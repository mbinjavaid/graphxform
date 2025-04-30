import argparse
import copy
import importlib
import os
import datetime # Needed for default results path in config if not overridden
import warnings # To suppress potential torch.compile warnings if needed

import torch._dynamo
torch._dynamo.config.suppress_errors = True

import torch
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
# --- AMP & Compile ---
from torch.cuda.amp import GradScaler, autocast
# --- End AMP & Compile ---
from tqdm import tqdm
import numpy as np

from logger import Logger
from molecule_dataset import TransformationMoleculeDataset
from config import MoleculeConfig
from model.molecule_transformer import MoleculeTransformer, dict_to_cpu
from molecule_design import MoleculeDesign # Needed for collate_fn

# Suppress potential torch.compile warnings if they become noisy
# warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo")

def save_checkpoint(checkpoint: dict, filename: str, config: MoleculeConfig):
    """Saves training checkpoint."""
    os.makedirs(config.results_path, exist_ok=True)
    path = os.path.join(config.results_path, filename)
    # Use a temporary file and atomic move for safer saving
    tmp_path = path + ".tmp"
    try:
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path) # Atomic replace
        print(f"Checkpoint saved to {path}")
    except Exception as e:
        print(f"Error saving checkpoint to {path}: {e}")
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass # Ignore error during cleanup

# --- Modified train_for_one_epoch for AMP ---
def train_for_one_epoch(epoch: int, config: MoleculeConfig, network: MoleculeTransformer,
                        optimizer: torch.optim.Optimizer | None,
                        scaler: GradScaler | None, # Added scaler argument
                        dataloader: DataLoader, is_validation=False):
    """
    Trains or validates the model for one epoch using the provided DataLoader.
    Uses AMP (autocast and GradScaler) during training if scaler is provided.

    Args:
        epoch (int | None): Current epoch number (used for logging, None for validation).
        config (MoleculeConfig): Configuration object.
        network (MoleculeTransformer): The model.
        optimizer (torch.optim.Optimizer | None): Optimizer instance (None for validation).
        scaler (GradScaler | None): GradScaler instance for AMP (None for validation or CPU).
        dataloader (DataLoader): DataLoader providing batches.
        is_validation (bool): Flag indicating if this is a validation run.

    Returns:
        dict: Dictionary containing loss metrics for the epoch.
    """
    metrics = dict()
    network.train() if not is_validation else network.eval()
    use_amp = not is_validation and scaler is not None and config.training_device != "cpu"

    accumulated_loss = 0
    accumulated_loss_lvl_zero = 0
    accumulated_loss_lvl_one = 0
    accumulated_loss_lvl_two = 0
    num_batches = len(dataloader)

    if num_batches == 0:
        print(f"Warning: {'Validation' if is_validation else 'Training'} dataloader is empty. Skipping epoch.")
        metric_prefix = "val_" if is_validation else ""
        metrics[f"{metric_prefix}full_loss"] = float('nan')
        metrics[f"{metric_prefix}loss_level_zero"] = float('nan')
        metrics[f"{metric_prefix}loss_level_one"] = float('nan')
        metrics[f"{metric_prefix}loss_level_two"] = float('nan')
        return metrics

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} {'Validation' if is_validation else 'Training'}", total=num_batches)

    criterion = CrossEntropyLoss(reduction="mean", ignore_index=-1)

    for data in progress_bar:
        input_batch = {
            k: v.to(network.device)
            for k, v in data.items()
            if k not in ['target_zero', 'target_one', 'target_two']
        }
        target_zero = data["target_zero"].to(network.device)
        target_one = data["target_one"].to(network.device)
        target_two = data["target_two"].to(network.device)
        mask_zero = data["feasibility_mask_level_zero"].to(network.device)
        mask_one = data["feasibility_mask_level_one"].to(network.device)
        mask_two = data["feasibility_mask_level_two"].to(network.device)

        # --- Forward Pass with autocast for AMP ---
        # Disable gradient calculation if validating
        with torch.set_grad_enabled(not is_validation):
            # Use autocast context manager for mixed precision
            with autocast(enabled=use_amp):
                logits_zero, logits_one, logits_two = network(input_batch)

                # --- Masking (inside autocast context) ---
                if logits_zero.shape != mask_zero.shape:
                     raise ValueError(f"Shape mismatch L0: Logits {logits_zero.shape}, Mask {mask_zero.shape}")
                logits_zero[mask_zero] = float("-inf")

                if logits_one.shape != mask_one.shape:
                     raise ValueError(f"Shape mismatch L1: Logits {logits_one.shape}, Mask {mask_one.shape}")
                logits_one[mask_one] = float("-inf")

                if logits_two.shape != mask_two.shape:
                     raise ValueError(f"Shape mismatch L2: Logits {logits_two.shape}, Mask {mask_two.shape}")
                logits_two[mask_two] = float("-inf")

                # --- Loss Calculation (inside autocast context) ---
                # Calculate losses in potentially higher precision (autocast handles it)
                loss_zero = criterion(logits_zero, target_zero)
                loss_zero = torch.tensor(0., device=network.device) if torch.isnan(loss_zero) else loss_zero
                loss_one = criterion(logits_one, target_one)
                loss_one = torch.tensor(0., device=network.device) if torch.isnan(loss_one) else loss_one
                loss_two = criterion(logits_two, target_two)
                loss_two = torch.tensor(0., device=network.device) if torch.isnan(loss_two) else loss_two

                # Combine losses
                loss = loss_zero + config.scale_factor_level_one * loss_one + config.scale_factor_level_two * loss_two

        # --- Optimization Step (if training, outside autocast) ---
        if not is_validation:
            if optimizer is None:
                 raise ValueError("Optimizer cannot be None during training.")
            if scaler is None and config.training_device != "cpu":
                 raise ValueError("Scaler cannot be None during CUDA training.")

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                # Scale the loss using GradScaler
                scaler.scale(loss).backward()

                if config.optimizer["gradient_clipping"] > 0:
                    scaler.unscale_(optimizer) # Unscale gradients before clipping
                    torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=config.optimizer["gradient_clipping"])

                # scaler.step() checks for inf/NaN grads, adjusts scale
                scaler.step(optimizer)
                # scaler.update() updates the scale factor
                scaler.update()
            else: # Standard backward/step for CPU or if AMP is disabled
                loss.backward()
                if config.optimizer["gradient_clipping"] > 0:
                    torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=config.optimizer["gradient_clipping"])
                optimizer.step()

        # --- Accumulate Metrics ---
        batch_loss = loss.item()
        accumulated_loss += batch_loss
        accumulated_loss_lvl_zero += loss_zero.item()
        accumulated_loss_lvl_one += loss_one.item()
        accumulated_loss_lvl_two += loss_two.item()

        progress_bar.set_postfix({"batch_loss": f"{batch_loss:.4f}"})

        del input_batch, target_zero, target_one, target_two, mask_zero, mask_one, mask_two
        del logits_zero, logits_one, logits_two, loss, loss_zero, loss_one, loss_two

    metric_prefix = "val_" if is_validation else ""
    metrics[f"{metric_prefix}full_loss"] = accumulated_loss / num_batches if num_batches > 0 else float('nan')
    metrics[f"{metric_prefix}loss_level_zero"] = accumulated_loss_lvl_zero / num_batches if num_batches > 0 else float('nan')
    metrics[f"{metric_prefix}loss_level_one"] = accumulated_loss_lvl_one / num_batches if num_batches > 0 else float('nan')
    metrics[f"{metric_prefix}loss_level_two"] = accumulated_loss_lvl_two / num_batches if num_batches > 0 else float('nan')

    return metrics
# --- End Modified train_for_one_epoch ---

if __name__ == '__main__':
    default_train_dataset = "./data/chembl/transformation_datasets/transformations_train.pkl"
    default_val_dataset = "./data/chembl/transformation_datasets/transformations_valid.pkl"
    pretrain_num_epochs = 1000
    load_checkpoint_from_path = None # Default: look for last_model.pt

    print(">> Pretraining Molecule Design with Transformation Dataset")

    parser = argparse.ArgumentParser(description='Pretrain Molecule Transformer')
    parser.add_argument('--debug', help="debug flag to turn off server logging", action="store_true")
    parser.add_argument('--run-name', type=str, help="Descriptive run name", default="Pretrain_Run")
    parser.add_argument('--exp-name', type=str, help="MLflow Experiment name", default="Molecule_Pretraining")
    parser.add_argument('--config', help="Path to optional config relative to script")
    parser.add_argument('--train-data', type=str, default=default_train_dataset, help="Path to training data pickle")
    parser.add_argument('--val-data', type=str, default=default_val_dataset, help="Path to validation data pickle")
    parser.add_argument('--epochs', type=int, default=pretrain_num_epochs, help="Number of epochs to train")
    parser.add_argument('--load-checkpoint', type=str, default=load_checkpoint_from_path,
                        help="Path to load checkpoint from. If None, tries 'last_model.pt' in results path.")
    # Add argument to disable torch.compile if needed
    parser.add_argument('--no-compile', action='store_true', help="Disable torch.compile")
    args = parser.parse_args()

    if args.config is not None:
        config_module_path = args.config.replace('.py', '').replace('/', '.')
        try:
            MoleculeConfig = importlib.import_module(config_module_path).MoleculeConfig
            print(f"Loaded configuration from {args.config}")
        except ImportError as e:
            print(f"Error loading config from {args.config}: {e}. Using default config.")
            from config import MoleculeConfig
    else:
        from config import MoleculeConfig

    config = MoleculeConfig()
    if not hasattr(config, 'batch_size_training'): config.batch_size_training = 64
    if not hasattr(config, 'batch_size_validation'): config.batch_size_validation = config.batch_size_training
    if not hasattr(config, 'num_dataloader_workers'): config.num_dataloader_workers = 3
    # Ensure results_path exists in config
    if not hasattr(config, 'results_path'): config.results_path = f"./results/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"Results path: {config.results_path}")
    os.makedirs(config.results_path, exist_ok=True) # Ensure results path exists early

    logger = Logger(args, config.results_path, config.log_to_file)
    logger.log_hyperparams(config)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    print("Instantiating datasets...")
    try:
        train_dataset = TransformationMoleculeDataset(config, args.train_data, no_random=False)
        val_dataset = TransformationMoleculeDataset(config, args.val_data, no_random=True)
    except FileNotFoundError as e:
        print(f"Error: Dataset file not found. {e}")
        exit(1)
    except Exception as e:
        print(f"Error creating dataset: {e}")
        exit(1)

    print("Creating dataloaders...")
    persistent_workers_flag = config.num_dataloader_workers > 0
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size_training, shuffle=True,
        num_workers=config.num_dataloader_workers, collate_fn=MoleculeDesign.list_to_batch,
        pin_memory=True if config.training_device != "cpu" else False,
        persistent_workers=persistent_workers_flag, drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.batch_size_validation, shuffle=False,
        num_workers=config.num_dataloader_workers, collate_fn=MoleculeDesign.list_to_batch,
        pin_memory=True if config.training_device != "cpu" else False,
        persistent_workers=persistent_workers_flag, drop_last=False
    )
    print(f"Train dataloader: {len(train_dataloader)} batches of size {config.batch_size_training}")
    print(f"Validation dataloader: {len(val_dataloader)} batches of size {config.batch_size_validation}")

    print("Setting up model...")
    network = MoleculeTransformer(config, config.training_device)

    # --- Apply torch.compile (PyTorch 2.0+) ---
    use_compile = hasattr(torch, 'compile') and not args.no_compile
    if use_compile:
        print("Attempting to compile model with torch.compile...")
        try:
            # You might experiment with modes like 'reduce-overhead' or 'max-autotune'
            network = torch.compile(network, mode='default')
            print("Model compiled successfully.")
        except Exception as e:
            print(f"torch.compile failed: {e}. Proceeding without compiling.")
            use_compile = False # Fallback if compilation fails
    else:
        if not hasattr(torch, 'compile'):
             print("torch.compile not available (requires PyTorch 2.0+).")
        else:
             print("torch.compile disabled by --no-compile flag.")
    # --- End torch.compile ---

    network.to(network.device)
    print(f"Model is on device: {network.device}")
    num_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters.")

    # --- Checkpoint Loading Logic ---
    start_epoch = 0
    best_validation_loss = float("inf")
    best_epoch = 0
    checkpoint = None # Initialize checkpoint variable
    load_path = args.load_checkpoint # Path specified by user

    # If user didn't specify a path, try the default 'last_model.pt'
    if load_path is None:
        default_load_path = os.path.join(config.results_path, "last_model.pt")
        if os.path.exists(default_load_path):
            load_path = default_load_path
            print(f"No --load-checkpoint specified, found existing 'last_model.pt'. Resuming from: {load_path}")
        else:
            print("No checkpoint specified and 'last_model.pt' not found. Starting training from scratch.")
    else:
        print(f"Attempting to load checkpoint from specified path: {load_path}")

    # Proceed with loading if a path was determined
    if load_path and os.path.exists(load_path):
        try:
            checkpoint = torch.load(load_path, map_location=network.device)
            if "model_weights" in checkpoint:
                # Handle potential issues if model was compiled before saving
                # load_state_dict might need strict=False if compile adds attributes
                try:
                    network.load_state_dict(checkpoint["model_weights"], strict=True)
                except RuntimeError as e:
                    print(f"Warning: Strict loading failed ({e}). Attempting non-strict loading.")
                    network.load_state_dict(checkpoint["model_weights"], strict=False)
                print("Model weights loaded.")
            else:
                 print("Warning: Checkpoint does not contain 'model_weights'.")

            start_epoch = checkpoint.get("pretrain_epochs_trained", 0)
            best_validation_loss = checkpoint.get("pretrain_best_validation_loss", float("inf"))
            best_epoch = checkpoint.get("best_epoch", 0) # Load best epoch
            print(f"Resuming training from epoch {start_epoch + 1}")
            print(f"Previous best validation loss: {best_validation_loss:.4f} (at epoch {best_epoch})")

        except Exception as e:
            print(f"Error loading checkpoint from {load_path}: {e}. Starting from scratch.")
            start_epoch = 0
            best_validation_loss = float("inf")
            best_epoch = 0
            checkpoint = None # Reset checkpoint dict on error
    elif load_path: # User specified a path but it doesn't exist
        print(f"Error: Specified checkpoint file not found at {load_path}. Starting from scratch.")
        start_epoch = 0
        best_validation_loss = float("inf")
        best_epoch = 0
    # --- End Checkpoint Loading Logic ---

    print("Setting up optimizer and scheduler...")
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=config.optimizer["lr"],
        weight_decay=config.optimizer["weight_decay"]
    )
    # Load optimizer state if checkpoint exists and config allows
    if checkpoint and config.load_optimizer_state and "optimizer_state" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            # Move optimizer state tensors to the correct device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(network.device)
            print("Optimizer state loaded and moved to device.")
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}. Initializing new optimizer state.")

    # --- Initialize GradScaler for AMP ---
    # Enable only if using CUDA and AMP is desired (can add config flag later)
    scaler_enabled = config.training_device != "cpu"
    scaler = GradScaler(enabled=scaler_enabled)
    if checkpoint and scaler_enabled and "scaler_state" in checkpoint:
        try:
            scaler.load_state_dict(checkpoint["scaler_state"])
            print("GradScaler state loaded.")
        except Exception as e:
            print(f"Warning: Could not load GradScaler state: {e}. Initializing new scaler state.")
    print(f"AMP GradScaler enabled: {scaler_enabled}")
    # --- End GradScaler Init ---

    # Scheduler lambda depends on the starting epoch
    _lambda = lambda epoch: config.optimizer["schedule"]["decay_factor"] ** (
                (start_epoch + epoch) // config.optimizer["schedule"]["decay_lr_every_epochs"])
    scheduler = LambdaLR(optimizer, lr_lambda=_lambda)
    # Consider saving/loading scheduler state as well for exact resumption
    # if checkpoint and "scheduler_state" in checkpoint:
    #    scheduler.load_state_dict(checkpoint["scheduler_state"])

    # --- Training Loop ---
    print(f"Starting pre-training from epoch {start_epoch + 1} for {args.epochs} epochs.")
    total_epochs_to_run = args.epochs
    end_epoch = start_epoch + total_epochs_to_run

    for epoch in range(start_epoch, end_epoch):
        current_epoch_num = epoch + 1
        print(f"\n--- Epoch {current_epoch_num}/{end_epoch} ---")

        # Training Step
        print("Training...")
        train_metrics = train_for_one_epoch(
            current_epoch_num, config, network, optimizer, scaler, train_dataloader, is_validation=False
        )
        logger.log_metrics(train_metrics, step=current_epoch_num)
        print(f">> Epoch {current_epoch_num} Train Avg Losses | Full: {train_metrics['full_loss']:.4f}, L0: {train_metrics['loss_level_zero']:.4f}, L1: {train_metrics['loss_level_one']:.4f}, L2: {train_metrics['loss_level_two']:.4f}")

        # Validation Step
        print("Validating...")
        # No need for torch.cuda.empty_cache() generally
        with torch.no_grad(): # Ensure no gradients are computed
            validation_metrics = train_for_one_epoch(
                current_epoch_num, config, network, None, None, val_dataloader, is_validation=True # Pass None for optimizer and scaler
            )
        logger.log_metrics(validation_metrics, step=current_epoch_num)
        print(f">> Epoch {current_epoch_num} Valid Avg Losses | Full: {validation_metrics['val_full_loss']:.4f}, L0: {validation_metrics['val_loss_level_zero']:.4f}, L1: {validation_metrics['val_loss_level_one']:.4f}, L2: {validation_metrics['val_loss_level_two']:.4f}")

        # LR Scheduler Step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Current LR: {current_lr:.6f}")
        logger.log_metrics({"learning_rate": current_lr}, step=current_epoch_num)

        # --- Save Checkpoint ---
        # Prepare checkpoint data (always use CPU tensors for portability)
        checkpoint_data = {
            # Use state_dict directly; compile might wrap the model
            "model_weights": dict_to_cpu(network.state_dict()),
            "optimizer_state": dict_to_cpu(optimizer.state_dict()),
            "scaler_state": dict_to_cpu(scaler.state_dict()) if scaler_enabled else None, # Save scaler state if enabled
            "pretrain_epochs_trained": current_epoch_num,
            "pretrain_best_validation_loss": best_validation_loss,
            "best_epoch": best_epoch, # Include best epoch
            # "scheduler_state": dict_to_cpu(scheduler.state_dict()), # Optional
        }

        # Save latest model
        save_checkpoint(checkpoint_data, "last_model.pt", config)

        # Save best model based on validation loss
        current_val_loss = validation_metrics['val_full_loss']
        # Handle potential NaN in validation loss
        if np.isnan(current_val_loss):
             print(">> Validation loss is NaN. Cannot determine improvement.")
        elif current_val_loss < best_validation_loss:
            print(f">> New best validation loss: {current_val_loss:.4f} at epoch {current_epoch_num} (previous: {best_validation_loss:.4f} at epoch {best_epoch}). Saving best model.")
            best_validation_loss = current_val_loss
            best_epoch = current_epoch_num # Update best epoch
            # Update best loss and epoch in the dictionary before saving best model
            checkpoint_data["pretrain_best_validation_loss"] = best_validation_loss
            checkpoint_data["best_epoch"] = best_epoch
            save_checkpoint(checkpoint_data, "best_model.pt", config)
        else:
            print(f">> Validation loss ({current_val_loss:.4f}) did not improve from best ({best_validation_loss:.4f} at epoch {best_epoch}).")

    print(f"\n--- Pre-training Finished after {total_epochs_to_run} epochs ---")
    print(f"Final best validation loss: {best_validation_loss:.4f} achieved at epoch {best_epoch}")
