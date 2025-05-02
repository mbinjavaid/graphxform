import argparse
import copy
import importlib
import os
import datetime
# import warnings

# Comment out dynamo suppression if not needed/causing issues
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

import torch
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from tqdm import tqdm
import numpy as np

from logger import Logger
from molecule_dataset import TransformationMoleculeDataset # Reverted dataset
from config import MoleculeConfig
from model.molecule_transformer import MoleculeTransformer, dict_to_cpu

# Suppress potential torch.compile warnings if they become noisy
# warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo")


def save_checkpoint(checkpoint: dict, filename: str, config: MoleculeConfig):
    """Saves training checkpoint."""
    os.makedirs(config.results_path, exist_ok=True)
    path = os.path.join(config.results_path, filename)
    tmp_path = path + ".tmp"
    try:
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)
        # print(f"Checkpoint saved to {path}") # Less verbose saving
    except Exception as e:
        print(f"Error saving checkpoint to {path}: {e}")
        if os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except OSError: pass


# --- train_for_one_epoch with conditional AMP ---
def train_for_one_epoch(epoch: int | None,
                        config: MoleculeConfig,
                        network: MoleculeTransformer,
                        optimizer: torch.optim.Optimizer | None,
                        # Re-add scaler for optional AMP
                        scaler: GradScaler | None,
                        dataset: TransformationMoleculeDataset,
                        # Flag to know if AMP is globally enabled
                        amp_enabled: bool,
                        is_validation=False):
    """
    REVERTED structure + Optional AMP.
    - Creates DataLoader(batch_size=1) internally.
    - Uses [0] indexing for data access.
    - Conditionally uses AMP (autocast, scaler).
    """
    metrics = dict()
    network.train() if not is_validation else network.eval()

    # Determine if AMP should be used *in this specific call*
    # Needs to be training, AMP globally enabled, scaler provided, and on CUDA
    use_amp_here = (not is_validation and amp_enabled and scaler is not None and
                    config.training_device != "cpu")

    # --- Create DataLoader INSIDE (batch_size=1, no collate_fn) ---
    num_workers = getattr(config, 'num_dataloader_workers')
    pin_memory_flag = getattr(config, 'pin_memory', config.training_device != "cpu")
    persistent_workers_flag = num_workers > 0 and getattr(config, 'persistent_workers', True)

    # print(num_workers)
    # print(pin_memory_flag)
    # print(persistent_workers_flag)

    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=not is_validation,
        num_workers=num_workers, pin_memory=pin_memory_flag,
        persistent_workers=persistent_workers_flag
    )
    # --- End DataLoader Creation ---

    accumulated_loss = 0
    accumulated_loss_lvl_zero = 0
    accumulated_loss_lvl_one = 0
    accumulated_loss_lvl_two = 0
    num_batches = len(dataloader)

    if num_batches == 0: # Handle empty dataloader
        print(f"Warning: {'Validation' if is_validation else 'Training'} dataloader created 0 batches. Skipping epoch.")
        metric_prefix = "val_" if is_validation else ""
        metrics[f"{metric_prefix}full_loss"]=metrics[f"{metric_prefix}loss_level_zero"]=metrics[f"{metric_prefix}loss_level_one"]=metrics[f"{metric_prefix}loss_level_two"]=float('nan')
        return metrics

    progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch if epoch is not None else 'Validation'} {'Validation' if is_validation else 'Training'}")
    data_iter = iter(dataloader)
    criterion = CrossEntropyLoss(reduction="mean", ignore_index=-1)

    # --- Batch Loop ---
    for batch_idx in progress_bar:
        data = next(data_iter)

        # --- Data Loading and Transfer ([0] indexing) ---
        input_batch_dict = data['input']
        # Use non_blocking if pin_memory is True
        input_batch = {k: v[0].to(network.device, non_blocking=pin_memory_flag) for k, v in input_batch_dict.items()}
        target_zero = data["target_zero"][0].to(network.device, non_blocking=pin_memory_flag)
        target_one = data["target_one"][0].to(network.device, non_blocking=pin_memory_flag)
        target_two = data["target_two"][0].to(network.device, non_blocking=pin_memory_flag)
        mask_zero = input_batch["feasibility_mask_level_zero"]
        mask_one = input_batch["feasibility_mask_level_one"]
        mask_two = input_batch["feasibility_mask_level_two"]
        # --- End Data Loading ---

        # --- Model Forward/Backward/Step (Conditional AMP) ---
        with torch.set_grad_enabled(not is_validation):
            # Conditionally use autocast
            with autocast(enabled=use_amp_here):
                logits_zero, logits_one, logits_two = network(input_batch)

                # --- MASK APPLICATION (Indexed Assignment - Remains the same) ---
                try:
                    if logits_zero.shape[1] != mask_zero.shape[1]: raise ValueError(f"L0 shape mismatch: Logits {logits_zero.shape}, Mask {mask_zero.shape}")
                    logits_zero[mask_zero.bool()] = float("-inf")

                    batch_max_actions_l1 = mask_one.shape[1]
                    if logits_one.shape[1] < batch_max_actions_l1: raise ValueError(f"L1 shape mismatch: Fixed Logits ({logits_one.shape[1]}) < Dynamic Mask ({batch_max_actions_l1})")
                    logits_one[:, :batch_max_actions_l1][mask_one.bool()] = float("-inf")

                    if logits_two.shape[1] != mask_two.shape[1]: raise ValueError(f"L2 shape mismatch: Logits {logits_two.shape}, Mask {mask_two.shape}")
                    logits_two[mask_two.bool()] = float("-inf")
                except (IndexError, ValueError) as e:
                     print(f"\nERROR applying mask (Batch {batch_idx}, Epoch {epoch}): {e}")
                     print(f"  Logits L0 shape: {logits_zero.shape}, Mask L0 shape: {mask_zero.shape}")
                     print(f"  Logits L1 shape: {logits_one.shape}, Mask L1 shape: {mask_one.shape}")
                     print(f"  Logits L2 shape: {logits_two.shape}, Mask L2 shape: {mask_two.shape}")
                     raise e
                # --- END MASK APPLICATION ---

                # --- Loss Calculation (inside autocast context) ---
                loss_zero = criterion(logits_zero, target_zero)
                loss_zero = torch.tensor(0., device=network.device) if torch.isnan(loss_zero) else loss_zero
                loss_one = criterion(logits_one, target_one)
                loss_one = torch.tensor(0., device=network.device) if torch.isnan(loss_one) else loss_one
                loss_two = criterion(logits_two, target_two)
                loss_two = torch.tensor(0., device=network.device) if torch.isnan(loss_two) else loss_two
                loss = loss_zero + config.scale_factor_level_one * loss_one + config.scale_factor_level_two * loss_two
            # --- End autocast context ---

        # --- Optimization Step (Conditional Scaler) ---
        if not is_validation:
            if optimizer is None: raise ValueError("Optimizer cannot be None during training.")
            optimizer.zero_grad(set_to_none=True)

            if use_amp_here:
                # Use scaler
                scaler.scale(loss).backward()
                if config.optimizer.get("gradient_clipping", 0) > 0:
                    scaler.unscale_(optimizer) # Unscale before clipping
                    torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=config.optimizer["gradient_clipping"])
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard backward and step
                loss.backward()
                if config.optimizer.get("gradient_clipping", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=config.optimizer["gradient_clipping"])
                optimizer.step()
        # --- End Optimization Step ---

        # --- Accumulate Metrics ---
        batch_loss = loss.item()
        accumulated_loss += batch_loss
        accumulated_loss_lvl_zero += loss_zero.item()
        accumulated_loss_lvl_one += loss_one.item()
        accumulated_loss_lvl_two += loss_two.item()

        progress_bar.set_postfix({"batch_loss": f"{batch_loss:.4f}"})
        del data, input_batch, target_zero, target_one, target_two, mask_zero, mask_one, mask_two
        del logits_zero, logits_one, logits_two, loss, loss_zero, loss_one, loss_two


    # --- Calculate Average Metrics ---
    metric_prefix = "val_" if is_validation else ""
    metrics[f"{metric_prefix}full_loss"] = accumulated_loss / num_batches if num_batches > 0 else float('nan')
    metrics[f"{metric_prefix}loss_level_zero"] = accumulated_loss_lvl_zero / num_batches if num_batches > 0 else float('nan')
    metrics[f"{metric_prefix}loss_level_one"] = accumulated_loss_lvl_one / num_batches if num_batches > 0 else float('nan')
    metrics[f"{metric_prefix}loss_level_two"] = accumulated_loss_lvl_two / num_batches if num_batches > 0 else float('nan')

    return metrics


# --- Main Execution Block with AMP/Compile Flags and Deepcopy ---
if __name__ == '__main__':
    # --- Default Parameters ---
    pretrain_train_dataset = "./data/chembl/transformation_datasets/transformations_train.pkl"
    pretrain_val_dataset = "./data/chembl/transformation_datasets/transformations_valid.pkl"
    pretrain_num_epochs = 1000
    batch_size = 512
    num_batches_per_epoch = 3000
    batch_size_validation = 512
    load_checkpoint_from_path = None
    use_amp_default = True  # Enable AMP by default
    use_compile_default = False  # Enable torch.compile by default (if available)
    # --- End Parameters ---

    print(">> Pretraining Molecule Design (Reverted Structure + Optional AMP/Compile + Deepcopy Save)")

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Experiment')
    parser.add_argument('--debug', help="debug flag", action="store_true")
    parser.add_argument('--run-name', type=str, help="run name", default="Default_Pretrain_Run")
    parser.add_argument('--exp-name', type=str, help="MLflow experiment name", default="Molecule_Pretraining")
    parser.add_argument('--config', help="Path to optional config")
    parser.add_argument('--train-data', type=str, default=pretrain_train_dataset, help="Path to training data")
    parser.add_argument('--val-data', type=str, default=pretrain_val_dataset, help="Path to validation data")
    parser.add_argument('--epochs', type=int, default=pretrain_num_epochs, help="Number of epochs")
    parser.add_argument('--batch-size', type=int, default=batch_size, help="Dataset batch size (train)")
    parser.add_argument('--val-batch-size', type=int, default=batch_size_validation, help="Dataset batch size (val)")
    parser.add_argument('--num-batches', type=int, default=num_batches_per_epoch, help="Custom train batches per epoch")
    parser.add_argument('--load-checkpoint', type=str, default=load_checkpoint_from_path, help="Path to load checkpoint")
    parser.add_argument('--use-amp', action=argparse.BooleanOptionalAction, default=use_amp_default, help="Enable Automatic Mixed Precision (AMP)")
    parser.add_argument('--use-compile', action=argparse.BooleanOptionalAction, default=use_compile_default, help="Enable torch.compile (if available)")
    args = parser.parse_args()

    # --- Config Loading ---
    if args.config is not None:
        config_module_path = args.config.replace('.py', '').replace('/', '.')
        try:
            MoleculeConfig = importlib.import_module(config_module_path).MoleculeConfig
            print(f"Loaded configuration from {args.config}")
        except ImportError as e: print(f"Error loading config: {e}. Using default."); from config import MoleculeConfig
    else: from config import MoleculeConfig
    config = MoleculeConfig()
    # --- End Config Loading ---

    # --- Set Config Defaults & Overrides ---
    if not hasattr(config, 'results_path'): config.results_path = f"./results/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.run_name}"
    if not hasattr(config, 'log_to_file'): config.log_to_file = True
    if not hasattr(config, 'log_to_mlflow'): config.log_to_mlflow = False
    if not hasattr(config, 'mlflow_server_uri'): config.mlflow_server_uri = None
    if not hasattr(config, 'seed'): config.seed = 42
    if not hasattr(config, 'training_device'): config.training_device = "cuda" if torch.cuda.is_available() else "cpu"
    if not hasattr(config, 'num_dataloader_workers'): config.num_dataloader_workers = 4
    if not hasattr(config, 'pin_memory'): config.pin_memory = config.training_device != "cpu"
    if not hasattr(config, 'persistent_workers'): config.persistent_workers = config.num_dataloader_workers > 0
    if not hasattr(config, 'optimizer'): config.optimizer = {
            "lr": 1e-4,  # learning rate
            "weight_decay": 0,
            "gradient_clipping": 1.,
            "schedule": {
                "decay_lr_every_epochs": 1,
                "decay_factor": 1
            }
        }
    if not hasattr(config, 'load_optimizer_state'): config.load_optimizer_state = True
    if not hasattr(config, 'scale_factor_level_one'): config.scale_factor_level_one = 1.
    if not hasattr(config, 'scale_factor_level_two'): config.scale_factor_level_two = 1.
    # --- End Config Defaults ---

    print(f"Results path: {config.results_path}")
    os.makedirs(config.results_path, exist_ok=True)
    # logger = Logger(args, config.results_path, config.log_to_file, config.log_to_mlflow, config.mlflow_server_uri)
    logger = Logger(args, config.results_path, config.log_to_file)
    logger.log_hyperparams(config)
    logger.log_hyperparams(vars(args))
    np.random.seed(config.seed); torch.manual_seed(config.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(config.seed)

    # --- Model Setup ---
    print("Setting up model...")
    network = MoleculeTransformer(config, config.training_device)

    # --- Conditional torch.compile ---
    compile_available = hasattr(torch, 'compile')
    if args.use_compile and compile_available:
        print("Attempting to compile model with torch.compile...")
        try:
            network = torch.compile(network, mode='default')
            print("Model compiled successfully.")
        except Exception as e:
            print(f"torch.compile failed: {e}. Proceeding without compiling.")
            args.use_compile = False
    elif args.use_compile and not compile_available:
        print("torch.compile not available (requires PyTorch 2.0+).")
        args.use_compile = False
    else:
        print("torch.compile disabled via flag or unavailable.")
    # --- End torch.compile ---

    network.to(network.device); print(f"Model on: {network.device}")
    num_params=sum(p.numel() for p in network.parameters() if p.requires_grad); print(f"Params: {num_params:,}")
    # --- End Model Setup ---

    # --- Checkpoint Loading ---
    checkpoint = None; start_epoch=0; best_validation_loss=float("inf"); best_epoch=0
    load_path = args.load_checkpoint
    if load_path is None:
        default_load_path = os.path.join(config.results_path, "last_model.pt")
        if os.path.exists(default_load_path): load_path = default_load_path; print(f"Found 'last_model.pt'. Resuming: {load_path}")
        else: print("No checkpoint path. Starting fresh.")
    else: print(f"Attempting load from: {load_path}")

    if load_path and os.path.exists(load_path):
        try:
            checkpoint = torch.load(load_path, map_location=network.device)
            if "model_weights" in checkpoint:
                _model_state_dict = checkpoint["model_weights"]
                if '_orig_mod.' in list(_model_state_dict.keys())[0]:
                   _model_state_dict = {k.replace('_orig_mod.', ''): v for k, v in _model_state_dict.items()}
                   print("Adjusted keys from compiled model state_dict.")
                try: network.load_state_dict(_model_state_dict, strict=True)
                except RuntimeError as e: print(f"Warn: Strict load failed ({e}). Trying non-strict."); network.load_state_dict(_model_state_dict, strict=False)
                print("Model weights loaded.")
            start_epoch = checkpoint.get("pretrain_epochs_trained", 0)
            best_validation_loss = checkpoint.get("pretrain_best_validation_loss", float("inf"))
            best_epoch = checkpoint.get("best_epoch", 0)
            print(f"Resuming from epoch {start_epoch + 1}. Best loss: {best_validation_loss:.4f} (epoch {best_epoch})")
        except Exception as e: print(f"Error loading checkpoint: {e}. Starting fresh."); start_epoch=0; best_validation_loss=float("inf"); best_epoch=0; checkpoint=None
    else:
        if load_path: print(f"Error: Checkpoint not found: '{load_path}'. Starting fresh.")
        checkpoint = { "pretrain_epochs_trained": 0, "pretrain_best_validation_loss": float("inf"), "best_epoch": 0 }
    # --- End Checkpoint Loading ---

    # --- Optimizer, Scaler, Scheduler Setup ---
    print("Setting up optimizer, scaler, and scheduler...")
    optimizer = torch.optim.Adam(network.parameters(), lr=config.optimizer["lr"], weight_decay=config.optimizer["weight_decay"])
    if checkpoint and config.load_optimizer_state and "optimizer_state" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor): state[k] = v.to(network.device)
            print("Optimizer state loaded.")
        except Exception as e: print(f"Warn: Could not load optimizer state: {e}.")

    scaler_enabled = args.use_amp and config.training_device != "cpu"
    scaler = GradScaler(enabled=scaler_enabled)
    if checkpoint and scaler_enabled and "scaler_state" in checkpoint and checkpoint["scaler_state"] is not None:
        try: scaler.load_state_dict(checkpoint["scaler_state"]); print("GradScaler state loaded.")
        except Exception as e: print(f"Warn: Could not load GradScaler state: {e}.")
    print(f"AMP GradScaler enabled: {scaler_enabled}")

    schedule_config = config.optimizer.get("schedule", {"decay_factor": 0.95, "decay_lr_every_epochs": 1})
    _lambda = lambda e: schedule_config["decay_factor"] ** (e // schedule_config["decay_lr_every_epochs"])
    scheduler = LambdaLR(optimizer, lr_lambda=_lambda)
    scheduler.last_epoch = start_epoch -1
    print(f"Scheduler initial last_epoch set to {scheduler.last_epoch}")
    # --- End Optimizer/Scaler/Scheduler ---

    # --- Dataset Setup ---
    print("Instantiating datasets...")
    try:
        train_dataset = TransformationMoleculeDataset(
            config, args.train_data, batch_size=args.batch_size,
            custom_num_batches=args.num_batches, no_random=False
        )
        val_dataset = TransformationMoleculeDataset(
            config, args.val_data, batch_size=args.val_batch_size,
            custom_num_batches=None, no_random=True, is_validation=True
        )
        print(f"Train dataset: {len(train_dataset)} batches")
        print(f"Validation dataset: {len(val_dataset)} batches")
    except Exception as e: print(f"Error creating dataset: {e}"); exit(1)
    # --- End Dataset Setup ---

    # --- Training Loop ---
    print(f"Starting pre-training from epoch {start_epoch + 1} for {args.epochs} epochs.")
    total_epochs_to_run = args.epochs
    end_epoch = start_epoch + total_epochs_to_run

    for epoch in range(start_epoch, end_epoch):
        current_epoch_num = epoch + 1
        print(f"\n--- Epoch {current_epoch_num}/{end_epoch} ---")

        # --- Training Step ---
        print("Training...")
        train_metrics = train_for_one_epoch(
            current_epoch_num, config, network, optimizer, scaler, train_dataset,
            amp_enabled=args.use_amp, is_validation=False
        )
        logger.log_metrics(train_metrics, step=current_epoch_num)
        print(f">> Epoch {current_epoch_num} Train Avg Losses | Full: {train_metrics['full_loss']:.4f}, L0: {train_metrics['loss_level_zero']:.4f}, L1: {train_metrics['loss_level_one']:.4f}, L2: {train_metrics['loss_level_two']:.4f}")

        # --- Validation Step ---
        print("Validating...")
        if config.training_device != 'cpu': torch.cuda.empty_cache()
        with torch.no_grad():
            validation_metrics = train_for_one_epoch(
                current_epoch_num, config, network, None, None, val_dataset,
                amp_enabled=False, is_validation=True
            )
        logger.log_metrics(validation_metrics, step=current_epoch_num)
        print(f">> Epoch {current_epoch_num} Valid Avg Losses | Full: {validation_metrics['val_full_loss']:.4f}, L0: {validation_metrics['val_loss_level_zero']:.4f}, L1: {validation_metrics['val_loss_level_one']:.4f}, L2: {validation_metrics['val_loss_level_two']:.4f}")

        # --- LR Scheduler Step ---
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]; print(f"Current LR: {current_lr:.6f}")
        logger.log_metrics({"learning_rate": current_lr}, step=current_epoch_num)

        # --- Save Checkpoint (with Deepcopy) ---
        # Prepare state dict, handling compiled model if necessary
        model_state_dict_to_save = network.state_dict()
        if '_orig_mod.' in list(model_state_dict_to_save.keys())[0]:
             model_state_dict_to_save = {k.replace('_orig_mod.', ''): v for k, v in model_state_dict_to_save.items()}

        # --- Apply deepcopy here, mirroring old script ---
        checkpoint_data = {
            "pretrain_epochs_trained": current_epoch_num,
            "model_weights": copy.deepcopy(dict_to_cpu(model_state_dict_to_save)),
            "optimizer_state": copy.deepcopy(dict_to_cpu(optimizer.state_dict())),
            "scaler_state": copy.deepcopy(dict_to_cpu(scaler.state_dict())) if scaler_enabled and scaler.is_enabled() else None, # Also deepcopy scaler state if enabled
            "pretrain_best_validation_loss": best_validation_loss,
            "best_epoch": best_epoch,
        }
        # --- End deepcopy application ---

        save_checkpoint(checkpoint_data, "last_model.pt", config)
        current_val_loss = validation_metrics['val_full_loss']
        if np.isnan(current_val_loss): print(">> Validation loss is NaN.")
        elif current_val_loss < best_validation_loss:
            print(f">> New best validation loss: {current_val_loss:.4f} (epoch {current_epoch_num}). Saving best model.")
            best_validation_loss = current_val_loss; best_epoch = current_epoch_num
            # Update best loss/epoch in the dict *before* saving best_model.pt
            checkpoint_data["pretrain_best_validation_loss"] = best_validation_loss
            checkpoint_data["best_epoch"] = best_epoch
            save_checkpoint(checkpoint_data, "best_model.pt", config)
        else: print(f">> Val loss ({current_val_loss:.4f}) did not improve from best ({best_validation_loss:.4f} epoch {best_epoch}).")
        # --- End Save Checkpoint ---

    print(f"\n--- Pre-training Finished after {total_epochs_to_run} total epochs run ({start_epoch+1} -> {end_epoch}) ---")
    print(f"Final best validation loss: {best_validation_loss:.4f} achieved at epoch {best_epoch}")
# --- End Main Execution Block ---