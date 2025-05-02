import argparse
import copy
import importlib
import os
import time
import datetime # Added for potential default results path

from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
# --- Added for AMP ---
from torch.cuda.amp import GradScaler, autocast
# --- End Added ---
from tqdm import tqdm

from logger import Logger
# from molecule_dataset import RandomMoleculeDataset
from molecule_dataset import TransformationMoleculeDataset

os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
import ray
import torch
import numpy as np
from config import MoleculeConfig
from core.gumbeldore_dataset import GumbeldoreDataset
from model.molecule_transformer import MoleculeTransformer, dict_to_cpu
from molecule_evaluator import MoleculeObjectiveEvaluator


def save_checkpoint(checkpoint: dict, filename: str, config: MoleculeConfig):
    """Saves training checkpoint with atomic write.""" # Updated docstring slightly
    os.makedirs(config.results_path, exist_ok=True)
    path = os.path.join(config.results_path, filename)
    tmp_path = path + ".tmp" # Use temp file for atomic write
    try:
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path) # Atomic replace
    except Exception as e:
        print(f"Error saving checkpoint to {path}: {e}")
        if os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except OSError: pass


# --- Updated train_for_one_epoch signature and logic ---
def train_for_one_epoch(epoch: int,
                        config: MoleculeConfig,
                        network: MoleculeTransformer,
                        network_weights: dict, # Used for generation, keep it
                        optimizer: torch.optim.Optimizer,
                        objective_evaluator: MoleculeObjectiveEvaluator,
                        best_objective: float,
                        # --- Added for AMP ---
                        scaler: GradScaler | None,
                        amp_enabled: bool
                        # --- End Added ---
                       ):
    """
    Generates data using Gumbeldore, then trains the network for one epoch on that data.
    Uses DataLoader(batch_size=1). Incorporates conditional AMP and new masking.
    """
    # --- Generation Step (Unchanged) ---
    gumbeldore_dataset = GumbeldoreDataset(
        config=config, objective_evaluator=objective_evaluator
    )
    metrics = gumbeldore_dataset.generate_dataset(
        network_weights, # Pass the frozen weights for generation
        best_objective=best_objective,
        memory_aggressive=False
    )
    print("Generated molecules")
    print(f"Mean obj. over fresh best mols: {metrics['mean_best_gen_obj']:.3f}")
    print(f"Best / worst obj. over fresh best mols: {metrics['best_gen_obj']:.3f}, {metrics['worst_gen_obj']:.3f}")
    print(f"Mean obj. over all time top 20 mols: {metrics['mean_top_20_obj']:.3f}")
    print(f"All time best mol: {list(metrics['top_20_molecules'][0].values())[0]:.3f}")
    torch.cuda.empty_cache()
    time.sleep(1)
    # --- End Generation Step ---

    # --- Dataset and DataLoader Setup (Unchanged, specific to finetuning) ---
    print("---- Loading dataset")
    try:
        dataset = TransformationMoleculeDataset(config, config.gumbeldore_config["destination_path"], batch_size=config.batch_size_training,
                                        custom_num_batches=config.num_batches_per_epoch)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {config.gumbeldore_config['destination_path']}. Skipping training for epoch {epoch+1}.")
        # Return metrics from generation, setting losses to NaN
        metrics["loss_level_zero"] = float('nan')
        metrics["loss_level_one"] = float('nan')
        metrics["loss_level_two"] = float('nan')
        top_20_molecules = metrics.pop("top_20_molecules", []) # Safely get and remove
        return metrics, top_20_molecules

    # DataLoader setup specific to finetuning (batch_size=1)
    num_workers = getattr(config, 'num_dataloader_workers', 0) # Default to 0 if not set
    pin_memory_flag = getattr(config, 'pin_memory', config.training_device != "cpu")
    persistent_workers_flag = num_workers > 0 and getattr(config, 'persistent_workers', False) # Default to False if not set

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True,
                            num_workers=num_workers, pin_memory=pin_memory_flag,
                            persistent_workers=persistent_workers_flag)
    # --- End Dataset/DataLoader Setup ---

    # --- Training Loop Setup ---
    network.train() # Set network to training mode

    # Freeze layers except the last (Unchanged)
    for parameter in network.parameters():
        parameter.requires_grad = False
    # Check if layers exist before setting requires_grad
    if hasattr(network, 'virtual_atom_linear'):
        network.virtual_atom_linear.weight.requires_grad = True
        network.virtual_atom_linear.bias.requires_grad = True
    if hasattr(network, 'bond_atom_linear'):
        network.bond_atom_linear.weight.requires_grad = True
        network.bond_atom_linear.bias.requires_grad = True

    # --- Added for AMP ---
    use_amp_here = (amp_enabled and scaler is not None and config.training_device != "cpu")
    # --- End Added ---

    accumulated_loss = 0 # Also track total loss for potential debugging
    accumulated_loss_lvl_zero = 0
    accumulated_loss_lvl_one = 0
    accumulated_loss_lvl_two = 0
    num_batches = len(dataloader)

    if num_batches == 0: # Handle empty dataloader case
        print("Warning: DataLoader created 0 batches. Skipping training loop.")
        metrics["loss_level_zero"] = float('nan')
        metrics["loss_level_one"] = float('nan')
        metrics["loss_level_two"] = float('nan')
        top_20_molecules = metrics.pop("top_20_molecules", [])
        return metrics, top_20_molecules

    progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1} Training")
    data_iter = iter(dataloader)
    criterion = CrossEntropyLoss(reduction="mean", ignore_index=-1)
    # --- End Training Loop Setup ---

    # --- Batch Training Loop ---
    for batch_idx in progress_bar:
        try:
            data = next(data_iter)
            # --- Data Loading and Transfer (batch_size=1 means [0] indexing) ---
            input_data_dict = data['input']
            # Use non_blocking if pin_memory is True
            input_data = {k: v[0].to(network.device, non_blocking=pin_memory_flag) for k, v in input_data_dict.items()}
            target_zero = data["target_zero"][0].to(network.device, non_blocking=pin_memory_flag)
            target_one = data["target_one"][0].to(network.device, non_blocking=pin_memory_flag)
            target_two = data["target_two"][0].to(network.device, non_blocking=pin_memory_flag)
            # Get masks directly from input_data
            mask_zero = input_data["feasibility_mask_level_zero"]
            mask_one = input_data["feasibility_mask_level_one"]
            mask_two = input_data["feasibility_mask_level_two"]
            # --- End Data Loading ---

            # --- Model Forward/Backward/Step (Conditional AMP) ---
            # Use autocast for forward pass and loss calculation if AMP is enabled
            with autocast(enabled=use_amp_here):
                logits_zero, logits_one, logits_two = network(input_data)

                # --- NEW MASK APPLICATION LOGIC (from pretrain.py) ---
                try:
                    # L0: Direct Masking
                    if logits_zero.shape[1] != mask_zero.shape[1]: raise ValueError(f"L0 shape mismatch: Logits {logits_zero.shape}, Mask {mask_zero.shape}")
                    logits_zero[mask_zero.bool()] = float("-inf")

                    # L1: Slice-then-Mask
                    batch_max_actions_l1 = mask_one.shape[1] # Dynamic size from mask
                    if logits_one.shape[1] < batch_max_actions_l1: raise ValueError(f"L1 shape mismatch: Fixed Logits ({logits_one.shape[1]}) < Dynamic Mask ({batch_max_actions_l1})")
                    logits_one[:, :batch_max_actions_l1][mask_one.bool()] = float("-inf")
                    # Mask out the rest of the fixed dimension (optional but clean)
                    # logits_one[:, batch_max_actions_l1:] = float("-inf") # Commented out: might interfere if target indices can fall here, though unlikely with ignore_index=-1

                    # L2: Direct Masking
                    if logits_two.shape[1] != mask_two.shape[1]: raise ValueError(f"L2 shape mismatch: Logits {logits_two.shape}, Mask {mask_two.shape}")
                    logits_two[mask_two.bool()] = float("-inf")
                except (IndexError, ValueError) as e:
                     print(f"\nERROR applying mask (Epoch {epoch+1}, Batch {batch_idx}): {e}")
                     # Provide more context if needed
                     raise e
                # --- END NEW MASK APPLICATION ---

                # --- Loss Calculation (inside autocast context) ---
                loss_zero = criterion(logits_zero, target_zero)
                loss_zero = torch.tensor(0., device=network.device) if torch.isnan(loss_zero) else loss_zero
                loss_one = criterion(logits_one, target_one)
                loss_one = torch.tensor(0., device=network.device) if torch.isnan(loss_one) else loss_one
                loss_two = criterion(logits_two, target_two)
                loss_two = torch.tensor(0., device=network.device) if torch.isnan(loss_two) else loss_two
                loss = loss_zero + config.scale_factor_level_one * loss_one + config.scale_factor_level_two * loss_two
            # --- End autocast context ---

            # --- Optimization Step (Conditional Scaler - from pretrain.py) ---
            optimizer.zero_grad(set_to_none=True) # Use set_to_none=True for potential memory savings

            if use_amp_here:
                # Use scaler for backward and step
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

            del data, input_data, target_zero, target_one, target_two, mask_zero, mask_one, mask_two
            del logits_zero, logits_one, logits_two, loss, loss_zero, loss_one, loss_two

        except StopIteration:
            print("DataLoader exhausted.")
            break # Exit the loop if dataloader is exhausted prematurely
        except Exception as e:
            print(f"\nError during training batch {batch_idx} in epoch {epoch+1}: {e}")
            # Decide whether to continue, break, or raise
            # For now, let's just print and continue to the next batch
            continue
    # --- End Batch Training Loop ---

    # --- Calculate Average Metrics ---
    metrics["loss_level_zero"] = accumulated_loss_lvl_zero / num_batches if num_batches > 0 else float('nan')
    metrics["loss_level_one"] = accumulated_loss_lvl_one / num_batches if num_batches > 0 else float('nan')
    metrics["loss_level_two"] = accumulated_loss_lvl_two / num_batches if num_batches > 0 else float('nan')
    metrics["full_loss"] = accumulated_loss / num_batches if num_batches > 0 else float('nan') # Add full loss metric

    # Prepare return values
    top_20_molecules = metrics.pop("top_20_molecules", []) # Safely get and remove before returning
    return metrics, top_20_molecules
# --- End train_for_one_epoch ---


def evaluate(eval_type: str, config: MoleculeConfig, network: MoleculeTransformer, objective_evaluator: MoleculeObjectiveEvaluator):
    """Evaluates the model by generating molecules and calculating metrics."""
    config = copy.deepcopy(config)
    config.gumbeldore_config["destination_path"] = None # Don't save generated data during eval

    gumbeldore_dataset = GumbeldoreDataset(
        config=config, objective_evaluator=objective_evaluator
    )

    # --- Updated to use state_dict and handle compile prefix ---
    network.eval() # Ensure network is in eval mode
    with torch.no_grad():
        model_state_dict_eval = network.state_dict()
        # Handle potential prefix from compiled model if evaluate is called on it
        if '_orig_mod.' in list(model_state_dict_eval.keys())[0]:
             model_state_dict_eval = {k.replace('_orig_mod.', ''): v for k, v in model_state_dict_eval.items()}
        network_weights_eval = copy.deepcopy(dict_to_cpu(model_state_dict_eval))
    # --- End Update ---

    metrics = gumbeldore_dataset.generate_dataset(network_weights_eval, memory_aggressive=False)

    # Process metrics (Unchanged)
    top_20_mols = metrics.pop("top_20_molecules", []) # Use pop with default
    metrics_processed = {
        f"{eval_type}_mean_top_20_obj": metrics.get("mean_top_20_obj", float('nan')),
        f"{eval_type}_mean_top_20_sa_score": metrics.get("mean_top_20_sa_score", float('nan')),
        f"{eval_type}_best_obj": metrics.get('best_gen_obj', float('-inf')),
        f"{eval_type}_best_mol_sa_score": metrics.get('best_gen_sa_score', float('nan')),
    }
    print("Evaluation done")
    print(f"Eval ({eval_type}) best obj: {metrics_processed[f'{eval_type}_best_obj']:.3f}")
    print(f"Eval ({eval_type}) mean top 20 obj: {metrics_processed[f'{eval_type}_mean_top_20_obj']:.3f}")

    # Create text format for saving top 20 mols
    top_20_text = "\n".join([f"{smiles}: {obj:.4f}" for d in top_20_mols for smiles, obj in d.items()])

    return metrics_processed, top_20_text


# --- Main Execution Block ---
if __name__ == '__main__':
    print(">> Molecule Design Finetuning")

    # --- Argument Parsing (Added AMP/Compile) ---
    parser = argparse.ArgumentParser(description='Experiment')
    parser.add_argument('--config', help="Path to optional config relative to main.py")
    # Add AMP/Compile flags, mirroring pretrain.py
    parser.add_argument('--use-amp', action=argparse.BooleanOptionalAction, default=True, help="Enable Automatic Mixed Precision (AMP)")
    parser.add_argument('--use-compile', action=argparse.BooleanOptionalAction, default=False, help="Enable torch.compile (if available)")
    # Add run name/exp name for better logging, if desired
    parser.add_argument('--run-name', type=str, help="run name for logging", default="Default_Finetune_Run")
    parser.add_argument('--exp-name', type=str, help="MLflow experiment name", default="Molecule_Finetuning")
    args = parser.parse_args()
    # --- End Argument Parsing ---

    # --- Config Loading (Unchanged) ---
    if args.config is not None:
        config_module_path = args.config.replace('.py', '').replace('/', '.')
        try:
            MoleculeConfig = importlib.import_module(config_module_path).MoleculeConfig
            print(f"Loaded configuration from {args.config}")
        except ImportError as e: print(f"Error loading config: {e}. Using default."); from config import MoleculeConfig
    else: from config import MoleculeConfig
    config = MoleculeConfig()
    # --- End Config Loading ---

    # --- Set Config Defaults & Overrides (Ensure necessary paths/flags exist) ---
    if not hasattr(config, 'results_path'): config.results_path = f"./results/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.run_name}"
    if not hasattr(config, 'log_to_file'): config.log_to_file = True
    # Add other defaults if needed, similar to pretrain.py, especially for optimizer/schedule
    if not hasattr(config, 'seed'): config.seed = 42
    if not hasattr(config, 'training_device'): config.training_device = "cuda" if torch.cuda.is_available() else "cpu"
    if not hasattr(config, 'objective_gnn_device'): config.objective_gnn_device = config.training_device # Default objective GNN device
    if not hasattr(config, 'num_dataloader_workers'): config.num_dataloader_workers = 4 # Default workers for finetuning
    if not hasattr(config, 'pin_memory'): config.pin_memory = config.training_device != "cpu"
    if not hasattr(config, 'persistent_workers'): config.persistent_workers = False # Default for finetuning
    if not hasattr(config, 'optimizer'): config.optimizer = {
            "lr": 1e-4,  # learning rate
            "weight_decay": 0,
            "gradient_clipping": 1.,  # Clip gradient to given L2-norm. Set to 0 if no clipping should be performed.
            "schedule": {
                "decay_lr_every_epochs": 1,
                "decay_factor": 1
            }
        }
    if not hasattr(config, 'load_optimizer_state'): config.load_optimizer_state = True
    if not hasattr(config, 'scale_factor_level_one'): config.scale_factor_level_one = 1.
    if not hasattr(config, 'scale_factor_level_two'): config.scale_factor_level_two = 1.
    if not hasattr(config, 'num_epochs'): config.num_epochs = 1000 # Default epochs if not set
    if not hasattr(config, 'batch_size_training'): config.batch_size_training = 64 # Default training batch size if not set
    if not hasattr(config, 'num_batches_per_epoch'): config.num_batches_per_epoch = 1000 # Default batches per epoch if not set
    if not hasattr(config, 'gumbeldore_config'): raise ValueError("Config must contain 'gumbeldore_config' dictionary")
    if "destination_path" not in config.gumbeldore_config: raise ValueError("Config 'gumbeldore_config' must contain 'destination_path'")
    # --- End Config Defaults ---

    # --- Ray and Logger Setup (Unchanged) ---
    num_gpus_ray = 0
    if hasattr(config, 'CUDA_VISIBLE_DEVICES') and config.CUDA_VISIBLE_DEVICES:
        num_gpus_ray = len(config.CUDA_VISIBLE_DEVICES.split(","))
    try:
        ray.init(num_gpus=num_gpus_ray, logging_level="info")
        print(ray.available_resources())
    except Exception as e:
        print(f"Ray initialization failed: {e}")
        # Decide how to proceed - maybe run without Ray? For now, just print.

    os.makedirs(config.results_path, exist_ok=True) # Ensure results path exists for logger
    logger = Logger(args, config.results_path, config.log_to_file) # Assuming logger doesn't need MLflow for finetuning
    logger.log_hyperparams(config)
    logger.log_hyperparams(vars(args)) # Log command line args too
    # Fix random number generator seed (Unchanged)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(config.seed)
    # --- End Ray/Logger Setup ---

    # --- Model and Evaluator Setup ---
    print("Setting up model and objective evaluator...")
    network = MoleculeTransformer(config, config.training_device)
    objective_evaluator = MoleculeObjectiveEvaluator(config, device=config.objective_gnn_device)

    # --- Conditional torch.compile (copied from pretrain.py) ---
    compile_available = hasattr(torch, 'compile')
    if args.use_compile and compile_available:
        print("Attempting to compile model with torch.compile...")
        try:
            network = torch.compile(network, mode='default') # Use default mode
            print("Model compiled successfully.")
        except Exception as e:
            print(f"torch.compile failed: {e}. Proceeding without compiling.")
            args.use_compile = False # Disable compile if it fails
    elif args.use_compile and not compile_available:
        print("torch.compile not available (requires PyTorch 2.0+).")
        args.use_compile = False
    else:
        print("torch.compile disabled via flag or unavailable.")
    # --- End torch.compile ---

    print(f"Policy network is on device {config.training_device}")
    network.to(network.device)
    num_params=sum(p.numel() for p in network.parameters() if p.requires_grad); print(f"Trainable Params: {num_params:,}")
    # --- End Model Setup ---

    # --- Scaler Setup (Added) ---
    scaler_enabled = args.use_amp and config.training_device != "cpu"
    scaler = GradScaler(enabled=scaler_enabled)
    print(f"AMP GradScaler enabled: {scaler_enabled}")
    # --- End Scaler Setup ---

    # --- Checkpoint Loading (Updated) ---
    checkpoint = None; start_epoch=0; best_validation_metric=float("-inf")
    load_path = getattr(config, 'load_checkpoint_from_path', None) # Get path from config

    if load_path and os.path.exists(load_path):
        print(f"Loading checkpoint from path {load_path}")
        try:
            checkpoint = torch.load(load_path, map_location=network.device)
            print(f"Checkpoint keys: {list(checkpoint.keys())}")

            if "model_weights" in checkpoint and checkpoint["model_weights"] is not None:
                _model_state_dict = checkpoint["model_weights"]
                # Handle compile prefix removal
                if '_orig_mod.' in list(_model_state_dict.keys())[0]:
                   _model_state_dict = {k.replace('_orig_mod.', ''): v for k, v in _model_state_dict.items()}
                   print("Adjusted keys from compiled model state_dict.")
                # Load model state
                try: network.load_state_dict(_model_state_dict, strict=True)
                except RuntimeError as e: print(f"Warn: Strict load failed ({e}). Trying non-strict."); network.load_state_dict(_model_state_dict, strict=False)
                print("Model weights loaded.")
            else: print("No 'model_weights' found or is None in checkpoint.")

            # Load other states if they exist
            start_epoch = checkpoint.get("epochs_trained", 0) # Use 'epochs_trained' as key
            best_validation_metric = checkpoint.get("best_validation_metric", float("-inf")) # Use 'best_validation_metric'

            # Load scaler state (Added)
            if scaler_enabled and "scaler_state" in checkpoint and checkpoint["scaler_state"] is not None:
                try: scaler.load_state_dict(checkpoint["scaler_state"]); print("GradScaler state loaded.")
                except Exception as e: print(f"Warn: Could not load GradScaler state: {e}.")
            elif scaler_enabled: print("Warn: Scaler enabled but no 'scaler_state' found in checkpoint.")

            print(f"Resuming from epoch {start_epoch + 1}. Best validation metric (obj): {best_validation_metric:.4f}")

        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting fresh.")
            start_epoch=0; best_validation_metric=float("-inf"); checkpoint=None
    else:
        if load_path: print(f"Checkpoint path specified but not found: '{load_path}'. Starting fresh.")
        else: print("No checkpoint path specified. Starting fresh.")
        # Initialize checkpoint structure for fresh start
        checkpoint = {
            "model_weights": None, "best_model_weights": None, "optimizer_state": None, "scaler_state": None,
            "epochs_trained": 0, "validation_metric": float("-inf"), "best_validation_metric": float("-inf")
        }
    # --- End Checkpoint Loading ---

    # --- Optimizer and Scheduler Setup (Updated) ---
    if config.num_epochs > 0:
        print("Setting up optimizer and LR scheduler.")
        optimizer = torch.optim.Adam(
            network.parameters(), # Should correctly get params even if compiled
            lr=config.optimizer["lr"],
            weight_decay=config.optimizer["weight_decay"]
        )
        # Load optimizer state (Updated - includes moving tensors to device)
        if checkpoint and config.load_optimizer_state and "optimizer_state" in checkpoint and checkpoint["optimizer_state"] is not None:
            print("Loading optimizer state from checkpoint.")
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                # Move optimizer state tensors to the correct device (important!)
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor): state[k] = v.to(network.device)
                print("Optimizer state loaded and moved to device.")
            except Exception as e: print(f"Warn: Could not load optimizer state: {e}.")
        else: print("Starting with fresh optimizer state.")

        print("Setting up LR scheduler")
        schedule_config = config.optimizer.get("schedule", {"decay_factor": 1, "decay_lr_every_epochs": 1}) # Default no decay
        _lambda = lambda e: schedule_config["decay_factor"] ** (e // schedule_config["decay_lr_every_epochs"])
        scheduler = LambdaLR(optimizer, lr_lambda=_lambda)
        scheduler.last_epoch = start_epoch -1 # Ensure scheduler starts correctly
        print(f"Scheduler initial last_epoch set to {scheduler.last_epoch}")
    # --- End Optimizer/Scheduler Setup ---

    # --- Training Loop ---
    if config.num_epochs > 0:
        print(f"Starting training from epoch {start_epoch + 1} for {config.num_epochs} total epochs.")
        end_epoch = start_epoch + config.num_epochs

        # Retrieve best state from potentially loaded checkpoint
        best_model_weights = checkpoint.get("best_model_weights") # Can be None
        best_validation_metric = checkpoint.get("best_validation_metric", float("-inf"))

        start_time_counter = None
        if hasattr(config, 'wall_clock_limit') and config.wall_clock_limit is not None:
            print(f"Wall clock limit of training set to {config.wall_clock_limit / 3600:.2f} hours")
            start_time_counter = time.perf_counter()

        for epoch in range(start_epoch, end_epoch):
            current_epoch_num = epoch + 1
            print(f"\n------ Epoch {current_epoch_num}/{end_epoch} ------")
            print(f"Generating dataset for epoch {current_epoch_num}.")

            # Get network weights for generation (using state_dict, deepcopy, cpu)
            network.eval() # Ensure model is in eval mode for getting weights
            with torch.no_grad():
                current_model_state_dict = network.state_dict()
                # Handle compile prefix if needed
                if '_orig_mod.' in list(current_model_state_dict.keys())[0]:
                   current_model_state_dict = {k.replace('_orig_mod.', ''): v for k, v in current_model_state_dict.items()}
                network_weights_for_gen = copy.deepcopy(dict_to_cpu(current_model_state_dict))

            # --- Call Updated train_for_one_epoch ---
            generated_loggable_dict, generated_text_to_save = train_for_one_epoch(
                epoch, config, network, network_weights_for_gen, optimizer, objective_evaluator, best_validation_metric,
                scaler=scaler, amp_enabled=args.use_amp # Pass scaler and flag
            )
            # --- End Call ---

            # --- Post-Epoch Processing ---
            # Update epoch count in checkpoint structure *before* saving
            checkpoint["epochs_trained"] = current_epoch_num
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f">> Epoch {current_epoch_num} finished. LR: {current_lr:.6f}")
            if 'full_loss' in generated_loggable_dict: # Check if training actually happened
                 print(f"   Avg Losses | Full: {generated_loggable_dict['full_loss']:.4f}, L0: {generated_loggable_dict['loss_level_zero']:.4f}, L1: {generated_loggable_dict['loss_level_one']:.4f}, L2: {generated_loggable_dict['loss_level_two']:.4f}")
            logger.log_metrics({"learning_rate": current_lr}, step=current_epoch_num) # Log LR
            logger.log_metrics(generated_loggable_dict, step=current_epoch_num)
            # Save the top 20 molecules text
            if generated_text_to_save: # Ensure there's text to save
                logger.text_artifact(os.path.join(config.results_path, f"epoch_{current_epoch_num}_train_top_20_molecules.txt"), generated_text_to_save)

            # --- Save Checkpoint (Updated) ---
            # Prepare model state dict, handling compiled prefix
            model_state_dict_to_save = network.state_dict()
            if '_orig_mod.' in list(model_state_dict_to_save.keys())[0]:
                 model_state_dict_to_save = {k.replace('_orig_mod.', ''): v for k, v in model_state_dict_to_save.items()}

            # Update checkpoint dictionary using deepcopy and dict_to_cpu
            checkpoint["model_weights"] = copy.deepcopy(dict_to_cpu(model_state_dict_to_save))
            checkpoint["optimizer_state"] = copy.deepcopy(dict_to_cpu(optimizer.state_dict()))
            checkpoint["scaler_state"] = copy.deepcopy(dict_to_cpu(scaler.state_dict())) if scaler_enabled and scaler.is_enabled() else None # Save scaler state

            # Use best_gen_obj from metrics as validation metric
            val_metric = generated_loggable_dict.get("best_gen_obj", float("-inf"))
            checkpoint["validation_metric"] = val_metric
            save_checkpoint(checkpoint, "last_model.pt", config) # Save last model checkpoint

            if val_metric > best_validation_metric:
                print(f">> Got new best model (Obj: {val_metric:.4f} > {best_validation_metric:.4f}). Saving best_model.pt.")
                best_validation_metric = val_metric
                checkpoint["best_validation_metric"] = best_validation_metric # Update best metric in dict
                checkpoint["best_model_weights"] = copy.deepcopy(checkpoint["model_weights"]) # Store best weights
                best_model_weights = checkpoint["best_model_weights"] # Update local variable too
                save_checkpoint(checkpoint, "best_model.pt", config) # Save best model checkpoint
            else:
                 print(f">> Validation metric ({val_metric:.4f}) did not improve from best ({best_validation_metric:.4f}).")
            # --- End Save Checkpoint ---

            # Check wall clock limit
            if start_time_counter is not None and time.perf_counter() - start_time_counter > config.wall_clock_limit:
                print("Wall clock time limit exceeded. Stopping training.")
                break
            # --- End Post-Epoch Processing ---

    # --- Evaluation Phase ---
    print("\n--- Evaluation Phase ---")
    # Determine which checkpoint to evaluate
    eval_checkpoint_path = None
    if config.num_epochs > 0:
        best_model_path = os.path.join(config.results_path, "best_model.pt")
        if os.path.exists(best_model_path):
            print(f"Evaluating with best model: {best_model_path}")
            eval_checkpoint_path = best_model_path
        else:
            print("Warning: No best_model.pt found after training. Evaluating with last model.")
            last_model_path = os.path.join(config.results_path, "last_model.pt")
            if os.path.exists(last_model_path): eval_checkpoint_path = last_model_path
            else: print("Warning: No last_model.pt found either. Evaluating with current model state (might be untrained).")
    elif getattr(config, 'load_checkpoint_from_path', None):
         print(f"Evaluating with explicitly loaded model: {config.load_checkpoint_from_path}")
         # Use the already loaded network state if training didn't run
         eval_checkpoint_path = None # Signal not to reload
    else:
         print("Warning: No training performed and no checkpoint loaded. Evaluating with random model.")
         eval_checkpoint_path = None

    # Load the chosen checkpoint for evaluation if necessary
    if eval_checkpoint_path:
        try:
            checkpoint_eval = torch.load(eval_checkpoint_path, map_location=network.device)
            if "model_weights" in checkpoint_eval and checkpoint_eval["model_weights"] is not None:
                 _model_state_dict_eval = checkpoint_eval["model_weights"]
                 if '_orig_mod.' in list(_model_state_dict_eval.keys())[0]:
                     _model_state_dict_eval = {k.replace('_orig_mod.', ''): v for k, v in _model_state_dict_eval.items()}
                 network.load_state_dict(_model_state_dict_eval, strict=False) # Use non-strict loading for safety
                 print(f"Loaded model weights from {eval_checkpoint_path} for evaluation.")
            else: print("Warning: 'model_weights' not found in evaluation checkpoint.")
        except Exception as e: print(f"Error loading evaluation checkpoint {eval_checkpoint_path}: {e}")

    # Perform evaluation
    network.eval() # Ensure model is in eval mode
    if config.training_device != 'cpu': torch.cuda.empty_cache()
    with torch.no_grad():
        test_loggable_dict, test_text_to_save = evaluate('test', config, network, objective_evaluator)

    print("\n>> TEST RESULTS")
    print(test_loggable_dict)
    logger.log_metrics(test_loggable_dict, step=checkpoint.get("epochs_trained", 0), step_desc="test") # Log test metrics at final epoch
    # print(test_text_to_save) # Optionally print the smiles strings
    logger.text_artifact(os.path.join(config.results_path, "test_top_20_molecules.txt"),
                         test_text_to_save)
    # --- End Evaluation Phase ---

    print("\nFinished. Shutting down ray.")
    try: ray.shutdown()
    except Exception as e: print(f"Error shutting down Ray: {e}")
# --- End Main Execution Block ---