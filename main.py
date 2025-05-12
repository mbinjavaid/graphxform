import argparse
import copy
import importlib
import os
import time
import datetime

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from torch.cuda.amp import GradScaler, autocast

from tqdm import tqdm

from logger import Logger
# from molecule_dataset import RandomMoleculeDataset
from molecule_dataset import TransformationMoleculeDataset

os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
import ray
import numpy as np
from config import MoleculeConfig
from core.gumbeldore_dataset import GumbeldoreDataset # Ensure correct import path
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
    # --- Generation Step (Added Print Statements) ---
    print(f"\n--- Starting Gumbeldore dataset generation (Epoch {epoch+1})... ---") # <-- Added Print
    gen_start_time = time.time()
    gumbeldore_dataset = GumbeldoreDataset(
        config=config, objective_evaluator=objective_evaluator
    )
    # The generate_dataset call will now show worker progress logs
    metrics = gumbeldore_dataset.generate_dataset(
        network_weights, # Pass the frozen weights for generation
        best_objective=best_objective,
        memory_aggressive=False # Or get from config if needed
    )
    gen_duration = time.time() - gen_start_time
    print(f"--- Finished Gumbeldore dataset generation (Epoch {epoch+1}). Duration: {gen_duration:.2f}s ---") # <-- Added Print

    # --- Process Generation Metrics (Handle potential missing keys) ---
    print("Generated molecules - Summary:")
    print(f"  Mean obj. over fresh best mols: {metrics.get('mean_best_gen_obj', float('nan')):.3f}")
    print(f"  Best / worst obj. over fresh best mols: {metrics.get('best_gen_obj', float('-inf')):.3f}, {metrics.get('worst_gen_obj', float('nan')):.3f}")
    print(f"  Mean obj. over all time top 20 mols: {metrics.get('mean_top_20_obj', float('nan')):.3f}")
    top_20_list = metrics.get('top_20_molecules', [])
    if top_20_list and isinstance(top_20_list[0], dict):
         # Safely access the first value of the first dictionary in the list
         first_mol_dict = top_20_list[0]
         if first_mol_dict: # Check if the dictionary is not empty
             first_value = next(iter(first_mol_dict.values()), "N/A")
             if isinstance(first_value, (int, float)):
                 print(f"  All time best mol obj: {first_value:.3f}")
             else:
                 print(f"  All time best mol obj: {first_value}") # Print as is if not number
         else:
             print("  All time best mol obj: N/A (empty dict)")
    else: print("  All time best mol obj: N/A (no data or wrong format)")


    if torch.cuda.is_available(): torch.cuda.empty_cache()
    time.sleep(1) # Keep short delay
    # --- End Generation Step ---

    # --- Dataset and DataLoader Setup (Unchanged, specific to finetuning) ---
    print("---- Loading dataset for training")
    destination_file = config.gumbeldore_config["destination_path"]
    try:
        if not os.path.exists(destination_file):
             raise FileNotFoundError(f"Dataset file not found: {destination_file}")
        # Ensure the updated TransformationMoleculeDataset is used
        dataset = TransformationMoleculeDataset(config, destination_file, batch_size=config.batch_size_training,
                                        custom_num_batches=config.num_batches_per_epoch)
        print(f"Loaded dataset with {len(dataset)} batches.")
    except FileNotFoundError as e:
        print(f"ERROR: {e}. Skipping training for epoch {epoch+1}.")
        metrics["loss_level_zero"] = float('nan')
        metrics["loss_level_one"] = float('nan')
        metrics["loss_level_two"] = float('nan')
        metrics["full_loss"] = float('nan')
        top_20_molecules_text = "\n".join([f"{s}: {o:.4f}" for d in top_20_list for s, o in d.items()])
        return metrics, top_20_molecules_text
    except Exception as e:
         print(f"ERROR loading TransformationMoleculeDataset: {e}. Skipping training epoch.")
         metrics["loss_level_zero"] = float('nan'); metrics["loss_level_one"] = float('nan'); metrics["loss_level_two"] = float('nan'); metrics["full_loss"] = float('nan')
         top_20_molecules_text = "\n".join([f"{s}: {o:.4f}" for d in top_20_list for s, o in d.items()])
         return metrics, top_20_molecules_text

    # DataLoader setup specific to finetuning (batch_size=1)
    num_workers = getattr(config, 'num_dataloader_workers', 0)
    pin_memory_flag = getattr(config, 'pin_memory', config.training_device != "cpu")
    persistent_workers_flag = num_workers > 0 and getattr(config, 'persistent_workers', False)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, # Shuffle=True for training
                            num_workers=num_workers, pin_memory=pin_memory_flag,
                            persistent_workers=persistent_workers_flag)
    # --- End Dataset/DataLoader Setup ---

    # --- Training Loop Setup ---
    network.train() # Set network to training mode






    # # Freeze all parameters initially
    # for name, parameter in network.named_parameters():
        # parameter.requires_grad = False
    #
    # # Unfreeze only the final output linear layers
    # unfrozen_layers = []
    # # Correct layer names based on MoleculeTransformer definition
    # layers_to_unfreeze = [
    #     'linear_l0_terminate',
    #     'linear_l0_select_atom',
    #     'output_linear_level_one',
    #     'output_linear_level_two'
    # ]
    # for layer_name in layers_to_unfreeze:
    #     try:
    #         if hasattr(network, layer_name):
    #             layer = getattr(network, layer_name)
    #             # Check if the attribute is actually a PyTorch module
    #             if isinstance(layer, nn.Module):
    #                 for param in layer.parameters():
    #                     param.requires_grad = True
    #                 unfrozen_layers.append(layer_name)
    #                 print(f"Successfully unfrozen: {layer_name}")
    #             else:
    #                 print(f"Warning: Attribute '{layer_name}' found but is not an nn.Module.")
    #         else:
    #             print(f"Warning: Layer '{layer_name}' not found in network.")
    #     except Exception as e:
    #         print(f"Warning: Error accessing or unfreezing layer '{layer_name}': {e}")
    #
    # print(f"Unfrozen layers for training: {unfrozen_layers}")
    # if not unfrozen_layers:
    #     print("CRITICAL WARNING: No layers were unfrozen! Check layer names and model structure.")






    # added for AMP
    use_amp_here = (amp_enabled and scaler is not None and config.training_device != "cpu")

    accumulated_loss = 0
    accumulated_loss_lvl_zero = 0
    accumulated_loss_lvl_one = 0
    accumulated_loss_lvl_two = 0
    num_batches = len(dataloader)

    if num_batches == 0: # Handle empty dataloader case
        print("Warning: DataLoader created 0 batches. Skipping training loop.")
        metrics["loss_level_zero"] = float('nan')
        metrics["loss_level_one"] = float('nan')
        metrics["loss_level_two"] = float('nan')
        metrics["full_loss"] = float('nan')
        top_20_molecules_text = "\n".join([f"{s}: {o:.4f}" for d in top_20_list for s, o in d.items()])
        return metrics, top_20_molecules_text

    print(f"Starting training loop for {num_batches} batches...")
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
            input_data = {k: v[0].to(network.device, non_blocking=pin_memory_flag) for k, v in input_data_dict.items()}
            target_zero = data["target_zero"][0].to(network.device, non_blocking=pin_memory_flag)
            target_one = data["target_one"][0].to(network.device, non_blocking=pin_memory_flag)
            target_two = data["target_two"][0].to(network.device, non_blocking=pin_memory_flag)

            # Get masks directly from input_data (Ensure these keys exist)
            if "feasibility_mask_level_zero" not in input_data: raise KeyError("Missing 'feasibility_mask_level_zero' in batch input")
            if "feasibility_mask_level_one" not in input_data: raise KeyError("Missing 'feasibility_mask_level_one' in batch input")
            if "feasibility_mask_level_two" not in input_data: raise KeyError("Missing 'feasibility_mask_level_two' in batch input")
            mask_zero = input_data["feasibility_mask_level_zero"]
            mask_one = input_data["feasibility_mask_level_one"]
            mask_two = input_data["feasibility_mask_level_two"]
            # --- End Data Loading ---

            # --- Model Forward/Backward/Step (Conditional AMP) ---
            optimizer.zero_grad(set_to_none=True) # Zero gradients before forward pass

            with autocast(enabled=use_amp_here):
                logits_zero, logits_one, logits_two = network(input_data)

                # --- MASK APPLICATION LOGIC ---
                try:
                    # L0: Direct Masking
                    if logits_zero.shape[1] != mask_zero.shape[1]: raise ValueError(f"L0 shape mismatch: Logits {logits_zero.shape}, Mask {mask_zero.shape}")
                    logits_zero[mask_zero.bool()] = float("-inf")

                    # L1: Slice-then-Mask
                    batch_max_actions_l1 = mask_one.shape[1]
                    # Ensure logits_one has enough columns before slicing
                    if logits_one.shape[1] < batch_max_actions_l1:
                        # Pad logits_one if necessary (e.g., with -inf) or raise error
                        # This scenario might indicate a config mismatch (max_atoms vs actual)
                        # For now, raise error as padding might hide issues
                         raise ValueError(f"L1 shape mismatch: Fixed Logits dim ({logits_one.shape[1]}) < Dynamic Mask dim ({batch_max_actions_l1})")
                    logits_one_masked = logits_one[:, :batch_max_actions_l1] # Slice the relevant part
                    logits_one_masked[mask_one.bool()] = float("-inf") # Apply mask to the slice
                    # Update the original tensor (optional, depends if subsequent code uses logits_one directly)
                    logits_one[:, :batch_max_actions_l1] = logits_one_masked

                    # L2: Direct Masking
                    if logits_two.shape[1] != mask_two.shape[1]: raise ValueError(f"L2 shape mismatch: Logits {logits_two.shape}, Mask {mask_two.shape}")
                    logits_two[mask_two.bool()] = float("-inf")
                except (IndexError, ValueError, KeyError) as e: # Added KeyError
                     print(f"\nERROR applying mask (Epoch {epoch+1}, Batch {batch_idx}): {e}")
                     raise e
                # --- END MASK APPLICATION ---

                # --- Loss Calculation (inside autocast context) ---
                loss_zero = criterion(logits_zero, target_zero)
                # Clamp loss_zero if needed, or handle NaN appropriately
                loss_zero = torch.where(torch.isnan(loss_zero), torch.tensor(0.0, device=network.device), loss_zero)

                loss_one = criterion(logits_one, target_one)
                loss_one = torch.where(torch.isnan(loss_one), torch.tensor(0.0, device=network.device), loss_one)

                loss_two = criterion(logits_two, target_two)
                loss_two = torch.where(torch.isnan(loss_two), torch.tensor(0.0, device=network.device), loss_two)

                # Calculate combined loss
                loss = loss_zero + config.scale_factor_level_one * loss_one + config.scale_factor_level_two * loss_two
            # --- End autocast context ---

            # --- Optimization Step (Conditional Scaler) ---
            if use_amp_here:
                scaler.scale(loss).backward()
                # Gradient Clipping (Optional)
                if config.optimizer.get("gradient_clipping", 0) > 0:
                    scaler.unscale_(optimizer) # Unscale before clipping
                    params_to_clip = [p for p in network.parameters() if p.requires_grad]
                    if params_to_clip:
                         torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=config.optimizer["gradient_clipping"])
                scaler.step(optimizer) # Optimizer step
                scaler.update() # Update scaler
            else:
                loss.backward() # Standard backward pass
                # Gradient Clipping (Optional)
                if config.optimizer.get("gradient_clipping", 0) > 0:
                    params_to_clip = [p for p in network.parameters() if p.requires_grad]
                    if params_to_clip:
                         torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=config.optimizer["gradient_clipping"])
                optimizer.step() # Optimizer step
            # --- End Optimization Step ---

            # --- Accumulate Metrics ---
            batch_loss = loss.item()
            accumulated_loss += batch_loss
            accumulated_loss_lvl_zero += loss_zero.item()
            accumulated_loss_lvl_one += loss_one.item()
            accumulated_loss_lvl_two += loss_two.item()

            progress_bar.set_postfix({"batch_loss": f"{batch_loss:.4f}"})

            # --- Explicitly delete tensors to free memory ---
            del data, input_data, target_zero, target_one, target_two
            del mask_zero, mask_one, mask_two
            del logits_zero, logits_one, logits_two, loss, loss_zero, loss_one, loss_two
            if 'logits_one_masked' in locals(): del logits_one_masked # Delete if created
            # --- End Delete ---

        except StopIteration:
            print("DataLoader exhausted.")
            break
        except Exception as e:
            print(f"\nError during training batch {batch_idx} in epoch {epoch+1}: {e}")
            import traceback
            traceback.print_exc()
            continue # Continue to the next batch
    # --- End Batch Training Loop ---

    # --- Calculate Average Metrics ---
    metrics["loss_level_zero"] = accumulated_loss_lvl_zero / num_batches if num_batches > 0 else float('nan')
    metrics["loss_level_one"] = accumulated_loss_lvl_one / num_batches if num_batches > 0 else float('nan')
    metrics["loss_level_two"] = accumulated_loss_lvl_two / num_batches if num_batches > 0 else float('nan')
    metrics["full_loss"] = accumulated_loss / num_batches if num_batches > 0 else float('nan')

    top_20_molecules_text = "\n".join([f"{s}: {o:.4f}" for d in top_20_list for s, o in d.items()])
    metrics.pop("top_20_molecules", None) # Remove original list

    return metrics, top_20_molecules_text
# --- End train_for_one_epoch ---


def evaluate(eval_type: str, config: MoleculeConfig, network: MoleculeTransformer, objective_evaluator: MoleculeObjectiveEvaluator):
    """Evaluates the model by generating molecules and calculating metrics."""
    print(f"\n--- Starting Evaluation ({eval_type})... ---")
    eval_start_time = time.time()
    eval_config = copy.deepcopy(config) # Use a copy for evaluation settings
    eval_config.gumbeldore_config["destination_path"] = None # Don't save during eval

    gumbeldore_dataset = GumbeldoreDataset(
        config=eval_config, objective_evaluator=objective_evaluator
    )

    network.eval() # Ensure network is in eval mode
    with torch.no_grad():
        model_state_dict_eval = network.state_dict()
        # Handle potential prefix from torch.compile if needed
        # Check if the current network object is a compiled module proxy
        is_compiled = isinstance(network, torch.nn.Module) and hasattr(network, '_orig_mod')
        keys_have_prefix = any('_orig_mod.' in k for k in model_state_dict_eval.keys())

        if keys_have_prefix and not is_compiled:
             # Loading compiled weights into non-compiled model
             model_state_dict_eval = {k.replace('_orig_mod.', ''): v for k, v in model_state_dict_eval.items()}
        elif not keys_have_prefix and is_compiled:
             # Loading non-compiled weights into compiled model (should be handled by compile?)
             # This case might not require manual adjustment if compile handles it.
             # If errors occur, might need: model_state_dict_eval = {'_orig_mod.'+k: v for k,v in model_state_dict_eval.items()}
             pass

        network_weights_eval = copy.deepcopy(dict_to_cpu(model_state_dict_eval))


    # Call generate_dataset
    metrics = gumbeldore_dataset.generate_dataset(network_weights_eval, memory_aggressive=False)

    # Process metrics
    top_20_list_eval = metrics.pop("top_20_molecules", [])
    metrics_processed = {
        f"{eval_type}_mean_top_20_obj": metrics.get("mean_top_20_obj", float('nan')),
        f"{eval_type}_mean_top_20_sa_score": metrics.get("mean_top_20_sa_score", float('nan')),
        f"{eval_type}_best_obj": metrics.get('best_gen_obj', float('-inf')),
        f"{eval_type}_best_mol_sa_score": metrics.get('best_gen_sa_score', float('nan')),
    }
    eval_duration = time.time() - eval_start_time
    print(f"--- Evaluation ({eval_type}) finished. Duration: {eval_duration:.2f}s ---")
    print(f"  Eval ({eval_type}) best obj: {metrics_processed[f'{eval_type}_best_obj']:.3f}")
    print(f"  Eval ({eval_type}) mean top 20 obj: {metrics_processed[f'{eval_type}_mean_top_20_obj']:.3f}")

    top_20_text = "\n".join([f"{smiles}: {obj:.4f}" for d in top_20_list_eval for smiles, obj in d.items()])

    return metrics_processed, top_20_text


# --- Main Execution Block ---
if __name__ == '__main__':
    print(">> Molecule Design Finetuning")

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Experiment')
    parser.add_argument('--config', help="Path to optional config relative to main.py")
    parser.add_argument('--use-amp', action=argparse.BooleanOptionalAction, default=True, help="Enable Automatic Mixed Precision (AMP)")
    parser.add_argument('--use-compile', action=argparse.BooleanOptionalAction, default=False, help="Enable torch.compile (if available)")
    parser.add_argument('--run-name', type=str, help="run name for logging", default="Default_Finetune_Run")
    parser.add_argument('--exp-name', type=str, help="MLflow experiment name", default="Molecule_Finetuning")
    args = parser.parse_args()

    # --- Config Loading ---
    if args.config is not None:
        config_module_path = args.config.replace('.py', '').replace('/', '.')
        try:
            # Dynamically import the config class
            ConfigModule = importlib.import_module(config_module_path)
            if hasattr(ConfigModule, 'MoleculeConfig'):
                 MoleculeConfig = ConfigModule.MoleculeConfig
                 print(f"Loaded configuration from {args.config}")
            else: raise ImportError(f"'MoleculeConfig' class not found in {args.config}")
        except ImportError as e: print(f"Error loading config: {e}. Using default."); from config import MoleculeConfig
    else: from config import MoleculeConfig
    config = MoleculeConfig()

    # --- Set Config Defaults & Overrides ---
    # Ensure essential paths and parameters are set
    if not hasattr(config, 'results_path') or not config.results_path: config.results_path = f"./results/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.run_name}"
    if not hasattr(config, 'log_to_file'): config.log_to_file = True
    if not hasattr(config, 'seed'): config.seed = 42
    if not hasattr(config, 'training_device'): config.training_device = "cuda" if torch.cuda.is_available() else "cpu"
    if not hasattr(config, 'objective_gnn_device'): config.objective_gnn_device = config.training_device # Default to same device
    if not hasattr(config, 'num_dataloader_workers'): config.num_dataloader_workers = 0
    if not hasattr(config, 'pin_memory'): config.pin_memory = config.training_device != "cpu"
    if not hasattr(config, 'persistent_workers'): config.persistent_workers = False
    if not hasattr(config, 'optimizer'): config.optimizer = {
            "lr": 1e-4, "weight_decay": 0, "gradient_clipping": 1.,
            "schedule": {"decay_lr_every_epochs": 1, "decay_factor": 1}
        }
    if not hasattr(config, 'load_optimizer_state'): config.load_optimizer_state = True
    if not hasattr(config, 'scale_factor_level_one'): config.scale_factor_level_one = 1.
    if not hasattr(config, 'scale_factor_level_two'): config.scale_factor_level_two = 1.
    if not hasattr(config, 'num_epochs'): config.num_epochs = 100
    if not hasattr(config, 'batch_size_training'): config.batch_size_training = 64
    if not hasattr(config, 'num_batches_per_epoch'): config.num_batches_per_epoch = 500
    if not hasattr(config, 'gumbeldore_config'): raise ValueError("Config must contain 'gumbeldore_config' dictionary")
    if "destination_path" not in config.gumbeldore_config or not config.gumbeldore_config["destination_path"]:
         # Provide a default path if missing or empty
         default_data_path = "./data/generated_molecules.pickle"
         print(f"Warning: gumbeldore_config['destination_path'] not set, using default: {default_data_path}")
         config.gumbeldore_config["destination_path"] = default_data_path
         # Ensure directory exists
         os.makedirs(os.path.dirname(default_data_path), exist_ok=True)
    if "worker_log_interval" not in config.gumbeldore_config: config.gumbeldore_config["worker_log_interval"] = 1

    # --- Ray and Logger Setup ---
    num_gpus_ray = 0
    if hasattr(config, 'CUDA_VISIBLE_DEVICES') and config.CUDA_VISIBLE_DEVICES:
        try: num_gpus_ray = len(config.CUDA_VISIBLE_DEVICES.split(","))
        except: pass
    try:
        # Ensure Ray is initialized only once if running multiple times
        if not ray.is_initialized():
             ray.init(num_gpus=num_gpus_ray, logging_level="info")
             print(ray.available_resources())
        else: print("Ray already initialized.")
    except Exception as e: print(f"Ray initialization failed: {e}")

    os.makedirs(config.results_path, exist_ok=True)
    logger = Logger(args, config.results_path, config.log_to_file)
    logger.log_hyperparams(config)
    logger.log_hyperparams(vars(args))
    np.random.seed(config.seed); torch.manual_seed(config.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(config.seed)

    # --- Model and Evaluator Setup ---
    print("Setting up model and objective evaluator...")
    network = MoleculeTransformer(config, config.training_device)
    objective_evaluator = MoleculeObjectiveEvaluator(config, device=config.objective_gnn_device)

    # --- Conditional torch.compile ---
    compile_available = hasattr(torch, 'compile')
    compile_success = False
    if args.use_compile and compile_available:
        print("Attempting to compile model with torch.compile...")
        try:
            network = torch.compile(network, mode='default') # Consider 'reduce-overhead' or 'max-autotune'
            print("Model compiled successfully.")
            compile_success = True
        except Exception as e: print(f"torch.compile failed: {e}. Proceeding without compiling.")
    elif args.use_compile and not compile_available: print("torch.compile not available (requires PyTorch 2.0+).")
    else: print("torch.compile disabled via flag or unavailable.")

    print(f"Policy network is on device {config.training_device}")
    network.to(network.device)
    # Calculate trainable params *after* potential freezing in train_for_one_epoch
    # num_params=sum(p.numel() for p in network.parameters() if p.requires_grad); print(f"Trainable Params: {num_params:,}")

    # --- Scaler Setup ---
    scaler_enabled = args.use_amp and config.training_device != "cpu"
    scaler = GradScaler(enabled=scaler_enabled)
    print(f"AMP GradScaler enabled: {scaler_enabled}")

    # --- Checkpoint Loading ---
    checkpoint = None; start_epoch=0; best_validation_metric=float("-inf")
    load_path = getattr(config, 'load_checkpoint_from_path', None)

    if load_path and os.path.exists(load_path):
        print(f"Loading checkpoint from path {load_path}")
        try:
            checkpoint = torch.load(load_path, map_location=network.device)
            print(f"Checkpoint keys: {list(checkpoint.keys())}")

            if "model_weights" in checkpoint and checkpoint["model_weights"] is not None:
                _model_state_dict = checkpoint["model_weights"]
                # Handle compile prefix mismatch between checkpoint and current model state
                keys_have_prefix = any('_orig_mod.' in k for k in _model_state_dict.keys())
                if keys_have_prefix and not compile_success:
                   _model_state_dict = {k.replace('_orig_mod.', ''): v for k, v in _model_state_dict.items()}
                   print("Adjusted keys from compiled state_dict for non-compiled load.")
                elif not keys_have_prefix and compile_success:
                    # Need to add prefix if loading non-compiled weights into compiled model
                    _model_state_dict = {'_orig_mod.'+k: v for k,v in _model_state_dict.items()}
                    print("Adjusted keys from non-compiled state_dict for compiled model load.")

                # Load weights (use strict=False for flexibility, esp. during finetuning)
                missing_keys, unexpected_keys = network.load_state_dict(_model_state_dict, strict=False)
                if missing_keys: print(f"Warning: Missing keys in state_dict: {missing_keys}")
                if unexpected_keys: print(f"Warning: Unexpected keys in state_dict: {unexpected_keys}")
                print("Model weights loaded.")
            else: print("No 'model_weights' found or is None in checkpoint.")

            start_epoch = checkpoint.get("epochs_trained", 0)
            best_validation_metric = checkpoint.get("best_validation_metric", float("-inf"))

            # Load scaler state if available and enabled
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
        checkpoint = {"epochs_trained": 0, "best_validation_metric": float("-inf")}

    # --- Optimizer and Scheduler Setup ---
    # Note: params_to_optimize needs to be determined *after* freezing logic runs,
    # which happens inside train_for_one_epoch. We create the optimizer shell here,
    # but the actual parameters linked might be zero initially if resuming.
    # The parameters requiring grad will be correctly set within train_for_one_epoch before training starts.
    if config.num_epochs > 0:
        print("Setting up optimizer and LR scheduler shell.")
        # Create optimizer with potentially no parameters initially; they get set in train_for_one_epoch
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, network.parameters()), # Pass generator initially
            lr=config.optimizer["lr"],
            weight_decay=config.optimizer["weight_decay"]
        )

        # Load optimizer state if checkpoint exists and loading is enabled
        if checkpoint and config.load_optimizer_state and "optimizer_state" in checkpoint and checkpoint["optimizer_state"] is not None:
            print("Loading optimizer state from checkpoint.")
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                # Ensure optimizer state tensors are on the correct device
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor): state[k] = v.to(network.device)
                print("Optimizer state loaded and moved to device.")
            except Exception as e: print(f"Warn: Could not load optimizer state: {e}.")
        else: print("Starting with fresh optimizer state (or state will be loaded implicitly if resuming).")

        print("Setting up LR scheduler")
        schedule_config = config.optimizer.get("schedule", {"decay_factor": 1, "decay_lr_every_epochs": 1})
        _lambda = lambda e: schedule_config["decay_factor"] ** (e // schedule_config["decay_lr_every_epochs"])
        scheduler = LambdaLR(optimizer, lr_lambda=_lambda)
        # Set scheduler's last epoch correctly for resuming
        scheduler.last_epoch = start_epoch - 1
        print(f"Scheduler initial last_epoch set to {scheduler.last_epoch}")
    else:
        optimizer = None
        scheduler = None
        print("num_epochs is 0, skipping optimizer and scheduler setup.")

    # --- Training Loop ---
    if config.num_epochs > 0 and optimizer is not None:
        print(f"Starting training from epoch {start_epoch + 1} for {config.num_epochs} total epochs.")
        end_epoch = start_epoch + config.num_epochs # Calculate end epoch based on start and total

        # Load best weights if available in checkpoint (for tracking improvement)
        best_model_weights = checkpoint.get("best_model_weights")
        best_validation_metric = checkpoint.get("best_validation_metric", float("-inf"))

        start_time_counter = None
        if hasattr(config, 'wall_clock_limit') and config.wall_clock_limit is not None and config.wall_clock_limit > 0:
            print(f"Wall clock limit of training set to {config.wall_clock_limit / 3600:.2f} hours")
            start_time_counter = time.perf_counter()

        for epoch in range(start_epoch, end_epoch):
            current_epoch_num = epoch + 1
            print(f"\n------ Epoch {current_epoch_num}/{end_epoch} ------")

            # Get frozen network weights for generation step
            network.eval() # Ensure eval mode for getting weights
            with torch.no_grad():
                current_model_state_dict = network.state_dict()
                # Handle compile prefix if needed when getting weights for generation
                keys_have_prefix = any('_orig_mod.' in k for k in current_model_state_dict.keys())
                if keys_have_prefix: # Assume generation doesn't use compiled model
                   current_model_state_dict = {k.replace('_orig_mod.', ''): v for k, v in current_model_state_dict.items()}
                network_weights_for_gen = copy.deepcopy(dict_to_cpu(current_model_state_dict))

            # --- Run Training for One Epoch ---
            # Freezing/Unfreezing and optimizer param update happens inside here now
            generated_loggable_dict, generated_text_to_save = train_for_one_epoch(
                epoch, config, network, network_weights_for_gen, optimizer, objective_evaluator, best_validation_metric,
                scaler=scaler, amp_enabled=args.use_amp
            )
            # --- End Training ---

            # --- Scheduler Step
            if scheduler:
                scheduler.step()


            checkpoint["epochs_trained"] = current_epoch_num

            # --- Get Current LR (After scheduler step) ---
            if scheduler:
                try: current_lr = scheduler.get_last_lr()[0]
                except Exception: current_lr = optimizer.param_groups[0]['lr'] # Fallback
            else: current_lr = config.optimizer["lr"]
            # --- End Get Current LR ---

            print(f">> Epoch {current_epoch_num} finished. LR: {current_lr:.6f}")
            if 'full_loss' in generated_loggable_dict and not np.isnan(generated_loggable_dict['full_loss']):
                 print(f"   Avg Losses | Full: {generated_loggable_dict['full_loss']:.4f}, L0: {generated_loggable_dict['loss_level_zero']:.4f}, L1: {generated_loggable_dict['loss_level_one']:.4f}, L2: {generated_loggable_dict['loss_level_two']:.4f}")
            else: print("   Training skipped or failed for this epoch.")

            # --- Logging ---
            logger.log_metrics({"learning_rate": current_lr}, step=current_epoch_num)
            logger.log_metrics(generated_loggable_dict, step=current_epoch_num)
            if generated_text_to_save:
                try:
                    logger.text_artifact(os.path.join(config.results_path, f"epoch_{current_epoch_num}_train_top_20_molecules.txt"), generated_text_to_save)
                except Exception as log_e: print(f"Warning: Failed to save text artifact: {log_e}")
            # --- End Logging ---

            # --- Save Checkpoint ---
            network.eval() # Ensure consistent state for saving
            with torch.no_grad():
                 model_state_dict_to_save = network.state_dict()
                 # Handle compile prefix for saving consistency
                 keys_have_prefix_save = any('_orig_mod.' in k for k in model_state_dict_to_save.keys())
                 if keys_have_prefix_save: # Always save without prefix
                      model_state_dict_to_save = {k.replace('_orig_mod.', ''): v for k, v in model_state_dict_to_save.items()}

            checkpoint["model_weights"] = copy.deepcopy(dict_to_cpu(model_state_dict_to_save))
            # Save optimizer state (ensure it's on CPU)
            checkpoint["optimizer_state"] = copy.deepcopy(dict_to_cpu(optimizer.state_dict()))
            checkpoint["scaler_state"] = copy.deepcopy(dict_to_cpu(scaler.state_dict())) if scaler_enabled and scaler.is_enabled() else None

            # Use 'best_gen_obj' from the *current* epoch's generation as validation metric
            val_metric = generated_loggable_dict.get("best_gen_obj", float("-inf"))
            # Handle cases where generation might fail or return NaN
            if np.isnan(val_metric): val_metric = float("-inf")

            checkpoint["validation_metric"] = val_metric # Store current epoch's metric
            save_checkpoint(checkpoint, "last_model.pt", config)

            # --- Update Best Model ---
            if val_metric > best_validation_metric:
                print(f">> Got new best model (Obj: {val_metric:.4f} > {best_validation_metric:.4f}). Saving best_model.pt.")
                best_validation_metric = val_metric
                checkpoint["best_validation_metric"] = best_validation_metric
                # Save the weights that achieved this best metric
                checkpoint["best_model_weights"] = copy.deepcopy(checkpoint["model_weights"])
                # best_model_weights = checkpoint["best_model_weights"] # Update local var if needed elsewhere
                save_checkpoint(checkpoint, "best_model.pt", config)
            else:
                 print(f">> Validation metric ({val_metric:.4f}) did not improve from best ({best_validation_metric:.4f}).")
            # --- End Update Best Model ---

            if start_time_counter is not None and time.perf_counter() - start_time_counter > config.wall_clock_limit:
                print("Wall clock time limit exceeded. Stopping training.")
                break

        # --- End Epoch Loop ---
    elif config.num_epochs > 0:
         print("WARNING: Training loop skipped because optimizer setup failed or num_epochs <= 0.")


    # --- Evaluation Phase ---
    print("\n--- Evaluation Phase ---")
    eval_checkpoint_path = None
    model_loaded_for_eval = False

    # Determine which checkpoint to evaluate
    if config.num_epochs > 0: # If training occurred
        best_model_path = os.path.join(config.results_path, "best_model.pt")
        if os.path.exists(best_model_path):
            print(f"Evaluating with best model: {best_model_path}")
            eval_checkpoint_path = best_model_path
        else:
            print("Warning: No best_model.pt found after training. Evaluating with last model.")
            last_model_path = os.path.join(config.results_path, "last_model.pt")
            if os.path.exists(last_model_path): eval_checkpoint_path = last_model_path
            else: print("Warning: No last_model.pt found either. Evaluating with final model state.")
    elif load_path and os.path.exists(load_path): # No training, but checkpoint was loaded initially
        print(f"Evaluating with explicitly loaded model: {load_path}")
        # Network state should already be loaded from the initial checkpoint load
        model_loaded_for_eval = True # Flag that we don't need to load again
    else: # No training, no initial checkpoint
         print("Warning: No training performed and no checkpoint loaded. Evaluating with potentially random model.")

    # Load the chosen checkpoint if necessary
    if eval_checkpoint_path and not model_loaded_for_eval:
        print(f"Loading {eval_checkpoint_path} for evaluation...")
        try:
            checkpoint_eval = torch.load(eval_checkpoint_path, map_location=network.device)
            # Use 'best_model_weights' if available in best_model.pt, else 'model_weights'
            weights_key = "best_model_weights" if "best_model.pt" in eval_checkpoint_path else "model_weights"
            if weights_key in checkpoint_eval and checkpoint_eval[weights_key] is not None:
                 _model_state_dict_eval = checkpoint_eval[weights_key]
                 # Adjust keys for compile status mismatch if needed
                 keys_have_prefix_eval = any('_orig_mod.' in k for k in _model_state_dict_eval.keys())
                 if keys_have_prefix_eval and not compile_success: # Loading compiled into non-compiled
                     _model_state_dict_eval = {k.replace('_orig_mod.', ''): v for k,v in _model_state_dict_eval.items()}
                 elif not keys_have_prefix_eval and compile_success: # Loading non-compiled into compiled
                     _model_state_dict_eval = {'_orig_mod.'+k: v for k,v in _model_state_dict_eval.items()}

                 missing_keys, unexpected_keys = network.load_state_dict(_model_state_dict_eval, strict=False)
                 if missing_keys: print(f"Eval Load Warning: Missing keys: {missing_keys}")
                 if unexpected_keys: print(f"Eval Load Warning: Unexpected keys: {unexpected_keys}")
                 print(f"Loaded model weights from {eval_checkpoint_path} ({weights_key}) for evaluation.")
                 model_loaded_for_eval = True
            else: print(f"Warning: '{weights_key}' not found in evaluation checkpoint {eval_checkpoint_path}.")
        except Exception as e: print(f"Error loading evaluation checkpoint {eval_checkpoint_path}: {e}")

    # Perform evaluation
    if model_loaded_for_eval or not eval_checkpoint_path: # Proceed if model is loaded or no checkpoint was expected
        network.eval() # Ensure eval mode
        if config.training_device != 'cpu': torch.cuda.empty_cache()
        with torch.no_grad():
            test_loggable_dict, test_text_to_save = evaluate('test', config, network, objective_evaluator)

        print("\n>> TEST RESULTS")
        print(test_loggable_dict)
        # Log test metrics at final epoch or 0 if no training
        log_step = checkpoint.get("epochs_trained", 0) if checkpoint else 0
        logger.log_metrics(test_loggable_dict, step=log_step, step_desc="test")
        if test_text_to_save:
            try:
                logger.text_artifact(os.path.join(config.results_path, "test_top_20_molecules.txt"), test_text_to_save)
            except Exception as log_e: print(f"Warning: Failed to save test text artifact: {log_e}")
    else:
        print("Skipping evaluation as model weights could not be loaded.")


    print("\nFinished. Shutting down ray.")
    try:
        if ray.is_initialized(): ray.shutdown()
    except Exception as e: print(f"Error shutting down Ray: {e}")
