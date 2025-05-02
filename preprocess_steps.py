import argparse
import pickle
import os
from tqdm import tqdm
from config import MoleculeConfig
from molecule_design import MoleculeDesign, ActionType # Import ActionType if used
import time


def generate_precomputed_dataset(config_path, input_pickle_path, output_pickle_path):
    """
    Generates a new dataset by precomputing intermediate states from transformation sequences.
    Saves key config parameters alongside the data. Includes periodic logging.
    """
    start_time = time.time()
    # --- Load Config ---
    if config_path:
        # Implement loading config from path if needed
        raise NotImplementedError("Loading config from path not fully implemented yet")
    else:
        config = MoleculeConfig()
        print("Using default MoleculeConfig.")

    # --- Validate Essential Config Attributes ---
    if not hasattr(config, 'atom_vocabulary'): raise ValueError("Config missing 'atom_vocabulary'")
    if not hasattr(config, 'max_num_atoms') or config.max_num_atoms is None:
         raise ValueError("Config must provide 'max_num_atoms' (max REAL atoms).")
    vocab_size = len(config.atom_vocabulary)

    # --- Load Input Data ---
    print(f"Loading transformations from: {input_pickle_path}")
    if not os.path.exists(input_pickle_path):
        raise FileNotFoundError(f"Input pickle not found: {input_pickle_path}")
    with open(input_pickle_path, "rb") as f:
        transformations_dict = pickle.load(f)
    total_start_molecules = len(transformations_dict)
    print(f"Loaded {total_start_molecules} starting SMILES.")

    # --- Ensure Output Directory Exists ---
    output_dir = os.path.dirname(output_pickle_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Ensured output directory exists: {output_dir}")

    # --- Initialize Output and Counters ---
    precomputed_steps_data = [] # List to hold the step dictionaries
    total_steps_generated = 0
    skipped_sequences_init_error = 0
    skipped_sequences_action_error = 0
    processed_sequences = 0
    processed_start_molecules = 0 # Counter for logging
    log_interval = 10000 # Log every N starting molecules

    print("Generating precomputed steps...")
    # --- Process Sequences ---
    # Use enumerate on items() to track molecule count
    for mol_idx, (start_smiles, transformations_list) in enumerate(tqdm(transformations_dict.items(), desc="Processing SMILES", total=total_start_molecules)):
        processed_start_molecules += 1
        if not isinstance(transformations_list, list): continue

        # --- Periodic Logging ---
        if processed_start_molecules % log_interval == 0:
            elapsed_time = time.time() - start_time
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processed {processed_start_molecules}/{total_start_molecules} start molecules ({elapsed_time:.1f}s elapsed). "
                  f"Skipped (Init Err): {skipped_sequences_init_error}, Skipped (Action/Mask Err): {skipped_sequences_action_error}, Steps Gen: {total_steps_generated}")
        # --- End Periodic Logging ---


        for trans_idx, transformation_data in enumerate(transformations_list):
            try:
                # Basic format check
                if not (isinstance(transformation_data, tuple) and len(transformation_data) == 3 and isinstance(transformation_data[2], list)):
                    continue
                action_sequence = transformation_data[2]
                if not action_sequence: continue

                processed_sequences += 1
                sequence_failed_init = False # Track init failure separately for counting
                sequence_failed_action = False # Track action failure separately

                # 1. Initialize Molecule from start_smiles
                try:
                    mol, _ = MoleculeDesign.from_smiles(config, start_smiles)
                except Exception as e_init:
                    sequence_failed_init = True
                    # Don't increment skipped_sequences_init_error here yet, do it once per sequence below
                    continue # Skip to next transformation

                # 2. Iterate through actions to generate steps
                for step_idx, target_action in enumerate(action_sequence):
                    # a. Calculate mask for the CURRENT state (before taking target_action)
                    try:
                        mol.update_action_mask()
                    except Exception as e_mask:
                        sequence_failed_action = True
                        break # Stop processing this sequence

                    # b. Capture the state BEFORE target_action
                    state_dict = {
                        'atoms': mol.atoms.copy(),
                        'bonds': mol.bonds.copy(),
                        'is_original_atom': mol.is_original_atom.copy(),
                        'current_action_level': mol.current_action_level,
                        'l0_selected_atom_idx': mol.l0_selected_atom_idx,
                        'l1_action_type': mol.l1_action_type,
                        'l1_selected_existing_atom_idx': mol.l1_selected_existing_atom_idx,
                        'last_bond_action_details': mol.last_bond_action_details,
                        'current_action_mask': mol.current_action_mask.copy() if mol.current_action_mask is not None else None,
                        'target_action': target_action
                    }
                    precomputed_steps_data.append(state_dict)
                    total_steps_generated += 1

                    # c. Apply target_action to advance state for NEXT iteration
                    try:
                        mol.take_action(target_action)
                    except Exception as e_action:
                        sequence_failed_action = True
                        break # Stop processing this sequence

            except Exception as e_outer:
                # Catch any other unexpected error during processing of a single transformation tuple
                sequence_failed_action = True # Count as action error
                continue # Skip to next transformation
            finally:
                 # Increment skip counters once per sequence if it failed at any point
                 if sequence_failed_init:
                      skipped_sequences_init_error += 1
                 elif sequence_failed_action: # Only count action error if init didn't fail
                      skipped_sequences_action_error += 1


    # --- Final Summary ---
    total_time = time.time() - start_time
    print("\n--- Precomputation Summary ---")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Processed start molecules:     {processed_start_molecules}")
    print(f"Processed sequences (attempted): {processed_sequences}")
    print(f"Generated precomputed steps:     {total_steps_generated}")
    print(f"Skipped sequences (init failed): {skipped_sequences_init_error}")
    print(f"Skipped sequences (action/mask failed): {skipped_sequences_action_error}")
    print("-----------------------------")

    # --- Prepare Final Output Dictionary ---
    output_data_package = {
        'config_params': {
            'max_num_atoms': config.max_num_atoms,
            'vocab_size': vocab_size
        },
        'steps': precomputed_steps_data
    }

    # --- Save Output ---
    print(f"Saving {total_steps_generated} precomputed steps and config params to: {output_pickle_path}")
    try:
        with open(output_pickle_path, "wb") as f:
            pickle.dump(output_data_package, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Save complete.")
    except Exception as e_save:
        print(f"\nError saving output file: {e_save}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute training steps for molecule generation.")
    parser.add_argument('--config', type=str, default=None, help="Path to MoleculeConfig python file (optional).")
    parser.add_argument('--train-input', type=str, default="./data/chembl/transformation_datasets/transformations_train.pkl", help="Path to input training transformations pickle.")
    parser.add_argument('--valid-input', type=str, default="./data/chembl/transformation_datasets/transformations_valid.pkl", help="Path to input validation transformations pickle.")
    parser.add_argument('--train-output', type=str, default="./data/chembl/precomputed_steps/precomputed_train_steps.pkl", help="Path to save output precomputed training steps pickle.")
    parser.add_argument('--valid-output', type=str, default="./data/chembl/precomputed_steps/precomputed_valid_steps.pkl", help="Path to save output precomputed validation steps pickle.")
    parser.add_argument('--log-interval', type=int, default=10000, help="Log progress every N starting molecules.")


    cli_args = parser.parse_args()

    # Update log interval from args
    log_interval = cli_args.log_interval

    print("--- Starting Training Set Precomputation ---")
    generate_precomputed_dataset(
        config_path=cli_args.config,
        input_pickle_path=cli_args.train_input,
        output_pickle_path=cli_args.train_output
        # log_interval is handled internally now based on the variable
    )

    print("\n--- Starting Validation Set Precomputation ---")
    generate_precomputed_dataset(
        config_path=cli_args.config,
        input_pickle_path=cli_args.valid_input,
        output_pickle_path=cli_args.valid_output
        # log_interval is handled internally now
    )

    print("\n--- Precomputation Script Finished ---")