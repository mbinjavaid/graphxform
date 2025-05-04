import pickle
import random
import os
from typing import List, Tuple, Dict, Optional
import torch
from torch.utils.data import Dataset
from config import MoleculeConfig
# Crucially, this MUST be the UPDATED MoleculeDesign class with necessary methods
from molecule_design import MoleculeDesign, ActionType # Assuming ActionType might be needed if list_to_batch uses it


class TransformationMoleculeDataset(Dataset):
    """
    Dataset for supervised pretraining on molecule transformation sequences (A -> B).

    REVERTED STRUCTURE: Mimics RandomMoleculeDataset.
    - __init__ takes batch_size.
    - __len__ returns number of batches.
    - __getitem__ receives a batch index, reconstructs batch_size molecules,
      calls list_to_batch internally, and returns a complete batch dictionary.
    - Designed to be used with DataLoader(batch_size=1).
    """
    def __init__(self, config: MoleculeConfig, path_to_pickle: str,
                 # --- Arguments like old RandomMoleculeDataset ---
                 batch_size: int,
                 custom_num_batches: Optional[int] = None, # Keep optional support
                 no_random: bool = False,
                 # --- New argument, not in old, but potentially useful ---
                 is_validation: bool = False # Flag to disable random sampling if needed
                 ):
        """
        Args:
            config: MoleculeConfig instance.
            path_to_pickle: Path to the preprocessed transformations dictionary pickle file.
            batch_size: The number of individual samples to bundle into one batch.
            custom_num_batches: If set, overrides the length calculation.
            no_random: If True, selects samples sequentially for batches. If False, samples randomly.
            is_validation: If True, implies no_random=True (overrides no_random).
        """
        self.config = config
        self.path_to_pickle = path_to_pickle
        self.batch_size = batch_size
        self.custom_num_batches = custom_num_batches
        self.is_validation = is_validation
        # Validation implies deterministic order
        self.no_random = no_random or self.is_validation

        print(f"Loading transformation dataset from: {path_to_pickle}")
        if not os.path.exists(path_to_pickle):
             raise FileNotFoundError(f"Dataset pickle file not found at: {path_to_pickle}")

        try:
            with open(path_to_pickle, "rb") as f:
                self.transformations_dict: Dict[str, List[Tuple[str, str, List[int]]]] = pickle.load(f)
            if not isinstance(self.transformations_dict, dict):
                raise TypeError(f"Loaded data from {path_to_pickle} is not a dictionary.")
            print(f"Loaded data for {len(self.transformations_dict)} starting molecules.")
        except Exception as e:
            print(f"Error loading dataset pickle '{path_to_pickle}': {e}")
            raise

        # --- Create list of targets to sample (Same as before) ---
        # Each element: (start_smiles_key: str, transformation_list_idx: int, action_step_idx: int)
        self.targets_to_sample: List[Tuple[str, int, int]] = []
        # ... (Keep the loop to populate self.targets_to_sample as in your provided code) ...
        skipped_entries = 0
        total_transformations = 0
        print("Preprocessing dataset to identify sample points...")
        for start_smiles_key, transformations_list in self.transformations_dict.items():
            if not isinstance(transformations_list, list):
                 skipped_entries += 1
                 continue
            total_transformations += len(transformations_list)
            for trans_idx, transformation_data in enumerate(transformations_list):
                is_valid_format = (
                    isinstance(transformation_data, tuple) and len(transformation_data) == 3 and
                    isinstance(transformation_data[0], str) and isinstance(transformation_data[1], str) and
                    isinstance(transformation_data[2], list)
                )
                if not is_valid_format:
                     skipped_entries += 1
                     continue
                action_sequence = transformation_data[2]
                if not action_sequence: continue
                self.targets_to_sample.extend([(start_smiles_key, trans_idx, step_idx) for step_idx in range(len(action_sequence))])

        if skipped_entries > 0: print(f"Warning: Skipped {skipped_entries} invalid entries during dataset preprocessing.")
        if not self.targets_to_sample: raise ValueError("No valid action steps found.")

        # --- Calculate Length based on BATCHES (Like old RandomMoleculeDataset) ---
        num_total_samples = len(self.targets_to_sample)
        if custom_num_batches is not None:
            self.length = custom_num_batches
            print(f"Using custom_num_batches: {self.length}")
        elif self.batch_size > 0:
            # Ensure we don't drop the last partial batch if not shuffling,
            # but match old code's integer division for length.
            self.length = num_total_samples // self.batch_size
            if num_total_samples % self.batch_size != 0 and not self.no_random:
                 # If random sampling, we can draw enough samples anyway.
                 # If no_random, the last partial batch won't be reachable via index.
                 pass # Length calculation matches old script here.
            print(f"Dataset contains {num_total_samples} total action steps.")
            print(f"Batch size: {self.batch_size}, Num Batches (__len__): {self.length}")
        else:
             raise ValueError("batch_size must be positive.")

        # Optional: Pre-shuffle if no_random is True (matches old script behavior)
        # Note: If using custom_num_batches, this pre-shuffling might not be strictly necessary
        # if random sampling is used in __getitem__, but keep for consistency.
        if self.no_random:
             print("Note: Pre-shuffling sample list for deterministic epochs (no_random=True).")
             # This shuffle happens once at init if no_random is true, which seems counter-intuitive
             # but matches the old code's comment/logic pattern. Let's keep it.
             random.shuffle(self.targets_to_sample)


    def __len__(self):
        """Returns the total number of BATCHES in the dataset."""
        return self.length

    def __getitem__(self, idx: int) -> Dict:
        """
        Generates one full BATCH of training data.

        Args:
            idx: The index of the BATCH to generate.

        Returns:
            A dictionary structured like the old script's output:
            {
                'input': { output of MoleculeDesign.list_to_batch },
                'target_zero': Batch tensor for level 0 targets,
                'target_one': Batch tensor for level 1 targets,
                'target_two': Batch tensor for level 2 targets
            }

        Raises:
            IndexError: If idx is out of bounds for batches.
            RuntimeError: If molecule reconstruction fails.
        """
        if not 0 <= idx < self.length:
             # More specific error message for batch index
             raise IndexError(f"Batch index {idx} out of bounds for dataset with {self.length} batches.")

        partial_molecules: List[MoleculeDesign] = []   # Collect reconstructed molecules for the batch
        instance_target_actions: List[int] = []  # Collect corresponding target actions

        # --- Select individual samples for this batch ---
        if self.no_random:
            # Select a sequential slice of targets for this batch index
            start_idx = idx * self.batch_size
            # Make sure end_idx doesn't exceed total samples, especially for the last batch
            end_idx = min(start_idx + self.batch_size, len(self.targets_to_sample))
            batch_targets_to_process = self.targets_to_sample[start_idx:end_idx]
            # Handle potential last batch being smaller if needed (though length calculation might prevent reaching it)
            current_batch_size = len(batch_targets_to_process)
            if current_batch_size == 0 and self.length > 0: # Should not happen if length calc is right
                 raise RuntimeError(f"Calculated batch index {idx} resulted in zero samples.")

        else:
            # Randomly sample batch_size targets (with replacement, like old script)
            batch_targets_to_process = random.choices(self.targets_to_sample, k=self.batch_size)
            current_batch_size = self.batch_size # Assumes random.choices returns k items

        # --- Reconstruct molecules for the selected samples ---
        for i, target_info in enumerate(batch_targets_to_process):
            start_smiles_key, trans_idx, action_step_idx = target_info
            sample_identifier = f"batch_idx {idx}, sample {i}, source ('{start_smiles_key}', {trans_idx}, {action_step_idx})" # For errors

            try:
                # Retrieve sequence and target action
                transformation_data = self.transformations_dict[start_smiles_key][trans_idx]
                full_action_seq = transformation_data[2]
                target_action = full_action_seq[action_step_idx]

                # Initialize MoleculeDesign
                result_tuple = MoleculeDesign.from_smiles(self.config, start_smiles_key)
                if result_tuple is None:
                    raise RuntimeError(f"Failed to init from SMILES '{start_smiles_key}' for {sample_identifier}")
                molecule, _ = result_tuple

                # Apply actions up to the target step
                actions_to_apply = full_action_seq[:action_step_idx]
                for step_num, action in enumerate(actions_to_apply):
                    try:
                        molecule.take_action(action)
                    except Exception as e_take_action:
                         raise RuntimeError(f"Error during take_action({action}) at step {step_num} reconstructing state for {sample_identifier}: {e_take_action}")

                # Append successfully reconstructed molecule and its target action
                partial_molecules.append(molecule)
                instance_target_actions.append(target_action)

            except Exception as e_reconstruct:
                 # Catch errors during reconstruction for a single sample
                 print(f"\nERROR processing sample within batch: {e_reconstruct}")
                 # Option 1: Re-raise, stopping the batch creation (like old script implicitly did)
                 raise RuntimeError(f"Failed to process {sample_identifier}") from e_reconstruct
                 # Option 2: Skip this sample and continue (batch might be smaller than expected)
                 # print(f"Skipping problematic sample: {sample_identifier}")
                 # continue

        # --- Check if any molecules were successfully processed ---
        if not partial_molecules:
             # This could happen if all samples in the batch failed reconstruction
             raise RuntimeError(f"No molecules successfully reconstructed for batch index {idx}.")
             # OR return a dummy batch / handle differently if preferred

        # --- Call list_to_batch internally (like old script) ---
        # Use the dynamically padded version of list_to_batch
        # Pass include_feasibility_masks=True
        # Device can be CPU; transfer happens in training loop
        batch_input = MoleculeDesign.list_to_batch(
            list_of_samples=[{'molecule': m} for m in partial_molecules], # Adapt to list_to_batch input if needed
            device=torch.device("cpu"),
            # include_feasibility_masks=True # Assuming list_to_batch handles this implicitly now or is adapted
        )
        # NOTE: If your modified list_to_batch expects list_of_samples as [{'molecule': m, 'target_action': t}],
        # you might need to adjust the input structure here or modify list_to_batch further.
        # Let's assume list_to_batch primarily needs the molecules and generates targets separately below.

        # --- Create Target Tensors (like old script) ---
        # instance_target_actions contains the target action index for each molecule in the batch
        batch_targets = []
        ignore_index = -1
        for level in [0, 1, 2]:
            level_targets = []
            for i, target_act in enumerate(instance_target_actions):
                # Check the level of the corresponding reconstructed molecule
                mol = partial_molecules[i]
                if mol.current_action_level == level:
                    level_targets.append(target_act)
                else:
                    level_targets.append(ignore_index)
            # Convert list to tensor for this level
            batch_targets.append(torch.tensor(level_targets, dtype=torch.long)) # Use torch.tensor

        # --- Return the batch dictionary in the old format ---
        return dict(
            input=batch_input, # This now contains the batched inputs AND masks
            target_zero=batch_targets[0], # (B,) tensor
            target_one=batch_targets[1], # (B,) tensor
            target_two=batch_targets[2]  # (B,) tensor
        )


# --- Updated Test Block ---
if __name__ == "__main__":
    import traceback  # For printing stack traces on error
    from torch.utils.data import DataLoader  # To test dataset output format
    print("\n--- Running REVERTED TransformationMoleculeDataset Test ---")

    # --- Configuration ---
    # !! IMPORTANT: Replace with the ACTUAL path to your transformation pickle file !!
    TEST_DATA_PATH = "./data/chembl/transformation_datasets/transformations_train.pkl"
    # TEST_DATA_PATH = "./data/chembl/transformation_datasets/transformations_valid.pkl" # Or validation path

    TEST_BATCH_SIZE = 4       # Number of samples per batch for testing __getitem__
    NUM_BATCHES_TO_TEST = 3   # How many batches to fetch and inspect
    USE_RANDOM_SAMPLING = False # Test sequential fetching (no_random=True behavior)

    # --- Instantiate Config ---
    test_config = MoleculeConfig()
    # Ensure max_num_atoms is set, as MoleculeDesign might use it
    if not hasattr(test_config, 'max_num_atoms') or test_config.max_num_atoms is None:
         print("Warning: test_config.max_num_atoms not set, using default 50.")
         test_config.max_num_atoms = 50

    # --- Instantiate Dataset ---
    print(f"\nInstantiating REVERTED TransformationMoleculeDataset...")
    print(f"  Data path: {TEST_DATA_PATH}")
    print(f"  Batch size: {TEST_BATCH_SIZE}")
    print(f"  Random sampling (in __getitem__): {USE_RANDOM_SAMPLING}")

    dataset = None
    try:
        dataset = TransformationMoleculeDataset(
            config=test_config,
            path_to_pickle=TEST_DATA_PATH,
            batch_size=TEST_BATCH_SIZE,
            custom_num_batches=None, # Or set a number to test custom length
            no_random=not USE_RANDOM_SAMPLING
            # is_validation=False # Set True if testing validation behavior
        )
        print(f"Dataset instantiated successfully.")
        print(f"Total number of BATCHES (__len__): {len(dataset)}")
        print(f"Total number of individual samples available: {len(dataset.targets_to_sample)}")
    except FileNotFoundError as e:
        print(f"\nError: Dataset file not found at '{TEST_DATA_PATH}'.")
        print("Please ensure the path is correct and the file exists.")
        exit(1)
    except Exception as e:
        print(f"\nError during dataset instantiation: {e}")
        traceback.print_exc()
        exit(1)

    # --- Fetch and Inspect Batches (Using DataLoader(batch_size=1)) ---
    if dataset and len(dataset) > 0:
        print(f"\nFetching and inspecting {NUM_BATCHES_TO_TEST} batches using DataLoader(batch_size=1)...")

        # Use DataLoader with batch_size=1 to fetch pre-constructed batches from dataset.__getitem__
        # Shuffle=False is appropriate here because the dataset's no_random flag controls sampling order
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0) # num_workers=0 for simplicity in testing

        batches_inspected = 0
        for i, batch_data in enumerate(test_loader):
            if i >= NUM_BATCHES_TO_TEST:
                break
            batches_inspected += 1
            print(f"\n--- Batch {i+1} (from dataset index {i}) ---")
            try:
                # batch_data is the dictionary returned by dataset.__getitem__
                # wrapped in an extra list/batch dimension by DataLoader(batch_size=1)
                print(f"  Batch keys: {list(batch_data.keys())}")

                # Check 'input' dictionary (output of list_to_batch)
                if 'input' in batch_data and isinstance(batch_data['input'], dict):
                     input_dict = batch_data['input']
                     print(f"  Input keys: {list(input_dict.keys())}")
                     # Check shapes of tensors within 'input'
                     for key, value in input_dict.items():
                         if isinstance(value, torch.Tensor):
                             # Remove the extra batch_size=1 dimension added by DataLoader
                             actual_value = value[0]
                             actual_shape = actual_value.shape
                             # Determine expected batch dim based on whether it was the last batch
                             is_last_batch = (i == len(dataset) - 1)
                             if is_last_batch and len(dataset.targets_to_sample) % TEST_BATCH_SIZE != 0:
                                 expected_batch_dim = len(dataset.targets_to_sample) % TEST_BATCH_SIZE
                             else:
                                 expected_batch_dim = TEST_BATCH_SIZE

                             print(f"    Input['{key}'].shape: {tuple(actual_shape)} (Expected first dim: ~{expected_batch_dim})")
                         else:
                             print(f"    Input['{key}']: type {type(value)}")
                else:
                    print("  Error: 'input' key missing or not a dictionary.")

                # Check target tensors
                for key in ['target_zero', 'target_one', 'target_two']:
                    if key in batch_data and isinstance(batch_data[key], torch.Tensor):
                        # Remove extra batch_size=1 dimension
                        actual_value = batch_data[key][0]
                        actual_shape = actual_value.shape
                        print(f"  '{key}'.shape: {tuple(actual_shape)}")
                        # Check if length matches expected batch size
                        if len(actual_shape) > 0 and actual_shape[0] != expected_batch_dim:
                            print(f"    Warning: First dimension ({actual_shape[0]}) doesn't match expected batch size ({expected_batch_dim})")
                    else:
                        print(f"  Error: '{key}' key missing or not a Tensor.")

            except Exception as e:
                print(f"  Error processing batch {i}: {e}")
                traceback.print_exc()
                # Decide whether to continue testing or stop
                # break

        print(f"\n--- Test Complete ({batches_inspected} batches inspected) ---")

    elif dataset is not None:
         print("\nDataset instantiated but appears empty (length 0). Cannot fetch batches.")
    else:
        print("\nDataset instantiation failed. Cannot proceed with testing.")
