import pickle
import random
import os
from typing import List, Tuple, Dict, Optional
import torch
from torch.utils.data import Dataset
from config import MoleculeConfig
from molecule_design import MoleculeDesign


class TransformationMoleculeDataset(Dataset):
    """
    Dataset for supervised training on generated molecule action sequences.

    ADAPTED STRUCTURE for finetuning:
    - Loads data saved by GumbeldoreDataset (List[Dict]).
    - __init__ takes batch_size.
    - __len__ returns number of batches.
    - __getitem__ receives a batch index, reconstructs batch_size molecules based on start type,
      calls list_to_batch internally, and returns a complete batch dictionary.
    - Designed to be used with DataLoader(batch_size=1).
    """
    def __init__(self, config: MoleculeConfig, path_to_pickle: str,
                 batch_size: int,
                 custom_num_batches: Optional[int] = None,
                 no_random: bool = False,
                 is_validation: bool = False # Flag to disable random sampling if needed
                 ):
        """
        Args:
            config: MoleculeConfig instance.
            path_to_pickle: Path to the generated molecules pickle file (List[Dict]).
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
        self.no_random = no_random or self.is_validation

        print(f"Loading generated molecule data from: {path_to_pickle}")
        if not os.path.exists(path_to_pickle):
             raise FileNotFoundError(f"Dataset pickle file not found at: {path_to_pickle}")

        # --- MODIFIED: Load List[Dict] ---
        try:
            with open(path_to_pickle, "rb") as f:
                # Load the list of molecule dictionaries saved by GumbeldoreDataset
                self.molecule_data_list: List[Dict] = pickle.load(f)

            # Validate the loaded data format
            if not isinstance(self.molecule_data_list, list):
                raise TypeError(f"Loaded data from {path_to_pickle} is not a list.")
            if self.molecule_data_list and not isinstance(self.molecule_data_list[0], dict):
                 # Check the first element's type if list is not empty
                 raise TypeError(f"Loaded data from {path_to_pickle} does not appear to be a list of dictionaries.")
            print(f"Loaded data for {len(self.molecule_data_list)} generated molecules.")
        except Exception as e:
            print(f"Error loading dataset pickle '{path_to_pickle}': {e}")
            raise
        # --- END MODIFIED ---

        # --- MODIFIED: Create list of targets to sample from List[Dict] ---
        # Each element: (molecule_list_idx: int, action_step_idx: int, start_atom: int, start_smiles_key: Optional[str])
        self.targets_to_sample: List[Tuple[int, int, int, Optional[str]]] = [] # <-- MODIFIED TUPLE STRUCTURE
        skipped_entries = 0
        print("Preprocessing dataset to identify sample points...")
        for mol_idx, molecule_data in enumerate(self.molecule_data_list):
            if not isinstance(molecule_data, dict):
                 print(f"Warning: Skipping item at index {mol_idx}, not a dictionary.")
                 skipped_entries += 1; continue

            action_sequence = molecule_data.get("action_seq")
            # --- ADDED: Extract start info ---
            start_atom = molecule_data.get("start_atom")
            start_smiles_key = molecule_data.get("start_smiles_key") # Can be None
            # --- END ADDED ---

            # Check if action_sequence and start_atom are valid
            # (start_smiles_key being None is okay)
            if not isinstance(action_sequence, list) or not action_sequence or start_atom is None:
                 # smiles_for_log = molecule_data.get("smiles", f"index_{mol_idx}")
                 # print(f"Warning: Skipping molecule '{smiles_for_log}' due to missing/empty action sequence or start_atom.")
                 skipped_entries += 1; continue # Skip if no valid action sequence or start atom

            # Add a target for each step, including start info
            self.targets_to_sample.extend([(mol_idx, step_idx, start_atom, start_smiles_key)
                                            for step_idx in range(len(action_sequence))])

        if skipped_entries > 0: print(f"Warning: Skipped {skipped_entries} invalid entries during dataset preprocessing.")
        if not self.targets_to_sample: raise ValueError("No valid action steps found in the loaded molecule data.")
        # --- END MODIFIED ---

        # --- Calculate Length based on BATCHES (Logic Unchanged) ---
        num_total_samples = len(self.targets_to_sample)
        if custom_num_batches is not None:
            self.length = custom_num_batches
            print(f"Using custom_num_batches: {self.length}")
        elif self.batch_size > 0:
            self.length = num_total_samples // self.batch_size
            if num_total_samples % self.batch_size != 0 and not self.no_random:
                 pass # Length calculation matches old script here.
            print(f"Dataset contains {num_total_samples} total action steps.")
            print(f"Batch size: {self.batch_size}, Num Batches (__len__): {self.length}")
        else:
             raise ValueError("batch_size must be positive.")

        # --- Pre-shuffling logic (Unchanged) ---
        if self.no_random:
             print("Note: Pre-shuffling sample list for deterministic epochs (no_random=True).")
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
            A dictionary structured for training input/targets.
        """
        if not 0 <= idx < self.length:
             raise IndexError(f"Batch index {idx} out of bounds for dataset with {self.length} batches.")

        partial_molecules: List[MoleculeDesign] = []
        instance_target_actions: List[int] = []

        # --- Select individual samples for this batch (Logic Unchanged) ---
        if self.no_random:
            start_idx = idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(self.targets_to_sample))
            batch_targets_to_process = self.targets_to_sample[start_idx:end_idx]
            current_batch_size = len(batch_targets_to_process)
            if current_batch_size == 0 and self.length > 0:
                 raise RuntimeError(f"Calculated batch index {idx} resulted in zero samples.")
        else:
            batch_targets_to_process = random.choices(self.targets_to_sample, k=self.batch_size)
            current_batch_size = self.batch_size

        # --- MODIFIED: Reconstruct molecules for the selected samples ---
        for i, target_info in enumerate(batch_targets_to_process):
            # --- ADDED: Unpack all elements ---
            mol_list_idx, action_step_idx, start_atom, start_smiles_key = target_info
            # --- END ADDED ---
            sample_identifier = f"batch_idx {idx}, sample {i}, source (list_idx {mol_list_idx}, step {action_step_idx})"

            try:
                # Retrieve molecule data and action sequence
                molecule_data = self.molecule_data_list[mol_list_idx]
                full_action_seq = molecule_data.get("action_seq")

                # Validate data needed for reconstruction
                if full_action_seq is None: # start_atom/start_smiles_key checked during init
                     raise ValueError(f"Missing 'action_seq' for {sample_identifier}")

                target_action = full_action_seq[action_step_idx]

                # --- START CONDITIONAL INITIALIZATION ---
                molecule = None
                if start_smiles_key is not None:
                    # Initialize from SMILES
                    try:
                        result_tuple = MoleculeDesign.from_smiles(self.config, start_smiles_key)
                        if result_tuple is None:
                            raise RuntimeError(f"MoleculeDesign.from_smiles returned None for SMILES '{start_smiles_key}'")
                        molecule, _ = result_tuple # Assuming it returns (mol, bool)
                    except Exception as e_init_smiles:
                         raise RuntimeError(f"Failed to initialize from SMILES '{start_smiles_key}' for {sample_identifier}: {e_init_smiles}")
                else:
                    # Initialize from start_atom
                    try:
                        # Assumes the standard constructor works: __init__(self, config, initial_atom)
                        molecule = MoleculeDesign(config=self.config, initial_atom=start_atom)
                    except Exception as e_init_atom:
                         raise RuntimeError(f"Failed to initialize MoleculeDesign with start_atom={start_atom} for {sample_identifier}: {e_init_atom}")
                # --- END CONDITIONAL INITIALIZATION ---

                if molecule is None: # Should not happen if exceptions are raised correctly
                    raise RuntimeError(f"Molecule initialization failed unexpectedly for {sample_identifier}")

                # Apply actions up to the target step (logic unchanged)
                actions_to_apply = full_action_seq[:action_step_idx]
                for step_num, action in enumerate(actions_to_apply):
                    try:
                        # Assuming take_action modifies the molecule in-place
                        molecule.take_action(action)
                    except Exception as e_take_action:
                         # Log more details about the state before failure if possible
                         # current_smiles = molecule.smiles_string # Might fail if state is invalid
                         raise RuntimeError(f"Error during take_action({action}) at step {step_num} reconstructing state for {sample_identifier}: {e_take_action}")

                # Append successfully reconstructed molecule and its target action (logic unchanged)
                partial_molecules.append(molecule)
                instance_target_actions.append(target_action)

            except Exception as e_reconstruct:
                 print(f"\nERROR processing sample within batch: {e_reconstruct}")
                 raise RuntimeError(f"Failed to process {sample_identifier}") from e_reconstruct
        # --- END MODIFIED ---

        if not partial_molecules:
             raise RuntimeError(f"No molecules successfully reconstructed for batch index {idx}.")

        # --- Call list_to_batch internally (Logic Unchanged) ---
        # Assumes list_to_batch is compatible with the reconstructed partial_molecules
        try:
            # Assuming list_to_batch takes a list of dicts {'molecule': mol_obj}
            batch_input = MoleculeDesign.list_to_batch(
                list_of_samples=[{'molecule': m} for m in partial_molecules],
                device=torch.device("cpu"),
                # include_feasibility_masks=True # Assuming list_to_batch handles this
            )
        except Exception as e_list_to_batch:
             print(f"\nERROR during MoleculeDesign.list_to_batch for batch index {idx}: {e_list_to_batch}")
             # Optionally inspect partial_molecules here
             raise RuntimeError(f"list_to_batch failed for batch index {idx}") from e_list_to_batch


        # --- Create Target Tensors (Logic Unchanged) ---
        batch_targets = []
        ignore_index = -1
        for level in [0, 1, 2]:
            level_targets = []
            for i, target_act in enumerate(instance_target_actions):
                mol = partial_molecules[i]
                # Ensure current_action_level attribute exists
                current_level = getattr(mol, 'current_action_level', -1)
                if current_level == level:
                    level_targets.append(target_act)
                else:
                    level_targets.append(ignore_index)
            batch_targets.append(torch.tensor(level_targets, dtype=torch.long))

        # --- Return the batch dictionary (Structure Unchanged) ---
        return dict(
            input=batch_input,
            target_zero=batch_targets[0],
            target_one=batch_targets[1],
            target_two=batch_targets[2]
        )


# --- Test Block (Adapted for new data format) ---
if __name__ == "__main__":
    import traceback
    from torch.utils.data import DataLoader
    print("\n--- Running ADAPTED TransformationMoleculeDataset Test ---")

    # --- Configuration ---
    # !! IMPORTANT: Update this path to your actual generated molecules pickle !!
    # This should be the List[Dict] file saved by GumbeldoreDataset
    TEST_DATA_PATH = "./data/generated_molecules.pickle"

    TEST_BATCH_SIZE = 4
    NUM_BATCHES_TO_TEST = 3
    USE_RANDOM_SAMPLING = False # Test sequential fetching (no_random=True behavior)

    # --- Instantiate Config ---
    test_config = MoleculeConfig()
    # Ensure max_num_atoms is set, as MoleculeDesign might use it
    if not hasattr(test_config, 'max_num_atoms') or test_config.max_num_atoms is None:
         print("Warning: test_config.max_num_atoms not set, using default 50.")
         test_config.max_num_atoms = 50

    # --- Instantiate Dataset ---
    print(f"\nInstantiating ADAPTED TransformationMoleculeDataset...")
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
    except ValueError as e: # Catch empty dataset error
        print(f"\nError during dataset instantiation: {e}")
        print("Check if the pickle file contains valid action sequences.")
        exit(1)
    except Exception as e:
        print(f"\nError during dataset instantiation: {e}")
        traceback.print_exc()
        exit(1)

    # --- Fetch and Inspect Batches (Using DataLoader(batch_size=1)) ---
    if dataset and len(dataset) > 0:
        print(f"\nFetching and inspecting {min(NUM_BATCHES_TO_TEST, len(dataset))} batches using DataLoader(batch_size=1)...")

        # Use DataLoader with batch_size=1 to fetch pre-constructed batches from dataset.__getitem__
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        batches_inspected = 0
        for i, batch_data in enumerate(test_loader):
            if i >= NUM_BATCHES_TO_TEST:
                break
            batches_inspected += 1
            print(f"\n--- Batch {i+1} (from dataset index {i}) ---")
            try:
                # Determine expected batch dim for checks
                # This needs to be calculated carefully for the last batch
                is_last_batch = (i == len(dataset) - 1)
                num_total_samples = len(dataset.targets_to_sample)
                if is_last_batch and num_total_samples % TEST_BATCH_SIZE != 0:
                    # Last batch might be smaller if not using random sampling
                    expected_batch_dim = num_total_samples % TEST_BATCH_SIZE
                else:
                    # Otherwise, it should be the full batch size
                    expected_batch_dim = TEST_BATCH_SIZE
                # Handle edge case where dataset length is 0 but somehow we entered loop
                if expected_batch_dim == 0 and num_total_samples > 0:
                    expected_batch_dim = TEST_BATCH_SIZE

                print(f"  Batch keys: {list(batch_data.keys())}")
                if 'input' in batch_data and isinstance(batch_data['input'], dict):
                     input_dict = batch_data['input']
                     print(f"  Input keys: {list(input_dict.keys())}")
                     for key, value in input_dict.items():
                         if isinstance(value, torch.Tensor):
                             actual_value = value[0] # Remove DataLoader's batch dim
                             actual_shape = actual_value.shape
                             print(f"    Input['{key}'].shape: {tuple(actual_shape)} (Expected first dim: {expected_batch_dim})")
                             # Check if first dimension matches expectation
                             if len(actual_shape) > 0 and actual_shape[0] != expected_batch_dim:
                                 print(f"      WARNING: Dim 0 mismatch! Got {actual_shape[0]}")
                         else: print(f"    Input['{key}']: type {type(value)}")
                else: print("  Error: 'input' key missing or not a dictionary.")

                for key in ['target_zero', 'target_one', 'target_two']:
                    if key in batch_data and isinstance(batch_data[key], torch.Tensor):
                        actual_value = batch_data[key][0] # Remove DataLoader's batch dim
                        actual_shape = actual_value.shape
                        print(f"  '{key}'.shape: {tuple(actual_shape)}")
                        # Check if length matches expected batch size
                        if len(actual_shape) > 0 and actual_shape[0] != expected_batch_dim:
                            print(f"    Warning: First dimension ({actual_shape[0]}) doesn't match expected batch size ({expected_batch_dim})")
                    else: print(f"  Error: '{key}' key missing or not a Tensor.")

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
