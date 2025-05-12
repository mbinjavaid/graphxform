import pickle
import random
import os
from typing import List, Tuple, Dict, Optional
import torch
from torch.utils.data import Dataset
from config import MoleculeConfig
from molecule_design import MoleculeDesign  # Ensure ActionType is accessible if used


class TransformationMoleculeDataset(Dataset):
    """
    Dataset for supervised training on generated molecule action sequences.
    Handles loading data in:
    1. New List[Dict] format (from GumbeldoreDataset used in finetuning).
       Each Dict: {"action_seq": List[int], "start_atom": int, "start_smiles_key": Optional[str], ...}
    2. Pretraining Dict[str, List[Tuple]] format (from transformation generator script).
       Dict: {initial_smiles: List[(initial_smiles, final_smiles, action_seq_list)]}
    """

    def __init__(self, config: MoleculeConfig, path_to_pickle: str,
                 batch_size: int,
                 custom_num_batches: Optional[int] = None,
                 no_random: bool = False,
                 is_validation: bool = False
                 ):
        self.config = config
        self.path_to_pickle = path_to_pickle
        self.batch_size = batch_size
        self.custom_num_batches = custom_num_batches
        self.is_validation = is_validation
        self.no_random = no_random or self.is_validation

        # Determine a default start_atom if SMILES isn't provided (e.g., for Gumbeldore's "from scratch")
        # Or for pretraining tuples where start_atom isn't explicitly saved with the tuple.
        # Let's use 1 (assuming it's a valid 1-based vocab index, e.g., Carbon if it's the first).
        # This is primarily a placeholder if start_smiles_key is used for initialization.
        self.default_initial_atom_type_idx = 1
        if config.atom_vocabulary:
            first_atom_name = next(iter(config.atom_vocabulary))  # Get the first key
            # Vocab indices are 1-based for MoleculeDesign constructor
            self.default_initial_atom_type_idx = list(config.atom_vocabulary.keys()).index(first_atom_name) + 1

        print(f"Loading and processing dataset from: {path_to_pickle}")
        if not os.path.exists(path_to_pickle):
            raise FileNotFoundError(f"Dataset pickle file not found at: {path_to_pickle}")

        try:
            with open(path_to_pickle, "rb") as f:
                loaded_data_from_pickle = pickle.load(f)
        except Exception as e:
            print(f"Error loading dataset pickle '{path_to_pickle}': {e}")
            raise

        raw_molecule_entries_list = []  # This will hold either Dicts or Tuples

        if isinstance(loaded_data_from_pickle, list):
            # Assumed Gumbeldore format: List[Dict]
            raw_molecule_entries_list = loaded_data_from_pickle
            print(f"Loaded data is a list with {len(raw_molecule_entries_list)} entries (assumed Gumbeldore format).")
        elif isinstance(loaded_data_from_pickle, dict):
            # Assumed pretraining generator format: Dict[str, List[Tuple[str, str, List[int]]]]
            print(
                "Loaded data is a dictionary (assumed pretraining generator format). Extracting transformation tuples...")
            count_transformations = 0
            for list_of_tuples in loaded_data_from_pickle.values():
                if isinstance(list_of_tuples, list):
                    for item_tuple in list_of_tuples:
                        if isinstance(item_tuple, tuple) and len(item_tuple) == 3:
                            raw_molecule_entries_list.append(item_tuple)
                            count_transformations += 1
                        # else: # Optional: Log malformed tuples within the list
                        #     print(f"Warning: Skipping non-tuple or incorrectly sized tuple item: {type(item_tuple)}")
            print(f"Extracted {count_transformations} transformation tuples from the dictionary.")
        else:
            raise TypeError(
                f"Loaded data from '{path_to_pickle}' is not a list or a supported dictionary structure. Type found: {type(loaded_data_from_pickle)}")

        if not raw_molecule_entries_list:
            print(f"Warning: No molecule entries found or extracted from '{path_to_pickle}'. Dataset will be empty.")

        self.processed_molecule_data: List[Dict] = []
        skipped_entries_parsing = 0

        for entry_idx, raw_entry in enumerate(raw_molecule_entries_list):
            action_seq = None
            start_atom_val = None
            start_smiles_key_val = None

            if isinstance(raw_entry, dict):  # Gumbeldore format List[Dict]
                action_seq = raw_entry.get("action_seq")
                start_atom_val = raw_entry.get("start_atom")  # This should be present
                start_smiles_key_val = raw_entry.get("start_smiles_key")  # Can be None
                if start_atom_val is None and start_smiles_key_val is None:  # Must have one way to init
                    skipped_entries_parsing += 1
                    continue
                if start_atom_val is None and start_smiles_key_val is not None:  # If only SMILES, provide default atom for consistency
                    start_atom_val = self.default_initial_atom_type_idx

            elif isinstance(raw_entry, tuple) and len(raw_entry) == 3:  # Pretraining generator format Tuple
                # entry is (initial_smiles_str, final_smiles_str, action_sequence_list)
                start_smiles_key_val = raw_entry[0]
                # final_smiles_val = raw_entry[1] # Not directly used by dataset for input construction
                action_seq = raw_entry[2]
                # For this format, start_smiles_key_val is always present.
                # Set start_atom_val to a default, as MoleculeDesign.from_smiles will be used.
                start_atom_val = self.default_initial_atom_type_idx

                if not isinstance(start_smiles_key_val,
                                  str) or not start_smiles_key_val:  # Ensure SMILES is valid string
                    skipped_entries_parsing += 1
                    continue
            else:
                skipped_entries_parsing += 1
                continue

            if not isinstance(action_seq, list) or not action_seq:
                skipped_entries_parsing += 1
                continue

            self.processed_molecule_data.append({
                "action_seq": action_seq,
                "start_atom": start_atom_val,  # Will be a valid int or the default
                "start_smiles_key": start_smiles_key_val  # Can be None (only for Gumbeldore's "from scratch")
            })

        if skipped_entries_parsing > 0:
            print(f"Warning: Skipped {skipped_entries_parsing} entries during format parsing and validation.")

        if not self.processed_molecule_data:
            print(
                f"Warning: After processing, no valid molecule entries could be prepared from {path_to_pickle}. Dataset will be empty.")

        self.targets_to_sample: List[Tuple[int, int, int, Optional[str]]] = []
        for mol_idx, processed_data_entry in enumerate(self.processed_molecule_data):
            action_sequence = processed_data_entry["action_seq"]
            start_atom = processed_data_entry["start_atom"]  # This is now always an int
            start_smiles_key = processed_data_entry["start_smiles_key"]

            self.targets_to_sample.extend([(mol_idx, step_idx, start_atom, start_smiles_key)
                                           for step_idx in range(len(action_sequence))])

        if not self.targets_to_sample and self.processed_molecule_data:
            raise ValueError("Processed molecule data was available, but no valid action steps could be extracted.")
        if not self.targets_to_sample and not self.processed_molecule_data:
            print("Warning: No valid action steps found; dataset effectively empty.")

        num_total_samples = len(self.targets_to_sample)
        if custom_num_batches is not None:
            self.length = custom_num_batches
            print(f"Using custom_num_batches: {self.length}")
        elif self.batch_size > 0:
            self.length = num_total_samples // self.batch_size
            print(f"Dataset contains {num_total_samples} total action steps (potential training samples).")
            print(f"Batch size: {self.batch_size}, Num Batches (__len__): {self.length}")
        else:
            raise ValueError("batch_size must be positive.")

        if self.length == 0 and num_total_samples > 0 and custom_num_batches is None:
            print(
                f"Warning: Calculated dataset length is 0 (samples: {num_total_samples}, batch_size: {self.batch_size}).")

        if self.no_random and self.targets_to_sample:
            random.shuffle(self.targets_to_sample)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Dict:
        if not 0 <= idx < self.length:
            raise IndexError(f"Batch index {idx} out of bounds for dataset with {self.length} batches.")

        partial_molecules: List[MoleculeDesign] = []
        instance_target_actions: List[int] = []

        if self.no_random:
            start_idx_in_targets = idx * self.batch_size
            end_idx_in_targets = min(start_idx_in_targets + self.batch_size, len(self.targets_to_sample))
            batch_targets_to_process = self.targets_to_sample[start_idx_in_targets:end_idx_in_targets]
        else:
            if not self.targets_to_sample:
                raise RuntimeError(
                    "Attempting to sample from an empty targets_to_sample list (should be caught by __len__==0).")
            batch_targets_to_process = random.choices(self.targets_to_sample, k=self.batch_size)

        if not batch_targets_to_process:  # If after slicing/sampling, it's empty
            raise RuntimeError(
                f"Batch index {idx} yielded no targets to process. Dataset length: {self.length}, total targets: {len(self.targets_to_sample)}")

        for i, target_info in enumerate(batch_targets_to_process):
            mol_processed_idx, action_step_idx, start_atom_for_init, start_smiles_key_for_init = target_info
            sample_identifier = f"batch_idx {idx}, sample {i}, source (proc_idx {mol_processed_idx}, step {action_step_idx})"

            try:
                processed_data_entry = self.processed_molecule_data[mol_processed_idx]
                full_action_seq = processed_data_entry["action_seq"]

                target_action = full_action_seq[action_step_idx]
                molecule = None

                # Prioritize initialization from SMILES if available
                if start_smiles_key_for_init is not None and start_smiles_key_for_init != "":
                    try:
                        result_tuple = MoleculeDesign.from_smiles(self.config, start_smiles_key_for_init)
                        if result_tuple is None:
                            raise RuntimeError(
                                f"MoleculeDesign.from_smiles returned None for SMILES '{start_smiles_key_for_init}'")
                        molecule, _ = result_tuple
                    except Exception as e_init_smiles:
                        raise RuntimeError(
                            f"Failed to initialize from SMILES '{start_smiles_key_for_init}' for {sample_identifier}: {e_init_smiles}")
                else:  # Fallback to initializing from start_atom_for_init
                    try:
                        # start_atom_for_init should be a valid 1-based vocab index
                        molecule = MoleculeDesign(config=self.config, initial_atom=start_atom_for_init)
                    except Exception as e_init_atom:
                        raise RuntimeError(
                            f"Failed to initialize MoleculeDesign with start_atom={start_atom_for_init} for {sample_identifier}: {e_init_atom}")

                if molecule is None:  # Should be caught by specific exceptions above
                    raise RuntimeError(f"Molecule initialization failed unexpectedly for {sample_identifier}")

                actions_to_apply = full_action_seq[:action_step_idx]
                for step_num, action_val in enumerate(actions_to_apply):
                    try:
                        molecule.take_action(action_val)
                    except Exception as e_take_action:
                        raise RuntimeError(
                            f"Error during take_action({action_val}) at step {step_num} for {sample_identifier}: {e_take_action}")

                partial_molecules.append(molecule)
                instance_target_actions.append(target_action)

            except Exception as e_reconstruct:
                print(f"\nERROR processing sample within batch: {e_reconstruct}")
                raise RuntimeError(f"Failed to process {sample_identifier}") from e_reconstruct

        if not partial_molecules:
            raise RuntimeError(
                f"No molecules successfully reconstructed for batch index {idx}, though targets were selected.")

        try:
            batch_input = MoleculeDesign.list_to_batch(
                list_of_samples=[{'molecule': m} for m in partial_molecules],
                device=torch.device("cpu"),
            )
        except Exception as e_list_to_batch:
            print(f"\nERROR during MoleculeDesign.list_to_batch for batch index {idx}: {e_list_to_batch}")
            raise RuntimeError(f"list_to_batch failed for batch index {idx}") from e_list_to_batch

        batch_targets = []
        ignore_index = -1
        for level in [0, 1, 2]:
            level_targets = []
            for i, target_act_val in enumerate(instance_target_actions):
                mol = partial_molecules[i]
                current_level = getattr(mol, 'current_action_level', -1)
                if current_level == level:
                    level_targets.append(target_act_val)
                else:
                    level_targets.append(ignore_index)
            batch_targets.append(torch.tensor(level_targets, dtype=torch.long))

        return dict(
            input=batch_input,
            target_zero=batch_targets[0],
            target_one=batch_targets[1],
            target_two=batch_targets[2]
        )


# --- Test Block (Keep your existing test block, but ensure TEST_DATA_PATH can point to a pretraining file) ---
if __name__ == "__main__":
    import traceback
    from torch.utils.data import DataLoader

    print("\n--- Running ADAPTED TransformationMoleculeDataset Test ---")

    # --- Configuration ---
    # Test with a Gumbeldore-generated file (List[Dict])
    # TEST_DATA_PATH_GUMBEL = "./data/generated_molecules.pickle"

    # Test with a pretraining-generated file (Dict[str, List[Tuple]])
    # !! IMPORTANT: Update this path to your actual pretraining pickle !!
    TEST_DATA_PATH_PRETRAIN = "./data/chembl/transformation_datasets/transformations_train.pkl"  # Or _valid.pkl

    # Select which path to test:
    TEST_DATA_PATH = TEST_DATA_PATH_PRETRAIN  # <--- CHANGE THIS TO TEST DIFFERENT FORMATS

    TEST_BATCH_SIZE = 4
    NUM_BATCHES_TO_TEST = 3
    USE_RANDOM_SAMPLING_FOR_GETITEM = False  # This flag is for dataset's internal sampling for __getitem__ when no_random=False

    test_config = MoleculeConfig()
    if not hasattr(test_config, 'max_num_atoms') or test_config.max_num_atoms is None:
        print("Warning: test_config.max_num_atoms not set, using default 50.")
        test_config.max_num_atoms = 50

    print(f"\nInstantiating ADAPTED TransformationMoleculeDataset...")
    print(f"  Data path: {TEST_DATA_PATH}")
    print(f"  Batch size (for dataset's internal batching): {TEST_BATCH_SIZE}")
    print(f"  Dataset 'no_random' (sequential fetching for batches): {not USE_RANDOM_SAMPLING_FOR_GETITEM}")

    dataset = None
    try:
        dataset = TransformationMoleculeDataset(
            config=test_config,
            path_to_pickle=TEST_DATA_PATH,
            batch_size=TEST_BATCH_SIZE,
            custom_num_batches=None,
            no_random=not USE_RANDOM_SAMPLING_FOR_GETITEM  # if True, __getitem__ uses sequential target selection
        )
        print(f"Dataset instantiated successfully.")
        print(f"Total number of BATCHES (__len__): {len(dataset)}")
        print(f"Total number of individual SAMPLES available (targets_to_sample): {len(dataset.targets_to_sample)}")
        print(f"Number of PROCESSED MOLECULE ENTRIES (after format conversion): {len(dataset.processed_molecule_data)}")

    except FileNotFoundError as e:
        print(f"\nError: Dataset file not found at '{TEST_DATA_PATH}'.")
        print("Please ensure the path is correct and the file exists.")
        exit(1)
    except (ValueError, TypeError) as e:
        print(f"\nError during dataset instantiation: {e}")
        print("Check if the pickle file contains valid data in a supported format.")
        traceback.print_exc()
        exit(1)
    except Exception as e:
        print(f"\nError during dataset instantiation: {e}")
        traceback.print_exc()
        exit(1)

    if dataset and len(dataset) > 0:
        print(
            f"\nFetching and inspecting {min(NUM_BATCHES_TO_TEST, len(dataset))} batches using DataLoader(batch_size=1)...")
        # DataLoader shuffle should be False if dataset.no_random is True, to test determinism
        dataloader_shuffle = USE_RANDOM_SAMPLING_FOR_GETITEM  # Shuffle DataLoader if dataset samples randomly
        test_loader = DataLoader(dataset, batch_size=1, shuffle=dataloader_shuffle, num_workers=0)

        batches_inspected = 0
        for i, batch_data_from_loader in enumerate(test_loader):  # This is one pre-batched item from dataset
            if i >= NUM_BATCHES_TO_TEST:
                break
            batches_inspected += 1
            print(f"\n--- Batch {i + 1} (from dataset index {i}) ---")
            try:
                # batch_data_from_loader is already a batch dictionary from dataset.__getitem__,
                # but DataLoader wraps it in another list of size 1.
                # So, batch_data_from_loader['input'][0] gives the actual batched tensor.

                actual_samples_in_this_batch = batch_data_from_loader['target_zero'][0].shape[0]
                print(f"  Actual samples in this dataset-generated batch: {actual_samples_in_this_batch}")

                print(f"  Batch keys: {list(batch_data_from_loader.keys())}")
                if 'input' in batch_data_from_loader and isinstance(batch_data_from_loader['input'], dict):
                    input_dict_from_loader = batch_data_from_loader['input']
                    print(f"  Input keys: {list(input_dict_from_loader.keys())}")
                    for key, batched_tensor_from_loader in input_dict_from_loader.items():
                        if isinstance(batched_tensor_from_loader, torch.Tensor):
                            actual_tensor_from_dataset_batch = batched_tensor_from_loader[0]
                            actual_shape = actual_tensor_from_dataset_batch.shape
                            print(
                                f"    Input['{key}'].shape: {tuple(actual_shape)} (Expected first dim: {actual_samples_in_this_batch})")
                            if len(actual_shape) > 0 and actual_shape[0] != actual_samples_in_this_batch:
                                print(f"      WARNING: Dim 0 mismatch! Got {actual_shape[0]}")
                        else:
                            print(f"    Input['{key}']: type {type(batched_tensor_from_loader)}")
                else:
                    print("  Error: 'input' key missing or not a dictionary.")

                for key in ['target_zero', 'target_one', 'target_two']:
                    if key in batch_data_from_loader and isinstance(batch_data_from_loader[key], torch.Tensor):
                        actual_tensor_from_dataset_batch = batch_data_from_loader[key][0]
                        actual_shape = actual_tensor_from_dataset_batch.shape
                        print(f"  '{key}'.shape: {tuple(actual_shape)}")
                        if len(actual_shape) > 0 and actual_shape[0] != actual_samples_in_this_batch:
                            print(
                                f"    Warning: First dimension ({actual_shape[0]}) doesn't match expected samples in batch ({actual_samples_in_this_batch})")
                    else:
                        print(f"  Error: '{key}' key missing or not a Tensor.")
            except Exception as e:
                print(f"  Error processing batch {i}: {e}")
                traceback.print_exc()
        print(f"\n--- Test Complete ({batches_inspected} batches inspected) ---")
    elif dataset is not None:
        print("\nDataset instantiated but appears empty (length 0). Cannot fetch batches.")
    else:
        print("\nDataset instantiation failed. Cannot proceed with testing.")
