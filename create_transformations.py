import time
import pickle
import os
import random
import numpy as np
# import networkx as nx # No longer needed directly here
from rdkit import Chem, RDLogger
# from rdkit.Chem import rdmolfiles, rdmolops # No longer needed directly here
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional, Set, Any
from datetime import datetime
import copy
from collections import Counter # Added for probability calculation

# --- Global Debug Flag ---
# Set to True to enable max component tracking and extra logging
DEBUG_MODE = False
# --- End Global Debug Flag ---

# turn off RDKit warnings
RDLogger.DisableLog('rdApp.*')

# --- Import Custom Modules ---
try:
    from config import MoleculeConfig
    from molecule_design import MoleculeDesign, ActionType, build_reverse_atom_lookup
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print("Please ensure config.py and molecule_design.py are in the correct path.")
    exit(1)

# --- Configuration ---
try:
    CONFIG = MoleculeConfig()
    if not hasattr(CONFIG, 'min_actions'):
        print("Warning: 'min_actions' not found in MoleculeConfig, using default: 5")
        CONFIG.min_actions = 5
    if not hasattr(CONFIG, 'max_actions'):
        print("Warning: 'max_actions' not found in MoleculeConfig, using default: 50")
        CONFIG.max_actions = 50
except Exception as e:
    print(f"Error loading MoleculeConfig: {e}")
    exit(1)

# Use DEBUG_MOLECULE_LIMIT from config if exists, otherwise default
DEBUG_MOLECULE_LIMIT = 5

MAX_ATOMS = CONFIG.max_num_atoms
MIN_HIGH_LEVEL_ACTIONS = CONFIG.min_actions
MAX_HIGH_LEVEL_ACTIONS = CONFIG.max_actions
MAX_LOW_LEVEL_STEPS_SAFETY = MAX_HIGH_LEVEL_ACTIONS * 4

# TRANSFORMATIONS_PER_MOLECULE = 10
TRANSFORMATIONS_PER_MOLECULE = 2
MAX_ATTEMPTS_PER_TRANSFORMATION = 50
MAX_TOTAL_ATTEMPTS_PER_MOLECULE = TRANSFORMATIONS_PER_MOLECULE * MAX_ATTEMPTS_PER_TRANSFORMATION * 2 # Keep this calculation based on potentially updated values

RANDOM_SEED = CONFIG.seed
CHECKPOINT_DIR = "./data/chembl/checkpoints_transformations"
STATS_FREQUENCY = 10
CHECKPOINT_FREQUENCY = 100 # Use config value
RESULTS_DIR = "./data/chembl/transformation_datasets"

CHEMBL_TRAIN_PATH = "./data/chembl/chembl_train_filtered.smiles"
CHEMBL_VALID_PATH = "./data/chembl/chembl_valid_filtered.smiles"

DATATYPES = {
    "train": CHEMBL_TRAIN_PATH,
    "valid": CHEMBL_VALID_PATH
}

# Set random seed
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Ensure directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Molecule Loading and Filtering ---
def load_and_filter_molecules(path: str, max_atoms: int = MAX_ATOMS, datatype: str = "unknown") -> List[str]:
    """Loads SMILES, filters by atom count, removes duplicates, returns list of canonical SMILES."""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"filtered_molecules_{datatype}.pkl")
    if os.path.exists(checkpoint_path):
        print(f"Loading filtered {datatype} molecules from checkpoint {checkpoint_path}")
        try:
            with open(checkpoint_path, "rb") as f:
                filtered_smiles = pickle.load(f)
            # Basic validation
            if isinstance(filtered_smiles, list) and all(isinstance(s, str) for s in filtered_smiles):
                print(f"Loaded {len(filtered_smiles)} {datatype} molecules from checkpoint")
                return filtered_smiles
            else:
                print(f"Warning: Checkpoint data invalid type ({type(filtered_smiles)}). Reloading.")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Processing from scratch.")

    print(f"Loading and filtering molecules from {path}")
    filtered_smiles_list: List[str] = []
    processed_smiles: Set[str] = set()
    if not os.path.exists(path):
        print(f"Error: Input SMILES file not found at {path}")
        return []
    with open(path) as f:
        for line in tqdm(f, desc=f"Filtering {datatype} molecules"):
            smiles = line.strip()
            if not smiles: continue
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None: continue
                # Optional: Pre-sanitize here if needed, though MolFromSmiles handles some issues
                # Chem.SanitizeMol(mol)
                # Chem.Kekulize(mol)
                num_heavy = mol.GetNumHeavyAtoms() # Check before canonicalization if faster
                if num_heavy == 0 or num_heavy > max_atoms: continue

                # Get canonical SMILES *after* initial checks
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                if not canonical_smiles: continue # Skip if canonicalization fails
                if canonical_smiles in processed_smiles: continue

                # Final check on canonical SMILES (redundant if MolFromSmiles worked, but safe)
                mol_check = Chem.MolFromSmiles(canonical_smiles)
                if mol_check is None: continue
                # Re-check num heavy atoms on canonical mol (should be same, but safest)
                num_heavy_check = mol_check.GetNumHeavyAtoms()
                if num_heavy_check == 0 or num_heavy_check > max_atoms: continue

                filtered_smiles_list.append(canonical_smiles)
                processed_smiles.add(canonical_smiles)
            except Exception as e:
                 # print(f"Skipping SMILES '{smiles}' due to error: {e}") # Optional debug
                 continue # Skip malformed SMILES
    try:
        with open(checkpoint_path, "wb") as f:
            pickle.dump(filtered_smiles_list, f)
        print(f"Saved {len(filtered_smiles_list)} filtered {datatype} molecules to checkpoint {checkpoint_path}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
    return filtered_smiles_list


# --- Atom Probability Calculation ---
# --- Atom Probability Calculation (Revised to use Train+Valid) ---
def calculate_atom_probabilities_from_lists( # Renamed again
    train_smiles_list: List[str], # <-- Accept train list
    valid_smiles_list: List[str], # <-- Accept valid list
    config: MoleculeConfig,
    reverse_atom_lookup: Dict[Tuple[int, int, int], int],
    checkpoint_dir: str = CHECKPOINT_DIR
    # No max_atoms_filter needed, filtering done before call
) -> Tuple[Dict[str, float], List[str]]:
    """
    Calculates the probability distribution of atom types based on the combined
    pre-loaded training and validation SMILES lists, using the
    reverse_atom_lookup for mapping. Loads from/Saves to a checkpoint.
    Returns probabilities and the ordered list of allowed atom names.
    Handles duplicates across lists.
    """
    checkpoint_filename = "atom_probabilities_combined.pkl" # Checkpoint for the calculated probabilities
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

    # --- Get ordered list of allowed atom names from config (Unchanged) ---
    try:
        if hasattr(config, 'vocabulary_atom_names'):
             ordered_vocab_names = config.vocabulary_atom_names
        else:
             ordered_vocab_names = list(config.atom_vocabulary.keys())
        allowed_atom_names_ordered = [
            name for i, name in enumerate(ordered_vocab_names)
            if config.atom_vocabulary[name].get("allowed", False)
        ]
        allowed_atom_keys_set = set(allowed_atom_names_ordered)
        if not allowed_atom_names_ordered:
             raise ValueError("No allowed atoms found in config vocabulary.")
    except Exception as e:
        print(f"Error processing atom vocabulary from config: {e}")
        raise

    # --- Try loading from checkpoint (Unchanged) ---
    if os.path.exists(checkpoint_path):
        print(f"Loading combined atom probabilities from checkpoint: {checkpoint_path}")
        try:
            # ... (keep existing checkpoint loading logic for probabilities) ...
            with open(checkpoint_path, "rb") as f: probabilities = pickle.load(f)
            if isinstance(probabilities, dict) and all(isinstance(k, str) and isinstance(v, float) for k, v in probabilities.items()):
                 if set(probabilities.keys()) == allowed_atom_keys_set:
                     print("Atom probabilities loaded successfully.")
                     return probabilities, allowed_atom_names_ordered
                 else: print("Warning: Allowed vocab keys in checkpoint don't match current config. Recalculating.")
            else: print("Warning: Checkpoint data has incorrect format. Recalculating.")
        except Exception as e: print(f"Warning: Failed to load probabilities checkpoint: {e}. Recalculating.")


    # --- Calculation needed ---
    print(f"Calculating combined atom probabilities from pre-loaded lists (Train: {len(train_smiles_list)}, Valid: {len(valid_smiles_list)})...")
    atom_counts = Counter()
    total_atoms = 0
    processed_smiles_for_counts = set() # Track SMILES processed *during this calculation*

    # --- Combine unique SMILES from both lists ---
    all_smiles_to_process = []
    for smiles in train_smiles_list + valid_smiles_list:
         if smiles not in processed_smiles_for_counts:
              all_smiles_to_process.append(smiles)
              processed_smiles_for_counts.add(smiles)

    if not all_smiles_to_process:
        print("Error: No unique molecules found in combined lists. Cannot calculate probabilities.")
        # Fallback to uniform
        num_allowed = len(allowed_atom_names_ordered)
        if num_allowed == 0: return {}, allowed_atom_names_ordered
        uniform_prob = 1.0 / num_allowed
        probabilities = {key: uniform_prob for key in allowed_atom_names_ordered}
        print("Warning: Using uniform atom probabilities as fallback.")
        # Save this fallback state
        try:
            print(f"Saving fallback uniform probabilities to checkpoint: {checkpoint_path}")
            with open(checkpoint_path, "wb") as f: pickle.dump(probabilities, f)
        except Exception as e: print(f"Error saving fallback probabilities checkpoint: {e}")
        return probabilities, allowed_atom_names_ordered

    # --- Iterate through the combined, unique list for counting ---
    print(f"Counting atoms in {len(all_smiles_to_process)} unique molecules...")
    for smiles in tqdm(all_smiles_to_process, desc="Counting atoms in combined list"):
         # ... (keep existing atom counting logic using MolFromSmiles and reverse_lookup) ...
         try:
             mol = Chem.MolFromSmiles(smiles)
             if mol is None: continue
             for atom in mol.GetAtoms():
                 atomic_num = atom.GetAtomicNum()
                 charge = atom.GetFormalCharge()
                 rdkit_chiral = atom.GetChiralTag()
                 chiral_key_val = 0
                 if rdkit_chiral == Chem.ChiralType.CHI_TETRAHEDRAL_CW: chiral_key_val = 1
                 elif rdkit_chiral == Chem.ChiralType.CHI_TETRAHEDRAL_CCW: chiral_key_val = 2
                 key = (atomic_num, charge, chiral_key_val)
                 vocab_idx = reverse_atom_lookup.get(key)
                 if vocab_idx is None and chiral_key_val != 0:
                      key_no_chiral = (atomic_num, charge, 0)
                      vocab_idx = reverse_atom_lookup.get(key_no_chiral)
                 if vocab_idx is not None:
                     if 1 <= vocab_idx <= len(ordered_vocab_names):
                          atom_name = ordered_vocab_names[vocab_idx - 1]
                          if atom_name in allowed_atom_keys_set:
                              atom_counts[atom_name] += 1
                              total_atoms += 1
         except Exception as e:
             continue # Skip molecule on error


    # ... (keep existing probability calculation, zero handling, and saving logic) ...
    if total_atoms == 0:
        print("Error: No atoms counted from the combined dataset. Cannot calculate probabilities.")
        num_allowed = len(allowed_atom_names_ordered)
        if num_allowed == 0: return {}, allowed_atom_names_ordered
        uniform_prob = 1.0 / num_allowed
        probabilities = {key: uniform_prob for key in allowed_atom_names_ordered}
        print("Warning: Using uniform atom probabilities as fallback.")
    else:
        probabilities = {key: count / total_atoms for key, count in atom_counts.items()}
        for key in allowed_atom_names_ordered:
            if key not in probabilities:
                probabilities[key] = 0.0
                print(f"Note: Allowed atom type '{key}' not found in combined dataset, assigned probability 0.")

    print(f"Combined atom probabilities calculated. Total unique atoms counted: {total_atoms}")
    if DEBUG_MODE:
        print("Calculated Probabilities (Combined):")
        for key in allowed_atom_names_ordered:
             prob = probabilities.get(key, 0.0)
             print(f"  '{key}': {prob:.6f}")

    try:
        print(f"Saving combined atom probabilities to checkpoint: {checkpoint_path}")
        with open(checkpoint_path, "wb") as f:
            pickle.dump(probabilities, f)
        print("Checkpoint saved.")
    except Exception as e:
        print(f"Error saving probabilities checkpoint: {e}")

    return probabilities, allowed_atom_names_ordered


# --- Random Walk Generation Logic ---
def get_action_type_name(level: int, action: int, vocab_size: int, num_real_atoms: int) -> str:
    # (Implementation remains the same)
    if level == 0: return "Terminate" if action == 0 else "Select Atom"
    elif level == 1:
        remove_action_idx = vocab_size + num_real_atoms
        if 0 <= action < vocab_size: return "Add Atom"
        elif vocab_size <= action < remove_action_idx: return "Select Existing Atom"
        elif action == remove_action_idx: return "Remove Atom"
        else: return "Unknown L1"
    elif level == 2: return "Set Bond Order" if 0 <= action <= 5 else "Remove Bond" if action == 6 else "Unknown L2"
    else: return "Unknown Level"


# <<< Modified Action Selection Function >>>
def select_action_strategy(
    mol_design: MoleculeDesign,
    atom_probabilities: Dict[str, float], # Passed in for weighted Add Atom
    atom_vocab_list: List[str],          # Passed in for weighted Add Atom
    terminate_prob: float = 0.05
) -> Optional[int]:
    """
    Selects a random valid action based on the current level.
    Level 0: Biased sampling (terminate_prob for Terminate, 1-terminate_prob for Select Atom).
    Level 1 (Add Atom): Weighted sampling based on atom_probabilities.
    Level 1 (Other Cats) / Level 2: Equal probability sampling across valid action *categories*,
                                     and uniform sampling *within* those categories.
    """
    mask = mol_design.current_action_mask
    level = mol_design.current_action_level
    num_real_atoms = len(mol_design.atoms) - 1
    vocab_size = mol_design.vocab_size

    if mask is None:
        if DEBUG_MODE: print("DEBUG (Action Select): Mask is None.")
        return None

    valid_action_indices = [i for i, is_masked in enumerate(mask) if not is_masked]
    if not valid_action_indices:
        if DEBUG_MODE: print(f"DEBUG (Action Select): No valid actions at level {level}.")
        return None

    # --- Level 0: Biased Sampling (Unchanged) ---
    if level == 0:
        # ... (Keep existing Level 0 logic - selecting Terminate vs Select Atom) ...
        valid_terminate_actions = [i for i in valid_action_indices if i == 0]
        valid_select_atom_actions = [i for i in valid_action_indices if 1 <= i <= num_real_atoms]
        can_terminate = bool(valid_terminate_actions)
        can_select_atom = bool(valid_select_atom_actions)
        if not can_terminate and not can_select_atom:
             if DEBUG_MODE: print("DEBUG (Action Select L0): No valid Terminate or Select Atom actions found despite valid indices.")
             return None
        elif can_terminate and not can_select_atom: return valid_terminate_actions[0]
        elif not can_terminate and can_select_atom: return random.choice(valid_select_atom_actions)
        else:
            if random.random() < terminate_prob:
                if DEBUG_MODE: print(f"DEBUG (Action Select L0): Chose Terminate (Prob: {terminate_prob})")
                return valid_terminate_actions[0]
            else:
                selected_atom_action = random.choice(valid_select_atom_actions)
                if DEBUG_MODE: print(f"DEBUG (Action Select L0): Chose Select Atom {selected_atom_action} (Prob: {1.0-terminate_prob})")
                return selected_atom_action

    # --- Level 1 & 2: Modified Category Sampling ---
    else:
        action_groups: Dict[str, List[int]] = {}
        if level == 1:
            # Define end indices based on current state (num_real_atoms is before action)
            add_atom_end_idx = vocab_size
            select_existing_end_idx = vocab_size + num_real_atoms
            replace_atom_end_idx = select_existing_end_idx + vocab_size
            remove_atom_idx = replace_atom_end_idx

            # Group valid actions by category using the NEW index ranges
            action_groups["Add Atom"] = [i for i in valid_action_indices if 0 <= i < add_atom_end_idx]
            action_groups["Select Existing Atom"] = [i for i in valid_action_indices if add_atom_end_idx <= i < select_existing_end_idx]
            action_groups["Replace Atom"] = [i for i in valid_action_indices if select_existing_end_idx <= i < replace_atom_end_idx]
            action_groups["Remove Atom"] = [i for i in valid_action_indices if i == remove_atom_idx]

        elif level == 2:
            # Level 2 grouping remains the same
            action_groups["Set Bond Order"] = [i for i in valid_action_indices if 0 <= i <= 5]
            action_groups["Remove Bond"] = [i for i in valid_action_indices if i == 6]
        else:
            raise ValueError(f"Invalid action level for sampling: {level}")

        # Filter for categories that have at least one valid action
        valid_categories = {name: indices for name, indices in action_groups.items() if indices}
        if not valid_categories:
             if DEBUG_MODE: print(f"DEBUG (Action Select L{level}): No valid action categories found.")
             return None

        # --- Choose a category with equal probability ---
        selected_category_name = random.choice(list(valid_categories.keys()))

        # --- Choose an action WITHIN the selected category ---
        if selected_category_name == "Add Atom":
            # Use WEIGHTED sampling based on dataset probabilities
            valid_add_atom_indices = valid_categories["Add Atom"]
            if not valid_add_atom_indices:
                 if DEBUG_MODE: print("DEBUG ERROR: 'Add Atom' category chosen but no valid indices found.")
                 return None # Should not happen

            candidate_indices = []
            candidate_weights = []
            for index in valid_add_atom_indices:
                if 0 <= index < len(atom_vocab_list): # atom_vocab_list has allowed names
                    atom_type_str = atom_vocab_list[index]
                    prob = atom_probabilities.get(atom_type_str, 0.0)
                    candidate_indices.append(index)
                    candidate_weights.append(prob)
                else:
                     if DEBUG_MODE: print(f"DEBUG WARNING: Invalid 'Add Atom' index {index} encountered (allowed list size {len(atom_vocab_list)}).")

            if not candidate_indices or sum(candidate_weights) <= 0:
                 if DEBUG_MODE: print(f"DEBUG WARNING: No valid 'Add Atom' actions with positive probability found. Falling back to uniform choice among valid add actions.")
                 if not valid_add_atom_indices: return None
                 selected_action_index = random.choice(valid_add_atom_indices) # Fallback to uniform
            else:
                 selected_action_index = random.choices(candidate_indices, weights=candidate_weights, k=1)[0] # Weighted choice
                 if DEBUG_MODE:
                     selected_atom_type = atom_vocab_list[selected_action_index]
                     selected_prob = atom_probabilities.get(selected_atom_type, 0.0)
                     print(f"DEBUG (Action Select L1 Add): Weighted Sample Chose Action {selected_action_index} ('{selected_atom_type}', Prob={selected_prob:.4f})")

            return selected_action_index

        else:
            # For "Select Existing", "Replace Atom", "Remove Atom" (L1)
            # AND "Set Bond Order", "Remove Bond" (L2)
            # Choose an action UNIFORMLY from the valid actions within the chosen category
            selected_action_index = random.choice(valid_categories[selected_category_name])
            if DEBUG_MODE: print(f"DEBUG (Action Select L{level}): Chose Category '{selected_category_name}' (Uniform), Action {selected_action_index}")
            return selected_action_index


# --- generate_single_transformation ---
def generate_single_transformation(
    initial_mol_design: MoleculeDesign,
    start_smiles: str,
    config: MoleculeConfig,
    atom_probabilities: Dict[str, float], # <-- Added
    atom_vocab_list: List[str],          # <-- Added (List of allowed atom names)
    max_low_level_steps_safety_limit: int
) -> Optional[Tuple[str, str, List[int], int, int]]: # Added int for max_components
    """
    Attempts sequence generation, managing high-level action counts locally.
    Enforces min actions by preventing early termination. Uses _get_smiles_for_check.
    Tracks max components if DEBUG_MODE is True.
    Uses weighted sampling for 'Add Atom' actions.
    Returns: Tuple of (start_smiles, end_smiles, low_level_action_sequence, high_level_action_count, max_components) or None.
    """
    current_mol_design = copy.deepcopy(initial_mol_design)
    high_level_action_count = 0
    max_components_this_sequence = 1

    if DEBUG_MODE and current_mol_design.synthesis_done:
         print(f"DEBUG ERROR: Copied design is already finalized!")
         return None
    if start_smiles is None:
        if DEBUG_MODE: print(f"DEBUG FAIL (Start): Passed-in start_smiles is None.")
        return None

    visited_smiles: Set[str] = {start_smiles}
    action_sequence: List[int] = []
    terminated_early = False
    low_level_step_count = 0

    if DEBUG_MODE: print(f"\nDEBUG START: Molecule {start_smiles} (Min/Max High-Level Actions: {config.min_actions}/{config.max_actions})")

    while high_level_action_count < config.max_actions and low_level_step_count < max_low_level_steps_safety_limit:
        if current_mol_design.synthesis_done:
            terminated_early = True
            if DEBUG_MODE: print(f"DEBUG INFO: Terminated early flag set at low-level step {low_level_step_count} (High-level count: {high_level_action_count}).")
            break

        prev_level = current_mol_design.current_action_level
        prev_num_real = len(current_mol_design.atoms) - 1

        # <<< Use the selection strategy with probabilities >>>
        action = select_action_strategy(
            current_mol_design,
            atom_probabilities, # Pass probabilities
            atom_vocab_list     # Pass ordered allowed vocab list
        )
        # <<< End change >>>

        if action is None:
            # select_action_strategy now prints debug info if needed
            return None

        # --- Enforce Min High-Level Actions (Important: This overrides the 5% terminate probability if needed) ---
        if prev_level == 0 and action == 0:
            if high_level_action_count < config.min_actions:
                if DEBUG_MODE: print(f"DEBUG INFO: Min Action Override: Prevented Terminate at low-level step {low_level_step_count} (High-level count {high_level_action_count} < {config.min_actions}). Re-selecting.")
                # Force selection of a 'Select Atom' action if possible
                mask = current_mol_design.current_action_mask
                # Ensure mask is valid before accessing length
                if mask is None:
                    if DEBUG_MODE: print(f"DEBUG FAIL (Min Actions Override): Mask is None, cannot re-select.")
                    return None
                select_atom_actions = [i for i in range(1, len(mask)) if i <= prev_num_real and not mask[i]]
                if not select_atom_actions:
                    if DEBUG_MODE: print(f"DEBUG FAIL (Min Actions Override): Terminate prevented, but no valid 'Select Atom' actions found.")
                    return None
                action = random.choice(select_atom_actions) # Override action
                if DEBUG_MODE: print(f"DEBUG INFO: Min Action Override: Re-selected action: {action} (Select Atom)")
        # --- End Min Action Enforcement ---


        action_type_name = get_action_type_name(prev_level, action, current_mol_design.vocab_size, prev_num_real)
        modifies_structure = action_type_name in ["Remove Atom", "Set Bond Order", "Remove Bond"]
        if DEBUG_MODE: print(f"DEBUG STEP {low_level_step_count}: HighLvl={high_level_action_count}, LowLvl={prev_level}, Action={action} ({action_type_name}), ModStruct={modifies_structure}")

        try:
            current_mol_design.take_action(action)
            # next_mol_design = current_mol_design # No longer needed, take_action modifies in place
            current_level = current_mol_design.current_action_level # Get updated level

            if DEBUG_MODE:
                max_components_this_sequence = max(max_components_this_sequence, current_mol_design.num_components)
                # Reduce verbosity: Only print if components > 1
                # if current_mol_design.num_components > 1:
                #      print(f"DEBUG INFO: Step {low_level_step_count} - Components = {current_mol_design.num_components} (Max so far: {max_components_this_sequence})")

            if prev_level != 0 and current_level == 0:
                high_level_action_count += 1
                if DEBUG_MODE: print(f"DEBUG INFO: High-level action count incremented to {high_level_action_count} at low-level step {low_level_step_count}.")

        # Catch specific errors that indicate sequence failure
        except (ValueError, IndexError, RuntimeError) as e:
             if DEBUG_MODE: print(f"DEBUG FAIL (Take Action Error): Error during take_action(action={action}, L{prev_level}) at low-level step {low_level_step_count}: {e}")
             # traceback.print_exc() # Optional: Full traceback in debug
             return None
        # Do not catch generic Exception here, let it propagate if truly unexpected

        next_smiles = None
        perform_cycle_check = modifies_structure
        if perform_cycle_check:
            # if DEBUG_MODE: print(f"DEBUG INFO: Performing cycle check for low-level step {low_level_step_count} (structure modified).")
            cycle_detected = False
            # Use the internal check function which handles sanitization issues
            next_smiles = current_mol_design._get_smiles_for_check()

            if next_smiles is None:
                if DEBUG_MODE: print(f"DEBUG FAIL (Cycle Check SMILES Error/None): SMILES None or generation failed after low-level step {low_level_step_count}.")
                # This indicates an invalid intermediate state, fail the sequence
                return None

            if next_smiles != "" and next_smiles in visited_smiles:
                cycle_detected = True
                if DEBUG_MODE: print(f"DEBUG FAIL (Cycle): Modifying action led back to visited SMILES after low-level step {low_level_step_count}, SMILES: {next_smiles}")

            if cycle_detected: return None
        # else:
            # if DEBUG_MODE: print(f"DEBUG INFO: Skipping cycle check for low-level step {low_level_step_count} (structure not modified).")

        action_sequence.append(action)
        # Add to visited set only if cycle check was performed and passed
        if perform_cycle_check and next_smiles is not None and next_smiles != "" and next_smiles not in visited_smiles:
             visited_smiles.add(next_smiles)
             # if DEBUG_MODE: print(f"DEBUG INFO: Added SMILES to visited set: {next_smiles}")

        low_level_step_count += 1
    # --- End of While Loop ---

    # Check if loop ended due to safety break
    if not terminated_early and low_level_step_count >= max_low_level_steps_safety_limit:
        if DEBUG_MODE: print(f"DEBUG FAIL (Safety Break): Exceeded max low-level steps ({max_low_level_steps_safety_limit}).")
        return None

    # Check if loop ended due to max high-level actions
    if not terminated_early and high_level_action_count >= config.max_actions:
        if DEBUG_MODE: print(f"DEBUG INFO: Reached max high-level actions ({config.max_actions}). Checking if terminable.")
        # Attempt force termination if possible
        if current_mol_design.current_action_level == 0 and current_mol_design.is_terminable():
            try:
                terminate_action = 0
                mask = current_mol_design.current_action_mask
                # Check mask validity and if action 0 is allowed
                if mask is not None and len(mask) > terminate_action and not mask[terminate_action]:
                    current_mol_design.take_action(terminate_action) # This sets synthesis_done=True
                    action_sequence.append(terminate_action)
                    terminated_early = True # Mark as terminated
                    if DEBUG_MODE: print(f"DEBUG INFO: Force terminated at max high-level actions.")
                else:
                    if DEBUG_MODE: print(f"DEBUG FAIL (Force Terminate @ Max): Cannot force terminate. State L0 but Terminate masked or unavailable.")
                    return None # Failed to terminate
            except Exception as e:
                # Catch errors during the final take_action call
                if DEBUG_MODE: print(f"DEBUG FAIL (Force Terminate Error @ Max): Error during force termination: {e}")
                return None # Failed to terminate
        else:
            # Reached max actions but cannot terminate in current state
            if DEBUG_MODE: print(f"DEBUG FAIL (End State @ Max): Reached max high-level actions but cannot terminate. Level={current_mol_design.current_action_level}, Terminable={current_mol_design.is_terminable()}")
            return None # Failed sequence

    # --- Final Strict Validation ---

    # 1. Check if sequence actually terminated properly
    #    'terminated_early' flag is set if Terminate action was taken (naturally or forced)
    #    'synthesis_done' should be True if Terminate was taken
    if not current_mol_design.synthesis_done:
         if DEBUG_MODE: print(f"DEBUG FAIL (Final Flags): Sequence finished but synthesis_done is False.")
         return None

    # 2. Check for infeasibility flag set during take_action
    if current_mol_design.infeasibility_flag:
        if DEBUG_MODE: print(f"DEBUG FAIL (Final Flags): Infeasibility flag is True.")
        return None

    # 3. Check connectivity (only relevant if >1 atom)
    num_final_real_atoms = len(current_mol_design.atoms) - 1
    if num_final_real_atoms > 1 and not current_mol_design.is_currently_connected:
         if DEBUG_MODE: print(f"DEBUG FAIL (Final Connectivity): Molecule disconnected ({current_mol_design.num_components} components).")
         return None

    # 4. Get final SMILES (includes sanitization check via finalize/to_smiles)
    final_smiles = None
    try:
        # to_smiles calls finalize if needed, which caches results and handles sanitization
        final_smiles = current_mol_design.to_smiles(canonical=True)

        # Check if SMILES generation failed (finalize sets _cached_smiles to None on error)
        if final_smiles is None:
             # Allow empty SMILES only if 0 real atoms
             if num_final_real_atoms == 0:
                 final_smiles = ""
                 if DEBUG_MODE: print(f"DEBUG INFO: Final molecule has 0 atoms, SMILES is ''.")
             else:
                  if DEBUG_MODE: print(f"DEBUG FAIL (Final SMILES None/Sanitize): Final SMILES is None despite {num_final_real_atoms} atoms (likely sanitization failure within finalize/to_smiles).")
                  return None
        # else:
        #     if DEBUG_MODE: print(f"DEBUG INFO: Final molecule passed Sanitization (via to_smiles/finalize). SMILES: {final_smiles}")

    except Exception as e:
        # Catch unexpected errors during final SMILES generation/sanitization
        if DEBUG_MODE: print(f"DEBUG FAIL (Final RDKit/Sanitize Error during to_smiles/finalize): {e}")
        return None

    # 5. Check if final SMILES is same as start
    if final_smiles == start_smiles:
        if DEBUG_MODE: print(f"DEBUG FAIL (Start=End): Final SMILES matches start. Sequence: {action_sequence}")
        return None

    # 6. Check minimum high-level actions
    #    Need to recalculate high_level_action_count based on sequence if needed,
    #    or ensure the loop counter is accurate. Let's assume loop counter is correct for now.
    if high_level_action_count < config.min_actions:
         if DEBUG_MODE: print(f"DEBUG FAIL (Min High-Level Actions): Final count {high_level_action_count} < {config.min_actions}. (Should have been prevented by override logic)")
         # This might indicate a bug in the override or counting logic
         return None

    # --- Success ---
    if DEBUG_MODE:
        print(f"\nDEBUG SUCCESS (Strict): Sequence passed all checks!")
        print(f"  - Start SMILES: {start_smiles}")
        print(f"  - End SMILES:   {final_smiles}")
        print(f"  - Low-level Len: {len(action_sequence)}")
        print(f"  - High-level Count: {high_level_action_count}")
        print(f"  - Max Components: {max_components_this_sequence}")
        # print(f"  - Sequence: {action_sequence}") # Optional: print full sequence
        print("-" * 20)

    return start_smiles, final_smiles, action_sequence, high_level_action_count, max_components_this_sequence


# --- Main Function ---
def main():
    start_time = time.time()
    print(f"Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Current User's Login: {os.getenv('USER', 'unknown')}")
    print("Starting transformation dataset generation (with weighted atom sampling).") # Updated message
    print(f"Using random seed: {RANDOM_SEED}")
    print(f"Min HIGH-LEVEL actions per sequence: {MIN_HIGH_LEVEL_ACTIONS}")
    print(f"Max HIGH-LEVEL actions per sequence: {MAX_HIGH_LEVEL_ACTIONS}")
    print(f"Safety limit (max LOW-LEVEL steps): {MAX_LOW_LEVEL_STEPS_SAFETY}")
    print(f"Target transformations per molecule: {TRANSFORMATIONS_PER_MOLECULE}")
    print(f"Stats print frequency: {STATS_FREQUENCY} molecules")
    print(f"Checkpoint frequency: {CHECKPOINT_FREQUENCY} molecules")
    if DEBUG_MODE:
        print(f"*** DEBUG MODE ACTIVE: Detailed logging, Max Component Tracking, processing limit = {DEBUG_MOLECULE_LIMIT} molecules ***")
    else:
        print("*** PRODUCTION MODE ACTIVE ***")

    # <<< --- Build Reverse Lookup FIRST --- >>>
    try:
        reverse_atom_lookup = build_reverse_atom_lookup(CONFIG)
        print("Successfully built reverse atom lookup.")
    except Exception as e:
        print(f"Fatal Error: Could not build reverse atom lookup from config: {e}")
        return
    # <<< --- End Build Reverse Lookup --- >>>

    # <<< --- Load Train and Valid Data ONCE --- >>>
    print("\n--- Pre-loading Train and Validation Data ---")
    train_smiles_list = []
    valid_smiles_list = []
    try:
        print("Loading training data...")
        train_smiles_list = load_and_filter_molecules(CHEMBL_TRAIN_PATH, MAX_ATOMS, datatype="train")
        if not train_smiles_list:
             print("Warning: No training molecules loaded or filtered.")
             # Decide if this is fatal or just affects probability calc
        else:
             print(f"Pre-loaded {len(train_smiles_list)} training molecules.")

        print("Loading validation data...")
        valid_smiles_list = load_and_filter_molecules(CHEMBL_VALID_PATH, MAX_ATOMS, datatype="valid")
        if not valid_smiles_list:
             print("Warning: No validation molecules loaded or filtered.")
        else:
            print(f"Pre-loaded {len(valid_smiles_list)} validation molecules.")

        if not train_smiles_list and not valid_smiles_list:
             print("Fatal Error: No molecules loaded from train or valid paths. Cannot proceed.")
             return

    except Exception as e:
         print(f"Fatal Error loading train/validation data: {e}")
         return
    # <<< --- End Loading Data --- >>>


    # <<< --- Calculate or Load Atom Probabilities (using pre-loaded lists) --- >>>
    print("\n--- Calculating Atom Probabilities (Train+Valid) ---")
    try:
        # Calculate probabilities using the pre-loaded lists
        atom_probabilities, allowed_atom_vocab_list = calculate_atom_probabilities_from_lists(
            train_smiles_list=train_smiles_list, # Pass the loaded train list
            valid_smiles_list=valid_smiles_list, # Pass the loaded valid list
            config=CONFIG,
            reverse_atom_lookup=reverse_atom_lookup
        )
        if not atom_probabilities:
             print("Fatal Error: Atom probabilities could not be calculated or loaded (returned empty).")
             return
        if not allowed_atom_vocab_list and atom_probabilities:
             print("Fatal Error: Atom probabilities generated, but allowed atom list is empty.")
             return
        print(f"Using {len(allowed_atom_vocab_list)} allowed atom types for weighted sampling.")
    except Exception as e:
        print(f"Fatal Error during atom probability calculation/loading: {e}")
        return
    # <<< --- End Probability Calculation --- >>>

    global_total_sequences_attempted = 0
    global_total_valid_sequences_generated = 0
    global_total_molecules_processed = 0

    # Store pre-loaded lists in a dictionary for easy access in the loop
    preloaded_data = {
        "train": train_smiles_list,
        "valid": valid_smiles_list
    }

    for datatype, filepath in DATATYPES.items():
        datatype_start_time = time.time()
        print(f"\n--- Processing {datatype} data ---") # Removed 'from filepath' as we use preloaded

        # <<< --- Use the pre-loaded list for the current datatype --- >>>
        initial_smiles_list = preloaded_data.get(datatype)
        if initial_smiles_list is None:
            # This case should ideally not happen if DATATYPES matches preloaded_data keys
            print(f"Error: Pre-loaded data not found for datatype '{datatype}'. Skipping.")
            continue
        print(f"Using pre-loaded {datatype} data ({len(initial_smiles_list)} molecules).")
        # <<< --- End using pre-loaded list --- >>>

        total_source_molecules_in_datatype_full = len(initial_smiles_list)
        if total_source_molecules_in_datatype_full == 0:
            print(f"No valid starting molecules found in pre-loaded data for {datatype}. Skipping.")
            continue

        # Apply DEBUG limit if active
        if DEBUG_MODE:
            effective_molecule_list = initial_smiles_list[:DEBUG_MOLECULE_LIMIT]
            total_source_molecules_in_datatype_effective = len(effective_molecule_list)
            print(f"DEBUG: Processing first {total_source_molecules_in_datatype_effective} molecules.")
        else:
            effective_molecule_list = initial_smiles_list
            total_source_molecules_in_datatype_effective = total_source_molecules_in_datatype_full

        results_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"transformation_results_{datatype}.pkl")
        all_results: Dict[str, List[Tuple[str, str, List[int]]]] = {}
        start_index = 0
        datatype_sequences_attempted_session = 0
        datatype_valid_sequences_generated_session = 0
        datatype_molecules_processed_session = 0
        datatype_total_high_level_actions_session = 0
        initial_valid_sequences_count = 0
        datatype_total_max_components_session = 0 # For DEBUG mode

        # Load checkpoint if exists
        if os.path.exists(results_checkpoint_path):
            print(f"Loading existing results checkpoint: {results_checkpoint_path}")
            try:
                with open(results_checkpoint_path, "rb") as f:
                    all_results = pickle.load(f)
                processed_smiles_count_ckpt = len(all_results)
                initial_valid_sequences_count = sum(len(v) for v in all_results.values())
                start_index = 0
                processed_smiles_set_ckpt = set(all_results.keys())
                # Find the correct starting index based on the checkpoint
                for idx, smi in enumerate(effective_molecule_list):
                     # Stop if we find a molecule not in the checkpoint keys
                     if smi not in processed_smiles_set_ckpt:
                          start_index = idx
                          break
                     # If we reach the end and all are in the checkpoint, start_index is the total count
                     if idx == len(effective_molecule_list) - 1:
                          start_index = len(effective_molecule_list)

                print(f"Resuming {datatype} generation from index {start_index}/{total_source_molecules_in_datatype_effective}.")
                print(f"Checkpoint contains {processed_smiles_count_ckpt} processed molecules and {initial_valid_sequences_count} valid sequences.")
            except Exception as e:
                print(f"Warning: Failed to load results checkpoint: {e}. Starting {datatype} from scratch.")
                all_results = {}
                start_index = 0
                initial_valid_sequences_count = 0
        else:
            print(f"No results checkpoint found for {datatype}. Starting from scratch.")
            initial_valid_sequences_count = 0

        molecules_processed_since_checkpoint = 0
        molecules_processed_since_stats = 0

        tqdm_desc = f"Generating {datatype}" + (" (DEBUG)" if DEBUG_MODE else "")
        # Ensure the loop range is correct after loading checkpoint
        pbar = tqdm(range(start_index, total_source_molecules_in_datatype_effective), desc=tqdm_desc, initial=start_index, total=total_source_molecules_in_datatype_effective)

        for i in pbar:
            smiles_for_mol_design = effective_molecule_list[i]
            # Skip if we already have enough transformations for this molecule from checkpoint
            # Check >= because we might have loaded exactly the required number
            if smiles_for_mol_design in all_results and len(all_results.get(smiles_for_mol_design, [])) >= TRANSFORMATIONS_PER_MOLECULE:
                continue

            # Get existing transformations for this molecule if resuming
            valid_transformations: List[Tuple[str, str, List[int]]] = all_results.get(smiles_for_mol_design, [])
            total_attempts_for_molecule = 0
            sequences_found_this_mol_session = 0 # Tracks success for *this molecule* in *this run*

            # --- Initialize MoleculeDesign for the current SMILES ---
            try:
                # Use from_smiles which handles initialization and potential errors
                initial_mol_design, _ = MoleculeDesign.from_smiles(CONFIG, smiles_for_mol_design)
                # Double check it's not finalized (shouldn't be from from_smiles)
                if initial_mol_design.synthesis_done:
                    raise RuntimeError("MoleculeDesign.from_smiles returned a finalized instance!")
                # Get canonical SMILES from the *initialized* design for consistency checks
                actual_start_smiles = initial_mol_design._get_smiles_for_check()
                if actual_start_smiles is None:
                    # This indicates an issue even creating the initial state representation
                    raise ValueError("Initial molecule failed internal SMILES check after MoleculeDesign init.")
                # Ensure the SMILES used matches the one from the list if canonicalization changed it
                if actual_start_smiles != smiles_for_mol_design:
                     if DEBUG_MODE: print(f"DEBUG INFO: Canonical SMILES '{actual_start_smiles}' differs from input list '{smiles_for_mol_design}'. Using canonical.")
                     smiles_for_mol_design = actual_start_smiles # Use the canonical version going forward

            except (ValueError, RuntimeError, KeyError, IndexError) as e:
                if DEBUG_MODE: print(f"\nDEBUG FAIL (Init/SMILES): Failed initial MoleculeDesign for {smiles_for_mol_design}: {e}.")
                else: print(f"\nWarning: Skipping molecule {smiles_for_mol_design} due to initialization error: {e}")
                # Ensure an entry exists even if skipped, to prevent reprocessing
                if smiles_for_mol_design not in all_results: all_results[smiles_for_mol_design] = []
                datatype_molecules_processed_session += 1; molecules_processed_since_checkpoint += 1; molecules_processed_since_stats += 1; global_total_molecules_processed += 1
                continue # Skip to the next molecule

            # Check for empty SMILES after init (should only happen if input was truly empty)
            if not actual_start_smiles and initial_mol_design.GetNumAtoms() > 0:
                 if DEBUG_MODE: print(f"\nDEBUG FAIL (Init/SMILES Empty): Initial molecule check yielded empty SMILES despite atoms for {smiles_for_mol_design}.")
                 if smiles_for_mol_design not in all_results: all_results[smiles_for_mol_design] = []
                 datatype_molecules_processed_session += 1; molecules_processed_since_checkpoint += 1; molecules_processed_since_stats += 1; global_total_molecules_processed += 1
                 continue

            # --- Generate Transformations for the Molecule ---
            # Loop until enough transformations found OR max attempts reached
            while len(valid_transformations) < TRANSFORMATIONS_PER_MOLECULE and total_attempts_for_molecule < MAX_TOTAL_ATTEMPTS_PER_MOLECULE:
                total_attempts_for_molecule += 1
                datatype_sequences_attempted_session += 1
                global_total_sequences_attempted += 1

                # Create a fresh copy for each attempt
                initial_mol_design_copy = copy.deepcopy(initial_mol_design)

                # Call the generation function
                result = generate_single_transformation(
                    initial_mol_design_copy,
                    actual_start_smiles, # Use the verified canonical start SMILES
                    CONFIG,
                    atom_probabilities,        # Pass calculated probabilities
                    allowed_atom_vocab_list,   # Pass allowed vocab list
                    MAX_LOW_LEVEL_STEPS_SAFETY
                )

                # Process successful result
                if result is not None:
                    start_smi, end_smi, low_level_seq, high_level_count, max_components = result
                    # Optional: Extra check for start SMILES consistency
                    if start_smi != actual_start_smiles:
                         if DEBUG_MODE: print(f"DEBUG WARNING: Start SMILES mismatch! Expected '{actual_start_smiles}', got '{start_smi}'.")
                         # Continue using the result's start_smi for the tuple stored
                    valid_transformations.append((start_smi, end_smi, low_level_seq))
                    sequences_found_this_mol_session += 1
                    datatype_valid_sequences_generated_session += 1
                    global_total_valid_sequences_generated += 1
                    datatype_total_high_level_actions_session += high_level_count
                    if DEBUG_MODE:
                        datatype_total_max_components_session += max_components

            # --- Finished attempts for this molecule ---
            # Update counters regardless of success/failure for this molecule
            datatype_molecules_processed_session += 1
            molecules_processed_since_checkpoint += 1
            molecules_processed_since_stats += 1
            global_total_molecules_processed += 1

            # Store the list of transformations found (might be empty if none succeeded)
            all_results[smiles_for_mol_design] = valid_transformations

            # --- Stats Update ---
            # Check if it's time to print stats
            if molecules_processed_since_stats >= STATS_FREQUENCY or i == total_source_molecules_in_datatype_effective - 1:
                current_time = time.time()
                elapsed_since_start = current_time - start_time
                elapsed_since_datatype_start = current_time - datatype_start_time
                # Calculate overall valid sequences including those loaded from checkpoint
                total_valid_sequences_overall = initial_valid_sequences_count + datatype_valid_sequences_generated_session
                # Calculate overall processed molecules including those skipped due to checkpoint
                processed_mol_count_overall = start_index + datatype_molecules_processed_session
                avg_sequences_per_mol_overall = total_valid_sequences_overall / processed_mol_count_overall if processed_mol_count_overall > 0 else 0
                # Session success rate based on attempts *in this run*
                success_rate_session = (datatype_valid_sequences_generated_session / datatype_sequences_attempted_session * 100) if datatype_sequences_attempted_session > 0 else 0
                # Session average actions based on valid sequences found *in this run*
                avg_high_level_actions_session = (datatype_total_high_level_actions_session / datatype_valid_sequences_generated_session) if datatype_valid_sequences_generated_session > 0 else 0.0

                stats_header = f"--- Stats Update ({datatype} @ molecule {i+1}/{total_source_molecules_in_datatype_effective})" + (" (DEBUG MODE)" if DEBUG_MODE else "") + " ---"
                print(f"\n{stats_header}")
                print(f"  Elapsed Time: Total={elapsed_since_start:.2f}s | {datatype} Session={elapsed_since_datatype_start:.2f}s")
                print(f"  Molecules Processed: Session={datatype_molecules_processed_session} | Total Since Start={global_total_molecules_processed}")
                print(f"  Sequences Attempted: Session={datatype_sequences_attempted_session} | Total Since Start={global_total_sequences_attempted}")
                print(f"  Valid Sequences Found (Strict): Session={datatype_valid_sequences_generated_session} | Total Overall={total_valid_sequences_overall}")
                print(f"  Session Success Rate (Strict): {success_rate_session:.2f}%")
                print(f"  Avg Valid Seq/Molecule (Strict, Overall {datatype}): {avg_sequences_per_mol_overall:.2f}")
                print(f"  Avg High-Level Actions/Valid Seq (Session): {avg_high_level_actions_session:.2f}")

                if DEBUG_MODE:
                    avg_max_components_session = (datatype_total_max_components_session / datatype_valid_sequences_generated_session) if datatype_valid_sequences_generated_session > 0 else 0.0
                    print(f"  Avg Max Components/Valid Seq (DEBUG Session): {avg_max_components_session:.2f}")

                print(f"----------------------------------------------------")
                # Reset stats counter
                molecules_processed_since_stats = 0

            # --- Checkpoint Saving ---
            # Check if it's time to save checkpoint
            if molecules_processed_since_checkpoint >= CHECKPOINT_FREQUENCY or i == total_source_molecules_in_datatype_effective - 1:
                print(f"\nSaving results checkpoint for {datatype} at molecule index {i}...")
                try:
                    # Save the current state of all_results
                    with open(results_checkpoint_path, "wb") as f:
                        pickle.dump(all_results, f)
                    print("Checkpoint saved.")
                    # Reset checkpoint counter
                    molecules_processed_since_checkpoint = 0
                except Exception as e:
                    print(f"Error saving results checkpoint: {e}")

        # --- Save Final Results for Datatype ---
        # Determine final output path based on DEBUG mode
        if DEBUG_MODE:
            final_results_path = os.path.join(RESULTS_DIR, f"transformations_{datatype}_debug.pkl")
            print(f"\nFinished processing {datatype} (DEBUG). Saving final results to {final_results_path}...")
        else:
            final_results_path = os.path.join(RESULTS_DIR, f"transformations_{datatype}.pkl")
            print(f"\nFinished processing {datatype}. Saving final results to {final_results_path}...")
        try:
            # Save the final complete results for this datatype
            with open(final_results_path, "wb") as f:
                pickle.dump(all_results, f)
            final_mol_count = len(all_results)
            # Recalculate final sequence count from the saved dictionary
            final_seq_count = sum(len(v) for v in all_results.values())
            print(f"Successfully saved {final_mol_count} molecule entries with a total of {final_seq_count} valid (strict) sequences for {datatype}.")
        except Exception as e:
            print(f"Error saving final results for {datatype}: {e}")

    # --- Overall Summary ---
    end_time = time.time()
    summary_header = "\n--- Overall Summary" + (" (DEBUG MODE)" if DEBUG_MODE else "") + " ---"
    print(summary_header)
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print(f"Total source molecules processed (across all datatypes): {global_total_molecules_processed}")
    print(f"Total transformation sequences attempted: {global_total_sequences_attempted}")
    print(f"Total valid transformation sequences generated (Strict): {global_total_valid_sequences_generated}")
    # Calculate overall success rate based on global counters
    overall_success_rate = (global_total_valid_sequences_generated / global_total_sequences_attempted * 100) if global_total_sequences_attempted > 0 else 0
    print(f"Overall Success Rate (Strict): {overall_success_rate:.2f}%")
    completion_message = "Transformation dataset generation complete" + (" (DEBUG MODE)." if DEBUG_MODE else ".")
    print(completion_message)

# --- Need this for the script to run ---
if __name__ == "__main__":
    main()