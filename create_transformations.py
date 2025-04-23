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
def load_and_filter_molecules(path: str, max_atoms: int = MAX_ATOMS, datatype: str = "unknown") -> List[Tuple[str, Chem.Mol]]:
    # (Implementation remains the same)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"filtered_molecules_{datatype}.pkl")
    if os.path.exists(checkpoint_path):
        print(f"Loading filtered {datatype} molecules from checkpoint {checkpoint_path}")
        try:
            with open(checkpoint_path, "rb") as f:
                filtered_smiles = pickle.load(f)
            print(f"Loaded {len(filtered_smiles)} {datatype} molecules from checkpoint")
            return filtered_smiles
        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Processing from scratch.")

    print(f"Loading and filtering molecules from {path}")
    filtered_smiles: List[str] = []
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
                Chem.SanitizeMol(mol)
                Chem.Kekulize(mol)
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                if canonical_smiles in processed_smiles: continue
                mol_check = Chem.MolFromSmiles(canonical_smiles)
                if mol_check is None: continue
                num_heavy = mol_check.GetNumHeavyAtoms()
                if num_heavy == 0 or num_heavy > max_atoms: continue
                filtered_smiles.append(canonical_smiles)
                processed_smiles.add(canonical_smiles)
            except Exception as e: continue
    try:
        with open(checkpoint_path, "wb") as f:
            pickle.dump(filtered_smiles, f)
        print(f"Saved {len(filtered_smiles)} filtered {datatype} molecules to checkpoint {checkpoint_path}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
    return filtered_smiles


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

# <<< Renamed and Modified Action Selection Function >>>
def select_action_strategy(mol_design: MoleculeDesign, terminate_prob: float = 0.05) -> Optional[int]:
    """
    Selects a random valid action based on the current level.
    Level 0: Biased sampling (terminate_prob for Terminate, 1-terminate_prob for Select Atom).
    Level 1/2: Equal probability sampling across valid action *categories*.
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

    # --- Level 0: Biased Sampling ---
    if level == 0:
        valid_terminate_actions = [i for i in valid_action_indices if i == 0]
        valid_select_atom_actions = [i for i in valid_action_indices if 1 <= i <= num_real_atoms]

        can_terminate = bool(valid_terminate_actions)
        can_select_atom = bool(valid_select_atom_actions)

        if not can_terminate and not can_select_atom:
            # Should not happen if valid_action_indices is not empty
            if DEBUG_MODE: print("DEBUG (Action Select L0): No valid Terminate or Select Atom actions found despite valid indices.")
            return None
        elif can_terminate and not can_select_atom:
            # Only termination is possible (e.g., empty molecule, min actions met)
            return valid_terminate_actions[0] # Action 0
        elif not can_terminate and can_select_atom:
            # Only atom selection is possible (e.g., min actions not met)
            return random.choice(valid_select_atom_actions)
        else: # Both are possible
            # Sample based on probability
            if random.random() < terminate_prob: # terminate_prob chance (e.g., 0.05)
                if DEBUG_MODE: print(f"DEBUG (Action Select L0): Chose Terminate (Prob: {terminate_prob})")
                return valid_terminate_actions[0] # Action 0
            else: # 1 - terminate_prob chance (e.g., 0.95)
                selected_atom_action = random.choice(valid_select_atom_actions)
                if DEBUG_MODE: print(f"DEBUG (Action Select L0): Chose Select Atom {selected_atom_action} (Prob: {1.0-terminate_prob})")
                return selected_atom_action

    # --- Level 1 & 2: Equalized Category Sampling ---
    else:
        action_groups: Dict[str, List[int]] = {}
        if level == 1:
            remove_action_idx = vocab_size + num_real_atoms
            action_groups["Add Atom"] = [i for i in valid_action_indices if 0 <= i < vocab_size]
            action_groups["Select Existing Atom"] = [i for i in valid_action_indices if vocab_size <= i < remove_action_idx]
            action_groups["Remove Atom"] = [i for i in valid_action_indices if i == remove_action_idx]
        elif level == 2:
            action_groups["Set Bond Order"] = [i for i in valid_action_indices if 0 <= i <= 5]
            action_groups["Remove Bond"] = [i for i in valid_action_indices if i == 6]
        else:
            raise ValueError(f"Invalid action level for equalized sampling: {level}")

        valid_categories = {name: indices for name, indices in action_groups.items() if indices}
        if not valid_categories:
             if DEBUG_MODE: print(f"DEBUG (Action Select L{level}): No valid action categories found.")
             return None

        # Choose a category with equal probability
        selected_category_name = random.choice(list(valid_categories.keys()))
        # Choose an action within that category with equal probability
        selected_action_index = random.choice(valid_categories[selected_category_name])

        if DEBUG_MODE: print(f"DEBUG (Action Select L{level}): Chose Category '{selected_category_name}', Action {selected_action_index}")
        return selected_action_index


# --- generate_single_transformation (Tracks max components in DEBUG mode) ---
def generate_single_transformation(
    initial_mol_design: MoleculeDesign,
    start_smiles: str,
    config: MoleculeConfig,
    max_low_level_steps_safety_limit: int
) -> Optional[Tuple[str, str, List[int], int, int]]: # Added int for max_components
    """
    Attempts sequence generation, managing high-level action counts locally.
    Enforces min actions by preventing early termination. Uses _get_smiles_for_check.
    Tracks max components if DEBUG_MODE is True.
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

        # <<< Use the new selection strategy >>>
        action = select_action_strategy(current_mol_design)
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
                select_atom_actions = [i for i in range(1, len(mask)) if not mask[i]]
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
            next_mol_design = current_mol_design
            current_level = next_mol_design.current_action_level

            if DEBUG_MODE:
                max_components_this_sequence = max(max_components_this_sequence, current_mol_design.num_components)
                if current_mol_design.num_components > 1:
                     print(f"DEBUG INFO: Step {low_level_step_count} - Components = {current_mol_design.num_components} (Max so far: {max_components_this_sequence})")

            if prev_level != 0 and current_level == 0:
                high_level_action_count += 1
                if DEBUG_MODE: print(f"DEBUG INFO: High-level action count incremented to {high_level_action_count} at low-level step {low_level_step_count}.")

        except (ValueError, IndexError, RuntimeError) as e:
             if DEBUG_MODE: print(f"DEBUG FAIL (Take Action Error): Error during take_action(action={action}) at low-level step {low_level_step_count}: {e}")
             return None

        next_smiles = None
        perform_cycle_check = modifies_structure
        if perform_cycle_check:
            if DEBUG_MODE: print(f"DEBUG INFO: Performing cycle check for low-level step {low_level_step_count} (structure modified).")
            cycle_detected = False
            next_smiles = next_mol_design._get_smiles_for_check()

            if next_smiles is None:
                if DEBUG_MODE: print(f"DEBUG FAIL (Cycle Check SMILES Error/None): SMILES None or generation failed after low-level step {low_level_step_count}.")
                return None

            if next_smiles != "" and next_smiles in visited_smiles:
                cycle_detected = True
                if DEBUG_MODE: print(f"DEBUG FAIL (Cycle): Modifying action led back to visited SMILES after low-level step {low_level_step_count}, SMILES: {next_smiles}")

            if cycle_detected: return None
        else:
            if DEBUG_MODE: print(f"DEBUG INFO: Skipping cycle check for low-level step {low_level_step_count} (structure not modified).")

        action_sequence.append(action)
        if perform_cycle_check and next_smiles is not None and next_smiles != "" and next_smiles not in visited_smiles:
             visited_smiles.add(next_smiles)
             if DEBUG_MODE: print(f"DEBUG INFO: Added SMILES to visited set: {next_smiles}")

        low_level_step_count += 1
    # --- End of While Loop ---

    if not terminated_early and low_level_step_count >= max_low_level_steps_safety_limit:
        if DEBUG_MODE: print(f"DEBUG FAIL (Safety Break): Exceeded max low-level steps ({max_low_level_steps_safety_limit}).")
        return None

    if not terminated_early and high_level_action_count >= config.max_actions:
        if DEBUG_MODE: print(f"DEBUG INFO: Reached max high-level actions ({config.max_actions}). Checking if terminable.")
        if current_mol_design.current_action_level == 0 and current_mol_design.is_terminable():
            try:
                terminate_action = 0
                mask = current_mol_design.current_action_mask
                if mask is not None and not mask[terminate_action]:
                    current_mol_design.take_action(terminate_action)
                    action_sequence.append(terminate_action)
                    terminated_early = True
                    if DEBUG_MODE: print(f"DEBUG INFO: Force terminated at max high-level actions.")
                else:
                    if DEBUG_MODE: print(f"DEBUG FAIL (Force Terminate @ Max): Cannot force terminate. State L0 but Terminate masked or unavailable.")
                    return None
            except Exception as e:
                if DEBUG_MODE: print(f"DEBUG FAIL (Force Terminate Error @ Max): Error during force termination: {e}")
                return None
        else:
            if DEBUG_MODE: print(f"DEBUG FAIL (End State @ Max): Reached max high-level actions but cannot terminate. Level={current_mol_design.current_action_level}, Terminable={current_mol_design.is_terminable()}")
            return None

    if DEBUG_MODE:
        final_smiles_before_val = "(Error getting SMILES)"
        try:
             final_smiles_before_val = current_mol_design._get_smiles_for_check()
             if final_smiles_before_val is None: final_smiles_before_val = "(Check Failed)"
             elif final_smiles_before_val == "": final_smiles_before_val = "'' (Empty)"
        except Exception as e: final_smiles_before_val = f"(Error: {e})"
        print(f"\nDEBUG INFO (Pre-Validation):")
        print(f"  - synthesis_done: {current_mol_design.synthesis_done}")
        print(f"  - infeasibility_flag: {current_mol_design.infeasibility_flag}")
        print(f"  - is_currently_connected: {current_mol_design.is_currently_connected}")
        print(f"  - Num Atoms (Internal): {len(current_mol_design.atoms) - 1}")
        print(f"  - Final SMILES (Check Attempted): {final_smiles_before_val}")
        print(f"  - Low-Level Action Seq Len: {len(action_sequence)}")
        print(f"  - Final High-Level Action Count (Local): {high_level_action_count}")
        print(f"  - Max Components Encountered (DEBUG): {max_components_this_sequence}\n")


    # --- Final Strict Validation ---
    final_smiles = None
    if not current_mol_design.synthesis_done or current_mol_design.infeasibility_flag:
        if DEBUG_MODE: print(f"DEBUG FAIL (Final Flags): Failed env flags check. Done={current_mol_design.synthesis_done}, Infeasible={current_mol_design.infeasibility_flag}")
        return None
    num_final_real_atoms = len(current_mol_design.atoms) - 1
    if num_final_real_atoms > 1 and not current_mol_design.is_currently_connected:
         if DEBUG_MODE: print(f"DEBUG FAIL (Final Connectivity): Molecule disconnected.")
         return None
    try:
        final_smiles = current_mol_design.to_smiles(canonical=True)
        if final_smiles is None:
             if num_final_real_atoms == 0: final_smiles = ""
             else:
                  if DEBUG_MODE: print(f"DEBUG FAIL (Final SMILES None/Sanitize): Final SMILES is None despite atoms (likely sanitization failure within finalize).")
                  return None
        if DEBUG_MODE: print(f"DEBUG INFO: Final molecule passed Sanitization (via to_smiles/finalize).")
    except Exception as e:
        if DEBUG_MODE: print(f"DEBUG FAIL (Final RDKit/Sanitize Error during to_smiles/finalize): {e}")
        return None
    if final_smiles == start_smiles:
        if DEBUG_MODE: print(f"DEBUG FAIL (Start=End): Final SMILES matches start. Sequence: {action_sequence}")
        return None
    if high_level_action_count < config.min_actions:
         if DEBUG_MODE: print(f"DEBUG FAIL (Min High-Level Actions): Final count {high_level_action_count} < {config.min_actions}. Should have been prevented.")
         return None

    # --- Success ---
    if DEBUG_MODE: print(f"DEBUG SUCCESS (Strict): Sequence passed all checks! Low-level Len={len(action_sequence)}, High-level Count={high_level_action_count}, Max Components={max_components_this_sequence}")
    return start_smiles, final_smiles, action_sequence, high_level_action_count, max_components_this_sequence


# --- Main Function (Accumulates and prints max components in DEBUG) ---
def main():
    # (No changes needed in main function - it already calls the action selection function
    #  and handles the results correctly)
    start_time = time.time()
    print(f"Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Current User's Login: {os.getenv('USER', 'unknown')}")
    print("Starting transformation dataset generation.")
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

    global_total_sequences_attempted = 0
    global_total_valid_sequences_generated = 0
    global_total_molecules_processed = 0

    try:
        reverse_atom_lookup = build_reverse_atom_lookup(CONFIG)
        print("Successfully built reverse atom lookup.")
    except Exception as e:
        print(f"Fatal Error: Could not build reverse atom lookup from config: {e}")
        return

    for datatype, filepath in DATATYPES.items():
        datatype_start_time = time.time()
        print(f"\n--- Processing {datatype} data from {filepath} ---")

        initial_smiles_list = load_and_filter_molecules(filepath, MAX_ATOMS, datatype)
        total_source_molecules_in_datatype_full = len(initial_smiles_list)
        if total_source_molecules_in_datatype_full == 0:
            print(f"No valid starting molecules found or loaded for {datatype}. Skipping.")
            continue

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
        datatype_total_max_components_session = 0

        if os.path.exists(results_checkpoint_path):
            print(f"Loading existing results checkpoint: {results_checkpoint_path}")
            try:
                with open(results_checkpoint_path, "rb") as f:
                    all_results = pickle.load(f)
                processed_smiles_count_ckpt = len(all_results)
                initial_valid_sequences_count = sum(len(v) for v in all_results.values())
                start_index = 0
                processed_smiles_set_ckpt = set(all_results.keys())
                while start_index < total_source_molecules_in_datatype_effective and effective_molecule_list[start_index] in processed_smiles_set_ckpt:
                    start_index += 1
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
        pbar = tqdm(range(start_index, total_source_molecules_in_datatype_effective), desc=tqdm_desc, initial=start_index, total=total_source_molecules_in_datatype_effective)

        for i in pbar:
            smiles_for_mol_design = effective_molecule_list[i]
            if smiles_for_mol_design in all_results and len(all_results.get(smiles_for_mol_design, [])) >= TRANSFORMATIONS_PER_MOLECULE:
                continue

            valid_transformations: List[Tuple[str, str, List[int]]] = all_results.get(smiles_for_mol_design, [])
            attempts_for_mol = 0
            total_attempts_for_molecule = 0
            sequences_found_this_mol_session = 0

            try:
                initial_mol_design, _ = MoleculeDesign.from_smiles(CONFIG, smiles_for_mol_design)
                if initial_mol_design.synthesis_done:
                    raise RuntimeError("MoleculeDesign.from_smiles returned a finalized instance!")
                actual_start_smiles = initial_mol_design._get_smiles_for_check()
                if actual_start_smiles is None:
                    raise ValueError("Initial molecule failed SMILES check (e.g., sanitization)")
            except (ValueError, RuntimeError, KeyError, IndexError) as e:
                if DEBUG_MODE: print(f"\nDEBUG FAIL (Init/SMILES): Failed initial MoleculeDesign or getting initial SMILES for {smiles_for_mol_design}: {e}.")
                else: print(f"\nWarning: Skipping molecule {smiles_for_mol_design} due to initialization error: {e}")
                if smiles_for_mol_design not in all_results: all_results[smiles_for_mol_design] = []
                datatype_molecules_processed_session += 1; molecules_processed_since_checkpoint += 1; molecules_processed_since_stats += 1; global_total_molecules_processed += 1
                continue

            if not actual_start_smiles and initial_mol_design.GetNumAtoms() > 0:
                 if DEBUG_MODE: print(f"\nDEBUG FAIL (Init/SMILES Empty): Initial molecule check yielded empty SMILES despite atoms for {smiles_for_mol_design}.")
                 if smiles_for_mol_design not in all_results: all_results[smiles_for_mol_design] = []
                 datatype_molecules_processed_session += 1; molecules_processed_since_checkpoint += 1; molecules_processed_since_stats += 1; global_total_molecules_processed += 1
                 continue

            initial_mol_design_copy = copy.deepcopy(initial_mol_design)

            while len(valid_transformations) < TRANSFORMATIONS_PER_MOLECULE and total_attempts_for_molecule < MAX_TOTAL_ATTEMPTS_PER_MOLECULE:
                total_attempts_for_molecule += 1
                datatype_sequences_attempted_session += 1
                global_total_sequences_attempted += 1
                attempts_for_mol += 1

                result = generate_single_transformation(
                    initial_mol_design_copy,
                    actual_start_smiles,
                    CONFIG,
                    MAX_LOW_LEVEL_STEPS_SAFETY
                )

                if result is not None:
                    start_smi, end_smi, low_level_seq, high_level_count, max_components = result
                    valid_transformations.append((start_smi, end_smi, low_level_seq))
                    sequences_found_this_mol_session += 1
                    datatype_valid_sequences_generated_session += 1
                    global_total_valid_sequences_generated += 1
                    datatype_total_high_level_actions_session += high_level_count
                    if DEBUG_MODE:
                        datatype_total_max_components_session += max_components
                    attempts_for_mol = 0

                if attempts_for_mol >= MAX_ATTEMPTS_PER_TRANSFORMATION and len(valid_transformations) < TRANSFORMATIONS_PER_MOLECULE:
                    attempts_for_mol = 0

            datatype_molecules_processed_session += 1
            molecules_processed_since_checkpoint += 1
            molecules_processed_since_stats += 1
            global_total_molecules_processed += 1

            if valid_transformations:
                 all_results[smiles_for_mol_design] = valid_transformations
            elif smiles_for_mol_design not in all_results:
                 all_results[smiles_for_mol_design] = []

            if molecules_processed_since_stats >= STATS_FREQUENCY or i == total_source_molecules_in_datatype_effective - 1:
                current_time = time.time()
                elapsed_since_start = current_time - start_time
                elapsed_since_datatype_start = current_time - datatype_start_time
                total_valid_sequences_overall = initial_valid_sequences_count + datatype_valid_sequences_generated_session
                processed_mol_count_overall = start_index + datatype_molecules_processed_session
                avg_sequences_per_mol_overall = total_valid_sequences_overall / processed_mol_count_overall if processed_mol_count_overall > 0 else 0
                success_rate_session = (datatype_valid_sequences_generated_session / datatype_sequences_attempted_session * 100) if datatype_sequences_attempted_session > 0 else 0
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
                molecules_processed_since_stats = 0

            if molecules_processed_since_checkpoint >= CHECKPOINT_FREQUENCY or i == total_source_molecules_in_datatype_effective - 1:
                print(f"\nSaving results checkpoint for {datatype} at molecule index {i}...")
                try:
                    with open(results_checkpoint_path, "wb") as f:
                        pickle.dump(all_results, f)
                    print("Checkpoint saved.")
                    molecules_processed_since_checkpoint = 0
                except Exception as e:
                    print(f"Error saving results checkpoint: {e}")

        if DEBUG_MODE:
            final_results_path = os.path.join(RESULTS_DIR, f"transformations_{datatype}_debug.pkl")
            print(f"\nFinished processing {datatype} (DEBUG). Saving final results to {final_results_path}...")
        else:
            final_results_path = os.path.join(RESULTS_DIR, f"transformations_{datatype}.pkl")
            print(f"\nFinished processing {datatype}. Saving final results to {final_results_path}...")
        try:
            with open(final_results_path, "wb") as f:
                pickle.dump(all_results, f)
            final_mol_count = len(all_results)
            final_seq_count = sum(len(v) for v in all_results.values())
            print(f"Successfully saved {final_mol_count} molecule entries with a total of {final_seq_count} valid (strict) sequences for {datatype}.")
        except Exception as e:
            print(f"Error saving final results for {datatype}: {e}")

    end_time = time.time()
    summary_header = "\n--- Overall Summary" + (" (DEBUG MODE)" if DEBUG_MODE else "") + " ---"
    print(summary_header)
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print(f"Total source molecules processed (across all datatypes): {global_total_molecules_processed}")
    print(f"Total transformation sequences attempted: {global_total_sequences_attempted}")
    print(f"Total valid transformation sequences generated (Strict): {global_total_valid_sequences_generated}")
    overall_success_rate = (global_total_valid_sequences_generated / global_total_sequences_attempted * 100) if global_total_sequences_attempted > 0 else 0
    print(f"Overall Success Rate (Strict): {overall_success_rate:.2f}%")
    completion_message = "Transformation dataset generation complete" + (" (DEBUG MODE)." if DEBUG_MODE else ".")
    print(completion_message)


if __name__ == "__main__":
    main()