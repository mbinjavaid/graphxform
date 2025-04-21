import time
import pickle
import os
import glob
import copy
import numpy as np # Need numpy for mask checking potentially
from rdkit import Chem, RDLogger
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional, Any
import sys # For checking rdkit version attributes
import traceback

# Make sure these are correctly importable from your project structure
from config import MoleculeConfig
from molecule_design import MoleculeDesign


# # Suppress RDKit warnings (optional)
# RDLogger.DisableLog('rdApp.*')

# Configuration
CHECKPOINT_DIR = "./data/chembl/checkpoints"
FINAL_DATA_DIR = "./data/chembl"
OUTPUT_DIR = "./data/chembl/rl_datasets"
MAX_RL_STEPS = 1000 # Add a safeguard against infinite loops

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Helper: Build Reverse Atom Lookup ---
def build_reverse_atom_lookup(config: MoleculeConfig) -> Dict[Tuple[int, int, int], int]:
    """Creates a lookup from (atomic_num, charge, chiral) to vocab index."""
    lookup = {}
    # Ensure vocabulary_atom_names exists or derive it if needed
    if hasattr(config, 'vocabulary_atom_names'):
        vocab_names = config.vocabulary_atom_names
    else:
        # If not precomputed, derive from keys (assuming order matters and is consistent)
        vocab_names = list(config.atom_vocabulary.keys())
        # Consider attaching this to config instance if needed elsewhere?
        # config.vocabulary_atom_names = vocab_names

    if not vocab_names:
        raise ValueError("Atom vocabulary in config appears empty.")

    for i, name in enumerate(vocab_names):
        try:
            atom_config = config.atom_vocabulary[name]
        except KeyError:
            print(f"Warning: Atom name '{name}' found in vocab_names but not in atom_vocabulary keys.")
            continue

        try:
            atomic_num = atom_config['atomic_number']
            # Use .get() with default 0 for optional properties
            charge = atom_config.get('formal_charge', 0)
            chiral = atom_config.get('chiral_tag', 0) # 0: unspecified, 1: CW, 2: CCW (RDKit mapping)
        except KeyError as e:
            print(f"Warning: Missing expected property {e} for atom '{name}' in config. Skipping.")
            continue

        key = (atomic_num, charge, chiral)
        vocab_idx = i + 1 # 1-based index

        # Store the mapping for the specific properties
        if key in lookup:
             print(f"Warning: Duplicate atom definition found for key {key} ('{name}'). Overwriting.")
        lookup[key] = vocab_idx

        # Add a fallback mapping for non-chiral lookup if this entry is chiral
        # This allows finding a chiral atom even if the query is non-chiral
        if chiral != 0:
             key_no_chiral = (atomic_num, charge, 0)
             if key_no_chiral not in lookup:
                  # Only add if a non-chiral version *doesn't* already exist specifically
                  lookup[key_no_chiral] = vocab_idx

    if not lookup:
         print("Warning: Reverse atom lookup is empty. Check atom_vocabulary in config.")

    return lookup


# --- Helper: Bond Type to RL Order ---
# Check if rdkit version has these higher bond types, provide dummy values if not
# Needed for compatibility with older rdkit versions that might lack these constants
if not hasattr(Chem.BondType, 'QUADRUPLE'): Chem.BondType.QUADRUPLE = sys.maxsize # Assign dummy unique value
if not hasattr(Chem.BondType, 'QUINTUPLE'): Chem.BondType.QUINTUPLE = sys.maxsize - 1
if not hasattr(Chem.BondType, 'HEXTUPLE'): Chem.BondType.HEXTUPLE = sys.maxsize - 2

# Populate the dictionary ensuring keys exist, potentially with dummy values if rdkit is old
BOND_TYPE_TO_RL_ORDER = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
    Chem.BondType.QUADRUPLE: 4,
    Chem.BondType.QUINTUPLE: 5,
    Chem.BondType.HEXTUPLE: 6,
}
# Remove entries whose keys are dummy values (meaning rdkit version didn't have them)
BOND_TYPE_TO_RL_ORDER = {k: v for k, v in BOND_TYPE_TO_RL_ORDER.items() if v <= 6}

REMOVE_BOND_RL_ACTION = 6 # L2 action index for removing bond


def load_latest_transformation_data(datatype: str) -> List[Dict[str, Any]]:
    """Loads transformation data from final file or latest checkpoint."""
    final_output_path = os.path.join(FINAL_DATA_DIR, f"transformation_dataset_{datatype}.pickle")
    checkpoint_pattern = os.path.join(CHECKPOINT_DIR, f"transformation_data_{datatype}_*.pkl")

    if os.path.exists(final_output_path):
        print(f"Loading final transformation data from: {final_output_path}")
        try:
            with open(final_output_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading final file {final_output_path}: {e}. Trying checkpoints.")

    checkpoint_files = glob.glob(checkpoint_pattern)
    if not checkpoint_files:
        print(f"Warning: No final data or checkpoints found for {datatype}.")
        return []

    # Sort by modification time (or timestamp in filename if reliable)
    # Assuming timestamp format YYYYMMDD_HHMMSS in filename like transformation_data_train_YYYYMMDD_HHMMSS.pkl
    try:
        # Extract timestamp and sort
         checkpoint_files.sort(key=lambda f: f.split('_')[-1].split('.')[0], reverse=True)
    except Exception:
         # Fallback sort by modification time if filename parsing fails
         print("Warning: Could not parse timestamps from checkpoint filenames, sorting by modification time.")
         checkpoint_files.sort(key=os.path.getmtime, reverse=True)


    latest_checkpoint = checkpoint_files[0]
    print(f"Loading latest transformation checkpoint: {latest_checkpoint}")
    try:
        with open(latest_checkpoint, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading checkpoint {latest_checkpoint}: {e}")
        return []

def convert_single_ged_to_rl(
    transformation: Dict[str, Any],
    config: MoleculeConfig,
    reverse_atom_lookup: Dict[Tuple[int, int, int], int]
) -> Optional[List[int]]:
    """
    Attempts to convert a single GED edit path to an RL action sequence.
    Includes debugging logs and fixes for skipped edits.
    """
    source_smiles = transformation['source_smiles']
    target_smiles = transformation['target_smiles']
    ged_path = transformation['edit_path'][:-1] # Exclude metadata

    # 1. Initialization
    try:
        # from_smiles returns the instance and the rdkit_to_internal_map
        mol_state, rdkit_to_internal_map = MoleculeDesign.from_smiles(
            config, source_smiles, do_finish=False # Use the simplified direct construction
        )
        if mol_state is None or rdkit_to_internal_map is None:
             print(f"  Failed to initialize MoleculeDesign from source SMILES: {source_smiles}")
             return None
    except Exception as e:
        print(f"  Error initializing MoleculeDesign for {source_smiles}: {e}")
        traceback.print_exc() # More debug info
        return None

    rl_action_sequence = []
    all_edits = list(enumerate(ged_path)) # Keep original index for tracking
    processed_edit_indices = set()
    deferred_edit_indices = set() # Edits deferred in the current pass
    applied_in_pass = True
    total_rl_steps = 0

    # --- Mapping from GED atom properties to vocab index ---
    def get_vocab_idx(element, charge, chiral_tag) -> Optional[int]:
        key = (element, charge, chiral_tag)
        idx = reverse_atom_lookup.get(key)
        if idx is None and chiral_tag != 0: # Try without chirality
             key_no_chiral = (element, charge, 0)
             idx = reverse_atom_lookup.get(key_no_chiral)
        return idx

    # --- Helper to apply sequence and update state ---
    def apply_rl_sequence(edit_idx, sequence: List[int], current_state: MoleculeDesign, current_map: Dict[int, int]) -> \
            Tuple[Optional[MoleculeDesign], Optional[Dict[int, int]], bool]:
        """
        Applies a sequence of RL actions to a state, updating the state and index map.
        Includes detailed logging and explicit mask handling during copy.
        """
        nonlocal total_rl_steps  # Ensure this is correctly linked to the outer scope variable

        # --- Explicit Copying ---
        # Deepcopy the main state object
        temp_state = copy.deepcopy(current_state)
        # Explicitly copy the mask IF it exists, otherwise set to None
        if current_state.current_action_mask is not None:
            # *** ADDED EXPLICIT MASK COPY ***
            temp_state.current_action_mask = np.copy(current_state.current_action_mask)
            print(
                f"  DEBUG apply_rl_sequence: Edit={edit_idx}. EXPLICITLY COPIED mask. Len={len(temp_state.current_action_mask)}")
        else:
            # Ensure the copied state's mask is also None if original was None
            temp_state.current_action_mask = None
            print(f"  DEBUG apply_rl_sequence: Edit={edit_idx}. Original mask was None, setting copied mask to None.")

        temp_map = copy.deepcopy(current_map)  # Map copy seems fine
        # --- End Explicit Copying ---

        removed_internal_idx = None
        added_rdkit_idx = None
        new_internal_idx = None
        current_action_to_attempt = -1  # Initialize for error logging
        current_level = -1  # Initialize for error logging

        try:
            for i, action_in_seq in enumerate(sequence):
                current_action_to_attempt = action_in_seq  # Store the action for this iteration
                current_level = temp_state.current_action_level  # Get level BEFORE action

                if total_rl_steps >= MAX_RL_STEPS:
                    print(f"  Exceeded MAX_RL_STEPS ({MAX_RL_STEPS}). Aborting conversion.")
                    return None, None, False  # Indicate failure (not deferral)

                # --- Get the mask FROM THE TEMP STATE ---
                # This mask should now be the explicitly copied one or the one updated by take_action
                mask = temp_state.current_action_mask
                # ---

                # --- START: Mask Verification Log (Keep this) ---
                mask_value_at_action = "N/A"
                is_out_of_bounds = False
                mask_len_str = "None"
                if mask is None:
                    mask_value_at_action = "Mask is None"
                elif current_action_to_attempt >= len(mask):
                    mask_value_at_action = "Index OOB"
                    is_out_of_bounds = True
                    mask_len_str = str(len(mask))
                else:
                    mask_value_at_action = str(mask[current_action_to_attempt])
                    mask_len_str = str(len(mask))
                print(
                    f"    Verifying Mask Before Check: Action={current_action_to_attempt}, Level={current_level}, MaskValue={mask_value_at_action}, MaskLen={mask_len_str}, Edit={edit_idx}")
                # --- END: Mask Verification Log ---

                # --- Perform Mask Check using the retrieved mask ---
                is_masked = False
                if mask is None or is_out_of_bounds or mask[current_action_to_attempt]:
                    is_masked = True
                # --- End Mask Check ---

                if is_masked:
                    # --- Log Mask Failure ---
                    print(f"    --> Action {current_action_to_attempt} (L{current_level}) MASKED for edit {edit_idx}.")

                    # --- Optional: Log Valence Detail ---
                    # (Your existing detailed logging for L2 masked actions remains here)
                    try:
                        # Check if it was an L2 bond-setting action that got masked
                        if current_level == 2 and not temp_state.is_modifying_atom and current_action_to_attempt < REMOVE_BOND_RL_ACTION:
                            idx_A = temp_state.l0_selected_atom_idx
                            idx_B = None
                            if temp_state.l1_new_atom_type is not None:
                                idx_B = len(temp_state.atoms)
                            elif temp_state.l1_selected_existing_atom_idx is not None:
                                idx_B = temp_state.l1_selected_existing_atom_idx

                            if idx_A is not None and idx_B is not None and idx_A > 0:
                                is_valid_B = (temp_state.l1_new_atom_type is not None or (
                                            idx_B > 0 and (idx_B - 1) < (len(temp_state.atoms) - 1)))
                                if is_valid_B:
                                    rem_val = temp_state._get_remaining_valence()
                                    val_A_str, val_B_str = "N/A", "N/A"
                                    if (idx_A - 1) < len(rem_val): val_A_str = str(rem_val[idx_A - 1])
                                    if temp_state.l1_new_atom_type is not None:
                                        new_atom_vocab_idx = temp_state.l1_new_atom_type
                                        if 0 <= new_atom_vocab_idx < len(temp_state.vocabulary_valence):
                                            val_B_str = str(
                                                temp_state.vocabulary_valence[new_atom_vocab_idx]) + " (new total)"
                                        else:
                                            val_B_str = "? (new)"
                                    elif (idx_B - 1) < len(rem_val):
                                        val_B_str = str(rem_val[idx_B - 1])
                                    print(
                                        f"      L2 Bond Set({current_action_to_attempt + 1}) Mask Detail: A={idx_A}(rem={val_A_str}), B={idx_B}(rem={val_B_str})")
                                else:
                                    print(
                                        f"      L2 Bond Set Mask Detail: Invalid B index ({idx_B}) for valence check.")
                            else:
                                print(f"      L2 Bond Set Mask Detail: Invalid A({idx_A}) or B({idx_B}) index.")

                        # Check if it was an L2 remove bond action that got masked
                        elif current_level == 2 and not temp_state.is_modifying_atom and current_action_to_attempt == REMOVE_BOND_RL_ACTION:
                            idx_A = temp_state.l0_selected_atom_idx
                            idx_B = temp_state.l1_selected_existing_atom_idx
                            if idx_A is not None and idx_B is not None and idx_A > 0 and idx_B > 0 and idx_A != idx_B:
                                # Check bounds before accessing bonds array
                                if idx_A < temp_state.bonds.shape[0] and idx_B < temp_state.bonds.shape[1]:
                                    current_bond_val = temp_state.bonds[idx_A, idx_B]
                                    print(
                                        f"      L2 Remove Bond Mask Detail: Bond({idx_A},{idx_B}) exists = {current_bond_val > 0}")
                                else:
                                    print(
                                        f"      L2 Remove Bond Mask Detail: Invalid indices ({idx_A},{idx_B}) for bonds shape {temp_state.bonds.shape}")
                            else:
                                print(f"      L2 Remove Bond Mask Detail: Invalid A({idx_A}) or B({idx_B}) index.")

                    except Exception as log_e:
                        print(f"      Error during logging details for masked action: {log_e}")
                    # --- End Valence Log ---

                    return None, None, True  # Indicate deferral due to mask
                # --- End Mask Failure Handling ---

                # --- If not masked, proceed with action ---

                # Store potential index to be removed *before* action modifies state
                if current_level == 2 and temp_state.is_modifying_atom and current_action_to_attempt == temp_state.REMOVE_ATOM_ACTION_L2_MODIFY:
                    removed_internal_idx = temp_state.atom_to_modify

                # Store intended RDKit index if adding atom
                if current_level == 1 and current_action_to_attempt < temp_state.vocab_size:
                    original_edit = ged_path[edit_idx]
                    if original_edit['operation'] == 'insert_node':
                        added_rdkit_idx = original_edit['target_idx']

                # --- Apply the action to the temporary state ---
                temp_state.take_action(current_action_to_attempt)  # This updates temp_state including its mask
                total_rl_steps += 1
                # --- Action applied ---

                # Determine new internal index *after* action (if atom was added)
                if current_level == 1 and current_action_to_attempt < temp_state.vocab_size:
                    new_internal_idx = len(temp_state.atoms) - 1

            # --- End of loop through action sequence ---

            # --- Update map after successful application ---
            if removed_internal_idx is not None:
                removed_rdkit_idx = None
                for r_idx, i_idx in temp_map.items():
                    if i_idx == removed_internal_idx: removed_rdkit_idx = r_idx; break
                if removed_rdkit_idx is not None: del temp_map[removed_rdkit_idx]
                keys_to_update = [r_idx for r_idx, i_idx in temp_map.items() if i_idx > removed_internal_idx]
                for r_idx in keys_to_update: temp_map[r_idx] -= 1

            if added_rdkit_idx is not None and new_internal_idx is not None:
                temp_map[added_rdkit_idx] = new_internal_idx

            return temp_state, temp_map, False  # Success

        except Exception as e:
            print(f"    Error applying action {current_action_to_attempt} (L{current_level}) for edit {edit_idx}: {e}")
            traceback.print_exc(limit=5)
            return None, None, True  # Defer on error

    # --- Main Simulation Loop ---
    # Pre-index edits for faster lookup during insertion pairing
    atom_insertion_edits = {e['target_idx']: (idx, e) for idx, e in all_edits if e['operation'] == 'insert_node'}
    edge_insertion_edits = [(idx, e) for idx, e in all_edits if e['operation'] == 'insert_edge']
    processed_initial_bond_indices = set() # Track edge insertions used with atom insertions

    while applied_in_pass:
        if total_rl_steps >= MAX_RL_STEPS: break
        applied_in_pass = False
        current_deferred = list(deferred_edit_indices) # Work on copy for this pass
        deferred_edit_indices.clear()

        # --- Define order for processing ---
        edit_order_indices = [idx for idx, e in all_edits if e['operation'] == 'delete_edge' and idx not in processed_edit_indices and idx not in current_deferred] + \
                             [idx for idx, e in all_edits if e['operation'] == 'delete_node' and idx not in processed_edit_indices and idx not in current_deferred] + \
                             [idx for idx, e in all_edits if e['operation'] == 'substitute_edge' and idx not in processed_edit_indices and idx not in current_deferred] + \
                             [idx for idx, e in all_edits if e['operation'] == 'substitute_node' and idx not in processed_edit_indices and idx not in current_deferred] + \
                             [idx for idx, e in all_edits if e['operation'] == 'insert_node' and idx not in processed_edit_indices and idx not in current_deferred] + \
                             [idx for idx, e in all_edits if e['operation'] == 'insert_edge' and idx not in processed_edit_indices and idx not in current_deferred and idx not in processed_initial_bond_indices]

        # Add deferred edits at the end to retry them
        edit_order_indices += [idx for idx in current_deferred if idx not in processed_edit_indices]

        for edit_idx in edit_order_indices:
            if edit_idx in processed_edit_indices: continue
            if total_rl_steps >= MAX_RL_STEPS: break

            edit_info = ged_path[edit_idx]
            op = edit_info['operation']
            rl_seq = []
            initial_bond_edit_idx = -1 # Reset for each edit check

            # --- START FIX: Check for Deleted Atoms BEFORE Try Block ---
            involved_rdkit_indices = []
            # Collect all relevant RDKit indices from the edit info
            if 'atom1_idx' in edit_info: involved_rdkit_indices.extend([edit_info['atom1_idx'], edit_info['atom2_idx']])
            if 'source_atom1' in edit_info: involved_rdkit_indices.extend([edit_info['source_atom1'], edit_info['source_atom2']])
            if 'target_atom1' in edit_info: involved_rdkit_indices.extend([edit_info['target_atom1'], edit_info['target_atom2']]) # For substitute_edge
            if 'source_idx' in edit_info: involved_rdkit_indices.append(edit_info['source_idx'])
            # Don't check target_idx for insert_node here, as it *shouldn't* be in the map yet

            atom_deleted = False
            # Check if any involved RDKit index (except target_idx for insert) is missing from the map
            for r_idx in set(involved_rdkit_indices): # Use set to avoid redundant checks
                if r_idx not in rdkit_to_internal_map:
                    # Exception: If it's an insert_node edit, the target_idx is expected to be missing
                    if op == 'insert_node' and r_idx == edit_info.get('target_idx'):
                         continue # This specific index is allowed to be missing for insert_node
                    atom_deleted = True
                    break

            if atom_deleted:
                # print(f"  Skipping edit {edit_idx} ({op}) involving deleted atom. Marking as processed.") # Optional log
                processed_edit_indices.add(edit_idx)
                # If this was an insert_node, also mark its paired bond (if found) as processed to avoid dangling references
                if op == 'insert_node':
                     target_r_idx = edit_info['target_idx']
                     for edge_i, edge_edit in edge_insertion_edits:
                          if edge_i in processed_edit_indices or edge_i in processed_initial_bond_indices: continue
                          if target_r_idx in (edge_edit['atom1_idx'], edge_edit['atom2_idx']):
                               processed_edit_indices.add(edge_i)
                               processed_initial_bond_indices.add(edge_i)
                               # print(f"    Also marking paired bond edit {edge_i} as processed due to deleted atom.")
                               break # Mark only the first found paired bond
                applied_in_pass = True # Count skipping as progress
                continue # Move to the next edit_idx
            # --- END FIX ---

            try:
                # Determine RL sequence based on operation
                # Redundant checks for index presence removed here, handled by the fix above
                if op == 'delete_edge':
                    r_idx1, r_idx2 = edit_info['atom1_idx'], edit_info['atom2_idx']
                    i_idx1, i_idx2 = rdkit_to_internal_map[r_idx1], rdkit_to_internal_map[r_idx2]
                    l1_select_action = mol_state.vocab_size + (i_idx2 - 1)
                    rl_seq = [i_idx1, l1_select_action, REMOVE_BOND_RL_ACTION]

                elif op == 'delete_node':
                    r_idx = edit_info['source_idx']
                    i_idx = rdkit_to_internal_map[r_idx]
                    l1_modify_action = mol_state.vocab_size + (len(mol_state.atoms) - 1)
                    rl_seq = [i_idx, l1_modify_action, mol_state.REMOVE_ATOM_ACTION_L2_MODIFY]

                elif op == 'substitute_edge':
                    r_idx1, r_idx2 = edit_info['source_atom1'], edit_info['source_atom2']
                    i_idx1, i_idx2 = rdkit_to_internal_map[r_idx1], rdkit_to_internal_map[r_idx2]
                    new_bond_type = edit_info['to_bond_type']
                    new_order = BOND_TYPE_TO_RL_ORDER.get(new_bond_type)
                    if new_order is None: continue
                    l1_select_action = mol_state.vocab_size + (i_idx2 - 1)
                    l2_action = new_order - 1
                    rl_seq = [i_idx1, l1_select_action, l2_action]

                elif op == 'substitute_node':
                    r_idx = edit_info['source_idx']
                    i_idx = rdkit_to_internal_map[r_idx]
                    new_vocab_idx = get_vocab_idx(edit_info['to_element'], edit_info['to_charge'], edit_info['to_chiral'])
                    if new_vocab_idx is None: continue
                    l1_modify_action = mol_state.vocab_size + (len(mol_state.atoms) - 1)
                    l2_action = new_vocab_idx - 1
                    rl_seq = [i_idx, l1_modify_action, l2_action]

                elif op == 'insert_node':
                    target_r_idx = edit_info['target_idx']
                    initial_bond_edit_info = None
                    anchor_r_idx = -1
                    for edge_i, edge_edit in edge_insertion_edits:
                         if edge_i in processed_edit_indices or edge_i in processed_initial_bond_indices: continue
                         # Ensure the *other* atom in the bond edit exists
                         potential_anchor = edge_edit['atom1_idx'] if edge_edit['atom2_idx'] == target_r_idx else edge_edit['atom2_idx']
                         if target_r_idx in (edge_edit['atom1_idx'], edge_edit['atom2_idx']) and potential_anchor in rdkit_to_internal_map:
                              initial_bond_edit_idx = edge_i
                              initial_bond_edit_info = edge_edit
                              anchor_r_idx = potential_anchor
                              break # Found first valid initial bond
                    if initial_bond_edit_info is None: continue # Defer if no valid initial bond found yet

                    anchor_i_idx = rdkit_to_internal_map[anchor_r_idx]
                    new_vocab_idx = get_vocab_idx(edit_info['element'], edit_info['charge'], edit_info['chiral_tag'])
                    bond_order = BOND_TYPE_TO_RL_ORDER.get(initial_bond_edit_info['bond_type'])
                    if new_vocab_idx is None or bond_order is None: continue

                    l1_action_add = new_vocab_idx - 1
                    l2_action_bond = bond_order - 1
                    rl_seq = [anchor_i_idx, l1_action_add, l2_action_bond]

                elif op == 'insert_edge':
                    if edit_idx in processed_initial_bond_indices or edit_idx in processed_edit_indices: continue
                    r_idx1, r_idx2 = edit_info['atom1_idx'], edit_info['atom2_idx']
                    # Check handled by the fix above
                    i_idx1, i_idx2 = rdkit_to_internal_map[r_idx1], rdkit_to_internal_map[r_idx2]
                    bond_order = BOND_TYPE_TO_RL_ORDER.get(edit_info['bond_type'])
                    if bond_order is None: continue
                    l1_select_action = mol_state.vocab_size + (i_idx2 - 1)
                    l2_action = bond_order - 1
                    rl_seq = [i_idx1, l1_select_action, l2_action]

                # --- Apply the sequence if determined ---
                if rl_seq:
                    new_state, new_map, needs_deferral = apply_rl_sequence(edit_idx, rl_seq, mol_state, rdkit_to_internal_map)

                    if needs_deferral:
                        # --- START DEBUG LOG ---
                        print(f"  DEFERRING Edit {edit_idx}: {edit_info}")
                        # --- END DEBUG LOG ---
                        deferred_edit_indices.add(edit_idx)
                        # Also defer paired bond if atom insertion was deferred
                        if op == 'insert_node' and initial_bond_edit_idx != -1:
                             deferred_edit_indices.add(initial_bond_edit_idx)
                    elif new_state is not None:
                        mol_state = new_state
                        rdkit_to_internal_map = new_map
                        rl_action_sequence.extend(rl_seq)
                        applied_in_pass = True
                        # Mark edit(s) as processed
                        indices_processed_now = {edit_idx}
                        if op == 'insert_node' and initial_bond_edit_idx != -1:
                             indices_processed_now.add(initial_bond_edit_idx)
                             processed_initial_bond_indices.add(initial_bond_edit_idx)
                        processed_edit_indices.update(indices_processed_now)
                        deferred_edit_indices.difference_update(indices_processed_now) # Remove if was deferred
                    else:
                         # Failed permanently (e.g., MAX_RL_STEPS)
                         return None # Abort conversion for this pair

            except KeyError as e:
                 # This should be less frequent now with the pre-check, but handle defensively
                 print(f"  KeyError processing edit {edit_idx} ({op}): {e}. RDKit index likely missing from map unexpectedly. Deferring.")
                 deferred_edit_indices.add(edit_idx)
            except Exception as e:
                 print(f"  Unexpected error processing edit {edit_idx} ({op}): {e}")
                 traceback.print_exc(limit=5)
                 deferred_edit_indices.add(edit_idx) # Defer on unexpected error


    # --- End of Main Loop ---
    if total_rl_steps >= MAX_RL_STEPS:
        print(f"  Failed conversion due to exceeding MAX_RL_STEPS for {source_smiles} -> {target_smiles}")
        return None

    # Check if all edits were processed
    if len(processed_edit_indices) != len(all_edits):
        # --- START DEBUG LOG ---
        print(f"  Failed conversion: Not all edits processed for {source_smiles} -> {target_smiles}")
        print(f"  Processed: {len(processed_edit_indices)}, Total: {len(all_edits)}")
        unprocessed_indices = sorted([idx for idx, e in all_edits if idx not in processed_edit_indices])
        print(f"  Unprocessed indices: {unprocessed_indices}")
        print(f"  Final deferred indices in last pass: {sorted(list(deferred_edit_indices))}")
        print(f"  State when stuck:")
        try:
            print(f"    Current Atoms: {mol_state.atoms}")
            # print(f"    Current Bonds:\n{mol_state.bonds}") # Can be large, maybe skip
            print(f"    Current Map: {rdkit_to_internal_map}")
            print(f"    Remaining Valence: {mol_state._get_remaining_valence()}")
            print(f"    Is Connected: {mol_state.is_currently_connected}")
        except Exception as log_e:
            print(f"    Error logging state details: {log_e}")
        print(f"  First few unprocessed/deferred edits:")
        logged_count = 0
        indices_to_log = sorted(list(set(unprocessed_indices) | deferred_edit_indices)) # Combine and sort
        for idx in indices_to_log:
            if logged_count >= 10: break # Log more edits
            try:
                print(f"    - Index {idx}: {ged_path[idx]}")
                logged_count += 1
            except IndexError:
                 print(f"    - Index {idx}: Error retrieving edit info.")
        # --- END DEBUG LOG ---
        return None

    # 4. Finalization and Verification
    try:
        if mol_state.is_terminable():
            mask = mol_state.current_action_mask
            if mask is None or 0 >= len(mask) or mask[0]:
                 print(f"  Warning: Final terminate action (0) is masked for {target_smiles}.")
                 if not mol_state.is_currently_connected:
                      print("    Final state is disconnected. Marking as failure.")
                      return None
                 # If connected but masked, maybe allow it but log warning? Or fail? For now, fail.
                 print("    Final terminate masked despite connectivity. Marking as failure.")
                 return None
            else:
                 mol_state.take_action(0)
                 rl_action_sequence.append(0)
        else:
            print(f"  Warning: Final state not terminable for {target_smiles}. Connected: {mol_state.is_currently_connected}. Marking as failure.")
            return None

        mol_state.finalize(assert_feasible=False)
        final_smiles = mol_state.to_smiles()

        if final_smiles is None:
             print(f"  Failed conversion: Final state resulted in None SMILES for {target_smiles}")
             return None

        # Canonical comparison
        try:
            canon_final = Chem.CanonSmiles(final_smiles)
            canon_target = Chem.CanonSmiles(target_smiles)
        except Exception as smi_err:
             print(f"  Error canonicalizing SMILES during verification: {smi_err}")
             print(f"    Target: {target_smiles}")
             print(f"    Result: {final_smiles}")
             return None

        if canon_final != canon_target:
            print(f"  Failed conversion: Final SMILES mismatch for {target_smiles}")
            print(f"    Target:   {canon_target}")
            print(f"    Result:   {canon_final}")
            return None

    except Exception as e:
        print(f"  Error during finalization/verification for {target_smiles}: {e}")
        traceback.print_exc()
        return None

    return rl_action_sequence


def main():
    """Main function to convert GED datasets to RL action sequences."""

    print("Initializing MoleculeConfig...")
    try:
        # Ensure MoleculeConfig() does not require arguments or pass necessary ones
        config = MoleculeConfig()
        print("MoleculeConfig initialized successfully.")
        # Perform a basic check on the config if possible
        if not hasattr(config, 'atom_vocabulary') or not config.atom_vocabulary:
             print("Error: config.atom_vocabulary is missing or empty.")
             return
        print(f"Atom vocabulary size: {len(config.atom_vocabulary)}")
    except Exception as e:
        print(f"Error initializing MoleculeConfig: {e}")
        import traceback; traceback.print_exc()
        return

    print("Building reverse atom lookup...")
    try:
        reverse_atom_lookup = build_reverse_atom_lookup(config)
        print("Reverse atom lookup built successfully.")
        if not reverse_atom_lookup:
             print("Warning: Reverse atom lookup is empty.")
    except Exception as e:
        print(f"Error building reverse atom lookup: {e}")
        import traceback; traceback.print_exc()
        return

    datatypes = ["train", "valid"] # Or load dynamically

    for datatype in datatypes:
        print(f"\n{'='*50}")
        print(f"Processing {datatype.upper()} dataset for RL conversion")
        print(f"{'='*50}\n")

        transformation_data = load_latest_transformation_data(datatype)
        if not transformation_data:
            print(f"No transformation data found for {datatype}. Skipping.")
            continue

        print(f"Loaded {len(transformation_data)} transformations for {datatype}.")
        rl_dataset = []
        conversion_failures = 0
        start_time = time.time()

        # --- Optional: Filter for specific pairs for debugging ---
        # specific_source = "CC1=NN2C(C3=CC=CC(NC(=O)OC(C)C)=C3)=CC=NC2=C1C(=O)C1=CC=CS1"
        # specific_target = "COC1=CC=CC=C1C1(O)C(=O)NC2=CC=CC=C21"
        # transformation_data = [t for t in transformation_data if t['source_smiles'] == specific_source and t['target_smiles'] == specific_target]
        # print(f"Filtered to {len(transformation_data)} specific pairs for debugging.")
        # ---

        for i, transformation in enumerate(tqdm(transformation_data, desc=f"Converting {datatype}")):
            rl_sequence = convert_single_ged_to_rl(transformation, config, reverse_atom_lookup)

            if rl_sequence is not None:
                rl_dataset.append({
                    'source_smiles': transformation['source_smiles'],
                    'target_smiles': transformation['target_smiles'],
                    'rl_action_sequence': rl_sequence,
                    # 'original_ged_path': transformation['edit_path'] # Keep for reference if needed
                })
            else:
                conversion_failures += 1
                # Log failed pairs more visibly if needed
                # print(f"\n--- Conversion FAILED for pair {i} ---")
                # print(f"  Source: {transformation['source_smiles']}")
                # print(f"  Target: {transformation['target_smiles']}")
                # print(f"-------------------------------------\n")


        end_time = time.time()
        print(f"\nFinished converting {datatype} dataset.")
        print(f"  Successfully converted: {len(rl_dataset)}")
        print(f"  Failures: {conversion_failures}")
        print(f"  Total time: {end_time - start_time:.2f} seconds")

        # Save the RL dataset
        output_filename = os.path.join(OUTPUT_DIR, f"rl_action_dataset_{datatype}.pickle")
        with open(output_filename, "wb") as f:
            pickle.dump(rl_dataset, f)
        print(f"Saved RL action dataset for {datatype} to {output_filename}")

    print("\nAll datasets processed.")

if __name__ == "__main__":
    # Ensure MoleculeDesign and MoleculeConfig are correctly defined/imported
    # and MoleculeConfig can be instantiated.
    main()