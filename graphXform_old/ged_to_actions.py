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
# Assuming config.py and molecule_design.py are in the same directory or PYTHONPATH
from config import MoleculeConfig
from molecule_design import MoleculeDesign


# # Suppress RDKit warnings (optional)
# RDLogger.DisableLog('rdApp.*')

# Configuration
CHECKPOINT_DIR = "../data/chembl/checkpoints"
FINAL_DATA_DIR = "../data/chembl"
OUTPUT_DIR = "../data/chembl/rl_datasets"
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

def get_defined_valence(atomic_num: int, config: MoleculeConfig) -> int:
    """Gets the valence defined in the config for a standard atom type (ignoring charge/chirality)."""
    # Find the first entry matching the atomic number (e.g., "C" for 6, "N" for 7)
    for key, data in config.atom_vocabulary.items():
        # Check if 'atomic_number' exists and matches, and it's the base type (no charge/chirality suffix)
        if data.get('atomic_number') == atomic_num and \
           data.get('formal_charge', 0) == 0 and \
           data.get('chiral_tag', 0) == 0:
            try:
                return data['valence']
            except KeyError:
                print(f"Warning: Valence not defined for base atom number {atomic_num} in config.")
                return -1 # Indicate error or unknown
    print(f"Warning: Could not find base definition for atomic number {atomic_num} in config.")
    return -1 # Indicate error or unknown
# ---

# --- Helper: Find Redundant Delete Edges ---
def find_redundant_delete_edges(ged_path: List[Dict[str, Any]]) -> set[int]:
    """
    Identifies indices of delete_edge edits in a GED path that are redundant
    because one of the involved nodes was deleted by an earlier delete_node edit.
    """
    deleted_nodes_rdkit_indices = set()
    redundant_delete_edge_indices = set()
    for idx, edit_info in enumerate(ged_path):
        op = edit_info['operation']
        if op == 'delete_node':
            deleted_nodes_rdkit_indices.add(edit_info['source_idx'])
        elif op == 'delete_edge':
            atom1_idx = edit_info['atom1_idx']
            atom2_idx = edit_info['atom2_idx']
            # Check if either atom involved in the edge was deleted *before* this edit
            if atom1_idx in deleted_nodes_rdkit_indices or atom2_idx in deleted_nodes_rdkit_indices:
                redundant_delete_edge_indices.add(idx)
                print(f"DEBUG: Found redundant delete_edge at index {idx}: {edit_info} (Nodes deleted: {deleted_nodes_rdkit_indices})") # Optional debug
    return redundant_delete_edge_indices
# ---


def convert_single_ged_to_rl(
    transformation: Dict[str, Any],
    config: MoleculeConfig, # Pass the config object
    reverse_atom_lookup: Dict[Tuple[int, int, int], int]
) -> Optional[List[int]]:
    """
    Attempts to convert a single GED edit path to an RL action sequence,
    skipping redundant delete_edge edits and using refined heuristic ordering.
    """
    source_smiles = transformation['source_smiles']
    target_smiles = transformation['target_smiles']
    ged_path = transformation['edit_path'][:-1] # Exclude final state if present

    # *** ADD REDUNDANCY CHECK ***
    redundant_delete_indices = find_redundant_delete_edges(ged_path)
    if redundant_delete_indices:
        print(f"  INFO: Skipping {len(redundant_delete_indices)} redundant delete_edge edits for {source_smiles} -> {target_smiles}. Indices: {sorted(list(redundant_delete_indices))}")
    # *** END REDUNDANCY CHECK ***

    # 1. Initialization
    try:
        mol_state, rdkit_to_internal_map = MoleculeDesign.from_smiles(
            config, source_smiles, do_finish=False
        )
        if mol_state is None or rdkit_to_internal_map is None:
             print(f"  Failed initialization for {source_smiles}"); return None
    except Exception as e:
        print(f"  Error initializing MoleculeDesign for {source_smiles}: {e}"); traceback.print_exc(); return None

    rl_action_sequence = []
    all_edits_indexed = list(enumerate(ged_path))

    # Categorize Edits
    delete_edits = [(idx, e) for idx, e in all_edits_indexed if e['operation'].startswith('delete_')]
    substitute_edits = [(idx, e) for idx, e in all_edits_indexed if e['operation'].startswith('substitute_')]
    insert_node_edits = [(idx, e) for idx, e in all_edits_indexed if e['operation'] == 'insert_node']
    insert_edge_edits = [(idx, e) for idx, e in all_edits_indexed if e['operation'] == 'insert_edge']

    processed_edit_indices = set()
    deferred_edit_indices = set()
    processed_initial_bond_indices = set() # Track bonds processed as part of insert_node
    total_rl_steps = 0

    # --- Mapping helper ---
    def get_vocab_idx(element, charge, chiral_tag) -> Optional[int]:
        key = (element, charge, chiral_tag); idx = reverse_atom_lookup.get(key)
        # Fallback for chiral mismatch (e.g., query non-chiral, find chiral entry)
        if idx is None and chiral_tag != 0: idx = reverse_atom_lookup.get((element, charge, 0))
        # Fallback for chiral mismatch (e.g., query chiral, find non-chiral entry) - less common need?
        # if idx is None and chiral_tag == 0:
        #     # Try finding ANY chiral version if non-chiral query failed
        #     for ch in [1, 2]: # RDKit chiral tags
        #         idx = reverse_atom_lookup.get((element, charge, ch))
        #         if idx is not None: break
        return idx

    # --- apply_rl_sequence helper ---
    def apply_rl_sequence(edit_idx, sequence: List[int], current_state: MoleculeDesign, current_map: Dict[int, int]) -> \
            Tuple[Optional[MoleculeDesign], Optional[Dict[int, int]], bool]:
        nonlocal total_rl_steps
        # print(f"  DEBUG apply_rl_sequence: Entering for Edit {edit_idx}, Sequence: {sequence}") # Optional entry log
        temp_state = copy.deepcopy(current_state)
        if current_state.current_action_mask is not None:
            # Ensure mask is copied correctly
            temp_state.current_action_mask = np.copy(current_state.current_action_mask)
        else:
            temp_state.current_action_mask = None
        temp_map = copy.deepcopy(current_map)
        removed_internal_idx, added_rdkit_idx, new_internal_idx = None, None, None
        current_action_to_attempt, current_level = -1, -1
        try:
            for i, action_in_seq in enumerate(sequence):
                current_action_to_attempt, current_level = action_in_seq, temp_state.current_action_level
                # print(f"    DEBUG apply_rl_sequence: Step {i}, Action={current_action_to_attempt}, Level={current_level}") # Optional step log

                if total_rl_steps >= MAX_RL_STEPS:
                    print(f"    MAX_RL_STEPS ({MAX_RL_STEPS}) reached during apply_rl_sequence for edit {edit_idx}. Aborting.")
                    return None, None, False # Treat as permanent failure

                mask = temp_state.current_action_mask
                mask_value_at_action, is_out_of_bounds, mask_len_str = "N/A", False, "None"

                if mask is None: mask_value_at_action = "Mask is None"
                elif current_action_to_attempt < 0 or current_action_to_attempt >= len(mask): # Check bounds carefully
                     mask_value_at_action, is_out_of_bounds, mask_len_str = "Index OOB", True, str(len(mask))
                else: mask_value_at_action, mask_len_str = str(mask[current_action_to_attempt]), str(len(mask))

                is_masked = mask is None or is_out_of_bounds or mask[current_action_to_attempt]

                if is_masked:
                    print(f"    --> MASK FAILURE for Edit {edit_idx}: Action={current_action_to_attempt}, Level={current_level}. MaskValue={mask_value_at_action}, MaskLen={mask_len_str}. Deferring.")
                    # Add context if L2 bond path failed
                    if current_level == 2 and not temp_state.is_modifying_atom:
                         idx_A = temp_state.l0_selected_atom_idx
                         idx_B = -1
                         if temp_state.l1_new_atom_type is not None: idx_B = len(temp_state.atoms) - 1
                         elif temp_state.l1_selected_existing_atom_idx is not None: idx_B = temp_state.l1_selected_existing_atom_idx
                         bond_order_at_fail = "N/A"
                         if idx_A is not None and idx_B != -1 and 0 < idx_A < len(temp_state.bonds) and 0 < idx_B < len(temp_state.bonds):
                              bond_order_at_fail = temp_state.bonds[idx_A, idx_B]
                         print(f"        L2 Bond Path Context: A={idx_A}, B={idx_B}, Action={current_action_to_attempt} (0-5:SetOrder, 6:Remove), BondOrder={bond_order_at_fail}")
                    return None, None, True # Defer

                # *** ADD L2 BOND PATH PRE-ACTION LOGGING ***
                if current_level == 2 and not temp_state.is_modifying_atom:
                    idx_A_pre = temp_state.l0_selected_atom_idx
                    idx_B_pre = -1
                    if temp_state.l1_new_atom_type is not None: idx_B_pre = len(temp_state.atoms) - 1
                    elif temp_state.l1_selected_existing_atom_idx is not None: idx_B_pre = temp_state.l1_selected_existing_atom_idx

                    bond_order_pre_action = "N/A"
                    if idx_A_pre is not None and idx_B_pre != -1 and \
                       0 < idx_A_pre < len(temp_state.bonds) and \
                       0 < idx_B_pre < len(temp_state.bonds):
                           try:
                               bond_order_pre_action = temp_state.bonds[idx_A_pre, idx_B_pre]
                           except IndexError:
                               bond_order_pre_action = "IndexError accessing bonds"

                    print(f"    DEBUG apply_rl_sequence: PRE L2 Bond Action {current_action_to_attempt} for Edit {edit_idx}")
                    print(f"        Indices: A={idx_A_pre}, B={idx_B_pre}")
                    print(f"        Bond Order in temp_state BEFORE action: {bond_order_pre_action}")
                    print(f"        Action Meaning: { 'Set Order ' + str(current_action_to_attempt + 1) if current_action_to_attempt <= 5 else 'Remove Bond' if current_action_to_attempt == 6 else 'INVALID' }")
                # *** END L2 BOND PATH PRE-ACTION LOGGING ***


                # Track internal index if atom is removed (BEFORE action might change indices)
                removed_idx_internal_if_action_taken = -1
                if current_level == 2 and temp_state.is_modifying_atom and current_action_to_attempt == temp_state.REMOVE_ATOM_ACTION_L2_MODIFY:
                    removed_idx_internal_if_action_taken = temp_state.atom_to_modify # Store the index that *would* be removed

                # Track RDKit index if atom is inserted
                added_rdkit_idx_if_action_taken = None
                if current_level == 1 and current_action_to_attempt < temp_state.vocab_size:
                    try:
                        original_edit = ged_path[edit_idx]
                        if original_edit['operation'] == 'insert_node':
                             added_rdkit_idx_if_action_taken = original_edit['target_idx']
                    except IndexError:
                        print(f"Warning: Could not access ged_path[{edit_idx}] in apply_rl_sequence.")

                # --- Execute Action ---
                # print(f"      Calling temp_state.take_action({current_action_to_attempt})") # Optional call log
                temp_state.take_action(current_action_to_attempt)
                total_rl_steps += 1
                # print(f"      Returned from temp_state.take_action({current_action_to_attempt})") # Optional return log
                # --- End Execute ---

                # Check if take_action resulted in an infeasible state (error during execution)
                if temp_state.infeasibility_flag:
                     print(f"    --> Infeasibility flag set during take_action for Edit {edit_idx}, Action {current_action_to_attempt}. Deferring.")
                     return None, None, True # Defer if take_action failed internally

                # --- Post-Action Index Tracking ---
                # If atom was removed, set removed_internal_idx based on stored value
                if removed_idx_internal_if_action_taken != -1:
                    removed_internal_idx = removed_idx_internal_if_action_taken

                # If atom was inserted, set added_rdkit_idx and find new internal index
                if added_rdkit_idx_if_action_taken is not None:
                    added_rdkit_idx = added_rdkit_idx_if_action_taken
                    # This assumes take_action correctly appends the new atom
                    new_internal_idx = len(temp_state.atoms) - 1
                # --- End Post-Action Index Tracking ---


            # --- Update Map After FULL Sequence ---
            if removed_internal_idx is not None:
                # print(f"    DEBUG apply_rl_sequence: Updating map after removing internal index {removed_internal_idx}") # Optional map log
                removed_rdkit_idx = None
                # Find RDKit index corresponding to the removed internal index
                for r_idx, i_idx in temp_map.items():
                    if i_idx == removed_internal_idx:
                        removed_rdkit_idx = r_idx; break
                if removed_rdkit_idx is not None:
                    # print(f"      Removing RDKit index {removed_rdkit_idx} from map.") # Optional map log
                    del temp_map[removed_rdkit_idx] # Remove from map
                else:
                    print(f"      Warning: Could not find RDKit index for removed internal index {removed_internal_idx} in map.")

                # Decrement internal indices in map that were greater than the removed one
                keys_to_update = [r_idx for r_idx, i_idx in temp_map.items() if i_idx > removed_internal_idx]
                # print(f"      Decrementing internal indices in map for RDKit keys: {keys_to_update}") # Optional map log
                for r_idx in keys_to_update:
                    temp_map[r_idx] -= 1

            # Add mapping for newly inserted atom
            if added_rdkit_idx is not None and new_internal_idx is not None:
                # print(f"    DEBUG apply_rl_sequence: Adding map entry: RDKit {added_rdkit_idx} -> Internal {new_internal_idx}") # Optional map log
                temp_map[added_rdkit_idx] = new_internal_idx
            # --- End Update Map ---

            # print(f"  DEBUG apply_rl_sequence: SUCCESS for Edit {edit_idx}") # Optional success log
            return temp_state, temp_map, False # Success

        except Exception as e:
            print(f"    --> UNCAUGHT EXCEPTION in apply_rl_sequence for Edit {edit_idx}: Action={current_action_to_attempt}, Level={current_level}. Error: {e}. Deferring.")
            traceback.print_exc(limit=5) # More traceback
            return None, None, True # Defer on error


    # --- Main Simulation Loop ---
    applied_in_pass = True
    while applied_in_pass:
        if total_rl_steps >= MAX_RL_STEPS:
            print(f"  MAX_RL_STEPS ({MAX_RL_STEPS}) reached in main loop. Aborting conversion.")
            break
        applied_in_pass = False
        edits_to_retry_from_deferral = list(deferred_edit_indices)
        deferred_edit_indices.clear()
        edits_to_retry_from_deferral_set = set(edits_to_retry_from_deferral)

        # --- Define order for processing IN THIS PASS ---
        edit_indices_this_pass = []

        # Phase 1: Deletions
        edit_indices_this_pass.extend([
            idx for idx, e in delete_edits
            if idx not in processed_edit_indices and idx not in edits_to_retry_from_deferral_set
        ])

        # Phase 2: Substitutions (Refined Order using config valence)
        current_substitutions = [
            (idx, e) for idx, e in substitute_edits
            if idx not in processed_edit_indices and idx not in edits_to_retry_from_deferral_set
        ]
        bond_subs = [(idx, e) for idx, e in current_substitutions if e['operation'] == 'substitute_edge']
        atom_subs = [(idx, e) for idx, e in current_substitutions if e['operation'] == 'substitute_node']

        # 2a: Bond reductions (Largest reduction first)
        bond_reductions = sorted(
            [(idx, e) for idx, e in bond_subs if BOND_TYPE_TO_RL_ORDER.get(e['to_bond_type'], 99) < BOND_TYPE_TO_RL_ORDER.get(e['from_bond_type'], 0)],
            key=lambda item: BOND_TYPE_TO_RL_ORDER.get(item[1]['from_bond_type'], 0) - BOND_TYPE_TO_RL_ORDER.get(item[1]['to_bond_type'], 99),
            reverse=True
        )
        edit_indices_this_pass.extend([idx for idx, e in bond_reductions])

        # 2b: Atom substitutions - Valence Reduction (using config valence)
        atom_valence_reductions = []
        atom_other = []
        for idx, e in atom_subs:
            from_valence = get_defined_valence(e['from_element'], config)
            to_valence = get_defined_valence(e['to_element'], config)
            if from_valence > 0 and to_valence > 0 and to_valence < from_valence:
                atom_valence_reductions.append(((idx, e), from_valence - to_valence))
            else:
                atom_other.append((idx, e))
        atom_valence_reductions.sort(key=lambda item: item[1], reverse=True) # Largest reduction first
        edit_indices_this_pass.extend([idx for (idx, e), diff in atom_valence_reductions])

        # 2c: Other Atom substitutions
        edit_indices_this_pass.extend([idx for idx, e in atom_other])

        # 2d: Bond increases (Smallest increase first)
        bond_increases = sorted(
            [(idx, e) for idx, e in bond_subs if BOND_TYPE_TO_RL_ORDER.get(e['to_bond_type'], 0) > BOND_TYPE_TO_RL_ORDER.get(e['from_bond_type'], 99)],
             key=lambda item: BOND_TYPE_TO_RL_ORDER.get(item[1]['to_bond_type'], 0) - BOND_TYPE_TO_RL_ORDER.get(item[1]['from_bond_type'], 99)
        )
        edit_indices_this_pass.extend([idx for idx, e in bond_increases])

        # Phase 3: Insertions
        edit_indices_this_pass.extend([
            idx for idx, e in insert_node_edits
            if idx not in processed_edit_indices and idx not in edits_to_retry_from_deferral_set
        ])
        edit_indices_this_pass.extend([
            idx for idx, e in insert_edge_edits
            if idx not in processed_edit_indices and idx not in processed_initial_bond_indices and idx not in edits_to_retry_from_deferral_set
        ])

        # Add previously deferred edits at the end
        edit_indices_this_pass.extend([
            idx for idx in edits_to_retry_from_deferral if idx not in processed_edit_indices
        ])
        # --- End Defining Order ---

        if not edit_indices_this_pass: break # No edits left to try in this pass

        # --- Process Edits for This Pass ---
        for edit_idx in edit_indices_this_pass:
            # *** ADD SKIP FOR REDUNDANT DELETE_EDGE ***
            if edit_idx in redundant_delete_indices:
                if edit_idx not in processed_edit_indices: # Only process/log once
                    # print(f"    Skipping redundant delete_edge edit {edit_idx}") # Optional debug
                    processed_edit_indices.add(edit_idx) # Mark as processed so we don't retry
                    applied_in_pass = True # Ensure loop continues if only redundant edits were skipped
                continue
            # *** END SKIP ***

            # Skip if already processed in this pass (e.g., initial bond for insert_node)
            if edit_idx in processed_edit_indices: continue
            # Check step limit again before processing
            if total_rl_steps >= MAX_RL_STEPS: break

            edit_info = ged_path[edit_idx]; op = edit_info['operation']
            rl_seq = []; initial_bond_edit_idx = -1 # For tracking paired insert_node bond

            # --- Pre-check: Involved Atoms Still Exist? ---
            involved_rdkit_indices = []
            # Collect all potentially relevant RDKit indices from the edit info
            if 'atom1_idx' in edit_info: involved_rdkit_indices.extend([edit_info['atom1_idx'], edit_info['atom2_idx']])
            if 'source_atom1' in edit_info: involved_rdkit_indices.extend([edit_info['source_atom1'], edit_info['source_atom2']])
            if 'target_atom1' in edit_info: involved_rdkit_indices.extend([edit_info['target_atom1'], edit_info['target_atom2']]) # Should not happen with GED?
            if 'source_idx' in edit_info: involved_rdkit_indices.append(edit_info['source_idx'])
            # For insert_node, we only need the *anchor* atom to exist initially
            if op == 'insert_node':
                 # Find potential anchor atom for insert_node
                 target_r_idx = edit_info['target_idx']
                 anchor_r_idx = -1
                 for edge_i, edge_edit in insert_edge_edits:
                     if edge_i in processed_edit_indices or edge_i in processed_initial_bond_indices: continue
                     if target_r_idx in (edge_edit['atom1_idx'], edge_edit['atom2_idx']):
                         potential_anchor = edge_edit['atom1_idx'] if edge_edit['atom2_idx'] == target_r_idx else edge_edit['atom2_idx']
                         if potential_anchor in rdkit_to_internal_map: # Check if anchor exists
                             anchor_r_idx = potential_anchor; break
                 if anchor_r_idx == -1:
                      involved_rdkit_indices = [] # No valid anchor found yet, don't check existence
                 else:
                      involved_rdkit_indices = [anchor_r_idx] # Only check anchor existence

            atom_deleted = False
            missing_rdkit_indices = []
            for r_idx in set(involved_rdkit_indices):
                if r_idx not in rdkit_to_internal_map:
                    atom_deleted = True
                    missing_rdkit_indices.append(r_idx)

            if atom_deleted:
                print(f"  Skipping edit {edit_idx} ({op}): Involved RDKit atom(s) {missing_rdkit_indices} no longer exist in map.")
                processed_edit_indices.add(edit_idx) # Mark as processed
                applied_in_pass = True # Ensure loop continues
                # If node insertion failed because anchor deleted, try to mark paired bond too? Less critical.
                continue
            # --- End Pre-check ---


            # Try block for generating and applying RL sequence
            try:
                # --- Determine RL sequence based on operation ---
                if op == 'delete_edge':
                    r_idx1, r_idx2 = edit_info['atom1_idx'], edit_info['atom2_idx']
                    # *** ADDED PRE-CHECK LOG ***
                    internal_idx1 = rdkit_to_internal_map.get(r_idx1, -1)
                    internal_idx2 = rdkit_to_internal_map.get(r_idx2, -1)
                    bond_order_before_attempt = -99
                    if internal_idx1 > 0 and internal_idx2 > 0 and internal_idx1 < len(mol_state.atoms) and internal_idx2 < len(mol_state.atoms):
                         try:
                             bond_order_before_attempt = mol_state.bonds[internal_idx1, internal_idx2]
                         except IndexError:
                              bond_order_before_attempt = -98 # Index error accessing bonds
                    print(f"  Attempting delete_edge Edit {edit_idx}: RDKit({r_idx1}, {r_idx2}) -> Internal({internal_idx1}, {internal_idx2}). Bond order in current state BEFORE apply_rl_sequence: {bond_order_before_attempt}")
                    # *** END PRE-CHECK LOG ***

                    i_idx1, i_idx2 = rdkit_to_internal_map[r_idx1], rdkit_to_internal_map[r_idx2] # Indices guaranteed by pre-check
                    l1_select_action = mol_state.vocab_size + (i_idx2 - 1) # i_idx2 is 1-based, need 0-based offset
                    rl_seq = [i_idx1, l1_select_action, REMOVE_BOND_RL_ACTION]

                elif op == 'delete_node':
                    r_idx = edit_info['source_idx']
                    i_idx = rdkit_to_internal_map[r_idx]
                    l1_modify_action = mol_state.vocab_size + (len(mol_state.atoms) - 1) # Action to initiate modify on *last* possible atom index (placeholder)
                    rl_seq = [i_idx, l1_modify_action, mol_state.REMOVE_ATOM_ACTION_L2_MODIFY]

                elif op == 'substitute_edge':
                    r_idx1, r_idx2 = edit_info['source_atom1'], edit_info['source_atom2']
                    i_idx1, i_idx2 = rdkit_to_internal_map[r_idx1], rdkit_to_internal_map[r_idx2]
                    new_bond_type = edit_info['to_bond_type']
                    new_order = BOND_TYPE_TO_RL_ORDER.get(new_bond_type)
                    if new_order is None:
                        print(f"  Skipping edit {edit_idx}: Invalid bond type {new_bond_type} for substitution."); continue
                    l1_select_action = mol_state.vocab_size + (i_idx2 - 1)
                    l2_action = new_order - 1 # RL action is 0-based for orders 1-6
                    rl_seq = [i_idx1, l1_select_action, l2_action]

                elif op == 'substitute_node':
                    r_idx = edit_info['source_idx']
                    i_idx = rdkit_to_internal_map[r_idx]
                    new_vocab_idx = get_vocab_idx(edit_info['to_element'], edit_info['to_charge'], edit_info['to_chiral'])
                    if new_vocab_idx is None:
                        print(f"  Skipping edit {edit_idx}: Cannot find vocab index for substitution target {edit_info['to_element']}/{edit_info['to_charge']}/{edit_info['to_chiral']}."); continue
                    l1_modify_action = mol_state.vocab_size + (len(mol_state.atoms) - 1) # Placeholder for modify
                    l2_action = new_vocab_idx - 1 # RL action is 0-based for vocab indices 1+
                    rl_seq = [i_idx, l1_modify_action, l2_action]

                elif op == 'insert_node':
                    target_r_idx = edit_info['target_idx']
                    initial_bond_edit_info = None; anchor_r_idx = -1
                    # Find the first available insert_edge involving this new node and an existing anchor
                    for edge_i, edge_edit in insert_edge_edits:
                         if edge_i in processed_edit_indices or edge_i in processed_initial_bond_indices: continue
                         # Determine potential anchor (the atom already in the molecule)
                         if edge_edit['atom1_idx'] == target_r_idx and edge_edit['atom2_idx'] in rdkit_to_internal_map:
                             anchor_r_idx = edge_edit['atom2_idx']
                         elif edge_edit['atom2_idx'] == target_r_idx and edge_edit['atom1_idx'] in rdkit_to_internal_map:
                             anchor_r_idx = edge_edit['atom1_idx']
                         else: continue # This edge doesn't connect the new node to an existing one

                         # Found a suitable initial bond
                         initial_bond_edit_idx = edge_i
                         initial_bond_edit_info = edge_edit
                         break # Use the first valid one found

                    if initial_bond_edit_info is None:
                         print(f"  Deferring insert_node edit {edit_idx}: Cannot find valid initial bond connection to existing atom yet.")
                         deferred_edit_indices.add(edit_idx); continue

                    anchor_i_idx = rdkit_to_internal_map[anchor_r_idx]
                    new_vocab_idx = get_vocab_idx(edit_info['element'], edit_info['charge'], edit_info['chiral_tag'])
                    bond_order = BOND_TYPE_TO_RL_ORDER.get(initial_bond_edit_info['bond_type'])

                    if new_vocab_idx is None or bond_order is None:
                        print(f"  Skipping edit {edit_idx}: Invalid vocab index or bond type for insert_node. Vocab: {new_vocab_idx}, Bond: {bond_order}"); continue

                    l1_action_add = new_vocab_idx - 1 # RL action is 0-based
                    l2_action_bond = bond_order - 1 # RL action is 0-based
                    rl_seq = [anchor_i_idx, l1_action_add, l2_action_bond]

                elif op == 'insert_edge':
                    # Skip if this edge was already processed as part of an insert_node
                    if edit_idx in processed_initial_bond_indices: continue

                    r_idx1, r_idx2 = edit_info['atom1_idx'], edit_info['atom2_idx']
                    # Both atoms must exist (guaranteed by pre-check)
                    i_idx1, i_idx2 = rdkit_to_internal_map[r_idx1], rdkit_to_internal_map[r_idx2]
                    bond_order = BOND_TYPE_TO_RL_ORDER.get(edit_info['bond_type'])
                    if bond_order is None:
                        print(f"  Skipping edit {edit_idx}: Invalid bond type {edit_info['bond_type']} for insert_edge"); continue

                    l1_select_action = mol_state.vocab_size + (i_idx2 - 1)
                    l2_action = bond_order - 1
                    rl_seq = [i_idx1, l1_select_action, l2_action]

                # --- Apply RL sequence if generated ---
                if rl_seq:
                    new_state, new_map, needs_deferral = apply_rl_sequence(edit_idx, rl_seq, mol_state, rdkit_to_internal_map)

                    if needs_deferral:
                        if edit_idx not in deferred_edit_indices: # Avoid duplicate logging if retried
                             print(f"  DEFERRING Edit {edit_idx}: {edit_info}")
                             deferred_edit_indices.add(edit_idx)
                             # If insert_node is deferred, also defer its paired bond if not already processed/deferred
                             if op == 'insert_node' and initial_bond_edit_idx != -1:
                                  if initial_bond_edit_idx not in processed_edit_indices and initial_bond_edit_idx not in deferred_edit_indices:
                                       deferred_edit_indices.add(initial_bond_edit_idx)
                    elif new_state is not None and new_map is not None:
                        # --- Success ---
                        mol_state = new_state
                        rdkit_to_internal_map = new_map
                        rl_action_sequence.extend(rl_seq)
                        applied_in_pass = True # Mark that progress was made in this pass

                        # Mark edit(s) as processed
                        indices_processed_now = {edit_idx}
                        if op == 'insert_node' and initial_bond_edit_idx != -1:
                             indices_processed_now.add(initial_bond_edit_idx)
                             processed_initial_bond_indices.add(initial_bond_edit_idx) # Mark bond as handled

                        processed_edit_indices.update(indices_processed_now)
                        # Remove from deferred set if it was successfully processed now
                        deferred_edit_indices.difference_update(indices_processed_now)
                        # --- End Success ---
                    else:
                         # Permanent failure from apply_rl_sequence
                         print(f"  Permanent failure applying sequence for edit {edit_idx}. Aborting conversion for this pair.")
                         return None # Abort conversion for this source/target pair

            # --- Exception handling for RL sequence generation/application ---
            except KeyError as e:
                 # Likely due to rdkit_to_internal_map lookup failing unexpectedly
                 print(f"  KeyError processing edit {edit_idx} ({op}): {e}. Likely atom deleted unexpectedly. Deferring.")
                 if edit_idx not in deferred_edit_indices: deferred_edit_indices.add(edit_idx)
            except Exception as e:
                 print(f"  Unexpected error processing edit {edit_idx} ({op}): {e}")
                 traceback.print_exc(limit=5)
                 if edit_idx not in deferred_edit_indices: deferred_edit_indices.add(edit_idx) # Defer on unexpected error
            # --- End Try/Except ---
        # --- End Loop Over Edits in Pass ---

        # Safety break: If no edits were applied and none are deferred, exit.
        if not applied_in_pass and not deferred_edit_indices:
             # Check if all edits were processed (including skipped redundant ones)
             if len(processed_edit_indices) != len(all_edits_indexed):
                  unprocessed_indices = [idx for idx, _ in all_edits_indexed if idx not in processed_edit_indices]
                  print(f"  Warning: Loop finished but unprocessed edits remain: {unprocessed_indices}. Likely stuck.")
             break # Exit main while loop


    # --- Final Checks and Verification ---
    if total_rl_steps >= MAX_RL_STEPS:
        print(f"  Failed conversion: Exceeded MAX_RL_STEPS ({MAX_RL_STEPS}) for {source_smiles} -> {target_smiles}")
        return None

    if len(processed_edit_indices) != len(all_edits_indexed):
        unprocessed_indices = [idx for idx, _ in all_edits_indexed if idx not in processed_edit_indices]
        deferred_count = len(deferred_edit_indices.intersection(unprocessed_indices))
        other_unprocessed_count = len(unprocessed_indices) - deferred_count
        print(f"  Failed conversion: Not all edits processed for {source_smiles} -> {target_smiles}. Total: {len(all_edits_indexed)}, Processed: {len(processed_edit_indices)}, Unprocessed: {len(unprocessed_indices)} ({deferred_count} deferred, {other_unprocessed_count} other).")
        # print(f"    Unprocessed Indices: {sorted(unprocessed_indices)}") # Optional details
        # print(f"    Deferred Indices: {sorted(list(deferred_edit_indices))}") # Optional details
        return None

    # Attempt to terminate the molecule state
    try:
        if mol_state.is_terminable():
            mask = mol_state.current_action_mask
            terminate_action = 0 # Action index for termination
            if mask is None or terminate_action >= len(mask) or mask[terminate_action]:
                 # Termination is masked - check connectivity
                 mol_state._check_and_update_connectivity() # Ensure connectivity is up-to-date
                 if not mol_state.is_currently_connected:
                     print(f"  Failed conversion: Final state terminable but disconnected for {source_smiles} -> {target_smiles}")
                     return None
                 else:
                     print(f"  Failed conversion: Final state terminable but termination action masked (unexpected?) for {source_smiles} -> {target_smiles}")
                     return None
            else:
                # Apply termination action
                mol_state.take_action(terminate_action)
                rl_action_sequence.append(terminate_action)
                total_rl_steps += 1
        else:
            # Not terminable - check connectivity as a potential reason
            mol_state._check_and_update_connectivity()
            if not mol_state.is_currently_connected:
                 print(f"  Failed conversion: Final state not terminable (disconnected) for {source_smiles} -> {target_smiles}")
                 return None
            else:
                 print(f"  Failed conversion: Final state not terminable (connected?) for {source_smiles} -> {target_smiles}")
                 return None # Not terminable is a failure

        # Finalize and verify SMILES match
        mol_state.finalize(assert_feasible=False) # Don't assert feasibility, just get SMILES
        final_smiles = mol_state.to_smiles()

        if final_smiles is None:
            print(f"  Failed conversion: Final state produced None SMILES for {source_smiles} -> {target_smiles}")
            return None

        # Canonicalize and compare
        canon_final = Chem.CanonSmiles(final_smiles)
        canon_target = Chem.CanonSmiles(target_smiles)

        if canon_final != canon_target:
            print(f"  Failed conversion: Final SMILES mismatch for {source_smiles} -> {target_smiles}")
            print(f"    Target:  {canon_target}")
            print(f"    Result:  {canon_final}")
            # print(f"    RL Seq: {rl_action_sequence}") # Optional debug
            return None

    except Exception as e:
        print(f"  Error during finalization/verification for {source_smiles} -> {target_smiles}: {e}")
        traceback.print_exc(limit=5)
        return None

    # Success!
    # print(f"  Successfully converted {source_smiles} -> {target_smiles} ({total_rl_steps} RL steps)") # Optional success log
    return rl_action_sequence


# --- Main Execution ---
def main():
    """Main function to load data and run conversion."""
    start_time = time.time()

    # --- Configuration and Setup ---
    config = MoleculeConfig() # Assuming MoleculeConfig() initializes correctly
    reverse_atom_lookup = build_reverse_atom_lookup(config)
    if not reverse_atom_lookup:
        print("Error: Reverse atom lookup failed. Cannot proceed.")
        return
    # --- End Setup ---

    total_processed = 0
    total_successful = 0
    rl_datasets = {'train': [], 'validation': [], 'test': []}

    for datatype in ['train', 'validation', 'test']:
        print(f"\n--- Processing {datatype} data ---")
        transformation_data = load_latest_transformation_data(datatype)
        if not transformation_data:
            print(f"No transformation data found for {datatype}. Skipping.")
            continue

        successful_conversions = []
        failed_indices = []

        for i, transformation in enumerate(tqdm(transformation_data, desc=f"Converting {datatype}")):
            total_processed += 1
            rl_sequence = convert_single_ged_to_rl(transformation, config, reverse_atom_lookup)

            if rl_sequence is not None:
                successful_conversions.append({
                    'source_smiles': transformation['source_smiles'],
                    'target_smiles': transformation['target_smiles'],
                    'rl_actions': rl_sequence
                })
                total_successful += 1
            else:
                failed_indices.append(i) # Log index of failed transformation

        rl_datasets[datatype] = successful_conversions
        success_rate = (len(successful_conversions) / len(transformation_data) * 100) if transformation_data else 0
        print(f"Finished {datatype}: {len(successful_conversions)} / {len(transformation_data)} successful ({success_rate:.2f}%)")
        if failed_indices:
             print(f"  Failed indices (first 10): {failed_indices[:10]}") # Show some failed indices

        # Save the successful RL sequences for this datatype
        output_path = os.path.join(OUTPUT_DIR, f"rl_dataset_{datatype}.pickle")
        try:
            with open(output_path, "wb") as f:
                pickle.dump(successful_conversions, f)
            print(f"Saved successful {datatype} RL dataset to: {output_path}")
        except Exception as e:
            print(f"Error saving RL dataset for {datatype} to {output_path}: {e}")

    # --- Final Summary ---
    end_time = time.time()
    total_time = end_time - start_time
    overall_success_rate = (total_successful / total_processed * 100) if total_processed else 0
    print("\n--- Conversion Summary ---")
    print(f"Total transformations processed: {total_processed}")
    print(f"Total successful conversions:  {total_successful}")
    print(f"Overall success rate:          {overall_success_rate:.2f}%")
    print(f"Total execution time:          {total_time:.2f} seconds")
    print("--------------------------")

if __name__ == "__main__":
    main()