# -*- coding: utf-8 -*-
import time
import pickle
import os
import random
import numpy as np
import networkx as nx
from rdkit import Chem, RDLogger
from rdkit.Chem import rdmolfiles, rdmolops
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional, Set, Any
from datetime import datetime
import copy

# Assuming molecule_design.py and config.py are accessible
from molecule_design import MoleculeDesign
from config import MoleculeConfig

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# --- Configuration ---
CHECKPOINT_DIR = "./data/chembl/checkpoints" # Directory for GED checkpoints
OUTPUT_DIR = "./data/chembl/rl_sequences"   # Directory for RL sequence output
MAX_CONVERSION_STEPS = 500 # Safety limit for RL steps per pair

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helper Functions (load_latest_checkpoint, mol_to_nx_graph - unchanged) ---
def load_latest_checkpoint(datatype: str) -> List[Dict]:
    """Loads the most recent transformation data checkpoint for the given datatype."""
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith(f"transformation_data_{datatype}_") and f.endswith(".pkl")]
    if not checkpoint_files:
        print(f"No transformation checkpoints found for {datatype} in {CHECKPOINT_DIR}.")
        return []
    try:
        checkpoint_files.sort(key=lambda f: f.split('_')[-1].replace('.pkl', ''), reverse=True)
    except Exception as e:
        print(f"Warning: Could not sort checkpoint files reliably by timestamp: {e}. Using lexicographical sort.")
        checkpoint_files.sort(reverse=True) # Fallback sort
    latest_checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_files[0])
    print(f"Loading latest {datatype} transformation checkpoint: {latest_checkpoint_path}")
    try:
        with open(latest_checkpoint_path, "rb") as f:
            transformation_data = pickle.load(f)
        print(f"Loaded {len(transformation_data)} transformations from checkpoint.")
        return transformation_data
    except Exception as e:
        print(f"Error loading checkpoint {latest_checkpoint_path}: {e}")
        return []

def mol_to_nx_graph(mol: Chem.Mol) -> nx.Graph:
    """Converts RDKit Mol to NetworkX Graph with essential properties."""
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_props = {
            'atomic_num': atom.GetAtomicNum(),
            'formal_charge': atom.GetFormalCharge(),
            'chiral_tag': int(atom.GetChiralTag())
        }
        graph.add_node(atom_idx, **atom_props)
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        bond_props = {
            'bond_type': int(bond.GetBondTypeAsDouble() * 10)
        }
        graph.add_edge(begin_idx, end_idx, **bond_props)
    return graph

# --- GedToRlConverter Class ---

class GedToRlConverter:
    """
    Converts abstract GED edit paths into concrete, sequential RL actions
    for the MoleculeDesign environment, respecting validity constraints.
    """
    # __init__ and _initialize_mappings remain the same
    def __init__(self, config: MoleculeConfig):
        self.config = config
        self.vocab_size = len(config.atom_vocabulary)
        self.rdkit_bond_type_to_rl_action = {
            int(Chem.BondType.SINGLE * 10): 0,
            int(Chem.BondType.DOUBLE * 10): 1,
            int(Chem.BondType.TRIPLE * 10): 2,
            int(Chem.BondType.QUADRUPLE * 10): 3,
            int(Chem.BondType.QUINTUPLE * 10): 4,
            int(Chem.BondType.HEXTUPLE * 10): 5,
        }
        self.atomic_num_to_vocab_idx0 = {
            data['atomic_number']: i
            for i, data in enumerate(config.atom_vocabulary.values())
        }
        self.vocab_idx1_to_idx0 = {i+1: i for i in range(self.vocab_size)}

    def _initialize_mappings(self, design: MoleculeDesign, edit_path: List[Dict]) -> Tuple[
        Dict[int, int], Dict[int, int], Dict[int, int]]:
        """Initialize index mappings based on source molecule's internal state and GED node path."""
        source_idx_to_internal_idx: Dict[int, int] = {}
        internal_idx_to_source_idx: Dict[int, int] = {}
        target_idx_to_internal_idx: Dict[int, int] = {}

        # --- MODIFIED LINE ---
        # Get atom count from the internal state (excluding virtual atom 0)
        num_source_atoms = len(design.atoms) - 1
        # --- END MODIFICATION ---

        # RDKit indices in the original source SMILES are implicitly 0 to num_source_atoms - 1
        # Internal indices are 1 to num_source_atoms
        for i in range(num_source_atoms):
            internal_idx = i + 1
            # Map original RDKit index 'i' to internal index 'internal_idx'
            source_idx_to_internal_idx[i] = internal_idx
            # Map internal index 'internal_idx' back to original RDKit index 'i'
            internal_idx_to_source_idx[internal_idx] = i

        # --- Rest of the method remains the same ---
        node_mappings = []
        present_source_indices_in_ops = set()
        source_indices_in_substitute = set()

        for op in edit_path:
            op_type = op['operation']
            if op_type == 'substitute_node':
                # op['source_idx'] refers to the original RDKit index
                node_mappings.append((op['source_idx'], op['target_idx']))
                present_source_indices_in_ops.add(op['source_idx'])
                source_indices_in_substitute.add(op['source_idx'])
            elif op_type == 'delete_node':
                present_source_indices_in_ops.add(op['source_idx'])
            elif op_type == 'insert_node':
                pass

        for source_idx in range(num_source_atoms):
            if source_idx not in present_source_indices_in_ops:
                is_in_edge = any(op.get('atom1_idx') == source_idx or op.get('atom2_idx') == source_idx or
                                 op.get('source_atom1') == source_idx or op.get('source_atom2') == source_idx
                                 for op in edit_path if 'atom1_idx' in op or 'source_atom1' in op)
                if is_in_edge:
                    if not any(u == source_idx for u, v in node_mappings):
                        node_mappings.append((source_idx, source_idx))

        for u, v in node_mappings:
            if u is not None and v is not None:
                # Use the source_idx_to_internal_idx map we created
                if u in source_idx_to_internal_idx:
                    target_idx_to_internal_idx[v] = source_idx_to_internal_idx[u]

        return source_idx_to_internal_idx, internal_idx_to_source_idx, target_idx_to_internal_idx

    # _categorize_changes, _check_target_reached, _update_mappings_after_removal,
    # _update_mappings_after_addition remain the same
    def _categorize_changes(self, edit_path: List[Dict]) -> Dict[str, Any]:
        """Parse GED path into structured sets/dicts of required changes."""
        changes: Dict[str, Any] = {
            "nodes_to_delete": set(),
            "nodes_to_insert": {},
            "nodes_to_substitute": {},
            "edges_to_delete": set(),
            "edges_to_insert": {},
            "edges_to_substitute": {},
        }
        for op in edit_path:
            op_type = op['operation']
            if op_type == 'delete_node':
                changes["nodes_to_delete"].add(op['source_idx'])
            elif op_type == 'insert_node':
                changes["nodes_to_insert"][op['target_idx']] = {
                    'element': op['element'], 'charge': op['charge'], 'chiral': op['chiral_tag']
                }
            elif op_type == 'substitute_node':
                changes["nodes_to_substitute"][op['source_idx']] = {
                    'target_idx': op['target_idx'],
                    'target_props': {'element': op['to_element'], 'charge': op['to_charge'], 'chiral': op['to_chiral']}
                }
            elif op_type == 'delete_edge':
                key = tuple(sorted((op['atom1_idx'], op['atom2_idx'])))
                changes["edges_to_delete"].add(key)
            elif op_type == 'insert_edge':
                key = tuple(sorted((op['atom1_idx'], op['atom2_idx'])))
                changes["edges_to_insert"][key] = {'bond_type': op['bond_type']} # RDKit type * 10
            elif op_type == 'substitute_edge':
                source_key = tuple(sorted((op['source_atom1'], op['source_atom2'])))
                target_key = tuple(sorted((op['target_atom1'], op['target_atom2'])))
                changes["edges_to_substitute"][source_key] = {
                    'target_key': target_key,
                    'target_bond_type': op['to_bond_type'] # RDKit type * 10
                }
        return changes

    def _check_target_reached(self, required_changes: Dict[str, Any]) -> bool:
        """Check if all required changes have been processed."""
        return all(not v for k, v in required_changes.items())

    def _update_mappings_after_removal(self, removed_internal_idx: int, maps: Tuple[Dict, Dict, Dict]):
        """Update index mappings after an atom is removed."""
        source_map, internal_to_source_map, target_map = maps
        # print(f"Updating maps after removing internal index: {removed_internal_idx}")
        removed_source_idx = internal_to_source_map.get(removed_internal_idx, None)
        removed_target_idx = None
        for t_idx, i_idx in list(target_map.items()):
            if i_idx == removed_internal_idx:
                removed_target_idx = t_idx
                del target_map[t_idx]
                break
        if removed_internal_idx in internal_to_source_map:
             del internal_to_source_map[removed_internal_idx]
        if removed_source_idx is not None and removed_source_idx in source_map:
             del source_map[removed_source_idx]
        max_internal_idx = 0
        if internal_to_source_map:
             max_internal_idx = max(internal_to_source_map.keys())
        indices_to_shift = sorted([idx for idx in internal_to_source_map.keys() if idx > removed_internal_idx])
        for i in indices_to_shift:
              if i in internal_to_source_map:
                  src_idx = internal_to_source_map.pop(i)
                  internal_to_source_map[i - 1] = src_idx
                  if src_idx in source_map:
                       source_map[src_idx] = i - 1
              target_indices_to_update = [t_idx for t_idx, i_idx in target_map.items() if i_idx == i]
              for t_idx in target_indices_to_update:
                  target_map[t_idx] = i - 1
        # print(f"  Updated source_map: {source_map}")
        # print(f"  Updated internal_to_source_map: {internal_to_source_map}")
        # print(f"  Updated target_map: {target_map}")

    def _update_mappings_after_addition(self, new_internal_idx: int, target_idx: Optional[int], maps: Tuple[Dict, Dict, Dict]):
        """Update index mappings after an atom is added."""
        source_map, internal_to_source_map, target_map = maps
        # print(f"Updating maps after adding internal index: {new_internal_idx}, target_idx: {target_idx}")
        if target_idx is not None:
            target_map[target_idx] = new_internal_idx
        # print(f"  Updated target_map: {target_map}")

    # _find_rl_sequence_for_ged_op remains the same (including debug prints)
    def _find_rl_sequence_for_ged_op(self, design: MoleculeDesign, op_type: str, op_data: Any, maps: Tuple[Dict, Dict, Dict]) -> Optional[List[int]]:
        """
        Constructs the potential RL sequence for a GED op and checks first action validity.
        Returns the full sequence if the first step is valid, else None.
        Includes debugging prints.
        """
        source_map, internal_to_source_map, target_map = maps
        vocab_size = self.vocab_size
        potential_sequence: Optional[List[int]] = None

        # --- Determine Potential RL Sequence based on op_type ---
        current_num_real_atoms = len(design.atoms) - 1 # Needed frequently

        # --- Node Deletion ---
        if op_type == 'delete_node':
            source_idx = op_data
            if source_idx not in source_map: return None
            internal_idx = source_map[source_idx]
            remove_atom_l2_action = design.REMOVE_ATOM_ACTION_L2_MODIFY
            if current_num_real_atoms < 1: return None
            initiate_modify_action = vocab_size + current_num_real_atoms
            potential_sequence = [internal_idx, initiate_modify_action, remove_atom_l2_action]

        # --- Edge Deletion ---
        elif op_type == 'delete_edge':
            s_idx1, s_idx2 = op_data
            if s_idx1 not in source_map or s_idx2 not in source_map: return None
            internal_idx1 = source_map[s_idx1]
            internal_idx2 = source_map[s_idx2]
            remove_bond_l2_action = 6
            if internal_idx2 - 1 < 0: return None # Safety check
            select_existing_l1_action = vocab_size + (internal_idx2 - 1)
            potential_sequence = [internal_idx1, select_existing_l1_action, remove_bond_l2_action]

        # --- Edge Order Reduction ---
        elif op_type == 'reduce_edge_order':
            s_idx1, s_idx2 = op_data['source_key']
            target_bond_type_rdkit = op_data['target_bond_type']
            if s_idx1 not in source_map or s_idx2 not in source_map: return None
            internal_idx1 = source_map[s_idx1]
            internal_idx2 = source_map[s_idx2]
            rl_bond_action = self.rdkit_bond_type_to_rl_action.get(target_bond_type_rdkit)
            if rl_bond_action is None: return None
            if internal_idx2 - 1 < 0: return None
            select_existing_l1_action = vocab_size + (internal_idx2 - 1)
            potential_sequence = [internal_idx1, select_existing_l1_action, rl_bond_action]

        # --- Node Substitution ---
        elif op_type == 'substitute_node':
            source_idx = op_data['source_idx']
            target_props = op_data['target_props']
            if source_idx not in source_map: return None
            internal_idx = source_map[source_idx]
            target_atomic_num = target_props['element']
            target_vocab_idx0 = self.atomic_num_to_vocab_idx0.get(target_atomic_num)
            if target_vocab_idx0 is None: return None
            if current_num_real_atoms < 1: return None
            initiate_modify_action = vocab_size + current_num_real_atoms
            potential_sequence = [internal_idx, initiate_modify_action, target_vocab_idx0]

        # --- Edge Substitution (Same Atoms, Different Type) ---
        elif op_type == 'substitute_edge_type':
            s_idx1, s_idx2 = op_data['source_key']
            target_bond_type_rdkit = op_data['target_bond_type']
            if s_idx1 not in source_map or s_idx2 not in source_map: return None
            internal_idx1 = source_map[s_idx1]
            internal_idx2 = source_map[s_idx2]
            rl_bond_action = self.rdkit_bond_type_to_rl_action.get(target_bond_type_rdkit)
            if rl_bond_action is None: return None
            if internal_idx2 - 1 < 0: return None
            select_existing_l1_action = vocab_size + (internal_idx2 - 1)
            potential_sequence = [internal_idx1, select_existing_l1_action, rl_bond_action]

        # --- Node Insertion ---
        elif op_type == 'insert_node':
            target_idx = op_data['target_idx']
            props = op_data['props']
            target_atomic_num = props['element']
            target_vocab_idx0 = self.atomic_num_to_vocab_idx0.get(target_atomic_num)
            if target_vocab_idx0 is None: return None
            anchor_internal_idx = None
            target_bond_type_rdkit = int(Chem.BondType.SINGLE * 10) # Assume single bond
            current_mask_l0 = design.current_action_mask
            current_level_l0 = design.current_action_level
            if current_level_l0 != 0 or current_mask_l0 is None: return None # Cannot start if not at L0
            found_anchor = False
            for potential_anchor_internal_idx in range(1, len(design.atoms)):
                 if potential_anchor_internal_idx < len(current_mask_l0) and not current_mask_l0[potential_anchor_internal_idx]:
                      anchor_internal_idx = potential_anchor_internal_idx
                      found_anchor = True
                      break
            if not found_anchor: return None
            rl_bond_action = self.rdkit_bond_type_to_rl_action.get(target_bond_type_rdkit)
            if rl_bond_action is None: return None
            potential_sequence = [anchor_internal_idx, target_vocab_idx0, rl_bond_action]

        # --- Edge Insertion ---
        elif op_type == 'insert_edge':
            t_idx1, t_idx2 = op_data['target_key']
            target_bond_type_rdkit = op_data['bond_type']
            if t_idx1 not in target_map or t_idx2 not in target_map: return None
            internal_idx1 = target_map[t_idx1]
            internal_idx2 = target_map[t_idx2]
            rl_bond_action = self.rdkit_bond_type_to_rl_action.get(target_bond_type_rdkit)
            if rl_bond_action is None: return None
            if internal_idx2 - 1 < 0: return None
            select_existing_l1_action = vocab_size + (internal_idx2 - 1)
            potential_sequence = [internal_idx1, select_existing_l1_action, rl_bond_action]

        # --- Edge Order Increase ---
        elif op_type == 'increase_edge_order':
            s_idx1, s_idx2 = op_data['source_key']
            target_bond_type_rdkit = op_data['target_bond_type']
            if s_idx1 not in source_map or s_idx2 not in source_map: return None
            internal_idx1 = source_map[s_idx1]
            internal_idx2 = source_map[s_idx2]
            rl_bond_action = self.rdkit_bond_type_to_rl_action.get(target_bond_type_rdkit)
            if rl_bond_action is None: return None
            if internal_idx2 - 1 < 0: return None
            select_existing_l1_action = vocab_size + (internal_idx2 - 1)
            potential_sequence = [internal_idx1, select_existing_l1_action, rl_bond_action]

        # --- Check Validity of First Action ---
        if not potential_sequence:
            return None

        first_action = potential_sequence[0]
        current_mask = design.current_action_mask
        current_level = design.current_action_level

        # <<< DEBUGGING >>>
        # print(f"    DEBUG Check: Trying Op={op_type}, Data={op_data}")
        # print(f"    DEBUG Check: Design State: Level={current_level}, Connected={design.is_currently_connected}")
        if current_mask is None:
             # print(f"    DEBUG ERROR: current_mask is None!")
             return None
        # print(f"    DEBUG Check: Mask Length={len(current_mask)}")
        # print(f"    DEBUG Check: Potential RL Seq={potential_sequence}")
        # print(f"    DEBUG Check: First Action={first_action}")
        if current_level != 0:
             # print(f"    DEBUG ERROR: Attempting to start new op sequence but current_level is {current_level}!")
             return None # Should only start new op at L0
        if not (0 <= first_action < len(current_mask)):
             # print(f"    DEBUG ERROR: first_action {first_action} is out of bounds for mask (len {len(current_mask)})!")
             return None
        is_masked = current_mask[first_action]
        # print(f"    DEBUG Check: current_mask[{first_action}] = {is_masked}")
        # <<< END DEBUGGING >>>

        if not is_masked:
            return potential_sequence
        else:
            return None

    def convert(self, transformation_dict: Dict) -> Optional[List[int]]:
        """Converts a single GED transformation to an RL action sequence."""
        source_smiles = transformation_dict['source_smiles']
        target_smiles = transformation_dict['target_smiles']
        edit_path = transformation_dict['edit_path']

        print(f"\n--- Converting: {source_smiles} -> {target_smiles} ---")

        try:
            design = MoleculeDesign.from_smiles(self.config, source_smiles, do_finish=False)
        except Exception as e:
            print(f"Initialization failed for {source_smiles}: {e}")
            return None

        try:
            maps = self._initialize_mappings(design, edit_path)
            required_changes = self._categorize_changes(edit_path)
        except Exception as e:
            print(f"Mapping/Categorization failed: {e}")
            return None

        source_map, internal_to_source_map, target_map = maps

        rl_sequence: List[int] = []
        current_rl_sub_sequence: List[int] = []
        current_ged_op_info: Optional[Tuple[str, Any]] = None
        steps = 0
        removed_internal_idx_this_step: Optional[int] = None

        while not self._check_target_reached(required_changes) and steps < MAX_CONVERSION_STEPS:
            steps += 1
            current_mask = design.current_action_mask
            current_level = design.current_action_level
            action_to_take: Optional[int] = None
            ged_op_completed_this_step = False
            newly_added_internal_idx: Optional[int] = None
            newly_added_target_idx: Optional[int] = None
            removed_internal_idx_this_step = None  # Reset removal tracking

            if current_mask is None:
                print(f"Error: Reached None mask unexpectedly at step {steps}.")
                return None

            # --- A. Continue existing RL sub-sequence ---
            if current_rl_sub_sequence:
                action_to_take = current_rl_sub_sequence.pop(0)
                # print(f"  Step {steps}: Continuing sub-sequence. Action: {action_to_take} (Level {current_level})")

                # ---> Add Detailed Pre-Check Debugging <---
                # print(f"  DEBUG PRE-CHECK: Mask Length={len(current_mask)}, Action={action_to_take}")
                if 0 <= action_to_take < len(current_mask):
                    is_masked_pre_check = current_mask[action_to_take]
                    # print(f"  DEBUG PRE-CHECK: Mask[{action_to_take}] = {is_masked_pre_check}")
                else:
                    is_masked_pre_check = True  # Treat out of bounds as masked
                    # print(
                    #     f"  DEBUG PRE-CHECK: Action {action_to_take} is OUT OF BOUNDS for mask len {len(current_mask)}")
                # ---> End Add <---

                # Use the value captured *before* the if statement
                if action_to_take >= len(current_mask) or is_masked_pre_check:  # Check the captured value
                    print(f"Action level is {current_level}, but action_to_take is {action_to_take} (masked: {is_masked_pre_check})")
                    print(f"Bonds: {design.bonds}")
                    print(f"ERROR: Masked action {action_to_take} encountered in sub-sequence! Mask: {current_mask}")
                    return None

            # --- B. Find new RL sub-sequence based on priority (only if at Level 0) ---
            elif current_level == 0:
                processed_action_this_step = False
                candidate_info: Optional[Tuple[List[int], str, Any]] = None

                # --- Priority 1: Deletions/Reductions ---
                # Node Deletions
                for source_idx in list(required_changes["nodes_to_delete"]):
                    seq = self._find_rl_sequence_for_ged_op(design, 'delete_node', source_idx, maps)
                    if seq:
                        candidate_info = (seq, 'delete_node', source_idx)
                        break
                # Edge Deletions
                if not candidate_info:
                    for edge_key in list(required_changes["edges_to_delete"]):
                        seq = self._find_rl_sequence_for_ged_op(design, 'delete_edge', edge_key, maps)
                        if seq:
                            candidate_info = (seq, 'delete_edge', edge_key)
                            break
                # Edge Reductions
                if not candidate_info:
                    for edge_key, sub_data in list(required_changes["edges_to_substitute"].items()):
                        s_idx1, s_idx2 = edge_key
                        if s_idx1 in source_map and s_idx2 in source_map:
                            internal1, internal2 = source_map[s_idx1], source_map[s_idx2]
                            if 0 < internal1 < len(design.atoms) and 0 < internal2 < len(design.atoms):
                                current_order = design.bonds[internal1, internal2]
                                target_order_action = self.rdkit_bond_type_to_rl_action.get(
                                    sub_data['target_bond_type'])
                                if target_order_action is not None and (target_order_action + 1) < current_order:
                                    op_data = {'source_key': edge_key, 'target_bond_type': sub_data['target_bond_type']}
                                    seq = self._find_rl_sequence_for_ged_op(design, 'reduce_edge_order', op_data, maps)
                                    if seq:
                                        candidate_info = (seq, 'reduce_edge_order', op_data)
                                        break
                            else:
                                print(
                                    f"Warning: Invalid internal indices ({internal1}, {internal2}) for edge reduction check {edge_key}.")
                # Process if found in Priority 1
                if candidate_info:
                    seq, op_type, op_data = candidate_info
                    # print(f"  Step {steps}: Starting new op (Priority 1): {op_type} {op_data}. RL Seq: {seq}")
                    current_rl_sub_sequence = seq
                    current_ged_op_info = (op_type, op_data)
                    action_to_take = current_rl_sub_sequence.pop(0)
                    processed_action_this_step = True

                # --- Priority 2: Substitutions ---
                if not processed_action_this_step:
                    candidate_info = None
                    for source_idx, sub_data in list(required_changes["nodes_to_substitute"].items()):
                        op_data = {'source_idx': source_idx, 'target_props': sub_data['target_props']}
                        seq = self._find_rl_sequence_for_ged_op(design, 'substitute_node', op_data, maps)
                        if seq:
                            candidate_info = (seq, 'substitute_node', op_data)
                            break
                    if candidate_info:
                        seq, op_type, op_data = candidate_info
                        # print(f"  Step {steps}: Starting new op (Priority 2): {op_type} {op_data}. RL Seq: {seq}")
                        current_rl_sub_sequence = seq
                        current_ged_op_info = (op_type, op_data)
                        action_to_take = current_rl_sub_sequence.pop(0)
                        processed_action_this_step = True

                # --- Priority 3: Insertions/Increases ---
                if not processed_action_this_step:
                    candidate_info = None
                    # Node Insertions
                    for target_idx, props in list(required_changes["nodes_to_insert"].items()):
                        op_data = {'target_idx': target_idx, 'props': props}
                        seq = self._find_rl_sequence_for_ged_op(design, 'insert_node', op_data, maps)
                        if seq:
                            candidate_info = (seq, 'insert_node', op_data)
                            break
                    # Edge Insertions
                    if not candidate_info:
                        for edge_key, props in list(required_changes["edges_to_insert"].items()):
                            op_data = {'target_key': edge_key, 'bond_type': props['bond_type']}
                            seq = self._find_rl_sequence_for_ged_op(design, 'insert_edge', op_data, maps)
                            if seq:
                                candidate_info = (seq, 'insert_edge', op_data)
                                break
                    # Edge Increases
                    if not candidate_info:
                        for edge_key, sub_data in list(required_changes["edges_to_substitute"].items()):
                            s_idx1, s_idx2 = edge_key
                            if s_idx1 in source_map and s_idx2 in source_map:
                                internal1, internal2 = source_map[s_idx1], source_map[s_idx2]
                                if 0 < internal1 < len(design.atoms) and 0 < internal2 < len(design.atoms):
                                    current_order = design.bonds[internal1, internal2]
                                    target_order_action = self.rdkit_bond_type_to_rl_action.get(
                                        sub_data['target_bond_type'])
                                    if target_order_action is not None and (target_order_action + 1) > current_order:
                                        op_data = {'source_key': edge_key,
                                                   'target_bond_type': sub_data['target_bond_type']}
                                        seq = self._find_rl_sequence_for_ged_op(design, 'increase_edge_order', op_data,
                                                                                maps)
                                        if seq:
                                            candidate_info = (seq, 'increase_edge_order', op_data)
                                            break
                                else:
                                    print(
                                        f"Warning: Invalid internal indices ({internal1}, {internal2}) for edge increase check {edge_key}.")

                    # Process if found in Priority 3
                    if candidate_info:
                        seq, op_type, op_data = candidate_info
                        # print(f"  Step {steps}: Starting new op (Priority 3): {op_type} {op_data}. RL Seq: {seq}")
                        current_rl_sub_sequence = seq
                        current_ged_op_info = (op_type, op_data)
                        action_to_take = current_rl_sub_sequence.pop(0)
                        processed_action_this_step = True

                # --- Check if any action was determined ---
                if not processed_action_this_step:
                    print(f"Error: No valid RL action found at step {steps} for remaining changes: {required_changes}")
                    return None  # Stuck

            # --- C. Execute Action and Update State ---

            # ---> Added this print <---
            # print(f"  DEBUG CHECK C @ step {steps}: action_to_take is '{action_to_take}', type: {type(action_to_take)}")
            # ---> End Add <---

            if action_to_take is not None:
                prev_level = design.current_action_level
                prev_num_atoms = len(design.atoms)
                atom_to_modify_before_action = design.atom_to_modify

                try:
                    design.take_action(action_to_take)
                    rl_sequence.append(action_to_take)
                except Exception as e:
                    print(f"\nERROR during design.take_action({action_to_take}) at step {steps}: {e}")
                    print(f"  Current Level: {prev_level}")
                    print(f"  Current Mask: {current_mask}")
                    print(f"  Sub-sequence remaining: {current_rl_sub_sequence}")
                    print(f"  GED Op Info: {current_ged_op_info}")
                    return None

                # --- Update Mappings ---
                if prev_level == 1 and action_to_take < self.vocab_size:  # L1 Add New Atom completed
                    newly_added_internal_idx = len(design.atoms) - 1
                    newly_added_target_idx = None
                    if current_ged_op_info and current_ged_op_info[0] == 'insert_node':
                        newly_added_target_idx = current_ged_op_info[1]['target_idx']
                    self._update_mappings_after_addition(newly_added_internal_idx, newly_added_target_idx, maps)

                if prev_level == 2 and design.is_modifying_atom is False and atom_to_modify_before_action is not None:
                    if action_to_take == design.REMOVE_ATOM_ACTION_L2_MODIFY:
                        removed_internal_idx_this_step = atom_to_modify_before_action

            else:
                # This case should ideally be caught earlier if no candidate found
                print(f"Error: Internal logic error - action_to_take is None at step {steps}.")
                return None

            # --- Check if GED Op Completed (Level returned to 0) ---
            if design.current_action_level == 0 and not current_rl_sub_sequence:
                ged_op_completed_this_step = True

                # Update maps *after* removal op fully completed
                if removed_internal_idx_this_step is not None:
                    self._update_mappings_after_removal(removed_internal_idx_this_step, maps)
                    removed_internal_idx_this_step = None  # Reset

                # Remove completed op from required changes
                if ged_op_completed_this_step and current_ged_op_info:
                    op_type, op_data = current_ged_op_info
                    # print(f"  Completed GED Op: {op_type} {op_data}")
                    try:
                        if op_type == 'delete_node':
                            required_changes["nodes_to_delete"].discard(op_data)
                        elif op_type == 'insert_node':
                            required_changes["nodes_to_insert"].pop(op_data['target_idx'], None)
                        elif op_type == 'substitute_node':
                            required_changes["nodes_to_substitute"].pop(op_data['source_idx'], None)
                        elif op_type == 'delete_edge':
                            required_changes["edges_to_delete"].discard(op_data)
                        elif op_type == 'insert_edge':
                            required_changes["edges_to_insert"].pop(op_data['target_key'], None)
                        elif op_type in ['reduce_edge_order', 'increase_edge_order', 'substitute_edge_type']:
                            source_key = op_data.get('source_key')
                            if source_key: required_changes["edges_to_substitute"].pop(source_key, None)
                            target_key = op_data.get('target_key')  # May not exist for pure reduction/increase
                            if target_key: required_changes["edges_to_insert"].pop(target_key,
                                                                                   None)  # Remove corresponding insert if substitution
                            # Also remove from delete if substitution covered it
                            if source_key: required_changes["edges_to_delete"].discard(source_key)

                    except KeyError:
                        print(f"Warning: Tried to remove already processed change: {op_type} {op_data}")
                    current_ged_op_info = None

        # --- Loop End ---
        if steps >= MAX_CONVERSION_STEPS:
            print(f"Failed: Max steps ({MAX_CONVERSION_STEPS}) exceeded for {source_smiles} -> {target_smiles}")
            return None

        if not self._check_target_reached(required_changes):
            print(
                f"Failed: Loop finished but changes remain for {source_smiles} -> {target_smiles}: {required_changes}")
            return None

        # --- Final Termination Action ---
        if design.is_terminable():
            try:
                if not design.current_action_mask[0]:
                    design.take_action(0)
                    rl_sequence.append(0)
                    # print("  Added final terminate action.")
                else:
                    print(f"Warning: Final state terminable but action 0 masked for {source_smiles} -> {target_smiles}")
            except ValueError as e:
                print(f"Warning: Error during final terminate action for {source_smiles} -> {target_smiles}: {e}")
        else:
            print(f"Warning: Final state not terminable for {source_smiles} -> {target_smiles}")

        print(f"--- Conversion SUCCESS: {source_smiles} -> {target_smiles} ({len(rl_sequence)} RL steps) ---")
        return rl_sequence


# --- Main Execution ---

def main_converter():
    """Loads GED data and converts it to RL sequences."""
    config = MoleculeConfig() # Load default config
    converter = GedToRlConverter(config)

    for datatype in ["train", "valid"]:
        print(f"\n{'='*60}\nProcessing {datatype.upper()} GED data for RL conversion\n{'='*60}")
        ged_data = load_latest_checkpoint(datatype)
        if not ged_data:
            continue

        rl_dataset = []
        failed_conversions = 0

        # Use a smaller subset for initial debugging if needed
        # ged_data = ged_data[:10]

        for i, transformation in enumerate(tqdm(ged_data, desc=f"Converting {datatype}")):
            rl_actions = converter.convert(transformation)
            if rl_actions is not None:
                rl_dataset.append({
                    'source_smiles': transformation['source_smiles'],
                    'target_smiles': transformation['target_smiles'],
                    'rl_actions': rl_actions
                })
            else:
                failed_conversions += 1

            # Optional: Save intermediate results periodically
            # if (i + 1) % 100 == 0: # Save more frequently during debug
            #     intermediate_path = os.path.join(OUTPUT_DIR, f"rl_sequences_{datatype}_checkpoint_{i+1}.pkl")
            #     with open(intermediate_path, "wb") as f:
            #         pickle.dump(rl_dataset, f)
            #     print(f"\nSaved intermediate RL sequences to {intermediate_path}")

        # Save final RL dataset
        output_path = os.path.join(OUTPUT_DIR, f"rl_sequences_{datatype}.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(rl_dataset, f)

        print(f"\nFinished processing {datatype}:")
        print(f"  Successfully converted: {len(rl_dataset)}")
        print(f"  Failed conversions:     {failed_conversions}")
        print(f"  Saved RL sequences to:  {output_path}")

if __name__ == "__main__":
    print(f"Current User: {os.getlogin() if hasattr(os, 'getlogin') else 'N/A'}") # Get username if possible
    print(f"Current Time (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    main_converter()