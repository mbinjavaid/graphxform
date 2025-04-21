import copy
# import random
import numpy as np
import torch
from torch import nn
from rdkit import Chem, RDLogger
from rdkit.Chem import rdmolfiles, rdmolops
import networkx as nx

import traceback
from config import MoleculeConfig
from core.abstracts import BaseTrajectory
# from core.utils import softmax

from typing import List, Tuple, Dict, Optional

# # Suppress RDKit warnings
# RDLogger.DisableLog('rdApp.*')


def build_reverse_atom_lookup(config: MoleculeConfig) -> Dict[Tuple[int, int, int], int]:
    """
    Creates a lookup dictionary mapping atom properties back to vocabulary indices.

    Args:
        config: The MoleculeConfig instance containing the atom_vocabulary.

    Returns:
        A dictionary mapping (atomic_number, formal_charge, chiral_tag) -> vocab_idx (1-based).
    """
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
            chiral = atom_config.get('chiral_tag', 0)  # 0: unspecified, 1: CW, 2: CCW (RDKit mapping)
        except KeyError as e:
            print(f"Warning: Missing expected property {e} for atom '{name}' in config. Skipping.")
            continue

        key = (atomic_num, charge, chiral)
        vocab_idx = i + 1  # 1-based index

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


class MoleculeDesign(BaseTrajectory):
    """
    Environment for molecular design using a revised hierarchical action space (v2025-04-20 NetworkX).

    State Management: Internal NumPy arrays (self.atoms, self.bonds) are the primary source of truth.
                      NetworkX used for connectivity checks during simulation.
                      RDKit Mol object is constructed only during finalize().

    Action Levels:
        - Level 0: Terminate (if connected) or Select Existing Atom.
        - Level 1: Add New Atom, Select Existing Atom for Bond, or Initiate Modify Atom.
        - Level 2 (Bond Path): Set Bond Order 1-6 (creates if 0) or Remove Bond.
        - Level 2 (Modify Path): Replace Atom Type or Remove Atom.
    """
    maximum_bond_order = 6
    virtual_bond_idx = 7 # Keep for internal bond matrix representation
    maximum_num_atoms_overall = 100 # Still useful for padding limits? Maybe remove later if unused.
    bond_types = { # RDKit bond types for adding bonds *during finalize*
        1: Chem.rdchem.BondType.SINGLE,
        2: Chem.rdchem.BondType.DOUBLE,
        3: Chem.rdchem.BondType.TRIPLE,
        4: Chem.rdchem.BondType.QUADRUPLE,
        5: Chem.rdchem.BondType.QUINTUPLE,
        6: Chem.rdchem.BondType.HEXTUPLE
    }
    REMOVE_ATOM_ACTION_L2_MODIFY = -1 # Placeholder, set in __init__

    def __init__(self, config: MoleculeConfig, initial_atom: int):
        self.config = config
        self.atom_vocabulary = self.config.atom_vocabulary
        self.vocabulary_atom_idcs = list(range(1, len(self.atom_vocabulary) + 1)) # [1, ..., V]
        self.vocabulary_atom_names = list(self.atom_vocabulary.keys())
        self.vocabulary_valence = [-1] * (len(self.vocabulary_atom_names) + 1)
        for i, name in enumerate(self.vocabulary_atom_names):
             self.vocabulary_valence[i+1] = self.atom_vocabulary[name]["valence"]

        self.atom_feasibility_mask = [not self.atom_vocabulary[x]["allowed"] for x in self.vocabulary_atom_names]
        self.vocab_size = len(self.vocabulary_atom_idcs) # V
        self.REMOVE_ATOM_ACTION_L2_MODIFY = self.vocab_size # V
        self.upper_limit_atoms = self.config.max_num_atoms

        assert initial_atom in self.vocabulary_atom_idcs and not self.atom_feasibility_mask[initial_atom - 1], \
            f"Initial atom {initial_atom} must be in vocabulary {self.vocabulary_atom_idcs} and allowed in config."
        self.initial_atom = initial_atom

        # --- Internal State (Primary) ---
        self.atoms = np.array([0, initial_atom], dtype=np.uint8) # Includes virtual atom 0
        self.bonds = np.zeros((2, 2), dtype=np.uint8)
        self.bonds[0, 1] = self.bonds[1, 0] = self.virtual_bond_idx # Virtual connection

        # --- No rdkit_mol attribute here ---
        # self.rdkit_mol = None # Removed

        # --- No topological_distance_matrix attribute here ---
        # self.topological_distance_matrix = None # Removed
        # self.virtual_distance = self.maximum_num_atoms_overall + 1 # Removed
        # self.infinity_distance = self.maximum_num_atoms_overall + 2 # Removed

        # --- Trajectory State ---
        self.synthesis_done = False
        self.smiles_string: Optional[str] = None
        self.objective: Optional[float] = None
        self.sa_score: float = 0. # Keep, might be evaluated externally
        self.infeasibility_flag: bool = False
        self.is_currently_connected: bool = True # Assume initial single atom is connected

        # --- Action Handling State ---
        self.current_action_level = 0
        self.current_action_mask: Optional[np.array] = None
        self.history: List[int] = []
        self.l0_selected_atom_idx: Optional[int] = None # 1-based internal index
        self.is_modifying_atom: bool = False
        self.atom_to_modify: Optional[int] = None # 1-based internal index
        self.l1_new_atom_type: Optional[int] = None # 1-based vocab index
        self.l1_selected_existing_atom_idx: Optional[int] = None # 1-based internal index

        # Initial mask calculation relies on internal state
        self.update_action_mask()

    # --- Removed RDKit Helper Methods (_add_atom_to_rdkit, _update_rdkit_bond, etc.) ---
    # --- They are implicitly handled by internal state updates and finalize() ---

    def _check_and_update_connectivity(self):
        """Checks connectivity using NetworkX on the internal state."""
        num_real_atoms = len(self.atoms) - 1
        if num_real_atoms <= 1:
            self.is_currently_connected = True
            return

        G = nx.Graph()
        G.add_nodes_from(range(num_real_atoms))
        adj_matrix = self.bonds[1:, 1:]
        rows, cols = np.where(adj_matrix > 0)
        edges = zip(rows, cols)
        G.add_edges_from(edges)

        try:
            if G.number_of_nodes() > 0:
                 self.is_currently_connected = nx.is_connected(G)
            else:
                 self.is_currently_connected = True
        except Exception as e:
            print(f"WARNING: NetworkX connectivity check failed: {e}. Assuming disconnected.")
            self.is_currently_connected = False

    def _get_current_valence_usage(self) -> np.array:
        """Calculates the sum of explicit bond orders for each real atom from self.bonds."""
        num_real_atoms = len(self.atoms) - 1
        if num_real_atoms <= 0:
            return np.array([], dtype=int)
        current_explicit_usage = np.sum(self.bonds[1 : num_real_atoms + 1, 1 : num_real_atoms + 1], axis=1)
        return current_explicit_usage.astype(int)

    def _get_remaining_valence(self) -> np.array:
        """Calculates remaining valence for each real atom based on self.atoms and self.bonds."""
        num_real_atoms = len(self.atoms) - 1
        if num_real_atoms <= 0:
            return np.array([], dtype=int)
        current_usage = self._get_current_valence_usage()
        total_valence = np.array([self.vocabulary_valence[atom_vocab_idx]
                                  for atom_vocab_idx in self.atoms[1:]], dtype=int)
        remaining = total_valence - current_usage
        remaining = np.maximum(0, remaining)
        return remaining

    def update_action_mask(self):
        """Creates the action mask based on the internal state (self.atoms, self.bonds)."""
        if self.synthesis_done:
            self.current_action_mask = None
            return

        num_real_atoms = len(self.atoms) - 1
        remaining_valence = self._get_remaining_valence()

        if self.current_action_level == 0:
            action_space_size = 1 + num_real_atoms
            mask = np.zeros(action_space_size, dtype=bool)
            if not self.is_currently_connected or num_real_atoms <= 0:
                mask[0] = True
            if num_real_atoms == 0:
                mask[1:] = True
            self.current_action_mask = mask

        elif self.current_action_level == 1:
            action_space_size = self.vocab_size + num_real_atoms + 1
            mask = np.ones(action_space_size, dtype=bool)
            anchor_atom_internal_idx = self.l0_selected_atom_idx
            if anchor_atom_internal_idx is None or anchor_atom_internal_idx <= 0 or anchor_atom_internal_idx > num_real_atoms:
                print("ERROR: L1 Mask - Invalid anchor atom index.")
                self.current_action_mask = mask; return
            anchor_atom_0_idx = anchor_atom_internal_idx - 1

            if remaining_valence[anchor_atom_0_idx] <= 0:
                mask[self.vocab_size + num_real_atoms] = False # Unmask Initiate Modify Atom ONLY
                self.current_action_mask = mask; return

            if self.upper_limit_atoms is None or num_real_atoms < self.upper_limit_atoms:
                for i in range(self.vocab_size):
                    if not self.atom_feasibility_mask[i] and self.vocabulary_valence[i+1] >= 1:
                        mask[i] = False

            # for target_0_idx in range(num_real_atoms):
            #     target_internal_idx = target_0_idx + 1
            #     action_idx = self.vocab_size + target_0_idx
            #     if target_internal_idx == anchor_atom_internal_idx: continue
            #     bond_exists = self.bonds[anchor_atom_internal_idx, target_internal_idx] > 0
            #     target_has_valence = remaining_valence[target_0_idx] > 0
            #     if bond_exists or target_has_valence:
            #         mask[action_idx] = False

            print(
                f"DEBUG L1 Mask Calc: Anchor={anchor_atom_internal_idx}, RemValAnchor={remaining_valence[anchor_atom_0_idx]}")

            for target_0_idx in range(num_real_atoms):
                target_internal_idx = target_0_idx + 1
                action_idx = self.vocab_size + target_0_idx  # Action index for selecting this existing atom

                if target_internal_idx == anchor_atom_internal_idx: continue  # Skip selecting self

                bond_exists = self.bonds[anchor_atom_internal_idx, target_internal_idx] > 0
                target_has_valence = remaining_valence[target_0_idx] > 0

                # --- Add Detailed Debug Print ---
                should_unmask = bond_exists or target_has_valence
                print(
                    f"  L1 Mask Check: Target={target_internal_idx}, ActionIdx={action_idx}, BondExists={bond_exists}, TargetHasVal={target_has_valence}, ShouldUnmask={should_unmask}")
                # --- End Debug Print ---

                if should_unmask:
                    mask[action_idx] = False  # Unmask the action
                    print(f"    => UNMASKING Action {action_idx}")  # Confirm unmasking


            mask[self.vocab_size + num_real_atoms] = False
            self.current_action_mask = mask

        elif self.current_action_level == 2:
            if self.is_modifying_atom:
                action_space_size = self.vocab_size + 1
                mask = np.ones(action_space_size, dtype=bool)
                atom_internal_idx = self.atom_to_modify
                if atom_internal_idx is None or atom_internal_idx <= 0 or atom_internal_idx > num_real_atoms:
                    print("ERROR: L2 Modify Mask - Invalid atom_to_modify index.")
                    self.current_action_mask = mask; return
                atom_0_idx = atom_internal_idx - 1
                current_atom_type_idx = self.atoms[atom_internal_idx]
                current_usage = 0
                valence_usage_array = self._get_current_valence_usage()
                if atom_0_idx < len(valence_usage_array):
                    current_usage = valence_usage_array[atom_0_idx]
                else:
                    print(f"WARNING: L2 Modify Mask - Index {atom_0_idx} out of bounds for valence usage.")

                for vocab_idx_0 in range(self.vocab_size):
                    new_atom_type = vocab_idx_0 + 1
                    action_idx = vocab_idx_0
                    if self.atom_feasibility_mask[vocab_idx_0]: continue
                    if new_atom_type == current_atom_type_idx: continue
                    if self.vocabulary_valence[new_atom_type] >= current_usage:
                        mask[action_idx] = False
                if num_real_atoms > 0:
                    mask[self.REMOVE_ATOM_ACTION_L2_MODIFY] = False
                self.current_action_mask = mask


            else:  # Bond Path

                action_space_size = 7

                mask = np.ones(action_space_size, dtype=bool)

                atom_A_internal_idx = self.l0_selected_atom_idx

                atom_B_internal_idx = -1

                if self.l1_new_atom_type is not None:
                    atom_B_internal_idx = len(self.atoms) - 1

                elif self.l1_selected_existing_atom_idx is not None:
                    atom_B_internal_idx = self.l1_selected_existing_atom_idx

                else:
                    print("ERROR: L2 Bond Mask - L1 context missing."); self.current_action_mask = mask; return

                # --- Add Debug Print ---

                print(f"  DEBUG L2 Bond Mask Calc: A={atom_A_internal_idx}, B={atom_B_internal_idx}")

                # --- End Add ---

                # Add validation checks *before* indexing arrays

                num_real_atoms = len(self.atoms) - 1

                if (atom_A_internal_idx is None or atom_A_internal_idx <= 0 or atom_A_internal_idx > num_real_atoms or

                        atom_B_internal_idx <= 0 or atom_B_internal_idx > num_real_atoms):
                    print(
                        f"ERROR: L2 Bond Mask - Invalid indices A={atom_A_internal_idx}, B={atom_B_internal_idx} (NumReal={num_real_atoms})")

                    self.current_action_mask = mask;
                    return

                atom_A_0_idx = atom_A_internal_idx - 1

                atom_B_0_idx = atom_B_internal_idx - 1

                remaining_valence = self._get_remaining_valence()  # Get remaining valence

                if atom_A_0_idx >= len(remaining_valence) or atom_B_0_idx >= len(remaining_valence):
                    print(
                        f"ERROR: L2 Bond Mask - Indices {atom_A_0_idx} or {atom_B_0_idx} out of bounds for rem_val (len {len(remaining_valence)}).")

                    self.current_action_mask = mask;
                    return

                current_bond_order = self.bonds[atom_A_internal_idx, atom_B_internal_idx]

                # --- Add Debug Print ---

                print(
                    f"  DEBUG L2 Bond Mask Calc: Current Bond Order ({atom_A_internal_idx},{atom_B_internal_idx}) = {current_bond_order}")

                # --- End Add ---

                valence_A_rem = remaining_valence[atom_A_0_idx]

                valence_B_rem = remaining_valence[atom_B_0_idx]

                max_increase = min(valence_A_rem, valence_B_rem)

                effective_current_order = int(current_bond_order) if current_bond_order > 0 else 0

                max_allowed_final_order = min(effective_current_order + max_increase, self.maximum_bond_order)

                for order in range(1, self.maximum_bond_order + 1):

                    action_idx = order - 1

                    if order <= max_allowed_final_order: mask[action_idx] = False

                # Unmask Remove Bond action (6) if bond currently exists

                if current_bond_order > 0:
                    mask[6] = False

                self.current_action_mask = mask
        else:
             raise ValueError(f"Invalid current_action_level: {self.current_action_level}")

    # --- Removed update_topological_distance_matrix ---

    def _adjust_indices_after_removal(self, removed_internal_idx: int):
        """Adjusts stored internal indices after an atom removal."""
        if self.l0_selected_atom_idx is not None and self.l0_selected_atom_idx > removed_internal_idx:
            self.l0_selected_atom_idx -= 1
        if self.l1_selected_existing_atom_idx is not None and self.l1_selected_existing_atom_idx > removed_internal_idx:
            self.l1_selected_existing_atom_idx -= 1


    # Inside MoleculeDesign class
    def take_action(self, action: int):
        """Execute a given action, updating internal state directly."""
        if self.synthesis_done: raise RuntimeError("Cannot take action on terminated design.")

        # --- Log Mask BEFORE the initial check ---
        initial_mask_str = "None"
        initial_mask_len = 0
        if self.current_action_mask is not None:
            initial_mask_len = len(self.current_action_mask)
            try:
                # Check if action is within bounds before accessing mask value
                if action < initial_mask_len:
                    initial_mask_str = f"Len={initial_mask_len}, ValueAtAction={self.current_action_mask[action]}"
                else:
                    initial_mask_str = f"Len={initial_mask_len}, Action OOB"
            except IndexError:
                initial_mask_str = f"Len={initial_mask_len}, IndexError accessing action {action}"  # Defensive
        print(
            f"  DEBUG take_action: ENTRY Action={action}, Level={self.current_action_level}. Initial Mask Check: {initial_mask_str}")
        # --- End Log ---

        if self.current_action_mask is None or action >= len(self.current_action_mask) or self.current_action_mask[
            action]:
            # If this exception occurs, the log above shows the mask state that caused it.
            raise ValueError(f"Action {action} masked/invalid for level {self.current_action_level}.")

        current_level = self.current_action_level
        next_level = 0  # Default next level
        self.history.append(int(action))
        num_real_atoms_before = len(self.atoms) - 1

        try:
            atom_removed = False
            # --- Apply Action based on Level ---
            if current_level == 0:
                if action == 0:  # Terminate
                    self.synthesis_done = True
                    self.finalize()  # Builds local rdkit_mol
                    next_level = -1  # Special level for termination
                else:  # Select Atom
                    self.l0_selected_atom_idx = action
                    # Reset L1/L2 state variables explicitly when starting L1
                    self.is_modifying_atom = False;
                    self.atom_to_modify = None;
                    self.l1_new_atom_type = None;
                    self.l1_selected_existing_atom_idx = None
                    next_level = 1
            elif current_level == 1:
                modify_idx = self.vocab_size + num_real_atoms_before
                if action < self.vocab_size:  # Add Atom
                    self.l1_new_atom_type = action + 1
                    self.atoms = np.append(self.atoms, self.l1_new_atom_type)
                    new_size = len(self.atoms);
                    new_idx = new_size - 1
                    self.bonds = np.pad(self.bonds, [(0, 1), (0, 1)], 'constant', constant_values=0)
                    self.bonds[0, new_idx] = self.bonds[new_idx, 0] = self.virtual_bond_idx  # Connect to virtual node
                    self.is_modifying_atom = False  # Ensure correct path for L2
                    next_level = 2
                elif action < modify_idx:  # Select Existing
                    self.l1_selected_existing_atom_idx = (action - self.vocab_size) + 1
                    self.is_modifying_atom = False  # Ensure correct path for L2
                    next_level = 2
                elif action == modify_idx:  # Initiate Modify
                    self.atom_to_modify = self.l0_selected_atom_idx
                    self.is_modifying_atom = True  # Set flag for L2
                    next_level = 2
                else:
                    raise ValueError(f"Invalid L1 action index: {action}")
            elif current_level == 2:
                if self.is_modifying_atom:  # Modify Path
                    mod_idx = self.atom_to_modify
                    if mod_idx is None: raise ValueError("L2 Modify path entered but atom_to_modify not set")
                    if action < self.vocab_size:  # Replace Type
                        self.atoms[mod_idx] = action + 1
                    elif action == self.REMOVE_ATOM_ACTION_L2_MODIFY:  # Remove Atom
                        # --- Ensure _adjust_indices_after_removal is called AFTER state changes ---
                        removed_idx_for_adjust = mod_idx  # Store index before potential modification by delete
                        self.atoms = np.delete(self.atoms, removed_idx_for_adjust)
                        self.bonds = np.delete(np.delete(self.bonds, removed_idx_for_adjust, 0), removed_idx_for_adjust,
                                               1)
                        self._adjust_indices_after_removal(removed_idx_for_adjust)  # Adjust other indices
                        atom_removed = True
                        # --- End ---
                    else:
                        raise ValueError(f"Invalid L2 Modify action index: {action}")
                    # Reset modify state after completion
                    self.is_modifying_atom = False;
                    self.atom_to_modify = None;
                    next_level = 0  # Back to L0
                else:  # Bond Path
                    idx_A = self.l0_selected_atom_idx
                    # Determine B based on L1 action
                    idx_B = -1  # Initialize
                    if self.l1_new_atom_type is not None:
                        idx_B = len(self.atoms) - 1  # New atom is last one
                    elif self.l1_selected_existing_atom_idx is not None:
                        idx_B = self.l1_selected_existing_atom_idx
                    else:
                        raise ValueError("L2 Bond path entered but L1 context (new/existing) missing")

                    if idx_A is None or idx_B == -1: raise ValueError(f"L2 Bond indices invalid: A={idx_A}, B={idx_B}")

                    if action <= 5:  # Set Order (Action 0 = Order 1, ..., Action 5 = Order 6)
                        order = action + 1
                        self.bonds[idx_A, idx_B] = self.bonds[idx_B, idx_A] = order
                    elif action == 6:  # Remove Bond (Set Order 0)
                        self.bonds[idx_A, idx_B] = self.bonds[idx_B, idx_A] = 0
                    else:
                        raise ValueError(f"Invalid L2 Bond action index: {action}")
                    # Reset L1 state after completion
                    self.l1_new_atom_type = None;
                    self.l1_selected_existing_atom_idx = None;
                    next_level = 0  # Back to L0

            # --- Update Mask and Level (if not terminated) ---
            if next_level != -1:
                # Log mask state BEFORE calling update_action_mask
                mask_before_update_str = "None"
                if self.current_action_mask is not None: mask_before_update_str = f"Len={len(self.current_action_mask)}"  # Just log length maybe
                print(
                    f"  DEBUG take_action: Action={action}, Level={current_level}. Mask BEFORE update_action_mask call: {mask_before_update_str}")

                self._check_and_update_connectivity()  # Update connectivity based on new state

                self.current_action_level = next_level  # Set level for the NEXT step

                self.update_action_mask()  # Calculate mask for the NEXT step

                # Log mask state IMMEDIATELY AFTER update_action_mask returns
                mask_after_update_str = "None"
                if self.current_action_mask is not None:
                    mask_after_update_str = f"Len={len(self.current_action_mask)}, Vals={self.current_action_mask[:7]}..."  # Log more values
                print(
                    f"  DEBUG take_action: Action={action}, Level={current_level}. Mask AFTER update_action_mask call: {mask_after_update_str}")

            else:  # Termination action was taken
                # Log mask state BEFORE setting to None
                print(
                    f"  DEBUG take_action: Action={action}, Level={current_level}. Mask BEFORE setting to None (Termination): {'Exists' if self.current_action_mask is not None else 'None'}")
                self.current_action_mask = None  # No further actions possible

            # --- Log Mask state just before function returns ---
            final_mask_str = "None"
            if self.current_action_mask is not None:
                final_mask_str = f"Len={len(self.current_action_mask)}, Vals={self.current_action_mask[:7]}..."
            print(
                f"  DEBUG take_action: Action={action}, Level={current_level}. Mask JUST BEFORE RETURN: {final_mask_str}")
            # --- End Log ---

        except Exception as e:
            print(f"FATAL ERROR during take_action(action={action}, L{current_level}): {e}")
            traceback.print_exc()
            self.infeasibility_flag = True;
            self.synthesis_done = True;
            self.current_action_mask = None
            # Log mask state after error
            print(f"  DEBUG take_action: Action={action}, Level={current_level}. Mask AFTER EXCEPTION: None")


    def finalize(self, assert_feasible: bool = False):
        """Finalize molecule design: build RDKit mol, sanitize, generate SMILES."""
        # --- Build RDKit Mol from final internal state ---
        # Call to_rdkit_mol to generate the object needed for sanitization/SMILES
        rdkit_mol = self.to_rdkit_mol(sanitize=False)  # Build unsanitized first

        # Ensure connectivity check is based on the final internal state (using NetworkX)
        self._check_and_update_connectivity()

        if assert_feasible:
            try:
                self.assert_feasible()
            except AssertionError as e:
                print(f"Feasibility assertion failed during finalize: {e}")
                self.infeasibility_flag = True

        # Check connectivity required for valid SMILES (unless empty)
        if len(self.atoms) > 1 and not self.is_currently_connected:  # Check internal atom count
            print("WARNING: Final molecule is disconnected. SMILES may represent fragments.")
            # Optionally set infeasibility_flag here if disconnected molecules are invalid
            # self.infeasibility_flag = True

        if not self.infeasibility_flag:
            if rdkit_mol.GetNumAtoms() == 0 and len(self.atoms) > 1:
                # Handle case where internal state has atoms but RDKit conversion failed
                print("ERROR: RDKit molecule is empty despite internal state having atoms.")
                self.infeasibility_flag = True
                self.smiles_string = None
            elif rdkit_mol.GetNumAtoms() > 0:
                try:
                    Chem.SanitizeMol(rdkit_mol)  # Sanitize the newly built mol
                    self.smiles_string = Chem.MolToSmiles(rdkit_mol)
                except Exception as e:
                    print(f"Final sanitization/SMILES generation failed: {e}")
                    self.infeasibility_flag = True
                    self.smiles_string = None
            else:  # No real atoms internally either
                self.smiles_string = ""  # Empty SMILES for empty molecule
        else:
            self.smiles_string = None  # Ensure SMILES is None if infeasible

    def assert_feasible(self):
        """Check internal state consistency (NumPy arrays)."""
        assert self.atoms[0] == 0, "First atom should be virtual (0)"
        num_atoms = len(self.atoms)
        num_real_atoms = num_atoms - 1

        if num_real_atoms > 0:
             valid_indices = all(1 <= idx <= self.vocab_size for idx in self.atoms[1:])
             assert valid_indices, f"Invalid atom vocab index found: {self.atoms}"
             allowed_check = all(not self.atom_feasibility_mask[idx - 1] for idx in self.atoms[1:])
             assert allowed_check, f"Disallowed atom type found: {self.atoms}"

        assert self.upper_limit_atoms is None or num_real_atoms <= self.upper_limit_atoms, f"Max atoms exceeded"
        assert self.bonds.shape == (num_atoms, num_atoms), f"Bonds shape mismatch"
        # --- Removed topological distance matrix check ---
        if num_real_atoms > 0:
             assert np.all(self.bonds[0, 1:] == self.virtual_bond_idx), "Virtual bond missing"
             assert np.all(self.bonds[1:, 0] == self.virtual_bond_idx), "Virtual bond missing"
        assert not np.any(self.bonds.diagonal()), "Self-loops detected"
        assert np.all(self.bonds == self.bonds.T), "Bond matrix not symmetric"
        if num_real_atoms > 0:
             remaining_valence = self._get_remaining_valence()
             assert np.all(remaining_valence >= 0), f"Valence constraints violated: {remaining_valence}"

    def to_rdkit_mol(self, sanitize=True) -> Chem.RWMol:
        """Creates a *new* RDKit molecule from the internal state (atoms, bonds)."""
        mol = Chem.RWMol()
        num_total_atoms = len(self.atoms)
        if num_total_atoms <= 1: return mol

        rdkit_idx_map = {} # Map internal index -> new RDKit index
        for idx, atom_vocab_idx in enumerate(self.atoms):
            if idx == 0: continue # Skip virtual
            if not (1 <= atom_vocab_idx <= self.vocab_size):
                 print(f"WARNING: Invalid vocab index {atom_vocab_idx} in self.atoms during to_rdkit_mol.")
                 continue
            atom_config = self.atom_vocabulary[self.vocabulary_atom_names[atom_vocab_idx - 1]]
            a = Chem.Atom(atom_config["atomic_number"])
            if "formal_charge" in atom_config: a.SetFormalCharge(atom_config["formal_charge"])
            ct = atom_config.get("chiral_tag")
            if ct == 1: a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
            elif ct == 2: a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
            else: a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            new_rdkit_idx = mol.AddAtom(a)
            rdkit_idx_map[idx] = new_rdkit_idx # Store mapping

        for i in range(1, num_total_atoms):
            for j in range(i + 1, num_total_atoms):
                bond_order = self.bonds[i, j]
                if bond_order > 0 and bond_order <= self.maximum_bond_order:
                    # Ensure indices exist in map (they should if atoms were added correctly)
                    if i in rdkit_idx_map and j in rdkit_idx_map:
                        rdkit_i, rdkit_j = rdkit_idx_map[i], rdkit_idx_map[j]
                        rdkit_bond_type = self.bond_types.get(int(bond_order))
                        if rdkit_bond_type:
                            mol.AddBond(rdkit_i, rdkit_j, rdkit_bond_type)
                    else:
                        print(f"WARNING: Missing RDKit index map entry for internal indices {i} or {j} in to_rdkit_mol.")
                elif bond_order > self.maximum_bond_order and bond_order != self.virtual_bond_idx:
                     print(f"WARNING: Invalid bond order {bond_order} in self.bonds during to_rdkit_mol.")

        if sanitize:
            try: Chem.SanitizeMol(mol)
            except Exception as e: print(f"Sanitization failed in to_rdkit_mol: {e}")
        return mol

    def is_terminable(self):
        """Checks if the current state allows termination."""
        can_terminate = self.current_action_level == 0 and not self.synthesis_done
        connectivity_ok = (len(self.atoms) <= 2) or self.is_currently_connected
        return can_terminate and connectivity_ok

    def to_smiles(self) -> Optional[str]:
        """Returns the SMILES string if finalized and valid."""
        return self.smiles_string

    @staticmethod
    def init_batch_from_instance_list(config: MoleculeConfig, instances: List[int], network: Optional[nn.Module]=None, device: Optional[torch.device]=None):
        return [MoleculeDesign(config=config, initial_atom=atom) for atom in instances]

    @staticmethod
    def log_probability_fn(trajectories: List['MoleculeDesign'], network: nn.Module) -> List[np.array]:
        """Calculates masked log probabilities for the current action level of each trajectory."""
        log_probs_to_return: List[np.array] = []
        network.eval()
        with torch.no_grad():
            batch = MoleculeDesign.list_to_batch(molecules=trajectories, device=network.device)
            batch_logits_l0, batch_logits_l1, batch_logits_l2 = network(batch)
            batch_logits_l0 = batch_logits_l0.cpu().numpy()
            batch_logits_l1 = batch_logits_l1.cpu().numpy()
            batch_logits_l2 = batch_logits_l2.cpu().numpy()

            for i, mol in enumerate(trajectories):
                mask = mol.current_action_mask
                if mask is None: log_probs_to_return.append(np.array([])); continue
                if mol.current_action_level == 0: logits = batch_logits_l0[i]
                elif mol.current_action_level == 1: logits = batch_logits_l1[i]
                elif mol.current_action_level == 2: logits = batch_logits_l2[i]
                else: log_probs_to_return.append(np.array([])); continue

                mask_len = len(mask)
                if len(logits) > mask_len: logits = logits[:mask_len]
                elif len(logits) < mask_len: raise ValueError(f"Logits/Mask length mismatch L{mol.current_action_level}")

                logits[mask] = -np.inf
                max_logit = np.max(logits)
                if np.isneginf(max_logit): log_probs = logits
                else:
                     exp_logits = np.exp(logits - max_logit)
                     log_sum_exp = np.log(np.sum(exp_logits))
                     log_probs = logits - (max_logit + log_sum_exp)
                log_probs[mask] = -np.inf
                log_probs_to_return.append(log_probs)
        return log_probs_to_return

    def transition_fn(self, action: int) -> Tuple['BaseTrajectory', bool]:
        """Creates a copy, takes the action, and returns the new state and termination status."""
        copied_molecule = copy.deepcopy(self)
        copied_molecule.take_action(action)
        return copied_molecule, copied_molecule.synthesis_done

    def to_max_evaluation_fn(self) -> float:
        """Returns the objective value."""
        if self.objective is None: print("WARNING: Objective is None."); return float("-inf")
        return float("-inf") if self.infeasibility_flag else self.objective

    def num_actions(self) -> int:
        """Returns the number of valid (unmasked) actions at the current level."""
        if self.current_action_mask is None: return 0
        return int(np.sum(~self.current_action_mask))

    @staticmethod
    def list_to_batch(molecules: List['MoleculeDesign'], device: torch.device = None,
                      include_feasibility_masks: bool = False) -> dict:
        """Converts a list of MoleculeDesign instances to a batch dictionary (No distance)."""
        atoms_padding_idx = molecules[0].vocab_size + 1
        max_valence = max([-1] + [v for v in molecules[0].vocabulary_valence if v is not None and v >= 0])
        degree_padding_idx = max_valence + 2
        bond_padding_idx = MoleculeDesign.virtual_bond_idx + 1
        # --- Removed distance_padding_idx ---

        device = torch.device("cpu") if device is None else device
        num_atoms = [len(mol.atoms) for mol in molecules]
        max_num_atoms = max(num_atoms) if num_atoms else 0
        batch_level_idx = [mol.current_action_level for mol in molecules]

        batch_picked_atom_mhe = np.zeros((len(molecules), max_num_atoms), dtype=int)
        for i, mol in enumerate(molecules):
            if mol.current_action_level >= 1 and mol.l0_selected_atom_idx is not None:
                 if 0 <= mol.l0_selected_atom_idx < max_num_atoms:
                      batch_picked_atom_mhe[i, mol.l0_selected_atom_idx] = 1

        batch_atoms = np.stack([
            np.concatenate((mol.atoms, np.full(max_num_atoms - num_atoms[i], fill_value=atoms_padding_idx, dtype=np.uint8))) if num_atoms[i] > 0 else np.full(max_num_atoms, fill_value=atoms_padding_idx, dtype=np.uint8)
            for i, mol in enumerate(molecules)
        ])

        batch_atoms_degree = []
        for i, mol in enumerate(molecules):
             current_num_atoms = num_atoms[i]
             if current_num_atoms > 1:
                  real_bonds = mol.bonds[1:current_num_atoms, 1:current_num_atoms]
                  degree_real = (real_bonds > 0).sum(axis=1)
                  degree = np.concatenate(([0], degree_real))
                  padded_degree = np.concatenate((degree, np.full(max_num_atoms - current_num_atoms, fill_value=degree_padding_idx, dtype=int)))
             elif current_num_atoms == 1:
                  padded_degree = np.concatenate(([0], np.full(max_num_atoms - 1, fill_value=degree_padding_idx, dtype=int)))
             else: padded_degree = np.full(max_num_atoms, fill_value=degree_padding_idx, dtype=int)
             batch_atoms_degree.append(padded_degree)
        batch_atoms_degree = np.stack(batch_atoms_degree)

        bonds_list = []
        for i, mol in enumerate(molecules):
            current_num_atoms = num_atoms[i]
            if current_num_atoms > 0:
                 padded_bonds = np.pad(mol.bonds, [(0, max_num_atoms - current_num_atoms), (0, max_num_atoms - current_num_atoms)], mode="constant", constant_values=bond_padding_idx)
                 np.fill_diagonal(padded_bonds, bond_padding_idx)
            else: padded_bonds = np.full((max_num_atoms, max_num_atoms), fill_value=bond_padding_idx, dtype=int)
            bonds_list.append(padded_bonds)
        batch_bonds = np.stack(bonds_list)

        # --- Removed distance_matrices_list and batch_topological_distance ---

        additive_padding_masks = []
        for i, mol in enumerate(molecules):
             current_num_atoms = num_atoms[i]
             if current_num_atoms > 0:
                  mask = np.zeros((current_num_atoms, current_num_atoms), dtype=float)
                  padded_mask = np.pad(mask, [(0, max_num_atoms - current_num_atoms), (0, max_num_atoms - current_num_atoms)], mode="constant", constant_values=-np.inf)
                  np.fill_diagonal(padded_mask, 0)
             else:
                  padded_mask = np.full((max_num_atoms, max_num_atoms), fill_value=-np.inf, dtype=float)
                  np.fill_diagonal(padded_mask, 0)
             additive_padding_masks.append(padded_mask)
        batch_additive_padding_attn_mask = np.stack(additive_padding_masks)

        return_dict = dict(
            level_idx=torch.tensor(batch_level_idx, dtype=torch.long, device=device),
            picked_atom_mhe=torch.from_numpy(batch_picked_atom_mhe).long().to(device),
            num_atoms=torch.tensor(num_atoms, dtype=torch.long, device=device),
            atoms=torch.from_numpy(batch_atoms).long().to(device),
            atoms_degree=torch.from_numpy(batch_atoms_degree).long().to(device),
            bonds=torch.from_numpy(batch_bonds).long().to(device),
            # --- Removed topological_distance ---
            additive_padding_attn_mask=torch.from_numpy(batch_additive_padding_attn_mask).float().to(device),
        )

        if include_feasibility_masks:
            # Feasibility mask padding logic (remains the same structure)
            masks_l0, masks_l1, masks_l2 = [], [], []
            max_actions_l0, max_actions_l1, max_actions_l2 = 0, 0, 0
            for mol in molecules:
                 num_real = len(mol.atoms) - 1
                 max_actions_l0 = max(max_actions_l0, 1 + num_real)
                 max_actions_l1 = max(max_actions_l1, mol.vocab_size + num_real + 1)
                 max_actions_l2 = max(max_actions_l2, mol.vocab_size + 1, 7)
            for mol in molecules:
                 num_real = len(mol.atoms) - 1
                 mask_l0 = mol.current_action_mask if mol.current_action_level == 0 and mol.current_action_mask is not None else np.ones(1 + num_real, dtype=bool)
                 padded_mask_l0 = np.pad(mask_l0, (0, max_actions_l0 - len(mask_l0)), mode='constant', constant_values=True)
                 masks_l0.append(padded_mask_l0)
                 mask_l1 = mol.current_action_mask if mol.current_action_level == 1 and mol.current_action_mask is not None else np.ones(mol.vocab_size + num_real + 1, dtype=bool)
                 padded_mask_l1 = np.pad(mask_l1, (0, max_actions_l1 - len(mask_l1)), mode='constant', constant_values=True)
                 masks_l1.append(padded_mask_l1)
                 if mol.current_action_level == 2 and mol.current_action_mask is not None:
                      mask_l2 = mol.current_action_mask
                      padded_mask_l2 = np.pad(mask_l2, (0, max_actions_l2 - len(mask_l2)), mode='constant', constant_values=True)
                 else: padded_mask_l2 = np.ones(max_actions_l2, dtype=bool)
                 masks_l2.append(padded_mask_l2)
            return_dict["feasibility_mask_level_zero"] = torch.from_numpy(np.stack(masks_l0)).bool().to(device)
            return_dict["feasibility_mask_level_one"] = torch.from_numpy(np.stack(masks_l1)).bool().to(device)
            return_dict["feasibility_mask_level_two"] = torch.from_numpy(np.stack(masks_l2)).bool().to(device)

        return return_dict

    @staticmethod
    def batch_to_device(batch: dict, device: torch.device):
        return {k: v.to(device) for k, v in batch.items()}

    @staticmethod
    def get_single_atom_molecules(config: MoleculeConfig, repeat: int = 1) -> List['MoleculeDesign']:
        """Creates initial molecules with single allowed atoms."""
        atoms = []
        for i, atom_name in enumerate(config.atom_vocabulary.keys()):
            if config.atom_vocabulary[atom_name]["allowed"]:
                atoms.append(i + 1)
        return MoleculeDesign.init_batch_from_instance_list(config, atoms * repeat)


    @staticmethod
    def from_smiles(config: MoleculeConfig, smiles: str, **kwargs) -> Tuple[
        Optional['MoleculeDesign'], Optional[Dict[int, int]]]:
        """
        Creates a MoleculeDesign instance directly from a SMILES string.
        Handles canonicalization and renumbering before calling from_rdkit_mol.

        Returns the instance and a map from original canonical RDKit indices to internal indices,
        or (None, None) on failure.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: Invalid SMILES input: {smiles}. Returning None.")
            return None, None

        # --- Canonical Renumbering (Crucial for consistency with GED) ---
        # Ensure the molecule processed by from_rdkit_mol has the same atom indices
        # as the one used for GED generation.
        try:
            Chem.SanitizeMol(mol, catchErrors=True)  # Sanitize first
            # Kekulize BEFORE canonical ranking for consistency
            Chem.Kekulize(mol, clearAromaticFlags=True)
            # Renumber atoms based on canonical rank
            canonical_order = rdmolfiles.CanonicalRankAtoms(mol)
            mol = rdmolops.RenumberAtoms(mol, canonical_order)
            # Ensure sanitization again after potential changes? Might not be needed.
            # Chem.SanitizeMol(mol, catchErrors=True)
        except Exception as e:
            print(f"Warning: Could not sanitize/kekulize/canonically renumber input SMILES {smiles}: {e}")
            # Fail if preprocessing fails, as map consistency is critical
            return None, None

        # Call the simplified from_rdkit_mol
        try:
            # Pass the preprocessed mol
            design_instance, rdkit_map = MoleculeDesign.from_rdkit_mol(
                config, mol, smiles=smiles  # Pass smiles for logging if needed
            )
        except Exception as e:
            print(f"Error during from_rdkit_mol execution for {smiles}: {e}")
            import traceback;
            traceback.print_exc()  # Uncomment for detailed debug
            return None, None  # Return None for both on error

        return design_instance, rdkit_map

    @staticmethod
    def from_rdkit_mol(config: MoleculeConfig, rdkit_mol: Chem.Mol, smiles: Optional[str] = None) -> Tuple[
        Optional['MoleculeDesign'], Optional[Dict[int, int]]]:
        """
        Creates a MoleculeDesign instance directly from an RDKit molecule
        by constructing the internal state (atoms, bonds) without simulating actions.

        Assumes input rdkit_mol has been appropriately preprocessed
        (e.g., Kekulized, Canonically Renumbered) before calling this method.

        Returns the instance and a map from the RDKit indices of the input molecule
        to the internal indices of the created instance, or (None, None) on failure.
        """
        BOND_TYPE_TO_RL_ORDER = {
            Chem.BondType.SINGLE: 1,
            Chem.BondType.DOUBLE: 2,
            Chem.BondType.TRIPLE: 3,
            Chem.BondType.QUADRUPLE: 4,
            Chem.BondType.QUINTUPLE: 5,
            Chem.BondType.HEXTUPLE: 6,
        }

        # 1. Preprocessing (Remove Hs, assume already Kekulized/Renumbered)
        try:
            mol_copy = Chem.RemoveHs(rdkit_mol, sanitize=False)  # Keep H removal
            num_heavy_atoms = mol_copy.GetNumAtoms()
            if num_heavy_atoms == 0:
                print(f"Warning: Input molecule {smiles or ''} has no heavy atoms. Creating empty design.")
                # Handle empty molecule case: Create an instance with only virtual atom?
                instance = MoleculeDesign(config, initial_atom=1)  # Need a valid initial atom
                instance.atoms = np.array([0], dtype=np.uint8)  # Override to be empty
                instance.bonds = np.zeros((1, 1), dtype=np.uint8)
                instance.update_action_mask()  # Update mask for empty state
                return instance, {}  # Return empty map

        except Exception as e:
            print(f"Error during preprocessing in from_rdkit_mol for {smiles or ''}: {e}")
            return None, None

        # 2. Build Atom List and Index Map
        # Get reverse lookup: (atomic_num, charge, chiral) -> vocab_idx
        # This helper function should be defined elsewhere or passed in
        try:
            reverse_atom_lookup = build_reverse_atom_lookup(config)
        except NameError:
            print("Error: build_reverse_atom_lookup helper function not found.")
            return None, None

        internal_atoms_list = [0]  # Start with virtual atom
        rdkit_to_internal_map = {}
        internal_idx_counter = 1  # Internal indices start from 1

        for atom in mol_copy.GetAtoms():
            rdkit_idx = atom.GetIdx()
            atomic_num = atom.GetAtomicNum()
            charge = atom.GetFormalCharge()
            chiral = int(atom.GetChiralTag())

            # Find corresponding vocabulary index
            key = (atomic_num, charge, chiral)
            vocab_idx = reverse_atom_lookup.get(key)
            if vocab_idx is None and chiral != 0:  # Try without chirality if specific chiral not found
                key_no_chiral = (atomic_num, charge, 0)
                vocab_idx = reverse_atom_lookup.get(key_no_chiral)

            if vocab_idx is None:
                print(f"Error: Atom type (Num={atomic_num}, Charge={charge}, Chiral={chiral}) "
                      f"in molecule {smiles or ''} not found in vocabulary config.")
                return None, None  # Cannot proceed if atom is not representable

            # Check if atom type is allowed (optional, mask handles it later)
            # atom_name = config.atom_vocabulary_names[vocab_idx - 1] # Assuming this attr exists
            # if not config.atom_vocabulary[atom_name]["allowed"]:
            #     print(f"Error: Atom type {atom_name} is not allowed by config.")
            #     return None, None

            internal_atoms_list.append(vocab_idx)
            rdkit_to_internal_map[rdkit_idx] = internal_idx_counter
            internal_idx_counter += 1

        # 3. Build Bond Matrix
        num_total_atoms = len(internal_atoms_list)  # Includes virtual atom
        internal_bonds_matrix = np.zeros((num_total_atoms, num_total_atoms), dtype=np.uint8)

        for bond in mol_copy.GetBonds():
            rdkit_idx1 = bond.GetBeginAtomIdx()
            rdkit_idx2 = bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()

            # Map bond type to RL order (1-6),
            rl_order = BOND_TYPE_TO_RL_ORDER.get(bond_type)

            if rl_order is None:
                # Handle unsupported bond types (e.g., Aromatic if not Kekulized, Other)
                print(f"Warning: Unsupported bond type {bond_type} found in {smiles or ''}. Skipping bond.")
                continue

            # Get corresponding internal indices
            try:
                internal_idx1 = rdkit_to_internal_map[rdkit_idx1]
                internal_idx2 = rdkit_to_internal_map[rdkit_idx2]
            except KeyError:
                # This shouldn't happen if atom mapping worked correctly
                print(f"Error: RDKit index mapping failed for bond ({rdkit_idx1}, {rdkit_idx2}).")
                return None, None

            # Set bond order symmetrically
            internal_bonds_matrix[internal_idx1, internal_idx2] = rl_order
            internal_bonds_matrix[internal_idx2, internal_idx1] = rl_order

        # 4. Add Virtual Bonds
        if num_total_atoms > 1:
            virtual_bond_val = 7  # Get from config or class attribute
            internal_bonds_matrix[0, 1:] = virtual_bond_val
            internal_bonds_matrix[1:, 0] = virtual_bond_val

        # 5. Create Instance and Set State
        try:
            # Create a base instance - requires a valid initial atom, even if overridden
            # Find the first *allowed* atom in the vocab as a fallback initial atom
            first_allowed_atom_idx = 1
            for i, name in enumerate(config.atom_vocabulary.keys()):
                if config.atom_vocabulary[name]["allowed"]:
                    first_allowed_atom_idx = i + 1
                    break

            instance = MoleculeDesign(config, initial_atom=first_allowed_atom_idx)

            # Directly set the constructed state
            instance.atoms = np.array(internal_atoms_list, dtype=np.uint8)
            instance.bonds = internal_bonds_matrix

            # Initialize other relevant state attributes
            instance.synthesis_done = False
            instance.smiles_string = None  # Not finalized yet
            instance.objective = None
            instance.infeasibility_flag = False
            instance.current_action_level = 0
            instance.history = []  # No action history generated here
            instance.l0_selected_atom_idx = None
            # ... reset other action state variables ...
            instance.is_modifying_atom = False
            instance.atom_to_modify = None
            instance.l1_new_atom_type = None
            instance.l1_selected_existing_atom_idx = None

            # Check connectivity and update initial mask
            instance._check_and_update_connectivity()  # Uses NetworkX on new state
            instance.update_action_mask()

        except Exception as e:
            print(f"Error creating/setting state for MoleculeDesign instance for {smiles or ''}: {e}")
            import traceback;
            traceback.print_exc()
            return None, None

        # 6. Return instance and map
        return instance, rdkit_to_internal_map