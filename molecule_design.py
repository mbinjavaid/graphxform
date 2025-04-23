import copy
import numpy as np
import torch
from torch import nn
from rdkit import Chem, RDLogger
from rdkit.Chem import rdmolfiles, rdmolops
import networkx as nx
# import sys # For warnings/debug

# import traceback # Keep commented unless debugging
from config import MoleculeConfig
from core.abstracts import BaseTrajectory

from typing import List, Tuple, Dict, Optional

# Suppress RDKit warnings if desired
# RDLogger.DisableLog('rdApp.*')


class ActionType:
    """Enum-like class for Level 1 action types."""
    ADD_ATOM = 1
    SELECT_EXISTING_ATOM = 2
    REMOVE_SELECTED_ATOM = 3


def build_reverse_atom_lookup(config: MoleculeConfig) -> Dict[Tuple[int, int, int], int]:
    """
    Creates a lookup dictionary mapping atom properties back to vocabulary indices.
    (Implementation unchanged from previous version)
    """
    lookup = {}
    if hasattr(config, 'vocabulary_atom_names'):
        vocab_names = config.vocabulary_atom_names
    else:
        vocab_names = list(config.atom_vocabulary.keys())

    if not vocab_names:
        raise ValueError("Atom vocabulary in config appears empty.")

    for i, name in enumerate(vocab_names):
        try:
            atom_config = config.atom_vocabulary[name]
        except KeyError:
            raise KeyError(f"Atom name '{name}' found in vocab_names but not in atom_vocabulary keys.")

        try:
            atomic_num = atom_config['atomic_number']
            charge = atom_config.get('formal_charge', 0)
            chiral = atom_config.get('chiral_tag', 0)
        except KeyError as e:
            raise ValueError(f"Missing expected property {e} for atom '{name}' in config.")

        key = (atomic_num, charge, chiral)
        vocab_idx = i + 1  # 1-based index

        if key in lookup:
            pass # Allow overwriting, assumes config is consistent if duplicates exist
        lookup[key] = vocab_idx

        # Also add a non-chiral version if a chiral one exists, pointing to the same index
        if chiral != 0:
            key_no_chiral = (atomic_num, charge, 0)
            if key_no_chiral not in lookup:
                lookup[key_no_chiral] = vocab_idx

    if not lookup:
        raise ValueError("Reverse atom lookup is empty. Check atom_vocabulary in config.")

    return lookup


class MoleculeDesign(BaseTrajectory):
    """
    Environment for molecular design with cycle prevention (state hashing),
    fragmentation prevention (masking), and rules to guide generation.

    Rule 1: Only atoms present in the initial molecule can be removed.
    Rule 2: Immediate reversal of bond actions on the same atom pair is forbidden.
    Rule 3: Actions leading to molecule fragmentation are forbidden (masked).

    State Management: Internal NumPy arrays (self.atoms, self.bonds) are the primary source of truth.
                      NetworkX used for fragmentation checks during masking.
                      RDKit Mol object is constructed only during finalize() or to_smiles().

    Action Levels (Revised):
        - Level 0: Terminate (if >0 atoms) or Select Existing Atom.
        - Level 1: Add New Atom, Select Existing Atom for Bond, or Remove Selected Original Atom (if allowed by Rule 1 & 3).
        - Level 2: Set Bond Order 1-6 (creates if 0) or Remove Bond (if allowed by Rule 3).
    """
    maximum_bond_order = 6
    virtual_bond_idx = 7
    bond_types = {
        1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE,
        4: Chem.rdchem.BondType.QUADRUPLE, 5: Chem.rdchem.BondType.QUINTUPLE, 6: Chem.rdchem.BondType.HEXTUPLE
    }

    def __init__(self, config: MoleculeConfig, initial_atom: int):
        self.config = config
        self.atom_vocabulary = self.config.atom_vocabulary
        self.vocabulary_atom_idcs = list(range(1, len(self.atom_vocabulary) + 1))
        self.vocabulary_atom_names = list(self.atom_vocabulary.keys())
        self.vocabulary_valence = [-1] * (len(self.vocabulary_atom_names) + 1)
        for i, name in enumerate(self.vocabulary_atom_names):
             self.vocabulary_valence[i+1] = self.atom_vocabulary[name]["valence"]

        self.atom_feasibility_mask = [not self.atom_vocabulary[x]["allowed"] for x in self.vocabulary_atom_names]
        self.vocab_size = len(self.vocabulary_atom_idcs)
        self.upper_limit_atoms = self.config.max_num_atoms

        if not (initial_atom in self.vocabulary_atom_idcs and not self.atom_feasibility_mask[initial_atom - 1]):
             raise ValueError(f"Initial atom {initial_atom} must be in vocabulary {self.vocabulary_atom_idcs} and allowed in config.")
        self.initial_atom = initial_atom

        # --- Internal State (Primary) ---
        self.atoms = np.array([0, initial_atom], dtype=np.uint8) # Includes virtual atom 0
        self.bonds = np.zeros((2, 2), dtype=np.uint8)
        self.bonds[0, 1] = self.bonds[1, 0] = self.virtual_bond_idx # Virtual connection
        # --- Rule 1 State ---
        self.is_original_atom = np.array([False, True], dtype=bool) # Track original atoms

        # --- Trajectory State ---
        self.synthesis_done = False
        self._cached_smiles: Optional[str] = None
        self._cached_rdkit_mol: Optional[Chem.Mol] = None
        self.objective: Optional[float] = None
        self.sa_score: float = 0.
        self.infeasibility_flag: bool = False
        # REMOVED: self.is_currently_connected and self.num_components

        # --- Action Handling State ---
        self.current_action_level = 0
        self.current_action_mask: Optional[np.array] = None
        self.history: List[int] = []
        self.l0_selected_atom_idx: Optional[int] = None # 1-based internal index
        self.l1_action_type: Optional[ActionType] = None # Stores outcome of L1 for L2 context
        self.l1_new_atom_type: Optional[int] = None # 1-based vocab index (if L1 was Add Atom)
        self.l1_selected_existing_atom_idx: Optional[int] = None # 1-based internal index (if L1 was Select Existing)
        # --- Rule 2 State ---
        self.last_bond_action_details: Optional[Tuple[int, int]] = None # Stores (min_idx, max_idx) of last bond action pair

        self.update_action_mask() # Initial mask calculation

    def _get_smiles_for_check(self) -> Optional[str]:
        """
        Generates a canonical SMILES string for intermediate checks WITHOUT
        calling finalize() or modifying internal state caches/flags.
        Returns None if SMILES generation or sanitization fails.
        (Implementation unchanged)
        """
        try:
            # Create a temporary RDKit mol from current state
            temp_mol = self.to_rdkit_mol(sanitize=False)  # Get unsanitized first
            if temp_mol is None: return None # Handle failure in to_rdkit_mol

            if temp_mol.GetNumAtoms() > 0:
                # Try to sanitize the temporary molecule
                Chem.SanitizeMol(temp_mol)  # Raises exception on failure
                # Get canonical SMILES from the sanitized temporary mol
                smiles = Chem.MolToSmiles(temp_mol, canonical=True)
                return smiles
            else:
                return ""  # Empty molecule
        except Exception:
            # If to_rdkit_mol, SanitizeMol, or MolToSmiles fails
            return None

    # REMOVED: _check_and_update_connectivity method is no longer needed.

    def _get_current_valence_usage(self) -> np.array:
        """Calculates the sum of explicit bond orders for each real atom from self.bonds."""
        num_real_atoms = len(self.atoms) - 1
        if num_real_atoms <= 0: return np.array([], dtype=int)
        # Slice bonds to exclude virtual atom 0
        real_bonds = self.bonds[1 : num_real_atoms + 1, 1 : num_real_atoms + 1]
        # Sum bond orders, ensuring only valid orders contribute (<= max_bond_order)
        current_explicit_usage = np.sum(real_bonds * (real_bonds <= self.maximum_bond_order), axis=1)
        return current_explicit_usage.astype(int)

    def _get_remaining_valence(self) -> np.array:
        """Calculates remaining valence for each real atom based on self.atoms and self.bonds."""
        num_real_atoms = len(self.atoms) - 1
        if num_real_atoms <= 0: return np.array([], dtype=int)

        current_usage = self._get_current_valence_usage()

        try:
            # Get max valence from config for current atoms
            total_valence = np.array([self.vocabulary_valence[atom_vocab_idx]
                                      for atom_vocab_idx in self.atoms[1:]], dtype=int)
            # Handle potential -1 values if valence wasn't defined (treat as infinite for calc)
            total_valence[total_valence < 0] = 999 # Or some large number
        except IndexError as e:
            raise IndexError(f"Invalid atom vocab index found in self.atoms[1:]: {self.atoms[1:]}. Error: {e}")

        if len(total_valence) != len(current_usage):
             raise RuntimeError(f"Valence calculation mismatch: total_valence ({len(total_valence)}) vs current_usage ({len(current_usage)})")

        remaining = total_valence - current_usage
        remaining = np.maximum(0, remaining) # Valence cannot be negative
        return remaining

    def _check_fragmentation_on_simulated_action(self, action_type: str, **kwargs) -> bool:
        """
        Checks if a simulated action would result in a fragmented molecule.

        Args:
            action_type: 'remove_atom' or 'remove_bond'.
            **kwargs:
                For 'remove_atom': requires 'internal_idx_to_remove' (1-based).
                For 'remove_bond': requires 'internal_idx_A', 'internal_idx_B' (1-based).

        Returns:
            True if the molecule remains connected (or <= 1 atom) after the
                 simulated action, False if it would fragment.
        """
        num_real_atoms_before = len(self.atoms) - 1
        if num_real_atoms_before <= 1:
            # Removing the only atom leaves 0 (connected).
            # Removing a bond in a 2-atom molecule leaves 2 still connected atoms (via virtual node logic, or just by definition).
            # Cannot fragment if 0 or 1 atoms exist.
            return True

        temp_atoms = self.atoms.copy()
        temp_bonds = self.bonds.copy()
        num_real_atoms_after = num_real_atoms_before

        if action_type == 'remove_atom':
            idx_to_remove = kwargs.get('internal_idx_to_remove')
            if idx_to_remove is None or not (1 <= idx_to_remove <= num_real_atoms_before):
                print(f"Warning: Invalid index {idx_to_remove} for simulated atom removal check.")
                return False # Treat as potentially fragmenting if input is bad

            # Simulate removal
            temp_atoms = np.delete(temp_atoms, idx_to_remove)
            temp_bonds = np.delete(np.delete(temp_bonds, idx_to_remove, 0), idx_to_remove, 1)
            num_real_atoms_after -= 1

        elif action_type == 'remove_bond':
            idx_A = kwargs.get('internal_idx_A')
            idx_B = kwargs.get('internal_idx_B')
            if (idx_A is None or idx_B is None or
                    not (1 <= idx_A <= num_real_atoms_before) or
                    not (1 <= idx_B <= num_real_atoms_before) or
                    idx_A == idx_B):
                 print(f"Warning: Invalid indices ({idx_A}, {idx_B}) for simulated bond removal check.")
                 return False # Treat as potentially fragmenting

            # Simulate removal (set bond order to 0)
            temp_bonds[idx_A, idx_B] = temp_bonds[idx_B, idx_A] = 0
            # num_real_atoms_after remains the same

        else:
            raise ValueError(f"Unknown action_type '{action_type}' for fragmentation check.")

        # --- Check connectivity on the simulated state ---
        if num_real_atoms_after <= 1:
            return True # 0 or 1 atom is considered connected

        G = nx.Graph()
        # Add nodes using 1-based indices based on the *remaining* atoms
        G.add_nodes_from(range(1, num_real_atoms_after + 1))

        # Extract adjacency matrix for real atoms only from temp_bonds
        adj_matrix = temp_bonds[1 : num_real_atoms_after + 1, 1 : num_real_atoms_after + 1]

        rows, cols = np.where(adj_matrix > 0)
        # Edges use 1-based node indices
        edges = list(zip(rows + 1, cols + 1))
        G.add_edges_from(edges)

        try:
            # Check connectivity of the simulated graph
            if not G: # Handle case where graph might become empty unexpectedly
                return True
            is_connected_after = nx.is_connected(G)
            return is_connected_after
        except Exception as e:
            # If NetworkX fails on the simulated graph, assume it might fragment
            print(f"Warning: NetworkX check failed during simulation: {e}")
            return False

    def update_action_mask(self):
        """Creates the action mask, incorporating Rule 1, 2, and fragmentation checks."""
        if self.synthesis_done:
            self.current_action_mask = None
            return

        num_real_atoms = len(self.atoms) - 1
        remaining_valence = self._get_remaining_valence() # Can raise IndexError

        if self.current_action_level == 0:
            action_space_size = 1 + num_real_atoms
            mask = np.zeros(action_space_size, dtype=bool)
            # Mask terminate ONLY if no real atoms exist
            if num_real_atoms <= 0:
                mask[0] = True
            # Mask atom selections if no atoms exist
            if num_real_atoms == 0:
                 mask[1:] = True
            self.current_action_mask = mask

        elif self.current_action_level == 1:
            action_space_size = self.vocab_size + num_real_atoms + 1
            mask = np.ones(action_space_size, dtype=bool) # Start with all masked

            anchor_atom_internal_idx = self.l0_selected_atom_idx
            if anchor_atom_internal_idx is None or anchor_atom_internal_idx <= 0 or anchor_atom_internal_idx > num_real_atoms:
                raise ValueError(f"L1 Mask Error: Invalid anchor atom index: {anchor_atom_internal_idx} (NumReal={num_real_atoms})")
            anchor_atom_0_idx = anchor_atom_internal_idx - 1 # 0-based index for valence array

            # --- Unmask "Add Atom" ---
            # Condition: Can add atom if not exceeding limit AND anchor has valence
            if self.upper_limit_atoms is None or num_real_atoms < self.upper_limit_atoms:
                # Check anchor valence bounds first
                if anchor_atom_0_idx < len(remaining_valence) and remaining_valence[anchor_atom_0_idx] > 0:
                    for i in range(self.vocab_size):
                        action_idx = i
                        atom_type_vocab_idx = i + 1
                        # Check if atom type is allowed and has valence >= 1
                        if not self.atom_feasibility_mask[i] and self.vocabulary_valence[atom_type_vocab_idx] >= 1:
                            mask[action_idx] = False # Unmask Add Atom action

            # --- Unmask "Select Existing Atom" ---
            # Condition: Can select if target is different from anchor AND (bond exists OR both have valence)
            if anchor_atom_0_idx < len(remaining_valence): # Check anchor index validity
                for target_0_idx in range(num_real_atoms):
                    target_internal_idx = target_0_idx + 1
                    action_idx = self.vocab_size + target_0_idx
                    if target_internal_idx == anchor_atom_internal_idx: continue # Cannot select self

                    # Check target index validity
                    if target_0_idx >= len(remaining_valence):
                        # This indicates a state inconsistency if reached
                        raise IndexError(f"L1 Mask Error: Target index {target_0_idx} OOB for rem_val (len {len(remaining_valence)})")

                    bond_exists = self.bonds[anchor_atom_internal_idx, target_internal_idx] > 0
                    target_has_valence = remaining_valence[target_0_idx] > 0
                    anchor_has_valence = remaining_valence[anchor_atom_0_idx] > 0
                    # Unmask if bond exists OR if both have valence for potential new bond
                    if bond_exists or (target_has_valence and anchor_has_valence):
                         mask[action_idx] = False

            # --- Unmask "Remove Selected Atom" (Rule 1 + Fragmentation Check) ---
            remove_action_idx = self.vocab_size + num_real_atoms
            # Check index bounds before accessing is_original_atom
            if anchor_atom_internal_idx < len(self.is_original_atom):
                # Conditions: More than 1 atom exists, atom is original, AND removal doesn't fragment
                if num_real_atoms > 1 and self.is_original_atom[anchor_atom_internal_idx]:
                    # *** Fragmentation Check ***
                    would_remain_connected = self._check_fragmentation_on_simulated_action(
                        action_type='remove_atom',
                        internal_idx_to_remove=anchor_atom_internal_idx
                    )
                    if would_remain_connected:
                        mask[remove_action_idx] = False # Unmask only if it wouldn't fragment
            else:
                # This indicates a state inconsistency
                raise IndexError(f"L1 Mask Error: anchor_atom_internal_idx {anchor_atom_internal_idx} OOB for is_original_atom (len {len(self.is_original_atom)})")

            self.current_action_mask = mask

        elif self.current_action_level == 2:
            action_space_size = 7 # 0:BondOrder1, ..., 5:BondOrder6, 6:RemoveBond
            mask = np.ones(action_space_size, dtype=bool) # Start all masked

            atom_A_internal_idx = self.l0_selected_atom_idx
            atom_B_internal_idx = -1 # Determine B based on L1 action type
            if self.l1_action_type == ActionType.ADD_ATOM:
                # The new atom is the last one in the current atoms array
                atom_B_internal_idx = len(self.atoms) - 1
            elif self.l1_action_type == ActionType.SELECT_EXISTING_ATOM:
                atom_B_internal_idx = self.l1_selected_existing_atom_idx
            else:
                # Should not reach L2 if L1 was REMOVE_ATOM
                raise RuntimeError(f"L2 Mask Error: Invalid L1 action type context ({self.l1_action_type}).")

            # Validate indices A and B
            if (atom_A_internal_idx is None or atom_A_internal_idx <= 0 or atom_A_internal_idx > num_real_atoms or
                    atom_B_internal_idx <= 0 or atom_B_internal_idx > num_real_atoms or atom_A_internal_idx == atom_B_internal_idx):
                raise ValueError(f"L2 Mask Error: Invalid indices A={atom_A_internal_idx}, B={atom_B_internal_idx} (NumReal={num_real_atoms})")

            # --- Rule 2 Check (Prevent immediate reversal) ---
            current_min_idx = min(atom_A_internal_idx, atom_B_internal_idx)
            current_max_idx = max(atom_A_internal_idx, atom_B_internal_idx)
            if self.last_bond_action_details is not None and \
               self.last_bond_action_details[0] == current_min_idx and \
               self.last_bond_action_details[1] == current_max_idx:
                # Last action was a bond action on this same pair. Mask ALL L2 actions.
                mask[:] = True # Mask all 7 actions
                self.current_action_mask = mask
                return # Skip normal valence/fragmentation checks

            # --- Normal L2 Mask Logic (If Rule 2 check passes) ---
            atom_A_0_idx = atom_A_internal_idx - 1
            atom_B_0_idx = atom_B_internal_idx - 1

            # Check index validity for valence array
            if atom_A_0_idx >= len(remaining_valence) or atom_B_0_idx >= len(remaining_valence):
                raise IndexError(f"L2 Mask Error: Indices {atom_A_0_idx} or {atom_B_0_idx} OOB for rem_val (len {len(remaining_valence)}).")

            current_bond_order = self.bonds[atom_A_internal_idx, atom_B_internal_idx]
            valence_A_rem = remaining_valence[atom_A_0_idx]
            valence_B_rem = remaining_valence[atom_B_0_idx]

            # Calculate max possible increase based on remaining valence
            max_increase = min(valence_A_rem, valence_B_rem)
            # Effective current order (0 if no bond exists)
            effective_current_order = int(current_bond_order) if current_bond_order > 0 and current_bond_order <= self.maximum_bond_order else 0

            # Max final order allowed by valence and config limit
            max_allowed_final_order = min(effective_current_order + max_increase, self.maximum_bond_order)

            # Unmask "Set Bond Order" actions based on valence
            for order in range(1, self.maximum_bond_order + 1):
                action_idx = order - 1
                if order <= max_allowed_final_order:
                    mask[action_idx] = False

            # --- Unmask "Remove Bond" (Fragmentation Check) ---
            remove_bond_action_idx = 6
            # Condition: Bond must exist AND removal doesn't fragment
            if current_bond_order > 0 and current_bond_order <= self.maximum_bond_order: # Only consider removing if a valid bond exists
                # *** Fragmentation Check ***
                would_remain_connected = self._check_fragmentation_on_simulated_action(
                    action_type='remove_bond',
                    internal_idx_A=atom_A_internal_idx,
                    internal_idx_B=atom_B_internal_idx
                )
                if would_remain_connected:
                    mask[remove_bond_action_idx] = False # Unmask only if it wouldn't fragment

            self.current_action_mask = mask
        else:
            raise ValueError(f"Invalid current_action_level: {self.current_action_level}")

    def _adjust_indices_after_removal(self, removed_internal_idx: int):
        """Adjusts stored internal indices after an atom removal."""
        # Adjust L0 context if needed
        if self.l0_selected_atom_idx is not None and self.l0_selected_atom_idx > removed_internal_idx:
            self.l0_selected_atom_idx -= 1
        # Adjust L1 context if needed (only if L1 was Select Existing)
        if self.l1_action_type == ActionType.SELECT_EXISTING_ATOM:
            if self.l1_selected_existing_atom_idx is not None and self.l1_selected_existing_atom_idx > removed_internal_idx:
                self.l1_selected_existing_atom_idx -= 1

        # Adjust Rule 2 context (last_bond_action_details)
        if self.last_bond_action_details is not None:
            min_idx, max_idx = self.last_bond_action_details
            # Check if the removed atom was part of the last bond action pair
            if min_idx == removed_internal_idx or max_idx == removed_internal_idx:
                self.last_bond_action_details = None # Invalidate if removed atom was involved
            else:
                # Adjust indices if they were after the removed atom
                new_min = min_idx - 1 if min_idx > removed_internal_idx else min_idx
                new_max = max_idx - 1 if max_idx > removed_internal_idx else max_idx
                # Update only if indices actually changed
                if new_min != min_idx or new_max != max_idx:
                     self.last_bond_action_details = (new_min, new_max)

    def take_action(self, action: int):
        """Execute a given action, updating internal state directly."""
        if self.synthesis_done:
            raise RuntimeError("Cannot take action on terminated design.")

        # Check mask validity
        if self.current_action_mask is None or action < 0 or action >= len(self.current_action_mask) or \
                self.current_action_mask[action]:
            mask_len = "None" if self.current_action_mask is None else len(self.current_action_mask)
            raise ValueError(
                f"Action {action} masked or invalid for level {self.current_action_level}. MaskLen={mask_len}")

        current_level = self.current_action_level
        next_level = 0 # Default next level
        self.history.append(int(action))
        num_real_atoms_before = len(self.atoms) - 1

        try:
            reset_last_bond_action = True # Default: reset Rule 2 tracker unless it's an L2 bond action

            # --- Level 0 Actions ---
            if current_level == 0:
                if action == 0:  # Terminate
                    self.synthesis_done = True
                    self.finalize(assert_feasible=False) # Finalize internal state
                    next_level = -1 # Indicate termination
                else:  # Select Atom
                    selected_idx = action
                    if not (1 <= selected_idx <= num_real_atoms_before):
                        raise ValueError(f"L0 Select Atom: Invalid index {selected_idx}.")
                    self.l0_selected_atom_idx = selected_idx
                    # Reset L1 context
                    self.l1_action_type = None
                    self.l1_new_atom_type = None
                    self.l1_selected_existing_atom_idx = None
                    next_level = 1

            # --- Level 1 Actions ---
            elif current_level == 1:
                remove_action_idx = self.vocab_size + num_real_atoms_before
                anchor_idx = self.l0_selected_atom_idx
                if anchor_idx is None: # Should be caught by masking, but defensive
                    raise RuntimeError("L1 take_action: l0_selected_atom_idx is None.")

                if action < self.vocab_size:  # Add Atom
                    self.l1_action_type = ActionType.ADD_ATOM
                    self.l1_new_atom_type = action + 1 # Store vocab index
                    # Append new atom and expand bonds matrix
                    self.atoms = np.append(self.atoms, self.l1_new_atom_type)
                    new_size = len(self.atoms)
                    new_idx = new_size - 1
                    self.bonds = np.pad(self.bonds, [(0, 1), (0, 1)], 'constant', constant_values=0)
                    # Add virtual bond connection
                    self.bonds[0, new_idx] = self.bonds[new_idx, 0] = self.virtual_bond_idx
                    # Update original atom tracker
                    self.is_original_atom = np.append(self.is_original_atom, False)
                    next_level = 2
                elif action < remove_action_idx:  # Select Existing Atom
                    selected_internal_idx = (action - self.vocab_size) + 1
                    # Mask should prevent selecting anchor, but double check
                    if selected_internal_idx == anchor_idx:
                        raise ValueError("L1 Select Existing: Cannot select anchor.")
                    if not (1 <= selected_internal_idx <= num_real_atoms_before):
                        raise ValueError(f"L1 Select Existing: Invalid target index {selected_internal_idx}.")
                    # Store context for L2
                    self.l1_action_type = ActionType.SELECT_EXISTING_ATOM
                    self.l1_selected_existing_atom_idx = selected_internal_idx
                    next_level = 2
                elif action == remove_action_idx:  # Remove Selected Atom (Anchor)
                    # Mask should prevent removing last atom or non-original, but double check
                    if num_real_atoms_before <= 1:
                        raise RuntimeError("Attempted to remove last real atom.")
                    if not self.is_original_atom[anchor_idx]:
                        raise RuntimeError("Attempted to remove non-original atom.")
                    # Store context (though not strictly needed as we go to L0)
                    self.l1_action_type = ActionType.REMOVE_SELECTED_ATOM
                    removed_idx_for_adjust = anchor_idx
                    # Perform removal
                    self.atoms = np.delete(self.atoms, removed_idx_for_adjust)
                    self.bonds = np.delete(np.delete(self.bonds, removed_idx_for_adjust, 0), removed_idx_for_adjust, 1)
                    self.is_original_atom = np.delete(self.is_original_atom, removed_idx_for_adjust)
                    # Adjust any stored indices > removed index
                    self._adjust_indices_after_removal(removed_idx_for_adjust)
                    # Clear L0/L1 context as we return to L0
                    self.l0_selected_atom_idx = None
                    self.l1_new_atom_type = None
                    self.l1_selected_existing_atom_idx = None
                    self.l1_action_type = None
                    next_level = 0  # Return to Level 0
                else:
                    raise ValueError(f"Invalid L1 action index: {action}")

            # --- Level 2 Actions ---
            elif current_level == 2:
                reset_last_bond_action = False # This IS a bond action, update Rule 2 tracker
                idx_A = self.l0_selected_atom_idx
                idx_B = -1 # Determine B based on L1 context stored
                if self.l1_action_type == ActionType.ADD_ATOM:
                    idx_B = len(self.atoms) - 1 # The newly added atom
                elif self.l1_action_type == ActionType.SELECT_EXISTING_ATOM:
                    idx_B = self.l1_selected_existing_atom_idx
                else: # Should not happen
                    raise RuntimeError(f"L2 take_action: Invalid L1 context ({self.l1_action_type}).")

                # Validate indices obtained from context
                current_num_real_atoms = len(self.atoms) - 1
                if (idx_A is None or idx_A <= 0 or idx_A > current_num_real_atoms or
                        idx_B <= 0 or idx_B > current_num_real_atoms or idx_A == idx_B):
                     raise ValueError(f"L2 take_action: Invalid indices A={idx_A}, B={idx_B}.")

                # Perform bond modification
                if action <= 5: # Set Order (action 0 -> order 1, ..., action 5 -> order 6)
                    order = action + 1
                    self.bonds[idx_A, idx_B] = self.bonds[idx_B, idx_A] = order
                elif action == 6: # Remove Bond (set order to 0)
                    # Mask should prevent this if it causes fragmentation
                    self.bonds[idx_A, idx_B] = self.bonds[idx_B, idx_A] = 0
                else:
                    raise ValueError(f"Invalid L2 Bond action index: {action}")

                # Rule 2 Update: Store the pair involved in this bond action
                self.last_bond_action_details = (min(idx_A, idx_B), max(idx_A, idx_B))

                # Clear L0/L1 context as we return to L0
                self.l0_selected_atom_idx = None
                self.l1_new_atom_type = None
                self.l1_selected_existing_atom_idx = None
                self.l1_action_type = None
                next_level = 0 # Return to Level 0

            # --- Reset Rule 2 tracker if the action was NOT a bond modification ---
            if reset_last_bond_action:
                self.last_bond_action_details = None

            # --- Update Mask and Level for Next Step ---
            if next_level != -1: # If not terminated
                self.current_action_level = next_level
                # Update mask based on the new state and level (can raise errors)
                self.update_action_mask()
            else: # Terminated
                self.current_action_mask = None # No more actions possible

        # --- Exception Handling during Action Execution ---
        except (ValueError, IndexError) as e:
            # Errors related to action logic, indexing, masking. Mark as infeasible.
            self.infeasibility_flag = True
            self.current_action_mask = None # Stop further actions
            # Do NOT set synthesis_done here, let the caller handle sequence failure
            raise RuntimeError(f"Action logic error at L{current_level}, action {action}: {e}") from e
        except RuntimeError as e:
            # Catch specific RuntimeErrors (e.g., from context issues, or raised above)
            self.infeasibility_flag = True
            self.current_action_mask = None
            # Re-raise the original RuntimeError
            raise e
        except Exception as e:
            # Catch any other unexpected errors (e.g., NumPy issues)
            self.infeasibility_flag = True
            self.current_action_mask = None
            print(f"CRITICAL: Unexpected error in take_action(action={action}, L{current_level}): {e}")
            # traceback.print_exc() # Optional: print full traceback for debugging
            raise RuntimeError(f"Unexpected error: {e}") from e

    def finalize(self, assert_feasible: bool = False):
        """Finalize molecule design: build RDKit mol, sanitize, cache SMILES."""
        # Avoid re-finalizing
        if self._cached_smiles is not None or self._cached_rdkit_mol is not None:
             return

        # REMOVED: Connectivity check here is redundant due to masking.

        # Optional feasibility assertion
        if assert_feasible:
            try:
                self.assert_feasible()
            except AssertionError as e:
                print(f"Warning: Feasibility assertion failed during finalize: {e}")
                self.infeasibility_flag = True

        num_real_atoms = len(self.atoms) - 1

        # Generate RDKit Mol only if not already marked infeasible
        rdkit_mol = None
        if not self.infeasibility_flag:
            try:
                # Generate RDKit mol from internal state (don't sanitize yet)
                rdkit_mol = self.to_rdkit_mol(sanitize=False)

                if rdkit_mol.GetNumAtoms() == 0 and num_real_atoms > 0:
                    # Internal state has atoms, but RDKit mol is empty - error
                    print("Warning: RDKit mol empty despite internal atoms.")
                    self.infeasibility_flag = True
                elif rdkit_mol.GetNumAtoms() > 0:
                    # Try to sanitize and cache
                    try:
                        # Store copy before sanitization attempt (optional)
                        # self._cached_rdkit_mol = copy.deepcopy(rdkit_mol)
                        Chem.SanitizeMol(rdkit_mol) # Sanitize in-place
                        self._cached_rdkit_mol = rdkit_mol # Store sanitized version
                        self._cached_smiles = Chem.MolToSmiles(rdkit_mol, canonical=True)
                    except Exception as e:
                        # Sanitization or SMILES generation failed
                        print(f"Warning: Final sanitization/SMILES failed: {e}.")
                        self._cached_smiles = None
                        # Keep the unsanitized mol if available, but mark infeasible? Or clear cache?
                        # Let's clear cache and mark infeasible if sanitization fails
                        self._cached_rdkit_mol = None
                        self.infeasibility_flag = True
                else: # Zero atoms internally and in RDKit mol
                    self._cached_smiles = ""
                    self._cached_rdkit_mol = rdkit_mol # Store empty mol object
            except Exception as e:
                 # Error during to_rdkit_mol itself
                 print(f"Warning: Error during RDKit mol generation in finalize: {e}")
                 self.infeasibility_flag = True
                 self._cached_smiles = None
                 self._cached_rdkit_mol = None
        else: # Already infeasible before RDKit generation
            self._cached_smiles = None
            self._cached_rdkit_mol = None

        # Mark synthesis as done regardless of success/failure at this point
        self.synthesis_done = True

    def assert_feasible(self):
        """Check internal state consistency. Raises AssertionError on failure."""
        # Basic type checks
        if not isinstance(self.atoms, np.ndarray) or not isinstance(self.bonds, np.ndarray) or not isinstance(self.is_original_atom, np.ndarray):
             raise AssertionError("Internal state types incorrect.")
        assert self.atoms[0] == 0, "Virtual atom missing/incorrect."

        num_atoms = len(self.atoms); num_real_atoms = num_atoms - 1
        assert len(self.is_original_atom) == num_atoms, f"is_original_atom length mismatch ({len(self.is_original_atom)} vs {num_atoms})."
        assert not self.is_original_atom[0], "Virtual atom marked as original."

        # Atom checks
        if num_real_atoms > 0:
             # Check if all atom indices are valid vocab indices
             valid_indices = all(1 <= idx <= self.vocab_size for idx in self.atoms[1:])
             assert valid_indices, f"Invalid atom vocab index found: {self.atoms[1:]}"
             # Check if all atom types are allowed
             allowed_check = all(not self.atom_feasibility_mask[idx - 1] for idx in self.atoms[1:])
             assert allowed_check, f"Disallowed atom type found: {self.atoms[1:]}"

        # Size limit check
        if self.upper_limit_atoms is not None:
             assert num_real_atoms <= self.upper_limit_atoms, f"Max atoms exceeded ({num_real_atoms} > {self.upper_limit_atoms})."

        # Bond matrix checks
        assert self.bonds.shape == (num_atoms, num_atoms), f"Bonds shape mismatch ({self.bonds.shape} vs ({num_atoms},{num_atoms}))."
        assert not np.any(self.bonds.diagonal()), "Self-loops detected in bond matrix."
        assert np.all(self.bonds == self.bonds.T), "Bond matrix not symmetric."

        # Virtual bond checks
        if num_real_atoms > 0:
             assert np.all(self.bonds[0, 1:] == self.virtual_bond_idx), "Virtual bond missing/incorrect in row 0."
             assert np.all(self.bonds[1:, 0] == self.virtual_bond_idx), "Virtual bond missing/incorrect in col 0."

        # Valence check (only if atoms exist)
        if num_real_atoms > 0:
             try:
                  remaining_valence = self._get_remaining_valence()
                  assert np.all(remaining_valence >= 0), f"Valence constraints violated. Remaining: {remaining_valence}"
             except (IndexError, RuntimeError) as e:
                  # Catch errors during valence calculation itself
                  raise AssertionError(f"Valence check failed during calculation: {e}")

        # REMOVED: Connectivity assertion is implicitly handled by masking.

    def to_rdkit_mol(self, sanitize=True) -> Chem.RWMol:
        """Creates a *new* RDKit molecule from the internal state."""
        mol = Chem.RWMol()
        num_total_atoms = len(self.atoms)
        if num_total_atoms <= 1: return mol # Return empty mol if only virtual atom exists

        rdkit_idx_map = {} # Maps internal index (1-based) to RDKit index (0-based)
        try:
            for internal_idx, atom_vocab_idx in enumerate(self.atoms):
                if internal_idx == 0: continue # Skip virtual atom

                # Validate vocab index
                if not (1 <= atom_vocab_idx <= self.vocab_size):
                     raise ValueError(f"Invalid vocab index {atom_vocab_idx} at internal index {internal_idx}.")

                # Get atom properties from config
                try:
                    atom_name = self.vocabulary_atom_names[atom_vocab_idx - 1]
                    atom_config = self.atom_vocabulary[atom_name]
                except (IndexError, KeyError) as e:
                    raise RuntimeError(f"Cannot get config for vocab index {atom_vocab_idx}: {e}")

                # Create RDKit atom
                a = Chem.Atom(atom_config["atomic_number"])
                if "formal_charge" in atom_config:
                    a.SetFormalCharge(atom_config["formal_charge"])
                # Set chirality
                ct = atom_config.get("chiral_tag", 0)
                if ct == 1: a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
                elif ct == 2: a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
                else: a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED) # Default

                # Add atom to RDKit mol and store mapping
                new_rdkit_idx = mol.AddAtom(a)
                rdkit_idx_map[internal_idx] = new_rdkit_idx

            # Add bonds based on internal bond matrix
            for i in range(1, num_total_atoms): # Start from 1 to skip virtual
                for j in range(i + 1, num_total_atoms): # Avoid self-loops and duplicates
                    bond_order = self.bonds[i, j]

                    # Add bond only if order is valid (1-6)
                    if bond_order > 0 and bond_order <= self.maximum_bond_order:
                        # Get corresponding RDKit indices
                        if i not in rdkit_idx_map or j not in rdkit_idx_map:
                            # This indicates an inconsistency if reached
                            raise RuntimeError(f"Missing RDKit map entry for internal index {i} or {j}.")
                        rdkit_i, rdkit_j = rdkit_idx_map[i], rdkit_idx_map[j]

                        # Get RDKit bond type
                        rdkit_bond_type = self.bond_types.get(int(bond_order))
                        if rdkit_bond_type:
                            mol.AddBond(rdkit_i, rdkit_j, rdkit_bond_type)
                        else:
                            # Should not happen if bond_types map is correct
                            print(f"Warning: Could not find RDKit bond type for order {bond_order}.")
                    elif bond_order > self.maximum_bond_order and bond_order != self.virtual_bond_idx:
                         # Invalid bond order found in matrix (excluding virtual)
                         print(f"Warning: Invalid bond order {bond_order} found between internal atoms {i},{j}.")

        except Exception as e:
             # Catch errors during atom/bond addition
             print(f"Error building RDKit Mol from internal state: {e}")
             # Return potentially partially built mol? Or raise? Let's return empty for safety.
             return Chem.RWMol()


        # Optional sanitization
        if sanitize:
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                # Allow returning unsanitized mol even if sanitization fails, but print warning
                print(f"Warning: RDKit sanitization failed: {e}")
        return mol

    def is_terminable(self):
        """Checks if the current state allows termination."""
        num_real_atoms = len(self.atoms) - 1
        # Can terminate if at Level 0, not already done, AND has at least one real atom
        can_terminate = self.current_action_level == 0 and not self.synthesis_done
        has_atoms = (num_real_atoms > 0)
        # REMOVED: Connectivity check is no longer needed here
        return can_terminate and has_atoms

    def to_smiles(self, canonical: bool = True) -> Optional[str]:
        """Returns a canonical SMILES string. Finalizes if needed. Caches result."""
        # Finalize if not already done (this builds/caches _cached_rdkit_mol)
        if not self.synthesis_done:
             # finalize handles potential errors internally and sets infeasibility_flag
             self.finalize(assert_feasible=False)

        # If canonical SMILES is already cached, return it
        if canonical and self._cached_smiles is not None:
            return self._cached_smiles

        # If RDKit mol object exists (even if unsanitized), try to generate SMILES
        if self._cached_rdkit_mol is not None:
            try:
                # Use a copy to avoid modifying the cached version if sanitization is needed again
                mol_to_use = copy.deepcopy(self._cached_rdkit_mol)
                # Ensure it's sanitized before generating SMILES
                Chem.SanitizeMol(mol_to_use)
                smiles = Chem.MolToSmiles(mol_to_use, canonical=canonical)
                # Cache the canonical SMILES if requested
                if canonical:
                    self._cached_smiles = smiles
                return smiles
            except Exception as e:
                # SMILES generation failed (likely due to sanitization issues)
                print(f"Warning: Failed to generate SMILES (canonical={canonical}): {e}")
                return None # Return None on failure
        else:
            # No RDKit mol available (likely means infeasible or empty)
            # Return cached SMILES (which would be None or "" in this case)
            return self._cached_smiles

    # --- Batching and Static Methods (Largely Unchanged, review context usage) ---

    @staticmethod
    def init_batch_from_instance_list(config: MoleculeConfig, instances: List[int], network: Optional[nn.Module]=None, device: Optional[torch.device]=None):
        """Initializes a batch of MoleculeDesign instances from initial atom vocab indices."""
        # No change needed
        return [MoleculeDesign(config=config, initial_atom=atom) for atom in instances]

    @staticmethod
    def log_probability_fn(trajectories: List['MoleculeDesign'], network: nn.Module) -> List[np.array]:
        """Calculates masked log probabilities for the current action level."""
        # No change needed in logic, relies on current_action_mask
        log_probs_to_return: List[np.array] = []
        network.eval() # Ensure network is in evaluation mode
        device = next(network.parameters()).device # Get device from network

        with torch.no_grad():
            # Convert list of molecules to a batch dictionary suitable for the network
            batch = MoleculeDesign.list_to_batch(molecules=trajectories, device=device)
            # Get logits from the network
            batch_logits_l0, batch_logits_l1, batch_logits_l2 = network(batch)
            # Move logits to CPU for NumPy operations
            batch_logits_l0 = batch_logits_l0.cpu().numpy()
            batch_logits_l1 = batch_logits_l1.cpu().numpy()
            batch_logits_l2 = batch_logits_l2.cpu().numpy()

            for i, mol in enumerate(trajectories):
                mask = mol.current_action_mask
                # If molecule is done or mask is missing, return empty array
                if mask is None:
                    log_probs_to_return.append(np.array([]))
                    continue

                logits = None
                # Select appropriate logits based on current action level
                if mol.current_action_level == 0: logits = batch_logits_l0[i]
                elif mol.current_action_level == 1: logits = batch_logits_l1[i]
                elif mol.current_action_level == 2: logits = batch_logits_l2[i]
                else:
                    # Should not happen with valid levels 0, 1, 2
                    print(f"Warning: Invalid action level {mol.current_action_level} during log_prob calculation.")
                    log_probs_to_return.append(np.array([]))
                    continue

                # Ensure logits and mask have compatible lengths
                mask_len = len(mask)
                if len(logits) > mask_len:
                    logits = logits[:mask_len] # Truncate logits if longer
                elif len(logits) < mask_len:
                    # This indicates a mismatch between network output size and expected action space size
                    raise ValueError(f"Logits/Mask length mismatch L{mol.current_action_level}: {len(logits)} vs {mask_len}")

                # Apply mask and calculate log probabilities (using stable log-softmax approach)
                logits[mask] = -np.inf # Apply mask by setting masked logits to negative infinity
                max_logit = np.max(logits)

                # Check if all actions are masked
                if np.isneginf(max_logit):
                    # If all logits are -inf, log_probs should also be -inf
                    log_probs = logits
                else:
                    # Stable log-softmax calculation
                    exp_logits = np.exp(logits - max_logit)
                    log_sum_exp = np.log(np.sum(exp_logits))
                    log_probs = logits - (max_logit + log_sum_exp)
                    log_probs[mask] = -np.inf # Re-apply mask to log_probs

                log_probs_to_return.append(log_probs)
        return log_probs_to_return

    def transition_fn(self, action: int) -> Tuple['BaseTrajectory', bool]:
        """Creates a copy, takes the action, and returns the new state and termination status."""
        # Deep copy the current state
        copied_molecule = copy.deepcopy(self)
        try:
            # Apply the action to the copied state
            # This call handles internal state updates and can raise errors
            copied_molecule.take_action(action)
        except (ValueError, IndexError) as e:
            # Re-raise errors related to invalid actions (masking should prevent this)
            raise e
        except RuntimeError as e:
            # Catch RuntimeErrors from take_action (e.g., logic errors, unexpected failures)
            # The take_action method should set the infeasibility flag internally on such errors.
            # We don't need to set synthesis_done here, let the caller decide based on the flag.
            print(f"Warning: transition_fn caught RuntimeError in take_action({action}): {e}. Returning resulting state.")
            # The copied_molecule state (potentially marked infeasible) is returned.
        # No generic 'except Exception' needed if take_action handles its errors well

        # Return the new state and whether it's a terminal state (synthesis_done flag)
        return copied_molecule, copied_molecule.synthesis_done

    def to_max_evaluation_fn(self) -> float:
        """Returns the objective value, penalizing infeasible states."""
        # Return negative infinity if objective not set or if marked infeasible
        if self.objective is None or self.infeasibility_flag:
             return float("-inf")
        # Otherwise, return the valid objective value
        return self.objective # Assuming objective is already float

    def num_actions(self) -> int:
        """Returns the number of valid (unmasked) actions at the current level."""
        if self.current_action_mask is None: return 0
        # Count number of False entries in the boolean mask
        return int(np.sum(~self.current_action_mask))

    @staticmethod
    def list_to_batch(molecules: List['MoleculeDesign'], device: torch.device = None,
                      include_feasibility_masks: bool = False) -> dict:
        """Converts a list of MoleculeDesign instances to a batch dictionary."""
        # Logic appears generally sound, relies on instance variables.
        # Check usage of l1_action_type for mhe encoding.
        if not molecules: return {}
        first_mol = molecules[0]
        # Determine padding indices based on vocab/config
        atoms_padding_idx = first_mol.vocab_size + 1
        max_valence = max([-1] + [v for v in first_mol.vocabulary_valence if v is not None and v >= 0])
        degree_padding_idx = max_valence + 2 # Assuming degree includes virtual? Check definition.
        bond_padding_idx = MoleculeDesign.virtual_bond_idx + 1 # Padding for bond matrix

        device = torch.device("cpu") if device is None else device

        num_atoms = [len(mol.atoms) for mol in molecules] # Includes virtual atom
        max_num_atoms = max(num_atoms) if num_atoms else 0
        batch_level_idx = [mol.current_action_level for mol in molecules]

        # --- picked_atom_mhe Encoding ---
        # Size: (batch_size, max_num_atoms)
        # 0: Not picked
        # 1: Picked at Level 0 (Anchor)
        # 2: Picked/Targeted at Level 1 (New atom or Existing atom)
        batch_picked_atom_mhe = np.zeros((len(molecules), max_num_atoms), dtype=int)
        for i, mol in enumerate(molecules):
            # Mark anchor atom (L0 selection)
            anchor_idx = mol.l0_selected_atom_idx # 1-based internal index
            if mol.current_action_level >= 1 and anchor_idx is not None:
                if 0 <= anchor_idx < max_num_atoms: # Check bounds (index is 1-based, array is 0-based)
                    batch_picked_atom_mhe[i, anchor_idx] = 1
                # else: print(f"Warning: Anchor index {anchor_idx} OOB for mhe.") # Debug

            # Mark target atom (L1 outcome) if at Level 2
            if mol.current_action_level == 2:
                target_idx = None
                if mol.l1_action_type == ActionType.ADD_ATOM:
                    # Target is the last atom added (index = num_atoms - 1)
                    target_idx = len(mol.atoms) - 1
                elif mol.l1_action_type == ActionType.SELECT_EXISTING_ATOM:
                    # Target is the one selected in L1
                    target_idx = mol.l1_selected_existing_atom_idx

                if target_idx is not None:
                    if 0 <= target_idx < max_num_atoms: # Check bounds
                        if target_idx != anchor_idx: # Ensure target is not same as anchor
                            batch_picked_atom_mhe[i, target_idx] = 2
                        # else: print(f"Warning: Target index {target_idx} same as anchor {anchor_idx}.") # Debug
                    # else: print(f"Warning: Target index {target_idx} OOB for mhe.") # Debug
        # --- End picked_atom_mhe ---

        # --- Batch Atoms ---
        batch_atoms = np.stack([
            np.pad(mol.atoms, (0, max_num_atoms - num_atoms[i]), mode='constant', constant_values=atoms_padding_idx)
            if num_atoms[i] > 0 else np.full(max_num_atoms, fill_value=atoms_padding_idx, dtype=np.uint8)
            for i, mol in enumerate(molecules)
        ])

        # --- Batch Atom Degrees ---
        batch_atoms_degree = []
        for i, mol in enumerate(molecules):
            n = num_atoms[i] # Total atoms including virtual
            if n > 1: # If real atoms exist
                # Calculate degree for real atoms (indices 1 to n-1)
                real_bonds = mol.bonds[1:n, 1:n]
                # Count bonds with valid orders (1-6)
                d_real = ( (real_bonds > 0) & (real_bonds <= mol.maximum_bond_order) ).sum(axis=1)
                # Combine with virtual atom degree (usually 0?) and pad
                d = np.concatenate(([0], d_real)) # Degree of virtual atom is 0
                p_d = np.pad(d, (0, max_num_atoms - n), mode='constant', constant_values=degree_padding_idx)
            elif n == 1: # Only virtual atom
                p_d = np.pad(np.array([0]), (0, max_num_atoms - 1), mode='constant', constant_values=degree_padding_idx)
            else: # Empty molecule (should not happen with init logic)
                p_d = np.full(max_num_atoms, fill_value=degree_padding_idx, dtype=int)
            batch_atoms_degree.append(p_d)
        batch_atoms_degree = np.stack(batch_atoms_degree)
        # --- End Atom Degrees ---

        # --- Batch Bonds ---
        bonds_list = []
        for i, mol in enumerate(molecules):
            n = num_atoms[i]
            if n > 0:
                # Pad bond matrix
                p_b = np.pad(mol.bonds, [(0, max_num_atoms - n), (0, max_num_atoms - n)], mode="constant", constant_values=bond_padding_idx)
                # Ensure diagonal is padded correctly (should already be 0, but belt-and-suspenders)
                # np.fill_diagonal(p_b, bond_padding_idx) # Is diagonal padding needed? Usually diagonal is ignored in GNNs. Let's keep it 0 or original.
                # Ensure diagonal remains 0 after padding if original was 0
                diag_indices = np.arange(max_num_atoms)
                p_b[diag_indices, diag_indices] = 0 # Explicitly set diagonal to 0 (no self-loops)
            else: # Empty molecule
                p_b = np.full((max_num_atoms, max_num_atoms), fill_value=bond_padding_idx, dtype=int)
                np.fill_diagonal(p_b, 0) # Ensure diagonal is 0
            bonds_list.append(p_b)
        batch_bonds = np.stack(bonds_list)
        # --- End Bonds ---

        # --- Batch Attention Mask ---
        # Mask where attention should NOT be paid (between padding atoms or padding<->real)
        additive_padding_masks = []
        for i, mol in enumerate(molecules):
            n = num_atoms[i] # Includes virtual
            # Mask is 0.0 for valid atom pairs, -inf for pairs involving padding
            m = np.full((max_num_atoms, max_num_atoms), fill_value=-np.inf, dtype=float)
            if n > 0:
                m[:n, :n] = 0.0 # Allow attention between non-padding atoms (including virtual?)
            # Usually diagonal is masked in attention, but let's keep it 0.0 for now as per original code.
            # If diagonal should be masked: np.fill_diagonal(m, -np.inf)
            additive_padding_masks.append(m)
        batch_additive_padding_attn_mask = np.stack(additive_padding_masks)
        # --- End Attention Mask ---

        # --- Construct Return Dictionary ---
        return_dict = dict(
            level_idx=torch.tensor(batch_level_idx, dtype=torch.long, device=device),
            picked_atom_mhe=torch.from_numpy(batch_picked_atom_mhe).long().to(device),
            num_atoms=torch.tensor(num_atoms, dtype=torch.long, device=device), # Includes virtual
            atoms=torch.from_numpy(batch_atoms).long().to(device),
            atoms_degree=torch.from_numpy(batch_atoms_degree).long().to(device),
            bonds=torch.from_numpy(batch_bonds).long().to(device),
            additive_padding_attn_mask=torch.from_numpy(batch_additive_padding_attn_mask).float().to(device),
        )

        # --- Optional Feasibility Masks ---
        if include_feasibility_masks:
            masks_l0, masks_l1, masks_l2 = [], [], []
            # Determine max action space sizes needed for padding across the batch
            max_actions_l0, max_actions_l1, max_actions_l2 = 0, 0, 7 # L2 is fixed size
            for mol in molecules:
                num_real = len(mol.atoms) - 1
                max_actions_l0 = max(max_actions_l0, 1 + num_real) # Terminate + num_real
                max_actions_l1 = max(max_actions_l1, mol.vocab_size + num_real + 1) # Add + Select + Remove

            # Pad individual masks
            for mol in molecules:
                num_real = len(mol.atoms) - 1
                # Level 0 Mask
                len_l0 = 1 + num_real
                mask_l0 = mol.current_action_mask if mol.current_action_level == 0 and mol.current_action_mask is not None else np.ones(len_l0, dtype=bool)
                # Ensure mask length matches expected length before padding
                if len(mask_l0) != len_l0: mask_l0 = np.ones(len_l0, dtype=bool)
                p_mask_l0 = np.pad(mask_l0, (0, max_actions_l0 - len_l0), mode='constant', constant_values=True)
                masks_l0.append(p_mask_l0)

                # Level 1 Mask
                len_l1 = mol.vocab_size + num_real + 1
                mask_l1 = mol.current_action_mask if mol.current_action_level == 1 and mol.current_action_mask is not None else np.ones(len_l1, dtype=bool)
                if len(mask_l1) != len_l1: mask_l1 = np.ones(len_l1, dtype=bool)
                p_mask_l1 = np.pad(mask_l1, (0, max_actions_l1 - len_l1), mode='constant', constant_values=True)
                masks_l1.append(p_mask_l1)

                # Level 2 Mask (Fixed size)
                len_l2 = 7
                mask_l2 = mol.current_action_mask if mol.current_action_level == 2 and mol.current_action_mask is not None else np.ones(len_l2, dtype=bool)
                if len(mask_l2) != len_l2: mask_l2 = np.ones(len_l2, dtype=bool) # Should not happen
                masks_l2.append(mask_l2) # No padding needed

            # Add masks to return dictionary
            return_dict["feasibility_mask_level_zero"] = torch.from_numpy(np.stack(masks_l0)).bool().to(device)
            return_dict["feasibility_mask_level_one"] = torch.from_numpy(np.stack(masks_l1)).bool().to(device)
            return_dict["feasibility_mask_level_two"] = torch.from_numpy(np.stack(masks_l2)).bool().to(device)
        # --- End Feasibility Masks ---

        return return_dict

    @staticmethod
    def batch_to_device(batch: dict, device: torch.device):
        """Moves all tensors in a batch dictionary to the specified device."""
        # No change needed
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    @staticmethod
    def get_single_atom_molecules(config: MoleculeConfig, repeat: int = 1) -> List['MoleculeDesign']:
        """Creates a list of MoleculeDesign instances, each with one allowed atom type."""
        # No change needed
        atoms = [i + 1 for i, name in enumerate(config.atom_vocabulary.keys()) if config.atom_vocabulary[name]["allowed"]]
        if not atoms: raise ValueError("No allowed atoms found in vocabulary config.")
        return MoleculeDesign.init_batch_from_instance_list(config, atoms * repeat)

    @staticmethod
    def from_smiles(config: MoleculeConfig, smiles: str, **kwargs) -> Tuple['MoleculeDesign', Dict[int, int]]:
        """Creates instance from SMILES. Raises Error on failure."""
        # Logic for RDKit preprocessing seems sound.
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: raise ValueError(f"Invalid SMILES input: {smiles}")
        try:
            # Optional: Remove Hs first? Depends on downstream use.
            # mol = Chem.RemoveHs(mol, sanitize=False)
            # Sanitize and Kekulize are important
            Chem.SanitizeMol(mol, catchErrors=True) # Catch errors during sanitization
            Chem.Kekulize(mol, clearAromaticFlags=True) # Essential for consistent bond orders
            # Canonical atom ordering for consistent internal representation
            canonical_order = rdmolfiles.CanonicalRankAtoms(mol)
            mol = rdmolops.RenumberAtoms(mol, canonical_order)
        except Exception as e:
             raise ValueError(f"Could not preprocess input SMILES {smiles}: {e}") from e

        # Call from_rdkit_mol to build the internal state
        try:
             return MoleculeDesign.from_rdkit_mol(config, mol, smiles=smiles)
        except Exception as e:
             # Catch errors during the conversion from RDKit to internal state
             raise RuntimeError(f"Error creating MoleculeDesign state from RDKit mol for {smiles}: {e}") from e

    @staticmethod
    def from_rdkit_mol(config: MoleculeConfig, rdkit_mol: Chem.Mol, smiles: Optional[str] = None) -> Tuple['MoleculeDesign', Dict[int, int]]:
        """Creates instance from RDKit Mol. Raises Error on failure."""
        # Mapping from RDKit bond types to internal bond orders
        BOND_TYPE_TO_RL_ORDER = {
            Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3,
            Chem.BondType.QUADRUPLE: 4, Chem.BondType.QUINTUPLE: 5, Chem.BondType.HEXTUPLE: 6,
            # Aromatic bonds should be Kekulized before reaching here
        }
        num_heavy_atoms = rdkit_mol.GetNumAtoms()

        # Find the first allowed atom type to use for initialization
        first_allowed_atom_idx = -1
        for i, name in enumerate(config.atom_vocabulary.keys()):
             if config.atom_vocabulary[name]["allowed"]:
                  first_allowed_atom_idx = i + 1
                  break
        if first_allowed_atom_idx == -1:
             raise ValueError("No allowed atom types found in config for initialization.")

        # Handle empty input molecule
        if num_heavy_atoms == 0:
            print(f"Warning: Input RDKit mol {smiles or ''} has 0 atoms. Creating empty design.")
            instance = MoleculeDesign(config, initial_atom=first_allowed_atom_idx)
            # Set state for empty molecule (only virtual atom)
            instance.atoms = np.array([0], dtype=np.uint8)
            instance.bonds = np.zeros((1, 1), dtype=np.uint8)
            instance.is_original_atom = np.array([False], dtype=bool)
            instance.update_action_mask() # Update mask for empty state
            return instance, {} # Return empty instance and empty map

        # Build reverse lookup for atom properties to vocab index
        try:
            reverse_atom_lookup = build_reverse_atom_lookup(config)
        except Exception as e:
            raise RuntimeError("Failed to build reverse atom lookup.") from e

        # Build internal atoms list and RDKit-to-internal index map
        internal_atoms_list = [0] # Start with virtual atom
        rdkit_to_internal_map = {} # Map: RDKit index -> Internal index (1-based)
        internal_idx_counter = 1
        for atom in rdkit_mol.GetAtoms():
            rdkit_idx = atom.GetIdx()
            atomic_num = atom.GetAtomicNum()
            charge = atom.GetFormalCharge()
            # RDKit chiral tags might need mapping if config uses different convention
            chiral_rdkit = atom.GetChiralTag()
            chiral_config = 0 # Default to unspecified
            if chiral_rdkit == Chem.ChiralType.CHI_TETRAHEDRAL_CW: chiral_config = 1
            elif chiral_rdkit == Chem.ChiralType.CHI_TETRAHEDRAL_CCW: chiral_config = 2

            # Find vocab index using lookup
            key = (atomic_num, charge, chiral_config)
            vocab_idx = reverse_atom_lookup.get(key)
            # Fallback to non-chiral version if specific chiral tag not found
            if vocab_idx is None and chiral_config != 0:
                vocab_idx = reverse_atom_lookup.get((atomic_num, charge, 0))
            # If still not found, the atom type is not supported
            if vocab_idx is None:
                raise ValueError(f"Atom type ({atomic_num}, charge={charge}, chiral={chiral_config}) in {smiles or ''} not found in vocabulary lookup.")

            internal_atoms_list.append(vocab_idx)
            rdkit_to_internal_map[rdkit_idx] = internal_idx_counter
            internal_idx_counter += 1

        # Build internal bonds matrix
        num_total_atoms = len(internal_atoms_list) # Includes virtual
        internal_bonds_matrix = np.zeros((num_total_atoms, num_total_atoms), dtype=np.uint8)
        for bond in rdkit_mol.GetBonds():
            idx1_rdkit, idx2_rdkit = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()

            # Convert RDKit bond type to internal order
            rl_order = BOND_TYPE_TO_RL_ORDER.get(bond_type)
            if rl_order is None:
                # If aromatic or other unsupported type reaches here, input wasn't properly Kekulized/prepared
                raise ValueError(f"Unsupported bond type {bond_type} found in {smiles or ''}. Ensure input is Kekulized.")

            # Get corresponding internal indices
            try:
                int_idx1 = rdkit_to_internal_map[idx1_rdkit]
                int_idx2 = rdkit_to_internal_map[idx2_rdkit]
            except KeyError:
                # Should not happen if map was built correctly
                raise RuntimeError(f"RDKit index map failed for bond ({idx1_rdkit}, {idx2_rdkit}).")

            # Set bond order in symmetric matrix
            internal_bonds_matrix[int_idx1, int_idx2] = internal_bonds_matrix[int_idx2, int_idx1] = rl_order

        # Add virtual bonds
        if num_total_atoms > 1:
            internal_bonds_matrix[0, 1:] = internal_bonds_matrix[1:, 0] = MoleculeDesign.virtual_bond_idx

        # Create and initialize the MoleculeDesign instance
        try:
            # Initialize with the first allowed atom (needed by __init__)
            instance = MoleculeDesign(config, initial_atom=first_allowed_atom_idx)
            # Overwrite state with data derived from RDKit molecule
            instance.atoms = np.array(internal_atoms_list, dtype=np.uint8)
            instance.bonds = internal_bonds_matrix
            # Mark all real atoms as original for Rule 1
            instance.is_original_atom = np.array([False] + [True] * num_heavy_atoms, dtype=bool)
            # Reset trajectory state
            instance.synthesis_done = False
            instance._cached_smiles = None # Clear cache
            instance._cached_rdkit_mol = None
            instance.objective = None
            instance.infeasibility_flag = False
            instance.current_action_level = 0 # Start at level 0
            instance.history = []
            # Reset action context
            instance.l0_selected_atom_idx = None
            instance.l1_action_type = None
            instance.l1_new_atom_type = None
            instance.l1_selected_existing_atom_idx = None
            # Reset Rule 2 context
            instance.last_bond_action_details = None
            # Calculate initial action mask
            instance.update_action_mask()
        except Exception as e:
            # Catch errors during instance creation or initial mask update
            raise RuntimeError(f"Error creating/setting state for MoleculeDesign instance from {smiles or ''}: {e}") from e

        # Return the created instance and the RDKit-to-internal index map
        return instance, rdkit_to_internal_map
