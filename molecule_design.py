import copy
import random
import numpy as np
import torch
from torch import nn
from rdkit import Chem, RDLogger

from config import MoleculeConfig
from core.abstracts import BaseTrajectory
from core.utils import softmax

from typing import Optional, List, Tuple

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

class MoleculeDesign(BaseTrajectory):
    """
    Environment for molecular design using a revised hierarchical action space (v2025-04-20 Refactor).

    State Management: Internal NumPy arrays (self.atoms, self.bonds) are the primary source of truth.
                      self.rdkit_mol is maintained for visualization, distance calculation, and final SMILES.
                      No syncing *from* RDKit back to internal arrays.

    Action Levels:
        - Level 0: Terminate (if connected) or Select Existing Atom.
        - Level 1: Add New Atom, Select Existing Atom for Bond, or Initiate Modify Atom.
        - Level 2 (Bond Path): Set Bond Order 1-6 (creates if 0) or Remove Bond.
        - Level 2 (Modify Path): Replace Atom Type or Remove Atom.

    Connectivity constraints are only applied to the Terminate action.
    Atom/Bond removals are allowed even if they cause disconnection.
    """
    maximum_bond_order = 6
    virtual_bond_idx = 7
    maximum_num_atoms_overall = 100 # Used for distance padding
    bond_types = { # RDKit bond types for adding bonds
        1: Chem.rdchem.BondType.SINGLE,
        2: Chem.rdchem.BondType.DOUBLE,
        3: Chem.rdchem.BondType.TRIPLE,
        4: Chem.rdchem.BondType.QUADRUPLE,
        5: Chem.rdchem.BondType.QUINTUPLE,
        6: Chem.rdchem.BondType.HEXTUPLE
    }
    # Action index for Level 2 (Modify Path) Remove Atom action
    REMOVE_ATOM_ACTION_L2_MODIFY = -1 # Placeholder, set in __init__

    def __init__(self, config: MoleculeConfig, initial_atom: int):
        self.config = config
        self.atom_vocabulary = self.config.atom_vocabulary
        self.vocabulary_atom_idcs = list(range(1, len(self.atom_vocabulary) + 1)) # [1, ..., V]
        self.vocabulary_atom_names = list(self.atom_vocabulary.keys())
        # Ensure valence list matches vocab indices + virtual atom
        self.vocabulary_valence = [-1] * (len(self.vocabulary_atom_names) + 1)
        for i, name in enumerate(self.vocabulary_atom_names):
             self.vocabulary_valence[i+1] = self.atom_vocabulary[name]["valence"]

        self.atom_feasibility_mask = [not self.atom_vocabulary[x]["allowed"] for x in self.vocabulary_atom_names]
        self.vocab_size = len(self.vocabulary_atom_idcs) # V

        # Set the actual index for the Remove Atom action in Level 2 Modify path
        self.REMOVE_ATOM_ACTION_L2_MODIFY = self.vocab_size # V

        self.upper_limit_atoms = self.config.max_num_atoms
        assert initial_atom in self.vocabulary_atom_idcs and not self.atom_feasibility_mask[initial_atom - 1], \
            f"Initial atom {initial_atom} must be in vocabulary {self.vocabulary_atom_idcs} and allowed in config."
        self.initial_atom = initial_atom

        # Internal State (Primary)
        self.atoms = np.array([0, initial_atom], dtype=np.uint8) # Includes virtual atom 0
        self.bonds = np.zeros((2, 2), dtype=np.uint8)
        self.bonds[0, 1] = self.bonds[1, 0] = self.virtual_bond_idx
        self.virtual_distance = self.maximum_num_atoms_overall + 1
        self.infinity_distance = self.maximum_num_atoms_overall + 2
        self.topological_distance_matrix = np.array([[0, self.virtual_distance], [self.virtual_distance, 0]], dtype=np.uint8)

        # RDKit Representation (Secondary, for utils)
        self.rdkit_mol = Chem.RWMol()
        self._add_atom_to_rdkit(initial_atom) # Add the first real atom

        # Trajectory State
        self.synthesis_done = False
        self.smiles_string: Optional[str] = None
        self.objective: Optional[float] = None
        self.sa_score: float = 0.
        self.infeasibility_flag: bool = False
        self.is_currently_connected: bool = True # Assume initial single atom is connected

        # Action Handling State
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

    def _add_atom_to_rdkit(self, atom_vocab_idx: int):
        """Adds an atom (specified by 1-based vocabulary index) to self.rdkit_mol."""
        if not (1 <= atom_vocab_idx <= self.vocab_size):
             raise ValueError(f"Invalid vocabulary index {atom_vocab_idx} for adding atom.")
        atom_config = self.atom_vocabulary[self.vocabulary_atom_names[atom_vocab_idx - 1]]
        a = Chem.Atom(atom_config["atomic_number"])
        if "formal_charge" in atom_config:
            a.SetFormalCharge(atom_config["formal_charge"])
        chiral_tag_config = atom_config.get("chiral_tag")
        if chiral_tag_config == 1:
            a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
        elif chiral_tag_config == 2:
            a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        else:
            a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        return self.rdkit_mol.AddAtom(a) # Return the RDKit index

    def _update_rdkit_bond(self, rdkit_idx1: int, rdkit_idx2: int, new_order: int):
        """Adds or modifies a bond in self.rdkit_mol. new_order=0 removes."""
        try:
             # Ensure indices are valid before proceeding
             num_atoms_rdkit = self.rdkit_mol.GetNumAtoms()
             if not (0 <= rdkit_idx1 < num_atoms_rdkit and 0 <= rdkit_idx2 < num_atoms_rdkit):
                  raise ValueError(f"Invalid RDKit indices ({rdkit_idx1}, {rdkit_idx2}) for bond update. Max index: {num_atoms_rdkit - 1}")

             existing_bond = self.rdkit_mol.GetBondBetweenAtoms(rdkit_idx1, rdkit_idx2)
             if existing_bond:
                 self.rdkit_mol.RemoveBond(rdkit_idx1, rdkit_idx2)

             if new_order > 0 and new_order <= self.maximum_bond_order:
                 rdkit_bond_type = self.bond_types.get(new_order)
                 if rdkit_bond_type:
                     self.rdkit_mol.AddBond(rdkit_idx1, rdkit_idx2, rdkit_bond_type)
                 else:
                     print(f"WARNING: Invalid bond order {new_order} requested for RDKit.")
                     self.infeasibility_flag = True
             elif new_order > self.maximum_bond_order:
                  print(f"WARNING: Bond order {new_order} exceeds maximum {self.maximum_bond_order}.")
                  self.infeasibility_flag = True

             # Attempt to sanitize to update RDKit's internal state (valence etc.)
             # Failure here indicates a chemically invalid state in RDKit
             Chem.SanitizeMol(self.rdkit_mol)

        except ValueError as ve: # Catch specific index errors
             print(f"ERROR: {ve}")
             self.infeasibility_flag = True
        except Exception as e:
             print(f"WARNING: Error updating/sanitizing RDKit bond {rdkit_idx1}-{rdkit_idx2} order {new_order}: {e}")
             self.infeasibility_flag = True # Mark as infeasible if RDKit update fails

    def _replace_rdkit_atom(self, rdkit_idx: int, new_atom_type_idx: int):
        """Replace atom type in RDKit molecule and attempt sanitization."""
        if rdkit_idx < 0 or rdkit_idx >= self.rdkit_mol.GetNumAtoms():
             print(f"ERROR: Invalid rdkit index {rdkit_idx} for replacement.")
             self.infeasibility_flag = True
             return
        try:
            atom = self.rdkit_mol.GetAtomWithIdx(rdkit_idx)
            atom_config = self.atom_vocabulary[self.vocabulary_atom_names[new_atom_type_idx - 1]]
            atom.SetAtomicNum(atom_config["atomic_number"])
            atom.SetFormalCharge(atom_config.get("formal_charge", 0))
            chiral_tag_config = atom_config.get("chiral_tag")
            if chiral_tag_config == 1: atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
            elif chiral_tag_config == 2: atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
            else: atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            # Update property cache for the atom and its neighbors before sanitizing
            atom.UpdatePropertyCache(strict=False)
            for neighbor in atom.GetNeighbors():
                 neighbor.UpdatePropertyCache(strict=False)
            # Attempt sanitization
            Chem.SanitizeMol(self.rdkit_mol)
        except Exception as e:
            print(f"WARNING: Sanitization failed after replacing RDKit atom {rdkit_idx} with type {new_atom_type_idx}: {e}")
            self.infeasibility_flag = True

    def _remove_atom_rdkit_and_remap(self, rdkit_idx_to_remove: int) -> dict:
        """
        Removes an atom from self.rdkit_mol and returns a mapping
        from old RDKit indices to new RDKit indices.
        Handles potential errors during removal.
        """
        old_to_new_map = {}
        try:
            num_atoms_before = self.rdkit_mol.GetNumAtoms()
            if rdkit_idx_to_remove < 0 or rdkit_idx_to_remove >= num_atoms_before:
                raise ValueError(f"Invalid RDKit index {rdkit_idx_to_remove} for removal. Num atoms: {num_atoms_before}")

            # Build the map *before* removal
            for i in range(num_atoms_before):
                if i < rdkit_idx_to_remove:
                    old_to_new_map[i] = i
                elif i > rdkit_idx_to_remove:
                    old_to_new_map[i] = i - 1
            # Atom being removed is not in the new map

            self.rdkit_mol.RemoveAtom(rdkit_idx_to_remove)
            # Attempt sanitization after removal only if atoms remain
            if self.rdkit_mol.GetNumAtoms() > 0:
                 Chem.SanitizeMol(self.rdkit_mol)

        except ValueError as ve: # Catch specific index errors
            print(f"ERROR: {ve}")
            self.infeasibility_flag = True
            return {}
        except Exception as e:
            print(f"ERROR during RDKit atom removal ({rdkit_idx_to_remove}) or subsequent sanitization: {e}")
            self.infeasibility_flag = True
            return {} # Return empty map on failure

        return old_to_new_map

    def _check_and_update_connectivity(self):
        """Checks connectivity of the current rdkit_mol and updates the flag."""
        # This relies on rdkit_mol being kept up-to-date
        if self.rdkit_mol is None or self.rdkit_mol.GetNumAtoms() <= 1:
            self.is_currently_connected = True # Single atom or empty is considered "connected" for termination logic
            return
        try:
            # Use RDKit's fragment counter
            frags = Chem.GetMolFrags(self.rdkit_mol, asMols=False, sanitizeFrags=False)
            self.is_currently_connected = (len(frags) == 1)
        except Exception as e:
             # If GetMolFrags fails, it likely means the RDKit state is bad
             print(f"WARNING: GetMolFrags failed during connectivity check: {e}. Assuming disconnected.")
             self.is_currently_connected = False
             self.infeasibility_flag = True # Mark infeasible if connectivity check fails

    def _get_current_valence_usage(self) -> np.array:
        """Calculates the sum of explicit bond orders for each real atom from self.bonds."""
        num_real_atoms = len(self.atoms) - 1
        if num_real_atoms <= 0:
            return np.array([], dtype=int)
        # Use internal bonds array, excluding virtual atom 0
        # Slice self.bonds[1:num_real_atoms+1, 1:num_real_atoms+1] gives the adjacency matrix
        current_explicit_usage = np.sum(self.bonds[1 : num_real_atoms + 1, 1 : num_real_atoms + 1], axis=1)
        return current_explicit_usage.astype(int)

    def _get_remaining_valence(self) -> np.array:
        """Calculates remaining valence for each real atom based on self.atoms and self.bonds."""
        num_real_atoms = len(self.atoms) - 1
        if num_real_atoms <= 0:
            return np.array([], dtype=int)

        current_usage = self._get_current_valence_usage()
        # Get total valence capacity from config based on atom types in self.atoms[1:]
        # Ensure vocabulary_valence has entries for all possible vocab indices
        total_valence = np.array([self.vocabulary_valence[atom_vocab_idx]
                                  for atom_vocab_idx in self.atoms[1:]], dtype=int)
        remaining = total_valence - current_usage
        remaining = np.maximum(0, remaining) # Ensure non-negative
        return remaining

    def update_action_mask(self):
        """Creates the action mask based on the internal state (self.atoms, self.bonds)."""
        if self.synthesis_done:
            self.current_action_mask = None
            return

        num_real_atoms = len(self.atoms) - 1  # N = number of real atoms
        remaining_valence = self._get_remaining_valence()  # 0-indexed array for real atoms [0..N-1]

        if self.current_action_level == 0:
            # Level 0: Terminate (0) / Select Existing Atom (1..N)
            action_space_size = 1 + num_real_atoms
            mask = np.zeros(action_space_size, dtype=bool)
            # Mask Terminate if not connected OR if only 1 real atom (cannot terminate single atom)
            if not self.is_currently_connected or num_real_atoms <= 0:
                mask[0] = True
            # Mask atom selection if no real atoms exist
            if num_real_atoms == 0:
                mask[1:] = True

            # --- REMOVED VALENCE CHECK FOR L0 SELECTION ---
            # # Mask selecting atoms that have no remaining valence (cannot initiate modify/bond from it)
            # if num_real_atoms > 0:
            #      no_valence_indices = np.where(remaining_valence <= 0)[0]
            #      if len(no_valence_indices) > 0:
            #           mask[1 + no_valence_indices] = True # +1 because action indices are 1-based internal indices
            # --- END REMOVAL ---

            self.current_action_mask = mask

        elif self.current_action_level == 1:
            # Level 1: Add New Atom (0..V-1) / Select Existing Atom for Bond (V..V+N-1) / Initiate Modify Atom (V+N)
            action_space_size = self.vocab_size + num_real_atoms + 1
            mask = np.ones(action_space_size, dtype=bool)  # Mask all initially

            anchor_atom_internal_idx = self.l0_selected_atom_idx
            # Basic check: anchor must be valid
            if anchor_atom_internal_idx is None or anchor_atom_internal_idx <= 0 or anchor_atom_internal_idx > num_real_atoms:
                print("ERROR: L1 Mask - Invalid anchor atom index.")
                self.current_action_mask = mask;
                return  # Leave all masked
            anchor_atom_0_idx = anchor_atom_internal_idx - 1

            # Check if anchor has valence. If not, only Modify path is possible.
            if remaining_valence[anchor_atom_0_idx] <= 0:
                mask[self.vocab_size + num_real_atoms] = False  # Unmask Initiate Modify Atom ONLY
                self.current_action_mask = mask;
                return

            # --- Anchor HAS valence ---

            # Unmask Add New Atom (0..V-1)
            # Condition: Anchor has valence (checked above) AND we haven't hit the atom limit
            if self.upper_limit_atoms is None or num_real_atoms < self.upper_limit_atoms:
                for i in range(self.vocab_size):
                    # Check if atom type is allowed and has valence capacity >= 1
                    if not self.atom_feasibility_mask[i] and self.vocabulary_valence[i + 1] >= 1:
                        mask[i] = False  # Unmask allowed new atoms

            # Unmask Select Existing Atom for Bond (V..V+N-1)
            for target_0_idx in range(num_real_atoms):
                target_internal_idx = target_0_idx + 1
                action_idx = self.vocab_size + target_0_idx

                # Cannot select the anchor atom itself
                if target_internal_idx == anchor_atom_internal_idx: continue

                # Condition: Target atom must have valence OR a bond must already exist (for removal/modification)
                bond_exists = self.bonds[anchor_atom_internal_idx, target_internal_idx] > 0
                target_has_valence = remaining_valence[target_0_idx] > 0

                # Allow selection if a bond exists (can modify/remove) OR if target has valence (can add/increase)
                # Anchor valence already confirmed > 0
                if bond_exists or target_has_valence:
                    mask[action_idx] = False

            # Unmask Initiate Modify Atom (V+N) - Always allowed if anchor was validly selected at L0
            mask[self.vocab_size + num_real_atoms] = False
            self.current_action_mask = mask

        elif self.current_action_level == 2:
            # L2 logic remains the same as the previous version...
            if self.is_modifying_atom:
                # Level 2 (Modify Path): Replace Type (0..V-1) / Remove Atom (V)
                action_space_size = self.vocab_size + 1
                mask = np.ones(action_space_size, dtype=bool)  # Mask all initially

                atom_internal_idx = self.atom_to_modify
                if atom_internal_idx is None or atom_internal_idx <= 0 or atom_internal_idx > num_real_atoms:
                    print("ERROR: L2 Modify Mask - Invalid atom_to_modify index.")
                    self.current_action_mask = mask;
                    return  # Leave all masked
                atom_0_idx = atom_internal_idx - 1
                current_atom_type_idx = self.atoms[atom_internal_idx]

                # Unmask Replace Atom Type (0..V-1)
                # Ensure current_usage calculation is safe
                current_usage = 0
                # Use _get_current_valence_usage which returns array for real atoms (0-indexed)
                valence_usage_array = self._get_current_valence_usage()
                if atom_0_idx < len(valence_usage_array):
                    current_usage = valence_usage_array[atom_0_idx]
                else:
                    print(
                        f"WARNING: L2 Modify Mask - Index {atom_0_idx} out of bounds for valence usage array (len {len(valence_usage_array)}).")
                    # Proceed, but replacement might fail later if usage is actually high

                for vocab_idx_0 in range(self.vocab_size):
                    new_atom_type = vocab_idx_0 + 1
                    action_idx = vocab_idx_0

                    if self.atom_feasibility_mask[vocab_idx_0]: continue  # Skip disallowed types
                    if new_atom_type == current_atom_type_idx: continue  # Skip replacing with same type
                    # Check if new type's valence capacity is sufficient for current bonds
                    if self.vocabulary_valence[new_atom_type] >= current_usage:
                        mask[action_idx] = False

                # Unmask Remove Atom (Action V) - Allow removing any real atom
                if num_real_atoms > 0:  # Cannot remove the last real atom (or virtual)
                    mask[self.REMOVE_ATOM_ACTION_L2_MODIFY] = False
                self.current_action_mask = mask

            else:
                # Level 2 (Bond Path): Set Bond 1-6 (0..5) / Remove Bond (6)
                action_space_size = 7
                mask = np.ones(action_space_size, dtype=bool)  # Mask all initially

                atom_A_internal_idx = self.l0_selected_atom_idx
                atom_B_internal_idx = -1
                if self.l1_new_atom_type is not None:
                    atom_B_internal_idx = len(self.atoms) - 1  # B is the newly added atom
                elif self.l1_selected_existing_atom_idx is not None:
                    atom_B_internal_idx = self.l1_selected_existing_atom_idx
                else:
                    print("ERROR: L2 Bond Mask - L1 context missing.")
                    self.current_action_mask = mask;
                    return  # Leave all masked

                # Validate indices
                if (atom_A_internal_idx is None or atom_A_internal_idx <= 0 or atom_A_internal_idx > num_real_atoms or
                        atom_B_internal_idx <= 0 or atom_B_internal_idx > num_real_atoms):
                    print(f"ERROR: L2 Bond Mask - Invalid indices A={atom_A_internal_idx}, B={atom_B_internal_idx}")
                    self.current_action_mask = mask;
                    return  # Leave all masked

                atom_A_0_idx = atom_A_internal_idx - 1
                atom_B_0_idx = atom_B_internal_idx - 1

                current_bond_order = self.bonds[atom_A_internal_idx, atom_B_internal_idx]
                valence_A_rem = remaining_valence[atom_A_0_idx]
                valence_B_rem = remaining_valence[atom_B_0_idx]

                # Calculate max possible increase based on remaining valence
                max_increase = min(valence_A_rem, valence_B_rem)
                effective_current_order = int(current_bond_order) if current_bond_order > 0 else 0
                max_allowed_final_order = min(effective_current_order + max_increase, self.maximum_bond_order)

                # Unmask Set Bond Order actions (0..5) -> order (1..6)
                for order in range(1, self.maximum_bond_order + 1):
                    action_idx = order - 1
                    if order <= max_allowed_final_order:
                        mask[action_idx] = False

                # Unmask Remove Bond action (6) if bond currently exists
                if current_bond_order > 0:
                    mask[6] = False
                self.current_action_mask = mask
        else:
            raise ValueError(f"Invalid current_action_level: {self.current_action_level}")


    def update_topological_distance_matrix(self):
        """Updates the distance matrix based on self.rdkit_mol."""
        num_total_atoms = len(self.atoms) # Includes virtual
        num_real_atoms = num_total_atoms - 1

        # Ensure internal matrix has the correct size *before* calculation
        if self.topological_distance_matrix.shape[0] != num_total_atoms:
             print(f"WARNING: Resizing distance matrix in update_topological_distance_matrix. Should happen in take_action. From {self.topological_distance_matrix.shape} to ({num_total_atoms},{num_total_atoms})")
             # Resize and fill with infinity
             old_matrix = self.topological_distance_matrix
             old_size = old_matrix.shape[0]
             self.topological_distance_matrix = np.full((num_total_atoms, num_total_atoms), self.infinity_distance, dtype=np.uint8)
             # Copy old data if shrinking or growing
             size_to_copy = min(old_size, num_total_atoms)
             if size_to_copy > 0:
                  self.topological_distance_matrix[:size_to_copy, :size_to_copy] = old_matrix[:size_to_copy, :size_to_copy]
             # Set distances involving new atoms if growing
             if num_total_atoms > old_size:
                  new_indices = range(old_size, num_total_atoms)
                  self.topological_distance_matrix[0, new_indices] = self.virtual_distance
                  self.topological_distance_matrix[new_indices, 0] = self.virtual_distance
             # Always reset diagonal
             np.fill_diagonal(self.topological_distance_matrix, 0)


        # --- RDKit Distance Calculation ---
        if self.rdkit_mol and num_real_atoms > 0:
            try:
                # Ensure RDKit mol has the expected number of atoms
                if self.rdkit_mol.GetNumAtoms() != num_real_atoms:
                     print(f"WARNING: RDKit atom count ({self.rdkit_mol.GetNumAtoms()}) mismatch with internal ({num_real_atoms}) before GetDistanceMatrix.")
                     # If counts mismatch, RDKit matrix won't fit internal slice. Mark infeasible.
                     self.infeasibility_flag = True
                     return # Stop update if counts mismatch

                rdkit_dist_matrix_float = Chem.GetDistanceMatrix(self.rdkit_mol)

                # Process float matrix: Replace inf/large values BEFORE casting
                processed_matrix = np.where(
                    (rdkit_dist_matrix_float > 0) & (rdkit_dist_matrix_float <= self.maximum_num_atoms_overall),
                    rdkit_dist_matrix_float,
                    self.infinity_distance
                )
                np.fill_diagonal(processed_matrix, 0)
                rdkit_dist_matrix_uint8 = processed_matrix.astype(np.uint8)

                # Assign to the internal state slice for real atoms
                self.topological_distance_matrix[1:, 1:] = rdkit_dist_matrix_uint8

            except Exception as e:
                print(f"WARNING: Failed to update RDKit distance matrix: {e}.")
                self.infeasibility_flag = True
        elif num_real_atoms == 0 and num_total_atoms == 1:
             # Ensure matrix is correct for only virtual atom
             if self.topological_distance_matrix.shape != (1,1):
                  self.topological_distance_matrix = np.zeros((1,1), dtype=np.uint8)

    def _adjust_indices_after_removal(self, removed_internal_idx: int):
        """Adjusts stored internal indices after an atom removal."""
        if self.l0_selected_atom_idx is not None and self.l0_selected_atom_idx > removed_internal_idx:
            self.l0_selected_atom_idx -= 1
        if self.l1_selected_existing_atom_idx is not None and self.l1_selected_existing_atom_idx > removed_internal_idx:
            self.l1_selected_existing_atom_idx -= 1
        # atom_to_modify is cleared when exiting L2 Modify, so no adjustment needed here normally.

    def take_action(self, action: int):
        """Execute a given action, updating internal state directly and RDKit representation."""
        if self.synthesis_done:
            raise RuntimeError("Cannot take action on a terminated design.")
        if self.current_action_mask is None or action >= len(self.current_action_mask) or self.current_action_mask[action]:
            print(f"ERROR DETAILS: Action={action}, Level={self.current_action_level}, Mask Length={len(self.current_action_mask) if self.current_action_mask is not None else 'None'}")
            if self.current_action_mask is not None and action < len(self.current_action_mask):
                 print(f"Mask value at action index: {self.current_action_mask[action]}")
            else:
                 print("Action index out of bounds for mask.")
            # Also print relevant state for debugging mask issues
            print(f"l0_selected_atom_idx: {self.l0_selected_atom_idx}")
            print(f"is_modifying_atom: {self.is_modifying_atom}")
            print(f"atom_to_modify: {self.atom_to_modify}")
            print(f"l1_new_atom_type: {self.l1_new_atom_type}")
            print(f"l1_selected_existing_atom_idx: {self.l1_selected_existing_atom_idx}")
            raise ValueError(f"Action {action} is masked or invalid for level {self.current_action_level}.")

        current_level = self.current_action_level
        next_level = 0 # Default next level is 0
        self.history.append(int(action))
        num_real_atoms_before = len(self.atoms) - 1

        # --- RDKit Index Map ---
        # Create a map from current internal index (1-based) to current RDKit index (0-based)
        # This is needed *before* potential atom removals change RDKit indices
        internal_to_rdkit_map = { i: i-1 for i in range(1, num_real_atoms_before + 1) }

        # --- Execute Action ---
        try:
            atom_removed = False # Flag if atom removal happens

            if current_level == 0:
                if action == 0: # TERMINATE
                    self.synthesis_done = True
                    self.finalize()
                    next_level = -1 # Special value indicating termination
                else: # SELECT_EXISTING_ATOM (action = 1-based internal index)
                    self.l0_selected_atom_idx = action
                    # Reset L1/L2 context
                    self.is_modifying_atom = False
                    self.atom_to_modify = None
                    self.l1_new_atom_type = None
                    self.l1_selected_existing_atom_idx = None
                    next_level = 1

            elif current_level == 1:
                modify_atom_action_idx = self.vocab_size + num_real_atoms_before

                if action < self.vocab_size: # INITIATE_ADD_NEW_ATOM (action = 0-based vocab index)
                    self.l1_new_atom_type = action + 1 # Store 1-based vocab index
                    # --- Internal state update ---
                    self.atoms = np.append(self.atoms, self.l1_new_atom_type)
                    # Pad bonds and distance matrices BEFORE RDKit update
                    new_size = len(self.atoms)
                    self.bonds = np.pad(self.bonds, [(0, 1), (0, 1)], mode='constant', constant_values=0)
                    self.topological_distance_matrix = np.pad(self.topological_distance_matrix, [(0, 1), (0, 1)], mode='constant', constant_values=self.infinity_distance)
                    # Update connections/distances for the new atom
                    new_atom_internal_idx = new_size - 1
                    self.bonds[0, new_atom_internal_idx] = self.bonds[new_atom_internal_idx, 0] = self.virtual_bond_idx
                    self.topological_distance_matrix[0, new_atom_internal_idx] = self.topological_distance_matrix[new_atom_internal_idx, 0] = self.virtual_distance
                    self.topological_distance_matrix[new_atom_internal_idx, new_atom_internal_idx] = 0
                    # --- RDKit update ---
                    self._add_atom_to_rdkit(self.l1_new_atom_type)
                    # --- State flags ---
                    self.is_modifying_atom = False
                    next_level = 2
                elif action < modify_atom_action_idx: # SELECT_EXISTING_ATOM_FOR_BOND (action = V + 0-based real atom index)
                    target_0_idx = action - self.vocab_size
                    self.l1_selected_existing_atom_idx = target_0_idx + 1 # Store 1-based internal index
                    self.is_modifying_atom = False
                    next_level = 2
                elif action == modify_atom_action_idx: # INITIATE_MODIFY_ATOM
                    self.atom_to_modify = self.l0_selected_atom_idx # Store 1-based internal index
                    self.is_modifying_atom = True
                    next_level = 2
                else:
                     raise ValueError(f"Invalid action {action} received for Level 1.")

            elif current_level == 2:
                if self.is_modifying_atom:
                    # --- Modify Path ---
                    atom_internal_idx_to_modify = self.atom_to_modify
                    if atom_internal_idx_to_modify is None: raise ValueError("atom_to_modify not set in L2 Modify")
                    rdkit_idx_to_modify = internal_to_rdkit_map.get(atom_internal_idx_to_modify)
                    if rdkit_idx_to_modify is None: raise ValueError(f"Cannot map internal modify index {atom_internal_idx_to_modify} to RDKit index")

                    if action < self.vocab_size: # REPLACE_ATOM_TYPE (action = 0-based vocab index)
                        new_atom_type_vocab_idx = action + 1 # 1-based vocab index
                        # Internal state update
                        self.atoms[atom_internal_idx_to_modify] = new_atom_type_vocab_idx
                        # RDKit update
                        self._replace_rdkit_atom(rdkit_idx_to_modify, new_atom_type_vocab_idx)

                    elif action == self.REMOVE_ATOM_ACTION_L2_MODIFY: # REMOVE_ATOM
                        # --- RDKit Update First (to get mapping) ---
                        old_rdkit_to_new_rdkit_map = self._remove_atom_rdkit_and_remap(rdkit_idx_to_modify)
                        if self.infeasibility_flag: # Check if RDKit removal failed
                             raise ValueError(f"RDKit atom removal or sanitization failed for index {rdkit_idx_to_modify}")

                        # --- Internal State Update ---
                        # 1. Remove from atoms array
                        self.atoms = np.delete(self.atoms, atom_internal_idx_to_modify)
                        # 2. Remove from bonds matrix
                        self.bonds = np.delete(np.delete(self.bonds, atom_internal_idx_to_modify, axis=0), atom_internal_idx_to_modify, axis=1)
                        # 3. Remove from distance matrix
                        self.topological_distance_matrix = np.delete(np.delete(self.topological_distance_matrix, atom_internal_idx_to_modify, axis=0), atom_internal_idx_to_modify, axis=1)
                        # 4. Adjust stored indices
                        self._adjust_indices_after_removal(atom_internal_idx_to_modify)
                        atom_removed = True # Set flag

                    else:
                         raise ValueError(f"Invalid action {action} for Level 2 Modify Path.")

                    # Clear modify context
                    self.is_modifying_atom = False
                    self.atom_to_modify = None
                    next_level = 0 # Return to L0

                else:
                    # --- Bond Path ---
                    atom_A_internal_idx = self.l0_selected_atom_idx
                    atom_B_internal_idx = -1
                    if self.l1_new_atom_type is not None:
                        atom_B_internal_idx = len(self.atoms) - 1 # B is the last atom added
                    elif self.l1_selected_existing_atom_idx is not None:
                        atom_B_internal_idx = self.l1_selected_existing_atom_idx
                    else:
                         raise ValueError("L2 Bond path state inconsistent.")

                    # Get RDKit indices using the map created *before* this action
                    rdkit_idx_A = internal_to_rdkit_map.get(atom_A_internal_idx)
                    rdkit_idx_B = internal_to_rdkit_map.get(atom_B_internal_idx)
                    if rdkit_idx_A is None or rdkit_idx_B is None:
                         raise ValueError(f"Cannot map internal bond indices ({atom_A_internal_idx},{atom_B_internal_idx}) to RDKit indices")

                    if action <= 5: # SET_BOND_ORDER (action = 0..5 -> order 1..6)
                        new_bond_order = action + 1
                        # Internal state update
                        self.bonds[atom_A_internal_idx, atom_B_internal_idx] = new_bond_order
                        self.bonds[atom_B_internal_idx, atom_A_internal_idx] = new_bond_order
                        # RDKit update
                        self._update_rdkit_bond(rdkit_idx_A, rdkit_idx_B, new_bond_order)

                    elif action == 6: # REMOVE_BOND
                        # Internal state update
                        self.bonds[atom_A_internal_idx, atom_B_internal_idx] = 0
                        self.bonds[atom_B_internal_idx, atom_A_internal_idx] = 0
                        # RDKit update
                        self._update_rdkit_bond(rdkit_idx_A, rdkit_idx_B, 0)
                    else:
                         raise ValueError(f"Invalid action {action} for Level 2 Bond Path.")

                    # Clear L1 context
                    self.l1_new_atom_type = None
                    self.l1_selected_existing_atom_idx = None
                    next_level = 0 # Return to L0

            # --- Post Action Updates (if not terminated) ---
            if next_level != -1:
                 # Update connectivity and distances using the potentially modified RDKit mol
                 self._check_and_update_connectivity()
                 self.update_topological_distance_matrix() # Recalculate based on RDKit graph

                 # Transition level and update mask
                 self.current_action_level = next_level
                 self.update_action_mask()
            else:
                 # Termination happened
                 self.current_action_mask = None

        except Exception as e:
             # Catch any unexpected error during action execution
             print(f"FATAL ERROR during take_action(action={action}, level={current_level}): {e}")
             import traceback
             traceback.print_exc() # Print stack trace
             self.infeasibility_flag = True
             self.synthesis_done = True # Mark as done to prevent further actions
             self.current_action_mask = None
             # Do not proceed to mask update if a fatal error occurred

    def finalize(self, assert_feasible: bool = False):
        """Finalize molecule design, generate SMILES if valid using self.rdkit_mol."""
        # Ensure connectivity is checked based on final state
        self._check_and_update_connectivity()

        if assert_feasible:
             try: self.assert_feasible()
             except AssertionError as e:
                 print(f"Feasibility assertion failed during finalize: {e}")
                 self.infeasibility_flag = True

        # Check connectivity required for valid SMILES (unless empty)
        if self.rdkit_mol.GetNumAtoms() > 0 and not self.is_currently_connected:
             print("WARNING: Final molecule is disconnected. SMILES may represent fragments.")
             # Optionally set infeasibility_flag here if disconnected molecules are invalid
             # self.infeasibility_flag = True

        if not self.infeasibility_flag:
             try:
                 # Final sanitization attempt on the RDKit object
                 Chem.SanitizeMol(self.rdkit_mol)
                 self.smiles_string = Chem.MolToSmiles(self.rdkit_mol)
             except Exception as e:
                 print(f"Final sanitization/SMILES generation failed: {e}")
                 self.infeasibility_flag = True
                 self.smiles_string = None
        else:
             self.smiles_string = None # Ensure SMILES is None if infeasible

    def assert_feasible(self):
        """Check internal state consistency (NumPy arrays)."""
        assert self.atoms[0] == 0, "First atom should be virtual (0)"
        num_atoms = len(self.atoms)
        num_real_atoms = num_atoms - 1

        if num_real_atoms > 0:
             # Check atom indices are valid vocab indices
             valid_indices = all(1 <= idx <= self.vocab_size for idx in self.atoms[1:])
             assert valid_indices, f"Invalid atom vocab index found in self.atoms: {self.atoms}"
             # Check only allowed atoms are present
             allowed_check = all(not self.atom_feasibility_mask[idx - 1] for idx in self.atoms[1:])
             assert allowed_check, f"Disallowed atom type found in self.atoms: {self.atoms}"

        assert self.upper_limit_atoms is None or num_real_atoms <= self.upper_limit_atoms, f"Max atoms exceeded ({num_real_atoms} > {self.upper_limit_atoms})"

        # Check matrix dimensions
        assert self.bonds.shape == (num_atoms, num_atoms), f"Bonds shape mismatch: {self.bonds.shape} vs expected ({num_atoms},{num_atoms})"
        assert self.topological_distance_matrix.shape == (num_atoms, num_atoms), f"Distance matrix shape mismatch: {self.topological_distance_matrix.shape} vs expected ({num_atoms},{num_atoms})"

        if num_real_atoms > 0:
             assert np.all(self.bonds[0, 1:] == self.virtual_bond_idx), "Virtual atom connection missing in bonds"
             assert np.all(self.bonds[1:, 0] == self.virtual_bond_idx), "Virtual atom connection missing in bonds"
             # Distance check might be less strict if disconnected graphs are allowed temporarily
             # assert np.all(self.topological_distance_matrix[0, 1:] == self.virtual_distance), "Virtual atom distance missing"
             # assert np.all(self.topological_distance_matrix[1:, 0] == self.virtual_distance), "Virtual atom distance missing"

        assert not np.any(self.bonds.diagonal()), "Self-loops detected in bonds"
        assert np.all(self.topological_distance_matrix.diagonal() == 0), "Diagonal in distance matrix is not 0"
        assert np.all(self.bonds == self.bonds.T), "Bond matrix not symmetric" # Use np.all for element-wise comparison
        # Distance matrix might not be symmetric if using directed calculation, but GetDistanceMatrix is symmetric
        assert np.all(self.topological_distance_matrix == self.topological_distance_matrix.T), "Distance matrix not symmetric"

        if num_real_atoms > 0:
             remaining_valence = self._get_remaining_valence()
             assert np.all(remaining_valence >= 0), f"Valence constraints violated. Remaining: {remaining_valence}"

    def to_rdkit_mol(self, sanitize=True) -> Chem.RWMol:
        """Creates a *new* RDKit molecule from the internal state (atoms, bonds). Useful for debugging."""
        mol = Chem.RWMol()
        if len(self.atoms) <= 1: return mol # Return empty mol if only virtual atom

        # Add atoms (from internal self.atoms)
        for atom_vocab_idx in self.atoms[1:]:
            if not (1 <= atom_vocab_idx <= self.vocab_size):
                 print(f"WARNING: Invalid vocab index {atom_vocab_idx} in self.atoms during to_rdkit_mol.")
                 continue # Skip invalid atoms
            atom_config = self.atom_vocabulary[self.vocabulary_atom_names[atom_vocab_idx - 1]]
            a = Chem.Atom(atom_config["atomic_number"])
            if "formal_charge" in atom_config: a.SetFormalCharge(atom_config["formal_charge"])
            ct = atom_config.get("chiral_tag")
            if ct == 1: a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
            elif ct == 2: a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
            else: a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            mol.AddAtom(a)

        # Add bonds (from internal self.bonds)
        real_bonds = self.bonds[1:, 1:]
        num_real_atoms = len(self.atoms) - 1
        for i in range(num_real_atoms):
            for j in range(i + 1, num_real_atoms): # Avoid double counting and self-loops
                bond_order = real_bonds[i, j]
                if bond_order > 0 and bond_order <= self.maximum_bond_order:
                    rdkit_bond_type = self.bond_types.get(int(bond_order)) # Ensure int
                    if rdkit_bond_type:
                        mol.AddBond(i, j, rdkit_bond_type)
                elif bond_order > self.maximum_bond_order and bond_order != self.virtual_bond_idx:
                     print(f"WARNING: Invalid bond order {bond_order} in self.bonds during to_rdkit_mol.")

        if sanitize:
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                print(f"Sanitization failed in to_rdkit_mol generated from internal state: {e}")
                # Optionally return None or unsanitized mol depending on desired behavior
        return mol

    def is_terminable(self):
        """Checks if the current state allows termination."""
        # Terminable if at Level 0, not already done, and connected (or empty/single atom)
        can_terminate = self.current_action_level == 0 and not self.synthesis_done
        # Allow termination if 0 or 1 real atoms, or if multiple atoms and connected
        connectivity_ok = (len(self.atoms) <= 2) or self.is_currently_connected
        return can_terminate and connectivity_ok

    def to_smiles(self) -> Optional[str]:
        """Returns the SMILES string if finalized and valid."""
        return self.smiles_string

    # --- Static and Abstract Methods (Largely unchanged, review batching if needed) ---

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
                if mask is None: # Already terminated
                     log_probs_to_return.append(np.array([]))
                     continue

                if mol.current_action_level == 0:
                    logits = batch_logits_l0[i]
                elif mol.current_action_level == 1:
                    logits = batch_logits_l1[i]
                elif mol.current_action_level == 2:
                    logits = batch_logits_l2[i]
                else: # Should not happen
                    print(f"WARNING: Invalid action level {mol.current_action_level} in log_probability_fn.")
                    log_probs_to_return.append(np.array([]))
                    continue

                # Ensure logits match mask length before applying mask
                mask_len = len(mask)
                if len(logits) > mask_len:
                    logits = logits[:mask_len]
                elif len(logits) < mask_len:
                     raise ValueError(f"Logits length ({len(logits)}) < Mask length ({mask_len}) for level {mol.current_action_level}.")

                logits[mask] = -np.inf # Apply mask
                # Manual log_softmax: handle potential all -inf case
                max_logit = np.max(logits)
                if np.isneginf(max_logit): # All actions were masked
                     log_probs = logits # Keep all as -inf
                else:
                     exp_logits = np.exp(logits - max_logit) # Subtract max for numerical stability
                     log_sum_exp = np.log(np.sum(exp_logits))
                     log_probs = logits - (max_logit + log_sum_exp)

                log_probs[mask] = -np.inf # Re-apply mask for safety
                log_probs_to_return.append(log_probs)
        return log_probs_to_return

    def transition_fn(self, action: int) -> Tuple['BaseTrajectory', bool]:
        """Creates a copy, takes the action, and returns the new state and termination status."""
        copied_molecule = copy.deepcopy(self)
        copied_molecule.take_action(action)
        return copied_molecule, copied_molecule.synthesis_done

    def to_max_evaluation_fn(self) -> float:
        """Returns the objective value."""
        if self.objective is None:
             # Optionally evaluate here if objective is None, or raise error
             # For now, assume objective should be set externally
             print("WARNING: Objective is None during to_max_evaluation_fn call.")
             return float("-inf") # Or raise error
        # Return negative infinity if infeasible
        return float("-inf") if self.infeasibility_flag else self.objective

    def num_actions(self) -> int:
        """Returns the number of valid (unmasked) actions at the current level."""
        if self.current_action_mask is None: return 0
        return int(np.sum(~self.current_action_mask))

    @staticmethod
    def list_to_batch(molecules: List['MoleculeDesign'], device: torch.device = None,
                      include_feasibility_masks: bool = False) -> dict:
        """Converts a list of MoleculeDesign instances to a batch dictionary (Revised for new state)."""
        # Padding indices remain the same
        atoms_padding_idx = molecules[0].vocab_size + 1
        # Ensure degree padding index accounts for max possible valence
        max_valence = max([-1] + [v for v in molecules[0].vocabulary_valence if v is not None and v >= 0])
        degree_padding_idx = max_valence + 2
        bond_padding_idx = MoleculeDesign.virtual_bond_idx + 1
        distance_padding_idx = MoleculeDesign.maximum_num_atoms_overall + 3

        device = torch.device("cpu") if device is None else device
        num_atoms = [len(mol.atoms) for mol in molecules] # Includes virtual atom
        max_num_atoms = max(num_atoms) if num_atoms else 0

        batch_level_idx = [mol.current_action_level for mol in molecules]

        # picked_atom_mhe: 1 for L0 anchor
        batch_picked_atom_mhe = np.zeros((len(molecules), max_num_atoms), dtype=int)
        for i, mol in enumerate(molecules):
            if mol.current_action_level >= 1 and mol.l0_selected_atom_idx is not None:
                 if 0 <= mol.l0_selected_atom_idx < max_num_atoms: # Check bounds (l0 is 1-based index)
                      batch_picked_atom_mhe[i, mol.l0_selected_atom_idx] = 1

        batch_atoms = np.stack([
            np.concatenate((mol.atoms, np.full(max_num_atoms - num_atoms[i], fill_value=atoms_padding_idx, dtype=np.uint8))) if num_atoms[i] > 0 else np.full(max_num_atoms, fill_value=atoms_padding_idx, dtype=np.uint8)
            for i, mol in enumerate(molecules)
        ])

        # Degree calculation based on internal bonds array
        batch_atoms_degree = []
        for i, mol in enumerate(molecules):
             current_num_atoms = num_atoms[i]
             if current_num_atoms > 1: # Need at least one real atom
                  real_bonds = mol.bonds[1:current_num_atoms, 1:current_num_atoms]
                  degree_real = (real_bonds > 0).sum(axis=1)
                  degree = np.concatenate(([0], degree_real)) # Add 0 for virtual atom
                  padded_degree = np.concatenate((degree, np.full(max_num_atoms - current_num_atoms, fill_value=degree_padding_idx, dtype=int)))
             elif current_num_atoms == 1: # Only virtual atom
                  padded_degree = np.concatenate(([0], np.full(max_num_atoms - 1, fill_value=degree_padding_idx, dtype=int)))
             else: # Empty case
                  padded_degree = np.full(max_num_atoms, fill_value=degree_padding_idx, dtype=int)
             batch_atoms_degree.append(padded_degree)
        batch_atoms_degree = np.stack(batch_atoms_degree)

        # Bonds and Distances padding
        bonds_list = []
        for i, mol in enumerate(molecules):
            current_num_atoms = num_atoms[i]
            if current_num_atoms > 0:
                 padded_bonds = np.pad(mol.bonds, [(0, max_num_atoms - current_num_atoms), (0, max_num_atoms - current_num_atoms)], mode="constant", constant_values=bond_padding_idx)
                 np.fill_diagonal(padded_bonds, bond_padding_idx) # Keep diagonal padding consistent
            else:
                 padded_bonds = np.full((max_num_atoms, max_num_atoms), fill_value=bond_padding_idx, dtype=int)
            bonds_list.append(padded_bonds)
        batch_bonds = np.stack(bonds_list)

        distance_matrices_list = []
        for i, mol in enumerate(molecules):
             current_num_atoms = num_atoms[i]
             if current_num_atoms > 0:
                  # Ensure distance matrix has correct shape before padding
                  if mol.topological_distance_matrix.shape != (current_num_atoms, current_num_atoms):
                       print(f"WARNING: Correcting distance matrix shape in list_to_batch for mol {i}. Was {mol.topological_distance_matrix.shape}, expected ({current_num_atoms},{current_num_atoms})")
                       # Fallback: create a default padded matrix
                       padded_dist = np.full((max_num_atoms, max_num_atoms), fill_value=distance_padding_idx, dtype=int)
                       np.fill_diagonal(padded_dist, 0)
                  else:
                       padded_dist = np.pad(mol.topological_distance_matrix, [(0, max_num_atoms - current_num_atoms), (0, max_num_atoms - current_num_atoms)], mode="constant", constant_values=distance_padding_idx)
             else:
                  padded_dist = np.full((max_num_atoms, max_num_atoms), fill_value=distance_padding_idx, dtype=int)
             distance_matrices_list.append(padded_dist)
        batch_topological_distance = np.stack(distance_matrices_list)

        # Additive padding mask
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
            topological_distance=torch.from_numpy(batch_topological_distance).long().to(device),
            additive_padding_attn_mask=torch.from_numpy(batch_additive_padding_attn_mask).float().to(device),
        )

        # Feasibility mask padding needs adjustment based on action space sizes
        if include_feasibility_masks:
            masks_l0, masks_l1, masks_l2 = [], [], []
            max_actions_l0, max_actions_l1, max_actions_l2 = 0, 0, 0

            for mol in molecules:
                 num_real = len(mol.atoms) - 1
                 max_actions_l0 = max(max_actions_l0, 1 + num_real)
                 max_actions_l1 = max(max_actions_l1, mol.vocab_size + num_real + 1)
                 # L2 max size is fixed across bond/modify paths in this version
                 max_actions_l2 = max(max_actions_l2, mol.vocab_size + 1, 7)

            for mol in molecules:
                 num_real = len(mol.atoms) - 1
                 # L0 Mask
                 mask_l0 = mol.current_action_mask if mol.current_action_level == 0 and mol.current_action_mask is not None else np.ones(1 + num_real, dtype=bool)
                 padded_mask_l0 = np.pad(mask_l0, (0, max_actions_l0 - len(mask_l0)), mode='constant', constant_values=True) # Pad with True (masked)
                 masks_l0.append(padded_mask_l0)

                 # L1 Mask
                 mask_l1 = mol.current_action_mask if mol.current_action_level == 1 and mol.current_action_mask is not None else np.ones(mol.vocab_size + num_real + 1, dtype=bool)
                 padded_mask_l1 = np.pad(mask_l1, (0, max_actions_l1 - len(mask_l1)), mode='constant', constant_values=True)
                 masks_l1.append(padded_mask_l1)

                 # L2 Mask - Pad to the max of bond/modify paths
                 if mol.current_action_level == 2 and mol.current_action_mask is not None:
                      mask_l2 = mol.current_action_mask
                      padded_mask_l2 = np.pad(mask_l2, (0, max_actions_l2 - len(mask_l2)), mode='constant', constant_values=True)
                 else:
                      padded_mask_l2 = np.ones(max_actions_l2, dtype=bool) # Default to all masked if not L2
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
                atoms.append(i + 1) # 1-based vocab index
        return MoleculeDesign.init_batch_from_instance_list(config, atoms * repeat)

    # --- Construction from SMILES/RDKit Mol ---

    @staticmethod
    def from_smiles(config: MoleculeConfig, smiles: str, do_finish=True, compare_smiles=False, max_steps=500) -> 'MoleculeDesign':
        """Creates a MoleculeDesign instance by simulating actions from a SMILES string."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: raise ValueError(f"Invalid SMILES input: {smiles}")
        try:
             # Basic sanitization, but allow failures as construction might handle intermediates
             Chem.SanitizeMol(mol, catchErrors=True)
        except Exception as e:
             print(f"Warning: Input SMILES {smiles} failed initial sanitization: {e}")
        return MoleculeDesign.from_rdkit_mol(config, mol, smiles, do_finish, compare_smiles, max_steps)

    @staticmethod
    def from_rdkit_mol(config: MoleculeConfig, rdkit_mol: Chem.Mol, smiles: Optional[str] = None,
                       do_finish: bool = True, compare_smiles: bool = False,
                       max_steps: int = 500) -> 'MoleculeDesign':
        """
        Creates a MoleculeDesign instance by simulating actions from an RDKit molecule (v2025-04-20 Refactor).
        Relies on internal state updates within take_action.
        """
        if not isinstance(rdkit_mol, Chem.Mol):
             raise TypeError("Input rdkit_mol must be an RDKit Mol object.")

        # --- Preprocessing ---
        # Work on a copy, remove Hs (important for matching internal state which is heavy-atom only)
        try:
            mol_copy = Chem.RemoveHs(rdkit_mol, sanitize=False) # Don't sanitize yet
            if mol_copy.GetNumAtoms() == 0:
                 raise ValueError("Input molecule has no heavy atoms.")
        except Exception as e:
             raise ValueError(f"Failed to remove hydrogens from input molecule: {e}")

        # Attempt Kekulization (best effort)
        try:
            Chem.Kekulize(mol_copy, clearAromaticFlags=True)
        except Exception as e:
            print(f"Warning: Kekulization failed for input mol copy: {e}")

        target_atoms = mol_copy.GetAtoms()
        num_target_atoms = len(target_atoms)
        if num_target_atoms == 0: # Double check after potential H removal failure
             raise ValueError("Input molecule has no heavy atoms.")

        # --- Map Target Atoms to Vocabulary ---
        prop_to_vocab_idx = {}
        vocab_names = list(config.atom_vocabulary.keys())
        for i, name in enumerate(vocab_names):
            cfg = config.atom_vocabulary[name]
            key = f"{cfg['atomic_number']}_{cfg.get('formal_charge', 0)}_{cfg.get('chiral_tag', 0)}"
            prop_to_vocab_idx[key] = i + 1

        target_atom_vocab_indices = [] # 1-based vocab indices for each target atom
        target_rdkit_indices = [atom.GetIdx() for atom in target_atoms] # Original RDKit indices in mol_copy

        for atom in target_atoms:
            key_parts = [str(atom.GetAtomicNum()), str(atom.GetFormalCharge())]
            ct = atom.GetChiralTag()
            chiral_tag_int = 0
            if ct == Chem.ChiralType.CHI_TETRAHEDRAL_CW: chiral_tag_int = 1
            elif ct == Chem.ChiralType.CHI_TETRAHEDRAL_CCW: chiral_tag_int = 2
            key_parts.append(str(chiral_tag_int))
            key = "_".join(key_parts)
            vocab_idx = prop_to_vocab_idx.get(key)
            # Fallback without chiral
            if vocab_idx is None and chiral_tag_int != 0:
                key_no_chiral = f"{atom.GetAtomicNum()}_{atom.GetFormalCharge()}_0"
                vocab_idx = prop_to_vocab_idx.get(key_no_chiral)

            if vocab_idx is None or not config.atom_vocabulary[vocab_names[vocab_idx-1]]["allowed"]:
                 raise ValueError(f"Target atom {atom.GetIdx()} (Props: {key}) cannot be mapped to an allowed vocabulary entry.")
            target_atom_vocab_indices.append(vocab_idx)

        # --- Build Target Adjacency (Bond Orders) ---
        bond_type_to_order = { Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3,
                               Chem.BondType.QUADRUPLE: 4, Chem.BondType.QUINTUPLE: 5, Chem.BondType.HEXTUPLE: 6,
                               Chem.BondType.AROMATIC: 1 } # Treat aromatic as 1 after Kekulize attempt
        target_adjacency_orders = {} # Map (min_rdkit_idx, max_rdkit_idx) -> order
        for bond in mol_copy.GetBonds():
            i_rdkit, j_rdkit = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            order = bond_type_to_order.get(bond.GetBondType(), 0)
            if order > 0:
                target_adjacency_orders[tuple(sorted((i_rdkit, j_rdkit)))] = order

        # --- Initialize Simulation ---
        initial_atom_vocab_idx = target_atom_vocab_indices[0]
        design = MoleculeDesign(config, initial_atom_vocab_idx)
        # Map: RDKit index (in mol_copy) -> internal MoleculeDesign index (1-based)
        rdkit_idx_to_internal_idx = {target_rdkit_indices[0]: 1}
        steps_taken = 0

        # --- Simulate Building Process ---
        for i in range(1, num_target_atoms): # Iterate through remaining target atoms
            if steps_taken > max_steps:
                print(f"Warning: Exceeded max_steps ({max_steps}) during construction from RDKit Mol.")
                design.infeasibility_flag = True; break
            if design.infeasibility_flag: break # Stop if an action failed

            current_target_rdkit_idx = target_rdkit_indices[i]
            atom_to_add_vocab_idx = target_atom_vocab_indices[i]
            atom_to_add_l1_action = atom_to_add_vocab_idx - 1 # 0-based for L1 action

            connection_found = False
            # Find first connection to an *already placed* atom j
            for j in range(i):
                anchor_target_rdkit_idx = target_rdkit_indices[j]
                bond_key = tuple(sorted((current_target_rdkit_idx, anchor_target_rdkit_idx)))
                bond_order = target_adjacency_orders.get(bond_key, 0)

                if bond_order > 0 and anchor_target_rdkit_idx in rdkit_idx_to_internal_idx:
                    anchor_internal_idx = rdkit_idx_to_internal_idx[anchor_target_rdkit_idx]
                    connection_found = True

                    # --- Action Sequence: Add Atom + First Bond ---
                    try:
                        # L0: Select anchor
                        design.take_action(anchor_internal_idx); steps_taken += 1
                        # L1: Add New Atom type
                        design.take_action(atom_to_add_l1_action); steps_taken += 1
                        # L2: Set Bond order
                        design.take_action(bond_order - 1); steps_taken += 1
                    except Exception as e:
                         raise ValueError(f"Error during initial atom add/bond for target RDKit {current_target_rdkit_idx}: {e}")

                    # Record mapping for the newly added atom
                    new_atom_internal_idx = len(design.atoms) - 1 # It's always the last one added
                    rdkit_idx_to_internal_idx[current_target_rdkit_idx] = new_atom_internal_idx

                    # --- Add Bonds to Other Previously Placed Atoms (k < i, k != j) ---
                    for k in range(i):
                        if k == j: continue # Skip the anchor used for initial placement
                        other_target_rdkit_idx = target_rdkit_indices[k]
                        extra_bond_key = tuple(sorted((current_target_rdkit_idx, other_target_rdkit_idx)))
                        extra_bond_order = target_adjacency_orders.get(extra_bond_key, 0)

                        if extra_bond_order > 0 and other_target_rdkit_idx in rdkit_idx_to_internal_idx:
                            other_internal_idx = rdkit_idx_to_internal_idx[other_target_rdkit_idx]
                            # --- Action Sequence: Add Bond Between Existing ---
                            try:
                                # L0: Select the *other* placed atom (k)
                                design.take_action(other_internal_idx); steps_taken += 1
                                # L1: Select the *newly added* atom (i) using its internal index
                                # L1 action index = vocab_size + (internal_idx - 1)
                                l1_select_action = design.vocab_size + (new_atom_internal_idx - 1)
                                design.take_action(l1_select_action); steps_taken += 1
                                # L2: Set Bond order
                                design.take_action(extra_bond_order - 1); steps_taken += 1
                            except Exception as e:
                                 raise ValueError(f"Error adding extra bond {other_target_rdkit_idx}-{current_target_rdkit_idx}: {e}")

                    break # Move to the next target atom i after handling atom j and all k

            if not connection_found and num_target_atoms > 1:
                 # This indicates disconnected components in the input heavy-atom graph
                 raise ValueError(f"Target molecule appears disconnected. Cannot connect atom RDKitIdx={current_target_rdkit_idx}.")

        # --- Finalization ---
        if not design.infeasibility_flag and do_finish:
            try:
                if design.is_terminable():
                     design.take_action(0); steps_taken += 1
                else:
                     # It might not be terminable if the construction process failed validation
                     print("WARNING: Molecule not terminable after construction.")
                     design.finalize(assert_feasible=False) # Finalize anyway to get SMILES if possible
            except ValueError as e:
                 print(f"WARNING: Error during final terminate action: {e}")
                 design.finalize(assert_feasible=False)
        elif design.infeasibility_flag:
             print("Skipping final terminate action due to infeasibility flag.")
             design.finalize(assert_feasible=False) # Still finalize to set SMILES etc.

        # --- Optional: Final SMILES Comparison ---
        if compare_smiles and smiles is not None and not design.infeasibility_flag:
            try:
                # Generate SMILES from the final internal state via RDKit
                final_mol_internal = design.to_rdkit_mol(sanitize=True)
                final_smiles_internal = Chem.MolToSmiles(final_mol_internal) if final_mol_internal else None

                # Generate canonical SMILES from the *original* input SMILES (heavy atoms)
                ref_mol_orig_noH = Chem.RemoveHs(Chem.MolFromSmiles(smiles))
                ref_smiles_canon = Chem.MolToSmiles(ref_mol_orig_noH) if ref_mol_orig_noH else None

                if final_smiles_internal and ref_smiles_canon and \
                   Chem.CanonSmiles(final_smiles_internal) != Chem.CanonSmiles(ref_smiles_canon):
                     print(f"WARNING: SMILES mismatch after construction")
                     print(f"  Constructed: {final_smiles_internal} -> Canon: {Chem.CanonSmiles(final_smiles_internal)}")
                     print(f"  Reference:   {smiles} -> Heavy Canon: {Chem.CanonSmiles(ref_smiles_canon)}")
            except Exception as smi_err:
                 print(f"Warning: Error during final SMILES comparison: {smi_err}")

        return design
