import copy
import numpy as np
import torch
from torch import nn
from rdkit import Chem, RDLogger
from rdkit.Chem import rdmolfiles, rdmolops
# import networkx as nx
# import collections

from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

# import traceback
from config import MoleculeConfig
from core.abstracts import BaseTrajectory # Assuming this import exists and is correct

from typing import List, Tuple, Dict, Optional

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')


class ActionType:
    """Enum to track the type of action taken at Level 1."""
    ADD_ATOM = 1
    SELECT_EXISTING_ATOM = 2
    REMOVE_SELECTED_ATOM = 3
    REPLACE_ATOM = 4


def build_reverse_atom_lookup(config: MoleculeConfig) -> Dict[Tuple[int, int, int], int]:
    """
    Creates a lookup dictionary mapping atom properties back to vocabulary indices.
    (Implementation remains the same)
    """
    lookup = {}
    vocab_names = list(config.atom_vocabulary.keys())

    # if not vocab_names:
    #     raise ValueError("Atom vocabulary in config appears empty.")

    for i, name in enumerate(vocab_names):
        try:
            atom_config = config.atom_vocabulary[name]
        except KeyError:
            raise KeyError(f"Atom name '{name}' found in vocab_names but not in atom_vocabulary keys.")

        try:
            atomic_num = atom_config['atomic_number']
            charge = atom_config.get('formal_charge', 0)
            # Map chiral tag from config (0, 1=@, 2=@@) to key value (0, 1, 2)
            chiral = atom_config.get('chiral_tag', 0)
        except KeyError as e:
            raise ValueError(f"Missing expected property {e} for atom '{name}' in config.")

        key = (atomic_num, charge, chiral)
        vocab_idx = i + 1  # 1-based index for internal use

        if key in lookup:
            # Allow overwriting, assumes config is consistent if duplicates exist
            # Could add a warning here if needed
            print("WARNING: Duplicate key found in atom vocabulary. Overwriting existing entry.")
            pass

        lookup[key] = vocab_idx

        # # Add fallback for non-chiral lookup if a chiral version exists
        # if chiral != 0:
        #     key_no_chiral = (atomic_num, charge, 0)
        #     if key_no_chiral not in lookup:
        #         lookup[key_no_chiral] = vocab_idx
        #     # raise ValueError(f"Chiral atom '{name}' with key {key} already exists in lookup. Check config for duplicates.")

    # if not lookup:
    #     raise ValueError("Reverse atom lookup is empty. Check atom_vocabulary in config.")

    # print(lookup)
    return lookup


class MoleculeDesign(BaseTrajectory):
    """
    Environment for molecular design with cycle prevention (state hashing),
    connectivity filtering (post-hoc), and rules to guide generation.

    Rule 1: Only atoms present in the initial molecule can be removed.
    Rule 2: Immediate reversal of bond actions on the same atom pair is forbidden.

    State Management: Internal NumPy arrays (self.atoms, self.bonds) are the primary source of truth.
                      NetworkX used for connectivity checks.
                      RDKit Mol object is constructed only during finalize() or to_smiles().

    Action Levels (Revised with Replace Atom):
        - Level 0: Terminate (if valid) or Select Existing Atom (index 1 to N).
        - Level 1 (Anchor Atom = A):
            - Add New Atom (Indices 0 to V-1): Choose atom type T from vocab, add T connected to A. -> Level 2
            - Select Existing Atom (Indices V to V+N-1): Choose existing atom B (B!=A, index 0 to N-1). -> Level 2
            - Replace Atom (Indices V+N to V+N+V-1): Choose atom type T' from vocab, replace A with T'. -> Level 0
            - Remove Selected Atom (Index V+N+V): Remove A (if original & N>1). -> Level 0
        - Level 2 (Atom Pair = A, B from L1): Set Bond Order 1-6 (creates if 0) or Remove Bond. -> Level 0
    """
    maximum_bond_order = 6
    virtual_bond_idx = 7 # Used for padding/virtual connections
    bond_types = {
        1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE,
        4: Chem.rdchem.BondType.QUADRUPLE, 5: Chem.rdchem.BondType.QUINTUPLE, 6: Chem.rdchem.BondType.HEXTUPLE
    }

    def __init__(self, config: MoleculeConfig, initial_atom: int):
        self.config = config
        self.atom_vocabulary = self.config.atom_vocabulary
        # Ensure consistency in vocab access
        if hasattr(config, 'vocabulary_atom_names'):
             self.vocabulary_atom_names = config.vocabulary_atom_names
        else:
             self.vocabulary_atom_names = list(self.atom_vocabulary.keys())
        self.vocab_size = len(self.vocabulary_atom_names)
        self.vocabulary_atom_idcs = list(range(1, self.vocab_size + 1)) # 1-based indices

        # Precompute valence and feasibility
        self.vocabulary_valence = [-1] * (self.vocab_size + 1) # Index 0 unused
        self.atom_feasibility_mask = [True] * self.vocab_size # True if *masked* (infeasible/disallowed)
        for i, name in enumerate(self.vocabulary_atom_names):
             vocab_idx = i + 1
             try:
                 self.vocabulary_valence[vocab_idx] = self.atom_vocabulary[name]["valence"]
                 if not self.atom_vocabulary[name].get("allowed", False):
                      self.atom_feasibility_mask[i] = True # Keep True if disallowed
                 else:
                      self.atom_feasibility_mask[i] = False # Set False if allowed
             except KeyError as e:
                  raise ValueError(f"Missing property {e} or name mismatch for '{name}' in atom_vocabulary.")

        self.upper_limit_atoms = self.config.max_num_atoms

        # Validate initial atom
        initial_atom_0_idx = initial_atom - 1
        # if not (0 <= initial_atom_0_idx < self.vocab_size and not self.atom_feasibility_mask[initial_atom_0_idx]):
        #      raise ValueError(f"Initial atom {initial_atom} (name: {self.vocabulary_atom_names[initial_atom_0_idx]}) must be in vocabulary and allowed in config.")
        self.initial_atom = initial_atom

        # --- Internal State (Primary) ---
        self.atoms = np.array([0, initial_atom], dtype=np.uint8) # Includes virtual atom 0 (index 0)
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
        # self.is_currently_connected: bool = True # Assume connected initially

        # --- Action Handling State ---
        self.current_action_level = 0
        self.current_action_mask: Optional[np.array] = None
        self.history: List[int] = []
        self.l0_selected_atom_idx: Optional[int] = None # 1-based internal index of the anchor atom
        self.l1_action_type: Optional[ActionType] = None # Store the type of L1 action taken
        self.l1_new_atom_type: Optional[int] = None # 1-based vocab index (used if ADD_ATOM)
        self.l1_selected_existing_atom_idx: Optional[int] = None # 1-based internal index (used if SELECT_EXISTING_ATOM)
        # --- Rule 2 State ---
        self.last_bond_action_details: Optional[Tuple[int, int]] = None # Stores (min_idx, max_idx) of last bond action pair

        # Initialize mask
        self.update_action_mask()

    def _check_connectivity_after_simulated_removal(self,
                                action_type: str,
                                atom_idx_to_remove: Optional[int] = None,
                                bond_indices_to_remove: Optional[Tuple[int, int]] = None) -> bool:

        """
        Checks connectivity using SciPy's connected_components after simulating removal.
        """
        num_real_atoms = len(self.atoms) - 1

        if action_type == "Remove Atom":
            if atom_idx_to_remove is None or not (1 <= atom_idx_to_remove <= num_real_atoms):
                return False  # Invalid index for simulation
            num_real_atoms_after_removal = num_real_atoms - 1
            if num_real_atoms_after_removal <= 1:
                return True  # Single atom or none left is connected

            # Create adjacency matrix, *excluding* the removed atom's row/col implicitly
            atom_0idx_to_remove = atom_idx_to_remove - 1
            indices_to_keep = [i for i in range(num_real_atoms) if i != atom_0idx_to_remove]
            # Ensure indices_to_keep is not empty before proceeding
            if not indices_to_keep:
                # This case implies removing the last atom, handled by num_real_atoms_after_removal check, but safe fallback.
                return True

            # Extract the submatrix corresponding to remaining atoms
            # Using advanced indexing creates a *copy* but might be faster than np.delete twice
            adj_matrix = self.bonds[1:, 1:][np.ix_(indices_to_keep, indices_to_keep)]

        elif action_type == "Remove Bond":
            if bond_indices_to_remove is None or len(bond_indices_to_remove) != 2: return False
            idx_A, idx_B = bond_indices_to_remove
            if not (1 <= idx_A <= num_real_atoms and 1 <= idx_B <= num_real_atoms): return False

            num_real_atoms_after_removal = num_real_atoms  # Atom count doesn't change
            if num_real_atoms_after_removal <= 1: return True

            # Copy the relevant part of the bond matrix and remove the bond
            adj_matrix = self.bonds[1:, 1:].copy()
            idx_A_0idx, idx_B_0idx = idx_A - 1, idx_B - 1
            adj_matrix[idx_A_0idx, idx_B_0idx] = adj_matrix[idx_B_0idx, idx_A_0idx] = 0

        else:
            raise ValueError(f"Invalid action_type for connectivity simulation: {action_type}")

        # # --- Check connectivity using SciPy ---
        # if adj_matrix.shape[0] == 0:  # Should be covered by num_real_atoms checks, but safe
        #     return True

        try:
            # Convert to sparse matrix for efficiency, ensure it's boolean or integer type
            adj_sparse = csr_matrix(adj_matrix > 0, dtype=int)  # Use >0 to get boolean adjacency
            n_components, labels = connected_components(csgraph=adj_sparse, directed=False,
                                                        return_labels=True)  # Get labels too

            # Check if all nodes belong to the same component (component 0)
            # This handles cases where adj_matrix might have isolated nodes after removal
            # We need to ensure all *expected* nodes are in the first component.
            # The number of nodes in the matrix IS num_real_atoms_after_removal.
            return n_components <= 1  # If 0 or 1 components, it's connected

        except Exception as e:
            # print(f"WARNING: Error during SciPy connectivity check ({action_type}): {e}")
            # return False # Assume disconnected on error
            # Keep consistent with original code's tendency to raise on unexpected errors
            raise ValueError(f"Error during SciPy connectivity check ({action_type}): {e}")

    def _get_smiles_for_check(self) -> Optional[str]:
        """
        Generates a canonical SMILES string for intermediate checks WITHOUT
        calling finalize() or modifying internal state caches/flags.
        Crucially, catches sanitization errors.
        Returns None if RDKit mol creation or sanitization fails.
        """
        try:
            temp_mol = self.to_rdkit_mol(sanitize=False)
            if temp_mol is None or temp_mol.GetNumAtoms() == 0:
                return "" # Empty molecule is valid

            # Explicitly try sanitization here
            sanitize_status = Chem.SanitizeMol(temp_mol, catchErrors=True)
            if sanitize_status != Chem.SanitizeFlags.SANITIZE_NONE:
                 # print(f"DEBUG: _get_smiles_for_check - Sanitization failed with status {sanitize_status}") # Optional Debug
                 return None # Sanitization failed

            # If sanitization succeeded, get canonical SMILES
            smiles = Chem.MolToSmiles(temp_mol, canonical=True)
            return smiles
        except Exception as e:
            # Catch errors during RDKit mol creation or SMILES generation
            # print(f"DEBUG: _get_smiles_for_check - Exception during mol/SMILES gen: {e}") # Optional Debug
            return None

    def _get_current_valence_usage(self, atom_internal_idx: Optional[int] = None) -> np.array:
        """
        Calculates the sum of explicit bond orders for each real atom (or a specific one)
        from self.bonds, ignoring virtual bonds.
        """
        num_real_atoms = len(self.atoms) - 1
        if num_real_atoms <= 0: return np.array([], dtype=int)

        if atom_internal_idx is not None:
            # Calculate for a single atom
            if not (1 <= atom_internal_idx <= num_real_atoms):
                 raise IndexError(f"Invalid internal index {atom_internal_idx} for valence usage.")
            # Sum bond orders for this atom with all *other* real atoms
            current_usage = np.sum(self.bonds[atom_internal_idx, 1 : num_real_atoms + 1])
            # Ensure diagonal (self-bond) is not counted if somehow present
            current_usage -= self.bonds[atom_internal_idx, atom_internal_idx]
            return np.array([int(current_usage)])
        else:
            # Calculate for all real atoms
            # Extract bond matrix for real atoms
            real_bonds = self.bonds[1 : num_real_atoms + 1, 1 : num_real_atoms + 1]
            # Sum bond orders along rows (axis=1)
            current_explicit_usage = np.sum(real_bonds, axis=1)
            return current_explicit_usage.astype(int)


    def _get_remaining_valence(self) -> np.array:
        """Calculates remaining valence for each real atom based on self.atoms and self.bonds."""
        num_real_atoms = len(self.atoms) - 1
        # if num_real_atoms <= 0: return np.array([], dtype=int)

        current_usage = self._get_current_valence_usage() # Gets usage for all real atoms

        try:
            # Get max valence for each atom based on its type index in self.atoms
            total_valence = np.array([self.vocabulary_valence[atom_vocab_idx]
                                      for atom_vocab_idx in self.atoms[1:]], dtype=int)
        except IndexError as e:
            raise IndexError(f"Invalid atom vocab index found in self.atoms[1:]: {self.atoms[1:]}. Error: {e}")

        if len(total_valence) != len(current_usage):
             raise RuntimeError(f"Valence calculation mismatch: total_valence ({len(total_valence)}) vs current_usage ({len(current_usage)})")

        remaining = total_valence - current_usage
        remaining = np.maximum(0, remaining) # Valence cannot be negative
        return remaining

    def update_action_mask(self):
        """Creates the action mask based on the internal state and rules.
           Includes proactive checks to mask removals that would disconnect the molecule.
        """
        if self.synthesis_done:
            self.current_action_mask = None
            return

        num_real_atoms = len(self.atoms) - 1

        # if num_real_atoms == 0:
        #     # This case should ideally not be reached if starting molecules are valid
        #     # and removal masking works correctly.
        #     raise ValueError("No real atoms present. Cannot update action mask.")

        remaining_valence = self._get_remaining_valence()  # Get remaining valence for all atoms

        # --- Level 0 Mask ---
        if self.current_action_level == 0:
            action_space_size = 1 + num_real_atoms # Terminate + Select Atom 1..N
            mask = np.zeros(action_space_size, dtype=bool) # Start with all False (unmasked)
            # Terminate is always allowed if connected and atoms exist
            self.current_action_mask = mask

        # --- Level 1 Mask ---
        elif self.current_action_level == 1:
            action_space_size = 2 * self.vocab_size + num_real_atoms + 1
            mask = np.ones(action_space_size, dtype=bool) # Start with all True (masked)

            anchor_atom_internal_idx = self.l0_selected_atom_idx
            # if anchor_atom_internal_idx is None or not (1 <= anchor_atom_internal_idx <= num_real_atoms):
            #     raise ValueError(f"L1 Mask Error: Invalid anchor atom index: {anchor_atom_internal_idx} (NumReal={num_real_atoms})")
            anchor_atom_0_idx = anchor_atom_internal_idx - 1

            # --- Unmask "Add Atom" (Indices 0 to V-1) ---
            # (Logic unchanged)
            if (self.upper_limit_atoms is None or num_real_atoms < self.upper_limit_atoms) and \
               remaining_valence[anchor_atom_0_idx] > 0:
                for i in range(self.vocab_size):
                    action_idx = i
                    atom_type_vocab_idx = i + 1
                    if not self.atom_feasibility_mask[i] and self.vocabulary_valence[atom_type_vocab_idx] >= 1:
                        mask[action_idx] = False

            # --- Unmask "Select Existing Atom" (Indices V to V+N-1) ---
            # (Logic unchanged)
            for target_0_idx in range(num_real_atoms):
                target_internal_idx = target_0_idx + 1
                action_idx = self.vocab_size + target_0_idx
                if target_internal_idx == anchor_atom_internal_idx: continue
                bond_exists = self.bonds[anchor_atom_internal_idx, target_internal_idx] > 0
                can_form_new = (remaining_valence[anchor_atom_0_idx] > 0 and remaining_valence[target_0_idx] > 0)
                if bond_exists or can_form_new:
                    mask[action_idx] = False

            # --- Unmask "Replace Atom" (Indices V+N to V+N+V-1) ---
            # (Logic unchanged)
            replace_start_idx = self.vocab_size + num_real_atoms
            current_atom_vocab_idx = self.atoms[anchor_atom_internal_idx]
            current_anchor_usage = self._get_current_valence_usage(anchor_atom_internal_idx)[0]
            for i in range(self.vocab_size):
                action_idx = replace_start_idx + i
                replacement_vocab_idx = i + 1
                if replacement_vocab_idx == current_atom_vocab_idx: continue
                if self.atom_feasibility_mask[i]: continue
                replacement_max_valence = self.vocabulary_valence[replacement_vocab_idx]
                if current_anchor_usage <= replacement_max_valence:
                    mask[action_idx] = False

            # --- Unmask "Remove Selected Atom" (Index V+N+V) ---
            remove_action_idx = 2 * self.vocab_size + num_real_atoms
            # Check index bounds before accessing is_original_atom
            if anchor_atom_internal_idx < len(self.is_original_atom):
                # Condition 1: More than 1 atom exists
                # Condition 2: Anchor is an original atom (Rule 1)
                if num_real_atoms > 1 and self.is_original_atom[anchor_atom_internal_idx]:
                    # Condition 3: Removing this atom would NOT disconnect the molecule
                    if self._check_connectivity_after_simulated_removal(action_type="Remove Atom",
                                                                        atom_idx_to_remove=anchor_atom_internal_idx):
                        mask[remove_action_idx] = False
            else:
                raise IndexError(f"L1 Mask Error: anchor_atom_internal_idx {anchor_atom_internal_idx} OOB for is_original_atom (len {len(self.is_original_atom)})")

            self.current_action_mask = mask

        # --- Level 2 Mask ---
        elif self.current_action_level == 2:
            action_space_size = 7 # Set Bond 1-6, Remove Bond
            mask = np.ones(action_space_size, dtype=bool) # Start masked

            atom_A_internal_idx = self.l0_selected_atom_idx
            # atom_B_internal_idx = -1
            if self.l1_action_type == ActionType.ADD_ATOM:
                atom_B_internal_idx = len(self.atoms) - 1
            elif self.l1_action_type == ActionType.SELECT_EXISTING_ATOM:
                atom_B_internal_idx = self.l1_selected_existing_atom_idx
            else:
                raise ValueError(f"L2 Mask Error: Invalid L1 action type context ({self.l1_action_type}).")

            # if not (1 <= atom_A_internal_idx <= num_real_atoms and 1 <= atom_B_internal_idx <= num_real_atoms):
            #     raise ValueError(f"L2 Bond Mask Error: Invalid indices A={atom_A_internal_idx}, B={atom_B_internal_idx} (NumReal={num_real_atoms})")

            # --- Rule 2 Check: Prevent immediate reversal ---
            # (Logic unchanged)
            current_min_idx = min(atom_A_internal_idx, atom_B_internal_idx)
            current_max_idx = max(atom_A_internal_idx, atom_B_internal_idx)
            if (self.last_bond_action_details is not None and self.last_bond_action_details[0] == current_min_idx and
                    self.last_bond_action_details[1] == current_max_idx):
                mask[:] = True
                self.current_action_mask = mask
                return

            # --- Normal L2 Mask Logic (Valence Check for Set Bond) ---
            # (Logic unchanged)
            atom_A_0_idx = atom_A_internal_idx - 1
            atom_B_0_idx = atom_B_internal_idx - 1
            if atom_A_0_idx >= len(remaining_valence) or atom_B_0_idx >= len(remaining_valence):
                raise IndexError(f"L2 Bond Mask Error: Indices {atom_A_0_idx} or {atom_B_0_idx} OOB for rem_val (len {len(remaining_valence)}).")

            current_bond_order = self.bonds[atom_A_internal_idx, atom_B_internal_idx]
            valence_A_rem = remaining_valence[atom_A_0_idx]
            valence_B_rem = remaining_valence[atom_B_0_idx]
            max_increase = min(valence_A_rem, valence_B_rem)
            effective_current_order = int(current_bond_order) if current_bond_order > 0 else 0
            max_allowed_final_order = min(effective_current_order + max_increase, self.maximum_bond_order)

            for order in range(1, self.maximum_bond_order + 1):
                action_idx = order - 1
                if order <= max_allowed_final_order:
                    mask[action_idx] = False

            # --- Unmask "Remove Bond" action (Index 6) ---
            # Condition 1: A bond currently exists
            if current_bond_order > 0:
                # Condition 2: Removing this bond would NOT disconnect the molecule
                # (Only relevant if there's more than 1 real atom)
                if num_real_atoms <= 1 or self._check_connectivity_after_simulated_removal(action_type="Remove Bond",
                                                                                           bond_indices_to_remove=(atom_A_internal_idx, atom_B_internal_idx)):
                    mask[6] = False

            self.current_action_mask = mask
        else:
            raise ValueError(f"Invalid current_action_level: {self.current_action_level}")

    def _adjust_indices_after_removal(self, removed_internal_idx: int):
        """Adjusts stored internal indices after an atom removal."""
        # Adjust L0 anchor if it was after the removed atom
        if self.l0_selected_atom_idx is not None and self.l0_selected_atom_idx > removed_internal_idx:
            self.l0_selected_atom_idx -= 1
        # Adjust L1 selected existing atom if it was after the removed atom
        if self.l1_selected_existing_atom_idx is not None and self.l1_selected_existing_atom_idx > removed_internal_idx:
            self.l1_selected_existing_atom_idx -= 1
        # Adjust Rule 2 tracker if needed
        if self.last_bond_action_details is not None:
            min_idx, max_idx = self.last_bond_action_details
            # If removed atom was part of the pair, invalidate the tracker
            if min_idx == removed_internal_idx or max_idx == removed_internal_idx:
                self.last_bond_action_details = None
            else:
                # Otherwise, adjust indices if they were after the removed one
                new_min = min_idx - 1 if min_idx > removed_internal_idx else min_idx
                new_max = max_idx - 1 if max_idx > removed_internal_idx else max_idx
                if new_min != min_idx or new_max != max_idx:
                     self.last_bond_action_details = (new_min, new_max) # Update adjusted indices

    def take_action(self, action: int):
        """Execute a given action, updating internal state directly."""
        if self.synthesis_done:
            # Changed to ValueError to match user preference in other exceptions
            raise ValueError("Cannot take action on terminated design.")

        # # Validate action against mask
        # if self.current_action_mask is None or not (0 <= action < len(self.current_action_mask)) or self.current_action_mask[action]:
        #     mask_len = "None" if self.current_action_mask is None else len(self.current_action_mask)
        #     raise ValueError(
        #         f"Action {action} masked or invalid for level {self.current_action_level}. MaskLen={mask_len}")

        current_level = self.current_action_level
        next_level = 0  # Default next level is 0 unless specified otherwise
        self.history.append(int(action))
        num_real_atoms_before = len(self.atoms) - 1

        try:
            reset_last_bond_action = True  # Default: reset Rule 2 tracker unless it's an L2 bond action

            # --- Level 0 Actions ---
            if current_level == 0:
                if action == 0:  # Terminate
                    # External check (is_terminable) should happen before calling take_action(0)
                    self.synthesis_done = True
                    # User change kept: assert_feasible=True
                    self.finalize(assert_feasible=True)  # Final sanitization happens here
                    next_level = -1  # Special level indicating termination
                else:  # Select Atom (action = 1 to N)
                    selected_internal_idx = action
                    # if not (1 <= selected_internal_idx <= num_real_atoms_before):
                    #     raise ValueError(f"L0 Select Atom: Invalid index {action} for {num_real_atoms_before} atoms.")
                    self.l0_selected_atom_idx = selected_internal_idx
                    # Reset L1 context
                    self.l1_action_type = None
                    self.l1_new_atom_type = None
                    self.l1_selected_existing_atom_idx = None
                    next_level = 1  # Transition to Level 1

            # --- Level 1 Actions ---
            elif current_level == 1:
                anchor_idx = self.l0_selected_atom_idx
                # if anchor_idx is None: # Should be set from L0
                #     # User change kept: ValueError
                #     raise ValueError("L1 take_action: l0_selected_atom_idx (anchor) is None.")

                # Define index boundaries based on the new structure
                add_atom_end_idx = self.vocab_size
                select_existing_end_idx = self.vocab_size + num_real_atoms_before
                replace_atom_end_idx = select_existing_end_idx + self.vocab_size
                remove_atom_idx = replace_atom_end_idx  # The single index for remove

                # 1. Add Atom (0 <= action < V)
                if action < add_atom_end_idx:
                    self.l1_action_type = ActionType.ADD_ATOM
                    self.l1_new_atom_type = action + 1  # 1-based vocab index
                    # Append new atom
                    self.atoms = np.append(self.atoms, self.l1_new_atom_type)
                    new_atom_internal_idx = len(self.atoms) - 1
                    # Expand bonds matrix
                    self.bonds = np.pad(self.bonds, [(0, 1), (0, 1)], 'constant', constant_values=0)
                    # Add virtual bond for the new atom
                    self.bonds[0, new_atom_internal_idx] = self.bonds[new_atom_internal_idx, 0] = self.virtual_bond_idx
                    # Expand and update original atom tracker
                    self.is_original_atom = np.append(self.is_original_atom, False)
                    next_level = 2  # Transition to Level 2 for bond setting

                # 2. Select Existing Atom (V <= action < V+N)
                elif action < select_existing_end_idx:
                    # Calculate target atom index (0-based relative to existing real atoms)
                    target_0_idx = action - self.vocab_size
                    selected_internal_idx = target_0_idx + 1  # Convert to 1-based internal index

                    # if selected_internal_idx == anchor_idx: # Should be masked, but double check
                    #     raise ValueError("L1 Select Existing: Cannot select anchor atom itself.")
                    # if not (1 <= selected_internal_idx <= num_real_atoms_before): # Validate index
                    #     raise ValueError(f"L1 Select Existing: Invalid target index {selected_internal_idx} for {num_real_atoms_before} atoms.")

                    self.l1_action_type = ActionType.SELECT_EXISTING_ATOM
                    self.l1_selected_existing_atom_idx = selected_internal_idx
                    next_level = 2  # Transition to Level 2 for bond setting

                # 3. Replace Atom (V+N <= action < V+N+V)
                elif action < replace_atom_end_idx:
                    # Calculate replacement atom type (1-based vocab index)
                    replacement_vocab_idx = (action - select_existing_end_idx) + 1

                    # # Basic validation (should be guaranteed by mask, but good to double-check)
                    # current_atom_vocab_idx = self.atoms[anchor_idx]
                    # if replacement_vocab_idx == current_atom_vocab_idx:
                    #      raise ValueError("L1 Replace Atom: Attempted to replace with the same type.")
                    # replacement_0_idx = replacement_vocab_idx - 1
                    # if not (0 <= replacement_0_idx < self.vocab_size and not self.atom_feasibility_mask[replacement_0_idx]):
                    #      raise ValueError(f"L1 Replace Atom: Replacement type {replacement_vocab_idx} is invalid or disallowed.")

                    # --- Execute Replacement ---
                    self.atoms[anchor_idx] = replacement_vocab_idx
                    self.is_original_atom[anchor_idx] = False  # Mark as no longer original
                    self.l1_action_type = ActionType.REPLACE_ATOM

                    # --- Post-Replacement Sanitization Check REMOVED ---
                    # smiles_check = self._get_smiles_for_check()
                    # if smiles_check is None:
                    #     self.infeasibility_flag = True # Mark state as infeasible
                    #     raise RuntimeError("Post-replacement sanitization failed.") # Raise error to stop sequence

                    # Reset L1/L2 context variables
                    self.l0_selected_atom_idx = None
                    self.l1_new_atom_type = None
                    self.l1_selected_existing_atom_idx = None
                    # next_level remains 0 (transition back to L0 after replacement)

                # 4. Remove Selected Atom (action == V+N+V)
                elif action == remove_atom_idx:
                    # Validation (should be guaranteed by mask)
                    # if num_real_atoms_before <= 1:
                    #     # User change kept: ValueError
                    #     raise ValueError("L1 Remove Atom: Attempted to remove last real atom.")
                    # if not self.is_original_atom[anchor_idx]:
                    #     # User change kept: ValueError
                    #     raise ValueError("L1 Remove Atom: Attempted to remove non-original atom (Rule 1 violation).")

                    self.l1_action_type = ActionType.REMOVE_SELECTED_ATOM
                    removed_idx_for_adjust = anchor_idx  # Store index before deletion

                    # --- Execute Removal ---
                    self.atoms = np.delete(self.atoms, removed_idx_for_adjust)
                    self.bonds = np.delete(np.delete(self.bonds, removed_idx_for_adjust, axis=0),
                                           removed_idx_for_adjust, axis=1)
                    self.is_original_atom = np.delete(self.is_original_atom, removed_idx_for_adjust)
                    self._adjust_indices_after_removal(removed_idx_for_adjust)  # Adjust stored indices

                    # <<< --- Post-Atom-Removal Sanitization Check REMOVED --- >>>
                    # smiles_check = self._get_smiles_for_check()
                    # if smiles_check is None:
                    #     self.infeasibility_flag = True # Mark state as infeasible
                    #     raise RuntimeError("Post-atom-removal sanitization failed.") # Raise error to stop sequence
                    # <<< --- END REMOVED CHECK --- >>>

                    # Reset L1/L2 context
                    self.l0_selected_atom_idx = None
                    self.l1_new_atom_type = None
                    self.l1_selected_existing_atom_idx = None
                    # next_level remains 0 (transition back to L0 after removal)

                else:  # Should not be reachable if mask is correct
                    raise ValueError(f"Invalid L1 action index: {action}")

            # --- Level 2 Actions ---
            elif current_level == 2:
                reset_last_bond_action = False  # It's a bond action, update Rule 2 tracker
                idx_A = self.l0_selected_atom_idx
                idx_B = -1
                # Determine idx_B based on the L1 action that led here
                if self.l1_action_type == ActionType.ADD_ATOM:
                    idx_B = len(self.atoms) - 1  # Newly added atom is last
                elif self.l1_action_type == ActionType.SELECT_EXISTING_ATOM:
                    idx_B = self.l1_selected_existing_atom_idx
                else:  # Should not happen
                    # User change kept: ValueError
                    raise ValueError(f"L2 take_action: Invalid L1 action type context ({self.l1_action_type}).")

                # Validate indices A and B
                current_num_real_atoms = len(self.atoms) - 1
                # if not (1 <= idx_A <= current_num_real_atoms and 1 <= idx_B <= current_num_real_atoms):
                #     raise ValueError(f"L2 take_action: Invalid indices A={idx_A}, B={idx_B} for {current_num_real_atoms} atoms.")

                # Execute bond action
                if 0 <= action <= 5:  # Set Bond Order (action 0 -> order 1, ..., action 5 -> order 6)
                    order = action + 1
                    self.bonds[idx_A, idx_B] = self.bonds[idx_B, idx_A] = order
                    # NO sanitization check needed here (was never present)

                elif action == 6:  # Remove Bond
                    self.bonds[idx_A, idx_B] = self.bonds[idx_B, idx_A] = 0

                    # <<< --- Post-Bond-Removal Sanitization Check REMOVED --- >>>
                    # smiles_check = self._get_smiles_for_check()
                    # if smiles_check is None:
                    #     self.infeasibility_flag = True # Mark state as infeasible
                    #     raise RuntimeError("Post-bond-removal sanitization failed.") # Raise error to stop sequence
                    # <<< --- END REMOVED CHECK --- >>>

                else:  # Should not be reachable
                    raise ValueError(f"Invalid L2 Bond action index: {action}")

                # Update Rule 2 tracker (applies to both Set and Remove bond actions)
                self.last_bond_action_details = (min(idx_A, idx_B), max(idx_A, idx_B))

                # Reset L1/L2 context
                self.l0_selected_atom_idx = None
                self.l1_action_type = None
                self.l1_new_atom_type = None
                self.l1_selected_existing_atom_idx = None
                # next_level remains 0 (transition back to L0 after bond action)

            # --- Reset Rule 2 tracker if it wasn't a bond action ---
            if reset_last_bond_action:
                self.last_bond_action_details = None

            # --- Update State After Action (if not terminated) ---
            if next_level != -1:
                # Update current level BEFORE updating mask
                self.current_action_level = next_level
                # Update the mask for the new level
                self.update_action_mask()  # Can raise IndexError/ValueError
            else:
                # Action was Terminate, clear mask
                self.current_action_mask = None

        # --- Exception Handling ---
        except (ValueError, IndexError) as e:
            # Errors related to invalid action indices, masking logic, array bounds
            self.infeasibility_flag = True  # Mark state as infeasible
            self.current_action_mask = None  # Prevent further actions
            # Re-raise to signal sequence failure (User change kept: ValueError)
            raise ValueError(f"Masking/Action logic error at L{current_level}, action {action}: {e}") from e
        except RuntimeError as e:
            # Catch specific RuntimeErrors (e.g., from OUR sanitization checks, though now only in finalize)
            # Infeasibility flag should already be set if raised internally by us
            if not self.infeasibility_flag: self.infeasibility_flag = True
            self.current_action_mask = None
            # Re-raise the original RuntimeError
            raise e
        except Exception as e:
            # Catch any other unexpected errors during action execution
            # print(f"CRITICAL: Unexpected error in take_action(action={action}, L{current_level}): {e}") # Optional Debug
            self.infeasibility_flag = True
            self.current_action_mask = None
            # Re-raise as ValueError to ensure generate_single_transformation catches it (User change kept: ValueError)
            raise ValueError(f"Unexpected error during action execution: {e}") from e

    def finalize(self, assert_feasible: bool = False):
        """Finalize molecule design: build RDKit mol, sanitize, cache SMILES."""
        # Avoid re-finalizing
        if self._cached_smiles is not None or self._cached_rdkit_mol is not None: return

        # Optional feasibility assertion
        if assert_feasible:
            try: self.assert_feasible()
            except AssertionError as e:
                print(f"Warning: Feasibility assertion failed during finalize: {e}"); self.infeasibility_flag = True
                raise RuntimeError("Feasibility assertion failed during finalize.") from e

        num_real_atoms = len(self.atoms) - 1
        # # Mark disconnected molecules (>1 atom) as infeasible during finalize
        # if num_real_atoms > 1 and not self.is_currently_connected:
        #     # print("Warning: Final molecule is disconnected.") # Optional
        #     self.infeasibility_flag = True
        #     raise ValueError("Final molecule is disconnected.")  # indicate disconnectedness, should not happen

        # Attempt to generate RDKit Mol and SMILES only if not already flagged infeasible
        rdkit_mol = None
        if not self.infeasibility_flag:
            try:
                # 1. Generate RDKit Mol (unsanitized first)
                rdkit_mol = self.to_rdkit_mol(sanitize=False)

                # Check if mol creation failed or resulted in empty mol unexpectedly
                if rdkit_mol is None or (rdkit_mol.GetNumAtoms() == 0 and num_real_atoms > 0):
                    print("Warning: RDKit mol creation failed or empty despite internal atoms."); self.infeasibility_flag = True
                elif rdkit_mol.GetNumAtoms() > 0:
                    # 2. Attempt Sanitization and SMILES generation
                    try:
                        self._cached_rdkit_mol = copy.deepcopy(rdkit_mol) # Cache unsanitized version first
                        # Attempt sanitization
                        sanitize_status = Chem.SanitizeMol(rdkit_mol, catchErrors=True)
                        if sanitize_status != Chem.SanitizeFlags.SANITIZE_NONE:
                             print(f"Warning: Final sanitization failed with status {sanitize_status}.")
                             self._cached_smiles = None # Ensure SMILES is None if sanitize fails
                             # Keep the unsanitized mol in cache? Optional. For now, clear SMILES.
                             self.infeasibility_flag = True # Optionally mark sanitize failure as infeasible
                        else:
                             # Sanitization succeeded, update cache and get SMILES
                             self._cached_rdkit_mol = rdkit_mol # Overwrite cache with sanitized version
                             self._cached_smiles = Chem.MolToSmiles(rdkit_mol, canonical=True)
                    except Exception as e:
                        # Catch errors during SanitizeMol or MolToSmiles
                        print(f"Warning: Final sanitization/SMILES generation failed: {e}.")
                        self._cached_smiles = None
                        # If SMILES failed, likely infeasible
                        if self._cached_rdkit_mol is None: self.infeasibility_flag = True
                else: # 0 real atoms resulted in 0 RDKit atoms
                    # self._cached_smiles = "" # Empty SMILES for empty molecule
                    # self._cached_rdkit_mol = rdkit_mol # Cache the empty mol
                    raise ValueError("Empty RDKit mol with >0 real atoms.") # Raise to indicate failure
            except Exception as e:
                 # Catch errors during to_rdkit_mol itself
                 print(f"Warning: Error during RDKit mol generation in finalize: {e}")
                 self.infeasibility_flag = True; self._cached_smiles = None; self._cached_rdkit_mol = None
        else:
            # If already infeasible, ensure caches are None
            self._cached_smiles = None; self._cached_rdkit_mol = None

        # Mark synthesis as done regardless of success/failure of finalization steps
        self.synthesis_done = True

    def assert_feasible(self):
        """Check internal state consistency. Raises AssertionError on failure."""
        # (Implementation remains the same)
        if not isinstance(self.atoms, np.ndarray) or not isinstance(self.bonds, np.ndarray) or not isinstance(self.is_original_atom, np.ndarray):
             raise AssertionError("Internal state types incorrect.")
        assert self.atoms[0] == 0, "Virtual atom missing/incorrect."
        num_atoms = len(self.atoms); num_real_atoms = num_atoms - 1
        assert len(self.is_original_atom) == num_atoms, "is_original_atom length mismatch."
        assert not self.is_original_atom[0], "Virtual atom marked as original."

        if num_real_atoms > 0:
             valid_indices = all(1 <= idx <= self.vocab_size for idx in self.atoms[1:])
             assert valid_indices, f"Invalid atom vocab index found: {self.atoms[1:]}"
             allowed_check = all(not self.atom_feasibility_mask[idx - 1] for idx in self.atoms[1:])
             assert allowed_check, f"Disallowed atom type found: {self.atoms[1:]}"
        if self.upper_limit_atoms is not None: assert num_real_atoms <= self.upper_limit_atoms, f"Max atoms exceeded ({num_real_atoms} > {self.upper_limit_atoms})."
        assert self.bonds.shape == (num_atoms, num_atoms), f"Bonds shape mismatch ({self.bonds.shape} vs ({num_atoms},{num_atoms}))."
        assert not np.any(self.bonds.diagonal()), "Self-loops detected in bonds diagonal."
        assert np.all(self.bonds == self.bonds.T), "Bond matrix not symmetric."

        # Check virtual bonds explicitly
        if num_atoms > 1:
             assert np.all(self.bonds[0, 1:] == self.virtual_bond_idx), "Virtual bond missing/incorrect in row 0."
             assert np.all(self.bonds[1:, 0] == self.virtual_bond_idx), "Virtual bond missing/incorrect in col 0."
        # Check real bonds bounds
        if num_real_atoms > 0:
             real_bonds = self.bonds[1:, 1:]
             assert np.all(real_bonds <= self.maximum_bond_order), f"Bond order > {self.maximum_bond_order} found."

        # Check valence constraints
        if num_real_atoms > 0:
             try:
                  remaining_valence = self._get_remaining_valence()
                  assert np.all(remaining_valence >= 0), f"Valence constraints violated (negative remaining): {remaining_valence}"
             except (IndexError, RuntimeError) as e: raise AssertionError(f"Valence check failed during assertion: {e}")

    def to_rdkit_mol(self, sanitize=True) -> Chem.RWMol:
        """Creates a *new* RDKit molecule from the internal state."""
        # (Implementation remains the same)
        mol = Chem.RWMol()
        num_total_atoms = len(self.atoms)
        if num_total_atoms <= 1: return mol # Return empty mol if no real atoms

        rdkit_idx_map = {} # Maps internal 1-based index to RDKit 0-based index
        for internal_idx, atom_vocab_idx in enumerate(self.atoms):
            if internal_idx == 0: continue # Skip virtual atom

            # # Validate vocab index
            # if not (1 <= atom_vocab_idx <= self.vocab_size):
            #     raise ValueError(f"Invalid vocab index {atom_vocab_idx} at internal index {internal_idx} during RDKit conversion.")

            try:
                # Get atom properties from config using the name derived from vocab index
                atom_name = self.vocabulary_atom_names[atom_vocab_idx - 1]
                atom_config = self.atom_vocabulary[atom_name]
            except (IndexError, KeyError) as e:
                raise RuntimeError(f"Cannot get config for vocab index {atom_vocab_idx} (name: {atom_name}): {e}")

            # Create RDKit atom
            a = Chem.Atom(atom_config["atomic_number"])
            if "formal_charge" in atom_config: a.SetFormalCharge(atom_config["formal_charge"])
            # Set chirality based on config tag (0=None, 1=@/R, 2=@@/S)
            # RDKit: 1=CW(R), 2=CCW(S)
            ct = atom_config.get("chiral_tag", 0)
            if ct == 1: a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW) # R
            elif ct == 2: a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW) # S
            else: a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)

            # Add atom to RDKit mol and store mapping
            new_rdkit_idx = mol.AddAtom(a)
            rdkit_idx_map[internal_idx] = new_rdkit_idx

        # Add bonds between real atoms
        for i in range(1, num_total_atoms):
            for j in range(i + 1, num_total_atoms): # Avoid double counting and self-loops
                bond_order = self.bonds[i, j]
                # Add bond if order is valid (1 to max)
                if 1 <= bond_order <= self.maximum_bond_order:
                    # Get corresponding RDKit indices
                    if i not in rdkit_idx_map or j not in rdkit_idx_map:
                        # This should not happen if loop is correct
                        raise RuntimeError(f"Missing RDKit map entry for internal index {i} or {j}.")
                    rdkit_i, rdkit_j = rdkit_idx_map[i], rdkit_idx_map[j]

                    # Get RDKit bond type from order
                    rdkit_bond_type = self.bond_types.get(int(bond_order))
                    if rdkit_bond_type:
                        mol.AddBond(rdkit_i, rdkit_j, rdkit_bond_type)
                    else:
                        # Should not happen if bond_types dict is complete
                        # print(f"Warning: Could not find RDKit bond type for order {bond_order} between internal atoms {i},{j}.")
                        raise RuntimeError(f"Missing RDKit bond type for order {bond_order} between internal atoms {i},{j}.")
                elif bond_order > self.maximum_bond_order and bond_order != self.virtual_bond_idx:
                     # Log invalid bond orders found in matrix (excluding virtual)
                     # print(f"Warning: Invalid bond order {bond_order} found between internal atoms {i},{j} during RDKit conversion.")
                    raise RuntimeError(f"Invalid bond order {bond_order} found between internal atoms {i},{j}.")

        # Optional sanitization
        if sanitize:
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                # Don't raise here, allow returning unsanitized mol, but warn
                print(f"Warning: RDKit sanitization failed during to_rdkit_mol: {e}")
        return mol

    def is_terminable(self):
        """Checks if the current state allows termination."""
        num_real_atoms = len(self.atoms) - 1
        # Can terminate if:
        # - Currently at Level 0 AND
        # - Not already terminated AND
        # - >0 real atoms AND connected
        can_terminate = self.current_action_level == 0 and not self.synthesis_done
        # structure_ok =  (num_real_atoms == 0) or (num_real_atoms > 0 and self.is_currently_connected)
        structure_ok = num_real_atoms > 0
        return can_terminate and structure_ok

    def to_smiles(self, canonical: bool = True) -> Optional[str]:
        """Returns a canonical SMILES string. Finalizes if needed. Caches result."""
        # Ensure molecule is finalized before returning SMILES
        if not self.synthesis_done:
             self.finalize(assert_feasible=False) # Finalize handles caching

        # Return cached SMILES if available and canonical required
        if canonical and self._cached_smiles is not None:
             return self._cached_smiles
        # If non-canonical needed or cache miss, try generating from cached mol
        elif self._cached_rdkit_mol is not None:
             try:
                 # Work on a copy to avoid modifying cache if sanitize needed again
                 mol_to_use = copy.deepcopy(self._cached_rdkit_mol)
                 # Ensure sanitization before generating SMILES if not guaranteed by finalize
                 sanitize_status = Chem.SanitizeMol(mol_to_use, catchErrors=True)
                 if sanitize_status != Chem.SanitizeFlags.SANITIZE_NONE:
                      print(f"Warning: Sanitization failed during to_smiles call (status {sanitize_status}).")
                      return None # Cannot generate valid SMILES

                 smiles = Chem.MolToSmiles(mol_to_use, canonical=canonical)
                 # Update canonical cache if generated
                 if canonical: self._cached_smiles = smiles
                 return smiles
             except Exception as e:
                 print(f"Warning: Failed to generate SMILES (canonical={canonical}) from cached mol: {e}")
                 return None
        else:
             # If no cached mol (e.g., finalization failed completely), return None
             return self._cached_smiles # Which should be None or ""

    # --- Methods below likely used by RL framework ---

    @staticmethod
    def init_batch_from_instance_list(config: MoleculeConfig, instances: List[int], network: Optional[nn.Module]=None, device: Optional[torch.device]=None):
        """Creates a list of MoleculeDesign instances from initial atom types."""
        # (Implementation remains the same)
        return [MoleculeDesign(config=config, initial_atom=atom_type) for atom_type in instances]

    @staticmethod
    def log_probability_fn(trajectories: List['MoleculeDesign'], network: nn.Module) -> List[np.array]:
        """Calculates masked log probabilities for the current action level using a network."""
        # (Implementation remains the same - relies on correct mask generation)
        log_probs_to_return: List[np.array] = []
        network.eval() # Set network to evaluation mode
        with torch.no_grad(): # Disable gradient calculation
            # Convert list of molecules to a batch dictionary for the network
            batch = MoleculeDesign.list_to_batch(molecules=trajectories, device=network.device) # Assuming network has a device attribute
            # Get logits from the network for all levels
            batch_logits_l0, batch_logits_l1, batch_logits_l2 = network(batch)
            # Move logits to CPU and convert to NumPy for easier handling
            batch_logits_l0 = batch_logits_l0.cpu().numpy()
            batch_logits_l1 = batch_logits_l1.cpu().numpy()
            batch_logits_l2 = batch_logits_l2.cpu().numpy()

            # Process each trajectory in the batch
            for i, mol in enumerate(trajectories):
                mask = mol.current_action_mask
                # If mask is None (e.g., terminated), return empty array
                if mask is None:
                    log_probs_to_return.append(np.array([])); continue

                # Select the appropriate logits based on the current level
                logits = None
                if mol.current_action_level == 0: logits = batch_logits_l0[i]
                elif mol.current_action_level == 1: logits = batch_logits_l1[i]
                elif mol.current_action_level == 2: logits = batch_logits_l2[i]
                else: # Invalid level
                    log_probs_to_return.append(np.array([])); continue

                # Ensure logits match mask length (handle potential padding differences)
                mask_len = len(mask)
                if len(logits) > mask_len:
                    logits = logits[:mask_len] # Truncate logits if longer
                elif len(logits) < mask_len:
                    # This indicates a mismatch between network output size and expected action space size
                    raise ValueError(f"Logits/Mask length mismatch L{mol.current_action_level}: {len(logits)} vs {mask_len}")

                # Apply mask (set masked actions to -infinity)
                logits[mask] = -np.inf
                # Calculate log probabilities using log-softmax trick for numerical stability
                max_logit = np.max(logits)
                if np.isneginf(max_logit): # If all actions are masked
                    log_probs = logits # Keep as -inf
                else:
                    exp_logits = np.exp(logits - max_logit)
                    log_sum_exp = np.log(np.sum(exp_logits))
                    log_probs = logits - (max_logit + log_sum_exp)
                    log_probs[mask] = -np.inf # Ensure masked actions remain -inf

                log_probs_to_return.append(log_probs)
        return log_probs_to_return

    def transition_fn(self, action: int) -> Tuple['BaseTrajectory', bool]:
        """Creates a copy, takes the action, and returns the new state and termination status."""
        # (Implementation remains the same)
        copied_molecule = copy.deepcopy(self)
        try:
            # take_action modifies the copied_molecule in place
            # It raises errors (ValueError, IndexError, RuntimeError) on invalid actions or internal failures
            copied_molecule.take_action(action)
        except (ValueError, IndexError) as e:
            # Errors related to invalid/masked actions - indicates problem in caller logic
            # Re-raise these immediately as they shouldn't occur if caller uses mask correctly
            raise e
        except RuntimeError as e:
            # Catch RuntimeErrors from within take_action (e.g., sanitization failure)
            # The take_action method should have set infeasibility_flag
            # Ensure synthesis_done is True only if Terminate action was the cause
            if not copied_molecule.synthesis_done: # If error wasn't Terminate action
                 copied_molecule.synthesis_done = True # Mark as done to stop trajectory
                 copied_molecule.current_action_mask = None # Clear mask
            print(f"Warning: transition_fn caught RuntimeError in take_action({action}): {e}. Returning infeasible state.")
        # Do not catch generic 'except Exception' to let unexpected errors propagate

        # Return the modified copy and its termination status
        return copied_molecule, copied_molecule.synthesis_done

    def to_max_evaluation_fn(self) -> float:
        """Returns the objective value, penalizing infeasible states."""
        # (Implementation remains the same)
        if self.objective is None: return float("-inf")
        # Return negative infinity if the state is flagged as infeasible
        return float("-inf") if self.infeasibility_flag else self.objective

    def num_actions(self) -> int:
        """Returns the number of valid (unmasked) actions at the current level."""
        if self.current_action_mask is None: return 0
        # Count False entries in the boolean mask
        return int(np.sum(~self.current_action_mask))

    @staticmethod
    def list_to_batch(list_of_samples: List[Dict], device: torch.device = None) -> dict:
        """
        Converts a list of sample dictionaries [{'molecule': MoleculeDesign, 'target_action': int}, ...]
        to a batch dictionary suitable for network input.

        Handles padding of atoms, bonds, attention masks, feasibility masks, and target tensors.
        Feasibility masks are padded to global maximum sizes derived from config.max_num_atoms
        (max REAL atoms) + virtual atom.
        Target tensors are created with -1 for inactive levels.
        """
        if not list_of_samples: return {}

        # --- Extract molecules and config ---
        molecules = [sample['molecule'] for sample in list_of_samples]
        first_mol = molecules[0]
        config = first_mol.config  # Get config from the first molecule
        vocab_size = first_mol.vocab_size

        # # --- Get Global Max Sizes and Padding Indices ---
        # if not hasattr(config, 'max_num_atoms') or config.max_num_atoms is None:
        #     raise ValueError("MoleculeConfig must provide 'max_num_atoms' (max REAL atoms) for list_to_batch padding.")

        # config.max_num_atoms = max number of REAL atoms allowed
        max_real_atoms_allowed = config.max_num_atoms

        # CORRECT Global Size for L0 indices: 0 (virtual/terminate) to max_real_atoms_allowed
        global_total_indices_l0 = max_real_atoms_allowed + 1

        # CORRECT Global Size for L1 actions: Select Existing (max_real) + Add (V) + Replace (V) + Remove (1)
        global_max_l1_actions = max_real_atoms_allowed + vocab_size + vocab_size + 1

        # Padding indices calculation (remains the same)
        atoms_padding_idx = vocab_size + 1
        max_valence = max([0] + [v for v in first_mol.vocabulary_valence if v is not None and v >= 0])
        degree_padding_idx = max_valence + 1
        bond_padding_idx = MoleculeDesign.virtual_bond_idx + 1

        device = torch.device("cpu") if device is None else device
        num_atoms_per_mol = [len(mol.atoms) for mol in molecules]  # List of actual atom counts per mol (incl. virtual)
        batch_max_atoms = max(num_atoms_per_mol) if num_atoms_per_mol else 0  # Max atoms *in this specific batch*

        # --- Multi-Hot Encoding for Picked Atoms (Padded to batch_max_atoms) ---
        # (Logic remains the same, using batch_max_atoms for padding this specific input)
        batch_picked_atom_mhe = np.zeros((len(molecules), batch_max_atoms), dtype=int)
        for i, mol in enumerate(molecules):
            anchor_idx = mol.l0_selected_atom_idx
            if mol.current_action_level >= 1 and anchor_idx is not None:
                if 0 <= anchor_idx < batch_max_atoms:
                    batch_picked_atom_mhe[i, anchor_idx] = 1
                else:
                    raise IndexError(f"Anchor index {anchor_idx} out of bounds for mhe (batch max={batch_max_atoms})")

                if mol.current_action_level == 2:
                    target_idx = None
                    if mol.l1_action_type == ActionType.ADD_ATOM:
                        target_idx = len(mol.atoms) - 1
                    elif mol.l1_action_type == ActionType.SELECT_EXISTING_ATOM:
                        target_idx = mol.l1_selected_existing_atom_idx

                    if target_idx is not None:
                        if 0 <= target_idx < batch_max_atoms:
                            if target_idx != anchor_idx:
                                batch_picked_atom_mhe[i, target_idx] = 2
                            else:
                                raise IndexError(f"Target index {target_idx} is same as anchor index {anchor_idx}")
                        else:
                            raise IndexError(
                                f"Target index {target_idx} out of bounds for mhe (batch max={batch_max_atoms})")

        # --- Batch Atoms, Degrees, Bonds, Attention Mask (Padded to batch_max_atoms) ---
        # (Logic remains the same, using batch_max_atoms for padding these inputs)
        batch_atoms = np.stack([
            np.pad(mol.atoms, (0, batch_max_atoms - n), mode='constant', constant_values=atoms_padding_idx) if n > 0
            else np.full(batch_max_atoms, fill_value=atoms_padding_idx, dtype=np.uint8)
            for mol, n in zip(molecules, num_atoms_per_mol)
        ])

        batch_atoms_degree = []
        for mol, n in zip(molecules, num_atoms_per_mol):
            if n > 1:
                d_real = (mol.bonds[1:n, 1:n] > 0).sum(axis=1)
                d = np.concatenate(([0], d_real))
                p_d = np.pad(d, (0, batch_max_atoms - n), mode='constant', constant_values=degree_padding_idx)
            elif n == 1:
                p_d = np.pad(np.array([0]), (0, batch_max_atoms - 1), mode='constant',
                             constant_values=degree_padding_idx)
            else:
                p_d = np.full(batch_max_atoms, fill_value=degree_padding_idx, dtype=int)
            batch_atoms_degree.append(p_d)
        batch_atoms_degree = np.stack(batch_atoms_degree)

        bonds_list = []
        for mol, n in zip(molecules, num_atoms_per_mol):
            if n > 0:
                p_b = np.pad(mol.bonds, [(0, batch_max_atoms - n), (0, batch_max_atoms - n)], mode="constant",
                             constant_values=bond_padding_idx)
                np.fill_diagonal(p_b, bond_padding_idx)
            else:
                p_b = np.full((batch_max_atoms, batch_max_atoms), fill_value=bond_padding_idx, dtype=int)
            bonds_list.append(p_b)
        batch_bonds = np.stack(bonds_list)

        additive_padding_masks = []
        for mol, n in zip(molecules, num_atoms_per_mol):
            if n > 0:
                m = np.zeros((n, n), dtype=float)
                p_m = np.pad(m, [(0, batch_max_atoms - n), (0, batch_max_atoms - n)], mode="constant",
                             constant_values=-np.inf)
                np.fill_diagonal(p_m, 0.0)
            else:
                p_m = np.full((batch_max_atoms, batch_max_atoms), fill_value=-np.inf, dtype=float)
                np.fill_diagonal(p_m, 0.0)
            additive_padding_masks.append(p_m)
        batch_additive_padding_attn_mask = np.stack(additive_padding_masks)

        # --- Batch Level Index (Unchanged) ---
        batch_level_idx = [mol.current_action_level for mol in molecules]

        # --- *** CORRECTED: Batch Feasibility Masks (Padded to GLOBAL Max Sizes) *** ---
        masks_l0, masks_l1, masks_l2 = [], [], []
        for mol, n in zip(molecules, num_atoms_per_mol):
            num_real = n - 1  # Number of real atoms in *this* molecule

            # --- Level 0 Mask ---
            # Expected size for *this molecule*: Terminate/Virtual (1) + Select Existing (num_real)
            expected_len_l0 = 1 + num_real
            mask_l0 = mol.current_action_mask if mol.current_action_level == 0 and mol.current_action_mask is not None else np.ones(
                expected_len_l0, dtype=bool)
            # Sanity check
            if len(mask_l0) != expected_len_l0:
                # print(f"Warning: L0 mask length mismatch for mol {i}. Expected {expected_len_l0}, got {len(mask_l0)}. Using default mask.") # Debug print
                mask_l0 = np.ones(expected_len_l0, dtype=bool)
            # Pad to CORRECT GLOBAL total size (global_total_indices_l0 = max_real_atoms + 1)
            p_mask_l0 = np.pad(mask_l0, (0, global_total_indices_l0 - expected_len_l0), mode='constant',
                               constant_values=True)  # True = Masked/Infeasible
            masks_l0.append(p_mask_l0)

            # --- Level 1 Mask ---
            # Expected size for *this molecule*: Select Existing (num_real) + Add (V) + Replace (V) + Remove (1)
            expected_len_l1 = num_real + vocab_size + vocab_size + 1
            mask_l1 = mol.current_action_mask if mol.current_action_level == 1 and mol.current_action_mask is not None else np.ones(
                expected_len_l1, dtype=bool)
            # Sanity check
            if len(mask_l1) != expected_len_l1:
                # print(f"Warning: L1 mask length mismatch for mol {i}. Expected {expected_len_l1}, got {len(mask_l1)}. Using default mask.") # Debug print
                mask_l1 = np.ones(expected_len_l1, dtype=bool)
            # Pad to CORRECT GLOBAL max size (global_max_l1_actions = max_real_atoms + 2V + 1)
            p_mask_l1 = np.pad(mask_l1, (0, global_max_l1_actions - expected_len_l1), mode='constant',
                               constant_values=True)  # True = Masked/Infeasible
            masks_l1.append(p_mask_l1)

            # --- Level 2 Mask (Remains fixed size 7) ---
            expected_len_l2 = 7  # Fixed size
            mask_l2 = mol.current_action_mask if mol.current_action_level == 2 and mol.current_action_mask is not None else np.ones(
                expected_len_l2, dtype=bool)
            # Sanity check
            if len(mask_l2) != expected_len_l2:
                # print(f"Warning: L2 mask length mismatch for mol {i}. Expected {expected_len_l2}, got {len(mask_l2)}. Using default mask.") # Debug print
                mask_l2 = np.ones(expected_len_l2, dtype=bool)
            # No padding needed as it's fixed size
            masks_l2.append(mask_l2)

        # Convert lists of masks to tensors
        batch_mask_l0 = torch.from_numpy(np.stack(masks_l0)).bool().to(device)
        batch_mask_l1 = torch.from_numpy(np.stack(masks_l1)).bool().to(device)
        batch_mask_l2 = torch.from_numpy(np.stack(masks_l2)).bool().to(device)

        # --- Batch Target Tensors (Logic remains the same) ---
        targets_l0, targets_l1, targets_l2 = [], [], []
        ignore_index = -1  # Value for inactive levels

        for sample in list_of_samples:
            mol = sample['molecule']
            target_action = sample['target_action']
            current_level = mol.current_action_level

            targets_l0.append(target_action if current_level == 0 else ignore_index)
            targets_l1.append(target_action if current_level == 1 else ignore_index)
            targets_l2.append(target_action if current_level == 2 else ignore_index)

        batch_target_l0 = torch.tensor(targets_l0, dtype=torch.long, device=device)
        batch_target_l1 = torch.tensor(targets_l1, dtype=torch.long, device=device)
        batch_target_l2 = torch.tensor(targets_l2, dtype=torch.long, device=device)

        # --- Construct Final Batch Dictionary ---
        return_dict = dict(
            # Inputs padded to batch_max_atoms (dynamic size for Transformer layers)
            level_idx=torch.tensor(batch_level_idx, dtype=torch.long, device=device),
            picked_atom_mhe=torch.from_numpy(batch_picked_atom_mhe).long().to(device),
            num_atoms=torch.tensor(num_atoms_per_mol, dtype=torch.long, device=device),  # Actual atom counts per sample
            atoms=torch.from_numpy(batch_atoms).long().to(device),
            atoms_degree=torch.from_numpy(batch_atoms_degree).long().to(device),
            bonds=torch.from_numpy(batch_bonds).long().to(device),
            additive_padding_attn_mask=torch.from_numpy(batch_additive_padding_attn_mask).float().to(device),

            # Feasibility masks padded to GLOBAL max sizes (for matching fixed logit shapes)
            feasibility_mask_level_zero=batch_mask_l0,  # Shape: (B, max_real_atoms + 1)
            feasibility_mask_level_one=batch_mask_l1,  # Shape: (B, max_real_atoms + 2V + 1)
            feasibility_mask_level_two=batch_mask_l2,  # Shape: (B, 7)

            # Target tensors with ignore_index for inactive levels
            target_zero=batch_target_l0,  # Shape: (B,)
            target_one=batch_target_l1,  # Shape: (B,)
            target_two=batch_target_l2  # Shape: (B,)
        )

        return return_dict

    @staticmethod
    def batch_to_device(batch: dict, device: torch.device):
        """Moves all tensors in the batch dictionary to the specified device."""
        # (Implementation remains the same)
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    @staticmethod
    def get_single_atom_molecules(config: MoleculeConfig, repeat: int = 1) -> List['MoleculeDesign']:
        """Creates initial single-atom MoleculeDesign instances for all allowed atom types."""
        # (Implementation remains the same)
        allowed_atom_indices = [
            i + 1 for i, name in enumerate(config.atom_vocabulary.keys())
            if config.atom_vocabulary[name].get("allowed", False)
        ]
        # if not allowed_atom_indices:
        #     raise ValueError("No allowed atoms found in vocabulary config to initialize single-atom molecules.")
        # Repeat the list of allowed atom indices if needed
        initial_instances = allowed_atom_indices * repeat
        # Create MoleculeDesign instances
        return MoleculeDesign.init_batch_from_instance_list(config, initial_instances)

    @staticmethod
    def from_smiles(config: MoleculeConfig, smiles: str, **kwargs) -> Tuple['MoleculeDesign', Dict[int, int]]:
        """Creates a MoleculeDesign instance from a SMILES string."""
        if '.' in smiles:
            raise ValueError(f"SMILES input {smiles} indicates a fragmented molecule.")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: raise ValueError(f"Invalid SMILES input: {smiles}")
        try:
            # Preprocess: Sanitize, Kekulize, Canonical Rank + Renumber
            sanitize_status = Chem.SanitizeMol(mol, catchErrors=True)
            if sanitize_status != Chem.SanitizeFlags.SANITIZE_NONE:
                 raise ValueError(f"RDKit Sanitization failed for SMILES {smiles} with status {sanitize_status}")
            Chem.Kekulize(mol, clearAromaticFlags=True) # Ensure Kekule form
            canonical_order = rdmolfiles.CanonicalRankAtoms(mol) # Get canonical atom order
            mol = rdmolops.RenumberAtoms(mol, canonical_order) # Renumber atoms based on canonical rank
        except Exception as e:
            # Catch errors during preprocessing
            raise ValueError(f"Could not preprocess input SMILES {smiles}: {e}") from e

        # Create MoleculeDesign instance from the processed RDKit Mol
        try:
            return MoleculeDesign.from_rdkit_mol(config, mol, smiles=smiles)
        except Exception as e:
            # Catch errors during MoleculeDesign state initialization
            raise RuntimeError(f"Error creating MoleculeDesign state from RDKit Mol for {smiles}: {e}") from e

    @staticmethod
    def from_rdkit_mol(config: MoleculeConfig, rdkit_mol: Chem.Mol, smiles: Optional[str] = None) -> Tuple['MoleculeDesign', Dict[int, int]]:
        """Creates a MoleculeDesign instance from an RDKit Mol object."""
        # (Implementation remains the same)
        # Mapping from RDKit bond types to internal bond orders
        BOND_TYPE_TO_RL_ORDER = {
            Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3,
            Chem.BondType.AROMATIC: -1,  # Should be handled by Kekulize, treat as error if seen
            Chem.BondType.QUADRUPLE: 4, Chem.BondType.QUINTUPLE: 5, Chem.BondType.HEXTUPLE: 6,
        }
        num_heavy_atoms = rdkit_mol.GetNumAtoms()

        # Find the first allowed atom in the vocabulary to use for initializing the instance
        first_allowed_atom_idx = -1
        try:
             for i, name in enumerate(config.atom_vocabulary.keys()):
                  if config.atom_vocabulary[name].get("allowed", False):
                       first_allowed_atom_idx = i + 1 # 1-based index
                       break
             # if first_allowed_atom_idx == -1: raise ValueError("No allowed atom found in config.")
        except Exception as e:
             raise RuntimeError(f"Error finding first allowed atom in config: {e}")

        # Handle empty input molecule
        if num_heavy_atoms == 0:
            print(f"Warning: Input RDKit mol {smiles or ''} is empty. Creating empty design state.")
            instance = MoleculeDesign(config, initial_atom=first_allowed_atom_idx)
            # Reset state to represent empty molecule
            instance.atoms = np.array([0], dtype=np.uint8) # Only virtual atom
            instance.bonds = np.zeros((1, 1), dtype=np.uint8)
            instance.is_original_atom = np.array([False], dtype=bool)
            instance.update_action_mask() # Update mask for empty state
            return instance, {} # Return empty instance and empty map

        # Build reverse atom lookup for mapping RDKit atoms to vocab indices
        try:
            reverse_atom_lookup = build_reverse_atom_lookup(config)
        except Exception as e:
            raise RuntimeError(f"Failed to build reverse atom lookup needed for from_rdkit_mol: {e}") from e

        # --- Convert RDKit Mol to Internal State ---
        internal_atoms_list = [0] # Start with virtual atom
        rdkit_to_internal_map = {} # Map RDKit index (0-based) to internal index (1-based)
        internal_idx_counter = 1

        # 1. Process Atoms
        for atom in rdkit_mol.GetAtoms():
            rdkit_idx = atom.GetIdx()
            atomic_num = atom.GetAtomicNum()
            charge = atom.GetFormalCharge()
            # Map RDKit chiral tag (0, 1=CW/R, 2=CCW/S) to config key (0, 1=@, 2=@@)
            rdkit_chiral = atom.GetChiralTag()
            chiral_key_val = 0
            if rdkit_chiral == Chem.ChiralType.CHI_TETRAHEDRAL_CW: chiral_key_val = 1
            elif rdkit_chiral == Chem.ChiralType.CHI_TETRAHEDRAL_CCW: chiral_key_val = 2

            key = (atomic_num, charge, chiral_key_val)
            vocab_idx = reverse_atom_lookup.get(key)
            # # Fallback: If specific chiral type not found, try non-chiral version
            # if vocab_idx is None and chiral_key_val != 0:
            #     key_no_chiral = (atomic_num, charge, 0)
            #     vocab_idx = reverse_atom_lookup.get(key_no_chiral)
            #     raise ValueError(f"Chiral atom type ({atomic_num}, charge={charge}, chiral={chiral_key_val}) from input SMILES '{smiles or ''}' not found in configured atom vocabulary. Fallback to non-chiral version failed.")

            # if vocab_idx is None:
            #     # Atom type in input molecule not found in vocabulary
            #     raise ValueError(f"Atom type ({atomic_num}, charge={charge}, chiral={chiral_key_val}) from input SMILES '{smiles or ''}' not found in configured atom vocabulary.")

            # Add atom to internal list and update map
            internal_atoms_list.append(vocab_idx)
            rdkit_to_internal_map[rdkit_idx] = internal_idx_counter
            internal_idx_counter += 1

        # 2. Process Bonds
        num_total_atoms = len(internal_atoms_list) # Includes virtual atom
        internal_bonds_matrix = np.zeros((num_total_atoms, num_total_atoms), dtype=np.uint8)
        for bond in rdkit_mol.GetBonds():
            idx1, idx2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()

            rl_order = BOND_TYPE_TO_RL_ORDER.get(bond_type)
            # if rl_order is None: # Unsupported bond type
            #      raise ValueError(f"Unsupported RDKit bond type '{bond_type}' found in input SMILES '{smiles or ''}'. Ensure molecule is Kekulized and only contains supported bond types.")
            # if rl_order == -1: # Aromatic bond type should not be present after Kekulization
            #      raise ValueError(f"Aromatic bond type found in input SMILES '{smiles or ''}' after preprocessing. Kekulization might have failed.")

            # Get corresponding internal indices
            try:
                int_idx1, int_idx2 = rdkit_to_internal_map[idx1], rdkit_to_internal_map[idx2]
            except KeyError:
                # This indicates an issue with the rdkit_to_internal_map construction
                raise RuntimeError(f"Internal error: RDKit index map failed for bond between atoms {idx1} and {idx2}.")

            # Set bond order in the symmetric matrix
            internal_bonds_matrix[int_idx1, int_idx2] = internal_bonds_matrix[int_idx2, int_idx1] = rl_order

        # 3. Add Virtual Bonds
        if num_total_atoms > 1:
            internal_bonds_matrix[0, 1:] = internal_bonds_matrix[1:, 0] = MoleculeDesign.virtual_bond_idx

        # --- Create and Initialize MoleculeDesign Instance ---
        try:
            # Initialize with the first allowed atom (state will be overwritten)
            instance = MoleculeDesign(config, initial_atom=first_allowed_atom_idx)
            # Overwrite state with data from input molecule
            instance.atoms = np.array(internal_atoms_list, dtype=np.uint8)
            instance.bonds = internal_bonds_matrix
            # Mark all real atoms from input as original (Rule 1)
            instance.is_original_atom = np.array([False] + [True] * num_heavy_atoms, dtype=bool)
            # Reset trajectory state variables
            instance.synthesis_done = False
            instance._cached_smiles = None # Clear cache
            instance._cached_rdkit_mol = None
            instance.objective = None
            instance.infeasibility_flag = False
            instance.current_action_level = 0 # Start at Level 0
            instance.history = []
            instance.l0_selected_atom_idx = None
            instance.l1_action_type = None
            instance.l1_new_atom_type = None
            instance.l1_selected_existing_atom_idx = None
            instance.last_bond_action_details = None # Reset Rule 2 tracker

            # Update initial mask based on the loaded state
            instance.update_action_mask()
        except Exception as e:
            # Catch errors during instance creation or state setting
            raise RuntimeError(f"Error creating/setting MoleculeDesign state from RDKit Mol for {smiles or ''}: {e}") from e

        # Return the initialized instance and the RDKit-to-internal index map
        return instance, rdkit_to_internal_map
