import copy
import numpy as np
import torch
from torch import nn
from rdkit import Chem, RDLogger
from rdkit.Chem import rdmolfiles, rdmolops
import networkx as nx

import traceback
from config import MoleculeConfig
from core.abstracts import BaseTrajectory

from typing import List, Tuple, Dict, Optional

# Suppress RDKit warnings if desired, but often useful during debugging
# RDLogger.DisableLog('rdApp.*')


def build_reverse_atom_lookup(config: MoleculeConfig) -> Dict[Tuple[int, int, int], int]:
    """
    Creates a lookup dictionary mapping atom properties back to vocabulary indices.

    Args:
        config: The MoleculeConfig instance containing the atom_vocabulary.

    Returns:
        A dictionary mapping (atomic_number, formal_charge, chiral_tag) -> vocab_idx (1-based).

    Raises:
        ValueError: If vocabulary is empty or essential properties are missing.
        KeyError: If vocabulary names mismatch keys.
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
            # This indicates a mismatch between config.vocabulary_atom_names and config.atom_vocabulary keys
            raise KeyError(f"Atom name '{name}' found in vocab_names but not in atom_vocabulary keys.")

        try:
            atomic_num = atom_config['atomic_number']
            # Use .get() with default 0 for optional properties
            charge = atom_config.get('formal_charge', 0)
            chiral = atom_config.get('chiral_tag', 0)  # 0: unspecified, 1: CW, 2: CCW (RDKit mapping)
        except KeyError as e:
            raise ValueError(f"Missing expected property {e} for atom '{name}' in config.")

        key = (atomic_num, charge, chiral)
        vocab_idx = i + 1  # 1-based index

        # Store the mapping for the specific properties
        if key in lookup:
            # Allow overwriting but maybe log a warning if strict checking is needed later
            # print(f"Warning: Duplicate atom definition found for key {key} ('{name}'). Overwriting.")
            pass
        lookup[key] = vocab_idx

        # Add a fallback mapping for non-chiral lookup if this entry is chiral
        # This allows finding a chiral atom even if the query is non-chiral
        if chiral != 0:
            key_no_chiral = (atomic_num, charge, 0)
            if key_no_chiral not in lookup:
                # Only add if a non-chiral version *doesn't* already exist specifically
                lookup[key_no_chiral] = vocab_idx

    if not lookup:
        raise ValueError("Reverse atom lookup is empty. Check atom_vocabulary in config.")

    return lookup


class MoleculeDesign(BaseTrajectory):
    """
    Environment for molecular design using a simplified hierarchical action space (v2025-04-21).

    State Management: Internal NumPy arrays (self.atoms, self.bonds) are the primary source of truth.
                      NetworkX used for connectivity checks.
                      RDKit Mol object is constructed only during finalize() or to_smiles().

    Action Levels (Revised):
        - Level 0: Terminate (if connected & >0 atoms) or Select Existing Atom.
        - Level 1: Add New Atom, Select Existing Atom for Bond, or Remove Selected Atom (from L0).
        - Level 2: Set Bond Order 1-6 (creates if 0) or Remove Bond.
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
    # Removed REMOVE_ATOM_ACTION_L2_MODIFY

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
        # Removed REMOVE_ATOM_ACTION_L2_MODIFY definition
        self.upper_limit_atoms = self.config.max_num_atoms

        if not (initial_atom in self.vocabulary_atom_idcs and not self.atom_feasibility_mask[initial_atom - 1]):
             raise ValueError(f"Initial atom {initial_atom} must be in vocabulary {self.vocabulary_atom_idcs} and allowed in config.")
        self.initial_atom = initial_atom

        # --- Internal State (Primary) ---
        self.atoms = np.array([0, initial_atom], dtype=np.uint8) # Includes virtual atom 0
        self.bonds = np.zeros((2, 2), dtype=np.uint8)
        self.bonds[0, 1] = self.bonds[1, 0] = self.virtual_bond_idx # Virtual connection

        # --- Trajectory State ---
        self.synthesis_done = False
        self._cached_smiles: Optional[str] = None # Cache SMILES after finalization
        self._cached_rdkit_mol: Optional[Chem.Mol] = None # Cache RDKit mol after finalization
        self.objective: Optional[float] = None
        self.sa_score: float = 0.
        self.infeasibility_flag: bool = False
        self.is_currently_connected: bool = True # Assume initial single atom is connected
        self.num_components: int = 1 # Track number of connected components

        # --- Action Handling State ---
        self.current_action_level = 0
        self.current_action_mask: Optional[np.array] = None
        self.history: List[int] = []
        self.l0_selected_atom_idx: Optional[int] = None # 1-based internal index
        # Removed is_modifying_atom, atom_to_modify
        self.l1_new_atom_type: Optional[int] = None # 1-based vocab index
        self.l1_selected_existing_atom_idx: Optional[int] = None # 1-based internal index

        # Initial mask calculation relies on internal state
        self.update_action_mask()

    def _check_and_update_connectivity(self):
        """Checks connectivity using NetworkX on the internal state and updates self.num_components."""
        num_real_atoms = len(self.atoms) - 1
        if num_real_atoms <= 0:
            self.is_currently_connected = True
            self.num_components = 0 if num_real_atoms == 0 else 1
            return

        G = nx.Graph()
        # Use internal indices (1 to num_real_atoms) as nodes for clarity
        G.add_nodes_from(range(1, num_real_atoms + 1))
        adj_matrix = self.bonds[1:, 1:] # Get the submatrix for real atoms
        rows, cols = np.where(adj_matrix > 0)
        # Adjust indices to match internal node IDs (1-based)
        edges = zip(rows + 1, cols + 1)
        G.add_edges_from(edges)

        try:
            if G.number_of_nodes() > 0:
                 # Use number_connected_components for more info
                 self.num_components = nx.number_connected_components(G)
                 self.is_currently_connected = (self.num_components == 1)
            else: # Should be caught by num_real_atoms <= 0, but defensive check
                 self.num_components = 0
                 self.is_currently_connected = True
        except Exception as e:
            # Raise error here as connectivity check is critical for masking/termination
            raise RuntimeError(f"NetworkX connectivity check failed unexpectedly: {e}")

    def _get_current_valence_usage(self) -> np.array:
        """Calculates the sum of explicit bond orders for each real atom from self.bonds."""
        num_real_atoms = len(self.atoms) - 1
        if num_real_atoms <= 0:
            return np.array([], dtype=int)
        # Slice bonds matrix correctly for real atoms (indices 1 to num_real_atoms)
        current_explicit_usage = np.sum(self.bonds[1 : num_real_atoms + 1, 1 : num_real_atoms + 1], axis=1)
        return current_explicit_usage.astype(int)

    def _get_remaining_valence(self) -> np.array:
        """Calculates remaining valence for each real atom based on self.atoms and self.bonds."""
        num_real_atoms = len(self.atoms) - 1
        if num_real_atoms <= 0:
            return np.array([], dtype=int)
        current_usage = self._get_current_valence_usage()
        try:
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
        """Creates the action mask based on the internal state (self.atoms, self.bonds)."""
        if self.synthesis_done:
            self.current_action_mask = None
            return

        num_real_atoms = len(self.atoms) - 1
        remaining_valence = self._get_remaining_valence()

        if self.current_action_level == 0:
            action_space_size = 1 + num_real_atoms
            mask = np.zeros(action_space_size, dtype=bool)
            # Mask terminate if not connected or no real atoms exist
            if num_real_atoms <= 0 or not self.is_currently_connected:
                mask[0] = True
            # Mask atom selection if no atoms
            if num_real_atoms == 0:
                mask[1:] = True
            self.current_action_mask = mask

        elif self.current_action_level == 1:
            # L1: Add New (V), Select Existing (N), Remove Selected (1)
            action_space_size = self.vocab_size + num_real_atoms + 1
            mask = np.ones(action_space_size, dtype=bool)  # Start with everything masked

            anchor_atom_internal_idx = self.l0_selected_atom_idx
            if anchor_atom_internal_idx is None or anchor_atom_internal_idx <= 0 or anchor_atom_internal_idx > num_real_atoms:
                raise ValueError(f"L1 Mask Error: Invalid anchor atom index: {anchor_atom_internal_idx} (NumReal={num_real_atoms})")

            anchor_atom_0_idx = anchor_atom_internal_idx - 1 # 0-based index for valence array

            # --- Unmask "Add Atom" actions ---
            if self.upper_limit_atoms is None or num_real_atoms < self.upper_limit_atoms:
                for i in range(self.vocab_size): # i is 0-based vocab index
                    action_idx = i
                    atom_type_vocab_idx = i + 1 # 1-based vocab index for valence lookup
                    # Check feasibility mask and if the atom type *can* form bonds
                    if not self.atom_feasibility_mask[i] and self.vocabulary_valence[atom_type_vocab_idx] >= 1:
                        # Only unmask if the anchor atom *also* has valence to form a bond
                        if remaining_valence[anchor_atom_0_idx] > 0:
                            mask[action_idx] = False

            # --- Unmask "Select Existing Atom" actions ---
            for target_0_idx in range(num_real_atoms):
                target_internal_idx = target_0_idx + 1
                action_idx = self.vocab_size + target_0_idx # Action index for selecting this existing atom

                if target_internal_idx == anchor_atom_internal_idx: continue  # Cannot select self

                # Check if indices are valid before accessing arrays
                if target_0_idx >= len(remaining_valence):
                    raise IndexError(f"L1 Mask Error: Target index {target_0_idx} out of bounds for remaining_valence (len {len(remaining_valence)})")

                bond_exists = self.bonds[anchor_atom_internal_idx, target_internal_idx] > 0
                target_has_valence = remaining_valence[target_0_idx] > 0

                # Unmask if: bond exists (for modification/deletion) OR both atoms have valence (for insertion)
                should_unmask = bond_exists or (target_has_valence and remaining_valence[anchor_atom_0_idx] > 0)
                if should_unmask:
                    mask[action_idx] = False

            # --- Unmask "Remove Selected Atom" action ---
            remove_action_idx = self.vocab_size + num_real_atoms
            # Only allow removal if more than one real atom exists
            if num_real_atoms > 1:
                mask[remove_action_idx] = False

            self.current_action_mask = mask

        elif self.current_action_level == 2:
            # L2 is now ONLY the Bond Path
            action_space_size = 7  # 0-5 for bond orders 1-6, 6 for remove
            mask = np.ones(action_space_size, dtype=bool)  # Mask all initially

            atom_A_internal_idx = self.l0_selected_atom_idx
            atom_B_internal_idx = -1

            # Determine atom B index based on L1 action
            if self.l1_new_atom_type is not None:
                # If L1 added a new atom, its index is the last one
                atom_B_internal_idx = len(self.atoms) - 1
            elif self.l1_selected_existing_atom_idx is not None:
                atom_B_internal_idx = self.l1_selected_existing_atom_idx
            else:
                # This should not happen if L1 logic is correct
                raise RuntimeError("L2 Bond Mask Error: L1 context (new/existing atom) missing.")

            # --- Index Validation ---
            # Recalculate num_real_atoms in case L1 added an atom
            num_real_atoms = len(self.atoms) - 1
            if (atom_A_internal_idx is None or atom_A_internal_idx <= 0 or atom_A_internal_idx > num_real_atoms or
                    atom_B_internal_idx <= 0 or atom_B_internal_idx > num_real_atoms):
                raise ValueError(
                    f"L2 Bond Mask Error: Invalid indices A={atom_A_internal_idx}, B={atom_B_internal_idx} (NumReal={num_real_atoms})")

            atom_A_0_idx = atom_A_internal_idx - 1
            atom_B_0_idx = atom_B_internal_idx - 1

            # --- Valence Array Validation ---
            if atom_A_0_idx >= len(remaining_valence) or atom_B_0_idx >= len(remaining_valence):
                raise IndexError(
                    f"L2 Bond Mask Error: Indices {atom_A_0_idx} or {atom_B_0_idx} out of bounds for rem_val (len {len(remaining_valence)}).")

            current_bond_order = self.bonds[atom_A_internal_idx, atom_B_internal_idx]
            valence_A_rem = remaining_valence[atom_A_0_idx]
            valence_B_rem = remaining_valence[atom_B_0_idx]

            # Calculate max possible increase based on remaining valence
            max_increase = min(valence_A_rem, valence_B_rem)
            effective_current_order = int(current_bond_order) if current_bond_order > 0 else 0

            # Calculate the highest bond order allowed (considering current bond and available valence)
            max_allowed_final_order = min(effective_current_order + max_increase, self.maximum_bond_order)

            # Unmask bond orders from 1 up to the maximum allowed
            for order in range(1, self.maximum_bond_order + 1):
                action_idx = order - 1  # Action 0 = Order 1, etc.
                if order <= max_allowed_final_order:
                    mask[action_idx] = False

            # Unmask "Remove Bond" action (index 6) if a bond currently exists
            if current_bond_order > 0:
                mask[6] = False

            self.current_action_mask = mask
        else:
            raise ValueError(f"Invalid current_action_level: {self.current_action_level}")

    def _adjust_indices_after_removal(self, removed_internal_idx: int):
        """Adjusts stored internal indices after an atom removal."""
        # Adjust L0 selection if it pointed to an atom after the removed one
        if self.l0_selected_atom_idx is not None and self.l0_selected_atom_idx > removed_internal_idx:
            self.l0_selected_atom_idx -= 1
        # Adjust L1 selection if it pointed to an atom after the removed one
        # (This case might be less likely to occur depending on action flow, but included for safety)
        if self.l1_selected_existing_atom_idx is not None and self.l1_selected_existing_atom_idx > removed_internal_idx:
            self.l1_selected_existing_atom_idx -= 1
        # Note: l1_new_atom_type refers to vocab index, no adjustment needed.

    def take_action(self, action: int):
        """Execute a given action, updating internal state directly."""
        if self.synthesis_done:
            raise RuntimeError("Cannot take action on terminated design.")

        if self.current_action_mask is None or action < 0 or action >= len(self.current_action_mask) or self.current_action_mask[action]:
            mask_len = "None" if self.current_action_mask is None else len(self.current_action_mask)
            raise ValueError(f"Action {action} masked or invalid for level {self.current_action_level}. MaskLen={mask_len}")

        current_level = self.current_action_level
        next_level = 0  # Default next level
        self.history.append(int(action))
        num_real_atoms_before = len(self.atoms) - 1

        try:
            # --- Apply Action based on Level ---
            if current_level == 0:
                if action == 0:  # Terminate
                    if not self.is_terminable(): # Double-check condition before executing
                         raise RuntimeError("Attempted to take Terminate action when not allowed.")
                    self.synthesis_done = True
                    self.finalize(assert_feasible=False) # Build RDKit mol on termination
                    next_level = -1  # Special level for termination
                else:  # Select Atom (action is 1-based internal index)
                    if not (1 <= action <= num_real_atoms_before):
                         raise ValueError(f"L0 Select Atom: Invalid action index {action} for {num_real_atoms_before} real atoms.")
                    self.l0_selected_atom_idx = action
                    # Reset L1 state variables explicitly when starting L1
                    self.l1_new_atom_type = None
                    self.l1_selected_existing_atom_idx = None
                    next_level = 1

            elif current_level == 1:
                remove_action_idx = self.vocab_size + num_real_atoms_before
                anchor_idx = self.l0_selected_atom_idx # Should have been set by L0

                if anchor_idx is None: # Should not happen if logic flow is correct
                    raise RuntimeError("L1 take_action: l0_selected_atom_idx is None.")

                if action < self.vocab_size:  # Add Atom (action is 0-based vocab index)
                    self.l1_new_atom_type = action + 1 # Store 1-based vocab index
                    # --- State Update ---
                    self.atoms = np.append(self.atoms, self.l1_new_atom_type)
                    new_size = len(self.atoms)
                    new_idx = new_size - 1
                    self.bonds = np.pad(self.bonds, [(0, 1), (0, 1)], 'constant', constant_values=0)
                    self.bonds[0, new_idx] = self.bonds[new_idx, 0] = self.virtual_bond_idx # Connect to virtual node
                    # --- End State Update ---
                    next_level = 2 # Go to L2 Bond Path

                elif action < remove_action_idx:  # Select Existing Atom
                    # action index V corresponds to internal index 1, etc.
                    selected_internal_idx = (action - self.vocab_size) + 1
                    if selected_internal_idx == anchor_idx:
                         raise ValueError("L1 Select Existing: Cannot select the anchor atom.")
                    if not (1 <= selected_internal_idx <= num_real_atoms_before):
                         raise ValueError(f"L1 Select Existing: Invalid target index {selected_internal_idx} derived from action {action}.")
                    self.l1_selected_existing_atom_idx = selected_internal_idx
                    next_level = 2 # Go to L2 Bond Path

                elif action == remove_action_idx:  # Remove Selected Atom (from L0)
                    if num_real_atoms_before <= 1: # Should be caught by mask, but double-check
                         raise RuntimeError("Attempted to remove the last real atom.")
                    removed_idx_for_adjust = anchor_idx
                    # --- State Update ---
                    self.atoms = np.delete(self.atoms, removed_idx_for_adjust)
                    self.bonds = np.delete(np.delete(self.bonds, removed_idx_for_adjust, 0), removed_idx_for_adjust, 1)
                    # --- End State Update ---
                    self._adjust_indices_after_removal(removed_idx_for_adjust)
                    # Reset L0/L1 state after removal
                    self.l0_selected_atom_idx = None
                    self.l1_new_atom_type = None
                    self.l1_selected_existing_atom_idx = None
                    next_level = 0 # Back to L0

                else:
                    raise ValueError(f"Invalid L1 action index: {action}")

            elif current_level == 2:
                # L2 is now ONLY the Bond Path
                idx_A = self.l0_selected_atom_idx
                # Determine B based on L1 action
                idx_B = -1
                if self.l1_new_atom_type is not None:
                    idx_B = len(self.atoms) - 1 # New atom is last one
                elif self.l1_selected_existing_atom_idx is not None:
                    idx_B = self.l1_selected_existing_atom_idx
                else:
                    # Should not happen if L1 logic is correct
                    raise RuntimeError("L2 take_action: L1 context (new/existing atom) missing.")

                # --- Index Validation ---
                current_num_real_atoms = len(self.atoms) - 1
                if (idx_A is None or idx_A <= 0 or idx_A > current_num_real_atoms or
                        idx_B <= 0 or idx_B > current_num_real_atoms):
                     raise ValueError(f"L2 take_action: Invalid indices A={idx_A}, B={idx_B} (NumReal={current_num_real_atoms})")

                # --- Apply Bond Change ---
                if action <= 5:  # Set Order (Action 0 = Order 1, ..., Action 5 = Order 6)
                    order = action + 1
                    # Check if change exceeds valence (should be caught by mask, but good sanity check)
                    # current_order = self.bonds[idx_A, idx_B]
                    # rem_val_A = self._get_remaining_valence()[idx_A-1]
                    # rem_val_B = self._get_remaining_valence()[idx_B-1]
                    # increase = order - (current_order if current_order > 0 else 0)
                    # if increase > rem_val_A or increase > rem_val_B:
                    #      raise RuntimeError(f"L2 Set Order {order} violates valence constraints (A_rem={rem_val_A}, B_rem={rem_val_B}, increase={increase})")
                    self.bonds[idx_A, idx_B] = self.bonds[idx_B, idx_A] = order
                elif action == 6:  # Remove Bond (Set Order 0)
                    # if self.bonds[idx_A, idx_B] == 0: # Check if bond exists (should be caught by mask)
                    #      raise RuntimeError("L2 Remove Bond: Attempted to remove non-existent bond.")
                    self.bonds[idx_A, idx_B] = self.bonds[idx_B, idx_A] = 0
                else:
                    raise ValueError(f"Invalid L2 Bond action index: {action}")
                # --- End Bond Change ---

                # Reset L0/L1 state after completing the bond action
                self.l0_selected_atom_idx = None
                self.l1_new_atom_type = None
                self.l1_selected_existing_atom_idx = None
                next_level = 0  # Back to L0

            # --- Update Mask and Level (if not terminated) ---
            if next_level != -1:
                # Connectivity check *must* happen before mask update
                try:
                    self._check_and_update_connectivity()
                except Exception as e:
                     # If connectivity check fails, something is very wrong
                     self.infeasibility_flag = True
                     self.synthesis_done = True
                     self.current_action_mask = None
                     raise RuntimeError(f"Connectivity check failed after action {action}: {e}") from e

                self.current_action_level = next_level # Set level for the NEXT step
                self.update_action_mask() # Calculate mask for the NEXT step
            else:
                # Termination action was taken
                self.current_action_mask = None # No further actions possible

        except Exception as e:
            # Catch any unexpected errors during action execution
            self.infeasibility_flag = True
            self.synthesis_done = True
            self.current_action_mask = None
            # Re-raise as a RuntimeError to halt execution and provide traceback
            raise RuntimeError(f"Error during take_action(action={action}, L{current_level}): {e}") from e

    def finalize(self, assert_feasible: bool = False):
        """Finalize molecule design: build RDKit mol, sanitize, cache SMILES."""
        if self._cached_smiles is not None or self._cached_rdkit_mol is not None:
             return # Already finalized

        # Ensure connectivity is up-to-date based on the final internal state
        try:
             self._check_and_update_connectivity()
        except Exception as e:
             self.infeasibility_flag = True
             print(f"Warning: Connectivity check failed during finalize: {e}. Marking as infeasible.")

        if assert_feasible:
            try:
                self.assert_feasible()
            except AssertionError as e:
                print(f"Warning: Feasibility assertion failed during finalize: {e}")
                self.infeasibility_flag = True

        # Check connectivity required for valid SMILES (unless empty)
        num_real_atoms = len(self.atoms) - 1
        if num_real_atoms > 0 and not self.is_currently_connected:
            print("Warning: Final molecule is disconnected. SMILES may represent fragments.")
            # Decide if disconnected final state is infeasible
            # self.infeasibility_flag = True

        rdkit_mol = None
        if not self.infeasibility_flag:
            try:
                # Build unsanitized first
                rdkit_mol = self.to_rdkit_mol(sanitize=False)
                if rdkit_mol.GetNumAtoms() == 0 and num_real_atoms > 0:
                    # Handle case where internal state has atoms but RDKit conversion failed
                    print("Warning: RDKit molecule is empty despite internal state having atoms.")
                    self.infeasibility_flag = True
                elif rdkit_mol.GetNumAtoms() > 0:
                    try:
                        # Store unsanitized version first
                        self._cached_rdkit_mol = copy.deepcopy(rdkit_mol)
                        # Attempt sanitization
                        Chem.SanitizeMol(rdkit_mol)
                        # Store sanitized version if successful
                        self._cached_rdkit_mol = rdkit_mol
                        self._cached_smiles = Chem.MolToSmiles(rdkit_mol)
                    except Exception as e:
                        print(f"Warning: Final sanitization/SMILES generation failed: {e}. Using unsanitized state if possible.")
                        # SMILES will remain None, but keep unsanitized RDKit mol if created
                        self._cached_smiles = None
                        if self._cached_rdkit_mol is None: # If even unsanitized failed somehow
                            self.infeasibility_flag = True
                else: # No real atoms internally
                    self._cached_smiles = ""
                    self._cached_rdkit_mol = rdkit_mol # Store empty mol

            except Exception as e:
                 print(f"Warning: Error during RDKit mol generation in finalize: {e}")
                 self.infeasibility_flag = True
                 self._cached_smiles = None
                 self._cached_rdkit_mol = None
        else:
            self._cached_smiles = None
            self._cached_rdkit_mol = None

        # Ensure synthesis_done is True after finalize is called
        self.synthesis_done = True


    def assert_feasible(self):
        """Check internal state consistency (NumPy arrays). Raises AssertionError on failure."""
        if not isinstance(self.atoms, np.ndarray) or not isinstance(self.bonds, np.ndarray):
             raise AssertionError("Internal state (atoms/bonds) is not numpy array.")

        assert self.atoms[0] == 0, "First atom should be virtual (0)"
        num_atoms = len(self.atoms)
        num_real_atoms = num_atoms - 1

        if num_real_atoms > 0:
             valid_indices = all(1 <= idx <= self.vocab_size for idx in self.atoms[1:])
             assert valid_indices, f"Invalid atom vocab index found: {self.atoms[1:]}"
             allowed_check = all(not self.atom_feasibility_mask[idx - 1] for idx in self.atoms[1:])
             assert allowed_check, f"Disallowed atom type found: {self.atoms[1:]}"

        if self.upper_limit_atoms is not None:
             assert num_real_atoms <= self.upper_limit_atoms, f"Max atoms exceeded ({num_real_atoms} > {self.upper_limit_atoms})"

        assert self.bonds.shape == (num_atoms, num_atoms), f"Bonds shape mismatch: {self.bonds.shape} vs ({num_atoms},{num_atoms})"
        assert not np.any(self.bonds.diagonal()), "Self-loops detected in bond matrix"
        assert np.all(self.bonds == self.bonds.T), "Bond matrix not symmetric"

        if num_real_atoms > 0:
             # Check virtual bonds
             assert np.all(self.bonds[0, 1:] == self.virtual_bond_idx), "Virtual bond missing from row 0"
             assert np.all(self.bonds[1:, 0] == self.virtual_bond_idx), "Virtual bond missing from col 0"
             # Check valence
             try:
                  remaining_valence = self._get_remaining_valence()
                  assert np.all(remaining_valence >= 0), f"Valence constraints violated: {remaining_valence}"
             except IndexError as e:
                  raise AssertionError(f"Index error during valence check in assert_feasible: {e}")
             except RuntimeError as e:
                  raise AssertionError(f"Runtime error during valence check in assert_feasible: {e}")


    def to_rdkit_mol(self, sanitize=True) -> Chem.RWMol:
        """Creates a *new* RDKit molecule from the internal state (atoms, bonds)."""
        mol = Chem.RWMol()
        num_total_atoms = len(self.atoms)
        if num_total_atoms <= 1: return mol # Return empty mol if only virtual atom

        rdkit_idx_map = {} # Map internal index (1-based) -> new RDKit index (0-based)
        for internal_idx, atom_vocab_idx in enumerate(self.atoms):
            if internal_idx == 0: continue # Skip virtual atom at index 0

            if not (1 <= atom_vocab_idx <= self.vocab_size):
                 # This indicates corruption in self.atoms
                 raise ValueError(f"Invalid vocab index {atom_vocab_idx} at internal index {internal_idx} during to_rdkit_mol.")

            try:
                atom_config = self.atom_vocabulary[self.vocabulary_atom_names[atom_vocab_idx - 1]]
            except IndexError:
                raise IndexError(f"Cannot find atom name for vocab index {atom_vocab_idx} (0-based index {atom_vocab_idx-1}).")
            except KeyError:
                # Should be caught by IndexError above if names/vocab match
                raise KeyError(f"Cannot find config for atom name corresponding to vocab index {atom_vocab_idx}.")

            a = Chem.Atom(atom_config["atomic_number"])
            if "formal_charge" in atom_config: a.SetFormalCharge(atom_config["formal_charge"])

            # Handle chirality mapping
            ct = atom_config.get("chiral_tag", 0)
            if ct == 1: a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
            elif ct == 2: a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
            else: a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED) # Default

            new_rdkit_idx = mol.AddAtom(a)
            rdkit_idx_map[internal_idx] = new_rdkit_idx # Store mapping: internal (1-based) -> rdkit (0-based)

        # Add bonds between real atoms
        for i in range(1, num_total_atoms): # internal indices i, j
            for j in range(i + 1, num_total_atoms):
                bond_order = self.bonds[i, j]
                if bond_order > 0 and bond_order <= self.maximum_bond_order:
                    # Ensure indices exist in map (they should if atoms were added correctly)
                    if i not in rdkit_idx_map or j not in rdkit_idx_map:
                         # This indicates an internal inconsistency
                         raise RuntimeError(f"Missing RDKit index map entry for internal indices {i} or {j} in to_rdkit_mol.")

                    rdkit_i, rdkit_j = rdkit_idx_map[i], rdkit_idx_map[j]
                    rdkit_bond_type = self.bond_types.get(int(bond_order))
                    if rdkit_bond_type:
                        mol.AddBond(rdkit_i, rdkit_j, rdkit_bond_type)
                    else:
                        # Should not happen if bond_order <= maximum_bond_order
                        print(f"Warning: Could not find RDKit bond type for order {bond_order}.")
                elif bond_order > self.maximum_bond_order and bond_order != self.virtual_bond_idx:
                     # Invalid bond order in the matrix
                     print(f"Warning: Invalid bond order {bond_order} found between internal atoms {i},{j} during to_rdkit_mol.")

        if sanitize:
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                # Don't raise error here, allow returning unsanitized mol, but log clearly
                print(f"Warning: RDKit sanitization failed in to_rdkit_mol: {e}")
                # Optionally, return None or raise if sanitization is mandatory
                # raise ValueError(f"Sanitization failed: {e}") from e
        return mol

    def is_terminable(self):
        """Checks if the current state allows termination."""
        num_real_atoms = len(self.atoms) - 1
        # Allowed if L0, not done, connected, and has at least one real atom
        can_terminate = self.current_action_level == 0 and not self.synthesis_done
        connectivity_ok = (num_real_atoms > 0 and self.is_currently_connected)
        return can_terminate and connectivity_ok

    def to_smiles(self, canonical: bool = True) -> Optional[str]:
        """
        Returns a canonical SMILES string representation of the molecule.
        Finalizes the molecule if not already done. Caches the result.
        """
        if not self.synthesis_done:
            self.finalize(assert_feasible=False) # Finalize if needed

        if self._cached_smiles is not None and canonical:
            return self._cached_smiles # Return cached canonical SMILES

        # If only non-canonical is cached or need re-generation
        if self._cached_rdkit_mol is not None:
            try:
                # Ensure sanitization before generating canonical SMILES
                mol_to_use = copy.deepcopy(self._cached_rdkit_mol) # Work on copy
                Chem.SanitizeMol(mol_to_use) # Re-sanitize just in case
                smiles = Chem.MolToSmiles(mol_to_use, canonical=canonical)
                if canonical: self._cached_smiles = smiles # Update cache if canonical requested
                return smiles
            except Exception as e:
                print(f"Warning: Failed to generate SMILES (canonical={canonical}) from cached RDKit mol: {e}")
                return None
        else: # No RDKit mol could be generated
            return self._cached_smiles # Return None or "" depending on finalize outcome

    # --- Batching and Static Methods (Mostly Unchanged, check padding sizes) ---

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

                logits = None
                if mol.current_action_level == 0: logits = batch_logits_l0[i]
                elif mol.current_action_level == 1: logits = batch_logits_l1[i]
                elif mol.current_action_level == 2: logits = batch_logits_l2[i]
                else: log_probs_to_return.append(np.array([])); continue # Should not happen

                mask_len = len(mask)
                if len(logits) > mask_len: logits = logits[:mask_len]
                elif len(logits) < mask_len:
                     # This indicates a mismatch between network output size and expected action space size
                     raise ValueError(f"Logits/Mask length mismatch L{mol.current_action_level}: Logits {len(logits)}, Mask {mask_len}")

                # Apply mask and calculate log probabilities safely
                logits[mask] = -np.inf
                max_logit = np.max(logits)
                if np.isneginf(max_logit): # All actions masked or logits are -inf
                    log_probs = logits # Keep as -inf
                else:
                     exp_logits = np.exp(logits - max_logit)
                     log_sum_exp = np.log(np.sum(exp_logits))
                     log_probs = logits - (max_logit + log_sum_exp)
                     log_probs[mask] = -np.inf # Ensure masked entries remain -inf

                log_probs_to_return.append(log_probs)
        return log_probs_to_return

    def transition_fn(self, action: int) -> Tuple['BaseTrajectory', bool]:
        """Creates a copy, takes the action, and returns the new state and termination status."""
        copied_molecule = copy.deepcopy(self)
        try:
            copied_molecule.take_action(action)
        except (ValueError, RuntimeError, IndexError) as e:
            # If take_action fails internally, mark as infeasible and done
            copied_molecule.infeasibility_flag = True
            copied_molecule.synthesis_done = True
            copied_molecule.current_action_mask = None
            print(f"Warning: transition_fn caught error in take_action({action}): {e}. Returning infeasible state.")
            # Optionally re-raise if halting is preferred: raise e
        return copied_molecule, copied_molecule.synthesis_done

    def to_max_evaluation_fn(self) -> float:
        """Returns the objective value, penalizing infeasible states."""
        if self.objective is None:
             # print("Warning: Objective is None.") # Reduce noise
             return float("-inf")
        # Return -inf if the state is marked as infeasible
        return float("-inf") if self.infeasibility_flag else self.objective

    def num_actions(self) -> int:
        """Returns the number of valid (unmasked) actions at the current level."""
        if self.current_action_mask is None: return 0
        # Count False values in the boolean mask
        return int(np.sum(~self.current_action_mask))

    @staticmethod
    def list_to_batch(molecules: List['MoleculeDesign'], device: torch.device = None,
                      include_feasibility_masks: bool = False) -> dict:
        """Converts a list of MoleculeDesign instances to a batch dictionary."""
        if not molecules: return {} # Handle empty list

        # Use attributes from the first molecule assuming homogeneity
        first_mol = molecules[0]
        atoms_padding_idx = first_mol.vocab_size + 1
        max_valence = max([-1] + [v for v in first_mol.vocabulary_valence if v is not None and v >= 0])
        degree_padding_idx = max_valence + 2
        bond_padding_idx = MoleculeDesign.virtual_bond_idx + 1

        device = torch.device("cpu") if device is None else device
        num_atoms = [len(mol.atoms) for mol in molecules]
        max_num_atoms = max(num_atoms) if num_atoms else 0
        batch_level_idx = [mol.current_action_level for mol in molecules]

        # --- Batch Picked Atom (L0 Selection) ---
        batch_picked_atom_mhe = np.zeros((len(molecules), max_num_atoms), dtype=int)
        for i, mol in enumerate(molecules):
            # Picked atom is relevant for L1 and L2 (Bond Path) mask generation/interpretation
            if mol.current_action_level >= 1 and mol.l0_selected_atom_idx is not None:
                 # Check index validity before assigning
                 if 0 <= mol.l0_selected_atom_idx < max_num_atoms:
                      batch_picked_atom_mhe[i, mol.l0_selected_atom_idx] = 1
                 # else: # Should not happen if l0_selected_atom_idx is valid internal index
                 #     print(f"Warning: Invalid l0_selected_atom_idx {mol.l0_selected_atom_idx} during batching.")

        # --- Batch Atoms ---
        batch_atoms = np.stack([
            np.pad(mol.atoms, (0, max_num_atoms - num_atoms[i]), mode='constant', constant_values=atoms_padding_idx)
            if num_atoms[i] > 0 else np.full(max_num_atoms, fill_value=atoms_padding_idx, dtype=np.uint8)
            for i, mol in enumerate(molecules)
        ])

        # --- Batch Atom Degrees ---
        batch_atoms_degree = []
        for i, mol in enumerate(molecules):
             current_num_atoms = num_atoms[i]
             if current_num_atoms > 1:
                  # Slice bonds correctly for real atoms (1 to current_num_atoms-1)
                  real_bonds = mol.bonds[1:current_num_atoms, 1:current_num_atoms]
                  degree_real = (real_bonds > 0).sum(axis=1)
                  # Prepend 0 for virtual atom degree
                  degree = np.concatenate(([0], degree_real))
                  padded_degree = np.pad(degree, (0, max_num_atoms - current_num_atoms), mode='constant', constant_values=degree_padding_idx)
             elif current_num_atoms == 1: # Only virtual atom
                  padded_degree = np.pad(np.array([0]), (0, max_num_atoms - 1), mode='constant', constant_values=degree_padding_idx)
             else: # Empty molecule state
                  padded_degree = np.full(max_num_atoms, fill_value=degree_padding_idx, dtype=int)
             batch_atoms_degree.append(padded_degree)
        batch_atoms_degree = np.stack(batch_atoms_degree)

        # --- Batch Bonds ---
        bonds_list = []
        for i, mol in enumerate(molecules):
            current_num_atoms = num_atoms[i]
            if current_num_atoms > 0:
                 padded_bonds = np.pad(mol.bonds, [(0, max_num_atoms - current_num_atoms), (0, max_num_atoms - current_num_atoms)], mode="constant", constant_values=bond_padding_idx)
                 # Ensure diagonal is padding index
                 np.fill_diagonal(padded_bonds, bond_padding_idx)
            else:
                 padded_bonds = np.full((max_num_atoms, max_num_atoms), fill_value=bond_padding_idx, dtype=int)
            bonds_list.append(padded_bonds)
        batch_bonds = np.stack(bonds_list)

        # --- Batch Attention Mask ---
        additive_padding_masks = []
        for i, mol in enumerate(molecules):
             current_num_atoms = num_atoms[i]
             if current_num_atoms > 0:
                  # Mask has shape (current_num_atoms, current_num_atoms), values are 0.0
                  mask = np.zeros((current_num_atoms, current_num_atoms), dtype=float)
                  # Pad to (max_num_atoms, max_num_atoms) with -inf
                  padded_mask = np.pad(mask, [(0, max_num_atoms - current_num_atoms), (0, max_num_atoms - current_num_atoms)], mode="constant", constant_values=-np.inf)
                  # Diagonal should be 0.0 for self-attention
                  np.fill_diagonal(padded_mask, 0.0)
             else: # Empty molecule
                  padded_mask = np.full((max_num_atoms, max_num_atoms), fill_value=-np.inf, dtype=float)
                  np.fill_diagonal(padded_mask, 0.0) # Still allow self-attention if max_num_atoms > 0
             additive_padding_masks.append(padded_mask)
        batch_additive_padding_attn_mask = np.stack(additive_padding_masks)

        # --- Assemble Dictionary ---
        return_dict = dict(
            level_idx=torch.tensor(batch_level_idx, dtype=torch.long, device=device),
            picked_atom_mhe=torch.from_numpy(batch_picked_atom_mhe).long().to(device),
            num_atoms=torch.tensor(num_atoms, dtype=torch.long, device=device),
            atoms=torch.from_numpy(batch_atoms).long().to(device),
            atoms_degree=torch.from_numpy(batch_atoms_degree).long().to(device),
            bonds=torch.from_numpy(batch_bonds).long().to(device),
            additive_padding_attn_mask=torch.from_numpy(batch_additive_padding_attn_mask).float().to(device),
        )

        # --- Include Feasibility Masks (Action Masks) if requested ---
        if include_feasibility_masks:
            masks_l0, masks_l1, masks_l2 = [], [], []
            # Determine max action space size for each level across the batch
            max_actions_l0, max_actions_l1, max_actions_l2 = 0, 0, 7 # L2 is fixed size
            for mol in molecules:
                 num_real = len(mol.atoms) - 1
                 max_actions_l0 = max(max_actions_l0, 1 + num_real)
                 # L1 size: V + N + 1 (Add, Select Existing, Remove)
                 max_actions_l1 = max(max_actions_l1, mol.vocab_size + num_real + 1)
                 # L2 size is fixed at 7 (Set Order 1-6, Remove Bond)

            for mol in molecules:
                 num_real = len(mol.atoms) - 1
                 # L0 Mask
                 current_mask_l0 = mol.current_action_mask if mol.current_action_level == 0 and mol.current_action_mask is not None else np.ones(1 + num_real, dtype=bool)
                 expected_len_l0 = 1 + num_real
                 if len(current_mask_l0) != expected_len_l0: current_mask_l0 = np.ones(expected_len_l0, dtype=bool) # Fallback if length mismatch
                 padded_mask_l0 = np.pad(current_mask_l0, (0, max_actions_l0 - expected_len_l0), mode='constant', constant_values=True)
                 masks_l0.append(padded_mask_l0)

                 # L1 Mask
                 expected_len_l1 = mol.vocab_size + num_real + 1
                 current_mask_l1 = mol.current_action_mask if mol.current_action_level == 1 and mol.current_action_mask is not None else np.ones(expected_len_l1, dtype=bool)
                 if len(current_mask_l1) != expected_len_l1: current_mask_l1 = np.ones(expected_len_l1, dtype=bool) # Fallback
                 padded_mask_l1 = np.pad(current_mask_l1, (0, max_actions_l1 - expected_len_l1), mode='constant', constant_values=True)
                 masks_l1.append(padded_mask_l1)

                 # L2 Mask (Fixed size 7)
                 expected_len_l2 = 7
                 current_mask_l2 = mol.current_action_mask if mol.current_action_level == 2 and mol.current_action_mask is not None else np.ones(expected_len_l2, dtype=bool)
                 if len(current_mask_l2) != expected_len_l2: current_mask_l2 = np.ones(expected_len_l2, dtype=bool) # Fallback
                 # No padding needed if max_actions_l2 is correctly fixed at 7
                 masks_l2.append(current_mask_l2)

            return_dict["feasibility_mask_level_zero"] = torch.from_numpy(np.stack(masks_l0)).bool().to(device)
            return_dict["feasibility_mask_level_one"] = torch.from_numpy(np.stack(masks_l1)).bool().to(device)
            return_dict["feasibility_mask_level_two"] = torch.from_numpy(np.stack(masks_l2)).bool().to(device)

        return return_dict

    @staticmethod
    def batch_to_device(batch: dict, device: torch.device):
        """Moves all tensors in a batch dictionary to the specified device."""
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    @staticmethod
    def get_single_atom_molecules(config: MoleculeConfig, repeat: int = 1) -> List['MoleculeDesign']:
        """Creates initial molecules with single allowed atoms."""
        atoms = []
        for i, atom_name in enumerate(config.atom_vocabulary.keys()):
            if config.atom_vocabulary[atom_name]["allowed"]:
                atoms.append(i + 1) # 1-based vocab index
        if not atoms:
            raise ValueError("No allowed atoms found in vocabulary config to create initial molecules.")
        return MoleculeDesign.init_batch_from_instance_list(config, atoms * repeat)

    # --- from_smiles / from_rdkit_mol (Error Handling Refined) ---

    @staticmethod
    def from_smiles(config: MoleculeConfig, smiles: str, **kwargs) -> Tuple[
        Optional['MoleculeDesign'], Optional[Dict[int, int]]]:
        """
        Creates a MoleculeDesign instance directly from a SMILES string.
        Handles canonicalization and renumbering before calling from_rdkit_mol.

        Returns the instance and a map from original canonical RDKit indices to internal indices,
        or raises an error on failure.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES input: {smiles}")

        # --- Canonical Renumbering (Crucial for consistency) ---
        try:
            # Remove Hs early, before canonicalization/kekulization
            mol = Chem.RemoveHs(mol, sanitize=False)
            # Sanitize first to handle aromaticity etc.
            Chem.SanitizeMol(mol, catchErrors=True)
            # Kekulize BEFORE canonical ranking for consistency
            Chem.Kekulize(mol, clearAromaticFlags=True)
            # Renumber atoms based on canonical rank
            canonical_order = rdmolfiles.CanonicalRankAtoms(mol)
            mol = rdmolops.RenumberAtoms(mol, canonical_order)
        except Exception as e:
            raise ValueError(f"Could not sanitize/kekulize/canonically renumber input SMILES {smiles}: {e}") from e

        # Call the simplified from_rdkit_mol
        try:
            # Pass the preprocessed mol
            design_instance, rdkit_map = MoleculeDesign.from_rdkit_mol(
                config, mol, smiles=smiles # Pass smiles for context in errors
            )
            # from_rdkit_mol now raises errors on failure
            return design_instance, rdkit_map
        except Exception as e:
            # Catch potential errors from from_rdkit_mol and re-raise
            raise RuntimeError(f"Error during from_rdkit_mol execution for {smiles}: {e}") from e


    @staticmethod
    def from_rdkit_mol(config: MoleculeConfig, rdkit_mol: Chem.Mol, smiles: Optional[str] = None) -> Tuple[
        'MoleculeDesign', Dict[int, int]]:
        """
        Creates a MoleculeDesign instance directly from an RDKit molecule
        by constructing the internal state (atoms, bonds) without simulating actions.

        Assumes input rdkit_mol has been appropriately preprocessed
        (e.g., Hs removed, Kekulized, Canonically Renumbered) before calling this method.

        Returns the instance and a map from the RDKit indices of the input molecule
        to the internal indices of the created instance. Raises Error on failure.
        """
        BOND_TYPE_TO_RL_ORDER = {
            Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3,
            Chem.BondType.QUADRUPLE: 4, Chem.BondType.QUINTUPLE: 5, Chem.BondType.HEXTUPLE: 6,
            # Aromatic bonds should have been Kekulized before calling this
        }

        num_heavy_atoms = rdkit_mol.GetNumAtoms()
        if num_heavy_atoms == 0:
            # Handle empty molecule case consistently
            print(f"Warning: Input molecule {smiles or ''} has no heavy atoms. Creating empty design.")
            first_allowed_atom_idx = 1 # Default needed for init
            try:
                for i, name in enumerate(config.atom_vocabulary.keys()):
                    if config.atom_vocabulary[name]["allowed"]: first_allowed_atom_idx = i + 1; break
            except Exception: pass # Ignore if finding first allowed fails here
            instance = MoleculeDesign(config, initial_atom=first_allowed_atom_idx)
            instance.atoms = np.array([0], dtype=np.uint8)
            instance.bonds = np.zeros((1, 1), dtype=np.uint8)
            instance._check_and_update_connectivity()
            instance.update_action_mask()
            return instance, {}

        # 2. Build Atom List and Index Map
        try:
            reverse_atom_lookup = build_reverse_atom_lookup(config)
        except (NameError, ValueError, KeyError) as e:
            raise RuntimeError("Failed to build reverse atom lookup.") from e

        internal_atoms_list = [0]  # Start with virtual atom
        rdkit_to_internal_map = {}
        internal_idx_counter = 1

        for atom in rdkit_mol.GetAtoms():
            rdkit_idx = atom.GetIdx()
            atomic_num = atom.GetAtomicNum()
            charge = atom.GetFormalCharge()
            chiral = int(atom.GetChiralTag())

            key = (atomic_num, charge, chiral)
            vocab_idx = reverse_atom_lookup.get(key)
            if vocab_idx is None and chiral != 0:
                key_no_chiral = (atomic_num, charge, 0)
                vocab_idx = reverse_atom_lookup.get(key_no_chiral)

            if vocab_idx is None:
                raise ValueError(f"Atom type (Num={atomic_num}, Charge={charge}, Chiral={chiral}) "
                                 f"in molecule {smiles or ''} not found in vocabulary config.")

            internal_atoms_list.append(vocab_idx)
            rdkit_to_internal_map[rdkit_idx] = internal_idx_counter
            internal_idx_counter += 1

        # 3. Build Bond Matrix
        num_total_atoms = len(internal_atoms_list)
        internal_bonds_matrix = np.zeros((num_total_atoms, num_total_atoms), dtype=np.uint8)

        for bond in rdkit_mol.GetBonds():
            rdkit_idx1 = bond.GetBeginAtomIdx()
            rdkit_idx2 = bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()

            rl_order = BOND_TYPE_TO_RL_ORDER.get(bond_type)
            if rl_order is None:
                raise ValueError(f"Unsupported bond type {bond_type} found in {smiles or ''} after preprocessing. Ensure Kekulization.")

            try:
                internal_idx1 = rdkit_to_internal_map[rdkit_idx1]
                internal_idx2 = rdkit_to_internal_map[rdkit_idx2]
            except KeyError:
                raise RuntimeError(f"RDKit index mapping failed for bond ({rdkit_idx1}, {rdkit_idx2}).")

            internal_bonds_matrix[internal_idx1, internal_idx2] = rl_order
            internal_bonds_matrix[internal_idx2, internal_idx1] = rl_order

        # 4. Add Virtual Bonds
        if num_total_atoms > 1:
            internal_bonds_matrix[0, 1:] = MoleculeDesign.virtual_bond_idx
            internal_bonds_matrix[1:, 0] = MoleculeDesign.virtual_bond_idx

        # 5. Create Instance and Set State
        try:
            first_allowed_atom_idx = 1 # Default needed for init
            try:
                for i, name in enumerate(config.atom_vocabulary.keys()):
                    if config.atom_vocabulary[name]["allowed"]: first_allowed_atom_idx = i + 1; break
            except Exception: pass # Ignore if finding first allowed fails here
            instance = MoleculeDesign(config, initial_atom=first_allowed_atom_idx)

            instance.atoms = np.array(internal_atoms_list, dtype=np.uint8)
            instance.bonds = internal_bonds_matrix
            instance.synthesis_done = False
            instance._cached_smiles = None
            instance._cached_rdkit_mol = None
            instance.objective = None
            instance.infeasibility_flag = False
            instance.current_action_level = 0
            instance.history = []
            instance.l0_selected_atom_idx = None
            instance.l1_new_atom_type = None
            instance.l1_selected_existing_atom_idx = None

            instance._check_and_update_connectivity()
            instance.update_action_mask()
        except Exception as e:
            raise RuntimeError(f"Error creating/setting state for MoleculeDesign instance for {smiles or ''}: {e}") from e

        return instance, rdkit_to_internal_map
