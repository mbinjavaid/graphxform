import copy
import numpy as np
import torch
from torch import nn
from rdkit import Chem, RDLogger
from rdkit.Chem import rdmolfiles, rdmolops
import networkx as nx

# import traceback
from config import MoleculeConfig
from core.abstracts import BaseTrajectory

from typing import List, Tuple, Dict, Optional

# Suppress RDKit warnings
# RDLogger.DisableLog('rdApp.*')


class ActionType:
    ADD_ATOM = 1
    SELECT_EXISTING_ATOM = 2
    REMOVE_SELECTED_ATOM = 3


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
    connectivity filtering (post-hoc), and rules to guide generation.

    Rule 1: Only atoms present in the initial molecule can be removed.
    Rule 2: Immediate reversal of bond actions on the same atom pair is forbidden.

    State Management: Internal NumPy arrays (self.atoms, self.bonds) are the primary source of truth.
                      NetworkX used for connectivity checks.
                      RDKit Mol object is constructed only during finalize() or to_smiles().

    Action Levels (Revised):
        - Level 0: Terminate (if connected & >0 atoms) or Select Existing Atom.
        - Level 1: Add New Atom, Select Existing Atom for Bond, or Remove Selected Original Atom (from L0).
        - Level 2: Set Bond Order 1-6 (creates if 0) or Remove Bond.
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
        self.is_currently_connected: bool = True
        self.num_components: int = 1

        # --- Action Handling State ---
        self.current_action_level = 0
        self.current_action_mask: Optional[np.array] = None
        self.history: List[int] = []
        self.l0_selected_atom_idx: Optional[int] = None # 1-based internal index
        self.l1_new_atom_type: Optional[int] = None # 1-based vocab index
        self.l1_selected_existing_atom_idx: Optional[int] = None # 1-based internal index
        # --- Rule 2 State ---
        self.last_bond_action_details: Optional[Tuple[int, int]] = None # Stores (min_idx, max_idx) of last bond action pair

        self.update_action_mask()

    def _get_smiles_for_check(self) -> Optional[str]:
        """
        Generates a canonical SMILES string for intermediate checks WITHOUT
        calling finalize() or modifying internal state caches/flags.
        Returns None if SMILES generation or sanitization fails.
        """
        try:
            # Create a temporary RDKit mol from current state
            temp_mol = self.to_rdkit_mol(sanitize=False)  # Get unsanitized first
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

    def _check_and_update_connectivity(self):
        """Checks connectivity using NetworkX on the internal state and updates self.num_components."""
        num_real_atoms = len(self.atoms) - 1
        # # --- DEBUG PRINTS ---
        # print(f"\nDEBUG Connectivity Check: Start")
        # print(f"DEBUG Connectivity Check: num_real_atoms={num_real_atoms}")
        # print(f"DEBUG Connectivity Check: self.atoms={self.atoms}")
        # print(f"DEBUG Connectivity Check: self.bonds=\n{self.bonds}")
        # # --- END DEBUG PRINTS ---

        if num_real_atoms <= 0:
            self.is_currently_connected = True
            self.num_components = 0 if num_real_atoms == 0 else 1
            print(f"DEBUG Connectivity Check: <=0 real atoms. num_components={self.num_components}")  # DEBUG
            return

        G = nx.Graph()
        # Add nodes using 1-based internal indices
        G.add_nodes_from(range(1, num_real_atoms + 1))
        # Extract adjacency matrix for real atoms only
        adj_matrix = self.bonds[1: num_real_atoms + 1, 1: num_real_atoms + 1]
        # # --- DEBUG PRINTS ---
        # print(f"DEBUG Connectivity Check: adj_matrix (real atoms only)=\n{adj_matrix}")
        # # --- END DEBUG PRINTS ---
        rows, cols = np.where(adj_matrix > 0)
        # Edges need to use the 1-based node indices
        edges = list(zip(rows + 1, cols + 1))  # Convert to list for printing
        # print(f"DEBUG Connectivity Check: Edges derived for graph: {edges}")  # DEBUG
        G.add_edges_from(edges)

        try:
            # --- DEBUG PRINTS ---
            # print(f"DEBUG Connectivity Check: Graph nodes={list(G.nodes())}, edges={list(G.edges())}")
            # --- END DEBUG PRINTS ---
            if G.number_of_nodes() > 0:
                # Use nx.is_connected and nx.number_connected_components
                # is_connected is faster if you only need the boolean check first
                self.is_currently_connected = nx.is_connected(G)
                if self.is_currently_connected:
                    self.num_components = 1
                    # print(f"DEBUG Connectivity Check: nx.is_connected=True. num_components=1")  # DEBUG
                else:
                    # Only calculate components if not connected
                    self.num_components = nx.number_connected_components(G)
                    # print(
                    #     f"DEBUG Connectivity Check: nx.is_connected=False. num_components={self.num_components}")  # DEBUG

            else:  # Should not happen if num_real_atoms > 0, but defensive
                self.num_components = 0
                self.is_currently_connected = True
                # print(f"DEBUG Connectivity Check: G has 0 nodes. num_components=0")  # DEBUG
        except Exception as e:
            # If NetworkX fails, something is fundamentally wrong with the graph construction
            print(f"ERROR during NetworkX check. Graph nodes={list(G.nodes())}, edges={list(G.edges())}")  # DEBUG
            raise RuntimeError(f"NetworkX connectivity check failed unexpectedly: {e}")

        # print(
        #     f"DEBUG Connectivity Check: End. Final num_components={self.num_components}, is_connected={self.is_currently_connected}\n")  # DEBUG

    def _get_current_valence_usage(self) -> np.array:
        """Calculates the sum of explicit bond orders for each real atom from self.bonds."""
        num_real_atoms = len(self.atoms) - 1
        if num_real_atoms <= 0: return np.array([], dtype=int)
        current_explicit_usage = np.sum(self.bonds[1 : num_real_atoms + 1, 1 : num_real_atoms + 1], axis=1)
        return current_explicit_usage.astype(int)

    def _get_remaining_valence(self) -> np.array:
        """Calculates remaining valence for each real atom based on self.atoms and self.bonds."""
        num_real_atoms = len(self.atoms) - 1
        if num_real_atoms <= 0: return np.array([], dtype=int)
        current_usage = self._get_current_valence_usage()
        try:
            total_valence = np.array([self.vocabulary_valence[atom_vocab_idx]
                                      for atom_vocab_idx in self.atoms[1:]], dtype=int)
        except IndexError as e:
            raise IndexError(f"Invalid atom vocab index found in self.atoms[1:]: {self.atoms[1:]}. Error: {e}")
        if len(total_valence) != len(current_usage):
             raise RuntimeError(f"Valence calculation mismatch: total_valence ({len(total_valence)}) vs current_usage ({len(current_usage)})")
        remaining = total_valence - current_usage
        remaining = np.maximum(0, remaining)
        return remaining

    def update_action_mask(self):
        """Creates the action mask based on the internal state, incorporating Rule 1 and Rule 2."""
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
            if num_real_atoms == 0: mask[1:] = True
            self.current_action_mask = mask

        elif self.current_action_level == 1:
            action_space_size = self.vocab_size + num_real_atoms + 1
            mask = np.ones(action_space_size, dtype=bool)

            anchor_atom_internal_idx = self.l0_selected_atom_idx
            if anchor_atom_internal_idx is None or anchor_atom_internal_idx <= 0 or anchor_atom_internal_idx > num_real_atoms:
                raise ValueError(f"L1 Mask Error: Invalid anchor atom index: {anchor_atom_internal_idx} (NumReal={num_real_atoms})")
            anchor_atom_0_idx = anchor_atom_internal_idx - 1

            # Unmask "Add Atom"
            if self.upper_limit_atoms is None or num_real_atoms < self.upper_limit_atoms:
                for i in range(self.vocab_size):
                    action_idx = i
                    atom_type_vocab_idx = i + 1
                    if not self.atom_feasibility_mask[i] and self.vocabulary_valence[atom_type_vocab_idx] >= 1:
                        if remaining_valence[anchor_atom_0_idx] > 0:
                            mask[action_idx] = False

            # Unmask "Select Existing Atom"
            for target_0_idx in range(num_real_atoms):
                target_internal_idx = target_0_idx + 1
                action_idx = self.vocab_size + target_0_idx
                if target_internal_idx == anchor_atom_internal_idx: continue
                if target_0_idx >= len(remaining_valence):
                    raise IndexError(f"L1 Mask Error: Target index {target_0_idx} OOB for rem_val (len {len(remaining_valence)})")
                bond_exists = self.bonds[anchor_atom_internal_idx, target_internal_idx] > 0
                target_has_valence = remaining_valence[target_0_idx] > 0
                should_unmask = bond_exists or (target_has_valence and remaining_valence[anchor_atom_0_idx] > 0)
                if should_unmask: mask[action_idx] = False

            # Unmask "Remove Selected Atom" (Rule 1 check)
            remove_action_idx = self.vocab_size + num_real_atoms
            # Check index bounds before accessing is_original_atom
            if anchor_atom_internal_idx < len(self.is_original_atom):
                if num_real_atoms > 1 and self.is_original_atom[anchor_atom_internal_idx]:
                    mask[remove_action_idx] = False
            else:
                # This indicates a state inconsistency
                raise IndexError(f"L1 Mask Error: anchor_atom_internal_idx {anchor_atom_internal_idx} OOB for is_original_atom (len {len(self.is_original_atom)})")

            self.current_action_mask = mask

        elif self.current_action_level == 2:
            action_space_size = 7
            mask = np.ones(action_space_size, dtype=bool)

            atom_A_internal_idx = self.l0_selected_atom_idx
            atom_B_internal_idx = -1
            if self.l1_new_atom_type is not None: atom_B_internal_idx = len(self.atoms) - 1
            elif self.l1_selected_existing_atom_idx is not None: atom_B_internal_idx = self.l1_selected_existing_atom_idx
            else: raise RuntimeError("L2 Bond Mask Error: L1 context missing.")

            num_real_atoms = len(self.atoms) - 1
            if (atom_A_internal_idx is None or atom_A_internal_idx <= 0 or atom_A_internal_idx > num_real_atoms or
                    atom_B_internal_idx <= 0 or atom_B_internal_idx > num_real_atoms):
                raise ValueError(f"L2 Bond Mask Error: Invalid indices A={atom_A_internal_idx}, B={atom_B_internal_idx} (NumReal={num_real_atoms})")

            # --- Rule 2 Check ---
            current_min_idx = min(atom_A_internal_idx, atom_B_internal_idx)
            current_max_idx = max(atom_A_internal_idx, atom_B_internal_idx)
            if self.last_bond_action_details is not None and \
               self.last_bond_action_details[0] == current_min_idx and \
               self.last_bond_action_details[1] == current_max_idx:
                # Last action was a bond action on this same pair. Mask ALL L2 actions.
                mask[:] = True
                self.current_action_mask = mask
                return # Skip normal valence checks

            # --- Normal L2 Mask Logic (If Rule 2 check passes) ---
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
                if order <= max_allowed_final_order: mask[action_idx] = False
            if current_bond_order > 0: mask[6] = False # Unmask Remove Bond

            self.current_action_mask = mask
        else:
            raise ValueError(f"Invalid current_action_level: {self.current_action_level}")

    def _adjust_indices_after_removal(self, removed_internal_idx: int):
        """Adjusts stored internal indices after an atom removal."""
        if self.l0_selected_atom_idx is not None and self.l0_selected_atom_idx > removed_internal_idx:
            self.l0_selected_atom_idx -= 1
        if self.l1_selected_existing_atom_idx is not None and self.l1_selected_existing_atom_idx > removed_internal_idx:
            self.l1_selected_existing_atom_idx -= 1
        # last_bond_action_details uses internal indices, needs adjustment if involved indices > removed_idx
        if self.last_bond_action_details is not None:
            min_idx, max_idx = self.last_bond_action_details
            new_min = min_idx - 1 if min_idx > removed_internal_idx else min_idx
            new_max = max_idx - 1 if max_idx > removed_internal_idx else max_idx
            # Check if the removed atom was part of the last bond action pair
            if min_idx == removed_internal_idx or max_idx == removed_internal_idx:
                self.last_bond_action_details = None # Invalidate if removed atom was involved
            elif new_min != min_idx or new_max != max_idx:
                 self.last_bond_action_details = (new_min, new_max) # Update adjusted indices


    # def take_action(self, action: int):
    #     """Execute a given action, updating internal state directly."""
    #     if self.synthesis_done: raise RuntimeError("Cannot take action on terminated design.")
    #
    #     if self.current_action_mask is None or action < 0 or action >= len(self.current_action_mask) or self.current_action_mask[action]:
    #         mask_len = "None" if self.current_action_mask is None else len(self.current_action_mask)
    #         raise ValueError(f"Action {action} masked or invalid for level {self.current_action_level}. MaskLen={mask_len}")
    #
    #     current_level = self.current_action_level
    #     next_level = 0
    #     self.history.append(int(action))
    #     num_real_atoms_before = len(self.atoms) - 1
    #
    #     try:
    #         reset_last_bond_action = True # Default: reset unless it's an L2 bond action
    #
    #         if current_level == 0:
    #             if action == 0: # Terminate
    #                 if not self.is_terminable(): raise RuntimeError("Attempted Terminate when not allowed.")
    #                 self.synthesis_done = True
    #                 self.finalize(assert_feasible=False)
    #                 next_level = -1
    #             else: # Select Atom
    #                 if not (1 <= action <= num_real_atoms_before): raise ValueError(f"L0 Select Atom: Invalid index {action}.")
    #                 self.l0_selected_atom_idx = action
    #                 self.l1_new_atom_type = None
    #                 self.l1_selected_existing_atom_idx = None
    #                 next_level = 1
    #         elif current_level == 1:
    #             remove_action_idx = self.vocab_size + num_real_atoms_before
    #             anchor_idx = self.l0_selected_atom_idx
    #             if anchor_idx is None: raise RuntimeError("L1 take_action: l0_selected_atom_idx is None.")
    #
    #             if action < self.vocab_size: # Add Atom
    #                 self.l1_new_atom_type = action + 1
    #                 self.atoms = np.append(self.atoms, self.l1_new_atom_type)
    #                 new_size = len(self.atoms); new_idx = new_size - 1
    #                 self.bonds = np.pad(self.bonds, [(0, 1), (0, 1)], 'constant', constant_values=0)
    #                 self.bonds[0, new_idx] = self.bonds[new_idx, 0] = self.virtual_bond_idx
    #                 # Rule 1 Update
    #                 self.is_original_atom = np.append(self.is_original_atom, False)
    #                 next_level = 2
    #             elif action < remove_action_idx: # Select Existing Atom
    #                 selected_internal_idx = (action - self.vocab_size) + 1
    #                 if selected_internal_idx == anchor_idx: raise ValueError("L1 Select Existing: Cannot select anchor.")
    #                 if not (1 <= selected_internal_idx <= num_real_atoms_before): raise ValueError(f"L1 Select Existing: Invalid target index {selected_internal_idx}.")
    #                 self.l1_selected_existing_atom_idx = selected_internal_idx
    #                 next_level = 2
    #             elif action == remove_action_idx: # Remove Selected Atom
    #                 if num_real_atoms_before <= 1: raise RuntimeError("Attempted to remove last real atom.")
    #                 if not self.is_original_atom[anchor_idx]: raise RuntimeError("Attempted to remove non-original atom.") # Rule 1 check
    #                 removed_idx_for_adjust = anchor_idx
    #                 self.atoms = np.delete(self.atoms, removed_idx_for_adjust)
    #                 self.bonds = np.delete(np.delete(self.bonds, removed_idx_for_adjust, 0), removed_idx_for_adjust, 1)
    #                 # Rule 1 Update
    #                 self.is_original_atom = np.delete(self.is_original_atom, removed_idx_for_adjust)
    #                 self._adjust_indices_after_removal(removed_idx_for_adjust)
    #                 self.l0_selected_atom_idx = None; self.l1_new_atom_type = None; self.l1_selected_existing_atom_idx = None
    #                 next_level = 0
    #
    #             else: raise ValueError(f"Invalid L1 action index: {action}")
    #         elif current_level == 2:
    #             reset_last_bond_action = False # Don't reset, update instead
    #             idx_A = self.l0_selected_atom_idx
    #             idx_B = -1
    #             if self.l1_new_atom_type is not None: idx_B = len(self.atoms) - 1
    #             elif self.l1_selected_existing_atom_idx is not None: idx_B = self.l1_selected_existing_atom_idx
    #             else: raise RuntimeError("L2 take_action: L1 context missing.")
    #
    #             current_num_real_atoms = len(self.atoms) - 1
    #             if (idx_A is None or idx_A <= 0 or idx_A > current_num_real_atoms or
    #                     idx_B <= 0 or idx_B > current_num_real_atoms):
    #                  raise ValueError(f"L2 take_action: Invalid indices A={idx_A}, B={idx_B}.")
    #
    #             if action <= 5: # Set Order
    #                 order = action + 1
    #                 self.bonds[idx_A, idx_B] = self.bonds[idx_B, idx_A] = order
    #             elif action == 6: # Remove Bond
    #                 self.bonds[idx_A, idx_B] = self.bonds[idx_B, idx_A] = 0
    #             else: raise ValueError(f"Invalid L2 Bond action index: {action}")
    #
    #             # Rule 2 Update
    #             self.last_bond_action_details = (min(idx_A, idx_B), max(idx_A, idx_B))
    #
    #             self.l0_selected_atom_idx = None; self.l1_new_atom_type = None; self.l1_selected_existing_atom_idx = None
    #             next_level = 0
    #
    #         # --- Reset Rule 2 tracker if necessary ---
    #         if reset_last_bond_action:
    #             self.last_bond_action_details = None
    #
    #         # --- Update Mask and Level ---
    #         if next_level != -1:
    #             try: self._check_and_update_connectivity()
    #             except Exception as e:
    #                  self.infeasibility_flag = True; self.synthesis_done = True; self.current_action_mask = None
    #                  raise RuntimeError(f"Connectivity check failed after action {action}: {e}") from e
    #             self.current_action_level = next_level
    #             self.update_action_mask()
    #         else:
    #             self.current_action_mask = None
    #
    #     except Exception as e:
    #         self.infeasibility_flag = True; self.synthesis_done = True; self.current_action_mask = None
    #         raise RuntimeError(f"Error during take_action(action={action}, L{current_level}): {e}") from e

    def take_action(self, action: int):
        """Execute a given action, updating internal state directly."""
        if self.synthesis_done:
            raise RuntimeError("Cannot take action on terminated design.")

        if self.current_action_mask is None or action < 0 or action >= len(self.current_action_mask) or \
                self.current_action_mask[action]:
            mask_len = "None" if self.current_action_mask is None else len(self.current_action_mask)
            raise ValueError(
                f"Action {action} masked or invalid for level {self.current_action_level}. MaskLen={mask_len}")

        current_level = self.current_action_level
        next_level = 0
        self.history.append(int(action))
        num_real_atoms_before = len(self.atoms) - 1

        try:
            reset_last_bond_action = True

            if current_level == 0:
                if action == 0:  # Terminate
                    # is_terminable check done externally in generate_single_transformation
                    # if not self.is_terminable(): raise RuntimeError("Attempted Terminate when not allowed.")
                    self.synthesis_done = True  # Correctly set only on explicit Terminate
                    self.finalize(assert_feasible=False)
                    next_level = -1
                else:  # Select Atom
                    if not (1 <= action <= num_real_atoms_before):
                        raise ValueError(f"L0 Select Atom: Invalid index {action}.")
                    self.l0_selected_atom_idx = action
                    self.l1_new_atom_type = None
                    self.l1_selected_existing_atom_idx = None
                    self.l1_action_type = None
                    next_level = 1
            elif current_level == 1:
                remove_action_idx = self.vocab_size + num_real_atoms_before
                anchor_idx = self.l0_selected_atom_idx
                if anchor_idx is None:
                    raise RuntimeError("L1 take_action: l0_selected_atom_idx is None.")

                if action < self.vocab_size:  # Add Atom
                    self.l1_action_type = ActionType.ADD_ATOM
                    self.l1_new_atom_type = action + 1
                    self.atoms = np.append(self.atoms, self.l1_new_atom_type)
                    new_size = len(self.atoms)
                    new_idx = new_size - 1
                    self.bonds = np.pad(self.bonds, [(0, 1), (0, 1)], 'constant', constant_values=0)
                    self.bonds[0, new_idx] = self.bonds[new_idx, 0] = self.virtual_bond_idx
                    self.is_original_atom = np.append(self.is_original_atom, False)
                    next_level = 2
                elif action < remove_action_idx:  # Select Existing Atom
                    selected_internal_idx = (action - self.vocab_size) + 1
                    if selected_internal_idx == anchor_idx:
                        raise ValueError("L1 Select Existing: Cannot select anchor.")
                    if not (1 <= selected_internal_idx <= num_real_atoms_before):
                        raise ValueError(f"L1 Select Existing: Invalid target index {selected_internal_idx}.")
                    self.l1_action_type = ActionType.SELECT_EXISTING_ATOM
                    self.l1_selected_existing_atom_idx = selected_internal_idx
                    next_level = 2
                elif action == remove_action_idx:  # Remove Selected Atom
                    if num_real_atoms_before <= 1:
                        raise RuntimeError("Attempted to remove last real atom.")
                    if not self.is_original_atom[anchor_idx]:
                        raise RuntimeError("Attempted to remove non-original atom.")
                    self.l1_action_type = ActionType.REMOVE_SELECTED_ATOM
                    removed_idx_for_adjust = anchor_idx
                    self.atoms = np.delete(self.atoms, removed_idx_for_adjust)
                    self.bonds = np.delete(np.delete(self.bonds, removed_idx_for_adjust, 0), removed_idx_for_adjust, 1)
                    self.is_original_atom = np.delete(self.is_original_atom, removed_idx_for_adjust)
                    self._adjust_indices_after_removal(removed_idx_for_adjust)
                    self.l0_selected_atom_idx = None
                    self.l1_new_atom_type = None
                    self.l1_selected_existing_atom_idx = None
                    self.l1_action_type = None
                    # High-level count handled externally
                    next_level = 0
                else:
                    raise ValueError(f"Invalid L1 action index: {action}")
            elif current_level == 2:
                reset_last_bond_action = False
                idx_A = self.l0_selected_atom_idx
                idx_B = -1
                if self.l1_action_type == ActionType.ADD_ATOM:
                    idx_B = len(self.atoms) - 1
                elif self.l1_action_type == ActionType.SELECT_EXISTING_ATOM:
                    idx_B = self.l1_selected_existing_atom_idx
                else:
                    raise RuntimeError(f"L2 take_action: L1 context missing or invalid ({self.l1_action_type}).")

                current_num_real_atoms = len(self.atoms) - 1
                if (idx_A is None or idx_A <= 0 or idx_A > current_num_real_atoms or
                        idx_B <= 0 or idx_B > current_num_real_atoms):
                    raise ValueError(f"L2 take_action: Invalid indices A={idx_A}, B={idx_B}.")

                if action <= 5:
                    self.bonds[idx_A, idx_B] = self.bonds[idx_B, idx_A] = action + 1
                elif action == 6:
                    self.bonds[idx_A, idx_B] = self.bonds[idx_B, idx_A] = 0
                else:
                    raise ValueError(f"Invalid L2 Bond action index: {action}")

                self.last_bond_action_details = (min(idx_A, idx_B), max(idx_A, idx_B))
                self.l0_selected_atom_idx = None
                self.l1_new_atom_type = None
                self.l1_selected_existing_atom_idx = None
                self.l1_action_type = None
                # High-level count handled externally
                next_level = 0

            if reset_last_bond_action:
                self.last_bond_action_details = None

            # --- Update Mask and Level ---
            # Moved connectivity check and mask update inside the main try block
            if next_level != -1:
                # Update connectivity and mask *before* returning from the successful action path
                self._check_and_update_connectivity()  # Can raise RuntimeError
                self.current_action_level = next_level
                self.update_action_mask()  # Can raise IndexError/ValueError
            else:  # next_level == -1 (Terminate action)
                self.current_action_mask = None

        # <<< Modified Exception Handling >>>
        except (ValueError, IndexError) as e:
            # These indicate problems with action selection/masking logic or indices
            # Mark as infeasible and re-raise to stop the current sequence generation attempt
            self.infeasibility_flag = True
            self.current_action_mask = None  # Stop further actions
            # Do NOT set synthesis_done here
            raise RuntimeError(f"Masking/Action logic error at L{current_level}, action {action}: {e}") from e
        except RuntimeError as e:
            # Catch specific RuntimeErrors (e.g., from connectivity check, or raised above)
            self.infeasibility_flag = True
            self.current_action_mask = None
            # Do NOT set synthesis_done here
            # Re-raise the original RuntimeError
            raise e
            # Or wrap it:
            # raise RuntimeError(f"Runtime error during take_action(action={action}, L{current_level}): {e}") from e
        except Exception as e:
            # Catch any other unexpected errors
            self.infeasibility_flag = True
            self.current_action_mask = None
            # Do NOT set synthesis_done here
            print(f"CRITICAL: Unexpected error in take_action(action={action}, L{current_level}): {e}")
            # Re-raise as RuntimeError to ensure generate_single_transformation catches it
            raise RuntimeError(f"Unexpected error: {e}") from e
        # <<< End Modified Exception Handling >>>

    def finalize(self, assert_feasible: bool = False):
        """Finalize molecule design: build RDKit mol, sanitize, cache SMILES."""
        if self._cached_smiles is not None or self._cached_rdkit_mol is not None: return

        try: self._check_and_update_connectivity()
        except Exception as e:
             self.infeasibility_flag = True; print(f"Warning: Connectivity check failed during finalize: {e}.")

        if assert_feasible:
            try: self.assert_feasible()
            except AssertionError as e:
                print(f"Warning: Feasibility assertion failed during finalize: {e}"); self.infeasibility_flag = True

        num_real_atoms = len(self.atoms) - 1
        if num_real_atoms > 0 and not self.is_currently_connected:
            # print("Warning: Final molecule is disconnected.")
            self.infeasibility_flag = True # Optional: Mark disconnected as infeasible

        rdkit_mol = None
        if not self.infeasibility_flag:
            try:
                rdkit_mol = self.to_rdkit_mol(sanitize=False)
                if rdkit_mol.GetNumAtoms() == 0 and num_real_atoms > 0:
                    print("Warning: RDKit mol empty despite internal atoms."); self.infeasibility_flag = True
                elif rdkit_mol.GetNumAtoms() > 0:
                    try:
                        self._cached_rdkit_mol = copy.deepcopy(rdkit_mol) # Store unsanitized
                        Chem.SanitizeMol(rdkit_mol)
                        self._cached_rdkit_mol = rdkit_mol # Overwrite with sanitized
                        self._cached_smiles = Chem.MolToSmiles(rdkit_mol)
                    except Exception as e:
                        print(f"Warning: Final sanitization/SMILES failed: {e}."); self._cached_smiles = None
                        if self._cached_rdkit_mol is None: self.infeasibility_flag = True
                else:
                    self._cached_smiles = ""; self._cached_rdkit_mol = rdkit_mol
            except Exception as e:
                 print(f"Warning: Error during RDKit mol generation in finalize: {e}")
                 self.infeasibility_flag = True; self._cached_smiles = None; self._cached_rdkit_mol = None
        else:
            self._cached_smiles = None; self._cached_rdkit_mol = None
        self.synthesis_done = True

    def assert_feasible(self):
        """Check internal state consistency. Raises AssertionError on failure."""
        if not isinstance(self.atoms, np.ndarray) or not isinstance(self.bonds, np.ndarray) or not isinstance(self.is_original_atom, np.ndarray):
             raise AssertionError("Internal state types incorrect.")
        assert self.atoms[0] == 0, "Virtual atom missing/incorrect."
        num_atoms = len(self.atoms); num_real_atoms = num_atoms - 1
        assert len(self.is_original_atom) == num_atoms, "is_original_atom length mismatch."
        assert not self.is_original_atom[0], "Virtual atom marked as original."

        if num_real_atoms > 0:
             valid_indices = all(1 <= idx <= self.vocab_size for idx in self.atoms[1:])
             assert valid_indices, f"Invalid atom vocab index: {self.atoms[1:]}"
             allowed_check = all(not self.atom_feasibility_mask[idx - 1] for idx in self.atoms[1:])
             assert allowed_check, f"Disallowed atom type: {self.atoms[1:]}"
        if self.upper_limit_atoms is not None: assert num_real_atoms <= self.upper_limit_atoms, "Max atoms exceeded."
        assert self.bonds.shape == (num_atoms, num_atoms), "Bonds shape mismatch."
        assert not np.any(self.bonds.diagonal()), "Self-loops detected."
        assert np.all(self.bonds == self.bonds.T), "Bond matrix not symmetric."

        if num_real_atoms > 0:
             assert np.all(self.bonds[0, 1:] == self.virtual_bond_idx), "Virtual bond missing row 0."
             assert np.all(self.bonds[1:, 0] == self.virtual_bond_idx), "Virtual bond missing col 0."
             try:
                  remaining_valence = self._get_remaining_valence()
                  assert np.all(remaining_valence >= 0), f"Valence constraints violated: {remaining_valence}"
             except (IndexError, RuntimeError) as e: raise AssertionError(f"Valence check failed: {e}")

    def to_rdkit_mol(self, sanitize=True) -> Chem.RWMol:
        """Creates a *new* RDKit molecule from the internal state."""
        mol = Chem.RWMol()
        num_total_atoms = len(self.atoms)
        if num_total_atoms <= 1: return mol

        rdkit_idx_map = {}
        for internal_idx, atom_vocab_idx in enumerate(self.atoms):
            if internal_idx == 0: continue
            if not (1 <= atom_vocab_idx <= self.vocab_size): raise ValueError(f"Invalid vocab index {atom_vocab_idx} at internal {internal_idx}.")
            try: atom_config = self.atom_vocabulary[self.vocabulary_atom_names[atom_vocab_idx - 1]]
            except (IndexError, KeyError) as e: raise RuntimeError(f"Cannot get config for vocab index {atom_vocab_idx}: {e}")

            a = Chem.Atom(atom_config["atomic_number"])
            if "formal_charge" in atom_config: a.SetFormalCharge(atom_config["formal_charge"])
            ct = atom_config.get("chiral_tag", 0)
            if ct == 1: a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
            elif ct == 2: a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
            else: a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            new_rdkit_idx = mol.AddAtom(a)
            rdkit_idx_map[internal_idx] = new_rdkit_idx

        for i in range(1, num_total_atoms):
            for j in range(i + 1, num_total_atoms):
                bond_order = self.bonds[i, j]
                if bond_order > 0 and bond_order <= self.maximum_bond_order:
                    if i not in rdkit_idx_map or j not in rdkit_idx_map: raise RuntimeError(f"Missing RDKit map entry for {i} or {j}.")
                    rdkit_i, rdkit_j = rdkit_idx_map[i], rdkit_idx_map[j]
                    rdkit_bond_type = self.bond_types.get(int(bond_order))
                    if rdkit_bond_type: mol.AddBond(rdkit_i, rdkit_j, rdkit_bond_type)
                    else: print(f"Warning: Could not find RDKit bond type for order {bond_order}.")
                elif bond_order > self.maximum_bond_order and bond_order != self.virtual_bond_idx:
                     print(f"Warning: Invalid bond order {bond_order} between {i},{j}.")

        if sanitize:
            try: Chem.SanitizeMol(mol)
            except Exception as e: print(f"Warning: RDKit sanitization failed: {e}")
        return mol

    def is_terminable(self):
        """Checks if the current state allows termination."""
        num_real_atoms = len(self.atoms) - 1
        can_terminate = self.current_action_level == 0 and not self.synthesis_done
        connectivity_ok = (num_real_atoms > 0 and self.is_currently_connected)
        return can_terminate and connectivity_ok

    def to_smiles(self, canonical: bool = True) -> Optional[str]:
        """Returns a canonical SMILES string. Finalizes if needed. Caches result."""
        if not self.synthesis_done: self.finalize(assert_feasible=False)
        if canonical and self._cached_smiles is not None: return self._cached_smiles

        if self._cached_rdkit_mol is not None:
            try:
                mol_to_use = copy.deepcopy(self._cached_rdkit_mol)
                Chem.SanitizeMol(mol_to_use)
                smiles = Chem.MolToSmiles(mol_to_use, canonical=canonical)
                if canonical: self._cached_smiles = smiles
                return smiles
            except Exception as e:
                print(f"Warning: Failed to generate SMILES (canonical={canonical}): {e}")
                return None
        else: return self._cached_smiles # None or ""

    @staticmethod
    def init_batch_from_instance_list(config: MoleculeConfig, instances: List[int], network: Optional[nn.Module]=None, device: Optional[torch.device]=None):
        return [MoleculeDesign(config=config, initial_atom=atom) for atom in instances]

    @staticmethod
    def log_probability_fn(trajectories: List['MoleculeDesign'], network: nn.Module) -> List[np.array]:
        """Calculates masked log probabilities for the current action level."""
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
                else: log_probs_to_return.append(np.array([])); continue

                mask_len = len(mask)
                if len(logits) > mask_len: logits = logits[:mask_len]
                elif len(logits) < mask_len: raise ValueError(f"Logits/Mask length mismatch L{mol.current_action_level}: {len(logits)} vs {mask_len}")

                logits[mask] = -np.inf; max_logit = np.max(logits)
                if np.isneginf(max_logit): log_probs = logits
                else: exp_logits = np.exp(logits - max_logit); log_sum_exp = np.log(np.sum(exp_logits)); log_probs = logits - (max_logit + log_sum_exp); log_probs[mask] = -np.inf
                log_probs_to_return.append(log_probs)
        return log_probs_to_return

    def transition_fn(self, action: int) -> Tuple['BaseTrajectory', bool]:
        """Creates a copy, takes the action, and returns the new state and termination status."""
        copied_molecule = copy.deepcopy(self)
        try:
            # This call should raise ValueError/IndexError on invalid/masked actions
            copied_molecule.take_action(action)
        except (ValueError, IndexError) as e:
            # Re-raise errors related to invalid actions immediately
            raise e
        except RuntimeError as e:
            # Catch only unexpected RuntimeErrors from deep within take_action (like connectivity check failing)
            copied_molecule.infeasibility_flag = True
            copied_molecule.synthesis_done = True
            copied_molecule.current_action_mask = None
            print(
                f"Warning: transition_fn caught RuntimeError in take_action({action}): {e}. Returning infeasible state.")
        # No generic 'except Exception'
        return copied_molecule, copied_molecule.synthesis_done

    def to_max_evaluation_fn(self) -> float:
        """Returns the objective value, penalizing infeasible states."""
        if self.objective is None: return float("-inf")
        return float("-inf") if self.infeasibility_flag else self.objective

    def num_actions(self) -> int:
        """Returns the number of valid (unmasked) actions at the current level."""
        if self.current_action_mask is None: return 0
        return int(np.sum(~self.current_action_mask))

    @staticmethod
    def list_to_batch(molecules: List['MoleculeDesign'], device: torch.device = None,
                      include_feasibility_masks: bool = False) -> dict:
        """Converts a list of MoleculeDesign instances to a batch dictionary."""
        if not molecules: return {}
        first_mol = molecules[0]
        atoms_padding_idx = first_mol.vocab_size + 1
        max_valence = max([-1] + [v for v in first_mol.vocabulary_valence if v is not None and v >= 0])
        degree_padding_idx = max_valence + 2
        bond_padding_idx = MoleculeDesign.virtual_bond_idx + 1
        device = torch.device("cpu") if device is None else device
        num_atoms = [len(mol.atoms) for mol in molecules];
        max_num_atoms = max(num_atoms) if num_atoms else 0
        batch_level_idx = [mol.current_action_level for mol in molecules]

        batch_picked_atom_mhe = np.zeros((len(molecules), max_num_atoms), dtype=int)
        for i, mol in enumerate(molecules):
            anchor_idx = mol.l0_selected_atom_idx
            if mol.current_action_level >= 1 and anchor_idx is not None:
                # Mark the anchor atom (L0 selection) with 1
                if 0 <= anchor_idx < max_num_atoms:
                    batch_picked_atom_mhe[i, anchor_idx] = 1
                else:
                    print(
                        f"Warning: Anchor index {anchor_idx} out of bounds for mhe (max={max_num_atoms})")  # Defensiveness

                # --- START NEW LOGIC ---
                if mol.current_action_level == 2:
                    target_idx = None
                    # Determine target based on L1 action outcome stored in state
                    if mol.l1_action_type == ActionType.ADD_ATOM:
                        # Target is the last atom added (use its 1-based internal index)
                        target_idx = len(mol.atoms) - 1
                    elif mol.l1_action_type == ActionType.SELECT_EXISTING_ATOM:
                        # Target is the one explicitly selected (already 1-based internal index)
                        target_idx = mol.l1_selected_existing_atom_idx
                    # Note: No target if L1 was REMOVE_SELECTED_ATOM (level would be 0 anyway)

                    # Mark the target atom (L1 selection outcome) with 2 if applicable
                    if target_idx is not None:
                        if 0 <= target_idx < max_num_atoms:  # Check bounds for target
                            if target_idx != anchor_idx:  # Ensure target is not the same as anchor
                                batch_picked_atom_mhe[i, target_idx] = 2
                            else:
                                # This case might indicate an issue upstream (e.g., masking)
                                print(f"Warning: Target index {target_idx} is same as anchor index {anchor_idx}")
                        else:
                            print(
                                f"Warning: Target index {target_idx} out of bounds for mhe (max={max_num_atoms})")  # Defensiveness
                # --- END NEW LOGIC ---

        batch_atoms = np.stack([np.pad(mol.atoms, (0, max_num_atoms - num_atoms[i]), mode='constant',
                                       constant_values=atoms_padding_idx) if num_atoms[i] > 0 else np.full(
            max_num_atoms, fill_value=atoms_padding_idx, dtype=np.uint8) for i, mol in enumerate(molecules)])

        batch_atoms_degree = []
        for i, mol in enumerate(molecules):
            n = num_atoms[i]
            if n > 1:
                d_real = (mol.bonds[1:n, 1:n] > 0).sum(axis=1); d = np.concatenate(([0], d_real)); p_d = np.pad(d, (
                0, max_num_atoms - n), mode='constant', constant_values=degree_padding_idx)
            elif n == 1:
                p_d = np.pad(np.array([0]), (0, max_num_atoms - 1), mode='constant', constant_values=degree_padding_idx)
            else:
                p_d = np.full(max_num_atoms, fill_value=degree_padding_idx, dtype=int)
            batch_atoms_degree.append(p_d)
        batch_atoms_degree = np.stack(batch_atoms_degree)

        bonds_list = []
        for i, mol in enumerate(molecules):
            n = num_atoms[i]
            if n > 0:
                p_b = np.pad(mol.bonds, [(0, max_num_atoms - n), (0, max_num_atoms - n)], mode="constant",
                             constant_values=bond_padding_idx); np.fill_diagonal(p_b, bond_padding_idx)
            else:
                p_b = np.full((max_num_atoms, max_num_atoms), fill_value=bond_padding_idx, dtype=int)
            bonds_list.append(p_b)
        batch_bonds = np.stack(bonds_list)

        additive_padding_masks = []
        for i, mol in enumerate(molecules):
            n = num_atoms[i]
            if n > 0:
                m = np.zeros((n, n), dtype=float); p_m = np.pad(m, [(0, max_num_atoms - n), (0, max_num_atoms - n)],
                                                                mode="constant",
                                                                constant_values=-np.inf); np.fill_diagonal(p_m, 0.0)
            else:
                p_m = np.full((max_num_atoms, max_num_atoms), fill_value=-np.inf, dtype=float); np.fill_diagonal(p_m,
                                                                                                                 0.0)
            additive_padding_masks.append(p_m)
        batch_additive_padding_attn_mask = np.stack(additive_padding_masks)

        return_dict = dict(
            level_idx=torch.tensor(batch_level_idx, dtype=torch.long, device=device),
            picked_atom_mhe=torch.from_numpy(batch_picked_atom_mhe).long().to(device),
            num_atoms=torch.tensor(num_atoms, dtype=torch.long, device=device),
            atoms=torch.from_numpy(batch_atoms).long().to(device),
            atoms_degree=torch.from_numpy(batch_atoms_degree).long().to(device),
            bonds=torch.from_numpy(batch_bonds).long().to(device),
            additive_padding_attn_mask=torch.from_numpy(batch_additive_padding_attn_mask).float().to(device),
        )

        if include_feasibility_masks:
            masks_l0, masks_l1, masks_l2 = [], [], []
            max_actions_l0, max_actions_l1, max_actions_l2 = 0, 0, 7
            for mol in molecules:
                num_real = len(mol.atoms) - 1;
                max_actions_l0 = max(max_actions_l0, 1 + num_real);
                max_actions_l1 = max(max_actions_l1, mol.vocab_size + num_real + 1)
            for mol in molecules:
                num_real = len(mol.atoms) - 1
                len_l0 = 1 + num_real;
                mask_l0 = mol.current_action_mask if mol.current_action_level == 0 and mol.current_action_mask is not None else np.ones(
                    len_l0, dtype=bool);
                mask_l0 = mask_l0 if len(mask_l0) == len_l0 else np.ones(len_l0, dtype=bool);
                p_mask_l0 = np.pad(mask_l0, (0, max_actions_l0 - len_l0), mode='constant', constant_values=True);
                masks_l0.append(p_mask_l0)
                len_l1 = mol.vocab_size + num_real + 1;
                mask_l1 = mol.current_action_mask if mol.current_action_level == 1 and mol.current_action_mask is not None else np.ones(
                    len_l1, dtype=bool);
                mask_l1 = mask_l1 if len(mask_l1) == len_l1 else np.ones(len_l1, dtype=bool);
                p_mask_l1 = np.pad(mask_l1, (0, max_actions_l1 - len_l1), mode='constant', constant_values=True);
                masks_l1.append(p_mask_l1)
                len_l2 = 7;
                mask_l2 = mol.current_action_mask if mol.current_action_level == 2 and mol.current_action_mask is not None else np.ones(
                    len_l2, dtype=bool);
                mask_l2 = mask_l2 if len(mask_l2) == len_l2 else np.ones(len_l2, dtype=bool);
                masks_l2.append(mask_l2)
            return_dict["feasibility_mask_level_zero"] = torch.from_numpy(np.stack(masks_l0)).bool().to(device)
            return_dict["feasibility_mask_level_one"] = torch.from_numpy(np.stack(masks_l1)).bool().to(device)
            return_dict["feasibility_mask_level_two"] = torch.from_numpy(np.stack(masks_l2)).bool().to(device)
        return return_dict

    @staticmethod
    def batch_to_device(batch: dict, device: torch.device):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    @staticmethod
    def get_single_atom_molecules(config: MoleculeConfig, repeat: int = 1) -> List['MoleculeDesign']:
        atoms = [i + 1 for i, name in enumerate(config.atom_vocabulary.keys()) if config.atom_vocabulary[name]["allowed"]]
        if not atoms: raise ValueError("No allowed atoms found in vocabulary config.")
        return MoleculeDesign.init_batch_from_instance_list(config, atoms * repeat)

    @staticmethod
    def from_smiles(config: MoleculeConfig, smiles: str, **kwargs) -> Tuple['MoleculeDesign', Dict[int, int]]:
        """Creates instance from SMILES. Raises Error on failure."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: raise ValueError(f"Invalid SMILES input: {smiles}")
        try:
            # mol = Chem.RemoveHs(mol, sanitize=False)
            Chem.SanitizeMol(mol, catchErrors=True)
            Chem.Kekulize(mol, clearAromaticFlags=True)
            canonical_order = rdmolfiles.CanonicalRankAtoms(mol)
            mol = rdmolops.RenumberAtoms(mol, canonical_order)
        except Exception as e: raise ValueError(f"Could not preprocess input SMILES {smiles}: {e}") from e
        try: return MoleculeDesign.from_rdkit_mol(config, mol, smiles=smiles)
        except Exception as e: raise RuntimeError(f"Error during from_rdkit_mol for {smiles}: {e}") from e

    @staticmethod
    def from_rdkit_mol(config: MoleculeConfig, rdkit_mol: Chem.Mol, smiles: Optional[str] = None) -> Tuple['MoleculeDesign', Dict[int, int]]:
        """Creates instance from RDKit Mol. Raises Error on failure."""
        BOND_TYPE_TO_RL_ORDER = {
            Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3,
            Chem.BondType.QUADRUPLE: 4, Chem.BondType.QUINTUPLE: 5, Chem.BondType.HEXTUPLE: 6,
        }
        num_heavy_atoms = rdkit_mol.GetNumAtoms()
        first_allowed_atom_idx = 1 # Default needed for init
        try:
             for i, name in enumerate(config.atom_vocabulary.keys()):
                  if config.atom_vocabulary[name]["allowed"]: first_allowed_atom_idx = i + 1; break
        except Exception: pass

        if num_heavy_atoms == 0:
            print(f"Warning: Input mol {smiles or ''} empty. Creating empty design.")
            instance = MoleculeDesign(config, initial_atom=first_allowed_atom_idx)
            instance.atoms = np.array([0], dtype=np.uint8); instance.bonds = np.zeros((1, 1), dtype=np.uint8)
            instance.is_original_atom = np.array([False], dtype=bool); instance._check_and_update_connectivity(); instance.update_action_mask()
            return instance, {}

        try: reverse_atom_lookup = build_reverse_atom_lookup(config)
        except Exception as e: raise RuntimeError("Failed to build reverse atom lookup.") from e

        internal_atoms_list = [0]; rdkit_to_internal_map = {}; internal_idx_counter = 1
        for atom in rdkit_mol.GetAtoms():
            rdkit_idx = atom.GetIdx(); atomic_num = atom.GetAtomicNum(); charge = atom.GetFormalCharge(); chiral = int(atom.GetChiralTag())
            key = (atomic_num, charge, chiral); vocab_idx = reverse_atom_lookup.get(key)
            if vocab_idx is None and chiral != 0: vocab_idx = reverse_atom_lookup.get((atomic_num, charge, 0))
            if vocab_idx is None: raise ValueError(f"Atom type ({atomic_num},{charge},{chiral}) in {smiles or ''} not in vocab.")
            internal_atoms_list.append(vocab_idx); rdkit_to_internal_map[rdkit_idx] = internal_idx_counter; internal_idx_counter += 1

        num_total_atoms = len(internal_atoms_list)
        internal_bonds_matrix = np.zeros((num_total_atoms, num_total_atoms), dtype=np.uint8)
        for bond in rdkit_mol.GetBonds():
            idx1, idx2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(); bond_type = bond.GetBondType()
            rl_order = BOND_TYPE_TO_RL_ORDER.get(bond_type)
            if rl_order is None: raise ValueError(f"Unsupported bond type {bond_type} in {smiles or ''}. Ensure Kekulization.")
            try: int_idx1, int_idx2 = rdkit_to_internal_map[idx1], rdkit_to_internal_map[idx2]
            except KeyError: raise RuntimeError(f"RDKit index map failed for bond ({idx1}, {idx2}).")
            internal_bonds_matrix[int_idx1, int_idx2] = internal_bonds_matrix[int_idx2, int_idx1] = rl_order
        if num_total_atoms > 1: internal_bonds_matrix[0, 1:] = internal_bonds_matrix[1:, 0] = MoleculeDesign.virtual_bond_idx

        try:
            instance = MoleculeDesign(config, initial_atom=first_allowed_atom_idx)
            instance.atoms = np.array(internal_atoms_list, dtype=np.uint8)
            instance.bonds = internal_bonds_matrix
            # Rule 1 Init
            instance.is_original_atom = np.array([False] + [True] * num_heavy_atoms, dtype=bool)
            instance.synthesis_done = False; instance._cached_smiles = None; instance._cached_rdkit_mol = None
            instance.objective = None; instance.infeasibility_flag = False; instance.current_action_level = 0
            instance.history = []; instance.l0_selected_atom_idx = None; instance.l1_new_atom_type = None; instance.l1_selected_existing_atom_idx = None
            # Rule 2 Init
            instance.last_bond_action_details = None
            instance._check_and_update_connectivity(); instance.update_action_mask()
        except Exception as e: raise RuntimeError(f"Error creating/setting state for instance from {smiles or ''}: {e}") from e

        return instance, rdkit_to_internal_map