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
    Environment for molecular design using a revised hierarchical action space (v2025-04-18).

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
        self.vocabulary_valence = [-1] + [self.atom_vocabulary[x]["valence"] for x in self.vocabulary_atom_names]
        self.atom_feasibility_mask = [not self.atom_vocabulary[x]["allowed"] for x in self.vocabulary_atom_names]
        self.vocab_size = len(self.vocabulary_atom_idcs) # V

        # Set the actual index for the Remove Atom action in Level 2 Modify path
        self.REMOVE_ATOM_ACTION_L2_MODIFY = self.vocab_size # V

        self.upper_limit_atoms = self.config.max_num_atoms
        assert initial_atom in self.vocabulary_atom_idcs and not self.atom_feasibility_mask[initial_atom - 1], \
            f"Initial atom {initial_atom} must be in vocabulary {self.vocabulary_atom_idcs} and allowed in config."
        self.initial_atom = initial_atom

        # Internal State
        self.atoms = np.array([0, initial_atom], dtype=np.uint8)
        self.bonds = np.zeros((2, 2), dtype=np.uint8)
        self.bonds[0, 1] = self.bonds[1, 0] = self.virtual_bond_idx
        self.virtual_distance = self.maximum_num_atoms_overall + 1
        self.infinity_distance = self.maximum_num_atoms_overall + 2
        self.topological_distance_matrix = np.array([[0, self.virtual_distance], [self.virtual_distance, 0]], dtype=np.uint8)
        self.rdkit_mol = Chem.RWMol()
        self._add_atom_to_rdkit(initial_atom)

        # Trajectory State
        self.synthesis_done = False
        self.smiles_string: Optional[str] = None
        self.objective: Optional[float] = None
        self.sa_score: float = 0.
        self.infeasibility_flag: bool = False
        self.is_currently_connected: bool = True

        # Action Handling State
        self.current_action_level = 0
        self.current_action_mask: Optional[np.array] = None
        self.history: List[int] = []
        self.l0_selected_atom_idx: Optional[int] = None
        self.is_modifying_atom: bool = False
        self.atom_to_modify: Optional[int] = None
        self.l1_new_atom_type: Optional[int] = None
        self.l1_selected_existing_atom_idx: Optional[int] = None

        self._check_and_update_connectivity()
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
        self.rdkit_mol.AddAtom(a)

    def _check_and_update_connectivity(self):
        """Checks connectivity of the current rdkit_mol and updates the flag."""
        if self.rdkit_mol is None or self.rdkit_mol.GetNumAtoms() <= 1:
            self.is_currently_connected = True
            return
        try:
            frags = Chem.GetMolFrags(self.rdkit_mol, asMols=False, sanitizeFrags=False)
            self.is_currently_connected = (len(frags) == 1)
        except Exception as e:
             print(f"WARNING: GetMolFrags failed during connectivity check: {e}. Assuming disconnected.")
             self.is_currently_connected = False
             self.infeasibility_flag = True

    def _get_current_valence_usage(self) -> np.array:
        """Calculates the sum of explicit bond orders for each real atom."""
        num_real_atoms = len(self.atoms) - 1
        if num_real_atoms <= 0:
            return np.array([], dtype=int)

        # Sum bond orders directly from the bonds matrix (excluding virtual atom 0)
        # self.bonds[1:, 1:] is the adjacency matrix for real atoms
        # Sum along axis 1 to get usage for each row (atom)
        current_explicit_usage = np.sum(self.bonds[1: num_real_atoms + 1, 1: num_real_atoms + 1], axis=1)

        # --- DEBUG (Optional) ---
        # print(f"DEBUG _get_current_valence_usage: Explicit Usage = {current_explicit_usage}")
        # --- END DEBUG ---

        return current_explicit_usage.astype(int)

    # Inside MoleculeDesign class
    def _get_remaining_valence(self) -> np.array:
        """
        Calculates remaining valence available for new explicit bonds for each real atom,
        using the valence defined in the config for the specific atom type.
        """
        num_real_atoms = len(self.atoms) - 1
        if num_real_atoms <= 0:
            return np.array([], dtype=int)

        current_usage = self._get_current_valence_usage()  # Sum of explicit bonds

        # Get total valence for each real atom based on its type in self.atoms
        # self.atoms[1:] contains the 1-based vocabulary indices of real atoms
        # self.vocabulary_valence should map 1-based vocab index to total valence
        total_valence = np.array([self.vocabulary_valence[atom_vocab_idx]
                                  for atom_vocab_idx in self.atoms[1: num_real_atoms + 1]], dtype=int)

        # Calculate remaining valence for adding *explicit* bonds
        remaining = total_valence - current_usage

        # Ensure remaining valence is not negative (can happen if state is invalid)
        remaining = np.maximum(0, remaining)

        # --- DEBUG (Optional) ---
        # print(f"\n--- DEBUG: Entering _get_remaining_valence (Implicit H) ---")
        # print(f"Current atoms (real): {self.atoms[1:]}")
        # print(f"Total valence from vocab: {total_valence}")
        # print(f"Current explicit usage: {current_usage}")
        # print(f"Calculated remaining valence: {remaining}")
        # print(f"--- DEBUG: Exiting _get_remaining_valence ---\n")
        # --- END DEBUG ---

        return remaining

    def _sync_internal_state_from_rdkit(self):
        """Rebuilds internal state (atoms, bonds, distances) from self.rdkit_mol,
           attempting to preserve explicit bond orders by Kekulizing if aromaticity is detected."""
        # <<< --- START DEBUGGING --- >>>
        # print("\n--- DEBUG: Entering _sync_internal_state_from_rdkit ---") # Keep if needed
        if not isinstance(self.rdkit_mol, Chem.Mol):
            print("DEBUG _sync: self.rdkit_mol is not a valid Mol object. Skipping sync.")
            self.atoms = np.zeros(1, dtype=np.uint8)
            self.bonds = np.zeros((1, 1), dtype=np.uint8)
            self.topological_distance_matrix = np.zeros((1, 1), dtype=np.uint8)
            self.infeasibility_flag = True
            return

        try:
            num_atoms_rdkit_before = self.rdkit_mol.GetNumAtoms()
            num_bonds_rdkit_before = self.rdkit_mol.GetNumBonds()
            # print(f"DEBUG _sync: BEFORE sync: self.rdkit_mol has {num_atoms_rdkit_before} atoms and {num_bonds_rdkit_before} bonds.") # Keep if needed
        except Exception as pre_sync_err:
            print(f"DEBUG _sync: Error accessing rdkit_mol properties before sync: {pre_sync_err}")
            self.infeasibility_flag = True
            return
        # <<< --- END DEBUGGING --- >>>

        try:
            rdkit_atoms = self.rdkit_mol.GetAtoms()
            num_rdkit_atoms = len(rdkit_atoms)

            if num_rdkit_atoms == 0 and len(self.atoms) > 1:
                # print("DEBUG _sync: RDKit mol is empty, resetting internal state.") # Keep if needed
                self.atoms = np.zeros(1, dtype=np.uint8)
                self.bonds = np.zeros((1, 1), dtype=np.uint8)
                self.topological_distance_matrix = np.zeros((1, 1), dtype=np.uint8)
                return

            # 1. Rebuild self.atoms (Keep existing logic)
            self.atoms = np.zeros(1 + num_rdkit_atoms, dtype=np.uint8)
            prop_to_vocab_idx = {}
            for i, name in enumerate(self.vocabulary_atom_names):
                config = self.atom_vocabulary[name]
                key_parts = [str(config["atomic_number"])]
                key_parts.append(str(config.get("formal_charge", 0)))
                key_parts.append(str(config.get("chiral_tag", 0)))
                prop_to_vocab_idx["_".join(key_parts)] = i + 1

            for rdkit_idx, atom in enumerate(rdkit_atoms):
                key_parts = [str(atom.GetAtomicNum())]
                fc = atom.GetFormalCharge()
                ct = atom.GetChiralTag()
                key_parts.append(str(fc))
                chiral_tag_int = 0
                if ct == Chem.ChiralType.CHI_TETRAHEDRAL_CW: chiral_tag_int = 1
                elif ct == Chem.ChiralType.CHI_TETRAHEDRAL_CCW: chiral_tag_int = 2
                key_parts.append(str(chiral_tag_int))
                key = "_".join(key_parts)

                vocab_idx = prop_to_vocab_idx.get(key)
                if vocab_idx is None:
                    key_parts_no_chiral = [str(atom.GetAtomicNum()), str(fc), '0']
                    key_no_chiral = "_".join(key_parts_no_chiral)
                    vocab_idx = prop_to_vocab_idx.get(key_no_chiral)
                    # if vocab_idx is not None: print(f"DEBUG _sync: Warning - Atom {rdkit_idx} matched without chiral tag...") # Keep if needed

                if vocab_idx is None:
                    raise ValueError(f"RDKit atom {rdkit_idx} (Props: {key} or {key_no_chiral}) could not be mapped back to vocabulary.")

                internal_idx = rdkit_idx + 1
                self.atoms[internal_idx] = vocab_idx
            # print(f"DEBUG _sync: Rebuilt self.atoms: {self.atoms}") # Keep if needed

            # 2. Rebuild self.bonds (Revised Logic)
            num_total_atoms = 1 + num_rdkit_atoms
            if self.bonds.shape[0] != num_total_atoms:
                self.bonds = np.zeros((num_total_atoms, num_total_atoms), dtype=np.uint8)
            else:
                self.bonds.fill(0)

            if num_rdkit_atoms > 0:
                self.bonds[0, 1:] = self.virtual_bond_idx
                self.bonds[1:, 0] = self.virtual_bond_idx
                adj_matrix_view = self.bonds[1:, 1:] # View for real atoms

                bond_type_to_order = { # Map non-aromatic types
                    Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3,
                    Chem.BondType.QUADRUPLE: 4, Chem.BondType.QUINTUPLE: 5, Chem.BondType.HEXTUPLE: 6,
                }
                needs_kekulization = False
                aromatic_bond_indices = [] # Store indices of aromatic bonds

                # First pass: Set explicit bond orders, flag aromatic bonds
                for bond in self.rdkit_mol.GetBonds():
                    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    b_type = bond.GetBondType()
                    order = bond_type_to_order.get(b_type)
                    if order is not None:
                        adj_matrix_view[i, j] = adj_matrix_view[j, i] = order
                    elif b_type == Chem.BondType.AROMATIC:
                        needs_kekulization = True
                        aromatic_bond_indices.append(tuple(sorted((i, j))))
                    else:
                        print(f"DEBUG _sync: Warning - Unknown bond type {b_type} between atoms {i}-{j}. Ignored.")

                # Second pass: Handle aromatic bonds if needed
                if needs_kekulization:
                    # print("DEBUG _sync: Aromatic bonds detected, attempting Kekulization for sync...") # Keep if needed
                    try:
                        mol_copy_for_kekule = Chem.Mol(self.rdkit_mol)
                        Chem.Kekulize(mol_copy_for_kekule, clearAromaticFlags=True)
                        # Get explicit orders from the Kekulized copy
                        for bond_kekule in mol_copy_for_kekule.GetBonds():
                            i_k, j_k = bond_kekule.GetBeginAtomIdx(), bond_kekule.GetEndAtomIdx()
                            # Check if this bond corresponds to one that was originally aromatic
                            if tuple(sorted((i_k, j_k))) in aromatic_bond_indices:
                                b_type_k = bond_kekule.GetBondType()
                                order_k = bond_type_to_order.get(b_type_k)
                                if order_k is not None:
                                    adj_matrix_view[i_k, j_k] = adj_matrix_view[j_k, i_k] = order_k
                                else: # Should not happen after successful Kekulization
                                    print(f"DEBUG _sync: Warning - Bond type {b_type_k} still unknown after Kekulization attempt for {i_k}-{j_k}.")
                                    adj_matrix_view[i_k, j_k] = adj_matrix_view[j_k, i_k] = 1 # Fallback to 1
                    except Exception as kekule_err:
                        print(f"DEBUG _sync: Kekulization failed during sync ({kekule_err}). Representing aromatic bonds as order 1.")
                        # Fallback: Set order 1 for aromatic bonds if Kekulization failed
                        for i, j in aromatic_bond_indices:
                             adj_matrix_view[i, j] = adj_matrix_view[j, i] = 1 # Fallback to order 1

            # print(f"DEBUG _sync: Rebuilt self.bonds (shape: {self.bonds.shape})") # Keep if needed

            # 3. Rebuild self.topological_distance_matrix (Keep existing logic)
            if self.topological_distance_matrix.shape[0] != num_total_atoms:
                self.topological_distance_matrix = np.full((num_total_atoms, num_total_atoms), self.infinity_distance, dtype=np.uint8)
            else:
                self.topological_distance_matrix.fill(self.infinity_distance)
            np.fill_diagonal(self.topological_distance_matrix, 0)

            if num_rdkit_atoms > 0:
                self.topological_distance_matrix[0, 1:] = self.virtual_distance
                self.topological_distance_matrix[1:, 0] = self.virtual_distance
                try:
                    rdkit_dist_matrix_float = Chem.GetDistanceMatrix(self.rdkit_mol)
                    rdkit_dist_matrix = np.where(
                        (rdkit_dist_matrix_float > 0) & (rdkit_dist_matrix_float <= self.maximum_num_atoms_overall),
                        rdkit_dist_matrix_float.astype(np.uint8),
                        self.infinity_distance
                    )
                    np.fill_diagonal(rdkit_dist_matrix, 0)
                    self.topological_distance_matrix[1:num_total_atoms, 1:num_total_atoms] = rdkit_dist_matrix
                except Exception as e:
                    print(f"ERROR rebuilding distance matrix from RDKit: {e}. Distances may be inconsistent.")
                    self.infeasibility_flag = True
            # print(f"DEBUG _sync: Rebuilt self.topological_distance_matrix (shape: {self.topological_distance_matrix.shape})") # Keep if needed

        except Exception as sync_err:
            print(f"ERROR during main sync logic in _sync_internal_state_from_rdkit: {sync_err}")
            self.infeasibility_flag = True
            self.atoms = np.zeros(1, dtype=np.uint8)
            self.bonds = np.zeros((1, 1), dtype=np.uint8)
            self.topological_distance_matrix = np.zeros((1, 1), dtype=np.uint8)

        # print("--- DEBUG: Exiting _sync_internal_state_from_rdkit ---") # Keep if needed

    def update_action_mask(self):
        """Creates the action mask for the current action level based on the new action space."""
        if self.synthesis_done:
            self.current_action_mask = None
            return

        num_real_atoms = len(self.atoms) - 1 # N
        remaining_valence = self._get_remaining_valence() # 0-indexed array for real atoms

        # --- DEBUG START ---
        print(f"\n--- DEBUG: Entering update_action_mask ---")
        print(f"Current Level: {self.current_action_level}")
        print(f"Num Real Atoms (N): {num_real_atoms}")
        # --- DEBUG END ---

        if self.current_action_level == 0:
            # Level 0: Terminate (0) / Select Existing Atom (1..N)
            action_space_size = 1 + num_real_atoms
            mask = np.zeros(action_space_size, dtype=bool)
            if not self.is_currently_connected:
                mask[0] = True
            if num_real_atoms == 0:
                 mask[1:] = True
            self.current_action_mask = mask

        elif self.current_action_level == 1:
            # Level 1: Add New (0..V-1) / Select Existing (V..V+N-1) / Initiate Modify (V+N)
            action_space_size = self.vocab_size + num_real_atoms + 1
            mask = np.ones(action_space_size, dtype=bool)

            anchor_atom_internal_idx = self.l0_selected_atom_idx
            if anchor_atom_internal_idx is None or anchor_atom_internal_idx == 0:
                 raise ValueError("Level 1 reached without a valid L0 selected atom.")
            anchor_atom_0_idx = anchor_atom_internal_idx - 1

            # --- DEBUG START ---
            print(f"--- DEBUG: update_action_mask L1 ---")
            print(f"Anchor Atom Internal Idx: {anchor_atom_internal_idx}")
            print(f"Anchor Atom 0-Based Idx: {anchor_atom_0_idx}")
            print(f"Remaining Valence array: {remaining_valence}")
            if anchor_atom_0_idx < len(remaining_valence):
                print(
                    f"Remaining Valence for Anchor Atom (idx {anchor_atom_0_idx}): {remaining_valence[anchor_atom_0_idx]}")
            else:
                print(
                    f"ERROR: anchor_atom_0_idx {anchor_atom_0_idx} out of bounds for remaining_valence (len {len(remaining_valence)})")
            # --- DEBUG END ---

            # Unmask Add New Atom (0..V-1)
            can_add_new = False  # Default to false
            if anchor_atom_0_idx < len(remaining_valence):  # Bounds check
                can_add_new = remaining_valence[anchor_atom_0_idx] > 0

            # --- DEBUG START ---
            print(f"Condition 'can_add_new' (remaining_valence[anchor] > 0): {can_add_new}")
            # --- DEBUG END ---

            # Unmask Add New Atom (0..V-1)
            # can_add_new = remaining_valence[anchor_atom_0_idx] > 0
            if can_add_new:
                # --- DEBUG START ---
                print(f"DEBUG: Unmasking 'Add New Atom' actions...")
                # --- DEBUG END ---
                for i in range(self.vocab_size):
                    if not self.atom_feasibility_mask[i] and self.vocabulary_valence[i+1] >= 1:
                        mask[i] = False
            # --- DEBUG START ---
            else:
                print(f"DEBUG: NOT Unmasking 'Add New Atom' actions because can_add_new is False.")
            # --- DEBUG END ---

            # Unmask Select Existing Atom (V..V+N-1)
            for target_0_idx in range(num_real_atoms):
                target_internal_idx = target_0_idx + 1
                action_idx = self.vocab_size + target_0_idx

                if target_internal_idx == anchor_atom_internal_idx: continue

                bond_exists = self.bonds[anchor_atom_internal_idx, target_internal_idx] > 0
                can_increase_bond = (remaining_valence[anchor_atom_0_idx] > 0 and
                                     remaining_valence[target_0_idx] > 0)

                if bond_exists or can_increase_bond:
                    mask[action_idx] = False

            # Unmask Initiate Modify Atom (V+N)
            mask[self.vocab_size + num_real_atoms] = False
            self.current_action_mask = mask

            # --- DEBUG START ---
            print(f"Final L1 Mask (True=Masked): {mask}")
            print(f"--- DEBUG: Exiting update_action_mask L1 ---")
            # --- DEBUG END ---

        elif self.current_action_level == 2:
            if self.is_modifying_atom:
                # Level 2 (Modify Path): Replace Type (0..V-1) / Remove Atom (V)
                action_space_size = self.vocab_size + 1
                mask = np.ones(action_space_size, dtype=bool)

                atom_internal_idx = self.atom_to_modify
                if atom_internal_idx is None or atom_internal_idx == 0:
                     raise ValueError("L2 Modify path reached without atom_to_modify set.")
                rdkit_idx = atom_internal_idx - 1
                current_atom_type_idx = self.atoms[atom_internal_idx]

                # Unmask Replace Atom Type (0..V-1)
                current_usage = self._get_current_valence_usage()[rdkit_idx]
                for vocab_idx_0 in range(self.vocab_size):
                    new_atom_type = vocab_idx_0 + 1
                    action_idx = vocab_idx_0

                    if self.atom_feasibility_mask[vocab_idx_0]: continue
                    if new_atom_type == current_atom_type_idx: continue
                    if self.vocabulary_valence[new_atom_type] < current_usage: continue

                    # Simple check passed, assume RDKit check might pass (defer full check)
                    mask[action_idx] = False

                # Unmask Remove Atom (Action V)
                mask[self.REMOVE_ATOM_ACTION_L2_MODIFY] = False
                self.current_action_mask = mask

            else:
                # Level 2 (Bond Path): Set Bond 1-6 (0..5) / Remove Bond (6)
                action_space_size = 7
                mask = np.ones(action_space_size, dtype=bool)

                atom_A_internal_idx = self.l0_selected_atom_idx
                atom_B_internal_idx = -1
                if self.l1_new_atom_type is not None:
                     atom_B_internal_idx = len(self.atoms) - 1
                elif self.l1_selected_existing_atom_idx is not None:
                     atom_B_internal_idx = self.l1_selected_existing_atom_idx
                else:
                     raise ValueError("L2 Bond path reached without L1 context.")

                if atom_A_internal_idx is None or atom_A_internal_idx <= 0 or atom_B_internal_idx <= 0:
                     raise ValueError("Invalid atom indices for L2 Bond path.")

                atom_A_0_idx = atom_A_internal_idx - 1
                atom_B_0_idx = atom_B_internal_idx - 1

                current_bond_order = self.bonds[atom_A_internal_idx, atom_B_internal_idx]
                valence_A_rem = remaining_valence[atom_A_0_idx]
                valence_B_rem = remaining_valence[atom_B_0_idx]
                max_increase = min(valence_A_rem, valence_B_rem)
                max_allowed_final_order = min(int(current_bond_order + max_increase), self.maximum_bond_order)

                # Unmask Set Bond Order actions (0..5) -> order (1..6)
                for order in range(1, self.maximum_bond_order + 1):
                    action_idx = order - 1
                    if order <= max_allowed_final_order:
                        mask[action_idx] = False

                # Unmask Remove Bond action (6) if bond exists
                if current_bond_order > 0:
                    mask[6] = False
                self.current_action_mask = mask
        else:
             raise ValueError(f"Invalid current_action_level: {self.current_action_level}")

    def update_topological_distance_matrix(self):
        """Updates the distance matrix after structural changes."""
        num_total_atoms = len(self.atoms)
        current_size = self.topological_distance_matrix.shape[0]
        if num_total_atoms > current_size:
             pad_width = num_total_atoms - current_size
             self.topological_distance_matrix = np.pad(
                 self.topological_distance_matrix, [(0, pad_width), (0, pad_width)],
                 mode='constant', constant_values=self.infinity_distance
             )
             new_indices = range(current_size, num_total_atoms)
             self.topological_distance_matrix[0, new_indices] = self.virtual_distance
             self.topological_distance_matrix[new_indices, 0] = self.virtual_distance
             for i in new_indices: self.topological_distance_matrix[i, i] = 0
        elif num_total_atoms < current_size:
             # Handled by _sync_internal_state_from_rdkit calling this
             pass

        if self.rdkit_mol and self.rdkit_mol.GetNumAtoms() > 0:
             try:
                 rdkit_dist_matrix = Chem.GetDistanceMatrix(self.rdkit_mol).astype(np.uint8)
                 rdkit_dist_matrix[rdkit_dist_matrix <= 0] = self.infinity_distance
                 rdkit_dist_matrix[rdkit_dist_matrix > self.maximum_num_atoms_overall] = self.infinity_distance
                 np.fill_diagonal(rdkit_dist_matrix, 0)
                 self.topological_distance_matrix[1:num_total_atoms, 1:num_total_atoms] = rdkit_dist_matrix
             except Exception as e:
                 print(f"WARNING: Failed to update RDKit distance matrix: {e}.")
                 self.infeasibility_flag = True
        elif num_total_atoms > 1:
             pass # Initial state handled

    def _update_rdkit_bond(self, rdkit_idx1: int, rdkit_idx2: int, new_order: int):
        """Adds or modifies a bond in self.rdkit_mol. new_order=0 removes."""
        try:
             existing_bond = self.rdkit_mol.GetBondBetweenAtoms(rdkit_idx1, rdkit_idx2)
             if existing_bond:
                 # Store flags before removing
                 is_aromatic = existing_bond.GetIsAromatic() # Check if it WAS aromatic
                 self.rdkit_mol.RemoveBond(rdkit_idx1, rdkit_idx2)

             if new_order > 0 and new_order <= self.maximum_bond_order:
                 rdkit_bond_type = self.bond_types.get(new_order)
                 if rdkit_bond_type:
                     self.rdkit_mol.AddBond(rdkit_idx1, rdkit_idx2, rdkit_bond_type)
                     # Note: AddBond does NOT automatically set aromatic flags.
                     # Sanitization below will handle aromaticity perception.
                 else:
                     print(f"WARNING: Invalid bond order {new_order} requested for RDKit.")
                     self.infeasibility_flag = True
             elif new_order > self.maximum_bond_order:
                  print(f"WARNING: Bond order {new_order} exceeds maximum {self.maximum_bond_order}.")
                  self.infeasibility_flag = True

             # Crucial: Sanitize to update valences, aromaticity, etc.
             Chem.SanitizeMol(self.rdkit_mol)

        except Exception as e:
             print(f"WARNING: Error updating/sanitizing RDKit bond {rdkit_idx1}-{rdkit_idx2} order {new_order}: {e}")
             self.infeasibility_flag = True

    def take_action(self, action: int):
        """Execute a given action at the current action level (Revised)."""
        if self.synthesis_done:
            raise RuntimeError("Cannot take action on a terminated design.")
        if self.current_action_mask is None or action >= len(self.current_action_mask) or self.current_action_mask[action]:
            raise ValueError(f"Action {action} is masked or invalid for level {self.current_action_level}.")

        current_level = self.current_action_level
        next_level = 0
        self.history.append(int(action))

        if current_level == 0:
            if action == 0: # TERMINATE
                self.synthesis_done = True
                self.finalize()
                next_level = -1
            else: # SELECT_EXISTING_ATOM
                self.l0_selected_atom_idx = action
                self.is_modifying_atom = False
                self.atom_to_modify = None
                self.l1_new_atom_type = None
                self.l1_selected_existing_atom_idx = None
                next_level = 1

        elif current_level == 1:
            num_real_atoms = len(self.atoms) - 1
            modify_atom_action_idx = self.vocab_size + num_real_atoms

            if action < self.vocab_size: # INITIATE_ADD_NEW_ATOM
                self.l1_new_atom_type = action + 1
                self.atoms = np.append(self.atoms, self.l1_new_atom_type)
                self.bonds = np.pad(self.bonds, [(0, 1), (0, 1)], mode='constant', constant_values=0)
                new_atom_internal_idx = len(self.atoms) - 1
                self.bonds[0, new_atom_internal_idx] = self.bonds[new_atom_internal_idx, 0] = self.virtual_bond_idx
                self._add_atom_to_rdkit(self.l1_new_atom_type)
                self.update_topological_distance_matrix()
                self.is_modifying_atom = False
                next_level = 2
            elif action < modify_atom_action_idx: # SELECT_EXISTING_ATOM_FOR_BOND
                target_0_idx = action - self.vocab_size
                self.l1_selected_existing_atom_idx = target_0_idx + 1
                self.is_modifying_atom = False
                next_level = 2
            elif action == modify_atom_action_idx: # INITIATE_MODIFY_ATOM
                self.atom_to_modify = self.l0_selected_atom_idx
                self.is_modifying_atom = True
                next_level = 2
            else:
                 raise ValueError(f"Invalid action {action} received for Level 1.")

        elif current_level == 2:
            if self.is_modifying_atom:
                # Modify Path
                atom_internal_idx = self.atom_to_modify
                rdkit_idx_to_modify = atom_internal_idx - 1

                if action < self.vocab_size: # REPLACE_ATOM_TYPE
                    new_atom_type = action + 1
                    self.atoms[atom_internal_idx] = new_atom_type
                    self.replace_atom(atom_internal_idx, new_atom_type)
                elif action == self.REMOVE_ATOM_ACTION_L2_MODIFY: # REMOVE_ATOM
                    if rdkit_idx_to_modify < 0 or rdkit_idx_to_modify >= self.rdkit_mol.GetNumAtoms():
                         print(f"ERROR: Invalid RDKit index {rdkit_idx_to_modify} for atom removal.")
                         self.infeasibility_flag = True
                    else:
                         try:
                             self.rdkit_mol.RemoveAtom(rdkit_idx_to_modify)
                             self._sync_internal_state_from_rdkit()
                             self._check_and_update_connectivity()
                             self.update_topological_distance_matrix()
                         except Exception as e:
                             print(f"ERROR during atom removal or state sync: {e}")
                             self.infeasibility_flag = True
                else:
                     raise ValueError(f"Invalid action {action} for Level 2 Modify Path.")

                self.is_modifying_atom = False
                self.atom_to_modify = None
                next_level = 0

            else:
                # Bond Path
                atom_A_internal_idx = self.l0_selected_atom_idx
                atom_B_internal_idx = -1
                if self.l1_new_atom_type is not None:
                    atom_B_internal_idx = len(self.atoms) - 1
                elif self.l1_selected_existing_atom_idx is not None:
                    atom_B_internal_idx = self.l1_selected_existing_atom_idx
                else:
                     raise ValueError("L2 Bond path state inconsistent.")

                rdkit_idx_A = atom_A_internal_idx - 1
                rdkit_idx_B = atom_B_internal_idx - 1

                if action <= 5: # SET_BOND_ORDER (0..5 -> order 1..6)
                    new_bond_order = action + 1
                    self.bonds[atom_A_internal_idx, atom_B_internal_idx] = new_bond_order
                    self.bonds[atom_B_internal_idx, atom_A_internal_idx] = new_bond_order
                    self._update_rdkit_bond(rdkit_idx_A, rdkit_idx_B, new_bond_order)
                    self.update_topological_distance_matrix()
                    self._check_and_update_connectivity()
                elif action == 6: # REMOVE_BOND
                    self.bonds[atom_A_internal_idx, atom_B_internal_idx] = 0
                    self.bonds[atom_B_internal_idx, atom_A_internal_idx] = 0
                    self._update_rdkit_bond(rdkit_idx_A, rdkit_idx_B, 0)
                    self.update_topological_distance_matrix()
                    self._check_and_update_connectivity()
                else:
                     raise ValueError(f"Invalid action {action} for Level 2 Bond Path.")

                self.l1_new_atom_type = None
                self.l1_selected_existing_atom_idx = None
                next_level = 0

        # Transition
        if next_level != -1:
             self.current_action_level = next_level
             self.update_action_mask()
        else:
             self.current_action_mask = None

    def validate_atom_replacement(self, rdkit_atom_idx, new_atom_type, neighbor_bonds):
        """Checks if replacement seems possible based on valence."""
        atom_internal_idx = rdkit_atom_idx + 1
        # Ensure index is valid before accessing valence usage
        if rdkit_atom_idx < 0 or rdkit_atom_idx >= len(self.atoms) - 1:
            return False # Invalid index
        current_usage = self._get_current_valence_usage()[rdkit_atom_idx]
        if self.vocabulary_valence[new_atom_type] < current_usage:
             return False
        return True

    def replace_atom(self, atom_internal_idx: int, new_atom_type_idx: int):
        """Replace atom type in RDKit molecule and attempt sanitization."""
        rdkit_atom_idx = atom_internal_idx - 1
        if rdkit_atom_idx < 0 or rdkit_atom_idx >= self.rdkit_mol.GetNumAtoms():
             print(f"ERROR: Invalid rdkit index {rdkit_atom_idx} for replacement.")
             self.infeasibility_flag = True
             return
        try:
            atom = self.rdkit_mol.GetAtomWithIdx(rdkit_atom_idx)
            atom_config = self.atom_vocabulary[self.vocabulary_atom_names[new_atom_type_idx - 1]]
            atom.SetAtomicNum(atom_config["atomic_number"])
            atom.SetFormalCharge(atom_config.get("formal_charge", 0))
            chiral_tag_config = atom_config.get("chiral_tag")
            if chiral_tag_config == 1: atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
            elif chiral_tag_config == 2: atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
            else: atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            atom.UpdatePropertyCache(strict=False)
            for neighbor in atom.GetNeighbors():
                 neighbor.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(self.rdkit_mol)
        except Exception as e:
            print(f"WARNING: Sanitization failed after replacing atom {rdkit_atom_idx} with type {new_atom_type_idx}: {e}")
            self.infeasibility_flag = True

    def finalize(self, assert_feasible: bool = False):
        """Finalize molecule design, generate SMILES if valid."""
        if assert_feasible:
             try: self.assert_feasible()
             except AssertionError as e:
                 print(f"Feasibility assertion failed: {e}")
                 self.infeasibility_flag = True

        if not self.infeasibility_flag:
             try:
                 Chem.SanitizeMol(self.rdkit_mol)
                 self.smiles_string = Chem.MolToSmiles(self.rdkit_mol)
             except Exception as e:
                 print(f"Final sanitization/SMILES generation failed: {e}")
                 self.infeasibility_flag = True
                 self.smiles_string = None
        else:
             self.smiles_string = None

    def assert_feasible(self):
        """Check internal state consistency (Revised)."""
        assert self.atoms[0] == 0, "First atom should be virtual (0)"
        if len(self.atoms) > 1:
             assert np.all([not self.atom_feasibility_mask[x - 1] for x in self.atoms[1:]]), "Only allowed atoms permitted"
        assert self.upper_limit_atoms is None or len(self.atoms) - 1 <= self.upper_limit_atoms, "Max atoms exceeded"
        if len(self.atoms) > 1:
             assert np.all(self.bonds[0, 1:] == self.virtual_bond_idx), "Virtual atom connection missing"
             assert np.all(self.bonds[1:, 0] == self.virtual_bond_idx), "Virtual atom connection missing"
        assert not np.any(self.bonds.diagonal()), "Self-loops detected"
        assert not np.any(self.bonds - self.bonds.T), "Bond matrix not symmetric"
        if len(self.atoms) > 1:
             remaining_valence = self._get_remaining_valence()
             assert np.all(remaining_valence >= 0), f"Valence constraints violated. Remaining: {remaining_valence}"

    def to_rdkit_mol(self, sanitize=True) -> Chem.RWMol:
        """Creates an RDKit molecule from the internal state."""
        # This might be redundant if self.rdkit_mol is kept consistent
        # But can be useful for debugging or creating a clean copy
        mol = Chem.RWMol()
        if len(self.atoms) <= 1: return mol # Return empty mol if only virtual atom

        # Add atoms
        for atom_vocab_idx in self.atoms[1:]:
            atom_config = self.atom_vocabulary[self.vocabulary_atom_names[atom_vocab_idx - 1]]
            a = Chem.Atom(atom_config["atomic_number"])
            if "formal_charge" in atom_config: a.SetFormalCharge(atom_config["formal_charge"])
            ct = atom_config.get("chiral_tag")
            if ct == 1: a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
            elif ct == 2: a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
            else: a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            mol.AddAtom(a)

        # Add bonds
        real_bonds = self.bonds[1:, 1:]
        num_real_atoms = len(self.atoms) - 1
        for i in range(num_real_atoms):
            for j in range(i + 1, num_real_atoms): # Avoid double counting and self-loops
                bond_order = real_bonds[i, j]
                if bond_order > 0 and bond_order <= self.maximum_bond_order:
                    rdkit_bond_type = self.bond_types.get(bond_order)
                    if rdkit_bond_type:
                        mol.AddBond(i, j, rdkit_bond_type)

        if sanitize:
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                print(f"Sanitization failed in to_rdkit_mol: {e}")
                # Optionally return None or unsanitized mol depending on desired behavior
        return mol

    def is_terminable(self):
        """Checks if the current state allows termination."""
        # Terminable if at Level 0 and not already done and connected
        return self.current_action_level == 0 and not self.synthesis_done and self.is_currently_connected

    def to_smiles(self) -> Optional[str]:
        """Returns the SMILES string if finalized and valid."""
        return self.smiles_string

    @staticmethod
    def init_batch_from_instance_list(config: MoleculeConfig, instances: List[int], network: Optional[nn.Module]=None, device: Optional[torch.device]=None):
        # Network and device arguments are optional now
        return [MoleculeDesign(config=config, initial_atom=atom) for atom in instances]

    @staticmethod
    def log_probability_fn(trajectories: List['MoleculeDesign'], network: nn.Module) -> List[np.array]:
        """Calculates masked log probabilities for the current action level of each trajectory."""
        log_probs_to_return: List[np.array] = []
        network.eval()
        with torch.no_grad():
            batch = MoleculeDesign.list_to_batch(molecules=trajectories, device=network.device)
            # Assuming network outputs logits for L0, L1, L2 based on level_idx
            batch_logits_l0, batch_logits_l1, batch_logits_l2 = network(batch)

            batch_logits_l0 = batch_logits_l0.cpu().numpy()
            batch_logits_l1 = batch_logits_l1.cpu().numpy()
            batch_logits_l2 = batch_logits_l2.cpu().numpy()

            for i, mol in enumerate(trajectories):
                if mol.current_action_level == 0:
                    logits = batch_logits_l0[i]
                elif mol.current_action_level == 1:
                    logits = batch_logits_l1[i]
                elif mol.current_action_level == 2:
                    logits = batch_logits_l2[i]
                else: # Should not happen
                    log_probs_to_return.append(np.array([])) # Empty log probs if level invalid
                    continue

                mask = mol.current_action_mask
                if mask is None: # Already terminated
                     log_probs_to_return.append(np.array([]))
                     continue

                # Ensure logits match mask length before applying mask
                if len(logits) > len(mask):
                    logits = logits[:len(mask)]
                elif len(logits) < len(mask):
                     # This case indicates a problem - mask is larger than network output
                     print(f"WARNING: Logits length ({len(logits)}) < Mask length ({len(mask)}) for level {mol.current_action_level}. Masking based on logits length.")
                     mask = mask[:len(logits)]

                logits[mask] = -np.inf # Apply mask
                with np.errstate(divide='ignore', invalid='ignore'): # Ignore log(0) warnings
                    log_probs = logits - np.log(np.sum(np.exp(logits))) # Manual log_softmax
                log_probs[np.isneginf(logits)] = -np.inf # Ensure masked actions remain -inf
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
            raise ValueError("Objective is `None`. Evaluate molecule first.")
        # Return negative infinity if infeasible
        return float("-inf") if self.infeasibility_flag else self.objective

    def num_actions(self) -> int:
        """Returns the number of valid actions at the current level."""
        if self.current_action_mask is None: return 0
        return int(np.sum(~self.current_action_mask))

    @staticmethod
    def list_to_batch(molecules: List['MoleculeDesign'], device: torch.device = None,
                      include_feasibility_masks: bool = False) -> dict:
        """Converts a list of MoleculeDesign instances to a batch dictionary (Revised)."""
        atoms_padding_idx = molecules[0].vocab_size + 1
        degree_padding_idx = max(molecules[0].vocabulary_valence) + 2
        bond_padding_idx = MoleculeDesign.virtual_bond_idx + 1
        distance_padding_idx = MoleculeDesign.maximum_num_atoms_overall + 3

        device = torch.device("cpu") if device is None else device
        num_atoms = [len(mol.atoms) for mol in molecules]
        max_num_atoms = max(num_atoms) if num_atoms else 0

        batch_level_idx = [mol.current_action_level for mol in molecules]

        batch_picked_atom_mhe = np.zeros((len(molecules), max_num_atoms), dtype=int)
        for i, mol in enumerate(molecules):
            if mol.current_action_level >= 1 and mol.l0_selected_atom_idx is not None:
                 if mol.l0_selected_atom_idx < max_num_atoms:
                      batch_picked_atom_mhe[i, mol.l0_selected_atom_idx] = 1

        batch_atoms = np.stack([
            np.concatenate((mol.atoms, np.full(max_num_atoms - num_atoms[i], fill_value=atoms_padding_idx, dtype=int))) if num_atoms[i] > 0 else np.full(max_num_atoms, fill_value=atoms_padding_idx, dtype=int)
            for i, mol in enumerate(molecules)
        ])

        batch_atoms_degree = []
        for i, mol in enumerate(molecules):
             if num_atoms[i] > 0:
                  degree = (mol.bonds > 0).sum(axis=1) - 1
                  degree[0] = 0
                  padded_degree = np.concatenate((degree, np.full(max_num_atoms - num_atoms[i], fill_value=degree_padding_idx, dtype=int)))
             else:
                  padded_degree = np.full(max_num_atoms, fill_value=degree_padding_idx, dtype=int)
             batch_atoms_degree.append(padded_degree)
        batch_atoms_degree = np.stack(batch_atoms_degree)

        bonds_list = []
        for i, mol in enumerate(molecules):
            if num_atoms[i] > 0:
                 padded_bonds = np.pad(mol.bonds, [(0, max_num_atoms - num_atoms[i]), (0, max_num_atoms - num_atoms[i])], mode="constant", constant_values=bond_padding_idx)
                 np.fill_diagonal(padded_bonds, bond_padding_idx)
            else:
                 padded_bonds = np.full((max_num_atoms, max_num_atoms), fill_value=bond_padding_idx, dtype=int)
            bonds_list.append(padded_bonds)
        batch_bonds = np.stack(bonds_list)

        distance_matrices_list = []
        for i, mol in enumerate(molecules):
             if num_atoms[i] > 0:
                  padded_dist = np.pad(mol.topological_distance_matrix, [(0, max_num_atoms - num_atoms[i]), (0, max_num_atoms - num_atoms[i])], mode="constant", constant_values=distance_padding_idx)
             else:
                  padded_dist = np.full((max_num_atoms, max_num_atoms), fill_value=distance_padding_idx, dtype=int)
             distance_matrices_list.append(padded_dist)
        batch_topological_distance = np.stack(distance_matrices_list)

        additive_padding_masks = []
        for i, mol in enumerate(molecules):
             if num_atoms[i] > 0:
                  mask = np.zeros((num_atoms[i], num_atoms[i]), dtype=float)
                  padded_mask = np.pad(mask, [(0, max_num_atoms - num_atoms[i]), (0, max_num_atoms - num_atoms[i])], mode="constant", constant_values=-np.inf)
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

        if include_feasibility_masks:
            masks_l0, masks_l1, masks_l2 = [], [], []
            max_actions_l0, max_actions_l1, max_actions_l2 = 0, 0, 0

            for mol in molecules:
                 num_real = len(mol.atoms) - 1
                 max_actions_l0 = max(max_actions_l0, 1 + num_real)
                 max_actions_l1 = max(max_actions_l1, mol.vocab_size + num_real + 1)
                 if mol.is_modifying_atom: max_actions_l2 = max(max_actions_l2, mol.vocab_size + 1)
                 else: max_actions_l2 = max(max_actions_l2, 7)

            for mol in molecules:
                 num_real = len(mol.atoms) - 1
                 mask_l0 = mol.current_action_mask if mol.current_action_level == 0 and mol.current_action_mask is not None else np.ones(1 + num_real, dtype=bool)
                 padded_mask_l0 = np.pad(mask_l0, (0, max_actions_l0 - len(mask_l0)), mode='constant', constant_values=1)
                 masks_l0.append(padded_mask_l0)

                 mask_l1 = mol.current_action_mask if mol.current_action_level == 1 and mol.current_action_mask is not None else np.ones(mol.vocab_size + num_real + 1, dtype=bool)
                 padded_mask_l1 = np.pad(mask_l1, (0, max_actions_l1 - len(mask_l1)), mode='constant', constant_values=1)
                 masks_l1.append(padded_mask_l1)

                 if mol.current_action_level == 2 and mol.current_action_mask is not None:
                      mask_l2 = mol.current_action_mask
                      padded_mask_l2 = np.pad(mask_l2, (0, max_actions_l2 - len(mask_l2)), mode='constant', constant_values=1)
                 else:
                      padded_mask_l2 = np.ones(max_actions_l2, dtype=bool)
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

    @staticmethod
    def from_smiles(config: MoleculeConfig, smiles: str, do_finish=False, compare_smiles=False) -> 'MoleculeDesign':
        """Creates a MoleculeDesign instance by simulating actions from a SMILES string."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: raise ValueError(f"Invalid SMILES input: {smiles}")
        try:
             Chem.SanitizeMol(mol)
        except Exception as e:
             print(f"Warning: Input SMILES {smiles} failed sanitization: {e}")
        return MoleculeDesign.from_rdkit_mol(config, mol, smiles, do_finish, compare_smiles)

    @staticmethod
    def from_rdkit_mol(config: MoleculeConfig, rdkit_mol: Chem.Mol, smiles: Optional[str] = None,
                       do_finish: bool = True, compare_smiles: bool = False,
                       max_steps: int = 500) -> 'MoleculeDesign':
        """
        Creates a MoleculeDesign instance by simulating actions from an RDKit molecule,
        handling implicit hydrogens by working with the heavy-atom graph.
        """
        # --- Input Validation and Preparation ---
        if not isinstance(rdkit_mol, Chem.Mol):
             raise TypeError("Input rdkit_mol must be an RDKit Mol object.")
        if rdkit_mol.GetNumAtoms() == 0:
            raise ValueError("Cannot create from empty RDKit molecule.")

        # Work on a copy
        mol_copy = Chem.Mol(rdkit_mol)

        # --- IMPLICIT HYDROGEN: Remove explicit H ---
        try:
            # Preserve charges/isotopes etc. during H removal
            mol_copy_noH = Chem.RemoveHs(mol_copy, sanitize=False, implicitOnly=False)
            print(f"DEBUG from_rdkit_mol: Removed H, atoms before: {mol_copy.GetNumAtoms()}, after: {mol_copy_noH.GetNumAtoms()}")
            if mol_copy_noH.GetNumAtoms() == 0:
                raise ValueError("Molecule contains no heavy atoms after H removal.")
        except Exception as e:
             print(f"ERROR: Chem.RemoveHs failed: {e}. Cannot proceed.")
             raise ValueError(f"Failed to remove hydrogens from input molecule: {e}")

        # Attempt Kekulization on the heavy-atom graph
        try:
            Chem.Kekulize(mol_copy_noH, clearAromaticFlags=True)
        except Exception as e:
            # Don't fail, but warn. Vocab might handle aromaticity implicitly.
            print(f"Warning: Kekulization failed for heavy-atom graph: {e}")
        # --- END IMPLICIT HYDROGEN ---

        heavy_atoms = mol_copy_noH.GetAtoms()
        num_heavy_atoms = len(heavy_atoms)

        # --- Map RDKit Heavy Atoms to Vocabulary ---
        prop_to_vocab_idx = {}
        vocab_names = list(config.atom_vocabulary.keys())
        for i, name in enumerate(vocab_names):
            cfg = config.atom_vocabulary[name]
            # Create unique key: atomic_num_charge_chiral
            key_parts = [str(cfg["atomic_number"])]
            key_parts.append(str(cfg.get("formal_charge", 0)))
            key_parts.append(str(cfg.get("chiral_tag", 0))) # Uses 0, 1, 2 from config
            prop_to_vocab_idx["_".join(key_parts)] = i + 1

        atom_vocab_indices = [] # 1-based vocab indices for each heavy atom
        rdkit_idx_map = {} # Map original RDKit index (in mol_copy_noH) to 0-based index in heavy_atoms list
        for idx, atom in enumerate(heavy_atoms):
            rdkit_idx_map[atom.GetIdx()] = idx # Store mapping

            key_parts = [str(atom.GetAtomicNum())]
            fc = atom.GetFormalCharge()
            ct = atom.GetChiralTag() # RDKit enum
            key_parts.append(str(fc))
            # Map RDKit ChiralType enum to 0, 1, 2 used in config
            chiral_tag_int = 0
            if ct == Chem.ChiralType.CHI_TETRAHEDRAL_CW: chiral_tag_int = 1
            elif ct == Chem.ChiralType.CHI_TETRAHEDRAL_CCW: chiral_tag_int = 2
            key_parts.append(str(chiral_tag_int))
            key = "_".join(key_parts)

            vocab_idx = prop_to_vocab_idx.get(key)
            # Fallback: Try matching without chiral tag if exact match failed
            if vocab_idx is None:
                key_parts_no_chiral = [str(atom.GetAtomicNum()), str(fc), '0']
                key_no_chiral = "_".join(key_parts_no_chiral)
                vocab_idx = prop_to_vocab_idx.get(key_no_chiral)
                if vocab_idx is not None:
                    print(f"DEBUG from_rdkit_mol: Atom {atom.GetIdx()} matched without chiral tag (key: {key_no_chiral})")

            if vocab_idx is None or not config.atom_vocabulary[vocab_names[vocab_idx-1]]["allowed"]:
                 final_key_tried = key if prop_to_vocab_idx.get(key) is None else key_no_chiral
                 raise ValueError(f"Heavy atom {atom.GetIdx()} (Props: {final_key_tried}) cannot be mapped to an allowed vocabulary entry.")
            atom_vocab_indices.append(vocab_idx)

        # --- Initialize MoleculeDesign and Adjacency for Heavy Atoms ---
        design = MoleculeDesign(config, atom_vocab_indices[0]) # Start with the first heavy atom
        # Map: RDKit heavy atom index (in mol_copy_noH) -> internal MoleculeDesign index (1-based)
        placed_atoms = {heavy_atoms[0].GetIdx(): 1}

        # Build adjacency matrix with bond orders between heavy atoms
        bond_type_to_order = {
            Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3,
            Chem.BondType.QUADRUPLE: 4, Chem.BondType.QUINTUPLE: 5, Chem.BondType.HEXTUPLE: 6,
            Chem.BondType.AROMATIC: 1, # Treat aromatic as single after Kekulization attempt
        }
        # Use original RDKit indices from mol_copy_noH for adjacency keys
        adjacency_orders = {} # Using dict: (min_idx, max_idx) -> order
        for bond in mol_copy_noH.GetBonds():
            i_rdkit, j_rdkit = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            order = bond_type_to_order.get(bond.GetBondType(), 0)
            if order > 0:
                key = tuple(sorted((i_rdkit, j_rdkit)))
                adjacency_orders[key] = order
            else:
                print(f"Warning: Unknown bond type {bond.GetBondType()} between heavy atoms {i_rdkit}-{j_rdkit}. Treating as 0.")


        # --- Simulate Building Process (Heavy Atoms Only) ---
        vocab_size = len(config.atom_vocabulary)
        steps_taken = 0
        for i_heavy_idx in range(1, num_heavy_atoms): # Iterate through remaining heavy atoms
            if steps_taken > max_steps:
                print(f"Warning: Exceeded max_steps ({max_steps}) during construction from RDKit Mol.")
                design.infeasibility_flag = True
                break

            current_heavy_atom_rdkit_idx = heavy_atoms[i_heavy_idx].GetIdx()
            atom_to_add_vocab_idx = atom_vocab_indices[i_heavy_idx]
            atom_to_add_l1_action = atom_to_add_vocab_idx - 1 # 0-based for action

            connection_found = False
            # Find an already placed heavy atom (j) connected to the current heavy atom (i)
            for j_heavy_idx in range(i_heavy_idx):
                anchor_heavy_atom_rdkit_idx = heavy_atoms[j_heavy_idx].GetIdx()
                bond_key = tuple(sorted((current_heavy_atom_rdkit_idx, anchor_heavy_atom_rdkit_idx)))
                bond_order = adjacency_orders.get(bond_key, 0)

                if bond_order > 0 and anchor_heavy_atom_rdkit_idx in placed_atoms:
                    anchor_internal_idx = placed_atoms[anchor_heavy_atom_rdkit_idx]

                    # --- Action Sequence: Add New Heavy Atom and First Bond ---
                    print(f"\nDEBUG LOOP (Heavy Atom Idx {i_heavy_idx}, RDKit {current_heavy_atom_rdkit_idx}): Connecting to Anchor RDKit {anchor_heavy_atom_rdkit_idx} (Internal {anchor_internal_idx})")
                    try:
                        # L0: Select anchor atom
                        print(f"  Action L0: Select {anchor_internal_idx}")
                        design.take_action(anchor_internal_idx); steps_taken += 1
                        # L1: Add New Atom type
                        print(f"  Action L1: Add type {atom_to_add_l1_action+1} ({vocab_names[atom_to_add_l1_action]})")
                        design.take_action(atom_to_add_l1_action); steps_taken += 1
                        # L2: Set Bond order between anchor and new atom
                        l2_bond_action = bond_order - 1 # 0-based for action
                        print(f"  Action L2: Set bond order {l2_bond_action+1}")
                        design.take_action(l2_bond_action); steps_taken += 1
                        # Check for immediate infeasibility after action
                        if design.infeasibility_flag:
                            raise ValueError("Action resulted in infeasible state.")
                    except ValueError as e:
                         raise ValueError(f"Error taking action during construction for heavy atom RDKitIdx={current_heavy_atom_rdkit_idx} connected to {anchor_heavy_atom_rdkit_idx}: {e}")

                    # Record the internal index of the newly added heavy atom
                    new_atom_internal_idx = len(design.atoms) - 1
                    placed_atoms[current_heavy_atom_rdkit_idx] = new_atom_internal_idx
                    connection_found = True
                    print(f"  Added: RDKit Idx {current_heavy_atom_rdkit_idx} -> Internal Idx {new_atom_internal_idx}")

                    # --- Add Bonds to Other Previously Placed Heavy Atoms ---
                    for k_heavy_idx in range(i_heavy_idx):
                        if k_heavy_idx == j_heavy_idx: continue # Skip the anchor atom
                        other_heavy_atom_rdkit_idx = heavy_atoms[k_heavy_idx].GetIdx()
                        extra_bond_key = tuple(sorted((current_heavy_atom_rdkit_idx, other_heavy_atom_rdkit_idx)))
                        bond_order_extra = adjacency_orders.get(extra_bond_key, 0)

                        if bond_order_extra > 0 and other_heavy_atom_rdkit_idx in placed_atoms:
                            other_placed_internal_idx = placed_atoms[other_heavy_atom_rdkit_idx]
                            print(f"\nDEBUG LOOP (Heavy Atom Idx {i_heavy_idx}, RDKit {current_heavy_atom_rdkit_idx}): Adding extra bond to RDKit {other_heavy_atom_rdkit_idx} (Internal {other_placed_internal_idx})")
                            try:
                                # L0: Select the *other* placed atom (k)
                                print(f"  Action L0: Select {other_placed_internal_idx}")
                                design.take_action(other_placed_internal_idx); steps_taken += 1
                                # L1: Select the *newly added* atom (i)
                                l1_select_action = vocab_size + (new_atom_internal_idx - 1) # 0-based relative index
                                print(f"  Action L1: Select existing {new_atom_internal_idx}")
                                design.take_action(l1_select_action); steps_taken += 1
                                # L2: Set Bond order between k and i
                                l2_bond_action_extra = bond_order_extra - 1 # 0-based for action
                                print(f"  Action L2: Set bond order {l2_bond_action_extra+1}")
                                design.take_action(l2_bond_action_extra); steps_taken += 1
                                if design.infeasibility_flag:
                                     raise ValueError("Action resulted in infeasible state.")
                            except ValueError as e:
                                raise ValueError(f"Error taking action during construction for extra bond {other_heavy_atom_rdkit_idx}-{current_heavy_atom_rdkit_idx}: {e}")
                    # Once the first connection is handled and extra bonds are added, move to the next heavy atom i
                    break # Break from inner loop (j)

            if not connection_found:
                 # Should not happen for connected heavy-atom graphs
                 raise ValueError(f"Could not find connection for heavy atom RDKitIdx={current_heavy_atom_rdkit_idx} to previously placed atoms.")
            if design.infeasibility_flag:
                print(f"Warning: Construction became infeasible at heavy atom RDKitIdx={current_heavy_atom_rdkit_idx}.")
                break # Stop if an action failed

        # --- Finalization and State Sync ---
        if not design.infeasibility_flag and do_finish:
            try:
                if design.is_terminable():
                     print("DEBUG from_rdkit_mol: Attempting final terminate action.")
                     design.take_action(0); steps_taken += 1
                else:
                     print("WARNING: Molecule cannot be terminated after construction (heavy atom graph).")
                     design.finalize(assert_feasible=False)
            except ValueError as e:
                 print(f"WARNING: Error during final terminate action: {e}")
                 design.finalize(assert_feasible=False)

        # <<< --- ENSURE FINAL STATE SYNCHRONIZATION --- >>>
        # This remains crucial to ensure internal arrays match the state achieved by take_action calls.
        print("DEBUG from_rdkit_mol: Performing final sync...")
        try:
            design._sync_internal_state_from_rdkit() # Syncs internal arrays from design.rdkit_mol
            design._check_and_update_connectivity()  # Updates connectivity flag based on synced state
            design.update_action_mask()             # Updates mask based on synced state and level
            print("DEBUG from_rdkit_mol: Final sync complete.")
        except Exception as e:
            print(f"ERROR during final sync in from_rdkit_mol: {e}")
            design.infeasibility_flag = True
            design.synthesis_done = True
            design.current_action_mask = None

        # --- Optional: Final SMILES Comparison ---
        # Compare against the original SMILES if provided, using heavy-atom graph from internal state
        if compare_smiles and smiles is not None:
            final_smiles_heavy = None
            try:
                # Get RDKit mol from the *synced* internal state (should be heavy atoms only)
                final_mol = design.get_rdkit_mol()
                if final_mol:
                    # Add Hs back for canonical comparison? Or compare heavy atom SMILES?
                    # Let's compare heavy atom SMILES first for simplicity
                    # Chem.SanitizeMol(final_mol) # Might fail if valence is weird before adding H
                    final_smiles_heavy = Chem.MolToSmiles(final_mol)

                    # Compare heavy atom SMILES (less robust but direct)
                    ref_mol_noH = Chem.RemoveHs(Chem.MolFromSmiles(smiles))
                    ref_smiles_heavy = Chem.MolToSmiles(ref_mol_noH)

                    if Chem.CanonSmiles(final_smiles_heavy) != Chem.CanonSmiles(ref_smiles_heavy):
                         print(f"WARNING: Heavy atom SMILES mismatch after construction")
                         print(f"  Constructed (Heavy): {final_smiles_heavy} -> Canon: {Chem.CanonSmiles(final_smiles_heavy)}")
                         print(f"  Reference   (Heavy): {ref_smiles_heavy} -> Canon: {Chem.CanonSmiles(ref_smiles_heavy)}")
            except Exception as smi_err:
                 print(f"Warning: Error during final SMILES comparison: {smi_err}")
                 pass

        return design