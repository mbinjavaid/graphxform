import copy
import random
import numpy as np
import torch
from torch import nn
from rdkit import Chem, RDLogger
import networkx as nx # Import NetworkX

from config import MoleculeConfig
from core.abstracts import BaseTrajectory
from core.utils import softmax

from typing import Optional, List, Tuple

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

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

            for target_0_idx in range(num_real_atoms):
                target_internal_idx = target_0_idx + 1
                action_idx = self.vocab_size + target_0_idx
                if target_internal_idx == anchor_atom_internal_idx: continue
                bond_exists = self.bonds[anchor_atom_internal_idx, target_internal_idx] > 0
                target_has_valence = remaining_valence[target_0_idx] > 0
                if bond_exists or target_has_valence:
                    mask[action_idx] = False

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

    def take_action(self, action: int):
        """Execute a given action, updating internal state directly."""
        if self.synthesis_done: raise RuntimeError("Cannot take action on terminated design.")
        if self.current_action_mask is None or action >= len(self.current_action_mask) or self.current_action_mask[action]:
            raise ValueError(f"Action {action} masked/invalid for level {self.current_action_level}.")

        current_level = self.current_action_level
        next_level = 0
        self.history.append(int(action))
        num_real_atoms_before = len(self.atoms) - 1

        try:
            atom_removed = False
            if current_level == 0:
                if action == 0: self.synthesis_done = True; self.finalize(); next_level = -1  # finalize now builds local rdkit_mol
                else:
                    self.l0_selected_atom_idx = action
                    self.is_modifying_atom=False; self.atom_to_modify=None; self.l1_new_atom_type=None; self.l1_selected_existing_atom_idx=None
                    next_level = 1
            elif current_level == 1:
                modify_idx = self.vocab_size + num_real_atoms_before
                if action < self.vocab_size: # Add Atom
                    self.l1_new_atom_type = action + 1
                    self.atoms = np.append(self.atoms, self.l1_new_atom_type)
                    new_size = len(self.atoms); new_idx = new_size - 1
                    self.bonds = np.pad(self.bonds, [(0,1),(0,1)], 'constant', constant_values=0)
                    self.bonds[0, new_idx] = self.bonds[new_idx, 0] = self.virtual_bond_idx
                    self.is_modifying_atom = False; next_level = 2
                elif action < modify_idx: # Select Existing
                    self.l1_selected_existing_atom_idx = (action - self.vocab_size) + 1
                    self.is_modifying_atom = False; next_level = 2
                elif action == modify_idx: # Initiate Modify
                    self.atom_to_modify = self.l0_selected_atom_idx
                    self.is_modifying_atom = True; next_level = 2
                else: raise ValueError("Invalid L1 action")
            elif current_level == 2:
                if self.is_modifying_atom: # Modify Path
                    mod_idx = self.atom_to_modify
                    if mod_idx is None: raise ValueError("atom_to_modify not set")
                    if action < self.vocab_size: # Replace Type
                        self.atoms[mod_idx] = action + 1
                    elif action == self.REMOVE_ATOM_ACTION_L2_MODIFY: # Remove Atom
                        self.atoms = np.delete(self.atoms, mod_idx)
                        self.bonds = np.delete(np.delete(self.bonds, mod_idx, 0), mod_idx, 1)
                        self._adjust_indices_after_removal(mod_idx); atom_removed = True
                    else: raise ValueError("Invalid L2 Modify action")
                    self.is_modifying_atom=False; self.atom_to_modify=None; next_level = 0
                else: # Bond Path
                    idx_A = self.l0_selected_atom_idx
                    idx_B = len(self.atoms) - 1 if self.l1_new_atom_type is not None else self.l1_selected_existing_atom_idx
                    if idx_A is None or idx_B is None: raise ValueError("L2 Bond indices missing")
                    if action <= 5: # Set Order
                        order = action + 1
                        self.bonds[idx_A, idx_B] = self.bonds[idx_B, idx_A] = order
                    elif action == 6: # Remove Bond
                        self.bonds[idx_A, idx_B] = self.bonds[idx_B, idx_A] = 0
                    else: raise ValueError("Invalid L2 Bond action")
                    self.l1_new_atom_type=None; self.l1_selected_existing_atom_idx=None; next_level = 0

            if next_level != -1:
                 self._check_and_update_connectivity() # Use NetworkX version
                 # --- No topological matrix update ---
                 self.current_action_level = next_level
                 self.update_action_mask()
            else: self.current_action_mask = None
        except Exception as e:
             print(f"FATAL ERROR take_action(action={action}, L{current_level}): {e}")
             import traceback; traceback.print_exc()
             self.infeasibility_flag = True; self.synthesis_done = True; self.current_action_mask = None


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

    # --- from_smiles / from_rdkit_mol (Keep Simulation Logic) ---
    @staticmethod
    def from_smiles(config: MoleculeConfig, smiles: str, do_finish=True, compare_smiles=False, max_steps=500) -> 'MoleculeDesign':
        """Creates a MoleculeDesign instance by simulating actions from a SMILES string."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: raise ValueError(f"Invalid SMILES input: {smiles}")
        try: Chem.SanitizeMol(mol, catchErrors=True)
        except Exception as e: print(f"Warning: Input SMILES {smiles} failed initial sanitization: {e}")
        return MoleculeDesign.from_rdkit_mol(config, mol, smiles, do_finish, compare_smiles, max_steps)

    @staticmethod
    def from_rdkit_mol(config: MoleculeConfig, rdkit_mol: Chem.Mol, smiles: Optional[str] = None,
                       do_finish: bool = True, compare_smiles: bool = False,
                       max_steps: int = 500) -> 'MoleculeDesign':
        """
        Creates a MoleculeDesign instance by simulating actions from an RDKit molecule (v2025-04-20 Sim Debug No RDKit).
        Attempts to build the molecule step-by-step using take_action, minimizing RDKit use during simulation.
        """
        # print(f"\n--- DEBUG: Entering from_rdkit_mol (Simulation) for SMILES: {smiles} ---") # DEBUG
        if not isinstance(rdkit_mol, Chem.Mol): raise TypeError("Input rdkit_mol must be an RDKit Mol object.")

        # Preprocessing (same)
        try:
            mol_copy = Chem.RemoveHs(rdkit_mol, sanitize=False)
            if mol_copy.GetNumAtoms() == 0: raise ValueError("Input molecule has no heavy atoms.")
        except Exception as e: raise ValueError(f"Failed to remove hydrogens: {e}")
        try: Chem.Kekulize(mol_copy, clearAromaticFlags=True)
        except Exception as e: print(f"Warning: Kekulization failed: {e}")
        target_atoms = mol_copy.GetAtoms(); num_target_atoms = len(target_atoms)
        if num_target_atoms == 0: raise ValueError("Input molecule has no heavy atoms after processing.")

        # Map Target Atoms (same)
        prop_to_vocab_idx = {}; vocab_names = list(config.atom_vocabulary.keys())
        for i, name in enumerate(vocab_names):
            cfg = config.atom_vocabulary[name]; key = f"{cfg['atomic_number']}_{cfg.get('formal_charge', 0)}_{cfg.get('chiral_tag', 0)}"
            prop_to_vocab_idx[key] = i + 1
        target_atom_vocab_indices = []; target_rdkit_indices = [atom.GetIdx() for atom in target_atoms]
        for atom in target_atoms:
            key_parts = [str(atom.GetAtomicNum()), str(atom.GetFormalCharge())]; ct = atom.GetChiralTag(); chiral_tag_int = 0
            if ct == Chem.ChiralType.CHI_TETRAHEDRAL_CW: chiral_tag_int = 1
            elif ct == Chem.ChiralType.CHI_TETRAHEDRAL_CCW: chiral_tag_int = 2
            key_parts.append(str(chiral_tag_int)); key = "_".join(key_parts)
            vocab_idx = prop_to_vocab_idx.get(key)
            if vocab_idx is None and chiral_tag_int != 0: key_no_chiral = f"{atom.GetAtomicNum()}_{atom.GetFormalCharge()}_0"; vocab_idx = prop_to_vocab_idx.get(key_no_chiral)
            if vocab_idx is None or not config.atom_vocabulary[vocab_names[vocab_idx-1]]["allowed"]: raise ValueError(f"Target atom {atom.GetIdx()} cannot be mapped.")
            target_atom_vocab_indices.append(vocab_idx)

        # Build Target Adjacency (same)
        bond_type_to_order = { Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3, Chem.BondType.QUADRUPLE: 4, Chem.BondType.QUINTUPLE: 5, Chem.BondType.HEXTUPLE: 6, Chem.BondType.AROMATIC: 1 }
        target_adjacency_orders = {}
        for bond in mol_copy.GetBonds():
            i_rdkit, j_rdkit = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(); order = bond_type_to_order.get(bond.GetBondType(), 0)
            if order > 0: target_adjacency_orders[tuple(sorted((i_rdkit, j_rdkit)))] = order

        # Initialize Simulation
        initial_atom_vocab_idx = target_atom_vocab_indices[0]
        design = MoleculeDesign(config, initial_atom_vocab_idx)
        rdkit_idx_to_internal_idx = {target_rdkit_indices[0]: 1}
        steps_taken = 0
        # print(f"  DEBUG from_rdkit_mol: Initialized with atom {initial_atom_vocab_idx}. Map: {rdkit_idx_to_internal_idx}") # DEBUG

        # Simulate Building Process
        for i in range(1, num_target_atoms):
            if steps_taken > max_steps: print(f"Warning: Exceeded max_steps"); design.infeasibility_flag = True; break
            if design.infeasibility_flag: break
            current_target_rdkit_idx = target_rdkit_indices[i]
            atom_to_add_vocab_idx = target_atom_vocab_indices[i]
            atom_to_add_l1_action = atom_to_add_vocab_idx - 1
            # print(f"\n  DEBUG: Processing target atom i={i} (RDKit={current_target_rdkit_idx}, Type={atom_to_add_vocab_idx})") # DEBUG
            connection_found = False
            for j in range(i):
                anchor_target_rdkit_idx = target_rdkit_indices[j]
                bond_key = tuple(sorted((current_target_rdkit_idx, anchor_target_rdkit_idx)))
                bond_order = target_adjacency_orders.get(bond_key, 0)
                if bond_order > 0 and anchor_target_rdkit_idx in rdkit_idx_to_internal_idx:
                    anchor_internal_idx = rdkit_idx_to_internal_idx[anchor_target_rdkit_idx]
                    connection_found = True
                    # print(f"    DEBUG: Connect to anchor j={j} (Internal={anchor_internal_idx}), Order={bond_order}") # DEBUG
                    action_seq = [anchor_internal_idx, atom_to_add_l1_action, bond_order - 1]
                    level_seq = [0, 1, 2]
                    # print(f"    DEBUG: Seq Add+Bond: {action_seq}") # DEBUG
                    try:
                        for idx, action in enumerate(action_seq):
                             level = level_seq[idx]
                             # print(f"      DEBUG: Try action {action} L{level} (Expect {design.current_action_level})") # DEBUG
                             if design.current_action_level != level: raise RuntimeError(f"Level mismatch {level} vs {design.current_action_level}")
                             mask = design.current_action_mask
                             if mask is None or action >= len(mask) or mask[action]: raise ValueError(f"Action {action} masked L{level}")
                             design.take_action(action); steps_taken += 1
                             # print(f"      DEBUG: OK. New L{design.current_action_level}") # DEBUG
                    except Exception as e: print(f"    DEBUG ERROR Add+Bond: {e}"); design.infeasibility_flag = True; break
                    if design.infeasibility_flag: break
                    new_atom_internal_idx = len(design.atoms) - 1
                    rdkit_idx_to_internal_idx[current_target_rdkit_idx] = new_atom_internal_idx
                    # print(f"    DEBUG: Added map {current_target_rdkit_idx} -> {new_atom_internal_idx}") # DEBUG
                    for k in range(i):
                        if k == j: continue
                        if design.infeasibility_flag: break
                        other_target_rdkit_idx = target_rdkit_indices[k]
                        extra_bond_key = tuple(sorted((current_target_rdkit_idx, other_target_rdkit_idx)))
                        extra_bond_order = target_adjacency_orders.get(extra_bond_key, 0)
                        if extra_bond_order > 0 and other_target_rdkit_idx in rdkit_idx_to_internal_idx:
                            other_internal_idx = rdkit_idx_to_internal_idx[other_target_rdkit_idx]
                            # print(f"    DEBUG: Extra bond to k={k} (Internal={other_internal_idx}), Order={extra_bond_order}") # DEBUG
                            l1_select_action = design.vocab_size + (new_atom_internal_idx - 1)
                            extra_bond_seq = [other_internal_idx, l1_select_action, extra_bond_order - 1]
                            level_seq_extra = [0, 1, 2]
                            # print(f"    DEBUG: Seq ExtraBond: {extra_bond_seq}") # DEBUG
                            try:
                                for idx, action in enumerate(extra_bond_seq):
                                     level = level_seq_extra[idx]
                                     # print(f"      DEBUG: Try action {action} L{level} (Expect {design.current_action_level})") # DEBUG
                                     if design.current_action_level != level: raise RuntimeError(f"Level mismatch {level} vs {design.current_action_level}")
                                     mask = design.current_action_mask
                                     if mask is None or action >= len(mask) or mask[action]: raise ValueError(f"Action {action} masked L{level}")
                                     design.take_action(action); steps_taken += 1
                                     # print(f"      DEBUG: OK. New L{design.current_action_level}") # DEBUG
                            except Exception as e: print(f"    DEBUG ERROR ExtraBond: {e}"); design.infeasibility_flag = True; break
                    break # Break j loop
            if not connection_found and num_target_atoms > 1: print(f"ERROR: Disconnected target RDKit={current_target_rdkit_idx}."); design.infeasibility_flag = True; break

        # Finalization
        # print(f"--- DEBUG: Finished simulation for {smiles}. Steps={steps_taken}, Infeasible={design.infeasibility_flag} ---") # DEBUG
        if not design.infeasibility_flag and do_finish:
            try:
                if design.is_terminable():
                     mask0 = design.current_action_mask
                     if mask0 is not None and not mask0[0]: design.take_action(0); steps_taken += 1
                     else: print(f"  DEBUG WARNING: Final terminate masked.")
                else: print("WARNING: Not terminable after construction."); design.finalize(assert_feasible=False)
            except ValueError as e: print(f"WARNING: Error during final terminate: {e}"); design.finalize(assert_feasible=False)
        elif design.infeasibility_flag: print("Skipping final terminate (infeasible)."); design.finalize(assert_feasible=False)
        else: design.finalize(assert_feasible=False)

        # SMILES Comparison (optional, keep as is)
        if compare_smiles and smiles is not None and not design.infeasibility_flag:
             try:
                 final_mol_internal = design.to_rdkit_mol(sanitize=True); final_smiles_internal = Chem.MolToSmiles(final_mol_internal) if final_mol_internal else None
                 ref_mol_orig_noH = Chem.RemoveHs(Chem.MolFromSmiles(smiles)); ref_smiles_canon = Chem.MolToSmiles(ref_mol_orig_noH) if ref_mol_orig_noH else None
                 if final_smiles_internal and ref_smiles_canon and Chem.CanonSmiles(final_smiles_internal) != Chem.CanonSmiles(ref_smiles_canon):
                      print(f"WARNING: SMILES mismatch"); print(f"  Constructed: {final_smiles_internal} -> {Chem.CanonSmiles(final_smiles_internal)}"); print(f"  Reference:   {smiles} -> {Chem.CanonSmiles(ref_smiles_canon)}")
             except Exception as smi_err: print(f"Warning: Error during SMILES compare: {smi_err}")

        # print(f"--- DEBUG: Exiting from_rdkit_mol for {smiles}. History Len: {len(design.history)} ---") # DEBUG
        return design
