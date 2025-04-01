import copy
import random
import numpy as np
import torch
from torch import nn
from rdkit import Chem

from config import MoleculeConfig
from core.abstracts import BaseTrajectory
from core.utils import softmax

from typing import Optional, List, Tuple


class MoleculeDesign(BaseTrajectory):
    """
    Environment for the molecular design.
    Actions are chosen hierarchically in three levels.
        - Level 0: Terminate or pick a first atom.
            - Choose to terminate (index 0)
            - Create a new atom and pick that (index 1 up to (length of vocabulary))
            - Pick an existing atom (index (length of vocabulary + 1) up to (length of vocabulary + 1 + number of atoms)
        - Level 1: If not terminating, pick a second atom on which a bond decision must be made. (index 0 up to number of atoms)
            - Special action at the end allows replacing an existing atom
        - Level 2:
            - Normal mode: Set bond order (index 0-5) or remove bond (index 6)
            - Replace mode: Pick new atom type to replace selected atom with

    Level 0 and 1 are predicted simultaneously by the network, while for level 2 we mark the chosen atom for the network.

    Atom types are specified in the config under `atom_vocabulary`. Indexing starts at 1. Index 0 is for a virtual atom.
    - Index 0: Virtual Atom, which is connected (with special bond order) to every other atom (and vice versa).

    We store all actions in a history, which is a list of indices indicating how you get from the initial atom to the current
    molecule. For example, with a vocabulary of [C, N, O], and starting from the atom C, the action history
    [1, 4, 1, 0] means that we add a C atom (1), connect it to the existing C atom (4), with a bond order of 2 (1) and
    then terminate (0), resulting in C=C.
    """
    maximum_bond_order = 6
    virtual_bond_idx = 7  # index for the virtual bond between virtual atom and other atoms. Is one more than the maximum bond order possible.
    maximum_num_atoms_overall = 100
    bond_types = {
        1: Chem.rdchem.BondType.SINGLE,
        2: Chem.rdchem.BondType.DOUBLE,
        3: Chem.rdchem.BondType.TRIPLE,
        4: Chem.rdchem.BondType.QUADRUPLE,
        5: Chem.rdchem.BondType.QUINTUPLE,
        6: Chem.rdchem.BondType.HEXTUPLE
    }

    REPLACE_ACTION = "replace_atom"  # Special action identifier for atom replacement

    def __init__(self, config: MoleculeConfig, initial_atom: int):
        """
        Parameters:
            config [MoleculeConfig]: Config
            initial_atom [int]: We always start with already one atom in the molecule to be able to diversify
                the starting point for the network.
        """
        self.config = config
        self.atom_vocabulary = self.config.atom_vocabulary
        self.vocabulary_atom_idcs = list(range(1, len(self.atom_vocabulary) + 1))  # [1, ..., num of atoms in vocab]
        self.vocabulary_atom_names = list(self.atom_vocabulary.keys())
        self.vocabulary_valence = [-1] + [self.atom_vocabulary[x]["valence"] for x in self.vocabulary_atom_names]  # have an entry "-1" for the first virtual atom
        self.atom_feasibility_mask = [not self.atom_vocabulary[x]["allowed"] for x in self.vocabulary_atom_names]  # if not allowed, then feasibility mask must be set to True

        # Extract relevant indexing information that depends on the size of the vocabulary.
        self.pick_existing_atoms_start_action_idx_lvl_0 = len(self.vocabulary_atom_idcs) + 1  # Level 0, where does (after terminate and create new atom) the indexing of the existing atoms start?

        self.upper_limit_atoms = self.config.max_num_atoms
        assert not self.atom_feasibility_mask[initial_atom - 1] and initial_atom in self.vocabulary_atom_idcs, f"Initial atom must be in {self.vocabulary_atom_idcs} and set to allowed in config."
        self.initial_atom = initial_atom

        # Keeps track of all atoms present (including virtual atom)
        self.atoms = np.array([0, initial_atom], dtype=np.uint8)

        # Keeps track of the design as an RDKit molecule
        self.rdkit_mol = Chem.RWMol()

        # Keeps track of all bonds with order. Is a matrix of shape (len(atoms), len(atoms)), where the (i,j)-th entry
        # indicates connection of i-th atom with j-th atom. Note that the virtual atom has a bond of special order with
        # all other atoms.
        self.bonds = np.zeros((2, 2), dtype=np.uint8)
        self.bonds[0, 1] = self.bonds[1, 0] = self.virtual_bond_idx  # connect with virtual atom
        # The topological distance matrix keeps the shortest path between any two atoms. We set a special distance
        # for the distance between virtual atom and any other atom, and also for an atom that is not yet connected
        self.virtual_distance = self.maximum_num_atoms_overall + 1  # for distance between virtual to any atom
        self.infinity_distance = self.maximum_num_atoms_overall + 2 # for distance between new atom (not bonded yet) to any atom
        self.topological_distance_matrix = np.array([[0, self.virtual_distance], [self.virtual_distance, 0]], dtype=np.uint8)

        # Additional tracking variables for atom replacement
        self.is_replacing_atom = False  # Flag to indicate we're in replacement mode
        self.atom_to_replace = None  # Index of atom being replaced

        self.synthesis_done = False
        self.smiles_string: Optional[str] = None  # Is set after synthesis is done
        self.current_objective = float("-inf")

        # Current action level. Can be 0, 1, 2
        self.current_action_level = 0  # start by choosing <terminate>/<create new atom and pick>/<pick existing atom>

        # The action mask indicates before each action what is feasible at the current level.
        # It is set for each level when transitioning to that level.
        # A `1` indicates that the action should be masked, i.e., cannot be taken.
        self.current_action_mask: Optional[np.array] = None

        # History is a list of `actions_taken` above, indicating how you get from the initial atom to the current
        # molecule.
        self.history: List[int] = []

        self.objective: Optional[float] = None
        # Synthetic accessibility score, obtained from RDKit, ranging from 1 [easiest] to 10 [hardest]
        self.sa_score: float = 0.

        # Set this to True if anything goes wrong and the molecule will always evaluate to objective -inf
        self.infeasibility_flag: bool = False

        self.update_action_mask()
        self.update_rdkit_mol(new_atom=initial_atom)

    def is_connected_without_bond(self, atom1: int, atom2: int) -> bool:
        """
        Checks whether the molecule would remain connected if the bond between atom1 and atom2 were removed,
        using RDKit's GetMolFrags function.

        Note:
            - The input atom indices refer to the internal representation which includes the virtual atom at index 0.
            - The rdkit_mol attribute, however, only contains the "real" atoms (i.e. indices starting from 0).
            - Thus, we adjust the indices by subtracting 1.
        """
        # If the molecule has less than two real atoms, removal of any bond will disconnect it.
        if len(self.atoms) <= 3:
            return False

        # Map atom indices from the internal representation to the indices in rdkit_mol.
        # Our internal indices: virtual atom at index 0, then real atoms 1,2,... map to rdkit indices 0,1,...
        rdkit_atom1 = atom1 - 1
        rdkit_atom2 = atom2 - 1

        # Create a copy of the RDKit molecule so that modifications don't affect the original.
        mol_copy = Chem.RWMol(self.rdkit_mol)

        # Remove the bond if it exists.
        bond = mol_copy.GetBondBetweenAtoms(rdkit_atom1, rdkit_atom2)
        if bond is None:
            # If there is no bond between the atoms, then connectivity remains unaffected.
            return True

        mol_copy.RemoveBond(rdkit_atom1, rdkit_atom2)

        # Compute the fragments. GetMolFrags returns a tuple where each element is a tuple of atom indices in that fragment.
        frags = Chem.GetMolFrags(mol_copy, asMols=False)

        # The molecule is connected if and only if there is exactly one fragment.
        return len(frags) == 1

    def keep_fragment(self, fragment_idx):
        """
        Keeps the specified fragment and removes the other.
        """
        if not hasattr(self, 'fragments') or fragment_idx >= len(self.fragments):
            return

        # Replace current rdkit_mol with the selected fragment
        self.rdkit_mol = Chem.RWMol(self.fragments[fragment_idx])

        # Rebuild atoms and bonds arrays from the selected fragment
        self.rebuild_from_rdkit()

        # Clean up fragment data
        del self.fragments

    def rebuild_from_rdkit(self):
        """
        Rebuilds the internal representation (atoms and bonds arrays) based on the current rdkit_mol.
        Used after selecting a fragment to keep.
        """
        # Get atoms from RDKit molecule
        rdkit_atoms = self.rdkit_mol.GetAtoms()
        num_rdkit_atoms = len(rdkit_atoms)

        # Pre-allocate the atoms array (virtual atom + fragment atoms)
        self.atoms = np.zeros(1 + num_rdkit_atoms, dtype=np.uint8)

        # Map atoms from RDKit to our vocabulary indices
        atom_mapping = {}
        for i, atom_name in enumerate(self.vocabulary_atom_names):
            config = self.atom_vocabulary[atom_name]
            key = config["atomic_number"]
            if "formal_charge" in config:
                key = f"{key}_{config['formal_charge']}"
            if "chiral_tag" in config:
                key = f"{key}@{config['chiral_tag']}"
            atom_mapping[key] = i + 1

        # Process all atoms in one go
        for i, atom in enumerate(rdkit_atoms):
            key = atom.GetAtomicNum()
            formal_charge = int(atom.GetFormalCharge())
            if formal_charge != 0:
                key = f"{key}_{formal_charge}"
            chiral_tag = int(atom.GetChiralTag())
            if chiral_tag != 0:
                key = f"{key}@{chiral_tag}"

            # The fragment should only contain atoms from our vocabulary
            self.atoms[i + 1] = atom_mapping[key]  # No default needed - should be in mapping

        # Get adjacency matrix with bond orders directly from RDKit
        adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(self.rdkit_mol, useBO=True)

        # Create bonds matrix with correct dimensions
        num_atoms = 1 + num_rdkit_atoms  # Virtual atom + real atoms
        self.bonds = np.zeros((num_atoms, num_atoms), dtype=np.uint8)

        # Set virtual bonds in one operation
        self.bonds[0, 1:] = self.bonds[1:, 0] = self.virtual_bond_idx

        # Copy adjacency matrix to bonds matrix (offset by 1 for virtual atom)
        self.bonds[1:, 1:] = adjacency_matrix

        # Update topological distance matrix
        self.topological_distance_matrix = np.full((num_atoms, num_atoms),
                                                   self.infinity_distance,
                                                   dtype=np.uint8)

        # Set distances to self and virtual atom in one go
        np.fill_diagonal(self.topological_distance_matrix, 0)
        self.topological_distance_matrix[0, 1:] = self.virtual_distance
        self.topological_distance_matrix[1:, 0] = self.virtual_distance

        # Get distance matrix from RDKit and copy it directly
        if num_rdkit_atoms > 0:
            rdkit_distance_matrix = Chem.GetDistanceMatrix(self.rdkit_mol, force=True).astype(np.uint8)
            self.topological_distance_matrix[1:num_atoms, 1:num_atoms] = rdkit_distance_matrix

        # Check if the molecule is actually connected
        # Get fragments after rebuilding
        frags = Chem.GetMolFrags(self.rdkit_mol, asMols=False)

        # Update disconnected fragments flag based on actual connectivity
        if len(frags) > 1:
            # Still disconnected even after keeping only one fragment from the current operation
            self.has_disconnected_fragments = True
        else:
            # Now connected - safe to remove the flag
            if hasattr(self, 'has_disconnected_fragments'):
                del self.has_disconnected_fragments

        # Ensure molecule is valid
        try:
            Chem.SanitizeMol(self.rdkit_mol)
        except Exception as e:
            print(f"Error sanitizing molecule after fragment rebuild: {e}")
            self.infeasibility_flag = True

        # Update action mask
        self.update_action_mask()

    def remove_bond(self, atom_a_idx, atom_b_idx):
        """
        Removes a bond between two atoms and handles potential fragmentation.
        Returns True if molecule entered fragment handling mode (Level 3).
        """
        # Remove the bond from internal representation
        self.bonds[atom_a_idx, atom_b_idx] = self.bonds[atom_b_idx, atom_a_idx] = 0

        # Remove from RDKit representation (adjusting indices for RDKit)
        self.update_rdkit_mol(remove_bond=(atom_a_idx - 1, atom_b_idx - 1))

        # Create an empty list for atom indices
        atom_indices = []

        # Get both the fragment molecules AND atom indices in a single call
        fragments_mol = Chem.GetMolFrags(self.rdkit_mol,
                                         asMols=True,
                                         sanitizeFrags=True,
                                         fragsMolAtomMapping=atom_indices)

        if len(fragments_mol) > 1:
            # Store fragment information for Level 3 decision
            self.fragments = fragments_mol
            self.fragment_atom_indices = atom_indices
            self.current_action_level = 3
            self.update_action_mask()
            return True

        return False

    def update_action_mask(self):
        """
        Creates the action mask for the current action level. Here, we take
        into account the valence of the present atoms.
        """
        if self.synthesis_done:
            self.current_action_mask = None
            return

        atom_valence = np.array([self.vocabulary_valence[x] for x in self.atoms[1:]])
        atom_valence_remaining = atom_valence - self.bonds[1:, 1:].sum(axis=1)
        ex_action_idx = self.pick_existing_atoms_start_action_idx_lvl_0

        if self.current_action_level == 0:
            self.current_action_mask = np.zeros(len(self.vocabulary_atom_idcs) + len(self.atoms), dtype=bool)
            self.current_action_mask[1:ex_action_idx] = self.atom_feasibility_mask
            if (self.upper_limit_atoms is not None and len(self.atoms) - 1 == self.upper_limit_atoms) or (
                    not np.any(atom_valence_remaining)):
                self.current_action_mask[1:ex_action_idx] = 1
            existing_bond = (self.bonds[1:, 1:] > 0).any(axis=1)
            masked = np.zeros(len(self.atoms) - 1, dtype=bool)
            masked[np.where(atom_valence_remaining <= 0)] = True
            masked[np.where(existing_bond)] = False
            self.current_action_mask[ex_action_idx:] = masked

            # Check for atoms with modifiable bonds (bonds that can be changed)
            has_modifiable_bond = np.zeros(len(self.atoms) - 1, dtype=bool)
            for i in range(len(self.atoms) - 1):
                atom_idx = i + 1  # Adjust for virtual atom

                # Check for bonds that can be decreased (order > 1)
                if np.any(self.bonds[atom_idx, 1:] > 1):
                    has_modifiable_bond[i] = True
                    continue

                # Check for bonds that can be increased (single bonds with remaining valence)
                for j in range(1, len(self.atoms)):
                    if atom_idx != j and self.bonds[atom_idx, j] == 1:  # Single bond exists
                        # If both atoms have remaining valence, the bond can be increased
                        if atom_valence_remaining[atom_idx - 1] > 0 and atom_valence_remaining[j - 1] > 0:
                            has_modifiable_bond[i] = True
                            break

            # Original free non-neighbor calculation
            bond_indicator = np.zeros_like(self.bonds[1:, 1:])
            bond_indicator[np.where(self.bonds[1:, 1:] == 0)] = 1
            np.fill_diagonal(bond_indicator, 0)
            has_free_nonneighbor = np.matmul(bond_indicator, (atom_valence_remaining > 0)[:, None]).squeeze()

            # Only mask atoms with no free non-neighbors AND no modifiable bonds
            no_valid_actions = (has_free_nonneighbor == 0) & ~has_modifiable_bond
            self.current_action_mask[ex_action_idx:][np.where(no_valid_actions)] = 1

            # Only allow termination if molecule is connected
            if hasattr(self, 'has_disconnected_fragments') and self.has_disconnected_fragments:
                self.current_action_mask[0] = True  # Mask termination action

        elif self.current_action_level == 1:
            vocab_size = len(self.vocabulary_atom_idcs)
            num_existing_atoms = len(self.atoms) - 1  # Count of real atoms (excluding virtual)

            # Create action space with 1-based indexing:
            # - Action 0: Invalid (always masked)
            # - Actions 1 to N: Create new atom
            # - Actions N+1 to N+M: Select existing atom
            # - Action N+M+1: Replace atom

            total_actions = 1 + vocab_size + num_existing_atoms + 1

            self.current_action_mask = np.ones(total_actions, dtype=bool)  # Default: all masked

            # Determine which atom was selected at level 0
            if len(self.history) > 0:
                latest_action = self.history[-1]
                if latest_action < self.pick_existing_atoms_start_action_idx_lvl_0:
                    # We added a new atom - can't replace it yet
                    atom_picked_on_lvl_0 = len(self.atoms) - 2
                    can_replace = False
                else:
                    # We selected an existing atom - can potentially replace it
                    atom_picked_on_lvl_0 = latest_action - self.pick_existing_atoms_start_action_idx_lvl_0
                    can_replace = True
            else:
                atom_picked_on_lvl_0 = len(self.atoms) - 2
                can_replace = False

            # Find the actual index in the atoms array (adding 1 for virtual atom)
            atom_idx_in_array = atom_picked_on_lvl_0 + 1

            # Unmask new atom creation actions (1 to N) - applying feasibility masks from config
            if len(self.atom_feasibility_mask) == vocab_size:

                # This handles the case where we have exactly one mask entry per atom type
                self.current_action_mask[1:vocab_size + 1] = np.array(self.atom_feasibility_mask)
            else:
                # Fallback - copy as many mask entries as possible
                n = min(len(self.atom_feasibility_mask), vocab_size)
                self.current_action_mask[1:n + 1] = np.array(self.atom_feasibility_mask[:n])

            # Apply valence constraints to atom creation - fixed vectorized version
            # First, filter indices that are within the bounds of atom_valence_remaining
            valid_atom_indices = np.arange(1, min(vocab_size + 1, len(self.atoms)))

            # Then check which ones have insufficient valence
            if len(valid_atom_indices) > 0:  # Only check if we have valid indices
                insufficient_valence_indices = valid_atom_indices[
                    atom_valence_remaining[valid_atom_indices - 1] < 1]
                if len(insufficient_valence_indices) > 0:
                    self.current_action_mask[insufficient_valence_indices] = True

            # Handle existing atom selection (N+1 to N+M)
            target_atoms = np.arange(1, num_existing_atoms + 1)  # Real atom indices in internal representation

            selection_actions = np.arange(vocab_size + 1, vocab_size + num_existing_atoms + 1)  # N+1 to N+M

            # Create masks for different conditions
            # 1. Can't bond with itself

            self_mask = target_atoms == atom_idx_in_array

            # 2. Check if atoms already have a bond
            has_bond_mask = self.bonds[atom_idx_in_array, target_atoms] > 0

            # 3. Check valence constraints for new bonds
            selected_atom_valence = atom_valence_remaining[atom_idx_in_array - 1]
            target_valences = atom_valence_remaining[target_atoms - 1]
            sufficient_valence_mask = (selected_atom_valence > 0) & (target_valences > 0)

            # Valid actions: either has existing bond OR has sufficient valence for new bond
            valid_bonding_mask = has_bond_mask | (sufficient_valence_mask & ~self_mask)

            # Apply the mask to actions
            self.current_action_mask[selection_actions] = ~valid_bonding_mask

            # Handle replacement action (N+M+1)
            replace_action_idx = vocab_size + num_existing_atoms + 1
            self.current_action_mask[replace_action_idx] = not can_replace

        elif self.current_action_level == 2:
            # Unified action space for Level 2:
            # - Action 0: Invalid (always masked)
            # - Actions 1 to N: Replace atom with type from vocabulary
            # - Actions N+1 to N+6: Set bond order 1-6
            # - Action N+7: Remove bond

            vocab_size = len(self.vocabulary_atom_idcs)
            total_actions = 1 + vocab_size + 7  # Invalid + replacement options + bond operations
            self.current_action_mask = np.ones(total_actions, dtype=bool)  # Default: all masked

            if self.is_replacing_atom:
                # === ATOM REPLACEMENT MODE ===
                # All bond actions are masked in replacement mode
                # Get information about the atom to be replaced

                atom_idx = self.atom_to_replace
                rdkit_atom_idx = atom_idx - 1
                current_atom_type = self.atoms[atom_idx]

                # Calculate total bond order for valence check (excluding virtual bonds)
                current_bonds = self.bonds[atom_idx, 1:]

                mask = (current_bonds > 0) & (current_bonds != self.virtual_bond_idx) & (
                        np.arange(len(current_bonds)) > 0)
                real_bond_sum = np.sum(current_bonds[mask])

                # Get neighboring bonds for validation
                neighbor_indices = np.where(mask)[0]
                neighbor_bonds = [(i, current_bonds[i]) for i in neighbor_indices]

                # Check each possible replacement atom type
                valid_replacements = []

                for atom_type_idx, atom_type in enumerate(self.vocabulary_atom_idcs):
                    action_idx = atom_type_idx + 1  # 1-based indexing

                    # Apply basic constraints
                    if (self.atom_feasibility_mask[atom_type_idx] or  # Not allowed in config
                            atom_type == current_atom_type or  # Same as current atom
                            self.vocabulary_valence[atom_type] < real_bond_sum):  # Insufficient valence
                        continue

                    # Chemical validity check through RDKit
                    if self.validate_atom_replacement(rdkit_atom_idx, atom_type, neighbor_bonds):
                        valid_replacements.append(action_idx)

                # Unmask valid replacement actions
                if valid_replacements:
                    self.current_action_mask[valid_replacements] = False

            else:
                # === BOND MODIFICATION MODE ===
                # Determine the atoms involved in this operation
                if hasattr(self, 'selected_bond'):
                    atom_a_idx, atom_b_idx = self.selected_bond
                else:
                    atom_picked_on_lvl_0 = (
                        len(self.atoms) - 2 if self.history[-2] < self.pick_existing_atoms_start_action_idx_lvl_0
                        else self.history[-2] - self.pick_existing_atoms_start_action_idx_lvl_0)

                    atom_picked_on_lvl_1 = self.history[-1]
                    atom_a_idx = atom_picked_on_lvl_0 + 1
                    atom_b_idx = atom_picked_on_lvl_1 + 1

                # All replacement actions are masked in bond mode
                atom_valence_remaining = np.array([self.vocabulary_valence[x] for x in self.atoms[1:]]) - \
                                         self.bonds[1:, 1:].sum(axis=1)

                current_bond_order = self.bonds[atom_a_idx, atom_b_idx]

                # Calculate maximum allowed bond order based on valence
                extra_increase = min(atom_valence_remaining[atom_a_idx - 1], atom_valence_remaining[atom_b_idx - 1])
                allowed_final_order = min(int(current_bond_order + extra_increase), self.maximum_bond_order)

                # Unmask valid bond order actions (N+1 to N+allowed_final_order)
                if allowed_final_order > 0:
                    valid_bond_actions = np.arange(vocab_size + 1, vocab_size + 1 + allowed_final_order)
                    self.current_action_mask[valid_bond_actions] = False

                # Unmask remove bond action if it would maintain connectivity
                if current_bond_order > 0 and self.is_connected_without_bond(atom_a_idx, atom_b_idx):
                    self.current_action_mask[vocab_size + 7] = False

        elif self.current_action_level == 3:
            # Level 3: Fragment handling
            # If this level is reached, we always have exactly 2 fragments + "do nothing" option
            # Always allow discarding either fragment or keeping both
            # No masking needed here as all 3 options are always valid
            self.current_action_mask = np.zeros(3, dtype=bool)

    def update_topological_distance_matrix(self, new_atom_created: bool = False):
        if new_atom_created:
            new_atom_idx = len(self.atoms) - 1
            self.topological_distance_matrix = np.pad(
                self.topological_distance_matrix, [(0, 1), (0, 1)],
                mode='constant', constant_values=self.infinity_distance
            )
            self.topological_distance_matrix[0, new_atom_idx] = self.topological_distance_matrix[new_atom_idx, 0] = self.virtual_distance
            self.topological_distance_matrix[new_atom_idx, new_atom_idx] = 0
        else:
            self.topological_distance_matrix[1:, 1:] = Chem.GetDistanceMatrix(self.rdkit_mol, force=True).astype(np.uint8)

    def update_rdkit_mol(self, new_atom: Optional[int] = None, set_bond: Optional[Tuple[int, int, int]] = None,
                         remove_bond: Optional[Tuple[int, int]] = None):
        if new_atom is not None:
            atom_idx = new_atom
            atom_config = self.atom_vocabulary[self.vocabulary_atom_names[atom_idx - 1]]
            a = Chem.Atom(atom_config["atomic_number"])
            if "formal_charge" in atom_config:
                a.SetFormalCharge(atom_config["formal_charge"])
            if "chiral_tag" in atom_config:
                if atom_config["chiral_tag"] == 1:
                    a.SetChiralTag(Chem.CHI_TETRAHEDRAL_CW)
                elif atom_config["chiral_tag"] == 2:
                    a.SetChiralTag(Chem.CHI_TETRAHEDRAL_CCW)
            self.rdkit_mol.AddAtom(a)
        elif set_bond is not None:
            i, j, bond_order = set_bond
            if self.rdkit_mol.GetBondBetweenAtoms(i, j) is not None:
                self.rdkit_mol.RemoveBond(i, j)
            self.rdkit_mol.AddBond(i, j, self.bond_types[bond_order])
        elif remove_bond is not None:
            i, j = remove_bond
            if self.rdkit_mol.GetBondBetweenAtoms(i, j) is not None:
                self.rdkit_mol.RemoveBond(i, j)

    def masked_log_probs_for_current_action_level(self, logits: np.array) -> np.array:
        mask = self.current_action_mask
        logits[mask] = np.NINF
        with np.errstate(divide='ignore'):
            log_probs = np.log(softmax(logits))
        return log_probs

    def take_action(self, action: int):
        """
        Execute a given action at the current action level.

        Args:
            action (int): The action to take, based on the current level's action space
        """
        assert not self.synthesis_done, "Taking action on already terminated design. No no!"
        assert self.current_action_mask[
                   action] == False, f"Trying to take action {action} on level {self.current_action_level}, but it is set to infeasible"
        next_level = 0

        if self.current_action_level == 0:
            # Level 0: Termination or Atom Selection
            if action == 0:
                # Terminate molecule design
                self.synthesis_done = True
                self.finalize()
            elif 1 <= action < self.pick_existing_atoms_start_action_idx_lvl_0:
                # Create new atom
                self.atoms = np.append(self.atoms, action)
                self.bonds = np.pad(self.bonds, [(0, 1), (0, 1)], mode='constant', constant_values=0)
                new_atom_idx = len(self.atoms) - 1
                self.bonds[0, new_atom_idx] = self.bonds[new_atom_idx, 0] = self.virtual_bond_idx
                self.update_rdkit_mol(new_atom=action)
                self.update_topological_distance_matrix(new_atom_created=True)
                self.history.append(int(action))
                self.base_atom_idx = new_atom_idx
                next_level = 1
            else:
                # Select existing atom
                selected_atom_idx = action - self.pick_existing_atoms_start_action_idx_lvl_0 + 1
                self.base_atom_idx = selected_atom_idx
                self.history.append(int(action))
                next_level = 1

        elif self.current_action_level == 1:
            # Level 1: Second Atom Selection or Replacement
            vocab_size = len(self.vocabulary_atom_idcs)
            num_existing_atoms = len(self.atoms) - 1
            replace_action_idx = vocab_size + num_existing_atoms + 1  # N+M+1

            if action == 0:
                # Invalid action - should never happen due to masking
                raise ValueError("Invalid action 0 selected in Level 1")
            elif action <= vocab_size:
                # Create new atom (actions 1 to N)
                atom_type = action  # In 1-based indexing, action is the atom type
                self.atoms = np.append(self.atoms, atom_type)
                self.bonds = np.pad(self.bonds, [(0, 1), (0, 1)], mode='constant', constant_values=0)
                new_atom_idx = len(self.atoms) - 1
                self.bonds[0, new_atom_idx] = self.bonds[new_atom_idx, 0] = self.virtual_bond_idx
                self.update_rdkit_mol(new_atom=atom_type)
                self.update_topological_distance_matrix(new_atom_created=True)
                self.history.append(int(action))
                self.last_created_atom_idx = new_atom_idx
                next_level = 2
            elif action == replace_action_idx:
                # Replace atom action (N+M+1)
                atom_picked_on_lvl_0 = self.history[-1] - self.pick_existing_atoms_start_action_idx_lvl_0
                self.atom_to_replace = atom_picked_on_lvl_0 + 1  # +1 for virtual atom
                self.is_replacing_atom = True
                self.history.append(int(action))
                next_level = 2
            else:
                # Bond with existing atom (actions N+1 to N+M)
                existing_atom_idx = action - vocab_size - 1  # Convert from N+1+idx to idx
                target_atom_idx = existing_atom_idx + 1  # +1 for virtual atom

                # Handle the case where the selected atom is the same as base atom
                if target_atom_idx == self.base_atom_idx:
                    raise ValueError("Cannot bond an atom with itself")

                self.selected_bond = (self.base_atom_idx, target_atom_idx)
                self.history.append(int(action))
                next_level = 2

        elif self.current_action_level == 2:
            # Level 2: Bond Order Setting or Atom Replacement
            vocab_size = len(self.vocabulary_atom_idcs)

            if action == 0:
                # Invalid action - should never happen due to masking
                raise ValueError("Invalid action 0 selected in Level 2")

            if self.is_replacing_atom and action <= vocab_size:
                # Atom replacement (actions 1 to N)
                new_atom_type = action  # Direct use of 1-based indexing
                self.replace_atom(self.atom_to_replace, new_atom_type)
                self.history.append(int(action))
                self.is_replacing_atom = False
                self.atom_to_replace = None
                self.update_topological_distance_matrix()
                next_level = 0

            elif not self.is_replacing_atom:
                # Determine atoms for bonding
                if hasattr(self, 'selected_bond'):
                    atom_a_idx, atom_b_idx = self.selected_bond
                else:
                    atom_a_idx = self.base_atom_idx
                    atom_b_idx = self.last_created_atom_idx

                if vocab_size < action <= vocab_size + 6:
                    # Set bond order (actions N+1 to N+6)
                    bond_order = action - vocab_size
                    self.bonds[atom_a_idx, atom_b_idx] = self.bonds[atom_b_idx, atom_a_idx] = bond_order
                    self.update_rdkit_mol(set_bond=(atom_a_idx - 1, atom_b_idx - 1, bond_order))
                    next_level = 0

                elif action == vocab_size + 7:
                    # Remove bond (action N+7)
                    entered_fragment_mode = self.remove_bond(atom_a_idx, atom_b_idx)
                    next_level = 3 if entered_fragment_mode else 0

                # Update distances after bond modification
                self.update_topological_distance_matrix()
                self.history.append(int(action))

                # Clean up
                if hasattr(self, 'selected_bond'):
                    del self.selected_bond

            # Invalid combination - should never happen
            else:
                raise ValueError(f"Invalid action {action} for Level 2 in current mode")

        elif self.current_action_level == 3:
            # Level 3: Fragment Handling
            if action < 2:
                # Keep fragment 0 or 1
                self.keep_fragment(action)
            else:  # action == 2: Keep both fragments
                # Flag that molecule has disconnected fragments
                self.has_disconnected_fragments = True

            self.history.append(int(action))
            next_level = 0

        self.current_action_level = next_level
        self.update_action_mask()

    def validate_atom_replacement(self, rdkit_atom_idx, new_atom_type, neighbor_bonds):
        """
        Performs a detailed check if replacing an atom would create a valid molecule.

        Parameters:
            rdkit_atom_idx (int): RDKit index of atom to replace
            new_atom_type (int): New atom type index from vocabulary
            neighbor_bonds (list): List of (neighbor_idx, bond_order) tuples

        Returns:
            bool: True if replacement would be valid, False otherwise
        """
        # # Skip Fluorine validation to test theory
        # atom_name = self.vocabulary_atom_names[new_atom_type - 1]
        # if atom_name == "F":  # Skip Fluorine validation
        #     print("DEBUG: Skipping Fluorine validation check")
        #     return False  # Just return false without testing

        # Create a copy of the RDKit molecule to test the replacement
        test_mol = Chem.RWMol(self.rdkit_mol)

        # Get atom configuration from vocabulary
        atom_name = self.vocabulary_atom_names[new_atom_type - 1]
        atom_config = self.atom_vocabulary[atom_name]

        # Create the new atom with proper configuration
        new_atom = Chem.Atom(atom_config["atomic_number"])
        if "formal_charge" in atom_config:
            new_atom.SetFormalCharge(atom_config["formal_charge"])
        if "chiral_tag" in atom_config:
            if atom_config["chiral_tag"] == 1:
                new_atom.SetChiralTag(Chem.CHI_TETRAHEDRAL_CW)
            elif atom_config["chiral_tag"] == 2:
                new_atom.SetChiralTag(Chem.CHI_TETRAHEDRAL_CCW)

        # Replace the atom in the test molecule
        test_mol.ReplaceAtom(rdkit_atom_idx, new_atom)

        # Try to sanitize the molecule with the new atom
        try:
            Chem.SanitizeMol(test_mol)
            return True
        except:
            return False

    def replace_atom(self, atom_idx: int, new_atom_type: int):
        """
        Replace an existing atom with a new atom type while preserving bonds.

        Parameters:
            atom_idx (int): Index of the atom to replace in self.atoms
            new_atom_type (int): New atom type index from vocabulary
        """
        # print(f"Replacing atom {atom_idx} with type {new_atom_type}")
        # print(f"Before replacement, atoms are: {self.atoms}")

        # Update internal representation
        self.atoms[atom_idx] = new_atom_type

        # Update RDKit representation
        rdkit_atom_idx = atom_idx - 1
        atom_name = self.vocabulary_atom_names[new_atom_type - 1]
        atom_config = self.atom_vocabulary[atom_name]

        # print(f"DEBUG: Replacing with {atom_name}, atomic_number={atom_config['atomic_number']}")

        # Work on a copy of the RDKit molecule
        updated_mol = Chem.RWMol(self.rdkit_mol)
        atom = updated_mol.GetAtomWithIdx(rdkit_atom_idx)
        # print(f"DEBUG: RDKit atom before: {atom.GetSymbol()}, atomic num: {atom.GetAtomicNum()}")

        # Update the atom type
        atom.SetAtomicNum(atom_config["atomic_number"])

        # Update formal charge and chirality if specified
        atom.SetFormalCharge(atom_config.get("formal_charge", 0))

        if atom_config.get("chiral_tag") == 1:
            atom.SetChiralTag(Chem.CHI_TETRAHEDRAL_CW)
        elif atom_config.get("chiral_tag") == 2:
            atom.SetChiralTag(Chem.CHI_TETRAHEDRAL_CCW)
        else:
            atom.SetChiralTag(Chem.CHI_UNSPECIFIED)

        # Reset hydrogen counts - let RDKit handle them implicitly
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)

        # Try to sanitize to recalculate implicit hydrogens
        try:
            # First reset property cache
            for a in updated_mol.GetAtoms():
                a.UpdatePropertyCache(strict=False)

            # Sanitize the molecule
            Chem.SanitizeMol(updated_mol)

            # If successful, update our rdkit_mol
            self.rdkit_mol = updated_mol
        except Exception as e:
            print(f"Error sanitizing molecule after replacement: {e}")
            print("Keeping original molecule")
            self.infeasibility_flag = True

        # print(f"After replacement, atoms are: {self.atoms}")

    def finalize(self, assert_feasible: bool = False):
        if assert_feasible:
            self.assert_feasible()
        try:
            Chem.SanitizeMol(self.rdkit_mol)
        except:
            self.infeasibility_flag = True
        if not self.infeasibility_flag:
            self.smiles_string = Chem.MolToSmiles(self.rdkit_mol)
            if self.smiles_string == "C":
                self.infeasibility_flag = True

    def assert_feasible(self):
        assert self.atoms[0] == 0, "First atom should be virtual (0)"
        assert np.all([not self.atom_feasibility_mask[x - 1] for x in self.atoms[1:]]) and np.all(self.atoms[1:] > 0), "Only atoms allowed that are also allowd in config vocabulary"
        assert self.upper_limit_atoms is None or len(self.atoms) - 1 <= self.upper_limit_atoms, "Exceeded maximum number of atoms"
        assert np.all(self.bonds[0, 1:] == self.virtual_bond_idx) and np.all(self.bonds[1:, 0] == self.virtual_bond_idx), "Virtual atom must be connected to all other atoms."
        assert not np.any(self.bonds.diagonal()), "An atom (even virtual) may not be connected to itself"
        assert not np.any(self.bonds - self.bonds.T), "Bond matrix must be symmetric"
        assert np.all(np.array([self.vocabulary_valence[x] for x in self.atoms[1:]]) - self.bonds[1:, 1:].sum(axis=1) >= 0), "Valence constraints not satisfied"
        if self.current_action_level == 0 and len(self.atoms) > 2:
            assert np.all(self.bonds[1:, 1:].sum(axis=1) > 0), "An atom must be connected to at least another atom"

    def to_rdkit_mol(self, sanitize=True) -> Chem.RWMol:
        mol = Chem.RWMol()
        num_atoms = len(self.atoms) - 1
        for atom_idx in self.atoms[1:]:
            atom_config = self.atom_vocabulary[self.vocabulary_atom_names[atom_idx - 1]]
            a = Chem.Atom(atom_config["atomic_number"])
            if "formal_charge" in atom_config:
                a.SetFormalCharge(atom_config["formal_charge"])
            if "chiral_tag" in atom_config:
                if atom_config["chiral_tag"] == 1:
                    a.SetChiralTag(Chem.CHI_TETRAHEDRAL_CW)
                elif atom_config["chiral_tag"] == 2:
                    a.SetChiralTag(Chem.CHI_TETRAHEDRAL_CCW)
            mol.AddAtom(a)
        bond_type = {
            1: Chem.rdchem.BondType.SINGLE,
            2: Chem.rdchem.BondType.DOUBLE,
            3: Chem.rdchem.BondType.TRIPLE,
            4: Chem.rdchem.BondType.QUADRUPLE,
            5: Chem.rdchem.BondType.QUINTUPLE,
            6: Chem.rdchem.BondType.HEXTUPLE
        }
        bonds = self.bonds[1:, 1:]
        for i in range(num_atoms):
            for j in range(i, num_atoms):
                if bonds[i, j] > 0:
                    mol.AddBond(i, j, bond_type[bonds[i, j]])
        if sanitize:
            try:
                Chem.SanitizeMol(mol)
            except:
                self.infeasibility_flag = True
        return mol

    def is_terminable(self):
        return self.current_action_level == 0 and not self.synthesis_done

    def to_smiles(self) -> str:
        return Chem.MolToSmiles(self.rdkit_mol)

    @staticmethod
    def init_batch_from_instance_list(config: MoleculeConfig, instances: List[int], network: nn.Module, device: torch.device):
        return [MoleculeDesign(config=config, initial_atom=atom) for atom in instances]

    @staticmethod
    def log_probability_fn(trajectories: List['MoleculeDesign'], network: nn.Module) -> List[np.array]:
        log_probs_to_return: List[np.array] = []
        network.eval()
        with torch.no_grad():
            batch = MoleculeDesign.list_to_batch(molecules=trajectories, device=network.device)
            batch_logits_per_level = list(network(batch))

            # Make sure we have logits for all levels (0-3)
            while len(batch_logits_per_level) < 4:
                # Add dummy logits for missing levels (like level 3)
                batch_logits_per_level.append(torch.zeros((len(trajectories), 3), device=network.device))

            for lvl in range(4):  # Changed from 3 to 4 to include Level 3
                batch_logits_per_level[lvl] = batch_logits_per_level[lvl].cpu().numpy()

            for i, mol in enumerate(trajectories):
                logits = batch_logits_per_level[mol.current_action_level][i]
                if mol.current_action_level != 2:  # Keep special handling for level 2
                    logits = logits[:len(mol.current_action_mask)]
                log_probs_to_return.append(mol.masked_log_probs_for_current_action_level(logits))
        return log_probs_to_return

    def transition_fn(self, action: int) -> Tuple['BaseTrajectory', bool]:
        copied_molecule = copy.deepcopy(self)
        copied_molecule.take_action(action)
        return copied_molecule, copied_molecule.synthesis_done

    def to_max_evaluation_fn(self) -> float:
        if self.objective is None:
            raise ValueError("Objective is `None`. Evaluate molecule with `MoleculeObjectiveEvaluator` first.")
        return self.objective

    def num_actions(self) -> int:
        return int((1 - self.current_action_mask).sum())

    @staticmethod
    def list_to_batch(molecules: List['MoleculeDesign'], device: torch.device = None,
                      include_feasibility_masks: bool = False) -> dict:
        atoms_padding_idx = len(molecules[0].vocabulary_atom_idcs) + 1
        degree_padding_idx = max(molecules[0].vocabulary_valence) + 1
        bond_padding_idx = MoleculeDesign.virtual_bond_idx + 1
        distance_padding_idx = MoleculeDesign.maximum_num_atoms_overall + 3

        device = torch.device("cpu") if device is None else device
        num_atoms = [len(mol.atoms) for mol in molecules]
        max_num_atoms = max(num_atoms)

        batch_level_idx = [mol.current_action_level for mol in molecules]

        batch_picked_atom_mhe = np.zeros((len(molecules), max_num_atoms), dtype=int)
        ex_pick_idx_start = molecules[0].pick_existing_atoms_start_action_idx_lvl_0
        for i, mol in enumerate(molecules):
            if mol.current_action_level == 0:
                pass
            elif mol.current_action_level == 1:
                atom_picked_on_lvl_0 = len(mol.atoms) - 1 if mol.history[-1] < ex_pick_idx_start else mol.history[-1] - ex_pick_idx_start + 1
                batch_picked_atom_mhe[i, atom_picked_on_lvl_0] = 1
            elif mol.current_action_level == 2:
                atom_picked_on_lvl_0 = len(mol.atoms) - 1 if mol.history[-2] < ex_pick_idx_start else mol.history[-2] - ex_pick_idx_start + 1
                batch_picked_atom_mhe[i, atom_picked_on_lvl_0] = 1
                atom_picked_on_lvl_1 = mol.history[-1] + 1
                batch_picked_atom_mhe[i, atom_picked_on_lvl_1] = 2

        batch_atoms = np.stack([
            np.concatenate((mol.atoms, np.full(max_num_atoms - num_atoms[i], fill_value=atoms_padding_idx, dtype=int)))
            for i, mol in enumerate(molecules)
        ])

        batch_atoms_degree = np.stack([
            np.concatenate((
                (mol.bonds > 0).sum(axis=1) - 1,
                np.full(max_num_atoms - num_atoms[i], fill_value=degree_padding_idx, dtype=int)
            ))
            for i, mol in enumerate(molecules)
        ])

        bonds_list = []
        for i, mol in enumerate(molecules):
            padded_bonds = np.pad(
                mol.bonds, [(0, max_num_atoms - num_atoms[i]), (0, max_num_atoms - num_atoms[i])],
                mode="constant", constant_values=bond_padding_idx
            )
            np.fill_diagonal(padded_bonds, bond_padding_idx)
            bonds_list.append(padded_bonds)
        batch_bonds = np.stack(bonds_list)

        distance_matrices_list = [
            np.pad(
                mol.topological_distance_matrix, [(0, max_num_atoms - num_atoms[i]), (0, max_num_atoms - num_atoms[i])],
                mode="constant", constant_values=distance_padding_idx
            )
            for i, mol in enumerate(molecules)
        ]
        batch_topological_distance = np.stack(distance_matrices_list)

        additive_padding_masks = []
        for i, mol in enumerate(molecules):
            mask = np.zeros_like(mol.bonds).astype(float)
            mask = np.pad(
                mask, [(0, max_num_atoms - num_atoms[i]), (0, max_num_atoms - num_atoms[i])],
                mode="constant", constant_values=np.NINF
            )
            np.fill_diagonal(mask, 0)
            additive_padding_masks.append(mask)
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
            feasibility_masks_per_level = []
            num_actions_per_level_and_mol = [
                [mol.pick_existing_atoms_start_action_idx_lvl_0 + len(mol.atoms) - 1 for mol in molecules],
                [len(mol.atoms) - 1 for mol in molecules],
                [molecules[0].maximum_bond_order + 1] * len(molecules),  # +1 for the remove bond action
                [3] * len(molecules)  # Level 3: Always 3 actions (discard fragment 0, discard fragment 1, keep both)
            ]
            for lvl, num_actions_per_mol in enumerate(num_actions_per_level_and_mol):
                max_num_actions = max(num_actions_per_mol)
                feasibility_masks_per_level.append(
                    torch.from_numpy(
                        np.stack([
                            np.pad(
                                mol.current_action_mask,
                                [(0, max_num_actions - num_actions_per_mol[i])],
                                mode='constant', constant_values=1
                            ) if mol.current_action_level == lvl else np.zeros(max_num_actions, dtype=bool)
                            for i, mol in enumerate(molecules)
                        ])
                    ).bool().to(device)
                )

            return_dict["feasibility_mask_level_zero"] = feasibility_masks_per_level[0]
            return_dict["feasibility_mask_level_one"] = feasibility_masks_per_level[1]
            return_dict["feasibility_mask_level_two"] = feasibility_masks_per_level[2]

            # Add mask for level three
            if len(feasibility_masks_per_level) > 3:
                return_dict["feasibility_mask_level_three"] = feasibility_masks_per_level[3]

        return return_dict

    @staticmethod
    def batch_to_device(batch: dict, device: torch.device):
        return {k: v.to(device) for k, v in batch.items()}

    @staticmethod
    def get_c_chains(config: MoleculeConfig) -> List['MoleculeDesign']:
        carbon_atom_idx = list(config.atom_vocabulary.keys()).index("C") + 1
        instance_list = []
        for num_c_to_add in range(min(config.max_num_atoms - 1, config.start_c_chain_max_len)):
            mol = MoleculeDesign(config, initial_atom=1)
            for i in range(num_c_to_add):
                mol.take_action(carbon_atom_idx)  # add C at level 0
                mol.take_action(len(mol.atoms) - 3)  # attach to last added atom
            instance_list.append(mol)
        return instance_list

    @staticmethod
    def get_single_atom_molecules(config: MoleculeConfig, repeat: int = 1) -> List['MoleculeDesign']:
        atoms = []
        for i, atom in enumerate(config.atom_vocabulary.keys()):
            if config.atom_vocabulary[atom]["allowed"]:
                atoms.append(i + 1)
        return MoleculeDesign.init_batch_from_instance_list(config, atoms * repeat, None, None)

    @staticmethod
    def random_atom_order_in_smiles(smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES input.")
        num_atoms = mol.GetNumAtoms()
        atom_indices = list(range(num_atoms))
        random.shuffle(atom_indices)
        reordered_mol = Chem.RenumberAtoms(mol, atom_indices)
        return Chem.MolToSmiles(reordered_mol, isomericSmiles=True, canonical=False)

    @staticmethod
    def from_smiles(config: MoleculeConfig, smiles: str, do_finish=False, compare_smiles=False) -> 'MoleculeDesign':
        mol = Chem.MolFromSmiles(smiles)
        Chem.SanitizeMol(mol)
        return MoleculeDesign.from_rdkit_mol(config, mol, smiles, do_finish, compare_smiles)

    @staticmethod
    def from_rdkit_mol(config: MoleculeConfig, rdkit_mol: Chem.RWMol, smiles: str, do_finish=True,
                       compare_smiles=True) -> 'MoleculeDesign':
        """
        Creates an instance of `MoleculeDesign` from an RDKit molecule.
        Directly constructs the internal representation by bypassing the action system.
        """
        # Create an empty molecule design with the first atom
        Chem.Kekulize(rdkit_mol)
        atoms = rdkit_mol.GetAtoms()
        atom_idcs_for_design = []  # Atom types in our vocabulary

        # Map atomic numbers to indices in our vocabulary
        atomic_num_to_atom_idx = {}
        for i, atom_name in enumerate(config.atom_vocabulary.keys()):
            k = config.atom_vocabulary[atom_name]["atomic_number"]
            if "formal_charge" in config.atom_vocabulary[atom_name]:
                k = f"{k}_{config.atom_vocabulary[atom_name]['formal_charge']}"
            if "chiral_tag" in config.atom_vocabulary[atom_name]:
                k = f"{k}@{config.atom_vocabulary[atom_name]['chiral_tag']}"
            atomic_num_to_atom_idx[k] = i + 1

        # Get atom types for all atoms in the molecule
        for atom in atoms:
            k = atom.GetAtomicNum()
            formal_charge = int(atom.GetFormalCharge())
            if formal_charge != 0:
                k = f"{k}_{formal_charge}"
            chiral_tag = int(atom.GetChiralTag())
            if chiral_tag != 0:
                k = f"{k}@{chiral_tag}"
            atom_idx = atomic_num_to_atom_idx[k]
            atom_idcs_for_design.append(atom_idx)

        # Initialize with the first atom
        design = MoleculeDesign(config, atom_idcs_for_design[0])

        # CRUCIAL CHANGE: Instead of using the action system, we'll build the molecule directly

        # 1. First, recreate the RDKit molecule from scratch
        design.rdkit_mol = Chem.RWMol()

        # 2. Add all atoms to both the design.atoms array and the RDKit molecule
        for i in range(len(atoms)):
            if i == 0:
                # First atom is already added during initialization
                atom_config = config.atom_vocabulary[list(config.atom_vocabulary.keys())[atom_idcs_for_design[0] - 1]]
                a = Chem.Atom(atom_config["atomic_number"])
                design.rdkit_mol.AddAtom(a)
            else:
                atom_type = atom_idcs_for_design[i]
                design.atoms = np.append(design.atoms, atom_type)
                atom_config = config.atom_vocabulary[list(config.atom_vocabulary.keys())[atom_type - 1]]
                a = Chem.Atom(atom_config["atomic_number"])
                design.rdkit_mol.AddAtom(a)

        # 3. Update the bonds matrix and add bonds to the RDKit molecule
        num_atoms = len(design.atoms)
        design.bonds = np.zeros((num_atoms, num_atoms), dtype=np.uint8)
        design.bonds[0, 1:] = design.bonds[1:, 0] = design.virtual_bond_idx  # Connect virtual atom

        # 4. Add bonds between atoms based on the adjacency matrix
        adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(rdkit_mol, useBO=True)
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                bond_order = int(adjacency_matrix[i, j])
                if bond_order > 0:
                    # Add bond to the design's bond matrix
                    design.bonds[i + 1, j + 1] = design.bonds[j + 1, i + 1] = bond_order

                    # Add bond to the RDKit molecule
                    design.rdkit_mol.AddBond(i, j, design.bond_types[bond_order])

        # 5. Initialize and update the topological distance matrix
        # First create a distance matrix of the right size (including virtual atom)
        design.topological_distance_matrix = np.full((num_atoms, num_atoms),
                                                     design.infinity_distance,
                                                     dtype=np.uint8)

        # Set the diagonal to 0 (distance to self)
        np.fill_diagonal(design.topological_distance_matrix, 0)

        # Set distances to virtual atom
        design.topological_distance_matrix[0, 1:] = design.topological_distance_matrix[1:, 0] = design.virtual_distance

        # Set actual distances between atoms
        if len(atoms) > 0:
            rdkit_distance_matrix = Chem.GetDistanceMatrix(design.rdkit_mol, force=True).astype(np.uint8)
            design.topological_distance_matrix[1:len(atoms) + 1, 1:len(atoms) + 1] = rdkit_distance_matrix

        # 6. Set the current action level to 0 (choosing atoms)
        design.current_action_level = 0
        design.update_action_mask()

        # 7. Finalize if needed
        if do_finish:
            design.synthesis_done = True
            design.finalize()
            if compare_smiles:
                assert Chem.CanonSmiles(design.smiles_string) == Chem.CanonSmiles(
                    smiles), f"Converted: {Chem.CanonSmiles(design.smiles_string)}, RDKit: {Chem.CanonSmiles(smiles)}"

        return design