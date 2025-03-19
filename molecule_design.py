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
        - Level 1: If not terminating, pick a second atom on which a bind decision must be made. (index 0 up to number of atoms)
        - Level 2: Pick the type of bond (index 0 up to order 6)

    Level 0 and 1 are predicted simultaneously by the network, while for level 2 we mark the chosen atom for the network.

    Atom types are specified in the config under `atom_vocabulary`. Indexing starts at 1. Index 0 is for a virtual atom.
    - Index 0: Virtual Atom, which is connected (with special bond order) to every other atom (and vice versa).

    We store all actions in a history, which is a list of indices indicating the action that was taken on a certain level.
    For example, with a vocabulary of [C, N, O], and starting from the atom C, the action history
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
        self.atom_feasibility_mask = [not self.atom_vocabulary[x]["allowed"] for x in self.vocabulary_atom_names]  # if not allowed, then feasbility mask must be set to True

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

        self.synthesis_done = False
        self.smiles_string: Optional[str] = None  # Is set after synthesis is done
        self.current_objective = float("-inf")

        # Current action level. Can be 0, 1, 2
        self.current_action_level = 0  # start by choosing <terminate>/<create new atom and pick>/<pick existing atom>

        # The action mask indicates before each action what is feasible at the current level.
        # It is set for each level when transitioning to that level
        # A `1` indicates that the action should be masked, i.e., cannot be taken.
        self.current_action_mask: Optional[np.array] = None

        # History is a list of `actions_taken` above, indicating how you get from the initial atom to the current
        # molecule. See class docstring for an example.
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
        Checks whether the molecule would remain connected if the bond between atom1 and atom2 were removed.
        This is done by performing a depth-first search from atom1 and checking if atom2 is still reachable.

        Parameters:
            atom1, atom2: Atom indices in the internal representation (including the virtual atom at index 0)

        Returns:
            bool: True if the molecule would remain connected after removing the bond, False otherwise
        """
        # Special case: If there are only two real atoms, removing a bond always disconnects the molecule
        if len(self.atoms) <= 3:
            return False

        # Create a copy of the bonds matrix without the bond between atom1 and atom2
        bonds_copy = self.bonds.copy()
        bonds_copy[atom1, atom2] = 0
        bonds_copy[atom2, atom1] = 0

        # Skip the virtual atom (index 0) for the connectivity check
        real_atoms = len(self.atoms) - 1

        # Using a depth-first search to check connectivity
        visited = np.zeros(len(self.atoms), dtype=bool)
        stack = [atom1]
        visited[atom1] = True

        while stack:
            current = stack.pop()
            # If we can reach atom2 from atom1 without using the direct bond between them, the molecule remains connected
            if current == atom2:
                return True

            for neighbor in range(1, len(self.atoms)):  # Skip virtual atom (0)
                if bonds_copy[current, neighbor] > 0 and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)

        # If we've explored all reachable atoms and haven't found atom2, the molecule would be disconnected
        return False

    def update_action_mask(self):
        """
        Creates the action mask for the current action level. Here, we take
        into account the valence of the present atoms.
        """
        if self.synthesis_done:
            self.current_action_mask = None
            return
        atom_valence = np.array([self.vocabulary_valence[x] for x in self.atoms])
        atom_valence_remaining = atom_valence - self.bonds[:, 1:].sum(axis=1)
        ex_action_idx = self.pick_existing_atoms_start_action_idx_lvl_0  # alias

        print("----- update_action_mask: START -----")
        print("Current bonds matrix:")
        print(self.bonds)

        print("Pre-update action mask:")
        print(self.current_action_mask)  # Example: print current state before updating the mask


        if self.current_action_level == 0:
            self.current_action_mask = np.zeros(len(self.vocabulary_atom_idcs) + len(self.atoms), dtype=bool)
            self.current_action_mask[1:ex_action_idx] = self.atom_feasibility_mask
            if (self.upper_limit_atoms is not None and len(self.atoms) - 1 == self.upper_limit_atoms) or \
                    (not np.any(atom_valence_remaining[1:])):
                self.current_action_mask[1:ex_action_idx] = 1
            existing_bond = (self.bonds[1:, 1:] > 0).any(axis=1)
            mask_valence = (atom_valence_remaining[1:] <= 0)
            final_mask = np.logical_and(mask_valence, np.logical_not(existing_bond))
            masked = np.zeros(len(self.atoms) - 1, dtype=bool)
            masked[np.where(atom_valence_remaining[1:] <= 0)] = True
            masked[np.where(existing_bond)] = False
            self.current_action_mask[ex_action_idx:] = masked
            bond_indicator = np.zeros_like(self.bonds[1:, 1:])
            bond_indicator[np.where(self.bonds[1:, 1:] == 0)] = 1
            np.fill_diagonal(bond_indicator, 0)
            has_free_nonneighbor = np.matmul(bond_indicator, (atom_valence_remaining[1:] > 0)[:, None]).squeeze()
            self.current_action_mask[ex_action_idx:][np.where(has_free_nonneighbor == 0)] = 1

        elif self.current_action_level == 1:
            # Expanded action space for level 1:
            # Block 1 (indices [0, new_atom_action_count - 1]): New atom creation.
            # Block 2 (indices starting at new_atom_action_count): Existing bond selection.
            new_atom_action_count = len(self.vocabulary_atom_idcs)
            existing_bond_action_count = len(self.atoms) - 1  # real atoms excluding virtual.
            total_actions = new_atom_action_count + existing_bond_action_count
            self.current_action_mask = np.zeros(total_actions, dtype=bool)
            atom_picked_on_lvl_0 = (
                len(self.atoms) - 2 if self.history[-1] < self.pick_existing_atoms_start_action_idx_lvl_0
                else self.history[-1] - self.pick_existing_atoms_start_action_idx_lvl_0)
            self.current_action_mask[:new_atom_action_count] = np.array(self.atom_feasibility_mask)
            for idx in range(new_atom_action_count):
                if idx < len(self.atoms) - 1:
                    if atom_valence_remaining[idx + 1] < 1:
                        self.current_action_mask[idx] = 1
            for idx in range(existing_bond_action_count):
                if idx == atom_picked_on_lvl_0:
                    self.current_action_mask[new_atom_action_count + idx] = 1
                elif self.bonds[atom_picked_on_lvl_0 + 1, idx + 1] > 0:
                    self.current_action_mask[new_atom_action_count + idx] = 0
                else:
                    self.current_action_mask[new_atom_action_count + idx] = 1





        elif self.current_action_level == 2:

            expected_mask_length = 2 * self.maximum_bond_order

            self.current_action_mask = np.ones(expected_mask_length, dtype=bool)

            # Debug print: Show history and selected_bond if it exists.

            print("DEBUG: In level 2 update_action_mask")

            print("DEBUG: self.history =", self.history)

            if hasattr(self, 'selected_bond'):

                print("DEBUG: self.selected_bond =", self.selected_bond)

                atom_a_idx, atom_b_idx = self.selected_bond

            else:

                # Fall back on using history.

                # NOTE: This branch might select a different bond than intended.

                atom_picked_on_lvl_0 = (

                    len(self.atoms) - 2 if self.history[-2] < self.pick_existing_atoms_start_action_idx_lvl_0

                    else self.history[-2] - self.pick_existing_atoms_start_action_idx_lvl_0)

                atom_picked_on_lvl_1 = self.history[-1]

                atom_a_idx = atom_picked_on_lvl_0 + 1

                atom_b_idx = atom_picked_on_lvl_1 + 1

            # Debug prints for the bond selection.

            print("Atom indices for bond update: atom_a_idx =", atom_a_idx, ", atom_b_idx =", atom_b_idx)

            # Fetch remaining valence info.

            atom_valence = np.array([self.vocabulary_valence[x] for x in self.atoms])

            atom_valence_remaining = atom_valence - self.bonds[:, 1:].sum(axis=1)

            print("Atom valences:", atom_valence)

            print("Atom valence remaining:", atom_valence_remaining)

            current_bond_order = self.bonds[atom_a_idx, atom_b_idx]

            print("Current bond order:", current_bond_order)

            # Compute available extra increase based solely on remaining valence.

            extra_increase = min(atom_valence_remaining[atom_a_idx], atom_valence_remaining[atom_b_idx])

            allowed_final_order = current_bond_order + extra_increase

            print("Extra available increase:", extra_increase)

            print("Allowed final order for increase actions:", allowed_final_order)

            # Unmask available increasing actions.

            self.current_action_mask[:int(allowed_final_order)] = False

            print("Action mask after unmasking increase actions:", self.current_action_mask)

            # Unmask reduction candidates:

            if current_bond_order > 0:

                for reduction in range(1, current_bond_order + 1):

                    reduction_idx = self.maximum_bond_order + reduction - 1

                    new_order = current_bond_order - reduction

                    print("Reduction move:", reduction, "would result in order:", new_order, "at reduction_idx:",
                          reduction_idx)

                    if new_order > 0:

                        self.current_action_mask[reduction_idx] = False

                        print("Partial reduction unmasked at index", reduction_idx)

                    else:

                        if self.is_connected_without_bond(atom_a_idx, atom_b_idx):

                            self.current_action_mask[reduction_idx] = False

                            print("Full removal allowed (connectivity holds) unmasked at index", reduction_idx)

                        else:

                            self.current_action_mask[reduction_idx] = True

                            print("Full removal disallowed (connectivity fails) at index", reduction_idx)

            reduction_indices = list(range(self.maximum_bond_order, len(self.current_action_mask)))

            print("Reduction indices computed: ", reduction_indices)

            feasible_reduction = [i for i in reduction_indices if not self.current_action_mask[i]]

            print("Feasible reduction candidates (should be non-empty if a bond is reducible):", feasible_reduction)

            print("Final updated action mask:", self.current_action_mask)

        # Log the computed reduction indices
        reduction_indices = list(range(self.maximum_bond_order, len(self.current_action_mask)))
        print("Reduction indices computed: ", reduction_indices)
        feasible_reduction = [i for i in reduction_indices if
                              self.current_action_mask[i] == 0 or self.current_action_mask[i] == False]
        print("Feasible reduction candidates (should be non-empty if a bond is reducible):", feasible_reduction)

        # Final mask log
        print("Updated action mask:")
        print(self.current_action_mask)
        print("----- update_action_mask: END -----")

    def update_topological_distance_matrix(self, new_atom_created: bool = False):
        if new_atom_created:
            new_atom_idx = len(self.atoms) - 1
            self.topological_distance_matrix = np.pad(self.topological_distance_matrix, [(0, 1), (0, 1)], mode='constant',
                                                      constant_values=self.infinity_distance)
            self.topological_distance_matrix[0, new_atom_idx] = self.topological_distance_matrix[new_atom_idx, 0] = self.virtual_distance
            self.topological_distance_matrix[new_atom_idx, new_atom_idx] = 0
        else:
            # re-compute the topological distance matrix from the current molecule and set it
            self.topological_distance_matrix[1:, 1:] = Chem.GetDistanceMatrix(self.rdkit_mol, force=True).astype(np.uint8)

    def update_rdkit_mol(self, new_atom: Optional[int] = None, set_bond: Optional[Tuple[int, int, int]] = None,
                         remove_bond: Optional[Tuple[int, int]] = None):
        """
        Updates the RDKit mol by either adding a new atom or setting/removing the bond between two atoms.
        Parameters:
            new_atom: Atom to add as index in vocabulary
            set_bond: Tuple of ints (i, j, bond order), where i,j start from 0 (so we do not count virtual atom)
            remove_bond: Tuple of ints (i, j), where i,j start from 0, to remove a bond
        """
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
            # Check if bond already exists - if so, remove it first
            if self.rdkit_mol.GetBondBetweenAtoms(i, j) is not None:
                self.rdkit_mol.RemoveBond(i, j)
            self.rdkit_mol.AddBond(i, j, self.bond_types[bond_order])

        elif remove_bond is not None:
            i, j = remove_bond
            if self.rdkit_mol.GetBondBetweenAtoms(i, j) is not None:
                self.rdkit_mol.RemoveBond(i, j)

    def masked_log_probs_for_current_action_level(self, logits: np.array) -> np.array:
        """
        Given an np.array of `logits` masks infeasible actions, determined
        by `current_action_mask`, and returns normalized log probabilities.
        """
        mask = self.current_action_mask
        logits[mask] = np.NINF
        with np.errstate(divide='ignore'):
            log_probs = np.log(softmax(logits))
        return log_probs

    def take_action(self, action: int):
        """
        Takes an action on the current action level and updates everything accordingly (see inline comments).
        Note that the updates are performed in-place!
        """
        assert not self.synthesis_done, "Taking action on already terminated design. No no!"

        assert self.current_action_mask[action] == 0, \
            f"Trying to take action {action} on level {self.current_action_level}, but it is set to infeasible"

        next_level = 0
        if self.current_action_level == 0:
            if action == 0:  # terminate
                self.synthesis_done = True
                self.finalize()
            elif 1 <= action < self.pick_existing_atoms_start_action_idx_lvl_0:  # create a new atom
                self.atoms = np.append(self.atoms, action)
                # add a row and column for the new atom
                self.bonds = np.pad(self.bonds, [(0, 1), (0, 1)], mode='constant', constant_values=0)
                new_atom_idx = len(self.atoms) - 1
                self.bonds[0, new_atom_idx] = self.bonds[
                    new_atom_idx, 0] = self.virtual_bond_idx  # Connect with virtual atom
                self.update_rdkit_mol(new_atom=action)
                self.update_topological_distance_matrix(new_atom_created=True)
                self.history.append(int(action))
                # Save base atom index from level 0.
                self.base_atom_idx = new_atom_idx
                next_level = 1
            else:  # already existing atom was picked as first atom
                next_level = 1



        elif self.current_action_level == 1:

            # Expanded action space for level 1:

            # Block 1 (indices [0, new_atom_action_count - 1]): New atom creation.

            # Block 2 (indices starting at new_atom_action_count): Existing bond modification.

            new_atom_action_count = len(self.vocabulary_atom_idcs)

            # existing bond actions relate only to real atoms (excluding the virtual atom).

            existing_bond_action_count = len(self.atoms) - 1

            total_actions = new_atom_action_count + existing_bond_action_count

            self.current_action_mask = np.zeros(total_actions, dtype=bool)

            # Compute remaining valence for real atoms only.

            # Note: self.atoms[0] is the virtual atom, so we consider atoms[1:]

            if len(self.atoms) > 1:

                real_atom_valence = np.array([self.vocabulary_valence[x] for x in self.atoms[1:]])

                # For real atoms, consider only bonds among real atoms via self.bonds[1:, 1:].

                real_atom_bonds_sum = self.bonds[1:, 1:].sum(axis=1)

                real_atom_valence_remaining = real_atom_valence - real_atom_bonds_sum

            else:

                real_atom_valence_remaining = np.array([])

            # Mask new atom creation actions.

            # self.atom_feasibility_mask is assumed to be a boolean array for new atom candidates.

            self.current_action_mask[:new_atom_action_count] = np.array(self.atom_feasibility_mask)

            # Refine the mask based on the remaining valence of candidate atoms.

            for idx in range(new_atom_action_count):

                # Candidate real atom index is idx + 1 (because index 0 is virtual).

                if idx < len(real_atom_valence_remaining):

                    if real_atom_valence_remaining[idx] < 1:
                        self.current_action_mask[idx] = 1

            # Mask for existing bond modification actions.

            # Loop over candidate atoms (real atoms excluding the current last atom) for modification.

            for idx in range(existing_bond_action_count):

                # For the last real atom, we prevent re-selecting it.

                if idx == (len(self.atoms) - 2):

                    self.current_action_mask[new_atom_action_count + idx] = 1

                # If there is an existing bond from the most recently added atom to candidate atom (idx+1), unmask that option.

                elif self.bonds[len(self.atoms) - 1, idx + 1] > 0:

                    self.current_action_mask[new_atom_action_count + idx] = 0

                else:

                    self.current_action_mask[new_atom_action_count + idx] = 1

            # Process the action.

            if action < new_atom_action_count:

                # Block 1: New atom creation.

                # Append the atom (action represents an atom type or index from the vocabulary).

                self.atoms = np.append(self.atoms, action)

                # Expand the bonds matrix to include the new atom.

                self.bonds = np.pad(self.bonds, [(0, 1), (0, 1)],

                                    mode='constant', constant_values=0)

                new_atom_idx = len(self.atoms) - 1

                # Set up the bond between the virtual atom (index 0) and the new atom.

                self.bonds[0, new_atom_idx] = self.bonds[new_atom_idx, 0] = self.virtual_bond_idx

                self.update_rdkit_mol(new_atom=action)

                self.update_topological_distance_matrix(new_atom_created=True)

                self.history.append(int(action))

                # Record the new atom index for later existing bond modification.

                self.last_created_atom_idx = new_atom_idx

            else:

                # Block 2: Existing bond modification.

                # If last_created_atom_idx is available, use that; otherwise, compute based on the action offset.

                if hasattr(self, 'last_created_atom_idx'):

                    candidate_atom_idx = self.last_created_atom_idx

                else:

                    existing_bond_action_index = action - new_atom_action_count

                    candidate_atom_idx = existing_bond_action_index + 1  # offset to match self.atoms indexing

                # Set the selected bond using the base atom and the candidate atom.

                self.selected_bond = (self.base_atom_idx, candidate_atom_idx)

                self.history.append(int(action))

            next_level = 2

        elif self.current_action_level == 2:
            # Level 2: Set or modify the bond order.
            if hasattr(self, 'selected_bond'):
                # Block 2: Using previously selected existing bond.
                atom_a_idx, atom_b_idx = self.selected_bond
            else:
                # Block 1: Use base atom from level 0 and the new atom from level 1.
                atom_a_idx = self.base_atom_idx
                atom_b_idx = self.last_created_atom_idx

            if action < self.maximum_bond_order:
                # Increase or set bond order: action index 0 means order 1, index 1 means order 2, etc.
                new_order = action + 1
                print(f"DEBUG: Bonding atoms at indices {atom_a_idx} and {atom_b_idx} with new order {new_order}")
                self.bonds[atom_a_idx, atom_b_idx] = self.bonds[atom_b_idx, atom_a_idx] = new_order
                # RDKit atoms exclude the virtual atom, so subtract 1 from indices.
                self.update_rdkit_mol(set_bond=(atom_a_idx - 1, atom_b_idx - 1, new_order))
            else:
                # Bond reduction action: second half of level 2.
                reduction = action - self.maximum_bond_order + 1
                current_order = self.bonds[atom_a_idx, atom_b_idx]
                new_order = max(0, current_order - reduction)
                print(
                    f"DEBUG: Reducing bond between atoms at indices {atom_a_idx} and {atom_b_idx} from order {current_order} to {new_order}")
                if new_order > 0:
                    self.bonds[atom_a_idx, atom_b_idx] = self.bonds[atom_b_idx, atom_a_idx] = new_order
                    self.rdkit_mol.RemoveBond(atom_a_idx - 1, atom_b_idx - 1)
                    self.update_rdkit_mol(set_bond=(atom_a_idx - 1, atom_b_idx - 1, new_order))
                else:
                    self.bonds[atom_a_idx, atom_b_idx] = self.bonds[atom_b_idx, atom_a_idx] = 0
                    self.rdkit_mol.RemoveBond(atom_a_idx - 1, atom_b_idx - 1)
            self.history.append(int(action))
            self.current_action_level = 0
            if hasattr(self, 'selected_bond'):
                del self.selected_bond

        self.current_action_level = next_level
        self.update_action_mask()

    def finalize(self, assert_feasible: bool = False):
        """
        Called when terminating. Creates a corresponding RDKit molecule and SMILES, as well as makes some premilinary
        checks.
        """
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
        """
        Checks whether the current molecule is feasible, i.e., if only atoms are there that are allowed
        and whether all bonds are consistent
        """
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
        """
        @Deprecated
        Returns the representation of the current molecule as
        rdkit's RWMol
        """
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
        bonds = self.bonds[1:, 1:]  # disregard virtual atom
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
        """
        Returns the current molecule as a SMILES string.
        """
        return Chem.MolToSmiles(self.rdkit_mol)

    # ---- Implementation of abstract methods from `BaseTrajectory`
    @staticmethod
    def init_batch_from_instance_list(config: MoleculeConfig, instances: List[int], network: nn.Module, device: torch.device):
        """
        An instance is given by the first atom placed on the molecule, so 1 (C), 2 (N) or 3 (O)
        """
        return [MoleculeDesign(config=config, initial_atom=atom) for atom in instances]

    @staticmethod
    def log_probability_fn(trajectories: List['MoleculeDesign'], network: nn.Module) -> List[np.array]:
        """
        Given a list of trajectories and a policy network,
        returns a list of numpy arrays, each having length num_actions, where each numpy array is a log-probability
        distribution over the next action level.

        Parameters:
            trajectories [List[BaseTrajectory]]
            network [torch.nn.Module]: Policy network
        Returns:
            List of numpy arrays, where i-th entry corresponds to the log-probabilities for i-th trajectory.
        """
        log_probs_to_return: List[np.array] = []
        network.eval()
        with torch.no_grad():
            batch = MoleculeDesign.list_to_batch(molecules=trajectories, device=network.device)
            batch_logits_per_level = list(network(batch))
            for lvl in range(3):
                batch_logits_per_level[lvl] = batch_logits_per_level[lvl].cpu().numpy()

            for i, mol in enumerate(trajectories):
                # get logits for this molecule and corresponding level
                logits = batch_logits_per_level[mol.current_action_level][i]
                # Now we need to trim the padding corresponding to current action level
                if mol.current_action_level != 2:
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
        """
        Returns number of current _feasible_ actions.
        """
        return int((1 - self.current_action_mask).sum())

    @staticmethod
    def list_to_batch(molecules: List['MoleculeDesign'], device: torch.device = None,
                      include_feasibility_masks: bool = False) -> dict:
        """
        Given a list of molecule designs, prepares a batch that can be passed through the network.
        In the following, when referring to `number of atoms`, we _include_ the virtual atom.
        The batch is given as a dictionary with the following keys and values:
        * "level_idx": This is "0" for level 0, "1" for level 1 or "2" for level 2. It is used to mark the virtual atom to inform
            the network what decision to make.
        * "picked_atom_mhe": A zero vector of length <max num atoms> with a 1 for the index of the
            atom that was picked/created at level 0, and a 2 for the index of the atom that was picked at level 1.
        * "num_atoms": torch.LongTensor of shape (batch_size,), which holds the number of atoms for each molecule
            (including virtual)
        * "atoms": torch.LongTensor of shape (batch_size, <max num atoms>) containing the
            indices (including virtual=0) of the atoms present in a molecule.
            We pad with 'num atoms in vocab + 1' to the maximum number of atoms in the batch.
        * "atoms_degree": torch.LongTensor of shape (batch_size, <max num atoms>), where an entry corresponds to the
            degree of the atom, i.e., to how many other atoms it is connected (excluding virtual).
             Hence, we pad with 'max possible valence + 1' to the maximum number of atoms in the batch (an atom can be connected to at most 6
             other atoms).
             Note that the virtual atom is also included here, but it is later not used in the network.
        * "bonds": torch.LongTensor of shape (batch_size, <max num atoms>, <max num atoms>), indicating the connection
            between atoms. We pad both columns and rows with 'virtual bond idx + 1' to the maximum number of atoms in the batch.
        * "topological_distance": torch.LongTensor of shape (batch_size, <max num atoms>, <max num atoms>), indicating the topological
            distance (i.e., smallest number of bonds) between atoms.
            We pad both columns and rows with 103 (100 max atoms overall + 1 for distance to virtual + 1 for infinity
             distance + 1 for padding) to the maximum number of atoms in the batch.
        * "additive_padding_attn_mask": torch.FloatTensor of shape (batch_size, <max num atoms>, <max num atoms>), which
            is zero everywhere except at the padding positions, where it's -inf. For numerical reasons, the diagonal is 0,
            so that still every padding token can attend to itself. This padding mask will later be added to the learned
            attention mask.

        if `include_feasibility_masks` is set to True, we also return
        """
        atoms_padding_idx = len(molecules[0].vocabulary_atom_idcs) + 1
        degree_padding_idx = max(molecules[0].vocabulary_valence) + 1
        bond_padding_idx = MoleculeDesign.virtual_bond_idx + 1
        distance_padding_idx = MoleculeDesign.maximum_num_atoms_overall + 3

        device = torch.device("cpu") if device is None else device
        num_atoms = [len(mol.atoms) for mol in molecules]
        max_num_atoms = max(num_atoms)

        batch_level_idx = [mol.current_action_level == 0 for mol in molecules]

        batch_picked_atom_mhe = np.zeros(((len(molecules), max_num_atoms)), dtype=int)
        ex_pick_idx_start = molecules[0].pick_existing_atoms_start_action_idx_lvl_0  # alias
        for i, mol in enumerate(molecules):
            if mol.current_action_level == 0:
                pass  # nothing has been chosen yet
            elif mol.current_action_level == 1:
                # an atom has been picked/created at level 0
                atom_picked_on_lvl_0 = len(mol.atoms) - 1 if mol.history[-1] < ex_pick_idx_start else mol.history[-1] - ex_pick_idx_start + 1
                batch_picked_atom_mhe[i, atom_picked_on_lvl_0] = 1
            elif mol.current_action_level == 2:
                # an atom has been picked/created at level 0
                atom_picked_on_lvl_0 = len(mol.atoms) - 1 if mol.history[-2] < ex_pick_idx_start else mol.history[-2] - ex_pick_idx_start + 1
                batch_picked_atom_mhe[i, atom_picked_on_lvl_0] = 1
                # an atom has been picked at level 1
                atom_picked_on_lvl_1 = mol.history[-1] + 1
                batch_picked_atom_mhe[i, atom_picked_on_lvl_1] = 2

        batch_atoms = np.stack([
                np.concatenate((mol.atoms, np.full(max_num_atoms - num_atoms[i], fill_value=atoms_padding_idx, dtype=int)))
                for i, mol in enumerate(molecules)
        ])

        batch_atoms_degree = np.stack([
                np.concatenate((
                    (mol.bonds > 0).sum(axis=1) - 1,  # implicity, we have a +1 here which we need to subtract, as this includes the connection to the virtual node
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
            # Fill the diagonal of the bonds with `8`, which will always embed to zeros in the network. This is to
            # not perturb the attention of an atom to itself.
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
            np.fill_diagonal(mask, 0)  # let padded tokens attend to themselves
            additive_padding_masks.append(mask)
        batch_additive_padding_attn_mask = np.stack(additive_padding_masks)

        return_dict = dict(
            level_idx=torch.tensor(batch_level_idx, dtype=torch.long, device=device),  # (B,)
            picked_atom_mhe=torch.from_numpy(batch_picked_atom_mhe).long().to(device),  # (B, <max num atoms>)
            num_atoms=torch.tensor(num_atoms, dtype=torch.long, device=device),  # (B,)
            atoms=torch.from_numpy(batch_atoms).long().to(device),  # (B, <max num atoms>)
            atoms_degree=torch.from_numpy(batch_atoms_degree).long().to(device),  # (B, <max num atoms>)
            bonds=torch.from_numpy(batch_bonds).long().to(device),  # (B, <max num atoms>, <max num atoms>)
            topological_distance=torch.from_numpy(batch_topological_distance).long().to(device),  # (B, <max num atoms>, <max num atoms>)
            additive_padding_attn_mask=torch.from_numpy(batch_additive_padding_attn_mask).float().to(device),  # (B, <max num atoms>, <max num atoms>)
        )

        if include_feasibility_masks:
            # This is only relevant for training. We prepare the masks with respect to action feasibility for all
            # action levels, also masking padded atoms. For molecules which are at a different action level, we also
            # return masks, but these are 0 everywhere (so all feasible). Account for this during training in
            # cross-entropy loss with `ignore_index`.
            feasibility_masks_per_level = []
            num_actions_per_level_and_mol = [
                [mol.pick_existing_atoms_start_action_idx_lvl_0 + len(mol.atoms) - 1 for mol in molecules],  # lvl 0
                [len(mol.atoms) - 1 for mol in molecules],  # lvl 1
                [molecules[0].maximum_bond_order] * len(molecules)  # lvl 2
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
                        for i, mol in enumerate(molecules)])
                    ).bool().to(device)
                )

            return_dict["feasibility_mask_level_zero"] = feasibility_masks_per_level[0]  # (B, <num atoms in vocab + max_num_atoms>)
            return_dict["feasibility_mask_level_one"] = feasibility_masks_per_level[1]  # (B, <max_num_atoms - 1>)
            return_dict["feasibility_mask_level_two"] = feasibility_masks_per_level[2]  # (B, <max_num_atoms - 1>)

        return return_dict

    @staticmethod
    def batch_to_device(batch: dict, device: torch.device):
        """
        Takes batch as returned from `list_to_batch` and moves it onto the given device.
        """
        return {k: v.to(device) for k, v in batch.items()}

    @staticmethod
    def get_c_chains(config: MoleculeConfig) -> List['MoleculeDesign']:
        """
        Returns list of designs which to use as a starting point.
        These are chains of C-atoms of length 1 to max number of atoms - 1.
        """
        carbon_atom_idx = list(config.atom_vocabulary.keys()).index("C") + 1
        instance_list = []
        for num_c_to_add in range(min(config.max_num_atoms - 1, config.start_c_chain_max_len)):
            mol = MoleculeDesign(config, initial_atom=1)
            for i in range(num_c_to_add):
                mol.take_action(carbon_atom_idx)  # choose to add C at level 0
                mol.take_action(len(mol.atoms) - 3)  # attach to last added atom
            instance_list.append(mol)
        return instance_list

    @staticmethod
    def get_single_atom_molecules(config: MoleculeConfig, repeat: int = 1) -> List['MoleculeDesign']:
        """
        Returns molecules with only a single C, N (if nitrogen is allowed), or O atom. These molecules
        are repeated `repeat` times.
        """
        atoms = []
        for i, atom in enumerate(config.atom_vocabulary.keys()):
            if config.atom_vocabulary[atom]["allowed"]:
                atoms.append(i + 1)

        return MoleculeDesign.init_batch_from_instance_list(config, atoms * repeat, None, None)

    @staticmethod
    def random_atom_order_in_smiles(smiles:str) -> str:
        """
        Given a SMILES string, returns an equivalent SMILES string with random atom order. Helpful for
        data augmentation.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES input.")

        # Number of atoms
        num_atoms = mol.GetNumAtoms()

        # Create a random permutation of atom indices
        atom_indices = list(range(num_atoms))
        random.shuffle(atom_indices)

        # Renumber the atoms according to the shuffled indices
        reordered_mol = Chem.RenumberAtoms(mol, atom_indices)

        # Convert the reordered molecule back to SMILES.
        # Use canonical=False to preserve non-canonical (i.e., randomized) ordering.
        return Chem.MolToSmiles(reordered_mol, isomericSmiles=True, canonical=False)

    @staticmethod
    def from_smiles(config: MoleculeConfig, smiles: str, do_finish=False, compare_smiles=False) -> 'MoleculeDesign':
        mol = Chem.MolFromSmiles(smiles)
        Chem.SanitizeMol(mol)
        return MoleculeDesign.from_rdkit_mol(config, mol, smiles, do_finish, compare_smiles)

    @staticmethod
    def from_rdkit_mol(config: MoleculeConfig, rdkit_mol: Chem.RWMol, smiles: str, do_finish=True, compare_smiles=True) -> 'MoleculeDesign':
        """
        Creates an instance of `MoleculeDesign` from an RDKit molecule.
        """
        Chem.Kekulize(rdkit_mol)
        atoms = rdkit_mol.GetAtoms()
        atom_idcs_for_design = []  # idcs to choose at level 0 to add the corresponding atom
        adjacency_matrix: np.ndarray = Chem.rdmolops.GetAdjacencyMatrix(rdkit_mol, useBO=True)

        atomic_num_to_atom_idx = dict()
        for i, atom_name in enumerate(config.atom_vocabulary.keys()):
            k = config.atom_vocabulary[atom_name]["atomic_number"]
            if "formal_charge" in config.atom_vocabulary[atom_name]:
                k = f"{k}_{config.atom_vocabulary[atom_name]['formal_charge']}"
            if "chiral_tag" in config.atom_vocabulary[atom_name]:
                k = f"{k}@{config.atom_vocabulary[atom_name]['chiral_tag']}"
            atomic_num_to_atom_idx[k] = i + 1  # account for 0 = Terminate action

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

        # Convert adjacency matrix to int to not track .5-bonds
        #adjacency_matrix = adjacency_matrix.astype(int)

        # Start by creating the design and setting the first atom as the start atom
        design = MoleculeDesign(config, atom_idcs_for_design[0])
        # We now iterate over each atom.
        for i in range(1, len(atom_idcs_for_design)):
            atom_to_add = atom_idcs_for_design[i]

            # We now want to get the most recent index to which the new atom is bonded. This is used as the initial
            # bond of the new atom.
            atom_is_placed = False
            #for j in range(i-1, -1, -1):  # Loop backwards
            for j in range(0, i):
                desired_bond_order = adjacency_matrix[i, j]
                if desired_bond_order > 0:
                    if not atom_is_placed:
                        design.take_action(atom_to_add)
                        atom_is_placed = True
                    else:
                        design.take_action(1 + len(config.atom_vocabulary.keys()) + len(design.atoms) - 2)

                    design.take_action(j)
                    design.take_action(int(desired_bond_order - 1))

        if do_finish:
            design.take_action(0)
            if compare_smiles:
                assert Chem.CanonSmiles(design.smiles_string) == Chem.CanonSmiles(smiles), f"Converted: {Chem.CanonSmiles(design.smiles_string)}, RDKit: {Chem.CanonSmiles(smiles)}"
            design.assert_feasible()

        return design