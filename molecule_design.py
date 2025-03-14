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
        self._smiles_string = None  # Use _smiles_string as the internal attribute
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

    # Fix the update_action_mask method
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

        if self.current_action_level == 0:
            # Same as before
            self.current_action_mask = np.zeros(len(self.vocabulary_atom_idcs) + len(self.atoms), dtype=bool)
            self.current_action_mask[1:ex_action_idx] = self.atom_feasibility_mask
            if (self.upper_limit_atoms is not None and len(self.atoms) - 1 == self.upper_limit_atoms) or \
                    (not np.any(atom_valence_remaining[1:])):
                self.current_action_mask[1:ex_action_idx] = 1
            self.current_action_mask[ex_action_idx:][np.where(atom_valence_remaining[1:] <= 0)] = 1
            bond_indicator = np.zeros_like(self.bonds[1:, 1:])
            bond_indicator[np.where(self.bonds[1:, 1:] == 0)] = 1
            np.fill_diagonal(bond_indicator, 0)
            has_free_nonneighbor = np.matmul(bond_indicator, (atom_valence_remaining[1:] > 0)[:, None]).squeeze()
            self.current_action_mask[ex_action_idx:][np.where(has_free_nonneighbor == 0)] = 1

        elif self.current_action_level == 1:
            self.current_action_mask = np.zeros(len(self.atoms) - 1, dtype=bool)
            atom_picked_on_lvl_0 = len(self.atoms) - 2 if self.history[-1] < ex_action_idx else self.history[
                                                                                                    -1] - ex_action_idx
            self.current_action_mask[atom_picked_on_lvl_0] = 1  # Can't bond to itself
            self.current_action_mask[np.where(atom_valence_remaining[1:] < 1)] = 1  # Atoms with no free valence

            # REMOVE THIS LINE to allow cycle formation:
            # self.current_action_mask[np.where(self.bonds[atom_picked_on_lvl_0 + 1, 1:] > 0)] = 1

        elif self.current_action_level == 2:
            # Same as before
            self.current_action_mask = np.zeros(self.maximum_bond_order, dtype=bool)
            atom_picked_on_lvl_0 = len(self.atoms) - 2 if self.history[-2] < ex_action_idx else self.history[
                                                                                                    -2] - ex_action_idx
            atom_picked_on_lvl_1 = self.history[-1]
            max_bond_order = min(atom_valence_remaining[atom_picked_on_lvl_0 + 1],
                                 atom_valence_remaining[atom_picked_on_lvl_1 + 1])
            self.current_action_mask[int(max_bond_order):] = 1

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

    def update_rdkit_mol(self, new_atom: Optional[int] = None, set_bond: Optional[Tuple[int, int, int]] = None):
        """
        Updates the RDKit mol by either adding a new atom or setting the bond order between two atoms.
        Parameters:
            new_atom: Atom to add as index in vocabulary
            set_bond: Tuple of ints (i, j, bond order), where i,j start from 0 (so we do not count virtual atom)
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
            self.rdkit_mol.AddBond(i, j, self.bond_types[bond_order])

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
        Takes an action on the current action level and updates everything accordingly.
        """
        assert not self.synthesis_done, "Taking action on already terminated design. No no!"
        assert self.current_action_mask[action] == 0, \
            f"Trying to take action {action} on level {self.current_action_level}, but it is set to infeasible"
        ex_action_idx = self.pick_existing_atoms_start_action_idx_lvl_0  # alias

        if self.current_action_level == 0:
            self.history.append(action)
            if action == 0:
                self.synthesis_done = True
            elif action < ex_action_idx:
                # Add a new atom
                self.atoms = np.append(self.atoms, action)
                # Add a new row and column to bonds matrix
                self.bonds = np.pad(self.bonds, ((0, 1), (0, 1)), 'constant', constant_values=0)
                # Set connections to virtual atom
                self.bonds[0, -1] = 7
                self.bonds[-1, 0] = 7
                self.current_action_level = 1
            else:
                # Pick an existing atom
                self.current_action_level = 1
            self.update_action_mask()

        elif self.current_action_level == 1:
            self.history.append(action)
            self.second_atom = action + 1  # The second atom is 1-indexed in atoms array
            self.current_action_level = 2
            self.update_action_mask()

        elif self.current_action_level == 2:
            # Create a bond with the specified order between the atom selected on level 0 and the atom selected on level 1.
            self.history.append(action)

            # Determine which atoms to bond
            if self.history[-3] < ex_action_idx:
                # If we added a new atom on level 0
                atom_picked_on_lvl_0 = len(self.atoms) - 1  # Last atom (1-indexed)
            else:
                # If we selected an existing atom on level 0
                atom_picked_on_lvl_0 = self.history[-3] - ex_action_idx + 1  # Convert action to atom index (1-indexed)

            atom_picked_on_lvl_1 = self.history[-2] + 1  # Convert from 0-indexed to 1-indexed
            bond_order = action + 1  # Bond order is 1-indexed

            print(
                f"DEBUG: Bonding atoms at indices {atom_picked_on_lvl_0} and {atom_picked_on_lvl_1} with order {bond_order}")

            # Update bonds matrix
            self.bonds[atom_picked_on_lvl_0, atom_picked_on_lvl_1] = bond_order
            self.bonds[atom_picked_on_lvl_1, atom_picked_on_lvl_0] = bond_order

            # Reset to level 0
            self.current_action_level = 0
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
            self._smiles_string = Chem.MolToSmiles(self.rdkit_mol)

            if self.smiles_string == "C":
                self.infeasibility_flag = True

    def is_connected_without_bond(self, atom1_idx: int, atom2_idx: int) -> bool:
        """
        Checks if the molecular graph would remain connected if the bond between
        atom1_idx and atom2_idx were removed.

        Args:
            atom1_idx: Index of first atom (including virtual atom at index 0)
            atom2_idx: Index of second atom (including virtual atom at index 0)

        Returns:
            bool: True if graph would remain connected, False otherwise
        """
        # Validate indices
        if (atom1_idx <= 0 or atom2_idx <= 0 or
                atom1_idx >= len(self.atoms) or atom2_idx >= len(self.atoms)):
            return True  # Invalid indices, nothing to check

        # If there's no bond, nothing to check
        if self.bonds[atom1_idx, atom2_idx] == 0:
            return True

        # Get number of real atoms
        num_real_atoms = len(self.atoms) - 1

        # If only 2 real atoms, removing any bond disconnects the molecule
        if num_real_atoms <= 2:
            return False

        # Create a working copy of the bond matrix (only real atoms)
        bonds_copy = self.bonds[1:, 1:].copy()
        # bonds_copy = self.bonds.copy()

        # Adjust indices (account for virtual atom at 0)
        real_idx1 = atom1_idx - 1
        # real_idx1 = atom1_idx
        real_idx2 = atom2_idx - 1
        # real_idx2 = atom2_idx

        # Remove the bond
        bonds_copy[real_idx1, real_idx2] = 0
        bonds_copy[real_idx2, real_idx1] = 0

        # Check connectivity with BFS
        visited = np.zeros(num_real_atoms, dtype=bool)
        queue = [0]  # Start from first real atom
        visited[0] = True

        while queue:
            current = queue.pop(0)
            for neighbor in range(num_real_atoms):
                if bonds_copy[current, neighbor] > 0 and not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        # Connected if all nodes are visited
        return np.all(visited)

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

    def to_rdkit_mol(self):
        """Converts the Design to an RDKit molecule."""
        mol = Chem.RWMol()
        atom_indices_in_rdkit = []

        # Add atoms
        for atom_idx in self.atoms[1:]:  # Skip virtual atom
            # Create proper atom objects
            atom_type_name = list(self.config.atom_vocabulary.keys())[atom_idx - 1]
            atomic_number = self.config.atom_vocabulary[atom_type_name]["atomic_number"]
            atom = Chem.Atom(atomic_number)
            rdkit_atom_idx = mol.AddAtom(atom)
            atom_indices_in_rdkit.append(rdkit_atom_idx)

        # Add bonds
        bonds = self.bonds[1:, 1:]  # Remove virtual atom connections
        for i in range(bonds.shape[0]):
            for j in range(i + 1, bonds.shape[1]):  # Only upper triangle to avoid duplicates
                if bonds[i, j] > 0 and bonds[i, j] != 7:  # Skip virtual bonds
                    # Add the bond with the correct type
                    bond_type = Chem.BondType.UNSPECIFIED
                    if bonds[i, j] == 1:
                        bond_type = Chem.BondType.SINGLE
                    elif bonds[i, j] == 2:
                        bond_type = Chem.BondType.DOUBLE
                    elif bonds[i, j] == 3:
                        bond_type = Chem.BondType.TRIPLE
                    elif bonds[i, j] == 4:
                        bond_type = Chem.BondType.AROMATIC

                    mol.AddBond(atom_indices_in_rdkit[i], atom_indices_in_rdkit[j], bond_type)

        # Ensure the molecule is valid
        try:
            Chem.SanitizeMol(mol)
        except:
            print("Warning: RDKit couldn't sanitize the molecule")

        return mol

    @property
    def smiles_string(self):
        """Returns the SMILES string of the molecule."""
        if self.synthesis_done:
            if self._smiles_string is None:
                try:
                    rdkit_mol = self.to_rdkit_mol()
                    self._smiles_string = Chem.MolToSmiles(rdkit_mol)
                except:
                    print("Error generating SMILES string")
                    return None
            return self._smiles_string
        return None

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
                if mol.current_action_level == 2:
                    # Modified: ensure we have enough logits for expanded action space
                    if logits.shape[0] < 2 * mol.maximum_bond_order:
                        # Need to handle existing model that doesn't output expanded action space
                        # This will be needed during transition to new model
                        expanded_logits = np.full(2 * mol.maximum_bond_order, -np.inf)
                        expanded_logits[:logits.shape[0]] = logits
                        logits = expanded_logits
                elif mol.current_action_level != 2:
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
    def from_rdkit_mol(config: MoleculeConfig, rdkit_mol: Chem.RWMol, smiles: str, do_finish=True,
                       compare_smiles=True) -> 'MoleculeDesign':
        """
        Creates an instance of `MoleculeDesign` from an RDKit molecule.
        """
        Chem.Kekulize(rdkit_mol)
        atoms = rdkit_mol.GetAtoms()
        atom_idcs_for_design = []  # idcs to choose at level 0 to add the corresponding atom
        adjacency_matrix: np.ndarray = Chem.rdmolops.GetAdjacencyMatrix(rdkit_mol, useBO=True)

        print(f"RDKit adjacency matrix for {smiles}:")
        print(adjacency_matrix)

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

        # Start by creating the design with the first atom
        design = MoleculeDesign(config, atom_idcs_for_design[0])

        # Track which bonds have been added
        bonds_added = set()

        # First add all atoms with a single bond to an existing atom
        for i in range(1, len(atom_idcs_for_design)):
            atom_to_add = atom_idcs_for_design[i]

            # Find a bond to an existing atom
            connected = False
            for j in range(i):
                # Check if atoms i and j should be connected
                if adjacency_matrix[i, j] > 0:
                    # Add the atom
                    design.take_action(atom_to_add)
                    # Connect to atom j
                    design.take_action(j)
                    # Set bond order
                    design.take_action(int(adjacency_matrix[i, j]) - 1)

                    # Mark this bond as added
                    bonds_added.add((i, j))
                    bonds_added.add((j, i))
                    connected = True
                    break

            if not connected:
                # Unlikely case - isolated atom
                design.take_action(atom_to_add)

        # Add any remaining bonds (cycle-closing bonds)
        for i in range(len(atom_idcs_for_design)):
            for j in range(i + 1, len(atom_idcs_for_design)):
                if adjacency_matrix[i, j] > 0 and (i, j) not in bonds_added:
                    # Need to select atom i using the existing atom selection action
                    ex_action_idx = 1 + len(config.atom_vocabulary.keys())
                    select_atom_i_action = ex_action_idx + i

                    # Add the bond
                    design.take_action(select_atom_i_action)
                    design.take_action(j)
                    design.take_action(int(adjacency_matrix[i, j]) - 1)

                    # Mark bond as added
                    bonds_added.add((i, j))
                    bonds_added.add((j, i))

        # Finish the molecule
        if do_finish:
            design.take_action(0)

            # Verify the structure
            if compare_smiles:
                try:
                    design_smiles = Chem.CanonSmiles(design.smiles_string)
                    expected_smiles = Chem.CanonSmiles(smiles)
                    if design_smiles != expected_smiles:
                        print(f"WARNING: SMILES mismatch - Generated: {design.smiles_string}, Expected: {smiles}")
                except:
                    print("ERROR: Failed to compare SMILES strings")

        return design