import copy

import numpy as np
import rdkit.Chem
import torch
from rdkit import Chem

from config import MoleculeConfig
from core.abstracts import BaseTrajectory
from core.utils import softmax
from model.molecule_transformer import MoleculeTransformer

from typing import Optional, List, Tuple


class MoleculeDesign(BaseTrajectory):
    """
    Environment for the molecular design.
    Actions are chosen hierarchically in three levels:
        - Level 0: Choose to terminate (0), add C atom (1), add N atom (2), add O atom (3), add positively charged nitrogen (4), or increase existing bond (5)
        - Level 1: If not terminating, pick an atom A on which the action should be performed. If at level 0s it was
            decided to add an atom, we are done. Else (increase bond), transition to level 2:
        - Level 2: Pick another atom B to increase the bond by 1 between A and B.

    Level 0 and 1 are predicted simultaneously by the network, while for level 2 we mark the chosen atom for the network.

    Atom types are specified in the config under `atom_vocabulary`. Indexing starts at 1. Index 0 is for virtual atom.
    - Index 0: Virtual Atom, which is connected (with special bond order) to every other atom (and vice versa).

    We store all actions in a history, which is a list of lists. Each list has either 2 elements and corresponds to
    the actions taken at level 0 and 1, or only one element corresponding to level 2. For example, a history
    (with vocabulary C,N,O, so "increase bond" is at index 4 for action level 0)
    [[1, 1], [4, 2], [1], [0, -1]] means that first a C atom is added to the atom at index 1, then the atom at
    index 2 increases its bond with the atom at index 1, and then we terminate.
    """

    def __init__(self, config: MoleculeConfig, initial_atom: int):
        """
        Parameters:
            config [MoleculeConfig]: Config
            initial_atom [int]: We always start with already one atom in the molecule to be able to diversify
                the starting point for the network. Must be 1,2,3, 4.
            minimal_init [bool]: If True, we return directly after setting the initial atom
        """
        self.config = config
        self.atom_vocabulary = self.config.atom_vocabulary
        self.vocabulary_atom_idcs = list(range(1, len(self.atom_vocabulary) + 1))  # [1, ..., num of atoms in vocab]
        self.vocabulary_atom_names = list(self.atom_vocabulary.keys())
        self.vocabulary_valence = [-1] + [self.atom_vocabulary[x]["valence"] for x in self.vocabulary_atom_names]  # have an entry "-1" for the first virtual atom
        self.atom_feasibility = [not self.atom_vocabulary[x]["allowed"] for x in self.vocabulary_atom_names]  # if not allowed, then feasbility mask must be set to True

        # Extract relevant indexing information that depends on the size of the vocabulary.
        self.increase_bond_action_idx = len(self.vocabulary_atom_idcs) + 1
        self.virtual_bond_idx = max(self.vocabulary_valence) + 1  # index for the virtual bond between virtual atom and other atoms. Is one more than the maximum valence.
        self.oxygen_atom_idx = self.vocabulary_atom_names.index("O") + 1

        # Extract information about minimum number of atoms
        self.min_required_atoms = []
        for i, atom_name in enumerate(self.vocabulary_atom_names):
            if "min_atoms" in self.atom_vocabulary[atom_name] and self.atom_vocabulary[atom_name]["min_atoms"] > 0:
                self.min_required_atoms.append((i + 1, self.atom_vocabulary[atom_name]["min_atoms"]))  # (tuple of atom index, min num atoms required)

        self.upper_limit_atoms = self.config.max_num_atoms
        assert not self.atom_feasibility[initial_atom - 1] and initial_atom in self.vocabulary_atom_idcs, f"Initial atom must be in {self.vocabulary_atom_idcs} and set to allowed in config."
        self.initial_atom = initial_atom

        # Keeps track of all atoms present (including virtual atom)
        self.atoms = np.array([0, initial_atom], dtype=np.uint8)

        # Keeps track of all bonds with order. Is a matrix of shape (len(atoms), len(atoms)), where the (i,j)-th entry
        # indicates connection of i-th atom with j-th atom. Note that the virtual atom has a bond of order 7 with
        # all other atoms.
        self.bonds = np.zeros((2, 2), dtype=np.uint8)
        self.bonds[0, 1] = self.bonds[1, 0] = self.virtual_bond_idx  # connect with virtual atom

        self.synthesis_done = False
        self.smiles_string: Optional[str] = None  # Is set after synthesis is done
        self.current_objective = float("-inf")

        # Current action level.
        self.current_action_level = 0  # start by choosing <terminate>/<add atom>/<bond increase>
        # The action mask indicates before each action what is feasible at the current level.
        # It is set for level 0 and level 1 when the current action level is reset to 0.
        # When choosing to increase the bond order, it is also set for level 2.
        # It's a list of length 3, while the mask for level 2 can be None.
        # A `1` indicates that the action should be masked, i.e., cannot be taken.
        self.current_action_mask: Optional[List[Optional[np.array]]] = None
        # Keeps track of the actions for each level, before saving it to the history (i.e., returning to action index 0)
        self.actions_taken: List[int] = []
        # At action level 0, the network can predict logits for level 0 and 1. These need to be registered here.
        # For consistency, we also register actions at level 2.
        # They will be reset to `None` once action level 1 or 2 have been passed.
        self.registered_logits: Optional[List[np.array]] = None
        # History is a list of `actions_taken` above, indicating how you get from the initial atom to the current
        # molecule. See class docstring for an example.
        self.history: List[List[int]] = []

        self.objective: Optional[float] = None
        # Synthetic accessibility score, obtained from RDKit, ranging from 1 [easiest] to 10 [hardest]
        self.sa_score: float = 0.

        # Set this to True if anything goes wrong and the molecule will always evaluate to objective -inf
        self.infeasibility_flag: bool = False
        self.rdkit_mol: Optional[rdkit.Chem.RWMol] = None

        self.reset_actions_to_level_0()

    def reset_actions_to_level_0(self):
        """
        Called after passing through the full hierarchy. Resets the action level to 0 and re-computes the action mask.
        Also adds the actions_taken (if any) to the history.
        """
        self.current_action_level = 0
        self.registered_logits = None
        if len(self.actions_taken):
            self.history.append(self.actions_taken)
            self.actions_taken = []
        self.current_action_mask = [np.zeros(len(self.vocabulary_atom_idcs) + 2, dtype=bool)]  # <terminate>/<add atom>/<increase bond>
        self.current_action_mask.append(None)  # <pick atom> (exclude virtual atom)
        self.current_action_mask.append(None)   # for level 2
        self.update_action_mask()

    def reset_actions_to_level_2(self):
        self.current_action_level = 2
        self.registered_logits = None
        self.history.append(self.actions_taken)
        self.actions_taken = []
        self.current_action_mask[0] = None
        self.current_action_mask[1] = None
        self.update_action_mask()

    def update_action_mask(self):
        """
        Depending on the current action level, updates the mask for each action level >= the current. Here, we take
        into account the valence of the present atoms.
        """
        if self.synthesis_done:
            self.current_action_mask = None
            return
        atom_valence = np.array([self.vocabulary_valence[x] for x in self.atoms])
        atom_valence_remaining = atom_valence - self.bonds[:, 1:].sum(axis=1)

        if self.current_action_level == 0:  # Directly set what to allow. Termination is always allowed.
            # Set all atoms to infeasible which cannot be chosen.
            self.current_action_mask[0][self.vocabulary_atom_idcs] = self.atom_feasibility

            # Check if we have reached the max number of atoms or if there is any valence remaining (skip virtual atom).
            # If not, only the terminate-action is possible.
            # If we are at the maximum number of atoms, we can only terminate or increase the bond
            if self.upper_limit_atoms is not None and len(self.atoms) - 1 == self.upper_limit_atoms:
                self.current_action_mask[0][1:-1] = 1
            # If there is no valence remaining, we can only terminate
            if not np.any(atom_valence_remaining[1:]):
                self.current_action_mask[0][1:] = 1
            # Else if there is only one atom left with valence, we cannot increase the bonds
            elif np.sum(atom_valence_remaining[1:] > 0) < 2:
                self.current_action_mask[0][-1] = 1

            # Special settings
            # Case 1: Do not allow oxygen to bond with another oxygen
            if self.config.disallow_oxygen_bonding and np.any(atom_valence_remaining[1:]):
                # If we are not allowed to bond oxygens with each other, we need to check if the only valence
                # remaining are oxygen atoms. If so, we are not allowed to add oxygen or increase the bonds.
                atom_idcs = np.where(atom_valence_remaining[1:] > 0)[0]  # account for virtual atom
                if len(atom_idcs) > 0:
                    if np.all(self.atoms[atom_idcs + 1] == self.oxygen_atom_idx):
                        self.current_action_mask[0][self.oxygen_atom_idx] = 1
                        self.current_action_mask[0][-1] = 1
            # Case 2: Do not allow rings.
            if self.config.disallow_rings and self.current_action_mask[0][-1] == 0:
                # We check if there are at least two atoms that have nonzero valence and are already bonded.
                # Otherwise we do not allow increasing bonds bonding.
                self.current_action_mask[0][-1] = 1
                for i in range(1, len(self.atoms)):
                    if atom_valence_remaining[i] > 0:
                        for j in range(i+1, len(self.atoms)):
                            if atom_valence_remaining[j] > 0 and self.bonds[i, j] > 0:
                                self.current_action_mask[0][-1] = 0  # allow increasing bonds
                                break
                    if self.current_action_mask[0][-1] == 0:
                        break

            # Decide for each atom if it can be picked at level 1. This is possible if it has free valence
            self.current_action_mask[1] = atom_valence_remaining[1:] < 1

        elif self.current_action_level == 1:
            if self.config.disallow_oxygen_bonding and self.actions_taken[0] == self.oxygen_atom_idx:
                # If the atom placed at level 0 is oxygen and we do not want bonds between oxygen atoms,
                # disallow choosing any other oxygen atom at level 1. Account for the virtual atom and the atom that
                # has just been added.
                self.current_action_mask[1][np.where(self.atoms[1:-1] == self.oxygen_atom_idx)] = 1

            if self.config.disallow_rings and self.actions_taken[0] == self.increase_bond_action_idx:
                # If we do not allow rings and have chosen to increase bond order,
                # then we also want to disallow picking atoms that do not share free valence
                # with another already bonded atom.
                has_free_valence_idcs = np.where(atom_valence_remaining[1:] > 0)[0]
                for atom_i in has_free_valence_idcs:
                    self.current_action_mask[1][atom_i] = 1  # at first, disallow
                    for atom_j in has_free_valence_idcs:
                        if atom_i != atom_j and self.bonds[atom_i + 1, atom_j + 1] > 0:
                            self.current_action_mask[1][atom_i] = 0  # allow
                            break

        elif self.current_action_level == 2:
            # At level 2, an atom can be picked if it has free valence and was not picked at level 1
            self.current_action_mask[2] = atom_valence_remaining[1:] < 1
            self.current_action_mask[2][self.history[-1][1]] = 1  # atom picked at level 1

            if self.config.disallow_oxygen_bonding and self.atoms[self.history[-1][1] + 1] == self.oxygen_atom_idx:
                # If we have chosen an O atom at level 1 to bond, then at level 2 disallow choosing another O
                self.current_action_mask[2][np.where(self.atoms[1:] == self.oxygen_atom_idx)] = 1

            if self.config.disallow_rings:
                # Prohibit choosing an atom that is not already bonded to the one chosen at level 1
                self.current_action_mask[2][np.where(self.bonds[self.history[-1][1] + 1, 1:] == 0)] = 1

    def masked_log_probs_for_current_action_level(self) -> np.array:
        """
        Assuming logits are registered, gets them for the current action index and masks infeasible actions, determined
        by `current_action_mask`.
        """
        logits = self.registered_logits[self.current_action_level]
        mask = self.current_action_mask[self.current_action_level]
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

        if self.current_action_level == 0:
            if action == 0:  # terminate
                self.synthesis_done = True
                self.finalize()
                self.actions_taken = [0, -1]
                self.reset_actions_to_level_0()
            elif 1 <= action <= len(self.vocabulary_atom_idcs):  # add atom
                self.atoms = np.append(self.atoms, action)
                # add a row and column for the new atom
                self.bonds = np.pad(self.bonds, [(0, 1), (0, 1)], mode='constant', constant_values=0)
                new_atom_idx = len(self.atoms) - 1
                self.bonds[0, new_atom_idx] = self.bonds[new_atom_idx, 0] = self.virtual_bond_idx  # Connect with virtual atom
                self.actions_taken = [action]
                self.current_action_level += 1
                self.update_action_mask()
            elif action == self.increase_bond_action_idx:  # increase bond
                self.actions_taken = [self.increase_bond_action_idx]
                self.current_action_level += 1
                self.update_action_mask()
        elif self.current_action_level == 1:
            self.actions_taken.append(action)
            if self.actions_taken[0] < self.increase_bond_action_idx:
                # action defines to which atom to bond the freshly added atom (we do not count virtual atom)
                new_atom_idx = len(self.atoms) - 1
                self.bonds[new_atom_idx, action + 1] = self.bonds[action + 1, new_atom_idx] = 1
                self.reset_actions_to_level_0()
            elif self.actions_taken[0] == self.increase_bond_action_idx:
                self.reset_actions_to_level_2()
            else:
                raise ValueError(f"Action at level 0 must be 1,2,3,4 when choosing atom at level 1. Is: {self.actions_taken[0]}")
        elif self.current_action_level == 2:
            # Increase the bond between the chosen atom and the atom picked at level 1.
            self.actions_taken = [action]
            atom_a = action
            atom_b = self.history[-1][1]
            assert atom_a != atom_b, f"Cannot bond atom {atom_a} with itself {atom_b}. {np.array([self.vocabulary_valence[x] for x in self.atoms]) - self.bonds[:, 1:].sum(axis=1)}"
            self.bonds[atom_a + 1, atom_b + 1] += 1
            self.bonds[atom_b + 1, atom_a + 1] += 1
            self.reset_actions_to_level_0()

    def finalize(self, assert_feasible: bool = False):
        """
        Called when terminating. Creates a corresponding RDKit molecule and SMILES, as well as makes some premilinary
        checks.
        """
        if assert_feasible:
            self.assert_feasible()
        mol = self.to_rdkit_mol()
        if not self.infeasibility_flag:
            self.rdkit_mol = mol
            self.smiles_string = Chem.MolToSmiles(self.rdkit_mol)

            if self.smiles_string == "C":
                self.infeasibility_flag = True

    def register_logits(self, logits_level_0: np.array, logits_level_1: np.array, logits_level_2: np.array):
        """
        The network predicts logits for level 0 and level 1 together, while for meaningful level 2 logits we first
        need to have a decision on level 1.
        """
        if self.registered_logits is not None:
            print("Warning: Registering logits before actions have been reset.")
        if self.current_action_level == 0:
            self.registered_logits = [logits_level_0, logits_level_1, None]
        elif self.current_action_level == 2:
            self.registered_logits = [None, None, logits_level_2]
        else:
            print("Warning: Registering logits is supposed to happen only at action level 0 or 2")

    def assert_feasible(self):
        """
        Checks whether the current molecule is feasible, i.e., if only atoms are there that are allowed
        and whether all bonds are consistent
        """
        assert self.atoms[0] == 0, "First atom should be virtual (0)"
        assert np.all([not self.atom_feasibility[x - 1] for x in self.atoms[1:]]) and np.all(self.atoms[1:] > 0), "Only atoms allowed that are also allowd in config vocabulary"
        assert self.upper_limit_atoms is None or len(self.atoms) - 1 <= self.upper_limit_atoms, "Exceeded maximum number of atoms"
        assert np.all(self.bonds[0, 1:] == self.virtual_bond_idx) and np.all(self.bonds[1:, 0] == self.virtual_bond_idx), "Virtual atom must be connected to all other atoms."
        assert not np.any(self.bonds.diagonal()), "An atom (even virtual) may not be connected to itself"
        assert not np.any(self.bonds - self.bonds.T), "Bond matrix must be symmetric"
        assert np.all(np.array([self.vocabulary_valence[x] for x in self.atoms[1:]]) - self.bonds[1:, 1:].sum(axis=1) >= 0), "Valence constraints not satisfied"
        if self.current_action_level == 0 and len(self.atoms) > 2:
            assert np.all(self.bonds[1:, 1:].sum(axis=1) > 0), "An atom must be connected to at least another atom"

    def to_rdkit_mol(self) -> Chem.RWMol:
        """
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
        return Chem.MolToSmiles(self.to_rdkit_mol())

    # ---- Implementation of abstract methods from `BaseTrajectory`
    @staticmethod
    def init_batch_from_instance_list(config: MoleculeConfig, instances: List[int], network: MoleculeTransformer, device: torch.device):
        """
        An instance is given by the first atom placed on the molecule, so 1 (C), 2 (N) or 3 (O)
        """
        return [MoleculeDesign(config=config, initial_atom=atom) for atom in instances]

    @staticmethod
    def log_probability_fn(trajectories: List['MoleculeDesign'], network: MoleculeTransformer) -> List[np.array]:
        """
        Given a list of trajectories and a policy network,
        returns a list of numpy arrays, each having length num_actions, where each numpy array is a log-probability
        distribution over the next action level.

        In our molecular case, we need to distinguish between trajectories which have registered logits and those who
        don't. If they don't, we need to pass them through the network. Otherwise, we can just collect the logits for
        the corresponding action level.

        Parameters:
            trajectories [List[BaseTrajectory]]
            network [torch.nn.Module]: Policy network
        Returns:
            List of numpy arrays, where i-th entry corresponds to the log-probabilities for i-th trajectory.
        """
        log_probs_to_return: List[np.array] = []

        batch_idx_to_list_idx = []
        to_eval_with_network = []
        for i, molecule_design in enumerate(trajectories):
            if molecule_design.registered_logits is None:
                to_eval_with_network.append(molecule_design)
                batch_idx_to_list_idx.append(i)

        if len(to_eval_with_network):
            network.eval()
            with torch.no_grad():
                batch = MoleculeDesign.list_to_batch(molecules=to_eval_with_network, device=network.device)
                batch_level_zero_logits, batch_level_one_logits, batch_level_two_logits = network(batch)
                batch_level_zero_logits = batch_level_zero_logits.cpu().numpy()
                batch_level_one_logits = batch_level_one_logits.cpu().numpy()
                batch_level_two_logits = batch_level_two_logits.cpu().numpy()

                # `level_zero_logits` is of shape (batch, atom vocabulary length + 2). There is no padding, so we don't need to remove anything
                # `level_one_logits` and `level_two_logits` are of shape (batch, max num_atoms - 1), so we need to remove any padded atoms from them.
                for i in range(len(to_eval_with_network)):
                    molecule_design = trajectories[batch_idx_to_list_idx[i]]
                    level_zero_logits = batch_level_zero_logits[i]
                    level_one_logits = batch_level_one_logits[i, :batch["num_atoms"][i].item() - 1]
                    level_two_logits = batch_level_two_logits[i, :batch["num_atoms"][i].item() - 1]
                    molecule_design.register_logits(
                        level_zero_logits, level_one_logits, level_two_logits
                    )

        for molecule_design in trajectories:
            log_probs_to_return.append(molecule_design.masked_log_probs_for_current_action_level())

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
        return int((1 - self.current_action_mask[self.current_action_level]).sum())

    @staticmethod
    def list_to_batch(molecules: List['MoleculeDesign'], device: torch.device = None,
                      include_feasibility_masks: bool = False) -> dict:
        """
        Given a list of molecule designs, prepares a batch that can be passed through the network.
        In the following, when referring to `number of atoms`, we _include_ the virtual atom.
        The batch is given as a dictionary with the following keys and values:
        - "level_idx": This is "0" for level 0 or "1" for level 2 [sic!]. It is used to mark the virtual atom to inform
            the network what decision to make.
        - "picked_atom_ohe": A zero vector of length <max num atoms> with a 1 with the index of the
            picked atom at level 1. This is only relevant for level 2.
        - "num_atoms": torch.LongTensor of shape (batch_size,), which holds the number of atoms for each molecule
            (including virtual)
        - "atoms": torch.LongTensor of shape (batch_size, <max num atoms>) containing the
            indices (including virtual=0) of the atoms present in a molecule.
            We pad with 'num atoms in vocab + 1' to the maximum number of atoms in the batch.
        - "atoms_degree": torch.LongTensor of shape (batch_size, <max num atoms>), where an entry corresponds to the
            degree of the atom, i.e., to how many other atoms it is connected (excluding virtual).
             Hence, we pad with 'max possible valence + 1' to the maximum number of atoms in the batch (an atom can be connected to at most 6
             other atoms).
             Note that the virtual atom is also included here, but it is later not used in the network.
        - "bonds": torch.LongTensor of shape (batch_size, <max num atoms>, <max num atoms>), indicating the connection
            between atoms. We pad both columns and rows with 'virtual bond idx + 1' to the maximum number of atoms in the batch.
        - "additive_padding_attn_mask": torch.FloatTensor of shape (batch_size, <max num atoms>, <max num atoms>), which
            is zero everywhere except at the padding positions, where it's -inf. For numerical reasons, the diagonal is 0,
            so that still every padding token can attend to itself. This padding mask will later be added to the learned
            attention mask.
        """
        atoms_padding_idx = len(molecules[0].vocabulary_atom_idcs) + 1
        degree_padding_idx = max(molecules[0].vocabulary_valence) + 1
        bond_padding_idx = molecules[0].virtual_bond_idx + 1

        device = torch.device("cpu") if device is None else device
        num_atoms = [len(mol.atoms) for mol in molecules]
        max_num_atoms = max(num_atoms)

        batch_level_idx = [0 if mol.current_action_level == 0 else 1 for mol in molecules]

        batch_picked_atom_ohe = np.zeros(((len(molecules), max_num_atoms)), dtype=int)
        for i, mol in enumerate(molecules):
            if mol.current_action_level == 2:
                batch_picked_atom_ohe[i, mol.history[-1][1] + 1] = 1

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
            picked_atom_ohe=torch.from_numpy(batch_picked_atom_ohe).long().to(device),  # (B, <max num atoms>)
            num_atoms=torch.tensor(num_atoms, dtype=torch.long, device=device),  # (B,)
            atoms=torch.from_numpy(batch_atoms).long().to(device),  # (B, <max num atoms>)
            atoms_degree=torch.from_numpy(batch_atoms_degree).long().to(device),  # (B, <max num atoms>)
            bonds=torch.from_numpy(batch_bonds).long().to(device),  # (B, <max num atoms>, <max num atoms>)
            additive_padding_attn_mask=torch.from_numpy(batch_additive_padding_attn_mask).float().to(device),  # (B, <max num atoms>, <max num atoms>)
        )

        if include_feasibility_masks:
            num_actions_level_0 = len(molecules[0].vocabulary_atom_idcs) + 2
            # This is only relevant for training. We prepare the masks with respect to action feasibility for all
            # action levels. Note that we do (!) return masks
            # also for padded atoms, but these are 0 everywhere (so all feasible). Account for this during training.
            feasibility_mask_level_zero = torch.from_numpy(
                np.stack([
                    mol.current_action_mask[0] if mol.current_action_level == 0 else np.zeros(num_actions_level_0)
                    for mol in molecules
                ])
            ).bool().to(device)

            feasibility_mask_level_one = torch.from_numpy(
                np.stack(
                    [np.pad(
                        mol.current_action_mask[1],
                        [(0, max_num_atoms - num_atoms[i])]
                    ) if mol.current_action_level == 0 else np.zeros(max_num_atoms - 1)
                    for i, mol in enumerate(molecules)]
                )
            ).bool().to(device)

            feasibility_mask_level_two = torch.from_numpy(
                np.stack(
                    [np.pad(
                        mol.current_action_mask[2],
                        [(0, max_num_atoms - num_atoms[i])]
                    ) if mol.current_action_level == 2 else np.zeros(max_num_atoms - 1)
                     for i, mol in enumerate(molecules)]
                )
            ).bool().to(device)

            return_dict["feasibility_mask_level_zero"] = feasibility_mask_level_zero  # (B, num atoms in vocab + 2)
            return_dict["feasibility_mask_level_one"] = feasibility_mask_level_one  # (B, <max_num_atoms - 1>)
            return_dict["feasibility_mask_level_two"] = feasibility_mask_level_two  # (B, <max_num_atoms - 1>)

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
        carbon_atom_idx = list(config.atom_vocabulary.keys()).index("O") + 1
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
    def from_smiles(config: MoleculeConfig, smiles:str) -> 'MoleculeDesign':
        mol = Chem.MolFromSmiles(smiles)
        Chem.SanitizeMol(mol)
        return MoleculeDesign.from_rdkit_mol(config, mol, smiles, False)

    @staticmethod
    def from_rdkit_mol(config: MoleculeConfig, rdkit_mol: Chem.RWMol, smiles: str, do_finish=True) -> 'MoleculeDesign':
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
            atomic_num_to_atom_idx[k] = i + 1

        for atom in atoms:
            k = atom.GetAtomicNum()
            if atom.GetFormalCharge() > 0:
                k = f"{k}_1"
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
            for j in range(i-1, -1, -1):  # Loop backwards
                desired_bond_order = adjacency_matrix[i, j]
                current_bond_order = 0
                if desired_bond_order > 0:
                    if not atom_is_placed:
                        design.take_action(atom_to_add)
                        design.take_action(j)
                        current_bond_order += 1
                        atom_is_placed = True
                    # Increase bond if needed
                    while current_bond_order < desired_bond_order:
                        design.take_action(len(config.atom_vocabulary) + 1)  # increase existing bond
                        design.take_action(i)  # freshly added atom
                        design.take_action(j)  # to bond with
                        current_bond_order += 1
        if do_finish:
            design.take_action(0)
            assert design.smiles_string == smiles, f"Converted: {design.smiles_string}, RDKit: {smiles}"
            design.assert_feasible()

        return design