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
        - Level 2: Pick the type of bond (index 0 up to order 6)

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
            if (self.upper_limit_atoms is not None and len(self.atoms) - 1 == self.upper_limit_atoms) or (not np.any(atom_valence_remaining)):
                self.current_action_mask[1:ex_action_idx] = 1
            existing_bond = (self.bonds[1:, 1:] > 0).any(axis=1)
            masked = np.zeros(len(self.atoms) - 1, dtype=bool)
            masked[np.where(atom_valence_remaining <= 0)] = True
            masked[np.where(existing_bond)] = False
            self.current_action_mask[ex_action_idx:] = masked
            bond_indicator = np.zeros_like(self.bonds[1:, 1:])
            bond_indicator[np.where(self.bonds[1:, 1:] == 0)] = 1
            np.fill_diagonal(bond_indicator, 0)
            has_free_nonneighbor = np.matmul(bond_indicator, (atom_valence_remaining > 0)[:, None]).squeeze()
            self.current_action_mask[ex_action_idx:][np.where(has_free_nonneighbor == 0)] = 1

        elif self.current_action_level == 1:
            new_atom_action_count = len(self.vocabulary_atom_idcs)
            existing_bond_action_count = len(self.atoms) - 1
            total_actions = new_atom_action_count + existing_bond_action_count
            self.current_action_mask = np.zeros(total_actions, dtype=bool)
            atom_picked_on_lvl_0 = (
                len(self.atoms) - 2 if self.history[-1] < self.pick_existing_atoms_start_action_idx_lvl_0
                else self.history[-1] - self.pick_existing_atoms_start_action_idx_lvl_0)
            self.current_action_mask[:new_atom_action_count] = np.array(self.atom_feasibility_mask)
            for idx in range(new_atom_action_count):
                if idx < len(self.atoms) - 1:
                    if atom_valence_remaining[idx] < 1:
                        self.current_action_mask[idx] = 1
            for idx in range(existing_bond_action_count):
                if idx == atom_picked_on_lvl_0:
                    self.current_action_mask[new_atom_action_count + idx] = 1
                elif self.bonds[len(self.atoms) - 1, idx + 1] > 0:
                    self.current_action_mask[new_atom_action_count + idx] = 0
                else:
                    self.current_action_mask[new_atom_action_count + idx] = 1

        elif self.current_action_level == 2:
            expected_mask_length = 2 * self.maximum_bond_order
            self.current_action_mask = np.ones(expected_mask_length, dtype=bool)
            if hasattr(self, 'selected_bond'):
                atom_a_idx, atom_b_idx = self.selected_bond
            else:
                atom_picked_on_lvl_0 = (
                    len(self.atoms) - 2 if self.history[-2] < self.pick_existing_atoms_start_action_idx_lvl_0
                    else self.history[-2] - self.pick_existing_atoms_start_action_idx_lvl_0)
                atom_picked_on_lvl_1 = self.history[-1]
                atom_a_idx = atom_picked_on_lvl_0 + 1
                atom_b_idx = atom_picked_on_lvl_1 + 1

            atom_valence = np.array([self.vocabulary_valence[x] for x in self.atoms[1:]])
            atom_valence_remaining = atom_valence - self.bonds[1:, 1:].sum(axis=1)
            current_bond_order = self.bonds[atom_a_idx, atom_b_idx]
            extra_increase = min(atom_valence_remaining[atom_a_idx-1], atom_valence_remaining[atom_b_idx-1])
            allowed_final_order = current_bond_order + extra_increase
            self.current_action_mask[:int(allowed_final_order)] = False

            if current_bond_order > 0:
                for reduction in range(1, current_bond_order + 1):
                    reduction_idx = self.maximum_bond_order + reduction - 1
                    new_order = current_bond_order - reduction
                    if new_order > 0:
                        self.current_action_mask[reduction_idx] = False
                    else:
                        if self.is_connected_without_bond(atom_a_idx, atom_b_idx):
                            self.current_action_mask[reduction_idx] = False
                        else:
                            self.current_action_mask[reduction_idx] = True

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
        assert not self.synthesis_done, "Taking action on already terminated design. No no!"
        assert self.current_action_mask[action] == 0, f"Trying to take action {action} on level {self.current_action_level}, but it is set to infeasible"
        next_level = 0

        if self.current_action_level == 0:
            if action == 0:
                self.synthesis_done = True
                self.finalize()
            elif 1 <= action < self.pick_existing_atoms_start_action_idx_lvl_0:
                self.atoms = np.append(self.atoms, action)
                self.bonds = np.pad(self.bonds, [(0, 1), (0, 1)],
                                    mode='constant', constant_values=0)
                new_atom_idx = len(self.atoms) - 1
                self.bonds[0, new_atom_idx] = self.bonds[new_atom_idx, 0] = self.virtual_bond_idx
                self.update_rdkit_mol(new_atom=action)
                self.update_topological_distance_matrix(new_atom_created=True)
                self.history.append(int(action))
                self.base_atom_idx = new_atom_idx
                next_level = 1
            else:
                next_level = 1

        elif self.current_action_level == 1:
            new_atom_action_count = len(self.vocabulary_atom_idcs)
            existing_bond_action_count = len(self.atoms) - 1
            total_actions = new_atom_action_count + existing_bond_action_count
            self.current_action_mask = np.zeros(total_actions, dtype=bool)
            if len(self.history) > 0:
                atom_picked_on_lvl_0 = (
                    len(self.atoms) - 2 if self.history[-1] < self.pick_existing_atoms_start_action_idx_lvl_0
                    else self.history[-1] - self.pick_existing_atoms_start_action_idx_lvl_0)
            else:
                atom_picked_on_lvl_0 = len(self.atoms) - 2

            self.current_action_mask[:new_atom_action_count] = np.array(self.atom_feasibility_mask)
            for idx in range(new_atom_action_count):
                if idx < len(self.atoms) - 1:
                    real_atom_valence = np.array([self.vocabulary_valence[x] for x in self.atoms[1:]])
                    real_atom_bonds_sum = self.bonds[1:, 1:].sum(axis=1)
                    real_atom_valence_remaining = real_atom_valence - real_atom_bonds_sum
                    if real_atom_valence_remaining[idx] < 1:
                        self.current_action_mask[idx] = 1

            for idx in range(existing_bond_action_count):
                if idx == atom_picked_on_lvl_0:
                    self.current_action_mask[new_atom_action_count + idx] = 1
                elif self.bonds[len(self.atoms) - 1, idx + 1] > 0:
                    self.current_action_mask[new_atom_action_count + idx] = 0
                else:
                    self.current_action_mask[new_atom_action_count + idx] = 1

            if action < new_atom_action_count:
                self.atoms = np.append(self.atoms, action)
                self.bonds = np.pad(self.bonds, [(0, 1), (0, 1)],
                                    mode='constant', constant_values=0)
                new_atom_idx = len(self.atoms) - 1
                self.bonds[0, new_atom_idx] = self.bonds[new_atom_idx, 0] = self.virtual_bond_idx
                self.update_rdkit_mol(new_atom=action)
                self.update_topological_distance_matrix(new_atom_created=True)
                self.history.append(int(action))
                self.last_created_atom_idx = new_atom_idx
            else:
                if hasattr(self, 'last_created_atom_idx'):
                    candidate_atom_idx = self.last_created_atom_idx
                else:
                    existing_bond_action_index = action - new_atom_action_count
                    candidate_atom_idx = existing_bond_action_index + 1
                    if candidate_atom_idx == self.base_atom_idx:
                        candidate_atom_idx += 1
                        if candidate_atom_idx >= len(self.atoms):
                            raise IndexError("No valid candidate atom for bond modification.")
                self.selected_bond = (self.base_atom_idx, candidate_atom_idx)
                self.history.append(int(action))
            next_level = 2

        elif self.current_action_level == 2:
            if hasattr(self, 'selected_bond'):
                atom_a_idx, atom_b_idx = self.selected_bond
            else:
                atom_a_idx = self.base_atom_idx
                atom_b_idx = self.last_created_atom_idx

            if action < self.maximum_bond_order:
                new_order = action + 1
                self.bonds[atom_a_idx, atom_b_idx] = self.bonds[atom_b_idx, atom_a_idx] = new_order
                self.update_rdkit_mol(set_bond=(atom_a_idx - 1, atom_b_idx - 1, new_order))
            else:
                reduction = action - self.maximum_bond_order + 1
                current_order = self.bonds[atom_a_idx, atom_b_idx]
                new_order = max(0, current_order - reduction)
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
            for lvl in range(3):
                batch_logits_per_level[lvl] = batch_logits_per_level[lvl].cpu().numpy()
            for i, mol in enumerate(trajectories):
                logits = batch_logits_per_level[mol.current_action_level][i]
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
                [molecules[0].maximum_bond_order] * len(molecules)
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