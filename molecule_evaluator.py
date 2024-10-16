import math
from typing import List, Union
import os
import ray
import rdkit.Chem
import torch
import numpy as np
import pandas as pd
from config import MoleculeConfig
from rdkit import Chem, RDLogger
from rdkit.Contrib.SA_Score import sascorer

from molecule_design import MoleculeDesign
from objective_predictor.GH_GNN_IDAC.src.models.utilities.mol2graph import get_dataloader_pairs_T, sys2graph, atom_features, n_atom_features, n_bond_features
from objective_predictor.GH_GNN_IDAC.src.models.GHGNN_architecture import GHGNN


@ray.remote
class PredictorWorker:
    def __init__(self, config: MoleculeConfig, device: torch.device):
        # Silence RDKit warnings
        RDLogger.DisableLog('rdApp.*')

        if config.CUDA_VISIBLE_DEVICES:
            # override ray's limiting of GPUs
            os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES

        self.device = device
        self.config = config
        self.model = self._load_model()

        # Pre-calculate molecules from SMILES:
        self.pre_molecules = {
            "COC1=CC(=CC(=C1)C=O)OC": Chem.MolFromSmiles("COC1=CC(=CC(=C1)C=O)OC"),
            "COC=1C=C(C=C(C1)OC)C(=O)[C@H](O)C1=CC(=CC(=C1)OC)OC": Chem.MolFromSmiles(
                "COC=1C=C(C=C(C1)OC)C(=O)[C@H](O)C1=CC(=CC(=C1)OC)OC"),
            "O": Chem.MolFromSmiles("O"),
            "CC(C)CO": Chem.MolFromSmiles("CC(C)CO")
        }

    def predict_objectives_from_rdkit_mols(self, feasible_molecules: List[Chem.RWMol]):
        constraint_value = self.predict_constraint(feasible_molecules)  # must be exp(.) > 4
        if self.config.objective_type == "DMBA_TMB":
            ln_y_DMBA_solv = self.predict_IDAC(l_solvent=feasible_molecules,
                                               l_smiles_solute=["COC1=CC(=CC(=C1)C=O)OC"] * len(feasible_molecules))
            ln_y_TMB_solv = self.predict_IDAC(l_solvent=feasible_molecules,
                                              l_smiles_solute=[
                                                                  "COC=1C=C(C=C(C1)OC)C(=O)[C@H](O)C1=CC(=CC(=C1)OC)OC"] * len(
                                                  feasible_molecules))
            with np.errstate(divide='ignore'):
                objs = np.where(
                    constraint_value > np.exp(4),
                    np.exp(ln_y_DMBA_solv) / np.exp(ln_y_TMB_solv),
                    np.NINF
                )
        elif self.config.objective_type == "IBA":
            ln_y_IPA_solv = self.predict_IDAC(l_solvent=feasible_molecules,
                                              l_smiles_solute=["CC(C)CO"] * len(feasible_molecules))
            with np.errstate(divide='ignore'):
                objs = np.where(
                    constraint_value > np.exp(4),
                    1. / np.exp(ln_y_IPA_solv),
                    np.NINF
                )
        else:
            raise ValueError("Objective type unknown")

        if self.config.synthetic_accessibility_in_objective_scale > 0:
            sa_scores = np.array([self.calc_SA_score(x) for x in feasible_molecules])
            objs = objs - self.config.synthetic_accessibility_in_objective_scale * sa_scores

        return objs

    def calc_SA_score(self, mol: Chem.RWMol):
        """
        SA score: Synthetic accessibility of drug-like molecules (or ease of synthesis) http://www.jcheminf.com/content/1/1/8
        Gives a score of of 1 (easiest) to 10 (hardest)
        Breaks down molecule into fragments and calculates score based on "ease of synthesis" divded by number of fragments
            Therefore small molecules like methane (C) and water (O) have high scores.
        Because this value was developed for drug-like (water-soluble) molecules, and we are designing water-insoluble
            molecules, this may not be useful in the end but worth trying
        """
        return sascorer.calculateScore(mol)

    def predict_constraint(self, l_mols: List[Chem.RWMol]) -> np.array:
        ln_y_water_solv = self.predict_IDAC(l_solvent=l_mols, l_smiles_solute=["O"] * len(l_mols))
        ln_y_solv_water = self.predict_IDAC(l_solvent=[self.pre_molecules["O"]] * len(l_mols), l_smiles_solute=l_mols)

        constr_value = np.exp(ln_y_water_solv) * np.exp(ln_y_solv_water)
        return constr_value

    def predict_IDAC(self, l_solvent: List[Chem.RWMol], l_smiles_solute: List[Union[str, Chem.RWMol]], l_T=None):
        # Preprocess data
        mol_solvents = l_solvent
        mol_solutes = []
        for solute in l_smiles_solute:
            if isinstance(solute, str):
                mol_solutes.append(self.pre_molecules[solute] if solute in self.pre_molecules else Chem.MolFromSmiles(solute))
            else:
                # is Chem.RWMol
                mol_solutes.append(solute)
        Temps = l_T if l_T is not None else [25] * len(mol_solvents)
        ys = [np.nan] * len(mol_solvents)

        ## Create dummy df to be able to use original data processing functions
        mol_column_solvent = 'Molecule_Solvent'
        mol_column_solute = 'Molecule_Solute'
        target = 'log-gamma'
        df = pd.DataFrame(
            {
                mol_column_solvent: mol_solvents,
                mol_column_solute: mol_solutes,
                "T": Temps,
                "log-gamma": ys
            }
        )

        graphs_solv, graphs_solu = 'g_solv', 'g_solu'
        df[graphs_solv], df[graphs_solu] = sys2graph(
            df=df,
            mol_column_1=mol_column_solvent,
            mol_column_2=mol_column_solute,
            target=target,
            y_scaler=None,
            single_system=False,
            silent=True
        )
        ## Dataloader
        indices = df.index.tolist()
        with torch.no_grad():
            predict_loader = get_dataloader_pairs_T(df,
                                                    indices,
                                                    graphs_solv,
                                                    graphs_solu,
                                                    batch_size=self.config.objective_predictor_batch_size,
                                                    shuffle=False,
                                                    drop_last=False)

            # Batch-wise prediction
            y_pred_final = np.array([])
            for batch_solvent, batch_solute, batch_T in predict_loader:
                batch_solvent = batch_solvent.to(self.device)
                batch_solute = batch_solute.to(self.device)
                batch_T = batch_T.to(self.device)
                with torch.no_grad():
                    y_pred = self.model(
                        batch_solvent.to(self.device), batch_solute.to(self.device), batch_T.to(self.device),
                        scaler=None, ln_gamma=True).reshape(
                        -1, ).cpu().numpy()
                    y_pred_final = np.concatenate((y_pred_final, y_pred))

        return y_pred_final

    def _load_model(self):
        v_in = n_atom_features()
        e_in = n_bond_features()
        u_in = 3  # ap, bp, topopsa
        model = GHGNN(v_in, e_in, u_in, self.config.GHGNN_hidden_dim, device=self.device)
        model.load_state_dict(torch.load(self.config.GHGNN_model_path, map_location="cpu"))
        model = model.to(self.device)
        model.eval()
        return model


class MoleculeObjectiveEvaluator:
    def __init__(self, config: MoleculeConfig, device: torch.device = None):
        self.config = config
        self.device = torch.device("cpu") if device is None else device
        self.predictor_workers = [PredictorWorker.remote(self.config, self.device) for _ in range(self.config.num_predictor_workers)]

    def predict_objective(self, molecule_designs: List[Union[MoleculeDesign, str]]) -> np.array:
        """
        Takes list of molecules (either as `MoleculeDesign` or directly as SMILES string
        and predicts the objective function on them. Returns the objectives as a numpy array, but also sets the
        objective directly on the objects.
        """
        # Get molecules that are known to be feasible for the predictor / RDKit / by the constraints,
        # i.e., molecules that could be sanitized and are not single carbon atoms.
        feasible_molecules: List[Chem.RWMol] = []
        feasible_idcs = []  # indices of feasible molecules in the original `molecule_designs` list

        for i, mol in enumerate(molecule_designs):
            if isinstance(mol, MoleculeDesign):
                assert mol.synthesis_done
                if not self.infeasible_by_special_constraints(mol):
                    feasible_idcs.append(i)
                    feasible_molecules.append(mol.rdkit_mol)
            elif mol != "C":
                # is a string
                try:
                    mol = Chem.MolFromSmiles(mol)
                    Chem.SanitizeMol(mol)
                    feasible_idcs.append(i)
                    feasible_molecules.append(mol)
                except:
                    continue

        # Distribute the list of feasible molecules to the predictor workers.
        num_per_worker = math.ceil(len(feasible_molecules) / len(self.predictor_workers))
        future_objs = [
            worker.predict_objectives_from_rdkit_mols.remote(feasible_molecules[i * num_per_worker: (i+1) * num_per_worker])
            for i, worker in enumerate(self.predictor_workers)
        ]
        future_objs = ray.get(future_objs)
        objs = np.concatenate(future_objs)
        all_objs = np.array([np.NINF] * len(molecule_designs))
        all_objs[feasible_idcs] = objs

        return all_objs

    def infeasible_by_special_constraints(self, mol: MoleculeDesign) -> bool:
        """
        We check special constraints such as number of rings, nitrogen-to-nitrogen bond order, etc. and if the
        constraints are not satisfied, return true.
        """
        if mol.infeasibility_flag:
            return True

        try:
            atoms = mol.rdkit_mol.GetAtoms()
            node_f = [atom_features(atom) for atom in atoms]
        except:
            return True

        if self.config.disallow_nitrogen_nitrogen_single_bond:
            # Do not allow nitrogen to nitrogen bonds of single order (only double or triple)
            for i, atom in enumerate(mol.atoms):
                if atom == 2 or atom == 4:  # N
                    for j in range(i+1, len(mol.atoms)):
                        if (mol.atoms[j] == 2 or mol.atoms[j] == 4) and mol.bonds[i, j] == 1:
                            return True

        if not self.config.disallow_rings and self.config.disallow_rings_larger_than > 3:
            rd_mol = mol.rdkit_mol
            ring_info = rd_mol.GetRingInfo()
            rings = ring_info.AtomRings()
            for ring in rings:
                if len(ring) > self.config.disallow_rings_larger_than:
                    return True

        min_required_atoms = []
        for i, atom_name in enumerate(self.config.atom_vocabulary.keys()):
            if self.config.atom_vocabulary[atom_name]["allowed"] and "min_atoms" in self.config.atom_vocabulary[atom_name] and self.config.atom_vocabulary[atom_name]["min_atoms"] > 0:
                min_required_atoms.append((i + 1, self.config.atom_vocabulary[atom_name]["min_atoms"]))  # (tuple of atom index, min num atoms required)
        if len(min_required_atoms):
            for atom in mol.atoms:
                for j, (req_atom, req_num) in enumerate(min_required_atoms):
                    if req_atom == atom:
                        min_required_atoms[j] = (req_atom, req_num - 1)
        for _, req_num in min_required_atoms:
            if req_num > 0:
                return True

        return False
