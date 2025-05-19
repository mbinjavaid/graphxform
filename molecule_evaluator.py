import math
from typing import List, Union
import os
import ray
import torch
import numpy as np
import pandas as pd
from config import MoleculeConfig
from rdkit import Chem, RDLogger
from rdkit.Contrib.SA_Score import sascorer

from molecule_design import MoleculeDesign
from objective_predictor.GH_GNN_IDAC.src.models.utilities.mol2graph import get_dataloader_pairs_T, sys2graph, atom_features, n_atom_features, n_bond_features
from objective_predictor.GH_GNN_IDAC.src.models.GHGNN_architecture import GHGNN

from guacamol.benchmark_suites import goal_directed_suite_v2


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
                    -np.inf
                )
        elif self.config.objective_type == "IBA":
            ln_y_IPA_solv = self.predict_IDAC(l_solvent=feasible_molecules,
                                              l_smiles_solute=["CC(C)CO"] * len(feasible_molecules))
            with np.errstate(divide='ignore'):
                objs = np.where(
                    constraint_value > np.exp(4),
                    1. / np.exp(ln_y_IPA_solv),
                    -np.inf
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
        # initialize GuacaMol benchmarks
        guacamol_goal_directed_suite = goal_directed_suite_v2()
        self.guacamol_benchmarks = dict(
            celecoxib_rediscovery=guacamol_goal_directed_suite[0],
            troglitazone_rediscovery=guacamol_goal_directed_suite[1],
            thiothixene_rediscovery=guacamol_goal_directed_suite[2],
            aripiprazole_similarity=guacamol_goal_directed_suite[3],
            albuterol_similarity=guacamol_goal_directed_suite[4],
            mestranol_similarity=guacamol_goal_directed_suite[5],
            isomers_c11h24=guacamol_goal_directed_suite[6],
            isomers_c9h10n2o2pf2cl=guacamol_goal_directed_suite[7],
            median_camphor_menthol=guacamol_goal_directed_suite[8],
            median_tadalafil_sildenafil=guacamol_goal_directed_suite[9],
            osimertinib_mpo=guacamol_goal_directed_suite[10],
            fexofenadine_mpo=guacamol_goal_directed_suite[11],
            ranolazine_mpo=guacamol_goal_directed_suite[12],
            perindopril_rings=guacamol_goal_directed_suite[13],
            amlodipine_rings=guacamol_goal_directed_suite[14],
            sitagliptin_replacement=guacamol_goal_directed_suite[15],
            zaleplon_mpo=guacamol_goal_directed_suite[16],
            valsartan_smarts=guacamol_goal_directed_suite[17],
            deco_hop=guacamol_goal_directed_suite[18],
            scaffold_hop=guacamol_goal_directed_suite[19]
        )

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

        if self.config.objective_type in self.guacamol_benchmarks:
            # Drug design tasks
            objs = np.array([
                self.guacamol_benchmarks[self.config.objective_type].objective.score(
                    Chem.MolToSmiles(rdkit_mol)
                )
                for rdkit_mol in feasible_molecules
            ])
        else:
            # Distribute the list of feasible molecules to the predictor workers.
            num_per_worker = math.ceil(len(feasible_molecules) / len(self.predictor_workers))
            future_objs = [
                worker.predict_objectives_from_rdkit_mols.remote(feasible_molecules[i * num_per_worker: (i+1) * num_per_worker])
                for i, worker in enumerate(self.predictor_workers)
            ]
            future_objs = ray.get(future_objs)
            objs = np.concatenate(future_objs)
        all_objs = np.array([-np.inf] * len(molecule_designs))
        all_objs[feasible_idcs] = objs

        return all_objs

    # def predict_objective(self, molecule_designs: List[MoleculeDesign]) -> np.array:
    #     """
    #     Takes list of MoleculeDesign objects, predicts their base objectives,
    #     applies penalties, and sets 'original_objective' and 'objective' (penalized)
    #     on each MoleculeDesign instance.
    #     Returns an array of the *penalized* objectives for search guidance.
    #     Also ensures 'sa_score' is set on each MoleculeDesign instance.
    #     """
    #
    #     # Ensure all inputs are MoleculeDesign instances, as expected by this refined logic
    #     for md_input_idx, md_input in enumerate(molecule_designs):
    #         if not isinstance(md_input, MoleculeDesign):
    #             raise TypeError(
    #                 f"predict_objective expects a list of MoleculeDesign objects. "
    #                 f"Got type {type(md_input)} at index {md_input_idx}."
    #             )
    #         # Ensure molecule is finalized if it's a leaf being evaluated
    #         if not md_input.synthesis_done:
    #             md_input.finalize(assert_feasible=False)
    #
    #     # --- Step 1: Prepare lists for base objective calculation ---
    #     # These lists will hold items for which base objectives need to be computed.
    #     # They will correspond one-to-one with `valid_md_for_objective_calc`.
    #     items_for_base_calc: List[Union[str, Chem.RWMol]] = []  # RDKit mols or SMILES strings
    #     valid_md_for_objective_calc: List[MoleculeDesign] = []  # Corresponding MoleculeDesign objects
    #     original_indices_of_valid_md: List[int] = []  # Original indices of these valid MDs in input list
    #
    #     for i, md in enumerate(molecule_designs):
    #         # `infeasible_by_special_constraints` checks `md.rdkit_mol`
    #         if md.rdkit_mol is not None and not self.infeasible_by_special_constraints(md):
    #             valid_md_for_objective_calc.append(md)
    #             original_indices_of_valid_md.append(i)
    #             if self.config.objective_type in self.guacamol_benchmarks:
    #                 items_for_base_calc.append(Chem.MolToSmiles(md.rdkit_mol))
    #             else:
    #                 items_for_base_calc.append(md.rdkit_mol)  # Pass RDKit mol to PredictorWorker
    #         else:
    #             # This MoleculeDesign instance is infeasible or problematic from the start
    #             md.original_objective = -np.inf
    #             md.objective = -np.inf  # For search
    #             md.sa_score = 10.0  # Worst SA score
    #
    #     # --- Step 2: Calculate base objectives for the valid items ---
    #     base_objectives_for_valid_md = np.array([-np.inf] * len(valid_md_for_objective_calc))
    #
    #     if items_for_base_calc:  # Only if there are valid items
    #         if self.config.objective_type in self.guacamol_benchmarks:
    #             # items_for_base_calc contains SMILES strings here
    #             base_objectives_for_valid_md = np.array([
    #                 self.guacamol_benchmarks[self.config.objective_type].objective.score(smiles)
    #                 for smiles in items_for_base_calc
    #             ])
    #         else:  # Custom objective via PredictorWorker
    #             # items_for_base_calc contains RDKit Mols here
    #             num_per_worker = math.ceil(
    #                 len(items_for_base_calc) / len(self.predictor_workers)) if self.predictor_workers else len(
    #                 items_for_base_calc)
    #
    #             if not self.predictor_workers:  # Should not happen if not GuacaMol task
    #                 raise RuntimeError("PredictorWorkers not initialized for non-GuacaMol task.")
    #
    #             future_objs_promises = [
    #                 worker.predict_objectives_from_rdkit_mols.remote(
    #                     items_for_base_calc[j * num_per_worker: (j + 1) * num_per_worker]
    #                 )
    #                 for j, worker in enumerate(self.predictor_workers) if
    #                 items_for_base_calc[j * num_per_worker: (j + 1) * num_per_worker]  # Ensure non-empty slice
    #             ]
    #
    #             if future_objs_promises:
    #                 future_objs_results = ray.get(future_objs_promises)
    #                 # Concatenate results carefully, handling potential empty arrays from workers
    #                 concatenated_results_list = [res for res in future_objs_results if
    #                                              isinstance(res, np.ndarray) and res.size > 0]
    #                 if concatenated_results_list:
    #                     base_objectives_for_valid_md = np.concatenate(concatenated_results_list)
    #                 # If all workers returned empty or invalid results, base_objectives_for_valid_md remains -np.inf initialized.
    #             # If items_for_base_calc was empty or future_objs_promises was empty, it also remains -np.inf initialized.
    #
    #     # Check for length mismatch (should only happen if base objective calculation failed for some valid items)
    #     if len(base_objectives_for_valid_md) != len(valid_md_for_objective_calc) and items_for_base_calc:
    #         print(
    #             f"Warning: Mismatch in base objective calculation. Expected {len(valid_md_for_objective_calc)}, got {len(base_objectives_for_valid_md)}. "
    #             f"Some base objectives might be -np.inf.")
    #         # Ensure base_objectives_for_valid_md is padded to the correct length if it's shorter
    #         if len(base_objectives_for_valid_md) < len(valid_md_for_objective_calc):
    #             padding = np.array([-np.inf] * (len(valid_md_for_objective_calc) - len(base_objectives_for_valid_md)))
    #             base_objectives_for_valid_md = np.concatenate((base_objectives_for_valid_md, padding))
    #
    #     # --- Step 3: Set original, penalized objectives, and SA score on MoleculeDesign instances ---
    #     final_penalized_objectives_to_return = np.array([-np.inf] * len(molecule_designs))
    #
    #     for i, md_instance in enumerate(valid_md_for_objective_calc):
    #         original_input_idx = original_indices_of_valid_md[i]
    #
    #         base_obj = base_objectives_for_valid_md[i] if i < len(base_objectives_for_valid_md) else -np.inf
    #         md_instance.original_objective = base_obj  # This base_obj already includes SA penalty from PredictorWorker if applicable
    #
    #         penalty = self.config.high_level_action_penalty_factor * md_instance.num_high_level_actions
    #         penalized_obj = base_obj - penalty
    #
    #         md_instance.objective = penalized_obj  # This is used by SBS/search
    #         final_penalized_objectives_to_return[original_input_idx] = penalized_obj
    #
    #         # Set SA score on MoleculeDesign object
    #         # PredictorWorker already uses SA score to modify the objective for non-GuacaMol tasks.
    #         # For consistent reporting in process_results, we ensure sa_score attribute is set.
    #         # if md_instance.rdkit_mol:
    #         md_instance.sa_score = sascorer.calculateScore(md_instance.rdkit_mol)
    #         # else:  # Should not happen if it was in valid_md_for_objective_calc
    #         #     md_instance.sa_score = 10.0  # Default worst SA score
    #
    #     # For any MoleculeDesign instances that were initially infeasible (not in valid_md_for_objective_calc)
    #     # their original_objective and objective were already set to -np.inf.
    #     # Their sa_score was also set to 10.0.
    #     # final_penalized_objectives_to_return is already -np.inf for them.
    #
    #     return final_penalized_objectives_to_return

    def infeasible_by_special_constraints(self, mol: MoleculeDesign) -> bool:
        """
        We check special constraints such as number of rings, nitrogen-to-nitrogen bond order, etc. and if the
        constraints are not satisfied, return true.
        """
        # print("mol infeasibility flag: ", mol.infeasibility_flag)
        # check whether the molecule is synthesis complete
        # print("mol synthesis done: ", mol.synthesis_done)
        if mol.infeasibility_flag:
            return True

        try:
            atoms = mol.rdkit_mol.GetAtoms()
            node_f = [atom_features(atom) for atom in atoms]
        except:
            return True
            # print("Error in getting atoms of molecule")

        if self.config.objective_type in ["IBA", "DMBA_TMB"] and self.config.include_structural_constraints:
            """
            Check for a ring with more than 6 atoms or less than 5
            """
            for ring in mol.rdkit_mol.GetRingInfo().AtomRings():
                if len(ring) < 5 or len(ring) > 6: # adjust according to max/min ring size
                    return True
            """
            Check for a O-O single bond in the molecule
            """
            for bond in mol.rdkit_mol.GetBonds():
                if (bond.GetBondType() == Chem.BondType.SINGLE and
                    mol.rdkit_mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomicNum() == 8 and
                        mol.rdkit_mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomicNum() == 8):
                    return True
            """
            Check for an N-N single bond
            """
            for bond in mol.rdkit_mol.GetBonds():
                if (bond.GetBondType() == Chem.BondType.SINGLE and
                    mol.rdkit_mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomicNum() == 7 and
                        mol.rdkit_mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomicNum() == 7):
                    return True
            """
            Check for an N-C-N bond (with exception for C=0)
            """
            for atom in mol.rdkit_mol.GetAtoms():
                if atom.GetAtomicNum() == 6:
                    neighbors = atom.GetNeighbors()
                    nitrogen_count = sum(1 for nbr in neighbors if
                                         nbr.GetAtomicNum() == 7 and
                                         mol.rdkit_mol.GetBondBetweenAtoms(atom.GetIdx(),
                                                                              nbr.GetIdx()).GetBondType() == Chem.BondType.SINGLE)

                    # Check if carbon is also double-bonded to oxygen (C=O)
                    has_carbonyl = any(
                        nbr.GetAtomicNum() == 8 and  # Oxygen
                        mol.rdkit_mol.GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
                        for nbr in neighbors
                    )

                    if nitrogen_count >= 2 and not has_carbonyl:
                        return True
            """
            Don't allow O-C(X)-N
            """
            for atom in mol.rdkit_mol.GetAtoms():
                if atom.GetAtomicNum() == 6:  # Carbon atom
                    neighbors = atom.GetNeighbors()

                    # Count the types of bonded atoms
                    n_count = sum(1 for nbr in neighbors if nbr.GetAtomicNum() == 7)  # Nitrogen
                    o_count = sum(1 for nbr in neighbors if nbr.GetAtomicNum() == 8)  # Oxygen
                    h_count = atom.GetTotalNumHs()  # Hydrogen

                    # Condition: Carbon is bonded to both N and O and has exactly 1 H
                    if n_count >= 1 and o_count >= 1 and h_count == 1:
                        return True  # Restriction is violated



            if self.config.objective_type == 'DMBA_TMB':
                # The molecule must be an alcohol (-OH group must be present):
                # Check if any atom has an -OH group
                has_alcohol = any(
                    atom.GetAtomicNum() == 8 and  # Oxygen
                    atom.GetDegree() == 1 and  # Oxygen is connected to only one other atom (the carbon)
                    atom.GetTotalNumHs() == 1  # Oxygen has one hydrogen
                    for atom in mol.rdkit_mol.GetAtoms()
                )
                if not has_alcohol:
                    return True

        return False