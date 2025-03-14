#**********************************************************************************
# Copyright (c) 2023 Process Systems Engineering (AVT.SVT), RWTH Aachen University
#
# This program and the accompanying materials are made available under the
# terms of the Eclipse Public License 2.0 which is available at
# http://www.eclipse.org/legal/epl-2.0.
#
# SPDX-License-Identifier: EPL-2.0
#
# The source code can be found here:
# https://git.rwth-aachen.de/avt-svt/public/GDI-NN
#
# Notes:
# - This code was adpated from the original implementation by Qin, S., Jiang, S., Li, J., Balaprakash, P., Van Lehn, R. C., & Zavala, V. M. (2023). Capturing molecular interactions in graph neural networks: a case study in multi-component phase equilibrium. Digital Discovery, 2(1), 138-151.
# - The original implementation can be found here: https://github.com/zavalab/ML/tree/master/SolvGNN
#
#*********************************************************************************


from __future__ import absolute_import

# external imports
import os.path as osp
import torch
import numpy as np
import pandas as pd

# internal imports
from model.model_GNN import solvgnn_xMLP_binary
from util.generate_dataset_for_training import solvent_dataset_binary

default_model_path = osp.join(osp.dirname(osp.abspath(__file__)), "results")

default_hyperparameters_dict = {
    'model_type': "SolvGNNxMLP",
    'pinn_lambda': 1.0,
    'pinn_start_epoch': 0, 
    'batch_size': 2000, 
    'mlp_dropout_rate': 0.0,
    'mlp_activation': "softplus",
    'enc_activation': "relu",
    'mlp_num_hid_layers': 2,
    'hidden_dim': 256,
    'batch_adding': True,
    'use_lr_scheduler': True,
    'epochs': 100,
    'lr': 1e-3, 
    'seed': 2021, 
    'data': "binaryGamma",
    'data_split_mode': "system_inter",
    'num_splits': 5,
}

class GDIGNN_Predictor():
    def __init__(self, model_path=None, hyperparameters=None, device=None):
        self.model_path =  model_path if model_path is not None else default_model_path
        self.hyperparameters = hyperparameters if hyperparameters is not None else default_hyperparameters_dict
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_list = [self._load_model(cv_index=cv_index) for cv_index in [0,1,2,3,4]]
        
    def _load_model(self, cv_index=0):
        # model parameters
        model_type = self.hyperparameters["model_type"]
        mlp_dropout_rate = self.hyperparameters["mlp_dropout_rate"]
        mlp_activation = self.hyperparameters["mlp_activation"]
        enc_activation = self.hyperparameters["enc_activation"]
        mlp_num_hid_layers = self.hyperparameters["mlp_num_hid_layers"]
        hidden_dim = self.hyperparameters["hidden_dim"]

        # training parameters 
        pinn_lambda = self.hyperparameters["pinn_lambda"]
        pinn_start_epoch = self.hyperparameters["pinn_start_epoch"]
        batch_adding = self.hyperparameters["batch_adding"]
        use_lr_scheduler = self.hyperparameters["use_lr_scheduler"]
        epochs = self.hyperparameters["epochs"]
        lr = self.hyperparameters["lr"]

        # data parameters
        data = self.hyperparameters["data"]
        data_split_mode = self.hyperparameters["data_split_mode"]
        n_splits = self.hyperparameters["num_splits"]

        save_add = f"_{data}_split-{data_split_mode}_nums{n_splits}-{model_type}_pinn_l{pinn_lambda}_se{pinn_start_epoch}_dropout{mlp_dropout_rate}_act{mlp_activation}_encAct{enc_activation}_nhl{mlp_num_hid_layers}_batchA{batch_adding}_lrsched{use_lr_scheduler}_epochs{epochs}_lr{lr}"

        # initialize model
        model = solvgnn_xMLP_binary(in_dim=74, hidden_dim=hidden_dim, n_classes=1, mlp_dropout_rate=mlp_dropout_rate, mlp_activation=mlp_activation, mlp_num_hid_layers=mlp_num_hid_layers).to(self.device)
        model.load_state_dict(torch.load('./results/final_model_cv{}{}.pth'.format(cv_index, save_add))["model_state_dict"])
        model.to(self.device)
        return model

    def predict_IDAC(self, l_smiles_solvent, l_smiles_solute): 

        # Preprocess data
        x1s = [1.0 for _ in range(len(l_smiles_solvent))] # set all x1 = 1 to get infinite dilution AC
        ys = [np.nan for _ in range(len(l_smiles_solvent))]

        ## Create dummy df to be able to use original data processing functions
        df = pd.DataFrame(
            {
                "solv1_smiles": l_smiles_solvent,
                "solv2_smiles": l_smiles_solute,
                "solv1_x": x1s,
                "gamma1": ys,
                "gamma2": ys
            }
        )
        solvent_list_path = './data/solvent_list.csv'
        dataset = solvent_dataset_binary(
            dataset=df,
            solvent_list_path=solvent_list_path,
            generate_all=False
            )
        samples = []
        for idx, _ in enumerate(l_smiles_solvent):
            chemical_list = [l_smiles_solvent[idx], l_smiles_solute[idx]]
            smiles_list = chemical_list.copy()
            sample = dataset.generate_sample(chemical_list=chemical_list, smiles_list=smiles_list, solv1_x=1.0)
            sample["solv1_x"] = torch.tensor([sample["solv1_x"]])
            sample["inter_hb"] = torch.tensor([sample["inter_hb"]])
            sample["intra_hb1"] = torch.tensor([sample["intra_hb1"]])
            sample["intra_hb2"] = torch.tensor([sample["intra_hb2"]])
            samples.append(sample)
        empty_solvsys = dataset.generate_solvsys(1).to("cuda")
        
        y_pred_final = np.array([])
        with torch.set_grad_enabled(True):
            for i, solvdata in enumerate(samples):
                y_pred = None
                with torch.backends.cudnn.flags(enabled=False):
                    y_pred = np.array([])
                    for model in self.model_list:
                        model.eval()
                        y_pred_i = model(solvdata, empty_solvsys, gamma_grad=False)
                        y_pred_i = y_pred_i[0][1].detach().cpu().numpy().reshape(-1,)
                        y_pred = np.concatenate((y_pred, y_pred_i))
                    y_pred = np.array([np.average(y_pred)])
                y_pred_final = np.concatenate((y_pred_final, y_pred))
        return y_pred_final

def predict_solvent_obj(predictor, solvent_smiles):
    ln_y_DMBA_solv = predictor.predict_IDAC(l_smiles_solvent=[solvent_smiles], l_smiles_solute=["COC1=CC(=CC(=C1)C=O)OC"]) 
    ln_y_TMB_solv = predictor.predict_IDAC(l_smiles_solvent=[solvent_smiles], l_smiles_solute=["COC=1C=C(C=C(C1)OC)C(=O)[C@H](O)C1=CC(=CC(=C1)OC)OC"]) 

    return np.exp(ln_y_DMBA_solv) / np.exp(ln_y_TMB_solv)

def predict_solvent_constr(predictor, solvent_smiles, raw_value=False):
    ln_y_water_solv = predictor.predict_IDAC(l_smiles_solvent=[solvent_smiles], l_smiles_solute=["O"]) 
    ln_y_solv_water = predictor.predict_IDAC(l_smiles_solvent=["O"], l_smiles_solute=[solvent_smiles]) 

    constr_value = np.exp(ln_y_water_solv) * np.exp(ln_y_solv_water)

    if raw_value:
        return constr_value
    return constr_value > np.exp(4)

if __name__ == '__main__':

    # ================ Minimal example =================== #
    predictor = GDIGNN_Predictor()
    solv_cand_smiles = "CCCO"
    obj = predict_solvent_obj(predictor=predictor, solvent_smiles=solv_cand_smiles) # returns obj value 
    constr = predict_solvent_constr(predictor=predictor, solvent_smiles=solv_cand_smiles) # returns boolean (use raw_value to get constr value)
    print(obj)
    print(constr)
    # ==================================================== #

    # Case study
    # DMBA -> 3,5-dimethoxybenzaldehyde: COC1=CC(=CC(=C1)C=O)OC
    # Water: O
    # TMB -> (R)-3,3’,5,5’-tetramethoxy-benzoin - check SMILES (differences in AC are quite small)
    # COC1=CC(=CC(=C1)C(=O)C(O)C1=CC(=CC(=C1)OC)OC)OC (without chirality)
    # COC1=CC(=CC(=C1)C(=O)[C@@H](C1=CC(=CC(=C1)OC)OC)O)OC
    # COC1=CC(=CC(=C1)C(=O)[C@@](C1=CC(=CC(=C1)OC)OC)O)OC
    # COC=1C=C(C=C(C1)OC)C(=O)[C@H](O)C1=CC(=CC(=C1)OC)OC -> this should be it according to https://opsin.ch.cam.ac.uk/

    y = predictor.predict_IDAC(l_smiles_solvent=["O"], l_smiles_solute=["CCO"]) 
    print(f"Solvent: water, Solute: Ethanol, T=25 °C, ln(y_inf) = {y[0]:.5f}, y_inf = {np.exp(y[0]):.2f}")

    y = predictor.predict_IDAC(l_smiles_solvent=["C(Cl)(Cl)Cl"], l_smiles_solute=["CC(=O)C"]) 
    print(f"Solvent: Chloroform, Solute: Acetone, T=25 °C, ln(y_inf) = {y[0]:.5f}, y_inf = {np.exp(y[0]):.2f}")

    y = predictor.predict_IDAC(l_smiles_solvent=["O"], l_smiles_solute=["COC1=CC(=CC(=C1)C=O)OC"]) 
    print(f"Solvent: water, Solute: DMBA, T=25 °C, ln(y_inf) = {y[0]:.5f}, y_inf = {np.exp(y[0]):.2f}")

    #y = predictor.predict_IDAC(l_smiles_solvent=["O"], l_smiles_solute=["COC1=CC(=CC(=C1)C(=O)C(O)C1=CC(=CC(=C1)OC)OC)OC"]) 
    #print(f"Solvent: water, Solute: TMB, T=25 °C, ln(y_inf) = {y[0]:.5f}, y_inf = {np.exp(y[0]):.2f}")

    #y = predictor.predict_IDAC(l_smiles_solvent=["O"], l_smiles_solute=["COC1=CC(=CC(=C1)C(=O)[C@@H](C1=CC(=CC(=C1)OC)OC)O)OC"]) 
    #print(f"Solvent: water, Solute: TMB, T=25 °C, ln(y_inf) = {y[0]:.5f}, y_inf = {np.exp(y[0]):.2f}")

    #y = predictor.predict_IDAC(l_smiles_solvent=["O"], l_smiles_solute=["COC1=CC(=CC(=C1)C(=O)[C@@](C1=CC(=CC(=C1)OC)OC)O)OC"]) 
    #print(f"Solvent: water, Solute: TMB, T=25 °C, ln(y_inf) = {y[0]:.5f}, y_inf = {np.exp(y[0]):.2f}")

    y = predictor.predict_IDAC(l_smiles_solvent=["O"], l_smiles_solute=["COC=1C=C(C=C(C1)OC)C(=O)[C@H](O)C1=CC(=CC(=C1)OC)OC"]) 
    print(f"Solvent: water, Solute: TMB, T=25 °C, ln(y_inf) = {y[0]:.5f}, y_inf = {np.exp(y[0]):.2f}")

    # Candidate solvents from Table 1 in original study, i.e., DOI: https://doi.org/10.1002/bit.21867
    cand_solv_names = ["Cis-decalin", "Cylcohexanone", "n-Heptane", "2-Propanol", "n-Hexane", "Cyclohexane", "Pentane", "1-Propanol", "Tetrahydrofuran", "Toluene", "Ethoxybenzene", "Ethylacetate", "Benzene", "2-Pentanone", "Piperidine", "Anisole", "1-Butanol", "MTBE", "Acetophenone", "1-Hexanol", "MIBK", "1-Pentanol", "Diethylether"]

    # we used pura to translate mol_names to smiles 
    # note that cis-decalin was manually corrected from C1CCC2CCCCC2C1 to C1CCC[C@@H]2CCCC[C@H]12
    #cand_solv_smiles = []
    #for mol in cand_solv_names:
    #    cand_solv_smiles += mol_names2smiles_pura([mol]) 
    #print(cand_solv_smiles)

    cand_solv_smiles = ['C1CCC[C@@H]2CCCC[C@H]12', 'O=C1CCCCC1', 'CCCCCCC', 'CC(C)O', 'CCCCCC', 'C1CCCCC1', 'CCCCC', 'CCCO', 'C1CCOC1', 'Cc1ccccc1', 'CCOc1ccccc1', 'CCOC(C)=O', 'c1ccccc1', 'CCCC(C)=O', 'C1CCNCC1', 'COc1ccccc1', 'CCCCO', 'COC(C)(C)C', 'CC(=O)c1ccccc1', 'CCCCCCO', 'CC(=O)CC(C)C', 'CCCCCO', 'CCOCC']

    objs = []
    constrs = []
    constrs_raw = []
    for idx, cand_i_smiles in enumerate(cand_solv_smiles):
        obj = predict_solvent_obj(predictor=predictor, solvent_smiles=cand_i_smiles)[0]
        constr = predict_solvent_constr(predictor=predictor, solvent_smiles=cand_i_smiles, raw_value=False)[0]
        constr_raw = predict_solvent_constr(predictor=predictor, solvent_smiles=cand_i_smiles, raw_value=True)[0]
        objs.append(obj)
        constrs.append(constr)
        constrs_raw.append(constr_raw)
        print(f"-> OBJ: {obj:.3f}, constraint: {constr} \t -> solvent: {cand_solv_names[idx]} - {cand_i_smiles}")

    df =  pd.DataFrame({
        "Name": cand_solv_names,
        "SMILES": cand_solv_smiles,
        "OBJ": objs,
        "Miscibility constraint": constrs,
        "Constraint raw": constrs_raw
    })
    df.to_csv("Solvents.csv")