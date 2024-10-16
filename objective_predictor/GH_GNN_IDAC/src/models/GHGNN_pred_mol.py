'''
Project: GNN_IAC_T
                    GNN-Gibbs-Helmholtz - T prediction
                    
Author: Edgar Ivan Sanchez Medina
Email: sanchez@mpi-magdeburg.mpg.de

Modified by: Jan G. Rittig (AVT.SVT, RWTH Aachen)
-------------------------------------------------------------------------------
'''

# external imports
import pandas as pd
from rdkit import Chem
import torch
import os.path as osp
import numpy as np

# internal imports
from utilities.mol2graph import get_dataloader_pairs_T, sys2graph, n_atom_features, n_bond_features
from GHGNN_architecture import GHGNN


# path to original pretrained GHGNN mdoel
default_model_path = osp.join(osp.dirname(osp.abspath(__file__)), "..", "..", "models", "GHGNN.pth")

# from original GHGNN git
default_hyperparameters_dict = {
    'hidden_dim'  : 113,
    'lr'          : 0.0002532501358651798,
    'batch_size'  : 32
}

class GHGNN_Predictor():
    def __init__(self, model_path=None, hyperparameters=None, device=None):
        self.model_path =  model_path if model_path is not None else default_model_path
        self.hyperparameters = hyperparameters if hyperparameters is not None else default_hyperparameters_dict
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self._load_model()

    def _load_model(self):
        # Model
        v_in = n_atom_features()
        e_in = n_bond_features()
        u_in = 3 # ap, bp, topopsa
        model    = GHGNN(v_in, e_in, u_in, self.hyperparameters["hidden_dim"])
        model.load_state_dict(torch.load(self.model_path, 
                                        map_location=torch.device(self.device)))
        device   = torch.device(self.device)
        model    = model.to(device)
        return model 


    def predict_IDAC(self, l_smiles_solvent, l_smiles_solute, l_T=None):    
        
        # Preprocess data
        ## Build molecule from SMILES
        mol_solvents = [Chem.MolFromSmiles(smi_solv) for smi_solv in l_smiles_solvent]
        mol_solutes = [Chem.MolFromSmiles(smi_solu) for smi_solu in l_smiles_solute]
        Temps = l_T if l_T is not None else [25 for _ in range(len(mol_solvents))]
        ys = [np.nan for _ in range(len(mol_solvents))]

        ## Create dummy df to be able to use original data processing functions
        mol_column_solvent     = 'Molecule_Solvent'
        mol_column_solute      = 'Molecule_Solute'
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
        predict_loader = get_dataloader_pairs_T(df, 
                                            indices, 
                                            graphs_solv,
                                            graphs_solu,
                                            batch_size=self.hyperparameters["batch_size"], 
                                            shuffle=False, 
                                            drop_last=False)


        # Predict 
        self.model.eval()
        y_pred_final = np.array([])
        with torch.no_grad():
            for batch_solvent, batch_solute, batch_T in predict_loader:
                batch_solvent = batch_solvent.to(self.device)
                batch_solute  = batch_solute.to(self.device)
                batch_T = batch_T.to(self.device)
                with torch.no_grad():
                    if torch.cuda.is_available():
                        y_pred  = self.model(batch_solvent.cuda(), batch_solute.cuda(), batch_T.cuda(), scaler=None, ln_gamma=True)
                        y_pred = y_pred.cpu().numpy().reshape(-1,)
                
                        
                    else:
                        y_pred  = self.model(batch_solvent, batch_solute, batch_T, scaler=None, ln_gamma=True).reshape(-1,)
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

"""
---
    not needed anymore, just for documentation
---
def mol_names2smiles_pura(mol_name_list):
    # Import pura
    from pura.resolvers import resolve_identifiers
    from pura.compound import CompoundIdentifierType
    from pura.services import PubChem, CIR, CAS

    # Resolve names to SMILES
    resolved = resolve_identifiers(
        mol_name_list,
        input_identifer_type=CompoundIdentifierType.NAME,
        output_identifier_type=CompoundIdentifierType.SMILES,
        backup_identifier_types=[
            CompoundIdentifierType.INCHI_KEY,
            CompoundIdentifierType.CAS_NUMBER,
        ],
        services=[PubChem(autocomplete=True), CIR(), CAS()],
        agreement=1,
        silent=True,
    )
    print("\nResults\n")
    smiles = []
    for input_compound, resolved_identifiers in resolved:
        print(input_compound, resolved_identifiers, "\n")
        smiles.append(resolved_identifiers[0])
    return smiles 
"""

if __name__ == "__main__":

    # ================ Minimal example =================== #
    predictor = GHGNN_Predictor()
    solv_cand_smiles = "CCCO"
    obj = predict_solvent_obj(predictor=predictor, solvent_smiles=solv_cand_smiles) # returns obj value 
    constr = predict_solvent_constr(predictor=predictor, solvent_smiles=solv_cand_smiles) # returns boolean (use raw_value to get constr value)
    print(obj)
    print(constr)
    # ==================================================== #

    # Run some sanity checks
    y = predictor.predict_IDAC(l_smiles_solvent=["CCCCOC(=O)COC(=O)c1ccccc1C(=O)OCCCC", "C(CCCCCCC#N)CCCCCC#N"], l_smiles_solute=["CC1CCCCC1C", "CC#N"], l_T=[120, 100.6])
    # first value from GHGNN_brouwer_extrapolation_pred.csv, second value from brouwer_edge_pred.csv
    assert (np.isclose(y[0], 0.46824193)) 
    assert (np.isclose(y[1], -0.39795938))
    # value from from brouwer_edge_pred.csv for T=25
    y = predictor.predict_IDAC(l_smiles_solvent=["C1CCC2CCCCC2C1", "C=CC1CCC[S]1(=O)=O"], l_smiles_solute=["C1C=CC=CC1", "c1ccccc1"]) 
    # Apparently slight deviations at 5th floating point number
    assert (np.isclose(y[0], 0.114886284, rtol=4))
    assert (np.isclose(y[1], 0.301235437, rtol=4))

    # Case study
    # DMBA -> 3,5-dimethoxybenzaldehyde: COC1=CC(=CC(=C1)C=O)OC
    # Water: O
    # TMB -> (R)-3,3’,5,5’-tetramethoxy-benzoin - check SMILES (differences in AC are quite small)
    # COC1=CC(=CC(=C1)C(=O)C(O)C1=CC(=CC(=C1)OC)OC)OC (without chirality)
    # COC1=CC(=CC(=C1)C(=O)[C@@H](C1=CC(=CC(=C1)OC)OC)O)OC
    # COC1=CC(=CC(=C1)C(=O)[C@@](C1=CC(=CC(=C1)OC)OC)O)OC
    # COC=1C=C(C=C(C1)OC)C(=O)[C@H](O)C1=CC(=CC(=C1)OC)OC -> this should be it according to https://opsin.ch.cam.ac.uk/

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

