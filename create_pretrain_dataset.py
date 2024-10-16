"""
In this script, we create a dataset from a file of SMILES strings to pretrain our model. We create a `MoleculeDesign`
from each SMILES to obtain a sequence of actions.
"""
import pickle
from rdkit import Chem
from config import MoleculeConfig
from molecule_design import MoleculeDesign
from typing import List

destination_path = "./data/pretrain_data.pickle"
allowed_vocabulary = [  # put multi-character occurences first if needed
    "o", "8", "N", "1", "4", "6", "-", ")", "5", "c", "(", "#", "n", "3", "=", "2", "7", "C", "O"
]
limit_num_smiles_to = None  # Set to integer to limit number

# ----------
molecules: List[Chem.RWMol] = []
molecule_designs: List[dict] = []

print("Opening ChEMBL and filtering for vocabulary")
unfiltered_smiles = []
with open("data/chembl.tab") as f:
    for line in f:
        s = line.rstrip().split("\t")
        if s[0] == "ID":
            continue
        smiles = s[1][1:-1]
        if len(smiles) > 0:
            unfiltered_smiles.append(smiles)

print(f"Num unfiltered: {len(unfiltered_smiles)}")

filtered_smiles = []
for mol in unfiltered_smiles:
    temp = mol
    for voc in allowed_vocabulary:
        temp = temp.replace(voc, "")
    if len(temp) == 0:
        filtered_smiles.append(mol)

print("Num filtered:", len(filtered_smiles))

print("---")
print("Creating dataset")

num_differing_smiles = 0
for line in filtered_smiles:
    smiles = line.rstrip()
    if len(smiles) > 0:
        mol = Chem.MolFromSmiles(smiles)
        Chem.SanitizeMol(mol)
        s = Chem.MolToSmiles(mol)
        if s != smiles:
            num_differing_smiles += 1
        molecules.append(mol)
    if len(molecules) == limit_num_smiles_to:
        break

print(f"Created {len(molecules)} RDkit molecules from SMILES. SMILES differing in {num_differing_smiles} occasions.")
max_num_atoms = max([x.GetNumAtoms() for x in molecules])
print(f"Maximum number of atoms: {max_num_atoms}")

config = MoleculeConfig()
"""
Allow everything in the config
"""
config.max_num_atoms = max_num_atoms
config.min_ratio_c = None  # minimum ratio of C atoms to all atoms
config.disallow_oxygen_bonding = False
config.disallow_nitrogen_nitrogen_single_bond = False
config.disallow_rings = False
config.disallow_rings_larger_than = -1

full_len = len(molecules)
i = 0
while len(molecules):
    mol = molecules.pop(0)
    i += 1
    # Create an instance of MoleculeDesign from it
    smiles = Chem.MolToSmiles(mol)
    print(f"Converting {i}/{full_len} {Chem.MolToSmiles(mol)} ...")
    molecule_design = MoleculeDesign.from_rdkit_mol(config, mol, smiles)
    instance = dict(
        start_atom=molecule_design.initial_atom,
        action_seq=molecule_design.history,
        smiles=molecule_design.smiles_string,
        obj=0.0,
        sa_score=0.0
    )
    molecule_designs.append(instance)

with open(destination_path, "wb") as f:
    pickle.dump(molecule_designs, f)