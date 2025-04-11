"""
This script generates paired molecule transformation datasets for both training and validation data.
We find pairs of molecules, calculate their Graph Edit Distance (GED),
and record the edit sequences required to transform one into the other.
Includes checkpointing to save intermediate results.
"""
import time
import pickle
import os
import random
import numpy as np
import networkx as nx
from rdkit import Chem, RDLogger
from tqdm import tqdm
from typing import List, Tuple, Dict
from datetime import datetime

# Chem.CanonicalRankAtoms

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Configuration
MAX_ATOMS = 50
# NUM_PAIRS is now determined by dataset size
RANDOM_SEED = 42
CHECKPOINT_DIR = "./data/chembl/checkpoints"
CHECKPOINT_FREQUENCY = 5000  # Save progress after every 5000 transformations
DATATYPES = ["train", "valid"]

# Set random seed for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def load_and_filter_molecules(path: str, max_atoms: int = MAX_ATOMS) -> List[Tuple[Chem.Mol, str]]:
    """
    Load molecules from SMILES file and filter by size.
    Loads from checkpoint if available.

    Args:
        path: Path to SMILES file
        max_atoms: Maximum number of atoms allowed

    Returns:
        List of (RDKit molecule, canonical SMILES) tuples
    """
    # Extract datatype from path for checkpoint naming
    datatype = "train" if "train" in path else "valid"
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"filtered_molecules_{datatype}.pkl")

    # Try to load from checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading filtered {datatype} molecules from checkpoint {checkpoint_path}")
        try:
            with open(checkpoint_path, "rb") as f:
                molecules = pickle.load(f)
            print(f"Loaded {len(molecules)} {datatype} molecules from checkpoint")
            return molecules
        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Processing from scratch.")

    print(f"Loading and filtering molecules from {path}")
    molecules = []

    with open(path) as f:
        for line in tqdm(f):
            smiles = line.strip()
            if not smiles:
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            try:
                Chem.SanitizeMol(mol)
                canonical_smiles = Chem.CanonSmiles(Chem.MolToSmiles(mol))
                order = Chem.CanonicalRankAtoms(mol, inclueChirality=True)
                # enforce canonical atom ordering
                mol = Chem.RenumberAtoms(mol, order)

                if mol.GetNumAtoms() <= max_atoms:
                    molecules.append((mol, canonical_smiles))
            except:
                continue

    print(f"Loaded {len(molecules)} valid molecules with ≤{max_atoms} atoms")

    # Save checkpoint
    with open(checkpoint_path, "wb") as f:
        pickle.dump(molecules, f)
    print(f"Saved filtered molecules to {checkpoint_path}")

    return molecules


def create_molecule_pairs(molecules: List[Tuple[Chem.Mol, str]], datatype: str) -> List[
    Tuple[Chem.Mol, Chem.Mol, str, str]]:
    """
    Create random pairs of molecules where each molecule appears in exactly two pairs.
    Returns kekulized molecules and their canonical SMILES.
    Uses the number of molecules as the target number of pairs.

    Args:
        molecules: List of (molecule, SMILES) tuples
        datatype: "train" or "valid" for checkpoint naming

    Returns:
        List of (mol1, mol2, smiles1, smiles2) tuples with kekulized molecules and canonical SMILES
    """
    pairs_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"molecule_pairs_{datatype}.pkl")

    # Use the number of molecules as the target number of pairs
    num_pairs = len(molecules)

    # Check if we already have the final pairs
    if os.path.exists(pairs_checkpoint_path):
        print(f"Loading {datatype} molecule pairs from checkpoint {pairs_checkpoint_path}")
        try:
            with open(pairs_checkpoint_path, "rb") as f:
                pairs = pickle.load(f)
            print(f"Loaded {len(pairs)} {datatype} molecule pairs from checkpoint")
            return pairs
        except Exception as e:
            print(f"Failed to load pairs checkpoint: {e}. Creating pairs from scratch.")

    # Calculate total molecules needed
    num_molecules = len(molecules)
    if num_molecules < 2:
        raise ValueError("Need at least 2 molecules to create pairs")

    print(f"Creating {num_pairs} {datatype} pairs with each molecule appearing twice")

    # Simple algorithm to pair molecules:
    # 1. Create a list of molecule indices that should appear in pairs
    # 2. Each index should appear twice in this list
    # 3. Shuffle this list
    # 4. Create pairs by taking consecutive pairs from the list

    # Create list where each molecule index appears exactly twice
    mol_indices = []
    for i in range(num_molecules):
        mol_indices.extend([i, i])
        if len(mol_indices) >= 2 * num_pairs:
            break

    # Shuffle the indices
    random.shuffle(mol_indices)

    # Make sure no molecule is paired with itself
    # by swapping indices if needed
    for i in range(0, len(mol_indices), 2):
        if i + 1 < len(mol_indices):
            if mol_indices[i] == mol_indices[i+1]:
                # Find a position to swap with
                for j in range(i+2, len(mol_indices), 2):
                    if mol_indices[j] != mol_indices[i] and mol_indices[j+1] != mol_indices[i]:
                        # Swap i+1 with j
                        mol_indices[i+1], mol_indices[j] = mol_indices[j], mol_indices[i+1]
                        break

    # Create the pairs
    pairs = []
    paired_together = set()  # Track pairs to avoid duplicates

    for i in range(0, len(mol_indices), 2):
        if i + 1 >= len(mol_indices):
            break

        idx1 = mol_indices[i]
        idx2 = mol_indices[i+1]

        # Skip if same molecule or already paired
        pair_key = tuple(sorted([idx1, idx2]))
        if idx1 == idx2 or pair_key in paired_together:
            continue

        paired_together.add(pair_key)

        mol1, smiles1 = molecules[idx1]
        mol2, smiles2 = molecules[idx2]

        try:
            # # Create kekulized copies
            # mol1_kekulized = Chem.Mol(mol1)
            # mol2_kekulized = Chem.Mol(mol2)

            Chem.Kekulize(mol1, clearAromaticFlags=True)
            Chem.Kekulize(mol2, clearAromaticFlags=True)

            # # Get canonical SMILES of kekulized molecules
            # smiles1_kek = Chem.MolToSmiles(mol1_kekulized, canonical=True)
            # smiles2_kek = Chem.MolToSmiles(mol2_kekulized, canonical=True)

            # Store the pair
            pairs.append((mol1, mol2, smiles1, smiles2))

        except Exception as e:
            print(f"Warning: Failed to kekulize molecule pair: {e}")
            continue

    print(f"Created {len(pairs)} {datatype} molecule pairs")

    # Save the pairs
    with open(pairs_checkpoint_path, "wb") as f:
        pickle.dump(pairs, f)
    print(f"Saved molecule pairs to {pairs_checkpoint_path}")

    return pairs


def mol_to_nx_graph(mol: Chem.Mol) -> nx.Graph:
    """
    Convert an RDKit molecule to a NetworkX graph suitable for GED,
    using only essential properties for matching.

    Args:
        mol: RDKit molecule

    Returns:
        NetworkX graph representation with minimal atom and bond properties
    """
    graph = nx.Graph()

    # Add nodes (atoms) with only essential properties
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_props = {
            'atomic_num': atom.GetAtomicNum(),
            'formal_charge': atom.GetFormalCharge(),
            'chiral_tag': int(atom.GetChiralTag())
        }
        graph.add_node(atom_idx, **atom_props)

    # Add edges (bonds) with only bond type
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        bond_props = {
            'bond_type': int(bond.GetBondType())
        }
        graph.add_edge(begin_idx, end_idx, **bond_props)

    return graph


def node_match(node1: Dict, node2: Dict) -> bool:
    """
    Custom node matching function for GED.
    Nodes match if they have the same atom type, formal charge, and chirality.
    """
    return (node1['atomic_num'] == node2['atomic_num'] and
            node1['formal_charge'] == node2['formal_charge'] and
            node1['chiral_tag'] == node2['chiral_tag'])


def edge_match(edge1: Dict, edge2: Dict) -> bool:
    """
    Custom edge matching function for GED.
    Edges match if they have the same bond type.
    """
    return edge1['bond_type'] == edge2['bond_type']


def calculate_edit_path(mol1: Chem.Mol, mol2: Chem.Mol) -> List[dict]:
    """
    Calculate a chemically meaningful edit path between molecules with full chemical details.

    Args:
        mol1: Source molecule
        mol2: Target molecule

    Returns:
        List of operation dictionaries with complete chemical information
    """
    graph1 = mol_to_nx_graph(mol1)
    graph2 = mol_to_nx_graph(mol2)

    # Get raw edit path
    path_generator = nx.optimize_edit_paths(
        graph1, graph2,
        node_match=node_match,
        edge_match=edge_match
    )
    node_edit_path, edge_edit_path, cost = next(path_generator)

    # Enhanced chemical edit path
    chemical_path = []

    # Process node operations with chemical details
    for u, v in node_edit_path:
        if u is None:  # Node insertion
            atom_props = graph2.nodes[v]
            chemical_path.append({
                'operation': 'insert_node',
                'target_idx': v,
                'element': atom_props['atomic_num'],
                'charge': atom_props.get('formal_charge', 0),
                'chiral_tag': atom_props.get('chiral_tag', 0)
            })
        elif v is None:  # Node deletion
            atom_props = graph1.nodes[u]
            chemical_path.append({
                'operation': 'delete_node',
                'source_idx': u,
                'element': atom_props['atomic_num'],
                'charge': atom_props.get('formal_charge', 0),
                'chiral_tag': atom_props.get('chiral_tag', 0)
            })
        elif not node_match(graph1.nodes[u], graph2.nodes[v]):  # Node substitution
            source_props = graph1.nodes[u]
            target_props = graph2.nodes[v]
            chemical_path.append({
                'operation': 'substitute_node',
                'source_idx': u,
                'target_idx': v,
                'from_element': source_props['atomic_num'],
                'to_element': target_props['atomic_num'],
                'from_charge': source_props.get('formal_charge', 0),
                'to_charge': target_props.get('formal_charge', 0),
                'from_chiral': source_props.get('chiral_tag', 0),
                'to_chiral': target_props.get('chiral_tag', 0)
            })

    # Process edge operations with bond chemistry details
    for edge1, edge2 in edge_edit_path:
        if edge1 is None:  # Edge insertion
            u, v = edge2
            bond_props = graph2.edges[edge2]
            chemical_path.append({
                'operation': 'insert_edge',
                'atom1_idx': u,
                'atom2_idx': v,
                'bond_type': bond_props['bond_type'],
                'bond_name': str(Chem.BondType.values[bond_props['bond_type']])
            })
        elif edge2 is None:  # Edge deletion
            u, v = edge1
            bond_props = graph1.edges[edge1]
            chemical_path.append({
                'operation': 'delete_edge',
                'atom1_idx': u,
                'atom2_idx': v,
                'bond_type': bond_props['bond_type'],
                'bond_name': str(Chem.BondType.values[bond_props['bond_type']])
            })
        else:  # Edge substitution
            u1, v1 = edge1
            u2, v2 = edge2
            bond1_props = graph1.edges[edge1]
            bond2_props = graph2.edges[edge2]

            if not edge_match(bond1_props, bond2_props):
                chemical_path.append({
                    'operation': 'substitute_edge',
                    'source_atom1': u1,
                    'source_atom2': v1,
                    'target_atom1': u2,
                    'target_atom2': v2,
                    'from_bond_type': bond1_props['bond_type'],
                    'to_bond_type': bond2_props['bond_type'],
                    'from_bond_name': str(Chem.BondType.values[bond1_props['bond_type']]),
                    'to_bond_name': str(Chem.BondType.values[bond2_props['bond_type']])
                })

    # Add molecule-level metadata to help with mapping to action space
    chemical_path.append({
        'operation': 'metadata',
        'source_num_atoms': mol1.GetNumAtoms(),
        'target_num_atoms': mol2.GetNumAtoms(),
        'source_smiles': Chem.MolToSmiles(mol1),
        'target_smiles': Chem.MolToSmiles(mol2),
        'edit_distance': cost
    })

    return chemical_path


def process_dataset(datatype):
    """Process a single dataset (train or valid)"""
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set up paths and checkpoints
    transformation_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"transformation_data_{datatype}_{timestamp}.pkl")
    final_output_path = f"./data/chembl/transformation_dataset_{datatype}.pickle"

    # Load previously saved transformation data if exists
    transformation_data = []
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith(f"transformation_data_{datatype}_")]

    if checkpoint_files:
        # Sort by timestamp to get the most recent checkpoint
        checkpoint_files.sort(reverse=True)
        latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoint_files[0])
        print(f"Found {datatype} transformation checkpoint: {latest_checkpoint}")
        try:
            with open(latest_checkpoint, "rb") as f:
                transformation_data = pickle.load(f)
            print(f"Loaded {len(transformation_data)} transformations from checkpoint")

            # Extract already processed pairs to skip them
            processed_pairs = {(item['source_smiles'], item['target_smiles']) for item in transformation_data}
            print(f"Will skip {len(processed_pairs)} already processed pairs")
        except Exception as e:
            print(f"Failed to load transformation checkpoint: {e}")
            transformation_data = []
            processed_pairs = set()
    else:
        processed_pairs = set()

    # 1. Load molecules
    molecules = load_and_filter_molecules(f"./data/chembl/chembl_{datatype}_filtered.smiles")

    # 2. Create molecule pairs - using the number of molecules as target
    pairs = create_molecule_pairs(molecules, datatype)

    # 3. Calculate GED and edit paths for each pair
    print(f"Calculating Graph Edit Distances and edit paths for {datatype} dataset")
    print(f"Processing {len(pairs)} molecule pairs")

    for i, (mol1, mol2, smiles1, smiles2) in enumerate(tqdm(pairs)):
        # Skip already processed pairs
        if (smiles1, smiles2) in processed_pairs:
            continue

        try:
            edit_path = calculate_edit_path(mol1, mol2)

            # Store the transformation data
            transformation = {
                'source_smiles': smiles1,
                'target_smiles': smiles2,
                'edit_path': edit_path
            }
            transformation_data.append(transformation)

            # Save checkpoint periodically
            if (i + 1) % CHECKPOINT_FREQUENCY == 0:
                with open(transformation_checkpoint_path, "wb") as f:
                    pickle.dump(transformation_data, f)
                print(f"Checkpoint saved after {len(transformation_data)} transformations")

        except Exception as e:
            print(f"Error processing {datatype} pair ({smiles1}, {smiles2}): {str(e)}")

    # 4. Save the final transformation dataset
    with open(final_output_path, "wb") as f:
        pickle.dump(transformation_data, f)

    print(f"Saved {len(transformation_data)} {datatype} transformations to {final_output_path}")
    print(f"Total processing time for {datatype}: {time.time() - start_time:.2f} seconds")


def main():
    """Main function to create transformation datasets for both train and validation"""

    # Process each dataset separately
    for datatype in DATATYPES:
        print(f"\n{'='*50}")
        print(f"Processing {datatype.upper()} dataset")
        print(f"{'='*50}\n")
        process_dataset(datatype)

    print("\nAll datasets processed successfully!")


if __name__ == "__main__":
    main()