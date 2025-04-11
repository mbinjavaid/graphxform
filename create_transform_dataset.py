"""
This script generates paired molecule transformation datasets.
We find pairs of molecules, calculate their Graph Edit Distance (GED),
and record the edit sequences required to transform one into the other.
Includes checkpointing to save intermediate results.
"""
import time
import pickle
import os
import random
from collections import defaultdict
import numpy as np
import networkx as nx
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional, Set, Any
from datetime import datetime

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Configuration
MAX_ATOMS = 50
SIMILARITY_LOWER_BOUND = 0.4  # Min similarity for pairs
SIMILARITY_UPPER_BOUND = 0.7  # Max similarity for pairs
# NUM_PAIRS = 10000  # Target number of pairs to generate
NUM_PAIRS = int(1574970*1)  # Target number of pairs to generate
RANDOM_SEED = 42
CHECKPOINT_DIR = "./data/chembl/checkpoints"
CHECKPOINT_FREQUENCY = 100  # Save progress after every 100 pairs

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
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "filtered_molecules.pkl")

    # Try to load from checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading filtered molecules from checkpoint {checkpoint_path}")
        try:
            with open(checkpoint_path, "rb") as f:
                molecules = pickle.load(f)
            print(f"Loaded {len(molecules)} molecules from checkpoint")
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


def calculate_similarity(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    """
    Calculate Tanimoto similarity between two molecules using Morgan fingerprints.

    Args:
        mol1: First molecule
        mol2: Second molecule

    Returns:
        Similarity score between 0 and 1
    """
    # Generate Morgan fingerprints with radius 2 (ECFP4-like)
    fp1 = AllChem.GetMorganFingerprint(mol1, 2)
    fp2 = AllChem.GetMorganFingerprint(mol2, 2)

    # Calculate Tanimoto similarity
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def create_molecule_pairs(
        molecules: List[Tuple[Chem.Mol, str]],
        similarity_lower: float = SIMILARITY_LOWER_BOUND,
        similarity_upper: float = SIMILARITY_UPPER_BOUND,
        num_pairs: int = NUM_PAIRS,
        max_pairs_per_molecule: int = 1000  # New parameter to limit pairs per molecule
) -> List[Tuple[Chem.Mol, Chem.Mol, float]]:
    """
    Create pairs of molecules within a specified similarity range using optimized approach.
    Each molecule can appear in up to max_pairs_per_molecule pairs.
    Uses checkpointing for fingerprints and progress.

    Args:
        molecules: List of (molecule, SMILES) tuples
        similarity_lower: Minimum similarity threshold
        similarity_upper: Maximum similarity threshold
        num_pairs: Target number of pairs to generate
        max_pairs_per_molecule: Maximum number of pairs a single molecule can appear in

    Returns:
        List of (mol1, mol2, similarity) tuples
    """
    # Define checkpoint paths
    fingerprint_checkpoint_path = os.path.join(CHECKPOINT_DIR, "molecule_fingerprints.pkl")
    pairs_checkpoint_path = os.path.join(CHECKPOINT_DIR, "molecule_pairs.pkl")
    pairs_progress_checkpoint_path = os.path.join(CHECKPOINT_DIR, "pairs_progress.pkl")

    # Initialize state variables
    fingerprints = []
    mol_smiles_map = {}

    # Step 1: Load or compute fingerprints
    if os.path.exists(fingerprint_checkpoint_path):
        print(f"Loading fingerprints from checkpoint {fingerprint_checkpoint_path}")
        try:
            with open(fingerprint_checkpoint_path, "rb") as f:
                fingerprint_data = pickle.load(f)
                fingerprints = fingerprint_data['fingerprints']
                mol_smiles_map = fingerprint_data['mol_smiles_map']
            print(f"Loaded {len(fingerprints)} fingerprints from checkpoint")
        except Exception as e:
            print(f"Failed to load fingerprint checkpoint: {e}. Computing from scratch.")
            fingerprints = []
            mol_smiles_map = {}

    # Compute fingerprints if not loaded from checkpoint
    if not fingerprints:
        print("Pre-computing all fingerprints...")
        for i, (mol, smiles) in enumerate(tqdm(molecules)):
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fingerprints.append(fp)
            mol_smiles_map[i] = (mol, smiles)

        # Save fingerprint checkpoint
        fingerprint_data = {
            'fingerprints': fingerprints,
            'mol_smiles_map': mol_smiles_map
        }
        with open(fingerprint_checkpoint_path, "wb") as f:
            pickle.dump(fingerprint_data, f)
        print(f"Saved {len(fingerprints)} fingerprints to {fingerprint_checkpoint_path}")

    # Step 2: Try to load final pairs result
    if os.path.exists(pairs_checkpoint_path):
        print(f"Loading final molecule pairs from checkpoint {pairs_checkpoint_path}")
        try:
            with open(pairs_checkpoint_path, "rb") as f:
                pairs = pickle.load(f)
            print(f"Loaded {len(pairs)} molecule pairs from final checkpoint")
            return pairs
        except Exception as e:
            print(f"Failed to load final pairs checkpoint: {e}. Continuing from progress checkpoint if available.")

    # Step 3: Try to load progress checkpoint
    pairs = []
    molecule_pair_counts = defaultdict(int)  # Replace used_indices with a counter
    processed_count = 0
    indices = list(range(len(molecules)))
    random.shuffle(indices)

    # Variable to track checkpoint milestones
    prev_checkpoint_milestone = 0

    if os.path.exists(pairs_progress_checkpoint_path):
        print(f"Loading pairs progress from checkpoint {pairs_progress_checkpoint_path}")
        try:
            with open(pairs_progress_checkpoint_path, "rb") as f:
                progress_data = pickle.load(f)
                pairs = progress_data['pairs']

                # Handle backward compatibility with old checkpoint format
                if 'molecule_pair_counts' in progress_data:
                    molecule_pair_counts = progress_data['molecule_pair_counts']
                elif 'used_indices' in progress_data:
                    # Convert old format to new format
                    used_indices = progress_data['used_indices']
                    for idx in used_indices:
                        molecule_pair_counts[idx] = 1
                    print("Converted old checkpoint format to new format")

                processed_count = progress_data['processed_count']

            # Calculate actual molecules used statistics from pairs
            molecules_in_pairs = set()
            for mol1, mol2, _ in pairs:
                for i, (m, _) in enumerate(molecules):
                    if Chem.MolToSmiles(m) == Chem.MolToSmiles(mol1) or Chem.MolToSmiles(m) == Chem.MolToSmiles(mol2):
                        molecules_in_pairs.add(i)

            print(f"Resuming from checkpoint: {len(pairs)} pairs found, {len(molecules_in_pairs)} molecules used, "
                  f"processed {processed_count} indices")

            # Set the checkpoint milestone based on loaded data
            prev_checkpoint_milestone = len(pairs) // 5000

        except Exception as e:
            print(f"Failed to load progress checkpoint: {e}. Starting from beginning.")
            pairs = []
            molecule_pair_counts = defaultdict(int)
            processed_count = 0

    print(f"Finding molecule pairs with similarity between {similarity_lower} and {similarity_upper}")
    print(f"Each molecule can appear in up to {max_pairs_per_molecule} pairs")

    # Step 4: Find pairs
    pbar = tqdm(total=num_pairs, initial=len(pairs))

    for idx, i in enumerate(indices):
        # Skip already processed indices
        if idx < processed_count:
            continue

        if len(pairs) >= num_pairs:
            break

        # Skip if molecule has reached its maximum allowed pairs
        if molecule_pair_counts[i] >= max_pairs_per_molecule:
            processed_count += 1
            continue

        mol1, smiles1 = mol_smiles_map[i]
        fp1 = fingerprints[i]

        # Calculate similarities in batches
        batch_size = 1000000
        # Only consider molecules that haven't reached their pair limit
        remaining_indices = [j for j in indices if j > i and molecule_pair_counts[j] < max_pairs_per_molecule]

        for batch_start in range(0, len(remaining_indices), batch_size):
            batch_indices = remaining_indices[batch_start:batch_start + batch_size]

            # Use bulk similarity calculation
            batch_fps = [fingerprints[j] for j in batch_indices]

            # Calculate similarities for the batch
            similarities = DataStructs.BulkTanimotoSimilarity(fp1, batch_fps)

            # Find matches in the right similarity range
            for k, sim in enumerate(similarities):
                if similarity_lower <= sim <= similarity_upper:
                    j = batch_indices[k]

                    # Double-check that neither molecule has reached its limit
                    # (could happen if another batch just paired it)
                    if (molecule_pair_counts[i] >= max_pairs_per_molecule or
                            molecule_pair_counts[j] >= max_pairs_per_molecule):
                        continue

                    mol2, smiles2 = mol_smiles_map[j]

                    pairs.append((mol1, mol2, sim))
                    molecule_pair_counts[i] += 1
                    molecule_pair_counts[j] += 1
                    pbar.update(1)

                    # Check if we've crossed a checkpoint milestone (every 5000 pairs)
                    current_milestone = len(pairs) // 5000
                    if current_milestone > prev_checkpoint_milestone:
                        progress_data = {
                            'pairs': pairs,
                            'molecule_pair_counts': molecule_pair_counts,
                            'processed_count': processed_count
                        }
                        with open(pairs_progress_checkpoint_path, "wb") as f:
                            pickle.dump(progress_data, f)

                        # Calculate statistics about molecule reuse
                        used_molecules = sum(1 for count in molecule_pair_counts.values() if count > 0)
                        max_reuse = max(molecule_pair_counts.values()) if molecule_pair_counts else 0
                        avg_reuse = sum(molecule_pair_counts.values()) / used_molecules if used_molecules > 0 else 0

                        print(
                            f"\nCheckpoint saved at {len(pairs)} pairs: {processed_count}/{len(indices)} molecules processed")
                        print(
                            f"Molecules used: {used_molecules}, Max pairs per molecule: {max_reuse}, Avg pairs per molecule: {avg_reuse:.2f}")

                        # Update the milestone
                        prev_checkpoint_milestone = current_milestone

                    if len(pairs) >= num_pairs:
                        break

                    # If this molecule has reached its limit, stop looking for more partners
                    if molecule_pair_counts[i] >= max_pairs_per_molecule:
                        break

            if len(pairs) >= num_pairs or molecule_pair_counts[i] >= max_pairs_per_molecule:
                break

        # Update processed count
        processed_count += 1

    pbar.close()
    print(f"Created {len(pairs)} molecule pairs")

    # Save final checkpoint
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


def calculate_edit_path(mol1: Chem.Mol, mol2: Chem.Mol) -> List[Tuple[str, Any, Any]]:
    """
    Calculate the edit path between two molecules using Graph Edit Distance.

    Args:
        mol1: Source molecule
        mol2: Target molecule

    Returns:
        Edit path as a list of (operation, source, target) tuples
    """
    # Convert molecules to NetworkX graphs
    graph1 = mol_to_nx_graph(mol1)
    graph2 = mol_to_nx_graph(mol2)

    # Get the edit path directly using optimize_edit_paths
    path_generator = nx.optimize_edit_paths(
        graph1, graph2,
        node_match=node_match,
        edge_match=edge_match
    )

    # Take the first (optimal) path
    node_edit_path, edge_edit_path, cost = next(path_generator)

    # Convert the path to a readable format
    readable_path = []

    # Process node operations
    for u, v in node_edit_path:
        if u is None:  # Node insertion (v was inserted)
            readable_path.append(('insert_node', None, v))
        elif v is None:  # Node deletion (u was deleted)
            readable_path.append(('delete_node', u, None))
        elif not node_match(graph1.nodes[u], graph2.nodes[v]):  # Node substitution
            readable_path.append(('substitute_node', u, v))

    # Process edge operations
    for edge1, edge2 in edge_edit_path:
        if edge1 is None:  # Edge insertion
            readable_path.append(('insert_edge', None, edge2))
        elif edge2 is None:  # Edge deletion
            readable_path.append(('delete_edge', edge1, None))
        else:  # Edge might be substituted (if attributes don't match)
            if not edge_match(graph1.edges[edge1], graph2.edges[edge2]):
                readable_path.append(('substitute_edge', edge1, edge2))

    return readable_path


def main():
    """Main function to create the transformation dataset with checkpointing"""
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Checkpoint path for transformation data
    transformation_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"transformation_data_{timestamp}.pkl")
    final_output_path = "./data/chembl/transformation_dataset.pickle"

    # Load previously saved transformation data if exists
    transformation_data = []
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("transformation_data_")]

    if checkpoint_files:
        # Sort by timestamp to get the most recent checkpoint
        checkpoint_files.sort(reverse=True)
        latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoint_files[0])
        print(f"Found transformation checkpoint: {latest_checkpoint}")
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
    molecules = load_and_filter_molecules("./data/chembl/chembl_train_filtered.smiles")

    # 2. Create molecule pairs
    pairs = create_molecule_pairs(molecules)

    # 3. Calculate GED and edit paths for each pair
    print("Calculating Graph Edit Distances and edit paths")
    pairs_to_process = []
    for mol1, mol2, similarity in pairs:
        smiles1 = Chem.MolToSmiles(mol1)
        smiles2 = Chem.MolToSmiles(mol2)
        if (smiles1, smiles2) not in processed_pairs:
            pairs_to_process.append((mol1, mol2, similarity, smiles1, smiles2))

    print(f"Processing {len(pairs_to_process)} remaining pairs")

    for i, (mol1, mol2, similarity, smiles1, smiles2) in enumerate(tqdm(pairs_to_process)):
        try:
            edit_path = calculate_edit_path(mol1, mol2)

            # Store the transformation data
            transformation = {
                'source_smiles': smiles1,
                'target_smiles': smiles2,
                'similarity': similarity,
                'edit_path': edit_path
            }
            transformation_data.append(transformation)

            # Save checkpoint periodically
            if (i + 1) % CHECKPOINT_FREQUENCY == 0:
                with open(transformation_checkpoint_path, "wb") as f:
                    pickle.dump(transformation_data, f)
                print(f"Checkpoint saved after {len(transformation_data)} transformations")

        except Exception as e:
            print(f"Error processing pair ({smiles1}, {smiles2}): {str(e)}")

    # 4. Save the final transformation dataset
    with open(final_output_path, "wb") as f:
        pickle.dump(transformation_data, f)

    # Also save final checkpoint
    with open(transformation_checkpoint_path, "wb") as f:
        pickle.dump(transformation_data, f)

    print(f"Saved {len(transformation_data)} transformations to {final_output_path}")
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")

    # Load and display transformation data
    print("\n===== Viewing Sample Transformation Data =====")
    try:
        # Load the transformation data if we don't already have it
        if not transformation_data:
            with open(final_output_path, "rb") as f:
                transformation_data = pickle.load(f)

        # Display 10 samples (or fewer if we have less than 10)
        sample_size = min(10, len(transformation_data))
        print(f"\nShowing {sample_size} sample transformations:")

        for i, transform in enumerate(transformation_data[:sample_size]):
            print(f"\n[{i + 1}] Source → Target (similarity: {transform['similarity']:.3f})")
            print(f"Source SMILES: {transform['source_smiles']}")
            print(f"Target SMILES: {transform['target_smiles']}")

            # Print first 5 edit operations (or all if fewer than 5)
            print("Edit sequence (first 5 operations):")
            for j, operation in enumerate(transform['edit_path']):
                print(f"  {j + 1}. {operation[0]}: {operation[1]} → {operation[2]}")
                print(f"Total number of operations: {len(transform['edit_path'])})")

    except Exception as e:
        print(f"Error displaying transformation data: {e}")


if __name__ == "__main__":
    main()