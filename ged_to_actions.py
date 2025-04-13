"""
Simple GED Analysis - First step towards mapping GED operations to MoleculeDesign actions.
This script loads transformation data and provides a basic analysis of edit operations.
"""
import pickle
import os
import numpy as np
from rdkit.Chem import rdmolfiles, rdmolops
from rdkit import Chem, RDLogger
from collections import defaultdict

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Configuration
CHECKPOINT_DIR = "./data/chembl/checkpoints"
DEBUG = True  # Enable detailed debugging output
MAX_TRANSFORMATIONS = 10  # Process just one for debugging


def load_transformation_data(datatype="train"):
    """Load existing transformation data from latest checkpoint."""
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith(f"transformation_data_{datatype}_")]

    if checkpoint_files:
        # Sort by timestamp to get the most recent checkpoint
        checkpoint_files.sort(reverse=True)
        latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoint_files[0])
        print(f"Loading from checkpoint: {latest_checkpoint}")
        with open(latest_checkpoint, "rb") as f:
            transformation_data = pickle.load(f)
        print(f"Loaded {len(transformation_data)} transformations")
        return transformation_data[:MAX_TRANSFORMATIONS]  # Limit number of transformations

    print(f"No transformation data found for {datatype}")
    return []


def categorize_operations(edit_path):
    """
    Categorize GED operations into types and extract metadata.

    Args:
        edit_path: List of GED operations

    Returns:
        Dictionary with categorized operations
    """
    # Extract metadata (last operation)
    metadata = None
    for i, op in enumerate(edit_path):
        if op['operation'] == 'metadata':
            metadata = op
            edit_path = edit_path[:i]  # Remove metadata from edit path
            break

    print(f"Metadata: {metadata}")
    print(f"Total GED operations: {len(edit_path)}")

    # Categorize operations
    substitutions = []
    deletions = []
    insertions = []
    edge_operations = []

    for op in edit_path:
        if op['operation'] == 'substitute_node':
            substitutions.append(op)
        elif op['operation'] == 'delete_node':
            deletions.append(op)
        elif op['operation'] == 'insert_node':
            insertions.append(op)
        elif op['operation'] in ['insert_edge', 'delete_edge', 'substitute_edge']:
            edge_operations.append(op)

    print(f"Substitutions: {len(substitutions)}")
    print(f"Deletions: {len(deletions)}")
    print(f"Insertions: {len(insertions)}")
    print(f"Edge operations: {len(edge_operations)}")

    return {
        'metadata': metadata,
        'substitutions': substitutions,
        'deletions': deletions,
        'insertions': insertions,
        'edge_operations': edge_operations
    }


def identify_fragments_from_bond_removals(operations, source_mol):
    """
    Identify fragments that would be created by bond removal operations.

    Args:
        operations: Categorized operations from categorize_operations
        source_mol: Source RDKit molecule

    Returns:
        Dictionary with fragment information
    """
    print("\n=== Analyzing Bond Removals and Fragments ===")

    # Get bond deletion operations
    bond_removals = [op for op in operations['edge_operations'] if op['operation'] == 'delete_edge']
    if not bond_removals:
        print("No bond removal operations found")
        return {'fragments': [], 'connections': []}

    print(f"Found {len(bond_removals)} bond removal operations")

    fragments = []
    connections = []

    # Create a copy of the molecule that we'll modify
    working_mol = Chem.Mol(source_mol)

    # Process each bond removal and check for fragments
    for i, op in enumerate(bond_removals):
        atom1_idx = op['atom1_idx']
        atom2_idx = op['atom2_idx']

        print(f"\nBond removal {i + 1}: atoms {atom1_idx}-{atom2_idx}")

        # Make a copy of the current working molecule
        test_mol = Chem.Mol(working_mol)

        # Check if the atoms exist and if a bond exists between them
        if atom1_idx >= test_mol.GetNumAtoms() or atom2_idx >= test_mol.GetNumAtoms():
            print(f"  Atoms {atom1_idx} or {atom2_idx} out of range (molecule has {test_mol.GetNumAtoms()} atoms)")
            continue

        bond = test_mol.GetBondBetweenAtoms(atom1_idx, atom2_idx)
        if bond is None:
            print(f"  No bond found between atoms {atom1_idx}-{atom2_idx}")
            continue

        # Now we can proceed with bond removal
        bond_idx = bond.GetIdx()
        print(f"  Found bond at index {bond_idx}, removing...")

        # Use RWMol for easier editing
        rwmol = Chem.RWMol(test_mol)
        rwmol.RemoveBond(atom1_idx, atom2_idx)
        modified_mol = Chem.Mol(rwmol)

        # Check if this creates fragments
        atom_groups = Chem.GetMolFrags(modified_mol, asMols=False)

        if len(atom_groups) > 1:
            print(f"  Creates {len(atom_groups)} fragments")
            for j, atoms in enumerate(atom_groups):
                print(f"  Fragment {j + 1}: {len(atoms)} atoms - {list(atoms)}")

            # Add to our fragments list
            fragments.append({
                'operation_idx': i,
                'bond': (atom1_idx, atom2_idx),
                'fragments': atom_groups
            })

            # Record which fragment each atom belongs to
            for frag_idx, atoms in enumerate(atom_groups):
                for atom in atoms:
                    if atom in [atom1_idx, atom2_idx]:
                        connections.append({
                            'atom': atom,
                            'fragment_idx': frag_idx,
                            'operation_idx': i,
                            'is_connection_point': True
                        })
        else:
            print("  No fragmentation occurs")

        # Update our working molecule with the bond removed
        working_mol = modified_mol

    return {
        'fragments': fragments,
        'connections': connections
    }


def analyze_atom_additions(operations):
    """
    Analyze atom additions and their connections.

    Args:
        operations: Categorized operations

    Returns:
        Dictionary with atom addition information
    """
    insertions = operations['insertions']
    edge_operations = operations['edge_operations']

    print("\n=== Analyzing Atom Additions ===")
    print(f"Found {len(insertions)} atom insertions")

    # Track connections for each inserted atom
    atom_connections = defaultdict(list)

    # Find edges that connect inserted atoms
    for edge_op in edge_operations:
        if edge_op['operation'] == 'insert_edge':
            atom1_idx = edge_op['atom1_idx']
            atom2_idx = edge_op['atom2_idx']

            # Check if either end is an inserted atom
            for insertion in insertions:
                new_atom_idx = insertion['target_idx']
                if atom1_idx == new_atom_idx or atom2_idx == new_atom_idx:
                    # This edge connects to an inserted atom
                    existing_idx = atom2_idx if atom1_idx == new_atom_idx else atom1_idx

                    # Check if this is an existing atom or another inserted one
                    is_existing = True
                    for other_insertion in insertions:
                        if other_insertion['target_idx'] == existing_idx:
                            is_existing = False
                            break

                    connection_type = "existing atom" if is_existing else "new atom"

                    atom_connections[new_atom_idx].append({
                        'connects_to': existing_idx,
                        'bond_order': edge_op['bond_type'],
                        'connection_type': connection_type
                    })

    # Print connection information
    for atom_idx, connections in atom_connections.items():
        print(f"Atom {atom_idx} has {len(connections)} connections:")
        for conn in connections:
            print(f"  - Connects to {conn['connects_to']} with bond order {conn['bond_order']} ({conn['connection_type']})")

    return {
        'atom_insertions': insertions,
        'atom_connections': atom_connections
    }

def main():
    """Main function to analyze GED edit sequences"""

    # Load transformation data
    transformations = load_transformation_data("train")
    if not transformations:
        print("No transformations found to analyze.")
        return

    for i in range(len(transformations)):
        if i >= MAX_TRANSFORMATIONS:
            break
        # Process the first transformation
        transformation = transformations[i]
        source_smiles = transformation['source_smiles']
        target_smiles = transformation['target_smiles']
        edit_path = transformation['edit_path']

        print(f"\nAnalyzing transformation: {source_smiles} -> {target_smiles}")

        # Create RDKit molecules for analysis
        source_mol = Chem.MolFromSmiles(source_smiles)
        source_order = rdmolfiles.CanonicalRankAtoms(source_mol)
        # enforce canonical atom ordering
        source_mol = rdmolops.RenumberAtoms(source_mol, source_order)

        target_mol = Chem.MolFromSmiles(target_smiles)
        target_order = rdmolfiles.CanonicalRankAtoms(target_mol)
        # enforce canonical atom ordering
        target_mol = rdmolops.RenumberAtoms(target_mol, target_order)

        if not source_mol or not target_mol:
            print("Error creating RDKit molecules from SMILES")
            return

        # Categorize operations
        operations = categorize_operations(edit_path)

        # Identify fragments from bond removals
        fragment_info = identify_fragments_from_bond_removals(operations, source_mol)

        # Analyze atom additions
        addition_info = analyze_atom_additions(operations)

    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()