import numpy as np
import pytest
from molecule_design import MoleculeDesign
from rdkit import Chem
import matplotlib.pyplot as plt
from rdkit.Chem import Draw


# Create a dummy configuration for testing.
class DummyConfig:
    def __init__(self):
        self.atom_vocabulary = {
            "C": {"atomic_number": 6, "valence": 4, "allowed": True},
            "N": {"atomic_number": 7, "valence": 3, "allowed": True},
            "O": {"atomic_number": 8, "valence": 2, "allowed": True}
        }
        # Allow up to 4 real atoms (plus the virtual atom)
        self.max_num_atoms = 4
        self.start_c_chain_max_len = 3


def create_dummy_config():
    return DummyConfig()


@pytest.fixture
def config():
    return create_dummy_config()


@pytest.fixture
def mol(config):
    # Start with an initial atom A (using "C", index 1)
    return MoleculeDesign(config, initial_atom=1)


def visualize_molecule(mol, title="Molecule Structure", highlight_atoms=None, highlight_bond=None):
    """Helper function to visualize the molecule structure with highlighted atoms/bonds"""
    rdkit_mol = mol.rdkit_mol

    # Create a copy to avoid modifying the original
    viz_mol = Chem.RWMol(rdkit_mol)

    # Add atom indices for clarity
    for atom in viz_mol.GetAtoms():
        atom.SetProp("atomNote", str(atom.GetIdx()))

    # Set up highlighting
    highlight_atom_list = highlight_atoms or []
    highlight_bond_list = []

    if highlight_bond:
        # Convert from internal indices to RDKit indices
        atom1, atom2 = highlight_bond
        rdkit_atom1 = atom1 - 1  # -1 because internal has virtual atom
        rdkit_atom2 = atom2 - 1

        # Find the bond index
        bond = viz_mol.GetBondBetweenAtoms(rdkit_atom1, rdkit_atom2)
        if bond:
            highlight_bond_list = [bond.GetIdx()]

    # Generate the image
    img = Draw.MolToImage(viz_mol, size=(300, 300),
                          highlightAtoms=highlight_atom_list,
                          highlightBonds=highlight_bond_list)

    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def test_bond_addition_and_reduction(mol):
    """
    Integration test scenario navigating through the proper action space.
    """
    vocab_size = len(mol.vocabulary_atom_idcs)

    # Steps 1-2: Initial setup with atom A, then add atom B
    # At level 0, select atom A (index 1)
    mol.take_action(1)  # Select atom 1 (A)

    # At level 1, create new atom B (carbon, index 0 in vocabulary)
    mol.take_action(0)  # Create new atom B (C)

    # At level 2, set bond order 1
    mol.take_action(vocab_size + 0)  # Create single bond (V+0)

    # Step 3: Add atom C and set B-C bond to order 2
    # At level 0, select atom B (index 2)
    mol.take_action(2)  # Select atom 2 (B)

    # At level 1, create new atom C (carbon, index 0 in vocabulary)
    mol.take_action(0)  # Create new atom C (C)

    # At level 2, set bond order 2
    mol.take_action(vocab_size + 1)  # Create double bond (V+1)
    assert mol.bonds[2, 3] == 2

    # Print current state
    print("\nAfter Step 3:")
    print(f"Atoms: {mol.atoms}")
    print(f"Bonds matrix:\n{mol.bonds}")
    print(f"Current action level: {mol.current_action_level}")

    # === Step 4: Add A-C bond with order 1 ===
    # At Level 0, select atom A (index 1)
    print(f"\nStep 4: Selecting atom A (index 1)")
    assert mol.current_action_level == 0
    mol.take_action(1)  # Directly select atom 1

    # At Level 1, select atom C (index 3)
    atom_c_action_idx = vocab_size + 2  # V+3 for atom at RDKit index 3 (internal index 2)
    print(f"Step 4: Selecting atom C with action {atom_c_action_idx}")
    assert mol.current_action_level == 1
    mol.take_action(atom_c_action_idx)

    # At Level 2, set bond order 1 (V+0 for bond order 1)
    print("Step 4: Setting A-C bond to order 1")
    assert mol.current_action_level == 2
    mol.take_action(vocab_size + 0)  # Action V+0 = set bond order 1
    assert mol.bonds[1, 3] == 1

    # === Step 5: Reduce B-C bond from order 2 to 1 ===
    # At Level 0, select atom B (index 2)
    print(f"\nStep 5: Selecting atom B (index 2)")
    assert mol.current_action_level == 0
    mol.take_action(2)  # Directly select atom 2

    # At Level 1, select atom C (index 3)
    print(f"Step 5: Selecting atom C with action {atom_c_action_idx}")
    assert mol.current_action_level == 1
    mol.take_action(atom_c_action_idx)

    # At Level 2, set bond order 1 (reducing from 2)
    print("Step 5: Setting B-C bond to order 1")
    assert mol.current_action_level == 2
    mol.take_action(vocab_size + 0)  # Action V+0 = set bond order 1
    assert mol.bonds[2, 3] == 1

    # === Step 6: Create A-B bond to ensure connectivity ===
    # At Level 0, select atom A (index 1)
    print(f"\nStep 6: Selecting atom A (index 1)")
    assert mol.current_action_level == 0
    mol.take_action(1)  # Directly select atom 1

    # At Level 1, select atom B (index 2)
    atom_b_level1_action_idx = vocab_size + 1  # V+2 for atom at RDKit index 2 (internal index 1)
    print(f"Step 6: Selecting atom B with action {atom_b_level1_action_idx}")
    assert mol.current_action_level == 1
    mol.take_action(atom_b_level1_action_idx)

    # At Level 2, set bond order 1
    print("Step 6: Setting A-B bond to order 1")
    assert mol.current_action_level == 2
    mol.take_action(vocab_size + 0)  # Action V+0 = set bond order 1
    assert mol.bonds[1, 2] == 1

    # Visualize molecule after creating A-B bond
    print("\nMolecule structure after adding A-B bond:")
    print(f"Bonds matrix:\n{mol.bonds}")
    visualize_molecule(mol, "After Adding A-B Bond", highlight_bond=(1, 2))

    # Check connectivity
    is_connected_ab_bc = mol.is_connected_without_bond(1, 3)
    print(f"Would molecule remain connected without A-C bond? {is_connected_ab_bc}")

    # === Step 7: Remove A-C bond ===
    # At Level 0, select atom A (index 1)
    print(f"\nStep 7: Selecting atom A (index 1)")
    assert mol.current_action_level == 0
    mol.take_action(1)  # Directly select atom 1

    # At Level 1, select atom C (index 3)
    print(f"Step 7: Selecting atom C with action {atom_c_action_idx}")
    assert mol.current_action_level == 1
    mol.take_action(atom_c_action_idx)

    # Check action mask
    print(f"Action mask at level 2: {mol.current_action_mask}")

    # In new action space, remove bond is action V+6
    print(f"Remove bond action (index {vocab_size + 6}) masked? {mol.current_action_mask[vocab_size + 6]}")

    # At Level 2, remove bond
    print(f"Step 7: Removing A-C bond with action {vocab_size + 6}")
    assert mol.current_action_level == 2
    mol.take_action(vocab_size + 6)  # Action V+6 = remove bond
    assert mol.bonds[1, 3] == 0

    # Final visualization
    print("\nFinal molecule structure:")
    print(f"Bonds matrix:\n{mol.bonds}")
    visualize_molecule(mol, "After A-C Bond Removal")