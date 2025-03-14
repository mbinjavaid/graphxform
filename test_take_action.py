import unittest
import numpy as np
from rdkit import Chem
from config import MoleculeConfig
from molecule_design import MoleculeDesign


class TestMoleculeDesign(unittest.TestCase):

    def setUp(self):
        # Create a minimal config for testing
        self.config = MoleculeConfig()
        self.config.atom_vocabulary = {
            "C": {"atomic_number": 6, "allowed": True, "valence": 4},
            "N": {"atomic_number": 7, "allowed": True, "valence": 3},
            "O": {"atomic_number": 8, "allowed": True, "valence": 2},
        }
        self.config.max_num_atoms = 10
        self.config.start_c_chain_max_len = 3

    def test_take_action_linear_molecule(self):
        """Test creating a linear molecule C-C-C."""
        # Start with a carbon atom
        mol = MoleculeDesign(self.config, initial_atom=1)  # 1 is the index for Carbon

        # Add second carbon and bond to first carbon with single bond
        mol.take_action(1)  # Add C
        mol.take_action(0)  # Select first C (index 0)
        mol.take_action(0)  # Create single bond (index 0)

        # Add third carbon and bond to second carbon with single bond
        mol.take_action(1)  # Add C
        mol.take_action(1)  # Select second C (index 1)
        mol.take_action(0)  # Create single bond (index 0)

        # Terminate
        mol.take_action(0)

        # Check the resulting structure
        print("Linear C-C-C molecule:")
        print(f"Atoms: {mol.atoms}")
        print(f"Bonds matrix:")
        print(mol.bonds)

        # Check SMILES string is correct
        self.assertEqual(Chem.CanonSmiles(mol.smiles_string), Chem.CanonSmiles("CCC"))

        # Check bonds matrix structure
        # [0, 7, 7, 7] - Virtual atom connections
        # [7, 0, 1, 0] - First C connected to second C
        # [7, 1, 0, 1] - Second C connected to first and third C
        # [7, 0, 1, 0] - Third C connected to second C
        expected_bonds = np.array([
            [0, 7, 7, 7],
            [7, 0, 1, 0],
            [7, 1, 0, 1],
            [7, 0, 1, 0]
        ])
        np.testing.assert_array_equal(mol.bonds, expected_bonds)

    def test_take_action_cyclic_molecule(self):
        """Test creating cyclopropane (C1CC1)."""
        # Start with a carbon atom
        mol = MoleculeDesign(self.config, initial_atom=1)  # 1 is the index for Carbon

        # Add second carbon and bond to first carbon
        mol.take_action(1)  # Add C
        mol.take_action(0)  # Bond to first C (index 0)
        mol.take_action(0)  # Single bond (index 0)

        # Add third carbon and bond to second carbon
        mol.take_action(1)  # Add C
        mol.take_action(1)  # Bond to second C (index 1)
        mol.take_action(0)  # Single bond (index 0)

        # Now close the cycle by bonding third carbon to first carbon
        # Calculate the appropriate action to select the third carbon atom
        select_third_c_action = 1 + len(self.config.atom_vocabulary.keys()) + 2  # 1 + 3 + 2 = 6
        print(f"Action to select third C: {select_third_c_action}")

        mol.take_action(select_third_c_action)  # Select third C
        mol.take_action(0)  # Bond to first C (index 0)
        mol.take_action(0)  # Single bond (index 0)

        # Terminate
        mol.take_action(0)

        # Check the resulting structure
        print("Cyclopropane molecule:")
        print(f"Atoms: {mol.atoms}")
        print(f"Bonds matrix:")
        print(mol.bonds)

        # Debug the actions history
        print(f"Actions history: {mol.history}")

        # Check SMILES string is correct for cyclopropane
        self.assertEqual(Chem.CanonSmiles(mol.smiles_string), Chem.CanonSmiles("C1CC1"))

        # Check bonds matrix structure for a cycle
        # [0, 7, 7, 7] - Virtual atom connections
        # [7, 0, 1, 1] - First C connected to second and third C
        # [7, 1, 0, 1] - Second C connected to first and third C
        # [7, 1, 1, 0] - Third C connected to first and second C
        expected_bonds = np.array([
            [0, 7, 7, 7],
            [7, 0, 1, 1],
            [7, 1, 0, 1],
            [7, 1, 1, 0]
        ])
        np.testing.assert_array_equal(mol.bonds, expected_bonds)

    def test_from_rdkit_mol_cyclic(self):
        """Test creating a molecule from RDKit's representation."""
        # Create a cyclopropane molecule with RDKit
        rdkit_mol = Chem.MolFromSmiles("C1CC1")
        Chem.SanitizeMol(rdkit_mol)
        smiles = Chem.MolToSmiles(rdkit_mol)

        # Convert to our molecule design
        mol = MoleculeDesign.from_rdkit_mol(self.config, rdkit_mol, smiles)

        # Check the resulting structure
        print("Cyclopropane from RDKit:")
        print(f"Atoms: {mol.atoms}")
        print(f"Bonds matrix:")
        print(mol.bonds)

        # Check SMILES string is correct
        self.assertEqual(Chem.CanonSmiles(mol.smiles_string), Chem.CanonSmiles("C1CC1"))

        # Check bonds matrix structure for a cycle
        expected_bonds = np.array([
            [0, 7, 7, 7],
            [7, 0, 1, 1],
            [7, 1, 0, 1],
            [7, 1, 1, 0]
        ])
        np.testing.assert_array_equal(mol.bonds, expected_bonds)

    def test_detailed_cyclic_molecule(self):
        """Test creating cyclopropane (C1CC1) with detailed debugging."""
        config = MoleculeConfig()
        config.atom_vocabulary = {"C": {"atomic_number": 6, "allowed": True, "valence": 4}}
        config.max_num_atoms = 10

        # Start with a carbon atom
        mol = MoleculeDesign(config, initial_atom=1)
        print("After creating first C:")
        print(f"Atoms: {mol.atoms}")
        print(f"Bonds:\n{mol.bonds}")

        # Add second carbon and bond to first carbon
        mol.take_action(1)  # Add C
        print("After adding second C:")
        print(f"Atoms: {mol.atoms}")
        print(f"Bonds:\n{mol.bonds}")

        mol.take_action(0)  # Bond to first C (index 0)
        print("After selecting first C to bond with:")
        print(f"Bonds:\n{mol.bonds}")

        mol.take_action(0)  # Single bond (index 0)
        print("After creating C1-C2 bond:")
        print(f"Bonds:\n{mol.bonds}")

        # Add third carbon and bond to second carbon
        mol.take_action(1)  # Add C
        print("After adding third C:")
        print(f"Atoms: {mol.atoms}")
        print(f"Bonds:\n{mol.bonds}")

        mol.take_action(1)  # Bond to second C (index 1)
        print("After selecting second C to bond with:")
        print(f"Bonds:\n{mol.bonds}")

        mol.take_action(0)  # Single bond (index 0)
        print("After creating C2-C3 bond:")
        print(f"Bonds:\n{mol.bonds}")

        # Now close the cycle by bonding third carbon to first carbon
        select_third_c_action = 1 + len(config.atom_vocabulary.keys()) + 2  # 1 + 1 + 2 = 4
        print(f"Action to select third C: {select_third_c_action}")

        mol.take_action(select_third_c_action)  # Select third C
        print("After selecting third C to create cycle:")
        print(f"Bonds:\n{mol.bonds}")

        mol.take_action(0)  # Bond to first C (index 0)
        print("After selecting first C to bond with:")
        print(f"Bonds:\n{mol.bonds}")

        mol.take_action(0)  # Single bond (index 0)
        print("After creating C3-C1 bond to close cycle:")
        print(f"Bonds:\n{mol.bonds}")

        # Terminate
        mol.take_action(0)
        print("Final molecule:")
        print(f"SMILES: {mol.smiles_string}")
        print(f"Atoms: {mol.atoms}")
        print(f"Bonds:\n{mol.bonds}")


if __name__ == "__main__":
    unittest.main()