import unittest
from rdkit import Chem
from molecule_design import MoleculeDesign
from config import MoleculeConfig


class TestModifyExistingBonds(unittest.TestCase):
    def test_modify_existing_bonds_from_smiles(self):
        # Setup configuration to initialize from an existing SMILES string.
        config = MoleculeConfig()
        config.start_from_smiles = "CCO"  # Ethanol
        # Create a MoleculeDesign instance from the provided SMILES.
        md = MoleculeDesign.from_smiles(config, config.start_from_smiles, do_finish=False)

        # In the generated molecule, we expect three atoms (two carbons and one oxygen).
        # For this test, we'll focus on the bond between the two carbon atoms.
        rdkit_mol = md.rdkit_mol
        # Get indices of the carbon atoms from the RDKit molecule.
        carbon_indices = [atom.GetIdx() for atom in rdkit_mol.GetAtoms() if atom.GetSymbol() == "C"]
        print("carbon indices: ", carbon_indices)
        self.assertTrue(len(carbon_indices) >= 2, "Not enough carbon atoms to proceed with the test.")
        atom_a = carbon_indices[0]
        atom_b = carbon_indices[1]

        print(atom_a)

        # Verify the original bond order between the two carbons.
        original_bond = rdkit_mol.GetBondBetweenAtoms(atom_a, atom_b)
        # print("original bond: ", original_bond)
        original_order = original_bond.GetBondTypeAsDouble() if original_bond else 0
        self.assertEqual(original_order, 1, "Expected initial carbon-carbon bond to be a single bond (order 1).")

        # Now, simulate a modification of the bond order.
        # In the MoleculeDesign.take_action method at level 2, an action value of 1 would set the bond order to 2.
        # For testing purposes—since routing through the full RL action history can be tricky—we will directly
        # update the bonds matrix and invoke update_rdkit_mol to apply the change.
        # Adjust the bonds matrix (note: index 0 corresponds to the virtual atom, so we add 1).
        md.bonds[atom_a + 1, atom_b + 1] = 2
        md.bonds[atom_b + 1, atom_a + 1] = 2
        # Call update_rdkit_mol to reflect these changes in the RDKit molecule.
        md.update_rdkit_mol(set_bond=(atom_a, atom_b, 2))

        # Re-extract the bond from the RDKit molecule after modification.
        updated_bond = md.rdkit_mol.GetBondBetweenAtoms(atom_a, atom_b)
        updated_order = updated_bond.GetBondTypeAsDouble() if updated_bond else 0

        self.assertEqual(updated_order, 2, f"Expected bond order to be updated to 2, got {updated_order}")


if __name__ == "__main__":
    unittest.main()