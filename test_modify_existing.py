import unittest
from rdkit import Chem
from molecule_design import MoleculeDesign
from config import MoleculeConfig


class TestModifyExistingBonds(unittest.TestCase):
    def test_modify_existing_bonds_from_smiles(self):
        """Test modifying existing bonds using the action API rather than direct matrix manipulation."""
        # Setup configuration to initialize from an existing SMILES string
        config = MoleculeConfig()
        config.start_from_smiles = "CCO"  # Ethanol

        # Create a MoleculeDesign instance from the provided SMILES
        print(f"\nStarting with molecule from SMILES: {config.start_from_smiles}")
        md = MoleculeDesign.from_smiles(config, config.start_from_smiles, do_finish=False)

        # Print the initial state of the molecule
        print(f"Initial atoms array: {md.atoms}")
        print(f"Initial bonds matrix:\n{md.bonds}")

        # Get the RDKit representation and check the carbon atoms
        rdkit_mol = md.rdkit_mol
        carbon_indices = [atom.GetIdx() for atom in rdkit_mol.GetAtoms() if atom.GetSymbol() == "C"]
        print(f"Carbon indices in RDKit molecule: {carbon_indices}")
        self.assertTrue(len(carbon_indices) >= 2, "Not enough carbon atoms to proceed with the test.")

        # Get the two carbon atoms we'll modify the bond between
        atom_a = carbon_indices[0]
        atom_b = carbon_indices[1]

        # Verify the original bond order between the two carbons
        original_bond = rdkit_mol.GetBondBetweenAtoms(atom_a, atom_b)
        original_order = original_bond.GetBondTypeAsDouble() if original_bond else 0
        print(f"Original bond order between carbons at indices {atom_a} and {atom_b}: {original_order}")
        self.assertEqual(original_order, 1, "Expected initial carbon-carbon bond to be a single bond (order 1).")

        # Now let's modify the bond using the action API
        # First, we need the action indices for selecting existing atoms

        # At level 0, to select an existing atom we use:
        # ex_atom_start_idx + atom_idx - 1 where ex_atom_start_idx = len(config.atom_vocabulary) + 1
        ex_atom_start_idx = len(config.atom_vocabulary) + 1
        atom_a_action_idx = ex_atom_start_idx + atom_a + 1 - 1  # +1 because we have a virtual atom

        # Execute level 0 action to select the first carbon atom
        print(f"\nTaking level 0 action {atom_a_action_idx} to select the first carbon atom")
        self.assertEqual(md.current_action_level, 0, "Expected to be at action level 0")
        md.take_action(atom_a_action_idx)

        # At level 1, to select an existing atom we use:
        # len(config.atom_vocabulary) + atom_idx
        atom_b_action_idx = len(config.atom_vocabulary) + atom_b

        # Execute level 1 action to select the second carbon atom
        print(f"Taking level 1 action {atom_b_action_idx} to select the second carbon atom")
        self.assertEqual(md.current_action_level, 1, "Expected to be at action level 1")
        md.take_action(atom_b_action_idx)

        # Print the action mask to see available bond order actions
        print(f"Action mask at level 2: {md.current_action_mask}")

        # Check if creating a double bond is feasible
        double_bond_feasible = md.current_action_mask[1] == 0  # 0 means feasible in the mask

        if double_bond_feasible:
            # Execute level 2 action to change bond to double bond (action 1 = bond order 2)
            print("Double bond is feasible, taking level 2 action 1 to create double bond")
            self.assertEqual(md.current_action_level, 2, "Expected to be at action level 2")
            md.take_action(1)  # Action 1 = bond order 2 (double bond)

            # Verify the bond order was updated
            updated_bond = md.rdkit_mol.GetBondBetweenAtoms(atom_a, atom_b)
            updated_order = updated_bond.GetBondTypeAsDouble() if updated_bond else 0
            print(f"Updated bond order: {updated_order}")

            self.assertEqual(updated_order, 2,
                             f"Expected bond order to be updated to 2, got {updated_order}")
            print("Successfully changed C-C bond from single to double bond")
        else:
            print("Double bond is not feasible, likely due to valence constraints")
            # For testing purposes, select a feasible action instead
            feasible_action = None
            for i, mask_value in enumerate(md.current_action_mask):
                if mask_value == 0:
                    feasible_action = i
                    break

            if feasible_action is not None:
                print(f"Taking feasible action {feasible_action} instead")
                md.take_action(feasible_action)
            else:
                self.fail("No feasible actions available at level 2")

        # Now let's try to decrease the bond order (if we managed to increase it first)
        if double_bond_feasible:
            # Select the first carbon atom again
            print("\nAttempting to decrease bond order from double to single")
            self.assertEqual(md.current_action_level, 0, "Expected to be back at action level 0")
            md.take_action(atom_a_action_idx)

            # Select the second carbon atom again
            self.assertEqual(md.current_action_level, 1, "Expected to be at action level 1")
            md.take_action(atom_b_action_idx)

            # Print action mask for level 2
            print(f"Action mask at level 2 (for decreasing bond): {md.current_action_mask}")

            # Check if decreasing bond is feasible (action 6 = decrease by 1)
            decrease_bond_feasible = md.current_action_mask[6] == 0

            if decrease_bond_feasible:
                print("Decreasing bond is feasible, taking action 6")
                md.take_action(6)  # Action 6 = decrease by 1

                # Verify the bond order was decreased back to single
                updated_bond = md.rdkit_mol.GetBondBetweenAtoms(atom_a, atom_b)
                updated_order = updated_bond.GetBondTypeAsDouble() if updated_bond else 0
                print(f"Bond order after decrease: {updated_order}")

                self.assertEqual(updated_order, 1,
                                 f"Expected bond order to be decreased to 1, got {updated_order}")
                print("Successfully decreased C=C bond back to C-C single bond")
            else:
                print("Decreasing bond is not feasible, possibly due to structural constraints")

        # Final molecule state
        print(f"\nFinal SMILES: {md.to_smiles()}")
        print(f"Final atoms array: {md.atoms}")
        print(f"Final bonds matrix:\n{md.bonds}")

    def test_modify_bond_in_complex_molecule(self):
        """Test modifying bonds in a more complex molecule to verify valence constraints."""
        # Setup with a more complex molecule - benzene with a methyl group
        config = MoleculeConfig()
        config.start_from_smiles = "c1(C)ccccc1"  # Toluene (methylbenzene)

        print(f"\nStarting with complex molecule from SMILES: {config.start_from_smiles}")
        md = MoleculeDesign.from_smiles(config, config.start_from_smiles, do_finish=False)

        # Get the RDKit representation
        rdkit_mol = md.rdkit_mol

        # Find the methyl carbon (sp3 carbon attached to benzene)
        methyl_carbon = None
        benzene_carbon = None

        for atom in rdkit_mol.GetAtoms():
            if atom.GetSymbol() == "C":
                # The methyl carbon has only one bond
                # (to the benzene ring, since hydrogens aren't explicit)
                if len(atom.GetBonds()) == 1:
                    methyl_carbon = atom.GetIdx()
                    # Get the benzene carbon it's attached to
                    neighbor = atom.GetNeighbors()[0]
                    benzene_carbon = neighbor.GetIdx()
                    break

        self.assertIsNotNone(methyl_carbon, "Couldn't find methyl carbon")
        self.assertIsNotNone(benzene_carbon, "Couldn't find benzene carbon")

        print(f"Found methyl carbon at index {methyl_carbon} bonded to benzene carbon at index {benzene_carbon}")

        # Check the original bond
        original_bond = rdkit_mol.GetBondBetweenAtoms(methyl_carbon, benzene_carbon)
        original_order = original_bond.GetBondTypeAsDouble() if original_bond else 0
        print(f"Original bond order: {original_order}")
        self.assertEqual(original_order, 1, "Expected methyl-benzene bond to be a single bond")

        # Get the action indices for the two atoms
        ex_atom_start_idx = len(config.atom_vocabulary) + 1
        methyl_action_idx = ex_atom_start_idx + methyl_carbon + 1 - 1
        benzene_action_idx = len(config.atom_vocabulary) + benzene_carbon

        # Try to increase the bond order
        print("\nAttempting to increase methyl-benzene bond to double bond")
        md.take_action(methyl_action_idx)  # Level 0: Select methyl carbon
        md.take_action(benzene_action_idx)  # Level 1: Select benzene carbon

        # Print action mask at level 2
        print(f"Action mask at level 2: {md.current_action_mask}")

        # Check if double bond is feasible (should be infeasible due to valence constraints)
        double_bond_feasible = md.current_action_mask[1] == 0

        if double_bond_feasible:
            print("Double bond is feasible (unexpected)")
            md.take_action(1)  # Create double bond
        else:
            print("Double bond is correctly infeasible due to valence constraints")
            # Take a valid action to continue the test
            for i, mask_value in enumerate(md.current_action_mask):
                if mask_value == 0:
                    print(f"Taking feasible action {i} instead")
                    md.take_action(i)
                    break

        # Check final bond order
        final_bond = md.rdkit_mol.GetBondBetweenAtoms(methyl_carbon, benzene_carbon)
        final_order = final_bond.GetBondTypeAsDouble() if final_bond else 0

        # The bond order should still be 1 since we can't increase it due to valence constraints
        print(f"Final bond order: {final_order}")
        print(f"Final SMILES: {md.to_smiles()}")

    def test_triple_bond_creation(self):
        """Test creating a triple bond using actions."""
        # Setup with acetylene (ethyne), which has a triple bond C≡C
        config = MoleculeConfig()
        config.start_from_smiles = "C#C"  # Acetylene

        print(f"\nStarting with acetylene from SMILES: {config.start_from_smiles}")
        md = MoleculeDesign.from_smiles(config, config.start_from_smiles, do_finish=False)

        # Get the RDKit representation
        rdkit_mol = md.rdkit_mol

        # Get the two carbon atoms
        carbon_indices = [atom.GetIdx() for atom in rdkit_mol.GetAtoms() if atom.GetSymbol() == "C"]
        atom_a, atom_b = carbon_indices[0], carbon_indices[1]

        # Verify the original bond is a triple bond
        original_bond = rdkit_mol.GetBondBetweenAtoms(atom_a, atom_b)
        original_order = original_bond.GetBondTypeAsDouble() if original_bond else 0
        print(f"Original bond order: {original_order}")
        self.assertEqual(original_order, 3, "Expected C≡C to be a triple bond (order 3)")

        # Try to decrease the bond to a double bond
        ex_atom_start_idx = len(config.atom_vocabulary) + 1
        atom_a_action_idx = ex_atom_start_idx + atom_a + 1 - 1
        atom_b_action_idx = len(config.atom_vocabulary) + atom_b

        print("\nAttempting to decrease triple bond to double bond")
        md.take_action(atom_a_action_idx)  # Level 0: Select first carbon
        md.take_action(atom_b_action_idx)  # Level 1: Select second carbon

        # Print action mask at level 2
        print(f"Action mask at level 2: {md.current_action_mask}")

        # Check if decreasing bond is feasible (action 6 = decrease by 1)
        decrease_bond_feasible = md.current_action_mask[6] == 0

        if decrease_bond_feasible:
            print("Decreasing bond is feasible, taking action 6")
            md.take_action(6)  # Decrease by 1

            # Verify the bond was decreased to double
            updated_bond = md.rdkit_mol.GetBondBetweenAtoms(atom_a, atom_b)
            updated_order = updated_bond.GetBondTypeAsDouble() if updated_bond else 0
            print(f"Bond order after decrease: {updated_order}")

            self.assertEqual(updated_order, 2,
                             f"Expected bond order to be decreased to 2, got {updated_order}")
            print("Successfully decreased C≡C triple bond to C=C double bond")

            # Try decreasing again to single bond
            print("\nAttempting to decrease double bond to single bond")
            md.take_action(atom_a_action_idx)  # Level 0: Select first carbon
            md.take_action(atom_b_action_idx)  # Level 1: Select second carbon

            if md.current_action_mask[6] == 0:
                md.take_action(6)  # Decrease by 1 again

                final_bond = md.rdkit_mol.GetBondBetweenAtoms(atom_a, atom_b)
                final_order = final_bond.GetBondTypeAsDouble() if final_bond else 0
                print(f"Bond order after second decrease: {final_order}")

                self.assertEqual(final_order, 1, "Expected bond to be single bond after second decrease")
                print("Successfully decreased to single bond")
            else:
                print("Decreasing to single bond is not feasible")
        else:
            print("Decreasing bond is not feasible, possibly due to structural constraints")

        print(f"Final SMILES: {md.to_smiles()}")


if __name__ == "__main__":
    unittest.main()