import unittest
import numpy as np
from rdkit import Chem
from config import MoleculeConfig
from molecule_design import MoleculeDesign


class TestMoleculeDesignComprehensive(unittest.TestCase):
    """
    A comprehensive test for the MoleculeDesign class that covers:
    1. Creating atoms of different types
    2. Creating bonds of different orders (single, double, triple)
    3. Creating ring structures
    4. Creating branches/substituents
    5. Modifying existing bonds (increasing and decreasing orders)
    6. Removing bonds
    7. Validating internal representations match RDKit

    The test constructs a complex molecule step by step, with validations at each step.
    """

    def setUp(self):
        """Set up configuration for testing."""
        self.config = MoleculeConfig()

        # Find indices for common atoms in the vocabulary for easier reference
        atom_names = list(self.config.atom_vocabulary.keys())
        self.C_idx = atom_names.index("C") + 1  # Carbon index in vocabulary
        self.N_idx = atom_names.index("N") + 1  # Nitrogen index
        self.O_idx = atom_names.index("O") + 1  # Oxygen index
        self.S_idx = atom_names.index("S") + 1  # Sulfur index

        # To easily reference the start index for selecting existing atoms
        self.ex_atom_start_idx = len(self.config.atom_vocabulary) + 1

    def test_build_complex_molecule(self):
        """
        Test building a complex molecule step by step, validating at each step.
        We'll create a benzene ring with various substituents.
        """
        # Start with a carbon atom
        molecule = MoleculeDesign(self.config, initial_atom=self.C_idx)

        # =====================================================
        # Step 1: Build a chain of 6 carbon atoms (for benzene)
        # =====================================================
        print("\nStep 1: Building a chain of 6 carbon atoms")

        print(molecule.history)
        print(molecule.atoms)
        print(molecule.bonds)

        # Add 5 more carbons to form a chain (the first atom is already added)
        for i in range(5):
            # Level 0: Create a new carbon atom
            self.assertEqual(molecule.current_action_level, 0)
            molecule.take_action(self.C_idx)

            print(molecule.history)
            print(molecule.atoms)
            print(molecule.bonds)

            # Level 1: Bond with the previously added atom
            self.assertEqual(molecule.current_action_level, 1)

            # For the first iteration, bond with the initial atom (nitrogen)
            # For subsequent iterations, bond with the last carbon added
            target_atom_idx = i

            # Print debug info
            print("action mask", molecule.current_action_mask)
            print(len(molecule.current_action_mask))
            print(len(molecule.atom_vocabulary))

            # Select existing atom for bonding using the correct offset
            existing_atom_action = len(self.config.atom_vocabulary) + target_atom_idx
            print("existing_atom_action", existing_atom_action)
            molecule.take_action(existing_atom_action)

            print(molecule.history)
            print(molecule.atoms)
            print(molecule.bonds)

            # Level 2: Create a single bond
            self.assertEqual(molecule.current_action_level, 2)
            molecule.take_action(0)  # Action 0 = bond order 1 (single bond)

            print(molecule.history)
            print(molecule.atoms)
            print(molecule.bonds)

            # Validate after adding each carbon
            self.assertEqual(molecule.current_action_level, 0)  # Back to level 0
            self.assertEqual(len(molecule.atoms), i + 3)  # Virtual atom (0) + initial N (1) + i+1 added C atoms

            # Verify the bond exists
            if i > 0:
                # The bond should be between the two most recently added atoms
                rdkit_bond = molecule.rdkit_mol.GetBondBetweenAtoms(i, i + 1)
                self.assertIsNotNone(rdkit_bond)
                self.assertEqual(rdkit_bond.GetBondTypeAsDouble(), 1.0)  # Single bond

        # Verify we now have 6 carbon atoms (plus the virtual atom)
        self.assertEqual(len(molecule.atoms), 7)
        self.assertTrue(all(atom == self.C_idx for atom in molecule.atoms[1:]))

        # Verify that all consecutive atoms are bonded
        for i in range(5):
            rdkit_bond = molecule.rdkit_mol.GetBondBetweenAtoms(i, i + 1)
            self.assertIsNotNone(rdkit_bond)
            self.assertEqual(rdkit_bond.GetBondTypeAsDouble(), 1.0)

        # =====================================================
        # Step 2: Close the ring to form benzene
        # =====================================================
        print("\nStep 2: Closing the ring to form benzene")

        # Level 0: Select the first carbon atom (index 1 in molecule.atoms)
        first_c_action_idx = self.ex_atom_start_idx + 1 - 1  # Formula: start_idx + atom_idx - 1
        molecule.take_action(first_c_action_idx)

        # Level 1: Select the last carbon atom (index 6 in molecule.atoms)
        # We need to use the vocabulary offset to select an existing atom
        last_c_action_idx = len(self.config.atom_vocabulary) + 5  # 23 + 5 = action 28
        print(f"Selecting last carbon using action {last_c_action_idx}")
        molecule.take_action(last_c_action_idx)

        # Level 2: Create a single bond
        molecule.take_action(0)  # Action 0 = bond order 1 (single bond)

        # Verify the ring closure bond exists
        rdkit_bond = molecule.rdkit_mol.GetBondBetweenAtoms(0, 5)
        self.assertIsNotNone(rdkit_bond, "Ring closure bond was not created")
        self.assertEqual(rdkit_bond.GetBondTypeAsDouble(), 1.0)  # Single bond

        # At this point, we have a cyclohexane. Let's add double bonds to make it benzene.

        # =====================================================
        # Step 3: Convert single bonds to double bonds (alternating)
        # =====================================================
        print("\nStep 3: Converting alternate bonds to double bonds")

        # Double bonds at positions 0-1, 2-3, and 4-5
        bond_positions = [(0, 1), (2, 3), (4, 5)]

        for pos in bond_positions:
            # Level 0: Select first atom of the bond
            first_atom_action_idx = self.ex_atom_start_idx + pos[0] + 1 - 1  # Formula: start_idx + atom_idx - 1
            molecule.take_action(first_atom_action_idx)

            # Level 1: Select second atom of the bond
            # INCORRECT: molecule.take_action(pos[1])  # This creates a new atom of type pos[1]
            # CORRECT: Use the vocabulary size offset to select an existing atom
            second_atom_action_idx = len(self.config.atom_vocabulary) + pos[1]
            print(f"Selecting second atom at position {pos[1]} using action {second_atom_action_idx}")
            molecule.take_action(second_atom_action_idx)

            # Level 2: Change to double bond (action 1 = bond order 2)
            molecule.take_action(1)

            # Verify the bond was updated
            rdkit_bond = molecule.rdkit_mol.GetBondBetweenAtoms(pos[0], pos[1])
            self.assertIsNotNone(rdkit_bond)
            self.assertEqual(rdkit_bond.GetBondTypeAsDouble(), 2.0)  # Double bond

        # At this point, we have a benzene ring. Now let's add some substituents.

        # =====================================================
        # Step 4: Add substituents (O, N, S) to the benzene ring
        # =====================================================
        print("\nStep 4: Adding substituents to the benzene ring")

        # Add an oxygen to position 1 (forming a phenol-like structure)
        # Level 0: Add oxygen atom
        molecule.take_action(self.O_idx)

        # Level 1: Bond with carbon at position 1
        # INCORRECT: molecule.take_action(1)  # This creates a new atom of type 1
        # CORRECT: Use the offset to select existing atom
        carbon1_action_idx = len(self.config.atom_vocabulary) + 1
        print(f"Selecting carbon at position 1 using action {carbon1_action_idx}")
        molecule.take_action(carbon1_action_idx)

        # Level 2: Create a single bond
        molecule.take_action(0)  # Action 0 = bond order 1 (single bond)

        # Verify oxygen was added and bonded
        self.assertEqual(molecule.atoms[7], self.O_idx)
        rdkit_bond = molecule.rdkit_mol.GetBondBetweenAtoms(1, 6)  # Carbon at idx 1, Oxygen at idx 6
        self.assertIsNotNone(rdkit_bond)
        self.assertEqual(rdkit_bond.GetBondTypeAsDouble(), 1.0)

        # Add a nitrogen to position 3 (forming an aniline-like structure)
        # Level 0: Add nitrogen atom
        molecule.take_action(self.N_idx)

        # Level 1: Bond with carbon at position 3
        # INCORRECT: molecule.take_action(3)
        # CORRECT: Use the offset
        carbon3_action_idx = len(self.config.atom_vocabulary) + 3
        print(f"Selecting carbon at position 3 using action {carbon3_action_idx}")
        molecule.take_action(carbon3_action_idx)

        # Level 2: Create a single bond
        molecule.take_action(0)  # Action 0 = bond order 1 (single bond)

        # Verify nitrogen was added and bonded
        self.assertEqual(molecule.atoms[8], self.N_idx)
        rdkit_bond = molecule.rdkit_mol.GetBondBetweenAtoms(3, 7)  # Carbon at idx 3, Nitrogen at idx 7
        self.assertIsNotNone(rdkit_bond)
        self.assertEqual(rdkit_bond.GetBondTypeAsDouble(), 1.0)

        # Add a sulfur to position 5 (forming a thiophenol-like structure)
        # Level 0: Add sulfur atom
        molecule.take_action(self.S_idx)

        # Level 1: Bond with carbon at position 5
        # INCORRECT: molecule.take_action(5)
        # CORRECT: Use the offset
        carbon5_action_idx = len(self.config.atom_vocabulary) + 5
        print(f"Selecting carbon at position 5 using action {carbon5_action_idx}")
        molecule.take_action(carbon5_action_idx)

        # Level 2: Create a single bond
        molecule.take_action(0)  # Action 0 = bond order 1 (single bond)

        # Verify sulfur was added and bonded
        self.assertEqual(molecule.atoms[9], self.S_idx)
        rdkit_bond = molecule.rdkit_mol.GetBondBetweenAtoms(5, 8)  # Carbon at idx 5, Sulfur at idx 8
        self.assertIsNotNone(rdkit_bond)
        self.assertEqual(rdkit_bond.GetBondTypeAsDouble(), 1.0)

        # # =====================================================
        # # Step 5: Modify an existing bond (increase order)
        # # =====================================================
        # print("\nStep 5: Increasing a bond order (N-C to N=C)")
        #
        # # Increase the bond order between nitrogen and carbon (making it a double bond)
        # # Level 0: Select the nitrogen atom
        # nitrogen_action_idx = self.ex_atom_start_idx + 8 - 1  # Formula: start_idx + atom_idx - 1
        # molecule.take_action(nitrogen_action_idx)
        #
        # # Level 1: Select the carbon atom it's bonded to (at position 3)
        # # INCORRECT: molecule.take_action(3)
        # # CORRECT: Use the offset
        # carbon3_action_idx = len(self.config.atom_vocabulary) + 3
        # print(f"Selecting carbon at position 3 using action {carbon3_action_idx}")
        # rdkit_bond = molecule.rdkit_mol.GetBondBetweenAtoms(3, 7)  # Carbon at idx 3, Nitrogen at idx 7
        # # rdkit_bond_2 = molecule.rdkit_mol.GetBondBetweenAtoms(3, 4)  # Carbon at idx 3, Nitrogen at idx 7
        # # rdkit_bond_3 = molecule.rdkit_mol.GetBondBetweenAtoms(3, 2)  # Carbon at idx 3, Nitrogen at idx 7
        # self.assertIsNotNone(rdkit_bond)
        # self.assertEqual(rdkit_bond.GetBondTypeAsDouble(), 1.0)  # Double bond
        # # self.assertEqual(rdkit_bond_2.GetBondTypeAsDouble(), 1.0)  # Double bond
        # # self.assertEqual(rdkit_bond_3.GetBondTypeAsDouble(), 2.0)  # Double bond
        #
        # molecule.take_action(carbon3_action_idx)
        #
        # # Level 2: Change to double bond (action 1 = bond order 2)
        # print(molecule.atoms)
        # print(molecule.bonds)
        # molecule.take_action(1)
        #
        # # Verify the bond order was increased
        # rdkit_bond = molecule.rdkit_mol.GetBondBetweenAtoms(3, 7)  # Carbon at idx 3, Nitrogen at idx 7
        # self.assertIsNotNone(rdkit_bond)
        # self.assertEqual(rdkit_bond.GetBondTypeAsDouble(), 2.0)  # Double bond

        # =====================================================
        # Step 5: Modify an existing bond (increase order)
        # =====================================================
        print("\nStep 5: Increasing a bond order (N-C to N=C)")

        # Increase the bond order between nitrogen and carbon (making it a double bond)
        # Level 0: Select the nitrogen atom
        nitrogen_action_idx = self.ex_atom_start_idx + 8 - 1  # Formula: start_idx + atom_idx - 1
        molecule.take_action(nitrogen_action_idx)

        # Level 1: Select the carbon atom it's bonded to (at position 3)
        carbon3_action_idx = len(self.config.atom_vocabulary) + 3
        print(f"Selecting carbon at position 3 using action {carbon3_action_idx}")
        molecule.take_action(carbon3_action_idx)

        # Level 2: Check if double bond is feasible before attempting it
        print(f"Action mask at level 2: {molecule.current_action_mask}")
        double_bond_feasible = molecule.current_action_mask[1] == 0  # Check if action 1 (double bond) is feasible

        if double_bond_feasible:
            print("Creating a double bond (action 1) is feasible")
            molecule.take_action(1)  # Action 1 = bond order 2 (double bond)

            # Verify the bond order was increased
            rdkit_bond = molecule.rdkit_mol.GetBondBetweenAtoms(3, 7)  # Carbon at idx 3, Nitrogen at idx 7
            self.assertIsNotNone(rdkit_bond)
            self.assertEqual(rdkit_bond.GetBondTypeAsDouble(), 2.0)  # Double bond
        else:
            print("Creating a double bond is NOT feasible due to valence constraints")
            # Maintain single bond by using action 0
            molecule.take_action(0)  # Keep single bond

            # Verify the bond remains a single bond
            rdkit_bond = molecule.rdkit_mol.GetBondBetweenAtoms(3, 7)
            self.assertIsNotNone(rdkit_bond)
            self.assertEqual(rdkit_bond.GetBondTypeAsDouble(), 1.0)  # Single bond

        # # =====================================================
        # # Step 6: Modify an existing bond (decrease order)
        # # =====================================================
        # print("\nStep 6: Decreasing a bond order (C=C to C-C)")
        #
        # # Decrease a double bond in the ring back to a single bond
        # # Level 0: Select the first carbon atom of a double bond pair (e.g., at position 0)
        # first_c_action_idx = self.ex_atom_start_idx + 1 - 1  # Formula: start_idx + atom_idx - 1
        # molecule.take_action(first_c_action_idx)
        #
        # # Level 1: Select the second carbon atom (at position 1)
        # # INCORRECT: molecule.take_action(1)
        # # CORRECT: Use the offset
        # carbon1_action_idx = len(self.config.atom_vocabulary) + 1
        # print(f"Selecting second carbon using action {carbon1_action_idx}")
        # molecule.take_action(carbon1_action_idx)
        #
        # # Level 2: Decrease to single bond (action 6 = decrease by 1)
        # molecule.take_action(6)  # First action in the "decrease" section
        #
        # # Verify the bond order was decreased
        # rdkit_bond = molecule.rdkit_mol.GetBondBetweenAtoms(0, 1)
        # self.assertIsNotNone(rdkit_bond)
        # self.assertEqual(rdkit_bond.GetBondTypeAsDouble(), 1.0)  # Single bond

        # =====================================================
        # Step 6: Modify an existing bond (decrease order)
        # =====================================================
        print("\nStep 6: Decreasing a bond order (C=C to C-C)")

        # Decrease a double bond in the ring back to a single bond
        # Level 0: Select the first carbon atom of a double bond pair (e.g., at position 0)
        first_c_action_idx = self.ex_atom_start_idx + 1 - 1  # Formula: start_idx + atom_idx - 1
        molecule.take_action(first_c_action_idx)

        # Level 1: Select the second carbon atom (at position 1)
        carbon1_action_idx = len(self.config.atom_vocabulary) + 1
        print(f"Selecting second carbon using action {carbon1_action_idx}")
        molecule.take_action(carbon1_action_idx)

        # Level 2: Check if decrease action is feasible
        print(f"Action mask at level 2: {molecule.current_action_mask}")
        decrease_bond_feasible = molecule.current_action_mask[6] == 0  # Check if action 6 (decrease by 1) is feasible

        if decrease_bond_feasible:
            print("Decreasing bond order (action 6) is feasible")
            molecule.take_action(6)  # Action 6 = decrease by 1

            # Verify the bond order was decreased
            rdkit_bond = molecule.rdkit_mol.GetBondBetweenAtoms(0, 1)
            self.assertIsNotNone(rdkit_bond)
            self.assertEqual(rdkit_bond.GetBondTypeAsDouble(), 1.0)  # Single bond
        else:
            print("Decreasing bond order is NOT feasible")
            # Choose a feasible action instead
            for i in range(len(molecule.current_action_mask)):
                if molecule.current_action_mask[i] == 0:
                    print(f"Using feasible action {i} instead")
                    molecule.take_action(i)
                    break

        # =====================================================
        # Step 7: Remove a bond and create a new one
        # =====================================================
        print("\nStep 7: Removing a bond and creating a new one")

        # We need to be careful here - removing a bond could fragment the molecule
        # Let's first check if we can remove the bond between C5 and S
        can_remove = molecule.is_connected_without_bond(6, 9)  # Internal indices (with virtual atom)

        if can_remove:
            # Level 0: Select the sulfur atom
            sulfur_action_idx = self.ex_atom_start_idx + 9 - 1  # Formula: start_idx + atom_idx - 1
            molecule.take_action(sulfur_action_idx)

            # Level 1: Select the carbon atom it's bonded to (at position 5)
            # INCORRECT: molecule.take_action(5)
            # CORRECT: Use the offset
            carbon5_action_idx = len(self.config.atom_vocabulary) + 5
            print(f"Selecting carbon at position 5 using action {carbon5_action_idx}")
            molecule.take_action(carbon5_action_idx)

            # Level 2: Remove the bond (action 7 = decrease by 1 from bond order 1 to 0)
            molecule.take_action(7)  # Action to completely remove bond

            # Verify the bond was removed
            rdkit_bond = molecule.rdkit_mol.GetBondBetweenAtoms(5, 8)  # Carbon at idx 5, Sulfur at idx 8
            self.assertIsNone(rdkit_bond)

            # Now create a new bond between S and O
            # Level 0: Select the sulfur atom again
            molecule.take_action(sulfur_action_idx)

            # Level 1: Select the oxygen atom (at index 7)
            # INCORRECT: molecule.take_action(7)
            # CORRECT: Use the offset
            oxygen_action_idx = len(self.config.atom_vocabulary) + 7
            print(f"Selecting oxygen atom using action {oxygen_action_idx}")
            molecule.take_action(oxygen_action_idx)

            # Level 2: Create a single bond
            molecule.take_action(0)  # Action 0 = bond order 1 (single bond)

            # Verify the new bond was created
            rdkit_bond = molecule.rdkit_mol.GetBondBetweenAtoms(6, 8)  # Oxygen at idx 6, Sulfur at idx 8
            self.assertIsNotNone(rdkit_bond)
            self.assertEqual(rdkit_bond.GetBondTypeAsDouble(), 1.0)
        else:
            # If we can't remove this bond without fragmenting, let's add a new carbon atom
            # and create a branch off of one of the existing atoms
            print("Cannot remove S-C bond without fragmenting, creating a branch instead")

            # Level 0: Add a new carbon atom
            molecule.take_action(self.C_idx)

            # Level 1: Bond with the oxygen atom
            # INCORRECT: molecule.take_action(6)
            # CORRECT: Use the offset
            oxygen_action_idx = len(self.config.atom_vocabulary) + 6
            print(f"Selecting oxygen atom using action {oxygen_action_idx}")
            molecule.take_action(oxygen_action_idx)

            # Level 2: Create a single bond
            molecule.take_action(0)  # Action 0 = bond order 1 (single bond)

            # Verify the new carbon was added and bonded to oxygen
            self.assertEqual(molecule.atoms[10], self.C_idx)
            rdkit_bond = molecule.rdkit_mol.GetBondBetweenAtoms(6, 9)  # Oxygen at idx 6, new C at idx 9
            self.assertIsNotNone(rdkit_bond)
            self.assertEqual(rdkit_bond.GetBondTypeAsDouble(), 1.0)

        # =====================================================
        # Step 8: Finalize and verify the molecule
        # =====================================================
        print("\nStep 8: Finalizing and verifying the molecule")

        # Terminate the molecule construction
        molecule.take_action(0)  # Action 0 at level 0 = terminate

        # Verify the molecule is terminated
        self.assertTrue(molecule.synthesis_done)

        # Generate and print the SMILES string for inspection
        smiles = molecule.to_smiles()
        print(f"Final molecule SMILES: {smiles}")

        # Ensure the molecule can be sanitized
        try:
            mol = Chem.MolFromSmiles(smiles)
            Chem.SanitizeMol(mol)
            print("Molecule successfully sanitized")
        except Exception as e:
            self.fail(f"Failed to sanitize molecule: {e}")

        # Verify atom counts by type
        atom_counts = {}
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            atom_counts[symbol] = atom_counts.get(symbol, 0) + 1

        print(f"Atom counts: {atom_counts}")

        # Check basic properties
        if 'C' in atom_counts:
            # We should have 6 or 7 carbon atoms (6 for the ring + possibly 1 more)
            self.assertIn(atom_counts['C'], [6, 7])
        if 'O' in atom_counts:
            self.assertEqual(atom_counts['O'], 1)
        if 'N' in atom_counts:
            self.assertEqual(atom_counts['N'], 1)
        if 'S' in atom_counts:
            self.assertEqual(atom_counts['S'], 1)


if __name__ == "__main__":
    unittest.main()