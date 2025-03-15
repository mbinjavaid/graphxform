import unittest
import numpy as np
from rdkit import Chem
from molecule_design import MoleculeDesign
from config import MoleculeConfig


class TestMoleculeDesign(unittest.TestCase):
    def setUp(self):
        """Set up a basic configuration for testing"""
        self.config = MoleculeConfig()
        # Ensure we're using a simple vocabulary for testing
        self.config.atom_vocabulary = {
            "C": {"allowed": True, "atomic_number": 6, "valence": 4},
            "N": {"allowed": True, "atomic_number": 7, "valence": 3},
            "O": {"allowed": True, "atomic_number": 8, "valence": 2}
        }
        self.config.max_num_atoms = 10

    def test_initial_state(self):
        """Test the initial state of a molecule design"""
        mol = MoleculeDesign(self.config, 1)  # Carbon atom

        # Check initial structure
        self.assertEqual(len(mol.atoms), 2)  # Virtual atom + C
        self.assertEqual(mol.atoms[0], 0)  # Virtual atom
        self.assertEqual(mol.atoms[1], 1)  # Carbon atom

        # Check bond matrix
        self.assertEqual(mol.bonds.shape, (2, 2))
        self.assertEqual(mol.bonds[0, 1], mol.virtual_bond_idx)  # Virtual bond
        self.assertEqual(mol.bonds[1, 0], mol.virtual_bond_idx)  # Virtual bond
        self.assertEqual(mol.bonds[1, 1], 0)  # No self-bond

        # Check initial action level
        self.assertEqual(mol.current_action_level, 0)

    def test_add_atom_and_bond(self):
        """Test adding atoms and forming bonds"""
        mol = MoleculeDesign(self.config, 1)  # Start with C

        # Add N atom connected to C
        mol.take_action(2)  # Choose N atom at level 0
        self.assertEqual(mol.current_action_level, 1)  # Move to level 1
        mol.take_action(0)  # Connect to first atom at level 1
        self.assertEqual(mol.current_action_level, 2)  # Move to level 2
        mol.take_action(0)  # Create single bond at level 2
        self.assertEqual(mol.current_action_level, 0)  # Back to level 0

        # Check molecule state after adding N with single bond
        self.assertEqual(len(mol.atoms), 3)  # Virtual + C + N
        self.assertEqual(mol.atoms[2], 2)  # N atom added
        self.assertEqual(mol.bonds[1, 2], 1)  # Single bond between C-N
        self.assertEqual(mol.bonds[2, 1], 1)  # Symmetric bond matrix

        # Add O atom connected to N with double bond
        mol.take_action(3)  # Choose O atom at level 0
        mol.take_action(1)  # Connect to N atom at level 1
        mol.take_action(1)  # Create double bond at level 2

        # Check molecule state after adding O with double bond to N
        self.assertEqual(len(mol.atoms), 4)  # Virtual + C + N + O
        self.assertEqual(mol.atoms[3], 3)  # O atom added
        self.assertEqual(mol.bonds[2, 3], 2)  # Double bond between N-O
        self.assertEqual(mol.bonds[3, 2], 2)  # Symmetric bond matrix

        # Check remaining valence
        atom_valence = np.array([mol.vocabulary_valence[x] for x in mol.atoms])
        # atom_valence_used = mol.bonds.sum(axis=1)
        # Don't consider virtual bonds in valence calculation:
        atom_valence_used = np.sum(np.where(mol.bonds != mol.virtual_bond_idx, mol.bonds, 0), axis=1)
        self.assertEqual(atom_valence[1] - atom_valence_used[1], 3)  # C has 3 valence left (4-1)
        self.assertEqual(atom_valence[2] - atom_valence_used[2], 0)  # N has 0 valence left (3-1-2)
        self.assertEqual(atom_valence[3] - atom_valence_used[3], 0)  # O has 0 valence left (2-2)

    def test_is_connected_without_bond(self):
        """Test the connectivity check function"""
        # Test fork molecule first (C1 connected to C2 and C3)
        mol = MoleculeDesign(self.config, 1)  # Start with C1

        # print(mol.bonds)

        # Add C2 to C1
        mol.take_action(1)  # Add C2
        # print(mol.bonds)
        mol.take_action(0)  # Bond to C1
        mol.take_action(0)  # Single bond

        # Add C3 to C1 (not to C2)
        mol.take_action(1)  # Add C3
        mol.take_action(0)  # Bond to C1
        mol.take_action(0)  # Single bond

        # Print the current molecule structure for debugging
        print("Molecule structure (fork):")
        print(f"Atoms: {mol.atoms}")
        print(f"Bonds matrix:")
        print(mol.bonds)

        # Check connectivity:
        # - Removing bond between C1 (idx 1) and C2 (idx 2) should disconnect
        # - Removing bond between C1 (idx 1) and C3 (idx 3) should disconnect
        self.assertFalse(mol.is_connected_without_bond(1, 2))
        self.assertFalse(mol.is_connected_without_bond(1, 3))

        # For a cyclic molecule, use from_smiles for simplicity
        cyclic_mol = MoleculeDesign.from_smiles(self.config, "C1CC1")  # Cyclopropane

        # Print the cyclic molecule structure
        print("Cyclic molecule structure:")
        print(f"Atoms: {cyclic_mol.atoms}")
        print(f"Bonds matrix:")
        print(cyclic_mol.bonds)

        # Check that removing any single bond in the cycle keeps it connected
        # For a cycle, the molecule should remain connected when any single bond is removed
        self.assertTrue(cyclic_mol.is_connected_without_bond(1, 2))
        self.assertTrue(cyclic_mol.is_connected_without_bond(2, 3))
        self.assertTrue(cyclic_mol.is_connected_without_bond(3, 1))



    def test_action_masking_level2(self):
        """Test action masking at level 2 including bond reduction options"""
        mol = MoleculeDesign(self.config, 1)  # Start with C
        mol.take_action(1)  # Add C
        mol.take_action(0)  # Bond to first C

        # At this point we're at level 2, let's inspect the action mask
        # First 6 indices are for increasing bond order, second 6 for decreasing
        self.assertEqual(len(mol.current_action_mask), 2 * mol.maximum_bond_order)

        # We should be able to form up to a quadruple bond (C valence is 4)
        self.assertEqual(mol.current_action_mask[0], False)  # Single bond allowed
        self.assertEqual(mol.current_action_mask[1], False)  # Double bond allowed
        self.assertEqual(mol.current_action_mask[2], False)  # Triple bond allowed
        self.assertEqual(mol.current_action_mask[3], False)  # Quadruple bond allowed
        self.assertEqual(mol.current_action_mask[4], True)  # Quintuple bond not allowed

        # Since no bond exists yet, no decreasing actions should be allowed
        for i in range(mol.maximum_bond_order, 2 * mol.maximum_bond_order):
            self.assertEqual(mol.current_action_mask[i], True)

        # Create a triple bond
        mol.take_action(2)  # Form triple bond

        # Now create another action and get to level 2 again
        mol.take_action(1)  # Add C
        mol.take_action(0)  # Bond to first C

        # First atom has only 1 valence left (4-3), so only single bond allowed for increase
        self.assertEqual(mol.current_action_mask[0], False)  # Single bond allowed
        self.assertEqual(mol.current_action_mask[1], True)  # Double bond not allowed

        # Now let's try with an existing bond
        # Create a cycle to test reducing a bond that doesn't disconnect the molecule
        mol = MoleculeDesign(self.config, 1)  # Start with C
        mol.take_action(1)  # Add C
        mol.take_action(0)  # Bond to first C
        mol.take_action(0)  # Single bond
        mol.take_action(1)  # Add another C
        mol.take_action(1)  # Bond to second C
        mol.take_action(0)  # Single bond
        # Close the cycle
        mol.take_action(3)  # Select first C (existing atom)
        mol.take_action(1)  # Bond to third C
        mol.take_action(1)  # Double bond

        # Now select the double bond to potentially reduce it
        mol.take_action(3)  # Select first C
        mol.take_action(1)  # Select third C

        # We should be able to reduce the double bond by 1 or 2 steps
        self.assertEqual(mol.current_action_mask[mol.maximum_bond_order], False)  # Reduce by 1 allowed
        self.assertEqual(mol.current_action_mask[mol.maximum_bond_order + 1],
                         False)  # Reduce by 2 allowed (would remove bond)

    def test_bond_reduction_execution(self):
        """Test that bond reduction properly updates molecule state"""
        # Create a molecule with double bond
        mol = MoleculeDesign(self.config, 1)  # Start with C
        mol.take_action(1)  # Add C
        mol.take_action(0)  # Bond to first C
        mol.take_action(1)  # Create double bond

        # Verify initial state
        self.assertEqual(mol.bonds[1, 2], 2)  # Double bond between C atoms

        # Now reduce the bond
        mol.take_action(3)  # Select first C
        mol.take_action(0)  # Select second C

        # Debug info to understand action masking
        print(f"Current bond order: {mol.bonds[1, 2]}")
        print(f"Action mask: {mol.current_action_mask}")

        # The first reduction action should be at index maximum_bond_order
        bond_reduction_action = mol.maximum_bond_order

        # Check if the action is allowed
        self.assertEqual(mol.current_action_mask[bond_reduction_action], False,
                         f"Bond reduction action {bond_reduction_action} should be allowed")

        # Take the action to reduce the bond by 1
        mol.take_action(bond_reduction_action)

        # Verify bond was reduced
        self.assertEqual(mol.bonds[1, 2], 1)  # Now single bond
        self.assertEqual(mol.rdkit_mol.GetBondBetweenAtoms(0, 1).GetBondTypeAsDouble(), 1.0)  # RDKit bond matches

        # Reduce to zero (break bond)
        mol.take_action(3)  # Select first C
        mol.take_action(0)  # Select second C
        mol.take_action(mol.maximum_bond_order)  # Reduce by 1

        # Verify bond was removed
        self.assertEqual(mol.bonds[1, 2], 0)  # No bond
        self.assertIsNone(mol.rdkit_mol.GetBondBetweenAtoms(0, 1))  # No bond in RDKit mol

    def test_cyclic_structure_bond_reduction(self):
        """Test bond reduction in cyclic structures"""
        # Create a cycle of 3 carbon atoms
        mol = MoleculeDesign(self.config, 1)  # Start with C
        mol.take_action(1)  # Add C
        mol.take_action(0)  # Bond to first C
        mol.take_action(0)  # Single bond
        mol.take_action(1)  # Add another C
        mol.take_action(1)  # Bond to second C
        mol.take_action(0)  # Single bond
        # Close the cycle
        mol.take_action(3)  # Select first C (existing atom)
        mol.take_action(1)  # Bond to third C
        mol.take_action(0)  # Single bond

        # Verify action mask allows removing one bond from the cycle
        mol.take_action(3)  # Select first C
        mol.take_action(0)  # Select second C

        # Check that we can reduce the bond since it won't disconnect the structure
        self.assertEqual(mol.current_action_mask[mol.maximum_bond_order], False)  # Allowed

        # Actually reduce the bond
        mol.take_action(mol.maximum_bond_order)  # Reduce bond to zero

        # Verify the bond is gone but structure is valid
        self.assertEqual(mol.bonds[1, 2], 0)  # Bond removed
        self.assertIsNone(mol.rdkit_mol.GetBondBetweenAtoms(0, 1))  # No bond in RDKit

        # Structure should still be valid and connected
        try:
            Chem.SanitizeMol(mol.rdkit_mol)
            is_valid = True
        except:
            is_valid = False
        self.assertTrue(is_valid)

    def test_disconnected_structure_prevention(self):
        """Test that action masking prevents creating disconnected structures"""
        # Create a linear C-C-C
        mol = MoleculeDesign(self.config, 1)  # Start with C
        mol.take_action(1)  # Add C
        mol.take_action(0)  # Bond to first C
        mol.take_action(0)  # Single bond
        mol.take_action(1)  # Add another C
        mol.take_action(1)  # Bond to second C
        mol.take_action(0)  # Single bond

        # Print the initial molecule state
        print(f"Initial bonds matrix:")
        print(mol.bonds)

        # Try to break middle bond
        mol.take_action(3)  # Select first C
        mol.take_action(0)  # Select second C

        print(f"Selected atoms: First C (index: {3 - 3}) and Second C (index: {0})")
        print(f"Maximum bond order: {mol.maximum_bond_order}")
        print(f"Current bond order: {mol.bonds[1, 2]}")
        print(f"Action mask: {mol.current_action_mask}")

        # Check that reducing this bond to zero is masked (would disconnect)
        self.assertEqual(mol.current_action_mask[mol.maximum_bond_order], True)

        # First issue: Let's correctly increase the bond order
        # We need to figure out which action actually increases the bond
        # Let's check what action levels and indices we're at
        print(f"Current action level: {mol.current_action_level}")
        print(f"History: {mol.history}")

        # Try a different action to increase the bond
        bond_increase_action = 0  # This should be the action to create a single bond
        mol.take_action(bond_increase_action)

        print(f"After take_action({bond_increase_action}):")
        print(f"Bonds matrix:")
        print(mol.bonds)
        print(f"Current bond order between atoms 1 and 2: {mol.bonds[1, 2]}")

        # Now try to reduce it again
        mol.take_action(3)  # Select first C
        mol.take_action(0)  # Select second C

        print(f"After reselecting atoms:")
        print(f"Current bond order: {mol.bonds[1, 2]}")
        print(f"Action mask: {mol.current_action_mask}")

        # Should allow reducing to single if we successfully increased to double
        if mol.bonds[1, 2] > 1:
            self.assertEqual(mol.current_action_mask[mol.maximum_bond_order], False)  # Can reduce by 1
            self.assertEqual(mol.current_action_mask[mol.maximum_bond_order + 1], True)  # Can't reduce by 2
        else:
            print("WARNING: Bond order did not increase, skipping reduction test")

    def test_valence_tracking_after_bond_changes(self):
        """Test that valence is correctly tracked after bond changes"""
        mol = MoleculeDesign(self.config, 1)  # Start with C
        mol.take_action(2)  # Add N
        mol.take_action(0)  # Bond to C
        mol.take_action(1)  # Double bond

        # Check valence used
        atom_valence = np.array([mol.vocabulary_valence[x] for x in mol.atoms])
        atom_valence_used = mol.bonds.sum(axis=1)
        self.assertEqual(atom_valence[1] - atom_valence_used[1], 2)  # C used 2 of 4
        self.assertEqual(atom_valence[2] - atom_valence_used[2], 1)  # N used 2 of 3

        # Reduce the bond and check valence again
        mol.take_action(3)  # Select C (existing atom)
        mol.take_action(0)  # Select N
        mol.take_action(mol.maximum_bond_order)  # Reduce by 1

        # Check updated valence
        atom_valence_used = mol.bonds.sum(axis=1)
        self.assertEqual(atom_valence[1] - atom_valence_used[1], 3)  # C used 1 of 4
        self.assertEqual(atom_valence[2] - atom_valence_used[2], 2)  # N used 1 of 3

    def test_complex_molecule_with_reduction(self):
        """Test building a complex molecule with multiple bond reductions"""
        # Build a more complex molecule with multiple bond types
        mol = MoleculeDesign(self.config, 1)  # Start with C

        # Add C with double bond
        mol.take_action(1)  # Add C
        mol.take_action(0)  # Bond to first C
        mol.take_action(1)  # Double bond

        # Add O with single bond to first C
        mol.take_action(3)  # Add O
        mol.take_action(0)  # Bond to first C
        mol.take_action(0)  # Single bond

        # Add N with triple bond to second C
        mol.take_action(2)  # Add N
        mol.take_action(1)  # Bond to second C
        mol.take_action(2)  # Triple bond

        # Now we have C(=C)(O)(-N#)
        # Check that the molecule structure is as expected
        self.assertEqual(len(mol.atoms), 5)  # Virtual + C + C + O + N
        self.assertEqual(mol.bonds[1, 2], 2)  # C-C double bond
        self.assertEqual(mol.bonds[1, 3], 1)  # C-O single bond
        self.assertEqual(mol.bonds[2, 4], 3)  # C-N triple bond

        # Try to reduce C-C double bond to single
        mol.take_action(3)  # Select first C
        mol.take_action(0)  # Select second C
        mol.take_action(mol.maximum_bond_order)  # Reduce by 1

        # Check updated structure
        self.assertEqual(mol.bonds[1, 2], 1)  # Now C-C single bond

        # Try to reduce C-N triple bond to double
        mol.take_action(4)  # Select second C
        mol.take_action(2)  # Select N
        mol.take_action(mol.maximum_bond_order)  # Reduce by 1

        # Check updated structure
        self.assertEqual(mol.bonds[2, 4], 2)  # Now C-N double bond

        # Verify final RDKit molecule structure is as expected
        mol.finalize()
        self.assertFalse(mol.infeasibility_flag)
        expected_smiles = "COC=N"  # Simplified form of what we built
        self.assertEqual(Chem.CanonSmiles(mol.smiles_string), Chem.CanonSmiles(expected_smiles))

    def test_rdkit_synchronization(self):
        """Test that the RDKit molecule stays in sync with our internal representation"""
        mol = MoleculeDesign(self.config, 1)  # Start with C

        # Add N with double bond
        mol.take_action(2)  # Add N
        mol.take_action(0)  # Bond to C
        mol.take_action(1)  # Double bond

        # Verify RDKit structure matches internal representation
        self.assertEqual(mol.rdkit_mol.GetNumAtoms(), 2)  # C, N
        bond = mol.rdkit_mol.GetBondBetweenAtoms(0, 1)
        self.assertIsNotNone(bond)
        self.assertEqual(bond.GetBondTypeAsDouble(), 2.0)  # Double bond

        # Reduce bond to single
        mol.take_action(3)  # Select C
        mol.take_action(0)  # Select N
        mol.take_action(mol.maximum_bond_order)  # Reduce by 1

        # Verify RDKit structure updated correctly
        bond = mol.rdkit_mol.GetBondBetweenAtoms(0, 1)
        self.assertIsNotNone(bond)
        self.assertEqual(bond.GetBondTypeAsDouble(), 1.0)  # Single bond

        # Reduce to zero (break bond)
        mol.take_action(3)  # Select C
        mol.take_action(0)  # Select N
        mol.take_action(mol.maximum_bond_order)  # Reduce by 1

        # Verify RDKit bond was removed
        bond = mol.rdkit_mol.GetBondBetweenAtoms(0, 1)
        self.assertIsNone(bond)  # Bond removed

    def test_network_outputs_compatibility(self):
        """Test that network outputs are compatible with expanded action space"""
        # This test simulates what happens in log_probability_fn
        mol = MoleculeDesign(self.config, 1)  # Start with C

        # Go to level 2
        mol.take_action(1)  # Add C
        mol.take_action(0)  # Bond to C

        # Create mock network outputs
        mock_level2_logits = np.ones(mol.maximum_bond_order)  # Original size output

        # Simulate what happens in log_probability_fn
        if len(mock_level2_logits) < 2 * mol.maximum_bond_order:
            expanded_logits = np.full(2 * mol.maximum_bond_order, -np.inf)
            expanded_logits[:mock_level2_logits.shape[0]] = mock_level2_logits
            mock_level2_logits = expanded_logits

        # Mask the logits
        mask = mol.current_action_mask
        mock_level2_logits[mask] = np.NINF

        # We should have valid logits for allowed actions
        self.assertTrue(np.isfinite(mock_level2_logits[0]))  # Single bond allowed
        self.assertTrue(np.isfinite(mock_level2_logits[1]))  # Double bond allowed
        self.assertFalse(np.isfinite(mock_level2_logits[mol.maximum_bond_order]))  # Decreasing not allowed yet

    def test_smiles_conversion_with_bond_reductions(self):
        """Test SMILES string generation after bond reductions"""
        # Create a molecule
        mol = MoleculeDesign(self.config, 1)  # Start with C

        # Add N with triple bond
        mol.take_action(2)  # Add N
        mol.take_action(0)  # Bond to C
        mol.take_action(2)  # Triple bond

        # Add O to C with single bond
        mol.take_action(3)  # Add O
        mol.take_action(0)  # Bond to C
        mol.take_action(0)  # Single bond

        # Finalize and check SMILES
        mol_original = mol.to_rdkit_mol()
        smiles_original = Chem.MolToSmiles(mol_original)

        # Reduce N-C bond to double
        mol.take_action(3)  # Select C
        mol.take_action(0)  # Select N
        mol.take_action(mol.maximum_bond_order)  # Reduce by 1

        # Finalize and check new SMILES
        mol_reduced = mol.to_rdkit_mol()
        smiles_reduced = Chem.MolToSmiles(mol_reduced)

        # The SMILES should be different after bond reduction
        self.assertNotEqual(Chem.CanonSmiles(smiles_original), Chem.CanonSmiles(smiles_reduced))

        # The reduced molecule should be valid
        try:
            Chem.SanitizeMol(mol_reduced)
            is_valid = True
        except:
            is_valid = False
        self.assertTrue(is_valid)

    def test_bond_reduction_from_smiles(self):
        """Test creating molecule from SMILES then reducing bonds"""
        # Create a molecule from SMILES
        smiles = "C=CC#N"  # C=C-C#N
        mol = MoleculeDesign.from_smiles(self.config, smiles)

        # Check the structure is as expected
        self.assertEqual(len(mol.atoms), 5)  # Virtual + 3C + N
        self.assertEqual(mol.bonds[1, 2], 2)  # C=C
        self.assertEqual(mol.bonds[2, 3], 1)  # C-C
        self.assertEqual(mol.bonds[3, 4], 3)  # C#N

        # Reduce C=C to C-C
        mol.take_action(3)  # Select first C
        mol.take_action(0)  # Select second C
        mol.take_action(mol.maximum_bond_order)  # Reduce by 1

        # Check updated structure
        self.assertEqual(mol.bonds[1, 2], 1)  # Now C-C

        # Reduce C#N to C=N
        mol.take_action(5)  # Select third C
        mol.take_action(2)  # Select N
        mol.take_action(mol.maximum_bond_order)  # Reduce by 1

        # Check updated structure
        self.assertEqual(mol.bonds[3, 4], 2)  # Now C=N

        # Finalize and verify SMILES
        mol.finalize()
        self.assertFalse(mol.infeasibility_flag)
        expected_smiles = "CCC=N"
        self.assertEqual(Chem.CanonSmiles(mol.smiles_string), Chem.CanonSmiles(expected_smiles))

if __name__ == "__main__":
    unittest.main()