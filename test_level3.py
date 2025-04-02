import unittest
import numpy as np
from rdkit import Chem

from config import MoleculeConfig
from molecule_design import MoleculeDesign


class DummyMoleculeConfig(MoleculeConfig):
    def __init__(self):
        # Dummy configuration with sufficient atom types for testing
        self.atom_vocabulary = {
            "C": {"valence": 4, "allowed": True, "atomic_number": 6},
            "N": {"valence": 3, "allowed": True, "atomic_number": 7},
            "O": {"valence": 2, "allowed": True, "atomic_number": 8},
            "F": {"valence": 1, "allowed": True, "atomic_number": 9},
        }
        self.max_num_atoms = 20  # Allow more atoms for complex test structures
        self.start_c_chain_max_len = 5


class TestFragmentHandling(unittest.TestCase):
    def setUp(self):
        self.config = DummyMoleculeConfig()
        # Start with a Carbon atom
        self.initial_atom = 1  # Carbon
        self.mol = MoleculeDesign(self.config, self.initial_atom)

        # Store atom indices for convenience
        self.atom_types = {
            "C": 1,
            "N": 2,
            "O": 3,
            "F": 4
        }

        # Store vocabulary size for action indexing
        self.vocab_size = len(self.config.atom_vocabulary)

    def build_bridge_molecule(self):
        """Create a simple linear chain: C-C-C where removing the middle bond creates two fragments"""
        print("DEBUG: Building bridge molecule")

        # Start with C
        mol = MoleculeDesign(self.config, initial_atom=1)

        # Add second C directly connected to first C
        # Level 0: Select atom 1 (first C)
        mol.take_action(1)
        # Level 1: Create new C atom (action 0 for atom type 1)
        mol.take_action(0)
        # Level 2: Set single bond (action V+0 for bond order 1)
        mol.take_action(self.vocab_size + 0)

        # Add third C directly connected to second C
        # Level 0: Select atom 2 (second C)
        mol.take_action(2)
        # Level 1: Create new C atom (action 0 for atom type 1)
        mol.take_action(0)
        # Level 2: Set single bond (action V+0 for bond order 1)
        mol.take_action(self.vocab_size + 0)

        print("Atoms: ", mol.atoms)
        print("Bonds:\n", mol.bonds)

        # Verify we have a single connected molecule
        frags = Chem.GetMolFrags(mol.rdkit_mol, asMols=False)
        print(f"DEBUG: Molecule structure: {Chem.MolToSmiles(mol.rdkit_mol)}")
        print(f"DEBUG: Initial fragment count: {len(frags)}")

        # The bridge bond is between atoms 2 and 3 (internal indices)
        # When we remove this bond, we should get exactly two fragments
        return mol, (2, 3)

    def build_ring_molecule(self):
        """
        Builds a molecule with a ring structure:

        C---C
        |   |
        C---C

        Where removing any bond will create a linear fragment
        """
        # Start with single C atom
        mol = MoleculeDesign(self.config, self.atom_types["C"])

        # Build a ring of 4 carbon atoms
        # 1. Select atom 1 (first C)
        mol.take_action(1)
        # Create second carbon
        mol.take_action(0)  # C atom type (atom type 1 → action 0)
        mol.take_action(self.vocab_size + 0)  # Single bond

        # 2. Select atom 2 (second C)
        mol.take_action(2)
        # Create third carbon
        mol.take_action(0)  # C atom type
        mol.take_action(self.vocab_size + 0)  # Single bond

        # 3. Select atom 3 (third C)
        mol.take_action(3)
        # Create fourth carbon
        mol.take_action(0)  # C atom type
        mol.take_action(self.vocab_size + 0)  # Single bond

        # 4. Select atom 4 (fourth C)
        mol.take_action(4)
        # Connect back to atom 1 to close the ring
        mol.take_action(self.vocab_size + 0)  # Select atom 1
        mol.take_action(self.vocab_size + 0)  # Single bond

        # Verify we've built the expected structure
        self.assertEqual(len(mol.atoms), 5)  # 1 virtual + 4 carbon atoms

        # Check that we have a ring structure
        # Each atom should have exactly 2 bonds
        for i in range(1, 5):
            self.assertEqual(np.sum(mol.bonds[i, 1:5] > 0), 2)

        # Return the molecule with our ring structure
        return mol, (1, 4)  # Return the molecule and indices of a bond that can be broken

    def print_molecule_state(self, mol):
        """Helper to print the current state of the molecule for debugging"""
        print(f"\nMolecule state:")
        print(f"Atoms: {mol.atoms}")
        print(f"Bonds:\n{mol.bonds}")
        if hasattr(mol, 'has_disconnected_fragments'):
            print(f"Has disconnected fragments: {mol.has_disconnected_fragments}")
        print(f"Current action level: {mol.current_action_level}")
        print(f"SMILES: {mol.to_smiles()}")

    def test_triggering_level3(self):
        """Test that removing a bridge bond correctly triggers Level 3"""
        # Build a molecule with a bridge bond
        mol, bridge_bond = self.build_bridge_molecule()
        bridge_atom1, bridge_atom2 = bridge_bond

        # Store the SMILES representation before bond removal
        original_smiles = mol.to_smiles()
        print(f"Original molecule SMILES: {original_smiles}")

        # Select the first bridge atom
        mol.take_action(bridge_atom1)  # Level 0: Select atom directly

        # Select the second bridge atom
        mol.take_action(self.vocab_size + bridge_atom2 - 1)  # Level 1: Select existing atom

        # Verify we're at Level 2
        self.assertEqual(mol.current_action_level, 2)

        # Get the action for removing a bond
        remove_bond_action = self.vocab_size + 6  # V+6 for bond removal

        # Remove the bridge bond
        mol.take_action(remove_bond_action)

        # Verify we're now at Level 3
        self.assertEqual(mol.current_action_level, 3)

        # Check that the bond has been removed
        self.assertEqual(mol.bonds[bridge_atom1, bridge_atom2], 0)
        self.assertEqual(mol.bonds[bridge_atom2, bridge_atom1], 0)

        # Verify that fragments were created
        self.assertTrue(hasattr(mol, 'fragments'))
        self.assertEqual(len(mol.fragments), 2)

        # The action mask should have exactly 3 actions, all unmasked
        self.assertEqual(len(mol.current_action_mask), 3)
        self.assertFalse(np.any(mol.current_action_mask))  # All actions should be unmasked

        print("Successfully triggered Level 3 fragment handling")

    def test_keep_first_fragment(self):
        """Test keeping only the first fragment after bond removal"""
        # Build a molecule with a bridge bond
        mol, bridge_bond = self.build_bridge_molecule()
        bridge_atom1, bridge_atom2 = bridge_bond

        # Store initial atom count
        initial_atom_count = len(mol.atoms)

        # Get initial SMILES
        original_smiles = mol.to_smiles()
        print(f"Original molecule SMILES: {original_smiles}")

        # Remove the bridge bond and enter Level 3
        mol.take_action(bridge_atom1)  # Level 0: Select atom
        mol.take_action(self.vocab_size + bridge_atom2 - 1)  # Level 1: Select target atom
        mol.take_action(self.vocab_size + 6)  # Level 2: Remove bond

        # Verify we're at Level 3
        self.assertEqual(mol.current_action_level, 3)

        # Keep the first fragment (action 0)
        mol.take_action(0)

        # Verify we're back at Level 0
        self.assertEqual(mol.current_action_level, 0)

        # Verify fragment cleanup
        self.assertFalse(hasattr(mol, 'fragments'))

        # The atom count should be less than the original
        self.assertLess(len(mol.atoms), initial_atom_count)

        # The molecule should be connected
        self.assertFalse(hasattr(mol, 'has_disconnected_fragments'))

        # Verify we can terminate the molecule
        self.assertFalse(mol.current_action_mask[0])  # Termination should be allowed

        # Get the final SMILES
        final_smiles = mol.to_smiles()
        print(f"Fragment 1 molecule SMILES: {final_smiles}")

        # The final SMILES should be different from the original
        self.assertNotEqual(original_smiles, final_smiles)

        print("Successfully kept first fragment")

    def test_keep_second_fragment(self):
        """Test keeping only the second fragment after bond removal"""
        # Build a molecule with a bridge bond
        mol, bridge_bond = self.build_bridge_molecule()
        bridge_atom1, bridge_atom2 = bridge_bond

        # Store initial atom count
        initial_atom_count = len(mol.atoms)

        # Get initial SMILES
        original_smiles = mol.to_smiles()
        print(f"Original molecule SMILES: {original_smiles}")

        # Remove the bridge bond and enter Level 3
        mol.take_action(bridge_atom1)  # Level 0: Select atom
        mol.take_action(self.vocab_size + bridge_atom2 - 1)  # Level 1: Select target atom
        mol.take_action(self.vocab_size + 6)  # Level 2: Remove bond

        # Verify we're at Level 3
        self.assertEqual(mol.current_action_level, 3)

        # Keep the second fragment (action 1)
        mol.take_action(1)

        # Verify we're back at Level 0
        self.assertEqual(mol.current_action_level, 0)

        # Verify fragment cleanup
        self.assertFalse(hasattr(mol, 'fragments'))

        # The atom count should be less than the original
        self.assertLess(len(mol.atoms), initial_atom_count)

        # The molecule should be connected
        self.assertFalse(hasattr(mol, 'has_disconnected_fragments'))

        # Verify we can terminate the molecule
        self.assertFalse(mol.current_action_mask[0])  # Termination should be allowed

        # Get the final SMILES
        final_smiles = mol.to_smiles()
        print(f"Fragment 2 molecule SMILES: {final_smiles}")

        # The final SMILES should be different from the original
        self.assertNotEqual(original_smiles, final_smiles)

        print("Successfully kept second fragment")

    def test_keep_both_fragments(self):
        """Test keeping both fragments which creates a disconnected molecule"""
        # Build a molecule with a bridge bond
        mol, bridge_bond = self.build_bridge_molecule()
        bridge_atom1, bridge_atom2 = bridge_bond

        # Store initial atom count
        initial_atom_count = len(mol.atoms)

        # Get initial SMILES
        original_smiles = mol.to_smiles()
        print(f"Original molecule SMILES: {original_smiles}")

        # Remove the bridge bond and enter Level 3
        mol.take_action(bridge_atom1)  # Level 0: Select atom
        mol.take_action(self.vocab_size + bridge_atom2 - 1)  # Level 1: Select target atom
        mol.take_action(self.vocab_size + 6)  # Level 2: Remove bond

        # Verify we're at Level 3
        self.assertEqual(mol.current_action_level, 3)

        # Keep both fragments (action 2)
        mol.take_action(2)

        # Verify we're back at Level 0
        self.assertEqual(mol.current_action_level, 0)

        # Verify fragment cleanup
        self.assertFalse(hasattr(mol, 'fragments'))

        # The atom count should remain the same
        self.assertEqual(len(mol.atoms), initial_atom_count)

        # The molecule should be marked as disconnected
        self.assertTrue(hasattr(mol, 'has_disconnected_fragments'))
        self.assertTrue(mol.has_disconnected_fragments)

        # Verify we cannot terminate the molecule due to disconnection
        self.assertTrue(mol.current_action_mask[0])  # Termination should be masked

        # Get the SMILES - should show disconnected fragments with a "." separator
        final_smiles = mol.to_smiles()
        print(f"Disconnected molecule SMILES: {final_smiles}")

        # The SMILES should contain a "." indicating disconnected fragments
        self.assertIn(".", final_smiles)

        print("Successfully kept both fragments (disconnected molecule)")

    def test_fragment_consistency(self):
        """Test that fragment handling correctly maintains internal representation consistency"""
        # Build a molecule with a bridge bond
        mol, bridge_bond = self.build_bridge_molecule()
        bridge_atom1, bridge_atom2 = bridge_bond

        # Store initial state
        initial_atoms = mol.atoms.copy()
        initial_bonds = mol.bonds.copy()

        # Remove the bridge bond
        mol.take_action(bridge_atom1)  # Level 0: Select atom
        mol.take_action(self.vocab_size + bridge_atom2 - 1)  # Level 1: Select target atom
        mol.take_action(self.vocab_size + 6)  # Level 2: Remove bond

        # Keep the first fragment (action 0)
        mol.take_action(0)

        # Verify the internal representation
        # 1. Check that atoms array is consistent with RDKit mol
        rdkit_atoms = mol.rdkit_mol.GetAtoms()
        self.assertEqual(len(mol.atoms) - 1, len(rdkit_atoms))  # -1 for virtual atom

        # 2. Check that bonds matrix is symmetric
        self.assertTrue(np.array_equal(mol.bonds, mol.bonds.T))

        # 3. Check that all real atoms have a virtual bond to the virtual atom
        self.assertTrue(np.all(mol.bonds[0, 1:] == mol.virtual_bond_idx))
        self.assertTrue(np.all(mol.bonds[1:, 0] == mol.virtual_bond_idx))

        # 4. Verify topological distance matrix
        # - Diagonal should be 0 (distance to self)
        self.assertTrue(np.all(np.diag(mol.topological_distance_matrix) == 0))

        # - Distances to virtual atom should be the virtual_distance constant
        self.assertTrue(np.all(mol.topological_distance_matrix[0, 1:] == mol.virtual_distance))
        self.assertTrue(np.all(mol.topological_distance_matrix[1:, 0] == mol.virtual_distance))

        # - The distance matrix should be symmetric
        self.assertTrue(np.array_equal(mol.topological_distance_matrix, mol.topological_distance_matrix.T))

        # Verify RDKit consistency by converting to SMILES and back
        rdkit_smiles = mol.to_smiles()
        new_mol = MoleculeDesign.from_smiles(self.config, rdkit_smiles, do_finish=False)
        self.assertEqual(len(mol.atoms), len(new_mol.atoms))

        print("Fragment handling maintains internal consistency")

    def test_reconnect_fragments(self):
        """Test reconnecting fragments after choosing to keep both"""
        # Build a molecule with a bridge bond
        mol, bridge_bond = self.build_bridge_molecule()
        bridge_atom1, bridge_atom2 = bridge_bond

        # Remove the bridge bond and keep both fragments
        mol.take_action(bridge_atom1)  # Level 0: Select atom
        mol.take_action(self.vocab_size + bridge_atom2 - 1)  # Level 1: Select target atom
        mol.take_action(self.vocab_size + 6)  # Level 2: Remove bond
        mol.take_action(2)  # Level 3: Keep both fragments

        # Verify we have disconnected fragments
        self.assertTrue(hasattr(mol, 'has_disconnected_fragments'))
        self.assertTrue(mol.has_disconnected_fragments)

        # Termination should be masked (can't terminate with disconnected fragments)
        self.assertTrue(mol.current_action_mask[0])

        # Recreate the bridge bond
        mol.take_action(bridge_atom1)  # Level 0: Select atom
        mol.take_action(self.vocab_size + bridge_atom2 - 1)  # Level 1: Select target atom
        mol.take_action(self.vocab_size + 0)  # Level 2: Create single bond

        # After reconnecting fragments, the molecule should no longer have the disconnected flag
        self.assertFalse(hasattr(mol, 'has_disconnected_fragments'))

        # Termination should now be unmasked
        self.assertFalse(mol.current_action_mask[0])

        # SMILES should be back to a single molecule (no ".")
        reconnected_smiles = mol.to_smiles()
        self.assertNotIn(".", reconnected_smiles)

        print("Successfully reconnected fragments")

    def build_complex_molecule(self):
        """Helper to build a linear chain with different atom types: C-C-N-O-C-C-F"""
        mol = MoleculeDesign(self.config, self.atom_types["C"])

        # Build linear chain: C-C-N-O-C-C-F
        atom_sequence = [self.atom_types["C"], self.atom_types["N"], self.atom_types["O"],
                         self.atom_types["C"], self.atom_types["C"], self.atom_types["F"]]

        prev_atom_idx = 1  # Start with the first carbon
        for i, atom_type in enumerate(atom_sequence):
            # Select previous atom
            mol.take_action(prev_atom_idx)  # Level 0: Select atom

            # Create new atom (action = atom_type - 1)
            mol.take_action(atom_type - 1)  # Level 1: Create new atom

            # Set single bond
            mol.take_action(self.vocab_size + 0)  # Level 2: Single bond (V+0)

            # Update previous atom index
            prev_atom_idx += 1

        return mol, (4, 5)  # Return molecule and indices of O-C bond

    def test_complex_fragment_handling(self):
        """Test fragment handling with more complex molecules containing different atom types"""
        # Create a molecule with multiple atom types: C-C-N-O-C-C-F
        mol, (o_atom, c_atom) = self.build_complex_molecule()

        # Store initial SMILES
        original_smiles = mol.to_smiles()
        print(f"Original complex molecule: {original_smiles}")

        # Break the bond between O and C
        mol.take_action(o_atom)  # Level 0: Select O atom
        mol.take_action(self.vocab_size + c_atom - 1)  # Level 1: Select C atom
        mol.take_action(self.vocab_size + 6)  # Level 2: Remove bond

        # Verify we're at Level 3
        self.assertEqual(mol.current_action_level, 3)

        # Keep first fragment (C-C-N-O)
        mol.take_action(0)

        # Verify we're back at Level 0
        self.assertEqual(mol.current_action_level, 0)

        # Check the atoms are correct
        self.assertEqual(len(mol.atoms), 5)  # Virtual + 4 atoms
        self.assertEqual(mol.atoms[1], self.atom_types["C"])
        self.assertEqual(mol.atoms[2], self.atom_types["C"])
        self.assertEqual(mol.atoms[3], self.atom_types["N"])
        self.assertEqual(mol.atoms[4], self.atom_types["O"])

        # Make sure it's a valid fragment
        fragment1_smiles = mol.to_smiles()
        print(f"First fragment: {fragment1_smiles}")
        self.assertNotIn(".", fragment1_smiles)

        # Now do the same but keep second fragment
        mol, (o_atom, c_atom) = self.build_complex_molecule()

        # Break the bond between O and C
        mol.take_action(o_atom)  # Level 0: Select O atom
        mol.take_action(self.vocab_size + c_atom - 1)  # Level 1: Select C atom
        mol.take_action(self.vocab_size + 6)  # Level 2: Remove bond

        # Keep second fragment (C-C-F)
        mol.take_action(1)

        # Check the atoms are correct
        self.assertEqual(len(mol.atoms), 4)  # Virtual + 3 atoms
        self.assertEqual(mol.atoms[1], self.atom_types["C"])
        self.assertEqual(mol.atoms[2], self.atom_types["C"])
        self.assertEqual(mol.atoms[3], self.atom_types["F"])

        # Make sure it's a valid fragment
        fragment2_smiles = mol.to_smiles()
        print(f"Second fragment: {fragment2_smiles}")
        self.assertNotIn(".", fragment2_smiles)

        print("Complex fragment handling successful")

    def test_fragment_handling_with_rings(self):
        """Test fragment handling with ring structures"""
        # Build a ring molecule
        mol, ring_bond = self.build_ring_molecule()
        ring_atom1, ring_atom2 = ring_bond

        # Print initial structure
        print(f"Ring molecule: {mol.to_smiles()}")

        # Breaking a bond in a ring should not create disconnected fragments
        mol.take_action(ring_atom1)  # Level 0: Select first atom
        mol.take_action(self.vocab_size + ring_atom2 - 1)  # Level 1: Select second atom

        # Check if remove bond is allowed
        remove_bond_action = self.vocab_size + 6  # V+6 for remove bond
        self.assertFalse(mol.current_action_mask[remove_bond_action])  # Should be unmasked

        # Remove the bond
        mol.take_action(remove_bond_action)

        # Because the ring has other paths to maintain connectivity,
        # it should NOT enter Level 3
        self.assertEqual(mol.current_action_level, 0)
        self.assertFalse(hasattr(mol, 'has_disconnected_fragments'))

        # Bond should be removed
        self.assertEqual(mol.bonds[ring_atom1, ring_atom2], 0)

        # The molecule should still be a valid, connected structure
        ring_opened_smiles = mol.to_smiles()
        print(f"Ring opened: {ring_opened_smiles}")
        self.assertNotIn(".", ring_opened_smiles)

        print("Successfully handled ring bond removal")

    def test_termination_with_disconnected_fragments(self):
        """Test that termination is blocked when molecule has disconnected fragments"""
        # Build a molecule with a bridge bond
        mol, bridge_bond = self.build_bridge_molecule()
        bridge_atom1, bridge_atom2 = bridge_bond

        # Remove the bridge bond and keep both fragments
        mol.take_action(bridge_atom1)  # Level 0: Select atom
        mol.take_action(self.vocab_size + bridge_atom2 - 1)  # Level 1: Select target atom
        mol.take_action(self.vocab_size + 6)  # Level 2: Remove bond
        mol.take_action(2)  # Level 3: Keep both fragments

        # Termination should be masked
        self.assertTrue(mol.current_action_mask[0])

        # Try to terminate directly (this should raise an assertion error)
        with self.assertRaises(AssertionError):
            mol.take_action(0)

        # Reconnect the fragments
        mol.take_action(bridge_atom1)  # Level 0: Select atom
        mol.take_action(self.vocab_size + bridge_atom2 - 1)  # Level 1: Select target atom
        mol.take_action(self.vocab_size + 0)  # Level 2: Create single bond (V+0)

        # Now termination should be allowed
        self.assertFalse(mol.current_action_mask[0])
        mol.take_action(0)  # Should succeed

        # Molecule should be terminated
        self.assertTrue(mol.synthesis_done)

        print("Termination correctly blocked with disconnected fragments")

    def test_bond_addition_between_fragments(self):
        """Test adding bonds between disconnected fragments"""
        # Build a molecule with a bridge bond
        mol, bridge_bond = self.build_bridge_molecule()
        bridge_atom1, bridge_atom2 = bridge_bond

        # Remove the bridge bond and keep both fragments
        mol.take_action(bridge_atom1)  # Level 0: Select atom
        mol.take_action(self.vocab_size + bridge_atom2 - 1)  # Level 1: Select target atom
        mol.take_action(self.vocab_size + 6)  # Level 2: Remove bond
        mol.take_action(2)  # Level 3: Keep both fragments

        # Now try to add a bond between atoms in different fragments
        # Use the same atoms we disconnected
        atom_from_fragment1 = bridge_atom1
        atom_from_fragment2 = bridge_atom2

        # Adding a bond should be allowed
        mol.take_action(atom_from_fragment1)  # Level 0: Select atom from fragment 1

        # The other atom should be selectable
        atom2_action = self.vocab_size + atom_from_fragment2 - 1  # Level 1: Select atom from fragment 2
        self.assertFalse(mol.current_action_mask[atom2_action])

        # Select the atom from fragment 2
        mol.take_action(atom2_action)

        # Create a single bond
        bond_action = self.vocab_size + 0  # V+0 for single bond
        self.assertFalse(mol.current_action_mask[bond_action])  # Should be unmasked
        mol.take_action(bond_action)

        # After reconnecting, the molecule should no longer have disconnected fragments
        self.assertFalse(hasattr(mol, 'has_disconnected_fragments'))

        # Verify the bond was created
        self.assertEqual(mol.bonds[atom_from_fragment1, atom_from_fragment2], 1)

        # The SMILES should be a single connected molecule
        smiles = mol.to_smiles()
        print(f"Reconnected molecule: {smiles}")
        self.assertNotIn(".", smiles)

        print("Successfully reconnected fragments by adding a bond")

    def test_valence_constraints_in_fragments(self):
        """Test that valence constraints are enforced when working with fragments"""
        # Build a complex molecule with different atom types
        mol, (o_atom, c_atom) = self.build_complex_molecule()  # O=atom 4, C=atom 5

        # Remove the bond between O and C to create fragments
        mol.take_action(o_atom)  # Level 0: Select O atom
        mol.take_action(self.vocab_size + c_atom - 1)  # Level 1: Select C atom
        mol.take_action(self.vocab_size + 6)  # Level 2: Remove bond
        mol.take_action(2)  # Level 3: Keep both fragments

        # Try to reconnect with a double bond
        mol.take_action(o_atom)  # Level 0: Select O atom
        mol.take_action(self.vocab_size + c_atom - 1)  # Level 1: Select C atom

        # Check if double bond is allowed (O already has 1 bond, so should only have 1 valence left)
        double_bond_action = self.vocab_size + 1  # V+1 for double bond

        # Oxygen atom (with valence 2) should not allow a triple bond
        triple_bond_action = self.vocab_size + 2  # V+2 for triple bond
        self.assertTrue(mol.current_action_mask[triple_bond_action])  # Should be masked

        # But a double bond should be allowed
        self.assertFalse(mol.current_action_mask[double_bond_action])  # Should be unmasked
        mol.take_action(double_bond_action)

        # Verify the bond was created with order 2
        self.assertEqual(mol.bonds[o_atom, c_atom], 2)

        # The molecule should be connected and valid
        final_smiles = mol.to_smiles()
        print(f"Molecule with double bond: {final_smiles}")
        self.assertNotIn(".", final_smiles)

        print("Valence constraints correctly enforced with fragments")

    def test_multi_step_fragment_operations(self):
        """Test a complex multi-step process with fragment handling"""
        # Build a complex molecule
        mol, (o_atom, c_atom) = self.build_complex_molecule()
        initial_smiles = mol.to_smiles()
        print(f"Initial molecule: {initial_smiles}")

        # Step 1: Create fragments by removing O-C bond
        mol.take_action(o_atom)  # Level 0: Select O atom
        mol.take_action(self.vocab_size + c_atom - 1)  # Level 1: Select C atom
        mol.take_action(self.vocab_size + 6)  # Level 2: Remove bond
        mol.take_action(2)  # Level 3: Keep both fragments

        # Verify we have disconnected fragments
        self.assertTrue(hasattr(mol, 'has_disconnected_fragments'))

        # Step 2: Replace an atom in first fragment (change N to O)
        n_atom = 3  # The nitrogen atom
        mol.take_action(n_atom)  # Level 0: Select N atom
        mol.take_action(self.vocab_size + len(mol.atoms) - 1)  # Level 1: Replacement action (V+N)
        mol.take_action(self.atom_types["O"] - 1)  # Level 2: Replace with O (type 3 → action 2)

        # Step 3: Replace an atom in second fragment (change C to N)
        second_c_atom = 6  # The second carbon in fragment 2
        mol.take_action(second_c_atom)  # Level 0: Select C atom
        mol.take_action(self.vocab_size + len(mol.atoms) - 1)  # Level 1: Replacement action (V+N)
        mol.take_action(self.atom_types["N"] - 1)  # Level 2: Replace with N (type 2 → action 1)

        # Step 4: Connect the fragments with a double bond between O (was N) and C
        mol.take_action(n_atom)  # Level 0: Select former N atom (now O)
        mol.take_action(self.vocab_size + c_atom - 1)  # Level 1: Select C atom
        mol.take_action(self.vocab_size + 1)  # Level 2: Double bond (V+1)

        # Verify molecule is now connected
        self.assertFalse(hasattr(mol, 'has_disconnected_fragments'))

        # Check the final structure is valid
        final_smiles = mol.to_smiles()
        print(f"Final molecule after multiple operations: {final_smiles}")
        self.assertNotIn(".", final_smiles)
        self.assertNotEqual(initial_smiles, final_smiles)

        # Verify we can terminate
        self.assertFalse(mol.current_action_mask[0])
        mol.take_action(0)
        self.assertTrue(mol.synthesis_done)

        print("Successfully completed multi-step fragment operations")

if __name__ == '__main__':
    unittest.main()

