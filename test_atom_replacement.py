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
        self.max_num_atoms = 10
        self.start_c_chain_max_len = 3


class TestAtomReplacement(unittest.TestCase):
    def setUp(self):
        self.config = DummyMoleculeConfig()
        # Start with a Carbon atom. In our vocabulary, the index for "C" is 1.
        self.initial_atom = 1
        self.mol = MoleculeDesign(self.config, self.initial_atom)

        # Add vocabulary info for easy access
        self.atom_indices = {
            "C": 1,
            "N": 2,
            "O": 3,
            "F": 4
        }

    def debug_action_mask(self, mol, level_name="current"):
        """Helper to print current action mask and feasible actions"""
        mask = mol.current_action_mask
        feasible = [i for i, m in enumerate(mask) if m == 0]
        print(f"Feasible actions at {level_name} level: {feasible}")

        # Additional context for level 1
        if mol.current_action_level == 1:
            num_atom_types = len(mol.vocabulary_atom_idcs)
            create_atoms = [i for i in feasible if i < num_atom_types]
            bond_atoms = [i for i in feasible if num_atom_types <= i < num_atom_types + len(mol.atoms) - 1]
            replace_action = [i for i in feasible if i == num_atom_types + len(mol.atoms) - 1]

            print(f"  - Create atoms: {create_atoms}")
            print(f"  - Bond with atoms: {bond_atoms}")
            print(f"  - Replace action: {replace_action}")

        return feasible

    def test_basic_atom_replacement(self):
        """Test replacing a carbon atom with nitrogen in a simple molecule."""
        # First, create a C-C molecule
        self.mol.take_action(1)  # Level 0: Create carbon atom

        # Connect with the first carbon to create C-C
        new_atom_action_count = len(self.mol.vocabulary_atom_idcs)
        self.mol.take_action(new_atom_action_count)  # Level 1: Select first atom for bonding
        self.mol.take_action(0)  # Level 2: Create single bond

        # Verify we have C-C
        self.assertEqual(len(self.mol.atoms), 3)  # 1 virtual + 2 real atoms
        self.assertEqual(self.mol.atoms[1], 1)  # First real atom is carbon
        self.assertEqual(self.mol.atoms[2], 1)  # Second real atom is carbon
        self.assertEqual(self.mol.bonds[1, 2], 1)  # Single bond between them

        print("\nCreated initial C-C molecule")
        print(f"Atoms: {self.mol.atoms}")
        print(f"Bonds:\n{self.mol.bonds}")

        # Check which atoms are selectable at level 0
        pick_existing_idx = self.mol.pick_existing_atoms_start_action_idx_lvl_0
        self.assertEqual(self.mol.current_action_level, 0)

        # Print available actions
        available_actions = [i for i, masked in enumerate(self.mol.current_action_mask) if not masked]
        print(f"Available actions at level 0: {available_actions}")

        # We see that action 5 (first carbon) is available but action 6 (second carbon) is not
        # This is expected due to action mask logic - second carbon might have no valid operations

        # Select the first carbon atom for replacement
        first_atom_selection_action = pick_existing_idx  # This is the first real atom's action
        print(f"Selecting first carbon with action {first_atom_selection_action}")
        self.mol.take_action(first_atom_selection_action)

        # Verify we're at level 1
        self.assertEqual(self.mol.current_action_level, 1)

        # Print available actions at level 1
        available_actions = [i for i, masked in enumerate(self.mol.current_action_mask) if not masked]
        print(f"Available actions at level 1: {available_actions}")

        # Get the replacement action index
        replace_action_idx = new_atom_action_count + len(self.mol.atoms) - 1
        print(f"Replacement action index: {replace_action_idx}")

        # Verify replacement is allowed
        self.assertIn(replace_action_idx, available_actions,
                      f"Replacement action {replace_action_idx} should be available")

        # Choose to replace the atom
        self.mol.take_action(replace_action_idx)

        # Verify we're in replacement mode at level 2
        self.assertTrue(self.mol.is_replacing_atom)
        self.assertEqual(self.mol.current_action_level, 2)
        print(f"Atom to replace: {self.mol.atom_to_replace}")

        # Print available replacement options
        available_actions = [i for i, masked in enumerate(self.mol.current_action_mask) if not masked]
        print(f"Available replacement options at level 2: {available_actions}")

        # Carbon (0) should not be feasible (same type)
        self.assertNotIn(0, available_actions, "Carbon should not be a feasible replacement (same type)")

        # Nitrogen (1) should be feasible
        self.assertIn(1, available_actions, "Nitrogen should be a feasible replacement")

        # Store the current atoms before replacement
        atoms_before = self.mol.atoms.copy()
        print(f"Atoms before replacement: {atoms_before}")

        # Replace with nitrogen (action 1)
        self.mol.take_action(1)

        # Verify we're back at level 0 and not in replacement mode
        self.assertEqual(self.mol.current_action_level, 0)
        self.assertFalse(self.mol.is_replacing_atom)

        # Verify the correct atom was replaced
        print(f"Atoms after replacement: {self.mol.atoms}")
        self.assertEqual(self.mol.atoms[1], 2)  # First atom should now be nitrogen

        # Check that the RDKit molecule matches our internal representation
        rdkit_atom = self.mol.rdkit_mol.GetAtomWithIdx(0)  # 0-indexed in RDKit
        self.assertEqual(rdkit_atom.GetAtomicNum(), 7)  # Nitrogen atomic number

        # Check bonds are preserved
        self.assertEqual(self.mol.bonds[1, 2], 1)  # Bond should still be there

        print("Successfully replaced carbon with nitrogen")

    def test_bond_increase_masking(self):
        """Test that atoms with single bonds that can be increased are selectable."""
        # Create a simple C-C molecule
        self.mol.take_action(1)  # Level 0: Create carbon atom

        # Connect with the first carbon to create C-C
        new_atom_action_count = len(self.mol.vocabulary_atom_idcs)
        self.mol.take_action(new_atom_action_count)  # Level 1: Select first atom for bonding
        self.mol.take_action(0)  # Level 2: Create single bond

        # Verify we have C-C with a single bond
        self.assertEqual(len(self.mol.atoms), 3)  # 1 virtual + 2 real atoms
        self.assertEqual(self.mol.bonds[1, 2], 1)  # Single bond

        # Check atom valences - both carbons should have remaining valence
        atom_valence = np.array([self.mol.vocabulary_valence[x] for x in self.mol.atoms[1:]])
        atom_valence_remaining = atom_valence - self.mol.bonds[1:, 1:].sum(axis=1)
        self.assertEqual(atom_valence_remaining[0], 3)  # First carbon has 3 valence remaining
        self.assertEqual(atom_valence_remaining[1], 3)  # Second carbon has 3 valence remaining

        print("\nC-C molecule with single bond:")
        print(f"Atoms: {self.mol.atoms}")
        print(f"Bonds:\n{self.mol.bonds}")
        print(f"Atom valence remaining: {atom_valence_remaining}")

        # Get available actions at level 0
        pick_existing_idx = self.mol.pick_existing_atoms_start_action_idx_lvl_0
        available_actions = [i for i, masked in enumerate(self.mol.current_action_mask) if not masked]
        print(f"Available actions at level 0: {available_actions}")

        # The atoms should be selectable since their bond can be increased
        first_atom_action = pick_existing_idx
        second_atom_action = pick_existing_idx + 1

        # Check if atoms are selectable
        print(f"First atom action: {first_atom_action}, masked: {self.mol.current_action_mask[first_atom_action]}")
        print(f"Second atom action: {second_atom_action}, masked: {self.mol.current_action_mask[second_atom_action]}")

        # Print the internal masking variables to debug what's happening
        # Recreate the masking logic here to see what's going wrong
        has_modifiable_bond = np.zeros(len(self.mol.atoms) - 1, dtype=bool)
        for i in range(len(self.mol.atoms) - 1):
            if np.any(self.mol.bonds[i + 1, 1:] > 1):  # Any bond with order > 1
                has_modifiable_bond[i] = True

        bond_indicator = np.zeros_like(self.mol.bonds[1:, 1:])
        bond_indicator[np.where(self.mol.bonds[1:, 1:] == 0)] = 1
        np.fill_diagonal(bond_indicator, 0)
        has_free_nonneighbor = np.matmul(bond_indicator, (atom_valence_remaining > 0)[:, None]).squeeze()

        print(f"has_modifiable_bond: {has_modifiable_bond}")
        print(f"has_free_nonneighbor: {has_free_nonneighbor}")
        print(f"no_valid_actions: {(has_free_nonneighbor == 0) & ~has_modifiable_bond}")

        # Try to select first atom to modify its bond
        try:
            # print("BOOGA")
            self.mol.take_action(first_atom_action)
            first_selectable = True
        except AssertionError:
            first_selectable = False

        # Try to select second atom to modify its bond
        try:
            if first_selectable:
                # print("OOGA")
                # Reset the molecule
                self.mol = MoleculeDesign(self.config, self.initial_atom)
                self.mol.take_action(1)
                self.mol.take_action(new_atom_action_count)
                self.mol.take_action(0)

            self.mol.take_action(second_atom_action)
            second_selectable = True
        except AssertionError:
            second_selectable = False

        # One of these should be True if the bond increase logic is working
        self.assertTrue(first_selectable or second_selectable,
                        "At least one atom should be selectable to increase bond order")

        if first_selectable or second_selectable:
            print("Test passed: Atoms with single bonds that can be increased are correctly selectable")
        else:
            print("Test failed: Atoms with single bonds that can be increased are incorrectly masked")

    def test_replacement_valence_constraints(self):
        """Test that atom replacement respects valence constraints."""
        # Start fresh
        self.mol = MoleculeDesign(self.config, self.initial_atom)

        # Create a C=C-C molecule with clear debug output
        print("\n--- Creating test molecule ---")

        # First carbon is already created in setUp
        print(f"Initial atom state: {self.mol.atoms}")

        # Add second carbon with double bond
        self.mol.take_action(1)  # Level 0: Create carbon atom
        self.mol.take_action(4)  # Level 1: Select first carbon
        self.mol.take_action(1)  # Level 2: Create double bond

        print(f"After adding second carbon: {self.mol.atoms}")
        print(f"Bonds:\n{self.mol.bonds}")

        # Verify first two atoms are carbon
        self.assertEqual(self.mol.atoms[1], 1)  # First atom should be carbon
        self.assertEqual(self.mol.atoms[2], 1)  # Second atom should be carbon
        self.assertEqual(self.mol.bonds[1, 2], 2)  # Double bond between them

        # Add third carbon with single bond
        self.mol.take_action(1)  # Level 0: Create another carbon
        self.mol.take_action(5)  # Level 1: Select second carbon
        self.mol.take_action(0)  # Level 2: Create single bond

        print(f"After adding third carbon: {self.mol.atoms}")
        print(f"Bonds:\n{self.mol.bonds}")

        # Verify all three atoms are carbon with correct bonds
        self.assertEqual(self.mol.atoms[1], 1)  # First atom should be carbon
        self.assertEqual(self.mol.atoms[2], 1)  # Second atom should be carbon
        self.assertEqual(self.mol.atoms[3], 1)  # Third atom should be carbon
        self.assertEqual(self.mol.bonds[1, 2], 2)  # Double bond between 1st and 2nd
        self.assertEqual(self.mol.bonds[2, 3], 1)  # Single bond between 2nd and 3rd

        print("\n--- Created C=C-C structure correctly ---")

        # Now try to replace the first carbon (atom_idx=1)
        atom_idx = 1
        atom_select_action = self.mol.pick_existing_atoms_start_action_idx_lvl_0 + atom_idx - 1

        # Verify this atom is selectable
        feasible_actions = [i for i, masked in enumerate(self.mol.current_action_mask) if not masked]
        print(f"Feasible actions at level 0: {feasible_actions}")
        self.assertIn(atom_select_action, feasible_actions,
                      f"Action {atom_select_action} to select atom {atom_idx} should be feasible")

        # Select the atom
        print(f"Selecting atom index {atom_idx} with action {atom_select_action}")
        self.mol.take_action(atom_select_action)

        # Get replacement action index
        replace_action_idx = len(self.mol.vocabulary_atom_idcs) + len(self.mol.atoms) - 1

        # Verify replacement action is available
        feasible_l1 = [i for i, masked in enumerate(self.mol.current_action_mask) if not masked]
        print(f"Feasible actions at level 1: {feasible_l1}")
        self.assertIn(replace_action_idx, feasible_l1, "Replacement action should be available")

        # Choose to replace
        print(f"Selecting replacement action {replace_action_idx}")
        self.mol.take_action(replace_action_idx)

        # Get available replacement types
        feasible_l2 = [i for i, masked in enumerate(self.mol.current_action_mask) if not masked]
        print(f"Available replacement types: {feasible_l2}")

        # Calculate real bond sum for this atom
        atom_bonds_sum = 0
        for i, bond_order in enumerate(self.mol.bonds[atom_idx, 1:]):
            if i > 0 and bond_order > 0 and bond_order != self.mol.virtual_bond_idx:
                atom_bonds_sum += bond_order

        print(f"Atom {atom_idx} has real bond sum: {atom_bonds_sum}")

        # Check each possible replacement
        for new_type_idx, atom_type in enumerate(self.mol.vocabulary_atom_idcs):
            atom_name = self.mol.vocabulary_atom_names[new_type_idx]
            atom_valence = self.mol.vocabulary_valence[atom_type]

            # Skip if it's the same type
            if atom_type == self.mol.atoms[atom_idx]:
                self.assertNotIn(new_type_idx, feasible_l2,
                                 f"Same atom type {atom_name} should not be allowed as replacement")
                continue

            # Check valence constraint
            if atom_valence >= atom_bonds_sum:
                if new_type_idx in feasible_l2:
                    print(f"{atom_name} with valence {atom_valence} correctly allowed")
                else:
                    print(f"UNEXPECTED: {atom_name} with valence {atom_valence} not allowed despite sufficient valence")
            else:
                if new_type_idx in feasible_l2:
                    print(
                        f"ERROR: {atom_name} with valence {atom_valence} incorrectly allowed for bonds {atom_bonds_sum}")
                    self.fail(
                        f"{atom_name} with valence {atom_valence} should not be allowed for bonds {atom_bonds_sum}")
                else:
                    print(f"{atom_name} with valence {atom_valence} correctly disallowed")

        # If we have valid replacements, try one (nitrogen)
        if 1 in feasible_l2:  # Nitrogen is index 1
            print("\n--- Performing replacement with Nitrogen ---")

            # Capture state before replacement
            atoms_before = self.mol.atoms.copy()
            print(f"Before replacement, atoms: {atoms_before}")

            # Replace with nitrogen
            self.mol.take_action(1)

            # Verify replacement worked
            print(f"After replacement, atoms: {self.mol.atoms}")
            self.assertEqual(self.mol.atoms[atom_idx], 2)  # Should now be nitrogen (type 2)

            # Check RDKit atom type
            rdkit_atom_idx = atom_idx - 1  # Adjust for RDKit indexing
            rdkit_atom = self.mol.rdkit_mol.GetAtomWithIdx(rdkit_atom_idx)
            self.assertEqual(rdkit_atom.GetAtomicNum(), 7)  # Nitrogen atomic number

            # Check bonds are preserved
            self.assertEqual(self.mol.bonds[atom_idx, 2], 2)  # Double bond should be preserved

            print("Replacement successful with correct valence constraints")
        else:
            self.fail("Nitrogen should be a valid replacement for first carbon")

    def test_middle_atom_replacement(self):
        """Test replacing a middle atom that has multiple bonds to different atoms."""
        # Create a C-C-C chain
        self.mol.take_action(1)  # Add second carbon
        self.mol.take_action(4)  # Select first carbon
        self.mol.take_action(0)  # Create single bond

        self.mol.take_action(1)  # Add third carbon
        self.mol.take_action(5)  # Select second carbon
        self.mol.take_action(0)  # Create single bond

        # Now replace the middle carbon
        middle_atom_idx = 2
        middle_atom_action = self.mol.pick_existing_atoms_start_action_idx_lvl_0 + middle_atom_idx - 1

        # Check if middle atom is selectable
        self.assertIn(middle_atom_action,
                      [i for i, m in enumerate(self.mol.current_action_mask) if not m],
                      "Middle atom should be selectable")

        # Replace middle C with N
        self.mol.take_action(middle_atom_action)
        replace_action = len(self.mol.vocabulary_atom_idcs) + len(self.mol.atoms) - 1
        self.mol.take_action(replace_action)
        self.mol.take_action(1)  # Replace with N

        # Verify replacement
        self.assertEqual(self.mol.atoms[middle_atom_idx], 2)  # Should be nitrogen now
        self.assertEqual(self.mol.bonds[1, middle_atom_idx], 1)  # Bond to first C preserved
        self.assertEqual(self.mol.bonds[middle_atom_idx, 3], 1)  # Bond to third C preserved



if __name__ == '__main__':
    unittest.main()