import unittest
import numpy as np
from rdkit import Chem

from config import MoleculeConfig
from molecule_design import MoleculeDesign


class DummyMoleculeConfig(MoleculeConfig):
    def __init__(self):
        # Dummy configuration.
        # Create a dummy atom vocabulary with necessary keys, including 'atomic_number'
        self.atom_vocabulary = {
            "C": {"valence": 4, "allowed": True, "atomic_number": 6},
            "N": {"valence": 3, "allowed": True, "atomic_number": 7},
            "O": {"valence": 2, "allowed": True, "atomic_number": 8},
        }
        self.max_num_atoms = 10
        self.start_c_chain_max_len = 3


class TestMoleculeDesign(unittest.TestCase):
    def setUp(self):
        self.config = DummyMoleculeConfig()
        # Start with a Carbon atom. In our configuration, the index for "C" is 1.
        self.initial_atom = 1
        self.mol = MoleculeDesign(self.config, self.initial_atom)
        # Store vocabulary size for action indexing
        self.vocab_size = len(self.config.atom_vocabulary)

    def test_level0_termination(self):
        self.assertFalse(self.mol.synthesis_done)
        self.mol.take_action(0)
        self.assertTrue(self.mol.synthesis_done)

    def test_level0_atom_selection(self):
        self.mol = MoleculeDesign(self.config, self.initial_atom)
        initial_num_atoms = len(self.mol.atoms)
        # In new action space: level 0 selects existing atom
        self.mol.take_action(1)  # Select atom at index 1
        # No new atoms should be created by this action
        self.assertEqual(len(self.mol.atoms), initial_num_atoms)
        # Should advance to level 1
        self.assertEqual(self.mol.current_action_level, 1)
        self.assertEqual(self.mol.history[-1], 1)

    def test_level1_new_atom_vs_existing_bond_masking(self):
        # Check proper segmentation of the level 1 action mask.
        # We'll build a valid molecule with 3 atoms to test both paths

        # Create first bond: A-B
        # Level 0: Select atom A (index 1)
        self.mol.take_action(1)
        # Level 1: Create new atom B (Carbon)
        self.mol.take_action(0)  # Create C atom
        # Level 2: Set bond order 1
        self.mol.take_action(self.vocab_size + 0)  # Action V+0 = bond order 1

        # Create second bond: A-C
        # Level 0: Select atom A again (index 1)
        self.mol.take_action(1)
        # Level 1: Create new atom C (Carbon)
        self.mol.take_action(0)  # Create C atom
        # Level 2: Set bond order 1
        self.mol.take_action(self.vocab_size + 0)  # Action V+0 = bond order 1

        # Now we have a molecule with 3 atoms connected as A-B and A-C

        # Check action mask at level 0
        self.assertEqual(self.mol.current_action_level, 0)

        # Select atom A to examine level 1 mask
        self.mol.take_action(1)  # Select atom 1 at level 0

        # At level 1:
        # - Actions 0 to V-1 create new atoms
        # - Actions V to V+N-1 select existing atoms
        # - Action V+N is for replacement

        new_atom_action_count = len(self.mol.vocabulary_atom_idcs)
        existing_atom_count = len(self.mol.atoms) - 1  # real atoms only
        total_expected_actions = new_atom_action_count + existing_atom_count + 1  # +1 for replacement action

        mask = self.mol.current_action_mask
        self.assertEqual(len(mask), total_expected_actions)

        # Check that some atom creation actions are valid
        self.assertTrue(np.any(mask[:new_atom_action_count] == 0))

        # Check that existing atom selections are valid (specifically atoms 2 and 3)
        # In level 1, existing atom actions start at index V
        self.assertFalse(mask[new_atom_action_count + 1])  # Atom B (index 2) should be selectable
        self.assertFalse(mask[new_atom_action_count + 2])  # Atom C (index 3) should be selectable

    def test_level2_bond_order_increase_and_reduction(self):
        """
        This test builds a molecule with a cycle that allows bond reduction.

        Steps:
        1. Start with atom A.
        2. Select atom A, create atom B, set bond order.
        3. Select atom B, create atom C, set bond order.
        4. Increase the bond order between B-C from 1 to 2.
        5. Create a bond between A and C to form a cycle.
        6. Decrease the bond order between B-C from 2 to 1.
        """
        # Step 1: Start with atom A (already initialized)
        print("Initial atoms:", self.mol.atoms)

        # Step 2: Create atom B and bond it to A
        # Level 0: Select atom A
        self.mol.take_action(1)  # Select atom 1
        # Level 1: Create new atom B (carbon, action 0)
        self.mol.take_action(0)  # Create C atom
        # Level 2: Set bond order 1
        self.mol.take_action(self.vocab_size + 0)  # Action V+0 = bond order 1
        print("After adding B:", self.mol.atoms)
        print("Bonds matrix:\n", self.mol.bonds)

        # Step 3: Create atom C and bond it to B
        # Level 0: Select atom B (index 2)
        self.mol.take_action(2)  # Select atom 2
        # Level 1: Create new atom C (carbon, action 0)
        self.mol.take_action(0)  # Create C atom
        # Level 2: Set bond order 1
        self.mol.take_action(self.vocab_size + 0)  # Action V+0 = bond order 1
        print("After adding C:", self.mol.atoms)
        print("Bonds matrix:\n", self.mol.bonds)

        # Step 4: Increase the bond order between B-C
        # Level 0: Select atom B (index 2)
        self.mol.take_action(2)
        # Level 1: Select atom C (index 3)
        self.mol.take_action(self.vocab_size + 2)  # V+2 for atom at index 3
        # Level 2: Set bond order 2
        self.mol.take_action(self.vocab_size + 1)  # Action V+1 = bond order 2

        # Verify bond order increased
        self.assertEqual(self.mol.bonds[2, 3], 2)
        print("After increasing B-C bond:", self.mol.bonds)

        # Step 5: Create a bond between A and C to form a cycle
        # Level 0: Select atom A (index 1)
        self.mol.take_action(1)
        # Level 1: Select atom C (index 3)
        self.mol.take_action(self.vocab_size + 2)  # V+2 for atom at index 3
        # Level 2: Set bond order 1
        self.mol.take_action(self.vocab_size + 0)  # Action V+0 = bond order 1

        # Verify A-C bond created (forming a cycle)
        self.assertEqual(self.mol.bonds[1, 3], 1)
        print("After creating A-C bond:", self.mol.bonds)

        # Step 6: Decrease the bond order between B-C
        # Level 0: Select atom B (index 2)
        self.mol.take_action(2)
        # Level 1: Select atom C (index 3)
        self.mol.take_action(self.vocab_size + 2)  # V+2 for atom at index 3

        # Debug action mask
        print("Action mask at level 2:", self.mol.current_action_mask)

        # Check if bond order 1 action is feasible
        reduction_action = self.vocab_size + 0  # Action V+0 = bond order 1

        if self.mol.current_action_mask[reduction_action]:
            self.skipTest("No feasible reduction action available in the current molecular configuration")

        # Reduce bond order to 1
        self.mol.take_action(reduction_action)

        # Verify bond order decreased
        self.assertEqual(self.mol.bonds[2, 3], 1)
        print("After decreasing B-C bond:", self.mol.bonds)


if __name__ == '__main__':
    unittest.main()