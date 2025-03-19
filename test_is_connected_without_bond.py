import numpy as np
import unittest

class DummyConfig:
    def __init__(self):
        # The Atom vocabulary should include the "virtual" atom plus real atoms.
        # For testing, we just provide a simple list.
        self.atom_vocabulary = ['virtual', 'A', 'B', 'C']

# This is our MoleculeDesign class with the expected __init__ signature.
class MoleculeDesign:
    def __init__(self, config, initial_atom: int):
        """
        For testing we ignore initial_atom aside from storing atoms and bonds.
        """
        self.config = config
        self.atom_vocabulary = self.config.atom_vocabulary
        # For testing, we'll initialize atoms and bonds later.
        self.atoms = []
        self.bonds = None

    def is_connected_without_bond(self, atom1: int, atom2: int) -> bool:
        """
        Checks whether the molecule would remain connected if the bond between atom1 and atom2 were removed.
        Uses a depth-first search from atom1.
        Note: atom indices include virtual atom at index 0.
        """
        # Special case: If there are only two real atoms, removal disconnects.
        if len(self.atoms) <= 3:
            return False

        # Create a copy of the bonds matrix without the bond between atom1 and atom2.
        bonds_copy = self.bonds.copy()
        bonds_copy[atom1, atom2] = 0
        bonds_copy[atom2, atom1] = 0

        # Perform DFS for connectivity (skip virtual atom at index 0).
        visited = np.zeros(len(self.atoms), dtype=bool)
        stack = [atom1]
        visited[atom1] = True

        while stack:
            current = stack.pop()
            if current == atom2:
                return True
            for neighbor in range(1, len(self.atoms)):  # Skip the virtual atom (index 0)
                if bonds_copy[current, neighbor] > 0 and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        return False

class TestIsConnectedWithoutBond(unittest.TestCase):

    def test_two_real_atoms(self):
        """
        For a molecule with only two real atoms (total atoms = virtual + 2 = 3),
        the function should always return False.
        """
        # Create dummy config and instance.
        config = DummyConfig()
        mol = MoleculeDesign(config, initial_atom=1)
        # Set up atoms list: index 0 is virtual, 1 and 2 are real.
        mol.atoms = ['virtual', 'A', 'B']
        # Set up bonds matrix for 3 atoms.
        mol.bonds = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0]
        ])
        # Since there are only two real atoms, removal of their bond should disconnect.
        self.assertFalse(mol.is_connected_without_bond(1, 2),
                         "With two real atoms, removing the bond should disconnect the molecule.")

    def test_disconnected_removal(self):
        """
        Test a linear chain: virtual atom, A-B, B-C.
        Removal of the bond between A and B (or B and C) should disconnect the structure.
        """
        config = DummyConfig()
        mol = MoleculeDesign(config, initial_atom=1)
        # Atoms: index 0 virtual, 1 A, 2 B, 3 C.
        mol.atoms = ['virtual', 'A', 'B', 'C']
        mol.bonds = np.array([
            [0, 0, 0, 0],
            [0, 0, 1, 0],  # A bonded only to B.
            [0, 1, 0, 1],  # B bonded to A and C.
            [0, 0, 1, 0]   # C bonded only to B.
        ])
        # Removal of A-B should disconnect A from B.
        self.assertFalse(mol.is_connected_without_bond(1, 2),
                         "In a chain, removal of the direct bond without an alternate path should disconnect.")
        # Removal of B-C should disconnect B and C.
        self.assertFalse(mol.is_connected_without_bond(2, 3),
                         "Removal of the bond in a chain disconnects the two atoms if no alternate path exists.")

    def test_connected_via_cycle(self):
        """
        Test a cycle: virtual, then real atoms A, B, C with bonds forming a cycle (A-B, B-C, A-C).
        Removal of any one bond should still leave the molecule connected.
        """
        config = DummyConfig()
        mol = MoleculeDesign(config, initial_atom=1)
        mol.atoms = ['virtual', 'A', 'B', 'C']
        mol.bonds = np.array([
            [0, 0, 0, 0],
            [0, 0, 1, 1],  # A bonded to B and C.
            [0, 1, 0, 1],  # B bonded to A and C.
            [0, 1, 1, 0]   # C bonded to A and B.
        ])
        # Removing A-B: path A -> C -> B exists.
        self.assertTrue(mol.is_connected_without_bond(1, 2),
                        "In a cycle, removal of one bond should still leave the molecule connected.")
        # Removing A-C: path A -> B -> C.
        self.assertTrue(mol.is_connected_without_bond(1, 3),
                        "In a cycle, removal of one bond should not disconnect the molecule.")
        # Removing B-C: path B -> A -> C.
        self.assertTrue(mol.is_connected_without_bond(2, 3),
                        "In a cycle, removing a bond should still leave an alternate path.")

if __name__ == '__main__':
    unittest.main()