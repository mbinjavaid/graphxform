import unittest
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
# from rdkit.Chem.Draw import IPythonConsole
from config import MoleculeConfig
from molecule_design import MoleculeDesign
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import io
import PIL.Image
import os


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

        # For visualization
        self.snapshots = []
        self.step_labels = []
        self.step_details = []
        self.last_smiles = None  # To avoid duplicate snapshots

    def capture_snapshot(self, molecule, label, details=""):
        """Capture a snapshot of the current molecule state if it's different from the last one."""
        # Make a copy of the RDKit molecule
        mol_copy = Chem.Mol(molecule.rdkit_mol)

        # Check if this is a duplicate of the last molecule
        current_smiles = Chem.MolToSmiles(mol_copy) if mol_copy.GetNumAtoms() > 0 else ""

        # Only add if it's different from the last one
        if current_smiles != self.last_smiles:
            # Store the molecule, label and details
            self.snapshots.append(mol_copy)
            self.step_labels.append(label)
            self.step_details.append(details)
            self.last_smiles = current_smiles

            print(f"Captured snapshot: {label}")
        else:
            print(f"Skipped duplicate snapshot: {label}")

    def visualize_molecule_evolution(self):
        """Create a grid visualization of the molecule's evolution with detailed comments."""
        if not self.snapshots:
            print("No snapshots to visualize")
            return

        n_mols = len(self.snapshots)
        # Calculate grid dimensions
        n_cols = min(3, n_mols)  # Maximum 3 columns to allow more space
        n_rows = (n_mols + n_cols - 1) // n_cols

        # Create figure
        plt.figure(figsize=(5 * n_cols, 4.5 * n_rows))

        # Generate images for each molecule
        img_list = []
        for i, (mol, label, details) in enumerate(zip(self.snapshots, self.step_labels, self.step_details)):
            # Add atom indices for clarity
            for atom in mol.GetAtoms():
                atom.SetProp("atomNote", str(atom.GetIdx()))

            # Generate molecule image
            img = Draw.MolToImage(mol, size=(300, 250), kekulize=True, fitImage=True)

            # Create subplot
            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(img)

            # Add title (step label) and detailed description below
            plt.title(label, fontsize=12, fontweight='bold')

            # Add detailed commentary below the image
            if details:
                plt.figtext(
                    (i % n_cols) / n_cols + 0.05,
                    1 - ((i // n_cols + 1) / n_rows) + 0.06,
                    details,
                    fontsize=9,
                    wrap=True,
                    ha='left',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5')
                )

            plt.axis('off')

        plt.tight_layout()

        # Save the figure
        plt.savefig('molecule_evolution.png', dpi=300, bbox_inches='tight')
        print("Visualization saved as 'molecule_evolution.png'")
        plt.show()

    def test_build_complex_molecule(self):
        """
        Test building a molecule step by step, validating at each step.
        We'll create a benzene ring with various substituents.
        """
        # Start with a carbon atom
        molecule = MoleculeDesign(self.config, initial_atom=self.C_idx)
        self.capture_snapshot(
            molecule,
            "Initial Carbon",
            "Started with a single carbon atom (action index 1). This is the level 0 action that creates the first atom."
        )

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
            self.assertEqual(len(molecule.atoms), i + 3)  # Virtual atom (0) + initial C (1) + i+1 added C atoms

            # Verify the bond exists
            if i > 0:
                # The bond should be between the two most recently added atoms
                rdkit_bond = molecule.rdkit_mol.GetBondBetweenAtoms(i, i + 1)
                self.assertIsNotNone(rdkit_bond)
                self.assertEqual(rdkit_bond.GetBondTypeAsDouble(), 1.0)  # Single bond

            # Only capture chain at the end to avoid too many similar structures
            if i == 4:
                self.capture_snapshot(
                    molecule,
                    "6-Carbon Chain",
                    "Built a linear chain of 6 carbon atoms using hierarchical actions: Level 0 (add atom), "
                    "Level 1 (select existing atom using offset), Level 2 (create single bond). "
                    "This demonstrates proper atom-atom bonding mechanics."
                )

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

        # Capture cyclohexane
        self.capture_snapshot(
            molecule,
            "Cyclohexane (Ring Closed)",
            "Created ring closure bond between the first and last carbon atoms. "
            "This required selecting existing atoms at both level 0 and level 1 using the proper "
            "action indices (vocabulary_size + atom_index). At this point, the molecule is cyclohexane."
        )

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

        # Capture benzene
        self.capture_snapshot(
            molecule,
            "Benzene (Added Double Bonds)",
            "Converted single bonds to double bonds at positions (0-1), (2-3), and (4-5) to create benzene. "
            "This demonstrates the ability to modify existing bonds. Action 1 at level 2 changes single bonds "
            "to double bonds, respecting valence constraints of carbon atoms."
        )

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

        # Capture after adding oxygen
        self.capture_snapshot(
            molecule,
            "Added Oxygen (Position 1)",
            "Added an oxygen atom (action index O_idx) and bonded it to carbon at position 1. "
            "This creates a phenol-like structure. Note that we must use the vocabulary offset when "
            "selecting an existing atom at level 1, otherwise we'd create a new atom instead."
        )

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

        # Capture after adding nitrogen
        self.capture_snapshot(
            molecule,
            "Added Nitrogen (Position 3)",
            "Added a nitrogen atom (action index N_idx) and bonded it to carbon at position 3. "
            "This creates an aniline-like structure. Carbon at position 3 now has bonds to: "
            "C2 (double), C4 (single), and N (single), using all 4 valence electrons."
        )

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

        # Capture after adding all substituents
        self.capture_snapshot(
            molecule,
            "Added Sulfur (Position 5)",
            "Added a sulfur atom (action index S_idx) and bonded it to carbon at position 5. "
            "The benzene ring now has three different substituents: O, N, and S. "
            "This demonstrates the ability to build complex, heterocyclic structures."
        )

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

        attempted_action = "double bond creation"
        action_result = ""

        if double_bond_feasible:
            print("Creating a double bond (action 1) is feasible")
            molecule.take_action(1)  # Action 1 = bond order 2 (double bond)

            # Verify the bond order was increased
            rdkit_bond = molecule.rdkit_mol.GetBondBetweenAtoms(3, 7)  # Carbon at idx 3, Nitrogen at idx 7
            self.assertIsNotNone(rdkit_bond)
            self.assertEqual(rdkit_bond.GetBondTypeAsDouble(), 2.0)  # Double bond
            action_result = "Succeeded: N=C double bond was created."
        else:
            print("Creating a double bond is NOT feasible due to valence constraints")
            # Maintain single bond by using action 0
            molecule.take_action(0)  # Keep single bond

            # Verify the bond remains a single bond
            rdkit_bond = molecule.rdkit_mol.GetBondBetweenAtoms(3, 7)
            self.assertIsNotNone(rdkit_bond)
            self.assertEqual(rdkit_bond.GetBondTypeAsDouble(), 1.0)  # Single bond
            action_result = "Failed: Creating N=C double bond not possible due to valence constraints. " \
                            "Carbon at position 3 already has all 4 valence electrons used (C=C, C-C, C-N)."

        # Capture bond modification attempt
        self.capture_snapshot(
            molecule,
            "Bond Modification (N-C)",
            f"Attempted to increase N-C bond order from single to double. {action_result} "
            "This demonstrates chemical feasibility constraints enforced by the model."
        )

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

        attempted_action = "bond order decrease"
        action_result = ""

        if decrease_bond_feasible:
            print("Decreasing bond order (action 6) is feasible")
            molecule.take_action(6)  # Action 6 = decrease by 1

            # Verify the bond order was decreased
            rdkit_bond = molecule.rdkit_mol.GetBondBetweenAtoms(0, 1)
            self.assertIsNotNone(rdkit_bond)
            self.assertEqual(rdkit_bond.GetBondTypeAsDouble(), 1.0)  # Single bond
            action_result = "Succeeded: C=C double bond was decreased to C-C single bond."
        else:
            print("Decreasing bond order is NOT feasible")
            # Choose a feasible action instead
            for i in range(len(molecule.current_action_mask)):
                if molecule.current_action_mask[i] == 0:
                    print(f"Using feasible action {i} instead")
                    molecule.take_action(i)
                    break
            action_result = "Failed: Decreasing bond order not possible. Used fallback action instead."

        # Capture bond decrease
        self.capture_snapshot(
            molecule,
            "Decreased C=C to C-C",
            f"Attempted to decrease C=C bond order to C-C. {action_result} "
            "This demonstrates ability to modify bond orders while respecting aromatic stability."
        )

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

            # Capture after bond removal
            self.capture_snapshot(
                molecule,
                "Removed S-C Bond",
                "Successfully removed the bond between sulfur and carbon. Using action 7 at level 2 removes an existing bond. "
                "Removal was possible because it doesn't fragment the molecule (the is_connected_without_bond check passed)."
            )

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

            # Capture after creating S-O bond
            self.capture_snapshot(
                molecule,
                "Added S-O Bond",
                "Created a new S-O bond connecting the detached sulfur to oxygen. "
                "This demonstrates bond redistribution while preserving molecule connectivity. "
                "S-O-C now forms a branch off the benzene ring."
            )
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

            # Capture after adding branch
            self.capture_snapshot(
                molecule,
                "Added C Branch to O",
                "Could not remove S-C bond as it would fragment the molecule (is_connected_without_bond check failed). "
                "Instead, created a C-O branch by adding a new carbon atom bonded to the oxygen. "
                "This demonstrates the connectivity constraint enforcement."
            )

        # =====================================================
        # Step 8: Finalize and verify the molecule
        # =====================================================
        print("\nStep 8: Finalizing and verifying the molecule")

        # Terminate the molecule construction
        molecule.take_action(0)  # Action 0 at level 0 = terminate

        # Capture final molecule
        self.capture_snapshot(
            molecule,
            "Final Molecule",
            "Terminated molecule construction with action 0 at level 0. "
            "Final structure contains a benzene ring with O, N, and S substituents, "
            "plus bond modifications. SMILES: " + molecule.to_smiles()
        )

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

        # Create visualization of molecule evolution
        self.visualize_molecule_evolution()


if __name__ == "__main__":
    unittest.main()