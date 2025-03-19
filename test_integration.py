import numpy as np
import pytest
from molecule_design import MoleculeDesign
from rdkit import Chem

# Create a dummy configuration for testing.
# This minimal configuration defines three atom types ("C", "N", "O") with typical valences.
class DummyConfig:
    def __init__(self):
        self.atom_vocabulary = {
            "C": {"atomic_number": 6, "valence": 4, "allowed": True},
            "N": {"atomic_number": 7, "valence": 3, "allowed": True},
            "O": {"atomic_number": 8, "valence": 2, "allowed": True}
        }
        # Allow up to 4 real atoms (plus the virtual atom)
        self.max_num_atoms = 4
        self.start_c_chain_max_len = 3

def create_dummy_config():
    return DummyConfig()

@pytest.fixture
def config():
    return create_dummy_config()

@pytest.fixture
def mol(config):
    # Start with an initial atom A (using "C", index 1)
    return MoleculeDesign(config, initial_atom=1)

def test_bond_addition_and_reduction(mol):
    """
    Integration test scenario:

    1. Create molecule with initial atom A.
    2. Level 0: Add atom B (new atom creation). This sets B as the base atom.
    3. Level 1: Add atom C (new atom creation); then at level 2,
       select an increase action to establish a bond between B and C with order 2.
    4. Form a cycle: add an additional bond connecting A and C with bond order 1.
    5. Modify the bond between B and C: reduce its order from 2 to 1.
    6. Verify that bond orders and connectivity are as expected.
    """

    # ---------------------------
    # Step 1: Initial molecule already has A.
    # At this point, mol.atoms = [0, A] (index 0 is virtual).

    # ---------------------------
    # Step 2: Level 0 – Add atom B.
    # Action index 1 is chosen (0 is termination; indices >=1 are new atom actions).
    mol.take_action(1)
    # Now, mol.atoms should be [0, A, B] and base_atom_idx is set to B (index 2).

    # ---------------------------
    # Step 3: Level 1 – Add atom C.
    # Choose new atom creation for C in level 1.
    mol.take_action(1)
    # mol.atoms becomes [0, A, B, C]; last_created_atom_idx is set to C (index 3).

    # Level 2 – Increase bond order between base (B, index 2) and new atom (C, index 3).
    # In level 2, if action < maximum_bond_order then the bond order becomes action + 1.
    # Choose action = 1, which sets bond order = 1 + 1 = 2.
    mol.take_action(1)
    # Verify that bond between B and C is now order 2.
    assert mol.bonds[2, 3] == 2, f"Expected bond order 2 between B and C, got {mol.bonds[2, 3]}"

    # ---------------------------
    # Step 4: Form a cycle by adding an extra bond between A and C with bond order 1.
    # To do this, we force a level 1 modification that selects an existing bond addition.
    # We want to add a bond connecting A (index 1) and C (index 3).
    # Manually set the current_action_level to 1, and change base_atom_idx to A.
    mol.current_action_level = 1
    mol.base_atom_idx = 1  # A is at index 1.
    # Remove any existing last_created_atom_idx so that in block2 the fallback computes candidate.
    if hasattr(mol, 'last_created_atom_idx'):
        del mol.last_created_atom_idx
    # Set history so that the branch for existing bond modification is triggered.
    # (In our level1 code, if history[-1] < pick_existing_atoms_start_action_idx_lvl_0, candidate is computed from history.)
    mol.history = [mol.pick_existing_atoms_start_action_idx_lvl_0]
    # Calling take_action with an action equal to new_atom_action_count will fall in block2.
    new_atom_action_count = len(mol.vocabulary_atom_idcs)  # e.g., 3 for "C", "N", "O"
    mol.take_action(new_atom_action_count)
    # Level 2 – Now, add a bond from base (A, index 1) to candidate.
    # Choose an increase action with action = 0, which will set bond order = 1.
    mol.take_action(0)
    # Check that the bond between A and C is now created with order 1.
    assert mol.bonds[1, 3] == 1, f"Expected bond order 1 between A and C, got {mol.bonds[1, 3]}"

    # ---------------------------
    # Step 5: Modify the bond between B and C: reduce its order from 2 to 1.
    # To do this, we want to select the B–C bond for modification.
    # Reset the action level for modification by setting current_action_level to 1.
    mol.current_action_level = 1
    # Set base_atom_idx back to B (index 2) for the modification.
    mol.base_atom_idx = 2  # B is at index 2.
    # Ensure that last_created_atom_idx points to C (index 3) so that block2 selects the B–C bond.
    mol.last_created_atom_idx = 3
    # Simulate history so that block2 is used.
    mol.history = [mol.pick_existing_atoms_start_action_idx_lvl_0]
    # Call take_action with a value that falls into block2.
    mol.take_action(new_atom_action_count)
    # At level 2, we now want to perform a bond reduction.
    # In level 2, reduction actions are those with action >= maximum_bond_order.
    # Since the current bond order between B and C is 2, reduction by 1 is achieved by choosing action = maximum_bond_order
    mol.take_action(mol.maximum_bond_order)
    # Verify that the bond between B and C has been reduced to 1.
    assert mol.bonds[2, 3] == 1, f"Expected bond order 1 between B and C after reduction, got {mol.bonds[2, 3]}"

    # ---------------------------
    # Final Checks:
    # Verify that virtual bonds remain intact for all real atoms.
    for i in range(1, len(mol.atoms)):
        assert mol.bonds[0, i] == mol.virtual_bond_idx, f"Virtual bond missing at row 0, column {i}."
        assert mol.bonds[i, 0] == mol.virtual_bond_idx, f"Virtual bond missing at row {i}, column 0."
    # Ensure that all real atoms are connected (at least one nonvirtual bond exists for each).
    real_atom_indices = range(1, len(mol.atoms))
    for i in real_atom_indices:
        connected = any(mol.bonds[i, j] > 0 for j in real_atom_indices if j != i)
        assert connected, f"Atom at index {i} is not connected to any other real atom."