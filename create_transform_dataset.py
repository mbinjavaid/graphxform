"""
Generate molecular transformation sequences between pairs of molecules using a hybrid
MCS + Edit Distance approach to create minimal edit paths.
"""
import time
import pickle
# import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS
from config import MoleculeConfig
from molecule_design import MoleculeDesign
from typing import Optional, Tuple, List, Dict, Set
from tqdm import tqdm
import random
import os

RDLogger.DisableLog('rdApp.*')  # Disable RDKit logging


def verify_transformation(config: MoleculeConfig,
                          source_smiles: str,
                          target_smiles: str,
                          actions: List[int]) -> Tuple[bool, str, str]:
    """
    Verify that the action sequence transforms the source molecule into the target.

    Args:
        config: Molecule configuration
        source_smiles: Source molecule SMILES
        target_smiles: Target molecule SMILES
        actions: List of transformation actions

    Returns:
        Tuple of (success: bool, result_smiles: str, debug_info: str)
    """
    # Standardize the target SMILES for comparison
    target_mol = Chem.MolFromSmiles(target_smiles)
    standard_target_smiles = Chem.MolToSmiles(target_mol, isomericSmiles=True, canonical=True)

    debug_info = ""

    try:
        # Initialize molecule design with the source molecule
        mol_design = MoleculeDesign.from_smiles(config, source_smiles, do_finish=False)

        # Apply each action in sequence
        for i, action in enumerate(actions):
            try:
                # In MoleculeDesign, take_action doesn't return a result object,
                # it modifies the molecule directly
                mol_design.take_action(action)

                # Check if the molecule became infeasible after this action
                if mol_design.infeasibility_flag:
                    return False, f"Molecule became infeasible at action {i}: {action}", debug_info

                # Log current action and level for debugging
                current_level = mol_design.current_action_level
                debug_info += f"Action {i}: {action} (Level {current_level})\n"

                # Check if terminated
                if mol_design.synthesis_done:
                    debug_info += "Synthesis terminated\n"
                    break

            except Exception as e:
                return False, f"Failed at action {i}: {action} - {str(e)}", debug_info

        # Get the resulting molecule
        if not mol_design.synthesis_done:
            # Force termination if the action sequence didn't terminate
            try:
                mol_design.take_action(0)  # 0 is the termination action
                debug_info += "Forced termination\n"
            except:
                return False, "Failed to terminate synthesis", debug_info

        result_smiles = mol_design.to_smiles()
        result_mol = Chem.MolFromSmiles(result_smiles)

        if result_mol is None:
            return False, "Invalid resulting molecule", debug_info

        # Standardize the result SMILES
        standard_result_smiles = Chem.MolToSmiles(result_mol, isomericSmiles=True, canonical=True)

        # Compare standardized SMILES
        success = standard_result_smiles == standard_target_smiles
        if success:
            return True, standard_result_smiles, debug_info
        else:
            return False, f"SMILES mismatch\nExpected: {standard_target_smiles}\nGot: {standard_result_smiles}", debug_info

    except Exception as e:
        return False, f"Exception during verification: {str(e)}", debug_info


def generate_molecule_pairs(smiles_list: List[str],
                            min_similarity: float = 0.4,
                            max_similarity: float = 0.8,
                            max_atoms: int = 50,  # Added parameter for max atom count
                            cache_dir: str = "./data/cache") -> List[Tuple[str, str]]:
    """
    Generate pairs of molecules with similarity within the specified range.

    Args:
        smiles_list: List of valid SMILES strings
        min_similarity: Minimum Tanimoto similarity
        max_similarity: Maximum Tanimoto similarity
        max_atoms: Maximum number of atoms allowed in each molecule
        cache_dir: Directory to store cached fingerprints and molecules

    Returns:
        List of (smiles_a, smiles_b) pairs
    """
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    import hashlib

    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Generate a hash of the SMILES list and parameters to use as a unique identifier
    # Include max_atoms in the hash to ensure different caches for different atom limits
    params_str = f"{min_similarity}_{max_similarity}_{max_atoms}"
    cache_id = hashlib.md5((''.join(smiles_list[:100]) + params_str).encode()).hexdigest()
    fingerprints_cache_path = os.path.join(cache_dir, f"fingerprints_{cache_id}.pkl")

    # Check if we have cached fingerprints and valid molecules
    if os.path.exists(fingerprints_cache_path):
        print(f"Loading cached fingerprints and molecules from {fingerprints_cache_path}")
        with open(fingerprints_cache_path, "rb") as f:
            cache_data = pickle.load(f)
            valid_mols = cache_data["valid_mols"]
            fingerprints = cache_data["fingerprints"]
            print(f"Loaded {len(valid_mols)} molecules and fingerprints from cache")
    else:
        # Convert SMILES to molecules and compute Morgan fingerprints
        print(f"Converting SMILES to molecules and generating fingerprints (max atoms: {max_atoms})...")
        valid_mols = []
        fingerprints = []
        skipped_count = 0

        for smiles in tqdm(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Check if molecule has at most max_atoms
                if mol.GetNumAtoms() <= max_atoms:
                    valid_mols.append((mol, smiles))
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    fingerprints.append(fp)
                else:
                    skipped_count += 1

        print(f"Filtered out {skipped_count} molecules with more than {max_atoms} atoms")
        print(f"Retained {len(valid_mols)} valid molecules")

        # Save to cache
        print(f"Saving fingerprints and molecules to cache: {fingerprints_cache_path}")
        cache_data = {
            "valid_mols": valid_mols,
            "fingerprints": fingerprints
        }
        with open(fingerprints_cache_path, "wb") as f:
            pickle.dump(cache_data, f)

    # Calculate number of pairs to generate (half the dataset size)
    num_pairs = len(valid_mols) // 2
    print(f"Will generate {num_pairs} molecule pairs (half of dataset size)")

    # PRE-FILTER BY SIZE: Insert the size grouping code here
    size_groups_cache_path = os.path.join(cache_dir, f"size_groups_{cache_id}.pkl")

    # Check if we have cached size groups
    if os.path.exists(size_groups_cache_path):
        print(f"Loading cached size groups from {size_groups_cache_path}")
        with open(size_groups_cache_path, "rb") as f:
            cache_data = pickle.load(f)
            size_groups = cache_data["size_groups"]
            compatible_indices = cache_data["compatible_indices"]
    else:
        print("Pre-filtering molecules by size compatibility...")
        size_groups = {}
        for i, (mol, _) in enumerate(valid_mols):
            size = mol.GetNumAtoms()
            size_groups.setdefault(size, []).append(i)

        # Create a lookup for compatible size ranges
        compatible_indices = {}
        for size, indices in size_groups.items():
            compatible = []
            for potential_size in range(int(size / 2), int(size * 2) + 1):
                if potential_size in size_groups:
                    compatible.extend(size_groups[potential_size])
            compatible_indices[size] = compatible

        # Save to cache
        cache_data = {
            "size_groups": size_groups,
            "compatible_indices": compatible_indices
        }
        with open(size_groups_cache_path, "wb") as f:
            pickle.dump(cache_data, f)

    # Use an efficient approach to select molecule pairs
    pairs = []
    print(f"Selecting molecule pairs with similarity between {min_similarity} and {max_similarity}...")

    # Create pairs based on similarity using size-compatible molecules
    attempts = 0
    max_attempts = num_pairs * 10

    with tqdm(total=num_pairs) as pbar:
        while len(pairs) < num_pairs and attempts < max_attempts:
            # Choose a random molecule
            i = random.randrange(len(valid_mols))
            mol_a, smiles_a = valid_mols[i]
            size_a = mol_a.GetNumAtoms()

            # Only consider size-compatible molecules
            if size_a in compatible_indices and compatible_indices[size_a]:
                # Pick a random compatible molecule
                j = random.choice(compatible_indices[size_a])

                if i != j:
                    sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])

                    if min_similarity <= sim <= max_similarity:
                        mol_b, smiles_b = valid_mols[j]
                        pairs.append((smiles_a, smiles_b))
                        pbar.update(1)

            attempts += 1

    print(f"Generated {len(pairs)} pairs after {attempts} attempts")
    return pairs


def get_mcs_mapping(mol_a: Chem.Mol, mol_b: Chem.Mol, timeout: int = 5) -> Dict[int, int]:
    """
    Get atom mapping between molecules based on Maximum Common Substructure.
    Uses a fast approach with relaxed constraints.

    Args:
        mol_a: First molecule
        mol_b: Second molecule
        timeout: Maximum time in seconds for MCS calculation (default: 5)

    Returns:
        Dictionary mapping atom indices from mol_a to mol_b
    """
    try:
        # Try fast MCS with relaxed parameters
        mcs_result = rdFMCS.FindMCS(
            [mol_a, mol_b],
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
            completeRingsOnly=True,  # Changed from True - major speedup
            matchValences=True,
            ringMatchesRingOnly=False,  # Changed from True - major speedup
            matchChiralTag=True,
            timeout=timeout  # Shortened timeout (5 seconds)
        )

        # If no MCS found or empty result, use fallback method
        if mcs_result.numAtoms == 0:
            return get_morgan_fallback_mapping(mol_a, mol_b)

        # Get the common substructure pattern
        mcs_pattern = Chem.MolFromSmarts(mcs_result.smartsString)

        # Get atom mappings for both molecules
        match_a = mol_a.GetSubstructMatch(mcs_pattern)
        match_b = mol_b.GetSubstructMatch(mcs_pattern)

        # Create mapping dictionary
        return {match_a[i]: match_b[i] for i in range(len(match_a))}
    except Exception as e:
        print(f"MCS calculation error: {e}")
        return get_morgan_fallback_mapping(mol_a, mol_b)


def get_morgan_fallback_mapping(mol_a: Chem.Mol, mol_b: Chem.Mol) -> Dict[int, int]:
    """
    Fallback mapping strategy using Morgan fingerprints.
    Much more accurate than simple element mapping.

    Args:
        mol_a: First molecule
        mol_b: Second molecule

    Returns:
        Dictionary mapping atom indices from mol_a to mol_b
    """
    from rdkit.Chem import AllChem
    from rdkit import DataStructs

    # Get atom-level Morgan fingerprints
    fps_a = {}
    for atom in mol_a.GetAtoms():
        idx = atom.GetIdx()
        env = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=1024, fromAtoms=[idx])
        fps_a[idx] = (atom.GetSymbol(), env)

    fps_b = {}
    for atom in mol_b.GetAtoms():
        idx = atom.GetIdx()
        env = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=1024, fromAtoms=[idx])
        fps_b[idx] = (atom.GetSymbol(), env)

    # Score all atom pairs by environment similarity
    similarity_scores = []
    for idx_a, (symbol_a, fp_a) in fps_a.items():
        for idx_b, (symbol_b, fp_b) in fps_b.items():
            if symbol_a == symbol_b:  # Only consider atoms of the same element
                sim = DataStructs.TanimotoSimilarity(fp_a, fp_b)
                similarity_scores.append((idx_a, idx_b, sim))

    # Sort by similarity (highest first)
    similarity_scores.sort(key=lambda x: x[2], reverse=True)

    # Create mapping greedily
    mapping = {}
    used_b = set()

    for idx_a, idx_b, sim in similarity_scores:
        if idx_a not in mapping and idx_b not in used_b and sim > 0.3:
            mapping[idx_a] = idx_b
            used_b.add(idx_b)

    return mapping


def optimize_mapping(mol_a: Chem.Mol, mol_b: Chem.Mol,
                    initial_mapping: Dict[int, int]) -> Dict[int, int]:
    """
    Optimize atom mapping to minimize edit operations.

    Args:
        mol_a: First molecule
        mol_b: Second molecule
        initial_mapping: Initial mapping from MCS

    Returns:
        Optimized mapping
    """
    from rdkit.Chem import rdmolops

    # Start with the initial mapping
    optimized = initial_mapping.copy()

    # Get unmatched atoms from both molecules
    mapped_a = set(optimized.keys())
    mapped_b = set(optimized.values())

    unmatched_a = set(range(mol_a.GetNumAtoms())) - mapped_a
    unmatched_b = set(range(mol_b.GetNumAtoms())) - mapped_b

    # Try to map similar unmatched atoms based on environment similarity
    if unmatched_a and unmatched_b:
        # Get Morgan fingerprints for each atom
        fp_a = {}
        for atom_idx in unmatched_a:
            env = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol_a, 2, nBits=1024, fromAtoms=[atom_idx]
            )
            fp_a[atom_idx] = env

        fp_b = {}
        for atom_idx in unmatched_b:
            env = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol_b, 2, nBits=1024, fromAtoms=[atom_idx]
            )
            fp_b[atom_idx] = env

        # Find best additional mappings
        from rdkit import DataStructs

        # Score potential new mappings by atom similarity
        potential_mappings = []
        for a_idx in unmatched_a:
            atom_a = mol_a.GetAtomWithIdx(a_idx)
            for b_idx in unmatched_b:
                atom_b = mol_b.GetAtomWithIdx(b_idx)

                # If atoms have same element, prioritize them
                if atom_a.GetSymbol() == atom_b.GetSymbol():
                    # Calculate environment similarity
                    similarity = DataStructs.TanimotoSimilarity(fp_a[a_idx], fp_b[b_idx])
                    potential_mappings.append((a_idx, b_idx, similarity))

        # Sort by similarity (highest first)
        potential_mappings.sort(key=lambda x: x[2], reverse=True)

        # Add best mappings
        for a_idx, b_idx, sim in potential_mappings:
            if a_idx in unmatched_a and b_idx in unmatched_b and sim > 0.5:
                optimized[a_idx] = b_idx
                unmatched_a.remove(a_idx)
                unmatched_b.remove(b_idx)

    return optimized


def get_atom_type_index(config: MoleculeConfig, atom: Chem.Atom) -> int:
    """
    Get the vocabulary index for an atom by constructing its symbol key.

    Args:
        config: Molecule configuration
        atom: RDKit atom

    Returns:
        Index in vocabulary (0-indexed)
    """
    # Get the base symbol
    symbol = atom.GetSymbol()

    # Add charge suffix if needed
    formal_charge = atom.GetFormalCharge()
    if formal_charge > 0:
        symbol += "+"
    elif formal_charge < 0:
        symbol += "-"

    # Add chirality suffix if needed
    chiral_tag = atom.GetChiralTag()
    if chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
        symbol += "@"
    elif chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
        symbol += "@@"

    # Look up in vocabulary directly
    atom_names = list(config.atom_vocabulary.keys())
    if symbol in atom_names:
        return atom_names.index(symbol)
    else:
        raise ValueError(f"Atom type {symbol} not found in vocabulary, please check the config.")


def generate_transformation_actions(config: MoleculeConfig, mol_a: Chem.Mol, mol_b: Chem.Mol) -> List[int]:
    """
    Generate a minimal sequence of actions to transform molecule A into molecule B.

    Args:
        config: Molecule configuration
        mol_a: Source molecule
        mol_b: Target molecule

    Returns:
        List of action indices corresponding to our action space
    """
    # Get optimized mapping between molecules
    initial_mapping = get_mcs_mapping(mol_a, mol_b)
    mapping = optimize_mapping(mol_a, mol_b, initial_mapping)

    # Start with molecule A
    mol_design = MoleculeDesign.from_rdkit_mol(config, mol_a, Chem.MolToSmiles(mol_a), do_finish=False)

    # Track generated actions
    actions = []

    # Create reverse mapping for lookup efficiency
    reverse_mapping = {v: k for k, v in mapping.items()}

    # Track which atoms have been processed
    processed_atoms_a = set()

    # Get vocab size for action indices
    vocab_size = len(config.atom_vocabulary)

    # 1. Handle atom replacements (highest priority)
    for atom_idx_a, atom_idx_b in mapping.items():
        atom_a = mol_a.GetAtomWithIdx(atom_idx_a)
        atom_b = mol_b.GetAtomWithIdx(atom_idx_b)

        # Skip if atoms are already the same
        if (atom_a.GetSymbol() == atom_b.GetSymbol() and
            atom_a.GetFormalCharge() == atom_b.GetFormalCharge() and
            atom_a.GetChiralTag() == atom_b.GetChiralTag()):
            processed_atoms_a.add(atom_idx_a)
            continue

        # We need to replace this atom
        # Level 0: Select atom (using internal index)
        actions.append(atom_idx_a + 1)  # +1 for virtual atom offset

        # Level 1: Select replace action (V+N)
        replace_action = vocab_size + mol_a.GetNumAtoms()
        actions.append(replace_action)

        # Determine the new atom type index in our vocabulary
        new_atom_type = get_atom_type_index(config, atom_b)

        # Level 2: Select new atom type (directly using vocabulary index)
        actions.append(new_atom_type)

        processed_atoms_a.add(atom_idx_a)

    # 2. Handle bond modifications between mapped atoms
    bond_changes = identify_bond_changes(mol_a, mol_b, mapping)

    # Process bond changes in an efficient order: modifications, additions, removals
    for bond_type, changes in [("modify", bond_changes["modify"]),
                              ("add", bond_changes["add"]),
                              ("remove", bond_changes["remove"])]:
        for a_idx_1, a_idx_2, new_order in changes:
            # Level 0: Select first atom
            actions.append(a_idx_1 + 1)  # +1 for virtual atom offset

            # Level 1: Select second atom
            actions.append(vocab_size + a_idx_2)  # V + atom_idx for existing atom

            if bond_type == "remove":
                # Level 2: Remove bond
                actions.append(vocab_size + 6)  # V + 6 = remove bond

                # Note: In a real implementation, we would need to check if this causes
                # fragmentation and handle Level 3 actions accordingly. This would require
                # actually executing the action and checking the result.
                # For now, we'll assume "keep both fragments" (action 2) if removal happens
                actions.append(2)  # Level 3: Keep both fragments
            else:
                # Level 2: Set bond order (V + order - 1)
                actions.append(vocab_size + (new_order - 1))

    # 3. Handle atom additions (atoms in B not mapped from A)
    unmapped_atoms_b = set(range(mol_b.GetNumAtoms())) - set(mapping.values())

    # Process atom additions - this is complex and needs to be done in the right order
    if unmapped_atoms_b:
        # First, build a graph of unmapped atoms in molecule B
        unmapped_graph = {}
        for atom_idx in unmapped_atoms_b:
            unmapped_graph[atom_idx] = []
            atom = mol_b.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                bond = mol_b.GetBondBetweenAtoms(atom_idx, neighbor_idx)
                unmapped_graph[atom_idx].append((neighbor_idx, int(bond.GetBondTypeAsDouble())))

        # We need to add atoms in a specific order - always connecting to either:
        # 1. An atom that's already in the current molecule (mapped from A)
        # 2. An atom that we've already added in this process

        # Keep track of atoms available for bonding
        available_atoms = set(mapping.values())  # Start with mapped atoms
        added_atoms = {}  # Maps B atom indices to their corresponding index in the current molecule

        # Add atoms until we've processed all unmapped atoms
        while unmapped_atoms_b:
            # Find an unmapped atom that connects to an available atom
            for atom_b_idx in list(unmapped_atoms_b):
                # Check if this atom connects to any available atom
                connects_to_available = False
                connection_point = None
                connection_order = None

                for neighbor_idx, bond_order in unmapped_graph[atom_b_idx]:
                    if neighbor_idx in available_atoms:
                        connects_to_available = True
                        connection_point = neighbor_idx
                        connection_order = bond_order
                        break

                if connects_to_available:
                    # This atom can be added next
                    # Determine the atom's type
                    atom_b = mol_b.GetAtomWithIdx(atom_b_idx)
                    atom_type_idx = get_atom_type_index(config, atom_b)

                    # Figure out which atom in our current molecule corresponds to the connection point
                    if connection_point in mapping.values():
                        # It's a mapped atom from A
                        attachment_idx = next(k for k, v in mapping.items() if v == connection_point)
                    else:
                        # It's an atom we added previously
                        attachment_idx = added_atoms[connection_point]

                    # Level 0: Select attachment point
                    actions.append(attachment_idx + 1)  # +1 for virtual atom offset

                    # Level 1: Create new atom
                    actions.append(atom_type_idx)  # Directly use vocabulary index

                    # Level 2: Set bond order
                    actions.append(vocab_size + (connection_order - 1))  # V + (order - 1)

                    # Update tracking
                    # The new atom's index in the current molecule will be the current size
                    new_atom_idx_in_mol = mol_a.GetNumAtoms() + len(added_atoms)
                    added_atoms[atom_b_idx] = new_atom_idx_in_mol
                    available_atoms.add(atom_b_idx)
                    unmapped_atoms_b.remove(atom_b_idx)

                    # Need to break since we've modified unmapped_atoms_b
                    break
            else:
                # If we get here, we couldn't find any unmapped atoms that connect to available atoms
                # This indicates a problem with the connectivity - should never happen with valid pairs
                raise ValueError("Cannot find a valid atom addition sequence - disconnected atoms")

        # Now we need to add any remaining bonds between newly added atoms
        # This is similar to the bond addition logic above, but for newly added atoms
        for atom_b_idx, mol_idx in added_atoms.items():
            for neighbor_idx, bond_order in unmapped_graph[atom_b_idx]:
                # Skip neighbors that are mapped (we already added these bonds during atom addition)
                if neighbor_idx in mapping.values():
                    continue

                # Skip neighbors with lower indices (avoid processing bonds twice)
                if neighbor_idx in added_atoms and neighbor_idx < atom_b_idx:
                    continue

                # Add bond between newly added atoms
                neighbor_mol_idx = added_atoms[neighbor_idx]

                # Level 0: Select first atom
                actions.append(mol_idx + 1)  # +1 for virtual atom offset

                # Level 1: Select second atom
                actions.append(vocab_size + neighbor_mol_idx)  # V + atom_idx

                # Level 2: Set bond order
                actions.append(vocab_size + (bond_order - 1))  # V + (order - 1)

    return actions

def identify_bond_changes(mol_a: Chem.Mol, mol_b: Chem.Mol,
                         mapping: Dict[int, int]) -> Dict[str, List]:
    """
    Identify bond changes needed between molecules.

    Args:
        mol_a: Source molecule
        mol_b: Target molecule
        mapping: Atom mapping from A to B

    Returns:
        Dictionary with lists of bond modifications/additions/removals
    """
    # Create reverse mapping for convenience
    reverse_mapping = {v: k for k, v in mapping.items()}

    # Track changes
    changes = {
        "modify": [],  # (atom_idx_1, atom_idx_2, new_order)
        "add": [],     # (atom_idx_1, atom_idx_2, new_order)
        "remove": []   # (atom_idx_1, atom_idx_2, None)
    }

    # Track bonds in molecule B
    bonds_b = set()
    for bond in mol_b.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        order = int(bond.GetBondTypeAsDouble())
        bonds_b.add((min(i, j), max(i, j), order))

    # Check each bond in molecule A
    for bond in mol_a.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        order_a = int(bond.GetBondTypeAsDouble())

        # Only process if both atoms are mapped
        if i in mapping and j in mapping:
            i_b = mapping[i]
            j_b = mapping[j]

            # Ensure consistent ordering
            i_b, j_b = min(i_b, j_b), max(i_b, j_b)

            # Check if bond exists in B
            bond_exists = False
            for b_i, b_j, order_b in bonds_b:
                if b_i == i_b and b_j == j_b:
                    bond_exists = True
                    # If bond exists but with different order, modify it
                    if order_a != order_b:
                        changes["modify"].append((i, j, order_b))
                    # Remove from B set to mark as processed
                    bonds_b.remove((b_i, b_j, order_b))
                    break

            # If bond doesn't exist in B, remove it
            if not bond_exists:
                changes["remove"].append((i, j, None))

    # Remaining bonds in B need to be added
    for i_b, j_b, order in bonds_b:
        # Only process if both atoms are mapped from A
        if i_b in reverse_mapping and j_b in reverse_mapping:
            i_a = reverse_mapping[i_b]
            j_a = reverse_mapping[j_b]
            changes["add"].append((i_a, j_a, order))

    return changes


def create_transform_dataset(config: MoleculeConfig,
                             smiles_path: str,
                             destination_path: str,
                             cache_dir: str = "./data/cache"):
    """
    Create a dataset of molecular transformations.

    Args:
        config: Molecule configuration
        smiles_path: Path to SMILES file
        destination_path: Where to save the dataset
        cache_dir: Directory to store cached data
    """
    start_time = time.perf_counter()

    # Load SMILES
    print(f"Loading SMILES from {smiles_path}")
    with open(smiles_path) as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    # Generate molecule pairs with appropriate similarity
    # Number of pairs is automatically set to half the dataset size
    pairs = generate_molecule_pairs(smiles_list, max_atoms=50, cache_dir=cache_dir)

    pairs = pairs[:10]  # Test with just 10 pairs
    print(f"TESTING WITH ONLY 10 MOLECULE PAIRS")

    # Generate transformation sequences
    transform_sequences = []
    print(f"Generating transformation sequences for {len(pairs)} molecule pairs")

    # Statistics tracking
    total_pairs = len(pairs)
    success_count = 0
    timeout_count = 0
    error_count = 0
    verification_failure_count = 0

    for smiles_a, smiles_b in tqdm(pairs):
        # Convert to RDKit molecules
        mol_a = Chem.MolFromSmiles(smiles_a)
        mol_b = Chem.MolFromSmiles(smiles_b)

        if mol_a and mol_b:
            try:
                # Print the molecules being processed
                print(f"\nProcessing pair: {smiles_a} → {smiles_b}")
                print(f"Molecule A: {mol_a.GetNumAtoms()} atoms, Molecule B: {mol_b.GetNumAtoms()} atoms")

                # Time the MCS computation
                mcs_start = time.perf_counter()
                print("Finding MCS mapping...")
                initial_mapping = get_mcs_mapping(mol_a, mol_b)
                print(f"Found initial mapping with {len(initial_mapping)} atoms mapped")

                print("Optimizing mapping...")
                mapping = optimize_mapping(mol_a, mol_b, initial_mapping)
                print(f"Optimized mapping has {len(mapping)} atoms mapped")

                print("Generating transformation actions...")
                actions = generate_transformation_actions(config, mol_a, mol_b)

                mcs_time = time.perf_counter() - mcs_start
                print(f"MCS and transformation generation took {mcs_time:.2f} seconds")
                print(f"Action sequence length: {len(actions)}")
                print("-" * 80)

                # Verify the transformation
                print("Verifying transformation...")
                verification_success, result_info, debug_details = verify_transformation(
                    config, smiles_a, smiles_b, actions
                )

                if verification_success:
                    print("Verification successful - actions correctly transform source to target")

                    # Store the sequence
                    sequence = {
                        'source_smiles': smiles_a,
                        'target_smiles': smiles_b,
                        'action_seq': actions,
                        'obj': 0.0  # Placeholder
                    }
                    transform_sequences.append(sequence)
                    success_count += 1
                else:
                    print(f"Verification failed: {result_info}")
                    print(f"Debug details:\n{debug_details}")
                    verification_failure_count += 1  # Increment failure counter

                print("-" * 80)

            except Exception as e:
                    print(f"\nError processing pair {smiles_a} → {smiles_b}: {e}")
                    if "timeout" in str(e).lower():
                        timeout_count += 1
                    else:
                        error_count += 1
                    print("-" * 80)

    print(f"Generated {len(transform_sequences)} transformation sequences")
    print(f"Generation took {time.perf_counter() - start_time:.2f} seconds")

    print(f"Dataset statistics:")
    print(f"  Total molecule pairs: {total_pairs}")
    print(f"  Successful transformations: {success_count} ({success_count / total_pairs * 100:.1f}%)")
    print(f"  MCS timeouts: {timeout_count} ({timeout_count / total_pairs * 100:.1f}%)")
    print(f"  Other errors: {error_count} ({error_count / total_pairs * 100:.1f}%)")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)

    # Save to file
    with open(destination_path, "wb") as f:
        pickle.dump(transform_sequences, f)
    print(f"Saved transformation dataset to {destination_path}")

if __name__ == "__main__":
    # Setup configuration
    config = MoleculeConfig()

    # Process both train and validation sets
    for datatype in ["train", "valid"]:
        smiles_path = f"./data/chembl/chembl_{datatype}_filtered.smiles"
        destination_path = f"./data/chembl/transform_sequences/chembl_{datatype}_transform.pickle"

        # Create transform dataset - now uses half of dataset size automatically
        create_transform_dataset(config, smiles_path, destination_path)

        # Read and print the destination file
        with open(destination_path, "rb") as f:
            data = pickle.load(f)
            print(f"Loaded {len(data)} sequences from {destination_path}")
            for seq in data[:5]:
                print(seq)
                print("-" * 80)
        print("-" * 80)
    print("All done!")


