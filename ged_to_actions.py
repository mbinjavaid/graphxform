# -*- coding: utf-8 -*-
"""
GED to MoleculeDesign Mapping - Graph-based efficient fragmentation strategy.
Version: 2025-04-18 01:41:50 UTC (Strict Sanitization)
"""
import pickle
import os
import numpy as np
from rdkit.Chem import Draw
from rdkit import Chem, RDLogger
from rdkit.Chem import rdmolfiles, rdmolops, AllChem
from collections import defaultdict, deque
import datetime # For timestamp
from tqdm import tqdm # Added tqdm for main loop

# --- User Info ---
CURRENT_USER = "mbinjavaid"
# --- End User Info ---


# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Configuration
CHECKPOINT_DIR = "./data/chembl/checkpoints"
DEBUG = True
MAX_TRANSFORMATIONS = 100
CHECKPOINT_FREQUENCY = 100 # Lowered for testing, adjust as needed


def log_message(message):
    """Helper function for timestamped logging."""
    if DEBUG:
        timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"[{timestamp}] {message}")

# --- Real Mappings (Derived from config.py and molecule_design.py) ---
ELEMENT_TO_VOCAB = {
    '6': 0,   # C
    '7': 5,   # N
    '8': 8,   # O
    '9': 11,  # F
    '15': 12, # P
    '16': 15, # S
    '17': 20, # Cl
    '35': 21, # Br
    '53': 22, # I
    '?': 0    # Default fallback to Carbon index
}
VOCAB_SIZE = 23 # Total entries in config.atom_vocabulary

RDKIT_BOND_TYPE_TO_ACTION = {
    Chem.BondType.SINGLE:    23, # V+0
    Chem.BondType.DOUBLE:    24, # V+1
    Chem.BondType.TRIPLE:    25, # V+2
    Chem.BondType.QUADRUPLE: 26, # V+3
    Chem.BondType.QUINTUPLE: 27, # V+4
    Chem.BondType.HEXTUPLE:  28, # V+5
    Chem.BondType.AROMATIC:  23, # Map AROMATIC to SINGLE action
}
BOND_REMOVE_ACTION = 29 # V+6
# --- End Real Mappings ---


def load_transformation_data(datatype="train"):
    """Load existing transformation data from latest checkpoint."""
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith(f"transformation_data_{datatype}_")]

    if checkpoint_files:
        checkpoint_files.sort(reverse=True)
        latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoint_files[0])
        log_message(f"Loading from checkpoint: {latest_checkpoint}")
        try: # Added try-except for loading
            with open(latest_checkpoint, "rb") as f:
                # Assuming checkpoint stores full transformation dicts including edit paths
                transformation_data = pickle.load(f)
            log_message(f"Loaded {len(transformation_data)} transformations from checkpoint")
            # Return only the raw data needed (smiles, edit_path)
            processed_data = [{'source_smiles': t['source_smiles'],
                               'target_smiles': t['target_smiles'],
                               'edit_path': t['edit_path']}
                              for t in transformation_data if 'edit_path' in t] # Ensure edit_path exists
            return processed_data[:MAX_TRANSFORMATIONS]
        except Exception as e:
            log_message(f"ERROR loading checkpoint {latest_checkpoint}: {e}. Loading from scratch if possible.")
            # Fall through to load from original file if checkpoint fails

    # If no checkpoint or checkpoint failed, load from original (assuming this logic exists elsewhere or is added)
    # Placeholder: Assuming you have a function like `load_raw_data(datatype)`
    # raw_data = load_raw_data(datatype)
    # return raw_data[:MAX_TRANSFORMATIONS]
    log_message(f"No usable transformation data checkpoint found for {datatype}. Need raw data source.")
    return []


def categorize_operations(edit_path):
    """Categorize GED operations by type."""
    log_message("Categorizing GED operations...")
    metadata = None
    clean_edit_path = [] # Use a new list to avoid modifying original

    if not edit_path: # Handle empty edit_path
        log_message("Warning: Empty edit path provided.")
        return {'metadata': None, 'substitutions': [], 'deletions': [],
                'insertions': [], 'edge_operations': [], 'raw_edit_path': []}

    for i, op in enumerate(edit_path):
        if isinstance(op, dict) and op.get('operation') == 'metadata':
            metadata = op
            # Don't modify the original edit_path, just stop iterating here for categorization
            break
        elif isinstance(op, dict): # Ensure op is a dict before proceeding
             clean_edit_path.append(op)
        else:
             log_message(f"Warning: Non-dict item found in edit path at index {i}: {op}. Skipping.")


    substitutions, deletions, insertions, edge_operations = [], [], [], []

    for op in clean_edit_path: # Iterate over the cleaned path
        op_type = op.get('operation')
        if op_type == 'substitute_node': substitutions.append(op)
        elif op_type == 'delete_node': deletions.append(op)
        elif op_type == 'insert_node': insertions.append(op)
        elif op_type in ['insert_edge', 'delete_edge', 'substitute_edge']: edge_operations.append(op)
        else: log_message(f"Warning: Unknown operation type encountered: {op_type}")

    edge_insertions = [op for op in edge_operations if op.get('operation') == 'insert_edge']
    edge_deletions = [op for op in edge_operations if op.get('operation') == 'delete_edge']
    log_message(f"Categorized Operations: {len(substitutions)} subs, {len(deletions)} dels, {len(insertions)} ins, {len(edge_operations)} edge ops ({len(edge_insertions)} insert_edge, {len(edge_deletions)} delete_edge).")

    return {
        'metadata': metadata, 'substitutions': substitutions, 'deletions': deletions,
        'insertions': insertions, 'edge_operations': edge_operations, 'raw_edit_path': edit_path # Keep original path if needed
    }


def build_atom_fate_map(operations, source_mol, target_mol):
    """Build a map of which atoms are kept vs. doomed."""
    log_message("\n=== Building Atom Fate Map ===")
    num_atoms_source = source_mol.GetNumAtoms()
    atom_fate = {i: "unknown" for i in range(num_atoms_source)}
    source_to_target = {}

    for del_op in operations.get('deletions', []):
        source_idx = del_op.get('source_idx')
        element_num_str = str(del_op.get('element', '?')) # Get atomic number as string
        if source_idx is not None and 0 <= source_idx < num_atoms_source:
            atom_fate[source_idx] = "doomed"
            log_message(f"Atom {source_idx} ({element_num_str}) is explicitly deleted")

    for sub_op in operations.get('substitutions', []):
        source_idx = sub_op.get('source_idx')
        target_idx = sub_op.get('target_idx')
        from_element_num_str = str(sub_op.get('from_element', '?'))
        to_element_num_str = str(sub_op.get('to_element', '?'))
        if source_idx is not None and 0 <= source_idx < num_atoms_source:
            atom_fate[source_idx] = "kept"
            if target_idx is not None: source_to_target[source_idx] = target_idx
            log_message(f"Atom {source_idx} -> {target_idx if target_idx is not None else '?'} (substituted {from_element_num_str} -> {to_element_num_str})")

    # Infer remaining atoms as kept
    for i in range(num_atoms_source):
        if atom_fate[i] == "unknown":
            atom_fate[i] = "kept"
            # Find corresponding target index if possible (simple case: not deleted, not substituted)
            # More complex inference might be needed if GED path is minimal
            # For now, just mark as kept
            log_message(f"Atom {i} implicitly kept")


    kept_count = sum(1 for fate in atom_fate.values() if fate == "kept")
    doomed_count = sum(1 for fate in atom_fate.values() if fate == "doomed")
    log_message(f"Atoms: {kept_count} kept, {doomed_count} doomed")

    return {'atom_fate': atom_fate, 'source_to_target': source_to_target}


def build_adjacency_list(mol):
    """Build an adjacency list representation of the molecule."""
    adj = defaultdict(list)
    if mol:
        for bond in mol.GetBonds():
            begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            adj[begin_idx].append(end_idx)
            adj[end_idx].append(begin_idx)
    return adj


def analyze_graph_structure(source_mol, atom_fate, operations):
    """Identifies Kept Components, Doomed Fragments, and Critical Bonds."""
    log_message("\n=== Analyzing Graph Structure (Kept/Doomed) ===")
    adj = build_adjacency_list(source_mol)
    visited, kept_components, doomed_fragments = set(), [], []
    num_atoms = source_mol.GetNumAtoms()

    # Find Kept Components
    for i in range(num_atoms):
        if atom_fate.get(i) == "kept" and i not in visited:
            component, q = set(), deque([i])
            while q:
                u = q.popleft()
                if u in visited or atom_fate.get(u) != "kept": continue
                visited.add(u); component.add(u)
                for v in adj.get(u, []):
                    if v not in visited and atom_fate.get(v) == "kept": q.append(v)
            if component: kept_components.append(component)

    # Find initial Doomed Fragments based on explicit deletions and connectivity
    visited_doomed = set()
    for i in range(num_atoms):
        if atom_fate.get(i) == "doomed" and i not in visited_doomed:
            component, q = set(), deque([i])
            while q:
                u = q.popleft()
                # Only traverse through other doomed atoms
                if u in visited_doomed or atom_fate.get(u) != "doomed": continue
                visited_doomed.add(u); component.add(u)
                for v in adj.get(u, []):
                     # Only add neighbors that are also doomed to the queue
                    if v not in visited_doomed and atom_fate.get(v) == "doomed": q.append(v)
            if component: doomed_fragments.append(component)


    log_message(f"Identified {len(kept_components)} Kept Components (Sizes: {[len(c) for c in kept_components]}).")
    log_message(f"Identified {len(doomed_fragments)} initial Doomed Fragments (Sizes: {[len(f) for f in doomed_fragments]}).")

    # Identify single doomed atoms attached only to kept atoms (form their own fragment)
    critically_attached_single_doomed = set()
    all_doomed_in_fragments = set().union(*doomed_fragments) if doomed_fragments else set()
    for i in range(num_atoms):
        if atom_fate.get(i) == "doomed" and i not in all_doomed_in_fragments:
            neighbors = adj.get(i, [])
            neighbor_fates = [atom_fate.get(n) for n in neighbors]
            # Check if it has neighbors AND all neighbors are 'kept'
            if neighbors and all(fate == "kept" for fate in neighbor_fates):
                critically_attached_single_doomed.add(i)
    if critically_attached_single_doomed:
        log_message(f"Identified {len(critically_attached_single_doomed)} single doomed atoms connected only to Kept components: {critically_attached_single_doomed}")
        for idx in critically_attached_single_doomed: doomed_fragments.append({idx}) # Add as single-atom fragments
        log_message(f"Total Doomed Fragments (including singles): {len(doomed_fragments)}")


    all_kept_atoms = set().union(*kept_components) if kept_components else set()

    # Identify critical bonds (kept <-> doomed) based on atom fate and doomed fragments
    critical_bonds = []
    delete_edge_ops = [op for op in operations.get('edge_operations', []) if op.get('operation') == 'delete_edge']
    for op in delete_edge_ops:
        a1, a2 = op.get('atom1_idx'), op.get('atom2_idx')
        if a1 is None or a2 is None or not (0 <= a1 < num_atoms and 0 <= a2 < num_atoms): continue

        fate1, fate2 = atom_fate.get(a1), atom_fate.get(a2)
        if (fate1 == "kept" and fate2 == "doomed") or (fate1 == "doomed" and fate2 == "kept"):
            kept_atom = a1 if fate1 == "kept" else a2
            doomed_atom = a1 if fate1 == "doomed" else a2
            # Check if the doomed atom belongs to one of the identified doomed fragments
            fragment_involved = next((frag for frag in doomed_fragments if doomed_atom in frag), None)
            kept_component_involved = next((comp for comp in kept_components if kept_atom in comp), None)

            # Only consider it critical if the doomed atom is part of a defined doomed fragment
            if fragment_involved is not None:
                 critical_bonds.append({
                     'bond': tuple(sorted((a1, a2))), 'kept_atom': kept_atom, 'doomed_atom': doomed_atom,
                     'doomed_fragment': fragment_involved, # The specific fragment
                     'kept_component': kept_component_involved, # Can be None if kept atom isolated? Unlikely.
                     'original_op': op
                 })

    log_message(f"Identified {len(critical_bonds)} critical bonds connecting Kept atoms to identified Doomed fragments.")

    # Identify internal kept deletions (kept <-> kept)
    internal_kept_deletions = []
    critical_bond_keys = {cb['bond'] for cb in critical_bonds}
    for op in delete_edge_ops:
         a1, a2 = op.get('atom1_idx'), op.get('atom2_idx')
         if a1 is None or a2 is None or not (0 <= a1 < num_atoms and 0 <= a2 < num_atoms): continue
         bond_key = tuple(sorted((a1, a2)))
         # Ensure both atoms are marked as 'kept' AND this bond wasn't already classified as critical
         if atom_fate.get(a1) == "kept" and atom_fate.get(a2) == "kept" and bond_key not in critical_bond_keys:
             internal_kept_deletions.append(op)

    log_message(f"Identified {len(internal_kept_deletions)} internal bond deletions within Kept components.")

    return {
        'kept_components': kept_components, 'all_kept_atoms': all_kept_atoms, 'doomed_fragments': doomed_fragments,
        'critical_bonds': critical_bonds, 'internal_kept_deletions': internal_kept_deletions
    }


def analyze_fragment(fragment_indices, atom_fate):
    """Analyze a fragment based on atom fates (indices only)."""
    if not hasattr(fragment_indices, '__iter__') or not fragment_indices: # Check if iterable and not empty
        return {"kept_count": 0, "should_discard": True} # Empty fragment should be discarded
    kept_count = sum(1 for idx in fragment_indices if atom_fate.get(idx) == "kept")
    return {"kept_count": kept_count, "should_discard": kept_count == 0}


# ==============================================================================
# START OF MODIFIED analyze_bond_removals_efficient
# ==============================================================================
def analyze_bond_removals_efficient(graph_analysis, source_mol, fate_info, operations):
    """
    Analyzes bond removals. Focuses on the split component for L3 actions.
    Stops analysis for the entire pair if sanitization fails at any step.
    """
    log_message("\n=== Analyzing Bond Removals (Strict Sanitization Strategy) ===")
    critical_bonds = graph_analysis.get('critical_bonds', [])
    internal_kept_deletions = graph_analysis.get('internal_kept_deletions', [])
    atom_fate = fate_info.get('atom_fate', {})
    bond_analyses, processed_bond_keys = [], set()
    working_mol = Chem.Mol(source_mol) # Start with a copy

    log_message(f"\n--- Processing {len(critical_bonds)} Critical Bonds ---")
    for i, cb_info in enumerate(critical_bonds):
        op, bond_key = cb_info['original_op'], cb_info['bond']
        atom1_idx, atom2_idx = bond_key
        kept_atom_idx = cb_info.get('kept_atom')
        original_doomed_atom = cb_info.get('doomed_atom')

        if bond_key in processed_bond_keys or original_doomed_atom is None or kept_atom_idx is None: continue
        log_message(f"\nCritical Bond {i + 1}: atoms {atom1_idx}-{atom2_idx} (Kept: {kept_atom_idx}, Doomed: {original_doomed_atom})")

        bond = working_mol.GetBondBetweenAtoms(atom1_idx, atom2_idx)
        if bond is None:
            atom1_exists = working_mol.GetAtomWithIdx(atom1_idx) is not None
            atom2_exists = working_mol.GetAtomWithIdx(atom2_idx) is not None
            reason = "Unknown"
            if not atom1_exists: reason = f"Atom {atom1_idx} no longer exists"
            elif not atom2_exists: reason = f"Atom {atom2_idx} no longer exists"
            else: reason = "Bond between existing atoms not found"
            log_message(f"  WARNING: Bond {bond_key} not found in working_mol. Reason: {reason}. Skipping this bond op (pair might still be processed).")
            continue # Skip this specific bond, but don't fail the whole pair yet

        # --- Identify Original Component ---
        original_component_indices = None
        prev_global_components_indices = []
        try:
            prev_global_components_indices = Chem.GetMolFrags(working_mol, asMols=False, sanitizeFrags=False)
            for comp in prev_global_components_indices:
                 if kept_atom_idx in comp:
                      original_component_indices = set(comp)
                      log_message(f"  Bond is within original component of size {len(original_component_indices)}")
                      break
        except Exception as e:
            log_message(f"  ERROR: GetMolFrags failed BEFORE critical removal: {e}. Skipping analysis for this transformation pair.")
            return None # FATAL for pair

        if original_component_indices is None:
             log_message(f"  ERROR: Could not find original component containing atom {kept_atom_idx}. Skipping analysis for this transformation pair.")
             return None # FATAL for pair
        # ---

        # --- Simulate removal ---
        rwmol = Chem.RWMol(working_mol)
        rwmol.RemoveBond(atom1_idx, atom2_idx)
        modified_mol = None
        try:
            # kekulize rwmol:
            Chem.Kekulize(rwmol, clearAromaticFlags=True) # Clear aromatic flags to avoid issues
            Chem.SanitizeMol(rwmol) # Attempt sanitization
            modified_mol = Chem.Mol(rwmol) # Convert if sanitization succeeded
        except Exception as e:
            # --- STRICT CHANGE HERE (Critical Bonds) ---
            log_message(f"  FATAL: Sanitization failed after critical bond {bond_key} removal: {e}. Skipping analysis for this transformation pair.")
            return None # Indicate failure for this pair
            # --- END STRICT CHANGE ---

        if modified_mol is None:
             # This case might be reached if SanitizeMol raises error but doesn't return None explicitly? Unlikely but safe check.
             log_message(f"  ERROR: Mol creation failed unexpectedly after removal (bond {bond_key}). Skipping analysis for this transformation pair.")
             return None # FATAL for pair
        # --- End Simulate removal ---


        # --- Analyze fragments AFTER removal ---
        new_global_components_indices = []
        try:
            new_global_components_indices = Chem.GetMolFrags(modified_mol, asMols=False, sanitizeFrags=False)
            log_message(f"  After removal, molecule has {len(new_global_components_indices)} global components (previously {len(prev_global_components_indices)}).")
        except Exception as e:
            log_message(f"  ERROR: GetMolFrags failed AFTER critical removal: {e}. Skipping analysis for this transformation pair.")
            return None # FATAL for pair
        # ---

        # --- Identify Derived Fragments and the Doomed One ---
        derived_components = []
        derived_doomed_component_indices = None
        derived_doomed_component_global_idx = -1

        for global_idx, new_comp_indices_tuple in enumerate(new_global_components_indices):
            new_comp_set = set(new_comp_indices_tuple)
            if new_comp_set.issubset(original_component_indices):
                derived_components.append({'global_idx': global_idx, 'indices': new_comp_set})
                if original_doomed_atom in new_comp_set:
                    derived_doomed_component_indices = new_comp_set
                    derived_doomed_component_global_idx = global_idx
                    log_message(f"  Found derived component {global_idx} (size {len(new_comp_set)}) containing doomed atom {original_doomed_atom}.")

        log_message(f"  Identified {len(derived_components)} components derived from the original split component.")
        # ---

        # --- Determine Level 3 Action Based on Derived Components ---
        env_level3_action = 2
        analysis_type = 'internal_kept_core'
        creates_fragments_flag = len(new_global_components_indices) > len(prev_global_components_indices)

        if derived_doomed_component_indices is not None:
            frag_analysis = analyze_fragment(derived_doomed_component_indices, atom_fate)
            if len(derived_components) == 2 and frag_analysis['should_discard']:
                analysis_type = 'critical_disconnection'
                creates_fragments_flag = True
                other_derived_comp_info = next((dc for dc in derived_components if dc['global_idx'] != derived_doomed_component_global_idx), None)

                if other_derived_comp_info is not None:
                    other_derived_comp_global_idx = other_derived_comp_info['global_idx']
                    if derived_doomed_component_global_idx < other_derived_comp_global_idx:
                         env_level3_action = 1
                         log_message(f"  Clean split & discard verified. Doomed global index ({derived_doomed_component_global_idx}) < Other ({other_derived_comp_global_idx}). Level 3 Env Action: 1 (Keep Other)")
                    else:
                         env_level3_action = 0
                         log_message(f"  Clean split & discard verified. Doomed global index ({derived_doomed_component_global_idx}) > Other ({other_derived_comp_global_idx}). Level 3 Env Action: 0 (Keep Other)")
                else:
                    log_message("  ERROR: Could not find the 'other' derived component despite clean split. Defaulting Level 3: Keep Both (Action 2).")
                    env_level3_action = 2
            else:
                if len(derived_components) != 2:
                     log_message(f"  Split was not clean (1 -> {len(derived_components)} derived components). Defaulting Level 3: Keep Both (Action 2).")
                if not frag_analysis['should_discard']:
                     log_message(f"  Doomed component {derived_doomed_component_global_idx} contains kept atoms ({frag_analysis['kept_count']}). Defaulting Level 3: Keep Both (Action 2).")
                env_level3_action = 2
        else:
             log_message(f"  ERROR: Could not find derived component containing doomed atom {original_doomed_atom}. Defaulting Level 3: Keep Both (Action 2).")
             env_level3_action = 2
        # ---

        # --- Append analysis and update state ---
        bond_analyses.append({
            'bond': bond_key, 'type': analysis_type, 'creates_fragments': creates_fragments_flag,
            'level3_action': env_level3_action, 'original_op': op,
            'kept_atom': kept_atom_idx, 'doomed_atom': original_doomed_atom
        })
        processed_bond_keys.add(bond_key)
        working_mol = modified_mol # Update working mol for next iteration
        # ---

    # --- Processing Internal Kept Core Bonds ---
    log_message(f"\n--- Processing {len(internal_kept_deletions)} initially classified Internal Kept Core Bonds ---")
    for i, op in enumerate(internal_kept_deletions):
        atom1_idx, atom2_idx = op.get('atom1_idx'), op.get('atom2_idx')
        if atom1_idx is None or atom2_idx is None: continue
        bond_key = tuple(sorted((atom1_idx, atom2_idx)))
        if bond_key in processed_bond_keys: continue
        log_message(f"\nInternal Bond {i + 1}: atoms {atom1_idx}-{atom2_idx}")

        bond = working_mol.GetBondBetweenAtoms(atom1_idx, atom2_idx)
        if bond is None:
            atom1_exists = working_mol.GetAtomWithIdx(atom1_idx) is not None
            atom2_exists = working_mol.GetAtomWithIdx(atom2_idx) is not None
            reason = "Unknown"
            if not atom1_exists: reason = f"Atom {atom1_idx} no longer exists"
            elif not atom2_exists: reason = f"Atom {atom2_idx} no longer exists"
            else: reason = "Bond between existing atoms not found"
            log_message(f"  WARNING: Bond {bond_key} not found in working_mol. Reason: {reason}. Skipping this bond op (pair might still be processed).")
            continue # Skip this specific bond

        # --- Identify Original Component ---
        original_component_indices_internal = None
        prev_global_components_indices_internal = []
        try:
            prev_global_components_indices_internal = Chem.GetMolFrags(working_mol, asMols=False, sanitizeFrags=False)
            for comp in prev_global_components_indices_internal:
                 if atom1_idx in comp:
                      original_component_indices_internal = set(comp)
                      break
        except Exception as e:
            log_message(f"  ERROR: GetMolFrags failed BEFORE internal removal: {e}. Skipping analysis for this transformation pair.")
            return None # FATAL for pair
        if original_component_indices_internal is None:
            log_message(f"  ERROR: Could not find original component for internal bond {bond_key}. Skipping analysis for this transformation pair.")
            return None # FATAL for pair
        # ---

        # --- Simulate removal ---
        rwmol_internal = Chem.RWMol(working_mol)
        rwmol_internal.RemoveBond(atom1_idx, atom2_idx)
        modified_mol_internal = None
        try:
            Chem.SanitizeMol(rwmol_internal)
            modified_mol_internal = Chem.Mol(rwmol_internal)
        except Exception as e:
            # --- STRICT CHANGE HERE (Internal Bonds) ---
            log_message(f"  FATAL: Sanitization failed after internal bond {bond_key} removal: {e}. Skipping analysis for this transformation pair.")
            return None # Indicate failure for this pair
            # --- END STRICT CHANGE ---
        if modified_mol_internal is None:
             log_message(f"  ERROR: Mol creation failed unexpectedly after internal removal (bond {bond_key}). Skipping analysis for this transformation pair.")
             return None # FATAL for pair
        # ---

        # --- Analyze fragments AFTER internal removal ---
        new_global_components_indices_internal = []
        try:
            new_global_components_indices_internal = Chem.GetMolFrags(modified_mol_internal, asMols=False, sanitizeFrags=False)
        except Exception as e:
            log_message(f"  ERROR: GetMolFrags failed AFTER internal removal: {e}. Skipping analysis for this transformation pair.")
            return None # FATAL for pair
        # ---

        # --- Check if THIS component split cleanly ---
        derived_components_internal = []
        for global_idx, new_comp_indices_tuple in enumerate(new_global_components_indices_internal):
            new_comp_set = set(new_comp_indices_tuple)
            if new_comp_set.issubset(original_component_indices_internal):
                derived_components_internal.append({'global_idx': global_idx, 'indices': new_comp_set})

        analysis = {'bond': bond_key, 'type': 'internal_kept_core', 'creates_fragments': False, 'level3_action': -1, 'original_op': op}
        if len(derived_components_internal) == 2:
             analysis['creates_fragments'] = True
             analysis['level3_action'] = 2
             log_message(f"  Internal bond removal caused clean split (1 -> 2 derived). Level 3: Keep Both (Action 2).")
        elif len(new_global_components_indices_internal) > len(prev_global_components_indices_internal):
             analysis['creates_fragments'] = True
             analysis['level3_action'] = 2
             log_message(f"  Internal bond removal caused complex fragmentation ({len(prev_global_components_indices_internal)}->{len(new_global_components_indices_internal)} global). Level 3: Keep Both (Action 2).")
        else:
             log_message("  Internal bond removal did not create fragments.")
        # ---

        # --- Append analysis and update state ---
        bond_analyses.append(analysis)
        processed_bond_keys.add(bond_key)
        working_mol = modified_mol_internal # Update working mol
        # ---

    # --- Check for unprocessed deletions ---
    all_delete_ops = [tuple(sorted((op.get('atom1_idx'), op.get('atom2_idx'))))
                      for op in operations.get('edge_operations', [])
                      if op.get('operation') == 'delete_edge' and op.get('atom1_idx') is not None and op.get('atom2_idx') is not None]
    unprocessed_deletions = set(all_delete_ops) - processed_bond_keys
    if unprocessed_deletions:
        log_message(f"\nWARNING: {len(unprocessed_deletions)} delete_edge operations were not processed (likely missing in intermediate states): {unprocessed_deletions}")

    return bond_analyses # Return the list if all removals succeeded
# ==============================================================================
# END OF MODIFIED analyze_bond_removals_efficient
# ==============================================================================


def map_operations_to_action_sequence_efficient(operations, source_mol, graph_analysis, bond_analyses, fate_info):
    """Maps operations to actions with heuristic ordering and multi-pass insertions."""
    log_message("\n=== Mapping Operations to Action Sequences (Efficient Strategy) ===")

    all_kept_atoms = graph_analysis.get('all_kept_atoms', set())
    atom_fate = fate_info.get('atom_fate', {})
    num_atoms_source = source_mol.GetNumAtoms()

    global VOCAB_SIZE, BOND_REMOVE_ACTION
    # Calculate ACTION_REPLACE_ATOM relative to the *initial* number of atoms
    # This assumes the environment action indices for selecting atoms stay fixed
    # relative to the initial state + additions.
    ACTION_REPLACE_ATOM = VOCAB_SIZE + num_atoms_source

    action_sequence = []

    def get_bond_action(rdkit_bond_type_or_val):
        if isinstance(rdkit_bond_type_or_val, (int, float)):
             # Convert numeric bond type (e.g., 1.0, 2.0) to RDKit type
             bond_type_map = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE, 1.5: Chem.BondType.AROMATIC}
             rdkit_bond_type = bond_type_map.get(float(rdkit_bond_type_or_val), Chem.BondType.SINGLE) # Use float for 1.5
        else:
             rdkit_bond_type = rdkit_bond_type_or_val # Assume it's already an RDKit type
        action = RDKIT_BOND_TYPE_TO_ACTION.get(rdkit_bond_type)
        if action is None:
             log_message(f"  WARNING: RDKit bond type {rdkit_bond_type} (from value {rdkit_bond_type_or_val}) not in action map. Defaulting to SINGLE bond action.")
             action = RDKIT_BOND_TYPE_TO_ACTION.get(Chem.BondType.SINGLE)
        return int(action)

    def get_vocab_index(element_atomic_num):
        element_atomic_num_str = str(element_atomic_num)
        fallback_index = ELEMENT_TO_VOCAB.get('?') # Should be 0 for Carbon
        index = ELEMENT_TO_VOCAB.get(element_atomic_num_str, fallback_index)
        if index == fallback_index and element_atomic_num_str not in ELEMENT_TO_VOCAB:
             log_message(f"  WARNING: Atomic number '{element_atomic_num_str}' not in vocab mapping. Using fallback index {fallback_index}.")
        return int(index)

    def generate_add_bond_actions(atom_env_idx1, atom_env_idx2, bond_type_val):
        # Ensure indices are 0-based for action calculation
        level0_action = int(atom_env_idx1) + 1 # Action is 1-based index
        level1_action = VOCAB_SIZE + int(atom_env_idx2) # Action uses 0-based index
        level2_action = get_bond_action(bond_type_val)
        action_tuple = (level0_action, level1_action, level2_action)
        log_message(f"    Add/Set Bond Action Tuple: {action_tuple} (env_idx {atom_env_idx1} - env_idx {atom_env_idx2}, Type {bond_type_val})")
        return action_tuple

    # --- 1. Doomed Fragment Removals (Successful Discards) ---
    log_message("\n-- Step 1: Doomed Fragment Removals (Successful Discards) --")
    critical_bond_actions = [ba for ba in bond_analyses if ba.get('type') == 'critical_disconnection']
    for ba in critical_bond_actions:
        bond = ba.get('bond')
        env_level3_action = ba.get('level3_action', -1) # Should be 0 or 1
        kept_atom_idx, doomed_atom_idx = ba.get('kept_atom'), ba.get('doomed_atom')

        if bond is None or env_level3_action not in [0, 1] or kept_atom_idx is None or doomed_atom_idx is None: continue

        log_message(f"Remove Doomed Fragment via bond {kept_atom_idx}-{doomed_atom_idx}:")
        level0_action = kept_atom_idx + 1
        level1_action = VOCAB_SIZE + doomed_atom_idx
        level2_action = BOND_REMOVE_ACTION
        level3_action = int(env_level3_action)
        action_tuple = (level0_action, level1_action, level2_action, level3_action)
        log_message(f"  Action Tuple: {action_tuple}")
        action_sequence.append({'type': 'remove_doomed_fragment', 'bond': bond, 'level3': level3_action, 'action_tuple': action_tuple})

    # --- 2. Internal Kept Core Bond Edits (incl. Failed Discards treated as Keep Both) ---
    log_message("\n-- Step 2: Internal Kept Core Bond Edits --")
    internal_bond_actions = [ba for ba in bond_analyses if ba.get('type') == 'internal_kept_core']
    for ba in internal_bond_actions:
        bond = ba.get('bond')
        if bond is None: continue
        atom1_idx, atom2_idx = bond
        creates_fragments = ba.get('creates_fragments', False)
        level3_action_val = ba.get('level3_action', -1) # Should be 2 if fragments created

        log_message(f"Internal Bond removal {atom1_idx}-{atom2_idx}:")
        level0_action = atom1_idx + 1
        level1_action = VOCAB_SIZE + atom2_idx
        level2_action = BOND_REMOVE_ACTION
        action_tuple = None
        if creates_fragments:
            if level3_action_val != 2:
                 log_message(f"  WARNING: Internal bond removal created fragments but L3 action is not 2 (got {level3_action_val}). Forcing L3=2.")
            level3_action = 2 # Force Keep Both for internal splits
            action_tuple = (level0_action, level1_action, level2_action, level3_action)
            log_message(f"  Action Tuple (with L3=Keep Both): {action_tuple}")
        else:
            action_tuple = (level0_action, level1_action, level2_action)
            log_message(f"  Action Tuple: {action_tuple}")
        action_sequence.append({'type': 'internal_bond_removal', 'bond': bond, 'level3': 2 if creates_fragments else None, 'action_tuple': action_tuple})

    # --- 3. Atom Substitutions ---
    log_message("\n-- Step 3: Atom Substitutions --")
    substitutions_ops = operations.get('substitutions', [])
    # Use source_to_target map generated earlier if available
    source_to_target = fate_info.get('source_to_target', {})
    for op in substitutions_ops:
        source_idx = op.get('source_idx')
        # Check if the atom is meant to be kept
        if source_idx is not None and atom_fate.get(source_idx) == "kept":
            to_element_num = op.get('to_element', '?')
            from_element_num = op.get('from_element', '?')
            log_message(f"Substitute Atom {source_idx} ({from_element_num} -> {to_element_num}):")
            level0_action = source_idx + 1
            level1_action = ACTION_REPLACE_ATOM # Use the fixed replace action index
            level2_action = get_vocab_index(to_element_num)
            action_tuple = (level0_action, level1_action, level2_action)
            log_message(f"  Action Tuple: {action_tuple}")
            action_sequence.append({'type': 'substitute_atom', 'source_idx': source_idx, 'to_element': str(to_element_num), 'action_tuple': action_tuple})
        elif source_idx is not None:
            log_message(f"Skipping substitution for doomed atom {source_idx}.")


    # --- 4. Atom Insertions (Multi-Pass) ---
    log_message("\n-- Step 4: Atom Insertions (Multi-Pass) --")
    insertions_ops = operations.get('insertions', [])
    edge_insertions = [op for op in operations.get('edge_operations', []) if op.get('operation') == 'insert_edge']

    insertions_grouped = defaultdict(lambda: {'op': None, 'connections_to_kept': [], 'connections_to_new': []})
    # Use target indices from insertion ops
    new_atom_target_indices = {op.get('target_idx') for op in insertions_ops if op.get('target_idx') is not None}

    log_message(f"Found {len(new_atom_target_indices)} new atom target indices from insert_node ops: {new_atom_target_indices}")

    for op in insertions_ops:
         target_idx = op.get('target_idx')
         if target_idx is not None: insertions_grouped[target_idx]['op'] = op

    log_message(f"Processing {len(edge_insertions)} insert_edge ops for connections...")
    for edge_op in edge_insertions:
         # Indices in insert_edge ops refer to the TARGET graph's indices
         a1_target, a2_target = edge_op.get('atom1_idx'), edge_op.get('atom2_idx')
         bond_type = edge_op.get('bond_type', 1)
         if a1_target is None or a2_target is None: continue

         conn_info = {'bond_type': bond_type}

         # Check if atoms belong to new atoms or map back to kept source atoms
         a1_is_new = a1_target in new_atom_target_indices
         a2_is_new = a2_target in new_atom_target_indices

         # Map target indices back to source indices for kept atoms
         target_to_source = {v: k for k, v in source_to_target.items()}
         a1_source = target_to_source.get(a1_target) if not a1_is_new else None
         a2_source = target_to_source.get(a2_target) if not a2_is_new else None

         # Case 1: New <-> New
         if a1_is_new and a2_is_new:
              insertions_grouped[a1_target]['connections_to_new'].append({'to_new_target_idx': a2_target, **conn_info})
              insertions_grouped[a2_target]['connections_to_new'].append({'to_new_target_idx': a1_target, **conn_info})
         # Case 2: New <-> Kept
         elif a1_is_new and not a2_is_new:
              if a2_source is not None and atom_fate.get(a2_source) == "kept":
                   insertions_grouped[a1_target]['connections_to_kept'].append({'to_kept_source_idx': a2_source, **conn_info})
              else: log_message(f"  Warning: New atom (target {a1_target}) connects to non-kept/unmapped target atom {a2_target}. Ignoring edge.")
         elif a2_is_new and not a1_is_new:
              if a1_source is not None and atom_fate.get(a1_source) == "kept":
                   insertions_grouped[a2_target]['connections_to_kept'].append({'to_kept_source_idx': a1_source, **conn_info})
              else: log_message(f"  Warning: New atom (target {a2_target}) connects to non-kept/unmapped target atom {a1_target}. Ignoring edge.")
         # Case 3: Kept <-> Kept (This is an edge insertion between existing atoms - handle separately?)
         # For now, assuming insert_edge only involves at least one new atom based on GED common practice.
         # If Kept<->Kept insertions are possible, they need separate handling after atom insertions.

    # --- Multi-Pass Mapping ---
    # env_idx_map maps NEW atom TARGET indices to their assigned ENV index
    env_idx_map = {}
    # current_atom_count tracks the next available index in the environment state
    current_atom_count = num_atoms_source # Starts after the initial source atoms

    unmapped_insertions = set(new_atom_target_indices)
    pass_num = 0
    max_passes = len(new_atom_target_indices) + 2 # Safety break

    while unmapped_insertions and pass_num < max_passes:
        pass_num += 1
        log_message(f"\n--- Insertion Pass {pass_num} ---")
        newly_mapped_in_pass = set()
        sorted_unmapped = sorted(list(unmapped_insertions)) # Process deterministically

        for target_idx in sorted_unmapped:
            data = insertions_grouped[target_idx]
            op = data['op']
            if op is None: # Should not happen if logic is correct
                 log_message(f"  ERROR: No insert_node op found for target_idx {target_idx}. Skipping.")
                 continue
            element_num = op.get('element', '?')
            conns_kept = data['connections_to_kept']
            conns_new = data['connections_to_new']

            anchor_info = None

            # Prioritize anchoring to already existing ('kept') atoms
            if conns_kept:
                first_conn = conns_kept[0]
                # Anchor is the SOURCE index of the kept atom
                anchor_kept_source_idx = first_conn['to_kept_source_idx']
                anchor_info = ('kept', anchor_kept_source_idx, first_conn['bond_type'])
                log_message(f"  Trying to map new atom (target {target_idx}, {element_num}) via kept anchor (source {anchor_info[1]})")
            else:
                # If no connection to kept, try anchoring to already mapped NEW atoms
                for conn_new in conns_new:
                    anchor_new_target_idx = conn_new['to_new_target_idx']
                    # Check if the anchor NEW atom has been mapped in a previous pass
                    if anchor_new_target_idx in env_idx_map:
                         # Anchor is the ENV index of the already mapped new atom
                        anchor_new_env_idx = env_idx_map[anchor_new_target_idx]
                        anchor_info = ('new', anchor_new_env_idx, conn_new['bond_type'])
                        log_message(f"  Trying to map new atom (target {target_idx}, {element_num}) via new anchor (target {anchor_new_target_idx}, env {anchor_info[1]})")
                        break

            # If an anchor was found, generate the add_atom action
            if anchor_info:
                anchor_type, anchor_env_or_source_idx, bond_type_val = anchor_info

                # The Level 0 action is always the ENV index of the anchor + 1
                anchor_env_idx = int(anchor_env_or_source_idx) # Already the correct env index for both 'kept' and 'new'

                log_message(f"  Mapping Add Atom (target {target_idx}, {element_num}) anchored to {anchor_type} atom (env_idx={anchor_env_idx})")

                level0_action = anchor_env_idx + 1
                level1_action = get_vocab_index(element_num)
                level2_action = get_bond_action(bond_type_val)
                add_atom_tuple = (level0_action, level1_action, level2_action)
                log_message(f"    Action Tuple (Add Atom): {add_atom_tuple}")
                action_sequence.append({'type': 'add_atom', 'target_idx': target_idx, 'element': str(element_num), 'anchor_env_idx': anchor_env_idx, 'action_tuple': add_atom_tuple})

                # Assign the new atom its environment index and update count
                new_atom_env_idx = current_atom_count
                env_idx_map[target_idx] = new_atom_env_idx
                current_atom_count += 1
                newly_mapped_in_pass.add(target_idx)

                # --- Generate subsequent add_bond actions for this new atom ---
                # Connections to other KEPT atoms (use their source indices)
                # Skip the first one if it was used as the anchor
                other_conns_kept = conns_kept[1:] if anchor_type == 'kept' else conns_kept
                for conn_kept in other_conns_kept:
                    other_kept_source_idx = conn_kept['to_kept_source_idx']
                    add_bond_tuple = generate_add_bond_actions(new_atom_env_idx, other_kept_source_idx, conn_kept['bond_type'])
                    action_sequence.append({'type': 'add_bond', 'from_new_target_idx': target_idx, 'to_kept_source_idx': other_kept_source_idx, 'action_tuple': add_bond_tuple})

                # Connections to other NEW atoms (use their ENV indices if already mapped)
                for conn_new in conns_new:
                    other_new_target_idx = conn_new['to_new_target_idx']
                    # Check if the other new atom is already mapped AND wasn't the anchor
                    if other_new_target_idx in env_idx_map and (anchor_type != 'new' or env_idx_map[other_new_target_idx] != anchor_env_idx):
                       other_new_env_idx = env_idx_map[other_new_target_idx]
                       add_bond_tuple = generate_add_bond_actions(new_atom_env_idx, other_new_env_idx, conn_new['bond_type'])
                       action_sequence.append({'type': 'add_bond', 'from_new_target_idx': target_idx, 'to_new_target_idx': other_new_target_idx, 'action_tuple': add_bond_tuple})
                # --- End generating add_bond actions ---

            else:
                # Check if it only connects to unmapped new atoms
                connects_only_to_unmapped_new = False
                if not conns_kept and conns_new:
                    all_connected_new_are_unmapped = True
                    for conn_new in conns_new:
                        if conn_new['to_new_target_idx'] in env_idx_map:
                            all_connected_new_are_unmapped = False; break
                    if all_connected_new_are_unmapped: connects_only_to_unmapped_new = True

                if connects_only_to_unmapped_new:
                     log_message(f"  Atom (target {target_idx}, {element_num}) only connects to other UNMAPPED new atoms in pass {pass_num}. Deferring.")
                else:
                     log_message(f"  Could not find suitable anchor for new atom (target {target_idx}, {element_num}) in pass {pass_num}. Connections: Kept={len(conns_kept)}, New={len(conns_new)}")


        if not newly_mapped_in_pass:
            log_message(f"--- No insertions mapped in pass {pass_num}. Exiting loop. ---")
            break

        unmapped_insertions -= newly_mapped_in_pass
        log_message(f"--- Mapped {len(newly_mapped_in_pass)} atoms in pass {pass_num}. Remaining: {len(unmapped_insertions)} ---")
        if not unmapped_insertions:
             log_message("--- All insertions mapped. ---")
             break

    if pass_num == max_passes and unmapped_insertions:
         log_message(f"\nWARNING: Exceeded max insertion passes ({max_passes}).")

    if unmapped_insertions:
        log_message(f"\nWARNING: {len(unmapped_insertions)} insert_node operations could not be mapped:")
        for idx in sorted(list(unmapped_insertions)):
             op = insertions_grouped[idx]['op']
             element = op.get('element', '?') if op else '?'
             log_message(f"  - Target Index {idx} ({element})")

    # --- 5. Add Bond Insertions between EXISTING Kept Atoms ---
    # Placeholder: If GED allows inserting edges between atoms that both exist in the source
    # and are kept, these operations need to be mapped here after all atom modifications.
    log_message("\n-- Step 5: Add Bonds Between Existing Kept Atoms (Placeholder) --")
    kept_kept_insertions = []
    for edge_op in edge_insertions:
        a1_target, a2_target = edge_op.get('atom1_idx'), edge_op.get('atom2_idx')
        bond_type = edge_op.get('bond_type', 1)
        if a1_target is None or a2_target is None: continue
        a1_is_new = a1_target in new_atom_target_indices
        a2_is_new = a2_target in new_atom_target_indices
        if not a1_is_new and not a2_is_new:
            # Map target indices back to source indices
            target_to_source = {v: k for k, v in source_to_target.items()}
            a1_source = target_to_source.get(a1_target)
            a2_source = target_to_source.get(a2_target)
            if a1_source is not None and a2_source is not None and atom_fate.get(a1_source) == "kept" and atom_fate.get(a2_source) == "kept":
                 # Ensure this bond doesn't already exist from source_mol processing
                 # This check might be complex; assume for now GED only adds necessary bonds
                 add_bond_tuple = generate_add_bond_actions(a1_source, a2_source, bond_type)
                 action_sequence.append({'type': 'add_bond_kept', 'source_idx1': a1_source, 'source_idx2': a2_source, 'action_tuple': add_bond_tuple})
                 kept_kept_insertions.append(edge_op)
    if kept_kept_insertions:
         log_message(f"Mapped {len(kept_kept_insertions)} add_edge operations between existing kept atoms.")


    return action_sequence


# ==============================================================================
# START OF MODIFIED main
# ==============================================================================
def main():
    """Main function to analyze and map GED operations."""
    log_message(f"Starting analysis run by {CURRENT_USER} at {datetime.datetime.utcnow().isoformat()}Z")
    # Load raw transformation data (SMILES pairs + edit path)
    # Assuming load_transformation_data now returns list of dicts like:
    # {'source_smiles': '...', 'target_smiles': '...', 'edit_path': [...]}
    transformations_raw = load_transformation_data("train")
    if not transformations_raw:
        log_message("No raw transformation data loaded. Exiting.")
        return

    results = []
    skipped_count = 0 # Counter for skipped pairs

    output_dir = "./data/chembl/action_results" # Define output dir
    os.makedirs(output_dir, exist_ok=True) # Ensure output dir exists
    final_output_path = os.path.join(output_dir, f"action_tuples_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pkl")


    log_message(f"Processing {min(len(transformations_raw), MAX_TRANSFORMATIONS)} transformations...")

    # Use tqdm for progress bar
    for i, transformation in enumerate(tqdm(transformations_raw[:MAX_TRANSFORMATIONS], desc="Mapping GED to Actions")):
        source_smiles = transformation.get('source_smiles')
        target_smiles = transformation.get('target_smiles')
        edit_path = transformation.get('edit_path') # Get the raw edit path

        if not source_smiles or not target_smiles or edit_path is None:
             log_message(f"\nSkipping transformation {i+1} due to missing SMILES or edit_path.")
             skipped_count += 1
             continue

        log_message(f"\n\n==================================================")
        log_message(f"Analyzing transformation {i+1}/{min(len(transformations_raw), MAX_TRANSFORMATIONS)}")
        log_message(f"Source: {source_smiles}")
        log_message(f"Target: {target_smiles}")
        log_message(f"==================================================\n")

        source_mol, target_mol = None, None

        try:
            # --- Prepare Molecules Consistently ---
            source_mol = Chem.MolFromSmiles(source_smiles)
            target_mol = Chem.MolFromSmiles(target_smiles)

            if not source_mol or not target_mol:
                raise ValueError("MolFromSmiles failed")

            # Sanitize first
            Chem.SanitizeMol(source_mol)
            Chem.SanitizeMol(target_mol)

            # Kekulize consistently
            Chem.Kekulize(source_mol, clearAromaticFlags=True)
            Chem.Kekulize(target_mol, clearAromaticFlags=True)

            # Canonicalize consistently
            source_mol = rdmolops.RenumberAtoms(source_mol, list(rdmolfiles.CanonicalRankAtoms(source_mol)))
            target_mol = rdmolops.RenumberAtoms(target_mol, list(rdmolfiles.CanonicalRankAtoms(target_mol)))
            # --- End Prepare Molecules ---

        except Exception as e:
            log_message(f"ERROR preparing RDKit molecules for pair ({source_smiles}, {target_smiles}): {e}")
            skipped_count += 1
            continue # Skip this pair if molecule preparation fails

        try:
            # --- Perform Analysis ---
            operations = categorize_operations(list(edit_path)) # Pass a copy of edit_path
            fate_info = build_atom_fate_map(operations, source_mol, target_mol)
            graph_analysis = analyze_graph_structure(source_mol, fate_info['atom_fate'], operations)

            # --- Call bond analysis (Strict Check) ---
            bond_analyses = analyze_bond_removals_efficient(graph_analysis, source_mol, fate_info, operations)

            if bond_analyses is None:
                # The reason for skipping is already logged inside analyze_bond_removals_efficient
                log_message(f"Skipping pair ({source_smiles}, {target_smiles}) due to fatal error during bond removal analysis (e.g., sanitization failure).")
                skipped_count += 1
                continue # Skip to the next pair
            # --- End Bond Analysis Check ---

            # --- Map to actions (only if bond analysis succeeded) ---
            action_sequence = map_operations_to_action_sequence_efficient(operations, source_mol, graph_analysis, bond_analyses, fate_info)
            # --- End Map to Actions ---

            log_message(f"\n--- Generated Action Sequence ({len(action_sequence)} items) ---")

            # Store results
            results.append({
                'transformation_index': i,
                'source_smiles': source_smiles,
                'target_smiles': target_smiles,
                # Store only the action tuples for the final dataset
                'action_tuples': [item['action_tuple'] for item in action_sequence if 'action_tuple' in item]
                # Optionally store the raw edit path if needed for debugging
                # 'raw_edit_path': edit_path
            })

        except Exception as analysis_err:
             log_message(f"\nERROR during analysis/mapping for pair ({source_smiles}, {target_smiles}): {analysis_err}")
             skipped_count += 1
             # Optionally add more detailed error traceback here if needed
             import traceback
             log_message(traceback.format_exc())
             continue # Skip pair on general analysis errors


        # --- Checkpoint Saving ---
        # Save checkpoint periodically based on the number of SUCCESSFULLY processed results
        if results and (len(results) % CHECKPOINT_FREQUENCY == 0):
            checkpoint_timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            # Save to the dedicated output directory
            current_checkpoint_path = os.path.join(output_dir, f"action_tuples_checkpoint_{checkpoint_timestamp}.pkl")
            try:
                with open(current_checkpoint_path, "wb") as f:
                    pickle.dump(results, f)
                log_message(f"\nCheckpoint saved: {current_checkpoint_path} ({len(results)} successfully processed transformations)")
            except Exception as cp_err:
                log_message(f"\nERROR saving checkpoint {current_checkpoint_path}: {cp_err}")
        # --- End Checkpoint Saving ---


    log_message("\n=== Analysis Complete ===")
    log_message(f"Successfully processed {len(results)} transformations.")
    log_message(f"Skipped {skipped_count} transformations due to errors or sanitization failures.")

    # Save the final results
    try:
        with open(final_output_path, "wb") as f:
             pickle.dump(results, f)
        log_message(f"Saved final action sequence results ({len(results)} items) to {final_output_path}")
    except Exception as final_save_err:
        log_message(f"ERROR saving final results to {final_output_path}: {final_save_err}")

# ==============================================================================
# END OF MODIFIED main
# ==============================================================================


if __name__ == "__main__":
    main()