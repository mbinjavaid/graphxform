"""
GED to MoleculeDesign Mapping - Graph-based efficient fragmentation strategy.
Version: 2025-04-14 02:31:48 UTC (Handle failed discards, add insertion logging)
"""
import pickle
import os
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdmolfiles, rdmolops, AllChem
from collections import defaultdict, deque
import datetime # For timestamp

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Configuration
CHECKPOINT_DIR = "./data/chembl/checkpoints"
DEBUG = True
MAX_TRANSFORMATIONS = 100


def log_message(message):
    """Helper function for timestamped logging."""
    if DEBUG:
        timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"[{timestamp}] {message}")

def load_transformation_data(datatype="train"):
    """Load existing transformation data from latest checkpoint."""
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith(f"transformation_data_{datatype}_")]

    if checkpoint_files:
        checkpoint_files.sort(reverse=True)
        latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoint_files[0])
        log_message(f"Loading from checkpoint: {latest_checkpoint}")
        with open(latest_checkpoint, "rb") as f:
            transformation_data = pickle.load(f)
        log_message(f"Loaded {len(transformation_data)} transformations")
        return transformation_data[:MAX_TRANSFORMATIONS]

    log_message(f"No transformation data found for {datatype}")
    return []


def categorize_operations(edit_path):
    """Categorize GED operations by type."""
    log_message("Categorizing GED operations...")
    metadata = None
    # Extract metadata (often the last operation)
    for i, op in enumerate(edit_path):
        if op['operation'] == 'metadata':
            metadata = op
            edit_path = edit_path[:i] # Remove metadata from main path
            break

    # Ensure edit_path is a list if it became None
    edit_path = edit_path or []

    substitutions = []
    deletions = []
    insertions = []
    edge_operations = []

    for op in edit_path:
        op_type = op.get('operation')
        if op_type == 'substitute_node':
            substitutions.append(op)
        elif op_type == 'delete_node':
            deletions.append(op)
        elif op_type == 'insert_node':
            insertions.append(op)
        elif op_type in ['insert_edge', 'delete_edge', 'substitute_edge']:
            edge_operations.append(op)
        else:
             log_message(f"Warning: Unknown operation type encountered: {op_type}")

    # --- Insertion Debug Logging ---
    edge_insertions = [op for op in edge_operations if op.get('operation') == 'insert_edge']
    log_message(f"Categorized Operations: {len(substitutions)} substitutions, {len(deletions)} deletions, {len(insertions)} insertions, {len(edge_operations)} edge ops ({len(edge_insertions)} insert_edge).")
    # --- End Debug Logging ---

    return {
        'metadata': metadata,
        'substitutions': substitutions,
        'deletions': deletions,
        'insertions': insertions,
        'edge_operations': edge_operations
    }


def build_atom_fate_map(operations, source_mol, target_mol):
    """Build a map of which atoms are kept vs. doomed."""
    log_message("\n=== Building Atom Fate Map ===")

    atom_fate = {i: "unknown" for i in range(source_mol.GetNumAtoms())}
    source_to_target = {} # Map source index to target index for kept/substituted atoms

    # Mark explicitly deleted atoms as "doomed"
    for del_op in operations.get('deletions', []):
        source_idx = del_op.get('source_idx')
        if source_idx is not None and 0 <= source_idx < source_mol.GetNumAtoms(): # Add bounds check
            atom_fate[source_idx] = "doomed"
            log_message(f"Atom {source_idx} ({del_op.get('element', '?')}) is explicitly deleted")

    # Mark substituted atoms as "kept" and establish source->target mappings
    for sub_op in operations.get('substitutions', []):
        source_idx = sub_op.get('source_idx')
        target_idx = sub_op.get('target_idx')
        if source_idx is not None and 0 <= source_idx < source_mol.GetNumAtoms(): # Add bounds check
            atom_fate[source_idx] = "kept"
            if target_idx is not None:
                 source_to_target[source_idx] = target_idx
            log_message(f"Atom {source_idx} -> {target_idx if target_idx is not None else '?'} (substituted {sub_op.get('from_element', '?')} -> {sub_op.get('to_element', '?')})")

    # Any atom not marked as doomed or kept is implicitly kept without modification
    for i in range(source_mol.GetNumAtoms()):
        if atom_fate[i] == "unknown":
            atom_fate[i] = "kept"
            log_message(f"Atom {i} implicitly kept but target index unknown (needs inference)")

    kept_count = sum(1 for fate in atom_fate.values() if fate == "kept")
    doomed_count = sum(1 for fate in atom_fate.values() if fate == "doomed")
    log_message(f"Atoms: {kept_count} kept, {doomed_count} doomed")

    return {
        'atom_fate': atom_fate,
        'source_to_target': source_to_target
    }


def build_adjacency_list(mol):
    """Build an adjacency list representation of the molecule."""
    adj = defaultdict(list)
    if mol: # Check if mol is valid
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            adj[begin_idx].append(end_idx)
            adj[end_idx].append(begin_idx)
    return adj


def analyze_graph_structure(source_mol, atom_fate, operations):
    """
    Identifies Kept Components, Doomed Fragments, and Critical Bonds using graph analysis.
    Considers all kept atoms and connections. Includes fix for single doomed atoms.
    """
    log_message("\n=== Analyzing Graph Structure (Kept/Doomed) ===")
    adj = build_adjacency_list(source_mol)
    visited = set()
    kept_components = []  # Store all kept components (sets of atom indices)
    doomed_fragments = [] # Store all doomed fragments (sets of atom indices)
    num_atoms = source_mol.GetNumAtoms()

    # Find all connected components of kept atoms (Kept Components)
    for i in range(num_atoms):
        if atom_fate.get(i) == "kept" and i not in visited:
            component = set()
            q = deque([i])
            while q:
                u = q.popleft()
                if u in visited or atom_fate.get(u) != "kept":
                    continue
                visited.add(u)
                component.add(u)
                for v in adj.get(u, []): # Use .get for safety
                    if v not in visited and atom_fate.get(v) == "kept":
                        q.append(v)
            if component:
                kept_components.append(component)

    # Find all connected components of doomed atoms (Doomed Fragments)
    visited_doomed = set()
    for i in range(num_atoms):
        if atom_fate.get(i) == "doomed" and i not in visited_doomed:
            component = set()
            q = deque([i])
            while q:
                u = q.popleft()
                if u in visited_doomed or atom_fate.get(u) != "doomed":
                    continue
                visited_doomed.add(u)
                component.add(u)
                for v in adj.get(u, []):
                    if v not in visited_doomed and atom_fate.get(v) == "doomed":
                         q.append(v)
            if component:
                 doomed_fragments.append(component)

    log_message(f"Identified {len(kept_components)} Kept Components (Sizes: {[len(c) for c in kept_components]}).")
    log_message(f"Identified {len(doomed_fragments)} initial Doomed Fragments (Sizes: {[len(f) for f in doomed_fragments]}).")

    # --- Identify single doomed atoms attached only to kept atoms ---
    critically_attached_single_doomed = set()
    all_doomed_in_fragments = set().union(*doomed_fragments) if doomed_fragments else set()

    for i in range(num_atoms):
        if atom_fate.get(i) == "doomed":
            if i not in all_doomed_in_fragments:
                neighbors = adj.get(i, [])
                neighbor_fates = [atom_fate.get(n) for n in neighbors]
                if neighbors and all(fate == "kept" for fate in neighbor_fates):
                    critically_attached_single_doomed.add(i)

    if critically_attached_single_doomed:
        log_message(f"Identified {len(critically_attached_single_doomed)} single doomed atoms connected only to Kept components: {critically_attached_single_doomed}")
        for single_doomed_idx in critically_attached_single_doomed:
             doomed_fragments.append({single_doomed_idx}) # Add as a fragment
        log_message(f"Total Doomed Fragments (including singles): {len(doomed_fragments)}")

    # Combine all kept atoms into a single set for easier checking
    all_kept_atoms = set().union(*kept_components) if kept_components else set()

    # Identify critical disconnection bonds (connecting ANY kept to ANY doomed)
    critical_bonds = []

    for op in [op for op in operations.get('edge_operations', []) if op.get('operation') == 'delete_edge']:
        a1 = op.get('atom1_idx')
        a2 = op.get('atom2_idx')

        if a1 is None or a2 is None or not (0 <= a1 < num_atoms and 0 <= a2 < num_atoms): continue

        bond_key = tuple(sorted((a1, a2)))
        fragment_involved = None
        kept_component_involved = None
        is_critical = False

        if (a1 in all_kept_atoms and atom_fate.get(a2) == "doomed") or \
           (a2 in all_kept_atoms and atom_fate.get(a1) == "doomed"):

            kept_atom = a1 if atom_fate.get(a1) == "kept" else a2
            doomed_atom = a1 if atom_fate.get(a1) == "doomed" else a2

            for fragment in doomed_fragments:
                if doomed_atom in fragment:
                    fragment_involved = fragment
                    is_critical = True
                    break

            for component in kept_components:
                 if kept_atom in component:
                     kept_component_involved = component
                     break

        if is_critical and fragment_involved is not None:
             critical_bonds.append({
                 'bond': bond_key,
                 'kept_atom': kept_atom,
                 'doomed_atom': doomed_atom,
                 'doomed_fragment': fragment_involved,
                 'kept_component': kept_component_involved,
                 'original_op': op
             })

    log_message(f"Identified {len(critical_bonds)} critical bonds connecting Kept/Doomed components.")

    # Identify internal kept deletions (bond deletion where BOTH atoms are kept)
    internal_kept_deletions = []
    critical_bond_keys = {cb['bond'] for cb in critical_bonds} # Set for faster lookup

    for op in [op for op in operations.get('edge_operations', []) if op.get('operation') == 'delete_edge']:
         a1 = op.get('atom1_idx')
         a2 = op.get('atom2_idx')

         if a1 is None or a2 is None or not (0 <= a1 < num_atoms and 0 <= a2 < num_atoms): continue

         bond_key = tuple(sorted((a1, a2)))
         if atom_fate.get(a1) == "kept" and atom_fate.get(a2) == "kept" and \
            bond_key not in critical_bond_keys:
             internal_kept_deletions.append(op)

    log_message(f"Identified {len(internal_kept_deletions)} internal bond deletions within Kept components.")

    return {
        'kept_components': kept_components,
        'all_kept_atoms': all_kept_atoms,
        'doomed_fragments': doomed_fragments,
        'critical_bonds': critical_bonds,
        'internal_kept_deletions': internal_kept_deletions
    }


def analyze_fragment(fragment_indices, atom_fate):
    """Analyze a fragment based on atom fates (indices only)."""
    if not hasattr(fragment_indices, '__iter__'):
        return {"kept_count": 0, "doomed_count": 0, "should_discard": True, "action": "discard", "atoms": []}

    kept_count = sum(1 for atom_idx in fragment_indices if atom_fate.get(atom_idx) == "kept")
    doomed_count = sum(1 for atom_idx in fragment_indices if atom_fate.get(atom_idx) == "doomed")
    should_discard = kept_count == 0

    return {
        "atoms": list(fragment_indices),
        "kept_count": kept_count,
        "doomed_count": doomed_count,
        "should_discard": should_discard,
        "action": "discard" if should_discard else "keep"
    }


def analyze_bond_removals_efficient(graph_analysis, source_mol, fate_info, operations):
    """
    Analyzes bond removals focusing on critical disconnections and internal core edits.
    If critical bond verification fails, treats it as internal (keep both).
    """
    log_message("\n=== Analyzing Bond Removals (Efficient Strategy) ===")

    critical_bonds = graph_analysis.get('critical_bonds', [])
    internal_kept_deletions = graph_analysis.get('internal_kept_deletions', [])
    atom_fate = fate_info.get('atom_fate', {})

    bond_analyses = []
    working_mol = Chem.Mol(source_mol)

    log_message(f"\n--- Processing {len(critical_bonds)} Critical Bonds ---")
    processed_bond_keys = set()

    for i, cb_info in enumerate(critical_bonds):
        op = cb_info['original_op']
        atom1_idx = op.get('atom1_idx')
        atom2_idx = op.get('atom2_idx')
        bond_key = tuple(sorted((atom1_idx, atom2_idx)))

        if atom1_idx is None or atom2_idx is None or bond_key in processed_bond_keys:
            continue

        original_doomed_atom_in_bond = cb_info.get('doomed_atom')
        if original_doomed_atom_in_bond is None:
             log_message(f"  WARNING: Critical bond info missing doomed_atom for {bond_key}. Skipping.")
             continue

        log_message(f"\nCritical Bond {i + 1}: atoms {atom1_idx}-{atom2_idx} (Targeting removal including doomed atom {original_doomed_atom_in_bond})")

        bond = working_mol.GetBondBetweenAtoms(atom1_idx, atom2_idx)
        if bond is None:
            log_message(f"  WARNING: Bond {atom1_idx}-{atom2_idx} not found in current state. Skipping.")
            continue

        rwmol = Chem.RWMol(working_mol)
        rwmol.RemoveBond(atom1_idx, atom2_idx)
        modified_mol = None # Initialize
        try:
            Chem.SanitizeMol(rwmol)
            modified_mol = Chem.Mol(rwmol)
        except Exception as e:
            log_message(f"  WARNING: RDKit Sanitization failed after removing bond {atom1_idx}-{atom2_idx}: {e}. Proceeding with unsanitized mol.")
            modified_mol = rwmol.GetMol()

        if modified_mol is None: # Should not happen with GetMol, but safety check
             log_message(f"  ERROR: Failed to create modified molecule after removing bond {bond_key}. Skipping.")
             continue

        new_components = []
        try:
            new_components = Chem.GetMolFrags(modified_mol, asMols=False, sanitizeFrags=False)
        except Exception as e:
             log_message(f"  ERROR: RDKit GetMolFrags failed after removing bond {atom1_idx}-{atom2_idx}: {e}. Skipping analysis for this bond.")
             working_mol = modified_mol
             processed_bond_keys.add(bond_key)
             continue

        log_message(f"  After removal, molecule has {len(new_components)} components.")

        doomed_frag_num = -1
        identified_doomed_component_indices = set()

        for frag_idx, frag_comp in enumerate(new_components):
            if not frag_comp or not hasattr(frag_comp, '__iter__'): continue
            frag_set = set(frag_comp)
            if original_doomed_atom_in_bond in frag_set:
                doomed_frag_num = frag_idx
                identified_doomed_component_indices = frag_set
                log_message(f"  Identified Component {frag_idx} (Size: {len(frag_set)}) as containing the doomed atom {original_doomed_atom_in_bond}.")
                break

        # --- MODIFICATION START: Handle failed verification ---
        analysis_added = False
        if doomed_frag_num != -1:
            frag_analysis = analyze_fragment(identified_doomed_component_indices, atom_fate)
            if frag_analysis['should_discard']: # Verification SUCCESS
                level3_action = doomed_frag_num
                log_message(f"  Verified fragment {doomed_frag_num} contains only doomed atoms.")
                log_message(f"  Level 3 action: {level3_action} (discard fragment {doomed_frag_num})")

                bond_analyses.append({
                    'bond': bond_key, 'type': 'critical_disconnection', 'creates_fragments': True,
                    'level3_action': level3_action, 'doomed_fragment_indices': identified_doomed_component_indices,
                    'original_op': op, 'kept_atom': cb_info.get('kept_atom'), 'doomed_atom': original_doomed_atom_in_bond
                })
                analysis_added = True

            else: # Verification FAILED
                log_message(f"  ERROR: Component {doomed_frag_num} containing doomed atom {original_doomed_atom_in_bond} also has {frag_analysis['kept_count']} kept atoms.")
                log_message(f"  Treating bond {bond_key} as internal removal (keep both fragments).")
                # Add analysis as an 'internal' type that creates fragments and keeps both
                bond_analyses.append({
                    'bond': bond_key, 'type': 'internal_kept_core', 'creates_fragments': True,
                    'level3_action': 2, # Keep both
                    'original_op': op
                    # Note: We don't need doomed_fragment_indices here as we keep both
                })
                analysis_added = True
        # --- MODIFICATION END ---
        else:
             log_message(f"  ERROR: Could not find component containing doomed atom {original_doomed_atom_in_bond} after removal. Bond {bond_key}.")

        if analysis_added:
             processed_bond_keys.add(bond_key) # Mark as processed ONLY if analysis was added

        working_mol = modified_mol

    # 2. Process truly Internal Kept Core Deletions (those not already handled above)
    log_message(f"\n--- Processing {len(internal_kept_deletions)} initially classified Internal Kept Core Bonds ---")
    for i, op in enumerate(internal_kept_deletions):
        atom1_idx = op.get('atom1_idx')
        atom2_idx = op.get('atom2_idx')

        if atom1_idx is None or atom2_idx is None: continue
        bond_key = tuple(sorted((atom1_idx, atom2_idx)))

        # Skip if already processed (e.g., handled as a failed critical bond)
        if bond_key in processed_bond_keys:
            # log_message(f"Skipping internal bond {bond_key} as it was already processed.")
            continue

        log_message(f"\nInternal Bond {i + 1}: atoms {atom1_idx}-{atom2_idx}")

        bond = working_mol.GetBondBetweenAtoms(atom1_idx, atom2_idx)
        if bond is None:
            log_message(f"  WARNING: Internal bond {atom1_idx}-{atom2_idx} not found in current state. Skipping.")
            continue

        prev_components = []
        try:
            prev_components = Chem.GetMolFrags(working_mol, asMols=False, sanitizeFrags=False)
        except Exception as e:
             log_message(f"  ERROR: GetMolFrags failed BEFORE internal removal {atom1_idx}-{atom2_idx}: {e}. Skipping.")
             continue

        rwmol = Chem.RWMol(working_mol)
        rwmol.RemoveBond(atom1_idx, atom2_idx)
        modified_mol = None
        try:
            Chem.SanitizeMol(rwmol)
            modified_mol = Chem.Mol(rwmol)
        except Exception as e:
            log_message(f"  WARNING: RDKit Sanitization failed after internal removal {atom1_idx}-{atom2_idx}: {e}. Proceeding unsanitized.")
            modified_mol = rwmol.GetMol()

        if modified_mol is None:
             log_message(f"  ERROR: Failed to create modified molecule after internal removal {bond_key}. Skipping.")
             continue

        new_components = []
        try:
            new_components = Chem.GetMolFrags(modified_mol, asMols=False, sanitizeFrags=False)
        except Exception as e:
             log_message(f"  ERROR: GetMolFrags failed AFTER internal removal {atom1_idx}-{atom2_idx}: {e}. Skipping analysis.")
             working_mol = modified_mol
             processed_bond_keys.add(bond_key)
             continue

        analysis = {
            'bond': bond_key, 'type': 'internal_kept_core', 'creates_fragments': False, 'original_op': op
        }

        if len(new_components) > len(prev_components):
            analysis['creates_fragments'] = True
            log_message("  Internal bond removal created new fragments within the Kept structure.")

            if len(new_components) == len(prev_components) + 1:
                frag1_indices, frag2_indices = set(), set()
                found_frags = 0
                for frag_comp in new_components:
                    if not frag_comp or not hasattr(frag_comp, '__iter__'): continue
                    frag_set = set(frag_comp)
                    if atom1_idx in frag_set or atom2_idx in frag_set:
                        if found_frags == 0: frag1_indices = frag_set
                        elif found_frags == 1: frag2_indices = frag_set
                        found_frags += 1

                level3_action = 2 # Default keep both
                if not frag1_indices or not frag2_indices:
                     log_message("  WARNING: Could not reliably identify the two fragments resulting from internal split. Defaulting Level 3.")
                else:
                    frag1_analysis = analyze_fragment(frag1_indices, atom_fate)
                    frag2_analysis = analyze_fragment(frag2_indices, atom_fate)
                    log_message(f"  Fragment 1 (Size: {len(frag1_indices)}): Kept={frag1_analysis['kept_count']}, Doomed={frag1_analysis['doomed_count']} -> {frag1_analysis['action']}")
                    log_message(f"  Fragment 2 (Size: {len(frag2_indices)}): Kept={frag2_analysis['kept_count']}, Doomed={frag2_analysis['doomed_count']} -> {frag2_analysis['action']}")

                    if frag1_analysis['should_discard'] and frag2_analysis['should_discard']: level3_action = 0
                    elif frag1_analysis['should_discard']: level3_action = 0
                    elif frag2_analysis['should_discard']: level3_action = 1
                    else: level3_action = 2
                    log_message(f"  Level 3 action: {level3_action} ({'discard frag 1' if level3_action==0 else ('discard frag 2' if level3_action==1 else 'keep both')})")
                analysis['level3_action'] = level3_action
            else:
                 log_message(f"  WARNING: Internal removal resulted in unexpected fragmentation ({len(prev_components)} -> {len(new_components)}). Defaulting Level 3.")
                 analysis['level3_action'] = 2
        else:
             log_message("  Internal bond removal did not create new fragments.")

        bond_analyses.append(analysis)
        processed_bond_keys.add(bond_key)
        working_mol = modified_mol

    # Final check for unprocessed deletions
    all_delete_ops = [tuple(sorted((op.get('atom1_idx'), op.get('atom2_idx'))))
                      for op in operations.get('edge_operations', [])
                      if op.get('operation') == 'delete_edge' and op.get('atom1_idx') is not None and op.get('atom2_idx') is not None]
    unprocessed_deletions = set(all_delete_ops) - processed_bond_keys
    if unprocessed_deletions:
        log_message(f"\nWARNING: {len(unprocessed_deletions)} delete_edge operations were not processed:")
        # for bond_key in unprocessed_deletions: log_message(f"  - Bond {bond_key}")

    return bond_analyses


def map_operations_to_action_sequence_efficient(operations, source_mol, graph_analysis, bond_analyses, fate_info):
    """
    Maps operations to MoleculeDesign actions using the efficient strategy.
    """
    log_message("\n=== Mapping Operations to Action Sequences (Efficient Strategy) ===")

    all_kept_atoms = graph_analysis.get('all_kept_atoms', set())
    vocab_size = 10
    num_atoms_source = source_mol.GetNumAtoms()

    ACTION_REPLACE_ATOM = vocab_size + num_atoms_source
    ACTION_REMOVE_BOND = vocab_size + 6
    ACTION_ADD_ATOM = vocab_size + num_atoms_source + 1
    ACTION_ADD_BOND = vocab_size + 7

    action_sequence = []

    # --- 1. Atom Substitutions (on Kept Atoms) ---
    log_message("\n-- Atom Substitutions --")
    substitutions_ops = operations.get('substitutions', [])
    for op in substitutions_ops:
        source_idx = op.get('source_idx')
        if source_idx is not None and source_idx in all_kept_atoms:
            from_element = op.get('from_element', '?')
            to_element = op.get('to_element', '?')
            log_message(f"Substitute Atom {source_idx} ({from_element} -> {to_element}):")

            action_step1 = f"Level 0: Select atom {source_idx}"
            action_step2 = f"Level 1: Choose 'Replace atom' (Action {ACTION_REPLACE_ATOM})"
            action_step3 = f"Level 2: Select new atom type {to_element}"

            log_message(f"  {action_step1}")
            log_message(f"  {action_step2}")
            log_message(f"  {action_step3}")
            action_sequence.append({'type': 'substitute_atom', 'source_idx': source_idx, 'to_element': to_element, 'actions': [action_step1, action_step2, action_step3]})
        elif source_idx is not None:
             log_message(f"Skipping substitution for doomed atom {source_idx}.")


    # --- 2. Critical Bond Removals (Successful Doomed Fragment Removal) ---
    log_message("\n-- Doomed Fragment Removals (Successful Discards) --")
    # Only map actions that were verified and added as 'critical_disconnection'
    critical_bond_actions = [ba for ba in bond_analyses if ba.get('type') == 'critical_disconnection']
    for ba in critical_bond_actions:
        atom1, atom2 = ba.get('bond', (None, None))
        level3_action = ba.get('level3_action', -1)
        kept_atom = ba.get('kept_atom')
        doomed_atom = ba.get('doomed_atom')

        if atom1 is None or atom2 is None or level3_action == -1 or kept_atom is None or doomed_atom is None:
            log_message(f"Skipping mapping for invalid critical bond analysis: {ba.get('bond')}")
            continue

        log_message(f"Remove Doomed Fragment via bond {kept_atom}-{doomed_atom}:")
        action_step1 = f"Level 0: Select atom {kept_atom}"
        action_step2 = f"Level 1: Select atom {doomed_atom} (Action {vocab_size + doomed_atom})"
        action_step3 = f"Level 2: Remove bond (Action {ACTION_REMOVE_BOND})"
        action_step4 = f"Level 3: Discard fragment {level3_action}"

        log_message(f"  {action_step1}")
        log_message(f"  {action_step2}")
        log_message(f"  {action_step3}")
        log_message(f"  {action_step4}")
        action_sequence.append({'type': 'remove_doomed_fragment', 'bond': (kept_atom, doomed_atom), 'level3': level3_action, 'actions': [action_step1, action_step2, action_step3, action_step4]})


    # --- 3. Internal Kept Core Bond Removals (including failed criticals treated as internal) ---
    log_message("\n-- Internal Kept Core Bond Edits (incl. Failed Discards) --")
    internal_bond_actions = [ba for ba in bond_analyses if ba.get('type') == 'internal_kept_core']
    for ba in internal_bond_actions:
        atom1, atom2 = ba.get('bond', (None, None))
        creates_fragments = ba.get('creates_fragments', False)
        # If it creates fragments, level3_action should be set (defaulted to 2 if failed critical)
        level3_action = ba.get('level3_action', -1 if not creates_fragments else 2)

        if atom1 is None or atom2 is None: continue

        log_message(f"Internal Bond removal {atom1}-{atom2}:")
        action_step1 = f"Level 0: Select atom {atom1}"
        action_step2 = f"Level 1: Select atom {atom2} (Action {vocab_size + atom2})"
        action_step3 = f"Level 2: Remove bond (Action {ACTION_REMOVE_BOND})"

        log_message(f"  {action_step1}")
        log_message(f"  {action_step2}")
        log_message(f"  {action_step3}")
        actions = [action_step1, action_step2, action_step3]

        if creates_fragments:
            level3_text = "Keep both fragments"
            if level3_action == 0: level3_text = "Discard fragment 0"
            elif level3_action == 1: level3_text = "Discard fragment 1"
            action_step4 = f"Level 3: {level3_text}"
            log_message(f"  {action_step4}")
            actions.append(action_step4)

        action_sequence.append({'type': 'internal_bond_removal', 'bond': (atom1, atom2), 'level3': level3_action if creates_fragments else None, 'actions': actions})


    # --- 4. Atom Insertions (Connect to Kept Structure) ---
    log_message("\n-- Atom Insertions --")
    insertions_ops = operations.get('insertions', [])
    edge_insertions = [op for op in operations.get('edge_operations', []) if op.get('operation') == 'insert_edge']

    insertions_grouped = defaultdict(lambda: {'op': None, 'connections': []})
    new_atom_target_indices = {op.get('target_idx') for op in insertions_ops if op.get('target_idx') is not None}

    # --- Insertion Debug Logging ---
    log_message(f"Found {len(new_atom_target_indices)} new atom target indices: {new_atom_target_indices}")
    log_message(f"All kept atoms: {all_kept_atoms}")
    # --- End Debug Logging ---

    for op in insertions_ops:
         target_idx = op.get('target_idx')
         if target_idx is not None:
             insertions_grouped[target_idx]['op'] = op

    log_message(f"Processing {len(edge_insertions)} insert_edge operations for connections...")
    for edge_op in edge_insertions:
         a1 = edge_op.get('atom1_idx')
         a2 = edge_op.get('atom2_idx')
         bond_type = edge_op.get('bond_type', 1)

         if a1 is None or a2 is None: continue

         # --- Insertion Debug Logging ---
         # log_message(f" Checking edge {a1}-{a2}: a1_new={a1 in new_atom_target_indices}, a2_new={a2 in new_atom_target_indices}, a1_kept={a1 in all_kept_atoms}, a2_kept={a2 in all_kept_atoms}")
         # --- End Debug Logging ---

         if a1 in new_atom_target_indices and a2 not in new_atom_target_indices:
             if a2 in all_kept_atoms:
                 log_message(f"  Found connection for new atom {a1} to kept atom {a2}")
                 insertions_grouped[a1]['connections'].append({'to_atom': a2, 'bond_type': bond_type})
             else:
                  log_message(f"  Warning: Insertion edge {a1}-{a2} connects new atom {a1} to a non-kept/unknown existing atom {a2}.")
         elif a2 in new_atom_target_indices and a1 not in new_atom_target_indices:
             if a1 in all_kept_atoms:
                 log_message(f"  Found connection for new atom {a2} to kept atom {a1}")
                 insertions_grouped[a2]['connections'].append({'to_atom': a1, 'bond_type': bond_type})
             else:
                  log_message(f"  Warning: Insertion edge {a1}-{a2} connects new atom {a2} to a non-kept/unknown existing atom {a1}.")
         elif a1 in new_atom_target_indices and a2 in new_atom_target_indices:
              log_message(f"  Info: Insertion edge {a1}-{a2} connects two new atoms. Requires ordered insertion or multi-atom add.")

    log_message("Mapping grouped insertions to actions...")
    processed_insertions = set()
    sorted_target_indices = sorted(insertions_grouped.keys())

    for target_idx in sorted_target_indices:
         data = insertions_grouped[target_idx]
         op = data['op']
         connections = data['connections'] # Connections to EXISTING kept atoms
         element = op.get('element', '?')

         # --- Insertion Debug Logging ---
         log_message(f" Attempting to map insertion for target_idx {target_idx} ({element}). Connections to kept: {connections}")
         # --- End Debug Logging ---

         if not connections:
             # Logic to check if it connects ONLY to other NEW atoms (copied from previous version)
             connects_only_to_new = False
             all_connected_atoms = set()
             # ... (rest of checking logic) ...
             if connects_only_to_new and all_connected_atoms:
                  log_message(f"Atom {target_idx} ({element}): Connects only to other new atoms {all_connected_atoms}. Requires dependent insertion mapping.")
             else:
                  log_message(f"Atom {target_idx} ({element}): No connections found to kept atoms in edge ops. Cannot map insertion simply.")
             continue

         first_conn = connections[0]
         anchor_atom = first_conn['to_atom']
         bond_type = first_conn['bond_type']

         log_message(f"Add Atom {target_idx} ({element}) connected to kept atom {anchor_atom}:")
         action_step1 = f"Level 0: Select atom {anchor_atom}"
         action_step2 = f"Level 1: Choose 'Add atom' (Action {ACTION_ADD_ATOM})"
         action_step3 = f"Level 2: Select new atom type {element}"
         action_step4 = f"Level 3: Set bond type {bond_type}"

         log_message(f"  {action_step1}")
         log_message(f"  {action_step2}")
         log_message(f"  {action_step3}")
         log_message(f"  {action_step4}")
         actions = [action_step1, action_step2, action_step3, action_step4]

         new_atom_placeholder_id = f"newly_added_{target_idx}"

         additional_bond_actions = []
         for conn in connections[1:]:
             existing_atom = conn['to_atom']
             bond_type = conn['bond_type']
             log_message(f"  + Connect new atom {new_atom_placeholder_id} to kept atom {existing_atom}:")

             action_addbond_1 = f"Level 0: Select atom {new_atom_placeholder_id}"
             action_addbond_2 = f"Level 1: Select atom {existing_atom} (Action {vocab_size + existing_atom})"
             action_addbond_3 = f"Level 2: Add bond (Action {ACTION_ADD_BOND})"
             action_addbond_4 = f"Level 3: Set bond type {bond_type}"

             log_message(f"    {action_addbond_1}")
             log_message(f"    {action_addbond_2}")
             log_message(f"    {action_addbond_3}")
             log_message(f"    {action_addbond_4}")
             additional_bond_actions.extend([action_addbond_1, action_addbond_2, action_addbond_3, action_addbond_4])

         actions.extend(additional_bond_actions)
         action_sequence.append({'type': 'add_atom', 'target_idx': target_idx, 'element': element, 'connections': connections, 'actions': actions})
         processed_insertions.add(target_idx)

    unprocessed_inserts = new_atom_target_indices - processed_insertions
    if unprocessed_inserts:
        log_message(f"\nWARNING: {len(unprocessed_inserts)} insert_node operations could not be mapped (e.g., no connection to kept atoms found, or connects only to other new atoms):")
        # for idx in unprocessed_inserts: log_message(f"  - Target Index {idx}")

    return action_sequence


def main():
    """Main function to analyze and map GED operations."""
    log_message("Starting analysis...")
    transformations = load_transformation_data("train")
    if not transformations:
        log_message("No transformations found to analyze.")
        return

    shared_data = {}

    for i in range(min(len(transformations), MAX_TRANSFORMATIONS)):
        transformation = transformations[i]
        source_smiles = transformation.get('source_smiles')
        target_smiles = transformation.get('target_smiles')
        edit_path = transformation.get('edit_path')

        if not source_smiles or not target_smiles or edit_path is None:
             log_message(f"Skipping transformation {i+1} due to missing data.")
             continue

        log_message(f"\n\n==================================================")
        log_message(f"Analyzing transformation {i+1}/{min(len(transformations), MAX_TRANSFORMATIONS)}")
        log_message(f"Source: {source_smiles}")
        log_message(f"Target: {target_smiles}")
        log_message(f"==================================================\n")

        source_mol, target_mol = None, None
        try:
            source_mol = Chem.MolFromSmiles(source_smiles)
            if source_mol:
                source_order = list(rdmolfiles.CanonicalRankAtoms(source_mol))
                source_mol = rdmolops.RenumberAtoms(source_mol, source_order)

            target_mol = Chem.MolFromSmiles(target_smiles)
            if target_mol:
                target_order = list(rdmolfiles.CanonicalRankAtoms(target_mol))
                target_mol = rdmolops.RenumberAtoms(target_mol, target_order)
        except Exception as e:
             log_message(f"ERROR creating/renumbering RDKit molecules: {e}")
             continue

        if not source_mol or not target_mol:
            log_message("Error creating RDKit molecules from SMILES")
            continue

        operations = categorize_operations(list(edit_path))
        shared_data['operations'] = operations

        num_ops_list = [len(ops) for ops in [operations.get('substitutions', []),
                                             operations.get('deletions', []),
                                             operations.get('insertions', []),
                                             operations.get('edge_operations', [])]]
        log_message(f"Total operations from GED path: {sum(num_ops_list)} (Subs: {num_ops_list[0]}, Dels: {num_ops_list[1]}, Ins: {num_ops_list[2]}, Edges: {num_ops_list[3]})")

        fate_info = build_atom_fate_map(operations, source_mol, target_mol)
        shared_data['fate_info'] = fate_info

        graph_analysis = analyze_graph_structure(source_mol, fate_info['atom_fate'], operations)
        shared_data['graph_analysis'] = graph_analysis

        bond_analyses = analyze_bond_removals_efficient(graph_analysis, source_mol, fate_info, operations)
        shared_data['bond_analyses'] = bond_analyses

        action_sequence = map_operations_to_action_sequence_efficient(operations, source_mol, graph_analysis, bond_analyses, fate_info)
        shared_data['action_sequence'] = action_sequence

        log_message(f"\n--- Generated Action Sequence ({len(action_sequence)} high-level steps) ---")

    log_message("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()