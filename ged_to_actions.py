"""
GED to MoleculeDesign Mapping - Graph-based efficient fragmentation strategy.
Version: 2025-04-16 14:57:00 UTC (Fix Level 3 action mapping)
"""
import pickle
import os
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdmolfiles, rdmolops, AllChem
from collections import defaultdict, deque
import datetime # For timestamp

# --- User Info ---
CURRENT_USER = "mbinjavaid"
# --- End User Info ---


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
    for i, op in enumerate(edit_path):
        if op['operation'] == 'metadata':
            metadata = op
            edit_path = edit_path[:i]
            break

    edit_path = edit_path or []

    substitutions, deletions, insertions, edge_operations = [], [], [], []

    for op in edit_path:
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
        'insertions': insertions, 'edge_operations': edge_operations
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

    for i in range(num_atoms_source):
        if atom_fate[i] == "unknown":
            atom_fate[i] = "kept"
            log_message(f"Atom {i} implicitly kept but target index unknown (needs inference)")

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

    # Find initial Doomed Fragments
    visited_doomed = set()
    for i in range(num_atoms):
        if atom_fate.get(i) == "doomed" and i not in visited_doomed:
            component, q = set(), deque([i])
            while q:
                u = q.popleft()
                if u in visited_doomed or atom_fate.get(u) != "doomed": continue
                visited_doomed.add(u); component.add(u)
                for v in adj.get(u, []):
                    if v not in visited_doomed and atom_fate.get(v) == "doomed": q.append(v)
            if component: doomed_fragments.append(component)

    log_message(f"Identified {len(kept_components)} Kept Components (Sizes: {[len(c) for c in kept_components]}).")
    log_message(f"Identified {len(doomed_fragments)} initial Doomed Fragments (Sizes: {[len(f) for f in doomed_fragments]}).")

    critically_attached_single_doomed = set()
    all_doomed_in_fragments = set().union(*doomed_fragments) if doomed_fragments else set()
    for i in range(num_atoms):
        if atom_fate.get(i) == "doomed" and i not in all_doomed_in_fragments:
            neighbors = adj.get(i, [])
            neighbor_fates = [atom_fate.get(n) for n in neighbors]
            if neighbors and all(fate == "kept" for fate in neighbor_fates):
                critically_attached_single_doomed.add(i)
    if critically_attached_single_doomed:
        log_message(f"Identified {len(critically_attached_single_doomed)} single doomed atoms connected only to Kept components: {critically_attached_single_doomed}")
        for idx in critically_attached_single_doomed: doomed_fragments.append({idx})
        log_message(f"Total Doomed Fragments (including singles): {len(doomed_fragments)}")

    all_kept_atoms = set().union(*kept_components) if kept_components else set()

    critical_bonds = []
    delete_edge_ops = [op for op in operations.get('edge_operations', []) if op.get('operation') == 'delete_edge']
    for op in delete_edge_ops:
        a1, a2 = op.get('atom1_idx'), op.get('atom2_idx')
        if a1 is None or a2 is None or not (0 <= a1 < num_atoms and 0 <= a2 < num_atoms): continue

        fate1, fate2 = atom_fate.get(a1), atom_fate.get(a2)
        if (fate1 == "kept" and fate2 == "doomed") or (fate1 == "doomed" and fate2 == "kept"):
            kept_atom = a1 if fate1 == "kept" else a2
            doomed_atom = a1 if fate1 == "doomed" else a2
            fragment_involved = next((frag for frag in doomed_fragments if doomed_atom in frag), None)
            kept_component_involved = next((comp for comp in kept_components if kept_atom in comp), None)

            if fragment_involved is not None:
                 critical_bonds.append({
                     'bond': tuple(sorted((a1, a2))), 'kept_atom': kept_atom, 'doomed_atom': doomed_atom,
                     'doomed_fragment': fragment_involved, 'kept_component': kept_component_involved, 'original_op': op
                 })

    log_message(f"Identified {len(critical_bonds)} critical bonds connecting Kept/Doomed components.")

    internal_kept_deletions = []
    critical_bond_keys = {cb['bond'] for cb in critical_bonds}
    for op in delete_edge_ops:
         a1, a2 = op.get('atom1_idx'), op.get('atom2_idx')
         if a1 is None or a2 is None or not (0 <= a1 < num_atoms and 0 <= a2 < num_atoms): continue
         bond_key = tuple(sorted((a1, a2)))
         if atom_fate.get(a1) == "kept" and atom_fate.get(a2) == "kept" and bond_key not in critical_bond_keys:
             internal_kept_deletions.append(op)

    log_message(f"Identified {len(internal_kept_deletions)} internal bond deletions within Kept components.")

    return {
        'kept_components': kept_components, 'all_kept_atoms': all_kept_atoms, 'doomed_fragments': doomed_fragments,
        'critical_bonds': critical_bonds, 'internal_kept_deletions': internal_kept_deletions
    }


def analyze_fragment(fragment_indices, atom_fate):
    """Analyze a fragment based on atom fates (indices only)."""
    if not hasattr(fragment_indices, '__iter__'): return {"kept_count": 0, "should_discard": True}
    kept_count = sum(1 for idx in fragment_indices if atom_fate.get(idx) == "kept")
    return {"kept_count": kept_count, "should_discard": kept_count == 0}


def analyze_bond_removals_efficient(graph_analysis, source_mol, fate_info, operations):
    """Analyzes bond removals. Failed criticals treated as internal."""
    log_message("\n=== Analyzing Bond Removals (Efficient Strategy) ===")
    critical_bonds = graph_analysis.get('critical_bonds', [])
    internal_kept_deletions = graph_analysis.get('internal_kept_deletions', [])
    atom_fate = fate_info.get('atom_fate', {})
    bond_analyses, processed_bond_keys = [], set()
    working_mol = Chem.Mol(source_mol)

    log_message(f"\n--- Processing {len(critical_bonds)} Critical Bonds ---")
    for i, cb_info in enumerate(critical_bonds):
        op, bond_key = cb_info['original_op'], cb_info['bond']
        atom1_idx, atom2_idx = bond_key
        original_doomed_atom = cb_info.get('doomed_atom')

        if bond_key in processed_bond_keys or original_doomed_atom is None: continue
        log_message(f"\nCritical Bond {i + 1}: atoms {atom1_idx}-{atom2_idx} (Targeting doomed {original_doomed_atom})")

        bond = working_mol.GetBondBetweenAtoms(atom1_idx, atom2_idx)
        if bond is None: log_message(f"  WARNING: Bond {bond_key} not found. Skipping."); continue

        # --- Store previous component count ---
        prev_components = []
        try: prev_components = Chem.GetMolFrags(working_mol, asMols=False, sanitizeFrags=False)
        except Exception as e: log_message(f"  ERROR: GetMolFrags failed BEFORE critical removal: {e}. Skipping."); continue
        # ---

        rwmol = Chem.RWMol(working_mol)
        rwmol.RemoveBond(atom1_idx, atom2_idx)
        modified_mol = None
        try: Chem.SanitizeMol(rwmol); modified_mol = Chem.Mol(rwmol)
        except Exception as e: log_message(f"  WARNING: Sanitization failed: {e}. Proceeding unsanitized."); modified_mol = rwmol.GetMol()
        if modified_mol is None: log_message(f"  ERROR: Mol creation failed. Skipping."); continue

        new_components = []
        try: new_components = Chem.GetMolFrags(modified_mol, asMols=False, sanitizeFrags=False)
        except Exception as e: log_message(f"  ERROR: GetMolFrags failed AFTER critical removal: {e}. Skipping analysis."); working_mol = modified_mol; processed_bond_keys.add(bond_key); continue
        log_message(f"  After removal, molecule has {len(new_components)} components (previously {len(prev_components)}).")

        doomed_frag_num, identified_doomed_indices = -1, set()
        for frag_idx, frag_comp in enumerate(new_components):
            if frag_comp and hasattr(frag_comp, '__iter__') and original_doomed_atom in frag_comp:
                doomed_frag_num, identified_doomed_indices = frag_idx, set(frag_comp)
                log_message(f"  Identified Component {frag_idx} (Size: {len(frag_comp)}) contains doomed atom {original_doomed_atom}.")
                break

        analysis_added = False
        if doomed_frag_num != -1:
            frag_analysis = analyze_fragment(identified_doomed_indices, atom_fate)
            if frag_analysis['should_discard']: # SUCCESS - Attempt to map to env actions 0 or 1
                env_level3_action = -1 # Default invalid action
                # --- Check if exactly two fragments resulted ---
                if len(new_components) == len(prev_components) + 1 and len(new_components) == 2:
                    if doomed_frag_num == 0:
                        env_level3_action = 1 # Keep fragment 1
                        log_message(f"  Verified fragment {doomed_frag_num} is purely doomed. Level 3 Env Action: {env_level3_action} (Keep Fragment 1)")
                    elif doomed_frag_num == 1:
                        env_level3_action = 0 # Keep fragment 0
                        log_message(f"  Verified fragment {doomed_frag_num} is purely doomed. Level 3 Env Action: {env_level3_action} (Keep Fragment 0)")
                    else: # Should not happen if doomed_frag_num is 0 or 1 and len is 2
                         log_message(f"  ERROR: doomed_frag_num ({doomed_frag_num}) out of range for 2 fragments. Defaulting to keep both (Action 2).")
                         env_level3_action = 2
                else:
                    log_message(f"  WARNING: Critical bond removal resulted in {len(new_components)} fragments (expected 2). Cannot map to specific discard action. Defaulting to keep both (Action 2).")
                    env_level3_action = 2 # Fallback action

                if env_level3_action != -1: # Only proceed if we determined a valid action (0, 1, or 2)
                    bond_analyses.append({
                        'bond': bond_key,
                        'type': 'critical_disconnection',
                        'creates_fragments': True, # It created fragments, even if > 2
                        'level3_action': env_level3_action, # STORE THE CORRECT ENV ACTION (0, 1, or 2)
                        'original_op': op,
                        'kept_atom': cb_info.get('kept_atom'),
                        'doomed_atom': original_doomed_atom
                    })
                    analysis_added = True
            else: # FAILED VERIFICATION (fragment contained kept atoms)
                log_message(f"  ERROR: Component {doomed_frag_num} also has {frag_analysis['kept_count']} kept atoms. Treating as internal (keep both).")
                # Treat as internal, level 3 action is 2
                bond_analyses.append({'bond': bond_key, 'type': 'internal_kept_core', 'creates_fragments': True, 'level3_action': 2, 'original_op': op })
                analysis_added = True
        else: # Doomed atom not found in any fragment after removal
             log_message(f"  ERROR: Could not find component containing doomed atom {original_doomed_atom}. Treating as internal (keep both).")
             # Also treat as internal failure case, assume it didn't disconnect if we can't find the doomed atom fragment
             bond_analyses.append({'bond': bond_key, 'type': 'internal_kept_core', 'creates_fragments': False, 'original_op': op })
             analysis_added = True # Mark as processed anyway

        if analysis_added: processed_bond_keys.add(bond_key)
        working_mol = modified_mol

    log_message(f"\n--- Processing {len(internal_kept_deletions)} initially classified Internal Kept Core Bonds ---")
    # ... (rest of the function remains the same) ...
    for i, op in enumerate(internal_kept_deletions):
        atom1_idx, atom2_idx = op.get('atom1_idx'), op.get('atom2_idx')
        if atom1_idx is None or atom2_idx is None: continue
        bond_key = tuple(sorted((atom1_idx, atom2_idx)))
        if bond_key in processed_bond_keys: continue
        log_message(f"\nInternal Bond {i + 1}: atoms {atom1_idx}-{atom2_idx}")

        bond = working_mol.GetBondBetweenAtoms(atom1_idx, atom2_idx)
        if bond is None: log_message(f"  WARNING: Bond {bond_key} not found. Skipping."); continue

        prev_components = []
        try: prev_components = Chem.GetMolFrags(working_mol, asMols=False, sanitizeFrags=False)
        except Exception as e: log_message(f"  ERROR: GetMolFrags failed BEFORE internal removal: {e}. Skipping."); continue

        rwmol = Chem.RWMol(working_mol)
        rwmol.RemoveBond(atom1_idx, atom2_idx)
        modified_mol = None
        try: Chem.SanitizeMol(rwmol); modified_mol = Chem.Mol(rwmol)
        except Exception as e: log_message(f"  WARNING: Sanitization failed: {e}. Proceeding unsanitized."); modified_mol = rwmol.GetMol()
        if modified_mol is None: log_message(f"  ERROR: Mol creation failed. Skipping."); continue

        new_components = []
        try: new_components = Chem.GetMolFrags(modified_mol, asMols=False, sanitizeFrags=False)
        except Exception as e: log_message(f"  ERROR: GetMolFrags failed AFTER internal removal: {e}. Skipping analysis."); working_mol = modified_mol; processed_bond_keys.add(bond_key); continue

        analysis = {'bond': bond_key, 'type': 'internal_kept_core', 'creates_fragments': False, 'original_op': op}
        if len(new_components) > len(prev_components):
            analysis['creates_fragments'] = True
            log_message("  Internal bond removal created fragments.")
            level3_action = 2 # Default keep both for internal removals unless proven otherwise (though GED shouldn't specify discard here)

            # --- Refined check for internal fragment analysis (optional but safer) ---
            # Generally, for 'internal_kept_core' based on GED, we expect to keep both parts
            # unless the GED somehow implied a discard which would be unusual.
            # Sticking with default action 2 for internal removals seems safest.
            # The complex fragment analysis logic here might be unnecessary if we trust the initial classification.
            # Keeping it for now, but simplifying to always use action 2 might be valid.
            if len(new_components) == len(prev_components) + 1 and len(new_components) == 2:
                 frag1_indices, frag2_indices = set(), set()
                 found_frags = 0
                 for frag_comp in new_components:
                     if frag_comp and hasattr(frag_comp, '__iter__'):
                         frag_set = set(frag_comp)
                         if atom1_idx in frag_set or atom2_idx in frag_set:
                             if found_frags == 0: frag1_indices = frag_set
                             elif found_frags == 1: frag2_indices = frag_set
                             found_frags += 1

                 if frag1_indices and frag2_indices:
                     frag1_analysis = analyze_fragment(frag1_indices, atom_fate)
                     frag2_analysis = analyze_fragment(frag2_indices, atom_fate)
                     # Even if one fragment is purely doomed (which shouldn't happen for internal_kept_core),
                     # the safest action based on 'internal' classification is likely 'keep both'.
                     level3_action = 2 # Override based on 'internal' classification
                     log_message(f"  Internal split into 2 fragments. Defaulting Level 3: keep both (Action {level3_action}).")
                 else:
                     log_message("  WARNING: Internal removal created 2 fragments, but couldn't identify them reliably. Defaulting Level 3: keep both (Action 2).")
                     level3_action = 2
            elif len(new_components) > len(prev_components): # More complex fragmentation
                 log_message(f"  WARNING: Internal removal caused complex fragmentation ({len(prev_components)}->{len(new_components)}). Defaulting Level 3: keep both (Action 2).")
                 level3_action = 2
            # --- End refined check ---

            analysis['level3_action'] = level3_action
        else: log_message("  Internal bond removal did not create fragments.")

        bond_analyses.append(analysis)
        processed_bond_keys.add(bond_key)
        working_mol = modified_mol

    all_delete_ops = [tuple(sorted((op.get('atom1_idx'), op.get('atom2_idx')))) for op in operations.get('edge_operations', []) if op.get('operation') == 'delete_edge' and op.get('atom1_idx') is not None and op.get('atom2_idx') is not None]
    unprocessed_deletions = set(all_delete_ops) - processed_bond_keys
    if unprocessed_deletions: log_message(f"\nWARNING: {len(unprocessed_deletions)} delete_edge operations were not processed.")

    return bond_analyses


def map_operations_to_action_sequence_efficient(operations, source_mol, graph_analysis, bond_analyses, fate_info):
    """Maps operations to actions with heuristic ordering and multi-pass insertions."""
    log_message("\n=== Mapping Operations to Action Sequences (Efficient Strategy) ===")

    all_kept_atoms = graph_analysis.get('all_kept_atoms', set())
    atom_fate = fate_info.get('atom_fate', {})
    num_atoms_source = source_mol.GetNumAtoms()

    global VOCAB_SIZE, BOND_REMOVE_ACTION
    ACTION_REPLACE_ATOM = VOCAB_SIZE + num_atoms_source

    action_sequence = []

    def get_bond_action(rdkit_bond_type_or_val):
        if isinstance(rdkit_bond_type_or_val, (int, float)):
             bond_type_map = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE, 1.5: Chem.BondType.AROMATIC}
             rdkit_bond_type = bond_type_map.get(rdkit_bond_type_or_val, Chem.BondType.SINGLE)
        else:
             rdkit_bond_type = rdkit_bond_type_or_val
        action = RDKIT_BOND_TYPE_TO_ACTION.get(rdkit_bond_type)
        if action is None:
             log_message(f"  WARNING: RDKit bond type {rdkit_bond_type} not in action map. Defaulting to SINGLE bond action.")
             action = RDKIT_BOND_TYPE_TO_ACTION.get(Chem.BondType.SINGLE)
        return int(action)

    def get_vocab_index(element_atomic_num):
        element_atomic_num_str = str(element_atomic_num)
        fallback_index = ELEMENT_TO_VOCAB.get('?')
        index = ELEMENT_TO_VOCAB.get(element_atomic_num_str, fallback_index)
        if index == fallback_index and element_atomic_num_str not in ELEMENT_TO_VOCAB:
             log_message(f"  WARNING: Atomic number '{element_atomic_num_str}' not in vocab mapping. Using fallback.")
        return int(index)

    def generate_add_bond_actions(atom_env_idx1, atom_env_idx2, bond_type_val):
        level0_action = atom_env_idx1 + 1
        level1_action = VOCAB_SIZE + atom_env_idx2
        level2_action = get_bond_action(bond_type_val)
        action_tuple = (level0_action, level1_action, level2_action)
        log_message(f"    Add/Set Bond Action Tuple: {action_tuple} (env_idx {atom_env_idx1} - env_idx {atom_env_idx2}, Type {bond_type_val})")
        return action_tuple

    # --- 1. Doomed Fragment Removals (Successful Discards) ---
    log_message("\n-- Step 1: Doomed Fragment Removals (Successful Discards) --")
    critical_bond_actions = [ba for ba in bond_analyses if ba.get('type') == 'critical_disconnection']
    for ba in critical_bond_actions:
        bond = ba.get('bond')
        # Use the corrected env_level3_action (0, 1, or 2) stored in bond_analyses
        env_level3_action = ba.get('level3_action', -1)
        kept_atom_idx, doomed_atom_idx = ba.get('kept_atom'), ba.get('doomed_atom')

        if bond is None or env_level3_action == -1 or kept_atom_idx is None or doomed_atom_idx is None: continue

        log_message(f"Remove Doomed Fragment via bond {kept_atom_idx}-{doomed_atom_idx}:")
        level0_action = kept_atom_idx + 1
        level1_action = VOCAB_SIZE + doomed_atom_idx
        level2_action = BOND_REMOVE_ACTION
        level3_action = int(env_level3_action) # Already 0, 1, or 2
        action_tuple = (level0_action, level1_action, level2_action, level3_action)
        log_message(f"  Action Tuple: {action_tuple}")
        # Store original level3 value if needed for context, but tuple uses corrected one
        action_sequence.append({'type': 'remove_doomed_fragment', 'bond': bond, 'level3': env_level3_action, 'action_tuple': action_tuple})

    # --- 2. Internal Kept Core Bond Edits (incl. Failed Discards) ---
    log_message("\n-- Step 2: Internal Kept Core Bond Edits (incl. Failed Discards) --")
    internal_bond_actions = [ba for ba in bond_analyses if ba.get('type') == 'internal_kept_core']
    for ba in internal_bond_actions:
        bond = ba.get('bond')
        if bond is None: continue
        atom1_idx, atom2_idx = bond
        creates_fragments = ba.get('creates_fragments', False)
        # Use the level3_action stored (should be 2 if fragments created, or None/-1 otherwise)
        level3_action_val = ba.get('level3_action', -1)
        log_message(f"Internal Bond removal {atom1_idx}-{atom2_idx}:")
        level0_action = atom1_idx + 1
        level1_action = VOCAB_SIZE + atom2_idx
        level2_action = BOND_REMOVE_ACTION
        action_tuple = None
        if creates_fragments:
            level3_action = int(level3_action_val) # Should be 2
            action_tuple = (level0_action, level1_action, level2_action, level3_action)
            level3_text = {0: "Discard fragment 0", 1: "Discard fragment 1", 2: "Keep both fragments"}.get(level3_action, "Keep both fragments")
            log_message(f"  Action Tuple (with L3={level3_text}): {action_tuple}")
        else:
            action_tuple = (level0_action, level1_action, level2_action)
            log_message(f"  Action Tuple: {action_tuple}")
        action_sequence.append({'type': 'internal_bond_removal', 'bond': bond, 'level3': level3_action_val if creates_fragments else None, 'action_tuple': action_tuple})

    # --- 3. Atom Substitutions ---
    log_message("\n-- Step 3: Atom Substitutions --")
    substitutions_ops = operations.get('substitutions', [])
    for op in substitutions_ops:
        source_idx = op.get('source_idx')
        if source_idx is not None and source_idx in all_kept_atoms:
            to_element_num = op.get('to_element', '?')
            from_element_num = op.get('from_element', '?')
            log_message(f"Substitute Atom {source_idx} ({from_element_num} -> {to_element_num}):")
            level0_action = source_idx + 1
            level1_action = ACTION_REPLACE_ATOM
            level2_action = get_vocab_index(to_element_num)
            action_tuple = (level0_action, level1_action, level2_action)
            log_message(f"  Action Tuple: {action_tuple}")
            action_sequence.append({'type': 'substitute_atom', 'source_idx': source_idx, 'to_element': str(to_element_num), 'action_tuple': action_tuple})
        elif source_idx is not None: log_message(f"Skipping substitution for doomed atom {source_idx}.")

    # --- 4. Atom Insertions (Multi-Pass) ---
    log_message("\n-- Step 4: Atom Insertions (Multi-Pass) --")
    insertions_ops = operations.get('insertions', [])
    edge_insertions = [op for op in operations.get('edge_operations', []) if op.get('operation') == 'insert_edge']

    insertions_grouped = defaultdict(lambda: {'op': None, 'connections_to_kept': [], 'connections_to_new': []})
    new_atom_target_indices = {op.get('target_idx') for op in insertions_ops if op.get('target_idx') is not None}

    log_message(f"Found {len(new_atom_target_indices)} new atom target indices: {new_atom_target_indices}")

    for op in insertions_ops:
         target_idx = op.get('target_idx')
         if target_idx is not None: insertions_grouped[target_idx]['op'] = op

    log_message(f"Processing {len(edge_insertions)} insert_edge ops for connections...")
    for edge_op in edge_insertions:
         a1, a2 = edge_op.get('atom1_idx'), edge_op.get('atom2_idx')
         bond_type = edge_op.get('bond_type', 1)
         if a1 is None or a2 is None: continue

         conn_info = {'bond_type': bond_type}
         if a1 in new_atom_target_indices and a2 in new_atom_target_indices:
              insertions_grouped[a1]['connections_to_new'].append({'to_atom': a2, **conn_info})
              insertions_grouped[a2]['connections_to_new'].append({'to_atom': a1, **conn_info})
         elif a1 in new_atom_target_indices and a2 not in new_atom_target_indices:
              if a2 in all_kept_atoms:
                   insertions_grouped[a1]['connections_to_kept'].append({'to_atom': a2, **conn_info})
              else: log_message(f"  Warning: New atom {a1} connects to non-kept existing atom {a2}. Ignoring.")
         elif a2 in new_atom_target_indices and a1 not in new_atom_target_indices:
              if a1 in all_kept_atoms:
                   insertions_grouped[a2]['connections_to_kept'].append({'to_atom': a1, **conn_info})
              else: log_message(f"  Warning: New atom {a2} connects to non-kept existing atom {a1}. Ignoring.")

    mapped_new_atom_ids = {}
    current_atom_count = num_atoms_source
    env_idx_map = {}

    unmapped_insertions = set(new_atom_target_indices)
    pass_num = 0
    while True:
        pass_num += 1
        log_message(f"\n--- Insertion Pass {pass_num} ---")
        newly_mapped_in_pass = set()

        sorted_unmapped = sorted(list(unmapped_insertions))

        for target_idx in sorted_unmapped:
            data = insertions_grouped[target_idx]
            op = data['op']
            element_num = op.get('element', '?')
            conns_kept = data['connections_to_kept']
            conns_new = data['connections_to_new']

            anchor_info = None

            if conns_kept:
                first_conn = conns_kept[0]
                anchor_info = ('kept', first_conn['to_atom'], first_conn['bond_type'])
                log_message(f"  Trying to map {target_idx} ({element_num}) via kept anchor {anchor_info[1]}")
            else:
                for conn_new in conns_new:
                    anchor_new_target_idx = conn_new['to_atom']
                    if anchor_new_target_idx in mapped_new_atom_ids:
                        anchor_info = ('new', anchor_new_target_idx, conn_new['bond_type'])
                        log_message(f"  Trying to map {target_idx} ({element_num}) via new anchor (target_idx={anchor_new_target_idx})")
                        break

            if anchor_info:
                anchor_type, anchor_original_or_target_idx, bond_type_val = anchor_info

                if anchor_type == 'kept':
                    anchor_env_idx = anchor_original_or_target_idx
                else:
                    anchor_env_idx = env_idx_map.get(anchor_original_or_target_idx)
                    if anchor_env_idx is None:
                         log_message(f"  ERROR: Anchor new atom {anchor_original_or_target_idx} not found in env_idx_map. Skipping {target_idx}.")
                         continue

                log_message(f"  Mapping Add Atom {target_idx} ({element_num}) anchored to {anchor_type} atom (env_idx={anchor_env_idx})")

                level0_action = anchor_env_idx + 1
                level1_action = get_vocab_index(element_num)
                level2_action = get_bond_action(bond_type_val)
                add_atom_tuple = (level0_action, level1_action, level2_action)
                log_message(f"    Action Tuple: {add_atom_tuple}")
                action_sequence.append({'type': 'add_atom', 'target_idx': target_idx, 'element': str(element_num), 'anchor_env_idx': anchor_env_idx, 'action_tuple': add_atom_tuple})

                placeholder_id = f"newly_added_{target_idx}"
                mapped_new_atom_ids[target_idx] = placeholder_id
                new_atom_env_idx = current_atom_count
                env_idx_map[target_idx] = new_atom_env_idx
                current_atom_count += 1

                newly_mapped_in_pass.add(target_idx)

                for conn_kept in conns_kept[1:] if anchor_type == 'kept' else conns_kept:
                    other_kept_env_idx = conn_kept['to_atom']
                    add_bond_tuple = generate_add_bond_actions(new_atom_env_idx, other_kept_env_idx, conn_kept['bond_type'])
                    action_sequence.append({'type': 'add_bond', 'from_env_idx': new_atom_env_idx, 'to_env_idx': other_kept_env_idx, 'action_tuple': add_bond_tuple})

                for conn_new in conns_new:
                    other_new_target_idx = conn_new['to_atom']
                    if other_new_target_idx in env_idx_map and other_new_target_idx != anchor_original_or_target_idx :
                       other_new_env_idx = env_idx_map[other_new_target_idx]
                       add_bond_tuple = generate_add_bond_actions(new_atom_env_idx, other_new_env_idx, conn_new['bond_type'])
                       action_sequence.append({'type': 'add_bond', 'from_env_idx': new_atom_env_idx, 'to_env_idx': other_new_env_idx, 'action_tuple': add_bond_tuple})

            else:
                connects_only_to_unmapped_new = False
                if not conns_kept and conns_new:
                    all_connected_new_are_unmapped = True
                    for conn_new in conns_new:
                        if conn_new['to_atom'] in mapped_new_atom_ids:
                            all_connected_new_are_unmapped = False; break
                    if all_connected_new_are_unmapped: connects_only_to_unmapped_new = True

                if connects_only_to_unmapped_new:
                     log_message(f"  Atom {target_idx} ({element_num}) only connects to other UNMAPPED new atoms in pass {pass_num}. Deferring.")
                else:
                     log_message(f"  Could not find anchor for {target_idx} ({element_num}) in pass {pass_num}")

        if not newly_mapped_in_pass:
            log_message(f"--- No insertions mapped in pass {pass_num}. Exiting loop. ---")
            break

        unmapped_insertions -= newly_mapped_in_pass
        log_message(f"--- Mapped {len(newly_mapped_in_pass)} atoms in pass {pass_num}. Remaining: {len(unmapped_insertions)} ---")
        if not unmapped_insertions:
             log_message("--- All insertions mapped. ---")
             break

    if unmapped_insertions:
        log_message(f"\nWARNING: {len(unmapped_insertions)} insert_node operations could not be mapped after {pass_num} passes:")
        for idx in sorted(list(unmapped_insertions)): log_message(f"  - Target Index {idx} ({insertions_grouped[idx]['op'].get('element', '?')})")

    return action_sequence


def main():
    """Main function to analyze and map GED operations."""
    log_message(f"Starting analysis run by {CURRENT_USER} at {datetime.datetime.utcnow().isoformat()}Z")
    transformations = load_transformation_data("train")
    if not transformations: return

    results = []

    for i in range(min(len(transformations), MAX_TRANSFORMATIONS)):
        transformation = transformations[i]
        source_smiles, target_smiles = transformation.get('source_smiles'), transformation.get('target_smiles')
        edit_path = transformation.get('edit_path')

        if not source_smiles or not target_smiles or edit_path is None:
             log_message(f"Skipping transformation {i+1} due to missing data."); continue

        log_message(f"\n\n==================================================")
        log_message(f"Analyzing transformation {i+1}/{min(len(transformations), MAX_TRANSFORMATIONS)}")
        log_message(f"Source: {source_smiles}")
        log_message(f"Target: {target_smiles}")
        log_message(f"==================================================\n")

        source_mol, target_mol = None, None
        try:
            source_mol = Chem.MolFromSmiles(source_smiles)
            if source_mol: source_mol = rdmolops.RenumberAtoms(source_mol, list(rdmolfiles.CanonicalRankAtoms(source_mol)))
            target_mol = Chem.MolFromSmiles(target_smiles)
            if target_mol: target_mol = rdmolops.RenumberAtoms(target_mol, list(rdmolfiles.CanonicalRankAtoms(target_mol)))
        except Exception as e: log_message(f"ERROR creating/renumbering RDKit molecules: {e}"); continue
        if not source_mol or not target_mol: log_message("Error creating RDKit molecules from SMILES"); continue

        operations = categorize_operations(list(edit_path))
        fate_info = build_atom_fate_map(operations, source_mol, target_mol)
        graph_analysis = analyze_graph_structure(source_mol, fate_info['atom_fate'], operations)
        bond_analyses = analyze_bond_removals_efficient(graph_analysis, source_mol, fate_info, operations) # Corrected version
        action_sequence = map_operations_to_action_sequence_efficient(operations, source_mol, graph_analysis, bond_analyses, fate_info)

        log_message(f"\n--- Generated Action Sequence ({len(action_sequence)} items) ---")

        results.append({
            'transformation_index': i,
            'source_smiles': source_smiles,
            'target_smiles': target_smiles,
            'action_sequence': action_sequence
        })

    log_message("\n=== Analysis Complete ===")

    # output_filename = f"ged_to_action_tuples_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pkl"
    # with open(output_filename, "wb") as f:
    #      pickle.dump(results, f)
    # log_message(f"Saved action sequence results to {output_filename}")


if __name__ == "__main__":
    main()