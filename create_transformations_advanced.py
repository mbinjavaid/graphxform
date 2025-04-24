import time
import pickle
import os
import random # Ensure random is imported
import numpy as np
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional, Set, Any
from datetime import datetime
import copy
import heapq
from multiprocessing import Pool, cpu_count
import itertools
import sys # Needed for Butina progress printing
import traceback # For detailed exception logging in A*

# --- Global Debug Flags ---
# Controls overall debug mode (sampling, limits)
DEBUG_MODE = True # <<<<<<< ENABLE DEBUG MODE FOR SINGLE PAIR RUN
# Controls specific A* search internal printing
DEBUG_ASTAR = False # <<<<<<< ENABLE DETAILED A* PRINTING
# --- End Global Debug Flags ---

# Turn off RDKit warnings
RDLogger.DisableLog('rdApp.*')

# --- Import Custom Modules ---
try:
    from config import MoleculeConfig
    # Assuming ActionType might be defined in molecule_design or accessible
    from molecule_design import MoleculeDesign, ActionType, build_reverse_atom_lookup
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print("Please ensure config.py and molecule_design.py are in the correct path.")
    print("Ensure MoleculeDesign class and potentially ActionType enum are defined and accessible.")
    exit(1)

# --- Configuration ---
try:
    CONFIG = MoleculeConfig()
except Exception as e:
    print(f"Error loading MoleculeConfig: {e}")
    exit(1)

# --- Script Parameters ---
# Debugging
DEBUG_PAIRING_LIMIT = 10000 # Affects sampling if DEBUG_MODE is on
DEBUG_ASTAR_LIMIT = 1 # Process only one pair in debug mode
# Sampling Parameter
SAMPLE_SIZE = 10000  # Sample size used when DEBUG_MODE is False
# Pairing & Clustering
SIMILARITY_THRESHOLD = 0.7
FP_RADIUS = 2
FP_BITS = 2048
# A* Search
MAX_SEARCH_STEPS = 1000 # Use the high limit for debugging
# Data Paths
MAX_ATOMS = CONFIG.max_num_atoms
RANDOM_SEED = CONFIG.seed
CHECKPOINT_DIR = "./data/chembl/checkpoints_astar" # Checkpoints for THIS script
RESULTS_DIR = "./data/chembl/astar_datasets"
CHEMBL_TRAIN_PATH = "./data/chembl/chembl_train_filtered.smiles"
CHEMBL_VALID_PATH = "./data/chembl/chembl_valid_filtered.smiles"
# Checkpoint Filenames within CHECKPOINT_DIR
FPS_CHECKPOINT_FILENAME_TPL = f"fps_r{FP_RADIUS}_b{FP_BITS}_sample{{}}.pkl"
PAIRS_CHECKPOINT_FILENAME_TPL = f"pairs_thresh{SIMILARITY_THRESHOLD}_sample{{}}.pkl"
# Output
FINAL_RESULTS_FILENAME_BASE = "astar_clustered_sequences"

# Set random seed
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Ensure directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Molecule Loading (Unchanged) ---
def load_and_filter_molecules(path: str, max_atoms: int = MAX_ATOMS, datatype: str = "unknown") -> List[str]:
    """Loads SMILES, filters by atom count, returns unique canonical SMILES list."""
    base_checkpoint_dir = "./data/chembl/checkpoints_transformations" # Use base checkpoint dir
    checkpoint_path = os.path.join(base_checkpoint_dir, f"filtered_molecules_{datatype}.pkl")
    if os.path.exists(checkpoint_path):
        print(f"Loading filtered {datatype} molecules from checkpoint {checkpoint_path}")
        try:
            with open(checkpoint_path, "rb") as f:
                filtered_smiles: List[str] = pickle.load(f)
            print(f"Loaded {len(filtered_smiles)} {datatype} molecules from checkpoint")
            return filtered_smiles
        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Processing from scratch.")

    print(f"Loading and filtering molecules from {path}")
    filtered_smiles_list: List[str] = []
    processed_smiles: Set[str] = set()
    if not os.path.exists(path):
        print(f"Error: Input SMILES file not found at {path}")
        return []
    try:
        with open(path) as f:
            try: total_lines = sum(1 for line in open(path))
            except: total_lines = None

            f.seek(0)
            for line in tqdm(f, desc=f"Filtering {datatype} molecules", total=total_lines):
                smiles = line.strip()
                if not smiles: continue
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None: continue
                    try: Chem.SanitizeMol(mol)
                    except Exception: continue

                    num_heavy = mol.GetNumHeavyAtoms()
                    if num_heavy == 0 or num_heavy > max_atoms: continue

                    try:
                        Chem.Kekulize(mol)
                        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                    except Exception: continue

                    if canonical_smiles in processed_smiles: continue

                    mol_check = Chem.MolFromSmiles(canonical_smiles)
                    if mol_check is None: continue
                    num_heavy_check = mol_check.GetNumHeavyAtoms()
                    if num_heavy_check == 0 or num_heavy_check > max_atoms: continue

                    filtered_smiles_list.append(canonical_smiles)
                    processed_smiles.add(canonical_smiles)
                except Exception: continue
    except FileNotFoundError:
         print(f"Error: Input SMILES file not found at {path}")
         return []
    except Exception as e:
         print(f"An unexpected error occurred during file processing: {e}")
         return []


    os.makedirs(base_checkpoint_dir, exist_ok=True)
    try:
        with open(checkpoint_path, "wb") as f:
            pickle.dump(filtered_smiles_list, f)
        print(f"Saved {len(filtered_smiles_list)} filtered {datatype} molecules to checkpoint {checkpoint_path}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
    return filtered_smiles_list


# --- Fingerprint & Heuristic Helpers (Unchanged) ---
def calculate_morgan_fp(mol_or_smiles, radius=FP_RADIUS, nBits=FP_BITS):
    if mol_or_smiles is None: return None
    mol = mol_or_smiles
    if isinstance(mol_or_smiles, str):
        mol = Chem.MolFromSmiles(mol_or_smiles)
    if mol is None: return None
    try: Chem.Kekulize(mol)
    except: pass
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return fp
    except Exception: return None

def heuristic_tanimoto(current_mol_design, target_fp, radius=FP_RADIUS, nBits=FP_BITS):
    if target_fp is None: return float('inf')
    try:
        current_smiles = current_mol_design._get_smiles_for_check()
        if current_smiles is None: return float('inf')
        if not current_smiles:
            return 0.0 if target_fp.GetNumOnBits() == 0 else 1.0

        current_fp = calculate_morgan_fp(current_smiles, radius, nBits)
        if current_fp is None: return float('inf')

        if not isinstance(target_fp, DataStructs.ExplicitBitVect):
             return float('inf')

        similarity = DataStructs.TanimotoSimilarity(current_fp, target_fp)
        similarity = max(0.0, min(1.0, similarity))
        distance = 1.0 - similarity
        return distance
    except Exception: return float('inf')


# --- Helper to create the state key tuple ---
def get_state_key(state_object: 'MoleculeDesign', smiles: Optional[str]) -> Optional[Tuple]:
    """Generates the unique state key tuple for the A* search."""
    if smiles is None:
        if DEBUG_ASTAR: print("DEBUG get_state_key: Received None SMILES, returning None key.")
        return None

    try:
        # Safely get attributes, defaulting to None if they don't exist or aren't set
        level = getattr(state_object, 'current_action_level', None)
        l0_idx = getattr(state_object, 'l0_selected_atom_idx', None)
        l1_type = getattr(state_object, 'l1_action_type', None)
        l1_target = getattr(state_object, 'l1_selected_existing_atom_idx', None)
        l1_new = getattr(state_object, 'l1_new_atom_type', None) # This might be vocab index

        # Ensure l1_type is hashable (e.g., convert enum to value/name if needed)
        # Check if ActionType exists and l1_type is an instance of it
        if 'ActionType' in globals() and isinstance(l1_type, ActionType):
             l1_type = l1_type.value # Use the enum value (assuming it's hashable like int/str)
        elif l1_type is not None and not isinstance(l1_type, (int, str, type(None), float, bool)):
             # Fallback if it's not an expected hashable type or known enum
             if DEBUG_ASTAR: print(f"DEBUG get_state_key: Converting non-hashable l1_type ({type(l1_type)}) to string.")
             l1_type = str(l1_type)

        key = (smiles, level, l0_idx, l1_type, l1_target, l1_new)
        # Optional: Check hashability early
        # hash(key)
        return key

    except Exception as e:
        if DEBUG_ASTAR: print(f"DEBUG get_state_key: EXCEPTION generating key: {e}")
        # traceback.print_exc() # Uncomment for full traceback if needed
        return None


# --- A* Search Function (Using State Key Tuple and DEBUG_ASTAR flag) ---
def a_star_search(smiles_A, smiles_B, config, max_search_steps=MAX_SEARCH_STEPS, fp_radius=FP_RADIUS, fp_bits=FP_BITS):
    """
    Performs A* search using a comprehensive state key tuple.
    Debug prints controlled by DEBUG_ASTAR flag.
    """
    try:
        # --- Initial state setup ---
        initial_mol_design, _ = MoleculeDesign.from_smiles(config, smiles_A)
        if initial_mol_design is None or getattr(initial_mol_design, 'synthesis_done', False):
            if DEBUG_ASTAR: print(f"DEBUG A*: Invalid initial state for {smiles_A}")
            return None

        # --- Target state setup ---
        target_mol = Chem.MolFromSmiles(smiles_B)
        if target_mol is None:
            if DEBUG_ASTAR: print(f"DEBUG A*: Invalid target SMILES {smiles_B}")
            return None
        try:
            Chem.Kekulize(target_mol)
            target_smiles_canonical = Chem.MolToSmiles(target_mol, canonical=True)
        except Exception:
            if DEBUG_ASTAR: print(f"DEBUG A*: Failed to canonicalize target {smiles_B}")
            return None
        target_fp = calculate_morgan_fp(target_mol, fp_radius, fp_bits)
        if target_fp is None:
             if DEBUG_ASTAR: print(f"DEBUG A*: Failed to get fingerprint for target {smiles_B}")
             return None

        # --- Start state key and check ---
        start_smiles_canonical = initial_mol_design._get_smiles_for_check()
        if start_smiles_canonical is None:
            if DEBUG_ASTAR: print(f"DEBUG A*: Failed to get SMILES for initial state {smiles_A}")
            return None
        if start_smiles_canonical == target_smiles_canonical:
             if DEBUG_ASTAR: print(f"DEBUG A*: Start SMILES is already target SMILES.")
             return smiles_A, smiles_B, []

        start_state_key = get_state_key(initial_mol_design, start_smiles_canonical)
        if start_state_key is None:
            if DEBUG_ASTAR: print(f"DEBUG A*: Failed to generate initial state key.")
            return None

        # --- A* Initialization ---
        initial_h_score = heuristic_tanimoto(initial_mol_design, target_fp, fp_radius, fp_bits)
        if initial_h_score == float('inf'):
            if DEBUG_ASTAR: print(f"DEBUG A*: Initial heuristic is infinity for {smiles_A}")
            return None

        pq = [(initial_h_score, 0, start_state_key, initial_mol_design, [])]
        visited_states = {start_state_key: 0}
        nodes_expanded = 0

        if DEBUG_ASTAR:
            print(f"\nStarting A* for A={smiles_A}, B={smiles_B}")
            print(f"Target Canonical SMILES: {target_smiles_canonical}")
        log_interval = max(1, max_search_steps // 1000)

        # --- A* Main Loop ---
        while pq and nodes_expanded < max_search_steps:
            f_score, g_score, current_state_key, current_state, current_path = heapq.heappop(pq)
            current_smiles = current_state_key[0]

            if DEBUG_ASTAR and nodes_expanded % log_interval == 0:
                print(f"\nStep {nodes_expanded}/{max_search_steps}:")
                print(f"  Popped: f={f_score:.4f}, g={g_score}")
                # Limit printing long SMILES in key
                smiles_part = current_smiles[:50] + "..." if len(current_smiles) > 50 else current_smiles
                print(f"  State Key: (SMILES={smiles_part}, Lvl={current_state_key[1]}, L0Idx={current_state_key[2]}, L1Type={current_state_key[3]}, ...)")

            # --- Goal check ---
            if current_smiles == target_smiles_canonical:
                 # Optional stricter goal: check if state is back at level 0 with no context
                 # is_final_state = (current_state_key[1] == 0 and current_state_key[2] is None and
                 #                   current_state_key[3] is None and current_state_key[4] is None and
                 #                   current_state_key[5] is None)
                 # if is_final_state:
                 if DEBUG_ASTAR: print(f"!!! Goal SMILES found at step {nodes_expanded} !!!")
                 return smiles_A, smiles_B, current_path

            # --- Visited check ---
            if g_score > visited_states.get(current_state_key, float('inf')):
                if DEBUG_ASTAR and nodes_expanded % (log_interval * 10) == 0: # Less frequent log for skipping
                     print(f"  Skipping state (found better path with g={visited_states.get(current_state_key)})")
                continue

            nodes_expanded += 1

            # --- Get valid actions ---
            if current_state is None: continue
            action_mask = getattr(current_state, 'current_action_mask', None)
            if action_mask is None:
                if DEBUG_ASTAR: print(f"  State has None action mask at step {nodes_expanded}. Dead end?")
                continue
            valid_actions = [i for i, masked in enumerate(action_mask) if not masked]

            # --- Explore neighbors ---
            action_taken_success = False
            for action in valid_actions:
                action_log_prefix = f"    Action {action}: "
                next_state = None
                try:
                    next_state = copy.deepcopy(current_state)
                    if DEBUG_ASTAR: print(action_log_prefix + "Calling take_action...")
                    next_state.take_action(action)

                    if DEBUG_ASTAR:
                         infeasibility_status = getattr(next_state, 'infeasibility_flag', 'ERROR_FLAG_MISSING')
                         print(action_log_prefix + f"Returned from take_action. Infeasibility flag: {infeasibility_status}")

                    # Check 1: Infeasibility Flag
                    if getattr(next_state, 'infeasibility_flag', False):
                        if DEBUG_ASTAR:
                            reason = getattr(next_state, 'infeasibility_reason', 'N/A')
                            print(action_log_prefix + f"-> Infeasible state (Reason: {reason})")
                        continue

                except Exception as e:
                    if DEBUG_ASTAR: print(action_log_prefix + f"-> EXCEPTION during deepcopy/take_action: {e}")
                    # traceback.print_exc() # Uncomment for full traceback
                    continue

                # --- Get SMILES for the next state ---
                next_smiles = next_state._get_smiles_for_check()
                if DEBUG_ASTAR: print(action_log_prefix + f"Result of _get_smiles_for_check: '{next_smiles}' (Type: {type(next_smiles)})")
                if next_smiles is None:
                    if DEBUG_ASTAR: print(action_log_prefix + f"-> Next state produced None SMILES")
                    continue

                # --- Generate the next state key TUPLE ---
                next_state_key = get_state_key(next_state, next_smiles)
                if next_state_key is None:
                     if DEBUG_ASTAR: print(action_log_prefix + f"-> Failed to generate next state key.")
                     continue

                # --- Calculate cost and check visited (using the TUPLE key) ---
                new_g_score = g_score + 1
                if new_g_score < visited_states.get(next_state_key, float('inf')):
                    visited_states[next_state_key] = new_g_score
                    h_score = heuristic_tanimoto(next_state, target_fp, fp_radius, fp_bits)

                    if h_score == float('inf'):
                        if DEBUG_ASTAR: print(action_log_prefix + f"-> h_score=inf for {next_smiles}")
                        continue

                    # --- Push to priority queue ---
                    new_f_score = new_g_score + h_score
                    new_path = current_path + [action]
                    heapq.heappush(pq, (new_f_score, new_g_score, next_state_key, next_state, new_path))
                    action_taken_success = True
                    if DEBUG_ASTAR:
                        # Optional: Log pushing state details
                        smiles_part_push = next_smiles[:50] + "..." if len(next_smiles) > 50 else next_smiles
                        # print(action_log_prefix + f"-> Pushing: f={new_f_score:.4f}, g={new_g_score}, h={h_score:.4f}, key=(SMILES={smiles_part_push}, Lvl={next_state_key[1]}, ...)")
                        pass # Keep this less verbose by default

                else: # Optional logging if g_score is not better
                     if DEBUG_ASTAR and nodes_expanded % (log_interval * 20) == 0: # Very infrequent log for not pushing
                         print(action_log_prefix + f"-> Not pushing state key (visited g={visited_states.get(next_state_key)} <= new_g={new_g_score})")


            if DEBUG_ASTAR and not action_taken_success and nodes_expanded <= 1:
                 print(f"  DEBUG: No valid successor states were pushed to the queue for the initial state.")


        # If loop finishes without finding the goal
        if DEBUG_ASTAR: print(f"--- A* search finished for pair ({smiles_A}, {smiles_B}) after {nodes_expanded} steps (limit {max_search_steps}) ---")
        return None # Path not found

    except Exception as e:
        if DEBUG_ASTAR: print(f"!!! UNCAUGHT EXCEPTION in a_star_search({smiles_A}, {smiles_B}): {e} !!!")
        if DEBUG_ASTAR: traceback.print_exc()
        return None


# --- Butina ClusterData Function (Unchanged from previous version with progress) ---
def ClusterData_with_progress(data, nPts, distThresh, isDistData=False, distFunc=None, reordering=False):
    """ Butina clustering with progress printing.
        distFunc MUST be provided if isDistData is False.
    """
    if not isDistData and distFunc is None:
        raise ValueError("distFunc must be provided when isDistData is False")
    if isDistData and len(data) > (nPts * (nPts - 1) / 2):
        print("WARNING: Butina.ClusterData: Distance matrix is too long")

    print(f"Butina: Calculating neighbors for {nPts} points...")
    nbrLists = [[] for _ in range(nPts)]
    dmIdx = 0
    print_interval = max(1, nPts // 100)

    for i in range(nPts):
        if (i + 1) % print_interval == 0 or (i + 1) == nPts:
            percent = (i + 1) / nPts * 100
            print(f"\rButina: Processing point {i + 1}/{nPts} ({percent:.1f}%)", end='')

        for j in range(i):
          if not isDistData:
            dij = distFunc(data[i], data[j])
          else:
            if dmIdx < len(data): dij = data[dmIdx]; dmIdx += 1
            else: print(f"\nERROR: Butina: Distance matrix index out of bounds at i={i}, j={j}"); dij = float('inf')

          if dij is not None and dij <= distThresh:
            nbrLists[i].append(j)
            nbrLists[j].append(i)

    print("\nButina: Neighbor calculation complete.")
    print("Butina: Sorting points by neighbor count...")
    try:
        tLists = [(len(y) if y is not None else 0, x) for x, y in enumerate(nbrLists)]
        tLists.sort(reverse=True)
    except Exception as e:
        print(f"\nERROR: Butina: Failed during initial sort: {e}")
        return tuple()

    print(f"Butina: Starting clustering loop (initial points: {len(tLists)})...")
    res = []
    seen = [0] * nPts
    clusters_processed = 0
    initial_tlist_len = len(tLists)
    print_interval_cluster = max(1, initial_tlist_len // 100)

    while tLists:
        try: neighbors_count, idx = tLists.pop(0)
        except IndexError: break

        clusters_processed += 1
        if clusters_processed % print_interval_cluster == 0 or not tLists:
             percent_clustered = clusters_processed / initial_tlist_len * 100
             remaining = len(tLists)
             print(f"\rButina: Clustering... Centroids considered: {clusters_processed}/{initial_tlist_len}. Remaining points: {remaining} ({percent_clustered:.1f}% est.)", end='')

        if seen[idx]: continue
        tRes = [idx]; seen[idx] = 1
        current_neighbors = nbrLists[idx] if nbrLists[idx] is not None else []
        for nbr in current_neighbors:
          if not seen[nbr]: tRes.append(nbr); seen[nbr] = 1

        if reordering and tLists:
            members_indices = set(tRes)
            potential_neighbors_of_members = set()
            for member_idx in members_indices:
                member_neighbors = nbrLists[member_idx] if nbrLists[member_idx] is not None else []
                potential_neighbors_of_members.update(member_neighbors)

            tLists_indices = {item[1] for item in tLists}
            affected_indices_in_tLists = potential_neighbors_of_members.intersection(tLists_indices)

            if affected_indices_in_tLists:
                needs_resort = False
                new_tLists = []
                processed_indices_in_loop = set()

                for neighbor_count_old, point_idx in tLists:
                    if point_idx in processed_indices_in_loop: continue
                    if point_idx in affected_indices_in_tLists:
                        original_neighbors = set(nbrLists[point_idx] if nbrLists[point_idx] is not None else [])
                        current_unseen_neighbors = [n for n in original_neighbors if not seen[n]]
                        new_neighbor_count = len(current_unseen_neighbors)
                        if neighbor_count_old != new_neighbor_count: needs_resort = True
                        new_tLists.append((new_neighbor_count, point_idx))
                    else: new_tLists.append((neighbor_count_old, point_idx))
                    processed_indices_in_loop.add(point_idx)

                tLists = new_tLists
                if needs_resort: tLists.sort(reverse=True)

        res.append(tuple(tRes))

    print("\nButina: Clustering complete.")
    return tuple(res)


# --- Pair Molecules via Clustering (Unchanged from previous version) ---
def pair_molecules_via_clustering(
    smiles_list,
    sample_size_tag,
    checkpoint_dir=CHECKPOINT_DIR,
    fps_checkpoint_filename_tpl=FPS_CHECKPOINT_FILENAME_TPL,
    pairs_checkpoint_filename_tpl=PAIRS_CHECKPOINT_FILENAME_TPL,
    similarity_threshold=SIMILARITY_THRESHOLD,
    fp_radius=FP_RADIUS,
    fp_bits=FP_BITS
):
    """Generates pairs using Butina clustering with distFunc (memory efficient),
       with checkpointing for fps and pairs. Uses ClusterData_with_progress.
       Checkpoints are specific to the sample size."""

    fps_checkpoint_filename = fps_checkpoint_filename_tpl.format(sample_size_tag)
    pairs_checkpoint_filename = pairs_checkpoint_filename_tpl.format(sample_size_tag)
    fps_checkpoint_path = os.path.join(checkpoint_dir, fps_checkpoint_filename)
    pairs_checkpoint_path = os.path.join(checkpoint_dir, pairs_checkpoint_filename)
    if DEBUG_MODE: # Print paths only in debug mode for clarity
        print(f"DEBUG: Using FPS checkpoint path: {fps_checkpoint_path}")
        print(f"DEBUG: Using Pairs checkpoint path: {pairs_checkpoint_path}")

    if os.path.exists(fps_checkpoint_path):
        print(f"Loading fingerprints from checkpoint: {fps_checkpoint_path}")
        try:
            with open(fps_checkpoint_path, "rb") as f:
                checkpoint_data = pickle.load(f)
                fps = checkpoint_data['fps']
                valid_smiles = checkpoint_data['valid_smiles']
                if not all(isinstance(fp, DataStructs.ExplicitBitVect) for fp in fps):
                     raise TypeError("Loaded fingerprints are not all ExplicitBitVect")
            print(f"Loaded {len(fps)} fingerprints and {len(valid_smiles)} corresponding SMILES from sample {sample_size_tag}.")
        except Exception as e:
            print(f"Error loading/validating fingerprint checkpoint: {e}. Recalculating...")
            fps, valid_smiles = [], []
    else:
        print(f"No fingerprint checkpoint found for sample {sample_size_tag} at {fps_checkpoint_path}.")
        fps, valid_smiles = [], []

    if not fps:
        print(f"Calculating fingerprints for {len(smiles_list)} molecules (sample {sample_size_tag})...")
        smiles_to_fp_idx_calc = {}
        for smi in tqdm(smiles_list, desc=f"Fingerprinting sample {sample_size_tag}"):
            fp = calculate_morgan_fp(smi, fp_radius, fp_bits)
            if fp:
                fps.append(fp)
                valid_smiles.append(smi)
        print(f"Calculated {len(fps)} valid fingerprints for sample {sample_size_tag}.")
        if not fps: print("Warning: No valid fingerprints generated. Cannot proceed."); return []
        try:
            print(f"Saving fingerprints checkpoint to {fps_checkpoint_path}...")
            with open(fps_checkpoint_path, "wb") as f:
                pickle.dump({'fps': fps, 'valid_smiles': valid_smiles}, f)
            print("Fingerprints checkpoint saved.")
        except Exception as e: print(f"Error saving fingerprint checkpoint: {e}")

    nfps = len(fps)
    if nfps < 2:
        print(f"Warning: Less than 2 valid fingerprints available for sample {sample_size_tag}. Cannot cluster or pair.")
        return []

    if os.path.exists(pairs_checkpoint_path):
        print(f"Loading pairs from checkpoint: {pairs_checkpoint_path}")
        try:
            with open(pairs_checkpoint_path, "rb") as f: final_pairs = pickle.load(f)
            if not isinstance(final_pairs, list) or (final_pairs and not isinstance(final_pairs[0], tuple)):
                 raise TypeError("Loaded pairs data has incorrect format.")
            print(f"Loaded {len(final_pairs)} pairs for sample {sample_size_tag}.")
            return final_pairs
        except Exception as e:
            print(f"Error loading/validating pairs checkpoint: {e}. Recalculating...")
            final_pairs = []
    else:
         print(f"No pairs checkpoint found for sample {sample_size_tag} at {pairs_checkpoint_path}.")
         final_pairs = []

    if not final_pairs:
        print(f"Starting clustering for sample {sample_size_tag} with threshold > {similarity_threshold} using distFunc...")
        distance_threshold = 1.0 - similarity_threshold

        def distij(fp1, fp2):
             if not isinstance(fp1, DataStructs.ExplicitBitVect) or not isinstance(fp2, DataStructs.ExplicitBitVect): return 1.0
             try: return 1.0 - DataStructs.TanimotoSimilarity(fp1, fp2)
             except Exception: return 1.0

        print("Performing clustering...")
        start_cluster_time = time.time()
        try: clusters = ClusterData_with_progress(fps, nfps, distance_threshold, isDistData=False, distFunc=distij, reordering=True)
        except Exception as e: print(f"\nERROR: Clustering failed: {e}"); return []
        cluster_time = time.time() - start_cluster_time
        print(f"Clustering completed in {cluster_time:.2f}s. Found {len(clusters)} clusters for sample {sample_size_tag}.")

        pairs_set = set()
        print("Generating pairs from clusters...")
        if not clusters: print("Warning: No clusters found.")
        else:
             for cluster in tqdm(clusters, desc=f"Pairing sample {sample_size_tag}"):
                 if not hasattr(cluster, '__iter__'): continue
                 valid_indices = [idx for idx in cluster if isinstance(idx, int) and 0 <= idx < nfps]
                 if len(valid_indices) > 1:
                     for i, j in itertools.combinations(valid_indices, 2):
                         smi_i = valid_smiles[i]; smi_j = valid_smiles[j]
                         if isinstance(smi_i, str) and isinstance(smi_j, str):
                              pairs_set.add(tuple(sorted((smi_i, smi_j))))

        final_pairs = list(pairs_set)
        print(f"Generated {len(final_pairs)} unique candidate pairs for sample {sample_size_tag}.")

        if final_pairs:
             try:
                 print(f"Saving pairs checkpoint to {pairs_checkpoint_path}...")
                 with open(pairs_checkpoint_path, "wb") as f: pickle.dump(final_pairs, f)
                 print("Pairs checkpoint saved.")
             except Exception as e: print(f"Error saving pairs checkpoint: {e}")
        else: print("No pairs generated, skipping save.")

    return final_pairs


# --- Multiprocessing Wrapper (Unchanged) ---
def run_a_star_wrapper(args):
    try:
        smiles_A, smiles_B, config, max_steps, radius, bits = args
        return a_star_search(smiles_A, smiles_B, config, max_steps, radius, bits)
    except Exception as e: return None


# --- Main Execution (Modified for Single Pair Debugging) ---
def main():
    overall_start_time = time.time()
    print(f"--- Script Start: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC ---")
    print(f"Current User: mbinjavaid")
    print("Mode: A* search for transformation dataset generation (using clustered existing pairs).")
    print(f"Random Seed: {RANDOM_SEED}")

    effective_sample_size = SAMPLE_SIZE
    if DEBUG_MODE:
        effective_sample_size = DEBUG_PAIRING_LIMIT
        print("--- DEBUG MODE ACTIVE ---")
        print(f"  A* Debug Prints (DEBUG_ASTAR): {DEBUG_ASTAR}")
        print(f"  Pairing Sample Size (DEBUG_PAIRING_LIMIT): {effective_sample_size}")
        print(f"  A* Processing Limit (DEBUG_ASTAR_LIMIT): {DEBUG_ASTAR_LIMIT}")
        print("-------------------------")
    elif SAMPLE_SIZE is not None and SAMPLE_SIZE > 0:
        print(f"Sampling Active: Target subset size {SAMPLE_SIZE} molecules for pairing.")
    else:
        print("Sampling Disabled: Using all loaded molecules for pairing.")

    print(f"Max Atoms (Filtering): {MAX_ATOMS}")
    print(f"Clustering Similarity Threshold: > {SIMILARITY_THRESHOLD}")
    print(f"Fingerprint Radius/Bits: {FP_RADIUS} / {FP_BITS}")
    print(f"A* Search Step Limit Per Pair: {MAX_SEARCH_STEPS}")
    print(f"Checkpoint Directory: {CHECKPOINT_DIR}")


    all_successful_sequences = []
    total_pairs_processed_astar = 0
    total_candidate_pairs_generated = 0

    print("\nLoading and combining molecules...")
    combined_smiles_list = []
    seen_smiles = set()
    DATATYPES = { "train": CHEMBL_TRAIN_PATH, "valid": CHEMBL_VALID_PATH }
    for datatype, filepath in DATATYPES.items():
        smiles_list = load_and_filter_molecules(filepath, MAX_ATOMS, datatype)
        for smi in smiles_list:
            if smi not in seen_smiles:
                combined_smiles_list.append(smi)
                seen_smiles.add(smi)
    print(f"Total unique molecules loaded: {len(combined_smiles_list)}")
    if not combined_smiles_list: print("Error: No molecules loaded. Exiting."); return

    pairing_smiles_list = combined_smiles_list
    actual_sample_size = len(combined_smiles_list)
    sample_size_tag_for_checkpoints = "all"

    if effective_sample_size is not None and effective_sample_size > 0 and effective_sample_size < len(combined_smiles_list):
        actual_sample_size = effective_sample_size
        pairing_smiles_list = random.sample(combined_smiles_list, actual_sample_size)
        sample_size_tag_for_checkpoints = str(actual_sample_size)
        print(f"Using {actual_sample_size} molecules sampled for pairing.")
    else:
        actual_sample_size = len(combined_smiles_list)
        sample_size_tag_for_checkpoints = "all"
        print(f"Using all {actual_sample_size} loaded molecules for pairing.")

    candidate_pairs = pair_molecules_via_clustering(pairing_smiles_list, sample_size_tag_for_checkpoints)
    total_candidate_pairs_generated = len(candidate_pairs)
    if not candidate_pairs: print("No candidate pairs available (loaded or generated). Exiting."); return

    if DEBUG_MODE:
        print("\n--- DEBUG MODE: Selecting first pair for A* ---")
        if candidate_pairs:
            pairs_to_process = [candidate_pairs[0]]
            DEBUG_SMILES_A, DEBUG_SMILES_B = pairs_to_process[0]
            print(f"  Selected Pair:"); print(f"    A: {DEBUG_SMILES_A}"); print(f"    B: {DEBUG_SMILES_B}")
        else: print("  No candidate pairs found to select from."); pairs_to_process = []
    else: pairs_to_process = candidate_pairs

    if not pairs_to_process: print("No pairs selected for A* processing. Exiting."); return

    print(f"\nStarting A* search for {len(pairs_to_process)} pair(s) (sequentially)...")
    astar_start_time = time.time()
    all_successful_sequences = []
    total_pairs_processed_astar = 0

    for smiles_a, smiles_b in pairs_to_process:
        result = a_star_search(smiles_a, smiles_b, CONFIG, max_search_steps=MAX_SEARCH_STEPS, fp_radius=FP_RADIUS, fp_bits=FP_BITS)
        total_pairs_processed_astar += 1
        if result is not None:
            if isinstance(result, tuple) and len(result) == 3:
                 all_successful_sequences.append(result)
                 print(f"!!! SUCCESS FOUND FOR PAIR ({smiles_a}, {smiles_b}) !!!")
            else: print(f"Warning: Invalid result format received from A*: {result}")
        else:
            # Print this message even if DEBUG_ASTAR is off, as it's a summary for the pair
            print(f"--- No path found for pair ({smiles_a}, {smiles_b}) within step limit ---")

    astar_time = time.time() - astar_start_time
    print(f"\nA* search phase completed in {astar_time:.2f}s.")

    sampling_tag = f"_sampled{sample_size_tag_for_checkpoints}"
    debug_tag = "_debugSinglePair" if DEBUG_MODE else ""
    results_filename = f"{FINAL_RESULTS_FILENAME_BASE}{sampling_tag}_thresh{SIMILARITY_THRESHOLD}_limit{MAX_SEARCH_STEPS}{debug_tag}.pkl"
    final_results_path = os.path.join(RESULTS_DIR, results_filename)

    print(f"\nSaving final results ({len(all_successful_sequences)} sequences) to {final_results_path}...")
    if not all_successful_sequences: print("Warning: No successful sequences found. Saving empty list.")
    try:
        with open(final_results_path, "wb") as f: pickle.dump(all_successful_sequences, f)
        print(f"Successfully saved.")
    except Exception as e: print(f"Error saving final results: {e}")

    overall_end_time = time.time()
    summary_header = "\n--- Overall A* (Clustered Pairs) Summary" + (" (DEBUG MODE - Single Pair)" if DEBUG_MODE else "") + " ---"
    print(summary_header)
    print(f"Total Execution Time: {overall_end_time - overall_start_time:.2f} seconds")
    print(f"Total Unique Molecules Loaded: {len(combined_smiles_list)}")
    print(f"Molecules Used for Pairing Step: {actual_sample_size}")
    print(f"Total Candidate Pairs Generated (from sample {sample_size_tag_for_checkpoints}): {total_candidate_pairs_generated}")
    print(f"Total Pairs Processed by A*: {total_pairs_processed_astar}")
    print(f"Total Successful Action Sequences Found: {len(all_successful_sequences)}")
    overall_success_rate = (len(all_successful_sequences) / total_pairs_processed_astar * 100) if total_pairs_processed_astar > 0 else 0
    print(f"Overall Success Rate (A* Found Path): {overall_success_rate:.2f}%")
    completion_message = "A* sequence generation (clustered pairs) complete" + (" (DEBUG MODE - Single Pair)." if DEBUG_MODE else ".")
    print(f"\n--- {completion_message} ({datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC) ---")

if __name__ == "__main__":
    main()