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

# --- Global Debug Flag ---
DEBUG_MODE = True # <<<<<<< ENABLE DEBUG MODE FOR SINGLE PAIR RUN
# --- End Global Debug Flag ---

# Turn off RDKit warnings
RDLogger.DisableLog('rdApp.*')

# --- Import Custom Modules ---
try:
    from config import MoleculeConfig
    from molecule_design import MoleculeDesign, build_reverse_atom_lookup
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print("Please ensure config.py and molecule_design.py are in the correct path and MoleculeDesign has required methods/properties.")
    exit(1)

# --- Configuration ---
try:
    CONFIG = MoleculeConfig()
except Exception as e:
    print(f"Error loading MoleculeConfig: {e}")
    exit(1)

# --- Script Parameters ---
# Debugging
# These might be overridden by DEBUG_MODE logic below, but set reasonable values
DEBUG_PAIRING_LIMIT = 10000 # Affects sampling if DEBUG_MODE is on
DEBUG_ASTAR_LIMIT = 1 # Process only one pair in debug mode
# <<< --- Sampling Parameter --- >>>
SAMPLE_SIZE = 10000  # Sample size used when DEBUG_MODE is False
# <<< --- End Sampling Parameter --- >>>
# Pairing & Clustering
SIMILARITY_THRESHOLD = 0.7
FP_RADIUS = 2
FP_BITS = 2048
# A* Search
MAX_SEARCH_STEPS = 200000 # Use the high limit for debugging
# Data Paths
MAX_ATOMS = CONFIG.max_num_atoms
RANDOM_SEED = CONFIG.seed
CHECKPOINT_DIR = "./data/chembl/checkpoints_astar" # Checkpoints for THIS script
RESULTS_DIR = "./data/chembl/astar_datasets"
CHEMBL_TRAIN_PATH = "./data/chembl/chembl_train_filtered.smiles"
CHEMBL_VALID_PATH = "./data/chembl/chembl_valid_filtered.smiles"
# Checkpoint Filenames within CHECKPOINT_DIR
# These names are now tied to the sample size used during their creation
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

# --- Molecule Loading (Reused - Unchanged) ---
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
            # Estimate total lines for tqdm
            try: total_lines = sum(1 for line in open(path))
            except: total_lines = None # Handle potential errors opening file again

            f.seek(0) # Reset file pointer
            for line in tqdm(f, desc=f"Filtering {datatype} molecules", total=total_lines):
                smiles = line.strip()
                if not smiles: continue
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None: continue
                    # Basic sanitization check
                    try:
                        Chem.SanitizeMol(mol)
                    except Exception:
                        continue # Skip molecules failing sanitization

                    # Filter by atom count *before* canonicalization can save time
                    num_heavy = mol.GetNumHeavyAtoms()
                    if num_heavy == 0 or num_heavy > max_atoms: continue

                    # Canonicalize and check uniqueness
                    try:
                        Chem.Kekulize(mol) # Kekulize before canonical SMILES
                        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                    except Exception:
                        continue # Skip if canonicalization fails

                    if canonical_smiles in processed_smiles: continue

                    # Final check on canonical mol (redundant if initial check is good, but safe)
                    mol_check = Chem.MolFromSmiles(canonical_smiles)
                    if mol_check is None: continue
                    num_heavy_check = mol_check.GetNumHeavyAtoms()
                    if num_heavy_check == 0 or num_heavy_check > max_atoms: continue # Should not happen if filtered before

                    filtered_smiles_list.append(canonical_smiles)
                    processed_smiles.add(canonical_smiles)
                except Exception:
                    # Catch any other unexpected RDKit errors for a specific SMILES
                    continue
    except FileNotFoundError:
         print(f"Error: Input SMILES file not found at {path}")
         return []
    except Exception as e:
         print(f"An unexpected error occurred during file processing: {e}")
         return [] # Return empty list on other file errors


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
    # Robustness: Handle potential None input
    if mol_or_smiles is None: return None
    mol = mol_or_smiles
    if isinstance(mol_or_smiles, str):
        mol = Chem.MolFromSmiles(mol_or_smiles)
    if mol is None: return None
    try: Chem.Kekulize(mol) # Kekulize before fingerprinting
    except: pass # Ignore kekulization errors for FP calc if possible
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return fp
    except Exception:
        # Catch potential errors during fingerprint generation
        return None

def heuristic_tanimoto(current_mol_design, target_fp, radius=FP_RADIUS, nBits=FP_BITS):
    if target_fp is None: return float('inf')
    try:
        current_smiles = current_mol_design._get_smiles_for_check()
        if current_smiles is None: return float('inf')
        # Handle empty SMILES case (e.g., if molecule reduced to nothing)
        if not current_smiles:
            # If target also has no bits set (e.g., invalid), distance is 0, else max distance
            return 0.0 if target_fp.GetNumOnBits() == 0 else 1.0

        current_fp = calculate_morgan_fp(current_smiles, radius, nBits)
        if current_fp is None: return float('inf') # Handle failure in FP calc

        # Ensure target_fp is valid before similarity calculation
        if not isinstance(target_fp, DataStructs.ExplicitBitVect):
             return float('inf')

        similarity = DataStructs.TanimotoSimilarity(current_fp, target_fp)
        # Clamp similarity to [0, 1] range just in case of numerical issues
        similarity = max(0.0, min(1.0, similarity))
        distance = 1.0 - similarity
        return distance
    except Exception:
        # Catch unexpected errors during heuristic calculation
        return float('inf')

# --- A* Search Function (WITH DEBUG LOGGING) ---
def a_star_search(smiles_A, smiles_B, config, max_search_steps=200000, fp_radius=2, fp_bits=2048): # Using example values
    """
    Performs A* search to find a sequence of actions transforming smiles_A to smiles_B.

    Args:
        smiles_A: Starting SMILES string.
        smiles_B: Target SMILES string.
        config: Configuration object (presumably MoleculeConfig).
        max_search_steps: Maximum number of nodes to expand.
        fp_radius: Radius for Morgan fingerprints used in heuristic.
        fp_bits: Number of bits for Morgan fingerprints used in heuristic.

    Returns:
        Tuple (smiles_A, smiles_B, path_list) if successful, None otherwise.
    """
    try:
        # Initial state setup
        # Assumes MoleculeDesign.from_smiles returns (MoleculeDesign_instance, initial_mask_or_None)
        initial_mol_design, _ = MoleculeDesign.from_smiles(config, smiles_A)
        if initial_mol_design is None or getattr(initial_mol_design, 'synthesis_done', False): # Check if initial state is valid/already done
            print(f"DEBUG A*: Invalid initial state for {smiles_A}") # DEBUG
            return None

        # Target state setup
        target_mol = Chem.MolFromSmiles(smiles_B)
        if target_mol is None:
            print(f"DEBUG A*: Invalid target SMILES {smiles_B}") # DEBUG
            return None
        try:
            Chem.Kekulize(target_mol)
            target_smiles_canonical = Chem.MolToSmiles(target_mol, canonical=True)
        except Exception:
            print(f"DEBUG A*: Failed to canonicalize target {smiles_B}") # DEBUG
            return None # Failed to canonicalize target

        target_fp = calculate_morgan_fp(target_mol, fp_radius, fp_bits)
        if target_fp is None:
             print(f"DEBUG A*: Failed to get fingerprint for target {smiles_B}") # DEBUG
             return None # Failed to get target FP

        # Start state SMILES and check if already at goal
        start_smiles_canonical = initial_mol_design._get_smiles_for_check() # Assumes this method exists
        if start_smiles_canonical is None:
            print(f"DEBUG A*: Failed to get SMILES for initial state {smiles_A}") # DEBUG
            return None # Failed to get initial SMILES
        if start_smiles_canonical == target_smiles_canonical:
            print(f"DEBUG A*: Start SMILES is already target SMILES.") # DEBUG
            return smiles_A, smiles_B, []

        # A* Initialization
        initial_h_score = heuristic_tanimoto(initial_mol_design, target_fp, fp_radius, fp_bits)
        if initial_h_score == float('inf'):
            print(f"DEBUG A*: Initial heuristic is infinity for {smiles_A}") # DEBUG
            return None # Invalid start heuristic

        # Priority Queue: (f_score, g_score, smiles_key, state_object, path_list)
        pq = [(initial_h_score, 0, start_smiles_canonical, initial_mol_design, [])] # g_score is 0 initially
        visited_smiles = {start_smiles_canonical: 0} # Store lowest g_score found for a SMILES
        nodes_expanded = 0

        # <<< --- DEBUG LOGGING --- >>>
        print(f"\nStarting A* for A={smiles_A}, B={smiles_B}")
        print(f"Target Canonical SMILES: {target_smiles_canonical}")
        log_interval = max(1, max_search_steps // 1000) # Print progress periodically (~1000 lines max)
        # <<< --- END DEBUG --- >>>

        # A* Main Loop
        while pq and nodes_expanded < max_search_steps:
            f_score, g_score, current_smiles, current_state, current_path = heapq.heappop(pq)

            # <<< --- DEBUG LOGGING --- >>>
            if nodes_expanded % log_interval == 0:
                print(f"\nStep {nodes_expanded}/{max_search_steps}:")
                print(f"  Popped: f={f_score:.4f}, g={g_score}, smiles={current_smiles}")
            # <<< --- END DEBUG --- >>>

            # Goal check (using canonical SMILES)
            if current_smiles == target_smiles_canonical:
                print(f"!!! Goal found at step {nodes_expanded} !!!") # DEBUG
                return smiles_A, smiles_B, current_path # Found path

            # Check if we found a shorter path to this state already
            if g_score > visited_smiles.get(current_smiles, float('inf')):
                # Optional: Log skipping due to better path found
                # if nodes_expanded % log_interval == 0: print(f"  Skipping state {current_smiles} (found better path with g={visited_smiles.get(current_smiles)})")
                continue # Already found a better path to this SMILES state

            nodes_expanded += 1

            # Get valid actions (handle potential None mask)
            if current_state is None:
                 print(f"ERROR: current_state became None at step {nodes_expanded}") # Should not happen if checks are done
                 continue
            action_mask = current_state.current_action_mask # Assumes this property exists
            if action_mask is None:
                print(f"  State {current_smiles} has None action mask at step {nodes_expanded}. Dead end.") # DEBUG
                continue # Cannot proceed from this state

            valid_actions = [i for i, masked in enumerate(action_mask) if not masked]

            # Explore neighbors
            action_taken_success = False # DEBUG Flag
            for action in valid_actions:
                action_log_prefix = f"    Action {action}: "
                next_state = None # Initialize for exception handling
                try:
                    next_state = copy.deepcopy(current_state)
                    print(action_log_prefix + "Calling take_action...")  # PRINT 1 (Seen)
                    next_state.take_action(action)  # Apply action

                    infeasibility_status = getattr(next_state, 'infeasibility_flag', 'ERROR_FLAG_MISSING')
                    print(
                        action_log_prefix + f"Returned from take_action. Checking infeasibility flag: {infeasibility_status}")  # PRINT NEW (Seen)

                    # Check 1: Infeasibility
                    if next_state.infeasibility_flag:
                        reason = getattr(next_state, 'infeasibility_reason', 'N/A')
                        print(action_log_prefix + f"-> Infeasible state (Reason: {reason})")  # PRINT 2 (Not Seen)
                        continue

                except Exception as e:
                    # Exception Handling
                    print(action_log_prefix + f"-> EXCEPTION during deepcopy/take_action: {e}")  # PRINT 3 (Not Seen)
                    continue

                    # Get SMILES for the next state
                next_smiles = next_state._get_smiles_for_check()

                # ====> ADD THIS DIAGNOSTIC PRINT <====
                print(
                    action_log_prefix + f"Result of _get_smiles_for_check: '{next_smiles}' (Type: {type(next_smiles)})")  # <<< **** NEW DIAGNOSTIC PRINT ****

                # Check 2: None SMILES
                if next_smiles is None:
                    print(action_log_prefix + f"-> Next state produced None SMILES")  # PRINT 4 (Not Seen)
                    continue  # Skip to next action

                # Calculate cost to reach neighbor
                new_g_score = g_score + 1

                # Check if this path to next_smiles is better than any previous path
                if new_g_score < visited_smiles.get(next_smiles, float('inf')):
                    visited_smiles[next_smiles] = new_g_score
                    h_score = heuristic_tanimoto(next_state, target_fp, fp_radius, fp_bits)

                    # Check 3: Infinite Heuristic
                    if h_score == float('inf'):
                        print(action_log_prefix + f"-> h_score=inf for {next_smiles}")  # PRINT 5 (Not Seen)
                        continue  # Skip to next action

                    # Calculate f_score and add to priority queue
                    new_f_score = new_g_score + h_score
                    new_path = current_path + [action]
                    heapq.heappush(pq, (new_f_score, new_g_score, next_smiles, next_state, new_path))
                    action_taken_success = True
                    # Optional: Log pushing state
                    # print(action_log_prefix + f"-> Pushing: f={new_f_score:.4f}, g={new_g_score}, h={h_score:.4f}, smiles={next_smiles}")

                # else: # Optional logging if g_score is not better
                    # Optional: Log why not pushing
                    # print(action_log_prefix + f"-> Not pushing {next_smiles} (visited g={visited_smiles.get(next_smiles)} <= new_g={new_g_score})")


            # <<< --- DEBUG: Check if any action succeeded --- >>>
            if not action_taken_success and nodes_expanded <= 1: # Only print this for the first node
                 print(f"  DEBUG: No valid successor states were pushed to the queue for the initial state.")
            # <<< --- END DEBUG --- >>>


        # If loop finishes without finding the goal
        print(f"--- A* search finished for pair ({smiles_A}, {smiles_B}) after {nodes_expanded} steps (limit {max_search_steps}) ---")
        return None # Path not found

    except Exception as e:
        # Catch any unexpected errors during the entire A* process for a pair
        print(f"!!! UNCAUGHT EXCEPTION in a_star_search({smiles_A}, {smiles_B}): {e} !!!") # DEBUG
        traceback.print_exc() # DEBUG
        return None


# --- Butina ClusterData Function (with progress, minor print changes) ---
def ClusterData_with_progress(data, nPts, distThresh, isDistData=False, distFunc=None, reordering=False):
    """ Butina clustering with progress printing.
        distFunc MUST be provided if isDistData is False.
    """
    if not isDistData and distFunc is None:
        raise ValueError("distFunc must be provided when isDistData is False")

    # Existing warning check
    if isDistData and len(data) > (nPts * (nPts - 1) / 2):
        print("WARNING: Butina.ClusterData: Distance matrix is too long")

    # --- Start of Neighbor Calculation ---
    print(f"Butina: Calculating neighbors for {nPts} points...") # Initial message
    nbrLists = [[] for _ in range(nPts)] # Initialize lists directly

    dmIdx = 0
    print_interval = max(1, nPts // 100) # Aim for ~100 updates

    for i in range(nPts):
        # --- Progress Printing Start ---
        if (i + 1) % print_interval == 0 or (i + 1) == nPts:
            percent = (i + 1) / nPts * 100
            # Use print with '\r' and end='' to overwrite line
            print(f"\rButina: Processing point {i + 1}/{nPts} ({percent:.1f}%)", end='')
        # --- Progress Printing End ---

        for j in range(i):
          if not isDistData:
            dij = distFunc(data[i], data[j]) # Assumes distFunc takes data points
          else:
            # Ensure dmIdx stays within bounds for precomputed distances
            if dmIdx < len(data):
                dij = data[dmIdx]
                dmIdx += 1
            else:
                 # This case should ideally not happen if data is correct length
                 print(f"\nERROR: Butina: Distance matrix index out of bounds at i={i}, j={j}")
                 dij = float('inf') # Assign max distance on error

          # Check if distance is below threshold (handle potential None/inf)
          if dij is not None and dij <= distThresh:
            nbrLists[i].append(j)
            nbrLists[j].append(i)

    print("\nButina: Neighbor calculation complete.") # Print newline after loop
    # --- End of Neighbor Calculation ---


    # --- Start of Clustering Phase ---
    print("Butina: Sorting points by neighbor count...")
    try:
        # Ensure neighbor lists are valid before calculating length
        tLists = [(len(y) if y is not None else 0, x) for x, y in enumerate(nbrLists)]
        tLists.sort(reverse=True)
    except Exception as e:
        print(f"\nERROR: Butina: Failed during initial sort: {e}")
        return tuple() # Return empty tuple on error

    print(f"Butina: Starting clustering loop (initial points: {len(tLists)})...")

    res = []
    seen = [0] * nPts
    clusters_processed = 0
    initial_tlist_len = len(tLists) # Store initial length for percentage
    print_interval_cluster = max(1, initial_tlist_len // 100) # Progress for clustering loop

    while tLists:
        # Pop the point with the most neighbors
        try:
            neighbors_count, idx = tLists.pop(0)
        except IndexError:
             break # Should not happen if while condition is correct, but safe

        # --- Clustering Progress (Before skipping) ---
        clusters_processed += 1 # Count points considered as centroids
        if clusters_processed % print_interval_cluster == 0 or not tLists:
             # Estimate percentage based on points processed (pop attempts)
             percent_clustered = clusters_processed / initial_tlist_len * 100
             remaining = len(tLists)
             print(f"\rButina: Clustering... Centroids considered: {clusters_processed}/{initial_tlist_len}. Remaining points: {remaining} ({percent_clustered:.1f}% est.)", end='')
        # --- End Clustering Progress ---


        if seen[idx]:
            continue # Skip if already assigned to a cluster

        # Start new cluster with the centroid
        tRes = [idx]
        seen[idx] = 1 # Mark centroid as seen

        # Add neighbors to the cluster
        current_neighbors = nbrLists[idx] if nbrLists[idx] is not None else []
        for nbr in current_neighbors:
          if not seen[nbr]:
            tRes.append(nbr)
            seen[nbr] = 1 # Mark neighbor as seen

        # --- Reordering Logic ---
        if reordering and tLists: # Only reorder if requested and items remain
          # Identify neighbors of the current cluster members that are still in tLists
          members_indices = set(tRes)
          potential_neighbors_of_members = set()

          # Gather all neighbors of the new cluster members
          for member_idx in members_indices:
              member_neighbors = nbrLists[member_idx] if nbrLists[member_idx] is not None else []
              potential_neighbors_of_members.update(member_neighbors)

          # Find which remaining points in tLists were neighbors of the new cluster
          tLists_indices = {item[1] for item in tLists} # Indices currently in tLists
          affected_indices_in_tLists = potential_neighbors_of_members.intersection(tLists_indices)

          if affected_indices_in_tLists: # Only proceed if there are affected points
              needs_resort = False
              new_tLists = []
              processed_indices_in_loop = set() # Track indices processed in this reorder step

              # Iterate through old tLists to build new_tLists
              for neighbor_count_old, point_idx in tLists:
                  if point_idx in processed_indices_in_loop: continue # Already handled

                  if point_idx in affected_indices_in_tLists:
                      # Recalculate neighbor count excluding seen points
                      original_neighbors = set(nbrLists[point_idx] if nbrLists[point_idx] is not None else [])
                      # Count only neighbors that are NOT already seen (members of previous clusters)
                      # This requires checking against the 'seen' array
                      current_unseen_neighbors = [n for n in original_neighbors if not seen[n]]
                      new_neighbor_count = len(current_unseen_neighbors)

                      # Check if count changed
                      if neighbor_count_old != new_neighbor_count:
                          needs_resort = True
                      new_tLists.append((new_neighbor_count, point_idx))
                      processed_indices_in_loop.add(point_idx)
                  else:
                       # If not affected, keep the old entry
                       new_tLists.append((neighbor_count_old, point_idx))
                       processed_indices_in_loop.add(point_idx)


              tLists = new_tLists # Update tLists
              if needs_resort:
                  tLists.sort(reverse=True) # Resort only if counts changed

        # Add the completed cluster to results
        res.append(tuple(tRes))

    print("\nButina: Clustering complete.") # Print newline
    # --- End of Clustering Phase ---

    return tuple(res)


# --- Pair Molecules via Clustering (Uses ClusterData_with_progress) ---
def pair_molecules_via_clustering(
    smiles_list, # This will be the (potentially sampled) list
    sample_size_tag, # Added tag for checkpoint naming
    checkpoint_dir=CHECKPOINT_DIR,
    # Use templates for checkpoint filenames
    fps_checkpoint_filename_tpl=FPS_CHECKPOINT_FILENAME_TPL,
    pairs_checkpoint_filename_tpl=PAIRS_CHECKPOINT_FILENAME_TPL,
    similarity_threshold=SIMILARITY_THRESHOLD,
    fp_radius=FP_RADIUS,
    fp_bits=FP_BITS
):
    """Generates pairs using Butina clustering with distFunc (memory efficient),
       with checkpointing for fps and pairs. Uses ClusterData_with_progress.
       Checkpoints are specific to the sample size."""

    # Generate specific checkpoint names using the sample size tag
    fps_checkpoint_filename = fps_checkpoint_filename_tpl.format(sample_size_tag)
    pairs_checkpoint_filename = pairs_checkpoint_filename_tpl.format(sample_size_tag)
    fps_checkpoint_path = os.path.join(checkpoint_dir, fps_checkpoint_filename)
    pairs_checkpoint_path = os.path.join(checkpoint_dir, pairs_checkpoint_filename)
    print(f"Using FPS checkpoint: {fps_checkpoint_path}") # DEBUG
    print(f"Using Pairs checkpoint: {pairs_checkpoint_path}") # DEBUG


    # --- Fingerprint Calculation or Loading ---
    if os.path.exists(fps_checkpoint_path):
        print(f"Loading fingerprints from checkpoint: {fps_checkpoint_path}")
        try:
            with open(fps_checkpoint_path, "rb") as f:
                checkpoint_data = pickle.load(f)
                fps = checkpoint_data['fps']
                valid_smiles = checkpoint_data['valid_smiles']
                # Ensure fps are valid BitVect objects
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
        # Use the input smiles_list (which IS the sample)
        for smi in tqdm(smiles_list, desc=f"Fingerprinting sample {sample_size_tag}"):
            # No need to check smiles_to_fp_idx_calc if smiles_list is already unique (from random.sample)
            fp = calculate_morgan_fp(smi, fp_radius, fp_bits)
            if fp: # Check if fingerprint calculation was successful
                fps.append(fp)
                # valid_smiles directly corresponds to the input smiles_list order
                valid_smiles.append(smi)
                # Store index if needed later, though maybe not required if valid_smiles matches fps order
                # smiles_to_fp_idx_calc[smi] = len(fps) - 1

        print(f"Calculated {len(fps)} valid fingerprints for sample {sample_size_tag}.")
        if not fps:
             print("Warning: No valid fingerprints generated. Cannot proceed.")
             return []
        try:
            print(f"Saving fingerprints checkpoint to {fps_checkpoint_path}...")
            with open(fps_checkpoint_path, "wb") as f:
                # Save the fps and the corresponding smiles list (which is the sample)
                pickle.dump({'fps': fps, 'valid_smiles': valid_smiles}, f)
            print("Fingerprints checkpoint saved.")
        except Exception as e:
            print(f"Error saving fingerprint checkpoint: {e}")

    # --- Clustering and Pair Generation or Loading ---
    nfps = len(fps)
    if nfps < 2:
        print(f"Warning: Less than 2 valid fingerprints available for sample {sample_size_tag}. Cannot cluster or pair.")
        return []

    # --- Check for existing pairs checkpoint FIRST ---
    if os.path.exists(pairs_checkpoint_path):
        print(f"Loading pairs from checkpoint: {pairs_checkpoint_path}")
        try:
            with open(pairs_checkpoint_path, "rb") as f:
                final_pairs = pickle.load(f)
            # Basic validation: check if it's a list of tuples
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

    # --- If pairs not loaded, proceed with clustering using distFunc ---
    if not final_pairs:
        print(f"Starting clustering for sample {sample_size_tag} with threshold > {similarity_threshold} using distFunc...")
        distance_threshold = 1.0 - similarity_threshold

        # --- Define the distance function (distij) ---
        def distij(fp1, fp2):
             """Calculates Tanimoto distance between two fingerprint objects."""
             # Add check for valid fingerprint types
             if not isinstance(fp1, DataStructs.ExplicitBitVect) or not isinstance(fp2, DataStructs.ExplicitBitVect):
                 return 1.0 # Return max distance if types are wrong
             try:
                 return 1.0 - DataStructs.TanimotoSimilarity(fp1, fp2)
             except Exception:
                 return 1.0 # Return max distance on calculation error

        # --- Perform Clustering using ClusterData_with_progress ---
        print("Performing clustering...")
        start_cluster_time = time.time()
        try:
            # Use the modified Butina function
            clusters = ClusterData_with_progress(fps, nfps, distance_threshold, isDistData=False, distFunc=distij, reordering=True)
        except Exception as e:
             print(f"\nERROR: Clustering failed: {e}")
             return [] # Return empty list on clustering error

        cluster_time = time.time() - start_cluster_time
        print(f"Clustering completed in {cluster_time:.2f}s. Found {len(clusters)} clusters for sample {sample_size_tag}.")
        # --- End Clustering ---

        # --- Generate Pairs ---
        pairs_set = set()
        print("Generating pairs from clusters...")
        if not clusters:
             print("Warning: No clusters found.")
        else:
             for cluster in tqdm(clusters, desc=f"Pairing sample {sample_size_tag}"):
                 # Ensure cluster is iterable and contains indices
                 if not hasattr(cluster, '__iter__'): continue
                 # valid_smiles list directly corresponds to fps list (indices 0 to nfps-1)
                 valid_indices = [idx for idx in cluster if isinstance(idx, int) and 0 <= idx < nfps]

                 if len(valid_indices) > 1:
                     for i, j in itertools.combinations(valid_indices, 2):
                         # Indices i, j are from the cluster, map back to valid_smiles
                         smi_i = valid_smiles[i]
                         smi_j = valid_smiles[j]
                         # Add pair (ensure SMILES are strings)
                         if isinstance(smi_i, str) and isinstance(smi_j, str):
                              pairs_set.add(tuple(sorted((smi_i, smi_j)))) # Store sorted tuple for uniqueness

        final_pairs = list(pairs_set)
        print(f"Generated {len(final_pairs)} unique candidate pairs for sample {sample_size_tag}.")
        # --- End Pair Generation ---

        # --- Save Generated Pairs ---
        if final_pairs: # Only save if pairs were generated
             try:
                 print(f"Saving pairs checkpoint to {pairs_checkpoint_path}...")
                 with open(pairs_checkpoint_path, "wb") as f:
                     pickle.dump(final_pairs, f)
                 print("Pairs checkpoint saved.")
             except Exception as e:
                 print(f"Error saving pairs checkpoint: {e}")
        else:
             print("No pairs generated, skipping save.")
        # --- End Save ---

    return final_pairs


# --- Multiprocessing Wrapper (Unchanged) ---
def run_a_star_wrapper(args):
    # Add error handling within the wrapper is good practice
    try:
        smiles_A, smiles_B, config, max_steps, radius, bits = args
        return a_star_search(smiles_A, smiles_B, config, max_steps, radius, bits)
    except Exception as e:
        # print(f"Error in run_a_star_wrapper for ({args[0]}, {args[1]}): {e}", file=sys.stderr) # Optional debug
        return None # Return None if the wrapper itself fails


# --- Main Execution (Modified for Single Pair Debugging) ---
def main():
    overall_start_time = time.time()
    print(f"--- Script Start: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC ---")
    print(f"Current User: mbinjavaid") # Hardcoded as requested
    print("Mode: A* search for transformation dataset generation (using clustered existing pairs).")
    print(f"Random Seed: {RANDOM_SEED}")

    # Determine effective sample size based on DEBUG_MODE
    effective_sample_size = SAMPLE_SIZE
    if DEBUG_MODE:
        effective_sample_size = DEBUG_PAIRING_LIMIT
        print("--- DEBUG MODE ACTIVE ---")
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

    # --- Load & Combine Data ---
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
    if not combined_smiles_list:
        print("Error: No molecules loaded. Exiting.")
        return

    # --- Perform Sampling ---
    pairing_smiles_list = combined_smiles_list # Default to using all
    actual_sample_size = len(combined_smiles_list) # Track the size used
    sample_size_tag_for_checkpoints = "all" # Tag for checkpoint filenames

    if effective_sample_size is not None and effective_sample_size > 0 and effective_sample_size < len(combined_smiles_list):
        actual_sample_size = effective_sample_size
        pairing_smiles_list = random.sample(combined_smiles_list, actual_sample_size)
        sample_size_tag_for_checkpoints = str(actual_sample_size) # Use actual size in tag
        print(f"Using {actual_sample_size} molecules sampled for pairing.")
    else:
        # Using all molecules
        actual_sample_size = len(combined_smiles_list)
        sample_size_tag_for_checkpoints = "all"
        print(f"Using all {actual_sample_size} loaded molecules for pairing.")


    # --- Generate Pairs via Clustering (using the potentially sampled list) ---
    # Pass the potentially smaller pairing_smiles_list and the tag
    candidate_pairs = pair_molecules_via_clustering(pairing_smiles_list, sample_size_tag_for_checkpoints)
    total_candidate_pairs_generated = len(candidate_pairs)

    if not candidate_pairs:
        print("No candidate pairs available (loaded or generated). Exiting.")
        return

    # --- <<< --- DEBUG: Select ONLY the first pair --- >>> ---
    if DEBUG_MODE:
        print("\n--- DEBUG MODE: Selecting first pair for A* ---")
        if candidate_pairs:
            pairs_to_process = [candidate_pairs[0]] # Select the first pair
            DEBUG_SMILES_A, DEBUG_SMILES_B = pairs_to_process[0]
            print(f"  Selected Pair:")
            print(f"    A: {DEBUG_SMILES_A}")
            print(f"    B: {DEBUG_SMILES_B}")
        else:
            print("  No candidate pairs found to select from.")
            pairs_to_process = []
    else:
        # Normal mode: process all pairs (or limited by DEBUG_ASTAR_LIMIT if that was set differently)
        pairs_to_process = candidate_pairs
    # --- <<< --- END DEBUG SELECTION --- >>> ---


    if not pairs_to_process:
         print("No pairs selected for A* processing. Exiting.")
         return

    # --- Run A* Search (SEQUENTIALLY for Debugging) ---
    print(f"\nStarting A* search for {len(pairs_to_process)} pair(s) (sequentially)...")
    astar_start_time = time.time()
    all_successful_sequences = []
    total_pairs_processed_astar = 0

    # Run directly, not in parallel
    for smiles_a, smiles_b in pairs_to_process:
        result = a_star_search(smiles_a, smiles_b, CONFIG, max_search_steps=MAX_SEARCH_STEPS, fp_radius=FP_RADIUS, fp_bits=FP_BITS)
        total_pairs_processed_astar += 1
        if result is not None:
            # Basic validation of result format (tuple of 3 elements)
            if isinstance(result, tuple) and len(result) == 3:
                 all_successful_sequences.append(result)
                 print(f"!!! SUCCESS FOUND FOR PAIR ({smiles_a}, {smiles_b}) !!!")
            else:
                 print(f"Warning: Invalid result format received from A*: {result}")
        else:
            print(f"--- No path found for pair ({smiles_a}, {smiles_b}) within step limit ---")


    astar_time = time.time() - astar_start_time
    print(f"\nA* search phase completed in {astar_time:.2f}s.")

    # --- Save Results ---
    # Modify filename if sampling was used
    sampling_tag = f"_sampled{sample_size_tag_for_checkpoints}"
    debug_tag = "_debugSinglePair" if DEBUG_MODE else "" # Specific tag for this debug run
    results_filename = f"{FINAL_RESULTS_FILENAME_BASE}{sampling_tag}_thresh{SIMILARITY_THRESHOLD}_limit{MAX_SEARCH_STEPS}{debug_tag}.pkl"
    final_results_path = os.path.join(RESULTS_DIR, results_filename)


    print(f"\nSaving final results ({len(all_successful_sequences)} sequences) to {final_results_path}...")
    if not all_successful_sequences:
        print("Warning: No successful sequences found. Saving empty list.")

    try:
        with open(final_results_path, "wb") as f:
            pickle.dump(all_successful_sequences, f)
        print(f"Successfully saved.")
    except Exception as e:
        print(f"Error saving final results: {e}")

    # --- Final Summary ---
    overall_end_time = time.time()
    summary_header = "\n--- Overall A* (Clustered Pairs) Summary" + (" (DEBUG MODE - Single Pair)" if DEBUG_MODE else "") + " ---"
    print(summary_header)
    print(f"Total Execution Time: {overall_end_time - overall_start_time:.2f} seconds")
    print(f"Total Unique Molecules Loaded: {len(combined_smiles_list)}")
    print(f"Molecules Used for Pairing Step: {actual_sample_size}") # Show actual size used
    print(f"Total Candidate Pairs Generated (from sample {sample_size_tag_for_checkpoints}): {total_candidate_pairs_generated}")
    print(f"Total Pairs Processed by A*: {total_pairs_processed_astar}")
    print(f"Total Successful Action Sequences Found: {len(all_successful_sequences)}")
    overall_success_rate = (len(all_successful_sequences) / total_pairs_processed_astar * 100) if total_pairs_processed_astar > 0 else 0
    print(f"Overall Success Rate (A* Found Path): {overall_success_rate:.2f}%")
    completion_message = "A* sequence generation (clustered pairs) complete" + (" (DEBUG MODE - Single Pair)." if DEBUG_MODE else ".")
    print(f"\n--- {completion_message} ({datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC) ---")

if __name__ == "__main__":
    main()