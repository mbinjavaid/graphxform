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

# --- Global Debug Flag ---
DEBUG_MODE = False
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
DEBUG_PAIRING_LIMIT = 1000 # Keep for potential fine-grained debugging if needed
DEBUG_ASTAR_LIMIT = 100
# <<< --- NEW: Sampling Parameter --- >>>
# SAMPLE_SIZE = 200000  # Set the desired sample size (e.g., 200k). Set to None or 0 to disable sampling.
SAMPLE_SIZE = 10000  # Set the desired sample size (e.g., 200k). Set to None or 0 to disable sampling.
# <<< --- End Sampling Parameter --- >>>
# Pairing & Clustering
SIMILARITY_THRESHOLD = 0.7
FP_RADIUS = 2
FP_BITS = 2048
# A* Search
# MAX_SEARCH_STEPS = 1000
MAX_SEARCH_STEPS = 200000
# Data Paths
MAX_ATOMS = CONFIG.max_num_atoms
RANDOM_SEED = CONFIG.seed
CHECKPOINT_DIR = "./data/chembl/checkpoints_astar" # Checkpoints for THIS script
RESULTS_DIR = "./data/chembl/astar_datasets"
CHEMBL_TRAIN_PATH = "./data/chembl/chembl_train_filtered.smiles"
CHEMBL_VALID_PATH = "./data/chembl/chembl_valid_filtered.smiles"
# Checkpoint Filenames within CHECKPOINT_DIR
# Adjust checkpoint names if sampling affects what they represent (optional but good practice)
# For now, keeping them generic - they will represent fps/pairs *from the sample* if sampling is active
FPS_CHECKPOINT_FILENAME = f"combined_fps_r{FP_RADIUS}_b{FP_BITS}.pkl"
PAIRS_CHECKPOINT_FILENAME = f"clustered_pairs_thresh{SIMILARITY_THRESHOLD}.pkl"
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

# --- A* Search Function (Unchanged, but relies on robust helpers) ---
def a_star_search(smiles_A, smiles_B, config, max_search_steps=MAX_SEARCH_STEPS, fp_radius=FP_RADIUS, fp_bits=FP_BITS):
    try:
        # Initial state setup
        initial_mol_design, _ = MoleculeDesign.from_smiles(config, smiles_A)
        if initial_mol_design is None or initial_mol_design.synthesis_done: return None # Check if initial state is valid

        # Target state setup
        target_mol = Chem.MolFromSmiles(smiles_B)
        if target_mol is None: return None
        try:
            Chem.Kekulize(target_mol)
            target_smiles_canonical = Chem.MolToSmiles(target_mol, canonical=True)
        except Exception: return None # Failed to canonicalize target

        target_fp = calculate_morgan_fp(target_mol, fp_radius, fp_bits)
        if target_fp is None: return None # Failed to get target FP

        # Start state SMILES and check if already at goal
        start_smiles_canonical = initial_mol_design._get_smiles_for_check()
        if start_smiles_canonical is None: return None # Failed to get initial SMILES
        if start_smiles_canonical == target_smiles_canonical: return smiles_A, smiles_B, []

        # A* Initialization
        initial_h_score = heuristic_tanimoto(initial_mol_design, target_fp, fp_radius, fp_bits)
        if initial_h_score == float('inf'): return None # Invalid start heuristic

        # Priority Queue: (f_score, g_score, smiles_key, state_object, path_list)
        pq = [(initial_h_score, 0, start_smiles_canonical, initial_mol_design, [])] # g_score is 0 initially
        visited_smiles = {start_smiles_canonical: 0} # Store lowest g_score found for a SMILES
        nodes_expanded = 0

        # A* Main Loop
        while pq and nodes_expanded < max_search_steps:
            f_score, g_score, current_smiles, current_state, current_path = heapq.heappop(pq)

            # Goal check (using canonical SMILES)
            if current_smiles == target_smiles_canonical:
                return smiles_A, smiles_B, current_path # Found path

            # Check if we found a shorter path to this state already
            if g_score > visited_smiles.get(current_smiles, float('inf')):
                continue # Already found a better path to this SMILES state

            nodes_expanded += 1

            # Get valid actions (handle potential None mask)
            action_mask = current_state.current_action_mask
            if action_mask is None: continue # Cannot proceed from this state
            valid_actions = [i for i, masked in enumerate(action_mask) if not masked]

            # Explore neighbors
            for action in valid_actions:
                # Create a deep copy to avoid modifying the parent state
                try:
                    next_state = copy.deepcopy(current_state)
                    next_state.take_action(action) # Apply action
                    # Check if action resulted in an infeasible state (as defined in MoleculeDesign)
                    if next_state.infeasibility_flag: continue
                except Exception:
                    # Catch errors during deepcopy or take_action
                    continue # Skip this action/neighbor

                # Get SMILES for the next state
                next_smiles = next_state._get_smiles_for_check()
                if next_smiles is None: continue # Skip if SMILES generation failed

                # Calculate cost to reach neighbor
                new_g_score = g_score + 1 # Assuming uniform cost of 1 per step

                # Check if this path to next_smiles is better than any previous path
                if new_g_score < visited_smiles.get(next_smiles, float('inf')):
                    visited_smiles[next_smiles] = new_g_score # Update cost for this state
                    h_score = heuristic_tanimoto(next_state, target_fp, fp_radius, fp_bits)

                    # If heuristic is invalid, don't add to queue
                    if h_score == float('inf'): continue

                    # Calculate f_score and add to priority queue
                    new_f_score = new_g_score + h_score
                    new_path = current_path + [action]
                    heapq.heappush(pq, (new_f_score, new_g_score, next_smiles, next_state, new_path))

        # If loop finishes without finding the goal
        return None
    except Exception as e:
        # Catch any unexpected errors during the entire A* process for a pair
        # print(f"Error in a_star_search({smiles_A}, {smiles_B}): {e}", file=sys.stderr) # Optional debug print
        return None


# --- Butina ClusterData Function (with progress) ---
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
            print(f"\rButina: Processing point {i + 1}/{nPts} ({percent:.1f}%)")
            # sys.stderr.flush()
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

    print("\nButina: Neighbor calculation complete.\n")
    # sys.stderr.flush()
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
    print_interval_cluster = max(1, len(tLists) // 100) # Progress for clustering loop

    while tLists:
        clusters_processed += 1
        # --- Clustering Progress ---
        if clusters_processed % print_interval_cluster == 0 or not tLists:
             percent_clustered = clusters_processed / nPts * 100 # Approx % based on points processed
             remaining = len(tLists)
             print(f"\rButina: Clustering... Processed ~{clusters_processed}. Remaining points: {remaining} ({percent_clustered:.1f}% est.)")
             # sys.stderr.flush()
        # --- End Clustering Progress ---

        # Pop the point with the most neighbors
        try:
            neighbors_count, idx = tLists.pop(0)
        except IndexError:
             break # Should not happen if while condition is correct, but safe

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
          affected_indices_in_tLists = set()
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

    print("\nButina: Clustering complete.\n")
    # sys.stderr.flush()
    # --- End of Clustering Phase ---

    return tuple(res)


# --- Pair Molecules via Clustering (Uses ClusterData_with_progress) ---
def pair_molecules_via_clustering(
    smiles_list, # This will be the (potentially sampled) list
    checkpoint_dir=CHECKPOINT_DIR,
    fps_checkpoint_filename=FPS_CHECKPOINT_FILENAME,
    pairs_checkpoint_filename=PAIRS_CHECKPOINT_FILENAME,
    similarity_threshold=SIMILARITY_THRESHOLD,
    fp_radius=FP_RADIUS,
    fp_bits=FP_BITS
):
    """Generates pairs using Butina clustering with distFunc (memory efficient),
       with checkpointing for fps and pairs. Uses ClusterData_with_progress."""
    fps_checkpoint_path = os.path.join(checkpoint_dir, fps_checkpoint_filename)
    pairs_checkpoint_path = os.path.join(checkpoint_dir, pairs_checkpoint_filename)

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
            print(f"Loaded {len(fps)} fingerprints and {len(valid_smiles)} corresponding SMILES.")
        except Exception as e:
            print(f"Error loading/validating fingerprint checkpoint: {e}. Recalculating...")
            fps, valid_smiles = [], []
    else:
        print(f"No fingerprint checkpoint found at {fps_checkpoint_path}.")
        fps, valid_smiles = [], []

    if not fps:
        print(f"Calculating fingerprints for {len(smiles_list)} molecules...")
        smiles_to_fp_idx_calc = {}
        # Use the input smiles_list (which might be sampled)
        for smi in tqdm(smiles_list, desc="Fingerprinting"):
            if smi not in smiles_to_fp_idx_calc: # Ensure uniqueness within the list being processed
                fp = calculate_morgan_fp(smi, fp_radius, fp_bits)
                if fp: # Check if fingerprint calculation was successful
                    fps.append(fp)
                    current_idx = len(fps) - 1
                    valid_smiles.append(smi)
                    smiles_to_fp_idx_calc[smi] = current_idx
        print(f"Calculated {len(fps)} unique valid fingerprints.")
        if not fps:
             print("Warning: No valid fingerprints generated. Cannot proceed.")
             return []
        try:
            print(f"Saving fingerprints checkpoint to {fps_checkpoint_path}...")
            with open(fps_checkpoint_path, "wb") as f:
                # Save only the fps and the corresponding smiles list
                pickle.dump({'fps': fps, 'valid_smiles': valid_smiles}, f)
            print("Fingerprints checkpoint saved.")
        except Exception as e:
            print(f"Error saving fingerprint checkpoint: {e}")

    # --- Clustering and Pair Generation or Loading ---
    nfps = len(fps)
    if nfps < 2:
        print("Warning: Less than 2 valid fingerprints available. Cannot cluster or pair.")
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
            print(f"Loaded {len(final_pairs)} pairs.")
            return final_pairs
        except Exception as e:
            print(f"Error loading/validating pairs checkpoint: {e}. Recalculating...")
            final_pairs = []
    else:
         print(f"No pairs checkpoint found at {pairs_checkpoint_path}.")
         final_pairs = []

    # --- If pairs not loaded, proceed with clustering using distFunc ---
    if not final_pairs:
        print(f"Starting clustering with threshold > {similarity_threshold} using distFunc...")
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
        print(f"Clustering completed in {cluster_time:.2f}s. Found {len(clusters)} clusters.")
        # --- End Clustering ---

        # --- Generate Pairs ---
        pairs_set = set()
        print("Generating pairs from clusters...")
        if not clusters:
             print("Warning: No clusters found.")
        else:
             for cluster in tqdm(clusters, desc="Pairing"):
                 # Ensure cluster is iterable and contains indices
                 if not hasattr(cluster, '__iter__'): continue
                 valid_indices = [idx for idx in cluster if isinstance(idx, int) and 0 <= idx < len(valid_smiles)]

                 if len(valid_indices) > 1:
                     for i, j in itertools.combinations(valid_indices, 2):
                         # Indices i, j are from the cluster, map back to valid_smiles
                         smi_i = valid_smiles[i]
                         smi_j = valid_smiles[j]
                         # Add pair (ensure SMILES are strings)
                         if isinstance(smi_i, str) and isinstance(smi_j, str):
                              pairs_set.add(tuple(sorted((smi_i, smi_j)))) # Store sorted tuple for uniqueness

        final_pairs = list(pairs_set)
        print(f"Generated {len(final_pairs)} unique candidate pairs.")
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


# --- Main Execution (Modified for Sampling) ---
def main():
    overall_start_time = time.time()
    print(f"--- Script Start: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC ---")
    print(f"Current User: {os.getenv('USER', 'unknown')}")
    print("Mode: A* search for transformation dataset generation (using clustered existing pairs).")
    print(f"Random Seed: {RANDOM_SEED}")
    # <<< --- Log Sampling Setting --- >>>
    if SAMPLE_SIZE is not None and SAMPLE_SIZE > 0:
        print(f"Sampling Active: Using a subset of {SAMPLE_SIZE} molecules for pairing.")
    else:
        print("Sampling Disabled: Using all loaded molecules for pairing.")
    # <<< --- End Log Sampling --- >>>
    print(f"Max Atoms (Filtering): {MAX_ATOMS}")
    print(f"Clustering Similarity Threshold: > {SIMILARITY_THRESHOLD}")
    print(f"Fingerprint Radius/Bits: {FP_RADIUS} / {FP_BITS}")
    print(f"A* Search Step Limit Per Pair: {MAX_SEARCH_STEPS}")
    print(f"Checkpoint Directory: {CHECKPOINT_DIR}")
    if DEBUG_MODE:
        print("--- DEBUG MODE ACTIVE ---")
        print(f"  Pairing Molecule Limit (Sample Size Override): {DEBUG_PAIRING_LIMIT}") # Note DEBUG overrides SAMPLE_SIZE
        print(f"  A* Processing Limit: {DEBUG_ASTAR_LIMIT}")
        print("-------------------------")

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

    # --- <<< --- Perform Sampling --- >>> ---
    pairing_smiles_list = combined_smiles_list # Default to using all
    actual_sample_size = len(combined_smiles_list) # Track the size used

    # If DEBUG_MODE has a pairing limit, it overrides SAMPLE_SIZE
    if DEBUG_MODE and DEBUG_PAIRING_LIMIT > 0 and DEBUG_PAIRING_LIMIT < len(combined_smiles_list):
        print(f"DEBUG: Applying debug pairing limit, overriding SAMPLE_SIZE.")
        actual_sample_size = DEBUG_PAIRING_LIMIT
        pairing_smiles_list = random.sample(combined_smiles_list, actual_sample_size)
        print(f"Using {actual_sample_size} molecules for pairing due to DEBUG_PAIRING_LIMIT.")
    # Otherwise, apply SAMPLE_SIZE if it's set and smaller than total
    elif SAMPLE_SIZE is not None and SAMPLE_SIZE > 0 and SAMPLE_SIZE < len(combined_smiles_list):
        actual_sample_size = SAMPLE_SIZE
        pairing_smiles_list = random.sample(combined_smiles_list, actual_sample_size)
        print(f"Using {actual_sample_size} molecules sampled for pairing.")
    else:
        # Using all molecules (either sampling disabled or sample size >= total)
        print(f"Using all {actual_sample_size} loaded molecules for pairing.")
    # --- <<< --- End Sampling --- >>> ---


    # --- Generate Pairs via Clustering (using the potentially sampled list) ---
    # Pass the potentially smaller pairing_smiles_list
    candidate_pairs = pair_molecules_via_clustering(pairing_smiles_list)
    total_candidate_pairs_generated = len(candidate_pairs)

    if not candidate_pairs:
        print("No candidate pairs available (loaded or generated). Exiting.")
        return

    # Apply A* limit in debug mode (operates on the generated candidate_pairs)
    pairs_to_process = candidate_pairs
    if DEBUG_MODE and DEBUG_ASTAR_LIMIT > 0:
        if len(candidate_pairs) > DEBUG_ASTAR_LIMIT:
            print(f"DEBUG: Subsampling to {DEBUG_ASTAR_LIMIT} pairs for A* search.")
            pairs_to_process = random.sample(candidate_pairs, DEBUG_ASTAR_LIMIT)
        else:
            print(f"DEBUG: Processing all {len(candidate_pairs)} generated pairs with A*.")

    # --- Run A* Search in Parallel ---
    num_workers = cpu_count()
    print(f"\nStarting A* search for {len(pairs_to_process)} pairs using {num_workers} workers...")

    # Prepare arguments for the parallel pool
    search_args = [(a, b, CONFIG, MAX_SEARCH_STEPS, FP_RADIUS, FP_BITS) for a, b in pairs_to_process]

    astar_start_time = time.time()
    all_successful_sequences = [] # Ensure list is initialized here
    total_pairs_processed_astar = 0 # Reset counter

    # Use try-except block for the pool to catch potential errors during parallel processing
    try:
        with Pool(num_workers) as pool:
            # Using imap_unordered for potentially better performance as results come in
            results_iterator = pool.imap_unordered(run_a_star_wrapper, search_args)

            # Progress bar for A* search
            pbar = tqdm(results_iterator, total=len(pairs_to_process), desc="A* Search")

            for result in pbar:
                total_pairs_processed_astar += 1
                if result is not None:
                    # Basic validation of result format (tuple of 3 elements)
                    if isinstance(result, tuple) and len(result) == 3:
                         all_successful_sequences.append(result)
                         success_rate = (len(all_successful_sequences) / total_pairs_processed_astar) * 100
                         pbar.set_postfix({"Success": f"{success_rate:.1f}% ({len(all_successful_sequences)})"}, refresh=True)
                    # else: print(f"Warning: Invalid result format received from A*: {result}", file=sys.stderr) # Optional warning

    except Exception as e:
        print(f"\nERROR during A* parallel processing: {e}")
        # Decide how to handle partial results: maybe save what was collected?
        print(f"Attempting to save {len(all_successful_sequences)} sequences collected before error.")


    astar_time = time.time() - astar_start_time
    print(f"A* search phase completed in {astar_time:.2f}s.")

    # --- Save Results ---
    # Modify filename if sampling was used
    sampling_tag = f"_sampled{actual_sample_size}" if (SAMPLE_SIZE is not None and SAMPLE_SIZE > 0 and SAMPLE_SIZE < len(combined_smiles_list)) else ""
    debug_tag = "_debug" if DEBUG_MODE else ""
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
    summary_header = "\n--- Overall A* (Clustered Pairs) Summary" + (" (DEBUG MODE)" if DEBUG_MODE else "") + " ---"
    print(summary_header)
    print(f"Total Execution Time: {overall_end_time - overall_start_time:.2f} seconds")
    print(f"Total Unique Molecules Loaded: {len(combined_smiles_list)}")
    print(f"Molecules Used for Pairing Step: {actual_sample_size}") # Show actual size used
    print(f"Total Candidate Pairs Generated (from sample): {total_candidate_pairs_generated}")
    print(f"Total Pairs Processed by A*: {total_pairs_processed_astar}")
    print(f"Total Successful Action Sequences Found: {len(all_successful_sequences)}")
    overall_success_rate = (len(all_successful_sequences) / total_pairs_processed_astar * 100) if total_pairs_processed_astar > 0 else 0
    print(f"Overall Success Rate (A* Found Path): {overall_success_rate:.2f}%")
    completion_message = "A* sequence generation (clustered pairs) complete" + (" (DEBUG MODE)." if DEBUG_MODE else ".")
    print(f"\n--- {completion_message} ({datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC) ---")

if __name__ == "__main__":
    main()