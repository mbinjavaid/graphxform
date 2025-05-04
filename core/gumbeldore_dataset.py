import copy
import os # <-- Added
import pickle
import sys
import time
import ray
import torch
import numpy as np
from ray.thirdparty_files import psutil
from tqdm import tqdm
from rdkit import RDLogger, Chem # <-- Added Chem for type hint if needed later

# Import necessary classes (ensure paths are correct)
from model.molecule_transformer import MoleculeTransformer
from molecule_design import MoleculeDesign
from core.abstracts import Config, Instance # Removed BaseTrajectory as not used
import core.stochastic_beam_search as sbs
from core.incremental_sbs import IncrementalSBS
from config import MoleculeConfig
from molecule_evaluator import MoleculeObjectiveEvaluator

from typing import List, Tuple, Any, Optional # Removed unused types

os.environ["RAY_DEDUP_LOGS"] = "0"


@ray.remote
class JobPool:
    # --- No changes needed in JobPool ---
    def __init__(self, problem_instances: List[Instance]):
        self.jobs = [(i, instance) for i, instance in enumerate(problem_instances)]
        self.job_results = []
        self.total_jobs = len(self.jobs) # Store total for progress reporting

    def get_jobs(self, n_items: int):
        if len(self.jobs) > 0:
            items = self.jobs[:n_items]
            self.jobs = self.jobs[n_items:]
            return items
        else:
            return None

    def push_results(self, results: List[Tuple[int, Any]]):
        self.job_results.extend(results)

    def fetch_results(self):
        results = self.job_results
        self.job_results = []
        return results

    def get_remaining_jobs_count(self): # Helper for progress
        return len(self.jobs)

    def get_total_jobs_count(self): # Helper for progress
        return self.total_jobs
# --- End JobPool ---


class GumbeldoreDataset:
    # --- Modified __init__ and generate_dataset ---
    def __init__(self, config: MoleculeConfig,
                 objective_evaluator: MoleculeObjectiveEvaluator
                ):
        self.config = config
        self.gumbeldore_config = config.gumbeldore_config
        self.objective_evaluator = objective_evaluator
        self.devices_for_workers: List[str] = self.gumbeldore_config.get("devices_for_workers", ["cpu"]) # Added default

    def generate_dataset(self, network_weights: dict, best_objective: Optional[float] = None, memory_aggressive: bool = False):
        """
        Generates dataset using Ray workers with enhanced progress reporting.
        """
        batch_size_gpu = self.gumbeldore_config.get("batch_size_per_worker", 16) # Added default
        batch_size_cpu = self.gumbeldore_config.get("batch_size_per_cpu_worker", batch_size_gpu) # Added default

        # --- Determine starting molecules ---
        if hasattr(self.config, 'start_from_c_chains') and self.config.start_from_c_chains:
            problem_instances = MoleculeDesign.get_c_chains(self.config)
            print(f"Starting generation from {len(problem_instances)} C-chains.")
        elif hasattr(self.config, 'start_from_smiles') and self.config.start_from_smiles:
            try:
                # Assuming from_smiles returns a list, take the first element
                start_mol = MoleculeDesign.from_smiles(self.config, self.config.start_from_smiles)
                if isinstance(start_mol, list):
                    problem_instances = start_mol
                else: # If it returns a single instance
                    problem_instances = [start_mol]
                print(f"Starting generation from SMILES: {self.config.start_from_smiles} ({len(problem_instances)} instance(s))")
            except Exception as e:
                 print(f"ERROR creating starting molecule from SMILES '{self.config.start_from_smiles}': {e}")
                 raise
        else:
            repeat_start = getattr(self.config, 'repeat_start_instances', 1)
            problem_instances = MoleculeDesign.get_single_atom_molecules(self.config, repeat=repeat_start)
            print(f"Starting generation from {len(problem_instances)} single atoms (repeat={repeat_start}).")
        # --- End starting molecules ---

        if not problem_instances:
             print("ERROR: No starting problem instances generated. Cannot proceed.")
             # Return empty metrics or raise error
             return {"mean_best_gen_obj": float('nan'), "best_gen_obj": float('-inf'), "worst_gen_obj": float('nan'),
                     "mean_top_20_obj": float('nan'), "top_20_molecules": []} # Match expected keys

        job_pool = JobPool.remote(copy.deepcopy(problem_instances))
        total_problem_instances = len(problem_instances)
        results = [None] * total_problem_instances

        # --- Worker setup (unchanged) ---
        cpu_cores = [None] * len(self.devices_for_workers)
        pin_workers = self.gumbeldore_config.get("pin_workers_to_core", False) # Added default
        if pin_workers and sys.platform == "linux":
            try:
                affinity = list(os.sched_getaffinity(0))
                cpu_cores = [affinity[i % len(affinity)] for i in range(len(self.devices_for_workers))] # Use modulo affinity length
                print(f"Pinning workers to cores: {cpu_cores}")
            except AttributeError: print("Warning: os.sched_getaffinity not available. Cannot pin workers.")

        log_interval_batches = self.gumbeldore_config.get("worker_log_interval", 10) # Configurable log interval

        # Kick off workers
        print(f"Launching {len(self.devices_for_workers)} generation workers...")
        future_tasks = [
            async_sbs_worker.remote(
                self.config, job_pool, network_weights, device,
                batch_size_gpu if device != "cpu" else batch_size_cpu,
                cpu_cores[i], best_objective, memory_aggressive,
                log_interval_batches # Pass log interval
            )
            for i, device in enumerate(self.devices_for_workers)
        ]
        # --- End worker setup ---

        # --- Progress Monitoring Loop ---
        print("Monitoring worker progress...")
        with tqdm(total=total_problem_instances, desc="Generation Jobs Completed") as progress_bar:
            last_reported_done = 0
            while True:
                # Check for completed tasks without blocking indefinitely
                ready, remaining = ray.wait(future_tasks, num_returns=len(future_tasks), timeout=2.0)

                # Fetch results non-blockingly if possible, update progress bar
                try:
                    fetched_results = ray.get(job_pool.fetch_results.remote())
                    for (i, result) in fetched_results:
                        if results[i] is None: # Avoid double counting if somehow fetched again
                             results[i] = result
                             progress_bar.update(1)
                    last_reported_done = progress_bar.n
                except ray.exceptions.RayActorError as e:
                    print(f"\nERROR fetching results from JobPool: {e}")
                    # Potentially try to continue or break depending on severity
                    break # Safer to break if JobPool actor died

                # Check if all workers have finished *and* all results have been processed
                # (progress_bar.n should equal total_problem_instances)
                if not remaining and progress_bar.n >= total_problem_instances:
                    print("\nAll workers finished and all results processed.")
                    break

                # Optional: Add a check for dead actors if things seem stalled
                if not ready and not fetched_results and last_reported_done == progress_bar.n:
                     # If no tasks became ready and no results were fetched for a while, check actor status
                     try:
                         actor_statuses = [ray.get(task) for task in remaining] # This might block slightly
                     except ray.exceptions.RayActorError as e:
                         print(f"\nERROR: A worker actor seems to have died: {e}")
                         # Decide how to handle - maybe attempt recovery or just break
                         break # Break loop if a worker died
        # --- End Progress Monitoring Loop ---

        # Final check to ensure all tasks completed (handles potential race conditions)
        try:
            ray.get(future_tasks)
            print("All worker tasks confirmed complete.")
        except ray.exceptions.RayActorError as e:
             print(f"ERROR: An error occurred in a worker during final get: {e}")
        except Exception as e:
             print(f"ERROR: An unexpected error occurred during final worker get: {e}")


        del job_pool # Explicitly delete remote actor reference
        del network_weights
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # Check if results list was fully populated
        num_missing = sum(1 for r in results if r is None)
        if num_missing > 0:
            print(f"WARNING: {num_missing}/{total_problem_instances} results were missing after generation.")

        print("Processing generated results...")
        return self.process_results(problem_instances, results)
    # --- End generate_dataset ---

    # --- process_results (minor robustness added) ---
    def process_results(self, problem_instances, results):
        """
        Processes the results from Gumbeldore search and save it to a pickle.
        """
        metrics_return = dict()
        instances_dict = dict()
        num_processed_results = 0
        num_valid_objective = 0
        num_unique_smiles = 0

        # Iterate through results, handling potential None values
        for i, result_list in enumerate(results):
            if result_list is None:
                # print(f"Warning: No result found for starting instance index {i}")
                continue # Skip if result is missing

            num_processed_results += len(result_list)
            for molecule in result_list:  # type: MoleculeDesign
                 # Ensure objective is valid before processing
                 obj = getattr(molecule, 'objective', float("-inf")) # Safely get objective
                 if obj is None or obj <= float("-inf"): continue # Skip molecules with invalid/no objective

                 num_valid_objective += 1
                 smiles = getattr(molecule, 'smiles_string', None) # Safely get smiles
                 if smiles is None: continue # Skip if SMILES string is None (shouldn't happen if obj is valid)

                 # Add to dict if SMILES is new or objective is better
                 if smiles not in instances_dict or obj > instances_dict[smiles]["obj"]:
                      if smiles not in instances_dict: num_unique_smiles += 1
                      instances_dict[smiles] = dict(
                          start_atom=getattr(molecule, 'initial_atom', None), # Safely get start atom
                          action_seq=getattr(molecule, 'history', []), # Safely get history
                          smiles=smiles,
                          obj=obj,
                          sa_score=getattr(molecule, 'sa_score', float('nan')) # Safely get SA score
                      )

        print(f"Processed {num_processed_results} total generated molecules.")
        print(f"Found {num_valid_objective} with valid objectives.")
        print(f"Retained {len(instances_dict)} unique, valid SMILES.")

        generated_mols = list(instances_dict.values())
        num_to_keep = self.gumbeldore_config.get("num_trajectories_to_keep", 1000) # Added default
        generated_mols = sorted(generated_mols, key=lambda x: x["obj"], reverse=True)[:num_to_keep]

        if not generated_mols:
             print("Warning: No valid molecules retained after filtering.")
             # Return default/empty metrics
             return {"mean_best_gen_obj": float('nan'), "best_gen_obj": float('-inf'), "worst_gen_obj": float('nan'),
                     "mean_top_20_obj": float('nan'), "top_20_molecules": [],
                     "mean_best_gen_sa_score": float('nan'), "best_gen_sa_score": float('nan'), "worst_gen_sa_score": float('nan'),
                     "mean_kept_obj": float('nan'), "mean_top_20_sa_score": float('nan')}

        # Calculate metrics based on retained molecules
        generated_objs = np.array([x["obj"] for x in generated_mols])
        generated_sa_scores = np.array([x["sa_score"] for x in generated_mols if not np.isnan(x["sa_score"])]) # Filter NaN SA scores

        metrics_return["mean_best_gen_obj"] = generated_objs.mean() if len(generated_objs) > 0 else float('nan')
        metrics_return["mean_best_gen_sa_score"] = generated_sa_scores.mean() if len(generated_sa_scores) > 0 else float('nan')
        metrics_return["best_gen_obj"] = generated_objs[0] if len(generated_objs) > 0 else float('-inf')
        metrics_return["best_gen_sa_score"] = generated_sa_scores[0] if len(generated_sa_scores) > 0 else float('nan')
        metrics_return["worst_gen_obj"] = generated_objs[-1] if len(generated_objs) > 0 else float('nan')
        metrics_return["worst_gen_sa_score"] = generated_sa_scores[-1] if len(generated_sa_scores) > 0 else float('nan')

        # --- Merging with existing data (logic unchanged, added print) ---
        destination_path = self.gumbeldore_config.get("destination_path")
        merged_mols = generated_mols
        if destination_path is not None:
            print(f"Checking for existing dataset at: {destination_path}")
            merged = False
            if os.path.isfile(destination_path):
                try:
                    with open(destination_path, "rb") as f:
                        existing_mols = pickle.load(f)
                    if isinstance(existing_mols, list):
                        print(f"Merging {len(generated_mols)} new molecules with {len(existing_mols)} existing molecules.")
                        temp_d = {x["smiles"]: x for x in existing_mols + merged_mols} # Keep best based on obj implicitly
                        merged_mols = list(temp_d.values())
                        merged = True
                    else: print("Warning: Existing file is not a list. Overwriting.")
                except Exception as e: print(f"Warning: Error loading existing dataset: {e}. Overwriting.")

            merged_mols = sorted(merged_mols, key=lambda x: x["obj"], reverse=True)[:num_to_keep]
            print(f"Saving {len(merged_mols)} {'merged ' if merged else ''}molecules to {destination_path}.")
            try:
                with open(destination_path, "wb") as f:
                    pickle.dump(merged_mols, f)
            except Exception as e: print(f"ERROR saving merged dataset: {e}")
        # --- End Merging ---

        # --- Calculate final metrics based on potentially merged data ---
        top_20_merged = merged_mols[:20]
        metrics_return["mean_top_20_obj"] = np.array([x["obj"] for x in top_20_merged]).mean() if top_20_merged else float('nan')
        metrics_return["mean_kept_obj"] = np.array([x["obj"] for x in merged_mols]).mean() if merged_mols else float('nan')
        top_20_sa_scores = np.array([x["sa_score"] for x in top_20_merged if not np.isnan(x["sa_score"])])
        metrics_return["mean_top_20_sa_score"] = top_20_sa_scores.mean() if len(top_20_sa_scores) > 0 else float('nan')
        # Format top 20 for return
        metrics_return["top_20_molecules"] = [{x["smiles"]: x["obj"]} for x in top_20_merged]

        print(f"Final processed metrics ready. Best overall obj: {metrics_return.get('best_gen_obj', float('-inf')):.3f}")
        return metrics_return
    # --- End process_results ---


@ray.remote(max_calls=1)
def async_sbs_worker(config: Config, job_pool: JobPool, network_weights: dict,
                     device: str, batch_size: int,
                     cpu_core: Optional[int] = None,
                     best_objective: Optional[float] = None,
                     memory_aggressive: bool = False,
                     # --- Added ---
                     log_interval: int = 10 # Interval for logging progress
                    ):
    # --- Added worker identification and counters ---
    worker_id = f"Worker-{os.getpid()}"
    batches_processed = 0
    total_molecules_processed = 0
    total_valid_molecules = 0
    start_time = time.time()
    print(f"[{worker_id}] Starting on device {device}. Log interval: {log_interval} batches.")
    # --- End Added ---

    # --- Helper functions (unchanged) ---
    def child_log_probability_fn(trajectories: List[MoleculeDesign]) -> List[np.array]:
        # Ensure log_probability_fn returns List[np.array]
        return MoleculeDesign.log_probability_fn(trajectories=trajectories, network=network)

    def batch_leaf_evaluation_fn(trajectories: List[MoleculeDesign]) -> np.array:
        objs = objective_evaluator.predict_objective(trajectories)
        for i, obj in enumerate(objs):
            # Safely set objective, handle potential missing attribute
            setattr(trajectories[i], 'objective', obj)
        return objs

    def child_transition_fn(trajectory_action_pairs: List[Tuple[MoleculeDesign, int]]) -> List[MoleculeDesign]:
        # Assuming transition_fn returns (new_molecule, is_done)
        # We need only the new_molecule state for SBS framework
        new_states = []
        for traj, action in trajectory_action_pairs:
            try:
                result = traj.transition_fn(action)
                if isinstance(result, tuple) and len(result) > 0 and isinstance(result[0], MoleculeDesign):
                    new_states.append(result[0])
                else:
                    # Log error if unexpected format is returned
                     print(f"[{worker_id}] ERROR: Unexpected return from transition_fn for action {action}. Got: {type(result)}. Skipping.")
            except Exception as e:
                 print(f"[{worker_id}] EXCEPTION during transition_fn for action {action}: {e}. Skipping state.")
                 # Optionally re-raise if needed: raise e
        return new_states
    # --- End helper functions ---

    # Silence RDKit warnings
    RDLogger.DisableLog('rdApp.*')

    # Pin worker to core if wanted (unchanged)
    if cpu_core is not None:
        try:
            os.sched_setaffinity(0, {cpu_core})
            psutil.Process().cpu_affinity([cpu_core])
        except Exception as e: print(f"[{worker_id}] Warning: Failed to pin to core {cpu_core}: {e}")

    # --- Main Worker Loop ---
    try: # Wrap main logic in try-finally for cleanup
        with torch.no_grad():
            # --- Setup (unchanged) ---
            if hasattr(config, 'CUDA_VISIBLE_DEVICES') and config.CUDA_VISIBLE_DEVICES:
                os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES

            device = torch.device(device)
            network = MoleculeTransformer(config, device)
            network.load_state_dict(network_weights)
            network.to(network.device)
            network.eval()

            objective_gnn_device = getattr(config, 'objective_gnn_device', device) # Use worker device if not specified
            objective_evaluator = MoleculeObjectiveEvaluator(config, torch.device(objective_gnn_device))
            # --- End Setup ---

            while True:
                # --- Fetch Batch ---
                batch_fetch_start = time.time()
                batch = ray.get(job_pool.get_jobs.remote(batch_size))
                if batch is None:
                    print(f"[{worker_id}] No more jobs. Exiting loop.")
                    break # Exit loop if no more jobs
                batches_processed += 1
                batch_fetch_time = time.time() - batch_fetch_start
                # --- End Fetch Batch ---

                idx_list = [i for i, _ in batch]
                root_nodes = [instance for _, instance in batch]
                if not root_nodes:
                     print(f"[{worker_id}] Warning: Fetched batch {batches_processed} but root_nodes list is empty. Skipping.")
                     continue

                # --- Perform Search (logic unchanged) ---
                search_start_time = time.time()
                try:
                    if config.gumbeldore_config.get("search_type", "tasar") == "beam_search": # Added default
                        beam_leaves_batch: List[List[sbs.BeamLeaf]] = sbs.stochastic_beam_search(
                            child_log_probability_fn=child_log_probability_fn,
                            child_transition_fn=child_transition_fn,
                            root_states=root_nodes,
                            beam_width=config.gumbeldore_config.get("beam_width", 100), # Added default
                            deterministic=True
                        )
                    else:
                        inc_sbs = IncrementalSBS(root_nodes, child_log_probability_fn, child_transition_fn,
                                                leaf_evaluation_fn=MoleculeDesign.to_max_evaluation_fn,
                                                batch_leaf_evaluation_fn=batch_leaf_evaluation_fn,
                                                memory_aggressive=memory_aggressive) # Use passed flag

                        if config.gumbeldore_config.get("search_type", "tasar") == "tasar":
                            beam_leaves_batch = inc_sbs.perform_tasar(
                                beam_width=config.gumbeldore_config.get("beam_width", 100),
                                deterministic=config.gumbeldore_config.get("deterministic", False),
                                nucleus_top_p=config.gumbeldore_config.get("nucleus_top_p", 0.9),
                                replan_steps=config.gumbeldore_config.get("replan_steps", 0),
                                sbs_keep_intermediate=config.gumbeldore_config.get("keep_intermediate_trajectories", False)
                            )
                        elif config.gumbeldore_config.get("search_type") == "wor":
                            beam_leaves_batch = inc_sbs.perform_incremental_sbs(
                                beam_width=config.gumbeldore_config.get("beam_width", 100),
                                num_rounds=config.gumbeldore_config.get("num_rounds", 1),
                                nucleus_top_p=config.gumbeldore_config.get("nucleus_top_p", 0.9),
                                sbs_keep_intermediate=config.gumbeldore_config.get("keep_intermediate_trajectories", False),
                                best_objective=best_objective
                            )
                        else:
                             print(f"[{worker_id}] ERROR: Unknown search_type '{config.gumbeldore_config.get('search_type')}'. Skipping batch.")
                             continue # Skip batch if search type is invalid
                except Exception as search_error:
                     print(f"[{worker_id}] ERROR during search for batch {batches_processed}: {search_error}")
                     # Decide how to handle - skip batch?
                     continue # Skip processing this batch on error
                search_time = time.time() - search_start_time
                # --- End Perform Search ---

                # --- Process Results & Check Validity (Added) ---
                results_to_push = []
                batch_molecules_generated = 0
                batch_valid_molecules = 0
                process_start_time = time.time()

                if len(beam_leaves_batch) != len(idx_list):
                     print(f"[{worker_id}] WARNING: Mismatch between beam_leaves_batch ({len(beam_leaves_batch)}) and idx_list ({len(idx_list)}) lengths for batch {batches_processed}.")
                     # Attempt to process based on the shorter length to avoid index errors
                     min_len = min(len(beam_leaves_batch), len(idx_list))
                     idx_list = idx_list[:min_len]
                     beam_leaves_batch = beam_leaves_batch[:min_len]


                for j, result_idx in enumerate(idx_list):
                    try:
                        # Safely get states, handle potential errors in BeamLeaf structure
                        leaves = beam_leaves_batch[j]
                        num_to_keep_worker = config.gumbeldore_config.get("num_trajectories_to_keep", 1000) # Use same num as in process_results
                        result_mols: List[MoleculeDesign] = [leaf.state for leaf in leaves[:num_to_keep_worker] if isinstance(getattr(leaf, 'state', None), MoleculeDesign)]

                        if not result_mols: # Skip if no valid states found for this index
                             # print(f"[{worker_id}] No valid MoleculeDesign states found for index {result_idx} in batch {batches_processed}.")
                             results_to_push.append((result_idx, [])) # Push empty list for this index
                             continue

                        batch_molecules_generated += len(result_mols)

                        # Check objective evaluation requirement (only for beam search typically)
                        # Safely check objective on the first molecule
                        first_mol_obj = getattr(result_mols[0], 'objective', None)
                        if first_mol_obj is None:
                            try: batch_leaf_evaluation_fn(result_mols)
                            except Exception as eval_err: print(f"[{worker_id}] ERROR during batch_leaf_evaluation_fn: {eval_err}")

                        # Check validity based on _cached_smiles
                        for mol in result_mols:
                            if getattr(mol, '_cached_smiles', None) is not None:
                                batch_valid_molecules += 1
                            # else: # Optional: Log invalid SMILES if needed for debugging
                            #    print(f"[{worker_id}] Invalid/Unfinalized mol generated (SMILES: None)")

                        results_to_push.append((result_idx, result_mols))
                    except IndexError:
                         print(f"[{worker_id}] ERROR: IndexError accessing beam_leaves_batch[{j}] for batch {batches_processed}. Skipping index {result_idx}.")
                         results_to_push.append((result_idx, [])) # Push empty list
                    except Exception as proc_err:
                         print(f"[{worker_id}] ERROR processing results for index {result_idx} in batch {batches_processed}: {proc_err}")
                         results_to_push.append((result_idx, [])) # Push empty list on error

                # Update cumulative counts
                total_molecules_processed += batch_molecules_generated
                total_valid_molecules += batch_valid_molecules
                process_time = time.time() - process_start_time
                # --- End Process Results & Check Validity ---

                # --- Push Results ---
                push_start_time = time.time()
                if results_to_push: # Only push if there's something to push
                    try: ray.get(job_pool.push_results.remote(results_to_push))
                    except Exception as push_err: print(f"[{worker_id}] ERROR pushing results for batch {batches_processed}: {push_err}")
                push_time = time.time() - push_start_time
                # --- End Push Results ---

                # --- Periodic Logging (Added) ---
                if batches_processed % log_interval == 0:
                    total_invalid = total_molecules_processed - total_valid_molecules
                    valid_ratio = (total_valid_molecules / total_molecules_processed * 100) if total_molecules_processed > 0 else 0
                    elapsed_time = time.time() - start_time
                    print(f"[{worker_id} @ {elapsed_time:.1f}s] Batch {batches_processed} done. "
                          f"Batch Stats(Mols/Valid): {batch_molecules_generated}/{batch_valid_molecules}. "
                          f"Cumulative(Mols/Valid/Ratio%): {total_molecules_processed}/{total_valid_molecules}/{valid_ratio:.2f}. "
                          f"Timings(Fetch/Search/Proc/Push): {batch_fetch_time:.2f}s/{search_time:.2f}s/{process_time:.2f}s/{push_time:.2f}s")
                # --- End Periodic Logging ---

                # --- GPU Cache Clear (Unchanged) ---
                if device != "cpu":
                    torch.cuda.empty_cache()
                # --- End GPU Cache Clear ---

    except Exception as worker_err:
         print(f"[{worker_id}] UNHANDLED EXCEPTION in main loop: {worker_err}")
         # Potentially log stack trace: import traceback; traceback.print_exc()
    finally:
        # --- Final Worker Log (Added) ---
        elapsed_time = time.time() - start_time
        total_invalid = total_molecules_processed - total_valid_molecules
        valid_ratio = (total_valid_molecules / total_molecules_processed * 100) if total_molecules_processed > 0 else 0
        print(f"[{worker_id}] Finished. Total time: {elapsed_time:.2f}s. "
              f"Processed Batches: {batches_processed}. "
              f"Total Mols: {total_molecules_processed}. Valid Mols: {total_valid_molecules}. Invalid Mols: {total_invalid}. Valid Ratio: {valid_ratio:.2f}%")
        # --- End Final Worker Log ---

        # --- Cleanup (Unchanged) ---
        del network # Free up memory
        del network_weights
        del objective_evaluator
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        # --- End Cleanup ---

# --- End async_sbs_worker ---