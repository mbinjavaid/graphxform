import os
import datetime


class MoleculeConfig:
    def __init__(self):
        self.seed = 42

        # Network and environment
        self.latent_dimension = 128
        self.num_transformer_blocks = 8
        self.num_heads = 8
        self.dropout = 0.
        self.use_rezero_transformer = True

        # Environment options
        self.wall_clock_limit = None  # in seconds. If no limit, set to None
        self.max_num_atoms = 25

        self.atom_vocabulary = {  # Attention! Order matters!
            "C":    {"allowed": True, "atomic_number": 6, "valence": 4, "min_atoms": 0},
            "N":    {"allowed": True, "atomic_number": 7, "valence": 3, "min_atoms": 0},
            "O":    {"allowed": True, "atomic_number": 8, "valence": 2, "min_atoms": 0}
        }

        self.start_from_c_chains = True
        self.start_c_chain_max_len = 1
        self.start_from_smiles = None
        self.repeat_start_instances = 1
        # Positive value x, where the actual objective with our molecule score will be set to obj = score - x * SA_score
        self.synthetic_accessibility_in_objective_scale = 0
        # Don't allow oxygen to bond with other oxygen
        self.disallow_oxygen_bonding = True
        # Don't allow nitrogen to bond with nitrogen with single bond (only double or triple)
        self.disallow_nitrogen_nitrogen_single_bond = True
        # Don't allow rings, i.e., do not allow to increase the bond between atoms that aren't bonded yet at level 2.
        self.disallow_rings = False
        self.disallow_rings_larger_than = 0  # if this is greater than 3, we disallow all rings that are larger than the given value.

        # Objective molecule predictor
        self.GHGNN_model_path = os.path.join("objective_predictor/GH_GNN_IDAC/models/GHGNN.pth")
        self.GHGNN_hidden_dim = 113
        self.objective_type = "IBA"  # either "IBA" or "DMBA_TMB"
        self.num_predictor_workers = 10  # num of parallel workers that operate on a given list of molecules
        self.objective_predictor_batch_size = 64
        self.objective_gnn_device = "cpu"  # device on which the GNN should live

        # Loading trained checkpoints to resume training or evaluate
        self.load_checkpoint_from_path = None  # If given, model checkpoint is loaded from this path.
        self.load_optimizer_state = False  # If True, the optimizer state is also loaded.

        # Training
        self.num_dataloader_workers = 30  # Number of workers for creating batches for training
        self.CUDA_VISIBLE_DEVICES = "0,1"  # Must be set, as ray can have problems detecting multiple GPUs
        self.training_device = "cuda:0"  # Device on which to perform the supervised training
        self.num_epochs = 1000  # Number of epochs (i.e., passes through training set) to train
        self.scale_factor_level_one = 1.
        self.scale_factor_level_two = 1.
        self.batch_size_training = 64
        self.num_batches_per_epoch = 20  # Can be None, then we just do one pass through generated dataset

        # Optimizer
        self.optimizer = {
            "lr": 1e-4,  # learning rate
            "weight_decay": 0,
            "gradient_clipping": 1.,  # Clip gradient to given L2-norm. Set to 0 if no clipping should be performed.
            "schedule": {
                "decay_lr_every_epochs": 1,
                "decay_factor": 1
            }
        }

        # Self-improvement sequence decoding
        self.gumbeldore_config = {
            # Number of trajectories with the the highest objective function evaluation to keep for training
            "num_trajectories_to_keep": 500,
            "keep_intermediate_trajectories": False,  # if True, we consider all intermediate, terminable trajectories
            "devices_for_workers": ["cuda:0"] * 1,
            "destination_path": "./data/generated_molecules.pickle",
            "batch_size_per_worker": 1,  # Keep at one, as we only have three atoms from which we can start
            "batch_size_per_cpu_worker": 1,
            "search_type": "wor",
            "beam_width": 1024,
            "num_rounds": (10, 50),  # if it's a tuple, then we sample as long as it takes to obtain a better trajectory, but for a minimum of first entry rounds and a maximum of second entry rounds
            "deterministic": False,  # Only use for gumbeldore_eval=True below, switches to regular beam search.
            "nucleus_top_p": 1.,
            "pin_workers_to_core": False
        }

        # Results and logging
        self.results_path = os.path.join("./results",
                                         datetime.datetime.now().strftime(
                                             "%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights
        self.log_to_file = True

