# GraphXForm: Graph transformer for computer-aided molecular design with application to extraction

This is the repository for the paper **[GraphXForm: Graph transformer for computer-aided molecular design with application to extraction](arxiv.org)**.

## Preliminary note on structure

The learning algorithm used here is a hybrid between the deep cross-entropy method and self-improvement learning (SIL) 
(from natural language processing and neural combinatorial optimization). The general structure of our code thus follows
the SIL repository [https://github.com/grimmlab/gumbeldore](https://github.com/grimmlab/gumbeldore). Please check out
this repo for general information on how the code is structured and what is happening in the folder `core`.
So note that regarding the `gumbeldore_config` dict in `config.py`: In accordance with our paper, where sequences are 
sampled without replacement for multiple rounds, the setting `gumbeldore_config.search_type` is set to `wor` 
('without replacement').

## Install

Install all requirements from `requirements.txt`. Additionally, you must install **[torch_scatter](https://github.com/rusty1s/pytorch_scatter)** with the options 
corresponding to your hardware.

Create subdirectories `results` (where model weights and experimental results will be saved) and `data`. 

## Pretraining

To pretrain with the settings in the paper, do the following: 

1. Please download the file `chembl.tab` from this link: [Download File](https://drive.google.com/file/d/1UKNivLk5tgXzUwuKH2ZxYCCczcM-M7jl/view?usp=sharing)
and put it under the `./data` directory.
2. Run `$ python create_pretrain_dataset.py`. This will filter the SMILES in `chembl.tab` for all strings containing the C,N,O-alphabet and
converts them to instances of `MoleculeDesign` (the class in `molecule_design.py`, which takes the role of the molecular graph environment).
The pickled molecules are saved in `./data/pretrain_data.pickle`.
3. Run `$ python pretrain.py` to perform pretraining of the model. The general config to use (e.g., architecture) is under `config.py`.
In `pretrain.py`, you can specify pretrain-specific options directly at the entrypoint to adjust to your hardware, i.e.:
```python
if __name__ == '__main__':
    pretrain_train_dataset = "./data/pretrain_data.pickle"  # Path to the pretraining dataset
    pretrain_num_epochs = 1000   # For how many epochs to train
    batch_size = 128  # Minibatch size
    num_batches_per_epoch = 2500   # Number of minibatches per epoch.
    training_device = "cuda:0"  # Device on which to train. Set to "cpu" if no CUDA available.
    num_dataloader_workers = 30  # Number of dataloader workers for creating batches for training
    load_checkpoint_from_path = None   # Path to checkpoint if training should be continued from existing weights.
```
4. The terminal output will show under which subdirectory (named after timestamp) in `./results` the script will save the model checkpoints.

## Finetuning

Finetuning the model is performed by running `$ python main.py`. The first thing that should be done is modify `config.py`
to configure everything to your needs. The default `config.py` here in the repo reflects in general the setup in the paper, 
and we list all options and their meaning below in detail.

In each epoch during finetuning, the agent generates a high number of potential molecules, and stores the best ones 
under `./data/generated_molecules.pickle` (can be changed via `gumbeldore_config.destination_path` in the config, see below).
**Important:** This file is loaded as the supervised training dataset in each epoch, and it is always _merged_ with freshly
generated molecules. In particular, if the file is not deleted before an experiment run, __the experiment will continue
from this file__. 

Results are saved to a subfolder with the current timestamp under `results` (see config below to change it).
During a run, after an epoch, the current network weights will be saved: `best_model.pt` for the epoch where the overall best 
molecule so far has been found. `last_model.pt` for the most recent epoch. We also save a text file with the overall best
20 molecules found so far. We'd recommend logging everything that's being logged to `log.txt` also to Mlflow or similar.
See the method `Logger.log_metrics` in `logger.py`. 

### Config

We'll go through each config parameter in `config.py`. Also see the SIL repository [https://github.com/grimmlab/gumbeldore](https://github.com/grimmlab/gumbeldore)
for more detailed info on the parallelization (which is not really used here).

- `seed`: [int] Random seed. Defaults to 42.

#### Network architecture options

- `latent_dimension`: [int] Latent dimension $`d`$. Defaults to 128.
- `num_transformer_blocks`: [int] Number of layers in the stack of transformer blocks for the architecture. Defaults to 8.
- `num_heads`: [int] Number of heads in the multihead attention. Defaults to 8.
- `dropout`: [float] Dropout for feedforward layer in a transformer block. Defaults to 0.
- `use_rezero_transformer`: [bool] Whether ReZero normalization should be used or regular layer normalization. Defaults to `True`.

#### Molecule environment options

- `wall_clock_limit`: Optional[int]. Defaults to `None`, but is set to `3600*8` in the paper. Number of seconds how long an experiment may run.
    If `None`, no limit is set.
- `max_num_atoms`: [int] Maximum number of atoms that may exist in a designed molecule.

- `atom_vocabulary`: [dict] Specifies the vocabulary that is allowed in the design process. Important: Order matters, 
   because the indexing is derived from it. To explain how it works, it's best to look at the default:
   ```python
   self.atom_vocabulary = {  # Attention! Order matters!
      "C":    {"allowed": True, "atomic_number": 6, "valence": 4, "min_atoms": 0},
      "N":    {"allowed": True, "atomic_number": 7, "valence": 3, "min_atoms": 0},
      "O":    {"allowed": True, "atomic_number": 8, "valence": 2, "min_atoms": 0}
   }
   ```
   Each key is an atom name (e.g., a single letter, but naming is arbitrary) corresponding to a node type that can be placed
   on the graph. The value is again a dictionary, with `allowed` indicating if the atom may be placed (otherwise, it will be masked,
   useful for seeing how turning atom types on and off affect the performance). `atomic_number` is used to identify the atom type
   in `rdkit`. `min_atoms` is an optional int to force the agent to use a minimum number of atoms of that type in each design.

- `start_from_c_chains`: [bool] If `True` (Default), the agent starts each design from a carbon chain. See `start_c_chain_max_len` below in that case.
- `start_c_chain_max_len`: [int] Defaults to 1. Only relevant if `start_from_c_chains` is `True`. Then, if integer `n` is given here, 
    the agent starts its design from multiple start points with SMILES 'C', 'CC', 'CCC', ... 'CCC...C' (`n` times). So when set to 1 (default),
    the agent always starts from a single C-atom.
- `start_from_smiles`: Optional[str] Only relevant if `start_from_c_chains` is `False`. If set, then the agent will start all its designs
   from a graph corresponding to the given SMILES string. Defaults to `None`.
- `repeat_start_instances`: [int] Defaults to 1. If larger than 1, the agent uses multiple, independent search trees (for sampling) for each
    design from which it starts. 
            # Positive value x, where the actual objective with our molecule score will be set to obj = score - x * SA_score
- `synthetic_accessibility_in_objective_scale`: [float] Defaults to 0. This is actually not used in the paper. If a positive
    value `x` is given, the objective value will be augmented with synthetic accessibility score. I.e., let `score` be the original
    objective value, then the updated one is set to `score - x * SA_score`.
- `disallow_oxygen_bonding`: [bool] If `True`, oxygen may not bond with other oxygen atoms.
- `disallow_nitrogen_nitrogen_single_bond`: [bool] If `True`, nitrogen may not bond with nitrogen with single bonds (only double or triple).
- `disallow_rings`: [bool] If `True`, no rings are allowed, i.e., do not allow to increase the bond order between atoms that aren't bonded yet.
- `disallow_rings_larger_than`: [int] Defaults to 0. If this is greater than 3, all rings larger than the given value are not allowed.

#### Objective function options

- `GHGNN_model_path`: [str] Path to activity coefficient prediction model. Defaults to `objective_predictor/GH_GNN_IDAC/models/GHGNN.pth`.
- `GHGNN_hidden_dim`: [int] Latent dim of activity coefficient prediction model. Defaults to 113 and should not be altered.
- `objective_type`: [str] Objective function to use for finetuning as described in paper. 
Must be either **"IBA"** or **"DMBA"** (for TMB/DMBA).
- `num_predictor_workers`: [int] Number of parallel workers that distribute objective function evaluations between each other. Defaults to 10.
- `objective_predictor_batch_size`: [int] Batch size for inference of the activity coefficient model.
- `objective_gnn_device`: [str] Device on which the activity model lives. Defaults to "cpu".

#### Training flow options

- `load_checkpoint_from_path`: Optional[str]. If given, model checkpoint is loaded from this path. This should normally be set to the
    checkpoint after pretraining. Defaults to `None`.
- `load_optimizer_state`: [bool] If `True`, the optimizer state is also loaded from the checkpoint path above. Defaults to `False`.
- `num_dataloader_workers`: [int] Number of workers for preparing batches for supervised training. Defaults to 30.
- `CUDA_VISIBLE_DEVICES`: [str] ray.io sometimes needs help to recognize multiple GPUs. 
This variable will be set as the env var of the same name in each worker. Defaults to "0,1" (for two GPUs). Modify as needed.
- `training_device`: [str] Device on which network training is performed. Defaults to "cuda:0".
- `num_epochs`: [int] Number of epochs to train in total. Is overridden by `wall_clock_limit` above. Defaults to 1000.
- `scale_factor_level_one`: [float] Scale factor for loss at level 1 (pick atom to bond new atom to / pick first atom for bond increase). Defaults to 1. 
- `scale_factor_level_two`: [float] Scale factor for loss at level 2 (pick second atom for bond increase). Defaults to 1.
- `batch_size_training`: [int] Batch size to use for the supervised training during finetuning. Defaults to 64.
- `num_batches_per_epoch`: [int] Number of batches to use for supervised training during finetuning in each epoch. Defaults to 20. Can be `None`, then 
    one pass through the current generated dataset is done.
- `optimizer`: [dict] Optimizer configuration. Options are commented on in `config.py`.

#### Learning algorithm for finetuning options

- `gumbeldore_config`: [dict] This is the config for the self-improvement part. We go through the options below but only explain how they are used
    in our setup. For a more general discussion, please see the SIL repository [https://github.com/grimmlab/gumbeldore](https://github.com/grimmlab/gumbeldore).
    - `num_trajectories_to_keep:` [int] Number of best designed molecules to keep, which are used for supervised training during finetuning.
    - `keep_intermediate_trajectories`: [bool] If this is `True`, all designed molecules encountered in the trie are considered,
        not only the leaves. This is not used in the paper and defaults to `False`.
    - `devices_for_workers`: List[str] Number of parallel workers and on which devices their models live. Defaults to `["cuda:0"] * 1`.
    - `destination_path`: [str] Path where the generated molecules after each epoch are stored (and then loaded from to use as training dataset).
        Defaults to `"./data/generated_molecules.pickle"`. **Note**: You need to manually delete this file if you want to start fresh in a new run.
    Otherwise, it will always only be updated.
    - `batch_size_per_worker`: [int] If you start from a single atom, keep at 1.
    - `batch_size_per_cpu_worker`: [int] Same as above. This value is used for workers whose models live on the CPU.
    - `search_type`: [str] Keep at `'wor'` (sampling without replacement using stochastic beam search in multiple rounds).
    - `beam_width`: [int] Beam width for stochastic beam search. Defaults to 1024. Adjust to you hardware.
    - `num_rounds`: Union[int, Tuple[int, int]]. If it's a single integer, we sample for this many rounds exactly. If it's an (int, int)-tuple
      (as used in the paper), then we sample as long as it takes to obtain a new best molecule, but for a minimum of first entry rounds and a maximum of second entry rounds.
    - `deterministic`: [bool] Set to `True` to switch to deterministic beam seach. Not relevant for paper.
    - `nucleus_top_p`: [float] Top-p sampling nucleus size. Defaults to 1.0 (no nucleus sampling)
    - `pin_workers_to_core`: [bool] Default to `False`. If `True`, workers are pinned to single CPU threads, which can help with many workers on the CPU to prevent them from jamming each other with their numpy and pytorch operations.

#### Results and logging options

- `results_path`: [str] Path where to save results (checkpoint and top 20 molecules) to. Defaults to 
    ```python
    os.path.join("./results", datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))
    ```
- `log_to_file`: [bool] If logging output should also be saved to text file. Defaults to `True`.

## Acknowledgments

Thanks to the following repositories:

- [rezero](https://github.com/majumderb/rezero)
- [gumbeldore](https://github.com/grimmlab/gumbeldore), using
	- [unique-randomizer](https://github.com/google-research/unique-randomizer)
	- [stochastic-beam-search](https://github.com/wouterkool/stochastic-beam-search/tree/stochastic-beam-search)
