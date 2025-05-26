import torch
from torch import nn
from torch.nn.modules import TransformerEncoderLayer
from model.rztx import RZTXEncoderLayer  # Assuming this is a custom ReZero-Transformer layer
from config import MoleculeConfig
from molecule_design import MoleculeDesign  # For bond_types, virtual_bond_idx


class MoleculeTransformer(nn.Module):
    """
    Molecular Transformer architecture.
    - L0 logits (Terminate, Select Existing Atom) are dynamic based on batch atom count.
    - L1 logits (Add Atom, Select Existing Atom, Replace Atom, Remove Atom) are dynamic.
        - Add, Replace, Remove components from virtual_atom_state.
        - Select Existing Atom component from real_atom_states.
    - L2 logits (Set Bond Type) are fixed size, from virtual_atom_state.
    """

    def __init__(self, config: MoleculeConfig, device: torch.device = None):
        super().__init__()
        self.config = config
        self.device = torch.device("cpu") if device is None else device
        self.latent_dim = self.config.latent_dimension
        self.num_heads = self.config.num_heads
        self.num_blocks = self.config.num_transformer_blocks

        # --- Vocabulary and Dimension Info ---
        # self.max_real_atoms = self.config.max_num_atoms # No longer needed for output head sizing
        # self.max_atoms_padded = self.max_real_atoms + 1   # No longer needed for output head sizing

        # Determine max_bond_actions for L2 output
        # Ensure MoleculeDesign.bond_types is accessible and correct
        try:
            # Accessing as a class attribute if available
            bond_types_keys = MoleculeDesign.bond_types.keys()
        except AttributeError:
            # Fallback if it's an instance attribute or needs instantiation (adjust as needed)
            # This might indicate a need to pass MoleculeDesign instance or bond_types directly
            print("Warning: MoleculeDesign.bond_types not directly accessible as class attribute. Attempting fallback.")
            # Example fallback: You might need to instantiate or get from config if truly dynamic
            # For now, assuming a default or that it's resolvable.
            # If MoleculeDesign.bond_types is simple like {0:'NONE', 1:'SINGLE', ...},
            # you could hardcode a default if necessary, but ideally it's sourced correctly.
            default_bond_types_len = 4  # Example: None, Single, Double, Triple
            max_bond_actions = default_bond_types_len + 1  # +1 for a specific "no bond" or "remove bond" action
            print(f"Using fallback max_bond_actions: {max_bond_actions}")
        else:
            max_bond_actions = len(bond_types_keys) + 1  # +1 for the "no bond" or equivalent action

        self.vocab_size = len(self.config.atom_vocabulary)
        # Calculate max_possible_valence robustly
        valid_valences = []
        if hasattr(self.config, 'atom_vocabulary') and isinstance(self.config.atom_vocabulary, dict):
            for atom_data in self.config.atom_vocabulary.values():
                if isinstance(atom_data, dict) and "valence" in atom_data and atom_data["valence"] is not None:
                    if isinstance(atom_data["valence"], (int, float)) and atom_data["valence"] >= 0:
                        valid_valences.append(atom_data["valence"])
        max_possible_valence = max([0] + valid_valences) if valid_valences else 0  # Default to 0 if no valid valences
        degree_padding_idx = max_possible_valence + 1

        num_atom_embeddings = self.vocab_size + 2  # vocab + virtual + padding
        atom_padding_idx = self.vocab_size + 1  # Assuming virtual is 0, then vocab, then padding

        # Ensure MoleculeDesign.virtual_bond_idx is accessible
        try:
            virtual_bond_idx = MoleculeDesign.virtual_bond_idx
        except AttributeError:
            print(
                "Warning: MoleculeDesign.virtual_bond_idx not accessible. Using a default (e.g., 0 or max_bond_type+1).")
            # This needs to be consistent with how bond matrices are prepared.
            # If bond types are 0,1,2,3 and virtual_bond_idx is, say, 4
            virtual_bond_idx = max_bond_actions  # A plausible default if it's beyond real bonds
        bond_padding_idx = virtual_bond_idx + 1
        num_bond_embeddings = virtual_bond_idx + 2

        # --- Input Embeddings ---
        self.virtual_atom_level_embedding = nn.Embedding(3, self.latent_dim)  # For L0, L1, L2 states
        self.atom_learnable_embedding = nn.Embedding(num_atom_embeddings, self.latent_dim, padding_idx=atom_padding_idx)
        self.degree_learnable_embedding = nn.Embedding(max_possible_valence + 2, self.latent_dim,
                                                       padding_idx=degree_padding_idx)
        self.bond_learnable_embedding = nn.Embedding(num_bond_embeddings, self.num_blocks * self.num_heads,
                                                     padding_idx=bond_padding_idx)
        self.picked_atom_embedding = nn.Embedding(3, self.latent_dim)  # 0:not picked, 1:L0 picked, 2:L1 picked

        # --- Output Linear Layers ---
        # L0: Dynamic Size (B, N_batch)
        self.linear_l0_terminate = nn.Linear(self.latent_dim, 1)  # Virtual atom -> Terminate logit
        self.linear_l0_select_atom = nn.Linear(self.latent_dim, 1)  # Real atoms -> Select Atom i logits

        # L1: Dynamic Size, components from virtual and real states
        # Actions from virtual_atom_state: Add Atom (V types), Replace Atom (V types), Remove Atom (1 type)
        l1_virtual_output_size = self.vocab_size + self.vocab_size + 1
        self.linear_l1_virtual_add_replace_remove = nn.Linear(self.latent_dim, l1_virtual_output_size)
        # Actions from real_atom_states: Select Existing Atom (1 logit per real atom)
        self.linear_l1_select_existing = nn.Linear(self.latent_dim, 1)

        # L2: Fixed Size (from virtual_atom_state)
        self.output_linear_level_two = nn.Linear(self.latent_dim, max_bond_actions)
        # --- End of Output Linear Layers ---

        # --- Transformer Encoder ---
        self.encoder = nn.ModuleList([])
        for _ in range(config.num_transformer_blocks):
            if not config.use_rezero_transformer:
                block = TransformerEncoderLayer(
                    d_model=self.latent_dim, nhead=self.num_heads,
                    dim_feedforward=4 * self.latent_dim, dropout=config.dropout,
                    activation="gelu", batch_first=True, norm_first=True
                )
            else:  # Assuming RZTXEncoderLayer is a custom ReZero-Transformer layer
                block = RZTXEncoderLayer(
                    d_model=self.latent_dim, nhead=self.num_heads,
                    dim_feedforward=4 * self.latent_dim, dropout=config.dropout,
                    activation="gelu", batch_first=True
                )
            self.encoder.append(block)

    def forward(self, x: dict):
        """
        Forward pass for the Molecule Transformer.
        Outputs dynamic logits for L0 and L1, and fixed-size logits for L2.
        """
        batch_size, num_atoms_in_batch = x["atoms"].shape  # N_batch = num atoms in this batch (incl. virtual)

        # --- 1. Construct Initial Atom Features ---
        atom_sequence = self.atom_learnable_embedding(x["atoms"])  # (B, N_batch, D)
        if num_atoms_in_batch > 1:  # Avoid indexing error if N_batch is 1 (only virtual atom)
            # Degree embedding is for real atoms, so starts from index 1 of atoms_degree
            degree_embeddings = self.degree_learnable_embedding(x["atoms_degree"][:, 1:])  # (B, N_batch-1, D)
            atom_sequence[:, 1:] = atom_sequence[:, 1:] + degree_embeddings

        level_embedding = self.virtual_atom_level_embedding(x["level_idx"])  # (B, D)

        # INCORRECT:
        # atom_sequence[:, 0] = atom_sequence[:, 0] + level_embedding.unsqueeze(1)  # Add to virtual atom, ensure broadcast

        # corrected:
        atom_sequence[:, 0] = atom_sequence[:, 0] + level_embedding  # Add to virtual atom features

        picked_embedding = self.picked_atom_embedding(x["picked_atom_mhe"])  # (B, N_batch, D)
        atom_sequence = atom_sequence + picked_embedding

        # --- 2. Prepare Attention Masks ---
        # Ensure bond embeddings are correctly shaped for multi-head attention bias
        attn_mask_bias = self.bond_learnable_embedding(x["bonds"])  # (B, N_batch, N_batch, num_blocks*num_heads)
        attn_mask_bias = torch.permute(attn_mask_bias, (0, 3, 1, 2))  # (B, num_blocks*num_heads, N_batch, N_batch)
        # Reshape to (B, num_blocks, num_heads, N_batch, N_batch) for block-wise processing
        attn_mask_bias = attn_mask_bias.view(batch_size, self.num_blocks, self.num_heads, num_atoms_in_batch,
                                             num_atoms_in_batch)

        # Additive padding mask from input
        padding_attn_mask = x["additive_padding_attn_mask"].unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N_batch, N_batch)
        # No need to expand padding_attn_mask if it's directly added in the loop,
        # or expand it once if preferred:
        # padding_attn_mask = padding_attn_mask.expand(-1, self.num_blocks, self.num_heads, -1, -1)
        # final_attn_mask = attn_mask_bias + padding_attn_mask # If expanded

        # --- 3. Process through Transformer Encoder ---
        current_src = atom_sequence
        for i, trf_block in enumerate(self.encoder):
            # Get bias for the current block, across all heads
            block_attn_bias = attn_mask_bias[:, i, :, :, :]  # (B, num_heads, N_batch, N_batch)
            # Add padding mask for this block
            current_block_mask = block_attn_bias + padding_attn_mask.squeeze(
                1)  # Squeeze block dim if not expanded before

            # Fold batch and head dimensions for TransformerEncoderLayer's expected mask format if needed
            # Or pass directly if layer handles (B, num_heads, N, N)
            # Standard PyTorch TransformerEncoderLayer expects (N_total_heads, N, N) or (N,N)
            # If batch_first=True, it expects (B*num_heads, N, N) if mask is 3D.
            # For (B, num_heads, N, N) it should be handled by setting attn_mask in trf_block(src, attn_mask=...)
            # Let's assume the layer can handle (B, num_heads, N_query, N_key) or will be reshaped.
            # For standard layer, it's often (B*num_heads, N, N)
            mask_for_block_folded = current_block_mask.reshape(batch_size * self.num_heads, num_atoms_in_batch,
                                                               num_atoms_in_batch)
            current_src = trf_block(current_src, src_mask=mask_for_block_folded)
        atom_sequence = current_src  # Final atom representations

        # --- 4. Generate Logits ---
        virtual_atom_state = atom_sequence[:, 0, :]  # (B, D)

        # Handle cases where there are no real atoms (N_batch = 1)
        if num_atoms_in_batch > 1:
            real_atom_states = atom_sequence[:, 1:, :]  # (B, N_batch-1, D)
        else:
            real_atom_states = torch.empty((batch_size, 0, self.latent_dim),
                                           dtype=virtual_atom_state.dtype,
                                           device=virtual_atom_state.device)

        # L0 Logits: Dynamic Size (B, N_batch) or (B,1) if N_batch=1
        logits_l0_terminate = self.linear_l0_terminate(virtual_atom_state)  # (B, 1)
        if num_atoms_in_batch > 1:
            logits_l0_select = self.linear_l0_select_atom(real_atom_states).squeeze(-1)  # (B, N_batch-1)
            logits_zero = torch.cat((logits_l0_terminate, logits_l0_select), dim=1)  # (B, N_batch)
        else:  # Only virtual atom exists, only terminate action is sensible for L0.
            logits_zero = logits_l0_terminate  # (B, 1) (Mask should also reflect this)

        # L1 Logits: Dynamic Size (B, 2*V + N_batch)
        # Part 1: From virtual_atom_state (Add, Replace, Remove)
        virtual_l1_logits_combined = self.linear_l1_virtual_add_replace_remove(virtual_atom_state)  # (B, 2*V + 1)

        logits_l1_add = virtual_l1_logits_combined[:, :self.vocab_size]  # (B, V)
        logits_l1_replace = virtual_l1_logits_combined[:, self.vocab_size: 2 * self.vocab_size]  # (B, V)
        logit_l1_remove = virtual_l1_logits_combined[:, 2 * self.vocab_size:]  # (B, 1)

        # Part 2: From real_atom_states (Select Existing)
        if num_atoms_in_batch > 1:
            logits_l1_select_existing = self.linear_l1_select_existing(real_atom_states).squeeze(-1)  # (B, N_batch-1)
        else:  # No real atoms to select
            logits_l1_select_existing = torch.empty((batch_size, 0),
                                                    dtype=virtual_atom_state.dtype,
                                                    device=virtual_atom_state.device)

        # Concatenate all L1 logits in the defined order:
        # [Add(V) | SelectExisting(N_batch-1) | Replace(V) | Remove(1)]
        # Total L1 actions: V (add) + (N_batch-1) (select) + V (replace) + 1 (remove)
        # = 2V + N_batch
        logits_one = torch.cat(
            (logits_l1_add, logits_l1_select_existing, logits_l1_replace, logit_l1_remove),
            dim=1
        )

        # L2 Logits: Fixed Size
        logits_two = self.output_linear_level_two(virtual_atom_state)  # (B, max_bond_actions)

        return logits_zero, logits_one, logits_two

    def get_weights(self):
        """Returns the model's state dict with tensors moved to CPU."""
        return dict_to_cpu(self.state_dict())


def dict_to_cpu(dictionary: dict) -> dict:
    """Recursively moves all tensors in a dictionary (and its sub-dictionaries) to CPU."""
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)  # Recurse for nested dictionaries
        else:
            cpu_dict[key] = value
    return cpu_dict
