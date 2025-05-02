import torch
from torch import nn
from torch.nn.modules import TransformerEncoderLayer
from model.rztx import RZTXEncoderLayer
from config import MoleculeConfig
from molecule_design import MoleculeDesign


class MoleculeTransformer(nn.Module):
    """
    Molecular Transformer HYBRID architecture.
    - Uses OLD style output for L0 (virtual + real atoms -> dynamic size).
    - Uses NEW style output for L1 (virtual atom -> fixed global size).
    - Uses NEW/OLD style output for L2 (virtual atom -> fixed size).
    """
    def __init__(self, config: MoleculeConfig, device: torch.device = None):
        super().__init__()
        self.config = config
        self.device = torch.device("cpu") if device is None else device
        self.latent_dim = self.config.latent_dimension
        self.num_heads = self.config.num_heads
        self.num_blocks = self.config.num_transformer_blocks

        # --- Vocabulary and Dimension Info (Mostly from NEW model) ---
        self.max_real_atoms = self.config.max_num_atoms # Max REAL atoms allowed by config
        self.max_atoms_padded = self.max_real_atoms + 1 # Max padded size (incl. virtual)

        self.vocab_size = len(self.config.atom_vocabulary)
        valid_valences = [v for data in self.config.atom_vocabulary.values() if (v := data.get("valence")) is not None and v >= 0]
        max_possible_valence = max([0] + valid_valences)
        degree_padding_idx = max_possible_valence + 1

        num_atom_embeddings = self.vocab_size + 2
        atom_padding_idx = self.vocab_size + 1

        bond_padding_idx = MoleculeDesign.virtual_bond_idx + 1
        num_bond_embeddings = MoleculeDesign.virtual_bond_idx + 2

        # --- Input Embeddings (Keep from NEW model) ---
        self.virtual_atom_level_embedding = nn.Embedding(3, self.latent_dim)
        self.atom_learnable_embedding = nn.Embedding(num_atom_embeddings, self.latent_dim, padding_idx=atom_padding_idx)
        self.degree_learnable_embedding = nn.Embedding(max_possible_valence + 2, self.latent_dim, padding_idx=degree_padding_idx)
        self.bond_learnable_embedding = nn.Embedding(num_bond_embeddings, self.num_blocks * self.num_heads, padding_idx=bond_padding_idx)
        self.picked_atom_embedding = nn.Embedding(3, self.latent_dim) # padding_idx=0 removed based on new model

        # --- HYBRID Output Linear Layers ---
        # L0: OLD style (virtual + real atoms) -> Dynamic Size (B, N_batch)
        self.linear_l0_terminate = nn.Linear(self.latent_dim, 1) # Virtual atom -> Terminate logit
        self.linear_l0_select_atom = nn.Linear(self.latent_dim, 1) # Real atoms -> Select Atom i logits

        # L1: NEW style (virtual atom only) -> Fixed Global Size
        # Size = Add(V) + Select(N_max) + Replace(V) + Remove(1)
        max_l1_actions_global = self.max_real_atoms + self.vocab_size + self.vocab_size + 1
        self.output_linear_level_one = nn.Linear(self.latent_dim, max_l1_actions_global)

        # L2: NEW/OLD style (virtual atom only) -> Fixed Size 7
        self.output_linear_level_two = nn.Linear(self.latent_dim, 7)
        # --- End of HYBRID Output Linear Layers ---

        # --- Transformer Encoder (Keep from NEW model) ---
        self.encoder = nn.ModuleList([])
        # ... (encoder block creation logic remains the same) ...
        for _ in range(config.num_transformer_blocks):
            if not config.use_rezero_transformer:
                block = TransformerEncoderLayer(
                    d_model=self.latent_dim, nhead=self.num_heads,
                    dim_feedforward=4*self.latent_dim, dropout=config.dropout,
                    activation="gelu", batch_first=True, norm_first=True
                )
            else:
                block = RZTXEncoderLayer(
                    d_model=self.latent_dim, nhead=self.num_heads,
                    dim_feedforward=4 * self.latent_dim, dropout=config.dropout,
                    activation="gelu", batch_first=True
                )
            self.encoder.append(block)


    def forward(self, x: dict):
        """
        Forward pass for the HYBRID Molecule Transformer.
        Outputs dynamic logits for L0, fixed global for L1, fixed for L2.
        """
        batch_size, num_atoms_in_batch = x["atoms"].shape # N_batch = num atoms in this batch (incl. virtual)

        # --- 1. Construct Initial Atom Features (Keep from NEW model) ---
        atom_sequence = self.atom_learnable_embedding(x["atoms"]) # (B, N_batch, D)
        if num_atoms_in_batch > 1:
            degree_embeddings = self.degree_learnable_embedding(x["atoms_degree"][:, 1:]) # (B, N_batch-1, D)
            atom_sequence[:, 1:] = atom_sequence[:, 1:] + degree_embeddings
        level_embedding = self.virtual_atom_level_embedding(x["level_idx"]) # (B, D)
        atom_sequence[:, 0] = atom_sequence[:, 0] + level_embedding # Auto-broadcasts
        picked_embedding = self.picked_atom_embedding(x["picked_atom_mhe"]) # (B, N_batch, D)
        atom_sequence = atom_sequence + picked_embedding

        # --- 2. Prepare Attention Masks (Keep from NEW model) ---
        attn_mask = self.bond_learnable_embedding(x["bonds"]) # (B, N_batch, N_batch, num_blocks*num_heads)
        attn_mask = torch.permute(attn_mask, (0, 3, 1, 2))
        attn_mask = attn_mask.view(batch_size, self.num_blocks, self.num_heads, num_atoms_in_batch, num_atoms_in_batch)
        padding_attn_mask = x["additive_padding_attn_mask"].unsqueeze(1).unsqueeze(1) # (B, 1, 1, N_batch, N_batch)
        padding_attn_mask = padding_attn_mask.expand(-1, self.num_blocks, self.num_heads, -1, -1)
        final_attn_mask = attn_mask + padding_attn_mask

        # --- 3. Process through Transformer Encoder (Keep from NEW model) ---
        for i, trf_block in enumerate(self.encoder):
            mask_block_i = final_attn_mask[:, i, :, :, :]
            mask_block_folded = mask_block_i.reshape(batch_size * self.num_heads, num_atoms_in_batch, num_atoms_in_batch)
            atom_sequence = trf_block(atom_sequence, src_mask=mask_block_folded) # (B, N_batch, D)

        # --- 4. Generate Logits using HYBRID approach ---
        virtual_atom_state = atom_sequence[:, 0, :]  # (B, D)
        real_atom_states = atom_sequence[:, 1:, :] # (B, N_batch-1, D) <- States for atoms 1 to N_batch-1

        # L0: Dynamic Size (B, N_batch)
        logits_l0_terminate = self.linear_l0_terminate(virtual_atom_state) # (B, 1)
        logits_l0_select = self.linear_l0_select_atom(real_atom_states).squeeze(-1) # (B, N_batch-1)
        # Concatenate: Logit 0 (Terminate) + Logits 1..N_batch-1 (Select Atom i)
        logits_zero = torch.cat((logits_l0_terminate, logits_l0_select), dim=1) # (B, 1 + N_batch-1) = (B, N_batch)

        # L1: Fixed Global Size (using NEW model's layer)
        logits_one = self.output_linear_level_one(virtual_atom_state) # (B, GlobalL1Size)

        # L2: Fixed Size 7
        logits_two = self.output_linear_level_two(virtual_atom_state) # (B, 7)

        return logits_zero, logits_one, logits_two

    def get_weights(self):
        """Returns the model's state dict with tensors moved to CPU."""
        return dict_to_cpu(self.state_dict())


# Helper function (consider moving to a utils file if used elsewhere)
def dict_to_cpu(dictionary):
    """Recursively moves all tensors in a dictionary to CPU."""
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value) # Recurse for nested dictionaries
        else:
            cpu_dict[key] = value
    return cpu_dict
