import torch
from torch import nn
from torch.nn.modules import TransformerEncoderLayer
from model.rztx import RZTXEncoderLayer
from config import MoleculeConfig
from molecule_design import MoleculeDesign


class MoleculeTransformer(nn.Module):
    """
    Molecular Transformer adapted for the 3-level transformation action space.
    Outputs logits corresponding to the action space defined by TransformationMoleculeDataset,
    generating predictions based on the final virtual atom state.
    """
    def __init__(self, config: MoleculeConfig, device: torch.device = None):
        super().__init__()
        self.config = config
        self.device = torch.device("cpu") if device is None else device
        self.latent_dim = self.config.latent_dimension
        self.num_heads = self.config.num_heads
        self.num_blocks = self.config.num_transformer_blocks

        # --- Vocabulary and Dimension Info ---
        # Ensure config provides these values
        if not hasattr(config, 'max_atoms'):
            raise ValueError("MoleculeConfig must provide 'max_atoms' (maximum padded atom count)")
        self.max_atoms = self.config.max_num_atoms + 1  # Max padded size (including virtual atom)

        if not hasattr(config, 'atom_vocabulary'):
             raise ValueError("MoleculeConfig must provide 'atom_vocabulary'")
        self.vocab_size = len(self.config.atom_vocabulary) # Number of actual atom types (e.g., C, O, N...)

        # Max valence needed for degree embedding padding index
        valid_valences = [v for atom, data in self.config.atom_vocabulary.items() if (v := data.get("valence")) is not None and v >= 0]
        max_possible_valence = max([0] + valid_valences) # Use 0 if no valid valence info
        degree_padding_idx = max_possible_valence + 1 # +1 because degree=0 is valid, +1 for padding itself

        # Number of atom types for embedding = vocab_size + virtual_atom + padding_atom
        num_atom_embeddings = self.vocab_size + 2
        atom_padding_idx = self.vocab_size + 1 # Index for padding atom type

        # Number of bond types for embedding = max_order + virtual_bond + padding_bond
        # Assuming MoleculeDesign.maximum_bond_order (e.g., 6) and MoleculeDesign.virtual_bond_idx (e.g., 7) are defined
        # Bond orders 0..6 -> indices 0..6
        # Virtual bond -> index 7 (MoleculeDesign.virtual_bond_idx)
        # Padding bond -> index 8
        bond_padding_idx = MoleculeDesign.virtual_bond_idx + 1
        num_bond_embeddings = MoleculeDesign.virtual_bond_idx + 2 # Total embeddings needed

        # --- Input Embeddings (Largely Unchanged) ---
        # Embedding for action level (0, 1, 2) - added to virtual atom
        self.virtual_atom_level_embedding = nn.Embedding(num_embeddings=3, embedding_dim=self.latent_dim)

        # Embedding for atom types (C, N, O, ..., virtual, padding)
        self.atom_learnable_embedding = nn.Embedding(
            num_embeddings=num_atom_embeddings,
            embedding_dim=self.latent_dim,
            padding_idx=atom_padding_idx
        )

        # Embedding for atom degrees (0..max_valence, padding)
        self.degree_learnable_embedding = nn.Embedding(
            num_embeddings=max_possible_valence + 2, # Degree 0 to max_valence + padding_idx
            embedding_dim=self.latent_dim,
            padding_idx=degree_padding_idx
        )

        # Embedding for bond types (0..max_order, virtual, padding) - used for attention bias
        self.bond_learnable_embedding = nn.Embedding(
            num_embeddings=num_bond_embeddings,
            embedding_dim=self.num_blocks * self.num_heads,
            padding_idx=bond_padding_idx
        )

        # Embedding for picked atom status (0=not picked, 1=anchor i, 2=target j)
        self.picked_atom_embedding = nn.Embedding(
            num_embeddings=3, # 0, 1, 2
            embedding_dim=self.latent_dim,
            # padding_idx=0 # Optional: Can treat 'not picked' as padding if desired.
                          # If not using padding_idx=0, ensure index 0 has a meaningful learned embedding.
        )

        # --- NEW: Output Linear Layers ---
        # Takes final virtual atom state -> Logits for each level based on MAX possible size

        # Level 0: Select atom i (indices 0 to max_atoms-1)
        self.output_linear_level_zero = nn.Linear(self.latent_dim, self.max_atoms)

        # Level 1: Select Existing j (max_atoms-1 indices) + Add New j (vocab_size types) + Replace i (vocab_size types) + Remove i (1 action)
        # Total size = (max_atoms - 1) + vocab_size + vocab_size + 1
        max_l1_actions = (self.max_atoms - 1) + self.vocab_size + self.vocab_size + 1
        self.output_linear_level_one = nn.Linear(self.latent_dim, max_l1_actions)

        # Level 2: Set Bond Order 0-6 (7 actions)
        self.output_linear_level_two = nn.Linear(self.latent_dim, 7)
        # --- End of NEW Output Linear Layers ---

        # --- Transformer Encoder (Unchanged Logic) ---
        self.encoder = nn.ModuleList([])
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
        Forward pass for the Molecule Transformer.

        Args:
            x (dict): Batch dictionary from TransformationMoleculeDataset containing keys like:
                      'atoms', 'atoms_degree', 'bonds', 'level_idx', 'picked_atom_mhe',
                      'additive_padding_attn_mask', etc. The shapes of tensors like 'atoms',
                      'bonds' will depend on the number of atoms in the largest molecule
                      in the *current batch* (`num_atoms_in_batch`).

        Returns:
            tuple: (logits_zero, logits_one, logits_two)
                   Logits for each action level. Shapes are fixed based on config.max_atoms:
                   - logits_zero: (Batch, config.max_atoms)
                   - logits_one:  (Batch, (config.max_atoms-1) + 2*vocab_size + 1)
                   - logits_two:  (Batch, 7)
                   The training loop must use feasibility masks from the input 'x'
                   (which have shapes based on the *current batch's* num_atoms_in_batch)
                   to correctly calculate the loss against these fixed-size logit tensors.
        """
        batch_size, num_atoms_in_batch = x["atoms"].shape # N = num_atoms_in_batch varies per batch

        # --- 1. Construct Initial Atom Features ---
        # Base atom type embeddings
        atom_sequence = self.atom_learnable_embedding(x["atoms"])  # (B, N, D) D=latent_dim

        # Add degree embedding to REAL atoms (indices 1 to N-1)
        # Ensure degrees are clipped or handled if they exceed embedding range (padding_idx handles this)
        if num_atoms_in_batch > 1: # Avoid slicing error if batch only has virtual atoms (unlikely)
            degree_embeddings = self.degree_learnable_embedding(x["atoms_degree"][:, 1:]) # (B, N-1, D)
            atom_sequence[:, 1:] = atom_sequence[:, 1:] + degree_embeddings

        # Add level embedding to VIRTUAL atom (index 0)
        level_embedding = self.virtual_atom_level_embedding(x["level_idx"]) # (B, D)
        atom_sequence[:, 0] = atom_sequence[:, 0] + level_embedding.unsqueeze(1).squeeze(1) # Ensure correct broadcasting

        # Add picked atom embedding to ALL atoms (virtual + real)
        picked_embedding = self.picked_atom_embedding(x["picked_atom_mhe"]) # (B, N, D)
        atom_sequence = atom_sequence + picked_embedding

        # --- 2. Prepare Attention Masks ---
        # Bond embeddings -> Additive Attention Bias
        attn_mask = self.bond_learnable_embedding(x["bonds"])  # (B, N, N, num_blocks*num_heads)
        attn_mask = torch.permute(attn_mask, (0, 3, 1, 2)) # (B, num_blocks*num_heads, N, N)
        attn_mask = attn_mask.view(batch_size, self.num_blocks, self.num_heads, num_atoms_in_batch, num_atoms_in_batch)

        # Additive padding mask (prevents attention to padding atoms beyond N)
        # Input mask shape: (B, N, N) -> Expand for blocks and heads
        padding_attn_mask = x["additive_padding_attn_mask"] # (B, N, N)
        padding_attn_mask = padding_attn_mask.unsqueeze(1).unsqueeze(1) # (B, 1, 1, N, N)
        # Repeat mask across all blocks and heads
        padding_attn_mask = padding_attn_mask.expand(-1, self.num_blocks, self.num_heads, -1, -1) # Use expand for efficiency

        # Combine bond bias and padding mask
        # Where padding_attn_mask is -inf, the result will be -inf. Where it's 0, the bond bias is kept.
        final_attn_mask = attn_mask + padding_attn_mask # Additive combination

        # --- 3. Process through Transformer Encoder ---
        for i, trf_block in enumerate(self.encoder):
            # Extract mask for the current block, shape (B, num_heads, N, N)
            mask_block_i = final_attn_mask[:, i, :, :, :]
            # Reshape for TransformerEncoderLayer: Needs (B*num_heads, N, N) if using default MHA masking
            mask_block_folded = mask_block_i.reshape(batch_size * self.num_heads, num_atoms_in_batch, num_atoms_in_batch)

            # Pass through the transformer block
            # Note: Make sure the TransformerEncoderLayer implementation correctly handles the mask shape.
            # PyTorch default expects (N, N) or (B*num_heads, N, N) or (B, N, N) depending on version/args.
            # The folded mask (B*num_heads, N, N) is usually safe for batch_first=True.
            atom_sequence = trf_block(atom_sequence, src_mask=mask_block_folded)

        # --- 4. Generate Logits from Final Virtual Atom State ---
        # Extract the final state of the virtual atom (index 0)
        virtual_atom_state = atom_sequence[:, 0, :]  # Shape: (batch_size, self.latent_dim)

        # Apply the output linear layers defined in __init__
        logits_zero = self.output_linear_level_zero(virtual_atom_state) # Shape: (B, config.max_atoms)
        logits_one = self.output_linear_level_one(virtual_atom_state)   # Shape: (B, max_l1_actions)
        logits_two = self.output_linear_level_two(virtual_atom_state)   # Shape: (B, 7)

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
