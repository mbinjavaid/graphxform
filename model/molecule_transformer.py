import torch
from torch import nn
from torch.nn.modules import TransformerEncoderLayer
from model.rztx import RZTXEncoderLayer
from config import MoleculeConfig


class MoleculeTransformer(nn.Module):
    """
    Molecular Transformer for our solvent design.
    """
    def __init__(self, config: MoleculeConfig, device: torch.device = None):
        super().__init__()
        self.config = config
        self.device = torch.device("cpu") if device is None else device
        self.latent_dim = self.config.latent_dimension
        self.num_heads = self.config.num_heads
        self.num_blocks = self.config.num_transformer_blocks

        self.num_possible_atoms = len(self.config.atom_vocabulary) + 1  # all atoms in vocab plus one virtual atom
        max_possible_valence = max([self.config.atom_vocabulary[x]["valence"] for x in self.config.atom_vocabulary])

        self.virtual_atom_level_embedding = nn.Embedding(num_embeddings=2, embedding_dim=self.latent_dim)
        self.atom_learnable_embedding = nn.Embedding(num_embeddings=self.num_possible_atoms + 1, embedding_dim=self.latent_dim,
                                                     padding_idx=self.num_possible_atoms)
        self.degree_learnable_embedding = nn.Embedding(num_embeddings=max_possible_valence + 2, embedding_dim=self.latent_dim,
                                                       padding_idx=max_possible_valence + 1)  # an atom can be connected to at most 4 other atoms
        # We map each bond to a scalar, additive attention bias for each transformer block and head. We perform the
        # embedding for all layers in one go.
        self.bond_learnable_embedding = nn.Embedding(num_embeddings=max_possible_valence + 3,
                                                     embedding_dim=self.num_blocks * self.num_heads,
                                                     padding_idx=max_possible_valence + 2)

        # Maps the One-(or zero)-hot encoded vector to an indicator for each atom if it was picked at level 1 (only
        # relevant for level 2). In particular, an index of 0 maps to all 0s.
        self.picked_atom_embedding = nn.Embedding(num_embeddings=2, embedding_dim=self.latent_dim, padding_idx=0)

        # Mapping the transformed virtual atom to the "pick atom or terminate" logits
        self.virtual_atom_linear = nn.Linear(self.latent_dim, self.num_possible_atoms + 1)
        # Mapping latent atoms to two logits: One for level 1, and one for level 2
        self.bond_atom_linear = nn.Linear(self.latent_dim, 2)

        # Transformer itself
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
        batch_size, num_atoms = x["atoms"].shape

        atom_sequence = self.atom_learnable_embedding(x["atoms"])  # (B, num_atoms, latent_dim)
        # add the embedded degree to all but the virtual atom. Shape stays (B, num_atoms, latent_dim)
        #print(x["atoms_degree"].shape)
        #print(x["atoms_degree"][:, 1:])
        atom_sequence[:, 1:] = atom_sequence[:, 1:] + self.degree_learnable_embedding(x["atoms_degree"][:, 1:])
        # add the embedded level index to the virtual atom.
        atom_sequence[:, 0] = atom_sequence[:, 0] + self.virtual_atom_level_embedding(x["level_idx"])
        # add the embedding indicating whether an atom was picked to the sequence
        atom_sequence = atom_sequence + self.picked_atom_embedding(x["picked_atom_ohe"])


        # Prepare the additive attention masks
        attn_mask = self.bond_learnable_embedding(x["bonds"])  # (B, num_atoms, num_atoms, num_trf_blocks*num_heads)
        attn_mask = torch.permute(attn_mask, (0, 3, 1, 2)).view(batch_size, self.num_blocks, self.num_heads, num_atoms, num_atoms)
        padding_attn_mask = x["additive_padding_attn_mask"][:, None, :, :].repeat((1, self.num_blocks*self.num_heads, 1, 1))
        padding_attn_mask = padding_attn_mask.view(batch_size, self.num_blocks, self.num_heads, num_atoms, num_atoms)
        attn_mask = attn_mask + padding_attn_mask

        for i, trf_block in enumerate(self.encoder):
            # get the additive mask for the i-th block, and fold the heads into the batch dimension.
            mask_block_folded = attn_mask[:, i, :, :, :].reshape(batch_size * self.num_heads, num_atoms, num_atoms)
            atom_sequence = trf_block(atom_sequence, src_mask=mask_block_folded)

        virtual_atom = atom_sequence[:, 0, :]  # (B, latent_dim)
        level_zero_logits = self.virtual_atom_linear(virtual_atom)  # (B, num possible atoms + 2)

        level_one_and_two_logits = self.bond_atom_linear(atom_sequence[:, 1:, :])  # (B, num_atoms - 1, 2)
        level_one_logits = level_one_and_two_logits[:, :, 0]
        level_two_logits = level_one_and_two_logits[:, :, 1]

        return level_zero_logits, level_one_logits, level_two_logits

    def get_weights(self):
        return dict_to_cpu(self.state_dict())


def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict
