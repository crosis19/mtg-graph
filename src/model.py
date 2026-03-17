"""Phase 4: Heterogeneous Graph Transformer (HGT) for MTG metagame prediction.

Two-headed model:
  Head 1 — Archetype Emergence: Predict which archetypes will gain meta share
  Head 2 — Tournament Top 8: Predict which archetypes will place in a tournament's top 8

Architecture:
  1. Per-type linear projections to shared hidden dim
  2. HGT message-passing layers (heterogeneous attention)
  3. Task-specific prediction heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear


class MTGMetagameHGT(nn.Module):
    """Two-headed HGT for metagame prediction.

    Parameters
    ----------
    metadata : tuple
        (node_types, edge_types) from HeteroData.metadata()
    node_dims : dict
        {node_type: input_feature_dim} for each node type
    hidden_dim : int
        Hidden dimension for all layers
    num_heads : int
        Number of attention heads in HGT
    num_layers : int
        Number of HGT message-passing layers
    dropout : float
        Dropout rate
    """

    def __init__(
        self,
        metadata: tuple,
        node_dims: dict,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Per-type input projections to shared hidden dim
        self.input_projections = nn.ModuleDict()
        for node_type in metadata[0]:
            self.input_projections[node_type] = Linear(
                node_dims[node_type], hidden_dim
            )

        # HGT convolution layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HGTConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=metadata,
                    heads=num_heads,
                )
            )
            # Per-type layer norms
            norm_dict = nn.ModuleDict()
            for node_type in metadata[0]:
                norm_dict[node_type] = nn.LayerNorm(hidden_dim)
            self.norms.append(norm_dict)

        self.dropout = nn.Dropout(dropout)

        # ── Head 1: Archetype Emergence ──
        # Predicts change in meta share for each archetype
        self.emergence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),  # Regression: predicted share delta
        )

        # ── Head 2: Tournament Top 8 ──
        # Given (archetype_emb, tournament_emb) → probability of top 8 placement
        self.top8_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, data: HeteroData) -> dict:
        """Forward pass through HGT backbone + both prediction heads.

        Returns dict with:
          - 'node_embeddings': {node_type: tensor} after HGT
          - 'emergence_scores': (n_archetypes,) predicted share changes
          - 'top8_logits': None (computed separately via predict_top8)
        """
        # Project each node type to shared hidden dim
        x_dict = {}
        for node_type in data.node_types:
            x_dict[node_type] = self.input_projections[node_type](data[node_type].x)

        # Build edge_index dict for HGT
        edge_index_dict = {}
        for edge_type in data.edge_types:
            edge_index_dict[edge_type] = data[edge_type].edge_index

        # HGT message passing
        for i, conv in enumerate(self.convs):
            x_dict_new = conv(x_dict, edge_index_dict)
            # Residual + norm + dropout
            for node_type in x_dict:
                if node_type in x_dict_new:
                    x_dict[node_type] = self.norms[i][node_type](
                        x_dict[node_type] + self.dropout(x_dict_new[node_type])
                    )

        # Head 1: Emergence prediction (per archetype)
        arch_embeddings = x_dict["archetype"]
        emergence_scores = self.emergence_head(arch_embeddings).squeeze(-1)

        return {
            "node_embeddings": x_dict,
            "emergence_scores": emergence_scores,
        }

    def predict_top8(
        self,
        arch_embeddings: torch.Tensor,
        tourney_embeddings: torch.Tensor,
        arch_indices: torch.Tensor,
        tourney_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Predict top-8 probability for (archetype, tournament) pairs.

        Parameters
        ----------
        arch_embeddings : (n_archetypes, hidden_dim)
        tourney_embeddings : (n_tournaments, hidden_dim)
        arch_indices : (n_pairs,) indices into arch_embeddings
        tourney_indices : (n_pairs,) indices into tourney_embeddings

        Returns
        -------
        probs : (n_pairs,) probability of top 8 placement
        """
        arch_emb = arch_embeddings[arch_indices]
        tourney_emb = tourney_embeddings[tourney_indices]
        combined = torch.cat([arch_emb, tourney_emb], dim=-1)
        return self.top8_head(combined).squeeze(-1)
