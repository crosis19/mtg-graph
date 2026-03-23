"""Deck Composition Predictor — HGT backbone + attention-pooled archetype embeddings.

Predicts which cards belong in which deck archetypes using BPR ranking loss
and leave-one-out archetype pooling to prevent information leakage.

Architecture:
  1. Card embeddings: HGT message passing over card-card synergy edges and
     set edges (no archetype edges — cards don't know their decks)
  2. Archetype embeddings: attention-weighted pool over their deck cards'
     HGT embeddings. During training, the card being predicted is excluded
     from the pool (leave-one-out) to prevent information leakage.
  3. Prediction head: concat(card_emb, arch_emb, card_emb * arch_emb) → logit
"""

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear


# Only use card-level edges for HGT — no archetype/tournament edges
CARD_EDGE_TYPES = {
    ("card", "keyword_synergy", "card"),
    ("card", "mechanical_synergy", "card"),
    ("card", "semantic_synergy", "card"),
    ("set", "printed_in", "card"),
    ("card", "from_set", "set"),
}


class DeckPredictor(nn.Module):
    """Deck composition predictor with HGT card backbone + attention-pooled archetypes.

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
        dropout: float = 0.3,
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

        # HGT convolution layers (card-level only)
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
            norm_dict = nn.ModuleDict()
            for node_type in metadata[0]:
                norm_dict[node_type] = nn.LayerNorm(hidden_dim)
            self.norms.append(norm_dict)

        self.dropout = nn.Dropout(dropout)

        # Attention pooling for archetype embeddings
        self.arch_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Deck inclusion prediction head
        # Input: [card_emb, arch_emb, card_emb * arch_emb]
        self.deck_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _run_hgt(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """Run HGT message passing and return node embeddings."""
        x_dict = {}
        for node_type in data.node_types:
            x_dict[node_type] = self.input_projections[node_type](data[node_type].x)

        edge_index_dict = {}
        for edge_type in data.edge_types:
            if edge_type in CARD_EDGE_TYPES:
                edge_index_dict[edge_type] = data[edge_type].edge_index

        for i, conv in enumerate(self.convs):
            x_dict_new = conv(x_dict, edge_index_dict)
            for node_type in x_dict:
                if node_type in x_dict_new:
                    x_dict[node_type] = self.norms[i][node_type](
                        x_dict[node_type] + self.dropout(x_dict_new[node_type])
                    )

        return x_dict

    def _get_arch_card_indices(self, data: HeteroData) -> dict[int, list[int]]:
        """Gather card indices per archetype from maindecks + sideboards edges."""
        arch_card_indices: dict[int, list[int]] = {}
        for edge_name in ["maindecks", "sideboards"]:
            et = ("archetype", edge_name, "card")
            if et in data.edge_types:
                ei = data[et].edge_index
                for j in range(ei.shape[1]):
                    a_idx = int(ei[0, j])
                    c_idx = int(ei[1, j])
                    arch_card_indices.setdefault(a_idx, []).append(c_idx)
        return arch_card_indices

    def _attention_pool(
        self,
        card_emb: torch.Tensor,
        card_indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Attention-pool card embeddings for one archetype.

        Returns (pooled_embedding, attention_weights, card_embeddings_subset).
        """
        c_indices = torch.tensor(card_indices, dtype=torch.long, device=card_emb.device)
        deck_card_embs = card_emb[c_indices]  # (n_deck_cards, hidden_dim)
        attn_scores = self.arch_attn(deck_card_embs)  # (n_deck_cards, 1)
        attn_weights = torch.softmax(attn_scores, dim=0)  # (n_deck_cards, 1)
        pooled = (attn_weights * deck_card_embs).sum(dim=0)  # (hidden_dim,)
        return pooled, attn_weights, deck_card_embs

    def build_archetype_embeddings(
        self,
        card_emb: torch.Tensor,
        data: HeteroData,
    ) -> torch.Tensor:
        """Build archetype embeddings via attention pooling (inference path).

        No leave-one-out — uses all deck cards.
        """
        n_archetypes = data["archetype"].x.shape[0]
        arch_embeddings = torch.zeros(
            n_archetypes, self.hidden_dim, device=card_emb.device
        )
        arch_card_indices = self._get_arch_card_indices(data)

        for a_idx, card_indices in arch_card_indices.items():
            pooled, _, _ = self._attention_pool(card_emb, card_indices)
            arch_embeddings[a_idx] = pooled

        return arch_embeddings

    def build_loo_arch_embeddings(
        self,
        card_emb: torch.Tensor,
        data: HeteroData,
        a_idx: int,
        exclude_card_indices: list[int],
    ) -> torch.Tensor:
        """Build leave-one-out archetype embeddings for training.

        For each card in exclude_card_indices, compute the archetype embedding
        with that card removed from the attention pool.

        Uses the subtraction trick for efficiency:
          full = sum(w_i * e_i)
          loo_j = (full - w_j * e_j) / (1 - w_j)

        Parameters
        ----------
        card_emb : (n_cards, hidden_dim) all card embeddings
        data : HeteroData graph
        a_idx : archetype index
        exclude_card_indices : card indices to exclude (one per LOO embedding)

        Returns
        -------
        loo_embeddings : (len(exclude_card_indices), hidden_dim)
        """
        arch_card_indices = self._get_arch_card_indices(data)
        deck_cards = arch_card_indices.get(a_idx, [])

        if not deck_cards:
            return torch.zeros(
                len(exclude_card_indices), self.hidden_dim, device=card_emb.device
            )

        # Compute full attention pool
        c_indices = torch.tensor(deck_cards, dtype=torch.long, device=card_emb.device)
        deck_card_embs = card_emb[c_indices]  # (n_deck, H)
        attn_scores = self.arch_attn(deck_card_embs)  # (n_deck, 1)
        attn_weights = torch.softmax(attn_scores, dim=0)  # (n_deck, 1)
        full_emb = (attn_weights * deck_card_embs).sum(dim=0)  # (H,)

        # Map card index → position in deck_cards list
        card_to_pos = {}
        for pos, c_idx in enumerate(deck_cards):
            card_to_pos.setdefault(c_idx, pos)

        loo_embeddings = []
        for exc_card in exclude_card_indices:
            if exc_card in card_to_pos:
                pos = card_to_pos[exc_card]
                w_j = attn_weights[pos]  # (1,)
                e_j = deck_card_embs[pos]  # (H,)
                denom = (1.0 - w_j).clamp(min=1e-6)
                loo = (full_emb - w_j * e_j) / denom
                loo_embeddings.append(loo)
            else:
                # Card not in this archetype's deck — no exclusion needed
                loo_embeddings.append(full_emb)

        return torch.stack(loo_embeddings)  # (n_excluded, H)

    def forward(self, data: HeteroData) -> dict:
        """Forward pass: HGT for cards, attention pooling for archetypes.

        Returns dict with 'node_embeddings': {node_type: tensor}.
        """
        x_dict = self._run_hgt(data)

        # Build archetype embeddings via standard attention pooling
        card_emb = x_dict["card"]
        x_dict["archetype"] = self.build_archetype_embeddings(card_emb, data)

        return {"node_embeddings": x_dict}

    def forward_train(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """Training forward pass: runs HGT only, returns card embeddings.

        Archetype embeddings are built per-pair with LOO in the training loop.
        """
        return self._run_hgt(data)

    def predict_deck(
        self,
        card_embeddings: torch.Tensor,
        arch_embeddings: torch.Tensor,
        card_indices: torch.Tensor,
        arch_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Predict deck inclusion for (card, archetype) pairs.

        Parameters
        ----------
        card_embeddings : (n_cards, hidden_dim)
        arch_embeddings : (n_archetypes, hidden_dim)
        card_indices : (n_pairs,) indices into card_embeddings
        arch_indices : (n_pairs,) indices into arch_embeddings

        Returns
        -------
        logits : (n_pairs,) raw logits — apply sigmoid for probabilities
        """
        card_emb = card_embeddings[card_indices]
        arch_emb = arch_embeddings[arch_indices]
        interaction = card_emb * arch_emb
        combined = torch.cat([card_emb, arch_emb, interaction], dim=-1)
        return self.deck_head(combined).squeeze(-1)

    def score_pairs(
        self,
        card_emb_batch: torch.Tensor,
        arch_emb_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Score (card, archetype) pairs from pre-gathered embeddings.

        Parameters
        ----------
        card_emb_batch : (n_pairs, hidden_dim)
        arch_emb_batch : (n_pairs, hidden_dim)

        Returns
        -------
        logits : (n_pairs,)
        """
        interaction = card_emb_batch * arch_emb_batch
        combined = torch.cat([card_emb_batch, arch_emb_batch, interaction], dim=-1)
        return self.deck_head(combined).squeeze(-1)
