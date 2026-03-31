"""Autoregressive Deck Constructor — GNN + Cross-Attention Deck Builder.

Two-stage architecture for predicting competitive 60-card MTG decklists:

  Stage 1 (HeteroGNN):
    MLP-based message passing over the heterogeneous metagame graph.
    Learns card, archetype, and set embeddings via separate MLPs per
    edge type with additive skip connections back to layer-0 embeddings.
    Messages are scaled by edge weights from the graph before aggregation.

  Stage 2 (DeckConstructor):
    Given an archetype embedding, autoregressively selects cards and
    predicts copy counts via cross-attention over the card pool.
    Builds a deck one card at a time until the 60-card budget is spent.

Training uses Gumbel-softmax straight-through estimation for card and
count selection. Context correction provides partial teacher forcing on
the deck context: when the model selects a wrong card, a random
ground-truth card is substituted into the context to prevent error
snowballing. The loss still penalizes the model's actual wrong pick.
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from src.config import (
    D_COUNT,
    D_MESSAGE,
    D_MODEL,
    DROPOUT,
    MAX_BASIC_LAND_COUNT,
    MAX_DECK_SIZE,
    MAX_NONBASIC_COUNT,
    NUM_ATTN_HEADS,
    NUM_GNN_LAYERS,
)
from src.graph_builder import BASIC_LAND_NAMES


# ════════════════════════════════════════════════════════════════════
#  Stage 1: GNN Message Passing
# ════════════════════════════════════════════════════════════════════


class NodeEncoder(nn.Module):
    """Project raw node features into the shared d_model space.

    One instance per node type (card, archetype, set).
    """

    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MessageMLP(nn.Module):
    """Two-layer MLP for edge-type-specific message transformation.

    Transforms source node embeddings (d_model) into the message space
    (d_message) before aggregation. One instance per (edge_type, GNN_layer).
    """

    def __init__(self, d_model: int, d_message: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_message),
            nn.ReLU(),
            nn.Linear(d_message, d_message),
            nn.LayerNorm(d_message),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpdateMLP(nn.Module):
    """Two-layer MLP that processes aggregated messages for a node type.

    Projects from message space (d_message) back to node space (d_model)
    so the output can be added to layer-0 embeddings via skip connection.
    One instance per (node_type, GNN_layer).
    """

    def __init__(self, d_message: int, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_message, d_message),
            nn.ReLU(),
            nn.Linear(d_message, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HeteroGNNLayer(nn.Module):
    """One round of heterogeneous message passing.

    For each edge type, source embeddings are transformed by a dedicated
    MessageMLP, scaled by edge weights (if present), mean-aggregated at
    the destination, then all edge-type contributions are summed per node
    type and passed through an UpdateMLP.
    """

    def __init__(
        self,
        edge_types: list[tuple[str, str, str]],
        node_types: list[str],
        d_model: int,
        d_message: int,
        dropout: float,
    ):
        super().__init__()
        # One MessageMLP per edge type: d_model → d_message
        self.message_mlps = nn.ModuleDict()
        for src, rel, dst in edge_types:
            key = f"{src}__{rel}__{dst}"
            self.message_mlps[key] = MessageMLP(d_model, d_message, dropout)

        # One UpdateMLP per node type: d_message → d_model
        self.update_mlps = nn.ModuleDict()
        for nt in node_types:
            self.update_mlps[nt] = UpdateMLP(d_message, d_model, dropout)

        self._edge_types = edge_types
        self._node_types = node_types
        self._d_message = d_message

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
        edge_weight_dict: dict[tuple, torch.Tensor | None],
    ) -> dict[str, torch.Tensor]:
        """Compute per-node-type update vectors (before skip connection).

        Returns a dict of {node_type: update_tensor} to be added to the
        layer-0 embeddings by the caller.
        """
        # Accumulate messages per destination node type (in d_message space)
        agg = {
            nt: torch.zeros(x_dict[nt].shape[0], self._d_message, device=x_dict[nt].device)
            for nt in self._node_types
        }

        for src_type, rel, dst_type in self._edge_types:
            key = f"{src_type}__{rel}__{dst_type}"
            et = (src_type, rel, dst_type)
            if et not in edge_index_dict:
                continue

            edge_index = edge_index_dict[et]
            src_idx, dst_idx = edge_index[0], edge_index[1]

            # Transform source embeddings with edge-type-specific MLP
            src_emb = x_dict[src_type][src_idx]           # (n_edges, d_model)
            messages = self.message_mlps[key](src_emb)     # (n_edges, d_message)

            # Scale messages by edge weight (if available)
            weights = edge_weight_dict.get(et)
            if weights is not None:
                messages = messages * weights.unsqueeze(1)  # (n_edges, 1) broadcast

            # Mean aggregation at destination nodes
            n_dst = x_dict[dst_type].shape[0]
            # Sum messages per destination node
            summed = torch.zeros(n_dst, self._d_message, device=messages.device)
            summed.scatter_add_(0, dst_idx.unsqueeze(1).expand_as(messages), messages)
            # Count incoming edges per destination for mean
            counts = torch.zeros(n_dst, 1, device=messages.device)
            counts.scatter_add_(0, dst_idx.unsqueeze(1), torch.ones_like(dst_idx.unsqueeze(1).float()))
            counts = counts.clamp(min=1.0)
            mean_msg = summed / counts

            # Accumulate across edge types targeting the same node type
            agg[dst_type] = agg[dst_type] + mean_msg

        # Apply UpdateMLP to aggregated messages
        updates = {}
        for nt in self._node_types:
            updates[nt] = self.update_mlps[nt](agg[nt])

        return updates


class HeteroGNN(nn.Module):
    """Heterogeneous GNN with MLP message passing and layer-0 skip connections.

    Stacks L HeteroGNNLayers. After each layer, the update is added to the
    original (layer-0) projected embeddings — NOT the previous layer output.
    This preserves node identity across message-passing rounds.
    """

    def __init__(
        self,
        node_dims: dict[str, int],
        edge_types: list[tuple[str, str, str]],
        node_types: list[str],
        d_model: int = D_MODEL,
        d_message: int = D_MESSAGE,
        num_layers: int = NUM_GNN_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        # Per-node-type input projection
        self.encoders = nn.ModuleDict()
        for nt in node_types:
            self.encoders[nt] = NodeEncoder(node_dims[nt], d_model)

        # Message-passing layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(HeteroGNNLayer(edge_types, node_types, d_model, d_message, dropout))

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """Run GNN and return final embeddings for all node types."""
        # Layer-0 embeddings (the anchor for skip connections)
        x_dict_0 = {}
        for nt in data.node_types:
            if nt in self.encoders:
                x_dict_0[nt] = self.encoders[nt](data[nt].x)

        # Collect edge indices and weights
        edge_index_dict = {}
        edge_weight_dict = {}
        for et in data.edge_types:
            edge_index_dict[et] = data[et].edge_index
            edge_weight_dict[et] = getattr(data[et], "edge_weight", None)

        # Message passing with additive skip to layer 0
        x_dict = {nt: x_dict_0[nt].clone() for nt in x_dict_0}
        for layer in self.layers:
            updates = layer(x_dict, edge_index_dict, edge_weight_dict)
            for nt in x_dict:
                if nt in updates:
                    x_dict[nt] = x_dict_0[nt] + updates[nt]

        return x_dict


# ════════════════════════════════════════════════════════════════════
#  Stage 2: Autoregressive Deck Construction
# ════════════════════════════════════════════════════════════════════


class DeckContextEncoder(nn.Module):
    """Encode the growing deck context into a single query vector.

    Applies multi-head self-attention over the context (archetype embedding
    + previously selected cards), then mean-pools to produce q_t.
    """

    def __init__(self, d_model: int = D_MODEL, num_heads: int = NUM_ATTN_HEADS, dropout: float = DROPOUT):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Produce query vector q_t from deck context.

        Args:
            context: (seq_len, d_model) — archetype + selected cards so far.

        Returns:
            q_t: (d_model,) — mean-pooled context query.
        """
        # Add batch dim for nn.MultiheadAttention
        ctx = context.unsqueeze(0)                   # (1, seq_len, d_model)
        attn_out, _ = self.self_attn(ctx, ctx, ctx)  # (1, seq_len, d_model)
        ctx = self.norm(ctx + attn_out)              # residual + layernorm
        q_t = ctx.squeeze(0).mean(dim=0)             # mean pool → (d_model,)
        return q_t


class CardSelector(nn.Module):
    """Score candidate cards via multi-head cross-attention.

    Projects the context query q_t and card embeddings into key/query
    subspaces, then computes scaled dot-product scores. Returns raw
    logits (pre-softmax) for compatibility with Gumbel-softmax.
    """

    def __init__(self, d_model: int = D_MODEL, num_heads: int = NUM_ATTN_HEADS):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(num_heads, 1, bias=False)  # combine heads → scalar per card

    def forward(
        self,
        q_t: torch.Tensor,
        card_embeddings: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute selection logits over all cards.

        Args:
            q_t: (d_model,) — context query.
            card_embeddings: (n_cards, d_model) — all card embeddings.
            mask: (n_cards,) — True for eligible cards, False otherwise.

        Returns:
            logits: (n_cards,) — raw selection scores (masked ineligible = -inf).
        """
        n_cards = card_embeddings.shape[0]

        # Project query and keys into multi-head subspaces
        Q = self.W_Q(q_t)                                     # (d_model,)
        K = self.W_K(card_embeddings)                          # (n_cards, d_model)

        # Reshape for multi-head: (num_heads, d_k) and (n_cards, num_heads, d_k)
        Q = Q.view(self.num_heads, self.d_k)                   # (H, d_k)
        K = K.view(n_cards, self.num_heads, self.d_k)          # (N, H, d_k)

        # Scaled dot-product per head: (N, H)
        scores = (K * Q.unsqueeze(0)).sum(dim=-1) / math.sqrt(self.d_k)

        # Combine heads via learned projection → (N, 1) → (N,)
        logits = self.W_O(scores).squeeze(-1)

        # Mask ineligible cards
        logits = logits.masked_fill(~mask, float("-inf"))
        return logits


class CountPredictor(nn.Module):
    """Predict copy count (1–20) for a selected card.

    Takes the context query, selected card embedding, and their element-wise
    product as input. The Hadamard product enables multiplicative feature
    interactions between deck context and card identity.
    """

    def __init__(self, d_model: int = D_MODEL, max_count: int = MAX_BASIC_LAND_COUNT):
        super().__init__()
        self.max_count = max_count
        self.mlp = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.head = nn.Linear(d_model, max_count)

    def forward(
        self,
        q_t: torch.Tensor,
        card_emb: torch.Tensor,
        is_basic_land: bool,
        remaining_budget: int,
    ) -> torch.Tensor:
        """Produce masked count logits.

        Args:
            q_t: (d_model,) — context query.
            card_emb: (d_model,) — selected card embedding.
            is_basic_land: whether the selected card is a basic land.
            remaining_budget: cards remaining in the 60-card budget.

        Returns:
            logits: (max_count,) — count logits with invalid positions set to -inf.
                    Position i corresponds to count (i+1).
        """
        # Concatenate [q_t, card_emb, q_t * card_emb]
        h = torch.cat([q_t, card_emb, q_t * card_emb])
        h = self.mlp(h)
        logits = self.head(h)  # (max_count,)

        # Build count validity mask
        max_allowed = MAX_BASIC_LAND_COUNT if is_basic_land else MAX_NONBASIC_COUNT
        max_allowed = min(max_allowed, remaining_budget)

        mask = torch.zeros(self.max_count, dtype=torch.bool, device=logits.device)
        mask[:max_allowed] = True  # positions 0..max_allowed-1 → counts 1..max_allowed

        logits = logits.masked_fill(~mask, float("-inf"))
        return logits


class CountEmbedding(nn.Module):
    """Learned embedding table for copy counts 1–20."""

    def __init__(self, max_count: int = MAX_BASIC_LAND_COUNT, d_count: int = D_COUNT):
        super().__init__()
        self.embedding = nn.Embedding(max_count, d_count)

    def forward(self, count_idx: int) -> torch.Tensor:
        """Look up embedding for a count index (0-based: idx 0 = count 1)."""
        idx = torch.tensor(count_idx, device=self.embedding.weight.device)
        return self.embedding(idx)


class ContextMerge(nn.Module):
    """Fuse a card embedding with its count embedding for context update.

    Produces a d_model-dimensional vector that gets appended to the
    growing deck context D_t.
    """

    def __init__(self, d_model: int = D_MODEL, d_count: int = D_COUNT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model + d_count, d_model),
            nn.ReLU(),
        )

    def forward(self, card_emb: torch.Tensor, count_emb: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([card_emb, count_emb]))


def gumbel_softmax_sample(logits: torch.Tensor, tau: float, hard: bool = True) -> torch.Tensor:
    """Gumbel-softmax with straight-through estimation.

    During training (hard=True): forward pass uses argmax (hard one-hot),
    backward pass uses the continuous relaxation for gradient flow.

    Args:
        logits: raw scores (may contain -inf for masked positions).
        tau: temperature — higher = more uniform, lower = more peaked.
        hard: if True, use straight-through (hard forward, soft backward).

    Returns:
        one_hot: (len(logits),) — one-hot vector (hard) or soft probabilities.
    """
    return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)


class DeckConstructor(nn.Module):
    """Autoregressive deck builder with context correction.

    Given an archetype embedding and the full card pool, iteratively:
      1. Self-attend over the current deck context
      2. Cross-attend to score all eligible cards
      3. Select one card (Gumbel-softmax in training, argmax in inference)
      4. Predict copy count for the selected card
      5. Context correction (training only): if the selected card is not
         in the target deck, substitute a random remaining ground-truth
         card into the context to prevent error snowballing
      6. Append the count-conditioned card to the context
      7. Repeat until the 60-card budget is exhausted
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        d_count: int = D_COUNT,
        num_heads: int = NUM_ATTN_HEADS,
        dropout: float = DROPOUT,
        max_count: int = MAX_BASIC_LAND_COUNT,
    ):
        super().__init__()
        self.context_encoder = DeckContextEncoder(d_model, num_heads, dropout)
        self.card_selector = CardSelector(d_model, num_heads)
        self.count_predictor = CountPredictor(d_model, max_count)
        self.count_embedding = CountEmbedding(max_count, d_count)
        self.context_merge = ContextMerge(d_model, d_count)
        self.d_model = d_model

    def forward(
        self,
        arch_embedding: torch.Tensor,
        card_embeddings: torch.Tensor,
        basic_land_mask: torch.Tensor,
        tau: float = 1.0,
        greedy: bool = False,
        target_deck: dict[int, int] | None = None,
    ) -> dict:
        """Build a deck autoregressively.

        Args:
            arch_embedding: (d_model,) — archetype embedding from GNN.
            card_embeddings: (n_cards, d_model) — all card embeddings from GNN.
            basic_land_mask: (n_cards,) bool — True for basic land cards.
            tau: Gumbel-softmax temperature (ignored if greedy=True).
            greedy: if True, use argmax instead of Gumbel-softmax.
            target_deck: {card_idx: count} ground truth for context correction
                during training. When the model picks a wrong card, a random
                remaining target card is substituted into the context so that
                subsequent steps condition on a valid deck state. The loss
                still penalizes the wrong pick. None disables correction.

        Returns:
            dict with:
              - deck: {card_idx: count} — the predicted decklist
              - steps: list of per-step info dicts containing logits and probs
                       for loss computation during training
        """
        n_cards = card_embeddings.shape[0]
        device = card_embeddings.device

        # Each card is selected at most once; the count prediction determines copies
        selected = torch.zeros(n_cards, dtype=torch.bool, device=device)
        budget = MAX_DECK_SIZE
        deck: dict[int, int] = {}
        steps: list[dict] = []

        # Track which target cards have been consumed (for context correction)
        target_remaining = set(target_deck.keys()) if target_deck else set()

        # Seed context with archetype embedding
        context = arch_embedding.unsqueeze(0)  # (1, d_model)

        while budget > 0:
            # 1. Encode context → query vector
            q_t = self.context_encoder(context)

            # 2. Build eligibility mask: only cards not yet selected
            eligible = ~selected  # (n_cards,) bool

            if not eligible.any():
                break  # no more eligible cards (shouldn't happen in practice)

            # 3. Score all cards via cross-attention
            select_logits = self.card_selector(q_t, card_embeddings, eligible)

            # 4. Select a card
            if greedy:
                card_idx = select_logits.argmax().item()
                card_emb = card_embeddings[card_idx]
                select_probs = F.softmax(select_logits, dim=-1)
            else:
                # Gumbel-softmax: hard forward, soft backward
                select_one_hot = gumbel_softmax_sample(select_logits, tau, hard=True)
                card_idx = select_one_hot.argmax().item()
                # Differentiable card embedding (soft blend in backward pass)
                card_emb = select_one_hot @ card_embeddings
                select_probs = F.softmax(select_logits, dim=-1)

            # 5. Predict copy count
            is_basic = basic_land_mask[card_idx].item()
            count_logits = self.count_predictor(q_t, card_emb, is_basic, budget)

            if greedy:
                count_idx = count_logits.argmax().item()
                count_probs = F.softmax(count_logits, dim=-1)
            else:
                count_one_hot = gumbel_softmax_sample(count_logits, tau, hard=True)
                count_idx = count_one_hot.argmax().item()
                count_probs = F.softmax(count_logits, dim=-1)

            count = count_idx + 1  # positions are 0-indexed, counts are 1-indexed
            count = min(count, budget)  # safety clamp

            # 6. Context correction: if the model picked wrong during training,
            #    substitute a random remaining target card into the context
            sub_idx = None
            use_model_pick = True
            if target_deck is not None and not greedy and card_idx not in target_deck:
                # Find eligible target cards (in GT and not yet selected)
                eligible_targets = [
                    c for c in target_remaining if not selected[c].item()
                ]
                if eligible_targets:
                    use_model_pick = False
                    sub_idx = random.choice(eligible_targets)
                    sub_count = target_deck[sub_idx]
                    sub_count = min(sub_count, budget)
                    target_remaining.discard(sub_idx)

                    # Update deck state with substituted card
                    deck[sub_idx] = sub_count
                    selected[sub_idx] = True
                    budget -= sub_count

                    # Build context from substituted card (no Gumbel gradient)
                    sub_card_emb = card_embeddings[sub_idx]
                    sub_count_idx = sub_count - 1
                    sub_count_emb = self.count_embedding(sub_count_idx)
                    merged = self.context_merge(sub_card_emb, sub_count_emb)
                    context = torch.cat([context, merged.unsqueeze(0)], dim=0)

            # 7. Record step info for loss computation (always the model's actual pick).
            #    Include substituted_card_idx so the loss function can remove the
            #    substituted card from its own target tracking.
            steps.append({
                "card_idx": card_idx,
                "count": count,
                "select_logits": select_logits,
                "select_probs": select_probs,
                "count_logits": count_logits,
                "count_probs": count_probs,
                "substituted_card_idx": sub_idx,
            })

            if use_model_pick:
                # Correct pick (or greedy / no target) — use model's selection
                if card_idx in target_remaining:
                    target_remaining.discard(card_idx)

                deck[card_idx] = count
                selected[card_idx] = True
                budget -= count

                # Update context with count-conditioned card representation
                if greedy:
                    count_emb = self.count_embedding(count_idx)
                else:
                    # Differentiable count embedding (soft blend in backward pass)
                    count_emb = count_one_hot @ self.count_embedding.embedding.weight

                merged = self.context_merge(card_emb, count_emb)
                context = torch.cat([context, merged.unsqueeze(0)], dim=0)

        return {"deck": deck, "steps": steps}


# ════════════════════════════════════════════════════════════════════
#  Top-Level Model
# ════════════════════════════════════════════════════════════════════


class MTGDeckModel(nn.Module):
    """Complete model: HeteroGNN + Autoregressive DeckConstructor.

    Stage 1 learns card/archetype/set embeddings from the metagame graph.
    Stage 2 uses those embeddings to build decks card-by-card.
    """

    def __init__(
        self,
        node_dims: dict[str, int],
        edge_types: list[tuple[str, str, str]],
        node_types: list[str],
        card_names: list[str],
        d_model: int = D_MODEL,
        d_message: int = D_MESSAGE,
        d_count: int = D_COUNT,
        num_gnn_layers: int = NUM_GNN_LAYERS,
        num_attn_heads: int = NUM_ATTN_HEADS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.gnn = HeteroGNN(
            node_dims=node_dims,
            edge_types=edge_types,
            node_types=node_types,
            d_model=d_model,
            d_message=d_message,
            num_layers=num_gnn_layers,
            dropout=dropout,
        )
        self.decoder = DeckConstructor(
            d_model=d_model,
            d_count=d_count,
            num_heads=num_attn_heads,
            dropout=dropout,
        )

        # Precompute which cards are basic lands
        self.register_buffer(
            "basic_land_mask",
            torch.tensor([name in BASIC_LAND_NAMES for name in card_names]),
        )

    def forward(
        self,
        data: HeteroData,
        archetype_idx: int,
        tau: float = 1.0,
        greedy: bool = False,
        target_deck: dict[int, int] | None = None,
    ) -> dict:
        """Run full model: GNN embedding + autoregressive deck construction.

        Args:
            data: the HeteroData metagame graph.
            archetype_idx: which archetype to build a deck for.
            tau: Gumbel temperature for training.
            greedy: if True, use argmax (inference mode).
            target_deck: {card_idx: count} ground truth for context correction
                during training (see DeckConstructor.forward).

        Returns:
            dict with 'deck' ({card_idx: count}) and 'steps' (per-step info).
        """
        x_dict = self.gnn(data)
        arch_emb = x_dict["archetype"][archetype_idx]
        card_emb = x_dict["card"]

        return self.decoder(
            arch_emb, card_emb, self.basic_land_mask, tau, greedy, target_deck,
        )

    def forward_with_embeddings(
        self,
        x_dict: dict[str, torch.Tensor],
        archetype_idx: int,
        tau: float = 1.0,
        greedy: bool = False,
        target_deck: dict[int, int] | None = None,
    ) -> dict:
        """Run deck construction using pre-computed GNN embeddings.

        Use this when the GNN output has been cached (e.g., to avoid
        redundant GNN passes across multiple archetypes in the same epoch).
        Gradients still flow through x_dict back to the GNN.

        Args:
            x_dict: pre-computed {node_type: embeddings} from self.gnn(data).
            archetype_idx: which archetype to build a deck for.
            tau: Gumbel temperature for training.
            greedy: if True, use argmax (inference mode).
            target_deck: {card_idx: count} ground truth for context correction.

        Returns:
            dict with 'deck' ({card_idx: count}) and 'steps' (per-step info).
        """
        arch_emb = x_dict["archetype"][archetype_idx]
        card_emb = x_dict["card"]

        return self.decoder(
            arch_emb, card_emb, self.basic_land_mask, tau, greedy, target_deck,
        )

    @torch.no_grad()
    def predict_deck(
        self,
        data: HeteroData,
        archetype_idx: int,
        x_dict: dict[str, torch.Tensor] | None = None,
    ) -> dict[int, int]:
        """Convenience method for greedy inference.

        Args:
            data: the HeteroData graph (used if x_dict is None).
            archetype_idx: which archetype to build a deck for.
            x_dict: optional pre-computed GNN embeddings. If provided,
                skips the GNN forward pass.

        Returns:
            {card_idx: count} — a valid 60-card decklist.
        """
        self.eval()
        if x_dict is not None:
            result = self.forward_with_embeddings(
                x_dict, archetype_idx, greedy=True,
            )
        else:
            result = self.forward(data, archetype_idx, greedy=True)
        return result["deck"]
