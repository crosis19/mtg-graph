"""Autoregressive Deck Constructor — GNN + MLP Deck Builder.

Two-stage architecture for predicting competitive 60-card MTG decklists:

  Stage 1 (HeteroGNN):
    MLP-based message passing over the heterogeneous metagame graph.
    Learns card, archetype, and set embeddings via separate MLPs per
    edge type with additive skip connections back to layer-0 embeddings.
    Edge weights from the graph (copy counts, synergy scores, win rates)
    are treated as edge features concatenated with source embeddings
    before the MessageMLP. Each edge type has a learnable scalar weight
    that controls its overall contribution to message aggregation.

  Stage 2 (DeckConstructor):
    Given an archetype embedding, autoregressively selects cards and
    predicts copy counts via MLP scoring over the card pool. At each step,
    the archetype embedding is concatenated with the mean of previously
    selected card+count embeddings and projected via MLP to produce a
    fixed-size query vector. Separate count heads for non-basic cards
    (1–4) and basic lands (1–20). Builds a deck one card at a time
    until the 60-card budget is spent.

Training uses argmax card selection with several mechanisms:
  1. Scheduled sampling: anneals teacher forcing rate so the model
     progressively sees its own errors during context correction.
  2. Deterministic card ordering: context correction substitutes the
     highest-consensus remaining GT card (not random).
  3. Count teacher forcing: when the model selects a correct card, the
     ground-truth count is used for the deck state and context update
     (preventing budget drift), while the loss trains on the model's
     predicted count.
  4. Budget signal: injects remaining budget, color density, and type
     density as explicit features into the context encoder.
  5. Temperature annealing: softmax temperature on card scorer logits
     controls gradient distribution during training.
  6. Consensus-weighted loss: penalizes missing staples more heavily
     than flex slots based on card inclusion rates.
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from torch.utils.checkpoint import checkpoint as torch_checkpoint

from src.config import (
    D_COUNT,
    D_MESSAGE,
    D_MODEL,
    DROPOUT,
    EDGE_CHUNK_SIZE,
    GRADIENT_CHECKPOINTING,
    MAX_BASIC_LAND_COUNT,
    MAX_DECK_SIZE,
    MAX_NONBASIC_COUNT,
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

    Transforms source node embeddings concatenated with edge features
    (d_input = d_model + n_edge_features) into the message space (d_message)
    before aggregation. One instance per (edge_type, GNN_layer).
    """

    def __init__(self, d_input: int, d_message: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, d_message),
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

    For each edge type, source embeddings are concatenated with edge
    features (the graph's edge weights treated as input features) and
    transformed by a dedicated MessageMLP. Messages are then scaled by
    a learnable per-edge-type scalar weight, mean-aggregated at the
    destination, and all edge-type contributions are summed per node
    type and passed through an UpdateMLP.
    """

    def __init__(
        self,
        edge_types: list[tuple[str, str, str]],
        node_types: list[str],
        d_model: int,
        d_message: int,
        dropout: float,
        edge_chunk_size: int = 0,
    ):
        super().__init__()
        # One MessageMLP per edge type: (d_model + 1 edge feature) → d_message
        self.message_mlps = nn.ModuleDict()
        for src, rel, dst in edge_types:
            key = f"{src}__{rel}__{dst}"
            self.message_mlps[key] = MessageMLP(d_model + 1, d_message, dropout)

        # Learnable per-edge-type scalar weight (initialized to 1.0)
        self.edge_type_weights = nn.ParameterDict()
        for src, rel, dst in edge_types:
            key = f"{src}__{rel}__{dst}"
            self.edge_type_weights[key] = nn.Parameter(torch.ones(1))

        # One UpdateMLP per node type: d_message → d_model
        self.update_mlps = nn.ModuleDict()
        for nt in node_types:
            self.update_mlps[nt] = UpdateMLP(d_message, d_model, dropout)

        self._edge_types = edge_types
        self._node_types = node_types
        self._d_message = d_message
        self._edge_chunk_size = edge_chunk_size

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
        edge_weight_dict: dict[tuple, torch.Tensor | None],
    ) -> dict[str, torch.Tensor]:
        """Compute per-node-type update vectors (before skip connection).

        Edge weights from edge_weight_dict are used as input features
        (concatenated with source embeddings before the MessageMLP), not
        as message scaling factors. Learned per-edge-type scalars control
        the contribution of each relationship type.

        When ``edge_chunk_size > 0``, large edge types are processed in
        mini-batches to cap GPU memory.  The scatter-add aggregation is
        fully decomposable: ``sum(partial) / count(partial) = global mean``.

        Returns a dict of {node_type: update_tensor} to be added to the
        layer-0 embeddings by the caller.
        """
        # Accumulate messages per destination node type (in d_message space)
        agg = {
            nt: torch.zeros(x_dict[nt].shape[0], self._d_message, device=x_dict[nt].device)
            for nt in self._node_types
        }

        chunk = self._edge_chunk_size

        for src_type, rel, dst_type in self._edge_types:
            key = f"{src_type}__{rel}__{dst_type}"
            et = (src_type, rel, dst_type)
            if et not in edge_index_dict:
                continue

            edge_index = edge_index_dict[et]
            n_edges = edge_index.shape[1]
            n_dst = x_dict[dst_type].shape[0]
            device = x_dict[src_type].device

            # Move edge data to compute device if needed (CPU offloading)
            if edge_index.device != device:
                edge_index = edge_index.to(device, non_blocking=True)
            src_idx, dst_idx = edge_index[0], edge_index[1]

            ew_raw = edge_weight_dict.get(et)
            if ew_raw is not None and ew_raw.device != device:
                ew_raw = ew_raw.to(device, non_blocking=True)

            # ── Chunked path for large edge types ──
            if chunk > 0 and n_edges > chunk:
                total_summed = torch.zeros(n_dst, self._d_message, device=device)
                total_counts = torch.zeros(n_dst, 1, device=device)

                for start in range(0, n_edges, chunk):
                    end = min(start + chunk, n_edges)
                    si_c = src_idx[start:end]
                    di_c = dst_idx[start:end]

                    src_emb_c = x_dict[src_type][si_c]
                    if ew_raw is not None:
                        ef_c = ew_raw[start:end].unsqueeze(1)
                    else:
                        ef_c = torch.ones(end - start, 1, device=device)

                    mlp_in_c = torch.cat([src_emb_c, ef_c], dim=1)
                    msg_c = self.message_mlps[key](mlp_in_c)
                    msg_c = msg_c * self.edge_type_weights[key]

                    total_summed.scatter_add_(
                        0, di_c.unsqueeze(1).expand_as(msg_c), msg_c,
                    )
                    total_counts.scatter_add_(
                        0, di_c.unsqueeze(1),
                        torch.ones_like(di_c.unsqueeze(1).float()),
                    )
                    del src_emb_c, ef_c, mlp_in_c, msg_c, si_c, di_c

                mean_msg = total_summed / total_counts.clamp(min=1.0)
                del total_summed, total_counts

            # ── Standard path for small edge types ──
            else:
                src_emb = x_dict[src_type][src_idx]
                if ew_raw is not None:
                    edge_feat = ew_raw.unsqueeze(1)
                else:
                    edge_feat = torch.ones(n_edges, 1, device=device)

                mlp_input = torch.cat([src_emb, edge_feat], dim=1)
                messages = self.message_mlps[key](mlp_input)
                messages = messages * self.edge_type_weights[key]
                del src_emb, edge_feat, mlp_input

                summed = torch.zeros(n_dst, self._d_message, device=device)
                summed.scatter_add_(
                    0, dst_idx.unsqueeze(1).expand_as(messages), messages,
                )
                counts = torch.zeros(n_dst, 1, device=device)
                counts.scatter_add_(
                    0, dst_idx.unsqueeze(1),
                    torch.ones_like(dst_idx.unsqueeze(1).float()),
                )
                mean_msg = summed / counts.clamp(min=1.0)
                del messages, summed, counts

            agg[dst_type] = agg[dst_type] + mean_msg
            del mean_msg

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
        gradient_checkpointing: bool = GRADIENT_CHECKPOINTING,
        edge_chunk_size: int = EDGE_CHUNK_SIZE,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing

        # Per-node-type input projection
        self.encoders = nn.ModuleDict()
        for nt in node_types:
            self.encoders[nt] = NodeEncoder(node_dims[nt], d_model)

        # Message-passing layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(HeteroGNNLayer(
                edge_types, node_types, d_model, d_message, dropout,
                edge_chunk_size=edge_chunk_size,
            ))

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """Run GNN and return final embeddings for all node types.

        When gradient_checkpointing is enabled, intermediate activations in
        the message-passing layers are NOT stored during forward; they are
        recomputed during backward. This cuts peak memory ~60% at ~25% more
        backward compute time.
        """
        # Layer-0 embeddings (the anchor for skip connections)
        x_dict_0 = {}
        for nt in data.node_types:
            if nt in self.encoders:
                x_dict_0[nt] = self.encoders[nt](data[nt].x)

        # Collect edge indices and weights — keep on CPU to save GPU memory.
        # HeteroGNNLayer.forward() moves each edge type to GPU on demand.
        edge_index_dict = {}
        edge_weight_dict = {}
        for et in data.edge_types:
            ei = data[et].edge_index
            ew = getattr(data[et], "edge_weight", None)
            edge_index_dict[et] = ei.cpu() if ei.is_cuda else ei
            edge_weight_dict[et] = ew.cpu() if (ew is not None and ew.is_cuda) else ew

        # Message passing with additive skip to layer 0
        x_dict = {nt: x_dict_0[nt].clone() for nt in x_dict_0}
        use_ckpt = self.gradient_checkpointing and torch.is_grad_enabled()

        for layer in self.layers:
            if use_ckpt:
                updates = torch_checkpoint(
                    layer, x_dict, edge_index_dict, edge_weight_dict,
                    use_reentrant=False,
                )
            else:
                updates = layer(x_dict, edge_index_dict, edge_weight_dict)
            for nt in x_dict:
                if nt in updates:
                    x_dict[nt] = x_dict_0[nt] + updates[nt]

        return x_dict


# ════════════════════════════════════════════════════════════════════
#  Stage 2: Autoregressive Deck Construction
# ════════════════════════════════════════════════════════════════════


class DeckContextEncoder(nn.Module):
    """Encode archetype + selected-card context into a query vector.

    Mean-pools the selected card embeddings and concatenates with the
    archetype embedding (and optional budget features), then projects
    via a 2-layer MLP to produce q_t.
    Fixed-size input every step — the archetype signal is never diluted.
    """

    def __init__(self, d_model: int = D_MODEL, dropout: float = DROPOUT,
                 n_budget_features: int = 0):
        super().__init__()
        input_dim = 2 * d_model + n_budget_features
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.d_model = d_model

    def forward(self, arch_emb: torch.Tensor, card_embs: torch.Tensor | None,
                budget_features: torch.Tensor | None = None) -> torch.Tensor:
        """Produce query vector q_t from archetype + selected card context.

        Args:
            arch_emb: (d_model,) — archetype embedding from GNN.
            card_embs: (num_selected, d_model) — merged card+count embeddings
                       selected so far, or None if no cards selected yet.
            budget_features: optional (n_budget_features,) — partial deck
                state features (remaining budget, color/type densities).

        Returns:
            q_t: (d_model,) — context query for card scoring.
        """
        if card_embs is not None and card_embs.shape[0] > 0:
            card_mean = card_embs.mean(dim=0)
        else:
            card_mean = torch.zeros(self.d_model, device=arch_emb.device)
        parts = [arch_emb, card_mean]
        if budget_features is not None:
            parts.append(budget_features)
        return self.proj(torch.cat(parts))


class MLPCardScorer(nn.Module):
    """Score candidate cards via MLP on context–card feature interactions.

    For each card, concatenates [q_t, card_emb, q_t * card_emb] and
    passes through a shared MLP to produce a scalar selection logit.
    All cards are scored in a single batched forward pass.
    """

    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )
        self.head = nn.Linear(d_model, 1)

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
        # Broadcast q_t to match card count
        q_expanded = q_t.unsqueeze(0).expand_as(card_embeddings)  # (N, d_model)

        # Concatenate context, card, and element-wise interaction
        x = torch.cat([q_expanded, card_embeddings, q_expanded * card_embeddings], dim=-1)

        # MLP → scalar logit per card
        logits = self.head(self.mlp(x)).squeeze(-1)  # (N,)

        # Mask ineligible cards
        logits = logits.masked_fill(~mask, float("-inf"))
        return logits


class NonBasicCountHead(nn.Module):
    """Predict copy count (1–4) for a selected non-basic card."""

    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.max_count = MAX_NONBASIC_COUNT
        self.mlp = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.head = nn.Linear(d_model, self.max_count)

    def forward(
        self,
        q_t: torch.Tensor,
        card_emb: torch.Tensor,
        remaining_budget: int,
    ) -> torch.Tensor:
        """Produce masked count logits for non-basic cards.

        Args:
            q_t: (d_model,) — context query.
            card_emb: (d_model,) — selected card embedding.
            remaining_budget: cards remaining in the 60-card budget.

        Returns:
            logits: (4,) — count logits with invalid positions set to -inf.
                    Position i corresponds to count (i+1).
        """
        h = torch.cat([q_t, card_emb, q_t * card_emb])
        h = self.mlp(h)
        logits = self.head(h)  # (4,)

        max_allowed = min(self.max_count, remaining_budget)
        mask = torch.zeros(self.max_count, dtype=torch.bool, device=logits.device)
        mask[:max_allowed] = True
        logits = logits.masked_fill(~mask, float("-inf"))
        return logits


class BasicLandCountHead(nn.Module):
    """Predict copy count (1–20) for a selected basic land."""

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
        remaining_budget: int,
    ) -> torch.Tensor:
        """Produce masked count logits for basic lands.

        Args:
            q_t: (d_model,) — context query.
            card_emb: (d_model,) — selected card embedding.
            remaining_budget: cards remaining in the 60-card budget.

        Returns:
            logits: (max_count,) — count logits with invalid positions set to -inf.
                    Position i corresponds to count (i+1).
        """
        h = torch.cat([q_t, card_emb, q_t * card_emb])
        h = self.mlp(h)
        logits = self.head(h)  # (max_count,)

        max_allowed = min(self.max_count, remaining_budget)
        mask = torch.zeros(self.max_count, dtype=torch.bool, device=logits.device)
        mask[:max_allowed] = True
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


class DeckConstructor(nn.Module):
    """Autoregressive deck builder with scheduled sampling, consensus-
    ordered context correction, count teacher forcing, budget signal,
    and temperature-annealed scoring.

    Given an archetype embedding and the full card pool, iteratively:
      1. Build budget feature vector (remaining cards, color/type density)
      2. Encode archetype + card context + budget features → query vector
      3. Score all eligible cards via MLP (with temperature scaling)
      4. Select one card via argmax
      5. Predict copy count for the selected card
      6. Scheduled sampling + context correction (training only):
         - Wrong card + coin says teacher-force: substitute the highest-
           priority remaining GT card (deterministic order) into context
         - Wrong card + coin says use model: keep the model's pick
         - Correct card: use GT count for deck state (count teacher forcing)
      7. Append the count-conditioned card to the context
      8. Repeat until the 60-card budget is exhausted
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        d_count: int = D_COUNT,
        dropout: float = DROPOUT,
        max_count: int = MAX_BASIC_LAND_COUNT,
        card_color_flags: torch.Tensor | None = None,
        card_type_flags: torch.Tensor | None = None,
    ):
        super().__init__()
        n_budget = 14 if card_color_flags is not None else 0
        self.context_encoder = DeckContextEncoder(d_model, dropout, n_budget_features=n_budget)
        self.card_selector = MLPCardScorer(d_model)
        self.nonbasic_count_head = NonBasicCountHead(d_model)
        self.basic_land_count_head = BasicLandCountHead(d_model, max_count)
        self.count_embedding = CountEmbedding(max_count, d_count)
        self.context_merge = ContextMerge(d_model, d_count)
        self.d_model = d_model
        self._has_budget_signal = card_color_flags is not None

        if card_color_flags is not None:
            self.register_buffer("card_color_flags", card_color_flags)
            self.register_buffer("card_type_flags", card_type_flags)

    def forward(
        self,
        arch_embedding: torch.Tensor,
        card_embeddings: torch.Tensor,
        basic_land_mask: torch.Tensor,
        greedy: bool = False,
        target_deck: dict[int, int] | None = None,
        curriculum_mask: torch.Tensor | None = None,
        target_order: list[int] | None = None,
        teacher_forcing_rate: float = 1.0,
        temperature: float = 1.0,
    ) -> dict:
        """Build a deck autoregressively.

        Args:
            arch_embedding: (d_model,) — archetype embedding from GNN.
            card_embeddings: (n_cards, d_model) — all card embeddings from GNN.
            basic_land_mask: (n_cards,) bool — True for basic land cards.
            greedy: if True, use argmax without teacher forcing (inference).
            target_deck: {card_idx: count} ground truth for teacher forcing
                during training. Context correction substitutes GT cards on
                wrong picks; count teacher forcing uses GT counts on correct
                picks to prevent budget drift. None disables both.
            curriculum_mask: (n_cards,) bool — True for cards eligible in this
                epoch's curriculum. None means all cards eligible. Only used
                during training to restrict the decoder's candidate pool.
            target_order: deterministic ordering of target cards (highest
                consensus first). Used for context correction substitution.
            teacher_forcing_rate: probability of substituting a GT card on
                wrong picks (scheduled sampling). 1.0 = always substitute.
            temperature: softmax temperature for card selection scores.
                Applied during training only (logits / T before softmax).

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

        # Track selected card embeddings for context encoding
        card_context: list[torch.Tensor] = []

        # Budget signal accumulators
        color_counts = torch.zeros(5, device=device)
        type_counts = torch.zeros(7, device=device)
        cards_placed = 0

        while budget > 0:
            # 1. Build budget features if enabled
            budget_features = None
            if self._has_budget_signal:
                placed_f = max(cards_placed, 1)
                budget_features = torch.cat([
                    torch.tensor([budget / MAX_DECK_SIZE], device=device),
                    color_counts / placed_f,
                    type_counts / placed_f,
                    torch.tensor([cards_placed / MAX_DECK_SIZE], device=device),
                ])

            # 2. Encode archetype + card context + budget → query vector
            card_embs = torch.stack(card_context) if card_context else None
            q_t = self.context_encoder(arch_embedding, card_embs, budget_features)

            # 3. Build eligibility mask: only cards not yet selected
            eligible = ~selected  # (n_cards,) bool
            if curriculum_mask is not None:
                eligible = eligible & curriculum_mask

            if not eligible.any():
                break  # no more eligible cards (shouldn't happen in practice)

            # 4. Score all cards via MLP
            select_logits = self.card_selector(q_t, card_embeddings, eligible)

            # Temperature scaling (training only): divide logits before softmax
            # to control gradient distribution. Argmax is unaffected (monotonic).
            if temperature != 1.0 and not greedy:
                select_logits = select_logits / temperature

            # 5. Select a card (always argmax)
            card_idx = select_logits.argmax().item()
            card_emb = card_embeddings[card_idx]
            select_probs = F.softmax(select_logits, dim=-1)

            # 6. Predict copy count via type-specific head
            is_basic = basic_land_mask[card_idx].item()
            if is_basic:
                count_logits = self.basic_land_count_head(q_t, card_emb, budget)
            else:
                count_logits = self.nonbasic_count_head(q_t, card_emb, budget)
            count_idx = count_logits.argmax().item()
            count_probs = F.softmax(count_logits, dim=-1)

            count = count_idx + 1  # positions are 0-indexed, counts are 1-indexed
            count = min(count, budget)  # safety clamp

            # 7. Context correction with scheduled sampling:
            #    If the model picked wrong during training, flip a coin weighted
            #    by teacher_forcing_rate. Heads: substitute a GT card (in
            #    deterministic order). Tails: keep the model's own wrong pick.
            sub_idx = None
            use_model_pick = True
            if target_deck is not None and not greedy and card_idx not in target_deck:
                # Find eligible target cards (in GT and not yet selected)
                eligible_targets = [
                    c for c in target_remaining if not selected[c].item()
                ]
                if eligible_targets and random.random() < teacher_forcing_rate:
                    # Teacher forcing: substitute GT card (deterministic order)
                    use_model_pick = False
                    if target_order is not None:
                        # Pick the first card in deterministic order that is eligible
                        sub_idx = next(
                            (c for c in target_order if c in target_remaining and not selected[c].item()),
                            None,
                        )
                    if sub_idx is None:
                        sub_idx = random.choice(eligible_targets)
                    sub_count = target_deck[sub_idx]
                    sub_count = min(sub_count, budget)
                    target_remaining.discard(sub_idx)

                    # Update deck state with substituted card
                    deck[sub_idx] = sub_count
                    selected[sub_idx] = True
                    budget -= sub_count

                    # Update budget signal accumulators
                    if self._has_budget_signal:
                        color_counts += self.card_color_flags[sub_idx] * sub_count
                        type_counts += self.card_type_flags[sub_idx] * sub_count
                        cards_placed += sub_count

                    # Build context from substituted card
                    sub_card_emb = card_embeddings[sub_idx]
                    sub_count_idx = sub_count - 1
                    sub_count_emb = self.count_embedding(sub_count_idx)
                    merged = self.context_merge(sub_card_emb, sub_count_emb)
                    card_context.append(merged)

            # 8. Record step info for loss computation (always the model's actual pick).
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
                # Correct pick, greedy, no target, or scheduled sampling
                # chose to keep model's own pick
                if card_idx in target_remaining:
                    target_remaining.discard(card_idx)

                # Count teacher forcing: during training, use GT count for
                # deck state and context to prevent budget drift. The model's
                # predicted count is still in the step dict for loss.
                if target_deck is not None and not greedy and card_idx in target_deck:
                    gt_count = target_deck[card_idx]
                    gt_count = min(gt_count, budget)
                    deck[card_idx] = gt_count
                    selected[card_idx] = True
                    budget -= gt_count
                    actual_count = gt_count
                    count_emb = self.count_embedding(gt_count - 1)
                else:
                    deck[card_idx] = count
                    selected[card_idx] = True
                    budget -= count
                    actual_count = count
                    count_emb = self.count_embedding(count_idx)

                # Update budget signal accumulators
                if self._has_budget_signal:
                    color_counts += self.card_color_flags[card_idx] * actual_count
                    type_counts += self.card_type_flags[card_idx] * actual_count
                    cards_placed += actual_count

                merged = self.context_merge(card_emb, count_emb)
                card_context.append(merged)

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
        dropout: float = DROPOUT,
        gradient_checkpointing: bool = GRADIENT_CHECKPOINTING,
        edge_chunk_size: int = EDGE_CHUNK_SIZE,
        card_color_flags: torch.Tensor | None = None,
        card_type_flags: torch.Tensor | None = None,
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
            gradient_checkpointing=gradient_checkpointing,
            edge_chunk_size=edge_chunk_size,
        )
        self.decoder = DeckConstructor(
            d_model=d_model,
            d_count=d_count,
            dropout=dropout,
            card_color_flags=card_color_flags,
            card_type_flags=card_type_flags,
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
        greedy: bool = False,
        target_deck: dict[int, int] | None = None,
        curriculum_mask: torch.Tensor | None = None,
        target_order: list[int] | None = None,
        teacher_forcing_rate: float = 1.0,
        temperature: float = 1.0,
    ) -> dict:
        """Run full model: GNN embedding + autoregressive deck construction.

        Args:
            data: the HeteroData metagame graph.
            archetype_idx: which archetype to build a deck for.
            greedy: if True, use argmax without teacher forcing (inference).
            target_deck: {card_idx: count} ground truth for teacher forcing
                during training (see DeckConstructor.forward).
            curriculum_mask: (n_cards,) bool — curriculum eligibility mask
                (see DeckConstructor.forward). None = all cards eligible.
            target_order: deterministic card ordering for context correction.
            teacher_forcing_rate: scheduled sampling rate (1.0 = always sub).
            temperature: softmax temperature for card scorer.

        Returns:
            dict with 'deck' ({card_idx: count}) and 'steps' (per-step info).
        """
        x_dict = self.gnn(data)
        arch_emb = x_dict["archetype"][archetype_idx]
        card_emb = x_dict["card"]

        return self.decoder(
            arch_emb, card_emb, self.basic_land_mask, greedy, target_deck,
            curriculum_mask, target_order, teacher_forcing_rate, temperature,
        )

    def forward_with_embeddings(
        self,
        x_dict: dict[str, torch.Tensor],
        archetype_idx: int,
        greedy: bool = False,
        target_deck: dict[int, int] | None = None,
        curriculum_mask: torch.Tensor | None = None,
        target_order: list[int] | None = None,
        teacher_forcing_rate: float = 1.0,
        temperature: float = 1.0,
    ) -> dict:
        """Run deck construction using pre-computed GNN embeddings.

        Use this when the GNN output has been cached (e.g., to avoid
        redundant GNN passes across multiple archetypes in the same epoch).
        Gradients still flow through x_dict back to the GNN.

        Args:
            x_dict: pre-computed {node_type: embeddings} from self.gnn(data).
            archetype_idx: which archetype to build a deck for.
            greedy: if True, use argmax without teacher forcing (inference).
            target_deck: {card_idx: count} ground truth for teacher forcing.
            curriculum_mask: (n_cards,) bool — curriculum eligibility mask
                (see DeckConstructor.forward). None = all cards eligible.
            target_order: deterministic card ordering for context correction.
            teacher_forcing_rate: scheduled sampling rate (1.0 = always sub).
            temperature: softmax temperature for card scorer.

        Returns:
            dict with 'deck' ({card_idx: count}) and 'steps' (per-step info).
        """
        arch_emb = x_dict["archetype"][archetype_idx]
        card_emb = x_dict["card"]

        return self.decoder(
            arch_emb, card_emb, self.basic_land_mask, greedy, target_deck,
            curriculum_mask, target_order, teacher_forcing_rate, temperature,
        )

    @torch.no_grad()
    def predict_deck(
        self,
        data: HeteroData,
        archetype_idx: int,
        x_dict: dict[str, torch.Tensor] | None = None,
        legality_mask: torch.Tensor | None = None,
    ) -> dict[int, int]:
        """Convenience method for greedy inference.

        Args:
            data: the HeteroData graph (used if x_dict is None).
            archetype_idx: which archetype to build a deck for.
            x_dict: optional pre-computed GNN embeddings. If provided,
                skips the GNN forward pass.
            legality_mask: optional (n_cards,) bool — format legality mask.
                Only format-legal cards are eligible for selection.

        Returns:
            {card_idx: count} — a valid 60-card decklist.
        """
        self.eval()
        if x_dict is not None:
            result = self.forward_with_embeddings(
                x_dict, archetype_idx, greedy=True,
                curriculum_mask=legality_mask,
            )
        else:
            result = self.forward(
                data, archetype_idx, greedy=True,
                curriculum_mask=legality_mask,
            )
        return result["deck"]
