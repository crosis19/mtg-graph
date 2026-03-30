# MTG Metagame GNN: Autoregressive Deck Constructor

## Architecture Specification

---

## 1. System Overview

This architecture predicts competitive 60-card Magic: The Gathering mainboard decklists given an archetype identity. It operates in two stages:

1. **GNN Message Passing** — Learn card, archetype, and set embeddings from a heterogeneous metagame graph.
2. **Autoregressive Cross-Attention Deck Construction** — An archetype embedding iteratively selects cards and predicts copy counts via cross-attention over the card pool, building a deck one card-selection at a time until the 60-card budget is exhausted.

The model trains end-to-end using its own predictions (no teacher forcing), with **context correction** that substitutes ground-truth cards into the deck context when the model makes wrong selections during training. Gradient flow through discrete sampling decisions is handled via Gumbel-softmax relaxation.

---

## 2. Input Graph Structure

The input is a heterogeneous graph constructed from Scryfall card data, MTGGoldfish metagame/decklist data, and MTGDecks.net matchup data. All node embeddings are projected to dimension `d_model` (default: 128).

### 2.1 Node Types

| Node Type | Count | Feature Dim | Description |
|-----------|-------|-------------|-------------|
| **Card** | ~4,168 | 407 | Individual Standard-legal MTG card. Features: 384-dim sentence-transformer embedding (oracle text + type line + keywords + mana cost via `all-MiniLM-L6-v2`) + 23 scalar features. |
| **Archetype** | ~16 | 14 | Named deck archetype (e.g., "Azorius Control"). Top N archetypes by meta share percentage. |
| **Set** | ~12-16 | 385 | Standard-legal expansion set. Features: 384-dim mean embedding of all cards in the set + 1 recency score. |

#### Card Scalar Features (23-dim)

| Feature | Dim | Encoding |
|---------|-----|----------|
| `cmc_norm` | 1 | Normalized converted mana cost |
| Type flags | 8 | `is_creature`, `is_instant`, `is_sorcery`, `is_land`, `is_planeswalker`, `is_enchantment`, `is_artifact`, `is_battle` |
| `rarity_enc` | 1 | common=0.25, uncommon=0.5, rare=0.75, mythic=1.0 |
| Color identity | 6 | `is_white`, `is_blue`, `is_black`, `is_red`, `is_green`, `is_colorless` |
| Mana generation | 6 | `gen_white`, `gen_blue`, `gen_black`, `gen_red`, `gen_green`, `gen_colorless` (detected from oracle text) |

#### Archetype Features (14-dim)

| Feature | Dim | Encoding |
|---------|-----|----------|
| Color presence | 6 | Binary flags: `contains_white`, `contains_blue`, `contains_black`, `contains_red`, `contains_green`, `contains_colorless` |
| Card type ratios | 8 | Normalized: `creature_ratio`, `instant_ratio`, `sorcery_ratio`, `land_ratio`, `planeswalker_ratio`, `enchantment_ratio`, `artifact_ratio`, `battle_ratio` |

#### Set Features (385-dim)

| Feature | Dim | Encoding |
|---------|-----|----------|
| Semantic average | 384 | Mean of all card embeddings in the set |
| Recency | 1 | Normalized release date: `(release_date - min_date) / (max_date - min_date)` [0=oldest, 1=newest] |

### 2.2 Edge Types (11 total)

#### Card-to-Card Synergy (undirected, stored as bidirectional)

| Edge Type | Description |
|-----------|-------------|
| `(card, keyword_synergy, card)` | Complementary keyword abilities (e.g., Deathtouch + First Strike). Weight = number of synergy rules matched. 26 predefined rules + fuzzy category matching. |
| `(card, semantic_synergy, card)` | Embedding cosine similarity >= 0.65 (configurable). Excludes pairs already connected by keyword edges. Weights normalized to [0, 1]. |

#### Card-to-Card Interaction (directed, with reverse edges)

| Edge Type | Description |
|-----------|-------------|
| `(card, counters, card)` | Card can counter another card's spells. 8 regex patterns on oracle text (universal counter, creature spell, noncreature, etc.). |
| `(card, countered_by, card)` | Reverse of `counters`. |
| `(card, removes, card)` | Card can destroy/exile/damage/bounce another. 16 removal patterns with target type restrictions. |
| `(card, removed_by, card)` | Reverse of `removes`. |

#### Set-Card Relations (directed)

| Edge Type | Description |
|-----------|-------------|
| `(set, printed_in, card)` | Set contains this card. Weight = 1.0. |
| `(card, member_of, set)` | Reverse of `printed_in`. |

#### Archetype-Card Relations (directed)

| Edge Type | Description |
|-----------|-------------|
| `(archetype, contains, card)` | Archetype includes this card in decklists. Weight encodes copy count: non-lands use `min(copies, 4) * 0.25` (1->0.25, 2->0.5, 3->0.75, 4->1.0); basic lands use `actual_count / 20`. |
| `(card, in_deck, archetype)` | Reverse of `contains`. |

#### Archetype-Archetype Matchups (directed)

| Edge Type | Description |
|-----------|-------------|
| `(archetype, win_rate, archetype)` | Source's win rate vs target (continuous, 0-1). Scraped from MTGDecks.net (last 30 days). |
| `(archetype, lose_rate, archetype)` | Source's loss rate vs target (= target's win rate vs source). |

---

## 3. Stage 1: GNN Message Passing

### 3.1 Architecture

Use MLP-based message passing with a configurable message dimension (`d_message`) and **additive skip connections** to preserve original node identity.

For each GNN layer `l` (default: 2 layers):

```
message_j^(l) = AGG({ MLP_edge_type^(l)(c_k^(l)) : k in N_type(j) })
c_j^(l+1) = c_j^(0) + MLP_update^(l)(message_j^(l))
```

Where:
- `c_j^(0)` is the **original** embedding of node `j` (not the previous layer output -- this is the skip connection)
- `AGG` is mean aggregation over neighbors
- `MLP_edge_type^(l)` is a separate 2-layer MLP per edge type (since this is a heterogeneous graph)
- `MLP_update^(l)` is a 2-layer MLP that processes aggregated messages and projects back to `d_model`
- The addition `c_j^(0) + ...` is the skip connection

### 3.2 MLP Specifications

**MessageMLP** (one per edge type per layer):
- Projects from node space to message space: `d_model -> d_message -> d_message`
- ReLU activation between layers
- LayerNorm after the final layer
- Dropout (default: 0.1)

**UpdateMLP** (one per node type per layer):
- Projects from message space back to node space: `d_message -> d_message -> d_model`
- ReLU activation between layers
- LayerNorm after the final layer
- Dropout (default: 0.1)

Setting `d_message = d_model` recovers the standard same-dimension message passing. Setting `d_message < d_model` creates a bottleneck; `d_message > d_model` creates an expansion.

### 3.3 Heterogeneous Handling

Since different edge types carry different semantic meaning, use **separate message MLPs per edge type**. If a node receives messages from multiple edge types, aggregate each type independently (mean per type) and then sum:

```
message_j^(l) = SUM_{type} MEAN({ MLP_type^(l)(c_k^(l)) : k in N_type(j) })
```

### 3.4 Output

After `L` message-passing layers, the final embeddings are:
- Card embeddings: `C in R^{|cards| x d_model}`
- Archetype embeddings: `A in R^{|archetypes| x d_model}`
- Set embeddings: `S in R^{|sets| x d_model}`

Card and archetype embeddings are the inputs to Stage 2.

---

## 4. Stage 2: Autoregressive Deck Construction

### 4.1 High-Level Loop

Given an archetype embedding `a_i`, the model constructs a deck through the following loop:

```
Initialize:
    D_0 = [a_i]                     # deck context starts with just the archetype
    budget = 60                      # remaining card slots
    selected = {}                    # set of already-selected card indices
    deck = {}                        # card_index -> count mapping

Loop (step t = 0, 1, 2, ...):
    1. Compute context query q_t from D_t
    2. Score all eligible candidate cards via cross-attention
    3. Select card s_t (highest scoring eligible card)
    4. Predict count n_t for selected card
    5. Context correction (training only):
        - If s_t is NOT in the target deck and remaining GT cards exist:
          substitute a random remaining target card into the context
        - Otherwise: use the model's selection
    6. Update D_{t+1}, budget, selected, deck
    7. If budget == 0: terminate and return deck
```

### 4.2 Context Aggregation

At step `t`, the deck context `D_t` is a sequence of embeddings: the archetype embedding plus all previously selected card embeddings (count-conditioned). Stack these into a matrix `D_t in R^{(t+1) x d_model}`.

Apply a **self-attention block** over `D_t` to produce a context-aware representation:

```
D_t_attn = MultiHeadSelfAttention(D_t) + D_t     # with residual connection
D_t_norm = LayerNorm(D_t_attn)
q_t = MeanPool(D_t_norm) in R^{d_model}
```

Self-attention config:
- Heads: 4
- d_k = d_model / num_heads
- Single self-attention layer (not a full transformer block -- keep it lightweight)
- Standard residual connection + LayerNorm

The mean-pooled output `q_t` is the query vector for cross-attention over the card pool.

### 4.3 Card Selection via Cross-Attention

Project the context query and candidate card embeddings:

```
Q = q_t @ W_Q in R^{d_k}               # single query vector
K = C_eligible @ W_K in R^{|eligible| x d_k}
```

Where `C_eligible` is the subset of card embeddings not yet selected (i.e., indices not in `selected`).

Compute raw scores:

```
scores_j = (Q . K_j) / sqrt(d_k)    for each eligible card j
```

Apply masking: set `scores_j = -inf` for any card `j` that is not eligible.

Card selection distribution:

```
p(select j at step t) = softmax(scores)_j
```

At inference: select `s_t = argmax_j p(j)`.

During training (see Section 5): use Gumbel-softmax to produce a differentiable soft selection.

### 4.4 Count Prediction

Once card `s_t` is selected, predict how many copies to include. The count range depends on card type:

| Card Type | Valid Counts |
|-----------|--------------|
| Basic Land (Plains, Island, Swamp, Mountain, Forest) | 1 to min(20, remaining_budget) |
| All other cards | 1 to min(4, remaining_budget) |

Note: minimum count is always 1 (if you selected it, you're including at least one copy).

**Count prediction head:**

```
h = MLP_count([q_t || c_{s_t} || q_t * c_{s_t}]) in R^{d_h}
count_logits = W_count @ h + b_count in R^{20}      # 20 = max possible count
```

Where:
- `||` denotes concatenation
- `*` denotes element-wise (Hadamard) product
- MLP_count: `3 * d_model -> d_model -> d_model` (2-layer MLP with ReLU and LayerNorm)
- `W_count in R^{20 x d_model}`, `b_count in R^{20}`

**Masking invalid counts:**

```
mask_k = 0 if k is a valid count for this card, else -inf
p(n = k | s_t, D_t) = softmax(count_logits + mask)_k
```

For non-basic cards, mask positions 5-20 to `-inf`.
For basic lands, mask any count exceeding `remaining_budget`.
Always mask position 0 (count of zero is invalid -- card was selected).

At inference: `n_t = argmax_k p(n = k)`.

During training: use Gumbel-softmax over valid count logits.

### 4.5 Context Update

After selecting card `s_t` with count `n_t`:

1. Create a count-conditioned card representation:
   ```
   count_embed = CountEmbedding(n_t) in R^{d_count}     # learned embedding table, 20 entries
   card_with_count = MLP_merge([c_{s_t} || count_embed]) in R^{d_model}
   ```
   - `MLP_merge`: `(d_model + d_count) -> d_model` (single linear layer + ReLU)
   - Default `d_count`: 16

2. Append to context:
   ```
   D_{t+1} = stack(D_t, card_with_count)      # now shape (t+2) x d_model
   ```

3. Update budget: `budget -= n_t`

4. Add `s_t` to `selected` set

### 4.6 Multi-Head Cross-Attention

Use multiple attention heads for the card selection cross-attention (default: 4 heads). Each head operates in a `d_k = d_model / num_heads` dimensional subspace with its own `W_Q^(h)`, `W_K^(h)` projections. The scores from all heads are combined via a learned output projection before the final softmax:

```
head_h = (q_t @ W_Q^(h)) . (C_eligible @ W_K^(h))^T / sqrt(d_k)
combined = Concat(head_1, ..., head_H) @ W_O in R^{|eligible|}
p(select j) = softmax(combined)_j
```

---

## 5. Training Procedure

### 5.1 Self-Prediction with Context Correction

The model uses its own predictions during training with **context correction**: when the model selects a wrong card (not in the ground-truth deck), a random remaining target card is substituted into the deck context for the next step. This prevents error snowballing through the autoregressive loop while still penalizing the wrong selection via the loss.

- **Correct pick:** The model's selected card and count are used for both the loss and the context update. The Gumbel-softmax differentiable path is preserved for gradient flow through the context.
- **Wrong pick:** The loss penalizes the wrong selection. For the context update, a random remaining ground-truth card is substituted with its ground-truth count. The substitute card's embedding is a raw lookup (no Gumbel gradient through context for wrong picks). If no remaining ground-truth cards are available, the model's pick is used as fallback.

At inference, no context correction is applied -- the model runs purely on its own greedy selections.

### 5.2 Gumbel-Softmax for Differentiable Sampling

Since argmax is non-differentiable, use the **straight-through Gumbel-softmax estimator**:

**Forward pass:** Use hard argmax (pick one card / one count).

**Backward pass:** Use the continuous Gumbel-softmax relaxation for gradient computation.

For card selection at step t:

```
# Forward (hard):
s_t = argmax_j(scores_j)
one_hot_hard = one_hot(s_t, |eligible|)

# Backward (soft relaxation):
g_j ~ Gumbel(0, 1)                                    # sample Gumbel noise
soft_j = softmax((scores_j + g_j) / tau)              # temperature-scaled softmax

# Straight-through:
selection = one_hot_hard - soft.detach() + soft        # hard forward, soft backward
```

The soft card embedding used for context update during backprop:

```
c_soft_t = SUM_j selection_j * c_j      # weighted blend during backward pass
```

Same procedure for count prediction -- apply Gumbel-softmax over valid count logits.

### 5.3 Temperature Schedule

```
tau(epoch) = max(0.1, 1.0 * exp(-0.01 * epoch))
```

- Early training (tau ~ 1.0): Soft selections, broad gradient signal, exploratory.
- Late training (tau -> 0.1): Near-hard selections, fine-tuning precise card choices.
- Never anneal to exactly 0 (gradients vanish).

### 5.4 Loss Function

At each step `t`, the model incurs loss for both card selection and count prediction:

```
L_select_t = -log p(s_t* in target_deck | D_t)
L_count_t  = -log p(n_t* = ground_truth_count | s_t, D_t)
```

Where `s_t*` and `n_t*` are evaluated against the ground-truth target deck.

**Handling mismatches:** Since the model makes its own picks, it may select a card that isn't in the target deck at all.

- If `s_t` IS in the target deck: `L_select_t = -log p(s_t)` (reward the selection), `L_count_t = -log p(n = target_count_for_s_t)` (train count accuracy)
- If `s_t` is NOT in the target deck: `L_select_t` penalizes the selection by rewarding the probability mass that *should* have gone to target deck cards. Specifically: `L_select_t = -log( SUM_{j in target_remaining} p(j) )` -- the negative log of total probability on all remaining target cards. Mask out `L_count` for this step (count is meaningless for a wrong card).

Total loss per deck:

```
L = (SUM_t L_select_t) / n_steps + lambda * (SUM_t L_count_t) / n_count_steps
```

Default `lambda = 1.0`. If count accuracy lags behind selection accuracy during training, increase `lambda`.

### 5.5 Ground Truth Deck Representation

Each training example is an (archetype_index, deck) pair where `deck` is a dictionary:

```python
{
    card_index: count,    # e.g., {42: 4, 107: 3, 256: 2, ...}
    ...
}
```

The set of card indices present in `deck` is the target card set. The model's selections are compared against this set (order-independent). Non-basic card counts are clamped to [1, 4], basic land counts to [1, 20].

### 5.6 Training Enhancements

| Feature | Description |
|---------|-------------|
| **AdamW optimizer** | Decoupled weight decay regularization |
| **LR warmup + scheduling** | Linear warmup (default 5 epochs) followed by cosine annealing (also supports linear or no schedule) |
| **Archetype-holdout cross-validation** | K-fold CV (default: 5 folds, 3 held-out archetypes per fold). Evaluates generalization to unseen archetypes. |
| **Early stopping** | Patience-based on Jaccard metric (default: 10 validation rounds) |
| **Gradient clipping** | `clip_grad_norm_(max_norm=1.0)` |
| **Gradient accumulation** | Gradients accumulated across all training archetypes before optimizer step |
| **Recency filtering** | Decklists filtered to last N days (default: 30) for freshness |

---

## 6. Inference Procedure

```
Input: archetype_index i
Output: deck dictionary {card_index: count}

1. Run GNN forward pass to get all card/archetype/set embeddings
2. Initialize D_0 = [a_i], budget = 60, selected = {}, deck = {}
3. While budget > 0:
    a. Self-attend over D_t -> q_t
    b. Score all eligible cards via cross-attention
    c. s_t = argmax(scores)
    d. Predict count: n_t = argmax(count_logits) (after masking invalid counts)
    e. Clamp: n_t = min(n_t, budget)
    f. Update context, budget, selected, deck
4. Return deck
```

No context correction is applied at inference -- the model runs greedy end-to-end.

### 6.1 Beam Search (Optional Enhancement, Not Yet Implemented)

For higher-quality predictions, maintain `B` candidate partial decks at each step. At each step, expand each candidate with its top-`k` card choices and top-`k` count choices, score all `B * k * k` expansions, and keep the top `B`. This is expensive but can improve prediction quality at inference time.

---

## 7. Model Hyperparameters (Defaults)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `d_model` | 128 | Embedding dimension for all nodes |
| `d_message` | 128 | GNN message-passing hidden dimension. Can differ from `d_model` for bottleneck/expansion. |
| `d_count` | 16 | Count embedding dimension |
| `num_gnn_layers` | 2 | Message-passing rounds |
| `num_attn_heads` | 4 | For both self-attention and cross-attention |
| `d_k` | 32 | = d_model / num_attn_heads |
| `dropout` | 0.1 | Applied in MLPs and attention |
| `gumbel_tau_start` | 1.0 | Initial Gumbel temperature |
| `gumbel_tau_min` | 0.1 | Minimum Gumbel temperature |
| `gumbel_decay` | 0.01 | Exponential decay rate |
| `learning_rate` | 1e-4 | AdamW optimizer |
| `weight_decay` | 1e-5 | Decoupled L2 regularization |
| `count_loss_weight` | 1.0 | Lambda for count loss term |
| `max_deck_size` | 60 | Mainboard only |
| `max_basic_land_count` | 20 | Per basic land type |
| `max_nonbasic_count` | 4 | Standard rules |
| `warmup_epochs` | 5 | Linear warmup before main LR schedule |
| `lr_scheduler` | cosine | Schedule after warmup: cosine, linear, or none |
| `patience` | 10 | Early stopping patience (validation rounds) |
| `num_cv_folds` | 5 | Cross-validation folds |
| `num_val_archetypes` | 3 | Held-out archetypes per fold |
| `recency_days` | 30 | Decklist freshness filter |

---

## 8. Module Breakdown

The following PyTorch `nn.Module` classes comprise the implementation:

### 8.1 `NodeEncoder`
- Generic encoder used for all node types (card, archetype, set)
- Projects raw node features into `d_model`-dimensional embeddings
- Single linear layer + ReLU + LayerNorm
- One instance per node type
- Input: raw feature vector (dimension varies by node type)
- Output: `R^{d_model}`

### 8.2 `MessageMLP`
- Edge-type-specific message transformation in GNN
- Projects from node space to message space: `d_model -> d_message -> d_message`
- ReLU + LayerNorm + Dropout
- One instance per (edge_type, GNN_layer)
- Input: source node embeddings `R^{d_model}`
- Output: messages `R^{d_message}`

### 8.3 `UpdateMLP`
- Node-type-specific aggregation processor in GNN
- Projects from message space back to node space: `d_message -> d_message -> d_model`
- ReLU + LayerNorm + Dropout
- One instance per (node_type, GNN_layer)
- Input: aggregated messages `R^{d_message}`
- Output: update vector `R^{d_model}` (added to layer-0 embeddings via skip connection)

### 8.4 `HeteroGNNLayer`
- One message-passing round with separate MessageMLPs per edge type
- Mean aggregation per edge type, then sum across types targeting same node
- UpdateMLP per node type processes aggregated messages
- Input: node embeddings `{type: R^{N x d_model}}`, edge indices per type
- Output: per-node-type update vectors `{type: R^{N x d_model}}`

### 8.5 `HeteroGNN`
- Stacks `num_gnn_layers` instances of `HeteroGNNLayer`
- Stores layer-0 embeddings for additive skip connections
- Input: HeteroData graph with raw node features and edge indices
- Output: final embeddings for all node types `{type: R^{N x d_model}}`

### 8.6 `DeckContextEncoder`
- Multi-head self-attention over the growing deck context
- Residual connection + LayerNorm
- Mean pooling to produce single query vector `q_t`
- Input: `D_t in R^{(t+1) x d_model}`
- Output: `q_t in R^{d_model}`

### 8.7 `CardSelector`
- Multi-head cross-attention from `q_t` to eligible card embeddings
- Learned `W_Q`, `W_K`, `W_O` projections
- Produces selection logits over candidate pool
- Handles masking of already-selected cards (`-inf` for ineligible)
- Input: `q_t`, `C_eligible`, eligibility mask
- Output: selection logits `in R^{|cards|}`

### 8.8 `CountPredictor`
- MLP that takes `[q_t || c_{s_t} || q_t * c_{s_t}]` and produces count logits
- `3 * d_model -> d_model -> d_model` (ReLU + LayerNorm) then linear head to 20
- Handles masking based on card type (basic vs. non-basic) and remaining budget
- Input: `q_t`, `c_{s_t}`, card type flag, remaining budget
- Output: count logits `in R^{20}` (masked)

### 8.9 `CountEmbedding`
- `nn.Embedding(20, d_count)` -- learned embedding for counts 1-20
- Used in context update to condition card embeddings on their selected count

### 8.10 `ContextMerge`
- Takes a card embedding and a count embedding, produces a `d_model`-dimensional vector for context update
- Single linear layer: `(d_model + d_count) -> d_model` + ReLU
- Input: `c_{s_t} in R^{d_model}`, `count_embed in R^{d_count}`
- Output: `R^{d_model}`

### 8.11 `DeckConstructor`
- Top-level module that orchestrates the full autoregressive loop
- Holds: `DeckContextEncoder`, `CardSelector`, `CountPredictor`, `CountEmbedding`, `ContextMerge`
- Implements the step-by-step deck construction loop with context correction
- Accepts optional `target_deck` for context correction during training
- Handles Gumbel-softmax during training, argmax during inference
- Tracks budget, selected set, target remaining set, and context matrix
- Input: archetype embedding `a_i`, all card embeddings `C`, basic land mask, tau, greedy flag, optional target_deck
- Output: dict with `deck` ({card_index: count}) and `steps` (per-step logits/probs for loss)

### 8.12 `MTGDeckModel`
- Top-level wrapper that combines everything
- Holds: `HeteroGNN` (with `NodeEncoder` per type), `DeckConstructor`
- Precomputes and registers basic land mask from card names
- Input: HeteroData graph, archetype index, tau, greedy flag, optional target_deck
- Output: dict with `deck` and `steps`
- Convenience method `predict_deck()` for greedy inference

---

## 9. Data Pipeline

### 9.1 Phase 1: Data Ingestion

| Script | Input | Output | Description |
|--------|-------|--------|-------------|
| `ingest_cards.py` | Scryfall bulk JSON | `cards.parquet` (~4,168 cards) | Downloads oracle cards, filters to Standard-legal, normalizes features |
| `ingest_metagame.py` | MTGGoldfish scrape | `metagame.parquet` + `decklists.parquet` | Top N archetypes by meta share, card breakdowns per archetype |
| `ingest_matchups.py` | MTGDecks.net scrape | `matchups.parquet` | Head-to-head win rates between archetype pairs (last 30 days) |

### 9.2 Phase 2: Edge Construction

| Script | Output | Description |
|--------|--------|-------------|
| `keyword_matrix.py` | keyword synergy edges | 26 predefined keyword compatibility rules + fuzzy category matching |
| `card_embeddings.py` | semantic synergy edges + `card_embeddings.npy` | Pairwise cosine similarity >= 0.65, excluding keyword-connected pairs |
| `card_interaction_edges.py` | counter + removal edges | Regex patterns on oracle text for counterspells (8 patterns) and removal (16 patterns) |

### 9.3 Phase 3: Graph Construction

`graph_builder.py` assembles all nodes, features, and edges into a PyTorch Geometric `HeteroData` object saved as `graph.pt`.

### 9.4 Basic Land Identification

Hardcoded set of basic land card names:
```python
BASIC_LAND_NAMES = {"Plains", "Island", "Swamp", "Mountain", "Forest"}
```

Used to set the basic land mask per card, which controls count masking in `CountPredictor`.

---

## 10. Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Card Inclusion Accuracy (Precision)** | Fraction of cards in predicted deck that are in the ground truth deck |
| **Card Recall** | Fraction of ground truth deck cards that appear in predicted deck |
| **Exact Count Match** | Fraction of correctly-included cards where predicted count matches ground truth exactly |
| **Count MAE** | Mean absolute error of predicted count vs. ground truth count (for correctly-included cards only) |
| **Deck Similarity (Jaccard)** | Intersection-over-union of predicted vs. ground truth card sets. **Primary metric** for early stopping and model selection. |
| **Budget Compliance** | Fraction of predicted decks that sum to exactly 60 cards (should be 100% by construction) |
| **Rules Compliance** | Fraction of predicted decks with no card exceeding its copy limit (should be 100% by construction) |

---

## 11. Key Design Decisions and Rationale

1. **Additive skip connections in GNN:** Prevents embedding drift over message-passing rounds. The cross-attention in Stage 2 needs embeddings that still encode "what card this is," not just "what this card's neighborhood looks like."

2. **Configurable message dimension (`d_message`):** Decouples the GNN message-passing capacity from the node embedding dimension. A bottleneck (`d_message < d_model`) can regularize message passing; an expansion (`d_message > d_model`) increases representational capacity in the message space without affecting the rest of the architecture.

3. **Self-attention over deck context (not addition):** Addition is lossy -- after 15+ cards, a summed vector cannot distinguish which specific cards are present. Self-attention preserves compositional structure and enables the model to learn inter-card dependencies.

4. **Context correction during training:** Pure self-prediction causes error snowballing -- one wrong card pollutes the context, causing subsequent picks to go further off-track. Substituting a random ground-truth card into the context keeps the autoregressive state grounded while the loss still penalizes the wrong selection. This is a form of partial teacher forcing that applies only to the context, not to the loss or selection mechanism.

5. **Gumbel-softmax training (no teacher forcing on selections):** Eliminates exposure bias and the card ordering problem simultaneously. The model trains on its own selection logits exactly as it runs at inference.

6. **Classification over regression for counts:** Card count distributions are non-smooth (a card is often 0 or 4, rarely 2). Classification with masking respects both the discrete nature and the domain constraints.

7. **Hadamard product in count predictor:** The element-wise product `q_t * c_{s_t}` enables the MLP to learn multiplicative feature interactions between the deck context and the candidate card, which is more expressive than concatenation alone.

8. **Count-conditioned context update:** Embedding a card as "3 copies of Sheoldred" rather than just "Sheoldred" lets subsequent attention steps reason about deck composition more precisely (e.g., "I already have 3 Sheoldred, so I don't need more top-end threats").

9. **Archetype-holdout cross-validation:** Evaluates whether the model can generalize to unseen archetypes, which is the realistic deployment scenario when new archetypes emerge in the metagame.

10. **Multi-source graph edges:** Keyword synergy, semantic synergy, counter/removal interactions, set membership, deck composition, and matchup data each provide different structural signals. Separate MessageMLPs per edge type allow the GNN to learn type-specific transformations.
