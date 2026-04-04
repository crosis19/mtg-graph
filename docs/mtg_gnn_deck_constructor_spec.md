# MTG Metagame GNN: Autoregressive Deck Constructor

## Architecture Specification

---

## 1. System Overview

This architecture predicts competitive 60-card Magic: The Gathering mainboard decklists given an archetype identity. It supports multiple formats (Standard, Pioneer) in a unified graph with format-prefixed archetype nodes and shared card embeddings, enabling transfer learning across formats. It operates in two stages:

1. **GNN Message Passing** — Learn card, archetype, and set embeddings from a heterogeneous metagame graph.
2. **Autoregressive MLP-Scored Deck Construction** — At each step, the archetype embedding is concatenated with the mean of previously selected card+count embeddings and projected via MLP to produce a fixed-size query vector. This query is used to score all eligible cards via MLP scoring over the card pool and predict copy counts via separate classification heads for non-basic cards (1–4) and basic lands (1–20), building a deck one card-selection at a time until the 60-card budget is exhausted.

The model trains end-to-end using argmax card selection with two forms of teacher forcing: **context correction** substitutes ground-truth cards into the deck context when the model makes wrong selections, and **count teacher forcing** uses ground-truth copy counts for deck state on correct picks to prevent budget drift.

---

## 2. Input Graph Structure

The input is a heterogeneous graph constructed from Scryfall card data, MTGGoldfish metagame/decklist data, and MTGDecks.net matchup data across all active formats. Card nodes are shared; archetype nodes are format-prefixed (e.g., `standard::Azorius Control`, `pioneer::Rakdos Vampires`). All node embeddings are projected to dimension `d_model` (default: 128).

### 2.1 Node Types

| Node Type | Count | Feature Dim | Description |
|-----------|-------|-------------|-------------|
| **Card** | ~4,168 (Standard) / ~8-10k (Standard+Pioneer) | 407 | Individual MTG card legal in any active format. Features: 384-dim sentence-transformer embedding (oracle text + type line + keywords + mana cost via `all-MiniLM-L6-v2`) + 23 scalar features. |
| **Archetype** | ~16 Standard + ~32 Pioneer | 14 + N_formats | Named deck archetype, prefixed with format (e.g., `standard::Azorius Control`). Top N archetypes per format by meta share percentage. |
| **Set** | Dynamic (all sets with legal cards) | 385 | Expansion set containing at least one legal card. Features: 384-dim mean embedding of all cards in the set + 1 recency score. Sets are discovered dynamically from the card pool. |

#### Card Scalar Features (23-dim)

| Feature | Dim | Encoding |
|---------|-----|----------|
| `cmc_norm` | 1 | Normalized converted mana cost |
| Type flags | 8 | `is_creature`, `is_instant`, `is_sorcery`, `is_land`, `is_planeswalker`, `is_enchantment`, `is_artifact`, `is_battle` |
| `rarity_enc` | 1 | common=0.25, uncommon=0.5, rare=0.75, mythic=1.0 |
| Color identity | 6 | `is_white`, `is_blue`, `is_black`, `is_red`, `is_green`, `is_colorless` |
| Mana generation | 6 | `gen_white`, `gen_blue`, `gen_black`, `gen_red`, `gen_green`, `gen_colorless` (detected from oracle text) |

#### Archetype Features (14 + N_formats dim)

| Feature | Dim | Encoding |
|---------|-----|----------|
| Color presence | 6 | Binary flags: `contains_white`, `contains_blue`, `contains_black`, `contains_red`, `contains_green`, `contains_colorless` |
| Card type ratios | 8 | Normalized: `creature_ratio`, `instant_ratio`, `sorcery_ratio`, `land_ratio`, `planeswalker_ratio`, `enchantment_ratio`, `artifact_ratio`, `battle_ratio` |
| Format one-hot | N_formats | Binary flags per supported format (e.g., `is_standard`, `is_pioneer`). Derived from the archetype's `format::` prefix. |

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
message_j^(l) = AGG({ alpha_type^(l) * MLP_edge_type^(l)([c_k^(l) || f_{k->j}]) : k in N_type(j) })
c_j^(l+1) = c_j^(0) + MLP_update^(l)(message_j^(l))
```

Where:
- `f_{k->j}` is the **edge feature** from the graph (e.g., copy count, synergy strength, win rate), concatenated with the source embedding before the MessageMLP. Edges without stored features default to 1.0. These are semantic inputs to the MLP, not scaling factors.
- `alpha_type^(l)` is a **learnable per-edge-type scalar weight** (one per edge type per GNN layer, initialized to 1.0). The model learns which relationship types contribute most to message aggregation.
- `c_j^(0)` is the **original** embedding of node `j` (not the previous layer output -- this is the skip connection)
- `AGG` is mean aggregation over neighbors
- `MLP_edge_type^(l)` is a separate 2-layer MLP per edge type (since this is a heterogeneous graph)
- `MLP_update^(l)` is a 2-layer MLP that processes aggregated messages and projects back to `d_model`
- The addition `c_j^(0) + ...` is the skip connection

### 3.2 MLP Specifications

**MessageMLP** (one per edge type per layer):
- Takes source embedding concatenated with edge feature: `(d_model + 1) -> d_message -> d_message`
- Edge features (graph edge weights such as copy counts, synergy scores, win rates) are concatenated as input features, not used as scaling factors
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
message_j^(l) = SUM_{type} alpha_type^(l) * MEAN({ MLP_type^(l)([c_k^(l) || f_{k->j}]) : k in N_type(j) })
```

Edge weights from the graph (copy counts, synergy scores, win rates) are treated as **edge features** concatenated with source embeddings before the MessageMLP. This allows the MLP to learn nonlinear, weight-dependent transformations rather than simple magnitude scaling. Edge types without stored features default to 1.0.

Each edge type has a **learnable scalar weight** `alpha_type^(l)` (one per edge type per GNN layer, initialized to 1.0) that scales messages after the MLP. This allows the model to learn which relationship types are most informative for each layer of message passing.

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
    a_i                              # archetype embedding (fixed throughout)
    card_context = []                # list of selected card+count embeddings
    budget = 60                      # remaining card slots
    selected = {}                    # set of already-selected card indices
    deck = {}                        # card_index -> count mapping

Loop (step t = 0, 1, 2, ...):
    1. Compute context query q_t from [a_i, mean(card_context)] via MLP
    2. Score all eligible candidate cards via MLP scoring
    3. Select card s_t (highest scoring eligible card)
    4. Predict count n_t for selected card
    5. Context correction (training only):
        - If s_t is NOT in the target deck and remaining GT cards exist:
          substitute a random remaining target card into the context
        - Otherwise: use the model's selection
    6. Append count-conditioned card embedding to card_context
    7. Update budget, selected, deck
    8. If budget == 0: terminate and return deck
```

### 4.2 Context Aggregation

At step `t`, the deck context consists of the archetype embedding `a_i` (fixed throughout) and a list of previously selected card+count embeddings. Rather than attending over a growing sequence, the selected card embeddings are mean-pooled into a single summary vector and concatenated with the archetype embedding. A 2-layer MLP projects this fixed-size input into the query vector `q_t`:

```
card_mean = MeanPool(card_context) in R^{d_model}   # zero vector if no cards selected yet
q_t = MLP_proj([a_i || card_mean]) in R^{d_model}
```

MLP projection config:
- Input: `2 * d_model` (archetype + card mean concatenation)
- Architecture: `Linear(2*d_model, d_model) -> ReLU -> Linear(d_model, d_model) -> LayerNorm -> Dropout`
- Output: `d_model`

This design ensures:
- The archetype signal is always at full strength (never diluted by mean-pooling over a growing sequence)
- The MLP input is fixed-size at every step, enabling more stable learning
- The projection learns to combine "what am I building" (archetype) with "what have I picked so far" (card summary)

The output `q_t` is the query vector for MLP scoring over the card pool.

### 4.3 Card Selection via MLP Scoring

For each candidate card, concatenate the context query with the card embedding and their element-wise product, then score via a shared MLP. All cards are scored in a single batched forward pass:

```
x_j = [q_t || c_j || q_t * c_j] in R^{3 * d_model}    for each card j
score_j = Linear(MLP(x_j)) in R                         # scalar logit per card
```

Where:
- `||` denotes concatenation
- `*` denotes element-wise (Hadamard) product
- MLP: `3 * d_model -> d_model` (Linear + ReLU + LayerNorm)
- Final linear head: `d_model -> 1`

Apply masking: set `score_j = -inf` for any card `j` that is not eligible (already selected).

Card selection distribution:

```
p(select j at step t) = softmax(scores)_j
```

Both training and inference use `s_t = argmax_j p(j)`. During training, context correction and count teacher forcing (see Section 5) keep the autoregressive state grounded.

### 4.4 Count Prediction

Once card `s_t` is selected, predict how many copies to include using a **type-specific count head**. The selected card is routed to one of two separate classification heads based on whether it is a basic land:

| Card Type | Head | Output Dim | Valid Counts |
|-----------|------|------------|--------------|
| Basic Land (Plains, Island, Swamp, Mountain, Forest) | `BasicLandCountHead` | 20 | 1 to min(20, remaining_budget) |
| All other cards | `NonBasicCountHead` | 4 | 1 to min(4, remaining_budget) |

Note: minimum count is always 1 (if you selected it, you're including at least one copy).

**Count prediction heads:**

Both heads share the same architecture but with different output sizes:

```
h = MLP_count([q_t || c_{s_t} || q_t * c_{s_t}]) in R^{d_model}
count_logits = W_count @ h + b_count in R^{K}
```

Where:
- `||` denotes concatenation
- `*` denotes element-wise (Hadamard) product
- MLP_count: `3 * d_model -> d_model -> d_model` (2-layer MLP with ReLU and LayerNorm)
- K = 4 for `NonBasicCountHead`, K = 20 for `BasicLandCountHead`

**Masking invalid counts:**

```
mask_k = 0 if k is a valid count for this card, else -inf
p(n = k | s_t, D_t) = softmax(count_logits + mask)_k
```

Each head masks any count exceeding `remaining_budget`.

Both training and inference use `n_t = argmax_k p(n = k)`. During training, when the model selects a correct card, the ground-truth count is teacher-forced into the deck state and context update (preventing budget drift), while the model's predicted count is still used for loss computation.

### 4.5 Context Update

After selecting card `s_t` with count `n_t` (or ground-truth count during training for correct picks):

1. Create a count-conditioned card representation:
   ```
   count_embed = CountEmbedding(n_t) in R^{d_count}     # learned embedding table, 20 entries
   card_with_count = MLP_merge([c_{s_t} || count_embed]) in R^{d_model}
   ```
   - `MLP_merge`: `(d_model + d_count) -> d_model` (single linear layer + ReLU)
   - Default `d_count`: 16

2. Append to card context list:
   ```
   card_context.append(card_with_count)
   ```

3. Update budget: `budget -= n_t`

4. Add `s_t` to `selected` set

---

## 5. Training Procedure

### 5.1 Argmax Selection with Context Correction and Count Teacher Forcing

The model uses argmax for both card selection and count prediction during training, with two forms of teacher forcing to keep the autoregressive state grounded:

- **Correct pick:** The model's selected card is used for the loss. The **ground-truth count** is teacher-forced into the deck state and context update (preventing budget drift), while the model's predicted count is used for loss computation.
- **Wrong pick:** The loss penalizes the wrong selection. For the context update, a random remaining ground-truth card is substituted with its ground-truth count. If no remaining ground-truth cards are available, the model's pick is used as fallback.

At inference, no teacher forcing is applied -- the model runs purely on its own greedy selections.

### 5.2 Loss Function

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

### 5.3 Ground Truth Deck Representation

Each training example is an (archetype_index, deck) pair where `deck` is a dictionary:

```python
{
    card_index: count,    # e.g., {42: 4, 107: 3, 256: 2, ...}
    ...
}
```

The set of card indices present in `deck` is the target card set. The model's selections are compared against this set (order-independent). Non-basic card counts are clamped to [1, 4], basic land counts to [1, 20].

### 5.4 Training Enhancements

| Feature | Description |
|---------|-------------|
| **AdamW optimizer** | Decoupled weight decay regularization with differential learning rates: GNN backbone uses `base_lr * gnn_lr_scale` (default 0.1), decoder uses `base_lr` |
| **LR warmup + scheduling** | Linear warmup (default 5 epochs) followed by cosine annealing (also supports linear or no schedule). Both parameter groups share the same schedule. |
| **Archetype-holdout cross-validation** | K-fold CV (default: 5 folds, 3 held-out archetypes per fold). Evaluates generalization to unseen archetypes. |
| **Early stopping** | Patience-based on Jaccard metric (default: 10 validation rounds) |
| **Gradient clipping** | `clip_grad_norm_(max_norm=1.0)` |
| **Gradient accumulation** | Gradients accumulated across all training archetypes before optimizer step |
| **Recency filtering** | Decklists filtered to last N days (default: 30) for freshness |

### 5.5 Multi-Format Training and Transfer Learning

The training pipeline supports two modes controlled by the `--mode` CLI argument:

**Pre-train mode** (`--mode pretrain`, default):
- Trains on ALL archetypes from ALL active formats in the unified graph
- Cross-validation folds sample from the full archetype pool (Standard + Pioneer)
- Ground truth built from all format-namespaced decklists
- Saves checkpoint as `model_fold*.pt`

**Fine-tune mode** (`--mode finetune`):
- Loads a pre-trained model checkpoint (`--pretrained-path`)
- Filters ground truth to a target format (`--format-filter standard`)
- Reduces learning rate by `TRANSFER_FINETUNE_LR_SCALE` (default: 0.1x)
- **GNN freeze strategy**: Freezes `model.gnn.parameters()` for the first `TRANSFER_GNN_FREEZE_EPOCHS` (default: 5) epochs, then unfreezes. This lets the decoder adapt to format-specific patterns while preserving card representations learned during pre-training.
- Cross-validation over target format archetypes only
- Fewer epochs (`TRANSFER_FINETUNE_EPOCHS`, default: 30)

**Rationale**: Pioneer provides 2-3x more competitive archetypes and diverse deck compositions. Pre-training on this larger dataset yields richer card embeddings that transfer well to Standard, since Pioneer is a superset of Standard's card pool. The GNN freezing strategy prevents catastrophic forgetting of card relationships while allowing the decoder to specialize.

---

## 6. Inference Procedure

```
Input: archetype_index i
Output: deck dictionary {card_index: count}

1. Run GNN forward pass to get all card/archetype/set embeddings
2. Initialize a_i = archetype embedding, card_context = [], budget = 60, selected = {}, deck = {}
3. While budget > 0:
    a. q_t = MLP_proj([a_i || mean(card_context)])  (zero vector if empty)
    b. Score all eligible cards via MLP scoring
    c. s_t = argmax(scores)
    d. Predict count: n_t = argmax(count_logits) (after masking invalid counts)
    e. Clamp: n_t = min(n_t, budget)
    f. Append count-conditioned card embedding to card_context
    g. Update budget, selected, deck
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
| `dropout` | 0.1 | Applied in MLPs |
| `learning_rate` | 1e-4 | AdamW optimizer (decoder base LR) |
| `gnn_lr_scale` | 0.1 | GNN LR = `learning_rate * gnn_lr_scale`. Lower LR stabilizes embeddings. Set to 1.0 for uniform LR. |
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
- Takes source embeddings concatenated with edge features: `(d_model + 1) -> d_message -> d_message`
- Edge features (graph edge weights) are treated as input features, not scaling factors
- ReLU + LayerNorm + Dropout
- One instance per (edge_type, GNN_layer)
- Input: source node embeddings concatenated with edge feature `R^{d_model + 1}`
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
- Edge features (graph weights) concatenated with source embeddings before MessageMLP
- Learnable per-edge-type scalar weight scales messages after the MLP (initialized to 1.0)
- Mean aggregation per edge type, then sum across types targeting same node
- UpdateMLP per node type processes aggregated messages
- Input: node embeddings `{type: R^{N x d_model}}`, edge indices and features per type
- Output: per-node-type update vectors `{type: R^{N x d_model}}`

### 8.5 `HeteroGNN`
- Stacks `num_gnn_layers` instances of `HeteroGNNLayer`
- Stores layer-0 embeddings for additive skip connections
- Input: HeteroData graph with raw node features and edge indices
- Output: final embeddings for all node types `{type: R^{N x d_model}}`

### 8.6 `DeckContextEncoder`
- Mean-pools selected card embeddings and concatenates with the archetype embedding
- 2-layer MLP projection: `Linear(2*d_model, d_model) -> ReLU -> Linear(d_model, d_model) -> LayerNorm -> Dropout`
- Fixed-size input at every step — archetype signal never diluted
- Falls back to zero vector when no cards selected yet (step 0)
- Input: `arch_emb in R^{d_model}`, `card_embs in R^{num_selected x d_model}` (or None)
- Output: `q_t in R^{d_model}`

### 8.7 `MLPCardScorer`
- MLP-based scoring of all candidate cards against the context query
- For each card: concatenates `[q_t || c_j || q_t * c_j]` and scores via shared MLP
- All cards scored in a single batched forward pass (parallelized on GPU)
- `3 * d_model -> d_model` (Linear + ReLU + LayerNorm) then linear head to 1
- Handles masking of already-selected cards (`-inf` for ineligible)
- Input: `q_t in R^{d_model}`, all card embeddings `C in R^{N x d_model}`, eligibility mask
- Output: selection logits `in R^{|cards|}`

### 8.8 `NonBasicCountHead` / `BasicLandCountHead`
- Two separate MLP heads for count prediction, routed by card type
- Both take `[q_t || c_{s_t} || q_t * c_{s_t}]` and produce count logits
- `3 * d_model -> d_model -> d_model` (ReLU + LayerNorm) then linear head
- `NonBasicCountHead`: output `in R^4` (counts 1–4)
- `BasicLandCountHead`: output `in R^{20}` (counts 1–20)
- Each head masks counts exceeding remaining budget
- Input: `q_t`, `c_{s_t}`, remaining budget
- Output: count logits (masked)

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
- Holds: `DeckContextEncoder`, `MLPCardScorer`, `NonBasicCountHead`, `BasicLandCountHead`, `CountEmbedding`, `ContextMerge`
- Implements the step-by-step deck construction loop with context correction and count teacher forcing
- Accepts optional `target_deck` for teacher forcing during training
- Uses argmax for both card selection and count prediction
- During training: context correction substitutes GT cards on wrong picks; count teacher forcing uses GT counts on correct picks
- Tracks budget, selected set, target remaining set, and card context list
- Input: archetype embedding `a_i`, all card embeddings `C`, basic land mask, greedy flag, optional target_deck
- Output: dict with `deck` ({card_index: count}) and `steps` (per-step logits/probs for loss)

### 8.12 `MTGDeckModel`
- Top-level wrapper that combines everything
- Holds: `HeteroGNN` (with `NodeEncoder` per type), `DeckConstructor`
- Precomputes and registers basic land mask from card names
- Input: HeteroData graph, archetype index, greedy flag, optional target_deck
- Output: dict with `deck` and `steps`
- Convenience method `predict_deck()` for greedy inference

---

## 9. Data Pipeline

### 9.1 Phase 1: Data Ingestion

Cards are shared across all formats; metagame/decklists/matchups are per-format.

| Script | Input | Output | Description |
|--------|-------|--------|-------------|
| `ingest_cards.py` | Scryfall bulk JSON | `cards.parquet` (shared) | Downloads oracle cards, filters to cards legal in any active format, adds per-format legality columns |
| `ingest_metagame.py` | MTGGoldfish scrape | `{format}/metagame.parquet` + `{format}/decklists.parquet` | Top N archetypes per format by meta share, card breakdowns per archetype |
| `ingest_matchups.py` | MTGDecks.net scrape | `{format}/matchups.parquet` | Head-to-head win rates between archetype pairs per format (last 30 days) |

Per-format data is stored in `data/processed/{format}/` subdirectories. The `ingest_all.py` orchestrator calls card ingestion once (shared), then loops over `ACTIVE_FORMATS` for metagame and matchups.

### 9.2 Phase 2: Edge Construction (format-agnostic)

| Script | Output | Description |
|--------|--------|-------------|
| `keyword_matrix.py` | keyword synergy edges | 26 predefined keyword compatibility rules + fuzzy category matching |
| `card_embeddings.py` | semantic synergy edges + `card_embeddings.npy` | Pairwise cosine similarity >= 0.65, excluding keyword-connected pairs |
| `card_interaction_edges.py` | counter + removal edges | Regex patterns on oracle text for counterspells (8 patterns) and removal (16 patterns) |

All edge construction operates on the shared card pool and is format-agnostic.

### 9.3 Phase 3: Graph Construction

`graph_builder.py` assembles all nodes, features, and edges into a PyTorch Geometric `HeteroData` object saved as `graph.pt`. It loads metagame/decklists/matchups from all format subdirectories, prefixes archetype names with `format::`, and builds a unified multi-format graph. Set nodes are discovered dynamically from the card pool rather than a hardcoded set list.

### 9.4 Basic Land Identification

Hardcoded set of basic land card names:
```python
BASIC_LAND_NAMES = {"Plains", "Island", "Swamp", "Mountain", "Forest"}
```

Used to set the basic land mask per card, which controls routing between `NonBasicCountHead` and `BasicLandCountHead`.

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

1. **Additive skip connections in GNN:** Prevents embedding drift over message-passing rounds. The MLP card scorer in Stage 2 needs embeddings that still encode "what card this is," not just "what this card's neighborhood looks like."

2. **Configurable message dimension (`d_message`):** Decouples the GNN message-passing capacity from the node embedding dimension. A bottleneck (`d_message < d_model`) can regularize message passing; an expansion (`d_message > d_model`) increases representational capacity in the message space without affecting the rest of the architecture.

3. **Split-pool context with MLP projection (not self-attention over growing sequence):** The archetype embedding and mean of selected card embeddings are concatenated and projected via a 2-layer MLP. This gives the model a fixed-size input at every step, preserves the archetype signal at full strength regardless of how many cards have been selected, and avoids the learning instability of self-attention over variable-length sequences (1 to ~25 tokens). The MLP learns to combine "what archetype am I building" with "what does my deck look like on average so far."

4. **Context correction during training:** Pure self-prediction causes error snowballing -- one wrong card pollutes the context, causing subsequent picks to go further off-track. Substituting a random ground-truth card into the context keeps the autoregressive state grounded while the loss still penalizes the wrong selection. This is a form of partial teacher forcing that applies only to the context, not to the loss or selection mechanism.

5. **Argmax selection with context correction and count teacher forcing:** The model uses the same argmax selection at training and inference, eliminating train/test mismatch. Context correction prevents error snowballing from wrong card picks. Count teacher forcing prevents budget drift from wrong copy counts, keeping the autoregressive state grounded. This approach eliminates 3 Gumbel-softmax hyperparameters (tau_start, tau_min, decay) while providing more direct supervision.

6. **Classification over regression for counts:** Card count distributions are non-smooth (a card is often 0 or 4, rarely 2). Classification with masking respects both the discrete nature and the domain constraints.

7. **Hadamard product in card scorer and count heads:** The element-wise product `q_t * c_j` enables the MLPs to learn multiplicative feature interactions between the deck context and the candidate card, which is more expressive than concatenation alone. Used in both the card selection MLP and both count prediction heads.

8. **Separate count heads for basic vs non-basic:** Non-basic counts (1–4) and basic land counts (1–20) have very different distributions. Separate heads give each sub-problem the right-sized output space, eliminating masking gymnastics and allowing each to specialize.

9. **Count-conditioned context update:** Embedding a card as "3 copies of Sheoldred" rather than just "Sheoldred" lets subsequent attention steps reason about deck composition more precisely (e.g., "I already have 3 Sheoldred, so I don't need more top-end threats").

10. **Archetype-holdout cross-validation:** Evaluates whether the model can generalize to unseen archetypes, which is the realistic deployment scenario when new archetypes emerge in the metagame.

11. **Multi-source graph edges:** Keyword synergy, semantic synergy, counter/removal interactions, set membership, deck composition, and matchup data each provide different structural signals. Separate MessageMLPs per edge type allow the GNN to learn type-specific transformations.

12. **Edge weights as features, not scaling factors:** Graph edge weights (copy counts, synergy scores, win rates) encode semantic information that shouldn't be used as simple magnitude scaling. A copy count of 0.75 (meaning 3 copies) shouldn't halve the message strength. Concatenating these values as input features to the MessageMLP lets the model learn nonlinear, weight-dependent transformations.

13. **Learnable per-edge-type weights:** A single learnable scalar per edge type per GNN layer controls the overall contribution of each relationship type to message aggregation. This lets the model discover which edge types are most informative (e.g., synergy edges vs. matchup edges) without overfitting to individual edges. Per-edge-type weights generalize to new graphs (new cards, meta shifts) since they are tied to relationship types, not specific edge instances.

14. **Differential learning rates (GNN vs decoder):** The GNN backbone learns foundational graph representations that all downstream predictions depend on — unstable GNN updates ripple into every card selection. The decoder gets direct loss gradients, while the GNN receives more diffuse gradients that flow back through cached embeddings. A lower LR for the GNN (default: 10% of base LR) stabilizes embedding learning while letting the decoder adapt faster. The `gnn_lr_scale` multiplier (1.0 = uniform LR) is included in hyperparameter sweeps.

15. **Unified multi-format graph with format-prefixed archetypes:** Card-to-card edges (synergy, counters, removal) derive from oracle text and are format-agnostic — sharing them avoids redundant computation and enables cross-format learning. Archetype names are prefixed with `format::` (e.g., `standard::Mono-Red Aggro`, `pioneer::Rakdos Vampires`) to prevent collisions, since the same archetype name can exist in both formats with different card compositions. Format one-hot flags in archetype features let the model distinguish formats without separate graphs.

16. **Dynamic set discovery over hardcoded set lists:** Rather than maintaining a static list of legal set codes (which requires manual updates each set release), set nodes are created for every set code found in the card pool. This automatically handles set rotations and multi-format set overlap.

17. **Transfer learning via GNN freezing:** During fine-tuning, the GNN is frozen for the first N epochs while the decoder adapts to the target format. This prevents catastrophic forgetting of card relationship patterns learned during pre-training while allowing the autoregressive decoder to specialize. After unfreezing, the GNN fine-tunes at a reduced learning rate.
