"""Training loop for the Autoregressive Deck Constructor.

Trains a GNN + cross-attention model to predict 60-card MTG decklists
from archetype identity. The model builds decks autoregressively using
argmax selection with teacher forcing for context correction and counts.

Key design:
  1. Argmax card selection with context correction on wrong picks
  2. Count teacher forcing: GT counts used for deck state on correct picks
  3. Per-step loss: card selection CE + count prediction CE
  4. Archetype-holdout cross-validation for robust evaluation
"""

import gc
import json
import logging
import random
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric  # noqa: F401 — must be fully loaded before torch.load unpickles HeteroData

from src.config import (
    ACTIVE_FORMATS,
    BUDGET_SIGNAL,
    CARDS_PARQUET,
    CONSENSUS_LOSS_MIN_WEIGHT,
    CONSENSUS_LOSS_WEIGHTING,
    COUNT_LOSS_WEIGHT,
    CURRICULUM_NEG_EPOCHS,
    CURRICULUM_NEG_SCHEDULE,
    CURRICULUM_NEG_START,
    D_COUNT,
    D_MESSAGE,
    D_MODEL,
    DECK_BATCH_SIZE,
    DECK_BATCH_STRATEGY,
    DECK_COLOR_LOSS_WEIGHTING,
    DECK_LEARNING_RATE,
    DECK_TOP_N_ARCHETYPES,
    DECK_LR_SCHEDULER,
    DECKLISTS_PARQUET,
    DROPOUT,
    METAGAME_PARQUET,
    EDGE_CHUNK_SIZE,
    GNN_LR_SCALE,
    DECK_NUM_CV_FOLDS,
    DECK_NUM_EPOCHS,
    DECK_NUM_VAL_ARCHETYPES,
    DECK_PATIENCE,
    DECK_RECENCY_DAYS,
    DECK_VAL_EVERY,
    DECK_WARMUP_EPOCHS,
    GRAPH_PATH,
    MAX_BASIC_LAND_COUNT,
    MAX_DECK_SIZE,
    MAX_NONBASIC_COUNT,
    NUM_GNN_LAYERS,
    SCHEDULED_SAMPLING_END,
    SCHEDULED_SAMPLING_SCHEDULE,
    SCHEDULED_SAMPLING_START,
    SCORER_TEMP_END,
    SCORER_TEMP_SCHEDULE,
    SCORER_TEMP_START,
    TRANSFER_FINETUNE_EPOCHS,
    TRANSFER_FINETUNE_LR_SCALE,
    TRANSFER_GNN_FREEZE_EPOCHS,
    WEIGHT_DECAY,
    create_run_dir,
    format_path,
)
from src.deck_predictor import MTGDeckModel
from src.graph_builder import BASIC_LAND_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
#  Data Preparation
# ════════════════════════════════════════════════════════════════════


def _filter_recent_decklists(decklists: pd.DataFrame, days: int) -> pd.DataFrame:
    """Filter decklists to only include snapshots from the last N days."""
    if "snapshot_date" not in decklists.columns:
        return decklists
    decklists = decklists.copy()
    decklists["snapshot_date"] = pd.to_datetime(decklists["snapshot_date"], errors="coerce")
    cutoff = pd.Timestamp.now(tz="UTC") - timedelta(days=days)
    if decklists["snapshot_date"].dt.tz is None:
        cutoff = cutoff.tz_localize(None)
    recent = decklists[decklists["snapshot_date"] >= cutoff]
    if recent.empty:
        log.warning(f"  No decklists in last {days} days, using all data")
        return decklists
    log.info(f"  Filtered to {len(recent)} decklist rows from last {days} days "
             f"(dropped {len(decklists) - len(recent)})")
    return recent


def _build_ground_truth(
    data,
    decklists: pd.DataFrame,
) -> tuple[dict[int, dict[int, int]], dict[int, dict[int, float]]]:
    """Build per-archetype ground truth with raw integer copy counts.

    Uses only the latest snapshot per archetype so counts reflect the
    current meta. Returns:
      - ground_truth: {arch_idx: {card_idx: count}} where counts sum to 60
      - inclusion_rates: {arch_idx: {card_idx: float}} from inclusion_pct column
    Non-basic cards are clamped to [1, 4], basic lands to [1, 20].
    """
    arch_names = data["archetype"].names
    card_names = data["card"].names
    arch_to_idx = {a: i for i, a in enumerate(arch_names)}
    card_to_idx = {c: i for i, c in enumerate(card_names)}
    # Front-face aliases for multi-face cards (MTGGoldfish → Scryfall)
    for name, idx in list(card_to_idx.items()):
        if " // " in name:
            card_to_idx.setdefault(name.split(" // ")[0], idx)

    copies_col = "avg_copies" if "avg_copies" in decklists.columns else "copies"
    has_inclusion = "inclusion_pct" in decklists.columns

    # Filter to mainboard only
    if "board" in decklists.columns:
        decklists = decklists[decklists["board"] == "main"]

    # Use only the latest snapshot per archetype so counts reflect the
    # current meta rather than accumulating historical maximums.
    if "snapshot_date" in decklists.columns:
        latest = decklists.groupby("archetype")["snapshot_date"].transform("max")
        decklists = decklists[decklists["snapshot_date"] == latest]

    # One row per (archetype, card) — no cross-snapshot aggregation needed
    agg_cols = [copies_col]
    if has_inclusion:
        agg_cols.append("inclusion_pct")
    agg = (
        decklists.groupby(["archetype", "card_name"])[agg_cols]
        .first()
        .reset_index()
    )

    ground_truth: dict[int, dict[int, int]] = {}
    inclusion_rates: dict[int, dict[int, float]] = {}

    for _, row in agg.iterrows():
        arch = row["archetype"]
        card = row["card_name"]
        if arch not in arch_to_idx:
            continue
        # Normalize split card separator: "A/B" → "A // B"
        if card not in card_to_idx and "/" in card and " // " not in card:
            card = card.replace("/", " // ")
        if card not in card_to_idx:
            copies = round(float(row[copies_col]))
            log.warning("  Dropped %s (%d copies) from %s — card not in graph",
                         card, copies, arch)
            continue

        a_idx = arch_to_idx[arch]
        c_idx = card_to_idx[card]
        copies = round(float(row[copies_col]))

        # Clamp to legal range
        if card in BASIC_LAND_NAMES:
            copies = max(1, min(copies, MAX_BASIC_LAND_COUNT))
        else:
            copies = max(1, min(copies, MAX_NONBASIC_COUNT))

        ground_truth.setdefault(a_idx, {})[c_idx] = copies

        # Capture inclusion rate
        inc = float(row["inclusion_pct"]) if has_inclusion else 1.0
        inclusion_rates.setdefault(a_idx, {})[c_idx] = inc

    # Log summary
    for a_idx, cards in ground_truth.items():
        total = sum(cards.values())
        log.info(f"  Archetype {arch_names[a_idx]}: {len(cards)} unique cards, {total} total copies")

    return ground_truth, inclusion_rates


# ════════════════════════════════════════════════════════════════════
#  Curriculum Negative Sampling
# ════════════════════════════════════════════════════════════════════


def _identify_deck_and_negative_cards(
    ground_truth: dict[int, dict[int, int]],
    total_cards: int,
) -> tuple[set[int], list[int]]:
    """Partition card indices into deck cards and negatives.

    Deck cards = any card appearing in at least one archetype's ground-truth
    decklist. Negatives = all other cards in the graph.
    """
    deck_cards: set[int] = set()
    for cards in ground_truth.values():
        deck_cards.update(cards.keys())
    negative_cards = sorted(set(range(total_cards)) - deck_cards)
    return deck_cards, negative_cards


def _compute_curriculum_mask(
    epoch: int,
    deck_card_set: set[int],
    negative_cards: list[int],
    total_cards: int,
    neg_start: int,
    neg_epochs: int,
    schedule: str,
) -> torch.Tensor:
    """Build a boolean mask over all cards for this epoch's curriculum.

    All deck cards are always eligible. The number of additional negative
    (distractor) cards grows from *neg_start* to all negatives over
    *neg_epochs* epochs, then stays at the full pool.

    Returns a CPU bool tensor of shape (total_cards,).
    """
    total_neg = len(negative_cards)

    # After curriculum period or no negatives — full pool
    if epoch >= neg_epochs or total_neg == 0:
        return torch.ones(total_cards, dtype=torch.bool)

    # Clamp start to available negatives
    neg_start = min(neg_start, total_neg)

    # Compute how many negatives this epoch
    frac = epoch / neg_epochs
    if schedule == "exponential" and neg_start > 0:
        n_neg = int(neg_start * (total_neg / neg_start) ** frac)
    else:  # linear (default)
        n_neg = int(neg_start + (total_neg - neg_start) * frac)
    n_neg = min(n_neg, total_neg)

    # Sample a random subset for diversity across epochs
    sampled = set(random.sample(negative_cards, n_neg))

    mask = torch.zeros(total_cards, dtype=torch.bool)
    for idx in deck_card_set:
        mask[idx] = True
    for idx in sampled:
        mask[idx] = True
    return mask


# ════════════════════════════════════════════════════════════════════
#  Annealing Schedule
# ════════════════════════════════════════════════════════════════════


def _anneal_value(
    epoch: int, num_epochs: int, start: float, end: float, schedule: str,
) -> float:
    """Linearly or exponentially interpolate between *start* and *end*.

    Used for scheduled sampling rate, scorer temperature, and any other
    value that needs to anneal over the course of training.
    """
    frac = min(epoch / max(num_epochs - 1, 1), 1.0)
    if schedule == "exponential" and start > 0 and end > 0:
        return start * (end / start) ** frac
    return start + (end - start) * frac


# ════════════════════════════════════════════════════════════════════
#  Deterministic Card Ordering
# ════════════════════════════════════════════════════════════════════


def _build_target_order(
    ground_truth: dict[int, dict[int, int]],
    inclusion_rates: dict[int, dict[int, float]],
    card_names: list[str],
) -> dict[int, list[int]]:
    """Sort each archetype's target cards for deterministic decoding.

    Ordering: highest inclusion rate first, then copy count descending,
    then card name ascending (alphabetical tiebreaker). The model learns
    to lock in staples early and fill flex slots later.
    """
    target_orders: dict[int, list[int]] = {}
    for a_idx, cards in ground_truth.items():
        inc = inclusion_rates.get(a_idx, {})
        order = sorted(
            cards.keys(),
            key=lambda c: (-inc.get(c, 1.0), -cards[c], card_names[c]),
        )
        target_orders[a_idx] = order
    return target_orders


# ════════════════════════════════════════════════════════════════════
#  Loss Computation
# ════════════════════════════════════════════════════════════════════


def compute_step_losses(
    steps: list[dict],
    target_deck: dict[int, int],
    count_loss_weight: float,
    inclusion_rates: dict[int, float] | None = None,
    consensus_min_weight: float = 0.2,
) -> tuple[torch.Tensor, dict]:
    """Compute per-step card selection and count prediction losses.

    For each autoregressive step:
      - If the selected card IS in the target deck:
          L_select = -log(p(selected_card))
          L_count  = CrossEntropy(count_logits, ground_truth_count - 1)
      - If the selected card is NOT in the target deck:
          L_select = -log(sum of probs on all remaining target cards)
          L_count  = 0 (masked)

    When *inclusion_rates* is provided, correct-card losses are weighted
    by the card's consensus rate (clamped to *consensus_min_weight*).
    Staples get weight ~1.0; flex slots get the floor weight.

    Args:
        steps: per-step dicts from DeckConstructor.forward().
        target_deck: {card_idx: count} ground truth.
        count_loss_weight: lambda weighting for count loss.
        inclusion_rates: optional {card_idx: float} consensus rates.
        consensus_min_weight: floor weight for low-inclusion cards.

    Returns:
        (total_loss, metrics_dict)
    """
    target_remaining = dict(target_deck)  # mutable copy
    total_select_loss = torch.tensor(0.0, device=steps[0]["select_logits"].device)
    total_count_loss = torch.tensor(0.0, device=steps[0]["select_logits"].device)
    n_select = 0
    n_count = 0
    n_correct_cards = 0

    for step in steps:
        card_idx = step["card_idx"]
        count = step["count"]
        select_probs = step["select_probs"]
        count_logits = step["count_logits"]

        if card_idx in target_remaining:
            # Correct card selected — reward this choice
            prob = select_probs[card_idx].clamp(min=1e-10)
            step_select_loss = -torch.log(prob)

            # Consensus weighting: penalize missing staples more heavily
            if inclusion_rates is not None:
                cw = max(inclusion_rates.get(card_idx, 1.0), consensus_min_weight)
                step_select_loss = step_select_loss * cw

            total_select_loss = total_select_loss + step_select_loss
            n_select += 1
            n_correct_cards += 1

            # Count loss: cross-entropy against ground truth count
            gt_count = target_remaining[card_idx]
            gt_count_idx = gt_count - 1  # 0-indexed (count 1 → idx 0)

            # Only compute if the ground truth count is within valid logit range
            if gt_count_idx < count_logits.shape[0]:
                # Mask out -inf positions before computing CE
                valid_mask = count_logits > float("-inf")
                if valid_mask[gt_count_idx]:
                    target_tensor = torch.tensor(gt_count_idx, device=count_logits.device)
                    # Use only valid positions for cross-entropy
                    ce = F.cross_entropy(count_logits.unsqueeze(0), target_tensor.unsqueeze(0))
                    if inclusion_rates is not None:
                        cw = max(inclusion_rates.get(card_idx, 1.0), consensus_min_weight)
                        ce = ce * cw
                    total_count_loss = total_count_loss + ce
                    n_count += 1

            # Remove from remaining targets (card has been selected)
            del target_remaining[card_idx]

        else:
            # Wrong card — penalize by rewarding total prob on remaining targets
            if target_remaining:
                remaining_indices = list(target_remaining.keys())
                remaining_probs = select_probs[remaining_indices].sum().clamp(min=1e-10)
                total_select_loss = total_select_loss + (-torch.log(remaining_probs))
                n_select += 1
            # No count loss for wrong card selections

            # If context correction substituted a GT card into the context,
            # remove it from our tracking so we don't penalize future steps
            # for failing to pick a card that is already selected/ineligible.
            sub_idx = step.get("substituted_card_idx")
            if sub_idx is not None and sub_idx in target_remaining:
                del target_remaining[sub_idx]

    # Average losses
    avg_select = total_select_loss / max(n_select, 1)
    avg_count = total_count_loss / max(n_count, 1)
    total_loss = avg_select + count_loss_weight * avg_count

    metrics = {
        "select_loss": avg_select.item(),
        "count_loss": avg_count.item(),
        "total_loss": total_loss.item(),
        "correct_cards": n_correct_cards,
        "total_steps": len(steps),
    }
    return total_loss, metrics


# ════════════════════════════════════════════════════════════════════
#  Evaluation
# ════════════════════════════════════════════════════════════════════


@torch.no_grad()
def evaluate_fold(
    model: MTGDeckModel,
    data,
    ground_truth: dict[int, dict[int, int]],
    val_archetypes: list[int],
) -> dict:
    """Evaluate model on held-out archetypes via greedy decoding.

    Metrics:
      - precision: fraction of predicted cards that are in the target
      - recall: fraction of target cards that appear in the prediction
      - exact_count_match: among common cards, fraction with matching counts
      - count_mae: among common cards, mean absolute error of counts
      - jaccard: intersection-over-union of predicted vs target card sets
      - budget_compliance: fraction of decks summing to exactly 60
      - rules_compliance: fraction of decks with all counts within limits
    """
    model.eval()
    use_amp = next(model.parameters()).device.type == "cuda"

    # Cache GNN embeddings once for all validation archetypes
    with torch.amp.autocast("cuda", enabled=use_amp):
        x_dict = model.gnn(data)

    precision_list, recall_list = [], []
    exact_match_list, mae_list = [], []
    jaccard_list = []
    budget_ok, rules_ok = 0, 0
    n_evaluated = 0

    for a_idx in val_archetypes:
        gt = ground_truth.get(a_idx, {})
        if not gt:
            continue

        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model.predict_deck(data, a_idx, x_dict=x_dict)
        n_evaluated += 1
        gt_set = set(gt.keys())
        pred_set = set(pred.keys())

        # Precision and recall over card sets
        common = gt_set & pred_set
        precision = len(common) / max(len(pred_set), 1)
        recall = len(common) / max(len(gt_set), 1)
        precision_list.append(precision)
        recall_list.append(recall)

        # Jaccard similarity
        jaccard = len(common) / max(len(gt_set | pred_set), 1)
        jaccard_list.append(jaccard)

        # Count metrics on common cards
        if common:
            exact_matches = sum(1 for c in common if pred[c] == gt[c])
            exact_match_list.append(exact_matches / len(common))
            mae = np.mean([abs(pred[c] - gt[c]) for c in common])
            mae_list.append(mae)

        # Budget compliance: total copies == 60
        total_copies = sum(pred.values())
        if total_copies == 60:
            budget_ok += 1

        # Rules compliance: all counts within legal limits
        all_valid = all(
            (1 <= count <= MAX_BASIC_LAND_COUNT) if is_basic else (1 <= count <= MAX_NONBASIC_COUNT)
            for card_idx, count in pred.items()
            for is_basic in [model.basic_land_mask[card_idx].item()]
        )
        if all_valid:
            rules_ok += 1

    n = max(n_evaluated, 1)
    return {
        "precision": np.mean(precision_list) if precision_list else 0.0,
        "recall": np.mean(recall_list) if recall_list else 0.0,
        "exact_count_match": np.mean(exact_match_list) if exact_match_list else 0.0,
        "count_mae": np.mean(mae_list) if mae_list else 0.0,
        "jaccard": np.mean(jaccard_list) if jaccard_list else 0.0,
        "budget_compliance": budget_ok / n,
        "rules_compliance": rules_ok / n,
    }


# ════════════════════════════════════════════════════════════════════
#  Cross-Validation Folds
# ════════════════════════════════════════════════════════════════════


def _create_cv_folds(
    archetype_indices: list[int],
    n_folds: int,
    n_val: int,
) -> list[tuple[list[int], list[int]]]:
    """Create cross-validation folds for archetype holdout."""
    n = len(archetype_indices)
    indices = archetype_indices.copy()
    random.shuffle(indices)

    folds = []
    for fold_i in range(n_folds):
        start = (fold_i * n_val) % n
        val_indices = [indices[(start + j) % n] for j in range(n_val)]
        train_indices = [i for i in indices if i not in val_indices]
        folds.append((train_indices, val_indices))

    return folds


# ════════════════════════════════════════════════════════════════════
#  Color-Balanced Batching & Loss Weighting Helpers
# ════════════════════════════════════════════════════════════════════

COLOR_NAMES = ["W", "U", "B", "R", "G", "C"]


def _get_archetype_color_groups(
    data,
    archetype_indices: list[int],
) -> dict[str, list[int]]:
    """Group archetypes by the card-derived color flags in the graph.

    Reads the first 6 dims of archetype node features (W/U/B/R/G/colorless
    binary flags, computed from the union of card color identities in the
    deck). Multi-color archetypes appear in multiple buckets.

    Returns dict mapping color letter -> list of archetype indices.
    """
    groups: dict[str, list[int]] = {c: [] for c in COLOR_NAMES}
    for a_idx in archetype_indices:
        flags = data["archetype"].x[a_idx, :6]
        has_any = False
        for i, color in enumerate(COLOR_NAMES):
            if flags[i] > 0.5:
                groups[color].append(a_idx)
                has_any = True
        if not has_any:
            groups["C"].append(a_idx)  # colorless fallback
    return {c: idxs for c, idxs in groups.items() if idxs}


def _color_balanced_batches(
    color_groups: dict[str, list[int]],
    batch_size: int,
    all_archetypes: list[int],
    epoch: int,
) -> list[list[int]]:
    """Generate batches with approximate color balance.

    Works with any batch size — no minimum required. For each batch,
    greedily selects archetypes that best improve color coverage by
    prioritizing underrepresented colors. Multi-color archetypes
    contribute to all their colors simultaneously.

    Every archetype is guaranteed to appear at least once per epoch.
    """
    rng = random.Random(42 + epoch)

    # Build archetype -> set of colors mapping
    arch_colors: dict[int, set[str]] = {}
    for color, indices in color_groups.items():
        for a_idx in indices:
            arch_colors.setdefault(a_idx, set()).add(color)

    # Target color distribution (proportional to how many archetypes
    # have each color, so we're balancing relative representation)
    active_colors = list(color_groups.keys())

    # Pool of archetypes to draw from, shuffled for this epoch
    pool = list(all_archetypes)
    rng.shuffle(pool)

    seen = set()
    batches = []
    pool_offset = 0  # tracks position through shuffled pool

    n_batches = max((len(all_archetypes) + batch_size - 1) // batch_size, 1)
    for _ in range(n_batches):
        batch = []
        # Track color counts within this batch
        batch_color_counts = {c: 0 for c in active_colors}

        for _ in range(batch_size):
            # Score each candidate: sum of 1/(1+count) for each color it covers.
            # This naturally prioritizes archetypes whose colors are
            # underrepresented in the current batch.
            best_idx = None
            best_score = -1.0

            # Check a window of candidates from the shuffled pool
            window_size = min(len(pool), max(batch_size * 3, 10))
            candidates = []
            for j in range(window_size):
                c_idx = pool[(pool_offset + j) % len(pool)]
                candidates.append(c_idx)

            for c_idx in candidates:
                score = sum(
                    1.0 / (1.0 + batch_color_counts.get(c, 0))
                    for c in arch_colors.get(c_idx, set())
                )
                # Small tiebreaker: prefer unseen archetypes
                if c_idx not in seen:
                    score += 0.1
                if score > best_score:
                    best_score = score
                    best_idx = c_idx

            if best_idx is not None:
                batch.append(best_idx)
                seen.add(best_idx)
                for c in arch_colors.get(best_idx, set()):
                    batch_color_counts[c] = batch_color_counts.get(c, 0) + 1
                # Advance past the picked candidate in the pool
                pool_offset = (pool_offset + 1) % len(pool)

        if batch:
            batches.append(batch)

    # If some archetypes still unseen, add a cleanup batch
    unseen = set(all_archetypes) - seen
    if unseen:
        cleanup = list(unseen)
        rng.shuffle(cleanup)
        batches.append(cleanup)

    return batches


def _compute_color_loss_weights(
    data,
    archetype_indices: list[int],
) -> dict[int, float]:
    """Compute inverse color-frequency loss weights per archetype.

    Archetypes with the same exact color identity (e.g., both WU) share
    a group. Weight = 1 / group_size, then normalized so weights average 1.0.
    """
    from collections import Counter

    # Map each archetype to its color identity as a frozenset
    # Uses the card-derived color flags from the graph node features
    arch_to_identity: dict[int, frozenset] = {}
    for a_idx in archetype_indices:
        flags = data["archetype"].x[a_idx, :6]
        colors = frozenset(
            COLOR_NAMES[i] for i in range(6) if flags[i] > 0.5
        )
        if not colors:
            colors = frozenset(["C"])
        arch_to_identity[a_idx] = colors

    # Count archetypes per identity
    identity_counts = Counter(arch_to_identity.values())

    # Raw weight = inverse frequency
    raw_weights = {
        a_idx: 1.0 / identity_counts[identity]
        for a_idx, identity in arch_to_identity.items()
    }

    # Normalize so weights average to 1.0
    n = len(raw_weights)
    total = sum(raw_weights.values())
    scale = n / total if total > 0 else 1.0

    weights = {a_idx: w * scale for a_idx, w in raw_weights.items()}

    # Log distribution
    identity_weights = {}
    for a_idx, identity in arch_to_identity.items():
        key = "".join(sorted(identity))
        identity_weights[key] = weights[a_idx]
    log.info(f"  Color loss weights: {identity_weights}")

    return weights


# ════════════════════════════════════════════════════════════════════
#  Main Training Loop
# ════════════════════════════════════════════════════════════════════


def train_deck(device=None, trial=None, **overrides):
    """Train the autoregressive deck constructor.

    Parameters
    ----------
    device : torch.device, optional
    trial : optuna.trial.Trial, optional
        For hyperparameter sweeps.
    **overrides : keyword args for hyperparameters

    Returns dict with model, data, metrics, and run info.
    """
    if device is None:
        device = torch.device("cpu")

    hp = {
        "d_model": overrides.get("d_model", D_MODEL),
        "d_message": overrides.get("d_message", D_MESSAGE),
        "d_count": overrides.get("d_count", D_COUNT),
        "num_gnn_layers": overrides.get("num_gnn_layers", NUM_GNN_LAYERS),
        "dropout": overrides.get("dropout", DROPOUT),
        "learning_rate": overrides.get("learning_rate", DECK_LEARNING_RATE),
        "weight_decay": overrides.get("weight_decay", WEIGHT_DECAY),
        "num_epochs": overrides.get("num_epochs", DECK_NUM_EPOCHS),
        "patience": overrides.get("patience", DECK_PATIENCE),
        "recency_days": overrides.get("recency_days", DECK_RECENCY_DAYS),
        "warmup_epochs": overrides.get("warmup_epochs", DECK_WARMUP_EPOCHS),
        "lr_scheduler": overrides.get("lr_scheduler", DECK_LR_SCHEDULER),
        "count_loss_weight": overrides.get("count_loss_weight", COUNT_LOSS_WEIGHT),
        "n_val_archetypes": overrides.get("n_val_archetypes", DECK_NUM_VAL_ARCHETYPES),
        "n_cv_folds": overrides.get("n_cv_folds", DECK_NUM_CV_FOLDS),
        "batch_size": overrides.get("batch_size", DECK_BATCH_SIZE),
        "val_every": overrides.get("val_every", DECK_VAL_EVERY),
        "gnn_lr_scale": overrides.get("gnn_lr_scale", GNN_LR_SCALE),
        "curriculum_neg_start": overrides.get("curriculum_neg_start", CURRICULUM_NEG_START),
        "curriculum_neg_epochs": overrides.get("curriculum_neg_epochs", CURRICULUM_NEG_EPOCHS),
        "curriculum_neg_schedule": overrides.get("curriculum_neg_schedule", CURRICULUM_NEG_SCHEDULE),
        "edge_chunk_size": overrides.get("edge_chunk_size", EDGE_CHUNK_SIZE),
        "batch_strategy": overrides.get("batch_strategy", DECK_BATCH_STRATEGY),
        "color_loss_weighting": overrides.get("color_loss_weighting", DECK_COLOR_LOSS_WEIGHTING),
        "top_n_archetypes": overrides.get("top_n_archetypes", DECK_TOP_N_ARCHETYPES),
        "consensus_loss_weighting": overrides.get("consensus_loss_weighting", CONSENSUS_LOSS_WEIGHTING),
        "consensus_loss_min_weight": overrides.get("consensus_loss_min_weight", CONSENSUS_LOSS_MIN_WEIGHT),
        "scheduled_sampling_start": overrides.get("scheduled_sampling_start", SCHEDULED_SAMPLING_START),
        "scheduled_sampling_end": overrides.get("scheduled_sampling_end", SCHEDULED_SAMPLING_END),
        "scheduled_sampling_schedule": overrides.get("scheduled_sampling_schedule", SCHEDULED_SAMPLING_SCHEDULE),
        "budget_signal": overrides.get("budget_signal", BUDGET_SIGNAL),
        "scorer_temp_start": overrides.get("scorer_temp_start", SCORER_TEMP_START),
        "scorer_temp_end": overrides.get("scorer_temp_end", SCORER_TEMP_END),
        "scorer_temp_schedule": overrides.get("scorer_temp_schedule", SCORER_TEMP_SCHEDULE),
    }

    results_base = overrides.get("results_base", None)
    run_dir = create_run_dir(task="deck", results_base=results_base)
    log.info(f"Results will be saved to {run_dir}")
    log.info(f"Hyperparameters: {hp}")

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Use module-level reference so Colab patching of config paths works
    import src.config as _cfg

    # ── Load graph ──
    log.info("Loading graph...")
    data = torch.load(_cfg.GRAPH_PATH, weights_only=False)
    log.info(f"  Nodes: {', '.join(f'{nt}={data[nt].x.shape[0]}' for nt in data.node_types)}")
    log.info(f"  Edge types: {len(data.edge_types)}")

    # ── Load and filter decklists from all format subdirs ──
    mode = overrides.get("mode", "pretrain")
    format_filter = overrides.get("format_filter", None)

    _dl_base = _cfg.DECKLISTS_PARQUET

    dl_frames = []
    for fmt in ACTIVE_FORMATS:
        dl_path = format_path(_dl_base, fmt)
        if dl_path.exists():
            df = pd.read_parquet(dl_path)
            if "format" not in df.columns:
                df["format"] = fmt
            # Prefix archetype names to match graph node names
            df["archetype"] = fmt + "::" + df["archetype"]
            dl_frames.append(df)
    decklists = pd.concat(dl_frames, ignore_index=True) if dl_frames else pd.DataFrame()

    # In finetune mode, filter to target format only
    if mode == "finetune" and format_filter:
        prefix = format_filter + "::"
        before = len(decklists)
        decklists = decklists[decklists["archetype"].str.startswith(prefix)]
        log.info(f"  Finetune filter: {format_filter} → {len(decklists)}/{before} rows")

    log.info(f"\nFiltering decklists to last {hp['recency_days']} days...")
    decklists = _filter_recent_decklists(decklists, hp["recency_days"])

    # ── Build ground truth ──
    log.info("\nBuilding ground truth (raw integer counts)...")
    ground_truth, inclusion_rates = _build_ground_truth(data, decklists)

    # ── Identify deck vs negative cards for curriculum sampling ──
    n_total_cards = data["card"].x.shape[0]
    deck_card_set, negative_cards = _identify_deck_and_negative_cards(
        ground_truth, n_total_cards,
    )
    log.info(f"  Curriculum pool: {len(deck_card_set)} deck cards, "
             f"{len(negative_cards)} negative cards "
             f"(start={hp['curriculum_neg_start']}, "
             f"epochs={hp['curriculum_neg_epochs']}, "
             f"schedule={hp['curriculum_neg_schedule']})")

    # ── Filter to top N archetypes per format if requested ──
    # Graph retains all archetypes; this only limits which ones we train on.
    top_n = hp["top_n_archetypes"]
    if top_n > 0:
        arch_names_pre = data["archetype"].names
        # Load metagame data to rank by meta share
        top_arch_names: set[str] = set()
        for fmt in ACTIVE_FORMATS:
            meta_path = format_path(METAGAME_PARQUET, fmt)
            if not meta_path.exists():
                continue
            meta = pd.read_parquet(meta_path)
            latest = (
                meta.sort_values("snapshot_date")
                .groupby("archetype").last().reset_index()
            )
            latest = latest.dropna(subset=["meta_share_pct"])
            top_fmt = latest.nlargest(top_n, "meta_share_pct")
            # Prefix to match graph node names
            top_arch_names |= {fmt + "::" + a for a in top_fmt["archetype"]}

        # Filter ground_truth to only top N archetypes
        before = len(ground_truth)
        ground_truth = {
            a_idx: gt for a_idx, gt in ground_truth.items()
            if arch_names_pre[a_idx] in top_arch_names
        }
        log.info(f"  Top-{top_n} filter: {len(ground_truth)}/{before} archetypes kept for training")
    else:
        log.info(f"  Training on all {len(ground_truth)} archetypes with ground truth")

    # ── Deterministic card ordering for context correction ──
    card_names_pre = data["card"].names
    target_orders = _build_target_order(ground_truth, inclusion_rates, card_names_pre)
    log.info(f"  Built deterministic target orders for {len(target_orders)} archetypes")

    # ── Extract card metadata for budget signal ���─
    card_color_flags = None
    card_type_flags = None
    if hp["budget_signal"]:
        emb_dim = 384
        card_x = data["card"].x
        # Scalar feature layout after embedding: cmc_norm(0), is_creature(1)..is_artifact(7),
        # is_battle(8), rarity_enc(9), is_white(10)..is_green(14), is_colorless(15), ...
        card_type_flags = card_x[:, emb_dim + 1 : emb_dim + 8].clone()   # 7 type flags
        card_color_flags = card_x[:, emb_dim + 10 : emb_dim + 15].clone()  # 5 color flags (WUBRG)
        log.info(f"  Budget signal: extracted type ({card_type_flags.shape[1]}) "
                 f"and color ({card_color_flags.shape[1]}) flags for {card_x.shape[0]} cards")

    # ── Move to device ──
    data = data.to(device)
    card_names = data["card"].names
    arch_names = data["archetype"].names
    if card_color_flags is not None:
        card_color_flags = card_color_flags.to(device)
        card_type_flags = card_type_flags.to(device)

    # ── Create CV folds ──
    all_arch_indices = sorted(ground_truth.keys())
    folds = _create_cv_folds(all_arch_indices, hp["n_cv_folds"], hp["n_val_archetypes"])
    log.info(f"\n{hp['n_cv_folds']}-fold CV with {hp['n_val_archetypes']} held-out archetypes per fold")

    for fold_i, (train_archs, val_archs) in enumerate(folds):
        train_names = [arch_names[i] for i in train_archs]
        val_names = [arch_names[i] for i in val_archs]
        log.info(f"  Fold {fold_i}: train={train_names}, val={val_names}")

    # For Optuna: single fold. Otherwise: all folds.
    folds_to_run = [folds[0]] if trial is not None else folds
    all_fold_results = []

    for fold_i, (train_archs, val_archs) in enumerate(folds_to_run):
        fold_label = f"Fold {fold_i}/{len(folds_to_run)}"
        log.info(f"\n{'='*60}")
        log.info(f"{fold_label}: Training on {len(train_archs)} archetypes, "
                 f"validating on {len(val_archs)}")
        log.info(f"{'='*60}")

        # ── Initialize model ──
        node_dims = {nt: data[nt].x.shape[1] for nt in data.node_types}
        model = MTGDeckModel(
            node_dims=node_dims,
            edge_types=list(data.edge_types),
            node_types=list(data.node_types),
            card_names=card_names,
            d_model=hp["d_model"],
            d_message=hp["d_message"],
            d_count=hp["d_count"],
            num_gnn_layers=hp["num_gnn_layers"],
            dropout=hp["dropout"],
            edge_chunk_size=hp["edge_chunk_size"],
            card_color_flags=card_color_flags,
            card_type_flags=card_type_flags,
        ).to(device)

        # Load pretrained weights for finetune mode
        pretrained_path = overrides.get("pretrained_path", None)
        if mode == "finetune" and pretrained_path and fold_i == 0:
            log.info(f"Loading pretrained weights from {pretrained_path}")
            ckpt = torch.load(pretrained_path, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)

        if fold_i == 0:
            total_params = sum(p.numel() for p in model.parameters())
            log.info(f"Model parameters: {total_params:,}")

        # Differential learning rates: GNN backbone gets scaled LR
        gnn_params = list(model.gnn.parameters())
        gnn_param_ids = {id(p) for p in gnn_params}
        decoder_params = [p for p in model.parameters() if id(p) not in gnn_param_ids]

        base_lr = hp["learning_rate"]
        if mode == "finetune":
            base_lr *= TRANSFER_FINETUNE_LR_SCALE
        gnn_lr = base_lr * hp["gnn_lr_scale"]

        # In finetune mode, optionally freeze GNN for first N epochs
        gnn_freeze_epochs = TRANSFER_GNN_FREEZE_EPOCHS if mode == "finetune" else 0
        if gnn_freeze_epochs > 0 and fold_i == 0:
            log.info(f"  Freezing GNN for first {gnn_freeze_epochs} epochs (finetune)")
            for p in gnn_params:
                p.requires_grad = False

        optimizer = torch.optim.AdamW(
            [
                {"params": gnn_params, "lr": gnn_lr},
                {"params": decoder_params, "lr": base_lr},
            ],
            weight_decay=hp["weight_decay"],
        )
        if fold_i == 0:
            log.info(f"LR: GNN={gnn_lr:.1e} (scale={hp['gnn_lr_scale']}), Decoder={base_lr:.1e}"
                     f"{' [finetune]' if mode == 'finetune' else ''}")

        # ── LR scheduler ──
        warmup_epochs = hp["warmup_epochs"]
        sched_type = hp["lr_scheduler"].lower()
        post_warmup = max(hp["num_epochs"] - warmup_epochs, 1)

        if sched_type == "cosine":
            base_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=post_warmup)
        elif sched_type == "linear":
            base_sched = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.01, total_iters=post_warmup,
            )
        else:
            base_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

        if warmup_epochs > 0:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_sched, base_sched], milestones=[warmup_epochs],
            )
        else:
            scheduler = base_sched

        # ── Mixed precision ──
        use_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        # NOTE: torch.compile with mode="reduce-overhead" uses CUDA Graphs
        # which pre-allocate massive GPU memory pools (~60+ GB) that cannot
        # be freed between folds or sweep trials. Disabled to avoid OOM.
        # If you want to enable it for single-run training (not sweeps),
        # uncomment the block below.
        # if device.type == "cuda" and hasattr(torch, "compile") and trial is None:
        #     try:
        #         model.gnn = torch.compile(model.gnn, mode="reduce-overhead")
        #         if fold_i == 0:
        #             log.info("  torch.compile() applied to GNN")
        #     except Exception as e:
        #         log.warning(f"  torch.compile() failed, continuing without: {e}")

        # ── Training loop ──
        best_val_metric = -1.0
        best_epoch = 0
        patience_counter = 0
        train_losses = []
        val_metrics_history = []
        model_path = run_dir / f"model_fold{fold_i}.pt"

        # Effective batch size: 0 means all archetypes in one step
        eff_batch = hp["batch_size"] if hp["batch_size"] > 0 else len(train_archs)
        val_every = hp["val_every"]

        # Color-balanced batching setup
        color_groups = None
        if hp["batch_strategy"] == "color_balanced" and eff_batch < len(train_archs):
            color_groups = _get_archetype_color_groups(data, train_archs)
            if fold_i == 0:
                log.info(f"  Color-balanced batching: {', '.join(f'{c}={len(v)}' for c, v in color_groups.items())}")

        # Inverse color-frequency loss weighting
        color_weights: dict[int, float] = {}
        if hp["color_loss_weighting"]:
            color_weights = _compute_color_loss_weights(data, train_archs)

        log.info(f"\n{'Epoch':>5} {'Loss':>8} {'SelL':>8} {'CntL':>8} "
                 f"{'Prec':>8} {'Rec':>8} {'Jacc':>8} "
                 f"{'ExCnt':>8} {'MAE':>8} {'LR':>10}")
        log.info("-" * 104)

        for epoch in range(1, hp["num_epochs"] + 1):
            model.train()

            # Scheduled sampling: anneal teacher forcing rate
            tf_rate = _anneal_value(
                epoch, hp["num_epochs"],
                hp["scheduled_sampling_start"],
                hp["scheduled_sampling_end"],
                hp["scheduled_sampling_schedule"],
            )

            # Temperature annealing on card scorer
            temperature = _anneal_value(
                epoch, hp["num_epochs"],
                hp["scorer_temp_start"],
                hp["scorer_temp_end"],
                hp["scorer_temp_schedule"],
            )

            if epoch == 1:
                log.info(f"  Scheduled sampling: tf_rate={tf_rate:.3f}, temperature={temperature:.3f}")

            # Curriculum mask: restrict decoder's candidate pool early, grow to full
            curriculum_mask = _compute_curriculum_mask(
                epoch=epoch,
                deck_card_set=deck_card_set,
                negative_cards=negative_cards,
                total_cards=n_total_cards,
                neg_start=hp["curriculum_neg_start"],
                neg_epochs=hp["curriculum_neg_epochs"],
                schedule=hp["curriculum_neg_schedule"],
            ).to(device)
            n_eligible = int(curriculum_mask.sum().item())
            if epoch == 1 or epoch == hp["curriculum_neg_epochs"]:
                log.info(f"  Epoch {epoch}: curriculum pool = {n_eligible}/{n_total_cards} cards")

            # Unfreeze GNN after freeze period in finetune mode
            if gnn_freeze_epochs > 0 and epoch == gnn_freeze_epochs + 1:
                log.info(f"  Unfreezing GNN at epoch {epoch}")
                for p in gnn_params:
                    p.requires_grad = True

            epoch_loss = 0.0
            epoch_select_loss = 0.0
            epoch_count_loss = 0.0
            epoch_correct = 0
            epoch_steps = 0

            # Build batches (only archetypes with ground truth)
            epoch_archs = [a for a in train_archs if ground_truth.get(a)]

            if color_groups is not None:
                # Color-balanced batching: greedy color coverage per batch
                batches = _color_balanced_batches(
                    color_groups, eff_batch, epoch_archs, epoch,
                )
            else:
                # Random batching (default)
                random.shuffle(epoch_archs)
                batches = [epoch_archs[i:i + eff_batch]
                           for i in range(0, len(epoch_archs), eff_batch)]

            n_archs = sum(len(b) for b in batches)

            # Process archetypes in batches using detached GNN embeddings.
            #
            # Memory optimization: instead of keeping the GNN computation
            # graph alive across all archetype backward passes (via
            # retain_graph=True), we:
            #   1. Run GNN forward → get x_dict (with grad graph)
            #   2. Detach x_dict into leaf tensors that collect decoder grads
            #   3. Run decoder forward+backward for batch archetypes
            #      (no retain_graph needed — decoder graph is independent)
            #   4. Backprop accumulated decoder grads through GNN
            #   5. Optimizer step
            #
            # This frees GNN activations before the decoder runs, cutting
            # peak memory from ~128 GB to ~51 GB with gradient checkpointing.

            for batch_archs in batches:
                batch_size = len(batch_archs)

                optimizer.zero_grad()

                # Step 1: GNN forward (builds computation graph)
                if device.type == "cuda" and epoch == 1 and batch_archs is batches[0]:
                    torch.cuda.reset_peak_memory_stats()
                    log.info(f"  GPU before GNN: {torch.cuda.memory_allocated()/1e9:.1f} GB")
                with torch.amp.autocast("cuda", enabled=use_amp):
                    x_dict = model.gnn(data)
                if device.type == "cuda" and epoch == 1 and batch_archs is batches[0]:
                    log.info(f"  GPU after GNN:  {torch.cuda.memory_allocated()/1e9:.1f} GB "
                             f"(peak: {torch.cuda.max_memory_allocated()/1e9:.1f} GB) "
                             f"ckpt={model.gnn.gradient_checkpointing}")

                # Step 2: Detach — x_dict_d are leaf tensors that collect
                # decoder gradients without keeping the GNN graph alive
                x_dict_d = {
                    nt: x_dict[nt].detach().requires_grad_(True)
                    for nt in x_dict
                }

                # Step 3: Decoder forward + backward for each archetype
                for a_idx in batch_archs:
                    gt = ground_truth[a_idx]

                    with torch.amp.autocast("cuda", enabled=use_amp):
                        result = model.forward_with_embeddings(
                            x_dict_d, a_idx, greedy=False, target_deck=gt,
                            curriculum_mask=curriculum_mask,
                            target_order=target_orders.get(a_idx),
                            teacher_forcing_rate=tf_rate,
                            temperature=temperature,
                        )
                        loss, metrics = compute_step_losses(
                            result["steps"], gt, hp["count_loss_weight"],
                            inclusion_rates=inclusion_rates.get(a_idx) if hp["consensus_loss_weighting"] else None,
                            consensus_min_weight=hp["consensus_loss_min_weight"],
                        )

                    # Decoder backward — grads accumulate in x_dict_d[nt].grad
                    # and in decoder parameters. No retain_graph needed.
                    arch_weight = color_weights.get(a_idx, 1.0)
                    scaler.scale(loss * arch_weight / batch_size).backward()

                    epoch_loss += metrics["total_loss"]
                    epoch_select_loss += metrics["select_loss"]
                    epoch_count_loss += metrics["count_loss"]
                    epoch_correct += metrics["correct_cards"]
                    epoch_steps += metrics["total_steps"]

                # Step 4: Backprop accumulated decoder grads through GNN.
                # Extract gradients from x_dict_d, then free it and reclaim
                # fragmented GPU memory before the GNN backward.  The GNN
                # backward (with gradient checkpointing) must recompute the
                # forward pass, which temporarily recreates large message
                # tensors — freeing decoder state first gives it headroom.
                gnn_tensors = []
                gnn_grads = []
                for nt in x_dict:
                    gnn_tensors.append(x_dict[nt])
                    grad = x_dict_d[nt].grad
                    if grad is not None:
                        gnn_grads.append(grad.clone())
                    else:
                        gnn_grads.append(torch.zeros_like(x_dict[nt]))
                del x_dict_d
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                torch.autograd.backward(gnn_tensors, grad_tensors=gnn_grads)
                del gnn_grads

                # Step 5: Optimizer step
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            scheduler.step()

            n_train = max(n_archs, 1)  # accounts for oversampling in color_balanced mode
            train_losses.append({
                "loss": epoch_loss / n_train,
                "select_loss": epoch_select_loss / n_train,
                "count_loss": epoch_count_loss / n_train,
                "correct_cards": epoch_correct,
                "total_steps": epoch_steps,
            })

            # ── Logging ──
            lr = scheduler.get_last_lr()[0]
            do_val = (epoch % val_every == 0 or epoch == 1)

            if do_val:
                val_metrics = evaluate_fold(model, data, ground_truth, val_archs)
                val_metrics_history.append({"epoch": epoch, **val_metrics})

                log.info(
                    f"{epoch:5d} {epoch_loss / n_train:8.4f} "
                    f"{epoch_select_loss / n_train:8.4f} "
                    f"{epoch_count_loss / n_train:8.4f} "
                    f"{val_metrics['precision']:8.3f} "
                    f"{val_metrics['recall']:8.3f} "
                    f"{val_metrics['jaccard']:8.3f} "
                    f"{val_metrics['exact_count_match']:8.3f} "
                    f"{val_metrics['count_mae']:8.4f} "
                    f"{lr:10.6f}"
                )
            else:
                log.info(
                    f"{epoch:5d} {epoch_loss / n_train:8.4f} "
                    f"{epoch_select_loss / n_train:8.4f} "
                    f"{epoch_count_loss / n_train:8.4f} "
                    f"{'---':>8} {'---':>8} {'---':>8} "
                    f"{'---':>8} {'---':>8} "
                    f"{lr:10.6f}"
                )

            # ── Model saving + early stopping (only on validation epochs) ──
            if do_val:
                current_metric = val_metrics["jaccard"]

                if current_metric > best_val_metric:
                    best_val_metric = current_metric
                    best_epoch = epoch
                    patience_counter = 0
                    torch.save({
                        "epoch": epoch,
                        "fold": fold_i,
                        "model_state_dict": {
                            k.replace("_orig_mod.", ""): v
                            for k, v in model.state_dict().items()
                        },
                        "val_metrics": val_metrics,
                        "hyperparameters": hp,
                    }, model_path)
                else:
                    patience_counter += 1

                # Optuna pruning
                if trial is not None:
                    trial.report(current_metric, epoch)
                    if trial.should_prune():
                        log.info(f"\nOptuna pruned at epoch {epoch}")
                        try:
                            import optuna
                            raise optuna.TrialPruned()
                        except ImportError:
                            break

                if patience_counter >= hp["patience"]:
                    log.info(f"\nEarly stopping at epoch {epoch} "
                             f"(no improvement for {hp['patience']} eval rounds)")
                    break

        log.info(f"\n{fold_label}: Best epoch {best_epoch}, jaccard={best_val_metric:.4f}")

        all_fold_results.append({
            "fold": fold_i,
            "train_archetypes": [arch_names[i] for i in train_archs],
            "val_archetypes": [arch_names[i] for i in val_archs],
            "best_epoch": best_epoch,
            "best_val_metric": best_val_metric,
            "train_losses": train_losses,
            "val_metrics_history": val_metrics_history,
            "model_path": str(model_path),
        })

        # Free GPU memory before next fold (keep last fold's model for return)
        if fold_i < len(folds_to_run) - 1:
            del model, optimizer, scheduler, scaler
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    # ── Summary across folds ──
    if len(all_fold_results) > 1:
        jaccard_scores = [r["best_val_metric"] for r in all_fold_results]
        log.info(f"\n{'='*60}")
        log.info(f"Cross-Validation Summary ({len(all_fold_results)} folds):")
        log.info(f"  Jaccard: {np.mean(jaccard_scores):.4f} +/- {np.std(jaccard_scores):.4f}")
        for r in all_fold_results:
            log.info(f"  Fold {r['fold']}: Jaccard={r['best_val_metric']:.4f} "
                     f"(val: {', '.join(r['val_archetypes'])})")
        log.info(f"{'='*60}")

    # ── Save training log ──
    training_log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task": "autoregressive_deck_constructor",
        "mode": mode,
        "format_filter": format_filter,
        "architecture": "HeteroGNN + CrossAttention DeckConstructor",
        "hyperparameters": hp,
        "graph_info": {
            "node_types": {nt: data[nt].x.shape[0] for nt in data.node_types},
            "edge_types": len(data.edge_types),
        },
        "curriculum": {
            "neg_start": hp["curriculum_neg_start"],
            "neg_epochs": hp["curriculum_neg_epochs"],
            "schedule": hp["curriculum_neg_schedule"],
            "total_deck_cards": len(deck_card_set),
            "total_negative_cards": len(negative_cards),
        },
        "fold_results": all_fold_results,
        "run_dir": str(run_dir),
    }

    log_path = run_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    log.info(f"\nTraining log saved to {log_path}")

    return {
        "model": model,
        "data": data,
        "hp": hp,
        "ground_truth": ground_truth,
        "inclusion_rates": inclusion_rates,
        "fold_results": all_fold_results,
        "run_dir": run_dir,
    }


# ════════════════════════════════════════════════════════════════════
#  Detailed Evaluation
# ════════════════════════════════════════════════════════════════════


def evaluate_deck(result: dict):
    """Load best model from each fold and print detailed per-archetype results."""
    data = result["data"]
    ground_truth = result["ground_truth"]
    arch_names = data["archetype"].names
    card_names = data["card"].names
    hp = result["hp"]

    log.info("\n" + "=" * 60)
    log.info("Final Evaluation — Autoregressive Deck Constructor")
    log.info("=" * 60)

    for fold_result in result["fold_results"]:
        fold_i = fold_result["fold"]
        model_path = fold_result["model_path"]
        val_arch_names = fold_result["val_archetypes"]

        log.info(f"\nFold {fold_i}: Held-out archetypes = {val_arch_names}")

        # Re-initialize and load best model
        node_dims = {nt: data[nt].x.shape[1] for nt in data.node_types}
        # Extract card metadata for budget signal (same as training)
        _eval_color = None
        _eval_type = None
        if hp.get("budget_signal", False):
            emb_dim = 384
            _eval_type = data["card"].x[:, emb_dim + 1 : emb_dim + 8].clone()
            _eval_color = data["card"].x[:, emb_dim + 10 : emb_dim + 15].clone()
        model = MTGDeckModel(
            node_dims=node_dims,
            edge_types=list(data.edge_types),
            node_types=list(data.node_types),
            card_names=card_names,
            d_model=hp["d_model"],
            d_message=hp["d_message"],
            d_count=hp["d_count"],
            num_gnn_layers=hp["num_gnn_layers"],
            dropout=hp["dropout"],
            edge_chunk_size=hp.get("edge_chunk_size", 0),
            card_color_flags=_eval_color,
            card_type_flags=_eval_type,
        ).to(data["card"].x.device)

        checkpoint = torch.load(model_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Cache GNN embeddings once for all archetypes in this fold
        eval_amp = data["card"].x.device.type == "cuda"
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=eval_amp):
            x_dict = model.gnn(data)

        arch_to_idx = {a: i for i, a in enumerate(arch_names)}

        for arch_name in val_arch_names:
            a_idx = arch_to_idx[arch_name]
            gt = ground_truth.get(a_idx, {})
            if not gt:
                continue

            with torch.no_grad(), torch.amp.autocast("cuda", enabled=eval_amp):
                pred = model.predict_deck(data, a_idx, x_dict=x_dict)
            gt_set = set(gt.keys())
            pred_set = set(pred.keys())
            common = gt_set & pred_set

            log.info(f"\n  {arch_name}:")
            log.info(f"  {'#':>3} {'Card Name':<40} {'Pred':>4} {'GT':>4} {'Match':>5}")
            log.info(f"  {'-'*60}")

            # Show predicted deck in selection order
            for rank, (card_idx, count) in enumerate(pred.items(), 1):
                name = card_names[card_idx]
                gt_count = gt.get(card_idx, "-")
                match = "Y" if card_idx in common and pred[card_idx] == gt.get(card_idx) else ""
                if card_idx in common and pred[card_idx] != gt.get(card_idx):
                    match = "~"  # right card, wrong count
                log.info(f"  {rank:3d} {name:<40} x{count:<3d} {str(gt_count):>4} {match:>5}")

            # Summary
            precision = len(common) / max(len(pred_set), 1)
            recall = len(common) / max(len(gt_set), 1)
            jaccard = len(common) / max(len(gt_set | pred_set), 1)
            total = sum(pred.values())

            exact = sum(1 for c in common if pred[c] == gt[c]) if common else 0
            log.info(f"\n  Precision={precision:.3f} Recall={recall:.3f} Jaccard={jaccard:.3f}")
            log.info(f"  Exact count matches: {exact}/{len(common)}")
            log.info(f"  Total cards: {total} (budget={'OK' if total == 60 else 'FAIL'})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MTG deck predictor")
    parser.add_argument("--mode", choices=["pretrain", "finetune"], default="pretrain",
                        help="pretrain: all formats; finetune: target format only")
    parser.add_argument("--format-filter", type=str, default=None,
                        help="Target format for finetune mode. Options: 'standard', 'pioneer'")
    parser.add_argument("--pretrained-path", type=str, default=None,
                        help="Path to pretrained model checkpoint for finetune mode")
    args = parser.parse_args()

    overrides = {
        "mode": args.mode,
        "format_filter": args.format_filter,
        "pretrained_path": args.pretrained_path,
    }
    if args.mode == "finetune":
        overrides["num_epochs"] = TRANSFER_FINETUNE_EPOCHS

    result = train_deck(**overrides)
    evaluate_deck(result)
