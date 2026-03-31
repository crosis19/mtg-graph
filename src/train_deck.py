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
    CARDS_PARQUET,
    COUNT_LOSS_WEIGHT,
    D_COUNT,
    D_MESSAGE,
    D_MODEL,
    DECK_BATCH_SIZE,
    DECK_LEARNING_RATE,
    DECK_LR_SCHEDULER,
    DECK_NUM_CV_FOLDS,
    DECK_NUM_EPOCHS,
    DECK_NUM_VAL_ARCHETYPES,
    DECK_PATIENCE,
    DECK_RECENCY_DAYS,
    DECK_VAL_EVERY,
    DECK_WARMUP_EPOCHS,
    DECKLISTS_PARQUET,
    DROPOUT,
    GRAPH_PATH,
    MAX_BASIC_LAND_COUNT,
    MAX_NONBASIC_COUNT,
    NUM_ATTN_HEADS,
    NUM_GNN_LAYERS,
    WEIGHT_DECAY,
    create_run_dir,
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
) -> dict[int, dict[int, int]]:
    """Build per-archetype ground truth with raw integer copy counts.

    Uses only the latest snapshot per archetype so counts reflect the
    current meta. Returns {arch_idx: {card_idx: count}} where counts
    sum to 60. Non-basic cards are clamped to [1, 4], basic lands to
    [1, 20].
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

    # Filter to mainboard only
    if "board" in decklists.columns:
        decklists = decklists[decklists["board"] == "main"]

    # Use only the latest snapshot per archetype so counts reflect the
    # current meta rather than accumulating historical maximums.
    if "snapshot_date" in decklists.columns:
        latest = decklists.groupby("archetype")["snapshot_date"].transform("max")
        decklists = decklists[decklists["snapshot_date"] == latest]

    # One row per (archetype, card) — no cross-snapshot aggregation needed
    agg = (
        decklists.groupby(["archetype", "card_name"])[copies_col]
        .first()
        .reset_index()
    )

    ground_truth: dict[int, dict[int, int]] = {}

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

    # Log summary
    for a_idx, cards in ground_truth.items():
        total = sum(cards.values())
        log.info(f"  Archetype {arch_names[a_idx]}: {len(cards)} unique cards, {total} total copies")

    return ground_truth


# ════════════════════════════════════════════════════════════════════
#  Temperature Schedule
# ════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════
#  Loss Computation
# ════════════════════════════════════════════════════════════════════


def compute_step_losses(
    steps: list[dict],
    target_deck: dict[int, int],
    count_loss_weight: float,
) -> tuple[torch.Tensor, dict]:
    """Compute per-step card selection and count prediction losses.

    For each autoregressive step:
      - If the selected card IS in the target deck:
          L_select = -log(p(selected_card))
          L_count  = CrossEntropy(count_logits, ground_truth_count - 1)
      - If the selected card is NOT in the target deck:
          L_select = -log(sum of probs on all remaining target cards)
          L_count  = 0 (masked)

    Args:
        steps: per-step dicts from DeckConstructor.forward().
        target_deck: {card_idx: count} ground truth.
        count_loss_weight: lambda weighting for count loss.

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
            total_select_loss = total_select_loss + (-torch.log(prob))
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
        "num_attn_heads": overrides.get("num_attn_heads", NUM_ATTN_HEADS),
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
    }

    results_base = overrides.get("results_base", None)
    run_dir = create_run_dir(task="deck", results_base=results_base)
    log.info(f"Results will be saved to {run_dir}")
    log.info(f"Hyperparameters: {hp}")

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # ── Load graph ──
    log.info("Loading graph...")
    data = torch.load(GRAPH_PATH, weights_only=False)
    log.info(f"  Nodes: {', '.join(f'{nt}={data[nt].x.shape[0]}' for nt in data.node_types)}")
    log.info(f"  Edge types: {len(data.edge_types)}")

    # ── Load and filter decklists ──
    decklists = pd.read_parquet(DECKLISTS_PARQUET)
    log.info(f"\nFiltering decklists to last {hp['recency_days']} days...")
    decklists = _filter_recent_decklists(decklists, hp["recency_days"])

    # ── Build ground truth ──
    log.info("\nBuilding ground truth (raw integer counts)...")
    ground_truth = _build_ground_truth(data, decklists)

    # ── Move to device ──
    data = data.to(device)
    card_names = data["card"].names
    arch_names = data["archetype"].names

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
            num_attn_heads=hp["num_attn_heads"],
            dropout=hp["dropout"],
        ).to(device)

        if fold_i == 0:
            total_params = sum(p.numel() for p in model.parameters())
            log.info(f"Model parameters: {total_params:,}")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=hp["learning_rate"],
            weight_decay=hp["weight_decay"],
        )

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

        # ── Mixed precision & torch.compile ──
        use_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        if device.type == "cuda" and hasattr(torch, "compile"):
            try:
                model.gnn = torch.compile(model.gnn, mode="reduce-overhead")
                if fold_i == 0:
                    log.info("  torch.compile() applied to GNN")
            except Exception as e:
                log.warning(f"  torch.compile() failed, continuing without: {e}")

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

        log.info(f"\n{'Epoch':>5} {'Loss':>8} {'SelL':>8} {'CntL':>8} "
                 f"{'Prec':>8} {'Rec':>8} {'Jacc':>8} "
                 f"{'ExCnt':>8} {'MAE':>8} {'LR':>10}")
        log.info("-" * 104)

        for epoch in range(1, hp["num_epochs"] + 1):
            model.train()

            epoch_loss = 0.0
            epoch_select_loss = 0.0
            epoch_count_loss = 0.0
            epoch_correct = 0
            epoch_steps = 0

            # Shuffle archetypes for stochastic batching (only those with GT)
            epoch_archs = [a for a in train_archs if ground_truth.get(a)]
            random.shuffle(epoch_archs)
            n_archs = len(epoch_archs)

            # Process archetypes in batches
            optimizer.zero_grad()
            batch_count = 0

            # Cache GNN embeddings (recomputed after each optimizer step).
            # Multiple backward() calls through the same cached x_dict require
            # retain_graph=True on all but the last call in each batch.
            with torch.amp.autocast("cuda", enabled=use_amp):
                x_dict = model.gnn(data)

            for arch_i, a_idx in enumerate(epoch_archs):
                gt = ground_truth[a_idx]

                with torch.amp.autocast("cuda", enabled=use_amp):
                    result = model.forward_with_embeddings(
                        x_dict, a_idx, greedy=False, target_deck=gt,
                    )
                    loss, metrics = compute_step_losses(
                        result["steps"], gt, hp["count_loss_weight"],
                    )

                # Determine if this is the last backward through the current
                # GNN graph: either the batch is about to be full, or we've
                # reached the last archetype in the epoch.
                batch_count += 1
                is_last_in_batch = (batch_count >= eff_batch) or (arch_i == n_archs - 1)

                # Scale loss by effective batch size for consistent gradient magnitude.
                # retain_graph=True keeps the GNN computation graph alive for
                # subsequent archetypes sharing the same cached x_dict.
                scaler.scale(loss / eff_batch).backward(retain_graph=not is_last_in_batch)

                epoch_loss += metrics["total_loss"]
                epoch_select_loss += metrics["select_loss"]
                epoch_count_loss += metrics["count_loss"]
                epoch_correct += metrics["correct_cards"]
                epoch_steps += metrics["total_steps"]

                # Step when batch is full
                if batch_count >= eff_batch:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    batch_count = 0
                    # Recompute GNN embeddings since weights changed
                    if arch_i < n_archs - 1:
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            x_dict = model.gnn(data)

            # Handle final partial batch
            if batch_count > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            scheduler.step()

            n_train = max(len(train_archs), 1)
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
        "architecture": "HeteroGNN + CrossAttention DeckConstructor",
        "hyperparameters": hp,
        "graph_info": {
            "node_types": {nt: data[nt].x.shape[0] for nt in data.node_types},
            "edge_types": len(data.edge_types),
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
        model = MTGDeckModel(
            node_dims=node_dims,
            edge_types=list(data.edge_types),
            node_types=list(data.node_types),
            card_names=card_names,
            d_model=hp["d_model"],
            d_message=hp["d_message"],
            d_count=hp["d_count"],
            num_gnn_layers=hp["num_gnn_layers"],
            num_attn_heads=hp["num_attn_heads"],
            dropout=hp["dropout"],
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
    result = train_deck()
    evaluate_deck(result)
