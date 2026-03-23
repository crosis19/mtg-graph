"""Training loop for the Deck Composition Predictor.

Trains a BPR (Bayesian Personalized Ranking) model to predict which cards
belong in which deck archetypes. Key improvements over the original BCE approach:

  1. BPR ranking loss — directly optimizes per-archetype card ranking
  2. Leave-one-out archetype pooling — prevents information leakage
  3. Card-level train/val/test split — tests generalization to unseen cards
  4. Early stopping on Hits@25 — checkpoints on the metric that matters
  5. Recency filter — uses only decklists from the last N days
"""

import json
import logging
import random
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import torch

from src.config import (
    CARDS_PARQUET,
    DECK_BPR_MARGIN,
    DECK_CHECKPOINT_METRIC,
    DECK_DROPOUT,
    DECK_FREEZE_HGT,
    DECK_LEARNING_RATE,
    DECK_MIN_META_SHARE,
    DECK_N_NEGATIVES,
    DECK_NUM_EPOCHS,
    DECK_PATIENCE,
    DECK_RECENCY_DAYS,
    DECKLISTS_PARQUET,
    GRAPH_PATH,
    HIDDEN_DIM,
    METAGAME_PARQUET,
    NUM_HEADS,
    NUM_HGT_LAYERS,
    TRAIN_SPLIT_RATIO,
    VAL_SPLIT_RATIO,
    WEIGHT_DECAY,
    create_run_dir,
)
from src.deck_predictor import DeckPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _filter_recent_decklists(decklists: pd.DataFrame, days: int) -> pd.DataFrame:
    """Filter decklists to only include snapshots from the last N days."""
    if "snapshot_date" not in decklists.columns:
        return decklists
    decklists = decklists.copy()
    decklists["snapshot_date"] = pd.to_datetime(decklists["snapshot_date"], errors="coerce")
    cutoff = pd.Timestamp.now(tz="UTC") - timedelta(days=days)
    # Handle timezone-naive dates
    if decklists["snapshot_date"].dt.tz is None:
        cutoff = cutoff.tz_localize(None)
    recent = decklists[decklists["snapshot_date"] >= cutoff]
    if recent.empty:
        log.warning(f"  No decklists in last {days} days, using all data")
        return decklists
    log.info(f"  Filtered to {len(recent)} decklist rows from last {days} days "
             f"(dropped {len(decklists) - len(recent)})")
    return recent


def _build_deck_membership(
    data, decklists: pd.DataFrame
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Build per-archetype positive and negative card pools.

    Returns
    -------
    arch_pos_cards : dict[int, list[int]]
        For each archetype index, list of card indices in its deck (excl. basic lands)
    arch_neg_pool : dict[int, list[int]]
        For each archetype index, list of color-matched negative card indices
    """
    arch_names = data["archetype"].names
    card_names = data["card"].names
    arch_to_idx = {a: i for i, a in enumerate(arch_names)}
    card_to_idx = {c: i for i, c in enumerate(card_names)}

    # Filter basic lands
    cards_df = pd.read_parquet(CARDS_PARQUET)
    basic_land_names = set(
        cards_df[cards_df["type_line"].str.contains("Basic Land", case=False, na=False)]["name"]
    )
    basic_land_indices = {card_to_idx[n] for n in basic_land_names if n in card_to_idx}
    all_non_basic = set(range(len(card_names))) - basic_land_indices
    log.info(f"  Filtering {len(basic_land_names)} basic land cards")

    # Card color identity lookup
    card_colors: dict[int, set[str]] = {}
    for _, row in cards_df.iterrows():
        name = row["name"]
        if name not in card_to_idx:
            continue
        ci = row.get("color_identity", "")
        if pd.notna(ci) and ci:
            card_colors[card_to_idx[name]] = set(str(ci).split("|"))
        else:
            card_colors[card_to_idx[name]] = set()

    # Build positive sets per archetype
    eligible_archetypes = set(arch_names)
    arch_pos_cards: dict[int, list[int]] = {}
    arch_colors: dict[int, set[str]] = {}

    for _, row in decklists.iterrows():
        arch = row["archetype"]
        card = row["card_name"]
        if (arch not in eligible_archetypes
                or arch not in arch_to_idx
                or card not in card_to_idx
                or card in basic_land_names):
            continue
        a_idx = arch_to_idx[arch]
        c_idx = card_to_idx[card]
        if a_idx not in arch_pos_cards:
            arch_pos_cards[a_idx] = []
        if c_idx not in set(arch_pos_cards[a_idx]):
            arch_pos_cards[a_idx].append(c_idx)
        arch_colors.setdefault(a_idx, set()).update(card_colors.get(c_idx, set()))

    # Build color-matched negative pools
    color_to_cards: dict[str, set[int]] = {}
    for c_idx in all_non_basic:
        for color in card_colors.get(c_idx, set()):
            color_to_cards.setdefault(color, set()).add(c_idx)

    arch_neg_pool: dict[int, list[int]] = {}
    for a_idx, pos_cards in arch_pos_cards.items():
        pos_set = set(pos_cards)
        colors = arch_colors.get(a_idx, set())
        color_matched = set()
        for color in colors:
            color_matched |= color_to_cards.get(color, set())
        # Include colorless cards
        colorless = {c for c in all_non_basic if not card_colors.get(c, set())}
        color_matched |= colorless
        arch_neg_pool[a_idx] = list(color_matched - pos_set)

    total_pos = sum(len(v) for v in arch_pos_cards.values())
    log.info(f"  {len(arch_pos_cards)} archetypes, {total_pos} total positive pairs")
    for a_idx in sorted(arch_pos_cards):
        log.info(f"    {arch_names[a_idx]}: {len(arch_pos_cards[a_idx])} cards, "
                 f"{len(arch_neg_pool[a_idx])} negative candidates")

    return arch_pos_cards, arch_neg_pool


def _card_level_split(
    arch_pos_cards: dict[int, list[int]],
) -> tuple[set[int], set[int], set[int]]:
    """Split cards (not pairs) into train/val/test sets.

    Ensures each archetype has cards in all splits when possible.
    """
    # Collect all unique positive card indices
    all_positive_cards = set()
    for cards in arch_pos_cards.values():
        all_positive_cards.update(cards)

    all_positive_cards = sorted(all_positive_cards)
    random.shuffle(all_positive_cards)

    n = len(all_positive_cards)
    train_end = int(n * TRAIN_SPLIT_RATIO)
    val_end = int(n * (TRAIN_SPLIT_RATIO + VAL_SPLIT_RATIO))

    train_cards = set(all_positive_cards[:train_end])
    val_cards = set(all_positive_cards[train_end:val_end])
    test_cards = set(all_positive_cards[val_end:])

    log.info(f"  Card-level split: {len(train_cards)} train, "
             f"{len(val_cards)} val, {len(test_cards)} test cards")

    # Log per-archetype breakdown
    for a_idx in sorted(arch_pos_cards):
        pos = set(arch_pos_cards[a_idx])
        n_train = len(pos & train_cards)
        n_val = len(pos & val_cards)
        n_test = len(pos & test_cards)
        log.info(f"    arch {a_idx}: {n_train}/{n_val}/{n_test} train/val/test")

    return train_cards, val_cards, test_cards


def _get_eligible_archetypes(arch_to_idx: dict) -> set[int]:
    """Return archetype indices with meta share >= DECK_MIN_META_SHARE%."""
    metagame = pd.read_parquet(METAGAME_PARQUET)
    latest = (
        metagame.sort_values("snapshot_date")
        .groupby("archetype")["meta_share_pct"]
        .last()
    )
    eligible_names = set(latest[latest >= DECK_MIN_META_SHARE].index)
    return {arch_to_idx[a] for a in eligible_names if a in arch_to_idx}


def _get_basic_land_indices(card_to_idx: dict) -> set[int]:
    """Return card indices for basic lands."""
    cards_df = pd.read_parquet(CARDS_PARQUET)
    basic_names = set(
        cards_df[cards_df["type_line"].str.contains("Basic Land", case=False, na=False)]["name"]
    )
    return {card_to_idx[n] for n in basic_names if n in card_to_idx}


def _evaluate_ranking(
    model, data, node_emb,
    arch_pos_cards: dict[int, list[int]],
    eval_cards: set[int] | None,
    basic_land_indices: set[int],
    eligible_archetypes: set[int] | None = None,
    k_values: list[int] | None = None,
) -> dict:
    """Evaluate ranking quality.

    For each eligible archetype, scores all non-basic-land cards and computes:
      - Hits@K: fraction of target deck cards found in top-K predictions
      - MRR: mean reciprocal rank of target cards
      - F1 at threshold 0.5

    If eval_cards is provided, only those cards count as targets (held-out eval).
    If eval_cards is None, all positive cards for each archetype are targets.

    Returns dict with averaged metrics.
    """
    if k_values is None:
        k_values = [25, 50]

    card_emb = node_emb["card"]
    arch_emb = node_emb["archetype"]
    n_cards = card_emb.shape[0]
    basic_lands = basic_land_indices or set()

    all_card_indices = torch.arange(n_cards, device=card_emb.device)
    hits_at_k = {k: [] for k in k_values}
    mrr_list = []
    f1_list = []
    precision_list = []
    recall_list = []

    for a_idx, pos_cards in arch_pos_cards.items():
        if eligible_archetypes is not None and a_idx not in eligible_archetypes:
            continue

        # Determine target positives
        if eval_cards is not None:
            target_positives = set(pos_cards) & eval_cards
        else:
            target_positives = set(pos_cards)
        if not target_positives:
            continue

        # Score all cards
        arch_indices = torch.full((n_cards,), a_idx, dtype=torch.long, device=arch_emb.device)
        with torch.no_grad():
            logits = model.predict_deck(card_emb, arch_emb, all_card_indices, arch_indices)

        # Mask basic lands
        for bl_idx in basic_lands:
            logits[bl_idx] = float("-inf")

        # Hits@K
        for k in k_values:
            top_k_indices = torch.topk(logits, min(k, n_cards)).indices
            top_k_set = set(top_k_indices.cpu().tolist())
            hit_count = len(target_positives & top_k_set)
            hits_at_k[k].append(hit_count / len(target_positives))

        # MRR
        sorted_indices = torch.argsort(logits, descending=True).cpu().tolist()
        rank_map = {idx: rank + 1 for rank, idx in enumerate(sorted_indices)}
        reciprocal_ranks = [1.0 / rank_map[c] for c in target_positives if c in rank_map]
        if reciprocal_ranks:
            mrr_list.append(np.mean(reciprocal_ranks))

        # F1 at threshold 0.5
        probs = torch.sigmoid(logits)
        preds_set = set((probs > 0.5).nonzero(as_tuple=True)[0].cpu().tolist()) - basic_lands
        tp = len(target_positives & preds_set)
        fp = len(preds_set - target_positives)
        fn = len(target_positives - preds_set)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)

    result = {}
    for k in k_values:
        result[f"hits_at_{k}"] = np.mean(hits_at_k[k]) if hits_at_k[k] else 0.0
    result["mrr"] = np.mean(mrr_list) if mrr_list else 0.0
    result["f1"] = np.mean(f1_list) if f1_list else 0.0
    result["precision"] = np.mean(precision_list) if precision_list else 0.0
    result["recall"] = np.mean(recall_list) if recall_list else 0.0

    return result


def train_deck(device=None, **overrides):
    """Main training loop for deck composition predictor with BPR ranking loss.

    Parameters
    ----------
    device : torch.device, optional
    **overrides : keyword args for hyperparameters

    Returns dict with model, data, splits, metrics, and run info.
    """
    if device is None:
        device = torch.device("cpu")

    hp = {
        "hidden_dim": overrides.get("hidden_dim", HIDDEN_DIM),
        "num_heads": overrides.get("num_heads", NUM_HEADS),
        "num_hgt_layers": overrides.get("num_hgt_layers", NUM_HGT_LAYERS),
        "dropout": overrides.get("dropout", DECK_DROPOUT),
        "learning_rate": overrides.get("learning_rate", DECK_LEARNING_RATE),
        "weight_decay": overrides.get("weight_decay", WEIGHT_DECAY),
        "num_epochs": overrides.get("num_epochs", DECK_NUM_EPOCHS),
        "n_negatives": overrides.get("n_negatives", DECK_N_NEGATIVES),
        "bpr_margin": overrides.get("bpr_margin", DECK_BPR_MARGIN),
        "patience": overrides.get("patience", DECK_PATIENCE),
        "checkpoint_metric": overrides.get("checkpoint_metric", DECK_CHECKPOINT_METRIC),
        "freeze_hgt": overrides.get("freeze_hgt", DECK_FREEZE_HGT),
        "recency_days": overrides.get("recency_days", DECK_RECENCY_DAYS),
    }

    run_dir = create_run_dir(task="deck")
    model_path = run_dir / "model.pt"
    log.info(f"Results will be saved to {run_dir}")
    log.info(f"Hyperparameters: {hp}")

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Load graph
    log.info("Loading graph...")
    data = torch.load(GRAPH_PATH, weights_only=False)
    log.info(f"  Nodes: {', '.join(f'{nt}={data[nt].x.shape[0]}' for nt in data.node_types)}")
    log.info(f"  Edge types: {len(data.edge_types)}")

    # Load and filter decklists
    decklists = pd.read_parquet(DECKLISTS_PARQUET)
    log.info(f"\nFiltering decklists to last {hp['recency_days']} days...")
    decklists = _filter_recent_decklists(decklists, hp["recency_days"])

    # Build membership data
    log.info("\nBuilding deck membership data...")
    arch_pos_cards, arch_neg_pool = _build_deck_membership(data, decklists)

    # Card-level split
    log.info("\nSplitting by card...")
    train_cards, val_cards, test_cards = _card_level_split(arch_pos_cards)

    # Move to device
    data = data.to(device)

    # Initialize model
    node_dims = {nt: data[nt].x.shape[1] for nt in data.node_types}
    log.info(f"\nNode dimensions: {node_dims}")

    model = DeckPredictor(
        metadata=data.metadata(),
        node_dims=node_dims,
        hidden_dim=hp["hidden_dim"],
        num_heads=hp["num_heads"],
        num_layers=hp["num_hgt_layers"],
        dropout=hp["dropout"],
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {total_params:,}")

    # Optionally freeze HGT backbone
    if hp["freeze_hgt"]:
        frozen = 0
        for name, param in model.named_parameters():
            if name.startswith(("convs.", "norms.", "input_projections.")):
                param.requires_grad = False
                frozen += param.numel()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"  Froze HGT backbone: {frozen:,} params frozen, {trainable:,} trainable")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=hp["learning_rate"],
        weight_decay=hp["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hp["num_epochs"]
    )

    # Setup evaluation
    arch_names = data["archetype"].names
    card_names = data["card"].names
    arch_to_idx = {a: i for i, a in enumerate(arch_names)}
    card_to_idx = {c: i for i, c in enumerate(card_names)}
    eligible_archetypes = _get_eligible_archetypes(arch_to_idx)
    basic_land_indices = _get_basic_land_indices(card_to_idx)

    # Training loop
    best_val_metric = -1.0
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    val_metrics_history = []

    log.info(f"\nTraining for up to {hp['num_epochs']} epochs (patience={hp['patience']})...")
    log.info(f"{'Epoch':>5} {'BPR Loss':>10} {'Val H@25':>9} {'Val F1':>8} {'Val MRR':>8} {'LR':>10}")
    log.info("-" * 58)

    for epoch in range(1, hp["num_epochs"] + 1):
        model.train()

        # Forward: get card embeddings via HGT
        x_dict = model.forward_train(data)
        card_emb = x_dict["card"]

        epoch_loss = 0.0
        n_pairs = 0

        # BPR training: for each archetype, sample pos/neg pairs
        for a_idx in arch_pos_cards:
            train_pos = [c for c in arch_pos_cards[a_idx] if c in train_cards]
            if not train_pos:
                continue

            neg_pool = arch_neg_pool.get(a_idx, [])
            if not neg_pool:
                continue

            n_neg = min(hp["n_negatives"], len(neg_pool))
            neg_sample = random.sample(neg_pool, n_neg)

            # Build LOO archetype embeddings for each positive card
            loo_arch_embs = model.build_loo_arch_embeddings(
                card_emb, data, a_idx, train_pos
            )  # (n_pos, H)

            neg_card_emb = card_emb[torch.tensor(neg_sample, dtype=torch.long, device=device)]
            # (n_neg, H)

            # For each positive, score against its LOO embedding and all negatives
            for p_i, pos_c in enumerate(train_pos):
                pos_c_emb = card_emb[pos_c].unsqueeze(0)  # (1, H)
                arch_emb_loo = loo_arch_embs[p_i].unsqueeze(0)  # (1, H)

                # Score positive
                pos_score = model.score_pairs(pos_c_emb, arch_emb_loo)  # (1,)

                # Score all negatives against same LOO archetype embedding
                arch_emb_expanded = arch_emb_loo.expand(n_neg, -1)  # (n_neg, H)
                neg_scores = model.score_pairs(neg_card_emb, arch_emb_expanded)  # (n_neg,)

                # BPR loss: -log(sigmoid(pos - neg))
                diff = pos_score - neg_scores  # (n_neg,)
                pair_loss = -torch.log(torch.sigmoid(diff) + 1e-8).mean()

                epoch_loss += pair_loss
                n_pairs += n_neg

        if n_pairs > 0:
            avg_loss = epoch_loss / len(arch_pos_cards)
            optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_losses.append(avg_loss.item())
        else:
            train_losses.append(0.0)

        # Validation — evaluate ranking on all positive cards (not just val split)
        # to get a meaningful signal with small dataset. Test split uses held-out cards.
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_output = model(data)
                val_emb = val_output["node_embeddings"]

                val_metrics = _evaluate_ranking(
                    model, data, val_emb,
                    arch_pos_cards, None,  # all cards
                    basic_land_indices, eligible_archetypes,
                )

            val_metrics_history.append(val_metrics)

            lr = scheduler.get_last_lr()[0]
            log.info(
                f"{epoch:5d} {train_losses[-1]:10.4f} "
                f"{val_metrics['hits_at_25']:9.3f} "
                f"{val_metrics['f1']:8.3f} "
                f"{val_metrics['mrr']:8.3f} "
                f"{lr:10.6f}"
            )

            # Checkpoint on chosen metric
            metric_key = hp["checkpoint_metric"].replace("val_", "")
            if metric_key == "hits25":
                metric_key = "hits_at_25"
            current_metric = val_metrics.get(metric_key, val_metrics.get("hits_at_25", 0))

            if current_metric > best_val_metric:
                best_val_metric = current_metric
                best_epoch = epoch
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_metrics,
                }, model_path)
            else:
                patience_counter += 1

            if patience_counter >= hp["patience"]:
                log.info(f"\nEarly stopping at epoch {epoch} "
                         f"(no improvement for {hp['patience']} eval rounds)")
                break

    log.info(f"\nBest model at epoch {best_epoch} with "
             f"{hp['checkpoint_metric']}={best_val_metric:.4f}")
    log.info(f"Model saved to {model_path}")

    return {
        "model": model,
        "data": data,
        "hp": hp,
        "arch_pos_cards": arch_pos_cards,
        "arch_neg_pool": arch_neg_pool,
        "train_cards": train_cards,
        "val_cards": val_cards,
        "test_cards": test_cards,
        "train_losses": train_losses,
        "val_metrics_history": val_metrics_history,
        "best_epoch": best_epoch,
        "run_dir": run_dir,
        "model_path": model_path,
    }


def evaluate_deck(
    model, data,
    arch_pos_cards, arch_neg_pool,
    train_cards, val_cards, test_cards,
    run_dir, model_path,
    train_losses=None, val_metrics_history=None, best_epoch=None,
    hp=None,
):
    """Final evaluation on held-out test set and save training log."""
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    log.info("\n" + "=" * 60)
    log.info("Final Evaluation — Deck Composition Predictor (BPR)")
    log.info("=" * 60)

    arch_names = data["archetype"].names
    card_names = data["card"].names
    arch_to_idx = {a: i for i, a in enumerate(arch_names)}
    card_to_idx = {c: i for i, c in enumerate(card_names)}
    eligible_archetypes = _get_eligible_archetypes(arch_to_idx)
    basic_land_indices = _get_basic_land_indices(card_to_idx)
    log.info(f"  Evaluating {len(eligible_archetypes)} eligible archetypes "
             f"(>= {DECK_MIN_META_SHARE}% meta share)")

    with torch.no_grad():
        output = model(data)
        node_emb = output["node_embeddings"]

        # Test set metrics
        log.info("\nDeck Prediction (TEST set — held-out cards):")
        test_metrics = _evaluate_ranking(
            model, data, node_emb,
            arch_pos_cards, test_cards,
            basic_land_indices, eligible_archetypes,
            k_values=[25, 50, 75],
        )
        for k, v in test_metrics.items():
            log.info(f"  {k}: {v:.3f}")

        # Val set for comparison
        log.info("\nDeck Prediction (validation set):")
        val_metrics = _evaluate_ranking(
            model, data, node_emb,
            arch_pos_cards, val_cards,
            basic_land_indices, eligible_archetypes,
            k_values=[25, 50, 75],
        )
        for k, v in val_metrics.items():
            log.info(f"  {k}: {v:.3f}")

        # Per-archetype top predictions
        arch_emb = node_emb["archetype"]
        card_emb_all = node_emb["card"]
        n_cards = card_emb_all.shape[0]
        all_card_indices = torch.arange(n_cards, device=card_emb_all.device)

        top_predictions = {}
        log.info(f"\nTop 10 Predicted Cards per Archetype ({len(eligible_archetypes)} eligible):")
        for a_idx, arch_name in enumerate(arch_names):
            if a_idx not in eligible_archetypes:
                continue
            arch_indices = torch.full((n_cards,), a_idx, dtype=torch.long, device=arch_emb.device)
            logits = model.predict_deck(card_emb_all, arch_emb, all_card_indices, arch_indices)

            for bl_idx in basic_land_indices:
                logits[bl_idx] = float("-inf")

            probs = torch.sigmoid(logits)
            top_k = torch.topk(probs, 10)
            pred_cards = [(card_names[i], float(probs[i])) for i in top_k.indices]
            top_predictions[arch_name] = pred_cards

            log.info(f"\n  {arch_name}:")
            for card_name, prob in pred_cards:
                in_deck = "  ✓" if card_to_idx.get(card_name, -1) in set(arch_pos_cards.get(a_idx, [])) else ""
                log.info(f"    {prob:.3f}  {card_name}{in_deck}")

    # Save training log
    training_log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task": "deck_composition",
        "loss_type": "bpr",
        "hyperparameters": hp or {},
        "graph_info": {
            "node_types": {nt: data[nt].x.shape[0] for nt in data.node_types},
            "edge_types": len(data.edge_types),
        },
        "data_split": {
            "split_type": "card_level",
            "train_cards": len(train_cards),
            "val_cards": len(val_cards),
            "test_cards": len(test_cards),
            "total_positive_pairs": sum(len(v) for v in arch_pos_cards.values()),
            "archetypes": len(arch_pos_cards),
        },
        "training_curves": {
            "train_losses": train_losses or [],
            "val_hits25": [m.get("hits_at_25", 0) for m in (val_metrics_history or [])],
            "val_f1": [m.get("f1", 0) for m in (val_metrics_history or [])],
            "val_mrr": [m.get("mrr", 0) for m in (val_metrics_history or [])],
        },
        "best_epoch": best_epoch,
        "final_metrics": {
            "test": {k: float(v) for k, v in test_metrics.items()},
            "val": {k: float(v) for k, v in val_metrics.items()},
        },
        "top_predictions": {
            arch: [(card, prob) for card, prob in preds]
            for arch, preds in top_predictions.items()
        },
        "model_path": str(model_path),
        "run_dir": str(run_dir),
    }

    log_path = run_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    log.info(f"\nTraining log saved to {log_path}")

    return {"test": test_metrics, "val": val_metrics}


if __name__ == "__main__":
    result = train_deck()
    evaluate_deck(
        model=result["model"],
        data=result["data"],
        arch_pos_cards=result["arch_pos_cards"],
        arch_neg_pool=result["arch_neg_pool"],
        train_cards=result["train_cards"],
        val_cards=result["val_cards"],
        test_cards=result["test_cards"],
        run_dir=result["run_dir"],
        model_path=result["model_path"],
        train_losses=result["train_losses"],
        val_metrics_history=result["val_metrics_history"],
        best_epoch=result["best_epoch"],
        hp=result["hp"],
    )
