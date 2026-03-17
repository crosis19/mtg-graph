"""Phase 4: Training loop for the MTG Metagame HGT.

Trains both heads jointly:
  - Head 1 (Emergence): MSE loss on predicted vs actual meta share changes
  - Head 2 (Top 8): BCE loss on archetype-tournament placement prediction

Uses temporal 3-way split: train (60%) / val (20%) / test (20%).
Val is used for checkpoint selection; test is evaluated only once at the end.
"""

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from src.config import (
    DECKLISTS_PARQUET,
    DROPOUT,
    EMERGENCE_LOSS_WEIGHT,
    GRAPH_PATH,
    HIDDEN_DIM,
    LEARNING_RATE,
    METAGAME_PARQUET,
    MODEL_PATH,
    NUM_EPOCHS,
    NUM_HEADS,
    NUM_HGT_LAYERS,
    TOP8_LOSS_WEIGHT,
    TOURNAMENTS_PARQUET,
    TRAIN_SPLIT_RATIO,
    VAL_SPLIT_RATIO,
    WEIGHT_DECAY,
)
from src.model import MTGMetagameHGT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _build_emergence_targets(data: HeteroData) -> torch.Tensor:
    """Build regression targets for archetype emergence.

    Target = change in meta share from average to latest snapshot.
    Positive = growing archetype, negative = shrinking.
    """
    metagame = pd.read_parquet(METAGAME_PARQUET)
    arch_names = data["archetype"].names
    n_archetypes = len(arch_names)

    targets = torch.zeros(n_archetypes)
    for i, arch in enumerate(arch_names):
        arch_data = metagame[metagame["archetype"] == arch].sort_values("snapshot_date")
        if len(arch_data) < 2:
            continue
        # Target: difference between last snapshot and overall mean
        mean_share = arch_data["meta_share_pct"].mean()
        last_share = arch_data["meta_share_pct"].iloc[-1]
        targets[i] = (last_share - mean_share) / 100.0  # Normalize

    return targets


def _build_top8_samples(data: HeteroData) -> tuple:
    """Build positive and negative samples for top-8 link prediction.

    Returns (arch_indices, tourney_indices, labels) tensors.
    """
    decklists = pd.read_parquet(DECKLISTS_PARQUET)
    tournaments = pd.read_parquet(TOURNAMENTS_PARQUET)

    arch_names = data["archetype"].names
    event_ids = data["tournament"].event_ids
    arch_to_idx = {a: i for i, a in enumerate(arch_names)}
    event_to_idx = {e: i for i, e in enumerate(event_ids)}
    n_arch = len(arch_names)
    n_tourney = len(event_ids)

    # Positive samples: actual top-8 placements
    positives = set()
    top8 = decklists[["event_id", "archetype"]].drop_duplicates()
    for _, row in top8.iterrows():
        if row["archetype"] in arch_to_idx and row["event_id"] in event_to_idx:
            positives.add((arch_to_idx[row["archetype"]], event_to_idx[row["event_id"]]))

    arch_list, tourney_list, label_list = [], [], []

    # Add positives
    for a_idx, t_idx in positives:
        arch_list.append(a_idx)
        tourney_list.append(t_idx)
        label_list.append(1.0)

    # Negative samples: archetypes that did NOT place in a tournament
    # Sample roughly equal number of negatives
    all_pairs = set()
    for a in range(n_arch):
        for t in range(n_tourney):
            all_pairs.add((a, t))
    negatives = list(all_pairs - positives)
    random.shuffle(negatives)
    n_neg = min(len(negatives), len(positives) * 2)  # 2:1 neg:pos ratio
    for a_idx, t_idx in negatives[:n_neg]:
        arch_list.append(a_idx)
        tourney_list.append(t_idx)
        label_list.append(0.0)

    return (
        torch.tensor(arch_list, dtype=torch.long),
        torch.tensor(tourney_list, dtype=torch.long),
        torch.tensor(label_list, dtype=torch.float32),
    )


def _temporal_split(
    tourney_indices: torch.Tensor,
    n_tournaments: int,
) -> tuple:
    """Split samples by temporal order of tournaments into train/val/test.

    Returns (train_mask, val_mask, test_mask) boolean tensors.
    """
    train_end = int(n_tournaments * TRAIN_SPLIT_RATIO)
    val_end = int(n_tournaments * (TRAIN_SPLIT_RATIO + VAL_SPLIT_RATIO))
    train_mask = tourney_indices < train_end
    val_mask = (tourney_indices >= train_end) & (tourney_indices < val_end)
    test_mask = tourney_indices >= val_end
    return train_mask, val_mask, test_mask


def _evaluate_top8(model, data, node_emb, arch_idx, tourney_idx, labels, mask, criterion):
    """Evaluate top8 head on a given split. Returns (loss, accuracy, precision, recall, f1)."""
    probs = model.predict_top8(
        node_emb["archetype"],
        node_emb["tournament"],
        arch_idx[mask],
        tourney_idx[mask],
    )
    loss = criterion(probs, labels[mask])
    preds = (probs > 0.5).float()
    split_labels = labels[mask]

    accuracy = (preds == split_labels).float().mean()
    if split_labels.sum() > 0:
        precision = (preds * split_labels).sum() / max(preds.sum(), 1)
        recall = (preds * split_labels).sum() / split_labels.sum()
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    else:
        precision = recall = f1 = torch.tensor(0.0)

    return loss, accuracy, precision, recall, f1


def train():
    """Main training loop."""
    # Seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Load graph
    log.info("Loading graph...")
    data = torch.load(GRAPH_PATH, weights_only=False)
    log.info(f"  Nodes: {', '.join(f'{nt}={data[nt].x.shape[0]}' for nt in data.node_types)}")
    log.info(f"  Edge types: {len(data.edge_types)}")

    # Build targets
    log.info("\nBuilding targets...")
    emergence_targets = _build_emergence_targets(data)
    arch_idx, tourney_idx, top8_labels = _build_top8_samples(data)
    log.info(f"  Emergence targets: {emergence_targets.shape}")
    log.info(f"  Top8 samples: {len(top8_labels)} ({top8_labels.sum().int()} pos, "
             f"{(1 - top8_labels).sum().int()} neg)")

    # Temporal 3-way split for top8 task
    train_mask, val_mask, test_mask = _temporal_split(
        tourney_idx, data["tournament"].x.shape[0]
    )
    log.info(f"  Train samples: {train_mask.sum()}, Val samples: {val_mask.sum()}, "
             f"Test samples: {test_mask.sum()}")

    # Node dimensions
    node_dims = {nt: data[nt].x.shape[1] for nt in data.node_types}
    log.info(f"\nNode dimensions: {node_dims}")

    # Initialize model
    model = MTGMetagameHGT(
        metadata=data.metadata(),
        node_dims=node_dims,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_HGT_LAYERS,
        dropout=DROPOUT,
    )
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    emergence_criterion = nn.MSELoss()
    top8_criterion = nn.BCELoss()

    # ── Training loop ──
    best_val_loss = float("inf")
    best_epoch = 0

    log.info(f"\nTraining for {NUM_EPOCHS} epochs...")
    log.info(f"{'Epoch':>5} {'Train Loss':>10} {'Emrg Loss':>10} {'Top8 Loss':>10} "
             f"{'Val Loss':>10} {'Val Acc':>8} {'LR':>10}")
    log.info("-" * 70)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()

        # Forward pass
        output = model(data)
        node_emb = output["node_embeddings"]

        # Head 1: Emergence loss
        emergence_loss = emergence_criterion(
            output["emergence_scores"], emergence_targets
        )

        # Head 2: Top8 loss (train split only)
        top8_probs = model.predict_top8(
            node_emb["archetype"],
            node_emb["tournament"],
            arch_idx[train_mask],
            tourney_idx[train_mask],
        )
        top8_loss = top8_criterion(top8_probs, top8_labels[train_mask])

        # Combined loss
        total_loss = (
            EMERGENCE_LOSS_WEIGHT * emergence_loss
            + TOP8_LOSS_WEIGHT * top8_loss
        )

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # ── Validation ──
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_output = model(data)
                val_emb = val_output["node_embeddings"]

                val_top8_loss, val_acc, _, _, _ = _evaluate_top8(
                    model, data, val_emb, arch_idx, tourney_idx,
                    top8_labels, val_mask, top8_criterion,
                )

                val_emergence_loss = emergence_criterion(
                    val_output["emergence_scores"], emergence_targets
                )
                val_loss = val_emergence_loss + val_top8_loss

            lr = scheduler.get_last_lr()[0]
            log.info(
                f"{epoch:5d} {total_loss.item():10.4f} {emergence_loss.item():10.4f} "
                f"{top8_loss.item():10.4f} {val_loss.item():10.4f} {val_acc.item():8.3f} "
                f"{lr:10.6f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss.item(),
                    "val_acc": val_acc.item(),
                }, MODEL_PATH)

    log.info(f"\nBest model at epoch {best_epoch} with val_loss={best_val_loss:.4f}")
    log.info(f"Model saved to {MODEL_PATH}")

    # ── Final evaluation on held-out TEST set ──
    log.info("\n" + "=" * 60)
    log.info("Final Evaluation (held-out test set)")
    log.info("=" * 60)

    checkpoint = torch.load(MODEL_PATH, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        output = model(data)
        node_emb = output["node_embeddings"]

        # Emergence predictions
        log.info("\nArchetype Emergence Predictions (predicted share change):")
        scores = output["emergence_scores"]
        arch_names = data["archetype"].names
        ranked = sorted(zip(arch_names, scores.tolist()), key=lambda x: -x[1])
        for name, score in ranked:
            direction = "\u2191" if score > 0 else "\u2193"
            log.info(f"  {direction} {name:25s} {score:+.4f}")

        # Top8 predictions on TEST set (never seen during training or checkpoint selection)
        log.info("\nTop 8 Prediction (TEST set - held out):")
        test_loss, test_acc, test_prec, test_rec, test_f1 = _evaluate_top8(
            model, data, node_emb, arch_idx, tourney_idx,
            top8_labels, test_mask, top8_criterion,
        )

        log.info(f"  Accuracy:  {test_acc:.3f}")
        log.info(f"  Precision: {test_prec:.3f}")
        log.info(f"  Recall:    {test_rec:.3f}")
        log.info(f"  F1 Score:  {test_f1:.3f}")
        log.info(f"  BCE Loss:  {test_loss:.4f}")

        # Also show val set for comparison
        log.info("\nTop 8 Prediction (validation set - for comparison):")
        val_loss, val_acc, val_prec, val_rec, val_f1 = _evaluate_top8(
            model, data, node_emb, arch_idx, tourney_idx,
            top8_labels, val_mask, top8_criterion,
        )

        log.info(f"  Accuracy:  {val_acc:.3f}")
        log.info(f"  Precision: {val_prec:.3f}")
        log.info(f"  Recall:    {val_rec:.3f}")
        log.info(f"  F1 Score:  {val_f1:.3f}")
        log.info(f"  BCE Loss:  {val_loss:.4f}")


if __name__ == "__main__":
    train()
