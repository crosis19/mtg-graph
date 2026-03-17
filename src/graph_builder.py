"""Phase 3: Build a PyTorch Geometric HeteroData graph from all processed data.

Assembles four node types (Card, Archetype, Tournament, Set) and multiple edge
types into a single heterogeneous graph suitable for HGT training.

Node types:
  - card:       ~4,168 Standard-legal cards with numeric features
  - archetype:  ~12-32 meta archetypes with metagame features
  - tournament: ~15-24 events with metadata features
  - set:        ~12-15 Standard-legal sets with release/rotation features

Edge types:
  - (card, keyword_synergy, card)       — complementary keywords
  - (card, mechanical_synergy, card)    — trigger/enabler patterns
  - (card, semantic_synergy, card)      — embedding similarity (distinct text/mechanics)
  - (card, co_occurrence, card)         — deck co-occurrence (how often played together)
  - (archetype, contains, card)         — decklist inclusion
  - (archetype, counters, archetype)    — favorable matchup (directed)
  - (archetype, top8, tournament)       — placed in top 8, weighted by slot count
  - (set, printed_in, card)             — set contains this card
  - (card, from_set, set)               — reverse: card belongs to set
  - (card, in_deck, archetype)          — reverse of contains
  - (tournament, hosted, archetype)     — reverse of top8

Outputs:
  - data/processed/graph.pt  (serialized HeteroData)
"""

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from src.config import (
    CARD_EMBEDDINGS_PATH,
    CARD_EMBEDDING_INDEX_PATH,
    CARDS_PARQUET,
    CO_OCCURRENCE_EDGES_PATH,
    CO_OCCURRENCE_MIN_DECKS,
    COUNTERS_WIN_RATE_THRESHOLD,
    DECKLISTS_PARQUET,
    GRAPH_PATH,
    KEYWORD_MATRIX_PATH,
    MATCHUPS_PARQUET,
    MECHANICAL_EDGES_PATH,
    METAGAME_PARQUET,
    SEMANTIC_EDGES_PATH,
    TOURNAMENTS_PARQUET,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Standard-legal sets: authoritative list from whatsinstandard.com ──
# Only these set codes get Set nodes. Everything else is either rotated out
# or a Scryfall printing artifact (reprint tagged to its original set).
# Source: https://whatsinstandard.com/api/v6/standard.json (as of 2026-03-16)
STANDARD_LEGAL_SET_CODES = {
    "woe",   # Wilds of Eldraine
    "lci",   # The Lost Caverns of Ixalan
    "mkm",   # Murders at Karlov Manor
    "otj",   # Outlaws of Thunder Junction
    "big",   # The Big Score
    "blb",   # Bloomburrow
    "dsk",   # Duskmourn: House of Horror
    "fdn",   # Foundations (evergreen, exits Q1 2030)
    "dft",   # Aetherdrift
    "tdm",   # Tarkir: Dragonstorm
    "fin",   # Final Fantasy
    "eoe",   # Edge of Eternities
    "spm",   # Marvel's Spider-Man
    "tla",   # Avatar: The Last Airbender
    "ecl",   # Lorwyn Eclipsed
    "tmt",   # Teenage Mutant Ninja Turtles
}

# ── Banned cards in Standard (as of 2026-03-16) ──
# Source: whatsinstandard.com
BANNED_CARDS = {
    "Heartfire Hero",
    "Screaming Nemesis",
    "Cori-Steel Cutter",
    "Vivi Ornitier",
}


def _load_cards() -> tuple[pd.DataFrame, dict, torch.Tensor]:
    """Load card data and build card node features.

    Returns (cards_df, card_name_to_idx, card_features_tensor).
    Card features: [384-dim embedding + 11 numeric].
    set_recency is no longer a card feature — it lives on the Set node.
    is_banned flags cards currently banned in Standard.
    """
    cards = pd.read_parquet(CARDS_PARQUET)
    card_name_to_idx = {name: i for i, name in enumerate(cards["name"])}

    # Load pre-computed 384-dim embeddings as primary card features
    embeddings = np.load(CARD_EMBEDDINGS_PATH)
    with open(CARD_EMBEDDING_INDEX_PATH) as f:
        emb_index = json.load(f)
    emb_name_to_idx = {name: i for i, name in enumerate(emb_index)}

    # Build feature matrix: [embedding(384) + numeric(10)]
    # Note: is_instant and is_sorcery are separate features (mechanically distinct)
    n_cards = len(cards)
    emb_dim = embeddings.shape[1]

    # Handle backward compatibility: if cards.parquet still has old column
    if "is_instant_sorcery" in cards.columns and "is_instant" not in cards.columns:
        cards["is_instant"] = cards["type_line"].str.contains("Instant", na=False)
        cards["is_sorcery"] = cards["type_line"].str.contains("Sorcery", na=False)

    # Add banned flag: 1.0 if the card is banned in Standard, 0.0 otherwise
    cards["is_banned"] = cards["name"].isin(BANNED_CARDS).astype(np.float32)
    n_banned = int(cards["is_banned"].sum())
    if n_banned > 0:
        banned_names = cards.loc[cards["is_banned"] == 1.0, "name"].tolist()
        log.info(f"  Banned cards flagged ({n_banned}): {', '.join(banned_names)}")

    numeric_cols = [
        "cmc", "color_count", "keyword_count",
        "is_creature", "is_instant", "is_sorcery", "is_land",
        "is_planeswalker", "is_enchantment", "is_artifact",
        "is_banned",
    ]

    numeric = cards[numeric_cols].values.astype(np.float32)

    # Normalize numeric features
    for col_idx in range(numeric.shape[1]):
        col = numeric[:, col_idx]
        col_max = col.max()
        if col_max > 0:
            numeric[:, col_idx] = col / col_max

    features = np.zeros((n_cards, emb_dim + len(numeric_cols)), dtype=np.float32)
    for i, name in enumerate(cards["name"]):
        if name in emb_name_to_idx:
            features[i, :emb_dim] = embeddings[emb_name_to_idx[name]]
        features[i, emb_dim:] = numeric[i]

    log.info(f"  Card features: {features.shape} (384 embedding + {len(numeric_cols)} numeric)")
    return cards, card_name_to_idx, torch.tensor(features)


def _load_sets(cards: pd.DataFrame) -> tuple[list, dict, torch.Tensor]:
    """Build Set nodes from the card pool.

    Only includes sets in STANDARD_LEGAL_SET_CODES (authoritative whitelist
    from whatsinstandard.com). Cards from non-Standard sets are Scryfall
    printing artifacts — those cards won't get set edges but still exist
    as card nodes.

    Features: [set_recency, set_size_norm, avg_cmc_norm]
      - set_recency: normalized release date (0=oldest in Standard, 1=newest)
      - set_size_norm: number of cards in this set / max set size
      - avg_cmc_norm: average CMC of cards in this set / max avg CMC

    Returns (set_codes, set_names, set_code_to_idx, set_features_tensor).
    """
    # Get unique sets and their properties
    set_info = []
    skipped_sets = []
    for set_code in cards["set"].unique():
        set_cards = cards[cards["set"] == set_code]
        set_name = set_cards["set_name"].iloc[0] if "set_name" in set_cards.columns else set_code
        n_cards = len(set_cards)
        avg_cmc = set_cards["cmc"].mean() if "cmc" in set_cards.columns else 0.0

        # Get release date from first card in the set
        release_date = None
        if "released_at" in set_cards.columns:
            dates = pd.to_datetime(set_cards["released_at"], errors="coerce")
            valid_dates = dates.dropna()
            if len(valid_dates) > 0:
                release_date = valid_dates.iloc[0]

        # Filter: only sets in the Standard-legal whitelist
        if set_code.lower() not in STANDARD_LEGAL_SET_CODES:
            skipped_sets.append((set_code, set_name, n_cards, release_date))
            continue

        set_info.append({
            "set_code": set_code,
            "set_name": set_name,
            "n_cards": n_cards,
            "avg_cmc": avg_cmc,
            "release_date": release_date,
        })

    if skipped_sets:
        n_cards_skipped = sum(s[2] for s in skipped_sets)
        log.info(f"  Skipped {len(skipped_sets)} non-Standard sets "
                 f"({n_cards_skipped} cards without set edges):")
        for code, name, count, date in skipped_sets:
            date_str = date.strftime("%Y-%m-%d") if pd.notna(date) else "?"
            log.info(f"    {code:6s} {name:40s} cards={count}  released={date_str}")

    set_df = pd.DataFrame(set_info)

    # Sort by release date for consistent ordering
    set_df["release_date"] = pd.to_datetime(set_df["release_date"], errors="coerce")
    set_df = set_df.sort_values("release_date", na_position="first").reset_index(drop=True)

    set_codes = set_df["set_code"].tolist()
    set_names = set_df["set_name"].tolist()
    set_code_to_idx = {code: i for i, code in enumerate(set_codes)}

    # Build features
    n_sets = len(set_codes)

    # set_recency: normalized release date ordinal [0, 1]
    release_dates = set_df["release_date"]
    min_date = release_dates.min()
    max_date = release_dates.max()
    date_range = (max_date - min_date).days if pd.notna(min_date) and pd.notna(max_date) else 1
    if date_range <= 0:
        date_range = 1
    recency = ((release_dates - min_date).dt.days / date_range).fillna(0.5).values.astype(np.float32)

    # set_size_norm: card count normalized
    sizes = set_df["n_cards"].values.astype(np.float32)
    max_size = sizes.max() if sizes.max() > 0 else 1.0
    sizes_norm = sizes / max_size

    # avg_cmc_norm
    avg_cmcs = set_df["avg_cmc"].values.astype(np.float32)
    max_avg_cmc = avg_cmcs.max() if avg_cmcs.max() > 0 else 1.0
    avg_cmcs_norm = avg_cmcs / max_avg_cmc

    features = np.stack([recency, sizes_norm, avg_cmcs_norm], axis=1)

    log.info(f"  Set features: {features.shape} (recency, size, avg_cmc) — {n_sets} sets")
    for i, (code, name) in enumerate(zip(set_codes, set_names)):
        date_str = set_df["release_date"].iloc[i].strftime("%Y-%m-%d") if pd.notna(set_df["release_date"].iloc[i]) else "?"
        log.info(f"    [{i}] {code:6s} {name:35s} released={date_str}  cards={int(sizes[i]):3d}  recency={recency[i]:.2f}")

    return set_codes, set_names, set_code_to_idx, torch.tensor(features)


def _build_set_card_edges(
    cards: pd.DataFrame,
    card_name_to_idx: dict,
    set_code_to_idx: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build set -> card 'printed_in' edges.

    Every card is connected to its set. Weight is uniform (1.0) since
    membership is binary — a card either is or isn't in the set.
    """
    src_list, dst_list = [], []
    for _, row in cards.iterrows():
        card = row["name"]
        set_code = row["set"]
        if card not in card_name_to_idx or set_code not in set_code_to_idx:
            continue
        src_list.append(set_code_to_idx[set_code])
        dst_list.append(card_name_to_idx[card])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
    log.info(f"  printed_in: {edge_index.shape[1]} edges (set -> card)")
    return edge_index, edge_weight


def _load_archetypes(metagame: pd.DataFrame) -> tuple[list, dict, torch.Tensor]:
    """Build archetype nodes from metagame data.

    Uses the latest snapshot for each archetype as node features.
    Returns (archetype_names, arch_name_to_idx, arch_features_tensor).
    """
    # Get latest snapshot per archetype
    latest = metagame.sort_values("snapshot_date").groupby("archetype").last().reset_index()
    archetype_names = sorted(latest["archetype"].tolist())
    arch_name_to_idx = {name: i for i, name in enumerate(archetype_names)}

    # Features: [meta_share, win_rate, n_colors]
    features = np.zeros((len(archetype_names), 3), dtype=np.float32)
    for _, row in latest.iterrows():
        idx = arch_name_to_idx[row["archetype"]]
        features[idx, 0] = row["meta_share_pct"] / 100.0
        features[idx, 1] = row["win_rate"] / 100.0
        n_colors = len(row["color_identity"].split("|")) if pd.notna(row["color_identity"]) else 0
        features[idx, 2] = n_colors / 5.0

    log.info(f"  Archetype features: {features.shape} (share, wr, colors)")
    return archetype_names, arch_name_to_idx, torch.tensor(features)


def _load_tournaments(tournaments: pd.DataFrame) -> tuple[list, dict, torch.Tensor]:
    """Build tournament nodes.

    Returns (event_ids, event_id_to_idx, tourney_features_tensor).
    """
    event_ids = tournaments["event_id"].tolist()
    event_id_to_idx = {eid: i for i, eid in enumerate(event_ids)}

    # Features: [player_count_norm, day_ordinal_norm]
    dates = pd.to_datetime(tournaments["date"], errors="coerce")
    min_date = dates.min()
    day_ordinals = (dates - min_date).dt.days.fillna(0).values.astype(np.float32)
    max_days = day_ordinals.max() if day_ordinals.max() > 0 else 1.0
    day_ordinals /= max_days

    player_counts = pd.to_numeric(tournaments["player_count"], errors="coerce").fillna(0).values.astype(np.float32)
    max_players = player_counts.max() if player_counts.max() > 0 else 1.0
    player_counts /= max_players

    features = np.stack([player_counts, day_ordinals], axis=1)
    log.info(f"  Tournament features: {features.shape} (players, time)")
    return event_ids, event_id_to_idx, torch.tensor(features)


def _build_card_synergy_edges(
    card_name_to_idx: dict,
) -> dict:
    """Load all three synergy layers and build edge index tensors.

    Returns dict of {edge_type_str: (edge_index, edge_weight)}.
    """
    edges = {}
    synergy_files = [
        (KEYWORD_MATRIX_PATH, "keyword_synergy", "synergy_count"),
        (MECHANICAL_EDGES_PATH, "mechanical_synergy", "synergy_count"),
        (SEMANTIC_EDGES_PATH, "semantic_synergy", "similarity"),
    ]

    for path, edge_type, weight_col in synergy_files:
        if not path.exists():
            log.warning(f"  {edge_type}: file not found at {path}, skipping")
            continue

        df = pd.read_parquet(path)
        src_list, dst_list, weight_list = [], [], []
        skipped = 0

        for _, row in df.iterrows():
            card_a = row["card_a"]
            card_b = row["card_b"]
            if card_a not in card_name_to_idx or card_b not in card_name_to_idx:
                skipped += 1
                continue
            src_idx = card_name_to_idx[card_a]
            dst_idx = card_name_to_idx[card_b]
            weight = float(row[weight_col]) if weight_col in df.columns else 1.0

            # Undirected: add both directions
            src_list.extend([src_idx, dst_idx])
            dst_list.extend([dst_idx, src_idx])
            weight_list.extend([weight, weight])

        if src_list:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_weight = torch.tensor(weight_list, dtype=torch.float32)
            # Normalize weights to [0, 1]
            if edge_weight.max() > 0:
                edge_weight = edge_weight / edge_weight.max()
            edges[edge_type] = (edge_index, edge_weight)
            log.info(f"  {edge_type}: {edge_index.shape[1]} edges (skipped {skipped} unknown cards)")
        else:
            log.warning(f"  {edge_type}: no valid edges!")

    return edges


def _build_co_occurrence_edges(
    decklists: pd.DataFrame,
    card_name_to_idx: dict,
    min_decks: int = CO_OCCURRENCE_MIN_DECKS,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Build card-card co-occurrence edges from decklist data.

    Two cards get an edge when they appear in the same decklist at least
    `min_decks` times. Weight is the number of shared decklists, normalized.

    This captures empirical synergy: cards players actually choose to play
    together, regardless of whether their text is similar.
    """
    if decklists.empty:
        log.warning("  co_occurrence: no decklists available, skipping")
        return None

    # Build a unique deck identifier from (event_id, archetype, placement)
    # Each unique combo represents one decklist
    decklists = decklists.copy()
    decklists["deck_uid"] = (
        decklists["event_id"].astype(str) + "_"
        + decklists["archetype"].astype(str) + "_"
        + decklists["placement"].astype(str)
    )

    # For each deck, get the set of cards (only cards we know about)
    deck_cards: dict[str, list[int]] = {}
    for _, row in decklists.iterrows():
        card = row["card_name"]
        if card not in card_name_to_idx:
            continue
        uid = row["deck_uid"]
        deck_cards.setdefault(uid, []).append(card_name_to_idx[card])

    # Deduplicate cards within each deck (a card appears once per deck edge-wise)
    for uid in deck_cards:
        deck_cards[uid] = list(set(deck_cards[uid]))

    # Count co-occurrences: for each pair of cards, how many decks contain both
    pair_counts: Counter = Counter()
    for uid, card_indices in deck_cards.items():
        card_indices.sort()
        for i in range(len(card_indices)):
            for j in range(i + 1, len(card_indices)):
                pair_counts[(card_indices[i], card_indices[j])] += 1

    # Filter by minimum deck threshold
    src_list, dst_list, weight_list = [], [], []
    for (a, b), count in pair_counts.items():
        if count < min_decks:
            continue
        # Undirected: add both directions
        src_list.extend([a, b])
        dst_list.extend([b, a])
        weight_list.extend([count, count])

    if not src_list:
        log.warning("  co_occurrence: no edges met the minimum threshold")
        return None

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_weight = torch.tensor(weight_list, dtype=torch.float32)
    # Normalize weights to [0, 1]
    if edge_weight.max() > 0:
        edge_weight = edge_weight / edge_weight.max()

    log.info(f"  co_occurrence: {edge_index.shape[1]} edges "
             f"({len(pair_counts)} unique pairs, min_decks={min_decks})")
    return edge_index, edge_weight


def _build_contains_edges(
    decklists: pd.DataFrame,
    card_name_to_idx: dict,
    arch_name_to_idx: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build archetype -> card 'contains' edges from decklists.

    Edge weight is the average number of copies across appearances.
    """
    # Aggregate: for each (archetype, card), average copies across all decklists
    agg = (
        decklists.groupby(["archetype", "card_name"])["copies"]
        .mean()
        .reset_index()
    )

    src_list, dst_list, weight_list = [], [], []
    for _, row in agg.iterrows():
        arch = row["archetype"]
        card = row["card_name"]
        if arch not in arch_name_to_idx or card not in card_name_to_idx:
            continue
        src_list.append(arch_name_to_idx[arch])
        dst_list.append(card_name_to_idx[card])
        weight_list.append(float(row["copies"]) / 4.0)  # Normalize by max copies

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_weight = torch.tensor(weight_list, dtype=torch.float32)
    log.info(f"  contains: {edge_index.shape[1]} edges (archetype -> card)")
    return edge_index, edge_weight


def _build_counters_edges(
    matchups: pd.DataFrame,
    arch_name_to_idx: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build directed archetype -> archetype 'counters' edges.

    Only creates edge when win_rate > COUNTERS_WIN_RATE_THRESHOLD.
    """
    src_list, dst_list, weight_list = [], [], []

    for _, row in matchups.iterrows():
        arch_a = row["archetype_a"]
        arch_b = row["archetype_b"]
        if arch_a not in arch_name_to_idx or arch_b not in arch_name_to_idx:
            continue

        # A counters B
        if row["win_rate_a"] > COUNTERS_WIN_RATE_THRESHOLD:
            src_list.append(arch_name_to_idx[arch_a])
            dst_list.append(arch_name_to_idx[arch_b])
            weight_list.append(row["win_rate_a"] / 100.0)

        # B counters A
        if row["win_rate_b"] > COUNTERS_WIN_RATE_THRESHOLD:
            src_list.append(arch_name_to_idx[arch_b])
            dst_list.append(arch_name_to_idx[arch_a])
            weight_list.append(row["win_rate_b"] / 100.0)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_weight = torch.tensor(weight_list, dtype=torch.float32)
    log.info(f"  counters: {edge_index.shape[1]} edges (archetype -> archetype)")
    return edge_index, edge_weight


def _build_top8_edges(
    decklists: pd.DataFrame,
    arch_name_to_idx: dict,
    event_id_to_idx: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build archetype -> tournament 'top8' edges from decklist placements.

    Weight reflects how many top-8 slots the archetype occupies in the event,
    normalized to [0, 1] (i.e., slots / 8). This captures dominance: an
    archetype taking 5/8 slots is qualitatively different from taking 1/8.
    """
    # Count how many top-8 slots each archetype occupies per event
    # Each unique (event_id, archetype, placement) row = one slot
    slots = (
        decklists[["event_id", "archetype", "placement"]]
        .drop_duplicates()
        .groupby(["event_id", "archetype"])
        .size()
        .reset_index(name="slot_count")
    )

    src_list, dst_list, weight_list = [], [], []
    for _, row in slots.iterrows():
        arch = row["archetype"]
        event = row["event_id"]
        if arch not in arch_name_to_idx or event not in event_id_to_idx:
            continue
        src_list.append(arch_name_to_idx[arch])
        dst_list.append(event_id_to_idx[event])
        # Weight = fraction of top-8 slots occupied (1 slot = 0.125, 8 slots = 1.0)
        weight_list.append(row["slot_count"] / 8.0)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_weight = torch.tensor(weight_list, dtype=torch.float32)
    log.info(f"  top8: {edge_index.shape[1]} edges (archetype -> tournament, weighted by slot count)")
    return edge_index, edge_weight


def build_graph() -> HeteroData:
    """Assemble the full heterogeneous graph."""
    log.info("Loading node data...")
    cards, card_name_to_idx, card_features = _load_cards()

    metagame = pd.read_parquet(METAGAME_PARQUET)
    arch_names, arch_name_to_idx, arch_features = _load_archetypes(metagame)

    tournaments = pd.read_parquet(TOURNAMENTS_PARQUET)
    event_ids, event_id_to_idx, tourney_features = _load_tournaments(tournaments)

    set_codes, set_names, set_code_to_idx, set_features = _load_sets(cards)

    decklists = pd.read_parquet(DECKLISTS_PARQUET)
    matchups = pd.read_parquet(MATCHUPS_PARQUET)

    # ── Build HeteroData ──
    data = HeteroData()

    # Node features
    data["card"].x = card_features
    data["card"].names = list(cards["name"])
    data["archetype"].x = arch_features
    data["archetype"].names = arch_names
    data["tournament"].x = tourney_features
    data["tournament"].event_ids = event_ids
    data["set"].x = set_features
    data["set"].codes = set_codes
    data["set"].names = set_names

    log.info(f"\nNodes: card={data['card'].x.shape[0]}, "
             f"archetype={data['archetype'].x.shape[0]}, "
             f"tournament={data['tournament'].x.shape[0]}, "
             f"set={data['set'].x.shape[0]}")

    # ── Card-to-card synergy edges (keyword, mechanical, semantic) ──
    log.info("\nBuilding card synergy edges...")
    synergy_edges = _build_card_synergy_edges(card_name_to_idx)

    for edge_type, (edge_index, edge_weight) in synergy_edges.items():
        data["card", edge_type, "card"].edge_index = edge_index
        data["card", edge_type, "card"].edge_weight = edge_weight

    # ── Co-occurrence edges (card-card from decklists) ──
    log.info("\nBuilding co-occurrence edges from decklists...")
    co_occ_result = _build_co_occurrence_edges(decklists, card_name_to_idx)
    if co_occ_result is not None:
        co_occ_idx, co_occ_w = co_occ_result
        data["card", "co_occurrence", "card"].edge_index = co_occ_idx
        data["card", "co_occurrence", "card"].edge_weight = co_occ_w

        # Also save the co-occurrence edges to parquet for reference
        _save_co_occurrence_parquet(co_occ_idx, co_occ_w, list(cards["name"]))

    # ── Set -> Card edges (printed_in / from_set) ──
    log.info("\nBuilding set-card edges...")
    printed_in_idx, printed_in_w = _build_set_card_edges(cards, card_name_to_idx, set_code_to_idx)
    data["set", "printed_in", "card"].edge_index = printed_in_idx
    data["set", "printed_in", "card"].edge_weight = printed_in_w

    # Reverse: card -> set (from_set)
    data["card", "from_set", "set"].edge_index = torch.stack([
        printed_in_idx[1], printed_in_idx[0]
    ])
    data["card", "from_set", "set"].edge_weight = printed_in_w

    # ── Contains edges (archetype -> card) ──
    log.info("\nBuilding structural edges...")
    contains_idx, contains_w = _build_contains_edges(decklists, card_name_to_idx, arch_name_to_idx)
    data["archetype", "contains", "card"].edge_index = contains_idx
    data["archetype", "contains", "card"].edge_weight = contains_w

    # ── Counters edges (archetype -> archetype) ──
    counters_idx, counters_w = _build_counters_edges(matchups, arch_name_to_idx)
    data["archetype", "counters", "archetype"].edge_index = counters_idx
    data["archetype", "counters", "archetype"].edge_weight = counters_w

    # ── Top8 edges (archetype -> tournament, weighted by slot count) ──
    top8_idx, top8_w = _build_top8_edges(decklists, arch_name_to_idx, event_id_to_idx)
    data["archetype", "top8", "tournament"].edge_index = top8_idx
    data["archetype", "top8", "tournament"].edge_weight = top8_w

    # ── Reverse edges (PyG convention for message passing) ──
    # card -> archetype (reverse of contains)
    data["card", "in_deck", "archetype"].edge_index = torch.stack([
        contains_idx[1], contains_idx[0]
    ])
    data["card", "in_deck", "archetype"].edge_weight = contains_w

    # tournament -> archetype (reverse of top8)
    data["tournament", "hosted", "archetype"].edge_index = torch.stack([
        top8_idx[1], top8_idx[0]
    ])
    data["tournament", "hosted", "archetype"].edge_weight = top8_w

    # ── Summary ──
    log.info(f"\n{'='*60}")
    log.info(f"Graph summary:")
    log.info(f"  Node types: {data.node_types}")
    log.info(f"  Edge types: {data.edge_types}")
    total_edges = sum(
        data[et].edge_index.shape[1] for et in data.edge_types
    )
    log.info(f"  Total edges: {total_edges:,}")
    log.info(f"{'='*60}")

    return data


def _save_co_occurrence_parquet(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    card_names: list[str],
) -> None:
    """Save co-occurrence edges to parquet for inspection/reuse."""
    # Only save unique pairs (one direction)
    seen = set()
    records = []
    for k in range(edge_index.shape[1]):
        a, b = int(edge_index[0, k]), int(edge_index[1, k])
        pair = (min(a, b), max(a, b))
        if pair in seen:
            continue
        seen.add(pair)
        records.append({
            "card_a": card_names[pair[0]],
            "card_b": card_names[pair[1]],
            "edge_type": "co_occurrence",
            "shared_decks": float(edge_weight[k]),  # Already normalized
        })

    df = pd.DataFrame(records)
    df.to_parquet(CO_OCCURRENCE_EDGES_PATH, index=False)
    log.info(f"  Saved {len(df)} co-occurrence pairs to {CO_OCCURRENCE_EDGES_PATH}")


def main():
    data = build_graph()

    # Save
    torch.save(data, GRAPH_PATH)
    log.info(f"\nGraph saved to {GRAPH_PATH}")

    # Validate
    log.info("\nValidation checks:")
    for node_type in data.node_types:
        n = data[node_type].x.shape[0]
        d = data[node_type].x.shape[1]
        has_nan = torch.isnan(data[node_type].x).any().item()
        log.info(f"  {node_type}: {n} nodes, {d}-dim features, NaN={has_nan}")

    for edge_type in data.edge_types:
        ei = data[edge_type].edge_index
        n_edges = ei.shape[1]
        src_type, _, dst_type = edge_type
        src_max = data[src_type].x.shape[0]
        dst_max = data[dst_type].x.shape[0]
        src_valid = (ei[0] >= 0).all() and (ei[0] < src_max).all()
        dst_valid = (ei[1] >= 0).all() and (ei[1] < dst_max).all()
        log.info(f"  {edge_type}: {n_edges:,} edges, indices_valid={src_valid and dst_valid}")


if __name__ == "__main__":
    main()
