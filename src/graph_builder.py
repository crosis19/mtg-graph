"""Phase 3: Build a PyTorch Geometric HeteroData graph from all processed data.

Assembles three node types (Card, Archetype, Set) and multiple edge types
into a single heterogeneous graph suitable for HGT training.

Node types:
  - card:       ~4,168 Standard-legal cards with numeric features
  - archetype:  ~5-10 meta archetypes (>= DECK_MIN_META_SHARE%) with metagame features
  - set:        ~12-16 Standard-legal sets with release/rotation features

Edge types (11):
  - (card, keyword_synergy, card)        — complementary keywords
  - (card, mechanical_synergy, card)     — trigger/enabler patterns
  - (card, semantic_synergy, card)       — embedding similarity
  - (archetype, maindecks, card)         — mainboard inclusion (edge_attr: copies, inclusion_pct)
  - (card, maindeck_of, archetype)       — reverse
  - (archetype, sideboards, card)        — sideboard inclusion (edge_attr: copies, inclusion_pct)
  - (card, sideboard_of, archetype)      — reverse
  - (archetype, counters, archetype)     — favorable matchup (directed)
  - (archetype, countered_by, archetype) — reverse
  - (set, printed_in, card)              — set contains this card
  - (card, from_set, set)               — reverse

Outputs:
  - data/processed/graph.pt  (serialized HeteroData)
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from src.config import (
    CARD_EMBEDDINGS_PATH,
    CARD_EMBEDDING_INDEX_PATH,
    CARDS_PARQUET,
    COUNTERS_WIN_RATE_THRESHOLD,
    DECK_MIN_META_SHARE,
    DECKLISTS_PARQUET,
    GRAPH_PATH,
    KEYWORD_MATRIX_PATH,
    MATCHUPS_PARQUET,
    MECHANICAL_EDGES_PATH,
    METAGAME_PARQUET,
    SEMANTIC_EDGES_PATH,
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


def _load_archetypes(
    metagame: pd.DataFrame,
    decklists: pd.DataFrame,
    cards_df: pd.DataFrame,
    card_embeddings: np.ndarray,
    card_emb_name_to_idx: dict[str, int],
) -> tuple[list, dict, torch.Tensor]:
    """Build archetype nodes from metagame data with rich features.

    Features per archetype (~390-dim):
      - meta_share (1): normalized meta share percentage
      - n_colors / 5 (1): number of colors in the archetype's identity
      - mean_card_embedding (384): average card embedding across deck cards
      - creature_ratio (1): fraction of deck that's creatures
      - spell_ratio (1): fraction of deck that's instants/sorceries
      - avg_cmc / 7 (1): average converted mana cost
      - cmc_variance / 10 (1): variance of CMC across deck cards

    Returns (archetype_names, arch_name_to_idx, arch_features_tensor).
    """
    # Get latest snapshot per archetype, filtered by meta share threshold
    latest = metagame.sort_values("snapshot_date").groupby("archetype").last().reset_index()
    latest = latest[latest["meta_share_pct"] >= DECK_MIN_META_SHARE]
    archetype_names = sorted(latest["archetype"].tolist())
    log.info(f"  {len(archetype_names)} archetypes with >= {DECK_MIN_META_SHARE}% meta share")
    arch_name_to_idx = {name: i for i, name in enumerate(archetype_names)}

    # ── Derive color identity from decklist cards ──
    card_colors: dict[str, set[str]] = {}
    for _, row in cards_df.iterrows():
        ci = row.get("color_identity")
        if pd.notna(ci) and ci:
            card_colors[row["name"]] = set(ci.split("|"))
        else:
            card_colors[row["name"]] = set()

    # ── Per-archetype stats from decklists ──
    emb_dim = card_embeddings.shape[1]
    n_archetypes = len(archetype_names)
    n_scalar = 6  # meta_share, n_colors, creature_ratio, spell_ratio, avg_cmc, cmc_var
    feature_dim = n_scalar + emb_dim

    features = np.zeros((n_archetypes, feature_dim), dtype=np.float32)

    for arch in archetype_names:
        idx = arch_name_to_idx[arch]
        arch_cards = decklists.loc[decklists["archetype"] == arch, "card_name"].unique()

        # Color identity
        colors = set()
        for card_name in arch_cards:
            colors |= card_colors.get(card_name, set())
        features[idx, 1] = len(colors) / 5.0

        # Mean card embedding
        emb_list = []
        cmcs = []
        n_creatures = 0
        n_spells = 0
        for card_name in arch_cards:
            if card_name in card_emb_name_to_idx:
                emb_list.append(card_embeddings[card_emb_name_to_idx[card_name]])
            card_row = cards_df.loc[cards_df["name"] == card_name]
            if card_row.empty:
                continue
            card_row = card_row.iloc[0]
            if pd.notna(card_row.get("cmc")):
                cmcs.append(float(card_row["cmc"]))
            type_line = str(card_row.get("type_line", ""))
            if "Creature" in type_line:
                n_creatures += 1
            if "Instant" in type_line or "Sorcery" in type_line:
                n_spells += 1

        n_cards = max(len(arch_cards), 1)
        features[idx, 2] = n_creatures / n_cards  # creature_ratio
        features[idx, 3] = n_spells / n_cards      # spell_ratio
        if cmcs:
            features[idx, 4] = np.mean(cmcs) / 7.0  # avg_cmc
            features[idx, 5] = np.var(cmcs) / 10.0   # cmc_variance
        if emb_list:
            features[idx, n_scalar:] = np.mean(emb_list, axis=0)

    # Fill meta_share from metagame data
    for _, row in latest.iterrows():
        idx = arch_name_to_idx[row["archetype"]]
        if pd.notna(row.get("meta_share_pct")):
            features[idx, 0] = row["meta_share_pct"] / 100.0

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    log.info(f"  Archetype features: {features.shape} "
             f"({n_scalar} scalar + {emb_dim} mean embedding)")
    log.info(f"  NaN check: {np.isnan(features).sum()} NaN remaining")
    return archetype_names, arch_name_to_idx, torch.tensor(features)



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


def _build_deck_edges(
    decklists: pd.DataFrame,
    card_name_to_idx: dict,
    arch_name_to_idx: dict,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Build archetype -> card deck edges, split by mainboard/sideboard.

    Returns dict mapping edge_type_name to (edge_index, edge_attr).
    edge_attr is (num_edges, 2): [avg_copies/4.0, inclusion_pct].
    """
    results = {}

    if "board" not in decklists.columns:
        log.warning("  'board' column missing from decklists — treating all as mainboard. "
                    "Re-run Phase 1 scraping to get main/side split.")
        boards = {"maindecks": decklists}
    else:
        boards = {
            "maindecks": decklists[decklists["board"] == "main"],
            "sideboards": decklists[decklists["board"] == "side"],
        }

    # Detect column names (new schema: avg_copies/inclusion_pct, old: copies)
    has_new_schema = "avg_copies" in decklists.columns
    copies_col = "avg_copies" if has_new_schema else "copies"

    for edge_name, board_df in boards.items():
        if board_df.empty:
            log.warning(f"  {edge_name}: no data, creating empty edge set")
            results[edge_name] = (
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0, 2), dtype=torch.float32),
            )
            continue

        # Group by (archetype, card_name) — may already be unique per new schema
        agg = (
            board_df.groupby(["archetype", "card_name"])[copies_col]
            .mean()
            .reset_index()
        )
        # Get inclusion_pct if available
        if has_new_schema and "inclusion_pct" in board_df.columns:
            incl = (
                board_df.groupby(["archetype", "card_name"])["inclusion_pct"]
                .mean()
                .reset_index()
            )
            agg = agg.merge(incl, on=["archetype", "card_name"], how="left")
            agg["inclusion_pct"] = agg["inclusion_pct"].fillna(1.0)
        else:
            agg["inclusion_pct"] = 1.0

        src_list, dst_list, attr_copies, attr_incl = [], [], [], []
        for _, row in agg.iterrows():
            arch = row["archetype"]
            card = row["card_name"]
            if arch not in arch_name_to_idx or card not in card_name_to_idx:
                continue
            src_list.append(arch_name_to_idx[arch])
            dst_list.append(card_name_to_idx[card])
            attr_copies.append(float(row[copies_col]) / 4.0)
            attr_incl.append(float(row["inclusion_pct"]))

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.tensor(
            list(zip(attr_copies, attr_incl)), dtype=torch.float32
        )  # (n_edges, 2)
        log.info(f"  {edge_name}: {edge_index.shape[1]} edges (archetype -> card)")
        results[edge_name] = (edge_index, edge_attr)

    return results


def _fuzzy_match_archetypes(
    matchup_names: set[str],
    graph_names: dict[str, int],
    threshold: float = 0.75,
) -> dict[str, int]:
    """Build a mapping from matchup archetype names to graph node indices.

    Tries exact match first, then falls back to fuzzy matching that requires
    the color identity (first word) to match. This prevents false positives
    like 'Mardu Midrange' → 'Dimir Midrange'.
    """
    from difflib import SequenceMatcher

    mapping: dict[str, int] = {}
    fuzzy_matches: list[tuple[str, str, float]] = []

    def _normalize(s: str) -> str:
        return s.lower().replace("-", " ").strip()

    def _color_word(s: str) -> str:
        """Extract the color identity prefix (e.g., 'mono green', 'izzet')."""
        return _normalize(s).split()[0] if s.strip() else ""

    for name in matchup_names:
        # Exact match
        if name in graph_names:
            mapping[name] = graph_names[name]
            continue

        # Fuzzy match: require the color identity word to match
        name_color = _color_word(name)
        best_match = None
        best_score = 0.0
        name_norm = _normalize(name)

        for gname in graph_names:
            gname_color = _color_word(gname)

            # Color identity must match (izzet=izzet, mono=mono, dimir=dimir)
            if name_color != gname_color:
                continue

            gname_norm = _normalize(gname)
            score = SequenceMatcher(None, name_norm, gname_norm).ratio()
            if score > best_score:
                best_score = score
                best_match = gname

        if best_match and best_score >= threshold:
            mapping[name] = graph_names[best_match]
            fuzzy_matches.append((name, best_match, best_score))

    if fuzzy_matches:
        log.info("  Fuzzy-matched archetype names:")
        for src, dst, score in fuzzy_matches:
            log.info(f"    '{src}' → '{dst}' (score={score:.2f})")

    return mapping


def _build_counters_edges(
    matchups: pd.DataFrame,
    arch_name_to_idx: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build directed archetype -> archetype 'counters' edges.

    Only creates edge when win_rate > COUNTERS_WIN_RATE_THRESHOLD.
    Uses fuzzy matching to handle naming differences between data sources.
    """
    # Build fuzzy name mapping
    matchup_names = set(matchups["archetype_a"].unique()) | set(matchups["archetype_b"].unique())
    name_to_idx = _fuzzy_match_archetypes(matchup_names, arch_name_to_idx)
    log.info(f"  Matched {len(name_to_idx)}/{len(matchup_names)} matchup archetypes "
             f"to {len(arch_name_to_idx)} graph archetypes")

    src_list, dst_list, weight_list = [], [], []

    for _, row in matchups.iterrows():
        arch_a = row["archetype_a"]
        arch_b = row["archetype_b"]
        if arch_a not in name_to_idx or arch_b not in name_to_idx:
            continue

        # A counters B
        if row["win_rate_a"] > COUNTERS_WIN_RATE_THRESHOLD:
            src_list.append(name_to_idx[arch_a])
            dst_list.append(name_to_idx[arch_b])
            weight_list.append(row["win_rate_a"] / 100.0)

        # B counters A
        if row["win_rate_b"] > COUNTERS_WIN_RATE_THRESHOLD:
            src_list.append(name_to_idx[arch_b])
            dst_list.append(name_to_idx[arch_a])
            weight_list.append(row["win_rate_b"] / 100.0)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_weight = torch.tensor(weight_list, dtype=torch.float32)
    log.info(f"  counters: {edge_index.shape[1]} edges (archetype -> archetype)")
    return edge_index, edge_weight



def build_graph() -> HeteroData:
    """Assemble the full heterogeneous graph."""
    log.info("Loading node data...")
    cards, card_name_to_idx, card_features = _load_cards()

    metagame = pd.read_parquet(METAGAME_PARQUET)
    decklists = pd.read_parquet(DECKLISTS_PARQUET)
    cards_df = pd.read_parquet(CARDS_PARQUET)

    # Load card embeddings for archetype mean-embedding features
    card_embeddings = np.load(CARD_EMBEDDINGS_PATH)
    with open(CARD_EMBEDDING_INDEX_PATH) as f:
        card_emb_index = json.load(f)
    card_emb_name_to_idx = {name: i for i, name in enumerate(card_emb_index)}

    arch_names, arch_name_to_idx, arch_features = _load_archetypes(
        metagame, decklists, cards_df, card_embeddings, card_emb_name_to_idx
    )

    set_codes, set_names, set_code_to_idx, set_features = _load_sets(cards)

    matchups = pd.read_parquet(MATCHUPS_PARQUET)

    # ── Build HeteroData ──
    data = HeteroData()

    # Node features
    data["card"].x = card_features
    data["card"].names = list(cards["name"])
    data["archetype"].x = arch_features
    data["archetype"].names = arch_names
    data["set"].x = set_features
    data["set"].codes = set_codes
    data["set"].names = set_names

    log.info(f"\nNodes: card={data['card'].x.shape[0]}, "
             f"archetype={data['archetype'].x.shape[0]}, "
             f"set={data['set'].x.shape[0]}")

    # ── Card-to-card synergy edges (keyword, mechanical, semantic) ──
    log.info("\nBuilding card synergy edges...")
    synergy_edges = _build_card_synergy_edges(card_name_to_idx)

    for edge_type, (edge_index, edge_weight) in synergy_edges.items():
        data["card", edge_type, "card"].edge_index = edge_index
        data["card", edge_type, "card"].edge_weight = edge_weight

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

    # ── Deck edges (archetype -> card, split by mainboard/sideboard) ──
    log.info("\nBuilding deck edges...")
    deck_edges = _build_deck_edges(decklists, card_name_to_idx, arch_name_to_idx)

    for edge_name, (edge_index, edge_attr) in deck_edges.items():
        # Forward: archetype -> card
        data["archetype", edge_name, "card"].edge_index = edge_index
        data["archetype", edge_name, "card"].edge_attr = edge_attr

        # Reverse: card -> archetype
        reverse_name = "maindeck_of" if edge_name == "maindecks" else "sideboard_of"
        data["card", reverse_name, "archetype"].edge_index = torch.stack([
            edge_index[1], edge_index[0]
        ])
        data["card", reverse_name, "archetype"].edge_attr = edge_attr

    # ── Counters edges (archetype -> archetype) ──
    counters_idx, counters_w = _build_counters_edges(matchups, arch_name_to_idx)
    data["archetype", "counters", "archetype"].edge_index = counters_idx
    data["archetype", "counters", "archetype"].edge_weight = counters_w

    # ── Countered_by edges (reverse of counters) ──
    data["archetype", "countered_by", "archetype"].edge_index = torch.stack([
        counters_idx[1], counters_idx[0]
    ])
    data["archetype", "countered_by", "archetype"].edge_weight = counters_w

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


def main():
    data = build_graph()

    # Sanitize all node features — ensure no NaN/Inf before saving
    for node_type in data.node_types:
        x = data[node_type].x
        nan_count = torch.isnan(x).sum().item()
        if nan_count > 0:
            log.warning(f"  Replacing {nan_count} NaN values in {node_type} features before save")
            data[node_type].x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

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
