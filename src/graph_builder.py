"""Phase 3: Build a PyTorch Geometric HeteroData graph from all processed data.

Assembles three node types (Card, Archetype, Set) and multiple edge types
into a single heterogeneous graph suitable for GNN training.

Node types:
  - card:       ~4,168 Standard-legal cards (407-dim: 384 embedding + 23 scalar)
  - archetype:  ~10 meta archetypes (14-dim: 6 color flags + 8 type ratios)
  - set:        ~12-16 Standard-legal sets (385-dim: 384 avg embedding + 1 recency)

Edge types (11):
  - (card, keyword_synergy, card)        — complementary keywords (undirected)
  - (card, semantic_synergy, card)       — embedding similarity (undirected)
  - (card, counters, card)               — card counters another card (directed)
  - (card, countered_by, card)           — reverse of counters
  - (card, removes, card)                — card removes another card (directed)
  - (card, removed_by, card)             — reverse of removes
  - (set, printed_in, card)              — set contains this card (directed)
  - (archetype, contains, card)          — deck includes this card (directed)
  - (card, in_deck, archetype)           — reverse of contains
  - (archetype, win_rate, archetype)     — source's win rate vs target (directed)
  - (archetype, lose_rate, archetype)    — source's loss rate vs target (directed)

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
    COUNTER_EDGES_PATH,
    DECK_BASIC_LAND_MAX_COPIES,
    DECK_NUM_ARCHETYPES,
    DECKLISTS_PARQUET,
    GRAPH_PATH,
    KEYWORD_MATRIX_PATH,
    MATCHUPS_PARQUET,
    METAGAME_PARQUET,
    REMOVAL_EDGES_PATH,
    SEMANTIC_EDGES_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Standard-legal sets: authoritative list from whatsinstandard.com ──
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

# ── Basic land names ──
BASIC_LAND_NAMES = {"Plains", "Island", "Swamp", "Mountain", "Forest"}


def _load_cards() -> tuple[pd.DataFrame, dict, torch.Tensor]:
    """Load card data and build card node features.

    Returns (cards_df, card_name_to_idx, card_features_tensor).
    Card features: [384-dim embedding + 23 scalar].

    Scalar features (23):
      cmc_norm, is_creature, is_instant, is_sorcery, is_land, is_planeswalker,
      is_enchantment, is_artifact, is_battle, rarity_enc,
      is_white, is_blue, is_black, is_red, is_green, is_colorless,
      gen_white, gen_blue, gen_black, gen_red, gen_green, gen_colorless
    """
    cards = pd.read_parquet(CARDS_PARQUET)

    # Sort so per-card output neurons have structured locality:
    # color identity → card type → CMC → set → name
    cards = cards.sort_values(
        by=["color_identity", "type_line", "cmc", "set", "name"],
        na_position="last",
    ).reset_index(drop=True)

    card_name_to_idx = {name: i for i, name in enumerate(cards["name"])}
    # Add front-face aliases so MTGGoldfish names resolve to Scryfall names.
    # E.g. "Bramble Familiar" → idx of "Bramble Familiar // Fetch Quest"
    for name, idx in list(card_name_to_idx.items()):
        if " // " in name:
            front = name.split(" // ")[0]
            card_name_to_idx.setdefault(front, idx)

    # Load pre-computed 384-dim embeddings
    embeddings = np.load(CARD_EMBEDDINGS_PATH)
    with open(CARD_EMBEDDING_INDEX_PATH) as f:
        emb_index = json.load(f)
    emb_name_to_idx = {name: i for i, name in enumerate(emb_index)}

    n_cards = len(cards)
    emb_dim = embeddings.shape[1]

    numeric_cols = [
        "cmc_norm",
        "is_creature", "is_instant", "is_sorcery", "is_land",
        "is_planeswalker", "is_enchantment", "is_artifact", "is_battle",
        "rarity_enc",
        "is_white", "is_blue", "is_black", "is_red", "is_green", "is_colorless",
        "gen_white", "gen_blue", "gen_black", "gen_red", "gen_green", "gen_colorless",
    ]

    # Verify all columns exist
    missing = [c for c in numeric_cols if c not in cards.columns]
    if missing:
        raise ValueError(f"Missing columns in cards.parquet: {missing}. "
                         f"Re-run ingest_cards.py to generate new features.")

    numeric = cards[numeric_cols].values.astype(np.float32)

    features = np.zeros((n_cards, emb_dim + len(numeric_cols)), dtype=np.float32)
    for i, name in enumerate(cards["name"]):
        if name in emb_name_to_idx:
            features[i, :emb_dim] = embeddings[emb_name_to_idx[name]]
        features[i, emb_dim:] = numeric[i]

    log.info(f"  Card features: {features.shape} (384 embedding + {len(numeric_cols)} scalar)")
    return cards, card_name_to_idx, torch.tensor(features)


def _load_sets(
    cards: pd.DataFrame,
    card_embeddings: np.ndarray,
    card_emb_name_to_idx: dict[str, int],
) -> tuple[list, list, dict, torch.Tensor]:
    """Build Set nodes with average card embedding + recency features.

    Features per set (385-dim):
      - semantic_embedding_avg (384): mean of card embeddings in this set
      - set_recency (1): normalized release date [0=oldest, 1=newest]

    Returns (set_codes, set_names, set_code_to_idx, set_features_tensor).
    """
    set_info = []
    skipped_sets = []

    for set_code in cards["set"].unique():
        set_cards = cards[cards["set"] == set_code]
        set_name = set_cards["set_name"].iloc[0] if "set_name" in set_cards.columns else set_code
        n_cards = len(set_cards)

        release_date = None
        if "released_at" in set_cards.columns:
            dates = pd.to_datetime(set_cards["released_at"], errors="coerce")
            valid_dates = dates.dropna()
            if len(valid_dates) > 0:
                release_date = valid_dates.iloc[0]

        if set_code.lower() not in STANDARD_LEGAL_SET_CODES:
            skipped_sets.append((set_code, set_name, n_cards, release_date))
            continue

        # Compute average card embedding for this set
        emb_list = []
        for card_name in set_cards["name"]:
            if card_name in card_emb_name_to_idx:
                emb_list.append(card_embeddings[card_emb_name_to_idx[card_name]])
        avg_emb = np.mean(emb_list, axis=0) if emb_list else np.zeros(card_embeddings.shape[1])

        set_info.append({
            "set_code": set_code,
            "set_name": set_name,
            "n_cards": n_cards,
            "release_date": release_date,
            "avg_embedding": avg_emb,
        })

    if skipped_sets:
        n_cards_skipped = sum(s[2] for s in skipped_sets)
        log.info(f"  Skipped {len(skipped_sets)} non-Standard sets "
                 f"({n_cards_skipped} cards without set edges)")

    set_df = pd.DataFrame(set_info)
    set_df["release_date"] = pd.to_datetime(set_df["release_date"], errors="coerce")
    set_df = set_df.sort_values("release_date", na_position="first").reset_index(drop=True)

    set_codes = set_df["set_code"].tolist()
    set_names = set_df["set_name"].tolist()
    set_code_to_idx = {code: i for i, code in enumerate(set_codes)}

    # Build features: [avg_embedding(384), set_recency(1)]
    n_sets = len(set_codes)
    emb_dim = card_embeddings.shape[1]

    release_dates = set_df["release_date"]
    min_date = release_dates.min()
    max_date = release_dates.max()
    date_range = (max_date - min_date).days if pd.notna(min_date) and pd.notna(max_date) else 1
    if date_range <= 0:
        date_range = 1
    recency = ((release_dates - min_date).dt.days / date_range).fillna(0.5).values.astype(np.float32)

    features = np.zeros((n_sets, emb_dim + 1), dtype=np.float32)
    for i in range(n_sets):
        features[i, :emb_dim] = set_df.iloc[i]["avg_embedding"]
        features[i, emb_dim] = recency[i]

    log.info(f"  Set features: {features.shape} (384 avg embedding + 1 recency) — {n_sets} sets")
    return set_codes, set_names, set_code_to_idx, torch.tensor(features)


def _load_archetypes(
    metagame: pd.DataFrame,
    decklists: pd.DataFrame,
    cards_df: pd.DataFrame,
) -> tuple[list, dict, torch.Tensor]:
    """Build archetype nodes with color flags and type ratios.

    Features per archetype (14-dim):
      - contains_white, contains_blue, contains_black, contains_red,
        contains_green, contains_colorless (6 binary)
      - creature_ratio, instant_ratio, sorcery_ratio, land_ratio,
        planeswalker_ratio, enchantment_ratio, artifact_ratio, battle_ratio (8 continuous)

    Returns (archetype_names, arch_name_to_idx, arch_features_tensor).
    """
    latest = metagame.sort_values("snapshot_date").groupby("archetype").last().reset_index()

    # Take top N archetypes by meta share (no minimum threshold)
    latest = latest.dropna(subset=["meta_share_pct"])
    latest = latest.nlargest(DECK_NUM_ARCHETYPES, "meta_share_pct")
    archetype_names = sorted(latest["archetype"].tolist())
    log.info(f"  {len(archetype_names)} archetypes (top {DECK_NUM_ARCHETYPES} by meta share)")
    arch_name_to_idx = {name: i for i, name in enumerate(archetype_names)}

    # Card color identity lookup
    card_colors: dict[str, set[str]] = {}
    card_types: dict[str, str] = {}
    for _, row in cards_df.iterrows():
        ci = row.get("color_identity")
        card_colors[row["name"]] = set(ci.split("|")) if pd.notna(ci) and ci else set()
        card_types[row["name"]] = str(row.get("type_line", ""))

    n_archetypes = len(archetype_names)
    features = np.zeros((n_archetypes, 14), dtype=np.float32)

    for arch in archetype_names:
        idx = arch_name_to_idx[arch]
        arch_cards = decklists.loc[decklists["archetype"] == arch, "card_name"].unique()
        n_cards = max(len(arch_cards), 1)

        # Color presence flags
        all_colors = set()
        type_counts = {
            "Creature": 0, "Instant": 0, "Sorcery": 0, "Land": 0,
            "Planeswalker": 0, "Enchantment": 0, "Artifact": 0, "Battle": 0,
        }

        for card_name in arch_cards:
            all_colors |= card_colors.get(card_name, set())
            tl = card_types.get(card_name, "")
            for t in type_counts:
                if t in tl:
                    type_counts[t] += 1

        # Color flags (6)
        features[idx, 0] = float("W" in all_colors)
        features[idx, 1] = float("U" in all_colors)
        features[idx, 2] = float("B" in all_colors)
        features[idx, 3] = float("R" in all_colors)
        features[idx, 4] = float("G" in all_colors)
        # contains_colorless: has colorless non-land cards
        has_colorless = any(
            len(card_colors.get(c, set())) == 0 and "Land" not in card_types.get(c, "")
            for c in arch_cards
        )
        features[idx, 5] = float(has_colorless)

        # Type ratios (8)
        features[idx, 6] = type_counts["Creature"] / n_cards
        features[idx, 7] = type_counts["Instant"] / n_cards
        features[idx, 8] = type_counts["Sorcery"] / n_cards
        features[idx, 9] = type_counts["Land"] / n_cards
        features[idx, 10] = type_counts["Planeswalker"] / n_cards
        features[idx, 11] = type_counts["Enchantment"] / n_cards
        features[idx, 12] = type_counts["Artifact"] / n_cards
        features[idx, 13] = type_counts["Battle"] / n_cards

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    log.info(f"  Archetype features: {features.shape} (6 color flags + 8 type ratios)")
    return archetype_names, arch_name_to_idx, torch.tensor(features)


def _build_set_card_edges(
    cards: pd.DataFrame,
    card_name_to_idx: dict,
    set_code_to_idx: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build set -> card 'printed_in' edges (directed, weight=1.0)."""
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


def _build_card_synergy_edges(card_name_to_idx: dict) -> dict:
    """Load keyword and semantic synergy edges.

    Returns dict of {edge_type_str: (edge_index, edge_weight)}.
    """
    edges = {}
    synergy_files = [
        (KEYWORD_MATRIX_PATH, "keyword_synergy", "synergy_count"),
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
            if edge_weight.max() > 0:
                edge_weight = edge_weight / edge_weight.max()
            edges[edge_type] = (edge_index, edge_weight)
            log.info(f"  {edge_type}: {edge_index.shape[1]} edges (skipped {skipped} unknown cards)")
        else:
            log.warning(f"  {edge_type}: no valid edges!")

    return edges


def _build_card_interaction_edges(card_name_to_idx: dict) -> dict:
    """Load counter and removal edges between cards.

    Returns dict of {edge_type_str: (edge_index, edge_weight)}.
    Creates bidirectional pairs: counters/countered_by and removes/removed_by.
    """
    edges = {}
    interaction_files = [
        (COUNTER_EDGES_PATH, "counters", "countered_by"),
        (REMOVAL_EDGES_PATH, "removes", "removed_by"),
    ]

    for path, forward_type, reverse_type in interaction_files:
        if not path.exists():
            log.warning(f"  {forward_type}: file not found at {path}, skipping")
            continue

        df = pd.read_parquet(path)
        fwd_src, fwd_dst = [], []
        skipped = 0

        for _, row in df.iterrows():
            source = row["source_card"]
            target = row["target_card"]
            if source not in card_name_to_idx or target not in card_name_to_idx:
                skipped += 1
                continue
            fwd_src.append(card_name_to_idx[source])
            fwd_dst.append(card_name_to_idx[target])

        if fwd_src:
            fwd_index = torch.tensor([fwd_src, fwd_dst], dtype=torch.long)
            fwd_weight = torch.ones(fwd_index.shape[1], dtype=torch.float32)
            edges[forward_type] = (fwd_index, fwd_weight)

            # Reverse
            rev_index = torch.stack([fwd_index[1], fwd_index[0]])
            edges[reverse_type] = (rev_index, fwd_weight.clone())

            log.info(f"  {forward_type}: {fwd_index.shape[1]} edges "
                     f"(skipped {skipped} unknown cards)")
        else:
            log.warning(f"  {forward_type}: no valid edges!")

    return edges


def _build_deck_edges(
    decklists: pd.DataFrame,
    card_name_to_idx: dict,
    arch_name_to_idx: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build archetype -> card 'contains' edges from mainboard only.

    Edge weight encodes copy count:
      - Non-basic cards: 0.25=1, 0.5=2, 0.75=3, 1.0=4+
      - Basic lands: actual_count / DECK_BASIC_LAND_MAX_COPIES

    Returns (edge_index, edge_weight).
    """
    # Detect column names
    has_new_schema = "avg_copies" in decklists.columns
    copies_col = "avg_copies" if has_new_schema else "copies"

    # Filter to mainboard only — sideboard cards can confuse core deck identity
    if "board" in decklists.columns:
        decklists = decklists[decklists["board"] == "main"]
        log.info(f"  Filtered to mainboard: {len(decklists)} decklist rows")

    # Aggregate copies per archetype-card pair
    agg = (
        decklists.groupby(["archetype", "card_name"])[copies_col]
        .max()
        .reset_index()
    )

    src_list, dst_list, weight_list = [], [], []
    for _, row in agg.iterrows():
        arch = row["archetype"]
        card = row["card_name"]
        # Normalize split card separator: "A/B" → "A // B"
        if card not in card_name_to_idx and "/" in card and " // " not in card:
            card = card.replace("/", " // ")
        if arch not in arch_name_to_idx or card not in card_name_to_idx:
            continue

        copies = float(row[copies_col])
        is_basic = card in BASIC_LAND_NAMES

        if is_basic:
            # Uncapped encoding for basic lands
            weight = min(copies / DECK_BASIC_LAND_MAX_COPIES, 1.0)
        else:
            # Encode copy count: 1→0.25, 2→0.5, 3→0.75, 4+→1.0
            copies_clamped = min(max(round(copies), 1), 4)
            weight = copies_clamped * 0.25

        src_list.append(arch_name_to_idx[arch])
        dst_list.append(card_name_to_idx[card])
        weight_list.append(weight)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_weight = torch.tensor(weight_list, dtype=torch.float32)
    log.info(f"  contains: {edge_index.shape[1]} edges (archetype -> card)")
    return edge_index, edge_weight


def _fuzzy_match_archetypes(
    matchup_names: set[str],
    graph_names: dict[str, int],
    threshold: float = 0.75,
) -> dict[str, int]:
    """Build a mapping from matchup archetype names to graph node indices.

    Tries exact match first, then fuzzy matching requiring color identity match.
    """
    from difflib import SequenceMatcher

    mapping: dict[str, int] = {}
    fuzzy_matches: list[tuple[str, str, float]] = []

    def _normalize(s: str) -> str:
        return s.lower().replace("-", " ").strip()

    def _color_word(s: str) -> str:
        return _normalize(s).split()[0] if s.strip() else ""

    for name in matchup_names:
        if name in graph_names:
            mapping[name] = graph_names[name]
            continue

        name_color = _color_word(name)
        best_match = None
        best_score = 0.0
        name_norm = _normalize(name)

        for gname in graph_names:
            gname_color = _color_word(gname)
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
            log.info(f"    '{src}' -> '{dst}' (score={score:.2f})")

    return mapping


def _build_winrate_edges(
    matchups: pd.DataFrame,
    arch_name_to_idx: dict,
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """Build two directed archetype -> archetype win rate edge types.

    Returns dict with two entries:
      - "win_rate": A→B edges carrying A's win rate against B (my win rate vs opponent)
      - "lose_rate": A→B edges carrying A's loss rate against B (opponent's win rate vs me)

    These are separate edge types so the GNN learns distinct message transformations
    for "I beat this archetype" vs "I lose to this archetype".

    All pairwise rates stored as continuous 0-1 (no threshold filter).
    """
    matchup_names = set(matchups["archetype_a"].unique()) | set(matchups["archetype_b"].unique())
    name_to_idx = _fuzzy_match_archetypes(matchup_names, arch_name_to_idx)
    log.info(f"  Matched {len(name_to_idx)}/{len(matchup_names)} matchup archetypes "
             f"to {len(arch_name_to_idx)} graph archetypes")

    # win_rate: edge from source to target, weight = source's win rate against target
    # lose_rate: edge from source to target, weight = target's win rate against source
    wr_src, wr_dst, wr_weight = [], [], []
    lr_src, lr_dst, lr_weight = [], [], []

    for _, row in matchups.iterrows():
        arch_a = row["archetype_a"]
        arch_b = row["archetype_b"]
        if arch_a not in name_to_idx or arch_b not in name_to_idx:
            continue

        a_idx = name_to_idx[arch_a]
        b_idx = name_to_idx[arch_b]
        wr_a = row["win_rate_a"] / 100.0
        wr_b = row["win_rate_b"] / 100.0

        # A's perspective: A→B with A's win rate, A→B with B's win rate (A's loss)
        wr_src.append(a_idx); wr_dst.append(b_idx); wr_weight.append(wr_a)
        lr_src.append(a_idx); lr_dst.append(b_idx); lr_weight.append(wr_b)

        # B's perspective: B→A with B's win rate, B→A with A's win rate (B's loss)
        wr_src.append(b_idx); wr_dst.append(a_idx); wr_weight.append(wr_b)
        lr_src.append(b_idx); lr_dst.append(a_idx); lr_weight.append(wr_a)

    edges = {}
    wr_index = torch.tensor([wr_src, wr_dst], dtype=torch.long)
    wr_w = torch.tensor(wr_weight, dtype=torch.float32)
    edges["win_rate"] = (wr_index, wr_w)
    log.info(f"  win_rate: {wr_index.shape[1]} edges (archetype -> archetype, source's win rate)")

    lr_index = torch.tensor([lr_src, lr_dst], dtype=torch.long)
    lr_w = torch.tensor(lr_weight, dtype=torch.float32)
    edges["lose_rate"] = (lr_index, lr_w)
    log.info(f"  lose_rate: {lr_index.shape[1]} edges (archetype -> archetype, source's loss rate)")

    return edges


def build_graph() -> HeteroData:
    """Assemble the full heterogeneous graph."""
    log.info("Loading node data...")
    cards, card_name_to_idx, card_features = _load_cards()

    metagame = pd.read_parquet(METAGAME_PARQUET)
    decklists = pd.read_parquet(DECKLISTS_PARQUET)
    cards_df = pd.read_parquet(CARDS_PARQUET)
    matchups = pd.read_parquet(MATCHUPS_PARQUET)

    # Load card embeddings for set avg embedding features
    card_embeddings = np.load(CARD_EMBEDDINGS_PATH)
    with open(CARD_EMBEDDING_INDEX_PATH) as f:
        card_emb_index = json.load(f)
    card_emb_name_to_idx = {name: i for i, name in enumerate(card_emb_index)}

    arch_names, arch_name_to_idx, arch_features = _load_archetypes(
        metagame, decklists, cards_df
    )

    set_codes, set_names, set_code_to_idx, set_features = _load_sets(
        cards, card_embeddings, card_emb_name_to_idx
    )

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

    # ── Card-to-card synergy edges (keyword, semantic) ──
    log.info("\nBuilding card synergy edges...")
    synergy_edges = _build_card_synergy_edges(card_name_to_idx)
    for edge_type, (edge_index, edge_weight) in synergy_edges.items():
        data["card", edge_type, "card"].edge_index = edge_index
        data["card", edge_type, "card"].edge_weight = edge_weight

    # ── Card-to-card interaction edges (counters, removes) ──
    log.info("\nBuilding card interaction edges...")
    interaction_edges = _build_card_interaction_edges(card_name_to_idx)
    for edge_type, (edge_index, edge_weight) in interaction_edges.items():
        data["card", edge_type, "card"].edge_index = edge_index
        data["card", edge_type, "card"].edge_weight = edge_weight

    # ── Set <-> Card edges (printed_in + member_of reverse) ──
    log.info("\nBuilding set-card edges...")
    printed_in_idx, printed_in_w = _build_set_card_edges(cards, card_name_to_idx, set_code_to_idx)
    data["set", "printed_in", "card"].edge_index = printed_in_idx
    data["set", "printed_in", "card"].edge_weight = printed_in_w

    # Reverse: card → set (cards send messages to their set nodes)
    data["card", "member_of", "set"].edge_index = torch.stack([
        printed_in_idx[1], printed_in_idx[0]
    ])
    data["card", "member_of", "set"].edge_weight = printed_in_w
    log.info(f"  member_of: {printed_in_idx.shape[1]} edges (card -> set)")

    # ── Deck edges (archetype <-> card, unified contains/in_deck) ──
    log.info("\nBuilding deck edges...")
    contains_idx, contains_w = _build_deck_edges(decklists, card_name_to_idx, arch_name_to_idx)
    data["archetype", "contains", "card"].edge_index = contains_idx
    data["archetype", "contains", "card"].edge_weight = contains_w

    # Reverse: card -> archetype (in_deck)
    data["card", "in_deck", "archetype"].edge_index = torch.stack([
        contains_idx[1], contains_idx[0]
    ])
    data["card", "in_deck", "archetype"].edge_weight = contains_w

    # ── Win rate edges (archetype -> archetype, two distinct types) ──
    log.info("\nBuilding win rate edges...")
    winrate_edges = _build_winrate_edges(matchups, arch_name_to_idx)
    for edge_type, (edge_index, edge_weight) in winrate_edges.items():
        data["archetype", edge_type, "archetype"].edge_index = edge_index
        data["archetype", edge_type, "archetype"].edge_weight = edge_weight

    # ── Summary ──
    log.info(f"\n{'='*60}")
    log.info("Graph summary:")
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

    # Sanitize all node features
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
