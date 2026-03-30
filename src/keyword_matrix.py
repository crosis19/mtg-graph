"""Phase 2, Layer 1: Keyword compatibility matrix for keyword synergy edges.

Uses Scryfall's structured keyword arrays to identify complementary keyword pairs.
No oracle text parsing needed — keywords come pre-extracted from Scryfall.

Example synergies:
  Deathtouch + First Strike → combat synergy
  Trample + power-boosting → damage throughput
  Flying + evasion support → evasion enabler

Outputs: keyword_matrix.parquet (card-to-card keyword synergy edges)
"""

import logging
from itertools import combinations

import pandas as pd

from src.config import CARDS_PARQUET, KEYWORD_MATRIX_PATH, DATA_PROCESSED

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Keyword compatibility rules: (keyword_a, keyword_b) → synergy_label
# These define mechanically complementary keyword pairs
KEYWORD_SYNERGIES = {
    # Combat synergies
    ("Deathtouch", "First strike"): "combat_synergy",
    ("Deathtouch", "Double strike"): "combat_synergy",
    ("Trample", "Double strike"): "damage_throughput",
    ("Menace", "Deathtouch"): "combat_evasion",
    ("First strike", "Lifelink"): "combat_sustain",
    ("Double strike", "Lifelink"): "combat_sustain",

    # Evasion synergies
    ("Flying", "Vigilance"): "evasion_pressure",
    ("Flying", "Lifelink"): "evasion_sustain",

    # Keyword-to-mechanic synergies
    ("Trample", "Ward"): "protected_threat",
    ("Hexproof", "Trample"): "protected_threat",
    ("Vigilance", "Ward"): "resilient_blocker",
    ("Indestructible", "Trample"): "unstoppable_threat",
    ("Indestructible", "Lifelink"): "unkillable_sustain",

    # Resource synergies
    ("Landfall", "Reach"): "land_ramp_defense",
    ("Flash", "Flying"): "instant_speed_evasion",
    ("Flash", "Ward"): "protected_surprise",

    # Tribal/archetype enablers
    ("Convoke", "Vigilance"): "convoke_enabler",
    ("Convoke", "Lifelink"): "token_convoke",
    ("Prowess", "Flash"): "spell_matters",

    # Engine synergies
    ("Fabricate", "Trample"): "token_power",
    ("Exploit", "Deathtouch"): "sacrifice_value",
    ("Escape", "Deathtouch"): "recursive_threat",
    ("Escape", "Lifelink"): "recursive_sustain",
    ("Disturb", "Flying"): "graveyard_evasion",
}

# Broader keyword categories for fuzzy matching
KEYWORD_CATEGORIES = {
    "evasion": {"Flying", "Menace", "Fear", "Intimidate", "Shadow", "Skulk", "Horsemanship"},
    "protection": {"Ward", "Hexproof", "Shroud", "Indestructible", "Protection"},
    "combat_boost": {"First strike", "Double strike", "Trample", "Deathtouch"},
    "sustain": {"Lifelink", "Vigilance"},
    "recursion": {"Escape", "Disturb", "Unearth", "Embalm", "Eternalize"},
    "cost_reduction": {"Convoke", "Delve", "Affinity", "Improvise"},
    "spell_matters": {"Prowess", "Magecraft", "Storm"},
    "land_matters": {"Landfall"},
    "sacrifice": {"Exploit", "Emerge"},
}


def _normalize_keyword(kw: str) -> str:
    """Normalize keyword casing for matching."""
    return kw.strip().lower()


def _keywords_synergize(kw_a: str, kw_b: str) -> str | None:
    """Check if two keywords have a defined synergy. Returns label or None."""
    # Direct lookup
    pair = (kw_a, kw_b)
    if pair in KEYWORD_SYNERGIES:
        return KEYWORD_SYNERGIES[pair]
    # Reverse lookup
    pair_rev = (kw_b, kw_a)
    if pair_rev in KEYWORD_SYNERGIES:
        return KEYWORD_SYNERGIES[pair_rev]

    # Category-based fuzzy matching: different complementary categories
    cat_a = {cat for cat, kws in KEYWORD_CATEGORIES.items() if kw_a in kws}
    cat_b = {cat for cat, kws in KEYWORD_CATEGORIES.items() if kw_b in kws}

    # Evasion + protection = hard-to-answer threat
    if "evasion" in cat_a and "protection" in cat_b:
        return "evasion_protection"
    if "protection" in cat_a and "evasion" in cat_b:
        return "evasion_protection"

    # Combat boost + sustain = dominant board presence
    if "combat_boost" in cat_a and "sustain" in cat_b:
        return "combat_sustain"
    if "sustain" in cat_a and "combat_boost" in cat_b:
        return "combat_sustain"

    # Cost reduction + spell_matters = engine synergy
    if "cost_reduction" in cat_a and "spell_matters" in cat_b:
        return "engine_synergy"
    if "spell_matters" in cat_a and "cost_reduction" in cat_b:
        return "engine_synergy"

    return None


def build_keyword_edges(cards_df: pd.DataFrame) -> pd.DataFrame:
    """Build card-to-card keyword synergy edges from structured keyword data.

    For each pair of cards that share complementary keywords, create an edge.
    Only considers cards that have at least one keyword.
    """
    # Parse keywords from pipe-separated string
    has_keywords = cards_df["keywords"].apply(
        lambda x: bool(x) and pd.notna(x) and len(x.strip()) > 0
    )
    cards_with_kw = cards_df[has_keywords].copy()
    cards_with_kw["keyword_list"] = cards_with_kw["keywords"].str.split("|")

    log.info("Cards with keywords: %d / %d", len(cards_with_kw), len(cards_df))

    # Build index: keyword → list of card names
    kw_to_cards: dict[str, list[str]] = {}
    card_keywords: dict[str, list[str]] = {}

    for _, row in cards_with_kw.iterrows():
        name = row["name"]
        kws = row["keyword_list"]
        card_keywords[name] = kws
        for kw in kws:
            kw_to_cards.setdefault(kw, []).append(name)

    # Find synergy edges between card pairs
    edges = []
    seen_pairs = set()

    card_names = list(card_keywords.keys())
    total_pairs = len(card_names) * (len(card_names) - 1) // 2
    log.info("Checking %d card pairs for keyword synergies...", total_pairs)

    for i, card_a in enumerate(card_names):
        for card_b in card_names[i + 1:]:
            pair_key = (card_a, card_b) if card_a < card_b else (card_b, card_a)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            # Check all keyword pairs between the two cards
            synergies_found = []
            for kw_a in card_keywords[card_a]:
                for kw_b in card_keywords[card_b]:
                    if kw_a == kw_b:
                        continue
                    label = _keywords_synergize(kw_a, kw_b)
                    if label:
                        synergies_found.append({
                            "keyword_a": kw_a,
                            "keyword_b": kw_b,
                            "synergy_type": label,
                        })

            if synergies_found:
                edges.append({
                    "card_a": pair_key[0],
                    "card_b": pair_key[1],
                    "edge_type": "keyword_synergy",
                    "synergy_count": len(synergies_found),
                    "synergy_types": "|".join(s["synergy_type"] for s in synergies_found),
                    "keyword_pairs": "|".join(
                        f"{s['keyword_a']}+{s['keyword_b']}" for s in synergies_found
                    ),
                })

    df = pd.DataFrame(edges)
    log.info("Found %d keyword synergy edges.", len(df))
    return df


def run() -> pd.DataFrame:
    """Full pipeline: load cards → build keyword edges → save."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    if not CARDS_PARQUET.exists():
        raise FileNotFoundError(
            f"Cards data not found at {CARDS_PARQUET}. Run ingest_cards.py first."
        )

    cards_df = pd.read_parquet(CARDS_PARQUET)
    edges_df = build_keyword_edges(cards_df)

    edges_df.to_parquet(KEYWORD_MATRIX_PATH, index=False)
    log.info("Saved %d keyword edges to %s", len(edges_df), KEYWORD_MATRIX_PATH)
    return edges_df


if __name__ == "__main__":
    run()
