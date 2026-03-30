"""Phase 2b: Extract counter and removal edges between cards from oracle text.

Parses Scryfall oracle text to identify:
- Counter relationships: cards that can counter other cards' spells
- Removal relationships: cards that can destroy/exile/damage other cards

Outputs counter_edges.parquet and removal_edges.parquet with (source_card, target_card) pairs.
"""

import logging
import re

import pandas as pd

from src.config import (
    CARDS_PARQUET,
    COUNTER_EDGES_PATH,
    DATA_PROCESSED,
    REMOVAL_EDGES_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Counter patterns ────────────────────────────────────────────────────────

# Patterns that identify a card as a counterspell and what it can target.
# Each tuple: (regex pattern on oracle text, set of targetable type_line substrings).
# None means "any non-land spell".
COUNTER_PATTERNS = [
    (r"counter target spell", None),  # universal counter
    (r"counter target creature spell", {"Creature"}),
    (r"counter target noncreature spell", {"noncreature"}),  # special handling
    (r"counter target instant or sorcery spell", {"Instant", "Sorcery"}),
    (r"counter target artifact spell", {"Artifact"}),
    (r"counter target enchantment spell", {"Enchantment"}),
    (r"counter target planeswalker spell", {"Planeswalker"}),
    (r"counter target activated or triggered ability", None),  # broad
]

# ── Removal patterns ───────────────────────────────────────────────────────

# Patterns that identify targeted removal and what they can remove.
# Each tuple: (regex, set of removable type_line substrings, removal_type label).
REMOVAL_PATTERNS = [
    # Destroy targeted
    (r"destroy target creature", {"Creature"}, "destroy"),
    (r"destroy target permanent", None, "destroy"),  # hits anything on battlefield
    (r"destroy target artifact", {"Artifact"}, "destroy"),
    (r"destroy target enchantment", {"Enchantment"}, "destroy"),
    (r"destroy target planeswalker", {"Planeswalker"}, "destroy"),
    (r"destroy target artifact or enchantment", {"Artifact", "Enchantment"}, "destroy"),
    (r"destroy target nonland permanent", {"nonland"}, "destroy"),  # special handling
    # Exile targeted
    (r"exile target creature", {"Creature"}, "exile"),
    (r"exile target permanent", None, "exile"),
    (r"exile target artifact", {"Artifact"}, "exile"),
    (r"exile target enchantment", {"Enchantment"}, "exile"),
    (r"exile target nonland permanent", {"nonland"}, "exile"),
    # Board wipes
    (r"destroy all creatures", {"Creature"}, "board_wipe"),
    (r"exile all creatures", {"Creature"}, "board_wipe"),
    (r"destroy all nonland permanents", {"nonland"}, "board_wipe"),
    # Damage-based removal (simplified: treat as removing creatures)
    (r"deals? \d+ damage to (?:target creature|any target|each creature)", {"Creature"}, "damage"),
    # -N/-N effects
    (r"target creature gets -\d+/-\d+", {"Creature"}, "minus_stats"),
    # Bounce
    (r"return target creature to its owner's hand", {"Creature"}, "bounce"),
    (r"return target nonland permanent to its owner's hand", {"nonland"}, "bounce"),
]


def _is_non_land_spell(type_line: str) -> bool:
    """Check if a card is a non-land spell (can be countered)."""
    return not pd.isna(type_line) and "Land" not in type_line


def _matches_target_types(type_line: str, target_types: set[str] | None) -> bool:
    """Check if a card's type line matches the targeting restriction."""
    if target_types is None:
        return True  # hits anything
    if pd.isna(type_line):
        return False
    tl = type_line

    for tt in target_types:
        if tt == "noncreature":
            if "Creature" not in tl and "Land" not in tl:
                return True
        elif tt == "nonland":
            if "Land" not in tl:
                return True
        elif tt in tl:
            return True
    return False


def build_counter_edges(cards_df: pd.DataFrame) -> pd.DataFrame:
    """Build directed counter edges: source_card counters target_card.

    A counter source is any card whose oracle text matches a counter pattern.
    A target is any non-land spell matching the counter's targeting restriction.
    """
    log.info("Building counter edges...")

    # Identify counter sources
    counter_sources = []  # list of (card_name, target_types)
    for _, card in cards_df.iterrows():
        text = (card.get("oracle_text") or "").lower()
        for pattern, target_types in COUNTER_PATTERNS:
            if re.search(pattern, text):
                counter_sources.append((card["name"], target_types))
                break  # one match per card is enough

    log.info("Found %d counter source cards", len(counter_sources))

    # Build edges
    edges = []
    for source_name, target_types in counter_sources:
        for _, target in cards_df.iterrows():
            if target["name"] == source_name:
                continue
            if not _is_non_land_spell(target.get("type_line", "")):
                continue
            if _matches_target_types(target.get("type_line", ""), target_types):
                edges.append({"source_card": source_name, "target_card": target["name"]})

    df = pd.DataFrame(edges)
    log.info("Built %d counter edges", len(df))
    return df


def build_removal_edges(cards_df: pd.DataFrame) -> pd.DataFrame:
    """Build directed removal edges: source_card removes target_card.

    A removal source is any card whose oracle text matches a removal pattern.
    A target is any card whose type line matches the removal's targeting restriction.
    """
    log.info("Building removal edges...")

    # Identify removal sources
    removal_sources = []  # list of (card_name, target_types, removal_type)
    for _, card in cards_df.iterrows():
        text = (card.get("oracle_text") or "").lower()
        for pattern, target_types, removal_type in REMOVAL_PATTERNS:
            if re.search(pattern, text):
                removal_sources.append((card["name"], target_types, removal_type))
                break  # one match per card

    log.info("Found %d removal source cards", len(removal_sources))

    # Build edges
    edges = []
    for source_name, target_types, removal_type in removal_sources:
        for _, target in cards_df.iterrows():
            if target["name"] == source_name:
                continue
            # Removal targets must be permanents (not instants/sorceries unless "permanent")
            target_tl = target.get("type_line", "")
            if _matches_target_types(target_tl, target_types):
                edges.append({
                    "source_card": source_name,
                    "target_card": target["name"],
                    "removal_type": removal_type,
                })

    df = pd.DataFrame(edges)
    log.info("Built %d removal edges", len(df))
    return df


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full pipeline: load cards → build counter/removal edges → save."""
    log.info("Loading cards from %s", CARDS_PARQUET)
    cards_df = pd.read_parquet(CARDS_PARQUET)
    log.info("Loaded %d cards", len(cards_df))

    counter_df = build_counter_edges(cards_df)
    removal_df = build_removal_edges(cards_df)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    counter_df.to_parquet(COUNTER_EDGES_PATH, index=False)
    log.info("Saved %d counter edges to %s", len(counter_df), COUNTER_EDGES_PATH)

    removal_df.to_parquet(REMOVAL_EDGES_PATH, index=False)
    log.info("Saved %d removal edges to %s", len(removal_df), REMOVAL_EDGES_PATH)

    return counter_df, removal_df


if __name__ == "__main__":
    run()
