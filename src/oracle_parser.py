"""Phase 2, Layer 2: Mechanical synergy extraction from oracle text.

Uses regex pattern matching to identify trigger/enable relationships between cards.
When Card A has a trigger pattern that Card B enables, draw a mechanical synergy edge.

Example:
  Card A: "Whenever you cast an instant or sorcery..." (trigger: spell-matters)
  Card B: "Counter target spell" (enabler: is an instant)
  → Mechanical synergy edge: Card B enables Card A's trigger

Outputs: mechanical_edges.parquet
"""

import logging
import re
from dataclasses import dataclass
from itertools import product

import pandas as pd

from src.config import CARDS_PARQUET, MECHANICAL_EDGES_PATH, DATA_PROCESSED

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


@dataclass
class TriggerPattern:
    """A pattern that detects a trigger condition in oracle text."""
    name: str
    pattern: re.Pattern
    category: str


@dataclass
class EnablerPattern:
    """A pattern that detects a card enabling a specific trigger category."""
    name: str
    pattern: re.Pattern  # Matches oracle text
    type_pattern: re.Pattern | None  # Matches type line
    category: str  # Which trigger category this enables


# === TRIGGER PATTERNS ===
# These detect "whenever X happens" conditions on cards
TRIGGERS = [
    TriggerPattern(
        "spell_cast",
        re.compile(r"whenever you cast (?:a|an) (?:instant|sorcery|noncreature)", re.I),
        "spell_matters",
    ),
    TriggerPattern(
        "any_spell_cast",
        re.compile(r"whenever you cast (?:a|an) (?:spell|nonland)", re.I),
        "spell_cast",
    ),
    TriggerPattern(
        "creature_dies",
        re.compile(r"whenever (?:a|another) creature (?:you control )?dies", re.I),
        "death_trigger",
    ),
    TriggerPattern(
        "self_dies",
        re.compile(r"when .+ dies", re.I),
        "death_trigger",
    ),
    TriggerPattern(
        "draw_trigger",
        re.compile(r"whenever you draw (?:a|your second|one or more) card", re.I),
        "draw_matters",
    ),
    TriggerPattern(
        "landfall",
        re.compile(r"landfall|whenever a land enters the battlefield under your control", re.I),
        "land_matters",
    ),
    TriggerPattern(
        "etb_trigger",
        re.compile(r"whenever (?:a|another) .+ enters the battlefield", re.I),
        "etb_matters",
    ),
    TriggerPattern(
        "self_etb",
        re.compile(r"when .+ enters the battlefield", re.I),
        "etb_matters",
    ),
    TriggerPattern(
        "attack_trigger",
        re.compile(r"whenever .+ attacks", re.I),
        "attack_matters",
    ),
    TriggerPattern(
        "gain_life",
        re.compile(r"whenever you gain life", re.I),
        "lifegain_matters",
    ),
    TriggerPattern(
        "counter_placed",
        re.compile(r"whenever (?:a|one or more) .+counter.+ (?:is|are) (?:placed|put) on", re.I),
        "counter_matters",
    ),
    TriggerPattern(
        "token_created",
        re.compile(r"whenever (?:you create|a token)", re.I),
        "token_matters",
    ),
    TriggerPattern(
        "discard_trigger",
        re.compile(r"whenever (?:a player|you|an opponent) discards", re.I),
        "discard_matters",
    ),
    TriggerPattern(
        "graveyard_trigger",
        re.compile(
            r"whenever a (?:card|creature) (?:is put into|enters) "
            r"(?:a|your) graveyard",
            re.I,
        ),
        "graveyard_matters",
    ),
]

# === ENABLER PATTERNS ===
# These detect cards that produce effects matching trigger categories
ENABLERS = [
    # Spell matters enablers (instants/sorceries that do useful things)
    EnablerPattern(
        "removal_spell",
        re.compile(r"(?:destroy|exile|deal \d+ damage to|return .+ to .+ hand)", re.I),
        re.compile(r"Instant|Sorcery", re.I),
        "spell_matters",
    ),
    EnablerPattern(
        "counterspell",
        re.compile(r"counter target spell", re.I),
        re.compile(r"Instant", re.I),
        "spell_matters",
    ),
    EnablerPattern(
        "cantrip",
        re.compile(r"draw (?:a|two|three) card", re.I),
        re.compile(r"Instant|Sorcery", re.I),
        "spell_matters",
    ),
    EnablerPattern(
        "generic_instant_sorcery",
        None,  # No oracle text pattern — just match type
        re.compile(r"Instant|Sorcery", re.I),
        "spell_cast",
    ),

    # Death trigger enablers (sacrifice outlets, removal)
    EnablerPattern(
        "sacrifice_outlet",
        re.compile(r"sacrifice (?:a|another) creature", re.I),
        None,
        "death_trigger",
    ),
    EnablerPattern(
        "board_wipe",
        re.compile(r"destroy all creatures|exile all creatures|all creatures get -", re.I),
        None,
        "death_trigger",
    ),
    EnablerPattern(
        "creature_removal",
        re.compile(r"destroy target creature|exile target creature", re.I),
        None,
        "death_trigger",
    ),

    # Draw matters enablers
    EnablerPattern(
        "draw_engine",
        re.compile(r"draw (?:a|two|three|\d+) card", re.I),
        None,
        "draw_matters",
    ),

    # Land matters enablers (ramp, fetch)
    EnablerPattern(
        "land_search",
        re.compile(r"search your library for (?:a|up to).* land", re.I),
        None,
        "land_matters",
    ),
    EnablerPattern(
        "extra_land",
        re.compile(r"you may play an additional land", re.I),
        None,
        "land_matters",
    ),

    # ETB matters enablers (blink, bounce)
    EnablerPattern(
        "blink",
        re.compile(r"exile .+ then return .+ to the battlefield|flicker", re.I),
        None,
        "etb_matters",
    ),
    EnablerPattern(
        "bounce_own",
        re.compile(r"return .+ you control .+ to .+ hand", re.I),
        None,
        "etb_matters",
    ),
    EnablerPattern(
        "token_producer_for_etb",
        re.compile(r"create (?:a|two|three|\d+) .+ (?:creature )?token", re.I),
        None,
        "etb_matters",
    ),

    # Lifegain enablers
    EnablerPattern(
        "lifegain_source",
        re.compile(r"you gain \d+ life|lifelink", re.I),
        None,
        "lifegain_matters",
    ),

    # Counter enablers
    EnablerPattern(
        "counter_placer",
        re.compile(r"put (?:a|one|two|\d+) \+1/\+1 counter", re.I),
        None,
        "counter_matters",
    ),

    # Token enablers
    EnablerPattern(
        "token_maker",
        re.compile(r"create (?:a|two|three|\d+) .+ token", re.I),
        None,
        "token_matters",
    ),

    # Discard enablers
    EnablerPattern(
        "discard_source",
        re.compile(r"(?:target (?:player|opponent) )?discards? (?:a|two|\d+) card", re.I),
        None,
        "discard_matters",
    ),
    EnablerPattern(
        "self_discard",
        re.compile(r"discard (?:a|two|\d+) card", re.I),
        None,
        "discard_matters",
    ),

    # Graveyard enablers (mill, self-mill)
    EnablerPattern(
        "mill",
        re.compile(r"(?:target player )?(?:mills?|puts?) .+ cards? .+ into .+ graveyard", re.I),
        None,
        "graveyard_matters",
    ),

    # Protection / interaction
    EnablerPattern(
        "protection",
        re.compile(r"counter target spell|can't be countered", re.I),
        None,
        "anti_interaction",
    ),
    EnablerPattern(
        "recursion",
        re.compile(r"return .+ from (?:your|a) graveyard", re.I),
        None,
        "recursion",
    ),
]


def detect_triggers(oracle_text: str) -> list[tuple[str, str]]:
    """Return list of (trigger_name, category) found in oracle text."""
    if not oracle_text:
        return []
    return [
        (t.name, t.category) for t in TRIGGERS
        if t.pattern.search(oracle_text)
    ]


def detect_enablers(oracle_text: str, type_line: str) -> list[tuple[str, str]]:
    """Return list of (enabler_name, category) found in oracle text / type line."""
    results = []
    for e in ENABLERS:
        text_match = e.pattern.search(oracle_text) if e.pattern and oracle_text else False
        type_match = e.type_pattern.search(type_line) if e.type_pattern and type_line else False

        if e.pattern is None:
            # Type-only enabler
            if type_match:
                results.append((e.name, e.category))
        elif e.type_pattern is None:
            # Text-only enabler
            if text_match:
                results.append((e.name, e.category))
        else:
            # Both must match
            if text_match and type_match:
                results.append((e.name, e.category))

    return results


def build_mechanical_edges(cards_df: pd.DataFrame) -> pd.DataFrame:
    """Build card-to-card mechanical synergy edges from trigger/enable patterns.

    For each card pair (A, B):
      If A has trigger in category C and B enables category C → edge A↔B
    """
    # Pre-compute triggers and enablers for all cards
    card_triggers: dict[str, list[tuple[str, str]]] = {}
    card_enablers: dict[str, list[tuple[str, str]]] = {}

    for _, row in cards_df.iterrows():
        name = row["name"]
        oracle = row.get("oracle_text", "") or ""
        type_line = row.get("type_line", "") or ""

        triggers = detect_triggers(oracle)
        enablers = detect_enablers(oracle, type_line)

        if triggers:
            card_triggers[name] = triggers
        if enablers:
            card_enablers[name] = enablers

    log.info(
        "Cards with triggers: %d, cards with enablers: %d",
        len(card_triggers), len(card_enablers),
    )

    # Build edges: for each (trigger_card, enabler_card) pair with matching categories
    edges = []
    seen_pairs = set()

    for trigger_card, triggers in card_triggers.items():
        trigger_cats = {cat for _, cat in triggers}

        for enabler_card, enablers in card_enablers.items():
            if trigger_card == enabler_card:
                continue

            pair_key = tuple(sorted([trigger_card, enabler_card]))
            if pair_key in seen_pairs:
                continue

            enabler_cats = {cat for _, cat in enablers}
            shared_cats = trigger_cats & enabler_cats

            if shared_cats:
                seen_pairs.add(pair_key)

                # Collect the specific trigger/enabler pairs
                trigger_names = [
                    name for name, cat in triggers if cat in shared_cats
                ]
                enabler_names = [
                    name for name, cat in enablers if cat in shared_cats
                ]

                edges.append({
                    "card_a": pair_key[0],
                    "card_b": pair_key[1],
                    "edge_type": "mechanical_synergy",
                    "shared_categories": "|".join(sorted(shared_cats)),
                    "trigger_patterns": "|".join(trigger_names),
                    "enabler_patterns": "|".join(enabler_names),
                    "synergy_count": len(shared_cats),
                })

    df = pd.DataFrame(edges)
    log.info("Found %d mechanical synergy edges.", len(df))
    return df


def run() -> pd.DataFrame:
    """Full pipeline: load cards → extract patterns → build edges → save."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    if not CARDS_PARQUET.exists():
        raise FileNotFoundError(
            f"Cards data not found at {CARDS_PARQUET}. Run ingest_cards.py first."
        )

    cards_df = pd.read_parquet(CARDS_PARQUET)
    edges_df = build_mechanical_edges(cards_df)

    edges_df.to_parquet(MECHANICAL_EDGES_PATH, index=False)
    log.info("Saved %d mechanical edges to %s", len(edges_df), MECHANICAL_EDGES_PATH)
    return edges_df


if __name__ == "__main__":
    run()
