"""Phase 2b: Extract rules-accurate counter and removal edges between cards.

Parses Scryfall oracle text to identify:
- Counter relationships: cards that can counter other cards' spells
- Removal relationships: cards that can destroy/exile/damage/bounce/sacrifice

Applies MTG rules interactions:
- Hard blocks: indestructible, hexproof, shroud, protection, "can't be countered"
- Soft weights: ward (cost penalty), bounce (CMC-scaled), damage/toughness ratio,
  forced sacrifice (opponent chooses)

Outputs counter_edges.parquet and removal_edges.parquet.
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

# Each tuple: (regex pattern on oracle text, set of targetable type_line substrings).
# None means "any non-land spell".
COUNTER_PATTERNS = [
    (r"counter target spell", None),
    (r"counter target creature spell", {"Creature"}),
    (r"counter target noncreature spell", {"noncreature"}),
    (r"counter target instant or sorcery spell", {"Instant", "Sorcery"}),
    (r"counter target artifact spell", {"Artifact"}),
    (r"counter target enchantment spell", {"Enchantment"}),
    (r"counter target planeswalker spell", {"Planeswalker"}),
    (r"counter target activated or triggered ability", None),
]

# ── Removal patterns ───────────────────────────────────────────────────────

# Each tuple: (regex, target_types, removal_type, value_capture_group).
# value_capture_group: index of the regex group containing the numeric value
# (damage amount or toughness reduction). None if no value to capture.
REMOVAL_PATTERNS = [
    # Destroy targeted
    (r"destroy target creature", {"Creature"}, "destroy", None),
    (r"destroy target permanent", None, "destroy", None),
    (r"destroy target artifact", {"Artifact"}, "destroy", None),
    (r"destroy target enchantment", {"Enchantment"}, "destroy", None),
    (r"destroy target planeswalker", {"Planeswalker"}, "destroy", None),
    (r"destroy target artifact or enchantment", {"Artifact", "Enchantment"}, "destroy", None),
    (r"destroy target nonland permanent", {"nonland"}, "destroy", None),
    # Exile targeted
    (r"exile target creature", {"Creature"}, "exile", None),
    (r"exile target permanent", None, "exile", None),
    (r"exile target artifact", {"Artifact"}, "exile", None),
    (r"exile target enchantment", {"Enchantment"}, "exile", None),
    (r"exile target nonland permanent", {"nonland"}, "exile", None),
    # Board wipes — split by destroy vs exile for indestructible filtering
    (r"destroy all creatures", {"Creature"}, "board_wipe_destroy", None),
    (r"destroy all nonland permanents", {"nonland"}, "board_wipe_destroy", None),
    (r"exile all creatures", {"Creature"}, "board_wipe_exile", None),
    # Damage-based removal (capture damage amount for toughness comparison)
    (r"deals? (\d+) damage to (?:target creature|any target)", {"Creature"}, "damage", 1),
    (r"deals? (\d+) damage to each creature", {"Creature"}, "board_wipe_damage", 1),
    # -N/-N effects (capture toughness reduction)
    (r"target creature gets -\d+/-(\d+)", {"Creature"}, "minus_stats", 1),
    (r"(?:all|each) creatures? get -\d+/-(\d+)", {"Creature"}, "board_wipe_minus", 1),
    # Bounce
    (r"return target creature to its owner's hand", {"Creature"}, "bounce", None),
    (r"return target nonland permanent to its owner's hand", {"nonland"}, "bounce", None),
    # Forced sacrifice (opponent chooses — soft removal)
    (r"(?:each |target )?opponents? sacrifices? a creature", {"Creature"}, "sacrifice", None),
    (r"each player sacrifices? a creature", {"Creature"}, "sacrifice", None),
    (r"opponents? sacrifices? a creature or planeswalker",
     {"Creature", "Planeswalker"}, "sacrifice", None),
    (r"opponents? sacrifices? a permanent", None, "sacrifice", None),
    (r"opponents? sacrifices? a nonland permanent", {"nonland"}, "sacrifice", None),
    (r"(?:that|its) (?:player|controller) sacrifices? (?:it|that creature)",
     {"Creature"}, "sacrifice", None),
]

# Non-targeting removal types (bypass hexproof, shroud, protection)
NON_TARGETING_TYPES = frozenset({
    "board_wipe_destroy", "board_wipe_exile", "board_wipe_damage",
    "board_wipe_minus", "sacrifice",
})

# Removal types blocked by indestructible
BLOCKED_BY_INDESTRUCTIBLE = frozenset({
    "destroy", "board_wipe_destroy", "damage", "board_wipe_damage",
    "minus_stats", "board_wipe_minus",
})

# Color abbreviation to name mapping
_COLOR_NAMES = {"W": "white", "U": "blue", "B": "black", "R": "red", "G": "green"}


# ── Helper functions ──────────────────────────────────────────────────────


def _is_non_land_spell(type_line: str) -> bool:
    """Check if a card is a non-land spell (can be countered)."""
    return not pd.isna(type_line) and "Land" not in type_line


def _matches_target_types(type_line: str, target_types: set[str] | None) -> bool:
    """Check if a card's type line matches the targeting restriction."""
    if target_types is None:
        return True
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


def _has_keyword(keywords_str, keyword: str) -> bool:
    """Check if a keyword is in the pipe-separated keywords string."""
    if pd.isna(keywords_str) or not keywords_str:
        return False
    return keyword in keywords_str.split("|")


def _has_cant_be_countered(oracle_text: str) -> bool:
    """Check if this card itself can't be countered.

    Matches 'this spell can't be countered' and standalone 'can't be countered'
    at the start of oracle text. Avoids matching cards that grant uncounterability
    to other spells (e.g., 'Spells you control can't be countered').
    """
    if not oracle_text:
        return False
    text = oracle_text.lower()
    if "this spell can't be countered" in text:
        return True
    if "can't be countered" in text:
        # Check it's about this card, not granting to others
        # If the line starts with "can't be countered" it's about this card
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("can't be countered"):
                return True
    return False


def _parse_protection(oracle_text: str) -> set[str]:
    """Parse 'protection from X' clauses from oracle text.

    Returns a set of protected-from qualities: colors, types, or 'everything'.
    """
    if not oracle_text:
        return set()

    result = set()
    # Match "protection from X" — X can be a color, type, or "everything"
    for m in re.finditer(r"protection from ([\w\s,]+?)(?:\.|$|\n|\")", oracle_text, re.I):
        raw = m.group(1).strip().lower()
        # Handle "X and from Y" — e.g., "protection from white and from black"
        parts = re.split(r"\s+and\s+(?:from\s+)?", raw)
        for part in parts:
            part = part.strip()
            if part:
                result.add(part)
    return result


def _parse_ward_mana_value(oracle_text: str) -> float:
    """Parse Ward cost from oracle text and return estimated mana value.

    Returns 0 if no Ward found.
    """
    if not oracle_text:
        return 0.0

    # Try mana cost: Ward {N} or Ward {N}{N}
    m = re.search(r"[Ww]ard \{(\d+)\}", oracle_text)
    if m:
        return float(m.group(1))

    # Try complex mana: Ward {2}{U} etc. — sum the numbers
    m = re.search(r"[Ww]ard ((?:\{[^}]+\})+)", oracle_text)
    if m:
        cost_str = m.group(1)
        total = 0.0
        for symbol in re.findall(r"\{([^}]+)\}", cost_str):
            if symbol.isdigit():
                total += float(symbol)
            elif symbol in ("W", "U", "B", "R", "G"):
                total += 1.0
            elif symbol.startswith("Pay ") and "life" in symbol:
                life_m = re.search(r"(\d+)", symbol)
                total += float(life_m.group(1)) / 2.0 if life_m else 1.0
        if total > 0:
            return total

    # Life payment: Ward—Pay N life
    m = re.search(r"[Ww]ard.{0,3}[Pp]ay (\d+) life", oracle_text)
    if m:
        return float(m.group(1)) / 2.0

    # Discard: Ward—Discard a card
    if re.search(r"[Ww]ard.{0,3}[Dd]iscard", oracle_text):
        return 2.0

    # Sacrifice: Ward—Sacrifice ...
    if re.search(r"[Ww]ard.{0,3}[Ss]acrifice", oracle_text):
        return 2.0

    # Fallback for any other Ward cost
    if re.search(r"[Ww]ard", oracle_text):
        return 2.0

    return 0.0


def _bounce_weight(target_cmc: float) -> float:
    """Weight for bounce removal: scales with target's CMC.

    Bouncing a cheap card is weak (they recast easily), bouncing an
    expensive card is devastating (huge tempo loss).
    """
    return min(max(target_cmc, 0.0) / 6.0, 1.0)


def _damage_vs_toughness_weight(amount: float, toughness: float | None) -> float:
    """Weight for damage or -N/-N vs a creature's toughness.

    Returns 1.0 if damage >= toughness (kills it), otherwise a fraction.
    Returns 1.0 for non-numeric toughness (can't compare).
    """
    if toughness is None or toughness <= 0:
        return 1.0
    return min(amount / toughness, 1.0)


def _color_identity_set(ci_str) -> set[str]:
    """Convert pipe-separated color identity to a set of color names."""
    if pd.isna(ci_str) or not ci_str:
        return set()
    return {_COLOR_NAMES.get(c, c.lower()) for c in ci_str.split("|") if c}


def _protection_blocks_source(
    protection: set[str],
    source_colors: set[str],
    source_type_line: str,
) -> bool:
    """Check if a card's protection blocks a specific removal source.

    Protection from a color blocks all targeted effects from that color.
    Protection from a type blocks effects from that type.
    Protection from everything blocks all targeted effects.
    """
    if "everything" in protection:
        return True

    # Color protection: "protection from red" blocks red sources
    if source_colors & protection:
        return True

    # Multicolored protection
    if "multicolored" in protection and len(source_colors) >= 2:
        return True

    # Type protection: "protection from Demons" blocks Demon sources
    if source_type_line:
        tl_lower = source_type_line.lower()
        for prot in protection:
            # Handle plurals: "Demons" -> check "demon" in type_line
            prot_singular = prot.rstrip("s") if prot.endswith("s") else prot
            if prot_singular in tl_lower or prot in tl_lower:
                return True

    # Creature protection: "protection from creatures" blocks creature sources
    if "creatures" in protection and source_type_line and "Creature" in source_type_line:
        return True

    # Instant protection: "protection from instants" blocks instant sources
    if "instants" in protection and source_type_line and "Instant" in source_type_line:
        return True

    return False


def _build_card_protections(cards_df: pd.DataFrame) -> dict:
    """Pre-compute protection attributes for all cards.

    Returns {card_name: {indestructible, hexproof, shroud, protection,
    ward_mana_value, toughness}}.
    """
    protections = {}
    for _, card in cards_df.iterrows():
        name = card["name"]
        kw = card.get("keywords", "") or ""
        ot = card.get("oracle_text", "") or ""

        # Parse numeric toughness (None for non-numeric like "*")
        toughness = None
        raw_t = card.get("toughness")
        if raw_t is not None and not pd.isna(raw_t):
            try:
                toughness = float(raw_t)
            except (ValueError, TypeError):
                pass

        protections[name] = {
            "indestructible": _has_keyword(kw, "Indestructible"),
            "hexproof": _has_keyword(kw, "Hexproof"),
            "shroud": _has_keyword(kw, "Shroud"),
            "protection": _parse_protection(ot) if _has_keyword(kw, "Protection") else set(),
            "ward_mana_value": _parse_ward_mana_value(ot) if _has_keyword(kw, "Ward") else 0.0,
            "toughness": toughness,
        }

    # Log protection summary
    counts = {k: sum(1 for p in protections.values() if p.get(k)) for k in
              ["indestructible", "hexproof", "shroud"]}
    counts["protection"] = sum(1 for p in protections.values() if p.get("protection"))
    counts["ward"] = sum(1 for p in protections.values() if p.get("ward_mana_value", 0) > 0)
    log.info("  Card protections: %s", counts)

    return protections


# ── Edge builders ─────────────────────────────────────────────────────────


def build_counter_edges(cards_df: pd.DataFrame) -> pd.DataFrame:
    """Build directed counter edges with rules-accurate filtering.

    Filters out targets with "can't be countered" in oracle text.
    """
    log.info("Building counter edges...")

    # Pre-compute uncounterable cards
    uncounterable = set()
    for _, card in cards_df.iterrows():
        if _has_cant_be_countered(card.get("oracle_text", "") or ""):
            uncounterable.add(card["name"])
    log.info("  %d uncounterable cards (will be excluded from targets)", len(uncounterable))

    # Identify counter sources
    counter_sources = []
    for _, card in cards_df.iterrows():
        text = (card.get("oracle_text") or "").lower()
        for pattern, target_types in COUNTER_PATTERNS:
            if re.search(pattern, text):
                counter_sources.append((card["name"], target_types))
                break
    log.info("  Found %d counter source cards", len(counter_sources))

    # Build edges
    edges = []
    skipped_uncounterable = 0
    for source_name, target_types in counter_sources:
        for _, target in cards_df.iterrows():
            if target["name"] == source_name:
                continue
            if not _is_non_land_spell(target.get("type_line", "")):
                continue
            if target["name"] in uncounterable:
                skipped_uncounterable += 1
                continue
            if _matches_target_types(target.get("type_line", ""), target_types):
                edges.append({"source_card": source_name, "target_card": target["name"]})

    df = pd.DataFrame(edges)
    log.info("  Built %d counter edges (skipped %d uncounterable targets)",
             len(df), skipped_uncounterable)
    return df


def build_removal_edges(cards_df: pd.DataFrame) -> pd.DataFrame:
    """Build directed removal edges with rules-accurate filtering and weights.

    Applies:
    - Hard blocks: indestructible, hexproof, shroud, protection from [X]
    - Soft weights: ward, bounce (CMC-scaled), damage/toughness, sacrifice
    """
    log.info("Building removal edges...")

    protections = _build_card_protections(cards_df)

    # Pre-compute source card metadata for protection-from-[color] checks
    source_meta = {}
    for _, card in cards_df.iterrows():
        source_meta[card["name"]] = {
            "colors": _color_identity_set(card.get("color_identity")),
            "type_line": card.get("type_line", "") or "",
        }

    # Identify removal sources (with captured numeric values)
    removal_sources = []  # (card_name, target_types, removal_type, captured_value)
    for _, card in cards_df.iterrows():
        text = (card.get("oracle_text") or "").lower()
        for pattern, target_types, removal_type, value_group in REMOVAL_PATTERNS:
            m = re.search(pattern, text)
            if m:
                captured_value = None
                if value_group is not None:
                    try:
                        captured_value = float(m.group(value_group))
                    except (IndexError, ValueError, TypeError):
                        pass
                removal_sources.append(
                    (card["name"], target_types, removal_type, captured_value)
                )
                break
    log.info("  Found %d removal source cards", len(removal_sources))

    # Build edges with rules filtering
    edges = []
    stats = {"indestructible": 0, "hexproof_shroud": 0, "protection": 0}

    for source_name, target_types, removal_type, captured_value in removal_sources:
        src_meta = source_meta[source_name]
        is_targeted = removal_type not in NON_TARGETING_TYPES

        for _, target in cards_df.iterrows():
            if target["name"] == source_name:
                continue
            target_tl = target.get("type_line", "")
            if not _matches_target_types(target_tl, target_types):
                continue

            prot = protections.get(target["name"], {})

            # ── Hard blocks ──

            # Indestructible blocks destroy, damage, -N/-N (but not exile/bounce/sacrifice)
            if prot.get("indestructible") and removal_type in BLOCKED_BY_INDESTRUCTIBLE:
                stats["indestructible"] += 1
                continue

            # Hexproof/Shroud block all targeted effects
            if (prot.get("hexproof") or prot.get("shroud")) and is_targeted:
                stats["hexproof_shroud"] += 1
                continue

            # Protection from [X] blocks targeted from matching sources
            if prot.get("protection") and is_targeted:
                if _protection_blocks_source(
                    prot["protection"], src_meta["colors"], src_meta["type_line"],
                ):
                    stats["protection"] += 1
                    continue

            # ── Soft weights ──
            weight = 1.0

            # Bounce: scale by target CMC (cheap targets → weak removal)
            if removal_type == "bounce":
                weight *= _bounce_weight(float(target.get("cmc", 0) or 0))

            # Damage vs toughness
            if removal_type in ("damage", "board_wipe_damage") and captured_value is not None:
                weight *= _damage_vs_toughness_weight(captured_value, prot.get("toughness"))

            # -N/-N vs toughness
            if removal_type in ("minus_stats", "board_wipe_minus") and captured_value is not None:
                weight *= _damage_vs_toughness_weight(captured_value, prot.get("toughness"))

            # Forced sacrifice: flat 0.5 (opponent chooses)
            if removal_type == "sacrifice":
                weight *= 0.5

            # Ward: reduce weight for targeted effects
            ward_mv = prot.get("ward_mana_value", 0)
            if ward_mv > 0 and is_targeted:
                weight *= 1.0 / (1.0 + ward_mv)

            edges.append({
                "source_card": source_name,
                "target_card": target["name"],
                "removal_type": removal_type,
                "removal_weight": round(weight, 4),
            })

    df = pd.DataFrame(edges)
    log.info("  Built %d removal edges", len(df))
    log.info("  Edges blocked: indestructible=%d, hexproof/shroud=%d, protection=%d",
             stats["indestructible"], stats["hexproof_shroud"], stats["protection"])

    if not df.empty:
        # Weight distribution summary
        log.info("  Weight distribution: mean=%.3f, min=%.3f, max=%.3f",
                 df["removal_weight"].mean(), df["removal_weight"].min(),
                 df["removal_weight"].max())
        for rt in sorted(df["removal_type"].unique()):
            sub = df[df["removal_type"] == rt]
            log.info("    %s: %d edges, mean_weight=%.3f", rt, len(sub), sub["removal_weight"].mean())

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
