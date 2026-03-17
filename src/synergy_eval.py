"""Phase 2: Evaluate synergy edge quality against known synergistic pairs.

Provides ground truth validation for the three synergy edge layers:
1. Keyword synergy (Layer 1)
2. Mechanical synergy (Layer 2)
3. Semantic synergy (Layer 3)

Uses a curated set of known-good synergy pairs and known-bad pairs
to compute precision, recall, and coverage metrics.
"""

import logging
from dataclasses import dataclass

import pandas as pd

from src.config import (
    KEYWORD_MATRIX_PATH,
    MECHANICAL_EDGES_PATH,
    SEMANTIC_EDGES_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# Known synergistic card pairs — ground truth for evaluation
# Uses current Standard-legal cards confirmed in the Scryfall data
KNOWN_SYNERGIES = [
    # Prowess + instants/sorceries (mechanical: spell_matters)
    ("Queen Brahne", "Witch's Mark", "prowess_spell_trigger"),
    ("Heartfire Immolator", "Torch the Tower", "prowess_spell_trigger"),
    ("Lilah, Undefeated Slickshot", "End-Blaze Epiphany", "prowess_spell_trigger"),

    # Deathtouch + First Strike (keyword: combat_synergy)
    ("South Wind Avatar", "Cosmic Spider-Man", "deathtouch_first_strike"),
    ("South Wind Avatar", "Brightglass Gearhulk", "deathtouch_first_strike"),
    ("South Wind Avatar", "Breeches, Eager Pillager", "deathtouch_first_strike"),

    # Landfall + land search (mechanical: land_matters)
    ("Seedship Agrarian", "Escape Tunnel", "landfall_ramp"),
    ("Seedship Agrarian", "Fabled Passage", "landfall_ramp"),
    ("Territorial Bruntar", "Solemn Simulacrum", "landfall_ramp"),

    # Death triggers + sacrifice (mechanical: death_trigger)
    ("Flesh Burrower", "Braids, Arisen Nightmare", "death_sacrifice"),

    # Flying + Lifelink (keyword: evasion_sustain)
    ("Fear of Infinity", "Screaming Phantom", "flying_keyword"),
]

# Known non-synergistic pairs — should NOT have synergy edges
# Cards from entirely different strategic axes with no shared mechanics
KNOWN_NON_SYNERGIES = [
    ("Escape Tunnel", "Cosmic Spider-Man", "no_shared_strategy"),
    ("Torch the Tower", "Fabled Passage", "no_shared_strategy"),
    ("Screaming Phantom", "Brightglass Gearhulk", "different_strategy"),
]


@dataclass
class EvalResult:
    """Evaluation metrics for a synergy edge set."""
    layer_name: str
    total_edges: int
    known_pairs_found: int
    known_pairs_total: int
    false_positives: int  # known non-synergies that appear as edges
    false_positives_total: int
    precision_on_known: float  # known_found / known_total
    false_positive_rate: float  # false_pos / false_pos_total


def _normalize_name(name: str) -> str:
    """Normalize card name for fuzzy matching."""
    return name.lower().strip().split(",")[0].split("//")[0].strip()


def _load_edge_pairs(path) -> set[tuple[str, str]]:
    """Load a synergy edge file and return normalized card pairs."""
    if not path.exists():
        return set()
    df = pd.read_parquet(path)
    pairs = set()
    for _, row in df.iterrows():
        a = _normalize_name(row["card_a"])
        b = _normalize_name(row["card_b"])
        pairs.add((min(a, b), max(a, b)))
    return pairs


def evaluate_layer(
    layer_name: str,
    edge_pairs: set[tuple[str, str]],
) -> EvalResult:
    """Evaluate a synergy layer against known ground truth."""
    # Check known synergies
    found = 0
    for card_a, card_b, _ in KNOWN_SYNERGIES:
        a = _normalize_name(card_a)
        b = _normalize_name(card_b)
        pair = (min(a, b), max(a, b))
        if pair in edge_pairs:
            found += 1

    # Check known non-synergies (should NOT be edges)
    false_pos = 0
    for card_a, card_b, _ in KNOWN_NON_SYNERGIES:
        a = _normalize_name(card_a)
        b = _normalize_name(card_b)
        pair = (min(a, b), max(a, b))
        if pair in edge_pairs:
            false_pos += 1

    total_known = len(KNOWN_SYNERGIES)
    total_non = len(KNOWN_NON_SYNERGIES)

    result = EvalResult(
        layer_name=layer_name,
        total_edges=len(edge_pairs),
        known_pairs_found=found,
        known_pairs_total=total_known,
        false_positives=false_pos,
        false_positives_total=total_non,
        precision_on_known=found / total_known if total_known > 0 else 0.0,
        false_positive_rate=false_pos / total_non if total_non > 0 else 0.0,
    )

    return result


def run() -> dict[str, EvalResult]:
    """Evaluate all synergy layers."""
    layers = {
        "keyword": KEYWORD_MATRIX_PATH,
        "mechanical": MECHANICAL_EDGES_PATH,
        "semantic_synergy": SEMANTIC_EDGES_PATH,
    }

    results = {}

    for name, path in layers.items():
        pairs = _load_edge_pairs(path)
        if not pairs:
            log.warning("No edges found for layer '%s' at %s", name, path)

        result = evaluate_layer(name, pairs)
        results[name] = result

        log.info(
            "[%s] %d total edges | Known synergies found: %d/%d (%.1f%%) | "
            "False positives: %d/%d",
            name,
            result.total_edges,
            result.known_pairs_found,
            result.known_pairs_total,
            result.precision_on_known * 100,
            result.false_positives,
            result.false_positives_total,
        )

    # Combined coverage: how many known synergies are found by ANY layer
    all_pairs = set()
    for path in layers.values():
        all_pairs |= _load_edge_pairs(path)

    combined = evaluate_layer("combined", all_pairs)
    results["combined"] = combined

    log.info(
        "[COMBINED] %d total unique edges | Known synergies found: %d/%d (%.1f%%) | "
        "False positives: %d/%d",
        combined.total_edges,
        combined.known_pairs_found,
        combined.known_pairs_total,
        combined.precision_on_known * 100,
        combined.false_positives,
        combined.false_positives_total,
    )

    return results


if __name__ == "__main__":
    run()
