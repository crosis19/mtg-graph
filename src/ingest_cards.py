"""Phase 1: Download and normalize Scryfall oracle card data for Standard-legal cards.

Downloads the oracle_cards bulk JSON from Scryfall, filters to Standard-legal cards,
and produces a cleaned parquet with the fields needed for graph construction.
"""

import json
import logging
import requests
import pandas as pd
from pathlib import Path

from src.config import (
    SCRYFALL_BULK_URL, ORACLE_CARDS_PATH, CARDS_PARQUET,
    DATA_RAW, DATA_PROCESSED,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Fields we extract from each Scryfall card object
KEEP_FIELDS = [
    "id", "name", "oracle_text", "mana_cost", "cmc", "type_line",
    "color_identity", "colors", "keywords", "set", "set_name",
    "released_at", "legalities", "rarity", "power", "toughness",
    "rulings_uri", "scryfall_uri",
]


def download_bulk_data(force: bool = False) -> Path:
    """Download Scryfall oracle_cards bulk JSON if not already cached."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    if ORACLE_CARDS_PATH.exists() and not force:
        log.info("Using cached bulk data at %s", ORACLE_CARDS_PATH)
        return ORACLE_CARDS_PATH

    log.info("Fetching bulk data catalog from Scryfall...")
    resp = requests.get(SCRYFALL_BULK_URL, timeout=30)
    resp.raise_for_status()
    catalog = resp.json()

    # Find the oracle_cards entry
    oracle_entry = next(
        (item for item in catalog["data"] if item["type"] == "oracle_cards"),
        None,
    )
    if oracle_entry is None:
        raise RuntimeError("Could not find oracle_cards in Scryfall bulk data catalog")

    download_url = oracle_entry["download_uri"]
    log.info("Downloading oracle_cards from %s ...", download_url)
    resp = requests.get(download_url, timeout=120, stream=True)
    resp.raise_for_status()

    with open(ORACLE_CARDS_PATH, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)

    size_mb = ORACLE_CARDS_PATH.stat().st_size / (1 << 20)
    log.info("Downloaded %.1f MB to %s", size_mb, ORACLE_CARDS_PATH)
    return ORACLE_CARDS_PATH


def _card_to_record(card: dict) -> dict | None:
    """Convert a Scryfall card object to a flat record dict, or None to skip."""
    # Filter: must be legal in Standard
    legalities = card.get("legalities", {})
    if legalities.get("standard") != "legal":
        return None

    # Skip tokens, emblems, art series, etc.
    layout = card.get("layout", "")
    if layout in ("token", "emblem", "art_series", "double_faced_token"):
        return None

    record = {}
    for field in KEEP_FIELDS:
        val = card.get(field)
        # Convert lists to pipe-separated strings for parquet compatibility
        if isinstance(val, list):
            val = "|".join(str(v) for v in val)
        elif isinstance(val, dict):
            val = json.dumps(val)
        record[field] = val

    return record


def load_and_filter(path: Path) -> pd.DataFrame:
    """Load oracle cards JSON and filter to Standard-legal cards."""
    log.info("Loading %s ...", path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    log.info("Total cards in bulk data: %d", len(raw))

    records = []
    for card in raw:
        record = _card_to_record(card)
        if record is not None:
            records.append(record)

    df = pd.DataFrame(records)
    log.info("Standard-legal cards from bulk data: %d", len(df))
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and add derived columns."""
    # Parse color identity from pipe-separated back to count
    df["color_count"] = df["color_identity"].apply(
        lambda x: len(x.split("|")) if pd.notna(x) and x else 0
    )

    # Extract keyword count
    df["keyword_count"] = df["keywords"].apply(
        lambda x: len(x.split("|")) if pd.notna(x) and x else 0
    )

    # Clean oracle text — replace None with empty string
    df["oracle_text"] = df["oracle_text"].fillna("")

    # Parse CMC to float
    df["cmc"] = pd.to_numeric(df["cmc"], errors="coerce").fillna(0.0)

    # Add supertype flags
    df["is_creature"] = df["type_line"].str.contains("Creature", na=False)
    df["is_instant"] = df["type_line"].str.contains("Instant", na=False)
    df["is_sorcery"] = df["type_line"].str.contains("Sorcery", na=False)
    df["is_land"] = df["type_line"].str.contains("Land", na=False)
    df["is_planeswalker"] = df["type_line"].str.contains("Planeswalker", na=False)
    df["is_enchantment"] = df["type_line"].str.contains("Enchantment", na=False)
    df["is_artifact"] = df["type_line"].str.contains("Artifact", na=False)

    return df


def run(force_download: bool = False) -> pd.DataFrame:
    """Full pipeline: download → filter → normalize → save."""
    path = download_bulk_data(force=force_download)
    df = load_and_filter(path)

    df = normalize(df)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CARDS_PARQUET, index=False)
    log.info("Saved %d cards to %s", len(df), CARDS_PARQUET)
    return df


if __name__ == "__main__":
    run()
