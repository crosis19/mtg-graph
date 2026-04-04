"""Phase 1: Download and normalize Scryfall oracle card data.

Downloads the oracle_cards bulk JSON from Scryfall, filters to cards legal in any
active format (Standard, Pioneer, etc.), and produces a cleaned parquet with the
fields needed for graph construction.
"""

import json
import logging
import requests
import pandas as pd
from pathlib import Path

from src.config import (
    SCRYFALL_BULK_URL, ORACLE_CARDS_PATH, CARDS_PARQUET,
    DATA_RAW, DATA_PROCESSED, ACTIVE_FORMATS, SUPPORTED_FORMATS,
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


def _card_to_record(card: dict, formats: list[str] | None = None) -> dict | None:
    """Convert a Scryfall card object to a flat record dict, or None to skip.

    Keeps a card if it is legal in ANY of the specified formats.
    Adds per-format legality boolean columns (e.g. legal_standard, legal_pioneer).
    """
    if formats is None:
        formats = ACTIVE_FORMATS

    legalities = card.get("legalities", {})
    legality_keys = [SUPPORTED_FORMATS[f]["legality_key"] for f in formats]

    # Keep card if legal in at least one active format
    if not any(legalities.get(k) == "legal" for k in legality_keys):
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

    # Per-format legality flags
    for fmt in formats:
        key = SUPPORTED_FORMATS[fmt]["legality_key"]
        record[f"legal_{fmt}"] = legalities.get(key) == "legal"

    # For multi-face cards (transform, adventure, modal DFC), Scryfall puts
    # oracle_text and type_line on card_faces, not the top level.  Merge both
    # faces so downstream embeddings and interaction edges see the full card.
    faces = card.get("card_faces")
    if faces and len(faces) >= 2:
        if not record.get("oracle_text"):
            record["oracle_text"] = " // ".join(
                f.get("oracle_text", "") for f in faces
            )
        if not record.get("type_line"):
            record["type_line"] = " // ".join(
                f.get("type_line", "") for f in faces
            )
        if not record.get("mana_cost"):
            record["mana_cost"] = " // ".join(
                f.get("mana_cost", "") for f in faces
            )

    return record


def load_and_filter(path: Path, formats: list[str] | None = None) -> pd.DataFrame:
    """Load oracle cards JSON and filter to cards legal in any active format."""
    if formats is None:
        formats = ACTIVE_FORMATS

    log.info("Loading %s ...", path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    log.info("Total cards in bulk data: %d", len(raw))

    records = []
    for card in raw:
        record = _card_to_record(card, formats=formats)
        if record is not None:
            records.append(record)

    df = pd.DataFrame(records)
    log.info("Cards legal in %s: %d", formats, len(df))
    for fmt in formats:
        col = f"legal_{fmt}"
        if col in df.columns:
            log.info("  %s-legal: %d", fmt, df[col].sum())
    return df


def _parse_mana_generation(oracle_text: str, type_line: str) -> dict[str, bool]:
    """Detect which colors of mana a card can produce from oracle text and land types."""
    import re

    result = {
        "gen_white": False, "gen_blue": False, "gen_black": False,
        "gen_red": False, "gen_green": False, "gen_colorless": False,
    }
    text = (oracle_text or "").lower()
    tl = (type_line or "").lower()

    # Basic land types in type line (e.g., "Basic Land — Plains" or dual "Land — Forest Island")
    if "plains" in tl:
        result["gen_white"] = True
    if "island" in tl:
        result["gen_blue"] = True
    if "swamp" in tl:
        result["gen_black"] = True
    if "mountain" in tl:
        result["gen_red"] = True
    if "forest" in tl:
        result["gen_green"] = True

    # Oracle text patterns: "{T}: Add {W}", "{T}: Add {C}", etc.
    mana_symbols = re.findall(r"add\s+\{([WUBRGC])\}", text)
    for sym in mana_symbols:
        mapping = {"W": "gen_white", "U": "gen_blue", "B": "gen_black",
                    "R": "gen_red", "G": "gen_green", "C": "gen_colorless"}
        if sym in mapping:
            result[mapping[sym]] = True

    # "Add one mana of any color" / "add one mana of any type"
    if re.search(r"add\s+.*\bany\s+(color|type)\b", text):
        for key in result:
            if key != "gen_colorless":
                result[key] = True

    # "Add {C}" also appears as "add one colorless mana"
    if re.search(r"add\s+.*\bcolorless\b", text):
        result["gen_colorless"] = True

    return result


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and add derived columns."""
    # Clean oracle text — replace None with empty string
    df["oracle_text"] = df["oracle_text"].fillna("")

    # Parse CMC to float and normalize
    df["cmc"] = pd.to_numeric(df["cmc"], errors="coerce").fillna(0.0)
    max_cmc = df["cmc"].max()
    df["cmc_norm"] = df["cmc"] / max_cmc if max_cmc > 0 else 0.0

    # Card type flags
    df["is_creature"] = df["type_line"].str.contains("Creature", na=False)
    df["is_instant"] = df["type_line"].str.contains("Instant", na=False)
    df["is_sorcery"] = df["type_line"].str.contains("Sorcery", na=False)
    df["is_land"] = df["type_line"].str.contains("Land", na=False)
    df["is_planeswalker"] = df["type_line"].str.contains("Planeswalker", na=False)
    df["is_enchantment"] = df["type_line"].str.contains("Enchantment", na=False)
    df["is_artifact"] = df["type_line"].str.contains("Artifact", na=False)
    df["is_battle"] = df["type_line"].str.contains("Battle", na=False)

    # Rarity encoding: common/basic=0.25, uncommon=0.5, rare=0.75, mythic=1.0
    rarity_map = {"common": 0.25, "basic": 0.25, "uncommon": 0.5, "rare": 0.75, "mythic": 1.0}
    df["rarity_enc"] = df["rarity"].map(rarity_map).fillna(0.25)

    # Color identity flags (parsed from pipe-separated color_identity field)
    def _parse_colors(ci_str):
        colors = set(ci_str.split("|")) if pd.notna(ci_str) and ci_str else set()
        return {
            "is_white": "W" in colors,
            "is_blue": "U" in colors,
            "is_black": "B" in colors,
            "is_red": "R" in colors,
            "is_green": "G" in colors,
            "is_colorless": len(colors) == 0,
        }

    color_flags = df["color_identity"].apply(_parse_colors).apply(pd.Series)
    df = pd.concat([df, color_flags], axis=1)

    # Mana generation flags (from oracle text + land type)
    mana_flags = df.apply(
        lambda row: _parse_mana_generation(row["oracle_text"], row["type_line"]),
        axis=1,
    ).apply(pd.Series)
    df = pd.concat([df, mana_flags], axis=1)

    # Convert boolean columns to float for downstream tensor construction
    bool_cols = [
        "is_creature", "is_instant", "is_sorcery", "is_land", "is_planeswalker",
        "is_enchantment", "is_artifact", "is_battle",
        "is_white", "is_blue", "is_black", "is_red", "is_green", "is_colorless",
        "gen_white", "gen_blue", "gen_black", "gen_red", "gen_green", "gen_colorless",
    ]
    for col in bool_cols:
        df[col] = df[col].astype(float)

    return df


def run(force_download: bool = False, formats: list[str] | None = None) -> pd.DataFrame:
    """Full pipeline: download → filter → normalize → save."""
    path = download_bulk_data(force=force_download)
    df = load_and_filter(path, formats=formats)

    df = normalize(df)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CARDS_PARQUET, index=False)
    log.info("Saved %d cards to %s", len(df), CARDS_PARQUET)
    return df


if __name__ == "__main__":
    run()
