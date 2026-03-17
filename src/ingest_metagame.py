"""Phase 1: Scrape weekly metagame snapshots from MTGGoldfish.

Collects archetype-level data:
- Meta share percentage per archetype per week
- Win rate (where available)
- Trend direction (rising/falling/stable)
- Color identity per archetype

Outputs: metagame.parquet
"""

import logging
import re
import time
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import METAGAME_PARQUET, DATA_PROCESSED, SCRAPE_DELAY

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

GOLDFISH_BASE = "https://www.mtggoldfish.com"
METAGAME_URL = f"{GOLDFISH_BASE}/metagame/standard/full"

HEADERS = {
    "User-Agent": "MTG-Graph-Research/1.0 (academic project; respectful scraping)",
    "Accept": "text/html,application/xhtml+xml",
}

# Color abbreviations used in MTGGoldfish archetype names
COLOR_MAP = {
    "Mono-White": "W", "Mono-Blue": "U", "Mono-Black": "B",
    "Mono-Red": "R", "Mono-Green": "G",
    "Azorius": "WU", "Dimir": "UB", "Rakdos": "BR",
    "Gruul": "RG", "Selesnya": "WG", "Orzhov": "WB",
    "Izzet": "UR", "Golgari": "BG", "Boros": "WR",
    "Simic": "UG", "Esper": "WUB", "Grixis": "UBR",
    "Jund": "BRG", "Naya": "WRG", "Bant": "WUG",
    "Abzan": "WBG", "Jeskai": "WUR", "Sultai": "UBG",
    "Mardu": "WBR", "Temur": "URG",
    "Domain": "WUBRG", "5c": "WUBRG", "Five-Color": "WUBRG",
}


def infer_color_identity(archetype_name: str) -> str:
    """Infer color identity from archetype name."""
    for prefix, colors in COLOR_MAP.items():
        if prefix.lower() in archetype_name.lower():
            return colors
    return ""


def scrape_current_metagame() -> list[dict]:
    """Scrape the current Standard metagame page from MTGGoldfish."""
    log.info("Fetching metagame from %s", METAGAME_URL)

    time.sleep(SCRAPE_DELAY)
    resp = requests.get(METAGAME_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    snapshot_date = datetime.now().strftime("%Y-%m-%d")
    archetypes = []

    # MTGGoldfish metagame page has tiles/rows for each archetype
    # Look for the archetype containers
    tiles = soup.select(
        ".archetype-tile, .metagame-decks-item, tr.deck-tile, div[class*='archetype']"
    )

    if not tiles:
        # Fallback: look for any structure with percentage data
        tiles = soup.find_all("div", class_=re.compile(r"archetype|metagame|deck"))

    for tile in tiles:
        name_el = tile.select_one(
            ".deck-price-paper a, .archetype-name a, a[href*='/archetype/'], span.deck-name"
        )
        if not name_el:
            name_el = tile.find("a")
        if not name_el:
            continue

        name = name_el.get_text(strip=True)
        if not name or len(name) < 3:
            continue

        # Find meta share percentage
        meta_share = None
        pct_match = re.search(r"([\d.]+)\s*%", tile.get_text())
        if pct_match:
            meta_share = float(pct_match.group(1))

        # Find deck count or additional info
        deck_count = None
        count_match = re.search(r"(\d+)\s*deck", tile.get_text(), re.I)
        if count_match:
            deck_count = int(count_match.group(1))

        archetypes.append({
            "snapshot_date": snapshot_date,
            "archetype": name,
            "meta_share_pct": meta_share,
            "deck_count": deck_count,
            "color_identity": infer_color_identity(name),
            "source": "mtggoldfish",
        })

    log.info("Parsed %d archetypes from current metagame.", len(archetypes))
    return archetypes


def scrape_metagame_history(weeks_back: int = 12) -> list[dict]:
    """Scrape multiple weeks of metagame data.

    MTGGoldfish archives are limited, so we scrape what's available
    and build historical data over time through repeated runs.
    """
    all_data = []

    # Current metagame snapshot
    current = scrape_current_metagame()
    all_data.extend(current)

    # Try to get historical pages if available
    # MTGGoldfish sometimes has date-parameterized URLs
    # For now, we capture the current state and accumulate over time
    log.info(
        "Captured %d archetype entries. "
        "Historical data will accumulate with repeated runs.",
        len(all_data),
    )

    return all_data


def run(weeks_back: int = 12) -> pd.DataFrame:
    """Full pipeline: scrape → normalize → save."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    snapshots = scrape_metagame_history(weeks_back=weeks_back)

    df = pd.DataFrame(snapshots)

    # If we have prior data, append rather than overwrite
    if METAGAME_PARQUET.exists():
        existing = pd.read_parquet(METAGAME_PARQUET)
        df = pd.concat([existing, df], ignore_index=True)
        # Deduplicate on (snapshot_date, archetype)
        df = df.drop_duplicates(subset=["snapshot_date", "archetype"], keep="last")

    df.to_parquet(METAGAME_PARQUET, index=False)
    log.info("Saved %d metagame rows to %s", len(df), METAGAME_PARQUET)
    return df


if __name__ == "__main__":
    run()
