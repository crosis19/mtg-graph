"""Phase 1: Scrape metagame snapshots and decklists from MTGGoldfish.

Collects:
  - Archetype-level metagame data (share %, deck count, color identity)
  - Per-card decklists for the top N archetypes by meta share

Uses https://www.mtggoldfish.com/metagame/standard#paper which defaults to
30-day paper metagame data. Each archetype tile links to an archetype page
with a "Card Breakdown" section showing average copies and inclusion rate
per card.

Fallback: If Card Breakdown parsing fails, downloads the text decklist
from the /deck/download/ endpoint.

Outputs:
  - metagame.parquet  (archetype-level stats)
  - decklists.parquet (per-card data for top N archetypes)
"""

import logging
import re
import time
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import (
    DATA_PROCESSED,
    DECK_NUM_ARCHETYPES,
    DECKLISTS_PARQUET,
    METAGAME_PARQUET,
    SCRAPE_DELAY,
    SUPPORTED_FORMATS,
    format_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

GOLDFISH_BASE = "https://www.mtggoldfish.com"

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
    "4c": "WUBR", "4C": "WUBR",
}


def infer_color_identity(archetype_name: str) -> str:
    """Infer color identity from archetype name."""
    for prefix, colors in COLOR_MAP.items():
        if prefix.lower() in archetype_name.lower():
            return colors
    return ""


def _fetch(url: str) -> BeautifulSoup:
    """Fetch a URL with rate limiting and return parsed soup."""
    time.sleep(SCRAPE_DELAY)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")


# ── Metagame page scraper ───────────────────────────────────────────────


def scrape_current_metagame(format_name: str = "standard") -> list[dict]:
    """Scrape the current metagame page from MTGGoldfish for a given format.

    Parses .archetype-tile elements from the 30-day default view.
    Extracts archetype names from .archetype-tile-title (not the card image
    label, which is a key card name, not the archetype name).

    Returns list of dicts with archetype name, meta share, deck count,
    color identity, and archetype page URL.
    """
    fmt_cfg = SUPPORTED_FORMATS[format_name]
    metagame_url = GOLDFISH_BASE + fmt_cfg["goldfish_path"]
    log.info("Fetching %s metagame from %s", format_name, metagame_url)
    soup = _fetch(metagame_url)

    snapshot_date = datetime.now().strftime("%Y-%m-%d")
    archetypes = []
    seen_urls = set()  # deduplicate by archetype URL

    tiles = soup.select(".archetype-tile")
    if not tiles:
        log.warning("No .archetype-tile elements found on page")
        return []

    log.info("Found %d archetype tiles", len(tiles))

    for tile in tiles:
        # Get archetype name from the title section (paper version)
        # This is the real archetype name, NOT the key card displayed as image
        name_el = tile.select_one(".archetype-tile-title .deck-price-paper a")
        if not name_el:
            # Fallback to any title link
            name_el = tile.select_one(".archetype-tile-title a")
        if not name_el:
            continue

        name = name_el.get_text(strip=True)
        archetype_url = name_el.get("href", "")

        if not name or len(name) < 3:
            continue

        # Deduplicate: some pages list the same archetype URL twice
        # (e.g., different card images but same archetype)
        url_key = archetype_url.split("#")[0]
        if url_key in seen_urls:
            continue
        seen_urls.add(url_key)

        # Get meta share percentage
        meta_share = None
        pct_el = tile.select_one(".metagame-percentage .archetype-tile-statistic-value")
        if pct_el:
            pct_match = re.search(r"([\d.]+)\s*%", pct_el.get_text())
            if pct_match:
                meta_share = float(pct_match.group(1))

        # Get deck count from the (N) after the percentage
        deck_count = None
        if pct_el:
            count_match = re.search(r"\((\d+)\)", pct_el.get_text())
            if count_match:
                deck_count = int(count_match.group(1))

        archetypes.append({
            "snapshot_date": snapshot_date,
            "archetype": name,
            "meta_share_pct": meta_share,
            "deck_count": deck_count,
            "color_identity": infer_color_identity(name),
            "archetype_url": archetype_url,
            "source": "mtggoldfish",
            "format": format_name,
        })

    log.info("Parsed %d unique archetypes from metagame page.", len(archetypes))
    return archetypes


# ── Archetype page scraper ──────────────────────────────────────────────


def scrape_archetype_cards(archetype_url: str) -> list[dict]:
    """Scrape the Card Breakdown from an archetype page.

    Returns list of dicts: card_name, avg_copies, inclusion_pct, board.
    Falls back to downloading the text decklist if Card Breakdown parsing fails.
    """
    if not archetype_url:
        log.warning("  No archetype URL provided, skipping")
        return []

    full_url = archetype_url
    if not archetype_url.startswith("http"):
        full_url = GOLDFISH_BASE + archetype_url

    # Use the paper version
    if "#" not in full_url:
        full_url += "#paper"

    log.info("  Fetching archetype page: %s", full_url)
    soup = _fetch(full_url)

    # Primary strategy: Download text decklist (clean, structured format)
    cards = _download_text_decklist(soup)
    if cards:
        log.info("    Parsed %d cards from text decklist download", len(cards))
        return cards

    # Fallback: Try parsing Card Breakdown HTML
    log.info("    Text download failed, trying Card Breakdown HTML...")
    cards = _parse_card_breakdown(soup)
    if cards:
        log.info("    Parsed %d cards from Card Breakdown", len(cards))
        return cards

    log.warning("    Failed to extract cards from archetype page")
    return []


def _parse_card_breakdown(soup: BeautifulSoup) -> list[dict]:
    """Parse the Card Breakdown section for avg copies and inclusion rates.

    The Card Breakdown typically has rows like:
      Card Name | 3.9 | 100%
    Split into mainboard sections (Creatures, Instants, etc.) and Sideboard.
    """
    cards = []
    board = "main"

    # Look for the card breakdown / metagame-deck-breakdown section
    breakdown = soup.select_one(
        ".archetype-breakdown, .metagame-breakdown, "
        "#tab-tabletop .archetype-breakdown, "
        "#tab-paper .archetype-breakdown"
    )

    if not breakdown:
        breakdown = soup

    rows = breakdown.select(
        "tr, .archetype-breakdown-card, "
        ".archetype-card-row, div[class*='card-row']"
    )

    for row in rows:
        text = row.get_text(" ", strip=True)

        # Detect sideboard section header
        if re.search(r"\bsideboard\b", text, re.I) and len(text) < 30:
            board = "side"
            continue

        # Try to extract: card name, quantity/avg copies, percentage
        card_link = row.select_one("a[href*='/price/']")
        if not card_link:
            card_link = row.select_one("a")

        if not card_link:
            continue

        card_name = card_link.get_text(strip=True)
        if not card_name or len(card_name) < 2:
            continue

        # Extract numbers from the row text
        numbers = re.findall(r"(\d+\.?\d*)", text.replace(card_name, "", 1))

        avg_copies = 0.0
        inclusion_pct = 1.0

        if len(numbers) >= 2:
            avg_copies = float(numbers[0])
            pct_val = float(numbers[1])
            inclusion_pct = pct_val / 100.0 if pct_val > 1 else pct_val
        elif len(numbers) == 1:
            avg_copies = float(numbers[0])

        if avg_copies > 0:
            cards.append({
                "card_name": card_name,
                "avg_copies": avg_copies,
                "inclusion_pct": inclusion_pct,
                "board": board,
            })

    return cards


def _download_text_decklist(soup: BeautifulSoup) -> list[dict]:
    """Download and parse a text decklist from the download endpoint.

    Looks for /deck/download/{id} links on the page, fetches the text file,
    and parses lines of the form "4 Card Name".
    """
    download_link = soup.select_one("a[href*='/deck/download/']")
    if not download_link:
        return []

    url = download_link["href"]
    if not url.startswith("http"):
        url = GOLDFISH_BASE + url

    time.sleep(SCRAPE_DELAY)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code != 200:
        return []

    text = resp.text.strip()
    cards = []
    board = "main"
    seen_cards = False  # Track if we've seen any cards yet

    for line in text.split("\n"):
        line = line.strip()

        # Blank line after mainboard cards = sideboard boundary
        if not line:
            if seen_cards and board == "main":
                board = "side"
            continue

        # Explicit sideboard header (some formats use this)
        if re.match(r"(?i)sideboard", line):
            board = "side"
            continue

        # Parse "N Card Name" pattern
        match = re.match(r"(\d+)\s+(.+)", line)
        if match:
            copies = int(match.group(1))
            name = match.group(2).strip()
            # Remove set codes like "(WOE)" at end
            name = re.sub(r"\s*\([A-Z0-9]+\)\s*$", "", name).strip()
            if name:
                seen_cards = True
                cards.append({
                    "card_name": name,
                    "avg_copies": float(copies),
                    "inclusion_pct": 1.0,  # Single deck — assume 100%
                    "board": board,
                })

    return cards


# ── Main pipeline ───────────────────────────────────────────────────────


def run(format_name: str = "standard") -> pd.DataFrame:
    """Full pipeline: scrape metagame + decklists for eligible archetypes."""
    fmt_cfg = SUPPORTED_FORMATS[format_name]
    num_archetypes = fmt_cfg["num_archetypes"]

    # Use format-namespaced paths
    meta_path = format_path(METAGAME_PARQUET, format_name)
    dl_path = format_path(DECKLISTS_PARQUET, format_name)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Scrape metagame overview
    snapshots = scrape_current_metagame(format_name)
    meta_df = pd.DataFrame(snapshots)

    # Append to existing data if available
    if meta_path.exists():
        existing = pd.read_parquet(meta_path)
        meta_df = pd.concat([existing, meta_df], ignore_index=True)
        meta_df = meta_df.drop_duplicates(subset=["snapshot_date", "archetype"], keep="last")

    meta_df.to_parquet(meta_path, index=False)
    log.info("Saved %d %s metagame rows to %s", len(meta_df), format_name, meta_path)

    # Step 2: Select archetypes by meta share (0 = all available)
    latest = (
        meta_df.sort_values("snapshot_date")
        .groupby("archetype")
        .last()
        .reset_index()
    )
    latest = latest.dropna(subset=["meta_share_pct"])
    if num_archetypes > 0:
        eligible = latest.nlargest(num_archetypes, "meta_share_pct")
    else:
        eligible = latest.sort_values("meta_share_pct", ascending=False)
    log.info("\n%d %s archetypes by meta share:", len(eligible), format_name)
    for _, row in eligible.sort_values("meta_share_pct", ascending=False).iterrows():
        log.info("  %5.1f%%  %s", row["meta_share_pct"], row["archetype"])

    if eligible.empty:
        log.warning("No archetypes found!")
        empty_dl = pd.DataFrame(columns=[
            "archetype", "card_name", "avg_copies", "inclusion_pct",
            "board", "snapshot_date", "format",
        ])
        empty_dl.to_parquet(dl_path, index=False)
        return meta_df

    # Step 3: Scrape Card Breakdown for each eligible archetype
    snapshot_date = datetime.now().strftime("%Y-%m-%d")
    decklist_records = []

    for _, row in eligible.iterrows():
        arch_name = row["archetype"]
        arch_url = row.get("archetype_url", "") or ""
        arch_url = str(arch_url).strip() if pd.notna(arch_url) else ""

        if not arch_url:
            log.warning("  No URL for %s, skipping", arch_name)
            continue

        cards = scrape_archetype_cards(arch_url)
        if not cards:
            log.warning("  No cards found for %s — archetype will have no deck data", arch_name)
            continue

        for card in cards:
            decklist_records.append({
                "archetype": arch_name,
                "card_name": card["card_name"],
                "avg_copies": card["avg_copies"],
                "inclusion_pct": card["inclusion_pct"],
                "board": card["board"],
                "snapshot_date": snapshot_date,
                "format": format_name,
            })

    dl_df = pd.DataFrame(decklist_records)

    # Append to existing if available
    if dl_path.exists():
        existing_dl = pd.read_parquet(dl_path)
        if set(dl_df.columns) == set(existing_dl.columns):
            dl_df = pd.concat([existing_dl, dl_df], ignore_index=True)
            dl_df = dl_df.drop_duplicates(
                subset=["snapshot_date", "archetype", "card_name", "board"],
                keep="last",
            )

    dl_df.to_parquet(dl_path, index=False)
    log.info("\nSaved %d %s decklist rows to %s", len(dl_df), format_name, dl_path)

    # Summary
    if not dl_df.empty:
        for arch in sorted(dl_df["archetype"].unique()):
            arch_cards = dl_df[dl_df["archetype"] == arch]
            main = len(arch_cards[arch_cards["board"] == "main"])
            side = len(arch_cards[arch_cards["board"] == "side"])
            log.info("  %s: %d mainboard + %d sideboard cards", arch, main, side)

    return meta_df


if __name__ == "__main__":
    run()
