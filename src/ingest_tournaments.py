"""Phase 1: Scrape tournament results and decklists from MTGTOP8.

Collects historical Standard tournament data including:
- Tournament metadata (name, date, format, player count)
- Top 8 decklists with archetype labels
- Individual card lists per deck

Outputs: tournaments.parquet and decklists.parquet
"""

import logging
import re
import time
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import (
    TOURNAMENTS_PARQUET, DECKLISTS_PARQUET,
    DATA_PROCESSED, SCRAPE_DELAY,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "https://www.mtgtop8.com"
STANDARD_FORMAT_CODE = "ST"

HEADERS = {
    "User-Agent": "MTG-Graph-Research/1.0 (academic project; respectful scraping)",
}


def _get_soup(url: str) -> BeautifulSoup:
    """Fetch a page with rate limiting and return parsed soup."""
    time.sleep(SCRAPE_DELAY)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")


def scrape_tournament_list(
    max_pages: int = 10,
    format_code: str = STANDARD_FORMAT_CODE,
) -> list[dict]:
    """Scrape the list of Standard tournaments from MTGTOP8."""
    tournaments = []

    for page in range(1, max_pages + 1):
        url = f"{BASE_URL}/format?f={format_code}&meta=&cp=&page={page}"
        log.info("Fetching tournament list page %d: %s", page, url)

        try:
            soup = _get_soup(url)
        except requests.RequestException as e:
            log.warning("Failed to fetch page %d: %s", page, e)
            break

        # MTGTOP8 lists tournaments in table rows with links to event pages
        event_links = soup.select("a[href*='event?e=']")
        if not event_links:
            log.info("No more tournaments found on page %d, stopping.", page)
            break

        for link in event_links:
            href = link.get("href", "")
            match = re.search(r"e=(\d+)", href)
            if not match:
                continue

            event_id = match.group(1)
            name = link.get_text(strip=True)

            # Try to find the date in nearby cells
            parent_row = link.find_parent("tr")
            date_str = None
            player_count = None
            if parent_row:
                cells = parent_row.find_all("td")
                for cell in cells:
                    text = cell.get_text(strip=True)
                    # Look for date pattern (dd/mm/yy or similar)
                    date_match = re.search(r"(\d{1,2}/\d{1,2}/\d{2,4})", text)
                    if date_match:
                        date_str = date_match.group(1)
                    # Look for player count
                    player_match = re.search(r"(\d+)\s*(?:players?|pl\.)", text, re.I)
                    if player_match:
                        player_count = int(player_match.group(1))

            tournaments.append({
                "event_id": event_id,
                "name": name,
                "date_raw": date_str,
                "player_count": player_count,
                "format": "Standard",
                "url": f"{BASE_URL}/event?e={event_id}&f={format_code}",
            })

        log.info("Found %d tournaments so far.", len(tournaments))

    # Deduplicate by event_id (same event may appear multiple times)
    seen = set()
    unique = []
    for t in tournaments:
        if t["event_id"] not in seen:
            seen.add(t["event_id"])
            unique.append(t)

    return unique


def scrape_event_page(event_id: str, format_code: str = STANDARD_FORMAT_CODE) -> dict:
    """Scrape an event page to get metadata and deck links.

    Returns dict with 'player_count', 'date', and 'decks' list of
    {deck_id, archetype, placement, player}.
    """
    url = f"{BASE_URL}/event?e={event_id}&f={format_code}"
    log.info("Scraping event page %s", event_id)

    try:
        soup = _get_soup(url)
    except requests.RequestException as e:
        log.warning("Failed to fetch event %s: %s", event_id, e)
        return {"decks": []}

    text = soup.get_text()

    # Extract player count from text like "384 players"
    player_match = re.search(r"(\d+)\s*players?", text)
    player_count = int(player_match.group(1)) if player_match else None

    # Find deck links with archetype names — pick those with non-empty text
    deck_links = soup.find_all("a", href=True)
    decks = []
    seen_deck_ids = set()

    for link in deck_links:
        href = link.get("href", "")
        if "e=" not in href or "d=" not in href:
            continue
        deck_match = re.search(r"d=(\d+)", href)
        if not deck_match:
            continue

        deck_id = deck_match.group(1)
        archetype = link.get_text(strip=True)

        # Skip empty text, arrows, and non-archetype links
        if not archetype or archetype in ("", "\u2192", "Switch to Visual") or len(archetype) < 3:
            continue
        if "switch" in href.lower() or "visual" in archetype.lower():
            continue

        if deck_id in seen_deck_ids:
            continue
        seen_deck_ids.add(deck_id)

        decks.append({
            "deck_id": deck_id,
            "archetype": archetype,
        })

    # Assign placements: first 8 are top 8
    for i, deck in enumerate(decks):
        if i < 2:
            deck["placement"] = i + 1  # 1st, 2nd
        elif i < 4:
            deck["placement"] = 3  # 3-4th
        elif i < 8:
            deck["placement"] = 5  # 5-8th
        else:
            deck["placement"] = 9  # 9+

    return {
        "player_count": player_count,
        "decks": decks,
    }


def scrape_deck_cards(event_id: str, deck_id: str, format_code: str = STANDARD_FORMAT_CODE) -> list[dict]:
    """Scrape individual card list from a deck page.

    MTGTOP8 encodes the deck as a single concatenated text block like:
    "22 LANDS4 Breeding Pool 1 Forest ... SIDEBOARD1 Card Name ..."
    """
    url = f"{BASE_URL}/event?e={event_id}&d={deck_id}&f={format_code}"

    try:
        soup = _get_soup(url)
    except requests.RequestException as e:
        log.warning("Failed to fetch deck %s: %s", deck_id, e)
        return []

    # The decklist appears as a blob of text, often in a single line
    # containing section headers and "N CardName" entries
    full_text = soup.get_text()

    # Find the decklist section — look for "MD XX SB YY" pattern
    md_match = re.search(r"MD\s+\d+\s+SB\s+\d+", full_text)
    if md_match:
        # Extract everything after the MD/SB marker
        deck_start = md_match.end()
        # Find the end of the decklist (usually ends at Privacy Policy or similar)
        deck_end = full_text.find("Privacy Policy", deck_start)
        if deck_end == -1:
            deck_end = len(full_text)
        deck_text = full_text[deck_start:deck_end]
    else:
        # Fallback: look for the section with LANDS/CREATURES/etc headers
        land_match = re.search(r"\d+\s+LANDS?\s*\d", full_text)
        if land_match:
            deck_start = land_match.start()
            deck_end = full_text.find("Privacy Policy", deck_start)
            if deck_end == -1:
                deck_end = len(full_text)
            deck_text = full_text[deck_start:deck_end]
        else:
            return []

    # Parse the deck text: extract "N CardName" patterns
    # First, remove section headers (LANDS, CREATURES, INSTANTS, SORCERIES, etc.)
    # and split at the SIDEBOARD marker
    board = "main"
    cards = []

    # Split at SIDEBOARD
    parts = re.split(r"SIDEBOARD", deck_text, flags=re.IGNORECASE)
    main_text = parts[0] if parts else deck_text
    side_text = parts[1] if len(parts) > 1 else ""

    def extract_cards(text, board_label):
        """Extract card entries from a text block."""
        result = []
        # Remove section headers
        cleaned = re.sub(
            r"\d+\s+(?:LANDS?|CREATURES?|INSTANTS?\s*(?:and\s*)?SORC(?:ERIES|\.)?|"
            r"OTHER\s+SPELLS?|ENCHANTMENTS?|ARTIFACTS?|PLANESWALKERS?|BATTLES?)",
            " ",
            text,
            flags=re.IGNORECASE,
        )
        # Find all "N CardName" patterns
        # Card names start with uppercase and can contain spaces, apostrophes, commas, hyphens
        matches = re.findall(r"(\d+)\s+([A-Z][A-Za-z\s,'\-/.!]+?)(?=\s*\d+\s+[A-Z]|\s*$)", cleaned)
        for qty, name in matches:
            name = name.strip().rstrip(".")
            if len(name) >= 2 and not re.match(r"^(?:LANDS?|CREATURES?|INSTANTS?|SORC)$", name, re.I):
                result.append({
                    "quantity": int(qty),
                    "name": name,
                    "board": board_label,
                })
        return result

    cards.extend(extract_cards(main_text, "main"))
    if side_text:
        cards.extend(extract_cards(side_text, "side"))

    return cards


def parse_date(date_str: str | None) -> str | None:
    """Try to parse MTGTOP8 date formats (dd/mm/yy)."""
    if not date_str:
        return None
    for fmt in ("%d/%m/%y", "%d/%m/%Y", "%m/%d/%y", "%m/%d/%Y"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str


def run(max_pages: int = 5, max_events: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full pipeline: scrape tournaments -> scrape decklists -> save parquets.

    Args:
        max_pages: Number of tournament list pages to scrape.
        max_events: Maximum number of events to scrape decklists from.
    """
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Step 1: Get tournament list
    tournaments_raw = scrape_tournament_list(max_pages=max_pages)
    log.info("Found %d unique tournaments.", len(tournaments_raw))

    # Step 2: Scrape event pages and decklists for top events
    tournament_records = []
    decklist_records = []

    for t in tournaments_raw[:max_events]:
        event_info = scrape_event_page(t["event_id"])

        # Update tournament record
        t["player_count"] = t.get("player_count") or event_info.get("player_count")
        t["date"] = parse_date(t.get("date_raw"))
        tournament_records.append({
            "event_id": t["event_id"],
            "name": t["name"],
            "date": t["date"],
            "player_count": t["player_count"],
            "format": t["format"],
        })

        # Scrape decklists for top 8 decks
        for deck in event_info["decks"][:8]:
            cards = scrape_deck_cards(t["event_id"], deck["deck_id"])
            log.info("  Deck %s (%s): %d cards", deck["deck_id"], deck["archetype"], len(cards))

            for card in cards:
                decklist_records.append({
                    "event_id": t["event_id"],
                    "archetype": deck["archetype"],
                    "placement": deck["placement"],
                    "player": f"Player_{t['event_id']}_{deck['deck_id']}",
                    "card_name": card["name"],
                    "copies": card["quantity"],
                })

    # Step 3: Save
    tourn_df = pd.DataFrame(tournament_records)
    tourn_df.to_parquet(TOURNAMENTS_PARQUET, index=False)
    log.info("Saved %d tournaments to %s", len(tourn_df), TOURNAMENTS_PARQUET)

    deck_df = pd.DataFrame(decklist_records)
    if not deck_df.empty:
        deck_df.to_parquet(DECKLISTS_PARQUET, index=False)
    log.info("Saved %d decklist rows to %s", len(deck_df), DECKLISTS_PARQUET)

    return tourn_df, deck_df


if __name__ == "__main__":
    run()
