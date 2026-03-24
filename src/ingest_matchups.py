"""Phase 1: Scrape head-to-head archetype matchup win rates.

Collects pairwise win rates between archetypes from MTGDecks.net (last 30 days).
These become the weighted "counters" edges in the graph.

Outputs: matchups.parquet
"""

import logging
import re
import time
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup

try:
    import cloudscraper
except ImportError:
    cloudscraper = None

import requests

from src.config import MATCHUPS_PARQUET, DATA_PROCESSED, SCRAPE_DELAY, MTGDECKS_WINRATES_URL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
}


def scrape_mtgdecks_matchups() -> list[dict]:
    """Scrape head-to-head matchup data from MTGDecks.net.

    Returns list of dicts with columns:
        archetype_a, archetype_b, win_rate_a, win_rate_b,
        sample_size, snapshot_date, source
    Win rates are stored as percentages (0-100).
    """
    log.info("Fetching matchup data from %s", MTGDECKS_WINRATES_URL)

    time.sleep(SCRAPE_DELAY)
    try:
        # MTGDecks uses Cloudflare JS challenge — cloudscraper handles it
        if cloudscraper is not None:
            scraper = cloudscraper.create_scraper()
            resp = scraper.get(MTGDECKS_WINRATES_URL, timeout=30)
        else:
            log.warning("cloudscraper not installed; falling back to requests "
                        "(may fail on Cloudflare-protected sites)")
            resp = requests.get(MTGDECKS_WINRATES_URL, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.warning("Failed to fetch matchup data: %s", e)
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", id="winrates")
    if not table:
        log.warning("No table#winrates found on page")
        return []

    return _parse_winrate_table(table)


def _parse_winrate_table(table) -> list[dict]:
    """Parse the MTGDecks winrate matrix table.

    Structure:
    - Header row: ['', 'Overall', arch1, arch2, ...]
    - Data rows: [arch_name, overall_cell, cell1, cell2, ...]
    - Each cell has class 'winrate-cell' and data-winrate='N' (percentage int)
    - Sample size in cell text as 'N matches'
    - Mirror/empty cells show '--' with no data-winrate
    """
    records = []
    rows = table.find_all("tr")
    if len(rows) < 2:
        return records

    # Extract column archetype names from header (skip '' and 'Overall')
    header_cells = rows[0].find_all(["th", "td"])
    col_names = [c.get_text(strip=True) for c in header_cells]

    snapshot_date = datetime.now().strftime("%Y-%m-%d")

    for row in rows[1:]:
        cells = row.find_all(["th", "td"])
        if not cells:
            continue

        row_archetype = cells[0].get_text(strip=True)
        if not row_archetype:
            continue

        # cells[1:] correspond to col_names[1:]
        for j, cell in enumerate(cells[1:], start=1):
            if j >= len(col_names):
                break

            col_archetype = col_names[j]

            # Skip 'Overall' column and mirror matches
            if col_archetype == "Overall" or col_archetype == row_archetype:
                continue

            # Get win rate from data attribute
            win_rate_str = cell.get("data-winrate")
            if not win_rate_str:
                continue  # '--' empty cell

            try:
                win_rate_a = float(win_rate_str)  # percentage (0-100)
            except (ValueError, TypeError):
                continue

            if win_rate_a < 0 or win_rate_a > 100:
                continue

            # Extract sample size from cell text
            cell_text = cell.get_text()
            match = re.search(r"([\d,]+)\s*matches", cell_text)
            sample_size = int(match.group(1).replace(",", "")) if match else None

            records.append({
                "archetype_a": row_archetype,
                "archetype_b": col_archetype,
                "win_rate_a": win_rate_a,
                "win_rate_b": 100.0 - win_rate_a,
                "sample_size": sample_size,
                "snapshot_date": snapshot_date,
                "source": "mtgdecks",
            })

    # Normalize archetype names to match MTGGoldfish convention
    # MTGDecks: "Mono Green Landfall" → MTGGoldfish: "Mono-Green Landfall"
    for rec in records:
        rec["archetype_a"] = _normalize_archetype_name(rec["archetype_a"])
        rec["archetype_b"] = _normalize_archetype_name(rec["archetype_b"])

    log.info("Parsed %d matchup records.", len(records))
    return records


def _normalize_archetype_name(name: str) -> str:
    """Normalize archetype names so MTGDecks names match MTGGoldfish convention.

    'Mono Green Landfall' → 'Mono-Green Landfall'
    'Mono Red Aggro'      → 'Mono-Red Aggro'
    """
    return re.sub(r"\bMono\s+", "Mono-", name)


def run() -> pd.DataFrame:
    """Full pipeline: scrape → save (replaces stale data)."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    matchups = scrape_mtgdecks_matchups()
    df = pd.DataFrame(matchups)

    if not df.empty:
        # Deduplicate: keep one record per archetype pair per snapshot
        df = df.drop_duplicates(
            subset=["archetype_a", "archetype_b", "snapshot_date"],
            keep="last",
        )
        df.to_parquet(MATCHUPS_PARQUET, index=False)
        log.info("Saved %d matchup rows to %s", len(df), MATCHUPS_PARQUET)

        # Summary
        unique_archs = set(df["archetype_a"].unique()) | set(df["archetype_b"].unique())
        log.info("  Unique archetypes: %d", len(unique_archs))
        log.info("  Win rate range: %.1f%% to %.1f%%",
                 df["win_rate_a"].min(), df["win_rate_a"].max())
        if df["sample_size"].notna().any():
            log.info("  Sample size range: %d to %d",
                     df["sample_size"].dropna().astype(int).min(),
                     df["sample_size"].dropna().astype(int).max())
    else:
        log.warning("No matchup data collected. Saving empty parquet.")
        df = pd.DataFrame(columns=[
            "archetype_a", "archetype_b", "win_rate_a", "win_rate_b",
            "sample_size", "snapshot_date", "source",
        ])
        df.to_parquet(MATCHUPS_PARQUET, index=False)

    return df


if __name__ == "__main__":
    run()
