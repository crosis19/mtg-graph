"""Phase 1: Scrape head-to-head archetype matchup win rates.

Collects pairwise win rates between archetypes from MTGDecks or similar sources.
These become the weighted "counters" edges in the graph.

Outputs: matchups.parquet
"""

import logging
import re
import time
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import MATCHUPS_PARQUET, DATA_PROCESSED, SCRAPE_DELAY

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "MTG-Graph-Research/1.0 (academic project; respectful scraping)",
}

# MTGDecks.net winrates page for Standard
MTGDECKS_WINRATES_URL = "https://mtgdecks.net/Standard/winrates"


def scrape_mtgdecks_matchups() -> list[dict]:
    """Scrape head-to-head matchup data from MTGDecks.net."""
    log.info("Fetching matchup data from %s", MTGDECKS_WINRATES_URL)

    time.sleep(SCRAPE_DELAY)
    try:
        resp = requests.get(MTGDECKS_WINRATES_URL, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.warning("Failed to fetch matchup data: %s", e)
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    matchups = []

    # MTGDecks typically shows matchups in a matrix table
    tables = soup.find_all("table")

    for table in tables:
        matchups.extend(_parse_matchup_table(table))

    if not matchups:
        # Fallback: try parsing from any structured data on the page
        matchups = _parse_matchup_fallback(soup)

    log.info("Parsed %d matchup records.", len(matchups))
    return matchups


def _parse_matchup_table(table) -> list[dict]:
    """Parse a matchup matrix table into pairwise records."""
    records = []

    # Get header row for archetype names
    header_row = table.find("tr")
    if not header_row:
        return records

    headers = []
    for th in header_row.find_all(["th", "td"]):
        text = th.get_text(strip=True)
        if text:
            headers.append(text)

    if len(headers) < 3:  # Need at least row label + 2 archetypes
        return records

    # Parse data rows
    for row in table.find_all("tr")[1:]:
        cells = row.find_all(["td", "th"])
        if not cells:
            continue

        row_archetype = cells[0].get_text(strip=True)
        if not row_archetype:
            continue

        for i, cell in enumerate(cells[1:], start=1):
            if i >= len(headers):
                break

            col_archetype = headers[i]
            text = cell.get_text(strip=True)

            # Look for win rate percentage
            pct_match = re.search(r"([\d.]+)\s*%?", text)
            if not pct_match:
                continue

            win_rate = float(pct_match.group(1))
            # Normalize to 0-1 range if given as percentage
            if win_rate > 1:
                win_rate /= 100.0

            # Skip mirror matches and invalid data
            if row_archetype == col_archetype:
                continue
            if win_rate < 0 or win_rate > 1:
                continue

            # Look for sample size
            sample_match = re.search(r"(\d+)\s*(?:games?|matches?|samples?)", text, re.I)
            sample_size = int(sample_match.group(1)) if sample_match else None

            records.append({
                "archetype_a": row_archetype,
                "archetype_b": col_archetype,
                "win_rate_a": win_rate,
                "win_rate_b": 1.0 - win_rate,
                "sample_size": sample_size,
                "snapshot_date": datetime.now().strftime("%Y-%m-%d"),
                "source": "mtgdecks",
            })

    return records


def _parse_matchup_fallback(soup: BeautifulSoup) -> list[dict]:
    """Fallback parser for when the table structure doesn't match."""
    records = []

    # Look for any text patterns like "Archetype A vs Archetype B: 55%"
    text = soup.get_text()
    pattern = re.compile(
        r"([A-Z][\w\s-]+?)\s+vs\.?\s+([A-Z][\w\s-]+?)[\s:]+(\d+(?:\.\d+)?)\s*%",
        re.MULTILINE,
    )

    for match in pattern.finditer(text):
        arch_a = match.group(1).strip()
        arch_b = match.group(2).strip()
        win_rate = float(match.group(3)) / 100.0

        if 0 < win_rate < 1 and arch_a != arch_b:
            records.append({
                "archetype_a": arch_a,
                "archetype_b": arch_b,
                "win_rate_a": win_rate,
                "win_rate_b": 1.0 - win_rate,
                "sample_size": None,
                "snapshot_date": datetime.now().strftime("%Y-%m-%d"),
                "source": "mtgdecks",
            })

    return records


def compute_counters_edges(
    matchups_df: pd.DataFrame,
    threshold: float = 0.55,
    min_samples: int = 10,
) -> pd.DataFrame:
    """Derive directed 'counters' edges from matchup data.

    A 'counters' edge A → B means archetype A has a favorable matchup
    against B (win rate > threshold).

    Args:
        matchups_df: Raw matchup data.
        threshold: Minimum win rate to constitute a "counters" relationship.
        min_samples: Minimum sample size to trust the matchup data.
    """
    if matchups_df.empty:
        return pd.DataFrame()

    df = matchups_df.copy()

    # Filter by sample size if available
    if "sample_size" in df.columns:
        has_samples = df["sample_size"].notna()
        df = df[~has_samples | (df["sample_size"] >= min_samples)]

    # Create counters edges where win rate exceeds threshold
    counters = df[df["win_rate_a"] >= threshold].copy()
    counters = counters.rename(columns={
        "archetype_a": "source",
        "archetype_b": "target",
        "win_rate_a": "weight",
    })
    counters["edge_type"] = "counters"
    counters = counters[["source", "target", "weight", "edge_type", "sample_size", "snapshot_date"]]

    log.info(
        "Derived %d counters edges (threshold=%.2f, min_samples=%d).",
        len(counters), threshold, min_samples,
    )
    return counters


def run() -> pd.DataFrame:
    """Full pipeline: scrape → derive counters → save."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    matchups = scrape_mtgdecks_matchups()
    df = pd.DataFrame(matchups)

    # Append to existing data if present
    if MATCHUPS_PARQUET.exists():
        existing = pd.read_parquet(MATCHUPS_PARQUET)
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(
            subset=["archetype_a", "archetype_b", "snapshot_date"],
            keep="last",
        )

    if not df.empty:
        df.to_parquet(MATCHUPS_PARQUET, index=False)
        log.info("Saved %d matchup rows to %s", len(df), MATCHUPS_PARQUET)
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
