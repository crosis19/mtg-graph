"""Generate synthetic tournament, decklist, metagame, and matchup data for development.

Creates realistic-looking data based on the actual Standard card pool so we can
build and test the full graph construction and model pipeline without live scraping.

Outputs:
  - tournaments.parquet
  - decklists.parquet
  - metagame.parquet
  - matchups.parquet
"""

import logging
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.config import (
    CARDS_PARQUET,
    TOURNAMENTS_PARQUET,
    DECKLISTS_PARQUET,
    METAGAME_PARQUET,
    MATCHUPS_PARQUET,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Archetype definitions: name → (color_identity, strategy keywords for card selection)
ARCHETYPES = {
    "Azorius Control": {
        "colors": {"W", "U"},
        "prefer_types": ["Instant", "Sorcery", "Enchantment", "Planeswalker"],
        "avoid_types": [],
    },
    "Esper Midrange": {
        "colors": {"W", "U", "B"},
        "prefer_types": ["Creature", "Instant", "Planeswalker"],
        "avoid_types": [],
    },
    "Domain Ramp": {
        "colors": {"W", "U", "B", "R", "G"},
        "prefer_types": ["Creature", "Sorcery", "Land"],
        "avoid_types": [],
    },
    "Mono-Red Aggro": {
        "colors": {"R"},
        "prefer_types": ["Creature", "Instant", "Sorcery"],
        "avoid_types": ["Enchantment"],
    },
    "Golgari Midrange": {
        "colors": {"B", "G"},
        "prefer_types": ["Creature", "Instant", "Sorcery"],
        "avoid_types": [],
    },
    "Boros Convoke": {
        "colors": {"W", "R"},
        "prefer_types": ["Creature", "Instant"],
        "avoid_types": ["Planeswalker"],
    },
    "Dimir Control": {
        "colors": {"U", "B"},
        "prefer_types": ["Instant", "Sorcery", "Creature"],
        "avoid_types": [],
    },
    "Gruul Stompy": {
        "colors": {"R", "G"},
        "prefer_types": ["Creature", "Sorcery"],
        "avoid_types": ["Enchantment"],
    },
    "Orzhov Midrange": {
        "colors": {"W", "B"},
        "prefer_types": ["Creature", "Enchantment", "Instant"],
        "avoid_types": [],
    },
    "Simic Ramp": {
        "colors": {"U", "G"},
        "prefer_types": ["Creature", "Sorcery", "Instant"],
        "avoid_types": [],
    },
    "Mono-White Aggro": {
        "colors": {"W"},
        "prefer_types": ["Creature", "Instant", "Enchantment"],
        "avoid_types": [],
    },
    "Rakdos Sacrifice": {
        "colors": {"B", "R"},
        "prefer_types": ["Creature", "Instant", "Artifact"],
        "avoid_types": [],
    },
}

TOURNAMENT_TEMPLATES = [
    ("Pro Tour {city}", [400, 500, 600]),
    ("Regional Championship {region}", [200, 300, 400]),
    ("Arena Championship {n}", [100, 150, 200]),
    ("Standard Challenge {n}", [64, 128, 256]),
]

CITIES = ["Chicago", "Barcelona", "Tokyo", "Amsterdam", "Sydney", "Toronto"]
REGIONS = ["North America", "Europe", "Asia-Pacific", "Latin America"]


def _color_set_from_identity(color_identity_str: str) -> set:
    """Parse color identity string like 'W|U|B' into a set."""
    if pd.isna(color_identity_str) or color_identity_str == "":
        return set()
    return set(color_identity_str.split("|"))


def _card_fits_archetype(card: pd.Series, archetype_info: dict) -> bool:
    """Check if a card's colors are a subset of the archetype's colors."""
    card_colors = _color_set_from_identity(card["color_identity"])
    # Colorless cards fit any archetype
    if len(card_colors) == 0:
        return True
    # Lands fit any archetype
    if card["is_land"]:
        return True
    return card_colors.issubset(archetype_info["colors"])


def _type_score(card: pd.Series, archetype_info: dict) -> float:
    """Score how well a card's type matches archetype preferences."""
    score = 1.0
    type_line = str(card["type_line"]).lower()
    for pref in archetype_info["prefer_types"]:
        if pref.lower() in type_line:
            score += 2.0
    for avoid in archetype_info["avoid_types"]:
        if avoid.lower() in type_line:
            score *= 0.1
    # Prefer lower CMC for aggro archetypes
    if "Aggro" in str(archetype_info.get("name", "")):
        score *= max(0.1, 1.0 - card["cmc"] / 10.0)
    return score


def build_archetype_pools(cards: pd.DataFrame) -> dict:
    """For each archetype, build a weighted pool of eligible cards."""
    pools = {}
    for arch_name, arch_info in ARCHETYPES.items():
        eligible = cards[cards.apply(lambda c: _card_fits_archetype(c, arch_info), axis=1)]
        weights = eligible.apply(lambda c: _type_score(c, arch_info), axis=1).values
        weights = weights / weights.sum()
        pools[arch_name] = (eligible["name"].values, weights)
        log.info(f"  {arch_name}: {len(eligible)} eligible cards")
    return pools


def generate_decklist(pool_names: np.ndarray, pool_weights: np.ndarray) -> list:
    """Generate a synthetic 60-card decklist from a card pool."""
    # Pick 25-35 unique cards, then fill to 60 with duplicates (up to 4 copies)
    n_unique = random.randint(25, 35)
    n_unique = min(n_unique, len(pool_names))
    chosen_idx = np.random.choice(len(pool_names), size=n_unique, replace=False, p=pool_weights)
    chosen_names = pool_names[chosen_idx]

    decklist = []
    for name in chosen_names:
        copies = random.choices([1, 2, 3, 4], weights=[15, 30, 30, 25])[0]
        decklist.extend([name] * copies)

    # Trim or pad to 60
    if len(decklist) > 60:
        decklist = random.sample(decklist, 60)
    while len(decklist) < 60:
        name = random.choice(chosen_names)
        decklist.append(name)

    return decklist


def generate_tournaments(n_tournaments: int = 24, start_date: str = "2024-06-01") -> pd.DataFrame:
    """Generate synthetic tournament records spread over ~18 months."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    records = []
    city_idx = 0
    region_idx = 0
    challenge_n = 1

    for i in range(n_tournaments):
        days_offset = int(i * (540 / n_tournaments)) + random.randint(-7, 7)
        date = start + timedelta(days=max(0, days_offset))
        template = TOURNAMENT_TEMPLATES[i % len(TOURNAMENT_TEMPLATES)]
        name_template, player_counts = template
        player_count = random.choice(player_counts)

        if "{city}" in name_template:
            name = name_template.format(city=CITIES[city_idx % len(CITIES)])
            city_idx += 1
        elif "{region}" in name_template:
            name = name_template.format(region=REGIONS[region_idx % len(REGIONS)])
            region_idx += 1
        elif "{n}" in name_template:
            name = name_template.format(n=challenge_n)
            challenge_n += 1
        else:
            name = name_template

        records.append({
            "event_id": f"evt_{i+1:04d}",
            "name": name,
            "date": date.strftime("%Y-%m-%d"),
            "player_count": player_count,
            "format": "Standard",
        })

    return pd.DataFrame(records)


def generate_decklists(
    tournaments: pd.DataFrame,
    archetype_pools: dict,
    metagame_shares: dict,
) -> pd.DataFrame:
    """Generate top-8 decklists for each tournament."""
    archetype_names = list(ARCHETYPES.keys())
    shares = np.array([metagame_shares.get(a, 5.0) for a in archetype_names])
    shares = shares / shares.sum()

    records = []
    for _, tourney in tournaments.iterrows():
        # Pick 8 archetypes for top 8, weighted by meta share + noise
        noisy_shares = shares + np.random.dirichlet(np.ones(len(shares)) * 2) * 0.3
        noisy_shares = noisy_shares / noisy_shares.sum()
        top8_indices = np.random.choice(
            len(archetype_names), size=8, replace=True, p=noisy_shares
        )
        for placement, arch_idx in enumerate(top8_indices, 1):
            arch_name = archetype_names[arch_idx]
            pool_names, pool_weights = archetype_pools[arch_name]
            decklist = generate_decklist(pool_names, pool_weights)

            for card_name in set(decklist):
                copies = decklist.count(card_name)
                records.append({
                    "event_id": tourney["event_id"],
                    "archetype": arch_name,
                    "placement": placement,
                    "player": f"Player_{tourney['event_id']}_{placement}",
                    "card_name": card_name,
                    "copies": copies,
                })

    return pd.DataFrame(records)


def generate_metagame(tournaments: pd.DataFrame) -> pd.DataFrame:
    """Generate metagame snapshot data (archetype share over time)."""
    records = []
    base_shares = {
        "Azorius Control": 8.5,
        "Esper Midrange": 12.0,
        "Domain Ramp": 14.0,
        "Mono-Red Aggro": 11.0,
        "Golgari Midrange": 10.0,
        "Boros Convoke": 7.5,
        "Dimir Control": 6.0,
        "Gruul Stompy": 5.5,
        "Orzhov Midrange": 6.5,
        "Simic Ramp": 5.0,
        "Mono-White Aggro": 8.0,
        "Rakdos Sacrifice": 6.0,
    }

    # Simulate metagame drift over time
    dates = sorted(tournaments["date"].unique())
    for i, date in enumerate(dates):
        drift = i / len(dates)  # 0→1 over time
        for arch, base in base_shares.items():
            # Add sinusoidal drift + noise to simulate metagame cycling
            share = base + 3.0 * np.sin(drift * 2 * np.pi + hash(arch) % 7)
            share += random.gauss(0, 1.5)
            share = max(1.0, min(25.0, share))
            win_rate = 48.0 + random.gauss(0, 3.0)
            win_rate = max(40.0, min(60.0, win_rate))

            colors = "|".join(sorted(ARCHETYPES[arch]["colors"]))
            records.append({
                "snapshot_date": date,
                "archetype": arch,
                "meta_share_pct": round(share, 2),
                "win_rate": round(win_rate, 2),
                "color_identity": colors,
            })

    return pd.DataFrame(records), base_shares


def generate_matchups() -> pd.DataFrame:
    """Generate synthetic head-to-head matchup data."""
    archetype_names = list(ARCHETYPES.keys())
    records = []

    # Create somewhat realistic rock-paper-scissors dynamics
    # Aggro beats control, control beats midrange, midrange beats aggro
    archetype_roles = {}
    for name in archetype_names:
        if "Aggro" in name or "Convoke" in name or "Stompy" in name:
            archetype_roles[name] = "aggro"
        elif "Control" in name or "Ramp" in name:
            archetype_roles[name] = "control"
        else:
            archetype_roles[name] = "midrange"

    for i, arch_a in enumerate(archetype_names):
        for j, arch_b in enumerate(archetype_names):
            if i >= j:
                continue
            role_a = archetype_roles[arch_a]
            role_b = archetype_roles[arch_b]

            # Base 50/50
            wr_a = 50.0
            # Rock-paper-scissors adjustments
            if role_a == "aggro" and role_b == "control":
                wr_a += random.uniform(3, 8)
            elif role_a == "control" and role_b == "aggro":
                wr_a -= random.uniform(3, 8)
            elif role_a == "control" and role_b == "midrange":
                wr_a += random.uniform(2, 6)
            elif role_a == "midrange" and role_b == "control":
                wr_a -= random.uniform(2, 6)
            elif role_a == "midrange" and role_b == "aggro":
                wr_a += random.uniform(2, 6)
            elif role_a == "aggro" and role_b == "midrange":
                wr_a -= random.uniform(2, 6)

            # Add noise
            wr_a += random.gauss(0, 2.0)
            wr_a = max(30.0, min(70.0, wr_a))
            wr_b = 100.0 - wr_a

            records.append({
                "archetype_a": arch_a,
                "archetype_b": arch_b,
                "win_rate_a": round(wr_a, 1),
                "win_rate_b": round(wr_b, 1),
                "sample_size": random.randint(50, 500),
                "snapshot_date": "2025-01-15",
            })

    return pd.DataFrame(records)


def main():
    random.seed(42)
    np.random.seed(42)

    log.info("Loading card pool...")
    cards = pd.read_parquet(CARDS_PARQUET)
    log.info(f"  {len(cards)} Standard-legal cards")

    log.info("Building archetype card pools...")
    archetype_pools = build_archetype_pools(cards)

    log.info("Generating tournaments...")
    tournaments = generate_tournaments(n_tournaments=24)
    tournaments.to_parquet(TOURNAMENTS_PARQUET, index=False)
    log.info(f"  {len(tournaments)} tournaments → {TOURNAMENTS_PARQUET}")

    log.info("Generating metagame snapshots...")
    metagame, base_shares = generate_metagame(tournaments)
    metagame.to_parquet(METAGAME_PARQUET, index=False)
    log.info(f"  {len(metagame)} metagame rows → {METAGAME_PARQUET}")

    log.info("Generating decklists...")
    decklists = generate_decklists(tournaments, archetype_pools, base_shares)
    decklists.to_parquet(DECKLISTS_PARQUET, index=False)
    log.info(f"  {len(decklists)} decklist rows → {DECKLISTS_PARQUET}")

    log.info("Generating matchup data...")
    matchups = generate_matchups()
    matchups.to_parquet(MATCHUPS_PARQUET, index=False)
    log.info(f"  {len(matchups)} matchup rows → {MATCHUPS_PARQUET}")

    log.info("Synthetic data generation complete!")


if __name__ == "__main__":
    main()
