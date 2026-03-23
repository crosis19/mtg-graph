"""Data ingestion orchestrator — run all Phase 1-3 steps with validation.

Runs each pipeline step sequentially and prints a validation report after
each one so you can inspect the data before training.

Usage:
    python -m src.ingest_all              # Full pipeline (Phases 1-3)
    python -m src.ingest_all --phase 1    # Phase 1 only (scrape data)
    python -m src.ingest_all --phase 2    # Phase 2 only (synergy edges)
    python -m src.ingest_all --phase 3    # Phase 3 only (build graph)
    python -m src.ingest_all --skip-scrape # Phases 2-3 only (reuse existing data)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from src.config import (
    CARDS_PARQUET,
    CARD_EMBEDDINGS_PATH,
    DECKLISTS_PARQUET,
    GRAPH_PATH,
    KEYWORD_MATRIX_PATH,
    MATCHUPS_PARQUET,
    MECHANICAL_EDGES_PATH,
    METAGAME_PARQUET,
    SEMANTIC_EDGES_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# ── Validation helpers ──────────────────────────────────────────────────


def _validate_parquet(path: Path, name: str, required_cols: list[str] | None = None):
    """Read a parquet file and print a summary. Returns the DataFrame or None."""
    if not path.exists():
        log.warning(f"  MISSING: {name} ({path})")
        return None

    df = pd.read_parquet(path)
    log.info(f"  {name}: {len(df):,} rows, {len(df.columns)} cols")

    if df.empty:
        log.warning(f"    WARNING: {name} is empty!")
        return df

    # Null check
    null_counts = df.isnull().sum()
    nulls = null_counts[null_counts > 0]
    if len(nulls) > 0:
        for col, count in nulls.items():
            pct = count / len(df) * 100
            level = "WARNING" if pct > 20 else "INFO"
            log.log(
                logging.WARNING if pct > 20 else logging.INFO,
                f"    {col}: {count} nulls ({pct:.1f}%)",
            )

    # Required columns check
    if required_cols:
        missing = set(required_cols) - set(df.columns)
        if missing:
            log.error(f"    MISSING COLUMNS: {missing}")

    return df


def _validate_cards(df: pd.DataFrame):
    """Card-specific validation."""
    if df is None or df.empty:
        return
    log.info(f"    Unique cards: {df['name'].nunique():,}")
    if "type_line" in df.columns:
        type_counts = df["type_line"].str.split(" — ").str[0].value_counts().head(5)
        log.info(f"    Top card types: {dict(type_counts)}")



def _validate_metagame(df: pd.DataFrame):
    """Metagame-specific validation."""
    if df is None or df.empty:
        return
    log.info(f"    Unique archetypes: {df['archetype'].nunique()}")
    if "snapshot_date" in df.columns:
        log.info(f"    Snapshots: {df['snapshot_date'].nunique()} dates")
        log.info(f"    Date range: {df['snapshot_date'].min()} to {df['snapshot_date'].max()}")
    if "meta_share_pct" in df.columns:
        total = df.groupby("snapshot_date")["meta_share_pct"].sum()
        log.info(f"    Share sum range: {total.min():.1f}% to {total.max():.1f}%")


def _validate_matchups(df: pd.DataFrame):
    """Matchup-specific validation."""
    if df is None or df.empty:
        return
    log.info(f"    Unique matchup pairs: {len(df):,}")
    unique_archs = set(df["archetype_a"].unique()) | set(df["archetype_b"].unique())
    log.info(f"    Archetypes with matchup data: {len(unique_archs)}")
    if "win_rate_a" in df.columns:
        log.info(f"    Win rate range: {df['win_rate_a'].min():.1f}% to {df['win_rate_a'].max():.1f}%")
    if "sample_size" in df.columns:
        log.info(f"    Sample size range: {df['sample_size'].min()} to {df['sample_size'].max()}")


def _validate_synergy_edges(path: Path, name: str):
    """Validate a synergy edge parquet."""
    df = _validate_parquet(path, name)
    if df is not None and not df.empty:
        if "card_a" in df.columns:
            unique_cards = set(df["card_a"].unique()) | set(df["card_b"].unique())
            log.info(f"    Unique cards with {name} edges: {len(unique_cards)}")
    return df


def _validate_graph():
    """Validate the built graph."""
    if not GRAPH_PATH.exists():
        log.warning(f"  MISSING: graph ({GRAPH_PATH})")
        return

    import torch

    data = torch.load(GRAPH_PATH, weights_only=False)

    log.info("  Graph loaded successfully")
    for nt in data.node_types:
        shape = data[nt].x.shape
        log.info(f"    {nt}: {shape[0]} nodes, {shape[1]}-dim features")

    problems = []
    for et in data.edge_types:
        ei = data[et].edge_index
        src_type, rel, dst_type = et
        n_src = data[src_type].x.shape[0]
        n_dst = data[dst_type].x.shape[0]
        n_edges = ei.shape[1]

        if n_edges == 0:
            log.warning(f"    WARNING: ({src_type}, {rel}, {dst_type}) has 0 edges")
            continue

        log.info(f"    ({src_type}, {rel}, {dst_type}): {n_edges:,} edges")

        if ei[0].max().item() >= n_src:
            problems.append(
                f"    ERROR: ({src_type}, {rel}, {dst_type}) "
                f"src index {ei[0].max().item()} >= {n_src} nodes"
            )
        if ei[1].max().item() >= n_dst:
            problems.append(
                f"    ERROR: ({src_type}, {rel}, {dst_type}) "
                f"dst index {ei[1].max().item()} >= {n_dst} nodes"
            )

    if problems:
        for p in problems:
            log.warning(p)
        log.error("  GRAPH VALIDATION FAILED — fix before training!")
        return False
    else:
        log.info("  All edge indices valid")
        return True


# ── Pipeline steps ──────────────────────────────────────────────────────


def run_phase1():
    """Phase 1: Scrape all external data sources."""
    log.info("")
    log.info("=" * 70)
    log.info("PHASE 1: Data Ingestion")
    log.info("=" * 70)

    # Step 1: Cards
    log.info("\n── Step 1/3: Scryfall Cards ──")
    from src.ingest_cards import run as ingest_cards

    ingest_cards()
    cards_df = _validate_parquet(CARDS_PARQUET, "cards", ["name", "type_line"])
    _validate_cards(cards_df)

    # Step 2: Metagame + Decklists (MTGGoldfish)
    log.info("\n── Step 2/3: MTGGoldfish Metagame + Decklists ──")
    from src.ingest_metagame import run as ingest_metagame

    ingest_metagame()
    meta_df = _validate_parquet(METAGAME_PARQUET, "metagame", ["archetype", "meta_share_pct"])
    _validate_metagame(meta_df)
    d_df = _validate_parquet(DECKLISTS_PARQUET, "decklists", ["archetype", "card_name", "board"])
    if d_df is not None and not d_df.empty:
        log.info(f"    Unique archetypes in decklists: {d_df['archetype'].nunique()}")
        log.info(f"    Unique cards in decklists: {d_df['card_name'].nunique()}")

    # Step 3: Matchups (MTGDecks)
    log.info("\n── Step 3/3: Archetype Matchups ──")
    from src.ingest_matchups import run as ingest_matchups

    ingest_matchups()
    matchup_df = _validate_parquet(MATCHUPS_PARQUET, "matchups", ["archetype_a", "archetype_b"])
    _validate_matchups(matchup_df)

    # Cross-validation: archetype name consistency
    log.info("\n── Cross-Validation ──")
    arch_sources = {}
    if d_df is not None and not d_df.empty:
        arch_sources["decklists"] = set(d_df["archetype"].unique())
    if meta_df is not None and not meta_df.empty:
        arch_sources["metagame"] = set(meta_df["archetype"].unique())
    if matchup_df is not None and not matchup_df.empty:
        arch_sources["matchups"] = (
            set(matchup_df["archetype_a"].unique()) | set(matchup_df["archetype_b"].unique())
        )

    if len(arch_sources) >= 2:
        all_archs = set()
        for archs in arch_sources.values():
            all_archs |= archs
        log.info(f"  Total unique archetypes across sources: {len(all_archs)}")
        for src_name, archs in arch_sources.items():
            log.info(f"    {src_name}: {len(archs)} archetypes")


def run_phase2():
    """Phase 2: Build synergy edges."""
    log.info("")
    log.info("=" * 70)
    log.info("PHASE 2: Synergy Edge Construction")
    log.info("=" * 70)

    # Step 1: Keyword matrix
    log.info("\n── Step 1/3: Keyword Synergy Edges ──")
    from src.keyword_matrix import run as build_keywords

    build_keywords()
    _validate_synergy_edges(KEYWORD_MATRIX_PATH, "keyword_synergy")

    # Step 2: Oracle text parsing (mechanical edges)
    log.info("\n── Step 2/3: Mechanical Synergy Edges ──")
    try:
        from src.oracle_parser import run as build_mechanical

        build_mechanical()
    except ImportError:
        log.info("  oracle_parser not available, skipping")
    _validate_synergy_edges(MECHANICAL_EDGES_PATH, "mechanical_synergy")

    # Step 3: Semantic embeddings
    log.info("\n── Step 3/3: Semantic Synergy Edges ──")
    from src.card_embeddings import run as build_embeddings

    build_embeddings()
    _validate_synergy_edges(SEMANTIC_EDGES_PATH, "semantic_synergy")

    if CARD_EMBEDDINGS_PATH.exists():
        import numpy as np

        emb = np.load(CARD_EMBEDDINGS_PATH)
        log.info(f"    Embedding matrix: {emb.shape[0]} cards x {emb.shape[1]}-dim")


def run_phase3():
    """Phase 3: Build heterogeneous graph."""
    log.info("")
    log.info("=" * 70)
    log.info("PHASE 3: Graph Construction")
    log.info("=" * 70)

    log.info("\n── Building HeteroData graph ──")
    from src.graph_builder import main as build_graph

    build_graph()

    log.info("\n── Graph Validation ──")
    valid = _validate_graph()

    if valid:
        log.info("\n  Graph is ready for training!")
        log.info(f"  Run: python -m src.train_deck")
    else:
        log.error("\n  Graph has problems — inspect and fix before training.")
        sys.exit(1)


def print_summary():
    """Print a final summary of all output files."""
    log.info("")
    log.info("=" * 70)
    log.info("PIPELINE SUMMARY")
    log.info("=" * 70)

    outputs = [
        ("Cards", CARDS_PARQUET),
        ("Metagame", METAGAME_PARQUET),
        ("Decklists", DECKLISTS_PARQUET),
        ("Matchups", MATCHUPS_PARQUET),
        ("Keyword Synergy", KEYWORD_MATRIX_PATH),
        ("Mechanical Synergy", MECHANICAL_EDGES_PATH),
        ("Semantic Synergy", SEMANTIC_EDGES_PATH),
        ("Graph", GRAPH_PATH),
    ]

    for name, path in outputs:
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            if path.suffix == ".parquet":
                try:
                    rows = len(pd.read_parquet(path))
                    log.info(f"  OK  {name:25s} {rows:>8,} rows  ({size_mb:.1f} MB)")
                except Exception:
                    log.info(f"  OK  {name:25s} ({size_mb:.1f} MB)")
            else:
                log.info(f"  OK  {name:25s} ({size_mb:.1f} MB)")
        else:
            log.info(f"  --  {name:25s} not found")


# ── CLI entry point ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run the MTG data ingestion pipeline with validation.",
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        help="Run only a specific phase (1=scrape, 2=synergy, 3=graph)",
    )
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip Phase 1 (use existing scraped data)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Don't run any steps, just validate existing output files",
    )
    args = parser.parse_args()

    start = time.time()

    if args.validate_only:
        log.info("Validation-only mode — checking existing outputs")
        log.info("\n── Phase 1 Outputs ──")
        cards = _validate_parquet(CARDS_PARQUET, "cards", ["name", "type_line"])
        _validate_cards(cards)
        meta = _validate_parquet(METAGAME_PARQUET, "metagame", ["archetype", "meta_share_pct"])
        _validate_metagame(meta)
        d_df = _validate_parquet(DECKLISTS_PARQUET, "decklists", ["archetype", "card_name", "board"])
        matchups = _validate_parquet(MATCHUPS_PARQUET, "matchups", ["archetype_a", "archetype_b"])
        _validate_matchups(matchups)
        log.info("\n── Phase 2 Outputs ──")
        _validate_synergy_edges(KEYWORD_MATRIX_PATH, "keyword_synergy")
        _validate_synergy_edges(MECHANICAL_EDGES_PATH, "mechanical_synergy")
        _validate_synergy_edges(SEMANTIC_EDGES_PATH, "semantic_synergy")
        log.info("\n── Phase 3 Outputs ──")
        _validate_graph()
        print_summary()
    elif args.phase == 1:
        run_phase1()
        print_summary()
    elif args.phase == 2:
        run_phase2()
        print_summary()
    elif args.phase == 3:
        run_phase3()
        print_summary()
    elif args.skip_scrape:
        run_phase2()
        run_phase3()
        print_summary()
    else:
        run_phase1()
        run_phase2()
        run_phase3()
        print_summary()

    elapsed = time.time() - start
    log.info(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
