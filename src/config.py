"""Project-wide paths and constants."""

import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EMBEDDINGS = PROJECT_ROOT / "data" / "embeddings"
RESULTS_DIR = PROJECT_ROOT / "results"

# Scryfall
SCRYFALL_BULK_URL = os.getenv("SCRYFALL_BULK_URL", "https://api.scryfall.com/bulk-data")
ORACLE_CARDS_PATH = DATA_RAW / "oracle_cards.json"

# Processed outputs
CARDS_PARQUET = DATA_PROCESSED / "cards.parquet"
TOURNAMENTS_PARQUET = DATA_PROCESSED / "tournaments.parquet"
DECKLISTS_PARQUET = DATA_PROCESSED / "decklists.parquet"
METAGAME_PARQUET = DATA_PROCESSED / "metagame.parquet"
MATCHUPS_PARQUET = DATA_PROCESSED / "matchups.parquet"

# Synergy outputs
KEYWORD_MATRIX_PATH = DATA_PROCESSED / "keyword_matrix.parquet"
MECHANICAL_EDGES_PATH = DATA_PROCESSED / "mechanical_edges.parquet"
SEMANTIC_EDGES_PATH = DATA_PROCESSED / "semantic_edges.parquet"
CO_OCCURRENCE_EDGES_PATH = DATA_PROCESSED / "co_occurrence_edges.parquet"
CARD_EMBEDDINGS_PATH = DATA_EMBEDDINGS / "card_embeddings.npy"
CARD_EMBEDDING_INDEX_PATH = DATA_EMBEDDINGS / "card_embedding_index.json"

# Standard-legal sets filter
STANDARD_FORMAT = "standard"

# Embedding model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
SEMANTIC_SIMILARITY_THRESHOLD = float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.65"))

# Rate limiting for scrapers (seconds between requests)
SCRAPE_DELAY = float(os.getenv("SCRAPE_DELAY", "2.0"))

# Graph construction
GRAPH_PATH = DATA_PROCESSED / "graph.pt"
COUNTERS_WIN_RATE_THRESHOLD = float(os.getenv("COUNTERS_WIN_RATE_THRESHOLD", "55.0"))
CO_OCCURRENCE_MIN_DECKS = int(os.getenv("CO_OCCURRENCE_MIN_DECKS", "2"))

# Model architecture
HIDDEN_DIM = int(os.getenv("HIDDEN_DIM", "128"))
NUM_HEADS = int(os.getenv("NUM_HEADS", "4"))
NUM_HGT_LAYERS = int(os.getenv("NUM_HGT_LAYERS", "3"))

# Training hyperparameters
DROPOUT = float(os.getenv("DROPOUT", "0.2"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.0001"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "100"))
EMERGENCE_LOSS_WEIGHT = float(os.getenv("EMERGENCE_LOSS_WEIGHT", "1.0"))
TOP8_LOSS_WEIGHT = float(os.getenv("TOP8_LOSS_WEIGHT", "1.0"))
TRAIN_SPLIT_RATIO = float(os.getenv("TRAIN_SPLIT_RATIO", "0.6"))
VAL_SPLIT_RATIO = float(os.getenv("VAL_SPLIT_RATIO", "0.2"))
# Test split is implicit: 1.0 - TRAIN_SPLIT_RATIO - VAL_SPLIT_RATIO

# Results — each run gets a timestamped subfolder
def create_run_dir() -> Path:
    """Create and return a timestamped results directory for this training run."""
    run_name = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    run_dir = RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    # Symlink 'latest' for easy access
    latest = RESULTS_DIR / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    try:
        latest.symlink_to(run_dir.name)
    except OSError:
        pass  # Symlinks may not work on all platforms (e.g., Windows without admin)
    return run_dir
