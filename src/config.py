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
DECKLISTS_PARQUET = DATA_PROCESSED / "decklists.parquet"
METAGAME_PARQUET = DATA_PROCESSED / "metagame.parquet"
MATCHUPS_PARQUET = DATA_PROCESSED / "matchups.parquet"

# Synergy outputs
KEYWORD_MATRIX_PATH = DATA_PROCESSED / "keyword_matrix.parquet"
MECHANICAL_EDGES_PATH = DATA_PROCESSED / "mechanical_edges.parquet"
SEMANTIC_EDGES_PATH = DATA_PROCESSED / "semantic_edges.parquet"
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

# Model architecture
HIDDEN_DIM = int(os.getenv("HIDDEN_DIM", "128"))
NUM_HEADS = int(os.getenv("NUM_HEADS", "4"))
NUM_HGT_LAYERS = int(os.getenv("NUM_HGT_LAYERS", "3"))

# Training hyperparameters
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.0001"))
TRAIN_SPLIT_RATIO = float(os.getenv("TRAIN_SPLIT_RATIO", "0.6"))
VAL_SPLIT_RATIO = float(os.getenv("VAL_SPLIT_RATIO", "0.2"))
# Test split is implicit: 1.0 - TRAIN_SPLIT_RATIO - VAL_SPLIT_RATIO

# Deck predictor hyperparameters
DECK_LEARNING_RATE = float(os.getenv("DECK_LEARNING_RATE", "0.001"))
DECK_NUM_EPOCHS = int(os.getenv("DECK_NUM_EPOCHS", "50"))
DECK_DROPOUT = float(os.getenv("DECK_DROPOUT", "0.3"))
DECK_MIN_META_SHARE = float(os.getenv("DECK_MIN_META_SHARE", "5.0"))
DECK_N_NEGATIVES = int(os.getenv("DECK_N_NEGATIVES", "50"))
DECK_BPR_MARGIN = float(os.getenv("DECK_BPR_MARGIN", "1.0"))
DECK_PATIENCE = int(os.getenv("DECK_PATIENCE", "10"))
DECK_CHECKPOINT_METRIC = os.getenv("DECK_CHECKPOINT_METRIC", "val_hits25")
DECK_FREEZE_HGT = bool(int(os.getenv("DECK_FREEZE_HGT", "0")))
DECK_RECENCY_DAYS = int(os.getenv("DECK_RECENCY_DAYS", "30"))

# Results — each run gets a timestamped subfolder
def create_run_dir(task: str | None = None) -> Path:
    """Create and return a timestamped results directory for this training run.

    Parameters
    ----------
    task : str or None
        Optional task name for results separation (e.g. "metagame", "deck").
        When provided, results go to results/<task>/<timestamp>/ with a
        results/<task>/latest symlink. When None, uses flat results/<timestamp>/.
    """
    base = RESULTS_DIR / task if task else RESULTS_DIR
    run_name = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    run_dir = base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    # Symlink 'latest' for easy access
    latest = base / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    try:
        latest.symlink_to(run_dir.name)
    except OSError:
        pass  # Symlinks may not work on all platforms (e.g., Windows without admin)
    return run_dir
