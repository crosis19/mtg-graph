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
SEMANTIC_EDGES_PATH = DATA_PROCESSED / "semantic_edges.parquet"
CARD_EMBEDDINGS_PATH = DATA_EMBEDDINGS / "card_embeddings.npy"
CARD_EMBEDDING_INDEX_PATH = DATA_EMBEDDINGS / "card_embedding_index.json"

# Card interaction edges (counter/removal)
COUNTER_EDGES_PATH = DATA_PROCESSED / "counter_edges.parquet"
REMOVAL_EDGES_PATH = DATA_PROCESSED / "removal_edges.parquet"

# Standard-legal sets filter
STANDARD_FORMAT = "standard"

# Embedding model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
SEMANTIC_SIMILARITY_THRESHOLD = float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.65"))

# Rate limiting for scrapers (seconds between requests)
SCRAPE_DELAY = float(os.getenv("SCRAPE_DELAY", "2.0"))

# MTGDecks matchup data
MTGDECKS_WINRATES_URL = os.getenv(
    "MTGDECKS_WINRATES_URL",
    "https://mtgdecks.net/Standard/winrates/range:last30days",
)

# Graph construction
GRAPH_PATH = DATA_PROCESSED / "graph.pt"

# ── Model architecture ──
D_MODEL = int(os.getenv("D_MODEL", "128"))          # shared embedding dim for all nodes
D_MESSAGE = int(os.getenv("D_MESSAGE", "128"))       # message-passing hidden dim in GNN
D_COUNT = int(os.getenv("D_COUNT", "16"))            # count embedding dim
NUM_GNN_LAYERS = int(os.getenv("NUM_GNN_LAYERS", "2"))
NUM_ATTN_HEADS = int(os.getenv("NUM_ATTN_HEADS", "4"))
DROPOUT = float(os.getenv("DROPOUT", "0.1"))

# Gumbel-softmax temperature schedule
GUMBEL_TAU_START = float(os.getenv("GUMBEL_TAU_START", "1.0"))
GUMBEL_TAU_MIN = float(os.getenv("GUMBEL_TAU_MIN", "0.1"))
GUMBEL_DECAY = float(os.getenv("GUMBEL_DECAY", "0.01"))

# Loss weights
COUNT_LOSS_WEIGHT = float(os.getenv("COUNT_LOSS_WEIGHT", "1.0"))

# Deck construction constraints
MAX_DECK_SIZE = int(os.getenv("MAX_DECK_SIZE", "60"))
MAX_BASIC_LAND_COUNT = int(os.getenv("MAX_BASIC_LAND_COUNT", "20"))
MAX_NONBASIC_COUNT = int(os.getenv("MAX_NONBASIC_COUNT", "4"))

# Training hyperparameters
DECK_LEARNING_RATE = float(os.getenv("DECK_LEARNING_RATE", "0.0001"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.00001"))
DECK_NUM_EPOCHS = int(os.getenv("DECK_NUM_EPOCHS", "50"))
DECK_PATIENCE = int(os.getenv("DECK_PATIENCE", "10"))
DECK_RECENCY_DAYS = int(os.getenv("DECK_RECENCY_DAYS", "30"))
DECK_WARMUP_EPOCHS = int(os.getenv("DECK_WARMUP_EPOCHS", "5"))
DECK_LR_SCHEDULER = os.getenv("DECK_LR_SCHEDULER", "cosine")  # cosine | linear | none
DECK_NUM_ARCHETYPES = int(os.getenv("DECK_NUM_ARCHETYPES", "16"))
DECK_NUM_VAL_ARCHETYPES = int(os.getenv("DECK_NUM_VAL_ARCHETYPES", "3"))
DECK_NUM_CV_FOLDS = int(os.getenv("DECK_NUM_CV_FOLDS", "5"))
DECK_BATCH_SIZE = int(os.getenv("DECK_BATCH_SIZE", "0"))  # archetypes per step (0 = all)
DECK_VAL_EVERY = int(os.getenv("DECK_VAL_EVERY", "5"))    # epochs between validation

# Legacy: still used by graph_builder for copy-count edge weight encoding
DECK_BASIC_LAND_MAX_COPIES = int(os.getenv("DECK_BASIC_LAND_MAX_COPIES", "20"))

# Results — each run gets a timestamped subfolder
def create_run_dir(task: str | None = None, results_base: Path | None = None) -> Path:
    """Create and return a timestamped results directory for this training run.

    Parameters
    ----------
    task : str or None
        Optional task name for results separation (e.g. "metagame", "deck").
        When provided, results go to <base>/<task>/<timestamp>/ with a
        <base>/<task>/latest symlink. When None, uses flat <base>/<timestamp>/.
    results_base : Path or None
        Override the default results directory (RESULTS_DIR). Useful for
        writing intermediate outputs to fast local storage (e.g., Colab's
        /content/) instead of Google Drive during sweeps.
    """
    root = results_base if results_base is not None else RESULTS_DIR
    base = root / task if task else root
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
