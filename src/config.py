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

# ── Multi-format support ──
SUPPORTED_FORMATS: dict[str, dict] = {
    "standard": {
        "legality_key": "standard",
        "goldfish_path": "/metagame/standard",
        "mtgdecks_url": "https://mtgdecks.net/Standard/winrates/range:last30days",
        "num_archetypes": 0,  # 0 = all available archetypes
    },
    "pioneer": {
        "legality_key": "pioneer",
        "goldfish_path": "/metagame/pioneer",
        "mtgdecks_url": "https://mtgdecks.net/Pioneer/winrates/range:last30days",
        "num_archetypes": 0,  # 0 = all available archetypes
    },
}

ACTIVE_FORMATS: list[str] = os.getenv("ACTIVE_FORMATS", "standard,pioneer").split(",")


def format_path(base_path: Path, fmt: str) -> Path:
    """Insert format name into a data path for per-format namespacing.

    E.g. data/processed/metagame.parquet -> data/processed/standard/metagame.parquet
    """
    return base_path.parent / fmt / base_path.name


# Scryfall
SCRYFALL_BULK_URL = os.getenv("SCRYFALL_BULK_URL", "https://api.scryfall.com/bulk-data")
ORACLE_CARDS_PATH = DATA_RAW / "oracle_cards.json"

# Processed outputs — shared (format-agnostic)
CARDS_PARQUET = DATA_PROCESSED / "cards.parquet"

# Per-format processed outputs (use format_path() to get actual path)
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

# Legacy alias (kept for any external references)
STANDARD_FORMAT = "standard"

# Embedding model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
SEMANTIC_SIMILARITY_THRESHOLD = float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.65"))

# Rate limiting for scrapers (seconds between requests)
SCRAPE_DELAY = float(os.getenv("SCRAPE_DELAY", "2.0"))

# MTGDecks matchup data (legacy alias — use SUPPORTED_FORMATS for multi-format)
MTGDECKS_WINRATES_URL = os.getenv(
    "MTGDECKS_WINRATES_URL",
    SUPPORTED_FORMATS["standard"]["mtgdecks_url"],
)

# Graph construction
GRAPH_PATH = DATA_PROCESSED / "graph.pt"

# ── Model architecture ──
D_MODEL = int(os.getenv("D_MODEL", "128"))          # shared embedding dim for all nodes
D_MESSAGE = int(os.getenv("D_MESSAGE", "128"))       # message-passing hidden dim in GNN
D_COUNT = int(os.getenv("D_COUNT", "16"))            # count embedding dim
NUM_GNN_LAYERS = int(os.getenv("NUM_GNN_LAYERS", "2"))
DROPOUT = float(os.getenv("DROPOUT", "0.1"))

# Memory optimization: recompute GNN activations during backward instead of storing them.
# Cuts GNN memory ~60% at the cost of ~25% slower backward. Recommended for large graphs.
GRADIENT_CHECKPOINTING = bool(int(os.getenv("GRADIENT_CHECKPOINTING", "1")))

# Chunked message passing: process large edge types in mini-batches to cap
# GPU memory.  Set to 0 to disable chunking (process all edges at once).
EDGE_CHUNK_SIZE = int(os.getenv("EDGE_CHUNK_SIZE", "2000000"))

# Loss weights
COUNT_LOSS_WEIGHT = float(os.getenv("COUNT_LOSS_WEIGHT", "1.0"))
CONSENSUS_LOSS_WEIGHTING = bool(int(os.getenv("CONSENSUS_LOSS_WEIGHTING", "0")))
CONSENSUS_LOSS_MIN_WEIGHT = float(os.getenv("CONSENSUS_LOSS_MIN_WEIGHT", "0.2"))

# Scheduled sampling: anneal teacher forcing rate from START to END over training
SCHEDULED_SAMPLING_START = float(os.getenv("SCHEDULED_SAMPLING_START", "1.0"))
SCHEDULED_SAMPLING_END = float(os.getenv("SCHEDULED_SAMPLING_END", "0.4"))
SCHEDULED_SAMPLING_SCHEDULE = os.getenv("SCHEDULED_SAMPLING_SCHEDULE", "linear")

# Budget signal: inject partial deck state features into context encoder
BUDGET_SIGNAL = bool(int(os.getenv("BUDGET_SIGNAL", "1")))

# Temperature annealing on MLP card scorer during training
SCORER_TEMP_START = float(os.getenv("SCORER_TEMP_START", "2.0"))
SCORER_TEMP_END = float(os.getenv("SCORER_TEMP_END", "0.5"))
SCORER_TEMP_SCHEDULE = os.getenv("SCORER_TEMP_SCHEDULE", "linear")

# Deck construction constraints
MAX_DECK_SIZE = int(os.getenv("MAX_DECK_SIZE", "60"))
MAX_BASIC_LAND_COUNT = int(os.getenv("MAX_BASIC_LAND_COUNT", "20"))
MAX_NONBASIC_COUNT = int(os.getenv("MAX_NONBASIC_COUNT", "4"))

# Training hyperparameters — three independent learning rates by functional pathway:
#   GNN:      model.gnn (message passing embeddings)
#   Selector: context_encoder + card_selector (card selection pathway)
#   Count:    count heads + count_embedding + context_merge (count prediction pathway)
DECK_LEARNING_RATE = float(os.getenv("DECK_LEARNING_RATE", "0.0001"))      # selector LR (legacy alias)
GNN_LEARNING_RATE = float(os.getenv("GNN_LEARNING_RATE", "0.00001"))       # GNN LR
SELECTOR_LEARNING_RATE = float(os.getenv("SELECTOR_LEARNING_RATE", "0.0001"))  # card selector LR
COUNT_LEARNING_RATE = float(os.getenv("COUNT_LEARNING_RATE", "0.0001"))    # count prediction LR
GNN_LR_SCALE = float(os.getenv("GNN_LR_SCALE", "0.1"))  # legacy: GNN LR = base LR * this (ignored when GNN_LEARNING_RATE is set explicitly)
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.00001"))
DECK_NUM_EPOCHS = int(os.getenv("DECK_NUM_EPOCHS", "50"))
DECK_PATIENCE = int(os.getenv("DECK_PATIENCE", "10"))
DECK_RECENCY_DAYS = int(os.getenv("DECK_RECENCY_DAYS", "30"))
DECK_WARMUP_EPOCHS = int(os.getenv("DECK_WARMUP_EPOCHS", "5"))
DECK_LR_SCHEDULER = os.getenv("DECK_LR_SCHEDULER", "cosine")  # cosine | linear | none
DECK_NUM_ARCHETYPES = int(os.getenv("DECK_NUM_ARCHETYPES", "0"))  # 0 = all available
DECK_NUM_VAL_ARCHETYPES = int(os.getenv("DECK_NUM_VAL_ARCHETYPES", "3"))
DECK_NUM_CV_FOLDS = int(os.getenv("DECK_NUM_CV_FOLDS", "5"))
DECK_BATCH_SIZE = int(os.getenv("DECK_BATCH_SIZE", "0"))  # archetypes per step (0 = all)
DECK_BATCH_STRATEGY = os.getenv("DECK_BATCH_STRATEGY", "random")  # "random" | "color_balanced"
DECK_COLOR_LOSS_WEIGHTING = bool(int(os.getenv("DECK_COLOR_LOSS_WEIGHTING", "0")))
DECK_TOP_N_ARCHETYPES = int(os.getenv("DECK_TOP_N_ARCHETYPES", "0"))  # 0 = all, N = top N per format
DECK_VAL_EVERY = int(os.getenv("DECK_VAL_EVERY", "5"))    # epochs between validation

# Curriculum negative sampling — gradually expand the decoder's candidate card
# pool during training.  "Deck cards" (any card in any archetype's decklist) are
# always eligible; the curriculum controls how many additional distractor cards
# the decoder sees.  Validation always uses the full pool.
CURRICULUM_NEG_START = int(os.getenv("CURRICULUM_NEG_START", "150"))       # initial negatives
CURRICULUM_NEG_EPOCHS = int(os.getenv("CURRICULUM_NEG_EPOCHS", "20"))      # epochs to full pool
CURRICULUM_NEG_SCHEDULE = os.getenv("CURRICULUM_NEG_SCHEDULE", "linear")   # linear | exponential

# Legacy: still used by graph_builder for copy-count edge weight encoding
DECK_BASIC_LAND_MAX_COPIES = int(os.getenv("DECK_BASIC_LAND_MAX_COPIES", "20"))

# Transfer learning (pretrain on all formats, finetune on target format)
TRANSFER_FINETUNE_LR_SCALE = float(os.getenv("TRANSFER_FINETUNE_LR_SCALE", "0.1"))
TRANSFER_GNN_FREEZE_EPOCHS = int(os.getenv("TRANSFER_GNN_FREEZE_EPOCHS", "5"))
TRANSFER_FINETUNE_EPOCHS = int(os.getenv("TRANSFER_FINETUNE_EPOCHS", "30"))

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
