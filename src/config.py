"""Project-wide paths and constants."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EMBEDDINGS = PROJECT_ROOT / "data" / "embeddings"

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
MODEL_PATH = DATA_PROCESSED / "model.pt"
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
TEMPORAL_SPLIT_RATIO = float(os.getenv("TEMPORAL_SPLIT_RATIO", "0.75"))
