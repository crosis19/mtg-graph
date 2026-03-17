"""Phase 2, Layer 3: Semantic synergy via oracle text embeddings.

Embeds card oracle text using a sentence transformer, computes pairwise cosine
similarity, and creates semantic synergy edges where similarity exceeds a threshold
AND no mechanical or keyword edge already exists (to avoid redundancy).

Outputs:
  - card_embeddings.npy (N x D embedding matrix)
  - card_embedding_index.json (ordered list of card names matching rows)
  - semantic_edges.parquet (card-to-card semantic synergy edges)
"""

import json
import logging

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    CARDS_PARQUET,
    CARD_EMBEDDINGS_PATH,
    CARD_EMBEDDING_INDEX_PATH,
    SEMANTIC_EDGES_PATH,
    KEYWORD_MATRIX_PATH,
    MECHANICAL_EDGES_PATH,
    DATA_EMBEDDINGS,
    DATA_PROCESSED,
    EMBEDDING_MODEL,
    SEMANTIC_SIMILARITY_THRESHOLD,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def build_embedding_text(row: pd.Series) -> str:
    """Build a rich text representation for embedding.

    Combines oracle text, type line, and keywords into a single string
    that captures the card's strategic identity.
    """
    parts = []

    type_line = row.get("type_line", "")
    if type_line:
        parts.append(type_line)

    oracle_text = row.get("oracle_text", "")
    if oracle_text:
        parts.append(oracle_text)

    keywords = row.get("keywords", "")
    if keywords:
        parts.append(f"Keywords: {keywords.replace('|', ', ')}")

    mana_cost = row.get("mana_cost", "")
    if mana_cost:
        parts.append(f"Cost: {mana_cost}")

    return " | ".join(parts) if parts else "Unknown card"


def compute_embeddings(cards_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Compute sentence embeddings for all card oracle texts."""
    DATA_EMBEDDINGS.mkdir(parents=True, exist_ok=True)

    # Build embedding texts
    card_names = cards_df["name"].tolist()
    texts = cards_df.apply(build_embedding_text, axis=1).tolist()

    log.info("Embedding %d cards with model '%s'...", len(texts), EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=128)
    embeddings = np.array(embeddings)

    log.info("Embedding shape: %s", embeddings.shape)

    # Save embeddings and index
    np.save(CARD_EMBEDDINGS_PATH, embeddings)
    with open(CARD_EMBEDDING_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(card_names, f)

    log.info("Saved embeddings to %s", CARD_EMBEDDINGS_PATH)
    return embeddings, card_names


def load_existing_edges() -> set[tuple[str, str]]:
    """Load card pairs that already have keyword or mechanical synergy edges."""
    existing_pairs = set()

    for path in [KEYWORD_MATRIX_PATH, MECHANICAL_EDGES_PATH]:
        if path.exists():
            df = pd.read_parquet(path)
            for _, row in df.iterrows():
                pair = tuple(sorted([row["card_a"], row["card_b"]]))
                existing_pairs.add(pair)

    log.info("Loaded %d existing synergy pairs to exclude.", len(existing_pairs))
    return existing_pairs


def build_semantic_edges(
    embeddings: np.ndarray,
    card_names: list[str],
    threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
    exclude_existing: bool = True,
) -> pd.DataFrame:
    """Build semantic synergy edges from embedding similarity.

    Args:
        embeddings: (N, D) card embedding matrix.
        card_names: List of card names (same order as embedding rows).
        threshold: Minimum cosine similarity to create an edge.
        exclude_existing: If True, skip pairs that already have keyword/mechanical edges.
    """
    log.info(
        "Computing pairwise cosine similarity for %d cards (threshold=%.2f)...",
        len(card_names), threshold,
    )

    # Compute similarity matrix
    sim_matrix = cosine_similarity(embeddings)

    # Load existing edges to avoid redundancy
    existing_pairs = load_existing_edges() if exclude_existing else set()

    # Extract edges above threshold
    edges = []
    n = len(card_names)

    for i in range(n):
        for j in range(i + 1, n):
            sim = sim_matrix[i, j]
            if sim < threshold:
                continue

            pair = tuple(sorted([card_names[i], card_names[j]]))
            if pair in existing_pairs:
                continue

            edges.append({
                "card_a": pair[0],
                "card_b": pair[1],
                "edge_type": "semantic_synergy",  # renamed from co_occurrence
                "similarity": round(float(sim), 4),
            })

    df = pd.DataFrame(edges)
    log.info(
        "Found %d semantic synergy edges (threshold=%.2f, excluded %d existing).",
        len(df), threshold, len(existing_pairs),
    )
    return df


def run(threshold: float = SEMANTIC_SIMILARITY_THRESHOLD) -> pd.DataFrame:
    """Full pipeline: load cards → embed → similarity → edges → save."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    if not CARDS_PARQUET.exists():
        raise FileNotFoundError(
            f"Cards data not found at {CARDS_PARQUET}. Run ingest_cards.py first."
        )

    cards_df = pd.read_parquet(CARDS_PARQUET)

    # Compute or load embeddings
    if CARD_EMBEDDINGS_PATH.exists() and CARD_EMBEDDING_INDEX_PATH.exists():
        log.info("Loading cached embeddings...")
        embeddings = np.load(CARD_EMBEDDINGS_PATH)
        with open(CARD_EMBEDDING_INDEX_PATH, "r", encoding="utf-8") as f:
            card_names = json.load(f)

        # Check if stale (different card count)
        if len(card_names) != len(cards_df):
            log.info("Card count changed, recomputing embeddings.")
            embeddings, card_names = compute_embeddings(cards_df)
    else:
        embeddings, card_names = compute_embeddings(cards_df)

    edges_df = build_semantic_edges(embeddings, card_names, threshold=threshold)

    edges_df.to_parquet(SEMANTIC_EDGES_PATH, index=False)
    log.info("Saved %d semantic edges to %s", len(edges_df), SEMANTIC_EDGES_PATH)
    return edges_df


if __name__ == "__main__":
    run()
