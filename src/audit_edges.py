"""Edge quality audit tool — spot-check synergy, counter, and removal edges.

Samples random edges from the processed edge files, joins with card oracle
text, and prints human-readable pairs for manual quality assessment.

Usage:
    python -m src.audit_edges --type semantic --n 50 --threshold 0.65
    python -m src.audit_edges --type counter --n 50
    python -m src.audit_edges --type removal --n 50
"""

import argparse
import random
import textwrap

import pandas as pd

from src.config import (
    CARDS_PARQUET,
    COUNTER_EDGES_PATH,
    REMOVAL_EDGES_PATH,
    SEMANTIC_EDGES_PATH,
)


def _load_cards() -> dict[str, str]:
    """Load card name → oracle text mapping."""
    cards = pd.read_parquet(CARDS_PARQUET)
    oracle = {}
    for _, row in cards.iterrows():
        name = row["name"]
        text = str(row.get("oracle_text", "")) if pd.notna(row.get("oracle_text")) else ""
        oracle[name] = text
    return oracle


def _wrap(text: str, width: int = 72, indent: str = "      ") -> str:
    """Wrap text with indentation for readability."""
    lines = textwrap.wrap(text, width=width) if text else ["(no oracle text)"]
    return "\n".join(indent + line for line in lines)


def audit_semantic(n: int, threshold: float | None):
    """Audit semantic_synergy edges."""
    oracle = _load_cards()
    edges = pd.read_parquet(SEMANTIC_EDGES_PATH)

    # Detect column names
    if "card_a" in edges.columns:
        col_a, col_b, col_score = "card_a", "card_b", "similarity"
    elif "source" in edges.columns:
        col_a, col_b, col_score = "source", "target", "similarity"
    else:
        cols = list(edges.columns)
        print(f"Unknown columns: {cols}")
        return

    if threshold is not None and col_score in edges.columns:
        edges = edges[edges[col_score] >= threshold]

    print(f"\nSemantic synergy edges: {len(edges)} total (threshold >= {threshold})")
    print(f"Sampling {min(n, len(edges))} random edges\n")

    sample = edges.sample(n=min(n, len(edges)), random_state=random.randint(0, 9999))

    scores = []
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        a, b = row[col_a], row[col_b]
        score = row[col_score] if col_score in row.index else 0.0
        scores.append(score)
        print(f"--- Edge {i} (similarity: {score:.4f}) ---")
        print(f"  A: {a}")
        print(_wrap(oracle.get(a, "(not found)")))
        print(f"  B: {b}")
        print(_wrap(oracle.get(b, "(not found)")))
        print()

    if scores:
        print(f"Summary: min={min(scores):.4f}, max={max(scores):.4f}, "
              f"mean={sum(scores)/len(scores):.4f}, n={len(scores)}")


def audit_counter(n: int):
    """Audit counter edges."""
    oracle = _load_cards()
    edges = pd.read_parquet(COUNTER_EDGES_PATH)

    print(f"\nCounter edges: {len(edges)} total")
    print(f"Columns: {list(edges.columns)}")
    print(f"Sampling {min(n, len(edges))} random edges\n")

    # Detect column names
    if "source" in edges.columns:
        col_a, col_b = "source", "target"
    else:
        cols = list(edges.columns)
        col_a, col_b = cols[0], cols[1]

    sample = edges.sample(n=min(n, len(edges)), random_state=random.randint(0, 9999))

    for i, (_, row) in enumerate(sample.iterrows(), 1):
        a, b = row[col_a], row[col_b]
        print(f"--- Edge {i} ---")
        print(f"  Source (counters): {a}")
        print(_wrap(oracle.get(a, "(not found)")))
        print(f"  Target (countered): {b}")
        print(_wrap(oracle.get(b, "(not found)")))
        print()


def audit_removal(n: int):
    """Audit removal edges."""
    oracle = _load_cards()
    edges = pd.read_parquet(REMOVAL_EDGES_PATH)

    print(f"\nRemoval edges: {len(edges)} total")
    print(f"Columns: {list(edges.columns)}")
    print(f"Sampling {min(n, len(edges))} random edges\n")

    # Detect column names
    if "source" in edges.columns:
        col_a, col_b = "source", "target"
    else:
        cols = list(edges.columns)
        col_a, col_b = cols[0], cols[1]
    col_w = "removal_weight" if "removal_weight" in edges.columns else None

    sample = edges.sample(n=min(n, len(edges)), random_state=random.randint(0, 9999))

    weights = []
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        a, b = row[col_a], row[col_b]
        w = row[col_w] if col_w else None
        if w is not None:
            weights.append(w)
        w_str = f" (weight: {w:.3f})" if w is not None else ""
        print(f"--- Edge {i}{w_str} ---")
        print(f"  Source (removes): {a}")
        print(_wrap(oracle.get(a, "(not found)")))
        print(f"  Target (removed): {b}")
        print(_wrap(oracle.get(b, "(not found)")))
        print()

    if weights:
        print(f"Summary: min={min(weights):.4f}, max={max(weights):.4f}, "
              f"mean={sum(weights)/len(weights):.4f}, n={len(weights)}")


def main():
    parser = argparse.ArgumentParser(description="Audit edge quality")
    parser.add_argument("--type", choices=["semantic", "counter", "removal"],
                        required=True, help="Edge type to audit")
    parser.add_argument("--n", type=int, default=50, help="Number of edges to sample")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Minimum similarity threshold (semantic only)")
    args = parser.parse_args()

    if args.type == "semantic":
        audit_semantic(args.n, args.threshold or 0.65)
    elif args.type == "counter":
        audit_counter(args.n)
    elif args.type == "removal":
        audit_removal(args.n)


if __name__ == "__main__":
    main()
