# MTG Deck Composition Predictor

A two-stage GNN + autoregressive MLP system that predicts competitive 60-card Magic: The Gathering decklists given an archetype identity. Supports multiple formats (Standard, Pioneer) in a unified graph with transfer learning.

## Overview

This project models MTG competitive metagames as a heterogeneous graph with three node types (card, archetype, set) and 11 edge types, then trains an autoregressive deck constructor to build complete 60-card decklists one card at a time. Multiple formats share card nodes in a unified graph, enabling transfer learning from Pioneer to Standard.

### Key Features

- **Multi-format support** — Standard and Pioneer archetypes in a unified graph with shared card embeddings and format-prefixed archetype nodes
- **Transfer learning** — pre-train on all formats, fine-tune on Standard with GNN freezing and reduced learning rate
- **Two-stage architecture** — HeteroGNN learns embeddings via message passing, then an MLP-scored autoregressive decoder constructs decks card-by-card
- **Edge features with learnable edge-type weights** — graph edge weights (copy counts, synergy scores, win rates) are treated as input features to the MessageMLP, while learnable per-edge-type scalars control each relationship type's contribution
- **Scheduled sampling** — anneals teacher forcing rate from 100% to 40% over training, forcing the model to recover from its own mistakes
- **Consensus-weighted loss** — penalizes missing staples (high inclusion rate) more heavily than flex slots
- **Deterministic card ordering** — sorts target cards by consensus rate for context correction, teaching "staples first, flex later"
- **Budget signal** — injects remaining budget, color density, and type density as explicit features into the context encoder
- **Temperature-annealed scoring** — softmax temperature on card selection scores starts high (flat gradients) and anneals low (sharp gradients)
- **Separate count heads** — non-basic cards (1-4 copies) and basic lands (1-20 copies) use dedicated classification heads
- **Archetype-holdout cross-validation** — tests generalization to unseen archetypes
- **30-day recency filter** — uses only recent decklists from MTGGoldfish
- **Early stopping on Jaccard similarity** — checkpoints on the metric that matters
- **Edge quality audit** — standalone tool to spot-check semantic, counter, and removal edges

## Architecture

### Heterogeneous Graph Structure

| Node Type | Count | Features |
|-----------|-------|----------|
| Card | ~4,168 (Standard) / ~8-10k (Standard+Pioneer) | 384-dim semantic embedding + 23 scalar features (cmc, colors, types, rarity, mana generation) |
| Archetype | ~16 Standard + ~32 Pioneer (format-prefixed) | 16-dim (6 color flags + 8 type ratios + 2 format one-hot) |
| Set | Dynamic (all sets with legal cards) | 385-dim (mean card embedding + recency score) |

| Edge Type | Description |
|-----------|-------------|
| `(card, keyword_synergy, card)` | Complementary keywords (e.g., deathtouch + first strike) |
| `(card, semantic_synergy, card)` | Oracle text embedding similarity >= 0.65 |
| `(card, counters/countered_by, card)` | Counterspell interactions (directed + reverse) |
| `(card, removes/removed_by, card)` | Removal interactions (directed + reverse) |
| `(set, printed_in, card)` | Set membership |
| `(card, member_of, set)` | Reverse of printed_in |
| `(archetype, contains, card)` | Archetype includes card (feature: copy count) |
| `(card, in_deck, archetype)` | Reverse of contains |
| `(archetype, win_rate, archetype)` | Head-to-head win rate |
| `(archetype, lose_rate, archetype)` | Head-to-head loss rate |

### Model

**Stage 1 (HeteroGNN):** MLP-based message passing with additive skip connections. Edge weights from the graph (copy counts, synergy scores, win rates) are concatenated as input features to edge-type-specific MessageMLPs. Learnable per-edge-type scalar weights control each relationship type's contribution.

**Stage 2 (Autoregressive Deck Constructor):** Given archetype embeddings from the GNN, builds a 60-card deck one card at a time using MLP scoring over the card pool with separate count classification heads for non-basic cards (1-4) and basic lands (1-20).

- **Training**: Argmax selection with scheduled sampling, consensus-weighted loss, deterministic card ordering, budget signal, and temperature-annealed scoring
- **Metrics**: Jaccard similarity, precision, recall, exact count match

## Project Structure

```
mtg-graph/
├── src/
│   ├── config.py              # Paths, constants, format registry, hyperparameters
│   ├── ingest_all.py          # CLI orchestrator for Phases 1-3 with validation
│   ├── ingest_cards.py        # Phase 1: Download cards from Scryfall API
│   ├── ingest_metagame.py     # Phase 1: Scrape metagame + decklists (MTGGoldfish)
│   ├── ingest_matchups.py     # Phase 1: Scrape matchup win rates (MTGDecks)
│   ├── oracle_parser.py       # Utility for parsing card oracle text
│   ├── keyword_matrix.py      # Phase 2: Keyword synergy edge extraction
│   ├── card_embeddings.py     # Phase 2: Semantic embeddings (SentenceTransformer)
│   ├── synergy_eval.py        # Phase 2: Synergy edge quality evaluation
│   ├── card_interaction_edges.py # Phase 2: Counter/removal edge extraction
│   ├── audit_edges.py         # Edge quality audit (spot-check synergy/counter/removal)
│   ├── graph_builder.py       # Phase 3: Assemble multi-format HeteroData graph
│   ├── deck_predictor.py      # Phase 4: HeteroGNN + autoregressive deck constructor
│   ├── train_deck.py          # Phase 4: Training loop with pretrain/finetune modes
│   ├── visualize_deck.py      # Phase 5: Interactive HTML dashboard
│   └── synthetic_data.py      # Generate synthetic data for development
├── data/                      # Generated by pipeline (not committed)
│   ├── raw/                   #   Scryfall bulk JSON
│   ├── processed/             #   Shared: cards.parquet, synergy edges, graph.pt
│   │   ├── standard/          #   Standard: metagame, decklists, matchups
│   │   └── pioneer/           #   Pioneer: metagame, decklists, matchups
│   └── embeddings/            #   Card embeddings (shared)
├── results/
│   └── deck/                  # Training runs (timestamped)
├── notebooks/
│   ├── train_deck_colab.ipynb # GPU training on Google Colab
│   ├── sweep_deck_colab.ipynb # Optuna hyperparameter sweep
│   └── inference_deck_colab.ipynb # Inference / deck generation
├── pyproject.toml
├── requirements.txt
└── .env.example
```

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended but not required)

### Installation

```bash
git clone https://github.com/<your-username>/mtg-graph.git
cd mtg-graph

python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

pip install -e .
python -m spacy download en_core_web_sm
```

### Configuration

```bash
cp .env.example .env
# Edit .env to adjust hyperparameters as needed
```

## Usage

### Data Ingestion (Phases 1-3)

```bash
python -m src.ingest_all                      # Full pipeline, all formats
python -m src.ingest_all --phase 1            # Phase 1 only (scrape)
python -m src.ingest_all --phase 2            # Phase 2 only (synergy edges)
python -m src.ingest_all --phase 3            # Phase 3 only (build graph)
python -m src.ingest_all --format standard    # Single format only
python -m src.ingest_all --skip-scrape        # Phases 2-3 (reuse scraped data)
python -m src.ingest_all --validate-only      # Check existing output files
```

### Training (Phase 4)

```bash
# Pre-train on all formats (Standard + Pioneer)
python -m src.train_deck --mode pretrain

# Fine-tune on Standard using a pretrained checkpoint
python -m src.train_deck --mode finetune \
    --format-filter standard \
    --pretrained-path results/deck/latest/model_fold0.pt

# GPU training via Colab — see notebooks/train_deck_colab.ipynb
```

Each training run saves to a timestamped folder under `results/deck/`:
- `model_fold*.pt` — best model checkpoint per fold
- `training_log.json` — hyperparameters, epoch curves, final metrics
- `dashboard.html` — interactive visualization

### Edge Quality Audit

```bash
python -m src.audit_edges --type semantic --n 50 --threshold 0.65
python -m src.audit_edges --type counter --n 50
python -m src.audit_edges --type removal --n 50
```

### Dashboard Generation

```bash
python -m src.visualize_deck
```

## Data Sources

All data is downloaded or scraped by the pipeline — no data files are committed to the repo. Per-format data (metagame, decklists, matchups) is stored in `data/processed/{format}/` subdirectories.

| Source | Data | Module |
|--------|------|--------|
| [Scryfall API](https://scryfall.com/docs/api) | Card oracle text, attributes, set info (shared across formats) | `ingest_cards.py` |
| [MTGGoldfish](https://www.mtggoldfish.com/metagame/) | Metagame share + archetype decklists per format (30-day paper) | `ingest_metagame.py` |
| [MTGDecks](https://www.mtgdecks.net) | Archetype matchup win rates per format | `ingest_matchups.py` |

### Active Formats

Configured via `ACTIVE_FORMATS` in `config.py` or `ACTIVE_FORMATS` env var (comma-separated):

| Format | Cards | Archetypes | MTGGoldfish Path | MTGDecks Path |
|--------|-------|------------|------------------|---------------|
| Standard | ~4,168 | 16 | `/metagame/standard` | `/Standard/winrates/` |
| Pioneer | ~8-10k | 32 | `/metagame/pioneer` | `/Pioneer/winrates/` |

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
