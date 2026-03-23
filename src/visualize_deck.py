"""Generate interactive dashboard for the Deck Composition Predictor.

Produces a self-contained HTML file with:
  1. Model performance stats (Hits@K, MRR, F1)
  2. Training curves (BPR loss, Hits@25, F1)
  3. Per-archetype predicted vs actual card tables
  4. Hyperparameters panel

Reads from training_log.json and model checkpoint in the run directory.
"""

import json
import logging
from pathlib import Path

import torch

from src.config import (
    DECKLISTS_PARQUET,
    GRAPH_PATH,
    RESULTS_DIR,
)
from src.deck_predictor import DeckPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _get_deck_predictions(data, model, top_n: int = 20) -> dict:
    """Score all cards for each archetype and return top predictions + actuals."""
    model.eval()
    with torch.no_grad():
        output = model(data)
        node_emb = output["node_embeddings"]
        card_emb = node_emb["card"]
        arch_emb = node_emb["archetype"]

    arch_names = data["archetype"].names
    card_names = data["card"].names
    n_cards = card_emb.shape[0]
    all_card_idx = torch.arange(n_cards, device=card_emb.device)

    # Build ground truth per archetype
    arch_actual: dict[int, set[str]] = {}
    for edge_name in ["maindecks", "sideboards"]:
        et = ("archetype", edge_name, "card")
        if et in data.edge_types:
            ei = data[et].edge_index
            for j in range(ei.shape[1]):
                a_idx = int(ei[0, j])
                c_idx = int(ei[1, j])
                arch_actual.setdefault(a_idx, set()).add(card_names[c_idx])

    results = {}
    with torch.no_grad():
        for a_idx, arch_name in enumerate(arch_names):
            arch_indices = torch.full(
                (n_cards,), a_idx, dtype=torch.long, device=arch_emb.device
            )
            logits = model.predict_deck(card_emb, arch_emb, all_card_idx, arch_indices)
            probs = torch.sigmoid(logits)

            top_k = torch.topk(probs, min(top_n, n_cards))
            predicted = [
                {"card": card_names[i], "prob": float(probs[i])}
                for i in top_k.indices
            ]

            actual_cards = sorted(arch_actual.get(a_idx, set()))

            results[arch_name] = {
                "predicted": predicted,
                "actual": actual_cards,
                "n_actual": len(actual_cards),
            }

    return results


def generate_deck_html(training_log: dict, deck_predictions: dict) -> str:
    """Generate the full interactive HTML dashboard."""

    metrics = training_log.get("final_metrics", {})
    test = metrics.get("test", {})
    hp = training_log.get("hyperparameters", {})
    curves = training_log.get("training_curves", {})
    best_epoch = training_log.get("best_epoch", 0)
    loss_type = training_log.get("loss_type", "bce")

    train_losses = curves.get("train_losses", [])
    # Support both old and new format
    val_hits25 = curves.get("val_hits25", curves.get("val_accs", []))
    val_f1 = curves.get("val_f1", [])

    # Build archetype cards HTML
    arch_cards_html = ""
    for arch_name, pred_data in sorted(deck_predictions.items()):
        actual_set = set(pred_data["actual"])
        n_actual = pred_data["n_actual"]

        top_cards = pred_data["predicted"]
        hits_in_top = sum(1 for p in top_cards if p["card"] in actual_set)

        rows = ""
        for p in top_cards:
            in_actual = p["card"] in actual_set
            badge = '<span class="badge badge-green">IN DECK</span>' if in_actual else ""
            prob_color = "#1D9E75" if p["prob"] > 0.7 else "#BA7517" if p["prob"] > 0.4 else "#888"
            rows += f"""
            <tr>
              <td style="color:{prob_color};font-weight:600">{p['prob']:.1%}</td>
              <td>{p['card']}</td>
              <td>{badge}</td>
            </tr>"""

        arch_cards_html += f"""
        <div class="arch-section">
          <div class="arch-header">
            <span class="arch-name">{arch_name}</span>
            <span class="arch-stats">{hits_in_top}/{len(top_cards)} predicted in actual deck ({n_actual} cards total)</span>
          </div>
          <table>
            <thead><tr><th>Confidence</th><th>Card</th><th>Status</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>"""

    train_losses_json = json.dumps(train_losses)
    val_hits25_json = json.dumps(val_hits25)
    val_f1_json = json.dumps(val_f1)
    num_epochs = hp.get("num_epochs", 50)

    # Data split info
    data_split = training_log.get("data_split", {})
    split_type = data_split.get("split_type", "random")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MTG Deck Composition Predictor Dashboard</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f0f12;
      color: #e0e0db;
      min-height: 100vh;
      padding: 24px 16px;
    }}
    .header {{ text-align: center; margin-bottom: 24px; }}
    h1 {{ font-size: 22px; font-weight: 600; color: #fff; margin-bottom: 4px; }}
    .subtitle {{ font-size: 13px; color: #888; }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      max-width: 1400px;
      margin: 0 auto;
    }}
    .panel {{
      background: #1a1a22;
      border: 1px solid #2a2a35;
      border-radius: 12px;
      overflow: hidden;
    }}
    .panel-header {{
      padding: 12px 16px;
      border-bottom: 1px solid #2a2a35;
      font-size: 14px;
      font-weight: 600;
      color: #fff;
    }}
    .panel-body {{ padding: 12px 16px; }}
    .full-width {{ grid-column: 1 / -1; }}
    .stat-grid {{
      display: grid;
      grid-template-columns: repeat(6, 1fr);
      gap: 12px;
      margin-bottom: 8px;
    }}
    .stat {{
      text-align: center;
      padding: 12px 8px;
      background: #12121a;
      border-radius: 8px;
    }}
    .stat-value {{ font-size: 24px; font-weight: 700; color: #fff; }}
    .stat-label {{
      font-size: 11px; color: #666; margin-top: 4px;
      text-transform: uppercase; letter-spacing: 0.05em;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th {{
      text-align: left; padding: 6px 8px; color: #888; font-weight: 500;
      border-bottom: 1px solid #2a2a35; font-size: 11px;
      text-transform: uppercase; letter-spacing: 0.05em;
    }}
    td {{ padding: 6px 8px; border-bottom: 1px solid #1e1e28; }}
    tr:hover td {{ background: #22222e; }}
    .badge {{
      display: inline-block; padding: 2px 8px; border-radius: 4px;
      font-size: 10px; font-weight: 600;
    }}
    .badge-green {{ background: rgba(29,158,117,0.2); color: #1D9E75; }}
    .arch-section {{
      margin-bottom: 16px;
      border: 1px solid #2a2a35;
      border-radius: 8px;
      overflow: hidden;
    }}
    .arch-header {{
      padding: 10px 12px;
      background: #12121a;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .arch-name {{ font-weight: 600; color: #7F77DD; font-size: 14px; }}
    .arch-stats {{ font-size: 12px; color: #888; }}
    .arch-section table {{ margin: 0; }}
    svg {{ display: block; width: 100%; }}
    .hp-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4px 16px; font-size: 12px; }}
    .hp-grid dt {{ color: #888; }}
    .hp-grid dd {{ color: #e0e0db; font-weight: 600; }}
  </style>
</head>
<body>
  <div class="header">
    <h1>MTG Deck Composition Predictor</h1>
    <p class="subtitle">BPR Ranking with Leave-One-Out Pooling &mdash; Card-Archetype Inclusion Prediction</p>
  </div>

  <!-- Stats Row -->
  <div class="grid" style="margin-bottom:16px">
    <div class="panel full-width">
      <div class="panel-body">
        <div class="stat-grid">
          <div class="stat">
            <div class="stat-value">{test.get('hits_at_25', test.get('accuracy', 0)):.1%}</div>
            <div class="stat-label">Hits@25</div>
          </div>
          <div class="stat">
            <div class="stat-value">{test.get('hits_at_50', 0):.1%}</div>
            <div class="stat-label">Hits@50</div>
          </div>
          <div class="stat">
            <div class="stat-value">{test.get('mrr', 0):.3f}</div>
            <div class="stat-label">MRR</div>
          </div>
          <div class="stat">
            <div class="stat-value">{test.get('f1', 0):.1%}</div>
            <div class="stat-label">F1 Score</div>
          </div>
          <div class="stat">
            <div class="stat-value">{test.get('precision', 0):.1%}</div>
            <div class="stat-label">Precision</div>
          </div>
          <div class="stat">
            <div class="stat-value">{best_epoch}</div>
            <div class="stat-label">Best Epoch</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="grid">
    <!-- Training Curves -->
    <div class="panel">
      <div class="panel-header">Training Curves</div>
      <div class="panel-body">
        <svg id="loss-chart" viewBox="0 0 600 250"></svg>
      </div>
    </div>

    <!-- Hyperparameters -->
    <div class="panel">
      <div class="panel-header">Hyperparameters</div>
      <div class="panel-body">
        <dl class="hp-grid">
          <dt>Loss Type</dt><dd>{"BPR" if loss_type == "bpr" else "BCE"}</dd>
          <dt>Hidden Dim</dt><dd>{hp.get('hidden_dim', '?')}</dd>
          <dt>HGT Layers</dt><dd>{hp.get('num_hgt_layers', '?')}</dd>
          <dt>Attention Heads</dt><dd>{hp.get('num_heads', '?')}</dd>
          <dt>Dropout</dt><dd>{hp.get('dropout', '?')}</dd>
          <dt>Learning Rate</dt><dd>{hp.get('learning_rate', '?')}</dd>
          <dt>Weight Decay</dt><dd>{hp.get('weight_decay', '?')}</dd>
          <dt>N Negatives</dt><dd>{hp.get('n_negatives', hp.get('neg_ratio', '?'))}</dd>
          <dt>Epochs</dt><dd>{hp.get('num_epochs', '?')}</dd>
          <dt>Patience</dt><dd>{hp.get('patience', '?')}</dd>
          <dt>Checkpoint</dt><dd>{hp.get('checkpoint_metric', '?')}</dd>
          <dt>Freeze HGT</dt><dd>{hp.get('freeze_hgt', False)}</dd>
          <dt>Split</dt><dd>{split_type}</dd>
          <dt>Recency</dt><dd>{hp.get('recency_days', '?')}d</dd>
        </dl>
      </div>
    </div>

    <!-- Per-archetype predictions -->
    <div class="panel full-width">
      <div class="panel-header">Top Predicted Cards per Archetype</div>
      <div class="panel-body">
        {arch_cards_html}
      </div>
    </div>
  </div>

  <script>
    // Training curves
    const trainLosses = {train_losses_json};
    const valHits25 = {val_hits25_json};
    const valF1 = {val_f1_json};
    const numEpochs = {num_epochs};
    const bestEpoch = {best_epoch};

    const valEpochs = [1].concat(Array.from({{length: Math.floor(numEpochs/5)}}, (_,i) => (i+1)*5));

    const margin = {{top: 20, right: 50, bottom: 35, left: 50}};
    const width = 600 - margin.left - margin.right;
    const height = 250 - margin.top - margin.bottom;

    const svg = d3.select('#loss-chart')
      .append('g').attr('transform', `translate(${{margin.left}},${{margin.top}})`);

    // Loss scale (left axis)
    const xScale = d3.scaleLinear().domain([1, Math.max(numEpochs, trainLosses.length)]).range([0, width]);
    const yLoss = d3.scaleLinear()
      .domain([0, d3.max(trainLosses) * 1.1 || 1])
      .range([height, 0]);

    // Metric scale (right axis, 0-1)
    const yMetric = d3.scaleLinear().domain([0, 1]).range([height, 0]);

    // Axes
    svg.append('g').attr('transform', `translate(0,${{height}})`)
      .call(d3.axisBottom(xScale).ticks(10))
      .selectAll('text,line,path').attr('stroke','#555').attr('fill','#888').attr('font-size',9);

    svg.append('g')
      .call(d3.axisLeft(yLoss).ticks(5))
      .selectAll('text,line,path').attr('stroke','#555').attr('fill','#888').attr('font-size',9);

    svg.append('g').attr('transform', `translate(${{width}},0)`)
      .call(d3.axisRight(yMetric).ticks(5).tickFormat(d3.format('.0%')))
      .selectAll('text,line,path').attr('stroke','#555').attr('fill','#1D9E75').attr('font-size',9);

    // Train loss line
    const trainLine = d3.line()
      .x((d,i) => xScale(i+1))
      .y(d => yLoss(d));
    svg.append('path').datum(trainLosses)
      .attr('fill','none').attr('stroke','#7F77DD').attr('stroke-width',1.5)
      .attr('stroke-opacity',0.5).attr('d', trainLine);

    // Val Hits@25 line
    if (valHits25.length > 0) {{
      const hitsLine = d3.line()
        .x((d,i) => xScale(valEpochs[i] || 1))
        .y(d => yMetric(d));
      svg.append('path').datum(valHits25)
        .attr('fill','none').attr('stroke','#1D9E75').attr('stroke-width',2)
        .attr('d', hitsLine);
    }}

    // Val F1 line
    if (valF1.length > 0) {{
      const f1Line = d3.line()
        .x((d,i) => xScale(valEpochs[i] || 1))
        .y(d => yMetric(d));
      svg.append('path').datum(valF1)
        .attr('fill','none').attr('stroke','#BA7517').attr('stroke-width',2)
        .attr('stroke-dasharray','4,3')
        .attr('d', f1Line);
    }}

    // Best epoch line
    svg.append('line')
      .attr('x1', xScale(bestEpoch)).attr('x2', xScale(bestEpoch))
      .attr('y1', 0).attr('y2', height)
      .attr('stroke','#E24B4A').attr('stroke-dasharray','4,3').attr('stroke-opacity',0.6);

    // Labels
    svg.append('text').attr('x', width/2).attr('y', height+30)
      .attr('text-anchor','middle').attr('fill','#888').attr('font-size',11).text('Epoch');
    svg.append('text').attr('transform','rotate(-90)')
      .attr('x', -height/2).attr('y', -35)
      .attr('text-anchor','middle').attr('fill','#7F77DD').attr('font-size',11).text('BPR Loss');
    svg.append('text').attr('transform','rotate(90)')
      .attr('x', height/2).attr('y', -width-35)
      .attr('text-anchor','middle').attr('fill','#1D9E75').attr('font-size',11).text('Metrics');

    // Legend
    const legend = svg.append('g').attr('transform', `translate(${{width-180}},5)`);
    [['BPR Loss','#7F77DD'],['Hits@25','#1D9E75'],['F1','#BA7517'],['Best','#E24B4A']].forEach(([label,color],i) => {{
      legend.append('rect').attr('x',0).attr('y',i*16).attr('width',12).attr('height',3).attr('fill',color);
      legend.append('text').attr('x',16).attr('y',i*16+4).attr('fill','#888').attr('font-size',10).text(label);
    }});
  </script>
</body>
</html>"""

    return html


def main(run_dir=None, model_path=None):
    """Generate deck predictor dashboard and save to the run directory."""
    if run_dir is None:
        run_dir = RESULTS_DIR / "deck" / "latest"
    run_dir = Path(run_dir)

    if model_path is None:
        model_path = run_dir / "model.pt"

    # Load training log
    log_path = run_dir / "training_log.json"
    if log_path.exists():
        with open(log_path) as f:
            training_log = json.load(f)
    else:
        log.warning(f"No training_log.json found at {log_path}")
        training_log = {}

    # Load graph and model for live predictions
    log.info("Loading graph and model...")
    data = torch.load(GRAPH_PATH, weights_only=False)
    node_dims = {nt: data[nt].x.shape[1] for nt in data.node_types}

    model = DeckPredictor(
        metadata=data.metadata(),
        node_dims=node_dims,
        hidden_dim=training_log.get("hyperparameters", {}).get("hidden_dim", 128),
        num_heads=training_log.get("hyperparameters", {}).get("num_heads", 4),
        num_layers=training_log.get("hyperparameters", {}).get("num_hgt_layers", 3),
        dropout=0.0,  # No dropout during inference
    )

    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    log.info("Getting deck predictions...")
    deck_predictions = _get_deck_predictions(data, model, top_n=20)

    log.info("Generating HTML dashboard...")
    html = generate_deck_html(training_log, deck_predictions)

    dashboard_path = run_dir / "dashboard.html"
    dashboard_path.write_text(html, encoding="utf-8")
    log.info(f"Dashboard saved to {dashboard_path}")
    log.info(f"Open in browser: file:///{dashboard_path.as_posix()}")


if __name__ == "__main__":
    main()
