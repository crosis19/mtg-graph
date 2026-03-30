"""Generate interactive dashboard for the Autoregressive Deck Constructor.

Produces a self-contained HTML file with:
  1. Cross-validation summary metrics (Jaccard, Precision, Recall, Count MAE)
  2. Training curves (select loss + count loss, Gumbel temperature)
  3. Per-archetype predicted vs actual card tables with copy counts
  4. Hyperparameters panel

Reads from training_log.json and model checkpoints in the run directory.
"""

import json
import logging
from pathlib import Path

import torch

from src.config import GRAPH_PATH, MAX_BASIC_LAND_COUNT, MAX_NONBASIC_COUNT, RESULTS_DIR
from src.deck_predictor import MTGDeckModel
from src.graph_builder import BASIC_LAND_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _get_deck_predictions(data, model) -> dict:
    """Run greedy inference for each archetype and return predictions + actuals."""
    arch_names = data["archetype"].names
    card_names = data["card"].names

    # Build ground truth from contains edges
    arch_actual: dict[int, dict[str, int]] = {}
    et = ("archetype", "contains", "card")
    if et in data.edge_types:
        ei = data[et].edge_index
        ew = data[et].edge_weight
        for j in range(ei.shape[1]):
            a_idx = int(ei[0, j])
            c_idx = int(ei[1, j])
            card = card_names[c_idx]
            weight = float(ew[j])

            # Decode edge weight back to approximate copy count
            is_basic = card in BASIC_LAND_NAMES
            if is_basic:
                count = round(weight * MAX_BASIC_LAND_COUNT)
            else:
                count = round(weight * 4)
            count = max(1, count)
            arch_actual.setdefault(a_idx, {})[card] = count

    results = {}
    for a_idx, arch_name in enumerate(arch_names):
        pred_raw = model.predict_deck(data, a_idx)
        actual = arch_actual.get(a_idx, {})
        actual_set = set(actual.keys())

        # Convert card indices to names
        predicted = []
        for card_idx, count in pred_raw.items():
            name = card_names[card_idx]
            predicted.append({"card": name, "count": count, "in_actual": name in actual_set})

        pred_set = {p["card"] for p in predicted}
        hits = len(pred_set & actual_set)

        results[arch_name] = {
            "predicted": predicted,
            "actual": actual,
            "n_actual": len(actual),
            "hits": hits,
            "total_pred_copies": sum(p["count"] for p in predicted),
        }

    return results


def generate_deck_html(training_log: dict, deck_predictions: dict) -> str:
    """Generate the full interactive HTML dashboard."""
    hp = training_log.get("hyperparameters", {})
    fold_results = training_log.get("fold_results", [])

    # Compute CV summary
    jaccard_scores = [r.get("best_val_metric", 0) for r in fold_results]
    mean_jaccard = sum(jaccard_scores) / max(len(jaccard_scores), 1)

    # Get first fold's training curves
    if fold_results:
        first_fold = fold_results[0]
        train_losses = first_fold.get("train_losses", [])
        select_losses = [t.get("select_loss", 0) for t in train_losses]
        count_losses = [t.get("count_loss", 0) for t in train_losses]
        taus = [t.get("tau", 1.0) for t in train_losses]
        best_epoch = first_fold.get("best_epoch", 0)

        last_val = (first_fold.get("val_metrics_history", [{}])[-1]
                    if first_fold.get("val_metrics_history") else {})
        precision = last_val.get("precision", 0)
        recall = last_val.get("recall", 0)
        count_mae = last_val.get("count_mae", 0)
        budget_compliance = last_val.get("budget_compliance", 0)
    else:
        select_losses, count_losses, taus = [], [], []
        best_epoch = 0
        precision = recall = count_mae = budget_compliance = 0

    # Build archetype cards HTML
    arch_cards_html = ""
    for arch_name, pred_data in sorted(deck_predictions.items()):
        actual = pred_data["actual"]
        actual_set = set(actual.keys())
        n_actual = pred_data["n_actual"]
        hits = pred_data["hits"]
        total = pred_data["total_pred_copies"]

        rows = ""
        for rank, p in enumerate(pred_data["predicted"], 1):
            in_actual = p["in_actual"]
            gt_count = actual.get(p["card"], None)

            if in_actual and gt_count is not None:
                if p["count"] == gt_count:
                    badge = f'<span class="badge badge-green">EXACT x{gt_count}</span>'
                else:
                    badge = f'<span class="badge badge-yellow">GT x{gt_count}</span>'
            else:
                badge = '<span class="badge badge-red">NOT IN DECK</span>'

            rows += f"""
            <tr>
              <td style="color:#888">{rank}</td>
              <td style="font-weight:600">x{p['count']}</td>
              <td>{p['card']}</td>
              <td>{badge}</td>
            </tr>"""

        budget_tag = "OK" if total == 60 else f"FAIL ({total})"
        arch_cards_html += f"""
        <div class="arch-section">
          <div class="arch-header">
            <span class="arch-name">{arch_name}</span>
            <span class="arch-stats">{hits}/{n_actual} cards found &middot; {total} total copies ({budget_tag})</span>
          </div>
          <table>
            <thead><tr><th>#</th><th>Copies</th><th>Card</th><th>Status</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>"""

    select_losses_json = json.dumps(select_losses)
    count_losses_json = json.dumps(count_losses)
    taus_json = json.dumps(taus)
    num_epochs = hp.get("num_epochs", 50)

    # CV fold summary
    fold_summary_html = ""
    for r in fold_results:
        val_archs = ", ".join(r.get("val_archetypes", []))
        jacc = r.get("best_val_metric", 0)
        fold_summary_html += f"""
        <tr>
          <td>Fold {r.get('fold', '?')}</td>
          <td>{val_archs}</td>
          <td style="font-weight:600">{jacc:.3f}</td>
          <td>{r.get('best_epoch', '?')}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MTG Autoregressive Deck Constructor Dashboard</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f0f12; color: #e0e0db;
      min-height: 100vh; padding: 24px 16px;
    }}
    .header {{ text-align: center; margin-bottom: 24px; }}
    h1 {{ font-size: 22px; font-weight: 600; color: #fff; margin-bottom: 4px; }}
    .subtitle {{ font-size: 13px; color: #888; }}
    .grid {{
      display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
      max-width: 1400px; margin: 0 auto;
    }}
    .panel {{
      background: #1a1a22; border: 1px solid #2a2a35;
      border-radius: 12px; overflow: hidden;
    }}
    .panel-header {{
      padding: 12px 16px; border-bottom: 1px solid #2a2a35;
      font-size: 14px; font-weight: 600; color: #fff;
    }}
    .panel-body {{ padding: 12px 16px; }}
    .full-width {{ grid-column: 1 / -1; }}
    .stat-grid {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; }}
    .stat {{ text-align: center; padding: 12px 8px; background: #12121a; border-radius: 8px; }}
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
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 600; }}
    .badge-green {{ background: rgba(29,158,117,0.2); color: #1D9E75; }}
    .badge-yellow {{ background: rgba(186,117,23,0.2); color: #BA7517; }}
    .badge-red {{ background: rgba(231,76,60,0.15); color: #E24B4A; }}
    .arch-section {{ margin-bottom: 16px; border: 1px solid #2a2a35; border-radius: 8px; overflow: hidden; }}
    .arch-header {{
      padding: 10px 12px; background: #12121a;
      display: flex; justify-content: space-between; align-items: center;
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
    <h1>MTG Autoregressive Deck Constructor</h1>
    <p class="subtitle">HeteroGNN + Cross-Attention &mdash; Gumbel-Softmax Autoregressive Deck Building</p>
  </div>

  <div class="grid" style="margin-bottom:16px">
    <div class="panel full-width">
      <div class="panel-body">
        <div class="stat-grid">
          <div class="stat">
            <div class="stat-value">{mean_jaccard:.1%}</div>
            <div class="stat-label">Mean Jaccard</div>
          </div>
          <div class="stat">
            <div class="stat-value">{precision:.1%}</div>
            <div class="stat-label">Precision</div>
          </div>
          <div class="stat">
            <div class="stat-value">{recall:.1%}</div>
            <div class="stat-label">Recall</div>
          </div>
          <div class="stat">
            <div class="stat-value">{count_mae:.2f}</div>
            <div class="stat-label">Count MAE</div>
          </div>
          <div class="stat">
            <div class="stat-value">{budget_compliance:.0%}</div>
            <div class="stat-label">Budget OK</div>
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
    <!-- CV Fold Results -->
    <div class="panel">
      <div class="panel-header">Cross-Validation Folds</div>
      <div class="panel-body">
        <table>
          <thead><tr><th>Fold</th><th>Held-out Archetypes</th><th>Jaccard</th><th>Best Epoch</th></tr></thead>
          <tbody>{fold_summary_html}</tbody>
        </table>
      </div>
    </div>

    <!-- Hyperparameters -->
    <div class="panel">
      <div class="panel-header">Hyperparameters</div>
      <div class="panel-body">
        <dl class="hp-grid">
          <dt>Architecture</dt><dd>GNN + Autoregressive</dd>
          <dt>d_model</dt><dd>{hp.get('d_model', '?')}</dd>
          <dt>d_count</dt><dd>{hp.get('d_count', '?')}</dd>
          <dt>Attn Heads</dt><dd>{hp.get('num_attn_heads', '?')}</dd>
          <dt>GNN Layers</dt><dd>{hp.get('num_gnn_layers', '?')}</dd>
          <dt>Dropout</dt><dd>{hp.get('dropout', '?')}</dd>
          <dt>LR</dt><dd>{hp.get('learning_rate', '?')}</dd>
          <dt>Weight Decay</dt><dd>{hp.get('weight_decay', '?')}</dd>
          <dt>Gumbel &tau;</dt><dd>{hp.get('gumbel_tau_start', '?')} &rarr; {hp.get('gumbel_tau_min', '?')}</dd>
          <dt>Gumbel Decay</dt><dd>{hp.get('gumbel_decay', '?')}</dd>
          <dt>Count Loss &lambda;</dt><dd>{hp.get('count_loss_weight', '?')}</dd>
          <dt>LR Schedule</dt><dd>{hp.get('warmup_epochs', 0)}ep warmup &rarr; {hp.get('lr_scheduler', '?')}</dd>
          <dt>Recency</dt><dd>{hp.get('recency_days', '?')}d</dd>
        </dl>
      </div>
    </div>

    <!-- Training Curves -->
    <div class="panel full-width">
      <div class="panel-header">Training Curves (Fold 0)</div>
      <div class="panel-body">
        <svg id="loss-chart" viewBox="0 0 900 250"></svg>
      </div>
    </div>

    <!-- Per-archetype predictions -->
    <div class="panel full-width">
      <div class="panel-header">Predicted Decklists (All Archetypes)</div>
      <div class="panel-body">
        {arch_cards_html}
      </div>
    </div>
  </div>

  <script>
    const selectLosses = {select_losses_json};
    const countLosses = {count_losses_json};
    const tauValues = {taus_json};
    const bestEpoch = {best_epoch};

    const margin = {{top: 20, right: 60, bottom: 35, left: 50}};
    const width = 900 - margin.left - margin.right;
    const height = 250 - margin.top - margin.bottom;

    const svg = d3.select('#loss-chart')
      .append('g').attr('transform', `translate(${{margin.left}},${{margin.top}})`);

    const nEpochs = Math.max(selectLosses.length, 1);
    const xScale = d3.scaleLinear().domain([1, nEpochs]).range([0, width]);

    const allLoss = selectLosses.concat(countLosses).filter(v => isFinite(v) && v > 0);
    const maxLoss = allLoss.length > 0 ? d3.max(allLoss) * 1.1 : 1;
    const yLoss = d3.scaleLinear().domain([0, maxLoss]).range([height, 0]);
    const yTau = d3.scaleLinear().domain([0, 1.1]).range([height, 0]);

    // Axes
    svg.append('g').attr('transform', `translate(0,${{height}})`)
      .call(d3.axisBottom(xScale).ticks(10))
      .selectAll('text,line,path').attr('stroke','#555').attr('fill','#888').attr('font-size',9);
    svg.append('g').call(d3.axisLeft(yLoss).ticks(5))
      .selectAll('text,line,path').attr('stroke','#555').attr('fill','#888').attr('font-size',9);
    svg.append('g').attr('transform', `translate(${{width}},0)`)
      .call(d3.axisRight(yTau).ticks(5))
      .selectAll('text,line,path').attr('stroke','#555').attr('fill','#E8A838').attr('font-size',9);

    // Select loss line
    const selLine = d3.line().defined(d => isFinite(d)).x((d,i) => xScale(i+1)).y(d => yLoss(d));
    svg.append('path').datum(selectLosses)
      .attr('fill','none').attr('stroke','#7F77DD').attr('stroke-width',1.5).attr('d', selLine);

    // Count loss line
    const cntLine = d3.line().defined(d => isFinite(d)).x((d,i) => xScale(i+1)).y(d => yLoss(d));
    svg.append('path').datum(countLosses)
      .attr('fill','none').attr('stroke','#1D9E75').attr('stroke-width',1.5).attr('d', cntLine);

    // Temperature line (right axis)
    const tauLine = d3.line().defined(d => isFinite(d)).x((d,i) => xScale(i+1)).y(d => yTau(d));
    svg.append('path').datum(tauValues)
      .attr('fill','none').attr('stroke','#E8A838').attr('stroke-width',1)
      .attr('stroke-dasharray','4,3').attr('d', tauLine);

    // Best epoch marker
    if (bestEpoch > 0) {{
      svg.append('line')
        .attr('x1', xScale(bestEpoch)).attr('x2', xScale(bestEpoch))
        .attr('y1', 0).attr('y2', height)
        .attr('stroke','#E24B4A').attr('stroke-dasharray','4,3').attr('stroke-opacity',0.6);
    }}

    // Axis labels
    svg.append('text').attr('x', width/2).attr('y', height+30)
      .attr('text-anchor','middle').attr('fill','#888').attr('font-size',11).text('Epoch');
    svg.append('text').attr('transform','rotate(-90)')
      .attr('x', -height/2).attr('y', -35)
      .attr('text-anchor','middle').attr('fill','#888').attr('font-size',11).text('Loss');

    // Legend
    const legend = svg.append('g').attr('transform', `translate(${{width-180}},5)`);
    [['Select Loss','#7F77DD'],['Count Loss','#1D9E75'],['Gumbel \\u03C4','#E8A838'],['Best','#E24B4A']].forEach(([label,color],i) => {{
      legend.append('rect').attr('x',0).attr('y',i*16).attr('width',12).attr('height',3).attr('fill',color);
      legend.append('text').attr('x',16).attr('y',i*16+4).attr('fill','#888').attr('font-size',10).text(label);
    }});
  </script>
</body>
</html>"""

    return html


def main(run_dir=None, fold_idx=0):
    """Generate deck constructor dashboard and save to the run directory."""
    if run_dir is None:
        run_dir = RESULTS_DIR / "deck" / "latest"
    run_dir = Path(run_dir)

    # Load training log
    log_path = run_dir / "training_log.json"
    if log_path.exists():
        with open(log_path) as f:
            training_log = json.load(f)
    else:
        log.warning(f"No training_log.json found at {log_path}")
        training_log = {}

    # Find model for the specified fold
    model_path = run_dir / f"model_fold{fold_idx}.pt"
    if not model_path.exists():
        model_path = run_dir / "model.pt"

    # Load graph and model
    log.info("Loading graph and model...")
    data = torch.load(GRAPH_PATH, weights_only=False)
    hp = training_log.get("hyperparameters", {})

    # Load checkpoint (contains hyperparameters)
    checkpoint = torch.load(model_path, weights_only=False)
    ckpt_hp = checkpoint.get("hyperparameters", {})
    # Prefer checkpoint HPs over training log HPs
    for k, v in ckpt_hp.items():
        if k not in hp:
            hp[k] = v

    node_dims = {nt: data[nt].x.shape[1] for nt in data.node_types}
    card_names = data["card"].names

    model = MTGDeckModel(
        node_dims=node_dims,
        edge_types=list(data.edge_types),
        node_types=list(data.node_types),
        card_names=card_names,
        d_model=hp.get("d_model", 128),
        d_message=hp.get("d_message", 128),
        d_count=hp.get("d_count", 16),
        num_gnn_layers=hp.get("num_gnn_layers", 2),
        num_attn_heads=hp.get("num_attn_heads", 4),
        dropout=0.0,  # no dropout at inference
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    log.info("Getting deck predictions...")
    deck_predictions = _get_deck_predictions(data, model)

    log.info("Generating HTML dashboard...")
    html = generate_deck_html(training_log, deck_predictions)

    dashboard_path = run_dir / "dashboard.html"
    dashboard_path.write_text(html, encoding="utf-8")
    log.info(f"Dashboard saved to {dashboard_path}")
    log.info(f"Open in browser: file:///{dashboard_path.as_posix()}")


if __name__ == "__main__":
    main()
