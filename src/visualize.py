"""Phase 5: Generate interactive visualization dashboard.

Produces a self-contained HTML file with:
  1. Force-directed graph of the metagame (archetypes, top cards, tournaments)
  2. Model prediction results (emergence rankings, top-8 predictions)
  3. Edge-type legend and hover details
  4. Comparison of predicted vs actual top-8 placements

Reads from the saved graph.pt and model.pt files.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import (
    CARDS_PARQUET,
    GRAPH_PATH,
    METAGAME_PARQUET,
    RESULTS_DIR,
    TOURNAMENTS_PARQUET,
    DECKLISTS_PARQUET,
    create_run_dir,
)
from src.model import MTGMetagameHGT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _get_top_cards_per_archetype(data, decklists: pd.DataFrame, top_n: int = 5) -> dict:
    """Get the most-played cards for each archetype."""
    if decklists.empty:
        return {}

    result = {}
    for arch in decklists["archetype"].unique():
        arch_cards = decklists[decklists["archetype"] == arch]
        if "copies" in arch_cards.columns:
            top = arch_cards.groupby("card_name")["copies"].sum().nlargest(top_n)
        else:
            top = arch_cards["card_name"].value_counts().head(top_n)
        result[arch] = list(top.index)
    return result


def _get_model_predictions(data, model_path=None) -> dict:
    """Run inference and get predictions.

    Parameters
    ----------
    model_path : Path, optional
        Path to model checkpoint. If None, uses latest run dir.
    """
    node_dims = {nt: data[nt].x.shape[1] for nt in data.node_types}

    model = MTGMetagameHGT(
        metadata=data.metadata(),
        node_dims=node_dims,
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        dropout=0.0,
    )

    if model_path is None:
        latest = RESULTS_DIR / "latest" / "model.pt"
        model_path = latest
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        output = model(data)
        node_emb = output["node_embeddings"]

        # Emergence scores
        emergence = output["emergence_scores"]
        arch_names = data["archetype"].names
        emergence_ranked = sorted(
            zip(arch_names, emergence.tolist()),
            key=lambda x: -x[1],
        )

        # Top8 predictions for all (arch, tournament) pairs
        n_arch = len(arch_names)
        n_tourn = data["tournament"].x.shape[0]
        event_ids = data["tournament"].event_ids

        arch_idx = []
        tourn_idx = []
        for a in range(n_arch):
            for t in range(n_tourn):
                arch_idx.append(a)
                tourn_idx.append(t)

        arch_tensor = torch.tensor(arch_idx, dtype=torch.long)
        tourn_tensor = torch.tensor(tourn_idx, dtype=torch.long)

        probs = model.predict_top8(
            node_emb["archetype"],
            node_emb["tournament"],
            arch_tensor,
            tourn_tensor,
        )

        # Reshape to (n_arch, n_tourn) matrix
        prob_matrix = probs.reshape(n_arch, n_tourn)

    return {
        "emergence": emergence_ranked,
        "top8_probs": prob_matrix.numpy(),
        "arch_names": arch_names,
        "event_ids": event_ids,
        "val_acc": checkpoint.get("val_acc", 0),
        "val_loss": checkpoint.get("val_loss", 0),
        "epoch": checkpoint.get("epoch", 0),
    }


def _build_graph_data(data, predictions: dict, top_cards: dict) -> dict:
    """Build JSON data for the D3 force graph."""
    arch_names = data["archetype"].names
    event_ids = data["tournament"].event_ids

    nodes = []
    links = []

    # Archetype nodes
    emergence_map = {name: score for name, score in predictions["emergence"]}
    for i, name in enumerate(arch_names):
        score = emergence_map.get(name, 0)
        meta_share = float(data["archetype"].x[i, 0]) * 100
        win_rate = float(data["archetype"].x[i, 1]) * 100
        nodes.append({
            "id": f"arch_{i}",
            "label": name,
            "type": "arch",
            "emergence": round(score, 4),
            "meta_share": round(meta_share, 1),
            "win_rate": round(win_rate, 1),
            "detail": f"Meta share: {meta_share:.1f}% | Win rate: {win_rate:.1f}% | Emergence: {score:+.4f}",
        })

    # Top cards — collect unique cards across all archetypes
    card_set = set()
    card_archetypes = {}  # card_name -> [archetype_names]
    for arch, cards in top_cards.items():
        for card in cards:
            card_set.add(card)
            card_archetypes.setdefault(card, []).append(arch)

    card_name_to_idx = {name: i for i, name in enumerate(data["card"].names)}
    for card_name in card_set:
        if card_name not in card_name_to_idx:
            continue
        decks = ", ".join(card_archetypes.get(card_name, []))
        nodes.append({
            "id": f"card_{card_name}",
            "label": card_name,
            "type": "card",
            "detail": f"Played in: {decks}",
        })

    # Tournament nodes
    tournaments = pd.read_parquet(TOURNAMENTS_PARQUET)
    for i, eid in enumerate(event_ids):
        row = tournaments[tournaments["event_id"] == eid]
        name = row["name"].values[0] if not row.empty else eid
        date = row["date"].values[0] if not row.empty and "date" in row.columns else ""
        players = row["player_count"].values[0] if not row.empty and "player_count" in row.columns else "?"

        # Get predicted top archetypes for this tournament
        probs = predictions["top8_probs"][:, i]
        top_pred = sorted(
            zip(arch_names, probs.tolist()),
            key=lambda x: -x[1],
        )[:3]
        pred_str = ", ".join(f"{n} ({p:.0%})" for n, p in top_pred)

        nodes.append({
            "id": f"tourn_{i}",
            "label": name[:30] if len(str(name)) > 30 else str(name),
            "type": "tourn",
            "detail": f"Date: {date} | Players: {players} | Predicted top: {pred_str}",
        })

    # Contains edges (archetype -> card)
    for arch, cards in top_cards.items():
        arch_idx = arch_names.index(arch) if arch in arch_names else -1
        if arch_idx < 0:
            continue
        for card_name in cards:
            if card_name not in card_name_to_idx:
                continue
            links.append({
                "source": f"arch_{arch_idx}",
                "target": f"card_{card_name}",
                "type": "contains",
            })

    # Co-occurrence edges between cards in the viz
    viz_card_names = {n["label"] for n in nodes if n["type"] == "card"}
    if ("card", "co_occurrence", "card") in data.edge_types:
        ei = data["card", "co_occurrence", "card"].edge_index
        ew = data["card", "co_occurrence", "card"].edge_weight
        card_names_list = data["card"].names
        seen_pairs = set()
        for k in range(ei.shape[1]):
            s, t = int(ei[0, k]), int(ei[1, k])
            pair = (min(s, t), max(s, t))
            if pair in seen_pairs:
                continue
            src_name = card_names_list[s]
            dst_name = card_names_list[t]
            if src_name in viz_card_names and dst_name in viz_card_names:
                seen_pairs.add(pair)
                links.append({
                    "source": f"card_{src_name}",
                    "target": f"card_{dst_name}",
                    "type": "co_occurrence",
                    "weight": round(float(ew[k]), 2),
                })

    # Counters edges
    if ("archetype", "counters", "archetype") in data.edge_types:
        ei = data["archetype", "counters", "archetype"].edge_index
        ew = data["archetype", "counters", "archetype"].edge_weight
        for k in range(ei.shape[1]):
            s, t = int(ei[0, k]), int(ei[1, k])
            w = float(ew[k])
            links.append({
                "source": f"arch_{s}",
                "target": f"arch_{t}",
                "type": "counters",
                "weight": round(w, 2),
            })

    # Top8 edges
    if ("archetype", "top8", "tournament") in data.edge_types:
        ei = data["archetype", "top8", "tournament"].edge_index
        ew = data["archetype", "top8", "tournament"].edge_weight
        for k in range(ei.shape[1]):
            s, t = int(ei[0, k]), int(ei[1, k])
            w = float(ew[k])
            links.append({
                "source": f"arch_{s}",
                "target": f"tourn_{t}",
                "type": "top8",
                "weight": round(w, 2),
            })

    # Set nodes (if present)
    if hasattr(data.get("set", None), "x") if "set" in data.node_types else False:
        set_codes = data["set"].codes if hasattr(data["set"], "codes") else []
        set_names_list = data["set"].names if hasattr(data["set"], "names") else set_codes
        for i, (code, sname) in enumerate(zip(set_codes, set_names_list)):
            recency = float(data["set"].x[i, 0]) * 100
            size = float(data["set"].x[i, 1])
            nodes.append({
                "id": f"set_{i}",
                "label": str(sname)[:25],
                "type": "set",
                "detail": f"Set: {sname} ({code}) | Recency: {recency:.0f}% | Cards: {size:.0f}",
            })

        # printed_in edges: set -> card (only for cards already in the viz)
        if ("set", "printed_in", "card") in data.edge_types:
            ei = data["set", "printed_in", "card"].edge_index
            viz_card_names = {n["label"] for n in nodes if n["type"] == "card"}
            card_names_list = data["card"].names
            for k in range(ei.shape[1]):
                s, t = int(ei[0, k]), int(ei[1, k])
                card_name = card_names_list[t]
                if card_name in viz_card_names:
                    links.append({
                        "source": f"set_{s}",
                        "target": f"card_{card_name}",
                        "type": "printed_in",
                    })

    return {"nodes": nodes, "links": links}


def generate_html(graph_data: dict, predictions: dict) -> str:
    """Generate the full interactive HTML dashboard."""
    nodes_json = json.dumps(graph_data["nodes"])
    links_json = json.dumps(graph_data["links"])

    # Build emergence table rows
    emergence_rows = ""
    for i, (name, score) in enumerate(predictions["emergence"]):
        direction = "rising" if score > 0 else "falling"
        arrow = "&#9650;" if score > 0 else "&#9660;"
        color = "#1D9E75" if score > 0 else "#E24B4A"
        emergence_rows += f"""
        <tr>
          <td>{i+1}</td>
          <td>{name}</td>
          <td style="color:{color};font-weight:600">{arrow} {score:+.4f}</td>
        </tr>"""

    # Build top8 probability matrix
    arch_names = predictions["arch_names"]
    event_ids = predictions["event_ids"]
    probs = predictions["top8_probs"]

    # For the heatmap, take the last few tournaments
    n_show = min(8, len(event_ids))
    tournaments = pd.read_parquet(TOURNAMENTS_PARQUET)
    tourn_labels = []
    for eid in event_ids[-n_show:]:
        row = tournaments[tournaments["event_id"] == eid]
        label = row["name"].values[0][:20] if not row.empty else eid
        tourn_labels.append(str(label))

    heatmap_data = []
    for i, arch in enumerate(arch_names):
        for j in range(len(event_ids) - n_show, len(event_ids)):
            heatmap_data.append({
                "arch": arch,
                "tourn": tourn_labels[j - (len(event_ids) - n_show)],
                "prob": round(float(probs[i, j]), 3),
            })
    heatmap_json = json.dumps(heatmap_data)
    tourn_labels_json = json.dumps(tourn_labels)
    arch_names_json = json.dumps(list(arch_names))

    val_acc = predictions["val_acc"]
    val_loss = predictions["val_loss"]
    epoch = predictions["epoch"]

    # Count graph stats
    n_cards = len([n for n in graph_data["nodes"] if n["type"] == "card"])
    n_arch = len([n for n in graph_data["nodes"] if n["type"] == "arch"])
    n_tourn = len([n for n in graph_data["nodes"] if n["type"] == "tourn"])
    n_sets = len([n for n in graph_data["nodes"] if n["type"] == "set"])
    n_edges = len(graph_data["links"])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MTG Metagame GNN Dashboard</title>
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
    .header {{
      text-align: center;
      margin-bottom: 24px;
    }}
    h1 {{
      font-size: 22px;
      font-weight: 600;
      color: #fff;
      margin-bottom: 4px;
    }}
    .subtitle {{
      font-size: 13px;
      color: #888;
    }}
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
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .panel-body {{
      padding: 12px 16px;
    }}
    .full-width {{
      grid-column: 1 / -1;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      padding: 10px 16px;
      border-bottom: 1px solid #2a2a35;
      font-size: 12px;
      color: #888;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 5px;
    }}
    .dot {{
      width: 10px; height: 10px;
      border-radius: 50%;
      display: inline-block;
    }}
    .line-sample {{
      width: 20px; height: 2px;
      display: inline-block;
    }}
    #graph {{ display: block; width: 100%; }}
    #info {{
      min-height: 42px;
      padding: 8px 16px;
      font-size: 12px;
      color: #888;
      border-top: 1px solid #2a2a35;
      line-height: 1.5;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th {{
      text-align: left;
      padding: 6px 8px;
      color: #888;
      font-weight: 500;
      border-bottom: 1px solid #2a2a35;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    td {{
      padding: 6px 8px;
      border-bottom: 1px solid #1e1e28;
    }}
    tr:hover td {{
      background: #22222e;
    }}
    .stat-grid {{
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 12px;
      margin-bottom: 8px;
    }}
    .stat {{
      text-align: center;
      padding: 12px 8px;
      background: #12121a;
      border-radius: 8px;
    }}
    .stat-value {{
      font-size: 24px;
      font-weight: 700;
      color: #fff;
    }}
    .stat-label {{
      font-size: 11px;
      color: #666;
      margin-top: 4px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    #heatmap {{
      width: 100%;
      overflow-x: auto;
    }}
    .badge {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 600;
    }}
    .badge-green {{ background: rgba(29,158,117,0.2); color: #1D9E75; }}
    .badge-red {{ background: rgba(226,75,74,0.2); color: #E24B4A; }}
  </style>
</head>
<body>
  <div class="header">
    <h1>MTG Standard Metagame GNN</h1>
    <p class="subtitle">Heterogeneous Graph Transformer &mdash; Archetype Emergence &amp; Tournament Top 8 Predictions</p>
  </div>

  <!-- Stats Row -->
  <div class="grid" style="margin-bottom:16px">
    <div class="panel full-width">
      <div class="panel-body">
        <div class="stat-grid">
          <div class="stat">
            <div class="stat-value">{n_arch}</div>
            <div class="stat-label">Archetypes</div>
          </div>
          <div class="stat">
            <div class="stat-value">{n_tourn}</div>
            <div class="stat-label">Tournaments</div>
          </div>
          <div class="stat">
            <div class="stat-value">{n_sets}</div>
            <div class="stat-label">Sets</div>
          </div>
          <div class="stat">
            <div class="stat-value">{val_acc:.1%}</div>
            <div class="stat-label">Val Accuracy</div>
          </div>
          <div class="stat">
            <div class="stat-value">{epoch}</div>
            <div class="stat-label">Best Epoch</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="grid">
    <!-- Graph Panel -->
    <div class="panel full-width">
      <div class="panel-header">Metagame Graph</div>
      <div class="legend">
        <span class="legend-item"><span class="dot" style="background:#7F77DD"></span>Archetype</span>
        <span class="legend-item"><span class="dot" style="background:#1D9E75"></span>Card</span>
        <span class="legend-item"><span class="dot" style="background:#BA7517"></span>Tournament</span>
        <span class="legend-item"><span class="dot" style="background:#C75CDB"></span>Set</span>
        <span class="legend-item"><span class="line-sample" style="background:#7F77DD"></span>Contains</span>
        <span class="legend-item"><span class="line-sample" style="background:#E24B4A;height:0;border-top:2px dashed #E24B4A"></span>Counters</span>
        <span class="legend-item"><span class="line-sample" style="background:#BA7517"></span>Top 8</span>
        <span class="legend-item"><span class="line-sample" style="background:#3CB4A0;height:0;border-top:2px dotted #3CB4A0"></span>Co-occurrence</span>
        <span class="legend-item"><span class="line-sample" style="background:#C75CDB;height:0;border-top:1px solid #C75CDB"></span>Printed In</span>
      </div>
      <svg id="graph" viewBox="0 0 1360 500"></svg>
      <div id="info">Hover a node to see details. Drag nodes to rearrange. Edge colors: purple=contains, red dashed=counters, orange=top 8, teal dotted=co-occurrence.</div>
    </div>

    <!-- Emergence Panel -->
    <div class="panel">
      <div class="panel-header">Archetype Emergence Predictions</div>
      <div class="panel-body">
        <table>
          <thead>
            <tr><th>#</th><th>Archetype</th><th>Predicted Change</th></tr>
          </thead>
          <tbody>
            {emergence_rows}
          </tbody>
        </table>
      </div>
    </div>

    <!-- Heatmap Panel -->
    <div class="panel">
      <div class="panel-header">Top 8 Probability Heatmap</div>
      <div class="panel-body" id="heatmap"></div>
    </div>
  </div>

  <script>
    // ── Graph Data ──
    const nodes = {nodes_json};
    const links = {links_json};

    const colorMap = {{arch:'#7F77DD', card:'#1D9E75', tourn:'#BA7517', set:'#C75CDB'}};
    const rMap = {{arch:24, card:14, tourn:18, set:20}};
    const edgeColors = {{
      contains: '#7F77DD',
      counters: '#E24B4A',
      top8: '#BA7517',
      co_occurrence: '#3CB4A0',
      printed_in: '#C75CDB',
    }};

    const W = 1360, H = 500;
    const svg = d3.select('#graph');

    // Arrow markers
    svg.append('defs').html(`
      <marker id="arr-counters" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
        <path d="M1 1L9 5L1 9" fill="none" stroke="#E24B4A" stroke-width="1.5" stroke-linecap="round"/>
      </marker>
    `);

    const linkG = svg.append('g');
    const nodeG = svg.append('g');

    const sim = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(d=>d.id).distance(d => {{
        if(d.type==='counters') return 160;
        if(d.type==='top8') return 120;
        return 80;
      }}).strength(0.4))
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(W/2, H/2))
      .force('collision', d3.forceCollide(d => rMap[d.type] + 12))
      .force('x', d3.forceX(W/2).strength(0.03))
      .force('y', d3.forceY(H/2).strength(0.03));

    const linkSel = linkG.selectAll('line').data(links).enter().append('line')
      .attr('stroke', d => edgeColors[d.type] || '#555')
      .attr('stroke-width', d => d.type==='counters' ? 1.8 : d.type==='top8' ? 1.5 : 1)
      .attr('stroke-dasharray', d => d.type==='counters' ? '5,4' : d.type==='co_occurrence' ? '2,3' : null)
      .attr('stroke-opacity', 0.5)
      .attr('marker-end', d => d.type==='counters' ? 'url(#arr-counters)' : null);

    const nodeGroup = nodeG.selectAll('g').data(nodes).enter().append('g')
      .attr('cursor','pointer')
      .call(d3.drag()
        .on('start', (e,d) => {{ if(!e.active) sim.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; }})
        .on('drag',  (e,d) => {{ d.fx=e.x; d.fy=e.y; }})
        .on('end',   (e,d) => {{ if(!e.active) sim.alphaTarget(0); d.fx=null; d.fy=null; }})
      )
      .on('mouseover', (e,d) => {{
        d3.select(e.currentTarget).select('circle').attr('r', rMap[d.type]+4);
        linkSel.attr('stroke-opacity', l => (l.source.id===d.id || l.target.id===d.id) ? 0.9 : 0.1);
        const typeLabel = d.type==='arch'?'Archetype':d.type==='card'?'Card':d.type==='set'?'Set':'Tournament';
        let html = `<strong style="color:#fff">${{d.label}}</strong> <span style="color:${{colorMap[d.type]}};font-size:11px;text-transform:uppercase">${{typeLabel}}</span><br>${{d.detail}}`;
        if (d.emergence !== undefined) {{
          const cls = d.emergence > 0 ? 'badge-green' : 'badge-red';
          html += ` <span class="badge ${{cls}}">Emergence: ${{d.emergence > 0 ? '+' : ''}}${{d.emergence.toFixed(4)}}</span>`;
        }}
        document.getElementById('info').innerHTML = html;
      }})
      .on('mouseout', (e,d) => {{
        d3.select(e.currentTarget).select('circle').attr('r', rMap[d.type]);
        linkSel.attr('stroke-opacity', 0.5);
        document.getElementById('info').innerHTML = 'Hover a node to see details. Drag nodes to rearrange.';
      }});

    nodeGroup.append('circle')
      .attr('r', d => rMap[d.type])
      .attr('fill', d => colorMap[d.type])
      .attr('fill-opacity', 0.9)
      .attr('stroke', d => colorMap[d.type])
      .attr('stroke-width', 1.5);

    nodeGroup.append('text')
      .attr('text-anchor','middle')
      .attr('dominant-baseline','central')
      .attr('font-size', d => d.type==='card' ? 8 : 9)
      .attr('font-weight','600')
      .attr('fill','white')
      .attr('pointer-events','none')
      .text(d => {{
        if(d.type==='arch') return d.label.split(' ')[0].substring(0,4).toUpperCase();
        if(d.type==='tourn') return '\\u2605';
        if(d.type==='set') return '\\u2B22';
        return '\\u25C6';
      }});

    nodeGroup.append('text')
      .attr('text-anchor','middle')
      .attr('y', d => rMap[d.type] + 12)
      .attr('font-size', 10)
      .attr('fill','#888')
      .attr('pointer-events','none')
      .text(d => d.label.length > 18 ? d.label.substring(0,18) + '...' : d.label);

    function clamp(v,lo,hi){{ return Math.max(lo,Math.min(hi,v)); }}

    sim.on('tick', () => {{
      nodes.forEach(d => {{
        d.x = clamp(d.x, 30, W-30);
        d.y = clamp(d.y, 30, H-30);
      }});
      linkSel
        .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => {{
          const dx=d.target.x-d.source.x, dy=d.target.y-d.source.y;
          const dist=Math.sqrt(dx*dx+dy*dy)||1;
          const off=d.type==='counters' ? rMap[d.target.type||'arch']+8 : 0;
          return d.target.x-(dx/dist)*off;
        }})
        .attr('y2', d => {{
          const dx=d.target.x-d.source.x, dy=d.target.y-d.source.y;
          const dist=Math.sqrt(dx*dx+dy*dy)||1;
          const off=d.type==='counters' ? rMap[d.target.type||'arch']+8 : 0;
          return d.target.y-(dy/dist)*off;
        }});
      nodeGroup.attr('transform', d => `translate(${{d.x}},${{d.y}})`);
    }});

    // ── Heatmap ──
    const heatData = {heatmap_json};
    const tournLabels = {tourn_labels_json};
    const archNames = {arch_names_json};

    const margin = {{top: 80, right: 10, bottom: 10, left: 140}};
    const cellW = Math.min(50, (500 - margin.left - margin.right) / tournLabels.length);
    const cellH = 22;
    const hmW = margin.left + cellW * tournLabels.length + margin.right;
    const hmH = margin.top + cellH * archNames.length + margin.bottom;

    const hmSvg = d3.select('#heatmap').append('svg')
      .attr('width', hmW).attr('height', hmH);

    const g = hmSvg.append('g').attr('transform', `translate(${{margin.left}},${{margin.top}})`);

    const colorScale = d3.scaleSequential(d3.interpolateYlOrRd).domain([0, 1]);

    // Column headers
    g.selectAll('.col-label')
      .data(tournLabels).enter().append('text')
      .attr('x', (d,i) => i * cellW + cellW/2)
      .attr('y', -8)
      .attr('text-anchor', 'end')
      .attr('transform', (d,i) => `rotate(-45,${{i*cellW+cellW/2}},-8)`)
      .attr('font-size', 9).attr('fill', '#888')
      .text(d => d.length > 15 ? d.substring(0,15)+'...' : d);

    // Row labels
    g.selectAll('.row-label')
      .data(archNames).enter().append('text')
      .attr('x', -6)
      .attr('y', (d,i) => i * cellH + cellH/2 + 4)
      .attr('text-anchor', 'end')
      .attr('font-size', 11).attr('fill', '#aaa')
      .text(d => d);

    // Cells
    g.selectAll('.cell')
      .data(heatData).enter().append('rect')
      .attr('x', d => tournLabels.indexOf(d.tourn) * cellW)
      .attr('y', d => archNames.indexOf(d.arch) * cellH)
      .attr('width', cellW - 1).attr('height', cellH - 1)
      .attr('rx', 3)
      .attr('fill', d => colorScale(d.prob))
      .attr('opacity', 0.85)
      .append('title').text(d => `${{d.arch}} @ ${{d.tourn}}: ${{(d.prob*100).toFixed(1)}}%`);

    // Cell text
    g.selectAll('.cell-text')
      .data(heatData).enter().append('text')
      .attr('x', d => tournLabels.indexOf(d.tourn) * cellW + cellW/2)
      .attr('y', d => archNames.indexOf(d.arch) * cellH + cellH/2 + 4)
      .attr('text-anchor', 'middle')
      .attr('font-size', 9)
      .attr('fill', d => d.prob > 0.6 ? '#fff' : '#666')
      .text(d => cellW >= 35 ? (d.prob*100).toFixed(0)+'%' : '');
  </script>
</body>
</html>"""

    return html


def main(run_dir=None, model_path=None):
    """Generate dashboard and save to the run directory.

    Parameters
    ----------
    run_dir : Path, optional
        Results directory for this run. If None, uses 'results/latest'.
    model_path : Path, optional
        Path to model checkpoint. If None, uses run_dir/model.pt.
    """
    if run_dir is None:
        run_dir = RESULTS_DIR / "latest"
    if model_path is None:
        model_path = run_dir / "model.pt"

    log.info("Loading graph and model...")
    data = torch.load(GRAPH_PATH, weights_only=False)

    log.info("Getting model predictions...")
    predictions = _get_model_predictions(data, model_path=model_path)

    log.info("Building archetype card pools...")
    try:
        decklists = pd.read_parquet(DECKLISTS_PARQUET)
    except Exception:
        decklists = pd.DataFrame()
    top_cards = _get_top_cards_per_archetype(data, decklists)

    log.info("Building graph visualization data...")
    graph_data = _build_graph_data(data, predictions, top_cards)
    log.info(f"  {len(graph_data['nodes'])} nodes, {len(graph_data['links'])} links in viz")

    log.info("Generating HTML dashboard...")
    html = generate_html(graph_data, predictions)

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dashboard_path = run_dir / "dashboard.html"
    dashboard_path.write_text(html, encoding="utf-8")
    log.info(f"Dashboard saved to {dashboard_path}")
    log.info(f"Open in browser: file:///{dashboard_path.as_posix()}")


if __name__ == "__main__":
    main()
