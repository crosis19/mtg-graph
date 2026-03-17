const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageNumber, PageBreak, LevelFormat,
} = require("docx");

// ── Color palette ──
const COLORS = {
  card: "4472C4",       // blue
  archetype: "ED7D31",  // orange
  tournament: "70AD47",  // green
  set: "9B59B6",         // purple
  edge: "BDD7EE",        // light blue
  header: "2E75B6",      // dark blue
  headerText: "FFFFFF",
  lightGray: "F2F2F2",
  medGray: "D9D9D9",
  white: "FFFFFF",
  black: "000000",
  loss1: "FF6B6B",       // red for MSE
  loss2: "4ECDC4",       // teal for BCE
  gradient1: "E8F4FD",
  gradient2: "D1ECF1",
  gradient3: "BEE5EB",
};

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const noBorder = { style: BorderStyle.NONE, size: 0 };
const noBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder };
const thickBorder = { style: BorderStyle.SINGLE, size: 3, color: "2E75B6" };

const CONTENT_WIDTH = 9360; // US Letter with 1" margins

function makeCell(text, opts = {}) {
  const {
    width, bold, color, fill, alignment, font, size, colspan, rowspan,
    cellBorders, italic, verticalAlign,
  } = opts;
  const cellOpts = {
    borders: cellBorders || borders,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [
      new Paragraph({
        alignment: alignment || AlignmentType.LEFT,
        children: [
          new TextRun({
            text,
            bold: bold || false,
            italic: italic || false,
            color: color || COLORS.black,
            font: font || "Arial",
            size: size || 20,
          }),
        ],
      }),
    ],
  };
  if (width) cellOpts.width = { size: width, type: WidthType.DXA };
  if (fill) cellOpts.shading = { fill, type: ShadingType.CLEAR };
  if (colspan) cellOpts.columnSpan = colspan;
  if (rowspan) cellOpts.rowSpan = rowspan;
  if (verticalAlign) cellOpts.verticalAlign = verticalAlign;
  return new TableCell(cellOpts);
}

function makeMultiLineCell(lines, opts = {}) {
  const { width, fill, cellBorders, alignment } = opts;
  const cellOpts = {
    borders: cellBorders || borders,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: lines.map(
      (line) =>
        new Paragraph({
          alignment: alignment || AlignmentType.LEFT,
          spacing: { after: 40 },
          children: Array.isArray(line) ? line : [line],
        })
    ),
  };
  if (width) cellOpts.width = { size: width, type: WidthType.DXA };
  if (fill) cellOpts.shading = { fill, type: ShadingType.CLEAR };
  return new TableCell(cellOpts);
}

function sectionHeading(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text, bold: true, font: "Arial", size: 32, color: COLORS.header })],
    spacing: { before: 360, after: 200 },
  });
}

function subHeading(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text, bold: true, font: "Arial", size: 26, color: COLORS.header })],
    spacing: { before: 280, after: 160 },
  });
}

function bodyText(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120 },
    children: [
      new TextRun({
        text,
        font: "Arial",
        size: 21,
        bold: opts.bold || false,
        italic: opts.italic || false,
        color: opts.color || COLORS.black,
      }),
    ],
  });
}

function bodyTextMulti(runs) {
  return new Paragraph({
    spacing: { after: 120 },
    children: runs.map(
      (r) =>
        new TextRun({
          text: r.text,
          font: "Arial",
          size: 21,
          bold: r.bold || false,
          italic: r.italic || false,
          color: r.color || COLORS.black,
        })
    ),
  });
}

function arrowRow(label, colWidths, spanCols) {
  const totalSpan = spanCols || colWidths.length;
  return new TableRow({
    children: [
      makeCell(label, {
        width: colWidths.reduce((a, b) => a + b, 0),
        colspan: totalSpan,
        alignment: AlignmentType.CENTER,
        bold: true,
        size: 22,
        color: COLORS.header,
        cellBorders: noBorders,
        fill: COLORS.gradient1,
      }),
    ],
  });
}

// ══════════════════════════════════════════════════════════════
// BUILD DOCUMENT
// ══════════════════════════════════════════════════════════════

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial", color: COLORS.header },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 },
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial", color: COLORS.header },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 1 },
      },
    ],
  },
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [
          {
            level: 0, format: LevelFormat.BULLET, text: "-",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } },
          },
        ],
      },
      {
        reference: "numbers",
        levels: [
          {
            level: 0, format: LevelFormat.DECIMAL, text: "%1.",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } },
          },
        ],
      },
    ],
  },
  sections: [
    // ════════════════════════════════════════════════
    // TITLE PAGE
    // ════════════════════════════════════════════════
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
        },
      },
      children: [
        new Paragraph({ spacing: { before: 3600 } }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 200 },
          children: [
            new TextRun({ text: "MTG Metagame GNN", font: "Arial", size: 52, bold: true, color: COLORS.header }),
          ],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 100 },
          children: [
            new TextRun({ text: "Signal Propagation & Training Architecture", font: "Arial", size: 32, color: "666666" }),
          ],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 600 },
          border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: COLORS.header, space: 1 } },
          children: [new TextRun({ text: " " })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 100 },
          children: [
            new TextRun({ text: "Heterogeneous Graph Transformer (HGT)", font: "Arial", size: 24, color: "444444" }),
          ],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 100 },
          children: [
            new TextRun({ text: "Two-Headed Architecture for Metagame Prediction", font: "Arial", size: 24, color: "444444" }),
          ],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 1200 },
          children: [
            new TextRun({ text: "March 2026", font: "Arial", size: 22, color: "888888" }),
          ],
        }),
      ],
    },

    // ════════════════════════════════════════════════
    // SECTION 1: GRAPH STRUCTURE OVERVIEW
    // ════════════════════════════════════════════════
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
        },
      },
      headers: {
        default: new Header({
          children: [
            new Paragraph({
              border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: COLORS.header, space: 4 } },
              children: [
                new TextRun({ text: "MTG Metagame GNN  |  Architecture Guide", font: "Arial", size: 18, color: "888888", italic: true }),
              ],
            }),
          ],
        }),
      },
      footers: {
        default: new Footer({
          children: [
            new Paragraph({
              alignment: AlignmentType.CENTER,
              children: [
                new TextRun({ text: "Page ", font: "Arial", size: 18, color: "888888" }),
                new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 18, color: "888888" }),
              ],
            }),
          ],
        }),
      },
      children: [
        // ── 1. Graph Structure ──
        sectionHeading("1. Heterogeneous Graph Structure"),
        bodyText("The model operates on a heterogeneous graph with four node types and eleven edge types. Each node type carries distinct feature vectors, and each edge type represents a different relationship between entities in the MTG Standard metagame."),

        subHeading("1.1 Node Types"),
        // Node types diagram
        new Table({
          width: { size: CONTENT_WIDTH, type: WidthType.DXA },
          columnWidths: [2340, 2340, 2340, 2340],
          rows: [
            new TableRow({
              children: [
                makeCell("CARD", { width: 2340, bold: true, fill: COLORS.card, color: COLORS.white, alignment: AlignmentType.CENTER, size: 24 }),
                makeCell("ARCHETYPE", { width: 2340, bold: true, fill: COLORS.archetype, color: COLORS.white, alignment: AlignmentType.CENTER, size: 24 }),
                makeCell("TOURNAMENT", { width: 2340, bold: true, fill: COLORS.tournament, color: COLORS.white, alignment: AlignmentType.CENTER, size: 24 }),
                makeCell("SET", { width: 2340, bold: true, fill: COLORS.set, color: COLORS.white, alignment: AlignmentType.CENTER, size: 24 }),
              ],
            }),
            new TableRow({
              children: [
                makeMultiLineCell([
                  new TextRun({ text: "~4,168 nodes", font: "Arial", size: 18, bold: true }),
                  new TextRun({ text: "384-dim embedding", font: "Arial", size: 17 }),
                  new TextRun({ text: "+ 11 numeric features", font: "Arial", size: 17 }),
                  new TextRun({ text: "(cmc, colors, types,", font: "Arial", size: 17 }),
                  new TextRun({ text: "is_banned, etc.)", font: "Arial", size: 17 }),
                ], { width: 2340, fill: "D6E4F0", alignment: AlignmentType.CENTER }),
                makeMultiLineCell([
                  new TextRun({ text: "~12-32 nodes", font: "Arial", size: 18, bold: true }),
                  new TextRun({ text: "3 features:", font: "Arial", size: 17 }),
                  new TextRun({ text: "meta_share", font: "Arial", size: 17 }),
                  new TextRun({ text: "win_rate", font: "Arial", size: 17 }),
                  new TextRun({ text: "n_colors", font: "Arial", size: 17 }),
                ], { width: 2340, fill: "FCEBD0", alignment: AlignmentType.CENTER }),
                makeMultiLineCell([
                  new TextRun({ text: "~15-24 nodes", font: "Arial", size: 18, bold: true }),
                  new TextRun({ text: "2 features:", font: "Arial", size: 17 }),
                  new TextRun({ text: "player_count", font: "Arial", size: 17 }),
                  new TextRun({ text: "time_ordinal", font: "Arial", size: 17 }),
                  new TextRun({ text: " ", font: "Arial", size: 17 }),
                ], { width: 2340, fill: "D5EADA", alignment: AlignmentType.CENTER }),
                makeMultiLineCell([
                  new TextRun({ text: "~12-15 nodes", font: "Arial", size: 18, bold: true }),
                  new TextRun({ text: "3 features:", font: "Arial", size: 17 }),
                  new TextRun({ text: "recency", font: "Arial", size: 17 }),
                  new TextRun({ text: "size_norm", font: "Arial", size: 17 }),
                  new TextRun({ text: "avg_cmc_norm", font: "Arial", size: 17 }),
                ], { width: 2340, fill: "E8D5F0", alignment: AlignmentType.CENTER }),
              ],
            }),
            new TableRow({
              children: [
                makeCell("Input: 395-dim", { width: 2340, alignment: AlignmentType.CENTER, size: 18, italic: true, fill: COLORS.lightGray }),
                makeCell("Input: 3-dim", { width: 2340, alignment: AlignmentType.CENTER, size: 18, italic: true, fill: COLORS.lightGray }),
                makeCell("Input: 2-dim", { width: 2340, alignment: AlignmentType.CENTER, size: 18, italic: true, fill: COLORS.lightGray }),
                makeCell("Input: 3-dim", { width: 2340, alignment: AlignmentType.CENTER, size: 18, italic: true, fill: COLORS.lightGray }),
              ],
            }),
          ],
        }),

        new Paragraph({ spacing: { after: 200 } }),
        subHeading("1.2 Edge Types (Relationships)"),
        bodyText("Edges encode different types of relationships. Synergy edges connect cards that work well together. Structural edges connect cards to their archetypes, sets, and tournaments."),

        // Edge types table
        new Table({
          width: { size: CONTENT_WIDTH, type: WidthType.DXA },
          columnWidths: [3800, 3200, 2360],
          rows: [
            new TableRow({
              children: [
                makeCell("Edge Type", { width: 3800, bold: true, fill: COLORS.header, color: COLORS.white, size: 20 }),
                makeCell("Connection", { width: 3200, bold: true, fill: COLORS.header, color: COLORS.white, size: 20 }),
                makeCell("Direction", { width: 2360, bold: true, fill: COLORS.header, color: COLORS.white, size: 20 }),
              ],
            }),
            ...([
              ["keyword_synergy", "Card \u2194 Card", "Undirected"],
              ["mechanical_synergy", "Card \u2194 Card", "Undirected"],
              ["semantic_synergy", "Card \u2194 Card", "Undirected"],
              ["co_occurrence", "Card \u2194 Card", "Undirected"],
              ["contains / in_deck", "Archetype \u2194 Card", "Bidirectional"],
              ["counters / countered_by", "Archetype \u2194 Archetype", "Bidirectional"],
              ["top8 / hosted", "Archetype \u2194 Tournament", "Bidirectional"],
              ["printed_in / from_set", "Set \u2194 Card", "Bidirectional"],
            ].map((row, i) =>
              new TableRow({
                children: [
                  makeCell(row[0], { width: 3800, size: 19, fill: i % 2 === 0 ? COLORS.white : COLORS.lightGray, bold: true }),
                  makeCell(row[1], { width: 3200, size: 19, fill: i % 2 === 0 ? COLORS.white : COLORS.lightGray }),
                  makeCell(row[2], { width: 2360, size: 19, fill: i % 2 === 0 ? COLORS.white : COLORS.lightGray }),
                ],
              })
            )),
          ],
        }),

        // ════════════════════════════════════════════════
        // SECTION 2: SIGNAL PROPAGATION
        // ════════════════════════════════════════════════
        new Paragraph({ children: [new PageBreak()] }),
        sectionHeading("2. Signal Propagation Through the Model"),
        bodyText("The model processes the heterogeneous graph in three stages: input projection, HGT message passing, and task-specific prediction heads. Here is how information flows end-to-end."),

        subHeading("2.1 Stage 1: Input Projection"),
        bodyText("Each node type has a different feature dimensionality. Before message passing, per-type linear projections map all nodes into a shared 128-dimensional hidden space."),

        // Input projection diagram
        new Table({
          width: { size: CONTENT_WIDTH, type: WidthType.DXA },
          columnWidths: [2200, 560, 2200, 560, 2200, 560, 1080],
          rows: [
            // Row 1: Input dimensions
            new TableRow({
              children: [
                makeCell("Card Features", { width: 2200, bold: true, fill: COLORS.card, color: COLORS.white, alignment: AlignmentType.CENTER, size: 18 }),
                makeCell("", { width: 560, cellBorders: noBorders }),
                makeCell("Archetype Features", { width: 2200, bold: true, fill: COLORS.archetype, color: COLORS.white, alignment: AlignmentType.CENTER, size: 18 }),
                makeCell("", { width: 560, cellBorders: noBorders }),
                makeCell("Tournament Features", { width: 2200, bold: true, fill: COLORS.tournament, color: COLORS.white, alignment: AlignmentType.CENTER, size: 18 }),
                makeCell("", { width: 560, cellBorders: noBorders }),
                makeCell("Set", { width: 1080, bold: true, fill: COLORS.set, color: COLORS.white, alignment: AlignmentType.CENTER, size: 18 }),
              ],
            }),
            new TableRow({
              children: [
                makeCell("395 dimensions", { width: 2200, alignment: AlignmentType.CENTER, size: 18, fill: "D6E4F0" }),
                makeCell("", { width: 560, cellBorders: noBorders }),
                makeCell("3 dimensions", { width: 2200, alignment: AlignmentType.CENTER, size: 18, fill: "FCEBD0" }),
                makeCell("", { width: 560, cellBorders: noBorders }),
                makeCell("2 dimensions", { width: 2200, alignment: AlignmentType.CENTER, size: 18, fill: "D5EADA" }),
                makeCell("", { width: 560, cellBorders: noBorders }),
                makeCell("3 dim", { width: 1080, alignment: AlignmentType.CENTER, size: 18, fill: "E8D5F0" }),
              ],
            }),
            // Arrow row
            arrowRow("\u2193  Linear(in_dim, 128)  \u2193", [2200, 560, 2200, 560, 2200, 560, 1080], 7),
            // Output row
            new TableRow({
              children: [
                makeCell("128-dim", { width: 2200, alignment: AlignmentType.CENTER, size: 18, bold: true, fill: COLORS.gradient2 }),
                makeCell("", { width: 560, cellBorders: noBorders }),
                makeCell("128-dim", { width: 2200, alignment: AlignmentType.CENTER, size: 18, bold: true, fill: COLORS.gradient2 }),
                makeCell("", { width: 560, cellBorders: noBorders }),
                makeCell("128-dim", { width: 2200, alignment: AlignmentType.CENTER, size: 18, bold: true, fill: COLORS.gradient2 }),
                makeCell("", { width: 560, cellBorders: noBorders }),
                makeCell("128-dim", { width: 1080, alignment: AlignmentType.CENTER, size: 18, bold: true, fill: COLORS.gradient2 }),
              ],
            }),
          ],
        }),

        new Paragraph({ spacing: { after: 60 } }),
        bodyTextMulti([
          { text: "Result: ", bold: true },
          { text: "All node types now share the same 128-dimensional representation space, enabling cross-type message passing." },
        ]),

        new Paragraph({ spacing: { after: 200 } }),
        subHeading("2.2 Stage 2: HGT Message Passing (3 Layers)"),
        bodyText("The core of the model is three layers of Heterogeneous Graph Transformer (HGT) convolution. Each layer allows nodes to attend to their neighbors across all edge types, aggregating information with type-specific attention weights."),

        // Single HGT Layer detail
        bodyText("Each HGT layer performs the following operations:", { bold: true }),

        new Table({
          width: { size: CONTENT_WIDTH, type: WidthType.DXA },
          columnWidths: [1200, 8160],
          rows: [
            new TableRow({
              children: [
                makeCell("Step", { width: 1200, bold: true, fill: COLORS.header, color: COLORS.white, alignment: AlignmentType.CENTER }),
                makeCell("Operation", { width: 8160, bold: true, fill: COLORS.header, color: COLORS.white }),
              ],
            }),
            ...([
              ["1", "Heterogeneous Attention: For each edge type, compute attention scores between source and target nodes using type-specific weight matrices. Each of the 4 attention heads learns different relationship patterns."],
              ["2", "Message Aggregation: Source nodes send weighted messages to target nodes along each edge type. Messages from different edge types are combined."],
              ["3", "Residual Connection: Add the input node embeddings back to the aggregated messages (x + dropout(HGT(x)))."],
              ["4", "Layer Normalization: Apply per-type LayerNorm to stabilize training and normalize each node type independently."],
            ].map((row, i) =>
              new TableRow({
                children: [
                  makeCell(row[0], { width: 1200, bold: true, alignment: AlignmentType.CENTER, size: 20, fill: COLORS.gradient1 }),
                  makeCell(row[1], { width: 8160, size: 19, fill: i % 2 === 0 ? COLORS.white : COLORS.lightGray }),
                ],
              })
            )),
          ],
        }),

        new Paragraph({ spacing: { after: 200 } }),

        // 3-layer stacking diagram
        bodyText("Stacked Layer Architecture:", { bold: true }),
        new Table({
          width: { size: 7000, type: WidthType.DXA },
          columnWidths: [7000],
          rows: [
            new TableRow({ children: [makeCell("All Node Embeddings (128-dim each)", { width: 7000, alignment: AlignmentType.CENTER, bold: true, fill: COLORS.gradient2, size: 20 })] }),
            new TableRow({ children: [makeCell("\u2193", { width: 7000, alignment: AlignmentType.CENTER, cellBorders: noBorders, size: 22, bold: true, color: COLORS.header })] }),
            new TableRow({ children: [makeCell("HGT Layer 1:  HGTConv(128, 128, heads=4)  +  Residual  +  LayerNorm  +  Dropout(0.2)", { width: 7000, alignment: AlignmentType.CENTER, fill: "D6E4F0", size: 19, bold: true })] }),
            new TableRow({ children: [makeCell("\u2193  Enriched embeddings carry 1-hop neighborhood info", { width: 7000, alignment: AlignmentType.CENTER, cellBorders: noBorders, size: 18, italic: true, color: "666666" })] }),
            new TableRow({ children: [makeCell("HGT Layer 2:  HGTConv(128, 128, heads=4)  +  Residual  +  LayerNorm  +  Dropout(0.2)", { width: 7000, alignment: AlignmentType.CENTER, fill: "BDD7EE", size: 19, bold: true })] }),
            new TableRow({ children: [makeCell("\u2193  Embeddings now encode 2-hop neighborhood patterns", { width: 7000, alignment: AlignmentType.CENTER, cellBorders: noBorders, size: 18, italic: true, color: "666666" })] }),
            new TableRow({ children: [makeCell("HGT Layer 3:  HGTConv(128, 128, heads=4)  +  Residual  +  LayerNorm  +  Dropout(0.2)", { width: 7000, alignment: AlignmentType.CENTER, fill: "9DC3E6", size: 19, bold: true })] }),
            new TableRow({ children: [makeCell("\u2193", { width: 7000, alignment: AlignmentType.CENTER, cellBorders: noBorders, size: 22, bold: true, color: COLORS.header })] }),
            new TableRow({ children: [makeCell("Final Node Embeddings  {card: [...], archetype: [...], tournament: [...], set: [...]}", { width: 7000, alignment: AlignmentType.CENTER, bold: true, fill: COLORS.gradient3, size: 19 })] }),
          ],
        }),

        new Paragraph({ spacing: { after: 120 } }),
        subHeading("2.3 What Each Layer Learns"),
        bodyTextMulti([
          { text: "Layer 1: ", bold: true },
          { text: "Direct relationships. A card learns about its synergistic partners (keyword, mechanical, semantic edges). An archetype learns about the cards it contains." },
        ]),
        bodyTextMulti([
          { text: "Layer 2: ", bold: true },
          { text: "Extended context. A card now knows about the synergy partners of its synergy partners. An archetype absorbs information about synergy clusters within its decklist." },
        ]),
        bodyTextMulti([
          { text: "Layer 3: ", bold: true },
          { text: "Global structure. Information has propagated across archetype-tournament-card boundaries. A tournament node now encodes which synergy patterns its top-8 archetypes share." },
        ]),

        // ════════════════════════════════════════════════
        // MESSAGE PROPAGATION EXAMPLE
        // ════════════════════════════════════════════════
        new Paragraph({ children: [new PageBreak()] }),
        subHeading("2.4 Concrete Example: Message Flow"),
        bodyText("Consider a single card node (e.g., \"Sheoldred, the Apocalypse\") and how it gathers information across 3 layers of HGT:"),

        new Table({
          width: { size: CONTENT_WIDTH, type: WidthType.DXA },
          columnWidths: [1400, 7960],
          rows: [
            new TableRow({ children: [
              makeCell("Layer", { width: 1400, bold: true, fill: COLORS.header, color: COLORS.white, alignment: AlignmentType.CENTER }),
              makeCell("Information Gathered by Sheoldred", { width: 7960, bold: true, fill: COLORS.header, color: COLORS.white }),
            ]}),
            new TableRow({ children: [
              makeCell("Input", { width: 1400, bold: true, alignment: AlignmentType.CENTER, fill: COLORS.lightGray }),
              makeMultiLineCell([
                new TextRun({ text: "384-dim text embedding of oracle text + [cmc=4, colors=1, is_creature=1, ...]", font: "Arial", size: 19 }),
                new TextRun({ text: "Projected to 128-dim via Linear(395, 128)", font: "Arial", size: 19, italic: true, color: "666666" }),
              ], { width: 7960 }),
            ]}),
            new TableRow({ children: [
              makeCell("Layer 1", { width: 1400, bold: true, alignment: AlignmentType.CENTER, fill: "D6E4F0" }),
              makeMultiLineCell([
                new TextRun({ text: "keyword_synergy: Receives from cards with lifelink, deathtouch synergies", font: "Arial", size: 19 }),
                new TextRun({ text: "semantic_synergy: Receives from cards with similar oracle text themes", font: "Arial", size: 19 }),
                new TextRun({ text: "co_occurrence: Receives from cards frequently played alongside it", font: "Arial", size: 19 }),
                new TextRun({ text: "in_deck: Receives from archetypes that include Sheoldred", font: "Arial", size: 19 }),
                new TextRun({ text: "from_set: Receives from the Dominaria United set node", font: "Arial", size: 19 }),
              ], { width: 7960 }),
            ]}),
            new TableRow({ children: [
              makeCell("Layer 2", { width: 1400, bold: true, alignment: AlignmentType.CENTER, fill: "BDD7EE" }),
              makeMultiLineCell([
                new TextRun({ text: "Now knows about synergy partners OF its synergy partners", font: "Arial", size: 19 }),
                new TextRun({ text: "Absorbs archetype-level info: meta share, win rates of decks it appears in", font: "Arial", size: 19 }),
                new TextRun({ text: "Learns which tournaments its archetypes have placed in (via archetype\u2192top8\u2192tournament)", font: "Arial", size: 19 }),
              ], { width: 7960 }),
            ]}),
            new TableRow({ children: [
              makeCell("Layer 3", { width: 1400, bold: true, alignment: AlignmentType.CENTER, fill: "9DC3E6" }),
              makeMultiLineCell([
                new TextRun({ text: "Global metagame awareness: knows about counter-archetypes and competing strategies", font: "Arial", size: 19 }),
                new TextRun({ text: "Has absorbed tournament field composition through multi-hop paths", font: "Arial", size: 19 }),
                new TextRun({ text: "Embedding now encodes: card identity + synergy context + metagame position + competitive history", font: "Arial", size: 19 }),
              ], { width: 7960 }),
            ]}),
          ],
        }),

        // ════════════════════════════════════════════════
        // ATTENTION MECHANISM
        // ════════════════════════════════════════════════
        new Paragraph({ spacing: { after: 200 } }),
        subHeading("2.5 Heterogeneous Attention Mechanism"),
        bodyText("Unlike standard graph attention (GAT), HGT computes attention weights that are specific to each edge type. This means the model learns different importance patterns for different relationships:"),

        new Table({
          width: { size: CONTENT_WIDTH, type: WidthType.DXA },
          columnWidths: [4680, 4680],
          rows: [
            new TableRow({ children: [
              makeCell("Standard GAT", { width: 4680, bold: true, fill: COLORS.medGray, alignment: AlignmentType.CENTER, size: 20 }),
              makeCell("HGT (This Model)", { width: 4680, bold: true, fill: COLORS.gradient2, alignment: AlignmentType.CENTER, size: 20 }),
            ]}),
            new TableRow({ children: [
              makeMultiLineCell([
                new TextRun({ text: "Single attention function", font: "Arial", size: 19 }),
                new TextRun({ text: "All edges treated the same", font: "Arial", size: 19 }),
                new TextRun({ text: "Cannot distinguish edge types", font: "Arial", size: 19 }),
              ], { width: 4680 }),
              makeMultiLineCell([
                new TextRun({ text: "Per-edge-type attention weights", font: "Arial", size: 19 }),
                new TextRun({ text: "keyword_synergy uses different", font: "Arial", size: 19 }),
                new TextRun({ text: "  weights than co_occurrence", font: "Arial", size: 19 }),
                new TextRun({ text: "4 heads per edge type = 4 views", font: "Arial", size: 19 }),
              ], { width: 4680 }),
            ]}),
          ],
        }),

        new Paragraph({ spacing: { after: 60 } }),
        bodyTextMulti([
          { text: "Key insight: ", bold: true },
          { text: "The 4 attention heads allow the model to simultaneously weigh different aspects of each relationship. For example, one head might focus on mana curve compatibility while another attends to color identity alignment." },
        ]),

        // ════════════════════════════════════════════════
        // SECTION 3: PREDICTION HEADS
        // ════════════════════════════════════════════════
        new Paragraph({ children: [new PageBreak()] }),
        sectionHeading("3. Prediction Heads"),
        bodyText("After HGT message passing produces enriched node embeddings, two task-specific heads decode the embeddings into predictions."),

        subHeading("3.1 Head 1: Archetype Emergence (Regression)"),
        bodyText("Predicts the change in metagame share for each archetype. Positive values indicate a rising archetype; negative values indicate decline."),

        new Table({
          width: { size: 7000, type: WidthType.DXA },
          columnWidths: [7000],
          rows: [
            new TableRow({ children: [makeCell("Archetype Embeddings (n_archetypes x 128)", { width: 7000, alignment: AlignmentType.CENTER, bold: true, fill: COLORS.archetype, color: COLORS.white, size: 19 })] }),
            new TableRow({ children: [makeCell("\u2193", { width: 7000, alignment: AlignmentType.CENTER, cellBorders: noBorders, size: 22, bold: true, color: COLORS.header })] }),
            new TableRow({ children: [makeCell("Linear(128, 64)  +  ReLU  +  Dropout(0.2)", { width: 7000, alignment: AlignmentType.CENTER, fill: COLORS.lightGray, size: 19 })] }),
            new TableRow({ children: [makeCell("\u2193", { width: 7000, alignment: AlignmentType.CENTER, cellBorders: noBorders, size: 22, bold: true, color: COLORS.header })] }),
            new TableRow({ children: [makeCell("Linear(64, 1)", { width: 7000, alignment: AlignmentType.CENTER, fill: COLORS.lightGray, size: 19 })] }),
            new TableRow({ children: [makeCell("\u2193", { width: 7000, alignment: AlignmentType.CENTER, cellBorders: noBorders, size: 22, bold: true, color: COLORS.header })] }),
            new TableRow({ children: [makeCell("Emergence Score  (scalar per archetype)", { width: 7000, alignment: AlignmentType.CENTER, bold: true, fill: COLORS.loss1, color: COLORS.white, size: 19 })] }),
          ],
        }),

        new Paragraph({ spacing: { after: 60 } }),
        bodyTextMulti([
          { text: "Target: ", bold: true },
          { text: "(latest_meta_share - mean_meta_share) / 100. Trained with MSE loss." },
        ]),

        new Paragraph({ spacing: { after: 200 } }),
        subHeading("3.2 Head 2: Tournament Top 8 (Link Prediction)"),
        bodyText("Predicts the probability that a given archetype will place in a tournament's top 8. Takes pairs of (archetype, tournament) embeddings as input."),

        new Table({
          width: { size: CONTENT_WIDTH, type: WidthType.DXA },
          columnWidths: [4680, 4680],
          rows: [
            new TableRow({ children: [
              makeCell("Archetype Embedding (128-dim)", { width: 4680, alignment: AlignmentType.CENTER, bold: true, fill: COLORS.archetype, color: COLORS.white, size: 19 }),
              makeCell("Tournament Embedding (128-dim)", { width: 4680, alignment: AlignmentType.CENTER, bold: true, fill: COLORS.tournament, color: COLORS.white, size: 19 }),
            ]}),
          ],
        }),
        new Table({
          width: { size: CONTENT_WIDTH, type: WidthType.DXA },
          columnWidths: [CONTENT_WIDTH],
          rows: [
            new TableRow({ children: [makeCell("\u2193  Concatenate  \u2193", { width: CONTENT_WIDTH, alignment: AlignmentType.CENTER, cellBorders: noBorders, bold: true, color: COLORS.header, size: 20 })] }),
            new TableRow({ children: [makeCell("Combined Vector (256-dim)", { width: CONTENT_WIDTH, alignment: AlignmentType.CENTER, fill: COLORS.gradient2, bold: true, size: 19 })] }),
            new TableRow({ children: [makeCell("\u2193", { width: CONTENT_WIDTH, alignment: AlignmentType.CENTER, cellBorders: noBorders, bold: true, color: COLORS.header, size: 22 })] }),
            new TableRow({ children: [makeCell("Linear(256, 128)  +  ReLU  +  Dropout(0.2)", { width: CONTENT_WIDTH, alignment: AlignmentType.CENTER, fill: COLORS.lightGray, size: 19 })] }),
            new TableRow({ children: [makeCell("\u2193", { width: CONTENT_WIDTH, alignment: AlignmentType.CENTER, cellBorders: noBorders, bold: true, color: COLORS.header, size: 22 })] }),
            new TableRow({ children: [makeCell("Linear(128, 1)  +  Sigmoid", { width: CONTENT_WIDTH, alignment: AlignmentType.CENTER, fill: COLORS.lightGray, size: 19 })] }),
            new TableRow({ children: [makeCell("\u2193", { width: CONTENT_WIDTH, alignment: AlignmentType.CENTER, cellBorders: noBorders, bold: true, color: COLORS.header, size: 22 })] }),
            new TableRow({ children: [makeCell("Top 8 Probability  (0.0 to 1.0)", { width: CONTENT_WIDTH, alignment: AlignmentType.CENTER, bold: true, fill: COLORS.loss2, color: COLORS.white, size: 19 })] }),
          ],
        }),

        new Paragraph({ spacing: { after: 60 } }),
        bodyTextMulti([
          { text: "Target: ", bold: true },
          { text: "Binary label (1 = archetype placed in tournament top 8, 0 = did not). Trained with BCE loss. Negative samples are generated at a 2:1 ratio." },
        ]),

        // ════════════════════════════════════════════════
        // SECTION 4: TRAINING PROCESS
        // ════════════════════════════════════════════════
        new Paragraph({ children: [new PageBreak()] }),
        sectionHeading("4. Training Process"),
        bodyText("The model trains both heads jointly with a combined loss function, using a temporal train/validation split to simulate real-world prediction scenarios."),

        subHeading("4.1 Temporal Split Strategy"),
        bodyText("Unlike random splits, this model splits data by time. Earlier tournaments form the training set; later tournaments form the validation set. This tests the model's ability to predict future metagame trends from past data."),

        new Table({
          width: { size: CONTENT_WIDTH, type: WidthType.DXA },
          columnWidths: [7020, 2340],
          rows: [
            new TableRow({ children: [
              makeCell("TRAINING SET (75%)", { width: 7020, bold: true, fill: "70AD47", color: COLORS.white, alignment: AlignmentType.CENTER, size: 22 }),
              makeCell("VAL SET (25%)", { width: 2340, bold: true, fill: COLORS.loss1, color: COLORS.white, alignment: AlignmentType.CENTER, size: 22 }),
            ]}),
            new TableRow({ children: [
              makeCell("Tournament 1  \u2192  Tournament 2  \u2192  ...  \u2192  Tournament N*0.75", { width: 7020, alignment: AlignmentType.CENTER, size: 18, fill: "D5EADA" }),
              makeCell("Tournament N*0.75+1  \u2192  N", { width: 2340, alignment: AlignmentType.CENTER, size: 18, fill: "FADBD8" }),
            ]}),
            new TableRow({ children: [
              makeCell("Earlier events (known outcomes)", { width: 7020, alignment: AlignmentType.CENTER, size: 17, italic: true, fill: COLORS.lightGray }),
              makeCell("Future events (predict)", { width: 2340, alignment: AlignmentType.CENTER, size: 17, italic: true, fill: COLORS.lightGray }),
            ]}),
          ],
        }),

        new Paragraph({ spacing: { after: 200 } }),
        subHeading("4.2 Training Loop (Per Epoch)"),
        bodyText("Each of the 100 training epochs follows this sequence:"),

        // Training loop diagram
        new Table({
          width: { size: CONTENT_WIDTH, type: WidthType.DXA },
          columnWidths: [600, 8760],
          rows: [
            // Step 1
            new TableRow({ children: [
              makeCell("1", { width: 600, bold: true, alignment: AlignmentType.CENTER, fill: COLORS.header, color: COLORS.white, size: 22 }),
              makeMultiLineCell([
                new TextRun({ text: "Forward Pass", font: "Arial", size: 20, bold: true }),
                new TextRun({ text: "Run full graph through model: Input Projection \u2192 3x HGT Layers \u2192 Get node embeddings", font: "Arial", size: 19 }),
              ], { width: 8760, fill: COLORS.gradient1 }),
            ]}),
            // Step 2
            new TableRow({ children: [
              makeCell("2", { width: 600, bold: true, alignment: AlignmentType.CENTER, fill: COLORS.header, color: COLORS.white, size: 22 }),
              makeMultiLineCell([
                new TextRun({ text: "Head 1: Emergence Loss (MSE)", font: "Arial", size: 20, bold: true, color: COLORS.loss1 }),
                new TextRun({ text: "Compare predicted meta share changes against actual changes for ALL archetypes", font: "Arial", size: 19 }),
                new TextRun({ text: "Loss = mean((predicted_delta - actual_delta)^2)", font: "Arial", size: 19, italic: true }),
              ], { width: 8760, fill: COLORS.white }),
            ]}),
            // Step 3
            new TableRow({ children: [
              makeCell("3", { width: 600, bold: true, alignment: AlignmentType.CENTER, fill: COLORS.header, color: COLORS.white, size: 22 }),
              makeMultiLineCell([
                new TextRun({ text: "Head 2: Top 8 Loss (BCE) \u2014 Train Split Only", font: "Arial", size: 20, bold: true, color: "0E918C" }),
                new TextRun({ text: "For training-set (archetype, tournament) pairs:", font: "Arial", size: 19 }),
                new TextRun({ text: "  Concatenate archetype + tournament embeddings \u2192 predict probability", font: "Arial", size: 19 }),
                new TextRun({ text: "Loss = BCE(predicted_prob, actual_label)", font: "Arial", size: 19, italic: true }),
              ], { width: 8760, fill: COLORS.lightGray }),
            ]}),
            // Step 4
            new TableRow({ children: [
              makeCell("4", { width: 600, bold: true, alignment: AlignmentType.CENTER, fill: COLORS.header, color: COLORS.white, size: 22 }),
              makeMultiLineCell([
                new TextRun({ text: "Combined Loss", font: "Arial", size: 20, bold: true }),
                new TextRun({ text: "total_loss = 1.0 * emergence_loss + 1.0 * top8_loss", font: "Arial", size: 19, bold: true }),
                new TextRun({ text: "Both tasks are weighted equally", font: "Arial", size: 19, italic: true }),
              ], { width: 8760, fill: COLORS.gradient1 }),
            ]}),
            // Step 5
            new TableRow({ children: [
              makeCell("5", { width: 600, bold: true, alignment: AlignmentType.CENTER, fill: COLORS.header, color: COLORS.white, size: 22 }),
              makeMultiLineCell([
                new TextRun({ text: "Backpropagation + Optimization", font: "Arial", size: 20, bold: true }),
                new TextRun({ text: "Gradient clipping (max_norm=1.0) \u2192 AdamW step (lr=1e-3, decay=1e-4) \u2192 Cosine LR schedule", font: "Arial", size: 19 }),
              ], { width: 8760, fill: COLORS.white }),
            ]}),
            // Step 6
            new TableRow({ children: [
              makeCell("6", { width: 600, bold: true, alignment: AlignmentType.CENTER, fill: COLORS.header, color: COLORS.white, size: 22 }),
              makeMultiLineCell([
                new TextRun({ text: "Validation (every 5 epochs)", font: "Arial", size: 20, bold: true }),
                new TextRun({ text: "Evaluate on future tournaments (val split). Track val_loss + val_accuracy.", font: "Arial", size: 19 }),
                new TextRun({ text: "Save model checkpoint if val_loss improves (early stopping by best checkpoint).", font: "Arial", size: 19 }),
              ], { width: 8760, fill: COLORS.lightGray }),
            ]}),
          ],
        }),

        new Paragraph({ spacing: { after: 200 } }),
        subHeading("4.3 Loss Function Visualization"),
        bodyText("The combined loss balances two objectives:"),

        new Table({
          width: { size: CONTENT_WIDTH, type: WidthType.DXA },
          columnWidths: [4680, 4680],
          rows: [
            new TableRow({ children: [
              makeCell("MSE Loss (Emergence)", { width: 4680, bold: true, fill: COLORS.loss1, color: COLORS.white, alignment: AlignmentType.CENTER, size: 22 }),
              makeCell("BCE Loss (Top 8)", { width: 4680, bold: true, fill: "0E918C", color: COLORS.white, alignment: AlignmentType.CENTER, size: 22 }),
            ]}),
            new TableRow({ children: [
              makeMultiLineCell([
                new TextRun({ text: "Regression task", font: "Arial", size: 19, bold: true }),
                new TextRun({ text: "Penalizes squared distance", font: "Arial", size: 19 }),
                new TextRun({ text: "between predicted and actual", font: "Arial", size: 19 }),
                new TextRun({ text: "meta share changes", font: "Arial", size: 19 }),
                new TextRun({ text: " ", font: "Arial", size: 10 }),
                new TextRun({ text: "Gradients flow back through", font: "Arial", size: 18, italic: true }),
                new TextRun({ text: "archetype embeddings only", font: "Arial", size: 18, italic: true }),
              ], { width: 4680 }),
              makeMultiLineCell([
                new TextRun({ text: "Classification task", font: "Arial", size: 19, bold: true }),
                new TextRun({ text: "Penalizes confident wrong", font: "Arial", size: 19 }),
                new TextRun({ text: "predictions on archetype-", font: "Arial", size: 19 }),
                new TextRun({ text: "tournament placement", font: "Arial", size: 19 }),
                new TextRun({ text: " ", font: "Arial", size: 10 }),
                new TextRun({ text: "Gradients flow back through", font: "Arial", size: 18, italic: true }),
                new TextRun({ text: "archetype + tournament embeddings", font: "Arial", size: 18, italic: true }),
              ], { width: 4680 }),
            ]}),
          ],
        }),

        new Paragraph({ spacing: { after: 120 } }),
        bodyTextMulti([
          { text: "Multi-task benefit: ", bold: true },
          { text: "Both losses jointly optimize the shared HGT backbone. The emergence task encourages archetype embeddings to capture meta-share dynamics, while the top-8 task ensures tournament and archetype embeddings capture competitive placement patterns. The shared backbone means card embeddings benefit from both signals." },
        ]),

        // ════════════════════════════════════════════════
        // SECTION 5: END-TO-END SUMMARY
        // ════════════════════════════════════════════════
        new Paragraph({ children: [new PageBreak()] }),
        sectionHeading("5. End-to-End Data Flow Summary"),
        bodyText("Complete signal path from raw features to predictions:"),

        new Table({
          width: { size: CONTENT_WIDTH, type: WidthType.DXA },
          columnWidths: [CONTENT_WIDTH],
          rows: [
            new TableRow({ children: [makeCell("RAW NODE FEATURES", { width: CONTENT_WIDTH, alignment: AlignmentType.CENTER, bold: true, fill: COLORS.medGray, size: 22 })] }),
            new TableRow({ children: [makeCell("Card: 384-dim embedding + 11 numeric  |  Archetype: 3 features  |  Tournament: 2 features  |  Set: 3 features", { width: CONTENT_WIDTH, alignment: AlignmentType.CENTER, size: 18, fill: COLORS.lightGray })] }),
            new TableRow({ children: [makeCell("\u2193  Per-Type Linear Projection", { width: CONTENT_WIDTH, alignment: AlignmentType.CENTER, cellBorders: noBorders, bold: true, color: COLORS.header, size: 20 })] }),
            new TableRow({ children: [makeCell("SHARED 128-DIM HIDDEN SPACE", { width: CONTENT_WIDTH, alignment: AlignmentType.CENTER, bold: true, fill: COLORS.gradient2, size: 22 })] }),
            new TableRow({ children: [makeCell("\u2193  HGT Layer 1: Heterogeneous multi-head attention across all 12 edge types", { width: CONTENT_WIDTH, alignment: AlignmentType.CENTER, cellBorders: noBorders, bold: false, color: COLORS.header, size: 19 })] }),
            new TableRow({ children: [makeCell("128-DIM (1-hop neighborhood encoded)", { width: CONTENT_WIDTH, alignment: AlignmentType.CENTER, fill: "D6E4F0", size: 19 })] }),
            new TableRow({ children: [makeCell("\u2193  HGT Layer 2: Residual + LayerNorm + Dropout", { width: CONTENT_WIDTH, alignment: AlignmentType.CENTER, cellBorders: noBorders, bold: false, color: COLORS.header, size: 19 })] }),
            new TableRow({ children: [makeCell("128-DIM (2-hop patterns encoded)", { width: CONTENT_WIDTH, alignment: AlignmentType.CENTER, fill: "BDD7EE", size: 19 })] }),
            new TableRow({ children: [makeCell("\u2193  HGT Layer 3: Global metagame structure captured", { width: CONTENT_WIDTH, alignment: AlignmentType.CENTER, cellBorders: noBorders, bold: false, color: COLORS.header, size: 19 })] }),
            new TableRow({ children: [makeCell("FINAL ENRICHED EMBEDDINGS (128-dim per node)", { width: CONTENT_WIDTH, alignment: AlignmentType.CENTER, bold: true, fill: "9DC3E6", size: 22 })] }),
            new TableRow({ children: [makeCell("\u2193                                                                                               \u2193", { width: CONTENT_WIDTH, alignment: AlignmentType.CENTER, cellBorders: noBorders, bold: true, color: COLORS.header, size: 20 })] }),
          ],
        }),
        new Table({
          width: { size: CONTENT_WIDTH, type: WidthType.DXA },
          columnWidths: [4680, 4680],
          rows: [
            new TableRow({ children: [
              makeCell("HEAD 1: EMERGENCE", { width: 4680, alignment: AlignmentType.CENTER, bold: true, fill: COLORS.loss1, color: COLORS.white, size: 20 }),
              makeCell("HEAD 2: TOP 8", { width: 4680, alignment: AlignmentType.CENTER, bold: true, fill: "0E918C", color: COLORS.white, size: 20 }),
            ]}),
            new TableRow({ children: [
              makeMultiLineCell([
                new TextRun({ text: "archetype_emb \u2192 Linear(128,64)", font: "Arial", size: 18 }),
                new TextRun({ text: "\u2192 ReLU \u2192 Dropout \u2192 Linear(64,1)", font: "Arial", size: 18 }),
                new TextRun({ text: " ", font: "Arial", size: 10 }),
                new TextRun({ text: "Output: share_delta per archetype", font: "Arial", size: 18, bold: true }),
                new TextRun({ text: "Loss: MSE vs actual delta", font: "Arial", size: 18, italic: true }),
              ], { width: 4680, fill: "FADBD8" }),
              makeMultiLineCell([
                new TextRun({ text: "concat(arch_emb, tourn_emb)", font: "Arial", size: 18 }),
                new TextRun({ text: "\u2192 Linear(256,128) \u2192 ReLU", font: "Arial", size: 18 }),
                new TextRun({ text: "\u2192 Dropout \u2192 Linear(128,1) \u2192 Sigmoid", font: "Arial", size: 18 }),
                new TextRun({ text: "Output: P(top8) per pair", font: "Arial", size: 18, bold: true }),
                new TextRun({ text: "Loss: BCE vs binary label", font: "Arial", size: 18, italic: true }),
              ], { width: 4680, fill: "D1F2EB" }),
            ]}),
          ],
        }),
        new Table({
          width: { size: CONTENT_WIDTH, type: WidthType.DXA },
          columnWidths: [CONTENT_WIDTH],
          rows: [
            new TableRow({ children: [makeCell("\u2193                                    Combined: total = 1.0 * MSE + 1.0 * BCE                                    \u2193", { width: CONTENT_WIDTH, alignment: AlignmentType.CENTER, cellBorders: noBorders, bold: true, color: COLORS.header, size: 19 })] }),
            new TableRow({ children: [makeCell("BACKPROPAGATE \u2192 AdamW (lr=1e-3, decay=1e-4) \u2192 Cosine LR Schedule \u2192 Grad Clip (1.0)", { width: CONTENT_WIDTH, alignment: AlignmentType.CENTER, bold: true, fill: COLORS.header, color: COLORS.white, size: 19 })] }),
          ],
        }),

        new Paragraph({ spacing: { after: 300 } }),
        subHeading("5.1 Key Training Hyperparameters"),
        new Table({
          width: { size: CONTENT_WIDTH, type: WidthType.DXA },
          columnWidths: [3600, 2880, 2880],
          rows: [
            new TableRow({ children: [
              makeCell("Parameter", { width: 3600, bold: true, fill: COLORS.header, color: COLORS.white }),
              makeCell("Value", { width: 2880, bold: true, fill: COLORS.header, color: COLORS.white }),
              makeCell("Purpose", { width: 2880, bold: true, fill: COLORS.header, color: COLORS.white }),
            ]}),
            ...([
              ["Hidden Dimension", "128", "Shared embedding space size"],
              ["Attention Heads", "4", "Parallel attention patterns"],
              ["HGT Layers", "3", "Message passing depth"],
              ["Dropout", "0.2", "Regularization"],
              ["Learning Rate", "1e-3", "AdamW step size"],
              ["Weight Decay", "1e-4", "L2 regularization"],
              ["Epochs", "100", "Training iterations"],
              ["LR Schedule", "Cosine Annealing", "Gradual LR reduction"],
              ["Grad Clip", "max_norm=1.0", "Prevent exploding gradients"],
              ["Temporal Split", "75% / 25%", "Train on past, validate on future"],
              ["Neg Sampling", "2:1 ratio", "Negatives per positive (top8 task)"],
            ].map((row, i) =>
              new TableRow({
                children: [
                  makeCell(row[0], { width: 3600, bold: true, size: 19, fill: i % 2 === 0 ? COLORS.white : COLORS.lightGray }),
                  makeCell(row[1], { width: 2880, size: 19, fill: i % 2 === 0 ? COLORS.white : COLORS.lightGray }),
                  makeCell(row[2], { width: 2880, size: 19, fill: i % 2 === 0 ? COLORS.white : COLORS.lightGray }),
                ],
              })
            )),
          ],
        }),

        new Paragraph({ spacing: { after: 200 } }),
        subHeading("5.2 Evaluation Metrics"),
        bodyText("At the end of training, the best checkpoint (lowest val_loss) is loaded for final evaluation:"),

        new Table({
          width: { size: CONTENT_WIDTH, type: WidthType.DXA },
          columnWidths: [4680, 4680],
          rows: [
            new TableRow({ children: [
              makeCell("Emergence Head", { width: 4680, bold: true, fill: COLORS.loss1, color: COLORS.white, alignment: AlignmentType.CENTER }),
              makeCell("Top 8 Head", { width: 4680, bold: true, fill: "0E918C", color: COLORS.white, alignment: AlignmentType.CENTER }),
            ]}),
            new TableRow({ children: [
              makeMultiLineCell([
                new TextRun({ text: "Ranked archetype predictions", font: "Arial", size: 19 }),
                new TextRun({ text: "Each archetype gets a signed", font: "Arial", size: 19 }),
                new TextRun({ text: "score indicating predicted", font: "Arial", size: 19 }),
                new TextRun({ text: "meta share movement", font: "Arial", size: 19 }),
              ], { width: 4680 }),
              makeMultiLineCell([
                new TextRun({ text: "Accuracy: overall correctness", font: "Arial", size: 19 }),
                new TextRun({ text: "Precision: of predicted top8s,", font: "Arial", size: 19 }),
                new TextRun({ text: "  how many actually placed?", font: "Arial", size: 19 }),
                new TextRun({ text: "Recall: of actual top8s,", font: "Arial", size: 19 }),
                new TextRun({ text: "  how many were predicted?", font: "Arial", size: 19 }),
                new TextRun({ text: "F1: harmonic mean of P and R", font: "Arial", size: 19 }),
              ], { width: 4680 }),
            ]}),
          ],
        }),
      ],
    },
  ],
});

// ── Write file ──
const OUTPUT_PATH = process.argv[2] || "mtg_architecture_guide.docx";
Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(OUTPUT_PATH, buffer);
  console.log(`Document written to ${OUTPUT_PATH}`);
});
