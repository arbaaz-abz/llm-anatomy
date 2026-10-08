// decoder-anatomy frames 1–3: pieces in, the embedding lookup, the residual stream. Each draw is a pure function of progress.
import * as G from '@shared/glyphs.js';
import { paramBreakdown, PRESETS } from '@math/params.js';
import { TOKENS, VOCAB, D_MODEL, X, E, MAX_ABS } from './stream.js';
import { CELL, SMALL, STREAM, LANE, PATCH, STACK_H, EAR, seg, lerp, layer, note, chips, chipX, stream, lane, sheet, emptyFrame } from './layout.js';

const STREAM_W = 8 * CELL;
const STREAM_H = TOKENS.length * CELL;
const E_POS = Object.freeze({ x: 430, y: 58 }); // frame 2: the four lit rows of E, collapsed (storyboard §4 fallback)
const LANE_ROWS = 0.7; // frame 2: rows copy into the stream during [0, 0.7], then the image side lane runs
const E_PARAMS = paramBreakdown(PRESETS.toy).parts.embedding; // 16 × 8 = 128

function emptyStream(svg) {
  emptyFrame(svg, { x: STREAM.x, y: STREAM.y, w: STREAM_W, h: STREAM_H, label: 'residual stream [4 × 8]', rowLabels: TOKENS });
}

// Frame 1: the chips slide in one by one; the image patch fades in at the right.
export function drawFrame1(svg, p) {
  const t = (i) => seg(p, i * 0.15, i * 0.15 + 0.3);
  chips(svg, TOKENS, { opacity: t, dx: (i) => -24 * (1 - t(i)) });
  const patchIn = seg(p, 0.55, 0.9);
  G.patch(layer(svg, patchIn), { x: PATCH.x, y: PATCH.y, pixels: EAR, state: 'dim' });
  note(layer(svg, patchIn), PATCH.x + 32, PATCH.y + 16, 'image patch');
  emptyStream(svg);
  note(svg, STREAM.x, 246, 'n = 4 pieces · d_model = 8 numbers per row');
  note(svg, STREAM.x, 264, 'an audio frame would be one more piece');
}

function embeddingTable(svg) {
  const g = layer(svg, 0.45);
  G.matrix(g, { x: E_POS.x, y: E_POS.y, values: E.slice(0, TOKENS.length), cell: SMALL, maxAbs: MAX_ABS, rowLabels: TOKENS });
  note(svg, E_POS.x, E_POS.y - 10, 'E [16 × 8]');
  note(svg, PATCH.x, E_POS.y + 4 * SMALL + 16, '12 more rows (unused here)');
  note(svg, PATCH.x, E_POS.y + 4 * SMALL + 32, `${VOCAB.length} × ${D_MODEL} = ${E_PARAMS} parameters`);
}

// One row's lookup: chip → its row of E lights up → the row slides into the stream, growing to readable cells.
function lookupRow(svg, i, t) {
  const from = { x: E_POS.x, y: E_POS.y + i * SMALL };
  const to = { x: STREAM.x, y: STREAM.y + i * CELL };
  const reach = seg(t, 0, 0.4);
  const slide = seg(t, 0.4, 1);
  G.flow(svg, { from: [chipX(i) + 14, 32], to: [E_POS.x - 40, from.y + SMALL / 2], carry: 'activation', progress: reach });
  if (slide === 0) G.vector(svg, { x: from.x, y: from.y, values: E[i], cell: SMALL, orient: 'row', maxAbs: MAX_ABS });
  else G.vector(svg, { x: lerp(from.x, to.x, slide), y: lerp(from.y, to.y, slide), values: E[i], cell: lerp(SMALL, CELL, slide), orient: 'row', maxAbs: MAX_ABS });
}

function sideLane(svg, t) {
  note(svg, PATCH.x, PATCH.y - 8, 'if the input had an image:');
  G.patch(svg, { x: PATCH.x, y: PATCH.y, pixels: EAR, state: 'dim' });
  const stops = [PATCH.y + 26, PATCH.y + 48, PATCH.y + 90, PATCH.y + 132];
  G.block(svg, { x: PATCH.x, y: stops[1], w: 160, h: 22, label: 'vision encoder' });
  G.block(svg, { x: PATCH.x, y: stops[2], w: 160, h: 22, label: 'projector [→ 8]' });
  const flowsFrom = [stops[0], stops[1] + 22, stops[2] + 22];
  const flowsTo = [stops[1], stops[2], stops[3]];
  flowsFrom.forEach((y1, k) => {
    const dot = seg(t, k / 3, (k + 1) / 3);
    if (t > k / 3) G.flow(svg, { from: [PATCH.x + 12, y1], to: [PATCH.x + 12, flowsTo[k] - 2], carry: 'activation', progress: dot });
  });
  G.vector(layer(svg, 0.45), { x: PATCH.x, y: stops[3], values: E[4], cell: SMALL, orient: 'row', maxAbs: MAX_ABS });
  note(svg, PATCH.x, stops[3] + SMALL + 14, 'would be row 5');
}

// Frame 2: each token looks up its row of E (top to bottom); then the image patch takes the longer road.
export function drawFrame2(svg, p) {
  chips(svg);
  embeddingTable(svg);
  const per = LANE_ROWS / TOKENS.length;
  const done = TOKENS.filter((_, i) => p >= (i + 1) * per).length;
  if (done === TOKENS.length) stream(svg, X, { cell: CELL });
  else {
    emptyStream(svg);
    X.slice(0, done).forEach((row, i) => G.vector(svg, { x: STREAM.x, y: STREAM.y + i * CELL, values: row, cell: CELL, orient: 'row', maxAbs: MAX_ABS }));
    lookupRow(svg, done, seg(p, done * per, (done + 1) * per));
  }
  note(svg, STREAM.x, 246, 'x_cat = E[cat]: a lookup copies row 2 of E');
  sideLane(svg, seg(p, LANE_ROWS, 1));
}

// Frame 3: the stream with its numbers; the block lane draws downward and the four rows pass along it as one sheet.
export function drawFrame3(svg, p) {
  chips(svg);
  stream(svg, X, { cell: CELL });
  const draw = seg(p, 0, 0.6);
  const clipId = `${svg.dataset.hatchId}-lane`;
  const clip = G.svgEl('clipPath', { id: clipId }, svg.querySelector('defs'));
  G.svgEl('rect', { x: LANE.x - 4, y: LANE.y - 4, width: LANE.w + 8, height: (STACK_H.two + 8) * draw }, clip);
  const drawn = G.svgEl('g', { 'clip-path': `url(#${clipId})` }, svg);
  lane(drawn, { opacity: 0.55 });
  const travel = seg(p, 0.3, 0.95);
  if (travel > 0 && travel < 1) sheet(svg, X, lerp(LANE.y, LANE.y + STACK_H.two - 20, travel), Math.sin(Math.PI * travel));
  note(svg, STREAM.x, 246, 'each block reads X and adds to it; nothing is overwritten');
}
