// decoder-anatomy frames 1–3: pieces in, the embedding lookup, the residual stream. Each draw is a pure function of progress.
import * as G from '@shared/glyphs.js';
import { paramBreakdown, PRESETS } from '@math/params.js';
import { TOKENS, VOCAB, D_MODEL, X, E, PATCH_VEC, MAX_ABS } from './numbers.js';
import { CELL, SMALL, STREAM, LANE, PATCH, STACK_H, EAR, HANDOFF, seg, lerp, arriving, leaving, layer, note, chips, chipX, stream, lane, sheet, emptyFrame } from './stage.js';

const STREAM_W = 8 * CELL;
const STREAM_H = TOKENS.length * CELL;
const E_POS = Object.freeze({ x: 430, y: 58 }); // frame 2: the four lit rows of E, collapsed (storyboard §4 fallback)
const LANE_ROWS = 0.7; // frame 2: rows copy into the stream during [HANDOFF, 0.7], then the image side lane runs
const E_PARAMS = paramBreakdown(PRESETS.toy).parts.embedding; // 16 × 8 = 128
const LANE_STOPS = Object.freeze([PATCH.y + 26, PATCH.y + 48, PATCH.y + 90, PATCH.y + 132]); // patch, encoder, projector, vector

function emptyStream(svg) {
  emptyFrame(svg, { x: STREAM.x, y: STREAM.y, w: STREAM_W, h: STREAM_H, label: 'residual stream [4 × 8]', rowLabels: TOKENS });
}

function frame1Notes(svg) {
  note(svg, PATCH.x + 32, PATCH.y + 16, 'image patch');
  note(svg, STREAM.x, 246, 'n = 4 pieces · d_model = 8 numbers per row');
  note(svg, STREAM.x, 264, 'an audio frame would be one more piece');
}

// Frame 1: the chips slide in one by one; the image patch fades in at the right.
export function drawFrame1(svg, p) {
  const t = (i) => seg(p, i * 0.15, i * 0.15 + 0.3);
  chips(svg, TOKENS, { opacity: t, dx: (i) => -24 * (1 - t(i)) });
  const patchIn = seg(p, 0.55, 0.9);
  G.patch(layer(svg, patchIn), { x: PATCH.x, y: PATCH.y, pixels: EAR, state: 'dim' });
  emptyStream(svg);
  frame1Notes(layer(svg, patchIn));
}

function embeddingTable(parent) {
  G.matrix(layer(parent, 0.45), { x: E_POS.x, y: E_POS.y, values: E.slice(0, TOKENS.length), cell: SMALL, maxAbs: MAX_ABS, rowLabels: TOKENS });
  note(parent, E_POS.x, E_POS.y - 10, 'E [16 × 8]');
  note(parent, PATCH.x, E_POS.y + 4 * SMALL + 16, '12 more rows (unused here)');
  note(parent, PATCH.x, E_POS.y + 4 * SMALL + 32, `${VOCAB.length} × ${D_MODEL} = ${E_PARAMS} parameters`);
}

// One row's lookup: chip → its row of E lights up → the row slides into the stream, growing to readable cells.
function lookupRow(svg, i, t) {
  const from = { x: E_POS.x, y: E_POS.y + i * SMALL };
  const to = { x: STREAM.x, y: STREAM.y + i * CELL };
  const reach = seg(t, 0, 0.4);
  const slide = seg(t, 0.4, 1);
  if (reach > 0) G.flow(svg, { from: [chipX(i) + 14, 32], to: [E_POS.x - 40, from.y + SMALL / 2], carry: 'activation', progress: reach });
  if (slide === 0) G.vector(svg, { x: from.x, y: from.y, values: E[i], cell: SMALL, orient: 'row', maxAbs: MAX_ABS });
  else G.vector(svg, { x: lerp(from.x, to.x, slide), y: lerp(from.y, to.y, slide), values: E[i], cell: lerp(SMALL, CELL, slide), orient: 'row', maxAbs: MAX_ABS });
}

// The image's longer road: patch → vision encoder → projector → a vector that would be row 5 (a branch, never joined).
function sideLane(parent, t) {
  note(parent, PATCH.x, PATCH.y - 8, 'if the input had an image:');
  G.block(parent, { x: PATCH.x, y: LANE_STOPS[1], w: 160, h: 22, label: 'vision encoder' });
  G.block(parent, { x: PATCH.x, y: LANE_STOPS[2], w: 160, h: 22, label: 'projector [→ 8]' });
  [LANE_STOPS[0], LANE_STOPS[1] + 22, LANE_STOPS[2] + 22].forEach((y1, k) => {
    const dot = seg(t, k / 3, (k + 1) / 3);
    if (dot > 0) G.flow(parent, { from: [PATCH.x + 12, y1], to: [PATCH.x + 12, LANE_STOPS[k + 1] - 2], carry: 'activation', progress: dot });
  });
  G.vector(layer(parent, 0.45), { x: PATCH.x, y: LANE_STOPS[3], values: PATCH_VEC, cell: SMALL, orient: 'row', maxAbs: MAX_ABS });
  note(parent, PATCH.x, LANE_STOPS[3] + SMALL + 14, 'would be row 5');
}

// Frame 2's new marks (the lookup table, the side lane, the x_cat note), at the given opacity and side-lane progress.
function frame2Marks(svg, opacity, laneT) {
  const g = layer(svg, opacity);
  embeddingTable(g);
  sideLane(g, laneT);
  note(g, STREAM.x, 246, 'x_cat = E[cat]: a lookup copies row 2 of E');
}

function frame2Stream(svg, p) {
  const per = (LANE_ROWS - HANDOFF) / TOKENS.length;
  const done = TOKENS.filter((_, i) => p >= HANDOFF + (i + 1) * per).length;
  if (done === TOKENS.length) return stream(svg, X, { cell: CELL });
  emptyStream(svg);
  X.slice(0, done).forEach((row, i) => G.vector(svg, { x: STREAM.x, y: STREAM.y + i * CELL, values: row, cell: CELL, orient: 'row', maxAbs: MAX_ABS }));
  if (p > HANDOFF) lookupRow(svg, done, seg(p, HANDOFF + done * per, HANDOFF + (done + 1) * per));
  return null;
}

// Frame 2: each token looks up its row of E (top to bottom); then the image patch takes the longer road.
export function drawFrame2(svg, p) {
  chips(svg);
  frame1Notes(layer(svg, leaving(p)));
  G.patch(svg, { x: PATCH.x, y: PATCH.y, pixels: EAR, state: 'dim' });
  frame2Marks(svg, arriving(p), seg(p, LANE_ROWS, 1));
  frame2Stream(svg, p);
}

// Frame 3: the stream with its numbers; the block lane draws downward and the four rows pass along it as one sheet.
export function drawFrame3(svg, p) {
  chips(svg);
  stream(svg, X, { cell: CELL });
  const out = 1 - seg(p, 0, 0.2);
  if (out > 0) {
    G.patch(layer(svg, out), { x: PATCH.x, y: PATCH.y, pixels: EAR, state: 'dim' });
    frame2Marks(svg, out, 1);
  }
  const draw = seg(p, 0.1, 0.6);
  const clipId = `${svg.dataset.hatchId}-lane`;
  const clip = G.svgEl('clipPath', { id: clipId }, svg.querySelector('defs'));
  G.svgEl('rect', { x: LANE.x - 4, y: LANE.y - 4, width: LANE.w + 8, height: (STACK_H.two + 8) * draw }, clip);
  lane(G.svgEl('g', { 'clip-path': `url(#${clipId})` }, svg), { opacity: 0.55 });
  const travel = seg(p, 0.3, 0.95);
  if (travel > 0 && travel < 1) sheet(svg, X, lerp(LANE.y, LANE.y + STACK_H.two - 20, travel), Math.sin(Math.PI * travel));
  note(layer(svg, seg(p, 0.1, 0.3)), STREAM.x, 246, 'each block reads X and adds to it; nothing is overwritten');
}
