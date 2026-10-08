// decoder-anatomy frames 7–9: the stack × N, the output head (scores → probabilities), and the decode loop with its KV cache.
import * as G from '@shared/glyphs.js';
import { paramBreakdown, PRESETS } from '@math/params.js';
import { TOKENS, X, E, SAT, DOWN, X2_SAT, MAX_ABS, PROB_MAX_ABS, SHOWN_WORDS, OTHERS, SCORE_CELLS, PROB_CELLS, PROB_TOTAL } from './stream.js';
import { CELL, SMALL, STREAM, LANE, STRIP, STACK_H, CHIPS, seg, lerp, layer, note, chips, chipX, stream, streamRowY, lane, laneBlockTop, linkedRow, sheet } from './layout.js';
import { mlpBox, moeBox, rowsWith } from './frames-block.js';
import { int } from './format.js';

const DENSE = paramBreakdown(PRESETS.toy);
const E7 = Object.freeze({ x: 274, y: 58 }); // frame 7: E collapsed to its four lit rows, beside the stream
const HEAD = Object.freeze({ // the final norm and the unembedding: waiting in frame 7, in the pipeline in frame 8
  norm: { w: 76, from: { x: 64, y: 300 }, to: { x: 228, y: 160 } },
  wu: { w: 96, from: { x: 166, y: 300 }, to: { x: 320, y: 160 } },
  h: 28,
});
const OUT = Object.freeze({ scores: 212, probs: 290 });
const WORDS5 = [...TOKENS, 'on'];
const minus = (s) => s.replace('-', '−');

function headBlocks(svg, t, opacity = 1) {
  const g = layer(svg, opacity);
  const at = (b) => ({ x: lerp(b.from.x, b.to.x, t), y: lerp(b.from.y, b.to.y, t) });
  const n = at(HEAD.norm);
  const u = at(HEAD.wu);
  G.block(g, { x: n.x, y: n.y, w: HEAD.norm.w, h: HEAD.h, label: 'final norm', state: t === 0 ? 'dim' : 'idle' });
  G.block(g, { x: u.x, y: u.y, w: HEAD.wu.w, h: HEAD.h, label: 'W_U [8 × 16]', state: 'dim' });
  note(g, u.x, u.y - 6, 'unembedding');
  return { n, u };
}

function stackTexts(svg, opacity) {
  const g = layer(svg, opacity);
  const top = LANE.y + STACK_H.folded + 16;
  note(g, LANE.x, top, 'dense MLP restored');
  note(g, LANE.x, top + 16, 'per block, dense toy:');
  note(g, LANE.x, top + 32, `${int(DENSE.perLayer.attention)} + ${int(DENSE.perLayer.mlp)} + ${int(DENSE.perLayer.norms)} = ${int(DENSE.perLayer.attention + DENSE.perLayer.mlp + DENSE.perLayer.norms)}`);
  note(g, LANE.x, top + 48, `N = ${PRESETS.toy.layers} here`);
}

function collapsedE(svg, opacity) {
  const g = layer(svg, opacity * 0.6);
  G.matrix(g, { x: E7.x, y: E7.y, values: E.slice(0, TOKENS.length), cell: SMALL, maxAbs: MAX_ABS, rowLabels: TOKENS });
  note(layer(svg, opacity), E7.x, E7.y - 10, 'E [16 × 8]');
  note(layer(svg, opacity), E7.x, E7.y + 4 * SMALL + 16, '12 more rows');
}

// Frame 7 (key frame): the dense MLP is back, the row docks, the stream passes both blocks, the stack folds to "× N".
export function drawFrame7(svg, p) {
  chips(svg);
  const dock = seg(p, 0.2, 0.4);
  stream(svg, rowsWith(SAT, X2_SAT), { selected: [SAT] });
  const restore = seg(p, 0, 0.2);
  const away = 1 - seg(p, 0.2, 0.4);
  if (away > 0 && restore < 1) moeBox(svg, { opacity: (1 - restore) * away });
  if (away > 0) mlpBox(svg, { opacity: restore * away, title: 'dense MLP restored' });
  collapsedE(svg, seg(p, 0.3, 0.5));
  if (dock < 1) {
    const cell = lerp(CELL, SMALL, dock);
    const y = lerp(STRIP.a, streamRowY(SAT), dock);
    linkedRow(svg, 'x', { x: STRIP.x, y, values: X2_SAT, cell });
    G.selectionMark(svg, { x: STRIP.x, y, w: 8 * cell, h: cell });
  }
  const fold = seg(p, 0.8, 1);
  if (fold < 1) lane(svg, { opacity: 1 - fold });
  if (fold > 0) lane(svg, { fold: true, opacity: fold });
  const pass = seg(p, 0.4, 0.8);
  if (pass > 0 && pass < 1) sheet(svg, X, lerp(LANE.y, LANE.y + STACK_H.two - 20, pass), Math.sin(Math.PI * pass));
  stackTexts(svg, restore);
  headBlocks(svg, 0);
}

function wordLabels(svg) {
  SHOWN_WORDS.forEach((w, i) => note(svg, STRIP.x + i * CELL + CELL / 2, OUT.scores - 6, w === '.' ? '"."' : w, { anchor: 'middle' }));
  // The collapsed cell's label is wider than a cell, so it starts at the cell instead of centering on it.
  note(svg, STRIP.x + SHOWN_WORDS.length * CELL + 2, OUT.scores - 6, `${OTHERS} others`);
}

function eachNote(svg, y, text) {
  note(svg, STRIP.x + 4.5 * CELL, y, text, { anchor: 'middle' });
}

// The two five-cell rows (scores, then probabilities) with the line that says what is being compared (README lesson 21).
function outputRows(svg, { scoresT = 1, probsT = 1 } = {}) {
  if (scoresT === 0) return;
  wordLabels(svg);
  linkedRow(svg, 'z', { x: STRIP.x, y: OUT.scores, values: SCORE_CELLS.slice(0, Math.ceil(5 * scoresT)), cell: CELL, maxAbs: MAX_ABS, label: 'scores' });
  if (scoresT === 1) eachNote(svg, OUT.scores + CELL + 14, `${minus(SCORE_CELLS[4].toFixed(1))} each`);
  if (probsT === 0) return;
  note(svg, STRIP.x, OUT.probs - 10, 'same five words; scores can be any size, probabilities add to 1');
  const values = SCORE_CELLS.map((z, i) => lerp(z, PROB_CELLS[i], probsT));
  linkedRow(svg, 'p', { x: STRIP.x, y: OUT.probs, values, cell: CELL, maxAbs: lerp(MAX_ABS, PROB_MAX_ABS, probsT), label: 'probs' });
  if (probsT === 1) {
    eachNote(svg, OUT.probs + CELL + 14, `${PROB_CELLS[4].toFixed(3)} each`);
    note(svg, STRIP.x + 5 * CELL + 12, OUT.probs + CELL / 2 + 4, `Σ = ${PROB_TOTAL.toFixed(3)}`);
  }
}

// x_down lifted out and the flows through the final norm and W_U.
function headPipeline(svg, { move, lift, through, opacity = 1 }) {
  const g = layer(svg, opacity);
  const { n, u } = headBlocks(g, move);
  const y = lerp(streamRowY(DOWN), STRIP.a, lift);
  linkedRow(g, 'x', { x: STRIP.x, y, values: X[DOWN], cell: SMALL, label: lift === 1 ? 'x_down' : undefined });
  G.selectionMark(g, { x: STRIP.x, y, w: 8 * SMALL, h: SMALL });
  if (through === 0) return;
  const mid = HEAD.h / 2;
  G.flow(g, { from: [STRIP.x + 8 * SMALL + 2, STRIP.a + SMALL / 2], to: [n.x - 2, n.y + mid], carry: 'activation', progress: seg(through, 0, 0.5) });
  G.flow(g, { from: [n.x + HEAD.norm.w + 2, n.y + mid], to: [u.x - 2, u.y + mid], carry: 'activation', progress: seg(through, 0.5, 1) });
}

// Frame 8: the last row after block 2 is normalized and multiplied by W_U; five score cells, then the same five as probabilities.
export function drawFrame8(svg, p) {
  chips(svg);
  stream(svg, rowsWith(SAT, X2_SAT), { selected: [DOWN] });
  lane(svg, { fold: true });
  headPipeline(svg, { move: seg(p, 0, 0.2), lift: seg(p, 0.1, 0.3), through: seg(p, 0.3, 0.5) });
  const scoresT = seg(p, 0.5, 0.7);
  if (scoresT > 0) G.flow(svg, { from: [HEAD.wu.to.x + HEAD.wu.w / 2, HEAD.wu.to.y + HEAD.h + 2], to: [STRIP.x + 5 * CELL + 4, OUT.scores + CELL / 2], carry: 'activation', progress: scoresT });
  outputRows(svg, { scoresT, probsT: seg(p, 0.7, 1) });
}

function kvStacks(svg, grown) {
  [0, 1].forEach((b) => G.kvStack(svg, { x: 340, y: laneBlockTop(b, true) + 30, count: grown ? 5 : 4, tile: 14, highlight: grown ? [4] : [], label: `KV · block ${b === 0 ? '1' : 'N'}` }));
}

// Frame 9: "on" is picked and appended; only its new row is computed, and each block's KV cache grows by one position.
export function drawFrame9(svg, p) {
  chips(svg);
  const travel = seg(p, 0, 0.35);
  const slot = { x: chipX(4, WORDS5), y: CHIPS.y };
  G.token(svg, { x: lerp(STRIP.x, slot.x, travel), y: lerp(OUT.probs + 8, slot.y, travel), text: 'on', index: 5 });
  const grow = seg(p, 0.35, 0.5);
  const rows = [...rowsWith(SAT, X2_SAT), E[4]];
  if (grow < 1) {
    stream(svg, rows.slice(0, 4));
    G.vector(layer(svg, grow), { x: STREAM.x, y: streamRowY(4), values: E[4], cell: SMALL, orient: 'row', maxAbs: MAX_ABS });
  } else stream(svg, rows, { selected: [4], words: WORDS5 });
  lane(svg, { fold: true });
  kvStacks(svg, p >= 0.65);
  const fade = 1 - seg(p, 0, 0.2);
  if (fade > 0) headPipeline(svg, { move: 1, lift: 1, through: 1, opacity: fade });
  outputRows(svg);
  G.selectionMark(svg, { x: STRIP.x, y: OUT.probs, w: CELL, h: CELL });
  const loop = seg(p, 0.8, 1);
  if (loop > 0) G.flow(svg, { from: [slot.x + 15, CHIPS.y + 26], to: [STREAM.x + 8 * SMALL + 4, streamRowY(4) + SMALL / 2], carry: 'token', progress: loop });
}

