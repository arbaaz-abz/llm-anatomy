// decoder-anatomy frames 7–9: the stack × N, the output head (scores → probabilities), and the decode loop with its KV cache.
import * as G from '@shared/glyphs.js';
import { paramBreakdown, PRESETS } from '@math/params.js';
import { TOKENS, X, E, SAT, DOWN, X1_SAT, X2_SAT, MAX_ABS, PROB_MAX_ABS, SHOWN_WORDS, OTHERS, SCORE_CELLS, PROB_CELLS, PROB_TOTAL } from './numbers.js';
import { CELL, SMALL, STREAM, LANE, STRIP, STACK_H, CHIPS, seg, lerp, arriving, leaving, layer, note, chips, chipX, stream, streamRowY, lane, laneBlend, laneBlockTop, linkedRow, blendRows, sheet } from './stage.js';
import { mlpBox, moeBox, moeReadout, rowsWith } from './frames-block.js';
import { int } from './format.js';

const DENSE = paramBreakdown(PRESETS.toy);
const E7 = Object.freeze({ x: 274, y: 58 }); // frame 7: E collapsed to its four lit rows, beside the stream
const HEAD = Object.freeze({ // the final norm and the unembedding: waiting in frame 7, in the pipeline in frame 8
  norm: { w: 76, from: { x: 64, y: 300 }, to: { x: 228, y: 160 } },
  wu: { w: 96, from: { x: 166, y: 300 }, to: { x: 320, y: 160 } },
  h: 28,
});
const OUT = Object.freeze({ scores: 212, probs: 290 });
const WORDS5 = Object.freeze([...TOKENS, 'on']);
const ON = 4; // the appended token's row and chip
const MOE_LANE = Object.freeze({ halves: ['attention', 'MoE'], active: { block: 1, half: 1 } });
const minus = (s) => s.replace('-', '−');
const P3 = (v) => v.toFixed(3); // probabilities at the storyboard's 3 d.p.

// Waiting (t = 0, dim: off the current path) or moving into / in the pipeline (idle: the learner reads it).
function headBlocks(parent, t) {
  const at = (b) => ({ x: lerp(b.from.x, b.to.x, t), y: lerp(b.from.y, b.to.y, t) });
  const n = at(HEAD.norm);
  const u = at(HEAD.wu);
  const state = t === 0 ? 'dim' : 'idle';
  G.block(parent, { x: n.x, y: n.y, w: HEAD.norm.w, h: HEAD.h, label: 'final norm', state });
  G.block(parent, { x: u.x, y: u.y, w: HEAD.wu.w, h: HEAD.h, label: 'W_U [8 × 16]', state });
  note(parent, u.x, u.y - 6, 'unembedding');
  return { n, u };
}

function stackTexts(parent) {
  const top = LANE.y + STACK_H.folded + 16;
  note(parent, LANE.x, top, 'dense MLP restored');
  note(parent, LANE.x, top + 16, 'per block, dense toy:');
  note(parent, LANE.x, top + 32, `${int(DENSE.perLayer.attention)} + ${int(DENSE.perLayer.mlp)} + ${int(DENSE.perLayer.norms)} = ${int(DENSE.perLayer.attention + DENSE.perLayer.mlp + DENSE.perLayer.norms)}`);
  note(parent, LANE.x, top + 48, `N = ${PRESETS.toy.layers} here`);
}

function collapsedE(parent) {
  G.matrix(layer(parent, 0.6), { x: E7.x, y: E7.y, values: E.slice(0, TOKENS.length), cell: SMALL, maxAbs: MAX_ABS, rowLabels: TOKENS });
  note(parent, E7.x, E7.y - 10, 'E [16 × 8]');
  note(parent, E7.x, E7.y + 4 * SMALL + 16, '12 more rows');
}

// The row docks back from the strip into the stream; until then the dense MLP turns x′ into x″ in place.
function dockingRow(svg, restore, dock) {
  if (dock === 1) return;
  const cell = lerp(CELL, SMALL, dock);
  const y = lerp(STRIP.a, streamRowY(SAT), dock);
  linkedRow(svg, 'x', { x: STRIP.x, y, values: blendRows(X1_SAT, X2_SAT, restore), cell, label: dock === 0 ? 'x′_sat' : undefined });
  G.selectionMark(svg, { x: STRIP.x, y, w: 8 * cell, h: cell });
}

// The stack: MoE branch → dense MLP (restore), the stream passes both blocks, then the fold to "⋮ × N".
function stackFold(svg, restore, fold) {
  if (fold < 1) laneBlend(svg, MOE_LANE, {}, restore);
  if (fold > 0) lane(svg, { fold: true, opacity: fold });
}

// Frame 7 (key frame): the dense MLP is back, the row docks, the stream passes both blocks, the stack folds to "× N".
export function drawFrame7(svg, p) {
  const restore = seg(p, 0, 0.2);
  const dock = seg(p, 0.2, 0.4);
  const away = 1 - seg(p, 0.2, 0.4);
  chips(svg, TOKENS, { follow: [SAT] });
  stream(svg, rowsWith(SAT, dock === 1 ? X2_SAT : X1_SAT), { selected: [SAT] });
  moeReadout(layer(svg, leaving(p)), 1);
  if (away > 0 && restore < 1) moeBox(svg, { opacity: (1 - restore) * away });
  if (away > 0 && restore > 0) mlpBox(svg, { opacity: restore * away, title: 'dense MLP restored' });
  collapsedE(layer(svg, seg(p, 0.3, 0.5)));
  dockingRow(svg, restore, dock);
  const fold = seg(p, 0.8, 1);
  stackFold(svg, restore, fold);
  const pass = seg(p, 0.4, 0.8);
  if (pass > 0 && pass < 1) sheet(svg, X, lerp(LANE.y, LANE.y + STACK_H.two - 20, pass), Math.sin(Math.PI * pass));
  stackTexts(layer(svg, restore));
  headBlocks(layer(svg, seg(p, 0.4, 0.6)), 0);
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
// The probability row interpolates between the two computed states; it prints one format per state (scores, then 3 d.p.).
function outputRows(svg, { scoresT = 1, probsT = 1 } = {}) {
  if (scoresT === 0) return;
  wordLabels(svg);
  linkedRow(svg, 'z', { x: STRIP.x, y: OUT.scores, values: SCORE_CELLS.slice(0, Math.ceil(5 * scoresT)), cell: CELL, maxAbs: MAX_ABS, label: 'scores' });
  if (scoresT === 1) eachNote(svg, OUT.scores + CELL + 14, `${minus(SCORE_CELLS[4].toFixed(1))} each`);
  if (probsT === 0) return;
  note(svg, STRIP.x, OUT.probs - 10, 'same five words; scores can be any size, probabilities add to 1');
  const values = SCORE_CELLS.map((z, i) => lerp(z, PROB_CELLS[i], probsT));
  linkedRow(svg, 'p', { x: STRIP.x, y: OUT.probs, values, cell: CELL, maxAbs: lerp(MAX_ABS, PROB_MAX_ABS, probsT), label: 'probs', format: probsT >= 0.5 ? P3 : undefined });
  if (probsT === 1) {
    eachNote(svg, OUT.probs + CELL + 14, `${P3(PROB_CELLS[4])} each`);
    note(svg, STRIP.x + 5 * CELL + 12, OUT.probs + CELL / 2 + 4, `Σ = ${PROB_TOTAL.toFixed(3)}`);
  }
}

// x_down lifted out, the flows through the final norm and W_U, and W_U's arrow down to the scores.
function headPipeline(parent, { move, lift, through, scores }) {
  const { n, u } = headBlocks(parent, move);
  if (lift > 0) {
    const y = lerp(streamRowY(DOWN), STRIP.a, lift);
    linkedRow(parent, 'x', { x: STRIP.x, y, values: X[DOWN], cell: SMALL, label: lift === 1 ? 'x_down' : undefined });
    G.selectionMark(parent, { x: STRIP.x, y, w: 8 * SMALL, h: SMALL });
  }
  const mid = HEAD.h / 2;
  if (through > 0) G.flow(parent, { from: [STRIP.x + 8 * SMALL + 2, STRIP.a + SMALL / 2], to: [n.x - 2, n.y + mid], carry: 'activation', progress: seg(through, 0, 0.5) });
  if (through > 0.5) G.flow(parent, { from: [n.x + HEAD.norm.w + 2, n.y + mid], to: [u.x - 2, u.y + mid], carry: 'activation', progress: seg(through, 0.5, 1) });
  if (scores > 0) G.flow(parent, { from: [u.x + HEAD.wu.w / 2, u.y + HEAD.h + 2], to: [STRIP.x + 5 * CELL + 4, OUT.scores + CELL / 2], carry: 'activation', progress: scores });
}

// Frame 8: the last row after block 2 is normalized and multiplied by W_U; five score cells, then the same five as probabilities.
export function drawFrame8(svg, p) {
  const out = leaving(p);
  const inn = arriving(p);
  chips(svg, TOKENS, { follow: [SAT], selOpacity: out });
  followChip(svg, DOWN, TOKENS, inn);
  stream(svg, rowsWith(SAT, X2_SAT), { selected: [SAT], selOpacity: out });
  if (inn > 0) G.selectionMark(layer(svg, inn), { x: STREAM.x, y: streamRowY(DOWN), w: 8 * SMALL, h: SMALL });
  lane(svg, { fold: true });
  if (out > 0) {
    const old = layer(svg, out);
    collapsedE(old);
    stackTexts(old);
  }
  const scoresT = seg(p, 0.5, 0.7);
  headPipeline(svg, { move: seg(p, 0, 0.2), lift: seg(p, 0.1, 0.3), through: seg(p, 0.3, 0.5), scores: scoresT });
  outputRows(svg, { scoresT, probsT: seg(p, 0.7, 1) });
}

// The selection outline on one chip (the followed token), at the given opacity.
function followChip(svg, i, words, opacity) {
  if (opacity > 0) G.selectionMark(layer(svg, opacity), { x: chipX(i, words), y: CHIPS.y, w: G.tokenWidth(words[i]), h: 24 });
}

function kvStacks(parent, grown) {
  [0, 1].forEach((b) => G.kvStack(parent, { x: 340, y: laneBlockTop(b, true) + 30, count: grown ? 5 : 4, tile: 14, highlight: grown ? [4] : [], label: `KV · block ${b === 0 ? '1' : 'N'}` }));
}

// "on" travels from its probability cell to the fifth chip slot, marked as the followed token all the way.
function travellingChip(svg, travel, opacity) {
  const slot = { x: chipX(ON, WORDS5), y: CHIPS.y };
  const x = lerp(STRIP.x, slot.x, travel);
  const y = lerp(OUT.probs + 8, slot.y, travel);
  const g = layer(svg, opacity);
  G.token(g, { x, y, text: 'on', index: ON + 1 });
  G.selectionMark(g, { x, y, w: G.tokenWidth('on'), h: 24 });
  return slot;
}

// Frame 9: "on" is picked and appended; only its new row is computed, and each block's KV cache grows by one position.
export function drawFrame9(svg, p) {
  const out = leaving(p);
  const inn = arriving(p);
  chips(svg);
  followChip(svg, DOWN, TOKENS, out);
  const slot = travellingChip(svg, seg(p, 0, 0.35), inn);
  const grow = seg(p, 0.35, 0.5);
  const rows = [...rowsWith(SAT, X2_SAT), E[ON]];
  if (grow < 1) {
    stream(svg, rows.slice(0, ON), { selected: [DOWN], selOpacity: out });
    if (grow > 0) G.vector(layer(svg, grow), { x: STREAM.x, y: streamRowY(ON), values: E[ON], cell: SMALL, orient: 'row', maxAbs: MAX_ABS });
  } else stream(svg, rows, { selected: [ON], words: WORDS5 });
  lane(svg, { fold: true });
  kvStacks(layer(svg, seg(p, 0.35, 0.5)), p >= 0.65);
  const fade = 1 - seg(p, 0, 0.2);
  if (fade > 0) headPipeline(layer(svg, fade), { move: 1, lift: 1, through: 1, scores: 1 });
  outputRows(svg);
  G.selectionMark(layer(svg, inn), { x: STRIP.x, y: OUT.probs, w: CELL, h: CELL });
  const loop = seg(p, 0.8, 1);
  if (loop > 0) G.flow(svg, { from: [slot.x + 15, CHIPS.y + 26], to: [STREAM.x + 8 * SMALL + 4, streamRowY(ON) + SMALL / 2], carry: 'token', progress: loop });
}
