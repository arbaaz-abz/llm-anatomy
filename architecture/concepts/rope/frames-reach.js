// Frames 6-10: the score by offset, the base, training length, the two stretches, and partial RoPE / NoPE (storyboard §5).
// Frames 6-9 share one offset row and two key dials (their positions never change); each frame is a pure function of p.
import * as G from '@shared/glyphs.js';
import { ropeScore, wavelengths } from '@math/rope.js';
import { fmt1, fmt3, trimNumber, TRAINED_LENGTH, Q_SAT } from './format.js';
import {
  FREQS, FREQS_10K, PI_FREQS, YARN_FREQS, WAVELENGTHS, ROW as ROWS, COVERAGE, POS, STRETCH_FACTOR, TRAINED_MAX_OFFSET, READ_MAX_OFFSET, K_CAT,
  Q_PAIRS,
} from './numbers.js';
import {
  CELL, ROW, BOTTOM_X, REACH_DIAL, REACH_TEXT_Y, BOTTOM_Y, DIAL_R, SCALE, seg, lerp, ease, fade, label, linked, tokenRow, offsetRow, reachDial,
  revealed, blendRows,
} from './stage.js';
import { drawFrame5 } from './frames-rotate.js';

const ROW_TITLE = 'score of q_sat with k_cat, as the two tokens move apart';
const TRAINED_TITLE = `trained on ${TRAINED_LENGTH} tokens, now reading ${POS.readTo}`;
const MID = REACH_DIAL.x;
const GHOST_UNTIL = 0.75;
const SEEN = [[0, COVERAGE.plain[0].seenMax], [0, COVERAGE.plain[1].seenMax]]; // pair 1 turned fully, pair 2 up to 1.5 rad

// The previous frame's end state, fading out over the first part of this one (a layout change is never a hard cut).
function outgoing(svg, drawPrevious, opacity) {
  if (opacity <= 0) return;
  const g = G.svgEl('g', {}, svg);
  drawPrevious(g, 1);
  fade(g, opacity);
}

// Frame 6: the score by offset. Each hand has turned by offset x speed; the fast one finishes a turn after offset 6.
export function drawFrame6(svg, p) {
  outgoing(svg, drawFrame5, 1 - seg(p, 0, 0.3));
  tokenRow(svg);
  const t = seg(p, 0.15, 1);
  const offset = (ROW.n - 1) * t;
  offsetRow(svg, { values: revealed(ROWS.base100, offset), title: ROW_TITLE, opacity: seg(p, 0.1, 0.3) });
  [0, 1].forEach((i) => {
    reachDial(svg, i, { angle: FREQS[i] * offset, seen: [0, FREQS[i] * offset], opacity: seg(p, 0.1, 0.3) });
    label(svg, MID[i], REACH_TEXT_Y[0], `wavelength ${fmt1(WAVELENGTHS[i])} tokens`, { anchor: 'middle', opacity: seg(p, 0.7, 1) });
  });
  label(svg, BOTTOM_X, BOTTOM_Y[0], 'each hand has turned by offset × speed', { opacity: seg(p, 0.7, 1) });
}

const BASE_BOX = { x: 70, y: 196, w: 220, h: 64 };
// The base readout of frame 7: "base 100" steps to "base 10,000", and the slow hand's speed follows.
function baseReadout(svg, { stepped, opacity }) {
  const g = linked(svg, 'base', BASE_BOX);
  const speed = stepped ? FREQS_10K[1] : FREQS[1];
  label(g, BASE_BOX.x + 8, BASE_BOX.y + 12, 'base', { opacity });
  label(g, BASE_BOX.x + 48, BASE_BOX.y + 12, stepped ? '10,000' : '100', { cls: '', opacity });
  label(g, BASE_BOX.x + 8, BASE_BOX.y + 36, `slow hand: ${trimNumber(speed)} per token`, { opacity });
}

// Frame 7: a bigger base slows the slowest hand and leaves the row almost unchanged.
export function drawFrame7(svg, p) {
  const t = ease(seg(p, 0.2, 1));
  const stepped = t >= 0.5;
  tokenRow(svg);
  offsetRow(svg, { values: blendRows(ROWS.base100, ROWS.base10000, t), title: ROW_TITLE });
  const offset = ROW.n - 1;
  const slow = lerp(FREQS[1], FREQS_10K[1], t) * offset;
  reachDial(svg, 0, { angle: FREQS[0] * offset, seen: [0, FREQS[0] * offset], opacity: 1 - seg(p, 0, 0.3) });
  reachDial(svg, 1, { angle: slow, seen: [0, slow] });
  label(svg, MID[0], REACH_TEXT_Y[0], `wavelength ${fmt1(WAVELENGTHS[0])} tokens`, { anchor: 'middle', opacity: 1 - seg(p, 0, 0.3) });
  label(svg, MID[1], REACH_TEXT_Y[0], `wavelength ${fmt1(wavelengths([stepped ? FREQS_10K[1] : FREQS[1]])[0])} tokens`, { anchor: 'middle' });
  baseReadout(svg, { stepped, opacity: seg(p, 0.1, 0.4) });
  label(svg, BOTTOM_X, BOTTOM_Y[0], 'a bigger base slows the slowest hand; the fast one is unchanged', { opacity: seg(p, 0.7, 1) });
}

const speedText = (i, freqs) => `speed ${trimNumber(freqs[i])} per token`;
function speedLines(svg, i, { from, to, t, opacity = 1 }) {
  label(svg, MID[i], REACH_TEXT_Y[1], speedText(i, from), { anchor: 'middle', opacity: (1 - seg(t, 0.4, 0.6)) * opacity });
  label(svg, MID[i], REACH_TEXT_Y[1], speedText(i, to), { anchor: 'middle', opacity: seg(t, 0.4, 0.6) * opacity });
}

function trainedLines(svg) {
  label(svg, MID[0], REACH_TEXT_Y[0], 'seen: a full turn', { anchor: 'middle' });
  label(svg, MID[1], REACH_TEXT_Y[0], `seen: up to ${trimNumber(SEEN[1][1])} rad`, { anchor: 'middle' });
}

// Frame 8: past the trained length the slow hand reaches angles it never saw; position interpolation pulls it back.
export function drawFrame8(svg, p) {
  const back = ease(seg(p, 0, 0.15));
  const swing = seg(p, 0.15, 0.5);
  const squeeze = ease(seg(p, 0.55, 1));
  const offset = swing > 0 ? lerp(TRAINED_MAX_OFFSET, READ_MAX_OFFSET, swing) : lerp(ROW.n - 1, TRAINED_MAX_OFFSET, back);
  const unsqueezed = [FREQS[0], lerp(FREQS_10K[1], FREQS[1], back)]; // the speeds before the squeeze: the ghost hand's
  const speeds = [lerp(FREQS[0], PI_FREQS[0], squeeze), lerp(unsqueezed[1], PI_FREQS[1], squeeze)];
  tokenRow(svg);
  offsetRow(svg, { values: blendRows(blendRows(ROWS.base10000, ROWS.base100, back), ROWS.pi, squeeze), title: TRAINED_TITLE });
  // The ghost hand (where the hand would point without the squeeze) goes once the squeeze is well under way, so the
  // rest frame shows only the hand the caption talks about.
  const ghost = (i) => (p < GHOST_UNTIL ? [0, unsqueezed[i] * offset] : null);
  reachDial(svg, 0, { angle: speeds[0] * offset, seen: SEEN[0], reached: ghost(0), opacity: seg(p, 0, 0.2) });
  reachDial(svg, 1, { angle: speeds[1] * offset, seen: SEEN[1], reached: ghost(1) });
  trainedLines(svg);
  speedLines(svg, 0, { from: FREQS, to: PI_FREQS, t: squeeze, opacity: seg(p, 0.15, 0.3) });
  speedLines(svg, 1, { from: FREQS, to: PI_FREQS, t: squeeze, opacity: seg(p, 0.15, 0.3) });
  const reach = (n) => trimNumber(n, n < 10 ? 3 : 1);
  label(svg, BOTTOM_X, BOTTOM_Y[0], `at ${POS.readTo} tokens: pair 2 reaches ${reach(COVERAGE.plain[1].reachedMax)} rad, outside what it saw`, { opacity: seg(p, 0.2, 0.35) * (1 - seg(p, 0.5, 0.6)) });
  label(svg, BOTTOM_X, BOTTOM_Y[0], `position interpolation, ÷ ${STRETCH_FACTOR}: every speed drops to a quarter`, { opacity: seg(p, 0.6, 0.75) });
  label(svg, BOTTOM_X, BOTTOM_Y[1], `pair 2 now reaches ${reach(COVERAGE.pi[1].reachedMax)} rad, one token's step past the ${trimNumber(COVERAGE.pi[1].seenMax)} it saw`, { opacity: seg(p, 0.8, 1) });
}

// Frame 9: YaRN-style squeezes only the slow pair; the fast hand returns to full speed.
export function drawFrame9(svg, p) {
  const v = ease(seg(p, 0.1, 0.6));
  const offset = READ_MAX_OFFSET;
  tokenRow(svg);
  offsetRow(svg, { values: blendRows(ROWS.pi, ROWS.yarn, v), title: TRAINED_TITLE });
  label(svg, ROW.x + ROW.n * CELL, ROW.y + CELL + 34, 'branch: YaRN-style', { anchor: 'end', opacity: seg(p, 0.05, 0.25) });
  reachDial(svg, 0, { angle: lerp(PI_FREQS[0], YARN_FREQS[0], v) * offset, seen: SEEN[0] });
  reachDial(svg, 1, { angle: PI_FREQS[1] * offset, seen: SEEN[1] });
  trainedLines(svg);
  speedLines(svg, 0, { from: PI_FREQS, to: YARN_FREQS, t: v });
  label(svg, MID[1], REACH_TEXT_Y[1], speedText(1, PI_FREQS), { anchor: 'middle' });
  label(svg, BOTTOM_X, BOTTOM_Y[0], 'YaRN-style: only the slow hand is squeezed; the fast hand returns to full speed', { opacity: seg(p, 0.2, 0.4) });
  label(svg, BOTTOM_X, BOTTOM_Y[1], `cost of squeezing all: “cat” must now be ${STRETCH_FACTOR} tokens back to score what 1 back did`, { opacity: seg(p, 0.6, 0.85) });
}

// ---- frame 10: partial RoPE and NoPE ----
const STRIPS = Object.freeze([
  { y: 112, name: 'partial RoPE', note: 'only pair 1 turns', freqs: [FREQS[0], 0] },
  { y: 252, name: 'NoPE', note: 'no pair turns', freqs: [0, 0] },
]);
const DIAL_X = Object.freeze([230, 340]);
const SCORE_X = 410;
const unwind = (t, angle) => lerp(angle, 0, ease(t));
const gray = (t) => lerp(1, 0.4, ease(t));

function strip(svg, p, { y, name, note, freqs }) {
  const turns = freqs.map((f) => f > 0);
  const timings = [seg(p, 0, 0.45), seg(p, 0.45, 0.9)]; // pair 2 stops first, then pair 1
  label(svg, 14, y - 6, name, { cls: '' });
  label(svg, 14, y + 12, note, { opacity: seg(p, 0.3, 0.5) });
  [0, 1].forEach((i) => {
    const t = turns[i] ? 0 : timings[1 - i];
    fade(G.dial(svg, { x: DIAL_X[i], y, r: DIAL_R, vector: Q_PAIRS[i], angle: unwind(t, POS.q * FREQS[i]), scale: SCALE.vector, label: `pair ${i + 1}` }), gray(t));
    if (!turns[i]) label(svg, DIAL_X[i], y + DIAL_R + 30, 'position-free', { anchor: 'middle', opacity: seg(t, 0.6, 1) });
  });
  const { score } = ropeScore(Q_SAT, K_CAT, { qPos: POS.q, kPos: POS.k, freqs });
  const text = turns.some(Boolean) ? `score at ${POS.q} and ${POS.k}: ${fmt3(score)}` : `score = ${fmt1(score)} again`;
  label(svg, SCORE_X, y - 4, text, { opacity: seg(p, 0.8, 1) });
}

export function drawFrame10(svg, p) {
  outgoing(svg, drawFrame9, 1 - seg(p, 0, 0.2));
  tokenRow(svg);
  STRIPS.forEach((s) => strip(svg, p, s));
}
