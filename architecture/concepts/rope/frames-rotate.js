// Frames 1-5: one query and one key, turned pair by pair (storyboard §5). Each frame is a pure function of its
// progress p (0 -> 1); the end of frame n is the start of frame n + 1.
import * as G from '@shared/glyphs.js';
import { fmt1, fmt3, trimNumber } from './format.js';
import { FREQS, SCORE, PLAIN_SCORE, POS, Q_PAIRS, K_PAIRS } from './numbers.js';
import {
  STAGE, CELL, TOKENS, TOKEN_X, GHOST_X, TAG_Y, SCORE_ROW, NOTE_Y, READOUT, PROMPT_Y, DIAL_Y, DIAL_R, TEXT_Y, SCALE,
  seg, lerp, ease, fade, label, linked, numberRow, tokenRow, tagsRow, pairBlock, pairCenterX, strokeStyle,
} from './stage.js';

const SCORES_FOR_SAT = Object.freeze([-1, 3, 0.5, -Infinity]); // attention's row for "sat": the future key is masked
const SPEED_NOTE = 'speed: how many radians a hand turns per token';
const GHOST_LABEL = 'what if “cat” were here?';
const MID_X = (STAGE.w - 8) / 2; // centre of the stage, for notes that span both blocks

// The "cat" chip's ghost sits at position 9 once it has slid there (frame 1 slides it; frame 2 fades it).
const ghostAt = (t, opacity = 1) => ({ x: lerp(TOKEN_X[1], GHOST_X, t), index: 9, opacity: 0.6 * opacity });

// attention's score row for "sat", with its labels, as one group so a frame can fade it whole.
function scoreRow(svg, opacity = 1) {
  const g = G.svgEl('g', {}, svg);
  numberRow(g, { ...SCORE_ROW, values: SCORES_FOR_SAT, maxAbs: SCALE.score });
  label(g, SCORE_ROW.x - 10, SCORE_ROW.y + CELL / 2, 'q_sat · k:', { anchor: 'end' });
  TOKENS.forEach((t, j) => label(g, SCORE_ROW.x + j * CELL + CELL / 2, SCORE_ROW.y + CELL + 12, t, { anchor: 'middle' }));
  label(g, MID_X, NOTE_Y, 'q_sat · k_cat = 3.0, wherever “cat” sits', { anchor: 'middle' });
  return fade(g, opacity);
}

// Frame 1: attention cannot see order. The score row for "sat" is the one from the attention lesson.
export function drawFrame1(svg, p) {
  const slide = ease(seg(p, 0.1, 0.7));
  tokenRow(svg, { ghost: ghostAt(slide, seg(p, 0.1, 0.25)) });
  tagsRow(svg);
  label(svg, GHOST_X + G.tokenWidth(TOKENS[1]) / 2, TAG_Y, GHOST_LABEL, { anchor: 'middle', opacity: seg(p, 0.6, 0.85) });
  pairBlock(svg, { who: 'q', pos: 0, gap: 0, name: 'q_sat' });
  pairBlock(svg, { who: 'k', pos: 0, gap: 0, name: 'k_cat' });
  scoreRow(svg); // nothing in the row changes while the ghost moves
}

// Frame 2: the divider opens and each pair becomes a clock hand.
export function drawFrame2(svg, p) {
  const out = 1 - seg(p, 0, 0.2);
  tokenRow(svg, { ghost: ghostAt(1, out) });
  tagsRow(svg);
  if (out > 0) {
    label(svg, GHOST_X + G.tokenWidth(TOKENS[1]) / 2, TAG_Y, GHOST_LABEL, { anchor: 'middle', opacity: out });
    scoreRow(svg, out);
  }
  const gap = ease(seg(p, 0.15, 0.5));
  const dials = seg(p, 0.45, 0.8);
  pairBlock(svg, { who: 'q', pos: 0, gap, name: 'q_sat', parts: { dials } });
  pairBlock(svg, { who: 'k', pos: 0, gap, name: 'k_cat', parts: { dials } });
}

const nameAt = (who, pos) => (pos > 0 ? `${who === 'q' ? 'q_sat' : 'k_cat'} at ${Math.round(pos)}` : `${who === 'q' ? 'q_sat' : 'k_cat'}`);

// Frame 3: the query's hands turn by position x speed.
export function drawFrame3(svg, p) {
  const t = seg(p, 0, 0.75);
  const qPos = POS.q * t;
  tokenRow(svg);
  tagsRow(svg);
  pairBlock(svg, { who: 'q', pos: qPos, gap: 1, name: nameAt('q', qPos), parts: { dials: 1, speed: seg(p, 0, 0.3), angle: seg(p, 0.75, 1) }, angleAt: POS.q });
  pairBlock(svg, { who: 'k', pos: 0, gap: 1, name: 'k_cat', parts: { dials: 1 } });
  label(svg, MID_X, NOTE_Y, SPEED_NOTE, { anchor: 'middle', opacity: seg(p, 0.1, 0.4) });
}

// The readout of frames 4 and 5: the two pair dots, their sum, and the unrotated score in gray. `typed` = how many have arrived.
export function readout(svg, { typed, opacity = 1 }) {
  const box = { x: READOUT.label - 52, y: READOUT.y[0] - 10, w: 230, h: 56 };
  const g = linked(svg, 'score', box);
  const rows = [['pair 1', fmt3(SCORE.pairs[0])], ['pair 2', fmt3(SCORE.pairs[1])], ['score', fmt3(SCORE.score)]];
  rows.forEach(([name, value], i) => {
    const shown = seg(typed, i / 3, (i + 1) / 3);
    label(g, READOUT.label, READOUT.y[i], name, { anchor: 'end', opacity: Math.min(shown * 4, 1) * opacity });
    label(g, READOUT.value, READOUT.y[i], value, { cls: '', opacity: Math.min(shown * 4, 1) * opacity });
  });
  label(g, READOUT.value + 70, READOUT.y[2], `(was ${fmt1(PLAIN_SCORE)})`, { opacity: seg(typed, 0.85, 1) * opacity });
}

function flows(svg, t) {
  if (t <= 0 || t >= 1) return;
  const bottom = TEXT_Y[1] + 14;
  G.flow(svg, { from: [pairCenterX('q', 1) + 4, bottom], to: [READOUT.label - 30, READOUT.y[0] - 14], carry: 'activation', progress: t });
  G.flow(svg, { from: [pairCenterX('k', 0) - 4, bottom], to: [READOUT.value + 70, READOUT.y[0] - 14], carry: 'activation', progress: t });
}

// Frame 4: the key's hands turn, then the pair dots add up to the score.
export function drawFrame4(svg, p) {
  const kPos = POS.k * seg(p, 0, 0.45);
  tokenRow(svg);
  tagsRow(svg);
  pairBlock(svg, { who: 'q', pos: POS.q, gap: 1, name: nameAt('q', POS.q), parts: { dials: 1, speed: 1, angle: 1 } });
  pairBlock(svg, { who: 'k', pos: kPos, gap: 1, name: nameAt('k', kPos), parts: { dials: 1, speed: seg(p, 0, 0.2), angle: seg(p, 0.45, 0.55) }, angleAt: POS.k });
  label(svg, MID_X, NOTE_Y, SPEED_NOTE, { anchor: 'middle' });
  flows(svg, seg(p, 0.5, 0.75));
  readout(svg, { typed: seg(p, 0.55, 1) });
}

// Frame 5 draws, in each query dial, a thin line for where the key's hand points and the arc between the two hands.
function relativeArcs(svg, { qAngle, kAngle, opacity }) {
  [0, 1].forEach((i) => {
    const cx = pairCenterX('q', i);
    const baseQ = Math.atan2(Q_PAIRS[i][1], Q_PAIRS[i][0]);
    const baseK = Math.atan2(K_PAIRS[i][1], K_PAIRS[i][0]);
    const [aq, ak] = [baseQ + qAngle * FREQS[i], baseK + kAngle * FREQS[i]];
    const g = linked(svg, 'off', { x: cx - DIAL_R, y: DIAL_Y - DIAL_R, w: 2 * DIAL_R, h: 2 * DIAL_R });
    drawArc(g, { cx, cy: DIAL_Y, r: DIAL_R, aq, ak, magnitude: Math.hypot(...K_PAIRS[i]) / SCALE.vector });
    fade(g, opacity);
  });
}

const TURN = 2 * Math.PI;
function drawArc(g, { cx, cy, r, aq, ak, magnitude }) {
  const polar = (len, a) => [cx + len * Math.cos(a), cy - len * Math.sin(a)];
  const tip = polar(r * magnitude, ak);
  strokeStyle(G.svgEl('line', { x1: cx, y1: cy, x2: tip[0].toFixed(2), y2: tip[1].toFixed(2) }, g), { opacity: 0.7 });
  const diff = ((((aq - ak) % TURN) + 3 * Math.PI) % TURN) - Math.PI; // the signed angle from the k hand to the q hand, in (-pi, pi]
  const [from, to] = diff >= 0 ? [ak, aq] : [aq, ak];
  const rho = r * 0.55;
  const [a, b] = [polar(rho, from), polar(rho, to)];
  strokeStyle(G.svgEl('path', { d: `M${a[0].toFixed(2)} ${a[1].toFixed(2)}A${rho} ${rho} 0 0 0 ${b[0].toFixed(2)} ${b[1].toFixed(2)}`, fill: 'none' }, g), { width: 1.5 });
}

// Frame 5: both words move ten tokens later; every hand turns, the angle between each pair does not.
export function drawFrame5(svg, p) {
  const t = seg(p, 0.12, 0.7);
  const [qPos, kPos] = [POS.q + POS.shift * t, POS.k + POS.shift * t];
  const settled = p >= 0.7;
  const textOpacity = p < 0.12 ? 1 - seg(p, 0.04, 0.12) : seg(p, 0.7, 0.8);
  tokenRow(svg);
  tagsRow(svg);
  label(svg, 250, 20, '+10 tokens →', { opacity: seg(p, 0.12, 0.25) });
  label(svg, 250, TAG_Y, `sat at ${POS.q + POS.shift}, cat at ${POS.k + POS.shift}`, { opacity: seg(p, 0.7, 0.85) });
  const shown = settled ? POS.shift : 0;
  pairBlock(svg, { who: 'q', pos: qPos, gap: 1, name: nameAt('q', qPos), parts: { dials: 1, speed: 1, angle: textOpacity }, angleAt: POS.q + shown });
  pairBlock(svg, { who: 'k', pos: kPos, gap: 1, name: nameAt('k', kPos), parts: { dials: 1, speed: 1, angle: textOpacity }, angleAt: POS.k + shown });
  relativeArcs(svg, { qAngle: qPos, kAngle: kPos, opacity: seg(p, 0, 0.12) });
  label(svg, MID_X, NOTE_Y, SPEED_NOTE, { anchor: 'middle', opacity: 1 - seg(p, 0, 0.12) });
  label(svg, MID_X, NOTE_Y, 'thin line: where the key\'s hand points · arc: the angle between the two hands', { anchor: 'middle', opacity: seg(p, 0.1, 0.25) });
  readout(svg, { typed: 1 });
  label(svg, 14, PROMPT_Y[0], 'Will the score change when both words move ten tokens later?', { cls: '', opacity: seg(p, 0, 0.12) });
  label(svg, 14, PROMPT_Y[1], `No: the score is still ${fmt3(SCORE.score)}.`, { cls: '', opacity: seg(p, 0.8, 1) });
}
