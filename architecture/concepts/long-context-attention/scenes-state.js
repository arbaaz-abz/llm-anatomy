// Frame 8: the linear-attention state. Three tokens feed a 4 × 4 state S; the query "sat" reads it with one multiplication.
// All numbers come from math/longctx.js (the state) and math/attention.js (the softmax row it is compared with).
import * as G from '@shared/glyphs.js';
import { TOY, attentionHead } from '@math/attention.js';
import { linearState, linearRead } from '@math/longctx.js';
import { STATE_SIZE, TOKENS } from './numbers.js';
import { trimNumber, formatFor } from './format.js';
import { scene, label, lines, select, linked, glyphValues, lerp, ease, seg, fade } from './stage.js';

const HEAD = TOY.heads.A;
const FED = 3; // The, cat, sat feed the state
const QUERY = 2; // "sat"
const KEYS = HEAD.K.slice(0, FED);
const VALUES = HEAD.V.slice(0, FED);
export const Q_SAT = HEAD.Q[QUERY];
export const STATES = Object.freeze(Array.from({ length: FED + 1 }, (_, t) => linearState({ keys: KEYS.slice(0, t), values: VALUES.slice(0, t), ...STATE_SIZE })));
export const OUTPUT = linearRead(STATES[FED], Q_SAT);
const SOFTMAX_ROW = attentionHead(HEAD.Q, HEAD.K, HEAD.V).output[QUERY];
const RAW_SCORES = attentionHead(HEAD.Q, HEAD.K, HEAD.V).scores[QUERY].slice(0, FED);
export const SCALE = 6; // storyboard §4: S and the output share one value scale

const CELL = G.NUMBER_CELL;
const S_AT = Object.freeze({ x: 150, y: 64 });
const Q_AT = Object.freeze({ x: 352, y: 64 });
const O_AT = Object.freeze({ x: 430, y: 64 });
const CHIP_X = 14;
const CHIP_Y = Object.freeze([72, 122, 172]);
const SEGMENT = 0.2; // each fed token owns this much of the frame, after the fade-in
const START = 0.12;

const typeIn = (row, t) => row.map((v, j) => (t * row.length >= j + 1 - 1e-9 ? v : null));

// The state after `p`: token t flies in during the first half of its segment, then S moves from state t−1 to state t.
export function stateAt(p) {
  const done = Math.min(Math.max(Math.floor((p - START) / SEGMENT + 1e-9), 0), FED);
  if (done >= FED) return { values: STATES[FED], flying: null };
  const local = (p - START - done * SEGMENT) / SEGMENT;
  if (p < START) return { values: STATES[0], flying: null };
  const move = ease(seg(local, 0.5, 1));
  return {
    values: STATES[done].map((row, i) => row.map((v, j) => lerp(v, STATES[done + 1][i][j], move))),
    flying: local < 0.5 ? { token: done, progress: local * 2 } : null,
  };
}

export function stateScene(parent, p, opacity = 1) {
  const holder = scene(parent, opacity);
  const { values, flying } = stateAt(p);
  TOY.tokens.slice(0, FED).forEach((name, i) => G.token(holder, { x: CHIP_X, y: CHIP_Y[i], text: name, index: i + 1 }));
  select(holder, CHIP_X, CHIP_Y[QUERY], G.tokenWidth(TOY.tokens[QUERY]), 24);
  const side = linked(holder, 'state', { x: S_AT.x, y: S_AT.y, w: 4 * CELL, h: 4 * CELL });
  G.matrix(side, { ...S_AT, values: values.map(glyphValues), cell: CELL, maxAbs: SCALE, label: 'state S', format: formatFor('state') });
  if (flying) {
    const from = [CHIP_X + G.tokenWidth(TOY.tokens[flying.token]) + 4, CHIP_Y[flying.token] + 12];
    G.flow(holder, { from, to: [S_AT.x - 6, S_AT.y + 2 * CELL], carry: 'activation', progress: flying.progress });
  }
  readStep(holder, p);
  lines(holder, CHIP_X, 262, [`cache after ${TOKENS} tokens: ${TOKENS * (STATE_SIZE.dKey + STATE_SIZE.dValue)} numbers · state: ${STATE_SIZE.dKey * STATE_SIZE.dValue}, always`]);
  lines(holder, CHIP_X, 300, noSoftmaxNote());
  return holder;
}

// "(−1, 3 and 0.5 times each value)": the raw scores the toy blends by, with no softmax.
function noSoftmaxNote() {
  const [a, b, c] = RAW_SCORES.map((s) => trimNumber(s));
  return ['No softmax here: the output is a raw-score blend', `(${a}, ${b} and ${c} times each value), so compare the shape`, 'with softmax\'s row, not the size.'];
}

// The query "sat" appears, then S · q fills the output.
function readStep(holder, p) {
  const show = seg(p, 0.72, 0.8);
  if (show <= 0) return;
  const q = G.vector(holder, { ...Q_AT, values: Q_SAT, cell: CELL, orient: 'col', maxAbs: SCALE, label: 'q_sat', format: formatFor('state') });
  fade(q, show);
  select(holder, Q_AT.x, Q_AT.y, CELL, 4 * CELL, show);
  label(holder, S_AT.x + 4 * CELL + 10, S_AT.y + 2 * CELL, '×', { opacity: show });
  label(holder, Q_AT.x + CELL + 14, S_AT.y + 2 * CELL, '=', { opacity: show });
  const out = typeIn(OUTPUT, seg(p, 0.8, 1));
  G.vector(holder, { ...O_AT, values: glyphValues(out), cell: CELL, orient: 'col', maxAbs: SCALE, label: 'o_sat', format: formatFor('output') });
  select(holder, O_AT.x, O_AT.y, CELL, 4 * CELL);
  lines(holder, CHIP_X, 282, [`softmax attention gave [${SOFTMAX_ROW.map((v) => trimNumber(v, 3)).join(', ')}]`], seg(p, 0.95, 1));
}
