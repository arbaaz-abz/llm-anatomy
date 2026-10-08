// pretraining frames 4–5: one pass grades every position under the causal mask; the mean loss and its perplexity.
import * as G from '@shared/glyphs.js';
import { tokenLoss, meanLoss, perplexity, uniformLoss } from '@math/lm.js';
import { TOKENS, STAND_INS, UNIFORM_VOCAB, FOLLOWED } from './numbers.js';
import { formatProb, formatLoss, formatPerplexity } from './format.js';
import { CELL, STRIP, DIM, seg, lerp, arriving, leaving, typed, slotX, slotCenter, cellX, layer, note, chips, lossCell, linked, select } from './stage.js';
import { trueNext, lossReadout, TARGET } from './frames-tokens.js';

const N = TOKENS.length; // 8 positions, 7 graded predictions
const LOSSES = Object.freeze(STAND_INS.map(tokenLoss));
const HEAT = Object.freeze({ x: 35, y: STRIP.y, cell: 16 });
const ROW_T = (r) => 0.15 + r * 0.1; // when mask row r (0-based) lights
const VISIBLE = Object.freeze(Array.from({ length: N }, () => Object.freeze(Array(N).fill(0)))); // the mask adds 0 where a position can see
const MASK = Object.freeze(Array.from({ length: N }, (_, i) => Object.freeze(Array.from({ length: N }, (__, j) => j <= i))));
const maskText = (v) => (v === -Infinity ? 'future' : '0: can see');
const MEAN = Object.freeze({ x: cellX(1), y: 200 });
const LABEL_X = slotX(N) + 6; // the strip's row labels, right of the last slot

// The causal mask: every row dim, or one row lit (its cells see the past and itself; the future is hatched).
function maskGrid(parent, { row = null } = {}) {
  const rows = row == null ? [0, N] : [row, row + 1];
  G.heatmap(parent, { x: HEAT.x, y: HEAT.y + rows[0] * HEAT.cell, values: VISIBLE.slice(...rows), mask: MASK.slice(...rows), cell: HEAT.cell, maxAbs: 1, format: maskText });
}

// The mask's title, row numbers and key: plain marks at full strength while its rows are still dim.
function maskNote(parent) {
  note(parent, HEAT.x, HEAT.y - 10, `causal mask [${N} × ${N}]`);
  TOKENS.forEach((_, i) => note(parent, HEAT.x - 8, HEAT.y + i * HEAT.cell + 12, String(i + 1), { anchor: 'end' }));
  note(parent, HEAT.x - 15, HEAT.y + N * HEAT.cell + 20, 'row = position; hatched = the future');
}

// The p printed above each target's loss cell, and the strip's two row labels.
function stripLabels(parent, shown) {
  STAND_INS.forEach((p, r) => {
    const t = shown(r);
    if (t > 0) note(layer(parent, t), slotCenter(r + 1), STRIP.pY, formatProb(p), { anchor: 'middle' });
  });
}
const rowLabels = (parent) => {
  note(parent, LABEL_X, STRIP.pY, 'p');
  note(parent, LABEL_X, STRIP.y + CELL / 2 + 4, 'loss');
};

// The loss strip under target chips 2–8; shown(r) types each cell in (with a short drop) from 0 to 1.
function lossStrip(parent, shown = () => 1) {
  const g = linked(parent, 'p', { x: cellX(1), y: STRIP.y, w: cellX(N - 1) + CELL - cellX(1), h: CELL });
  LOSSES.forEach((loss, r) => {
    const t = shown(r);
    if (t > 0) lossCell(layer(g, t), { x: cellX(r + 1), y: STRIP.y - 10 * (1 - t), loss });
  });
  stripLabels(parent, shown);
  const mark = shown(FOLLOWED - 1);
  select(parent, { x: cellX(TARGET), y: STRIP.y, w: CELL, h: CELL }, mark);
}

// Frame 4: the mask's rows light one after another; each lit row drops its loss cell under the token it predicted.
export function drawFrame4(svg, p) {
  const out = leaving(p);
  const inn = arriving(p);
  chips(svg, { opacity: (i) => (i < 4 ? 1 : lerp(DIM, 1, inn)), follow: [TARGET] });
  if (out > 0) {
    const old = layer(svg, out);
    trueNext(old);
    lossReadout(old);
  }
  maskGrid(layer(svg, 0.3 * inn));
  maskNote(layer(svg, inn));
  rowLabels(layer(svg, inn));
  for (let r = 0; r < N; r += 1) {
    const t = seg(p, ROW_T(r), ROW_T(r) + 0.05);
    if (t > 0) maskGrid(layer(svg, t), { row: r });
  }
  select(svg, { x: HEAT.x, y: HEAT.y + (FOLLOWED - 1) * HEAT.cell, w: N * HEAT.cell, h: HEAT.cell }, seg(p, ROW_T(FOLLOWED - 1), ROW_T(FOLLOWED - 1) + 0.05));
  lossStrip(svg, (r) => seg(p, ROW_T(r) + 0.02, ROW_T(r) + 0.08));
}

const meanLine = () => {
  const sum = LOSSES.reduce((s, l) => s + l, 0);
  return `mean = ${sum.toFixed(3)} / ${LOSSES.length} = ${formatLoss(meanLoss(STAND_INS))}`;
};
const pplLine = () => `perplexity = e^${formatLoss(meanLoss(STAND_INS))} = ${formatPerplexity(perplexity(meanLoss(STAND_INS)))}`;
const uniformLine = () => `uniform guess over ${UNIFORM_VOCAB} words: ln ${UNIFORM_VOCAB} = ${formatLoss(uniformLoss(UNIFORM_VOCAB))} → perplexity ${UNIFORM_VOCAB}`;

// The mean readout: the mean as a loss cell, its line, and the perplexity line typed to `pplT`.
function meanReadout(parent, { meanT = 1, pplT = 1, uniformT = 1 } = {}) {
  if (meanT > 0) {
    const g = layer(parent, meanT);
    lossCell(g, { x: MEAN.x, y: MEAN.y, loss: meanLoss(STAND_INS) });
    note(g, MEAN.x + CELL + 12, MEAN.y + 16, meanLine(), { cls: '' });
  }
  if (pplT > 0) note(parent, MEAN.x + CELL + 12, MEAN.y + 34, typed(pplLine(), pplT), { cls: '' });
  if (uniformT > 0) note(layer(parent, uniformT), MEAN.x, MEAN.y + CELL + 30, uniformLine());
}

// Frame 5's end state (also what frame 6 fades out).
export function lossScene(parent) {
  chips(parent, { follow: [TARGET] });
  rowLabels(parent);
  lossStrip(parent);
  meanReadout(parent);
}

// Frame 5 (key frame): copies of the seven cells slide together into the mean; the perplexity types in.
export function drawFrame5(svg, p) {
  const out = leaving(p);
  chips(svg, { follow: [TARGET] });
  rowLabels(svg);
  if (out > 0) {
    const old = layer(svg, out);
    maskGrid(layer(old, 0.3));
    for (let r = 0; r < N; r += 1) maskGrid(old, { row: r });
    maskNote(old);
    select(old, { x: HEAT.x, y: HEAT.y + (FOLLOWED - 1) * HEAT.cell, w: N * HEAT.cell, h: HEAT.cell });
  }
  lossStrip(svg);
  const slide = seg(p, 0.15, 0.5);
  const fade = 1 - seg(p, 0.4, 0.5);
  if (slide > 0 && fade > 0) {
    const g = layer(svg, fade);
    LOSSES.forEach((loss, r) => lossCell(g, { x: lerp(cellX(r + 1), MEAN.x, slide), y: lerp(STRIP.y, MEAN.y, slide), loss }));
  }
  meanReadout(svg, { meanT: seg(p, 0.45, 0.55), pplT: seg(p, 0.6, 0.8), uniformT: seg(p, 0.8, 0.95) });
}

