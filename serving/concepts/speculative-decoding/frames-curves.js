// speculative-decoding frames 7–9: tokens per round against guesses (one curve per acceptance rate), the speedup after the
// drafter's cost, and the speedup against batch size with the 128-user step bars. Values come from specdec.js sweeps.
import * as G from '@shared/glyphs.js';
import { expectedTokens, simpleSpeedup, batchSpeedup } from '@math/specdec.js';
import { formatDuration } from '@math/core.js';
import { ALPHA, ALPHAS, C, HEAD_USERS, K, K_RANGE, MODEL } from './numbers.js';
import { batchSeries, partialSeries, speedupSeries, stepParts, tokensSeries } from './sweeps.js';
import { probText, ratioText, tokensText } from './format.js';
import { BAR, arriving, layer, leaving, note, seg, stepBarAt, textLines } from './stage.js';
import { drawFrame6 } from './frames-verify.js';

const PLOT = Object.freeze({ x: 24, y: 8, w: 530 });
const MAX_USERS = 211; // the largest count that fits at 1,024 tokens of context (maxUsersPerGpu, pinned by the page test)
const K_AXIS = Object.freeze({ label: 'guesses per round k', domain: [1, 8], ticks: K_RANGE });
const FULL = 0.999; // a series counts as fully drawn from here: its label and markers appear

const kAxisY = (label) => ({ label, domain: [1, 7], ticks: [1, 2, 3, 4, 5, 6, 7] });
const ending = (t) => t >= FULL;

function tokensPlot(parent, t, { height = 250, onlySeven = false } = {}) {
  const alphas = onlySeven ? [ALPHA] : ALPHAS;
  const series = alphas.map((a) => ({
    points: partialSeries(tokensSeries(a), t), label: ending(t) ? (onlySeven ? 'tokens per round' : `α ${probText(a)}`) : '',
    style: a === ALPHA && !onlySeven ? 'solid' : onlySeven ? 'muted' : 'muted', tone: 'ink',
  }));
  const markers = !onlySeven && ending(t) ? [{ x: K, y: expectedTokens(ALPHA, K), label: tokensText(expectedTokens(ALPHA, K)), followed: true }] : [];
  G.curvePlot(parent, { ...PLOT, h: height, xAxis: K_AXIS, yAxis: kAxisY(onlySeven ? 'tokens per round · speedup' : 'tokens per round'), series, markers, label: 'tokens per round against guesses' });
}

const KS = [1, 3, 5, 8];
const at = (alpha, k) => `k ${k} → ${tokensText(expectedTokens(alpha, k))}`;
const LINE_7 = `α ${probText(ALPHA)}: ${KS.map((k) => at(ALPHA, k)).join(', ')} · α ${probText(0.9)}: ${at(0.9, K)}`;
const LINE_8 = `α ${probText(ALPHA)}, c ${probText(C)}, batch 1: ${KS.map((k) => `k ${k} → ${ratioText(simpleSpeedup(ALPHA, k, C))}`).join(', ')}`;
const GAP_LINES = ['two quantities on one y-axis: tokens per round (top curve) and', "speedup after drafting cost (lower curve); the gap is the drafter's cost"];

// Frame 7: the curves draw left to right; the α = 0.7, k = 3 point is framed.
export function drawFrame7(svg, p) {
  if (p < 0.15) drawFrame6(layer(svg, leaving(p)), 1);
  tokensPlot(layer(svg, arriving(p)), seg(p, 0.1, 0.8));
  note(layer(svg, seg(p, 0.8, 0.95)), PLOT.x, 292, LINE_7);
}

function speedupPlot(parent, t) {
  const tokens = [{ points: tokensSeries(ALPHA), label: 'tokens per round', style: 'muted', tone: 'ink' }];
  const speed = { points: partialSeries(speedupSeries(ALPHA, C), t), label: ending(t) ? `speedup, c ${probText(C)}` : '', style: 'solid', tone: 'ink' };
  const markers = ending(t) ? [{ x: K, y: simpleSpeedup(ALPHA, K, C), label: ratioText(simpleSpeedup(ALPHA, K, C)), followed: true }] : [];
  G.curvePlot(parent, { ...PLOT, h: 250, xAxis: K_AXIS, yAxis: kAxisY('tokens per round · speedup'), series: [...tokens, speed], markers, label: 'tokens per round and speedup against guesses' });
}

// Frame 8: the speedup curve draws under the token curve; the gap is the drafter's cost.
export function drawFrame8(svg, p) {
  if (p < 0.15) drawFrame7(layer(svg, leaving(p)), 1);
  speedupPlot(layer(svg, arriving(p)), seg(p, 0.1, 0.8));
  const t = seg(p, 0.8, 0.95);
  textLines(layer(svg, t), PLOT.x, 292, GAP_LINES);
  note(layer(svg, t), PLOT.x, 332, LINE_8);
}

const SPEEDUP_Y = Object.freeze({ label: 'speedup', domain: [1, 2.5], ticks: [1, 1.5, 2, 2.5] });
const MARKED = Object.freeze([64, HEAD_USERS, MAX_USERS]);
const speedupAt = (batch) => batchSpeedup({ alpha: ALPHA, k: K, c: C, batch, model: MODEL }).speedup;

const LABEL_RISE = 9; // a marker's value prints this far above its dot: the curve falls to the right, so above is clear

function batchPlot(parent, t) {
  const points = batchSeries({ alpha: ALPHA, k: K, c: C, max: MAX_USERS });
  const marked = ending(t) ? MARKED : [];
  const spec = {
    x: PLOT.x, y: PLOT.y, w: PLOT.w, h: 210, label: 'speedup against users in the batch',
    xAxis: { label: 'users in the batch', domain: [1, MAX_USERS], ticks: [1, 64, 128, MAX_USERS] }, yAxis: SPEEDUP_Y,
    series: [{ points: partialSeries(points, t), style: 'solid', tone: 'ink' }],
    markers: marked.map((u) => ({ x: u, y: speedupAt(u), followed: u === HEAD_USERS })),
  };
  G.curvePlot(parent, spec);
  const layout = G.curvePlotLayout(spec);
  layout.markers.forEach((m, i) => note(parent, PLOT.x + m.x, PLOT.y + m.y - LABEL_RISE, ratioText(speedupAt(marked[i])), { anchor: 'middle' }));
}

// Frame 9: the curve falls as the batch grows; at 128 users the verify pass is math-bound.
export function drawFrame9(svg, p) {
  if (p < 0.15) drawFrame8(layer(svg, leaving(p)), 1);
  batchPlot(layer(svg, arriving(p)), seg(p, 0.1, 0.7));
  const bars = layer(svg, seg(p, 0.7, 0.95));
  const plain = stepParts({ tokens: HEAD_USERS, seqs: HEAD_USERS });
  const verify = stepParts({ tokens: HEAD_USERS * (K + 1), seqs: HEAD_USERS });
  const row = { y: 252, titleY: 240 };
  stepBarAt(bars, { x: BAR.left, ...row, parts: plain, title: `plain step, ${HEAD_USERS} users: ${formatDuration(plain.timeS)}` });
  stepBarAt(bars, { x: BAR.right, ...row, parts: verify, title: `verify pass, ${HEAD_USERS * (K + 1)} tokens: ${formatDuration(verify.timeS)}` });
}
