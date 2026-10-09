// disaggregation frames 1–3: shared GPUs (batching's branch, in ms), the two service targets, then two pools.
// Each draw is a pure function of progress p; frame N at p = 1 is frame N + 1 at p = 0 (scene fades the handoff).
import * as G from '@shared/glyphs.js';
import { formatDuration } from '@math/core.js';
import { BRANCH_PROMPT, CHUNK_BUDGET, GOODPUT_BUDGETS, TTFT_TARGET_MS, TPOT_TARGET_MS } from './numbers.js';
import { branchSchedule, laneOf, colocatedMetrics, separatedTimes, SHOWN } from './colocated.js';
import { LEFT, LINE, seg, lerp, layer, note, lines, connector, scene, drawLane, laneExtent } from './stage.js';

const ms = (value) => formatDuration(value / 1000);
const BRANCHES = Object.freeze([{ budget: Infinity, label: 'no chunks' }, { budget: CHUNK_BUDGET, label: `chunks of ${CHUNK_BUDGET}` }]);
const SCHEDULES = Object.freeze(BRANCHES.map((b) => branchSchedule(b.budget)));
const END_MS = Math.max(...SCHEDULES.flatMap((s) => SHOWN.map((id) => laneOf(s, id).steps.at(-1).toMs)));
const AXIS = Object.freeze({ x: 40, w: 520 });
const PX_PER_MS = AXIS.w / END_MS;
const LANE_PITCH = 16;
const PANELS = Object.freeze([{ title: 92, lanes: 104 }, { title: 198, lanes: 210 }]);
const LANE_H = 10;

function gpuHeader(svg) {
  G.gpu(svg, { x: LEFT, y: 4, w: 64, h: 56, label: 'one GPU', showMem: false });
  note(svg, 88, 20, 'from Continuous batching\'s branch:', { cls: '' });
  note(svg, 88, 36, `D\'s ${BRANCH_PROMPT.toLocaleString('en-US')}-token prompt joins at step 3, seats = 3`);
}

function panel(svg, index, upToMs) {
  const { title, lanes } = PANELS[index];
  note(svg, AXIS.x - 32, title, BRANCHES[index].label, { cls: '' });
  SHOWN.forEach((id, i) => {
    const lane = laneOf(SCHEDULES[index], id);
    const y = lanes + i * LANE_PITCH;
    drawLane(svg, { x: AXIS.x, y, id, lane, pxPerMs: PX_PER_MS, upToMs });
    const extent = id === 'D' ? laneExtent(lane, PX_PER_MS, upToMs) : null;
    if (extent) G.selectionMark(svg, { x: AXIS.x + extent.from, y, w: Math.max(extent.to - extent.from, 4), h: LANE_H });
  });
}

const isFullSlice = (s) => s.prefill.some((q) => q.id === 'D') && s.tokens === CHUNK_BUDGET;
const FULL_SLICES = Object.freeze(SCHEDULES[1].filter(isFullSlice));
const seated = (id) => FULL_SLICES.filter((s) => s.decode.includes(id)).length; // full-size steps a running request sits through
const WORDS = Object.freeze({ 2: 'two', 4: 'four', 8: 'eight' });
const LONG_STEP = SCHEDULES[0].find((s) => s.prefill.some((q) => q.id === 'D'));
const CHUNKED_END = FULL_SLICES.at(-1).fromMs + FULL_SLICES.at(-1).ms;

// The labels under each panel appear once the step (or the run of steps) they name has been drawn.
function stallLabels(svg, upToMs) {
  const under = (i) => PANELS[i].lanes + 3 * LANE_PITCH + 12;
  if (upToMs[0] >= LONG_STEP.fromMs + LONG_STEP.ms) note(svg, AXIS.x + (LONG_STEP.fromMs + LONG_STEP.ms / 2) * PX_PER_MS, under(0), `one ${ms(LONG_STEP.ms)} step`, { cls: '', anchor: 'middle' });
  if (upToMs[1] >= CHUNKED_END) note(svg, AXIS.x + ((FULL_SLICES[0].fromMs + CHUNKED_END) / 2) * PX_PER_MS, under(1), `${WORDS[FULL_SLICES.length]} steps of ${ms(FULL_SLICES[0].ms)}`, { cls: '', anchor: 'middle' });
}

export function drawFrame1(svg, p) {
  gpuHeader(svg);
  const upTo = [seg(p, 0.05, 0.5), seg(p, 0.5, 0.95)].map((t) => t * END_MS);
  panel(svg, 0, upTo[0]);
  panel(svg, 1, p > 0.5 ? upTo[1] : 0);
  stallLabels(svg, upTo);
  const numbers = layer(svg, seg(p, 0.9, 1));
  lines(numbers, LEFT, 286, [
    `no chunks: one ${ms(LONG_STEP.ms)} step`,
    `chunks of ${CHUNK_BUDGET}: D's prompt takes ${WORDS[FULL_SLICES.length]} steps of ${ms(FULL_SLICES[0].ms)};`,
    `A sits through ${WORDS[seated('A')]} of them, C through ${WORDS[seated('C')]}`,
  ]);
  note(svg, LEFT, 352, 'block: prefill · tick: one token · grey: waiting for a seat · time runs left to right');
}

// Frame 2's gauges: time to first token and time per token against illustrative targets, one verdict per request.
const GAUGE = Object.freeze({ x: 40, w: 280, ttft: { y: 62, hi: 600 }, tpot: { y: 150, hi: 100 } });
const TABLE = Object.freeze({ x: 364, y: 30, col: 76, row: 28 });

function stopAt(p) {
  const t = p * (GOODPUT_BUDGETS.length - 1);
  const i = Math.min(Math.floor(t), GOODPUT_BUDGETS.length - 2);
  const move = seg(t - i, 0.55, 1);
  return { from: i, to: i + 1, move, shown: move >= 0.5 ? i + 1 : i };
}

function gauges(svg, stop) {
  const [a, b] = [colocatedMetrics(GOODPUT_BUDGETS[stop.from]), colocatedMetrics(GOODPUT_BUDGETS[stop.to])];
  G.clipLine(svg, { x: GAUGE.x, y: GAUGE.ttft.y, w: GAUGE.w, lo: 0, hi: GAUGE.ttft.hi, band: [0, TTFT_TARGET_MS], marker: lerp(a.worstTtftMs, b.worstTtftMs, stop.move) });
  note(svg, GAUGE.x, GAUGE.ttft.y - 34, 'time to first token, slowest request (ms)', { cls: '' });
  G.clipLine(svg, { x: GAUGE.x, y: GAUGE.tpot.y, w: GAUGE.w, lo: 0, hi: GAUGE.tpot.hi, band: [0, TPOT_TARGET_MS], marker: lerp(a.worstTpotMs, b.worstTpotMs, stop.move) });
  note(svg, GAUGE.x, GAUGE.tpot.y - 34, 'time per token, slowest request (ms)', { cls: '' });
  note(svg, GAUGE.x, GAUGE.ttft.y + 36, `target: at most ${TTFT_TARGET_MS} ms (illustrative)`);
  note(svg, GAUGE.x, GAUGE.tpot.y + 36, `target: at most ${TPOT_TARGET_MS} ms (illustrative)`);
}

function verdictTable(svg, metrics) {
  ['first token', 'per token', 'both'].forEach((h, c) => note(svg, TABLE.x + 34 + c * TABLE.col, TABLE.y, h, { anchor: 'middle' }));
  metrics.rows.forEach((r, i) => {
    const y = TABLE.y + 14 + i * TABLE.row;
    G.token(svg, { x: TABLE.x - 12, y, text: r.id, owner: r.id });
    [r.okTtft, r.okTpot, r.okTtft && r.okTpot].forEach((ok, c) => G.verdict(svg, { x: TABLE.x + 34 + c * TABLE.col, y: y + 12, ok }));
  });
}

function budgetChips(svg, shown) {
  note(svg, GAUGE.x, 232, 'chunk size: token budget per step', { cls: '' });
  GOODPUT_BUDGETS.forEach((budget, i) => {
    const label = budget === Infinity ? 'no chunks' : String(budget);
    G.block(svg, { x: GAUGE.x + i * 84, y: 242, w: 76, h: 28, label, state: i === shown ? 'active' : 'idle' });
  });
}

export function drawFrame2(svg, p) {
  scene(svg, p, (g) => drawFrame1(g, 1), (g) => {
    const stop = stopAt(p);
    const metrics = colocatedMetrics(GOODPUT_BUDGETS[stop.shown]);
    gauges(g, stop);
    verdictTable(g, metrics);
    budgetChips(g, stop.shown);
    note(g, LEFT, 306, `goodput: ${metrics.meeting} of ${metrics.total} requests meet both targets`, { cls: '' });
    note(g, LEFT, 322, 'goodput = requests per second that meet both (DistServe\'s definition)');
  });
}

// Frame 3: D's prefill on its own GPU, on the same time axis as A and C's steady decode steps.
export const POOLS = Object.freeze({ x0: 150, stepPx: 12, prefillY: 56, decodeY: { A: 160, C: 176, D: 192 }, gpuP: { x: LEFT, y: 24, w: 96, h: 64 }, gpuD: { x: LEFT, y: 134, w: 96, h: 64 } });
const TIMES = separatedTimes();
export const POOL_PX_PER_MS = POOLS.stepPx / TIMES.decodeMs;
export const PREFILL_PX = TIMES.prefillMs * POOL_PX_PER_MS;
const DECODE_STEPS = Object.freeze({ A: 2, C: 4 }); // what A and C have left at step 3 (batching's branch)

function poolRows(svg, upToMs) {
  G.gpu(svg, { ...POOLS.gpuP, label: 'prefill GPU', showMem: false });
  G.gpu(svg, { ...POOLS.gpuD, label: 'decode GPU', showMem: false });
  const prefillShown = Math.min(upToMs, TIMES.prefillMs);
  if (prefillShown > 0) G.request(svg, { x: POOLS.x0, y: POOLS.prefillY, owner: 'D', label: 'D', steps: [{ from: 0, to: prefillShown * POOL_PX_PER_MS, kind: 'prefill' }] });
  G.selectionMark(svg, { x: POOLS.x0 - 22, y: POOLS.prefillY, w: 18, h: 10 });
  Object.entries(DECODE_STEPS).forEach(([id, n]) => {
    const done = Math.min(n, Math.floor(upToMs / TIMES.decodeMs));
    if (done > 0) G.request(svg, { x: POOLS.x0, y: POOLS.decodeY[id], owner: id, label: id, steps: Array.from({ length: done }, (_, k) => ({ from: k * POOLS.stepPx, to: (k + 1) * POOLS.stepPx, kind: 'decode' })) });
  });
}

export function poolExtras(svg, p = 1) {
  if (p <= 0) return;
  const g = layer(svg, p);
  note(g, POOLS.x0 + PREFILL_PX / 2, POOLS.prefillY - 8, `prefill ${ms(TIMES.prefillMs)}`, { cls: '', anchor: 'middle' });
  connector(g, [POOLS.x0 + PREFILL_PX, POOLS.prefillY + 14], [POOLS.x0 + PREFILL_PX, POOLS.decodeY.D + 12]);
  note(g, POOLS.x0 + PREFILL_PX + 8, POOLS.decodeY.A + 4, `${ms(TIMES.decodeMs)} steps`, { cls: '' });
  note(g, LEFT, 252, `same time axis; D's bar is ${TIMES.stepsLong.toFixed(1)} of A's steps long`, { cls: '' });
  lines(g, LEFT, 284, [`decode steps ${ms(TIMES.decodeMs)} throughout`, `D's prefill ${ms(TIMES.prefillMs)} on its own GPU`]);
}

export function drawFrame3(svg, p) {
  scene(svg, p, (g) => drawFrame2(g, 1), (g) => {
    const t = seg(p, 0.1, 0.9) * TIMES.prefillMs;
    poolRows(g, t);
    poolExtras(g, seg(p, 0.1, 0.5));
  });
}

export { poolRows, TIMES };
