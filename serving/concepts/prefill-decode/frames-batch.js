// prefill-decode frames 7–10: batching decode (8 users share one read of the weights), KV reads growing with the batch,
// the memory cap at 105 users, and a measured per-user vs per-GPU curve. The batch bar holds the top-left place throughout.
import * as G from '@shared/glyphs.js';
import { formatBytes, formatCount, formatDuration, formatInt, formatRatio } from '@math/core.js';
import { STATES, MAX_USERS, CACHE_PER_USER, RIDGE, MODEL, decode, intensityOf, perGpu, perUser } from './model.js';
import { H200, CONTEXT, MEASURED } from './numbers.js';
import { fixed1, memoryParts } from './format.js';
import { LEFT, CELL, BATCH, seg, lerp, arriving, leaving, layer, note, lines, batchChips, stepBarAt, readCell } from './stage.js';
import { frame6End } from './frames-steps.js';

const RIGHT = Object.freeze({ cellX: 330, textX: 330 + CELL + 10 });
// Per user on a linear axis, per GPU on a log axis: the low batches (68 and 509 tok/s per GPU) stay apart.
const CURVE = Object.freeze({ x: 300, y: 20, w: 276, h: 180, xTicks: Object.freeze([30, 40, 50, 60, 70]), yTicks: Object.freeze([50, 100, 500, 1000, 5000]) });
// Three decades from 10, so the flat roof is long enough for its label beside the ridge's and the ticks stay apart.
const INSET = Object.freeze({ x: 300, y: 214, w: 262, h: 148, xDomain: Object.freeze([10, 1e4]), yDomain: Object.freeze([10, 1e4]) });
const MEMBAR = Object.freeze({ labelY: 188, y: 196, w: 260 });
const LOW_Y = 300; // frames 8–9: the numbers under the left column
const usersText = (u) => `${formatInt(u)} user${u === 1 ? '' : 's'}`;
const B1 = STATES.batch[1];
const B8 = STATES.batch[8];
const B64 = STATES.batch[64];
const BMAX = STATES.batch[MAX_USERS];
const tweenUsers = (from, to, t) => Math.round(lerp(from, to, t));

const contextLabel = (parent) => note(parent, LEFT, BATCH.contextY, `each user: ${formatInt(CONTEXT)} tokens of context`);

// One batch bar with its header, and (frame 7) its tokens-per-GPU cell and step time at the right.
function batchBar(parent, { users, y, step = decode(users), reveal = 1, name = 'batch', cell = false }) {
  note(parent, LEFT, y - 8, usersText(users), { cls: '' });
  stepBarAt(parent, { y, step, reveal, name });
  if (!cell) return;
  readCell(parent, { x: RIGHT.cellX, y, value: cell === 'empty' ? null : perGpu(users, step), format: formatInt, label: 'tok/s per GPU' });
  if (cell !== 'empty') note(parent, RIGHT.textX, y + 24, `step ${formatDuration(step.timeS)}`);
}

const frame7Lines = () => [
  `1 user: ${formatDuration(B1.timeS)}, ${formatInt(perGpu(1, B1))} tok/s per GPU · 8 users: ${formatDuration(B8.timeS)}, ${formatInt(perGpu(8, B8))} tok/s per GPU`,
  `8 users make ${formatRatio(perGpu(8, B8) / perGpu(1, B1))} the tokens per second in a step ${formatRatio(B8.timeS / B1.timeS)} as long`,
];

// Frame 7: request A's step moves down as the reference; seven more users join and the step barely grows.
export function drawFrame7(svg, p) {
  const out = leaving(p);
  if (out > 0) frame6End(layer(svg, out));
  const inP = arriving(p);
  batchChips(layer(svg, inP), { users: p >= 0.5 ? 8 : 1 });
  contextLabel(layer(svg, inP));
  const down = seg(p, 0.3, 0.5);
  batchBar(layer(svg, inP), { users: 1, y: lerp(BATCH.barY, BATCH.secondBarY, down), cell: true, name: 'one' });
  if (p >= 0.5) batchBar(svg, { users: 8, y: BATCH.barY, reveal: seg(p, 0.5, 0.8), cell: p >= 0.8 ? true : 'empty', name: 'eight' });
  lines(layer(svg, seg(p, 0.85, 1)), LEFT, LOW_Y + 14, frame7Lines());
}

const FRAME7_END = (parent) => {
  batchBar(parent, { users: 1, y: BATCH.secondBarY, cell: true, name: 'one' });
  readCell(parent, { x: RIGHT.cellX, y: BATCH.barY, value: perGpu(8, B8), format: formatInt, label: 'tok/s per GPU' });
  note(parent, RIGHT.textX, BATCH.barY + 24, `step ${formatDuration(B8.timeS)}`);
  lines(parent, LEFT, LOW_Y + 14, frame7Lines());
};

// The per-user vs per-GPU curve (curvePlot) for the users shown; the last one is followed.
function curve(parent, { shown, opacity = 1 }) {
  if (opacity <= 0 || shown.length === 0) return;
  const pts = shown.map((u) => ({ u, x: perUser(STATES.batch[u]), y: perGpu(u, STATES.batch[u]) }));
  G.curvePlot(layer(parent, opacity), {
    x: CURVE.x, y: CURVE.y, w: CURVE.w, h: CURVE.h, label: 'tokens per second per user against per GPU',
    xAxis: { label: 'tokens/s per user', ticks: CURVE.xTicks }, yAxis: { label: 'tokens/s per GPU', log: true, ticks: CURVE.yTicks },
    series: pts.length > 1 ? [{ points: pts.map((q) => [q.x, q.y]), style: 'muted' }] : [],
    // "1 user" names the unit; the other markers print their user count alone, so neighbours' labels stay apart.
    markers: pts.map((q, i) => ({ x: q.x, y: q.y, label: q.u === 1 ? usersText(1) : formatInt(q.u), followed: i === pts.length - 1 })),
  });
}

const batchTop = (parent, users) => {
  batchChips(parent, { users });
  contextLabel(parent);
  batchBar(parent, { users, y: BATCH.barY });
};

const frame8Lines = () => [`64 users: step ${formatDuration(B64.timeS)},`, `${formatCount(perUser(B64))} tok/s per user, ${formatInt(perGpu(64, B64))} per GPU`];

// Frame 8: markers for 1 and 8 users, then the batch grows to 64 and its KV read grows with it.
export function drawFrame8(svg, p) {
  const out = leaving(p);
  if (out > 0) FRAME7_END(layer(svg, out));
  const users = tweenUsers(8, 64, seg(p, 0.4, 0.7));
  batchTop(svg, users);
  curve(svg, { shown: users === 64 ? [1, 8, 64] : [1, 8], opacity: seg(p, 0.1, 0.35) });
  lines(layer(svg, seg(p, 0.8, 1)), LEFT, LOW_Y, frame8Lines());
}

// The H200's memory: weights, the users' KV cache, what is left (shareBar, printed in bytes). A part too thin to draw
// (the last 536 MB at 105 users) is named in the label instead, so the bar reads "full" and every byte is still printed.
function memoryBar(parent, users) {
  const m = memoryParts({ weights: MODEL.weightBytesPerGpu, kv: users * CACHE_PER_USER, total: H200.hbmBytes, w: MEMBAR.w });
  const thin = m.thin.length ? `: full, ${m.thin.map((q) => `${q.name} ${formatBytes(q.value)}`).join(', ')}` : '';
  note(parent, LEFT, MEMBAR.labelY, `${H200.label} memory, ${formatBytes(H200.hbmBytes)} ${H200.basis}${thin}`);
  G.shareBar(parent, { x: LEFT, y: MEMBAR.y, w: MEMBAR.w, tail: 'none', minSegment: 0, label: `${H200.label} memory`, format: (share) => formatBytes(share * m.drawnTotal), parts: m.parts });
}

function inset(parent, opacity) {
  if (opacity <= 0) return;
  G.roofline(layer(parent, opacity), {
    ...INSET, peakTflops: MODEL.peakTflops, bandwidthTBps: MODEL.bandwidthTBps, label: 'H200 roofline, FP8',
    points: [{ intensity: intensityOf(BMAX), label: usersText(MAX_USERS), followed: true }],
  });
}

const frame9Lines = () => [
  `${usersText(MAX_USERS)} fill it: step ${formatDuration(BMAX.timeS)}`,
  `${formatCount(perUser(BMAX))} tok/s per user, ${formatInt(perGpu(MAX_USERS, BMAX))} per GPU`,
  `intensity ${fixed1(intensityOf(BMAX))} vs ridge ${fixed1(RIDGE)}:`,
  'still memory-bound',
];

function frame9Scene(parent, { users, full }) {
  batchTop(parent, users);
  memoryBar(parent, users);
  curve(parent, { shown: full ? [1, 8, 64, MAX_USERS] : [1, 8, 64] });
}

// Frame 9: the batch grows until the KV cache fills the memory at 105 users, still far left of the ridge.
export function drawFrame9(svg, p) {
  const out = leaving(p);
  if (out > 0) lines(layer(svg, out), LEFT, LOW_Y, frame8Lines());
  const users = tweenUsers(64, MAX_USERS, seg(p, 0.15, 0.6));
  batchTop(svg, users);
  memoryBar(layer(svg, arriving(p)), users);
  curve(svg, { shown: users === MAX_USERS ? [1, 8, 64, MAX_USERS] : [1, 8, 64] });
  inset(svg, seg(p, 0.6, 0.8));
  lines(layer(svg, seg(p, 0.8, 1)), LEFT, LOW_Y - 4, frame9Lines());
}

// ---- frame 10: a measured curve ----
const MEAS = Object.freeze({ x: 24, y: 30, w: 530, h: 250, xTicks: Object.freeze([0, 10, 20, 30]), yTicks: Object.freeze([0, 4000, 8000, 12000]) });
const LIKE_WITH_LIKE = Object.freeze(['a different model and GPU; compare the shape (asking each user to go faster', 'costs tokens per GPU), not the numbers']);

function measured(parent, connector) {
  const [a, b] = MEASURED.points;
  note(parent, LEFT, 16, 'measured, not this toy\'s model', { cls: '' });
  G.curvePlot(parent, {
    x: MEAS.x, y: MEAS.y, w: MEAS.w, h: MEAS.h, label: `${MEASURED.model} on ${MEASURED.hardware}, measured`,
    xAxis: { label: 'tokens/s per user', ticks: MEAS.xTicks }, yAxis: { label: 'tokens/s per GPU', ticks: MEAS.yTicks },
    series: connector ? [{ points: [[a.perUser, a.perGpu], [b.perUser, b.perGpu]], style: 'muted' }] : [],
    markers: MEASURED.points.map((q) => ({ x: q.perUser, y: q.perGpu, label: `${q.perUser} tok/s/user · ${formatInt(q.perGpu)} tok/s/GPU` })),
  });
  lines(parent, LEFT, 300, [`${MEASURED.model} on ${MEASURED.hardware}`, MEASURED.conditions]);
  lines(parent, LEFT, 336, LIKE_WITH_LIKE, { cls: 'g-label' });
}

// Frame 10: two InferenceX points for DeepSeek-V4-Pro on GB300: twice the speed per user, about half the tokens per GPU.
export function drawFrame10(svg, p) {
  const out = leaving(p);
  if (out > 0) {
    const g = layer(svg, out);
    frame9Scene(g, { users: MAX_USERS, full: true });
    inset(g, 1);
    lines(g, LEFT, LOW_Y - 4, frame9Lines());
  }
  measured(layer(svg, arriving(p)), p >= 0.5);
}

