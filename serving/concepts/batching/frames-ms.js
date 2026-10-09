// batching frames 7-10: the mixed step, the long-prompt What-if in milliseconds, chunked prefill, and the 4-seat lane.
import * as G from '@shared/glyphs.js';
import { formatInt } from '@math/core.js';
import {
  X0, SLOT_TOP, AXIS_PX, CHIP_W, CHIP_H, seg, lerp, ease, arriving, leaving, layer, note, tint, axisSteps, axisMs, drawLane, blendRows, pxRow, headerY,
} from './stage.js';
import { REQ, STATIC, CONT, CONT4, pxRows, endState } from './stage-lanes.js';
import { bothLanes } from './frames-lanes.js';
import { laneTitle, summaryLine, mixedLine, stepLine, timesLine, budgetLine, changedLine, firstTokenLine, shortNote, slicesLine } from './format.js';
import { requestsFor, msLane, widenedLane, stepMs, LONG_PROMPT } from './model.js';
import { TITLES, MS_AXIS, BRANCH_BUDGET, MIXED_STEP, FOLLOWED } from './numbers.js';

const MS_SCALE = AXIS_PX / MS_AXIS;
const LONG = requestsFor({ dPrompt: LONG_PROMPT });
const ZOOM = Object.freeze({ x: 28, y: 150, pitch: 32, enter: 70, gpuX: 330, gpuY: 112, gpuW: 130, gpuH: 86 });
const STRIP = Object.freeze({ y: SLOT_TOP[0] + 18, h: 14, gap: 1.5 });
const MIXED = CONT.schedule[MIXED_STEP]; // step 3: D's six prompt tokens, A and C decoding
const MIXED_TIME = stepMs(MIXED) / 1e3;

// ---- Frame 7: one pass of the GPU holds D's prompt and two decode tokens ----
const ZOOM_CHIPS = Object.freeze([...Array(6).keys()].map((i) => ({ id: 'D', index: i + 1 })).concat([{ id: 'A' }, { id: 'C' }]));

export function zoomScene(svg, p, opacity = 1) {
  const g = layer(svg, opacity);
  note(g, 8, headerY(0), 'step 3 of the continuous lane');
  const enter = ease(seg(p, 0.1, 0.7));
  const shown = seg(p, 0.05, 0.4);
  const chips = layer(g, shown);
  ZOOM_CHIPS.forEach((c, i) => G.token(chips, { x: ZOOM.x + i * ZOOM.pitch - (1 - enter) * ZOOM.enter, y: ZOOM.y, text: c.id, owner: c.id, index: c.index }));
  const dEnd = ZOOM.x + 5 * ZOOM.pitch + CHIP_W - (1 - enter) * ZOOM.enter;
  G.selectionMark(chips, { x: ZOOM.x - (1 - enter) * ZOOM.enter, y: ZOOM.y, w: dEnd - ZOOM.x + (1 - enter) * ZOOM.enter, h: CHIP_H });
  const labels = layer(g, seg(p, 0.5, 0.9));
  note(labels, ZOOM.x + 2.5 * ZOOM.pitch + CHIP_W / 2, ZOOM.y + CHIP_H + 18, 'D: 6 prompt tokens', { anchor: 'middle' });
  note(labels, ZOOM.x + 6.5 * ZOOM.pitch + CHIP_W / 2, ZOOM.y + CHIP_H + 34, 'A, C: 1 token each', { anchor: 'middle' });
  G.gpu(g, { x: ZOOM.gpuX, y: ZOOM.gpuY, w: ZOOM.gpuW, h: ZOOM.gpuH, label: 'one H200', showMem: false });
  const midY = ZOOM.y + CHIP_H / 2;
  G.flow(g, { from: [ZOOM.x + 7 * ZOOM.pitch + CHIP_W + 8, midY], to: [ZOOM.gpuX - 6, ZOOM.gpuY + ZOOM.gpuH / 2 - 8], carry: 'token', progress: seg(p, 0.5, 1) });
  const readout = layer(g, seg(p, 0.7, 1));
  note(readout, 28, 262, mixedLine({ prefillTokens: MIXED.prefill[0].tokens, decoders: MIXED.decode.length, tokens: MIXED.tokens, timeS: MIXED_TIME }));
  note(readout, 28, 280, `memory-bound, as in ${TITLES.prefillDecode}`);
  return g;
}

export function drawFrame7(svg, p) {
  const out = leaving(p);
  if (out > 0) {
    bothLanes(svg, { opacity: out, staticRight: summaryLine(STATIC.sim), contRight: summaryLine(CONT.sim), markers: 1 });
  }
  zoomScene(svg, p, arriving(p));
}

// ---- The step strip: the run's steps as blocks whose widths are their times ----
// edges: ms positions of every step boundary; splits[j] = how far box j's gaps have opened (0 = flush, 1 = open).
function stepStrip(parent, { edges, labels, opacity = 1, gaps }) {
  const g = layer(parent, opacity);
  note(g, X0 - 14, STRIP.y + STRIP.h / 2 + 4, 'steps', { anchor: 'end' });
  edges.slice(0, -1).forEach((from, j) => {
    const x = X0 + from * MS_SCALE;
    const w = (edges[j + 1] - from) * MS_SCALE - STRIP.gap * gaps[j];
    tint(g, { x, y: STRIP.y, w, h: STRIP.h, rx: 2, fill: 'var(--surface-2)' });
    const text = labels[j];
    if (text && w >= text.width) note(g, x + w / 2, STRIP.y + STRIP.h / 2 + 4, text.text, { anchor: 'middle', opacity: text.opacity ?? 1 });
  });
  return g;
}

const CHAR_W = 6.6;
// A step box that prefills D's prompt prints its token count when it fits: "4,098 tokens" from ~90 px, the bare number from ~40 px.
function boxLabel(tokens, w) {
  const long = `${formatInt(tokens)} tokens`;
  const short = formatInt(tokens);
  const text = w >= long.length * CHAR_W + 8 ? long : short;
  return { text, width: text.length * CHAR_W + 4 };
}
function labelsFor(lane, opacityOf = () => 1) {
  return lane.schedule.map((e, j) => {
    const isPrefillOfD = e.prefill.some((x) => x.id === FOLLOWED);
    return { ...boxLabel(e.tokens, isPrefillOfD ? (lane.edges[j + 1] - lane.edges[j]) * MS_SCALE : 0), opacity: opacityOf(j) };
  });
}

// Shared by frames 8-9: the branch lane (rows in ms), the empty queue box and the seat tracks.
function branchLane(svg, rows, opacity = 1) {
  return drawLane(svg, { slot: 1, title: laneTitle('continuous', 3), rows, seats: 3, opacity });
}

const WHAT_IF = `What if D's prompt were ${formatInt(LONG_PROMPT)} tokens?`;
const FIRST = stepMs(CONT.schedule[0]);
const SHORT = stepMs(CONT.schedule[1]);
const SHORT_NOTE = shortNote(CONT.schedule[0].tokens, FIRST, SHORT);
const LINES_Y = Object.freeze([SLOT_TOP[0] + 58, SLOT_TOP[0] + 70, SLOT_TOP[0] + 82, SLOT_TOP[0] + 94]); // value lines; the short-step note sits above them
const NOTE_Y = SLOT_TOP[0] + 46;
const WIDE = widenedLane({ requests: LONG, shortRequests: REQ, blend: 1 });
const CHUNKED = msLane({ kind: 'continuous', requests: LONG, budget: BRANCH_BUDGET });
const WIDE_TIMES = Object.freeze({ a: WIDE.edges[WIDE.rows.find((r) => r.id === 'A').done + 1], c: WIDE.edges[WIDE.rows.find((r) => r.id === 'C').done + 1], d: WIDE.rows.find((r) => r.id === 'D').prefill.to });

// ---- Frame 8: the long prompt widens step 3 ----
export function drawFrame8(svg, p) {
  const lane = widenedLane({ requests: LONG, shortRequests: REQ, blend: p });
  const fade = arriving(p);
  const out = leaving(p);
  if (out > 0) zoomScene(svg, 1, out);
  const g = layer(svg, fade);
  axisMs(g, MS_AXIS);
  note(g, 8, headerY(0), WHAT_IF);
  stepStrip(g, { edges: lane.edges, labels: labelsFor(lane), gaps: lane.edges.slice(1).map(() => 1) });
  const lines = layer(g, seg(p, 0.6, 1));
  note(lines, 8, LINES_Y[0], stepLine(WIDE, MIXED_STEP));
  note(lines, 8, LINES_Y[1], timesLine(WIDE_TIMES));
  note(g, 8, NOTE_Y, SHORT_NOTE);
  branchLane(g, lane.rows.map((r) => pxRow(r, MS_SCALE)));
}

// ---- Frame 9: a 512-token budget slices the prompt ----
// Step 3 splits into the budget run's nine steps: the long run's step 3 is cut into equal parts, then every edge moves to its final time.
function splitEdges(wide, chunked) {
  const parts = chunked.durations.length - wide.durations.length + 1;
  const cut = (j) => {
    if (j <= MIXED_STEP) return wide.edges[j];
    if (j <= MIXED_STEP + parts) return lerp(wide.edges[MIXED_STEP], wide.edges[MIXED_STEP + 1], (j - MIXED_STEP) / parts);
    return wide.edges[j - parts + 1];
  };
  return chunked.edges.map((_, j) => cut(j));
}
const SPLIT_FROM = splitEdges(WIDE, CHUNKED);
const SPLIT_RANGE = Object.freeze([MIXED_STEP, MIXED_STEP + CHUNKED.durations.length - WIDE.durations.length]); // inclusive step indices
const inSplit = (j) => j >= SPLIT_RANGE[0] && j <= SPLIT_RANGE[1];

export function drawFrame9(svg, p) {
  const t = ease(seg(p, 0, 1));
  axisMs(svg, MS_AXIS);
  note(svg, 8, headerY(0), WHAT_IF);
  const edges = SPLIT_FROM.map((x, j) => lerp(x, CHUNKED.edges[j], t));
  const labels = labelsFor(CHUNKED, (j) => (inSplit(j) ? seg(p, 0.35, 0.8) : 1));
  stepStrip(svg, { edges, labels, gaps: CHUNKED.edges.slice(0, -1).map((_, j) => (inSplit(j) ? seg(p, 0, 0.4) : 1)) });
  const mergedLabel = 1 - seg(p, 0, 0.35);
  note(svg, X0 + ((SPLIT_FROM[SPLIT_RANGE[0]] + SPLIT_FROM[SPLIT_RANGE[1] + 1]) / 2) * MS_SCALE, STRIP.y + STRIP.h / 2 + 4, `${formatInt(WIDE.schedule[MIXED_STEP].tokens)} tokens`, { anchor: 'middle', opacity: mergedLabel });
  const was = layer(svg, 1 - seg(p, 0, 0.3));
  note(was, 8, LINES_Y[0], stepLine(WIDE, MIXED_STEP));
  note(was, 8, LINES_Y[1], timesLine(WIDE_TIMES));
  const now = layer(svg, seg(p, 0.5, 1));
  CHUNKED_LINES.forEach((line, i) => note(now, 8, LINES_Y[i], line));
  note(svg, 8, NOTE_Y, SHORT_NOTE);
  branchLane(svg, blendRows(WIDE.rows.map((r) => pxRow(r, MS_SCALE)), CHUNKED.rows.map((r) => pxRow(r, MS_SCALE)), t));
}

const timeOf = (lane, id) => lane.edges[lane.rows.find((r) => r.id === id).done + 1];
const firstOf = (lane) => lane.rows.find((r) => r.id === FOLLOWED).prefill.to;
const CHUNKED_LINES = Object.freeze([
  budgetLine(BRANCH_BUDGET, Math.max(...CHUNKED.durations)),
  `${changedLine('A', timeOf(CHUNKED, 'A'), timeOf(WIDE, 'A'))} · ${changedLine('C', timeOf(CHUNKED, 'C'), timeOf(WIDE, 'C'))}`,
  firstTokenLine(firstOf(CHUNKED), firstOf(WIDE)),
  slicesLine(CHUNKED.schedule.flatMap((e) => e.prefill.filter((x) => x.id === FOLLOWED).map((x) => x.tokens))),
]);

// ---- Frame 10: back to steps; the 4-seat lane ----
export function drawFrame10(svg, p) {
  const back = ease(seg(p, 0, 1));
  const msFade = 1 - seg(p, 0, 0.5);
  const stepFade = seg(p, 0.5, 1);
  axisMs(svg, MS_AXIS, msFade);
  axisSteps(svg, stepFade);
  const out = layer(svg, leaving(p));
  note(out, 8, headerY(0), WHAT_IF);
  stepStrip(out, { edges: CHUNKED.edges, labels: labelsFor(CHUNKED), gaps: CHUNKED.edges.slice(1).map(() => 1) });
  CHUNKED_LINES.forEach((line, i) => note(out, 8, LINES_Y[i], line));
  note(out, 8, NOTE_Y, SHORT_NOTE);
  endState(svg, { lane: STATIC, slot: 0, title: laneTitle('static', 3), right: summaryLine(STATIC.sim), opacity: 0.45 * arriving(p), endLabel: true });
  drawLane(svg, {
    slot: 1, title: laneTitle('continuous', 3), right: summaryLine(CONT.sim), seats: 3,
    rows: blendRows(CHUNKED.rows.map((r) => pxRow(r, MS_SCALE)), pxRows(CONT), back), endLabel: back >= 1,
  });
  const t4 = ease(seg(p, 0.1, 1));
  drawLane(svg, {
    slot: 2, title: laneTitle('continuousN', 4), right: summaryLine(CONT4.sim), seats: 4, hasQueue: false, opacity: arriving(p),
    rows: blendRows(pxRows(CONT), pxRows(CONT4), t4), endLabel: t4 >= 1, extraSeatOpacity: seg(p, 0.1, 0.6),
  });
  note(svg, 8, 330, `2023 engines reserved KV for the longest answer per seat: see ${TITLES.pagedAttention}`, { opacity: arriving(p) });
}

