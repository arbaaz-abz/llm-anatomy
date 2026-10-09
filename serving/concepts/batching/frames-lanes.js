// batching frames 1-6: the seat timeline in steps. Static lane (slot 0), continuous lane (slot 1); every frame a pure function of (svg, p).
import {
  STEP_PX, SLOT_TOP, seg, lerp, ease, arriving, leaving, layer, axisSteps, drawLane, queueChips, runClock, pulse, X0,
} from './stage.js';
import {
  REQ, STATIC, CONT, pxRows, requestTable, endState, scheduler, isBoundary, endMarker, noteAt,
} from './stage-lanes.js';
import { laneTitle, doneLine, idleLine, busyLine, summaryLine, admittedLine } from './format.js';
import { busyAt, idleAt, previousHolder } from './model.js';
import { TITLES } from './numbers.js';

const STATIC_END = STATIC.sim.steps; // 11: the static run ends after step 10
const CONT_END = CONT.sim.steps; // 7
const DIM = 0.45;
const STEP_RULE = 'Prefill ends with a request\'s first token; each tick after it is one decode step.'; // XS-1
const D_ROW = (lane) => lane.rows.find((r) => r.id === 'D');

// Frame 1: three empty seats, the queue, and the request table.
export function drawFrame1(svg, p) {
  const fade = seg(p, 0, 0.5);
  axisSteps(svg, fade);
  const chips = queueChips({ rows: STATIC.rows, t: 0, slot: 0 });
  drawLane(svg, { slot: 0, title: 'the GPU (3 seats)', rows: [], seats: 3, chips, opacity: fade });
  noteAt(svg, 'seat = a place in the running batch', SLOT_TOP[1] + 4, { opacity: fade, x: X0 });
  requestTable(svg, seg(p, 0.2, 0.8));
  noteAt(svg, `1 step ≈ 14.6 ms here (Llama-3.1-70B, FP8, one H200)`, 304, { opacity: fade });
  noteAt(svg, `memory is not modeled on this page, which is why ${TITLES.pagedAttention} comes next`, 322, { opacity: fade });
  noteAt(svg, STEP_RULE, 340, { opacity: fade });
}

// Frame 2: static lane. A, B, C slide onto seats at step 0 and run until C is done.
export function drawFrame2(svg, p) {
  const clock = runClock(p, { from: 0, to: STATIC_END - 4, pauses: [0] });
  axisSteps(svg);
  const chips = queueChips({ rows: STATIC.rows, t: clock.t, slot: 0, slide: clock.slide, slidDone: clock.slidDone });
  drawLane(svg, { slot: 0, title: laneTitle('static', 3), right: doneLine(STATIC.rows, clock.t), rows: pxRows(STATIC), seats: 3, rb: clock.t * STEP_PX, ri: 0, chips });
  requestTable(svg);
  const keep = leaving(p);
  noteAt(svg, `1 step ≈ 14.6 ms here (Llama-3.1-70B, FP8, one H200)`, 304, { opacity: keep });
  noteAt(svg, `memory is not modeled on this page, which is why ${TITLES.pagedAttention} comes next`, 322, { opacity: keep });
  noteAt(svg, STEP_RULE, 340, { opacity: keep });
}

// Frame 3: the idle hold fills in; D waits in the queue.
export function drawFrame3(svg, p) {
  const r = lerp(3, 7, p);
  axisSteps(svg);
  const chips = queueChips({ rows: STATIC.rows, t: r, slot: 0, seatedAt: [0] });
  drawLane(svg, { slot: 0, title: laneTitle('static', 3), right: idleLine(STATIC.rows, r, idleAt), rows: pxRows(STATIC), seats: 3, rb: (STATIC_END - 4) * STEP_PX, ri: r * STEP_PX, chips });
  requestTable(svg);
}

// Frame 4: D takes seat 1 at step 7; the busy counter climbs.
export function drawFrame4(svg, p) {
  const clock = runClock(p, { from: STATIC_END - 4, to: STATIC_END, pauses: [STATIC_END - 4] });
  axisSteps(svg);
  const chips = queueChips({ rows: STATIC.rows, t: clock.t, slot: 0, seatedAt: [0], slide: clock.slide, slidDone: clock.slidDone });
  const right = busyLine(STATIC.sim, busyAt(STATIC.sim, clock.t), clock.t >= STATIC_END);
  drawLane(svg, { slot: 0, title: laneTitle('static', 3), right, rows: pxRows(STATIC), seats: 3, rb: clock.t * STEP_PX, chips, endLabel: true });
  requestTable(svg);
}

// Frame 5: the continuous lane runs below; the static lane steps back.
export function drawFrame5(svg, p) {
  const clock = runClock(p, { from: 0, to: CONT_END, pauses: [0, 3] });
  axisSteps(svg);
  endState(svg, { lane: STATIC, slot: 0, title: laneTitle('static', 3), right: busyLine(STATIC.sim, STATIC.sim.busySeatSteps, true), opacity: lerp(1, DIM, arriving(p)), endLabel: true });
  const chips = queueChips({ rows: CONT.rows, t: clock.t, slot: 1, slide: clock.slide, slidDone: clock.slidDone });
  const d = D_ROW(CONT);
  const right = admittedLine(d, previousHolder(CONT.rows, d), clock.t);
  const g = layer(svg, arriving(p));
  drawLane(g, { slot: 1, title: laneTitle('continuous', 3), right, rows: pxRows(CONT), seats: 3, rb: clock.t * STEP_PX, chips, endLabel: true });
  scheduler(g, clock.slide !== null || isBoundary(clock.t));
  requestTable(svg, leaving(p));
}

// Both lanes complete, with their readouts and end markers. Frame 6 builds it; frame 7 fades it out.
export function bothLanes(svg, { opacity = 1, staticOpacity = 1, staticRight, contRight, markers = 1, contMarker = CONT_END }) {
  const g = layer(svg, opacity);
  axisSteps(g);
  endState(g, { lane: STATIC, slot: 0, title: laneTitle('static', 3), right: staticRight, opacity: staticOpacity, endLabel: true });
  endState(g, { lane: CONT, slot: 1, title: laneTitle('continuous', 3), right: contRight, endLabel: true });
  scheduler(g, false);
  endMarker(g, { slot: 0, seats: 3, x: STATIC_END * STEP_PX, opacity: markers });
  endMarker(g, { slot: 1, seats: 3, x: contMarker * STEP_PX, opacity: markers });
}

// Frame 6: the readouts replace the running counters and the continuous end marker settles at step 6.
export function drawFrame6(svg, p) {
  const swapped = seg(p, 0, 0.15) > 0;
  bothLanes(svg, {
    staticOpacity: lerp(DIM, 1, arriving(p)),
    staticRight: swapped ? summaryLine(STATIC.sim) : busyLine(STATIC.sim, STATIC.sim.busySeatSteps, true),
    contRight: swapped ? summaryLine(CONT.sim) : admittedLine(D_ROW(CONT), previousHolder(CONT.rows, D_ROW(CONT)), CONT_END),
    markers: seg(p, 0, 0.3),
    contMarker: lerp(STATIC_END, CONT_END, ease(seg(p, 0.1, 1))),
  });
}

export { pulse, REQ };
