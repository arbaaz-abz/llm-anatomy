// batching: the pieces of the lane frames (1-6, 10) that more than one frame draws. Pure functions of their arguments.
import * as G from '@shared/glyphs.js';
import { TOY_REQUESTS } from '@math/serving.js';
import { stepLane, requestsFor } from './model.js';
import {
  X0, STEP_PX, SLOT_TOP, CHIP_H, EDGE_RIGHT, seatY, headerY, layer, note, tint, rule, pxRow, drawLane, queueChips,
} from './stage.js';

export const REQ = requestsFor();
export const STATIC = stepLane({ kind: 'static', requests: REQ });
export const CONT = stepLane({ kind: 'continuous', requests: REQ });
export const CONT4 = stepLane({ kind: 'continuous', requests: REQ, seats: 4 });
export const pxRows = (lane, scale = STEP_PX) => lane.rows.map((r) => pxRow(r, scale));

// The request table of frame 1 (plain text numbers; the cast every later frame reuses).
const TABLE = Object.freeze({ headY: SLOT_TOP[1] + 22, firstY: SLOT_TOP[1] + 30, pitch: 26 });
const COLUMNS = Object.freeze([['arrives', 76], ['prompt', 150], ['decode steps', 232]]);

export function requestTable(parent, opacity = 1) {
  if (opacity <= 0) return;
  const g = layer(parent, opacity);
  note(g, X0, TABLE.headY, 'request');
  COLUMNS.forEach(([name, dx]) => note(g, X0 + dx, TABLE.headY, name, { anchor: 'start' }));
  TOY_REQUESTS.forEach((r, i) => {
    const y = TABLE.firstY + i * TABLE.pitch;
    G.token(g, { x: X0, y, text: r.id, owner: r.id });
    if (r.id === 'D') G.selectionMark(g, { x: X0, y, w: 28, h: CHIP_H });
    const text = [`step ${r.arrives}`, `${r.prompt} tokens`, `${r.output} steps`];
    COLUMNS.forEach(([, dx], c) => note(g, X0 + dx, y + CHIP_H / 2 + 4, text[c], { cls: 'g-label' }));
  });
}

// Either lane, complete: the static lane (slot 0) or the continuous lane (slot 1) at their end state, with optional pieces.
export function endState(parent, { lane, slot, title, right, opacity = 1, ri = Infinity, rb = Infinity, chips = [], endLabel = false }) {
  return drawLane(parent, { slot, title, right, rows: pxRows(lane), seats: lane.seats, opacity, rb, ri, chips, endLabel });
}

// The scheduler block of the continuous lane: it flashes (active) while a seat is handed out and at each step boundary.
const SCHED = Object.freeze({ x: 392, w: 78, h: 22 });
export function scheduler(parent, flashing, opacity = 1) {
  const g = layer(parent, opacity);
  G.block(g, { x: SCHED.x, y: SLOT_TOP[1] + 18, w: SCHED.w, h: SCHED.h, label: 'scheduler', state: flashing ? 'active' : 'idle' });
  return g;
}
export const isBoundary = (t) => Math.abs(t - Math.round(t)) < 0.06;

// A vertical marker across a lane's seat rows at axis position `x` (px from X0): the lane's last step.
export function endMarker(parent, { slot, seats, x, opacity }) {
  if (opacity <= 0) return;
  const top = seatY(slot, 0, true) - 5;
  const bottom = seatY(slot, seats - 1, true) + 15;
  rule(parent, X0 + x, top, X0 + x, bottom, { opacity });
}

export const noteAt = (parent, str, y, { x = 8, ...opts } = {}) => note(parent, x, y, str, opts);
export const rightEdge = EDGE_RIGHT;
export { tint, headerY };
