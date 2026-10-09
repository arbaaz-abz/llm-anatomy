// batching toy figure: both lanes, complete, in steps or milliseconds. Pure drawing from the toy view model (no state).
import * as G from '@shared/glyphs.js';
import { X0, AXIS_PX, AXIS_Y, layer, note, drawLane, pxRow } from './stage.js';

export const TOY_STAGE = Object.freeze({ w: 580, h: 276, laneTop: [24, 148], seatsDy: 50 });
const MIN_STEPS = 11; // the figure never packs fewer steps than the stage's 440 px axis holds at 40 px
const SMALL_AXIS_MS = 200;

function axis(parent, view, scale) {
  if (view.mode === 'steps') {
    note(parent, X0 - 14, AXIS_Y, 'step', { anchor: 'end' });
    for (let s = 0; s < view.axisSteps; s += 1) note(parent, X0 + (s + 0.5) * scale, AXIS_Y, String(s), { anchor: 'middle' });
    return;
  }
  note(parent, X0 - 14, AXIS_Y, 'ms', { anchor: 'end' });
  const tick = view.axisMs <= SMALL_AXIS_MS ? 50 : 100;
  for (let t = 0; t <= view.axisMs; t += tick) note(parent, X0 + t * scale, AXIS_Y, String(t), { anchor: 'middle' });
}

const waitsOf = (rows, scale) => rows.filter((r) => r.queue).map((r) => ({ id: r.id, from: r.queue.from * scale, to: r.queue.to * scale }));

export function drawToy(svg, view) {
  svg.replaceChildren();
  G.hatchFill(svg);
  const scale = view.mode === 'steps' ? AXIS_PX / Math.max(MIN_STEPS, view.axisSteps) : AXIS_PX / view.axisMs;
  const g = layer(svg);
  axis(g, view, scale);
  ['static', 'continuous'].forEach((kind, i) => {
    const lane = view.lanes[kind];
    const top = TOY_STAGE.laneTop[i];
    drawLane(g, {
      slot: 0, top, seatsTop: top + TOY_STAGE.seatsDy, title: view.titles[kind], seats: lane.seats, hasQueue: false,
      rows: lane.rows.map((r) => pxRow(r, scale)), waits: waitsOf(lane.rows, scale),
    });
    note(g, X0 - 24, top + 14 + 8, 'queue', { anchor: 'end' });
  });
  svg.setAttribute('aria-label', `Seat timelines, ${view.mode === 'steps' ? 'in steps' : 'in milliseconds'}: static batching above, continuous batching below.`);
  return svg;
}
