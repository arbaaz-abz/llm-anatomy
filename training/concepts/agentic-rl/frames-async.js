// agentic-rl frames 5–6: a synchronous iteration waits for the slowest episode; a partial one cuts at 6 of 8.
import * as G from '@shared/glyphs.js';
import { rolloutSchedule } from '@math/agentic.js';
import { DURATIONS, GROUP_SIZE } from './numbers.js';
import { minutes, ofTotal, percent1, rowsText, carriedRows } from './format.js';
import {
  TIMELINE, layer, leaving, note, readout, seg, timelineBars, timelineX, timelineY, trainerBlock,
} from './stage.js';

const SYNC = rolloutSchedule(DURATIONS);
const PARTIAL = rolloutSchedule(DURATIONS, { mode: 'partial', lambda: 0.75 });
const SLOT_UNIT = 'slot-minutes';
const MIN_IDLE_ROOM = 36; // an "idle" label needs this many px between a bar's end and the cut
const CUT_LAMBDA = '0.75';

const slots = (s) => GROUP_SIZE * s.iterationTime;

// Frame 5: the bars grow together; each stops at its length and the space after it is labeled idle.
export function drawFrame5(svg, p) {
  const reach = TIMELINE.minutes * seg(p, 0.05, 0.65);
  timelineBars(svg, { reach });
  DURATIONS.forEach((d, row) => {
    if (d < TIMELINE.minutes) note(layer(svg, seg(reach, d, d + 1)), timelineX(d) + 6, timelineY(row) + 8, 'idle');
  });
  trainerBlock(svg, seg(p, 0.65, 0.8));
  const { iterationTime, busy, idle } = SYNC;
  readout(svg, [
    `iteration ${minutes(iterationTime)} · busy ${ofTotal(busy, slots(SYNC))} ${SLOT_UNIT}`,
    `utilization ${busy} / (${GROUP_SIZE} × ${iterationTime}) = ${percent1(busy, slots(SYNC))} · idle ${idle} ${SLOT_UNIT}`,
  ], { y: 316, opacity: seg(p, 0.8, 1) });
}

// Frame 6: the same bars, a cut at minute 6, the trainer lights there, bars 4 and 7 run on under newer weights.
export function drawFrame6(svg, p) {
  const cutX = timelineX(PARTIAL.iterationTime);
  const axisY = timelineBars(svg);
  DURATIONS.forEach((d, row) => {
    const room = (PARTIAL.iterationTime - d) * TIMELINE.unit;
    const keep = room >= MIN_IDLE_ROOM ? 1 : leaving(p) * (d < TIMELINE.minutes ? 1 : 0);
    if (keep > 0) note(layer(svg, keep), timelineX(d) + 6, timelineY(row) + 8, 'idle');
  });
  const drop = seg(p, 0.1, 0.45);
  const top = timelineY(0) - 8;
  const g = G.svgEl('g', { class: 'glyph g-note' }, svg);
  G.svgEl('line', { x1: cutX, y1: top, x2: cutX, y2: top + (axisY - top) * drop, stroke: 'var(--ink-muted)', 'stroke-width': 1 }, g);
  note(layer(svg, seg(p, 0.1, 0.25)), cutX, 14, `cut here: ${ofTotal(DURATIONS.length - carriedRows(PARTIAL.carried).length, GROUP_SIZE)} done`, { anchor: 'middle' });
  trainerBlock(svg, seg(p, 0.45, 0.6));
  PARTIAL.carried.forEach((carried, row) => {
    if (carried) note(layer(svg, seg(p, 0.6, 0.8)), cutX + 4, timelineY(row) + 22, 'finishes under newer weights');
  });
  const { iterationTime, busy } = PARTIAL;
  readout(svg, [
    `cut at ${ofTotal(GROUP_SIZE - carriedRows(PARTIAL.carried).length, GROUP_SIZE)} (λ = ${CUT_LAMBDA}) · iteration ${minutes(iterationTime)}`,
    `busy ${ofTotal(busy, slots(PARTIAL))} ${SLOT_UNIT} · utilization ${percent1(busy, slots(PARTIAL))}`,
    `carried: ${rowsText(carriedRows(PARTIAL.carried))} (off-policy: the part of each sampled before the cut)`,
  ], { y: 304, opacity: seg(p, 0.8, 1) });
}
