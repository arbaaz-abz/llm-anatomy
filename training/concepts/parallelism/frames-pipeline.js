// Frames 5–8: pipeline parallelism, the GPipe schedule and 1F1B (storyboard §5). Pure functions of (progress).
import * as G from '@shared/glyphs.js';
import { formatShare } from '@shared/glyphs/bars.js';
import { pipelineBubble, pipelineSchedule } from '@math/parallel.js';
import { HANDOFF_BYTES, MORE_MICRO, PIPE, TOKENS } from './numbers.js';
import { CELL_PX, bytesExact, kindOf } from './format.js';
import { chipRow, counter, ease, fade, label, linked, seg, select } from './stage.js';

const SCHEDULE = (schedule, microBatches = PIPE.microBatches) => pipelineSchedule({ schedule, stages: PIPE.stages, microBatches });

// ---- frame 5: three stages of two blocks ----
const COL_X = Object.freeze([20, 215, 410]);
const COL_W = 150;
const HOPS = 2;

export function drawFrame5(svg, p) {
  chipRow(svg, 20, 16);
  label(svg, 190, 28, 'one micro-batch', { cls: 'g-label' });
  const clock = p * (2 * HOPS + 1); // stage 1 · hop · stage 2 · hop · stage 3, one unit each
  const lit = Math.min(Math.floor(clock / 2 + 1e-9), PIPE.stages - 1);
  COL_X.forEach((x, s) => {
    const state = s === lit ? 'active' : 'idle';
    for (let b = 0; b < PIPE.blocksPerStage; b += 1) G.block(svg, { x, y: 70 + b * 40, w: COL_W, h: 34, label: `block ${s * PIPE.blocksPerStage + b + 1}`, state });
    G.gpu(svg, { x: x + 27, y: 168, label: `GPU ${s + 1}`, showMem: false });
  });
  for (let hop = 0; hop < HOPS; hop += 1) {
    const t = seg(clock, 2 * hop + 1, 2 * hop + 2);
    if (t <= 0) continue;
    const x = COL_X[hop] + COL_W;
    G.flow(svg, { from: [x + 2, 104], to: [COL_X[hop + 1] - 2, 104], carry: 'activation', progress: ease(t) });
    label(svg, (x + COL_X[hop + 1]) / 2, 86, bytesExact(HANDOFF_BYTES), { anchor: 'middle', opacity: Math.min(t * 3, 1) });
  }
  label(svg, 20, 332, 'stand-in 6-block model', { cls: 'g-label' });
  label(svg, 20, 282, `per hand-off: ${TOKENS.length} tokens × 8 × 2 B = ${bytesExact(HANDOFF_BYTES)} forward`);
  label(svg, 20, 298, `backward: their gradients travel back, ${bytesExact(HANDOFF_BYTES)}`);
  label(svg, 20, 314, `${PIPE.stages} stages × ${PIPE.blocksPerStage} blocks`);
  counter(svg, `sent per GPU: ${bytesExact(Math.round(HANDOFF_BYTES * seg(clock, 1, 2)))}`);
}

// ---- frames 6–8: the schedule as lanes ----
const LANES = { x: 6, y: 96, w: 568 };

const lanesOf = (grid, { show }) => grid.map((row, s) => ({
  label: `stage ${s + 1}`,
  segments: row.flatMap((op, c) => (show(op, c) ? [{ from: c, to: c + 1, kind: kindOf(op), label: op === '.' ? undefined : op }] : [])),
}));

function drawLanes(parent, grid, show, link = false) {
  const lanes = lanesOf(grid, { show });
  const columns = grid[0].length;
  const scale = (LANES.w - laneGutter(lanes)) / columns;
  const opts = { x: LANES.x, y: LANES.y, w: LANES.w, lanes, scale, label: 'pipeline schedule' };
  const host = link ? linked(parent, 'bubble', { x: LANES.x, y: LANES.y, w: LANES.w, h: 88 }) : parent;
  return { group: G.laneTimeline(host, opts), layout: G.laneTimelineLayout({ ...opts, labels: true }) };
}

const laneGutter = (lanes) => Math.round(Math.max(...lanes.map((l) => l.label.length)) * 6.6) + 6;

function columnNumbers(svg, columns) {
  const gutter = laneGutter([{ label: 'stage 1' }]);
  for (let c = 0; c < columns; c += 1) label(svg, LANES.x + gutter + (c + 0.5) * CELL_PX, LANES.y - 12, String(c + 1), { anchor: 'middle', cls: 'g-label' });
  label(svg, LANES.x + gutter - 6, LANES.y - 12, 'time', { anchor: 'end', cls: 'g-label' });
}

function markFirst(svg, layout, opacity = 1) {
  const first = layout.segments[0];
  if (first) select(svg, first.x + LANES.x, first.y + LANES.y, first.width, first.height, opacity);
}

const idleCells = (grid, show) => grid.flat().filter((op, i) => op === '.' && show(op, i % grid[0].length)).length;
const cellsText = (n, total, fraction) => `bubble ${n} of ${total} cells = ${formatShare(fraction)}`;

export function drawFrame6(svg, p) {
  const { grid } = SCHEDULE('gpipe');
  columnNumbers(svg, grid[0].length);
  const forwardCols = Math.floor(seg(p, 0, 1) * 6 + 1e-9);
  const { layout } = drawLanes(svg, grid, (op, c) => op[0] === 'F' && c < forwardCols);
  markFirst(svg, layout);
  label(svg, 12, 216, `${PIPE.microBatches} micro-batches · forward done at column 6`);
  label(svg, 12, 236, 'outlined: micro-batch 1', { cls: 'g-label' });
  counter(svg, 'sent per GPU: 64 B per hand-off');
}

function gpipeFrame7(svg, p) {
  const { grid, peakInFlight } = SCHEDULE('gpipe');
  columnNumbers(svg, grid[0].length);
  const early = seg(p, 0, 0.2) > 0;
  const upTo = Math.floor(seg(p, 0.2, 1) * 6 + 1e-9);
  const show = (op, c) => (c < 6 ? (op[0] === 'F' || early) : c - 6 < upTo);
  const { layout } = drawLanes(svg, grid, show, true);
  markFirst(svg, layout);
  const total = grid.length * grid[0].length;
  label(svg, 12, 216, `${grid[0].length} columns × ${grid.length} stages = ${total} cells, ${grid.flat().filter((o) => o === '.').length} idle`);
  label(svg, 12, 236, `(${PIPE.stages} − 1) / (${PIPE.microBatches} + ${PIPE.stages} − 1) = ${formatShare(pipelineBubble({ stages: PIPE.stages, microBatches: PIPE.microBatches }))}`);
  label(svg, 12, 256, `stage 1 holds ${peakInFlight[0]} micro-batches' activations at its peak`);
  const done = idleCells(grid, show);
  label(svg, 12, 280, cellsText(done, total, done / total), { cls: '' });
}

export function drawFrame7(svg, p) { gpipeFrame7(svg, p); counter(svg, 'sent per GPU: 64 B per hand-off'); }

export function drawFrame8(svg, p) {
  const gpipe = SCHEDULE('gpipe');
  const f1b = SCHEDULE('1f1b');
  const t = ease(seg(p, 0, 0.7));
  columnNumbers(svg, gpipe.grid[0].length);
  const all = () => true;
  const a = drawLanes(svg, gpipe.grid, all, true);
  fade(a.group, 1 - t);
  const b = drawLanes(svg, f1b.grid, all);
  fade(b.group, t);
  markFirst(svg, b.layout);
  const more = { gpipe: SCHEDULE('gpipe', MORE_MICRO), f1b: SCHEDULE('1f1b', MORE_MICRO) };
  label(svg, 12, 216, `bubble ${formatShare(f1b.idleFraction)} (GPipe: ${formatShare(gpipe.idleFraction)}, unchanged)`);
  label(svg, 12, 236, `peak in flight per stage: GPipe ${gpipe.peakInFlight.join(', ')} → 1F1B ${f1b.peakInFlight.join(', ')}`);
  label(svg, 12, 256, `stage 1 holds at most ${f1b.peakInFlight[0]}`, { opacity: seg(p, 0.7, 1) });
  label(svg, 12, 280, `with ${MORE_MICRO} micro-batches: ${more.gpipe.peakInFlight[0]} vs ${more.f1b.peakInFlight[0]}, bubble ${formatShare(more.f1b.idleFraction)}`);
  counter(svg, 'sent per GPU: 64 B per hand-off');
}
