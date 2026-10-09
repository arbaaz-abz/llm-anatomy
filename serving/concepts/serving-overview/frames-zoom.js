// Frames 9–10: zoom out to the batch (the course's requests A–D from TOY_REQUESTS, replayed by simulateContinuous) and
// the map of the track (storyboard §5). A–D wear their --req hues and letters; your request is not among them (P4-R7).
import * as G from '@shared/glyphs.js';
import { BATCH, BATCH_STEP, LESSON_MAP, TITLES, batchAt } from './numbers.js';
import { IDLE, CHOSEN, seg, lerp, ease, arriving, leaving, pipeline, cacheTiles, note, label, region, layer } from './stage.js';
import { frame8End } from './frames-stream.js';

// ---- frame 9 layout: one row per request, steps 0…lastStep across ----
export const ROWS = Object.freeze({ chipX: 24, barX: 64, y0: 150, pitch: 32, stepW: 56 });
const rowY = (i) => ROWS.y0 + i * ROWS.pitch;
const COLUMN = Object.freeze({ top: ROWS.y0 - 8, bottom: rowY(BATCH.live.length - 1) + 32 });
const stepX = (s) => ROWS.barX + s * ROWS.stepW;

// One request's G.request steps (px from barX, before the arrival offset) for every step up to `upTo`: queue while it waits, its prefill step, decodes.
export function requestSteps(r, upTo) {
  const steps = [];
  const queueEnd = Math.min(r.admitted, upTo + 1);
  if (r.admitted > r.arrives && queueEnd > r.arrives) steps.push({ from: r.arrives * ROWS.stepW, to: queueEnd * ROWS.stepW, kind: 'queue' });
  if (r.admitted <= upTo) steps.push({ from: r.admitted * ROWS.stepW, to: (r.admitted + 1) * ROWS.stepW, kind: 'prefill' });
  for (let s = r.admitted + 1; s <= Math.min(r.finishes, upTo); s += 1) steps.push({ from: s * ROWS.stepW, to: (s + 1) * ROWS.stepW, kind: 'decode' });
  return steps;
}

function batchRows(svg, upTo, opacity) {
  if (opacity <= 0) return;
  const g = layer(svg, opacity);
  BATCH.live.forEach((r, i) => {
    G.token(g, { x: ROWS.chipX, y: rowY(i), text: r.id, owner: r.id });
    const offset = r.arrives * ROWS.stepW; // the rail starts when the request arrives
    const steps = requestSteps(r, upTo).map((s) => ({ ...s, from: s.from - offset, to: s.to - offset }));
    if (steps.length) G.request(g, { x: ROWS.barX + offset, y: rowY(i) + 7, steps, owner: r.id });
  });
}

const STEP_SENTENCE = (() => {
  const { advancing, done } = batchAt(BATCH_STEP);
  const names = `${advancing.slice(0, -1).join(', ')} and ${advancing.at(-1)}`;
  return `step ${BATCH_STEP}: ${names} each advance one token together; ${done.join(', ')} ${done.length === 1 ? 'is' : 'are'} done`;
})();
export const BATCH_NOTES = Object.freeze([STEP_SENTENCE, `four requests; ${TITLES.batching} follows these same four`]);

// ---- frame 9: the GPU never serves you alone ----
export function drawFrame9(svg, p) {
  pipeline(svg, p < 0.1 ? CHOSEN : IDLE);
  frame8End(svg, leaving(p));
  if (p >= 0.15) cacheTiles(svg, 0);
  const move = ease(seg(p, 0.2, 0.6));
  const landed = p >= 0.6;
  const shown = arriving(p);
  region(svg, { x: lerp(stepX(BATCH_STEP - 1), stepX(BATCH_STEP), move), y: COLUMN.top, w: ROWS.stepW, h: COLUMN.bottom - COLUMN.top }, shown);
  batchRows(svg, landed ? BATCH_STEP : BATCH_STEP - 1, shown);
  label(svg, lerp(stepX(BATCH_STEP - 1), stepX(BATCH_STEP), move) + ROWS.stepW / 2, COLUMN.bottom + 10, `step ${landed ? BATCH_STEP : BATCH_STEP - 1}`, { anchor: 'middle', cls: '', opacity: shown });
  BATCH_NOTES.forEach((text, i) => note(svg, 2 + i, text, seg(p, 0.65, 0.8)));
}

// ---- frame 10: every stop tagged with the lesson that speeds it up ----
export const MAP = Object.freeze({ stopX: 196, arrowX: 206, titleX: 220, y0: 144, pitch: 24 });

function batchEnd(svg, opacity) {
  if (opacity <= 0) return;
  region(svg, { x: stepX(BATCH_STEP), y: COLUMN.top, w: ROWS.stepW, h: COLUMN.bottom - COLUMN.top }, opacity);
  batchRows(svg, BATCH_STEP, opacity);
  label(svg, stepX(BATCH_STEP) + ROWS.stepW / 2, COLUMN.bottom + 10, `step ${BATCH_STEP}`, { anchor: 'middle', cls: '', opacity });
  BATCH_NOTES.forEach((text, i) => note(svg, 2 + i, text, opacity));
}

export function drawFrame10(svg, p) {
  pipeline(svg, IDLE);
  cacheTiles(svg, 0);
  batchEnd(svg, leaving(p));
  LESSON_MAP.forEach((m, i) => {
    const opacity = seg(p, 0.12 + 0.1 * i, 0.2 + 0.1 * i);
    const y = MAP.y0 + i * MAP.pitch;
    label(svg, MAP.stopX, y, m.stop, { anchor: 'end', opacity });
    label(svg, MAP.arrowX, y, '→', { opacity });
    label(svg, MAP.titleX, y, m.titles.join(', '), { cls: '', opacity });
  });
}

