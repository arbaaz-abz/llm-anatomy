// speculative-decoding computed series (pure): the step-bar parts for frames 1, 3 and 9 and the curve sweeps of frames 7–9
// and the toy. Every value comes from stepTime / expectedTokens / simpleSpeedup / batchSpeedup; nothing is typed.
import { stepTime } from '@math/serving.js';
import { expectedTokens, simpleSpeedup, batchSpeedup } from '@math/specdec.js';
import { K_RANGE, MODEL } from './numbers.js';

const BYTES_PER_TB = 1e12;

// One step as stepBar wants it (seconds): the reading row split into weights / KV / activations, and the arithmetic.
export function stepParts({ tokens, seqs, model = MODEL }) {
  const r = stepTime({ ...model, tokens, seqs });
  const perSecond = model.bandwidthTBps * BYTES_PER_TB;
  const reading = [
    { label: 'weights read', s: model.weightBytesPerGpu / perSecond },
    { label: 'KV read', s: r.kvBytes / perSecond },
    { label: 'activations', s: r.actBytes / perSecond },
  ];
  return { reading, mathS: r.computeS, timeS: r.timeS, bound: r.bound };
}

export const tokensSeries = (alpha) => K_RANGE.map((k) => [k, expectedTokens(alpha, k)]);
export const speedupSeries = (alpha, c) => K_RANGE.map((k) => [k, simpleSpeedup(alpha, k, c)]);

// Users 1, 5, 10 … up to `max` (always ending on max itself): a smooth curve, and the stops the storyboard prints are on it.
export function userGrid(max) {
  const grid = [1];
  for (let u = 5; u < max; u += 5) grid.push(u);
  return [...grid, max];
}

export function batchSeries({ alpha, k, c, max, model = MODEL }) {
  return userGrid(max).map((batch) => [batch, batchSpeedup({ alpha, k, c, batch, model }).speedup]);
}

// A series cut at progress t (0–1) of its x range, ending on an interpolated point: curves "draw left to right".
export function partialSeries(points, t) {
  if (t >= 1) return points;
  const last = points.length - 1;
  const reach = t * last;
  const whole = Math.floor(reach);
  if (whole >= last) return points;
  const [x0, y0] = points[whole];
  const [x1, y1] = points[whole + 1];
  const f = reach - whole;
  return [...points.slice(0, whole + 1), [x0 + (x1 - x0) * f, y0 + (y1 - y0) * f]];
}
