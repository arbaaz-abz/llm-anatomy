// Agentic RL (agentic-rl §11): rollout scheduling, the toy model of rounding mismatch, and importance-sampling
// treatments of one token. Pure: no DOM, inputs never mutated. Every bad argument throws RangeError('<fn>: <arg> must be …').
import { deepFreeze } from './core.js';

const SCHEDULE_MODES = deepFreeze(['sync', 'partial']);
const PRECISIONS = deepFreeze({ bf16: 1, fp16: 8 }); // FP16 rounds about 8× finer than BF16 (unit roundoff 2⁻¹¹ vs 2⁻⁸)
const CORRECTION_MODES = deepFreeze(['none', 'full', 'tis', 'icepop']);
const FLOAT_SLACK = 1e-9; // λ·G like 0.7 · 10 = 7.000000000000001 must not round up to 8

const isPositive = (x) => typeof x === 'number' && Number.isFinite(x) && x > 0;

function checkDurations(durations) {
  if (!Array.isArray(durations) || durations.length === 0) throw new RangeError('rolloutSchedule: durations must be a non-empty array');
  durations.forEach((d, i) => {
    if (!isPositive(d)) throw new RangeError(`rolloutSchedule: durations[${i}] must be a finite number > 0, got ${d}`);
  });
}

// One iteration of rollout scheduling. 'sync' waits for every episode; 'partial' ends the iteration when
// ceil(lambda·G) episodes have finished and carries the rest into the next iteration.
// utilization = busy ÷ (G · iterationTime): the page's single definition (README lesson 16).
//   [3,2,4,16,5,3,9,6] → { 16, 48, 80, 0.375, none carried } · partial 0.75 → { 6, 35, 13, 0.7292, rows 4 and 7 }
export function rolloutSchedule(durations, { mode = 'sync', lambda = 1 } = {}) {
  checkDurations(durations);
  if (!SCHEDULE_MODES.includes(mode)) throw new RangeError(`rolloutSchedule: mode must be one of ${SCHEDULE_MODES.join(', ')}, got ${mode}`);
  if (!(typeof lambda === 'number' && lambda > 0 && lambda <= 1)) throw new RangeError(`rolloutSchedule: lambda must be in (0, 1], got ${lambda}`);
  const sorted = [...durations].sort((a, b) => a - b);
  const waitFor = mode === 'sync' ? durations.length : Math.min(durations.length, Math.max(1, Math.ceil(lambda * durations.length - FLOAT_SLACK)));
  const iterationTime = sorted[waitFor - 1];
  const busy = durations.reduce((sum, d) => sum + Math.min(d, iterationTime), 0);
  const slots = durations.length * iterationTime;
  return { iterationTime, busy, idle: slots - busy, utilization: busy / slots, carried: durations.map((d) => d > iterationTime) };
}

// Toy model of rounding mismatch: FP16 rounds about 8× finer than BF16, so log ρ shrinks 8×.
//   (3.2) → 3.2 · (3.2, { precision: 'fp16' }) → 1.1565 · (0.4, fp16) → 0.8918 · (1, any) → 1
export function mismatchRatio(rhoBf16, { precision = 'bf16' } = {}) {
  if (!isPositive(rhoBf16)) throw new RangeError(`mismatchRatio: rho must be a finite number > 0, got ${rhoBf16}`);
  if (!(precision in PRECISIONS)) throw new RangeError(`mismatchRatio: precision must be one of ${Object.keys(PRECISIONS).join(', ')}, got ${precision}`);
  return PRECISIONS[precision] === 1 ? rhoBf16 : Math.exp(Math.log(rhoBf16) / PRECISIONS[precision]);
}

function checkBand(band) {
  if (!Array.isArray(band) || band.length !== 2 || !isPositive(band[0]) || !isPositive(band[1]) || !(band[0] < band[1])) {
    throw new RangeError(`isCorrection: band must be [low, high] with 0 < low < high, got ${JSON.stringify(band)}`);
  }
}

// Importance-sampling treatment of one token with ratio ρ = p_trainer / p_engine.
//   (3.2, none) → { 1, false } · (3.2, full) → { 3.2, false } · (3.2, tis) → { 2, false } · (3.2, icepop) → { 0, true }
export function isCorrection(rho, { mode = 'none', cap = 2, band = [0.5, 2] } = {}) {
  if (!isPositive(rho)) throw new RangeError(`isCorrection: rho must be a finite number > 0, got ${rho}`);
  if (!CORRECTION_MODES.includes(mode)) throw new RangeError(`isCorrection: mode must be one of ${CORRECTION_MODES.join(', ')}, got ${mode}`);
  if (!isPositive(cap)) throw new RangeError(`isCorrection: cap must be a finite number > 0, got ${cap}`);
  checkBand(band);
  if (mode === 'full') return { weight: rho, masked: false };
  if (mode === 'tis') return { weight: Math.min(rho, cap), masked: false };
  if (mode === 'icepop') {
    const masked = rho < band[0] || rho > band[1];
    return { weight: masked ? 0 : 1, masked };
  }
  return { weight: 1, masked: false };
}
