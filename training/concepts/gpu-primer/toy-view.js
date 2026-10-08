// gpu-primer toy view model (pure, no DOM): state + data → every string the toy prints and the specs of its two figures.
// Numbers come from format.js analyze (math/roofline.js); durations through formatDuration, ratios through formatRatio,
// counts through formatCount, bytes through formatBytes (README lesson 35).
import { formatBytes, formatCount, formatDuration, formatRatio } from '@math/core.js';
import { chipPreset, formatOptions } from './hardware.js';
import { analyze, checkWork, verdictText, fixed1, int, pct2, rangeText } from './format.js';

export const PLOT_DOMAIN = Object.freeze({ x: Object.freeze([0.1, 1e6]), y: Object.freeze([0.1, 1e5]) }); // fixed, so chips compare
export const FP8_NOTE = 'FP8 is counted at 8 bits per number (one per-tensor scale); the per-tile scales you meet in [[scale-reliability]] are not counted.';
export const SAME_FORMAT_NOTE = 'Inputs and outputs are stored in the same format as the weights.';
export const RESIDUAL_NOTE = 'residual add: element-wise ops are always memory-bound, so kernels fuse many of them into one trip to HBM';
const FMT_LABEL = Object.freeze({ bf16: 'BF16', fp8: 'FP8 (E4M3)', nvfp4: 'NVFP4', mxfp4: 'MXFP4' });
const US = 1e6;

// Rubin's bandwidth sources conflict: the mark printed under any Rubin point (storyboard §4).
export const conflictNote = (preset) => (preset.bandwidths.length > 1 ? `bandwidth: sources conflict, ${[...preset.bandwidths].sort((a, b) => a - b).join(' or ')} TB/s` : '');

function timeRatio(a) {
  const sides = a.ends.map((e) => (e.time.memoryS > e.time.computeS ? 'memory' : 'compute'));
  const ratios = a.ends.map((e) => Math.max(e.time.memoryS, e.time.computeS) / Math.min(e.time.memoryS, e.time.computeS));
  if (new Set(sides).size === 1) return { value: rangeText(ratios, formatRatio), sub: `${sides[0]} takes longer` };
  return { value: ratios.map(formatRatio).join(' · '), sub: a.ends.map((e, i) => `${sides[i]} longer at ${e.bandwidthTBps} TB/s`).join(' · ') };
}

function readouts(a) {
  const pick = (f) => a.ends.map(f);
  return {
    flops: formatCount(a.cost.flops),
    bytes: formatBytes(a.cost.bytes),
    intensity: fixed1(a.intensity),
    ridge: rangeText(pick((e) => e.ridge), int),
    tokensNeeded: rangeText(pick((e) => e.crossing), (c) => (Number.isFinite(c) ? fixed1(c) : 'never')),
    verdict: verdictText(a),
    attainable: `${rangeText(pick((e) => e.attainable), fixed1)} TFLOPS`,
    peakShare: rangeText(pick((e) => e.attainable), (t) => pct2(t, a.peakTflops)),
    memoryTime: rangeText(pick((e) => e.time.memoryS), formatDuration),
    computeTime: formatDuration(a.ends[0].time.computeS),
    timeRatio: timeRatio(a),
  };
}

// The time lanes on one axis (µs): memory per bandwidth end, then compute. With one end, the shorter lane's remainder is
// hatched idle (the resource waits); Rubin's two memory lanes are alternatives, so they get no idle remainder.
export function lanesSpec(a) {
  const memory = a.ends.map((e) => ({ label: a.ends.length > 1 ? `memory, ${e.bandwidthTBps} TB/s` : 'memory', us: e.time.memoryS * US, kind: 'memory' }));
  const compute = { label: 'compute', us: a.ends[0].time.computeS * US, kind: 'compute' };
  const rows = [...memory, compute];
  const full = Math.max(...rows.map((r) => r.us));
  const idle = (r) => (a.ends.length === 1 && r.us < full ? [{ from: r.us, to: full, kind: 'idle' }] : []);
  return {
    full,
    lanes: rows.map((r) => ({ label: r.label, segments: [{ from: 0, to: r.us, kind: r.kind, label: formatDuration(r.us / US) }, ...idle(r)] })),
  };
}

export function plotSpec(a, preset, fmt) {
  const fastest = a.ends[0];
  return {
    title: `${preset.label} · ${FMT_LABEL[a.format]}`,
    peakTflops: a.peakTflops,
    bandwidthTBps: fastest.bandwidthTBps,
    ridgeRange: a.ends.length > 1 ? [a.ends[0].ridge, a.ends[1].ridge] : null,
    xDomain: PLOT_DOMAIN.x,
    yDomain: PLOT_DOMAIN.y,
    points: [
      { intensity: a.intensity, label: `${int(a.tokens)} token${a.tokens === 1 ? '' : 's'}`, followed: true },
      { intensity: a.residIntensity, label: 'residual add' },
    ],
    fmt,
  };
}

export function toyView(state, data) {
  const preset = chipPreset(data, state.chip);
  const a = analyze(state, preset);
  const lanes = lanesSpec(a);
  return {
    readouts: readouts(a),
    formatOptions: formatOptions(preset),
    plot: plotSpec(a, preset, state.fmt),
    lanes,
    lanesWidth: `lanes share one time axis; full width = ${formatDuration(lanes.full / US)}`,
    conflict: conflictNote(preset),
    checkWork: checkWork(state, preset),
  };
}
