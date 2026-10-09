// prefill-decode toy view model (pure, no DOM): state + data → every string the toy prints and the specs of its figures.
// Numbers come from analyze (format.js: math/serving.js, math/roofline.js, math/memory.js); bytes through formatBytes,
// durations through formatDuration, exact counts through formatInt, per-user rates through formatCount (README lesson 35).
import { stepTime } from '@math/serving.js';
import { formatBytes, formatCount, formatDuration, formatInt } from '@math/core.js';
import { GPUS, gpuPreset, gpuMemoryText, hbmText, WEIGHT_FORMATS } from './hardware.js';
import { analyze, checkWork, boundText, fixed1, formatFlops, tflops, tbps, usersStops, usersLabel, memoryParts, thinText } from './format.js';
import { readParts } from './model.js';

export const TOY_STEP_W = 420; // the toy step bar's width; its scale is this state's step (the longer row spans it)
export const DOES_NOT_FIT = 'does not fit on one GPU: see [[serving-calculator]]';

const row = (label, sub, value, name, valueSub) => ({ label, ...(sub ? { sub } : {}), cells: [{ value, name, ...(valueSub ? { sub: valueSub } : {}) }] });

function stepTable(a) {
  const { step, model } = a;
  return {
    head: ['This step', ''], name: 'step',
    rows: [
      row('Operations', '2 × parameters × tokens', formatFlops(step.flops), 'flops'),
      row('Weights read', `${WEIGHT_FORMATS[a.weights].label}: ${model.actBytesPerElem} byte${model.actBytesPerElem === 1 ? '' : 's'} per weight`, formatBytes(model.weightBytesPerGpu), 'weights-bytes'),
      row('Activations read and written', '2 × parameters × bytes × tokens ÷ d_model', formatBytes(step.actBytes), 'act-bytes'),
      ...(a.phase === 'decode' ? [row('KV read', 'users × context × KV bytes per token', formatBytes(step.kvBytes), 'kv-bytes')] : []),
      row('Bytes moved', 'weights + activations' + (a.phase === 'decode' ? ' + KV' : ''), formatBytes(step.bytes), 'bytes'),
      row('Math time', 'operations ÷ peak', formatDuration(step.computeS), 'math-time'),
      row('Reading time', 'bytes ÷ bandwidth', formatDuration(step.memoryS), 'read-time'),
      row('Step time', 'the longer of the two: a floor', formatDuration(step.timeS), 'step-time'),
      row('Bound', null, boundText(step), 'bound'),
    ],
  };
}

function gpuTable(a) {
  const { model, preset } = a;
  return {
    head: ['On this GPU', ''], name: 'gpu-readouts',
    rows: [
      ...(a.step ? [row('Arithmetic intensity', 'operations per byte read', fixed1(a.intensity), 'intensity')] : []),
      row('Ridge point', 'peak ÷ bandwidth, operations per byte', fixed1(a.ridge), 'ridge'),
      row('Tokens per weight read to be compute-bound', 'A GPU for LLM people', fixed1(a.crossing), 'tokens-needed'),
      row('GPU memory', null, hbmText(preset), 'hbm'),
      row('Dense peak', null, tflops(model.peakTflops), 'peak', WEIGHT_FORMATS[a.weights].label),
      row('HBM bandwidth', null, tbps(model.bandwidthTBps), 'bandwidth'),
    ],
  };
}

function usersTable(a) {
  const rates = a.step ? [
    row('Tokens/s per user', '1 ÷ step time', formatCount(1 / a.step.timeS), 'per-user'),
    row('Tokens/s per GPU', 'users ÷ step time, output tokens', formatInt(a.users / a.step.timeS), 'per-gpu'),
  ] : [];
  return {
    head: ['Users and throughput', ''], name: 'users-readouts',
    rows: [
      ...rates,
      row('Max users that fit', 'free memory ÷ cache per user, rounded down', formatInt(a.maxUsers), 'max-users'),
      row('Cache per user', `${formatInt(a.context)} tokens × ${formatInt(a.model.kvBytesPerToken)} B`, formatBytes(a.cachePerUser), 'cache-per-user'),
      row('Free after the weights', null, formatBytes(a.free), 'free'),
    ],
  };
}

function prefillTable(a) {
  return {
    head: ['This prompt', ''], name: 'prefill-readouts',
    rows: [
      row('Prompt tokens per second', 'prompt ÷ step time', formatInt(a.tokens / a.step.timeS), 'prefill-rate'),
      row('TTFT for this prompt', 'the prefill step, no queue', formatDuration(a.step.timeS), 'ttft'),
      row('Ceiling if math were the only limit', 'peak ÷ (2 × parameters)', formatInt(a.ceiling), 'ceiling'),
    ],
  };
}

const fitTable = (a) => ({
  head: ['Fit', ''], name: 'fit-readouts',
  rows: [
    row('GPU memory', null, hbmText(a.preset), 'hbm'),
    row('Weights', `${WEIGHT_FORMATS[a.weights].label}: ${a.model.actBytesPerElem} bytes per weight`, formatBytes(a.model.weightBytesPerGpu), 'weights-bytes'),
    row('Max users that fit', null, formatInt(0), 'max-users'),
  ],
});

function tables(a) {
  if (!a.fits) return [fitTable(a)];
  const third = a.phase === 'prefill' ? prefillTable(a) : usersTable(a);
  return [...(a.step ? [stepTable(a)] : []), gpuTable(a), third];
}

function usersNote(a) {
  if (!a.fits || a.phase !== 'decode') return '';
  if (a.maxUsers === 0) return `No user fits: one user's cache at ${formatInt(a.context)} tokens (${formatBytes(a.cachePerUser)}) is more than the ${formatBytes(a.free)} free.`;
  return a.clamped ? `Users clamped to ${formatInt(a.maxUsers)}, the most that fit at ${formatInt(a.context)} tokens of context.` : '';
}

const stepBarSpec = (a) => ({
  caption: `Full width = this step, ${formatDuration(a.step.timeS)}: a floor, since real engines reach less than this bandwidth and real steps take longer.`,
  scaleS: a.step.timeS,
  reading: readParts(a.step, { weightBytes: a.model.weightBytesPerGpu, bandwidthTBps: a.model.bandwidthTBps }),
  mathS: a.step.computeS,
});

export const TOY_MEMORY_W = 420;

function memoryBarSpec(a) {
  const m = memoryParts({ weights: a.model.weightBytesPerGpu, kv: a.users * a.cachePerUser, total: a.preset.hbm.bytes, w: TOY_MEMORY_W });
  const label = `${a.preset.label} memory, ${hbmText(a.preset)}`;
  const thin = thinText(m.thin);
  return { parts: m.parts, drawnTotal: m.drawnTotal, label, thin, caption: thin ? `${label}; ${thin}.` : `${label}.` };
}

const decadeBelow = (v) => 10 ** Math.floor(Math.log10(v));
const decadeAbove = (v) => 10 ** Math.ceil(Math.log10(v));
// A linear per-user axis in steps of 10 (20 past a 60-wide span) around the points; a log per-GPU axis in decades.
function xAxisFor(xs) {
  const lo = Math.floor(Math.min(...xs) / 10) * 10;
  const hi = Math.max(Math.ceil(Math.max(...xs) / 10) * 10, lo + 10);
  const step = hi - lo > 60 ? 20 : 10;
  return { label: 'tokens/s per user', domain: [lo, hi], ticks: Array.from({ length: Math.floor((hi - lo) / step) + 1 }, (_, i) => lo + i * step) };
}
function yAxisFor(ys) {
  const lo = decadeBelow(Math.min(...ys));
  const hi = Math.max(decadeAbove(Math.max(...ys)), lo * 10);
  return { label: 'tokens/s per GPU', log: true, domain: [lo, hi], ticks: Array.from({ length: Math.round(Math.log10(hi / lo)) + 1 }, (_, i) => lo * 10 ** i) };
}

// The per-user vs per-GPU curve over the users stops; the current point is followed. Labels: "1 user", the current
// point and the max that fits (the rest are unlabeled dots, so neighbours' labels never collide).
function curveSpec(a) {
  const points = usersStops(a.maxUsers).map((u) => {
    const s = stepTime({ ...a.model, tokens: u, seqs: u, context: a.context });
    return { users: u, x: 1 / s.timeS, y: u / s.timeS };
  });
  const label = (p) => {
    if (p.users === 1) return usersLabel(1, 0);
    if (p.users === a.users) return formatInt(p.users);
    return p.users === a.maxUsers ? `${formatInt(p.users)}, max` : null;
  };
  return {
    caption: `One dot per stop of Users in the batch at ${formatInt(a.context)} tokens of context; the framed dot is this batch. Tokens per GPU count output tokens only.`,
    xAxis: xAxisFor(points.map((p) => p.x)),
    yAxis: yAxisFor(points.map((p) => p.y)),
    series: points.length > 1 ? [{ points: points.map((p) => [p.x, p.y]), style: 'muted' }] : [],
    markers: points.map((p) => ({ x: p.x, y: p.y, label: label(p), followed: p.users === a.users })),
  };
}

const figures = (a) => {
  const showDecode = a.fits && a.phase === 'decode';
  return {
    stepBar: a.step ? stepBarSpec(a) : null,
    memoryBar: showDecode ? memoryBarSpec(a) : null,
    curve: showDecode && a.step ? curveSpec(a) : null,
  };
};

// The line under the GPU chips: every chip's memory with its basis word.
export const gpuNote = (data) => `GPU memory: ${GPUS.map((g) => gpuMemoryText(gpuPreset(data, g.id)).replace(/^\S+: /, `${g.label} `)).join(' · ')}.`;

export function toyView(state, data) {
  const preset = gpuPreset(data, state.hw);
  const a = { ...analyze(state, preset), weights: state.weights };
  const stops = usersStops(a.maxUsers);
  return {
    fit: a.fits ? null : DOES_NOT_FIT,
    tables: tables(a),
    bound: a.step ? boundText(a.step) : null,
    users: { stops, value: a.users, label: usersLabel(a.users, a.maxUsers) },
    usersNote: usersNote(a),
    ...figures(a),
    checkWork: checkWork(a),
  };
}
