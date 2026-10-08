// parallelism pure helpers (no DOM): the toy's state, formatters, the schedule as lanes and the "Check my work" text.
// Every number comes from math/parallel.js or math/training-memory.js; this file only arranges and formats them.
import { formatCount } from '@math/core.js';
import { pipelineBubble, pipelineSchedule, gpuCount } from '@math/parallel.js';
import { TRAINING_RECIPES, zeroPerGpuBytes } from '@math/training-memory.js';
import { formatShare } from '@shared/glyphs/bars.js';
import { LLAMA, LLAMA_PRESETS } from './numbers.js';

export const SCHEDULES = Object.freeze(['gpipe', '1f1b']);
export const SCHEDULE_LABELS = Object.freeze({ gpipe: 'GPipe', '1f1b': '1F1B' });
export const STAGE_STOPS = Object.freeze([2, 3, 4, 8]);
export const MICRO_RANGE = Object.freeze({ min: 1, max: 32 });
export const MICRO_CHIPS = Object.freeze([4, 8, 16, 32]);
export const DEGREE_STOPS = Object.freeze({
  tp: [1, 2, 4, 8, 16], cp: [1, 2, 4, 8, 16], pp: [1, 2, 4, 8, 16, 32],
  dp: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
});
export const ZERO_STAGES = Object.freeze([0, 1, 2, 3]);
export const MAX_LABELLED_COLUMNS = 12; // up to 12 columns print "F3" / "B3" at 43 px; wider grids are label-free
export const CELL_PX = 43; // one column on the stage's 568 px lane (52 px gutter + 12 × 43 = 568)

export const INITIAL_STATE = Object.freeze({ schedule: 'gpipe', stages: 4, micro: 4, preset: '8k', ...LLAMA_PRESETS['8k'], zero: LLAMA.zeroStage });

export const int = (n) => (n < 0 ? '−' : '') + Math.abs(n).toLocaleString('en-US');
export const bytesExact = (n) => `${int(n)} B`; // toy-sized byte counts print exactly (README visual constraints)

// Training state per GPU: one fixed unit so the terms of a sum can be added by eye. 2 decimals, whole numbers from 100 GB.
export function formatStateGB(bytes) {
  const gb = bytes / 1e9;
  return `${gb >= 100 ? int(Math.round(gb)) : gb.toFixed(2)} GB`;
}

// The preset whose degrees match the sliders, or 'custom'.
export function presetFor({ tp, cp, pp, dp }) {
  const hit = Object.entries(LLAMA_PRESETS).find(([, d]) => d.tp === tp && d.cp === cp && d.pp === pp && d.dp === dp);
  return hit ? hit[0] : 'custom';
}

const KIND = { F: 'forward', B: 'backward', '.': 'idle' };
export const kindOf = (op) => KIND[op[0]];

// One lane per stage; each grid cell becomes a one-column segment ('.' = idle, hatched).
export function lanesFromGrid(grid, { labels = true } = {}) {
  return grid.map((row, s) => ({
    label: `stage ${s + 1}`,
    segments: row.map((op, c) => ({ from: c, to: c + 1, kind: kindOf(op), label: labels && op !== '.' ? op : undefined })),
  }));
}

// The pipeline panel: the schedule, the bubble (formula and counted idle cells agree, tested), peak in flight, step length.
export function pipelineView({ schedule, stages, micro }) {
  const s = pipelineSchedule({ schedule, stages, microBatches: micro });
  const bubble = pipelineBubble({ stages, microBatches: micro });
  return {
    grid: s.grid, columns: s.columns, bubble, idle: s.idleFraction, peak: s.peakInFlight,
    bubbleText: formatShare(bubble), labelled: s.columns <= MAX_LABELLED_COLUMNS,
  };
}

export function stateView({ tp, cp, pp, dp, zero }) {
  const parts = zeroPerGpuBytes({ params: LLAMA.params, recipe: TRAINING_RECIPES.adam, stage: zero, dp, modelShards: tp * pp });
  return { gpus: gpuCount({ tp, cp, pp, dp }), parts, perGpu: formatStateGB(parts.total) };
}

const adamBytes = (r) => r.master + r.optimizer; // 12 B: FP32 master copy + two moments
const shardNote = (sharded, dp) => (sharded && dp > 1 ? ` ÷ ${dp}` : '');

// The §6 box for any state. The default state gives the storyboard's text exactly; an unsharded part prints no "÷ dp".
export function checkWork(state) {
  const { stages, micro, tp, cp, pp, dp, zero } = state;
  const r = TRAINING_RECIPES.adam;
  const view = stateView(state);
  const P = formatCount(LLAMA.params / (tp * pp), { digits: 4 });
  const part = (bytes, perParam, sharded) => `${P} × ${perParam} B${shardNote(sharded, dp)} = ${formatStateGB(bytes)}`;
  return [
    `bubble  = (${stages} − 1) ÷ (${micro} + ${stages} − 1) = ${stages - 1} ÷ ${micro + stages - 1} = ${formatShare(pipelineBubble({ stages, microBatches: micro }))}`,
    `GPUs    = ${tp} × ${cp} × ${pp} × ${dp} = ${int(view.gpus)}`,
    `per GPU (ZeRO-${zero}): ${formatCount(LLAMA.params)} ÷ (${tp} × ${pp}) = ${P} parameters`,
    `  weights ${part(view.parts.weights, r.weight, zero >= 3)} · gradients ${part(view.parts.grads, r.grad, zero >= 2)}`,
    `  optimizer ${part(view.parts.optimizer, adamBytes(r), zero >= 1)} · total ${view.perGpu}`,
  ].join('\n');
}
