// parallelism toy view model (pure, no DOM): state → every string the toy prints, plus the try-this list.
// Numbers come from math/parallel.js and math/training-memory.js through format.js; this file only words them.
import { pipelineBubble, pipelineSchedule } from '@math/parallel.js';
import { formatShare } from '@shared/glyphs/bars.js';
import { INITIAL_STATE, MICRO_CHIPS, checkWork, int, lanesFromGrid, pipelineView, stateView, formatStateGB } from './format.js';
import { LLAMA } from './numbers.js';

export { INITIAL_STATE };

export const SCHEDULE_NOTE = 'Forward and backward each take one time unit here; real backward passes take about twice as long, which changes the bubble\'s size a little but not its shape.';
export const STATE_NOTE = 'State is weights, gradients and optimizer states for a 405B model with Adam (16 B per parameter). Activations are not modeled: 405B\'s layer shape is not in the course data. Context parallelism replicates the weights, so it does not divide the state.';
export const PRESET_CHIPS = Object.freeze([
  { value: '8k', label: '8K GPUs' },
  { value: '16k', label: '16K GPUs' },
  { value: 'long', label: 'long context' },
]);

export function toyView(state) {
  const p = pipelineView(state);
  const s = stateView(state);
  return {
    lanes: lanesFromGrid(p.grid, { labels: p.labelled }),
    columns: p.columns,
    labelled: p.labelled,
    bubble: p.bubbleText,
    peak: String(p.peak[0]),
    stepLength: String(p.columns),
    gpus: int(s.gpus),
    statePerGpu: s.perGpu,
    checkWork: checkWork(state),
  };
}

const arrow = (items) => items.join(' → ');

// The three try-this items as { prompt, insight, rest } with every number computed (README lesson 34).
export function tryThis() {
  const bubble = (stages, microBatches) => formatShare(pipelineBubble({ stages, microBatches }));
  const peak = (schedule) => pipelineSchedule({ schedule, stages: 4, microBatches: 16 }).peakInFlight[0];
  const llama = { ...INITIAL_STATE };
  const at = (patch) => stateView({ ...llama, ...patch });
  const one = { tp: 1, cp: 1, pp: 1, dp: 1, zero: 0 };
  const zero3Flat = at({ tp: 1, pp: 1, dp: 8192, zero: 3 });
  const zero3Split = at({ zero: 3 });
  return [
    {
      prompt: `GPipe, 4 stages: set micro-batches ${arrow(MICRO_CHIPS)} and read the bubble: ${arrow(MICRO_CHIPS.map((m) => bubble(4, m)))}. Then 8 stages with 32 micro-batches: ${bubble(8, 32)}.`,
      insight: 'the bubble is (p − 1)/(m + p − 1), so a deep pipeline needs many more micro-batches than stages.',
      rest: '',
    },
    {
      prompt: `4 stages, 16 micro-batches: GPipe holds ${peak('gpipe')} micro-batches of activations at stage 1; switch to 1F1B: ${peak('1f1b')}, while the bubble stays ${bubble(4, 16)}.`,
      insight: '1F1B doesn\'t shrink the bubble; it caps activation memory at p micro-batches, which is what makes large m affordable.',
      rest: ' Shrinking the bubble itself takes interleaved, zero-bubble or DualPipe schedules, which this page does not draw.',
    },
    {
      prompt: `Load the Llama 3.1 "8K GPUs" preset: ${int(at({}).gpus)} GPUs, ${at({}).perGpu} of state each with ZeRO-${LLAMA.zeroStage} (${formatStateGB(at({ zero: 1 }).parts.total)} if only optimizer states were sharded). Set all four degrees to 1: one GPU, ${at(one).perGpu}. Now set tp = pp = 1 and dp = ${int(8192)} with ZeRO-3: ${zero3Flat.perGpu}, the same as tp 8 × pp 16 × dp 64 with ZeRO-3${zero3Split.perGpu === zero3Flat.perGpu ? '' : ` (${zero3Split.perGpu})`}.`,
      insight: 'for state alone, ZeRO-3 over all GPUs would do; tensor and pipeline splits are there for what ZeRO cannot touch: activations, and the traffic of gathering every weight from thousands of GPUs every layer.',
      rest: '',
    },
  ];
}
