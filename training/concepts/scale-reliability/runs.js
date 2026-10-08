// scale-reliability runs: the Llama 3.1 405B preset as runPlan inputs, shared by the stage and the toy's defaults.
// Pure: every number comes from math/scale.js.
import { runPlan } from '@math/scale.js';
import { LLAMA, DEEPSEEK, H100, STAND_IN, PER_GPU_MTBF_H, BIG_GPUS } from './numbers.js';

const SECONDS_PER_HOUR = 3600;
const SECONDS_PER_MINUTE = 60;

export const LLAMA_RUN_ARGS = Object.freeze({
  params: LLAMA.params, tokens: LLAMA.tokens, gpus: LLAMA.gpus, peakTflops: H100.bf16, mfu: LLAMA.toyMfu,
  perGpuMtbfHours: PER_GPU_MTBF_H, saveH: STAND_IN.saveS / SECONDS_PER_HOUR, restartH: STAND_IN.restartMin / SECONDS_PER_MINUTE,
  dollarsPerGpuHour: DEEPSEEK.dollarsPerGpuHour,
});

// The Llama preset with some inputs changed, e.g. llamaRun({ gpus: 100000 }) (frame 10).
export const llamaRun = (overrides = {}) => runPlan({ ...LLAMA_RUN_ARGS, ...overrides });

// Frame 10's second bar is the widest of frames 2, 3 and 10: its GPU-hours are the full bar width ("same scale").
export const SAME_SCALE_GPU_HOURS = llamaRun({ gpus: BIG_GPUS }).gpuHours;
