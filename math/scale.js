// Training at scale: 6ND, GPU-hours, MFU, failures, checkpoints and cost (scale-reliability §6). Hours and TFLOPS in.
// Pure: no DOM, inputs never mutated. Owned by scale-reliability; scaling-laws imports trainingFlops and
// cluster-topology imports FLOPS_PER_PARAM_TOKEN, so 6ND has one exporter (README lesson 16).
// One definition per metric: run-average MFU is always mfuFrom(...) over all GPU-hours; MFU while training is an input
// to runPlan, which relates them: run-average = while-training × (1 − loss).
// FROZEN after Plan 3's shared prep (S3): change only through a shared patch.

export const FLOPS_PER_PARAM_TOKEN = 6; // 2 forward + 4 backward

const SECONDS_PER_HOUR = 3600;
const HOURS_PER_DAY = 24;
const TERA = 1e12;

function requirePositive(fn, name, value) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    throw new RangeError(`${fn}: ${name} must be a positive finite number, got ${value}`);
  }
}

// Positive, Infinity allowed (a component that never fails).
function requirePositiveOrInfinite(fn, name, value) {
  if (typeof value !== 'number' || Number.isNaN(value) || value <= 0) {
    throw new RangeError(`${fn}: ${name} must be a positive number (Infinity allowed), got ${value}`);
  }
}

function requireNonNegative(fn, name, value) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    throw new RangeError(`${fn}: ${name} must be a finite number ≥ 0, got ${value}`);
  }
}

function requireCount(fn, name, value) {
  if (!Number.isInteger(value) || value < 1) throw new RangeError(`${fn}: ${name} must be a positive integer, got ${value}`);
}

// FLOPS_PER_PARAM_TOKEN · N · D (ignores attention FLOPs, which grow at long context). Pass active parameters for a MoE.
export function trainingFlops({ params, tokens }) {
  requirePositive('trainingFlops', 'params', params);
  requirePositive('trainingFlops', 'tokens', tokens);
  return FLOPS_PER_PARAM_TOKEN * params * tokens;
}

export function gpuHoursAt({ flops, peakTflops, mfu }) {
  requirePositive('gpuHoursAt', 'flops', flops);
  requirePositive('gpuHoursAt', 'peakTflops', peakTflops);
  if (typeof mfu !== 'number' || !(mfu > 0 && mfu <= 1)) throw new RangeError(`gpuHoursAt: mfu must be a number in (0, 1], got ${mfu}`);
  return flops / (peakTflops * TERA * mfu) / SECONDS_PER_HOUR;
}

// Run-average MFU against the peak of the precision named on screen.
export function mfuFrom({ flops, gpuHours, peakTflops }) {
  requirePositive('mfuFrom', 'flops', flops);
  requirePositive('mfuFrom', 'gpuHours', gpuHours);
  requirePositive('mfuFrom', 'peakTflops', peakTflops);
  return flops / (gpuHours * SECONDS_PER_HOUR * peakTflops * TERA);
}

export function achievedTflopsPerGpu({ flops, gpuHours }) {
  requirePositive('achievedTflopsPerGpu', 'flops', flops);
  requirePositive('achievedTflopsPerGpu', 'gpuHours', gpuHours);
  return flops / (gpuHours * SECONDS_PER_HOUR) / TERA;
}

export function wallClockDays({ gpuHours, gpus }) {
  requirePositive('wallClockDays', 'gpuHours', gpuHours);
  requireCount('wallClockDays', 'gpus', gpus);
  return gpuHours / gpus / HOURS_PER_DAY;
}

// The cluster fails as often as all its GPUs together: per-GPU MTBF / GPUs.
export function clusterMtbfHours({ perGpuMtbfHours, gpus }) {
  requirePositiveOrInfinite('clusterMtbfHours', 'perGpuMtbfHours', perGpuMtbfHours);
  requireCount('clusterMtbfHours', 'gpus', gpus);
  return perGpuMtbfHours / gpus;
}

// First-order fraction of time lost, split by cause (frame 9): saves save/T, lost work (T/2)/MTBF, restarts
// restart/MTBF. Valid when T and restart ≪ MTBF (runPlan's `valid`). With no failures (MTBF Infinity) only saves cost.
export function checkpointLossParts({ intervalH, saveH, restartH, mtbfH }) {
  requirePositiveOrInfinite('checkpointLossParts', 'intervalH', intervalH);
  requireNonNegative('checkpointLossParts', 'saveH', saveH);
  requireNonNegative('checkpointLossParts', 'restartH', restartH);
  requirePositiveOrInfinite('checkpointLossParts', 'mtbfH', mtbfH);
  const save = saveH / intervalH;
  const failing = mtbfH !== Infinity;
  const lostWork = failing ? intervalH / 2 / mtbfH : 0;
  const restart = failing ? restartH / mtbfH : 0;
  return { save, lostWork, restart, total: save + lostWork + restart };
}

// The one definition of "lost to checkpointing and failures": checkpointLossParts(...).total.
export function checkpointLoss(args) {
  return checkpointLossParts(args).total;
}

// √(2 · save · MTBF), where checkpointLoss is smallest (Young's interval).
export function bestInterval({ saveH, mtbfH }) {
  requirePositive('bestInterval', 'saveH', saveH);
  requirePositiveOrInfinite('bestInterval', 'mtbfH', mtbfH);
  return Math.sqrt(2 * saveH * mtbfH);
}

// One training step's time in ms (ruling P3-R4: unqualified "step time" is Serving's per-iteration stepTime).
export function trainingStepMs({ computeMs, commMs, overlap }) {
  requireNonNegative('trainingStepMs', 'computeMs', computeMs);
  requireNonNegative('trainingStepMs', 'commMs', commMs);
  if (typeof overlap !== 'boolean') throw new RangeError(`trainingStepMs: overlap must be true or false, got ${overlap}`);
  return overlap ? Math.max(computeMs, commMs) : computeMs + commMs;
}

export function runCost({ gpuHours, dollarsPerGpuHour }) {
  requirePositive('runCost', 'gpuHours', gpuHours);
  requireNonNegative('runCost', 'dollarsPerGpuHour', dollarsPerGpuHour);
  return gpuHours * dollarsPerGpuHour;
}

// A whole run: mfu is MFU while training; intervalH defaults to bestInterval. gpuHours = useful / (1 − loss), and
// Infinity (with days and cost) once loss ≥ 1. valid = intervalH + restartH ≤ mtbfH / 2, the loss formula's range.
export function runPlan({ params, tokens, gpus, peakTflops, mfu, perGpuMtbfHours, saveH, restartH, intervalH, dollarsPerGpuHour }) {
  const flops = trainingFlops({ params, tokens });
  const usefulGpuHours = gpuHoursAt({ flops, peakTflops, mfu });
  const mtbfH = clusterMtbfHours({ perGpuMtbfHours, gpus });
  const interval = intervalH ?? bestInterval({ saveH, mtbfH });
  const loss = checkpointLoss({ intervalH: interval, saveH, restartH, mtbfH });
  const valid = interval + restartH <= mtbfH / 2;
  const plan = { flops, usefulGpuHours, mtbfH, intervalH: interval, loss, valid };
  if (loss >= 1) return { ...plan, gpuHours: Infinity, days: Infinity, cost: Infinity };
  const gpuHours = usefulGpuHours / (1 - loss);
  return { ...plan, gpuHours, days: wallClockDays({ gpuHours, gpus }), cost: runCost({ gpuHours, dollarsPerGpuHour }) };
}
