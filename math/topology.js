// Communication vs compute per parallelism cut on a link (cluster-topology §6). Peaks in TFLOPS, links in GB/s each way.
// Every ratio has one basis: (bytes sent per GPU in one full training step ÷ link) ÷ (that step's FLOPs per GPU ÷ peak).
// Pure: no DOM, inputs never mutated. Owned by cluster-topology; collective costs and the 6 FLOPs per parameter-token
// are imported, never redefined.
import { ringAllReduceBytes } from './parallel.js';
import { FLOPS_PER_PARAM_TOKEN } from './scale.js';

const BYTES_PER_ACTIVATION = 2; // BF16 activations and gradients
const TFLOPS_PER_GBPS = 1e12 / 1e9; // FLOP/s per TFLOPS over bytes/s per GB/s
const FULL_STEP_FLOPS_FACTOR = 3; // forward 1× + backward 2× the forward matmul FLOPs
const MATMUL_FLOPS_PER_TOKEN_UNIT = 24; // one layer's forward matmul FLOPs = 24·s·b·h²
const TP_ALLREDUCES_PER_STEP = 4; // 2 forward + 2 backward per layer
const PP_HANDOFFS_PER_STEP = 2; // one activation forward + one gradient backward per boundary
const DEFAULT_EP_FLOPS_PER_BYTE = 6144; // DeepSeek-V4's hiding condition, 2d (data/models.json deepseek-v4-pro.ep_hiding_flops_per_byte)

function requireCount(fn, name, value) {
  if (!Number.isInteger(value) || value < 1) throw new RangeError(`${fn}: ${name} must be a positive integer, got ${value}`);
}

function requirePositive(fn, name, value) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    throw new RangeError(`${fn}: ${name} must be a positive finite number, got ${value}`);
  }
}

// seconds to send `bytes` at linkGBps, over seconds to do `flops` at peakTflops
function ratio(bytes, linkGBps, flops, peakTflops) {
  return (bytes / (linkGBps * 1e9)) / (flops / (peakTflops * 1e12));
}

// One layer, one full step, per token: 4 ring all-reduces of h BF16 numbers vs 72·h²/t FLOPs (s and b cancel).
export function tpCommRatio({ tp, hidden, peakTflops, linkGBps }) {
  requireCount('tpCommRatio', 'tp', tp);
  requireCount('tpCommRatio', 'hidden', hidden);
  requirePositive('tpCommRatio', 'peakTflops', peakTflops);
  requirePositive('tpCommRatio', 'linkGBps', linkGBps);
  const bytes = TP_ALLREDUCES_PER_STEP * ringAllReduceBytes(hidden * BYTES_PER_ACTIVATION, tp);
  const flops = (FULL_STEP_FLOPS_FACTOR * MATMUL_FLOPS_PER_TOKEN_UNIT * hidden * hidden) / tp;
  return ratio(bytes, linkGBps, flops, peakTflops);
}

// One micro-batch at one stage boundary: 2 hand-offs of h numbers per token vs the stage's layers/pp layers of FLOPs.
export function ppCommRatio({ pp, hidden, layers, peakTflops, linkGBps }) {
  requireCount('ppCommRatio', 'pp', pp);
  requireCount('ppCommRatio', 'hidden', hidden);
  requireCount('ppCommRatio', 'layers', layers);
  requirePositive('ppCommRatio', 'peakTflops', peakTflops);
  requirePositive('ppCommRatio', 'linkGBps', linkGBps);
  const bytes = PP_HANDOFFS_PER_STEP * hidden * BYTES_PER_ACTIVATION;
  const flops = (FULL_STEP_FLOPS_FACTOR * MATMUL_FLOPS_PER_TOKEN_UNIT * hidden * hidden * layers) / pp;
  return ratio(bytes, linkGBps, flops, peakTflops);
}

// One step, per parameter: a ring all-reduce of its BF16 gradient vs FLOPS_PER_PARAM_TOKEN × tokens per replica.
export function dpCommRatio({ dp, tokensPerReplica, peakTflops, linkGBps }) {
  requireCount('dpCommRatio', 'dp', dp);
  requireCount('dpCommRatio', 'tokensPerReplica', tokensPerReplica);
  requirePositive('dpCommRatio', 'peakTflops', peakTflops);
  requirePositive('dpCommRatio', 'linkGBps', linkGBps);
  const bytes = ringAllReduceBytes(BYTES_PER_ACTIVATION, dp);
  return ratio(bytes, linkGBps, FLOPS_PER_PARAM_TOKEN * tokensPerReplica, peakTflops);
}

// DeepSeek-V4's rule: expert traffic hides when FLOPs per byte of link ≤ flopsPerByte; a ratio of 1 is the line.
export function epCommRatio({ peakTflops, linkGBps, flopsPerByte = DEFAULT_EP_FLOPS_PER_BYTE }) {
  requirePositive('epCommRatio', 'peakTflops', peakTflops);
  requirePositive('epCommRatio', 'linkGBps', linkGBps);
  requirePositive('epCommRatio', 'flopsPerByte', flopsPerByte);
  return (peakTflops * TFLOPS_PER_GBPS) / linkGBps / flopsPerByte;
}

// The slowest link (GB/s each way) that still hides expert traffic at this peak.
export function epMinLinkGBps({ peakTflops, flopsPerByte = DEFAULT_EP_FLOPS_PER_BYTE }) {
  requirePositive('epMinLinkGBps', 'peakTflops', peakTflops);
  requirePositive('epMinLinkGBps', 'flopsPerByte', flopsPerByte);
  return (peakTflops * TFLOPS_PER_GBPS) / flopsPerByte;
}
