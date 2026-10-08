// scale-reliability's hand-authored constants (storyboard §5, §6): the published runs the stage draws, the stand-in
// reliability numbers, and the toy's slider stops. Each published figure is pinned to data/*.json by the page test
// (ruling P3-R13), because a stage frame cannot read ctx.data. Pure, no DOM.
import { deepFreeze, mulberry32 } from '@math/core.js';

// Llama 3.1 405B (the followed run): Meta's paper and model card.
export const LLAMA = deepFreeze({
  params: 405e9,
  tokens: 15.6e12,
  gpus: 16384,
  gpuHours: 30.84e6, // model card; covers more stages than 6ND over 15.6T tokens counts
  mfuBand: [0.38, 0.43], // MFU while training (paper)
  effective: 0.9, // ">90%" effective training time, as a floor
  interruptions: 419,
  windowDays: 54,
  hardwareShare: 0.78,
  gpuShare: 0.587,
  toyMfu: 0.4, // the toy preset's MFU while training (inside the paper's band)
});

// DeepSeek-V3: the FP8 and cost example (H800s; H800 peaks are not in data, so H100 peaks are the stand-in).
export const DEEPSEEK = deepFreeze({
  params: 37e9, // active
  tokens: 14.8e12,
  gpus: 2048,
  pretrainGpuHours: 2.664e6,
  totalGpuHours: 2.788e6,
  dollarsPerGpuHour: 2,
  commSms: 20,
  smsPerGpu: 132,
  toyMfu: 0.35637, // back-solved so runPlan returns the reported 2.664M pre-training GPU-hours (storyboard §6, §13)
});

export const BEHEMOTH = deepFreeze({ tflopsPerGpu: 390 });
export const H100 = deepFreeze({ bf16: 989, fp8: 1979 }); // dense TFLOPS

// Stand-ins chosen so the Llama preset lands near its reported >90% effective time (labeled on screen).
export const STAND_IN = deepFreeze({ computeMs: 100, commMs: 40, saveS: 30, restartMin: 3 });

// Llama 3.1's 419 interruptions in 54 days across 16,384 GPUs → one GPU's MTBF (hours), 50,677 h.
export const PER_GPU_MTBF_H = LLAMA.windowDays * 24 / LLAMA.interruptions * LLAMA.gpus;

export const BIG_GPUS = 100000; // frame 10's second run; its total is the full bar width on frames 2, 3 and 10 (runs.js)
export const HOURS_PER_YEAR = 24 * 365.25;
export const FAILURE_SEED = 54; // mulberry32 seed of frame 8's illustrative failure positions (their count, 419, is the paper's)

// Frame 9: a 90-minute window, one failure shortly after a save (illustrative position; the 13.6 min interval is computed).
export const WINDOW_MIN = 90;
export const FAILURE_AT_MIN = 73;

// The toy's snapped slider stops (log sliders: README visual constraints; every preset and try-this value is a stop).
export const STOPS = deepFreeze({
  params: [1e9, 2e9, 3e9, 5e9, 7e9, 10e9, 20e9, 30e9, 37e9, 50e9, 70e9, 100e9, 200e9, 300e9, 405e9, 500e9, 700e9, 1e12],
  tokens: [100e9, 200e9, 500e9, 1e12, 2e12, 5e12, 10e12, 14.8e12, 15.6e12, 20e12, 30e12, 50e12],
  gpus: [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 100000, 131072, 200000],
  perGpuMtbfH: [10000, 20000, 30000, 40000, 50677, 70000, 100000, 150000, 200000],
  saveS: [5, 10, 15, 20, 30, 45, 60, 120, 300, 600],
  restartMin: [1, 2, 3, 5, 10, 15, 20, 30],
  intervalMin: [1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240],
});
export const MFU_RANGE = Object.freeze({ min: 15, max: 60, step: 0.1 }); // percent
export const PRICE_RANGE = Object.freeze({ min: 1, max: 6, step: 0.5 }); // dollars per GPU-hour

// DeepSeek-V3's FP8 recipe (frame 6), pinned to models.json/deepseek-v3.fp8_loss_error by the page test.
export const FP8 = deepFreeze({ lossError: '< 0.25%', kept: 'embeddings, output head, gating, norms, attention' });

// Frame 8: where the illustrative failures land. Positions come from mulberry32(54), sorted, in days; the count (419) is
// the paper's. Two racks each show one failed GPU, picked from the same seed.
const failureRandom = mulberry32(FAILURE_SEED);
export const FAILURE_DAYS = Object.freeze(Array.from({ length: LLAMA.interruptions }, () => failureRandom() * LLAMA.windowDays).sort((a, b) => a - b));
export const FAILED_GPUS = Object.freeze([Math.floor(failureRandom() * 8), Math.floor(failureRandom() * 8)]);
export const RACK_GPUS = 8;
export const OTHER_NODES = LLAMA.gpus / RACK_GPUS - 2; // 2,046 nodes besides the two drawn
