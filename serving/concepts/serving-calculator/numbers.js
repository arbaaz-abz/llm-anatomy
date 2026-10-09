// serving-calculator's hand-authored stage constants (storyboard §4–§5). The stage renders without ctx.data, so the worked
// example's numbers it draws are restated here; tests/serving-calculator-page.test.js asserts each equals data/*.json
// (the P3-R13 pattern: a page constant that restates data carries an equality test). Everything else the stage prints is
// computed from these through math/serving.js and friends.
import { deepFreeze } from '@math/core.js';

// DeepSeek-V4-Pro (models.json): 1.6T total / 49B active, d_model 7,168, 384 routed experts with 6 per token.
export const V4 = deepFreeze({
  label: 'DeepSeek-V4-Pro', totalParams: 1.6e12, activeParams: 49e9, dModel: 7168, experts: 384, expertsPerToken: 6,
  checkpointBytes: 865e9, // reported (models.json deepseek-v4-pro.checkpoint_gb)
  kvBytesPerToken: { low: 4000, high: 12000 }, // reported range (models.json deepseek-v4-pro.kv_bytes_per_token)
});

// The followed GPU: GB300 NVL72 (hardware.json gb300-nvl72). 288 GB is nominal (no usable figure in data); FP8 is the "as shipped" math rate.
export const GB300 = deepFreeze({ label: 'GB300', hbmBytes: 288e9, bandwidthTBps: 8, fp8Tflops: 5000 });
export const GB200_HBM_BYTES = 186e9; // hardware.json gb200-nvl72.hbm_gb: the rack total over 72 GPUs, nominal
export const EP_SIZE = 16; // GPUs per replica, experts spread evenly (expert parallelism)

// Tokens per user (input + output) the frames use: 8K in + 1K out, 128K in + 8K out, and V4-Pro's configured 1M.
export const CONTEXT = deepFreeze({ short: 8192 + 1024, mid: 131072 + 8192, long: 1_000_000 });
export const TARGET_TOK_S_USER = 27; // InferenceX interactivity point (serving.json inferencex-v4-pro-gb300.interactivity_tok_s_user)
export const ISL_OSL = deepFreeze({ input: 8192, output: 1024 });

// InferenceX, measured 2026-05-22 (serving.json): throughput per GPU at an operating point, and the GPU-hour price.
export const MEASURED = deepFreeze({
  gb300: { tokSGpu: 6182, tokSUser: 27, peakTokSGpu: 11056, peakTokSUser: 13.1, usdPerGpuHour: 2.65 },
  gb200: { tokSGpu: 2189, tokSUser: 27, usdPerGpuHour: 2.21 },
});
// DeepSeek V3/R1 production, February 2025 (serving.json deepseek-v3-production): tokens per second per H800 node.
export const PRODUCTION = deepFreeze({ inputNodeTokS: 73700, outputNodeTokS: 14800 });
// List prices read 2026-10-07 (serving.json pricing-*): DeepSeek V4-Pro off-peak, Anthropic's output-to-input ratio.
export const LIST_PRICES = deepFreeze({ inputUsd: 0.66, outputUsd: 1.98, anthropicRatio: 5 });

// The step model of one GB300 in the V4-Pro replica: its own users' tokens, 1/16 of the weights (storyboard §6 `V`).
export const STAGE_MODEL = deepFreeze({
  activeParamsPerGpu: V4.activeParams, weightBytesPerGpu: V4.checkpointBytes / EP_SIZE, dModel: V4.dModel, actBytesPerElem: 1,
  peakTflops: GB300.fp8Tflops, bandwidthTBps: GB300.bandwidthTBps,
});

// Dates the stage prints beside measured and list numbers; the page test checks each against data/serving.json.
export const MEASURED_DATE = '2026-05-22'; // inferencex-v4-pro-gb300.date
export const LIST_PRICES_DATE = '2026-10-07'; // pricing-*.date
export const PRODUCTION_DATE = 'Feb 2025'; // deepseek-v3-production.date ("2025-02") printed as the |date claim format does
