// parallelism's hand-authored stand-in numbers (storyboard §4–§5) and what follows from them. Pure, no DOM.
// Everything else on the stage comes from math/parallel.js, math/training-memory.js, math/moe.js or math/params.js.
import { deepFreeze } from '@math/core.js';
import { ROUTER_TOY, routeTopK } from '@math/moe.js';
import { paramBreakdown, PRESETS } from '@math/params.js';
import { ringAllReduceBytes } from '@math/parallel.js';

export const TOKENS = Object.freeze(['The', 'cat', 'sat', 'down']);
export const SAT = 2; // the row followed in every frame
export const BYTES_PER_NUMBER = 2; // BF16
export const D_MODEL = 8;
export const MLP_HIDDEN = 16;
export const TOY = paramBreakdown(PRESETS.toy); // total 1,576; perLayer.mlp 384 (the course toy)

// Data parallelism (frame 2): two GPUs, the whole toy model on each.
export const DP_GPUS = 2;
export const GRADIENT_BYTES = TOY.total * BYTES_PER_NUMBER; // 3,152 B
export const DP_SENT_BYTES = ringAllReduceBytes(GRADIENT_BYTES, DP_GPUS); // 3,152 B per GPU
export const GPT3_GRADIENT_BYTES = 350e9; // 175e9 parameters × 2 B (tests/parallelism-page.test.js ties it to data/models.json)
export const GPT3_DP_GPUS = 64;

// Tensor parallelism (frames 3–4): "sat" through the MLP's second matrix. GPU 1's partial sum is a hand-picked stand-in
// on the quarter grid; GPU 2's is what is left of decoder-anatomy's MLP output for sat (checked to equal its M_SAT).
export const MLP_OUT_SAT = deepFreeze([0, 0.25, 0, -0.5, 0.5, 0, -0.25, 0.25]);
export const PARTIAL_GPU1 = deepFreeze([0.25, 0.5, -0.25, -0.25, 0.25, 0.25, 0, 0]);
export const PARTIAL_GPU2 = deepFreeze(MLP_OUT_SAT.map((v, i) => v - PARTIAL_GPU1[i] + 0)); // + 0 turns −0 into 0
export const TP_OUTPUT_BYTES = TOKENS.length * D_MODEL * BYTES_PER_NUMBER; // a [4 × 8] partial output: 64 B
export const TP_SENT_BYTES = ringAllReduceBytes(TP_OUTPUT_BYTES, 2); // 64 B per GPU
export const MLP_PARAMS_PER_GPU = TOY.perLayer.mlp / 2; // 192 of 384

// Pipeline parallelism (frames 5–8): the stand-in 6-block model, 3 stages × 2 blocks, 4 micro-batches.
export const PIPE = deepFreeze({ stages: 3, blocksPerStage: 2, microBatches: 4 });
export const HANDOFF_BYTES = TOKENS.length * D_MODEL * BYTES_PER_NUMBER; // 64 B per hand-off, forward and backward
export const MORE_MICRO = 8; // frame 8's "with 8 micro-batches" line

// Context parallelism (frame 9): K and V of a token, 8 numbers each.
export const KV_NUMBERS_PER_TOKEN = 2 * D_MODEL; // 16
export const KV_BYTES_PER_TOKEN = KV_NUMBERS_PER_TOKEN * BYTES_PER_NUMBER; // 32 B
export const CP_TOKENS_SENT = 2; // The and cat travel from GPU 1 to GPU 2
export const CP_SENT_BYTES = CP_TOKENS_SENT * KV_BYTES_PER_TOKEN; // 64 B per layer

// Expert parallelism (frame 10): moe's ROUTER_TOY, top-2 of 8 experts, two experts per GPU, one token per GPU.
export const EP_EXPERTS = 8;
export const EP_GPUS = 4;
export const EXPERTS_PER_GPU = EP_EXPERTS / EP_GPUS;
export const EP_COPY_BYTES = D_MODEL * BYTES_PER_NUMBER; // one token copy: 16 B
export const ROUTES = deepFreeze(ROUTER_TOY.map((row) => routeTopK(row, 2))); // 0-based expert indices per token
export const gpuOfExpert = (expert) => Math.floor(expert / EXPERTS_PER_GPU); // 0-based GPU
export const copyCrosses = (token, expert) => gpuOfExpert(expert) !== token; // token t lives on GPU t
export const CROSSING = deepFreeze(ROUTES.flatMap((experts, token) => experts.filter((e) => copyCrosses(token, e)).map((e) => ({ token, expert: e }))));
export const COPIES_PER_GPU = deepFreeze(Array.from({ length: EP_GPUS }, (_, g) => ROUTES.flat().filter((e) => gpuOfExpert(e) === g).length));
export const EP_DISPATCH_BYTES = CROSSING.length * EP_COPY_BYTES; // 64 B out (and 64 B back)

// Real runs (frame 11, §8): the degrees as published; tests tie them to data/models.json.
export const LLAMA = deepFreeze({ params: 405e9, tp: 8, cp: 1, pp: 16, dp: 64, zeroStage: 2, gpus: 8192 });
export const LLAMA_PRESETS = deepFreeze({
  '8k': { tp: 8, cp: 1, pp: 16, dp: 64 },
  '16k': { tp: 8, cp: 1, pp: 16, dp: 128 },
  long: { tp: 8, cp: 16, pp: 16, dp: 8 },
});
export const DEEPSEEK_V3 = deepFreeze({ pp: 16, ep: 64, gpus: 2048 });
export const WEIGHT_BYTES_PER_PARAM = 16; // Adam, BF16 weights and gradients, FP32 master and moments
