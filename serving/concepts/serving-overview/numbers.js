// serving-overview's hand-authored stand-ins (storyboard §4–§6) and every stage number computed from them. Pure, no DOM.
// Times come from math/serving.js (the step-time floor of Prefill vs decode, RUNNING_EXAMPLE: Llama-3.1-70B FP8 on one H200);
// frame 9 replays math/batching.js on the course's TOY_REQUESTS. The page test pins the constants to data/*.json.
import { deepFreeze } from '@math/core.js';
import { stepTime, requestTimeline, RUNNING_EXAMPLE, TOY_REQUESTS } from '@math/serving.js';
import { simulateContinuous } from '@math/batching.js';

// ---- your request (frames 1–8; P4-R7: no letter, no --req hue, G.selectionMark) ----
export const PROMPT = Object.freeze(['The', 'cat', 'sat']); // positions 1–3
export const ANSWER = Object.freeze(['down', '.', '⟨end⟩']); // positions 4–6; the end token is never fed back
export const DECODE_CONTEXT = 2048; // the context both decode-speed chips assume (storyboard §6)
export const CHAT = deepFreeze({ prompt: 2000, output: 500 }); // the toy's default request, quoted on frame 5

// One user's decode speed: every user gets one token per step, so tokens/s per user = 1 ÷ the step time.
export function decodeRate(users, context = DECODE_CONTEXT) {
  return 1 / stepTime({ ...RUNNING_EXAMPLE, tokens: users, seqs: users, context }).timeS;
}
export const prefillSeconds = (promptTokens) => stepTime({ ...RUNNING_EXAMPLE, tokens: promptTokens, seqs: 0, context: 0 }).timeS;

export const RATE_ALONE = decodeRate(1); // 67.90… tok/s
export const PREFILL_S = prefillSeconds(PROMPT.length); // 14.59 ms: one read of the weights
export const TPOT_S = 1 / RATE_ALONE; // 14.73 ms, the TPOT frame 7 prints and the tick spacing it draws
export const CHAT_TIMELINE = requestTimeline({ promptTokens: CHAT.prompt, model: RUNNING_EXAMPLE, outputTokens: CHAT.output, decodeTokPerS: RATE_ALONE });

// The stage's GPU memory bar: the running example's weights in an H200's nominal HBM (hardware.json h200.hbm_gb).
export const H200_NOMINAL_BYTES = 141e9;
export const WEIGHTS_FILL = RUNNING_EXAMPLE.weightBytesPerGpu / H200_NOMINAL_BYTES;

// ---- frame 9: the course's four requests A–D (TOY_REQUESTS), three seats, continuous batching ----
export const SEATS = 3;
export const BATCH_STEP = 4; // the step the marker lands on
export const BATCH = simulateContinuous({ requests: TOY_REQUESTS, seats: SEATS });
export { TOY_REQUESTS };

// Who advances at step s (a decode step: admitted < s ≤ finishes) and who is already done (finishes < s).
export function batchAt(step) {
  const advancing = BATCH.live.filter((r) => r.admitted < step && step <= r.finishes).map((r) => r.id);
  const done = BATCH.live.filter((r) => r.finishes < step).map((r) => r.id);
  return { advancing, done };
}

// ---- frame 10: each stop and the lessons that speed it up (titles from shared/concepts.json, pinned by the page test) ----
export const LESSON_MAP = deepFreeze([
  { stop: 'router', slugs: ['prefix-caching'], titles: ['Prefix caching'] },
  { stop: 'scheduler', slugs: ['batching'], titles: ['Continuous batching'] },
  { stop: 'prefill and decode', slugs: ['prefill-decode'], titles: ['Prefill vs decode'] },
  { stop: 'KV cache', slugs: ['paged-attention', 'prefix-caching'], titles: ['PagedAttention', 'Prefix caching'] },
  { stop: 'decode loop', slugs: ['speculative-decoding'], titles: ['Speculative decoding'] },
  { stop: 'weights', slugs: ['quantization'], titles: ['Quantization'] },
  { stop: 'many GPUs', slugs: ['disaggregation'], titles: ['Disaggregated serving'] },
  { stop: 'all of it', slugs: ['serving-calculator'], titles: ['Serving a 1T model'] },
]);
export const TITLES = deepFreeze({ prefixCaching: 'Prefix caching', batching: 'Continuous batching', sampling: 'Picking the next token' });
