// Storyboard §6 "Try this": four prompts, each leading to a named insight. Every number is computed through analyze
// (format.js) from data/hardware.json and math/serving.js, printed by the formatter its readout uses. Pure, no DOM.
import { formatCount, formatDuration, formatInt, formatBytes } from '@math/core.js';
import { gpuPreset } from './hardware.js';
import { analyze, INITIAL_STATE } from './format.js';
import { H100_CROSSING } from './model.js';

const at = (data, patch) => {
  const state = { ...INITIAL_STATE, ...patch };
  return analyze(state, gpuPreset(data, state.hw));
};
const prefillAt = (data, promptTokens, patch = {}) => at(data, { phase: 'prefill', promptTokens, ...patch });
const chain = (list, f) => list.map(f).join(' → ');
const time = (a) => formatDuration(a.step.timeS);

function first(data) {
  const [p217, p1000, p8192] = [217, 1000, 8192].map((t) => prefillAt(data, t));
  const flip = formatInt(p217.crossing);
  const bf16 = formatInt(prefillAt(data, 217, { weights: 'bf16' }).crossing);
  const b200 = [prefillAt(data, 217, { hw: 'b200', weights: 'bf16' }), prefillAt(data, 217, { hw: 'b200' })].map((a) => formatInt(a.crossing));
  const b200Text = b200[0] === b200[1] ? `it moves to ${b200[0]}, in BF16 as in FP8` : `it moves to ${b200[0]} in BF16 and ${b200[1]} in FP8`;
  return {
    prompt: `Set Phase to prefill (GPU H200, Weight format FP8) and slide Prompt tokens from 1 to 8,192, watching the bound: memory-bound up to ${flip} tokens, compute-bound from there (${time(p217)} at 217; ${time(p1000)} at 1,000; ${time(p8192)} at 8,192). `
      + `Switch Weight format to BF16: the flip stays at ${bf16}. Switch GPU to B200: ${b200Text}.`,
    insight: 'the crossover, counted in tokens per weight read, is set by the GPU\'s ratio of math to bandwidth.',
    rest: ` FP8 halves the bytes and doubles the math rate, so on an H200 it moves the ridge but not the token crossover (the same point made for the H100's ${formatInt(H100_CROSSING)} in [[gpu-primer]]).`,
  };
}

function second(data) {
  const maxUsers = at(data, {}).maxUsers;
  const rows = [1, 8, 64, maxUsers].map((users) => at(data, { users }));
  return {
    prompt: `Set Phase to decode with Context per user at 2,048 and slide Users in the batch 1 → 8 → 64 → max that fits: per user ${chain(rows, (a) => formatCount(1 / a.step.timeS))} tok/s; per GPU ${chain(rows, (a) => formatInt(a.users / a.step.timeS))} tok/s.`,
    insight: 'batching trades each user\'s speed for the GPU\'s total,',
    rest: ' cheaply at first (one shared read of the weights), then more steeply as KV reads grow.',
  };
}

function third(data) {
  const rows = [2048, 8192, 32768, 131072].map((context) => at(data, { context, users: Infinity }));
  return {
    prompt: `Keep Users in the batch at max that fits and slide Context per user 2,048 → 8,192 → 32,768 → 131,072: max users ${chain(rows, (a) => formatInt(a.maxUsers))}; tokens per GPU ${chain(rows, (a) => formatInt(a.users / a.step.timeS))}.`,
    insight: 'at long context the KV cache, not the weights, decides how many users a GPU can hold,',
    rest: ' and with them its throughput. That is the problem [[paged-attention]], [[kv-compression]] and FP8 KV caches (see [[quantization]]) attack.',
  };
}

function fourth(data) {
  const a = at(data, { weights: 'bf16' });
  return {
    prompt: `Switch Weight format to BF16 on the H200 at Context per user 2,048: ${formatBytes(a.free)} is left and only ${formatInt(a.maxUsers)} user fits (${formatInt(a.users / a.step.timeS)} tok/s per GPU).`,
    insight: 'the weight format decides whether there is room for users at all.',
    rest: ' You\'ll see what FP8 and FP4 cost in quality in [[quantization]].',
  };
}

// [{ prompt, insight, rest }]; the page prints "prompt → Insight: insight rest", with [[slug]] as the lesson's title.
export const tryThis = (data) => [first(data), second(data), third(data), fourth(data)];
