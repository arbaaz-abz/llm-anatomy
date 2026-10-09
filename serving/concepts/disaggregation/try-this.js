// Storyboard §6 "Try this": three prompts, each leading to a named insight. Every number is computed from format.js (math/ functions
// over data/*.json), printed at the storyboard's prose precision. Pure, no DOM. Controls are named by their visible labels.
import { formatBytes, formatCount, formatDuration, formatInt } from '@math/core.js';
import { INITIAL_STATE, shipAnalysis, expertAnalysis, pct1, prefillCrossover } from './format.js';

const ship = (setup, state) => shipAnalysis({ ...INITIAL_STATE, ...state }, setup);
const experts = (setup, state) => expertAnalysis({ ...INITIAL_STATE, ...state }, setup);
const ms = (s) => formatDuration(s);
const ratio = (a) => pct1(a.transferS, a.prefillS);

function first(setup) {
  const [p512, p4096, p131k, p128] = [512, 4096, 131072, 128].map((prompt) => ship(setup, { prompt }));
  return {
    prompt: `Set Link between pools to network 400 Gb/s and KV cache to BF16. Slide Prompt length 512 → 4,096 → 131,072: transfer ${ms(p512.transferS)} → ${ms(p4096.transferS)} → ${ms(p131k.transferS)}, prefill ${ms(p512.prefillS)} → ${ms(p4096.prefillS)} → ${ms(p131k.prefillS)}, ratio ${ratio(p4096)} every time. `
      + `Now slide down to 128: transfer ${ms(p128.transferS)}, prefill ${ms(p128.prefillS)}, ratio ${ratio(p128)}, and a note beside the slider says why: below ${formatInt(prefillCrossover())} tokens a prefill is one weight read, and real transfers add a fixed start-up cost the toy does not model.`,
    insight: 'past the crossover, the transfer-to-prefill ratio is set by the model and the link, not the prompt.',
    rest: ' Long prompts do not make disaggregation worse in this model; slow links do.',
  };
}

function second(setup) {
  const links = ['net400', 'net800', 'nvlink'];
  const [bf, fp8] = ['bf16', 'fp8'].map((kv) => links.map((link) => ratio(ship(setup, { link, kv }))));
  const mla = ship(setup, { link: 'net400' }).mla;
  return {
    prompt: `Keep Prompt length at 4,096 tokens and switch Link between pools: network 400 Gb/s ${bf[0]} · network 800 Gb/s ${bf[1]} · NVLink5 ${bf[2]}. Then set KV cache to FP8, which halves each: ${fp8.join(', ')}. `
      + `The MLA readout, at 400 Gb/s with BF16 KV: ${formatBytes(mla.kvBytes)}, ${ms(mla.transferS)}.`,
    insight: 'a fast link or a smaller KV makes the split nearly free;',
    rest: ' this is one more reason MLA and FP8 KV matter ([[kv-compression]], [[quantization]]).',
  };
}

function third(setup) {
  const at = (ep, users = 64) => experts(setup, { ep, users });
  const [e1, e8, e16, e72] = [1, 8, 16, 72].map((ep) => at(ep));
  const big = at(72, 256);
  return {
    prompt: `Set Users per GPU to 64 and slide GPUs sharing the experts (EP size) from 1 to 72. At 1 the free memory reads "does not fit" (${formatBytes(e1.weightsBytes)} of weights against ${formatBytes(e1.weightsBytes + e1.freeBytes)} of memory) and each expert gets ${formatCount(e1.tokens)} token; `
      + `EP 8 gives ${formatCount(e8.tokens)} tokens per expert and ${formatBytes(e8.weightsBytes)} of weights per GPU; EP 16, ${formatCount(e16.tokens)} and ${formatBytes(e16.weightsBytes)}; EP 72, ${formatCount(e72.tokens)}, ${formatBytes(e72.weightsBytes)} and ${formatBytes(e72.freeBytes)} free.`,
    insight: 'wide EP both shrinks each GPU\'s share of the weights and pools users\' tokens at each expert,',
    rest: ` so every expert does more math per byte read. Even at EP 72 with 256 users per GPU (${formatCount(big.tokens)} tokens per expert) the intensity is ${formatCount(big.intensity)}, still under the FP4 ridge of ${formatInt(big.ridge)}, which an expert reaches only at ${formatInt(Math.ceil(big.tokensNeeded))} tokens.`,
  };
}

// [{ prompt, insight, rest }]; the page prints "prompt → Insight: insight rest", with [[slug]] as the lesson's title.
export const tryThis = (setup) => [first(setup), second(setup), third(setup)];
