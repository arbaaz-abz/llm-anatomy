// Storyboard §6 "Try this": four prompts, each leading to a named insight. Every number is computed through model.js (the toy's own
// functions) from data/*.json, printed at the toy's formats. Controls are named by their visible labels (XT-2). Pure, no DOM.
import { formatDuration, formatRatio } from '@math/core.js';
import { compute } from './model.js';
import { INITIAL_STATE, int, rate, speedupText } from './format.js';
import { PRODUCTION } from './numbers.js';

const LONG = 1_000_000;
const MID = 139264;
const at = (data, patch) => compute({ ...INITIAL_STATE, ...patch }, data);
const kvFit = (data, patch) => int(at(data, patch).chosen.maxFit);

function first(data) {
  const [short, mid, long] = [9216, MID, LONG].map((context) => at(data, { context }));
  const high = [9216, MID, LONG].map((context) => at(data, { context, kvEnd: 'high' }));
  const gb200 = at(data, { context: LONG, hw: 'gb200-nvl72' });
  const floorFull = at(data, { context: LONG }).stepModel;
  return {
    prompt: `Keep the defaults, then switch Tokens per user (in + out) from 8K + 1K to 128K + 8K to 1M (low KV): users that fit ${int(short.maxFit)} → ${int(mid.maxFit)} → ${int(long.maxFit)}; `
      + `at the ${INITIAL_STATE.target} tok/s target the limit changes from "${short.target.limit}" (${int(short.users)} users) to "${mid.target.limit}" (${int(mid.users)}, then ${int(long.users)}); `
      + `floor tokens/s per GPU ${int(short.tokSGpu)} → ${int(mid.tokSGpu)} → ${int(long.tokSGpu)}. Flip V4 KV estimate to high: ${high.map((h) => int(h.maxFit)).join(' → ')}. `
      + `Then pick GB200 NVL72 under GPU at 1M with 16 GPUs: ${int(gb200.maxFit)} users instead of ${int(long.maxFit)}.`,
    insight: 'at long context, memory for KV, not compute, sets how many users a GPU serves,',
    rest: ` and when memory is full every step re-reads it, so per-user speed tops out near HBM size over bandwidth (${int(short.gpu.hbmBytes / 1e9)} GB at ${floorFull.bandwidthTBps} TB/s, about ${formatDuration(short.gpu.hbmBytes / (floorFull.bandwidthTBps * 1e12))} per step).`,
  };
}

function second(data) {
  const d = at(data, {});
  const one = at(data, { users: 1 });
  const m = d.scenario.measured.gb300.tokSGpu;
  const outputOnly = m / 9;
  return {
    prompt: `Defaults: the floor is ${int(d.tokSGpu)} output tokens/s per GPU at ${INITIAL_STATE.target} tok/s per user; the measured row says ${int(m)} tokens/s per GPU, possibly counting input too, `
      + `so the gap is a range: ${formatRatio(d.tokSGpu / m)} to ${formatRatio(d.tokSGpu / outputOnly)}. Now set Users per GPU to 1: ${rate(one.tokSUser)} tok/s per user and per GPU.`,
    insight: 'the roofline is a ceiling, not a forecast.',
    rest: ` It tells you which wall is closest (here: math at 8K, memory at 1M), while the measured ${INITIAL_STATE.target} tok/s per user implies ${formatDuration(1 / INITIAL_STATE.target)} per token, `
      + `${formatRatio(1 / INITIAL_STATE.target / (d.stepModel.weightBytesPerGpu / (d.stepModel.bandwidthTBps * 1e12)))} the weight-read floor; plan with measurements and use the floor to reason about changes.`,
  };
}

function third(data) {
  const d = at(data, {});
  const [mid, long] = [MID, LONG].map((context) => at(data, { context }));
  const hit = at(data, { hit: d.scenario.hitRate });
  return {
    prompt: `Input side at defaults: prefill ceiling ${int(d.prefillTokS)} tok/s per GPU, the same as decode's ideal at 8K, so the floor's output ÷ input cost ratio is ${formatRatio(d.costRatio)}, and a visible line says why. `
      + `Switch to 128K: ${formatRatio(mid.costRatio)}; at 1M: ${formatRatio(long.costRatio)}. Turn Prompt cache hit rate to ${Number((d.scenario.hitRate * 100).toFixed(1))}%: input work per token falls to ${Number(((1 - d.scenario.hitRate) * 100).toFixed(1))}%, `
      + `so input cost per token falls ${formatRatio(d.inputCost / hit.inputCost)}.`,
    insight: 'in the ideal, output costs more only when KV memory caps the decode batch;',
    rest: ` in practice decode never reaches the ideal, measured nodes read ${formatRatio(PRODUCTION.inputNodeTokS / PRODUCTION.outputNodeTokS)} more input than output tokens (step 10 of the animation), and cache hits help only the input side.`,
  };
}

function fourth(data) {
  const gain = at(data, {}).scenario.lmsysGainPct;
  const [at256, atTarget, long128] = [at(data, { mtp: true, users: 256 }), at(data, { mtp: true }), at(data, { mtp: true, context: MID })];
  return {
    prompt: `Turn MTP speculation (k = 1, α = ${at(data, {}).scenario.mtpAlpha}) on. At 256 users and 8K: ${speedupText(at256.mtp.speedup)}. At "most at target" (${int(atTarget.users)} users, ${atTarget.step.bound}-bound): MTP slows each user down to ${speedupText(atTarget.mtp.speedup)}. `
      + `At 128K with ${int(long128.users)} users: ${speedupText(long128.mtp.speedup)}.`,
    insight: 'MTP pays where decode is memory-bound,',
    rest: ` as LMSYS measured at 128K on GB300 and the batch curve in [[speculative-decoding]] shows. The toy prints LMSYS's record beside its own number: measured +${gain}% (${formatRatio(1 + gain / 100)}, DeepSeek-R1, a different model) vs the floor's ${speedupText(long128.mtp.speedup)}, `
      + `which uses α = ${at(data, {}).scenario.mtpAlpha}, the low end of DeepSeek\'s reported ${at(data, {}).scenario.mtpRange.join(' to ')}% acceptance.`,
  };
}

// [{ prompt, insight, rest }]; the page prints "prompt → Insight: insight rest", with [[slug]] as the lesson's title.
export const tryThis = (data) => [first(data), second(data), third(data), fourth(data)];
