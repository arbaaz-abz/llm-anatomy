// serving-calculator model (pure, no DOM): a toy state + data → every number the toy prints, from math/ functions only.
// One definition per metric (README lesson 16): users per GPU = sequences in one GPU's decode batch; tokens/s per GPU =
// users ÷ step time, output tokens only; KV per user = KV bytes per token × context (math/serving.js, math/memory.js).
import {
  stepTime, freeHbmPerGpu, maxUsersPerGpu, prefillTokPerSecCeiling, minGpusForWeights, usersAtTarget, costPerMillion,
} from '@math/serving.js';
import { batchSpeedup } from '@math/specdec.js';
import { kvCacheBytes } from '@math/memory.js';
import { modelPreset, gpuPreset, scenarioPresets, kvBytesFor, weightsFor, mathPeak, supportsWeights, WEIGHT_FORMATS, V4_ID, NO_FP4 } from './presets.js';

export const MTP = Object.freeze({ k: 1, c: 0.05 }); // one guessed token, a draft step that costs 5% of a plain step (speculative-decoding)
const EPS = 1e-9;

// The KV rows: V4-Pro shows both ends of its reported range, low first; Llama one row.
function kvRows(model, state, free) {
  const ends = model.dense ? [{ end: null, bytesPerToken: kvBytesFor(model, state) }]
    : [{ end: 'low', bytesPerToken: model.kvBytes.low }, { end: 'high', bytesPerToken: model.kvBytes.high }];
  return ends.map(({ end, bytesPerToken }) => {
    const perUser = kvCacheBytes({ bytesPerToken, tokens: state.context });
    return { end, bytesPerToken, perUser, maxFit: maxUsersPerGpu(free, perUser) };
  });
}

// How many users the toy runs: "most at target", "max that fits", or a stop capped to what fits.
function pickUsers(users, target, maxFit) {
  if (users === 'target') return target.users;
  if (users === 'fits') return maxFit;
  return Math.min(users, maxFit);
}

function speedAndCost(state, stepModel, users) {
  if (users < 1) return { step: null, tokSUser: null, tokSGpu: null, cost: null };
  const step = stepTime({ ...stepModel, tokens: users, seqs: users });
  const tokSGpu = users / step.timeS;
  return { step, tokSUser: 1 / step.timeS, tokSGpu, cost: costPerMillion(state.price, tokSGpu) };
}

// Output ÷ input cost per token = input tokens/s ÷ output tokens/s; the floor never lets decode beat prefill, so it is at least 1.
function costRatio(inputTokS, tokSGpu) {
  const ratio = inputTokS / tokSGpu;
  return ratio < 1 && ratio > 1 - EPS ? 1 : ratio;
}

// Everything the toy and the Check-my-work box read for one state (preset objects come from presets.js).
export function compute(state, data) {
  const model = modelPreset(data, state.model);
  const gpu = gpuPreset(data, state.hw);
  const scenario = scenarioPresets(data);
  if (!supportsWeights(gpu, state.weights)) throw new RangeError(`compute: ${gpu.label} has no figure for ${WEIGHT_FORMATS[state.weights].label} (${NO_FP4})`);
  const weights = weightsFor(model, state.weights);
  const free = freeHbmPerGpu({ hbmBytes: gpu.hbmBytes, weightBytes: weights, gpus: state.gpus });
  const kv = kvRows(model, state, free);
  const chosen = kv[model.dense || state.kvEnd === 'low' ? 0 : 1];
  const stepModel = {
    activeParamsPerGpu: model.activeParams, weightBytesPerGpu: weights / state.gpus, dModel: model.dModel, actBytesPerElem: WEIGHT_FORMATS[state.weights].actBytesPerElem,
    kvBytesPerToken: chosen.bytesPerToken, context: state.context, peakTflops: mathPeak(gpu, state.weights), bandwidthTBps: gpu.bandwidthTBps,
  };
  const target = usersAtTarget({ targetTokPerUser: state.target, ...stepModel, maxUsers: chosen.maxFit });
  const users = pickUsers(state.users, target, chosen.maxFit);
  const run = speedAndCost(state, stepModel, users);
  const prefillTokS = prefillTokPerSecCeiling({ activeParamsPerGpu: model.activeParams, peakTflops: stepModel.peakTflops });
  const inputTokS = prefillTokS / (1 - state.hit);
  const mtp = state.mtp && users >= 1 ? batchSpeedup({ alpha: scenario.mtpAlpha, k: MTP.k, c: MTP.c, batch: users, model: stepModel }) : null;
  return {
    model, gpu, scenario, stepModel, weights, free, fits: free > 0, minGpus: minGpusForWeights(weights, gpu.hbmBytes),
    kv, chosen, maxFit: chosen.maxFit, target, users, clamped: typeof state.users === 'number' && users < state.users,
    ...run, prefillTokS, inputTokS, inputCost: costPerMillion(state.price, inputTokS),
    costRatio: run.tokSGpu === null ? null : costRatio(inputTokS, run.tokSGpu), mtp,
    measured: { gb300: costPerMillion(scenario.measured.gb300.usd, scenario.measured.gb300.tokSGpu), gb200: costPerMillion(scenario.measured.gb200.usd, scenario.measured.gb200.tokSGpu) },
  };
}

export const isV4 = (state) => state.model === V4_ID;
