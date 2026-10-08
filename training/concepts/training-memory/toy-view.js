// training-memory toy view model (pure, no DOM): state + data → every string the toy prints. Sizes come from
// math/training-memory.js, chip capacities and parameter counts from data (conventions: B200 reads hbm_usable_gb, "usable").
import { formatBytes, formatCount } from '@math/core.js';
import { TRAINING_RECIPES, trainingBytesPerParam, zeroPerGpuBytes, activationBytes, gpusToHoldStates } from '@math/training-memory.js';
import { lookupFact, fillText } from '@shared/claims.js';
import { CHIPS, MODELS, INITIAL_STATE, isGpt3, activationArgs, checkWork, gbText, trafficText, extraComputeText, SELECTIVE_HOVER } from './format.js';

export { INITIAL_STATE };
export const NOT_MODELED = 'activations: shape not modeled here (MoE / MLA layers); states only';
export const SCOPE_NOTE = 'Per-GPU memory counts training state and saved activations only; real runs also lose 20–50% of HBM to buffers, fragmentation and communication workspaces.';

const fact = (dataset, id, key) => lookupFact(dataset, id, key)?.value ?? null;

function required(value, what) {
  if (value === null) throw new Error(`training-memory: ${what} is missing from data/*.json`);
  return value;
}

export function modelParams(state, data) {
  const { dataId } = MODELS.find((m) => m.value === state.model);
  return required(fact(data?.models, dataId, 'total_params'), `${dataId}.total_params`);
}

// The chip's capacity in bytes and the words that say which capacity it is ("usable (192 nominal)" for B200).
export function chipInfo(chipValue, data) {
  const chip = CHIPS.find((c) => c.value === chipValue);
  const gb = required(fact(data?.hardware, chip.value, chip.key), `${chip.value}.${chip.key}`);
  const nominal = chip.basis === 'usable' ? required(fact(data?.hardware, chip.value, 'hbm_gb'), `${chip.value}.hbm_gb`) : null;
  return { label: chip.label, basis: nominal ? `usable (${nominal} nominal)` : chip.basis, bytes: gb * 1e9, word: chip.basis };
}

export function chipLabel(chipValue, data) {
  const info = chipInfo(chipValue, data);
  return `${info.label} ${info.bytes / 1e9} GB ${info.basis}`;
}

const MOMENT_NAMES = Object.freeze({ adam: 'Adam moments', adamBf16Moments: 'Adam moments (BF16)', muon: 'Muon momentum' });

function perParamParts(state) {
  const per = trainingBytesPerParam(TRAINING_RECIPES[state.recipe]);
  const parts = [
    { name: 'weight', value: per.weight, hue: 1 }, { name: 'gradient', value: per.grad, hue: 2 },
    { name: MOMENT_NAMES[state.recipe], value: per.optimizer, hue: 3 }, { name: 'master copy', value: per.master, hue: 4 },
  ];
  return { parts, total: per.total };
}

function compositionParts(z, act) {
  const parts = [
    { name: 'weights', value: z.weights, hue: 1 }, { name: 'gradients', value: z.grads, hue: 2 }, { name: 'optimizer', value: z.optimizer, hue: 3 },
    ...(act === null ? [] : [{ name: 'activations', value: act, hue: 5 }]),
  ];
  return parts.filter((p) => p.value > 0);
}

export function toyView(state, data) {
  const params = modelParams(state, data);
  const chip = chipInfo(state.gpu, data);
  const recipe = TRAINING_RECIPES[state.recipe];
  const per = perParamParts(state);
  const z = zeroPerGpuBytes({ params, recipe, stage: state.stage, dp: state.dp });
  const act = isGpt3(state) ? activationBytes(activationArgs(state)) : null;
  const total = z.total + (act ?? 0);
  const fits = total <= chip.bytes;
  const holders = gpusToHoldStates({ params, bytesPerParam: per.total, hbmBytes: chip.bytes });
  return {
    params: formatCount(params),
    perParam: { parts: per.parts, total: formatBytes(per.total), totalBytes: per.total },
    state: { weights: gbText(z.weights), grads: gbText(z.grads), optimizer: gbText(z.optimizer), total: gbText(z.total) },
    activations: act === null ? NOT_MODELED : gbText(act),
    total: gbText(total),
    capacity: gbText(chip.bytes),
    chipWords: `${chip.label}, ${chip.basis}`,
    fits,
    verdict: fits ? 'fits' : 'does not fit',
    verdictLine: `${fits ? 'fits' : 'does not fit'}: ${gbText(total)} vs ${gbText(chip.bytes)} (${chip.label}, ${chip.basis})`,
    composition: compositionParts(z, act),
    gpusToHold: `${holders.toLocaleString('en-US')} ${chip.label}s (${chip.word})`,
    traffic: trafficText(state.stage),
    extraCompute: { text: isGpt3(state) ? extraComputeText(state.recompute) : 'none: activations are not modeled for this model', hover: isGpt3(state) && state.recompute === 'selective' ? SELECTIVE_HOVER : '' },
    checkWork: checkWork(state, { params, chip }),
    holders,
  };
}

const num = (bytes) => gbText(bytes).replace(' GB', '');

// The try-this list, numbers computed from the same functions as the toy.
export function tryThis(data) {
  const adam = TRAINING_RECIPES.adam;
  const zero = (params, stage, recipe = adam) => zeroPerGpuBytes({ params, recipe, stage, dp: 64 }).total;
  const paper = required(fact(data?.models, 'zero-paper-7.5b', 'total_params'), 'zero-paper-7.5b.total_params');
  const g3 = required(fact(data?.models, 'gpt-3', 'total_params'), 'gpt-3.total_params');
  const v4 = required(fact(data?.models, 'deepseek-v4-pro', 'total_params'), 'deepseek-v4-pro.total_params');
  const a = (recompute) => activationBytes(activationArgs({ ...INITIAL_STATE, recompute }));
  const stages = [0, 1, 2, 3].map((s) => num(zero(paper, s))).join(' → ');
  const withAct = (recompute, stage) => gbText(zero(g3, stage) + a(recompute)).replace(' GB', '');
  const hold = (params, recipe, gpu) => gpusToHoldStates({ params, bytesPerParam: trainingBytesPerParam(TRAINING_RECIPES[recipe]).total, hbmBytes: chipInfo(gpu, data).bytes });
  const items = [
    {
      prompt: `Pick "ZeRO paper 7.5B", keep 64 GPUs, and step ZeRO stage 0 → 1 → 2 → 3: ${stages} GB per GPU, the ZeRO paper's table.`,
      insight: 'optimizer state is 12 of the 16 bytes, so sharding it alone (stage 1) removes most of the memory; stage 3 divides all 16 bytes by the number of GPUs.',
      rest: '',
    },
    {
      prompt: `GPT-3, ZeRO-3, 64 GPUs, H100: state is ${num(zero(g3, 3))} GB. With "store everything" the total is ${withAct('none', 3)} GB (does not fit); "skip attention scores" ${withAct('selective', 3)} GB (still no); "block inputs only" ${withAct('full', 3)} GB (fits). Now set ZeRO stage 0: ${withAct('full', 0)} GB even with full recomputation.`,
      insight: "ZeRO's stages shard state, never activations; recomputation shrinks activations, never state; you need both.",
      rest: ' Activations are per GPU because each GPU holds its own data; tensor and pipeline parallelism split them further (see [[parallelism]]).',
    },
    {
      prompt: `DeepSeek-V4-Pro's size ({deepseek-v4-pro.total_params|count} total, {deepseek-v4-pro.active_params|count} active) at the Adam recipe, H100: "GPUs just to hold the state" reads ${hold(v4, 'adam', 'h100')}. Switch to B200: ${hold(v4, 'adam', 'b200')}; B300: ${hold(v4, 'adam', 'b300')}. Switch the recipe to Muon (V4's own): ${hold(v4, 'muon', 'b300')}. Then pick GPT-3 (175B dense) with H100 and Adam again: ${hold(g3, 'adam', 'h100')} H100s.`,
      insight: 'memory follows total parameters, compute follows active ones, so a MoE that computes like a 49B model needs the training memory of a 1.6T one.',
      rest: " Muon's single momentum buffer saves a quarter of it.",
    },
  ];
  return items.map((item) => ({ ...item, prompt: fillText(item.prompt, data) }));
}
