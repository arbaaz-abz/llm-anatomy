// The toy's view model (pure): (state, shapes) → every string and number the toy prints. Unit-tested; toy.js only paints it.
import { formatBytes } from '@math/core.js';
import { kvCacheBytes } from '@math/memory.js';
import { TOY_SHAPE, BYTES_PER_NUMBER, TOY_LATENTS, REAL_LATENTS, DEFAULT_CONTEXT } from './numbers.js';
import { numbersPerLayer, bytesPerToken, timesSmaller, storedHeads, wiring, SCHEME_LABELS } from './scheme.js';
import { int, timesText, checkWork, groupsLine } from './format.js';
import { patternFor } from './pattern.js';

export const MODEL_CHIPS = Object.freeze([
  { value: 'toy', label: 'Toy layer' },
  { value: 'gpt3', label: 'GPT-3' },
  { value: 'llama', label: 'Llama-3.1-70B' },
  { value: 'minimax', label: 'MiniMax-M3' },
  { value: 'v3', label: 'DeepSeek-V3' },
]);
export const SCHEME_OPTIONS = Object.freeze(Object.entries(SCHEME_LABELS).map(([value, label]) => ({ value, label })));

const TOY_KV_DEFAULT = 2;
const GQA_DEFAULT = 8; // "8 KV heads became the common middle setting": where the GQA slider starts on a model that is not GQA

// Opens on the toy layer of the animation: plain MHA, so the first numbers are frame 1's.
export const INITIAL_STATE = Object.freeze({ model: 'toy', scheme: 'mha', kvHeads: TOY_KV_DEFAULT, latent: TOY_SHAPE.dLatent, context: DEFAULT_CONTEXT, shareKeys: true });

export const shapeOf = (state, shapes) => (state.model === 'toy' ? { ...TOY_SHAPE, name: 'Toy layer', contextLength: null } : shapes[state.model]);

export const divisors = (n) => Array.from({ length: n }, (_, i) => i + 1).filter((d) => n % d === 0);

// The slider stops that fit one model: KV heads divide its query heads; the latent stops are the toy's or the real ones.
export const stopsFor = (model, shapes) => {
  const shape = shapeOf({ model }, shapes);
  return { kvHeads: divisors(shape.queryHeads), latent: model === 'toy' ? TOY_LATENTS : REAL_LATENTS };
};

// What a model chip sets: the shape's real scheme, its KV heads (GQA) and its latent size.
export function presetPatch(model, shapes) {
  if (model === 'toy') return { model, scheme: INITIAL_STATE.scheme, kvHeads: TOY_KV_DEFAULT, latent: TOY_SHAPE.dLatent };
  const shape = shapes[model];
  return { model, scheme: shape.scheme, kvHeads: shape.scheme === 'gqa' ? shape.kvHeads : GQA_DEFAULT, latent: shape.dLatent };
}

const configOf = (state, shape) => ({ scheme: state.scheme, queryHeads: shape.queryHeads, kvHeads: state.kvHeads, headDim: shape.headDim, dLatent: state.latent, dRope: shape.dRope });

function specLine(state, shape) {
  if (state.model === 'toy') return `Toy layer: ${shape.layers} layer, ${shape.queryHeads} query heads of ${shape.headDim} numbers; every size below is for this one layer.`;
  const base = `${shape.name}: ${int(shape.layers)} layers, ${int(shape.queryHeads)} query heads of ${int(shape.headDim)} numbers.`;
  return state.scheme === 'mla' ? `${base} The position key stays at ${int(shape.dRope)} numbers.` : base;
}

function memoryNumbers(state, shape) {
  const config = configOf(state, shape);
  const basis = { layers: shape.layers, bytesPerElem: BYTES_PER_NUMBER };
  const bytes = bytesPerToken(config, basis);
  return {
    config, bytes,
    perLayer: numbersPerLayer(config),
    mhaPerLayer: numbersPerLayer({ ...config, scheme: 'mha' }),
    ratio: timesSmaller(config, basis),
    cache: kvCacheBytes({ bytesPerToken: bytes, tokens: state.context }),
  };
}

export function toyView(state, shapes) {
  const shape = shapeOf(state, shapes);
  const m = memoryNumbers(state, shape);
  const stored = state.scheme === 'mla' ? 1 : storedHeads(m.config);
  const pattern = patternFor();
  const beyond = shape.contextLength !== null && state.context > shape.contextLength;
  return {
    spec: specLine(state, shape),
    perLayer: int(m.perLayer),
    bytes: `${int(m.bytes)} B`,
    bytesSub: m.bytes >= 1000 ? formatBytes(m.bytes) : '',
    ratio: state.scheme === 'mha' ? `${timesText(m.ratio)} (this is MHA)` : timesText(m.ratio),
    cacheLabel: `Cache at ${int(state.context)} tokens`,
    cache: formatBytes(m.cache),
    whatIf: beyond ? `a what-if at this shape; its own context was ${int(shape.contextLength)}` : '',
    groups: groupsLine({ scheme: state.scheme, queryHeads: shape.queryHeads, stored }),
    wiring: shape.queryHeads <= TOY_SHAPE.queryHeads ? { groups: wiring(m.config), mla: state.scheme === 'mla', dLatent: state.latent, dRope: shape.dRope } : null,
    checkWork: checkWork({
      scheme: state.scheme, stored, headDim: shape.headDim, dLatent: state.latent, dRope: shape.dRope,
      perLayer: m.perLayer, layers: shape.layers, bytesPerElem: BYTES_PER_NUMBER, bytes: m.bytes, mhaPerLayer: m.mhaPerLayer, ratio: m.ratio,
    }),
    headA: pattern.a,
    headB: state.shareKeys ? pattern.shared : pattern.own,
    headBTitle: state.shareKeys ? 'head B, reading head A\'s keys' : 'head B, with its own keys',
  };
}
