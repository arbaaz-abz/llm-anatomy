// Pure helpers for the multimodal page: number text, the toy's state rules and the "Check my work" line. No DOM.
// Every count comes from math/vision.js; every percentage from sharePct (math/memory.js).
import { patchGrid, visionTokens } from '@math/vision.js';
import { sharePct } from '@math/memory.js';
import { lookupFact } from '@shared/claims.js';

export const CONTEXT_1M = 1048576;
export const CONTEXT_256K = 262144;
export const CONTEXTS = Object.freeze([CONTEXT_256K, CONTEXT_1M]);
export const MIN_SIDE = 112;
export const FALLBACK = Object.freeze({ maxSide: 3584, patch: 14, merge: 2, deepseekMerge: 3 });
export const PHONE = Object.freeze({ width: 1008, height: 1008 }); // the 1,008-pixel photo of frames 7–8
export const WIDE = Object.freeze({ width: 1008, height: 504 });
export const VIDEO_FRAME = Object.freeze({ width: 448, height: 448 }); // frame 9's video, 2 frames per second
export const SECONDS = Object.freeze([10, 60, 600, 3600]);
export const FPS = Object.freeze([1, 2]);

// The model facts the page reads from data/models.json (never typed): FALLBACK only keeps the pure functions total
// when no data is loaded (the validation lesson); the page test pins the data to the same values.
export function modelFacts(data) {
  const fact = (id, key, fallback) => lookupFact(data?.models, id, key)?.value ?? fallback;
  return Object.freeze({
    maxSide: fact('kimi-k3', 'max_image_side', FALLBACK.maxSide),
    patch: fact('kimi-k3', 'patch_size', FALLBACK.patch),
    merge: fact('kimi-k3', 'vision_merge', FALLBACK.merge),
    deepseekMerge: fact('deepseek-v4.1-flash', 'vision_merge', FALLBACK.deepseekMerge),
  });
}

export const TOY_IMAGE = Object.freeze({ width: 16, height: 16, patch: 4, merge: 2 }); // the animation's 16-pixel picture

// What each chip loads. DeepSeek's patch size is not in the data, so its chip keeps Kimi K3's and changes only the merge.
export const presetSettings = (facts) => Object.freeze({
  kimi: Object.freeze({ patch: facts.patch, merge: facts.merge }),
  deepseek: Object.freeze({ patch: facts.patch, merge: facts.deepseekMerge }),
  toy: TOY_IMAGE,
});

export const initialState = (facts) => Object.freeze({
  preset: 'kimi', width: PHONE.width, height: PHONE.height, patch: facts.patch, merge: facts.merge, media: 'image', seconds: 60, fps: 2, context: CONTEXT_1M,
});
export const INITIAL_STATE = initialState(modelFacts(null));

export const int = (n) => n.toLocaleString('en-US');

// ---- sizes the sliders may take: multiples of patch × merge from 112 px up to the model's largest side ----
export function sideValues({ patch, merge, maxSide }) {
  const step = patch * merge;
  const first = Math.ceil(MIN_SIDE / step) * step;
  const last = Math.floor(maxSide / step) * step;
  return Array.from({ length: (last - first) / step + 1 }, (_, i) => first + i * step);
}

export function snapSide(value, options) {
  const values = sideValues(options);
  return values.reduce((best, v) => (Math.abs(v - value) < Math.abs(best - value) ? v : best), values[0]);
}

// New state after a preset chip: the chip's settings, sizes snapped; leaving the toy image restores the phone photo.
export function applyPreset(state, preset, facts) {
  const settings = presetSettings(facts);
  const base = state.preset === 'toy' ? { ...state, ...PHONE, ...settings.kimi } : state;
  return { ...state, ...snapped({ ...base, ...settings[preset] }, facts.maxSide), preset };
}

// A change to one setting: sizes re-snapped to the new step, the preset becomes "custom".
export function applySetting(state, change, facts) {
  return { ...state, ...snapped({ ...state, ...change }, facts.maxSide), preset: 'custom' };
}

// The toy image (patch 4) keeps its 16 pixels; every other size is a multiple of patch × merge within [112, maxSide].
function snapped(next, maxSide) {
  const { patch, merge } = next;
  if (patch === TOY_IMAGE.patch) return { width: TOY_IMAGE.width, height: TOY_IMAGE.height, patch, merge };
  const options = { patch, merge, maxSide };
  return { width: snapSide(next.width, options), height: snapSide(next.height, options), patch, merge };
}

// ---- counts ----
export const framesOf = (state) => (state.media === 'video' ? state.seconds * state.fps : 1);

export function counts(state) {
  const { width, height, patch, merge } = state;
  const grid = patchGrid({ width, height, patch });
  const frames = framesOf(state);
  const tokens = visionTokens({ width, height, patch, merge, frames });
  return { ...grid, frames, tokensPerFrame: tokens.tokensPerFrame, tokens: tokens.tokens };
}

export const imagesThatFit = (context, tokensPerFrame) => Math.floor(context / tokensPerFrame);

// Share of the context window, one decimal from 1 % and two below ("0.12%", "1.6%", "175.8%").
export function shareText(tokens, context) {
  const exact = sharePct(tokens, context, { decimals: 6 });
  if (exact > 0 && exact < 0.005) return '<0.01%';
  const decimals = exact >= 1 ? 1 : 2;
  return `${sharePct(tokens, context, { decimals }).toFixed(decimals)}%`;
}

export const fitsContext = (tokens, context) => tokens <= context;

// ---- "Check my work" ----
export function checkWork(state) {
  const { width, height, patch, merge } = state;
  const c = counts(state);
  const sides = width === height ? [`${width} ÷ ${patch} = ${c.cols}`] : [`${width} ÷ ${patch} = ${c.cols}`, `${height} ÷ ${patch} = ${c.rows}`];
  const parts = [...sides, `${c.cols} × ${c.rows} = ${int(c.patches)}`];
  parts.push(merge === 1 ? `no merge = ${int(c.tokensPerFrame)}` : `÷ (${merge} × ${merge}) = ${int(c.tokensPerFrame)}`);
  if (state.media === 'video') parts.push(`× ${int(c.frames)} frames = ${int(c.tokens)}`);
  return parts.join(' · ');
}
