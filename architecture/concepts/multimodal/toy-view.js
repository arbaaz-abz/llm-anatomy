// multimodal toy view model (pure, no DOM): state + model facts → every string the toy prints.
// Counts come from math/vision.js through format.js; percentages from sharePct.
import { formatCount } from '@math/core.js';
import { counts, checkWork, imagesThatFit, shareText, fitsContext, int, TOY_IMAGE } from './format.js';

export const PRESET_CHIPS = Object.freeze([
  { value: 'kimi', label: 'Kimi K3' },
  { value: 'deepseek', label: '3 × 3 merge (DeepSeek, reported)' },
  { value: 'toy', label: 'toy image' },
  { value: 'custom', label: 'custom' },
]);

export const MERGE_OPTIONS = Object.freeze([
  { value: 1, label: 'none' }, { value: 2, label: '2 × 2' }, { value: 3, label: '3 × 3' },
]);

export const secondsText = (v) => int(v);

// The line under the chips: what the current settings are.
export function specLine(state, facts) {
  switch (state.preset) {
    case 'kimi': return `Kimi K3 settings: patch ${state.patch}, ${state.merge} × ${state.merge} merge, images up to ${int(facts.maxSide)} pixels a side.`;
    case 'deepseek': return `DeepSeek's encoder merges ${state.merge} × ${state.merge} (reported). Its patch size is not published in the data, so this chip keeps patch ${state.patch} and changes only the merge.`;
    case 'toy': return `Toy image: ${TOY_IMAGE.width} × ${TOY_IMAGE.height} pixels, patch ${TOY_IMAGE.patch}, ${TOY_IMAGE.merge} × ${TOY_IMAGE.merge} merge: the animation's own 4 tokens.`;
    default: return `Your settings: patch ${state.patch}, ${state.merge === 1 ? 'no merge' : `${state.merge} × ${state.merge} merge`}. Sizes snap to multiples of ${state.patch * state.merge} pixels so every count is a whole number.`;
  }
}

export function toyView(state, facts) {
  const c = counts(state);
  const video = state.media === 'video';
  const fit = imagesThatFit(state.context, c.tokensPerFrame);
  return {
    spec: specLine(state, facts),
    grid: `${c.cols} × ${c.rows} = ${int(c.patches)}`,
    perFrame: int(c.tokensPerFrame),
    perFrameLabel: video ? 'Tokens per frame' : 'Tokens per image',
    tokens: int(c.tokens),
    tokensApprox: c.tokens >= 1000 ? `≈ ${formatCount(c.tokens)}` : '',
    share: shareText(c.tokens, state.context),
    shareNote: fitsContext(c.tokens, state.context) ? `of ${int(state.context)} tokens` : 'does not fit',
    fit: int(fit),
    fitLabel: video ? 'Frames that fit' : 'Images of this size that fit',
    frames: video ? `${int(c.frames)} (${int(state.seconds)} s at ${state.fps} per second)` : '',
    checkWork: checkWork(state),
    showVideo: video,
    showSize: state.preset !== 'toy',
  };
}

// The three "try this" prompts of storyboard §6, each with the insight it leads to.
export function tryThis(facts) {
  const big = int(facts.maxSide);
  return [
    { prompt: `Kimi K3 preset, 1,008 × 1,008: 1,296 tokens. Set merge to none: 5,184. Set it back and drag to ${big} × ${big}: 16,384 tokens, 1.6% of 1M; 64 such images fill the window.`, insight: 'tokens grow with area; merging neighbors is the main lever on what an image costs.' },
    { prompt: 'Set 1,008 × 504: 648 tokens; the same photo squashed to 1,008 × 1,008 costs 1,296.', insight: 'keeping the aspect ratio (dynamic resolution) spends tokens only on real pixels.' },
    { prompt: 'Video, 448 × 448, 2 fps: 1 minute is 30,720 tokens; 10 minutes at 1 fps is 153,600 (14.6% of 1M; 58.6% of 262,144); 1 hour at 2 fps is 1,843,200, "does not fit".', insight: 'video cost is per frame, so long video needs pooling over time or fewer frames; these counts are before any pooling.' },
  ];
}
