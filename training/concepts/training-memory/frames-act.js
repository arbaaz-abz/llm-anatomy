// training-memory frames 5–7: the activations GPT-3's 96 blocks save for one 2,048-token sequence, what skipping the
// attention-score grid saves, and full recomputation (keep only each block's input, run the forward again).
import * as G from '@shared/glyphs.js';
import { sharePct } from '@math/memory.js';
import { activationBytesPerLayer } from '@math/training-memory.js';
import { sizeText, gbText, scoreSplit } from './format.js';
import { GPT3 } from './numbers.js';
import { seg, lerp, note, linkFrame, layer } from './stage.js';

const STACK = Object.freeze({ x: 20, y: 24, w: 220, count: 96, shown: 1 });
const TOPS = Object.freeze([STACK.y, STACK.y + 130]); // block 1 and block 96 (blockStack: 82 px blocks, 24 px fold, 12 px gaps)
const BAR_X = 290;
const PX_PER_BYTE = 60 / 1e9; // a block's bar is 60 px per GB, so its width is proportional to the GB it holds
const FLOW_X = { save: 10, rerun: 252 };
const STACK_BOTTOM = STACK.y + 212;
const ARGS = Object.freeze({ seq: GPT3.seq, microBatch: GPT3.microBatch, hidden: GPT3.hidden, heads: GPT3.heads });
const PER_LAYER = Object.freeze({
  none: activationBytesPerLayer({ ...ARGS, recompute: 'none' }),
  full: activationBytesPerLayer({ ...ARGS, recompute: 'full' }),
});
const SPLIT = scoreSplit(ARGS);

function stack(svg) {
  G.blockStack(svg, { x: STACK.x, y: STACK.y, w: STACK.w, count: STACK.count, shown: STACK.shown });
}

// One block's saved-activation bar at the block's top; `parts` = [{ name, value, hue }] in bytes, drawn at 60 px per GB.
function actBar(svg, top, parts) {
  const live = parts.filter((q) => q.value > 1e-6);
  const total = live.reduce((s, q) => s + q.value, 0);
  if (total <= 0) return;
  const w = Math.max(total * PX_PER_BYTE, 2);
  G.shareBar(svg, { x: BAR_X, y: top + 14, w, h: 14, tail: 'none', label: 'saved activations of one block', parts: live, format: (share) => sizeText(share * total) });
  linkFrame(svg, 'act', { x: BAR_X, y: top + 14, w, h: 14 });
}

function marks(svg, totalBytes, { sequence = true } = {}) {
  note(svg, STACK.x, STACK_BOTTOM + 26, `all ${GPT3.layers} blocks save: ${gbText(totalBytes)}`, { cls: 'g-text' });
  if (sequence) note(svg, STACK.x, STACK_BOTTOM + 46, `one ${GPT3.seq.toLocaleString('en-US')}-token sequence`);
  note(svg, STACK.x, STACK_BOTTOM + 62, 'formula: Korthikanti et al. 2022, micro-batch 1');
}

export function drawFrame5(svg, p) {
  stack(svg);
  const fill = [seg(p, 0.1, 0.35), seg(p, 0.45, 0.7)];
  TOPS.forEach((top, i) => actBar(svg, top, [{ name: 'saved activations', value: PER_LAYER.none * fill[i], hue: 5 }]));
  const run = seg(p, 0.1, 0.7);
  G.flow(svg, { from: [FLOW_X.save, STACK.y], to: [FLOW_X.save, STACK_BOTTOM], carry: 'activation', progress: run });
  marks(svg, PER_LAYER.none * GPT3.layers * seg(p, 0.1, 0.75));
}

export function drawFrame6(svg, p) {
  stack(svg);
  const shrink = seg(p, 0.25, 0.75);
  const scores = SPLIT.scores * (1 - shrink);
  TOPS.forEach((top) => actBar(svg, top, [{ name: 'attention scores', value: scores, hue: 2 }, { name: 'everything else', value: SPLIT.rest, hue: 5 }]));
  const intro = seg(p, 0, 0.15);
  const zoom = layer(svg, Math.min(intro, 1 - shrink));
  const parts = SPLIT.scores / (GPT3.seq * GPT3.microBatch * GPT3.hidden);
  const wholeParts = SPLIT.total / (GPT3.seq * GPT3.microBatch * GPT3.hidden);
  if (intro > 0 && shrink < 1) {
    G.block(zoom, { x: BAR_X, y: STACK.y + 86, w: 190, h: 32, label: 'attention scores', state: 'idle' });
    note(zoom, BAR_X, STACK.y + 134, `scores ${Math.round(parts)} of ${Math.round(wholeParts)} parts (${Math.round(sharePct(parts, wholeParts))}%)`);
  }
  marks(svg, (scores + SPLIT.rest) * GPT3.layers);
}

export function drawFrame7(svg, p) {
  stack(svg);
  const shrink = seg(p, 0.2, 0.7);
  const rest = lerp(SPLIT.rest, PER_LAYER.full, shrink);
  TOPS.forEach((top) => actBar(svg, top, [{ name: shrink >= 1 ? 'block input' : 'everything else', value: rest, hue: 5 }]));
  const rerun = seg(p, 0.15, 0.85);
  G.flow(svg, { from: [FLOW_X.rerun, STACK.y], to: [FLOW_X.rerun, STACK_BOTTOM], carry: 'activation', progress: rerun });
  note(svg, FLOW_X.rerun + 10, STACK.y - 6, 're-run');
  marks(svg, rest * GPT3.layers, { sequence: false });
  note(svg, STACK.x, STACK_BOTTOM + 46, 'extra compute: about +33%', { opacity: seg(p, 0.7, 1) });
  note(svg, STACK.x, STACK_BOTTOM + 78, 'forward 1 + backward 2 + re-run 1 = 4 units instead of 3', { opacity: seg(p, 0.7, 1) });
}
