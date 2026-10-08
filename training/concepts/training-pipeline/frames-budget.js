// training-pipeline frames 8–9: the token budget of GLM-5 as a share bar (storyboard §5), then the compute line.
import * as G from '@shared/glyphs.js';
import { GLM5_TOKENS, STAGE_FIGURES, REPLIES } from './numbers.js';
import { STAGE, seg, leaving, arriving, layer, note, stageRow, checkpointChip, replyRow } from './stage.js';
import { mergedBlock, polishLabels } from './frames-stages.js';

export const BAR = Object.freeze({ x: 14, y: 156, w: 330 });
export const BAR_LABEL = '28.5T (its published stages sum to 28.55T)';
// The unpublished post-training part is { value: null }: neutral, 24 px off the scale (P3-R6).
export const GLM5_PARTS = Object.freeze([
  Object.freeze({ name: 'pretrain', value: GLM5_TOKENS.pretrain, hue: 1 }),
  Object.freeze({ name: 'mid-train', value: GLM5_TOKENS.midTrain, hue: 2 }),
  Object.freeze({ name: 'post-training', value: null }),
]);
const BAR_OPTS = Object.freeze({ w: BAR.w, minSegment: 12, label: 'GLM-5 token budget' });

const ALL_IDLE = Object.freeze(Array(6).fill('idle'));

// The bar slides in from the left behind a clip that opens with `reveal`.
function budgetBar(svg, reveal) {
  const g = layer(svg);
  if (reveal < 1) {
    const clip = G.svgEl('clipPath', { id: 'tp-bar-reveal' }, g);
    G.svgEl('rect', { x: 0, y: BAR.y - 8, width: Math.max(1, reveal * STAGE.w), height: 150 }, clip);
    g.setAttribute('clip-path', 'url(#tp-bar-reveal)');
  }
  G.shareBar(g, { x: BAR.x, y: BAR.y, parts: GLM5_PARTS, ...BAR_OPTS });
  note(g, BAR.x, BAR.y - 16, `GLM-5: ${BAR_LABEL}`);
  note(g, BAR.x, BAR.y + 100, 'neutral = not published');
}

// Where the selection outline sits: the pretrain segment, or the post-training one (off the scale).
function segmentBox(name) {
  const layout = G.shareBarLayout(GLM5_PARTS, { w: BAR.w, minSegment: BAR_OPTS.minSegment });
  const s = [...layout.main, ...layout.unknown].find((q) => q.name === name);
  return { x: BAR.x + s.x, y: BAR.y, w: s.width, h: 14 };
}

function handoffFromPolish(svg, p) {
  const out = leaving(p);
  if (out <= 0) return;
  mergedBlock(svg, { opacity: out, state: 'idle', withSpecialists: 0 });
  polishLabels(svg, 1, out);
}

export function drawFrame8(svg, p) {
  stageRow(svg, ALL_IDLE);
  checkpointChip(svg, 5, leaving(p));
  handoffFromPolish(svg, p);
  budgetBar(svg, seg(p, 0.1, 0.8));
  G.selectionMark(layer(svg, arriving(p)), segmentBox('pretrain'));
  replyRow(svg, { words: REPLIES.chat });
}

export function drawFrame9(svg, p) {
  stageRow(svg, ALL_IDLE);
  budgetBar(svg, 1);
  G.selectionMark(layer(svg, leaving(p)), segmentBox('pretrain'));
  G.selectionMark(layer(svg, seg(p, 0.15, 0.4)), segmentBox('post-training'));
  note(layer(svg, seg(p, 0.3, 0.8)), BAR.x, BAR.y + 130, STAGE_FIGURES.v32Line);
  replyRow(svg, { words: REPLIES.chat });
}
