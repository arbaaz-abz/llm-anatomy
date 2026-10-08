// rlhf-dpo stage: render(index, progress, stage) draws frame index+1 into one fixed 580 × 366 <svg>.
// Every frame is a pure function of (index, progress): no module state, no clock, no randomness. During the first
// HANDOFF of a frame the previous frame's end state fades out, so each frame starts from where the last one ended.
import * as G from '@shared/glyphs.js';
import { STAGE, HANDOFF, layer, seg } from './stage.js';
import { CAPTIONS } from './captions.js';
import { drawFrame1, drawFrame2, drawFrame3, drawFrame4 } from './frames-reward.js';
import { drawFrame5, drawFrame6, drawFrame7, drawFrame8 } from './frames-anchor.js';
import { drawFrame9, drawFrame10, drawFrame11 } from './frames-dpo.js';

const FRAMES = Object.freeze([drawFrame1, drawFrame2, drawFrame3, drawFrame4, drawFrame5, drawFrame6, drawFrame7, drawFrame8, drawFrame9, drawFrame10, drawFrame11]);

export function render(index, progress, stage) {
  const draw = FRAMES[index];
  if (!draw) throw new RangeError(`rlhf-dpo: no frame ${index + 1} (there are ${FRAMES.length})`);
  const svg = stage.querySelector('svg') ?? G.svgEl('svg', { width: STAGE.w, height: STAGE.h, viewBox: `0 0 ${STAGE.w} ${STAGE.h}`, role: 'img' }, stage);
  svg.replaceChildren();
  G.hatchFill(svg); // claim the hatch pattern first, so every frame's DOM is identical however it was reached
  svg.setAttribute('aria-label', CAPTIONS[index]);
  if (index > 0 && progress < HANDOFF) FRAMES[index - 1](layer(svg, 1 - seg(progress, 0, HANDOFF)), 1);
  draw(svg, progress);
}
