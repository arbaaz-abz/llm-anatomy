// quantization stage: render(index, progress, stage) draws frame index+1 into one fixed 580 × 366 <svg>.
// Every frame is a pure function of (index, progress): no module state, no clock, no randomness.
import * as G from '@shared/glyphs.js';
import { STAGE } from './stage.js';
import { CAPTIONS } from './captions.js';
import { drawBlockFrame, drawFrame7, drawCompareEnd } from './frames-block.js';
import { drawFrame8, drawFrame9, drawFrame10, drawFrame11 } from './frames-model.js';

const FRAMES = Object.freeze([
  ...[0, 1, 2, 3, 4, 5].map((i) => (svg, p) => drawBlockFrame(svg, i, p)),
  drawFrame7,
  (svg, p) => drawFrame8(svg, p, drawCompareEnd),
  drawFrame9,
  drawFrame10,
  drawFrame11,
]);

export function render(index, progress, stage) {
  const draw = FRAMES[index];
  if (!draw) throw new RangeError(`quantization: no frame ${index + 1} (there are ${FRAMES.length})`);
  const svg = stage.querySelector('svg') ?? G.svgEl('svg', { width: STAGE.w, height: STAGE.h, viewBox: `0 0 ${STAGE.w} ${STAGE.h}`, role: 'img' }, stage);
  svg.replaceChildren();
  G.hatchFill(svg); // claim the hatch pattern first, so every frame's DOM is identical however it was reached
  svg.setAttribute('aria-label', CAPTIONS[index]);
  draw(svg, progress);
}
