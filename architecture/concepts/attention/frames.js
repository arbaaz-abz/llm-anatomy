// The attention stage: render(index, progress, stage) draws frame index + 1 of storyboard §5 at progress 0 → 1.
// One <svg>, cleared every frame, so the DOM is a pure function of (index, progress).
import * as G from '@shared/glyphs.js';
import { STAGE } from './stage.js';
import { CAPTIONS } from './captions.js';
import { drawFrame1, drawFrame2, drawFrame3, drawFrame4, drawFrame5, drawFrame6, drawFrame7 } from './frames-row.js';
import { drawFrame8, drawFrame9, drawFrame10 } from './frames-heads.js';

const FRAMES = Object.freeze([drawFrame1, drawFrame2, drawFrame3, drawFrame4, drawFrame5, drawFrame6, drawFrame7, drawFrame8, drawFrame9, drawFrame10]);

export function render(index, progress, stage) {
  const svg = stage.querySelector('svg') ?? G.svgEl('svg', { width: STAGE.w, height: STAGE.h, viewBox: `0 0 ${STAGE.w} ${STAGE.h}`, role: 'img' }, stage);
  G.hatchFill(svg); // claim the svg's hatch id first, so frames with and without hatching keep one svg state
  svg.replaceChildren();
  svg.setAttribute('aria-label', CAPTIONS[index]);
  FRAMES[index](svg, progress);
}
