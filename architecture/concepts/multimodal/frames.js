// The multimodal stage: render(index, progress, stage) draws frame index + 1 of storyboard §5 at progress 0 → 1.
// One <svg>, cleared every frame, so the DOM is a pure function of (index, progress).
import * as G from '@shared/glyphs.js';
import { STAGE } from './stage.js';
import { CAPTIONS } from './captions.js';
import { drawFrame1, drawFrame2, drawFrame3 } from './frames-patches.js';
import { drawFrame4, drawFrame5, drawFrame6 } from './frames-merge.js';
import { drawFrame7, drawFrame8, drawFrame9 } from './frames-counts.js';
import { drawFrame10 } from './frames-recipes.js';

const FRAMES = Object.freeze([drawFrame1, drawFrame2, drawFrame3, drawFrame4, drawFrame5, drawFrame6, drawFrame7, drawFrame8, drawFrame9, drawFrame10]);

// ctx: the model facts frames 7–10 print ({ patch, merge, maxSide }), read from the data by content.js.
export function render(index, progress, stage, ctx) {
  const draw = FRAMES[index];
  if (!draw) throw new RangeError(`multimodal: no frame ${index + 1} (there are ${FRAMES.length})`);
  const svg = stage.querySelector('svg') ?? G.svgEl('svg', { width: STAGE.w, height: STAGE.h, viewBox: `0 0 ${STAGE.w} ${STAGE.h}`, role: 'img' }, stage);
  svg.replaceChildren();
  G.hatchFill(svg); // the hatch pattern first, so every frame's DOM is identical however it was reached
  svg.setAttribute('aria-label', CAPTIONS[index]);
  draw(svg, progress, ctx);
}
