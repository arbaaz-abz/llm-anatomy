// The midtraining stage: renderFor(data)(index, progress, stage) draws frame index + 1 of storyboard §5 at progress 0 -> 1.
// One <svg>, cleared every frame, so the DOM is a pure function of (index, progress).
import * as G from '@shared/glyphs.js';
import { STAGE } from './stage.js';
import { CAPTIONS } from './captions.js';
import { runData, stageText } from './facts.js';
import { drawFrame1, drawFrame2, drawFrame3, drawFrame4, drawFrame5 } from './frames-curve.js';
import { drawFrame6, drawFrame7, drawFrame8, drawFrame9 } from './frames-context.js';

const FRAMES = Object.freeze([drawFrame1, drawFrame2, drawFrame3, drawFrame4, drawFrame5, drawFrame6, drawFrame7, drawFrame8, drawFrame9]);

// The render function of a lesson built from `data` (null = the numbers.js stand-ins).
export function renderFor(data) {
  const ctx = Object.freeze({ run: runData(data), text: stageText(data) });
  return (index, progress, stage) => {
    const draw = FRAMES[index];
    if (!draw) throw new RangeError(`midtraining: no frame ${index + 1} (there are ${FRAMES.length})`);
    const svg = stage.querySelector('svg') ?? G.svgEl('svg', { width: STAGE.w, height: STAGE.h, viewBox: `0 0 ${STAGE.w} ${STAGE.h}`, role: 'img' }, stage);
    svg.replaceChildren();
    G.hatchFill(svg); // the hatch pattern first, so every frame's DOM is identical however it was reached
    svg.setAttribute('aria-label', CAPTIONS[index]);
    draw(svg, progress, ctx);
  };
}
