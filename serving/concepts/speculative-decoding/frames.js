// speculative-decoding stage: renderFor(text) → render(index, progress, stage) draws frame index+1 into one fixed 580 × 366 <svg>.
// `text` is the dated stage text (facts.js stageText, filled from the data by lessonFor), so frame 10 prints the data's numbers.
// Every frame is a pure function of (index, progress, text): no module state, no clock, no randomness.
import * as G from '@shared/glyphs.js';
import { STAGE } from './stage.js';
import { CAPTIONS } from './captions.js';
import { drawFrame1, drawFrame2, drawFrame3, drawFrame4, drawFrame5 } from './frames-draft.js';
import { drawFrame6 } from './frames-verify.js';
import { drawFrame7, drawFrame8, drawFrame9 } from './frames-curves.js';
import { drawFrame10 } from './frames-facts.js';

const FRAMES = Object.freeze([drawFrame1, drawFrame2, drawFrame3, drawFrame4, drawFrame5, drawFrame6, drawFrame7, drawFrame8, drawFrame9, drawFrame10]);

export function renderFor(text) {
  return function render(index, progress, stage) {
    const draw = FRAMES[index];
    if (!draw) throw new RangeError(`speculative-decoding: no frame ${index + 1} (there are ${FRAMES.length})`);
    const svg = stage.querySelector('svg') ?? G.svgEl('svg', { width: STAGE.w, height: STAGE.h, viewBox: `0 0 ${STAGE.w} ${STAGE.h}`, role: 'img' }, stage);
    svg.replaceChildren();
    G.hatchFill(svg); // claim the hatch pattern first, so every frame's DOM is identical however it was reached
    svg.setAttribute('aria-label', CAPTIONS[index]);
    draw(svg, progress, text);
  };
}
