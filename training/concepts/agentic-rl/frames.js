// agentic-rl stage: renderFor(data) → render(index, progress, stage) draws frame index + 1 of storyboard §5 into one
// fixed 580 × 366 <svg>, cleared every frame, so the DOM is a pure function of (index, progress).
import * as G from '@shared/glyphs.js';
import { STAGE } from './stage.js';
import { CAPTIONS } from './captions.js';
import { stageText } from './facts.js';
import { drawFrame1, drawFrame2, drawFrame3, drawFrame4 } from './frames-episode.js';
import { drawFrame5, drawFrame6 } from './frames-async.js';
import { drawFrame7, drawFrame8, drawFrame9, drawFrame10 } from './frames-mismatch.js';

const FRAMES = Object.freeze([drawFrame1, drawFrame2, drawFrame3, drawFrame4, drawFrame5, drawFrame6, drawFrame7, drawFrame8, drawFrame9, drawFrame10]);

export function renderFor(data) {
  const text = stageText(data);
  return function render(index, progress, stage) {
    const draw = FRAMES[index];
    if (!draw) throw new RangeError(`agentic-rl: no frame ${index + 1} (there are ${FRAMES.length})`);
    const svg = stage.querySelector('svg') ?? G.svgEl('svg', { width: STAGE.w, height: STAGE.h, viewBox: `0 0 ${STAGE.w} ${STAGE.h}`, role: 'img' }, stage);
    svg.replaceChildren();
    G.hatchFill(svg); // the hatch pattern first, so every frame's DOM is identical however it was reached
    svg.setAttribute('aria-label', CAPTIONS[index]);
    draw(svg, progress, text);
  };
}
