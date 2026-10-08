// decoder-recap stage: render(index, progress, stage) draws frame index+1 into one fixed 580 × 366 <svg>.
// Every frame is a pure function of (index, progress): no module state, no clock, no randomness.
import * as G from '@shared/glyphs.js';
import { STAGE } from './stage.js';
import { CAPTIONS } from './captions.js';
import { drawFrame } from './frame-kit.js';
import { panel1, panel2, panel3, panel4 } from './frames-a.js';
import { panel5, panel6, panel7 } from './frames-b.js';
import { panel8, panel9, panel10 } from './frames-c.js';

const PANELS = Object.freeze([panel1, panel2, panel3, panel4, panel5, panel6, panel7, panel8, panel9, panel10]);

export function render(index, progress, stage) {
  if (!Number.isInteger(index) || index < 0 || index >= PANELS.length) throw new RangeError(`decoder-recap: no frame ${index + 1} (there are ${PANELS.length})`);
  const svg = stage.querySelector('svg') ?? G.svgEl('svg', { width: STAGE.w, height: STAGE.h, viewBox: `0 0 ${STAGE.w} ${STAGE.h}`, role: 'img' }, stage);
  svg.replaceChildren();
  G.hatchFill(svg); // claim the hatch pattern first, so every frame's DOM is identical however it was reached
  svg.setAttribute('aria-label', CAPTIONS[index]);
  drawFrame(svg, index + 1, progress, PANELS);
}
