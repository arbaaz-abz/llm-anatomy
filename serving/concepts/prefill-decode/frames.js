// prefill-decode stage: render(index, progress, stage) draws frame index+1 into one fixed 580 × 366 <svg> (stub until B5).
import * as G from '@shared/glyphs.js';
import { CAPTIONS } from './captions.js';

const STAGE = Object.freeze({ w: 580, h: 366 });

export function render(index, progress, stage) {
  if (!CAPTIONS[index]) throw new RangeError(`prefill-decode: no frame ${index + 1} (there are ${CAPTIONS.length})`);
  const svg = stage.querySelector('svg') ?? G.svgEl('svg', { width: STAGE.w, height: STAGE.h, viewBox: `0 0 ${STAGE.w} ${STAGE.h}`, role: 'img' }, stage);
  svg.replaceChildren();
  G.hatchFill(svg);
  svg.setAttribute('aria-label', CAPTIONS[index]);
}
