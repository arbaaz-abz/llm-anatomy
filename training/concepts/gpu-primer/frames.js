// gpu-primer stage: render(index, progress, stage) draws frame index+1 into one fixed 580 × 366 <svg> (stub until B5).
import * as G from '@shared/glyphs.js';
import { CAPTIONS } from './captions.js';

export function render(index, progress, stage) {
  if (!CAPTIONS[index]) throw new RangeError(`gpu-primer: no frame ${index + 1} (there are ${CAPTIONS.length})`);
  const svg = stage.querySelector('svg') ?? G.svgEl('svg', { width: 580, height: 366, viewBox: '0 0 580 366', role: 'img' }, stage);
  svg.replaceChildren();
  G.hatchFill(svg);
  svg.setAttribute('aria-label', CAPTIONS[index]);
}
