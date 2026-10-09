// The serving-overview stage (stub until B5).
import * as G from '@shared/glyphs.js';
import { CAPTIONS } from './captions.js';

export function render(index, progress, stage) {
  if (!CAPTIONS[index]) throw new RangeError(`serving-overview: no frame ${index + 1} (there are ${CAPTIONS.length})`);
  const svg = stage.querySelector('svg') ?? G.svgEl('svg', { width: 580, height: 366, viewBox: '0 0 580 366', role: 'img' }, stage);
  svg.replaceChildren();
  G.hatchFill(svg);
  svg.setAttribute('aria-label', CAPTIONS[index]);
}
