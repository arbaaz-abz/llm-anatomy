// pretraining stage (stub until B5).
import * as G from '@shared/glyphs.js';
import { CAPTIONS } from './captions.js';

export function renderFor() {
  return (index, progress, stage) => {
    const svg = stage.querySelector('svg') ?? G.svgEl('svg', { width: 580, height: 366, viewBox: '0 0 580 366', role: 'img' }, stage);
    svg.replaceChildren();
    svg.setAttribute('aria-label', CAPTIONS[index]);
  };
}
