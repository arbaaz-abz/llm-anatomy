// model-card stage: render(index, progress, stage, model) draws frame index+1 into one fixed 580 × 366 <svg>.
// Every frame is a pure function of (index, progress, model): the scene mix (scene.js) decides what is visible, the
// frames-*.js files draw it. No module state, no clock, no randomness.
import * as G from '@shared/glyphs.js';
import { STAGE } from './stage.js';
import { CAPTIONS } from './captions.js';
import { sceneAt } from './scene.js';
import { drawColumn } from './frames-card.js';
import { drawDiagram } from './frames-diagram.js';
import { drawExits, drawConflicts } from './frames-end.js';

export function render(index, progress, stage, model) {
  if (!model) throw new Error('model-card: the stage needs data/models.json (ctx.data), which the track app loads before mounting');
  const scene = sceneAt(index, progress); // throws a RangeError for an unknown frame
  const svg = stage.querySelector('svg') ?? G.svgEl('svg', { width: STAGE.w, height: STAGE.h, viewBox: `0 0 ${STAGE.w} ${STAGE.h}`, role: 'img' }, stage);
  svg.replaceChildren();
  G.hatchFill(svg); // claim the hatch pattern first, so every frame's DOM is identical however it was reached
  svg.setAttribute('aria-label', CAPTIONS[index]);
  drawDiagram(svg, model, scene);
  drawColumn(svg, model, scene);
  drawExits(svg, model, scene);
  drawConflicts(svg, model, scene);
}
