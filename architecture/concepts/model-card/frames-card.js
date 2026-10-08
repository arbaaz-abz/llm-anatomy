// model-card frames, the card column: the ten field chips (typed in top to bottom in frame 1), the followed chip's
// selection mark, and the chips added from the paper / config in frames 4, 7 and 9.
import * as G from '@shared/glyphs.js';
import { COL, TOKEN_H, layer, note, mark, chipY, seg } from './stage.js';

const TYPE_STEP = 0.085; // chip i starts typing at i · TYPE_STEP of the `chips` progress
const TYPE_SPAN = 0.2;

const chipBox = (label, y) => ({ x: COL.x, y, w: G.tokenWidth(label), h: TOKEN_H });

function drawChips(svg, model, scene) {
  model.chips.forEach((chip, i) => {
    const opacity = scene.col * seg(scene.chips, i * TYPE_STEP, i * TYPE_STEP + TYPE_SPAN);
    if (opacity <= 0) return;
    G.token(layer(svg, opacity), { x: COL.x, y: chipY(i), text: chip.label });
    mark(svg, scene.col * scene[`s${i}`], chipBox(chip.label, chipY(i)));
  });
}

const EXTRA_GROUPS = Object.freeze([['x4', 'frame4'], ['x7', 'frame7'], ['x9', 'frame9']]);

function drawExtras(svg, model, scene) {
  const strongest = Math.max(...EXTRA_GROUPS.map(([key]) => scene[key]));
  if (strongest <= 0) return;
  note(layer(svg, strongest), COL.x, COL.extraLabelY, 'from the paper / config');
  EXTRA_GROUPS.forEach(([key, group]) => {
    if (scene[key] <= 0) return;
    model.extras[group].forEach((label, slot) => G.token(layer(svg, scene[key]), { x: COL.x, y: COL.extraY[slot], text: label }));
  });
  model.extras.frame9.forEach((label, slot) => mark(svg, scene[slot === 0 ? 'sa' : 'sb'], chipBox(label, COL.extraY[slot])));
}

export function drawColumn(svg, model, scene) {
  if (scene.col <= 0) return;
  note(layer(svg, scene.col), COL.x, COL.titleY, `${model.name} card`);
  drawChips(svg, model, scene);
  drawExtras(svg, model, scene);
}
