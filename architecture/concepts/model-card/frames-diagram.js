// model-card frames, the decoder diagram on the right: the block stack (attention and MoE halves, "⋮ × N") and the
// input lane (four tokens, an unused image-patch slot, and the lane that stretches to the context length).
import * as G from '@shared/glyphs.js';
import { STACK, LANE, TOKEN_H, layer, note, mark, rule, lerp, laneWidth } from './stage.js';
import { drawPanel } from './frames-panels.js';

const HALVES = Object.freeze(['attention', 'MoE']);
const ATTENTION = Object.freeze({ block: 1, half: 0 });
const MOE = Object.freeze({ block: 1, half: 1 });

// The stack's count label types in the model's layer count ("⋮ × 6", then "⋮ × 61"); until then it reads N.
export function typedLayers(layers, typed) {
  if (typed <= 0) return 'N';
  return layers.slice(0, Math.max(1, Math.ceil(typed * layers.length)));
}

function drawStack(svg, model, scene) {
  const layers = typedLayers(model.layers, scene.typed);
  const active = scene.litAttn > 0.5 ? ATTENTION : scene.litMoe > 0.5 ? MOE : null;
  G.blockStack(svg, {
    x: STACK.x, y: STACK.y, w: STACK.w, count: Number(model.layers), shown: 1, halves: HALVES, active,
    countLabel: `⋮ × ${layers}`, lastLabel: `block ${scene.typed >= 1 ? model.layers : 'N'}`,
  });
  mark(svg, scene.selStack, { x: STACK.x, y: STACK.y, w: STACK.w, h: STACK.h });
}

// An empty slot where an image patch would enter: dimmed, never filled.
function patchSlot(svg) {
  const g = G.svgEl('g', { class: 'glyph g-patch g-patch--dim', transform: `translate(${LANE.patchX} ${LANE.y})`, role: 'img', 'aria-label': 'image patch slot, unused' }, svg);
  G.svgEl('rect', { class: 'g-frame', width: LANE.patchSize, height: LANE.patchSize, rx: 5 }, g);
}

function drawLane(svg, model, scene) {
  note(svg, LANE.x, LANE.labelY, 'input');
  let x = LANE.x;
  LANE.words.forEach((text, i) => {
    G.token(svg, { x, y: LANE.y, text, index: i + 1 });
    x += G.tokenWidth(text) + LANE.gap;
  });
  patchSlot(svg);
  if (scene.stretch > 0) {
    const g = layer(svg, scene.stretch);
    const end = lerp(LANE.x + laneWidth(), LANE.endX, scene.stretch);
    rule(g, LANE.x, LANE.barY, end, LANE.barY);
    note(g, LANE.endX, LANE.barY + 18, `${model.context.positions} positions`, { anchor: 'end' });
  }
  mark(svg, scene.selLane, { x: LANE.x, y: LANE.y, w: laneWidth(), h: TOKEN_H });
  if (scene.modal > 0) {
    const g = layer(svg, scene.modal);
    note(g, LANE.patchX, LANE.barY + 4, 'no vision encoder');
  }
  mark(svg, scene.selPatch, { x: LANE.patchX, y: LANE.y, w: LANE.patchSize, h: LANE.patchSize });
}

export function drawDiagram(svg, model, scene) {
  if (scene.diag <= 0) return;
  const g = layer(svg, scene.diag);
  drawStack(g, model, scene);
  drawPanel(g, model, scene);
  drawLane(g, model, scene);
}
