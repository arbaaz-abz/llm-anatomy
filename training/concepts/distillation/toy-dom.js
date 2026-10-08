// distillation toy stage: draws the teacher row(s), the student row and the reward row into one <svg>, and wires the
// selectable student cells (P3-R17: focusable role=button groups, Enter / Space select, arrows move focus, G.selectionMark).
import * as G from '@shared/glyphs.js';
import { CANDIDATES, PROB_MAX_ABS, REWARD_MAX_ABS, WRONG_TOKEN } from './numbers.js';
import { mountStageSelect } from '@shared/ui/stage-select.js';
import { signed } from './format.js';

export const TOY_STAGE = Object.freeze({ w: 340, h: 216, x0: 132, cell: G.NUMBER_CELL, headY: 14, firstRowY: 22, gap: 4 });
const DIM = 0.35;

// Plain labels sit in a `.glyph.g-note` group, which is what gives them the theme's label ink (as stage.js note).
const svgText = (parent, x, y, str, props = {}) => G.svgEl('text', { x, y, class: 'g-label', ...props }, G.svgEl('g', { class: 'glyph g-note' }, parent)).append(str);
const cellX = (i) => TOY_STAGE.x0 + i * TOY_STAGE.cell;
const rowLabel = (parent, y, str) => svgText(parent, TOY_STAGE.x0 - 10, y + TOY_STAGE.cell / 2, str, { 'text-anchor': 'end', 'dominant-baseline': 'central' });

// A row of four cells inside a linked group (the invisible frame is what the math-panel hover outlines).
function row(parent, { y, values, link, maxAbs, format, cellOpacity = () => 1, attrs = {} }) {
  const g = G.svgEl('g', { 'data-link': link, ...attrs }, parent);
  G.svgEl('rect', { class: 'g-frame', x: TOY_STAGE.x0 - 1, y: y - 1, width: 4 * TOY_STAGE.cell + 2, height: TOY_STAGE.cell + 2, rx: 3, fill: 'none', stroke: 'none' }, g);
  values.forEach((v, i) => {
    const o = cellOpacity(i);
    G.cell(G.svgEl('g', { opacity: o < 1 ? o : null }, g), { x: cellX(i), y, size: TOY_STAGE.cell, v, maxAbs, format });
  });
  return g;
}

// The student's cells as buttons: the frame stays in the paint, the four button groups persist (shared stage-select) and are
// redrawn into on every paint so keyboard focus survives.
function studentCells(picks, parent, { y, values, sampled }) {
  const g = G.svgEl('g', { 'data-link': 's' }, parent);
  G.svgEl('rect', { class: 'g-frame', x: TOY_STAGE.x0 - 1, y: y - 1, width: 4 * TOY_STAGE.cell + 2, height: TOY_STAGE.cell + 2, rx: 3, fill: 'none', stroke: 'none' }, g);
  values.forEach((v, i) => {
    const node = picks.node(i);
    node.removeAttribute('display');
    G.cell(node, { x: cellX(i), y, size: TOY_STAGE.cell, v, maxAbs: PROB_MAX_ABS });
    if (i === sampled) selection(node, i, y);
  });
}

const selection = (parent, i, y) => G.selectionMark(parent, { x: cellX(i), y, w: TOY_STAGE.cell, h: TOY_STAGE.cell });
// The four student cells as shared stage-select items (selection follows focus); they sit last in the svg, so they are on top.
export function mountPicks(svg, onPick) {
  const items = CANDIDATES.map((c, i) => ({ value: i, label: `sampled: ${c}` }));
  return mountStageSelect(svg, { items, value: WRONG_TOKEN, onSelect: onPick });
}

// Repaints the stage for one model (toy-view.js modelFor); the pick groups persist, so keyboard focus survives.
export function paintStage(svg, model, { onPolicy, picks }) {
  const { cell, firstRowY, gap, headY } = TOY_STAGE;
  [...svg.children].filter((c) => !c.classList.contains('stage-item')).forEach((c) => c.remove());
  picks.sync(model.sampled);
  CANDIDATES.forEach((c, i) => { const node = picks.node(i); node.replaceChildren(); node.setAttribute('display', 'none'); });
  G.hatchFill(svg);
  CANDIDATES.forEach((c, i) => svgText(svg, cellX(i) + cell / 2, headY, c, { 'text-anchor': 'middle' }));
  model.teachers.forEach((teacher, k) => {
    const y = firstRowY + k * (cell + gap);
    row(svg, { y, values: teacher.probs, link: 't', maxAbs: PROB_MAX_ABS, attrs: { 'data-teacher-row': teacher.short } });
    rowLabel(svg, y, teacher.name);
  });
  const studentY = firstRowY + model.teachers.length * (cell + gap) + gap;
  rowLabel(svg, studentY, 'student');
  if (onPolicy) studentCells(picks, svg, { y: studentY, values: model.student, sampled: model.sampled });
  else row(svg, { y: studentY, values: model.student, link: 's', maxAbs: PROB_MAX_ABS, cellOpacity: (i) => (model.method === 'traces' && i > 0 ? DIM : 1) });
  if (model.method === 'traces') selection(svg, 0, studentY);
  if (onPolicy) {
    const rewardY = studentY + cell + gap * 2;
    row(svg, { y: rewardY, values: model.rewards, link: 'r', maxAbs: REWARD_MAX_ABS, format: (v) => signed(v, 2) });
    rowLabel(svg, rewardY, 'reward');
    selection(svg, model.sampled, rewardY);
  }
}
