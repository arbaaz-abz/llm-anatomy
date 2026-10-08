// distillation toy stage: draws the teacher row(s), the student row and the reward row into one <svg>, and wires the
// selectable student cells (P3-R17: focusable role=button groups, Enter / Space select, arrows move focus, G.selectionMark).
import * as G from '@shared/glyphs.js';
import { CANDIDATES, PROB_MAX_ABS, REWARD_MAX_ABS } from './numbers.js';
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

// The student's cells as buttons: each is a focusable group that selects the token it stands for.
function studentButtons(parent, { y, values, sampled }) {
  const g = G.svgEl('g', { 'data-link': 's' }, parent);
  G.svgEl('rect', { class: 'g-frame', x: TOY_STAGE.x0 - 1, y: y - 1, width: 4 * TOY_STAGE.cell + 2, height: TOY_STAGE.cell + 2, rx: 3, fill: 'none', stroke: 'none' }, g);
  values.forEach((v, i) => {
    const button = G.svgEl('g', {
      role: 'button', tabindex: 0, 'aria-label': `sampled: ${CANDIDATES[i]}`, 'aria-pressed': String(i === sampled), 'data-sampled': i, class: 'toy-pick',
    }, g);
    G.cell(button, { x: cellX(i), y, size: TOY_STAGE.cell, v, maxAbs: PROB_MAX_ABS });
  });
  return g;
}

const selection = (parent, i, y) => G.selectionMark(parent, { x: cellX(i), y, w: TOY_STAGE.cell, h: TOY_STAGE.cell });
const focusedPick = (svg) => (svg.contains(document.activeElement) ? document.activeElement.dataset?.sampled ?? null : null);

// Repaints the stage for one model (toy-view.js modelFor). Keyboard focus on a student cell survives the repaint.
export function paintStage(svg, model, { onPolicy }) {
  const refocus = focusedPick(svg);
  const { cell, firstRowY, gap, headY } = TOY_STAGE;
  svg.replaceChildren();
  G.hatchFill(svg);
  CANDIDATES.forEach((c, i) => svgText(svg, cellX(i) + cell / 2, headY, c, { 'text-anchor': 'middle' }));
  model.teachers.forEach((teacher, k) => {
    const y = firstRowY + k * (cell + gap);
    row(svg, { y, values: teacher.probs, link: 't', maxAbs: PROB_MAX_ABS, attrs: { 'data-teacher-row': teacher.short } });
    rowLabel(svg, y, teacher.name);
  });
  const studentY = firstRowY + model.teachers.length * (cell + gap) + gap;
  rowLabel(svg, studentY, 'student');
  if (onPolicy) studentButtons(svg, { y: studentY, values: model.student, sampled: model.sampled });
  else row(svg, { y: studentY, values: model.student, link: 's', maxAbs: PROB_MAX_ABS, cellOpacity: (i) => (model.method === 'traces' && i > 0 ? DIM : 1) });
  if (model.method === 'traces') selection(svg, 0, studentY);
  if (onPolicy) {
    const rewardY = studentY + cell + gap * 2;
    row(svg, { y: rewardY, values: model.rewards, link: 'r', maxAbs: REWARD_MAX_ABS, format: (v) => signed(v, 2) });
    rowLabel(svg, rewardY, 'reward');
    selection(svg, model.sampled, studentY);
    selection(svg, model.sampled, rewardY);
  }
  if (refocus != null) svg.querySelector(`[data-sampled="${refocus}"]`)?.focus();
}

// Click, Enter / Space pick a token; ArrowLeft / ArrowRight move focus between the student cells (they stop at the ends).
export function bindPicks(svg, pick) {
  const cells = () => [...svg.querySelectorAll('[data-sampled]')];
  const targetOf = (event) => event.target.closest?.('[data-sampled]');
  const onClick = (event) => {
    const target = targetOf(event);
    if (target) pick(Number(target.dataset.sampled));
  };
  const onKey = (event) => {
    const target = targetOf(event);
    if (!target) return;
    const index = Number(target.dataset.sampled);
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); pick(index); return; }
    const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
    if (!step) return;
    event.preventDefault();
    cells()[Math.min(Math.max(index + step, 0), CANDIDATES.length - 1)]?.focus();
  };
  svg.addEventListener('click', onClick);
  svg.addEventListener('keydown', onKey);
  return () => { svg.removeEventListener('click', onClick); svg.removeEventListener('keydown', onKey); };
}
