// distillation frames 1–5: the teacher and student rows, learning from the teacher's text, its whole row, and the branch to
// on-policy distillation. Each draw is a pure function of progress; every number comes from numbers.js (math/ functions).
import * as G from '@shared/glyphs.js';
import {
  CANDIDATES, DEFAULT_STUDENT as STUDENT, DEFAULT_TEACHER as TEACHER, TRUE_TOKEN, WRONG_TOKEN, TRACE_LOSS, FORWARD_KL, FORWARD_KL_TERMS, REWARDS, EXPECTED_REWARD,
} from './numbers.js';
import { fixed, signed, prob } from './format.js';
import {
  COL, ROW, SAMPLE, CELL, CHIP_H, seg, lerp, cellX, cellCenter, layer, note, typed, promptChips, teacherBlock, studentBlock, candidateHeads, probRow,
  rewardCell,
} from './stage.js';

const DIM = 0.3; // opacity of what the frame does not use
const SEL_AT = 0.85; // the selection outline arrives once the cells it marks are drawn
const ROW_NOTE_X = COL.student - 12;
const CHIP_X = cellCenter(COL.student, WRONG_TOKEN) - G.tokenWidth(CANDIDATES[WRONG_TOKEN]) / 2; // the sampled `54` chip, under its cell
const SAMPLE_TEXT = `p(${CANDIDATES[WRONG_TOKEN]}) = ${prob(STUDENT[WRONG_TOKEN])}`;
const GRADE = REWARDS[WRONG_TOKEN];

const teacherRow = (svg, o = {}) => probRow(svg, { x: COL.teacher, y: ROW.vec, values: TEACHER, link: 't', ...o });
const studentRow = (svg, o = {}) => probRow(svg, { x: COL.student, y: ROW.vec, values: STUDENT, link: 's', ...o });
const heads = (svg, opacity = 1) => { candidateHeads(svg, COL.teacher, opacity); candidateHeads(svg, COL.student, opacity); };
const staggered = (p, a, b) => (i) => seg(p, lerp(a, b, i / 5), lerp(a, b, (i + 2) / 5));

// Frame 1: the teacher's row fills, then the student's; the `56` cells take the selection outline.
export function drawFrame1(svg, p) {
  const sel = seg(p, SEL_AT, 1);
  promptChips(svg);
  teacherBlock(svg, { opacity: seg(p, 0, 0.15) });
  studentBlock(svg, { opacity: seg(p, 0, 0.15) });
  heads(svg, seg(p, 0, 0.15));
  teacherRow(svg, { cellOpacity: staggered(p, 0.05, 0.45), follow: TRUE_TOKEN, selOpacity: sel });
  studentRow(svg, { cellOpacity: staggered(p, 0.45, 0.85), follow: TRUE_TOKEN, selOpacity: sel });
}

// Frame 2: the teacher writes 56; only the student's 56 cell counts and the other three dim; loss −ln 0.40 = 0.916.
export function drawFrame2(svg, p) {
  const dim = seg(p, 0.3, 0.7);
  const chipIn = seg(p, 0, 0.2);
  const chipX = COL.teacher + CELL / 2 - G.tokenWidth(CANDIDATES[TRUE_TOKEN]) / 2;
  promptChips(svg);
  teacherBlock(svg);
  studentBlock(svg);
  heads(svg);
  teacherRow(svg, { follow: TRUE_TOKEN });
  studentRow(svg, { cellOpacity: (i) => (i === TRUE_TOKEN ? 1 : lerp(1, DIM, dim)), follow: TRUE_TOKEN });
  G.token(layer(svg, chipIn), { x: chipX, y: SAMPLE.chipY, text: CANDIDATES[TRUE_TOKEN] });
  note(layer(svg, chipIn), COL.teacher, SAMPLE.chipY + CHIP_H + 18, `teacher wrote ${CANDIDATES[TRUE_TOKEN]}`);
  const reach = seg(p, 0.1, 0.6);
  if (reach > 0) G.flow(svg, { from: [chipX + G.tokenWidth(CANDIDATES[TRUE_TOKEN]) + 6, SAMPLE.chipY + 6], to: [cellCenter(COL.student, TRUE_TOKEN), ROW.vec + CELL + 6], carry: 'token', progress: reach });
  note(layer(svg, seg(p, 0.7, 1)), COL.student, SAMPLE.chipY + 24, `loss −ln ${prob(STUDENT[TRUE_TOKEN])} = ${fixed(TRACE_LOSS, 3)}`);
}

// Frame 3: a copy of the teacher's row slides under the student's; KL(teacher ‖ student) = 0.551, the 56 term shown.
export function drawFrame3(svg, p) {
  const slide = seg(p, 0.05, 0.55);
  const light = seg(p, 0, 0.2);
  const term = FORWARD_KL_TERMS[TRUE_TOKEN];
  promptChips(svg);
  teacherBlock(svg);
  studentBlock(svg);
  heads(svg);
  teacherRow(svg, { opacity: lerp(1, DIM, slide) });
  studentRow(svg, { cellOpacity: (i) => (i === TRUE_TOKEN ? 1 : lerp(DIM, 1, light)), follow: TRUE_TOKEN });
  const copy = probRow(svg, { x: lerp(COL.teacher, COL.student, slide), y: lerp(ROW.vec, ROW.under, slide), values: TEACHER, link: 't', follow: TRUE_TOKEN });
  copy.setAttribute('data-copy', '');
  const at = seg(p, 0.55, 0.7);
  note(layer(svg, at), ROW_NOTE_X, ROW.vec + CELL / 2 + 4, 'student', { anchor: 'end' });
  note(layer(svg, at), ROW_NOTE_X, ROW.under + CELL / 2 + 4, 'teacher', { anchor: 'end' });
  note(layer(svg, seg(p, 0.6, 0.75)), COL.student, ROW.under + CELL + 24, `${prob(TEACHER[TRUE_TOKEN])} ln ${(TEACHER[TRUE_TOKEN] / STUDENT[TRUE_TOKEN]).toFixed(2)} = ${fixed(term, 3)}`);
  note(svg, COL.student, ROW.under + CELL + 44, typed(`KL(teacher ‖ student) = ${fixed(FORWARD_KL, 3)}`, seg(p, 0.7, 1)));
}

// Frame 4 (branch): the student samples its own `54`, somewhere it never practised.
export function drawFrame4(svg, p) {
  const out = seg(p, 0, 0.15);
  promptChips(svg);
  teacherBlock(svg, { opacity: lerp(1, DIM, out) });
  studentBlock(svg);
  heads(svg);
  teacherRow(svg, { opacity: lerp(1, DIM, out) });
  studentRow(svg, { follow: WRONG_TOKEN });
  note(layer(svg, seg(p, 0.1, 0.3)), COL.teacher, 215, 'what if the student writes its own answer?');
  const reach = seg(p, 0.25, 0.65);
  if (reach > 0) G.flow(svg, { from: [cellCenter(COL.student, WRONG_TOKEN), ROW.vec + CELL + 4], to: [cellCenter(COL.student, WRONG_TOKEN), SAMPLE.chipY - 2], carry: 'token', progress: reach });
  const chip = layer(svg, seg(p, 0.3, 0.6));
  G.token(chip, { x: CHIP_X, y: SAMPLE.chipY, text: CANDIDATES[WRONG_TOKEN] });
  G.selectionMark(chip, { x: CHIP_X, y: SAMPLE.chipY, w: G.tokenWidth(CANDIDATES[WRONG_TOKEN]), h: CHIP_H });
  const labels = layer(svg, seg(p, 0.6, 0.9));
  note(labels, COL.student, SAMPLE.chipY + CHIP_H + 18, 'student\'s own sample');
  note(labels, COL.student, SAMPLE.chipY + CHIP_H + 36, 'never in its training text');
  note(labels, COL.student, SAMPLE.chipY + CHIP_H + 54, SAMPLE_TEXT);
}

// Frame 5: the teacher grades the student's sample: 54 earns ln 0.05 − ln 0.30 = −1.792; the chip cools to that value.
export function drawFrame5(svg, p) {
  const reach = seg(p, 0.1, 0.55);
  const t = (a, b) => seg(p, a, b);
  const others = [0, 2, 3].map((i) => `${CANDIDATES[i]} ${signed(REWARDS[i], 2)}`).join(' · ');
  promptChips(svg);
  teacherBlock(svg, { state: 'active' });
  studentBlock(svg);
  heads(svg);
  teacherRow(svg);
  studentRow(svg, { follow: WRONG_TOKEN });
  if (reach > 0) G.flow(svg, { from: [COL.teacher + 170, ROW.block + 20], to: [CHIP_X - 4, SAMPLE.chipY + 8], carry: 'gradient', progress: reach });
  const chip = G.svgEl('g', { 'data-link': 's' }, svg);
  G.token(chip, { x: CHIP_X, y: SAMPLE.chipY, text: CANDIDATES[WRONG_TOKEN], fill: G.valueColor(GRADE * t(0.3, 0.7), 3) });
  G.selectionMark(svg, { x: CHIP_X, y: SAMPLE.chipY, w: G.tokenWidth(CANDIDATES[WRONG_TOKEN]), h: CHIP_H });
  const gradeIn = t(0.6, 0.85);
  if (gradeIn > 0) {
    rewardCell(svg, { x: cellX(COL.student, WRONG_TOKEN), y: SAMPLE.cellY, v: GRADE, opacity: gradeIn });
    G.selectionMark(layer(svg, gradeIn), { x: cellX(COL.student, WRONG_TOKEN), y: SAMPLE.cellY, w: CELL, h: CELL });
  }
  note(layer(svg, t(0.7, 0.9)), COL.student, SAMPLE.cellY + CELL + 18, others);
  const formulas = layer(svg, t(0.75, 1));
  note(formulas, COL.teacher, 262, `${CANDIDATES[WRONG_TOKEN]}: ln ${prob(TEACHER[WRONG_TOKEN])} − ln ${prob(STUDENT[WRONG_TOKEN])} = ${fixed(GRADE, 3)}`);
  note(formulas, COL.teacher, 282, `${CANDIDATES[TRUE_TOKEN]}: ln ${prob(TEACHER[TRUE_TOKEN])} − ln ${prob(STUDENT[TRUE_TOKEN])} = ${signed(REWARDS[TRUE_TOKEN], 3)}`);
  note(formulas, COL.teacher, 306, `student's expected reward ${signed(EXPECTED_REWARD, 3)} (= minus the reverse KL)`);
}
