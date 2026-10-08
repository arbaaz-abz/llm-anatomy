// rlhf-dpo frames 9–11: the four-model shelf, DPO on the same pair, and where each method lives in 2026.
import * as G from '@shared/glyphs.js';
import { ANSWER_A, ANSWER_B, PIPELINE_STAGES, STAGE_WIDTHS, DPO, DPO_DEFAULT } from './numbers.js';
import { fmt1, fmt2, fmt3, fmtChange } from './format.js';
import { CELL, lerp, seg, typed, layer, note, select, linked, answerRow, rowMark, route } from './stage.js';

// ---- frame 9: four models, one sampling loop ----
const SHELF = Object.freeze([
  { label: 'policy', x: 16 }, { label: 'reference', x: 160 }, { label: 'reward model', x: 304 }, { label: 'critic', x: 448 },
]);
const SHELF_BOX = Object.freeze({ y: 60, w: 116, h: 44 });
const LOOP_Y = 170;

export function drawFrame9(svg, p) {
  SHELF.forEach((m, i) => G.block(svg, { ...SHELF_BOX, x: m.x, label: m.label, state: p >= lerp(0.1, 0.5, i / 3) ? 'active' : 'dim' }));
  note(layer(svg, seg(p, 0.1, 0.5)), 218, 128, 'in memory at once', { cls: '', anchor: 'middle' });
  const policyX = SHELF[0].x + SHELF_BOX.w / 2;
  const rewardX = SHELF[2].x + SHELF_BOX.w / 2;
  const open = seg(p, 0.5, 0.6);
  if (open > 0) {
    const g = layer(svg, open);
    note(g, (policyX + rewardX) / 2, LOOP_Y - 8, 'answers', { anchor: 'middle' });
    note(g, (policyX + rewardX) / 2, LOOP_Y + 22, 'sample every step', { cls: '', anchor: 'middle' });
    route(g, [[policyX, SHELF_BOX.y + SHELF_BOX.h + 4], [policyX, LOOP_Y], [rewardX, LOOP_Y], [rewardX, SHELF_BOX.y + SHELF_BOX.h + 2]], { t: seg(p, 0.55, 0.95) });
  }
  const tally = seg(p, 0.9, 1);
  note(layer(svg, tally), 290, 250, '4 models', { cls: '', anchor: 'middle' });
  note(layer(svg, tally), 290, 270, '1 sampling loop per step', { cls: '', anchor: 'middle' });
}

// ---- frame 10: DPO on the pair from frame 1 ----
const ROW_A = 40;
const ROW_B = 128;
const BADGE_X = 366;
const CHANGE_Y = Object.freeze({ a: ROW_A + 32, b: ROW_B + 32 });

export function drawFrame10(svg, p) {
  answerRow(svg, { y: ROW_A, letter: 'A', tokens: ANSWER_A, link: 'a' });
  answerRow(svg, { y: ROW_B, letter: 'B', tokens: ANSWER_B });
  rowMark(svg, ROW_A, ANSWER_A);
  G.verdict(svg, { x: BADGE_X, y: ROW_A + 12, ok: true });
  G.verdict(svg, { x: BADGE_X, y: ROW_B + 12, ok: false });
  note(svg, BADGE_X + 16, ROW_A + 16, 'chosen', { cls: '' });
  note(svg, BADGE_X + 16, ROW_B + 16, 'rejected', { cls: '' });
  const fill = seg(p, 0.1, 0.4);
  [['a', DPO_DEFAULT.dChosen], ['b', DPO_DEFAULT.dRejected]].forEach(([key, v]) => {
    const g = layer(svg, fill);
    G.vector(g, { x: 34, y: CHANGE_Y[key], values: [v * fill], cell: CELL, orient: 'row', maxAbs: 1, format: () => fmtChange(v) });
    note(g, 34 + CELL + 10, CHANGE_Y[key] + 24, 'change in log-probability vs the reference');
  });
  const beta = linked(svg, 'beta', { x: 34, y: 226, w: 64, h: 18 });
  note(beta, 34, 240, `β = ${fmt1(DPO_DEFAULT.beta)}`, { cls: '' });
  const lines = [
    `implicit rewards: A ${fmt2(DPO.rewardChosen)} · B ${fmt2(DPO.rewardRejected)}`,
    `gap ${fmt2(DPO.rewardChosen)} − (${fmt2(DPO.rewardRejected)}) = ${fmt2(DPO.margin)}`,
    `loss −ln σ(${fmt2(DPO.margin)}) = ${fmt3(DPO.loss)}  (ln 2 = ${fmt3(Math.LN2)} at zero gap)`,
    `update weight σ(−${fmt2(DPO.margin)}) = ${fmt3(DPO.weight)}`,
  ];
  lines.forEach((line, i) => note(svg, 34, 262 + i * 20, typed(line, seg(p, 0.4 + i * 0.12, 0.52 + i * 0.12)), { cls: i === 2 ? '' : 'g-label' }));
  const dim = seg(p, 0.1, 0.4);
  G.block(layer(svg, 0.4 + 0.6 * (1 - dim)), { x: 420, y: 250, w: 140, h: 30, label: 'reward model', state: 'dim' });
  G.block(layer(svg, 0.4 + 0.6 * (1 - dim)), { x: 420, y: 288, w: 140, h: 30, label: 'critic', state: 'dim' });
  note(layer(svg, seg(p, 0.3, 0.5)), 490, 336, 'not needed', { anchor: 'middle' });
}

// ---- frame 11: where each lives in 2026 ----
const STRIP = Object.freeze({ x: 5, y: 40, h: 40, gap: 6 });
const stageX = (i) => STRIP.x + STAGE_WIDTHS.slice(0, i).reduce((s, w) => s + w + STRIP.gap, 0);
const TAG = Object.freeze({ y: 112, h: 28 });

function tag(parent, { cx, w, label, drop }) {
  const g = layer(parent, drop);
  G.block(g, { x: cx - w / 2, y: lerp(TAG.y - 30, TAG.y, drop), w, h: TAG.h, label, state: 'active' });
}

export function drawFrame11(svg, p) {
  note(svg, STRIP.x, 28, 'the six-stage pipeline', { cls: 'g-label' });
  PIPELINE_STAGES.forEach((label, i) => G.block(svg, { x: stageX(i), y: STRIP.y, w: STAGE_WIDTHS[i], h: STRIP.h, label: `${i + 1} ${label}`, state: 'dim' }));
  const dpoX = (stageX(2) + STAGE_WIDTHS[2] + stageX(3)) / 2;
  const rlhfX = stageX(5) + STAGE_WIDTHS[5] / 2;
  const dpo = seg(p, 0.1, 0.45);
  const rlhf = seg(p, 0.5, 0.85);
  if (dpo > 0) {
    tag(svg, { cx: dpoX, w: 56, label: 'DPO', drop: dpo });
    select(svg, { x: dpoX - 28, y: lerp(TAG.y - 30, TAG.y, dpo), w: 56, h: TAG.h }, dpo);
    note(layer(svg, seg(p, 0.35, 0.5)), dpoX, 164, 'Olmo 3 (reported)', { cls: '', anchor: 'middle' });
  }
  if (rlhf > 0) {
    tag(svg, { cx: rlhfX, w: 64, label: 'RLHF', drop: rlhf });
    note(layer(svg, seg(p, 0.75, 0.9)), 575, 196, 'Nemotron 3 Super: judge-model reward', { cls: '', anchor: 'end' });
  }
  route(layer(svg, dpo), [[dpoX, TAG.y - 2], [dpoX, STRIP.y + STRIP.h + 2]], { t: seg(p, 0.3, 0.45) });
  route(layer(svg, rlhf), [[rlhfX, TAG.y - 2], [rlhfX, STRIP.y + STRIP.h + 2]], { t: seg(p, 0.7, 0.85) });
}
