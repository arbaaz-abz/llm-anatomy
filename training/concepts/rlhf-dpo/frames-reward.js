// rlhf-dpo frames 1–4: the preference pair, the reward model that learns it, the policy's fresh answer, the critic.
import * as G from '@shared/glyphs.js';
import { PROMPT, ANSWER_A, ANSWER_B, ANSWER_FRESH, REWARD_A, REWARD_B, REWARD_FRESH, CRITIC_VALUE, ADVANTAGE, BT } from './numbers.js';
import { fmt1, fmt3 } from './format.js';
import { CELL, ROW_X, lerp, seg, typed, layer, note, select, answerRow, rowMark, rowWidth, chipRow, route, signed } from './stage.js';

export const ROW_A_Y = 64;
export const ROW_B_Y = 112;
const BADGE_X = 366;
const SCORE_MAX_ABS = 1.5;
const scoreCell = (parent, { x, y, v }) => G.vector(parent, { x, y, values: [v], cell: CELL, orient: 'row', maxAbs: SCORE_MAX_ABS, format: fmt1 });
// A chip row that types in chip by chip over [a, b].
const typedIn = (p, a, b, n) => (i) => seg(p, lerp(a, b, i / n), lerp(a, b, (i + 1) / n));

// ---- frame 1: the person's choice ----
export function drawFrame1(svg, p) {
  note(svg, 16, 30, PROMPT, { cls: '' });
  answerRow(svg, { y: ROW_A_Y, letter: 'A', tokens: ANSWER_A, link: 'a', shown: typedIn(p, 0, 0.35, ANSWER_A.length) });
  answerRow(svg, { y: ROW_B_Y, letter: 'B', tokens: ANSWER_B, shown: typedIn(p, 0.1, 0.4, ANSWER_B.length) });
  G.block(layer(svg, seg(p, 0.1, 0.3)), { x: 470, y: 78, w: 96, h: 40, label: 'person' });
  const stamp = seg(p, 0.45, 0.65);
  const words = seg(p, 0.6, 0.8);
  if (stamp > 0) {
    G.verdict(layer(svg, stamp), { x: BADGE_X, y: ROW_A_Y + 12, ok: true });
    G.verdict(layer(svg, stamp), { x: BADGE_X, y: ROW_B_Y + 12, ok: false });
  }
  if (words > 0) {
    note(layer(svg, words), BADGE_X + 16, ROW_A_Y + 16, 'chosen', { cls: '' });
    note(layer(svg, words), BADGE_X + 16, ROW_B_Y + 16, 'rejected', { cls: '' });
  }
  rowMark(svg, ROW_A_Y, ANSWER_A, seg(p, 0.7, 0.85));
  note(layer(svg, seg(p, 0.85, 0.95)), 16, 190, 'pair: A ≻ B', { cls: '' });
}

// ---- frame 2: the reward model ----
const RM = Object.freeze({ x: ROW_X, y: 176, w: 290, h: 40 });
const SCORES_X = 350;

export function drawFrame2(svg, p) {
  answerRow(svg, { y: ROW_A_Y, letter: 'A', tokens: ANSWER_A, link: 'a' });
  answerRow(svg, { y: ROW_B_Y, letter: 'B', tokens: ANSWER_B });
  rowMark(svg, ROW_A_Y, ANSWER_A);
  G.block(layer(svg, seg(p, 0, 0.2)), { ...RM, label: 'reward model (SFT copy + score head)' });
  route(layer(svg, seg(p, 0.05, 0.15)), [[310, ROW_A_Y + CHIP_BELOW], [310, RM.y - 2]], { t: seg(p, 0.1, 0.45) });
  route(layer(svg, seg(p, 0.05, 0.15)), [[240, ROW_B_Y + CHIP_BELOW], [240, RM.y - 2]], { t: seg(p, 0.1, 0.45) });
  const fill = seg(p, 0.45, 0.65);
  if (fill > 0) {
    const g = layer(svg, fill);
    note(g, SCORES_X + CELL / 2, RM.y - 6, 'r(A)', { anchor: 'middle' });
    note(g, SCORES_X + CELL + 8 + CELL / 2, RM.y - 6, 'r(B)', { anchor: 'middle' });
    scoreCell(g, { x: SCORES_X, y: RM.y, v: REWARD_A });
    scoreCell(g, { x: SCORES_X + CELL + 8, y: RM.y, v: REWARD_B });
  }
  const t = seg(p, 0.65, 0.95);
  const lines = [
    `P(A preferred) ${fmt3(BT.right.pChosen)} · loss ${fmt3(BT.right.loss)}`,
    `= σ(${fmt1(REWARD_A)} − (${fmt1(REWARD_B)})) = σ(${fmt1(REWARD_A - REWARD_B)});  loss = −ln ${fmt3(BT.right.pChosen)}`,
    `if it had the order wrong: loss ${fmt3(BT.flipped.loss)}`,
  ];
  lines.forEach((line, i) => note(svg, ROW_X, 252 + i * 20, typed(line, seg(t, i / 3, (i + 1) / 3)), { cls: i === 0 ? '' : 'g-label' }));
}
const CHIP_BELOW = 24; // a chip row's height: arrows start right under it

// ---- frames 3–4: the policy writes, the reward model scores, the critic predicts ----
const POLICY = Object.freeze({ x: ROW_X, y: 24, w: 250, h: 34 });
const FRESH_Y = 90;
const RM3 = Object.freeze({ x: ROW_X, y: 160, w: 290, h: 40 });
const CRITIC = Object.freeze({ x: ROW_X, y: 226, w: 290, h: 40 });
const RESULT_X = 350;
const FRESH_END = ROW_X + rowWidth(ANSWER_FRESH);

function freshAnswer(parent, { fillT = 0, shown }) {
  chipRow(parent, { y: FRESH_Y, tokens: ANSWER_FRESH, shown, fill: fillT > 0 ? () => G.valueColor(ADVANTAGE * fillT, 1) : () => undefined });
}

export function drawFrame3(svg, p) {
  G.block(svg, { ...POLICY, label: 'policy: the model being trained', state: 'active' });
  route(layer(svg, seg(p, 0, 0.1)), [[150, POLICY.y + POLICY.h], [150, FRESH_Y - 2]], { t: seg(p, 0.05, 0.4) });
  freshAnswer(svg, { shown: typedIn(p, 0.25, 0.5, ANSWER_FRESH.length) });
  select(svg, { x: ROW_X - 3, y: FRESH_Y - 3, w: rowWidth(ANSWER_FRESH) + 6, h: CHIP_H_PAD }, seg(p, 0.5, 0.6));
  G.block(layer(svg, seg(p, 0.5, 0.6)), { ...RM3, label: 'reward model' });
  route(layer(svg, seg(p, 0.5, 0.6)), [[150, FRESH_Y + CHIP_BELOW + 2], [150, RM3.y - 2]], { t: seg(p, 0.55, 0.85) });
  const score = seg(p, 0.85, 1);
  if (score > 0) {
    note(layer(svg, score), RESULT_X + CELL / 2, RM3.y - 6, 'score', { anchor: 'middle' });
    scoreCell(layer(svg, score), { x: RESULT_X, y: RM3.y, v: REWARD_FRESH });
  }
}
const CHIP_H_PAD = 30;

export function drawFrame4(svg, p) {
  G.block(svg, { ...POLICY, label: 'policy: the model being trained' });
  const warm = seg(p, 0.5, 0.85);
  freshAnswer(svg, { fillT: warm, shown: () => 1 });
  select(svg, { x: ROW_X - 3, y: FRESH_Y - 3, w: rowWidth(ANSWER_FRESH) + 6, h: CHIP_H_PAD });
  G.block(svg, { ...RM3, label: 'reward model' });
  note(svg, RESULT_X + CELL / 2, RM3.y - 6, 'score', { anchor: 'middle' });
  scoreCell(svg, { x: RESULT_X, y: RM3.y, v: REWARD_FRESH });
  G.block(layer(svg, seg(p, 0.1, 0.25)), { ...CRITIC, label: 'critic (value model)', state: 'active' });
  const value = seg(p, 0.25, 0.4);
  if (value > 0) {
    scoreCell(layer(svg, value), { x: RESULT_X, y: CRITIC.y, v: CRITIC_VALUE });
    note(layer(svg, value), RESULT_X + CELL / 2, CRITIC.y + CELL + 14, 'usual score for this prompt', { anchor: 'middle' });
  }
  const sub = seg(p, 0.4, 0.55);
  note(svg, ROW_X, 300, typed(`beat it by ${signed(ADVANTAGE)}`, sub), { cls: '' });
  note(svg, ROW_X, 318, typed(`${fmt1(REWARD_FRESH)} − ${fmt1(CRITIC_VALUE)} = ${signed(ADVANTAGE)}`, sub));
  const send = seg(p, 0.55, 0.9);
  if (send > 0) route(layer(svg, seg(p, 0.55, 0.6)), [[RESULT_X + CELL + 4, CRITIC.y + CELL / 2], [548, CRITIC.y + CELL / 2], [548, FRESH_Y + 12], [FRESH_END + 6, FRESH_Y + 12]], { carry: 'gradient', t: send });
}
