// distillation frames 6–9: dense versus one-number grades, merging specialists, forgetting, and the small-model recipe.
// Each draw is a pure function of progress; the rewards come from numbers.js (opdTokenReward, groupAdvantages).
import * as G from '@shared/glyphs.js';
import { ROW_2, ROW_2_ADVANTAGE, ROW_2_REWARDS, FOLLOWED_TOKEN_INDEX } from './numbers.js';
import { fixed, signed } from './format.js';
import { CELL, CHIP_H, seg, lerp, layer, note, rewardCell, rewardChip } from './stage.js';

const GRID = Object.freeze({ x: 196, pitch: 56, grpoChipY: 50, grpoCellY: 80, opdChipY: 170, opdCellY: 200 });
const LABEL_X = 14;
const gridX = (i) => GRID.x + i * GRID.pitch;
const chipAt = (i) => gridX(i) + (CELL - G.tokenWidth(ROW_2[i])) / 2;
const sel = (parent, i, y, w, h, x) => G.selectionMark(parent, { x, y, w, h });

// One row of the answer: chips on the reward scale, the grade of each under it. `reach(i)` is how far chip i has taken its colour.
function answerRow(svg, { chipY, cellY, rewards, reach }) {
  rewards.forEach((v, i) => {
    const r = reach(i);
    rewardChip(svg, { x: chipAt(i), y: chipY, text: ROW_2[i], v, reach: r });
    if (r > 0) rewardCell(svg, { x: gridX(i), y: cellY, v, opacity: r });
  });
  const followed = FOLLOWED_TOKEN_INDEX;
  sel(svg, followed, chipY, G.tokenWidth(ROW_2[followed]), CHIP_H, chipAt(followed));
  if (reach(followed) > 0) sel(layer(svg, reach(followed)), followed, cellY, CELL, CELL, gridX(followed));
}

const threeDecimals = () => ROW_2.map((token, i) => `${token} ${signed(ROW_2_REWARDS[i], 3)}`).join(' · ');

// Frame 6 (key frame): GRPO gave every token the same −0.58; the teacher's grades land on 54.
export function drawFrame6(svg, p) {
  const grpo = ROW_2.map(() => ROW_2_ADVANTAGE);
  note(svg, LABEL_X, GRID.grpoChipY - 22, 'row 2: 7 × 8 = 54');
  note(svg, LABEL_X, GRID.grpoChipY + 16, 'GRPO (rlvr-grpo)');
  answerRow(svg, { chipY: GRID.grpoChipY, cellY: GRID.grpoCellY, rewards: grpo, reach: () => 1 });
  note(svg, gridX(ROW_2.length - 1) + CELL + 12, GRID.grpoCellY + 24, `${fixed(ROW_2_ADVANTAGE, 2)} × ${ROW_2.length}`);
  note(svg, LABEL_X, GRID.opdChipY + 16, 'on-policy distillation');
  answerRow(svg, { chipY: GRID.opdChipY, cellY: GRID.opdCellY, rewards: ROW_2_REWARDS, reach: (i) => seg(p, 0.05 + i * 0.16, 0.25 + i * 0.16) });
  note(layer(svg, seg(p, 0.7, 0.9)), gridX(0), GRID.opdCellY + CELL + 28, 'different units: compare the shape, not the size');
  note(layer(svg, seg(p, 0.8, 1)), gridX(0), GRID.opdCellY + CELL + 50, threeDecimals());
}

const SPECIALISTS = Object.freeze([
  { label: 'math & code', prompt: 'solve 7 × 8' },
  { label: 'agents', prompt: 'book a flight' },
  { label: 'chat', prompt: 'write a poem' },
]);
const MERGE = Object.freeze({ x: 14, pitch: 201, w: 150, y: 30, h: 36, promptY: 150, studentY: 248, studentW: 170 });
const merge = (k) => ({ x: MERGE.x + k * MERGE.pitch, cx: MERGE.x + k * MERGE.pitch + MERGE.w / 2 });
const STUDENT_X = (580 - MERGE.studentW) / 2;
const MERGE_NOTE = ['DeepSeek-V4\'s specialists are still trained with SFT and GRPO;', 'only the final mixed-RL stage was replaced.'];

// Frame 7: each prompt is graded by the teacher for its domain; one student learns every specialist's skill.
export function drawFrame7(svg, p) {
  const step = 0.28;
  SPECIALISTS.forEach(({ label, prompt }, k) => {
    const a = k * step;
    const { x, cx } = merge(k);
    const live = p >= a && p < a + step;
    G.block(svg, { x, y: MERGE.y, w: MERGE.w, h: MERGE.h, label, state: live ? 'active' : 'idle' });
    const drop = seg(p, a, a + 0.08);
    const chipW = G.tokenWidth(prompt);
    G.token(layer(svg, drop), { x: cx - chipW / 2, y: lerp(MERGE.promptY - 14, MERGE.promptY, drop), text: prompt });
    const grade = seg(p, a + 0.06, a + 0.18);
    if (grade > 0) G.flow(svg, { from: [cx, MERGE.y + MERGE.h + 2], to: [cx, MERGE.promptY - 4], carry: 'gradient', progress: grade });
    const toStudent = seg(p, a + 0.16, a + 0.28);
    if (toStudent > 0) G.flow(svg, { from: [cx, MERGE.promptY + CHIP_H + 2], to: [STUDENT_X + 25 + k * 60, MERGE.studentY - 2], carry: 'token', progress: toStudent });
  });
  G.block(svg, { x: STUDENT_X, y: MERGE.studentY, w: MERGE.studentW, h: 36, label: 'student', state: 'idle' });
  const mark = layer(svg, seg(p, 0.85, 1));
  MERGE_NOTE.forEach((line, i) => note(mark, 14, 318 + i * 18, line));
}

const SKILLS = Object.freeze(['math', 'code', 'agents']);
const FORGET = Object.freeze({ x: 14, pitch: 150, w: 120, h: 34, rlY: 70, allY: 200 });

// Frame 8: one domain after another erases the earlier skills; distilling from all specialists at once keeps them.
export function drawFrame8(svg, p) {
  note(svg, FORGET.x, FORGET.rlY - 14, 'RL one domain after another');
  note(svg, FORGET.x, FORGET.allY - 14, 'distill from all specialists at once');
  SKILLS.forEach((skill, k) => {
    const x = FORGET.x + k * FORGET.pitch;
    const fade = k < SKILLS.length - 1 ? seg(p, 0.15 + k * 0.3, 0.4 + k * 0.3) : 0;
    G.block(layer(svg, lerp(1, 0.35, fade)), { x, y: FORGET.rlY, w: FORGET.w, h: FORGET.h, label: skill });
    if (k > 0) G.flow(svg, { from: [x - 28, FORGET.rlY + FORGET.h / 2], to: [x - 4, FORGET.rlY + FORGET.h / 2], carry: 'activation', progress: 1 });
    G.block(svg, { x, y: FORGET.allY, w: FORGET.w, h: FORGET.h, label: skill });
  });
  note(layer(svg, seg(p, 0.6, 0.9)), FORGET.x, FORGET.rlY + FORGET.h + 20, 'earlier skills fade');
  note(layer(svg, seg(p, 0.5, 0.8)), FORGET.x, FORGET.allY + FORGET.h + 20, 'each teacher\'s peak kept (MiMo-V2-Flash)');
  note(layer(svg, seg(p, 0.75, 1)), FORGET.x, 300, 'GLM-5 distills from its own earlier checkpoints to undo this.');
}

const STEPS = Object.freeze(['pretrain small', 'SFT on teacher traces', 'on-policy distillation', 'short RLVR (optional)']);
const STRIP = Object.freeze({ x: 40, w: 230, h: 38, y: 24, pitch: 80 });

// Frame 9: the four steps that make a small model, lighting in order.
export function drawFrame9(svg, p) {
  STEPS.forEach((label, k) => {
    const y = STRIP.y + k * STRIP.pitch;
    const lit = seg(p, k * 0.22, k * 0.22 + 0.1);
    G.block(svg, { x: STRIP.x, y, w: STRIP.w, h: STRIP.h, label, state: lit >= 1 ? 'active' : 'dim' });
    if (k < STEPS.length - 1) G.flow(svg, { from: [STRIP.x + STRIP.w / 2, y + STRIP.h + 2], to: [STRIP.x + STRIP.w / 2, y + STRIP.pitch - 4], carry: 'activation', progress: seg(p, k * 0.22 + 0.08, k * 0.22 + 0.22) });
  });
}
