// distillation's hand-authored stand-in numbers (storyboard header, §4, §6, §11) and what follows from them. Pure, no DOM.
// The four candidates are the final tokens of rlvr-grpo's rows 1, 2, 4 and 6; every loss and reward is computed from them.
import { deepFreeze } from '@math/core.js';
import { GROUP_TOY, buildGroup, verifyFinalAnswer, groupAdvantages } from '@math/grpo.js';
import { opdTokenReward } from '@math/distill.js';
import { tokenLoss, klDivergence } from '@math/lm.js';

export const CANDIDATES = Object.freeze(['56', '54', '48', '63']);
export const PROMPT = Object.freeze(['7', '×', '8', '=']);
export const REWARD_MAX_ABS = 3; // the one reward scale on this page: chips, stage cells and toy cells alike (P3-R18)
export const PROB_MAX_ABS = 1; // the value scale for the printed probabilities
export const TRUE_TOKEN = 0; // the teacher wrote 56 (index into CANDIDATES)
export const WRONG_TOKEN = 1; // the followed item from frame 4: the student's own 54

export const TEACHERS = deepFreeze({ math: [0.90, 0.05, 0.03, 0.02], chat: [0.60, 0.20, 0.10, 0.10] });
export const STUDENTS = deepFreeze({ unsure: [0.40, 0.30, 0.20, 0.10], wrong: [0.10, 0.70, 0.10, 0.10], near: [0.85, 0.07, 0.05, 0.03] });
export const DEFAULT_STUDENT = STUDENTS.unsure;
export const DEFAULT_TEACHER = TEACHERS.math;

// Frame 6: rlvr-grpo's row 2 (`7 × 8 = 54`) from GROUP_TOY, and its GRPO advantage (the same on every token).
const GROUP = buildGroup(2, GROUP_TOY);
export const ROW_2 = deepFreeze(GROUP[1]);
export const ROW_2_ADVANTAGE = groupAdvantages(GROUP.map((tokens) => verifyFinalAnswer(tokens, GROUP_TOY.target)))[1];
// The per-token teacher / student probabilities for row 2 (stand-ins): 7 · × · 8 · = · 54.
export const ROW_2_PAIRS = deepFreeze([[0.50, 0.50], [0.95, 0.90], [0.98, 0.95], [0.97, 0.97], [0.05, 0.30]]);
export const ROW_2_REWARDS = deepFreeze(ROW_2_PAIRS.map(([teacher, student]) => opdTokenReward(teacher, student)));
export const FOLLOWED_TOKEN_INDEX = ROW_2.length - 1; // the `54` token of row 2

// What the stage prints for the default pair (unsure student, math teacher); every value is computed, none typed.
const sum = (values) => values.reduce((total, v) => total + v, 0);
export const TRACE_LOSS = tokenLoss(DEFAULT_STUDENT[0]); // −ln 0.40
export const FORWARD_KL_TERMS = deepFreeze(DEFAULT_TEACHER.map((t, i) => t * Math.log(t / DEFAULT_STUDENT[i])));
export const FORWARD_KL = klDivergence(DEFAULT_TEACHER, DEFAULT_STUDENT);
export const REWARDS = deepFreeze(DEFAULT_STUDENT.map((s, i) => opdTokenReward(DEFAULT_TEACHER[i], s)));
export const EXPECTED_REWARD = -klDivergence(DEFAULT_STUDENT, DEFAULT_TEACHER);
export const KL_TERMS_SUM = sum(FORWARD_KL_TERMS);
