// agentic-rl's hand-authored stand-ins (storyboard §5–§6) and what follows from them. Pure, no DOM.
// The group is rlvr-grpo's (GROUP_TOY, k = 2); the episode, durations and probabilities are this page's own.
import { deepFreeze } from '@math/core.js';
import { GROUP_TOY, buildGroup, groupAdvantages, verifyFinalAnswer } from '@math/grpo.js';

export const GROUP_K = 2;
export const ROWS = deepFreeze(buildGroup(GROUP_K, GROUP_TOY)); // 8 rows, 39 tokens
export const TOKEN_COUNT = ROWS.reduce((n, row) => n + row.length, 0);
export const REWARDS = deepFreeze(ROWS.map((tokens) => verifyFinalAnswer(tokens, GROUP_TOY.target)));
export const ADVANTAGES = deepFreeze(groupAdvantages(REWARDS));
export const A_MAX_ABS = 2.65; // the value scale for advantages, the same on rlvr-grpo

// Frames 1–3: row 1 as a tiny tool-using episode (whitespace-split). 8 policy tokens, 3 observation tokens.
export const EPISODE = deepFreeze([
  ...['<think>', 'use', 'python', '</think>', '<call>', 'print(7*8)', '</call>'].map((text) => ({ text, observation: false })),
  ...['<obs>', '56', '</obs>'].map((text) => ({ text, observation: true })),
  { text: '56', observation: false },
]);
export const POLICY_TOKENS = EPISODE.filter((t) => !t.observation).length; // 8
export const OBSERVATION_TOKENS = EPISODE.length - POLICY_TOKENS; // 3

// Episode minutes for rows 1–8 (frames 5–6 and the toy).
export const DURATIONS = deepFreeze([3, 2, 4, 16, 5, 3, 9, 6]);
export const GROUP_SIZE = DURATIONS.length;

// The followed token: row 4's second 48 (0-based row 3, position 7), on screen "row 4, token 8".
export const FOLLOWED = deepFreeze({ row: 3, pos: 7 });

// Engine probabilities for all 39 tokens (stand-ins). The trainer agrees exactly except the five tokens in TRAINER_BF16.
export const ENGINE_P = deepFreeze([
  [0.6, 0.9, 0.95, 0.9, 0.2],
  [0.6, 0.9, 0.95, 0.9, 0.1],
  [0.6, 0.1, 0.9, 0.9, 0.3],
  [0.6, 0.9, 0.95, 0.9, 0.1, 0.5, 0.3, 0.05],
  [0.6, 0.1, 0.9, 0.9, 0.2],
  [0.2],
  [0.6, 0.9, 0.95, 0.9, 0.05],
  [0.1, 0.9, 0.95, 0.9, 0.1],
]);
// The trainer's probability under BF16 where it differs from the engine's: "row:position" (0-based) → p.
export const TRAINER_BF16 = deepFreeze({ '0:4': 0.22, '4:4': 0.18, '3:7': 0.16, '5:0': 0.08, '6:4': 0.12 });

export const trainerBf16 = (row, pos) => TRAINER_BF16[`${row}:${pos}`] ?? ENGINE_P[row][pos];

// Tokens whose trainer probability equals the engine's (frame 7: "34 of 39 here").
export const AGREE_COUNT = TOKEN_COUNT - Object.keys(TRAINER_BF16).length;
