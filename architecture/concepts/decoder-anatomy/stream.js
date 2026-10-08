// decoder-anatomy's hand-authored stand-in numbers (storyboard §4, ruling 7) and what follows from them. Pure, no DOM.
// Quarter-grid constants so every add on the stage is one line of arithmetic; softmax and the attention row come from math/.
import { deepFreeze, randomMatrix, softmax } from '@math/core.js';
import { TOY, attentionHead } from '@math/attention.js';

export const TOKENS = TOY.tokens; // The cat sat down, shared with `attention`
export const VOCAB = Object.freeze(['The', 'cat', 'sat', 'down', 'on', '.', 'and', 'the', 'mat', 'a', 'dog', 'ran', 'up', 'big', 'was', 'then']);
export const D_MODEL = 8;
export const MAX_ABS = 2; // the value scale for the stream and the scores
export const PROB_MAX_ABS = 0.4; // the value scale for probabilities
export const SAT = 2; // row index of "sat", the row followed in frames 4–6
export const DOWN = 3; // row index of "down", the row followed in frame 8

export const X = deepFreeze([
  [1, 0, -0.5, 0, 0.5, 0, 0, -0.5],
  [0.5, 0.5, 0, -1, 0, 1, 0.5, 0],
  [0, 1, 0.5, 0, -0.5, 1, 0, 0.5],
  [0.5, 1, 0, 0, 0, -0.5, 1, 0.5],
]);
export const A_SAT = deepFreeze([0.25, 0.5, -0.25, 0.5, 0, -0.5, 0.25, 0]); // attention delta for sat, block 1
export const M_SAT = deepFreeze([0, 0.25, 0, -0.5, 0.5, 0, -0.25, 0.25]); // MLP delta for sat, block 1

const snapHalf = (v) => Math.round(v * 2) / 2 + 0; // + 0 turns −0 into 0
// The embedding table E [16 × 8]: the four lit rows are X; the 12 unused rows are seeded and only ever drawn dim.
export const E = deepFreeze([...X, ...randomMatrix(12, D_MODEL, 11, 1).map((row) => row.map(snapHalf))]);

export const addRows = (a, b) => {
  if (a.length !== b.length) throw new RangeError(`addRows: length ${a.length} vs ${b.length}`);
  return a.map((v, i) => v + b[i]);
};
export const X1_SAT = deepFreeze(addRows(X[SAT], A_SAT)); // x′ = x + a
export const X2_SAT = deepFreeze(addRows(X1_SAT, M_SAT)); // x″ = x′ + m

// Head A's weights row for "sat" from math/attention.js: a static reference here, computed on `attention`.
export const WEIGHTS_SAT = deepFreeze(attentionHead(TOY.heads.A.Q, TOY.heads.A.K, TOY.heads.A.V).weights[SAT]);

// Scores for the last position after block 2: on 2.0 · "." 1.5 · and 0.5 · the 0.0 · the other 12 words −1.0.
const SCORE_OF = { on: 2, '.': 1.5, and: 0.5, the: 0 };
const OTHER_SCORE = -1;
export const LOGITS = deepFreeze(VOCAB.map((w) => SCORE_OF[w] ?? OTHER_SCORE));
export const PROBS = deepFreeze(softmax(LOGITS));
export const SHOWN_WORDS = Object.freeze(['on', '.', 'and', 'the']);
export const OTHERS = VOCAB.length - SHOWN_WORDS.length; // 12, collapsed into one cell (README lesson 18)
const shown = SHOWN_WORDS.map((w) => VOCAB.indexOf(w));
const otherIndex = VOCAB.findIndex((w) => !SHOWN_WORDS.includes(w));
// Five cells: the four named words, then one "12 others" cell holding the value of each of them.
export const SCORE_CELLS = deepFreeze([...shown.map((i) => LOGITS[i]), LOGITS[otherIndex]]);
export const PROB_CELLS = deepFreeze([...shown.map((i) => PROBS[i]), PROBS[otherIndex]]);
export const PROB_TOTAL = PROB_CELLS.slice(0, SHOWN_WORDS.length).reduce((s, p) => s + p, 0) + OTHERS * PROB_CELLS[SHOWN_WORDS.length];
