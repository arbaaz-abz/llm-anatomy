// pretraining's hand-authored stand-in numbers (storyboard §4–§6) and what follows from them. Pure, no DOM.
// Position 4's probabilities are decoder-anatomy frame 8's exact softmax (recomputed here from the same logits: a track
// never imports another track's concepts); the other six probabilities are hand-picked stand-ins.
import { deepFreeze, softmax } from '@math/core.js';
import { tokenLoss } from '@math/lm.js';

export const TOKENS = Object.freeze(['The', 'cat', 'sat', 'down', 'on', 'the', 'mat', '.']);

// decoder-anatomy's 16-word vocabulary and its scores after "down": on 2.0 · "." 1.5 · and 0.5 · the 0.0 · 12 others −1.0.
export const LOGIT_WORDS = Object.freeze(['The', 'cat', 'sat', 'down', 'on', '.', 'and', 'the', 'mat', 'a', 'dog', 'ran', 'up', 'big', 'was', 'then']);
const SCORE_OF = Object.freeze({ on: 2, '.': 1.5, and: 0.5, the: 0 });
const OTHER_SCORE = -1;
export const LOGITS = deepFreeze(LOGIT_WORDS.map((w) => SCORE_OF[w] ?? OTHER_SCORE));
const PROBS = deepFreeze(softmax(LOGITS));
export const SHOWN_WORDS = Object.freeze(['on', '.', 'and', 'the']);
export const OTHERS = LOGIT_WORDS.length - SHOWN_WORDS.length; // 12, collapsed into one cell (README lesson 18)
const otherIndex = LOGIT_WORDS.findIndex((w) => !SHOWN_WORDS.includes(w));
// Five cells: the four named words, then one "12 others" cell holding the value of each of them.
export const PROB_CELLS = deepFreeze([...SHOWN_WORDS.map((w) => PROBS[LOGIT_WORDS.indexOf(w)]), PROBS[otherIndex]]);
export const PROB_TOTAL = PROB_CELLS.slice(0, SHOWN_WORDS.length).reduce((s, p) => s + p, 0) + OTHERS * PROB_CELLS[SHOWN_WORDS.length];
export const P_ON = PROBS[LOGIT_WORDS.indexOf('on')]; // 0.390253…, printed 0.390 (3 d.p.) or 0.3903 (4 d.p.)

// Positions 1–7 predict tokens 2–8: cat 0.10 · sat 0.25 · down 0.30 · on 0.390 · the 0.60 · mat 0.45 · . 0.80.
export const STAND_INS = deepFreeze([0.10, 0.25, 0.30, P_ON, 0.60, 0.45, 0.80]);
export const FOLLOWED = 4; // the followed prediction: position 4 → "on" (frames 2–5)

export const UNIFORM_VOCAB = 16; // the toy vocabulary of decoder-anatomy
export const REFERENCE_PS = Object.freeze([0.99, 0.01]); // frame 3's two reference marks
// One fixed value scale for every loss on the page (stage and toy): ln 100, the loss of a confident miss at p = 0.01.
export const LOSS_MAX_ABS = tokenLoss(0.01);
export const PROB_MAX_ABS = 0.4; // the value scale for probabilities, as on decoder-anatomy
