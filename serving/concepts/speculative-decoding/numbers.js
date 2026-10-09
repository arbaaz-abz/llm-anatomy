// speculative-decoding stand-ins and fixed inputs (storyboard §4–§6). The target is prefill-decode's running example with
// 1,024 tokens of context per user; the guesses, the two probability rows and the drafter cost are hand-picked stand-ins.
// Nothing here is a computed result: step times, speedups and user counts come from math/ functions at the call site.
import { deepFreeze } from '@math/core.js';
import { RUNNING_EXAMPLE } from '@math/serving.js';
import { VOCAB } from '@math/sampling.js';

export const CONTEXT = 1024; // tokens each user holds (frames 1–9 and the toy)
export const MODEL = deepFreeze({ ...RUNNING_EXAMPLE, context: CONTEXT });

export const ALPHA = 0.7; // the animation's acceptance rate
export const K = 3; // guesses per round in frames 1–6 and 9
export const C = 0.05; // drafter cost per guess, as a fraction of a target step
export const HEAD_USERS = 128; // frame 9's step bars

// "The cat sat" is serving-overview's prompt; the drafter's three guesses, the verdicts and the target's own pick (frame 5).
const word = (i) => VOCAB[i];
export const PROMPT = deepFreeze([word(0), word(1), word(2)]);
export const GUESSES = deepFreeze([word(3), word(4), word(9)]); // down on a
export const VERDICTS = deepFreeze([true, true, false]);
export const PICK = word(7); // the

// Frame 6 and the toy's position panel: four candidate words and the two models' probabilities over them.
export const ZOOM_WORDS = deepFreeze([word(3), word(4), word(12), word(13)]); // down on up big
export const P = deepFreeze([0.6, 0.25, 0.1, 0.05]); // the target
export const Q = deepFreeze([0.7, 0.2, 0.05, 0.05]); // the drafter
export const ZOOM_GUESS = 0; // the guess followed in frame 6 (down)

// Frames 7–9 sweeps.
export const ALPHAS = deepFreeze([0.5, 0.7, 0.85, 0.9]);
export const K_RANGE = deepFreeze([1, 2, 3, 4, 5, 6, 7, 8]);
export const USER_STOPS = deepFreeze([1, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 211]);

// One seconds scale for every step bar on the stage (README lesson 19): 40 ms holds the 36.2 ms verify pass of frame 9.
export const STEP_SCALE_S = 0.04;
