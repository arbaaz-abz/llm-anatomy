// moe's hand-authored stand-in numbers (storyboard §4) and what follows from them. Pure, no DOM.
// The router scores, expert outputs and fine-grained scores are quarter-grid stand-ins; picks, gate weights, loads and the
// balancing run all come from math/moe.js.
import { deepFreeze } from '@math/core.js';
import { TOY } from '@math/attention.js';
import { ROUTER_TOY, routeTopK, gateWeights, expertLoads, simulateBalancing } from '@math/moe.js';
import { BATCH } from './format.js';

export const TOKENS = TOY.tokens; // The cat sat down, shared with `attention` and `decoder-anatomy`
export const SAT = 2; // row index of "sat", the token followed in frames 1–7 and 9
export const SCORE_MAX = 2; // the value scale for router scores and the bias
export const GATE_MAX = 1; // the value scale for gate weights
export const TOP_K = 2;

// The row the router reads: x′_sat after block 1's attention, the same stand-in as decoder-anatomy frame 6.
export const X_SAT = deepFreeze([0.25, 1.5, 0.25, 0.5, -0.5, 0.5, 0.25, 0.5]);
// What experts E3 and E6 return for it (stand-ins; hover-only on the stage).
export const E3_OUT = deepFreeze([0.5, -0.5, 1, 0, -0.5, 0.5, 0, 0.5]);
export const E6_OUT = deepFreeze([0, 0.5, -0.5, 1, 0.5, 0, -1, 0.25]);

export const SCORES_SAT = ROUTER_TOY[SAT];
export const PICKS = deepFreeze(ROUTER_TOY.map((row) => routeTopK(row, TOP_K)));
export const GATES = deepFreeze(ROUTER_TOY.map((row, t) => gateWeights(row, PICKS[t])));
export const LOADS_FOUR = deepFreeze(expertLoads(ROUTER_TOY, TOP_K));
export const TOP1_SAT = deepFreeze(routeTopK(SCORES_SAT, 1)); // frame 7: the router now picks one routed expert

// Frame 9: the bias row, the sum the router chooses with, and what changes.
export const BIAS_SAT = deepFreeze([0, 0, 0, 0, 0, -0.75, 0.5, 0]);
export const CHOOSE_SAT = deepFreeze(SCORES_SAT.map((s, e) => s + BIAS_SAT[e]));
export const PICKS_BIASED = deepFreeze(routeTopK(SCORES_SAT, TOP_K, BIAS_SAT));
export const GATES_BIASED = deepFreeze(gateWeights(SCORES_SAT, PICKS_BIASED)); // from the raw scores, never the biased ones

// Frame 6: the same expert split in two (16 experts of hidden 4); a stand-in score per half (a, b) and the four it picks.
export const SAT_FINE = deepFreeze([0, 0.25, -0.5, -0.25, 1.75, 2.0, 0.25, 0.5, -1, -1.25, 1.5, 0.5, 0.5, 1.25, 0, 0.25]);
export const FINE_PICKS = deepFreeze(routeTopK(SAT_FINE, TOP_K * 2));

// Frames 8 and 10: the seeded batch with gamma 0.1, steps 0–6. Frame 8 shows step 0, frame 10 runs 0 → 6.
export const FRAME_GAMMA = 0.1;
export const LAST_STEP = 6;
export const RUN = deepFreeze(simulateBalancing({ ...BATCH, gamma: FRAME_GAMMA, steps: LAST_STEP + 1 }));
export const MAX_LOAD = 96; // held fixed in frames 8 and 10 so the flattening reads as motion
export const FAIR_SHARE = (BATCH.tokens * TOP_K) / BATCH.experts; // 128 × 2 ÷ 8 = 32
