// The home page's story on one clock: three acts, one per track, each the cover of its card.
// Act 1 a token goes through the model, act 2 the loss falls as the weights move, act 3 three requests share
// one GPU. Everything here is pure (time in, what to draw out); covers.js turns a scene into SVG.
import { easeInOut } from '../shared/ui/stepper.js';

export const LEAD_MS = 400; // a beat before the first act moves
export const ACT_MS = 2600;
export const ACTS = 3;
export const STORY_MS = LEAD_MS + ACTS * ACT_MS;

export const clamp01 = (v) => Math.min(Math.max(Number(v) || 0, 0), 1);
// Eased 0..1 inside the sub-window [from, to] of an act's progress p.
export const within = (p, from, to) => easeInOut(clamp01((p - from) / (to - from)));

// Progress (0..1) of every act at `ms` since the story started; `act` is the one still playing (-1 once all are done).
export function storyFrame(ms) {
  const t = ms === Infinity ? STORY_MS : Number.isFinite(ms) ? Math.max(ms, 0) : 0;
  const progress = Array.from({ length: ACTS }, (_, i) => clamp01((t - LEAD_MS - i * ACT_MS) / ACT_MS));
  const act = progress.findIndex((p) => p < 1);
  return { progress, act, done: act === -1 };
}

export const FINAL_FRAME = Object.freeze(storyFrame(STORY_MS));

// ---- act 1: a token through the model (cover viewBox 320 × 150) ----
export const MODEL = Object.freeze({
  input: Object.freeze(['The', 'cat', 'sat']),
  output: 'down',
  spine: Object.freeze({ x: 160, from: 120, to: 36 }),
  blockH: 22,
  blocks: Object.freeze([Object.freeze({ id: 'attention', label: 'attention', y: 84 }), Object.freeze({ id: 'mlp', label: 'MLP', y: 52 })]),
});
const TOKEN_IN_AT = 0.05; // the i-th input token appears at TOKEN_IN_AT + i × TOKEN_IN_STEP
const TOKEN_IN_STEP = 0.06;
const TRAVEL = Object.freeze({ from: 0.25, to: 0.85 });

export function modelScene(p) {
  const tokens = MODEL.input.map((text, i) => ({ text, state: p >= TOKEN_IN_AT + i * TOKEN_IN_STEP ? 'idle' : 'dim' }));
  const travelling = p >= TRAVEL.from && p < TRAVEL.to;
  const dotY = travelling ? MODEL.spine.from + (MODEL.spine.to - MODEL.spine.from) * within(p, TRAVEL.from, TRAVEL.to) : null;
  const active = MODEL.blocks.find((b) => dotY != null && dotY >= b.y && dotY <= b.y + MODEL.blockH)?.id ?? null;
  return { tokens, dotY, active, output: p >= TRAVEL.to ? 'active' : 'draft' };
}

// ---- act 2: the loss falls as the weights move ----
// Toy weights, not real ones: a noisy 4 × 4 that settles into a near-diagonal as training runs.
export const W_START = Object.freeze([[0.9, -0.6, 0.2, -0.8], [-0.3, 0.7, -0.9, 0.1], [0.5, -0.2, 0.8, -0.5], [-0.7, 0.4, -0.1, 0.6]].map(Object.freeze));
export const W_END = Object.freeze([[0.9, 0.1, 0, -0.1], [0.1, 0.8, 0.1, 0], [0, 0.1, 0.9, 0.1], [-0.1, 0, 0.1, 0.8]].map(Object.freeze));
const CURVE_POINTS = 25;
const lossAt = (x) => 0.12 + 0.88 * Math.exp(-3.5 * x);
export const LOSS_CURVE = Object.freeze(Array.from({ length: CURVE_POINTS }, (_, i) => Object.freeze([i / (CURVE_POINTS - 1), Number(lossAt(i / (CURVE_POINTS - 1)).toFixed(3))])));
const TRAIN = Object.freeze({ from: 0.1, to: 0.9, pulses: 3 });

export function trainScene(p) {
  const k = within(p, TRAIN.from, TRAIN.to);
  const weights = k >= 1 ? W_END : W_START.map((row, i) => row.map((v, j) => v + (W_END[i][j] - v) * k));
  const shown = Math.round(k * (CURVE_POINTS - 1)) + 1;
  const running = p > TRAIN.from && p < TRAIN.to;
  const gradient = running ? (((p - TRAIN.from) / (TRAIN.to - TRAIN.from)) * TRAIN.pulses) % 1 : null;
  return { weights, curve: LOSS_CURVE.slice(0, shown), gradient };
}

// ---- act 3: three requests share one GPU ----
export const SERVE = Object.freeze({
  owners: Object.freeze(['A', 'B', 'C']),
  enter: -34, // a chip starts off the left edge…
  dock: 80, // …and stops against the GPU
  laneY: 30, laneStep: 28,
  streamX: 216, streamStep: 32, streamMax: 3,
});
const ARRIVE = Object.freeze({ at: 0.02, stagger: 0.12, span: 0.28 });
const STREAM = Object.freeze({ at: 0.45, stagger: 0.08, step: 0.14 });
const MEM = Object.freeze({ from: 0.3, to: 0.95 });

export function serveScene(p) {
  const requests = SERVE.owners.map((owner, i) => {
    const start = ARRIVE.at + i * ARRIVE.stagger;
    return { owner, x: SERVE.enter + (SERVE.dock - SERVE.enter) * within(p, start, start + ARRIVE.span) };
  });
  const streams = SERVE.owners.map((_, i) => {
    const start = STREAM.at + i * STREAM.stagger;
    const count = p < start ? 0 : Math.floor((p - start) / STREAM.step) + 1;
    return Math.min(count, SERVE.streamMax);
  });
  return { requests, streams, memFill: within(p, MEM.from, MEM.to) };
}
