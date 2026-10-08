// One attention head, step by step (attention §7). Pure: no DOM, inputs never mutated.
import { matmul, transpose, softmax, causalMask, deepFreeze } from './core.js';

// Hand-picked on a quarter grid so every dot product is a one-line sum (attention §4, ruling 1).
export const TOY = deepFreeze({
  tokens: ['The', 'cat', 'sat', 'down'],
  dModel: 8,
  dHead: 4,
  heads: {
    A: {
      Q: [[1, 0, -0.5, 0], [0.5, 0.5, 0, -1], [0, 2, 0.5, 0], [0.5, 2, 0, 0]],
      K: [[1, -0.5, 0, 0.5], [0, 1.5, 0, -0.5], [-0.5, 0, 1, 0.5], [0.5, -0.5, 0.5, 1]],
      V: [[1, 0, -1, 0], [0, 2, 0, 1], [-1, 0, 1, 0.5], [0.5, -1, 0, 1]],
    },
    B: {
      Q: [[1, 0, 0, 0.5], [2, 0, 0.5, 0], [0, 2, 0, 0.5], [0.5, 0, 2, 0]],
      K: [[1.5, 0, -0.5, 0], [0, 1.5, 0, -0.5], [-0.5, 0, 1.5, 0], [0, -0.5, 0, 1.5]],
      V: [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
    },
  },
});

// S = Q · Kᵀ: [n × d] · [n × d] → [n × n]. matmul throws RangeError on a shape mismatch.
export const scores = (Q, K) => matmul(Q, transpose(K));

export function scaleScores(S, divisor) {
  if (!(divisor > 0) || !Number.isFinite(divisor)) throw new RangeError(`scaleScores: divisor must be > 0, got ${divisor}`);
  return S.map((row) => row.map((v) => v / divisor));
}

// true = visible (core.causalMask's convention); hidden cells become −Infinity.
export function applyMask(S, mask) {
  const sameShape = mask.length === S.length && S.every((row, i) => mask[i]?.length === row.length);
  if (!sameShape) throw new RangeError(`applyMask: mask shape [${mask.length} × ${mask[0]?.length}] differs from scores [${S.length} × ${S[0]?.length}]`);
  return S.map((row, i) => row.map((v, j) => (mask[i][j] ? v : -Infinity)));
}

// Row-wise softmax; a fully masked row throws (core.softmax: "every logit is masked").
export const attentionWeights = (S) => S.map((row) => softmax(row));

// O = A · V: [n × n] · [n × d] → [n × d].
export const weightedSum = (A, V) => matmul(A, V);

const allVisible = (n) => Array.from({ length: n }, () => Array.from({ length: n }, () => true));

// The whole pipeline for one head; every intermediate is returned for the animation and the toy.
export function attentionHead(Q, K, V, { causal = true, divisor = Math.sqrt(Q[0].length) } = {}) {
  const S = scores(Q, K);
  const scaled = scaleScores(S, divisor);
  const mask = causal ? causalMask(Q.length) : allVisible(Q.length);
  const masked = applyMask(scaled, mask);
  const weights = attentionWeights(masked);
  // The un-shifted exp row the storyboard prints ("exp [0.607, 4.482, 1.284, 0] · sum 6.372"); masked cells give exp(−∞) = 0.
  const exps = masked.map((row) => row.map(Math.exp));
  const expSums = exps.map((row) => row.reduce((s, x) => s + x, 0));
  return { scores: S, scaled, mask, masked, exps, expSums, weights, output: weightedSum(weights, V) };
}

// Several heads over the same tokens; concat row i = every head's output row i, left to right.
export function multiHead(heads, opts = {}) {
  if (!heads.length) throw new RangeError('multiHead: at least one head is required');
  const results = heads.map(({ Q, K, V }) => attentionHead(Q, K, V, opts));
  const concat = results[0].output.map((_, i) => results.flatMap((r) => r.output[i]));
  return { heads: results, concat };
}

// Q = X·W_Q, K = X·W_K, V = X·W_V (the deferred "from embeddings" preset and the gallery demo).
export const projectQKV = (X, { Wq, Wk, Wv }) => ({ Q: matmul(X, Wq), K: matmul(X, Wk), V: matmul(X, Wv) });
