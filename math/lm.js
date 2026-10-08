// Language-model loss (pretraining §11; reused by sft, rlhf-dpo and distillation).
// Pure: no DOM, inputs never mutated. Every bad argument throws RangeError('<fn>: <arg> must be …').
// FROZEN after Plan 3 S3: tokenLoss, meanLoss, perplexity, uniformLoss and klDivergence keep these signatures.

function requireArray(fn, name, value) {
  if (!Array.isArray(value) || value.length === 0) throw new RangeError(`${fn}: ${name} must be a non-empty array`);
}

function requireProbability(fn, name, value) {
  if (typeof value !== 'number' || !(value >= 0 && value <= 1)) throw new RangeError(`${fn}: ${name} must be a probability in [0, 1], got ${value}`);
}

// Loss for one position: minus the natural log of the probability given to the true token.
//   p(on) = 0.3903 (decoder-anatomy's logits) → 0.9410 · 0.01 → 4.6052 · 1 → 0 · 0.0625 → 2.7726
export function tokenLoss(p) {
  if (typeof p !== 'number' || !(p > 0 && p <= 1)) throw new RangeError(`tokenLoss: p must be a number with 0 < p ≤ 1, got ${p}`);
  return -Math.log(p) || 0; // `|| 0` turns −ln 1 = −0 into 0
}

function checkMask(mask, length) {
  if (mask === null) return;
  if (!Array.isArray(mask)) throw new RangeError('meanLoss: mask must be null or an array of booleans');
  if (mask.length !== length) throw new RangeError(`meanLoss: mask must have one entry per position (${length}), got ${mask.length}`);
  mask.forEach((m, i) => {
    if (typeof m !== 'boolean') throw new RangeError(`meanLoss: mask[${i}] must be true or false`);
  });
  if (!mask.includes(true)) throw new RangeError('meanLoss: mask must keep at least one position');
}

// Mean loss over positions; an optional boolean mask keeps only positions where mask[i] is true
// (sft builds it with lossMask). Every probability is checked, kept or not.
//   [0.10, 0.25, 0.30, 0.3903, 0.60, 0.45, 0.80] → 1.0523 · [0.5, 0.25, 0.8], [false, true, true] → 0.8047
export function meanLoss(probs, mask = null) {
  requireArray('meanLoss', 'probs', probs);
  checkMask(mask, probs.length);
  const losses = probs.map(tokenLoss);
  const kept = mask === null ? losses : losses.filter((_, i) => mask[i]);
  return kept.reduce((sum, loss) => sum + loss, 0) / kept.length;
}

// e^loss: the effective number of equally likely choices.   1.0523 → 2.8643 · 2.7726 → 16
export function perplexity(loss) {
  if (typeof loss !== 'number' || !Number.isFinite(loss) || loss < 0) throw new RangeError(`perplexity: loss must be a finite number ≥ 0, got ${loss}`);
  return Math.exp(loss);
}

// ln |V|, the loss of a uniform guess over the vocabulary.   16 → 2.7726 · 163840 → 12.0066 · 201088 → 12.2115
export function uniformLoss(vocabSize) {
  if (!Number.isInteger(vocabSize) || vocabSize < 1) throw new RangeError(`uniformLoss: vocabSize must be an integer ≥ 1, got ${vocabSize}`);
  return Math.log(vocabSize);
}

// KL divergence D(p ‖ q) = Σ p ln(p/q); terms with p = 0 contribute 0. Not symmetric:
// distillation's forward KL is klDivergence(teacher, student), its reverse KL klDivergence(student, teacher).
//   ([0.70,0.15,0.10,0.05], [0.40,0.30,0.20,0.10]) → 0.1838 · ([0.01,0.01,0.01,0.97], same) → 2.1031
//   ([0.90,0.05,0.03,0.02], [0.40,0.30,0.20,0.10]) → 0.5511 · the reverse → 0.7535 · (p, p) → 0
export function klDivergence(p, q) {
  requireArray('klDivergence', 'p', p);
  requireArray('klDivergence', 'q', q);
  if (p.length !== q.length) throw new RangeError(`klDivergence: p and q must have the same length (${p.length} vs ${q.length})`);
  p.forEach((v, i) => requireProbability('klDivergence', `p[${i}]`, v));
  q.forEach((v, i) => requireProbability('klDivergence', `q[${i}]`, v));
  return p.reduce((sum, pi, i) => {
    if (pi === 0) return sum;
    if (q[i] === 0) throw new RangeError(`klDivergence: q[${i}] must be > 0 where p[${i}] > 0`);
    return sum + pi * Math.log(pi / q[i]);
  }, 0);
}
