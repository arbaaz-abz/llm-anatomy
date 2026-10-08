// Rotary position embeddings, pair by pair (rope §6). Pure: no DOM, inputs never mutated.
// Pairs are adjacent dimensions (0, 1), (2, 3), … as in the RoPE paper; codebases that pair i with i + d/2
// do the same rotation on reordered dimensions.

const TURN = 2 * Math.PI;
const METHODS = Object.freeze(['none', 'pi', 'yarn-simple']);

const isCount = (n) => Number.isInteger(n) && n > 0;
const dot = (a, b) => a.reduce((sum, x, i) => sum + x * b[i], 0);

// A speed of 0 is a pair that does not rotate (partial RoPE, NoPE); only wavelengths and the stretch need speeds > 0.
function checkFreqs(fn, freqs, { allowZero = false } = {}) {
  const floor = allowZero ? (f) => f >= 0 : (f) => f > 0;
  if (!Array.isArray(freqs) || freqs.length === 0 || !freqs.every((f) => Number.isFinite(f) && floor(f))) {
    throw new RangeError(`${fn}: freqs must be a non-empty list of finite numbers ${allowZero ? '>= 0' : '> 0'}`);
  }
}

function checkVector(fn, vec, freqs) {
  if (!Array.isArray(vec) || vec.length !== 2 * freqs.length || !vec.every(Number.isFinite)) {
    throw new RangeError(`${fn}: vec must hold ${2 * freqs.length} finite numbers (one pair per frequency), got ${vec?.length}`);
  }
}

function checkPosition(fn, name, pos) {
  if (!Number.isFinite(pos)) throw new RangeError(`${fn}: ${name} must be a finite number, got ${pos}`);
}

// Speed of each pair: base^(−2i/d) for i = 0 … d/2 − 1.
export function ropeFrequencies(dHead, base) {
  if (!isCount(dHead) || dHead % 2 !== 0) throw new RangeError(`ropeFrequencies: dHead must be a positive even integer, got ${dHead}`);
  if (!(Number.isFinite(base) && base > 1)) throw new RangeError(`ropeFrequencies: base must be a finite number > 1, got ${base}`);
  return Array.from({ length: dHead / 2 }, (_, i) => base ** (-(2 * i) / dHead));
}

// Rotate each pair (x, y) by pos · freq: (x cos a − y sin a, x sin a + y cos a).
export function rotatePairs(vec, pos, freqs) {
  checkFreqs('rotatePairs', freqs, { allowZero: true });
  checkVector('rotatePairs', vec, freqs);
  checkPosition('rotatePairs', 'pos', pos);
  return freqs.flatMap((freq, i) => {
    const angle = pos * freq;
    const [x, y] = [vec[2 * i], vec[2 * i + 1]];
    return [x * Math.cos(angle) - y * Math.sin(angle), x * Math.sin(angle) + y * Math.cos(angle)];
  });
}

// The rotated dot product of q at qPos and k at kPos; pairs[i] is pair i's contribution.
export function ropeScore(q, k, { qPos, kPos, base, freqs = ropeFrequencies(q.length, base) }) {
  if (q.length !== k.length) throw new RangeError(`ropeScore: q has ${q.length} numbers but k has ${k.length}`);
  const qr = rotatePairs(q, qPos, freqs);
  const kr = rotatePairs(k, kPos, freqs);
  const pairs = freqs.map((_, i) => qr[2 * i] * kr[2 * i] + qr[2 * i + 1] * kr[2 * i + 1]);
  return { score: pairs.reduce((sum, x) => sum + x, 0), pairs };
}

// Score at each offset: the query sits at offset + 1, the key at 1 (positions are 1-based on screen).
export function scoreByOffset(q, k, { freqs, offsets }) {
  checkFreqs('scoreByOffset', freqs, { allowZero: true });
  if (!Array.isArray(offsets) || !offsets.every((o) => Number.isInteger(o) && o >= 0)) {
    throw new RangeError('scoreByOffset: offsets must be non-negative integers');
  }
  checkVector('scoreByOffset', q, freqs);
  checkVector('scoreByOffset', k, freqs);
  return offsets.map((offset) => dot(rotatePairs(q, offset + 1, freqs), rotatePairs(k, 1, freqs)));
}

// Tokens per full turn: 2π / freq.
export function wavelengths(freqs) {
  checkFreqs('wavelengths', freqs);
  return freqs.map((f) => TURN / f);
}

// Stretch the trained range by `factor`. 'pi': every freq ÷ factor. 'yarn-simple': a pair keeps its freq if its
// wavelength ≤ trainedLength (it already turned fully in training), else freq ÷ factor. A stated simplification of
// YaRN, which ramps smoothly between the two cases and adds an attention-temperature fix. 'none': a copy.
export function stretchFrequencies(freqs, { factor, method, trainedLength }) {
  checkFreqs('stretchFrequencies', freqs);
  if (!METHODS.includes(method)) throw new RangeError(`stretchFrequencies: method must be one of ${METHODS.join(', ')}, got "${method}"`);
  if (!(Number.isFinite(factor) && factor > 0)) throw new RangeError(`stretchFrequencies: factor must be a finite number > 0, got ${factor}`);
  if (!(Number.isFinite(trainedLength) && trainedLength > 0)) throw new RangeError(`stretchFrequencies: trainedLength must be > 0, got ${trainedLength}`);
  if (method === 'none') return [...freqs];
  if (method === 'pi') return freqs.map((f) => f / factor);
  return freqs.map((f) => (TURN / f <= trainedLength ? f : f / factor));
}

// Per pair: the largest angle seen in training (offsets 0 … trainedLength − 1, at the trained speed), the largest
// reached at `length` (at `freqs`), and whether every angle reached was seen: the pair turned fully in training,
// or it stays within one token step of the largest trained angle (the angles between trained positions).
export function angleCoverage(freqs, { trainedLength, length, trainedFreqs = freqs }) {
  checkFreqs('angleCoverage', freqs);
  if (!Array.isArray(trainedFreqs) || trainedFreqs.length !== freqs.length) throw new RangeError('angleCoverage: trainedFreqs must have one speed per pair');
  if (!isCount(trainedLength) || !isCount(length)) throw new RangeError(`angleCoverage: trainedLength and length must be positive integers, got ${trainedLength}, ${length}`);
  return freqs.map((freq, i) => {
    const seenMax = trainedFreqs[i] * (trainedLength - 1);
    const reachedMax = freq * (length - 1);
    const turnedFully = seenMax >= TURN;
    return { seenMax, reachedMax, allSeen: turnedFully || reachedMax <= seenMax + trainedFreqs[i] + 1e-9 };
  });
}
