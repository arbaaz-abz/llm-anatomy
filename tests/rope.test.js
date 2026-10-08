import { test } from 'node:test';
import assert from 'node:assert/strict';
import { TOY } from '../math/attention.js';
import { mulberry32 } from '../math/core.js';
import {
  ropeFrequencies, rotatePairs, ropeScore, scoreByOffset, wavelengths, stretchFrequencies, angleCoverage,
} from '../math/rope.js';

const { Q, K } = TOY.heads.A;
const Q_SAT = Q[2];
const [K_THE, K_CAT, K_SAT] = K;
const OFFSETS = [0, 1, 2, 3, 4, 5, 6, 7];

const near = (actual, expected, label, tol = 5e-4) => {
  assert.equal(actual.length, expected.length, `${label}: length`);
  actual.forEach((v, i) => assert.ok(Math.abs(v - expected[i]) < tol, `${label}[${i}]: ${v} vs ${expected[i]}`));
};

// ---- ropeFrequencies ----

test('ropeFrequencies: speed of pair i is base^(-2i/d)', () => {
  assert.deepEqual(ropeFrequencies(4, 100), [1, 0.1]);
  assert.deepEqual(ropeFrequencies(4, 10_000), [1, 0.01]);
});

test('ropeFrequencies: the slowest pair of a 128-wide head', () => {
  const f = ropeFrequencies(128, 10_000);
  assert.equal(f.length, 64);
  assert.equal(f[0], 1);
  assert.ok(Math.abs(f.at(-1) - 1.155e-4) < 5e-7);
});

test('ropeFrequencies throws RangeError on an odd or non-positive d_head and on base <= 1', () => {
  assert.throws(() => ropeFrequencies(3, 100), RangeError);
  assert.throws(() => ropeFrequencies(0, 100), RangeError);
  assert.throws(() => ropeFrequencies(4.5, 100), RangeError);
  assert.throws(() => ropeFrequencies(4, 1), RangeError);
  assert.throws(() => ropeFrequencies(4, Number.NaN), RangeError);
});

// ---- rotatePairs ----

test('rotatePairs: the storyboard\'s rotated vectors (frames 3, 4, 5)', () => {
  const f = ropeFrequencies(4, 100);
  near(rotatePairs(Q_SAT, 3, f), [-0.282, -1.980, 0.478, 0.148], 'q at 3');
  near(rotatePairs(K_CAT, 2, f), [-1.364, -0.624, 0.099, -0.490], 'k at 2');
  near(rotatePairs(Q_SAT, 13, f), [-0.840, 1.815, 0.134, 0.482], 'q at 13');
  near(rotatePairs(K_CAT, 12, f), [0.805, 1.266, 0.466, -0.181], 'k at 12');
});

test('rotatePairs: position 0 returns the vector exactly', () => {
  assert.deepEqual(rotatePairs(Q_SAT, 0, [1, 0.1]), Q_SAT);
});

test('rotatePairs: each pair keeps its length (|(-0.282, -1.980)| = 2.000)', () => {
  const rotated = rotatePairs(Q_SAT, 3, [1, 0.1]);
  assert.ok(Math.abs(Math.hypot(rotated[0], rotated[1]) - 2) < 1e-12);
  assert.ok(Math.abs(Math.hypot(rotated[2], rotated[3]) - 0.5) < 1e-12);
});

test('rotatePairs: a vector of the wrong length, a bad position or a bad speed throws RangeError', () => {
  assert.throws(() => rotatePairs([1, 2, 3], 1, [1]), RangeError);
  assert.throws(() => rotatePairs([1, 2, 3, 4], 1, [1]), RangeError);
  assert.throws(() => rotatePairs([1, 2], Number.NaN, [1]), RangeError);
  assert.throws(() => rotatePairs([1, 2], 1, [Number.POSITIVE_INFINITY]), RangeError);
});

test('rotatePairs does not mutate its inputs', () => {
  const v = [0, 2, 0.5, 0];
  const f = [1, 0.1];
  rotatePairs(v, 3, f);
  assert.deepEqual(v, [0, 2, 0.5, 0]);
  assert.deepEqual(f, [1, 0.1]);
});

// ---- ropeScore ----

test('ropeScore: "sat" at 3 against "cat" at 2, base 100: 1.621 + (-0.025) = 1.596', () => {
  const r = ropeScore(Q_SAT, K_CAT, { qPos: 3, kPos: 2, base: 100 });
  near([r.score], [1.596], 'score');
  near(r.pairs, [1.621, -0.025], 'pairs');
});

test('ropeScore: the same offset at 13/12 and 103/102 gives the same score', () => {
  near([ropeScore(Q_SAT, K_CAT, { qPos: 13, kPos: 12, base: 100 }).score], [1.596], '13/12');
  near([ropeScore(Q_SAT, K_CAT, { qPos: 103, kPos: 102, base: 100 }).score], [1.596], '103/102');
});

test('ropeScore: base 10,000 slows pair 2 and nearly freezes it', () => {
  const r = ropeScore(Q_SAT, K_CAT, { qPos: 3, kPos: 2, base: 10_000 });
  near([r.score], [1.618], 'score');
  near(r.pairs, [1.621, -0.002], 'pairs');
});

test('ropeScore: offset 0 leaves the dot product unrotated (to rounding: both vectors turn by the same angle)', () => {
  near([ropeScore(Q_SAT, K_SAT, { qPos: 3, kPos: 3, base: 100 }).score], [0.5], 'sat . sat', 1e-12);
  near([ropeScore(Q_SAT, K_CAT, { qPos: 7, kPos: 7, base: 100 }).score], [3], 'sat . cat', 1e-12);
});

test('ropeScore: the score is the sum of the pair contributions', () => {
  const r = ropeScore(Q_SAT, K_CAT, { qPos: 5, kPos: 1, base: 100 });
  assert.ok(Math.abs(r.pairs.reduce((s, x) => s + x, 0) - r.score) < 1e-12);
});

test('ropeScore: only the offset matters (20 random shifts, |delta| < 1e-9; frame 5 and README lesson 16)', () => {
  const rand = mulberry32(7);
  for (let t = 0; t < 20; t += 1) {
    const m = 1 + Math.floor(rand() * 60);
    const n = 1 + Math.floor(rand() * 60);
    const shift = Math.floor(rand() * 500);
    const base = [100, 10_000][t % 2];
    const a = ropeScore(Q_SAT, K_CAT, { qPos: m, kPos: n, base }).score;
    const b = ropeScore(Q_SAT, K_CAT, { qPos: m + shift, kPos: n + shift, base }).score;
    assert.ok(Math.abs(a - b) < 1e-9, `shift ${shift} at (${m}, ${n}): ${a} vs ${b}`);
  }
});

test('ropeScore: explicit freqs override base; a q and k of different length throw RangeError', () => {
  const viaBase = ropeScore(Q_SAT, K_CAT, { qPos: 3, kPos: 2, base: 100 });
  const viaFreqs = ropeScore(Q_SAT, K_CAT, { qPos: 3, kPos: 2, freqs: [1, 0.1] });
  assert.deepEqual(viaFreqs, viaBase);
  assert.throws(() => ropeScore([1, 2], [1, 2, 3, 4], { qPos: 1, kPos: 1, base: 100 }), RangeError);
});

test('ropeScore: the frame 5 row at +10: [-1.353, 1.596, 0.5] (base 100) and [-1.397, 1.618, 0.5] (base 10,000)', () => {
  const row = (base) => [
    ropeScore(Q_SAT, K_THE, { qPos: 13, kPos: 11, base }).score,
    ropeScore(Q_SAT, K_CAT, { qPos: 13, kPos: 12, base }).score,
    ropeScore(Q_SAT, K_SAT, { qPos: 13, kPos: 13, base }).score,
  ];
  near(row(100), [-1.353, 1.596, 0.5], 'base 100');
  near(row(10_000), [-1.397, 1.618, 0.5], 'base 10,000');
});

// ---- scoreByOffset ----

test('scoreByOffset: base 100 and base 10,000 rows (frames 6 and 7)', () => {
  near(scoreByOffset(Q_SAT, K_CAT, { freqs: [1, 0.1], offsets: OFFSETS }), [3, 1.596, -1.298, -3.044, -2.058, 0.731, 2.739, 2.101], 'base 100');
  near(scoreByOffset(Q_SAT, K_CAT, { freqs: [1, 0.01], offsets: OFFSETS }), [3, 1.618, -1.253, -2.977, -1.971, 0.838, 2.866, 2.244], 'base 10,000');
});

test('scoreByOffset: position interpolation (divide by 4) and the YaRN-style row (frames 8 and 9)', () => {
  near(scoreByOffset(Q_SAT, K_CAT, { freqs: [0.25, 0.025], offsets: OFFSETS }), [3, 2.9, 2.62, 2.176, 1.596, 0.915, 0.175, -0.578], 'PI');
  near(scoreByOffset(Q_SAT, K_CAT, { freqs: [1, 0.025], offsets: OFFSETS }), [3, 1.615, -1.261, -2.989, -1.986, 0.82, 2.843, 2.218], 'YaRN-style');
});

test('scoreByOffset: offset 0 is the plain dot product q . k', () => {
  const plain = Q_SAT.reduce((s, x, i) => s + x * K_CAT[i], 0);
  near(scoreByOffset(Q_SAT, K_CAT, { freqs: [1, 0.1], offsets: [0] }), [plain], 'offset 0', 1e-12);
});

test('scoreByOffset: an offset that is not a non-negative integer throws RangeError', () => {
  assert.throws(() => scoreByOffset(Q_SAT, K_CAT, { freqs: [1, 0.1], offsets: [-1] }), RangeError);
  assert.throws(() => scoreByOffset(Q_SAT, K_CAT, { freqs: [1, 0.1], offsets: [0.5] }), RangeError);
});

// ---- wavelengths ----

test('wavelengths: tokens per full turn is 2 pi / speed', () => {
  near(wavelengths([1, 0.1]), [6.283, 62.832], 'toy', 5e-4);
  assert.throws(() => wavelengths([0]), RangeError);
  assert.throws(() => wavelengths([-1]), RangeError);
});

test('wavelengths: the slowest pair of a 128-wide head at the §8 bases', () => {
  const slowest = (base) => Math.round(wavelengths(ropeFrequencies(128, base)).at(-1));
  assert.equal(slowest(10_000), 54_410);
  assert.equal(slowest(150_000), 782_338);
  assert.equal(slowest(10_000_000), 48_843_285);
});

// ---- stretchFrequencies ----

test('stretchFrequencies: PI slows every pair; YaRN-style keeps pairs that turned fully in training', () => {
  assert.deepEqual(stretchFrequencies([1, 0.1], { factor: 4, method: 'pi', trainedLength: 16 }), [0.25, 0.025]);
  assert.deepEqual(stretchFrequencies([1, 0.1], { factor: 4, method: 'yarn-simple', trainedLength: 16 }), [1, 0.025]);
});

test('stretchFrequencies: factor 1 or method "none" is the identity (and a copy)', () => {
  const f = [1, 0.1];
  assert.deepEqual(stretchFrequencies(f, { factor: 1, method: 'none', trainedLength: 16 }), [1, 0.1]);
  assert.deepEqual(stretchFrequencies(f, { factor: 1, method: 'pi', trainedLength: 16 }), [1, 0.1]);
  assert.deepEqual(stretchFrequencies(f, { factor: 1, method: 'yarn-simple', trainedLength: 16 }), [1, 0.1]);
  assert.deepEqual(stretchFrequencies(f, { factor: 4, method: 'none', trainedLength: 16 }), [1, 0.1]);
  assert.notEqual(stretchFrequencies(f, { factor: 1, method: 'none', trainedLength: 16 }), f);
});

test('stretchFrequencies: bad method, factor or trained length throws RangeError; the input is not mutated', () => {
  const f = [1, 0.1];
  assert.throws(() => stretchFrequencies(f, { factor: 4, method: 'ntk', trainedLength: 16 }), RangeError);
  assert.throws(() => stretchFrequencies(f, { factor: 0, method: 'pi', trainedLength: 16 }), RangeError);
  assert.throws(() => stretchFrequencies(f, { factor: 4, method: 'yarn-simple', trainedLength: 0 }), RangeError);
  stretchFrequencies(f, { factor: 4, method: 'pi', trainedLength: 16 });
  assert.deepEqual(f, [1, 0.1]);
});

// ---- angleCoverage ----

test('angleCoverage: trained on 16 tokens, read to 64: pair 1 turned fully, pair 2 reaches 6.3 rad it never saw', () => {
  const [p1, p2] = angleCoverage([1, 0.1], { trainedLength: 16, length: 64 });
  near([p1.seenMax, p1.reachedMax], [15, 63], 'pair 1');
  near([p2.seenMax, p2.reachedMax], [1.5, 6.3], 'pair 2', 1e-9);
  assert.deepEqual([p1.allSeen, p2.allSeen], [true, false]);
});

test('angleCoverage: after position interpolation pair 2 reaches 1.575, within one token step of the 1.5 it saw', () => {
  const [, p2] = angleCoverage([0.25, 0.025], { trainedLength: 16, length: 64, trainedFreqs: [1, 0.1] });
  near([p2.seenMax, p2.reachedMax], [1.5, 1.575], 'pair 2', 1e-9);
  assert.equal(p2.allSeen, true);
});

test('angleCoverage: reading exactly the trained length sees everything; the YaRN-style freqs stay seen', () => {
  assert.deepEqual(angleCoverage([1, 0.1], { trainedLength: 16, length: 16 }).map((p) => p.allSeen), [true, true]);
  assert.deepEqual(angleCoverage([1, 0.025], { trainedLength: 16, length: 64, trainedFreqs: [1, 0.1] }).map((p) => p.allSeen), [true, true]);
  assert.deepEqual(angleCoverage([1, 0.1], { trainedLength: 16, length: 32 }).map((p) => p.allSeen), [true, false]);
});

test('angleCoverage: bad lengths throw RangeError', () => {
  assert.throws(() => angleCoverage([1, 0.1], { trainedLength: 0, length: 4 }), RangeError);
  assert.throws(() => angleCoverage([1, 0.1], { trainedLength: 16, length: 0 }), RangeError);
  assert.throws(() => angleCoverage([1, 0.1], { trainedLength: 16, length: 8, trainedFreqs: [1] }), RangeError);
});
