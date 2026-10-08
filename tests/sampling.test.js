import { test } from 'node:test';
import assert from 'node:assert/strict';
import { VOCAB, LOGITS, NAMED, rankTokens, applyTemperature, topK, topP, samplingDistribution, sampleIndex, drawSamples, collapseOthers } from '../math/sampling.js';
import { softmax } from '../math/core.js';

const near = (actual, expected, tol = 5e-4) => {
  if (Array.isArray(expected)) { assert.equal(actual.length, expected.length); expected.forEach((e, i) => near(actual[i], e, tol)); return; }
  assert.ok(Math.abs(actual - expected) < tol, `${actual} ≉ ${expected}`);
};
const sum = (list) => list.reduce((a, b) => a + b, 0);
const P1 = applyTemperature(LOGITS, 1);
const ON = 4;
const [DOT, AND, THE] = [5, 6, 7];

// Counts of the four named words and "everything else" among `n` drawn indices.
const histogram = (draws) => {
  const named = NAMED.map((i) => draws.filter((d) => d === i).length);
  return [...named, draws.length - sum(named)];
};

test('the course vocabulary is the decoder-anatomy one: 16 words, frozen, the same scores', () => {
  assert.deepEqual([...VOCAB], ['The', 'cat', 'sat', 'down', 'on', '.', 'and', 'the', 'mat', 'a', 'dog', 'ran', 'up', 'big', 'was', 'then']);
  assert.equal(LOGITS.length, VOCAB.length);
  assert.deepEqual(NAMED.map((i) => VOCAB[i]), ['on', '.', 'and', 'the']);
  assert.deepEqual(NAMED.map((i) => LOGITS[i]), [2, 1.5, 0.5, 0]);
  assert.ok(LOGITS.filter((_, i) => !NAMED.includes(i)).every((z) => z === -1));
  assert.ok(Object.isFrozen(VOCAB) && Object.isFrozen(LOGITS) && Object.isFrozen(NAMED));
  assert.throws(() => { VOCAB.push('x'); }, TypeError);
});

test('applyTemperature: the storyboard\'s "on" probabilities at T = 1, 0.5 and 2', () => {
  near(applyTemperature(LOGITS, 1)[ON], 0.390);
  near(applyTemperature(LOGITS, 0.5)[ON], 0.682);
  near(applyTemperature(LOGITS, 2)[ON], 0.189);
  near(applyTemperature(LOGITS, 1).slice(4, 8), [0.390, 0.237, 0.087, 0.053]);
  near(applyTemperature(LOGITS, 0.5).slice(4, 8), [0.682, 0.251, 0.034, 0.012]);
  near(applyTemperature(LOGITS, 2).slice(4, 8), [0.189, 0.147, 0.089, 0.069]);
});

test('applyTemperature: T = 1 is core.softmax, and the other probabilities of the 12 words match the storyboard', () => {
  assert.deepEqual(applyTemperature(LOGITS, 1), softmax(LOGITS));
  near(applyTemperature(LOGITS, 1)[0], 0.019);
  near(applyTemperature(LOGITS, 0.5)[0], 0.002);
  near(applyTemperature(LOGITS, 2)[0], 0.042);
});

test('applyTemperature: temperature 0 is greedy, an exact one-hot on the top token (core.softmax would throw)', () => {
  const one = applyTemperature(LOGITS, 0);
  assert.deepEqual(one, VOCAB.map((_, i) => (i === ON ? 1 : 0)));
  assert.throws(() => softmax(LOGITS, 0), RangeError);
  assert.deepEqual(applyTemperature([1, 3, 3, 0], 0), [0, 1, 0, 0], 'a tie goes to the lower vocabulary index');
});

test('applyTemperature: bad temperatures throw', () => {
  assert.throws(() => applyTemperature(LOGITS, -1), RangeError);
  assert.throws(() => applyTemperature(LOGITS, Number.NaN), RangeError);
  assert.throws(() => applyTemperature([], 0), RangeError);
});

test('README lesson 16: below 1 the favorite gains, above 1 the tail gains (monotonic in T)', () => {
  const temps = [0.25, 0.5, 0.75, 1, 1.5, 2];
  const favorite = temps.map((t) => applyTemperature(LOGITS, t)[ON]);
  const tail = temps.map((t) => sum(applyTemperature(LOGITS, t).filter((_, i) => !NAMED.includes(i))));
  for (let i = 1; i < temps.length; i += 1) {
    assert.ok(favorite[i] < favorite[i - 1], `"on" should fall as T rises (${temps[i - 1]} → ${temps[i]})`);
    assert.ok(tail[i] > tail[i - 1], `the 12 others should gain as T rises (${temps[i - 1]} → ${temps[i]})`);
  }
  near(tail[3], 0.233);
  near(tail[5], 0.506);
  near(tail[1], 0.020);
});

test('rankTokens: highest first, ties by lower vocabulary index', () => {
  assert.deepEqual(rankTokens(P1), [4, 5, 6, 7, 0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15]);
  assert.deepEqual(rankTokens([0.2, 0.5, 0.2, 0.1]), [1, 0, 2, 3]);
  assert.deepEqual(rankTokens([]), []);
});

test('topK(3) at T = 1: on . and rescaled by 0.714', () => {
  const r = topK(P1, 3);
  near([r.probs[ON], r.probs[DOT], r.probs[AND]], [0.547, 0.331, 0.122]);
  assert.equal(r.kept, 3);
  near(r.mass, 0.714);
  assert.equal(sum(r.probs.filter((_, i) => ![ON, DOT, AND].includes(i))), 0);
});

test('topK(6) keeps on . and the The cat: the others cell reads "2 of 12 kept"', () => {
  const r = topK(P1, 6);
  assert.equal(r.kept, 6);
  const five = collapseOthers(r.probs, NAMED);
  near(five.named, [0.484, 0.294, 0.108, 0.066]);
  assert.equal(five.others.kept, 2);
  assert.equal(five.others.of, 12);
  near(five.others.each, 0.024);
  near(five.others.together, 0.048);
  assert.deepEqual([0, 1].map((i) => r.probs[i] > 0), [true, true]);
  assert.equal(r.probs[2], 0);
});

test('topK(1) is greedy: it equals temperature 0', () => {
  assert.deepEqual(topK(P1, 1).probs, applyTemperature(LOGITS, 0));
  assert.deepEqual(topK(applyTemperature(LOGITS, 0.5), 1).probs, applyTemperature(LOGITS, 0));
  assert.equal(topK(P1, 1).kept, 1);
});

test('topK(k >= 16) and topP(1) are identities', () => {
  [1, 0.5, 2].forEach((t) => {
    const p = applyTemperature(LOGITS, t);
    near(topK(p, 16).probs, p, 1e-12);
    near(topK(p, 99).probs, p, 1e-12);
    near(topP(p, 1).probs, p, 1e-12);
    assert.equal(topK(p, 16).kept, 16);
    assert.equal(topP(p, 1).kept, 16);
  });
});

test('topK rejects a k that is not a whole number of at least 1', () => {
  [0, -2, 1.5, Number.NaN].forEach((k) => assert.throws(() => topK(P1, k), RangeError, `k = ${k}`));
});

test('topP at T = 1: the storyboard\'s kept counts and masses', () => {
  const at = (p) => topP(P1, p);
  assert.equal(at(0.7).kept, 3);
  near(at(0.7).mass, 0.714);
  near([at(0.7).probs[ON], at(0.7).probs[DOT], at(0.7).probs[AND]], [0.547, 0.331, 0.122]);
  assert.equal(at(0.5).kept, 2);
  near(at(0.5).mass, 0.627);
  near([at(0.5).probs[ON], at(0.5).probs[DOT]], [0.622, 0.378]);
  assert.equal(at(0.75).kept, 4);
  near(at(0.75).mass, 0.767);
  assert.equal(at(0.9).kept, 11);
  near(at(0.9).mass, 0.903);
});

test('topP(0.9) cuts inside the tied 12: it keeps The, cat, sat, down, mat, a, dog (vocabulary order)', () => {
  const r = topP(P1, 0.9);
  const keptWords = VOCAB.filter((_, i) => r.probs[i] > 0);
  assert.deepEqual(keptWords, ['The', 'cat', 'sat', 'down', 'on', '.', 'and', 'the', 'mat', 'a', 'dog']);
  assert.equal(collapseOthers(r.probs, NAMED).others.kept, 7);
});

test('topP depends on how sure the model is: p = 0.7 keeps 2 at T = 0.5, 3 at T = 1, 9 at T = 2', () => {
  const kept = [0.5, 1, 2].map((t) => topP(applyTemperature(LOGITS, t), 0.7));
  assert.deepEqual(kept.map((r) => r.kept), [2, 3, 9]);
  near(kept.map((r) => r.mass), [0.933, 0.714, 0.705]);
  assert.equal(collapseOthers(kept[2].probs, NAMED).others.kept, 5, 'five of the twelve tied words');
});

test('topK does not adapt: top-3 keeps 3 at every temperature', () => {
  [0.5, 1, 2].forEach((t) => assert.equal(topK(applyTemperature(LOGITS, t), 3).kept, 3));
});

test('topP keeps the same set as topK when the cut falls on a boundary (p = 0.7 and k = 3 at T = 1)', () => {
  assert.deepEqual(topP(P1, 0.7).probs, topK(P1, 3).probs);
});

test('topP rejects a p outside (0, 1]', () => {
  [0, -0.1, 1.1, Number.NaN].forEach((p) => assert.throws(() => topP(P1, p), RangeError, `p = ${p}`));
});

test('every filter output adds up to 1', () => {
  const outputs = [topK(P1, 3), topK(P1, 6), topP(P1, 0.9), topP(P1, 0.5), samplingDistribution(LOGITS, { temperature: 2, topK: 5, topP: 0.7 }), samplingDistribution(LOGITS, { temperature: 0 })];
  outputs.forEach((r) => assert.ok(Math.abs(sum(r.probs) - 1) < 1e-12, `sum ${sum(r.probs)}`));
});

test('samplingDistribution: no filters is the temperature distribution; kept counts the tokens that can be drawn', () => {
  const none = samplingDistribution(LOGITS, {});
  near(none.probs, P1, 1e-12);
  assert.equal(none.kept, 16);
  near(none.mass, 1, 1e-12);
  const greedy = samplingDistribution(LOGITS, { temperature: 0, topK: 3 });
  assert.equal(greedy.kept, 1, 'at temperature 0 only one token has probability left');
  assert.deepEqual(greedy.probs, applyTemperature(LOGITS, 0));
});

test('samplingDistribution: order is temperature, then top-k, then top-p, then rescale', () => {
  const r = samplingDistribution(LOGITS, { temperature: 1, topK: 5, topP: 0.7 });
  assert.equal(r.kept, 2, 'top-k 5 rescales first (on is then 0.496), so p = 0.7 is reached after two tokens, not three');
  near(r.mass, 0.627);
  const both = samplingDistribution(LOGITS, { temperature: 1, topK: 2, topP: 0.9 });
  assert.equal(both.kept, 2, 'top-k 2 cuts first; top-p then has nothing more to cut');
  near(both.mass, 0.627);
  near([both.probs[ON], both.probs[DOT]], [0.622, 0.378]);
  const tempFirst = samplingDistribution(LOGITS, { temperature: 0.5, topP: 0.7 });
  assert.equal(tempFirst.kept, 2);
  near(tempFirst.mass, 0.933);
});

test('sampleIndex: the storyboard\'s u values at T = 1', () => {
  assert.deepEqual([0.2, 0.55, 0.7, 0.75, 0.95].map((u) => sampleIndex(P1, u)), [4, 5, 6, 7, 13]);
  assert.deepEqual([0.2, 0.55, 0.7, 0.75, 0.95].map((u) => VOCAB[sampleIndex(P1, u)]), ['on', '.', 'and', 'the', 'big']);
});

test('sampleIndex: slice boundaries 0.390, 0.627, 0.714, 0.767 and the ends of [0, 1)', () => {
  assert.equal(sampleIndex(P1, 0), ON);
  assert.equal(sampleIndex(P1, 0.389), ON);
  assert.equal(sampleIndex(P1, 0.391), DOT);
  assert.equal(sampleIndex(P1, 0.713), AND);
  assert.equal(sampleIndex(P1, 0.715), THE);
  assert.equal(sampleIndex(P1, 0.9999999999), 15, 'the last slice belongs to the last-ranked token');
});

test('sampleIndex never returns a token whose probability is 0', () => {
  const filtered = topP(P1, 0.7).probs;
  [0, 0.5, 0.713, 0.9999999999999999].forEach((u) => assert.ok(filtered[sampleIndex(filtered, u)] > 0, `u = ${u}`));
  assert.equal(sampleIndex(applyTemperature(LOGITS, 0), 0.99), ON);
});

test('sampleIndex rejects a u outside [0, 1)', () => {
  [-0.1, 1, 1.5, Number.NaN].forEach((u) => assert.throws(() => sampleIndex(P1, u), RangeError, `u = ${u}`));
});

test('drawSamples: 20 draws at seed 1, T = 1: on 6 . 8 and 1 the 1 others 4; first eight "and on . then was on . the"', () => {
  const draws = drawSamples(P1, 20, 1);
  assert.deepEqual(histogram(draws), [6, 8, 1, 1, 4]);
  assert.equal(draws.slice(0, 8).map((i) => VOCAB[i]).join(' '), 'and on . then was on . the');
});

test('drawSamples: T = 0.5 → 15 · 2 · 0 · 1 · 2, T = 2 → 4 · 2 · 2 · 4 · 8, top-p 0.7 → 13 · 4 · 3 · 0 · 0', () => {
  assert.deepEqual(histogram(drawSamples(applyTemperature(LOGITS, 0.5), 20, 1)), [15, 2, 0, 1, 2]);
  assert.deepEqual(histogram(drawSamples(applyTemperature(LOGITS, 2), 20, 1)), [4, 2, 2, 4, 8]);
  assert.deepEqual(histogram(drawSamples(topP(P1, 0.7).probs, 20, 1)), [13, 4, 3, 0, 0]);
});

test('drawSamples: equal seeds repeat exactly, different seeds differ, greedy always draws the same token', () => {
  assert.deepEqual(drawSamples(P1, 20, 3), drawSamples(P1, 20, 3));
  assert.notDeepEqual(drawSamples(P1, 20, 1), drawSamples(P1, 20, 2));
  assert.notDeepEqual(drawSamples(P1, 8, 2).join(), drawSamples(P1, 8, 3).join());
  assert.deepEqual(drawSamples(applyTemperature(LOGITS, 0), 20, 4), Array(20).fill(ON));
  assert.deepEqual(drawSamples(P1, 0, 1), []);
});

test('drawSamples rejects a bad count or seed', () => {
  assert.throws(() => drawSamples(P1, -1, 1), RangeError);
  assert.throws(() => drawSamples(P1, 2.5, 1), RangeError);
  assert.throws(() => drawSamples(P1, 5, 1.5), RangeError);
});

test('collapseOthers at T = 1: the four named cells and one "12 others" cell', () => {
  const five = collapseOthers(P1, NAMED);
  near(five.named, [0.390, 0.237, 0.087, 0.053]);
  near(five.others.each, 0.019);
  near(five.others.together, 0.233);
  assert.equal(five.others.kept, 12);
  assert.equal(five.others.of, 12);
  assert.ok(Math.abs(sum(five.named) + five.others.together - 1) < 1e-12);
});

test('collapseOthers at T = 0.5 and T = 2 matches the frame 4 numbers', () => {
  const half = collapseOthers(applyTemperature(LOGITS, 0.5), NAMED);
  near(half.named, [0.682, 0.251, 0.034, 0.012]);
  near([half.others.each, half.others.together], [0.002, 0.020]);
  const twice = collapseOthers(applyTemperature(LOGITS, 2), NAMED);
  near(twice.named, [0.189, 0.147, 0.089, 0.069]);
  near([twice.others.each, twice.others.together], [0.042, 0.506]);
});

test('collapseOthers: top-3 leaves 0 of 12 kept', () => {
  const five = collapseOthers(topK(P1, 3).probs, NAMED);
  near(five.named, [0.547, 0.331, 0.122, 0]);
  assert.deepEqual([five.others.each, five.others.together, five.others.kept, five.others.of], [0, 0, 0, 12]);
});

test('no function mutates its inputs', () => {
  const probs = [...P1];
  const logits = [...LOGITS];
  const snapshot = JSON.stringify([probs, logits]);
  rankTokens(probs);
  applyTemperature(logits, 0.5);
  applyTemperature(logits, 0);
  topK(probs, 3);
  topP(probs, 0.7);
  samplingDistribution(logits, { temperature: 2, topK: 5, topP: 0.7 });
  sampleIndex(probs, 0.5);
  drawSamples(probs, 20, 1);
  collapseOthers(probs, NAMED);
  assert.equal(JSON.stringify([probs, logits]), snapshot);
  assert.equal(JSON.stringify(LOGITS), JSON.stringify(logits));
});

test('frozen results: each filter returns a fresh array the caller may keep', () => {
  const a = topK(P1, 3).probs;
  const b = topK(P1, 3).probs;
  assert.notEqual(a, b);
  assert.deepEqual(a, b);
});
