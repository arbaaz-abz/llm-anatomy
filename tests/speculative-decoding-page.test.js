import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateLessonSpec } from '../shared/lesson-spec.js';
import { fillClaim, fillText } from '../shared/claims.js';
import { RUNNING_EXAMPLE, stepTime, maxUsersPerGpu } from '../math/serving.js';
import { batchSpeedup, simpleSpeedup, expectedTokens } from '../math/specdec.js';
import { VOCAB } from '../math/sampling.js';
import { LESSON, lessonFor } from '../serving/concepts/speculative-decoding/content.js';
import { checkWork, probText, ratioText, cellText } from '../serving/concepts/speculative-decoding/format.js';
import { BELOW, factRows, stageText } from '../serving/concepts/speculative-decoding/facts.js';
import { INITIAL_STATE, toyView, usersFit, batchStops, mtpPresets, usersNote } from '../serving/concepts/speculative-decoding/toy-view.js';
import { tryThis } from '../serving/concepts/speculative-decoding/try-this.js';
import { stepParts, partialSeries, userGrid } from '../serving/concepts/speculative-decoding/sweeps.js';
import * as N from '../serving/concepts/speculative-decoding/numbers.js';
import { CAPTIONS, CHECK_WORK, TRY_THIS, USERS_NOTE } from './speculative-decoding-expected.js';

const read = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const data = { models: await read('../data/models.json'), hardware: await read('../data/hardware.json'), serving: await read('../data/serving.json'), papers: await read('../data/papers.json') };
const graph = await read('../shared/concepts.json');
const M = { ...RUNNING_EXAMPLE, context: 1024 };

test('the lesson spec is complete, with and without data', () => {
  assert.deepEqual(validateLessonSpec(LESSON), []);
  assert.deepEqual(validateLessonSpec(lessonFor(data)), []);
});
test('captions are the storyboard\'s, verbatim and in order', () => assert.deepEqual(LESSON.animation.steps.map((s) => s.caption), CAPTIONS));
test('7 facts rows, every placeholder resolves against data/*.json, nothing unfilled', () => {
  assert.equal(LESSON.facts.rows.length, 7);
  LESSON.facts.rows.forEach((row, i) => assert.deepEqual(fillClaim(row.claim, data).missing, [], `row ${i + 1}`));
  const filled = factRows().map((r) => fillClaim(r.claim, data).segments.map((s) => s.text).join(''));
  filled.forEach((t) => assert.doesNotMatch(t, /[{}—]/, t));
  assert.match(filled[1], /^DeepSeek-V3 \(2024 report\).*accepted 85–90% of the time, about 1\.8× tokens per second\.$/);
  assert.match(filled[2], /^EAGLE-3 \(2025\): up to 6\.5× .* about 1\.4× better than EAGLE-2 .*; 1\.38× throughput at batch 64/);
  assert.match(filled[3], /up to 1\.69× over EAGLE-3 on B200/);
  assert.match(filled[5], /raised per-user throughput 87% for DeepSeek-R1/);
});
test('Next lists exactly the lessons that take this one as a prereq (README lesson 1)', () => {
  const dependents = graph.concepts.filter((c) => c.prereqs.includes('speculative-decoding')).map((c) => c.slug);
  assert.deepEqual([...LESSON.links.next].sort(), dependents.sort());
});
test('dated text under the stage fills from data and carries no unfilled mark', () => {
  const lesson = lessonFor(data);
  BELOW.flat().forEach((t) => assert.deepEqual(fillClaim(t, data).missing, [], t.slice(0, 60)));
  const text = [lesson.hook, ...lesson.intuition, ...BELOW.flatMap((_, i) => lesson.animation.belowFor(i))];
  text.forEach((t) => assert.doesNotMatch(t, /[{}—]/, t.slice(0, 60)));
  assert.match(lesson.animation.belowFor(8)[0], /^EAGLE-3's measured 1\.38× at batch 64 is lower than this toy's 2\.14×; its setup had less KV to read per step\.$/);
  assert.equal(lesson.animation.belowFor(42).length, 0);
  assert.match(lesson.intuition[2], /a 70% acceptance rate and 3 guesses, a round yields 2\.53 tokens on average.* about 2\.2× on the running example/);
});
test('frame 10 rows print the data back; without data they print dashes, never stale numbers', () => {
  const rows = stageText(data).rows;
  assert.deepEqual(rows.map((r) => r.result), ['second token accepted 85–90%, about 1.8× tokens/s', 'up to 6.5× at small batch, 1.38× at batch 64', 'up to 1.69× over EAGLE-3 on B200']);
  assert.deepEqual(rows.map((r) => r.setup), ['DeepSeek-V3 (2024 report)', 'EAGLE-3 (2025, SGLang)', 'parallel drafter (vLLM)']);
  assert.match(stageText(null).rows[1].result, /—/);
});
test('the stand-ins match the storyboard and the shared vocabulary', () => {
  assert.deepEqual(N.PROMPT, ['The', 'cat', 'sat']);
  assert.deepEqual(N.GUESSES, ['down', 'on', 'a']);
  assert.equal(N.PICK, 'the');
  assert.deepEqual(N.ZOOM_WORDS, ['down', 'on', 'up', 'big']);
  N.ZOOM_WORDS.forEach((w) => assert.ok(VOCAB.includes(w)));
  assert.deepEqual([N.P, N.Q], [[0.6, 0.25, 0.1, 0.05], [0.7, 0.2, 0.05, 0.05]]);
  assert.deepEqual(N.MODEL, { ...RUNNING_EXAMPLE, context: 1024 });
});

test('the step bars are stepTime\'s parts: weights + KV + activations is the memory time; frame 3 is 14.7 ms either way', () => {
  const plain = stepParts({ tokens: 1, seqs: 1 });
  const total = plain.reading.reduce((a, r) => a + r.s, 0);
  assert.ok(Math.abs(total - stepTime({ ...M, tokens: 1, seqs: 1 }).memoryS) < 1e-12);
  assert.equal(plain.bound, 'memory');
  const verify = stepParts({ tokens: 4, seqs: 1 });
  assert.ok(Math.abs(verify.mathS - 283e-6) < 1e-6);
  assert.deepEqual(plain.reading.map((r) => r.label), ['weights read', 'KV read', 'activations']);
});

test('"Check my work" at the default is the storyboard text and follows the state', () => {
  assert.equal(checkWork(INITIAL_STATE), CHECK_WORK);
  assert.equal(toyView(INITIAL_STATE, data).checkWork, CHECK_WORK);
  assert.match(checkWork({ alpha: 0.5, k: 8, c: 0.2 }), /= 2\.00 tokens per round\nspeedup = E \/ \(1 \+ k·c\) = 2\.00 \/ \(1 \+ 8 · 0\.2\) = 2\.00 \/ 2\.6 = 0\.77× /);
});

test('formatters: ratios below 1 are plain numbers, probabilities are shortest, cells keep two decimals', () => {
  assert.equal(ratioText(2.2026), '2.2×');
  assert.equal(ratioText(0.9114), '0.91×');
  assert.equal(ratioText(0.9999), '1×');
  assert.equal([0.7, 0.85, 0.05, 0.1, 0].map(probText).join(' '), '0.7 0.85 0.05 0.1 0');
  assert.deepEqual([0, 0.6, 0.5].map(cellText), ['0', '0.60', '0.50']);
});

test('toy view: the default state prints the animation\'s numbers', () => {
  const v = toyView(INITIAL_STATE, data);
  assert.deepEqual([v.tokens, v.simple, v.batchSpeedup, v.plain, v.verify, v.verifyBound], ['2.53', '2.2×', '2.2×', '14.7 ms', '14.7 ms', 'memory-bound']);
  assert.deepEqual([v.position.keep, v.position.acceptance, v.position.leftoverText, v.position.resultText], ['0.857', '0.900', '0 · 0.50 · 0.50 · 0', '0.60 · 0.25 · 0.10 · 0.05']);
  assert.equal(v.batchLabel, 'Speedup at 1 user');
});
test('toy view: every readout equals the shared function for the same inputs (Review Focus 1)', () => {
  for (const [alpha, k, c, batch] of [[0.7, 3, 0.05, 64], [0.7, 3, 0.05, 128], [0.85, 1, 0.05, 211], [0.5, 8, 0.2, 16]]) {
    const v = toyView({ ...INITIAL_STATE, alpha, k, c, batch }, data);
    const r = batchSpeedup({ alpha, k, c, batch, model: M });
    assert.equal(v.batchSpeedup, ratioText(r.speedup));
    assert.equal(v.tokens, expectedTokens(alpha, k).toFixed(2));
    assert.equal(v.simple, ratioText(simpleSpeedup(alpha, k, c)));
    assert.equal(v.verifyBound, `${stepTime({ ...M, tokens: batch * (k + 1), seqs: batch }).bound}-bound`);
  }
  const plain = toyView({ ...INITIAL_STATE, batch: 1 }, data).plain;
  assert.equal(plain, `${(stepTime({ ...M, tokens: 1, seqs: 1 }).timeS * 1e3).toPrecision(3)} ms`);
});
test('toy view: the storyboard\'s batch rows (128 users math-bound, 211 users at k 5 below 1)', () => {
  const at = (patch) => toyView({ ...INITIAL_STATE, ...patch }, data);
  assert.deepEqual([at({ batch: 64 }).batchSpeedup, at({ batch: 128 }).batchSpeedup, at({ batch: 211 }).batchSpeedup], ['2.14×', '1.53×', '1.19×']);
  const h = at({ batch: 128 });
  assert.deepEqual([h.plain, h.verify, h.verifyBound], ['24 ms', '36.2 ms', 'compute-bound']);
  assert.equal(at({ batch: 211, k: 5 }).batchSpeedup, '0.91×');
});
test('the batch slider\'s last stop is the largest count that fits: 211 at 1,024 tokens, 105 at 2,048', () => {
  assert.deepEqual(batchStops(data), [1, 4, 16, 64, 128, 211]);
  assert.equal(usersFit(data), 211);
  assert.equal(usersFit(data, 2048), 105);
  assert.equal(maxUsersPerGpu(71e9, 327680 * 1024), 211);
  assert.equal(usersNote(data), USERS_NOTE);
  assert.throws(() => usersFit({ hardware: { entries: [] } }), RangeError);
});
test('MTP presets come from deepseek-v3-mtp.acceptance_pct', () => {
  assert.deepEqual(mtpPresets(data), [{ value: 0.85, label: 'MTP 0.85' }, { value: 0.9, label: 'MTP 0.9' }]);
  assert.deepEqual(mtpPresets(null), []);
  const v = toyView({ ...INITIAL_STATE, alpha: 0.85, k: 1 }, data);
  assert.deepEqual([v.tokens, v.simple], ['1.85', '1.76×']);
});
test('position panel: only overrated guesses are rejected, the result is p every time', () => {
  const keeps = N.ZOOM_WORDS.map((guess) => toyView({ ...INITIAL_STATE, guess }, data).position);
  assert.deepEqual(keeps.map((p) => p.keep), ['0.857', '1.000', '1.000', '1.000']);
  keeps.forEach((p) => assert.equal(p.resultText, '0.60 · 0.25 · 0.10 · 0.05'));
  assert.throws(() => toyView({ ...INITIAL_STATE, guess: 'cat' }, data), RangeError);
});

test('try-this text is the storyboard\'s, computed', () => {
  const text = tryThis(data).map((t) => `${t.prompt} → Insight: ${t.insight}${t.rest}`);
  assert.deepEqual(text, TRY_THIS);
  tryThis(null).forEach((t) => assert.doesNotMatch(t.rest, /[{}]/));
});

test('sweeps: partial series end on an interpolated point; the user grid ends on the stop', () => {
  assert.deepEqual(partialSeries([[1, 1], [2, 3], [3, 5]], 1), [[1, 1], [2, 3], [3, 5]]);
  assert.deepEqual(partialSeries([[1, 1], [2, 3], [3, 5]], 0.75), [[1, 1], [2, 3], [2.5, 4]]);
  assert.equal(userGrid(211).at(-1), 211);
  assert.deepEqual(userGrid(10), [1, 5, 10]);
});

test('nothing here mutates its inputs', () => {
  const state = Object.freeze({ ...INITIAL_STATE, batch: 128 });
  const before = JSON.stringify([N.MODEL, N.P, N.Q]);
  toyView(state, data);
  fillText('x', data);
  assert.equal(JSON.stringify([N.MODEL, N.P, N.Q]), before);
});
