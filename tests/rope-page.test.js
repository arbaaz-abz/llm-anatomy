import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateLessonSpec } from '../shared/lesson-spec.js';
import { fillClaim, fillText } from '../shared/claims.js';
import { LESSON, lessonFor } from '../architecture/concepts/rope/content.js';
import {
  INITIAL_STATE, checkWork, cellText, trimNumber, fmt3, fixed, degrees, thousands, wavelengthText, listText,
} from '../architecture/concepts/rope/format.js';
import { belowNotes, realHeadRows, slowestWavelength } from '../architecture/concepts/rope/facts.js';
import { ROW, SCORE, COVERAGE, PI_FREQS, YARN_FREQS } from '../architecture/concepts/rope/numbers.js';
import { CAPTIONS, CHECK_WORK } from './rope-expected.js';

const read = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const data = { models: await read('../data/models.json'), hardware: await read('../data/hardware.json'), serving: await read('../data/serving.json'), papers: await read('../data/papers.json') };
const graph = await read('../shared/concepts.json');
const near = (actual, expected, tol = 5e-4) => actual.forEach((v, i) => assert.ok(Math.abs(v - expected[i]) < tol, `[${i}] ${v} vs ${expected[i]}`));

test('the lesson spec is complete, with and without data', () => {
  assert.deepEqual(validateLessonSpec(LESSON), []);
  assert.deepEqual(validateLessonSpec(lessonFor(data)), []);
});

test('captions are the storyboard\'s, verbatim and in order', () => {
  assert.deepEqual(LESSON.animation.steps.map((s) => s.caption), CAPTIONS);
});

test('6 facts rows, every placeholder resolves against data/models.json', () => {
  const lesson = lessonFor(data);
  assert.equal(lesson.facts.rows.length, 6);
  lesson.facts.rows.forEach((row, i) => assert.deepEqual(fillClaim(row.claim, data).missing, [], `row ${i + 1}`));
});

test('Next lists exactly the lessons that take this one as a prereq (README lesson 1)', () => {
  const dependents = graph.concepts.filter((c) => c.prereqs.includes('rope')).map((c) => c.slug);
  assert.deepEqual([...LESSON.links.next].sort(), dependents.sort());
});

test('every prose placeholder resolves and nothing printed is "—"', () => {
  const lesson = lessonFor(data);
  const below = [...Array(10).keys()].flatMap((i) => lesson.animation.belowFor(i));
  const prose = [lesson.hook, ...lesson.intuition, lesson.animation.standIn, lesson.toy.intro, lesson.facts.framing, ...below, ...lesson.takeaways];
  prose.forEach((text) => assert.deepEqual(fillClaim(text, data).missing, [], text.slice(0, 40)));
  [...prose, ...lesson.facts.rows.map((r) => r.claim)].forEach((text) => assert.ok(!fillText(text, data).includes('—'), text.slice(0, 50)));
});

test('the dated rows print the data: bases, partial RoPE, YaRN, NoPE and 1M', () => {
  const rows = lessonFor(data).facts.rows.map((r) => fillText(r.claim, data));
  assert.match(rows[0], /^GPT-3 \(2020\): learned absolute positions, a table of 2,048 rows\.$/);
  assert.match(rows[1], /DeepSeek-V4-Pro 10,000 \(160,000 for its compressed streams\) · gpt-oss 150,000 · MiniMax-M3 5,000,000 · GLM-5.3 8,000,000 · Qwen3.8 10,000,000\.$/);
  assert.match(rows[2], /MiniMax-M3 rotates 64 of 128 dimensions \(0\.5\); Qwen3\.5 rotates 25% \(0\.25\)\.$/);
  assert.match(rows[3], /gpt-oss \(2025\) stretched 4,096 → 131,072 tokens \(factor 32\); DeepSeek-V4-Pro uses factor 16\.$/);
  assert.match(rows[4], /Kimi K3's 24 MLA layers use no position encoding; order comes from its 69 linear-attention layers' decay/);
  assert.match(rows[5], /DeepSeek-V4-Pro 1M, Kimi K3 1\.05M, GLM-5\.3 1\.05M, MiniMax-M3 1\.05M\.$/);
});

test('the derived numbers under the stage equal the storyboard\'s (frames 3 to 10)', () => {
  const below = belowNotes(data);
  assert.match(below[2][0], /q at 3 = \[−0\.282, −1\.980, 0\.478, 0\.148\]/);
  assert.match(below[3][0], /k at 2 = \[−1\.364, −0\.624, 0\.099, −0\.490\]\. Pair dots 1\.621 \+ \(−0\.025\) = 1\.596; unrotated it was 3\.0\./);
  assert.match(below[4][0], /q at 13 = \[−0\.840, 1\.815, 0\.134, 0\.482\]; k at 12 = \[0\.805, 1\.266, 0\.466, −0\.181\]\. The score is 1\.596 at positions \(3, 2\), 1\.596 at \(13, 12\) and 1\.596 at \(103, 102\)\./);
  assert.match(below[4][1], /still \[−1\.353, 1\.596, 0\.500\]/);
  assert.match(below[5][0], /\[3\.000, 1\.596, −1\.298, −3\.044, −2\.058, 0\.731, 2\.739, 2\.101\]\. Wavelengths 6\.3 and 62\.8 tokens\./);
  assert.match(below[6][0], /\[3\.000, 1\.618, −1\.253, −2\.977, −1\.971, 0\.838, 2\.866, 2\.244\]/);
  assert.equal(below[6][1], 'Slowest turn of a head of 128 numbers (64 pairs): base 10,000 → 54,410 tokens · 150,000 (gpt-oss) → 782,338 · 10,000,000 → 48.8 million.');
  assert.match(below[7][0], /pair 2 reaches 1\.5 rad\. At 64 tokens it reaches 6\.3 rad, never seen\. After ÷ 4: 1\.575 rad\. New row \[3\.000, 2\.900, 2\.620, 2\.176, 1\.596, 0\.915, 0\.175, −0\.578\]\./);
  assert.match(below[8][0], /YaRN-style row: \[3\.000, 1\.615, −1\.261, −2\.989, −1\.986, 0\.820, 2\.843, 2\.218\]\. With position interpolation, offset 4 scored 1\.596, what offset 1 scored before\./);
  assert.equal(below[9][0], 'Qwen3.5: 25% of dimensions rotate · MiniMax-M3: 50% (64 of 128) · Kimi K3\'s MLA layers: none.');
});

test('"Check my work" for the opening state is the hand-checked text', () => {
  assert.equal(checkWork(INITIAL_STATE), CHECK_WORK);
});

test('"Check my work" for another state: base 10,000, "down" at position 1, sat at 13', () => {
  const lines = checkWork({ ...INITIAL_STATE, kToken: 3, qPos: 13, kPos: 1, base: 10_000 }).split('\n');
  assert.equal(lines.length, 7);
  assert.match(lines[0], /^pair 1 {2}q_sat \(0, 2\) turned 13 × 1 = 13 rad {9}→ \(−0\.840, 1\.815\)$/);
  assert.match(lines[3], /^pair 2 {2}q_sat \(0\.5, 0\) turned 13 × 0\.01 = 0\.13 rad {2}→ \(0\.496, 0\.065\)$/);
  assert.equal(lines[6], 'score  −0.307 + 0.308 = 0.001   (unrotated −0.750)');
});

test('stage cells: one format, 2 decimals, real minus and no negative zero (a signed cell fits five characters)', () => {
  assert.deepEqual([0, 2, 0.5, -0.282, -1.98, 0.478, 1.596, -1.298, -3.044, 3, -0.001].map(cellText), ['0.00', '2.00', '0.50', '−0.28', '−1.98', '0.48', '1.60', '−1.30', '−3.04', '3.00', '0.00']);
  assert.ok([0.5, -3.044, -Infinity].every((v) => /Infinity|^.{1,5}$/.test(cellText(v))));
  assert.equal(cellText(null), '');
  assert.equal(cellText(Number.NaN), '');
});

test('number helpers: no negative zero, real minus, thousands separators', () => {
  assert.deepEqual([fmt3(-0.0001), fmt3(-0.025), fixed(1)(171.887), trimNumber(-0.0001), trimNumber(0.3), trimNumber(-0.5)], ['0.000', '−0.025', '171.9', '0', '0.3', '−0.5']);
  assert.deepEqual([degrees(3), degrees(0.3), thousands(54_410), thousands(-1500), wavelengthText(62.83)], ['171.9°', '17.2°', '54,410', '−1,500', '62.8 tokens']);
  assert.equal(listText([3, -1.298]), '[3.000, −1.298]');
});

test('real heads: the slowest wavelengths at the data\'s bases equal the storyboard\'s (frame 7 strip)', () => {
  assert.deepEqual([10_000, 150_000, 10_000_000].map((b) => Math.round(slowestWavelength(b))), [54_410, 782_338, 48_843_285]);
  const rows = realHeadRows(data);
  assert.deepEqual(rows.map((r) => r.base), [10_000, 150_000, 5_000_000, 8_000_000, 10_000_000]);
  assert.deepEqual(realHeadRows(null).map((r) => r.tokens), [null, null, null, null, null]);
});

test('the animation\'s numbers are regenerated by math/rope.js', () => {
  near(SCORE.pairs, [1.621, -0.025]);
  near(ROW.pi, [3, 2.9, 2.62, 2.176, 1.596, 0.915, 0.175, -0.578]);
  near(ROW.yarn, [3, 1.615, -1.261, -2.989, -1.986, 0.82, 2.843, 2.218]);
  assert.deepEqual([...PI_FREQS], [0.25, 0.025]);
  assert.deepEqual([...YARN_FREQS], [1, 0.025]);
  assert.deepEqual(COVERAGE.plain.map((p) => p.allSeen), [true, false]);
  assert.deepEqual(COVERAGE.pi.map((p) => p.allSeen), [true, true]);
});

test('the lesson never mutates the data', () => {
  const before = JSON.stringify(data);
  lessonFor(data).animation.belowFor(6);
  assert.equal(JSON.stringify(data), before);
});
