import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateLessonSpec } from '../shared/lesson-spec.js';
import { fillClaim, fillText } from '../shared/claims.js';
import { LESSON, lessonFor } from '../architecture/concepts/sampling/content.js';
import { acceptanceRange, tokensPerStep, tokensPerStepText, mtpTex } from '../architecture/concepts/sampling/facts.js';
import {
  INITIAL_STATE, checkWork, workedTex, trimNumber, fmt3, probText, scoreText, sig, effectiveTemperature, filtersOf, distributionOf,
} from '../architecture/concepts/sampling/format.js';
import { CAPTIONS, CHECK_WORK, CHECK_WORK_HALF, CHECK_WORK_TOP_P, CHECK_WORK_GREEDY } from './sampling-expected.js';

const read = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const data = { models: await read('../data/models.json'), hardware: await read('../data/hardware.json'), serving: await read('../data/serving.json'), papers: await read('../data/papers.json') };
const graph = await read('../shared/concepts.json');
const STATE = INITIAL_STATE;

test('the lesson spec is complete, with and without data', () => {
  assert.deepEqual(validateLessonSpec(LESSON), []);
  assert.deepEqual(validateLessonSpec(lessonFor(data)), []);
});

test('captions are the storyboard\'s, verbatim and in order', () => {
  assert.deepEqual(LESSON.animation.steps.map((s) => s.caption), CAPTIONS);
  assert.equal(CAPTIONS.length, 9);
});

test('6 facts rows, every placeholder in them resolves against data/*.json', () => {
  assert.equal(LESSON.facts.rows.length, 6);
  LESSON.facts.rows.forEach((row, i) => assert.deepEqual(fillClaim(row.claim, data).missing, [], `row ${i + 1}`));
});

test('every prose placeholder resolves, and nothing prints "—" or a leftover brace', () => {
  const lesson = lessonFor(data);
  const below = [...Array(CAPTIONS.length).keys()].flatMap((i) => lesson.animation.belowFor(i));
  const prose = [lesson.hook, ...lesson.intuition, lesson.animation.standIn, lesson.toy.intro, lesson.facts.framing, ...lesson.math.notes, ...lesson.takeaways, ...below];
  prose.forEach((text) => assert.deepEqual(fillClaim(text, data).missing, [], text.slice(0, 40)));
  [...prose, ...lesson.facts.rows.map((r) => fillText(r.claim, data))].forEach((text) => {
    assert.ok(!text.includes('—'), text.slice(0, 40));
    assert.ok(!/[{}]/.test(text), `a placeholder was left unfilled: ${text.slice(0, 60)}`);
  });
  lesson.math.blocks.forEach((b) => assert.ok(!b.tex.includes('—')));
});

test('Next lists exactly the lessons that take this one as a prereq (README lesson 1)', () => {
  const dependents = graph.concepts.filter((c) => c.prereqs.includes('sampling')).map((c) => c.slug);
  assert.deepEqual([...LESSON.links.next].sort(), dependents.sort());
  assert.deepEqual(dependents, ['speculative-decoding']);
});

test('the hook prints Kimi K3\'s vocabulary from the data (163,840), not from the page', () => {
  assert.match(lessonFor(data).hook, /163,840 scores in Kimi K3/);
  assert.doesNotMatch(LESSON.hook, /163/);
});

test('the §8 rows print the storyboard numbers from the data', () => {
  const rows = lessonFor(data).facts.rows.map((r) => fillText(r.claim, data));
  assert.match(rows[0], /163,840 entries in Kimi K3, 201,088 in gpt-oss-120b/);
  assert.match(rows[1], /DeepSeek-V3 \(2024\).*depth 1.*accepted 85–90% of the time, about 1\.8× tokens per second \(for one user/);
  assert.match(rows[2], /DeepSeek-V4 \(2026\): MTP depth 1, also used as an auxiliary training loss/);
  assert.match(rows[3], /GLM-5 \(2026\): shares 3 MTP layers; mean accepted length 2\.76 tokens per step/);
  assert.match(rows[4], /Kimi K3: MTP depth 1, fine-tuned as an EAGLE-3-style draft/);
  assert.match(rows[5], /raised per-user speed by 87% for DeepSeek-R1 on GB300 NVL72 at 128K input \/ 8K output/);
});

test('frame 9\'s note derives "1.85–1.90" from the acceptance range, not from a typed number', () => {
  assert.deepEqual(acceptanceRange(data), [85, 90]);
  assert.deepEqual(tokensPerStep(data), ['1.85', '1.90']);
  assert.equal(tokensPerStepText(data), '1.85–1.90');
  assert.equal(tokensPerStepText(null), '—');
  assert.match(lessonFor(data).animation.belowFor(8)[0], /accepted 85–90% of the time, so one draft gives 1\.85–1\.90 tokens per step on average/);
  assert.match(lessonFor(data).animation.belowFor(7)[0], /MTP depth 1 in DeepSeek-V3 \(2024\) and 1 in DeepSeek-V4-Pro \(2026\); GLM-5 \(2026\) shares 3 MTP layers/);
  assert.match(lessonFor(data).takeaways[2], /85–90% of the time in DeepSeek-V3/);
});

test('the MTP math block: 1 + α with the range from the data, bare without it', () => {
  assert.equal(mtpTex(data), String.raw`\text{MTP (one draft): } \mathbb{E}[\text{tokens per step}] = 1 + \alpha,\qquad \alpha = 0.85\text{–}0.90 \Rightarrow 1.85\text{–}1.90`);
  assert.equal(mtpTex(null), String.raw`\text{MTP (one draft): } \mathbb{E}[\text{tokens per step}] = 1 + \alpha`);
  assert.equal(lessonFor(data).math.blocks[3].tex, mtpTex(data));
});

test('the math panel\'s worked softmax is templated from the toy\'s terms and equals the storyboard block', () => {
  const storyboard = String.raw`\text{worked } (T = 0.5):\ \frac{e^{4}}{e^{4} + e^{3} + e^{1} + e^{0} + 12\,e^{-2}} = 0.682`;
  assert.equal(workedTex(0.5), storyboard);
  assert.ok(LESSON.math.blocks[0].tex.endsWith(storyboard));
  assert.equal(workedTex(1), String.raw`\text{worked } (T = 1):\ \frac{e^{2}}{e^{2} + e^{1.5} + e^{0.5} + e^{0} + 12\,e^{-1}} = 0.390`);
  assert.match(LESSON.math.blocks[0].tex, /hl-p.*hl-z.*hl-t/);
});

test('"Check my work": the default state, the storyboard\'s T = 0.5 example, a top-p state and greedy', () => {
  assert.equal(checkWork(STATE), CHECK_WORK);
  assert.equal(checkWork({ ...STATE, temperature: 0.5 }), CHECK_WORK_HALF);
  assert.equal(checkWork({ ...STATE, topP: 0.7 }), CHECK_WORK_TOP_P);
  assert.equal(checkWork({ ...STATE, greedy: true }), CHECK_WORK_GREEDY);
  assert.equal(checkWork({ ...STATE, greedy: true, topK: 3 }), CHECK_WORK_GREEDY);
});

test('"Check my work" for other temperatures keeps its layout and uses a real minus', () => {
  const lines = checkWork({ ...STATE, temperature: 2 }).split('\n');
  assert.equal(lines.length, 7);
  assert.equal(lines[0], 'on         e^1           = 2.72');
  assert.equal(lines[2], 'and        e^0.25        = 1.28');
  assert.equal(lines[4], '12 others  12 × e^−0.5   = 7.28');
  assert.equal(lines[6], 'on = 2.72 ÷ 14.4 = 0.189');
  const cold = checkWork({ ...STATE, temperature: 0.25 }).split('\n');
  assert.equal(cold[0], 'on         e^8           = 2,981');
  assert.equal(cold[5], 'sum                      = 3,393');
  assert.equal(checkWork({ ...STATE, temperature: 0.75 }).split('\n')[1], '"."        e^2           = 7.39');
  assert.equal(checkWork({ ...STATE, temperature: 0.75 }).split('\n')[4], '12 others  12 × e^−1.33  = 3.16');
});

test('"Check my work" names a top-k cut, and one kept token reads "1 token"', () => {
  assert.equal(checkWork({ ...STATE, topK: 3 }).split('\n').slice(-2).join('\n'), 'top-k 3 kept 3 tokens, holding 0.714\non = 0.390 ÷ 0.714 = 0.547');
  assert.match(checkWork({ ...STATE, topK: 1 }), /top-k 1 kept 1 token, holding 0\.390/);
  assert.match(checkWork({ ...STATE, topK: 5, topP: 0.7 }), /top-k 5 and top-p 0.7 kept 2 tokens, holding 0\.627\non = 0\.390 ÷ 0\.627 = 0\.622$/);
});

test('number formats: real minus, exact zero and one on the stage, three decimals in the table', () => {
  assert.deepEqual([-1, 0.75, 4, -0.5, -0, 2.5].map(scoreText), ['−1', '0.75', '4', '−0.5', '0', '2.5']);
  assert.deepEqual([0, 1, 0.39, 0.0194, 0.5468].map(probText), ['0', '1', '0.390', '0.019', '0.547']);
  assert.deepEqual([0, 1, 0.2333].map(fmt3), ['0.000', '1.000', '0.233']);
  assert.deepEqual([54.598, 7.389, 1, 2981.2, 80.026, 18.934, 0.22].map((v, i) => sig(v, [3, 3, 3, 3, 4, 4, 3][i])), ['54.6', '7.39', '1', '2,981', '80.03', '18.93', '0.22']);
  assert.deepEqual([-2, 0.5, 1.5, 0, -1 / 0.75].map((v) => trimNumber(v)), ['−2', '0.5', '1.5', '0', '−1.33']);
});

test('the toy\'s state helpers: greedy is temperature 0, "off" filters are null, the start equals the animation\'s T = 1', () => {
  assert.deepEqual({ ...INITIAL_STATE }, { temperature: 1, greedy: false, topK: 0, topP: 1, seed: 1 });
  assert.equal(effectiveTemperature(STATE), 1);
  assert.equal(effectiveTemperature({ ...STATE, greedy: true, temperature: 2 }), 0);
  assert.deepEqual(filtersOf(STATE), { topK: null, topP: null });
  assert.deepEqual(filtersOf({ ...STATE, topK: 3, topP: 0.7 }), { topK: 3, topP: 0.7 });
  assert.equal(distributionOf(STATE).kept, 16);
  assert.equal(distributionOf({ ...STATE, topP: 0.7 }).kept, 3);
});

test('lessonFor never mutates the data it reads', () => {
  const before = JSON.stringify(data);
  lessonFor(data).animation.belowFor(8);
  assert.equal(JSON.stringify(data), before);
});

// ---- the toy's view model (toy-view.js) ----
const toyView = await import('../architecture/concepts/sampling/toy-view.js');

test('toy view, default state: the animation\'s probabilities, 16 of 16 kept, seed 1\'s 20 draws and the exact "Check my work"', () => {
  const v = toyView.view(STATE);
  assert.deepEqual(v.rows.map((r) => [r.label, r.pre, r.post]), [['on', '0.390', '0.390'], ['"."', '0.237', '0.237'], ['and', '0.087', '0.087'], ['the', '0.053', '0.053'], ['12 others', '0.019', '0.019']]);
  assert.equal(v.rows[4].postSub, 'together 0.233 · 12 of 12 kept');
  assert.deepEqual([v.kept, v.mass], ['16 of 16', '1.000']);
  assert.deepEqual(v.counts, [6, 8, 1, 1, 4]);
  assert.equal(v.firstEight, 'and on . then was on . the');
  assert.equal(v.checkWork, CHECK_WORK);
  assert.equal(v.drawsTitle, '20 draws, seed 1');
});

test('toy view, top-k 3: the cut tokens read 0.000 and "cut", the survivors are rescaled', () => {
  const v = toyView.view({ ...STATE, topK: 3 });
  assert.deepEqual(v.rows.map((r) => r.post), ['0.547', '0.331', '0.122', '0.000', '0.000']);
  assert.equal(v.rows[3].postSub, 'cut');
  assert.equal(v.rows[4].postSub, 'together 0.000 · 0 of 12 kept');
  assert.deepEqual([v.kept, v.mass], ['3 of 16', '0.714']);
});

test('toy view, greedy: probability 1 on "on", every draw "on"', () => {
  const v = toyView.view({ ...STATE, greedy: true });
  assert.equal(v.rows[0].post, '1.000');
  assert.deepEqual(v.counts, [20, 0, 0, 0, 0]);
  assert.equal(v.firstEight, 'on on on on on on on on');
  assert.equal(v.rows[0].pre, '1.000', 'temperature 0 is already a one-hot, so "after temperature" shows it too');
});

test('toy view, top-p 0.7 at temperature 2: 9 kept, 5 of the 12 tied words', () => {
  const v = toyView.view({ ...STATE, temperature: 2, topP: 0.7 });
  assert.deepEqual([v.kept, v.mass], ['9 of 16', '0.705']);
  assert.match(v.rows[4].postSub, /5 of 12 kept$/);
});

test('try-this items are { prompt, insight }, computed from the same functions, with the storyboard numbers', () => {
  const [one, two, three] = toyView.tryThis();
  assert.match(one.prompt, /"on" 0\.682, the 12 others 0\.020 together; seed 1 draws "on" 15 times in 20\. Temperature 2: "on" 0\.189, others 0\.506; 8 of 20 draws.*Tap Greedy: "on" every time\.$/);
  assert.match(two.prompt, /Top-p 0\.7 at temperature 1: 3 kept\. At 0\.5: 2 kept\. At 2: 9 kept \(5 of the 12 tied words/);
  assert.match(two.prompt, /top-k 3 instead: 3 kept at every temperature\.$/);
  assert.match(three.prompt, /seed 2: the on on \. a and \. on\)\. Return to seed 1 and they come back exactly \("and on \. then was on \. the"\)\.$/);
  assert.deepEqual([one, two, three].map((t) => t.insight), [
    'temperature is one dial from "always the favorite" to "almost uniform"; it changes how the draw spends probability, not what the model knows.',
    'top-p adapts to how sure the model is; top-k does not.',
    'the randomness lives in the sampler, not in the model; fix the seed and the text repeats.',
  ]);
});

test('no reported release year is printed: the Kimi K3 MTP row and the frame-1 note carry none (X-1), and the attention link is not a subject (X-4)', () => {
  const lesson = lessonFor(data);
  const kimiRow = fillText(lesson.facts.rows[4].claim, data);
  assert.equal(kimiRow, 'Kimi K3: MTP depth 1, fine-tuned as an EAGLE-3-style draft.');
  assert.equal(fillClaim(lesson.facts.rows[4].claim, data).reported, false);
  assert.match(lesson.animation.belowFor(0)[0], /^The 16-word vocabulary here is a toy: Kimi K3 has 163,840 entries and gpt-oss-120b \(2025\) has 201,088\./);
  assert.match(lesson.animation.belowFor(3)[0], /^The divisor slider on \[\[attention\]\] is a different temperature/);
});
