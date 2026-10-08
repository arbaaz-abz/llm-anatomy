import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateLessonSpec } from '../shared/lesson-spec.js';
import { fillClaim, fillText, lookupFact } from '../shared/claims.js';
import { LESSON, lessonFor } from '../training/concepts/rlhf-dpo/content.js';
import { checkWork, fmt3, fmtChange, trim1, INITIAL_STATE } from '../training/concepts/rlhf-dpo/format.js';
import { tryThis } from '../training/concepts/rlhf-dpo/try-this.js';
import { toyView } from '../training/concepts/rlhf-dpo/toy-view.js';
import { BT, KL, TOTALS, CLIP, DPO, ROWS as STAND_INS } from '../training/concepts/rlhf-dpo/numbers.js';
import { dpoLoss, bradleyTerry, klPenalizedReward, sigmoid } from '../math/preference.js';
import { klDivergence } from '../math/lm.js';
import { clippedSurrogate } from '../math/grpo.js';
import { CAPTIONS, CHECK_WORK, TRY_THIS } from './rlhf-dpo-expected.js';

const read = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const data = { models: await read('../data/models.json'), hardware: await read('../data/hardware.json'), serving: await read('../data/serving.json'), papers: await read('../data/papers.json') };
const graph = await read('../shared/concepts.json');
const at4 = (x) => Number(x.toFixed(4));

test('the lesson spec is complete, with and without data', () => {
  assert.deepEqual(validateLessonSpec(LESSON), []);
  assert.deepEqual(validateLessonSpec(lessonFor(data)), []);
});

test('captions are the storyboard\'s, verbatim and in order', () => {
  assert.equal(CAPTIONS.length, 11);
  assert.deepEqual(LESSON.animation.steps.map((s) => s.caption), CAPTIONS);
});

test('7 facts rows; every placeholder resolves against data/*.json', () => {
  assert.equal(LESSON.facts.rows.length, 7);
  LESSON.facts.rows.forEach((row, i) => assert.deepEqual(fillClaim(row.claim, data).missing, [], `row ${i + 1}`));
  assert.equal(LESSON.facts.rows.filter((r) => r.derived).length, 1);
});

test('the rows print the data: Nemotron 21 environments, Olmo 200K and 105K, SmolLM3 APO, five "not listed"', () => {
  const rows = lessonFor(data).facts.rows.map((r) => fillText(r.claim, data));
  assert.match(rows[0], /across 21 environments, then SWE-RL.*GenRM \(principle-following, from Qwen3-235B-A22B-Thinking-2507, HelpSteer 3\)\.$/);
  assert.match(rows[1], /rule \+ ORM \+ GRM; human-written anchors\.$/);
  assert.match(rows[2], /verifies with reward models, unit tests, LLM judges, and static checks\.$/);
  assert.match(rows[3], /about 200K "Delta Learning" pairs.*about 105K prompts\.$/);
  assert.match(rows[4], /^SmolLM3 used APO, a DPO-family method\.$/);
  assert.match(rows[5], /^DPO is not listed as a main stage in the DeepSeek-V4-Pro, GLM-5, Kimi K3, MiniMax-M2 and MiMo-V2-Flash reports/);
  ['deepseek-v4-pro', 'glm-5', 'kimi-k3', 'minimax-m2', 'mimo-v2-flash'].forEach((id) => assert.equal(lookupFact(data.models, id, 'dpo_stage').value, 'not listed', id));
});

test('row 4 is flagged "reported" through the data, and the frame 11 note says so', () => {
  assert.equal(fillClaim(LESSON.facts.rows[3].claim, data).reported, true);
  assert.equal(lookupFact(data.models, 'olmo-3', 'dpo_pairs').confidence, 'reported');
  const note = lessonFor(data).animation.belowFor(10)[0];
  assert.equal(note, 'Olmo 3: SFT → DPO (about 200K pairs) → RLVR (about 105K prompts), reported. Nemotron 3 Super: RLVR → SWE-RL → RLHF with a principle-following judge, confirmed.');
  assert.deepEqual(lessonFor(data).animation.belowFor(3), []);
});

test('every prose placeholder resolves and nothing prints "—" or a year', () => {
  const lesson = lessonFor(data);
  const prose = [lesson.hook, ...lesson.intuition, lesson.toy.intro, lesson.facts.framing, ...lesson.math.notes, ...lesson.takeaways, ...lesson.animation.belowFor(10)];
  prose.forEach((text) => assert.deepEqual(fillClaim(text, data).missing, [], text.slice(0, 40)));
  [...prose, ...lesson.facts.rows.map((r) => fillText(r.claim, data))].forEach((text) => {
    assert.ok(!fillText(text, data).includes('—'), text.slice(0, 40));
    assert.ok(!/\b20(1\d|2[0-5])\b/.test(fillText(text, data).replace(/arXiv \d{4}\.\d+|Qwen3-235B-A22B-Thinking-2507/g, '')), `no year in: ${text.slice(0, 40)}`);
  });
});

test('Next lists exactly the lessons that take this one as a prereq; the prereq is sft', () => {
  const dependents = graph.concepts.filter((c) => c.prereqs.includes('rlhf-dpo')).map((c) => c.slug);
  assert.deepEqual([...LESSON.links.next].sort(), dependents.sort());
  assert.deepEqual(graph.concepts.find((c) => c.slug === 'rlhf-dpo').prereqs, ['sft']);
});

test('"Check my work" for the default state is the storyboard text, and the default is where the animation ends', () => {
  assert.equal(checkWork(INITIAL_STATE), CHECK_WORK);
  assert.deepEqual({ ...INITIAL_STATE }, { dChosen: 0.5, dRejected: -0.2, beta: 0.1 });
});

test('"Check my work" keeps its layout for other states', () => {
  const lines = checkWork({ dChosen: -1, dRejected: -3, beta: 0.5 }).split('\n');
  assert.equal(lines[0], 'reward A = 0.5 × (−1) = −0.500');
  assert.equal(lines[1], 'reward B = 0.5 × (−3) = −1.500');
  assert.equal(lines[2], 'gap      = −0.500 − (−1.500) = 1.000');
  assert.equal(lines[3], 'loss     = −ln σ(1.000) = −ln 0.731 = 0.313');
  assert.equal(lines[4], 'weight   = σ(−1.000) = 0.269');
  const zero = checkWork({ dChosen: 0, dRejected: 0, beta: 0.1 }).split('\n');
  assert.equal(zero[2], 'gap      = 0.000 − 0.000 = 0.000');
  assert.equal(zero[4], 'weight   = σ(0.000) = 0.500');
});

test('number formats: real minus, no negative zero, signed slider text', () => {
  assert.deepEqual([fmt3(-0.02), fmt3(-0.0001), fmt3(0.05), fmt3(Number.NaN)], ['−0.020', '0.000', '0.050', '']);
  assert.deepEqual([fmtChange(0.5), fmtChange(-0.2), fmtChange(0), fmtChange(-10)], ['+0.5', '−0.2', '0', '−10.0']);
  assert.deepEqual([trim1(5), trim1(-0.2), trim1(0.1), trim1(-1)], ['5', '−0.2', '0.1', '−1']);
});

test('the toy view prints the storyboard\'s default outputs', () => {
  const v = toyView(INITIAL_STATE);
  assert.deepEqual(v.rewards, ['0.050', '−0.020']);
  assert.deepEqual([v.margin, v.pChosen, v.loss, v.weight], ['0.070', '0.517', '0.659', '0.483']);
  assert.equal(v.lossRef, '0.693 = ln 2 (no preference yet)');
  assert.equal(v.checkWork, CHECK_WORK);
});

test('every toy readout equals the shared function\'s output for the same inputs', () => {
  [[0.5, -0.2, 0.1], [5, -5, 0.1], [10, -10, 0.1], [-1, -3, 0.1], [0.5, -0.2, 0.5], [-7.3, 4.1, 0.5]].forEach(([dChosen, dRejected, beta]) => {
    const r = dpoLoss({ dChosen, dRejected, beta });
    const v = toyView({ dChosen, dRejected, beta });
    assert.deepEqual(v.rewards, [fmt3(r.rewardChosen), fmt3(r.rewardRejected)]);
    assert.equal(v.margin, fmt3(r.margin));
    assert.equal(v.pChosen, fmt3(sigmoid(r.margin)));
    assert.equal(v.loss, fmt3(r.loss));
    assert.equal(v.weight, fmt3(r.weight));
  });
});

test('the three try-this items print with their numbers from math/preference.js and a named insight', () => {
  const items = tryThis();
  assert.equal(items.length, 3);
  assert.deepEqual(items, TRY_THIS);
  assert.deepEqual(items.map(([, insight]) => insight), ['the update fades as a pair is learned.', 'DPO optimizes the gap, not the chosen answer.', 'β is the leash.']);
});

test('the stage numbers are the reproducer\'s (storyboard §11)', () => {
  assert.deepEqual([BT.right.pChosen, BT.right.loss, BT.flipped.loss].map(at4), [0.8176, 0.2014, 1.7014]);
  assert.deepEqual([KL.honest, KL.hacked].map(at4), [0.1838, 2.1031]);
  assert.deepEqual([TOTALS.honest, TOTALS.hacked].map(at4), [0.9081, 0.2484]);
  assert.deepEqual([klPenalizedReward(1.0, KL.honest, 0), klPenalizedReward(1.3, KL.hacked, 0)], [1, 1.3]);
  assert.deepEqual([CLIP.moved.objective, CLIP.safe.objective].map((x) => Number(x.toFixed(2))), [0.24, 0.22]);
  assert.deepEqual([CLIP.moved.clipped, CLIP.safe.clipped], [true, false]);
  assert.deepEqual([DPO.rewardChosen, DPO.rewardRejected, DPO.margin, DPO.loss, DPO.weight].map(at4), [0.05, -0.02, 0.07, 0.6588, 0.4825]);
  assert.equal(at4(klDivergence(STAND_INS.policy, STAND_INS.reference)), 0.1838);
  assert.deepEqual(clippedSurrogate(1.25, 0.2), CLIP.moved);
  assert.equal(at4(bradleyTerry(1.2, -0.3).loss), 0.2014);
});

test('lessonFor never mutates the data', () => {
  const before = JSON.stringify(data);
  lessonFor(data);
  assert.equal(JSON.stringify(data), before);
});
