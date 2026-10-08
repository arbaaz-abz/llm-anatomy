import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateLessonSpec } from '../shared/lesson-spec.js';
import { fillClaim, fillText } from '../shared/claims.js';
import { budgetShares } from '../math/pipeline.js';
import { formatCount } from '../math/core.js';
import { formatShare } from '../shared/glyphs/bars.js';
import { LESSON, lessonFor } from '../training/concepts/training-pipeline/content.js';
import { ROWS, belowFor } from '../training/concepts/training-pipeline/facts.js';
import { checkWork, totalText, NOTHING_PUBLISHED } from '../training/concepts/training-pipeline/format.js';
import { toyView, tokenParts, tryThis, stageStrip, stopsText, recipeOf, INITIAL_STATE, MODEL_CHIPS } from '../training/concepts/training-pipeline/toy-view.js';
import { RECIPES, RECIPE_ORDER } from '../training/concepts/training-pipeline/recipes.js';
import { nextStage } from '../training/concepts/training-pipeline/select.js';
import { GLM5_TOKENS, STAGE_FIGURES, STAGES, STOPS } from '../training/concepts/training-pipeline/numbers.js';
import { GLM5_PARTS } from '../training/concepts/training-pipeline/frames-budget.js';
import { CAPTIONS, CHECK_WORK, CHECK_WORK_KIMI } from './training-pipeline-expected.js';

const read = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const data = { models: await read('../data/models.json'), hardware: await read('../data/hardware.json'), serving: await read('../data/serving.json'), papers: await read('../data/papers.json') };
const graph = await read('../shared/concepts.json');
const fact = (id, key) => data.models.entries.find((e) => e.id === id).facts[key].value;

test('the lesson spec is complete, with and without data', () => {
  assert.deepEqual(validateLessonSpec(LESSON), []);
  assert.deepEqual(validateLessonSpec(lessonFor(data)), []);
});

test('captions are the storyboard\'s, verbatim and in order', () => assert.deepEqual(LESSON.animation.steps.map((s) => s.caption), CAPTIONS));

test('10 facts rows; every placeholder in rows, notes, hook and intuition resolves; nothing prints a dash', () => {
  assert.equal(LESSON.facts.rows.length, 10);
  assert.equal(ROWS.length, 10);
  const lesson = lessonFor(data);
  const texts = [...ROWS, ...Array.from({ length: 9 }, (_, i) => belowFor(i)).flat(), LESSON.hook, ...LESSON.intuition];
  texts.forEach((t) => assert.deepEqual(fillClaim(t, data).missing, [], t.slice(0, 60)));
  const filled = [...lesson.facts.rows.map((r) => r.claim), ...lesson.intuition, ...Array.from({ length: 9 }, (_, i) => lesson.animation.belowFor(i)).flat()];
  filled.forEach((t) => assert.doesNotMatch(fillText(t, data), /—/, t.slice(0, 60)));
});

test('Next lists exactly the lessons that take this one as a prereq', () => {
  const dependents = graph.concepts.filter((c) => c.prereqs.includes('training-pipeline')).map((c) => c.slug);
  assert.deepEqual([...LESSON.links.next].sort(), dependents.sort());
});

test('every stop link names a real lesson', () => {
  const slugs = new Set(graph.concepts.map((c) => c.slug));
  STOPS.flat().forEach((s) => assert.ok(slugs.has(s), s));
});

test('"Check my work": the GLM-5 default is the storyboard text; Kimi K3 prints one line', () => {
  assert.equal(toyView(INITIAL_STATE, data).checkWork, CHECK_WORK);
  assert.equal(toyView({ model: 'kimiK3', stage: 1 }, data).checkWork, CHECK_WORK_KIMI);
  assert.equal(checkWork([{ name: 'a', value: null }]), NOTHING_PUBLISHED);
});

test('every readout equals the shared budgetShares output for the same inputs', () => {
  for (const model of RECIPE_ORDER) {
    const parts = tokenParts(model, data);
    const r = budgetShares(parts);
    const view = toyView({ model, stage: 1 }, data);
    assert.equal(view.knownTotal, totalText(r.knownTotal));
    r.parts.filter((q) => !q.unknown).forEach((q) => assert.ok(view.checkWork.includes(`= ${formatShare(q.share)}`), `${model} ${q.name}`));
    r.parts.filter((q) => q.unknown && r.knownTotal > 0).forEach((q) => assert.ok(view.checkWork.includes(`${q.name}: not published`)));
  }
  assert.equal(toyView(INITIAL_STATE, data).knownTotal, '28.55T tokens');
  assert.equal(toyView({ model: 'kimiK3', stage: 1 }, data).knownTotal, 'none published');
  assert.equal(toyView({ model: 'olmo3', stage: 1 }, data).knownTotal, '6T tokens');
});

test('shares: GLM-5 94.6% / 5.4%, Olmo 3 98.3% / 1.7%, DeepSeek-V4 and Nemotron 100.0%, Kimi K3 none', () => {
  const pct = (model) => budgetShares(tokenParts(model, data)).parts.map((q) => (q.share == null ? null : formatShare(q.share)));
  assert.deepEqual(pct('glm5'), ['94.6%', '5.4%', null]);
  assert.deepEqual(pct('olmo3'), ['98.3%', '1.7%', null]);
  assert.deepEqual(pct('deepseekV4'), ['100.0%', null]);
  assert.deepEqual(pct('nemotron3Super'), ['100.0%', null]);
  assert.deepEqual(pct('kimiK3'), [null, null, null]);
  assert.equal(budgetShares(tokenParts('kimiK3', data)).unknownCount, 3);
});

test('stand-ins restate the data (P3-R13): GLM-5 stage counts and the printed figures', () => {
  assert.equal(GLM5_TOKENS.pretrain, fact('glm-5', 'base_tokens'));
  assert.equal(GLM5_TOKENS.midTrain, fact('glm-5', 'midtrain_tokens'));
  assert.equal(GLM5_TOKENS.headline, fact('glm-5', 'pretrain_tokens'));
  assert.deepEqual(GLM5_PARTS.map(({ name, value }) => [name, value]), [['pretrain', GLM5_TOKENS.pretrain], ['mid-train', GLM5_TOKENS.midTrain], ['post-training', null]]);
  assert.equal(formatCount(GLM5_TOKENS.headline), '28.5T');
  assert.equal(formatCount(GLM5_TOKENS.pretrain + GLM5_TOKENS.midTrain, { digits: 4 }), '28.55T');
  assert.match(String(fact('glm-5', 'context_stages')), /4K/);
  assert.match(String(fact('glm-5', 'context_stages')), /200K/);
  assert.match(STAGE_FIGURES.midTrainContext, /^4K → 200K$/);
  assert.equal(fact('nemotron-3-super', 'pretrain_tokens'), 25e12);
  assert.equal(fact('deepseek-v4-pro', 'pretrain_tokens'), 33e12);
  assert.match(String(fact('deepseek-v3.2', 'post_training_compute_share')), /10%/);
  assert.match(STAGE_FIGURES.v32Line, /> 10%/);
  assert.match(fillText('{deepseek-r1.release_date|year}', data), /^2025$/);
});

test('the toy: every stage of every preset fills from data; undescribed stages say so', () => {
  assert.equal(MODEL_CHIPS.length, 5);
  for (const model of RECIPE_ORDER) {
    for (let stage = 1; stage <= 6; stage += 1) {
      const view = toyView({ model, stage }, data);
      assert.deepEqual(fillClaim(view.inspector, data).missing, [], `${model} ${stage}`);
      assert.doesNotMatch(view.inspector, /[{}—]/, `${model} ${stage}`);
      const described = recipeOf(model).stages[stage - 1] !== null;
      assert.equal(view.strip[stage - 1].described, described);
      assert.equal(view.inspector.includes('not described in this report'), !described);
    }
  }
  assert.equal(toyView(INITIAL_STATE, data).inspector, 'Stage 1, pretraining: 27T tokens of base pretraining at 4K context');
});

test('try this 1: the merge stage is lit for exactly GLM-5, DeepSeek-V4 and Kimi K3', () => {
  const lit = RECIPE_ORDER.filter((m) => stageStrip(m)[4].described);
  assert.deepEqual(lit, ['glm5', 'deepseekV4', 'kimiK3']);
  assert.match(tryThis(data)[0].prompt, /lit for GLM-5, DeepSeek-V4 and Kimi K3, not described for Nemotron 3 Super and Olmo 3\./);
});

test('try this 2 and 3 print the computed shares', () => {
  const [, two, three] = tryThis(data);
  assert.match(two.prompt, /pretrain 94\.6%, mid-train 5\.4%, post-training not published/);
  assert.match(two.prompt, /Olmo 3: 98\.3% \/ 1\.7%/);
  assert.match(three.prompt, /no published shares/);
  assert.equal(tryThis(data).length, 3);
});

test('stage 4 of GLM-5 is counted in environments, Olmo 3 in pairs and prompts; Kimi K3 lights every stage', () => {
  assert.match(toyView({ model: 'glm5', stage: 4 }, data).inspector, /environments: >10,000/);
  assert.match(toyView({ model: 'olmo3', stage: 4 }, data).inspector, /200K preference pairs \(DPO\), then 105K prompts \(RLVR\)/);
  assert.ok(stageStrip('kimiK3').every((s) => s.described));
  assert.equal(RECIPES.kimiK3.tokens.every(([, key]) => key === null), true);
});

test('stage selection keys and stop text', () => {
  assert.equal(nextStage(1, 'ArrowLeft'), 1);
  assert.equal(nextStage(6, 'ArrowRight'), 6);
  assert.equal(nextStage(3, 'ArrowRight'), 4);
  assert.equal(nextStage(3, 'Home'), 1);
  assert.equal(nextStage(3, 'End'), 6);
  assert.equal(nextStage(3, 'x'), null);
  assert.equal(stopsText(4), 'Taught in [[rlhf-dpo]], [[rlvr-grpo]] and [[agentic-rl]].');
  assert.equal(STAGES.length, 6);
});

test('unknown models and unpublished keys throw clearly', () => {
  assert.throws(() => recipeOf('nope'), RangeError);
  assert.throws(() => tokenParts('glm5', { models: { entries: [] } }), /not a published number/);
});

test('inputs are not mutated', () => {
  const before = JSON.stringify(data.models);
  const state = Object.freeze({ model: 'olmo3', stage: 3 });
  toyView(state, data);
  tryThis(data);
  lessonFor(data);
  assert.equal(JSON.stringify(data.models), before);
  assert.deepEqual(state, { model: 'olmo3', stage: 3 });
});
