import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateLessonSpec } from '../shared/lesson-spec.js';
import { fillClaim, fillText, lookupFact } from '../shared/claims.js';
import { LESSON, lessonFor } from '../training/concepts/midtraining/content.js';
import { checkWork, INITIAL_STATE, DECAY_VALUES, sci, fmt3, ofPeak, percentText, SCHEDULES, scheduleOptions, mainShare, tailShare } from '../training/concepts/midtraining/format.js';
import { runData, stageText, glmBudget, contextLabels, labelK } from '../training/concepts/midtraining/facts.js';
import { toyView, stageRows, curves, nemotronRate } from '../training/concepts/midtraining/toy-view.js';
import { tryThis } from '../training/concepts/midtraining/try-this.js';
import { FALLBACK, DIAL_STAND_IN } from '../training/concepts/midtraining/numbers.js';
import { barParts, lastStageBox } from '../training/concepts/midtraining/context-bar.js';
import { CAPTIONS, CHECK_WORK, CHECK_WORK_IN_DECAY_LINE } from './midtraining-expected.js';
import { lrAt, attentionCostRatio } from '../math/schedule.js';
import { budgetShares } from '../math/pipeline.js';

const read = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const data = { models: await read('../data/models.json'), hardware: await read('../data/hardware.json'), serving: await read('../data/serving.json'), papers: await read('../data/papers.json') };
const graph = await read('../shared/concepts.json');
const fact = (id, key) => lookupFact(data.models, id, key).value;

test('the lesson spec is complete, with and without data', () => {
  assert.deepEqual(validateLessonSpec(LESSON), []);
  assert.deepEqual(validateLessonSpec(lessonFor(data)), []);
});
test('captions are the storyboard\'s, verbatim and in order', () => assert.deepEqual(LESSON.animation.steps.map((s) => s.caption), CAPTIONS));
test('9 facts rows, every placeholder resolves against data/models.json', () => {
  assert.equal(LESSON.facts.rows.length, 9);
  LESSON.facts.rows.forEach((row, i) => assert.deepEqual(fillClaim(row.claim, data).missing, [], `row ${i + 1}`));
});
test('Next lists exactly the lessons that take this one as a prereq (none)', () => {
  const dependents = graph.concepts.filter((c) => c.prereqs.includes('midtraining')).map((c) => c.slug);
  assert.deepEqual([...LESSON.links.next].sort(), dependents.sort());
});
test('every prose placeholder resolves and no filled text prints "—"', () => {
  const lesson = lessonFor(data);
  const prose = [lesson.hook, ...lesson.intuition, lesson.toy.intro, lesson.animation.standIn, ...lesson.math.notes, lesson.facts.framing, ...lesson.facts.prose, ...lesson.takeaways];
  prose.forEach((text) => assert.deepEqual(fillClaim(text, data).missing, [], text.slice(0, 40)));
  [...prose, ...lesson.facts.rows.map((r) => r.claim)].forEach((text) => assert.ok(!fillText(text, data).includes('—'), text.slice(0, 40)));
});

// ---- "Check my work" and the toy's numbers ----
test('"Check my work" for the default state is the storyboard text', () => assert.equal(checkWork(INITIAL_STATE), CHECK_WORK));
test('inside the decay the WSD line reads the storyboard\'s 90% example', () => {
  assert.ok(checkWork({ ...INITIAL_STATE, stopAt: 90 }).split('\n').includes(CHECK_WORK_IN_DECAY_LINE));
});
test('"Check my work" names the plateau, the decay start and the minus-sqrt fall by where the stop falls', () => {
  assert.equal(checkWork({ ...INITIAL_STATE, stopAt: 80 }).split('\n')[1], 'stop at 80%: where the decay begins → 1.000 of peak');
  const root = checkWork({ ...INITIAL_STATE, schedule: 'wsd-minus-sqrt', stopAt: 90 }).split('\n');
  assert.equal(root[0], 'WSD minus-sqrt: the decay starts at 100% − 20% = 80% of the run');
  assert.equal(root[1], 'stop at 90%: inside the decay → 1 − √((90% − 80%) ÷ 20%) = 0.293 of peak');
  assert.match(checkWork({ ...INITIAL_STATE, stopAt: 100 }).split('\n')[2], /½ × \(1 − 1\.000\) = 0\.000 of peak$/);
});
test('the toy starts where the animation ends', () => {
  assert.deepEqual({ ...INITIAL_STATE }, { schedule: 'wsd', decayFrac: 20, preset: 'custom', stopAt: 60, run: 'glm-5' });
});
test('the default readouts equal math/schedule.js for the same inputs', () => {
  const view = toyView(INITIAL_STATE, data);
  assert.equal(view.chosen.lr, fmt3(lrAt(0.6, { kind: 'wsd', total: 1, decayStart: 0.8 })));
  assert.equal(view.other.lr, fmt3(lrAt(0.6, { kind: 'cosine', total: 1 })));
  assert.deepEqual([view.chosen.lr, view.other.lr, view.nemotron], ['1.000', '0.345', null]);
  assert.equal(toyView({ ...INITIAL_STATE, schedule: 'cosine' }, data).chosen.lr, '0.345');
});
test('Nemotron 3 Super\'s absolute rate: 4.50 × 10⁻⁴ on the plateau, 2.27 × 10⁻⁴ at 85% of its run', () => {
  const on = { ...INITIAL_STATE, schedule: 'wsd-minus-sqrt', preset: 'nemotron' };
  assert.equal(toyView(on, data).nemotron, '4.50 × 10⁻⁴');
  assert.equal(toyView({ ...on, stopAt: 85 }, data).nemotron, '2.27 × 10⁻⁴');
  assert.equal(toyView({ ...on, stopAt: 85 }, data).chosen.lr, '0.500');
  assert.ok(Math.abs(nemotronRate({ ...on, stopAt: 100 }, runData(data).nemotron) - fact('nemotron-3-super', 'lr_floor')) < 1e-12);
});
test('sci keeps three significant figures with a real minus in the exponent', () => assert.deepEqual([sci(4.5e-4), sci(2.272e-4), sci(4.5e-6)], ['4.50 × 10⁻⁴', '2.27 × 10⁻⁴', '4.50 × 10⁻⁶']));
test('stage rows for GLM-5: tokens, shares of the run (budgetShares) and attention work (attentionCostRatio)', () => {
  const rows = stageRows('glm-5', data);
  assert.deepEqual(rows.map((r) => r.tokens), ['27T', '1T', '500B', '50B']);
  assert.deepEqual(rows.map((r) => r.share), ['94.57%', '3.50%', '1.75%', '0.18%']);
  assert.deepEqual(rows.map((r) => r.attention), ['1×', '8×', '32×', '50×']);
  const shares = budgetShares(runData(data).stages).parts.map((p) => p.share);
  assert.deepEqual(rows.map((r) => r.share), shares.map(mainShare));
  assert.equal(attentionCostRatio(200, 4), 50);
});
test('the other three runs publish no stage tokens: not published, and their own attention ratios', () => {
  ['minimax-m2', 'deepseek-v4-pro', 'kimi-k3'].forEach((id) => stageRows(id, data).forEach((r) => assert.deepEqual([r.tokens, r.share], ['not published', 'not published'])));
  assert.deepEqual(stageRows('kimi-k3', data).map((r) => r.attention), ['1×', '8×', '32×', '125×']);
  assert.deepEqual(stageRows('deepseek-v4-pro', data).map((r) => r.attention), ['1×', '4×', '16×', '250×']);
  assert.deepEqual(stageRows('minimax-m2', data).map((r) => r.attention), ['1×', '4×', '24×']);
});
test('the plot curves and the followed stop marker come from lrAt', () => {
  const c = curves(INITIAL_STATE);
  assert.deepEqual(c.marker, { x: 0.6, y: 1, label: '60%: 1.000 of peak', followed: true });
  assert.equal(c.decayStart, 0.8);
  assert.equal(curves({ ...INITIAL_STATE, schedule: 'cosine' }).decayStart, null);
  assert.equal(c.chosen.length, 101);
  assert.equal(c.other[60][1], lrAt(0.6, { kind: 'cosine', total: 1 }));
});
test('try this: the numbers are the storyboard\'s', () => {
  const [one, two, three] = tryThis(data).map((t) => t.join(' '));
  assert.match(one, /stop at 60%\. Under \*\*cosine\*\* the learning rate there is 0\.345 of peak.*\*\*WSD\*\*: 1\.000/);
  assert.match(two, /Nemotron 3 Super 20%.*MiniMax-M2 31\.8%/);
  assert.match(two, /5T of 25T, 9\.3T of 29\.2T/);
  assert.match(three, /4K 94\.57% of the run; zoomed, 32K 64\.5%, 128K 32\.3%, 200K 3\.2% of the last 1\.55T \(200K is 0\.18% of the run\).*8×, 32×, 50×/);
});

// ---- data: the stand-ins equal the data (P3-R13) and the printed strings come from it ----
test('numbers.js stand-ins equal the data', () => {
  const f = FALLBACK;
  assert.deepEqual(f.glm5.stages.map((s) => s[1]), [fact('glm-5', 'base_tokens'), fact('glm-5', 'stage_32k_tokens'), fact('glm-5', 'stage_128k_tokens'), fact('glm-5', 'stage_200k_tokens')]);
  assert.equal(f.glm5.reportedTotal, fact('glm-5', 'pretrain_tokens'));
  assert.deepEqual(f.nemotron, { peak: fact('nemotron-3-super', 'lr_peak'), floor: fact('nemotron-3-super', 'lr_floor'), warmupTokens: fact('nemotron-3-super', 'lr_warmup_tokens'), decayTokens: fact('nemotron-3-super', 'lr_decay_tokens'), total: fact('nemotron-3-super', 'pretrain_tokens'), shape: fact('nemotron-3-super', 'lr_decay_shape') });
  assert.deepEqual(f.minimax, { constantTokens: fact('minimax-m2', 'constant_phase_tokens'), decayTokens: fact('minimax-m2', 'decay_phase_tokens') });
  assert.deepEqual(runData(null).stages, runData(data).stages);
});
test('the context labels of each run equal the data\'s context_stages strings', () => {
  const runs = runData(data).runs;
  ['minimax-m2', 'deepseek-v4-pro', 'kimi-k3'].forEach((id) => assert.deepEqual(runs[id].stages.map((s) => s.name), FALLBACK.runs[id].labels));
  assert.deepEqual(contextLabels('4K → 16K → 64K → 1M'), ['4K', '16K', '64K', '1M']);
  assert.deepEqual([labelK('4K'), labelK('192K'), labelK('1M')], [4, 192, 1000]);
});
test('the decay-share chips: Nemotron 20% and MiniMax-M2 31.8% are on the slider, from the data', () => {
  const run = runData(data);
  assert.equal(run.nemotron.decayPercent, 20);
  assert.equal(run.minimax.decayPercent, 31.8);
  assert.ok(DECAY_VALUES.includes(run.nemotron.decayPercent) && DECAY_VALUES.includes(run.minimax.decayPercent));
  assert.equal(percentText(31.8), '31.8%');
});
test('the DeepSeek-V4 row fills Flash\'s schedule from its own data key, not typed text', () => {
  const row = LESSON.facts.rows[1].claim;
  assert.match(row, /\{deepseek-v4-flash\.lr_schedule\}/);
  assert.doesNotMatch(row, /2\.7e-4|75\.5M/);
  assert.deepEqual(fillClaim(row, data).missing, []);
  assert.match(fillText(row, data), /DeepSeek-V4-Flash: peak 2\.7e-4, batch ramped to 75\.5M tokens\./);
});
test('review fixes: schedule chips run cosine, WSD, WSD minus-sqrt; lessons are named by title; try-this insights are lower case', () => {
  assert.deepEqual(SCHEDULES.map((o) => o.value), ['cosine', 'wsd', 'wsd-minus-sqrt']);
  const text = JSON.stringify([lessonFor(data).intuition, LESSON.facts.framing]);
  assert.match(text, /look familiar \(see \[\[rope\]\]\)/);
  assert.match(text, /rescale their rotations \(see \[\[rope\]\]\)/);
  const items = tryThis(data);
  assert.deepEqual(items.map(([, insight]) => insight), ['WSD lets you choose the end late.', 'the decay window where the best data goes is a fifth to a third of these runs:', 'context is extended on a thin slice of tokens,']);
  items.forEach(([prompt]) => assert.match(prompt, /[.:]$|[.:] /, prompt.slice(-30)));
});
test('stage text and intuition print the data\'s numbers: 28.5T reported, 28.55T published, 5.4% mid-training', () => {
  const text = stageText(data);
  assert.equal(text.glmShare, 'GLM-5 mid-training 1.55T of 28.55T = 5.4%');
  assert.equal(text.glmSum, '(28.5T reported; its published stages sum to 28.55T)');
  assert.equal(text.nemotronDecay, 'Nemotron 3 Super decay: last 5T of 25T (20%)');
  assert.equal(text.minimaxDecay, 'MiniMax-M2 decay: 9.3T of 29.2T (31.8%)');
  assert.equal(text.olmo, 'Olmo 3: ~100B tokens (reported)');
  assert.equal(text.kimiLayers, 'Kimi K3: 24 full-attention layers with no positional encoding; nothing to rescale');
  const prose = lessonFor(data).intuition[2];
  assert.match(prose, /50 times as many earlier tokens as at 4K/);
  assert.match(prose, /GLM-5 goes 4K → 32K \(1T tokens\) → 128K \(500B\) → 200K \(50B\): the 200K stage is 0\.18% of the run \(GLM-5's 28\.5T; its published stages sum to 28\.55T\)/);
});
test('frame 6\'s zoomed tail shares are shares of the last 1.55T: 64.5, 32.3, 3.2', () => {
  const { tail } = glmBudget(runData(data));
  assert.equal(tail.knownTotal, 1.55e12);
  assert.deepEqual(tail.parts.map((p) => tailShare(p.share)), ['64.5%', '32.3%', '3.2%']);
});
test('the stage bar: unpublished stages are { value: null }; the 200K segment is the 18 px tail segment', () => {
  const kimi = runData(data).runs['kimi-k3'].stages;
  assert.ok(barParts(kimi).every((p) => p.value === null));
  assert.equal(lastStageBox(kimi, { x: 0, y: 0 }), null);
  const box = lastStageBox(runData(data).stages, { x: 10, y: 26 });
  assert.ok(Math.abs(box.w - 18.06) < 0.01 && box.y === 80 && box.h === 14);
});
test('the dial stand-in: the slowest pair turns 0.015 turn at 4K and ¾ turn at 200K', () => {
  assert.equal(DIAL_STAND_IN.turnsAt200K, 0.75);
  assert.ok(Math.abs(DIAL_STAND_IN.turnsAt4K - 0.015) < 1e-12);
  assert.equal(scheduleOptions(INITIAL_STATE).decayStart, 0.8);
  assert.equal(ofPeak(0), '0');
});
test('lessonFor and the toy never mutate the data', () => {
  const before = JSON.stringify(data);
  lessonFor(data);
  toyView(INITIAL_STATE, data);
  tryThis(data);
  assert.equal(JSON.stringify(data), before);
});
