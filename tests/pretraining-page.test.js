import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateLessonSpec } from '../shared/lesson-spec.js';
import { fillClaim } from '../shared/claims.js';
import { softmax, formatCount } from '../math/core.js';
import { tokenLoss, meanLoss, perplexity, uniformLoss } from '../math/lm.js';
import { LESSON, lessonFor, INTUITION, MATH_NOTES } from '../training/concepts/pretraining/content.js';
import { CAPTIONS as PAGE_CAPTIONS } from '../training/concepts/pretraining/captions.js';
import * as N from '../training/concepts/pretraining/numbers.js';
import {
  INITIAL_STATE, PRESETS, checkWork, formatProb, formatLoss, formatPerplexity, selectPosition, setProb, allUniform, resetProbs, targetOf,
} from '../training/concepts/pretraining/format.js';
import { toyView, tryThis } from '../training/concepts/pretraining/toy-view.js';
import { factRows, FRAMING, tokenRange, stageText, BELOW } from '../training/concepts/pretraining/facts.js';
import { keyStep } from '../training/concepts/pretraining/stage-select.js';
import { CAPTIONS, CHECK_WORK, DEFAULT_LOSSES, TRY_THIS } from './pretraining-expected.js';

const read = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const data = { models: await read('../data/models.json'), hardware: await read('../data/hardware.json'), serving: await read('../data/serving.json'), papers: await read('../data/papers.json') };
const graph = await read('../shared/concepts.json');
const model = (id, key) => data.models.entries.find((e) => e.id === id).facts[key].value;
const near = (a, b, eps = 5e-4) => assert.ok(Math.abs(a - b) < eps, `${a} ≉ ${b}`);
const filled = lessonFor(data);

test('the lesson spec is complete, with and without data', () => {
  assert.deepEqual(validateLessonSpec(LESSON), []);
  assert.deepEqual(validateLessonSpec(filled), []);
});

test('captions are the storyboard\'s, verbatim and in order', () => {
  assert.deepEqual(LESSON.animation.steps.map((s) => s.caption), CAPTIONS);
  assert.deepEqual([...PAGE_CAPTIONS], CAPTIONS);
});

test('the numbers the captions repeat match the data and the math (README lesson 29)', () => {
  assert.match(CAPTIONS[0], new RegExp(`vocabulary has ${model('kimi-k3', 'vocab_size').toLocaleString('en-US')} pieces`));
  assert.match(CAPTIONS[1], new RegExp(`got ${N.P_ON.toFixed(3)}\\.`));
  const { lo, hi } = tokenRange(data);
  assert.match(CAPTIONS[9], new RegExp(`on ${lo / 1e12} to ${hi / 1e12} trillion tokens`));
  assert.match(CAPTIONS[4], /about 2\.9 here/);
  assert.equal(perplexity(meanLoss(N.STAND_INS)).toFixed(1), '2.9');
});

test('12 facts rows, every placeholder resolves against data/*.json, nothing unfilled', () => {
  const rows = factRows(data);
  assert.equal(rows.length, 12);
  assert.equal(LESSON.facts.rows.length, 12);
  rows.forEach((row, i) => {
    const f = fillClaim(row.claim, data);
    assert.deepEqual(f.missing, [], `row ${i + 1}`);
    assert.ok(f.sources.length > 0, `row ${i + 1} has a source`);
    assert.doesNotMatch(f.segments.map((s) => s.text).join(''), /—/, `row ${i + 1}`);
  });
  const text = (i) => fillClaim(rows[i].claim, data).segments.map((s) => s.text).join('');
  assert.equal(text(0), 'DeepSeek-V4-Pro: 33T pretraining tokens.');
  assert.equal(text(1), 'DeepSeek-V4-Flash 32T · MiniMax-M2 29.2T · GLM-5 28.5T (27T base + mid-training) · MiMo-V2-Flash 27T · Nemotron 3 Super 25T.');
  assert.match(text(3), /^Llama 3\.1 405B \(2024\): 15\.6T tokens/);
  assert.match(text(4), /DeepSeek-V4 129,280 · GLM-5 151,552–154,880 .* · Kimi K3 163,840 · MiniMax-M2 200,064 · gpt-oss ~201K/);
  assert.match(text(5), /163,840 × 7,168 ≈ 1\.17B parameters/);
  assert.match(text(10), /about 10M issue–PR pairs \(about 160B unique tokens\)/);
  assert.match(text(11), /Qwen3: 36T tokens over 119 languages; Mistral Large 4: 160\+ languages/);
  assert.equal(fillClaim(rows[11].claim, data).reported, true, 'Qwen3\'s token count is reported');
  assert.equal(fillClaim(rows[0].claim, data).reported, false);
});

test('dated prose fills from data: hook, intuition, framing, notes, math notes, takeaways, stage text', () => {
  const texts = [filled.hook, ...filled.intuition, filled.facts.framing, ...filled.math.notes, ...filled.takeaways, filled.animation.standIn,
    ...BELOW.flatMap((_, i) => filled.animation.belowFor(i))];
  texts.forEach((t) => assert.doesNotMatch(t, /[{}]|—/, t.slice(0, 60)));
  [...BELOW.flat(), ...INTUITION, ...MATH_NOTES].forEach((t) => assert.deepEqual(fillClaim(t, data).missing, [], t.slice(0, 60)));
  assert.match(filled.intuition[2], /read 25–33 trillion tokens this way \(Kimi K3 did not say; smaller open models read far less, Olmo 3 about 5\.9T, reported\)/);
  assert.match(filled.takeaways[2], /read 25–33 trillion tokens;/);
  assert.match(filled.math.notes[0], /163,840 in Kimi K3/);
  assert.match(FRAMING, /next-token cross-entropy/);
  assert.equal(filled.animation.belowFor(99).length, 0);
});

test('stage text is computed from the data (frames 1, 8 and 10)', () => {
  const s = stageText(data);
  assert.equal(s.vocab, 'vocabulary: 163,840 pieces (Kimi K3)');
  assert.equal(s.vocabRange, 'vocabularies 129K–201K (DeepSeek-V4 to gpt-oss)');
  assert.equal(s.embedding, 'embedding table ≈ 163,840 × 7,168 ≈ 1.17B params');
  assert.equal(formatCount(model('kimi-k3', 'vocab_size') * model('kimi-k3', 'd_model')), '1.17B');
  assert.equal(s.glm, 'GLM-5: ~10M issue–PR pairs, ~160B tokens, for software engineering');
  assert.deepEqual(s.budget.map((r) => [r.name, r.tokens, r.dim]), [
    ['DeepSeek-V4-Pro', '33T', false], ['DeepSeek-V4-Flash', '32T', false], ['MiniMax-M2', '29.2T', false], ['GLM-5', '28.5T', false],
    ['MiMo-V2-Flash', '27T', false], ['Nemotron 3 Super', '25T', false], ['Kimi K3', 'not disclosed', false], ['Llama 3.1 405B (2024)', '15.6T', true]]);
  assert.equal(s.range, 'range 25T–33T');
  const empty = stageText(null);
  assert.equal(empty.vocab, 'vocabulary: — pieces (Kimi K3)');
  assert.equal(empty.range, 'range —');
});

test('the math panel: hl-p links, and the DeepSeek-V4 MTP weights the tex restates equal the data (ruling P3-R13)', () => {
  const tex = LESSON.math.blocks.map((b) => b.tex).join('\n');
  assert.match(tex, /\\htmlClass\{hl-p\}/);
  const [, depth, from, to] = model('deepseek-v4-pro', 'mtp_loss').match(/depth (\d+), weight ([\d.]+) → ([\d.]+)/);
  assert.match(tex, new RegExp(`depth ${depth}\\},\\\\ \\\\lambda = ${from} \\\\to ${to}`));
});

test('Next lists exactly the lessons that take this one as a prereq (README lesson 1)', () => {
  const dependents = graph.concepts.filter((c) => c.prereqs.includes('pretraining')).map((c) => c.slug);
  assert.deepEqual([...LESSON.links.next].sort(), dependents.sort());
});

test('stand-in numbers: decoder-anatomy\'s softmax row, the seven stand-ins and their losses (storyboard §4, §5, §11)', () => {
  [0.390, 0.237, 0.087, 0.053, 0.019].forEach((p, i) => near(N.PROB_CELLS[i], p));
  assert.equal(N.P_ON, softmax(N.LOGITS)[N.LOGIT_WORDS.indexOf('on')]);
  near(N.PROB_TOTAL, 1, 1e-12);
  assert.equal(N.OTHERS, 12);
  assert.deepEqual(N.STAND_INS, [0.10, 0.25, 0.30, N.P_ON, 0.60, 0.45, 0.80]);
  assert.deepEqual(N.STAND_INS.map((p) => formatLoss(tokenLoss(p))), DEFAULT_LOSSES);
  assert.equal(N.LOSS_MAX_ABS, tokenLoss(0.01));
  assert.equal(N.LOSS_MAX_ABS.toFixed(3), '4.605');
  assert.deepEqual(N.REFERENCE_PS.map((p) => formatLoss(tokenLoss(p))), ['0.010', '4.605']);
  assert.equal(formatLoss(tokenLoss(0.3903)), '0.941');
  assert.equal(formatLoss(uniformLoss(N.UNIFORM_VOCAB)), '2.773');
  assert.equal(N.TOKENS.length, 8);
});

test('number formats: probabilities, losses, perplexities', () => {
  assert.deepEqual([0.45, 0.1, 1, 0.0625, N.P_ON, 0.01, 0.3].map(formatProb), ['0.45', '0.10', '1.00', '0.0625', '0.3903', '0.01', '0.30']);
  assert.deepEqual([0, 0.79851, 12.00665].map(formatLoss), ['0.000', '0.799', '12.007']);
  assert.deepEqual([2.8643, 16.0000001, 4.9339].map(formatPerplexity), ['2.86', '16.00', '4.93']);
  assert.throws(() => formatProb(0), RangeError);
  assert.throws(() => formatProb(1.2), RangeError);
});

test('"Check my work" for the default state is the storyboard text; other states fill the same template', () => {
  assert.equal(checkWork(INITIAL_STATE), CHECK_WORK);
  const on = checkWork(selectPosition(INITIAL_STATE, 4)).split('\n');
  assert.equal(on[0], 'selected: on   −ln 0.3903 = 0.9410');
  const dot = checkWork(selectPosition(INITIAL_STATE, 7)).split('\n');
  assert.equal(dot[0], 'selected: "."   −ln 0.80 = 0.2231');
  const uniform = checkWork(allUniform(INITIAL_STATE)).split('\n');
  assert.equal(uniform[1], `sum   = ${Array(7).fill('2.7726').join(' + ')} = 19.4081`);
  assert.equal(uniform[2], 'mean  = 19.4081 ÷ 7 = 2.7726 → 2.773');
  assert.equal(uniform[3], 'perplexity = e^2.7726 = 16.000 → 16.00');
  assert.equal(checkWork(setProb(INITIAL_STATE, 1)).split('\n')[0], 'selected: mat   −ln 1.00 = 0.0000');
});

test('toy view: the default state prints the storyboard\'s numbers, each from math/lm.js', () => {
  const v = toyView(INITIAL_STATE, data);
  assert.deepEqual(v.strip.map((c) => c.lossText), DEFAULT_LOSSES);
  assert.deepEqual(v.strip.map((c) => c.word), ['cat', 'sat', 'down', 'on', 'the', 'mat', '.']);
  assert.deepEqual(v.strip.map((c) => c.chip), [2, 3, 4, 5, 6, 7, 8]);
  assert.deepEqual(v.strip.map((c) => c.label), ['position 2: cat', 'position 3: sat', 'position 4: down', 'position 5: on', 'position 6: the', 'position 7: mat', 'position 8: .']);
  assert.deepEqual(v.strip.map((c) => c.pText), ['0.10', '0.25', '0.30', '0.3903', '0.60', '0.45', '0.80']);
  assert.deepEqual(v.strip.map((c) => c.selected), [false, false, false, false, false, true, false]);
  v.strip.forEach((c, i) => assert.equal(c.loss, tokenLoss(N.STAND_INS[i])));
  assert.deepEqual([v.selected.word, v.selected.pText, v.selected.lossText], ['mat', '0.45', '0.799']);
  assert.equal(v.mean, formatLoss(meanLoss(N.STAND_INS)));
  assert.equal(v.mean, '1.052');
  assert.equal(v.perplexity, formatPerplexity(perplexity(meanLoss(N.STAND_INS))));
  assert.equal(v.perplexity, '2.86');
  assert.equal(v.ref16, formatLoss(uniformLoss(16)));
  assert.equal(v.refKimi, formatLoss(uniformLoss(model('kimi-k3', 'vocab_size'))));
  assert.deepEqual([v.ref16, v.refKimi, v.kimiVocab], ['2.773', '12.007', '163,840']);
  assert.equal(v.checkWork, CHECK_WORK);
  assert.equal(toyView(INITIAL_STATE, null).refKimi, '—');
});

test('toy view: try this 1–3 at the storyboard\'s values', () => {
  const certain = toyView(setProb(INITIAL_STATE, 1), data);
  assert.deepEqual([certain.mean, certain.selected.lossText], ['0.938', '0.000']);
  const miss = toyView(setProb(INITIAL_STATE, 0.01), data);
  assert.deepEqual([miss.selected.lossText, miss.mean, miss.perplexity], ['4.605', '1.596', '4.93']);
  assert.equal((meanLoss(setProb(INITIAL_STATE, 0.01).probs) - meanLoss(N.STAND_INS)).toFixed(3), '0.544');
  assert.equal((meanLoss(N.STAND_INS) - meanLoss(setProb(INITIAL_STATE, 1).probs)).toFixed(3), '0.114');
  const on = toyView(selectPosition(INITIAL_STATE, 4), data);
  assert.deepEqual([on.selected.word, on.selected.pText, on.selected.lossText, on.mean], ['on', '0.3903', '0.941', '1.052']);
  const uniform = toyView(allUniform(INITIAL_STATE), data);
  assert.deepEqual(uniform.strip.map((c) => c.lossText), Array(7).fill('2.773'));
  assert.deepEqual([uniform.mean, uniform.perplexity], ['2.773', '16.00']);
  assert.equal(uniformLoss(model('kimi-k3', 'vocab_size')).toFixed(2), '12.01');
});

test('the try-this list is the storyboard\'s, with its numbers computed and Kimi K3\'s vocabulary from the data', () => {
  assert.deepEqual(tryThis(data), TRY_THIS);
  assert.match(tryThis(null)[2].prompt, /vocabulary of — pieces would start near —\./);
});

test('state transitions: select, slide, presets, all uniform, reset; inputs never mutated', () => {
  const before = JSON.stringify(INITIAL_STATE);
  const s1 = selectPosition(INITIAL_STATE, 2);
  assert.equal(s1.pos, 2);
  assert.equal(s1.probs, INITIAL_STATE.probs, 'selecting keeps the probabilities (and the exact stand-in)');
  const s2 = setProb(s1, 0.5);
  assert.deepEqual(s2.probs, [0.10, 0.5, 0.30, N.P_ON, 0.60, 0.45, 0.80]);
  assert.deepEqual(PRESETS.map((p) => [p.label, p.p]), [['confident miss 0.01', 0.01], ['uniform 1/16', 0.0625], ['certain 1.00', 1]]);
  assert.deepEqual(allUniform(s2).probs, Array(7).fill(0.0625));
  assert.deepEqual(resetProbs(allUniform(s2)), { pos: 2, probs: N.STAND_INS });
  assert.equal(targetOf(6), 7);
  assert.throws(() => selectPosition(INITIAL_STATE, 8), RangeError);
  assert.throws(() => selectPosition(INITIAL_STATE, 0), RangeError);
  assert.throws(() => setProb(INITIAL_STATE, 0), RangeError);
  assert.throws(() => setProb(INITIAL_STATE, 1.01), RangeError);
  assert.equal(JSON.stringify(INITIAL_STATE), before);
  assert.ok(Object.isFrozen(INITIAL_STATE) && Object.isFrozen(INITIAL_STATE.probs));
});

test('stage selection keys (P3-R17): arrows move one, Home / End jump, the ends hold, other keys do nothing', () => {
  assert.equal(keyStep('ArrowRight', 2, 7), 3);
  assert.equal(keyStep('ArrowDown', 2, 7), 3);
  assert.equal(keyStep('ArrowLeft', 2, 7), 1);
  assert.equal(keyStep('ArrowUp', 0, 7), 0);
  assert.equal(keyStep('ArrowRight', 6, 7), 6);
  assert.equal(keyStep('Home', 4, 7), 0);
  assert.equal(keyStep('End', 1, 7), 6);
  assert.equal(keyStep('a', 3, 7), null);
});
