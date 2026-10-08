import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateLessonSpec } from '../shared/lesson-spec.js';
import { fillClaim } from '../shared/claims.js';
import { paramBreakdown, PRESETS } from '../math/params.js';
import { LESSON, lessonFor } from '../architecture/concepts/decoder-anatomy/content.js';
import { checkWork, gapLine, signedPct, toyConfig, perBlockLine, barParts } from '../architecture/concepts/decoder-anatomy/format.js';
import { toyView, kimiLine, v3GapLine, INITIAL_STATE, EXPERTS_EDGE_NOTE } from '../architecture/concepts/decoder-anatomy/toy-view.js';
import { BELOW, factRows, framing } from '../architecture/concepts/decoder-anatomy/facts.js';
import * as S from '../architecture/concepts/decoder-anatomy/stream.js';
import { CAPTIONS, CHECK_WORK } from './decoder-anatomy-expected.js';

const read = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const data = { models: await read('../data/models.json'), hardware: await read('../data/hardware.json') };
const graph = await read('../shared/concepts.json');
const model = (id, key) => data.models.entries.find((e) => e.id === id).facts[key].value;
const near = (a, b, eps = 5e-4) => assert.ok(Math.abs(a - b) < eps, `${a} ≉ ${b}`);

test('the lesson spec is complete', () => assert.deepEqual(validateLessonSpec(LESSON), []));
test('the lesson filled with real data is complete too', () => assert.deepEqual(validateLessonSpec(lessonFor(data)), []));
test('captions are the storyboard\'s, verbatim and in order', () => assert.deepEqual(LESSON.animation.steps.map((s) => s.caption), CAPTIONS));
test('13 facts rows, every placeholder resolves against data/models.json', () => {
  assert.equal(LESSON.facts.rows.length, 13);
  LESSON.facts.rows.forEach((row, i) => assert.deepEqual(fillClaim(row.claim, data).missing, [], `row ${i + 1}`));
  lessonFor(data).facts.rows.forEach((row, i) => assert.doesNotMatch(row.claim, /—/, `row ${i + 1} has an unfilled derived number`));
});
test('Next lists exactly the lessons that take this one as a prereq (README lesson 1)', () => {
  const dependents = graph.concepts.filter((c) => c.prereqs.includes('decoder-anatomy')).map((c) => c.slug);
  assert.deepEqual([...LESSON.links.next].sort(), dependents.sort());
});
test('"Check my work" for the toy preset is the storyboard text', () => assert.equal(checkWork(PRESETS.toy, paramBreakdown(PRESETS.toy)), CHECK_WORK));
test('gap lines and signed percentages', () => {
  assert.equal(gapLine(174_604_259_328, 175e9), 'computed 174.6B vs published 175B (−0.23%)');
  assert.equal(gapLine(116_829_149_760, 116.83e9), 'computed 116.83B vs published 116.83B (matches)');
  assert.equal(gapLine(671_026_404_352, 671e9), 'computed 671.03B vs published 671B (matches)');
  assert.equal(signedPct(0.00004), '+0.00%');
  assert.equal(gapLine(100_000_040_000, 100e9), 'computed 100B vs published 100B (matches)');
  assert.equal(gapLine(99_999_990_000, 100e9), 'computed 100B vs published 100B (matches)', 'a tiny negative gap is not "−0.00%"');
  assert.equal(signedPct(0.0123), '+1.23%');
});

test('dated text: hook, intuition and the notes under the stage fill from data, nothing missing', () => {
  const lesson = lessonFor(data);
  const texts = [lesson.hook, ...lesson.intuition, ...BELOW.flatMap((_, i) => lesson.animation.belowFor(i))];
  texts.forEach((t) => assert.doesNotMatch(t, /[{}—]/, t.slice(0, 60)));
  BELOW.flat().forEach((t) => assert.deepEqual(fillClaim(t, data).missing, [], t.slice(0, 60)));
  assert.match(lesson.hook, /1\.6T parameters/);
  assert.match(lesson.animation.belowFor(8)[0], /4,718,592 B ≈ 4\.72 MB; DeepSeek-V3 \(2024\) 70,272 B ≈ 70\.3 kB/);
  assert.match(lesson.animation.belowFor(6)[0], /toy 2 · GPT-3 \(2020\) 96 · gpt-oss-120b \(2025\) 36 · DeepSeek-V4-Pro \(2026\) 61 · Kimi K3 \(2026\) 93/);
  assert.equal(lesson.animation.belowFor(42).length, 0);
});

test('the text the data cannot fill matches the data it repeats (README lesson 29)', () => {
  assert.match(CAPTIONS[6], new RegExp(`Layers: ${model('deepseek-v4-pro', 'layers')}"`));
  assert.equal(model('kimi-k3', 'd_model'), model('deepseek-v4-pro', 'd_model'), 'intuition ¶1 names both models for one d_model');
  assert.equal(model('minimax-m3', 'patch_size'), model('kimi-k3', 'patch_size'), 'frame 1 note names both models for one patch size');
});

test('framing and the derived 2026-norm row are computed from the table\'s entries', () => {
  assert.equal(framing(data).match(/differ by (\S+)×/)[1], '19–33');
  assert.match(framing(data), /the stack is 36–93 blocks deep/);
  const norm = factRows(data)[10];
  assert.equal(norm.derived, true);
  assert.match(norm.claim, /^2026 norm: about 3–5% of parameters active per token \(the table's entries run 3\.1–5\.4%\)/);
  assert.match(factRows(data)[1].claim, /\(4\.4%\)/);
  assert.match(factRows(data)[3].claim, /\(3\.1%\)/);
  assert.match(factRows(data)[2].claim, /checkpoint is 685B/);
  assert.match(factRows(data)[0].claim, /our count 174\.6B\.$/);
  assert.match(factRows(null)[1].claim, /\(—\)/, 'without data a ratio prints a dash, never a stale number');
});

test('stand-in numbers: the adds, the attention row and the softmax match storyboard §4–§5', () => {
  assert.deepEqual(S.X1_SAT, [0.25, 1.5, 0.25, 0.5, -0.5, 0.5, 0.25, 0.5]);
  assert.deepEqual(S.X2_SAT, [0.25, 1.75, 0.25, 0, 0, 0.5, 0, 0.75]);
  [0.095, 0.703, 0.202, 0].forEach((w, i) => near(S.WEIGHTS_SAT[i], w));
  [0.390, 0.237, 0.087, 0.053, 0.019].forEach((p, i) => near(S.PROB_CELLS[i], p));
  assert.deepEqual(S.SCORE_CELLS, [2, 1.5, 0.5, 0, -1]);
  near(S.PROB_TOTAL, 1, 1e-12);
  near(S.OTHERS * S.PROB_CELLS[4], 0.233);
  assert.equal(S.E.length, 16);
  assert.deepEqual(S.E.slice(0, 4), S.X);
  assert.ok(S.E.flat().every((v) => Number.isInteger(v * 2) && !Object.is(v, -0)), 'E sits on the half grid, no −0');
  assert.throws(() => S.addRows([1], [1, 2]), RangeError);
});

test('the math panel\'s worked numbers come from the stand-ins', () => {
  const blocks = LESSON.math.blocks.map((b) => b.tex).join('\n');
  const rms = Math.sqrt(S.X[S.SAT].reduce((s, v) => s + v * v, 0) / S.D_MODEL);
  assert.match(blocks, new RegExp(`\\\\sqrt\\{${S.X[S.SAT].reduce((s, v) => s + v * v, 0)}/8\\} = ${rms.toFixed(3)}`));
  const sum = S.LOGITS.reduce((s, z) => s + Math.exp(z), 0);
  assert.match(blocks, new RegExp(`\\{${sum.toFixed(2)}\\} = ${S.PROB_CELLS[0].toFixed(3)}`));
  ['x', 'a', 'm', 'z', 'p'].forEach((l) => assert.match(blocks, new RegExp(`\\\\htmlClass\\{hl-${l}\\}`)));
});

test('toyConfig follows the sliders and rejects off-stop values', () => {
  const c = toyConfig(PRESETS.toy, { layers: 8, dModel: 64, experts: 8 });
  assert.equal(c.attention.dHead, 32);
  assert.equal(c.mlp.hidden, 128);
  assert.deepEqual(c.moe, { routed: 8, shared: 0, topK: 2, hidden: 64, denseLayers: 0 });
  assert.equal(toyConfig(PRESETS.toy, { layers: 2, dModel: 8, experts: 0 }).moe, null);
  assert.throws(() => toyConfig(PRESETS.toy, { layers: 9, dModel: 8, experts: 0 }), RangeError);
  assert.throws(() => toyConfig(PRESETS.toy, { layers: 2, dModel: 12, experts: 0 }), RangeError);
  assert.throws(() => toyConfig(PRESETS.toy, { layers: 2, dModel: 8, experts: 3 }), RangeError);
});

test('check my work fills the same template for experts and for one block', () => {
  const moe = toyConfig(PRESETS.toy, { layers: 2, dModel: 8, experts: 8 });
  const text = checkWork(moe, paramBreakdown(moe)).split('\n');
  assert.equal(text[2], '  experts     8 × (W_in, W_gate [8 × 8], W_out [8 × 8])   8 × 3 × 8 × 8   = 1,536');
  assert.equal(text[3], '  router      W_router [8 × 8]          8 × 8                =    64');
  assert.match(text.at(-2), /^total {56}= 4,008$/);
  assert.match(text.at(-1), /^active = total − unused experts − embedding lookup {11}= 1,576$/);
  const one = toyConfig(PRESETS.toy, { layers: 1, dModel: 8, experts: 0 });
  assert.match(checkWork(one, paramBreakdown(one)), /\n× 1 block {52}=   656\n/);
  const wide = toyConfig(PRESETS.toy, { layers: 8, dModel: 64, experts: 0 });
  assert.match(checkWork(wide, paramBreakdown(wide)), /\ntotal {56}= 330,816\n/);
});

test('per-block line and share-bar parts', () => {
  const moe = toyConfig(PRESETS.toy, { layers: 2, dModel: 8, experts: 8 });
  assert.equal(perBlockLine(moe, paramBreakdown(moe)), 'attention 256 · expert 192 × 8 · norms 16');
  assert.equal(perBlockLine(PRESETS.deepseekV3, paramBreakdown(PRESETS.deepseekV3), { exact: false }), 'attention 187M · MLP 396M · expert 44M × 257 · norms 14.3K');
  assert.deepEqual(barParts(paramBreakdown(PRESETS.toy).parts).map((p) => [p.name, p.value, p.hue]), [
    ['embedding', 128, 1], ['attention', 512, 2], ['MLP', 768, 3], ['other (router, norms)', 40, 4], ['head', 128, 5]]);
  assert.equal(barParts(paramBreakdown(PRESETS.gpt3).parts).some((p) => p.name === 'head'), false, 'tied GPT-3 has no separate head');
  assert.equal(barParts(paramBreakdown(PRESETS.deepseekV3).parts)[2].name, 'MLP + experts');
  assert.deepEqual(barParts({ experts: 3, unknown: 1 }).map((p) => [p.name, p.unknown ?? false]), [['experts', false], ['not published', true]]);
});

test('toy view: the default state prints the animation\'s numbers', () => {
  const v = toyView(INITIAL_STATE, data);
  assert.deepEqual([v.total, v.active, v.activeWithEmbedding, v.activePct], ['1,576', '1,448', '1,576', '91.9% of total']);
  assert.equal(v.perBlock, 'attention 256 · MLP 384 · norms 16');
  assert.equal(v.activeNote, 'Active 1,448 of 1,576: active leaves out the 128-parameter embedding table.');
  assert.equal(v.checkWork, CHECK_WORK);
  const row = (part) => v.rows.find((r) => r.part === part);
  assert.deepEqual([row('mlp').share, row('attention').share, row('embedding').share, row('norms').share], ['48.7%', '32.5%', '8.1%', '2.5%']);
  assert.deepEqual([row('embedding').activeShare, row('positional').share, row('positional').count], ['—', '—', '0']);
  assert.equal(v.expertsNote, '');
  assert.match(v.spec, /vocabulary 16, 2 heads/);
});

test('toy view: real presets', () => {
  const gpt3 = toyView({ ...INITIAL_STATE, preset: 'gpt3' }, data);
  assert.equal(gpt3.gap, 'computed 174.6B vs published 175B (−0.23%)');
  assert.match(gpt3.activeNote, /tied/);
  assert.equal(gpt3.rows.find((r) => r.part === 'embedding').share, '0.35%');
  const oss = toyView({ ...INITIAL_STATE, preset: 'gptOss120b' }, data);
  assert.deepEqual([oss.active, oss.gap], ['5.13B', 'computed 116.83B vs published 116.83B (matches)']);
  assert.equal(oss.activeGap, 'active: computed 5.1328B vs published 5.13B (+0.06%)');
  assert.equal(oss.rows.find((r) => r.part === 'head').activeShare, '11.3%');
  assert.equal(oss.rows.find((r) => r.part === 'attention').share, '0.82%');
  assert.match(oss.spec, /^gpt-oss-120b: vocab 201,088 · d_model 2,880 · 36 blocks · 64 heads × 64 \(8 KV heads\) · 128 experts of hidden 2,880, top-4/);
  const v3 = toyView({ ...INITIAL_STATE, preset: 'deepseekV3' }, data);
  assert.equal(v3.gap, 'computed 671.03B vs published 671B (matches)');
  assert.equal(v3.rows.find((r) => r.part === 'experts').share, '97.8%');
  assert.throws(() => toyView({ ...INITIAL_STATE, preset: 'nope' }, data), RangeError);
});

test('toy view: DeepSeek-V4-Pro prints its published total and active back, with a "not published" remainder', () => {
  const v = toyView({ ...INITIAL_STATE, preset: 'deepseekV4Pro' }, data);
  assert.deepEqual([v.total, v.active, v.activePct], ['1.6T', '49B', '3.1% of total']);
  assert.deepEqual(v.rows.map((r) => [r.part, r.share, r.activeShare]), [['experts', '97.0%', '57.6%'], ['unknown', '3.0%', '42.4%']]);
  assert.equal(v.perBlock, 'expert 66.1M × 385 · the rest is not published');
  assert.match(v.activeNote, /28\.2B of it is derivable \(7 experts × 61 blocks\)/);
  assert.equal(v.bar.at(-1).unknown, true);
  assert.match(v.spec, /384 \+ 1 experts of hidden 3,072, 6 \+ 1 used per token/);
});

test('toy view: the experts = 2 stop and the experts slider (try this 3)', () => {
  assert.equal(toyView({ ...INITIAL_STATE, experts: 2 }, data).expertsNote, EXPERTS_EDGE_NOTE);
  const eight = toyView({ ...INITIAL_STATE, experts: 8 }, data);
  const sixteen = toyView({ ...INITIAL_STATE, experts: 16 }, data);
  assert.deepEqual([eight.total, eight.active, sixteen.total, sixteen.active], ['4,008', '1,576', '7,208', '1,704']);
  assert.equal(toyView({ ...INITIAL_STATE, layers: 1 }, data).total, '920');
});

test('the Kimi K3 line and the DeepSeek-V3 gap line read the data', () => {
  assert.equal(kimiLine(data), 'Kimi K3 (2026): 2.35B of 2.78T, 0.08%');
  assert.match(v3GapLine(data), /checkpoint is 685B because it also ships the 14B multi-token-prediction \(MTP\) module; the paper's 671B/);
  assert.equal(kimiLine(null), '');
  assert.equal(v3GapLine(null), '');
});

test('nothing here mutates its inputs', () => {
  const state = Object.freeze({ ...INITIAL_STATE, experts: 8 });
  const before = JSON.stringify(PRESETS);
  toyView(state, data);
  toyConfig(PRESETS.toy, state);
  checkWork(PRESETS.toy, paramBreakdown(PRESETS.toy));
  assert.equal(JSON.stringify(PRESETS), before);
});
