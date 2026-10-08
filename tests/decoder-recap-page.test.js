import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateLessonSpec } from '../shared/lesson-spec.js';
import { fillClaim } from '../shared/claims.js';
import { paramBreakdown, PRESETS } from '../math/params.js';
import { LESSON, lessonFor } from '../architecture/concepts/decoder-recap/content.js';
import { CAPTIONS as PAGE_CAPTIONS } from '../architecture/concepts/decoder-recap/captions.js';
import { FACT_ROWS, belowTexts } from '../architecture/concepts/decoder-recap/facts.js';
import * as F from '../architecture/concepts/decoder-recap/format.js';
import * as N from '../architecture/concepts/decoder-recap/numbers.js';
import { X as ANATOMY_X } from '../architecture/concepts/decoder-anatomy/numbers.js';
import { CAPTIONS, FRAME5_EXACT, FRAME7_LINE, STAND_IN, RESIDUAL_NOTE } from './decoder-recap-expected.js';

const read = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const data = { models: await read('../data/models.json'), hardware: await read('../data/hardware.json') };
const graph = await read('../shared/concepts.json');
const near = (a, b, eps = 5e-4) => assert.ok(Math.abs(a - b) < eps, `${a} ≉ ${b}`);
const view = (patch = {}) => F.toyView({ ...F.INITIAL_STATE, ...patch });

test('the lesson spec is complete, with and without data', () => {
  assert.deepEqual(validateLessonSpec(LESSON), []);
  assert.deepEqual(validateLessonSpec(lessonFor(data)), []);
});
test('captions are the storyboard\'s, verbatim and in order', () => {
  assert.deepEqual(LESSON.animation.steps.map((s) => s.caption), CAPTIONS);
  assert.deepEqual([...PAGE_CAPTIONS], CAPTIONS);
});
test('every facts row resolves against data/models.json and none prints a dash', () => {
  assert.equal(FACT_ROWS.length, 12);
  FACT_ROWS.forEach((row, i) => assert.deepEqual(fillClaim(row.claim, data).missing, [], `row ${i + 1}`));
  FACT_ROWS.forEach((row, i) => assert.doesNotMatch(fillClaim(row.claim, data).segments.map((s) => s.text).join(''), /—/, `row ${i + 1}`));
  assert.equal(fillClaim(FACT_ROWS[0].claim, data).segments.map((s) => s.text).join(''), 'GPT-3 (2020): pre-norm LayerNorm, learned positions (2,048), a GELU MLP 4× wide, full multi-head attention (96 heads, each with its own keys and values), biases and tied embeddings.');
  assert.equal(fillClaim(FACT_ROWS[7].claim, data).reported, false);
  assert.equal(fillClaim(FACT_ROWS[0].claim, data).reported, true, 'GPT-3\'s learned positions are a reported fact');
});
test('the text under the stage fills from data, prints the storyboard\'s exact lines, and nothing is missing', () => {
  const lesson = lessonFor(data);
  belowTexts().flat().forEach((t) => assert.deepEqual(fillClaim(t, data).missing, [], t.slice(0, 50)));
  const below = (i) => lesson.animation.belowFor(i);
  assert.ok(below(4).includes(FRAME5_EXACT));
  assert.ok(below(6).includes(FRAME7_LINE));
  assert.match(below(6)[0], /\[−0\.792, 1\.841, 0\.396\] → \[0\.055, 0\.765, 0\.180\], unchanged at q × 10/);
  assert.match(below(6)[0], /plain scores ÷ 2: \[−0\.5, 1\.5, 0\.25\] · q × 10: \[−5, 15, 2\.5\] → \[0\.000, 1\.000, 0\.000\]/);
  assert.match(below(8)[0], /Qwen3\.8: 3 linear : 1 full · Kimi K3: 69 linear \+ 24 full · gpt-oss: window and full alternate \(1 full : 1 window\) · Qwen3\.5-397B: 15 of 60 full/);
  assert.match(below(2)[0], /2,048 positions/);
  assert.equal(below(1)[0], 'LayerNorm: [−0.630, 1.386, 0.378, −0.630, −1.638, 1.386, −0.630, 0.378] · RMSNorm: [0, 1.706, 0.853, 0, −0.853, 1.706, 0, 0.853]');
  assert.equal(below(7)[0], 'normalized scores: [−0.792, 1.841, 0.396], the row the sink joins.');
  assert.match(below(9)[1], new RegExp(RESIDUAL_NOTE.replace(/[()]/g, '\\$&')));
  assert.match(below(5)[0], /^2 × 96 × 96 × 128 × 2 B = 4,718,592 B → 2 × 96 × 8 × 128 × 2 B = 393,216 B\./);
  assert.equal(below(42).length, 0);
  assert.equal(lesson.animation.standIn, STAND_IN);
});
test('Next lists exactly the lessons that take this one as a prereq (README lesson 1)', () => {
  const dependents = graph.concepts.filter((c) => c.prereqs.includes('decoder-recap')).map((c) => c.slug);
  assert.deepEqual([...LESSON.links.next].sort(), dependents.sort());
});
test('stage cells print 2 d.p., the text under the stage 3 d.p. (template rule 7)', () => {
  assert.deepEqual(N.ROW_NORMS.layer.map(F.norm2), ['−0.63', '1.39', '0.38', '−0.63', '−1.64', '1.39', '−0.63', '0.38']);
  assert.deepEqual(N.ROW_NORMS.rms.map(F.norm2), ['0', '1.71', '0.85', '0', '−0.85', '1.71', '0', '0.85']);
  assert.deepEqual(N.QK.normed().map(F.fixed2), ['−0.79', '1.84', '0.40']);
  assert.deepEqual(N.X_SAT.map(F.plain), ['0', '1', '0.5', '0', '−0.5', '1', '0', '0.5']);
  assert.equal(F.fixed2(-0.001), '0.00');
});
test('the active definition names GPT-3\'s position table (README lesson 16)', () => {
  assert.match(F.ACTIVE_DEFINITION, /GPT-3's learned position table is a lookup too; it is so small, 0\.01 %, that this course counts it as active\.\)$/);
  assert.equal(Number((100 * N.GPT3_FACTS.positionTable / N.GPT3_FACTS.total).toFixed(2)), 0.01);
});
test('three takeaways, three further links, and every math term has a stage link', () => {
  assert.equal(LESSON.takeaways.length, 3);
  assert.equal(LESSON.links.further.length, 3);
  const tex = LESSON.math.blocks.map((b) => b.tex).join('\n');
  ['rms', 'glu', 'qk'].forEach((name) => assert.match(tex, new RegExp(`\\\\htmlClass\\{hl-${name}\\}`)));
});

test('the stand-in rows are the earlier pages\' (README lesson 29)', () => {
  assert.deepEqual([...N.X_SAT], ANATOMY_X[2], 'frame 2 uses decoder-anatomy\'s row "sat"');
  assert.deepEqual([...N.Q_SAT], [0, 2, 0.5, 0]);
  assert.match(LESSON.math.blocks[1].tex, new RegExp(`${N.ROW_NORMS.rootMeanSquare.toFixed(3)}`));
  assert.match(LESSON.math.blocks[1].tex, /\\sqrt\{2\.75\/8\}/);
});
test('frame 2 and frame 7 numbers match the storyboard', () => {
  [-0.630, 1.386, 0.378, -0.630, -1.638, 1.386, -0.630, 0.378].forEach((v, i) => near(N.ROW_NORMS.layer[i], v));
  [0, 1.706, 0.853, 0, -0.853, 1.706, 0, 0.853].forEach((v, i) => near(N.ROW_NORMS.rms[i], v));
  near(N.ROW_NORMS.mean, 0.3125); near(N.ROW_NORMS.spread, 0.496); near(N.ROW_NORMS.rootMeanSquare, 0.586);
  [0.095, 0.703, 0.202].forEach((v, i) => near(N.QK.plainWeights()[i], v));
  [0, 1, 0].forEach((v, i) => near(N.QK.plainWeights(10)[i], v));
  [0.055, 0.765, 0.180].forEach((v, i) => near(N.QK.normedWeights()[i], v));
  N.QK.normedWeights().forEach((v, i) => near(N.QK.normedWeights(10)[i], v, 1e-9));
  N.QK.normedWeights().forEach((v, i) => near(N.QK.normedWeights(100)[i], v, 1e-9));
});
test('GPT-3 and the modernized block: the stage counts', () => {
  assert.deepEqual([N.GPT3_FACTS.totalText, N.GPT3_FACTS.totalShort, N.GPT3_FACTS.cacheBytes, N.GPT3_FACTS.cacheText], ['174.6B', '175B', 4_718_592, '4.72 MB']);
  assert.deepEqual([N.MODERN_FACTS.stored, N.MODERN_FACTS.active, N.MODERN_FACTS.dense, N.MODERN_FACTS.cacheText, N.MODERN_FACTS.group], ['960B', '148B', '148B', '393 kB', 12]);
  assert.equal(N.GPT3_FACTS.positionTable, 25_165_824);
  assert.equal(N.MODERN_FACTS.router, 96 * 12288 * 64);
  assert.equal(N.LIT_EXPERTS.length, N.MODERN_FACTS.topK);
  assert.equal(new Set(N.LIT_EXPERTS).size, 8);
  assert.ok(N.LIT_EXPERTS.every((i) => i >= 0 && i < 64));
});

test('the switch states reproduce the storyboard\'s cumulative list (README lesson 16)', () => {
  const steps = [
    ['norm', 'rmsnorm', 174_601_887_744], ['position', 'rope', 174_576_721_920], ['mlp', 'swiglu', 174_578_294_784],
    ['biases', 'off', 174_566_105_088], ['kvHeads', 8, 147_990_994_944],
  ];
  assert.equal(F.breakdownFor(F.INITIAL_STATE).total, 174_604_259_328);
  const final = steps.reduce((state, [key, value, total]) => {
    const next = F.applySwitch(state, key, value);
    assert.equal(F.breakdownFor(next).total, total, key);
    return next;
  }, F.INITIAL_STATE);
  const moe = F.applySwitch(final, 'experts', 'moe');
  assert.deepEqual([F.breakdownFor(moe).total, F.breakdownFor(moe).active], [959_815_311_360, 148_066_492_416]);
  assert.deepEqual(moe, F.MODERN_STATE);
});
test('GELU 4d and SwiGLU 8/3·d MLPs are equal without biases at GPT-3\'s width; experts add exactly the router to active (lessons 16, 17)', () => {
  const noBias = { ...PRESETS.gpt3, biases: false };
  const gelu = paramBreakdown(noBias).perLayer.mlp;
  const glu = paramBreakdown({ ...noBias, mlp: { kind: 'swiglu', hidden: 32768 } }).perLayer.mlp;
  assert.equal(gelu, 1_207_959_552);
  assert.equal(glu, gelu);
  const dense = F.breakdownFor({ ...F.MODERN_STATE, experts: 'dense' });
  const moe = F.breakdownFor(F.MODERN_STATE);
  assert.equal(moe.active - dense.active, 96 * 12288 * 64);
});
test('experts need SwiGLU, and a bad switch is a RangeError', () => {
  assert.equal(F.applySwitch(F.INITIAL_STATE, 'experts', 'moe').mlp, 'swiglu');
  assert.equal(F.applySwitch(F.MODERN_STATE, 'mlp', 'gelu').experts, 'dense');
  assert.throws(() => F.applySwitch(F.INITIAL_STATE, 'colour', 'red'), RangeError);
  assert.throws(() => F.applySwitch(F.INITIAL_STATE, 'norm', 'batchnorm'), RangeError);
  assert.throws(() => F.configFor({ ...F.INITIAL_STATE, experts: 'moe' }), RangeError);
  assert.throws(() => F.configFor({ ...F.INITIAL_STATE, kvHeads: 7 }), RangeError);
});

test('toy view: the opening state is GPT-3', () => {
  const v = view();
  assert.deepEqual([v.total, v.active, v.change, v.changeSub, v.cache, v.cacheSub, v.longest], ['174,604,259,328', '174,604,259,328', '0', '0.00%', '4,718,592 B', '4.72 MB', '2,048 (table size)']);
  assert.equal(v.activeSub, '175B, 100% of total');
  const part = (name) => v.parts.find((p) => p.part === name);
  assert.deepEqual([part('mlp').count, part('positional').count, part('experts').count, part('embedding').count], ['115,970,015,232', '25,165,824', '0', '617,558,016']);
  assert.ok(v.parts.every((p) => p.delta === '0'));
});
test('toy view: try this 1 (norm, MLP)', () => {
  const rms = view({ norm: 'rmsnorm' });
  assert.deepEqual([rms.total, rms.change, rms.changeSub], ['174,601,887,744', '−2,371,584', '−0.0014%']);
  const glu = view({ mlp: 'swiglu' });
  assert.deepEqual([glu.parts.find((p) => p.part === 'mlp').count, glu.parts.find((p) => p.part === 'mlp').delta], ['115,971,588,096', '+1,572,864']);
});
test('toy view: try this 2 (RoPE, KV heads)', () => {
  const rope = view({ position: 'rope' });
  assert.deepEqual([rope.parts.find((p) => p.part === 'positional').delta, rope.longest], ['−25,165,824', 'set by training']);
  assert.deepEqual([view({ kvHeads: 8 }).cache, view({ kvHeads: 8 }).cacheSub], ['393,216 B', '393 kB (12× less than GPT-3)']);
  assert.deepEqual([view({ kvHeads: 1 }).cache, view({ kvHeads: 1 }).cacheSub], ['49,152 B', '49.2 kB (96× less than GPT-3)']);
});
test('toy view: try this 3 (experts)', () => {
  const dense = F.toyView({ ...F.MODERN_STATE, experts: 'dense' });
  const moe = F.toyView(F.MODERN_STATE);
  assert.deepEqual([dense.total, dense.active, moe.total, moe.active], ['147,990,994,944', '147,990,994,944', '959,815,311,360', '148,066,492,416']);
  const part = (v, name) => v.parts.find((p) => p.part === name);
  assert.deepEqual([part(moe, 'router').count, part(moe, 'experts').count, part(moe, 'mlp').count], ['75,497,472', '927,712,935,936', '0']);
  assert.equal(moe.changeSub, '+449.71%');
});
test('the try-this text reads its numbers back from the same functions', () => {
  const [one, two, three] = F.tryThis();
  assert.match(one.prompt, /^Flip norm: total 174,604,259,328 → 174,601,887,744 \(2,371,584 fewer: the β vectors\)\. Flip MLP: the MLP parts go 115,970,015,232 → 115,971,588,096 \(the bias vectors differ; without biases both are 96 × 1,207,959,552\)\.$/);
  assert.match(two.prompt, /25,165,824 fewer parameters.*cache 4,718,592 → 393,216 B per token \(12× less\)\. Turn biases off too, and the total is 147,990,994,944: W_K and W_V shrink\./);
  assert.match(three.prompt, /total 147,990,994,944 → 959,815,311,360; active 147,990,994,944 → 148,066,492,416 \(\+75,497,472: the router\)/);
  assert.match(two.rest, /\[\[kv-cache\]\].*\[\[kv-compression\]\]/);
});
test('formatters: real minus, leading zero, no signed zero', () => {
  assert.equal(F.fixed3(-0.792), '−0.792');
  assert.equal(F.fixed3(-0.0001), '0.000');
  assert.equal(F.norm3(0), '0');
  assert.equal(F.norm3(1.7056), '1.706');
  assert.equal(F.listText([-0.5, 1.5, 0.25]), '[−0.5, 1.5, 0.25]');
  assert.equal(F.listText([0, 1, 0], 3), '[0.000, 1.000, 0.000]');
  assert.equal(F.signedPct(0), '0.00%');
  assert.equal(F.signedPct(-0.0023), '−0.23%');
  assert.equal(F.signedInt(-5), '−5');
  assert.equal(F.signedInt(0), '0');
});
test('nothing here mutates its inputs', () => {
  const before = JSON.stringify([PRESETS, F.OPTIONS, F.INITIAL_STATE]);
  F.toyView(F.MODERN_STATE); F.configFor(F.INITIAL_STATE); F.tryThis(); F.applySwitch(F.INITIAL_STATE, 'experts', 'moe');
  assert.equal(JSON.stringify([PRESETS, F.OPTIONS, F.INITIAL_STATE]), before);
});
