import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { validateLessonSpec } from '../shared/lesson-spec.js';
import { fillClaim, fillText } from '../shared/claims.js';
import { LESSON, lessonFor } from '../architecture/concepts/moe/content.js';
import { belowFor, factRows, framing, hookFor, activeRange } from '../architecture/concepts/moe/facts.js';
import { toyView, tryThis, REAL_MODELS, INITIAL_STATE as VIEW_INITIAL } from '../architecture/concepts/moe/toy-view.js';
import { toyMoe, scoreText, combosText, combosApprox, biasText, picksText, gatesText, imbalanceText, routedText, routedShare, int, INITIAL_STATE, TOY_LIMITS } from '../architecture/concepts/moe/format.js';
import * as N from '../architecture/concepts/moe/numbers.js';
import { CAPTIONS } from './moe-expected.js';

const read = async (path) => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const data = { models: await read('../data/models.json'), hardware: await read('../data/hardware.json'), serving: await read('../data/serving.json'), papers: await read('../data/papers.json') };
const graph = await read('../shared/concepts.json');
const STATE = { ...INITIAL_STATE };
const view = (patch) => toyView({ ...STATE, ...patch }, data);

test('the lesson spec is complete, with and without data', () => {
  assert.deepEqual(validateLessonSpec(LESSON), []);
  assert.deepEqual(validateLessonSpec(lessonFor(data)), []);
});
test('captions are the storyboard\'s, verbatim and in order', () => assert.deepEqual(LESSON.animation.steps.map((s) => s.caption), CAPTIONS));
test('11 facts rows; every placeholder resolves and no filled row prints "—"', () => {
  const rows = factRows(data);
  assert.equal(LESSON.facts.rows.length, 11);
  rows.forEach((row, i) => {
    assert.deepEqual(fillClaim(row.claim, data).missing, [], `row ${i + 1}`);
    assert.ok(!fillText(row.claim, data).includes('—'), `row ${i + 1}: ${fillText(row.claim, data)}`);
  });
});
test('the rows print the storyboard numbers from the data', () => {
  const rows = factRows(data).map((r) => fillText(r.claim, data));
  assert.equal(rows[0], 'DeepSeek-V4-Pro (2026): 1.6T / 49B (3.1%), 384 routed + 1 shared, top-6, expert hidden 3,072.');
  assert.match(rows[1], /^Kimi K3: 2\.78T \/ 104\.2B \(3\.7%\), 896 routed \+ 2 shared, top-16; .*\(3,584, half the model width of 7,168\)/);
  assert.match(rows[2], /^Qwen3\.8 \(2026\): 2\.4T \/ 95B \(4\.0%\), 512 routed experts plus a shared one, top-10\.$/);
  assert.match(rows[3], /^GLM-5\.3: 753B \/ 40B \(5\.3%.*\), 256 routed \+ 1 shared, top-8\.$/);
  assert.match(rows[4], /^MiniMax-M3: about 428B \/ 23B \(5\.4%\), 128 routed \+ 1 shared, top-4, first 3 layers dense\.$/);
  assert.match(rows[5], /^gpt-oss-120b \(2025\): 116\.8B \/ 5\.13B \(4\.4%\), 128 experts, top-4, no shared expert\.$/);
  assert.match(rows[6], /^Mistral Large 4 \(preview, 2026\): 1\.05T total, 49B routed-active \(52B with embeddings\), a/);
  assert.match(rows[7], /^The trend: about 3–5% of parameters active per token in 2026 .*down from 9–28% in 2023–25 \(Mixtral 8x7B, Dec 2023, about 28%; Qwen3-235B, 2025, 9%\)\.$/);
  assert.match(rows[8], /DeepSeek-V4: sqrt-softplus; hash routing in first 3 MoE layers\.$/);
  assert.match(rows[9], /aux-loss-free bias \+ small sequence-level term; Kimi K3: Quantile Balancing instead\.$/);
  assert.equal(factRows(data).filter((r) => r.derived).length, 2);
});
test('Mistral and GLM rows carry the reported chip through their data; DeepSeek\'s does not', () => {
  const reported = factRows(data).map((r) => fillClaim(r.claim, data).reported);
  assert.deepEqual(reported, [false, false, false, true, false, false, true, false, false, false, false]);
});
test('Next lists exactly the lessons that take this one as a prereq (README lesson 1)', () => {
  const dependents = graph.concepts.filter((c) => c.prereqs.includes('moe')).map((c) => c.slug);
  assert.deepEqual([...LESSON.links.next].sort(), dependents.sort());
});
test('dated text: hook, intuition, framing and the notes under the stage fill from data, nothing missing', () => {
  const lesson = lessonFor(data);
  const below = belowFor(data);
  const texts = [lesson.hook, ...lesson.intuition, lesson.facts.framing, ...below.flat()];
  texts.forEach((t) => assert.deepEqual(fillClaim(t, data).missing, [], t.slice(0, 60)));
  [lesson.hook, ...lesson.intuition, lesson.facts.framing, ...below.flatMap((_, i) => lesson.animation.belowFor(i))].forEach((t) => assert.doesNotMatch(t, /[{}—]/, t.slice(0, 60)));
  assert.match(lesson.hook, /^DeepSeek-V4-Pro has 1\.6T parameters but uses only 49B for each token\. Who decides which 3% to use/);
  assert.match(lesson.intuition[1], /3–5% of the parameters active per token/);
  assert.match(lesson.facts.framing, /and 3–5% of its parameters active per token/);
  assert.equal(activeRange(data), '3–5');
  assert.equal(lesson.animation.belowFor(42).length, 0);
});
test('frame notes print the computed counts and combinations', () => {
  const lesson = lessonFor(data);
  assert.equal(lesson.animation.belowFor(4)[0], 'Whole toy (2 blocks): dense MLP 1,576 total / 1,448 active · 8 experts 4,008 total / 1,576 active · 16 experts 7,208 total / 1,704 active.');
  assert.equal(lesson.animation.belowFor(5)[0], 'DeepSeek-V4-Pro (2026): choosing 6 of 384 routed experts can be done in ≈ 4.3 × 10¹² ways (exactly 4,281,625,192,384).');
  assert.match(lesson.animation.belowFor(6)[0], /Shared experts in 2026: 1 in DeepSeek-V4-Pro, 2 in Kimi K3, 1 in GLM-5\.3, 1 in MiniMax-M3\.$/);
  assert.equal(lesson.animation.belowFor(0)[0], 'This opens the Mixture-of-Experts branch from frame 6 of [[decoder-anatomy]], with the same toy: 8 experts of hidden 8, top-2.');
  assert.match(lesson.animation.belowFor(2)[0], /^Softmax over the chosen two, as Mixtral and Qwen3 do; DeepSeek normalizes sigmoid scores instead\.$/);
  assert.match(lesson.animation.belowFor(3)[0], /not topics; this page makes no claim about what each expert learns\.$/);
});
test('hook and framing without data print a placeholder dash, never a typed number', () => {
  assert.match(fillText(hookFor(null), null), /—/);
  assert.match(framing(null), /—/);
});

// ---- numbers.js: the storyboard's worked numbers ----
test('picks, gates, loads and the biased choice are the storyboard\'s', () => {
  assert.deepEqual(N.PICKS, [[1, 4], [3, 5], [2, 5], [0, 6]]);
  assert.deepEqual(N.GATES.map((g) => g.map((x) => x.toFixed(3))), [['0.622', '0.378'], ['0.622', '0.378'], ['0.622', '0.378'], ['0.562', '0.438']]);
  assert.deepEqual(N.LOADS_FOUR, [1, 1, 1, 1, 1, 2, 1, 0]);
  assert.deepEqual(N.PICKS_BIASED, [2, 6]);
  assert.deepEqual(N.GATES_BIASED.map((x) => x.toFixed(3)), ['0.818', '0.182']);
  assert.deepEqual(N.CHOOSE_SAT, [0, -0.5, 2, 0.25, -1, 0.75, 1, 0]);
  assert.deepEqual(N.TOP1_SAT, [2]);
});
test('the fine-grained picks are four of 16 and differ from the halves of E3 and E6', () => {
  assert.equal(N.FINE_PICKS.length, 4);
  assert.deepEqual([...N.FINE_PICKS].sort((a, b) => a - b), [4, 5, 10, 13]);
});
test('the batch run: imbalance 3.00 → 1.13 over six steps, step 0 and 6 loads', () => {
  assert.deepEqual(N.RUN.map((r) => r.imbalance.toFixed(2)), ['3.00', '2.31', '2.03', '1.75', '1.44', '1.44', '1.13']);
  assert.deepEqual(N.RUN[0].loads, [96, 51, 12, 16, 25, 9, 21, 26]);
  assert.deepEqual(N.RUN[6].loads, [32, 35, 36, 33, 28, 28, 30, 34]);
  assert.equal(N.FAIR_SHARE, 32);
});

// ---- format.js ----
test('number formats: real minus, whole scores with one decimal, zero plain', () => {
  assert.deepEqual([0, 2, -1, 1.5, 0.25, -0.5, -0.75].map(scoreText), ['0', '2.0', '−1.0', '1.5', '0.25', '−0.5', '−0.75']);
  assert.deepEqual([-0.6000000000000001, 0.4, 0, -0.001].map(biasText), ['−0.60', '0.40', '0.00', '0.00']);
  assert.equal(imbalanceText(1.125), '1.13');
  assert.equal(int(-1576), '−1,576');
  assert.equal(picksText([2, 5]), 'E3, E6');
  assert.equal(gatesText([0.6224593, 0.3775407]), '0.622 · 0.378');
});
test('combination counts: exact below 10^15, then a power of ten', () => {
  assert.deepEqual([28n, 1820n, 10_518_300n, 4_281_625_192_384n].map(combosText), ['28', '1,820', '10,518,300', '4,281,625,192,384']);
  assert.equal(combosText(1_000_000_000_000_000n), '≈ 1.0 × 10¹⁵');
  assert.equal(combosText(409_663_695_276_000n), '409,663,695,276,000');
  assert.equal(combosApprox(4_281_625_192_384n), '≈ 4.3 × 10¹²');
  assert.equal(combosApprox(9_960_000_000_000n), '≈ 1.0 × 10¹³');
  assert.equal(combosApprox(10n ** 35n * 3n), '≈ 3.0 × 10³⁵');
  assert.throws(() => combosText(5), TypeError);
});
test('routed share text', () => {
  assert.equal(routedText(6, 384), '6 of 384');
  assert.equal(routedShare(6, 384), '1.6%');
});
test('toyMoe: the sliders give the storyboard\'s configurations and keep 384 active expert parameters', () => {
  assert.equal(toyMoe({ routed: 0, split: 1, shared: false }).dense, true);
  const cases = [
    [{ routed: 8, split: 1, shared: false }, { experts: 8, topK: 2, hidden: 8, shared: 0 }],
    [{ routed: 8, split: 2, shared: false }, { experts: 16, topK: 4, hidden: 4, shared: 0 }],
    [{ routed: 8, split: 4, shared: false }, { experts: 32, topK: 8, hidden: 2, shared: 0 }],
    [{ routed: 8, split: 1, shared: true }, { experts: 8, topK: 1, hidden: 8, shared: 1 }],
    [{ routed: 8, split: 2, shared: true }, { experts: 16, topK: 3, hidden: 4, shared: 1 }],
  ];
  cases.forEach(([state, want]) => {
    const m = toyMoe(state);
    assert.deepEqual({ experts: m.experts, topK: m.topK, hidden: m.hidden, shared: m.shared }, want);
    assert.equal((m.topK + m.shared) * 3 * 8 * m.hidden, 384);
  });
  assert.throws(() => toyMoe({ routed: 5, split: 1, shared: false }), /routed must be one of/);
  assert.throws(() => toyMoe({ routed: 8, split: 3, shared: false }), /split must be one of/);
});
test('the toy starts where the animation ends', () => {
  assert.deepEqual({ ...VIEW_INITIAL }, { real: 'toy', routed: 8, split: 1, shared: false, gamma: 0.1, step: 0 });
  assert.equal(TOY_LIMITS.routed.length, 6);
});

// ---- toy-view.js: every number the storyboard prints for the toy ----
test('panel A default: 4,008 / 1,576 / 39.3%, 384 per token, router 64, 28 combinations, the four words\' picks and gates', () => {
  const a = view({}).a;
  assert.deepEqual([a.total, a.active, a.activeShare, a.expertActive, a.router, a.combos], ['4,008', '1,576', '39.3%', '384', '64', '28']);
  assert.deepEqual(a.routing.rows.map((r) => [r.picks, r.gates]), [['E2, E5', '0.622 · 0.378'], ['E4, E6', '0.622 · 0.378'], ['E3, E6', '0.622 · 0.378'], ['E1, E7', '0.562 · 0.438']]);
  assert.deepEqual(a.routing.loads, [1, 1, 1, 1, 1, 2, 1, 0]);
  assert.equal(a.routingNote, '');
});
test('try this 1: routed 0 → 8 → 16', () => {
  const rows = [0, 8, 16].map((routed) => view({ routed }).a);
  assert.deepEqual(rows.map((a) => [a.total, a.active, a.router]), [['1,576', '1,448', '0'], ['4,008', '1,576', '64'], ['7,208', '1,704', '128']]);
  assert.equal(rows[0].combos, '—');
  assert.equal(rows[0].expertActive, '384');
  assert.equal(rows[2].routing, null);
  assert.match(rows[2].routingNote, /only for 8 experts and split 1/);
});
test('try this 2: split 2 and 4', () => {
  const a2 = view({ split: 2 }).a;
  const a4 = view({ split: 4 }).a;
  assert.deepEqual([a2.expertActive, a2.combos, a2.router, a2.total, a2.active], ['384', '1,820', '128', '4,136', '1,704']);
  assert.deepEqual([a4.expertActive, a4.combos, a4.router, a4.total, a4.active], ['384', '10,518,300', '256', '4,392', '1,960']);
});
test('a shared expert: 4,392 total, active 1,576, top-1 routing', () => {
  const a = view({ shared: true }).a;
  assert.deepEqual([a.total, a.active, a.expertActive], ['4,392', '1,576', '384']);
  assert.deepEqual(a.routing.rows.map((r) => r.gates), ['1.000', '1.000', '1.000', '1.000']);
  assert.match(a.spec, /plus 1 shared expert, top-1/);
});
test('two routed experts at top-2 use every expert: a dense MLP with a router', () => {
  assert.match(view({ routed: 2 }).a.edgeNote, /dense MLP with a router/);
  assert.equal(view({ routed: 2, shared: true }).a.edgeNote, '');
});
test('real chips print the published figures from the data', () => {
  const ds = view({ real: 'deepseek-v4-pro' });
  assert.equal(ds.real, true);
  assert.deepEqual([ds.a.total, ds.a.active, ds.a.share, ds.a.routed, ds.a.routedShare, ds.a.shared], ['1.6T', '49B', '3.1%', '6 of 384', '1.6%', '1']);
  assert.equal(ds.a.combos, '4,281,625,192,384');
  const rows = REAL_MODELS.map((m) => view({ real: m.value }).a);
  assert.deepEqual(rows.map((a) => a.share), ['3.1%', '3.7%', '4.0%', '5.3%', '5.4%', '4.4%']);
  assert.deepEqual(rows.map((a) => a.shared), ['1', '2', 'a shared expert', '1', '1', 'no shared expert']);
  assert.deepEqual(rows.map((a) => a.activeReported), [false, false, false, true, false, false]);
  assert.match(rows[1].combos, /^≈ /);
  assert.throws(() => toyView({ ...STATE, real: 'qwen3.8' }, { models: { entries: [] } }), /no experts/);
});
test('panel B: gamma 0.1 steps 0 and 6, gamma 0 step 9, gamma 0.2 steps 2 and 5', () => {
  const at = (gamma, step) => view({ gamma, step }).b;
  assert.deepEqual([at(0.1, 0).imbalance, at(0.1, 0).loadTexts.join(' ')], ['3.00', '96 51 12 16 25 9 21 26']);
  assert.equal(at(0.1, 6).imbalance, '1.13');
  assert.equal(at(0.1, 6).biasTexts.join(' '), '−0.60 −0.10 0.40 0.40 0.40 0.40 0.40 0.40');
  assert.equal(at(0, 9).imbalance, '2.91');
  assert.deepEqual([at(0.2, 2).imbalance, at(0.2, 5).imbalance], ['1.22', '1.84']);
  assert.equal(at(0.1, 6).fairShare, 32);
  assert.match(at(0.1, 0).note, /^Each step routes a fresh batch of 128 tokens, so the number wobbles; the bias keeps it near 1\.$/);
});
test('the active definition is the decoder-anatomy sentence, verbatim', () => {
  assert.equal(view({}).definition, 'Active = the parameters multiplied for one token: every block parameter except unused experts, plus the unembedding. The embedding table is left out: looking up a row is not a multiplication. (GPT-3\'s learned position table is a lookup too; it is so small, 0.01 %, that this course counts it as active.)');
});
test('try-this text fills DeepSeek-V4-Pro\'s numbers from the data', () => {
  const [one] = tryThis(data);
  assert.match(one.prompt, /Tap DeepSeek-V4-Pro: 1\.6T total, 49B active, 3\.1%, 6 of 384 routed experts per token \(1\.6%\)/);
  assert.equal(tryThis(data).length, 3);
  assert.match(tryThis(null)[0].prompt, /—/);
});
test('the toy and the page never mutate the data', () => {
  const before = JSON.stringify(data);
  lessonFor(data);
  view({ real: 'kimi-k3' });
  tryThis(data);
  assert.equal(JSON.stringify(data), before);
});
