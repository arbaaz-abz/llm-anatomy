import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { tpCommRatio, ppCommRatio, dpCommRatio, epCommRatio, epMinLinkGBps } from '../math/topology.js';

const H = 12288;
const pct = (x) => Number((100 * x).toFixed(1));
const round = (x, d) => Number(x.toFixed(d));
const close = (a, b, rel = 1e-12) => assert.ok(Math.abs(a - b) <= rel * Math.abs(b), `${a} ≉ ${b}`);
const tp = (t, peakTflops, linkGBps) => tpCommRatio({ tp: t, hidden: H, peakTflops, linkGBps });
const pp = (p, peakTflops, linkGBps) => ppCommRatio({ pp: p, hidden: H, layers: 96, peakTflops, linkGBps });
const dp = (n, tokensPerReplica, peakTflops, linkGBps) => dpCommRatio({ dp: n, tokensPerReplica, peakTflops, linkGBps });

// ---- §6 worked examples (GPT-3 shape, full step) ----

test('tpCommRatio: GPT-3 shape on H100 NVLink and network (§6 reproducer rows)', () => {
  assert.deepEqual([2, 4, 8, 16, 32, 64].map((t) => pct(tp(t, 989, 450))), [4.0, 11.9, 27.8, 59.6, 123.2, 250.4]);
  assert.deepEqual([2, 4, 8, 16, 32, 64].map((t) => pct(tp(t, 989, 50))), [35.8, 107.3, 250.4, 536.6, 1108.9, 2253.6]);
  assert.equal(round(tp(8, 989, 450), 4), 0.2782);
  assert.equal(round(tp(8, 989, 50), 4), 2.504);
  assert.equal(round(tp(16, 989, 50), 4), 5.3657);
});

test('tpCommRatio: GB200 NVL72 (2,500 TFLOPS, NVLink 900, network 100)', () => {
  assert.deepEqual([2, 4, 8, 16, 32, 64].map((t) => pct(tp(t, 2500, 900))), [5.0, 15.1, 35.2, 75.4, 155.7, 316.5]);
  assert.deepEqual([2, 4, 8, 16, 32, 64].map((t) => pct(tp(t, 2500, 100))), [45.2, 135.6, 316.5, 678.2, 1401.5, 2848.3]);
  assert.equal(round(tp(8, 2500, 900), 4), 0.3516);
  assert.equal(round(tp(16, 2500, 900), 4), 0.7535);
});

test('tpCommRatio equals the closed form 2(t − 1) · P / (9 · h · L)', () => {
  for (const [t, p, l] of [[8, 989, 450], [16, 2500, 900], [3, 400, 25]]) {
    close(tp(t, p, l), (2 * (t - 1) * p * 1000) / (9 * H * l));
  }
});

test('ppCommRatio: 16 and 4 stages (§6 reproducer rows)', () => {
  assert.equal(round(pp(16, 989, 50), 4), 0.0149);
  assert.equal(round(pp(16, 989, 450), 4), 0.0017);
  assert.deepEqual([4, 16].map((p) => pct(pp(p, 989, 50))), [0.4, 1.5]);
  assert.deepEqual([4, 16].map((p) => pct(pp(p, 989, 450))), [0.0, 0.2]);
  assert.deepEqual([4, 16].map((p) => pct(pp(p, 2500, 900))), [0.1, 0.2]);
  assert.deepEqual([4, 16].map((p) => pct(pp(p, 2500, 100))), [0.5, 1.9]);
});

test('ppCommRatio equals the closed form pp · P / (18 · h · layers · L)', () => {
  close(pp(16, 989, 50), (16 * 989 * 1000) / (18 * H * 96 * 50));
});

test('dpCommRatio: 64 replicas at 262,144 and 16,384 tokens (§6 reproducer rows)', () => {
  assert.deepEqual([16384, 262144].map((t) => pct(dp(64, t, 989, 50))), [79.2, 5.0]);
  assert.deepEqual([16384, 262144].map((t) => pct(dp(64, t, 989, 450))), [8.8, 0.6]);
  assert.deepEqual([16384, 262144].map((t) => pct(dp(64, t, 2500, 900))), [11.1, 0.7]);
  assert.deepEqual([16384, 262144].map((t) => pct(dp(64, t, 2500, 100))), [100.1, 6.3]);
});

test('dpCommRatio equals the closed form 2(N − 1) · P / (3 · N · T · L)', () => {
  close(dp(64, 262144, 989, 50), (2 * 63 * 989 * 1000) / (3 * 64 * 262144 * 50));
});

test('epCommRatio and epMinLinkGBps: DeepSeek-V4 hiding line at 6,144 FLOPs per byte', () => {
  assert.equal(round(epCommRatio({ peakTflops: 989, linkGBps: 450 }), 4), 0.3577);
  assert.equal(round(epCommRatio({ peakTflops: 989, linkGBps: 50 }), 4), 3.2194);
  assert.equal(round(epCommRatio({ peakTflops: 2500, linkGBps: 900 }), 4), 0.4521);
  assert.equal(round(epCommRatio({ peakTflops: 2500, linkGBps: 100 }), 3), 4.069);
  assert.equal(round(epMinLinkGBps({ peakTflops: 989 }), 2), 160.97);
  assert.equal(round(epMinLinkGBps({ peakTflops: 2500 }), 1), 406.9);
});

// ---- properties ----

test('every ratio is linear in peakTflops and inverse in linkGBps', () => {
  const cases = [
    (p, l) => tp(8, p, l), (p, l) => pp(16, p, l), (p, l) => dp(64, 262144, p, l), (p, l) => epCommRatio({ peakTflops: p, linkGBps: l }),
  ];
  for (const f of cases) {
    close(f(2000, 100), 2 * f(1000, 100));
    close(f(1000, 50), 2 * f(1000, 100));
  }
});

test('tpCommRatio is 0 at tp = 1 and dpCommRatio is 0 at dp = 1', () => {
  assert.equal(tp(1, 989, 450), 0);
  assert.equal(dp(1, 262144, 989, 450), 0);
});

test('epCommRatio at the minimum link is exactly 1', () => {
  for (const peakTflops of [989, 2500, 4500]) {
    close(epCommRatio({ peakTflops, linkGBps: epMinLinkGBps({ peakTflops }) }), 1);
  }
});

test('flopsPerByte overrides the DeepSeek-V4 default', () => {
  close(epCommRatio({ peakTflops: 989, linkGBps: 450, flopsPerByte: 3072 }), 2 * epCommRatio({ peakTflops: 989, linkGBps: 450 }));
  close(epMinLinkGBps({ peakTflops: 989, flopsPerByte: 3072 }), 2 * epMinLinkGBps({ peakTflops: 989 }));
});

// ---- P3-R13: one fact, two sources ----

test('epCommRatio\'s default flopsPerByte equals deepseek-v4-pro.ep_hiding_flops_per_byte in the data', async () => {
  const models = JSON.parse(await readFile(new URL('../data/models.json', import.meta.url), 'utf8'));
  const entry = models.entries.find((e) => e.id === 'deepseek-v4-pro');
  const fromData = entry.facts.ep_hiding_flops_per_byte.value;
  assert.equal(fromData, 6144);
  close(epCommRatio({ peakTflops: 989, linkGBps: 450 }), epCommRatio({ peakTflops: 989, linkGBps: 450, flopsPerByte: fromData }));
  close(epMinLinkGBps({ peakTflops: 989 }), epMinLinkGBps({ peakTflops: 989, flopsPerByte: fromData }));
});

// ---- errors and immutability ----

test('bad arguments throw RangeError("<fn>: <arg> must be …")', () => {
  const ok = { hidden: H, peakTflops: 989, linkGBps: 450 };
  assert.throws(() => tpCommRatio({ ...ok, tp: 0 }), { name: 'RangeError', message: /^tpCommRatio: tp must be a positive integer/ });
  assert.throws(() => tpCommRatio({ ...ok, tp: 2.5 }), /^RangeError: tpCommRatio: tp must be/);
  assert.throws(() => tpCommRatio({ ...ok, tp: 8, hidden: -1 }), /tpCommRatio: hidden must be/);
  assert.throws(() => tpCommRatio({ ...ok, tp: 8, peakTflops: 0 }), /tpCommRatio: peakTflops must be a positive finite number/);
  assert.throws(() => tpCommRatio({ ...ok, tp: 8, linkGBps: NaN }), /tpCommRatio: linkGBps must be/);
  assert.throws(() => ppCommRatio({ ...ok, pp: 0, layers: 96 }), /ppCommRatio: pp must be/);
  assert.throws(() => ppCommRatio({ ...ok, pp: 4, layers: 0 }), /ppCommRatio: layers must be/);
  assert.throws(() => ppCommRatio({ ...ok, pp: 4, layers: 96, hidden: 0 }), /ppCommRatio: hidden must be/);
  assert.throws(() => ppCommRatio({ ...ok, pp: 4, layers: 96, peakTflops: -1 }), /ppCommRatio: peakTflops must be/);
  assert.throws(() => ppCommRatio({ ...ok, pp: 4, layers: 96, linkGBps: Infinity }), /ppCommRatio: linkGBps must be/);
  assert.throws(() => dpCommRatio({ dp: 0, tokensPerReplica: 1, peakTflops: 1, linkGBps: 1 }), /dpCommRatio: dp must be/);
  assert.throws(() => dpCommRatio({ dp: 2, tokensPerReplica: 0, peakTflops: 1, linkGBps: 1 }), /dpCommRatio: tokensPerReplica must be/);
  assert.throws(() => dpCommRatio({ dp: 2, tokensPerReplica: 1, peakTflops: 0, linkGBps: 1 }), /dpCommRatio: peakTflops must be/);
  assert.throws(() => dpCommRatio({ dp: 2, tokensPerReplica: 1, peakTflops: 1, linkGBps: 0 }), /dpCommRatio: linkGBps must be/);
  assert.throws(() => epCommRatio({ peakTflops: 0, linkGBps: 1 }), /epCommRatio: peakTflops must be/);
  assert.throws(() => epCommRatio({ peakTflops: 1, linkGBps: 0 }), /epCommRatio: linkGBps must be/);
  assert.throws(() => epCommRatio({ peakTflops: 1, linkGBps: 1, flopsPerByte: 0 }), /epCommRatio: flopsPerByte must be/);
  assert.throws(() => epMinLinkGBps({ peakTflops: 0 }), /epMinLinkGBps: peakTflops must be/);
  assert.throws(() => epMinLinkGBps({ peakTflops: 1, flopsPerByte: -1 }), /epMinLinkGBps: flopsPerByte must be/);
});

test('inputs are not mutated', () => {
  const args = Object.freeze({ tp: 8, hidden: H, peakTflops: 989, linkGBps: 450 });
  assert.doesNotThrow(() => tpCommRatio(args));
  assert.doesNotThrow(() => ppCommRatio(Object.freeze({ pp: 4, hidden: H, layers: 96, peakTflops: 989, linkGBps: 450 })));
  assert.doesNotThrow(() => dpCommRatio(Object.freeze({ dp: 4, tokensPerReplica: 1024, peakTflops: 989, linkGBps: 450 })));
  assert.doesNotThrow(() => epCommRatio(Object.freeze({ peakTflops: 989, linkGBps: 450 })));
  assert.deepEqual(args, { tp: 8, hidden: H, peakTflops: 989, linkGBps: 450 });
});
