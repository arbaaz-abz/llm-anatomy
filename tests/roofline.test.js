import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import {
  FORMATS, bitsPerElement, bytesPerElement, matmulCost, elementwiseCost, arithmeticIntensity,
  ridgePoint, attainableTflops, rooflineTime, tokensToComputeBound,
} from '../math/roofline.js';

const round = (x, d) => Number(x.toFixed(d));
const close = (a, b, rel = 1e-9) => assert.ok(Math.abs(a - b) <= rel * Math.max(1, Math.abs(b)), `${a} ≉ ${b}`);
const H100 = { peakTflops: 989, bandwidthTBps: 3.35 };
const real = (m, bytesPerElem = 2) => matmulCost({ m, k: 8192, n: 8192, bytesPerElem });
const crossing = (peakTflops, bandwidthTBps, bytesPerElem, k = 8192, n = 8192) =>
  tokensToComputeBound({ peakTflops, bandwidthTBps, bytesPerElem, k, n });

// ---- FORMATS, bitsPerElement, bytesPerElement (gpu-primer §6; quantization, serving-calculator, prefill-decode) ----

test('FORMATS: the seven exact keys, never a bare fp8, with gpu-primer §6 layouts', () => {
  assert.deepEqual(Object.keys(FORMATS), ['fp32', 'bf16', 'fp16', 'fp8_e4m3', 'fp8_e5m2', 'mxfp4', 'nvfp4']);
  assert.deepEqual(FORMATS.fp32, { bits: 32, layout: '1/8/23' });
  assert.deepEqual(FORMATS.bf16, { bits: 16, layout: '1/8/7' });
  assert.deepEqual(FORMATS.fp16, { bits: 16, layout: '1/5/10' });
  assert.deepEqual(FORMATS.fp8_e4m3, { bits: 8, layout: '1/4/3' });
  assert.deepEqual(FORMATS.fp8_e5m2, { bits: 8, layout: '1/5/2' });
  assert.deepEqual(FORMATS.mxfp4, { bits: 4, layout: '1/2/1', blockSize: 32, scaleBits: 8 });
  assert.deepEqual(FORMATS.nvfp4, { bits: 4, layout: '1/2/1', blockSize: 16, scaleBits: 8 });
  assert.equal('fp8' in FORMATS, false);
});

test('FORMATS: every layout sums to its bits', () => {
  for (const [key, f] of Object.entries(FORMATS)) {
    assert.equal(f.layout.split('/').map(Number).reduce((a, b) => a + b, 0), f.bits, key);
  }
});

test('FORMATS is deeply frozen', () => {
  assert.ok(Object.isFrozen(FORMATS));
  for (const f of Object.values(FORMATS)) assert.ok(Object.isFrozen(f));
  assert.throws(() => { 'use strict'; FORMATS.bf16.bits = 8; }, TypeError);
});

test('bitsPerElement: gpu-primer and quantization examples (scales counted: bits + scaleBits / blockSize)', () => {
  assert.equal(bitsPerElement('bf16'), 16);
  assert.equal(bitsPerElement('fp8_e4m3'), 8);
  assert.equal(bitsPerElement('fp8_e5m2'), 8);
  assert.equal(bitsPerElement('nvfp4'), 4.5);
  assert.equal(bitsPerElement('mxfp4'), 4.25);
  assert.equal(bitsPerElement('fp32'), 32);
  assert.equal(bitsPerElement('fp16'), 16);
  // quantization §8 frame 8: "bits: BF16 16, FP8 8, MXFP4 4.25, NVFP4 4.5"; NVFP4 costs 0.25 more bits than MXFP4
  assert.equal(bitsPerElement('nvfp4') - bitsPerElement('mxfp4'), 0.25);
});

test('bytesPerElement: bitsPerElement / 8', () => {
  assert.equal(bytesPerElement('bf16'), 2);
  assert.equal(bytesPerElement('nvfp4'), 0.5625);
  assert.equal(bytesPerElement('mxfp4'), 0.53125);
  assert.equal(bytesPerElement('fp8_e4m3'), 1);
  assert.equal(bytesPerElement('fp32'), 4);
});

test('bitsPerElement and bytesPerElement throw on a bare fp8 or any unknown name', () => {
  for (const bad of ['fp8', 'FP8', 'int4', 'toString', '__proto__', '', 16, undefined, null]) {
    assert.throws(() => bitsPerElement(bad), /^RangeError: bitsPerElement: format must be one of fp32, bf16, fp16, fp8_e4m3, fp8_e5m2, mxfp4, nvfp4/);
    assert.throws(() => bytesPerElement(bad), /^RangeError: bytesPerElement: format must be one of/);
  }
});

test('P3-R13: FORMATS equals data/hardware.json formats for every fact both define', async () => {
  const hardware = JSON.parse(await readFile(new URL('../data/hardware.json', import.meta.url), 'utf8'));
  const facts = hardware.entries.find((e) => e.id === 'formats').facts;
  const layoutBits = (layout) => layout.split('/').map(Number).reduce((a, b) => a + b, 0);
  const scaleFormatKey = (name) => Object.keys(FORMATS).find((k) => k.endsWith(`_${name.toLowerCase()}`));
  const compare = {
    layout: (k, v) => { assert.equal(FORMATS[k].layout, v, `${k}.layout`); assert.equal(FORMATS[k].bits, layoutBits(v), `${k}.bits`); },
    block_size: (k, v) => assert.equal(FORMATS[k].blockSize, v, `${k}.blockSize`),
    scale_bits: (k, v) => assert.equal(FORMATS[k].scaleBits, v, `${k}.scaleBits`),
    bits_per_value: (k, v) => assert.equal(bitsPerElement(k), v, `bitsPerElement('${k}')`),
    scale_format: (k, v) => assert.equal(FORMATS[k].scaleBits, FORMATS[scaleFormatKey(v)].bits, `${k} scale ${v}`),
  };
  let compared = 0;
  for (const [factKey, fact] of Object.entries(facts)) {
    const format = Object.keys(FORMATS).find((k) => factKey.startsWith(`${k}_`));
    if (!format) continue;
    const suffix = factKey.slice(format.length + 1);
    assert.ok(Object.hasOwn(compare, suffix), `formats.${factKey}: no comparison with FORMATS for "${suffix}" (add one, P3-R13)`);
    compare[suffix](format, fact.value);
    compared += 1;
  }
  assert.ok(compared >= 10, `only ${compared} formats facts compared`);
});

// ---- matmulCost, elementwiseCost, arithmeticIntensity ----

test('matmulCost: flops 2mkn, bytes bytesPerElem·(mk + kn + mn) (gpu-primer §6)', () => {
  assert.deepEqual(matmulCost({ m: 4, k: 8, n: 8, bytesPerElem: 2 }), { flops: 512, bytes: 256 });
  assert.deepEqual(real(4), { flops: 536_870_912, bytes: 134_348_800 });
  assert.deepEqual(real(4096), { flops: 549_755_813_888, bytes: 268_435_456 });
});

test('elementwiseCost: reads `inputs` tensors, writes one', () => {
  const c = elementwiseCost({ elements: 32768, bytesPerElem: 2 });
  assert.deepEqual(c, { flops: 32768, bytes: 196608 });
  assert.equal(round(arithmeticIntensity(c), 4), 0.1667);
  assert.deepEqual(elementwiseCost({ elements: 10, inputs: 1, flopsPerElem: 3, bytesPerElem: 4 }), { flops: 30, bytes: 80 });
});

test('arithmeticIntensity: toy 2, 4-token real-size multiply 3.9961, 4,096 tokens 2048', () => {
  assert.equal(arithmeticIntensity({ flops: 512, bytes: 256 }), 2);
  assert.equal(round(arithmeticIntensity(real(4)), 4), 3.9961);
  assert.equal(arithmeticIntensity(real(4096)), 2048);
});

test('gpu-primer reproducer: H100 BF16 per-token table (intensity / attainable / % peak / compute µs / memory µs)', () => {
  const rows = [
    [1, 1.000, 3.3, 0.34, 0.136, 40.075, 'memory'],
    [4, 3.996, 13.4, 1.35, 0.543, 40.104, 'memory'],
    [64, 63.015, 211.1, 21.34, 8.685, 40.691, 'memory'],
    [256, 240.941, 807.2, 81.61, 34.742, 42.569, 'memory'],
    [512, 455.111, 989, 100, 69.484, 45.073, 'compute'],
    [4096, 2048, 989, 100, 555.87, 80.13, 'compute'],
  ];
  for (const [m, intensity, att, pct, computeUs, memoryUs, bound] of rows) {
    const c = real(m), I = arithmeticIntensity(c), a = attainableTflops({ intensity: I, ...H100 }), t = rooflineTime({ ...c, ...H100 });
    assert.equal(round(I, 3), intensity, `intensity ${m}`);
    assert.equal(round(a, 1), att, `attainable ${m}`);
    assert.equal(round((100 * a) / 989, 2), pct, `% peak ${m}`);
    assert.equal(round(t.computeS * 1e6, 3), computeUs, `compute ${m}`);
    assert.equal(round(t.memoryS * 1e6, m === 4096 ? 2 : 3), memoryUs, `memory ${m}`);
    assert.equal(t.bound, bound, `bound ${m}`);
  }
  assert.equal(round(arithmeticIntensity(real(8192)), 3), 2730.667);
});

test('gpu-primer reproducer: FP8 bytes at 256 and 4 tokens on the H100', () => {
  assert.equal(round(rooflineTime({ ...real(256, 1), ...H100 }).memoryS * 1e6, 2), 21.28);
  assert.equal(round(rooflineTime({ ...real(4, 1), ...H100 }).memoryS * 1e6, 2), 20.05);
});

// ---- ridgePoint, attainableTflops, rooflineTime ----

test('ridgePoint: peak / bandwidth in FLOPs per byte (gpu-primer §6)', () => {
  const cases = [[989, 3.35, 295.22], [1979, 3.35, 590.75], [2250, 8, 281.25], [2500, 8, 312.5], [15000, 8, 1875],
    [989, 4.8, 206.04], [2307, 7.38, 312.6], [35000, 22, 1590.91], [35000, 19.2, 1822.92]];
  for (const [peakTflops, bandwidthTBps, want] of cases) assert.equal(round(ridgePoint({ peakTflops, bandwidthTBps }), 2), want);
});

test('ridgePoint: Serving examples (prefill-decode 412.3; disaggregation GB300 FP4 1,875; chips table)', () => {
  assert.equal(round(ridgePoint({ peakTflops: 1979, bandwidthTBps: 4.8 }), 1), 412.3);
  assert.equal(ridgePoint({ peakTflops: 15000, bandwidthTBps: 8 }), 1875);
  assert.equal(ridgePoint({ peakTflops: 9000, bandwidthTBps: 8 }), 1125);
  assert.equal(ridgePoint({ peakTflops: 10000, bandwidthTBps: 8 }), 1250);
  assert.equal(ridgePoint({ peakTflops: 5000, bandwidthTBps: 8 }), 625);
  assert.equal(round(ridgePoint({ peakTflops: 4614, bandwidthTBps: 7.38 }), 1), 625.2);
});

test('attainableTflops: min(peak, intensity · bandwidth)', () => {
  assert.equal(round(attainableTflops({ intensity: 2, ...H100 }), 2), 6.7);
  assert.equal(round(attainableTflops({ intensity: 3.9961, ...H100 }), 2), 13.39);
  assert.equal(attainableTflops({ intensity: 2048, ...H100 }), 989);
});

test('attainableTflops ≤ peakTflops always, and equals it exactly at and past the ridge', () => {
  for (const peakTflops of [989, 2500, 35000]) {
    for (const bandwidthTBps of [3.35, 8, 22]) {
      for (const intensity of [0, 0.1, 1, 10, 100, 1e3, 1e4, 1e6]) {
        assert.ok(attainableTflops({ intensity, peakTflops, bandwidthTBps }) <= peakTflops);
      }
      assert.equal(attainableTflops({ intensity: ridgePoint({ peakTflops, bandwidthTBps }), peakTflops, bandwidthTBps }), peakTflops);
    }
  }
});

test('rooflineTime: the 4-token and 4,096-token multiplies on H100 BF16', () => {
  const small = rooflineTime({ ...real(4), ...H100 });
  assert.equal(Number(small.computeS.toPrecision(4)), 5.428e-7);
  assert.equal(Number(small.memoryS.toPrecision(5)), 4.0104e-5);
  assert.equal(small.timeS, small.memoryS);
  assert.equal(small.bound, 'memory');
  const big = rooflineTime({ ...real(4096), ...H100 });
  assert.equal(Number(big.computeS.toPrecision(5)), 5.5587e-4);
  assert.equal(Number(big.memoryS.toPrecision(4)), 8.013e-5);
  assert.equal(big.timeS, big.computeS);
  assert.equal(big.bound, 'compute');
});

test('rooflineTime: zero work takes zero time', () => {
  assert.deepEqual(rooflineTime({ flops: 0, bytes: 0, ...H100 }), { computeS: 0, memoryS: 0, timeS: 0, bound: 'compute' });
});

// ---- tokensToComputeBound ----

test('tokensToComputeBound: gpu-primer §6 examples', () => {
  assert.equal(round(crossing(989, 3.35, 2), 1), 318.2);
  assert.equal(round(crossing(1979, 3.35, 1), 1), 318.3);
  assert.equal(round(crossing(2500, 8, 2), 1), 338.3);
  assert.equal(round(crossing(15000, 8, 0.5625), 1), 605.3);
  assert.equal(round(crossing(35000, 22, 0.5625), 1), 502.3);
  assert.equal(round(crossing(35000, 19.2, 0.5625), 1), 586.1);
});

test('tokensToComputeBound: gpu-primer reproducer chip rows', () => {
  const rows = [[989, 4.8, 2, 217.0], [2250, 8, 2, 302.0], [9000, 8, 0.5625, 342.9], [5000, 8, 1, 338.3],
    [10000, 8, 0.53125, 361.3], [2307, 7.38, 2, 338.4], [4614, 7.38, 1, 338.4]];
  for (const [peak, bw, b, want] of rows) assert.equal(round(crossing(peak, bw, b), 1), want, `${peak}/${bw}/${b}`);
});

test('tokensToComputeBound: Serving examples (prefill-decode 217.1 / 217.0 / 302.0 / 318.2; disaggregation 698.7)', () => {
  assert.equal(round(crossing(1979, 4.8, 1), 1), 217.1);
  assert.equal(round(crossing(989, 4.8, 2), 1), 217.0);
  assert.equal(round(crossing(4500, 8, 1), 1), 302.0);
  assert.equal(round(crossing(989, 3.35, 2), 1), 318.2);
  assert.equal(round(crossing(15000, 8, 0.5625, 7168, 3072), 1), 698.7);
});

test('disaggregation: expert multiply intensity (7,168 × 3,072, NVFP4) at 1 / 8 / 16 / 32 / 72 / 288 tokens', () => {
  const expert = (m) => arithmeticIntensity(matmulCost({ m, k: 7168, n: 3072, bytesPerElem: bytesPerElement('nvfp4') }));
  assert.equal(round(expert(1), 1), 3.6);
  assert.deepEqual([8, 16, 32, 72, 288].map((m) => round(expert(m), 1)), [28.3, 56.5, 112.1, 247.7, 903.1]);
  assert.ok(expert(288) < ridgePoint({ peakTflops: 15000, bandwidthTBps: 8 }));
});

test('gpu-primer reproducer: intensity at 4 / 256 / 512 tokens per chip format', () => {
  const at = (m, b) => round(arithmeticIntensity(real(m, b)), m === 4 ? 2 : 1);
  assert.deepEqual([at(4, 2), at(256, 2), at(512, 2)], [4.00, 240.9, 455.1]);
  assert.deepEqual([at(4, 1), at(256, 1), at(512, 1)], [7.99, 481.9, 910.2]);
  assert.deepEqual([at(4, 0.5625), at(512, 0.5625)], [14.21, 1618.2]);
});

test('ridgePoint and tokensToComputeBound agree: the intensity at the returned m equals the ridge', () => {
  for (const [peak, bw, b, k, n] of [[989, 3.35, 2, 8192, 8192], [15000, 8, 0.5625, 8192, 8192], [1979, 4.8, 1, 8192, 8192],
    [15000, 8, 0.5625, 7168, 3072], [35000, 19.2, 0.5625, 8192, 8192]]) {
    const m = crossing(peak, bw, b, k, n);
    close(arithmeticIntensity(matmulCost({ m, k, n, bytesPerElem: b })), ridgePoint({ peakTflops: peak, bandwidthTBps: bw }));
  }
});

test('rooflineTime.bound flips exactly at tokensToComputeBound', () => {
  for (const [peak, bw, b] of [[989, 3.35, 2], [2500, 8, 2], [15000, 8, 0.5625], [1979, 4.8, 1]]) {
    const hw = { peakTflops: peak, bandwidthTBps: bw }, m = crossing(peak, bw, b);
    const at = (x) => rooflineTime({ ...matmulCost({ m: x, k: 8192, n: 8192, bytesPerElem: b }), ...hw }).bound;
    assert.equal(at(m * (1 - 1e-9)), 'memory');
    assert.equal(at(m * (1 + 1e-9)), 'compute');
    assert.equal(at(Math.floor(m)), 'memory');
    assert.equal(at(Math.ceil(m)), 'compute');
  }
});

test('tokensToComputeBound: Infinity when the multiply can never reach the ridge', () => {
  // As m grows the intensity tends to 2kn / (b(k + n)); here 2·8·8 / (2·16) = 4 < ridge 295.
  assert.equal(tokensToComputeBound({ ...H100, bytesPerElem: 2, k: 8, n: 8 }), Infinity);
  // Exactly at the limit: still never reached.
  assert.equal(tokensToComputeBound({ peakTflops: 4, bandwidthTBps: 1, bytesPerElem: 2, k: 8, n: 8 }), Infinity);
});

// ---- validation and purity ----

test('every bad argument throws "<fn>: <arg> must be …"', () => {
  const ok = { m: 4, k: 8, n: 8, bytesPerElem: 2 };
  for (const bad of [0, -1, Number.NaN, Infinity, '2', undefined]) {
    assert.throws(() => matmulCost({ ...ok, m: bad }), /^RangeError: matmulCost: m must be a positive finite number/);
    assert.throws(() => matmulCost({ ...ok, bytesPerElem: bad }), /^RangeError: matmulCost: bytesPerElem must be a positive finite number/);
    assert.throws(() => elementwiseCost({ elements: bad, bytesPerElem: 2 }), /^RangeError: elementwiseCost: elements must be a positive finite number/);
    assert.throws(() => arithmeticIntensity({ flops: 1, bytes: bad }), /^RangeError: arithmeticIntensity: bytes must be a positive finite number/);
    assert.throws(() => ridgePoint({ peakTflops: bad, bandwidthTBps: 1 }), /^RangeError: ridgePoint: peakTflops must be a positive finite number/);
    assert.throws(() => ridgePoint({ peakTflops: 1, bandwidthTBps: bad }), /^RangeError: ridgePoint: bandwidthTBps must be a positive finite number/);
    assert.throws(() => rooflineTime({ flops: 1, bytes: 1, peakTflops: bad, bandwidthTBps: 1 }), /^RangeError: rooflineTime: peakTflops must be/);
    assert.throws(() => tokensToComputeBound({ ...H100, bytesPerElem: 2, k: bad, n: 8 }), /^RangeError: tokensToComputeBound: k must be/);
    assert.throws(() => tokensToComputeBound({ peakTflops: bad, bandwidthTBps: 1, bytesPerElem: 2, k: 8, n: 8 }), /^RangeError: tokensToComputeBound: peakTflops must be/);
    assert.throws(() => attainableTflops({ intensity: 1, peakTflops: 1, bandwidthTBps: bad }), /^RangeError: attainableTflops: bandwidthTBps must be/);
  }
  for (const bad of [-1, Number.NaN, Infinity, '2', undefined]) {
    assert.throws(() => arithmeticIntensity({ flops: bad, bytes: 1 }), /^RangeError: arithmeticIntensity: flops must be a finite number ≥ 0/);
    assert.throws(() => rooflineTime({ flops: bad, bytes: 1, ...H100 }), /^RangeError: rooflineTime: flops must be a finite number ≥ 0/);
    assert.throws(() => rooflineTime({ flops: 1, bytes: bad, ...H100 }), /^RangeError: rooflineTime: bytes must be a finite number ≥ 0/);
    assert.throws(() => attainableTflops({ intensity: bad, ...H100 }), /^RangeError: attainableTflops: intensity must be a finite number ≥ 0/);
  }
  for (const bad of [-1, Number.NaN, Infinity, '2', null]) {
    assert.throws(() => elementwiseCost({ elements: 1, flopsPerElem: bad, bytesPerElem: 2 }), /^RangeError: elementwiseCost: flopsPerElem must be a finite number ≥ 0/);
  }
  for (const bad of [-1, 1.5, Number.NaN, '2']) {
    assert.throws(() => elementwiseCost({ elements: 1, inputs: bad, bytesPerElem: 2 }), /^RangeError: elementwiseCost: inputs must be an integer ≥ 0/);
  }
});

test('inputs are never mutated', () => {
  const args = Object.freeze({ m: 4, k: 8192, n: 8192, bytesPerElem: 2 });
  const cost = Object.freeze(matmulCost(args));
  const hw = Object.freeze({ ...H100 });
  rooflineTime({ ...cost, ...hw });
  arithmeticIntensity(cost);
  tokensToComputeBound(Object.freeze({ ...hw, bytesPerElem: 2, k: 8192, n: 8192 }));
  assert.deepEqual(args, { m: 4, k: 8192, n: 8192, bytesPerElem: 2 });
  assert.deepEqual(hw, H100);
});
