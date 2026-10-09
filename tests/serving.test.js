import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { FLOPS_PER_PARAM_TOKEN } from '../math/scale.js';
import { bitsPerElement, arithmeticIntensity, matmulCost, ridgePoint, tokensToComputeBound } from '../math/roofline.js';
import { kvCacheBytes, kvBytesPerTokenMla } from '../math/memory.js';
import { formatBytes } from '../math/core.js';
import {
  FORWARD_FLOPS_PER_PARAM_TOKEN, RUNNING_EXAMPLE, TOY_REQUESTS, hbmFor, weightBytes, stepTime, freeHbmPerGpu, maxUsersPerGpu,
  prefillTokPerSecCeiling, requestTimeline, kvTransferTime, tokensPerExpert, minGpusForWeights, usersAtTarget, costPerMillion,
} from '../math/serving.js';

const data = async (f) => JSON.parse(await readFile(new URL(`../data/${f}.json`, import.meta.url), 'utf8'));
const entry = (d, id) => d.entries.find((e) => e.id === id);
const fact = (e, k) => e.facts[k].value;
// A storyboard prints `x` to `d` decimals: the function must reproduce that print exactly.
const printed = (x, d) => Number(x.toFixed(d));
const GB = (x) => printed(x / 1e9, 2);

const L = RUNNING_EXAMPLE; // Llama-3.1-70B, FP8, one H200
const V = { activeParamsPerGpu: 49e9, weightBytesPerGpu: 865e9 / 16, dModel: 7168, actBytesPerElem: 1, peakTflops: 5000, bandwidthTBps: 8 };
const decode = (model, users, context, extra = {}) => stepTime({ ...model, tokens: users, seqs: users, context, ...extra });
const prefill = (model, tokens) => stepTime({ ...model, tokens, seqs: 0, context: 0 });

// ---- constants and data pins (P4-R3, R4, R21; Review Focus 1, 3) ----

test('the forward pass is a third of training\'s 6 FLOPs per parameter per token (P4-R4)', () => {
  assert.equal(FORWARD_FLOPS_PER_PARAM_TOKEN, FLOPS_PER_PARAM_TOKEN / 3);
  assert.equal(FORWARD_FLOPS_PER_PARAM_TOKEN, 2);
});

test('RUNNING_EXAMPLE restates the data (P4-R21, P3-R13)', async () => {
  const llama = entry(await data('models'), 'llama-3.1-70b');
  const h200 = entry(await data('hardware'), 'h200');
  assert.equal(L.activeParamsPerGpu, fact(llama, 'total_params'));
  assert.equal(L.weightBytesPerGpu, fact(llama, 'total_params')); // FP8: one byte per parameter
  assert.equal(L.dModel, fact(llama, 'd_model'));
  assert.equal(L.kvBytesPerToken, fact(llama, 'kv_bytes_per_token'));
  assert.equal(L.peakTflops, fact(h200, 'fp8_e4m3_dense_tflops'));
  assert.equal(L.bandwidthTBps, fact(h200, 'hbm_tbps'));
  assert.equal(L.actBytesPerElem, 1);
  assert.ok(Object.isFrozen(RUNNING_EXAMPLE));
  assert.deepEqual(Object.keys(L), ['activeParamsPerGpu', 'weightBytesPerGpu', 'dModel', 'actBytesPerElem', 'kvBytesPerToken', 'peakTflops', 'bandwidthTBps']);
});

test('hbmFor prints its basis: usable when the data has it, else nominal (Review Focus 3)', async () => {
  const hw = await data('hardware');
  assert.deepEqual(hbmFor(entry(hw, 'b200')), { bytes: 180e9, basis: 'usable' });
  assert.deepEqual(hbmFor(entry(hw, 'b200')), { bytes: fact(entry(hw, 'b200'), 'hbm_usable_gb') * 1e9, basis: 'usable' });
  assert.deepEqual(hbmFor(entry(hw, 'h200')), { bytes: fact(entry(hw, 'h200'), 'hbm_gb') * 1e9, basis: 'nominal' });
  assert.deepEqual(hbmFor(entry(hw, 'h100')), { bytes: 80e9, basis: 'nominal' });
  assert.deepEqual(hbmFor(entry(hw, 'gb300-nvl72')), { bytes: 288e9, basis: 'nominal' });
});

test('hbmFor: an entry with no HBM fact, or no facts, throws', () => {
  assert.throws(() => hbmFor({ id: 'x', facts: {} }), /^RangeError: hbmFor: entry x has neither/);
  assert.throws(() => hbmFor({ id: 'x' }), /^RangeError: hbmFor: entry must be a hardware entry/);
  assert.throws(() => hbmFor(null), RangeError);
  assert.throws(() => hbmFor({ id: 'x', facts: { hbm_gb: { value: -1 } } }), /hbm gigabytes must be a positive finite number/);
});

test('TOY_REQUESTS is frozen and holds A–D as paged-attention §5 gives them', () => {
  assert.deepEqual(TOY_REQUESTS.map((r) => r.id), ['A', 'B', 'C', 'D']);
  assert.deepEqual(TOY_REQUESTS.map((r) => [r.arrives, r.prompt, r.output]), [[0, 8, 4], [0, 5, 2], [0, 10, 6], [1, 6, 3]]);
  assert.ok(Object.isFrozen(TOY_REQUESTS) && TOY_REQUESTS.every(Object.isFrozen));
  assert.throws(() => { 'use strict'; TOY_REQUESTS[0].prompt = 9; }, TypeError);
});

// ---- weightBytes, roofline anchors (prefill-decode §6; quantization §6; serving-calculator) ----

test('weightBytes: params · bits / 8 (quantization §6)', () => {
  assert.deepEqual([16, 8, 4.5, 4.25].map((b) => weightBytes({ params: 70e9, bitsPerParam: b }) / 1e9), [140, 70, 39.375, 37.1875]);
  assert.equal(printed(weightBytes({ params: 70e9, bitsPerParam: 4.25 }) / 1e9, 2), 37.19);
  assert.deepEqual([16, 8, 4.5].map((b) => weightBytes({ params: 1.6e12, bitsPerParam: b }) / 1e9), [3200, 1600, 900]);
  assert.equal(weightBytes({ params: 1.6e12, bitsPerParam: bitsPerElement('nvfp4') }), 900e9);
  assert.throws(() => weightBytes({ params: 0, bitsPerParam: 8 }), /^RangeError: weightBytes: params must be a positive finite number/);
  assert.throws(() => weightBytes({ params: 1, bitsPerParam: -8 }), /bitsPerParam/);
});

test('the roofline anchors the Serving pages quote', () => {
  assert.equal(printed(ridgePoint({ peakTflops: 1979, bandwidthTBps: 4.8 }), 1), 412.3);
  const cross = (peakTflops, bandwidthTBps, bytesPerElem) => printed(tokensToComputeBound({ peakTflops, bandwidthTBps, bytesPerElem, k: 8192, n: 8192 }), 1);
  assert.deepEqual([cross(1979, 4.8, 1), cross(989, 4.8, 2), cross(4500, 8, 1), cross(989, 3.35, 2)], [217.1, 217.0, 302.0, 318.2]);
  assert.equal(printed(tokensToComputeBound({ peakTflops: 15000, bandwidthTBps: 8, bytesPerElem: 0.5625, k: 7168, n: 3072 }), 1), 698.7);
  assert.equal(Math.round(ridgePoint({ peakTflops: 15000, bandwidthTBps: 8 })), 1875);
});

// ---- stepTime (prefill-decode §6 and every importer) ----

test('stepTime: one token, one user, no context — compute 0.071 ms, memory 14.587 ms, memory-bound, intensity 2.0', () => {
  const s = decode(L, 1, 0);
  assert.equal(printed(s.computeS * 1e3, 3), 0.071);
  assert.equal(printed(s.memoryS * 1e3, 3), 14.587);
  assert.equal(printed(s.timeS * 1e3, 2), 14.59);
  assert.equal(s.bound, 'memory');
  assert.equal(printed(arithmeticIntensity(s), 1), 2.0);
  assert.deepEqual(Object.keys(s).sort(), ['actBytes', 'bound', 'bytes', 'computeS', 'flops', 'kvBytes', 'memoryS', 'timeS']);
});

test('stepTime: decode at one user and zero context is (weights + activations) / bandwidth', () => {
  const s = decode(L, 1, 0);
  assert.equal(s.timeS, (L.weightBytesPerGpu + s.actBytes) / (L.bandwidthTBps * 1e12));
  assert.equal(s.kvBytes, 0);
});

test('stepTime: prefill crossover at 217 / 218 tokens, and 1,000 and 8,192 tokens', () => {
  const at217 = prefill(L, 217);
  assert.equal(printed(at217.computeS * 1e3, 3), 15.351);
  assert.equal(printed(at217.memoryS * 1e3, 3), 15.356);
  assert.equal(at217.bound, 'memory');
  assert.equal(prefill(L, 218).bound, 'compute');
  const at1000 = prefill(L, 1000);
  assert.equal(printed(at1000.computeS * 1e3, 3), 70.743);
  assert.equal(printed(at1000.memoryS * 1e3, 3), 18.144);
  assert.equal(GB(at1000.bytes), 87.09);
  assert.equal(printed(at1000.timeS * 1e3, 2), 70.74);
  assert.equal(printed(arithmeticIntensity(at1000), 1), 1607.5);
  assert.equal(printed(prefill(L, 8192).timeS * 1e3, 2), 579.53);
});

test('stepTime: at tokensToComputeBound compute and memory times agree (the whole-model crossover)', () => {
  const m = tokensToComputeBound({ peakTflops: 1979, bandwidthTBps: 4.8, bytesPerElem: 1, k: 8192, n: 8192 });
  const s = prefill(L, m);
  assert.ok(Math.abs(s.computeS - s.memoryS) / s.timeS < 1e-3);
  assert.ok(Math.abs(prefill(L, 217).computeS - prefill(L, 217).memoryS) < 1e-4);
});

test('stepTime: timeS is max(computeS, memoryS), always', () => {
  for (const [tokens, seqs, context] of [[1, 1, 0], [5, 5, 2048], [217, 0, 0], [4096, 0, 0], [105, 105, 2048], [20000, 0, 0]]) {
    const s = stepTime({ ...L, tokens, seqs, context });
    assert.equal(s.timeS, Math.max(s.computeS, s.memoryS));
    assert.equal(s.bytes, L.weightBytesPerGpu + s.actBytes + s.kvBytes);
    assert.equal(s.flops, 2 * L.activeParamsPerGpu * tokens);
  }
});

test('stepTime: decode at context 2,048 for 1 / 8 / 64 / 105 users (step ms · tok/s per user · tok/s per GPU · intensity · KV read ms)', () => {
  const rows = { 1: [14.73, 67.9, 68, 2.0, 0.14], 8: [15.73, 63.6, 509, 14.8, 1.12], 64: [23.76, 42.1, 2694, 78.6, 8.95], 105: [29.64, 33.7, 3543, 103.3, 14.68] };
  for (const [users, [ms, perUser, perGpu, intensity, kvReadMs]] of Object.entries(rows)) {
    const s = decode(L, Number(users), 2048);
    assert.equal(printed(s.timeS * 1e3, 2), ms, `${users} users: step`);
    assert.equal(printed(1 / s.timeS, 1), perUser, `${users} users: per user`);
    assert.equal(Math.round(Number(users) / s.timeS), perGpu, `${users} users: per GPU`);
    assert.equal(printed(arithmeticIntensity(s), 1), intensity, `${users} users: intensity`);
    assert.equal(printed((s.kvBytes / (L.bandwidthTBps * 1e12)) * 1e3, 2), kvReadMs, `${users} users: KV read ms`);
  }
});

test('stepTime and maxUsersPerGpu at the longer contexts (prefill-decode §6)', () => {
  const cache = (tokens) => kvCacheBytes({ bytesPerToken: 327680, tokens });
  assert.deepEqual([2048, 8192, 32768, 131072].map((c) => maxUsersPerGpu(71e9, cache(c))), [105, 26, 6, 1]);
  const atMax = [[8192, 29.22, 890], [32768, 28.03, 214], [131072, 23.53, 42]];
  for (const [context, ms, perGpu] of atMax) {
    const users = maxUsersPerGpu(71e9, cache(context));
    const s = decode(L, users, context);
    assert.equal(printed(s.timeS * 1e3, 2), ms, `context ${context}`);
    assert.equal(Math.round(users / s.timeS), perGpu, `context ${context}: tok/s/GPU`);
  }
});

test('stepTime: BF16 on an H200 (weights 140 GB, BF16 activations, peak 989): free 1.0 GB, one user, 29.31 ms', () => {
  const bf16 = { ...L, weightBytesPerGpu: 140e9, actBytesPerElem: 2, peakTflops: 989 };
  const free = freeHbmPerGpu({ hbmBytes: 141e9, weightBytes: 140e9, gpus: 1 });
  assert.equal(printed(free / 1e9, 1), 1.0);
  const users = maxUsersPerGpu(free, kvCacheBytes({ bytesPerToken: 327680, tokens: 2048 }));
  assert.equal(users, 1);
  assert.equal(printed(decode(bf16, users, 2048).timeS * 1e3, 2), 29.31);
});

test('stepTime: B200 FP8 on 180 GB usable (not 192 nominal): 1 user 8.84 ms, 163 users, 22.77 ms, 7,158 tok/s/GPU', () => {
  const b200 = { ...L, peakTflops: 4500, bandwidthTBps: 8 };
  assert.equal(printed(decode(b200, 1, 2048).timeS * 1e3, 2), 8.84);
  const cache = kvCacheBytes({ bytesPerToken: 327680, tokens: 2048 });
  const users = maxUsersPerGpu(180e9 - L.weightBytesPerGpu, cache);
  assert.equal(users, 163);
  assert.equal(maxUsersPerGpu(192e9 - L.weightBytesPerGpu, cache), 181); // the nominal figure would overstate it
  const s = decode(b200, users, 2048);
  assert.equal(printed(s.timeS * 1e3, 2), 22.77);
  assert.equal(Math.round(users / s.timeS), 7158);
});

test('stepTime: prefix-caching\'s skipped prefill and the cache it holds', () => {
  assert.equal(printed(prefill(L, 10000).timeS * 1e3, 1), 707.4);
  assert.equal(kvCacheBytes({ bytesPerToken: 327680, tokens: 10000 }), 3_276_800_000);
});

test('stepTime: batching\'s short prompts — step 0 (23 prompt tokens) 14.7 ms, a three-user decode step 14.6 ms (context 16)', () => {
  assert.equal(printed(stepTime({ ...L, tokens: 23, seqs: 0, context: 16 }).timeS * 1e3, 1), 14.7);
  assert.equal(printed(stepTime({ ...L, tokens: 3, seqs: 3, context: 16 }).timeS * 1e3, 1), 14.6);
});

test('stepTime: quantization §6 — H200 users, decode and prefill by weight format', () => {
  const kv = (bytesPerToken) => kvCacheBytes({ bytesPerToken, tokens: 2048 });
  const rows = [
    // [weight bits, activation bytes, peak TFLOPS, users, decode ms, prefill-4096 ms]
    [16, 2, 989, 1, 29.31, 579.8],
    [8, 1, 1979, 105, 14.73, 289.8],
    [4.5, 2, 989, 151, 8.35, 579.8],
  ];
  for (const [bits, actBytesPerElem, peakTflops, users, decodeMs, prefillMs] of rows) {
    const weights = weightBytes({ params: 70e9, bitsPerParam: bits });
    const model = { ...L, weightBytesPerGpu: weights, actBytesPerElem, peakTflops };
    assert.equal(maxUsersPerGpu(freeHbmPerGpu({ hbmBytes: 141e9, weightBytes: weights, gpus: 1 }), kv(327680)), users, `${bits} bits: users`);
    assert.equal(printed(decode(model, 1, 2048).timeS * 1e3, 2), decodeMs, `${bits} bits: decode`);
    assert.equal(printed(prefill(model, 4096).timeS * 1e3, 1), prefillMs, `${bits} bits: prefill`);
  }
  const fp8Kv = [1, 105, 151].map((_, i) => [141e9 - weightBytes({ params: 70e9, bitsPerParam: [16, 8, 4.5][i] })]).map(([free]) => maxUsersPerGpu(free, kv(163840)));
  assert.deepEqual(fp8Kv, [2, 211, 302]);
});

test('stepTime: quantization §6 — B200 (180 GB usable) users, decode and prefill by format, with FP8 KV', () => {
  const rows = [
    // [bits, act bytes, peak, users, decode ms, prefill ms, users with FP8 KV]
    [16, 2, 2250, 59, 17.59, 254.9, 119],
    [8, 1, 4500, 163, 8.84, 127.4, 327],
    [bitsPerElement('nvfp4'), 1, 9000, 209, 5.01, 63.7, 419],
    [bitsPerElement('mxfp4'), 1, 9000, 212, 4.73, 63.7, 425],
  ];
  for (const [bits, actBytesPerElem, peakTflops, users, decodeMs, prefillMs, fp8KvUsers] of rows) {
    const weights = weightBytes({ params: 70e9, bitsPerParam: bits });
    const model = { ...L, weightBytesPerGpu: weights, actBytesPerElem, peakTflops, bandwidthTBps: 8 };
    const free = freeHbmPerGpu({ hbmBytes: 180e9, weightBytes: weights, gpus: 1 });
    assert.equal(maxUsersPerGpu(free, kvCacheBytes({ bytesPerToken: 327680, tokens: 2048 })), users, `${bits} bits: users`);
    assert.equal(maxUsersPerGpu(free, kvCacheBytes({ bytesPerToken: 163840, tokens: 2048 })), fp8KvUsers, `${bits} bits: FP8 KV users`);
    assert.equal(printed(decode(model, 1, 2048).timeS * 1e3, 2), decodeMs, `${bits} bits: decode`);
    assert.equal(printed(prefill(model, 4096).timeS * 1e3, 1), prefillMs, `${bits} bits: prefill`);
  }
});

test('stepTime: errors name the argument', () => {
  const ok = { ...L, tokens: 1, seqs: 1, context: 0 };
  for (const [key, bad] of [['tokens', 0], ['seqs', -1], ['context', -1], ['activeParamsPerGpu', 0], ['dModel', NaN], ['peakTflops', 0], ['bandwidthTBps', Infinity], ['kvBytesPerToken', -1], ['weightBytesPerGpu', -1], ['actBytesPerElem', 0]]) {
    assert.throws(() => stepTime({ ...ok, [key]: bad }), new RegExp(`^RangeError: stepTime: ${key} must be`), key);
  }
});

// ---- freeHbmPerGpu, maxUsersPerGpu (Review Focus 2) ----

test('a model that does not fit gives negative free memory and 0 users, never a negative count (Review Focus 2)', () => {
  const free = freeHbmPerGpu({ hbmBytes: 80e9, weightBytes: weightBytes({ params: 70e9, bitsPerParam: 16 }), gpus: 1 });
  assert.ok(free < 0);
  assert.equal(free, -60e9);
  assert.equal(maxUsersPerGpu(free, 327680 * 2048), 0);
  assert.equal(maxUsersPerGpu(-5e9, 1e6), 0);
  assert.equal(maxUsersPerGpu(0, 1e6), 0);
  assert.equal(maxUsersPerGpu(999_999, 1e6), 0);
  assert.throws(() => maxUsersPerGpu(71e9, 0), RangeError);
  assert.throws(() => maxUsersPerGpu(71e9, -1), /cacheBytesPerUser must be a positive finite number/);
  assert.throws(() => maxUsersPerGpu(NaN, 1), /freeBytes must be a finite number/);
  assert.throws(() => maxUsersPerGpu(71e9, kvCacheBytes({ bytesPerToken: 327680, tokens: 0 })), RangeError); // context 0 → cache 0 → throws, never Infinity
});

test('freeHbmPerGpu: V4-Pro\'s 865 GB over 1 / 8 / 16 / 32 / 72 GPUs of 288 GB (disaggregation §6, serving-calculator)', () => {
  const free = (gpus, hbmBytes = 288e9) => freeHbmPerGpu({ hbmBytes, weightBytes: 865e9, gpus });
  assert.deepEqual([1, 8, 16, 32, 72].map((g) => printed(free(g) / 1e9, 1)), [-577.0, 179.9, 233.9, 261.0, 276.0]);
  assert.deepEqual([1, 8, 16, 32, 72].map((g) => printed(865 / g, 1)), [865, 108.1, 54.1, 27.0, 12.0]);
  assert.deepEqual([8, 16, 72].map((g) => GB(free(g))), [179.88, 233.94, 275.99]);
  assert.deepEqual([8, 16].map((g) => GB(free(g, 186e9))), [77.88, 131.94]);
  assert.throws(() => freeHbmPerGpu({ hbmBytes: 288e9, weightBytes: 865e9, gpus: 0 }), /gpus must be a positive integer/);
  assert.throws(() => freeHbmPerGpu({ hbmBytes: 0, weightBytes: 1, gpus: 1 }), /hbmBytes/);
});

test('maxUsersPerGpu: speculative-decoding\'s 211 users at context 1,024', () => {
  assert.equal(maxUsersPerGpu(71e9, kvCacheBytes({ bytesPerToken: 327680, tokens: 1024 })), 211);
});

// ---- prefillTokPerSecCeiling (P4-R5) ----

test('prefillTokPerSecCeiling: peak / (2 · active) — 14,135.7 for the running example, 51,020 for V4-Pro on GB300', () => {
  assert.equal(printed(prefillTokPerSecCeiling({ activeParamsPerGpu: 70e9, peakTflops: 1979 }), 1), 14135.7);
  assert.equal(Math.round(prefillTokPerSecCeiling({ activeParamsPerGpu: 49e9, peakTflops: 5000 })), 51020);
  assert.throws(() => prefillTokPerSecCeiling({ activeParams: 70e9, peakTflops: 1979 }), /activeParamsPerGpu must be a positive finite number/);
  assert.throws(() => prefillTokPerSecCeiling({ activeParamsPerGpu: 1, peakTflops: 0 }), /peakTflops/);
});

// ---- requestTimeline (serving-overview §6) ----

test('requestTimeline: prefill times for 3 / 100 / 500 / 2,000 / 8,000 / 20,000 / 128,000 tokens', () => {
  const t = (promptTokens) => requestTimeline({ promptTokens, model: L, outputTokens: 2, decodeTokPerS: 67.9 }).prefillS;
  assert.deepEqual([3, 100, 500, 2000, 8000].map((n) => printed(t(n) * 1e3, n === 8000 ? 1 : 2)), [14.59, 14.94, 35.37, 141.49, 565.9]);
  assert.equal(printed(t(20000), 3), 1.415);
  assert.equal(printed(t(128000), 3), 9.055);
});

test('requestTimeline: 2,000-token prompt, 500 tokens out at 67.90 tok/s — TTFT 0.1415, TPOT 0.0147, e2e 7.49, decode share 0.981', () => {
  const r = requestTimeline({ promptTokens: 2000, model: L, outputTokens: 500, decodeTokPerS: 67.9 });
  assert.deepEqual(
    [printed(r.ttftS, 4), printed(r.tpotS, 4), printed(r.e2eS, 2), printed(r.decodeShare, 3), r.queueS],
    [0.1415, 0.0147, 7.49, 0.981, 0],
  );
  assert.deepEqual(Object.keys(r), ['queueS', 'prefillS', 'ttftS', 'tpotS', 'e2eS', 'decodeShare']);
});

test('requestTimeline: the other serving-overview rows', () => {
  const row = (extra) => requestTimeline({ promptTokens: 2000, model: L, outputTokens: 500, decodeTokPerS: 67.9, ...extra });
  const long = row({ promptTokens: 20000 });
  assert.deepEqual([printed(long.ttftS, 3), printed(long.e2eS, 2), printed(long.decodeShare, 3)], [1.415, 8.76, 0.839]);
  const short = row({ outputTokens: 50 });
  assert.deepEqual([printed(short.e2eS, 2), printed(short.decodeShare, 3)], [0.86, 0.836]);
  const one = row({ outputTokens: 1 });
  assert.equal(printed(one.e2eS, 4), 0.1415);
  assert.equal(one.e2eS, one.ttftS);
  assert.equal(one.decodeShare, 0);
  const slow = row({ decodeTokPerS: 33.74 });
  assert.deepEqual([printed(slow.tpotS, 4), printed(slow.e2eS, 2)], [0.0296, 14.93]);
  const queued = row({ decodeTokPerS: 33.74, queueS: 2 });
  assert.deepEqual([printed(queued.ttftS, 4), printed(queued.e2eS, 2), queued.queueS], [2.1415, 16.93, 2]);
  const tiny = requestTimeline({ promptTokens: 3, model: L, outputTokens: 2, decodeTokPerS: 67.9 });
  assert.deepEqual([printed(tiny.ttftS, 4), printed(tiny.e2eS, 3)], [0.0146, 0.029]);
});

test('requestTimeline: the chip rates are 1 / stepTime — 67.90 at one user, 33.74 at 105 (context 2,048)', () => {
  assert.equal(printed(1 / decode(L, 1, 2048).timeS, 2), 67.9);
  assert.equal(printed(1 / decode(L, 105, 2048).timeS, 2), 33.74);
});

test('requestTimeline properties: e2e ≥ TTFT, decode share in [0, 1), prefill is stepTime, no second definition, inputs untouched', () => {
  const model = Object.freeze({ ...L });
  for (const promptTokens of [1, 3, 217, 2000, 128000]) {
    for (const outputTokens of [1, 2, 50, 500]) {
      const r = requestTimeline({ promptTokens, model, outputTokens, decodeTokPerS: 50, queueS: 0.5 });
      assert.ok(r.e2eS >= r.ttftS);
      assert.ok(r.decodeShare >= 0 && r.decodeShare < 1);
      assert.equal(r.prefillS, stepTime({ ...model, tokens: promptTokens, seqs: 0, context: 0 }).timeS);
      assert.equal(r.ttftS, r.queueS + r.prefillS);
    }
  }
});

test('requestTimeline: a non-positive decode rate, outputTokens < 1, a negative queue or a missing model throws', () => {
  const ok = { promptTokens: 10, model: L, outputTokens: 5, decodeTokPerS: 50 };
  assert.throws(() => requestTimeline({ ...ok, decodeTokPerS: 0 }), /decodeTokPerS must be a positive finite number/);
  assert.throws(() => requestTimeline({ ...ok, outputTokens: 0 }), /outputTokens must be a positive integer/);
  assert.throws(() => requestTimeline({ ...ok, outputTokens: 1.5 }), RangeError);
  assert.throws(() => requestTimeline({ ...ok, queueS: -1 }), /queueS/);
  assert.throws(() => requestTimeline({ ...ok, promptTokens: 0 }), /promptTokens/);
  assert.throws(() => requestTimeline({ ...ok, model: undefined }), /model must be a step-time model object/);
});

// ---- kvTransferTime (disaggregation §6) ----

test('kvTransferTime: 4,096 tokens over 50 / 100 / 900 GB/s and FP8 KV (disaggregation §6)', () => {
  const prefillMs = prefill(L, 4096).timeS * 1e3;
  assert.equal(printed(prefillMs, 1), 289.8);
  const t = kvTransferTime(4096, 327680, 50e9);
  assert.equal(printed(t, 5), 0.02684);
  assert.equal(printed(kvCacheBytes({ bytesPerToken: 327680, tokens: 4096 }) / 1e9, 3), 1.342);
  assert.equal(printed((t * 1e3 * 100) / prefillMs, 2), 9.26);
  assert.equal(printed(kvTransferTime(4096, 327680, 100e9) * 1e3, 2), 13.42);
  assert.equal(printed((kvTransferTime(4096, 327680, 100e9) * 1e3 * 100) / prefillMs, 2), 4.63);
  assert.equal(printed(kvTransferTime(4096, 327680, 0.9e12) * 1e3, 2), 1.49);
  assert.equal(printed((kvTransferTime(4096, 327680, 0.9e12) * 1e3 * 100) / prefillMs, 2), 0.51);
  assert.equal(printed(kvTransferTime(4096, 163840, 50e9) * 1e3, 2), 13.42);
});

test('kvTransferTime: the ratio to prefill is constant above the crossover, smaller below it', () => {
  const ratio = (n) => kvTransferTime(n, 327680, 50e9) / prefill(L, n).timeS;
  const rows = [[512, 3.36, 36.2], [131072, 858.99, 9272.4]];
  for (const [n, transferMs, prefillMs] of rows) {
    assert.equal(printed(kvTransferTime(n, 327680, 50e9) * 1e3, 2), transferMs);
    assert.equal(printed(prefill(L, n).timeS * 1e3, 1), prefillMs);
    assert.equal(printed(ratio(n) * 100, 2), 9.26);
  }
  const below = [[100, 0.66, 14.94, 4.39], [128, 0.84, 15.04, 5.58], [200, 1.31, 15.3, 8.57], [217, 1.42, 15.36, 9.26]];
  for (const [n, transferMs, prefillMs, pct] of below) {
    assert.equal(printed(kvTransferTime(n, 327680, 50e9) * 1e3, 2), transferMs, `${n}: transfer`);
    assert.equal(printed(prefill(L, n).timeS * 1e3, 2), prefillMs, `${n}: prefill`);
    assert.equal(printed(ratio(n) * 100, 2), pct, `${n}: share`);
  }
  assert.ok(ratio(100) < ratio(217));
});

test('kvTransferTime: linear in the prompt and in the KV bytes; MLA example', () => {
  assert.equal(kvTransferTime(2000, 1000, 1e9), 2 * kvTransferTime(1000, 1000, 1e9));
  assert.equal(kvTransferTime(1000, 4000, 1e9), 2 * kvTransferTime(1000, 2000, 1e9));
  const mla = kvBytesPerTokenMla({ layers: 61, dLatent: 512, dRope: 64, bytesPerElem: 2 });
  assert.equal(mla, 70272);
  assert.equal(printed((mla * 4096) / 1e9, 3), 0.288);
  assert.equal(printed(kvTransferTime(4096, mla, 50e9) * 1e3, 2), 5.76);
  assert.throws(() => kvTransferTime(0, 1, 1), /promptTokens/);
  assert.throws(() => kvTransferTime(1, 1, 0), /linkBytesPerSecond/);
});

// ---- tokensPerExpert ----

test('tokensPerExpert: 64 users per GPU on V4-Pro (6 of 384) at EP 1 / 8 / 16 / 32 / 72, with the expert intensity', () => {
  const tokens = (epSize, usersPerGpu = 64) => tokensPerExpert({ usersPerGpu, epSize, expertsActive: 6, expertsTotal: 384 });
  const intensity = (m) => printed(arithmeticIntensity(matmulCost({ m, k: 7168, n: 3072, bytesPerElem: 0.5625 })), 1);
  const rows = [[1, 1, 3.6], [8, 8, 28.3], [16, 16, 56.5], [32, 32, 112.1], [72, 72, 247.7]];
  for (const [ep, expected, expectedIntensity] of rows) {
    assert.equal(tokens(ep), expected);
    assert.equal(intensity(tokens(ep)), expectedIntensity, `EP ${ep}`);
  }
  assert.equal(tokens(72, 256), 288);
  assert.equal(intensity(288), 903.1);
  assert.equal(tokensPerExpert({ usersPerGpu: 16, epSize: 16, expertsActive: 6, expertsTotal: 384 }), 4);
});

test('tokensPerExpert: DeepSeek decode at EP 144 — (256 + 32) / 144 = 2 experts per GPU, 72 tokens per expert; EP 1 is users · k / E', () => {
  assert.equal((256 + 32) / 144, 2);
  assert.equal(tokensPerExpert({ usersPerGpu: 16, epSize: 144, expertsActive: 8, expertsTotal: 256 }), 72);
  for (const [users, k, e] of [[7, 6, 384], [64, 8, 256], [1, 2, 8]]) {
    assert.equal(tokensPerExpert({ usersPerGpu: users, epSize: 1, expertsActive: k, expertsTotal: e }), (users * k) / e);
  }
  assert.throws(() => tokensPerExpert({ usersPerGpu: 1, epSize: 0, expertsActive: 1, expertsTotal: 1 }), /epSize/);
});

// ---- minGpusForWeights, usersAtTarget, costPerMillion (serving-calculator §6) ----

test('minGpusForWeights: ⌈weights / hbm⌉', () => {
  assert.deepEqual(
    [[865e9, 288e9], [865e9, 186e9], [3.2e12, 288e9], [1.6e12, 288e9], [0.9e12, 288e9], [288e9, 288e9]].map(([w, h]) => minGpusForWeights(w, h)),
    [4, 5, 12, 6, 4, 1],
  );
  assert.throws(() => minGpusForWeights(0, 1), /weightBytes/);
  assert.throws(() => minGpusForWeights(1, 0), /hbmBytes/);
});

test('serving-calculator: KV per user and users that fit — both KV ends, three contexts (V4-Pro on 16 GB300 GPUs)', () => {
  const free = freeHbmPerGpu({ hbmBytes: 288e9, weightBytes: 865e9, gpus: 16 });
  const row = (bytesPerToken, tokens) => {
    const perUser = kvCacheBytes({ bytesPerToken, tokens });
    return [formatBytes(perUser), maxUsersPerGpu(free, perUser)];
  };
  assert.deepEqual(row(4000, 9216), ['36.9 MB', 6345]);
  assert.deepEqual(row(12000, 9216), ['111 MB', 2115]);
  assert.deepEqual(row(4000, 139264), ['557 MB', 419]);
  assert.deepEqual(row(12000, 139264), ['1.67 GB', 139]);
  assert.deepEqual(row(4000, 1e6), ['4 GB', 58]);
  assert.deepEqual(row(12000, 1e6), ['12 GB', 19]);
  const gb200 = freeHbmPerGpu({ hbmBytes: 186e9, weightBytes: 865e9, gpus: 16 });
  assert.equal(maxUsersPerGpu(gb200, kvCacheBytes({ bytesPerToken: 4000, tokens: 1e6 })), 32);
});

test('serving-calculator: speed floor at 8K context — 1 / 64 / 256 / 1,024 users', () => {
  const rows = [[1, 6.76, 147.8, null], [64, 7.16, 139.6, 8936], [256, 8.37, 119.4, 30567], [1024, 20.07, 49.8, 51020]];
  for (const [users, ms, perUser, perGpu] of rows) {
    const s = decode(V, users, 9216, { kvBytesPerToken: 4000 });
    assert.equal(printed(s.timeS * 1e3, 2), ms, `${users} users`);
    assert.equal(printed(1 / s.timeS, 1), perUser, `${users} users: per user`);
    if (perGpu) assert.equal(Math.round(users / s.timeS), perGpu, `${users} users: per GPU`);
  }
  assert.equal(decode(V, 1024, 9216, { kvBytesPerToken: 4000 }).bound, 'compute');
});

test('serving-calculator: speed floor at the most users that fit — 8K, 128K, 1M, and the high KV end', () => {
  const at = (users, context, kvBytesPerToken) => decode(V, users, context, { kvBytesPerToken });
  const eight = at(6345, 9216, 4000);
  assert.equal(printed(eight.timeS * 1e3, 2), 124.36);
  assert.equal(eight.bound, 'compute');
  const k128 = at(419, 139264, 4000);
  assert.deepEqual([printed(k128.timeS * 1e3, 2), printed(1 / k128.timeS, 1), Math.round(419 / k128.timeS)], [36.65, 27.3, 11433]);
  const m1 = at(58, 1e6, 4000);
  assert.deepEqual([printed(m1.timeS * 1e3, 2), printed(1 / m1.timeS, 1), Math.round(58 / m1.timeS)], [35.86, 27.9, 1618]);
  const high = at(19, 1e6, 12000);
  assert.deepEqual([printed(high.timeS * 1e3, 2), printed(1 / high.timeS, 1), Math.round(19 / high.timeS)], [35.29, 28.3, 538]);
});

test('serving-calculator: the weight-read floor against the measured 27 tok/s is 5.48×', () => {
  const floorMs = (V.weightBytesPerGpu / (V.bandwidthTBps * 1e12)) * 1e3;
  assert.equal(printed(floorMs, 2), 6.76);
  assert.equal(printed(1000 / 27, 2), 37.04);
  assert.equal(printed(1000 / 27 / floorMs, 2), 5.48);
  assert.equal(GB(V.weightBytesPerGpu), 54.06);
});

test('usersAtTarget: 27 tok/s per user at 8K — 1,889 users, byCompute 1,889, byMem 4,793, step 37.02 ms, 51,020 tok/s/GPU', () => {
  const r = usersAtTarget({ targetTokPerUser: 27, ...V, context: 9216, kvBytesPerToken: 4000, maxUsers: 6345 });
  assert.deepEqual(r, { users: 1889, byMem: 4793, byCompute: 1889, maxUsers: 6345, limit: 'compute' });
  const s = decode(V, r.users, 9216, { kvBytesPerToken: 4000 });
  assert.equal(printed(s.timeS * 1e3, 2), 37.02);
  assert.equal(Math.round(r.users / s.timeS), 51020);
});

test('usersAtTarget: 128K and 1M are capped by memory capacity; FP4 math is bandwidth-limited; a 13.1 target is compute-limited', () => {
  const base = { targetTokPerUser: 27, ...V, kvBytesPerToken: 4000 };
  const k128 = usersAtTarget({ ...base, context: 139264, maxUsers: 419 });
  assert.deepEqual([k128.users, k128.limit], [419, 'memory capacity']);
  const m1 = usersAtTarget({ ...base, context: 1e6, maxUsers: 58 });
  assert.deepEqual([m1.users, m1.limit], [58, 'memory capacity']);
  const fp4 = usersAtTarget({ ...base, peakTflops: 15000, context: 9216, maxUsers: 6345 });
  assert.deepEqual([fp4.users, fp4.limit], [4793, 'bandwidth']);
  const t131 = usersAtTarget({ ...base, targetTokPerUser: 13.1, context: 9216, maxUsers: 6345 });
  assert.deepEqual([t131.users, t131.limit], [3894, 'compute']);
});

test('usersAtTarget properties: never above maxUsers; the step meets the target at `users` and misses it at users + 1', () => {
  for (const [target, context, maxUsers] of [[27, 9216, 6345], [13.1, 9216, 6345], [27, 139264, 419], [40, 9216, 6345], [60, 9216, 6345]]) {
    const r = usersAtTarget({ targetTokPerUser: target, ...V, context, kvBytesPerToken: 4000, maxUsers });
    assert.ok(r.users <= maxUsers);
    if (r.users > 0) assert.ok(decode(V, r.users, context, { kvBytesPerToken: 4000 }).timeS <= 1 / target, `${target}/${context}: meets`);
    if (r.limit !== 'memory capacity') assert.ok(decode(V, r.users + 1, context, { kvBytesPerToken: 4000 }).timeS > 1 / target, `${target}/${context}: misses at +1`);
  }
});

test('usersAtTarget: a target no user can meet gives 0 users, never a negative count; tie goes to compute', () => {
  const unreachable = usersAtTarget({ targetTokPerUser: 500, ...V, context: 9216, kvBytesPerToken: 4000, maxUsers: 100 });
  assert.equal(unreachable.users, 0);
  assert.equal(unreachable.byMem, 0);
  assert.ok(unreachable.byCompute >= 0);
  const tie = usersAtTarget({ targetTokPerUser: 27, ...V, context: 9216, kvBytesPerToken: 0, maxUsers: 1e9, peakTflops: 5000 });
  assert.equal(tie.limit, tie.byCompute <= tie.byMem ? 'compute' : 'bandwidth');
});

test('usersAtTarget: errors name the argument', () => {
  const ok = { targetTokPerUser: 27, ...V, context: 9216, kvBytesPerToken: 4000, maxUsers: 10 };
  for (const [key, bad] of [['targetTokPerUser', 0], ['maxUsers', -1], ['maxUsers', 1.5], ['context', -1], ['peakTflops', 0], ['dModel', 0]]) {
    assert.throws(() => usersAtTarget({ ...ok, [key]: bad }), new RegExp(`^RangeError: usersAtTarget: ${key} must be`), key);
  }
});

test('costPerMillion: $ / (tok/s · 3600) · 1e6 — the InferenceX anchors and the floor', () => {
  assert.equal(printed(costPerMillion(2.65, 6182), 4), 0.1191);
  assert.equal(printed(costPerMillion(2.65, 6182), 2), 0.12); // InferenceX's $0.12
  assert.equal(printed(costPerMillion(2.21, 2189), 4), 0.2804);
  assert.equal(printed(costPerMillion(2.65, 11056), 4), 0.0666);
  assert.equal(printed(costPerMillion(2.65, 51020), 4), 0.0144);
  // the storyboard's "(2.65, 148) → 4.979" feeds the unrounded one-user rate (147.84 tok/s), not the printed 148
  assert.equal(printed(costPerMillion(2.65, 1 / decode(V, 1, 9216, { kvBytesPerToken: 4000 }).timeS), 3), 4.979);
  assert.throws(() => costPerMillion(-1, 10), /dollarsPerGpuHour/);
  assert.throws(() => costPerMillion(1, 0), /tokPerSecPerGpu must be a positive finite number/);
});

test('serving-calculator: the measured-vs-floor gaps and the input side', () => {
  assert.equal(printed(6182 / 9, 1), 686.9);
  assert.equal(printed(51020 / (6182 / 9), 1), 74.3);
  assert.equal(printed(51020 / 6182, 2), 8.25);
  const ceiling = prefillTokPerSecCeiling({ activeParamsPerGpu: 49e9, peakTflops: 5000 });
  assert.equal(printed(ceiling / 11433, 2), 4.46);
  assert.equal(printed(ceiling / 1618, 1), 31.5);
});

test('serving-calculator: the Llama-3.1-70B preset on one H200 (FP8, 9,216 tokens) and at 131,072 tokens', () => {
  const cache = (tokens) => kvCacheBytes({ bytesPerToken: 327680, tokens });
  const free = freeHbmPerGpu({ hbmBytes: 141e9, weightBytes: L.weightBytesPerGpu, gpus: 1 });
  const users = maxUsersPerGpu(free, cache(9216));
  assert.equal(users, 23);
  const s = decode(L, users, 9216);
  assert.equal(printed(s.timeS * 1e3, 2), 29.14);
  assert.equal(printed(1 / s.timeS, 1), 34.3);
  const perGpu = users / s.timeS;
  assert.equal(Math.round(perGpu), 789);
  assert.equal(printed(costPerMillion(2.65, perGpu), 3), 0.932);
  const big = maxUsersPerGpu(free, cache(131072));
  assert.equal(big, 1);
  assert.equal(Math.round(big / decode(L, big, 131072).timeS), 42);
});

test('no function mutates its inputs', () => {
  const model = Object.freeze({ ...L });
  const args = Object.freeze({ ...model, tokens: 4, seqs: 2, context: 100 });
  assert.doesNotThrow(() => stepTime(args));
  assert.doesNotThrow(() => usersAtTarget(Object.freeze({ targetTokPerUser: 27, ...V, context: 9216, kvBytesPerToken: 4000, maxUsers: 6345 })));
  assert.doesNotThrow(() => requestTimeline(Object.freeze({ promptTokens: 10, model, outputTokens: 3, decodeTokPerS: 50 })));
  assert.deepEqual(stepTime(args), stepTime(args));
});
