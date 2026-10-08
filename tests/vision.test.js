import { test } from 'node:test';
import assert from 'node:assert/strict';
import { patchGrid, visionTokens } from '../math/vision.js';

// The worked examples are storyboard multimodal §6 (reproducer run 2026-10-07).

test('patchGrid: the toy image, a phone photo, the largest K3 input and a wide photo', () => {
  assert.deepEqual(patchGrid({ width: 16, height: 16, patch: 4 }), { cols: 4, rows: 4, patches: 16 });
  assert.deepEqual(patchGrid({ width: 1008, height: 1008, patch: 14 }), { cols: 72, rows: 72, patches: 5184 });
  assert.deepEqual(patchGrid({ width: 3584, height: 3584, patch: 14 }), { cols: 256, rows: 256, patches: 65536 });
  assert.deepEqual(patchGrid({ width: 1008, height: 504, patch: 14 }), { cols: 72, rows: 36, patches: 2592 });
});

test('patchGrid: a side that is not a multiple of the patch throws, naming the side', () => {
  assert.throws(() => patchGrid({ width: 1000, height: 1000, patch: 14 }), new RangeError('patchGrid: width must be a multiple of patch'));
  assert.throws(() => patchGrid({ width: 1008, height: 1000, patch: 14 }), new RangeError('patchGrid: height must be a multiple of patch'));
});

test('patchGrid: sizes must be positive integers', () => {
  assert.throws(() => patchGrid({ width: 0, height: 28, patch: 14 }), /width must be a positive integer/);
  assert.throws(() => patchGrid({ width: 28, height: 28.5, patch: 14 }), /height must be a positive integer/);
  assert.throws(() => patchGrid({ width: 28, height: 28, patch: 0 }), /patch must be a positive integer/);
  assert.throws(() => patchGrid({ width: 28, height: 28, patch: Number.NaN }), /patch must be a positive integer/);
});

test('visionTokens: the toy image gives 4 image tokens', () => {
  assert.deepEqual(visionTokens({ width: 16, height: 16, patch: 4, merge: 2 }), { patches: 16, tokensPerFrame: 4, tokens: 4 });
});

test('visionTokens: a 1,008-pixel photo is 5,184 patches and 1,296 tokens after a 2 × 2 merge', () => {
  assert.deepEqual(visionTokens({ width: 1008, height: 1008, patch: 14, merge: 2 }), { patches: 5184, tokensPerFrame: 1296, tokens: 1296 });
  assert.equal(visionTokens({ width: 1008, height: 1008, patch: 14, merge: 3 }).tokensPerFrame, 576);
  assert.equal(visionTokens({ width: 1008, height: 1008, patch: 14 }).tokensPerFrame, 5184, 'merge defaults to 1: no merge');
});

test('visionTokens: the largest Kimi K3 input and a wide photo', () => {
  assert.deepEqual(visionTokens({ width: 3584, height: 3584, patch: 14, merge: 2 }), { patches: 65536, tokensPerFrame: 16384, tokens: 16384 });
  const wide = visionTokens({ width: 1008, height: 504, patch: 14, merge: 2 });
  assert.equal(wide.patches, 2592);
  assert.equal(wide.tokensPerFrame, 648);
});

test('visionTokens: video pays per frame, tokens = tokensPerFrame × frames', () => {
  const base = { width: 448, height: 448, patch: 14, merge: 2 };
  assert.equal(visionTokens({ ...base, frames: 120 }).tokensPerFrame, 256);
  assert.equal(visionTokens({ ...base, frames: 120 }).tokens, 30720);
  assert.equal(visionTokens({ ...base, frames: 600 }).tokens, 153600);
  assert.equal(visionTokens({ ...base, frames: 7200 }).tokens, 1843200);
  [1, 3, 120].forEach((frames) => {
    const v = visionTokens({ ...base, frames });
    assert.equal(v.tokens, v.tokensPerFrame * frames);
  });
});

test('visionTokens: doubling both sides multiplies the tokens by 4 (tokens grow with area)', () => {
  const small = visionTokens({ width: 504, height: 504, patch: 14, merge: 2 }).tokens;
  const big = visionTokens({ width: 1008, height: 1008, patch: 14, merge: 2 }).tokens;
  assert.equal(big, 4 * small);
});

test('visionTokens: merge 2 divides the patch count by 4 and merge 3 by 9', () => {
  const size = { width: 1008, height: 1008, patch: 14 };
  const none = visionTokens(size).tokensPerFrame;
  assert.equal(visionTokens({ ...size, merge: 2 }).tokensPerFrame, none / 4);
  assert.equal(visionTokens({ ...size, merge: 3 }).tokensPerFrame, none / 9);
});

test('visionTokens: a grid that the merge does not divide throws, naming the axis', () => {
  assert.throws(() => visionTokens({ width: 1008, height: 1008, patch: 14, merge: 5 }), new RangeError('visionTokens: cols must be divisible by merge'));
  assert.throws(() => visionTokens({ width: 1008, height: 504, patch: 14, merge: 8 }), new RangeError('visionTokens: rows must be divisible by merge'));
});

test('visionTokens: merge and frames must be positive integers; patch errors come from patchGrid', () => {
  assert.throws(() => visionTokens({ width: 28, height: 28, patch: 14, merge: 0 }), /merge must be a positive integer/);
  assert.throws(() => visionTokens({ width: 28, height: 28, patch: 14, frames: 1.5 }), /frames must be a positive integer/);
  assert.throws(() => visionTokens({ width: 30, height: 28, patch: 14 }), /patchGrid: width must be a multiple of patch/);
});

test('inputs are never mutated', () => {
  const input = Object.freeze({ width: 1008, height: 504, patch: 14, merge: 2, frames: 3 });
  visionTokens(input);
  patchGrid(input);
  assert.deepEqual({ ...input }, { width: 1008, height: 504, patch: 14, merge: 2, frames: 3 });
});
