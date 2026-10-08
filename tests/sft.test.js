import { test } from 'node:test';
import assert from 'node:assert/strict';
import { lossMask, maskSummary, MASK_DEFAULTS } from '../math/sft.js';
import { meanLoss } from '../math/lm.js';

// The storyboard's 26-token transcript (sft §5), whitespace-split, in eight segments.
const SEG = [
  { kind: 'template', tokens: ['<user>'] },
  { kind: 'user', tokens: ['What', 'is', '7', '×', '8', '?'] },
  { kind: 'template', tokens: ['<assistant>'] },
  { kind: 'assistant', tokens: ['<think>'] },
  { kind: 'error', tokens: ['7', '×', '8', '=', '54'] },
  { kind: 'assistant', tokens: ['wait', ',', 'check', '</think>', '<call>', 'calc(7*8)', '</call>'] },
  { kind: 'observation', tokens: ['<obs>', '56', '</obs>'] },
  { kind: 'assistant', tokens: ['56', '<end>'] },
];
const bits = (mask) => mask.map((b) => (b ? 1 : 0)).join('');

test('lossMask: the default mask is the storyboard string, 26 entries, 15 true', () => {
  const mask = lossMask(SEG);
  assert.equal(mask.length, 26);
  assert.equal(bits(mask), '00000000111111111111100011');
  assert.equal(mask.filter(Boolean).length, 15);
});

test('lossMask: the short example from the signature', () => {
  assert.deepEqual(lossMask([{ kind: 'user', tokens: ['hi'] }, { kind: 'assistant', tokens: ['hello'] }]), [false, true]);
});

test('lossMask: each option flips exactly its own kinds', () => {
  assert.equal(bits(lossMask(SEG, { maskPrompt: false })), '11111111111111111111100011');
  assert.equal(bits(lossMask(SEG, { maskObservation: false })), '00000000111111111111111111');
  assert.equal(bits(lossMask(SEG, { maskError: true })), '00000000100000111111100011');
});

test('lossMask: assistant tokens are trained under every option', () => {
  const kinds = SEG.flatMap((s) => s.tokens.map(() => s.kind));
  for (const maskPrompt of [true, false]) for (const maskObservation of [true, false]) for (const maskError of [true, false]) {
    const mask = lossMask(SEG, { maskPrompt, maskObservation, maskError });
    kinds.forEach((k, i) => { if (k === 'assistant') assert.equal(mask[i], true); });
  }
});

test('lossMask: an unknown kind and bad arguments throw RangeError', () => {
  assert.throws(() => lossMask([{ kind: 'system', tokens: ['x'] }]), RangeError);
  assert.throws(() => lossMask('nope'), /lossMask: segments must be an array/);
  assert.throws(() => lossMask([{ kind: 'user', tokens: 'hi' }]), /lossMask: segments\[0\]\.tokens must be an array of strings/);
  assert.throws(() => lossMask(SEG, { maskPrompt: 1 }), /lossMask: maskPrompt must be true or false/);
  assert.throws(() => lossMask(SEG, null), /lossMask: options must be an object/);
});

test('maskSummary: the default state, 15 of 26, per-kind trained/masked', () => {
  const s = maskSummary(SEG);
  assert.deepEqual([s.total, s.trained, s.masked], [26, 15, 11]);
  assert.deepEqual(s.byKind, {
    template: { trained: 0, masked: 2 },
    user: { trained: 0, masked: 6 },
    assistant: { trained: 10, masked: 0 },
    error: { trained: 5, masked: 0 },
    observation: { trained: 0, masked: 3 },
  });
  assert.equal((100 * s.trained / s.total).toFixed(1), '57.7');
});

test('maskSummary: the storyboard reproducer for every try-this state', () => {
  const at = (o) => { const s = maskSummary(SEG, o); return [s.total, s.trained, s.masked, (100 * s.trained / s.total).toFixed(1)]; };
  assert.deepEqual(at({}), [26, 15, 11, '57.7']);
  assert.deepEqual(at({ maskPrompt: false }), [26, 23, 3, '88.5']);
  assert.deepEqual(at({ maskObservation: false }), [26, 18, 8, '69.2']);
  assert.deepEqual(at({ maskError: true }), [26, 10, 16, '38.5']);
  assert.deepEqual(maskSummary(SEG, { maskError: true }).byKind.error, { trained: 0, masked: 5 });
});

test('maskSummary: trained + masked = total, and unmasking never lowers trained', () => {
  for (const key of ['maskPrompt', 'maskObservation']) {
    const masked = maskSummary(SEG, { [key]: true });
    const open = maskSummary(SEG, { [key]: false });
    assert.equal(open.trained + open.masked, open.total);
    assert.ok(open.trained >= masked.trained);
  }
  const keepError = maskSummary(SEG, { maskError: false });
  assert.ok(keepError.trained >= maskSummary(SEG, { maskError: true }).trained);
});

test('maskSummary: a kind with no tokens still appears with zero counts', () => {
  const s = maskSummary([{ kind: 'assistant', tokens: ['a', 'b'] }]);
  assert.deepEqual(s.byKind.user, { trained: 0, masked: 0 });
  assert.deepEqual([s.total, s.trained, s.masked], [2, 2, 0]);
});

test('the mask feeds meanLoss from lm.js: only trained tokens count', () => {
  const probs = SEG.flatMap((s) => s.tokens.map(() => 0.5));
  assert.ok(Math.abs(meanLoss(probs, lossMask(SEG)) - Math.log(2)) < 1e-12);
});

test('defaults are exported frozen, and inputs are never mutated', () => {
  assert.deepEqual({ ...MASK_DEFAULTS }, { maskPrompt: true, maskObservation: true, maskError: false });
  assert.ok(Object.isFrozen(MASK_DEFAULTS));
  const before = JSON.stringify(SEG);
  const opts = Object.freeze({ maskError: true });
  lossMask(SEG, opts);
  maskSummary(SEG, opts);
  assert.equal(JSON.stringify(SEG), before);
});
