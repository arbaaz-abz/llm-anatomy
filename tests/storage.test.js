import { test } from 'node:test';
import assert from 'node:assert/strict';
import { safeStorage, toggleLearned, LEARNED_KEY } from '../shared/storage.js';

const memoryStorage = () => {
  const map = new Map();
  return { getItem: (k) => (map.has(k) ? map.get(k) : null), setItem: (k, v) => map.set(k, String(v)) };
};

test('round-trips JSON values', () => {
  const store = safeStorage(memoryStorage);
  assert.equal(store.set('k', { a: 1 }), true);
  assert.deepEqual(store.get('k', null), { a: 1 });
  assert.equal(store.get('missing', 'fallback'), 'fallback');
});

test('works when storage throws (private window, blocked data)', () => {
  const throwingGetter = safeStorage(() => { throw new Error('SecurityError'); });
  assert.equal(throwingGetter.get('k', 'fb'), 'fb');
  assert.equal(throwingGetter.set('k', 1), false);
  const throwingCalls = safeStorage(() => ({
    getItem: () => { throw new Error('denied'); }, setItem: () => { throw new Error('quota'); },
  }));
  assert.equal(throwingCalls.get('k', 'fb'), 'fb');
  assert.equal(throwingCalls.set('k', 1), false);
});

test('corrupt JSON falls back', () => {
  const store = safeStorage(() => ({ getItem: () => '{not json', setItem: () => {} }));
  assert.equal(store.get('k', 'fb'), 'fb');
});

test('toggleLearned adds and removes, sorted, without mutating', () => {
  const store = safeStorage(memoryStorage);
  assert.deepEqual(toggleLearned(store, 'rope'), ['rope']);
  assert.deepEqual(toggleLearned(store, 'attention'), ['attention', 'rope']);
  assert.deepEqual(toggleLearned(store, 'rope'), ['attention']);
  assert.deepEqual(store.get(LEARNED_KEY, []), ['attention']);
});

test('toggleLearned reports the true state when storage is blocked', () => {
  const store = safeStorage(() => { throw new Error('SecurityError'); });
  assert.deepEqual(toggleLearned(store, 'rope'), []);
  assert.deepEqual(toggleLearned(store, 'rope'), []);
});

test('toggleLearned treats non-array stored values as empty', () => {
  for (const stored of ['null', '"rope"', '{}', '5']) {
    const store = safeStorage(() => ({ getItem: () => stored, setItem: () => {} }));
    assert.deepEqual(toggleLearned(store, 'rope'), ['rope'], stored);
  }
});
