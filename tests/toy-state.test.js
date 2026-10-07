import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createToyState } from '../shared/ui/toy-state.js';

test('renders once at creation and once per set, with frozen states', () => {
  const seen = [];
  const toy = createToyState({ layers: 2, experts: 0 }, (s) => seen.push(s));
  toy.set({ experts: 8 });
  toy.set((s) => ({ layers: s.layers + 1 }));
  assert.deepEqual(seen.map((s) => [s.layers, s.experts]), [[2, 0], [2, 8], [3, 8]]);
  assert.ok(seen.every(Object.isFrozen));
  assert.notEqual(seen[0], seen[1]);
});

test('destroy stops renders: a toy left mid-interaction never paints again', () => {
  let renders = 0;
  const toy = createToyState({ n: 1 }, () => { renders += 1; });
  toy.destroy();
  toy.set({ n: 2 });
  assert.equal(renders, 1);
  assert.equal(toy.get().n, 1);
});

test('the initial object is copied, never mutated', () => {
  const initial = { n: 1 };
  createToyState(initial, () => {}).set({ n: 5 });
  assert.deepEqual(initial, { n: 1 });
  assert.throws(() => createToyState({}, null), /render must be a function/);
});
