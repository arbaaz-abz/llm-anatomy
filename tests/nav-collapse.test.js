import { test } from 'node:test';
import assert from 'node:assert/strict';
import { NAV_KEY, createNavState } from '../shared/ui/nav-collapse.js';
import { safeStorage } from '../shared/storage.js';

const memory = () => {
  const map = new Map();
  return { getItem: (k) => map.get(k) ?? null, setItem: (k, v) => map.set(k, v), map };
};

test('the lesson list starts shown when nothing is saved', () => {
  assert.equal(createNavState(safeStorage(() => memory())).collapsed(), false);
});

test('toggle flips the state and saves it under NAV_KEY', () => {
  const raw = memory();
  const state = createNavState(safeStorage(() => raw));
  assert.equal(state.toggle(), true);
  assert.equal(raw.map.get(NAV_KEY), 'true');
  assert.equal(state.toggle(), false);
  assert.equal(raw.map.get(NAV_KEY), 'false');
});

test('a saved collapsed choice is restored by the next page load', () => {
  const raw = memory();
  createNavState(safeStorage(() => raw)).toggle();
  assert.equal(createNavState(safeStorage(() => raw)).collapsed(), true);
});

test('junk in storage reads as shown, and storage that throws still toggles in memory', () => {
  const junk = memory();
  junk.setItem(NAV_KEY, '"yes"');
  assert.equal(createNavState(safeStorage(() => junk)).collapsed(), false);
  const broken = createNavState(safeStorage(() => { throw new Error('blocked'); }));
  assert.equal(broken.collapsed(), false);
  assert.equal(broken.toggle(), true);
});
