import { test } from 'node:test';
import assert from 'node:assert/strict';
import { THEME_KEY, resolveTheme, createTheme } from '../shared/ui/theme-toggle.js';
import { safeStorage } from '../shared/storage.js';

const memory = () => {
  const map = new Map();
  return { getItem: (k) => map.get(k) ?? null, setItem: (k, v) => map.set(k, v), map };
};
const fakeRoot = () => {
  const attrs = {};
  return { setAttribute: (k, v) => { attrs[k] = v; }, getAttribute: (k) => attrs[k] ?? null, attrs };
};

test('resolveTheme: a saved light or dark choice wins over the system preference', () => {
  assert.equal(resolveTheme('light', true), 'light');
  assert.equal(resolveTheme('dark', false), 'dark');
});

test('resolveTheme: with nothing (or junk) saved it follows the system preference', () => {
  assert.equal(resolveTheme(undefined, true), 'dark');
  assert.equal(resolveTheme(null, false), 'light');
  assert.equal(resolveTheme('sepia', true), 'dark');
});

test('first visit starts from the system preference and leaves the page attribute and storage alone', () => {
  const raw = memory();
  const root = fakeRoot();
  const theme = createTheme({ store: safeStorage(() => raw), root, prefersDark: () => true });
  assert.equal(theme.current(), 'dark');
  assert.equal(root.getAttribute('data-theme'), null);
  assert.equal(raw.map.size, 0);
});

test('a saved choice is applied to html[data-theme] on start', () => {
  const raw = memory();
  raw.setItem(THEME_KEY, JSON.stringify('light'));
  const root = fakeRoot();
  const theme = createTheme({ store: safeStorage(() => raw), root, prefersDark: () => true });
  assert.equal(theme.current(), 'light');
  assert.equal(root.getAttribute('data-theme'), 'light');
});

test('toggle flips light and dark, sets data-theme and persists the choice', () => {
  const raw = memory();
  const root = fakeRoot();
  const theme = createTheme({ store: safeStorage(() => raw), root, prefersDark: () => false });
  assert.equal(theme.toggle(), 'dark');
  assert.equal(root.getAttribute('data-theme'), 'dark');
  assert.equal(JSON.parse(raw.map.get(THEME_KEY)), 'dark');
  assert.equal(theme.toggle(), 'light');
  assert.equal(root.getAttribute('data-theme'), 'light');
  const again = createTheme({ store: safeStorage(() => raw), root: fakeRoot(), prefersDark: () => true });
  assert.equal(again.current(), 'light');
});

test('blocked storage: the toggle still works for this view', () => {
  const root = fakeRoot();
  const theme = createTheme({ store: safeStorage(() => { throw new Error('blocked'); }), root, prefersDark: () => false });
  assert.equal(theme.toggle(), 'dark');
  assert.equal(root.getAttribute('data-theme'), 'dark');
});
