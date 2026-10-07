import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const css = await readFile(new URL('../shared/theme.css', import.meta.url), 'utf8');
// The three theme blocks: light :root, the OS-dark media block, the explicit dark toggle.
// Computed inside the tests, so a missing marker fails one named test instead of crashing the file.
const MARKERS = { light: ':root {', osDark: ':root:not([data-theme="light"]) {', dark: ':root[data-theme="dark"] {' };
const block = (marker) => {
  const start = css.indexOf(marker);
  return start < 0 ? null : css.slice(start, css.indexOf('}', start));
};
const blocks = () => Object.fromEntries(Object.entries(MARKERS).map(([name, marker]) => [name, block(marker)]));
const SURFACE = { light: '#ffffff', osDark: '#1a1f28', dark: '#1a1f28' };
const tokens = (text, name) => Object.fromEntries([...text.matchAll(new RegExp(`--(${name}-\\d): (#[0-9a-f]{6})`, 'g'))].map((m) => [m[1], m[2]]));

const luminance = (hex) => {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255)
    .map((c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};
const contrast = (a, b) => { const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x); return (hi + 0.05) / (lo + 0.05); };

test('theme.css has the three theme blocks', () => {
  for (const [name, text] of Object.entries(blocks())) assert.ok(text, `theme.css is missing the ${name} block (${MARKERS[name]})`);
});

test('--part-1 … --part-5 are defined in all three theme blocks', () => {
  const b = blocks();
  for (const [name, text] of Object.entries(b)) assert.deepEqual(Object.keys(tokens(text ?? '', 'part')).sort(), ['part-1', 'part-2', 'part-3', 'part-4', 'part-5'], name);
  assert.deepEqual(tokens(b.osDark ?? '', 'part'), tokens(b.dark ?? '', 'part'), 'both dark blocks carry the same values');
});

test('every part hue clears 3:1 against its theme surface (WCAG 1.4.11)', () => {
  for (const [name, text] of Object.entries(blocks())) {
    for (const [token, hex] of Object.entries(tokens(text ?? '', 'part'))) assert.ok(contrast(hex, SURFACE[name]) >= 3, `${name} ${token} ${hex}: ${contrast(hex, SURFACE[name]).toFixed(2)}:1`);
  }
});

test('part hues are their own tokens, never a request-identity color', () => {
  for (const text of Object.values(blocks())) {
    const parts = new Set(Object.values(tokens(text ?? '', 'part')));
    for (const hex of Object.values(tokens(text ?? '', 'req'))) assert.ok(!parts.has(hex), `${hex} is both a part and a request color`);
  }
});

test('the "not published" part has its own neutral token in the light block', () => {
  assert.match(blocks().light ?? '', /--part-none: var\(--surface-2\);/);
});

test('the activation carry is the Architecture accent in every track (ruling 2026-10-07)', () => {
  assert.match(blocks().light ?? '', /--carry-activation: var\(--accent-arch\);/);
});
