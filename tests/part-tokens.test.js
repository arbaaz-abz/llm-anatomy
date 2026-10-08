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

// OKLab ΔE ×100 (Björn Ottosson's OKLab), the same metric the dataviz palette validator reports.
const srgbLinear = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const oklab = (hex) => {
  const [r, g, b] = [1, 3, 5].map((i) => srgbLinear(parseInt(hex.slice(i, i + 2), 16) / 255));
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s, 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s, 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s];
};
const deltaE = (a, b) => { const [x, y] = [oklab(a), oklab(b)]; return 100 * Math.hypot(x[0] - y[0], x[1] - y[1], x[2] - y[2]); };
const ACCENT_MIN_DELTA_E = 15;
const accents = (text) => Object.fromEntries([...text.matchAll(/--(accent-(?:arch|train|serve)): (#[0-9a-f]{6})/g)].map((m) => [m[1], m[2]]));

// --part-4 ("other") is the violet-adjacent slot; on an Architecture page it must never read as the accent (selection).
test('--part-4 stays clear of all three track accents in every theme (OKLab ΔE ≥ 15)', () => {
  for (const [name, text] of Object.entries(blocks())) {
    const acc = accents(text ?? '');
    assert.deepEqual(Object.keys(acc).sort(), ['accent-arch', 'accent-serve', 'accent-train'], `${name} block defines the three accents`);
    for (const [token, hex] of Object.entries(tokens(text ?? '', 'part')).filter(([t]) => t === 'part-4')) {
      for (const [accent, ahex] of Object.entries(acc)) {
        assert.ok(deltaE(hex, ahex) >= ACCENT_MIN_DELTA_E, `${name} ${token} ${hex} vs ${accent} ${ahex}: ΔE ${deltaE(hex, ahex).toFixed(1)}`);
      }
    }
  }
});

test('the "not published" segment is outlined in --ink-muted (≥ 3:1 in both themes), never a faint --line stroke', () => {
  assert.match(css, /\.g-share \.g-part-none \{ fill: var\(--part-none\); stroke: var\(--ink-muted\); stroke-width: 1; \}/);
});
