// Optional glyph options added by shared patch S0 (wave 1a review): `format` (shared-1),
// heatmap `hatch` (shared-2) and blockStack `lastLabel` (shared-3). Every option is optional,
// and leaving it out reproduces the pre-S0 output byte for byte (pinned below).
import { test, expect } from '@playwright/test';
import { BASELINE } from './fixtures/glyph-baseline.js';

// Draws one glyph into a fresh <svg> with a fixed hatch id and returns its outerHTML.
function draw(page, name, opts) {
  return page.evaluate(async ({ glyph, o }) => {
    const G = await import('/shared/glyphs.js');
    const revive = (v) => (v === '-Infinity' ? -Infinity : Array.isArray(v) ? v.map(revive) : v);
    const parsed = Object.fromEntries(Object.entries(o).map(([k, v]) => [k, k === 'values' ? revive(v) : v]));
    if (o.formatName === 'fixed3') parsed.format = (v) => v.toFixed(3);
    if (o.formatName === 'cell') parsed.format = G.formatCell;
    if (o.formatName === 'masked') parsed.format = (v) => (v === -Infinity ? 'none' : v.toFixed(1));
    delete parsed.formatName;
    const svg = G.svgEl('svg', { width: 400, height: 300 }, document.body);
    svg.dataset.hatchId = 'g-hatch-test';
    const g = G[glyph](svg, parsed);
    const html = g.outerHTML;
    svg.remove();
    return html;
  }, { glyph: name, o: opts });
}

const CALLS = {
  vector: ['vector', { x: 0, y: 0, values: [0.095, -1.5, 0, 12.3], cell: 40, orient: 'row', label: 'v' }],
  matrix: ['matrix', { x: 20, y: 20, values: [[0.5, -0.25], [1, '-Infinity']], cell: 40, label: 'M', rowLabels: ['a', 'b'] }],
  heatmap: ['heatmap', { x: 20, y: 30, values: [[0.095, 0.703], ['-Infinity', 0.2]], mask: [[true, true], [false, true]], cell: 40, label: 'H', colLabels: ['x', 'y'] }],
  blockStack: ['blockStack', { x: 0, y: 0, count: 61 }],
  blockStackShort: ['blockStack', { x: 0, y: 0, count: 3 }],
};

test.beforeEach(async ({ page }) => { await page.goto('/gallery/'); });

test('defaults reproduce the pre-S0 output byte for byte', async ({ page }) => {
  for (const [key, [glyph, opts]] of Object.entries(CALLS)) {
    expect(await draw(page, glyph, opts), key).toBe(BASELINE[key]);
  }
});

test('format: formatCell passed explicitly prints the same cell text as the default', async ({ page }) => {
  const texts = (html) => [...html.matchAll(/<text[^>]*class="g-text"[^>]*>([^<]*)<\/text>/g)].map((m) => m[1]);
  for (const key of ['vector', 'matrix', 'heatmap']) {
    const [glyph, opts] = CALLS[key];
    expect(texts(await draw(page, glyph, { ...opts, formatName: 'cell' })), key).toEqual(texts(BASELINE[key]));
  }
});

const cellTexts = (page, glyph, opts) => page.evaluate(async ({ glyph: name, o }) => {
  const G = await import('/shared/glyphs.js');
  const svg = G.svgEl('svg', { width: 400, height: 300 }, document.body);
  const fmt = { fixed3: (v) => v.toFixed(3), masked: (v) => (v === -Infinity ? 'none' : v.toFixed(1)) }[o.formatName];
  const values = o.values.map((r) => (Array.isArray(r) ? r.map((v) => (v === '-Infinity' ? -Infinity : v)) : (r === '-Infinity' ? -Infinity : r)));
  const g = G[name](svg, { ...o, values, format: fmt, formatName: undefined });
  const out = [...g.querySelectorAll('.g-cell')].map((c) => ({
    text: c.querySelector('.g-text')?.textContent ?? null,
    title: c.querySelector('title').textContent,
    masked: c.classList.contains('g-cell--masked'),
    hatched: !!c.querySelector('.g-hatch'),
    level: c.dataset.level,
  }));
  svg.remove();
  return out;
}, { glyph, o: opts });

test('format: heatmap({ values: [[0.095]], format: toFixed(3) }) prints "0.095" in the cell and its title', async ({ page }) => {
  const [c] = await cellTexts(page, 'heatmap', { x: 0, y: 0, values: [[0.095]], cell: 40, formatName: 'fixed3' });
  expect(c).toMatchObject({ text: '0.095', title: '0.095' });
});

test('format: vector and matrix use it for every cell', async ({ page }) => {
  const v = await cellTexts(page, 'vector', { x: 0, y: 0, values: [0.703, -0.2], cell: 40, orient: 'row', formatName: 'fixed3' });
  expect(v.map((c) => c.text)).toEqual(['0.703', '-0.200']);
  const m = await cellTexts(page, 'matrix', { x: 0, y: 0, values: [[0.5], [0.25]], cell: 40, formatName: 'fixed3' });
  expect(m.map((c) => c.text)).toEqual(['0.500', '0.250']);
});

test('format: a masked cell keeps "−∞" unless the format handles it', async ({ page }) => {
  const kept = await cellTexts(page, 'heatmap', { x: 0, y: 0, values: [[0.5, '-Infinity']], mask: [[true, false]], cell: 40, formatName: 'fixed3' });
  expect(kept.map((c) => [c.text, c.title])).toEqual([['0.500', '0.500'], ['−∞', 'masked (−∞)']]);
  const handled = await cellTexts(page, 'heatmap', { x: 0, y: 0, values: [['-Infinity']], cell: 40, formatName: 'masked' });
  expect(handled.map((c) => [c.text, c.masked])).toEqual([['none', true]]);
});

test('hatch: a hatched cell prints format(v), keeps its value color, and draws the hatch; mask still hides', async ({ page }) => {
  const cells = await cellTexts(page, 'heatmap', {
    x: 0, y: 0, values: [[0.095, 0.703, 0.202, 0], [0.5, '-Infinity', 0, 0]], cell: 40, formatName: 'fixed3',
    hatch: [[false, false, false, true], [false, false, true, false]],
    mask: [[true, true, true, true], [true, false, true, true]],
  });
  expect(cells.map((c) => c.text)).toEqual(['0.095', '0.703', '0.202', '0.000', '0.500', '−∞', '0.000', '0.000']);
  expect(cells.map((c) => c.hatched)).toEqual([false, false, false, true, false, true, true, false]);
  expect(cells.map((c) => c.masked)).toEqual([false, false, false, false, false, true, false, false]);
  expect(cells[3].title).toBe('0.000');
});

