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


test('lastLabel: printed on the last drawn block only; the default stays `block ${count}`', async ({ page }) => {
  const labels = (count, lastLabel) => page.evaluate(async ({ n, last }) => {
    const G = await import('/shared/glyphs.js');
    const svg = G.svgEl('svg', { width: 400, height: 400 }, document.body);
    const g = G.blockStack(svg, { x: 0, y: 0, count: n, ...(last ? { lastLabel: last } : {}) });
    const out = [...g.querySelectorAll(':scope > .g-sub')].map((t) => t.textContent);
    svg.remove();
    return out;
  }, { n: count, last: lastLabel });
  expect(await labels(61)).toEqual(['block 1', 'block 2', 'block 61']);
  expect(await labels(61, 'block N')).toEqual(['block 1', 'block 2', 'block N']);
  expect(await labels(1, 'block N')).toEqual(['block N']);
  expect(await labels(3, 'block N')).toEqual(['block 1', 'block 2', 'block N']);
});

// ---- Shared prep S3 (Plan 3, Task 3): Training options on existing glyphs ----
// Leaving every S3 option out must reproduce the pre-S3 output byte for byte (Architecture pages on main
// draw these glyphs). Pinned as SHA-256 of outerHTML, captured at 64e85d1 before any S3 change.
import { createHash } from 'node:crypto';

const sha = (html) => createHash('sha256').update(html).digest('hex').slice(0, 16);
const PRE_S3 = {
  gpu: [{ x: 10, y: 6, memFill: 0.3, label: 'H100 · 30 %' }, 'bd29feac8e2b2a57'],
  gpuBare: [{ x: 0, y: 0 }, '2390321fb8501279'],
  rack: [{ x: 10, y: 6, gpus: 8, linkWidth: 4, label: 'HGX, 8 NVLink' }, '5d6d6612d43c2b18'],
  rack72: [{ x: 0, y: 0, gpus: 72, cols: 9 }, '5ff689f9c867e922'],
  flowActivation: [{ from: [0, 0], to: [120, 40], carry: 'activation', progress: 0.25 }, 'c897f1204c71450c'],
  flowGradient: [{ from: [0, 0], to: [120, 0], carry: 'gradient', progress: 0.5 }, 'fdf1dafda17f31d9'],
  vector: [{ x: 0, y: 0, values: [1, 0, 0.25, -0.5], cell: 40, orient: 'col', label: 'R' }, '9af5c50b3af02524'],
  shareBar: [{ x: 4, y: 6, w: 172, label: 'toy model', parts: [
    { name: 'embedding', value: 128, hue: 1 }, { name: 'attention', value: 512, hue: 2 }, { name: 'MLP', value: 768, hue: 3 },
    { name: 'other (norms)', value: 40, hue: 4 }, { name: 'head', value: 128, hue: 5 }, { name: 'not published', value: 60, unknown: true },
  ] }, '352382e1ebe83354'],
  shareBarPlain: [{ x: 0, y: 0, w: 300, parts: [{ name: 'a', value: 3, hue: 1 }, { name: 'b', value: 1, hue: 2 }], tail: 'none' }, '06a2840a9e65f74a'],
};
const GLYPH_OF = { gpuBare: 'gpu', rack72: 'rack', flowActivation: 'flow', flowGradient: 'flow', shareBarPlain: 'shareBar' };

test('S3: defaults of gpu, rack, flow, vector and shareBar reproduce the pre-S3 output byte for byte', async ({ page }) => {
  const got = {};
  for (const [key, [opts]] of Object.entries(PRE_S3)) got[key] = sha(await draw(page, GLYPH_OF[key] ?? key, opts));
  expect(got).toEqual(Object.fromEntries(Object.entries(PRE_S3).map(([k, [, h]]) => [k, h])));
});
