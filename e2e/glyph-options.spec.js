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
  vector: [{ x: 0, y: 0, values: [1, 0, 0.25, -0.5], cell: 40, orient: 'col', label: 'R' }, 'e14527294151166d'], // re-pinned (shared-8): each cell group now also carries the `glyph` class
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

// Draws a glyph and returns facts read from its DOM (computed styles included, so both themes can be checked).
const probe = (page, name, opts, read, theme = 'light') => page.evaluate(async ({ glyph, o, fn, th }) => {
  document.documentElement.dataset.theme = th;
  const G = await import('/shared/glyphs.js');
  const svg = G.svgEl('svg', { width: 600, height: 400 }, document.body);
  const g = G[glyph](svg, o);
  const out = new Function('g', 'G', `return (${fn})(g, G);`)(g, G);
  svg.remove();
  return out;
}, { glyph: name, o: opts, fn: read.toString(), th: theme });

for (const theme of ['light', 'dark']) {
  test(`S3 flow carry 'weight': its own dot color, distinct from every other carry (${theme})`, async ({ page }) => {
    const fills = {};
    for (const carry of ['activation', 'gradient', 'kv', 'token', 'weight']) {
      fills[carry] = await probe(page, 'flow', { from: [0, 0], to: [100, 0], carry, progress: 0.5 }, (g) => getComputedStyle(g.querySelector('.g-dot')).fill, theme);
    }
    expect(new Set(Object.values(fills)).size).toBe(5);
    expect(fills.weight).not.toBe('none');
  });
}

test('S3 flow rejects an unknown carry, naming weight among the choices', async ({ page }) => {
  const message = await page.evaluate(async () => {
    const G = await import('/shared/glyphs.js');
    const svg = G.svgEl('svg', {}, document.body);
    try { G.flow(svg, { from: [0, 0], to: [1, 1], carry: 'bytes' }); return null; } catch (e) { return e.message; } finally { svg.remove(); }
  });
  expect(message).toBe('glyphs.flow: carry must be one of activation, gradient, kv, token, weight');
});

test('S3 gpu: showMem: false drops the memory bar and its words; litSms lights the given SM tiles and titles the count', async ({ page }) => {
  const bare = await probe(page, 'gpu', { x: 0, y: 0, label: 'GPU 1', showMem: false }, (g) => ({ aria: g.getAttribute('aria-label'), track: g.querySelectorAll('.g-mem-track, .g-mem-fill').length, sms: g.querySelectorAll('.g-sm').length }));
  expect(bare).toEqual({ aria: 'GPU 1', track: 0, sms: 12 });
  const lit = await probe(page, 'gpu', { x: 0, y: 0, label: 'H100', litSms: [0, 5], showMem: false }, (g) => ({
    lit: [...g.querySelectorAll('.g-sm')].map((r, i) => (r.classList.contains('g-sm--lit') ? i : -1)).filter((i) => i >= 0),
    title: g.querySelector(':scope > title')?.textContent,
    fills: [...new Set([...g.querySelectorAll('.g-sm')].map((r) => getComputedStyle(r).fill))].length,
  }));
  expect(lit).toEqual({ lit: [0, 5], title: 'H100: SM grid, 2 of 12 tiles lit', fills: 2 });
  const all = await probe(page, 'gpu', { x: 0, y: 0, litSms: Array.from({ length: 12 }, (_, i) => i) }, (g) => g.querySelectorAll('.g-sm--lit').length);
  expect(all).toBe(12);
  const bad = await page.evaluate(async () => {
    const G = await import('/shared/glyphs.js');
    const svg = G.svgEl('svg', {}, document.body);
    try { G.gpu(svg, { x: 0, y: 0, litSms: [12] }); return null; } catch (e) { return e.message; } finally { svg.remove(); }
  });
  expect(bad).toBe('glyphs.gpu: litSms must be SM tile indices 0–11, got [12]');
});

test('S3 rack labels: one printed label per GPU tile, in place of its SM square (cluster-topology frame 8)', async ({ page }) => {
  const labels = ['1', '2', '3', '4', '5', '6', '7', '8'];
  const out = await probe(page, 'rack', { x: 0, y: 0, gpus: 8, labels }, (g) => ({
    texts: [...g.querySelectorAll('.g-rack-label')].map((t) => t.textContent),
    sms: g.querySelectorAll('.g-sm').length,
    inside: [...g.querySelectorAll('.g-rack-label')].every((t, i) => {
      const tile = g.querySelectorAll(':scope > rect.g-frame')[i + 1].getBBox();
      const c = t.getBBox();
      return c.x >= tile.x - 0.5 && c.x + c.width <= tile.x + tile.width + 0.5;
    }),
  }));
  expect(out).toEqual({ texts: labels, sms: 0, inside: true });
  const partial = await probe(page, 'rack', { x: 0, y: 0, gpus: 4, cols: 4, labels: ['', '', '3', ''] }, (g) => ({ texts: [...g.querySelectorAll('.g-rack-label')].map((t) => t.textContent), sms: g.querySelectorAll('.g-sm').length }));
  expect(partial).toEqual({ texts: ['3'], sms: 3 });
});

for (const theme of ['light', 'dark']) {
  test(`S3 vector and cell fill 'ok' | 'bad': semantic fills with readable ink, the value scale untouched (rlvr-grpo R column, ${theme})`, async ({ page }) => {
    const out = await probe(page, 'vector', { x: 0, y: 0, values: [1, 0, 0, 1], cell: 40, fill: ['ok', 'bad', 'bad', 'ok'], label: 'R' }, (g) => [...g.querySelectorAll('.g-cell')].map((c) => ({
      cls: c.getAttribute('class'), level: c.dataset.level, fill: getComputedStyle(c.querySelector('rect')).fill, ink: getComputedStyle(c.querySelector('.g-text')).fill, text: c.querySelector('.g-text').textContent,
    })), theme);
    expect(out.map((c) => c.cls)).toEqual(['glyph g-cell g-cell--ok', 'glyph g-cell g-cell--bad', 'glyph g-cell g-cell--bad', 'glyph g-cell g-cell--ok']);
    expect(out.map((c) => c.text)).toEqual(['1', '0', '0', '1']);
    expect(out[0].fill).not.toBe(out[1].fill);
    expect(out.every((c) => c.level === '0')).toBe(true);
    const contrast = await page.evaluate(([a, b]) => {
      // Computed color-mix() fills come back as oklab(…): let a canvas resolve any CSS color to sRGB bytes.
      const ctx = Object.assign(document.createElement('canvas'), { width: 1, height: 1 }).getContext('2d', { willReadFrequently: true });
      const rgb = (css) => { ctx.clearRect(0, 0, 1, 1); ctx.fillStyle = css; ctx.fillRect(0, 0, 1, 1); return [...ctx.getImageData(0, 0, 1, 1).data.slice(0, 3)]; };
      const lum = (c) => { const [r, g, b2] = rgb(c).map((v) => v / 255).map((v) => (v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4)); return 0.2126 * r + 0.7152 * g + 0.0722 * b2; };
      const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p);
      return (x + 0.05) / (y + 0.05);
    }, [out[0].fill, out[0].ink]);
    expect(contrast).toBeGreaterThanOrEqual(4.5);
    const one = await probe(page, 'vector', { x: 0, y: 0, values: [1, 1], cell: 40, fill: 'ok' }, (g) => [...g.querySelectorAll('.g-cell--ok')].length, theme);
    expect(one).toBe(2);
  });
}

test('S3 matrix shards: each GPU\'s columns (or rows) get their own tint and a GPU title; other cells keep the value scale (parallelism frames 3–4)', async ({ page }) => {
  const zeros = (r, c) => Array.from({ length: r }, () => Array(c).fill(0));
  const cols = await probe(page, 'matrix', { x: 0, y: 0, values: zeros(2, 4), cell: 20, label: 'W_in', shards: [{ cols: [1, 2], gpu: 1 }, { cols: [3, 4], gpu: 2 }] }, (g) => [...g.querySelectorAll('.g-cell')].map((c) => [c.dataset.gpu, getComputedStyle(c.querySelector('rect')).fill, c.querySelector('title').textContent]));
  expect(cols.map((c) => c[0])).toEqual(['1', '1', '2', '2', '1', '1', '2', '2']);
  expect(new Set(cols.map((c) => c[1])).size).toBe(2);
  expect(cols[0][2]).toBe('GPU 1');
  const rows = await probe(page, 'matrix', { x: 0, y: 0, values: zeros(4, 2), cell: 20, shards: [{ rows: [1, 2], gpu: 1 }] }, (g) => [...g.querySelectorAll('.g-cell')].map((c) => c.dataset.gpu ?? null));
  expect(rows).toEqual(['1', '1', '1', '1', null, null, null, null]);
  const bad = await page.evaluate(async () => {
    const G = await import('/shared/glyphs.js');
    const svg = G.svgEl('svg', {}, document.body);
    try { G.matrix(svg, { x: 0, y: 0, values: [[0, 0]], shards: [{ cols: [1, 3], gpu: 1 }] }); return null; } catch (e) { return e.message; } finally { svg.remove(); }
  });
  expect(bad).toBe('glyphs.matrix: a shard needs gpu 1–4 and cols or rows [from, to] inside the 1 × 2 matrix (1-based, inclusive)');
});

test('S3 shareBar: hatched parts draw the hatch, value-null parts print "not published", an all-unknown bar prints "no published shares"', async ({ page }) => {
  const run = await probe(page, 'shareBar', { x: 0, y: 0, w: 400, parts: [{ name: 'useful', value: 26.62, hue: 3 }, { name: 'below peak', value: 4.22, hue: 4 }, { name: 'lost to failures', value: 2.62, hue: 5, hatched: true }] }, (g) => g.querySelectorAll('.g-hatch').length);
  expect(run).toBe(2); // the segment and its legend swatch
  const glm = await probe(page, 'shareBar', { x: 0, y: 0, w: 300, parts: [{ name: 'pretrain', value: 27, hue: 1 }, { name: 'mid-train', value: 1.55, hue: 2 }, { name: 'post-training', value: null }] }, (g) => ({ label: g.querySelector('.g-unknown-label').textContent, aria: g.getAttribute('aria-label') }));
  expect(glm.label).toBe('post-training: not published');
  expect(glm.aria).toBe('shares: pretrain 94.6%, mid-train 5.4%, post-training not published');
  const kimi = await probe(page, 'shareBar', { x: 0, y: 0, w: 300, parts: [{ name: 'pretrain', value: null }, { name: 'post-training', value: null }] }, (g) => ({ label: g.querySelector('.g-unknown-label').textContent, pct: g.querySelectorAll('.g-pct').length, segs: g.querySelectorAll('.g-seg').length }));
  expect(kimi).toEqual({ label: 'no published shares', pct: 0, segs: 2 });
  const tail = await probe(page, 'shareBar', { x: 0, y: 0, w: 560, tailBasis: 'tail', tailLabel: 'last 5%', parts: [{ name: '4K', value: 27, hue: 1 }, { name: '32K', value: 1, hue: 2 }, { name: '128K', value: 0.5, hue: 3 }, { name: '200K', value: 0.05, hue: 4 }] }, (g) => ({
    tailLabel: g.querySelector('.g-tail-label').textContent,
    pcts: [...g.querySelectorAll('.g-pct')].map((t) => t.textContent),
    legend: [...g.querySelectorAll(':scope > .g-label')].map((t) => t.textContent).filter((t) => t.includes('of last')),
  }));
  expect(tail.tailLabel).toBe('last 5%');
  expect(tail.pcts).toEqual(['94.6%', '64.5%', '32.3%']);
  expect(tail.legend).toEqual(['200K · 3.2% of last 5%', 'others: zoomed below, shares of last 5%']);
});
