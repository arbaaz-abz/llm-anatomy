// Theme rules shared by every page (shared patch S0, shared-8): no dashed frame except a draft,
// and the toy readout classes pages use instead of inline styles.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const css = await readFile(new URL('../shared/theme.css', import.meta.url), 'utf8');
const stripped = css.replace(/\/\*[\s\S]*?\*\//g, '');
// Every top-level or nested rule as { selector, body } (media blocks are opened, not matched as rules).
const rules = [...stripped.matchAll(/([^{}]+)\{([^{}]*)\}/g)].map((m) => ({ selector: m[1].trim().replace(/\s+/g, ' '), body: m[2].trim() }));
const declarations = (body) => Object.fromEntries(body.split(';').map((d) => d.trim()).filter(Boolean).map((d) => {
  const i = d.indexOf(':');
  return [d.slice(0, i).trim(), d.slice(i + 1).trim()];
}));
const ruleFor = (selector) => rules.filter((r) => r.selector.split(',').map((s) => s.trim()).includes(selector));
const merged = (selector) => Object.assign({}, ...ruleFor(selector).map((r) => declarations(r.body)));

test('dim blocks and block-stack frames are never dashed (dashed means draft, README lesson 24)', () => {
  assert.equal(merged('.g-block--dim .g-frame')['stroke-dasharray'], 'none');
  assert.equal(merged('.g-stack .g-stack-frame')['stroke-dasharray'], 'none');
});

// S1 ruling S1-R8: the `bars` reference line is a chart guide with its own printed label (moe §4 condition c),
// not a state of an item, so it is the one dashed line besides the draft token and the free KV slot.
// S3 (P3-R10, P3-R11): curvePlot's refY is drawn exactly like the bars reference line, the same allowed dash.
test('the only dashed strokes are the draft token, the free KV slot and the bars and curvePlot reference lines', () => {
  const dashed = rules.filter((r) => /stroke-dasharray:\s*\d/.test(r.body)).map((r) => r.selector);
  assert.ok(dashed.length > 0);
  for (const selector of dashed) assert.match(selector, /draft|g-slot--free|g-bars-ref|g-plot-ref/, `${selector} draws a dashed stroke`);
  assert.equal(merged('.g-plot .g-plot-ref')['stroke-dasharray'], merged('.g-bars .g-bars-ref')['stroke-dasharray']);
});

test('toy readout rows: theme classes with the reviewed declarations', () => {
  assert.deepEqual(merged('.toy-rows'), { display: 'grid', gap: '3px', 'overflow-x': 'auto', 'max-width': '100%' });
  assert.deepEqual(merged('.toy-row'), { display: 'flex', 'align-items': 'center', gap: '3px', 'font-family': 'var(--font-mono)', 'font-variant-numeric': 'tabular-nums', 'font-size': 'var(--fs-s)' });
  assert.deepEqual(merged('.toy-row-label'), { flex: '0 0 7.5em', color: 'var(--ink-muted)', 'font-family': 'var(--font-body)' });
  assert.deepEqual(merged('.toy-row .cell'), { flex: '0 0 4.6em', 'text-align': 'center', padding: '6px 0', 'border-radius': 'var(--radius-s)' });
  assert.deepEqual(merged('.toy-maps'), { display: 'flex', 'flex-wrap': 'wrap', gap: 'var(--space-4)' });
});

test('a masked HTML cell is hatched, and its text sits on a solid surface chip so no stripe crosses it', () => {
  assert.equal(merged('.cell--masked').background, 'repeating-linear-gradient(45deg, color-mix(in oklab, var(--ink-muted) 55%, transparent) 0 1px, var(--surface) 1px 8px)');
  assert.deepEqual(merged('.cell--masked .cell-text'), { background: 'var(--surface)', color: 'var(--ink)', padding: '0 3px', 'border-radius': '2px' });
});

// ---- Shared patch S1: math-panel links for the wave-1b storyboards, dial and bars styles ----
import { readdir } from 'node:fs/promises';

const STORYBOARDS = new URL('../docs/storyboards/', import.meta.url);
// The tracks whose storyboards' math-panel names must all have an outline rule. Plan 4 adds 'serving' here
// (its nine names, ttft … code, are not in theme.css yet).
const HL_TRACKS = ['architecture', 'training'];

test('every Architecture and Training storyboard\'s \\htmlClass{hl-…} name outlines its linked glyph (theme data-hl rule)', async () => {
  const files = (await readdir(STORYBOARDS)).filter((f) => f.endsWith('.md'));
  const names = new Set();
  const trackLine = new RegExp(`^Track: (${HL_TRACKS.join('|')}) `, 'm');
  for (const file of files) {
    const md = await readFile(new URL(file, STORYBOARDS), 'utf8');
    if (!trackLine.test(md)) continue;
    for (const m of md.matchAll(/htmlClass\{hl-(\w+)\}/g)) names.add(m[1]);
  }
  assert.ok(names.size >= 50, `found ${names.size} names`);
  for (const name of ['n', 'd', 'pol', 'beta', 'ref', 'r', 'ratio', 'rho', 'int', 'ridge', 'peak', 'bw', 'w', 'g', 'comm', 'bubble', 'link', 'flops', 'mfu', 'loss']) assert.ok(names.has(name), `the Training storyboards name hl-${name}`);
  const linked = rules.filter((r) => /stroke:\s*var\(--accent\)/.test(r.body) && /stroke-width:\s*2\b/.test(r.body)).flatMap((r) => r.selector.split(',').map((s) => s.trim()));
  for (const name of names) assert.ok(linked.includes(`[data-hl="${name}"] [data-link="${name}"] .g-frame`), `hl-${name} has no outline rule`);
});

test('dial: the hand is ink (never accent or value scale), the seen wedge a pale line fill with no outline (rope §4 b, c)', () => {
  assert.equal(merged('.g-dial .g-dial-hand').stroke, 'var(--ink)');
  assert.equal(merged('.g-dial .g-dial-seen').fill, 'var(--line)');
  assert.equal(merged('.g-dial .g-dial-seen').stroke, 'none');
  assert.equal(merged('.g-dial .g-dial-ghost')['stroke-dasharray'], undefined);
});

test('bars: neutral ink-tint fill with no outline; the reference line is dashed (moe §4 b, c)', () => {
  assert.equal(merged('.g-bars .g-bar').fill, 'color-mix(in oklab, var(--ink) 30%, var(--surface))');
  assert.equal(merged('.g-bars .g-bar').stroke, 'none');
  assert.match(merged('.g-bars .g-bars-ref')['stroke-dasharray'], /^\d/);
});

// Shared patch S2 (shared-2): fact-row source links stay on one line on desktop and wrap on a phone.
test('.fact-meta stays nowrap above 640 px and wraps at 640 px or less', () => {
  assert.equal(declarations(ruleFor('.fact-meta')[0].body)['white-space'], 'nowrap'); // the base rule, before the media query
  const phone = css.match(/@media \(max-width: 640px\) \{[^@]*?\.fact-meta \{([^}]*)\}/);
  assert.ok(phone, 'a .fact-meta rule inside the 640 px media query');
  const d = declarations(phone[1]);
  assert.equal(d['flex-wrap'], 'wrap');
  assert.equal(d['white-space'], 'normal');
  assert.equal(d['row-gap'], 'var(--space-1)');
});

// ---- Shared prep S3: the Training glyph styles ----
test('laneTimeline kinds: compute, memory, comm on --sem-*, forward and backward on the carries; idle and lost hatched (P3-R9)', () => {
  const fills = Object.fromEntries(['compute', 'memory', 'comm', 'forward', 'backward', 'lost', 'idle'].map((k) => [k, merged(`.g-lanes .g-lane--${k}`).fill]));
  assert.match(fills.compute, /var\(--sem-compute\)/);
  assert.match(fills.memory, /var\(--sem-memory\)/);
  assert.match(fills.comm, /var\(--sem-comm\)/);
  assert.match(fills.forward, /var\(--carry-activation\)/);
  assert.match(fills.backward, /var\(--carry-gradient\)/);
  assert.equal(fills.lost, fills.compute, 'lost work keeps the compute fill under its hatch');
  assert.equal(fills.idle, 'var(--surface)');
});

test('curvePlot: series never use the track accent; muted is lighter, never dashed; markers are ink', () => {
  const series = rules.filter((r) => /g-series|g-tone--/.test(r.selector));
  assert.ok(series.length > 0);
  for (const r of series) assert.doesNotMatch(r.body, /--accent/, `${r.selector} uses the accent`);
  assert.equal(merged('.g-plot .g-series--muted')['stroke-dasharray'], undefined);
  assert.equal(merged('.g-plot .g-marker').fill, 'var(--ink)');
});

test('the new tokens --carry-weight and --sem-comm drive the weight dot and the comm lanes', () => {
  assert.equal(merged('.g-flow--weight .g-dot').fill, 'var(--carry-weight)');
  assert.match(merged('.g-lanes .g-lane--comm').fill, /--sem-comm/);
});

// ---- Training review shared-3 and shared-5 ----
const blockOf = (marker) => { const start = css.indexOf(marker); return start < 0 ? '' : css.slice(start, css.indexOf('}', start)); };
const THEME_BLOCKS = { light: ':root {', osDark: ':root:not([data-theme="light"]) {', dark: ':root[data-theme="dark"] {' };
const BG = { light: '#f4f5f8', osDark: '#12161d', dark: '#12161d' };
const lum = (hex) => {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255).map((c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};
const ratio = (a, b) => { const [hi, lo] = [lum(a), lum(b)].sort((x, y) => y - x); return (hi + 0.05) / (lo + 0.05); };

test('verdict text classes use their own ink tokens', () => {
  assert.deepEqual(merged('.sem-text--memory'), { color: 'var(--sem-memory-ink)' });
  assert.deepEqual(merged('.sem-text--compute'), { color: 'var(--sem-compute-ink)' });
});

test('--sem-memory-ink and --sem-compute-ink pass 4.5:1 on --bg in all three theme blocks', () => {
  for (const [name, marker] of Object.entries(THEME_BLOCKS)) {
    const text = blockOf(marker);
    assert.ok(text, `missing ${name} block`);
    for (const token of ['sem-memory-ink', 'sem-compute-ink']) {
      const hex = text.match(new RegExp(`--${token}: (#[0-9a-f]{6})`))?.[1];
      assert.ok(hex, `${name} defines --${token}`);
      assert.ok(ratio(hex, BG[name]) >= 4.5, `${name} --${token} ${hex}: ${ratio(hex, BG[name]).toFixed(2)}:1`);
    }
  }
});

test('selectable stage items show a pointer and the course focus ring', () => {
  assert.equal(merged('.stage-item').cursor, 'pointer');
  assert.equal(merged('.stage-item:focus').outline, 'none');
  assert.deepEqual(merged('.stage-item:focus-visible'), { outline: '2px solid var(--focus)', 'outline-offset': '2px' });
});
