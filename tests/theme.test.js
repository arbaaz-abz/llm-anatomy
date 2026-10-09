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
// not a state of an item, so it is an allowed dash besides the draft token.
// S3 (P3-R10, P3-R11): curvePlot's refY is drawn exactly like the bars reference line, the same allowed dash.
// P4-R11 / P4-R15 (S6): the free KV slot is a faint fill, never dashed; no new dashed mark.
test('the only dashed strokes are the draft token and the bars and curvePlot reference lines', () => {
  const dashed = rules.filter((r) => /stroke-dasharray:\s*\d/.test(r.body)).map((r) => r.selector);
  assert.ok(dashed.length > 0);
  for (const selector of dashed) assert.match(selector, /draft|g-bars-ref|g-plot-ref/, `${selector} draws a dashed stroke`);
  assert.equal(merged('.g-plot .g-plot-ref')['stroke-dasharray'], merged('.g-bars .g-bars-ref')['stroke-dasharray']);
});

test('a free KV slot is a faint fill with no dash (P4-R11); a cached slot is its own muted fill', () => {
  for (const part of ['k', 'v']) {
    const free = merged(`.g-pool .g-slot--free .g-slot-${part}`);
    assert.equal(free['stroke-dasharray'], undefined);
    assert.equal(free.fill, 'var(--surface-2)');
    const cached = merged(`.g-pool .g-slot--cached .g-slot-${part}`);
    assert.equal(cached['stroke-dasharray'], undefined);
    assert.match(cached.fill, /var\(--ink-muted\)/);
  }
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
// The tracks whose storyboards' math-panel names must all have an outline rule (Serving joined in Plan 4, P4-R16).
const HL_TRACKS = ['architecture', 'training', 'serving'];
const SERVING_HL = ['ttft', 'tpot', 'compute', 'memory', 'table', 'waste', 'hit', 'scale', 'code'];

test('every Architecture, Training and Serving storyboard\'s \\htmlClass{hl-…} name outlines its linked glyph (theme data-hl rule)', async () => {
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
  for (const name of SERVING_HL) assert.ok(names.has(name), `the Serving storyboards name hl-${name}`);
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

// ---- Shared prep S6: the Serving glyph styles ----
test('P4-R16: the nine Serving math-panel names outline their linked glyph; the step bar and waste segment can be linked too', () => {
  const linked = rules.filter((r) => /stroke:\s*var\(--accent\)/.test(r.body)).flatMap((r) => r.selector.split(',').map((x) => x.trim()));
  for (const name of SERVING_HL) assert.ok(linked.includes(`[data-hl="${name}"] [data-link="${name}"] .g-frame`), `hl-${name} has no outline rule`);
  assert.ok(linked.includes('[data-hl="compute"] [data-link="compute"] .g-step--compute'));
  assert.ok(linked.includes('[data-hl="memory"] [data-link="memory"] .g-step--memory'));
  for (const sel of linked.filter((x) => /g-step--|g-reserved-bg/.test(x))) assert.match(merged(sel)['stroke-width'], /^2$/, `${sel} stays at the 2px ceiling`);
});

test('.g-faint steps a mark back by opacity alone (never a dash)', () => {
  const faint = merged('.g-faint');
  assert.ok(Number(faint.opacity) > 0 && Number(faint.opacity) < 1);
  assert.equal(faint['stroke-dasharray'], undefined);
});

test('step bar fills: the reading row is --sem-memory, the arithmetic row --sem-compute; the ink on them is --ink', () => {
  assert.match(merged('.g-stepbar .g-step--memory').fill, /var\(--sem-memory\)/);
  assert.match(merged('.g-stepbar .g-step--compute').fill, /var\(--sem-compute\)/);
  assert.equal(merged('.g-stepbar .g-step-text').fill, 'var(--ink)');
});

test('prefix tree: hit is --sem-ok, new takes the request hue, cached is muted, evicted is plain (the hatch is drawn over it)', () => {
  assert.match(merged('.g-ptree--hit .g-frame').fill, /var\(--sem-ok\)/);
  for (let n = 1; n <= 4; n += 1) assert.match(merged(`.g-ptree--new[data-req="${n}"] .g-frame`).fill, new RegExp(`var\\(--req-${n}\\)`));
  assert.match(merged('.g-ptree--cached .g-frame').fill, /var\(--ink-muted\)/);
  assert.equal(merged('.g-ptree--evicted .g-frame').fill, 'var(--surface)');
  assert.equal(merged('.g-ptree .g-ptree-edge')['stroke-dasharray'], undefined);
});

test('token owner and request owner take the --req hue', () => {
  for (let n = 1; n <= 4; n += 1) {
    assert.match(merged(`.g-token[data-req="${n}"] .g-frame`).fill, new RegExp(`var\\(--req-${n}\\)`));
    assert.equal(merged(`.g-request[data-req="${n}"] .g-prefill`).fill, `var(--req-${n})`);
    assert.equal(merged(`.g-request[data-req="${n}"] .g-decode`).stroke, `var(--req-${n})`);
  }
});

// Ink on the tinted fills Serving glyphs print text over: >= 4.5:1 in all three theme blocks. color-mix(in oklab) is
// reproduced here (OKLab, Björn Ottosson), so the check reads the same fill the browser paints.
const SURFACE_OF = { light: '#ffffff', osDark: '#1a1f28', dark: '#1a1f28' };
const INK_OF = { light: null, osDark: null, dark: null };
const srgbToLinear = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const linearToSrgb = (c) => (c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055);
const toOklab = (hex) => {
  const [r, g, b] = [1, 3, 5].map((i) => srgbToLinear(parseInt(hex.slice(i, i + 2), 16) / 255));
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const sCone = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * sCone, 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * sCone, 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * sCone];
};
const fromOklab = ([L, a, b]) => {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const sCone = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
  const lin = [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * sCone, -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * sCone, -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * sCone];
  return `#${lin.map((c) => Math.round(Math.min(Math.max(linearToSrgb(c), 0), 1) * 255).toString(16).padStart(2, '0')).join('')}`;
};
const mix = (top, pct, bottom) => {
  const [a, b] = [toOklab(top), toOklab(bottom)];
  return fromOklab(a.map((v, i) => v * (pct / 100) + b[i] * (1 - pct / 100)));
};
const tokenHex = (text, name) => text.match(new RegExp(`--${name}: (#[0-9a-f]{6})`))?.[1];

test('ink stays readable (4.5:1) on the Serving tints: step bar rows, hit and cached nodes, request-hued chips and nodes, cached slots', () => {
  const tints = [['sem-memory', 55], ['sem-compute', 55], ['sem-ok', 40], ['req-1', 40], ['req-2', 40], ['req-3', 40], ['req-4', 40], ['ink-muted', 35]];
  for (const [name, marker] of Object.entries(THEME_BLOCKS)) {
    const text = blockOf(marker);
    const ink = tokenHex(text, 'ink') ?? INK_OF[name] ?? tokenHex(blockOf(THEME_BLOCKS.light), 'ink');
    const surface = SURFACE_OF[name];
    for (const [token, pct] of tints) {
      const hex = tokenHex(text, token) ?? tokenHex(blockOf(THEME_BLOCKS.osDark), token);
      assert.ok(hex, `${name} defines --${token}`);
      const fill = mix(hex, pct, surface);
      assert.ok(ratio(ink, fill) >= 4.5, `${name}: --ink on ${pct}% --${token} (${fill}): ${ratio(ink, fill).toFixed(2)}:1`);
    }
  }
});
