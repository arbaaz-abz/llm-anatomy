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
test('the only dashed strokes are the draft token, the free KV slot and the bars reference line', () => {
  const dashed = rules.filter((r) => /stroke-dasharray:\s*\d/.test(r.body)).map((r) => r.selector);
  assert.ok(dashed.length > 0);
  for (const selector of dashed) assert.match(selector, /draft|g-slot--free|g-bars-ref/, `${selector} draws a dashed stroke`);
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

test('every Architecture storyboard\'s \\htmlClass{hl-…} name outlines its linked glyph (theme data-hl rule)', async () => {
  const files = (await readdir(STORYBOARDS)).filter((f) => f.endsWith('.md'));
  const names = new Set();
  for (const file of files) {
    const md = await readFile(new URL(file, STORYBOARDS), 'utf8');
    if (!/^Track: architecture/m.test(md)) continue;
    for (const m of md.matchAll(/htmlClass\{hl-(\w+)\}/g)) names.add(m[1]);
  }
  assert.ok(names.size >= 30, `found ${names.size} names`);
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
