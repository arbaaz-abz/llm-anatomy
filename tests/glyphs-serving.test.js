// Shared prep S6 (Plan 4, Task 2): the Serving glyphs and glyph options (rulings P4-R8 … P4-R14).
// Pure layouts are tested directly; the DOM builders run against a minimal SVG document (no jsdom in this repo).
// Byte-for-byte defaults of the changed glyphs are pinned in e2e/glyph-options.spec.js (PRE_S6).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  barSegments, requestSlot, token, request, requestLayout, rack, rackLayout, blockPool, memBar, memBarLayout, stepBar, stepBarLayout,
  numberLine, numberLineLayout, prefixTree, prefixTreeLayout, shareBar, shareBarLayout, clipLine, formatShare,
} from '../shared/glyphs.js';
import { formatDuration } from '../math/core.js';

const near = (a, b, eps = 1e-6) => Math.abs(a - b) < eps;

// ---- a minimal SVG document: attributes, children, text, style, and enough query support for the hatch pattern ----
function withSvg(run) {
  const find = (node, sel) => {
    const [tag, cls] = sel.split('.');
    for (const c of node.children) {
      if (c.tag === tag && (!cls || String(c.attrs.class ?? '').split(' ').includes(cls))) return c;
      const hit = find(c, sel);
      if (hit) return hit;
    }
    return null;
  };
  const make = (tag, root) => ({
    tag, tagName: tag, attrs: {}, dataset: {}, children: [], textContent: '', style: {}, ownerSVGElement: root,
    setAttribute(k, v) { this.attrs[k] = String(v); }, append(...kids) { this.children.push(...kids); }, querySelector(sel) { return find(this, sel); },
  });
  const root = make('svg');
  root.ownerSVGElement = undefined;
  const saved = globalThis.document;
  globalThis.document = { createElementNS: (_ns, tag) => make(tag, root) };
  try { return run(root); } finally { globalThis.document = saved; }
}
const walk = (node) => [node, ...node.children.flatMap(walk)];
const withClass = (node, cls) => walk(node).filter((n) => String(n.attrs.class ?? '').split(' ').includes(cls));
const titleOf = (node) => node.children.find((c) => c.tag === 'title')?.textContent;
const textsIn = (node) => walk(node).filter((n) => n.tag === 'text').map((n) => n.textContent);

// ---------------- P4-R10: token owner ----------------
test('token owner: sets data-req to the request hue; fill and owner are exclusive; owner is A–D', () => {
  withSvg((svg) => {
    const g = token(svg, { x: 0, y: 0, text: 'A', owner: 'C' });
    assert.equal(g.attrs['data-req'], String(requestSlot('C')));
    assert.equal(g.attrs['data-level'], undefined, 'no value-scale level on an owner chip');
    assert.equal(token(svg, { x: 0, y: 0, text: 'x' }).attrs['data-req'], undefined, 'no owner, no attribute');
    assert.throws(() => token(svg, { x: 0, y: 0, text: 'A', owner: 'A', fill: 'var(--val-pos)' }), new RangeError('token: owner and fill are exclusive'));
    assert.throws(() => token(svg, { x: 0, y: 0, text: 'E', owner: 'E' }), /token: owner must be one of A, B, C, D/);
  });
});

// ---------------- P4-R9: request owner / idle / steps ----------------
test('requestLayout: the default marks are the old ones; idle units follow the last tick; steps are placed at their px', () => {
  const base = requestLayout({ prefill: 8, decode: 4, unit: 6 });
  assert.equal(base.prefillW, 48);
  assert.deepEqual(base.ticks, [54, 60, 66, 72]);
  assert.equal(base.total, 72);
  assert.equal(base.idle, null);
  const idle = requestLayout({ prefill: 2, decode: 3, unit: 6, idle: 3 });
  assert.deepEqual(idle.idle, { x: 30, width: 18 });
  assert.equal(idle.end, 48);
  const steps = requestLayout({ steps: [{ from: 0, to: 20, kind: 'queue' }, { from: 20, to: 70, kind: 'prefill' }, { from: 70, to: 80, kind: 'decode' }, { from: 80, to: 95, kind: 'idle' }] });
  assert.deepEqual(steps.steps.map((s) => [s.kind, s.x, s.width]), [['queue', 0, 20], ['prefill', 20, 50], ['decode', 70, 10], ['idle', 80, 15]]);
  assert.equal(steps.end, 95);
});

test('requestLayout: bad steps throw named RangeErrors', () => {
  assert.throws(() => requestLayout({ steps: [{ from: 5, to: 5, kind: 'decode' }] }), /request: a step needs 0 ≤ from < to/);
  assert.throws(() => requestLayout({ steps: [{ from: 0, to: 5, kind: 'nap' }] }), /request: step kind must be one of queue, prefill, decode, idle/);
  assert.throws(() => requestLayout({ steps: [{ from: 0, to: 9, kind: 'decode' }, { from: 4, to: 12, kind: 'decode' }] }), /request: steps must not overlap/);
  assert.throws(() => requestLayout({ steps: [] }), /request: steps must be a non-empty array/);
  assert.throws(() => requestLayout({ prefill: 2, decode: 1, idle: -1 }), /request: idle must be an integer ≥ 0/);
});

test('request: owner sets data-req; steps draw one titled mark per step (queue, prefill, decode, idle); idle draws hatched units', () => {
  withSvg((svg) => {
    const owned = request(svg, { x: 0, y: 0, prefill: 8, decode: 4, label: 'A', owner: 'C' });
    assert.equal(owned.attrs['data-req'], String(requestSlot('C')));
    const stepped = request(svg, { x: 0, y: 20, owner: 'B', label: 'B', steps: [
      { from: 0, to: 20, kind: 'queue' }, { from: 20, to: 70, kind: 'prefill' }, { from: 70, to: 80, kind: 'decode' }, { from: 80, to: 95, kind: 'idle' }] });
    const marks = withClass(stepped, 'g-step');
    assert.deepEqual(marks.map(titleOf), ['queue', 'prefill', 'decode', 'idle']);
    assert.equal(withClass(stepped, 'g-prefill')[0].attrs.width, '50');
    assert.equal(withClass(stepped, 'g-hatch').length, 1, 'only the idle step is hatched');
    assert.equal(withClass(stepped, 'g-decode').length, 1, 'a decode step is one tick at its end');
    const waiting = request(svg, { x: 0, y: 40, prefill: 2, decode: 3, idle: 3, owner: 'D' });
    const hatch = withClass(waiting, 'g-hatch');
    assert.equal(hatch.length, 1);
    assert.equal(hatch[0].attrs.x, '30');
    assert.equal(hatch[0].attrs.width, '18');
    assert.equal(withClass(waiting, 'g-decode').length, 3, 'idle adds no ticks');
  });
});

// ---------------- P4-R11: blockPool refs and cached ----------------
test('blockPool refs: a count above 1 prints small in its block, nothing at 1; the count must match the blocks', () => {
  withSvg((svg) => {
    const slots = Array.from({ length: 12 }, (_, i) => ({ owner: 'A', state: i < 8 ? 'filled' : 'free' }));
    const g = blockPool(svg, { x: 0, y: 0, blocks: 3, blockSize: 4, slots, refs: [2, 1, 3] });
    assert.deepEqual(withClass(g, 'g-block-refs').map((t) => t.textContent), ['2', '3']);
    assert.equal(titleOf(withClass(g, 'g-refs')[0]), 'block 0: 2 references');
    assert.equal(withClass(blockPool(svg, { x: 0, y: 0, blocks: 3, blockSize: 4, slots }), 'g-block-refs').length, 0, 'no refs, no marks');
    assert.throws(() => blockPool(svg, { x: 0, y: 0, blocks: 3, blockSize: 4, slots, refs: [2, 1] }), /glyphs.blockPool: refs must be 3 integers ≥ 0, one per block/);
    assert.throws(() => blockPool(svg, { x: 0, y: 0, blocks: 3, blockSize: 4, slots, refs: [2, 1.5, 3] }), /glyphs.blockPool: refs must be 3 integers ≥ 0/);
  });
});

test('blockPool cached: a filled slot no request holds keeps its letter and takes no request hue', () => {
  withSvg((svg) => {
    const slots = [{ owner: 'A', state: 'filled' }, { owner: 'A', state: 'cached' }, { state: 'free' }, { owner: 'B', state: 'reserved' }];
    const g = blockPool(svg, { x: 0, y: 0, blocks: 1, blockSize: 4, slots });
    const cells = withClass(g, 'g-slot');
    assert.deepEqual(cells.map((c) => c.attrs.class), ['g-slot g-slot--filled', 'g-slot g-slot--cached', 'g-slot g-slot--free', 'g-slot g-slot--reserved']);
    assert.equal(cells[1].attrs['data-req'], undefined);
    assert.equal(cells[0].attrs['data-req'], '1');
    assert.deepEqual(textsIn(cells[1]), ['A']);
    assert.equal(titleOf(cells[1]), 'block 0, slot 1: cached (A)');
    assert.throws(() => blockPool(svg, { x: 0, y: 0, blocks: 1, blockSize: 4, slots: [{ state: 'lost' }, {}, {}, {}] }), /use filled, reserved, cached or free/);
  });
});

// ---------------- P4-R12: memBar minSegment ----------------
test('memBarLayout: the default (minSegment 0) is the old geometry; a narrow nonzero segment widens and the others rescale to the same total', () => {
  const plain = memBarLayout({ useful: 23, reserved: 5, free: 20, w: 280 });
  assert.deepEqual(plain.segments.map((s) => s.name), ['useful', 'reserved', 'free']);
  assert.deepEqual(plain.segments.map((s) => s.x), barSegments([23, 5, 20], 280).map((s) => s.x), 'the same geometry as barSegments');
  assert.equal(plain.segments.some((s) => s.widened), false);
  const tiny = memBarLayout({ useful: 98, reserved: 2, free: 0, w: 240, minSegment: 18 });
  const [useful, reserved] = tiny.segments;
  assert.equal(reserved.width, 18);
  assert.equal(reserved.widened, true);
  assert.equal(tiny.segments.length, 2, 'a zero part is not drawn');
  assert.ok(near(useful.x + useful.width + 2 + reserved.width, 240), 'the bar keeps its total width');
  assert.deepEqual(tiny.brackets.map((b) => [b.x0, b.x1]), [[reserved.x, reserved.x + 18]]);
  assert.equal(memBarLayout({ useful: 98, reserved: 2, free: 0, w: 240, minSegment: 18 }).segments[1].value, 2, 'the printed value stays true');
  const roomy = memBarLayout({ useful: 23, reserved: 5, free: 20, w: 280, minSegment: 18 });
  assert.deepEqual(roomy.segments.map((s) => s.x), plain.segments.map((s) => s.x), 'nothing narrow, nothing changes');
  assert.deepEqual(roomy.brackets, []);
  assert.throws(() => memBarLayout({ useful: 1, reserved: 0, free: 0, w: 240, minSegment: -1 }), /glyphs.memBar: minSegment must be a number ≥ 0/);
});

test('memBar minSegment: the widened segment has a bracket mark and the summary line keeps the true numbers', () => {
  withSvg((svg) => {
    const g = memBar(svg, { x: 0, y: 0, w: 240, useful: 98, reserved: 2, free: 0, minSegment: 18 });
    assert.equal(withClass(g, 'g-bracket').length, 1);
    assert.ok(textsIn(g).includes('useful 98 · reserved 2 · free 0'));
    assert.equal(withClass(memBar(svg, { x: 0, y: 0, w: 240, useful: 98, reserved: 2, free: 0 }), 'g-bracket').length, 0);
  });
});

// ---------------- P4-R13: rack groups ----------------
test('rackLayout: the default is the old grid; groups add a bracket under each run of cells and room for its label', () => {
  const plain = rackLayout({ gpus: 8, cols: 4 });
  assert.deepEqual([plain.w, plain.h, plain.cells[5].x, plain.cells[5].y], [4 * 26 + 10, 2 * 26 + 10, 10 + 26, 10 + 26]);
  const grouped = rackLayout({ gpus: 16, cols: 8, groups: [{ from: 0, to: 1, label: 'P1' }, { from: 2, to: 3, label: 'P2' }, { from: 8, to: 15, label: 'decode' }] });
  assert.equal(grouped.brackets.length, 3, 'one bracket per group');
  assert.deepEqual(grouped.brackets.map((b) => b.label), ['P1', 'P2', 'decode']);
  const [p1, , decode] = grouped.brackets;
  assert.equal(p1.x0, grouped.cells[0].x);
  assert.equal(p1.x1, grouped.cells[1].x + grouped.tile);
  assert.equal(decode.x0, grouped.cells[8].x);
  assert.equal(decode.x1, grouped.cells[15].x + grouped.tile);
  assert.ok(decode.y > grouped.cells[8].y + grouped.tile, 'the bracket sits under its cells');
  assert.ok(grouped.h > rackLayout({ gpus: 16, cols: 8 }).h, 'labels get room');
  const wrap = rackLayout({ gpus: 8, cols: 4, groups: [{ from: 2, to: 5, label: 'EP' }] });
  assert.equal(wrap.brackets.length, 2, 'a run across two rows is one bracket per row');
  assert.deepEqual(wrap.brackets.map((b) => b.first), [true, false], 'only the first prints the label');
});

test('rackLayout: bad groups throw', () => {
  assert.throws(() => rackLayout({ gpus: 8, groups: [{ from: 0, to: 8, label: 'x' }] }), /glyphs.rack: a group needs 0 ≤ from ≤ to < 8/);
  assert.throws(() => rackLayout({ gpus: 8, groups: [{ from: 0, to: 3, label: 'a' }, { from: 3, to: 4, label: 'b' }] }), /glyphs.rack: groups must not overlap/);
  assert.throws(() => rackLayout({ gpus: 8, groups: [{ from: 0, to: 1 }] }), /glyphs.rack: a group needs a label/);
});

test('rack groups: draws the bracket paths and labels; the default draws none', () => {
  withSvg((svg) => {
    const g = rack(svg, { x: 0, y: 0, gpus: 16, cols: 8, groups: [{ from: 0, to: 7, label: 'prefill' }, { from: 8, to: 15, label: 'decode' }] });
    assert.equal(withClass(g, 'g-bracket').length, 2);
    assert.deepEqual(withClass(g, 'g-group-label').map((t) => t.textContent), ['prefill', 'decode']);
    assert.equal(withClass(rack(svg, { x: 0, y: 0, gpus: 8 }), 'g-bracket').length, 0);
  });
});

// ---------------- P4-R8: the step bar ----------------
const MS = (n) => n / 1000;
test('stepBarLayout: both rows sit on the one seconds scale (14.587 ms of 30 ms over 300 px is 145.87 px)', () => {
  const L = stepBarLayout({ w: 300, scaleS: MS(30), reading: [{ label: 'weights read', s: MS(14.587) }], mathS: MS(0.07) });
  const [reading] = L.rows;
  assert.ok(near(reading.segments[0].width, 145.87, 1e-9));
  assert.ok(near(reading.end, 145.87, 1e-9));
  assert.equal(L.longer, 'reading');
  assert.equal(L.rows[0].faint, false);
  assert.equal(L.rows[1].faint, true);
  assert.deepEqual(L.overlapped && L.overlapped.text, 'overlapped');
  assert.equal(L.total.text, '14.6 ms');
  assert.ok(near(L.total.x, reading.end + 6), 'the step total prints at the right end of the longer row');
});

test('stepBarLayout: the math row wins when it is longer; the reading row is faint; a tie fades neither', () => {
  const math = stepBarLayout({ w: 300, scaleS: MS(80), reading: [{ label: 'weights read', s: MS(14.58) }, { label: 'activations', s: MS(3.56) }], mathS: MS(70.74) });
  assert.equal(math.longer, 'arithmetic');
  assert.equal(math.rows[0].faint, true);
  assert.equal(math.rows[1].faint, false);
  assert.equal(math.total.text, '70.7 ms');
  const tie = stepBarLayout({ w: 300, scaleS: MS(30), reading: [{ label: 'weights read', s: MS(10) }], mathS: MS(10) });
  assert.equal(tie.longer, 'tie');
  assert.deepEqual(tie.rows.map((r) => r.faint), [false, false]);
  assert.equal(tie.overlapped, null);
});

test('stepBarLayout: parts keep a 3 px gap; a part under 18 px widens to 18 under a bracket and keeps its true value', () => {
  const L = stepBarLayout({ w: 300, scaleS: MS(30), reading: [{ label: 'weights read', s: MS(14.58) }, { label: 'KV read', s: MS(8.95) }, { label: 'activations', s: MS(0.2) }], mathS: MS(0.07) });
  const [weights, kv, acts] = L.rows[0].segments;
  assert.ok(near(weights.width, (14.58 / 30) * 300 - 3));
  assert.ok(near(kv.x, weights.x + weights.width + 3));
  assert.equal(acts.width, 18);
  assert.equal(acts.widened, true);
  assert.equal(acts.s, MS(0.2));
  assert.match(acts.title, /activations: 200 µs/);
  assert.ok(L.brackets.some((b) => b.x0 === acts.x && b.x1 === acts.x + 18), 'a bracket under the widened part');
  const math = L.rows[1].segments[0];
  assert.equal(math.widened, true);
  assert.equal(math.width, 18);
  assert.match(math.title, /arithmetic: 70 µs/);
  assert.equal(L.total.text, formatMs(MS(14.58 + 8.95 + 0.2)));
});

function formatMs(s) { return `${Number((s * 1000).toPrecision(3))} ms`; }

test('stepBarLayout: text goes inside a segment that fits it and below the bar otherwise; below-labels never overlap', () => {
  const L = stepBarLayout({ w: 300, scaleS: MS(30), reading: [{ label: 'weights read', s: MS(14.58) }, { label: 'KV read', s: MS(1) }, { label: 'acts', s: MS(0.5) }], mathS: MS(0.07) });
  const [weights, kv, acts] = L.rows[0].segments;
  assert.equal(weights.textInside, true);
  assert.equal(weights.text, 'weights read 14.6 ms');
  assert.equal(kv.textInside, false);
  const below = L.rows[0].segments.filter((s) => !s.textInside && s.text);
  assert.equal(below.length, 2);
  assert.notEqual(kv.line, acts.line, 'two neighbors that would overlap take different lines');
});

test('stepBarLayout: the default format is formatDuration and a custom one is used for every printed number', () => {
  const L = stepBarLayout({ w: 300, scaleS: 30, reading: [{ label: 'weights', s: 14 }], mathS: 1, format: (s) => `${s} s!` });
  assert.equal(L.total.text, '14 s!');
  assert.equal(L.rows[0].segments[0].text, 'weights 14 s!');
});

test('stepBarLayout: bad input throws named RangeErrors (a part past the scale, a bad label, no parts)', () => {
  const ok = { w: 300, scaleS: MS(30), reading: [{ label: 'a', s: MS(1) }], mathS: MS(1) };
  assert.throws(() => stepBarLayout({ ...ok, scaleS: 0 }), /glyphs.stepBar: scaleS must be a finite number > 0/);
  assert.throws(() => stepBarLayout({ ...ok, reading: [] }), /glyphs.stepBar: reading must be a non-empty array/);
  assert.throws(() => stepBarLayout({ ...ok, reading: [{ label: '', s: 1 }] }), /glyphs.stepBar: reading part 0 needs a label/);
  assert.throws(() => stepBarLayout({ ...ok, reading: [{ label: 'a', s: -1 }] }), /glyphs.stepBar: reading part "a" s must be a finite number ≥ 0/);
  assert.throws(() => stepBarLayout({ ...ok, mathS: MS(31) }), /glyphs.stepBar: arithmetic 0.031 s runs past scaleS 0.03 s; hold one scale across frames/);
  assert.throws(() => stepBarLayout({ ...ok, reading: [{ label: 'a', s: MS(20) }, { label: 'b', s: MS(11) }] }), /glyphs.stepBar: reading .* runs past scaleS/);
  assert.throws(() => stepBarLayout({ ...ok, mathS: NaN }), /glyphs.stepBar: mathS must be a finite number ≥ 0/);
});

test('stepBar: draws the two rows with the faint class on the shorter one, its label, and the total', () => {
  withSvg((svg) => {
    const g = stepBar(svg, { x: 0, y: 0, w: 300, scaleS: MS(30), reading: [{ label: 'weights read', s: MS(14.58) }], mathS: MS(0.07) });
    assert.deepEqual(withClass(g, 'g-step-row').map((r) => r.attrs.class), ['g-step-row g-step-row--reading', 'g-step-row g-step-row--arithmetic g-faint']);
    assert.equal(withClass(g, 'g-step--memory').length, 1);
    assert.equal(withClass(g, 'g-step--compute').length, 1);
    assert.ok(textsIn(g).includes('overlapped'));
    assert.equal(withClass(g, 'g-step-total')[0].textContent, '14.6 ms');
    assert.match(g.attrs['aria-label'], /^step time: reading weights read 14\.6 ms; arithmetic 70 µs; step 14\.6 ms$/);
  });
});

// ---------------- P4-R14: numberLine ----------------
const INT4 = Array.from({ length: 15 }, (_, i) => i - 7);
const E2M1 = [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6];
test('numberLineLayout: lo and hi map to the two ends, ticks sit at the grid values, a point finds its nearest grid value', () => {
  const L = numberLineLayout({ w: 280, lo: -7, hi: 7, grid: INT4, points: [{ value: 2.1 }, { value: -0.4, snapped: true }, { value: 6.5, snapped: true }] });
  assert.equal(L.toX(-7), 0);
  assert.equal(L.toX(7), 280);
  assert.deepEqual(L.ticks.map((t) => t.value), INT4);
  assert.ok(near(L.ticks[3].x, 60));
  assert.deepEqual(L.points.map((p) => p.snappedValue), [2, 0, 6]);
  assert.deepEqual(L.points.map((p) => p.snapped), [false, true, true]);
  assert.ok(near(L.points[0].x, L.toX(2.1)));
  assert.ok(near(L.points[1].snappedX, L.toX(0)));
});

test('numberLineLayout: uneven grids (E2M1) keep their true spacing; a point halfway between two ticks snaps to the one nearer zero', () => {
  const L = numberLineLayout({ w: 360, lo: -6, hi: 6, grid: E2M1, points: [{ value: 5 }, { value: 0.25 }, { value: -0.25 }] });
  assert.ok(near(L.ticks[8].x - L.ticks[7].x, 15), '0 to 0.5 is 15 px');
  assert.ok(near(L.ticks[14].x - L.ticks[13].x, 60), '4 to 6 is 60 px');
  assert.equal(L.points[0].snappedValue, 4, '5 is equally far from 4 and 6: the lower magnitude wins');
  assert.equal(L.points[1].snappedValue, 0);
  assert.equal(L.points[2].snappedValue, 0);
});

test('numberLineLayout: tick labels skip any that would collide, always keep the ends and zero, and every tick keeps its value in the title', () => {
  const L = numberLineLayout({ w: 360, lo: -6, hi: 6, grid: E2M1, points: [] });
  const printed = L.ticks.filter((t) => t.label !== null);
  assert.ok(printed.length < E2M1.length, 'some labels are skipped at this width');
  for (let i = 1; i < printed.length; i += 1) assert.ok(printed[i].x - printed[i - 1].x >= 20, 'printed labels clear each other');
  assert.deepEqual([L.ticks[0].label, L.ticks[14].label, L.ticks[7].label], ['−6', '6', '0']);
  assert.ok(L.ticks.every((t) => /^-?\d|^−/.test(t.title.replace('−', '-'))));
  const wide = numberLineLayout({ w: 700, lo: -7, hi: 7, grid: INT4, points: [] });
  assert.equal(wide.ticks.every((t) => t.label !== null), true, 'with room, every tick prints');
});

test('numberLineLayout: bad input throws', () => {
  const ok = { w: 280, lo: -7, hi: 7, grid: INT4, points: [] };
  assert.throws(() => numberLineLayout({ ...ok, hi: -7 }), /glyphs.numberLine: hi \(-7\) must exceed lo \(-7\)/);
  assert.throws(() => numberLineLayout({ ...ok, grid: [] }), /glyphs.numberLine: grid must be a non-empty array of finite numbers/);
  assert.throws(() => numberLineLayout({ ...ok, grid: [0, 9] }), /glyphs.numberLine: grid value 9 lies outside \[-7, 7\]/);
  assert.throws(() => numberLineLayout({ ...ok, points: [{ value: 8 }] }), /glyphs.numberLine: point value 8 lies outside \[-7, 7\]/);
  assert.throws(() => numberLineLayout({ ...ok, w: 0 }), /glyphs.numberLine: w must be a finite number > 0/);
});

test('numberLine: draws ticks, dots, a drop line only for a snapped point, and the selection mark for a followed one', () => {
  withSvg((svg) => {
    const g = numberLine(svg, { x: 0, y: 0, w: 280, lo: -7, hi: 7, grid: INT4, label: 'INT4 grid', points: [{ value: 2.1 }, { value: 0.4, snapped: true, followed: true }] });
    assert.equal(withClass(g, 'g-nl-tick').length, 15);
    assert.equal(withClass(g, 'g-nl-dot').length, 2);
    assert.equal(withClass(g, 'g-nl-drop').length, 1);
    assert.equal(withClass(g, 'g-select').length, 1);
    assert.match(g.attrs['aria-label'], /^INT4 grid: 15 grid values from −7 to 7; 2 points/);
  });
});

// ---------------- P4-R14: prefixTree ----------------
const CHAIN = (n, extra = {}) => Array.from({ length: n }, (_, i) => ({ id: `n${i}`, parent: i === 0 ? null : `n${i - 1}`, label: i === 0 ? 'start' : `b${i}`, state: i === 0 ? 'cached' : 'hit', ...extra }));
test('prefixTreeLayout: a left-to-right tree, one column per depth, edges from parent to child', () => {
  const nodes = [
    { id: 'root', parent: null, label: 'start', state: 'cached' },
    { id: 'a1', parent: 'root', label: 'You are', state: 'hit' },
    { id: 'a2', parent: 'a1', label: 'a cat', state: 'new', owner: 'A' },
    { id: 'c2', parent: 'a1', label: 'a dog', state: 'new', owner: 'C' },
  ];
  const L = prefixTreeLayout({ nodes });
  const at = Object.fromEntries(L.nodes.map((n) => [n.id, n]));
  assert.ok(at.a1.x > at.root.x && at.a2.x > at.a1.x);
  assert.equal(at.a2.x, at.c2.x, 'siblings share a column');
  assert.ok(at.c2.y > at.a2.y, 'siblings stack');
  assert.equal(at.a2.col, 2);
  assert.equal(L.edges.length, 3);
  assert.deepEqual(L.edges.map((e) => [e.from, e.to]), [['root', 'a1'], ['a1', 'a2'], ['a1', 'c2']]);
  assert.equal(at.root.w >= 64 && at.root.h === 26, true);
  assert.deepEqual(L.nodes.map((n) => n.state), ['cached', 'hit', 'new', 'new']);
});

test('prefixTreeLayout: the 7-node chain at 64 px plus 14 px edges takes 532 px (prefix-caching §4) when it fits; a path past the 4th block wraps to a second row', () => {
  const flat = prefixTreeLayout({ nodes: CHAIN(7), wrapAfter: 6 });
  assert.equal(flat.width, 532);
  assert.equal(new Set(flat.nodes.map((n) => n.y)).size, 1);
  const wrapped = prefixTreeLayout({ nodes: CHAIN(7) });
  const rows = new Set(wrapped.nodes.map((n) => n.y));
  assert.equal(rows.size, 2, 'two rows');
  const byId = Object.fromEntries(wrapped.nodes.map((n) => [n.id, n]));
  assert.equal(byId.n4.y, byId.n0.y, 'the 4th block (depth 4) is still on the first row');
  assert.ok(byId.n5.y > byId.n4.y, 'the 5th block wraps');
  assert.equal(byId.n5.x, byId.n1.x, 'a wrapped node starts under the first block column');
  assert.ok(byId.n6.x > byId.n5.x);
  assert.ok(wrapped.width < flat.width);
});

test('prefixTreeLayout: a long label widens its node and its whole column; every node is wide enough for its text', () => {
  const L = prefixTreeLayout({ nodes: [{ id: 'r', parent: null, label: 'start', state: 'cached' }, { id: 'a', parent: 'r', label: 'The cat sat down', state: 'hit', followed: true }, { id: 'b', parent: 'r', label: 'Hi…', state: 'new', owner: 'B' }] });
  const [, a, b] = L.nodes;
  assert.ok(a.w >= 'The cat sat down'.length * 6.6);
  assert.equal(b.w, a.w, 'a column is as wide as its widest node');
  assert.equal(a.followed, true);
});

test('prefixTreeLayout: bad trees throw (duplicate id, unknown parent, cycle, bad state, a new node without an owner)', () => {
  assert.throws(() => prefixTreeLayout({ nodes: [] }), /glyphs.prefixTree: nodes must be a non-empty array/);
  assert.throws(() => prefixTreeLayout({ nodes: [...CHAIN(2), { id: 'n1', parent: null, label: 'x', state: 'hit' }] }), /glyphs.prefixTree: duplicate node id "n1"/);
  assert.throws(() => prefixTreeLayout({ nodes: [{ id: 'a', parent: 'zzz', label: 'x', state: 'hit' }] }), /glyphs.prefixTree: node "a" has unknown parent "zzz"/);
  assert.throws(() => prefixTreeLayout({ nodes: [{ id: 'a', parent: 'b', label: 'x', state: 'hit' }, { id: 'b', parent: 'a', label: 'y', state: 'hit' }] }), /glyphs.prefixTree: the nodes form a cycle/);
  assert.throws(() => prefixTreeLayout({ nodes: [{ id: 'a', parent: null, label: 'x', state: 'gone' }] }), /glyphs.prefixTree: node "a" state must be one of hit, new, cached, evicted/);
  assert.throws(() => prefixTreeLayout({ nodes: [{ id: 'a', parent: null, label: 'x', state: 'new' }] }), /glyphs.prefixTree: node "a" is new and needs an owner A–D/);
  assert.throws(() => prefixTreeLayout({ nodes: CHAIN(2), wrapAfter: 0 }), /glyphs.prefixTree: wrapAfter must be an integer ≥ 1/);
});

test('prefixTree: draws edges first, one titled node per state, the hatch only on evicted nodes, the request hue on new nodes, the selection mark on followed ones', () => {
  withSvg((svg) => {
    const nodes = [
      { id: 'r', parent: null, label: 'start', state: 'cached' },
      { id: 'h', parent: 'r', label: 'You are', state: 'hit' },
      { id: 'n', parent: 'h', label: 'a cat', state: 'new', owner: 'C', followed: true },
      { id: 'e', parent: 'h', label: 'a dog', state: 'evicted' },
    ];
    const g = prefixTree(svg, { x: 0, y: 0, nodes, label: 'prefix tree' });
    assert.equal(withClass(g, 'g-ptree-edge').length, 3);
    assert.deepEqual(withClass(g, 'g-ptree-node').map((n) => n.attrs.class), ['glyph g-ptree-node g-ptree--cached', 'glyph g-ptree-node g-ptree--hit', 'glyph g-ptree-node g-ptree--new', 'glyph g-ptree-node g-ptree--evicted'].map((c) => c.replace('glyph ', '')));
    assert.equal(withClass(g, 'g-ptree--new')[0].attrs['data-req'], String(requestSlot('C')));
    assert.equal(withClass(g, 'g-hatch').length, 1);
    assert.equal(withClass(g, 'g-select').length, 1);
    assert.equal(titleOf(withClass(g, 'g-ptree--evicted')[0]), 'a dog: evicted');
    assert.match(g.attrs['aria-label'], /^prefix tree: 4 blocks/);
  });
});

// ---------------- S7 shared-1: shareBar tail 'none' folds nothing ----------------
test("shareBarLayout: minSegment 0 never folds a part into 'others', however thin", () => {
  const parts = [{ name: 'w', value: 140, hue: 1 }, { name: 'kv', value: 0.6, hue: 2 }, { name: 'free', value: 0.4, hue: 3 }];
  assert.equal(shareBarLayout(parts, { w: 300, minSegment: 0 }).tail.length, 0);
  assert.ok(shareBarLayout(parts, { w: 300 }).tail.length > 0, 'the default still zooms');
});

test("shareBar tail 'none': no bracket and no zoomed bar; a thin nonzero part is drawn at least 1 px wide, its value in the title", () => {
  withSvg((svg) => {
    const parts = [{ name: 'w', value: 140, hue: 1 }, { name: 'kv', value: 0.6, hue: 2 }, { name: 'free', value: 0.4, hue: 3 }];
    const g = shareBar(svg, { x: 0, y: 0, w: 300, parts, tail: 'none' });
    assert.equal(withClass(g, 'g-bracket').length, 0);
    const segs = withClass(g, 'g-seg');
    assert.equal(segs.length, 3);
    assert.ok(segs.every((r) => Number(r.attrs.width) >= 1));
    assert.match(titleOf(segs[2]), /^free: /);
  });
});

// ---------------- S7 shared-2: memBar format ----------------
test('memBar format: the printed shares and the aria-label use it; the default stays whole percentages', () => {
  withSvg((svg) => {
    const g = memBar(svg, { x: 0, y: 0, w: 240, useful: 537, reserved: 463, free: 0, format: formatShare });
    assert.ok(textsIn(g).includes('53.7%'));
    assert.match(g.attrs['aria-label'], /useful 53\.7%/);
    const plain = memBar(svg, { x: 0, y: 0, w: 240, useful: 537, reserved: 463, free: 0 });
    assert.ok(textsIn(plain).includes('54%'));
    assert.match(plain.attrs['aria-label'], /useful 54%/);
  });
});

// ---------------- S7 shared-4: clipLine format ----------------
test('clipLine format: tick labels, the marker label and the aria-label use it; the default stays two decimals', () => {
  withSvg((svg) => {
    const g = clipLine(svg, { x: 0, y: 0, lo: 0, hi: 600, band: [0, 400], marker: 525.06, format: (ms) => formatDuration(ms / 1000) });
    const texts = textsIn(g);
    assert.ok(texts.includes('525 ms') && texts.includes('400 ms'), texts.join(' | '));
    assert.match(g.attrs['aria-label'], /525 ms/);
    const plain = textsIn(clipLine(svg, { x: 0, y: 0, lo: 0.8, hi: 1.2, band: [0.8, 1.2], marker: 1.0 }));
    assert.ok(plain.includes('0.80') && plain.includes('1.00'));
  });
});
