import { test } from 'node:test';
import assert from 'node:assert/strict';
import { valueColor, valueLevel, levelFromFill, tokenWidth, requestSlot, formatCell, pixelFill, blockStackLayout, barSegments, formatShare, shareBarLayout } from '../shared/glyphs.js';

test('zero, NaN and non-numeric inputs map to the zero token, never a "NaN%" string', () => {
  assert.equal(valueColor(0, 3), 'var(--val-zero)');
  assert.equal(valueColor(Number.NaN, 3), 'var(--val-zero)');
  assert.equal(valueColor(undefined, 3), 'var(--val-zero)');
  assert.equal(valueColor('x', 3), 'var(--val-zero)');
  assert.equal(valueColor(null, 3), 'var(--val-zero)'); // Number(null) is 0
  assert.equal(valueColor(0.004, 3), 'var(--val-zero)'); // rounds to 0 %
});

test('sign picks the pole and magnitude the mix', () => {
  assert.equal(valueColor(1.5, 3), 'color-mix(in oklab, var(--val-zero), var(--val-pos) 50%)');
  assert.equal(valueColor(-1.5, 3), 'color-mix(in oklab, var(--val-zero), var(--val-neg) 50%)');
  assert.equal(valueColor(3, 3), 'color-mix(in oklab, var(--val-zero), var(--val-pos) 100%)');
});

test('clamps beyond maxAbs, including infinities', () => {
  assert.equal(valueColor(99, 3), valueColor(3, 3));
  assert.equal(valueColor(-99, 3), valueColor(-3, 3));
  assert.equal(valueColor(-Infinity, 3), valueColor(-3, 3));
  assert.equal(valueColor(Infinity, 3), valueColor(3, 3));
});

test('rejects a useless maxAbs with a clear message', () => {
  assert.throws(() => valueColor(1, 0), /maxAbs/);
  assert.throws(() => valueColor(1, -2), /maxAbs/);
  assert.throws(() => valueColor(1, Number.NaN), /maxAbs/);
});

test('valueLevel buckets |v|/maxAbs into 0..10', () => {
  assert.equal(valueLevel(0, 4), 0);
  assert.equal(valueLevel(-2, 4), 5);
  assert.equal(valueLevel(4, 4), 10);
  assert.equal(valueLevel(40, 4), 10);
  assert.equal(valueLevel(-Infinity, 4), 10);
  assert.equal(valueLevel(Number.NaN, 4), 0);
  assert.equal(valueLevel(undefined, 4), 0);
  assert.equal(valueLevel('x', 4), 0);
});

test('levelFromFill recovers the cell ink level from a valueColor() string', () => {
  assert.equal(levelFromFill(valueColor(1.5, 3)), 5);
  assert.equal(levelFromFill(valueColor(-3, 3)), 10);
  assert.equal(levelFromFill(valueColor(0, 3)), 0);
  assert.equal(levelFromFill('var(--accent)'), 0);
  assert.equal(levelFromFill(undefined), 0);
});

test('tokenWidth grows with text but never below the minimum', () => {
  assert.equal(tokenWidth(''), 28);
  assert.ok(tokenWidth('The') < tokenWidth('down'));
  assert.equal(tokenWidth('ab'), tokenWidth('ab', 11)); // the one diagram label size
  assert.ok(tokenWidth('attention', 13) > tokenWidth('attention'));
});

test('requestSlot maps owner letters onto the four request hues, cycling', () => {
  assert.equal(requestSlot('A'), 1);
  assert.equal(requestSlot('D'), 4);
  assert.equal(requestSlot('E'), 1);
  assert.equal(requestSlot('b'), 2);
  assert.equal(requestSlot(''), 1);
});

test('formatCell prints compact numbers and the masked symbol', () => {
  assert.equal(formatCell(-Infinity), '−∞');
  assert.equal(formatCell(0.7031), '0.70');
  assert.equal(formatCell(-0.5), '−0.50');
  assert.equal(formatCell(-0.87), '−0.87');
  assert.equal(formatCell(3), '3');
  assert.equal(formatCell(1.407), '1.41');
  assert.equal(formatCell(-12.3), '−12');
  assert.ok([0.7031, -0.5, -0.87, 1.407, -12.3, -Infinity].every((v) => formatCell(v).length <= 5));
});

test('pixelFill: grey levels and rgb triples, never the value scale', () => {
  assert.equal(pixelFill(0), 'rgb(0 0 0)');
  assert.equal(pixelFill(1), 'rgb(255 255 255)');
  assert.equal(pixelFill(0.5), 'rgb(128 128 128)');
  assert.equal(pixelFill([200, 120, 40]), 'rgb(200 120 40)');
  assert.throws(() => pixelFill(1.2), /grey level 0–1/);
  assert.throws(() => pixelFill([300, 0, 0]), /grey level 0–1/);
  assert.throws(() => pixelFill('red'), /grey level 0–1/);
});

test('blockStackLayout: shows every block when nothing would be hidden', () => {
  assert.deepEqual(blockStackLayout({ count: 2 }), [{ kind: 'block', index: 1 }, { kind: 'block', index: 2 }]);
  assert.deepEqual(blockStackLayout({ count: 3 }).map((r) => r.kind), ['block', 'block', 'block']);
});

test('blockStackLayout: collapses the middle and keeps the last block', () => {
  assert.deepEqual(blockStackLayout({ count: 61 }), [
    { kind: 'block', index: 1 }, { kind: 'block', index: 2 }, { kind: 'collapse', hidden: 58 }, { kind: 'block', index: 61 },
  ]);
  assert.deepEqual(blockStackLayout({ count: 96, shown: 1 }).map((r) => r.index ?? r.hidden), [1, 94, 96]);
});

test('blockStackLayout: rejects counts that are not positive integers', () => {
  assert.throws(() => blockStackLayout({ count: 0 }), /count must be an integer ≥ 1/);
  assert.throws(() => blockStackLayout({ count: 2.5 }), /count must be an integer ≥ 1/);
  assert.throws(() => blockStackLayout({ count: 4, shown: 0 }), /shown must be an integer ≥ 1/);
});

test('barSegments: one geometry for every stacked bar; gaps between neighbors only', () => {
  const segs = barSegments([23, 5, 20], 280);
  const close = (a, b) => Math.abs(a - b) < 1e-9;
  assert.ok(close(segs[0].x, 0) && close(segs[0].width, (23 / 48) * 280 - 2));
  assert.ok(close(segs[1].x, (23 / 48) * 280) && close(segs[1].width, (5 / 48) * 280 - 2));
  assert.ok(close(segs[2].x, (28 / 48) * 280) && close(segs[2].width, (20 / 48) * 280));
  assert.ok(close(segs.reduce((s, x) => s + x.share, 0), 1));
});

test('barSegments rejects empty, negative, non-finite and all-zero inputs', () => {
  assert.throws(() => barSegments([], 100), /non-empty/);
  assert.throws(() => barSegments([1, -1], 100), /finite numbers ≥ 0/);
  assert.throws(() => barSegments([1, Infinity], 100), /finite numbers ≥ 0/);
  assert.throws(() => barSegments([0, 0], 100), /more than 0/);
});

test('formatShare: one decimal from 1 %, two below', () => {
  assert.equal(formatShare(768 / 1576), '48.7%');
  assert.equal(formatShare(0.00818), '0.82%');
  assert.equal(formatShare(0.00354), '0.35%');
  assert.equal(formatShare(1), '100.0%');
});

const TOY_PARTS = [
  { name: 'embedding', value: 128, hue: 1 }, { name: 'attention', value: 512, hue: 2 }, { name: 'MLP', value: 768, hue: 3 },
  { name: 'other', value: 40, hue: 4 }, { name: 'head', value: 128, hue: 5 },
];

test('shareBarLayout: parts under 18 px fold into "others" with a zoomed tail; shares are of the whole (lesson 19)', () => {
  // decoder-anatomy §4 at the toy's 300 px bar: embedding 24.4 · attention 97.5 · MLP 146.2 · other 7.6 · head 24.4 px.
  const { main, tail, unknown } = shareBarLayout(TOY_PARTS, { w: 300 });
  assert.deepEqual(main.map((s) => s.name), ['embedding', 'attention', 'MLP', 'head', 'others']);
  assert.equal(main.at(-1).others, true);
  assert.deepEqual(tail.map((s) => [s.name, s.x, s.width]), [['other', 0, 300]]);
  assert.equal(tail[0].share, 40 / 1576);
  assert.deepEqual(unknown, []);
  assert.deepEqual(shareBarLayout(TOY_PARTS, { w: 300, minSegment: 0 }).tail, []);
});

test('shareBarLayout: an unknown part sits off the scale at a fixed width; the known parts fill the bar', () => {
  const { main, unknown } = shareBarLayout([{ name: 'experts', value: 97, hue: 3 }, { name: 'not published', value: 3, unknown: true }], { w: 300 });
  assert.deepEqual(main.map(({ name, x, width, share }) => [name, x, width, share]), [['experts', 0, 300, 0.97]]);
  assert.deepEqual(unknown.map(({ name, x, width, share }) => [name, x, width, share]), [['not published', 306, 24, 0.03]]);
});

test('shareBarLayout rejects nameless, duplicate, hue-less and negative parts', () => {
  assert.throws(() => shareBarLayout([{ value: 1, hue: 1 }]), /needs a name/);
  assert.throws(() => shareBarLayout([{ name: 'a', value: 1, hue: 1 }, { name: 'a', value: 2, hue: 2 }]), /names must be unique/);
  assert.throws(() => shareBarLayout([{ name: 'a', value: 1 }]), /needs hue 1–5 or unknown: true/);
  assert.throws(() => shareBarLayout([{ name: 'a', value: -1, hue: 1 }]), /value must be a finite number ≥ 0/);
  assert.throws(() => shareBarLayout([{ name: 'a', value: 0, hue: 1 }, { name: 'b', value: 1, unknown: true }]), /at least one known part/); // all-unknown bars draw since S3 (P3-R6); known parts all 0 still throw
});

test('shareBarLayout folds by drawn width: a part whose ideal is 19 px draws at 17 px after its 2 px gap, so it folds (lesson 19)', () => {
  const { main, tail } = shareBarLayout([{ name: 'a', value: 19, hue: 1 }, { name: 'b', value: 81, hue: 2 }], { w: 100 });
  assert.deepEqual(main.map((s) => s.name), ['b', 'others']);
  assert.deepEqual(tail.map((s) => s.name), ['a']);
  const kept = shareBarLayout([{ name: 'a', value: 20, hue: 1 }, { name: 'b', value: 80, hue: 2 }], { w: 100 });
  assert.deepEqual(kept.main.map((s) => s.name), ['a', 'b']);
  assert.ok(kept.main[0].width >= 18);
});

// ---- Shared patch S1: dial (rope §4, conditions a–j) and bars (moe §4, conditions a–f) ----
import { dialLayout, barsLayout } from '../shared/glyphs.js';

const near = (a, b, eps = 1e-9) => Math.abs(a - b) < eps;
const TURN = 2 * Math.PI;

test('dialLayout: the hand is the pair rotated by angle; length = magnitude, direction = rotation (rope frames 2–3)', () => {
  // q pair 1 (0, 2) points straight up (SVG y grows downward), full radius when no scale is given.
  const up = dialLayout({ r: 34, vector: [0, 2] });
  assert.ok(near(up.hand.x, 0) && near(up.hand.y, -34));
  assert.equal(up.magnitudeLabel, '|·| = 2.00');
  // rotated by 3 rad: (−0.282, −1.980) as on the rope page, drawn at r · (x, −y) / |·|
  const turned = dialLayout({ r: 34, vector: [0, 2], angle: 3 });
  assert.ok(near(turned.hand.x, 34 * (-2 * Math.sin(3)) / 2) && near(turned.hand.y, -34 * (2 * Math.cos(3)) / 2));
  assert.equal(dialLayout({ r: 34, vector: [0.5, 0] }).magnitudeLabel, '|·| = 0.50');
});

test('dialLayout: a shared scale keeps two dials comparable; no vector = unit hand, no magnitude printed (condition g)', () => {
  const short = dialLayout({ r: 34, vector: [0.5, 0], scale: 2 });
  assert.ok(near(short.hand.x, 34 * 0.25) && near(short.hand.y, 0));
  const bare = dialLayout({ r: 20, angle: Math.PI / 2 });
  assert.ok(near(bare.hand.x, 0) && near(bare.hand.y, -20));
  assert.equal(bare.magnitudeLabel, null);
});

test('dialLayout: ticks at 0, ¼, ½, ¾ turn with radians for the tooltip only (condition e)', () => {
  const { ticks } = dialLayout({ r: 34 });
  assert.deepEqual(ticks.map((t) => t.title), ['0 rad', 'π/2 rad', 'π rad', '3π/2 rad']);
  assert.ok(near(ticks[1].x, 0) && near(ticks[1].y, -34));
});

test('dialLayout: angles past one turn print "2.4 turns"; within a turn nothing prints (condition i)', () => {
  assert.equal(dialLayout({ r: 34, angle: 2.4 * TURN }).turnsLabel, '2.4 turns');
  assert.equal(dialLayout({ r: 34, angle: 15 }).turnsLabel, '2.4 turns');
  assert.equal(dialLayout({ r: 34, angle: 3 }).turnsLabel, null);
  assert.equal(dialLayout({ r: 34, angle: -15 }).turnsLabel, '−2.4 turns');
});

test('dialLayout: the seen sector is a filled wedge, a full circle once the pair turned fully (condition b)', () => {
  const slow = dialLayout({ r: 34, vector: [0.5, 0], seen: [0, 1.5] });
  assert.equal(slow.seen.full, false);
  assert.match(slow.seen.path, /^M0 0L34 0A34 34 0 0 0 /);
  assert.equal(dialLayout({ r: 34, vector: [0, 2], seen: [0, 15] }).seen.full, true);
  assert.equal(dialLayout({ r: 34 }).seen, null);
  const big = dialLayout({ r: 34, seen: [0, 4] });
  assert.match(big.seen.path, /A34 34 0 1 0 /); // more than half a turn: the large-arc flag is set
});

test('dialLayout: reached draws a ghost hand; "never seen" only when it leaves the seen sector (condition h)', () => {
  const out = dialLayout({ r: 34, vector: [0.5, 0], seen: [0, 1.5], reached: [0, 6.3] });
  assert.ok(near(out.ghost.x, 34 * Math.cos(6.3)) && near(out.ghost.y, -34 * Math.sin(6.3)));
  assert.equal(out.neverSeen, true);
  assert.equal(dialLayout({ r: 34, vector: [0.5, 0], seen: [0, 1.5], reached: [0, 1.575 - 0.075] }).neverSeen, false);
  assert.equal(dialLayout({ r: 34, vector: [0, 2], seen: [0, 15], reached: [0, 63] }).neverSeen, false); // full circle: all seen
  assert.equal(dialLayout({ r: 34 }).ghost, null);
});

test('dialLayout rejects bad input with RangeError', () => {
  assert.throws(() => dialLayout({ r: 0 }), /glyphs.dial: r must be a finite number > 0/);
  assert.throws(() => dialLayout({ r: 34, vector: [1] }), /glyphs.dial: vector must be a pair/);
  assert.throws(() => dialLayout({ r: 34, angle: Number.NaN }), /glyphs.dial: angle must be finite/);
  assert.throws(() => dialLayout({ r: 34, seen: [1, 0] }), /glyphs.dial: seen must be \[from, to\]/);
  assert.throws(() => dialLayout({ r: 34, reached: 3 }), /glyphs.dial: reached must be \[from, to\]/);
  assert.throws(() => dialLayout({ r: 34, vector: [0, 1], scale: 0 }), /glyphs.dial: scale must be a finite number > 0/);
});

test('barsLayout: 8 bars fit 344 px at NUMBER_CELL pitch, heights on an explicit max (moe conditions d, e)', () => {
  const loads = [96, 51, 12, 16, 25, 9, 21, 26];
  const { bars, w, referenceY } = barsLayout({ values: loads, max: 96, h: 120, reference: { value: 32, label: 'fair share 32' } });
  assert.equal(w, 344);
  assert.deepEqual(bars.map((b) => b.x), [0, 43, 86, 129, 172, 215, 258, 301]);
  assert.ok(bars.every((b) => b.width === 40));
  assert.equal(bars[0].height, 120);
  assert.equal(bars[0].y, 0);
  assert.equal(bars[2].height, 15);
  assert.equal(referenceY, 80);
  assert.equal(barsLayout({ values: [32, 35], max: 96, h: 120 }).bars[0].height, 40); // the same max holds between frames
  assert.equal(barsLayout({ values: [1, 2], max: 2, h: 10 }).referenceY, null);
});

test('barsLayout: a value label that would sit on the reference line is lifted clear of it (shared-5)', () => {
  const reference = { value: 32, label: 'fair share 32' }; // max 96, h 120 → the line at y = 80
  const { bars, referenceY } = barsLayout({ values: [96, 32, 30, 36, 29, 12], max: 96, h: 120, reference });
  assert.equal(referenceY, 80);
  assert.deepEqual(bars.map((b) => b.labelY), [-5, 71, 73.5, 70, 74.75, 100]); // 96, 36 and 12 keep y − 5; the others end 4 px above the line or above their own place
  const clear = bars.every((b) => b.labelY <= referenceY - 4 || b.labelY - referenceY >= 8);
  assert.ok(clear, 'no value label baseline lies within 8 px of the reference line, nor below it within the glyph height');
  assert.equal(barsLayout({ values: [32], max: 96, h: 120 }).bars[0].labelY, 75); // no reference line → y − 5
});

test('barsLayout: an explicit width sets the pitch; zero is a bar of height 0', () => {
  const { bars, w } = barsLayout({ values: [6, 8, 1, 1, 4], max: 20, h: 100, w: 215 });
  assert.equal(w, 215);
  assert.deepEqual(bars.map((b) => b.x), [0, 43, 86, 129, 172]);
  assert.equal(barsLayout({ values: [0, 4], max: 4, h: 50 }).bars[0].height, 0);
});

test('barsLayout rejects empty, negative and over-max values, a missing max and mismatched labels', () => {
  assert.throws(() => barsLayout({ values: [], max: 1, h: 10 }), /glyphs.bars: values must be a non-empty array/);
  assert.throws(() => barsLayout({ values: [1, -1], max: 1, h: 10 }), /glyphs.bars: values must be finite numbers ≥ 0/);
  assert.throws(() => barsLayout({ values: [1, 2], h: 10 }), /glyphs.bars: max must be a finite number > 0/);
  assert.throws(() => barsLayout({ values: [1, 5], max: 4, h: 10 }), /glyphs.bars: value 5 exceeds max 4/);
  assert.throws(() => barsLayout({ values: [1, 2], max: 2, h: 10, labels: ['a'] }), /glyphs.bars: labels must match values/);
  assert.throws(() => barsLayout({ values: [1], max: 2, h: 10, reference: { value: 3, label: 'x' } }), /glyphs.bars: reference must be \{ value ≤ max, label \}/);
  assert.throws(() => barsLayout({ values: [1], max: 2, h: 0 }), /glyphs.bars: h must be a finite number > 0/);
});

// ---- Shared prep S3 (Plan 3, Task 3): curvePlot (P3-R10) and roofline (gpu-primer §4) ----
import { curvePlotLayout, rooflineLayout, formatTick } from '../shared/glyphs.js';

const LOG_AXES = { xAxis: { label: 'x', log: true, ticks: [0.1, 1, 10, 100, 1000, 10000] }, yAxis: { label: 'y', log: true, ticks: [0.1, 1, 10, 100, 1000, 10000] } };

test('curvePlotLayout: a log axis gives every decade the same width, and ticks sit on their values', () => {
  const L = curvePlotLayout({ w: 360, h: 240, ...LOG_AXES, series: [{ points: [[0.1, 0.335], [295.2, 989], [1e4, 989]], label: 'roof' }] });
  const xs = L.xTicks.map((t) => t.x);
  const widths = xs.slice(1).map((x, i) => x - xs[i]);
  widths.forEach((d) => assert.ok(near(d, widths[0], 1e-9), `decade widths ${widths.join(', ')}`));
  assert.ok(near(xs[0], L.plot.left) && near(xs.at(-1), L.plot.right));
  assert.ok(near(L.toX(Math.sqrt(10)), (L.toX(1) + L.toX(10)) / 2)); // half a decade is half the width
  assert.deepEqual(L.xTicks.map((t) => t.label), ['0.1', '1', '10', '100', '1,000', '10,000']);
  const ys = L.yTicks.map((t) => t.y);
  assert.ok(ys[0] > ys.at(-1), 'y grows upward on screen');
  assert.ok(near(ys[0], L.plot.bottom) && near(ys.at(-1), L.plot.top));
  assert.ok(L.plot.left > 0 && L.plot.right <= 360 && L.plot.bottom < 240 && L.plot.top > 0, 'the axes fit inside w × h');
});

test('curvePlotLayout: a linear axis maps the domain onto the plot; the domain defaults to the tick extent', () => {
  const L = curvePlotLayout({ w: 300, h: 200, xAxis: { label: 'step', ticks: [0, 50, 100] }, yAxis: { label: 'lr', ticks: [0, 1, 2, 3] }, series: [{ points: [[0, 0], [10, 3], [100, 0.3]], label: 'lr' }] });
  assert.ok(near(L.toX(50), (L.plot.left + L.plot.right) / 2));
  assert.ok(near(L.toY(1.5), (L.plot.top + L.plot.bottom) / 2));
  assert.equal(L.series[0].path.split('L').length, 3);
  assert.match(L.series[0].path, /^M/);
});

test('curvePlotLayout: refY is drawn at its value, like the bars reference line; markers, bands and labels are placed', () => {
  const L = curvePlotLayout({
    w: 360, h: 220, xAxis: { label: 'N', log: true, ticks: [1e8, 1e9, 1e10, 1e11] }, yAxis: { label: 'loss', ticks: [1.5, 2, 2.5, 3] },
    series: [{ points: [[1e8, 2.9], [1e11, 1.9]], label: 'fit', style: 'solid' }, { points: [[1e8, 2.5], [1e11, 1.6]], label: 'branch', style: 'muted' }],
    markers: [{ x: 1e10, y: 1.96, label: 'same loss', followed: true }, { x: 1e11, y: 1.9, label: 'right edge' }],
    bands: [{ from: 1e8, to: 1e9, label: 'warmup' }, { from: 1e9, to: 1e11, label: 'stable' }],
    refY: { value: 1.96, label: 'same loss 1.960' },
  });
  assert.ok(near(L.refY.y, L.toY(1.96)));
  assert.equal(L.refY.label, 'same loss 1.960');
  assert.ok(near(L.refY.x1, L.plot.left) && near(L.refY.x2, L.plot.right));
  assert.deepEqual(L.series.map((s) => s.style), ['solid', 'muted']);
  assert.ok(near(L.markers[0].x, L.toX(1e10)) && near(L.markers[0].y, L.toY(1.96)));
  assert.equal(L.markers[0].followed, true);
  assert.equal(L.markers[1].labelAnchor, 'end', 'a marker at the right edge prints its label to its left');
  assert.equal(L.markers[0].labelAnchor, 'start');
  const [warm, stable] = L.bands;
  assert.ok(warm.x + warm.width < stable.x, 'touching bands keep a visible gap');
  assert.ok(near(warm.x, L.toX(1e8) + 1) && near(stable.x + stable.width, L.toX(1e11) - 1));
  assert.equal(curvePlotLayout({ w: 300, h: 200, xAxis: { ticks: [0, 1] }, yAxis: { ticks: [0, 1] }, series: [] }).refY, null);
});

test('curvePlotLayout rejects dashed series, out-of-domain points, non-positive log values and bad sizes', () => {
  const base = { w: 300, h: 200, xAxis: { label: 'x', ticks: [0, 10] }, yAxis: { label: 'y', ticks: [0, 10] } };
  assert.throws(() => curvePlotLayout({ ...base, series: [{ points: [[1, 1]], style: 'dashed' }] }), /glyphs.curvePlot: series style must be solid or muted \(no dashed series, P3-R11\)/);
  assert.throws(() => curvePlotLayout({ ...base, series: [{ points: [[11, 1]] }] }), /glyphs.curvePlot: point \(11, 1\) is outside the domain/);
  assert.throws(() => curvePlotLayout({ ...base, markers: [{ x: 5, y: -1 }] }), /outside the domain/);
  assert.throws(() => curvePlotLayout({ ...base, xAxis: { log: true, ticks: [0, 10] } }), /glyphs.curvePlot: a log axis needs a domain > 0/);
  assert.throws(() => curvePlotLayout({ ...base, w: 0 }), /glyphs.curvePlot: w and h must be finite numbers > 0/);
  assert.throws(() => curvePlotLayout({ ...base, bands: [{ from: 4, to: 2 }] }), /glyphs.curvePlot: band from must be < to/);
  assert.throws(() => curvePlotLayout({ ...base, refY: { value: 12, label: 'x' } }), /glyphs.curvePlot: refY must be \{ value inside the y domain, label \}/);
  assert.throws(() => curvePlotLayout({ ...base, xAxis: { ticks: [3] } }), /glyphs.curvePlot: x axis needs a domain/);
  assert.throws(() => curvePlotLayout({ ...base, series: [{ points: [[1, 1]], tone: 'accent' }] }), /glyphs.curvePlot: series tone must be one of/);
});

test('formatTick: thousands separators, short decimals, a real minus', () => {
  assert.deepEqual([0.1, 1, 10, 1000, 10000, 2.5, -3, 0.25, 1e6].map(formatTick), ['0.1', '1', '10', '1,000', '10,000', '2.5', '−3', '0.25', '1,000,000']);
});

test('rooflineLayout: the ridge sits at peak ÷ bandwidth on the log axis; points ride the roof with their bound (gpu-primer frames 6–8)', () => {
  const L = rooflineLayout({ w: 360, h: 240, peakTflops: 989, bandwidthTBps: 3.35, points: [{ intensity: 2, label: '4 tokens', followed: true }, { intensity: 2048, label: '4,096 tokens' }] });
  assert.ok(near(L.ridge.value, (989 * 1e12) / (3.35 * 1e12)));
  assert.ok(near(L.ridge.x, L.plot.toX(989 / 3.35)));
  assert.ok(near(L.ridge.y, L.plot.toY(989)));
  assert.equal(L.ridge.label, 'ridge point 295');
  assert.deepEqual(L.points.map((p) => p.bound), ['memory', 'compute']);
  assert.ok(near(L.points[0].attainable, 6.7) && near(L.points[1].attainable, 989));
  assert.ok(near(L.points[0].y, L.plot.toY(6.7)));
  assert.deepEqual(L.plot.markers.map((m) => m.label), ['4 tokens · 2', '4,096 tokens · 2,048']);
  assert.equal(L.plot.markers[0].followed, true);
  assert.deepEqual(L.plot.series.map((s) => [s.tone, s.label]), [['memory', '3.35 TB/s'], ['compute', '989 TFLOPS']]);
  assert.deepEqual(L.plot.xTicks.map((t) => t.value), [0.1, 1, 10, 100, 1000, 10000]); // default xDomain [0.1, 1e4]
  assert.equal(L.plot.bands.length, 0);
});

test('rooflineLayout: ridgeRange draws a band with both ends printed instead of one bend label (Rubin)', () => {
  const L = rooflineLayout({ w: 360, h: 240, peakTflops: 35000, bandwidthTBps: 22, yDomain: [1, 1e5], ridgeRange: [1590.9, 1822.9], points: [] });
  assert.equal(L.ridge.label, null);
  assert.equal(L.plot.bands.length, 1);
  assert.equal(L.plot.bands[0].label, 'ridge 1,591–1,823');
  assert.ok(near(L.plot.bands[0].x, L.plot.toX(1590.9) + 1));
});

test('rooflineLayout: an intensity exactly at the ridge is compute-bound; bad input throws', () => {
  assert.equal(rooflineLayout({ w: 360, h: 240, peakTflops: 1000, bandwidthTBps: 4, points: [{ intensity: 250 }] }).points[0].bound, 'compute');
  assert.throws(() => rooflineLayout({ w: 360, h: 240, peakTflops: 0, bandwidthTBps: 3.35 }), /glyphs.roofline: peakTflops and bandwidthTBps must be finite numbers > 0/);
  assert.throws(() => rooflineLayout({ w: 360, h: 240, peakTflops: 989, bandwidthTBps: 3.35, points: [{ intensity: 0 }] }), /glyphs.roofline: point intensity must be a finite number > 0/);
  assert.throws(() => rooflineLayout({ w: 360, h: 240, peakTflops: 989, bandwidthTBps: 3.35, ridgeRange: [10, 5] }), /glyphs.roofline: ridgeRange must be \[low, high\]/);
});

// ---- laneTimeline (gpu-primer §4, P3-R9) ----
import { laneTimelineLayout } from '../shared/glyphs.js';

const GPIPE_STAGE_1 = ['F1', 'F2', 'F3', 'F4', null, null, null, null, 'B1', 'B2', 'B3', 'B4']; // 3 stages × 4 micro-batches, stage 1
const cells = (row) => row.map((c, t) => ({ from: t, to: t + 1, kind: c ? (c[0] === 'F' ? 'forward' : 'backward') : 'idle', label: c ?? undefined }));

test('laneTimelineLayout: segments scale onto the track; 12 columns at 43 px fit the 568 px stage row (parallelism frame 7)', () => {
  const L = laneTimelineLayout({ w: 568, lanes: [{ label: 'stage 1', segments: cells(GPIPE_STAGE_1) }] });
  assert.equal(L.gutter, 52);
  assert.equal(L.track, 516);
  assert.equal(L.scale, 43);
  const segs = L.segments;
  assert.deepEqual(segs.slice(0, 3).map((s) => [s.x, s.width]), [[52, 40], [95, 40], [138, 40]]); // a 3 px gap between touching cells
  assert.equal(segs.at(-1).x + segs.at(-1).width, 568);
  assert.deepEqual(segs.map((s) => s.kind).slice(3, 6), ['forward', 'idle', 'idle']);
  assert.equal(segs[0].textInside, true);
  assert.equal(segs[0].text, 'F1');
});

test('laneTimelineLayout: an explicit scale wins; a short segment prints its text beside it (gpu-primer frame 7)', () => {
  const L = laneTimelineLayout({ w: 580, scale: 520 / 40.1, lanes: [
    { label: 'HBM', segments: [{ from: 0, to: 40.1, kind: 'memory', label: 'memory 40.1 µs' }] },
    { label: 'math', segments: [{ from: 0, to: 0.54, kind: 'compute', label: 'compute 0.54 µs' }, { from: 0.54, to: 40.1, kind: 'idle' }] },
  ] });
  const [memory, compute, idle] = L.segments;
  assert.ok(near(memory.width, 520));
  assert.equal(memory.textInside, true);
  assert.equal(compute.textInside, false, 'a 7 px segment prints its label beside it');
  assert.ok(compute.textX > compute.x + compute.width);
  assert.equal(idle.hatched, true);
  assert.deepEqual(L.lanes.map((l) => l.label), ['HBM', 'math']);
  assert.ok(L.lanes[1].y > L.lanes[0].y);
});

test('laneTimelineLayout: labels: false prints no text, and every segment still has a title naming its lane and kind (parallelism, 78 columns)', () => {
  const row = Array.from({ length: 78 }, (_, t) => ({ from: t, to: t + 1, kind: t % 3 ? 'forward' : 'idle', label: `F${t}` }));
  const L = laneTimelineLayout({ w: 572, labels: false, lanes: [{ label: 'stage 1', segments: row }] });
  assert.equal(L.segments.length, 78);
  assert.ok(L.segments.every((s) => s.text === null));
  assert.ok(L.segments.every((s) => typeof s.title === 'string' && s.title.length > 0));
  assert.equal(L.segments[1].title, 'stage 1: forward'); // no micro-batch id, not even on hover (parallelism §6, lesson 18)
  assert.equal(L.segments[0].title, 'stage 1: idle');
  const on = laneTimelineLayout({ w: 572, lanes: [{ label: 'stage 1', segments: row.slice(0, 2) }] });
  assert.equal(on.segments[1].title, 'stage 1: F1 (forward)');
});

test('laneTimelineLayout: cap cuts a long lane at the track with an arrow and its printed label (cluster-topology frame 3)', () => {
  const L = laneTimelineLayout({ w: 572, scale: 1.2, cap: { at: 520 / 1.2, label: 'continues: 250%' }, lanes: [
    { label: 'compute', segments: [{ from: 0, to: 100, kind: 'compute', label: 'compute 100%' }] },
    { label: 'comm', segments: [{ from: 0, to: 250.4, kind: 'comm', label: 'comm 250%' }] },
    { label: 'network', segments: [{ from: 0, to: 2254, kind: 'comm', label: 'comm 2,254%' }] },
  ] });
  const [compute, nvlink, network] = L.segments;
  assert.ok(near(compute.width, 120));
  assert.equal(nvlink.capped, false);
  assert.equal(network.capped, true);
  assert.ok(near(network.arrow.tip, L.gutter + 520));
  assert.ok(network.x + network.width <= network.arrow.base + 1e-9);
  assert.equal(network.capLabel, 'continues: 250%');
  assert.equal(network.textAnchor, 'start', 'a capped segment prints its own label at its start, the cap label at its end');
});

test('laneTimelineLayout: ticks mark checkpoint saves and gaps open restarts (scale-reliability frame 9)', () => {
  const L = laneTimelineLayout({ w: 580, lanes: [{ label: 'run', segments: [
    { from: 0, to: 40, kind: 'compute' }, { from: 40, to: 47, kind: 'lost', label: 'lost' }, { from: 50, to: 90, kind: 'compute' },
  ] }], ticks: [{ t: 13.6, label: 'save' }, { t: 27.2, label: 'save' }], gaps: [{ from: 47, to: 50, label: 'restart' }] });
  assert.ok(near(L.ticks[0].x, L.gutter + 13.6 * L.scale));
  assert.equal(L.ticks[1].label, 'save');
  assert.ok(near(L.gaps[0].x, L.gutter + 47 * L.scale) && near(L.gaps[0].width, 3 * L.scale));
  assert.equal(L.gaps[0].label, 'restart');
  assert.equal(L.segments[1].hatched, true);
  assert.equal(L.segments[1].kind, 'lost');
  assert.ok(L.top > 0, 'labeled ticks get a row above the lanes');
  assert.ok(L.height > L.top + 24, 'gap labels get a row below the lanes');
});

test('laneTimelineLayout rejects unknown kinds, reversed segments, a lane past the track without a cap, and bad ticks', () => {
  const lane = (segments) => [{ label: 'a', segments }];
  assert.throws(() => laneTimelineLayout({ w: 300, lanes: lane([{ from: 0, to: 1, kind: 'save' }]) }), /glyphs.laneTimeline: kind must be one of compute, memory, comm, idle, forward, backward, lost/);
  assert.throws(() => laneTimelineLayout({ w: 300, lanes: lane([{ from: 2, to: 1, kind: 'compute' }]) }), /glyphs.laneTimeline: a segment needs 0 ≤ from < to/);
  assert.throws(() => laneTimelineLayout({ w: 300, scale: 10, lanes: lane([{ from: 0, to: 100, kind: 'compute' }]) }), /glyphs.laneTimeline: lane "a" runs past the track; pass cap or a smaller scale/);
  assert.throws(() => laneTimelineLayout({ w: 300, lanes: [] }), /glyphs.laneTimeline: lanes must be a non-empty array/);
  assert.throws(() => laneTimelineLayout({ w: 300, lanes: lane([{ from: 0, to: 1, kind: 'compute' }]), scale: 100, ticks: [{ t: 5 }] }), /glyphs.laneTimeline: tick t must lie on the track/);
  assert.throws(() => laneTimelineLayout({ w: 300, lanes: lane([{ from: 0, to: 1, kind: 'compute' }]), scale: -1 }), /glyphs.laneTimeline: scale must be a finite number > 0/);
});

// ---- bitLayout (gpu-primer §4 L110–113) ----
import { bitLayoutLayout, bitFields } from '../shared/glyphs.js';

// The storyboard's FORMATS entries (gpu-primer §8), the shape math/roofline.js exports.
const STORYBOARD_FORMATS = {
  fp32: { bits: 32, layout: '1/8/23' }, bf16: { bits: 16, layout: '1/8/7' }, fp16: { bits: 16, layout: '1/5/10' },
  fp8_e4m3: { bits: 8, layout: '1/4/3' }, fp8_e5m2: { bits: 8, layout: '1/5/2' },
  mxfp4: { bits: 4, layout: '1/2/1', blockSize: 32, scaleBits: 8 }, nvfp4: { bits: 4, layout: '1/2/1', blockSize: 16, scaleBits: 8 },
};
const widthsOf = (format) => bitLayoutLayout({ format }).fields.map((f) => [f.role, f.width]);

test('bitLayoutLayout: sign / exponent / mantissa widths per FORMATS key, 14 px a bit (gpu-primer frame 10)', () => {
  assert.deepEqual(widthsOf(STORYBOARD_FORMATS.bf16), [['sign', 14], ['exponent', 112], ['mantissa', 98]]);
  assert.deepEqual(widthsOf(STORYBOARD_FORMATS.fp8_e4m3), [['sign', 14], ['exponent', 56], ['mantissa', 42]]);
  assert.deepEqual(widthsOf(STORYBOARD_FORMATS.nvfp4), [['sign', 14], ['exponent', 28], ['mantissa', 14]]);
  for (const [key, format] of Object.entries(STORYBOARD_FORMATS)) {
    const L = bitLayoutLayout({ format });
    assert.equal(L.width, format.bits * 14, key);
    assert.equal(L.cells.length, format.bits, key);
  }
  assert.equal(bitLayoutLayout({ format: STORYBOARD_FORMATS.bf16 }).width, 224); // "16 cells × 14 px = 224 px"
  assert.deepEqual(bitLayoutLayout({ format: STORYBOARD_FORMATS.fp8_e4m3 }).fields.map((f) => f.text), ['S', 'E4', 'M3']);
});

const ROOFLINE = await import('../math/roofline.js').catch(() => null);
test('bitLayoutLayout: every key of math/roofline.js FORMATS draws its own bit count', { skip: ROOFLINE ? false : 'math/roofline.js lands with stream S3-B' }, () => {
  for (const [key, format] of Object.entries(ROOFLINE.FORMATS)) {
    const L = bitLayoutLayout({ format });
    assert.equal(L.cells.length, format.bits, key);
    assert.equal(L.fields.reduce((s, f) => s + f.bits, 0), format.bits, key);
  }
});

test('bitLayoutLayout: an explicit field list and a shared scale with its bracket (NVFP4 block)', () => {
  const numbers = [...bitFields('1/2/1'), ...bitFields('1/2/1'), ...bitFields('1/2/1')];
  const L = bitLayoutLayout({ fields: [...numbers, { role: 'scale', bits: 8 }], sharedBy: 16 });
  assert.equal(L.width, 280); // "3 × 4 + 8 = 20 bit cells × 14 px = 280 px"
  assert.deepEqual(L.bracket, { x0: 168, x1: 280, text: 'shared by 16 numbers', textX: 280 - (20 * 6.6) / 2 }); // slid left: the text never runs past the row
  assert.equal(L.fields.at(-1).text, 'scale');
  assert.deepEqual(L.fields.slice(0, 3).map((f) => f.text), ['S', 'E2', 'M1']);
  assert.deepEqual(bitLayoutLayout({ fields: bitFields('1/2/1'), bitW: 11 }).fields.map((f) => f.text), ['S', 'E2', null]); // "M1" is wider than 11 px + 2: its title names it
  assert.equal(bitLayoutLayout({ fields: [{ role: 'sign', bits: 1 }], bitW: 10 }).width, 10);
});

test('bitLayoutLayout rejects unknown roles, bad bit counts, a sharedBy with no scale field and a bad layout string', () => {
  assert.throws(() => bitLayoutLayout({ fields: [{ role: 'fraction', bits: 3 }] }), /glyphs.bitLayout: role must be one of sign, exponent, mantissa, scale/);
  assert.throws(() => bitLayoutLayout({ fields: [{ role: 'sign', bits: 0 }] }), /glyphs.bitLayout: bits must be an integer ≥ 1/);
  assert.throws(() => bitLayoutLayout({ fields: [{ role: 'sign', bits: 1 }], sharedBy: 16 }), /glyphs.bitLayout: sharedBy needs a scale field/);
  assert.throws(() => bitLayoutLayout({ format: { layout: '1/8' } }), /glyphs.bitLayout: layout must read "sign\/exponent\/mantissa"/);
  assert.throws(() => bitLayoutLayout({}), /glyphs.bitLayout: pass fields or format/);
});

// ---- shareBar options (P3-R6 unknown parts as value null and all-unknown bars, P3-R7 tail basis, lesson 24 hatched parts) ----
test('shareBarLayout: value null is an unknown part with no share; known shares are of the known total (training-pipeline frame 8)', () => {
  const glm5 = [{ name: 'pretrain', value: 27e12, hue: 1 }, { name: 'mid-train', value: 1.55e12, hue: 2 }, { name: 'post-training', value: null }];
  const { main, unknown, unknownLabel } = shareBarLayout(glm5, { w: 560 });
  assert.deepEqual(main.map((s) => s.name), ['pretrain', 'mid-train']);
  assert.equal(formatShare(main[0].share), '94.6%');
  assert.equal(formatShare(main[1].share), '5.4%');
  assert.deepEqual(unknown.map(({ name, x, width, share }) => [name, x, width, share]), [['post-training', 566, 24, null]]);
  assert.equal(unknownLabel, 'post-training: not published');
});

test('shareBarLayout: a bar whose parts are all unknown draws each at 24 px with "no published shares" and no scale (P3-R6, Kimi K3)', () => {
  const kimi = [{ name: 'pretrain', value: null }, { name: 'mid-train', value: null }, { name: 'post-training', value: null }];
  const { main, tail, unknown, unknownLabel, allUnknown } = shareBarLayout(kimi, { w: 300 });
  assert.equal(allUnknown, true);
  assert.deepEqual(main, []);
  assert.deepEqual(tail, []);
  assert.deepEqual(unknown.map((p) => [p.x, p.width]), [[0, 24], [26, 24], [52, 24]]);
  assert.equal(unknownLabel, 'no published shares');
  assert.equal(shareBarLayout([{ name: 'x', value: 5, unknown: true }]).allUnknown, true); // the older unknown: true form too
});

test('shareBarLayout: the existing unknown: true parts keep their value, share of the whole and printed name (decoder-anatomy)', () => {
  const { unknown, unknownLabel, allUnknown } = shareBarLayout([{ name: 'experts', value: 97, hue: 3 }, { name: 'not published', value: 3, unknown: true }], { w: 300 });
  assert.equal(unknown[0].share, 0.03);
  assert.equal(unknownLabel, 'not published');
  assert.equal(allUnknown, false);
});

test('shareBarLayout: tailBasis "tail" gives the zoomed bar shares of the tail that add to 100 % (P3-R7, midtraining frame 6)', () => {
  const glm5 = [{ name: '4K', value: 27e12, hue: 1 }, { name: '32K', value: 1e12, hue: 2 }, { name: '128K', value: 0.5e12, hue: 3 }, { name: '200K', value: 0.05e12, hue: 4 }];
  const whole = shareBarLayout(glm5, { w: 560 });
  const tailed = shareBarLayout(glm5, { w: 560, tailBasis: 'tail' });
  assert.deepEqual(tailed.tail.map((s) => s.name), ['32K', '128K', '200K']);
  assert.deepEqual(tailed.tail.map((s) => formatShare(s.share)), ['64.5%', '32.3%', '3.2%']);
  assert.ok(near(tailed.tail.reduce((s, p) => s + p.share, 0), 1));
  assert.deepEqual(tailed.main.map((s) => s.share), whole.main.map((s) => s.share), 'the main bar keeps shares of the whole');
  assert.equal(whole.tail.reduce((s, p) => s + p.share, 0) < 0.06, true, 'the default basis stays the whole');
  assert.throws(() => shareBarLayout(glm5, { tailBasis: 'part' }), /glyphs.shareBar: tailBasis must be "whole" or "tail"/);
});

test('shareBarLayout: a hatched part keeps its hue and share and is flagged for the hatch (README lesson 24)', () => {
  const run = [{ name: 'useful', value: 26.62, hue: 3 }, { name: 'below peak', value: 4.22, hue: 4 }, { name: 'lost to failures', value: 2.62, hue: 5, hatched: true }];
  const { main } = shareBarLayout(run, { w: 400 });
  assert.deepEqual(main.map((s) => [s.name, s.hatched === true]), [['useful', false], ['below peak', false], ['lost to failures', true]]);
  assert.throws(() => shareBarLayout([{ name: 'a', value: 1, hue: 1 }, { name: 'b', value: null, hatched: true }]), /glyphs.shareBar: part "b" is unknown and cannot be hatched/);
});

test('curvePlotLayout: a series label sits above its end, or above-left of its midpoint (labelAt: mid); a band label at the top or bottom', () => {
  const L = curvePlotLayout({ w: 300, h: 200, xAxis: { ticks: [0, 10] }, yAxis: { ticks: [0, 10] },
    series: [{ points: [[0, 0], [10, 10]], label: 'slope', labelAt: 'mid' }, { points: [[0, 5], [10, 5]], label: 'flat' }],
    bands: [{ from: 2, to: 4, label: 'top' }, { from: 6, to: 8, label: 'low', labelAt: 'bottom' }] });
  assert.ok(near(L.series[0].labelPos.x, L.toX(5) - 6) && near(L.series[0].labelPos.y, L.toY(5) - 6));
  assert.ok(near(L.series[1].labelPos.x, L.toX(10)) && near(L.series[1].labelPos.y, L.toY(5) - 6));
  assert.equal(L.bands[0].labelY, L.plot.top + 11);
  assert.equal(L.bands[1].labelY, L.plot.bottom - 5);
  assert.throws(() => curvePlotLayout({ w: 300, h: 200, xAxis: { ticks: [0, 10] }, yAxis: { ticks: [0, 10] }, series: [{ points: [], labelAt: 'start' }] }), /series labelAt must be end or mid/);
});
