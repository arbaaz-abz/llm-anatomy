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
  assert.throws(() => shareBarLayout([{ name: 'a', value: 1, unknown: true }]), /at least one known part/);
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
