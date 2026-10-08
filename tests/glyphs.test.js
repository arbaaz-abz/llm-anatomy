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
