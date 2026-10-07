import { test } from 'node:test';
import assert from 'node:assert/strict';
import { valueColor, valueLevel, levelFromFill, tokenWidth, requestSlot, formatCell, pixelFill, blockStackLayout } from '../shared/glyphs.js';

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
