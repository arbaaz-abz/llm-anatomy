import { test } from 'node:test';
import assert from 'node:assert/strict';
import { sliderValue, nearestIndex } from '../shared/ui/slider.js';

test('plain sliders return the numeric value', () => {
  assert.equal(sliderValue('42', {}), 42);
});

test('value lists snap: the range indexes into the list', () => {
  const values = [1024, 4096, 32768, 131072, 1048576];
  assert.equal(sliderValue('0', { values }), 1024);
  assert.equal(sliderValue('4', { values }), 1048576);
  assert.equal(sliderValue('9', { values }), 1048576);
});

test('nearestIndex snaps to the closest entry', () => {
  const values = [1024, 4096, 32768];
  assert.equal(nearestIndex(values, 4096), 1);
  assert.equal(nearestIndex(values, 3000), 1);
  assert.equal(nearestIndex(values, 2000), 0);
  assert.equal(nearestIndex(values, 1), 0);
  assert.equal(nearestIndex(values, 999999), 2);
});
