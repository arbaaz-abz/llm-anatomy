import { test } from 'node:test';
import assert from 'node:assert/strict';
import { checkOptions } from '../shared/ui/choice.js';

const OPTS = [{ value: 'A', label: 'A' }, { value: 'B', label: 'B' }, { value: 'both', label: 'both' }];

test('checkOptions accepts unique, labeled options containing the value', () => {
  assert.equal(checkOptions(OPTS, 'A'), true);
});

test('checkOptions names what is wrong', () => {
  assert.throws(() => checkOptions([], 'A'), /non-empty/);
  assert.throws(() => checkOptions([...OPTS, { value: 'A', label: 'again' }], 'A'), /unique/);
  assert.throws(() => checkOptions([{ value: 1, label: '' }], 1), /needs a label/);
  assert.throws(() => checkOptions(OPTS, 'C'), /value C is not one of A, B, both/);
});
