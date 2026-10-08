import { test } from 'node:test';
import assert from 'node:assert/strict';
import { checkOptions, noteLines, firstEnabled } from '../shared/ui/choice.js';

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

// ---- Per-option disabled with a visible note (S3.5, ruling P3-R12) ----
const FMT = [{ value: 'bf16', label: 'BF16' }, { value: 'fp8', label: 'FP8' }, { value: 'fp4', label: 'FP4', disabled: true, note: 'no FP4 figure in data' }];
const RUBIN = [
  { value: 'bf16', label: 'BF16', disabled: true, note: 'no settled BF16/FP8 figure' },
  { value: 'fp8', label: 'FP8', disabled: true, note: 'no settled BF16/FP8 figure' },
  { value: 'fp4', label: 'FP4', note: 'no FP4 figure in data' },
];

test('checkOptions accepts a disabled option with a note, and a note on an enabled option', () => {
  assert.equal(checkOptions(FMT, 'bf16'), true);
  assert.equal(checkOptions(RUBIN, 'fp4'), true);
});

test('checkOptions: a disabled option needs a note, the value must be enabled, and one option must stay enabled', () => {
  assert.throws(() => checkOptions([{ value: 'a', label: 'a' }, { value: 'b', label: 'b', disabled: true }], 'a'), /choice: disabled option b needs a note saying why/);
  assert.throws(() => checkOptions([{ value: 'a', label: 'a' }, { value: 'b', label: 'b', disabled: true, note: ' ' }], 'a'), /needs a note/);
  assert.throws(() => checkOptions(FMT, 'fp4'), /choice: value fp4 is disabled \(no FP4 figure in data\)/);
  assert.throws(() => checkOptions(FMT.map((o) => ({ ...o, disabled: true, note: 'off' })), 'bf16'), /at least one option must be enabled/);
  assert.throws(() => checkOptions([{ value: 'a', label: 'a', disabled: 'yes', note: 'x' }, { value: 'b', label: 'b' }], 'b'), /disabled must be true or false/);
});

test('noteLines: nothing while every option is enabled; one line per note, prefixed by the option labels it covers', () => {
  assert.deepEqual(noteLines(OPTS), []);
  assert.deepEqual(noteLines(FMT), ['FP4: no FP4 figure in data']);
  assert.deepEqual(noteLines(RUBIN), ['BF16, FP8: no settled BF16/FP8 figure'], 'a note shared by two options prints once');
  const mixed = [{ value: 1, label: 'one', disabled: true, note: 'x' }, { value: 2, label: 'two' }, { value: 3, label: 'three', disabled: true, note: 'y' }];
  assert.deepEqual(noteLines(mixed), ['one: x', 'three: y']);
});

test('firstEnabled: the value of the first option that is not disabled', () => {
  assert.equal(firstEnabled(FMT), 'bf16');
  assert.equal(firstEnabled(RUBIN), 'fp4');
});
