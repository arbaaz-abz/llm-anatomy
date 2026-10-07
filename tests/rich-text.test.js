import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseRichText } from '../shared/rich-text.js';

test('plain text stays one text segment', () => {
  assert.deepEqual(parseRichText('Just words.'), [{ type: 'text', text: 'Just words.' }]);
  assert.deepEqual(parseRichText(''), []);
});

test('[[slug]] links a lesson and `x` is code, in order', () => {
  assert.deepEqual(parseRichText('See [[kv-cache]] for `W_O` and [[rope]].'), [
    { type: 'text', text: 'See ' }, { type: 'lesson', slug: 'kv-cache' }, { type: 'text', text: ' for ' },
    { type: 'code', text: 'W_O' }, { type: 'text', text: ' and ' }, { type: 'lesson', slug: 'rope' }, { type: 'text', text: '.' },
  ]);
});

test('markup that is not a slug stays text; HTML is never parsed', () => {
  assert.deepEqual(parseRichText('[[Not A Slug]] <b>x</b>'), [{ type: 'text', text: '[[Not A Slug]] <b>x</b>' }]);
  assert.throws(() => parseRichText(42), /text must be a string/);
});
