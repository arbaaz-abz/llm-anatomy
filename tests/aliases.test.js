import { test } from 'node:test';
import assert from 'node:assert/strict';

test('@shared/ and @math/ resolve in node the way the import map resolves them in the browser', async () => {
  const { NUMBER_CELL } = await import('@shared/glyphs.js');
  const { softmax } = await import('@math/core.js');
  assert.equal(NUMBER_CELL, 40);
  assert.deepEqual(softmax([0, 0]), [0.5, 0.5]);
});
