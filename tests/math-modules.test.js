import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readdir } from 'node:fs/promises';

// The coverage gate is one aggregate 80% over math/**: a module without its own test file drags every branch down.
test('every math/<name>.js has tests/<name>.test.js', async () => {
  const modules = (await readdir(new URL('../math/', import.meta.url))).filter((f) => f.endsWith('.js'));
  const tests = new Set(await readdir(new URL('./', import.meta.url)));
  const missing = modules.filter((f) => !tests.has(f.replace(/\.js$/, '.test.js')));
  assert.deepEqual(missing, [], `math modules without a test file: ${missing.join(', ')}`);
});
