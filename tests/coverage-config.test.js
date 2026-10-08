import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readdir, readFile, stat } from 'node:fs/promises';

// The coverage gate must see every lesson's page formatter (format.js), in every track, built or not:
// a track whose glob is missing lets untested formatters slip under the 80% line.
const pkg = JSON.parse(await readFile(new URL('../package.json', import.meta.url), 'utf8'));
const exists = (p) => stat(new URL(`../${p}`, import.meta.url)).then(() => true, () => false);

const TRACKS = JSON.parse(await readFile(new URL('../shared/concepts.json', import.meta.url), 'utf8')).tracks.map((t) => t.id);

test('the coverage gate includes every track\'s lesson format.js, built or not', () => {
  assert.deepEqual(TRACKS, ['architecture', 'training', 'serving']);
  for (const track of TRACKS) {
    assert.match(pkg.scripts.coverage, new RegExp(`--test-coverage-include=\\\\?"${track}/concepts/\\*\\*/format\\.js\\\\?"`), `${track} format.js in coverage`);
  }
});

test('no concepts/ directory exists outside the tracks in shared/concepts.json', async () => {
  for (const dir of ['hub', 'gallery']) assert.equal(await exists(`${dir}/concepts`), false, `${dir}/concepts would escape the gate`);
  const topLevel = (await readdir(new URL('../', import.meta.url), { withFileTypes: true }))
    .filter((e) => e.isDirectory() && !e.name.startsWith('.') && e.name !== 'node_modules')
    .map((e) => e.name);
  const withConcepts = [];
  for (const dir of topLevel) if (await exists(`${dir}/concepts`)) withConcepts.push(dir);
  assert.deepEqual(withConcepts.filter((dir) => !TRACKS.includes(dir)), [], 'a concepts/ directory outside a track escapes the gate');
});
