import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readdir, readFile } from 'node:fs/promises';

const graph = JSON.parse(await readFile(new URL('../shared/concepts.json', import.meta.url), 'utf8'));
const DIR = new URL('../docs/storyboards/', import.meta.url);

// "Track: … · Prereqs: a, b (note)" → ['a', 'b']; "none (…)" → [].
const headerPrereqs = (text) => {
  const line = text.match(/^Track:.*Prereqs:\s*(.*)$/m)?.[1] ?? '';
  return line.split('(')[0].split('·')[0].split(',').map((s) => s.replace(/`/g, '').trim()).filter((s) => s && s !== 'none');
};

test('every storyboard header lists the same prereqs as shared/concepts.json (README lesson 1)', async () => {
  const files = (await readdir(DIR)).filter((f) => f.endsWith('.md') && f !== 'README.md' && !f.startsWith('_'));
  const mismatches = [];
  for (const file of files) {
    const slug = file.replace(/\.md$/, '');
    const concept = graph.concepts.find((c) => c.slug === slug);
    const header = headerPrereqs(await readFile(new URL(file, DIR), 'utf8'));
    const same = concept && [...header].sort().join() === [...concept.prereqs].sort().join();
    if (!same) mismatches.push(`${slug}: header [${header.join(', ')}] vs graph [${concept?.prereqs.join(', ') ?? 'no concept'}]`);
  }
  assert.deepEqual(mismatches, []);
});
