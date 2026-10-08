import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, relative, resolve, sep, posix } from 'node:path';
import { fileURLToPath } from 'node:url';
import { artifactFiles } from '../scripts/build-artifact.js';

const ROOT = fileURLToPath(new URL('..', import.meta.url));
// Static imports, re-exports (`export … from`) and dynamic imports; specifiers never contain quotes or newlines.
const IMPORT = /\b(?:import|export)\s*(?:[\w*{}\s,$]*\s*from\s*)?["']([^"'\n]+)["']|\bimport\(\s*["']([^"'\n]+)["']\s*\)/g;

async function walk(dir) {
  const entries = await readdir(dir, { withFileTypes: true }).catch((error) => (error.code === 'ENOENT' ? [] : Promise.reject(error)));
  const nested = await Promise.all(entries.map((e) => (e.isDirectory() ? walk(join(dir, e.name)) : [join(dir, e.name)])));
  return nested.flat().filter((f) => f.endsWith('.js'));
}

// Where a specifier lands in the published artifact (paths relative to the artifact root), or why it can't.
function publishedPath(spec, file, { track, pageDir, concepts }) {
  if (spec.startsWith('@shared/')) return `shared/${spec.slice(8)}`;
  if (spec.startsWith('@math/')) return `math/${spec.slice(6)}`;
  if (!spec.startsWith('./') && !spec.startsWith('../')) return { error: `bare or absolute specifier "${spec}"` };
  const target = resolve(file, '..', spec);
  if (!target.startsWith(concepts + sep)) return { error: `"${spec}" leaves ${track}/concepts/ (use @shared/ or @math/)` };
  return relative(pageDir, target).split(sep).join(posix.sep);
}

// Every problem with the imports of <root>/<track>/concepts/**, as "<file>: <why>" lines.
// <track>/index.html is published as its own artifact, so a concept may reach only @shared/, @math/ and its own concepts/.
async function importProblems(root, track) {
  const pageDir = join(root, track);
  const concepts = join(pageDir, 'concepts');
  const files = await artifactFiles(pageDir, root);
  const problems = [];
  for (const file of await walk(concepts)) {
    const name = relative(root, file).split(sep).join(posix.sep);
    for (const m of (await readFile(file, 'utf8')).matchAll(IMPORT)) {
      const spec = m[1] ?? m[2];
      const where = publishedPath(spec, file, { track, pageDir, concepts });
      if (typeof where !== 'string') problems.push(`${name}: ${where.error}`);
      else if (!(where in files)) problems.push(`${name}: "${spec}" → ${where} is not published`);
    }
  }
  return problems;
}

// Each track is published as its own artifact; a track with no concepts/ directory yet passes trivially.
const TRACK_IDS = ['architecture', 'training', 'serving'];

for (const track of TRACK_IDS) {
  test(`every import and re-export in ${track}/concepts/** resolves to a file the ${track} artifact publishes`, async () => {
    assert.deepEqual(await importProblems(ROOT, track), []);
  });
}

test('a planted bad concept is caught: a path out of concepts/, a missing re-export, a missing export-star', async (t) => {
  const repo = await mkdtemp(join(tmpdir(), 'artifact-imports-'));
  t.after(() => rm(repo, { recursive: true, force: true }));
  const put = async (rel, text = 'x') => { await mkdir(join(repo, rel, '..'), { recursive: true }); await writeFile(join(repo, rel), text); };
  for (const rel of ['shared/glyphs.js', 'math/core.js', 'data/models.json', 'data/hardware.json', 'architecture/index.html']) await put(rel);
  await put('architecture/concepts/bad.js', [
    "import * as G from '@shared/glyphs.js';",
    "import { x } from '../../shared/x.js';",
    "export { y } from './other.js';",
    "export * from './bad/also-missing.js';",
    "export const label = 'not an import';",
  ].join('\n'));
  assert.deepEqual(await importProblems(repo, 'architecture'), [
    'architecture/concepts/bad.js: "../../shared/x.js" leaves architecture/concepts/ (use @shared/ or @math/)',
    'architecture/concepts/bad.js: "./other.js" → concepts/other.js is not published',
    'architecture/concepts/bad.js: "./bad/also-missing.js" → concepts/bad/also-missing.js is not published',
  ]);
});

test('a training concept that imports another track\'s concept code is caught', async (t) => {
  const repo = await mkdtemp(join(tmpdir(), 'artifact-imports-'));
  t.after(() => rm(repo, { recursive: true, force: true }));
  const put = async (rel, text = 'x') => { await mkdir(join(repo, rel, '..'), { recursive: true }); await writeFile(join(repo, rel), text); };
  for (const rel of ['shared/glyphs.js', 'training/index.html', 'architecture/concepts/moe/stage.js']) await put(rel);
  await put('training/concepts/bad.js', "import { x } from '../../architecture/concepts/moe/stage.js';\n");
  assert.deepEqual(await importProblems(repo, 'training'), [
    'training/concepts/bad.js: "../../architecture/concepts/moe/stage.js" leaves training/concepts/ (use @shared/ or @math/)',
  ]);
});

for (const track of TRACK_IDS) {
  test(`the ${track} artifact ships every data file ctx.data holds, and no ${track} concept fetches JSON itself (facts come from ctx.data)`, async () => {
    const pageDir = join(ROOT, track);
    const files = await artifactFiles(pageDir, ROOT);
    for (const name of ['models', 'hardware', 'serving', 'papers']) assert.ok(`data/${name}.json` in files, `data/${name}.json is published`);
    const fetchers = [];
    for (const file of await walk(join(pageDir, 'concepts'))) {
      if (/\bfetch\s*\(|\bloadJSON\s*\(/.test(await readFile(file, 'utf8'))) fetchers.push(relative(ROOT, file));
    }
    assert.deepEqual(fetchers, []);
  });
}
