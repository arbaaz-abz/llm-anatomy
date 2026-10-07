import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { toArtifactHtml, artifactFiles } from '../scripts/build-artifact.js';

const page = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>LLM Architecture Anatomy</title>
<link rel="stylesheet" href="../shared/theme.css">
<script type="importmap">{ "imports": { "@shared/": "../shared/", "@math/": "../math/" } }</script>
</head>
<body>
<div id="app" data-track="architecture"></div>
<script type="module">
import('@shared/track-app.js').then(({ mountTrack }) => mountTrack({ root: document.getElementById('app'), track: 'architecture' }));
</script>
</body>
</html>`;

test('strips the document wrapper but keeps title, styles and scripts on top', () => {
  const out = toArtifactHtml(page);
  assert.doesNotMatch(out, /<!doctype|<html|<\/html>|<head|<\/head>|<body|<\/body>/i);
  assert.doesNotMatch(out, /charset|name="viewport"/);
  assert.match(out, /^<meta name="llm-anatomy-target" content="artifact">/);
  assert.ok(out.indexOf('<title>') < out.indexOf('<div id="app"'));
});

test('rebases ../shared, ../math, ../data to ./', () => {
  const out = toArtifactHtml(page);
  assert.match(out, /href="\.\/shared\/theme\.css"/);
  assert.match(out, /"@shared\/": "\.\/shared\/"/);
  assert.match(out, /"@math\/": "\.\/math\/"/);
  assert.doesNotMatch(out, /\.\.\//);
});

test('refuses body attributes, which the artifact skeleton would drop', () => {
  assert.throws(() => toArtifactHtml(page.replace('<body>', '<body class="x">')), /body/);
});

test('refuses pages without head/body', () => {
  assert.throws(() => toArtifactHtml('<div>hi</div>'), /head/);
});

test('rebases only attribute values and the import map, never inline script strings', () => {
  const html = page.replace('</body>', `<script type="module">
const data = await loadJSON('../data/models.json');
const other = "../shared/data.js";
</script>
</body>`);
  const out = toArtifactHtml(html);
  assert.match(out, /loadJSON\('\.\.\/data\/models\.json'\)/);
  assert.match(out, /"\.\.\/shared\/data\.js"/);
  assert.match(out, /href="\.\/shared\/theme\.css"/);
  assert.match(out, /"@shared\/": "\.\/shared\/"/);
});

test('artifactFiles maps published paths to sources, filters types, wraps extensionless files', async (t) => {
  const repo = await mkdtemp(join(tmpdir(), 'artifact-test-'));
  t.after(() => rm(repo, { recursive: true, force: true }));
  const put = async (rel) => { await mkdir(join(repo, rel, '..'), { recursive: true }); await writeFile(join(repo, rel), 'x'); };
  for (const rel of [
    'shared/a.js', 'shared/theme.css', 'shared/vendor/katex/LICENSE', 'shared/vendor/katex/VERSION',
    'shared/vendor/katex/fonts/F.woff2', 'shared/.DS_Store', 'shared/NOTES.md',
    'math/core.js', 'data/models.json', 'architecture/concepts/x.js', 'architecture/concepts/.DS_Store',
    'architecture/app.js', 'architecture/index.html', 'architecture/NOTES.md', 'architecture/.DS_Store',
  ]) await put(rel);
  const files = await artifactFiles(join(repo, 'architecture'), repo);
  assert.equal(files['shared/a.js'], 'shared/a.js');
  assert.equal(files['shared/theme.css'], 'shared/theme.css');
  assert.equal(files['shared/vendor/katex/fonts/F.woff2'], 'shared/vendor/katex/fonts/F.woff2');
  assert.equal(files['math/core.js'], 'math/core.js');
  assert.equal(files['data/models.json'], 'data/models.json');
  assert.equal(files['concepts/x.js'], 'architecture/concepts/x.js');
  assert.deepEqual(files['shared/vendor/katex/LICENSE'], { from: 'shared/vendor/katex/LICENSE', contentType: 'text/plain' });
  assert.deepEqual(files['shared/vendor/katex/VERSION'], { from: 'shared/vendor/katex/VERSION', contentType: 'text/plain' });
  assert.equal('shared/.DS_Store' in files, false);
  assert.equal('shared/NOTES.md' in files, false);
  assert.equal('concepts/.DS_Store' in files, false);
  assert.equal(files['app.js'], 'architecture/app.js');
  assert.equal('index.html' in files, false);
  assert.equal('NOTES.md' in files, false);
  assert.equal('.DS_Store' in files, false);
});
