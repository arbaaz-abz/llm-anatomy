import { test } from 'node:test';
import assert from 'node:assert/strict';
import { resolve, sep } from 'node:path';
import { resolveSafe } from '../scripts/serve.js';

const root = resolve('/srv/site');

test('resolves normal paths inside the root', () => {
  assert.equal(resolveSafe(root, '/architecture/index.html'), `${root}${sep}architecture${sep}index.html`);
  assert.equal(resolveSafe(root, '/shared/ui/stepper.js?v=2'), `${root}${sep}shared${sep}ui${sep}stepper.js`);
  assert.equal(resolveSafe(root, '/'), root);
});

test('blocks traversal and malformed encodings', () => {
  assert.equal(resolveSafe(root, '/../../etc/passwd'), null);
  assert.equal(resolveSafe(root, '/%2e%2e/%2e%2e/etc/passwd'), null);
  assert.equal(resolveSafe(root, '/%E0%A4%A'), null);
  assert.equal(resolveSafe(root, '/site-evil/../../site-evil/x'), null);
});

test('redirectFor sends bare directory paths to a trailing slash and keeps the query', async () => {
  const { redirectFor } = await import('../scripts/serve.js');
  assert.equal(redirectFor('/architecture', true), '/architecture/');
  assert.equal(redirectFor('/architecture?x=1', true), '/architecture/?x=1');
  assert.equal(redirectFor('/architecture/', true), null);
  assert.equal(redirectFor('/shared/concepts.json', false), null);
  assert.equal(redirectFor('/', true), null);
});

test('redirectFor never produces a protocol-relative Location', async () => {
  const { redirectFor } = await import('../scripts/serve.js');
  assert.equal(redirectFor('//architecture', true), '/architecture/');
  assert.equal(redirectFor('///evil.example', true), '/evil.example/');
});

test('mimeFor covers modern module/font/icon types and falls back to octet-stream', async () => {
  const { mimeFor } = await import('../scripts/serve.js');
  assert.equal(mimeFor('a.mjs'), 'text/javascript; charset=utf-8');
  assert.equal(mimeFor('a.woff'), 'font/woff');
  assert.equal(mimeFor('a.ttf'), 'font/ttf');
  assert.equal(mimeFor('favicon.ico'), 'image/x-icon');
  assert.equal(mimeFor('LICENSE'), 'application/octet-stream');
});

test('server: 301 for bare directory, 200 files, 404, 405, HEAD without body', async (t) => {
  const { createApp } = await import('../scripts/serve.js');
  const { mkdtemp, mkdir, writeFile, rm } = await import('node:fs/promises');
  const { tmpdir } = await import('node:os');
  const { join } = await import('node:path');
  const dir = await mkdtemp(join(tmpdir(), 'serve-test-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  await mkdir(join(dir, 'architecture'));
  await writeFile(join(dir, 'architecture', 'index.html'), '<p>hi</p>');
  await writeFile(join(dir, 'data.json'), '{}');
  const server = createApp(dir);
  await new Promise((done) => server.listen(0, '127.0.0.1', done));
  t.after(() => server.close());
  assert.equal(server.address().address, '127.0.0.1');
  const base = `http://127.0.0.1:${server.address().port}`;

  const redirect = await fetch(`${base}/architecture?x=1`, { redirect: 'manual' });
  assert.equal(redirect.status, 301);
  assert.equal(redirect.headers.get('location'), '/architecture/?x=1');

  const page = await fetch(`${base}/architecture/`);
  assert.equal(page.status, 200);
  assert.match(page.headers.get('content-type'), /text\/html/);
  assert.equal(await page.text(), '<p>hi</p>');

  assert.equal((await fetch(`${base}/data.json`)).headers.get('content-type'), 'application/json; charset=utf-8');
  assert.equal((await fetch(`${base}/nope.js`)).status, 404);

  const post = await fetch(`${base}/data.json`, { method: 'POST' });
  assert.equal(post.status, 405);
  assert.equal(post.headers.get('allow'), 'GET, HEAD');

  const head = await fetch(`${base}/data.json`, { method: 'HEAD' });
  assert.equal(head.status, 200);
  assert.equal(await head.text(), '');
});
