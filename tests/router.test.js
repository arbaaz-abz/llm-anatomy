import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseRoute, startRouter } from '../shared/router.js';

const known = new Set(['rope', 'kv-cache']);

test('parses a known slug', () => {
  assert.deepEqual(parseRoute('#rope', known), { slug: 'rope' });
  assert.deepEqual(parseRoute('#kv-cache', known), { slug: 'kv-cache' });
});

test('empty hash is the track index', () => {
  assert.deepEqual(parseRoute('', known), { slug: null });
  assert.deepEqual(parseRoute('#', known), { slug: null });
  assert.deepEqual(parseRoute(undefined, known), { slug: null });
});

test('rejects malformed hashes without throwing', () => {
  for (const hash of ['#key=value', '#RoPE', '#../x', '#%E0%A4%A', '#unknown', '#rope?x=1']) {
    const route = parseRoute(hash, known);
    assert.equal(route.slug, null, hash);
    assert.ok('invalid' in route, hash);
  }
});

test('startRouter fires immediately and on hashchange, and stops cleanly', () => {
  const win = new EventTarget();
  win.location = { hash: '#rope' };
  const seen = [];
  const stop = startRouter({ knownSlugs: known, onRoute: (r) => seen.push(r.slug), win });
  win.location.hash = '#kv-cache';
  win.dispatchEvent(new Event('hashchange'));
  stop();
  win.location.hash = '#rope';
  win.dispatchEvent(new Event('hashchange'));
  assert.deepEqual(seen, ['rope', 'kv-cache']);
});
