import { test } from 'node:test';
import assert from 'node:assert/strict';
import { loadConcept, mountSafely, safeUnmount } from '../shared/track-app.js';

test('a built concept loads', async () => {
  const module = { default: { mount: () => () => {} } };
  assert.deepEqual(await loadConcept('rope', async () => module), { status: 'ok', module: module.default });
});

test('missing module: a not-yet-built page reports "missing" rather than crashing', async (t) => {
  t.mock.method(console, 'warn', () => {});
  const notFound = async () => { throw new TypeError('Failed to fetch dynamically imported module: …/concepts/rope.js'); };
  assert.deepEqual(await loadConcept('rope', notFound), { status: 'missing' });
  const nodeStyle = async () => { const e = new Error('Cannot find module'); e.code = 'ERR_MODULE_NOT_FOUND'; throw e; };
  assert.deepEqual(await loadConcept('rope', nodeStyle), { status: 'missing' });
  const firefox = async () => { throw new TypeError('error loading dynamically imported module: …/concepts/rope.js'); };
  assert.deepEqual(await loadConcept('rope', firefox), { status: 'missing' });
});

test('a module with a bug reports "error"', async () => {
  const broken = async () => { throw new SyntaxError('Unexpected token'); };
  const result = await loadConcept('rope', broken);
  assert.equal(result.status, 'error');
  assert.match(result.error.message, /Unexpected/);
});

test('a module without mount() is an error', async () => {
  const result = await loadConcept('rope', async () => ({ default: {} }));
  assert.equal(result.status, 'error');
  assert.match(result.error.message, /mount/);
});

test('mountSafely returns the unmount function on success', () => {
  const unmount = () => {};
  assert.deepEqual(mountSafely({ mount: () => unmount }, {}, {}), { status: 'ok', unmount });
  assert.deepEqual(mountSafely({ mount: () => undefined }, {}, {}), { status: 'ok', unmount: null });
});

test('mountSafely reports a throwing mount() as an error instead of throwing', (t) => {
  t.mock.method(console, 'error', () => {});
  const result = mountSafely({ mount: () => { throw new RangeError('bad ctx'); } }, {}, {});
  assert.equal(result.status, 'error');
  assert.equal(result.unmount, null);
  assert.match(result.error.message, /bad ctx/);
});

test('safeUnmount swallows a throwing unmount and tolerates null', (t) => {
  const log = t.mock.method(console, 'error', () => {});
  assert.doesNotThrow(() => safeUnmount(null));
  assert.doesNotThrow(() => safeUnmount(() => { throw new Error('boom'); }));
  assert.equal(log.mock.callCount(), 1);
  let called = false;
  safeUnmount(() => { called = true; });
  assert.equal(called, true);
});
