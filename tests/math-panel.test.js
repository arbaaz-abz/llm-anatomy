import { test, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { renderTex, allowHtmlClassOnly } from '../shared/ui/math-panel.js';

const fakeTarget = () => {
  const classes = new Set();
  return { textContent: '', classList: { add: (c) => classes.add(c), has: (c) => classes.has(c) } };
};

afterEach(() => { delete globalThis.katex; });

test('uses KaTeX when present', () => {
  const calls = [];
  globalThis.katex = { render: (tex, el, opts) => calls.push([tex, opts.displayMode, opts.trust]) };
  assert.equal(renderTex('a^2', fakeTarget()), 'katex');
  assert.deepEqual(calls, [['a^2', true, allowHtmlClassOnly]]);
});

test('falls back to raw TeX when KaTeX is missing', () => {
  const el = fakeTarget();
  assert.equal(renderTex('\\frac{1}{2}', el), 'fallback');
  assert.equal(el.textContent, '\\frac{1}{2}');
  assert.ok(el.classList.has('tex-fallback'));
});

test('falls back to raw TeX when KaTeX throws', () => {
  globalThis.katex = { render: () => { throw new Error('parse error'); } };
  const el = fakeTarget();
  const originalError = console.error;
  console.error = () => {};
  try {
    assert.equal(renderTex('\\bad{', el), 'fallback');
  } finally {
    console.error = originalError;
  }
  assert.equal(el.textContent, '\\bad{');
});

test('trust allowlist permits only \\htmlClass', () => {
  assert.equal(allowHtmlClassOnly({ command: '\\htmlClass' }), true);
  for (const command of ['\\href', '\\url', '\\htmlId']) {
    assert.equal(allowHtmlClassOnly({ command }), false, command);
  }
});
