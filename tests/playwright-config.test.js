import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('..', import.meta.url));

// Reads the config in a child process, so each case gets its own PORT at module load.
function readConfig(port) {
  const script = `
    const { default: c } = await import('./playwright.config.js');
    const html = c.reporter.find(([name]) => name === 'html');
    process.stdout.write(JSON.stringify({ baseURL: c.use.baseURL, outputDir: c.outputDir, report: html[1].outputFolder,
      env: c.webServer.env, url: c.webServer.url, reuse: c.webServer.reuseExistingServer }));`;
  const env = { ...process.env, PORT: port };
  return JSON.parse(execFileSync(process.execPath, ['--input-type=module', '-e', script], { cwd: ROOT, env, encoding: 'utf8' }));
}

test('PORT drives the base URL, the server, the output dir and the html report', () => {
  assert.deepEqual(readConfig('4211'), {
    baseURL: 'http://127.0.0.1:4211',
    outputDir: 'test-results/port-4211',
    report: 'playwright-report/port-4211',
    env: { PORT: '4211' },
    url: 'http://127.0.0.1:4211/architecture/',
    reuse: false,
  });
});

test('no PORT (or an empty one) falls back to 4173', () => {
  assert.equal(readConfig('').baseURL, 'http://127.0.0.1:4173');
});

test('a server already on the port is never reused: a second agent on one port fails instead of testing another branch', () => {
  assert.equal(readConfig('4173').reuse, false);
});

test('a malformed PORT stops the run with a clear message', () => {
  assert.throws(() => readConfig('abc'), /PORT must be an integer 1024–65535/);
  assert.throws(() => readConfig('80'), /PORT must be an integer 1024–65535/);
});
