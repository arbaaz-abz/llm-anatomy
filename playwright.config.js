import { defineConfig, devices } from '@playwright/test';

// One port per agent (Plan 2 port table). Worktrees share 127.0.0.1, so a run on a busy port
// must fail at startup instead of quietly testing another branch's server: never reuse a server.
const DEFAULT_PORT = 4173;

export function resolvePort(raw) {
  if (raw === undefined || raw === '') return DEFAULT_PORT;
  const port = Number(raw);
  if (!Number.isInteger(port) || port < 1024 || port > 65535) {
    throw new Error(`playwright.config: PORT must be an integer 1024–65535, got "${raw}"`);
  }
  return port;
}

const PORT = resolvePort(process.env.PORT);
// 127.0.0.1 (not localhost): scripts/serve.js binds IPv4 only.
const BASE_URL = `http://127.0.0.1:${PORT}`;

export default defineConfig({
  testDir: 'e2e',
  outputDir: `test-results/port-${PORT}`,
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: [['list'], ['html', { open: 'never', outputFolder: `playwright-report/port-${PORT}` }]],
  use: { baseURL: BASE_URL, trace: 'retain-on-failure' },
  webServer: {
    command: 'node scripts/serve.js .',
    env: { PORT: String(PORT) },
    url: `${BASE_URL}/architecture/`,
    reuseExistingServer: false,
  },
  projects: [
    { name: 'chromium-desktop', use: { ...devices['Desktop Chrome'], viewport: { width: 1280, height: 800 } } },
    { name: 'chromium-phone', use: { ...devices['Desktop Chrome'], viewport: { width: 400, height: 860 }, hasTouch: true } },
    { name: 'webkit-desktop', use: { ...devices['Desktop Safari'], viewport: { width: 1280, height: 800 } } },
  ],
});
