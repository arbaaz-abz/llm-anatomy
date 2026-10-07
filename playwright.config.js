import { defineConfig, devices } from '@playwright/test';

// 127.0.0.1 (not localhost): scripts/serve.js binds IPv4 only.
const BASE_URL = 'http://127.0.0.1:4173';

export default defineConfig({
  testDir: 'e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: [['list'], ['html', { open: 'never' }]],
  use: { baseURL: BASE_URL, trace: 'retain-on-failure' },
  webServer: {
    command: 'node scripts/serve.js .',
    env: { PORT: '4173' },
    url: `${BASE_URL}/architecture/`,
    reuseExistingServer: !process.env.CI,
  },
  projects: [
    { name: 'chromium-desktop', use: { ...devices['Desktop Chrome'], viewport: { width: 1280, height: 800 } } },
    { name: 'chromium-phone', use: { ...devices['Desktop Chrome'], viewport: { width: 400, height: 860 }, hasTouch: true } },
    { name: 'webkit-desktop', use: { ...devices['Desktop Safari'], viewport: { width: 1280, height: 800 } } },
  ],
});
