import { test, expect } from '@playwright/test';
import { collectConsoleErrors, TRACKS } from './helpers.js';

for (const { id, count } of TRACKS) {
  test(`${id} index lists ${count} lessons with no console errors`, async ({ page }) => {
    const errors = collectConsoleErrors(page);
    await page.goto(`/${id}/`);
    await expect(page.locator('.concept-card')).toHaveCount(count);
    await expect(page.locator('.nav-link')).toHaveCount(count);
    expect(errors).toEqual([]);
  });
}

test('an unbuilt lesson shows the coming-soon card with prereq links', async ({ page }) => {
  // Training is built in a later plan, so this lesson stays unbuilt for all of Plan 2.
  await page.goto('/training/#pretraining');
  await expect(page.locator('.coming-soon h2')).toHaveText('Pretraining');
  await expect(page.locator('.coming-soon a')).toHaveText(['The 2026 training pipeline']);
});

for (const hash of ['#key=value', '#RoPE', '#..%2Fx', '#%E0%A4%A', '#']) {
  test(`odd hash ${hash} falls back to the index`, async ({ page }) => {
    const errors = collectConsoleErrors(page);
    await page.goto(`/architecture/${hash}`);
    await expect(page.locator('.concept-card')).toHaveCount(11);
    expect(errors).toEqual([]);
  });
}

test('nav marks the current lesson and back/forward routes work', async ({ page }) => {
  await page.goto('/serving/');
  await page.getByRole('button', { name: 'Lessons' }).click();
  await page.locator('.nav-link', { hasText: 'PagedAttention' }).click();
  await expect(page).toHaveURL(/#paged-attention$/);
  await expect(page.locator('.nav-link[aria-current="page"]')).toHaveText('PagedAttention');
  await page.goBack();
  await expect(page.locator('.concept-card')).toHaveCount(9);
});

test('blocked localStorage does not break the page', async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(window, 'localStorage', { get() { throw new DOMException('blocked', 'SecurityError'); } });
  });
  const errors = collectConsoleErrors(page);
  await page.goto('/training/');
  await expect(page.locator('.concept-card')).toHaveCount(14);
  const isBlocked = await page.evaluate(() => { try { window.localStorage; return false; } catch { return true; } });
  expect(isBlocked).toBe(true);
  expect(errors).toEqual([]);
});
