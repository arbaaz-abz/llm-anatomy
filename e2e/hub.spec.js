import { test, expect } from '@playwright/test';
import { collectConsoleErrors } from './helpers.js';

test('home page lists the three tracks and all 34 lessons, with no console errors', async ({ page }) => {
  const errors = collectConsoleErrors(page);
  await page.goto('/');
  await expect(page.locator('.hub-track')).toHaveCount(3);
  await expect(page.locator('.hub-track h2')).toHaveText(['Architecture', 'Training', 'Serving']);
  await expect(page.locator('.hub-section a')).toHaveCount(34);
  await expect(page.locator('.hub-track[data-track="training"] ol').nth(1)).toHaveAttribute('start', '10');
  expect(errors).toEqual([]);
});

test('home page has no horizontal page scroll', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('.hub-track')).toHaveCount(3);
  await page.evaluate(() => document.fonts.ready);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(0);
});

test('a track title opens the track, and the track header links back home', async ({ page }) => {
  await page.goto('/');
  await page.locator('.hub-track h2 a', { hasText: 'Serving' }).click();
  await expect(page).toHaveURL(/\/serving\/$/);
  await page.locator('.app-header a', { hasText: 'LLM Anatomy' }).click();
  await expect(page).toHaveURL(/127\.0\.0\.1:\d+\/$/);
  await expect(page.locator('.hub-track')).toHaveCount(3);
});

test('a lesson link opens that lesson inside its track', async ({ page }) => {
  await page.goto('/');
  await page.locator('.hub-section a', { hasText: 'Prefix caching' }).click();
  await expect(page).toHaveURL(/\/serving\/#prefix-caching$/);
  await expect(page.locator('.lesson-head h2')).toHaveText('Prefix caching');
});

test('"Start with Architecture" opens the first track', async ({ page }) => {
  await page.goto('/');
  await page.locator('.hub-start').click();
  await expect(page).toHaveURL(/\/architecture\/$/);
});
