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

test('the story plays act by act and rests on its final frame; Replay starts it over', async ({ page }) => {
  await page.goto('/');
  const covers = page.locator('.hub-cover-svg');
  await expect(covers).toHaveCount(3);
  await expect(covers.first()).toHaveAttribute('role', 'img');
  await expect(covers.first()).toHaveAccessibleName(/a token goes in/i);
  // Early on, the architecture act is still running: the next token is a draft "?".
  await expect(page.locator('.hub-track[data-track="architecture"] .g-token--draft')).toHaveCount(1);
  await expect(page.locator('.hub-tracks')).toHaveAttribute('data-story', 'playing');
  // Once the whole story (about 8 s) is over, every cover shows its final frame.
  await expect(page.locator('.hub-tracks')).toHaveAttribute('data-story', 'done', { timeout: 15000 });
  await expect(page.locator('.hub-track[data-track="serving"] .g-token[data-req]')).toHaveCount(12);
  await expect(page.locator('.hub-track[data-track="architecture"] .g-token--active')).toHaveText(/down/);
  await expect(page.locator('.hub-track[data-track="training"] .g-plot .g-series')).toHaveCount(1);
  const bars = await page.locator('.hub-track').evaluateAll((cards) => cards.map((c) => c.style.getPropertyValue('--act-p')));
  expect(bars).toEqual(['1', '1', '1']);
  await page.locator('.hub-replay').click();
  await expect(page.locator('.hub-tracks')).toHaveAttribute('data-story', 'playing');
  await expect(page.locator('.hub-track[data-track="architecture"] .g-token--draft')).toHaveCount(1);
});

test('with reduced motion the covers show their final frames at once and there is nothing to replay', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/');
  await expect(page.locator('.hub-tracks')).toHaveAttribute('data-story', 'still');
  await expect(page.locator('.hub-track[data-track="architecture"] .g-token--active')).toHaveText(/down/);
  await expect(page.locator('.hub-track[data-track="serving"] .g-token[data-req]')).toHaveCount(12);
  await expect(page.locator('.hub-replay')).toBeHidden();
});

test('the course map has a station per lesson; focusing one names what it builds on and lights its chain', async ({ page }) => {
  await page.goto('/');
  const stations = page.locator('.map-station');
  await expect(stations).toHaveCount(34);
  await expect(page.locator('.map-edge')).toHaveCount(46);
  await expect(page.locator('.map-edge--cross')).toHaveCount(9);
  await expect(page.locator('.hub-map-caption')).toHaveText(/Hover or tab/);
  const prefix = page.locator('.map-station[data-slug="prefix-caching"]');
  await expect(prefix).toHaveAttribute('href', './serving/#prefix-caching');
  await prefix.focus();
  await expect(page.locator('.hub-map-caption')).toHaveText(/^Prefix caching \(Serving 5\) builds on \d+ lessons: /);
  await expect(page.locator('.map-station.is-path')).toHaveCount(await page.locator('.hub-section li.is-path').count());
  await expect(page.locator('.hub-section li.is-path[data-slug="kv-cache"]')).toHaveCount(1);
  await prefix.blur();
  await expect(page.locator('.map-station.is-path')).toHaveCount(0);
  await expect(page.locator('.hub-map-caption')).toHaveText(/Hover or tab/);
});

test('a map station opens its lesson', async ({ page }) => {
  await page.goto('/');
  await page.locator('.map-station[data-slug="rope"]').click();
  await expect(page).toHaveURL(/\/architecture\/#rope$/);
  await expect(page.locator('.lesson-head h2')).toHaveText('RoPE');
});

test('learned lessons show on the hero, the cards and the map', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('llm-anatomy:learned', JSON.stringify(['decoder-anatomy', 'attention'])));
  await page.goto('/');
  await expect(page.locator('.hub-progress')).toHaveText(/2 of 34 learned\. Next for you: From GPT-3 to 2026/);
  await expect(page.locator('.hub-track[data-track="architecture"] .eyebrow')).toHaveText('Part 1 · 11 lessons · 2 learned');
  await expect(page.locator('.hub-section li.is-learned')).toHaveCount(2);
  await expect(page.locator('.map-station.is-learned')).toHaveCount(2);
});
