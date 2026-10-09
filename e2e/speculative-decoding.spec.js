import { test, expect } from '@playwright/test';
import { registerLessonContract } from './lesson-helpers.js';
import { collectConsoleErrors } from './helpers.js';
import { CAPTIONS, CHECK_WORK, USERS_NOTE, TRY_THIS } from '../tests/speculative-decoding-expected.js';

const URL = '/serving/#speculative-decoding';

registerLessonContract({ name: 'speculative-decoding', url: URL, captions: CAPTIONS, factRows: 7, returnHash: 'speculative-decoding' });

const readout = (page, name) => page.locator(`[data-section="toy"] [data-readout="${name}"]`);
// The batch slider snaps to 1, 4, 16, 64, 128 and the largest count that fits (211): index i is the i-th stop.
const BATCH_INDEX = Object.freeze({ 1: 0, 4: 1, 16: 2, 64: 3, 128: 4, 211: 5 });
const setBatch = (page, users) => page.locator('#batch').fill(String(BATCH_INDEX[users]));
const setK = (page, k) => page.locator('#k').fill(String(k));
const guess = (page, word) => page.locator(`#guess [data-value="${word}"]`);

async function expectReadouts(page, expected) {
  for (const [name, text] of Object.entries(expected)) await expect(readout(page, name), name).toHaveText(text);
}

test.describe('speculative-decoding toy: guess and check', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('opens on the storyboard default with its exact numbers and "Check my work"', async ({ page }) => {
    const errors = collectConsoleErrors(page);
    await expectReadouts(page, {
      'tokens-per-round': '2.53', 'speedup-simple': '2.2×', 'speedup-batch': '2.2×', 'plain-step': '14.7 ms', 'verify-pass': '14.7 ms',
      'verify-bound': 'memory-bound', 'keep-chance': '0.857', 'acceptance-rate': '0.900', leftover: '0 · 0.50 · 0.50 · 0', result: '0.60 · 0.25 · 0.10 · 0.05',
    });
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK);
    await expect(page.locator('#batch')).toHaveAttribute('max', '5');
    await expect(readout(page, 'users-note')).toHaveText(USERS_NOTE);
    expect(errors).toEqual([]);
  });

  test('the batch slider\'s last stop is the largest count that fits: 211 users', async ({ page }) => {
    await setBatch(page, 211);
    await expect(page.locator('[data-section="toy"] .slider', { has: page.locator('#batch') }).locator('output')).toHaveText('211 users');
  });

  test('the batch slider reads "1 user" at its first stop', async ({ page }) => {
    await setBatch(page, 1);
    await expect(page.locator('[data-section="toy"] .slider', { has: page.locator('#batch') }).locator('output')).toHaveText('1 user');
  });

  test('the try-this list closes the toy, verbatim', async ({ page }) => {
    await expect(page.locator('[data-section="toy"] ol.try-this > li')).toHaveText(TRY_THIS);
  });

  test('try this 1: guesses have diminishing returns', async ({ page }) => {
    await setK(page, 1);
    await expectReadouts(page, { 'tokens-per-round': '1.70', 'speedup-simple': '1.62×' });
    await setK(page, 5);
    await expectReadouts(page, { 'tokens-per-round': '2.94', 'speedup-simple': '2.35×' });
    await setK(page, 8);
    await expectReadouts(page, { 'tokens-per-round': '3.20', 'speedup-simple': '2.28×' });
    await setK(page, 3);
    await expectReadouts(page, { 'tokens-per-round': '2.53', 'speedup-simple': '2.2×' });
  });

  test('try this 2: the speedup fades as the batch fills the arithmetic, and can fall below 1', async ({ page }) => {
    await setBatch(page, 64);
    await expectReadouts(page, { 'speedup-batch': '2.14×', 'verify-bound': 'memory-bound' });
    await setBatch(page, 128);
    await expectReadouts(page, { 'speedup-batch': '1.53×', 'plain-step': '24 ms', 'verify-pass': '36.2 ms', 'verify-bound': 'compute-bound' });
    await setBatch(page, 211);
    await expectReadouts(page, { 'speedup-batch': '1.19×' });
    await setK(page, 5);
    await expectReadouts(page, { 'speedup-batch': '0.91 of the plain speed' });
  });

  test('try this 3: MTP 0.85 sets one guess that survives a large batch', async ({ page }) => {
    await page.locator('#mtp-preset [data-value="0.85"]').click();
    await expectReadouts(page, { 'tokens-per-round': '1.85', 'speedup-simple': '1.76×', 'speedup-batch': '1.76×' });
    await expect(page.locator('#k')).toHaveValue('1');
    await expect(page.locator('#alpha')).toHaveValue('0.85');
    await setBatch(page, 128);
    await expectReadouts(page, { 'speedup-batch': '1.73×' });
    // The preset chips are actions, never a selection (XT-3).
    await expect(page.locator('#mtp-preset [aria-pressed]')).toHaveCount(0);
  });

  test('drafter cost chips: a free drafter gives the token count; a dear one gives less', async ({ page }) => {
    await page.locator('#c [data-value="0"]').click();
    await expectReadouts(page, { 'speedup-simple': '2.53×' });
    await page.locator('#c [data-value="0.2"]').click();
    await expectReadouts(page, { 'speedup-simple': '1.58×' });
  });

  test('try this 4: only guesses the drafter overrates are rejected, and the result is the target\'s own', async ({ page }) => {
    await guess(page, 'on').click();
    await expectReadouts(page, { 'keep-chance': '1.000', leftover: '0 · 0.50 · 0.50 · 0', result: '0.60 · 0.25 · 0.10 · 0.05' });
    await guess(page, 'up').click();
    await expectReadouts(page, { 'keep-chance': '1.000', result: '0.60 · 0.25 · 0.10 · 0.05' });
    await guess(page, 'down').click();
    await expectReadouts(page, { 'keep-chance': '0.857', result: '0.60 · 0.25 · 0.10 · 0.05' });
  });

  test('"Check my work" follows the controls', async ({ page }) => {
    await setK(page, 5);
    await expect(readout(page, 'check-work')).toContainText('(1 − 0.7^6) / (1 − 0.7) = 2.94 tokens per round');
    await expect(readout(page, 'check-work')).toContainText('2.94 / (1 + 5 · 0.05) = 2.94 / 1.25 = 2.35×');
  });
});
