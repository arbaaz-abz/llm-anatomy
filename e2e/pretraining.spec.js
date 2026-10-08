import { test, expect } from '@playwright/test';
import { registerLessonContract } from './lesson-helpers.js';
import { collectConsoleErrors } from './helpers.js';
import { CAPTIONS, CHECK_WORK, DEFAULT_LOSSES, TRY_THIS } from '../tests/pretraining-expected.js';

const URL = '/training/#pretraining';

registerLessonContract({ name: 'pretraining', url: URL, captions: CAPTIONS, factRows: 12, returnHash: 'pretraining' });

const readout = (page, name) => page.locator(`[data-section="toy"] [data-readout="${name}"]`);
const chip = (page, position, word) => page.locator(`[data-section="toy"] [role="button"][aria-label="position ${position}: ${word}"]`);
const lossCells = (page) => page.locator('[data-section="toy"] [data-readout="loss-strip"] .g-cell .g-text');
const slider = (page) => page.locator('#p');
const sliderText = (page) => page.locator('[data-section="toy"] .slider output');

test.describe('pretraining toy: grade a sentence', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('opens on mat selected, the stand-ins, and the exact "Check my work" box', async ({ page }) => {
    await expect(lossCells(page)).toHaveText(DEFAULT_LOSSES);
    await expect(chip(page, 7, 'mat')).toHaveAttribute('aria-pressed', 'true');
    await expect(chip(page, 7, 'mat')).toHaveAttribute('tabindex', '0');
    await expect(chip(page, 5, 'on')).toHaveAttribute('tabindex', '-1');
    await expect(readout(page, 'selected-loss')).toHaveText('0.799');
    await expect(readout(page, 'mean-loss')).toHaveText('1.052');
    await expect(readout(page, 'perplexity')).toHaveText('2.86');
    await expect(readout(page, 'ref-16')).toHaveText('2.773');
    await expect(readout(page, 'ref-kimi')).toHaveText('12.007');
    await expect(sliderText(page)).toHaveText('0.45');
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK);
  });

  test('try this 1: certainty moves the mean down 0.114; a confident miss moves it up 0.544', async ({ page }) => {
    await slider(page).fill('1');
    await expect(readout(page, 'mean-loss')).toHaveText('0.938');
    await expect(readout(page, 'selected-loss')).toHaveText('0.000');
    await slider(page).fill('0.01');
    await expect(readout(page, 'selected-loss')).toHaveText('4.605');
    await expect(readout(page, 'mean-loss')).toHaveText('1.596');
    await expect(readout(page, 'perplexity')).toHaveText('4.93');
    await expect(lossCells(page).nth(5)).toHaveText('4.605');
  });

  test('try this 2: "on" is the top guess and still costs 0.941; its exact stand-in survives until a drag', async ({ page }) => {
    await chip(page, 5, 'on').click();
    await expect(chip(page, 5, 'on')).toHaveAttribute('aria-pressed', 'true');
    await expect(chip(page, 7, 'mat')).toHaveAttribute('aria-pressed', 'false');
    await expect(readout(page, 'selected-loss')).toHaveText('0.941');
    await expect(sliderText(page)).toHaveText('0.3903');
    await expect(readout(page, 'mean-loss')).toHaveText('1.052');
    await expect(readout(page, 'check-work')).toContainText('selected: on   −ln 0.3903 = 0.9410');
    await slider(page).fill('0.39');
    await expect(readout(page, 'selected-loss')).toHaveText('0.942');
  });

  test('try this 3: a uniform guess over 16 gives loss 2.773 everywhere and perplexity 16', async ({ page }) => {
    await page.locator('[data-section="toy"] button', { hasText: 'uniform guess over 16' }).click();
    await expect(lossCells(page)).toHaveText(Array(7).fill('2.773'));
    await expect(readout(page, 'mean-loss')).toHaveText('2.773');
    await expect(readout(page, 'perplexity')).toHaveText('16.00');
    await expect(sliderText(page)).toHaveText('0.0625');
    await page.locator('[data-section="toy"] button', { hasText: 'Reset' }).click();
    await expect(lossCells(page)).toHaveText(DEFAULT_LOSSES);
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK);
  });

  test('preset chips set the selected probability: confident miss, uniform 1/16, certain', async ({ page }) => {
    const preset = (name) => page.locator('[data-section="toy"] button', { hasText: name });
    await preset('confident miss 0.01').click();
    await expect(readout(page, 'selected-loss')).toHaveText('4.605');
    await preset('uniform 1/16').click();
    await expect(readout(page, 'selected-loss')).toHaveText('2.773');
    await expect(sliderText(page)).toHaveText('0.0625');
    await preset('certain 1.00').click();
    await expect(readout(page, 'selected-loss')).toHaveText('0.000');
    await expect(readout(page, 'mean-loss')).toHaveText('0.938');
  });

  test('selectable stage item (P3-R17): Tab reaches the selection, arrows move it, Enter and Space select, focus is visible', async ({ page }) => {
    const errors = collectConsoleErrors(page);
    await slider(page).focus();
    await page.keyboard.press('Shift+Tab'); // one tab stop for the strip: the selected chip
    await expect(chip(page, 7, 'mat')).toBeFocused();
    expect(await chip(page, 7, 'mat').evaluate((g) => getComputedStyle(g).outlineStyle)).not.toBe('none');
    await page.keyboard.press('ArrowRight');
    await expect(chip(page, 8, '.')).toBeFocused();
    await expect(chip(page, 8, '.')).toHaveAttribute('aria-pressed', 'true');
    await expect(readout(page, 'selected-loss')).toHaveText('0.223');
    await page.keyboard.press('ArrowRight'); // the last item: stays put
    await expect(chip(page, 8, '.')).toBeFocused();
    for (let i = 0; i < 4; i += 1) await page.keyboard.press('ArrowLeft');
    await expect(chip(page, 4, 'down')).toBeFocused();
    await expect(readout(page, 'selected-loss')).toHaveText('1.204');
    await expect(sliderText(page)).toHaveText('0.30');
    await page.keyboard.press('Home');
    await expect(chip(page, 2, 'cat')).toBeFocused();
    await expect(readout(page, 'selected-loss')).toHaveText('2.303');
    await page.keyboard.press('End');
    await expect(chip(page, 8, '.')).toHaveAttribute('aria-pressed', 'true');
    await chip(page, 5, 'on').focus();
    await page.keyboard.press('Enter');
    await expect(chip(page, 5, 'on')).toHaveAttribute('aria-pressed', 'true');
    await chip(page, 3, 'sat').focus();
    await page.keyboard.press('Space');
    await expect(chip(page, 3, 'sat')).toHaveAttribute('aria-pressed', 'true');
    await expect(readout(page, 'selected-loss')).toHaveText('1.386');
    await expect(chip(page, 3, 'sat')).toHaveAttribute('tabindex', '0');
    expect(errors).toEqual([]);
  });

  test('the try-this list sits at the end of the toy, its numbers computed (README lesson 34)', async ({ page }) => {
    const items = page.locator('[data-section="toy"] ol.try-this > li');
    await expect(items).toHaveCount(TRY_THIS.length);
    const strip = (s) => s.replace(/`/g, '');
    for (const [i, t] of TRY_THIS.entries()) {
      await expect(items.nth(i)).toHaveText(`${strip(t.prompt)} → Insight: ${t.insight}${t.rest}`);
    }
    const last = await page.locator('[data-section="toy"] .toy > *:last-child').evaluate((n) => n.matches('ol.try-this'));
    expect(last).toBe(true);
  });
});
