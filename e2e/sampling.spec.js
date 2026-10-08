import { test, expect } from '@playwright/test';
import { registerLessonContract } from './lesson-helpers.js';
import { CAPTIONS, CHECK_WORK, CHECK_WORK_HALF, CHECK_WORK_TOP_P, CHECK_WORK_GREEDY } from '../tests/sampling-expected.js';

const URL = '/architecture/#sampling';

registerLessonContract({ name: 'sampling', url: URL, captions: CAPTIONS, factRows: 6, returnHash: 'sampling' });

const readout = (page, name) => page.locator(`[data-readout="${name}"]`);
const subOf = (page, name) => page.locator(`td:has([data-readout="${name}"]) .ro-sub`);
const counts = (page) => page.locator('[data-section="toy"] .g-bars .g-bar-value');
// The sliders snap to lists, so fill() takes the list position: temperature [0.25, 0.5, 0.75, 1, 1.5, 2], top-k [off, 1, 2, 3, 5, 8],
// top-p [off, 0.95, 0.9, 0.75, 0.7, 0.5].
const TEMPERATURE = { 0.25: '0', 0.5: '1', 0.75: '2', 1: '3', 1.5: '4', 2: '5' };
const TOP_K = { off: '0', 1: '1', 2: '2', 3: '3', 5: '4', 8: '5' };
const TOP_P = { off: '0', 0.95: '1', 0.9: '2', 0.75: '3', 0.7: '4', 0.5: '5' };
const setTemperature = (page, t) => page.locator('#temperature').fill(TEMPERATURE[t]);
const setTopK = (page, k) => page.locator('#topK').fill(TOP_K[k]);
const setTopP = (page, p) => page.locator('#topP').fill(TOP_P[p]);

test.describe('sampling toy: shape the draw', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('opens on temperature 1, no filters, seed 1: the animation\'s probabilities and its 20 draws', async ({ page }) => {
    await expect(readout(page, 'p-on')).toHaveText('0.390');
    await expect(readout(page, 'p-dot')).toHaveText('0.237');
    await expect(readout(page, 'p-and')).toHaveText('0.087');
    await expect(readout(page, 'p-the')).toHaveText('0.053');
    await expect(readout(page, 'p-others')).toHaveText('0.019');
    await expect(subOf(page, 'p-others')).toHaveText('together 0.233 · 12 of 12 kept');
    await expect(readout(page, 'kept')).toHaveText('16 of 16');
    await expect(readout(page, 'mass')).toHaveText('1.000');
    await expect(counts(page)).toHaveText(['6', '8', '1', '1', '4']);
    await expect(readout(page, 'first-eight')).toHaveText('and on . then was on . the');
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK);
  });

  test('the order of operations is printed beside the controls', async ({ page }) => {
    await expect(page.locator('[data-section="toy"]')).toContainText('temperature → top-k → top-p → rescale → draw');
  });

  test('try this 1: temperature 0.5 sharpens, 2 flattens, greedy always draws "on"', async ({ page }) => {
    await setTemperature(page, 0.5);
    await expect(readout(page, 'p-on')).toHaveText('0.682');
    await expect(subOf(page, 'p-others')).toHaveText('together 0.020 · 12 of 12 kept');
    await expect(counts(page)).toHaveText(['15', '2', '0', '1', '2']);
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK_HALF);
    await setTemperature(page, 2);
    await expect(readout(page, 'p-on')).toHaveText('0.189');
    await expect(subOf(page, 'p-others')).toHaveText('together 0.506 · 12 of 12 kept');
    await expect(counts(page)).toHaveText(['4', '2', '2', '4', '8']);
    await page.locator('#greedy').click();
    await expect(page.locator('#greedy')).toHaveAttribute('aria-pressed', 'true');
    await expect(page.locator('#temperature')).toBeHidden();
    await expect(readout(page, 'p-on')).toHaveText('1.000');
    await expect(counts(page)).toHaveText(['20', '0', '0', '0', '0']);
    await expect(readout(page, 'first-eight')).toHaveText('on on on on on on on on');
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK_GREEDY);
    await page.locator('#greedy').click();
    await expect(page.locator('#temperature')).toBeVisible();
    await expect(readout(page, 'p-on')).toHaveText('0.189');
  });

  test('try this 2: top-p adapts to how sure the model is; top-k does not', async ({ page }) => {
    await setTopP(page, 0.7);
    await expect(readout(page, 'kept')).toHaveText('3 of 16');
    await expect(readout(page, 'mass')).toHaveText('0.714');
    await expect(readout(page, 'p-on')).toHaveText('0.547');
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK_TOP_P);
    await setTemperature(page, 0.5);
    await expect(readout(page, 'kept')).toHaveText('2 of 16');
    await expect(readout(page, 'mass')).toHaveText('0.933');
    await setTemperature(page, 2);
    await expect(readout(page, 'kept')).toHaveText('9 of 16');
    await expect(readout(page, 'mass')).toHaveText('0.705');
    await expect(subOf(page, 'p-others')).toContainText('5 of 12 kept');
    await setTopP(page, 'off');
    await setTopK(page, 3);
    for (const t of [0.5, 1, 2]) {
      await setTemperature(page, t);
      await expect(readout(page, 'kept')).toHaveText('3 of 16');
    }
  });

  test('top-k 3 cuts "the" and the 12 others: the table reads 0.000 and "cut"', async ({ page }) => {
    await setTopK(page, 3);
    await expect(readout(page, 'p-the')).toHaveText('0.000');
    await expect(subOf(page, 'p-the')).toHaveText('cut');
    await expect(readout(page, 'p-others')).toHaveText('0.000');
    await expect(subOf(page, 'p-others')).toHaveText('together 0.000 · 0 of 12 kept');
  });

  test('try this 3: the seed moves the draws; seed 1 brings the same text back', async ({ page }) => {
    await page.locator('#seed').fill('2');
    await expect(readout(page, 'first-eight')).toHaveText('the on on . a and . on');
    await page.locator('#seed').fill('3');
    await expect(readout(page, 'first-eight')).toHaveText('the on . on the . . on');
    await expect(counts(page)).toHaveText(['8', '5', '3', '2', '2']);
    await page.locator('#seed').fill('1');
    await expect(readout(page, 'first-eight')).toHaveText('and on . then was on . the');
    await page.locator('#new-seed').click();
    await expect(page.locator('#seed')).toHaveValue('2');
    await expect(readout(page, 'first-eight')).toHaveText('the on on . a and . on');
  });
});
