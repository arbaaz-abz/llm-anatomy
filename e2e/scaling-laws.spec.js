import { test, expect } from '@playwright/test';
import { registerLessonContract } from './lesson-helpers.js';
import { CAPTIONS, CHECK_WORK, TRY_THIS } from '../tests/scaling-laws-expected.js';

const URL = '/training/#scaling-laws';
const FACT_ROWS = 8; // storyboard §8

registerLessonContract({ name: 'scaling-laws', url: URL, captions: CAPTIONS, factRows: FACT_ROWS, returnHash: 'scaling-laws' });

const readout = (page, name) => page.locator(`[data-readout="${name}"]`);
const BUDGET = (exponent) => `#C [data-value="1e+${exponent}"]`;
const SERVED = (value) => `#Dinf [data-value="${value}"]`;
// The slider is index-based: 0 is 1B; at the 10^24 default its stops are 61 grid steps plus the optimum at index 40, so 61 is 1T.
const SLIDER_1B = '0';
const SLIDER_1T_AT_1E24 = '61';

test.describe('scaling-laws toy: spend a compute budget', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('opens on the storyboard default: 10^24 FLOPs at the compute-optimal size, with its exact numbers', async ({ page }) => {
    await expect(readout(page, 'n-this')).toHaveText('95.9B');
    await expect(readout(page, 'd-this')).toHaveText('1.74T');
    await expect(readout(page, 'ratio-this')).toHaveText('18.1');
    await expect(readout(page, 'loss-this')).toHaveText('1.960');
    await expect(readout(page, 'n-opt')).toHaveText('95.9B');
    await expect(readout(page, 'd-opt')).toHaveText('1.74T');
    await expect(readout(page, 'ratio-opt')).toHaveText('18.1');
    await expect(readout(page, 'loss-opt')).toHaveText('1.960');
    await expect(readout(page, 'life-opt')).toHaveText('1.00 × 10²⁴');
    await expect(readout(page, 'ratio-cheap')).toHaveText('18.1');
    await expect(readout(page, 'saving')).toHaveText('0.0% less lifetime compute than the compute-optimal model');
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK);
  });

  test('the toy ends with the storyboard\'s try-this list: three prompts, each with a named insight', async ({ page }) => {
    const items = page.locator('[data-section="toy"] ol.try-this > li');
    await expect(page.locator('[data-section="toy"] h4', { hasText: 'Try this' })).toHaveCount(1);
    await expect(items).toHaveCount(TRY_THIS.length);
    for (const [i, [prompt, insight]] of TRY_THIS.entries()) {
      await expect(items.nth(i)).toContainText(prompt.replaceAll('**', ''));
      await expect(items.nth(i).locator('strong').last()).toHaveText(`Insight: ${insight}`);
    }
  });

  test('try this 1: there is a valley: 1T is worse than the optimum and 1B is worse still', async ({ page }) => {
    await page.locator('#logN').fill(SLIDER_1T_AT_1E24);
    await expect(readout(page, 'loss-this')).toHaveText('2.013');
    await expect(readout(page, 'd-this')).toHaveText('167B');
    await expect(readout(page, 'ratio-this')).toHaveText('0.17');
    await expect(page.locator('[data-readout="ratio-this"] ~ .ro-sub')).toHaveText('extrapolated');
    await page.locator('#logN').fill(SLIDER_1B);
    await expect(readout(page, 'loss-this')).toHaveText('2.187');
    await expect(readout(page, 'd-this')).toHaveText('167T');
    await expect(readout(page, 'ratio-this')).toHaveText('166,666.7');
    await expect(page.locator('[data-readout="ratio-this"] ~ .ro-sub')).toHaveText('extrapolated');
    await expect(readout(page, 'n-opt')).toHaveText('95.9B'); // the optimum does not move with the slider
  });

  test('try this 2: the more a model will be used, the smaller and longer-trained the cheapest one is', async ({ page }) => {
    const expected = [
      ['1000000000000', '79B', '27.1', '1.6%'], ['10000000000000', '48.4B', '88.8', '24.0%'],
      ['100000000000000', '29.2B', '494.9', '58.5%'], ['1000000000000000', '21B', '3,005.5', '74.1%'],
    ];
    for (const [served, size, ratio, saving] of expected) {
      await page.locator(SERVED(served)).click();
      await expect(readout(page, 'n-cheap')).toHaveText(size);
      await expect(readout(page, 'ratio-cheap')).toHaveText(ratio);
      await expect(readout(page, 'saving')).toHaveText(`${saving} less lifetime compute than the compute-optimal model`);
    }
    await page.locator(SERVED('100000000000000')).click();
    await expect(readout(page, 'd-cheap')).toHaveText('14.4T');
    await expect(readout(page, 'life-cheap')).toHaveText('8.36 × 10²⁴');
    await expect(readout(page, 'life-opt')).toHaveText('2.02 × 10²⁵');
    await expect(readout(page, 'n-this')).toHaveText('95.9B'); // the slider stays at the optimum
  });

  test('try this 3: ten thousand times the budget buys about a hundred times more of each', async ({ page }) => {
    await page.locator(BUDGET(22)).click();
    await expect(readout(page, 'n-opt')).toHaveText('9.05B');
    await expect(readout(page, 'd-opt')).toHaveText('184B');
    await expect(readout(page, 'ratio-opt')).toHaveText('20.4');
    await expect(readout(page, 'n-this')).toHaveText('9.05B'); // the slider follows the new optimum
    await page.locator(BUDGET(26)).click();
    await expect(readout(page, 'n-opt')).toHaveText('1.02T');
    await expect(readout(page, 'd-opt')).toHaveText('16.4T');
    await expect(readout(page, 'ratio-opt')).toHaveText('16.1');
    await expect(readout(page, 'n-this')).toHaveText('1.02T');
  });

  test('Reset returns to 10^24 FLOPs, the optimum and nothing served', async ({ page }) => {
    await page.locator(BUDGET(25)).click();
    await page.locator(SERVED('100000000000000')).click();
    await page.locator('#logN').fill('3');
    await page.locator('#reset').click();
    await expect(readout(page, 'n-this')).toHaveText('95.9B');
    await expect(readout(page, 'saving')).toHaveText('0.0% less lifetime compute than the compute-optimal model');
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK);
    await expect(page.locator('#C [aria-pressed="true"]')).toHaveText('10²⁴ FLOPs');
  });

  test('the check box follows the slider: 1T at 10^24 FLOPs', async ({ page }) => {
    await page.locator('#logN').fill(SLIDER_1T_AT_1E24);
    await expect(readout(page, 'check-work')).toContainText('N = 1.00 × 10¹² parameters (active)');
    await expect(readout(page, 'check-work')).toContainText('= 2.013');
  });
});
