import { test, expect } from '@playwright/test';
import { registerLessonContract } from './lesson-helpers.js';
import { CAPTIONS, CHECK_WORK } from '../tests/rope-expected.js';

const URL = '/architecture/#rope';

registerLessonContract({ name: 'rope', url: URL, captions: CAPTIONS, factRows: 6, returnHash: 'rope' });

const readout = (page, name) => page.locator(`[data-readout="${name}"]`);
const cells = (page, name) => page.locator(`[data-readout="${name}"] .cell`);

test.describe('rope toy: turn the hands', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('opens on the animation\'s state: sat at 3, cat at 2, base 100, trained 16, no stretch', async ({ page }) => {
    await expect(readout(page, 'score')).toHaveText('1.596');
    await expect(readout(page, 'pair-dot-1')).toHaveText('1.621');
    await expect(readout(page, 'pair-dot-2')).toHaveText('−0.025');
    await expect(readout(page, 'score-plain')).toHaveText('3.000');
    await expect(readout(page, 'score-shifted')).toHaveText('1.596');
    await expect(cells(page, 'q-rotated')).toHaveText(['−0.282', '−1.980', '0.478', '0.148']);
    await expect(cells(page, 'k-rotated')).toHaveText(['−1.364', '−0.624', '0.099', '−0.490']);
    await expect(cells(page, 'offset-row')).toHaveText(['3.000', '1.596', '−1.298', '−3.044', '−2.058', '0.731', '2.739', '2.101']);
    await expect(readout(page, 'wavelength-1')).toHaveText('6.3 tokens');
    await expect(readout(page, 'wavelength-2')).toHaveText('62.8 tokens');
    await expect(readout(page, 'verdict-1')).toHaveText('all seen');
    await expect(readout(page, 'verdict-2')).toHaveText('all seen');
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK);
  });

  test('try this 1: the same offset gives the same score wherever the words sit', async ({ page }) => {
    await page.locator('#shift10').click();
    await expect(readout(page, 'score')).toHaveText('1.596');
    await expect(page.locator('#qPos')).toHaveValue('13');
    await expect(page.locator('#kPos')).toHaveValue('12');
    await page.locator('#qPos').fill('3');
    await page.locator('#kPos').fill('1');
    await expect(readout(page, 'score')).toHaveText('−1.298');
    await expect(readout(page, 'score-shifted')).toHaveText('−1.298');
  });

  test('the key never sits after the query: the key position is clamped to the query\'s', async ({ page }) => {
    await page.locator('#kPos').fill('10');
    await expect(page.locator('#kPos')).toHaveValue('3');
    await page.locator('#qPos').fill('1');
    await expect(page.locator('#kPos')).toHaveValue('1');
    await expect(readout(page, 'score')).toHaveText('3.000');
  });

  test('another key: "The" two tokens before "sat" scores −1.353', async ({ page }) => {
    await page.locator('#kToken [data-value="0"]').click();
    await page.locator('#kPos').fill('1');
    await expect(readout(page, 'score')).toHaveText('−1.353');
  });

  test('try this 2: a bigger base freezes the slow hand and leaves the fast one alone', async ({ page }) => {
    await page.locator('#base').fill('1');
    await expect(cells(page, 'offset-row')).toHaveText(['3.000', '1.618', '−1.253', '−2.977', '−1.971', '0.838', '2.866', '2.244']);
    await expect(readout(page, 'wavelength-1')).toHaveText('6.3 tokens');
    await expect(readout(page, 'wavelength-2')).toHaveText('628.3 tokens');
    await expect(readout(page, 'score')).toHaveText('1.618');
  });

  test('real heads: the slowest pair of a 128-number head at each model\'s base', async ({ page }) => {
    await expect(readout(page, 'real-deepseek-v4-pro')).toHaveText('54.4K');
    await expect(readout(page, 'real-gpt-oss-120b')).toHaveText('782K');
    await expect(readout(page, 'real-minimax-m3')).toHaveText('24.7M');
    await expect(readout(page, 'real-glm-5.3')).toHaveText('39.2M');
    await expect(readout(page, 'real-qwen3.8')).toHaveText('48.8M');
  });

  test('try this 3: reading past the trained length hits unseen angles; PI and YaRN-style bring them back', async ({ page }) => {
    await page.locator('#target').fill('2');
    await expect(readout(page, 'reached-2')).toHaveText('6.3 rad');
    await expect(readout(page, 'verdict-2')).toHaveText('never seen');
    await expect(readout(page, 'verdict-1')).toHaveText('all seen');
    await page.locator('#stretch [data-value="pi"]').click();
    await expect(readout(page, 'verdict-2')).toHaveText("all seen (within one token's step)");
    await expect(readout(page, 'verdict-1')).toHaveText('all seen');
    await expect(cells(page, 'offset-row')).toHaveText(['3.000', '2.900', '2.620', '2.176', '1.596', '0.915', '0.175', '−0.578']);
    await page.locator('#stretch [data-value="yarn-simple"]').click();
    await expect(cells(page, 'offset-row')).toHaveText(['3.000', '1.615', '−1.261', '−2.989', '−1.986', '0.820', '2.843', '2.218']);
    await expect(readout(page, 'verdict-2')).toHaveText("all seen (within one token's step)");
  });
});

