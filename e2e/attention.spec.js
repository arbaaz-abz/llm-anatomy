import { test, expect } from '@playwright/test';
import { registerLessonContract } from './lesson-helpers.js';
import { CAPTIONS, CHECK_WORK } from '../tests/attention-expected.js';

const URL = '/architecture/#attention';

registerLessonContract({ name: 'attention', url: URL, captions: CAPTIONS, factRows: 6, returnHash: 'attention' });

const row = (page, name) => page.locator(`[data-readout="${name}"] .cell`);

test.describe('attention toy: compute one row yourself', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('opens where the animation ended: query sat, head A, divisor 2, mask on', async ({ page }) => {
    await expect(row(page, 'scores')).toHaveText(['−1.00', '3.00', '0.50', '−0.75']);
    await expect(row(page, 'scaled')).toHaveText(['−0.500', '1.500', '0.250', '−0.375']);
    await expect(row(page, 'masked')).toHaveText(['−0.500', '1.500', '0.250', '−∞']);
    await expect(row(page, 'weights')).toHaveText(['0.095', '0.703', '0.202', '0']);
    await expect(row(page, 'output')).toHaveText(['−0.106', '1.407', '0.106', '0.804']);
    await expect(page.locator('[data-readout="weight-sum"]')).toHaveText('Σ = 1.000');
    await expect(page.locator('[data-readout="check-work"]')).toHaveText(CHECK_WORK);
  });

  test('try this 1: mask off — "down" gets weight and cat drops to 0.635', async ({ page }) => {
    await page.locator('#causal').click();
    await expect(page.locator('#causal')).toHaveAttribute('aria-pressed', 'false');
    await expect(row(page, 'weights')).toHaveText(['0.086', '0.635', '0.182', '0.097']);
  });

  test('try this 2: the divisor is a sharpness dial', async ({ page }) => {
    await page.locator('#divisor').fill('0');
    await expect(row(page, 'weights')).toHaveText(['0.000', '0.993', '0.007', '0']);
    await page.locator('#divisor').fill('4');
    await expect(row(page, 'weights')).toHaveText(['0.259', '0.428', '0.313', '0']);
    await page.locator('#divisor-reset').click();
    await expect(row(page, 'weights')).toHaveText(['0.095', '0.703', '0.202', '0']);
  });

  test('try this 3: same tokens, different pattern per head; both heads join into 8 numbers', async ({ page }) => {
    await page.locator('#query [data-value="3"]').click();
    await expect(row(page, 'weights')).toHaveText(['0.114', '0.656', '0.129', '0.101']);
    await page.locator('#head [data-value="B"]').click();
    await expect(row(page, 'weights')).toHaveText(['0.129', '0.146', '0.578', '0.146']);
    await page.locator('#query [data-value="2"]').click();
    await page.locator('#head [data-value="both"]').click();
    await expect(row(page, 'concat')).toHaveText(['−0.106', '1.407', '0.106', '0.804', '0.664', '0.168', '0', '0.168']);
    await expect(page.locator('[data-section="toy"] .g-heatmap')).toHaveCount(2);
  });
});
