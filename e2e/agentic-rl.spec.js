import { test, expect } from '@playwright/test';
import { registerLessonContract } from './lesson-helpers.js';
import { CAPTIONS, TRY_THIS } from '../tests/agentic-rl-expected.js';

const URL = '/training/#agentic-rl';

registerLessonContract({ name: 'agentic-rl', url: URL, captions: CAPTIONS, factRows: 10, returnHash: 'agentic-rl' });

const readout = (page, name) => page.locator(`[data-readout="${name}"]`);
const choose = (page, id, value) => page.locator(`#${id} [data-value="${value}"]`).click();
const item = (page, row, pos) => page.locator(`[data-section="toy"] .stage-item[data-value="${row}:${pos}"]`);

test.describe('agentic-rl toy: fix the mismatch', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('opens on row 4\'s second 48 with the storyboard default numbers', async ({ page }) => {
    await expect(readout(page, 'inspector-token')).toHaveText('row 4, token 8: 48');
    await expect(readout(page, 'inspector-a')).toHaveText('−0.577');
    await expect(readout(page, 'inspector-engine')).toHaveText('0.050');
    await expect(readout(page, 'inspector-trainer')).toHaveText('0.160');
    await expect(readout(page, 'inspector-rho')).toHaveText('3.200');
    await expect(readout(page, 'inspector-weight')).toHaveText('1.000');
    await expect(readout(page, 'inspector-masked')).toHaveText('no');
    await expect(readout(page, 'inspector-push')).toHaveText('−0.577');
    await expect(readout(page, 'masked')).toHaveText('0 of 39');
    await expect(readout(page, 'iteration')).toHaveText('16 min');
    await expect(readout(page, 'utilization')).toHaveText('37.5%');
    await expect(readout(page, 'carried')).toHaveText('none');
  });

  test('the toy ends with the storyboard\'s try-this list: three prompts, each with a named insight', async ({ page }) => {
    const items = page.locator('[data-section="toy"] ol.try-this > li');
    await expect(page.locator('[data-section="toy"] h4', { hasText: 'Try this' })).toHaveCount(1);
    await expect(items).toHaveCount(TRY_THIS.length);
    for (const [i, t] of TRY_THIS.entries()) {
      await expect(items.nth(i)).toContainText(t.prompt);
      await expect(items.nth(i).locator('strong')).toHaveText(`Insight: ${t.insight}`);
    }
  });

  test('try this 1: the corrections trade bias for variance', async ({ page }) => {
    await choose(page, 'correction', 'full');
    await expect(readout(page, 'inspector-weight')).toHaveText('3.200');
    await expect(readout(page, 'inspector-push')).toHaveText('−1.848');
    await choose(page, 'correction', 'tis');
    await expect(readout(page, 'inspector-weight')).toHaveText('2.000');
    await expect(readout(page, 'inspector-push')).toHaveText('−1.155');
    await choose(page, 'correction', 'icepop');
    await expect(readout(page, 'inspector-masked')).toHaveText('yes');
    await expect(readout(page, 'inspector-push')).toHaveText('0');
    await expect(readout(page, 'masked')).toHaveText('3 of 39');
  });

  test('try this 2: part of the mismatch is rounding', async ({ page }) => {
    await choose(page, 'correction', 'icepop');
    await choose(page, 'precision', 'fp16');
    await expect(readout(page, 'inspector-rho')).toHaveText('1.156');
    await expect(readout(page, 'masked')).toHaveText('0 of 39');
    await item(page, 6, 4).click();
    await expect(readout(page, 'inspector-rho')).toHaveText('1.116');
    await item(page, 5, 0).click();
    await expect(readout(page, 'inspector-rho')).toHaveText('0.892');
  });

  test('try this 3: async is a trade', async ({ page }) => {
    await choose(page, 'lambda', 'six');
    await expect(readout(page, 'iteration')).toHaveText('6 min');
    await expect(readout(page, 'utilization')).toHaveText('72.9%');
    await expect(readout(page, 'carried')).toHaveText('rows 4 and 7');
    await choose(page, 'lambda', 'four');
    await expect(readout(page, 'iteration')).toHaveText('4 min');
    await expect(readout(page, 'utilization')).toHaveText('87.5%');
    await expect(readout(page, 'carried')).toHaveText('rows 4, 5, 7 and 8');
  });

  test('a token is selected by click and by the arrow keys', async ({ page }) => {
    await item(page, 5, 0).click();
    await expect(readout(page, 'inspector-token')).toHaveText('row 6, token 1: 63');
    await expect(readout(page, 'inspector-rho')).toHaveText('0.400');
    await item(page, 5, 0).press('ArrowDown');
    await expect(readout(page, 'inspector-token')).toHaveText('row 7, token 1: 7');
    await item(page, 6, 0).press('ArrowRight');
    await expect(readout(page, 'inspector-token')).toHaveText('row 7, token 2: ×');
  });

  test('a masked token\'s selection shows masked: yes and a push of exactly 0', async ({ page }) => {
    await choose(page, 'correction', 'icepop');
    await item(page, 6, 4).click();
    await expect(readout(page, 'inspector-masked')).toHaveText('yes');
    await expect(readout(page, 'inspector-push')).toHaveText('0');
  });
});
