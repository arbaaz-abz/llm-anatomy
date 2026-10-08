import { test, expect } from '@playwright/test';
import { registerLessonContract } from './lesson-helpers.js';
import { CAPTIONS, CHECK_WORK } from '../tests/rlhf-dpo-expected.js';

const URL = '/training/#rlhf-dpo';

registerLessonContract({ name: 'rlhf-dpo', url: URL, captions: CAPTIONS, factRows: 7, returnHash: 'rlhf-dpo' });

const readout = (page, name) => page.locator(`[data-readout="${name}"]`);
const setSlider = (page, id, value) => page.locator(`#${id}`).evaluate((input, v) => {
  input.value = String(v);
  input.dispatchEvent(new Event('input', { bubbles: true }));
}, value);
const expectOutputs = async (page, { rewards, margin, p, loss, weight }) => {
  await expect(page.locator('[data-readout="rewards"] .g-cell .g-text')).toHaveText(rewards);
  await expect(readout(page, 'margin')).toHaveText(margin);
  await expect(readout(page, 'p-chosen')).toHaveText(p);
  await expect(readout(page, 'loss')).toHaveText(loss);
  await expect(readout(page, 'weight')).toHaveText(weight);
};

test.describe('rlhf-dpo toy: tune a DPO pair', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('opens on the storyboard default with its exact numbers', async ({ page }) => {
    await expectOutputs(page, { rewards: ['0.050', '−0.020'], margin: '0.070', p: '0.517', loss: '0.659', weight: '0.483' });
    await expect(page.locator('[data-section="toy"]')).toContainText('0.693 = ln 2 (no preference yet)');
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK);
    await expect(page.locator('#beta [data-value="0.1"]')).toHaveAttribute('aria-pressed', 'true');
    await expect(page.locator('#dChosen')).toHaveValue('0.5');
    await expect(page.locator('#dRejected')).toHaveValue('-0.2');
  });

  test('the toy ends with the storyboard\'s try-this list: three prompts, each with a named insight', async ({ page }) => {
    const items = page.locator('[data-section="toy"] ol.try-this > li');
    await expect(page.locator('[data-section="toy"] h4', { hasText: 'Try this' })).toHaveCount(1);
    await expect(items).toHaveCount(3);
    await expect(items.nth(0)).toContainText('loss 0.313, update weight 0.269');
    await expect(items.nth(0).locator('strong').last()).toHaveText('Insight: the update fades as a pair is learned.');
    await expect(items.nth(1).locator('strong').last()).toHaveText('Insight: DPO optimizes the gap, not the chosen answer.');
    await expect(items.nth(2)).toContainText('5× less drift');
    await expect(items.nth(2).locator('strong').last()).toHaveText('Insight: β is the leash.');
  });

  test('try this 1: the update fades as a pair is learned', async ({ page }) => {
    await setSlider(page, 'dChosen', 5);
    await setSlider(page, 'dRejected', -5);
    await expectOutputs(page, { rewards: ['0.500', '−0.500'], margin: '1.000', p: '0.731', loss: '0.313', weight: '0.269' });
    await setSlider(page, 'dChosen', 10);
    await setSlider(page, 'dRejected', -10);
    await expectOutputs(page, { rewards: ['1.000', '−1.000'], margin: '2.000', p: '0.881', loss: '0.127', weight: '0.119' });
  });

  test('try this 2: both answers get less likely and the loss still falls', async ({ page }) => {
    await setSlider(page, 'dChosen', -1);
    await setSlider(page, 'dRejected', -3);
    await expectOutputs(page, { rewards: ['−0.100', '−0.300'], margin: '0.200', p: '0.550', loss: '0.598', weight: '0.450' });
    await setSlider(page, 'dChosen', 0);
    await setSlider(page, 'dRejected', 0);
    await expect(readout(page, 'loss')).toHaveText('0.693');
  });

  test('try this 3: beta is the leash', async ({ page }) => {
    await page.locator('#beta [data-value="0.5"]').click();
    await expectOutputs(page, { rewards: ['0.250', '−0.100'], margin: '0.350', p: '0.587', loss: '0.533', weight: '0.413' });
    await page.locator('#beta [data-value="0.1"]').click();
    await setSlider(page, 'dChosen', 3.5);
    await setSlider(page, 'dRejected', 0);
    await expect(readout(page, 'loss')).toHaveText('0.533');
    await page.locator('#beta [data-value="0.5"]').click();
    await setSlider(page, 'dChosen', 0.7);
    await expect(readout(page, 'loss')).toHaveText('0.533');
  });

  test('reset returns to 0.5, −0.2 and β 0.1; check my work follows every change', async ({ page }) => {
    await setSlider(page, 'dChosen', 2);
    await expect(readout(page, 'check-work')).toContainText('reward A = 0.1 × 2    = 0.200');
    await page.locator('#beta [data-value="0.5"]').click();
    await page.locator('#toy-reset').click();
    await expectOutputs(page, { rewards: ['0.050', '−0.020'], margin: '0.070', p: '0.517', loss: '0.659', weight: '0.483' });
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK);
    await expect(page.locator('#beta [data-value="0.1"]')).toHaveAttribute('aria-pressed', 'true');
  });
});
