import { test, expect } from '@playwright/test';
import { registerLessonContract } from './lesson-helpers.js';
import { CAPTIONS, CHECK_WORK, CHECK_WORK_IN_DECAY_LINE } from '../tests/midtraining-expected.js';

const URL = '/training/#midtraining';

registerLessonContract({ name: 'midtraining', url: URL, captions: CAPTIONS, factRows: 9, returnHash: 'midtraining' });

const readout = (page, name) => page.locator(`[data-readout="${name}"]`);
const chip = (page, group, value) => page.locator(`#${group} [data-value="${value}"]`);

test.describe('midtraining toy: plan the end of a run', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('opens on the storyboard default with its exact numbers', async ({ page }) => {
    await expect(readout(page, 'lr-chosen')).toHaveText('1.000');
    await expect(readout(page, 'lr-other')).toHaveText('0.345');
    await expect(readout(page, 'lr-nemotron')).toHaveCount(0);
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK);
    await expect(readout(page, 'tokens-0')).toHaveText('27T');
    await expect(readout(page, 'share-0')).toHaveText('94.57%');
    await expect(readout(page, 'share-1')).toHaveText('3.50%');
    await expect(readout(page, 'share-2')).toHaveText('1.75%');
    await expect(readout(page, 'share-3')).toHaveText('0.18%');
    await expect([0, 1, 2, 3].map((i) => readout(page, `attn-${i}`)).length).toBe(4);
    await expect(readout(page, 'attn-0')).toHaveText('1×');
    await expect(readout(page, 'attn-1')).toHaveText('8×');
    await expect(readout(page, 'attn-2')).toHaveText('32×');
    await expect(readout(page, 'attn-3')).toHaveText('50×');
    await expect(page.locator('[data-section="toy"] .g-share')).toContainText('200K · 3.2% of last 5%');
  });

  test('try this 1: stopping at 60% leaves cosine mid-curve and WSD on the plateau', async ({ page }) => {
    await chip(page, 'schedule', 'cosine').click();
    await expect(readout(page, 'lr-chosen')).toHaveText('0.345');
    await expect(readout(page, 'lr-other')).toHaveText('1.000');
    await chip(page, 'schedule', 'wsd').click();
    await expect(readout(page, 'lr-chosen')).toHaveText('1.000');
    await expect(readout(page, 'lr-other')).toHaveText('0.345');
  });

  test('inside the decay the check line shows the linear fall (stop at 90%)', async ({ page }) => {
    await page.locator('#stopAt').fill('90');
    await expect(readout(page, 'lr-chosen')).toHaveText('0.500');
    await expect(readout(page, 'check-work')).toContainText(CHECK_WORK_IN_DECAY_LINE);
    await expect(readout(page, 'lr-other')).toHaveText('0.024');
  });

  test('minus-sqrt decay (Nemotron\'s shape) falls faster at first: stop at 90% reads 0.293', async ({ page }) => {
    await chip(page, 'schedule', 'wsd-minus-sqrt').click();
    await page.locator('#stopAt').fill('90');
    await expect(readout(page, 'lr-chosen')).toHaveText('0.293');
  });

  test('try this 2: the Nemotron 3 Super and MiniMax-M2 chips set the decay share and show the absolute rate', async ({ page }) => {
    await expect(page.locator('#decayPreset button')).toHaveText(['Nemotron 3 Super 20%', 'MiniMax-M2 31.8%']);
    await expect(page.locator('#decayPreset [aria-pressed]')).toHaveCount(0);
    await expect(page.locator('#stopAt').locator('xpath=..')).toContainText('60%');
    await chip(page, 'decayPreset', 'nemotron').click();
    await expect(page.locator('#decayFrac')).toHaveValue('3');
    await expect(chip(page, 'schedule', 'wsd-minus-sqrt')).toHaveAttribute('aria-pressed', 'true');
    await expect(readout(page, 'lr-nemotron')).toHaveText('4.50 × 10⁻⁴');
    await page.locator('#stopAt').fill('85');
    await expect(readout(page, 'lr-chosen')).toHaveText('0.500');
    await expect(readout(page, 'lr-nemotron')).toHaveText('2.27 × 10⁻⁴');
    await chip(page, 'decayPreset', 'minimax').click();
    await expect(page.locator('#decayFrac')).toHaveValue('6');
    await expect(readout(page, 'lr-nemotron')).toHaveCount(0);
    await expect(readout(page, 'check-work')).toContainText('100% − 31.8% = 68.2% of the run');
  });

  test('try this 3: the other runs publish no per-stage tokens, so their bars say so', async ({ page }) => {
    await chip(page, 'run', 'kimi-k3').click();
    await expect(page.locator('[data-section="toy"] .g-share')).toContainText('no published shares');
    await expect(readout(page, 'tokens-0')).toHaveText('not published');
    await expect(readout(page, 'attn-3')).toHaveText('125×');
    await chip(page, 'run', 'deepseek-v4-pro').click();
    await expect(readout(page, 'attn-3')).toHaveText('250×');
    await chip(page, 'run', 'glm-5').click();
    await expect(readout(page, 'attn-3')).toHaveText('50×');
  });

  test('the decay controls are hidden when the cosine schedule has no plateau', async ({ page }) => {
    await chip(page, 'schedule', 'cosine').click();
    await expect(page.locator('#decayFrac')).toBeHidden();
    await expect(page.locator('#decayPreset')).toBeHidden();
  });
});
