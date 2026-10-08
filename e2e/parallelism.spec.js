import { test, expect } from '@playwright/test';
import { registerLessonContract } from './lesson-helpers.js';
import { CAPTIONS, CHECK_WORK, TRY_THIS, SCHEDULE_NOTE } from '../tests/parallelism-expected.js';

const URL = '/training/#parallelism';

registerLessonContract({ name: 'parallelism', url: URL, captions: CAPTIONS, factRows: 8, returnHash: 'parallelism' });

const readout = (page, name) => page.locator(`[data-readout="${name}"]`);
const lastIndex = (stops) => String(stops.length - 1);
const DP_LAST = lastIndex([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]);

test.describe('parallelism toy: schedule a pipeline, then count the GPUs', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('opens on the storyboard default with its exact numbers', async ({ page }) => {
    await expect(readout(page, 'bubble')).toHaveText('42.9%');
    await expect(readout(page, 'peak')).toHaveText('4');
    await expect(readout(page, 'step-length')).toHaveText('14');
    await expect(readout(page, 'gpus')).toHaveText('8,192');
    await expect(readout(page, 'state-per-gpu')).toHaveText('7.02 GB');
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK);
    await expect(page.locator('[data-section="toy"]')).toContainText(SCHEDULE_NOTE);
  });

  test('the toy ends with the storyboard\'s try-this list: three prompts, each with a named insight', async ({ page }) => {
    const items = page.locator('[data-section="toy"] ol.try-this > li');
    await expect(items).toHaveCount(3);
    for (let i = 0; i < 3; i += 1) await expect(items.nth(i)).toHaveText(TRY_THIS[i]);
  });

  test('try this 1: more micro-batches shrink the bubble; a deep pipeline needs many more', async ({ page }) => {
    for (const [chip, bubble] of [['4', '42.9%'], ['8', '27.3%'], ['16', '15.8%'], ['32', '8.6%']]) {
      await page.locator(`#micro-chips [data-value="${chip}"]`).click();
      await expect(readout(page, 'bubble')).toHaveText(bubble);
      await expect(page.locator('#micro')).toHaveValue(chip);
    }
    await page.locator('#stages [data-value="8"]').click();
    await expect(readout(page, 'bubble')).toHaveText('17.9%');
  });

  test('try this 2: 1F1B caps activation memory without changing the bubble', async ({ page }) => {
    await page.locator('#micro-chips [data-value="16"]').click();
    await expect(readout(page, 'peak')).toHaveText('16');
    await expect(readout(page, 'bubble')).toHaveText('15.8%');
    await page.locator('#schedule [data-value="1f1b"]').click();
    await expect(readout(page, 'peak')).toHaveText('4');
    await expect(readout(page, 'bubble')).toHaveText('15.8%');
    await expect(readout(page, 'step-length')).toHaveText('38');
  });

  test('a small grid prints its cells; a wide grid is label-free', async ({ page }) => {
    await page.locator('#stages [data-value="3"]').click();
    await expect(readout(page, 'step-length')).toHaveText('12');
    await expect(page.locator('[data-readout="lanes"] svg text', { hasText: /^F1$/ }).first()).toBeVisible();
    await page.locator('#micro').fill('5');
    await expect(page.locator('[data-readout="lanes"] svg text', { hasText: /^F1$/ })).toHaveCount(0);
  });

  test('try this 3: degrees multiply into GPUs; ZeRO-3 alone reaches the same state', async ({ page }) => {
    await expect(page.locator('#preset [aria-pressed]')).toHaveCount(0);
    await page.locator('#zero [data-value="1"]').click();
    await expect(readout(page, 'state-per-gpu')).toHaveText('13.25 GB');
    await page.locator('#zero [data-value="2"]').click();
    for (const id of ['tp', 'cp', 'pp', 'dp']) await page.locator(`#${id}`).fill('0');
    await expect(readout(page, 'gpus')).toHaveText('1');
    await expect(readout(page, 'state-per-gpu')).toHaveText('6,480 GB');
    await expect(page.locator('#preset [data-value="custom"]')).toHaveCount(0);
    await expect(page.locator('#micro-chips [aria-pressed]')).toHaveCount(0);
    await page.locator('#zero [data-value="0"]').click();
    await expect(readout(page, 'state-per-gpu')).toHaveText('6,480 GB');
    await page.locator('#dp').fill(DP_LAST);
    await page.locator('#zero [data-value="3"]').click();
    await expect(readout(page, 'gpus')).toHaveText('8,192');
    await expect(readout(page, 'state-per-gpu')).toHaveText('0.79 GB');
    await page.locator('#preset [data-value="8k"]').click();
    await expect(readout(page, 'state-per-gpu')).toHaveText('0.79 GB');
  });

  test('presets set the degrees: 16K and long context are 16,384 GPUs', async ({ page }) => {
    await page.locator('#preset [data-value="16k"]').click();
    await expect(readout(page, 'gpus')).toHaveText('16,384');
    await page.locator('#preset [data-value="long"]').click();
    await expect(readout(page, 'gpus')).toHaveText('16,384');
    await expect(page.locator('#cp')).toHaveValue('4');
    await expect(readout(page, 'check-work')).toContainText('GPUs    = 8 × 16 × 16 × 8 = 16,384');
  });
});
