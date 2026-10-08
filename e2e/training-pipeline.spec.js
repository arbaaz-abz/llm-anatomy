import { test, expect } from '@playwright/test';
import { registerLessonContract } from './lesson-helpers.js';
import { collectConsoleErrors } from './helpers.js';
import { CAPTIONS, CHECK_WORK, CHECK_WORK_KIMI } from '../tests/training-pipeline-expected.js';

const URL = '/training/#training-pipeline';

registerLessonContract({ name: 'training-pipeline', url: URL, captions: CAPTIONS, factRows: 10, returnHash: 'training-pipeline' });

const readout = (page, name) => page.locator(`[data-readout="${name}"]`);
const stage = (page, n) => page.locator(`[data-section="toy"] g[role="button"][data-stage="${n}"]`);
const model = (page, value) => page.locator(`#model [data-value="${value}"]`);
const barText = (page) => page.locator('[data-section="toy"] .g-share').first();

test.describe('training-pipeline toy: read a recipe', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('opens on GLM-5 with stage 1 selected and the exact "Check my work" box', async ({ page }) => {
    const errors = collectConsoleErrors(page);
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK);
    await expect(readout(page, 'known-total')).toHaveText('28.55T tokens');
    await expect(readout(page, 'inspector')).toHaveText('Stage 1, pretraining: 27T tokens of base pretraining at 4K context');
    await expect(stage(page, 1)).toHaveAttribute('aria-pressed', 'true');
    await expect(page.locator('[data-section="toy"] g[role="button"]')).toHaveCount(6);
    await expect(barText(page).locator('.g-pct').first()).toHaveText('94.6%');
    expect(errors).toEqual([]);
  });

  test('stage blocks are keyboard buttons: arrows move, Enter and Space select, focus is kept', async ({ page }) => {
    await stage(page, 1).focus();
    await page.keyboard.press('ArrowRight');
    await expect(stage(page, 2)).toHaveAttribute('aria-pressed', 'true');
    await expect(stage(page, 2)).toBeFocused();
    await page.keyboard.press('ArrowRight');
    await page.keyboard.press('ArrowRight');
    await expect(stage(page, 4)).toBeFocused();
    await expect(stage(page, 4)).toHaveAttribute('aria-label', 'stage 4: specialist reinforcement learning');
    await expect(readout(page, 'inspector')).toContainText('Stage 4, specialist reinforcement learning:');
    await page.keyboard.press('ArrowLeft');
    await page.keyboard.press('Enter');
    await expect(stage(page, 3)).toHaveAttribute('aria-pressed', 'true');
    await page.keyboard.press('End');
    await expect(stage(page, 6)).toBeFocused();
    await page.keyboard.press('Home');
    await page.keyboard.press(' ');
    await expect(stage(page, 1)).toHaveAttribute('aria-pressed', 'true');
    await stage(page, 5).click();
    await expect(readout(page, 'inspector')).toContainText('Stage 5, merging by on-policy distillation:');
  });

  test('try this 1: the merge stage is lit for GLM-5, DeepSeek-V4 and Kimi K3, not described for Nemotron 3 Super and Olmo 3', async ({ page }) => {
    await stage(page, 5).click();
    const expected = { glm5: true, deepseekV4: true, kimiK3: true, nemotron3Super: false, olmo3: false };
    for (const [value, lit] of Object.entries(expected)) {
      await model(page, value).click();
      const text = lit ? /^Stage 5, merging by on-policy distillation: (?!not described)/ : 'Stage 5, merging by on-policy distillation: not described in this report.';
      await expect(readout(page, 'inspector')).toHaveText(text);
      await expect(stage(page, 5)).toHaveAttribute('aria-label', lit ? 'stage 5: merging by on-policy distillation' : 'stage 5: merging by on-policy distillation (not described)');
    }
  });

  test('try this 2: GLM-5 counts stage 4 in environments; Olmo 3 is 98.3% / 1.7% with traces, pairs and prompts', async ({ page }) => {
    await stage(page, 4).click();
    await expect(readout(page, 'inspector')).toContainText('software-engineering environments: >10,000 (9 languages)');
    await expect(page.locator('[data-section="toy"] .g-share').first()).toContainText('post-training: not published');
    await model(page, 'olmo3').click();
    await expect(readout(page, 'known-total')).toHaveText('6T tokens');
    await expect(readout(page, 'check-work')).toContainText('pretrain    = 5.9T ÷ 6T');
    await expect(readout(page, 'check-work')).toContainText('= 98.3%');
    await expect(readout(page, 'check-work')).toContainText('= 1.7%');
    await expect(readout(page, 'inspector')).toContainText('200K preference pairs (DPO), then 105K prompts (RLVR) (reported)');
    await stage(page, 3).click();
    await expect(readout(page, 'inspector')).toContainText('2.3M reasoning traces (reported)');
  });

  test('try this 3: Kimi K3 publishes no shares, yet every stage is described', async ({ page }) => {
    await model(page, 'kimiK3').click();
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK_KIMI);
    await expect(readout(page, 'known-total')).toHaveText('none published');
    await expect(page.locator('[data-section="toy"] .g-share').first()).toContainText('no published shares');
    await expect(page.locator('[data-section="toy"] g[role="button"][aria-label*="not described"]')).toHaveCount(0);
    await expect(page.locator('[data-section="toy"] .g-share .g-hatch')).toHaveCount(0);
  });

  test('the try-this list closes the toy', async ({ page }) => {
    const items = page.locator('[data-section="toy"] ol.try-this li');
    await expect(items).toHaveCount(3);
    await expect(items.first()).toContainText('Insight: merging specialists is the 2026 frontier step,');
    await expect(items.nth(1)).toContainText('pretrain 94.6%, mid-train 5.4%, post-training not published');
  });
});
