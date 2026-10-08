import { test, expect } from '@playwright/test';
import { registerLessonContract } from './lesson-helpers.js';
import { collectConsoleErrors } from './helpers.js';
import { CAPTIONS, CHECK_WORK, TRY_THIS } from '../tests/training-memory-expected.js';

const URL = '/training/#training-memory';

registerLessonContract({ name: 'training-memory', url: URL, captions: CAPTIONS, factRows: 10, returnHash: 'training-memory' });

const readout = (page, name) => page.locator(`[data-readout="${name}"]`);
const pick = (page, group, value) => page.locator(`#${group} [data-value="${value}"]`).click();

test.describe('training-memory toy: will it fit?', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('opens on GPT-3, Adam, ZeRO 0, 64 GPUs, H100 with the exact "Check my work" box', async ({ page }) => {
    const errors = collectConsoleErrors(page);
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK);
    await expect(readout(page, 'total')).toHaveText('3,075.41 GB');
    await expect(readout(page, 'verdict')).toHaveText('does not fit: 3,075.41 GB vs 80 GB (H100, nominal)');
    await expect(readout(page, 'gpus-to-hold')).toHaveText('35 H100s (nominal)');
    await expect(readout(page, 'traffic')).toHaveText('1×');
    await expect(readout(page, 'extra-compute')).toHaveText('0%');
    expect(errors).toEqual([]);
  });

  test('try this 1: the ZeRO paper table, stages 0 to 3, 120 → 31.41 → 16.64 → 1.88 GB per GPU', async ({ page }) => {
    await pick(page, 'model', 'zero-paper');
    for (const [stage, gb] of TRY_THIS.zeroPaper.entries()) {
      await pick(page, 'stage', String(stage));
      await expect(readout(page, 'state-total')).toHaveText(`${gb} GB`);
    }
    await expect(readout(page, 'traffic')).toHaveText('1.5×');
    await expect(readout(page, 'activations')).toHaveText('activations: shape not modeled here (MoE / MLA layers); states only');
    await expect(page.locator('#seq')).toBeHidden();
  });

  test('try this 2: ZeRO-3 on GPT-3 needs recomputation to fit; ZeRO-0 never fits', async ({ page }) => {
    const t = TRY_THIS.gpt3Zero3;
    await pick(page, 'stage', '3');
    await expect(readout(page, 'state-total')).toHaveText(`${t.state} GB`);
    await expect(readout(page, 'total')).toHaveText(`${t.none} GB`);
    await expect(readout(page, 'verdict')).toContainText('does not fit');
    await pick(page, 'recompute', 'selective');
    await expect(readout(page, 'total')).toHaveText(`${t.selective} GB`);
    await expect(readout(page, 'verdict')).toContainText('does not fit');
    await pick(page, 'recompute', 'full');
    await expect(readout(page, 'total')).toHaveText(`${t.full} GB`);
    await expect(readout(page, 'verdict')).toHaveText('fits: 48.58 GB vs 80 GB (H100, nominal)');
    await expect(readout(page, 'extra-compute')).toHaveText('about +33%');
    await pick(page, 'stage', '0');
    await expect(readout(page, 'total')).toHaveText(`${t.stage0Full} GB`);
  });

  test('try this 3: a 1.6T mixture of experts needs 320 H100s, 143 B200s, 89 B300s; Muon 67; GPT-3 35', async ({ page }) => {
    const k = TRY_THIS.v4Pro;
    await pick(page, 'model', 'deepseek-v4-pro');
    await expect(readout(page, 'gpus-to-hold')).toHaveText(`${k.h100} H100s (nominal)`);
    await pick(page, 'gpu', 'b200');
    await expect(readout(page, 'gpus-to-hold')).toHaveText(`${k.b200} B200s (usable)`);
    await pick(page, 'gpu', 'b300');
    await expect(readout(page, 'gpus-to-hold')).toHaveText(`${k.b300} B300s (nominal)`);
    await pick(page, 'recipe', 'muon');
    await expect(readout(page, 'gpus-to-hold')).toHaveText(`${k.muonB300} B300s (nominal)`);
    await pick(page, 'model', 'gpt-3');
    await pick(page, 'gpu', 'h100');
    await pick(page, 'recipe', 'adam');
    await expect(readout(page, 'gpus-to-hold')).toHaveText(`${k.gpt3H100} H100s (nominal)`);
  });

  test('the GPT-3 shape controls appear only for GPT-3, and the B200 chip says usable', async ({ page }) => {
    await expect(page.locator('#seq')).toBeVisible();
    await expect(page.locator('#recompute')).toBeVisible();
    await expect(page.locator('#gpu [data-value="b200"]')).toHaveText('B200 180 GB usable (192 nominal)');
    await expect(page.locator('#gpu [data-value="h100"]')).toHaveText('H100 80 GB nominal');
    await pick(page, 'model', 'kimi-k2');
    await expect(page.locator('#seq')).toBeHidden();
    await expect(page.locator('#recompute')).toBeHidden();
  });
});
