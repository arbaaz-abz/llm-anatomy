import { test, expect } from '@playwright/test';
import { registerLessonContract } from './lesson-helpers.js';
import { CAPTIONS } from '../tests/model-card-expected.js';

const URL = '/architecture/#model-card';

registerLessonContract({ name: 'model-card', url: URL, captions: CAPTIONS, factRows: 9, returnHash: 'model-card' });

const out = (page, name) => page.locator(`[data-section="toy"] [data-readout="${name}"]`);
const pick = (page, side, id) => page.locator(`#${side} [data-value="${id}"]`).click();

test.describe('model-card toy: decode a card', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('opens on DeepSeek-V4-Pro against Kimi K3, each card in its own column', async ({ page }) => {
    await expect(page.locator('#left [data-value="deepseek-v4-pro"]')).toHaveAttribute('aria-pressed', 'true');
    await expect(page.locator('#right [data-value="kimi-k3"]')).toHaveAttribute('aria-pressed', 'true');
    await expect(page.locator('#context [data-value="own"]')).toHaveAttribute('aria-pressed', 'true');
    await expect(out(page, 'left-total_params')).toHaveText('1.6T');
    await expect(out(page, 'left-active_params')).toHaveText('49B');
    await expect(out(page, 'left-layers')).toHaveText('61');
    await expect(out(page, 'left-experts_total')).toHaveText('384');
    await expect(out(page, 'left-context_length')).toHaveText('1M');
    await expect(out(page, 'left-modalities')).toHaveText('text');
    await expect(out(page, 'right-total_params')).toHaveText('2.78T');
    await expect(out(page, 'right-active_params')).toHaveText('104.2B');
    await expect(out(page, 'right-layers')).toHaveText('93');
    await expect(page.locator('[data-section="toy"] tr[data-field="total_params"] .fact-lesson')).toContainText('Mixture of Experts');
  });

  test('every field row prints its meaning and its lesson without hover', async ({ page }) => {
    const rows = page.locator('[data-section="toy"] table[data-readout="fields"] tbody tr');
    await expect(rows).toHaveCount(13);
    await expect(page.locator('tr[data-field="active_params"] .ro-sub').first()).toContainText('Weights one token multiplies; sets compute per token.');
    await expect(page.locator('tr[data-field="kv_bytes_per_token"] .ro-sub').first()).toContainText('Bytes of keys and values kept for each token.');
  });

  test('try this 1: V4-Pro against Kimi K3: the bigger total is not the bigger bill per token', async ({ page }) => {
    await expect(out(page, 'left-active-share')).toHaveText('3.06%');
    await expect(out(page, 'right-active-share')).toHaveText('3.75%');
    await expect(out(page, 'left-routed-share')).toHaveText('1.56%');
    await expect(out(page, 'right-routed-share')).toHaveText('1.79%');
  });

  test('the active-share bars fold the sliver into "others" and zoom the first 10% (README lesson 19)', async ({ page }) => {
    const bars = page.locator('[data-section="toy"] .g-share');
    await expect(bars).toHaveCount(2);
    for (const i of [0, 1]) {
      await expect(bars.nth(i).locator('.g-bracket')).toHaveCount(1);
      await expect(bars.nth(i).locator('.g-seg.g-part-others')).toHaveCount(1);
      await expect(bars.nth(i).locator('.g-hatch')).toHaveCount(0);
    }
    await expect(bars.first().locator('.g-pct').filter({ hasText: '3.06%' })).toHaveCount(1);
    await expect(page.locator('[data-section="toy"] .share-zoom-label').first()).toHaveText('zoom on the first 10%');
  });

  test('the cache lines: V4-Pro is a reported range, Kimi K3 publishes nothing to size it from', async ({ page }) => {
    await expect(out(page, 'left-cache-token')).toHaveText('4,000–12,000 B ≈ 4–12 kB');
    await expect(out(page, 'left-cache-conversation')).toHaveText('4–12 GB');
    await expect(out(page, 'left-gpu-share')).toHaveText('5.0–15.0%');
    await expect(out(page, 'right-cache-token')).toHaveText('not published');
    await expect(out(page, 'right-cache-conversation')).toHaveText('not published');
    await expect(out(page, 'right-gpu-share')).toHaveText('not published');
  });

  test('try this 2: the attention line, not the context number, decides what a long conversation costs', async ({ page }) => {
    await pick(page, 'right', 'minimax-m3');
    await expect(out(page, 'right-cache-token')).toHaveText('122,880 B ≈ 123 kB');
    await expect(out(page, 'right-cache-conversation')).toHaveText('129 GB');
    await expect(out(page, 'right-gpu-share')).toHaveText('161.1%');
    await expect(out(page, 'left-cache-conversation')).toHaveText('4–12 GB');
    await pick(page, 'right', 'gpt-oss-120b');
    await expect(out(page, 'right-cache-token')).toHaveText('36,864 B ≈ 36.9 kB + 4.72 MB fixed');
    await expect(out(page, 'right-cache-conversation')).toHaveText('4.84 GB');
    await page.locator('#context [data-value="131072"]').click();
    await expect(out(page, 'left-cache-conversation')).toHaveText('524 MB–1.57 GB');
    await expect(out(page, 'left-gpu-share')).toHaveText('0.7–2.0%');
    await expect(out(page, 'right-cache-conversation')).toHaveText('4.84 GB');
  });

  test('try this 3: GLM-5.3 against Mistral Large 4: a card is a set of claims, and ranges stay ranges', async ({ page }) => {
    await pick(page, 'left', 'glm-5.3');
    await pick(page, 'right', 'mistral-large-4');
    await expect(out(page, 'left-layers')).toHaveText('78–80');
    await expect(out(page, 'left-active_params')).toHaveText('40B');
    await expect(out(page, 'right-context_length')).toHaveText('512K–1M');
    await expect(out(page, 'right-active_params')).toHaveText('49B');
    await expect(out(page, 'right-layers')).toHaveText('not published');
    await expect(out(page, 'right-active-share-embedding')).toHaveText('4.95%');
    const notes = page.locator('[data-section="toy"] .labs-differ');
    await expect(notes).toContainText('config.json');
    await expect(notes).toContainText('GLM-5 paper');
    await expect(notes).toContainText('52B including embeddings');
    await expect(page.locator('tr[data-field="layers"] td[data-col="left"] .fact-reported')).toHaveCount(0);
    await expect(page.locator('tr[data-field="active_params"] td[data-col="left"] .fact-reported')).toHaveCount(1);
  });

  test('"none" leaves a single column', async ({ page }) => {
    await pick(page, 'right', 'none');
    await expect(out(page, 'right-total_params')).toHaveCount(0);
    await expect(out(page, 'left-total_params')).toHaveText('1.6T');
    await expect(page.locator('[data-section="toy"] .g-share')).toHaveCount(1);
    await pick(page, 'right', 'qwen3.8');
    await expect(out(page, 'right-context_length')).toHaveText('262K');
  });
});
