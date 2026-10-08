import { test, expect } from '@playwright/test';
import { registerLessonContract } from './lesson-helpers.js';
import { collectConsoleErrors } from './helpers.js';
import { CAPTIONS, CHECK_WORK, PATTERN_A, PATTERN_B_SHARED, PATTERN_B_OWN } from '../tests/kv-compression-expected.js';

const URL = '/architecture/#kv-compression';

registerLessonContract({ name: 'kv-compression', url: URL, captions: CAPTIONS, factRows: 9, returnHash: 'kv-compression' });

const readout = (page, name) => page.locator(`[data-readout="${name}"]`);
const pattern = (page, name) => page.locator(`[data-readout="${name}"] .g-cell .g-text`);
const pick = (page, id, value) => page.locator(`#${id} [data-value="${value}"]`).click();

test.describe('kv-compression toy: share, group or compress', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('opens on the toy: plain MHA, 8 query heads of 4 numbers, 64 numbers per token per layer, and the exact "Check my work" box', async ({ page }) => {
    const errors = collectConsoleErrors(page);
    await expect(readout(page, 'per-layer')).toHaveText('64');
    await expect(readout(page, 'bytes-per-token')).toHaveText('128 B');
    await expect(readout(page, 'times-smaller')).toHaveText('1×');
    await expect(readout(page, 'cache')).toHaveText('16.8 MB');
    await expect(readout(page, 'groups')).toHaveText('8 query heads read 8 KV heads: 1 query head per KV head');
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK);
    await expect(page.locator('#kvHeads')).toBeHidden();
    await expect(page.locator('#latent')).toBeHidden();
    expect(errors).toEqual([]);
  });

  test('toy sharing: GQA with 2 KV heads is 16 numbers, MQA 8, MLA 10 (between MQA and GQA-2); a wider latent grows it', async ({ page }) => {
    await pick(page, 'scheme', 'gqa');
    await expect(page.locator('#kvHeads')).toBeVisible();
    await expect(readout(page, 'per-layer')).toHaveText('16');
    await expect(readout(page, 'times-smaller')).toHaveText('4×');
    await expect(readout(page, 'groups')).toHaveText('8 query heads read 2 KV heads: 4 query heads per KV head');
    await pick(page, 'scheme', 'mqa');
    await expect(readout(page, 'per-layer')).toHaveText('8');
    await expect(readout(page, 'times-smaller')).toHaveText('8×');
    await pick(page, 'scheme', 'mla');
    await expect(page.locator('#latent')).toBeVisible();
    await expect(readout(page, 'per-layer')).toHaveText('10');
    await expect(readout(page, 'bytes-per-token')).toHaveText('20 B');
    await expect(readout(page, 'times-smaller')).toHaveText('6.4×');
    await page.locator('#latent').fill('2');
    await expect(readout(page, 'per-layer')).toHaveText('18');
  });

  test('try this 1: GPT-3 MHA is 4,718,592 B per token; GQA with 8 KV heads 393,216 B (12×); MQA 49,152 B (96×); the query heads stay', async ({ page }) => {
    await pick(page, 'model', 'gpt3');
    await expect(readout(page, 'bytes-per-token')).toHaveText('4,718,592 B');
    await expect(readout(page, 'cache')).toHaveText('618 GB');
    await expect(readout(page, 'what-if')).toContainText('a what-if at this shape; its own context was 2,048');
    await pick(page, 'scheme', 'gqa');
    await expect(readout(page, 'bytes-per-token')).toHaveText('393,216 B');
    await expect(readout(page, 'times-smaller')).toHaveText('12×');
    await expect(readout(page, 'groups')).toHaveText('96 query heads read 8 KV heads: 12 query heads per KV head');
    await pick(page, 'scheme', 'mqa');
    await expect(readout(page, 'bytes-per-token')).toHaveText('49,152 B');
    await expect(readout(page, 'times-smaller')).toHaveText('96×');
    await expect(readout(page, 'groups')).toHaveText('96 query heads read 1 KV head: 96 query heads per KV head');
  });

  test('try this 2: head B reading head A\'s keys puts 0.731 on "The" from "cat"; with its own keys, 0.798 on "The"; head A never changes', async ({ page }) => {
    await expect(pattern(page, 'pattern-a')).toHaveText(PATTERN_A);
    await expect(pattern(page, 'pattern-b')).toHaveText(PATTERN_B_SHARED);
    await expect(page.locator('#pattern')).toHaveAttribute('aria-pressed', 'true');
    await page.locator('#pattern').click();
    await expect(page.locator('#pattern')).toHaveAttribute('aria-pressed', 'false');
    await expect(pattern(page, 'pattern-b')).toHaveText(PATTERN_B_OWN);
    await expect(pattern(page, 'pattern-a')).toHaveText(PATTERN_A);
  });

  test('try this 3: DeepSeek-V3 MLA is 576 numbers, 70,272 B, 56.9×; GQA with 2 KV heads 512; MQA 256', async ({ page }) => {
    await pick(page, 'model', 'v3');
    await expect(page.locator('#scheme [data-value="mla"]')).toHaveAttribute('aria-pressed', 'true');
    await expect(readout(page, 'per-layer')).toHaveText('576');
    await expect(readout(page, 'bytes-per-token')).toHaveText('70,272 B');
    await expect(readout(page, 'times-smaller')).toHaveText('56.9×');
    await expect(readout(page, 'cache')).toHaveText('9.21 GB');
    await expect(readout(page, 'what-if')).toHaveText('');
    await pick(page, 'scheme', 'gqa');
    await page.locator('#kvHeads').fill('1');
    await expect(readout(page, 'per-layer')).toHaveText('512');
    await pick(page, 'scheme', 'mqa');
    await expect(readout(page, 'per-layer')).toHaveText('256');
  });

  test('real chips: Llama-3.1-70B is 327,680 B (GQA-8); MiniMax-M3 is 122,880 B (GQA-4) and 16.1 GB at 131,072 tokens', async ({ page }) => {
    await pick(page, 'model', 'llama');
    await expect(readout(page, 'bytes-per-token')).toHaveText('327,680 B');
    await expect(readout(page, 'cache')).toHaveText('42.9 GB');
    await expect(readout(page, 'what-if')).toHaveText('');
    await pick(page, 'model', 'minimax');
    await expect(readout(page, 'bytes-per-token')).toHaveText('122,880 B');
    await expect(readout(page, 'times-smaller')).toHaveText('16×');
    await expect(readout(page, 'cache')).toHaveText('16.1 GB');
    await pick(page, 'model', 'toy');
    await expect(readout(page, 'bytes-per-token')).toHaveText('128 B');
  });

  test('the context slider: past a model\'s own context the what-if label shows', async ({ page }) => {
    await pick(page, 'model', 'llama');
    await page.locator('#context').fill('3');
    await expect(readout(page, 'cache')).toHaveText('344 GB');
    await expect(readout(page, 'what-if')).toContainText('a what-if at this shape; its own context was 131,072');
    await page.locator('#context').fill('0');
    await expect(readout(page, 'what-if')).toHaveText('');
  });
});
