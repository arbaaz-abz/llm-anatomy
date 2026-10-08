import { test, expect } from '@playwright/test';
import { registerLessonContract } from './lesson-helpers.js';
import { CAPTIONS, CHECK_WORK_A, CHECK_WORK_B } from '../tests/kv-cache-expected.js';

const URL = '/architecture/#kv-cache';

registerLessonContract({ name: 'kv-cache', url: URL, captions: CAPTIONS, factRows: 6, returnHash: 'kv-cache' });

const readout = (page, name) => page.locator(`[data-readout="${name}"]`);
const model = (page, value) => page.locator(`#model [data-value="${value}"]`).click();
// Snapped sliders move by stop index, not by value.
const stop = (page, id, index) => page.locator(`#${id}`).fill(String(index));
const PROMPT = { 1000: 3 };
const REPLY = { 1000: 5 };
const CONTEXT = { 2048: 0, 131072: 3, 1048576: 5 };
const SEQUENCES = { 2: 1, 8: 3 };

test.describe('kv-cache toy: count the work, then weigh the memory', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('opens on the animation\'s numbers and the exact "Check my work" boxes', async ({ page }) => {
    await expect(readout(page, 'positions')).toHaveText('7');
    await expect(readout(page, 'key-reads')).toHaveText('28');
    await expect(readout(page, 'positions-flipped')).toHaveText('22');
    await expect(readout(page, 'key-reads-flipped')).toHaveText('74');
    await expect(readout(page, 'positions-ratio')).toHaveText('3.14×');
    await expect(readout(page, 'key-reads-ratio')).toHaveText('2.64×');
    await expect(readout(page, 'bytes-per-token')).toHaveText('64 B');
    await expect(readout(page, 'cache-one')).toHaveText('131 kB');
    await expect(readout(page, 'check-work-a')).toHaveText(CHECK_WORK_A);
    await expect(readout(page, 'check-work-b')).toHaveText(CHECK_WORK_B);
    await expect(page.locator('#cache')).toHaveAttribute('aria-pressed', 'true');
  });

  test('try this 1: the cache removes recomputation, not reading', async ({ page }) => {
    await page.locator('#cache').click();
    await expect(page.locator('#cache')).toHaveAttribute('aria-pressed', 'false');
    await expect(readout(page, 'positions')).toHaveText('22');
    await expect(readout(page, 'positions-flipped')).toHaveText('7');
    await page.locator('#cache').click();
    await stop(page, 'prompt', PROMPT[1000]);
    await stop(page, 'reply', REPLY[1000]);
    await expect(readout(page, 'positions')).toHaveText('1,999');
    await expect(readout(page, 'positions-flipped')).toHaveText('1,499,500');
    await expect(readout(page, 'positions-ratio')).toHaveText('750×');
    await expect(readout(page, 'key-reads')).toHaveText('1,999,000');
    await expect(readout(page, 'key-reads-flipped')).toHaveText('1,166,666,500');
    await expect(readout(page, 'key-reads-ratio')).toHaveText('584×');
  });

  test('try this 2: one 128K Llama-3.1-70B conversation is 54 % of an H100; two do not fit; GPT-3 at 8 conversations is 96.6 %', async ({ page }) => {
    await model(page, 'llama');
    await stop(page, 'context', CONTEXT[131072]);
    await expect(readout(page, 'bytes-per-token')).toHaveText('327,680 B');
    await expect(readout(page, 'cache-one')).toHaveText('42.9 GB');
    await expect(readout(page, 'share-h100')).toHaveText('53.7%');
    await stop(page, 'sequences', SEQUENCES[2]);
    await expect(readout(page, 'cache-all')).toHaveText('85.9 GB');
    await expect(readout(page, 'share-h100')).toHaveText('107.4%');
    await expect(readout(page, 'fit-h100')).toHaveText('does not fit (before the weights)');
    await model(page, 'gpt3');
    await stop(page, 'context', CONTEXT[2048]);
    await stop(page, 'sequences', SEQUENCES[8]);
    await expect(readout(page, 'cache-all')).toHaveText('77.3 GB');
    await expect(readout(page, 'share-h100')).toHaveText('96.6%');
    await expect(readout(page, 'context-note')).not.toContainText('beyond');
  });

  test('try this 3: GPT-3 → Llama → DeepSeek-V3 shrinks the per-token cache; KV heads 96 → 8 is the biggest lever', async ({ page }) => {
    await stop(page, 'context', CONTEXT[131072]);
    await model(page, 'gpt3');
    await expect(readout(page, 'bytes-per-token')).toHaveText('4,718,592 B');
    await expect(readout(page, 'cache-one')).toHaveText('618 GB');
    await expect(readout(page, 'context-note')).toContainText('beyond this model\'s 2,048-token context: a what-if at its shape');
    await model(page, 'llama');
    await expect(readout(page, 'bytes-per-token')).toHaveText('327,680 B');
    await expect(readout(page, 'cache-one')).toHaveText('42.9 GB');
    await expect(readout(page, 'context-note')).not.toContainText('beyond');
    await model(page, 'v3');
    await expect(readout(page, 'bytes-per-token')).toHaveText('70,272 B');
    await expect(readout(page, 'cache-one')).toHaveText('9.21 GB');
    await expect(readout(page, 'formula-note')).toContainText('latent');
    await model(page, 'gpt3');
    await model(page, 'toy');
    await expect(page.locator('#layers')).toBeVisible();
    await expect(page.locator('#kvHeads')).toHaveValue('96');
    await page.locator('#kvHeads').fill('8');
    await expect(readout(page, 'bytes-per-token')).toHaveText('393,216 B');
    await expect(readout(page, 'cache-one')).toHaveText('51.5 GB');
  });

  test('the toy chip builds GPT-3 by hand: 96 blocks, 96 KV heads, head size 128, 2 bytes', async ({ page }) => {
    await page.locator('#layers').fill('96');
    await page.locator('#kvHeads').fill('96');
    await stop(page, 'headDim', 2);
    await expect(readout(page, 'bytes-per-token')).toHaveText('4,718,592 B');
    await expect(readout(page, 'check-work-b')).toContainText('= 4,718,592 B');
  });

  test('DeepSeek-V4-Pro prints the reported range, hides the shape sliders and calls it an estimate', async ({ page }) => {
    await model(page, 'v4pro');
    await expect(readout(page, 'bytes-per-token')).toHaveText('4,000–12,000 B');
    await expect(page.locator('#layers')).toBeHidden();
    await expect(readout(page, 'formula-note')).toContainText('formula-derived estimate; the layer mix is uncertain');
    await stop(page, 'context', CONTEXT[1048576]);
    await expect(readout(page, 'cache-one')).toHaveText('4.19 GB–12.6 GB');
    await expect(readout(page, 'share-h100')).toHaveText('5.2%–15.7%');
    await expect(readout(page, 'context-note')).toContainText('beyond this model\'s 1,000,000-token context');
  });

  test('the second GPU is a B300 with its reported capacity', async ({ page }) => {
    await model(page, 'llama');
    await stop(page, 'context', CONTEXT[131072]);
    await expect(readout(page, 'share-b300')).toHaveText('14.9%');
    await expect(page.locator('[data-section="toy"] .g-gpu')).toHaveCount(2);
  });
});
