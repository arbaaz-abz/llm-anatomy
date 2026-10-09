import { test, expect } from '@playwright/test';
import { registerLessonContract } from './lesson-helpers.js';
import { collectConsoleErrors } from './helpers.js';
import { CAPTIONS, CHECK_WORK, TRY_THIS } from '../tests/disaggregation-expected.js';

const URL = '/serving/#disaggregation';

registerLessonContract({ name: 'disaggregation', url: URL, captions: CAPTIONS, factRows: 11, returnHash: 'disaggregation' });

const readout = (page, name) => page.locator(`[data-section="toy"] [data-readout="${name}"]`);
// Both sliders snap to stops: index i is the i-th stop.
const PROMPT_INDEX = Object.freeze({ 128: 0, 512: 1, 4096: 2, 8192: 3, 32768: 4, 131072: 5 });
const EP_INDEX = Object.freeze({ 1: 0, 8: 1, 16: 2, 32: 3, 72: 4 });
const setPrompt = (page, n) => page.locator('#prompt').fill(String(PROMPT_INDEX[n]));
const setEp = (page, n) => page.locator('#ep').fill(String(EP_INDEX[n]));
const link = (page, id) => page.locator(`#link [data-value="${id}"]`);

async function expectReadouts(page, expected) {
  for (const [name, text] of Object.entries(expected)) await expect(readout(page, name), name).toHaveText(text);
}

test.describe('disaggregation toy: ship the KV, feed the experts', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('opens on the storyboard default (4,096 tokens, 400 Gb/s, BF16 KV, EP 16, 64 users) with its exact numbers and "Check my work"', async ({ page }) => {
    const errors = collectConsoleErrors(page);
    await expectReadouts(page, {
      'kv-bytes': '1.34 GB', transfer: '26.8 ms', prefill: '290 ms', ratio: '9.3%', 'mla-kv': '288 MB', 'mla-transfer': '5.76 ms',
      'experts-per-gpu': '24', weights: '54.1 GB', free: '234 GB', 'tokens-per-expert': '16', intensity: '56.5', ridge: '1,875', 'tokens-needed': '699', verdict: 'memory-bound',
    });
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK);
    await expect(readout(page, 'hbm-note')).toHaveText('288 GB HBM per GPU (nominal); weights 865 GB as shipped (reported)');
    await expect(readout(page, 'short-note')).toBeHidden();
    await expect(page.locator('[data-section="toy"]')).toContainText('Floors from bytes, bandwidth and FLOPs; real transfers add start-up latency, and real all-to-all adds communication time not modeled here.');
    await expect(page.locator('[data-section="toy"] label[for="link"], [data-section="toy"] #link')).toContainText('each way');
    expect(errors).toEqual([]);
  });

  test('the try-this list closes the toy, verbatim', async ({ page }) => {
    await expect(page.locator('[data-section="toy"] ol.try-this > li')).toHaveText(TRY_THIS);
  });

  test('try this 1: past the crossover the ratio is the same at every prompt; below it, it is smaller', async ({ page }) => {
    await setPrompt(page, 512);
    await expectReadouts(page, { transfer: '3.36 ms', prefill: '36.2 ms', ratio: '9.3%' });
    await setPrompt(page, 131072);
    await expectReadouts(page, { transfer: '859 ms', prefill: '9.27 s', ratio: '9.3%', 'kv-bytes': '42.9 GB' });
    await setPrompt(page, 128);
    await expectReadouts(page, { transfer: '839 µs', prefill: '15 ms', ratio: '5.6%' });
    await expect(readout(page, 'short-note')).toHaveText('below 217 tokens prefill is one weight read, so the ratio is smaller here; real transfers add a fixed start-up cost the toy does not model, which is why short prompts gain least');
    await setPrompt(page, 512);
    await expect(readout(page, 'short-note')).toBeHidden();
  });

  test('try this 2: a fast link or a smaller KV makes the split nearly free', async ({ page }) => {
    await link(page, 'net800').click();
    await expectReadouts(page, { ratio: '4.6%', transfer: '13.4 ms' });
    await link(page, 'nvlink').click();
    await expectReadouts(page, { ratio: '0.5%', transfer: '1.49 ms' });
    await page.locator('#kv [data-value="fp8"]').click();
    await expectReadouts(page, { ratio: '0.3%', 'kv-bytes': '671 MB', 'mla-kv': '144 MB' });
    await link(page, 'net400').click();
    await expectReadouts(page, { ratio: '4.6%', transfer: '13.4 ms' });
    await page.locator('#kv [data-value="bf16"]').click();
    await link(page, 'net400').click();
    await expectReadouts(page, { 'mla-kv': '288 MB', 'mla-transfer': '5.76 ms' });
  });

  test('try this 3: wide expert parallelism shrinks the weights and pools the tokens; EP 1 does not fit', async ({ page }) => {
    await setEp(page, 1);
    await expectReadouts(page, { free: 'does not fit', 'tokens-per-expert': '1', weights: '865 GB', 'experts-per-gpu': '384' });
    await setEp(page, 8);
    await expectReadouts(page, { 'tokens-per-expert': '8', weights: '108 GB', free: '180 GB' });
    await setEp(page, 72);
    await expectReadouts(page, { 'tokens-per-expert': '72', weights: '12 GB', free: '276 GB', 'experts-per-gpu': '5.33' });
    await page.locator('#users [data-value="256"]').click();
    await expectReadouts(page, { 'tokens-per-expert': '288', intensity: '903', verdict: 'memory-bound', 'tokens-needed': '699', ridge: '1,875' });
    await expect(readout(page, 'check-work')).toContainText('256 × 72 × 6 ÷ 384 = 288');
  });

  test('switching the EP size away and back restores the default readouts exactly', async ({ page }) => {
    await setEp(page, 1);
    await setEp(page, 72);
    await setEp(page, 16);
    await expectReadouts(page, { 'tokens-per-expert': '16', weights: '54.1 GB', free: '234 GB' });
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK);
  });
});
