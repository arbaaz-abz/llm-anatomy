import { test, expect } from '@playwright/test';
import { registerLessonContract } from './lesson-helpers.js';
import { collectConsoleErrors } from './helpers.js';
import { CAPTIONS, CHECK_WORK, TRY_THIS, NO_FP4_NOTE } from '../tests/quantization-expected.js';

const URL = '/serving/#quantization';

registerLessonContract({ name: 'quantization', url: URL, captions: CAPTIONS, factRows: 9, returnHash: 'quantization' });

const readout = (page, name) => page.locator(`[data-section="toy"] [data-readout="${name}"]`);
const pick = (page, control, value) => page.locator(`#${control} [data-value="${value}"]`).click();
const choice = (page, control, value) => page.locator(`#${control} [data-value="${value}"]`);

async function expectReadouts(page, expected) {
  for (const [name, text] of Object.entries(expected)) await expect(readout(page, name), name).toHaveText(text);
}

test.describe('quantization toy: round a block, then shrink a model', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('opens on the storyboard default (INT4, block 8, outlier on; FP8 weights on an H200) with its exact numbers', async ({ page }) => {
    const errors = collectConsoleErrors(page);
    await expectReadouts(page, {
      'mean-error': '0.064', zeroed: '3', clipped: '0',
      bits: '8.00', weights: '70 GB', memory: '141 GB nominal', free: '71 GB', 'kv-user': '671 MB', users: '105', decode: '14.7 ms', prefill: '290 ms',
    });
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK);
    await expect(page.locator('[data-section="toy"] .g-nl-dot')).toHaveCount(8);
    await expect(page.locator('[data-section="toy"] .g-nl .g-select')).toHaveCount(1);
    expect(errors).toEqual([]);
  });

  test('the try-this list closes the toy, verbatim', async ({ page }) => {
    await expect(page.locator('[data-section="toy"] ol.try-this > li')).toHaveText(TRY_THIS);
  });

  test('try this 1: one outlier ruins a shared scale, and smaller blocks contain the damage', async ({ page }) => {
    await pick(page, 'outlier', 'calm');
    await expectReadouts(page, { 'mean-error': '0.024' });
    await pick(page, 'outlier', 'outlier');
    await pick(page, 'blockSize', '4');
    await expectReadouts(page, { 'mean-error': '0.032', zeroed: '1', clipped: '0' });
    await expect(readout(page, 'check-work')).toContainText('scale = max|w| / 7 = 0.47 / 7 = 0.0671');
  });

  test('try this 2: the scale\'s precision matters as much as the grid', async ({ page }) => {
    const errorOf = async (format, blockSize) => {
      await pick(page, 'format', format);
      await pick(page, 'blockSize', blockSize);
      return (await readout(page, 'mean-error').textContent());
    };
    expect([await errorOf('int4', '4'), await errorOf('mxfp4', '4'), await errorOf('nvfp4', '4')]).toEqual(['0.032', '0.062', '0.029']);
    await pick(page, 'format', 'mxfp4');
    await expectReadouts(page, { clipped: '1' });
    await expect(readout(page, 'check-work')).toContainText('7.52 (past 6, clamped to 6) → 6');
    expect([await errorOf('int4', '8'), await errorOf('mxfp4', '8'), await errorOf('nvfp4', '8')]).toEqual(['0.064', '0.073', '0.049']);
    expect([await errorOf('int4', '2'), await errorOf('mxfp4', '2'), await errorOf('nvfp4', '2')]).toEqual(['0.011', '0.054', '0.017']);
    await pick(page, 'format', 'mxfp4');
    await expectReadouts(page, { clipped: '2' });
  });

  test('try this 3: bytes speed up decode; only low-precision math speeds up prefill', async ({ page }) => {
    await pick(page, 'modelFormat', 'bf16');
    await expectReadouts(page, { weights: '140 GB', users: '1', decode: '29.3 ms', prefill: '580 ms' });
    await pick(page, 'modelFormat', 'fp8');
    await expectReadouts(page, { weights: '70 GB', users: '105', decode: '14.7 ms', prefill: '290 ms' });
    await pick(page, 'modelFormat', 'w4a16');
    await expectReadouts(page, { bits: '4.50', weights: '39.4 GB', users: '151', decode: '8.35 ms', prefill: '580 ms' });
    await pick(page, 'hw', 'b200');
    await pick(page, 'modelFormat', 'nvfp4');
    await expectReadouts(page, { memory: '180 GB usable', users: '209', decode: '5.01 ms', prefill: '63.7 ms' });
    await pick(page, 'modelFormat', 'mxfp4');
    await expectReadouts(page, { bits: '4.25', users: '212', decode: '4.73 ms', prefill: '63.7 ms' });
  });

  test('try this 4: the KV cache in FP8 doubles the users that fit', async ({ page }) => {
    await pick(page, 'kv', 'fp8');
    await expectReadouts(page, { users: '211', 'kv-user': '336 MB' });
  });

  test('the H200 disables both FP4 chips with the note; a chip change keeps a usable format and back again restores the defaults (Review Focus 5)', async ({ page }) => {
    await expect(choice(page, 'modelFormat', 'nvfp4')).toBeDisabled();
    await expect(choice(page, 'modelFormat', 'mxfp4')).toBeDisabled();
    await expect(page.locator('#modelFormat [data-choice-note]')).toHaveText(NO_FP4_NOTE);
    await pick(page, 'hw', 'b200');
    await expect(choice(page, 'modelFormat', 'nvfp4')).toBeEnabled();
    await expect(page.locator('#modelFormat [data-choice-note]')).toHaveCount(0);
    await pick(page, 'modelFormat', 'nvfp4');
    await pick(page, 'hw', 'h200');
    await expect(choice(page, 'modelFormat', 'w4a16')).toHaveAttribute('aria-pressed', 'true');
    await expectReadouts(page, { bits: '4.50', users: '151', decode: '8.35 ms', prefill: '580 ms' });
    await pick(page, 'modelFormat', 'fp8');
    await expectReadouts(page, { users: '105', decode: '14.7 ms', prefill: '290 ms' });
  });

  test('the visible notes state the toy\'s stand-ins', async ({ page }) => {
    await expect(page.locator('[data-section="toy"]')).toContainText('GPTQ and AWQ choose roundings more cleverly');
    await expect(page.locator('[data-section="toy"]')).toContainText('On a hand-sized block INT4 can match FP4\'s error');
  });
});
