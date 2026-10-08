import { test, expect } from '@playwright/test';
import { registerLessonContract } from './lesson-helpers.js';
import { collectConsoleErrors } from './helpers.js';
import { CAPTIONS, CHECK_WORK, COMPUTE_4, TRY_THIS } from '../tests/gpu-primer-expected.js';

const URL = '/training/#gpu-primer';

registerLessonContract({ name: 'gpu-primer', url: URL, captions: CAPTIONS, factRows: 11, returnHash: 'gpu-primer' });

const readout = (page, name) => page.locator(`[data-section="toy"] [data-readout="${name}"]`);
// The tokens slider snaps to powers of two: index i is 2^i tokens.
const TOKEN_INDEX = Object.freeze({ 4: 2, 64: 6, 256: 8, 512: 9, 4096: 12 });
const setTokens = (page, n) => page.locator('#tokens').fill(String(TOKEN_INDEX[n]));
const chip = (page, id) => page.locator(`#chip [data-value="${id}"]`);
const fmt = (page, id) => page.locator(`#fmt [data-value="${id}"]`);

async function expectReadouts(page, expected) {
  for (const [name, text] of Object.entries(expected)) await expect(readout(page, name), name).toHaveText(text);
}

test.describe('gpu-primer toy: where does this multiply sit on the roof?', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('opens on the storyboard default (H100, BF16, 4 tokens) with its exact numbers and "Check my work"', async ({ page }) => {
    const errors = collectConsoleErrors(page);
    await expectReadouts(page, {
      flops: '537M', bytes: '134 MB', intensity: '4.0', ridge: '295', 'tokens-needed': '318.2', verdict: 'memory-bound',
      attainable: '13.4 TFLOPS', 'peak-share': '1.35%', 'memory-time': '40.1 µs', 'compute-time': COMPUTE_4, 'time-ratio': '73.9×',
    });
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK);
    await expect(readout(page, 'lanes-width')).toHaveText('lanes share one time axis; full width = 40.1 µs');
    await expect(page.locator('[data-section="toy"] .g-roofline .g-marker')).toHaveCount(2);
    await expect(page.locator('[data-section="toy"] .g-roofline .g-select')).toHaveCount(1);
    await expect(fmt(page, 'fp4')).toBeDisabled();
    await expect(page.locator('#fmt [data-choice-note]')).toHaveText('FP4: no FP4 figure in data');
    expect(errors).toEqual([]);
  });

  test('the try-this list closes the toy, verbatim', async ({ page }) => {
    await expect(page.locator('[data-section="toy"] ol.try-this > li')).toHaveText(TRY_THIS);
  });

  test('try this 1: in BF16 the intensity is about the token count; the H100 crosses at 318.2 tokens', async ({ page }) => {
    await setTokens(page, 64);
    await expectReadouts(page, { 'peak-share': '21.34%', verdict: 'memory-bound', intensity: '63.0', 'tokens-needed': '318.2' });
    await setTokens(page, 256);
    await expectReadouts(page, { 'peak-share': '81.61%', verdict: 'memory-bound', intensity: '240.9' });
    await setTokens(page, 512);
    await expectReadouts(page, { 'peak-share': '100.00%', verdict: 'compute-bound', 'tokens-needed': '318.2' });
    // The token presets are actions, never a selection: none claims to be chosen while the slider sits between them.
    await expect(page.locator('#tokens-preset [aria-pressed]')).toHaveCount(0);
    await page.locator('#tokens-preset [data-value="4096"]').click();
    await expect(page.locator('#tokens')).toHaveValue(String(TOKEN_INDEX[4096]));
    await expectReadouts(page, { intensity: '2,048.0', 'compute-time': '556 µs', 'memory-time': '80.1 µs', 'time-ratio': '6.94×' });
    await expect(readout(page, 'lanes-width')).toHaveText('lanes share one time axis; full width = 556 µs');
  });

  test('try this 2: FP8 on an H100 doubles both sides; the batch to cross stays 318', async ({ page }) => {
    await page.locator('#tokens-preset [data-value="256"]').click();
    await expectReadouts(page, { intensity: '240.9', ridge: '295', 'memory-time': '42.6 µs', 'tokens-needed': '318.2' });
    await fmt(page, 'fp8').click();
    await expectReadouts(page, { intensity: '481.9', ridge: '591', verdict: 'memory-bound', 'memory-time': '21.3 µs', 'tokens-needed': '318.3', bytes: '71.3 MB' });
    await expect(readout(page, 'check-work')).toContainText('bytes     = 1 × (256 × 8,192 + 8,192 × 8,192 + 256 × 8,192) = 71,303,168');
  });

  test('try this 3: the BF16 ridge is flat across generations; FP4 needs more tokens on a B300, barely more on a B200', async ({ page }) => {
    await chip(page, 'b300').click();
    await setTokens(page, 512);
    await expectReadouts(page, { verdict: 'compute-bound', intensity: '455.1', ridge: '313', 'tokens-needed': '338.3' });
    await fmt(page, 'fp4').click();
    await expectReadouts(page, { verdict: 'memory-bound', intensity: '1,618.2', ridge: '1,875', 'tokens-needed': '605.3' });
    await chip(page, 'h100').click();
    await expect(fmt(page, 'bf16')).toHaveAttribute('aria-pressed', 'true');
    await expect(readout(page, 'ridge')).toHaveText('295');
    await chip(page, 'b200').click();
    await expect(readout(page, 'ridge')).toHaveText('281');
    await chip(page, 'b300').click();
    await expect(readout(page, 'ridge')).toHaveText('313');
    await chip(page, 'h200').click();
    await expectReadouts(page, { ridge: '206', 'tokens-needed': '217.0' });
    await chip(page, 'b200').click();
    await expect(readout(page, 'tokens-needed')).toHaveText('302.0');
    await fmt(page, 'fp4').click();
    await expect(readout(page, 'tokens-needed')).toHaveText('342.9');
  });

  test('Rubin is FP4 only: BF16 and FP8 disabled with the reason; ridge and crossing print both bandwidth ends', async ({ page }) => {
    const errors = collectConsoleErrors(page);
    await chip(page, 'rubin').click();
    await expect(fmt(page, 'fp4')).toHaveAttribute('aria-pressed', 'true');
    await expect(fmt(page, 'bf16')).toBeDisabled();
    await expect(fmt(page, 'fp8')).toBeDisabled();
    await expect(page.locator('#fmt [data-choice-note]')).toHaveText('BF16, FP8: no settled BF16/FP8 figure');
    await expectReadouts(page, { ridge: '1,591–1,823', 'tokens-needed': '502.3–586.1', verdict: 'memory-bound' });
    await expect(page.locator('[data-section="toy"]')).toContainText('bandwidth: sources conflict, 19.2 or 22 TB/s');
    await setTokens(page, 512);
    await expect(readout(page, 'verdict')).toHaveText('compute-bound at 22 TB/s, memory-bound at 19.2 TB/s');
    await chip(page, 'mi355x').click(); // the MI355X has FP4 too (MXFP4), so the format stays
    await expect(fmt(page, 'fp4')).toHaveAttribute('aria-pressed', 'true');
    await expect(fmt(page, 'bf16')).toBeEnabled();
    await expect(page.locator('#fmt [data-choice-note]')).toBeHidden();
    await expectReadouts(page, { ridge: '1,250', 'tokens-needed': '361.3' });
    await expect(readout(page, 'check-work')).toContainText('bytes     = 0.53125 × (');
    expect(errors).toEqual([]);
  });
});
