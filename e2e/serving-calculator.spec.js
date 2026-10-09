import { test, expect } from '@playwright/test';
import { registerLessonContract } from './lesson-helpers.js';
import { collectConsoleErrors } from './helpers.js';
import { CAPTIONS, CHECK_WORK, TRY_THIS } from '../tests/serving-calculator-expected.js';

const URL = '/serving/#serving-calculator';

registerLessonContract({ name: 'serving-calculator', url: URL, captions: CAPTIONS, factRows: 10, returnHash: 'serving-calculator' });

const readout = (page, name) => page.locator(`[data-section="toy"] [data-readout="${name}"]`);
const chip = (page, id, value) => page.locator(`#${id} [data-value="${value}"]`);
const USER_INDEX = { 1: 0, 256: 4, target: 8 };

async function expectReadouts(page, expected) {
  for (const [name, text] of Object.entries(expected)) await expect(readout(page, name), name).toHaveText(text);
}

const DEFAULTS = {
  'weights-total': '865 GB', 'gpus-min': '4', hbm: '288 GB nominal', 'weights-per-gpu': '54.1 GB', 'free-per-gpu': '234 GB', users: '1,889', 'step-time': '37 ms',
  bound: 'compute-bound', 'tok-user': '27 tok/s', 'tok-gpu': '51,020 tok/s', 'target-users': '1,889', 'target-limit': 'compute', 'cost-floor': '$0.0144',
  'cost-gb300': '$0.119', 'cost-gb200': '$0.28', 'cost-ratio': '1×', 'kv-per-user': '36.9 MB (low) · 111 MB (high)', 'users-fit': '6,345 (low) · 2,115 (high)',
};

test.describe('serving-calculator toy: size a deployment', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('opens on the storyboard default with its exact numbers and "Check my work"', async ({ page }) => {
    const errors = collectConsoleErrors(page);
    await expectReadouts(page, DEFAULTS);
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK);
    await expect(readout(page, 'conditions')).toContainText('288 GB nominal per GPU');
    await expect(chip(page, 'weights', 'nvfp4')).toBeEnabled();
    expect(errors).toEqual([]);
  });

  test('the try-this list closes the toy, verbatim', async ({ page }) => {
    await expect(page.locator('[data-section="toy"] ol.try-this > li')).toHaveText(TRY_THIS);
  });

  test('try this 1: context 8K → 128K → 1M moves users, limit and tok/s per GPU; high KV; GB200', async ({ page }) => {
    await chip(page, 'context', '139264').click();
    await expectReadouts(page, { 'users-fit': '419 (low) · 139 (high)', 'target-limit': 'memory capacity', users: '419', 'tok-gpu': '11,433 tok/s' });
    await chip(page, 'context', '1000000').click();
    await expectReadouts(page, { 'users-fit': '58 (low) · 19 (high)', users: '58', 'tok-gpu': '1,618 tok/s', 'tok-user': '27.9 tok/s' });
    await chip(page, 'kvEnd', 'high').click();
    await expectReadouts(page, { users: '19', 'step-time': '35.3 ms' });
    await chip(page, 'kvEnd', 'low').click();
    await chip(page, 'hw', 'gb200-nvl72').click();
    await expectReadouts(page, { 'users-fit': '32 (low) · 10 (high)', hbm: '186 GB nominal (the rack total over 72 GPUs)' });
  });

  test('try this 2: one user is 148 tok/s per user and per GPU', async ({ page }) => {
    await page.locator('#users').fill(String(USER_INDEX[1]));
    await expectReadouts(page, { users: '1', 'tok-user': '148 tok/s', 'tok-gpu': '148 tok/s', 'step-time': '6.76 ms', bound: 'memory-bound' });
  });

  test('try this 3: the input side', async ({ page }) => {
    await expectReadouts(page, { 'prefill-tps': '51,020 tok/s', 'cost-ratio': '1×' });
    await expect(readout(page, 'note-ratio')).toContainText('output as cheap as input');
    await chip(page, 'hit', '0.563').click();
    await expectReadouts(page, { 'cost-ratio': '2.29×', 'prefill-tps': '116,752 tok/s' });
    await chip(page, 'hit', '0').click();
    await chip(page, 'context', '139264').click();
    await expectReadouts(page, { 'cost-ratio': '4.46×' });
    await chip(page, 'context', '1000000').click();
    await expectReadouts(page, { 'cost-ratio': '31.5×' });
  });

  test('try this 4: MTP pays where decode is memory-bound', async ({ page }) => {
    await expect(readout(page, 'mtp-speedup')).toHaveCount(0);
    await page.locator('#mtp').click();
    await expectReadouts(page, { 'mtp-speedup': '0.9 of the plain speed' });
    await page.locator('#users').fill(String(USER_INDEX[256]));
    await expectReadouts(page, { 'mtp-speedup': '1.48×' });
    await page.locator('#users').fill(String(USER_INDEX.target));
    await chip(page, 'context', '139264').click();
    await expectReadouts(page, { 'mtp-speedup': '1.73×', 'mtp-record': /\+87% \(1\.87×\)/ });
  });

  test('too few GPUs for the weights prints "does not fit" and 0 users (Review Focus 2)', async ({ page }) => {
    await page.locator('#gpus').fill('0');
    await expectReadouts(page, { 'free-per-gpu': 'does not fit', users: '0', 'step-time': 'does not fit', 'target-limit': 'does not fit' });
    await expect(readout(page, 'check-work')).toContainText('= −577 GB free: does not fit');
  });

  test('model V4-Pro → Llama → V4-Pro restores the default readouts exactly (Review Focus 5)', async ({ page }) => {
    await chip(page, 'model', 'llama-3.1-70b').click();
    await expectReadouts(page, { users: '23', 'step-time': '29.1 ms', 'tok-gpu': '789 tok/s', 'cost-floor': '$0.932', hbm: '141 GB nominal' });
    await expect(chip(page, 'weights', 'shipped')).toHaveCount(0);
    await expect(chip(page, 'context', '1000000')).toHaveCount(0);
    await chip(page, 'model', 'deepseek-v4-pro').click();
    await expectReadouts(page, DEFAULTS);
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK);
  });

  test('the H200 disables NVFP4 with its note and moves the format to FP8 (Review Focus 5)', async ({ page }) => {
    await chip(page, 'weights', 'nvfp4').click();
    await chip(page, 'hw', 'h200').click();
    await expect(chip(page, 'weights', 'nvfp4')).toBeDisabled();
    await expect(page.locator('#weights [data-choice-note]')).toHaveText('NVFP4: no FP4 figure in data');
    await expect(chip(page, 'weights', 'fp8')).toHaveAttribute('aria-pressed', 'true');
  });
});
