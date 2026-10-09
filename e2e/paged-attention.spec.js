import { test, expect } from '@playwright/test';
import { registerLessonContract } from './lesson-helpers.js';
import { CAPTIONS } from '../tests/paged-attention-expected.js';

const URL = '/serving/#paged-attention';
registerLessonContract({ name: 'paged-attention', url: URL, captions: CAPTIONS, factRows: 12, returnHash: 'paged-attention' });

const readout = (page, name) => page.locator(`[data-readout="${name}"]`);
const BLOCK_SIZE_STOP = { 2: 0, 4: 1, 8: 2, 16: 3 };
const setBlock = (page, b) => page.locator('#blockSize').fill(String(BLOCK_SIZE_STOP[b]));
const setStep = (page, s) => page.locator('#step').fill(String(s));

test.describe('paged-attention toy', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('opens on step 3, block size 4, with the exact numbers', async ({ page }) => {
    await expect(readout(page, 'before-useful')).toHaveText('62.5%');
    await expect(readout(page, 'before-wasted')).toHaveText('37.5%');
    await expect(readout(page, 'after-useful')).toHaveText('66.7%');
    await expect(readout(page, 'after-wasted')).toHaveText('8.3%');
    await expect(readout(page, 'after-free')).toHaveText('25.0%');
    await expect(readout(page, 'after-blocks')).toHaveText('9 of 12 blocks');
    await expect(readout(page, 'before-average')).toHaveText('34.2%');
    await expect(readout(page, 'after-average')).toHaveText('8.0%');
    await expect(readout(page, 'table-3')).toHaveText('2');
    await expect(readout(page, 'blocks-saved')).toHaveText('0');
  });

  test('try this 1: scrubbing Time, before wastes 52.1% at step 0 and D waits; after peaks at 16.7%', async ({ page }) => {
    await setStep(page, 0);
    await expect(readout(page, 'before-wasted')).toHaveText('52.1%');
    await setStep(page, 2);
    await expect(readout(page, 'before-D-status')).toHaveText('waiting');
    await expect(readout(page, 'before-d-start')).toHaveText('step 3');
    await expect(readout(page, 'before-d-finish')).toHaveText('step 6');
    await expect(readout(page, 'after-d-start')).toHaveText('step 1');
    await expect(readout(page, 'after-d-finish')).toHaveText('step 4');
    await setStep(page, 1);
    await expect(readout(page, 'after-wasted')).toHaveText('16.7%');
  });

  test('try this 2: block size 16 is the before lane; block size 2 averages 2.7%', async ({ page }) => {
    await setBlock(page, 16);
    await setStep(page, 0);
    await expect(readout(page, 'after-wasted')).toHaveText('52.1%');
    await expect(readout(page, 'after-average')).toHaveText('34.2%');
    await setBlock(page, 2);
    await expect(readout(page, 'after-average')).toHaveText('2.7%');
  });

  test('try this 3: the shared system prompt saves 3 blocks at block size 4, none at 8', async ({ page }) => {
    await setStep(page, 1);
    await page.locator('#sharedPrefix').click();
    await expect(readout(page, 'after-blocks')).toHaveText('7 of 12 blocks');
    await expect(readout(page, 'blocks-saved')).toHaveText('3');
    await setBlock(page, 8);
    await expect(readout(page, 'blocks-saved')).toHaveText('0');
  });

  test('following D before it arrives says it holds no blocks; following C shows its table', async ({ page }) => {
    await setStep(page, 0);
    await page.locator('#follow [data-value="D"]').click();
    await expect(readout(page, 'follow-empty')).toContainText('D holds no blocks at step 0');
    await page.locator('#follow [data-value="C"]').click();
    await expect(readout(page, 'table-2')).toHaveText('6');
  });

  test('scale it up: Llama-3.1-70B reserves 42.9 GB, 53.7% of an H100 (80 GB nominal)', async ({ page }) => {
    await page.locator('#model [data-value="llama"]').click();
    await expect(readout(page, 'scale-per-token')).toHaveText('327,680 B');
    await expect(readout(page, 'scale-block')).toHaveText('5.24 MB');
    await expect(readout(page, 'scale-reserve')).toHaveText('42.9 GB');
    await expect(readout(page, 'scale-share')).toHaveText('53.7%');
    await expect(page.locator('[data-section="toy"]')).toContainText('80 GB nominal');
    await page.locator('#model [data-value="v3"]').click();
    await expect(readout(page, 'scale-block')).toHaveText('1.12 MB');
    await expect(readout(page, 'scale-reserve')).toHaveText('9.21 GB');
    await page.locator('#model [data-value="gpt3"]').click();
    await expect(readout(page, 'scale-block')).toHaveText('75.5 MB');
    await expect(readout(page, 'scale-reserve')).toHaveText('9.66 GB');
    await page.locator('#model [data-value="toy"]').click();
    await expect(readout(page, 'scale-empty')).toBeVisible();
  });

  test('no control setting runs the pool dry: every block size at every step renders', async ({ page }) => {
    for (const b of [2, 4, 8, 16]) {
      await setBlock(page, b);
      for (const s of [0, 3, 6]) {
        await setStep(page, s);
        await expect(readout(page, 'after-blocks')).toBeVisible();
      }
    }
  });
});
