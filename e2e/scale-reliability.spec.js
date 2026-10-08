import { test, expect } from '@playwright/test';
import { registerLessonContract } from './lesson-helpers.js';
import { collectConsoleErrors } from './helpers.js';
import { CAPTIONS, DEFAULT_READOUTS, LOSS_SPLIT, RECORD_LLAMA, RECORD_LLAMA_MFU_38, RECORD_DEEPSEEK, NO_RECORD, INVALID, DEEPSEEK_NOTE } from '../tests/scale-reliability-expected.js';

const URL = '/training/#scale-reliability';

registerLessonContract({ name: 'scale-reliability', url: URL, captions: CAPTIONS, factRows: 13, returnHash: 'scale-reliability' });

const readout = (page, name) => page.locator(`[data-readout="${name}"]`);
// The sliders snap to stops; fill() takes a stop's index (numbers.js STOPS order).
const STOP = Object.freeze({ gpus: { 32768: 7, 100000: 9, 200000: 11 }, interval: { 5: 2, 60: 8 }, save: { 15: 2 }, restart: { 30: 7 } });
const expectReadouts = async (page, expected) => {
  for (const [name, text] of Object.entries(expected)) await expect(readout(page, name), name).toHaveText(text);
};

test.describe('scale-reliability toy: plan a training run', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('opens on the Llama 3.1 405B preset with the storyboard\'s numbers and its record line', async ({ page }) => {
    const errors = collectConsoleErrors(page);
    await expectReadouts(page, DEFAULT_READOUTS);
    await expect(page.locator('[data-readout="loss"] + .ro-sub')).toHaveText(LOSS_SPLIT);
    await expect(readout(page, 'record')).toHaveText(RECORD_LLAMA);
    await expect(readout(page, 'invalid')).toBeHidden();
    await expect(page.locator('#interval')).toBeHidden();
    await expect(page.locator('[data-section="toy"] .g-share .g-hatch').first()).toBeVisible();
    expect(errors).toEqual([]);
  });

  test('try this 1: MFU 40% → 38% moves the run from 29.24M to 30.78M GPU-hours, next to the card\'s 30.84M', async ({ page }) => {
    await page.locator('#mfu').fill('38');
    await expect(readout(page, 'gpu-hours')).toHaveText('30.78M');
    await expect(readout(page, 'days')).toHaveText('78.3 days');
    await expect(readout(page, 'record')).toHaveText(RECORD_LLAMA_MFU_38);
  });

  test('try this 2: more GPUs finish sooner and pay a bigger failure tax', async ({ page }) => {
    await page.locator('#gpus').fill(String(STOP.gpus[32768]));
    await expectReadouts(page, { days: '39.2 days', loss: '13.6%', 'gpu-hours': '30.81M' });
    await expect(readout(page, 'record')).toHaveText(NO_RECORD);
    await page.locator('#gpus').fill(String(STOP.gpus[100000]));
    await expectReadouts(page, { mtbf: '30.4 min', interval: '5.51 min', days: '15.4 days', loss: '28.0%', 'gpu-hours': '36.97M' });
  });

  test('try this 3: the best checkpoint interval, and faster saves on a bigger cluster', async ({ page }) => {
    await page.locator('#interval-auto').click();
    await expect(page.locator('#interval')).toBeVisible();
    await page.locator('#interval').fill(String(STOP.interval[5]));
    await expectReadouts(page, { interval: '5 min', loss: '13.0%' });
    await page.locator('#interval').fill(String(STOP.interval[60]));
    await expectReadouts(page, { interval: '1 h', loss: '18.6%' });
    await page.locator('#interval-auto').click();
    await expectReadouts(page, { interval: '13.6 min', loss: '9.0%' });
    await page.locator('#save').fill(String(STOP.save[15]));
    await expectReadouts(page, { interval: '9.63 min', loss: '6.8%' });
    await page.locator('#gpus').fill(String(STOP.gpus[100000]));
    await expectReadouts(page, { loss: '22.7%' });
  });

  test('the DeepSeek-V3 preset reproduces its record by construction, and says what FP8 does to the percentage', async ({ page }) => {
    await expect(page.locator('#preset [aria-pressed]')).toHaveCount(0);
    await page.locator('#preset [data-value="deepseek"]').click();
    await expectReadouts(page, { 'gpu-hours': '2.664M', days: '54.2 days', 'useful-hours': '2.589M', loss: '2.8%' });
    await expect(readout(page, 'record')).toHaveText(RECORD_DEEPSEEK);
    await expect(readout(page, 'preset-note')).toHaveText(DEEPSEEK_NOTE);
    await page.locator('#preset [data-value="llama"]').click();
    await expectReadouts(page, DEFAULT_READOUTS);
    await expect(readout(page, 'preset-note')).toBeHidden();
  });

  test('outside the loss formula\'s range the readouts are replaced by the stated line (README lesson 22)', async ({ page }) => {
    await page.locator('#gpus').fill(String(STOP.gpus[200000]));
    await page.locator('#restart').fill(String(STOP.restart[30]));
    await expect(readout(page, 'invalid')).toHaveText(INVALID);
    await expectReadouts(page, { 'gpu-hours': '—', days: '—', cost: '—', 'run-average-mfu': '—' });
    await expect(page.locator('[data-section="toy"] .g-share')).toHaveCount(0);
  });
});
