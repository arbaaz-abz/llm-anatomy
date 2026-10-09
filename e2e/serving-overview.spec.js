import { test, expect } from '@playwright/test';
import { registerLessonContract } from './lesson-helpers.js';
import { collectConsoleErrors } from './helpers.js';
import { CAPTIONS, CHECK_WORK, TRY_THIS } from '../tests/serving-overview-expected.js';
import { stepTime, RUNNING_EXAMPLE } from '../math/serving.js';
import { formatDuration } from '../math/core.js';

const URL = '/serving/#serving-overview';

registerLessonContract({ name: 'serving-overview', url: URL, captions: CAPTIONS, factRows: 5, returnHash: 'serving-overview' });

const readout = (page, name) => page.locator(`[data-section="toy"] [data-readout="${name}"]`);
// Snapped sliders move by stop index, not by value.
const PROMPT = Object.freeze({ 3: 0, 2000: 3, 20000: 5 });
const OUTPUT = Object.freeze({ 50: 1, 500: 3 });
const setPrompt = (page, n) => page.locator('#prompt').fill(String(PROMPT[n]));
const setOutput = (page, n) => page.locator('#output').fill(String(OUTPUT[n]));
const setQueue = (page, s) => page.locator('#queue').fill(String(s));
const speed = (page, id) => page.locator(`#decodeRate [data-value="${id}"]`);

async function expectReadouts(page, expected) {
  for (const [name, text] of Object.entries(expected)) await expect(readout(page, name), name).toHaveText(text);
}

test.describe('serving-overview toy: where does the time go?', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('opens on the storyboard default (2,000-token prompt, 500-token answer, no queue, alone) with its exact numbers', async ({ page }) => {
    const errors = collectConsoleErrors(page);
    await expectReadouts(page, { prefill: '141 ms', ttft: '141 ms', tpot: '14.7 ms', total: '7.49 s', 'decode-share': '98.1%' });
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK);
    await expect(readout(page, 'timeline-width')).toHaveText('full width = 7.49 s (the whole request)');
    await expect(speed(page, 'alone')).toHaveAttribute('aria-pressed', 'true');
    await expect(speed(page, 'alone')).toHaveText('67.9 tok/s, alone on the GPU');
    await expect(speed(page, 'shared')).toHaveText('33.7 tok/s, sharing with 104 others');
    await expect(page.locator('[data-section="toy"]')).toContainText('141 GB (nominal)');
    // Your request is drawn in the semantic prefill / decode colors: no letter, no --req hue.
    const bar = page.locator('[data-section="toy"] .g-request');
    await expect(bar).toHaveCount(1);
    await expect(bar).not.toHaveAttribute('data-req', /.+/);
    expect(errors).toEqual([]);
  });

  test('the try-this list closes the toy, verbatim', async ({ page }) => {
    await expect(page.locator('[data-section="toy"] ol.try-this > li')).toHaveText(TRY_THIS);
  });

  test('try this 1: the prompt sets the wait for the first token; the answer\'s length sets the total', async ({ page }) => {
    await setPrompt(page, 20000);
    await expectReadouts(page, { ttft: '1.41 s', total: '8.76 s', 'decode-share': '83.9%' });
    await setPrompt(page, 2000);
    await expectReadouts(page, { ttft: '141 ms', total: '7.49 s', 'decode-share': '98.1%' });
    await setOutput(page, 50);
    await expectReadouts(page, { ttft: '141 ms', total: '863 ms', 'decode-share': '83.6%' });
  });

  test('try this 2: TTFT and TPOT are separate dials', async ({ page }) => {
    await speed(page, 'shared').click();
    await expect(speed(page, 'shared')).toHaveAttribute('aria-pressed', 'true');
    await expectReadouts(page, { ttft: '141 ms', tpot: '29.6 ms', total: '14.9 s' });
  });

  test('try this 3: a queue only delays the start', async ({ page }) => {
    await speed(page, 'shared').click();
    await setQueue(page, 2);
    await expectReadouts(page, { ttft: '2.14 s', total: '16.9 s', tpot: '29.6 ms' });
    await expect(readout(page, 'check-work')).toContainText('TTFT = queue + prefill = 2 s + 141 ms = 2.14 s');
  });

  test('try this 4: even a tiny prompt costs one full read of the weights', async ({ page }) => {
    await setPrompt(page, 3);
    await expectReadouts(page, { prefill: '14.6 ms', ttft: '14.6 ms' });
    await expect(page.locator('[data-section="toy"] [data-readout="prefill"]').locator('xpath=ancestor::tr')).toContainText('here: the weight read');
  });

  test('the queue slider moves TTFT by exactly the queue seconds, and never moves TPOT', async ({ page }) => {
    for (const prompt of [2000, 3]) {
      await setPrompt(page, prompt);
      const prefillS = stepTime({ ...RUNNING_EXAMPLE, tokens: prompt, seqs: 0, context: 0 }).timeS;
      for (const q of [0, 0.5, 2, 5]) {
        await setQueue(page, q);
        await expectReadouts(page, { ttft: formatDuration(q + prefillS), prefill: formatDuration(prefillS), tpot: '14.7 ms' });
      }
    }
  });
});
