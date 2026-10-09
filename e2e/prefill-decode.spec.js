import { test, expect } from '@playwright/test';
import { registerLessonContract } from './lesson-helpers.js';
import { collectConsoleErrors } from './helpers.js';
import { CAPTIONS, CHECK_WORK, CHECK_WORK_DOES_NOT_FIT, DOES_NOT_FIT, TRY_THIS } from '../tests/prefill-decode-expected.js';

const URL = '/serving/#prefill-decode';

registerLessonContract({ name: 'prefill-decode', url: URL, captions: CAPTIONS, factRows: 5, returnHash: 'prefill-decode' });

const readout = (page, name) => page.locator(`[data-section="toy"] [data-readout="${name}"]`);
const choose = (page, id, value) => page.locator(`#${id} [data-value="${value}"]`).click();
// Snapped sliders move by stop index, not by value.
const stop = (page, id, index) => page.locator(`#${id}`).fill(String(index));
const PROMPT = Object.freeze({ 1: 0, 217: 3, 512: 4, 1000: 5, 8192: 7 });
const CONTEXT = Object.freeze({ 2048: 1, 8192: 2, 32768: 3, 131072: 4 });
// Users stops at 2,048 tokens of context on an H200 in FP8: 1, 2, 4, 8, 16, 32, 64, 105 (max that fits).
const USERS = Object.freeze({ 1: 0, 8: 3, 64: 6, max: 7 });

async function expectReadouts(page, expected) {
  for (const [name, text] of Object.entries(expected)) await expect(readout(page, name), name).toHaveText(text);
}

test.describe('prefill-decode toy: step-time calculator', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('opens on the storyboard default (decode, H200, FP8, 8 users, 2,048 tokens) with its exact numbers', async ({ page }) => {
    const errors = collectConsoleErrors(page);
    await expectReadouts(page, {
      flops: '1.12 TFLOP', 'weights-bytes': '70 GB', 'act-bytes': '137 MB', 'kv-bytes': '5.37 GB', bytes: '75.5 GB',
      'math-time': '566 µs', 'read-time': '15.7 ms', 'step-time': '15.7 ms', bound: 'memory-bound',
      intensity: '14.8', ridge: '412.3', 'tokens-needed': '217.1', hbm: '141 GB nominal', peak: '1,979 TFLOP/s', bandwidth: '4.8 TB/s',
      'per-user': '63.6', 'per-gpu': '509', 'max-users': '105', 'cache-per-user': '671 MB', free: '71 GB',
    });
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK);
    await expect(readout(page, 'bound').locator('.sem-text--memory')).toHaveText('memory-bound');
    await expect(page.locator('#hw [data-value="b200"]')).toHaveText('B200 · 180 GB usable (192 nominal)');
    await expect(page.locator('#users + output, #users ~ output').first()).toHaveText('8 users');
    await expect(page.locator('#promptTokens')).toBeHidden();
    await expect(page.locator('[data-section="toy"] .g-stepbar')).toHaveCount(1);
    await expect(page.locator('[data-section="toy"] .g-plot .g-select')).toHaveCount(1);
    expect(errors).toEqual([]);
  });

  test('the try-this list closes the toy, verbatim', async ({ page }) => {
    await expect(page.locator('[data-section="toy"] ol.try-this > li')).toHaveText(TRY_THIS);
  });

  test('try this 1: the crossover is the GPU\'s ratio of math to bandwidth, counted in tokens per weight read', async ({ page }) => {
    await choose(page, 'phase', 'prefill');
    await expect(page.locator('#users')).toBeHidden();
    await stop(page, 'promptTokens', PROMPT[1]);
    await expectReadouts(page, { bound: 'memory-bound', 'tokens-needed': '217.1' });
    await stop(page, 'promptTokens', PROMPT[217]);
    await expectReadouts(page, { bound: 'memory-bound', 'step-time': '15.4 ms', 'math-time': '15.4 ms', 'read-time': '15.4 ms' });
    await stop(page, 'promptTokens', PROMPT[512]);
    await expect(readout(page, 'bound')).toHaveText('compute-bound');
    await stop(page, 'promptTokens', PROMPT[1000]);
    await expectReadouts(page, { 'step-time': '70.7 ms', 'read-time': '18.1 ms', ttft: '70.7 ms', 'prefill-rate': '14,136', ceiling: '14,136', intensity: '1,607.5' });
    await stop(page, 'promptTokens', PROMPT[8192]);
    await expect(readout(page, 'step-time')).toHaveText('580 ms');
    await choose(page, 'weights', 'bf16');
    await expect(readout(page, 'tokens-needed')).toHaveText('217.0');
    await choose(page, 'hw', 'b200');
    await expectReadouts(page, { 'tokens-needed': '302.0', hbm: '180 GB usable (192 nominal)' });
    await choose(page, 'weights', 'fp8');
    await expect(readout(page, 'tokens-needed')).toHaveText('302.0');
  });

  test('try this 2: batching trades each user\'s speed for the GPU\'s total', async ({ page }) => {
    const rows = [[USERS[1], '67.9', '68'], [USERS[8], '63.6', '509'], [USERS[64], '42.1', '2,694'], [USERS.max, '33.7', '3,543']];
    for (const [index, perUser, perGpu] of rows) {
      await stop(page, 'users', index);
      await expectReadouts(page, { 'per-user': perUser, 'per-gpu': perGpu });
    }
    await expect(readout(page, 'step-time')).toHaveText('29.6 ms');
    await expect(readout(page, 'intensity')).toHaveText('103.3');
  });

  test('try this 3: at long context the KV cache decides how many users fit', async ({ page }) => {
    await stop(page, 'users', USERS.max);
    await expectReadouts(page, { 'max-users': '105', 'per-gpu': '3,543' });
    const rows = [[CONTEXT[8192], '26', '890'], [CONTEXT[32768], '6', '214'], [CONTEXT[131072], '1', '42']];
    for (const [index, maxUsers, perGpu] of rows) {
      await stop(page, 'context', index);
      await expectReadouts(page, { 'max-users': maxUsers, 'per-gpu': perGpu });
    }
  });

  test('try this 4: BF16 on the H200 leaves 1 GB and one user', async ({ page }) => {
    await choose(page, 'weights', 'bf16');
    await expectReadouts(page, { free: '1 GB', 'max-users': '1', 'per-gpu': '34', 'step-time': '29.3 ms', 'weights-bytes': '140 GB', peak: '989 TFLOP/s' });
    await expect(readout(page, 'users-note')).toHaveText('Users clamped to 1, the most that fit at 2,048 tokens of context.');
  });

  test('raising context clamps users to the new max that fits, never above it (Review Focus 5)', async ({ page }) => {
    await stop(page, 'users', USERS[64]);
    await expect(readout(page, 'users-note')).toBeHidden();
    await stop(page, 'context', CONTEXT[8192]);
    const users = page.locator('#users');
    await expect(users).toHaveAttribute('max', '5'); // stops 1, 2, 4, 8, 16, 26
    await expect(users).toHaveValue('5');
    await expect(page.locator('#users ~ output').first()).toHaveText('26 users, max that fits');
    await expect(readout(page, 'users-note')).toHaveText('Users clamped to 26, the most that fit at 8,192 tokens of context.');
    await expectReadouts(page, { 'max-users': '26', 'per-gpu': '890', 'step-time': '29.2 ms' });
    await stop(page, 'context', CONTEXT[2048]);
    await expect(readout(page, 'users-note')).toBeHidden();
    await expectReadouts(page, { 'per-gpu': '2,694' });
  });

  test('BF16 weights on an H100 print "does not fit" and 0 users, never a negative count (Review Focus 2)', async ({ page }) => {
    const errors = collectConsoleErrors(page);
    await choose(page, 'hw', 'h100');
    await choose(page, 'weights', 'bf16');
    await expect(readout(page, 'fit')).toHaveText(DOES_NOT_FIT);
    await expect(readout(page, 'fit').locator('a')).toHaveAttribute('href', /serving-calculator/);
    await expectReadouts(page, { 'max-users': '0', hbm: '80 GB nominal', 'weights-bytes': '140 GB' });
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK_DOES_NOT_FIT);
    await expect(readout(page, 'step-time')).toHaveCount(0);
    await expect(page.locator('[data-section="toy"] .g-stepbar')).toHaveCount(0);
    await expect(page.locator('[data-section="toy"]')).not.toContainText(/NaN|Infinity|−\d/);
    await choose(page, 'phase', 'prefill');
    await expect(readout(page, 'fit')).toHaveText(DOES_NOT_FIT);
    await choose(page, 'weights', 'fp8');
    await expect(readout(page, 'fit')).toHaveCount(0);
    await expect(readout(page, 'step-time')).toHaveText('70.7 ms');
    expect(errors).toEqual([]);
  });

  test('no user fits when one cache outgrows the free memory: 0 users, no step', async ({ page }) => {
    await choose(page, 'weights', 'bf16');
    await stop(page, 'context', CONTEXT[8192]);
    await expectReadouts(page, { 'max-users': '0', free: '1 GB', 'cache-per-user': '2.68 GB' });
    await expect(readout(page, 'users-note')).toHaveText('No user fits: one user\'s cache at 8,192 tokens (2.68 GB) is more than the 1 GB free.');
    await expect(page.locator('#users ~ output').first()).toHaveText('0 users, none fit');
    await expect(readout(page, 'step-time')).toHaveCount(0);
    await stop(page, 'context', CONTEXT[2048]);
    await expectReadouts(page, { 'max-users': '1', 'per-gpu': '34' });
  });
});
