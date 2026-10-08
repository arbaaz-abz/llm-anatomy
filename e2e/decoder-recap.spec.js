import { test, expect } from '@playwright/test';
import { registerLessonContract } from './lesson-helpers.js';
import { collectConsoleErrors } from './helpers.js';
import { CAPTIONS, FRAME5_EXACT, STAND_IN, RESIDUAL_NOTE } from '../tests/decoder-recap-expected.js';

const URL = '/architecture/#decoder-recap';
const STEPPER = '[data-section="animation"] .stepper';

registerLessonContract({ name: 'decoder-recap', url: URL, captions: CAPTIONS, factRows: 12, returnHash: 'decoder-recap' });

const readout = (page, name) => page.locator(`[data-readout="${name}"]`);
const pick = (page, control, value) => page.locator(`#${control} [data-value="${value}"]`).click();
const goTo = (page, index) => page.locator(`${STEPPER} input[type="range"]`).fill(String(index));

test.describe('decoder-recap animation: page text under the stage', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('the stand-in line is always visible, and frames 5, 7 and 10 print their exact page text', async ({ page }) => {
    await expect(page.locator('.stand-in')).toHaveText(STAND_IN);
    await goTo(page, 4);
    await expect(page.locator('.below-stage')).toContainText(FRAME5_EXACT);
    await goTo(page, 6);
    await expect(page.locator('.below-stage')).toContainText('This row is the hero row of Attention, step by step, which computes it step by step.');
    await expect(page.locator('.below-stage')).toContainText('[−0.792, 1.841, 0.396]');
    await goTo(page, 9);
    await expect(page.locator('.below-stage')).toContainText(RESIDUAL_NOTE);
  });
});

test.describe('decoder-recap toy: modernize GPT-3, one switch at a time', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('opens on GPT-3: its count, its cache, its position limit, and every switch off', async ({ page }) => {
    await expect(readout(page, 'total')).toHaveText('174,604,259,328');
    await expect(readout(page, 'active')).toHaveText('174,604,259,328');
    await expect(readout(page, 'change')).toHaveText('0');
    await expect(readout(page, 'cache')).toHaveText('4,718,592 B');
    await expect(readout(page, 'longest')).toHaveText('2,048 (table size)');
    await expect(readout(page, 'part-mlp')).toHaveText('115,970,015,232');
    await expect(readout(page, 'part-positional')).toHaveText('25,165,824');
    await expect(page.locator('#norm [aria-pressed="true"]')).toHaveText('LayerNorm');
    await expect(page.locator('#kvHeads [aria-pressed="true"]')).toHaveText('96 (GPT-3)');
  });

  test('try this 1: norm and MLP change how the numbers flow, not how many there are', async ({ page }) => {
    await pick(page, 'norm', 'rmsnorm');
    await expect(readout(page, 'total')).toHaveText('174,601,887,744');
    await expect(readout(page, 'change')).toHaveText('−2,371,584');
    await expect(readout(page, 'part-norms')).toHaveText('2,371,584');
    await pick(page, 'norm', 'layernorm');
    await pick(page, 'mlp', 'swiglu');
    await expect(readout(page, 'part-mlp')).toHaveText('115,971,588,096');
    await expect(readout(page, 'part-mlp-delta')).toHaveText('+1,572,864');
  });

  test('try this 2: RoPE removes the position table; 8 KV heads cut the cache 12×', async ({ page }) => {
    await pick(page, 'position', 'rope');
    await expect(readout(page, 'part-positional')).toHaveText('0');
    await expect(readout(page, 'part-positional-delta')).toHaveText('−25,165,824');
    await expect(readout(page, 'longest')).toContainText('set by training');
    await pick(page, 'kvHeads', '8');
    await expect(readout(page, 'cache')).toHaveText('393,216 B');
    await expect(page.locator('[data-section="toy"] tr:has([data-readout="cache"]) .ro-sub')).toHaveText('393 kB (12× less than GPT-3)');
    await page.locator('#presets [data-value="2026"]').click();
    await pick(page, 'experts', 'dense');
    await expect(readout(page, 'total')).toHaveText('147,990,994,944');
  });

  test('try this 3: experts multiply what is stored, not what a token uses; the router is the only active growth', async ({ page }) => {
    await page.locator('#presets [data-value="2026"]').click();
    await pick(page, 'experts', 'dense');
    await expect(readout(page, 'total')).toHaveText('147,990,994,944');
    await expect(readout(page, 'active')).toHaveText('147,990,994,944');
    await pick(page, 'experts', 'moe');
    await expect(readout(page, 'total')).toHaveText('959,815,311,360');
    await expect(readout(page, 'active')).toHaveText('148,066,492,416');
    await expect(readout(page, 'part-router')).toHaveText('75,497,472');
    await expect(readout(page, 'part-experts')).toHaveText('927,712,935,936');
    await page.locator('#presets [data-value="gpt3"]').click();
    await expect(readout(page, 'total')).toHaveText('174,604,259,328');
    await expect(page.locator('#experts [aria-pressed="true"]')).toHaveText('dense');
    await expect(page.locator('#kvHeads [aria-pressed="true"]')).toHaveText('96 (GPT-3)');
  });

  test('experts need SwiGLU: switching them on flips the MLP, and going back to GELU turns them off', async ({ page }) => {
    const errors = collectConsoleErrors(page);
    await pick(page, 'experts', 'moe');
    await expect(page.locator('#mlp [aria-pressed="true"]')).toHaveText('SwiGLU, 8/3 × d');
    await pick(page, 'mlp', 'gelu');
    await expect(page.locator('#experts [aria-pressed="true"]')).toHaveText('dense');
    expect(errors).toEqual([]);
  });

  test('the active definition is printed once, and 1 KV head cuts the cache 96×', async ({ page }) => {
    await expect(page.locator('[data-section="toy"]')).toContainText('Active = the parameters multiplied for one token');
    await pick(page, 'kvHeads', '1');
    await expect(readout(page, 'cache')).toHaveText('49,152 B');
  });
});
