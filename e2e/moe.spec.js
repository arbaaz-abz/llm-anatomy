import { test, expect } from '@playwright/test';
import { registerLessonContract } from './lesson-helpers.js';
import { CAPTIONS } from '../tests/moe-expected.js';

const URL = '/architecture/#moe';

registerLessonContract({ name: 'moe', url: URL, captions: CAPTIONS, factRows: 11, returnHash: 'moe' });

const readout = (page, name) => page.locator(`[data-section="toy"] [data-readout="${name}"]`);
const ROUTED_STOPS = [0, 2, 4, 8, 16, 32]; // the #routed slider snaps to these; fill() takes the index
const pickRouted = (page, n) => page.locator('#routed').fill(String(ROUTED_STOPS.indexOf(n)));
const GAMMA_STOPS = [0, 0.05, 0.1, 0.2];
const pickGamma = (page, g) => page.locator('#gamma').fill(String(GAMMA_STOPS.indexOf(g)));
const imbalanceAt = async (page, step) => {
  await page.locator('#step').fill(String(step));
  return (await readout(page, 'imbalance').textContent()).trim();
};

test.describe('moe toy: route, count, rebalance (panel A)', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('opens on the animation\'s toy: 8 experts of hidden 8, top-2, 4,008 total and 1,576 active', async ({ page }) => {
    await expect(readout(page, 'total')).toHaveText('4,008');
    await expect(readout(page, 'active')).toHaveText('1,576');
    await expect(readout(page, 'active-share')).toHaveText('39.3%');
    await expect(readout(page, 'expert-active')).toHaveText('384');
    await expect(readout(page, 'router')).toHaveText('64');
    await expect(readout(page, 'combos')).toHaveText('28');
    await expect(page.locator('[data-section="toy"]')).toContainText('Active = the parameters multiplied for one token: every block parameter except unused experts, plus the unembedding. The embedding table is left out: looking up a row is not a multiplication.');
  });

  test('the routing table for the four words shows the hand-picked picks and gate weights, only at 8 experts and split 1', async ({ page }) => {
    await expect(readout(page, 'picks-1')).toHaveText('E2, E5');
    await expect(readout(page, 'picks-3')).toHaveText('E3, E6');
    await expect(readout(page, 'gates-3')).toHaveText('0.622 · 0.378');
    await expect(readout(page, 'picks-4')).toHaveText('E1, E7');
    await expect(readout(page, 'gates-4')).toHaveText('0.562 · 0.438');
    await expect(page.locator('[data-section="toy"] .g-bars').first()).toBeVisible();
    await pickRouted(page, 16);
    await expect(page.locator('[data-readout="routing-note"]')).toBeVisible();
    await expect(readout(page, 'picks-1')).toHaveCount(0);
  });

  test('try this 1: routed 0 → 8 → 16 grows total, active barely moves (only the router grows)', async ({ page }) => {
    await pickRouted(page, 0);
    await expect(readout(page, 'total')).toHaveText('1,576');
    await expect(readout(page, 'active')).toHaveText('1,448');
    await expect(readout(page, 'router')).toHaveText('0');
    await expect(readout(page, 'expert-active')).toHaveText('384');
    await pickRouted(page, 8);
    await expect(readout(page, 'total')).toHaveText('4,008');
    await expect(readout(page, 'active')).toHaveText('1,576');
    await pickRouted(page, 16);
    await expect(readout(page, 'total')).toHaveText('7,208');
    await expect(readout(page, 'active')).toHaveText('1,704');
    await expect(readout(page, 'router')).toHaveText('128');
  });

  test('try this 1: DeepSeek-V4-Pro keeps 1.6T to use 49B per token, 6 of 384 routed experts', async ({ page }) => {
    await page.locator('#real [data-value="deepseek-v4-pro"]').click();
    await expect(readout(page, 'real-total')).toHaveText('1.6T');
    await expect(readout(page, 'real-active')).toHaveText('49B');
    await expect(readout(page, 'real-share')).toHaveText('3.1%');
    await expect(readout(page, 'real-routed')).toHaveText('6 of 384');
    await expect(readout(page, 'real-routed-share')).toHaveText('1.6%');
    await expect(page.locator('#routed')).toBeHidden();
    await page.locator('#real [data-value="toy"]').click();
    await expect(page.locator('#routed')).toBeVisible();
    await expect(readout(page, 'total')).toHaveText('4,008');
  });

  test('try this 2: split 2 then 4 keeps 384 parameters per token and multiplies the choices', async ({ page }) => {
    await page.locator('#split [data-value="2"]').click();
    await expect(readout(page, 'expert-active')).toHaveText('384');
    await expect(readout(page, 'combos')).toHaveText('1,820');
    await expect(readout(page, 'router')).toHaveText('128');
    await expect(readout(page, 'total')).toHaveText('4,136');
    await expect(readout(page, 'active')).toHaveText('1,704');
    await page.locator('#split [data-value="4"]').click();
    await expect(readout(page, 'expert-active')).toHaveText('384');
    await expect(readout(page, 'combos')).toHaveText('10,518,300');
    await expect(readout(page, 'router')).toHaveText('256');
    await expect(readout(page, 'total')).toHaveText('4,392');
    await expect(readout(page, 'active')).toHaveText('1,960');
  });

  test('the shared expert takes one of the active slots: 4,392 total, active still 1,576', async ({ page }) => {
    await page.locator('#shared').click();
    await expect(page.locator('#shared')).toHaveAttribute('aria-pressed', 'true');
    await expect(readout(page, 'total')).toHaveText('4,392');
    await expect(readout(page, 'active')).toHaveText('1,576');
    await expect(readout(page, 'expert-active')).toHaveText('384');
  });
});

test.describe('moe toy: route, count, rebalance (panel B)', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('opens on step 0 with gamma 0.1: loads 96 · 51 · 12 · 16 · 25 · 9 · 21 · 26 and imbalance 3.00', async ({ page }) => {
    await expect(readout(page, 'imbalance')).toHaveText('3.00');
    const loads = await Promise.all([1, 2, 3, 4, 5, 6, 7, 8].map((e) => readout(page, `load-${e}`).textContent()));
    expect(loads.map((t) => t.trim())).toEqual(['96', '51', '12', '16', '25', '9', '21', '26']);
    await expect(page.locator('[data-section="toy"]')).toContainText('Each step routes a fresh batch of 128 tokens, so the number wobbles; the bias keeps it near 1.');
  });

  test('try this 3: gamma 0.1 balances to 1.13 by step 6, with the storyboard\'s biases', async ({ page }) => {
    await page.locator('#step').fill('6');
    await expect(readout(page, 'imbalance')).toHaveText('1.13');
    const biases = await Promise.all([1, 2, 3, 4, 5, 6, 7, 8].map((e) => readout(page, `bias-${e}`).textContent()));
    expect(biases.map((t) => t.trim())).toEqual(['−0.60', '−0.10', '0.40', '0.40', '0.40', '0.40', '0.40', '0.40']);
    const loads = await Promise.all([1, 2, 3, 4, 5, 6, 7, 8].map((e) => readout(page, `load-${e}`).textContent()));
    expect(loads.map((t) => t.trim())).toEqual(['32', '35', '36', '33', '28', '28', '30', '34']);
  });

  test('try this 3: gamma 0 stays near 3 (step 9: 2.91); gamma 0.2 overshoots (1.22 at step 2, 1.84 at step 5)', async ({ page }) => {
    await pickGamma(page, 0);
    expect(await imbalanceAt(page, 1)).toBe('2.53');
    expect(await imbalanceAt(page, 9)).toBe('2.91');
    await pickGamma(page, 0.2);
    expect(await imbalanceAt(page, 2)).toBe('1.22');
    expect(await imbalanceAt(page, 5)).toBe('1.84');
  });
});
