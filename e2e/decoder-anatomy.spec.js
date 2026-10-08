import { test, expect } from '@playwright/test';
import { registerLessonContract } from './lesson-helpers.js';
import { collectConsoleErrors } from './helpers.js';
import { CAPTIONS, CHECK_WORK } from '../tests/decoder-anatomy-expected.js';

const URL = '/architecture/#decoder-anatomy';

registerLessonContract({ name: 'decoder-anatomy', url: URL, captions: CAPTIONS, factRows: 13, returnHash: 'decoder-anatomy' });

const readout = (page, name) => page.locator(`[data-readout="${name}"]`);
const cell = (page, part, col) => page.locator(`table[data-readout="parts"] tr[data-part="${part}"] td[data-col="${col}"]`);

test.describe('decoder-anatomy toy: where do the parameters live?', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('opens on the toy preset with the animation\'s numbers and the exact "Check my work" box', async ({ page }) => {
    await expect(readout(page, 'total')).toHaveText('1,576');
    await expect(readout(page, 'active')).toHaveText('1,448');
    await expect(readout(page, 'active-with-embedding')).toHaveText('1,576');
    await expect(readout(page, 'per-block')).toHaveText('attention 256 · MLP 384 · norms 16');
    await expect(readout(page, 'active-note')).toHaveText('Active 1,448 of 1,576: active leaves out the 128-parameter embedding table.');
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK);
    await expect(cell(page, 'mlp', 'share')).toHaveText('48.7%');
    await expect(cell(page, 'attention', 'share')).toHaveText('32.5%');
    await expect(cell(page, 'embedding', 'share')).toHaveText('8.1%');
    await expect(cell(page, 'norms', 'share')).toHaveText('2.5%');
  });

  test('the share bar folds parts under 18 px into "others" with a zoomed tail (README lesson 19)', async ({ page }) => {
    const bar = page.locator('[data-section="toy"] .g-share').first();
    await expect(bar.locator('.g-seg.g-part-others')).toHaveCount(1); // the bar segment; the legend swatch carries the class too
    await expect(bar.locator('.g-bracket')).toHaveCount(1);
    await expect(bar.locator('.g-hatch')).toHaveCount(0);
  });

  test('try this 3: experts 0 → 8 → 16 grows total, active barely moves', async ({ page }) => {
    await page.locator('#experts').fill('3');
    await expect(readout(page, 'total')).toHaveText('4,008');
    await expect(readout(page, 'active')).toHaveText('1,576');
    await expect(readout(page, 'per-block')).toHaveText('attention 256 · expert 192 × 8 · norms 16');
    await page.locator('#experts').fill('4');
    await expect(readout(page, 'total')).toHaveText('7,208');
    await expect(readout(page, 'active')).toHaveText('1,704');
  });

  test('try this 2: one block makes the tables 27.8 %; eight wide blocks make them small', async ({ page }) => {
    await page.locator('#layers').fill('1');
    await expect(readout(page, 'total')).toHaveText('920');
    await expect(cell(page, 'embedding', 'share')).toHaveText('13.9%');
    await expect(cell(page, 'head', 'share')).toHaveText('13.9%');
    await page.locator('#layers').fill('8');
    await page.locator('#dModel').fill('3');
    await expect(readout(page, 'total')).toHaveText('330,816');
  });

  test('try this 1: real presets: GPT-3 is two-thirds MLP; gpt-oss is 98 % experts', async ({ page }) => {
    await page.locator('#preset [data-value="gpt3"]').click();
    await expect(cell(page, 'mlp', 'share')).toHaveText('66.4%');
    await expect(cell(page, 'attention', 'share')).toHaveText('33.2%');
    await expect(cell(page, 'embedding', 'share')).toHaveText('0.35%');
    await expect(readout(page, 'published-gap')).toHaveText('computed 174.6B vs published 175B (−0.23%)');
    await expect(page.locator('#layers')).toBeDisabled();
    await page.locator('#preset [data-value="gptOss120b"]').click();
    await expect(cell(page, 'experts', 'share')).toHaveText('98.2%');
    await expect(cell(page, 'attention', 'share')).toHaveText('0.82%');
    await expect(cell(page, 'head', 'active-share')).toHaveText('11.3%');
    await expect(readout(page, 'active')).toHaveText('5.13B');
    await expect(readout(page, 'published-gap')).toHaveText('computed 116.83B vs published 116.83B (matches)');
  });

  test('DeepSeek-V3 matches the paper; V4-Pro prints its published 1.6T / 49B and a neutral "not published" part', async ({ page }) => {
    await page.locator('#preset [data-value="deepseekV3"]').click();
    await expect(cell(page, 'experts', 'share')).toHaveText('97.8%');
    await expect(readout(page, 'published-gap')).toHaveText('computed 671.03B vs published 671B (matches)');
    await page.locator('#preset [data-value="deepseekV4Pro"]').click();
    await expect(cell(page, 'experts', 'share')).toHaveText('97.0%');
    await expect(cell(page, 'unknown', 'share')).toHaveText('3.0%');
    await expect(readout(page, 'total')).toContainText('1.6T');
    await expect(readout(page, 'active')).toContainText('49B');
    await expect(page.locator('[data-section="toy"] .g-share .g-part-none').first()).toBeVisible();
    await expect(page.locator('[data-section="toy"] .g-share .g-hatch')).toHaveCount(0);
    await page.locator('#preset [data-value="toy"]').click();
    await expect(readout(page, 'total')).toHaveText('1,576');
    await expect(page.locator('#layers')).toBeEnabled();
  });

  test('the experts = 2 stop says it is a dense MLP with a router (README lesson 22)', async ({ page }) => {
    await page.locator('#experts').fill('1');
    await expect(readout(page, 'experts-note')).toHaveText('2 of 2 experts used: this is a dense MLP with a router');
    await page.locator('#experts').fill('3');
    await expect(readout(page, 'experts-note')).toHaveText('');
  });

  test('the "8, like the animation" chip sets the experts slider, and real presets show the DeepSeek-V3 gap and the Kimi K3 line', async ({ page }) => {
    const errors = collectConsoleErrors(page);
    await page.locator('#experts-preset [data-value="8"]').click();
    await expect(readout(page, 'total')).toHaveText('4,008');
    await expect(page.locator('#experts')).toHaveValue('3');
    await expect(readout(page, 'kimi-line')).toHaveText('Kimi K3 (2026): 2.35B of 2.78T, 0.08%');
    await expect(readout(page, 'v3-gap')).toBeHidden();
    await page.locator('#preset [data-value="deepseekV3"]').click();
    await expect(readout(page, 'v3-gap')).toBeVisible();
    await expect(readout(page, 'v3-gap')).toContainText('685B');
    expect(errors).toEqual([]);
  });
});
