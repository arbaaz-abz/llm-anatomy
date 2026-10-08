import { test, expect } from '@playwright/test';
import { registerLessonContract } from './lesson-helpers.js';
import { CAPTIONS, TRY_THIS } from '../tests/rlvr-grpo-expected.js';

const URL = '/training/#rlvr-grpo';

registerLessonContract({ name: 'rlvr-grpo', url: URL, captions: CAPTIONS, factRows: 13, returnHash: 'rlvr-grpo' });

const readout = (page, name) => page.locator(`[data-readout="${name}"]`);
// A cell's printed number is its .g-text (the glyph's <title> repeats it for hover).
const cell = (page, row, col) => page.locator(`[data-section="toy"] [data-row="${row}"] [data-col="${col}"] .g-text`);
const chip = (page, row, tok) => page.locator(`[data-section="toy"] [data-row="${row}"] [data-tok="${tok}"]`);
const hatched = (page) => page.locator('[data-section="toy"] .toy-table .g-token .g-hatch');
const column = async (page, col) => {
  const out = [];
  for (let row = 1; row <= 8; row += 1) out.push((await cell(page, row, col).textContent()).trim());
  return out;
};
const setK = (page, k) => page.locator('#k').fill(String(k));

test.describe('rlvr-grpo toy: grade a group', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('opens on the storyboard default: k = 2, rewards, advantages, stats and the inspector on row 1\'s 56', async ({ page }) => {
    expect(await column(page, 'r')).toEqual(['1', '0', '0', '0', '1', '0', '0', '0']);
    expect(await column(page, 'a')).toEqual(['+1.73', '−0.58', '−0.58', '−0.58', '+1.73', '−0.58', '−0.58', '−0.58']);
    await expect(readout(page, 'mean')).toHaveText('0.250');
    await expect(readout(page, 'std')).toHaveText('0.433');
    await expect(readout(page, 'total-push')).toHaveText('6.93');
    await expect(readout(page, 'signal')).toBeHidden();
    const inspector = { 'insp-row': '1', 'insp-token': '56', 'insp-adv': '+1.7321', 'insp-weight': '0.0250', 'insp-push': '+0.0433', 'insp-sampled': '0.200', 'insp-now': '0.250', 'insp-ratio': '1.250', 'insp-objective': '2.078', 'insp-clipped': 'yes' };
    for (const [name, text] of Object.entries(inspector)) await expect(readout(page, name), name).toHaveText(text);
    await expect(page.locator('[data-section="toy"] .toy-table .g-select')).toHaveCount(1);
  });

  test('the toy ends with the storyboard\'s try-this list: three prompts, each with a named insight', async ({ page }) => {
    const items = page.locator('[data-section="toy"] ol.try-this > li');
    await expect(page.locator('[data-section="toy"] h4', { hasText: 'Try this' })).toHaveCount(1);
    await expect(items).toHaveCount(3);
    for (let i = 0; i < 3; i += 1) {
      const [prompt, insight, rest] = TRY_THIS[i];
      await expect(items.nth(i)).toHaveText(`${prompt} → Insight: ${insight}${rest}`);
    }
  });

  test('try this 1: advantages are relative; the std division amplifies the rare outcome', async ({ page }) => {
    await setK(page, 4);
    expect((await column(page, 'a'))[0]).toBe('+1.00');
    await expect(readout(page, 'total-push')).toHaveText('8.00');
    await setK(page, 8);
    expect(await column(page, 'a')).toEqual(Array(8).fill('0.00'));
    await expect(readout(page, 'signal')).toHaveText('No spread: every A = 0. Dynamic sampling would drop this group.');
    await setK(page, 1);
    expect(await column(page, 'a')).toEqual(['+2.65', ...Array(7).fill('−0.38')]);
    await expect(readout(page, 'total-push')).toHaveText('5.29');
    await setK(page, 0);
    expect(await column(page, 'a')).toEqual(Array(8).fill('0.00'));
    await expect(readout(page, 'signal')).toBeVisible();
    await page.locator('#norm').click();
    await expect(page.locator('#norm')).toHaveAttribute('aria-pressed', 'false');
    await setK(page, 1);
    await expect(readout(page, 'insp-adv')).toHaveText('+0.8750');
    await expect(readout(page, 'total-push')).toHaveText('1.75');
    await setK(page, 2);
    expect((await column(page, 'a'))[0]).toBe('+0.75');
    await expect(readout(page, 'total-push')).toHaveText('3.00');
    await setK(page, 4);
    expect((await column(page, 'a'))[0]).toBe('+0.50');
    await expect(readout(page, 'total-push')).toHaveText('4.00');
  });

  test('every token chip is a selectable stage item (pointer and focus ring)', async ({ page }) => {
    const items = page.locator('[data-section="toy"] g.stage-item[role="button"]');
    await expect(items).toHaveCount(39);
    await expect(items.first()).toHaveCSS('cursor', 'pointer');
  });

  test('the k chips set the slider', async ({ page }) => {
    await page.locator('#k-preset [data-value="4"]').click();
    await expect(page.locator('#k')).toHaveValue('4');
    expect((await column(page, 'r')).filter((r) => r === '1')).toHaveLength(4);
  });

  test('try this 2: sample-level averaging shields the long wrong answer; token-level loss does not', async ({ page }) => {
    await chip(page, 4, 5).click();
    await expect(readout(page, 'insp-push')).toHaveText('−0.0090');
    await expect(cell(page, 4, 'push')).toHaveText('−0.0722');
    await chip(page, 6, 1).click();
    await expect(readout(page, 'insp-push')).toHaveText('−0.0722');
    await expect(cell(page, 6, 'push')).toHaveText('−0.0722');
    await page.locator('#agg [data-value="token"]').click();
    await expect(readout(page, 'insp-push')).toHaveText('−0.0148');
    await expect(cell(page, 4, 'push')).toHaveText('−0.1184');
    await expect(cell(page, 6, 'push')).toHaveText('−0.0148');
  });

  test('try this 3: clip-higher un-clips row 1; row 5 and the lower bound stay clipped', async ({ page }) => {
    await expect(hatched(page)).toHaveCount(3);
    for (const [row, tok] of [[1, 5], [5, 5], [4, 5]]) await expect(chip(page, row, tok).locator('.g-hatch')).toHaveCount(1);
    await expect(readout(page, 'insp-objective')).toHaveText('2.078');
    await expect(readout(page, 'insp-clipped')).toHaveText('yes');
    await page.locator('#eps [data-value="0.28"]').click();
    await expect(chip(page, 1, 5).locator('.g-hatch')).toHaveCount(0);
    await expect(hatched(page)).toHaveCount(2);
    await expect(readout(page, 'insp-objective')).toHaveText('2.165');
    await expect(readout(page, 'insp-clipped')).toHaveText('no');
    await chip(page, 5, 5).click();
    await expect(readout(page, 'insp-ratio')).toHaveText('1.350');
    await expect(readout(page, 'insp-clipped')).toHaveText('yes');
    await chip(page, 4, 5).click();
    await expect(readout(page, 'insp-ratio')).toHaveText('0.750');
    await expect(readout(page, 'insp-clipped')).toHaveText('yes');
  });

  test('keyboard: arrows move the selection and the focus together; Enter keeps it', async ({ page }) => {
    await chip(page, 1, 5).focus();
    await page.keyboard.press('ArrowRight');
    await expect(readout(page, 'insp-token')).toHaveText('56');
    await page.keyboard.press('ArrowLeft');
    await expect(readout(page, 'insp-token')).toHaveText('=');
    await expect(chip(page, 1, 4)).toBeFocused();
    await page.keyboard.press('ArrowDown');
    await expect(readout(page, 'insp-row')).toHaveText('2');
    await expect(chip(page, 2, 4)).toBeFocused();
    await page.keyboard.press('Enter');
    await expect(readout(page, 'insp-row')).toHaveText('2');
    await page.keyboard.press('ArrowUp');
    await expect(readout(page, 'insp-row')).toHaveText('1');
    await expect(page.locator('[data-section="toy"] .toy-table .g-select')).toHaveCount(1);
  });

  test('the selection survives a change of k (clamped to the answer\'s last token)', async ({ page }) => {
    await chip(page, 4, 8).click();
    await setK(page, 0);
    await expect(readout(page, 'insp-row')).toHaveText('4');
    await expect(readout(page, 'insp-token')).toHaveText('63');
  });
});
