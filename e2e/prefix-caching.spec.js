import { test, expect } from '@playwright/test';
import { registerLessonContract } from './lesson-helpers.js';
import { collectConsoleErrors } from './helpers.js';
import { CAPTIONS, CHECK_WORK, DEFAULT_ROWS, TRY_THIS } from '../tests/prefix-caching-expected.js';

const URL = '/serving/#prefix-caching';

registerLessonContract({ name: 'prefix-caching', url: URL, captions: CAPTIONS, factRows: 10, returnHash: 'prefix-caching' });

const toy = '[data-section="toy"]';
const readout = (page, name) => page.locator(`${toy} [data-readout="${name}"]`);
const sizeIndex = { 1: 0, 2: 1, 4: 2, 8: 3, 16: 4 };
const setSize = (page, size) => page.locator('#blockSize').fill(String(sizeIndex[size]));
const pick = (page, group, value) => page.locator(`#${group} button[data-value="${value}"]`).click();
const stageText = (page) => page.locator('[data-section="animation"] .stepper-stage svg').evaluate((svg) => [...svg.querySelectorAll('text')].map((t) => t.textContent).join(' '));
const stage = (page) => page.locator('[data-section="animation"] .stepper-stage');
const goTo = async (page, step) => {
  await page.locator('[data-section="animation"] input[type="range"]').fill(String(step));
  await expect(stage(page)).toHaveAttribute('data-step', String(step));
  await expect(stage(page)).toHaveAttribute('data-progress', '1');
};

test.describe('prefix-caching toy', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('opens on the storyboard default with its exact numbers', async ({ page }) => {
    for (const [id, prompt, hit, computed, blocks, evicted] of DEFAULT_ROWS) {
      await expect(readout(page, `prompt-${id}`)).toHaveText(prompt);
      await expect(readout(page, `hit-${id}`)).toHaveText(hit);
      await expect(readout(page, `computed-${id}`)).toHaveText(computed);
      await expect(readout(page, `blocks-${id}`)).toHaveText(blocks);
      await expect(readout(page, `evicted-${id}`)).toHaveText(evicted);
    }
    await expect(readout(page, 'total-prompt')).toHaveText('57');
    await expect(readout(page, 'total-hit')).toHaveText('24');
    await expect(readout(page, 'total-computed')).toHaveText('33');
    await expect(readout(page, 'hit-rate')).toHaveText('42.1%');
    await expect(readout(page, 'skipped')).toHaveText('707 ms');
    await expect(readout(page, 'held')).toHaveText('3.28 GB');
    await expect(readout(page, 'rate-used')).toHaveText('42.1%');
    await expect(readout(page, 'price')).toHaveText('$1.53');
    await expect(readout(page, 'price-plain')).toHaveText('$2.00');
    await expect(page.locator(`${toy} #hitRate + output`)).toHaveText('42.1%');
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK);
    await expect(page.locator(`${toy} svg .g-ptree`)).toHaveCount(1);
  });

  test('try this 1: block size 16 falls to 28.1% (C still reuses 16), size 1 rises to 47.4% (D reuses 3)', async ({ page }) => {
    await setSize(page, 16);
    await expect(readout(page, 'hit-rate')).toHaveText('28.1%');
    await expect(readout(page, 'hit-B')).toHaveText('0');
    await expect(readout(page, 'hit-C')).toHaveText('16');
    await expect(page.locator('#blockSize + output')).toHaveText('16 tokens');
    await setSize(page, 1);
    await expect(readout(page, 'hit-rate')).toHaveText('47.4%');
    await expect(readout(page, 'hit-D')).toHaveText('3');
    await expect(page.locator(`${toy} svg .g-ptree`)).toHaveCount(0);
    await expect(page.locator(`${toy} svg`).first()).toContainText('too many blocks to draw as a tree');
    await setSize(page, 4);
    await expect(readout(page, 'hit-rate')).toHaveText('42.1%');
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK);
  });

  test('try this 2: B again evicts A\'s turn; pool 6 makes C lose B\'s blocks and D evict A\'s turn', async ({ page }) => {
    await page.locator('#req-B-again').click();
    await expect(readout(page, 'hit-B-again')).toHaveText('8');
    await expect(readout(page, 'evicted-B-again')).toHaveText('The cat sat down · Where did you sit');
    await expect(readout(page, 'blocks-B-again')).toHaveText('0, 1, 3, 2');
    await pick(page, 'pool', 6);
    await expect(readout(page, 'evicted-C')).toHaveText('? Yes , fish · Do you like fish');
    await expect(readout(page, 'evicted-D')).toHaveText('It was warm . · Why down there ? · The cat sat down · Where did you sit');
    await page.locator('#req-B-again').click();
    await pick(page, 'pool', 8);
    await expect(readout(page, 'hit-rate')).toHaveText('42.1%');
  });

  test('try this 3: the price follows the hit rate; 0% costs more than not caching, DeepSeek has no write fee', async ({ page }) => {
    await page.locator('#hitRate').fill('0');
    await expect(readout(page, 'price')).toHaveText('$2.50');
    await expect(readout(page, 'price-plain')).toHaveText('$2.00');
    await expect(readout(page, 'rate-used')).toHaveText('0.00%');
    await page.locator(`${toy} #hit-presets button`, { hasText: 'this toy' }).click();
    await expect(readout(page, 'price')).toHaveText('$1.53');
    await page.locator(`${toy} #hit-presets button`, { hasText: 'DeepSeek 2025' }).click();
    await expect(readout(page, 'price')).toHaveText('$1.21');
    await expect(page.locator('#hitRate + output')).toHaveText('56.3%');
    await expect(readout(page, 'check-work')).toContainText('h = 56.3% (set by hand)');
    await pick(page, 'provider', 'deepseek');
    await expect(readout(page, 'price')).toHaveText('$0.30');
    await expect(readout(page, 'price-plain')).toHaveText('$0.66');
    await expect(page.locator('#writePremium button[data-value="on"]')).toBeDisabled();
    await expect(page.locator(`${toy} [data-choice-note]`)).toContainText('DeepSeek charges no cache write');
    await expect(readout(page, 'check-work')).toContainText('price per M = (1 − h) · miss + h · hit');
    await page.locator('#hitRate').fill('0');
    await expect(readout(page, 'price')).toHaveText('$0.66');
  });

  test('switching the price away and back restores the defaults exactly (write premium included)', async ({ page }) => {
    await pick(page, 'provider', 'opus');
    await expect(readout(page, 'price')).toHaveText('$2.98');
    await pick(page, 'provider', 'deepseek');
    await pick(page, 'provider', 'sonnet');
    await expect(page.locator('#writePremium button[data-value="on"]')).toBeEnabled();
    await expect(page.locator('#writePremium button[data-value="on"]')).toHaveAttribute('aria-pressed', 'true');
    await expect(readout(page, 'price')).toHaveText('$1.53');
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK);
    await pick(page, 'provider', 'sonnet');
    await page.locator('#writePremium button[data-value="off"]').click();
    await expect(readout(page, 'price')).toHaveText('$1.24');
    await pick(page, 'provider', 'deepseek');
    await pick(page, 'provider', 'sonnet');
    await expect(page.locator('#writePremium button[data-value="off"]')).toHaveAttribute('aria-pressed', 'true');
  });

  test('turning every request off leaves an empty cache, not an error', async ({ page }) => {
    const errors = collectConsoleErrors(page);
    for (const id of ['A', 'B', 'C', 'D']) await page.locator(`#req-${id}`).click();
    await expect(readout(page, 'hit-rate')).toHaveText('—');
    await expect(readout(page, 'total-prompt')).toHaveText('0');
    await expect(readout(page, 'check-work')).toContainText('h = 0 (no request has arrived)');
    await page.locator('#req-A').click();
    await expect(readout(page, 'hit-A')).toHaveText('0');
    expect(errors).toEqual([]);
  });

  test('the try-this list sits at the end of the toy, its numbers computed (README lesson 34)', async ({ page }) => {
    const items = page.locator(`${toy} ol.try-this > li`);
    await expect(items).toHaveCount(TRY_THIS.length);
    const plain = (s) => s.replace(/`/g, '').replace('[[paged-attention]]', 'PagedAttention');
    for (const [i, t] of TRY_THIS.entries()) {
      await expect(items.nth(i)).toHaveText(`${plain(t.prompt)} → Insight: ${t.insight}${plain(t.rest)}`);
    }
    expect(await page.locator(`${toy} .toy > *:last-child`).evaluate((n) => n.matches('ol.try-this'))).toBe(true);
  });
});

test.describe('prefix-caching stage', () => {
  test('frame 7 ends with D admitted: the free queue is 3 2 1 0 and the system prompt survives', async ({ page }) => {
    await page.goto(URL);
    await goTo(page, 6);
    const text = await stageText(page);
    expect(text).toContain('free queue (evict from the left): 3 2 1 0');
    expect(text).toContain('after D finishes: 3 2 1 0 6 7 4 5');
    expect(text).toContain('evicted: blocks 5, 4, 7, 6');
  });

  test('frame 8 tallies 24 of 57 prompt tokens, 42.1%, beside DeepSeek\'s 56.3%', async ({ page }) => {
    await page.goto(URL);
    await goTo(page, 7);
    const text = await stageText(page);
    expect(text).toContain('24 of 57 prompt tokens came from the cache');
    expect(text).toContain('42.1%');
    expect(text).toContain('DeepSeek production (Feb 2025): 56.3%');
  });

  test('frame 9 sends C to replica 1 for 16 hit tokens and to replica 2 for 8', async ({ page }) => {
    await page.goto(URL);
    await goTo(page, 8);
    const text = await stageText(page);
    expect(text).toContain('KV-aware sends C to replica 1');
    expect(text).toMatch(/16\s+hit tokens/);
    expect(text).toMatch(/8\s+hit tokens/);
  });

  test('frame 11 prices: $2.00, $2.50, $0.20 at Anthropic; $0.66 and $0.022 at DeepSeek; 707 ms and 3.28 GB', async ({ page }) => {
    await page.goto(URL);
    await goTo(page, 10);
    const text = await stageText(page);
    for (const s of ['$2.00', '$2.50', '$0.20', '$0.66', '$0.022', '707 ms', '3.28 GB', 'prices read 2026-10-07; they change']) expect(text).toContain(s);
  });
});
