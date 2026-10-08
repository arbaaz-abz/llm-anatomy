import { test, expect } from '@playwright/test';
import { registerLessonContract, stepperParts, LESSON_STEPPER } from './lesson-helpers.js';
import { collectConsoleErrors } from './helpers.js';
import { CAPTIONS, CHECK_WORK } from '../tests/multimodal-expected.js';

const URL = '/architecture/#multimodal';

registerLessonContract({ name: 'multimodal', url: URL, captions: CAPTIONS, factRows: 7, returnHash: 'multimodal' });

const readout = (page, name) => page.locator(`[data-readout="${name}"]`);
// The side sliders step through multiples of patch × merge from 112 px; at patch 14 and merge 2 that is every 28 px.
const sideIndex = (px) => String((px - 112) / 28);

async function setSize(page, width, height) {
  await page.locator('#width').fill(sideIndex(width));
  await page.locator('#height').fill(sideIndex(height));
}

test.describe('multimodal stage: the counts on the page text', () => {
  test('frame 7 prints the tokens of a phone photo and of Kimi K3\'s largest input from the data', async ({ page }) => {
    await page.goto(URL);
    const p = stepperParts(LESSON_STEPPER);
    await page.locator(p.scrub).fill('6');
    await expect(page.locator(p.count)).toHaveText('7 / 10');
    const text = await page.locator(`${p.stage} svg`).evaluate((svg) => svg.textContent);
    ['72 × 72 = 5,184 patches', '1,296 tokens', '256 × 256 = 65,536 patches', '16,384 tokens', '64 of these fill a 1M window'].forEach((s) => expect(text).toContain(s));
  });

  test('frame 10 names the text-only models in the page text, not only in the facts list', async ({ page }) => {
    await page.goto(URL);
    const p = stepperParts(LESSON_STEPPER);
    await page.locator(p.scrub).fill('9');
    await expect(page.locator('[data-section="animation"] .below-stage')).toContainText('Not every 2026 frontier model sees images');
    await expect(page.locator('[data-section="animation"] .below-stage')).toContainText('DeepSeek-V4-Pro');
    await expect(page.locator('[data-section="animation"] .below-stage')).toContainText('gpt-oss-120b');
  });
});

test.describe('multimodal toy: what does a picture cost?', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('opens on the Kimi K3 preset with the animation\'s phone photo and the exact "Check my work" line', async ({ page }) => {
    await expect(readout(page, 'grid')).toHaveText('72 × 72 = 5,184');
    await expect(readout(page, 'tokens-per-frame')).toHaveText('1,296');
    await expect(readout(page, 'tokens')).toHaveText('1,296');
    await expect(readout(page, 'share')).toHaveText('0.12%');
    await expect(readout(page, 'fit')).toHaveText('809');
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK);
    await expect(page.locator('#seconds')).toBeHidden();
    await expect(page.locator('#fps')).toBeHidden();
  });

  test('try this 1: merging neighbors is the main lever; the largest input is 16,384 tokens, 64 fill a 1M window', async ({ page }) => {
    await page.locator('#merge [data-value="1"]').click();
    await expect(readout(page, 'tokens')).toHaveText('5,184');
    await page.locator('#merge [data-value="2"]').click();
    await expect(readout(page, 'tokens')).toHaveText('1,296');
    await setSize(page, 3584, 3584);
    await expect(readout(page, 'tokens')).toHaveText('16,384');
    await expect(readout(page, 'share')).toHaveText('1.6%');
    await expect(readout(page, 'fit')).toHaveText('64');
  });

  test('try this 2: keeping the aspect ratio spends tokens only on real pixels', async ({ page }) => {
    await page.locator('#height').fill(sideIndex(504));
    await expect(readout(page, 'grid')).toHaveText('72 × 36 = 2,592');
    await expect(readout(page, 'tokens')).toHaveText('648');
    await expect(readout(page, 'check-work')).toHaveText('1008 ÷ 14 = 72 · 504 ÷ 14 = 36 · 72 × 36 = 2,592 · ÷ (2 × 2) = 648');
    await page.locator('#height').fill(sideIndex(1008));
    await expect(readout(page, 'tokens')).toHaveText('1,296');
  });

  test('try this 3: video pays per frame; an hour at 2 fps does not fit a 1M window', async ({ page }) => {
    await page.locator('#media [data-value="video"]').click();
    await setSize(page, 448, 448);
    await expect(readout(page, 'tokens-per-frame')).toHaveText('256');
    await expect(readout(page, 'tokens')).toHaveText('30,720');
    await expect(readout(page, 'check-work')).toHaveText('448 ÷ 14 = 32 · 32 × 32 = 1,024 · ÷ (2 × 2) = 256 · × 120 frames = 30,720');
    await page.locator('#seconds').fill('2'); // 600 s
    await page.locator('#fps [data-value="1"]').click();
    await expect(readout(page, 'tokens')).toHaveText('153,600');
    await expect(readout(page, 'share')).toHaveText('14.6%');
    await page.locator('#context [data-value="262144"]').click();
    await expect(readout(page, 'share')).toHaveText('58.6%');
    await page.locator('#context [data-value="1048576"]').click();
    await page.locator('#seconds').fill('3'); // 3,600 s
    await page.locator('#fps [data-value="2"]').click();
    await expect(readout(page, 'tokens')).toHaveText('1,843,200');
    await expect(readout(page, 'share')).toHaveText('175.8%');
    await expect(page.locator('[data-section="toy"]')).toContainText('does not fit');
  });

  test('the toy preset is the animation\'s 16-pixel image: 4 tokens', async ({ page }) => {
    await page.locator('#preset [data-value="toy"]').click();
    await expect(readout(page, 'grid')).toHaveText('4 × 4 = 16');
    await expect(readout(page, 'tokens')).toHaveText('4');
    await expect(readout(page, 'check-work')).toHaveText('16 ÷ 4 = 4 · 4 × 4 = 16 · ÷ (2 × 2) = 4');
    await expect(page.locator('#width')).toBeHidden();
  });

  test('the 3 × 3 merge chip changes only the merge: 576 tokens for the phone photo', async ({ page }) => {
    await page.locator('#preset [data-value="deepseek"]').click();
    await expect(readout(page, 'tokens')).toHaveText('576');
    await expect(page.locator('#merge [data-value="3"]')).toHaveAttribute('aria-pressed', 'true');
  });

  test('editing a setting moves the preset chip to "custom"; no console errors', async ({ page }) => {
    const errors = collectConsoleErrors(page);
    await page.locator('#patch [data-value="16"]').click();
    await expect(page.locator('#preset [data-value="custom"]')).toHaveAttribute('aria-pressed', 'true');
    await expect(readout(page, 'grid')).toContainText('=');
    expect(errors).toEqual([]);
  });
});
