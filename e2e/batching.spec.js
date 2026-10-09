import { test, expect } from '@playwright/test';
import { registerLessonContract, LESSON_STEPPER, stepperParts } from './lesson-helpers.js';
import { collectConsoleErrors } from './helpers.js';
import { CAPTIONS, TRY_THIS, STAGE_TEXT } from '../tests/batching-expected.js';

const URL = '/serving/#batching';

registerLessonContract({ name: 'batching', url: URL, captions: CAPTIONS, factRows: 6, returnHash: 'batching' });

const readout = (page, name) => page.locator(`[data-readout="${name}"]`);
const parts = stepperParts(LESSON_STEPPER);
const stageText = (page) => page.locator(`${parts.stage} svg`).evaluate((svg) => svg.textContent);

async function goToFrame(page, index) {
  await page.locator(parts.scrub).fill(String(index));
  await expect(page.locator(parts.stage)).toHaveAttribute('data-step', String(index));
  await expect(page.locator(parts.stage)).toHaveAttribute('data-progress', '1');
  return page.locator(parts.stage).evaluate((s) => s.innerHTML);
}

test.describe('batching stage: the numbers printed on each frame', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('frames 2 to 6 print the storyboard\'s lane readouts', async ({ page }) => {
    const expectations = [[1, [STAGE_TEXT.frame2Done]], [2, [STAGE_TEXT.frame3Idle]], [3, [STAGE_TEXT.frame4Busy]], [4, [STAGE_TEXT.frame5Admitted]], [5, [STAGE_TEXT.frame6Static, STAGE_TEXT.frame6Continuous]]];
    for (const [index, texts] of expectations) {
      await goToFrame(page, index);
      const text = await stageText(page);
      texts.forEach((t) => expect(text, `frame ${index + 1}`).toContain(t));
    }
  });

  test('frames 7 to 10 print the mixed step, the long prompt, the budget and the fourth seat', async ({ page }) => {
    const expectations = [[6, [STAGE_TEXT.frame7Mixed]], [7, [STAGE_TEXT.frame8Step, STAGE_TEXT.frame8Times, STAGE_TEXT.frame8Note]], [8, [STAGE_TEXT.frame9Budget, STAGE_TEXT.frame9Done, STAGE_TEXT.frame9First]], [9, [STAGE_TEXT.frame10Four, 'D 1 → 4']]];
    for (const [index, texts] of expectations) {
      await goToFrame(page, index);
      const text = await stageText(page);
      texts.forEach((t) => expect(text, `frame ${index + 1}`).toContain(t));
    }
  });

  test('scrubbing frame 6 to 3 and back to 6 gives identical stage markup', async ({ page }) => {
    const first = await goToFrame(page, 5);
    await goToFrame(page, 2);
    expect(await goToFrame(page, 5)).toBe(first);
  });
});

test.describe('batching toy: the seat timeline', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('opens on the storyboard default with its exact numbers', async ({ page }) => {
    await expect(readout(page, 'static-last-step')).toHaveText('10');
    await expect(readout(page, 'continuous-last-step')).toHaveText('6');
    await expect(readout(page, 'static-utilization')).toHaveText('57.6%');
    await expect(readout(page, 'continuous-utilization')).toHaveText('90.5%');
    await expect(readout(page, 'static-tokens-per-step')).toHaveText('1.36');
    await expect(readout(page, 'continuous-tokens-per-step')).toHaveText('2.14');
    await expect(readout(page, 'static-D-admitted')).toHaveText('7');
    await expect(readout(page, 'continuous-D-admitted')).toHaveText('3');
    await expect(readout(page, 'longest-step')).toHaveText('14.7 ms');
    await expect(page.locator('#budget')).toBeHidden();
  });

  test('the toy lists the four try-this prompts with their insights', async ({ page }) => {
    const items = page.locator('[data-section="toy"] ol.try-this li');
    await expect(items).toHaveCount(TRY_THIS.length);
    await expect(items).toHaveText(TRY_THIS, { useInnerText: true });
  });

  test('try this 1: refilling a seat the step after it frees ends the run at step 6', async ({ page }) => {
    await expect(readout(page, 'static-D-waited')).toHaveText('6');
    await expect(readout(page, 'continuous-D-waited')).toHaveText('2');
  });

  test('try this 2: a longer answer from C moves static D to step 11 and leaves continuous D at steps 3 to 6', async ({ page }) => {
    await page.locator('#cOutput').fill('10');
    await expect(readout(page, 'static-D-admitted')).toHaveText('11');
    await expect(readout(page, 'static-D-done')).toHaveText('14');
    await expect(readout(page, 'static-utilization')).toHaveText('51.1%');
    await expect(readout(page, 'continuous-D-admitted')).toHaveText('3');
    await expect(readout(page, 'continuous-D-done')).toHaveText('6');
    await expect(readout(page, 'continuous-last-step')).toHaveText('10');
    await expect(readout(page, 'continuous-utilization')).toHaveText('69.7%');
  });

  test('try this 3: a fourth seat starts D at step 1 but drops utilization to 67.9%', async ({ page }) => {
    await page.locator('#seats').fill('2'); // the slider's stops are 2, 3, 4: index 2 is 4 seats
    await expect(readout(page, 'continuous-D-admitted')).toHaveText('1');
    await expect(readout(page, 'continuous-D-done')).toHaveText('4');
    await expect(readout(page, 'continuous-utilization')).toHaveText('67.9%');
    await expect(readout(page, 'static-D-admitted')).toHaveText('7');
  });

  test('try this 4: the long prompt, then each budget', async ({ page }) => {
    await page.locator('#dPrompt [data-value="4096"]').click();
    await expect(page.locator('#budget')).toBeVisible();
    await expect(readout(page, 'longest-step')).toHaveText('290 ms');
    await expect(readout(page, 'a-done')).toHaveText('348 ms');
    await expect(readout(page, 'c-done')).toHaveText('378 ms');
    await expect(readout(page, 'd-first-token')).toHaveText('334 ms');
    await expect(readout(page, 'a-gap')).toHaveText('290 ms');
    await page.locator('#budget [data-value="2048"]').click();
    await expect(readout(page, 'longest-step')).toHaveText('145 ms');
    await expect(readout(page, 'a-done')).toHaveText('334 ms');
    await page.locator('#budget [data-value="512"]').click();
    await expect(readout(page, 'longest-step')).toHaveText('36.2 ms');
    await expect(readout(page, 'a-done')).toHaveText('116 ms');
    await expect(readout(page, 'c-done')).toHaveText('189 ms');
    await expect(readout(page, 'd-first-token')).toHaveText('348 ms');
    // Steps differ in length here, so the lanes are compared in milliseconds, and continuous D finishes first.
    await expect(page.getByText('Steps differ in length once D\'s long prompt runs, so the lanes are compared in milliseconds.')).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Per request, in steps (steps differ in length)' })).toBeVisible();
    await expect(readout(page, 'static-run-ends')).toContainText('ms');
    await expect(readout(page, 'continuous-d-done')).toContainText('ms');
    await expect(readout(page, 'static-last-step')).toHaveCount(0);
    await page.locator('#dPrompt [data-value="6"]').click();
    await expect(readout(page, 'static-last-step')).toHaveText('10');
    await expect(page.getByText('compared in milliseconds')).toHaveCount(0);
  });

  test('every control combination draws both lanes without errors and the budget hides again with a short prompt', async ({ page }) => {
    const errors = collectConsoleErrors(page);
    await page.locator('#dPrompt [data-value="4096"]').click();
    await page.locator('#budget [data-value="512"]').click();
    await page.locator('#dPrompt [data-value="6"]').click();
    await expect(page.locator('#budget')).toBeHidden();
    await expect(readout(page, 'longest-step')).toHaveText('14.7 ms');
    await page.locator('#cOutput').fill('2');
    await page.locator('#seats').fill('0');
    await expect(page.locator('[data-section="toy"] svg .g-request').first()).toBeVisible();
    expect(errors).toEqual([]);
  });
});
