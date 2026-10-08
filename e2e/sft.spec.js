import { test, expect } from '@playwright/test';
import { registerLessonContract } from './lesson-helpers.js';
import { CAPTIONS, CHECK_WORK, CHECK_WORK_NO_PROMPT_MASK, CHECK_WORK_NO_OBSERVATION_MASK, CHECK_WORK_MASK_ERROR, TRY_THIS, DEFAULT_READOUTS } from '../tests/sft-expected.js';

const URL = '/training/#sft';

registerLessonContract({ name: 'sft', url: URL, captions: CAPTIONS, factRows: 10, returnHash: 'sft' });

const readout = (page, name) => page.locator(`[data-readout="${name}"]`);
const chips = (page) => page.locator('.toy-transcript [role="button"]');
const masked = (page) => page.locator('.toy-transcript [role="button"][data-trained="false"]');
const counts = async (page) => [await readout(page, 'trained').innerText(), await readout(page, 'share').innerText()];

test.describe('sft toy: which tokens teach?', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('opens on the storyboard default with its exact numbers', async ({ page }) => {
    await expect(chips(page)).toHaveCount(26);
    await expect(masked(page)).toHaveCount(11);
    await expect(readout(page, 'trained')).toHaveText(DEFAULT_READOUTS.trained);
    await expect(readout(page, 'masked')).toHaveText(DEFAULT_READOUTS.masked);
    await expect(readout(page, 'share')).toHaveText(DEFAULT_READOUTS.share);
    await expect(readout(page, 'selection')).toHaveText(DEFAULT_READOUTS.selection);
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK);
    await expect(page.locator('[data-section="toy"]')).toContainText('Tags are generic stand-ins; each lab\'s template differs.');
    await expect(readout(page, 'kind-assistant')).toHaveText('10/10');
    await expect(readout(page, 'kind-error')).toHaveText('5/5');
    await expect(readout(page, 'kind-observation')).toHaveText('0/3');
  });

  test('try this 1: without the prompt mask, 23 of 26 train and the user side is a target', async ({ page }) => {
    await page.locator('#maskPrompt').click();
    expect(await counts(page)).toEqual(['23 of 26', '88.5%']);
    await expect(masked(page)).toHaveCount(3);
    await expect(readout(page, 'kind-user')).toHaveText('6/6');
    await expect(readout(page, 'kind-template')).toHaveText('2/2');
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK_NO_PROMPT_MASK);
    await expect(page.locator('ol.try-this li').nth(0)).toContainText(TRY_THIS[0].prompt);
  });

  test('try this 2: without the observation mask, 18 of 26 train and the tool reply is a target', async ({ page }) => {
    await page.locator('#maskObservation').click();
    expect(await counts(page)).toEqual(['18 of 26', '69.2%']);
    await expect(readout(page, 'kind-observation')).toHaveText('3/3');
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK_NO_OBSERVATION_MASK);
    await expect(page.locator('ol.try-this li').nth(1)).toContainText(TRY_THIS[1].prompt);
  });

  test('try this 3: masking the mistake leaves 10 of 26, five chips still visible but hatched', async ({ page }) => {
    await page.locator('#maskError').click();
    expect(await counts(page)).toEqual(['10 of 26', '38.5%']);
    await expect(readout(page, 'masked')).toHaveText('16');
    await expect(readout(page, 'kind-error')).toHaveText('0/5');
    await expect(page.locator('.toy-transcript [data-kind="error"][data-trained="false"]')).toHaveCount(5);
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK_MASK_ERROR);
    await expect(page.locator('ol.try-this li').nth(2)).toContainText(TRY_THIS[2].prompt);
  });

  test('the template chips change tag text and never the counts', async ({ page }) => {
    const before = await chips(page).evaluateAll((els) => els.map((e) => e.getAttribute('aria-label')));
    await page.locator('#template .choice-option[data-value="harmony"]').click();
    await expect(readout(page, 'template-note')).toContainText('harmony (System > Developer > User > Assistant > Tool)');
    await expect(readout(page, 'trained')).toHaveText('15 of 26');
    await expect(chips(page)).toHaveCount(26);
    const after = await chips(page).evaluateAll((els) => els.map((e) => e.getAttribute('aria-label')));
    expect(after).not.toEqual(before);
    await page.locator('#template .choice-option[data-value="deepseek"]').click();
    await expect(readout(page, 'template-note')).toContainText('|DSML| XML tool calls');
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK);
  });

  test('selecting a token (click, then keyboard) says what it is and whether it trains', async ({ page }) => {
    await chips(page).nth(0).click();
    await expect(readout(page, 'selection')).toHaveText('Selected "<user>", a template tag: masked (context only).');
    await chips(page).nth(0).press('ArrowRight');
    await expect(readout(page, 'selection')).toHaveText('Selected "What", a user token: masked (context only).');
    await expect(chips(page).nth(1)).toBeFocused();
    await chips(page).nth(1).press('End');
    await expect(readout(page, 'selection')).toHaveText('Selected "<end>", an assistant token: trained.');
    await page.keyboard.press('Home');
    await page.keyboard.press('Enter');
    await expect(readout(page, 'selection')).toHaveText('Selected "<user>", a template tag: masked (context only).');
  });
});
