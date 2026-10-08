import { test, expect } from '@playwright/test';
import { registerLessonContract, LESSON_STEPPER, stepperParts } from './lesson-helpers.js';
import { CAPTIONS } from '../tests/long-context-attention-expected.js';

const URL = '/architecture/#long-context-attention';

registerLessonContract({ name: 'long-context-attention', url: URL, captions: CAPTIONS, factRows: 12, returnHash: 'long-context-attention' });

const readout = (page, name) => page.locator(`[data-readout="${name}"]`);
const cells = (page, name) => page.locator(`[data-readout="${name}"] .cell`);
const choose = (page, id, value) => page.locator(`#${id} [data-value="${value}"]`).click();
const setStop = (page, id, index) => page.locator(`#${id}`).fill(String(index)); // snapped sliders: the input holds a stop index
const stageText = (page) => page.locator(`${stepperParts(LESSON_STEPPER).stage} svg`).evaluate((svg) => svg.textContent.replace(/\s+/g, ' '));
const goToStep = async (page, k) => {
  await page.locator(stepperParts(LESSON_STEPPER).scrub).fill(String(k));
  await expect(page.locator(stepperParts(LESSON_STEPPER).count)).toHaveText(`${k + 1} / ${CAPTIONS.length}`);
  await expect(page.locator(`${stepperParts(LESSON_STEPPER).stage}[data-progress="1"]`)).toHaveCount(1);
};

test.describe('long-context-attention animation: the counters on the stage', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('frame 1 counts 16 reads, 16 stored and 136 cells; frame 6 reads 4 but still stores 16; frame 7 stores 8 and reads 5', async ({ page }) => {
    await goToStep(page, 0);
    let text = await stageText(page);
    expect(text).toContain('16 entries');
    expect(text).toContain('136');
    await goToStep(page, 5);
    text = await stageText(page);
    expect(text).toContain('reads (compute): 4 entries');
    expect(text).toContain('stored (memory): 16 entries');
    await goToStep(page, 6);
    text = await stageText(page);
    expect(text).toContain('reads (compute): 5 entries');
    expect(text).toContain('stored (memory): 8 entries');
  });

  test('frame 6 keeps its question on screen at rest, with the answer under it', async ({ page }) => {
    await goToStep(page, 5);
    const text = await stageText(page);
    expect(text).toContain('Did the cache shrink?');
    expect(text).toContain('No: it still stores 16.');
  });

  test('frame 8 prints the state counter and the shape-not-size line; frame 4 ends on the sink split', async ({ page }) => {
    await goToStep(page, 7);
    const text = await stageText(page);
    expect(text).toContain('cache after 16 tokens: 128 numbers');
    expect(text).toContain('compare the shape');
    await goToStep(page, 3);
    expect(await stageText(page)).toContain('Σ = 1.000');
  });
});

test.describe('long-context-attention toy: what does each trick read, and what does it keep?', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('opens on full attention: token 16 reads 16, 136 cells read, 16 stored; the real-scale table shows full attention at 1,048,576', async ({ page }) => {
    await expect(readout(page, 'reads')).toHaveText('16');
    await expect(readout(page, 'cells')).toHaveText('136');
    await expect(readout(page, 'stored')).toHaveText('16');
    await expect(readout(page, 'real-reads')).toHaveText('1,048,576');
    await expect(readout(page, 'real-stored')).toHaveText('1,048,576');
    await expect(page.locator('[data-section="toy"] .g-heatmap')).toHaveCount(1);
  });

  test('try this 1: sparse reads 4 and stores 16; compressed (merge 4, top 1) reads 5 and stores 8; GLM-5.3 reads 2,048 and stores 1,048,576; V4-Pro CSA reads 1,152 and stores 262,272', async ({ page }) => {
    await choose(page, 'pattern', 'sparse');
    await expect(readout(page, 'reads')).toHaveText('4');
    await expect(readout(page, 'stored')).toHaveText('16');
    await expect(readout(page, 'cells')).toHaveText('58');
    await choose(page, 'pattern', 'compressed');
    await setStop(page, 'topK', 0);
    await expect(readout(page, 'reads')).toHaveText('5');
    await expect(readout(page, 'stored')).toHaveText('8');
    await choose(page, 'real', 'glm-5.3');
    await expect(readout(page, 'real-reads')).toHaveText('2,048');
    await expect(readout(page, 'real-stored')).toHaveText('1,048,576');
    await choose(page, 'real', 'deepseek-v4-pro');
    await expect(readout(page, 'real-reads')).toHaveText('1,152');
    await expect(readout(page, 'real-stored')).toHaveText('262,272');
    await expect(readout(page, 'real-reads-hca')).toHaveText('8,320');
  });

  test('controls that do not apply are hidden, not disabled', async ({ page }) => {
    await expect(page.locator('#window')).toBeHidden();
    await expect(page.locator('#topK')).toBeHidden();
    await expect(page.locator('#merge')).toBeHidden();
    await choose(page, 'pattern', 'compressed');
    await expect(page.locator('#window')).toBeVisible();
    await expect(page.locator('#topK')).toBeVisible();
    await expect(page.locator('#merge')).toBeVisible();
    await choose(page, 'pattern', 'sink');
    await expect(page.locator('#sinkLogit')).toBeVisible();
    await expect(page.locator('#merge')).toBeHidden();
  });

  test('try this 2: the sink takes 0.600 at logit 1; at −2 it takes 0.069', async ({ page }) => {
    await choose(page, 'pattern', 'sink');
    await expect(cells(page, 'sink-weights')).toHaveText(['0.600', '0.081', '0.134', '0.081', '0.104']);
    await setStop(page, 'sinkLogit', 0);
    await expect(cells(page, 'sink-weights')).toHaveText(['0.069', '0.189', '0.311', '0.189', '0.242']);
  });

  test('try this 3: linear attention reads [−1.5, 6, 1.5, 3.25]; gate 0.5 gives [−0.75, 3, 0.75, 1.75]; the state stays 16 numbers', async ({ page }) => {
    await choose(page, 'pattern', 'linear');
    await expect(cells(page, 'linear-output')).toHaveText(['−1.5', '6', '1.5', '3.25']);
    await expect(page.locator('[data-section="toy"] .g-matrix .g-text')).toHaveText(['1.5', '−0.5', '−1', '0', '0', '3', '0', '−1', '−1.5', '0.5', '1', '0', '−0.25', '1.5', '0.5', '−0.25']);
    await setStop(page, 'gate', 1);
    await expect(cells(page, 'linear-output')).toHaveText(['−0.75', '3', '0.75', '1.75']);
    await expect(page.locator('[data-section="toy"] .g-matrix .g-text')).toHaveCount(16);
  });

  test('the try-this list is printed inside the toy, numbered, each with a bold insight', async ({ page }) => {
    const items = page.locator('[data-section="toy"] ol.try-this li');
    await expect(items).toHaveCount(3);
    await expect(items.nth(0)).toContainText('DSA reads 2,048 and stores 1,048,576; CSA reads 1,152 and stores 262,272.');
    await expect(items.nth(2).locator('strong')).toContainText('Insight: a linear layer trades exact lookup');
  });

  test('the sink row shows only in sink mode and the linear row only in linear mode', async ({ page }) => {
    await expect(readout(page, 'sink-weights')).toBeHidden();
    await expect(readout(page, 'linear-output')).toBeHidden();
    await choose(page, 'pattern', 'sink');
    await expect(readout(page, 'sink-weights')).toBeVisible();
    await expect(readout(page, 'linear-output')).toBeHidden();
    await choose(page, 'pattern', 'linear');
    await expect(readout(page, 'sink-weights')).toBeHidden();
    await expect(readout(page, 'linear-output')).toBeVisible();
  });

  test('the linear note follows the gate', async ({ page }) => {
    await choose(page, 'pattern', 'linear');
    await expect(page.locator('[data-section="toy"]')).toContainText('gate 1 keeps every addition');
    await setStop(page, 'gate', 1);
    await expect(page.locator('[data-section="toy"]')).toContainText('gate 0.5 halves the state before each new token');
  });

  test('the followed token is a slider: token 8 of a window reads its own last 4', async ({ page }) => {
    await choose(page, 'pattern', 'window');
    await page.locator('#query').fill('8');
    await expect(readout(page, 'reads')).toHaveText('4');
    await page.locator('#query').fill('2');
    await expect(readout(page, 'reads')).toHaveText('2');
  });

  test('real scale: gpt-oss stores 4.84 GB at 131,072 tokens; Qwen3.5 prints its growing part with a fixed-state note; V4-Pro prints a range labeled estimate', async ({ page }) => {
    await setStop(page, 'context', 0);
    await choose(page, 'real', 'gpt-oss');
    await expect(readout(page, 'real-cache')).toHaveText('4.84 GB');
    await choose(page, 'real', 'qwen3.5');
    await setStop(page, 'context', 1);
    await expect(readout(page, 'real-cache')).toHaveText('32.2 GB');
    await expect(page.locator('[data-section="toy"]')).toContainText('fixed state per linear layer');
    await choose(page, 'real', 'deepseek-v4-pro');
    await expect(page.locator('[data-section="toy"]')).toContainText('estimate');
  });
});
