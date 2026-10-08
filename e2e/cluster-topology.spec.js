// cluster-topology e2e: the lesson contract plus the toy "Put a cut on a link" (storyboard §6), driven by its control ids.
import { test, expect } from '@playwright/test';
import { registerLessonContract } from './lesson-helpers.js';
import { collectConsoleErrors } from './helpers.js';
import { CAPTIONS, CHECK_WORK, TRY_THIS } from '../tests/cluster-topology-expected.js';

const URL = '/training/#cluster-topology';
const STEPPER = '[data-section="animation"] .stepper';

registerLessonContract({ name: 'cluster-topology', url: URL, captions: CAPTIONS, factRows: 11, returnHash: 'cluster-topology' });

const readout = (page, name) => page.locator(`[data-readout="${name}"]`);
const pick = (page, control, value) => page.locator(`#${control} [data-value="${value}"]`).click();
const stop = (page, control, index) => page.locator(`#${control}`).fill(String(index));
const goTo = (page, index) => page.locator(`${STEPPER} input[type="range"]`).fill(String(index));
// slider stops: tensor 2,4,8,16,32,64 · pipeline 2,4,8,16,32 · data 2…1,024 · tokens 4,096 … 1,048,576
const TENSOR = { 8: 2, 16: 3, 64: 5 };
const PIPELINE = { 16: 3 };
const DATA = { 64: 5 };
const TOKENS = { 16384: 2, 262144: 6 };

test.describe('cluster-topology animation: page text under the stage', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('frame 1 and frame 2 print the each-way convention and the 9× ladder', async ({ page }) => {
    await expect(page.locator('.below-stage')).toContainText('NVIDIA quotes 900 GB/s per GPU for H100 NVLink, counting both directions; this page prints each way.');
    await goTo(page, 1);
    await expect(page.locator('.below-stage')).toContainText('HBM bandwidth is a separate total and is not on this ladder');
  });

  test('frame 4 states the stand-in tokens per replica and frame 6 explains why data sits outermost', async ({ page }) => {
    await goTo(page, 3);
    await expect(page.locator('.below-stage')).toContainText('Tokens per replica per step is a stand-in (262,144); the ratio rises to 79.2% at 16,384 tokens.');
    await goTo(page, 5);
    await expect(page.locator('.below-stage')).toContainText('Data parallelism sends more per unit of compute than pipeline here (5.0% against 1.5%), but its sync overlaps with the backward pass');
  });
});

test.describe('cluster-topology toy: put a cut on a link', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('opens on tensor 8, H100 HGX, inside: 27.8%, 352 MB, and the exact "Check my work" box', async ({ page }) => {
    await expect(readout(page, 'ratio')).toHaveText('27.8%');
    await expect(readout(page, 'bytes')).toHaveText('352 MB');
    await expect(readout(page, 'hide')).toHaveText('no: on the critical path');
    await expect(readout(page, 'lane-note')).toHaveText('both lanes in the same time units');
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK);
    await expect(page.locator('[data-section="toy"]')).toContainText('One full training step (forward and backward).');
    await expect(page.locator('[data-section="toy"] ol.try-this li')).toHaveText(TRY_THIS);
  });

  test('try this 1: tensor 8 inside 27.8%, network 250.4%; 16 runs over the network (536.6%); NVL72 35.2% and 75.4%', async ({ page }) => {
    await pick(page, 'where', 'network');
    await expect(readout(page, 'ratio')).toHaveText('250.4%');
    await expect(page.locator('[data-section="toy"] .g-lane-cap')).toHaveCount(0);
    await stop(page, 'degree', TENSOR[16]);
    await expect(readout(page, 'ratio')).toHaveText('536.6%');
    await expect(page.locator('[data-section="toy"] .g-lane-cap')).toHaveText('continues: 537%');
    await pick(page, 'system', 'gb200');
    await stop(page, 'degree', TENSOR[8]);
    await pick(page, 'where', 'inside');
    await expect(readout(page, 'ratio')).toHaveText('35.2%');
    await stop(page, 'degree', TENSOR[16]);
    await expect(readout(page, 'ratio')).toHaveText('75.4%');
  });

  test('"inside" is disabled with a visible note once the degree passes the NVLink domain, and comes back on NVL72', async ({ page }) => {
    const inside = page.locator('#where [data-value="inside"]');
    await expect(page.locator('#where [data-choice-note]')).toHaveCount(0);
    await stop(page, 'degree', TENSOR[16]);
    await expect(inside).toBeDisabled();
    await expect(page.locator('#where [data-choice-note]')).toHaveText('inside: this cut no longer fits in one NVLink domain');
    await expect(page.locator('#where [data-value="network"]')).toHaveAttribute('aria-pressed', 'true');
    await pick(page, 'system', 'gb200');
    await expect(inside).toBeEnabled();
    await expect(page.locator('#where [data-choice-note]')).toHaveCount(0);
  });

  test('try this 2: data 64 over the network 5.0% at 262,144 tokens, 79.2% at 16,384; pipeline 16 is 1.5%', async ({ page }) => {
    await pick(page, 'cut', 'data');
    await stop(page, 'degree', DATA[64]);
    await pick(page, 'where', 'network');
    await expect(readout(page, 'ratio')).toHaveText('5.0%');
    await expect(page.locator('#tokens')).toHaveValue(String(TOKENS[262144]));
    await stop(page, 'tokens', TOKENS[16384]);
    await expect(readout(page, 'ratio')).toHaveText('79.2%');
    await expect(readout(page, 'hide')).toHaveText('yes, during backward');
    await pick(page, 'cut', 'pipeline');
    await stop(page, 'degree', PIPELINE[16]);
    await expect(readout(page, 'ratio')).toHaveText('1.5%');
    await expect(readout(page, 'hide')).toHaveText('mostly, in 1F1B\'s steady state');
    await expect(page.locator('#tokens')).toBeHidden();
  });

  test('try this 3: expert on H100 needs 161 GB/s (35.8% inside, 321.9% over the network); on NVL72 407 GB/s (45.2%, 406.9%)', async ({ page }) => {
    await pick(page, 'cut', 'expert');
    await expect(page.locator('#degree')).toBeHidden();
    await expect(readout(page, 'ratio')).toHaveText('35.8%');
    await expect(readout(page, 'link-needed')).toHaveText('161 GB/s');
    await expect(readout(page, 'hide')).toHaveText('only below the hiding line');
    await pick(page, 'where', 'network');
    await expect(readout(page, 'ratio')).toHaveText('321.9%');
    await pick(page, 'system', 'gb200');
    await expect(readout(page, 'link-needed')).toHaveText('407 GB/s');
    await expect(readout(page, 'ratio')).toHaveText('406.9%');
    await pick(page, 'where', 'inside');
    await expect(readout(page, 'ratio')).toHaveText('45.2%');
  });

  test('a lane longer than the track is cut at 4.33× compute with its percentage printed', async ({ page }) => {
    const errors = collectConsoleErrors(page);
    await stop(page, 'degree', TENSOR[64]);
    await expect(readout(page, 'ratio')).toHaveText('2,253.6%');
    await expect(readout(page, 'lane-note')).toHaveText('both lanes in the same time units; comm lane cut at 4.33× compute');
    await expect(page.locator('[data-section="toy"] .g-lane-cap')).toHaveText('continues: 2,254%');
    expect(errors).toEqual([]);
  });
});
