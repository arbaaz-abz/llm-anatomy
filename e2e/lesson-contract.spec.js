import { test, expect } from '@playwright/test';
import { registerLessonContract, LESSON_STEPPER, MID_MS, MID_PROGRESS, startPausedClock, animateTo } from './lesson-helpers.js';

const FIXTURE = '/e2e/fixtures/lesson-fixture.html';
const CAPTIONS = [
  'Four tokens arrive. We follow "sat", token 3.',
  'The query row of "sat" lifts out so its numbers can be read.',
  'The row slides back into place, still marked as the one we follow.',
];

registerLessonContract({ name: 'fixture', url: `${FIXTURE}#fixture`, captions: CAPTIONS, factRows: 3, leaveHash: 'away', returnHash: 'fixture' });

test('fixture: clock sampling through the public UI equals the pure render at the same (step, progress)', async ({ page }) => {
  await startPausedClock(page, `${FIXTURE}#fixture`);
  await page.evaluate(() => document.fonts.ready);
  const mid = await animateTo(page, LESSON_STEPPER, 1, MID_MS);
  expect(mid.progress).toBe(MID_PROGRESS); // on the 16 ms grid, so exact
  // Oracle: the fixture's own exported pure render at the same (index, progress). No page hook.
  const oracle = await page.evaluate(async (p) => {
    const { render } = await import('./lesson-fixture.js');
    const stage = document.querySelector('[data-section="animation"] .stepper-stage');
    render(1, p, stage);
    return stage.innerHTML;
  }, MID_PROGRESS);
  expect(oracle).toBe(mid.html);
});

test('fixture: sections render in spec §4 order with rich text, stand-in line and below-stage text', async ({ page }) => {
  await page.goto(`${FIXTURE}#fixture`);
  const order = await page.locator('#lesson > [data-section]').evaluateAll((els) => els.map((e) => e.dataset.section));
  expect(order).toEqual(['hook', 'intuition', 'animation', 'toy', 'math', 'facts', 'takeaways', 'links']);
  await expect(page.locator('[data-section="intuition"] a code')).toHaveText('attention');
  await expect(page.locator('.stand-in')).toHaveText('These numbers are hand-picked stand-ins.');
  await page.locator('[data-section="animation"] input[type="range"]').fill('2');
  await expect(page.locator('.below-stage')).toHaveText('Page text under the stage on step 3.');
  await expect(page.locator('[data-section="takeaways"] li')).toHaveCount(3);
  await expect(page.locator('[data-section="facts"] .fact-row').first()).toContainText('1.6T parameters');
  await expect(page.locator('[data-section="facts"] .fact-row').nth(2)).toContainText('accepts 85–90%'); // sv: reads serving.json
});

test('fixture: the toy renders live from one state object', async ({ page }) => {
  await page.goto(`${FIXTURE}#fixture`);
  const out = page.locator('[data-readout="result"]');
  await expect(out).toHaveText('8');
  await page.locator('#fixture-n').fill('3');
  await expect(out).toHaveText('16');
  await page.locator('#fixture-mode [data-value="square"]').click();
  await expect(out).toHaveText('64');
  await expect(page.locator('#fixture-mode [data-value="square"]')).toHaveAttribute('aria-pressed', 'true');
});

test('fixture: a toy that throws shows its own error card and the animation still works', async ({ page }) => {
  await page.goto(`${FIXTURE}#broken-toy`);
  await expect(page.locator('[data-section="toy"] .load-error')).toBeVisible();
  await page.locator('[data-section="animation"] [data-act="next"]').click();
  await expect(page.locator('[data-section="animation"] .stepper-count')).toHaveText('2 / 3');
});
