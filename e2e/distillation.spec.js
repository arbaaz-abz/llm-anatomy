import { test, expect } from '@playwright/test';
import { registerLessonContract } from './lesson-helpers.js';
import { collectConsoleErrors } from './helpers.js';
import { CAPTIONS, CHECK_WORK, TRY_THIS } from '../tests/distillation-expected.js';

const URL = '/training/#distillation';

registerLessonContract({ name: 'distillation', url: URL, captions: CAPTIONS, factRows: 7, returnHash: 'distillation' });

const readout = (page, name) => page.locator(`[data-readout="${name}"]`);
const toySvg = (page) => page.locator('[data-section="toy"] svg.toy-stage');
const sampledCell = (page, i) => toySvg(page).locator(`[data-sampled="${i}"]`);
const choose = (page, id, value) => page.locator(`#${id} [data-value="${value}"]`).click();
const rewardRow = (page) => page.locator('table[data-readout="rewards"] tbody tr');

test.describe('distillation toy: grade the student', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('opens on on-policy, unsure, the math teacher and sampled 54, with the exact "Check my work" box', async ({ page }) => {
    await expect(readout(page, 'sampled-reward')).toHaveText('−1.792');
    await expect(readout(page, 'expected-reward')).toHaveText('−0.754');
    await expect(readout(page, 'multi-teacher-loss')).toHaveText('0.754');
    await expect(rewardRow(page)).toHaveText(['56 +0.811', '54 −1.792', '48 −1.897', '63 −1.609'], { useInnerText: true });
    await expect(readout(page, 'check-work')).toHaveText(CHECK_WORK);
    await expect(sampledCell(page, 1)).toHaveAttribute('aria-pressed', 'true');
    await expect(toySvg(page).locator('.g-select')).toHaveCount(2); // the sampled student cell and its reward cell
  });

  test('the try-this list is the storyboard\'s, with its insights', async ({ page }) => {
    const items = page.locator('[data-section="toy"] ol.try-this li');
    await expect(items).toHaveText(TRY_THIS);
  });

  test('try this 1: each sampled token gets its own grade (click and keyboard)', async ({ page }) => {
    await sampledCell(page, 0).click();
    await expect(readout(page, 'sampled-reward')).toHaveText('+0.811');
    await sampledCell(page, 2).click();
    await expect(readout(page, 'sampled-reward')).toHaveText('−1.897');
    await sampledCell(page, 3).click();
    await expect(readout(page, 'sampled-reward')).toHaveText('−1.609');
    await expect(readout(page, 'expected-reward')).toHaveText('−0.754');
  });

  test('the sampled student cells are keyboard buttons: Tab, arrows move focus, Enter and Space select', async ({ page }) => {
    const errors = collectConsoleErrors(page);
    for (let i = 0; i < 4; i += 1) {
      await expect(sampledCell(page, i)).toHaveAttribute('role', 'button');
      await expect(sampledCell(page, i)).toHaveAttribute('tabindex', '0');
    }
    await expect(sampledCell(page, 0)).toHaveAttribute('aria-label', 'sampled: 56');
    await sampledCell(page, 1).focus();
    await page.keyboard.press('ArrowRight');
    await expect(sampledCell(page, 2)).toBeFocused();
    await page.keyboard.press('Enter');
    await expect(readout(page, 'sampled-reward')).toHaveText('−1.897');
    await expect(sampledCell(page, 2)).toBeFocused(); // the repaint keeps focus
    await page.keyboard.press('ArrowLeft');
    await page.keyboard.press('ArrowLeft');
    await expect(sampledCell(page, 0)).toBeFocused();
    await page.keyboard.press('Space');
    await expect(readout(page, 'sampled-reward')).toHaveText('+0.811');
    await page.keyboard.press('ArrowLeft'); // stops at the first cell
    await expect(sampledCell(page, 0)).toBeFocused();
    expect(errors).toEqual([]);
  });

  test('try this 2: the signal shrinks as the student nears the teacher and grows when it is confidently wrong', async ({ page }) => {
    await choose(page, 'student', 'near');
    await expect(rewardRow(page)).toHaveText(['56 +0.057', '54 −0.336', '48 −0.511', '63 −0.405'], { useInnerText: true });
    await expect(readout(page, 'sampled-reward')).toHaveText('−0.336');
    await expect(readout(page, 'expected-reward')).toHaveText('−0.013');
    await choose(page, 'student', 'wrong');
    await expect(readout(page, 'sampled-reward')).toHaveText('−2.639');
    await expect(readout(page, 'expected-reward')).toHaveText('−1.909');
  });

  test('try this 3: the multi-teacher loss for a student that matches the math teacher', async ({ page }) => {
    await choose(page, 'student', 'near');
    await expect(readout(page, 'multi-teacher-loss')).toHaveText('0.013');
    await choose(page, 'teacher', 'chat');
    await expect(readout(page, 'multi-teacher-loss')).toHaveText('0.152');
    await choose(page, 'teacher', 'mix');
    await expect(readout(page, 'multi-teacher-loss')).toHaveText('0.082');
    await expect(toySvg(page).locator('[data-teacher-row]')).toHaveCount(2);
  });

  test('the other two methods print their own loss and hide the sampled-token control', async ({ page }) => {
    await choose(page, 'method', 'traces');
    await expect(readout(page, 'trace-loss')).toHaveText('0.916');
    await expect(sampledCell(page, 1)).toHaveCount(0);
    await expect(readout(page, 'check-work')).toHaveText('loss = −ln p_student(56) = −ln 0.40 = 0.916');
    await choose(page, 'method', 'logits');
    await expect(readout(page, 'forward-kl')).toHaveText('0.551');
    await expect(readout(page, 'sampled-reward')).toBeHidden();
    await choose(page, 'method', 'onPolicy');
    await expect(sampledCell(page, 1)).toHaveAttribute('aria-pressed', 'true'); // the choice survives the detour
  });
});
