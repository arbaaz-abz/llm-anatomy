import { test, expect } from '@playwright/test';
import { MID_MS, END_MS, MID_PROGRESS, startPausedClock, stageState, animateTo, glyphOverflows, stageOverflows, accentStrokeOffenders } from './lesson-helpers.js';

const ROOT = '#stepper-root';

test('the stepper reports its frame on the stage at rest', async ({ page }) => {
  await page.goto('/gallery/');
  const s = await stageState(page, ROOT);
  expect([s.step, s.progress]).toEqual([0, 1]);
});

test('clock sampling through the public UI lands exactly on the 16 ms frame grid', async ({ page }) => {
  await startPausedClock(page, '/gallery/');
  const mid = await animateTo(page, ROOT, 4, MID_MS);
  expect(mid.step).toBe(4);
  expect(mid.progress).toBe(MID_PROGRESS); // on the 16 ms grid, so exact
  await page.clock.runFor(END_MS - MID_MS);
  expect((await stageState(page, ROOT)).progress).toBe(1);
});

test('two routes to one step sample the same frame mid-transition and at rest', async ({ page }) => {
  await startPausedClock(page, '/gallery/');
  const a = await animateTo(page, ROOT, 2, MID_MS, 1);
  await page.clock.runFor(END_MS - MID_MS);
  const aEnd = await stageState(page, ROOT);
  const b = await animateTo(page, ROOT, 2, MID_MS, 4);
  await page.clock.runFor(END_MS - MID_MS);
  const bEnd = await stageState(page, ROOT);
  expect(b.progress).toBe(a.progress);
  expect(b.html).toBe(a.html);
  expect([aEnd.progress, bEnd.progress]).toEqual([1, 1]);
  expect(bEnd.html).toBe(aEnd.html);
});

test('overflow and accent helpers agree with the gallery as shipped', async ({ page }) => {
  await page.goto('/gallery/');
  await page.evaluate(() => document.fonts.ready);
  expect(await glyphOverflows(page, '#figures svg')).toEqual([]);
  expect(await accentStrokeOffenders(page, '#figures')).toEqual([]);
});

test('stageOverflows: nothing drawn on the gallery stepper stage clips, at mid-transition or at rest, on any step', async ({ page }) => {
  await startPausedClock(page, '/gallery/');
  await page.evaluate(() => document.fonts.ready);
  const svg = `${ROOT} .stepper-stage svg`;
  for (let k = 0; k < 5; k += 1) {
    await animateTo(page, ROOT, k, MID_MS);
    expect(await stageOverflows(page, svg), `step ${k + 1} mid`).toEqual([]);
    await page.clock.runFor(END_MS - MID_MS);
    expect(await stageOverflows(page, svg), `step ${k + 1} end`).toEqual([]);
  }
});

test('stageOverflows: nothing drawn in any gallery figure clips its svg', async ({ page }) => {
  await page.goto('/gallery/');
  await page.evaluate(() => document.fonts.ready);
  expect(await stageOverflows(page, '#figures svg')).toEqual([]);
});

test('stageOverflows catches a label drawn straight on the svg past its edge (not only glyphs)', async ({ page }) => {
  await page.goto('/gallery/');
  await page.evaluate(() => {
    const svg = document.querySelector('#figures svg');
    const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    t.setAttribute('x', '-30'); t.setAttribute('y', '12'); t.textContent = 'keys →';
    svg.append(t);
  });
  expect(await glyphOverflows(page, '#figures svg')).toEqual([]);
  expect((await stageOverflows(page, '#figures svg')).join()).toContain('text "keys →"');
});
