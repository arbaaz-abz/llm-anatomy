// Shared e2e helpers for every lesson page and the gallery. FROZEN after Plan 2 wave 0.
import { test, expect } from '@playwright/test';
import { collectConsoleErrors } from './helpers.js';

export const TRANSITION_MS = 750;
export const MID_MS = 368; // 23 frames of the fake clock's 16 ms grid
export const END_MS = 752; // 47 frames: the 750 ms transition has finished
export const CLOCK_START_MS = Date.UTC(2026, 9, 7); // non-zero (a 0 start can read as "now"); a multiple of 16
export const ALIGN_AT_MS = CLOCK_START_MS + 60_000; // 60 000 is a multiple of 16: transitions start on the frame grid
export const easeInOut = (p) => (p < 0.5 ? 2 * p * p : 1 - (-2 * p + 2) ** 2 / 2);
export const MID_PROGRESS = easeInOut(MID_MS / TRANSITION_MS);
export const STAGE_MAX = { width: 580, height: 366 };

export const stepperParts = (root) => ({
  stage: `${root} .stepper-stage`,
  caption: `${root} .stepper-caption`,
  count: `${root} .stepper-count`,
  scrub: `${root} input[type="range"]`,
  toggle: `${root} [data-act="toggle"]`,
  next: `${root} [data-act="next"]`,
  prev: `${root} [data-act="prev"]`,
});

// Every glyph (bbox grown by 2 px for strokes) must sit inside its own <svg>.
export function glyphOverflows(page, svgSelector) {
  return page.evaluate((sel) => {
    const out = [];
    for (const svg of document.querySelectorAll(sel)) {
      const box = svg.getBoundingClientRect();
      for (const el of svg.querySelectorAll('.glyph')) {
        const r = el.getBoundingClientRect();
        if (r.width === 0 && r.height === 0) continue;
        const pad = 2;
        const outside = r.left - pad < box.left - 0.5 || r.top - pad < box.top - 0.5
          || r.right + pad > box.right + 0.5 || r.bottom + pad > box.bottom + 0.5;
        if (outside) out.push(`${el.getAttribute('class')} in ${svg.getAttribute('aria-labelledby') ?? svg.getAttribute('aria-label') ?? sel}`);
      }
    }
    return out;
  }, svgSelector);
}

// Accent strokes are selection marks only, never wider than the 2 px focus ring.
export function accentStrokeOffenders(page, scopeSelector, maxWidth = 2) {
  return page.evaluate(({ scope, max }) => {
    const root = document.querySelector(scope) ?? document.body;
    const probe = document.createElement('i');
    probe.style.color = 'var(--accent)';
    root.append(probe);
    const accent = getComputedStyle(probe).color;
    probe.remove();
    const offenders = [];
    for (const el of root.querySelectorAll('svg *')) {
      const cs = getComputedStyle(el);
      if (cs.stroke === accent && parseFloat(cs.strokeWidth) > max) offenders.push(`${el.tagName}.${el.getAttribute('class')} ${cs.strokeWidth}`);
    }
    return offenders;
  }, { scope: scopeSelector, max: maxWidth });
}

export const pageOverflowX = (page) =>
  page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);

// Fake clock: install at a fixed start, load, then pause on the 16 ms grid. Time moves only via runFor.
export async function startPausedClock(page, url) {
  await page.clock.install({ time: CLOCK_START_MS });
  await page.goto(url);
  await page.clock.pauseAt(ALIGN_AT_MS);
}

export function stageState(page, root) {
  return page.locator(stepperParts(root).stage).evaluate((s) => ({
    step: Number(s.dataset.step), progress: Number(s.dataset.progress), html: s.innerHTML,
  }));
}

// Seek through the stepper's own range input, exactly as a drag does, then let the clock run.
// (Locator clicks wait on requestAnimationFrame for stability, which a paused fake clock never fires.)
export async function seekAndRun(page, root, index, ms) {
  await page.locator(stepperParts(root).scrub).evaluate((input, i) => {
    input.value = String(i);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  }, index);
  await page.clock.runFor(ms);
  return stageState(page, root);
}

// Reach `index` from another step (`from`, default the previous one) so its transition runs, sampled at `ms`.
export async function animateTo(page, root, index, ms, from = index === 0 ? 1 : index - 1) {
  const current = (await stageState(page, root)).step;
  if (current !== from) await seekAndRun(page, root, from, END_MS);
  const state = await seekAndRun(page, root, index, ms);
  expect(state.step).toBe(index);
  return state;
}

export const LESSON_STEPPER = '[data-section="animation"] .stepper';

async function openSettled(page, url, root) {
  await page.goto(url);
  await expect(page.locator(stepperParts(root).stage)).toHaveAttribute('data-progress', '1');
  await page.evaluate(() => document.fonts.ready);
}

function captionTests({ url, captions, root }) {
  test('every caption renders verbatim, in order, with no console errors', async ({ page }) => {
    const errors = collectConsoleErrors(page);
    const p = stepperParts(root);
    await openSettled(page, url, root);
    for (let k = 0; k < captions.length; k += 1) {
      await page.locator(p.scrub).fill(String(k));
      await expect(page.locator(p.count)).toHaveText(`${k + 1} / ${captions.length}`);
      await expect(page.locator(p.caption)).toHaveText(captions[k]);
    }
    expect(errors).toEqual([]);
  });
}

function determinismTests({ url, captions, root }) {
  test('every step is a pure function of (step, progress): two routes give the same mid and end frames', async ({ page }) => {
    const last = captions.length - 1;
    await startPausedClock(page, url);
    for (let k = 0; k <= last; k += 1) {
      const a = await animateTo(page, root, k, MID_MS, k === 0 ? Math.min(1, last) : k - 1);
      await page.clock.runFor(END_MS - MID_MS);
      const aEnd = await stageState(page, root);
      const b = await animateTo(page, root, k, MID_MS, k === last ? 0 : last);
      await page.clock.runFor(END_MS - MID_MS);
      const bEnd = await stageState(page, root);
      expect(a.progress, `step ${k + 1}: mid-transition sample`).toBe(MID_PROGRESS);
      expect(b.html, `step ${k + 1}: mid frame differs by route`).toBe(a.html);
      expect([aEnd.progress, bEnd.progress]).toEqual([1, 1]);
      expect(bEnd.html, `step ${k + 1}: end frame differs by route`).toBe(aEnd.html);
    }
  });
}

function layoutTests({ url, captions, root }) {
  test('no glyph clips at mid-transition or at rest on any step; the stage fits 580 × 366', async ({ page }) => {
    const stageSvg = `${stepperParts(root).stage} svg`;
    await startPausedClock(page, url);
    await page.evaluate(() => document.fonts.ready);
    for (let k = 0; k < captions.length; k += 1) {
      await animateTo(page, root, k, MID_MS);
      expect(await glyphOverflows(page, stageSvg), `step ${k + 1} mid`).toEqual([]);
      await page.clock.runFor(END_MS - MID_MS);
      expect(await glyphOverflows(page, stageSvg), `step ${k + 1} end`).toEqual([]);
      const size = await page.locator(stageSvg).first().evaluate((s) => ({ width: Number(s.getAttribute('width')), height: Number(s.getAttribute('height')) }));
      expect(size.width, `step ${k + 1} stage width`).toBeLessThanOrEqual(STAGE_MAX.width);
      expect(size.height, `step ${k + 1} stage height`).toBeLessThanOrEqual(STAGE_MAX.height);
      expect(await accentStrokeOffenders(page, '.concept'), `step ${k + 1} accent strokes`).toEqual([]);
    }
    expect(await glyphOverflows(page, '[data-section="toy"] svg')).toEqual([]);
  });

  test('no horizontal page scroll at 400 px with the math panel open', async ({ page }) => {
    await page.setViewportSize({ width: 400, height: 860 }); // builders run desktop only; this check must not depend on the project
    await openSettled(page, url, root);
    await page.locator('[data-section="math"] summary').click();
    await page.evaluate(() => document.fonts.ready);
    expect(await pageOverflowX(page)).toBeLessThanOrEqual(0);
  });
}

function controlTests({ url, captions, root }) {
  test('keyboard: arrows step, space plays and pauses, focus is visible', async ({ page }) => {
    const p = stepperParts(root);
    const n = captions.length;
    await openSettled(page, url, root);
    await page.locator(root).focus();
    await page.keyboard.press('ArrowRight');
    await expect(page.locator(p.count)).toHaveText(`2 / ${n}`);
    await page.keyboard.press('ArrowLeft');
    await expect(page.locator(p.count)).toHaveText(`1 / ${n}`);
    await page.keyboard.press('Space');
    await expect(page.locator(p.toggle)).toHaveText('Pause');
    await page.keyboard.press('Space');
    await expect(page.locator(p.toggle)).toHaveText('Play');
    await page.locator(p.next).focus();
    expect(await page.locator(p.next).evaluate((b) => getComputedStyle(b).outlineStyle)).not.toBe('none');
  });

  test('reduced motion: a step change renders its final frame within one animation frame', async ({ page }) => {
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await openSettled(page, url, root);
    const frame = await page.evaluate((r) => new Promise((resolve) => {
      document.querySelector(`${r} [data-act="next"]`).click();
      requestAnimationFrame(() => {
        const s = document.querySelector(`${r} .stepper-stage`);
        resolve({ step: s.dataset.step, progress: s.dataset.progress });
      });
    }), root);
    expect(frame).toEqual({ step: '1', progress: '1' });
  });
}

function lifecycleTests({ url, captions, root, leaveHash, returnHash }) {
  test('leaving mid-play stops rendering; returning starts fresh at step 1', async ({ page }) => {
    const errors = collectConsoleErrors(page);
    const p = stepperParts(root);
    await startPausedClock(page, url);
    await page.locator(p.toggle).dispatchEvent('click');
    await page.clock.runFor(MID_MS);
    const stage = await page.locator(p.stage).elementHandle();
    const before = await stage.evaluate((s) => s.dataset.progress);
    await page.evaluate((h) => { location.hash = h; }, leaveHash);
    await expect(page.locator(p.stage)).toHaveCount(0);
    await page.clock.runFor(5_000); // a leaked frame loop or dwell timer would repaint the detached stage
    expect(await stage.evaluate((s) => s.dataset.progress)).toBe(before);
    await page.evaluate((h) => { location.hash = h; }, returnHash);
    await expect(page.locator(p.count)).toHaveText(`1 / ${captions.length}`);
    await expect(page.locator(p.toggle)).toHaveText('Play');
    await expect(page.locator(p.caption)).toHaveText(captions[0]);
    expect(errors).toEqual([]);
  });
}

function factTests({ url, root, factRows }) {
  test(`facts: ${factRows} rows, each sourced row links its source, none missing`, async ({ page }) => {
    await openSettled(page, url, root);
    const rows = page.locator('[data-section="facts"] .fact-row');
    await expect(rows).toHaveCount(factRows);
    await expect(page.locator('[data-section="facts"] .fact-missing')).toHaveCount(0);
    const unsourced = await page.locator('[data-section="facts"] .fact-row:not([data-derived])').evaluateAll((lis) => lis.filter((li) => !li.querySelector('.fact-source')).map((li) => li.textContent.slice(0, 60)));
    expect(unsourced).toEqual([]);
  });

  test('dark theme: the whole page renders with no console errors', async ({ page }) => {
    const errors = collectConsoleErrors(page);
    await page.emulateMedia({ colorScheme: 'dark' });
    await openSettled(page, url, root);
    await page.locator('[data-section="math"] summary').click();
    await expect(page.locator('[data-section="math"] .katex').first()).toBeVisible();
    expect(errors).toEqual([]);
  });
}

// One call per lesson spec file. `captions` are the storyboard's, verbatim; `factRows` is its §8 row count.
export function registerLessonContract({ name, url, captions, factRows, leaveHash = 'unmount-check', returnHash, root = LESSON_STEPPER }) {
  if (!captions?.length || !Number.isInteger(factRows) || !returnHash) throw new Error('registerLessonContract: captions, factRows and returnHash are required');
  const opts = { url, captions, factRows, leaveHash, returnHash, root };
  test.describe(`${name}: lesson contract`, () => {
    captionTests(opts);
    determinismTests(opts);
    layoutTests(opts);
    controlTests(opts);
    lifecycleTests(opts);
    factTests(opts);
  });
}
