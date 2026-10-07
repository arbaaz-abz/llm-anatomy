// Shared e2e helpers for every lesson page and the gallery. FROZEN after Plan 2 wave 0.
import { expect } from '@playwright/test';

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
