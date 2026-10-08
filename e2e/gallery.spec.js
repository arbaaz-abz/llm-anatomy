import { test, expect } from '@playwright/test';
import { collectConsoleErrors } from './helpers.js';
import { glyphOverflows, startPausedClock, animateTo, MID_MS, END_MS } from './lesson-helpers.js';

const stepper = (page) => page.locator('#stepper-root');
const caption = (page) => page.locator('#stepper-root .stepper-caption');
const stage = (page) => page.locator('#stepper-root .stepper-stage');

// Everything that can change the page's width: fonts, the real fact, the open math panel.
async function settle(page) {
  await page.goto('/gallery/');
  await expect(page.locator('#figures figure')).toHaveCount(22);
  await expect(page.locator('#fact-root .fact-source')).toBeVisible();
  await page.locator('.math-panel summary').click();
  await expect(page.locator('.math-panel .katex').first()).toBeVisible();
  await page.evaluate(() => document.fonts.ready);
}

test('gallery loads with no console errors and renders every glyph figure', async ({ page }) => {
  const errors = collectConsoleErrors(page);
  await settle(page);
  await expect(page.locator('#figures svg .glyph').first()).toBeVisible();
  await expect(page.locator('#fact-root .fact-reported')).toHaveText('reported');
  await expect(page.locator('#figures svg[role="group"][aria-labelledby]')).toHaveCount(22);
  expect(errors).toEqual([]);
});

test('no horizontal page scroll once fonts, the fact and the math panel are in', async ({ page }) => {
  await settle(page);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(0);
});

for (const scheme of ['light', 'dark']) {
  test(`body has an explicit background in ${scheme} mode`, async ({ page }) => {
    await page.emulateMedia({ colorScheme: scheme });
    await page.goto('/gallery/');
    const bg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
    expect(bg).not.toBe('rgba(0, 0, 0, 0)');
  });
}

test('glyph text renders where the glyph put it: clipLine label inside its SVG, token index inside its chip', async ({ page }) => {
  await settle(page);
  const boxes = await page.evaluate(() => {
    const rect = (el) => el.getBoundingClientRect();
    const clip = document.querySelector('#figures .g-clip');
    const label = rect(clip.querySelector('.g-label'));
    const svgBox = rect(clip.closest('svg'));
    const token = document.querySelector('#figures .g-token');
    const chip = rect(token.querySelector('.g-frame'));
    const index = rect(token.querySelector('.g-sub'));
    return { labelLeft: label.left - svgBox.left, chip, index };
  });
  expect(boxes.labelLeft).toBeGreaterThanOrEqual(0);
  expect(boxes.index.left).toBeGreaterThanOrEqual(boxes.chip.left);
  expect(boxes.index.right).toBeLessThanOrEqual(boxes.chip.right + 0.5);
  expect(boxes.index.top).toBeGreaterThanOrEqual(boxes.chip.top);
  expect(boxes.index.bottom).toBeLessThanOrEqual(boxes.chip.bottom + 0.5);
});

test('stepper advances with Next and with the → key, and scrubbing back reproduces the first frame exactly', async ({ page }) => {
  await page.goto('/gallery/');
  await expect(stage(page)).toHaveAttribute('data-progress', '1');
  const first = await caption(page).textContent();
  const firstFrame = await stage(page).innerHTML();
  await stepper(page).locator('[data-act="next"]').click();
  await expect(stepper(page).locator('.stepper-count')).toHaveText('2 / 5');
  expect(await caption(page).textContent()).not.toBe(first);

  await stepper(page).focus();
  await page.keyboard.press('ArrowRight');
  await expect(stepper(page).locator('.stepper-count')).toHaveText('3 / 5');

  const scrub = stepper(page).locator('input[type="range"]');
  await scrub.fill('4');
  await expect(stepper(page).locator('.stepper-count')).toHaveText('5 / 5');
  await expect(stage(page)).toHaveAttribute('data-step', '4');
  await expect(stage(page)).toHaveAttribute('data-progress', '1');
  await scrub.fill('0');
  await expect(stepper(page).locator('.stepper-count')).toHaveText('1 / 5');
  await expect(stage(page)).toHaveAttribute('data-step', '0'); // the new step's first frame has rendered…
  await expect(stage(page)).toHaveAttribute('data-progress', '1'); // …and its transition has finished
  expect(await caption(page).textContent()).toBe(first);
  expect(await stage(page).innerHTML()).toBe(firstFrame);
});

test('with reduced motion a step change renders its final frame within one animation frame', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/gallery/');
  await expect(stage(page)).toHaveAttribute('data-step', '0');
  const frame = await page.evaluate(() => new Promise((resolve) => {
    document.querySelector('#stepper-root [data-act="next"]').click();
    requestAnimationFrame(() => {
      const el = document.querySelector('#stepper-root .stepper-stage');
      resolve({ step: el.dataset.step, progress: el.dataset.progress });
    });
  }));
  expect(frame).toEqual({ step: '1', progress: '1' });
});

test('stepper controls: arrow icon buttons with accessible names, seven speeds, 1× default', async ({ page }) => {
  await page.goto('/gallery/');
  const prev = stepper(page).locator('[data-act="prev"]');
  const next = stepper(page).locator('[data-act="next"]');
  await expect(prev).toHaveAttribute('aria-label', 'Previous step');
  await expect(next).toHaveAttribute('aria-label', 'Next step');
  await expect(prev.locator('svg')).toHaveCount(1);
  await expect(stepper(page).locator('[data-act="toggle"]')).toHaveText('Play');
  for (const button of [prev, next]) {
    const box = await button.boundingBox();
    expect(box.width).toBeGreaterThanOrEqual(40);
    expect(box.height).toBeGreaterThanOrEqual(40);
  }
  await next.focus();
  const ring = await next.evaluate((el) => getComputedStyle(el).outlineStyle);
  expect(ring).not.toBe('none');
  await expect(stepper(page).locator('select option')).toHaveCount(7);
  await expect(stepper(page).locator('select')).toHaveValue('1');
});

test('no glyph is clipped by its SVG: figures and every stepper step', async ({ page }) => {
  await settle(page);
  expect(await glyphOverflows(page, '#figures svg')).toEqual([]);
  const scrub = stepper(page).locator('input[type="range"]');
  for (let step = 0; step < 5; step += 1) {
    await scrub.fill(String(step));
    await expect(stage(page)).toHaveAttribute('data-step', String(step));
    await expect(stage(page)).toHaveAttribute('data-progress', '1');
    expect(await glyphOverflows(page, '#stepper-root .stepper-stage svg')).toEqual([]);
  }
});

test('step 5: V is plain, weight chips share the weights row\'s color scale, no accent stroke wider than the focus ring', async ({ page }) => {
  await page.goto('/gallery/');
  const scrub = stepper(page).locator('input[type="range"]');
  await scrub.fill('4');
  await expect(stage(page)).toHaveAttribute('data-step', '4');
  await expect(stage(page)).toHaveAttribute('data-progress', '1');
  const result = await page.evaluate(() => {
    const probe = document.createElement('i');
    probe.style.color = 'var(--accent)';
    document.body.append(probe);
    const accent = getComputedStyle(probe).color;
    probe.remove();
    const offenders = [];
    for (const el of document.querySelectorAll('#stepper-root .stepper-stage svg *')) {
      const cs = getComputedStyle(el);
      if (cs.stroke === accent && parseFloat(cs.strokeWidth) > 2) offenders.push(`${el.tagName}.${el.getAttribute('class')} ${cs.strokeWidth}`);
    }
    const stageEl = document.querySelector('#stepper-root .stepper-stage');
    // Effective opacity: the cell's own, times every ancestor's up to the stage (computed opacity does not compound).
    const effectiveOpacity = (el) => {
      let o = 1;
      for (let node = el; node && node !== stageEl; node = node.parentElement) o *= Number(getComputedStyle(node).opacity);
      return o;
    };
    const vOpacities = [...stageEl.querySelectorAll('[data-link="v"] .g-cell')].map((c) => String(effectiveOpacity(c)));
    const weightLevels = [...stageEl.querySelectorAll('[data-link="a"] .g-cell')].map((c) => c.dataset.level);
    const tags = [...stageEl.querySelectorAll('.g-tag')];
    return {
      offenders,
      vOpacities: [...new Set(vOpacities)],
      weightLevels,
      tagLevels: tags.map((t) => t.dataset.level),
      tagTexts: tags.map((t) => t.querySelector('.g-text').textContent),
      hatchedTags: tags.filter((t) => t.querySelector('.g-hatch')).length,
      ghosts: stageEl.querySelectorAll('.g-ghost').length,
    };
  });
  expect(result.offenders).toEqual([]);
  expect(result.vOpacities).toEqual(['1']);
  expect(result.tagLevels.slice(0, 3)).toEqual(result.weightLevels.slice(0, 3)); // same valueColor + maxAbs as the weights row
  expect(result.tagTexts).toEqual(['× 0.10', '× 0.70', '× 0.20', '× 0']);
  expect(result.hatchedTags).toBe(1); // the masked row's "× 0"
  expect(result.ghosts).toBe(0); // at progress 1 the weighted rows have merged into o_sat
  await expect(caption(page)).toContainText('70 % of cat');
});

test('the query is marked the same way in every frame: "query" under the sat chip, ▶ before the sat row of K and V', async ({ page }) => {
  await page.goto('/gallery/');
  const scrub = stepper(page).locator('input[type="range"]');
  const marks = async () => page.evaluate(() => {
    const stageEl = document.querySelector('#stepper-root .stepper-stage');
    const label = [...stageEl.querySelectorAll('text')].find((t) => t.textContent === 'query');
    const markers = [...stageEl.querySelectorAll('.g-row-marker')];
    const satChip = [...stageEl.querySelectorAll('.g-token')].find((t) => t.querySelector('.g-text')?.textContent === 'sat');
    const chip = satChip.getBoundingClientRect();
    const lab = label?.getBoundingClientRect();
    return {
      labelUnderChip: !!lab && lab.top >= chip.bottom - 1 && Math.abs((lab.left + lab.right) / 2 - (chip.left + chip.right) / 2) < 6,
      labelFill: label && getComputedStyle(label).fill,
      markerFills: [...new Set(markers.map((m) => getComputedStyle(m).fill))],
      markerCount: markers.length,
      markerRows: markers.map((m) => Math.round(m.getBoundingClientRect().top)),
      satRows: ['k', 'v'].map((k) => stageEl.querySelectorAll(`[data-link="${k}"] .g-cell`)[8]).filter(Boolean).map((c) => Math.round(c.getBoundingClientRect().top)), // row 2 ("sat"), first cell
    };
  });
  const expectedMarkers = { 0: 0, 1: 0, 2: 1, 3: 1, 4: 2 };
  for (let step = 0; step < 5; step += 1) {
    await scrub.fill(String(step));
    await expect(stage(page)).toHaveAttribute('data-step', String(step));
    await expect(stage(page)).toHaveAttribute('data-progress', '1');
    const m = await marks();
    expect(m.labelUnderChip, `step ${step}: query label under the sat chip`).toBe(true);
    expect(m.markerCount, `step ${step}: marker count`).toBe(expectedMarkers[step]);
    if (m.markerCount) {
      expect(m.markerFills).toEqual([m.labelFill]); // one accent style for every selection mark
      m.markerRows.forEach((top, i) => expect(Math.abs(top - m.satRows[i])).toBeLessThanOrEqual(6)); // beside the sat row
    }
  }
});

const DISSOLVE_MS = 480; // 30 frames of the 16 ms grid: eased progress 0.7408, inside the 0.6–0.9 dissolve

test('mid-step 5: the travelling rows are drawn on top of o_sat, dissolve, and never leave the SVG', async ({ page }) => {
  await startPausedClock(page, '/gallery/');
  await page.evaluate(() => document.fonts.ready);
  const sample = async () => {
    const overflow = await glyphOverflows(page, '#stepper-root .stepper-stage svg');
    const ghosts = await page.evaluate(() => {
      const svg = document.querySelector('#stepper-root .stepper-stage svg');
      const o = svg.querySelector('[data-link="o"]');
      return [...svg.querySelectorAll('.g-ghost')].map((g) => ({ opacity: Number(g.style.opacity), afterOutput: !!(o && (o.compareDocumentPosition(g) & Node.DOCUMENT_POSITION_FOLLOWING)) }));
    });
    return { overflow, ghosts };
  };
  await animateTo(page, '#stepper-root', 4, MID_MS);
  const mid = await sample();
  expect(mid.overflow).toEqual([]);
  expect(mid.ghosts).toHaveLength(3);
  expect(mid.ghosts.every((g) => g.afterOutput && g.opacity === 1)).toBe(true);
  await page.clock.runFor(DISSOLVE_MS - MID_MS);
  const dissolving = await sample();
  expect(dissolving.overflow).toEqual([]);
  expect(dissolving.ghosts.every((g) => g.afterOutput && g.opacity > 0 && g.opacity < 1)).toBe(true);
  await page.clock.runFor(END_MS - DISSOLVE_MS);
  expect((await sample()).ghosts).toHaveLength(0);
});

test('the slider demo updates its output on input', async ({ page }) => {
  await page.goto('/gallery/');
  const output = page.locator('#slider-root output');
  const before = await output.textContent();
  await page.locator('#ctx').fill('2');
  await expect(output).not.toHaveText(before);
  await expect(output).toHaveText('4.1K tokens');
  await expect(page.locator('#kv-readout')).toHaveText('302 MB');
});

test('KaTeX renders the math panel with hover-linked terms', async ({ page }) => {
  await page.goto('/gallery/');
  await page.locator('.math-panel summary').click();
  await expect(page.locator('.math-panel .katex').first()).toBeVisible();
  await expect(page.locator('.math-panel .hl-q').first()).toBeVisible();
  await expect(page.locator('.math-panel [tabindex]')).toHaveCount(0);
  await page.locator('.math-panel .hl-k').first().hover();
  await expect(stage(page)).toHaveAttribute('data-hl', 'k');
});

test('a draft token is dashed and muted, never hatched (README lesson 24)', async ({ page }) => {
  await page.goto('/gallery/');
  const draft = page.locator('#figures .g-token--draft').first();
  expect(await draft.locator('.g-frame').evaluate((r) => getComputedStyle(r).strokeDasharray)).not.toBe('none');
  await expect(draft.locator('.g-hatch')).toHaveCount(0);
  expect(await page.locator('#figures .g-patch--dim .g-frame').first().evaluate((r) => getComputedStyle(r).strokeDasharray)).toBe('none');
});

test('memBar keeps its exact output (labels, percentages, geometry) through the shareBar refactor', async ({ page }) => {
  await page.goto('/gallery/');
  const bar = page.locator('#figures .g-membar');
  await expect(bar).toHaveAttribute('aria-label', 'memory: useful 48%, reserved but empty 10%, free 42%');
  await expect(bar.locator('.g-pct')).toHaveText(['48%', '10%', '42%']);
  await expect(bar.locator('.g-label')).toHaveText('useful 23 · reserved 5 · free 20');
  const rects = await bar.evaluate((g) => [...g.querySelectorAll(':scope > rect')].map((r) => [r.getAttribute('class'), Number(r.getAttribute('x') ?? 0), Number(r.getAttribute('width'))]));
  const expected = [['g-useful', 0, 132.1667], ['g-reserved-bg', 134.1667, 27.1667], ['g-hatch', 134.1667, 27.1667], ['g-free', 163.3333, 116.6667]];
  expect(rects.map(([c]) => c)).toEqual(expected.map(([c]) => c));
  rects.forEach(([, x, w], i) => { expect(x).toBeCloseTo(expected[i][1], 3); expect(w).toBeCloseTo(expected[i][2], 3); });
});

for (const width of [1280, 400]) {
  test(`every figure fits its card at ${width} px: no svg wider than the card, no card that scrolls`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    await page.goto('/gallery/');
    await page.evaluate(() => document.fonts.ready);
    const misfits = await page.locator('#figures figure').evaluateAll((figs) => figs.flatMap((fig) => {
      const card = fig.querySelector('.scroll-x');
      const svg = fig.querySelector('svg');
      const name = fig.querySelector('figcaption code, figcaption')?.textContent.trim().slice(0, 30);
      const svgW = svg.getBoundingClientRect().width;
      return svgW <= card.clientWidth && card.scrollWidth <= card.clientWidth ? [] : [`${name}: svg ${svgW.toFixed(0)} / scroll ${card.scrollWidth} vs card ${card.clientWidth}`];
    }));
    expect(misfits).toEqual([]);
  });
}

test('shareBar prints "not published" beside its off-scale segment, inside the figure (decoder-anatomy §4)', async ({ page }) => {
  await page.goto('/gallery/');
  const share = page.locator('#figures .g-share');
  await expect(share.locator('.g-unknown-label')).toHaveText(['not published']);
  await page.evaluate(() => document.fonts.ready);
  // getBoundingClientRect, not locator.boundingBox(): WebKit's protocol box for SVG text is unreliable.
  const rect = (l) => l.evaluate((e) => { const r = e.getBoundingClientRect(); return { x: r.x, y: r.y, width: r.width, height: r.height }; });
  const [seg, label] = await Promise.all([rect(share.locator('.g-part-none').first()), rect(share.locator('.g-unknown-label').first())]);
  expect(label.x).toBeGreaterThanOrEqual(seg.x + seg.width); // beside, to the right
  expect(Math.abs(label.y + label.height / 2 - (seg.y + seg.height / 2))).toBeLessThanOrEqual(3); // on the bar's line
});
