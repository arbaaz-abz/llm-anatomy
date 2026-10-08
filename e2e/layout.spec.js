import { test, expect } from '@playwright/test';

for (const track of ['architecture', 'training', 'serving']) {
  test(`${track}: no horizontal page scroll`, async ({ page }) => {
    await page.goto(`/${track}/`);
    await expect(page.locator('.concept-card').first()).toBeVisible();
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    expect(overflow).toBeLessThanOrEqual(0);
  });

  for (const scheme of ['light', 'dark']) {
    test(`${track}: body has an explicit background in ${scheme} mode`, async ({ page }) => {
      await page.emulateMedia({ colorScheme: scheme });
      await page.goto(`/${track}/`);
      await expect(page.locator('.concept-card').first()).toBeVisible();
      const bg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
      expect(bg).not.toBe('rgba(0, 0, 0, 0)');
    });
  }
}

const LESSON = '/architecture/#decoder-anatomy';

test('the lesson column is centered on the viewport: prose at 680 px, figures wider', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.goto(LESSON);
  await expect(page.locator('.lesson-stepper')).toBeVisible();
  const box = async (sel) => page.locator(sel).first().evaluate((n) => { const r = n.getBoundingClientRect(); return { left: r.left, width: r.width }; });
  const prose = await box('.lesson-intuition p');
  const stepper = await box('.lesson-stepper');
  expect(prose.width).toBeLessThanOrEqual(680.5);
  expect(stepper.width).toBeGreaterThan(prose.width);
  expect(stepper.width).toBeLessThanOrEqual(960.5);
  for (const b of [prose, stepper]) expect(Math.abs(b.left + b.width / 2 - 640)).toBeLessThan(2);
});

test('the header theme toggle switches html[data-theme] and the choice survives a reload', async ({ page }) => {
  await page.emulateMedia({ colorScheme: 'light' });
  await page.goto('/architecture/');
  const toggle = page.getByRole('button', { name: 'Dark mode' });
  await expect(toggle).toHaveAttribute('aria-pressed', 'false');
  await toggle.click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await expect(toggle).toHaveAttribute('aria-pressed', 'true');
  await page.reload();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await expect(page.getByRole('button', { name: 'Dark mode' })).toHaveAttribute('aria-pressed', 'true');
});

test('first visit follows the system preference', async ({ page }) => {
  await page.emulateMedia({ colorScheme: 'dark' });
  await page.goto('/architecture/');
  await expect(page.getByRole('button', { name: 'Dark mode' })).toHaveAttribute('aria-pressed', 'true');
});

test('the Lessons menu opens, marks the current lesson, and closes on Escape', async ({ page }) => {
  await page.goto(LESSON);
  const button = page.getByRole('button', { name: 'Lessons', exact: true });
  await expect(page.locator('#lesson-nav')).toBeHidden();
  await button.click();
  await expect(button).toHaveAttribute('aria-expanded', 'true');
  await expect(page.locator('.nav-link[aria-current="page"]')).toBeFocused();
  await page.keyboard.press('Escape');
  await expect(page.locator('#lesson-nav')).toBeHidden();
  await expect(button).toBeFocused();
});

test('every lesson ends with previous and next links', async ({ page }) => {
  await page.goto('/architecture/#attention');
  await expect(page.locator('.lesson-pager')).toBeVisible();
  await expect(page.locator('.lesson-pager .pager-link')).not.toHaveCount(0);
});

test('from 1440 px the lesson list sits at the left edge and the column does not shift', async ({ page }) => {
  await page.setViewportSize({ width: 1600, height: 900 });
  await page.goto(LESSON);
  await expect(page.locator('#lesson-nav')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Lessons', exact: true })).toBeHidden();
  const { left, width } = await page.locator('.lesson-stepper').evaluate((n) => { const r = n.getBoundingClientRect(); return { left: r.left, width: r.width }; });
  expect(Math.abs(left + width / 2 - 800)).toBeLessThan(2);
});

test.describe('from 1440 px the lesson list can be hidden', () => {
  test.use({ viewport: { width: 1600, height: 900 } });

  test('the round button hides and shows the list, keeps the column centered, and the choice survives a reload', async ({ page }) => {
    await page.goto(LESSON);
    const button = page.getByRole('button', { name: 'Hide lessons' });
    await expect(button).toHaveAttribute('aria-expanded', 'true');
    await expect(button).toHaveAttribute('aria-controls', 'lesson-nav');
    const radius = await button.evaluate((n) => ({ radius: getComputedStyle(n).borderRadius, w: n.offsetWidth, h: n.offsetHeight }));
    expect(radius.w).toBe(radius.h);
    await button.click();
    const shown = page.getByRole('button', { name: 'Show lessons' });
    await expect(shown).toHaveAttribute('aria-expanded', 'false');
    await expect(page.locator('#lesson-nav')).toBeHidden();
    const center = async () => page.locator('.lesson-stepper').evaluate((n) => { const r = n.getBoundingClientRect(); return r.left + r.width / 2; });
    expect(Math.abs(await center() - 800)).toBeLessThan(2);
    await page.reload();
    await expect(page.getByRole('button', { name: 'Show lessons' })).toHaveAttribute('aria-expanded', 'false');
    await expect(page.locator('#lesson-nav')).toBeHidden();
    await page.getByRole('button', { name: 'Show lessons' }).click();
    await expect(page.locator('#lesson-nav')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Hide lessons' })).toHaveAttribute('aria-expanded', 'true');
  });
});

test('below 1440 px there is no collapse button; the Lessons menu is unchanged', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.goto(LESSON);
  await expect(page.locator('.nav-collapse')).toBeHidden();
  await expect(page.getByRole('button', { name: 'Lessons', exact: true })).toBeVisible();
});

test('buttons are pills and icon-only buttons are circles', async ({ page }) => {
  await page.goto(LESSON);
  await expect(page.locator('.lesson-stepper')).toBeVisible();
  const shape = (sel) => page.locator(sel).first().evaluate((n) => ({ radius: getComputedStyle(n).borderTopLeftRadius, w: n.offsetWidth, h: n.offsetHeight }));
  for (const sel of ['.stepper-play', '.menu-toggle', '[data-act="prev"]', '.theme-toggle']) {
    const s = await shape(sel);
    expect(parseFloat(s.radius), sel).toBeGreaterThanOrEqual(s.h / 2);
    expect(s.h, sel).toBeGreaterThanOrEqual(44);
  }
  for (const sel of ['[data-act="prev"]', '[data-act="next"]', '.theme-toggle']) {
    const s = await shape(sel);
    expect(s.w, sel).toBe(s.h);
  }
});

test('the lesson title and hook are centered; the prose after them is left-aligned', async ({ page }) => {
  await page.goto(LESSON);
  await expect(page.locator('.lesson-stepper')).toBeVisible();
  const align = (sel) => page.locator(sel).first().evaluate((n) => getComputedStyle(n).textAlign);
  expect(await align('.lesson-head h2')).toBe('center');
  expect(await align('.lesson-hook')).toBe('center');
  expect(['start', 'left']).toContain(await align('.lesson-intuition p'));
});

test('a segmented choice is a pill track: a sliding neutral thumb, and the keyboard still selects', async ({ page }) => {
  await page.goto('/architecture/#attention');
  const track = page.locator('#head .choice-track');
  await expect(track).toBeVisible();
  await expect(track).toHaveAttribute('data-thumb', '');
  expect(parseFloat(await track.evaluate((n) => getComputedStyle(n).borderTopLeftRadius))).toBeGreaterThan(20);
  const options = page.locator('#head .choice-option');
  await expect(options.nth(0)).toHaveAttribute('aria-pressed', 'true');
  const thumbX = () => track.evaluate((n) => n.style.getPropertyValue('--thumb-x'));
  const before = await thumbX();
  await options.nth(0).focus();
  await page.keyboard.press('Tab');
  await expect(options.nth(1)).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(options.nth(1)).toHaveAttribute('aria-pressed', 'true');
  await expect(options.nth(0)).toHaveAttribute('aria-pressed', 'false');
  expect(await thumbX()).not.toBe(before);
  await page.keyboard.press('Tab');
  await page.keyboard.press('Space');
  await expect(options.nth(2)).toHaveAttribute('aria-pressed', 'true');
  const fill = await options.nth(2).evaluate((n) => getComputedStyle(n).backgroundColor);
  expect(fill).toBe('rgba(0, 0, 0, 0)'); // the thumb carries the fill, not the option
});
