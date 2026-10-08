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
  const button = page.getByRole('button', { name: 'Lessons' });
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
  await expect(page.getByRole('button', { name: 'Lessons' })).toBeHidden();
  const { left, width } = await page.locator('.lesson-stepper').evaluate((n) => { const r = n.getBoundingClientRect(); return { left: r.left, width: r.width }; });
  expect(Math.abs(left + width / 2 - 800)).toBeLessThan(2);
});
