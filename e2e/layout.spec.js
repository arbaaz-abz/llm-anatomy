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
