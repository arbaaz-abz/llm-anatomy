import { test, expect } from '@playwright/test';
import { registerLessonContract } from './lesson-helpers.js';
import { CAPTIONS } from '../tests/model-card-expected.js';
import { stepperParts, startPausedClock, animateTo, END_MS, MID_MS } from './lesson-helpers.js';

const URL = '/architecture/#model-card';

registerLessonContract({ name: 'model-card', url: URL, captions: CAPTIONS, factRows: 9, returnHash: 'model-card' });

const out = (page, name) => page.locator(`[data-section="toy"] [data-readout="${name}"]`);
const pick = (page, side, id) => page.locator(`#${side} [data-value="${id}"]`).click();

test.describe('model-card toy: decode a card', () => {
  test.beforeEach(async ({ page }) => { await page.goto(URL); });

  test('opens on DeepSeek-V4-Pro against Kimi K3, each card in its own column', async ({ page }) => {
    await expect(page.locator('#left [data-value="deepseek-v4-pro"]')).toHaveAttribute('aria-pressed', 'true');
    await expect(page.locator('#right [data-value="kimi-k3"]')).toHaveAttribute('aria-pressed', 'true');
    await expect(page.locator('#context [data-value="own"]')).toHaveAttribute('aria-pressed', 'true');
    await expect(out(page, 'left-total_params')).toHaveText('1.6T');
    await expect(out(page, 'left-active_params')).toHaveText('49B');
    await expect(out(page, 'left-layers')).toHaveText('61');
    await expect(out(page, 'left-experts_total')).toHaveText('384');
    await expect(out(page, 'left-context_length')).toHaveText('1M');
    await expect(out(page, 'left-modalities')).toHaveText('text');
    await expect(out(page, 'right-total_params')).toHaveText('2.78T');
    await expect(out(page, 'right-active_params')).toHaveText('104.2B');
    await expect(out(page, 'right-layers')).toHaveText('93');
    await expect(page.locator('[data-section="toy"] tr[data-field="total_params"] .fact-lesson')).toContainText('Mixture of Experts');
  });

  test('every field row prints its meaning and its lesson without hover', async ({ page }) => {
    const rows = page.locator('[data-section="toy"] table[data-readout="fields"] tbody tr');
    await expect(rows).toHaveCount(13);
    await expect(page.locator('tr[data-field="active_params"] .ro-sub').first()).toContainText('Weights one token multiplies; sets compute per token.');
    await expect(page.locator('tr[data-field="kv_bytes_per_token"] .ro-sub').first()).toContainText('Bytes of keys and values kept for each token.');
  });

  test('try this 1: V4-Pro against Kimi K3: the bigger total is not the bigger bill per token', async ({ page }) => {
    await expect(out(page, 'left-active-share')).toHaveText('3.06%');
    await expect(out(page, 'right-active-share')).toHaveText('3.75%');
    await expect(out(page, 'left-routed-share')).toHaveText('1.56%');
    await expect(out(page, 'right-routed-share')).toHaveText('1.79%');
  });

  test('the active-share bar is one bar over the first 10%: the active slice prints its own share (README lesson 19)', async ({ page }) => {
    const bars = page.locator('[data-section="toy"] .g-share');
    await expect(bars).toHaveCount(2);
    for (const i of [0, 1]) {
      await expect(bars.nth(i).locator('.g-bracket')).toHaveCount(0);
      await expect(bars.nth(i).locator('.g-hatch')).toHaveCount(0);
    }
    await expect(bars.nth(0).locator('.g-pct')).toHaveText(['3.06%', '6.94%']);
    await expect(bars.nth(1).locator('.g-pct')).toHaveText(['3.75%', '6.25%']);
    await expect(page.locator('[data-section="toy"] .toy-bars')).toContainText('active 3.06% · not used 96.94%');
    await expect(page.locator('[data-section="toy"] .toy-bars-note')).toContainText('the other 90% is not used by this token either');
  });

  test('the cache lines: V4-Pro is a reported range, Kimi K3 has nothing in the data to size it from', async ({ page }) => {
    await expect(out(page, 'left-cache-token')).toHaveText('4,000–12,000 B ≈ 4–12 kB');
    await expect(out(page, 'left-cache-conversation')).toHaveText('4–12 GB');
    await expect(out(page, 'left-gpu-share')).toHaveText('5.0–15.0%');
    await expect(out(page, 'right-cache-token')).toHaveText('not in our data');
    await expect(out(page, 'right-cache-conversation')).toHaveText('not in our data');
    await expect(out(page, 'right-gpu-share')).toHaveText('not in our data');
  });

  test('try this 2: the attention line, not the context number, decides what a long conversation costs', async ({ page }) => {
    await pick(page, 'right', 'minimax-m3');
    await expect(out(page, 'right-cache-token')).toHaveText('122,880 B ≈ 123 kB');
    await expect(out(page, 'right-cache-conversation')).toHaveText('129 GB');
    await expect(out(page, 'right-gpu-share')).toHaveText('161.1%');
    await expect(out(page, 'left-cache-conversation')).toHaveText('4–12 GB');
    await pick(page, 'right', 'gpt-oss-120b');
    await expect(out(page, 'right-cache-token')).toHaveText('36,864 B ≈ 36.9 kB + 4.72 MB fixed');
    await expect(out(page, 'right-cache-conversation')).toHaveText('4.84 GB');
    await page.locator('#context [data-value="131072"]').click();
    await expect(out(page, 'left-cache-conversation')).toHaveText('524 MB–1.57 GB');
    await expect(out(page, 'left-gpu-share')).toHaveText('0.7–2.0%');
    await expect(out(page, 'right-cache-conversation')).toHaveText('4.84 GB');
  });

  test('try this 3: GLM-5.3 against Mistral Large 4: a card is a set of claims, and ranges stay ranges', async ({ page }) => {
    await pick(page, 'left', 'glm-5.3');
    await pick(page, 'right', 'mistral-large-4');
    await expect(out(page, 'left-layers')).toHaveText('78–80');
    await expect(out(page, 'left-active_params')).toHaveText('40B');
    await expect(out(page, 'right-context_length')).toHaveText('512K–1M');
    await expect(out(page, 'right-active_params')).toHaveText('49B');
    await expect(out(page, 'right-layers')).toHaveText('not in our data');
    await expect(out(page, 'right-active-share-embedding')).toHaveText('4.95%');
    const notes = page.locator('[data-section="toy"] .labs-differ');
    await expect(notes).toContainText('Its config says 78 layers; the GLM-5 paper says 80.');
    await expect(notes).toContainText('49B counts the routed experts only; 52B includes the embeddings.');
    await expect(page.locator('tr[data-field="layers"] td[data-side="left"] .fact-reported')).toHaveCount(0);
    await expect(page.locator('tr[data-field="active_params"] td[data-side="left"] .fact-reported')).toHaveCount(1);
  });

  test('"none" leaves a single column', async ({ page }) => {
    await pick(page, 'right', 'none');
    await expect(out(page, 'right-total_params')).toHaveCount(0);
    await expect(out(page, 'left-total_params')).toHaveText('1.6T');
    await expect(page.locator('[data-section="toy"] .g-share')).toHaveCount(1);
    await pick(page, 'right', 'qwen3.8');
    await expect(out(page, 'right-context_length')).toHaveText('262K');
  });
});

// The contract's clip check only asks "inside the svg"; this one asks for breathing room and for each selection box to hold what it marks.
test.describe('model-card stage: fit', () => {
  const ROOT = '[data-section="animation"] .stepper';
  const MARGIN = 6; // user units between any drawn element and the stage edge

  const fit = (page) => page.evaluate((margin) => {
    const svg = document.querySelector('.stepper-stage svg');
    const box = svg.getBoundingClientRect();
    const scale = svg.viewBox.baseVal.width / box.width;
    const rel = (el) => { const r = el.getBoundingClientRect(); return { l: (r.left - box.left) * scale, t: (r.top - box.top) * scale, r: (r.right - box.left) * scale, b: (r.bottom - box.top) * scale }; };
    const edge = [...svg.querySelectorAll('.g-token, .g-block, .g-kv, .g-gpu, .g-share, .g-note')].map((el) => ({ el, r: rel(el) }))
      .filter(({ r }) => r.l < margin || r.t < margin || r.r > svg.viewBox.baseVal.width - margin || r.b > svg.viewBox.baseVal.height - margin)
      .map(({ el }) => `${el.getAttribute('class')} ${el.textContent.trim().slice(0, 24)}`);
    const loose = [];
    for (const sel of svg.querySelectorAll('.g-select')) {
      const s = rel(sel);
      for (const el of svg.querySelectorAll('.g-kv, .g-token, .g-block, .g-note')) {
        const r = rel(el);
        const overlaps = r.r > s.l && r.l < s.r && r.b > s.t && r.t < s.b;
        const inside = r.l >= s.l - 1 && r.r <= s.r + 1 && r.t >= s.t - 1 && r.b <= s.b + 1;
        const opacity = Number(el.closest('g[opacity]')?.getAttribute('opacity') ?? 1);
        if (overlaps && !inside && opacity === 1 && !el.closest('.g-select')) loose.push(`${el.getAttribute('class')} ${el.textContent.trim().slice(0, 20)}`);
      }
    }
    return { edge, loose };
  }, MARGIN);

  test('at rest on every step, nothing sits within 6 units of the stage edge, and no selection box cuts through what it marks', async ({ page }) => {
    await page.goto(URL);
    await expect(page.locator(stepperParts(ROOT).stage)).toHaveAttribute('data-progress', '1');
    for (let k = 0; k < CAPTIONS.length; k += 1) {
      await page.locator(stepperParts(ROOT).scrub).fill(String(k));
      await expect(page.locator(stepperParts(ROOT).count)).toHaveText(`${k + 1} / ${CAPTIONS.length}`);
      await expect(page.locator(stepperParts(ROOT).stage)).toHaveAttribute('data-progress', '1');
      const { edge, loose } = await fit(page);
      expect(edge, `step ${k + 1}: too close to the edge`).toEqual([]);
      expect(loose, `step ${k + 1}: cut by a selection box`).toEqual([]);
    }
  });

  test('mid-transition too', async ({ page }) => {
    await startPausedClock(page, URL);
    for (let k = 0; k < CAPTIONS.length; k += 1) {
      await animateTo(page, ROOT, k, MID_MS);
      const { edge } = await fit(page);
      expect(edge, `step ${k + 1} mid`).toEqual([]);
      await page.clock.runFor(END_MS - MID_MS);
    }
  });
});
