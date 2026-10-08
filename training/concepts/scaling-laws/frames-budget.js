// Frames 1, 2, 5 and 8: the 6ND product, one budget split three ways, the optimum at three budgets, and the 2026 models' ratios.
import * as G from '@shared/glyphs.js';
import { trainingFlops } from '@math/scale.js';
import { isoFlopLoss, computeOptimal, tokensPerParam } from '@math/scaling.js';
import { MODELS, STAGE_BUDGET, FRAME_2_SIZES, CHINCHILLA_RULE, TABLE_ORDER } from './numbers.js';
import { sci, perParam, powerText, sizeText, tokensText } from './format.js';
import { seg, lerp, ease, fade, label, ink, select, linked, countedSci } from './stage.js';

const FLOPS_PER_PARAM_TOKEN = 6; // "× 6": two forward, four backward (the caption says so)
const { deepseekV4Pro: PRO, llama31: LLAMA } = MODELS;

// ---- frame 1: parameters × tokens × 6 = compute ----
const FACTORS = Object.freeze([
  { link: 'n', x: 70, w: 190, text: `parameters ${sizeText(PRO.active)} (active)` },
  { link: 'd', x: 276, w: 130, text: `tokens ${tokensText(PRO.tokens)}` },
  { link: null, x: 422, w: 70, text: `× ${FLOPS_PER_PARAM_TOKEN}` },
]);
const FACTOR_Y = 36;
const FACTOR_H = 44;
const PRODUCT = Object.freeze({ x: 100, y: 138, w: 380, h: 56 });
const flopsOf = (m) => trainingFlops({ params: m.active, tokens: m.tokens });
const sumLine = (m) => `${FLOPS_PER_PARAM_TOKEN} × ${sizeText(m.active)} × ${tokensText(m.tokens)} = ${sci(flopsOf(m))}`;

export function drawFrame1(svg, p) {
  const join = ease(seg(p, 0, 0.3));
  const count = ease(seg(p, 0.3, 0.8));
  FACTORS.forEach((f) => {
    const parent = f.link ? linked(svg, f.link, { x: f.x, y: FACTOR_Y, w: f.w, h: FACTOR_H }) : svg;
    G.block(parent, { x: f.x, y: FACTOR_Y, w: f.w, h: FACTOR_H, label: f.text });
    if (join > 0) G.flow(svg, { from: [f.x + f.w / 2, FACTOR_Y + FACTOR_H], to: [lerp(f.x + f.w / 2, PRODUCT.x + PRODUCT.w / 2, 0.5), PRODUCT.y], carry: 'activation', progress: join });
  });
  if (count > 0) {
    G.block(svg, { ...PRODUCT, label: `compute ≈ ${countedSci(flopsOf(PRO), count, sci(flopsOf(PRO)))} FLOPs`, state: 'idle' });
    select(svg, PRODUCT.x, PRODUCT.y, PRODUCT.w, PRODUCT.h);
  }
  const rows = seg(p, 0.7, 1);
  ink(svg, 100, 232, `${PRO.label}:  ${sumLine(PRO)}`, { opacity: rows });
  label(svg, 100, 256, `${LLAMA.label}:  ${sumLine(LLAMA)}`, { opacity: rows });
}

// ---- frame 2: one budget, three splits ----
const ROW_Y = Object.freeze([70, 150, 230]);
const BLOCK_H = 38;
const BLOCK_X = 16;
const BLOCK_MIN_W = 40;
const BLOCK_PER_DECADE = 55; // block width is model size on a log scale (printed beside each block)
const TEXT_X = 250;
const OPTIMUM_ROW = 1;
const blockWidth = (n) => BLOCK_MIN_W + BLOCK_PER_DECADE * (Math.log10(n) - Math.log10(FRAME_2_SIZES[0]));

function splitText(n) {
  const { D, tokensPerParam: ratio } = isoFlopLoss(STAGE_BUDGET, n);
  const tag = ratio < 1 || ratio > 1e4 ? ' · extrapolated' : '';
  return `${tokensText(D)} tokens · ${perParam(ratio, 0)} per param${tag}`;
}

export function drawFrame2(svg, p) {
  ink(svg, 16, 28, `budget: ${powerText(STAGE_BUDGET)} FLOPs`);
  FRAME_2_SIZES.forEach((n, i) => {
    const t = ease(seg(p, i * 0.28, i * 0.28 + 0.4));
    if (t <= 0) return;
    const w = lerp(BLOCK_MIN_W / 2, blockWidth(n), t);
    const parent = linked(svg, 'n', { x: BLOCK_X, y: ROW_Y[i], w, h: BLOCK_H });
    G.block(parent, { x: BLOCK_X, y: ROW_Y[i], w, h: BLOCK_H, label: t >= 1 ? sizeText(n) : '' });
    const tokens = linked(svg, 'd', { x: TEXT_X, y: ROW_Y[i] - 8, w: 320, h: 16 });
    ink(tokens, TEXT_X, ROW_Y[i] + BLOCK_H / 2, splitText(n), { opacity: seg(t, 0.5, 1) });
  });
  if (p >= 0.85) select(svg, BLOCK_X, ROW_Y[OPTIMUM_ROW], blockWidth(FRAME_2_SIZES[OPTIMUM_ROW]), BLOCK_H, seg(p, 0.85, 1));
}

// ---- frame 5: the optimum at three budgets ----
const TABLE_BUDGETS = Object.freeze([1e22, 1e24, 1e26]);
const TABLE = Object.freeze({ x: 24, w: 532, headY: 52, firstY: 80, rowH: 44, rowGap: 10, cols: { budget: 36, N: 250, D: 390, ratio: 536 } });
const HIGHLIGHT_ROW = 1; // 10^24: the budget the other frames use

export function drawFrame5(svg, p) {
  const head = seg(p, 0, 0.15);
  label(svg, TABLE.cols.budget, TABLE.headY, 'budget', { opacity: head });
  label(svg, TABLE.cols.N, TABLE.headY, 'model N*', { anchor: 'end', opacity: head });
  label(svg, TABLE.cols.D, TABLE.headY, 'tokens D*', { anchor: 'end', opacity: head });
  label(svg, TABLE.cols.ratio, TABLE.headY, 'tokens per parameter', { anchor: 'end', opacity: head });
  TABLE_BUDGETS.forEach((budget, i) => {
    const t = ease(seg(p, 0.15 + i * 0.25, 0.4 + i * 0.25));
    if (t <= 0) return;
    const y = TABLE.firstY + i * (TABLE.rowH + TABLE.rowGap);
    const best = computeOptimal(budget);
    const g = G.svgEl('g', { class: 'glyph g-note' }, svg);
    G.svgEl('rect', { class: 'g-frame', x: TABLE.x, y, width: TABLE.w, height: TABLE.rowH, rx: 6 }, g);
    fade(g, t);
    ink(svg, TABLE.cols.budget, y + TABLE.rowH / 2, `${powerText(budget)} FLOPs`, { opacity: t });
    ink(svg, TABLE.cols.N, y + TABLE.rowH / 2, sizeText(best.N), { anchor: 'end', opacity: t });
    ink(svg, TABLE.cols.D, y + TABLE.rowH / 2, tokensText(best.D), { anchor: 'end', opacity: t });
    ink(svg, TABLE.cols.ratio, y + TABLE.rowH / 2, perParam(best.tokensPerParam), { anchor: 'end', opacity: t });
    if (i === HIGHLIGHT_ROW) select(svg, TABLE.x, y, TABLE.w, TABLE.rowH, t);
  });
}

// ---- frame 8: tokens per active parameter, then per total ----
const RATIO = Object.freeze({ nameX: 12, calcX: 180, valueX: 318, totalCalcX: 344, totalValueX: 566, headY: 52, firstY: 76, rowH: 36, rowW: 556 });
const DENSE_NOTE = 'dense: same';
const fraction = (tokens, params) => `${tokensText(tokens)} ÷ ${sizeText(params)}`;

function ratioRow(svg, { name, calc, value, totalCalc, totalValue, y, rowOpacity, totalOpacity, followed }) {
  const mid = y + RATIO.rowH / 2;
  const frame = G.svgEl('g', { class: 'glyph g-note' }, svg);
  G.svgEl('rect', { class: 'g-frame', x: RATIO.nameX - 4, y, width: RATIO.rowW, height: RATIO.rowH, rx: 6 }, frame);
  fade(frame, rowOpacity);
  ink(svg, RATIO.nameX + 4, mid, name, { opacity: rowOpacity });
  label(svg, RATIO.calcX, mid, calc, { opacity: rowOpacity });
  ink(svg, RATIO.valueX, mid, value, { anchor: 'end', opacity: rowOpacity });
  label(svg, RATIO.totalCalcX, mid, totalCalc, { opacity: totalOpacity });
  label(svg, RATIO.totalValueX, mid, totalValue, { anchor: 'end', opacity: totalOpacity });
  if (followed) select(svg, RATIO.nameX - 4, y, RATIO.rowW, RATIO.rowH, rowOpacity);
}

export function drawFrame8(svg, p) {
  const head = seg(p, 0, 0.1);
  const totals = ease(seg(p, 0.8, 1));
  label(svg, RATIO.nameX + 4, RATIO.headY, 'per active parameter', { opacity: head });
  label(svg, RATIO.valueX, RATIO.headY, 'tokens', { anchor: 'end', opacity: head });
  label(svg, RATIO.totalCalcX, RATIO.headY, 'per total parameter', { opacity: totals });
  ratioRow(svg, { name: 'Chinchilla rule', calc: '', value: String(CHINCHILLA_RULE), totalCalc: '', totalValue: '', y: RATIO.firstY, rowOpacity: seg(p, 0.1, 0.25), totalOpacity: 0 });
  TABLE_ORDER.forEach((key, i) => {
    const m = MODELS[key];
    const dense = m.active === m.total;
    ratioRow(svg, {
      name: m.label, calc: fraction(m.tokens, m.active), value: perParam(tokensPerParam(m.tokens, m.active), 0),
      totalCalc: dense ? DENSE_NOTE : fraction(m.tokens, m.total), totalValue: perParam(tokensPerParam(m.tokens, m.total), 0),
      y: RATIO.firstY + (i + 1) * (RATIO.rowH + 6), rowOpacity: seg(p, 0.2 + i * 0.15, 0.4 + i * 0.15), totalOpacity: totals, followed: key === 'deepseekV4Pro',
    });
  });
}
