// scaling-laws toy view model (pure, no DOM): state { C, N, Dinf } → every string the toy prints, plus the plot's input.
// Numbers come from math/scaling.js; formats from format.js.
import { computeOptimal, isoFlopLoss, inferenceAwareOptimum, lifetimeFlops } from '@math/scaling.js';
import { sci, perParam, lossText, sizeText, tokensText, savingText, isExtrapolated, checkWork } from './format.js';

export const SERVED_OPTIONS = Object.freeze([
  { value: 0, label: '0' }, { value: 1e12, label: '1T' }, { value: 1e13, label: '10T' }, { value: 1e14, label: '100T' }, { value: 1e15, label: '1,000T' },
]);
export const VALIDITY_NOTE = 'The fit was made on 2022-scale runs, smaller than most budgets here, so every value is an extrapolation; sizes far from the optimum (fewer than about 1 or more than about 10,000 tokens per parameter) are furthest from its data, and the readout tags them extrapolated.';
export const EXTRAPOLATED = 'extrapolated';

const PLOT_LOG_MIN = 9;
const PLOT_LOG_MAX = Math.log10(1.2e12); // room for the 10^26 optimum (1.02T) at the right edge
const PLOT_POINTS = 120;
const LOSS_STEP = 0.02;
const TICK_STEPS = Object.freeze([0.02, 0.05, 0.1, 0.2, 0.5]);
const MAX_TICKS = 6;
const SAME_SIZE = 1e-9; // relative: the learner's size is the optimum

// With nothing served, the cheapest model for a loss is the compute-optimal one by definition; otherwise a grid search.
function cheapestPick(best, Dinf) {
  if (Dinf === 0) return { N: best.N, D: best.D, tokensPerParam: best.tokensPerParam, total: lifetimeFlops(best.N, best.D, 0) };
  return inferenceAwareOptimum({ targetLoss: best.loss, inferenceTokens: Dinf });
}

function niceTicks(lo, hi) {
  const step = TICK_STEPS.find((s) => Math.floor(hi / s + 1e-9) - Math.ceil(lo / s - 1e-9) + 1 <= MAX_TICKS) ?? TICK_STEPS[TICK_STEPS.length - 1];
  const ticks = [];
  for (let k = Math.ceil(lo / step - 1e-9); k * step <= hi + 1e-9; k += 1) ticks.push({ value: Number((k * step).toFixed(3)), label: (k * step).toFixed(2) });
  return ticks;
}

const sizeLabel = (n) => sizeText(n);

function plotInput(C, N, Dinf, best, cheap) {
  const points = Array.from({ length: PLOT_POINTS + 1 }, (_, i) => {
    const size = 10 ** (PLOT_LOG_MIN + ((PLOT_LOG_MAX - PLOT_LOG_MIN) * i) / PLOT_POINTS);
    return [size, isoFlopLoss(C, size).loss];
  });
  const losses = points.map((p) => p[1]);
  const lo = Math.floor((Math.min(...losses) - 0.005) / LOSS_STEP) * LOSS_STEP;
  const hi = Math.ceil(Math.max(...losses) / LOSS_STEP) * LOSS_STEP;
  const mine = isoFlopLoss(C, N);
  const sameAsBest = Math.abs(N / best.N - 1) < SAME_SIZE;
  const markers = [
    { x: N, y: mine.loss, label: sameAsBest ? sizeLabel(N) : `${sizeLabel(N)} · ${lossText(mine.loss)}`, followed: true },
    ...(sameAsBest ? [] : [{ x: best.N, y: best.loss, label: 'optimum' }]),
    ...(Dinf > 0 ? [{ x: cheap.N, y: best.loss, label: 'cheapest' }] : []),
  ];
  return {
    xAxis: { label: 'active parameters N', log: true, domain: [10 ** PLOT_LOG_MIN, 10 ** PLOT_LOG_MAX], ticks: [{ value: 1e9, label: '1B' }, { value: 1e10, label: '10B' }, { value: 1e11, label: '100B' }, { value: 1e12, label: '1T' }] },
    yAxis: { label: 'fitted loss', domain: [lo, hi], ticks: niceTicks(lo, hi) },
    series: [{ points, label: 'loss at this budget', labelAt: 'mid' }],
    markers,
    refY: Dinf > 0 ? { value: best.loss, label: `same loss ${lossText(best.loss)}` } : null,
  };
}

// The three columns of the table are { N, D, ratio, loss, life } for this size, the compute-optimal size and the cheapest size.
function column({ N, D, tokensPerParam: ratio, loss }, life) {
  return { N: sizeText(N), D: tokensText(D), ratio: perParam(ratio), loss: lossText(loss), life: sci(life), extrapolated: isExtrapolated(ratio) };
}

export function toyView({ C, N, Dinf }) {
  const best = computeOptimal(C);
  const mine = isoFlopLoss(C, N);
  const cheap = cheapestPick(best, Dinf);
  const bestLife = lifetimeFlops(best.N, best.D, Dinf);
  const cols = {
    this: column({ N, ...mine }, lifetimeFlops(N, mine.D, Dinf)),
    opt: column(best, bestLife),
    cheap: column({ N: cheap.N, D: cheap.D, tokensPerParam: cheap.tokensPerParam, loss: best.loss }, cheap.total),
  };
  const saving = 1 - cheap.total / bestLife;
  return {
    cols,
    saving: savingText(saving),
    savingLine: `${savingText(saving)} less lifetime compute than the compute-optimal model`,
    plot: plotInput(C, N, Dinf, best, cheap),
    check: checkWork({ C, N }),
  };
}
