// scaling-laws pure formatters, the toy's initial state, the slider's values and "Check my work" (no DOM).
// One formatter per quantity (README lesson 35): parameters and tokens through formatCount, FLOPs in scientific form,
// tokens per parameter through perParam, shares through formatShare.
import { formatCount } from '@math/core.js';
import { formatShare } from '@shared/glyphs/bars.js';
import { CHINCHILLA_FIT, computeOptimal, isoFlopLoss } from '@math/scaling.js';
import { STAGE_BUDGET } from './numbers.js';

const SUPERSCRIPT = Object.freeze({ '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '-': '⁻' });
const GRID_START_DECADE = 9; // the slider's grid runs 10^9 to 10^12 in steps of 0.05 decade
const GRID_END_DECADE = 12;
const GRID_STEP = 0.05;
const GRID_STEPS = Math.round((GRID_END_DECADE - GRID_START_DECADE) / GRID_STEP);
const EXTRAPOLATED_BELOW = 1; // tokens per parameter, storyboard §6 validity line
const EXTRAPOLATED_ABOVE = 10000;

export const superscript = (n) => String(n).split('').map((c) => SUPERSCRIPT[c]).join('');

// 9.586e10 → "9.59 × 10¹⁰": three significant figures, trailing zeros kept so every value has the same shape.
export function sci(x, digits = 3) {
  if (!Number.isFinite(x) || x <= 0) throw new RangeError(`sci: x must be a positive finite number, got ${x}`);
  const [mantissa, exponent] = x.toExponential(digits - 1).split('e');
  return `${mantissa} × 10${superscript(Number(exponent))}`;
}

// 1e24 → "10²⁴" when x is an exact power of ten, else sci(x).
export function powerText(x) {
  const exponent = Math.round(Math.log10(x));
  return 10 ** exponent === x ? `10${superscript(exponent)}` : sci(x);
}

// Tokens per parameter: the toy, the check box and the try-this print one decimal; the stage tables print the decimals the
// storyboard names (frame 2 and frame 8: none). A value below 1 keeps two significant figures ("0.17", "0.0017").
export function perParam(ratio, decimals = 1) {
  if (ratio < 1) return String(Number(ratio.toPrecision(2)));
  return ratio.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}

export const lossText = (loss) => loss.toFixed(3);
const termText = (term) => term.toPrecision(4);
export const sizeText = (n) => formatCount(n);
export const tokensText = (d) => formatCount(d);
// "58.5% less": the saving as a share of the compute-optimal model's lifetime compute (never negative: ties print 0.0%).
export const savingText = (saving) => (saving > 0 ? formatShare(saving) : "0.0%");
export const isExtrapolated = (ratio) => ratio < EXTRAPOLATED_BELOW || ratio > EXTRAPOLATED_ABOVE;

// The slider's snapped stops: 0.05-decade steps from 1B to 1T, plus the compute-optimal size of the chosen budget
// (the toy's default stop, which sits between two grid steps, or just past 1T at 10^26).
export function sliderSizes(C) {
  const grid = Array.from({ length: GRID_STEPS + 1 }, (_, i) => 10 ** (GRID_START_DECADE + i * GRID_STEP));
  return [...grid, computeOptimal(C).N].sort((a, b) => a - b);
}

export const INITIAL_STATE = Object.freeze({ C: STAGE_BUDGET, N: computeOptimal(STAGE_BUDGET).N, Dinf: 0 });

// "Check my work" (storyboard §6): the loss for the toy's current C and N, from isoFlopLoss and the fit.
export function checkWork({ C, N }) {
  const { E, A, B, alpha, beta } = CHINCHILLA_FIT;
  const { D, tokensPerParam: ratio, loss } = isoFlopLoss(C, N);
  const nTerm = A / N ** alpha;
  const dTerm = B / D ** beta;
  return [
    `N = ${sci(N)} parameters (active)`,
    `D = ${powerText(C)} ÷ (6 × ${sci(N)}) = ${sci(D)} tokens`,
    `tokens per parameter = D ÷ N = ${perParam(ratio)}`,
    `loss = ${E} + ${A} ÷ N^${alpha} + ${B} ÷ D^${beta}`,
    `     = ${E} + ${termText(nTerm)} + ${termText(dTerm)} = ${lossText(loss)}`,
  ].join('\n');
}

// A real minus everywhere (README lesson 35).
export const real = (text) => String(text).replace(/-/g, '−');

// Muon's (a, b, c): fixed decimals when given ("3.4445"), else the shortest form ("−1.5").
export const coefficientText = (coefficients, decimals = null) => `(${coefficients.map((c) => real(decimals === null ? String(c) : c.toFixed(decimals))).join(', ')})`;
