// speculative-decoding pure formatters and "Check my work" (storyboard §6); no DOM. Every number the page prints goes
// through one of these: ratios through formatRatio (a ratio below 1 is a plain number, never "N× smaller"), probabilities
// at their shortest two-decimal form, expected tokens at two decimals, keep chances at three.
import { formatRatio } from '@math/core.js';
import { expectedTokens, simpleSpeedup } from '@math/specdec.js';

export const probText = (x) => String(Number(x.toFixed(2))); // 0.7, 0.85, 0.05, 0.1
export const tokensText = (e) => e.toFixed(2); // 2.53
export const chanceText = (x) => x.toFixed(3); // 0.857, 1.000
export const cellText = (v) => (v === 0 ? '0' : v.toFixed(2)); // vector cells: 0.60, 0.25, 0 for none

// "2.2×", "1.38×"; below 1 the speedup is a loss, printed as the plain number at two decimals ("0.91×", the notes' form).
export function ratioText(x) {
  if (x >= 1) return formatRatio(x);
  const plain = Number(x.toFixed(2));
  return plain >= 1 ? formatRatio(1) : `${plain}×`;
}

// The toy's two-line check box for any (α, k, c): E from expectedTokens, speedup from simpleSpeedup, both rounded once.
// The default state's text is the storyboard's, pinned by the page test.
export function checkWork({ alpha, k, c }) {
  const a = probText(alpha);
  const tokens = tokensText(expectedTokens(alpha, k));
  const denominator = probText(1 + k * c);
  return [
    `E = (1 − α^(k+1)) / (1 − α) = (1 − ${a}^${k + 1}) / (1 − ${a}) = ${tokens} tokens per round`,
    `speedup = E / (1 + k·c) = ${tokens} / (1 + ${k} · ${probText(c)}) = ${tokens} / ${denominator} = ${ratioText(simpleSpeedup(alpha, k, c))} (rounded once, from the unrounded E)`,
  ].join('\n');
}
