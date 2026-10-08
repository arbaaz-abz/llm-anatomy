// rlhf-dpo pure formatters, the toy's starting state and the "Check my work" text (storyboard §6). No DOM.
import { dpoLoss, sigmoid } from '@math/preference.js';
import { DPO_DEFAULT } from './numbers.js';

const MINUS = '−';
const realMinus = (s) => s.replace(/^-/, MINUS);

// Fixed decimals with a real minus, never "−0.000".
export const fixed = (v, digits) => {
  if (v == null || Number.isNaN(v)) return '';
  const s = v.toFixed(digits);
  return /^-0(\.0+)?$/.test(s) ? s.slice(1) : realMinus(s);
};
export const fmt1 = (v) => fixed(v, 1);
export const fmt2 = (v) => fixed(v, 2);
export const fmt3 = (v) => fixed(v, 3);

// The slider's printed value: signed one decimal ("+0.5", "−0.2", "0").
export const fmtChange = (v) => (v === 0 ? '0' : `${v > 0 ? '+' : MINUS}${Math.abs(v).toFixed(1)}`);
// The shortest form of a slider or β value ("0.5", "−0.2", "5").
export const trim1 = (v) => (Number.isInteger(v) ? realMinus(String(v)) : realMinus(v.toFixed(1)));

export const INITIAL_STATE = Object.freeze({ ...DPO_DEFAULT });
export const TOY_LIMITS = Object.freeze({ change: Object.freeze({ min: -10, max: 10, step: 0.1 }), betas: Object.freeze([0.1, 0.5]) });

const factor = (v) => (v < 0 ? `(${trim1(v)})` : trim1(v));
const diff = (a, b) => (b < 0 ? `${fmt3(a)} − (${fmt3(b)})` : `${fmt3(a)} − ${fmt3(b)}`);

// "Check my work": the storyboard's five lines for the default state, the same layout for any other.
export function checkWork({ dChosen, dRejected, beta }) {
  const r = dpoLoss({ dChosen, dRejected, beta });
  const left = [`${trim1(beta)} × ${factor(dChosen)}`, `${trim1(beta)} × ${factor(dRejected)}`];
  const width = Math.max(...left.map((s) => s.length));
  return [
    `reward A = ${left[0].padEnd(width)} = ${fmt3(r.rewardChosen)}`,
    `reward B = ${left[1].padEnd(width)} = ${fmt3(r.rewardRejected)}`,
    `gap      = ${diff(r.rewardChosen, r.rewardRejected)} = ${fmt3(r.margin)}`,
    `loss     = −ln σ(${fmt3(r.margin)}) = −ln ${fmt3(sigmoid(r.margin))} = ${fmt3(r.loss)}`,
    `weight   = σ(${fmt3(-r.margin)}) = ${fmt3(r.weight)}`,
  ].join('\n');
}
