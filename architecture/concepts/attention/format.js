// Pure formatters for the attention page and the "Check my work" text (storyboard §6). No DOM.
import { TOY, attentionHead } from '@math/attention.js';

const MINUS = '−';
const realMinus = (s) => s.replace(/^-/, MINUS);
const isZero = (v) => v === 0; // true for −0 too

// Fixed decimals with a real minus; −∞ for a masked cell; an exact 0 prints "0".
const fixed = (digits) => (v) => {
  if (v === -Infinity) return `${MINUS}∞`;
  if (isZero(v)) return '0';
  return realMinus(v.toFixed(digits));
};
export const fmt2 = fixed(2);
export const fmt3 = fixed(3);

// The shortest exact form up to `digits` decimals ("−0.5", "1.25", "3"), real minus, never "−0".
export function trimNumber(v, digits = 3) {
  if (v === -Infinity) return `${MINUS}∞`;
  const s = v.toFixed(digits).replace(/\.?0+$/, '');
  return s === '-0' ? '0' : realMinus(s);
}

const STAGE_CELL_CHARS = 5; // a NUMBER_CELL holds five mono characters at the 11 px label size ("−0.87")
const integerOr = (v, text) => (Number.isInteger(v) ? realMinus(String(v)) : text);

// What a stage cell prints, per quantity (each quantity has one format on the stage):
//   score  exact quarter-grid values ("−0.75", "3")      scaled  exact up to 3 decimals, 2 when 3 would not fit
//   exp, weight  3 decimals ("0.095"; exact 0 and 1 print as integers)   output  2 decimals, as "Check my work"
export function cellText(v, kind) {
  if (v === -Infinity) return `${MINUS}∞`;
  switch (kind) {
    case 'score': return trimNumber(v, 2);
    case 'scaled': {
      const three = trimNumber(v, 3);
      return three.length <= STAGE_CELL_CHARS ? three : trimNumber(v, 2);
    }
    case 'exp':
    case 'weight': return integerOr(v, v.toFixed(3));
    case 'output': return integerOr(v, realMinus(v.toFixed(2)));
    default: throw new RangeError(`cellText: unknown kind "${kind}"`);
  }
}

// The divisor slider's printed value: 2 is the model's own choice.
export const divisorText = (v) => (v === Math.sqrt(TOY.dHead) ? `${v} = √d_head (the model's value)` : String(v));

export const headOf = (head) => TOY.heads[head === 'B' ? 'B' : 'A'];
export const runHead = ({ head, causal, divisor }) => {
  const { Q, K, V } = headOf(head);
  return attentionHead(Q, K, V, { causal, divisor });
};

// ---- "Check my work" (the storyboard's exact layout for the default state; the same layout for any other) ----
const KEY_NAME_WIDTH = 6; // "k_down"
const LABEL_WIDTH = 21; // "exp (calculator)     " — the arrow sits in column 22
const VALUE_COLUMNS = [3, 11, 20, 29]; // where each value starts after the arrow; a minus sign sits one left
const SUFFIX_COLUMN = 30;

const factor = (v) => (v < 0 ? `(${trimNumber(v)})` : trimNumber(v));
const termGap = (q, term) => {
  if (term.includes('(')) return 1;
  return q.length > 1 ? 3 : 4;
};
const signedScore = (v) => {
  const abs = Math.abs(v);
  const text = Number.isInteger(abs * 10) ? abs.toFixed(1) : trimNumber(abs, 2);
  return `${v < 0 ? MINUS : ' '}${text}`;
};

function dotLine(q, k, score) {
  const terms = q.map((qi, i) => {
    const qText = factor(qi);
    const term = `${qText}·${factor(k[i])}`;
    return `${term}${' '.repeat(termGap(qText, term))}`;
  });
  return `${terms.join('+ ')}= ${signedScore(score)}`;
}

// Values laid out in the storyboard's columns after "→", then an optional suffix ("sum 6.372").
function columns(values, suffix = '') {
  const placed = values.reduce((line, text, i) => {
    const start = (VALUE_COLUMNS[i] ?? line.length + 2) - (text.startsWith(MINUS) ? 1 : 0);
    return `${line.padEnd(Math.max(start, line.length + 1))}${text}`;
  }, '');
  return suffix ? `${placed.padEnd(Math.max(SUFFIX_COLUMN, placed.length + 5))}${suffix}` : placed;
}

const labelled = (label, values, suffix) => `${label.padEnd(LABEL_WIDTH)}→${columns(values, suffix)}`;

function divideLabel(divisor) {
  const root = Math.sqrt(TOY.dHead);
  return divisor === root ? `÷ √${TOY.dHead} = ÷ ${trimNumber(divisor)}` : `÷ ${trimNumber(divisor)}`;
}

export function checkWork(state) {
  const { query, divisor } = state;
  const { Q, K } = headOf(state.head);
  const r = runHead(state);
  const tokens = TOY.tokens;
  const visible = r.mask[query].map((on, j) => (on ? j : -1)).filter((j) => j >= 0);
  const pick = (row) => visible.map((j) => row[j]);
  const weights = pick(r.weights[query]);
  const dots = tokens.map((t, j) => {
    const head = `q_${tokens[query]} · ${`k_${t}`.padEnd(KEY_NAME_WIDTH)} = `;
    return r.mask[query][j] ? `${head}${dotLine(Q[query], K[j], r.scores[query][j])}` : `${head}masked (${MINUS}∞)`;
  });
  return [
    ...dots,
    labelled(divideLabel(divisor), pick(r.scaled[query]).map((v) => trimNumber(v))),
    labelled('exp (calculator)', pick(r.exps[query]).map((v) => v.toFixed(3)), `sum ${r.expSums[query].toFixed(3)}`),
    labelled('÷ sum', weights.map((v) => v.toFixed(3)), `(adds to ${weights.reduce((s, w) => s + w, 0).toFixed(3)})`),
    `output = ${visible.map((j) => `${r.weights[query][j].toFixed(3)}·v_${tokens[j]}`).join(' + ')}`,
    `       = [${r.output[query].map((v) => realMinus(v.toFixed(2)).replace(/^−0\.00$/, '0.00')).join(', ')}]`,
  ].join('\n');
}

// The math panel's worked-row block, templated from a toy state (brief 05 §4.3: template, never reparse).
export function workedRowTex(state) {
  const r = runHead(state);
  const tex = (v) => (v === -Infinity ? '-\\infty' : trimNumber(v).replace(MINUS, '-'));
  const row = r.masked[state.query].map(tex).join(',\\ ');
  const weights = r.weights[state.query].map((v) => fmt3(v).replace(MINUS, '-')).join(',\\ ');
  return `\\text{worked row: } A_{${state.query + 1},:} = \\operatorname{softmax}\\big([${row}]\\big) = [${weights}]`;
}
