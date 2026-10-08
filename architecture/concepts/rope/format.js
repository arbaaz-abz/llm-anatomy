// Pure formatters, the toy's starting state and the "Check my work" text for the rope page (storyboard §6). No DOM.
import { TOY } from '@math/attention.js';
import { ropeFrequencies, rotatePairs, ropeScore } from '@math/rope.js';

export const MINUS = '−';
export const TRAINED_LENGTH = 16; // the toy model was trained on 16 tokens (frame 8)
export const POSITION_MAX = 64;
export const TOY_BASES = Object.freeze([100, 10_000, 1_000_000]);
export const TARGETS = Object.freeze([16, 32, 64]);
export const STRETCH_OPTIONS = Object.freeze([
  { value: 'none', label: 'none' },
  { value: 'pi', label: 'squeeze all (PI)' },
  { value: 'yarn-simple', label: 'squeeze slow pairs (YaRN-style)' },
]);

// The query is always "sat" (attention's head A, row 3); the key is one of the four tokens.
export const Q_SAT = TOY.heads.A.Q[2];
export const keyVector = (tokenIndex) => TOY.heads.A.K[tokenIndex];
export const OFFSETS = Object.freeze([0, 1, 2, 3, 4, 5, 6, 7]);

// Where the animation ends: sat at 3, cat at 2, base 100, trained on 16, read to 16, no stretch.
export const INITIAL_STATE = Object.freeze({ qPos: 3, kToken: 1, kPos: 2, base: 100, target: TRAINED_LENGTH, stretch: 'none' });

const realMinus = (s) => s.replace(/^-/, MINUS);

// Fixed decimals with a real minus and no negative zero ("−0.025", "3.000").
export function fixed(digits) {
  return (v) => {
    const s = v.toFixed(digits);
    return Number(s) === 0 ? (0).toFixed(digits) : realMinus(s);
  };
}
export const fmt3 = fixed(3);
export const fmt1 = fixed(1);

// The shortest exact form up to `digits` decimals ("−0.5", "1.25", "3"), real minus, never "−0".
export function trimNumber(v, digits = 3) {
  const s = v.toFixed(digits).replace(/\.?0+$/, '');
  return s === '-0' || s === '' ? '0' : realMinus(s);
}

const STAGE_CELL_CHARS = 5; // a NUMBER_CELL holds five mono characters at the 11 px label size ("−0.87")
// What a stage cell prints: up to 3 decimals, 2 when 3 would not fit in five characters (the toy prints all three).
export function cellText(v) {
  if (v == null || Number.isNaN(v)) return '';
  const three = trimNumber(v, 3);
  return three.length <= STAGE_CELL_CHARS ? three : trimNumber(v, 2);
}
export const formatCell = (v) => cellText(v);

export const degrees = (rad) => `${fmt1((rad * 180) / Math.PI)}°`;
export const thousands = (n) => realMinus(Math.round(n).toLocaleString('en-US'));
export const wavelengthText = (tokens) => `${fmt1(tokens)} tokens`;

// ---- "Check my work" ----
const pairOf = (vec, i) => [vec[2 * i], vec[2 * i + 1]];
const pairText = ([a, b], f) => `(${f(a)}, ${f(b)})`;
const paren = (v) => `(${fmt3(v)})`;

function pairLines(i, { q, k, qName, kName, qPos, kPos, freqs, pairDot }) {
  const speed = trimNumber(freqs[i]);
  const turn = (name, vec, pos) => `${name} ${pairText(pairOf(vec, i), trimNumber)} turned ${pos} × ${speed} = ${trimNumber(pos * freqs[i])} rad`;
  const [qr, kr] = [rotatePairs(q, qPos, freqs), rotatePairs(k, kPos, freqs)].map((v) => pairOf(v, i));
  const head = i === 0 ? 'pair 1' : 'pair 2';
  const left = [`${head}  ${turn(qName, q, qPos)}`, `        ${turn(kName, k, kPos)}`];
  const right = [pairText(qr, fmt3), pairText(kr, fmt3)];
  const dot = `        dot ${paren(qr[0])}${paren(kr[0])} + ${paren(qr[1])}${paren(kr[1])} = ${fmt3(pairDot)}`;
  return { left, right, dot };
}

// The pair-by-pair arithmetic behind the toy's score, in the toy's current state (base freqs, no stretch).
export function checkWork(state) {
  const q = Q_SAT;
  const k = keyVector(state.kToken);
  const freqs = ropeFrequencies(q.length, state.base);
  const { score, pairs } = ropeScore(q, k, { qPos: state.qPos, kPos: state.kPos, base: state.base });
  const plain = q.reduce((sum, x, i) => sum + x * k[i], 0);
  const ctx = { q, k, qName: 'q_sat', kName: `k_${TOY.tokens[state.kToken]}`, qPos: state.qPos, kPos: state.kPos, freqs };
  const blocks = pairs.map((pairDot, i) => pairLines(i, { ...ctx, pairDot }));
  const width = Math.max(...blocks.flatMap((b) => b.left.map((l) => l.length)));
  const rows = blocks.flatMap((b) => [`${b.left[0].padEnd(width)}  → ${b.right[0]}`, `${b.left[1].padEnd(width)}  → ${b.right[1]}`, b.dot]);
  const second = pairs[1] < 0 ? paren(pairs[1]) : fmt3(pairs[1]);
  return [...rows, `score  ${fmt3(pairs[0])} + ${second} = ${fmt3(score)}   (unrotated ${fmt3(plain)})`].join('\n');
}

// ---- lines the page prints under the stage (exact values for the cells that round) ----
export const listText = (values) => `[${values.map(fmt3).join(', ')}]`;
