// The rope toy's view model (pure, no DOM): every string and number the toy prints, as a function of (state, data).
// The page test checks it against the storyboard's numbers; toy.js only puts it on the screen.
import {
  ropeFrequencies, rotatePairs, ropeScore, scoreByOffset, wavelengths, stretchFrequencies, angleCoverage,
} from '@math/rope.js';
import { formatCount } from '@math/core.js';
import {
  INITIAL_STATE, OFFSETS, TRAINED_LENGTH, Q_SAT, keyVector, checkWork, fmt3, trimNumber, degrees, thousands, wavelengthText,
} from './format.js';
import { realHeadRows, slowestWavelength } from './facts.js';
import { ROW, COVERAGE, FREQS_10K } from './numbers.js';

export { INITIAL_STATE };
export const SHIFT = 10;

const dot = (a, b) => a.reduce((sum, x, i) => sum + x * b[i], 0);
const radText = (v) => `${trimNumber(v, 3)} rad`;
const handText = (angle) => `${trimNumber(angle, 3)} rad · ${degrees(angle)}`;

// "all seen" is honest only when the pair turned fully in training or stays inside; a reach within one token's step past
// the seen range (position interpolation: 1.575 against 1.5) says so.
const verdictText = (c, turned) => {
  if (!c.allSeen) return 'never seen';
  return !turned && c.reachedMax > c.seenMax ? "all seen (within one token's step)" : 'all seen';
};

function coverageColumn(c, i, { freqs, base, target }) {
  const turned = base[i] * (TRAINED_LENGTH - 1) >= 2 * Math.PI;
  return {
    wavelength: wavelengthText((2 * Math.PI) / base[i]),
    seen: turned ? `${radText(c.seenMax)} (a full turn)` : radText(c.seenMax),
    reached: radText(c.reachedMax),
    verdict: verdictText(c, turned),
    speed: `${trimNumber(freqs[i])} per token`,
    target,
  };
}

export function toyView(state, data) {
  const qPos = state.qPos;
  const kPos = Math.min(state.kPos, qPos); // the key never sits after the query (the mask)
  const base = ropeFrequencies(Q_SAT.length, state.base);
  const q = Q_SAT;
  const k = keyVector(state.kToken);
  const { score, pairs } = ropeScore(q, k, { qPos, kPos, freqs: base });
  const shifted = ropeScore(q, k, { qPos: qPos + SHIFT, kPos: kPos + SHIFT, freqs: base }).score;
  const freqs = stretchFrequencies(base, { factor: state.target / TRAINED_LENGTH, method: state.stretch, trainedLength: TRAINED_LENGTH });
  const coverage = angleCoverage(freqs, { trainedLength: TRAINED_LENGTH, length: state.target, trainedFreqs: base });
  const offsetValues = scoreByOffset(q, k, { freqs, offsets: OFFSETS });
  const columns = coverage.map((c, i) => coverageColumn(c, i, { freqs, base, target: state.target }));
  const hands = [
    { who: 'query', i: 0, angle: qPos * base[0] }, { who: 'query', i: 1, angle: qPos * base[1] },
    { who: 'key', i: 0, angle: kPos * base[0] }, { who: 'key', i: 1, angle: kPos * base[1] },
  ].map((h) => ({ ...h, label: `${h.who} pair ${h.i + 1}`, text: handText(h.angle) }));
  return {
    qPos, kPos, q, k, hands,
    qRotated: rotatePairs(q, qPos, base),
    kRotated: rotatePairs(k, kPos, base),
    pairDots: pairs.map(fmt3),
    score: fmt3(score),
    plain: fmt3(dot(q, k)),
    shifted: fmt3(shifted),
    shiftedNote: `positions ${qPos + SHIFT} and ${kPos + SHIFT}: the same number`,
    offsetValues,
    offsetRow: offsetValues.map(fmt3),
    columns,
    reachTitle: `Reaches at ${state.target} tokens`,
    realHeads: realHeadRows(data).map((r) => ({
      ...r, baseText: r.base == null ? '—' : thousands(r.base), tokensText: r.tokens == null ? '—' : formatCount(r.tokens),
    })),
    checkWork: checkWork({ ...state, kPos }),
  };
}

// The three "try this" items of storyboard §6; every number comes from math/rope.js or the data.
export function tryThis(data) {
  const wave = (id) => realHeadRows(data).find((r) => r.id === id);
  const head = (id) => wave(id)?.base;
  const slowest = (id, text) => (head(id) == null ? '—' : text(slowestWavelength(head(id))));
  const million = (n) => `${trimNumber(n / 1e6, 1)} million`;
  const first = (row) => `[${row.slice(0, 4).map(fmt3).join(', ')}, …]`;
  const catScore = (qPos, kPos) => fmt3(ropeScore(Q_SAT, keyVector(1), { qPos, kPos, base: 100 }).score);
  const baseText = (id) => (head(id) == null ? '—' : thousands(head(id)));
  return [
    {
      prompt: `Key cat, positions 3 and 2: score ${catScore(3, 2)}. Tap same offset, +${SHIFT}: 13 and 12, still ${catScore(13, 12)}. Then set the query back to 3 and the key to 1 (offset 2): ${catScore(3, 1)}.`,
      insight: 'RoPE scores depend on how far apart two tokens are, never on where they are.',
      rest: '',
    },
    {
      prompt: `Watch the offset row at base 100: ${first(ROW.base100)}. Switch to ${thousands(10_000)}: ${first(ROW.base10000)}; the slow hand nearly freezes (${trimNumber(FREQS_10K[1])} per token, a turn every ${trimNumber(wavelengths([FREQS_10K[1]])[0], 0)} tokens), the fast one is unchanged.`,
      insight: 'the base only changes the slow pairs, and the slow pairs are what reach far.',
      rest: ` A head of 128 numbers (64 pairs) turns its slowest pair once every ${slowest('deepseek-v4-pro', thousands)} tokens at base ${baseText('deepseek-v4-pro')} and once every ${slowest('qwen3.8', million)} at base ${baseText('qwen3.8')}.`,
    },
    {
      prompt: `Set Read up to 64 with stretch none: pair 2 reaches ${radText(COVERAGE.plain[1].reachedMax)}, never seen. Pick squeeze all: pair 2 is back in range, but the offset row becomes ${first(ROW.pi)}: offset 4 now scores what offset 1 did. Pick squeeze slow pairs: ${first(ROW.yarn)}, close to the original.`,
      insight: 'stretching trades resolution for reach; YaRN spends that resolution only where the model had it to spare.',
      rest: '',
    },
  ];
}
