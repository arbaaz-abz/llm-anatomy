// rope's dated text (pure, no DOM): the §8 rows and framing, the real-head wavelengths, and the notes printed under the stage.
// Dated numbers are {entry.key|format} placeholders filled from data/models.json; ratios and wavelengths are computed from
// those same entries. Rows keep their placeholders so the scaffold adds each row's source link and "reported" chip.
import { ropeFrequencies, wavelengths, ropeScore } from '@math/rope.js';
import { lookupFact } from '@shared/claims.js';
import { listText, thousands, fmt3, fmt1, trimNumber, Q_SAT, keyVector, TRAINED_LENGTH } from './format.js';
import { ROW, POS, FREQS, WAVELENGTHS, SCORE, PLAIN_SCORE, Q_AT, K_AT, COVERAGE } from './numbers.js';

const fact = (data, id, key) => lookupFact(data?.models, id, key)?.value ?? null;
const percent = (v) => (v == null ? '—' : `${Math.round(v * 100)}%`);

export const HEAD_WIDTH = 128; // the real-head readouts use a head of 128 numbers (64 pairs)
// The models whose base the toy's real-head table and the frame 7 strip read from the data.
export const REAL_HEADS = Object.freeze([
  { id: 'deepseek-v4-pro', name: 'DeepSeek-V4-Pro' },
  { id: 'gpt-oss-120b', name: 'gpt-oss-120b' },
  { id: 'minimax-m3', name: 'MiniMax-M3' },
  { id: 'glm-5.3', name: 'GLM-5.3' },
  { id: 'qwen3.8', name: 'Qwen3.8' },
]);

// Tokens per full turn of the slowest pair of a 128-number head: wavelength of the last of 64 frequencies.
export const slowestWavelength = (base) => wavelengths(ropeFrequencies(HEAD_WIDTH, base)).at(-1);

export function realHeadRows(data) {
  return REAL_HEADS.map(({ id, name }) => {
    const base = fact(data, id, 'rope_theta');
    return { id, name, base, tokens: base == null ? null : slowestWavelength(base) };
  });
}

const million = (n) => `${fmt1(n / 1e6)} million`;

export function framing() {
  return 'Almost every 2026 model rotates its queries and keys (in its softmax-attention layers). The differences are the base, how much of each head rotates, which layers rotate at all, and how the model was stretched to its final length. '
    + 'The mainstream recipe for 1M tokens is a very large base, staged length training ([[midtraining]]), attention that is cheap at length ([[long-context-attention]]), and sometimes YaRN.';
}

export function factRows(data) {
  const rotated = fact(data, 'minimax-m3', 'partial_rotary_factor') * fact(data, 'minimax-m3', 'head_dim');
  return [
    { claim: 'GPT-3 ({gpt-3.release_date|year}): {gpt-3.positional} absolute positions, a table of {gpt-3.context_length|int} rows.' },
    { claim: 'RoPE base θ: DeepSeek-V4-Pro {deepseek-v4-pro.rope_theta|int} ({deepseek-v4-pro.rope_theta_compressed|int} for its compressed streams) · gpt-oss {gpt-oss-120b.rope_theta|int} · MiniMax-M3 {minimax-m3.rope_theta|int} · GLM-5.3 {glm-5.3.rope_theta|int} · Qwen3.8 {qwen3.8.rope_theta|int}.' },
    { claim: `Partial RoPE: MiniMax-M3 ({minimax-m3.release_date|year}) rotates ${Number.isFinite(rotated) ? rotated : '—'} of {minimax-m3.head_dim} dimensions ({minimax-m3.partial_rotary_factor|raw}); Qwen3.5 rotates ${percent(fact(data, 'qwen3.5-397b', 'partial_rotary_factor'))} ({qwen3.5-397b.partial_rotary_factor|raw}).` },
    { claim: 'YaRN: gpt-oss ({gpt-oss-120b.release_date|year}) stretched {gpt-oss-120b.rope_original_context|int} → {gpt-oss-120b.context_length|int} tokens (factor {gpt-oss-120b.yarn_factor}); DeepSeek-V4-Pro uses factor {deepseek-v4-pro.yarn_factor}.' },
    { claim: 'NoPE: Kimi K3\'s {kimi-k3.full_attention_layers} MLA layers ({kimi-k3.release_date|year}) use no position encoding; order comes from its {kimi-k3.linear_attention_layers} linear-attention layers\' decay, and it reaches a context of {kimi-k3.context_length|count} tokens without any position-encoding change.' },
    { claim: 'Most frontier open models list about 1M positions: DeepSeek-V4-Pro {deepseek-v4-pro.context_length|count}, Kimi K3 {kimi-k3.context_length|count}, GLM-5.3 {glm-5.3.context_length|count}, MiniMax-M3 {minimax-m3.context_length|count}.' },
  ];
}

// ---- the notes under the stage, one list per frame (storyboard §5 "Numbers shown", visible without hover) ----
// "sat" at 13 against The (11), cat (12) and sat (13): every word shifted by ten, so the offsets are 2, 1 and 0.
const shiftedRow = () => [0, 1, 2].map((i) => ropeScore(Q_SAT, keyVector(i), { qPos: POS.q + POS.shift, kPos: POS.q + POS.shift - (2 - i), freqs: FREQS }).score);
const catScore = (qPos, kPos) => ropeScore(Q_SAT, keyVector(1), { qPos, kPos, freqs: FREQS }).score;

function slowestStrip(data) {
  const heads = realHeadRows(data);
  const at = (id) => heads.find((h) => h.id === id);
  const [deepseek, ossHead, qwen] = ['deepseek-v4-pro', 'gpt-oss-120b', 'qwen3.8'].map(at);
  const tokens = (h, text) => (h.tokens == null ? '—' : text(h.tokens));
  const base = (h) => (h.base == null ? '—' : thousands(h.base));
  return `Slowest turn of a head of ${HEAD_WIDTH} numbers (${HEAD_WIDTH / 2} pairs): base ${base(deepseek)} → ${tokens(deepseek, thousands)} tokens · ${base(ossHead)} (gpt-oss) → ${tokens(ossHead, thousands)} · ${base(qwen)} → ${tokens(qwen, million)}.`;
}

export function belowNotes(data) {
  return [
    ['q_sat and k_cat are head A\'s vectors from [[attention]]: the 3.0 is the score computed there.'],
    ['Query pairs (0, 2) and (0.5, 0), key pairs (0, 1.5) and (0, −0.5). A hand\'s length is its pair\'s size, so turning never changes it.'],
    [`Cells round to fit; exact: q at ${POS.q} = ${listText(Q_AT(POS.q))}.`],
    [`Exact: k at ${POS.k} = ${listText(K_AT(POS.k))}. Pair dots ${fmt3(SCORE.pairs[0])} + (${fmt3(SCORE.pairs[1])}) = ${fmt3(SCORE.score)}; unrotated it was ${fmt1(PLAIN_SCORE)}.`],
    [
      `q at ${POS.q + POS.shift} = ${listText(Q_AT(POS.q + POS.shift))}; k at ${POS.k + POS.shift} = ${listText(K_AT(POS.k + POS.shift))}. The score is ${fmt3(catScore(3, 2))} at positions (3, 2), ${fmt3(catScore(13, 12))} at (13, 12) and ${fmt3(catScore(103, 102))} at (103, 102).`,
      `Shift all of The, cat and sat by ten and the whole row for "sat" is still ${listText(shiftedRow())}. Positions on screen start at 1 and code usually starts at 0; the shift shows it makes no difference.`,
    ],
    [`Score at offsets 0 to 7: ${listText(ROW.base100)}. Wavelengths ${fmt1(WAVELENGTHS[0])} and ${fmt1(WAVELENGTHS[1])} tokens.`],
    [`At base 10,000 the row is ${listText(ROW.base10000)}.`, slowestStrip(data)],
    [
      `Trained offsets 0 to ${TRAINED_LENGTH - 1}: pair 2 reaches ${trimNumber(COVERAGE.plain[1].seenMax)} rad. At ${POS.readTo} tokens it reaches ${trimNumber(COVERAGE.plain[1].reachedMax, 1)} rad, never seen. After ÷ 4: ${trimNumber(COVERAGE.pi[1].reachedMax)} rad. New row ${listText(ROW.pi)}.`,
      'A real model trains on thousands of tokens; 16 and 64 keep the toy small.',
    ],
    [
      `YaRN-style row: ${listText(ROW.yarn)}. With position interpolation, offset 4 scored ${fmt3(ROW.pi[4])}, what offset 1 scored before.`,
      'gpt-oss ({gpt-oss-120b.release_date|year}): {gpt-oss-120b.rope_original_context|int} → {gpt-oss-120b.context_length|int} tokens, factor {gpt-oss-120b.yarn_factor}. Real YaRN ramps smoothly between the two cases and also rescales attention; see the math below.',
    ],
    [partialNote(data)],
  ];
}

function partialNote(data) {
  const factor = fact(data, 'minimax-m3', 'partial_rotary_factor');
  const head = fact(data, 'minimax-m3', 'head_dim');
  const minimax = factor == null ? '—' : `${percent(factor)} (${factor * head} of ${head})`;
  return `Qwen3.5: ${percent(fact(data, 'qwen3.5-397b', 'partial_rotary_factor'))} of dimensions rotate · MiniMax-M3: ${minimax} · Kimi K3's MLA layers: none.`;
}
