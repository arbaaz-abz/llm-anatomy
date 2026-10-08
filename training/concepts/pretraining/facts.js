// pretraining's dated text (pure, no DOM): the §8 rows and framing, the dated stage text of frames 1, 8 and 10, and the
// notes printed under the stage. Dated numbers are {entry.key|format} placeholders filled from data/models.json; the token
// range is computed from the same entries. Rows keep their placeholders so the scaffold adds each source link and chip.
import { formatCount } from '@math/core.js';
import { lookupFact, fillText } from '@shared/claims.js';

const fact = (data, id, key) => lookupFact(data?.models, id, key)?.value ?? null;

// The open frontier MoEs reported in 2026 with a published pretraining budget (storyboard §3 ¶3, frame 10).
export const BUDGET_TABLE = Object.freeze([
  ['deepseek-v4-pro', 'DeepSeek-V4-Pro'], ['deepseek-v4-flash', 'DeepSeek-V4-Flash'], ['minimax-m2', 'MiniMax-M2'], ['glm-5', 'GLM-5'],
  ['mimo-v2-flash', 'MiMo-V2-Flash'], ['nemotron-3-super', 'Nemotron 3 Super'],
]);

// The 2026 budget range over the table's entries, in tokens; null bounds when the data is missing.
export function tokenRange(data) {
  const values = BUDGET_TABLE.map(([id]) => fact(data, id, 'pretrain_tokens')).filter(Number.isFinite);
  return values.length ? { lo: Math.min(...values), hi: Math.max(...values) } : { lo: null, hi: null };
}

// "25–33" (trillions), for prose that says "trillion tokens".
export function trillionRange(data) {
  const { lo, hi } = tokenRange(data);
  return lo == null ? '—' : `${formatCount(lo / 1e12)}–${formatCount(hi / 1e12)}`;
}

export const FRAMING = 'Every model below is pretrained on next-token cross-entropy, several with a multi-token-prediction term added. '
  + 'The reports differ on vocabulary, data pipeline and token budget, and they give recipes, not ratios: how much of the crawl each filter keeps is not published.';

// The 12 rows of storyboard §8, in order.
export function factRows(data) {
  const kimi = (key) => fact(data, 'kimi-k3', key);
  const table = kimi('vocab_size') && kimi('d_model') ? formatCount(kimi('vocab_size') * kimi('d_model')) : '—';
  return [
    { claim: 'DeepSeek-V4-Pro: {deepseek-v4-pro.pretrain_tokens|count} pretraining tokens.' },
    { claim: 'DeepSeek-V4-Flash {deepseek-v4-flash.pretrain_tokens|count} · MiniMax-M2 {minimax-m2.pretrain_tokens|count} · GLM-5 {glm-5.pretrain_tokens|count} ({glm-5.base_tokens|count} base + mid-training) · MiMo-V2-Flash {mimo-v2-flash.pretrain_tokens|count} · Nemotron 3 Super {nemotron-3-super.pretrain_tokens|count}.' },
    { claim: 'Kimi K3 did not disclose its token count{kimi-k3.pretrain_tokens|cite}.' },
    { claim: 'Llama 3.1 405B ({llama-3.1-405b.release_date|year}): {llama-3.1-405b.pretrain_tokens|count} tokens.' },
    { claim: 'Vocabularies: DeepSeek-V4 {deepseek-v4-pro.vocab_size} · GLM-5 {glm-5.vocab_size} (two columns in its report, shown as a range) · Kimi K3 {kimi-k3.vocab_size} · MiniMax-M2 {minimax-m2.vocab_size} · gpt-oss ~{gpt-oss-120b.vocab_size|count} (`o200k_harmony`).' },
    { claim: `Kimi K3's embedding table: {kimi-k3.vocab_size} × {kimi-k3.d_model} ≈ ${table} parameters.` },
    { claim: 'Multi-token prediction as an auxiliary loss: DeepSeek-V4 ({deepseek-v4-pro.mtp_loss} when LR decay starts), GLM-5{glm-5.mtp|cite}, MiniMax-M2{minimax-m2.mtp|cite}, Nemotron 3 Super{nemotron-3-super.mtp|cite} and Kimi K3{kimi-k3.mtp|cite}.' },
    { claim: 'Kimi K3: {kimi-k3.data_pipeline}.' },
    { claim: 'MiniMax-M2: {minimax-m2.data_pipeline}.' },
    { claim: 'DeepSeek-V4: {deepseek-v4-pro.data_pipeline}.' },
    { claim: 'GLM-5: about {glm-5.swe_issue_pr_pairs|count} issue–PR pairs (about {glm-5.swe_data_tokens|count} unique tokens) for software engineering.' },
    { claim: 'Qwen3: {qwen3.pretrain_tokens|count} tokens over {qwen3.languages} languages; Mistral Large 4: {mistral-large-4.languages} languages.' },
  ];
}

// Dated text drawn on the stage (frames 1, 8 and 10), filled from the data; "—" where the data is missing.
export function stageText(data) {
  const fill = (text) => fillText(text, data);
  const [vocab, d] = ['vocab_size', 'd_model'].map((k) => fact(data, 'kimi-k3', k));
  const { lo, hi } = tokenRange(data);
  const budget = BUDGET_TABLE.map(([id, name]) => ({ name, tokens: fill(`{${id}.pretrain_tokens|count}`), dim: false }));
  return Object.freeze({
    vocab: fill('vocabulary: {kimi-k3.vocab_size} pieces (Kimi K3)'),
    vocabRange: fill('vocabularies {deepseek-v4-pro.vocab_size|count}–{gpt-oss-120b.vocab_size|count} (DeepSeek-V4 to gpt-oss)'),
    embedding: fill(`embedding table ≈ {kimi-k3.vocab_size} × {kimi-k3.d_model} ≈ ${vocab && d ? formatCount(vocab * d) : '—'} params`),
    glm: fill('GLM-5: ~{glm-5.swe_issue_pr_pairs|count} issue–PR pairs, ~{glm-5.swe_data_tokens|count} tokens, for software engineering'),
    budget: Object.freeze([
      ...budget,
      { name: 'Kimi K3', tokens: fill('{kimi-k3.pretrain_tokens}'), dim: false },
      { name: fill('Llama 3.1 405B ({llama-3.1-405b.release_date|year})'), tokens: fill('{llama-3.1-405b.pretrain_tokens|count}'), dim: true },
    ]),
    range: lo == null ? 'range —' : `range ${formatCount(lo)}–${formatCount(hi)}`,
  });
}

// Page text under the stage, one list per frame (template rule 1: placeholders filled by lessonFor).
export const BELOW = Object.freeze([
  ['Tiktokenizer (go deeper) shows real BPE boundaries on your own text.'],
  [],
  [],
  ['The causal mask is the one built in [[attention]]: the same hatched future, now used to grade every position in one pass.'],
]);
