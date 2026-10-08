// decoder-recap's dated text (pure, no DOM): the §8 rows and framing, and the notes printed under the stage.
// Dated numbers are {entry.key|format} placeholders filled from data/models.json; the numbers computed on this page
// (GPT-3's exact counts, the QK-norm row) come from math/ through numbers.js.
import { QK, Q_MULTIPLES, KV_FORMULA, GPT3_FACTS, MODERN_FACTS } from './numbers.js';
import { listText } from './format.js';
import { FRAME5_EXACT_TEXT } from './below.js';

export function framing() {
  return "Every part below is in at least one 2026 frontier model; almost none is in all of them. Sebastian Raschka's architecture gallery compares 100+ models part by part (link below).";
}

// The rows of storyboard §8, in order (its last row, "beyond the block", is split in three so each row stays one claim with few sources). Booleans in the data (qk_norm, biases, attention_sink) print as the config key they come from.
export const FACT_ROWS = Object.freeze([
  { claim: 'GPT-3 ({gpt-3.release_date|year}): pre-norm LayerNorm, {gpt-3.positional} positions ({gpt-3.context_length}), a GELU MLP 4× wide, full multi-head attention ({gpt-3.n_heads} heads, each with its own keys and values), biases and tied embeddings.' },
  { claim: 'Pre-norm RMSNorm in most 2026 models, for example DeepSeek-V3 (its training {deepseek-v3.activation_tricks}); Gemma normalizes both before and after each half.' },
  { claim: 'RoPE in most models; in Kimi K3 the {kimi-k3.full_attention_layers} MLA layers use no position encoding: "{kimi-k3.attention}".' },
  { claim: 'QK-norm in Gemma 3 and Qwen3; DeepSeek-V4-Pro normalizes queries and KV entries too (qk_norm: {deepseek-v4-pro.qk_norm}), instead of Kimi\'s QK-Clip.' },
  { claim: 'Biases are mostly gone; gpt-oss-120b keeps its attention biases (biases: {gpt-oss-120b.biases}).' },
  { claim: 'A learned sink per head: gpt-oss-120b (attention_sink: {gpt-oss-120b.attention_sink}), MiMo-V2-Flash ({mimo-v2-flash.attention_sink}) and DeepSeek-V4-Pro ({deepseek-v4-pro.attention_sink}).' },
  { claim: 'GQA with {gpt-oss-120b.n_kv_heads} KV heads is typical (gpt-oss-120b; Llama 3.1 70B: {llama-3.1-70b.n_kv_heads}); MLA in DeepSeek-V3 (latent rank {deepseek-v3.mla_kv_rank}), Kimi K3 ({kimi-k3.mla_kv_rank}) and GLM-5 ({glm-5.attention_note}).' },
  { claim: 'Hybrid stacks: Qwen3.8 runs {qwen3.8.layer_pattern} (Gated DeltaNet to full attention); Kimi K3 has {kimi-k3.linear_attention_layers} linear and {kimi-k3.full_attention_layers} MLA layers.' },
  { claim: 'Residual redesigns in 2026: {deepseek-v4-pro.residual} in DeepSeek-V4-Pro and {kimi-k3.residual} in Kimi K3.' },
  { claim: 'Beyond the block, multi-token prediction heads: DeepSeek-V3 depth {deepseek-v3.mtp_depth}, V4-Pro {deepseek-v4-pro.mtp_depth}, Kimi K3 {kimi-k3.mtp_depth}, GLM-5 shares {glm-5.mtp_layers}.' },
  { claim: 'Beyond the block, the Muon optimizer: DeepSeek-V4-Pro {deepseek-v4-pro.optimizer}, Kimi K3 {kimi-k3.optimizer}, GLM-5 {glm-5.optimizer}.' },
  { claim: 'Beyond the block, fewer bits per number: DeepSeek-V4-Pro trains in {deepseek-v4-pro.pretrain_precision}; gpt-oss-120b ships {gpt-oss-120b.weight_format}; Kimi K3 uses {kimi-k3.post_training_qat}.' },
]);

export const FRAME7_LINE = "This row is [[attention]]'s hero row; that page computes it step by step.";
export const FRAME5_LINES = Object.freeze([FRAME5_EXACT_TEXT, 'The experts box is opened in [[moe]]. This 64-expert GPT-3 is illustrative, not a real model.']);

// Page text under the stage, one list per frame (dated numbers are placeholders, filled by the lesson from data).
export function belowTexts() {
  const [one, ten, hundred] = Q_MULTIPLES;
  const kvFormula = `${KV_FORMULA(GPT3_FACTS.layers)} → ${KV_FORMULA(MODERN_FACTS.sharedKv)}`;
  return [
    ['The forward pass is the one [[decoder-anatomy]] draws end to end; only the parts inside the block change.'],
    ['The formulas for both norms are in the math panel below. RMSNorm also drops the learned shift β that LayerNorm carries.'],
    ['GPT-3 could not read past its {gpt-3.context_length} positions. How RoPE works, and how a RoPE model is stretched beyond its trained length: [[rope]].'],
    ['The gate multiplies two projections of the row, one entry at a time, before the last matrix. The math panel writes it out.'],
    FRAME5_LINES,
    [`${kvFormula}. How the sharing works: [[kv-compression]]; building this number by hand: [[kv-cache]].`],
    [
      `plain scores ÷ 2: ${listText(QK.plain(one))} · q × ${ten}: ${listText(QK.plain(ten))} → ${listText(QK.plainWeights(ten), 3)} · normalized (rms q ${QK.qRms.toFixed(3)}; k ${QK.keyRms.map((r) => r.toFixed(3)).join(', ')}): ${listText(QK.normed(one), 3)} → ${listText(QK.normedWeights(one), 3)}, unchanged at q × ${ten} and at q × ${hundred}`,
      FRAME7_LINE,
    ],
    ['gpt-oss, MiMo-V2 and DeepSeek-V4 use one sink per head. The numbers and the window story: [[long-context-attention]].'],
    ['Qwen3.8: {qwen3.8.layer_pattern} · Kimi K3: {kimi-k3.linear_attention_layers} linear + {kimi-k3.full_attention_layers} full · gpt-oss: window and full alternate ({gpt-oss-120b.layer_pattern}) · Qwen3.5-397B: {qwen3.5-397b.full_attention_layers} of {qwen3.5-397b.layers} full. The mechanics: [[long-context-attention]].'],
    [
      `Beyond the block (not drawn): multi-token prediction, an extra head that drafts the token after next ([[sampling]]); the Muon optimizer, which replaces AdamW in V4, Kimi and GLM-5 ([[scaling-laws]]); FP8 training and FP4 / MXFP4 weights, fewer bits per number ([[quantization]]).`,
      'Residual redesigns (mHC, Attention Residuals) change the ⊕ itself. New in 2026 (DeepSeek-V4, Kimi K3); this course does not cover how they work.',
    ],
  ];
}
