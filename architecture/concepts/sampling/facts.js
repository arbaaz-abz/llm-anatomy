// sampling's dated text (pure, no DOM): the §8 rows and framing, the notes printed under the stage and the numbers
// derived from the data. Dated numbers are {entry.key|format} placeholders filled from data/*.json (sv: = serving.json).
import { lookupFact } from '@shared/claims.js';

export const FRAMING = 'Sampling settings are chosen by whoever serves the model, not baked into it. What is baked in, in several 2026 models, is a head that predicts more than one token per step.';

// The six rows of storyboard §8, in order. `|raw` keeps a decimal ("1.8", "2.76"); numbers otherwise print as whole numbers.
export const FACT_ROWS = Object.freeze([
  { claim: 'The softmax runs over the whole vocabulary: {kimi-k3.vocab_size|int} entries in Kimi K3, {gpt-oss-120b.vocab_size|int} in gpt-oss-120b.' },
  { claim: 'DeepSeek-V3 ({deepseek-v3.release_date|year}): one MTP module (depth {deepseek-v3.mtp_depth}) predicts one extra token; the second token is accepted {sv:deepseek-v3-mtp.acceptance_pct}% of the time, about {sv:deepseek-v3-mtp.tps_gain|raw}× tokens per second (for one user, as the V3 report states it).' },
  { claim: 'DeepSeek-V4 ({deepseek-v4-pro.release_date|year}): MTP depth {deepseek-v4-pro.mtp_depth}, also used as an auxiliary training loss.' },
  { claim: 'GLM-5 ({glm-5.release_date|year}): shares {glm-5.mtp_layers} MTP layers; mean accepted length {glm-5.mtp_accept_length|raw} tokens per step in its report.' },
  { claim: 'Kimi K3: MTP depth {kimi-k3.mtp_depth}, fine-tuned as an EAGLE-3-style draft.' },
  { claim: 'In serving, MTP raised per-user speed by {sv:lmsys-gb300-longctx.mtp_per_user_gain_pct}% for DeepSeek-R1 on GB300 NVL72 at 128K input / 8K output, with peak throughput kept (LMSYS, Feb 2026).' },
]);

// Page text under the stage, one list per frame (index 0 = frame 1): dated notes and the hand-offs to other lessons.
export const BELOW = Object.freeze([
  ['The 16-word vocabulary here is a toy: Kimi K3 has {kimi-k3.vocab_size|int} entries and gpt-oss-120b ({gpt-oss-120b.release_date|year}) has {gpt-oss-120b.vocab_size|int}. Where the logits come from: [[decoder-anatomy]]. The loss that shapes them: [[pretraining]].'],
  [],
  [],
  ['The divisor slider on [[attention]] is a different temperature: it works inside attention, not on these scores.'],
  [],
  [],
  [],
  ['MTP depth {deepseek-v3.mtp_depth} in DeepSeek-V3 ({deepseek-v3.release_date|year}) and {deepseek-v4-pro.mtp_depth} in DeepSeek-V4-Pro ({deepseek-v4-pro.release_date|year}); GLM-5 ({glm-5.release_date|year}) shares {glm-5.mtp_layers} MTP layers. The head adds parameters, and some work is wasted when its guess is wrong.'],
  ['DeepSeek-V3 ({deepseek-v3.release_date|year}): the second token is accepted {sv:deepseek-v3-mtp.acceptance_pct}% of the time, so one draft gives {tokensPerStep} tokens per step on average (1 plus the acceptance rate). The full method, with several drafts: [[speculative-decoding]].'],
]);

// The acceptance range in percent ([85, 90]) or null when the data is missing.
export function acceptanceRange(data) {
  const value = lookupFact(data?.serving, 'deepseek-v3-mtp', 'acceptance_pct')?.value;
  return Array.isArray(value) && value.length === 2 ? value : null;
}

// One draft per step: 1 + acceptance (the general function belongs to speculative-decoding and must agree at one draft).
export function tokensPerStep(data) {
  const range = acceptanceRange(data);
  if (!range) return null;
  return range.map((pct) => (1 + pct / 100).toFixed(2));
}

export const tokensPerStepText = (data) => tokensPerStep(data)?.join('–') ?? '—';

// The MTP block of the math panel: expectation = 1 + α. The numbers appear only when the data supplies them.
export function mtpTex(data) {
  const base = String.raw`\text{MTP (one draft): } \mathbb{E}[\text{tokens per step}] = 1 + \alpha`;
  const range = acceptanceRange(data);
  if (!range) return base;
  const alpha = range.map((pct) => (pct / 100).toFixed(2)).join(String.raw`\text{–}`);
  return String.raw`${base},\qquad \alpha = ${alpha} \Rightarrow ${tokensPerStep(data).join(String.raw`\text{–}`)}`;
}
