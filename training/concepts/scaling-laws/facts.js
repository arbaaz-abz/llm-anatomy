// scaling-laws "In today's models" rows (storyboard §8) and the lines under frames 6 and 8. Pure.
// Dated numbers are {entry.key|format} placeholders; ratios are computed from the same data (stand-ins when there is none).
import { lookupFact } from '@shared/claims.js';
import { formatRatio } from '@math/core.js';
import { tokensPerParam } from '@math/scaling.js';
import { MODELS } from './numbers.js';
import { perParam } from './format.js';

const FACT = (data, id, key, fallback) => lookupFact(data?.models, id, key)?.value ?? fallback;
const ratio = (data, m, paramKey) => perParam(tokensPerParam(FACT(data, m.id, 'pretrain_tokens', m.tokens), FACT(data, m.id, paramKey, paramKey === 'total_params' ? m.total : m.active)), 0);
const { deepseekV4Pro: PRO, deepseekV4Flash: FLASH, nemotron: NEMOTRON, llama31: LLAMA } = MODELS;

const OLMO = Object.freeze({ id: 'olmo-3', tokens: 5.9e12, small: 7e9, large: 32e9 });
const KIMI_FALLBACK = 2.5;

function olmoRange(data) {
  const tokens = FACT(data, OLMO.id, 'pretrain_tokens', OLMO.tokens);
  const lo = tokensPerParam(tokens, FACT(data, OLMO.id, 'total_params_large', OLMO.large));
  const hi = tokensPerParam(tokens, FACT(data, OLMO.id, 'total_params_small', OLMO.small));
  return `${perParam(lo, 0)}–${perParam(hi, 0)}`;
}

export function factRows(data) {
  const kimiRatio = formatRatio(FACT(data, 'kimi-k3', 'scaling_efficiency_vs_k2', KIMI_FALLBACK));
  return [
    { claim: `${PRO.label}: {deepseek-v4-pro.active_params|count} active, {deepseek-v4-pro.pretrain_tokens|count} tokens (${ratio(data, PRO, 'active_params')} tokens per active parameter; ${ratio(data, PRO, 'total_params')} per total parameter of {deepseek-v4-pro.total_params|count}).` },
    { claim: `${FLASH.label}: {deepseek-v4-flash.active_params|count} active, {deepseek-v4-flash.total_params|count} total, {deepseek-v4-flash.pretrain_tokens|count} tokens (${ratio(data, FLASH, 'active_params')} per active parameter).` },
    { claim: `${NEMOTRON.label}: {nemotron-3-super.active_params|count} active, {nemotron-3-super.total_params|count} total, {nemotron-3-super.pretrain_tokens|count} tokens (${ratio(data, NEMOTRON, 'active_params')} per active parameter).` },
    { claim: `Llama 3.1 405B ({llama-3.1-405b.release_date|year}): dense, {llama-3.1-405b.total_params|count} parameters, {llama-3.1-405b.pretrain_tokens|count} tokens (${ratio(data, LLAMA, 'total_params')} per parameter).` },
    { claim: `Olmo 3: about {olmo-3.pretrain_tokens|count} tokens for its {olmo-3.total_params_small|count} and {olmo-3.total_params_large|count} dense models (${olmoRange(data)} tokens per parameter).` },
    { claim: `Kimi K3 ran its own scaling-law studies (batch size, learning rate, tokens per parameter, shape) and claims about ${kimiRatio} scaling efficiency over K2 from architecture, data and recipe together.{kimi-k3.scaling_efficiency_vs_k2|cite}` },
    { claim: 'Optimizers: DeepSeek-V4-Pro {deepseek-v4-pro.optimizer} (AdamW kept for the embedding, head and norms), GLM-5 "{glm-5.optimizer}", Kimi K3 "{kimi-k3.optimizer}"; MiMo-V2-Flash {mimo-v2-flash.optimizer} and Nemotron 3 Super {nemotron-3-super.optimizer}.' },
    { claim: "DeepSeek-V4's Newton–Schulz schedule: {deepseek-v4-pro.muon_ns_schedule}." },
  ];
}

export const SERVE_BASIS_LINE = 'Training counts the full step (forward + backward) per token; serving counts the forward pass only.';
export const DEFINITION_LINE = 'Tokens per parameter = pretraining tokens ÷ active parameters (for a dense model, active = total). The second column divides by total parameters instead.';

// The lines that appear under the stage for one frame (0-based index).
export function belowFor(index) {
  if (index === 5) return [SERVE_BASIS_LINE];
  if (index === 7) return [DEFINITION_LINE];
  return [];
}
