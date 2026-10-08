// midtraining's dated numbers (pure, no DOM): the published runs read from the data (numbers.js is the fallback when there
// is none), the §8 rows and the framing. Rows keep their {entry.key|format} placeholders so the scaffold adds each row's
// source link and "reported" chip; stage text is filled with fillText.
import { lookupFact, fillText } from '@shared/claims.js';
import { budgetShares } from '@math/pipeline.js';
import { sharePct } from '@math/memory.js';
import { formatCount } from '@math/core.js';
import { attentionCostRatio } from '@math/schedule.js';
import { FALLBACK } from './numbers.js';

const value = (data, id, key, fallback) => lookupFact(data?.models, id, key)?.value ?? fallback;
const LABEL = /\d+[KM]/g; // "4K", "192K", "1M": the nominal context labels inside a context_stages string

// The nominal context labels of a run, in order ("8K → 64K (pretraining); 256K → 1M (cooldown)" → 8K, 64K, 256K, 1M).
export const contextLabels = (text) => text.match(LABEL) ?? [];
// A context label in thousands of tokens: 4K → 4, 1M → 1000 (the labels are nominal, so M counts as 1,000K).
export const labelK = (label) => Number.parseInt(label, 10) * (label.endsWith('M') ? 1000 : 1);

function glmStages(data) {
  const f = FALLBACK.glm5.stages;
  return [
    { name: '4K', value: value(data, 'glm-5', 'base_tokens', f[0][1]) },
    { name: '32K', value: value(data, 'glm-5', 'stage_32k_tokens', f[1][1]) },
    { name: '128K', value: value(data, 'glm-5', 'stage_128k_tokens', f[2][1]) },
    { name: '200K', value: value(data, 'glm-5', 'stage_200k_tokens', f[3][1]) },
  ];
}

// The four runs of the toy's context chips: stages with their token counts (null = not published, P3-R6).
export function runStages(data) {
  const out = {};
  Object.entries(FALLBACK.runs).forEach(([id, run]) => {
    const labels = id === 'glm-5' ? run.labels : contextLabels(value(data, id, 'context_stages', run.labels.join(' → ')));
    out[id] = { id, name: run.name, stages: id === 'glm-5' ? glmStages(data) : labels.map((name) => ({ name, value: null })) };
  });
  return out;
}

// Everything the frames and the toy compute with, from the data (or the fallback).
export function runData(data) {
  const n = FALLBACK.nemotron;
  const m = FALLBACK.minimax;
  const stages = glmStages(data);
  const midtrain = value(data, 'glm-5', 'midtrain_tokens', stages.slice(1).reduce((a, s) => a + s.value, 0));
  const nemotron = {
    peak: value(data, 'nemotron-3-super', 'lr_peak', n.peak),
    floor: value(data, 'nemotron-3-super', 'lr_floor', n.floor),
    warmupTokens: value(data, 'nemotron-3-super', 'lr_warmup_tokens', n.warmupTokens),
    decayTokens: value(data, 'nemotron-3-super', 'lr_decay_tokens', n.decayTokens),
    total: value(data, 'nemotron-3-super', 'pretrain_tokens', n.total),
    shape: value(data, 'nemotron-3-super', 'lr_decay_shape', n.shape),
  };
  const minimax = {
    constantTokens: value(data, 'minimax-m2', 'constant_phase_tokens', m.constantTokens),
    decayTokens: value(data, 'minimax-m2', 'decay_phase_tokens', m.decayTokens),
    total: value(data, 'minimax-m2', 'pretrain_tokens', m.constantTokens + m.decayTokens),
  };
  return {
    stages,
    midtrain,
    reportedTotal: value(data, 'glm-5', 'pretrain_tokens', FALLBACK.glm5.reportedTotal),
    nemotron: { ...nemotron, decayPercent: sharePct(nemotron.decayTokens, nemotron.total), warmupPercent: sharePct(nemotron.warmupTokens, nemotron.total) },
    minimax: { ...minimax, decayPercent: sharePct(minimax.decayTokens, minimax.total) },
    runs: runStages(data),
  };
}

// Shares of GLM-5's run by stage (budgetShares), the stage sum and the last-5% tail (the stages after the first).
export function glmBudget(run) {
  const whole = budgetShares(run.stages);
  const tail = budgetShares(run.stages.slice(1));
  return { whole, tail, wholeShare: (name) => whole.parts.find((p) => p.name === name).share };
}

const countText = (v, digits = 3) => formatCount(v, { digits });
export const midtrainShare = (run) => sharePct(run.midtrain, glmBudget(run).whole.knownTotal);
export { countText };

export function framing() {
  return 'The schedule family is unsettled: Nemotron 3 Super uses warmup-stable-decay, DeepSeek-V4 a constant rate with a cosine tail, and Kimi K3 found cosine better when each was tuned separately. '
    + 'All four 2026 reports in the table stage their context length near the end of the run; RoPE models also rescale their rotations ([[rope]]).';
}

export function factRows() {
  return [
    { claim: 'Nemotron 3 Super: {nemotron-3-super.lr_schedule}.{nemotron-3-super.lr_peak|cite}{nemotron-3-super.lr_floor|cite}{nemotron-3-super.lr_warmup_tokens|cite}{nemotron-3-super.lr_decay_tokens|cite}{nemotron-3-super.lr_decay_shape|cite}{nemotron-3-super.pretrain_tokens|cite}' },
    { claim: 'DeepSeek-V4-Pro: {deepseek-v4-pro.lr_schedule}; sequence length {deepseek-v4-pro.context_stages}. Flash: peak 2.7e-4, batch ramped to 75.5M tokens.' },
    { claim: 'Kimi K3: {kimi-k3.lr_schedule}; its scaling study found cosine beat WSD when each was tuned separately; context {kimi-k3.context_stages}; its {kimi-k3.full_attention_layers} full-attention layers use no positional encoding (NoPE).' },
    { claim: 'MiniMax-M2: {minimax-m2.lr_schedule}; context {minimax-m2.context_stages}.{minimax-m2.constant_phase_tokens|cite}{minimax-m2.decay_phase_tokens|cite}' },
    { claim: 'GLM-5: {glm-5.lr_schedule} decay; context 32K ({glm-5.stage_32k_tokens|count}) → 128K ({glm-5.stage_128k_tokens|count}) → 200K ({glm-5.stage_200k_tokens|count}) after a {glm-5.base_tokens|count} base at 4K; {glm-5.midtrain_data}; {glm-5.swe_data_tokens|count} tokens of SWE data.{glm-5.context_stages|cite}' },
    { claim: 'MiMo-V2-Flash: {mimo-v2-flash.lr_schedule}.' },
    { claim: 'Olmo 3: {olmo-3.midtrain_tokens|count}-token Dolmino mid-training, then {olmo-3.context_extension} context extension.' },
    { claim: 'DeepSeek-V4: {deepseek-v4-pro.midtrain_data}.' },
    { claim: 'Kimi K3 states that {kimi-k3.context_extension_note}.' },
  ];
}

// The printed lines of frames 4, 5, 8 and 9 that carry dated numbers, built from the data.
export function stageText(data) {
  const run = runData(data);
  const fill = (text) => fillText(text, data);
  const pct = (v) => `${Number(v.toFixed(1))}%`;
  const { nemotron, minimax } = run;
  const sum = glmBudget(run).whole.knownTotal;
  return {
    nemotronDecay: `Nemotron 3 Super decay: last ${countText(nemotron.decayTokens)} of ${countText(nemotron.total)} (${pct(nemotron.decayPercent)})`,
    minimaxDecay: `MiniMax-M2 decay: ${countText(minimax.decayTokens)} of ${countText(minimax.total)} (${pct(minimax.decayPercent)})`,
    glmMid: `GLM-5: ${countText(run.midtrain)} tokens, long documents and synthetic agent trajectories upsampled`,
    olmo: fill('Olmo 3: ~{olmo-3.midtrain_tokens|count} tokens (reported)'),
    deepseekMid: 'DeepSeek-V4: agentic data injected here',
    glmShare: `GLM-5 mid-training ${countText(run.midtrain)} of ${countText(sum, 4)} = ${pct(midtrainShare(run))}`,
    glmSum: `(${countText(run.reportedTotal)} reported; its published stages sum to ${countText(sum, 4)})`,
    kimiLayers: fill('Kimi K3: {kimi-k3.full_attention_layers} full-attention layers with no positional encoding; nothing to rescale'),
    kimiNote: fill('Kimi K3: “{kimi-k3.context_extension_note}”'),
    kimiRow: '8K → 64K in pretraining, 256K → 1M in cooldown',
    minimaxRow: `8K → 32K → 192K inside its ${countText(minimax.decayTokens)} decay`,
    deepseekRow: '4K → 16K → 64K → 1M',
  };
}

// The three intuition paragraphs (storyboard §3); the dated numbers are computed from the data, never typed.
export function intuition(data) {
  const run = runData(data);
  const [base, k32, k128, k200] = run.stages;
  const sum = glmBudget(run).whole.knownTotal;
  const times = attentionCostRatio(200, 4);
  const share200 = sharePct(k200.value, sum, { decimals: 2 }).toFixed(2);
  return [
    'The learning rate sets how big each weight update is. Runs warm it up from near zero, hold it high while the model learns broadly, and decay it at the end. With a cosine schedule the rate falls smoothly over the whole run, so the end has to be fixed on day one. With warmup-stable-decay (WSD) the rate stays flat for most of the run and falls only in a final stretch; any checkpoint on the plateau can be branched into its own short decay. That makes the end of the run a separate, cheaper decision.',
    'The decay matters because, as the steps shrink, the model stops bouncing around and settles into what it is reading at that moment. So labs put their best data there: this is annealing, and in 2026 it is a named stage, mid-training. The mix shifts toward high-quality text, reasoning, code and synthetic agent trajectories. It is still plain next-token prediction on documents: no chat template, no answers to imitate.',
    `Mid-training is also where context grows. Most of pretraining runs on 4K–8K-token sequences, because long documents are scarce and every token in a long sequence costs more attention work: at 200K each new token is compared with ${times} times as many earlier tokens as at 4K. So labs stretch the window in a few short stages near the end. GLM-5 goes 4K → 32K (${countText(k32.value)} tokens) → 128K (${countText(k128.value)}) → 200K (${countText(k200.value)}): the 200K stage is ${share200}% of the run (GLM-5's ${countText(run.reportedTotal)}; its published stages sum to ${countText(sum, 4)}). `
      + `Models that use RoPE also rescale their rotations so the new, larger offsets look familiar ([[rope]]); ${fillText('Kimi K3 avoids that: its {kimi-k3.full_attention_layers} full-attention layers have no positional encoding (its linear-attention layers carry order another way)', data)}. The cost of all this is choice: a bad anneal mix is baked in at exactly the moment the model is settling, and the briefs show labs still disagree on the schedule.`,
  ];
}

// The math panel's notes (a) to (c): DeepSeek-V4's schedule from the data, and the page's two stated assumptions.
export function mathNotes(data) {
  return [
    fillText('Shapes: η and t are scalars; t in tokens. (a) DeepSeek-V4-Pro: {deepseek-v4-pro.lr_schedule}. Its decay start is not published, so it has no decay-share chip.', data),
    '(b) Nemotron 3 Super calls its decay "minus-sqrt" without a formula; the page uses s(p) = 1 − √p, the standard form of that shape, as its own assumption.',
  ];
}
