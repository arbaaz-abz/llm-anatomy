// moe's dated text (pure, no DOM): the §8 rows and framing, the hook and the notes printed under the stage.
// Dated numbers are {entry.key|format} placeholders filled from data/models.json; ratios and ranges are computed from those
// same entries. Rows keep their placeholders so the scaffold adds each row's source link and "reported" chip.
import { paramBreakdown, mlpParams, PRESETS } from '@math/params.js';
import { sharePct } from '@math/memory.js';
import { expertCombinations } from '@math/moe.js';
import { lookupFact } from '@shared/claims.js';
import { int, combosApprox, combosText, toyMoe } from './format.js';

const fact = (data, id, key) => lookupFact(data?.models, id, key)?.value ?? null;
// The table's Mixture-of-Experts entries: the basis for "3–5% active". The earlier ones are the background ratios.
export const MOE_TABLE = Object.freeze(['deepseek-v4-pro', 'kimi-k3', 'qwen3.8', 'glm-5.3', 'minimax-m3', 'gpt-oss-120b', 'mistral-large-4']);
const EARLIER_MOE = Object.freeze(['mixtral-8x7b', 'qwen3-235b-a22b']);

const activeShare = (data, id) => {
  const [total, active] = [fact(data, id, 'total_params'), fact(data, id, 'active_params')];
  return total && active ? sharePct(active, total) : null;
};
const pct = (data, id) => (activeShare(data, id) == null ? '—' : `${activeShare(data, id).toFixed(1)}%`);

function range(values, format = (v) => String(Math.round(v))) {
  const finite = values.filter(Number.isFinite);
  return finite.length ? `${format(Math.min(...finite))}–${format(Math.max(...finite))}` : '—';
}

// "3–5" (whole percentages of the table's active shares).
export const activeRange = (data) => range(MOE_TABLE.map((id) => activeShare(data, id)));

export function hookFor(data) {
  const share = activeShare(data, 'deepseek-v4-pro');
  const whole = share == null ? '—' : `${Math.round(share)}%`;
  return `DeepSeek-V4-Pro has {deepseek-v4-pro.total_params|count} parameters but uses only {deepseek-v4-pro.active_params|count} for each token. Who decides which ${whole} to use, and what stops a few experts from doing all the work?`;
}

export function framing(data) {
  return `Every frontier open model on this list is a Mixture of Experts with many experts and ${activeRange(data)}% of its parameters active per token; most are fine-grained with one or two shared experts. `
    + 'Where they differ is the router and the balancing method.';
}

const MONTHS = Object.freeze(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']);
const monthYear = (date) => (/^\d{4}-\d{2}/.test(date) ? `${MONTHS[Number(date.slice(5, 7)) - 1]} ${date.slice(0, 4)}` : '—');

function trendRow(data) {
  const shares = MOE_TABLE.map((id) => activeShare(data, id));
  const [mixtral, qwen] = EARLIER_MOE.map((id) => ({ share: activeShare(data, id), date: String(fact(data, id, 'release_date') ?? '') }));
  const whole = (v) => (v == null ? '—' : `${Math.round(v)}%`);
  const years = mixtral.date && qwen.date ? `${mixtral.date.slice(0, 4)}–${qwen.date.slice(2, 4)}` : '—';
  return `The trend: about ${range(shares)}% of parameters active per token in 2026 (the table's entries run ${range(shares, (v) => v.toFixed(1))}%), `
    + `down from ${range([mixtral.share, qwen.share])}% in ${years} (Mixtral 8x7B, ${monthYear(mixtral.date)}, about ${whole(mixtral.share)}; `
    + `Qwen3-235B, ${qwen.date.slice(0, 4) || '—'}, ${whole(qwen.share)}).`;
}

// The 11 rows of storyboard §8, in order. Rows 8 and 11 name no single data entry, so they carry `derived: true`.
// Kimi K3, GLM-5.3 and MiniMax-M3 are printed without a year: their release dates are "reported" in the data, which would
// mark rows whose numbers are confirmed.
export function factRows(data) {
  return [
    { claim: `DeepSeek-V4-Pro ({deepseek-v4-pro.release_date|year}): {deepseek-v4-pro.total_params|count} / {deepseek-v4-pro.active_params|count} (${pct(data, 'deepseek-v4-pro')}), {deepseek-v4-pro.experts_total} routed + {deepseek-v4-pro.experts_shared} shared, top-{deepseek-v4-pro.experts_active}, expert hidden {deepseek-v4-pro.expert_hidden}.` },
    { claim: `Kimi K3: {kimi-k3.total_params|count} / {kimi-k3.active_params|count4} (${pct(data, 'kimi-k3')}), {kimi-k3.experts_total} routed + {kimi-k3.experts_shared} shared, top-{kimi-k3.experts_active}; the experts work in a smaller "latent" space ({kimi-k3.moe_latent_dim}, half the model width of {kimi-k3.d_model}) to cut traffic between GPUs.` },
    { claim: `Qwen3.8 ({qwen3.8.release_date|year}): {qwen3.8.total_params|count} / {qwen3.8.active_params|count} (${pct(data, 'qwen3.8')}), {qwen3.8.experts_total} routed experts plus a shared one, top-{qwen3.8.experts_active}.` },
    { claim: `GLM-5.3: {glm-5.3.total_params|count} / {glm-5.3.active_params|count} (${pct(data, 'glm-5.3')}, the active count carried over from GLM-5), {glm-5.3.experts_total} routed + {glm-5.3.experts_shared} shared, top-{glm-5.3.experts_active}.` },
    { claim: `MiniMax-M3: about {minimax-m3.total_params|count} / {minimax-m3.active_params|count} (${pct(data, 'minimax-m3')}), {minimax-m3.experts_total} routed + {minimax-m3.experts_shared} shared, top-{minimax-m3.experts_active}, first {minimax-m3.dense_layers} layers dense.` },
    { claim: `gpt-oss-120b ({gpt-oss-120b.release_date|year}): {gpt-oss-120b.total_params|count4} / {gpt-oss-120b.active_params|count} (${pct(data, 'gpt-oss-120b')}), {gpt-oss-120b.experts_total} experts, top-{gpt-oss-120b.experts_active}, no shared expert.` },
    { claim: 'Mistral Large 4 (preview, {mistral-large-4.release_date|year}): {mistral-large-4.total_params|count} total, {mistral-large-4.active_params|count} routed-active ({mistral-large-4.active_params_with_embeddings|count} with embeddings), a "granular MoE" whose expert count is unpublished.' },
    { claim: trendRow(data), derived: true },
    { claim: 'Routers: softmax top-k (Mixtral, Qwen3); sigmoid plus bias (DeepSeek-V3); DeepSeek-V4: {deepseek-v4-pro.router}.' },
    { claim: 'Balancing: aux-loss-free bias (DeepSeek, 2024); DeepSeek-V4-Pro: {deepseek-v4-pro.balancing}; Kimi K3: {kimi-k3.balancing} instead.' },
    { claim: 'The serving cost: at large batch nearly every expert is touched every step, so the whole model is read; spreading experts over many GPUs (expert parallelism) gathers enough tokens per expert: [[parallelism]].', derived: true },
  ];
}

// ---- the notes under the stage (storyboard "visible line" cells and the whole-toy counts) ----

// "Whole toy (2 blocks): dense MLP 1,576 total / 1,448 active · 8 experts 4,008 / 1,576 · 16 experts 7,208 / 1,704", from math/params.js.
function wholeToyLine() {
  const part = (label, state) => {
    const b = paramBreakdown(toyMoe(state).config);
    return `${label} ${int(b.total)} total / ${int(b.active)} active`;
  };
  const dense = paramBreakdown(PRESETS.toy);
  return `Whole toy (2 blocks): dense MLP ${int(dense.total)} total / ${int(dense.active)} active · ${part('8 experts', { routed: 8, split: 1, shared: false })} · ${part('16 experts', { routed: 16, split: 1, shared: false })}.`;
}

// DeepSeek-V4-Pro's choices per token, computed from its data entries: C(384, 6).
function deepseekChoices(data) {
  const [n, k] = [fact(data, 'deepseek-v4-pro', 'experts_total'), fact(data, 'deepseek-v4-pro', 'experts_active')];
  if (!n || !k) return 'DeepSeek-V4-Pro picks its routed experts from a far larger set than the toy: —.';
  const ways = expertCombinations(n, k);
  return `DeepSeek-V4-Pro ({deepseek-v4-pro.release_date|year}): choosing {deepseek-v4-pro.experts_active} of {deepseek-v4-pro.experts_total} routed experts can be done in ${combosApprox(ways)} ways (exactly ${combosText(ways)}).`;
}

const DENSE_EXPERT = mlpParams({ kind: 'swiglu', hidden: PRESETS.toy.mlp.hidden / 2 }, PRESETS.toy.dModel); // 192
const SHARED_LINE = `One shared expert of ${int(DENSE_EXPERT)} parameters plus one routed expert of ${int(DENSE_EXPERT)} is the same ${int(2 * DENSE_EXPERT)} as before.`;

export const belowFor = (data) => Object.freeze([
  ['This opens the Mixture-of-Experts branch from frame 6 of [[decoder-anatomy]], with the same toy: 8 experts of hidden 8, top-2.'],
  [],
  ['Softmax over the chosen two, as Mixtral and Qwen3 do; DeepSeek normalizes sigmoid scores instead.'],
  ['The router scores vectors, not topics; this page makes no claim about what each expert learns.'],
  [wholeToyLine()],
  [deepseekChoices(data)],
  [`${SHARED_LINE} Shared experts in 2026: {deepseek-v4-pro.experts_shared} in DeepSeek-V4-Pro, {kimi-k3.experts_shared} in Kimi K3, {glm-5.3.experts_shared} in GLM-5.3, {minimax-m3.experts_shared} in MiniMax-M3.`],
  [
    'This is a batch, not one word: 128 tokens each choose 2 of 8 experts, so the fair share is 128 × 2 ÷ 8 = 32 each. The scores are seeded random numbers plus a fixed lean toward E1 and E2, standing in for a router that has drifted.',
    'In practice the slowest expert sets the pace, on the GPU that holds it: [[parallelism]].',
  ],
  ['Here the bias is hand-picked to show its effect on one token; the next step shows it being learned.'],
  [
    'Each step routes a fresh batch of 128 tokens, so the number wobbles; the bias keeps it near 1.',
    'Every step adds 0.1 to the bias of experts below their fair share and subtracts 0.1 from those above it; real models use a much smaller nudge.',
  ],
]);
