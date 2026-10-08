// moe toy view model (pure, no DOM): state + data → every string and number the toy prints.
// Counts come from math/params.js, routing and balancing from math/moe.js, percentages from sharePct; published figures from ctx.data.
import { paramBreakdown, mlpParams } from '@math/params.js';
import { sharePct } from '@math/memory.js';
import { formatCount } from '@math/core.js';
import { ROUTER_TOY, routeTopK, gateWeights, expertLoads, simulateBalancing, expertCombinations } from '@math/moe.js';
import { lookupFact, fillText } from '@shared/claims.js';
import { INITIAL_STATE, BATCH, toyMoe, int, picksText, gatesText, combosText, routedText, routedShare, imbalanceText, biasText } from './format.js';
import { TOKENS } from './numbers.js';

export { INITIAL_STATE };

export const REAL_MODELS = Object.freeze([
  { value: 'deepseek-v4-pro', label: 'DeepSeek-V4-Pro' },
  { value: 'kimi-k3', label: 'Kimi K3' },
  { value: 'qwen3.8', label: 'Qwen3.8' },
  { value: 'glm-5.3', label: 'GLM-5.3' },
  { value: 'minimax-m3', label: 'MiniMax-M3' },
  { value: 'gpt-oss-120b', label: 'gpt-oss-120b' },
]);
// The two models whose shared-expert count is not in the data as a number.
const SHARED_WORDING = Object.freeze({ 'qwen3.8': 'a shared expert', 'gpt-oss-120b': 'no shared expert' });
const WOBBLE_NOTE = 'Each step routes a fresh batch of 128 tokens, so the number wobbles; the bias keeps it near 1.';
const ROUTING_NOTE = 'The hand-picked router scores exist only for 8 experts and split 1; with other settings only the counts apply.';
export const ACTIVE_DEFINITION = 'Active = the parameters multiplied for one token: every block parameter except unused experts, plus the unembedding. The embedding table is left out: looking up a row is not a multiplication.';
const SWIGLU = 'swiglu';
const D_MODEL = 8;

const fact = (data, id, key) => lookupFact(data?.models, id, key);
const pct = (part, whole) => `${sharePct(part, whole).toFixed(1)}%`;

// Panel A for the toy sliders: whole-toy counts, per-block costs and (at 8 experts, split 1) the four words' routing.
function toyPanel(state) {
  const moe = toyMoe(state);
  const b = paramBreakdown(moe.config);
  const layers = moe.config.layers;
  const perToken = moe.dense ? moe.config.mlp.hidden : moe.hidden;
  const active = moe.dense ? mlpParams({ kind: SWIGLU, hidden: perToken }, D_MODEL) : (moe.topK + moe.shared) * mlpParams({ kind: SWIGLU, hidden: perToken }, D_MODEL);
  return {
    spec: moe.dense
      ? 'Dense: one MLP of hidden 16 in each of the 2 blocks.'
      : `${moe.experts} experts of hidden ${moe.hidden}${moe.shared ? ' plus 1 shared expert' : ''}, top-${moe.topK}, in each of the 2 blocks.`,
    total: int(b.total),
    active: int(b.active),
    activeShare: pct(b.active, b.total),
    expertActive: int(active),
    router: int(b.parts.router / layers),
    combos: moe.dense ? '—' : combosText(expertCombinations(moe.experts, moe.topK)),
    edgeNote: !moe.dense && moe.topK >= moe.experts ? 'Every expert is used for every token: this is a dense MLP with a router.' : '',
    routing: state.routed === 8 && state.split === 1 ? routingRows(moe.topK) : null,
    routingNote: state.routed === 8 && state.split === 1 ? '' : ROUTING_NOTE,
  };
}

// The four words routed with the hand-picked scores: picks, gate weights and the loads of the whole batch.
function routingRows(topK) {
  return {
    rows: ROUTER_TOY.map((scores, t) => {
      const picks = routeTopK(scores, topK);
      return { token: TOKENS[t], picks: picksText(picks), gates: gatesText(gateWeights(scores, picks)), pickIndices: picks };
    }),
    loads: expertLoads(ROUTER_TOY, topK),
  };
}

// A real model from the data: published totals, the routed experts one token uses, the shared experts, the choices per token.
function realPanel(id, data) {
  const label = REAL_MODELS.find((m) => m.value === id).label;
  const total = fact(data, id, 'total_params');
  const active = fact(data, id, 'active_params');
  const experts = fact(data, id, 'experts_total')?.value;
  const picked = fact(data, id, 'experts_active')?.value;
  const shared = fact(data, id, 'experts_shared')?.value;
  if (!total || !active || !experts || !picked) throw new RangeError(`toyView: the data has no experts for "${id}"`);
  return {
    label,
    total: formatCount(total.value, { digits: 4 }),
    active: formatCount(active.value, { digits: 4 }),
    activeReported: active.confidence === 'reported',
    share: pct(active.value, total.value),
    routed: routedText(picked, experts),
    routedShare: routedShare(picked, experts),
    shared: shared == null ? SHARED_WORDING[id] ?? 'not published' : String(shared),
    combos: combosText(expertCombinations(experts, picked)),
  };
}

// Panel B at the chosen step: the step's loads, the biases it chose with, and busiest ÷ fair share.
function balancePanel({ gamma, step }) {
  const run = simulateBalancing({ ...BATCH, gamma, steps: step + 1 });
  const now = run.at(-1);
  return {
    loads: now.loads,
    loadTexts: now.loads.map(String),
    biasTexts: now.bias.map(biasText),
    imbalance: imbalanceText(now.imbalance),
    busiest: Math.max(...now.loads),
    fairShare: (BATCH.tokens * BATCH.k) / BATCH.experts,
    note: WOBBLE_NOTE,
  };
}

export function toyView(state, data) {
  const real = state.real !== 'toy';
  return {
    real,
    a: real ? realPanel(state.real, data) : toyPanel(state),
    b: balancePanel(state),
    definition: ACTIVE_DEFINITION,
  };
}

// Storyboard §6 try-this prompts: [prompt, insight, rest]. Try-this 1's real-model numbers are filled from the data.
const TRY_THIS = Object.freeze([
  ['Panel A, slide Routed experts 0 → 8 → 16: total 1,576 → 4,008 → 7,208; active 1,448 → 1,576 → 1,704 (the only growth is the router: 0 → 64 → 128 parameters per block). Tap DeepSeek-V4-Pro: {deepseek-v4-pro.total_params|count} total, {deepseek-v4-pro.active_params|count} active, %ACTIVE%, %ROUTED% routed experts per token (%ROUTED_SHARE%).',
    'Experts buy capacity in memory, not work per token.', ''],
  ['Set Split each expert into 2, then 4: 16 experts of hidden 4 top-4, then 32 of hidden 2 top-8. Expert parameters per token stay 384; combinations go 28 → 1,820 → 10,518,300; the router grows to 128 and 256 per block, so active ticks up to 1,704 and 1,960.',
    'Fine-grained experts keep the work fixed and multiply the router\'s choices; the price is a bigger router and more, smaller pieces to move between GPUs', ' ([[parallelism]] frame 10 draws exactly that).'],
  ['Panel B, gamma 0: scrub the batch slider; imbalance stays near 3 (2.50–3.00; step 9: 2.91). Gamma 0.1: 3.00 → 1.13 by step 6. Gamma 0.2: 1.22 by step 2, then it swings back up to 1.84 at step 5.',
    'A small selection-only bias balances the experts without touching the gate weights; too big a nudge overshoots.', ''],
]);

export function tryThis(data) {
  const total = fact(data, 'deepseek-v4-pro', 'total_params')?.value;
  const active = fact(data, 'deepseek-v4-pro', 'active_params')?.value;
  const picked = fact(data, 'deepseek-v4-pro', 'experts_active')?.value;
  const experts = fact(data, 'deepseek-v4-pro', 'experts_total')?.value;
  const known = total && active && picked && experts;
  const fills = { '%ACTIVE%': known ? pct(active, total) : '—', '%ROUTED%': known ? routedText(picked, experts) : '—', '%ROUTED_SHARE%': known ? routedShare(picked, experts) : '—' };
  return TRY_THIS.map(([prompt, insight, rest]) => ({
    prompt: Object.entries(fills).reduce((text, [token, value]) => text.replace(token, value), fillText(prompt, data)),
    insight,
    rest,
  }));
}
