// decoder-anatomy toy view model (pure, no DOM): state + data → every string the toy prints.
// Counts come from math/params.js, percentages from sharePct / formatShare; published figures from ctx.data.
import { paramBreakdown, partialBreakdown, mlpParams, PRESETS, V4_PRO_PARTIAL } from '@math/params.js';
import { sharePct } from '@math/memory.js';
import { formatCount } from '@math/core.js';
import { lookupFact } from '@shared/claims.js';
import { formatShare } from '@shared/glyphs/bars.js';
import { PART_ORDER, int, toyConfig, checkWork, gapLine, perBlockLine, barParts } from './format.js';

export const PRESET_CHIPS = Object.freeze([
  { value: 'toy', label: 'toy' },
  { value: 'gpt3', label: 'GPT-3 175B', dataId: 'gpt-3' },
  { value: 'gptOss120b', label: 'gpt-oss-120b', dataId: 'gpt-oss-120b' },
  { value: 'deepseekV3', label: 'DeepSeek-V3', dataId: 'deepseek-v3' },
  { value: 'deepseekV4Pro', label: 'DeepSeek-V4-Pro (partial)', dataId: 'deepseek-v4-pro' },
]);

export const PART_LABELS = Object.freeze({
  embedding: 'embedding table', positional: 'positions (learned)', attention: 'attention', mlp: 'MLP (dense)', experts: 'experts',
  router: 'router', norms: 'norms', head: 'unembedding (head)', unknown: 'not published',
});

export const INITIAL_STATE = Object.freeze({ preset: 'toy', layers: 2, dModel: 8, experts: 0 });
export const EXPERTS_EDGE_NOTE = '2 of 2 experts used: this is a dense MLP with a router';
const REAL_CHECK_NOTE = 'The line-by-line count is for the toy preset. Real presets use the same formulas, plus the biases, learned positions and latent projections their configs list.';

const fact = (data, id, key) => lookupFact(data?.models, id, key)?.value ?? null;
const pctText = (part, whole) => `${sharePct(part, whole).toFixed(1)}%`;
const shareOrDash = (value, share) => (value > 0 ? formatShare(share) : '—');

function attentionText({ attention: a }) {
  if (a.kind === 'mla') return `MLA, ${a.nHeads} heads`;
  return `${a.nHeads} heads × ${a.dHead}${a.nKvHeads < a.nHeads ? ` (${a.nKvHeads} KV heads)` : ''}`;
}

function mlpText({ mlp, moe }) {
  if (!moe) return `dense ${mlp.kind === 'gelu' ? 'GELU' : 'SwiGLU'} MLP, hidden ${int(mlp.hidden)}`;
  const dense = moe.denseLayers ? `, first ${moe.denseLayers} blocks dense` : '';
  return `${moe.routed}${moe.shared ? ` + ${moe.shared}` : ''} experts of hidden ${int(moe.hidden)}, top-${moe.topK}${dense}`;
}

export function specLine(preset, config) {
  if (preset === 'toy') return `Fixed toy constants: vocabulary ${config.vocab}, ${config.attention.nHeads} heads, SwiGLU, RMSNorm, no biases, untied embedding and unembedding.`;
  const label = PRESET_CHIPS.find((c) => c.value === preset).label;
  const extras = [config.norm === 'layernorm' ? 'LayerNorm' : 'RMSNorm', config.positional === 'learned' ? `learned positions (${int(config.maxPositions)})` : 'RoPE',
    config.biases ? 'biases' : 'no biases', config.tiedEmbeddings ? 'tied embedding' : 'untied'];
  return `${label}: vocab ${int(config.vocab)} · d_model ${int(config.dModel)} · ${config.layers} blocks · ${attentionText(config)} · ${mlpText(config)} · ${extras.join(', ')}.`;
}

function activeNote(config, b, count) {
  const head = `Active ${count(b.active)} of ${count(b.total)}`;
  if (config.tiedEmbeddings) return `${head}: the embedding table is also the unembedding (tied), so it is multiplied and counted once.`;
  return `${head}: active leaves out the ${count(b.parts.embedding)}-parameter embedding table.`;
}

function activeGapLine(preset, b, published) {
  if (preset === 'gpt3') return 'Dense and tied: active = total, and the paper prints no separate active count.';
  return published?.active ? `active: ${gapLine(b.active, published.active)}` : '';
}

function fullView(preset, config, data) {
  const exact = preset === 'toy';
  const b = paramBreakdown(config);
  const count = (n) => (exact ? int(n) : formatCount(n));
  const dataId = PRESET_CHIPS.find((c) => c.value === preset).dataId;
  const published = dataId ? { total: fact(data, dataId, 'total_params'), active: fact(data, dataId, 'active_params') } : null;
  return {
    exact,
    total: count(b.total),
    active: count(b.active),
    activePct: `${pctText(b.active, b.total)} of total`,
    activeWithEmbedding: count(b.activeWithEmbedding),
    activeNote: activeNote(config, b, count),
    perBlock: perBlockLine(config, b, { exact }),
    rows: PART_ORDER.map((part) => ({ part, label: PART_LABELS[part], count: count(b.parts[part]), share: shareOrDash(b.parts[part], b.share[part]), activeShare: shareOrDash(b.activeShare[part], b.activeShare[part]) })),
    bar: barParts(b.parts),
    gap: published?.total ? gapLine(b.total, published.total) : 'The toy has no published total: every count here is exact.',
    activeGap: exact ? '' : activeGapLine(preset, b, published),
    checkWork: exact ? checkWork(config, b) : REAL_CHECK_NOTE,
    expertsNote: exact && config.moe && config.moe.routed === config.moe.topK ? EXPERTS_EDGE_NOTE : '',
    spec: specLine(preset, config),
  };
}

// DeepSeek-V4-Pro: the experts are derivable from confirmed facts; the rest of the published total is "not published".
function partialView(data) {
  const p = partialBreakdown({ known: V4_PRO_PARTIAL.known, publishedTotal: V4_PRO_PARTIAL.publishedTotal });
  const a = partialBreakdown({ known: V4_PRO_PARTIAL.activeKnown, publishedTotal: V4_PRO_PARTIAL.publishedActive });
  const f = (key) => fact(data, 'deepseek-v4-pro', key);
  const expert = f('expert_hidden') && f('d_model') ? mlpParams({ kind: 'swiglu', hidden: f('expert_hidden') }, f('d_model')) : null;
  const unknownLabel = `${PART_LABELS.unknown} (${V4_PRO_PARTIAL.assumption})`;
  return {
    exact: false,
    total: formatCount(p.total),
    active: formatCount(a.total),
    activePct: `${pctText(a.total, p.total)} of total`,
    activeWithEmbedding: 'not published',
    activeNote: `Active ${formatCount(a.total)} of ${formatCount(p.total)}, both published: ${formatCount(a.parts.experts)} of it is derivable (${f('experts_active') + f('experts_shared')} experts × ${f('layers')} blocks), the rest is not published.`,
    perBlock: expert ? `expert ${formatCount(expert)} × ${f('experts_total') + f('experts_shared')} · the rest is not published` : 'not published',
    rows: [
      { part: 'experts', label: PART_LABELS.experts, count: formatCount(p.parts.experts), share: formatShare(p.share.experts), activeShare: formatShare(a.share.experts) },
      { part: 'unknown', label: unknownLabel, count: formatCount(p.parts.unknown), share: formatShare(p.share.unknown), activeShare: formatShare(a.share.unknown) },
    ],
    bar: barParts(p.parts),
    gap: `published ${formatCount(p.total)} total / ${formatCount(a.total)} active, printed back; the experts are ${formatShare(p.share.experts)} of the total ${V4_PRO_PARTIAL.assumption}.`,
    activeGap: '',
    checkWork: REAL_CHECK_NOTE,
    expertsNote: '',
    spec: `DeepSeek-V4-Pro: d_model ${int(f('d_model'))} · ${f('layers')} blocks · ${f('experts_total')} + ${f('experts_shared')} experts of hidden ${int(f('expert_hidden'))}, ${f('experts_active')} + ${f('experts_shared')} used per token · the rest of the config is not published.`,
  };
}

export function toyView(state, data) {
  if (state.preset === 'deepseekV4Pro') return partialView(data);
  if (state.preset === 'toy') return fullView('toy', toyConfig(PRESETS.toy, state), data);
  if (!PRESETS[state.preset]) throw new RangeError(`toyView: unknown preset "${state.preset}"`);
  return fullView(state.preset, PRESETS[state.preset], data);
}

// Try-this 2's far end: Kimi K3's embedding + unembedding as a share of its total (from data/models.json).
export function kimiLine(data) {
  const [vocab, d, total, date] = ['vocab_size', 'd_model', 'total_params', 'release_date'].map((k) => fact(data, 'kimi-k3', k));
  if (!vocab || !d || !total) return '';
  const p = partialBreakdown({ known: { embedding: vocab * d, head: vocab * d }, publishedTotal: total });
  const tables = p.parts.embedding + p.parts.head;
  return `Kimi K3 (${String(date).slice(0, 4)}): ${formatCount(tables)} of ${formatCount(total)}, ${sharePct(tables, total, { decimals: 2 }).toFixed(2)}%`;
}

// The "explain the gap" line under the DeepSeek-V3 chip.
export function v3GapLine(data) {
  const [total, mtp] = ['total_params', 'mtp_params'].map((k) => fact(data, 'deepseek-v3', k));
  if (!total || !mtp) return '';
  return `Explain the gap: the Hugging Face checkpoint is ${formatCount(total + mtp)} because it also ships the ${formatCount(mtp)} multi-token-prediction (MTP) module; the paper's ${formatCount(total)} is the main model alone.`;
}
