// model-card toy view model (pure, no DOM): state + data → every string the toy prints.
// Fields come from decodeCard (math/card.js), costs from activeShare / routedShare / cachePerToken / cacheForConversation,
// percentages from sharePct, sizes from formatBytes; nothing here is typed in except the card names.
import { FIELD_GUIDE, decodeCard, activeShare, routedShare, activeWithEmbeddings, cachePerToken, cacheForConversation } from '@math/card.js';
import { glossFor, LABS_DIFFER_KEYS } from './glosses.js';
import {
  NOT_IN_DATA, CONTEXT_SHORT, INITIAL_STATE, hbmBytes, int, sharePercent, gpuShareText, cacheTokenText, cacheConversationText, provenanceText, exactTokens,
} from './format.js';

export { INITIAL_STATE };

export const CARDS = Object.freeze([
  { id: 'deepseek-v4-pro', name: 'DeepSeek-V4-Pro' },
  { id: 'kimi-k3', name: 'Kimi K3' },
  { id: 'qwen3.8', name: 'Qwen3.8' },
  { id: 'glm-5.3', name: 'GLM-5.3' },
  { id: 'minimax-m3', name: 'MiniMax-M3' },
  { id: 'mistral-large-4', name: 'Mistral Large 4' },
  { id: 'gpt-oss-120b', name: 'gpt-oss-120b' },
]);
export const NONE = 'none';
export const CONTEXT_OPTIONS = Object.freeze([{ value: 'own', label: 'the card\'s own context' }, { value: String(CONTEXT_SHORT), label: int(CONTEXT_SHORT) }]);

const nameOf = (id) => CARDS.find((c) => c.id === id)?.name ?? id;
const lessonsOf = (key) => [FIELD_GUIDE[key].lesson].flat().filter(Boolean);

function entryOf(data, id) {
  const entry = data?.models?.entries?.find((e) => e.id === id);
  if (!entry) throw new RangeError(`model-card toy: data/models.json has no entry "${id}"`);
  return entry;
}

// The conversation length the cost rows use: the card's own context (a number or a range), or the short length.
export function tokensFor(entry, context) {
  if (context !== 'own') return Number(context);
  const own = entry.facts.context_length?.value;
  return own === undefined ? null : own;
}

// One of the 13 field rows. A missing field says so; the cache row shows how the figure was reached.
function fieldRow(entry, rows, key, cache) {
  const guide = FIELD_GUIDE[key];
  const base = { key, label: guide.label, gloss: guide.gloss, lessons: lessonsOf(key) };
  const row = rows.find((r) => r.key === key);
  if (!row) return { ...base, value: NOT_IN_DATA, present: false, tag: '', detail: '', sourceUrl: '' };
  if (key === 'kv_bytes_per_token') {
    return { ...base, value: cacheTokenText(cache), present: true, tag: cache.kind === 'reported' ? 'reported' : '', detail: provenanceText(cache.kind), sourceUrl: row.sourceUrl };
  }
  return {
    ...base, value: row.display, present: true, tag: row.confidence === 'reported' ? 'reported' : '',
    detail: key === 'context_length' && !row.isRange ? exactTokens(row.value) : '', sourceUrl: row.sourceUrl,
  };
}

// The learner glosses of this card's conflicts and carried-over figures (never the data file's maintainer notes).
const labsDiffer = (data, id, name, rows) => LABS_DIFFER_KEYS
  .filter((k) => k.startsWith(`${id}.`))
  .map((k) => ({ key: k.slice(id.length + 1), text: glossFor(data, id, k.slice(id.length + 1)) }))
  .filter((n) => n.text !== null)
  .map((n) => ({ label: `${name} · ${rows.find((r) => r.key === n.key).label}`, text: n.text }));

function costs(entry, cache, tokens, hbm) {
  const conversation = tokens === null ? null : cacheForConversation(entry, tokens);
  const share = activeShare(entry);
  const routed = routedShare(entry);
  const withEmbeddings = activeWithEmbeddings(entry);
  const total = entry.facts.total_params?.value;
  const ends = [conversation].flat();
  return {
    activeShare: share === null ? NOT_IN_DATA : sharePercent(share),
    activeEmbedding: withEmbeddings !== null && Number.isFinite(total) ? sharePercent(withEmbeddings / total) : null,
    routedShare: routed === null ? NOT_IN_DATA : sharePercent(routed),
    cacheToken: cacheTokenText(cache),
    cacheTokenSub: provenanceText(cache.kind),
    cacheConversation: cacheConversationText(conversation),
    gpuShare: gpuShareText(conversation, hbm),
    gpuSub: ends.some((b) => b !== null && b > hbm) ? 'more than one GPU\'s memory' : '',
  };
}

function bars(entry) {
  const [total, active] = [entry.facts.total_params?.value, entry.facts.active_params?.value];
  if (!Number.isFinite(total) || !Number.isFinite(active) || active >= total) return null;
  const share = activeShare(entry);
  return { total, active, formula: `active ÷ total = ${sharePercent(share)}`, line: `active ${sharePercent(share)} · not used ${sharePercent((total - active) / total)}` };
}

function column(side, id, state, data, hbm) {
  const entry = entryOf(data, id);
  const rows = decodeCard(entry);
  const cache = cachePerToken(entry);
  const tokens = tokensFor(entry, state.context);
  const name = nameOf(id);
  return {
    side, id, name,
    rows: Object.keys(FIELD_GUIDE).map((key) => fieldRow(entry, rows, key, cache)),
    costs: costs(entry, cache, tokens, hbm),
    bars: bars(entry),
    notes: labsDiffer(data, id, name, rows),
    tokensText: tokens === null ? NOT_IN_DATA : (Array.isArray(tokens) ? `${tokens.map(int).join('–')} tokens` : `${int(tokens)} tokens`),
  };
}

export function toyView(state, data) {
  const hbm = hbmBytes(data);
  const columns = [column('left', state.left, state, data, hbm), ...(state.right === NONE ? [] : [column('right', state.right, state, data, hbm)])];
  return {
    columns,
    hbmGb: hbm / 1e9,
    conversationLabel: state.context === 'own' ? 'at each card\'s own context' : `at ${int(Number(state.context))} tokens`,
    showEmbedding: columns.some((c) => c.costs.activeEmbedding !== null),
  };
}

// ---- try this (storyboard §6): each prompt leads to a named insight; every number is computed from the cards ----

function describe(data, id, context = 'own') {
  const entry = entryOf(data, id);
  const cache = cachePerToken(entry);
  const tokens = tokensFor(entry, context);
  return { entry, cache, tokens, conversation: tokens === null ? null : cacheForConversation(entry, tokens), rows: Object.fromEntries(decodeCard(entry).map((r) => [r.key, r])) };
}

export function tryThis(data) {
  const [v4, kimi, minimax, ossLong, glm, mistral] = ['deepseek-v4-pro', 'kimi-k3', 'minimax-m3', 'gpt-oss-120b', 'glm-5.3', 'mistral-large-4'].map((id) => describe(data, id));
  const card = (d) => `${d.rows.total_params.display} / ${d.rows.active_params.display} (${sharePercent(activeShare(d.entry))})`;
  const oneMillion = cacheForConversation(minimax.entry, 1_000_000);
  return [
    {
      prompt: `${nameOf('deepseek-v4-pro')} against ${nameOf('kimi-k3')}: ${card(v4)} against ${card(kimi)}; routed experts per token ${sharePercent(routedShare(v4.entry))} against ${sharePercent(routedShare(kimi.entry))}, lower than the active share because shared experts, attention and the head run for every token too.`,
      insight: 'compare active parameters for per-token cost and total for memory; the bigger total is not the bigger bill per token.',
    },
    {
      prompt: `${nameOf('deepseek-v4-pro')} against ${nameOf('minimax-m3')} at their own contexts: cache for one conversation ${cacheConversationText(v4.conversation)} (a reported estimate) against ${cacheConversationText(minimax.conversation)} (derived: ${int(minimax.cache.bytes)} bytes per token, or ${cacheConversationText(oneMillion)} at exactly one million tokens). Then pick ${nameOf('gpt-oss-120b')}: ${cacheTokenText(ossLong.cache)}, ${cacheConversationText(ossLong.conversation)} at its ${int(ossLong.tokens)} tokens.`,
      insight: 'the attention line, not the context number, decides what a long conversation costs.',
    },
    {
      prompt: `${nameOf('glm-5.3')} against ${nameOf('mistral-large-4')}: layers ${glm.rows.layers.display} (config against paper), context ${mistral.rows.context_length.display} (claimed against measured, both reported), Mistral active ${mistral.rows.active_params.display} (a higher figure with embeddings), GLM active ${glm.rows.active_params.display} (reported).`,
      insight: 'a card is a set of claims with sources; when they disagree, keep the range and the sources, and prefer the config file.',
    },
  ];
}

