// model-card "Show me the math" (storyboard §7): three blocks, every number worked from the data file by the earlier
// pages' functions (math/card.js → math/memory.js). hl-act links to the active chip and share bars, hl-kv to the cache readout.
import { decodeCard, activeShare, routedShare, cacheForConversation, cachePerToken, BYTES_PER_ELEM } from '@math/card.js';
import { formatBytes } from '@math/core.js';
import { lookupFact } from '@shared/claims.js';
import { int, sharePercent } from './format.js';

const WORKED_TOKENS = 1_000_000; // the worked lines use one million tokens, as the storyboard does
const GB = 1e9;
const tex = String.raw;
const dash = tex`\text{—}`;

const entryOf = (data, id) => data?.models?.entries?.find((e) => e.id === id) ?? null;
const fact = (data, id, key) => lookupFact(data?.models, id, key)?.value ?? null;
const text = (s) => tex`\text{${s}}`;
const percent = (s) => s.replace('%', tex`\%`);

function shareBlock(data) {
  const entry = entryOf(data, 'deepseek-v4-pro');
  if (!entry) return { tex: dash };
  const display = Object.fromEntries(decodeCard(entry).map((r) => [r.key, r.display]));
  return {
    tex: tex`\text{active share} = \frac{\htmlClass{hl-act}{\text{active}}}{\text{total}} = \frac{${text(display.active_params)}}{${text(display.total_params)}} = ${percent(sharePercent(activeShare(entry)))},\qquad \text{routed used} = \frac{k}{E} = \frac{${display.experts_active}}{${display.experts_total}} = ${percent(sharePercent(routedShare(entry)))}`,
    note: 'Active parameters ÷ total parameters (2 decimals), and the experts a token runs ÷ the experts the router chooses from. DeepSeek-V4-Pro\'s card, from the data file.',
  };
}

function cacheBlock(data) {
  const entry = entryOf(data, 'minimax-m3');
  if (!entry) return { tex: dash };
  const [layers, heads, headDim] = ['layers', 'n_kv_heads', 'head_dim'].map((k) => fact(data, 'minimax-m3', k));
  const bytes = cacheForConversation(entry, WORKED_TOKENS);
  return {
    tex: tex`\text{cache per conversation} = \htmlClass{hl-kv}{\text{bytes per token}} \times \text{tokens}\;(+\ \text{fixed window or state bytes}) \qquad \text{MiniMax-M3: } 2 \cdot ${layers} \cdot ${heads} \cdot ${headDim} \cdot ${BYTES_PER_ELEM} \times 10^6 = ${text(formatBytes(bytes))}`,
    note: `Keys and values (the first 2), ${BYTES_PER_ELEM} bytes per number, for every layer, KV head and head dimension, times the tokens kept: ${int(cachePerToken(entry).bytes)} bytes per token for MiniMax-M3, here at one million tokens.`,
  };
}

function rangeBlock(data) {
  const range = fact(data, 'deepseek-v4-pro', 'kv_bytes_per_token');
  if (!Array.isArray(range)) return { tex: dash };
  const gb = range.map((b) => ((b * WORKED_TOKENS) / GB).toFixed(1));
  const bytes = range.map((b) => int(b).replace(',', '{,}'));
  return {
    tex: tex`\text{a range stays a range: } [a, b] \times n = [a n,\ b n],\qquad [${bytes[0]},\ ${bytes[1]}]\ \text{B} \times 10^6 = [${gb[0]},\ ${gb[1]}]\ \text{GB}`,
    note: 'Both ends are multiplied; nothing is averaged. DeepSeek-V4-Pro\'s cache per token is a range, so its cache per conversation is too.',
  };
}

export const mathBlocks = (data) => [shareBlock(data), cacheBlock(data), rangeBlock(data)];

export const MATH_NOTES = Object.freeze([
  'Every formula here is an earlier page\'s: parameters in [[decoder-anatomy]] and [[moe]], the cache in [[kv-cache]] and [[long-context-attention]]. Hover a highlighted term to outline its glyph on the stage.',
]);
