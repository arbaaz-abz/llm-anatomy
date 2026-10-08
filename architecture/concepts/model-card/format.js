// model-card pure formatters (no DOM): every string the toy and the stage print about shares and cache sizes.
// Percentages come from sharePct (math/memory.js), byte sizes from formatBytes (decimal, 3 significant figures).
import { sharePct } from '@math/memory.js';
import { formatBytes } from '@math/core.js';
import { formatByteRange } from '@math/card.js';
import { lookupFact } from '@shared/claims.js';

export const NOT_PUBLISHED = 'not published';
export const CONTEXT_SHORT = 131_072; // the toy's "ordinary conversation" length, 2^17
export const ZOOM_FRACTION = 0.1; // the second share bar zooms on the first 10% of the total
export const INITIAL_STATE = Object.freeze({ left: 'deepseek-v4-pro', right: 'kimi-k3', context: 'own' });

// Whole numbers with thousands separators and a real minus: "122,880".
export const int = (n) => Math.round(n).toLocaleString('en-US').replace(/^-/, '−');

// A fraction as the course prints an active or routed share: two decimals ("3.06%").
export const sharePercent = (fraction) => `${sharePct(fraction, 1, { decimals: 2 }).toFixed(2)}%`;

// Share of one GPU's memory for a byte count or a [low, high] range (one decimal, the unit printed once).
export function gpuShareText(bytes, hbmBytes) {
  if (bytes === null) return NOT_PUBLISHED;
  const one = (b) => sharePct(b, hbmBytes).toFixed(1);
  return Array.isArray(bytes) ? `${one(bytes[0])}–${one(bytes[1])}%` : `${one(bytes)}%`;
}

// Cache per token with its exact bytes beside the rounded size (storyboard: "70,272 B ≈ 70.3 kB").
export function cacheTokenText(cache) {
  if (cache.kind === 'not published') return NOT_PUBLISHED;
  if (Array.isArray(cache.bytes)) return `${cache.bytes.map(int).join('–')} B ≈ ${formatByteRange(cache.bytes)}`;
  const fixed = cache.fixed ? ` + ${formatBytes(cache.fixed)} fixed` : '';
  return `${int(cache.bytes)} B ≈ ${formatBytes(cache.bytes)}${fixed}`;
}

export const cacheConversationText = (bytes) => {
  if (bytes === null) return NOT_PUBLISHED;
  return Array.isArray(bytes) ? formatByteRange(bytes) : formatBytes(bytes);
};

const PROVENANCE = Object.freeze({ derived: 'derived from the config', reported: 'a reported estimate', 'not published': '' });
export const provenanceText = (kind) => PROVENANCE[kind] ?? '';

// A single-valued context is also printed exactly: "1,048,576 tokens" under the rounded "1.05M".
export const exactTokens = (value) => (typeof value === 'number' ? `${int(value)} tokens` : '');

const GPU_ID = 'h100'; // the 80 GB GPU the cache lines are measured against

// One GPU's memory in bytes, from data/hardware.json (decimal GB, as vendors print it).
export function hbmBytes(data) {
  const gb = lookupFact(data?.hardware, GPU_ID, 'hbm_gb')?.value;
  if (!Number.isFinite(gb)) throw new RangeError(`model-card: data/hardware.json has no ${GPU_ID}.hbm_gb`);
  return gb * 1e9;
}
