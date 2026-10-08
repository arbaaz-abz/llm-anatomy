// Reading a model card (model-card §6): maps each data/models.json key to its meaning and lesson, and turns a card's
// figures into costs by calling math/memory.js. It adds no architecture formula. Pure: no DOM, inputs never mutated.
import { formatCount, formatBytes, deepFreeze } from './core.js';
import { kvBytesPerToken, stackKvBytes, kvCacheBytes, sharePct } from './memory.js';

const BYTES_PER_ELEM = 2; // BF16 keys and values, the convention data/models.json's derived cache sizes use
const PCT_DECIMALS = 2; // "3.06%": a share this small needs the second decimal

// One entry per data key the page shows: label, a one-line gloss (always visible), the lesson(s) that explain it.
export const FIELD_GUIDE = deepFreeze({
  total_params: { label: 'Total parameters', gloss: 'Every weight stored, all experts included.', lesson: 'moe' },
  active_params: { label: 'Active parameters', gloss: 'Weights one token multiplies; sets compute per token.', lesson: 'decoder-anatomy' },
  layers: { label: 'Layers', gloss: 'Blocks: attention plus MLP or experts, repeated.', lesson: 'decoder-anatomy' },
  experts_total: { label: 'Routed experts', gloss: 'Small MLPs the router chooses from.', lesson: 'moe' },
  experts_active: { label: 'Experts per token', gloss: 'How many routed experts each token runs.', lesson: 'moe' },
  attention: { label: 'Attention', gloss: 'How keys and values are stored and read.', lesson: 'kv-compression' },
  kv_bytes_per_token: { label: 'Cache per token', gloss: 'Bytes of keys and values kept for each token.', lesson: 'kv-cache' },
  context_length: { label: 'Context', gloss: 'Positions the model accepts, not how well it uses them.', lesson: ['long-context-attention', 'rope'] },
  modalities: { label: 'Inputs', gloss: 'What can enter the stream.', lesson: 'multimodal' },
  optimizer: { label: 'Optimizer', gloss: 'How the weights were trained.', lesson: 'scaling-laws' },
  pretrain_tokens: { label: 'Pretraining tokens', gloss: 'Text seen in pretraining.', lesson: 'pretraining' },
  license: { label: 'License', gloss: 'What you may do with the weights.', lesson: null },
  release_date: { label: 'Released', gloss: 'When the weights or preview appeared.', lesson: null },
});

const SUFFIX_SCALE = Object.freeze({ K: 1e3, M: 1e6, B: 1e9, T: 1e12 });
const MAX_DIGITS = 6;

// The shortest formatCount text that reads back as exactly n: 1.6T, 49B, 104.2B, 116.83B (never "104B" for 104.2B).
function exactCount(n) {
  for (let digits = 3; digits <= MAX_DIGITS; digits += 1) {
    const text = formatCount(n, { digits });
    const scale = SUFFIX_SCALE[text.slice(-1)] ?? 1;
    if (Math.abs(parseFloat(text) * scale - n) <= Math.abs(n) * 1e-9) return text;
  }
  return formatCount(n, { digits: MAX_DIGITS });
}

// A byte range such as [4000, 12000] prints "4–12 kB": the unit once when both ends share it.
export function formatByteRange([low, high]) {
  const [a, b] = [formatBytes(low), formatBytes(high)];
  const [number, unit] = a.split(' ');
  return unit === b.split(' ')[1] ? `${number}–${b}` : `${a}–${b}`;
}

const NUMBER_FORMATS = Object.freeze({
  total_params: exactCount,
  active_params: exactCount,
  pretrain_tokens: exactCount,
  context_length: (n) => formatCount(n),
  kv_bytes_per_token: (n) => formatBytes(n),
});

function display(key, value) {
  if (Array.isArray(value)) return key === 'kv_bytes_per_token' ? formatByteRange(value) : value.map((v) => (NUMBER_FORMATS[key] ?? String)(v)).join('–');
  return typeof value === 'number' ? (NUMBER_FORMATS[key] ?? String)(value) : String(value);
}

function requireEntry(fn, entry) {
  if (entry === null || typeof entry !== 'object' || entry.facts === null || typeof entry.facts !== 'object') {
    throw new TypeError(`${fn}: entry must be a data/models.json entry with facts`);
  }
}

const valueOf = (entry, key) => entry.facts[key]?.value ?? null;
const isNumber = (v) => typeof v === 'number' && Number.isFinite(v);

// Rows in FIELD_GUIDE order for the keys the entry has; a range keeps both ends (spec §7: conflicts stay open).
export function decodeCard(entry) {
  requireEntry('decodeCard', entry);
  return Object.entries(FIELD_GUIDE)
    .filter(([key]) => entry.facts[key] !== undefined)
    .map(([key, field]) => {
      const fact = entry.facts[key];
      return {
        key, label: field.label, gloss: field.gloss, lesson: field.lesson, value: fact.value, display: display(key, fact.value),
        confidence: fact.confidence, isRange: Array.isArray(fact.value), sourceUrl: fact.source_url, note: fact.note ?? '',
      };
    });
}

// A share as a fraction, from two published figures; null if either is missing or a range.
function share(entry, partKey, wholeKey) {
  requireEntry('share', entry);
  const [part, whole] = [valueOf(entry, partKey), valueOf(entry, wholeKey)];
  return isNumber(part) && isNumber(whole) ? sharePct(part, whole, { decimals: PCT_DECIMALS }) / 100 : null;
}

// Published active ÷ published total (README lesson 16: the course's one definition).
export const activeShare = (entry) => share(entry, 'active_params', 'total_params');
// Routed experts each token runs ÷ routed experts the router chooses from.
export const routedShare = (entry) => share(entry, 'experts_active', 'experts_total');

// Some labs also quote an active count with the embedding table; the figure is only ever read from the data note
// ("Routed-active; 52B including embeddings"), never typed on a page.
export function activeWithEmbeddings(entry) {
  requireEntry('activeWithEmbeddings', entry);
  const found = (entry.facts.active_params?.note ?? '').match(/(\d+(?:\.\d+)?)B including embeddings/);
  return found ? Number(found[1]) * SUFFIX_SCALE.B : null;
}

const MIX = /^(\d+) full : (\d+) window$/;

// The layer groups of a stack whose keys and values are plain GQA, from config keys; null when they cannot be derived.
function derivedGroups(entry) {
  const [layers, kvHeads, headDim] = ['layers', 'n_kv_heads', 'head_dim'].map((k) => valueOf(entry, k));
  if (![layers, kvHeads, headDim].every(Number.isInteger) || !/^GQA\b/.test(String(valueOf(entry, 'attention')))) return null;
  const perLayer = kvBytesPerToken({ layers: 1, kvHeads, headDim, bytesPerElem: BYTES_PER_ELEM });
  const pattern = valueOf(entry, 'layer_pattern');
  if (pattern === null) return [{ layers, kind: 'full', bytesPerTokenPerLayer: perLayer }];
  const mix = String(pattern).match(MIX);
  const window = valueOf(entry, 'window');
  if (!mix || !Number.isInteger(window)) return null;
  const [full, sliding] = [Number(mix[1]), Number(mix[2])];
  const fullLayers = (layers * full) / (full + sliding);
  if (!Number.isInteger(fullLayers) || fullLayers < 1 || fullLayers === layers) return null;
  return [
    { layers: fullLayers, kind: 'full', bytesPerTokenPerLayer: perLayer },
    { layers: layers - fullLayers, kind: 'window', window, bytesPerTokenPerLayer: perLayer },
  ];
}

const reportedBytes = (entry) => {
  const fact = entry.facts.kv_bytes_per_token;
  return fact?.confidence === 'reported' ? (Array.isArray(fact.value) ? [...fact.value] : fact.value) : null;
};

// Cache per token with its provenance: 'derived' from config keys through math/memory.js, 'reported' from the data's
// own figure, else 'not published'. `fixed` is the window layers' constant part.
export function cachePerToken(entry) {
  requireEntry('cachePerToken', entry);
  const groups = derivedGroups(entry);
  if (groups) {
    const { perToken, fixed } = stackKvBytes({ groups, tokens: groups.find((g) => g.kind === 'window')?.window ?? 1 });
    return fixed > 0 ? { kind: 'derived', bytes: perToken, fixed } : { kind: 'derived', bytes: perToken };
  }
  const bytes = reportedBytes(entry);
  return bytes === null ? { kind: 'not published' } : { kind: 'reported', bytes };
}

function checkTokens(tokens) {
  const ends = Array.isArray(tokens) ? tokens : [tokens, tokens];
  if (ends.length !== 2 || !ends.every((t) => Number.isInteger(t) && t >= 1) || ends[0] > ends[1]) {
    throw new RangeError(`cacheForConversation: tokens must be a positive integer or a [low, high] pair, got ${JSON.stringify(tokens)}`);
  }
  return ends;
}

// Cache for one conversation of `tokens` (a number or a [low, high] range): a number, or a [low, high] pair when an
// input is a range; null when the card publishes nothing to size it from.
export function cacheForConversation(entry, tokens) {
  requireEntry('cacheForConversation', entry);
  const [fewest, most] = checkTokens(tokens);
  const groups = derivedGroups(entry);
  const perToken = cachePerToken(entry);
  if (perToken.kind === 'not published') return null;
  const bytesEnds = Array.isArray(perToken.bytes) ? perToken.bytes : [perToken.bytes, perToken.bytes];
  const [low, high] = groups
    ? [stackKvBytes({ groups, tokens: fewest }).total, stackKvBytes({ groups, tokens: most }).total]
    : [kvCacheBytes({ bytesPerToken: bytesEnds[0], tokens: fewest }), kvCacheBytes({ bytesPerToken: bytesEnds[1], tokens: most })];
  return low === high ? low : [low, high];
}
