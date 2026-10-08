// decoder-anatomy pure helpers (no DOM): the toy's config, the "Check my work" text, gap lines and percentages.
// Every number comes from math/params.js or math/memory.js; this file only arranges and formats them.
import { publishedGap } from '@math/params.js';
import { sharePct } from '@math/memory.js';
import { formatCount } from '@math/core.js';

export const PART_ORDER = Object.freeze(['embedding', 'positional', 'attention', 'mlp', 'experts', 'router', 'norms', 'head']);
export const TOY_LIMITS = Object.freeze({ layers: [1, 8], dModel: [8, 16, 32, 64], experts: [0, 2, 4, 8, 16] });
const TOY_TOP_K = 2;

export const int = (n) => (n < 0 ? '−' : '') + Math.abs(n).toLocaleString('en-US');

// The toy preset with the sliders applied: d_head = d/2 (2 heads), dense hidden 2·d, expert hidden d, top-2.
export function toyConfig(base, { layers, dModel, experts }) {
  if (!Number.isInteger(layers) || layers < TOY_LIMITS.layers[0] || layers > TOY_LIMITS.layers[1]) throw new RangeError(`toyConfig: layers must be 1–8, got ${layers}`);
  if (!TOY_LIMITS.dModel.includes(dModel)) throw new RangeError(`toyConfig: dModel must be one of ${TOY_LIMITS.dModel.join(', ')}, got ${dModel}`);
  if (!TOY_LIMITS.experts.includes(experts)) throw new RangeError(`toyConfig: experts must be one of ${TOY_LIMITS.experts.join(', ')}, got ${experts}`);
  return {
    ...base,
    layers,
    dModel,
    attention: { ...base.attention, dHead: dModel / base.attention.nHeads },
    mlp: { ...base.mlp, hidden: 2 * dModel },
    moe: experts ? { routed: experts, shared: 0, topK: TOY_TOP_K, hidden: dModel, denseLayers: 0 } : null,
  };
}

// A signed percentage with two decimals and a real minus ("−0.23%", "+0.00%"), through sharePct.
export function signedPct(x) {
  const pct = sharePct(x, 1, { decimals: 2 });
  return `${pct < 0 ? '−' : '+'}${Math.abs(pct).toFixed(2)}%`;
}

// "computed 174.6B vs published 175B (−0.23%)"; a gap that rounds to 0.00% prints "(matches)", never a signed zero.
export function gapLine(computed, published) {
  const gap = publishedGap(computed, published);
  const shown = sharePct(gap, 1, { decimals: 2 }) === 0 ? 'matches' : signedPct(gap);
  return `computed ${formatCount(computed, { digits: 5 })} vs published ${formatCount(published, { digits: 5 })} (${shown})`;
}

// ---- "Check my work" (the toy preset only; real presets add biases and learned positions) ----
const EQ_COL = 61;
const MIN_GAP = 3;
const BLOCK_LABEL = 14;
const MATRICES = 26;
const TOP_LABEL = 21;
const padTo = (text, col) => text + ' '.repeat(Math.max(MIN_GAP, col - text.length));
const columns = (cells) => cells.map(([text, width], i) => (i < cells.length - 1 ? padTo(text, width) : text)).join('');

function blockRows(config, b) {
  const { dModel: d, layers, attention, mlp, moe } = config;
  const q = attention.nHeads * attention.dHead;
  const rows = [[columns([['  attention', BLOCK_LABEL], ['W_Q, W_K, W_V, W_O', MATRICES], [`4 × (${d} × ${q})`]]), b.perLayer.attention]];
  if (moe) {
    const he = moe.hidden;
    rows.push([columns([['  experts', BLOCK_LABEL], [`${moe.routed} × (W_in, W_gate, W_out)`, MATRICES], [`${moe.routed} × 3 × (${d} × ${he})`]]), b.parts.experts / layers]);
    rows.push([columns([['  router', BLOCK_LABEL], [`W_router [${d} × ${moe.routed}]`, MATRICES], [`${d} × ${moe.routed}`]]), b.parts.router / layers]);
  } else {
    rows.push([columns([['  MLP', BLOCK_LABEL], ['W_in, W_gate, W_out', MATRICES], [`3 × (${d} × ${mlp.hidden})`]]), b.perLayer.mlp]);
  }
  rows.push([columns([['  norms', BLOCK_LABEL], [`2 × ${d}`]]), b.perLayer.norms]);
  return rows;
}

// The default toy state gives the storyboard's §6 box exactly; other slider states fill the same template.
export function checkWork(config, b) {
  const { vocab, dModel: d, layers } = config;
  const blocks = blockRows(config, b);
  const perBlock = blocks.reduce((sum, [, v]) => sum + v, 0);
  const finalNorm = b.parts.norms - layers * b.perLayer.norms;
  const active = config.moe ? 'active = total − unused experts − embedding lookup' : 'active = total − embedding lookup';
  const rows = [
    ...blocks,
    [columns([['  block', BLOCK_LABEL]]), perBlock],
    [`× ${layers} block${layers === 1 ? '' : 's'}`, perBlock * layers],
    [columns([['embedding table', TOP_LABEL], [`${vocab} × ${d}`]]), b.parts.embedding],
    ['final norm', finalNorm],
    [columns([['unembedding', TOP_LABEL], [`${vocab} × ${d}`]]), b.parts.head],
    ['total', b.total],
    [active, b.active],
  ];
  const width = Math.max(5, ...rows.map(([, v]) => int(v).length));
  return ['per block', ...rows.map(([left, v]) => `${padTo(left, EQ_COL)}= ${int(v).padStart(width)}`)].join('\n');
}

// The per-block parts as table rows: attention, MLP (when dense), "expert 192 × 8" (value + multiplier), norms (exact counts for the toy).
export function perBlockRows(config, b, { exact = true } = {}) {
  const count = (n) => (exact ? int(n) : formatCount(n));
  const { moe } = config;
  const rows = [{ label: 'attention', value: count(b.perLayer.attention) }];
  if (!moe || moe.denseLayers > 0) rows.push({ label: 'MLP', value: count(b.perLayer.mlp) });
  if (moe) rows.push({ label: 'expert', value: count(b.perLayer.expert), sub: `× ${moe.routed + moe.shared}` });
  rows.push({ label: 'norms', value: count(b.perLayer.norms) });
  return rows;
}

// "attention 256 · MLP 384 · norms 16", "attention 256 · expert 192 × 8 · norms 16": the same rows as one line.
export const perBlockLine = (config, b, options) => perBlockRows(config, b, options).map((r) => `${r.label} ${r.value}${r.sub ? ` ${r.sub}` : ''}`).join(' · ');

// The shareBar's five hues (Task 4 mapping): 1 embedding + positional · 2 attention · 3 MLP + experts · 4 other · 5 head.
export function barParts(parts) {
  const big = (parts.mlp ?? 0) && (parts.experts ?? 0) ? 'MLP + experts' : (parts.experts ? 'experts' : 'MLP');
  const list = [
    { name: 'embedding', value: (parts.embedding ?? 0) + (parts.positional ?? 0), hue: 1 },
    { name: 'attention', value: parts.attention ?? 0, hue: 2 },
    { name: big, value: (parts.mlp ?? 0) + (parts.experts ?? 0), hue: 3 },
    { name: 'other (router, norms)', value: (parts.router ?? 0) + (parts.norms ?? 0), hue: 4 },
    { name: 'head', value: parts.head ?? 0, hue: 5 },
    { name: 'not published', value: parts.unknown ?? 0, unknown: true },
  ];
  return list.filter((p) => p.value > 0);
}
