// "In today's models" claims (pure): on-page wording with {entry.key} placeholders filled from data/*.json,
// so every dated number on a page comes from the data files (spec §7).
//   {gpt-3.layers}            models.json, default format     {hw:h100.hbm_gb}   hardware.json
//   {deepseek-v3.total_params|count}  a named format (CLAIM_FORMATS)
import { formatCount, formatBytes } from '../math/core.js';

const REF = /\{(?:(hw):)?([a-z0-9][a-z0-9.-]*)\.([a-z0-9_]+)(?:\|([a-z0-9]+))?\}/g;
const realMinus = (s) => s.replace(/^-/, '−');

export const CLAIM_FORMATS = Object.freeze({
  raw: (v) => String(v),
  int: (v) => realMinus(Math.round(v).toLocaleString('en-US')),
  count: (v) => formatCount(v),
  count4: (v) => formatCount(v, { digits: 4 }),
  bytes: (v) => formatBytes(v),
  year: (v) => String(v).slice(0, 4),
});

export function lookupFact(dataset, entryId, key) {
  const entry = dataset?.entries?.find((e) => e.id === entryId);
  return entry?.facts?.[key] ?? null;
}

function formatValue(value, formatName) {
  const name = formatName ?? (typeof value === 'number' || Array.isArray(value) ? 'int' : 'raw');
  const format = CLAIM_FORMATS[name];
  if (!format) throw new RangeError(`fillClaim: unknown format "${name}" (use ${Object.keys(CLAIM_FORMATS).join(', ')})`);
  return Array.isArray(value) ? `${format(value[0])}–${format(value[1])}` : format(value);
}

// → { segments: [{ type: 'text', text } | { type: 'fact', text, ref, note }], sources: [url…], reported, missing: [ref…] }
export function fillClaim(claim, data) {
  if (typeof claim !== 'string') throw new TypeError('fillClaim: claim must be a string');
  const segments = [];
  const sources = [];
  const missing = [];
  let reported = false;
  let last = 0;
  for (const m of claim.matchAll(REF)) {
    const [ref, hw, entryId, key, formatName] = m;
    if (m.index > last) segments.push({ type: 'text', text: claim.slice(last, m.index) });
    last = m.index + ref.length;
    const fact = lookupFact(hw ? data?.hardware : data?.models, entryId, key);
    if (!fact) {
      missing.push(ref.slice(1, -1));
      segments.push({ type: 'text', text: '—' });
      continue;
    }
    segments.push({ type: 'fact', text: formatValue(fact.value, formatName), ref: ref.slice(1, -1), note: fact.note ?? '' });
    if (!sources.includes(fact.source_url)) sources.push(fact.source_url);
    reported = reported || fact.confidence === 'reported';
  }
  if (last < claim.length) segments.push({ type: 'text', text: claim.slice(last) });
  return { segments, sources, reported, missing };
}
