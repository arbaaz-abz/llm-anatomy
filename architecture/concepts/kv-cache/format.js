// Pure formatters for the kv-cache toy and its "Check my work" boxes (storyboard §6). No DOM.
import { formatBytes, formatCount } from '@math/core.js';
import { decodeWork, sharePct } from '@math/memory.js';

const MINUS = '−';
const realMinus = (s) => s.replace(/^-/, MINUS);
const EQUALS_COLUMN = 41; // the storyboard's "=" sits in column 41 of every sum line
const SHORT_SUM = 4; // up to this many passes the sum lists every term
export const COMPACT_FROM = 10_000; // "formatCount past 10,000"

// Where the toy opens: the animation's reply of four tokens after a four-token prompt, the toy model at 2,048 tokens.
export const INITIAL_STATE = Object.freeze({
  cache: true, prompt: 4, reply: 4, model: 'toy', context: 2048, sequences: 1, layers: 2, kvHeads: 2, headDim: 4, bytes: 2,
});
export const SLIDER_VALUES = Object.freeze({
  prompt: [4, 16, 128, 1000, 8192, 100_000],
  reply: [1, 2, 4, 16, 128, 1000],
  context: [2048, 8192, 32_768, 131_072, 262_144, 1_048_576],
  sequences: [1, 2, 4, 8, 16, 32, 64],
  headDim: [4, 64, 128],
  bytes: [1, 2],
});
export const SHAPE_LIMITS = Object.freeze({ layers: [1, 96], kvHeads: [1, 96] });

export const int = (n) => realMinus(Math.round(n).toLocaleString('en-US'));
export const exactBytes = (n) => `${int(n)} B`;
// Exact bytes for a number or a [low, high] range: "4,000–12,000 B".
export const exactSpan = (value) => (Array.isArray(value) ? `${int(value[0])}–${int(value[1])} B` : exactBytes(value));
export const plural = (n, one, many = `${one}s`) => `${int(n)} ${n === 1 ? one : many}`;

// Three significant figures: "3.14×", "750×", "1,180×".
export function ratioText(x) {
  if (!(x > 0) || !Number.isFinite(x)) throw new RangeError(`ratioText: x must be a finite number > 0, got ${x}`);
  const rounded = Number(x.toPrecision(3));
  return `${rounded >= 1000 ? int(rounded) : String(rounded)}×`;
}

// A count: the exact integer, plus a compact form beside it once it passes 10,000.
export const compactSub = (n) => (n >= COMPACT_FROM ? formatCount(n) : '');

// A single number or a [low, high] range, both ends formatted the same way.
export const spanText = (value, format) => (Array.isArray(value) ? `${format(value[0])}–${format(value[1])}` : format(value));

// Share of a GPU: sharePct (math/memory.js) at one decimal; "<0.1%" for a real but tiny cache.
export function shareText(bytes, capacity) {
  const pct = sharePct(bytes, capacity);
  return pct === 0 && bytes > 0 ? '<0.1%' : `${pct.toFixed(1)}%`;
}

export function fitText(bytes, capacity) {
  const [low, high] = Array.isArray(bytes) ? bytes : [bytes, bytes];
  if (low > capacity) return 'does not fit (before the weights)';
  return high > capacity ? 'may not fit at the high end (before the weights)' : 'fits (before the weights)';
}

// Panel A: whole-reply work with and without the cache, from the one definition (decodeWork).
export function workFor({ prompt, reply, cache }) {
  return { current: decodeWork({ prompt, generated: reply, cache }), flipped: decodeWork({ prompt, generated: reply, cache: !cache }) };
}

// ---- "Check my work", panel A ----
function noCacheTerms(prompt, reply) {
  const at = (i) => int(prompt + i);
  if (reply === 1) return at(0);
  if (reply <= SHORT_SUM) return Array.from({ length: reply }, (_, i) => at(i)).join(' + ');
  return `${at(0)} + ${at(1)} + … + ${at(reply - 1)}`;
}

function cacheTerms(prompt, reply) {
  const first = `${int(prompt)} (prefill)`;
  if (reply === 1) return first;
  if (reply <= SHORT_SUM) return [first, ...Array.from({ length: reply - 1 }, () => '1')].join(' + ');
  return `${first} + ${int(reply - 1)} × 1`;
}

const withEquals = (left, right) => `${left.length >= EQUALS_COLUMN ? `${left} ` : left.padEnd(EQUALS_COLUMN)}= ${right}`;

export function checkWorkA({ prompt, reply }) {
  const off = decodeWork({ prompt, generated: reply, cache: false }).positions;
  const on = decodeWork({ prompt, generated: reply, cache: true }).positions;
  const width = Math.max(int(off).length, int(on).length);
  return [
    withEquals(`no cache: ${noCacheTerms(prompt, reply)}`, `${int(off).padStart(width)} positions`),
    withEquals(`cache:    ${cacheTerms(prompt, reply)}`, `${int(on).padStart(width)} positions`),
  ].join('\n');
}

// ---- "Check my work", panel B ----
// shape: { kind: 'mha' | 'mla' | 'range', ... } as built by toy-view.js; bytesPerToken is a number or, for 'range', [low, high].
const HEAD = 'bytes per token = ';
const CONTINUE = ' '.repeat(HEAD.length - 2); // the second "=" lines up under the first

function formulaLines(shape, bytesPerToken) {
  const bytes = plural(shape.bytesPerElem ?? 2, 'byte');
  if (shape.kind === 'mha') {
    return [`${HEAD}2 (K and V) × ${plural(shape.layers, 'block')} × ${plural(shape.kvHeads, 'KV head')} × ${plural(shape.headDim, 'number')} × ${bytes}`, `${CONTINUE}= ${exactBytes(bytesPerToken)}`];
  }
  if (shape.kind === 'mla') {
    return [`${HEAD}${plural(shape.layers, 'block')} × (${int(shape.dLatent)} latent + ${int(shape.dRope)} position key) numbers × ${bytes}`, `${CONTINUE}= ${exactBytes(bytesPerToken)}`];
  }
  return [`${HEAD}${exactSpan(bytesPerToken)} (reported)`, `${CONTINUE}a formula-derived estimate; the layer mix is uncertain`];
}

export function checkWorkB({ shape, bytesPerToken, tokens, sequences, total }) {
  const times = `× ${int(tokens)} tokens${sequences > 1 ? ` × ${int(sequences)} conversations` : ''}`;
  const result = `${exactSpan(total)} = ${spanText(total, formatBytes)}`;
  return [...formulaLines(shape, bytesPerToken), withEquals(times, result)].join('\n');
}
