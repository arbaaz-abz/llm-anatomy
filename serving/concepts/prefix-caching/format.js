// prefix-caching pure helpers (no DOM): the toy's state and its transitions, the one price format, the simulation the toy runs and
// "Check my work". Every number comes from math/prefix.js; this file only arranges and formats them.
import { deepFreeze, formatUsd } from '@math/core.js';
import { PREFIX_REQUESTS, blendedInputPrice, poolBlocksFor, simulatePrefixCache } from '@math/prefix.js';
import { formatShare } from '@shared/glyphs.js';

export const BLOCK_SIZES = Object.freeze([1, 2, 4, 8, 16]);
export const POOLS = Object.freeze([6, 8, 12]);
// The five requests the toy can run, in arrival order. "B again" repeats B after D.
export const REQUEST_KEYS = Object.freeze([
  { key: 'A', label: 'A', source: 0 }, { key: 'B', label: 'B', source: 1 }, { key: 'C', label: 'C', source: 2 },
  { key: 'D', label: 'D', source: 3 }, { key: 'B-again', label: 'B again', source: 1 },
]);

// hitMode: 'toy' (the live hit rate) or a percentage set by hand. writePremium: charge the cache-write multiple.
export const INITIAL_STATE = deepFreeze({ on: ['A', 'B', 'C', 'D'], blockSize: 4, pool: 8, provider: 'sonnet', writePremium: true, hitMode: 'toy' });

const MAX_PERCENT = 100;

function checkOne(fn, list, value, what) {
  if (!list.includes(value)) throw new RangeError(`${fn}: ${what} must be one of ${list.join(', ')}, got ${value}`);
}

// ---- state transitions (pure: every one returns a new frozen state) ----
export function toggleRequest(state, key) {
  checkOne('toggleRequest', REQUEST_KEYS.map((r) => r.key), key, 'key');
  const on = state.on.includes(key) ? state.on.filter((k) => k !== key) : [...state.on, key];
  return Object.freeze({ ...state, on: Object.freeze(REQUEST_KEYS.map((r) => r.key).filter((k) => on.includes(k))) });
}

export function setBlockSize(state, blockSize) {
  checkOne('setBlockSize', BLOCK_SIZES, blockSize, 'blockSize');
  return Object.freeze({ ...state, blockSize });
}

export function setPool(state, pool) {
  checkOne('setPool', POOLS, pool, 'pool');
  return Object.freeze({ ...state, pool });
}

export function setProvider(state, provider) {
  checkOne('setProvider', ['sonnet', 'opus', 'deepseek'], provider, 'provider');
  return Object.freeze({ ...state, provider });
}

export const setWritePremium = (state, writePremium) => Object.freeze({ ...state, writePremium: Boolean(writePremium) });

export function setHitMode(state, hitMode) {
  const ok = hitMode === 'toy' || (typeof hitMode === 'number' && hitMode >= 0 && hitMode <= MAX_PERCENT);
  if (!ok) throw new RangeError(`setHitMode: hitMode must be 'toy' or a percentage 0-${MAX_PERCENT}, got ${hitMode}`);
  return Object.freeze({ ...state, hitMode });
}

// ---- price format (one per quantity): $ per million tokens through the course's formatUsd (three significant figures, two decimals at least) ----
export function formatPrice(usd) {
  if (usd < 0) throw new RangeError(`formatPrice: usd must be ≥ 0, got ${usd}`);
  return formatUsd(usd);
}

const plain = (x) => String(Number(x.toPrecision(4))); // 1.25, 0.1, 0.05

// ---- the simulation the toy runs ----
export function requestsFor(state) {
  return REQUEST_KEYS.filter((r) => state.on.includes(r.key)).map((r) => ({ ...PREFIX_REQUESTS[r.source], label: r.label, key: r.key }));
}

export function simulateFor(state) {
  const requests = requestsFor(state);
  if (requests.length === 0) return { requests, log: [], promptTokens: 0, hitTokens: 0, hitRatePct: 0, cachedBlocks: 0, poolBlocks: poolBlocksFor(state.pool, state.blockSize) };
  const poolBlocks = poolBlocksFor(state.pool, state.blockSize);
  return { requests, poolBlocks, ...simulatePrefixCache({ requests, blockSize: state.blockSize, poolBlocks }) };
}

// The hit rate the price uses, as a fraction: the toy's own, or the value set by hand.
export function hitFraction(state, sim) {
  if (state.hitMode !== 'toy') return state.hitMode / MAX_PERCENT;
  return sim.promptTokens === 0 ? 0 : sim.hitTokens / sim.promptTokens;
}

// The price terms of one provider in one state: the multiples actually charged and the blended price (unrounded).
export function priceFor(state, provider, hit) {
  const write = provider.hasWrite && state.writePremium ? provider.write : 1;
  const miss = (1 - hit) * provider.base * write;
  const saved = hit * provider.base * provider.read;
  return { write, miss, saved, blended: blendedInputPrice({ hitRate: hit, basePrice: provider.base, readMult: provider.read, writeMult: write }) };
}

// ---- "Check my work" (storyboard §6) ----
export function checkWork(state, sim, provider) {
  const h = hitFraction(state, sim);
  const { write, miss, saved, blended } = priceFor(state, provider, h);
  const source = state.hitMode === 'toy'
    ? (sim.promptTokens === 0 ? 'h = 0 (no request has arrived)' : `h = ${sim.hitTokens} hit tokens ÷ ${sim.promptTokens} prompt tokens = ${formatShare(h)}`)
    : `h = ${formatShare(h)} (set by hand)`;
  const keep = (1 - h).toFixed(3);
  const hit = h.toFixed(3);
  const pad = '            ';
  if (provider.id === 'deepseek') {
    return [
      source,
      'price per M = (1 − h) · miss + h · hit',
      `${pad}= ${keep} · ${formatPrice(provider.base)} + ${hit} · ${formatPrice(provider.hitUsd)} = ${formatPrice(miss)} + ${formatPrice(saved)} = ${formatPrice(blended)}`,
    ].join('\n');
  }
  return [
    source,
    'price per M = (1 − h) · base · write + h · base · read',
    `${pad}= ${keep} · ${formatPrice(provider.base)} · ${plain(write)} + ${hit} · ${formatPrice(provider.base)} · ${plain(provider.read)} = ${formatPrice(miss)} + ${formatPrice(saved)} = ${formatPrice(blended)}`,
  ].join('\n');
}
