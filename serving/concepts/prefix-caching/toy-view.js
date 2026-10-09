// prefix-caching toy view model (pure, no DOM): state + data → every string the toy prints, the tree/pool the stage draws, and the
// try-this list. Hits, evictions and prices come from math/prefix.js; the scale-up line from math/serving.js (numbers.js).
import { formatBytes, formatDuration, formatInt } from '@math/core.js';
import { formatShare } from '@shared/glyphs.js';
import { fillText } from '@shared/claims.js';
import { INITIAL_STATE, checkWork, formatPrice, hitFraction, priceFor, simulateFor } from './format.js';
import { providersFor, deepseekHitRate } from './facts.js';
import { SCALE_UP } from './numbers.js';
import { nodeState, poolAfter, requestPaths, shortLabel, treeModel, treeNodes } from './scenes.js';

export const TREE_MIN_BLOCK_SIZE = 4; // below this the tree has too many blocks to draw at 580 px (the counters stay exact)
const PER_ROW = Object.freeze({ 1: 16, 2: 12, 4: 4, 8: 4, 16: 2 });
const noneYet = '—';

// The tree the toy draws for the last request that arrived: its hits green, its new blocks in its color, evicted blocks hatched.
function treeFor(state, sim) {
  if (state.blockSize < TREE_MIN_BLOCK_SIZE || sim.log.length === 0) return null;
  const last = sim.log.length - 1;
  const model = treeModel(sim.requests, state.blockSize);
  const paths = requestPaths(sim.requests, state.blockSize, sim.log);
  const held = new Set(sim.log[last].cachedKeys);
  const evicted = new Set(model.filter((n) => !held.has(n.id)).map((n) => n.id));
  return treeNodes(model, { state: nodeState({ paths, active: last, evicted }) });
}

const MAX_LISTED = 4; // more evicted labels than this print as a count
// Evicted blocks print with the tree's short labels ("Do…fish, It…."); the full words ride in the cell's title.
const evictedText = (list) => (list.length === 0 ? 'none' : list.length <= MAX_LISTED ? list.map(shortLabel).join(', ') : `${list.length} blocks`);
const evictedTitle = (list) => (list.length === 0 || list.length > MAX_LISTED ? '' : list.join(' · '));

function rowsFor(sim) {
  return sim.log.map((r, i) => ({
    key: sim.requests[i].key,
    label: sim.requests[i].label,
    prompt: formatInt(r.promptTokens),
    hit: formatInt(r.hitTokens),
    computed: formatInt(r.computed),
    blocks: r.blocks.join(', '),
    evicted: evictedText(r.evicted),
    evictedTitle: evictedTitle(r.evicted),
  }));
}

export function toyView(state, data) {
  const sim = simulateFor(state);
  const providers = providersFor(data);
  const provider = providers?.find((p) => p.id === state.provider) ?? null;
  const h = hitFraction(state, sim);
  const last = sim.log.at(-1) ?? null;
  const price = provider ? priceFor(state, provider, h) : null;
  const toyShare = sim.promptTokens === 0 ? noneYet : formatShare(sim.hitTokens / sim.promptTokens);
  return {
    sim,
    provider,
    rows: rowsFor(sim),
    totals: { prompt: formatInt(sim.promptTokens), hit: formatInt(sim.hitTokens), computed: formatInt(sim.promptTokens - sim.hitTokens), rate: toyShare },
    scale: { skipped: formatDuration(SCALE_UP.skippedS), held: formatBytes(SCALE_UP.heldBytes), tokens: formatInt(SCALE_UP.tokens) },
    price: price && { rate: formatShare(h), blended: formatPrice(price.blended), plain: formatPrice(provider.base), write: state.writePremium && provider.hasWrite },
    hitSlider: state.hitMode === 'toy' ? toyShare : formatShare(h),
    checkWork: provider ? checkWork(state, sim, provider) : '',
    stage: {
      tree: treeFor(state, sim),
      treeNote: state.blockSize < TREE_MIN_BLOCK_SIZE ? `blocks of ${state.blockSize}: too many blocks to draw as a tree; the counts below are exact` : null,
      legend: last ? `last to arrive: ${sim.requests.at(-1).label} · green hit · its color new · grey cached · hatched evicted` : 'no request has arrived: the cache is empty',
      poolBlocks: poolAfter(last, sim.poolBlocks),
      queue: last ? last.freeQueue : Array.from({ length: sim.poolBlocks }, (_, i) => i),
      blockSize: state.blockSize,
      perRow: PER_ROW[state.blockSize],
    },
  };
}

const sim = (patch) => simulateFor({ ...INITIAL_STATE, ...patch });
const rate = (s) => formatShare(s.hitTokens / s.promptTokens);
const dollars = (state, provider, h) => formatPrice(priceFor(state, provider, h).blended);

// Storyboard §6 "Try this", every number computed from the toy's own states. `x` marks code text; [[slug]] links a lesson.
export function tryThis(data) {
  const providers = providersFor(data);
  const [sonnet, , deepseek] = providers ?? [];
  const at = (blockSize) => sim({ blockSize });
  const prod = deepseekHitRate(data);
  const hit = (h) => (sonnet ? dollars(INITIAL_STATE, sonnet, h) : noneYet);
  const dsHit = (h) => (deepseek ? dollars({ ...INITIAL_STATE, provider: 'deepseek' }, deepseek, h) : noneYet);
  const again = simulateFor({ ...INITIAL_STATE, on: [...INITIAL_STATE.on, 'B-again'] });
  const small = simulateFor({ ...INITIAL_STATE, on: [...INITIAL_STATE.on, 'B-again'], pool: 6 });
  const live = at(4).hitTokens / at(4).promptTokens;
  const real = prod === null ? null : prod / 100;
  return [
    {
      prompt: `Slide Block size from 4 to 16, then to 1 (the order matters: the effect is not monotone). From 4 to 16 the hit rate falls from ${rate(at(4))} to ${rate(at(16))}: B's 8-token system prompt no longer fills a block (${at(16).log[1].hitTokens} hits), while C still reuses ${at(16).log[2].hitTokens} tokens because A's whole first turn filled exactly one block. Now slide to 1: ${rate(at(1))}, because D reuses \`You are a\`.`,
      insight: 'block size sets the grain of reuse.',
      rest: fillText(' Smaller blocks catch more, at the cost of bigger tables and smaller memory reads (see [[paged-attention]]); vLLM uses {sv:vllm.default_block_size}, SGLang matches token by token.', data),
    },
    {
      prompt: `Keep Pool size at 8 and turn on B again after D. D evicted B's and C's private blocks, but B still reuses its ${again.log.at(-1).hitTokens}-token system prompt; this time A's turn (\`Where did you sit\`, \`The cat sat down\`) is evicted to make room. Switch Pool size to 6: C already loses B's ${small.log[2].evicted.length} blocks, and D evicts A's turn too.`,
      insight: 'least-recently-used eviction keeps whatever keeps getting reused,',
      rest: ' which is usually the shared beginning.',
    },
    {
      prompt: `Pick Anthropic Sonnet 5.5 with Charge the cache write on. Hit rate for pricing at 0%: ${hit(0)} per M, more than not caching at all (${sonnet ? formatPrice(sonnet.base) : noneYet}). This toy's ${formatShare(live)}: ${hit(live)}. DeepSeek's ${real === null ? noneYet : formatShare(real)}: ${real === null ? noneYet : hit(real)}. Switch Price to DeepSeek V4-Pro: ${dsHit(0)} at 0%, ${real === null ? noneYet : dsHit(real)} at ${real === null ? noneYet : formatShare(real)}.`,
      insight: 'caching pays only when hits come back.',
      rest: ' A write costs extra because the provider must hold your KV for minutes; a hit is cheap because it skips prefill math.',
    },
  ];
}

