// The toy's view model (pure, no DOM): (state, data) → every string and matrix the toy prints, so the page test pins them.
import { attentionPattern, linearState, linearRead } from '@math/longctx.js';
import { TOY } from '@math/attention.js';
import { formatBytes } from '@math/core.js';
import { fillText } from '@shared/claims.js';
import { TOKENS, WINDOW, INDEXER, ONE_M, CONTEXTS, SINK_LOGITS, GATES, STATE_SIZE } from './numbers.js';
import { int, windowScoresFor, sinkSplit, sumOf, trimNumber } from './format.js';
import { layerColumns, gptOssCache, qwenCache, minimaxCache, v4Estimate, fact, PRESET_MODEL } from './real-scale.js';

// The toy opens on full attention at the real-scale default of 1M tokens, following the newest token.
export const INITIAL_STATE = Object.freeze({ pattern: 'full', window: WINDOW, topK: 4, merge: 4, query: TOKENS, real: 'full', context: ONE_M, sinkLogit: 1, gate: 1 });

export const PATTERN_OPTIONS = Object.freeze([
  { value: 'full', label: 'full' }, { value: 'window', label: 'window' }, { value: 'sink', label: 'window + sink' },
  { value: 'sparse', label: 'sparse top-k' }, { value: 'compressed', label: 'compressed' }, { value: 'linear', label: 'linear' },
]);

// Chip labels carry dated numbers as placeholders; the toy fills them from the data.
export const REAL_OPTIONS = Object.freeze([
  { value: 'full', label: 'full attention' },
  { value: 'gpt-oss', label: 'gpt-oss (window {gpt-oss-120b.window})' },
  { value: 'glm-5.3', label: 'GLM-5.3 (DSA {glm-5.3.sparse_top_k})' },
  { value: 'minimax-m3', label: 'MiniMax-M3 (MSA)' },
  { value: 'deepseek-v4-pro', label: 'DeepSeek-V4-Pro (CSA / HCA)' },
  { value: 'qwen3.5', label: 'Qwen3.5 (hybrid)' },
]);

export const realOptions = (data) => REAL_OPTIONS.map((o) => ({ value: o.value, label: fillText(o.label, data) }));

const USES = Object.freeze({
  window: ['window'], sink: ['sinkLogit'], sparse: ['topK'], compressed: ['window', 'topK', 'merge'], linear: ['gate'], full: [],
});

// Which sliders show for a pattern: the others are hidden, not disabled. `query` shows for every pattern but linear.
export function visibleControls({ pattern }) {
  const used = USES[pattern] ?? [];
  return { window: used.includes('window'), topK: used.includes('topK'), merge: used.includes('merge'), sinkLogit: used.includes('sinkLogit'), gate: used.includes('gate'), query: pattern !== 'linear' };
}

const PATTERN_ARGS = Object.freeze({
  full: () => ({ kind: 'full' }),
  window: (s) => ({ kind: 'window', window: s.window }),
  sink: () => ({ kind: 'window', window: WINDOW }),
  sparse: (s) => ({ kind: 'sparse', topK: s.topK, scores: INDEXER }),
  compressed: (s) => ({ kind: 'compressed', window: s.window, topK: s.topK, merge: s.merge, scores: INDEXER }),
});

// The 16-token pattern, the followed token's counters and the cache-stack tiles. null for the linear pattern (no pattern).
export function patternView(state) {
  const spec = PATTERN_ARGS[state.pattern];
  if (!spec) return null;
  const pattern = attentionPattern({ n: TOKENS, ...spec(state) });
  const row = state.query - 1;
  const compressed = state.pattern === 'compressed';
  const read = pattern.mask[row];
  const veils = Array.from({ length: TOKENS }, (_, i) => {
    if (state.pattern === 'window' || state.pattern === 'sink') return i >= pattern.stored ? 1 : 0;
    return state.pattern === 'sparse' && !read[i] ? 0.65 : 0;
  });
  return {
    mask: pattern.mask,
    row,
    reads: String(pattern.readsPerRow[row]),
    readsSub: compressed ? `1 entry = ${state.merge} tokens` : '',
    cells: String(pattern.cellsRead),
    cellsSub: compressed ? 'token cells covered' : '',
    stored: String(pattern.stored),
    storedSub: state.pattern === 'sparse' ? 'every token stays: the indexer needs them' : '',
    storedTiles: pattern.stored,
    veils,
    merged: compressed,
  };
}

// The sink split of the followed token's window (sink first): sink + window sum to 1.
export function sinkView(state) {
  const scores = windowScoresFor(state.query, WINDOW);
  const { sink, window } = sinkSplit(scores, state.sinkLogit);
  const first = Math.max(1, state.query - WINDOW + 1);
  const values = [sink, ...window];
  return { values, sum: sumOf(values), label: `sink, then tokens ${first}–${state.query}`, sumText: `Σ = ${sumOf(values).toFixed(3)}` };
}

const HEAD = TOY.heads.A;
const FED = 3; // The, cat, sat feed the state; the query is "sat"

// The linear state after The, cat, sat and the read S · q_sat.
export function linearView(state) {
  const S = linearState({ keys: HEAD.K.slice(0, FED), values: HEAD.V.slice(0, FED), gate: state.gate, ...STATE_SIZE });
  const q = HEAD.Q[2];
  return { S, q, output: linearRead(S, q), stateNumbers: STATE_SIZE.dKey * STATE_SIZE.dValue };
}

const contextNote = (preset, context, data) => {
  const limit = PRESET_MODEL[preset] ? fact(data, PRESET_MODEL[preset], 'context_length') : null;
  return limit != null && context > limit ? `Beyond this model's ${int(limit)}-token context: a what-if, not a published setting.` : '';
};

function cacheView(preset, data, context) {
  if (preset === 'gpt-oss') {
    const g = gptOssCache(data, context);
    return g && { value: formatBytes(g.total), sub: `${g.fullLayers} full + ${g.windowLayers} window layers, ${int(g.perToken)} B per token plus ${formatBytes(g.fixed)} fixed` };
  }
  if (preset === 'qwen3.5') {
    const q = qwenCache(data, context);
    return q && { value: formatBytes(q.growing), sub: `${int(q.perToken)} B per token from ${q.fullLayers} of ${q.layers} layers, plus a fixed state per linear layer` };
  }
  if (preset === 'minimax-m3') {
    const m = minimaxCache(data, context);
    return m && { value: formatBytes(m.total), sub: `${int(m.perToken)} B per token (derived, reported): sparse reads do not shrink it` };
  }
  if (preset === 'deepseek-v4-pro') {
    const v = v4Estimate(data, context);
    return v && { value: `${formatBytes(v.low)} to ${formatBytes(v.high)}`, sub: `estimate: ${v.entry} numbers per entry at 1 or 2 bytes, CSA : HCA layers 1 : 1 or 3 : 1` };
  }
  return { value: '—', sub: preset === 'glm-5.3' ? 'its cache size is not in the data; sparse reads do not shrink it' : 'pick a model to see its whole-model cache' };
}

const MISSING_ROWS = Object.freeze({ head: ['Per attention layer', '—'], rows: [], cache: { value: '—', sub: '' }, note: 'This model\'s numbers are not in the data.' });

// The "at real scale" panel: a per-layer table, the whole-model cache and a note.
export function realView(state, data) {
  const layers = layerColumns(state.real, data, state.context);
  const cache = layers && cacheView(state.real, data, state.context);
  if (!layers || !cache) return MISSING_ROWS;
  const cell = (column, key, name) => ({ value: column[key], sub: column[`${key}Sub`], name: name ?? undefined });
  const names = (key) => layers.columns.map((c, i) => (i === 0 ? `real-${key}` : `real-${key}-${['hca', 'linear'][i - 1]}`));
  const rows = [
    { label: 'Reads per token', sub: 'keys one query reads', cells: layers.columns.map((c, i) => cell(c, 'reads', names('reads')[i])) },
    { label: 'Stored entries', sub: 'cache entries in the layer', cells: layers.columns.map((c, i) => cell(c, 'stored', names('stored')[i])) },
    ...(layers.indexed ? [{ label: 'Indexer scores per token', sub: 'cheap, but one per stored key or block', cells: [{ value: layers.indexed, name: 'real-indexed' }] }] : []),
  ];
  return { head: ['Per attention layer', ...layers.columns.map((c) => c.label)], rows, cache, note: contextNote(state.real, state.context, data) };
}

export const contextLabel = (v) => int(v);
export const sinkLogitLabel = (v) => int(v);
export const gateLabel = (v) => String(v);
export const toyStops = Object.freeze({ contexts: CONTEXTS, sinkLogits: SINK_LOGITS, gates: GATES });

// Everything the toy prints for one state.
export const toyView = (state, data) => ({
  controls: visibleControls(state),
  pattern: patternView(state),
  sink: state.pattern === 'sink' ? sinkView(state) : null,
  linear: state.pattern === 'linear' ? { ...linearView(state), note: linearNote(state) } : null,
  real: realView(state, data),
});

// The three "Try this" prompts of storyboard §6 with their insights; every number comes from the same functions the toy calls.
export function tryThis(data) {
  const sparse = patternView({ ...INITIAL_STATE, pattern: 'sparse', topK: 4 });
  const merged = patternView({ ...INITIAL_STATE, pattern: 'compressed', topK: 1, merge: 4 });
  const col = (preset) => layerColumns(preset, data, ONE_M)?.columns[0];
  const [dsa, csa] = [col('glm-5.3'), col('deepseek-v4-pro')];
  const sinkOn = sinkSplit(windowScoresFor(TOKENS, WINDOW), 1);
  const sinkOff = sinkSplit(windowScoresFor(TOKENS, WINDOW), -2);
  const three = (xs) => `[${xs.map((x) => x.toFixed(3)).join(', ')}]`;
  const gate1 = linearView({ ...INITIAL_STATE, gate: 1 }).output;
  const gateHalf = linearView({ ...INITIAL_STATE, gate: 0.5 }).output;
  const list = (xs) => `[${xs.map((x) => trimNumber(x)).join(', ')}]`;
  return [
    {
      prompt: `Pattern sparse top-k, k = 4: token 16 reads ${sparse.reads} entries, stores ${sparse.stored}. Switch to compressed (merge 4, top 1): reads ${merged.reads}, stores ${merged.stored}. At real scale, tap GLM-5.3 then DeepSeek-V4-Pro: DSA reads ${dsa?.reads ?? '—'} and stores ${dsa?.stored ?? '—'}; CSA reads ${csa?.reads ?? '—'} and stores ${csa?.stored ?? '—'}.`,
      insight: 'picking what to read saves compute; only merging, windows or a fixed state save memory.',
    },
    {
      prompt: `Pattern window + sink, sink logit 1: the sink takes ${sinkOn.sink.toFixed(3)} and the four tokens share ${(1 - sinkOn.sink).toFixed(3)}. Set the sink logit to −2: ${three(sinkOff.window)} + sink ${sinkOff.sink.toFixed(3)}, almost the no-sink row.`,
      insight: 'the sink is a learned "nothing here" option; when a window holds nothing useful, the layer can attend to nothing instead of to noise.',
    },
    {
      prompt: `Pattern linear, gate 1: the output for "sat" is ${list(gate1)}, weighted by the raw scores −1, 3, 0.5 with no softmax. Set the gate to 0.5: the state halves before each new token, output ${list(gateHalf)}. The state stays ${STATE_SIZE.dKey * STATE_SIZE.dValue} numbers however long the text.`,
      insight: 'a linear layer trades exact lookup for a fixed-size memory that blends and fades; that is why hybrids keep some full layers.',
    },
  ];
}

// The linear note follows the gate (README lesson 5: text matches the state).
export function linearNote(state) {
  const base = `The state has ${STATE_SIZE.dKey * STATE_SIZE.dValue} numbers however long the text; `;
  return state.gate === 1 ? `${base}gate 1 keeps every addition.` : `${base}gate ${state.gate} halves the state before each new token.`;
}
