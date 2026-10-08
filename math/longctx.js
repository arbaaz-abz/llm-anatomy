// Which keys a layer reads and what it stores, from full attention to linear attention (long-context-attention §6).
// Pure: no DOM, inputs never mutated. Every bad argument throws RangeError('<fn>: <arg> must be …').

const PATTERN_KINDS = Object.freeze(['full', 'window', 'sparse', 'compressed']);
const SCALE_KINDS = Object.freeze(['full', 'window', 'sparse', 'msa', 'compressed', 'linear']);

const isCount = (v) => Number.isInteger(v) && v >= 1;

function requireCount(fn, name, value) {
  if (!isCount(value)) throw new RangeError(`${fn}: ${name} must be a positive integer`);
}

function requireKind(fn, kind, allowed) {
  if (!allowed.includes(kind)) throw new RangeError(`${fn}: kind must be one of ${allowed.join(', ')}, got ${kind}`);
}

function requireScores(n, scores) {
  const ok = Array.isArray(scores) && scores.length === n && scores.every((row) => Array.isArray(row) && row.length === n && row.every(Number.isFinite));
  if (!ok) throw new RangeError(`attentionPattern: scores must be a ${n} × ${n} matrix of finite numbers`);
}

const sum = (xs) => xs.reduce((acc, x) => acc + x, 0);
const range = (from, to) => Array.from({ length: Math.max(to - from, 0) }, (_, i) => from + i);
// Indices of the k largest values; ties go to the lower index.
const topIndices = (values, k) => values.map((v, i) => ({ v, i })).sort((a, b) => b.v - a.v || a.i - b.i).slice(0, k).map(({ i }) => i);

// One row's read set as the keys it reads (0-based) plus how many entries that is (a merged group counts 1).
function rowReads({ kind, i, window, topK, merge, scores }) {
  if (kind === 'full') return { keys: range(0, i + 1), entries: i + 1 };
  const windowStart = Math.max(0, i - window + 1);
  if (kind === 'window') return { keys: range(windowStart, i + 1), entries: i + 1 - windowStart };
  if (kind === 'sparse') {
    const keys = topIndices(scores[i].slice(0, i + 1), topK);
    return { keys, entries: keys.length };
  }
  // compressed: the window, plus the best merged groups that end before the window starts (a group scores the sum of its tokens)
  const groups = Array.from({ length: Math.floor(windowStart / merge) }, (_, b) => sum(scores[i].slice(b * merge, (b + 1) * merge)));
  const picked = topIndices(groups, topK);
  return { keys: [...picked.flatMap((b) => range(b * merge, (b + 1) * merge)), ...range(windowStart, i + 1)], entries: picked.length + i + 1 - windowStart };
}

function checkPattern({ n, kind, window, topK, merge, scores }) {
  const fn = 'attentionPattern';
  requireCount(fn, 'n', n);
  requireKind(fn, kind, PATTERN_KINDS);
  if (kind === 'window' || kind === 'compressed') requireCount(fn, 'window', window);
  if (kind === 'sparse' || kind === 'compressed') {
    requireCount(fn, 'topK', topK);
    requireScores(n, scores);
  }
  if (kind === 'compressed') requireCount(fn, 'merge', merge);
}

// Entries a layer keeps in an n-token toy: every token, the window, or the merged groups plus the window.
function storedEntries({ kind, n, window, merge }) {
  if (kind === 'full' || kind === 'sparse') return n;
  if (kind === 'window') return Math.min(n, window);
  return Math.ceil(n / merge) + Math.min(n, window);
}

// Which (query, key) cells a layer reads in an n-token toy, per-row reads, and what it stores.
// kind: 'full' | 'window' | 'sparse' | 'compressed'. sparse: the top-k keys at or before the query by `scores` (ties → lower index).
// compressed: the window, plus the top-k merged groups that end before the window (a group's score = the sum of its tokens' scores).
export function attentionPattern({ n, kind, window, topK, merge, scores }) {
  checkPattern({ n, kind, window, topK, merge, scores });
  const rows = Array.from({ length: n }, (_, i) => rowReads({ kind, i, window, topK, merge, scores }));
  const mask = rows.map(({ keys }) => {
    const read = new Set(keys);
    return Array.from({ length: n }, (_, j) => read.has(j));
  });
  return {
    mask,
    readsPerRow: rows.map(({ entries }) => entries),
    cellsRead: sum(mask.map((row) => row.filter(Boolean).length)),
    stored: storedEntries({ kind, n, window, merge }),
  };
}

function checkScale({ n, kind, window, topK, merge, topBlocks, blockSize }) {
  const fn = 'readsAndStores';
  requireCount(fn, 'n', n);
  requireKind(fn, kind, SCALE_KINDS);
  if (kind === 'window' || kind === 'compressed') requireCount(fn, 'window', window);
  if (kind === 'sparse') requireCount(fn, 'topK', topK);
  if (kind === 'msa') {
    requireCount(fn, 'topBlocks', topBlocks);
    requireCount(fn, 'blockSize', blockSize);
  }
  if (kind === 'compressed') {
    requireCount(fn, 'merge', merge);
    if (!isCount(topK) && topK !== Infinity) throw new RangeError(`${fn}: topK must be a positive integer or Infinity`);
  }
}

// Closed forms at real scale, per token per layer. 'msa': topBlocks · blockSize read (the always-kept recent block is counted
// inside topBlocks) and one score per block indexed. 'compressed': min(n / merge, topK) merged entries plus the window
// (topK = Infinity is dense attention over the merged entries). 'linear': a fixed state, nothing grows.
export function readsAndStores({ kind, n, window, topK, merge, topBlocks, blockSize }) {
  checkScale({ n, kind, window, topK, merge, topBlocks, blockSize });
  switch (kind) {
    case 'full': return { reads: n, stored: n };
    case 'window': return { reads: Math.min(n, window), stored: Math.min(n, window) };
    case 'sparse': return { reads: Math.min(n, topK), stored: n, indexed: n };
    case 'msa': return { reads: Math.min(n, topBlocks * blockSize), stored: n, indexed: Math.ceil(n / blockSize) };
    case 'compressed': {
      const entries = Math.ceil(n / merge);
      return { reads: Math.min(entries, topK) + window, stored: entries + window };
    }
    default: return { fixed: true };
  }
}

function checkLinear({ keys, values, gate, dKey, dValue }) {
  const fn = 'linearState';
  if (!Array.isArray(keys) || !Array.isArray(values) || keys.length !== values.length) throw new RangeError(`${fn}: keys and values must pair up, one value per key`);
  if (!(typeof gate === 'number' && gate >= 0 && gate <= 1)) throw new RangeError(`${fn}: gate must be a number in [0, 1], got ${gate}`);
  if (keys.some((k) => k.length !== keys[0].length)) throw new RangeError(`${fn}: every key must have the same length`);
  if (values.some((v) => v.length !== values[0].length)) throw new RangeError(`${fn}: every value must have the same length`);
  if (keys.length === 0) {
    requireCount(fn, 'dKey', dKey);
    requireCount(fn, 'dValue', dValue);
  }
}

// Linear attention with a scalar gate: S ← gate · S + v kᵀ for each token, in order. Returns [d_v × d_k].
// With no tokens the state is zeros, so the sizes come from `dKey` and `dValue`.
export function linearState({ keys, values, gate = 1, dKey, dValue }) {
  checkLinear({ keys, values, gate, dKey, dValue });
  const rows = values[0]?.length ?? dValue;
  const cols = keys[0]?.length ?? dKey;
  const zero = Array.from({ length: rows }, () => Array(cols).fill(0));
  return keys.reduce((state, k, t) => state.map((row, i) => row.map((x, j) => gate * x + values[t][i] * k[j])), zero);
}

// o = S q: the state read with one multiplication.
export function linearRead(S, q) {
  const cols = S[0]?.length ?? 0;
  if (q.length !== cols) throw new RangeError(`linearRead: q has ${q.length} entries but the state has ${cols} columns`);
  return S.map((row) => sum(row.map((x, j) => x * q[j])));
}
