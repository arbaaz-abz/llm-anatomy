// Learning-rate schedules and the attention cost of longer context (midtraining §11). Pure: no DOM, inputs never mutated.
// Every bad argument throws RangeError('<fn>: <arg> must be …').

const KINDS = ['cosine', 'wsd'];
const SHAPES = ['linear', 'minus-sqrt', 'cosine'];
const isNum = Number.isFinite;

// Fraction of the way down the decay at progress p in [0, 1]: 1 at the start, 0 at the end.
const DECAY = Object.freeze({
  linear: (p) => 1 - p,
  'minus-sqrt': (p) => 1 - Math.sqrt(p),
  cosine: (p) => 0.5 * (1 + Math.cos(Math.PI * p)),
});

function checkOptions(opts) {
  if (opts === null || typeof opts !== 'object') throw new RangeError('lrAt: options must be an object { kind, total, … }');
  const { kind, peak = 1, floor = 0, warmup = 0, total, decayStart, decayShape = 'linear' } = opts;
  if (!KINDS.includes(kind)) throw new RangeError(`lrAt: kind must be one of ${KINDS.join(', ')}, got ${kind}`);
  if (!(isNum(total) && total > 0)) throw new RangeError(`lrAt: total must be a finite number > 0, got ${total}`);
  if (!(isNum(peak) && peak > 0)) throw new RangeError(`lrAt: peak must be a finite number > 0, got ${peak}`);
  if (!(isNum(floor) && floor >= 0 && floor <= peak)) throw new RangeError(`lrAt: floor must be a finite number from 0 to peak, got ${floor}`);
  if (!(isNum(warmup) && warmup >= 0 && warmup < total)) throw new RangeError(`lrAt: warmup must be a finite number from 0 up to total (exclusive), got ${warmup}`);
  if (!SHAPES.includes(decayShape)) throw new RangeError(`lrAt: decayShape must be one of ${SHAPES.join(', ')}, got ${decayShape}`);
  if (kind === 'wsd' && !(isNum(decayStart) && decayStart >= warmup && decayStart < total)) {
    throw new RangeError(`lrAt: decayStart must be a finite number from warmup (${warmup}) up to total (${total}), exclusive, got ${decayStart}`);
  }
  return { kind, peak, floor, warmup, total, decayStart, decayShape };
}

// Learning rate at position t (tokens or a 0–1 fraction; any one unit). Warmup rises linearly from 0 to peak; then
//   cosine: peak → floor along half a cosine over [warmup, total];
//   wsd:    peak until decayStart, then peak → floor along decayShape ('linear' | 'minus-sqrt' | 'cosine').
//   (0.6, { kind: 'cosine', total: 1 })                 → 0.3455
//   (0.9, { kind: 'wsd', total: 1, decayStart: 0.8 })   → 0.5
export function lrAt(t, opts) {
  const o = checkOptions(opts);
  if (!(isNum(t) && t >= 0 && t <= o.total)) throw new RangeError(`lrAt: t must be a finite number from 0 to total (${o.total}), got ${t}`);
  if (t < o.warmup) return (o.peak * t) / o.warmup;
  const span = o.peak - o.floor;
  if (o.kind === 'cosine') return o.floor + span * DECAY.cosine((t - o.warmup) / (o.total - o.warmup));
  if (t <= o.decayStart) return o.peak;
  return o.floor + span * DECAY[o.decayShape]((t - o.decayStart) / (o.total - o.decayStart));
}

// [t, lrAt(t)] at `samples` + 1 evenly spaced positions from `from` to `to` (default 0 to total), for a plot.
export function lrCurve(opts, samples, { from = 0, to = opts?.total } = {}) {
  if (!(Number.isInteger(samples) && samples >= 1)) throw new RangeError(`lrCurve: samples must be an integer ≥ 1, got ${samples}`);
  if (!(isNum(from) && isNum(to) && from < to)) throw new RangeError(`lrCurve: from and to must be finite numbers with from < to, got ${from} → ${to}`);
  return Array.from({ length: samples + 1 }, (_, i) => {
    const t = i === samples ? to : from + ((to - from) * i) / samples;
    return [t, lrAt(t, opts)];
  });
}

// Attention work per new token grows with the context it looks back over: ctx / baseCtx (attention only; the rest of the
// forward pass costs the same per token).
//   (200, 4) → 50 · (32, 4) → 8
export function attentionCostRatio(ctx, baseCtx) {
  if (!(isNum(ctx) && ctx > 0)) throw new RangeError(`attentionCostRatio: ctx must be a finite number > 0, got ${ctx}`);
  if (!(isNum(baseCtx) && baseCtx > 0)) throw new RangeError(`attentionCostRatio: baseCtx must be a finite number > 0, got ${baseCtx}`);
  return ctx / baseCtx;
}
