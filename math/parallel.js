// Splitting training across GPUs: collective costs, GPU counts, pipeline bubbles and schedules (parallelism §6).
// Bytes are bytes sent per GPU. Pure: no DOM, inputs never mutated. Owned by parallelism; cluster-topology imports
// the collective costs. FROZEN after Plan 3's shared prep (S3): change only through a shared patch.

const SCHEDULES = ['gpipe', '1f1b'];

function requireCount(fn, name, value) {
  if (!Number.isInteger(value) || value < 1) throw new RangeError(`${fn}: ${name} must be a positive integer, got ${value}`);
}

function requireSize(fn, sizeBytes) {
  if (typeof sizeBytes !== 'number' || !Number.isFinite(sizeBytes) || sizeBytes < 0) {
    throw new RangeError(`${fn}: sizeBytes must be a finite number ≥ 0, got ${sizeBytes}`);
  }
}

// (N − 1)/N · S: each GPU sends all but its own 1/N share.
function shareSent(fn, sizeBytes, ranks) {
  requireSize(fn, sizeBytes);
  requireCount(fn, 'ranks', ranks);
  return ((ranks - 1) / ranks) * sizeBytes;
}

// 2(N − 1)/N · S: a reduce-scatter followed by an all-gather.
export function ringAllReduceBytes(sizeBytes, ranks) {
  return 2 * shareSent('ringAllReduceBytes', sizeBytes, ranks);
}

export function reduceScatterBytes(sizeBytes, ranks) {
  return shareSent('reduceScatterBytes', sizeBytes, ranks);
}

export function allGatherBytes(sizeBytes, ranks) {
  return shareSent('allGatherBytes', sizeBytes, ranks);
}

// Uniform traffic: each GPU keeps 1/N of what it holds and sends the rest.
export function allToAllBytes(sizeBytes, ranks) {
  return shareSent('allToAllBytes', sizeBytes, ranks);
}

export function gpuCount({ tp = 1, cp = 1, pp = 1, dp = 1 }) {
  requireCount('gpuCount', 'tp', tp);
  requireCount('gpuCount', 'cp', cp);
  requireCount('gpuCount', 'pp', pp);
  requireCount('gpuCount', 'dp', dp);
  return tp * cp * pp * dp;
}

// Idle share of a pipeline: b / (m + b) with b = (p − 1)/v. For v = 1 that is (p − 1)/(m + p − 1).
export function pipelineBubble({ stages, microBatches, virtualStages = 1 }) {
  requireCount('pipelineBubble', 'stages', stages);
  requireCount('pipelineBubble', 'microBatches', microBatches);
  requireCount('pipelineBubble', 'virtualStages', virtualStages);
  const b = (stages - 1) / virtualStages;
  return b / (microBatches + b);
}

const range = (from, to) => Array.from({ length: Math.max(0, to - from + 1) }, (_, i) => from + i);

// The order one stage (0-based s) runs its ops in. GPipe: every forward, then every backward.
// 1F1B: warm-up min(p − s − 1, m) forwards, then forward, backward, … until the forwards run out, then the rest.
function stageOrder(schedule, s, stages, microBatches) {
  const fwd = (j) => `F${j}`;
  const bwd = (j) => `B${j}`;
  if (schedule === 'gpipe') return [...range(1, microBatches).map(fwd), ...range(1, microBatches).map(bwd)];
  const warm = Math.min(stages - s - 1, microBatches);
  const steady = range(warm + 1, microBatches).flatMap((j, i) => [fwd(j), bwd(i + 1)]);
  const drain = range(microBatches - warm + 1, microBatches).map(bwd);
  return [...range(1, warm).map(fwd), ...steady, ...drain];
}

// What op `op` on stage s waits for: F needs F of stage s − 1; B needs B of stage s + 1, or F of the last stage.
function dependency(op, s, stages) {
  const j = op.slice(1);
  if (op[0] === 'F') return s === 0 ? null : [s - 1, op];
  return s === stages - 1 ? [s, `F${j}`] : [s + 1, `B${j}`];
}

// Unit-time ops; at each tick every stage runs its next op if that op's input finished on an earlier tick.
function simulate(orders, stages) {
  const finished = orders.map(() => new Map());
  const next = orders.map(() => 0);
  const total = orders.reduce((n, order) => n + order.length, 0);
  let placed = 0;
  for (let t = 0; placed < total; t += 1) {
    if (t > 2 * total) throw new RangeError('pipelineSchedule: the schedule deadlocked');
    const runnable = orders.map((order, s) => {
      const op = order[next[s]];
      const dep = op && dependency(op, s, stages);
      return op && (!dep || finished[dep[0]].get(dep[1]) < t) ? op : null;
    });
    runnable.forEach((op, s) => {
      if (!op) return;
      finished[s].set(op, t);
      next[s] += 1;
      placed += 1;
    });
  }
  return finished;
}

function peakInFlight(order) {
  return order.reduce(([held, peak], op) => {
    const now = held + (op[0] === 'F' ? 1 : -1);
    return [now, Math.max(peak, now)];
  }, [0, 0])[1];
}

// grid[s] holds 'F1' … 'Bm' or '.' (idle) per tick; columns = 2(m + p − 1); idleFraction = idle cells / all cells;
// peakInFlight[s] = most micro-batches whose activations stage s holds at once (forward done, backward not yet).
export function pipelineSchedule({ schedule, stages, microBatches }) {
  if (!SCHEDULES.includes(schedule)) throw new RangeError(`pipelineSchedule: schedule must be 'gpipe' or '1f1b', got ${schedule}`);
  requireCount('pipelineSchedule', 'stages', stages);
  requireCount('pipelineSchedule', 'microBatches', microBatches);
  const orders = range(0, stages - 1).map((s) => stageOrder(schedule, s, stages, microBatches));
  const finished = simulate(orders, stages);
  const columns = 1 + Math.max(...finished.flatMap((times) => [...times.values()]));
  const grid = finished.map((times) => {
    const opAt = new Map([...times].map(([op, t]) => [t, op]));
    return Array.from({ length: columns }, (_, t) => opAt.get(t) ?? '.');
  });
  const idle = grid.flat().filter((cell) => cell === '.').length;
  return { columns, grid, idleFraction: idle / (stages * columns), peakInFlight: orders.map(peakInFlight) };
}
