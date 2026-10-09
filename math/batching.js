// Batching schedulers on the toy requests (batching §6; serving-overview frame 9 and disaggregation frame 1 import it, P4-R3).
// Pure and deterministic: the same input gives a deep-equal output, so a scrubbed frame equals the same frame reached
// forward (Review Focus 4). Inputs are never mutated. Steps count from 0; this module has no memory addresses.
// FROZEN after Plan 4's shared prep (S6): change only through a shared patch.
//
// Step rule (the same as paged-attention's): the step a request is admitted runs its prefill; each later step is one decode
// step that adds one token; a request with `output` o admitted at step a is done at the end of step a + o, and its seat
// is free from the next step. A request is { id, arrives, prompt, output } (TOY_REQUESTS in math/serving.js).

const isInt = (x) => Number.isInteger(x);

function requireSeats(fn, seats) {
  if (!isInt(seats) || seats < 1) throw new RangeError(`${fn}: seats must be a positive integer, got ${seats}`);
}

function requireRequests(fn, requests) {
  if (!Array.isArray(requests) || requests.length === 0) throw new RangeError(`${fn}: requests must be a non-empty array`);
  const seen = new Set();
  for (const r of requests) {
    if (!r || typeof r.id !== 'string' || r.id === '') throw new RangeError(`${fn}: every request needs a string id`);
    if (seen.has(r.id)) throw new RangeError(`${fn}: request ids must be unique, got ${r.id} twice`);
    seen.add(r.id);
    if (!isInt(r.arrives) || r.arrives < 0) throw new RangeError(`${fn}: request ${r.id} arrives must be an integer ≥ 0, got ${r.arrives}`);
    if (!isInt(r.prompt) || r.prompt < 1) throw new RangeError(`${fn}: request ${r.id} prompt must be a positive integer, got ${r.prompt}`);
    if (!isInt(r.output) || r.output < 1) throw new RangeError(`${fn}: request ${r.id} output must be a positive integer, got ${r.output}`);
  }
}

// Arrival order, ties in the order given (a stable sort on a copy).
const inArrivalOrder = (requests) => requests.map((r, index) => ({ ...r, index })).sort((a, b) => a.arrives - b.arrives || a.index - b.index);

function summarize(kind, seats, requests, admittedAt) {
  const live = requests.map((r) => ({
    id: r.id, arrives: r.arrives, admitted: admittedAt.get(r.id), finishes: admittedAt.get(r.id) + r.output, waited: admittedAt.get(r.id) - r.arrives,
  }));
  const lastStep = Math.max(...live.map((r) => r.finishes));
  const steps = lastStep + 1;
  const seatSteps = seats * steps;
  // Busy seat-steps count every step from a request's admission to its last step inclusive (prefill and decode).
  const busySeatSteps = live.reduce((sum, r) => sum + (r.finishes - r.admitted + 1), 0);
  const outputTokens = requests.reduce((sum, r) => sum + r.output, 0);
  return {
    kind, seats, live, lastStep, steps, seatSteps, busySeatSteps,
    utilizationPct: (100 * busySeatSteps) / seatSteps, outputTokens, tokensPerStep: outputTokens / steps,
  };
}

// A batch starts only when the previous batch is entirely done; it takes up to `seats` arrived requests in arrival order.
export function simulateStatic({ requests, seats }) {
  requireRequests('simulateStatic', requests);
  requireSeats('simulateStatic', seats);
  const waiting = inArrivalOrder(requests);
  const admittedAt = new Map();
  let free = 0; // the first step the next batch may start
  while (waiting.length > 0) {
    const start = Math.max(free, waiting[0].arrives);
    const batch = waiting.filter((r) => r.arrives <= start).slice(0, seats);
    for (const r of batch) {
      admittedAt.set(r.id, start);
      waiting.splice(waiting.indexOf(r), 1);
    }
    free = Math.max(...batch.map((r) => start + r.output)) + 1;
  }
  return summarize('static', seats, requests, admittedAt);
}

// Before every step: drop requests done at the end of the previous step, then admit arrived requests in arrival order
// while seats are free.
export function simulateContinuous({ requests, seats }) {
  requireRequests('simulateContinuous', requests);
  requireSeats('simulateContinuous', seats);
  const waiting = inArrivalOrder(requests);
  const admittedAt = new Map();
  let running = []; // { id, finishes }
  for (let step = 0; waiting.length > 0; step += 1) {
    running = running.filter((r) => r.finishes >= step);
    while (running.length < seats && waiting.length > 0 && waiting[0].arrives <= step) {
      const r = waiting.shift();
      admittedAt.set(r.id, step);
      running.push({ id: r.id, finishes: step + r.output });
    }
  }
  return summarize('continuous', seats, requests, admittedAt);
}

// Continuous batching with a per-step token budget (chunked prefill). Each step: running decodes first (one token each),
// then waiting prompt tokens up to budget − decodes, requests already mid-prefill first, then new arrivals in arrival order
// while seats are free; a request decodes from the step after its last prefill slice. budget = Infinity prefills whole
// prompts, which reproduces simulateContinuous. Steps with no work are skipped.
export function scheduleTokens({ requests, seats, budget = Infinity }) {
  requireRequests('scheduleTokens', requests);
  requireSeats('scheduleTokens', seats);
  if (budget !== Infinity && (!isInt(budget) || budget < 1)) throw new RangeError(`scheduleTokens: budget must be a positive integer or Infinity, got ${budget}`);
  const waiting = inArrivalOrder(requests);
  let holding = []; // admitted, not finished, in admission order: { id, promptLeft, decodesLeft }
  const schedule = [];
  for (let step = 0; waiting.length > 0 || holding.length > 0; step += 1) {
    const decoders = holding.filter((r) => r.promptLeft === 0);
    let room = budget - decoders.length;
    const prefill = [];
    const slice = (r) => {
      const tokens = Math.max(0, Math.min(r.promptLeft, room));
      if (tokens > 0) prefill.push({ id: r.id, tokens });
      room -= tokens;
      return tokens;
    };
    const slices = new Map(holding.filter((r) => r.promptLeft > 0).map((r) => [r.id, slice(r)]));
    while (holding.length < seats && waiting.length > 0 && waiting[0].arrives <= step && room > 0) {
      const next = waiting.shift();
      const entry = { id: next.id, promptLeft: next.prompt, decodesLeft: next.output };
      holding.push(entry);
      slices.set(entry.id, slice(entry));
    }
    if (decoders.length > 0 || prefill.length > 0) {
      schedule.push({ step, decode: decoders.map((r) => r.id), prefill, tokens: decoders.length + prefill.reduce((s, p) => s + p.tokens, 0) });
    }
    holding = holding
      .map((r) => (r.promptLeft > 0 ? { ...r, promptLeft: r.promptLeft - slices.get(r.id) } : { ...r, decodesLeft: r.decodesLeft - 1 }))
      .filter((r) => r.promptLeft > 0 || r.decodesLeft > 0);
  }
  return schedule;
}

// Padding tokens wasted when a new prompt of n tokens joins a padded batch of B running requests: (n − 1)(B − 1)
// (Hugging Face's padded-batching example). `runningRequests` is the batch size or the array of running requests.
export function paddingWaste(newPromptTokens, runningRequests) {
  const running = Array.isArray(runningRequests) ? runningRequests.length : runningRequests;
  if (!isInt(newPromptTokens) || newPromptTokens < 1) throw new RangeError(`paddingWaste: newPromptTokens must be a positive integer, got ${newPromptTokens}`);
  if (!isInt(running) || running < 1) throw new RangeError(`paddingWaste: runningRequests must be a positive integer or a non-empty array, got ${runningRequests}`);
  return (newPromptTokens - 1) * (running - 1);
}
