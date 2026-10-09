import { test } from 'node:test';
import assert from 'node:assert/strict';
import { RUNNING_EXAMPLE, TOY_REQUESTS, stepTime } from '../math/serving.js';
import { simulateStatic, simulateContinuous, scheduleTokens, paddingWaste } from '../math/batching.js';

const R = TOY_REQUESTS;
const printed = (x, d) => Number(x.toFixed(d));
const span = (sim, id) => {
  const r = sim.live.find((x) => x.id === id);
  return `${r.admitted}–${r.finishes}`;
};
const lanes = (sim) => Object.fromEntries(sim.live.map((r) => [r.id, span(sim, r.id)]));
const withOutput = (id, output) => R.map((r) => (r.id === id ? { ...r, output } : r));
const withPrompt = (id, prompt) => R.map((r) => (r.id === id ? { ...r, prompt } : r));

test('simulateStatic, 3 seats: D waits for the whole first batch — D 7–10, last step 10, busy 19 of 33 = 57.6%, 1.36 tokens/step', () => {
  const s = simulateStatic({ requests: R, seats: 3 });
  assert.deepEqual(lanes(s), { A: '0–4', B: '0–2', C: '0–6', D: '7–10' });
  assert.deepEqual([s.kind, s.seats, s.lastStep, s.steps, s.seatSteps, s.busySeatSteps], ['static', 3, 10, 11, 33, 19]);
  assert.deepEqual([printed(s.utilizationPct, 1), s.outputTokens, printed(s.tokensPerStep, 2)], [57.6, 15, 1.36]);
  assert.deepEqual(s.live.map((r) => [r.id, r.arrives, r.waited]), [['A', 0, 0], ['B', 0, 0], ['C', 0, 0], ['D', 1, 6]]);
});

test('simulateContinuous, 3 seats: D takes the seat B frees — D 3–6, last step 6, busy 19 of 21 = 90.5%, 2.14 tokens/step', () => {
  const s = simulateContinuous({ requests: R, seats: 3 });
  assert.deepEqual(lanes(s), { A: '0–4', B: '0–2', C: '0–6', D: '3–6' });
  assert.deepEqual([s.kind, s.lastStep, s.steps, s.seatSteps, s.busySeatSteps], ['continuous', 6, 7, 21, 19]);
  assert.deepEqual([printed(s.utilizationPct, 1), printed(s.tokensPerStep, 2)], [90.5, 2.14]);
  assert.equal(s.live[3].waited, 2);
});

test('4 seats: continuous D 1–4, last step 6 (67.9%)', () => {
  const co = simulateContinuous({ requests: R, seats: 4 });
  assert.deepEqual([span(co, 'D'), co.lastStep, printed(co.utilizationPct, 1)], ['1–4', 6, 67.9]);
});

test('2 seats: static C 5–11, D 5–8 (79.2%); continuous C 3–9, D 5–8, last step 9 (95.0%)', () => {
  const st = simulateStatic({ requests: R, seats: 2 });
  assert.deepEqual(lanes(st), { A: '0–4', B: '0–2', C: '5–11', D: '5–8' });
  assert.equal(printed(st.utilizationPct, 1), 79.2);
  const co = simulateContinuous({ requests: R, seats: 2 });
  assert.deepEqual(lanes(co), { A: '0–4', B: '0–2', C: '3–9', D: '5–8' });
  assert.deepEqual([co.lastStep, printed(co.utilizationPct, 1)], [9, 95]);
});

test('C\'s answer grows to 10 tokens, 3 seats: static D 11–14, last step 14 (51.1%); continuous D 3–6, last step 10 (69.7%)', () => {
  const requests = withOutput('C', 10);
  const st = simulateStatic({ requests, seats: 3 });
  assert.deepEqual([span(st, 'D'), st.lastStep, printed(st.utilizationPct, 1)], ['11–14', 14, 51.1]);
  const co = simulateContinuous({ requests, seats: 3 });
  assert.deepEqual([span(co, 'D'), co.lastStep, printed(co.utilizationPct, 1)], ['3–6', 10, 69.7]);
});

test('4 seats, static: D still waits until the batch is done (step 7, 43.2%)', () => {
  const st = simulateStatic({ requests: R, seats: 4 });
  // all four requests share one batch only if they have all arrived when it starts: D arrives at step 1, after the batch began at 0
  assert.deepEqual(lanes(st), { A: '0–4', B: '0–2', C: '0–6', D: '7–10' });
  assert.equal(printed(st.utilizationPct, 1), 43.2);
});

test('simulateContinuous with 4 or more seats admits every request on arrival', () => {
  for (const seats of [4, 5, 9]) {
    const s = simulateContinuous({ requests: R, seats });
    assert.ok(s.live.every((r) => r.admitted === r.arrives), `${seats} seats`);
  }
});

test('properties for every seat count: utilization ≤ 100%, continuous never ends later than static, busy seat-steps are the same work', () => {
  for (const requests of [R, withOutput('C', 10), withPrompt('D', 4096), R.map((r) => ({ ...r, arrives: r.arrives * 3 }))]) {
    for (let seats = 1; seats <= 6; seats += 1) {
      const st = simulateStatic({ requests, seats });
      const co = simulateContinuous({ requests, seats });
      assert.ok(st.utilizationPct <= 100 && co.utilizationPct <= 100);
      assert.ok(co.lastStep <= st.lastStep, `${seats} seats`);
      assert.equal(st.busySeatSteps, co.busySeatSteps);
      assert.ok(st.live.every((r) => r.admitted >= r.arrives) && co.live.every((r) => r.admitted >= r.arrives));
    }
  }
});

test('a request that arrives after everything is done starts a fresh batch at its own arrival', () => {
  const requests = [{ id: 'X', arrives: 0, prompt: 2, output: 1 }, { id: 'Y', arrives: 9, prompt: 2, output: 2 }];
  assert.equal(span(simulateStatic({ requests, seats: 1 }), 'Y'), '9–11');
  assert.equal(span(simulateContinuous({ requests, seats: 1 }), 'Y'), '9–11');
});

test('the simulators are pure: frozen inputs, input order does not matter for arrival ties, same input twice gives deep-equal output (Review Focus 4)', () => {
  const frozen = Object.freeze(R.map((r) => Object.freeze({ ...r })));
  for (const fn of [simulateStatic, simulateContinuous]) {
    assert.deepEqual(fn({ requests: frozen, seats: 3 }), fn({ requests: frozen, seats: 3 }));
  }
  assert.deepEqual(scheduleTokens({ requests: frozen, seats: 3, budget: 512 }), scheduleTokens({ requests: frozen, seats: 3, budget: 512 }));
  const before = JSON.stringify(R);
  simulateContinuous({ requests: R, seats: 2 });
  assert.equal(JSON.stringify(R), before);
});

test('paddingWaste: (n − 1)(B − 1) — 100 new prompt tokens into a batch of 8 wastes 693', () => {
  assert.equal(paddingWaste(100, 8), 693);
  assert.equal(paddingWaste(100, new Array(8).fill({})), 693);
  assert.equal(paddingWaste(1, 8), 0);
  assert.equal(paddingWaste(100, 1), 0);
  assert.throws(() => paddingWaste(0, 8), /newPromptTokens must be a positive integer/);
  assert.throws(() => paddingWaste(10, 0), /runningRequests must be a positive integer or a non-empty array/);
  assert.throws(() => paddingWaste(10, []), RangeError);
});

test('input validation names the argument', () => {
  assert.throws(() => simulateStatic({ requests: [], seats: 3 }), /^RangeError: simulateStatic: requests must be a non-empty array/);
  assert.throws(() => simulateContinuous({ requests: R, seats: 0 }), /seats must be a positive integer/);
  assert.throws(() => simulateContinuous({ requests: R, seats: 1.5 }), RangeError);
  assert.throws(() => scheduleTokens({ requests: R, seats: 3, budget: 0 }), /budget must be a positive integer or Infinity/);
  assert.throws(() => scheduleTokens({ requests: R, seats: 3, budget: 2.5 }), RangeError);
  assert.throws(() => simulateStatic({ requests: [...R, R[0]], seats: 3 }), /request ids must be unique/);
  assert.throws(() => simulateStatic({ requests: [{ id: 1, arrives: 0, prompt: 1, output: 1 }], seats: 1 }), /every request needs a string id/);
  assert.throws(() => simulateStatic({ requests: [{ id: 'A', arrives: -1, prompt: 1, output: 1 }], seats: 1 }), /arrives must be an integer ≥ 0/);
  assert.throws(() => simulateStatic({ requests: [{ id: 'A', arrives: 0, prompt: 0, output: 1 }], seats: 1 }), /prompt must be a positive integer/);
  assert.throws(() => simulateStatic({ requests: [{ id: 'A', arrives: 0, prompt: 1, output: 0 }], seats: 1 }), /output must be a positive integer/);
  assert.throws(() => simulateStatic({ requests: null, seats: 1 }), RangeError);
});

// ---- scheduleTokens and the 4,096-token branch (batching §6; disaggregation frame 1) ----

// Step times from stepTime on the running example, context 16; returns each step with its ms and the clock at its end.
function timeline(requests, budget) {
  let clock = 0;
  return scheduleTokens({ requests, seats: 3, budget }).map((st) => {
    const ms = stepTime({ ...RUNNING_EXAMPLE, tokens: st.tokens, seqs: st.decode.length, context: 16 }).timeS * 1e3;
    clock += ms;
    return { ...st, ms, end: clock };
  });
}
const long = withPrompt('D', 4096);
const doneAt = (steps, id) => steps.filter((s) => s.decode.includes(id)).at(-1).end;
const prefillEnd = (steps, id) => steps.filter((s) => s.prefill.some((p) => p.id === id)).at(-1).end;
const stepsWith = (steps, id) => steps.filter((s) => s.decode.includes(id));

test('short prompts (defaults): step 0 carries 23 prompt tokens in 14.7 ms, every other step is 14.6 ms', () => {
  const steps = timeline(R, Infinity);
  assert.deepEqual(steps[0].prefill, [{ id: 'A', tokens: 8 }, { id: 'B', tokens: 5 }, { id: 'C', tokens: 10 }]);
  assert.equal(steps[0].tokens, 23);
  assert.equal(printed(steps[0].ms, 1), 14.7);
  assert.ok(steps.slice(1).every((s) => printed(s.ms, 1) === 14.6));
  assert.deepEqual(steps[1], { step: 1, decode: ['A', 'B', 'C'], prefill: [], tokens: 3, ms: steps[1].ms, end: steps[1].end });
});

test('budget off, D\'s prompt 4,096: step 3 = 4,098 tokens, 289.9 ms; D\'s prefill ends 333.8 ms; A done 348.4; C done 377.5', () => {
  const steps = timeline(long, Infinity);
  assert.equal(steps[3].tokens, 4098);
  assert.equal(printed(steps[3].ms, 1), 289.9);
  assert.deepEqual([printed(prefillEnd(steps, 'D'), 1), printed(doneAt(steps, 'A'), 1), printed(doneAt(steps, 'C'), 1)], [333.8, 348.4, 377.5]);
  const gap = Math.max(...steps.filter((s) => s.decode.includes('A') || s.decode.includes('C')).map((s) => s.ms));
  assert.equal(printed(gap, 1), 289.9);
});

test('budget 2048: steps 3–4 are 2,048 tokens, 144.9 ms each; D\'s prefill ends 348.2; A done 333.6; C 362.8', () => {
  const steps = timeline(long, 2048);
  assert.deepEqual([steps[3].tokens, steps[4].tokens], [2048, 2048]);
  assert.deepEqual([printed(steps[3].ms, 1), printed(steps[4].ms, 1)], [144.9, 144.9]);
  assert.deepEqual([printed(prefillEnd(steps, 'D'), 1), printed(doneAt(steps, 'A'), 1), printed(doneAt(steps, 'C'), 1)], [348.2, 333.6, 362.8]);
});

test('budget 512: slices 510, 510, 511, 511, 512 × 4, 6 over steps 3–11; steps 36.2 ms; D\'s prefill ends 348.2; A done 116.3; C 188.7', () => {
  const steps = timeline(long, 512);
  const slices = steps.filter((s) => s.prefill.some((p) => p.id === 'D'));
  assert.deepEqual(slices.map((s) => s.prefill.find((p) => p.id === 'D').tokens), [510, 510, 511, 511, 512, 512, 512, 512, 6]);
  assert.deepEqual([slices[0].step, slices.at(-1).step], [3, 11]);
  assert.deepEqual(slices.slice(0, 8).map((s) => printed(s.ms, 1)), new Array(8).fill(36.2));
  assert.deepEqual([printed(prefillEnd(steps, 'D'), 1), printed(doneAt(steps, 'A'), 1), printed(doneAt(steps, 'C'), 1)], [348.2, 116.3, 188.7]);
  assert.deepEqual([stepsWith(steps, 'A').filter((s) => printed(s.ms, 1) === 36.2).length, stepsWith(steps, 'C').filter((s) => printed(s.ms, 1) === 36.2).length], [2, 4]);
  assert.ok(steps.every((s) => s.tokens <= 512 || s.step < 3));
});

test('scheduleTokens: the budget is never exceeded and every prompt token is processed exactly once', () => {
  for (const requests of [R, long, withPrompt('C', 700)]) {
    for (const budget of [4, 16, 512, 2048, Infinity]) {
      const steps = scheduleTokens({ requests, seats: 3, budget });
      if (budget !== Infinity) {
        // decodes always run; only the prefill share is bounded by budget − decodes
        assert.ok(steps.every((s) => s.tokens <= Math.max(budget, s.decode.length)), `budget ${budget}`);
      }
      for (const r of requests) {
        const prefilled = steps.flatMap((s) => s.prefill).filter((p) => p.id === r.id).reduce((n, p) => n + p.tokens, 0);
        assert.equal(prefilled, r.prompt, `${r.id} at budget ${budget}`);
        assert.equal(steps.filter((s) => s.decode.includes(r.id)).length, r.output, `${r.id} decodes at budget ${budget}`);
      }
    }
  }
});

test('scheduleTokens with an Infinity budget reproduces simulateContinuous (admitted and last decode step)', () => {
  for (const requests of [R, withOutput('C', 10), long]) {
    for (let seats = 1; seats <= 5; seats += 1) {
      const sim = simulateContinuous({ requests, seats });
      const steps = scheduleTokens({ requests, seats });
      for (const r of sim.live) {
        assert.equal(steps.find((s) => s.prefill.some((p) => p.id === r.id)).step, r.admitted, `${r.id}/${seats}`);
        assert.equal(steps.filter((s) => s.decode.includes(r.id)).at(-1).step, r.finishes, `${r.id}/${seats}`);
      }
    }
  }
});

test('serving-overview frame 9: at step 4 A, C and D each advance one token together and B is done', () => {
  const at4 = scheduleTokens({ requests: R, seats: 3 }).find((s) => s.step === 4);
  assert.deepEqual(at4.decode, ['A', 'C', 'D']);
  assert.equal(simulateContinuous({ requests: R, seats: 3 }).live.find((r) => r.id === 'B').finishes, 2);
});

test('disaggregation frame 1: with chunks of 512 A sits through two 36.2 ms steps and C through four; unchunked it is one 289.9 ms step', () => {
  const chunked = timeline(long, 512);
  const heavy = (id) => stepsWith(chunked, id).filter((s) => printed(s.ms, 1) === 36.2).length;
  assert.deepEqual([heavy('A'), heavy('C')], [2, 4]);
  assert.equal(printed(Math.max(...timeline(long, Infinity).map((s) => s.ms)), 1), 289.9);
});

test('scheduleTokens: decodes always run, so a budget the decodes fill starves the prefill until they finish, then it catches up', () => {
  const requests = [{ id: 'A', arrives: 0, prompt: 2, output: 3 }, { id: 'B', arrives: 0, prompt: 2, output: 1 }];
  const steps = scheduleTokens({ requests, seats: 2, budget: 1 });
  assert.deepEqual(steps.slice(2, 5).map((s) => [s.decode, s.prefill]), [[['A'], []], [['A'], []], [['A'], []]]);
  assert.deepEqual(steps.slice(5).map((s) => s.prefill.map((p) => p.id).join('')), ['B', 'B', '']);
  assert.equal(steps.flatMap((s) => s.prefill).reduce((n, p) => n + p.tokens, 0), 4);
});

test('scheduleTokens skips steps with no work and waits for a late arrival', () => {
  const requests = [{ id: 'X', arrives: 0, prompt: 2, output: 1 }, { id: 'Y', arrives: 7, prompt: 3, output: 1 }];
  const steps = scheduleTokens({ requests, seats: 1 });
  assert.deepEqual(steps.map((s) => s.step), [0, 1, 7, 8]);
  assert.deepEqual(steps[2].prefill, [{ id: 'Y', tokens: 3 }]);
});
