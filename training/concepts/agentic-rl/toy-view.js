// agentic-rl toy view model (pure, no DOM): state → every string and mark the toy prints. toy.js only paints it.
import { isCorrection, mismatchRatio, rolloutSchedule } from '@math/agentic.js';
import { formatRatio } from '@math/core.js';
import { ADVANTAGES, DURATIONS, ENGINE_P, FOLLOWED, GROUP_SIZE, ROWS, TOKEN_COUNT, trainerBf16 } from './numbers.js';
import { advantage3, carriedRows, fixed3, minutes, ofTotal, percent1, rowsText, signed3 } from './format.js';

export const CORRECTIONS = Object.freeze([
  { value: 'none', label: 'ignore' }, { value: 'full', label: 'full IS' }, { value: 'tis', label: 'truncated IS' }, { value: 'icepop', label: 'IcePop' },
]);
export const PRECISIONS = Object.freeze([
  { value: 'bf16', label: 'BF16' }, { value: 'fp16', label: 'FP16: toy model of rounding only (reported fix)' },
]);
export const LAMBDAS = Object.freeze([
  { value: 'all', label: 'all 8 (sync)', mode: 'sync', lambda: 1 },
  { value: 'six', label: '6 of 8', mode: 'partial', lambda: 0.75 },
  { value: 'four', label: '4 of 8', mode: 'partial', lambda: 0.5 },
]);
export const INITIAL_STATE = Object.freeze({ correction: 'none', precision: 'bf16', lambda: 'all', row: FOLLOWED.row, pos: FOLLOWED.pos });

const CAP = 2;
const BAND = Object.freeze([0.5, 2]);

// Everything about one token under the state's correction and precision.
export function tokenAt(state, row, pos) {
  const engine = ENGINE_P[row][pos];
  const rho = mismatchRatio(trainerBf16(row, pos) / engine, { precision: state.precision });
  const { weight, masked } = isCorrection(rho, { mode: state.correction, cap: CAP, band: BAND });
  const advantage = ADVANTAGES[row];
  return { row, pos, text: ROWS[row][pos], advantage, engine, trainer: engine * rho, rho, weight, masked, push: advantage * weight + 0 };
}

export const allTokens = (state) => ROWS.flatMap((tokens, row) => tokens.map((_, pos) => tokenAt(state, row, pos)));
export const maskedCount = (state) => allTokens(state).filter((t) => t.masked).length;

export function schedule(state) {
  const { mode, lambda } = LAMBDAS.find((l) => l.value === state.lambda);
  return rolloutSchedule(DURATIONS, { mode, lambda });
}

// The inspector rows: [data-readout name, label, text].
export function inspectorRows(state) {
  const t = tokenAt(state, state.row, state.pos);
  return [
    ['inspector-token', 'Selected token', `row ${t.row + 1}, token ${t.pos + 1}: ${t.text}`],
    ['inspector-a', 'Advantage A', advantage3(t.advantage)],
    ['inspector-engine', 'Engine p (sampled it)', fixed3(t.engine)],
    ['inspector-trainer', 'Trainer p (updates on it)', fixed3(t.trainer)],
    ['inspector-rho', 'Ratio ρ = trainer / engine', fixed3(t.rho)],
    ['inspector-weight', 'IS weight', fixed3(t.weight)],
    ['inspector-masked', 'Masked', t.masked ? 'yes' : 'no'],
    ['inspector-push', 'Push = A × weight', signed3(t.push)],
  ];
}

export function timelineView(state) {
  const s = schedule(state);
  return {
    schedule: s,
    iteration: minutes(s.iterationTime),
    utilization: percent1(s.busy, GROUP_SIZE * s.iterationTime),
    carried: rowsText(carriedRows(s.carried)),
  };
}

export function toyView(state) {
  return {
    inspector: inspectorRows(state),
    masked: ofTotal(maskedCount(state), TOKEN_COUNT),
    timeline: timelineView(state),
    note: 'The probabilities are hand-picked stand-ins. The FP16 setting models rounding only; mismatch from MoE routing does not shrink with precision (DeepSeek\'s fix for that is Keep Routing).',
  };
}

const withState = (patch) => ({ ...INITIAL_STATE, ...patch });
const followed = (patch) => tokenAt(withState(patch), FOLLOWED.row, FOLLOWED.pos);

// The three try-this items of storyboard §6, every number computed from the same functions as the toy.
export function tryThis() {
  const steps = CORRECTIONS.map((c) => ({ ...c, t: followed({ correction: c.value }) }));
  const [none, full, tis, icepop] = steps.map((s) => s.t);
  const fp = (row, pos) => tokenAt(withState({ correction: 'icepop', precision: 'fp16' }), row, pos).rho;
  const bf = (row, pos) => tokenAt(withState({ correction: 'icepop' }), row, pos).rho;
  const lambdas = LAMBDAS.map((l) => timelineView(withState({ lambda: l.value })));
  const maskedBf = maskedCount(withState({ correction: 'icepop' }));
  const maskedFp = maskedCount(withState({ correction: 'icepop', precision: 'fp16' }));
  return [
    {
      prompt: `Predict first: should the trainer fully trust the ratio? With row 4's 48 selected, step through “How the trainer treats the mismatch”: ignore → weight ${fixed3(none.weight)}, push ${signed3(none.push)}; full IS → weight ${fixed3(full.weight)}, push ${signed3(full.push)}; truncated IS → weight ${fixed3(tis.weight)}, push ${signed3(tis.push)}; IcePop → masked, push ${signed3(icepop.push)}, and the readout says ${ofTotal(maskedBf, TOKEN_COUNT)} masked.`,
      insight: 'corrections trade bias for variance.',
      rest: ` Ignoring the gap is biased; full reweighting is unbiased but lets one token push ${formatRatio(full.weight)} as hard; truncation caps it (at ${CAP} in this toy), and IcePop simply drops tokens where engine and trainer disagree more than twofold.`,
    },
    {
      prompt: `Keep IcePop and switch “Rollout and trainer number format” to FP16. Row 4's ρ falls ${fixed3(bf(3, 7))} → ${fixed3(fp(3, 7))}, row 7's ${fixed3(bf(6, 4))} → ${fixed3(fp(6, 4))}, row 6's ${fixed3(bf(5, 0))} → ${fixed3(fp(5, 0))}, and the readout drops to ${ofTotal(maskedFp, TOKEN_COUNT)} masked.`,
      insight: 'part of the mismatch is rounding,',
      rest: ' and a finer number format shrinks it before any correction is needed (a reported fix). The rest, such as MoE routing that differs between engine and trainer, needs its own fix (Keep Routing).',
    },
    {
      prompt: `Step “Update when this share of episodes is done” from all 8 to 6 of 8 to 4 of 8: utilization ${lambdas.map((l) => l.utilization).join(' → ')}, while the rows that finish under newer weights go from ${lambdas.map((l) => l.carried).join(' to ')}.`,
      insight: 'async is a trade.',
      rest: ' The less the trainer waits, the more of each batch is off-policy, and the more work the corrections from try-this 1 have to do. (Kimi K3\'s partial rollouts pause at a fraction λ of finished episodes and resume the rest next iteration.)',
    },
  ];
}
