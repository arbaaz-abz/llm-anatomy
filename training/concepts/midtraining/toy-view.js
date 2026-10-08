// The midtraining toy's pure view model: (state, data) -> every string and curve the toy prints (storyboard §6). No DOM.
import { lrAt, lrCurve, attentionCostRatio } from '@math/schedule.js';
import { budgetShares } from '@math/pipeline.js';
import { formatCount, formatRatio } from '@math/core.js';
import { runData, labelK } from './facts.js';
import { FALLBACK } from './numbers.js';
import { fmt3, ofPeak, sci, mainShare, checkWork, scheduleOptions, lrAtStop, otherSchedule, scheduleName } from './format.js';

export const NOT_PUBLISHED = 'not published';
const CURVE_SAMPLES = 100;
const TOKENS_PER_T = 1e12;

// Nemotron 3 Super's absolute learning rate at the stop (t in trillions of tokens), only while its chip is on.
export function nemotronRate(state, nemotron) {
  const total = nemotron.total / TOKENS_PER_T;
  const t = (state.stopAt / 100) * total;
  return lrAt(t, {
    kind: 'wsd', peak: nemotron.peak, floor: nemotron.floor, warmup: nemotron.warmupTokens / TOKENS_PER_T, total,
    decayStart: total - nemotron.decayTokens / TOKENS_PER_T, decayShape: nemotron.shape,
  });
}

// One row per context stage of the chosen run: tokens, share of the run (budgetShares), attention work vs the first stage.
export function stageRows(runId, data) {
  const run = runData(data).runs[runId];
  const shares = budgetShares(run.stages);
  const first = labelK(run.stages[0].name);
  return run.stages.map((stage, i) => {
    const published = stage.value !== null;
    return {
      name: stage.name,
      tokens: published ? formatCount(stage.value) : NOT_PUBLISHED,
      share: published ? mainShare(shares.parts[i].share) : NOT_PUBLISHED,
      attention: formatRatio(attentionCostRatio(labelK(stage.name), first)),
    };
  });
}

// The curves the toy plots: the chosen schedule solid, the other muted, and the followed stop marker on the chosen one.
export function curves(state) {
  const chosen = scheduleOptions(state);
  const other = scheduleOptions(state, otherSchedule(state.schedule));
  const stop = lrAtStop(state);
  return {
    chosen: lrCurve(chosen, CURVE_SAMPLES),
    other: lrCurve(other, CURVE_SAMPLES),
    marker: { x: state.stopAt / 100, y: stop, label: `${state.stopAt}%: ${ofPeak(stop)} of peak`, followed: true },
    decayStart: state.schedule === 'cosine' ? null : chosen.decayStart,
  };
}

export function toyView(state, data) {
  const { nemotron } = runData(data);
  return {
    chosen: { name: scheduleName(state.schedule), lr: fmt3(lrAtStop(state)) },
    other: { name: scheduleName(otherSchedule(state.schedule)), lr: fmt3(lrAtStop(state, otherSchedule(state.schedule))) },
    nemotron: state.preset === 'nemotron' ? sci(nemotronRate(state, nemotron)) : null,
    check: checkWork(state),
    rows: stageRows(state.run, data),
    runName: FALLBACK.runs[state.run].name,
  };
}
