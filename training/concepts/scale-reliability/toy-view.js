// scale-reliability toy view (pure, no DOM): (state, data) → every string the toy "Plan a training run" prints, so the
// page shows exactly what the tests check. Hardware peaks come from data/hardware.json (`<chip>.<precision>_dense_tflops`).
import { deepFreeze } from '@math/core.js';
import { runPlan, gpuHoursAt, mfuFrom, checkpointLossParts } from '@math/scale.js';
import { lookupFact } from '@shared/claims.js';
import { LLAMA, DEEPSEEK, STAND_IN, PER_GPU_MTBF_H } from './numbers.js';
import { sci, int, gpuHours, hoursText, daysText, minutesText, dollarsM, pct, signedPct } from './format.js';

const SECONDS_PER_HOUR = 3600;
const MIN_PER_HOUR = 60;
const FP8_CHIP = 'h100-fp8';

// The chip-and-precision chips: the data entry and key each peak is read from.
export const CHIPS = deepFreeze([
  { value: 'h100-bf16', label: 'H100 BF16', entry: 'h100', key: 'bf16_dense_tflops' },
  { value: 'h100-fp8', label: 'H100 FP8', entry: 'h100', key: 'fp8_e4m3_dense_tflops' },
  { value: 'b200-bf16', label: 'B200 BF16', entry: 'b200', key: 'bf16_dense_tflops' },
  { value: 'b300-fp8', label: 'B300 FP8', entry: 'b300', key: 'fp8_e4m3_dense_tflops' },
]);

// Run presets: N (active parameters), D, GPUs, chip and MFU while training. DeepSeek-V3's MFU is back-solved (0.35637) so
// runPlan returns its reported 2.664M pre-training GPU-hours.
export const PRESETS = deepFreeze({
  llama: { label: 'Llama 3.1 405B', params: LLAMA.params, tokens: LLAMA.tokens, gpus: LLAMA.gpus, chip: 'h100-bf16', mfu: LLAMA.toyMfu, recordHours: LLAMA.gpuHours },
  deepseek: { label: 'DeepSeek-V3', params: DEEPSEEK.params, tokens: DEEPSEEK.tokens, gpus: DEEPSEEK.gpus, chip: 'h100-bf16', mfu: DEEPSEEK.toyMfu, recordHours: DEEPSEEK.pretrainGpuHours },
});
export const PRESET_FIELDS = Object.freeze(['params', 'tokens', 'gpus', 'chip', 'mfu']);

export const INITIAL_STATE = deepFreeze({
  preset: 'llama', ...Object.fromEntries(PRESET_FIELDS.map((k) => [k, PRESETS.llama[k]])),
  perGpuMtbfH: PER_GPU_MTBF_H, saveS: STAND_IN.saveS, restartMin: STAND_IN.restartMin, auto: true, intervalMin: 15, price: DEEPSEEK.dollarsPerGpuHour,
});

export const INVALID_LINE = "outside the formula's range: interval plus restart exceed half the cluster MTBF";
export const NEVER_LINE = 'saving takes longer than the interval allows, so the run never finishes';
export const NO_RECORD_LINE = 'No record to compare: this is a hypothetical run.';
export const STAND_IN_NOTE = "Save and restart times are stand-ins chosen so the Llama 3.1 405B preset lands near its reported >90% effective time; the per-GPU MTBF is derived from Llama 3.1's 419 interruptions and includes non-GPU causes.";
export const PRICE_NOTE = "$2 is DeepSeek's own assumption for V3, not a market price.";

export function peakFor(chip, data) {
  const spec = CHIPS.find((c) => c.value === chip);
  if (!spec) throw new RangeError(`scale-reliability: unknown chip ${chip}`);
  const fact = lookupFact(data?.hardware, spec.entry, spec.key);
  if (!fact) throw new RangeError(`scale-reliability: hardware.json has no ${spec.entry}.${spec.key}`);
  return fact.value;
}

// runPlan for a toy state. The result carries the peak that was divided by.
export function planFor(state, data) {
  const peakTflops = peakFor(state.chip, data);
  const plan = runPlan({
    params: state.params, tokens: state.tokens, gpus: state.gpus, peakTflops, mfu: state.mfu, perGpuMtbfHours: state.perGpuMtbfH,
    saveH: state.saveS / SECONDS_PER_HOUR, restartH: state.restartMin / MIN_PER_HOUR, intervalH: state.auto ? undefined : state.intervalMin / MIN_PER_HOUR,
    dollarsPerGpuHour: state.price,
  });
  return { ...plan, peakTflops };
}

const matchesPreset = (state, preset) => PRESET_FIELDS.filter((k) => k !== 'mfu').every((k) => state[k] === preset[k]);

// "Check against the record": the reported GPU-hours beside this run's, only while the run is the preset's own.
export function recordLine(state, plan) {
  const preset = PRESETS[state.preset];
  if (!matchesPreset(state, preset) || !plan.valid || !Number.isFinite(plan.gpuHours)) return NO_RECORD_LINE;
  if (state.preset === 'deepseek' && state.mfu === preset.mfu) {
    return `Reproduces ${gpuHours(preset.recordHours)} by construction (MFU while training set from the record): useful ${gpuHours(plan.usefulGpuHours)} GPU-hours plus a ${pct(plan.loss)} failure tax is ${gpuHours(plan.gpuHours)}.`;
  }
  const gap = signedPct(plan.gpuHours / preset.recordHours - 1);
  const scope = state.preset === 'llama' ? ' The card covers more stages than 6ND counts.' : '';
  const label = state.preset === 'llama' ? 'Model card' : 'Reported pre-training';
  return `${label}: ${gpuHours(preset.recordHours)} GPU-hours. This run: ${gpuHours(plan.gpuHours)} (${gap}).${scope}`;
}

function presetNote(state, plan, data) {
  if (state.preset !== 'deepseek') return '';
  const fp8 = mfuFrom({ flops: plan.flops, gpuHours: plan.gpuHours, peakTflops: peakFor(FP8_CHIP, data) });
  return `H100 peaks stand in for the H800 (its dense peaks are not in the data). DeepSeek ran its matmuls in FP8; against the FP8 peak the run-average is ${pct(fp8)}.`;
}

export function barParts(plan, peakTflops) {
  const ideal = gpuHoursAt({ flops: plan.flops, peakTflops, mfu: 1 });
  return [
    { name: 'useful (MFU)', value: ideal, hue: 1 },
    { name: 'below peak', value: Math.max(plan.usefulGpuHours - ideal, 0), hue: 2 },
    { name: 'lost to failures', value: Math.max(plan.gpuHours - plan.usefulGpuHours, 0), hue: 3, hatched: true },
  ];
}

const DASH = '—';

export function toyView(state, data) {
  const plan = planFor(state, data);
  const chip = CHIPS.find((c) => c.value === state.chip);
  const finite = Number.isFinite(plan.gpuHours);
  const ok = plan.valid && finite;
  const parts = checkpointLossParts({ intervalH: plan.intervalH, saveH: state.saveS / SECONDS_PER_HOUR, restartH: state.restartMin / MIN_PER_HOUR, mtbfH: plan.mtbfH });
  return {
    peak: int(plan.peakTflops),
    chipLabel: chip.label,
    flops: sci(plan.flops),
    usefulHours: gpuHours(plan.usefulGpuHours),
    mtbf: hoursText(plan.mtbfH),
    interval: minutesText(plan.intervalH * MIN_PER_HOUR),
    intervalSub: state.auto ? 'best' : 'chosen',
    loss: finite ? pct(plan.loss) : DASH,
    lossSplit: `save ${pct(parts.save)} · lost work ${pct(parts.lostWork)} · restarts ${pct(parts.restart)}`,
    gpuHours: ok ? gpuHours(plan.gpuHours) : DASH,
    days: ok ? daysText(plan.days) : DASH,
    cost: ok ? dollarsM(plan.cost) : DASH,
    runAverageMfu: ok ? `${pct(mfuFrom({ flops: plan.flops, gpuHours: plan.gpuHours, peakTflops: plan.peakTflops }))} of the ${chip.label} peak` : DASH,
    invalid: ok ? '' : (plan.valid ? NEVER_LINE : INVALID_LINE),
    record: recordLine(state, plan),
    presetNote: presetNote(state, plan, data),
    bar: ok ? barParts(plan, plan.peakTflops) : null,
    valid: ok,
  };
}

// The three try-this items, with every number computed by the same functions the toy uses.
export function tryThis(data) {
  const at = (patch) => toyView({ ...INITIAL_STATE, ...patch }, data);
  const [base, mfu38, g32, g100] = [{}, { mfu: 0.38 }, { gpus: 32768 }, { gpus: 100000 }].map(at);
  const [t5, t60, best, save15, save15Big] = [{ auto: false, intervalMin: 5 }, { auto: false, intervalMin: 60 }, {}, { saveS: 15 }, { saveS: 15, gpus: 100000 }].map(at);
  return [
    {
      prompt: `Llama 3.1 preset at MFU ${pct(LLAMA.toyMfu, 0)}: ${base.gpuHours} GPU-hours, ${base.days}, against the model card's ${gpuHours(LLAMA.gpuHours)}. Slide MFU to 38%: ${mfu38.gpuHours}.`,
      insight: "6ND ÷ (peak × MFU), plus a failure tax, reproduces a real run's GPU-hours to within a few percent; MFU is the number that turns FLOPs into days.",
      rest: " (The failure stand-ins were tuned to Llama 3.1's reported >90%, so this is a consistency check, not a prediction.)",
    },
    {
      prompt: `Keep everything and raise GPUs 16,384 → 32,768 → 100,000: ${[base, g32, g100].map((v) => v.days).join(' → ')}, but the loss grows ${[base, g32, g100].map((v) => v.loss).join(' → ')} and GPU-hours ${[base, g32, g100].map((v) => v.gpuHours).join(' → ')}.`,
      insight: "more GPUs finish sooner but pay a bigger failure tax, because the cluster's MTBF falls as the GPU count rises.",
      rest: '',
    },
    {
      prompt: `At 16,384 GPUs set the interval to 5 min (loss ${t5.loss}), 60 min (${t60.loss}), then "best" (${best.interval}, ${best.loss}). Halve the save time to 15 s: best becomes ${save15.interval} and the loss ${save15.loss}; at 100,000 GPUs the same change takes ${g100.loss} to ${save15Big.loss}.`,
      insight: 'the best checkpoint interval balances save cost against lost work, and faster checkpointing is worth the most on the biggest clusters.',
      rest: '',
    },
  ];
}
