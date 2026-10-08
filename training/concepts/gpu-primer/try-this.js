// Storyboard §6 "Try this": three prompts, each leading to a named insight. Every number is computed from data/hardware.json
// (chipPreset) and math/roofline.js (analyze), printed at the storyboard's prose precision. Pure, no DOM.
import { formatDuration } from '@math/core.js';
import { chipPreset } from './hardware.js';
import { analyze, fixed1, int, pctProse } from './format.js';

// One chip, format and token count → its analysis; the only bandwidth end of a single-bandwidth chip.
const at = (data, chip, fmt, tokens) => {
  const a = analyze({ chip, fmt, tokens }, chipPreset(data, chip));
  return { a, e: a.ends[0] };
};
const peakShare = ({ a, e }) => pctProse(e.attainable, a.peakTflops);
const ridge = ({ e }) => int(e.ridge);
const intensity = ({ a }) => int(a.intensity);
const crossing = ({ e }) => fixed1(e.crossing);
const crossingInt = ({ e }) => int(e.crossing);

function first(data) {
  const [t4, t64, t256] = [4, 64, 256].map((n) => at(data, 'h100', 'bf16', n));
  return {
    prompt: `H100, BF16, 4 tokens: ${peakShare(t4)} of peak, memory-bound, ${formatDuration(t4.e.time.memoryS)} of memory time for ${formatDuration(t4.e.time.computeS)} of math. `
      + `Predict how many tokens it takes to become compute-bound, then slide: 64 (${peakShare(t64)}), 256 (${peakShare(t256)}, still memory-bound), 512 (compute-bound). The readout says ${crossing(t4)}.`,
    insight: 'in BF16 the intensity is about the number of tokens sharing each weight read (in general, 2 × tokens ÷ bytes per number), so batch size is the lever.',
    rest: ' Training pushes thousands of tokens through every weight and is compute-bound; decoding for a few users is not (see [[prefill-decode]]).',
  };
}

function second(data) {
  const [bf, f8] = ['bf16', 'fp8'].map((f) => at(data, 'h100', f, 256));
  return {
    prompt: `H100 at 256 tokens: switch BF16 → FP8. Intensity ${intensity(bf)} → ${intensity(f8)}, ridge ${ridge(bf)} → ${ridge(f8)}, still memory-bound; `
      + `memory time ${formatDuration(bf.e.time.memoryS)} → ${formatDuration(f8.e.time.memoryS)}; "tokens needed" stays ${crossing(bf)} → ${crossing(f8)}.`,
    insight: 'on an H100, FP8 doubles both sides of the roof, so a multiply runs about twice as fast on either side of the ridge, but the batch needed to cross does not change.',
    rest: '',
  };
}

function third(data) {
  const [b3bf, b3f4] = ['bf16', 'fp4'].map((f) => at(data, 'b300', f, 512));
  const ridges = ['h100', 'b200', 'b300', 'h200'].map((c) => ridge(at(data, c, 'bf16', 512)));
  const [b2bf, b2f4] = ['bf16', 'fp4'].map((f) => at(data, 'b200', f, 512));
  const b200Shift = `${crossingInt(b2bf)} → ${crossingInt(b2f4)}`;
  return {
    prompt: `B300 at 512 tokens: BF16 is compute-bound (intensity ${intensity(b3bf)} vs ridge ${ridge(b3bf)}). Switch to FP4: intensity ${intensity(b3f4)} vs ridge ${ridge(b3f4)}, now memory-bound; `
      + `tokens needed ${crossingInt(b3bf)} → ${crossingInt(b3f4)}. Then step H100 → B200 → B300 in BF16: the ridge reads ${ridges.slice(0, 3).join(', ')}; the H200 reads ${ridges[3]}. `
      + `Last, B200 in FP4: tokens needed ${b200Shift}.`,
    insight: `the BF16 ridge is flat across NVIDIA's flagship generations (the H200's bandwidth bump is the exception, and it lowers the crossing); on the B300 and Rubin, FP4 compute outran memory, so FP4 needs bigger batches there, while on a B200 the crossing barely moves (${b200Shift}).`,
    rest: ' ("Lower precision pushes the ridge right" is true in FLOPs per byte; in tokens it depends on the chip.)',
  };
}

// [{ prompt, insight, rest }]; the page prints "prompt → Insight: insight rest", with [[slug]] as the lesson's title.
export const tryThis = (data) => [first(data), second(data), third(data)];
