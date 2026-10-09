// scale-reliability: the complete lesson (storyboard docs/storyboards/scale-reliability.md §3, §5, §7–§10). Pure, no DOM.
// lessonFor(data) fills the dated text from data/*.json; LESSON is the same lesson with no data (validation, tests).
import { render } from './frames.js';
import { mount } from './toy.js';
import { CAPTIONS } from './captions.js';
import { fillText } from '@shared/claims.js';
import { BELOW, FRAMING, factRows } from './facts.js';

const SLUG = 'scale-reliability';

const HOOK = 'Llama 3.1 405B needed 3.8 × 10²⁵ FLOPs{llama-3.1-405b.training_flops|cite}, which {llama-3.1-405b.training_gpus|int} H100s at full speed would finish in 27 days. Why did it take nearly three times as many GPU-hours as that?';

const INTUITION = Object.freeze([
  "Start with the arithmetic. Training costs about 6 FLOPs per parameter per token: 2 in the forward pass and 4 in the backward pass. That rule ignores attention's own FLOPs, which matter at long context, but for Llama 3.1 405B ({llama-3.1-405b.release_date|year}) it gives 6 × {llama-3.1-405b.total_params|count} × {llama-3.1-405b.pretrain_tokens|count} = 3.8 × 10²⁵, the paper's own figure. At an H100's {hw:h100.bf16_dense_tflops} TFLOPS that is 10.6 million GPU-hours. Meta's model card reports {llama-3.1-405b.training_gpu_hours|count4}. The card's figure also covers the long-context and post-training stages, where 6ND undercounts, so the run-average MFU it implies, 34.5%, is a floor.",
  'The first gap is utilization. The share of peak a run actually delivers is its model FLOPs utilization, MFU. Llama 3.1 ran at 38–43% MFU while training: the GPUs wait on memory-bound kernels ([[gpu-primer]]), on communication that could not hide behind compute ([[cluster-topology]]) and on pipeline bubbles ([[parallelism]]). Overlapping communication with compute and running matrix multiplies in FP8 win some of it back. The second gap is reliability. With 16,384 GPUs, something fails every few hours; each failure throws away the work since the last checkpoint plus the time to restart, and checkpoints themselves cost time. Llama 3.1 kept {llama-3.1-405b.effective_time} of its time productive. Divide the 34.5% run-average by that 0.9 and you are back inside the 38–43% band: utilization and failures together answer the hook.',
  "The same arithmetic prices a run. GPU-hours times a price per hour is the bill: DeepSeek put V3's {deepseek-v3.training_gpu_hours|count4} H800-hours at $2 each, ${deepseek-v3.training_cost_usd_reported|count4}, for the final run alone. The price of scale is the failure tax: it grows with the GPU count, so on the biggest runs faster checkpointing and recovery can be worth as much as faster chips (try this 3).",
]);

const tex = String.raw;
const MATH_BLOCKS = Object.freeze([
  { tex: tex`\htmlClass{hl-flops}{C} \approx 6\,N\,D \qquad \text{GPU-hours} = \frac{C}{\htmlClass{hl-peak}{P_{\text{peak}}}\cdot \htmlClass{hl-mfu}{\text{MFU}} \cdot 3600} \qquad \text{MFU} = \frac{C}{\text{GPU-hours}\cdot 3600 \cdot P_{\text{peak}}}` },
  { tex: tex`\text{MTBF}_{\text{cluster}} = \frac{\text{MTBF}_{\text{GPU}}}{n_{\text{GPU}}} \qquad \htmlClass{hl-loss}{\text{loss}} \approx \frac{t_{\text{save}}}{T} + \frac{T/2 + t_{\text{restart}}}{\text{MTBF}_{\text{cluster}}} \qquad T^{*} = \sqrt{2\, t_{\text{save}}\, \text{MTBF}_{\text{cluster}}}` },
  { tex: tex`\text{cost} = \frac{\text{GPU-hours}_{\text{useful}}}{1 - \text{loss}} \times \text{price per GPU-hour}` },
]);

const MATH_NOTES = Object.freeze([
  'Symbols: N parameters (active parameters for a Mixture of Experts), D tokens, T the checkpoint interval. (a) 6ND counts the matmuls of the forward pass (2ND) and the backward pass (4ND); attention\'s own FLOPs grow with context length and are ignored, so long-context stages cost more than 6ND says.',
  '(b) The loss formula is first-order: one failure per MTBF on average, losing half an interval of work plus a restart; T* (the Young/Daly rule) follows by setting its derivative to zero. It overstates the loss when the restart time approaches the cluster MTBF, which is why the toy flags an interval plus restart above half the MTBF. (c) Recomputation ([[training-memory]]) adds FLOPs that MFU does not count (hardware FLOPs utilization, HFU, does). Hover a highlighted term to outline its glyph on the stage: the FLOPs card, the "at peak" bar, the "below peak" part and the hatched "lost" parts.',
]);

const TAKEAWAYS = Object.freeze([
  "A run's FLOPs are about 6 × parameters × tokens; dividing by peak × MFU gives GPU-hours. Llama 3.1 405B: 3.8 × 10²⁵ FLOPs, 38–43% MFU while training, 30.84M H100-hours in all (a run-average of at least 34.5%). Always name the peak an MFU divides by, especially with FP8.",
  'Utilization is lost to memory-bound kernels, exposed communication and bubbles; overlap (DeepSeek-V3 gave 20 SMs to communication) and FP8 matmuls with fine-grained scales win part of it back.',
  "Failures scale with GPU count (Llama 3.1 405B: one every 3.1 hours on 16,384 GPUs); the checkpoint interval trades save time against lost work, and the failure tax grows with scale (9% to 28% from 16K to 100K GPUs with the same parts). GPU-hours times price is the bill (DeepSeek-V3: $5.6M by its own assumption).",
]);

const FURTHER = Object.freeze([
  { title: 'The Llama 3 Herd of Models (Meta)', href: 'https://arxiv.org/abs/2407.21783', note: 'the infrastructure and reliability sections' },
  { title: 'ML Engineering Open Book (Stas Bekman)', href: 'https://github.com/stas00/ml-engineering', note: 'fault tolerance, with the war stories' },
  { title: 'Transformer Math 101 (EleutherAI)', href: 'https://blog.eleuther.ai/transformer-math/', note: 'the same FLOPs and memory arithmetic, worked step by step' },
]);

export function lessonFor(data) {
  const fill = (text) => fillText(text, data);
  return {
    slug: SLUG,
    hook: fill(HOOK),
    intuition: INTUITION.map(fill),
    animation: {
      label: 'Training at 10,000 GPUs: from 6ND to GPU-hours, failures, checkpoints and the bill, in eleven steps',
      steps: CAPTIONS.map((caption) => ({ caption })),
      render,
      standIn: 'The 100 ms step and the 30 s save and 3 min restart are stand-ins; the Llama 3.1 and DeepSeek-V3 figures are published.',
      belowFor: (index) => (BELOW[index] ?? []).map(fill),
    },
    toy: {
      title: 'Plan a training run',
      intro: 'It opens on the Llama 3.1 405B preset, so the numbers match the animation; one chip jumps to DeepSeek-V3, and every slider is yours.',
      mount,
    },
    math: { blocks: MATH_BLOCKS, notes: MATH_NOTES },
    facts: { framing: FRAMING, rows: factRows(data) },
    takeaways: TAKEAWAYS,
    links: { next: [], further: FURTHER },
  };
}

export const LESSON = lessonFor(null);
