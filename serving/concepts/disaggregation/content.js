// disaggregation: the complete lesson (storyboard docs/storyboards/disaggregation.md §3, §5, §7–§10). Pure, no DOM.
// lessonFor(data) fills the dated text from data/*.json; LESSON is the same lesson with no data (validation only).
import { fillText } from '@shared/claims.js';
import { render } from './frames.js';
import { mount } from './toy.js';
import { CAPTIONS } from './captions.js';
import { hook, intuition, takeaways, framing, factRows, BELOW } from './facts.js';

const SLUG = 'disaggregation';

const tex = String.raw;
const MATH_BLOCKS = Object.freeze([
  { tex: tex`\htmlClass{hl-kv}{t_{\text{transfer}}} = \frac{\text{prompt}\cdot \text{KV bytes/token}}{\text{link bandwidth}}, \qquad t_{\text{prefill}} \approx \frac{2\,N_{\text{active}}\cdot\text{prompt}}{\text{peak FLOP/s}} \;\Rightarrow\; \frac{\htmlClass{hl-kv}{t_{\text{transfer}}}}{t_{\text{prefill}}} = \frac{\text{KV bytes/token}\cdot \text{peak}}{2\,N_{\text{active}}\cdot\text{link}}` },
  { tex: tex`\text{tokens per expert per step} = \frac{\text{users per GPU}\cdot \text{EP}\cdot k}{E}, \qquad \text{weights per GPU} \approx \frac{W}{\text{EP}}\ \ (\text{replicated non-expert weights ignored})` },
  { tex: tex`\text{goodput} = \max\{\text{request rate} : \text{TTFT} \le \text{target}_1 \text{ and } \text{TPOT} \le \text{target}_2\}` },
]);

const MATH_NOTES = Object.freeze([
  'Shapes: one expert\'s weights are a few [7168 × 3072] matrices in V4-Pro; with t tokens each multiply is [t × 7168] · [7168 × 3072], and its intensity comes from the roofline\'s matrix-multiply cost (about 2t ÷ bytes per weight while t is small). The prefill estimate ignores the attention term; the page uses the same step-time function as [[prefill-decode]].',
  'Hover t_transfer to outline the KV arrows on the stage (the KV the prefill pass wrote, moving to the decode GPU). The ratio holds only past the prompt length where prefill stops being one weight read.',
]);

const FURTHER = Object.freeze([
  { title: 'DeepSeek, Inference System Overview', href: 'https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md', note: 'prefill EP32 and decode EP144 in production' },
  { title: 'LMSYS, Large-scale expert parallelism', href: 'https://lmsys.org/blog/2025-05-05-large-scale-ep/', note: 'the open reproduction with prefill/decode disaggregation' },
  { title: 'Hao AI Lab, Throughput is not all you need (DistServe)', href: 'https://haoailab.com/blogs/distserve/', note: 'why goodput, not raw throughput, is the measure' },
]);

export function lessonFor(data) {
  const fill = (text) => fillText(text, data);
  return {
    slug: SLUG,
    hook: fill(hook),
    intuition: intuition(data).map(fill),
    animation: {
      label: 'Disaggregated serving: shared GPUs, two pools, the KV transfer and wide expert parallelism in ten steps',
      steps: CAPTIONS.map((caption) => ({ caption })),
      render,
      standIn: 'The 4,096-token branch, the two service targets (400 ms, 15 ms) and the even routing of tokens to experts are illustrations; every step time is computed for Llama-3.1-70B in FP8 on one H200.',
      belowFor: (index) => (BELOW[index] ?? []).map(fill),
    },
    toy: {
      title: 'Ship the KV, feed the experts',
      intro: 'Two panels. "Ship the KV" moves one request\'s cache between pools (Llama-3.1-70B, FP8, H200). "Feed the experts" spreads DeepSeek-V4-Pro over GB300 GPUs. Link speeds are per GPU, each way.',
      mount,
    },
    math: { blocks: MATH_BLOCKS, notes: MATH_NOTES },
    facts: { framing: framing(), rows: factRows(data) },
    takeaways: takeaways(data).map(fill),
    links: { next: ['serving-calculator'], further: FURTHER },
  };
}

export const LESSON = lessonFor(null);
