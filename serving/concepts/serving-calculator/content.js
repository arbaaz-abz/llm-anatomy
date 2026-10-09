// serving-calculator: the complete lesson (storyboard docs/storyboards/serving-calculator.md §3, §5, §7–§10). Pure, no DOM.
// lessonFor(data) fills the dated text from data/*.json; LESSON is the same lesson with no data (validation only).
import { fillText } from '@shared/claims.js';
import { render } from './frames.js';
import { mount } from './toy.js';
import { CAPTIONS } from './captions.js';
import { hook, intuition, takeaways, framing, factRows, BELOW } from './facts.js';

const SLUG = 'serving-calculator';

const tex = String.raw;
const MATH_BLOCKS = Object.freeze([
  { tex: tex`\text{GPUs}_{\min} = \left\lceil \frac{W}{\text{HBM}} \right\rceil, \qquad \text{free} = \text{HBM} - \frac{W}{G}, \qquad \text{users}_{\max} = \left\lfloor \frac{\text{free}}{\htmlClass{hl-kv}{\text{KV/token}}\cdot \text{context}} \right\rfloor` },
  { tex: tex`t_{\text{step}}(u) = \max\!\left(\frac{2 N_{\text{active}}\, u}{\text{peak}},\ \frac{W/G + \tfrac{2 N b}{d}\,u + u\cdot\text{context}\cdot\htmlClass{hl-kv}{\text{KV/token}}}{\text{BW}}\right), \qquad u_{\text{target}} = \max\{u : t_{\text{step}}(u) \le 1/\text{target}\}` },
  { tex: tex`\$\text{ per M tokens} = \frac{\$\text{ per GPU-hour}}{3600\cdot \text{tokens/s per GPU}}\cdot 10^6, \qquad \frac{\text{output cost}}{\text{input cost}} = \frac{\text{prefill tokens/s per GPU}}{\text{decode tokens/s per GPU}}` },
]);

const MATH_NOTES = Object.freeze([
  'Shapes: none (scalars per GPU). G = GPUs per replica, u = users per GPU, b = bytes per element, d = d_model. With experts spread evenly and attention data-parallel, each GPU holds its own users\' KV and W / G of the weights (replicated non-expert weights are ignored).',
  'Hover KV/token to outline the KV part of the followed GPU\'s memory bar, drawn twice (low and high) for V4-Pro. The toy\'s "Check my work" box prints each of these lines with the numbers of the current state.',
]);

const FURTHER = Object.freeze([
  { title: 'InferenceX (SemiAnalysis)', href: 'https://inferencex.semianalysis.com/about', note: 'live tokens-per-GPU against per-user curves, and $ per million tokens' },
  { title: 'DeepSeek, Inference System Overview', href: 'https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md', note: 'production throughput, cache hit rate and the cost accounting' },
  { title: 'LMSYS, GB300 long context', href: 'https://lmsys.org/blog/2026-02-19-gb300-longctx', note: 'multi-token prediction and concurrency at 128K on GB300 NVL72' },
]);

export function lessonFor(data) {
  const fill = (text) => fillText(text, data);
  return {
    slug: SLUG,
    hook: fill(hook),
    intuition: intuition().map(fill),
    animation: {
      label: 'Serving a 1T model: weights, KV, speed and cost in ten steps',
      steps: CAPTIONS.map((caption) => ({ caption })),
      render,
      standIn: 'Every number is a floor from bytes and FLOPs only, computed for DeepSeek-V4-Pro on GB300 NVL72 with 16 GPUs per replica; the measured points are InferenceX\'s, with its counting convention stated where it matters.',
      belowFor: (index) => (BELOW[index] ?? []).map(fill),
    },
    toy: {
      title: 'Size a deployment',
      intro: 'Pick a model, a GPU and a workload; every output shows its formula with your numbers in "Check my work". The first four presets are the worked example\'s defaults.',
      mount,
    },
    math: { blocks: MATH_BLOCKS, notes: MATH_NOTES },
    facts: { framing: framing(), rows: factRows() },
    takeaways: takeaways().map(fill),
    links: { next: [], further: FURTHER },
  };
}

export const LESSON = lessonFor(null);
