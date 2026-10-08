// The midtraining lesson (docs/storyboards/midtraining.md): every word on the page, plus the stage and the toy.
// No DOM at import time, so the node page test can import it.
import { renderFor } from './frames.js';
import { mount } from './toy.js';
import { CAPTIONS } from './captions.js';
import { intuition, framing, factRows, mathNotes } from './facts.js';

export { CAPTIONS };

const STAND_IN = 'Curves are drawn as a fraction of the peak learning rate because most labs do not publish their peak; Nemotron 3 Super\'s numbers are its published ones. The decay curve\'s shape is not published for every lab and is drawn linear unless stated.';

const math = {
  blocks: [
    {
      tex: '\\text{cosine:}\\ \\eta(t) = \\eta_{\\min} + (\\htmlClass{hl-peak}{\\eta_{\\max}} - \\eta_{\\min})\\,\\tfrac12\\Big(1 + \\cos \\frac{\\pi (t - t_w)}{T - t_w}\\Big)',
      note: 'η, η_max, η_min, t, t_w and T are scalars; t is in tokens.',
    },
    {
      tex: '\\text{WSD:}\\ \\eta(t) = \\begin{cases} \\eta_{\\max}\\, t / t_w & t < t_w \\\\ \\htmlClass{hl-peak}{\\eta_{\\max}} & t_w \\le t \\le t_d \\\\ \\eta_{\\min} + (\\eta_{\\max} - \\eta_{\\min})\\, \\htmlClass{hl-d}{s\\!\\big(\\tfrac{t - t_d}{T - t_d}\\big)} & t > t_d \\end{cases} \\qquad s(p) = 1 - p \\ \\text{(linear)},\\ \\ 1 - \\sqrt{p}\\ \\text{(minus-sqrt)}',
      note: 't_d is where the decay starts; the decay share is (T − t_d) / T. The outline on the stage marks the decay.',
    },
    {
      tex: '\\text{attention work per new token} \\propto \\text{context length } L \\quad\\Rightarrow\\quad \\frac{L_{200K}}{L_{4K}} = 50',
      note: 'Attention only: the rest of the forward pass costs the same per token.',
    },
  ],
};

// The whole lesson. Pure; every dated number is filled from `data` (or the numbers.js stand-ins when it is null).
export function lessonFor(data) {
  return Object.freeze({
    slug: 'midtraining',
    hook: 'Why do labs save their best data for the end of pretraining, and their longest documents for its last few percent?',
    intuition: intuition(data),
    animation: {
      label: 'The learning-rate schedule, annealing, and context extension',
      steps: CAPTIONS.map((caption) => ({ caption })),
      render: renderFor(data),
      standIn: STAND_IN,
    },
    toy: {
      title: 'Plan the end of a run.',
      intro: 'Switch the schedule, choose where to stop or branch, and see how 2026 runs spend their last tokens. Rates are fractions of the peak; the Nemotron 3 Super chip adds its published absolute rate.',
      mount,
    },
    math: { blocks: math.blocks, notes: mathNotes(data) },
    facts: {
      framing: framing(),
      rows: factRows(),
      prose: ['Continue the recipe at [[sft]]. Related: [[rope]] for how position scaling works, and [[long-context-attention]] for how 2026 models make 1M tokens affordable.'],
    },
    takeaways: [
      'Runs warm the learning rate up, hold it, and decay it; cosine fixes the end on day one, while warmup-stable-decay lets you branch a decay off any plateau checkpoint, and labs still disagree on which is better.',
      'The decay is where the model settles, so labs anneal on their best data there; in 2026 that end phase is the named mid-training stage (still next-token prediction on documents).',
      'Context grows late and in a few short stages (GLM-5: 4K to 200K on 5% of its tokens) because long tokens cost more attention work and long documents are scarce; RoPE models rescale positions to match.',
    ],
    links: {
      next: [],
      further: [
        { title: 'Hugging Face, The Smol Training Playbook', href: 'https://huggingfacetb-smol-training-playbook.hf.space/', note: 'schedules, annealing and ablations' },
        { title: 'Andrej Karpathy, nanochat', href: 'https://github.com/karpathy/nanochat', note: 'a repo with a separate midtrain stage' },
      ],
    },
  });
}

// For spec validation only; the page mounts lessonFor(ctx.data).
export const LESSON = lessonFor(null);
