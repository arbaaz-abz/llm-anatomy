// training-memory: the complete lesson (storyboard docs/storyboards/training-memory.md §3, §5, §7–§10). Pure, no DOM.
// lessonFor(data) fills the dated text from data/*.json; LESSON is the same lesson with no data (validation, tests).
import { render } from './frames.js';
import { mount } from './toy.js';
import { CAPTIONS } from './captions.js';
import { fillText } from '@shared/claims.js';
import { BELOW, factRows, framing } from './facts.js';

const SLUG = 'training-memory';

const HOOK = "GPT-3's weights fit in 350 GB. Why does training it need 2.8 TB before it has seen a single token?";

const INTUITION = Object.freeze([
  'Training keeps four things for every parameter. The weight itself, in BF16: 2 bytes. Its gradient, which backpropagation writes once per step: 2 more. Adam\'s two running averages, which decide how far each weight moves, kept in FP32 because they are tiny and must stay precise: 8 bytes. And an FP32 master copy of the weight, because adding a small update to a BF16 number would often round it away: 4 bytes. That is 16 bytes per parameter, eight times the 2 bytes that inference needs, and it answers the hook: the other 2.45 TB are 0.35 TB of gradients and 2.1 TB of optimizer state (moments and master copy).',
  'Then come activations, the intermediate results the forward pass saves because the backward pass needs them. They grow with the tokens in flight, not with the parameters, and at long sequence lengths they dominate. Training keeps no KV cache: the whole sequence goes through in one pass under the causal mask, and what grows with the tokens is the activations saved for the backward pass. The cheapest fix is to save less and recompute: keep only each block\'s input and run its forward again during backward. Training code usually calls this activation checkpointing (or gradient checkpointing). That costs about a third more compute and saves most of the memory.',
  'Finally, sharding. Many GPUs training on different data each hold a full copy of the state, so adding GPUs adds copies, not room. ZeRO (Zero Redundancy Optimizer; PyTorch\'s FSDP is the same idea) gives each GPU one slice of the state and has the GPUs pass the rest around just in time. Shard the optimizer state, then the gradients, then the weights, and per-GPU state falls from 2.8 TB to 44 GB on 64 GPUs. The price is traffic: the fully sharded stage moves about 1.5 times the data of plain data parallelism, and ZeRO\'s stages leave activations alone, so they still have to fit on each GPU (see [[parallelism]], which splits those).',
]);

const tex = String.raw;
const MATH_BLOCKS = Object.freeze([
  { tex: tex`\text{bytes per parameter} = \underbrace{\htmlClass{hl-w}{2}}_{\text{weight}} + \underbrace{\htmlClass{hl-g}{2}}_{\text{gradient}} + \underbrace{\htmlClass{hl-o}{8}}_{\text{Adam } m,\,v} + \underbrace{\htmlClass{hl-o}{4}}_{\text{FP32 master}} = 16` },
  { tex: tex`\text{ZeRO-1: } 4\Psi + \frac{12\Psi}{N} \qquad \text{ZeRO-2: } 2\Psi + \frac{14\Psi}{N} \qquad \text{ZeRO-3: } \frac{16\Psi}{N}` },
  { tex: tex`\htmlClass{hl-act}{A_{\text{layer}}} = s\,b\,h\left(34 + \frac{5\,a\,s}{h}\right) \ \text{bytes} \quad\to\quad 34\,s\,b\,h \ (\text{selective}) \quad\to\quad 2\,s\,b\,h \ (\text{full})` },
]);

const MATH_NOTES = Object.freeze([
  'Shapes: Ψ parameters; N data-parallel GPUs; s sequence length, b micro-batch, h hidden size, a heads (GPT-3: 2,048, 1, 12,288, 96). The 5as/h term is the attention-score grid [a × s × s] (softmax output, dropout mask and input). Hover a highlighted term to outline its segment on the stage.',
  'Notes: (a) the factor 34 assumes BF16 activations and a 4h MLP. (b) A full extra forward pass is about a third of a forward + backward step, because backward costs about two forwards. (c) ZeRO\'s traffic: stages 0 to 2 move about 2Ψ numbers per step (reduce-scatter + all-gather), stage 3 about 3Ψ.',
]);

const TAKEAWAYS = Object.freeze([
  'Mixed-precision training with Adam costs 16 bytes per parameter (weight 2, gradient 2, Adam moments 8, FP32 master copy 4), eight times inference\'s 2: GPT-3\'s shape needs 2.8 TB of state.',
  'Activations scale with tokens in flight and can exceed the state; recomputation trades about a third more compute for most of that memory (GPT-3, one sequence: 275 GB to 4.8 GB).',
  'Data parallelism copies the state; ZeRO / FSDP shards it (2.8 TB to 44 GB on 64 GPUs) at up to 1.5 times the traffic; its three stages leave activations alone. Memory follows total parameters, so a 1T mixture of experts needs about 200 H100s just to hold its state.',
]);

const FURTHER = Object.freeze([
  { title: 'ZeRO & DeepSpeed (Microsoft Research blog)', href: 'https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/', note: 'the animated sharding figure' },
  { title: 'Transformer Math 101 (EleutherAI)', href: 'https://blog.eleuther.ai/transformer-math/', note: 'memory and compute formulas worked out' },
  { title: 'The Ultra-Scale Playbook (Hugging Face)', href: 'https://huggingface.co/spaces/nanotron/ultrascale-playbook', note: 'every kind of parallelism, with experiments' },
]);

export function lessonFor(data) {
  const fill = (text) => fillText(text, data);
  return {
    slug: SLUG,
    hook: HOOK,
    intuition: INTUITION,
    animation: {
      label: 'Where training memory goes: one parameter\'s bytes, saved activations, and ZeRO sharding, in eleven steps',
      steps: CAPTIONS.map((caption) => ({ caption })),
      render,
      standIn: 'GPT-3\'s shape and the 64-GPU group are exact; the batch labels on the GPUs are illustrative.',
      belowFor: (index) => (BELOW[index] ?? []).map(fill),
    },
    toy: {
      title: 'Will it fit?',
      intro: 'Pick a model, an optimizer recipe, a ZeRO stage and a GPU. The calculator adds up training state and saved activations for one GPU, and says whether they fit.',
      mount,
    },
    math: { blocks: MATH_BLOCKS, notes: MATH_NOTES },
    facts: { framing: framing(), rows: factRows(data) },
    takeaways: TAKEAWAYS,
    links: { next: ['parallelism'], further: FURTHER },
  };
}

export const LESSON = lessonFor(null);
