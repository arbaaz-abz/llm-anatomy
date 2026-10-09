// batching: the complete lesson (storyboard docs/storyboards/batching.md §3, §5-§10). Pure, no DOM.
// lessonFor(data) fills the dated text from data/serving.json; LESSON is the same lesson with no data (validation, tests).
import { render } from './frames.js';
import { mount } from './toy.js';
import { CAPTIONS } from './captions.js';
import { fillText } from '@shared/claims.js';
import { FRAMING, factRows, INTUITION_FACTS } from './facts.js';
import { TOY_INTRO } from './toy-view.js';

const SLUG = 'batching';

const HOOK = 'If one read of the weights can serve dozens of users at once, why did early servers keep making new requests wait while seats sat empty?';

const INTUITION = Object.freeze([
  `In [[prefill-decode]] you saw that a decode step costs about the same whether it serves one user or dozens: the GPU reads the weights once and every user in the batch gets a token. So the server's job is to keep that batch full. The simplest way, static batching, gathers a group of requests, runs them together, and starts the next group only when the whole group is done. Answers have different lengths, though. When the short ones finish, their seats stay held and empty until the longest one is done, and requests that arrived in the meantime wait outside. Static batches also pad: each prompt is stretched to the longest one's length (adding a 100-token prompt to 8 running requests ${INTUITION_FACTS.pad}).`,
  `${INTUITION_FACTS.orca}, fixes this by deciding the batch one step at a time. Before every step the scheduler drops finished requests and seats waiting ones, so a seat is reused the step after it frees, and sequences of different lengths sit side by side without padding (ragged batching). The major engines (vLLM, SGLang, TensorRT-LLM) all schedule this way; vLLM's core loop is schedule, run the model, postprocess, repeated every step.`,
  'Mixing requests in one step has a catch. A new request\'s whole prompt is prefilled in the step it joins, and a long prompt makes that step long. Everyone else\'s next token waits for it. Chunked prefill gives each step a token budget: running decodes go first, and the prompt fills the rest of the budget a slice at a time, so no step gets much longer than the budget allows. The price is that the long prompt\'s own first token comes a little later.',
  `The scheduler has a few more knobs: a cap on seats, the token budget, priorities, and preemption, which pauses a request and recomputes it later when memory runs out (${INTUITION_FACTS.preempt}; older versions could also swap to CPU memory). The price of continuous batching is a scheduling pass before every step and the chance that an admitted request runs out of memory mid-reply, which is what preemption is for. Memory is the knob this page holds fixed. In 2023 engines the seat cap was really a memory limit, because each seat reserved KV for the longest possible answer; you'll see how paging lifted it in [[paged-attention]].`,
]);

const tex = String.raw;
const MATH_BLOCKS = Object.freeze([
  { tex: tex`\text{static: start}_{\text{next batch}} = \max_{r \in \text{batch}} \left(a_r + o_r\right) + 1 \qquad \text{continuous: seat free at step } a_r + o_r + 1` },
  { tex: tex`\text{seat utilization} = \frac{\sum_r (o_r + 1)}{\text{seats}\times\text{steps}} \qquad \text{tokens per step} = \frac{\sum_r o_r}{\text{steps}} \qquad \text{pad waste} = (n-1)(B-1)` },
  { tex: tex`\text{chunked prefill: } \text{slice}_s = \min\!\left(\text{prompt left},\ \text{budget} - \#\text{decodes}_s\right), \qquad t_s = \text{stepTime}(\#\text{decodes}_s + \text{slice}_s)` },
]);

const MATH_NOTES = Object.freeze([
  'Shapes: none. a_r is the step request r is admitted, o_r its output tokens (its decode steps), n the new prompt\'s length and B the number of running requests. The equations carry no color links to the stage.',
]);

const TAKEAWAYS = Object.freeze([
  'Static batching holds seats until the longest member is done and makes newcomers wait; continuous batching re-forms the batch every step, so a freed seat is reused at once (frames 2–6, try-this 1 and 2).',
  'Prefill and decode can share a step, so a long prompt can stall everyone\'s stream; chunked prefill caps each step\'s tokens and bounds the stall at a small cost to that prompt\'s TTFT (frames 7–9, try-this 4).',
  'A seat only helps if someone is waiting, and in 2023 engines each seat reserved KV for the longest answer, which kept seats scarce; [[paged-attention]] lifts that limit (frame 10, try-this 3).',
]);

const FURTHER = Object.freeze([
  { title: 'Continuous batching from first principles (Hugging Face)', href: 'https://huggingface.co/blog/continuous_batching', note: 'the padded-versus-ragged batching walkthrough this lesson borrows its pad-waste example from' },
  { title: 'Interactive vLLM guide (Ashwin Giridharan)', href: 'https://ashwing.github.io/vllm-guide/', note: 'a token-by-token scheduler walkthrough' },
  { title: 'LLM Inference Explained (Lynskey)', href: 'https://llm-inference-explained.vercel.app', note: 'serving from the request down to the GPU' },
]);

// Page text under the stage, one list per frame: hand-offs to the lessons that continue each idea.
export const BELOW = Object.freeze({
  0: ['The step times on this page come from the running example in [[prefill-decode]]: Llama-3.1-70B, FP8 weights, one H200.'],
  6: ['Why a shared step costs about the same as a decode step: see [[prefill-decode]].'],
  8: ['Prefill and decode can also move to separate GPU pools: see [[disaggregation]].'],
  9: ['Why memory sets the number of seats, and how paging lifts it: see [[paged-attention]].'],
});

export function lessonFor(data) {
  const fill = (text) => fillText(text, data);
  return {
    slug: SLUG,
    hook: HOOK,
    intuition: INTUITION.map(fill),
    intuitionNote: 'Routing requests by what their KV caches already hold needs prefix caching: see [[prefix-caching]].',
    animation: {
      label: 'Continuous batching: four requests through static batching, continuous batching, a long prompt, chunked prefill and a fourth seat, in ten steps',
      steps: CAPTIONS.map((caption) => ({ caption })),
      render,
      standIn: 'The four requests, the three seats and their lengths are hand-picked for this lesson; step times come from the running example in [[prefill-decode]].',
      belowFor: (index) => (BELOW[index] ?? []).map(fill),
    },
    toy: { title: 'Seat timeline', intro: TOY_INTRO, mount },
    math: { blocks: MATH_BLOCKS, notes: MATH_NOTES },
    facts: { framing: FRAMING, rows: factRows(data) },
    takeaways: TAKEAWAYS,
    links: { next: ['paged-attention', 'disaggregation'], further: FURTHER },
  };
}

export const LESSON = lessonFor(null);
