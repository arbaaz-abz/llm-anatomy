// batching dated text (pure, no DOM): the §8 rows and framing, filled from data/serving.json. Rows keep their placeholders so the
// scaffold adds each row's source link and "reported" chip; ratios are formatted with formatRatio (X-3), never typed.
import { formatRatio } from '@math/core.js';
import { lookupFact } from '@shared/claims.js';

const fact = (data, id, key) => lookupFact(data?.serving, id, key)?.value ?? null;
const ratio = (data, key) => {
  const value = fact(data, 'sarathi-serve', key);
  return value == null ? '—' : formatRatio(value);
};

export const FRAMING = 'Every engine named below schedules this way today. The numbers on the stage and in the toy are simulations on four hand-picked requests; these are the sourced facts around them.';

// The six rows of storyboard §8, in order.
export function factRows(data) {
  return [
    { claim: 'Padded static batching: adding a 100-token prompt to 8 running requests wastes {sv:hf-continuous-batching.pad_waste_example} pad tokens (Hugging Face\'s worked example).' },
    { claim: 'Continuous (iteration-level) batching: Orca, {sv:orca.venue}; finished requests leave and waiting ones join at every step.' },
    { claim: 'vLLM\'s engine loop, every step: {sv:vllm.engine_loop}.' },
    {
      claim: `Chunked prefill (Sarathi-Serve, {sv:sarathi-serve.venue}): up to ${ratio(data, 'gain_mistral7b_a100')} serving capacity within its latency targets for Mistral-7B on one A100, `
        + `and up to ${ratio(data, 'gain_falcon180b_8xa100')} for Falcon-180B on 8 A100s, compared with Orca and vLLM.{sv:sarathi-serve.gain_mistral7b_a100|cite}{sv:sarathi-serve.gain_falcon180b_8xa100|cite}`,
    },
    { claim: 'Engine knobs: {sv:vllm.scheduler_knobs}. vLLM V1 preempts by {sv:vllm.v1_preemption}.' },
    { claim: 'On GB200 with DeepSeek-R1, vLLM reports: {sv:vllm-gb200-dsr1.note_chunking}.' },
  ];
}

// Dated sentences inside prose: filled with fillText, so a missing fact prints a dash and the page test catches it.
export const INTUITION_FACTS = Object.freeze({
  pad: 'wastes {sv:hf-continuous-batching.pad_waste_example} pad tokens in Hugging Face\'s worked example',
  orca: 'Continuous batching, introduced by Orca at {sv:orca.venue}',
  preempt: 'vLLM V1 preempts by {sv:vllm.v1_preemption}',
});
