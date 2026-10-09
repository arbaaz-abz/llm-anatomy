// speculative-decoding's dated text (pure, no DOM): the §8 rows and framing, the frame-10 rows drawn on the stage and the
// notes printed under it. Dated numbers are {entry.key|format} placeholders filled from data/serving.json (prefix sv:) and
// data/models.json; ratios the page computes come from specdec.js. Rows keep their placeholders so the scaffold adds each
// row's source link and "reported" flag.
import { formatRatio } from '@math/core.js';
import { batchSpeedup } from '@math/specdec.js';
import { fillText, lookupFact } from '@shared/claims.js';
import { ALPHA, C, K, MODEL } from './numbers.js';
import { ratioText } from './format.js';

const BATCH_COMPARED = 64; // EAGLE-3's measured batch (data: eagle-3.batch64_throughput_gain)

export const FRAMING = 'In 2026 open-model serving the drafter is usually part of the model, and every number below was measured by its authors on their own setup, not by this page\'s toy.';

export function factRows() {
  return [
    { claim: 'vLLM: speculative decoding "preserves the verifier model\'s output distribution exactly" via rejection sampling.{sv:vllm-specdec.exact_distribution|cite}' },
    { claim: 'DeepSeek-V3 ({deepseek-v3.release_date|year} report): its multi-token-prediction (MTP) module predicts one extra token; the second token is accepted {sv:deepseek-v3-mtp.acceptance_pct}% of the time, about {sv:deepseek-v3-mtp.tps_gain|raw}× tokens per second.' },
    { claim: 'EAGLE-3 ({sv:eagle-3.release_date|year}): up to {sv:eagle-3.peak_speedup|raw}× over plain decoding at small batch, about {sv:eagle-3.gain_vs_eagle2|raw}× better than EAGLE-2 (the abstract\'s words); {sv:eagle-3.batch64_throughput_gain|raw}× throughput at batch 64 in SGLang.' },
    { claim: 'P-EAGLE parallel drafting (vLLM): the drafter emits all k tokens in one forward pass; up to {sv:p-eagle.gain_vs_eagle3|raw}× over EAGLE-3 on B200.' },
    { claim: 'vLLM: {sv:vllm-specdec.parallel_drafters_note|raw}.' },
    { claim: 'MTP raised per-user throughput {sv:lmsys-gb300-longctx.mtp_per_user_gain_pct}% for DeepSeek-R1 on GB300 NVL72 at 128K tokens in and 8K out, while keeping peak system throughput (LMSYS).' },
    { claim: '{sv:vllm-specdec.mainstream_note|raw}.' },
  ];
}

const value = (data, set, id, key) => lookupFact(data?.[set], id, key)?.value ?? null;
const ratio = (v) => (v == null ? '—' : formatRatio(v));
const range = (v) => (Array.isArray(v) ? `${v[0]}–${v[1]}` : '—');
const year = (v) => (typeof v === 'string' ? v.slice(0, 4) : '—');

// The three rows of frame 10: system, setup, result. "—" where the data lacks the figure.
export function stageText(data) {
  const sv = (id, key) => value(data, 'serving', id, key);
  const v3Year = year(value(data, 'models', 'deepseek-v3', 'release_date'));
  return Object.freeze({
    rows: Object.freeze([
      { name: 'MTP head', setup: `DeepSeek-V3 (${v3Year} report)`, result: `second token accepted ${range(sv('deepseek-v3-mtp', 'acceptance_pct'))}%, about ${ratio(sv('deepseek-v3-mtp', 'tps_gain'))} tokens/s` },
      { name: 'EAGLE-3 head', setup: `EAGLE-3 (${year(sv('eagle-3', 'release_date'))}, SGLang)`, result: `up to ${ratio(sv('eagle-3', 'peak_speedup'))} at small batch, ${ratio(sv('eagle-3', 'batch64_throughput_gain'))} at batch ${BATCH_COMPARED}` },
      { name: 'P-EAGLE', setup: 'parallel drafter (vLLM)', result: `up to ${ratio(sv('p-eagle', 'gain_vs_eagle3'))} over EAGLE-3 on B200` },
    ]),
  });
}

const ours = ratioText(batchSpeedup({ alpha: ALPHA, k: K, c: C, batch: BATCH_COMPARED, model: MODEL }).speedup);

// Page text under the stage, one list per frame. Frame 9's comparison fills EAGLE-3's measured figure from the data.
export const BELOW = Object.freeze([
  ['The target is the running example of [[prefill-decode]]: Llama-3.1-70B with FP8 weights on one H200, each user holding 1,024 tokens of context.'],
  [],
  [],
  [],
  [],
  ['The four words and both probability rows are stand-ins. How a real model turns scores into probabilities is [[sampling]].'],
  [],
  [],
  [
    `EAGLE-3's measured {sv:eagle-3.batch64_throughput_gain|raw}× at batch ${BATCH_COMPARED} is lower than this toy's ${ours}; its setup had less KV to read per step.`,
    'The drafter costs the same fraction of a target step at every batch size, which flatters large batches: a real curve falls at least this fast.',
  ],
  ['Closed providers do not publish their numbers.'],
]);

export const belowFor = (data) => (index) => (BELOW[index] ?? []).map((t) => fillText(t, data));
