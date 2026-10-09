// paged-attention toy view model (pure): every string and number the toy prints, from math/paging.js and the data files.
// toy.js only places these in the DOM, so the page test checks exactly what the learner reads.
import { formatBytes } from '@math/core.js';
import { kvBytesPerBlock, reserveMaxBytes } from '@math/paging.js';
import { hbmFor } from '@math/serving.js';
import { lookupFact } from '@shared/claims.js';
import {
  INITIAL_STATE, shareText, shareCell, plural, bytesPair, statusText, stepText, meanWaste,
} from './format.js';
import {
  contiguousAt, pagedAt, request, IDS, POOL_SLOTS, MAX_LEN, SYSTEM_PROMPT, LAST_STEP,
} from './numbers.js';
import { formatInt } from '@math/core.js';

export const MODEL_CHIPS = Object.freeze([
  { value: 'toy', label: 'Toy (slots only)' },
  { value: 'gpt3', label: 'GPT-3' },
  { value: 'llama', label: 'Llama-3.1-70B' },
  { value: 'v3', label: 'DeepSeek-V3' },
]);
const MODELS = Object.freeze({ gpt3: { id: 'gpt-3', name: 'GPT-3' }, llama: { id: 'llama-3.1-70b', name: 'Llama-3.1-70B' }, v3: { id: 'deepseek-v3', name: 'DeepSeek-V3' } });
export const FOLLOW_CHIPS = Object.freeze(IDS.map((id) => ({ value: id, label: id })));
export const SHARED_LABEL = 'All prompts start with the same 4-token system prompt';
export const KERNEL_NOTE = 'The toy counts slots, not kernel time, so smaller always looks better here; the 2023 paper measured that blocks under 16 could not keep the GPU busy.';
export const COMPARE_NOTE = 'Compared: the grain of allocation (one block) against the worst-case reservation (one request\'s whole context), both in bytes for one request.';

const STEPS = Array.from({ length: LAST_STEP + 1 }, (_, i) => i);

function need(data, dataset, id, key) {
  const value = lookupFact(data?.[dataset], id, key)?.value;
  if (value == null) throw new RangeError(`paged-attention: data/${dataset}.json has no ${id}.${key}`);
  return value;
}

// The two schemes at the state the toy shows (and at every step, for the averages and the start and finish steps).
export function simulate({ blockSize, step, sharedPrefix }) {
  const prefix = sharedPrefix ? SYSTEM_PROMPT : 0;
  const sims = (scheme) => STEPS.map((s) => (scheme === 'before' ? contiguousAt(s) : pagedAt(s, blockSize, prefix)));
  const all = { before: sims('before'), after: sims('after') };
  return { all, before: all.before[step], after: all.after[step], last: { before: all.before[LAST_STEP], after: all.after[LAST_STEP] } };
}

const finishOf = (live) => {
  const d = live.find((r) => r.id === 'D');
  return d.admitted == null ? null : d.admitted + request('D').output;
};

function laneCells(sim, last, all) {
  const d = last.live.find((r) => r.id === 'D');
  return {
    useful: shareCell(sim.useful, POOL_SLOTS),
    wasted: shareCell(sim.waste, POOL_SLOTS),
    free: shareCell(sim.free, POOL_SLOTS),
    dStart: stepText(d.admitted),
    dFinish: stepText(finishOf(last.live)),
    average: meanWaste(all.map((s) => s.waste), POOL_SLOTS),
  };
}

const slotsOrDash = (n, show) => (show ? formatInt(n) : '—');

function requestRows(before, after, step) {
  return IDS.map((id, i) => {
    const [b, a] = [before.live[i], after.live[i]];
    const arrives = request(id).arrives;
    return {
      id,
      before: { status: statusText(b, arrives, step), tokens: slotsOrDash(b.tokens, b.running), wasted: slotsOrDash(MAX_LEN - b.tokens, b.running) },
      after: { status: statusText(a, arrives, step), tokens: slotsOrDash(a.tokens, a.running), blocks: slotsOrDash(a.blocks, a.running), wasted: slotsOrDash(a.waste, a.running) },
    };
  });
}

function followView(after, follow, step) {
  const r = after.live.find((q) => q.id === follow);
  const status = statusText(r, request(follow).arrives, step);
  if (!r.running) return { id: follow, rows: [], empty: `${follow} holds no blocks at step ${step} (${status}).` };
  return { id: follow, rows: r.table.map((physical, logical) => ({ logical: String(logical), physical: String(physical) })), empty: null };
}

// "Scale it up": what one block and one reservation weigh for a real model, and the share of an H100 that reservation is.
export function scaleView(model, data) {
  if (model === 'toy') return null;
  const { id, name } = MODELS[model];
  const perToken = need(data, 'models', id, 'kv_bytes_per_token');
  const context = need(data, 'models', id, 'context_length');
  const blockSize = need(data, 'serving', 'vllm', 'default_block_size');
  const entry = data.hardware?.entries?.find((e) => e.id === 'h100');
  if (!entry) throw new RangeError('paged-attention: data/hardware.json has no h100');
  const hbm = hbmFor(entry);
  const reserve = reserveMaxBytes(perToken, context);
  return {
    name,
    perToken: bytesPair(perToken),
    block: { value: formatBytes(kvBytesPerBlock(perToken, blockSize)), sub: plural(blockSize, 'token') },
    reserve: { value: formatBytes(reserve), sub: `${formatInt(context)}-token context` },
    share: { value: shareText(reserve, hbm.bytes), sub: `of an H100 (${formatBytes(hbm.bytes)} ${hbm.basis})` },
  };
}

export function toyView(state, data) {
  const { step, follow } = state;
  const { all, before, after, last } = simulate(state);
  const blocks = (sim, poolBlocks, noun) => `${sim} of ${poolBlocks} ${noun}`;
  return {
    step,
    before: { ...laneCells(before, last.before, all.before), blocks: blocks(before.live.filter((r) => r.running).length, POOL_SLOTS / MAX_LEN, 'strips') },
    after: { ...laneCells(after, last.after, all.after), blocks: blocks(after.blocksUsed, after.poolBlocks, 'blocks') },
    saved: { value: formatInt(after.blocksSaved), sub: plural(after.blocksSaved, 'block') },
    requests: requestRows(before, after, step),
    follow: followView(after, follow, step),
    scale: scaleView(state.model, data),
    sims: { before, after },
  };
}

// The three try-this items: the storyboard's prompts with every number computed from the same simulators.
export function tryThis() {
  const run = (blockSize, sharedPrefix = false) => simulate({ blockSize, step: 0, sharedPrefix });
  const four = run(4);
  const peak = Math.max(...four.all.after.map((s) => s.waste));
  const dAfter = four.last.after.live.find((r) => r.id === 'D');
  const dBefore = four.last.before.live.find((r) => r.id === 'D');
  const sixteen = run(16);
  const two = run(2);
  const fine = pagedAt(1, 4, SYSTEM_PROMPT);
  const plain = pagedAt(1, 4, 0);
  const coarse = pagedAt(1, 8, SYSTEM_PROMPT);
  const entries = (n, b) => Math.ceil(n / b);
  return [
    {
      text: `Keep Block size at 4 and scrub Time from 0 to 6. Before: ${shareText(four.all.before[0].waste, POOL_SLOTS)} wasted at step 0, D waits until step ${dBefore.admitted} and finishes at step ${dBefore.admitted + request('D').output}. After: waste peaks at ${shareText(peak, POOL_SLOTS)} and D starts at step ${dAfter.admitted} and finishes at step ${dAfter.admitted + request('D').output}.`,
      insight: 'reserving for the worst case, not the tokens themselves, is what wastes memory and shrinks the batch.',
      rest: '',
    },
    {
      text: `Set Block size to 16: the After lane becomes identical to the Before lane (${shareText(sixteen.all.after[0].waste, POOL_SLOTS)} at step 0, D waits, ${meanWaste(sixteen.all.after.map((s) => s.waste), POOL_SLOTS)} averaged). Set it to 2: averaged waste ${meanWaste(two.all.after.map((s) => s.waste), POOL_SLOTS)}, but a 12-token request now needs ${entries(12, 2)} table entries and ${entries(12, 2)} scattered reads instead of ${entries(12, 4)}.`,
      insight: 'block size is a dial between internal waste and table or kernel overhead, and "one strip per request" is just block size equal to the maximum length.',
      rest: ' vLLM\'s 16 is a measured middle.',
    },
    {
      text: `Turn on "${SHARED_LABEL}" at Time 1 with Block size 4: the four requests share one physical block, ${fine.blocksUsed} blocks in use instead of ${plain.blocksUsed}. Switch Block size to 8: ${coarse.blocksSaved} blocks saved, because the ${SYSTEM_PROMPT} shared tokens never fill a whole block.`,
      insight: 'sharing works on whole blocks, so block size also sets the granularity of the prefix cache.',
      rest: ' The next lesson, [[prefix-caching]], builds on this.',
    },
  ];
}

export { INITIAL_STATE };
