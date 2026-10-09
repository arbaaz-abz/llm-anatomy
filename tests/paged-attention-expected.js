// Expected strings for paged-attention, exported once: the nine captions (storyboard §5) and the three try-this items.
export { CAPTIONS } from '../serving/concepts/paged-attention/captions.js';

export const TRY_THIS = Object.freeze([
  'Keep Block size at 4 and scrub Time from 0 to 6. Before: 52.1% wasted at step 0, D waits until step 3 and finishes at step 6. After: waste peaks at 16.7% and D starts at step 1 and finishes at step 4. → Insight: reserving for the worst case, not the tokens themselves, is what wastes memory and shrinks the batch.',
  'Set Block size to 16: the After lane becomes identical to the Before lane (52.1% at step 0, D waits, 34.2% averaged). Set it to 2: averaged waste 2.7%, but a 12-token request now needs 6 table entries and 6 scattered reads instead of 3. → Insight: block size is a dial between internal waste and table or kernel overhead, and "one strip per request" is just block size equal to the maximum length.',
  'Turn on "All prompts start with the same 4-token system prompt" at Time 1 with Block size 4: the four requests share one physical block, 7 blocks in use instead of 10. Switch Block size to 8: 0 blocks saved, because the 4 shared tokens never fill a whole block. → Insight: sharing works on whole blocks, so block size also sets the granularity of the prefix cache.',
]);
