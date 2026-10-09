// The nine captions of storyboard §5, verbatim (one idea each, at most two sentences and 30 words).
export const CAPTIONS = Object.freeze([
  'Three requests arrive. In attention layers, each token a request has seen keeps its K and V in GPU memory until that request finishes.',
  'Before paging, each request reserved a contiguous strip for the longest answer it might give. Three requests fill the whole pool before a single token is generated.',
  'D arrives and must wait: the pool is 100% reserved and 39.6% empty. In decode a bigger batch means more tokens per second, so waiting costs throughput.',
  'PagedAttention cuts KV memory into blocks of 4 tokens (vLLM uses 16) and hands a block out only when a request fills its current one.',
  'Each request keeps a block table saying where each of its blocks sits in the pool. A\'s third block landed at 7; the kernel follows the table to find it.',
  'A request gets a new block only when its current one fills, from anywhere in the pool. That leaves room for D, so D starts now.',
  'A finished request returns all its blocks at once, and any request can use them right away. Any free block fits anyone, so no gap is ever too small.',
  'What if D asks for two answers? Both point at the same prompt blocks, and each block counts its pointers: ref 2, one stored copy of K and V.',
  'A shared block still being filled is copied the first time a sequence writes into it: copy-on-write. Full shared blocks are never copied; prefix caching builds on that.',
]);
