// The nine captions of storyboard §5, verbatim (one idea each, at most two sentences and 30 words).
export const CAPTIONS = Object.freeze([
  'The model just chose "on". One decode step runs it through every block, and in each attention layer it must look at all five positions.',
  'Without a cache, every position is computed again for each new token. Rows 1 to 4 come out exactly as before, because a position never looks ahead.',
  'With a cache, only the new token is computed; the past\'s stored keys and values are read back. Work shrinks, but the read still covers every position.',
  'The prompt goes in all at once and fills the cache in one pass: that is prefill. After it, each decode step adds one row.',
  'Over a four-token reply, the cache computes 7 positions instead of 22. The stored rows are still read every step, so reads grow with the length.',
  'Each position stores one key row and one value row per head in every attention layer. In this toy that is 32 numbers, 64 bytes per token.',
  'GPT-3 stored 4.72 MB for every token, because all 96 heads in all 96 layers kept their own keys and values. Its whole 2,048-token context needed 9.66 GB.',
  'Every conversation has its own cache. One 128K-token Llama-3.1-70B conversation needs 42.9 GB, so a single 80 GB GPU cannot hold two.',
  'By 2026 the biggest open models keep a few kB per token, hundreds of times less than GPT-3. That is what makes a million-token context affordable.',
]);
