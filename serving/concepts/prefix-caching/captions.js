// prefix-caching storyboard §5 captions, verbatim (one per frame). content.js puts them on the steps; frames.js labels each
// stage with them. Frame 3's, 5's, 8's and 9's numbers are pinned to the simulation (and frame 8's DeepSeek figure to the data) by the page test.
export const CAPTIONS = Object.freeze([
  'Request A is done, but its four full blocks of keys and values stay in GPU memory instead of being freed. Each one is labeled by its tokens.',
  'A block\'s label is its own tokens plus everything before it. Keys and values depend on earlier tokens, so the same words after another start are a different block.',
  'B starts with the same system prompt. Its first two blocks match, so prefill skips 8 tokens and computes only the 5 that differ.',
  'B\'s last prompt block holds only one token, so it is not cached yet. Engines share only full blocks; this one becomes cacheable once B\'s answer fills it.',
  'C is A\'s next turn, so its prompt resends the whole conversation so far. Sixteen of its twenty prompt tokens hit, including A\'s own answer.',
  'D repeats A\'s question word for word, but its first block differs. A match has to start at the first token, so D reuses nothing.',
  'When D needs four blocks, the least recently used ones are evicted first: B\'s and C\'s own. The shared system prompt was touched most recently, so it survives.',
  'Across the four requests, 24 of 57 prompt tokens came from the cache, a 42.1% hit rate. DeepSeek measured 56.3% across its production traffic in 2025.',
  'With several replicas, the router decides whether a hit is even possible. Sent where A ran, C reuses 16 tokens; sent anywhere else, only 8.',
  'Newer engines move evicted blocks to CPU memory or storage instead of dropping them. Reloading a long prefix is cheaper than computing it again.',
  'Providers pass the saving on: at Anthropic a cached input token costs a tenth or less of a normal one. Cache writes cost a quarter more, to hold it.',
]);
