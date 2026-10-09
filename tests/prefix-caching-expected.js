// prefix-caching storyboard §5 captions, §6 "Check my work" and §6 try-this text, verbatim (numbers as the simulation and the
// data fill them). Imported by the page test and the e2e spec; page code never imports from tests/.
export const CAPTIONS = [
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
];

export const CHECK_WORK = [
  'h = 24 hit tokens ÷ 57 prompt tokens = 42.1%',
  'price per M = (1 − h) · base · write + h · base · read',
  '            = 0.579 · $2.00 · 1.25 + 0.421 · $2.00 · 0.1 = $1.45 + $0.084 = $1.53',
].join('\n');

// The toy's default readouts (storyboard §5 frame 8, §6): per request [prompt, from cache, computed, blocks, evicted].
export const DEFAULT_ROWS = [
  ['A', '12', '0', '12', '0, 1, 2, 3', 'none'],
  ['B', '13', '8', '5', '0, 1, 4, 5', 'none'],
  ['C', '20', '16', '4', '0, 1, 2, 3, 6, 7', 'none'],
  ['D', '12', '0', '12', '5, 4, 7, 6', '? Yes , fish · Do you like fish · It was warm . · Why down there ?'],
];

// Storyboard §6 "Try this": { prompt, insight, rest }; `x` marks code text; vLLM's block size is filled from the data.
export const TRY_THIS = [
  {
    prompt: 'Slide Block size from 4 to 16, then to 1 (the order matters: the effect is not monotone). From 4 to 16 the hit rate falls from 42.1% to 28.1%: B\'s 8-token system prompt no longer fills a block (0 hits), while C still reuses 16 tokens because A\'s whole first turn filled exactly one block. Now slide to 1: 47.4%, because D reuses `You are a`.',
    insight: 'block size sets the grain of reuse.',
    rest: ' Smaller blocks catch more, at the cost of bigger tables and smaller memory reads (see [[paged-attention]]); vLLM uses 16, SGLang matches token by token.',
  },
  {
    prompt: 'Keep Pool size at 8 and turn on B again after D. D evicted B\'s and C\'s private blocks, but B still reuses its 8-token system prompt; this time A\'s turn (`Where did you sit`, `The cat sat down`) is evicted to make room. Switch Pool size to 6: C already loses B\'s 2 blocks, and D evicts A\'s turn too.',
    insight: 'least-recently-used eviction keeps whatever keeps getting reused,',
    rest: ' which is usually the shared beginning.',
  },
  {
    prompt: 'Pick Anthropic Sonnet 5.5 with Charge the cache write on. Hit rate for pricing at 0%: $2.50 per M, more than not caching at all ($2.00). This toy\'s 42.1%: $1.53. DeepSeek\'s 56.3%: $1.21. Switch Price to DeepSeek V4-Pro: $0.66 at 0%, $0.30 at 56.3%.',
    insight: 'caching pays only when hits come back.',
    rest: ' A write costs extra because the provider must hold your KV for minutes; a hit is cheap because it skips prefill math.',
  },
];
