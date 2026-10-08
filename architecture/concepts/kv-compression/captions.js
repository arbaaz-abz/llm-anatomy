// The nine captions of storyboard §5, verbatim (one idea each, at most two sentences and 30 words).
export const CAPTIONS = Object.freeze([
  'In GPT-3-style attention, each query head has its own stored key head and value head, a KV head. Every token stores all of them, in every layer.',
  'Multi-query attention lets all the query heads share one key head and one value head. The cache is 8 times smaller, at some cost in quality.',
  'Grouped-query attention is the middle way: query heads share within groups. Two groups here; 8 KV heads became a common choice in real models.',
  'Two heads that share keys and values still have their own queries. Their patterns differ: from "cat", head B looks mostly at "The" while head A looks at "cat".',
  'Multi-head latent attention stores one short latent vector per token. Learned matrices rebuild each head\'s own key and value from it.',
  'A small extra key carries each token\'s position, because the latent itself cannot be rotated by position. It is stored beside the latent.',
  'At decode time the rebuild folds into the query and output matrices, so attention reads the latent directly. Memory is saved; compute per step grows.',
  'MLA stores close to what two shared KV heads would, yet every head keeps its own keys and values. That is the trade it was designed for.',
  'Real models stack these choices across every layer. From GPT-3 to DeepSeek-V3, sharing and latents cut the cache per token by 67 times.',
]);
