// kv-compression storyboard §5 captions and the toy's "Check my work" text for its default state. Imported by the page test and the e2e spec.
export const CAPTIONS = [
  "In GPT-3-style attention, each query head has its own stored key head and value head, a KV head. Every token stores all of them, in every layer.",
  "Multi-query attention lets all the query heads share one key head and one value head. The cache is 8 times smaller, at some cost in quality.",
  "Grouped-query attention is the middle way: query heads share within groups. Two groups here; 8 KV heads became a common choice in real models.",
  "Two heads that share keys and values still have their own queries. Their patterns differ: from \"cat\", head B looks mostly at \"The\" while head A looks at \"cat\".",
  "Multi-head latent attention stores one short latent vector per token. Learned matrices rebuild each head's own key and value from it.",
  "A small extra key carries each token's position, because the latent itself cannot be rotated by position. It is stored beside the latent.",
  "At decode time the rebuild folds into the query and output matrices, so attention reads the latent directly. Memory is saved; compute per step grows.",
  "MLA stores close to what two shared KV heads would, yet every head keeps its own keys and values. That is the trade it was designed for.",
  "Real models stack these choices across every layer. From GPT-3 to DeepSeek-V3, sharing and latents cut the cache per token by 67 times.",
];

export const CHECK_WORK = [
  'numbers per token per layer: 2 (K and V) × 8 KV heads × 4 numbers = 64',
  'bytes per token: 64 numbers × 1 layer × 2 bytes = 128 B',
].join('\n');

// The frame-4 pattern, row by row (4 × 4, masked cells print 0): head A, head B reading A's keys, head B with its own keys.
export const PATTERN_A = ['1', '0', '0', '0', '0.321', '0.679', '0', '0', '0.095', '0.703', '0.202', '0', '0.114', '0.656', '0.129', '0.101'];
export const PATTERN_B_SHARED = ['1', '0', '0', '0', '0.731', '0.269', '0', '0', '0.119', '0.685', '0.196', '0', '0.196', '0.153', '0.366', '0.285'];
export const PATTERN_B_OWN = ['1', '0', '0', '0', '0.798', '0.202', '0', '0', '0.168', '0.664', '0.168', '0', '0.129', '0.146', '0.578', '0.146'];
