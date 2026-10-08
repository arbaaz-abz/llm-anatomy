// long-context-attention storyboard §5 captions, verbatim. Imported by the page test and the e2e spec.
export const CAPTIONS = [
  "Full attention has two costs that grow with length. Each new token reads every stored key, and the cache keeps every token.",
  "A sliding-window layer reads only its last 4 tokens, so the cache stops growing at 4 entries. Anything older is invisible to this layer.",
  "Models interleave window layers with full ones. The full layers still read everything and carry old information forward through the stream.",
  "Softmax must spend all its weight somewhere, even when nothing in the window matters. A learned sink logit gives it a place to put that weight.",
  "Sparse attention runs a cheap indexer over every stored key, then lets the real attention read only the top few. Here token 16 reads 4 of 16.",
  "Reading fewer keys does not mean storing fewer. Every token stays in the cache, because the indexer must score all of them each step.",
  "Compressed attention merges every 4 tokens' keys and values into one entry, then reads the best few plus a short window. This cuts both reading and memory.",
  "Linear attention keeps a fixed 4 by 4 state instead of a cache. Each token adds its value-times-key grid; a query reads the state with one multiplication.",
  "Hybrid models make three of every four layers linear and keep one full layer for exact recall. Only the full layers grow a cache.",
  "2026 models split into two routes to a million tokens: sparse or compressed softmax attention, or linear-attention hybrids. Both also train in stages up to that length.",
];
