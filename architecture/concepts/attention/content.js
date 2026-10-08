// The attention lesson (docs/storyboards/attention.md): every word on the page, plus the stage and the toy.
// No DOM at import time, so the node page test can import it.
import { render } from './frames.js';
import { mount } from './toy.js';
import { workedRowTex } from './format.js';
import { CAPTIONS } from './captions.js';

export { CAPTIONS };

// The toy opens where the animation ends: query "sat", head A, divisor √d_head = 2, mask on.
export const TOY_DEFAULT = Object.freeze({ query: 2, divisor: 2, causal: true, head: 'A' });

const intuition = [
  'Each token in an attention layer makes three small vectors from its current vector (in the first layer, its embedding): a query (what it is looking for), a key (what other queries match against) and a value (what it hands over if chosen). The three come from three learned matrices, W_Q, W_K and W_V, multiplied with the same input vector. Each has d_head numbers; 4 in this toy.',
  'Attention is one scoring pass and one blend. A token\'s query is dotted with every key, giving one score per token. The scores are divided by √d_head, the future is masked out, and softmax turns the row into weights that sum to 1. The token\'s output is the weighted sum of the values. Five steps, and you can do them with a calculator; this page makes you do exactly that for one row.',
  'Why forbid the future? The model learns by predicting each next word, and in training the whole sentence goes in at once. If "sat" could see "down", it would copy the answer instead of learning to predict it. When the model is generating, "down" doesn\'t exist yet, so the mask also keeps training and generation the same.',
  'A head is one such pattern over the tokens. A layer runs several heads at once, each with its own W_Q, W_K, W_V, like looking at the same sentence through several lenses. Their outputs are concatenated and mixed by one more matrix, W_O, and the result is added back into each token\'s vector (the residual stream), where the next block starts. Most later lessons change what feeds these five steps (RoPE), what is stored between them (KV cache, GQA, MLA) or which keys a query reads (sliding windows, sparse attention). Two small additions sit around the softmax (QK-norm and a sink logit), and some 2026 layers swap softmax attention for linear attention altogether ([[long-context-attention]]).',
];

const math = {
  blocks: [
    {
      tex: '\\htmlClass{hl-q}{Q} = X\\,W_Q,\\qquad \\htmlClass{hl-k}{K} = X\\,W_K,\\qquad \\htmlClass{hl-v}{V} = X\\,W_V \\qquad X \\in \\mathbb{R}^{n \\times d_{\\text{model}}},\\; W_Q, W_K, W_V \\in \\mathbb{R}^{d_{\\text{model}} \\times d_{\\text{head}}}',
      note: 'X [n × d_model] (4 × 8); W_Q, W_K, W_V [d_model × d_head] (8 × 4) per head; Q, K, V [n × d_head] (4 × 4).',
    },
    {
      tex: '\\htmlClass{hl-s}{S} = \\frac{\\htmlClass{hl-q}{Q}\\,\\htmlClass{hl-k}{K}^{\\top}}{\\sqrt{d_{\\text{head}}}} + M, \\qquad M_{ij} = \\begin{cases} 0 & j \\le i \\\\ -\\infty & j > i \\end{cases}',
      note: 'S [n × n] (4 × 4): rows are queries, columns are keys. M is the causal mask.',
    },
    {
      tex: '\\htmlClass{hl-a}{A}_{ij} = \\frac{e^{S_{ij}}}{\\sum_{k=1}^{n} e^{S_{ik}}}, \\qquad \\htmlClass{hl-o}{O} = \\htmlClass{hl-a}{A}\\,\\htmlClass{hl-v}{V}',
      note: 'A [n × n] (4 × 4), every row sums to 1; O [n × d_head] (4 × 4).',
    },
    { tex: workedRowTex(TOY_DEFAULT), note: 'Query "sat" (row 3), head A, divided by √4 = 2, mask on: the toy\'s starting state.' },
    {
      tex: '\\text{head}_h = \\operatorname{Attention}\\!\\left(X W_Q^{h},\\, X W_K^{h},\\, X W_V^{h}\\right), \\qquad \\operatorname{MultiHead}(X) = \\big[\\text{head}_1 \\,\\|\\, \\cdots \\,\\|\\, \\text{head}_H\\big]\\, W_O',
      note: 'concat [n × H·d_head] (4 × 8); W_O [H·d_head × d_model] (8 × 8).',
    },
  ],
};

const facts = {
  framing: 'These five steps are the 2017 formula, and they are unchanged in every layer that still uses softmax attention. Real models often let several query heads share one set of keys and values to save memory ([[kv-compression]]). What 2026 models change is (1) how many keys and values each layer stores per token (→ [[kv-compression]]), (2) which past tokens a query reads, because every query scoring every key gives a grid of n × n cells, so doubling the sequence quadruples the work (→ [[long-context-attention]]), (3) two small additions around the softmax: normalizing Q and K before the dot product, and a learned \'sink\' logit that gives the row somewhere to put weight when nothing is relevant, since the row must still sum to 1, and (4) in some models, swapping most softmax-attention layers for linear attention (→ [[long-context-attention]]).',
  rows: [
    { claim: 'GPT-3 ({gpt-3.release_date|year}): full multi-head attention, {gpt-3.n_heads} heads per layer, each {gpt-3.head_dim} numbers wide, every head with its own keys and values; {gpt-3.layers} layers.' },
    { claim: 'gpt-oss-120b ({gpt-oss-120b.release_date|year}): {gpt-oss-120b.n_heads} query heads share {gpt-oss-120b.n_kv_heads} sets of keys and values, each {gpt-oss-120b.head_dim} numbers wide; layers alternate full attention with a {gpt-oss-120b.window}-token window; a learned sink per head.' },
    { claim: 'DeepSeek-V4-Pro ({deepseek-v4-pro.release_date|year}): {deepseek-v4-pro.n_kv_heads} set of keys and values per layer, {deepseek-v4-pro.head_dim} numbers wide, shared by all query heads, plus compression and sparsity ([[long-context-attention]]); Q and K normalized; a learned sink.' },
    { claim: 'Kimi K3 ({kimi-k3.release_date|year}): {kimi-k3.n_heads} heads; most of its {kimi-k3.layers} layers are linear-attention layers ([[long-context-attention]]), and only the rest run this softmax attention.' },
    { claim: 'MiniMax-M3 ({minimax-m3.release_date|year}): {minimax-m3.n_heads} query heads, {minimax-m3.n_kv_heads} sets of keys and values; after its first few layers, each query reads only the {minimax-m3.sparse_top_blocks} most relevant blocks of {minimax-m3.sparse_block} past tokens.' },
    { claim: 'Qwen3.8-2.4T-A95B ({qwen3.8.release_date|year}): {qwen3.8.n_heads} query heads, {qwen3.8.n_kv_heads} sets of keys and values; most of its layers are linear-attention layers, and only the rest are softmax attention.' },
  ],
  prose: [
    'In production kernels the 4 × 4 grid on this page is not written to GPU memory at full size: FlashAttention computes exactly these numbers tile by tile in on-chip SRAM with an online softmax.',
  ],
};

export const LESSON = Object.freeze({
  slug: 'attention',
  hook: 'When a model reads "The cat sat down", how does "sat" know to look at "cat", and why can it never look at "down"?',
  intuition,
  intuitionNote: 'We treat each word as one token here; how text is cut into tokens is covered in [[pretraining]].',
  animation: {
    label: 'One row of attention, then two heads',
    steps: CAPTIONS.map((caption) => ({ caption })),
    render,
  },
  toy: {
    title: 'Compute one row yourself.',
    intro: 'Q, K and V here are hand-picked so every number checks by hand; a real model computes them from each token\'s vector with W_Q, W_K, W_V.',
    mount,
  },
  math,
  facts,
  takeaways: [
    'One row of attention is five hand-sized steps: dot the query with every key, divide by √d_head, set the future to −∞, softmax, then blend the values by those weights.',
    'Attention weights are not parameters. They are recomputed from Q and K for every input; what training learns is W_Q, W_K, W_V and W_O.',
    'A head is one pattern; a layer runs several patterns in parallel over the same tokens and joins them with W_O. Later lessons change what feeds these steps (RoPE), what is stored between them (KV cache, GQA, MLA) or which keys are read (sparse, sliding window). Some 2026 layers replace these steps with linear attention ([[long-context-attention]]).',
  ],
  links: {
    next: ['rope', 'kv-cache', 'multimodal'],
    further: [
      { title: '3Blue1Brown, Attention in transformers', href: 'https://www.3blue1brown.com/lessons/attention', note: 'the intuition on video' },
      { title: 'Transformer Explainer', href: 'https://poloclub.github.io/transformer-explainer/', note: 'run real GPT-2 on your own sentence' },
      { title: 'Jay Alammar, The Illustrated Transformer', href: 'https://jalammar.github.io/illustrated-transformer/', note: 'the canonical static diagrams' },
    ],
  },
});
