# 01 - Architecture brief (state of the art as of 2026-10-07)

Confidence tags: **[C]** CONFIRMED (primary source: tech report, official model card, HF config.json), **[R]** REPORTED (secondary only), **[U]** UNVERIFIED / conflicting. Untagged items are long-settled background (2017-2025) that I did not re-check this session. "Derived" means I computed it from confirmed config values; the arithmetic is shown so builders can re-run it.

Note on method: some HF pages and PDFs were read through a summarizing fetcher (and the DeepSeek-V4, GLM-5, Kimi-K3 and MiniMax-MSA PDFs were read as full text locally). Where a number came only from a fetcher summary of a config.json, I still tag it [C] but builders should eyeball the config once before putting it on screen.

---

## 1. Evolution: GPT-3-era block -> 2026 block

| Component | 2020-23 baseline (GPT-3, early GPT-NeoX) | 2026 frontier open models | What changed and why |
|---|---|---|---|
| Norm | LayerNorm, post-norm (original) then pre-norm (GPT-2/3) | Pre-norm **RMSNorm** (sometimes pre+post, e.g. Gemma) | Pre-norm trains stably at depth; RMSNorm drops mean-centering, cheaper, same quality. |
| Position | Learned absolute embeddings (GPT-2/3) | **RoPE** (partial RoPE; or NoPE on some layers) | Relative, extrapolates/extends better, no table limit. See sec. 3. |
| MLP | Dense, GELU, 4x width | Dense SwiGLU for shared/dense layers; **fine-grained MoE** for most layers | Gated activation gives better quality per FLOP; MoE decouples total params (knowledge) from active params (cost). |
| Attention | Full MHA (every head has own K,V) | GQA (8 KV heads typical), **MLA** (DeepSeek/Kimi/GLM), sliding-window mixes, sparse/compressed, linear-attention hybrids | KV cache is the serving bottleneck; every change shrinks it or avoids reading all of it. |
| QK-norm | none | RMSNorm on Q and K (Gemma 3, Qwen3, DeepSeek-V4 norms queries and KV entries) | Prevents attention-logit blowups at scale. DeepSeek-V4 uses it instead of Kimi's QK-Clip [C] https://arxiv.org/pdf/2606.19348 |
| Biases | biases in linear layers | mostly none (exception: gpt-oss keeps `attention_bias: true`) [C] https://huggingface.co/openai/gpt-oss-120b/raw/main/config.json | Simpler, tiny speedup; not a quality lever. |
| Embeddings | tied in GPT-2 | tied only in small models; large models untie | Tying saves params that matter only when the embedding is a large share of the model. |
| Attention sink | none | learned sink logit per head (gpt-oss, MiMo-V2, DeepSeek-V4) | Gives softmax a "do nothing" option; makes sliding windows stable. [C] V4 eq. 27 https://arxiv.org/pdf/2606.19348 |
| Residual stream | plain `x + f(x)` | **mHC** (DeepSeek-V4), **Attention Residuals** (Kimi K3) | New in 2026: the residual connection itself is being redesigned. [C] https://arxiv.org/pdf/2606.19348 , https://arxiv.org/html/2607.24653v2 |
| Decoding | 1 token/step | **Multi-token prediction** heads used for training signal and speculative drafting (V3/V4 depth 1, K3 1 layer, GLM-5 shares 3 MTP layers) | Denser training signal; ~2.8 accepted tokens/step for GLM-5 in its table. [C] https://arxiv.org/pdf/2602.15763 |
| Optimizer | AdamW | **Muon** (V4, Kimi, GLM-5 "Muon Split") | Faster convergence; affects architecture choices (GLM-5 found MLA needed Muon Split to match GQA-8). [C] same sources |
| Precision | FP16/BF16 | FP8 training; FP4 experts (V4 QAT), MXFP4 (gpt-oss, K3) | Memory/bandwidth. |

Sources for the design drift: Raschka's gallery https://sebastianraschka.com/llm-architecture-gallery/ [R as a secondary], and the primary papers cited below.

---

## 2. Attention mechanisms that matter in Oct 2026

KV-cache formulas (bytes per token, `b` = bytes per element, `L` = layers that keep a growing cache):
- MHA: `2 * L * n_heads * d_head * b`
- MQA: `2 * L * 1 * d_head * b`
- GQA: `2 * L * n_kv * d_head * b`
- MLA: `L * (d_latent + d_rope) * b` (one shared latent per token, not per head)
- Sliding window: same as GQA but tokens capped at window `w` (cache is constant after `w`)
- Linear / delta-rule: **no per-token cache**; constant state `~ H * d_k * d_v * b` per layer

### MHA / MQA / GQA (settled)
MHA gives each query head its own K,V. MQA (Shazeer 2019) shares one K,V across all heads (smallest cache, some quality loss). GQA (Ainslie et al. 2023, https://arxiv.org/abs/2305.13245) shares K,V within groups of query heads; 8 KV heads became the default compromise (Llama 3, Qwen3, gpt-oss, MiniMax-M3 uses 4). Widely adopted and proven.
DeepSeek-V4 effectively went back to an MQA-style **1 KV head with head_dim 512** (plus compression) [C] https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/raw/main/config.json .

### MLA (DeepSeek-V2, 2024)
Intuition: instead of caching K and V for every head, cache one low-rank latent vector per token and re-expand it to K,V at attention time (the up-projection matrices are absorbed into the Q and output projections at decode). Cache per layer = `d_latent + d_rope` (DeepSeek-V3: 512 + 64 = 576 numbers vs 2*128*128 = 32,768 for MHA, about 57x smaller) while keeping per-head diversity. https://arxiv.org/pdf/2405.04434
Used by: DeepSeek V3/V3.2, Kimi K2 and K3 (in the global layers), GLM-5.x (`kv_lora_rank 512`, `q_lora_rank 2048`) [C] https://huggingface.co/zai-org/GLM-5.3/raw/main/config.json , Mistral Large 3 (DeepSeek-V3-like) [R] Raschka gallery. Proven and now mainstream among Chinese frontier labs. Caveats from the GLM-5 report: MLA's decode compute is high; they enlarge head dim 192->256 and cut head count ("MLA-256") to fix, and need Muon Split to match GQA-8 quality [C] https://arxiv.org/pdf/2602.15763 .
V4 dropped classic MLA for CSA/HCA (below), so "MLA is final" is not true; it is the incumbent, not the endpoint.

### Sliding-window / local-global interleaving
Intuition: most layers attend only to the last `w` tokens (cache is capped at `w`); every k-th layer attends to everything. Plus a learned **attention sink** so the softmax has somewhere to put mass (StreamingLLM, https://arxiv.org/abs/2309.17453).
- Gemma 3: 5 local : 1 global, window 1024. Gemma 4 (Apr 2, 2026): 5:1 (E2B 4:1), 31B window 1024, 256K context [R] https://www.sebastianraschka.com/blog/2026/gemma-4-release-notes.html , https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4
- gpt-oss-120b: alternating full / banded layers, window 128, 36 layers, learned sinks [C] config above.
- MiMo-V2-Flash (309B/15B): 5 SWA : 1 global, window 128, learned sink bias [C] https://arxiv.org/pdf/2601.02780 ; MiMo-V2.5/2.6-Pro (1T/42B) reported 1:7 global:SWA, 1M ctx [R] https://mimo.mi.com/docs/en-US/updates/model
- DeepSeek-V4 adds a 128-token sliding-window branch to *every* CSA/HCA layer [C].
Proven, widely used. Caveat (GLM-5 report): retrofitting SWA onto a trained full-attention model degraded long-context tasks badly even with half the layers kept full; works when trained in from scratch [C] https://arxiv.org/pdf/2602.15763 .

### Sparse attention: pick which past tokens to read
- **DeepSeek Sparse Attention (DSA)**, V3.2 (Dec 2025, https://arxiv.org/abs/2512.02556 [C]): a small "lightning indexer" (few heads, low precision) scores every past token for the current query; main attention reads only the top-k (k=2048). Cost is still O(L^2) for the indexer but cheap; main attention becomes O(L*k). **Important for teaching: DSA does not shrink the KV cache** (you still store everything so the indexer can look at it); it cuts compute and bandwidth. GLM-5 adopted DSA (continued-pretrain from MLA, lossless in their tests; index top-k 2048 in the 5.3 config) [C] https://arxiv.org/pdf/2602.15763 .
- **CSA + HCA, DeepSeek-V4 (Apr 2026)** [C] https://arxiv.org/pdf/2606.19348 : *compress first, then sparsify.* CSA merges every m=4 tokens' KV into one entry, then DSA-style top-k over the compressed entries (k=512 Flash, 1024 Pro). HCA merges every 128 tokens into one entry and attends densely over those. Layers alternate CSA/HCA, each with a 128-token sliding-window branch; KV stored mixed FP8/BF16 (only RoPE dims in BF16); indexer in FP4. Result at 1M tokens: Pro uses 27% of V3.2's per-token FLOPs and 10% of its KV cache; overall about 2% of a BF16 GQA-8/d128 baseline.
- **MiniMax Sparse Attention (MSA), M3 (Jun 2026)** [C] https://arxiv.org/abs/2606.13392 : block-sparse on top of GQA. Index branch max-pools scores per block of 128 tokens, picks top-16 blocks per GQA group, always keeps the most recent block. M3 config: 4 KV heads, sparse from layer 3 [C] https://huggingface.co/MiniMaxAI/MiniMax-M3/raw/main/config.json . Claims 28.4x less attention compute at 1M on a 109B test model. Like DSA, KV is still stored.
- **NSA** (DeepSeek, Feb 2025, https://arxiv.org/abs/2502.11089): compression + selection + sliding-window branches, trained natively; the conceptual ancestor of CSA. **MoBA** (Moonshot, 2502.13189): MoE-style gating over blocks. Both are now research-lineage; production models use DSA/CSA/MSA. 
Status: DSA is proven at 744B-scale (GLM-5) and 1M (V4); sparse+compressed is the leading 2026 route to 1M context for MLA/GQA lineages.

### Hybrid linear attention (O(1) memory layers mixed with a few global layers)
Intuition: replace softmax attention in ~3 of every 4 layers with a recurrent "fast-weight" memory matrix updated per token by a delta rule (Gated DeltaNet: https://arxiv.org/abs/2412.06464; gate controls forgetting). Those layers keep a fixed-size state instead of a growing cache; the 1-in-4 full-attention layers do exact retrieval.
- **Qwen3-Next (Sep 2025) -> Qwen3.5 (Feb 2026) -> Qwen3.8 (Aug 2026)**: 3 Gated-DeltaNet : 1 gated full attention. Qwen3.5-397B-A17B: 60 layers, 15 full-attention layers with 32 Q heads / 2 KV heads / head_dim 256; Qwen3.8-2.4T-A95B: 92 layers (23 x (3 GDN + 1 attn)), 64 Q / 4 KV heads [C] https://huggingface.co/Qwen/Qwen3.5-397B-A17B/raw/main/config.json , https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B
- **Kimi Linear (Oct 2025)** https://arxiv.org/abs/2510.26692 [C]: Kimi Delta Attention = GDN with **per-channel** decay; 3 KDA : 1 MLA, reports up to 75% KV-cache reduction and ~6x decode throughput at 1M for a 48B/3B test model. **Kimi K3 (Jul 2026)** scaled it: 93 layers = 69 KDA + 24 gated MLA, MLA layers use **NoPE** (position comes from KDA's decay) [C] https://arxiv.org/pdf/2607.24653 .
- **MiniMax lightning attention**: used in MiniMax-01 and M1 (2025). MiniMax then *abandoned* linear attention for M2 (full attention, Oct 2025 - from MiniMax's own blog, not re-fetched [U]) and moved to sparse attention for M3 [C: M3 uses MSA]. This is a useful counter-example: hybrids are not universal.
- **Mamba hybrids**: NVIDIA Nemotron 3 (Mamba-2 + attention + LatentMoE; Super 120B/12B active, Ultra ~550B/55B) [R] https://arxiv.org/pdf/2604.12374 , https://www.sebastianraschka.com/blog/2026/nemotron-3-ultra-latent-moe.html .
Adoption verdict: GDN/KDA hybrids are **proven at frontier scale** by two labs (Qwen, Moonshot) but the field is split: DeepSeek, GLM, MiniMax chose sparse/compressed softmax attention instead.

### FlashAttention (what it is)
Exact attention computed in SRAM tiles with an online softmax, so the N x N score matrix never goes to GPU HBM (IO-aware; speed and memory, not approximation). FA1 2022 (2205.14135), FA2 2023 (2307.08691), FA3 2024 Hopper/FP8 (2407.08608), **FA4 (Mar 2026), Blackwell-targeted**, reports ~1.6 PFLOP/s on B200, ~71% utilization [R] https://arxiv.org/abs/2603.05451 , https://lambda.ai/blog/flashattention-4-gives-the-nvidia-blackwell-platform-its-most-optimized-attention-kernel-yet . Orthogonal to all variants above: it makes whichever pattern you choose fast.

---

## 3. Positional encoding

- **RoPE** (Su et al., https://arxiv.org/abs/2104.09864): rotate each pair of Q/K dimensions by angle `pos * theta_i`, with `theta_i = base^(-2i/d)`. The Q.K dot product then depends only on relative offset. Low-index pairs spin fast (local order), high-index pairs spin slowly (long range). `base` (theta) controls the slowest wavelength: bigger base = longer reach. 2026 values from configs: gpt-oss 150,000; DeepSeek-V4 10,000 (and 160,000 for the compressed streams); MiniMax-M3 5,000,000; GLM-5.3 8,000,000; Qwen3.5/3.8 10,000,000 [C] (configs linked above).
- **Partial RoPE**: rotate only a fraction of dims. Qwen3.5: `partial_rotary_factor 0.25`; MiniMax-M3 0.5 (rotary dim 64 of 128) [C]. Rest of dims are position-free (content matching).
- **Decoupled RoPE in MLA**: the compressed latent can't be rotated (RoPE would not commute with the absorbed up-projection), so MLA keeps a small separate `d_rope`=64 key part that carries RoPE and is cached next to the latent: 512 latent + 64 rope. https://arxiv.org/pdf/2405.04434
- **NoPE**: no explicit position. Works because causal attention leaks order (Kazemnejad et al. 2023, 2305.19466). Llama 4 "iRoPE" put NoPE on global layers. **Kimi K3 uses NoPE on all MLA layers**, extrapolates to 1M "without any positional-encoding modification" [C] https://arxiv.org/pdf/2607.24653 .
- **Context extension**: Position Interpolation (2306.15595) squeezes positions into the trained range; NTK-aware scaling raises `base` instead; **YaRN** (2309.00071) = per-frequency interpolation + attention-temperature fix; needs far fewer tokens. gpt-oss: YaRN factor 32 from 4,096 -> 131,072 [C]; DeepSeek-V4 YaRN factor 16 [C].
- **What is mainstream for 1M (Oct 2026)**: (1) very large RoPE base (5M-10M) + (2) staged length training (K3: 8K->64K->256K->1M [C]; V4: 4K->16K->64K->1M [C]) + (3) an attention design that is cheap at length (sparse/compressed/linear) + optionally YaRN. Most frontier open models list ~1,048,576 positions: V4, K3, GLM-5.3, M3, MiMo-V2.6 [C/R]. Not all are equal: Mistral Large 4 claims 1M but evaluators measure ~512K [R] https://kingy.ai/blog/mistral-large-4-specs-benchmarks-pricing/ ; Qwen3.5/3.8 natively train 262K and extend to ~1.01M (HF card; method not named, likely YaRN [U]).

---

## 4. KV caching

Mechanism: at decode step t, the new token's Q is compared with the K of all earlier tokens. K and V of earlier tokens never change (causal mask), so compute them once and store them: that is the KV cache. It turns per-step cost from re-running the whole prefix to one token of compute plus a read of the whole cache; decoding becomes **memory-bandwidth-bound**, which is why every trick in sec. 2 attacks cache size or cache reads. Prefix caching (reuse cache across requests with the same prompt prefix) is table-stakes in serving.

Formula (GQA/MHA): `bytes/token = 2 * layers * kv_heads * head_dim * bytes_per_elem`; multiply by tokens and by concurrent sequences.

Worked numbers (BF16/FP16 = 2 bytes unless stated):

| Model | Calculation | Per token | At 128K (131,072 tok) | At 1M (1,048,576 tok = 2^20) |
|---|---|---|---|---|
| GPT-3 175B (96 L, 96 heads x128, MHA) | 2*96*96*128*2 | 4.5 MiB (4,718,592 B) | n/a (2K ctx: 9.7 GB) | n/a |
| Llama-3-70B (80 L, 8 KV x128, GQA) | 2*80*8*128*2 | 320 KiB (327,680 B) | 43 GB | 344 GB |
| DeepSeek-V3 (61 L, MLA 512+64) | 61*576*2 | 68.6 KiB (70,272 B) | 9.2 GB | 74 GB |
| GLM-5.3 (78 L, MLA 512 latent + 64 rope assumed) | 78*576*2 | ~88 KiB (89,856 B), derived; rope dim assumed 64 [U] | 11.8 GB | ~94 GB |
| MiniMax-M3 (60 L, 4 KV x128) | 2*60*4*128*2 | 120 KiB (122,880 B), derived | 16 GB | 129 GB (MSA saves compute, not storage) |
| Qwen3.5-397B (15 of 60 layers full attn, 2 KV x256) | 15*2*2*256*2 | 30 KiB (30,720 B), derived, + constant GDN state per layer | 4 GB | 32 GB |
| gpt-oss-120b (18 full layers x 8 KV x64; 18 SWA layers capped at 128 tok) | 18*2*8*64*2 | 36 KiB (36,864 B) + constant ~4.7 MB for the window layers, derived | 4.8 GB | n/a (131K max) |
| DeepSeek-V4-Pro | paper: "approximately 2%" of BF16 GQA-8/d128 baseline (baseline = 2*61*8*128*2 = 249,856 B) | ~5 KB, derived from paper's 2% | ~0.65 GB | ~5.2 GB |
| DeepSeek-V4.1-Flash | reported "890 bytes per token" global cache | 0.9 KB [R] https://www.contextstudios.ai/blog/deepseek-v4-1-flash-890-bytes-kv-cache-per-token | 0.1 GB | ~0.9 GB |

Headline for a teaching page: from GPT-3's 4.5 MiB/token to ~1-5 KB/token is a ~1000-5000x drop in 5 years, and that is what makes 1M-token contexts economical. Kimi K3: 24 MLA layers; per-token cache is roughly 24/61 of K2's if latent dims are unchanged (K2 = 61*576*2 = 70 KB -> ~28 KB) but the K3 latent dims were not read [U]; plus constant KDA state.

---

## 5. MoE as of 2026

Core ideas: replace the one big FFN with N small expert FFNs plus a router; each token uses only top-k. **Fine-grained experts** (DeepSeekMoE, 2401.06066): many small experts (instead of 8 big ones like Mixtral) for more combinatorial flexibility. **Shared experts**: 1-2 always-on experts capture common knowledge so routed ones can specialise. **Routers**: softmax top-k (Mixtral, Qwen3); sigmoid affinity + bias (DeepSeek-V3); **sqrt-softplus** affinity (V4 config `scoring_function: sqrtsoftplus`) and **hash routing** for the first 3 MoE layers (expert chosen by a fixed hash of token id) [C] https://arxiv.org/pdf/2606.19348 . **Aux-loss-free balancing** (DeepSeek, 2408.15664): add a per-expert bias to the routing score *for selection only*; nudge it up for under-used experts and down for over-used, no auxiliary loss gradient; V4 adds only a small sequence-level balance term [C]. Kimi K3 replaces it with **Quantile Balancing** [C]. **Latent MoE** (K3 latent dim 3584 = 0.5x hidden; Nemotron 3 latent 1024): route/compute experts in a down-projected space to cut all-to-all traffic [C K3 / R Nemotron].
**Multi-token prediction (MTP)**: extra head(s) predict token t+2 (and onward) during training; at inference serve as a built-in speculative draft model (https://arxiv.org/abs/2404.19737). V4: depth 1 [C]; GLM-5 reports mean accept length 2.76 [C]; K3: one MTP layer fine-tuned as an EAGLE-3 draft [C].

| Model (date) | Total / active | Experts | Notes / source |
|---|---|---|---|
| DeepSeek-V3 (Dec 2024) | 671B / 37B | 256 routed + 1 shared, top-8, 61 L | background |
| DeepSeek-V4-Pro (Apr 2026 preview, GA "0813" Aug 2026) | **1.6T / 49B** | 384 routed + 1 shared, top-6, expert width 3072, 61 L, d=7168 | [C] https://arxiv.org/pdf/2606.19348 ; GA date [R] https://tech.yahoo.com/ai/articles/deepseek-officially-launches-v4-pro-181255468.html |
| DeepSeek-V4-Flash | 284B / 13B | 256 routed + 1 shared, top-6, 43 L, d=4096 | [C] same |
| DeepSeek-V4.1-Flash (Sep 10, 2026) | 552B / 8B prefill, 16B decode | 384 routed +1 shared top-6 (config), 40 L (20 enc + 20 dec "causal encoder-decoder") | [R] HF card/config + https://arxiv.org/abs/2609.19969 (fetcher summaries; read twice before showing) |
| Kimi K2 (Jul 2025) | 1.04T / 32B | 384 routed + 1 shared, top-8, 61 L | background |
| Kimi K3 (weights Jul 2026) | **2.78T / 104.2B** | 896 routed (latent) + 2 shared, top-16, 93 L, d=7168, 96 heads, vocab 160K | [C] https://arxiv.org/pdf/2607.24653 , https://github.com/MoonshotAI/Kimi-K3 |
| Qwen3 (Apr 2025) | 235B / 22B | 128 routed, top-8, no shared | background |
| Qwen3.5-397B-A17B (Feb 2026) | 397B / 17B | 512 routed, top-10, + shared expert, 60 L | [C] HF config |
| Qwen3.8-2.4T-A95B (Aug 2026) | **2.4T / 95B** | 512 routed top-10 + 1 shared, 92 L, d=8192 | [C] https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B |
| GLM-5 (Feb 2026) | 744B / 40B | 256 experts, top-8 (+1 shared in config), 80 L per paper | [C] https://arxiv.org/pdf/2602.15763 |
| GLM-5.3 (Aug 2026) | 753B / 40B (active: carried over from GLM-5; HF card says "no architectural changes") | 256 routed + 1 shared, top-8, 78 L in HF config, d=6144 | [C] config; 40B [R] |
| gpt-oss-120b / 20b (Aug 2025) | 116.8B / 5.1B ; 20.9B / 3.6B | 128 experts top-4 ; 32 experts top-4 | [C] config + model card https://arxiv.org/abs/2508.10925 |
| Mistral Large 3 (Dec 2025) | ~675B / ~41B | granular MoE, DeepSeek-V3-like | [R] https://www.verdent.ai/guides/model/mistral-large-3 |
| Mistral Large 4 (preview Oct 6, 2026) | **1.05T / 49B** (52B incl. embeddings) | "granular MoE"; expert count unpublished | [R] https://kingy.ai/blog/mistral-large-4-specs-benchmarks-pricing/ ; Mistral news page says "1 trillion / 49B": https://mistral.ai/news/mistral-large-4 [C for headline numbers] |
| MiniMax-M3 (Jun 2026) | ~428B / ~23B | 128 routed + 1 shared, top-4, 60 L (3 dense), d=6144 | [C] HF card + config |
| MiMo-V2-Flash / V2.5-V2.6 | 309B/15B ; 1T/42B | n/a | [C] arXiv 2601.02780 / [R] mimo docs |

Active ratios: V4-Pro 3.1%, K3 3.7%, Qwen3.8-2.4T 4.0%, Mistral L4 4.7%, GLM-5.3 5.3%, M3 5.4%, gpt-oss-120b 4.4%, Qwen3.5 4.3%. Trend: **2.5-5% active** is the 2026 norm, down from ~9-28% (Qwen3-235B 9%, Mixtral 8x7B ~28%).

---

## 6. Multimodality

Two recipes:
1. **Adapter / late fusion (LLaVA-style, https://arxiv.org/abs/2304.08485)**: pretrained ViT (often SigLIP) -> small MLP projector -> image tokens interleaved with text tokens into a (usually already-trained) LLM. Cheap, modular.
2. **Native / early fusion**: text, image (and video/audio) tokens trained together from step 0 in one backbone, one next-token loss. Kimi K3's report states this explicitly and trains MoonViT-V2 **from scratch with next-token prediction** (SigLIP-initialised MoonViT-3D had unstable gradient norms) [C] https://arxiv.org/pdf/2607.24653 . MiniMax-M3: "mixed-modality training from the very first step" [C HF card]. Note both still have a **separate vision encoder + MLP projector**; "native" means joint pretraining, not "no encoder".

Mechanics:
- **Patch tokenization**: image cut into p x p patches (p=14 in K3, M3), each becomes a vector via linear projection; tokens = (H/p)*(W/p). 1008x1008 at p=14 = 5,184 patches. Pixel-shuffle/unshuffle merges 2x2 (K3, /4 tokens) or 3x3 (DeepSeek-ViT, /9) neighbours before the projector.
- **Dynamic resolution**: keep native aspect ratio rather than resize to a fixed square (Qwen2-VL naive dynamic resolution + M-RoPE, 2409.12191); K3 handles up to 3584x3584; M3 vision config max 2016x2016 with 3D RoPE [C].
- **Video**: frames sampled, factorised spatial-then-temporal attention in the encoder, temporal pooling (K3 [C]); 3D RoPE (M3 [C]). **Audio** (not covered by the reviewed flagship open LLMs; Qwen-Omni line does it with a Whisper-style encoder, roughly 25 tokens/s [U, from memory]).
- **Vision encoder sizes**: K3 MoonViT-V2 401M, 27 layers, patch 14 [C]; Qwen3.5 ViT 27 layers, width 1152 (SigLIP-sized, ~0.4B) [C config]; MiniMax-M3 ViT 32 layers, width 1280, patch 14 [C config]; Mistral Large 4 1.6B [R], Large 3 2.5B [R]; DeepSeek-V4.1-Flash "DeepSeek-ViT" 32 layers, width 1024, 2-layer MLP projector [R].

2026 reality check: **multimodal is not universal** among frontier open text leaders. DeepSeek-V4 (tech report lists multimodality as future work) [C] and GLM-5.3 and gpt-oss and Qwen3.8-2.4T ("text-only model", HF card) [C] are text-only; K3, M3, Mistral L4, Qwen3.5, MiMo-V2.6 are multimodal. Several secondary sources wrongly call V4 "natively multimodal" (e.g. https://www.digitalapplied.com/blog/deepseek-v4-trillion-parameter-open-source-multimodal) - the paper contradicts them. V4.1-Flash is the first DeepSeek model with a vision encoder (reported).

---

## 7. Frontier open-weight spec sheet (Oct 2026)

| Model (release) | Total / active | Attention | Context | Layers | Multimodal | Weights / license | Confidence |
|---|---|---|---|---|---|---|---|
| **DeepSeek-V4-Pro** (preview Apr 24; GA -0813 Aug 2026) | 1.6T / 49B | CSA(m=4,top-k 1024)+HCA(m'=128)+SWA128, 1 KV head d=512, mHC, Muon | 1M | 61 | No (text) | MIT | [C] https://arxiv.org/pdf/2606.19348 |
| DeepSeek-V4-Flash | 284B / 13B | same family, top-k 512 | 1M | 43 | No | MIT | [C] |
| DeepSeek-V4.1-Flash (Sep 10) | 552B / 8B pre, 16B dec | CSA2 (Full/Reindex/Reuse), causal enc-dec | 1M | 40 | Yes (DeepSeek-ViT) | MIT | [R] https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash |
| **Kimi K3** (weights ~Jul 27) | 2.78T / 104.2B | 69 KDA + 24 gated MLA (NoPE), AttnRes, LatentMoE | 1M | 93 | Yes, native (MoonViT-V2 401M) | Kimi K3 License (custom) | [C] https://github.com/MoonshotAI/Kimi-K3 |
| **GLM-5.3** (Aug 2026; GLM-5 Feb) | 753B / 40B | MLA + DSA (index top-k 2048), RoPE theta 8M | 1M (GLM-5: 200K) | 78 (cfg) / 80 (GLM-5 paper) | No | custom glm-5.3 licence; 5.3-Flash MIT [R] | [C]/[R] https://huggingface.co/zai-org/GLM-5.3 |
| **Qwen3.8-2.4T-A95B** (HF Aug 12) | 2.4T / 95B | 3 GDN : 1 gated attn, 64Q/4KV heads | 262K native, extensible to ~1.01M (method [U]) | 92 | No (text-only) | custom (Qwen3.8-Max terms) | [C] https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B |
| Qwen3.5-397B-A17B (Feb) | 397B / 17B | 3 GDN : 1 attn | 262K | 60 | Yes | Apache 2.0 [R] | [C] config |
| Qwen3.8-27B / Flash-Next 180B | 27B dense ; 180B | hybrid | - | - | - | 27B Apache 2.0 [R] | [R] |
| **MiniMax-M3** (Jun 1-11) | ~428B / ~23B | GQA 64Q/4KV + MSA (block 128, top-16) | 1M | 60 | Yes, native (ViT 32L) | minimax-community | [C] https://huggingface.co/MiniMaxAI/MiniMax-M3 |
| MiMo-V2.6-Pro (Sep 22) | ~1T / 42B | 1:7 global:SWA | 1M | - | Yes | HF lists 1T checkpoints (MIT per [R]) | [R] https://mimo.mi.com/docs/en-US/updates/model |
| gpt-oss-120b (Aug 2025; no successor found) | 116.8B / 5.1B | alternating full + SWA128, sinks, GQA 64Q/8KV | 131K | 36 | No | Apache 2.0 | [C] |
| **Mistral Large 4 "Le Chonk"** (preview Oct 6) | 1.05T / 49B (52B incl. emb.) | undisclosed | 1M claimed; evaluators ~512K | - | Yes (1.6B encoder) | **weights not yet out (ETA Oct 31); licence pending** | [R]/[C] see below |
| Llama | Llama 4 (Apr 2025) is the last release I could verify. Wikipedia says Muse Spark (closed) replaced Llama for Meta apps (Apr 2026). SEO pages claim a "Llama 5" (600B, 5M ctx) with contradictory context figures | | | | | | **[U] - leave blank, not relevant for frontier** |

### Verdicts on the chat claims
- **Mistral Large 4 ~1.05T / 49B, granular MoE, 1M context**: parameters and "granular MoE" **correct** (Mistral: "1 trillion"/49B in prose; model card 1.05T, 49B routed-active, 52B with embeddings). **1M context is Mistral's claim only; third-party evaluators measure ~512K**. Also: it is currently a *preview via API*; open weights promised ~Oct 31, 2026; licence not final. Source: https://mistral.ai/news/mistral-large-4 [C], https://kingy.ai/blog/mistral-large-4-specs-benchmarks-pricing/ [R].
- **DeepSeek V4 Pro 1.6T / 49B**: **correct** [C]. Additions: V4-Flash 284B/13B; **V4 is text-only**, 1M context, MIT; GA version V4-Pro-0813; V4.1-Flash (552B) exists.
- **Kimi K3 2.8T**: **correct** (2.78T, 104.2B active, 1M context) [C].
- **GLM-5.3 753B / 40B**: 753B **correct** [C HF card]; 40B active is the GLM-5 figure carried over, not independently printed on the 5.3 card [R]. Context 1M. Text-only.
- **MiniMax M3 428B / 23B**: **correct** per HF card; some secondary sites say 22B [C/R].
- Missing from the chat's list: **Qwen3.8-2.4T-A95B** (open, Aug 2026), **MiMo-V2.6-Pro 1T/42B**, DeepSeek-V4.1-Flash.

---

## 8. Teaching notes and existing explainers

**Interactive ideas (slider -> live output):**
1. *Evolution table*: toggle each component (LayerNorm->RMSNorm, GELU->SwiGLU, abs->RoPE, MHA->GQA, dense->MoE); live readout of params, active params, KV bytes/token.
2. *Attention variants*: slider "KV heads" 96 -> 8 -> 1 and a "latent dim" slider for MLA; animate which heads share a K/V block; live bytes/token and GB at chosen context.
3. *Attention mask visualizer*: 32x32 causal grid; buttons for full / sliding-window w / DSA top-k / CSA (compress m, then top-k) / hybrid 3:1; slider for context length shows "tokens read per step" and a bar of FLOPs vs full.
4. *Linear attention memory*: a small matrix S (fast weights) updated per token with delta rule; slider for forget-gate; show recall of a key inserted at t=0 as distance grows vs full attention.
5. *RoPE clock*: pairs of dims drawn as rotating hands; sliders for position, base theta (10K...10M); show dot-product vs offset; second panel shows YaRN/PI squeezing positions.
6. *KV-cache calculator*: dropdown of real models (table in sec. 4), context slider to 1M, batch slider; live GB vs an 80 GB H100 bar; "why does decode stall" bandwidth animation.
7. *MoE router*: tokens flow to top-k of N experts; sliders for N, k, shared experts; live active-parameter %; toggle the load-balancing bias and watch expert-load histogram flatten (aux-loss-free demo).
8. *Patch tokenizer*: upload/choose image, slider for patch size, resolution, pixel-unshuffle factor; live token count and "fraction of 1M context used"; video: fps slider.
9. *MTP*: show 1 vs 2 predicted tokens per step with an acceptance-probability slider -> tokens/sec.

**Existing explainers (URLs as known; fetched ones marked):**
- Raschka, LLM Architecture Gallery (109 models, diff tool, memory calculator) - https://sebastianraschka.com/llm-architecture-gallery/ (fetched OK)
- Raschka, "The Big LLM Architecture Comparison" - https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison
- bbycroft LLM visualization (3D walk through a GPT block) - https://bbycroft.net/llm
- Transformer Explainer (Polo Club, runs GPT-2 in browser) - https://poloclub.github.io/transformer-explainer/
- Maarten Grootendorst, A Visual Guide to Gemma 4 - https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4 (appeared in search)
- Jay Alammar, The Illustrated Transformer - https://jalammar.github.io/illustrated-transformer/
- 3Blue1Brown attention chapters - https://www.3blue1brown.com/lessons/attention
- Hugging Face Ultra-Scale Playbook (KV/parallelism) - https://huggingface.co/spaces/nanotron/ultrascale-playbook
- Qwen3.5 attention overview - https://huggingface.co/blog/mlabonne/qwen35
(I did not re-verify the liveness of the bbycroft, Polo Club, Alammar, 3B1B and HF playbook URLs this session.)

---

## Open uncertainties
- GLM-5.3 release date conflicts across secondary sources (Aug 14 vs Aug 18 API; weights Aug 28 reported); active params and rope dim not printed on the 5.3 card. GLM layer count: 80 (GLM-5 paper) vs 78 (HF config).
- Kimi K3 MLA latent dims and exact release date (weights ~Jul 27, 2026 [R]); tech report arXiv 2607.24653.
- Mistral Large 4: context (1M vs ~512K), expert count, attention type, licence, weight drop (promised ~Oct 31).
- DeepSeek-V4.1-Flash details come from fetcher summaries of the HF card/config and secondary articles.
- MiniMax M2 "back to full attention" and Qwen-Omni audio-token rate are from memory, not re-verified.
- "Llama 5" is unverified. MiMo-V2.6 licence/open status partially unverified.
