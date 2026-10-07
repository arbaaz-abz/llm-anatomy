# 04 - How LLMs Are Served at Scale (Inference), state of the art as of 2026-10-07

Research brief for page-builders. Learner: knows transformer basics, not an infra pro.

**Confidence tags** - CONFIRMED: read in a primary source (paper, official docs/blog, model card) this session. REPORTED: from a secondary source, a vendor claim without methodology, or a number I could not trace to a primary. UNVERIFIED: plausible but not checked. Items marked *(prior)* are classic results from papers I know well but did not re-fetch this session; the arXiv links are stable and worth a spot check before quoting exact numbers. "ESTIMATE" marks numbers I derived myself; formulas are shown so the toy can recompute them.

---

## 0. The 60-second story (for page intros)

1. A request has two very different phases: **prefill** (read the whole prompt in parallel; compute-bound) and **decode** (emit one token per step, re-reading all weights and the KV cache each step; memory-bandwidth-bound).
2. Everything in modern serving is a trick to make decode less wasteful: batch many users so one weight read serves many tokens (continuous batching), store KV efficiently (paging, prefix reuse, offload, quantization), guess several tokens at once (speculative decoding), shrink bytes (FP8/FP4), and split the phases and the experts across many GPUs (disaggregation, expert parallelism).
3. The 2026 frontier: 1T-1.6T-parameter MoE models with 1M-token context, served on rack-scale NVLink systems (GB200/GB300 NVL72) with prefill/decode disaggregation, wide expert parallelism, KV-aware routing and tiered KV storage, orchestrated by Dynamo or llm-d on Kubernetes, with vLLM / SGLang / TensorRT-LLM as engines.

---

## 1. The two phases, the metrics, and the back-of-envelope

### 1.1 Prefill vs decode

| | Prefill | Decode |
|---|---|---|
| What happens | All prompt tokens go through the network at once; K/V for every prompt token are computed and stored | One new token per step per request; reads all (active) weights + the whole KV cache |
| Parallelism | Over the prompt length (thousands of tokens per forward pass) | Over the batch only (one token per request) |
| Bound by | Compute (FLOPs). Big matrix-matrix products | Memory bandwidth (bytes). Skinny matrix-vector-ish products |
| Latency metric | TTFT (time to first token) | TPOT / ITL (time per output token / inter-token latency) |
| Attention cost | O(n^2 d) | O(n d) per token thanks to the KV cache (Hugging Face, [continuous batching from first principles](https://huggingface.co/blog/continuous_batching), CONFIRMED) |

**Why decode is memory-bound (arithmetic intensity).** One decoded token with weights in W-byte precision does ~2 FLOPs per parameter and must read each parameter once. For a batch of B requests, a dense layer does 2B FLOPs per parameter but reads that parameter once, so intensity is about `2B / bytes_per_param` FLOPs per byte. A GPU is compute-bound only above its "ridge point" = peak FLOPs / HBM bandwidth:

- H100: ~990 TFLOPs dense BF16 / 3.35 TB/s ~ 300 FLOP/byte (specs from memory, *(prior)*).
- B200: NVFP4 ~9,000 TFLOP/s and 8 TB/s HBM (InferenceX, [Kimi K2.5/K2.6 post](https://inferencex.semianalysis.com/blog/b200-nvfp4-vs-h200-int4-kimi-k2-vllm-perf-per-dollar), CONFIRMED) -> ridge ~1,100 FLOP/byte at FP4.
- So a dense BF16 model needs a batch of roughly 300 tokens in flight to become compute-bound. In MoE the effective batch **per expert** is `B * k / E` (k active of E experts), so a model that activates 6 of 384 experts needs ~64x more total tokens in the batch to feed each expert equally. This is the single biggest reason DeepSeek and others use huge expert-parallel groups (section 6): aggregate tokens from many users so each expert sees a decent batch.

### 1.2 Metrics

| Metric | Definition | Notes |
|---|---|---|
| TTFT | Time from request arrival to first output token = queue wait + prefill time | Dominated by prompt length and queueing; 128K-token prompt on DeepSeek-R1 / GB300 NVL72 with chunked pipeline-parallel prefill: 8.6 s ([LMSYS GB300 long-context](https://lmsys.org/blog/2026-02-19-gb300-longctx), CONFIRMED) |
| TPOT / ITL | Mean (or P99) time between successive output tokens | Interactivity = 1/TPOT, in tokens/s/user (InferenceX uses this on the x-axis) |
| Throughput | Tokens/s per GPU (total across all users) | The operator's number; sets $/token |
| Goodput | Max request rate that still meets the TTFT and TPOT SLOs (DistServe's definition) | Throughput that violates SLOs is worthless; goodput is why disaggregation wins ([DistServe](https://arxiv.org/abs/2401.09670): 7.4x more requests or 12.6x tighter SLO, REPORTED via secondary summary; paper *(prior)*) |
| Cost per 1M tokens | `($/GPU/hr) / (tok/s/GPU x 3600) x 1e6` | InferenceX convention: converts measured throughput at a fixed interactivity to $/M total (input+output) tokens ([InferenceX about](https://inferencex.semianalysis.com/about), CONFIRMED). I verified the arithmetic: $2.65/hr and 6,182 tok/s/GPU -> $0.119/M, matches their $0.12 |

### 1.3 Batch size trades latency for throughput (the central curve)

At small batch the step time is ~constant (weights dominate), so adding users is nearly free: throughput rises ~linearly while TPOT stays flat. As batch grows, KV reads (proportional to B x context) and eventually compute start to add to step time, TPOT rises, and throughput saturates. Result: a **Pareto frontier** of tok/s/GPU vs tok/s/user. Concrete measured anchor, DeepSeek-V4-Pro on GB300 vs GB200 at ISL 8K/OSL 1K ([InferenceX GB300 vs GB200](https://inferencex.semianalysis.com/blog/gb300-nvl72-vs-gb200-nvl72-dsv4-pro-vllm-fp4), CONFIRMED, measured 2026-05-22):

| Operating point | GB300 | GB200 |
|---|---|---|
| Max throughput (low interactivity) | 11,056 tok/s/GPU @ 13.1 tok/s/user | 8,933 tok/s/GPU @ 15.3 tok/s/user |
| 27 tok/s/user | 6,182 tok/s/GPU, $0.12/M | 2,189 tok/s/GPU, $0.28/M |

Read: demanding 2x the per-user speed cut per-GPU throughput roughly in half on the same hardware. Kimi K2.5 on B200 NVFP4 shows the same: $0.140/M at 32 tok/s/user -> $0.347/M at 90 tok/s/user ([InferenceX](https://inferencex.semianalysis.com/blog/b200-nvfp4-vs-h200-int4-kimi-k2-vllm-perf-per-dollar), CONFIRMED).

### 1.4 Back-of-envelope decode step time

```
t_step  ~  ( active_weight_bytes_read + KV_bytes_read ) / HBM_bandwidth      (+ comms, + kernel overhead)
```

- Dense, batch 1, Llama-class 70B in FP8 on one H200 (4.8 TB/s): 70 GB / 4.8 TB/s = 14.6 ms -> ceiling ~68 tok/s/user. (ESTIMATE; ignores KV and overhead; real engines reach maybe 60-80% of the bandwidth roofline.)
- With tensor parallelism over N GPUs, weights per GPU drop by N but all-reduces add latency, so speedup is sublinear.
- MoE at **small batch** reads only the experts that were routed to (about `min(E, B*k)` experts per layer); at **large batch** nearly every expert is touched every step so you pay for all weights. That is why MoE decode economics depend so heavily on batch size.
- KV term: `KV_bytes = batch x context x KV_bytes_per_token` (section 3.1). For a 70B GQA model at 128K context that is ~42 GB per sequence, easily larger than the weights share.
- Hardware constants for the toy: H100 3.35 TB/s / 80 GB; H200 4.8 TB/s / 141 GB; B200 8 TB/s / ~180 GB; GB200 192 GB per GPU; GB300 288 GB, 8 TB/s per GPU, ~20 TB HBM and 130 TB/s NVLink per 72-GPU rack ([NVIDIA GB300 NVL72](https://www.nvidia.com/en-gb/data-center/gb300-nvl72), REPORTED via search snippet; InferenceX confirms 192/288 GB and 8 vs 4.8 TB/s).

---

## 2. Batching and scheduling

### 2.1 Static -> continuous batching

- **Static batching**: group N requests, run until the *longest* finishes; finished slots idle, new requests wait. Padding and idle waste. HF's formula: adding an n-token prompt to B decoding prompts under padded dynamic batching wastes `(n-1)(B-1)` pad tokens, e.g. B=8, n=100 -> 693 ([HF](https://huggingface.co/blog/continuous_batching), CONFIRMED).
- **Continuous / in-flight / iteration-level batching** (Orca, OSDI 2022, *(prior)*; [paper PDF](https://www.usenix.org/conference/osdi22/presentation/yu)): the scheduler runs **once per decode iteration**; finished requests leave immediately and waiting requests join on the next step. **Ragged batching** (concatenate variable-length sequences, use masks) removes padding (HF, CONFIRMED). Orca reported order-of-magnitude throughput gains over FasterTransformer at equal latency *(prior, check number)*.
- vLLM's engine core is exactly this loop: schedule -> run model -> sample -> update -> repeat; walkthrough in [Inside vLLM](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) (CONFIRMED to exist and cover scheduling, paged attention, chunked prefill, prefix caching, spec decoding).

### 2.2 Chunked prefill (stall-free batching)

Problem: a long prompt's prefill monopolizes an iteration, so every decoding user sees a latency spike (TPOT stall). Fix: split the prompt into chunks (e.g., 512-8K tokens) and fill each iteration's **token budget** with all running decodes first, then a chunk of prefill. Sarathi-Serve: up to 2.6x serving throughput within SLO for Mistral-7B on one A100, up to 6.9x for Falcon-180B on 8 A100 vs Orca/vLLM ([Sarathi-Serve, OSDI'24](https://arxiv.org/html/2403.02310v3), CONFIRMED via abstract). Chunking also bounds activation memory and balances pipeline stages. Caveat: chunking adds KV re-reads (each chunk attends to all earlier chunks) and vLLM notes "minimized chunking overheads" mattered for the 64K-token prefill batches on GB200 ([vLLM GB200 post](https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1), CONFIRMED).

### 2.3 Scheduling policies

Typical engine scheduler knobs (vLLM/SGLang; general knowledge *(prior)*): FCFS by default; priority classes; a per-step `max_num_batched_tokens` budget; `max_num_seqs` cap; **preemption** when KV blocks run out (evict a request, recompute or swap it later); prefill-first vs decode-first. Research variants: shortest-job-first with predicted output length, SLO-aware / predicted-latency scheduling (llm-d's "predicted-latency scheduling GA" in v0.7, [llm-d 0.7 docs](https://llm-d.ai/docs/0.7), REPORTED via search snippet).

### 2.4 Routing across replicas

A plain round-robin load balancer destroys cache locality. 2026 stacks route by **KV-cache state**:

- **KV-aware / prefix-aware routing**: send the request to the replica already holding the longest matching prefix (system prompt, earlier turns, shared document), balanced against load. llm-d's router (Kubernetes Gateway API Inference Extension, "EPP") does prefix-cache-aware routing ([llm-d CNCF blog](https://www.cncf.io/blog/2026/03/24/welcome-llm-d-to-the-cncf-evolving-kubernetes-into-sota-ai-infrastructure/), REPORTED via snippet); NVIDIA Dynamo 1.0's KV-aware router sends requests "to GPUs that already have the most relevant short-term memory from earlier steps" ([NVIDIA press release](https://investor.nvidia.com/news/press-release-details/2026/NVIDIA-Enters-Production-With-Dynamo-the-Broadly-Adopted-Inference-Operating-System-for-AI-Factories/default.aspx), CONFIRMED). Engines emit KV events so routers know what each replica holds ([vLLM tiered offload post](https://vllm.ai/blog/2026-09-10-tiered-kv-offloading), CONFIRMED).
- **Mooncake** (Kimi): a global KV pool plus a KV-cache-aware scheduler; effective request capacity up 59%-498% under SLOs, and 115% / 107% more requests on A800 / H800 clusters in production; >100B tokens/day on thousands of nodes ([FAST'25 best paper coverage](https://www.tsinghua.edu.cn/en/info/1245/14138.htm), REPORTED; paper [arXiv 2407.00079](https://arxiv.org/abs/2407.00079) *(prior)*).
- Real-world hit-rate anchor: DeepSeek's production traffic had a **56.3% KV cache hit rate** (of 608B input tokens/day) ([DeepSeek inference system overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md), CONFIRMED).

---

## 3. KV-cache memory management

### 3.1 How big is the KV cache? (formulas for the toy)

```
KV_bytes_per_token = 2 (K and V) x n_layers x n_KV_heads x head_dim x bytes_per_element      # MHA / GQA / MQA
KV_bytes_per_token(MLA) = n_layers x (d_c + d_rope) x bytes_per_element                      # one shared latent
```

Careful: the heads term is **KV heads**, not query heads. Worked numbers (ESTIMATE, computed here):

| Model style | Calculation | Bytes/token | 1M tokens |
|---|---|---|---|
| Llama-2-7B, MHA (32 layers, 32 KV heads, d=128, BF16) | 2x32x32x128x2 | 524,288 (0.5 MB) | 524 GB |
| Llama-3-70B, GQA (80 layers, 8 KV heads, d=128, BF16) | 2x80x8x128x2 | 327,680 (320 KB) | 328 GB |
| DeepSeek-V3/Kimi-K2 MLA (61 layers, d_c=512, d_rope=64, BF16) | 61x(512+64)x2 | 70,272 (~70 KB) | 70 GB |
| V3.2-style MLA, FP8 cache incl. scales (656 B per layer-token, from FlashMLA layout *(prior)*) | 61x656 | ~40 KB | ~40 GB |
| DeepSeek-V4-Pro (compressed attention; see section 8) | formula-derived | ~4-12 KB | ~4-12 GB |

Flag: the HF continuous-batching article text, as returned by my fetch tool, quotes "16 KB per token" for Llama-2-7B; the correct fp16 MHA number is 512 KB (2x32x32x128x2 B). Treat HF's figure as a typo/garble and use 512 KB. Kimi-K2.x uses MLA with the same 61-layer backbone, so ~70 KB BF16 per token ([Kimi K2.6 specs, REPORTED](https://www.gmicloud.ai/en/blog/kimi-k2-6-architecture-benchmarks-and-what-it-means-for-production-ai)).

Takeaway for visuals: GQA -> MLA -> compressed/sparse attention is a ~5x then ~10x reduction in the thing that limits concurrency at long context.

### 3.2 PagedAttention (vLLM)

*(prior)* [Kwon et al., SOSP'23, arXiv 2309.06180](https://arxiv.org/abs/2309.06180). Before: each request reserved a contiguous max-length KV buffer -> internal fragmentation (reserved but unused), external fragmentation, and no sharing; the paper measured only ~20-40% of KV memory holding useful data. PagedAttention splits KV into fixed-size **blocks** (e.g., 16 tokens) allocated on demand from a free pool, with a per-request **block table** (logical block i -> physical block j), like OS virtual memory. Waste falls to under ~4% (last partially filled block only), and blocks can be **shared** copy-on-write (parallel sampling, beam search, common prefixes). Reported 2-4x throughput over FasterTransformer/Orca at equal latency. Now the default memory model in vLLM, SGLang and TensorRT-LLM (the latter calls it paged KV cache). Check the exact percentages against the paper before quoting.

### 3.3 Prefix caching and RadixAttention

- **Automatic prefix caching (vLLM)**: hash each full block by (parent hash, token ids); a new request reuses any blocks whose hashes match, skipping their prefill. LRU eviction of unreferenced blocks.
- **RadixAttention (SGLang)**: keep all cached KV in a **radix tree** keyed by token sequences, so any shared prefix (system prompt, few-shot examples, multi-turn history, tree-of-thought branches) is matched by tree walk with LRU leaf eviction ([SGLang paper, arXiv 2312.07104](https://arxiv.org/abs/2312.07104) *(prior)*). Good animation: a trie that grows as requests arrive, with matched nodes highlighted and "tokens skipped" counter.
- Agent workloads (long, append-only histories, tool loops) are why this is now table stakes and why routers are cache-aware (2.4).

### 3.4 Offload and tiering

Evicted KV does not have to be thrown away; recomputing a long prefix costs far more than reloading it.

- vLLM **tiered KV offloading** (post of 2026-09-10; framework available since v0.22): HBM -> host DRAM (a real LRU/ARC cache) -> secondary tiers (filesystem, S3-compatible via NIXL, peer-to-peer RDMA). All data flows through host memory; content-addressed naming lets KV be shared across nodes and warm-start new instances. In their Qwen-35B / 2xH100 test, up to 64 conversations fit in HBM; 64-128 needed CPU offload; beyond 128 storage offload "more than doubled throughput" ([vLLM tiered KV](https://vllm.ai/blog/2026-09-10-tiered-kv-offloading), CONFIRMED). Earlier: [KV offloading connector, Jan 2026](https://vllm.ai/blog/2026-01-08-kv-offloading-connector) (CONFIRMED to exist).
- **LMCache**, **Mooncake Store**, **Dynamo KVBM** (KV Block Manager: offload to CPU/SSD/network storage; [NVIDIA May 2025 blog](https://developer.nvidia.com/blog/nvidia-dynamo-accelerates-llm-d-community-initiatives-for-advancing-large-scale-distributed-inference), CONFIRMED), llm-d's tiered prefix cache ([docs](https://llm-d.ai/docs/well-lit-paths/foundations/tiered-prefix-cache), REPORTED), and Novita's PegaFlow external KV (vLLM blog 2026-05-18, title CONFIRMED).
- Hardware that helps: GB200/GB300 Grace-Blackwell NVLink-C2C gives fast CPU<->GPU transfer; vLLM's "weight offloading v2" for prefill instances exploits it ([vLLM GB200](https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1), CONFIRMED).
- Hybrid offload of *sparse* KV: "GLM 5.3 ... Hybrid HiSparse Offloading in vLLM" (2026-09-08, title only, UNVERIFIED content).

### 3.5 KV-cache quantization

- vLLM `--kv-cache-dtype fp8` (e4m3) halves KV bytes and runs QK and ScoreV in FP8. Findings ([vLLM FP8 KV post, 2026-04-22](https://vllm.ai/blog/2026-04-22-fp8-kvcache), CONFIRMED): a naive implementation dropped 128K needle-in-haystack accuracy from 91% to 13%; an accumulation fix restored near-BF16 (Llama-3.3-70B ~97-98% of baseline AUC@128k; Qwen3.5-27B matches at 1M); reasoning drops at most 1-2 points; decode ITL slope falls to ~54% of BF16 (Llama-3.1-8B); +14.9% throughput (Llama) / +4.8% (gpt-oss-20b); break-even ~7K tokens; avoid for short contexts; some models (Kimi-K2.5) need calibration.
- Teaching point: a quantization that looks harmless on short benchmarks can fail catastrophically at long context because errors accumulate across many attended tokens.

### 3.6 How public prompt-caching discounts relate

API "prompt caching" is prefix-KV reuse exposed as a price. Prefill for a cache hit is skipped, so the provider saves compute, but must hold the KV somewhere (HBM/DRAM/SSD) for a TTL, which is why writes cost *more* than base input and reads cost far less:

| Provider / model | Base input | Cache write | Cache read | Output |
|---|---|---|---|---|
| Anthropic, most models | 1x | 1.25x (5 min), 2x (1 h) | **0.1x** | 5x (e.g., Sonnet 5.5 $2 in / $10 out) |
| Anthropic Opus 5.5 | $4/M | $5 / $8 | 0.05x ($0.20) | $20 |
| Anthropic Fable 5.1 | $10/M | $12.5 / $20 | 0.025x ($0.25) | $50 |
| DeepSeek V4-Flash (off-peak) | $0.15 | n/a (no write fee) | $0.003 (2% of miss) | $0.60 |
| DeepSeek V4-Pro (off-peak) | $0.66 | n/a | $0.022 (3.3% of miss) | $1.98 |

Sources: [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing) (CONFIRMED, read 2026-10-07); [DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing) (CONFIRMED; peak rates are 2x off-peak, peak = 01-04 and 06-10 UTC Mon-Fri). OpenAI pricing: page returned 403, UNVERIFIED (omit). Why the exact discount sizes are chosen is a business decision, not published; the drop to 0.05x/0.025x on newer Anthropic models suggests cheaper KV retention/retrieval (ANALYSIS, not stated by the vendor). Note Claude 4.7+ tokenizers produce ~30% more tokens for the same text, so per-token prices are not apples-to-apples across generations (Anthropic pricing page, CONFIRMED).

---

## 4. Speculative decoding

### 4.1 Mechanism and math

Decode is memory-bound, so verifying k extra tokens in one forward pass costs little more than generating 1. Draft-verify loop ([Leviathan et al. arXiv 2211.17192](https://arxiv.org/abs/2211.17192); [Chen et al. 2302.01318](https://arxiv.org/abs/2302.01318), *(prior)*): a cheap drafter proposes k tokens; the target scores all k+1 positions in one pass; tokens are accepted left-to-right with rejection sampling; on the first rejection, resample from a corrected distribution and stop. The output distribution is **exactly** the target's (vLLM: "preserves the verifier model's output distribution exactly via rejection sampling", [vLLM spec-decode post](https://vllm.ai/blog/2026-07-28-speculators-parallel-drafting), CONFIRMED).

If each draft token is accepted independently with probability alpha:

```
E[tokens per target step] = (1 - alpha^(k+1)) / (1 - alpha)                (always >= 1; the +1 is the bonus/resampled token)
speedup ~ E[tokens] / (1 + k*c)       c = cost of one draft step / cost of one target step
```

Table (ESTIMATE, computed): expected tokens per step

| alpha | k=1 | k=2 | k=3 | k=5 |
|---|---|---|---|---|
| 0.50 | 1.50 | 1.75 | 1.88 | 1.97 |
| 0.70 | 1.70 | 2.19 | 2.53 | 2.94 |
| 0.85 | 1.85 | 2.57 | 3.19 | 4.15 |
| 0.90 | 1.90 | 2.71 | 3.44 | 4.69 |

Diminishing returns in k; with alpha=0.7 going from k=3 to 5 adds only 0.4 tokens but costs two more draft steps. Real alpha is not constant per position (it decays with depth) and depends on the task (code/boilerplate high, creative text low). **Caveat that matters for serving**: speedup shrinks as batch grows, because at large batch the target step is no longer memory-bound (verification of k extra tokens per request costs real FLOPs). EAGLE-3's own SGLang number is a 1.38x throughput gain at batch 64 versus up to 6.5x peak speedup at small batch ([EAGLE-3 abstract](https://arxiv.org/abs/2503.01840), CONFIRMED). Spec decoding is therefore mostly a **latency (per-user speed)** tool for interactive/low-batch regimes.

### 4.2 Families

| Method | Idea | Status |
|---|---|---|
| Separate small draft model | e.g., 1B drafting for 70B; must share tokenizer | Classic; acceptance typically lower than feature-based drafters (REPORTED) |
| n-gram / prompt-lookup | Copy-from-context drafting, no model | Cheap, great for RAG/code edits *(prior)* |
| Medusa | Extra decoding heads on the target predict future tokens; tree verification | [arXiv 2401.10774](https://arxiv.org/abs/2401.10774) *(prior)* |
| EAGLE-1/2/3 | Small autoregressive head over the target's hidden features; EAGLE-2 dynamic draft tree; EAGLE-3 drops feature regression, uses multi-layer feature fusion + training-time test. ~1.4x better than EAGLE-2, up to 6.5x vs vanilla (paper) | EAGLE-3 CONFIRMED (abstract); shipping in vLLM, SGLang, TensorRT-LLM and Vertex AI ([LMSYS x Vertex](https://lmsys.org/blog/2025-12-01-eagle3-vertex/), REPORTED) |
| MTP heads | Multi-token-prediction module trained with the model, reused as a drafter. DeepSeek-V3 predicts 1 extra token; 85-90% acceptance of the second token, ~1.8x TPS ([DeepSeek-V3 report](https://arxiv.org/pdf/2412.19437), CONFIRMED via excerpt) | Standard in DeepSeek-family and several 2026 open models (GLM-5 report cites it; REPORTED) |
| P-EAGLE (parallel drafting) | Drafter emits all k tokens in **one** forward pass, removing the sequential-draft ceiling; up to 1.69x over EAGLE-3 on B200 ([vLLM, 2026-03-13](https://vllm.ai/blog/2026-03-13-p-eagle), CONFIRMED); pre-trained heads for gpt-oss-120B/20B, Qwen3-Coder-30B |
| EAGLE 3.1 | Fixes "attention drift" (drafter attention moves off sink tokens as depth grows) with FC norm + post-norm hidden states; more robust to chat templates/long context ([vLLM, 2026-05-26](https://vllm.ai/blog/2026-05-26-eagle-3-1), CONFIRMED) |
| DFlash, DSpark | Block-diffusion-style parallel drafters; DSpark adds autoregressive correction and confidence scores ([vLLM, 2026-07-28](https://vllm.ai/blog/2026-07-28-speculators-parallel-drafting), CONFIRMED; no numeric speedups in the text I read) |

### 4.3 What is mainstream in 2026 production

- CONFIRMED: vLLM's mid-2026 blog frames parallel drafters (P-EAGLE, DFlash, DSpark) as state of the art versus autoregressive EAGLE-3, and they remove the need to retune speculation length as server load changes.
- CONFIRMED: MTP is used in practice at scale: lmsys reports MTP raising per-user throughput +87% for DeepSeek-R1 on GB300 NVL72 at 128K/8K with peak system throughput maintained ([LMSYS GB300](https://lmsys.org/blog/2026-02-19-gb300-longctx)).
- UNVERIFIED: exact share of public API traffic using spec decoding; closed providers do not publish it. Safe claim: self-drafting heads (MTP/EAGLE-family) are the norm in open-model serving; separate draft models are a legacy option.

---

## 5. Quantization for inference

| Scheme | What is quantized | Notes / evidence |
|---|---|---|
| **Weight-only INT4/INT3 (GPTQ, AWQ)** | Weights to 4 bit, activations stay 16-bit | GPTQ = second-order layer-wise rounding ([2210.17323](https://arxiv.org/abs/2210.17323)); AWQ = protect salient channels via activation-aware scaling ([2306.00978](https://arxiv.org/abs/2306.00978)) *(prior)*. Cuts bytes read in decode ~4x, big win when memory-bound; less useful at large compute-bound batches because math still runs in 16-bit |
| **FP8 W8A8** | Weights and activations to FP8 (E4M3, block scales) | Hopper+; native in DeepSeek V3/V4 checkpoints (FP8 block 128x128, UE8M0 scales; [V4-Pro config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/raw/main/config.json), CONFIRMED). Roughly 2x vs BF16 compute and bytes; quality loss usually small |
| **MXFP4** | 4-bit E2M1 values, 32-element blocks, power-of-two (E8M0) scale | OCP microscaling standard. **gpt-oss shipped post-trained with MXFP4 on the MoE weights**; 117B/5.1B-active model fits one 80 GB GPU ([model card](https://huggingface.co/openai/gpt-oss-120b), CONFIRMED) |
| **NVFP4** | 4-bit E2M1, **16-element** blocks, FP8 (E4M3) scale + FP32 per-tensor scale (~4.5 bits/value) | Finer scaling -> smaller error than MXFP4; Blackwell tensor cores run it natively, ~9 PFLOPs dense on B200 ([NVIDIA NVFP4 blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference), CONFIRMED via snippet) |
| **INT4 QAT** | Quantization-aware training | Kimi K2 Thinking/K2.5/K2.6 ship native INT4 MoE weights ([REPORTED](https://docs.vultr.com/inference-cookbook/rocm/model-guides/kimi-k2-5-1t)); InferenceX serves an NVFP4 requantization of those weights on B200 for full-rate FP4 cores (CONFIRMED) |
| **FP4 expert + FP8 rest** | Mixed | DeepSeek-V4: experts FP4, other params FP8 with QAT; Pro checkpoint ~865 GB, Flash ~160 GB ([model card summary](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro), CONFIRMED for FP4+FP8; sizes REPORTED) |
| **KV cache FP8** | See 3.5 | Accuracy cliffs at long context without the right kernels |

Why it matters: bytes drive decode (BF16 -> FP4 = 4x fewer weight bytes) and compute drives prefill (FP4 tensor cores 2x FP8). Measured effect of precision plus hardware: Kimi K2.5 B200 NVFP4 vs H200 INT4 gives 2.75-2.95x lower $/M tokens, decomposed by the source into 1.67x HBM bandwidth, FP4 tensor-core unlock, and lower TP overhead (InferenceX, CONFIRMED). vLLM's GB200 DeepSeek stack used NVFP4 GEMM for MoE experts and output projections, FP8 for MLA projections, and **NVFP4 dispatch (quantize activations before the all-to-all, ~4x less comm bandwidth)** (vLLM GB200 post, CONFIRMED).

Quality: QAT/post-training in the target format (gpt-oss, Kimi, DeepSeek-V4) loses little; naive post-training INT4/FP4 of an FP16-trained model is riskier, especially MXFP4 on larger models relative to NVFP4 (NVIDIA claim, REPORTED vendor). I found no neutral head-to-head benchmark; UNVERIFIED.

---

## 6. Disaggregation and parallelism

### 6.1 Prefill/decode (P/D) disaggregation

Idea: run prefill and decode on **different GPU pools**, transfer the KV cache between them. Gains: no prefill-induced decode stalls (TPOT), each pool gets its own parallelism and batch size, hardware can differ. Costs: KV transfer bandwidth/latency, two pools to size and autoscale, more complexity.

- **Splitwise** (Microsoft; [arXiv 2311.18677](https://arxiv.org/abs/2311.18677)): 1.4x throughput at 20% lower cost, or 2.35x throughput at same cost/power (REPORTED via secondary summary).
- **DistServe** ([arXiv 2401.09670](https://arxiv.org/abs/2401.09670)): goodput framing; 7.4x more requests or 12.6x tighter SLOs (REPORTED via summary).
- **Mooncake** (Kimi) adds a distributed KV pool (2.4).
- **KV transfer plumbing**: **NIXL** (NVIDIA Inference Xfer Library) moves KV over NVLink, InfiniBand/RoCE RDMA, with TCP fallback; used by Dynamo, llm-d, vLLM, Ray Serve LLM (NVIDIA May 2025 blog; vLLM Dec 2025 post, CONFIRMED).
- **When it helps most**: MoE with expert parallelism, because a single compute-bound prefill in the EP group delays the whole group's forward pass ([vLLM large-scale serving](https://vllm.ai/blog/large-scale-serving), CONFIRMED). vLLM published a practical guide "Taking vLLM Apart: Disaggregated Serving" (2026-09-29; title CONFIRMED, body not read) and "Disaggregated Serving for Hybrid SSM Models" (2026-04-21, title only).
- **When it hurts**: short prompts, small models, or slow interconnect, where KV transfer dominates; and prefill/decode ratios that shift with traffic. In vLLM's GB200 setup the best config was 4 prefill instances of **2 GPUs** each plus 1 decode instance of 8 GPUs = 16 GPUs, because shrinking compute-bound prefill groups removes collective overhead (vLLM GB200 post, CONFIRMED).

### 6.2 Tensor parallel (TP) vs expert parallel (EP) for MoE

- **TP**: shard every matrix across GPUs; two all-reduces per layer; excellent for latency at small scale, but each GPU still holds a slice of *every* expert, so per-expert batches stay small and communication grows with TP size.
- **EP**: each GPU holds whole experts; tokens are dispatched by all-to-all to the GPUs owning their chosen experts and combined afterwards. Larger EP -> fewer experts per GPU -> **less weight memory per GPU, more KV room, and a larger effective batch per expert** (many users' tokens pooled). Costs: all-to-all traffic and load imbalance (hot experts).
- **Attention under EP**: attention is usually run **data-parallel** (each GPU handles its own requests with its own KV; MLA's single latent head makes TP of attention wasteful), then MoE layers do all-to-all across the EP group. This "DP attention + EP MoE" combo is what vLLM calls Wide-EP: "a single set of experts are shared across ranks" ([vLLM](https://vllm.ai/blog/large-scale-serving), CONFIRMED).
- Supporting tricks: **DeepEP** (low-latency dispatch/combine kernels), **DeepGEMM**, **EPLB** (redundant experts + rebalancing from observed load), **dual-batch / two-batch overlap** (split the batch into two micro-batches so one's communication hides under the other's compute).

### 6.3 Reference deployments (all CONFIRMED unless noted)

| System | Setup | Result |
|---|---|---|
| **DeepSeek V3/R1 production** (Feb 2025) | Prefill EP32 over 4 nodes (9 routed + 1 shared expert/GPU); decode EP144 over 18 nodes (2 routed + 1 shared/GPU); 8 of 256 experts active; dual-batch overlap in prefill, 5-stage pipeline in decode; 3 load balancers; FP8 matmuls/dispatch, BF16 MLA | 24 h: 608B input tokens (56.3% cache hits), 168B output; per H800 node ~73.7K input tok/s (prefill), ~14.8K output tok/s (decode); avg KV length 4,989 tokens; 226.75 avg / 278 peak nodes; cost $87,072/day at $2/GPU-hr vs theoretical $562,027 revenue at R1 prices = "545% cost profit margin" (theoretical; free web/app and discounts make real revenue lower) [source](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) |
| **SGLang open reproduction** (May 2025) | 96 H100 (12 nodes), PD disaggregation, DeepEP+DeepGEMM, TBO, EPLB with 288 experts (256+32 redundant) | 52.3K input / 22.3K output tok/s per node (2K input); ~$0.20/M output tokens; up to 5x output throughput over TP16; EPLB 1.49x prefill / 2.54x decode; TBO +27-35% prefill [LMSYS](https://lmsys.org/blog/2025-05-05-large-scale-ep/) |
| **SGLang on GB200 NVL72** (Sep 2025) | FP8 attention + NVFP4 MoE, PD + large EP | 26,156 input and 13,386 output tok/s/GPU (2K input): 3.8x prefill, 4.8x decode over H100 [LMSYS](https://lmsys.org/blog/2025-09-25-gb200-part-2) |
| **vLLM wide-EP on H200** (Dec 2025) | CoreWeave H200 + InfiniBand, DBO, EPLB, DeepEP, DeepGEMM | 2.2K tok/s/H200 sustained (up from ~1.5K) [vLLM](https://vllm.ai/blog/large-scale-serving) |
| **vLLM on GB200** (Feb 2026) | 4x2-GPU prefill + 1x8-GPU decode, NVFP4 | 26.2K prefill / 10.1K decode tok/s/GPU at 2K/2K, 3-5x H200 baseline (which used 48 GPUs) [vLLM](https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1) |
| **InferenceX DeepSeek-V4-Pro** (May 2026) | Dynamo + vLLM, disaggregated, FP4, GB200 EP=8 vs GB300 EP=16 | See 1.3; GB300 extra HBM allowed wider EP and larger prefill batches [source](https://inferencex.semianalysis.com/blog/gb300-nvl72-vs-gb200-nvl72-dsv4-pro-vllm-fp4) |

Observation for a chart: across 2025 -> 2026, per-GPU decode throughput on DeepSeek-style models went from ~1.5-2.2K (H200) to ~10-13K (GB200) tok/s/GPU at 2K contexts and short outputs, a 4-6x jump from FP4 + 8 TB/s + NVL72's 72-GPU NVLink domain making wide EP cheap. These numbers are at specific interactivity levels and sequence lengths; do not compare across rows naively (DeepSeek's own number is a 24-hour production average at ~20-22 tok/s/user, 4,989-token average context).

### 6.4 Why rack-scale NVL72 matters

Wide EP needs all-to-all every layer. Inside an NVLink domain (130 TB/s aggregate per GB300 NVL72, REPORTED) that all-to-all is far cheaper than across InfiniBand nodes; 72 GPUs x 192-288 GB means the whole 865 GB V4-Pro fits across a rack's worth of EP with room for KV. Per-GPU memory also decides how wide EP can go: InferenceX attributes GB300's 2.83x per-GPU throughput at 27 tok/s/user to 288 GB enabling EP=16 and larger prefill batches (CONFIRMED).

---

## 7. The production stack, Oct 2026

### 7.1 Engines

| Engine | Strengths | Evidence |
|---|---|---|
| **vLLM** | Broadest model/hardware coverage, PagedAttention origin, active large-scale-serving work (wide-EP, DBO, EPLB, tiered KV, P/D, hybrid/SSM, FP8 KV, parallel spec decoding), releases ~v0.2x by mid-2026 | Own blog (26 posts in 2026, CONFIRMED); co-authors from Meta, NVIDIA, Red Hat, IBM, AWS; InferenceX uses vLLM v0.21.0 for several runs (CONFIRMED) |
| **SGLang** | RadixAttention, strong on multi-turn/agent prefix reuse and DeepSeek-style wide-EP; often first with day-0 recipes. Spun out as company **RadixArk** (valued ~$400M per TechCrunch Jan 2026; $100M seed led by Accel, co-led by Spark, with NVentures, AMD; public launch May 2026) ([TechCrunch](https://techcrunch.com/2026/01/21/sources-project-sglang-spins-out-as-radixark-with-400m-valuation-as-inference-market-explodes), REPORTED via search snippet) | "400,000+ GPUs, xAI default engine, trillions of tokens/day" is from SEO blogs, REPORTED only |
| **TensorRT-LLM** | NVIDIA's engine; best kernels first on new NVIDIA silicon; has a PyTorch-based runtime, wide-EP support via Dynamo, DeepSeek-V4 NVFP4 checkpoint loading (v1.3.0rc23) and V4-Pro GB300 sanity tests ([release notes](https://newreleases.io/project/github/NVIDIA/TensorRT-LLM/release/v1.3.0rc23), REPORTED) | Claimed 8-13% over vLLM in third-party blog, REPORTED/SEO; treat benchmark rankings as workload-specific. For neutral numbers use InferenceX |
| Others | LMDeploy, TGI (Hugging Face), llama.cpp / Ollama (local), MLC; proprietary in-house stacks at the big labs and clouds (not public) | UNVERIFIED for relative share |

**Honest caveat for pages**: no reliable public market-share number exists. The strongest verifiable signals of who matters: Dynamo 1.0's adopter list (AWS, Azure, Google Cloud, OCI, CoreWeave, Together, Nebius, Alibaba Cloud; Cursor, Perplexity, Baseten, Deep Infra, Fireworks, ByteDance, Meituan, PayPal, Pinterest; NVIDIA press release, CONFIRMED that NVIDIA says so); llm-d accepted into CNCF Sandbox on 2026-03-12; InferenceX being sponsored/used by vLLM, SGLang, OpenAI, Microsoft, Meta etc.

### 7.2 Orchestration, routing, autoscaling

- **NVIDIA Dynamo 1.0**: GA 2026-03-16 at GTC. Components: router (KV-aware), workers wrapping vLLM / SGLang / TensorRT-LLM in prefill or decode roles, **NIXL**, **KVBM**, **Planner** (scales P and D pools from input/output sequence-length mix instead of raw QPS), **Grove** (cluster scaling), TensorRT-LLM kernels contributed to FlashInfer; integrates with llm-d, LMCache, LangChain ([press release](https://investor.nvidia.com/news/press-release-details/2026/NVIDIA-Enters-Production-With-Dynamo-the-Broadly-Adopted-Inference-Operating-System-for-AI-Factories/default.aspx); [May 2025 blog](https://developer.nvidia.com/blog/nvidia-dynamo-accelerates-llm-d-community-initiatives-for-advancing-large-scale-distributed-inference), CONFIRMED). "Up to 7x on Blackwell" is a vendor claim with unspecified benchmark: REPORTED.
- **llm-d**: Kubernetes-native distributed inference on vLLM (and SGLang), launched at Red Hat Summit May 2025 with Google, IBM, NVIDIA, CoreWeave; CNCF Sandbox 2026-03-12; v0.7 May 2026 (predicted-latency scheduling GA, batch gateway experimental); features: inference scheduler on Gateway API Inference Extension, prefix-aware routing, P/D disaggregation, hierarchical KV offload, traffic- and hardware-aware autoscaling ([CNCF](https://www.cncf.io/blog/2026/03/24/welcome-llm-d-to-the-cncf-evolving-kubernetes-into-sota-ai-infrastructure/), [docs 0.7](https://llm-d.ai/docs/0.7), REPORTED via snippets; vLLM's own post names llm-d, Dynamo and Ray Serve LLM as the three deployment frameworks, CONFIRMED).
- **Autoscaling insight**: scale on queue depth / token load / SLO attainment, not CPU or QPS; P and D pools scale independently; cold start of a 1T-parameter model is minutes (weight loading: vLLM published "Large-Scale Sharded Weight Transfer with Ray Direct Transport", title only, UNVERIFIED body), so providers keep warm headroom and warm-start from shared KV/weights storage.

### 7.3 Multi-tenancy and LoRA

Many fine-tunes of one base model are served by keeping the base weights resident and swapping small low-rank adapters per request in the same batch (S-LoRA arXiv 2311.03285, Punica arXiv 2310.18547 *(prior)*; supported in vLLM/SGLang as multi-LoRA). Tenant isolation, per-tenant rate limits and priority classes sit in the gateway/scheduler. llm-d's v0.5 multi-tenant SaaS benchmark claims near-zero added latency up to ~120K tokens/s (REPORTED snippet).

### 7.4 Pricing: why output costs more than input

Anthropic: output = **5x** input across current tiers (Haiku 4.5 $1/$5; Sonnet 5.5 $2/$10; Opus 5.5 $4/$20; Fable 5.1 $10/$50). DeepSeek V4-Flash 4x ($0.15/$0.60 off-peak), V4-Pro 3x ($0.66/$1.98). Batch API: 50% off both; prompt-cache reads 0.1x (CONFIRMED, pages above).

Why (ANALYSIS, not vendor-stated, but follows from sections 1-3):
1. Input tokens are processed in parallel at high arithmetic intensity (compute-bound prefill at high MFU); output tokens are produced one step at a time at low intensity, so each GPU-second yields far fewer output tokens than input tokens.
2. Each output token also holds KV memory for the request's entire lifetime and ties up a batch slot.
3. Output latency is user-visible and so is provisioned with headroom (low batch -> lower utilization), while input can be queued/batched. Illustration: DeepSeek's own production node did ~73.7K input vs ~14.8K output tokens/s per node, a ~5x ratio (CONFIRMED), close to the 3-5x price ratios.
4. Prefill also benefits from cache hits (56% at DeepSeek), decode never does.

Other pricing levers: peak/off-peak (DeepSeek halves prices off-peak, CONFIRMED), "fast mode" at premium for lower latency (Opus 5.5 fast: $8/$40, CONFIRMED) i.e., sell lower batch size/more speculation as higher price per token, regional pinning +10%, and **1M context at flat per-token price** for Claude 4.6+ (CONFIRMED) even though attention cost and KV grow with length, which tells you how much architectural compression (section 3.1) matters to providers.

### 7.5 Long-context (1M token) serving concerns

1. **KV size per sequence** (3.1): 1M tokens = 328 GB in a Llama-3-70B-like GQA model, ~70 GB MLA BF16, ~4-12 GB with V4-style compression. One request can exceed a GPU's HBM.
2. **Prefill cost**: linear part ~`2 x active_params x N` FLOPs plus attention O(N^2) (reduced by sparse/compressed attention). A 128K prompt took 8.6 s TTFT on a GB300 NVL72 with chunked pipeline-parallel prefill (CONFIRMED); 1M would be tens of seconds to minutes (ESTIMATE). Mitigations: chunked prefill, context/sequence parallelism, chunked PP, offloading, caching.
3. **Decode cost** grows with N because every step reads the KV: at 1M and 70 KB/token that is 70 GB/step per sequence (~9 ms at 8 TB/s for one sequence), which kills batching; hence sparse/compressed attention (DeepSeek DSA in V3.2-Exp cut API prices 50%+ ([DeepSeek](https://deepseek.com/en/news/v3-2-exp/), CONFIRMED headline); V4's CSA/HCA gives ~10% of V3.2's KV and ~27% of its FLOPs at 1M ([HF card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro), CONFIRMED)).
4. **Concurrency collapse**: free HBM / KV-per-sequence sets max batch; FP8 KV, KV offload, hybrid sliding-window/SSM layers all raise it. Example: lmsys GB300 gets ~40 concurrent 128K requests per GPU vs 24 on GB200 (CONFIRMED).
5. **Prefix caching is essential**: agents resend huge histories; without cache reuse, cost scales with total re-sent tokens.
6. **Accuracy hazards**: FP8 KV long-context regression (3.5).

---

## 8. Worked example: ~1T-param / ~50B-active MoE with 1M context (ESTIMATE)

Use a real model that matches: **DeepSeek-V4-Pro**: 1.6T total, 49B active, 1M context, released 2026-04-24 under MIT; experts FP4, rest FP8; checkpoint ~865 GB ([HF card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro); config: 61 layers, hidden 7,168, 128 heads, **1 KV head**, head_dim 512 (RoPE 64), 384 routed + 1 shared experts, 6 routed per token, MoE intermediate 3,072, sliding window 128, compression ratios alternating 128/4, CONFIRMED from [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/raw/main/config.json)). (Alternative reference model: Kimi K2.x, 1T/32B active, 61 layers, MLA, 384 experts, 256K context, INT4 QAT, REPORTED.)

### 8.1 Weights

```
weights_bytes = sum(params x bits/8)
```
- V4-Pro stored: ~865 GB (REPORTED size). Cross-check: 865 GB / 1.6T params = 0.54 B/param, consistent with mostly-FP4 experts (4.25-4.5 bits incl. scales) plus FP8 elsewhere.
- If the same 1.6T were all FP8: ~1.6 TB; all BF16: ~3.2 TB (the Substack deep dive states 3.2 TB BF16, REPORTED); all NVFP4: ~0.9 TB.
- Per GPU with EP (experts sharded; attention/dense replicated, small): EP=8: ~108 GB; EP=16: ~54 GB; EP=72: ~12 GB (+ replicated non-expert weights, unknown, ignore).
- Fit: GB200 (192 GB): EP=8 leaves ~84 GB/GPU for KV + activations; GB300 (288 GB): EP=16 leaves ~230 GB. 8x B200 node (~1.4 TB) holds the weights with ~500 GB spare.
- Minimum GPUs just for weights: ceil(865/288) = 3 GB300, ceil(865/192) = 5 GB200. Practical replica: 8-16 GPUs (one NVL72 can host several replicas or one very wide-EP replica).

### 8.2 KV per token and at 1M context

Baseline for comparison, V3.2-style MLA at FP8 ~40 KB/token -> 40 GB per 1M tokens (formula in 3.1). V4-Pro keeps one KV head of dim 512 per layer but stores only **compressed entries**: a layer with ratio r keeps N/r entries (CSA r=4, HCA r=128). Estimate per 1M-token sequence, 512 values per entry (assuming K and V share the stored vector; if not, double):

```
entries = N x ( n_CSA_layers / 4  +  n_HCA_layers / 128 )
KV_bytes = entries x 512 x bytes_per_element
```
- Config says alternating 128/4 across 61 layers (~30 + 30): entries = 1e6 x (30/4 + 30/128) = 7.73M -> **3.96 GB (FP8) / 7.9 GB (BF16)**.
- Blog summaries say a 3:1 CSA:HCA mix (45 + 15): 11.4M entries -> **5.8 GB (FP8) / 11.6 GB (BF16)**. The two sources disagree (UNVERIFIED which is right); the answer lands at ~4-12 GB either way.
- Sanity check against the model card: "10% of V3.2's KV cache at 1M" -> 10% x 40 GB = ~4 GB. My FP8 numbers (4-6 GB) land within about 1.5x. Not included: the 128-token sliding window (negligible), the FP4 lightning-indexer keys for CSA (small, maybe a few hundred MB per 1M tokens), and paging overhead.
- Per token: ~4-12 KB, versus 320 KB for a Llama-3-70B-style GQA model at the same context: ~30-80x smaller.

### 8.3 Concurrency and a replica sketch

```
max_concurrent_seqs_per_GPU ~ (HBM - weights_share - activations) / KV_per_seq     (with attention data-parallel: each GPU holds its own sequences' KV)
```
- GB300, EP=16: free ~200 GB / ~4-12 GB per 1M-token sequence -> ~17-50 full-length sequences per GPU. At 8K context (~35-100 MB each) the KV is not the constraint; the binding limits become per-user latency target and all-to-all/compute.
- Check against a measured anchor: lmsys DeepSeek-R1 128K/8K on GB300 holds ~40 requests/GPU (MLA ~35-70 KB/token x 128K = 4.5-9 GB each -> roughly 200+ GB), consistent with this method (LMSYS GB300, CONFIRMED).

### 8.4 Throughput and latency ballpark

Roofline per decode step with wide EP (ESTIMATE): all experts touched at large batch, so `t_step ~ weights_per_GPU/HBM_BW + KV_read/HBM_BW + comm`:
- GB300, EP=16: 54 GB / 8 TB/s = 6.8 ms of weight read; GB200, EP=8: 108 GB / 8 TB/s = 13.5 ms; H200, EP=8: 22.5 ms (but H200 cannot hold FP4 natively).
- A 27 tok/s/user target allows 37 ms per token, so comms, MoE kernel inefficiency, attention and scheduling use most of the budget beyond the pure weight-read floor; real step time is ~3-5x the floor. (ESTIMATE; this ratio is why measured numbers sit far below the roofline.)
- Measured anchor (InferenceX, ISL 8K/OSL 1K, V4-Pro, FP4, disaggregated Dynamo+vLLM): **GB300 6,182 tok/s/GPU at 27 tok/s/user ($0.12/M tokens at $2.65/GPU-hr); GB200 2,189 tok/s/GPU ($0.28/M at $2.21/hr); peak throughput 11,056 (GB300) / 8,933 (GB200) tok/s/GPU at ~13-15 tok/s/user.** Implied concurrency ~ 6,182/27 ~ 229 active streams per GPU (if the metric counts only output tokens; if it counts input+output, as the cost formula suggests, divide by ~9 at 8K:1K, ~25 streams per GPU; the source does not make this explicit, UNVERIFIED).
- 1M-context regime (ESTIMATE): prefill linear FLOPs ~ 2 x 49e9 x 1e6 ~ 98 PFLOP plus attention; at, say, 8 GPUs x ~5 PFLOPs effective FP4/FP8 mixed -> ~2.5-10 s lower bound, more realistically tens of seconds with attention and chunking; decode throughput per GPU falls well below the 8K numbers, limited by concurrency and KV reads, roughly 2-10x lower. Anchor: R1 128K/8K on GB300 gets 226 tok/s/GPU peak (LMSYS, CONFIRMED) versus ~thousands at 8K, a ~10-25x drop for 16x longer input.
- Cost sanity: at $0.12-0.28 per M (8K/1K) the provider cost is below DeepSeek's off-peak list price for V4-Pro ($0.66 input / $1.98 output), consistent with the large listed margin DeepSeek reported in 2025 (theoretical).

---

## 9. Teaching notes: toys, formulas, and existing explainers

### 9.1 Toy ideas with the formulas each needs

| Concept | Toy | Formulas / state |
|---|---|---|
| Two phases | Roofline slider: batch size B on x-axis; show arithmetic intensity vs ridge; color decode memory-bound until B ~ ridge/(bytes per param). Add "MoE" toggle dividing effective B by E/k | `AI = 2B/bytes_per_param`; `ridge = peak_FLOPs / BW`; `t_step = max(compute, memory)` |
| Step-time calculator | Pick GPU, model (dense/MoE), precision, TP/EP, batch, context -> t_step, tok/s/user, tok/s/GPU | `t_step = (W_active_or_all + B x ctx x KV_tok)/BW + t_comm`; users/GPU = B; throughput = B / t_step; for MoE, bytes read = `E_touched x expert_bytes`, `E_touched = E x (1 - (1-k/E)^B)` (expected distinct experts hit; ESTIMATE-style simplification) |
| Static vs continuous batching | Timeline Gantt: requests arrive (arrival-rate slider), have random output lengths; two lanes, static (waits for longest) and continuous (slots refill each step). Show GPU-slot utilization and avg latency | utilization = useful token-slots / total slots; static pad waste `(n-1)(B-1)` |
| Chunked prefill | Same timeline with a long prompt arriving; show decode stalls (TPOT spike) without chunking and a flat TPOT with chunk-size slider | `token_budget = decode_tokens + chunk`; TPOT_spike ~ `prompt_len / prefill_tokens_per_s` |
| PagedAttention | Grid of GPU memory blocks; requests grow token by token; block table shown per request; toggle "contiguous max-length reservation" vs "paged" and show wasted bytes; fork a sequence to show copy-on-write sharing | `waste_paged <= block_size-1 tokens/request`; `waste_contig = max_len - actual_len` |
| Prefix cache / RadixAttention | Radix tree grows as prompts are typed or chosen from presets; highlight reused prefix; counter "prefill tokens skipped" and cost; LRU eviction when budget full | `hit_rate = matched_tokens/prompt_tokens`; `cost = (1-hit) x input_price + hit x cache_read_price (+ write)` |
| KV size explorer | Pick MHA/GQA/MLA/V4-compressed, layers, heads, dtype, context -> bytes/token and GB; bar vs GPU HBM; "max concurrent users" | formulas in 3.1 and 8.2 |
| KV tiering | Three-tier diagram (HBM/DRAM/SSD) with capacity sliders and conversation count; throughput curve with cliffs as in vLLM's 64/128 conversation result | hit tier latency vs recompute time `= prefix_tokens / prefill_tok_per_s` |
| Speculative decoding | Acceptance-rate alpha slider, k slider, draft-cost c slider -> expected tokens/step, speedup, plus animation: green accepted tokens, red rejection point, bonus token. Add "batch size" slider that shrinks speedup (model: verification cost multiplier grows once B passes the ridge) | `E = (1-a^(k+1))/(1-a)`; `speedup = E/(1+k*c)`; MTP preset: k=1, a=0.85-0.90 -> E=1.85-1.90, DeepSeek reports 1.8x |
| Quantization | Bit-width slider: show bytes, step time, and a "same weights, round to FP4/INT4" error visualization on a histogram; MXFP4 (32-block, pow2 scale) vs NVFP4 (16-block, FP8 scale) rounding demo | `bytes = params x bits/8 (+ scale overhead: NVFP4 4.5 b/val)` |
| Disaggregation | Two clusters animation: colocated (prefill blocks stall decode) vs disaggregated (KV arrow with transfer time); sliders: prompt length, interconnect GB/s, P:D ratio; goodput meter against TTFT/TPOT SLOs | `kv_transfer_time = prompt_len x KV_tok / link_BW`; goodput = max rate with P99 TTFT<=SLO_T and TPOT<=SLO_P |
| Expert parallelism | Grid of GPUs with experts; tokens flow in all-to-all; slider EP size shows weights/GPU shrinking and per-expert batch growing; "hot expert" imbalance and EPLB redundant experts | `weights/GPU = W/EP`; `tokens_per_expert = B x k / E` (pooled across the EP group: `B_total x k / E`) |
| Router | Cluster of replicas with different cached prefixes; routing policy toggle (round-robin vs KV-aware) with hit-rate and TTFT outputs | `TTFT = queue + (1-hit) x prompt/prefill_rate` |
| Cost calculator | GPU $/hr, tok/s/GPU at chosen interactivity, input:output ratio, cache hit rate -> $/M input/output tokens and "list price vs cost" margin | `$/Mtok = $/hr / (tok/s x 3600) x 1e6`; blended input cost uses hit rate; cost per output token / cost per input token ~ (prefill tok/s per GPU) / (decode tok/s per GPU); e.g., DeepSeek's 73.7K vs 14.8K tok/s per node gives ~5x |

Good defaults for presets: DeepSeek production (EP144 decode, 56.3% hit rate, 4,989-token avg KV, 14.8K out tok/s/node), V4-Pro on GB300 (6,182 tok/s/GPU @ 27 tok/s/user), Llama-3-70B on H200.

### 9.2 Existing explainers worth linking (verified to exist this session unless noted)

- Hugging Face, [Continuous batching from first principles](https://huggingface.co/blog/continuous_batching) (Nov 2025): derives it from attention + KV cache, with ragged batching and chunked prefill. Good static-vs-continuous reference.
- Aleksa Gordic, [Inside vLLM: anatomy of a high-throughput LLM inference system](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm): scheduler, paged attention, prefix caching, spec decoding, multi-GPU, serving layer.
- Ashwin Giridharan, [interactive 11-chapter vLLM guide](https://ashwing.github.io/vllm-guide/): PagedAttention allocator simulator, token-by-token scheduler walkthrough (URL from the author's [DEV post](https://dev.to/ashwin_giridharan_dc396df/i-built-an-interactive-11-chapter-guide-to-how-llm-inference-actually-works-1pb9); I did not open the guide itself).
- DeepSeek, [Inference System Overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md): short, primary, numeric.
- LMSYS blogs on large-scale EP and GB200 ([EP](https://lmsys.org/blog/2025-05-05-large-scale-ep/), [GB200 pt 2](https://lmsys.org/blog/2025-09-25-gb200-part-2)); vLLM [large-scale serving](https://vllm.ai/blog/large-scale-serving) and [tiered KV](https://vllm.ai/blog/2026-09-10-tiered-kv-offloading).
- InferenceX dashboard, [inferencex.semianalysis.com](https://inferencex.semianalysis.com/about): live tok/s/GPU vs tok/s/user Pareto curves; ideal inspiration for the cost calculator.
- *(prior, not re-fetched; check URLs)*: Kipply, "Transformer Inference Arithmetic" (kipp.ly/transformer-inference-arithmetic); JAX ML "How to Scale Your Model" inference chapter (jax-ml.github.io/scaling-book/inference); Hao AI Lab DistServe blog "Throughput is not all you need" (hao-ai-lab.github.io/blogs/distserve); Lilian Weng "Large Transformer Model Inference Optimization"; Jay Alammar-style diagrams for KV cache; NVIDIA "Mastering LLM Techniques: Inference Optimization".

### 9.3 Core papers (arXiv ids)

Orca (OSDI'22); PagedAttention 2309.06180; FlashAttention 2205.14135 (IO-aware tiled attention; enables fast prefill and long context, *(prior)*); Sarathi-Serve 2403.02310; Splitwise 2311.18677; DistServe 2401.09670; Mooncake 2407.00079; SGLang/RadixAttention 2312.07104; speculative decoding 2211.17192 and 2302.01318; Medusa 2401.10774; EAGLE 2401.15077, EAGLE-2 2406.16858, EAGLE-3 2503.01840; DeepSeek-V2 (MLA) 2405.04434; DeepSeek-V3 2412.19437; GPTQ 2210.17323; AWQ 2306.00978; S-LoRA 2311.03285. All *(prior)* unless linked earlier in this file.

---

## 10. Open uncertainties and cautions for page-builders

1. **Engine market share** is not verifiable from primary sources; the "SGLang 400K GPUs", "SGLang/LMDeploy 16.2K vs vLLM 12.5K tok/s" and "TRT-LLM 8-13% faster" figures all come from SEO blogs. Use qualitative statements, or InferenceX numbers.
2. **Dynamo "up to 7x"** is an NVIDIA claim without stated methodology.
3. **V4-Pro KV estimate** is formula-derived from config.json; secondary sources disagree on the CSA:HCA layer mix (config suggests alternating 1:1, summaries say 3:1) and I could not confirm whether K and V share the stored 512-d vector, or the indexer's extra footprint. Range 4-12 GB per 1M tokens; the model-card claim (10% of V3.2) supports the low end.
4. **InferenceX tok/s/GPU units**: unclear whether per-GPU throughput counts only output tokens or input+output; the cost conversion implies total tokens. Verify before using in a "tokens per GPU" chart.
5. **OpenAI pricing and OpenAI/Anthropic/Google serving internals** are not public; pages should say "closed providers do not disclose their serving stacks". Anthropic/DeepSeek price tables were read live on 2026-10-07 and will drift.
6. Classic-paper numbers (Orca, PagedAttention, Splitwise, DistServe) were not re-fetched; quote ranges, not decimals, or re-check arXiv.
7. Hugging Face's "16 KB per token for Llama-2-7B" (as returned by my fetch tool) is wrong by 32x; use 512 KB.
8. Spec-decoding gains are regime-dependent: big at small batch, ~1.4x at batch 64 for EAGLE-3 in SGLang. Do not present "2-3x" as a throughput multiplier at scale; it is a per-user latency multiplier.
