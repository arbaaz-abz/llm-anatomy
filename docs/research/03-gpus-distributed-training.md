# 03 — How Large LLMs Are Trained on GPUs (hardware + distributed systems)

Research brief for page-builders. Date of research: 2026-10-07. Scope: hardware, memory, parallelism, efficiency, reliability, recent runs. (Training recipe and inference serving are covered elsewhere.)

**Confidence tags:** CONFIRMED = read in a primary source (vendor page/doc or the lab's own paper) during this research. REPORTED = credible secondary source, or primary source not fully read. UNVERIFIED = could not confirm, or sources conflict. "DERIVED" = my arithmetic from confirmed inputs.

**Biggest cautions up front**
1. Vendor FLOPs are quoted in three flavors: dense, "with sparsity" (2x), and for NVIDIA FP4 "inference" numbers that include tricks. Use **dense** numbers for training toys. Every table below is dense unless stated.
2. Several widely repeated 2026 claims are wrong or unsupported (e.g. "GLM-5 was trained on 100,000 Huawei Ascend chips" — the GLM-5 paper does not say this; see section 6).
3. Rubin HBM bandwidth is stated as 22 TB/s in NVIDIA's tech blog but 19.2 TB/s on one NVIDIA product page. Use "~20 TB/s" or flag it.

---

## 1. GPU primer for an LLM learner

### 1.1 Anatomy (what to draw)
- **SM (streaming multiprocessor)**: the unit of parallel compute. H100 SXM has 132 SMs (DeepSeek-V3 paper mentions "132 SMs available in the H800", [arXiv 2412.19437](https://arxiv.org/abs/2412.19437), CONFIRMED). Rubin has 224 SMs, 2 compute dies, 336B transistors ([NVIDIA tech blog](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/), CONFIRMED).
- **Tensor cores**: matrix-multiply units inside each SM; they provide >90% of the FLOPs. Everything that is not a matmul (softmax, layernorm, exp) runs on much slower units. FlashAttention-4's paper makes this explicit: on Blackwell tensor-core throughput doubled while shared memory bandwidth and exponential units scaled far less ([arXiv 2603.05451](https://arxiv.org/pdf/2603.05451.pdf), REPORTED via abstract summary).
- **Memory hierarchy** (rough, per GPU): registers -> shared memory/L1 per SM (~hundreds of KB, tens of TB/s aggregate) -> L2 (tens of MB) -> **HBM** (80-288 GB, 3-22 TB/s) -> NVLink peers (0.9-3.6 TB/s) -> InfiniBand/Ethernet NIC (50-100 GB/s per GPU). Each step down is roughly 3-10x slower. Good "zoom out" animation.
- **HBM vs SRAM**: HBM is big and slow(ish); on-chip SRAM is tiny and fast. FlashAttention exists because attention's naive implementation shuttles the S x S score matrix through HBM; tiling keeps it in SRAM ([Dao et al. 2022](https://arxiv.org/abs/2205.14135), CONFIRMED).

### 1.2 Arithmetic intensity and the roofline
- **Arithmetic intensity** I = FLOPs performed / bytes moved. Source definition: [Scaling Book, Roofline chapter](https://jax-ml.github.io/scaling-book/roofline/) (CONFIRMED).
- **Roofline**: time = max(FLOPs / peak_FLOPs, bytes / bandwidth). Attainable throughput = min(peak, I x bandwidth). The "ridge point" (critical intensity) = peak_FLOPs / bandwidth.
- Critical intensities (DERIVED from tables below, BF16 dense unless noted): H100 989/3.35 = **295** FLOPs/byte (the Scaling Book also states ~295, CONFIRMED); B200 ~2250/8 = ~280; MI355X 2500/8 = ~312; TPU Ironwood BF16 2307/7.38 = ~313; **FP8 on Ironwood 4614/7.38 = ~625; NVFP4 on B300 15000/8 = ~1,875**. Trend to teach: **lower precision and newer chips push the ridge point right**, so "memory-bound" gets easier to hit.
- **Matmul** X[B,D] @ W[D,F] in BF16: intensity = BDF / (BD + DF + BF) FLOPs-per-byte (units: the Scaling Book's formula, with 2 bytes/element folded in -> intensity ~ B when B << D,F). So a matmul is **compute-bound once the token batch B exceeds ~240-300** (Scaling Book: "greater than 240 (tokens)" on TPU v5e). Training batches are millions of tokens -> compute-bound. This is why training is "easy" on FLOPs and hard on communication.
- **Decode** (inference): each generated token does a matmul with B = number of concurrent sequences (often 1-64) -> intensity ~B << 295 -> **memory-bound**: time ~ (bytes of weights + KV cache) / HBM bandwidth. (Inference is another agent's domain; just reuse the same roofline plot with a different operating point.)
- Elementwise ops (GELU, layernorm, residual add) have intensity ~1 -> always memory-bound -> **kernel fusion** (do many ops per HBM round trip). Classic reference: [Horace He, "Making Deep Learning Go Brrrr"](https://horace.io/brrr_intro.html) (page exists; CONFIRMED reachable).

### 1.3 Number formats
| Format | Bits | Layout | What it buys | Training status |
|---|---|---|---|---|
| FP32 | 32 | 1/8/23 | Master weights, optimizer states, accumulation | always used somewhere |
| BF16 | 16 | 1/8/7 | Same range as FP32, 2x less memory, no loss scaling needed | default for pre-training since ~2020 |
| FP16 | 16 | 1/5/10 | More mantissa, small range -> needs loss scaling | legacy |
| FP8 (E4M3 / E5M2) | 8 | 1/4/3 or 1/5/2 | 2x tensor-core FLOPs vs BF16; half the bytes | DeepSeek-V3 trained at 671B scale with it (CONFIRMED, section 5.3) |
| MXFP8 / MXFP4 | 8 / 4 | block of 32 elements shares an 8-bit power-of-2 scale | OCP "microscaling" standard; hardware does the scaling | MXFP4 used for DeepSeek-V4 expert weights QAT and Kimi K3 post-training QAT (CONFIRMED, [DeepSeek-V4](https://arxiv.org/abs/2606.19348), [Kimi K3](https://arxiv.org/abs/2607.24653)) |
| NVFP4 | 4 | FP4 (E2M1) with a block of **16** elements sharing an **E4M3** scale (plus a per-tensor FP32 scale) | ~4x FLOPs vs BF16, ~1/4 memory | NVIDIA's 12B/10T-token run matched FP8 loss ([NVIDIA blog](https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/), CONFIRMED); **Nemotron 3 Ultra (550B total/55B active) pre-trained ~20T tokens in NVFP4**, loss gap vs BF16 <0.4% ([arXiv 2606.15007](https://arxiv.org/pdf/2606.15007), REPORTED via search summary; model-card page confirms "Pretrained in NVFP4", CONFIRMED) |

NVFP4 recipe ingredients (NVIDIA blog, CONFIRMED): 16-element micro-blocks with E4M3 scales, random Hadamard transforms on GEMM inputs to tame outliers, stochastic rounding of gradients, 2D block quantization for weights, and some sensitive layers kept in higher precision.
Teaching point: low precision is about **bytes moved and tensor-core rate**, but the accumulators and optimizer states stay high-precision. DeepSeek-V3 also notes FP8 tensor-core accumulation keeps only ~14 bits, so it promotes partial sums to FP32 on CUDA cores ([arXiv 2412.19437](https://arxiv.org/abs/2412.19437), CONFIRMED).

### 1.4 Hardware table (per accelerator, dense, as of Oct 2026)

| Chip | HBM | HBM BW | BF16 dense | FP8 dense | FP4 dense | Scale-up link | Status | Source / tag |
|---|---|---|---|---|---|---|---|---|
| H100 SXM | 80 GB HBM3 | 3.35 TB/s | 989 TF | 1,979 TF | n/a | NVLink4 900 GB/s | mature | [Scaling Book](https://jax-ml.github.io/scaling-book/roofline/) (BF16, BW) CONFIRMED; FP8 = 2x BF16 (REPORTED) |
| H200 | 141 GB HBM3e | 4.8 TB/s | same as H100 | same | n/a | 900 GB/s | mature | [NVIDIA H200](https://www.nvidia.com/en-us/data-center/h200/) CONFIRMED |
| B200 (HGX, 1 kW) | 180 GB usable (192 GB nominal) | 8 TB/s | ~2.25 PF | ~4.5 PF | ~9 PF | NVLink5 1.8 TB/s | shipping since 2025 | [DGX B200 datasheet via search](https://openzeka.com/wp-content/uploads/2024/10/dgx-scale-ai-infrastructure-dgx-b200-datasheet-nvidia-web-1.pdf) REPORTED |
| B200 in GB200 NVL72 | ~186 GB (13.4 TB / 72; nominal 192) | 8 TB/s | ~2.5 PF | ~5 PF | ~10 PF | 1.8 TB/s | shipping | [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/): rack 13.4 TB HBM, 576 TB/s, 130 TB/s NVLink, FP4 720 PF dense/rack -> 10 PF/GPU (DERIVED), CONFIRMED |
| B300 (Blackwell Ultra) | 288 GB HBM3e (12-high) | 8 TB/s | ~2.5 PF | ~5 PF | **15 PF** | 1.8 TB/s | shipping | [GB300 NVL72 page](https://www.nvidia.com/en-us/data-center/gb300-nvl72/) (20 TB HBM, 576 TB/s, 130 TB/s) CONFIRMED; 15 PF dense FP4 and 1,400 W via [Introl](https://introl.com/blog/nvidia-blackwell-ultra-b300-infrastructure-requirements-2025) REPORTED |
| **Rubin (R200)** | 288 GB HBM4 | **22 TB/s** (NVIDIA tech blog) vs 19.2 TB/s (NVIDIA product page) | ~4 PF (NVIDIA product page; CONFIRMED as listed, but see flag) | ~17.5 PF (product page; CONFIRMED as listed) | **35 PF NVFP4 training / 50 PF NVFP4 inference** | NVLink6 3.6 TB/s | **shipping since ~Jul 2026** | [NVIDIA tech blog](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/) CONFIRMED (288 GB, 22 TB/s, 35/50 PF, 3.6 TB/s, NVL72 260 TB/s); [product page](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/) lists 19.2 TB/s, 3 TB/s -> conflict, UNVERIFIED which is current. **Flag:** product page's BF16 4 PF vs FP8 17.5 PF is a 4.4x jump (expected ~2x), so one of the numbers is probably sparse/dense-mismatched or stale; don't use Rubin BF16/FP8 in a toy without re-checking |
| MI355X (AMD) | 288 GB HBM3E | 8 TB/s | 2.5 PF | 5 PF | 10 PF (FP4/MXFP4) | Infinity Fabric (8-GPU) | shipping since mid-2025 | [Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/amd-reveals-core-specs-for-instinct-mi355x-cdna4-ai-accelerator-slated-for-shipping-in-the-second-half-of-2025) REPORTED |
| MI455X/MI450 (Helios) | 432 GB HBM4 | 19.6 TB/s | n/a | ~20 PF | ~40 PF | UALink-over-Ethernet, 72-GPU rack | "in production"; first shipments ~end Q3 2026; OpenAI deployment from Q4 2026 | [AMD press release](https://ir.amd.com/news-events/press-releases/detail/1294/aai-2026-amd-delivers-full-stack-compute-for-the-agentic-ai-era) (72 GPUs/rack, in production, OpenAI Q4 2026) CONFIRMED; chip numbers from [TechRadar](https://www.techradar.com/pro/amd-gets-ready-for-nvidias-vera-rubin-and-2026-with-432gb-mi400-gpu-monster-paired-with-256-core-epyc-venice-and-i-cant-wait-to-see-the-sparks-fly) REPORTED (these are 2025 targets; final specs UNVERIFIED) |
| TPU v6e (Trillium) | 32 GB | 1.64 TB/s | 918 TF | (int8 1,836 TOPs) | n/a | ICI 800 GB/s, 256-chip pod | GA | [Google docs](https://docs.cloud.google.com/tpu/docs/v6e) CONFIRMED |
| **TPU v7 "Ironwood" (TPU7x)** | 192 GiB | 7.38 TB/s | 2,307 TF | 4,614 TF | n/a | ICI 1.2 TB/s, 3D torus, **9,216-chip pod** | available | [Google docs](https://docs.cloud.google.com/tpu/docs/tpu7x) CONFIRMED |
| TPU 8t (training) / 8i (inference) | 8i: 288 GB HBM, 384 MB SRAM | n/a | n/a | n/a | 8t pod: 9,600 chips, "121 EF FP4" | 3D torus | announced Cloud Next '26, **GA late 2027** | [coverage](https://nand-research.com/google-cloud-8th-generation-tpu-family-splits-training-and-inference/) REPORTED |
| Trainium2 | 96 GiB | 2.9 TB/s | n/a | 1.3 PF | n/a | 64-chip UltraServer | shipping | [Neuron docs](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.25.0/general/arch/neuron-hardware/trainium2.html) REPORTED |
| **Trainium3** | 144 GB HBM3e | 4.9 TB/s | n/a | 2.52 PF (MXFP8) | MXFP4 supported | NeuronLink-v4 2 TB/s/chip; **144-chip UltraServer** (362 PF FP8, 20.7 TB HBM) | launched Dec 2025 | [AWS Trn3 page](https://aws.amazon.com/ec2/instance-types/trn3/) CONFIRMED |

Rack-scale systems:
- **GB200/GB300 NVL72**: 72 GPUs + 36 Grace CPUs in one NVLink domain, ~130 TB/s aggregate NVLink; GB300 = 20 TB HBM3e ([NVIDIA](https://www.nvidia.com/en-us/data-center/gb300-nvl72/), CONFIRMED).
- **Vera Rubin NVL72**: 72 Rubin GPUs + 36 Vera CPUs, 260 TB/s NVLink6 ([NVIDIA tech blog](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/), CONFIRMED). Naming note: NVIDIA moved from "NVL144" (counting dies) back to "NVL72" (counting packages) — the product page uses NVL72 (CONFIRMED); older articles say NVL144.
- **Shipping status**: NVIDIA reportedly confirmed full production on 21 Jul 2026, with shipments to OpenAI, CoreWeave, Google Cloud, Azure, Meta, Dell ([GCN](https://gcn.com/nvidia-vera-rubin-chips-begin-shipping/20421/), REPORTED; the NVIDIA product page marks it "Available Now", CONFIRMED). CoreWeave says Cognition is its first production customer ([CoreWeave](https://www.coreweave.com/news/coreweave-delivers-nvidia-vera-rubin-nvl72-performance-at-production-scale-starting-with-cognition), REPORTED).
- **Roadmap**: Rubin Ultra (4 dies/package, ~1 TB HBM4e, NVL576 Kyber rack) H2 2027; Feynman 2028 ([DCD](https://datacenterdynamics.com/en/news/nvidias-rubin-ultra-nvl576-rack-expected-to-be-600kw-coming-second-half-of-2027), REPORTED).
- **MI500 / Helios 500**: AMD says 2027 ([AMD](https://ir.amd.com/news-events/press-releases/detail/1294/aai-2026-amd-delivers-full-stack-compute-for-the-agentic-ai-era), CONFIRMED).

Headline trend for a chart: from H100 (2022) to Rubin (2026): HBM 80 -> 288 GB (3.6x), HBM BW 3.35 -> ~22 TB/s (6.6x), low-precision FLOPs 2 PF (FP8) -> 35 PF (NVFP4, dense training) (~17x, partly from the precision drop), NVLink 0.9 -> 3.6 TB/s (4x). FLOPs outrun bandwidth, so memory/communication are the bottleneck.

---

## 2. Interconnect and cluster topology

### 2.1 Layers of network (fastest to slowest)
| Layer | Technology | Per-GPU bandwidth | Domain size |
|---|---|---|---|
| On-package | die-to-die | multi-TB/s | 1 GPU |
| **Scale-up** | NVLink + NVSwitch | H100 900 GB/s; B200/B300 1.8 TB/s; Rubin 3.6 TB/s (bidirectional) | 8 (HGX), **72 (NVL72)**, 576 (Rubin Ultra Kyber, 2027) |
| Scale-up (others) | TPU ICI 3D torus (Ironwood 1.2 TB/s/chip, up to 9,216 chips); Trainium NeuronLink; AMD UALink/Ethernet (Helios 72) | | |
| **Scale-out** | InfiniBand (Quantum-X800) or Ethernet/RoCE (Spectrum-X) via ConnectX NICs | 400 Gb/s (50 GB/s) in H100-era; 800 Gb/s (100 GB/s) CX-8 in Blackwell; ConnectX-9 in Rubin (REPORTED) | thousands-100K+ |

Concrete numbers: DeepSeek-V3's cluster had NVLink 160 GB/s effective vs IB 50 GB/s per GPU (3.2x gap) — these are the H800 numbers ([arXiv 2412.19437](https://arxiv.org/abs/2412.19437), CONFIRMED; H800 has reduced NVLink vs H100). GB300 NVL72 uses ConnectX-8 at 800 Gb/s per GPU ([NVIDIA](https://www.nvidia.com/en-us/data-center/gb300-nvl72/), CONFIRMED).

### 2.2 Cluster shape
- **HGX node**: 8 GPUs, all-to-all via NVSwitch; nodes connected by NICs. Kimi K2: 8 GPUs/node, 2 TB RAM, 8x400 Gb/s RoCE ([arXiv 2507.20534](https://arxiv.org/abs/2507.20534), CONFIRMED).
- **NVL72 rack**: 72 GPUs in one coherent NVLink domain (~9x the scale-up domain of HGX). Nemotron's RL report notes expert-parallel groups must be co-located on the same rack to stay inside NVLink ([arXiv 2606.15007](https://arxiv.org/pdf/2606.15007), CONFIRMED text).
- **Meta's 24K RoCE cluster for Llama 3** (CONFIRMED, [arXiv 2407.21783](https://arxiv.org/abs/2407.21783)): 16 GPUs per rack (2 servers + 1 ToR switch); 192 racks form a **pod of 3,072 GPUs with full bisection bandwidth**; 8 pods form 24K GPUs with aggregation layer **oversubscribed 1:7**. 400 Gbps per GPU. Scheduler and parallelism are topology-aware to minimize cross-pod traffic. Load balancing: 16 flows per GPU pair + Enhanced-ECMP; no DCQCN.
- **Rail-optimized**: GPU i of every node connects to the same "rail" leaf switch, so same-index GPUs across nodes are one hop apart. DeepSeek's all-to-all exploits this: a token goes over IB to the same-index GPU on the target node, then is forwarded over NVLink to the expert's GPU, and each token goes to at most 4 nodes ([arXiv 2412.19437](https://arxiv.org/abs/2412.19437), CONFIRMED). (The general term "rail-optimized" is standard NVIDIA SuperPOD design; not separately sourced here.)

### 2.3 Why topology dictates parallelism placement
Rule: **put the chattiest parallelism on the fastest link.** Llama 3's order is [TP, CP, PP, DP], innermost to outermost: "innermost parallelism requires the highest network bandwidth and lowest latency, and hence is usually constrained to within the same server" ([arXiv 2407.21783](https://arxiv.org/abs/2407.21783), CONFIRMED). Rough hierarchy:
1. Tensor parallel: all-reduce on the critical path every layer -> inside NVLink domain (<= 8, or up to 72 on NVL72).
2. Expert parallel: all-to-all every MoE layer -> NVLink, or IB if heavily overlapped (DeepSeek does this).
3. Context parallel: ring/all-gather of K,V.
4. Pipeline parallel: only small activations point-to-point between stages -> tolerates IB / cross-rack.
5. Data parallel/FSDP: one gradient sync per step, can be overlapped/prefetched -> outermost, cross-pod.
The Scaling Book gives the quantitative rule: TP stays compute-bound only if its degree Y < F / (critical intensity vs ICI), which in practice caps TP at ~8-16 ([Scaling Book, Training](https://jax-ml.github.io/scaling-book/training/), CONFIRMED). NVL72 shifts the cap: bigger fast domains let EP/TP grow without hitting IB.

---

## 3. Memory accounting for training

### 3.1 Bytes per parameter (mixed precision with Adam)
Per parameter:
- BF16 weights: 2
- BF16 gradients: 2
- FP32 master weights: 4
- Adam m (FP32): 4
- Adam v (FP32): 4
**= 16 bytes/param** (ZeRO paper: 2 + 2 + K, K=12 for Adam; [arXiv 1910.02054](https://arxiv.org/abs/1910.02054), CONFIRMED by reference). Variants: FP32 gradient accumulation buffer instead of BF16 -> 20 B; no separate master copy -> 12 B; BF16 optimizer states (DeepSeek-V3 stores low-precision optimizer states in BF16, [arXiv 2412.19437](https://arxiv.org/abs/2412.19437), CONFIRMED) -> ~10-12 B. "16-18 B/param" is the safe teaching number. For Muon (Kimi K2/K3, DeepSeek-V4): **one** momentum buffer instead of two Adam buffers -> ~4 B less per param (Kimi K2 reports BF16 params + FP32 grad buffer = 6 B/param ≈ 6 TB for 1.04T params, [arXiv 2507.20534](https://arxiv.org/abs/2507.20534), CONFIRMED).

### 3.2 Activations
Per transformer layer, BF16, no recompute (Korthikanti et al. 2022, [arXiv 2205.05198](https://arxiv.org/abs/2205.05198), CONFIRMED by reference):
- no parallelism: `s*b*h*(34 + 5*a*s/h)` bytes (s = seq len, b = micro-batch, h = hidden, a = heads)
- with TP t + sequence parallel: divide the whole thing by t
- selective recomputation (recompute the attention-score part only): `34*s*b*h / t`
- full recomputation: store only layer inputs ~ `2*s*b*h` per layer, at ~33% extra compute (one extra forward).
The `5*a*s/h` term is the S x S attention matrix that FlashAttention never materializes. Modern runs add finer tricks: DeepSeek-V3 recomputes RMSNorm and MLA up-projections and keeps EMA weights on CPU ([arXiv 2412.19437](https://arxiv.org/abs/2412.19437), CONFIRMED); Kimi K3 uses a "unified activation manager" combining recompute, FP8-quantized activations, local CPU offload, and remote offload to other pipeline ranks ([arXiv 2607.24653](https://arxiv.org/abs/2607.24653), CONFIRMED).

### 3.3 Worked example: 1T-parameter model
- States alone: 1e12 x 16 B = **16 TB**.
- Per GPU: H100 80 GB -> 16e12/80e9 = **200 GPUs minimum just to hold states** (with perfect sharding, zero activations). H200 141 GB -> 114; B200 180 GB -> 89; B300/Rubin 288 GB -> **56**. DERIVED.
- Realistically activations, buffers, fragmentation, and communication workspaces consume 20-50%+ of HBM, so 2-4x more. Kimi K2 (1.04T total) holds its params + grad buffer (6 TB) over a 256-GPU model-parallel group, leaving ~30 GB/GPU for states, with optimizer state distributed across data-parallel ranks ([arXiv 2507.20534](https://arxiv.org/abs/2507.20534), CONFIRMED).
- The model can't fit on one GPU even for **weights alone**: 1T x 2 B = 2 TB = 7 B300s.
- MoE caveat: *compute* scales with active params (e.g. 32-104B), but *memory* scales with total params. This is why MoE training needs lots of GPUs per replica despite low FLOPs.

### 3.4 ZeRO / FSDP per-GPU memory (Ψ = params, N = data-parallel GPUs, mixed-precision Adam)
| Stage | Shards | Per-GPU bytes | Ψ=7.5B, N=64 (ZeRO paper example) |
|---|---|---|---|
| DDP baseline | nothing | 16Ψ | 120 GB |
| ZeRO-1 | optimizer states | 4Ψ + 12Ψ/N | 31.4 GB |
| ZeRO-2 | + gradients | 2Ψ + 14Ψ/N | 16.6 GB |
| ZeRO-3 / FSDP | + parameters | 16Ψ/N | 1.9 GB |
The arithmetic matches the ZeRO paper's 7.5B/64-GPU table (DERIVED; paper [arXiv 1910.02054](https://arxiv.org/abs/1910.02054), CONFIRMED by reference). Comms: DDP all-reduce ≈ 2Ψ words; ZeRO-1/2 same total; ZeRO-3 ≈ 3Ψ (extra all-gather of params in fwd and bwd) = 1.5x.

---

## 4. Parallelism strategies

Collective cost model (for the toys): ring all-reduce of S bytes over N ranks moves `2*(N-1)/N * S` bytes per rank (reduce-scatter + all-gather, each `(N-1)/N * S`); time ≈ that / link BW (+ latency terms). All-to-all: each rank sends `(N-1)/N` of its data to others.

| Strategy | What is split | Communication | Pain point |
|---|---|---|---|
| **Data parallel (DP)** | batch; model replicated | all-reduce gradients once/step | memory: full replica per GPU |
| **ZeRO-1/2/3, FSDP** | optimizer / +grads / +params across DP ranks | reduce-scatter grads; all-gather params (ZeRO-3 every layer fwd & bwd) | extra comms at stage 3; needs good overlap |
| **Tensor parallel (TP)** | each weight matrix (Megatron: MLP first matrix by **column**, second by **row**; attention by heads) | 2 all-reduces/layer fwd, 2 bwd (all-gather+reduce-scatter with sequence parallel) | latency-critical, only fits inside NVLink (TP ≤ 8 on HGX) |
| **Sequence / context parallel (CP)** | sequence dimension; ring attention passes K,V blocks around a ring while computing | P2P K,V ring or all-gather | long-context only; Llama 3 uses all-gather CP since GQA makes K,V small |
| **Pipeline parallel (PP)** | layers into stages | point-to-point activations between stages | **bubble**, stage imbalance, memory of in-flight micro-batches |
| **Expert parallel (EP)** | MoE experts across GPUs | **all-to-all** dispatch + combine every MoE layer | load imbalance, all-to-all latency across nodes |

### 4.1 Data parallel and ZeRO/FSDP
- Intuition: each GPU has the whole model and a slice of the batch; average gradients. All-reduce is bandwidth-optimal as ring/tree. When the model doesn't fit, shard states (ZeRO) — same math, "unshard just in time".
- Llama 3 used FSDP sharding optimizer states + gradients but **not resharding weights after forward** (to avoid an extra all-gather in backward) ([arXiv 2407.21783](https://arxiv.org/abs/2407.21783), CONFIRMED). DeepSeek-V3 used ZeRO-1 only (with 16-way PP and 64-way EP already shrinking per-GPU state) ([arXiv 2412.19437](https://arxiv.org/abs/2412.19437), CONFIRMED). Kimi K3 uses ZeRO-1 DP plus "Pipeline ZeRO-2" gradient sharding ([arXiv 2607.24653](https://arxiv.org/abs/2607.24653), CONFIRMED).
- Compute-bound condition: per-device token batch B/X must exceed C/W_ici (≈2,550 tokens/chip on TPU v5p) ([Scaling Book](https://jax-ml.github.io/scaling-book/training/), CONFIRMED). Mixing FSDP+TP drops the needed per-chip batch to ~100 tokens.

### 4.2 Tensor parallel (Megatron-LM)
- MLP: Y = GeLU(X·A), Z = Y·B. Split A by columns (each GPU computes its slice of the hidden activation independently), split B by rows (each GPU produces a partial sum of Z) -> one all-reduce per MLP; similarly one for attention (heads split across GPUs). Reference: Megatron-LM, [arXiv 1909.08053](https://arxiv.org/abs/1909.08053) (not fetched; REPORTED). Sequence parallelism shards layernorm/dropout activations along sequence, converting all-reduce into reduce-scatter + all-gather ([arXiv 2205.05198](https://arxiv.org/abs/2205.05198)).
- Pain: communication volume per layer ~ `4 x (t-1)/t x s·b·h x 2 bytes` and cannot overlap well -> NVLink only.
- Note: DeepSeek-V3 trained **without TP** (EP64+PP16+ZeRO-1) ([arXiv 2412.19437](https://arxiv.org/abs/2412.19437), CONFIRMED) because MoE experts are small and MLA is cheap.

### 4.3 Sequence / context parallel
- Ring attention: shard sequence over GPUs; each holds Q chunk, K/V blocks rotate around the ring, overlapped with block-wise attention ([Liu et al., arXiv 2310.01889](https://arxiv.org/abs/2310.01889), REPORTED).
- Llama 3: CP only in the final long-context stage (128K seq, CP=16); uses all-gather of K,V since attention FLOPs O(S²) dwarf the O(S) all-gather ([arXiv 2407.21783](https://arxiv.org/abs/2407.21783), CONFIRMED).
- Linear-attention hybrids need different CP: Kimi K3's KDA context parallelism passes recurrent state via a fixed-size all-gather ([arXiv 2607.24653](https://arxiv.org/abs/2607.24653), CONFIRMED). DeepSeek-V4 uses "two-stage contextual parallelism" for compressed attention ([arXiv 2606.19348](https://arxiv.org/abs/2606.19348), CONFIRMED).

### 4.4 Pipeline parallel
- Split L layers into p stages; split batch into m micro-batches.
- **GPipe**: all forwards, then all backwards. Bubble fraction = (p-1)/(m+p-1); needs memory for m micro-batches' activations.
- **1F1B**: alternate forward/backward after warm-up; same bubble, but activation memory capped at ~p micro-batches. (DeepSeek's Table 2: 1F1B bubble = (PP−1)(F+B), activation = PP×, [arXiv 2412.19437](https://arxiv.org/abs/2412.19437), CONFIRMED.)
- **Interleaved 1F1B (virtual stages)**: each GPU holds v non-contiguous chunks -> bubble divided by v, at the cost of v-fold more P2P messages. Used by Kimi K2/K3 (PP with virtual stages) ([arXiv 2507.20534](https://arxiv.org/abs/2507.20534), CONFIRMED) and Llama 3 (tunable N continuous micro-batches; adjusts first/last-stage imbalance) ([arXiv 2407.21783](https://arxiv.org/abs/2407.21783), CONFIRMED).
- **Zero-bubble (ZB1P)**: split backward into B (grad wrt input) and W (grad wrt weights); fill bubbles with W. Bubble = (PP−1)(F+B−2W).
- **DualPipe** (DeepSeek-V3): bidirectional pipeline — micro-batches fed from both ends; forward and backward chunks are paired so all-to-all/P2P communication overlaps compute. Bubble = (PP/2−1)(F&B + B − 3W); parameters 2× (two copies) and activation PP+1 (vs PP); bubble and activation memory don't grow with micro-batch count ([arXiv 2412.19437](https://arxiv.org/abs/2412.19437), CONFIRMED). Code: [github.com/deepseek-ai/DualPipe](https://github.com/deepseek-ai/DualPipe).
- Pain points: bubbles, last stage holds the loss/vocab layer (compute imbalance), first stage holds embeddings + most warm-up activations (memory imbalance) — Llama 3 explicitly mentions both.

### 4.5 Expert parallel (MoE)
- Each GPU holds a subset of experts. Per MoE layer: **dispatch** (all-to-all: tokens -> GPUs hosting their top-k experts), expert FFNs, **combine** (all-to-all: results back, weighted sum).
- DeepSeek-V3: 64-way EP across 8 nodes, each token limited to ≤4 nodes, IB then NVLink forwarding; custom kernels need only **20 of 132 SMs** for communication ([arXiv 2412.19437](https://arxiv.org/abs/2412.19437), CONFIRMED). Open-sourced as [DeepEP](https://github.com/deepseek-ai/DeepEP).
- DeepSeek-V4: fuses dispatch + expert GEMMs + combine into one "mega-kernel" with expert "waves" (comm for the next wave overlaps compute of the current), open-sourced as MegaMoE in DeepGEMM; **hiding condition: C/B ≤ V_comp/V_comm = 2d = 6,144 FLOPs/byte**, i.e. each GB/s of interconnect hides ~6.1 TFLOP/s of compute ([arXiv 2606.19348](https://arxiv.org/abs/2606.19348), CONFIRMED). Great formula for a toy.
- Kimi K3: **MoonEP** guarantees perfect load balance via dynamic redundant experts and static shapes ([arXiv 2607.24653](https://arxiv.org/abs/2607.24653), CONFIRMED).
- Pain: expert load imbalance (stragglers), all-to-all latency, memory for dispatch buffers.

### 4.6 Combining them (concrete configs)

| Run | Hardware | Config | Source / tag |
|---|---|---|---|
| **Llama 3 405B** (15.6T tokens, 3.8e25 FLOPs) | up to 16,384 H100 | 8K GPUs: TP8 / CP1 / PP16 / DP64, 430 TFLOPs/GPU, **43% MFU**; 16K GPUs: TP8/CP1/PP16/DP128, 400 TFLOPs, 41%; long-context stage 16K GPUs: TP8/**CP16**/PP16/DP8, 380 TFLOPs, 38% | [arXiv 2407.21783](https://arxiv.org/abs/2407.21783) CONFIRMED (Table 4). Note: CP=1 for the main stage — the user's "TP8/CP/PP16/DP" is right only with CP in the long-context phase |
| **DeepSeek-V3** (671B total/37B active, 14.8T tokens) | **2,048 H800** (8/node, NVLink+IB) | **PP16 / EP64 (8 nodes) / ZeRO-1 DP, no TP**; DualPipe; FP8 | [arXiv 2412.19437](https://arxiv.org/abs/2412.19437) CONFIRMED (2048 H800, 16-way PP, 64-way EP, ZeRO-1) |
| **Kimi K2** (1.04T/32B active, 15.5T tokens) | H800; trains on any multiple of 32 nodes (min 256 GPUs) | PP16 (virtual stages) / EP16 / ZeRO-1 DP; 256-GPU model-parallel group; ~30 GB/GPU states | [arXiv 2507.20534](https://arxiv.org/abs/2507.20534) CONFIRMED |
| **Kimi K3** (2.8T/104B active) | not disclosed | PP+VP / EP (MoonEP) / ZeRO-1 DP + Pipeline ZeRO-2 / CP | [arXiv 2607.24653](https://arxiv.org/abs/2607.24653) CONFIRMED (no GPU count in what I read) |
| **Llama 4 Behemoth** | 32K GPUs, FP8 | 390 TFLOPs/GPU | [Meta](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) CONFIRMED |

How to count GPUs: total = TP x CP x PP x DP (x EP overlays DP/TP dimension in MoE). Llama 3: 8 x 1 x 16 x 64 = 8,192 ✓.; 8 x 1 x 16 x 128 = 16,384 ✓.; 8 x 16 x 16 x 8 = 16,384 ✓..
"5D parallelism" in the literature = DP + TP + PP + CP + EP (Ultra-Scale Playbook framing). Selection heuristic (Playbook/Scaling Book): fit the model in memory using the least communication -> start with FSDP/ZeRO; add TP inside the NVLink domain; add PP across nodes when memory still doesn't fit or IB is the bottleneck; add CP only for long sequences; add EP for MoE.

---

## 5. Efficiency and reliability at scale

### 5.1 MFU
- **MFU** = (model FLOPs per second actually achieved, counting 6·N·tokens/s-ish) / (hardware peak FLOPs). **HFU** also counts recomputation.
- Published: Llama 3 405B **38-43% BF16 MFU** on H100 ([arXiv 2407.21783](https://arxiv.org/abs/2407.21783), CONFIRMED). Llama 4 Behemoth: 390 TFLOPs/GPU in FP8 on 32K GPUs; Meta gave no MFU, and the % depends on the denominator (~20% of H100's 1,979 TF FP8 peak, but ~39% of its 989 TF BF16 peak; DERIVED, convention ambiguous) ([Meta](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)). Scaling Book example assumes 50% for planning ([source](https://jax-ml.github.io/scaling-book/training/)). Rule of thumb: 35-55% for dense BF16, often reported lower when measured against FP8/FP4 peaks (peak doubles but memory-bound parts don't; but conventions differ), lower for MoE.
- NVIDIA Megatron Core on GB300 NVL72 reached **1,648 TFLOPs/GPU** on DeepSeek-V3 671B pre-training with 256 GPUs (98.5% per-GPU retention at 1,024 GPUs), ~3x GB200 NVL72's 606; "1.5x from software alone over six months" ([NVIDIA blog](https://developer.nvidia.com/blog/setting-a-world-record-for-moe-pre-training-on-nvidia-gb300-nvl72/), CONFIRMED figures; the blog doesn't state precision, so MFU is UNVERIFIED).
- FlashAttention-3 on H100 and FA-4 on B200: FA-4 BF16 forward reaches 1,613 TFLOPs/s = 71% of B200's peak ([arXiv 2603.05451](https://arxiv.org/pdf/2603.05451.pdf), REPORTED).
- Qwen reports 30% higher MFU for Qwen3-Max vs Qwen2.5-Max via PAI-FlashMoE pipeline strategy and 5x less failure-time loss ([search summary](https://deepinfra.com/Qwen/Qwen3-Max), REPORTED; low-quality source, treat as UNVERIFIED detail).

### 5.2 Overlap
- Overlap communication with compute: DP gradient reduce-scatter during backward; FSDP prefetch next layer's all-gather; TP async chunked collectives; PP P2P hidden in 1F1B steady state; EP via DualPipe or fused mega-kernels. DeepSeek-V3 states both all-to-all and PP comm "can be fully hidden" and reserves 20 SMs for comm ([arXiv 2412.19437](https://arxiv.org/abs/2412.19437), CONFIRMED); DeepSeek's hardware wish: offload comm from SMs to a co-processor and unify IB/NVLink domains (same paper, CONFIRMED). DeepSeek-V4 also warns that fully fused kernels make **power throttling** a limiter (CONFIRMED).
- Toy: a two-lane timeline (compute lane / network lane) with an overlap toggle.

### 5.3 FP8 training in practice (DeepSeek-V3)
Facts ([arXiv 2412.19437](https://arxiv.org/abs/2412.19437), CONFIRMED): all three linear GEMMs (Fprop, Dgrad, Wgrad) in FP8; **tile-wise 1x128 scaling for activations, block-wise 128x128 for weights**; promote partial accumulations to FP32 at intervals on CUDA cores; E4M3 everywhere; cache/dispatch activations in FP8, optimizer states BF16; embeddings, output head, gating, norms, attention ops kept in BF16/FP32; relative loss error vs BF16 < 0.25%. Fine-grained scaling = outlier in one tile doesn't ruin the whole tensor's range.

### 5.4 Failures
Llama 3 405B, 54-day window ([arXiv 2407.21783](https://arxiv.org/abs/2407.21783), CONFIRMED): **466 interruptions** (47 planned, **419 unexpected**) -> about one every 3 hours; 78% hardware-attributed; GPU issues = 58.7% of unexpected (faulty GPU 148, HBM3 72, SRAM 19, etc.); software bug 54; network switch/cable 35; only 3 needed manual intervention; **>90% effective training time**; storage fabric 240 PB, 2 TB/s sustained, checkpoints 1 MB-4 GB per GPU; stragglers: "even a single straggler can slow down thousands of other GPUs"; 1-2% diurnal throughput swing from temperature; silent data corruption: 6 cases. Toy: scale failure rate with GPU count (MTBF_cluster ≈ MTBF_gpu / N) and show how checkpoint interval trades lost work vs checkpoint overhead.
Kimi K2 claims zero loss spikes/crashes with MuonClip over 15.5T tokens ([arXiv 2507.20534](https://arxiv.org/abs/2507.20534), CONFIRMED) — training-stability, not hardware reliability.

### 5.5 Back-of-envelope time and cost
- Training FLOPs ≈ **6·N·D** (N = active params, D = tokens; 2ND forward + 4ND backward; ignores attention FLOPs, which matter at long context).
- GPU-hours = 6ND / (peak × MFU × 3600). Days = GPU-hours / (GPUs × 24).
- Checks (DERIVED):
  - Llama 3 405B: 6 × 405e9 × 15.6e12 = 3.79e25 FLOPs ✓ matches "3.8 × 10²⁵". Model card says 30.84M H100-hours for 405B ([model card](https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_1/MODEL_CARD.md), CONFIRMED). Average implied: 3.79e25/(30.84e6×3600) ≈ 340 TFLOPs/GPU ≈ 34% of 989 (below the 38-43% peak MFU because it includes downtime and the long-context phases).
  - DeepSeek-V3: 6 × 37e9 × 14.8e12 = 3.3e24 FLOPs; pre-training 2.664M H800-hours -> ≈ 340 TFLOPs/GPU. DeepSeek reports **2.788M GPU-hours total, 180K H800-hours per trillion tokens = 3.7 days on 2,048 GPUs, "$5.576M at $2/GPU-h"**, excluding research/ablations ([arXiv 2412.19437](https://arxiv.org/abs/2412.19437), CONFIRMED). 
  - Cost-slider: $ = GPU-hours × $/GPU-h. Typical rental $2-4/H100-h (REPORTED common knowledge; not sourced).

---

## 6. Recent big training runs (2025-2026)

Blanks are intentional. "GPUs" = disclosed by the lab or vendor unless noted.

| Model | Params (total/active) | Tokens | Hardware & GPU count | Notes | Source / tag |
|---|---|---|---|---|---|
| DeepSeek-V3 (Dec 2024) | 671B / 37B | 14.8T | **2,048 H800** | PP16/EP64/ZeRO-1, FP8, 2.788M GPU-h | [arXiv](https://arxiv.org/abs/2412.19437) CONFIRMED |
| Llama 3.1 405B (2024) | 405B dense | 15.6T | up to **16,384 H100** | 38-43% MFU | [arXiv](https://arxiv.org/abs/2407.21783) CONFIRMED |
| Llama 4 Behemoth (2025, unreleased at the time) | 288B active, 16 experts | >30T (family) | **32K GPUs**, FP8, 390 TFLOPs/GPU | | [Meta](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) CONFIRMED |
| Kimi K2 (Jul 2025) | 1.04T / 32B | 15.5T | H800 cluster; **count not disclosed** (min unit 256 GPUs) | MuonClip, zero crashes | [arXiv](https://arxiv.org/abs/2507.20534) CONFIRMED |
| Kimi K3 (Jul 2026; weights ~27 Jul) | 2.8T / 104B (16 of 896 experts) | not found | **not disclosed** | KDA hybrid attention, MoonEP, 1M context, ~2.5x scaling efficiency vs K2 | [arXiv](https://arxiv.org/abs/2607.24653) CONFIRMED (arch/infra), [Let's Data Science](https://letsdatascience.com/news/kimi-k3-debuts-as-28-trillion-parameter-open-weight-model-cac7da05) REPORTED (dates) |
| DeepSeek-V4 (2026) | Pro 1.6T / 49B; Flash 284B / 13B | Pro 33T, Flash 32T | **GPU count and chip mix not disclosed** in the paper; mega-kernel validated on NVIDIA GPUs and Huawei Ascend NPUs | FP4 expert weights (QAT), Muon, hybrid CSA/HCA attention, 1M ctx; at 1M ctx Pro uses 27% of V3.2's FLOPs and 10% KV | [arXiv 2606.19348](https://arxiv.org/abs/2606.19348) CONFIRMED |
| GLM-5 (Feb 2026) | 744B / 40B (256 experts, 8 active) | 28.5T | **UNVERIFIED.** Secondary sources say "100,000 Ascend 910B, no NVIDIA"; the arXiv paper only describes *adapting inference/kernels* to seven Chinese chip platforms | Paper's only Ascend content is inference/deployment kernels (grep for training cluster/NPU found none); treat Ascend-only-training claim as unsupported | [arXiv 2602.15763](https://arxiv.org/abs/2602.15763) CONFIRMED (adaptation text, no training-cluster claim found); claim itself from [Let's Data Science](https://letsdatascience.com/blog/china-trained-frontier-ai-model-glm-5-without-nvidia) UNVERIFIED |
| Qwen3-Max / Qwen3.x | >1T (Max); Qwen3 open models up to 235B | 36T (Qwen3) | not disclosed | MFU +30% vs Qwen2.5-Max (claim) | [Qwen3 report](https://arxiv.org/html/2505.09388v1) token count CONFIRMED by reference; MFU claim REPORTED |
| Mistral Large 3 (Dec 2025) | MoE | | ~**3,000 H200** | | [Mistral](https://mistral.ai/news/mistral-3) REPORTED (via search summary, page not fully read) |
| **Mistral Large 4** (announced **6 Oct 2026**, weights by end of Oct) | **1T / 49B** | not stated | "trained from scratch on **3,800 NVIDIA Grace Blackwell GPUs** in Mistral's own datacenters in Europe" | user's claim **confirmed** (GB200 vs GB300 not specified) | [Mistral](https://mistral.ai/news/mistral-large-4/) CONFIRMED |
| Nemotron 3 Ultra (Jun 2026) | 550B / 55B (hybrid Mamba-MoE) | ~20T | trained on **GB200**, NVFP4; count not disclosed | NVFP4 pre-training at scale | [arXiv 2606.15007](https://arxiv.org/pdf/2606.15007) CONFIRMED (GB200 mention), tokens REPORTED |
| xAI Colossus (Grok) | n/a | n/a | site-level claims: Colossus 1 ~230K GPUs, Colossus 2 ~550K GB200/GB300, 2 GW total | Not a single-run number | [Introl](https://introl.com/blog/xai-colossus-2-gigawatt-expansion-555k-gpus-january-2026) REPORTED (low quality; UNVERIFIED) |
| OpenAI / Google / Anthropic frontier runs | | | Not disclosed per run. Google trains on TPUs (Ironwood pods up to 9,216 chips); OpenAI deploying Rubin and MI450 (6 GW AMD deal, first 1 GW from H2 2026) | | [Google docs](https://docs.cloud.google.com/tpu/docs/tpu7x) CONFIRMED; [AMD](https://ir.amd.com/news-events/press-releases/detail/1294/aai-2026-amd-delivers-full-stack-compute-for-the-agentic-ai-era) CONFIRMED |

Takeaways: (1) disclosed single-run GPU counts cluster in 2K-32K (H800/H100-class) and ~4K Blackwell for 2026 European open-weight runs; (2) labs increasingly **don't** disclose counts (K3, V4); (3) 2026 shift: MoE at 1-3T total, FP4/NVFP4 and MXFP4 entering pre-training, Muon-family optimizers, hybrid/linear attention requiring new CP schemes.

---

## 7. Teaching notes: toys, formulas, existing explainers

### 7.1 Toy ideas and the formulas they need

1. **Memory calculator** (params slider 1B-3T, precision dropdown BF16/FP8/NVFP4 for weights, optimizer dropdown Adam/Muon/8-bit, ZeRO stage 0-3, DP size N, TP/PP sliders, seq len, micro-batch, recompute none/selective/full). Output: GB per GPU stacked bar (weights / grads / optimizer / activations) vs HBM line for H100 80 / B200 180 / B300 288. Formulas: section 3.1-3.4. Per-GPU states with TP t, PP p, ZeRO stage z: `16Ψ/(t·p)` scaled by stage formulas over N_DP. Activation per layer: `s·b·h·(34 + 5·a·s/h)/t`; layers per stage L/p; in-flight micro-batches ≈ p for 1F1B. "GPUs needed" = ceil(16Ψ / HBM_usable).
2. **Roofline plot**: log-log, x = arithmetic intensity, y = TFLOPs; ridge at peak/BW per chip and precision dropdown (use table 1.4). Sliders: batch size B (matmul intensity ≈ B), sequence length. Dots for "training matmul (B=4M)", "decode (B=8)", "layernorm (I≈1)", "attention naive vs flash". Formula: attainable = min(P, I·BW).
3. **Parallelism "slicer"**: show a weight matrix being cut column-wise/row-wise (TP), layers cut into stages (PP), batch cut (DP), experts scattered (EP). Click to see collectives and bytes.
4. **Ring all-reduce animation**: N GPUs in a ring, chunks flowing; counter of bytes/rank = 2(N−1)/N·S; compare with naive all-to-one. Toggle NVLink vs IB bandwidth -> time bar.
5. **Pipeline-bubble timeline**: grid of (stage × time); schedule dropdown GPipe / 1F1B / interleaved(v) / zero-bubble / DualPipe; sliders p and m. Show bubble% = (p−1)/(m+p−1) (GPipe, 1F1B), /v for interleaved; DualPipe: (p/2−1)(F&B+B−3W) using F=B/… ratios. DualPipe's official Table 2 gives formulas to cite.
6. **Expert-parallel all-to-all**: tokens colored by destination expert flying across GPUs; slider for top-k, number of EP ranks, imbalance (Zipf skew) -> idle GPUs; overlap toggle using V4's inequality C/B ≤ 6,144 FLOPs/byte.
7. **Topology explorer**: draw 8-GPU node, NVL72 rack, pods, oversubscribed spine (Llama 3: 16/rack, 3,072/pod, 24K cluster, 1:7); drag a parallelism dimension onto a layer of the network and see a "bandwidth needed vs available" meter.
8. **Failure simulator**: N GPUs, per-GPU MTBF slider, checkpoint interval; show effective training time. Calibrate with Llama 3: 419 unexpected interruptions/54 days on 16K GPUs ≈ one per 3.1 h, >90% effective.
9. **Cost/time calculator**: 6·N_active·D / (GPUs × peak × MFU) -> days; presets for DeepSeek-V3 (2,048 H800, 14.8T, 37B active; 2.664M GPU-h / 2,048 ≈ 54 days for pre-training), Llama 3 405B, Mistral Large 4 (1T/49B, 3,800 GB GPUs; token count unknown — leave a slider).
10. **Number-format explorer**: bit layouts FP32/BF16/FP8/MXFP4/NVFP4; scale-factor block of 16 vs 32; drag a distribution with an outlier to show why per-block scales beat per-tensor scales (DeepSeek's 1×128 tiles).

### 7.2 Best existing explainers / interactive pages
- Hugging Face **Ultra-Scale Playbook** (interactive, 5D parallelism, memory widgets): https://huggingface.co/spaces/nanotron/ultrascale-playbook (reachable; JS app, body not extractable here)
- Google **How to Scale Your Model** (rooflines, TPU/GPU, sharding math, Llama-3 examples): https://jax-ml.github.io/scaling-book/ (content CONFIRMED)
- Horace He, **Making Deep Learning Go Brrrr**: https://horace.io/brrr_intro.html
- Modal **GPU Glossary** (SMs, tensor cores, memory hierarchy, with diagrams): https://modal.com/gpu-glossary
- EleutherAI **Transformer Math 101** (6ND, memory formulas): https://blog.eleuther.ai/transformer-math/
- Lilian Weng, **How to Train Really Large Models on Many GPUs**: https://lilianweng.github.io/posts/2021-09-25-train-large/
- Stas Bekman, **ML Engineering Open Book** (networking, failure handling): https://github.com/stas00/ml-engineering
- Hugging Face **Smol Training Playbook**: https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook
- Microsoft **ZeRO/DeepSpeed** blog with animated memory-sharding figure: https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/
- Papers: ZeRO https://arxiv.org/abs/1910.02054 · Megatron sequence-parallel/activation recompute https://arxiv.org/abs/2205.05198 · Ring Attention https://arxiv.org/abs/2310.01889 · FlashAttention https://arxiv.org/abs/2205.14135 · Llama 3 https://arxiv.org/abs/2407.21783 · DeepSeek-V3 https://arxiv.org/abs/2412.19437 · DeepSeek-V4 https://arxiv.org/abs/2606.19348 · Kimi K2 https://arxiv.org/abs/2507.20534 · Kimi K3 https://arxiv.org/abs/2607.24653
- Code to link: DualPipe https://github.com/deepseek-ai/DualPipe · DeepEP https://github.com/deepseek-ai/DeepEP

(All URLs above returned HTTP 200 when checked on 2026-10-07; I did not read the interactive body of the Ultra-Scale Playbook, so don't quote specifics from it without opening it.)

---

## Open uncertainties / things I did not verify
- Rubin: HBM bandwidth (22 vs 19.2 TB/s), BF16/FP8 dense numbers, and whether "35 PF NVFP4 training" is strictly dense. Use NVFP4 35 PF (dense, training) for training toys.
- AMD MI455X/MI450 per-chip FLOPs/bandwidth: from 2025 announcements, final silicon numbers not confirmed. Helios first shipments ~end Q3 2026 per secondary reports; AMD itself says "in production".
- Megatron-LM TP paper and Ring Attention details cited from knowledge, not re-fetched.
- GLM-5 training hardware; Qwen3-Max infra; xAI numbers; Mistral Large 3 H200 count (only via search summaries).
- No MFU figures found for DeepSeek-V3/V4, Kimi, Mistral Large 4; my 340 TFLOPs/GPU figures are DERIVED estimates that ignore attention FLOPs.
- GPU counts for DeepSeek-V4, Kimi K3, Nemotron 3 Ultra are not disclosed in the sources I read.
- Typical rental $/GPU-hour is common knowledge, not sourced here.
