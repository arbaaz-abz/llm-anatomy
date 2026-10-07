# 02 — The Training Recipe (pretraining → mid-training → SFT → RL → distillation)

Research date: 2026-10-07. Scope: recipe layer only (hardware/parallelism is covered elsewhere).

**Confidence tags.** `[CONFIRMED]` = I read it in the primary document (tech report/model card/official blog) during this research. `[REPORTED]` = only in secondary press or blog coverage, or a primary I could only see through a search snippet. `[UNVERIFIED]` = plausible but I could not check it. `[BG]` = textbook/background knowledge I did not re-fetch (it carries an arXiv link where one exists).

Primary documents actually read in full text: DeepSeek-V4 report, DeepSeek-V3.2 report, GLM-5 report, Kimi K3 report, MiniMax-M2 series report, MiMo-V2-Flash report, Nemotron 3 Super report, gpt-oss model card, Mistral Large 4 blog.

---

## 0. Executive summary (the 2026 picture in ten lines)

1. Pretraining is still plain next-token cross-entropy, now at **25–33T tokens** for open frontier MoEs (DeepSeek-V4 32–33T, MiniMax-M2 29.2T, GLM-5 28.5T, MiMo-V2-Flash 27T, Nemotron 3 Super 25T) `[CONFIRMED]`. Kimi K3 did **not** disclose its token count `[CONFIRMED: absent from report]`.
2. **Muon** replaced AdamW at the frontier-open tier: Kimi K2/K3, GLM-4.5/5, DeepSeek-V4 all use it `[CONFIRMED]`. AdamW remains in MiMo-V2-Flash and Nemotron 3 Super `[CONFIRMED]`.
3. **Mid-training** is now a named stage: context extension plus quality/agentic/reasoning data, often during the LR-decay phase (GLM-5: 32K/1T → 128K/500B → 200K/50B tokens).
4. **RLVR with a GRPO-family algorithm** is the mainstream RL engine. Nobody uses a learned critic any more in the open reports. Every report patches GRPO for stability (clip-higher, no KL, token-level loss, importance-sampling corrections).
5. The headline 2026 shift is **specialist RL, then multi-teacher on-policy distillation (OPD)** to merge the specialists: DeepSeek-V4, Kimi K3, MiMo-V2-Flash and GLM-5 (cross-stage) all do this `[CONFIRMED]`.
6. **Agentic RL** in executable sandboxes (SWE, terminal, search) is now the dominant RL compute sink: thousands to hundreds of thousands of environments, fully async rollouts.
7. Non-verifiable domains use **generative reward models with rubrics** (DeepSeek-V4's actor-as-judge GRM, Kimi's agentic GRM), not scalar RMs.
8. The **training/inference mismatch** (different kernels, BF16 noise, MoE routing, FP4) is a first-class RL problem with named fixes (IcePop, TIS/MIS, Keep Routing, matching quantization in rollout and training).
9. Low precision: **native FP4 pretraining = Nemotron 3 Super only** (NVFP4). FP4 quantization-aware *post*-training = DeepSeek-V4, Kimi K3. gpt-oss's MXFP4 is a post-training quantization of MoE weights.
10. Mistral Large 4's disclosures are mostly confirmed; the ">10T tokens" and "hundreds of thousands of sandboxes" claims are not in Mistral's own blog (Section 6).

---

## 1. Pretraining

### 1.1 Objective and tokenization

- Objective: maximize likelihood of the next token; loss = cross-entropy `L = -(1/T) Σ log p_θ(x_t | x_<t)`. Perplexity = exp(L). MTP (multi-token prediction) is added as an auxiliary loss in DeepSeek-V4 (depth 1, weight 0.3, dropped to 0.1 when LR decay starts), GLM-5, MiniMax-M2, Nemotron 3 Super, and Kimi K3 `[CONFIRMED]`. MTP heads double as speculative-decoding drafts.
- Tokenization: byte-level BPE everywhere. Vocab sizes in the reports I read `[CONFIRMED]`:

| Model | Vocab |
|---|---|
| DeepSeek-V4 | 128K (V3 tokenizer plus a few special tokens) |
| GLM-5 | ~152K (151,552 in the report's table; 154,880 is the other column) |
| Kimi K3 | 160K |
| MiniMax-M2 | 200,064 |
| gpt-oss | `o200k_harmony` (extends o200k), ~201K `[CONFIRMED]` |

Pattern: 128K–200K, larger for multilingual models. Bigger vocab means fewer tokens per document but a larger embedding/softmax. DeepSeek also keeps Fill-in-Middle (FIM) and "token-splitting" from V3.

### 1.2 Data pipeline

Common shape across reports: crawl → heuristic filters → model-based quality classifiers → exact/fuzzy dedup → domain mixture with upsampling → (late) annealing mix.

- Kimi K3: four text domains (Web, Code, Math, Knowledge) each filtered by "rule-based heuristics, classifier-based quality scoring, and deduplication", with per-domain sampling rates set by small-model ablations; **knowledge and math corpora are rephrased** (style- and perspective-diverse prompting, chunk-wise generation, fidelity checks against the source). `[CONFIRMED]` (K3 report §3.1)
- MiniMax-M2: model-based reward scoring plus auxiliary classifiers; balanced sampling that upweights quality; "code, mathematics, and STEM significantly upsampled." `[CONFIRMED]` (§3)
- DeepSeek-V4: filters "batched auto-generated and templated content" to avoid model collapse; math/code remain core; larger multilingual corpus for long-tail culture; **long-document emphasis** (scientific papers, technical reports); agentic data injected in mid-training; documents packed to minimize truncation; sample-level attention masking. `[CONFIRMED]` (§4.1)
- GLM-5: code and reasoning prioritized early; ~10M issue–PR pairs (~160B unique tokens) concatenated with repo files and diffs for SWE. `[CONFIRMED]`
- Synthetic data: rephrasing (Kimi), agent-trajectory synthesis (GLM-5 mid-training), and LLM-generated reasoning traces are all standard. The tech reports give recipes, not ratios.
- Multilingual: Qwen3 36T tokens over 119 languages `[REPORTED: search snippet of arXiv 2505.09388]`; Mistral Large 4 160+ languages `[CONFIRMED: blog]`.

### 1.3 Token budgets and scaling laws

- Kaplan et al. 2020 (arXiv 2001.08361): loss follows power laws in params, data, compute; favored bigger models. `[BG]`
- Chinchilla (arXiv 2203.15556): compute-optimal is roughly **~20 tokens per parameter** with params and tokens scaling together. `[BG]`
- Reality in 2026 is heavy **over-training** because inference cost, not training cost, dominates the lifetime bill. For MoE the relevant denominator is arguably active params. Computed from `[CONFIRMED]` numbers: DeepSeek-V4-Pro 33T / 49B active ≈ 670 tokens per active param; V4-Flash 32T / 13B ≈ 2,460; Nemotron 3 Super 25T / 12B active ≈ 2,080; Olmo 3 ~5.9T for 7B/32B dense ≈ 185–840. These are 10–100x past Chinchilla.
- Kimi K3 ran dedicated scaling-law studies to retune batch size, LR, tokens-per-parameter and model shape, claiming **~2.5x scaling efficiency over K2** (architecture + data + recipe) `[CONFIRMED]`.
- Rule of thumb for an explainer: frontier open models ≈ 10–40T tokens, i.e. a few thousand tokens per active parameter.

### 1.4 Learning-rate schedules

| Model | Schedule | Source |
|---|---|---|
| DeepSeek-V4 | 2000-step linear warmup, **constant** peak (Flash 2.7e-4, Pro 2.0e-4) for most of training, then **cosine decay to 10% of peak** near the end; batch size ramped to 75.5M (Flash) / 94.4M (Pro) tokens; seq len 4K→16K→64K→1M | `[CONFIRMED]` |
| Nemotron 3 Super | **WSD**: 200B-token warmup to 4.5e-4, long plateau, **minus-sqrt decay over final 5T of 25T** tokens to 4.5e-6 | `[CONFIRMED]` |
| SmolLM3 | WSD, 2000 warmup steps, linear decay to 0 over last 10% | `[REPORTED]` (HF blog via search) |
| Kimi K3 | **Cosine**, 1% warmup, wd 0.1. Their scaling-law study explicitly found cosine beat WSD *when each is tuned separately* (optimal peak LR and batch differ by schedule) | `[CONFIRMED]` |
| MiMo-V2-Flash | AdamW, two-stage cosine (stage 1 decays to 1e-4 over 10T, stage 2 to 3e-5 over 4T) | `[CONFIRMED]` |
| GLM-5 | cosine decay (follows GLM-4.5) | `[CONFIRMED]` |

Takeaway: "WSD vs cosine" is not settled. WSD/constant-then-decay is popular because you can branch a decay ("annealing") off a stable checkpoint, and the decay phase is where labs inject the highest-quality data (mid-training). MiniMax-M2 splits 19.9T "constant phase" + 9.3T "decay phase" (which is also where context extends 8K→32K→192K) `[CONFIRMED]`.

### 1.5 Optimizers

- **AdamW** (β1 0.9, β2 0.95, wd 0.1) is the legacy default and is still used for embeddings, head, norms.
- **Muon** orthogonalizes the momentum matrix with Newton–Schulz iterations. Kimi's "Muon is Scalable" (arXiv 2502.16982) added weight decay and update-RMS matching so AdamW LRs transfer `[BG]`. **MuonClip** (Kimi K2): Muon plus QK-Clip, rescaling a head's Q/K weights when its max attention logit exceeds a threshold; K2 trained 15.5T tokens with zero loss spikes `[REPORTED: search result of K2 report, arXiv 2507.20534]`.
- Adoption `[CONFIRMED]` in 2026 reports:
  - **DeepSeek-V4**: Muon for most modules; AdamW for embedding, head, RMSNorm weights and mHC biases. Hybrid Newton–Schulz: 10 iterations, 8 with coefficients (3.4445, −4.7750, 2.0315) then 2 with (2, −1.5, 0.5). Momentum 0.95, wd 0.1, update RMS rescaled to 0.18. They **skip QK-Clip** because RMSNorm on queries and KV already prevents logit blow-up.
  - **Kimi K3**: **Per-Head Muon** (orthogonalize each attention head's Q/K/V block separately so large-scale heads don't dominate).
  - **GLM-5**: "Muon Split", same idea; needed to make MLA match GQA performance under Muon.
  - Not Muon: MiMo-V2-Flash (AdamW), Nemotron 3 Super (AdamW). MiniMax-M2's report text I extracted names no optimizer `[UNVERIFIED]`.

### 1.6 Stability tricks

- **QK-norm**: RMSNorm on Q and K (DeepSeek-V4 applies it to queries and KV entries `[CONFIRMED]`); used by many open dense/MoE models since OLMo-2/Qwen3 `[BG]`.
- **Logit soft-capping** (Gemma 2) was largely replaced by QK-norm in later Gemma versions `[BG, UNVERIFIED for 2026]`.
- **z-loss** (penalize log-partition drift of the output softmax; PaLM, OLMo) `[BG]`.
- New in V4 `[CONFIRMED]`: **Anticipatory Routing** (compute MoE routing indices with weights from Δt steps earlier, enabled automatically after a loss spike, ~20% overhead only when active), **SwiGLU clamping** (linear part clamped to [−10, 10], gate capped at 10; credited to OpenAI 2025), **mHC** (manifold-constrained hyper-connections). DeepSeek says spikes tracked MoE-layer outliers and that the theoretical reason these tricks work is open.
- MoE load balancing: aux-loss-free bias updates (V4: bias speed 0.001, tiny balance loss 1e-4); Kimi K3 uses **Quantile Balancing**.

### 1.7 Precision

- BF16 mixed precision is the baseline. **FP8** training is common in the open tier (DeepSeek-V3 lineage; MiMo-V2-Flash explicitly "FP8 mixed precision" over 27T tokens; V4 keeps non-expert params FP8) `[CONFIRMED]`.
- **FP4**: three distinct things, do not conflate:
  1. **Native NVFP4 pretraining**: Nemotron 3 Super trained 25T tokens in NVFP4 (2D block-scaled weights, 1D block-scaled activations/gradients, stochastic rounding on gradients; **final 15% of layers, attention projections, latent projections, MTP and embeddings stay BF16**, Mamba output projection MXFP8) `[CONFIRMED]`. NVIDIA's earlier paper trained a 12B model on 10T tokens in NVFP4 (random Hadamard transforms, 2D scaling, stochastic rounding, loss within ~1% of FP8) (arXiv 2509.25149) `[REPORTED via search]`. Nemotron says NVFP4 produces ~3x more zero-valued weight gradients than BF16 at the same token count, a cost they monitor `[CONFIRMED]`.
  2. **FP4 quantization-aware training in post-training**: DeepSeek-V4 (MXFP4 on MoE expert weights and the CSA indexer QK path; FP32 master weights → FP4 → lossless dequant to FP8; rollouts use true FP4) and Kimi K3 (MXFP4 weights + MXFP8 activations, QAT from SFT onward, rollout and training share the same scheme) `[CONFIRMED]`.
  3. **Post-training quantization (PTQ)**: gpt-oss quantizes MoE weights to MXFP4 (~4.25 bits/param) so the 120B fits an 80GB GPU `[CONFIRMED: model card]`. Press often says "natively trained in MXFP4"; the card says "post-trained the models with quantization of the MoE weights."

### 1.8 Lab-by-lab pretraining snapshot

| Model | Params (total/active) | Tokens | Optimizer | Notes |
|---|---|---|---|---|
| DeepSeek-V4-Pro / Flash | 1.6T/49B ; 284B/13B | 33T / 32T | Muon + AdamW | 1M ctx; FP4+FP8; CSA/HCA attention; dense attention first 1T tokens `[CONFIRMED]` |
| GLM-5 | 744B / – | 28.5T total (27T base + mid-training) | Muon Split | DSA attention; ctx 4K→200K `[CONFIRMED]` |
| Kimi K3 | 2.8T/104B | **undisclosed** | Per-Head Muon | NoPE + KDA linear attention; ctx 8K→64K pretrain, 256K→1M in cooldown `[CONFIRMED]` |
| MiniMax-M2 | 229.9B/9.8B | 29.2T (19.9T + 9.3T decay) | n/a in extracted text | vocab 200K, ctx 8K→32K→192K `[CONFIRMED]` |
| MiMo-V2-Flash | 309B/15B | 27T | AdamW | FP8 `[CONFIRMED]` |
| Nemotron 3 Super | 120B/12B | 25T | AdamW | native NVFP4, WSD `[CONFIRMED]` |
| Olmo 3 (fully open) | 7B / 32B | ~5.9T (Dolma 3 Mix) + 100B mid-train (Dolmino) | – | `[REPORTED]` (search + arXiv 2512.13961 skim) |
| SmolLM3 | 3B | 11.2T | – | 3 stages, WSD, 384 H100 x 24 days `[REPORTED]` |

---

## 2. Mid-training

Definition: a distinct stage between "main pretraining" and post-training, run on a smaller token budget with a re-weighted mixture: high-quality, reasoning, code, synthetic, agentic data, and **progressive context extension**. Often coincides with LR annealing.

Evidence `[CONFIRMED]`:
- **GLM-5 §2.3**: context grown over three stages, **32K (1T tokens) → 128K (500B) → 200K (50B)**; long documents and synthetic agent trajectories upsampled at later stages; SWE data (160B tokens). Their overview: base 27T → "distinct mid-training phase to progressively extend context length from 4K to 200K, focusing on long-context agentic data."
- **MiniMax-M2**: 9.3T-token decay phase mixes short-text decay data with long-context data (code concatenation, naturally long PDFs, thematically related document packing); context 8K → 32K → 192K.
- **Kimi K3**: four-stage curriculum, 8K→64K during pretraining, **256K→1M during the cooldown**, plus synthetic long-context tasks solvable only by attending across the full window. NoPE means no RoPE rescaling is needed.
- **DeepSeek-V4**: seq-len schedule 4K→16K→64K→1M inside pretraining; sparse attention introduced at 64K after 1T dense tokens with a lightning-indexer warm-up; agentic data in mid-training.
- **Olmo 3**: 100B-token Dolmino mid-training then 65K context extension `[REPORTED]`.
- Conceptual claim you can make: *context extension is cheap because it is concentrated in a small fraction of the budget* (K3 says this explicitly).

---

## 3. SFT / instruction tuning

- **Chat templates**: special tokens delimit roles and tool calls. Examples `[CONFIRMED]`: DeepSeek-V4 `<think>…</think>` plus a new XML-style `|DSML|` tool-call schema (they found XML reduces escaping errors); gpt-oss "harmony" roles System > Developer > User > Assistant > Tool; Kimi K3 XTML template; GLM-5 updated template with interleaved/preserved/turn-level thinking.
- **Loss masking**: standard SFT computes loss only on assistant tokens (prompt masked). GLM-5 goes further: in agent trajectories "erroneous segments are retained but masked out in the loss" so the model sees error-correction without reinforcing the error; environment/tool outputs are also excluded from the loss in agentic RL `[CONFIRMED]`.
- **Data scale**: Olmo 3 SFT ~2.3M reasoning traces distilled from QwQ-32B and DeepSeek-R1 `[REPORTED]`; DeepSeek-R1 used ~800K SFT samples (600K reasoning + 200K non-reasoning) and the same set to distil small models (arXiv 2501.12948) `[BG]`.
- **Reasoning traces / cold start**: SFT on long chain-of-thought gives RL a readable starting policy. K3: "SFT stage establishes a high-quality cold-start policy for the subsequent RL stage," trajectories synthesized by earlier Kimi specialists plus multi-stage verification and human-in-loop annotation. DeepSeek-V3.2 uses a cold-start to merge reasoning with tool-use ("thinking in tool-use") `[CONFIRMED]`. Qwen3's recipe: long-CoT cold-start → reasoning RL → thinking-mode fusion → general RL (arXiv 2505.09388) `[REPORTED]`.
- **SFT data builders**: rejection sampling (GLM-5 for logic problems, filtering to problems the previous model finds hard), expert-RL-generated trajectories, executable-environment trajectories.
- **Thinking modes as a product feature**: DeepSeek-V4 Non-think / Think High / Think Max (different length penalties and context windows during RL); Kimi K3 low/high/max effort; gpt-oss low/medium/high; Nemotron low-effort prompts (~1–2% of RL prompts) `[CONFIRMED]`.

---

## 4. Preference learning and RL (core section)

### 4.1 Classic RLHF (PPO)

1. Collect human preference pairs; train a **reward model** (RM) `r(x, y)` with Bradley–Terry loss.
2. Optimize the policy with **PPO**: sample responses, score with RM, estimate advantages with a learned **value model** (critic) via GAE, update with the clipped surrogate `min(r·A, clip(r, 1−ε, 1+ε)·A)`, plus a **KL penalty** `β·KL(π‖π_ref)` to the SFT reference to prevent reward hacking (InstructGPT, arXiv 2203.02155) `[BG]`.
3. Cost: four models in memory (policy, reference, RM, critic). That is why critic-free methods took over for reasoning RL.
4. 2026 status: RLHF survives as a **late-stage polish** for style/helpfulness/safety, typically with generative RMs. Nemotron 3 Super runs RLVR (21 environments), then SWE-RL, then a distinct **RLHF stage** with a principle-following **GenRM** (initialized from Qwen3-235B-A22B-Thinking-2507, trained on HelpSteer 3) `[CONFIRMED]`. GLM-5's General RL uses rule + outcome-RM + GRM hybrid rewards and anchors on human-written responses to avoid model-style drift `[CONFIRMED]`.

### 4.2 DPO and offline preference methods

- **DPO** (arXiv 2305.18290) turns the RLHF objective into a supervised loss on (chosen, rejected) pairs relative to a reference policy: no RM, no sampling loop. Variants: SimPO (reference-free, length-normalized), KTO (binary good/bad labels), ORPO (SFT+odds-ratio in one stage), APO `[BG]`.
- Where used in 2026: **open, resource-constrained, fully-open pipelines**. Olmo 3: SFT → DPO (~200K "Delta Learning" pairs, chosen/rejected from stronger vs weaker models) → RLVR (~105K prompts) `[REPORTED]`. SmolLM3 used APO `[REPORTED, BG]`. None of the big open MoE reports (DeepSeek-V4, GLM-5, Kimi K3, MiniMax-M2, MiMo-V2-Flash) list DPO as a main stage `[CONFIRMED: not present]`. Safe statement: **DPO is a cheap preference stage between SFT and RL for small/open models; frontier reasoning gains come from online RL.**

### 4.3 RLVR (verifiable rewards)

- Idea: replace the learned RM with a **program that checks the answer**: exact-match / symbolic-equivalence for math, unit tests or compilers for code, format checkers (`<think>…</think>`, boxed answer), constraint checkers for instruction following. Reward is mostly binary 0/1 (GLM-5: "binary outcome rewards" for math/science/code/TIR). Name introduced in Tülu 3 (arXiv 2411.15124) `[BG]`.
- **DeepSeek-R1-Zero / R1** (arXiv 2501.12948; Nature 2025) popularized it: GRPO directly on the base model with accuracy + format rewards produced long chain-of-thought, self-verification and "aha" reflection with no SFT; R1 added a small cold-start SFT, reasoning RL, rejection-sampling SFT (~800K), and a final RL stage; also released distilled small models `[BG]`.
- Why it creates "thinking" models: RL on checkable tasks rewards any token sequence that ends in the right answer, so the policy discovers that spending more tokens on decomposing, checking and backtracking raises reward; test-time compute (thinking length) becomes a trained, controllable quantity (hence reasoning-effort modes with per-problem token budgets: K3 overrides the reward with −1 if the trajectory exceeds τ·b₀(x)).
- Domain mix example (GLM-5 reasoning RL): math, science, code (Codeforces/TACO/SYNTHETIC-2-RL plus internal scientific coding), tool-integrated reasoning; prompts difficulty-filtered to ones GLM-4.7 solves rarely but stronger teachers can solve `[CONFIRMED]`.
- Env scale: Nemotron 3 Super 21 environments / 37 RL datasets `[CONFIRMED]`; DeepSeek-V3.2 1,827 synthetic environments (4,417 tasks) plus 85K agentic prompts `[CONFIRMED]`.

### 4.4 GRPO and its successors

**GRPO** (DeepSeekMath, arXiv 2402.03300) `[BG]`: for each prompt sample a group of G responses, score each, set advantage `A_i = (R_i − mean(R)) / std(R)`, apply PPO-style clipped ratio per token, optional KL to reference. **No critic**: the group mean is the baseline.

Variants and who actually uses what (all `[CONFIRMED]` unless noted):

| Method | Key change | Adopted by (in reports read) |
|---|---|---|
| **GRPO** | group-normalized advantage, no critic | DeepSeek-V4 specialists ("hyper-parameters closely aligned with prior research"), GLM-5 (backbone), MiMo-V2-Flash (ORM advantage in MOPD), Nemotron 3 Super ("asynchronous GRPO"), Olmo 3 (OlmoRL variant), Magistral |
| **DAPO** (arXiv 2503.14476) `[BG]` | clip-higher, dynamic sampling (drop all-correct/all-wrong groups), token-level loss, overlong penalty | Its tricks are everywhere: GLM-5 ε_high=0.28, Olmo 3 (zero-gradient filtering, clip-higher, token-level loss, no KL `[REPORTED]`), Magistral (ε_high 0.26–0.28, non-diverse group filtering `[REPORTED]`) |
| **Dr.GRPO** (arXiv 2503.20783) `[BG]` | remove length and std normalization (biases toward long wrong answers) | DeepSeek-V3.2 defines `A = R − mean(R)` with **no std division** (eq. in report) |
| **GSPO** (Qwen, arXiv 2507.18071) | sequence-level importance ratio and clipping; stabilizes MoE RL | Qwen3 2507 models per Qwen `[REPORTED]`; no other lab report I read adopts it |
| **CISPO** (MiniMax-M1, arXiv 2506.13585 `[BG]`) | clip the **importance weight** (stop-grad) instead of dropping tokens, so every token still gets gradient. M2 report: weight clipped to `[0, 1+ε_high]`, loss = `sg(r̂)·A·log π`, **reward-to-go minus trajectory baseline** (not group-normalized), composite reward = process reward + completion-time reward + task reward | MiniMax-M2/M2.5/M2.7 `[CONFIRMED]` |
| **IcePop-GRPO** (GLM-5) | mask tokens whose train/infer probability ratio ρ falls outside [1/β, β] (β=2); ε_low 0.2, ε_high 0.28; KL term **removed**; group size 32, batch 32, fully on-policy for reasoning RL | GLM-5 `[CONFIRMED]` |
| **DeepSeek-V3.2 GRPO patches** | (a) unbiased K3 KL estimator (importance-corrected), weak/zero KL for math; (b) **off-policy sequence masking** (drop sequences whose policy drifted too far); (c) **Keep Routing** (replay MoE expert choices from rollout); (d) **Keep Sampling Mask** (top-p/top-k mask reused in training) | DeepSeek-V3.2, V4 specialists |
| **Kimi policy optimization** | K1.5 used online mirror descent with KL regularization `[REPORTED]`; K3 follows the K2.5 algorithm with a **per-token regularization** tolerant of very stale data (partial rollouts span multiple iterations) | Kimi K2.5/K3 (exact loss not reproduced in K3 text) |
| **RLOO, REINFORCE++** (arXiv 2501.03262) | leave-one-out baseline; critic-free PPO tricks | Popular in open frameworks (OpenRLHF, TRL); **no frontier tech report I read lists them** `[UNVERIFIED adoption]` |

Cross-cutting recipe consensus (the "2026 GRPO"): **no or tiny KL**, **asymmetric clip with higher upper bound**, **token-level (not sample-level) loss aggregation**, **filter zero-variance groups**, **importance-sampling correction for rollout/trainer mismatch**, **MoE routing/sampling-mask replay**, and **length/effort control via reward**. Proprietary frontier labs (OpenAI, Anthropic, Google) do not disclose their RL algorithms; gpt-oss only says "similar CoT RL techniques as OpenAI o3" `[CONFIRMED]`.

### 4.5 Rewards for hard-to-verify tasks

- **Process vs outcome rewards**: PRMs score each reasoning step (Let's Verify Step by Step, arXiv 2305.20050 `[BG]`) but are costly and hackable; the R1-era consensus was outcome-only. The 2026 twist is *targeted* process signals in agent RL: MiniMax-M2 uses dense process rewards for language mixing, tool-format errors and well-structured intermediate steps, plus a wall-clock completion-time reward to encourage parallel tool calls `[CONFIRMED]`.
- **LLM-as-judge / rubric rewards (RLAIF)**:
  - **DeepSeek-V4 GRM**: no scalar RMs; curate rubric-guided RL data, and the **actor itself is the judge**, with RL applied to the judging ability too (joint optimization of generation and evaluation) `[CONFIRMED]`.
  - **Kimi K3 Agentic GRM**: tournament-style group reward with binary comparisons; the judge must read the output, **generate a rubric, score each candidate, write a scorepad**; a verbosity budget (σ·ℓ₀) makes over-long answers auto-lose to prevent length hacking `[CONFIRMED]`.
  - **GLM-5 hybrid**: rule rewards (precise, narrow) + ORMs (low variance, hackable) + GRMs (robust, higher variance) `[CONFIRMED]`.
  - Mistral Large 4: "reward models, unit tests, LLM judges, and static checks combined as needed for each task" `[CONFIRMED]`.

### 4.6 Agentic RL

Multi-turn RL where an episode is a long trajectory of reasoning, tool calls and observations in an executable environment; reward comes from tests, graders or judges. All `[CONFIRMED]`:

- **GLM-5**: fully **async, decoupled** inference/training via a Multi-Task Rollout Orchestrator on the `slime` framework; **TITO** (token-in-token-out) gateway avoids re-tokenization mismatch; **direct double-sided importance sampling** (token-level clip [1−ε_l, 1+ε_h] on rollout log-probs, without tracking historical checkpoints); DP-aware routing for KV-cache reuse; >10K verifiable SWE environments across 9 languages (RepoLaunch-built F2P/P2P tests), thousands of Docker terminal tasks (Harbor format), multi-hop search from a web knowledge graph (>2M pages), slide-generation environment.
- **Kimi K3**: three domain experts (general, general agents, coding agents) x three effort levels (low/high/max) = **9 experts**; **partial rollouts** (pause generation once fraction λ of trajectories finish; resume the rest next iteration inside persistent microVM sandboxes, **AgentEnv** with snapshot/resume/fork); million-token context agent trajectories with hundreds to thousands of tool calls; a **unified white-box environment** that instantiates harnesses like Claude Code, Codex, Kimi Code, OpenClaw, Hermes by composing modules (tools, prompts, context management, skills, subagents) so the model does not overfit one harness; knowledge-graph-guided task synthesis; kernel-optimization tasks with performance rewards.
- **MiniMax-M2 Forge**: agent-native RL; policy gradient on atomic (state, action) pairs so the policy is agnostic to context truncation/rewrites; windowed-FIFO scheduling, prefix-tree merging, white-box and black-box agent support; "hundreds of thousands of real-world environments" per M2.5 marketing `[REPORTED]`.
- **DeepSeek-V3.2/V4**: V3.2 synthesized 1,827 environments / 85K prompts and exceeded 10% of pretraining cost in post-training compute; V4 adds a preemptible, fault-tolerant rollout service (token-granular write-ahead log, because regenerating interrupted requests from scratch **biases toward short responses**), a million-token RL framework and sandbox infra; thinking traces are preserved across user turns in tool-use mode.
- **Nemotron 3 Super**: separate SWE-RL stage because SWE rollouts are slow and long; **NeMo Gym** open-sources most environments.
- **Qwen3.5**: described by Qwen as RL scaled across "million-agent environments" `[REPORTED]` (blog not fetchable; secondary only).

### 4.7 RL infrastructure

- **Async rollouts vs training**: synchronous RL leaves GPUs idle during long agent rollouts, so everyone decouples (GLM-5/slime, MiniMax Forge, Mistral, Nemotron "asynchronous GRPO", Kimi partial rollouts). Cost: data becomes **off-policy**, which is why importance-sampling corrections and staleness limits exist.
- **Training/inference mismatch**: rollout engines (SGLang/vLLM) and trainers (Megatron/FSDP) compute different probabilities for the same tokens. Fixes seen in the wild: IcePop masking (GLM-5), token-level truncated IS "TIS" (Yao et al.), sequence-level MIS, **FP16 instead of BF16** to shrink rounding mismatch (arXiv 2510.26788) `[REPORTED via search]`, deterministic top-k for sparse-attention indexers (GLM-5: non-deterministic CUDA top-k caused collapse in a few steps; `torch.topk` fixed it) `[CONFIRMED]`, MoE Keep Routing, and **using the same quantization in rollout and trainer** (K3, V4) `[CONFIRMED]`.
- **Frameworks**: slime (Z.ai) and Forge (MiniMax) are in-house; verl, OpenRLHF, TRL, prime-rl are the common open ones; NeMo RL/NeMo Gym (NVIDIA); MiMo-V2-Flash used SGLang + Megatron-LM with FP8 for RL and MOPD `[CONFIRMED]`. For verl/OpenRLHF/TRL/prime-rl specifics I did not read primary docs `[UNVERIFIED details]`.
- **Throughput** (Mistral): ~33B tokens/day at ~3k GPUs, ~16B trainable `[CONFIRMED]`.

### 4.8 Distillation and the on-policy shift

Three kinds, in order of how mainstream they are:

1. **Off-policy SFT distillation** (sequence-level): fine-tune a student on teacher-generated reasoning traces. Cheap, hugely common for small models (R1 distilled into Qwen/Llama ~800K samples; Olmo 3 SFT on QwQ/R1 traces) `[BG/REPORTED]`.
2. **Logit distillation** (token-level KL on teacher distributions) on fixed data: classic, used for small-model pretraining/mid-training in places; Qwen3 "strong-to-weak" includes an off-policy then on-policy phase `[REPORTED]`.
3. **On-policy distillation (OPD)**: the *student* samples; a teacher scores every token; loss = reverse KL (student ‖ teacher) per token, which is equivalent to using `sg(log π_T − log π_S)` as a dense per-token reward/advantage in an RL loop. Popularized by Thinking Machines (blog, 2025) and Qwen3; Thinking Machines claims large compute savings versus RL (an order of magnitude; "9–30x" figures are secondary) `[REPORTED]`.

2026 usage `[CONFIRMED]`:

| Model | OPD form |
|---|---|
| **DeepSeek-V4** | **Multi-teacher, >10 teachers, full-vocabulary reverse KL** `L = Σ w_i · KL(π_θ ‖ π_Ei)`; replaces the old mixed-RL stage (V3.2 pipeline otherwise unchanged); specialists still trained by SFT + GRPO. Full-vocab chosen because token-level KL estimates had high variance; teachers' last-layer hidden states cached and logits rebuilt on the fly |
| **Kimi K3 (MOPD)** | per-token reward `clip(sg log π_T/π_θ, ±R_max)` from one of the 9 experts chosen by (domain, effort); top-k logit variants gave no gain |
| **MiMo-V2-Flash (MOPD)** | `A = sg log π_T/π_θ + α·A_ORM` (adds GRPO outcome advantage); SFT → domain teachers by RL/SFT → MOPD; they report it preserves each teacher's peak without the usual capability trade-off |
| **GLM-5** | **on-policy cross-stage distillation** as the final stage: earlier checkpoints (SFT, Reasoning RL, General RL) are teachers, to undo catastrophic forgetting from sequential RL; advantage replaced by `sg log π_T/π_train` |

Correction to a popular summary: a Fireworks blog says V4 "replaces RL entirely with OPD." The paper (§5.1) says only the **mixed RL stage** was replaced; specialists are still RL-trained with GRPO. Use the paper.

How small models are made: (a) pretrain small on lots of tokens, (b) SFT on teacher traces, (c) OPD/logit distillation from a big specialist or merged model, (d) optional short RLVR. Mistral says Large 4 "leaned less on distillation" than peers `[REPORTED]`.

---

## 5. Canonical 2026 pipeline (in words)

```
1. PRETRAIN        10–35T tokens, next-token CE, Muon/AdamW, BF16/FP8 (FP4 rare),
                   4K–8K ctx, constant/WSD/cosine LR.        Adds: world knowledge, language, code/math ability.
2. MID-TRAIN       ~0.1–10T tokens (Olmo3 0.1T; GLM-5 1.55T in 3 ctx stages; MiniMax 9.3T decay),
   / ANNEAL        high-quality + reasoning + synthetic agent data, ctx -> 128K–1M.
                                                             Adds: long context, reasoning priors, agentic format fluency.
3. SFT (cold start) ~0.1–few M examples, loss only on assistant tokens, long-CoT + tool trajectories,
                   chat/tool template with <think> tags.     Adds: instruction-following, readable reasoning, tool syntax; gives RL a good start.
4. SPECIALIST RL   per domain (math/code/science, general chat, agentic SWE/search/terminal),
   (RLVR + GRM)    GRPO-family, G≈8–32 samples/prompt, thousands of envs, async + partial rollouts,
                   several effort levels (low/high/max).     Adds: the bulk of reasoning + agent capability.
                   Compute: DeepSeek-V3.2 post-training >10% of pretraining cost; K3 shows RL FLOPs scaling curves (no absolute number).
5. MERGE / OPD     multi-teacher on-policy distillation (DS-V4: >10 teachers; K3: 9; GLM-5: cross-stage).
                                                             Adds: one model with all specialists' peaks, no regressions.
6. FINAL POLISH    general RL / RLHF with GenRM, safety, style; MTP healing; QAT to FP4/FP8 for deployment.
```

Token/compute budget shape: pretraining dwarfs everything in tokens (tens of T) but RL is now measured as a fraction of pretraining *compute* (>10% at DeepSeek-V3.2) and is growing. Exact RL token counts are mostly undisclosed except Mistral's throughput figure `[CONFIRMED]`.

---

## 6. Mistral Large 4 ("Le Chonk"): claim check

Primary: https://mistral.ai/news/mistral-large-4/ (preview announced 2026-10-06; weights promised by end of October).

| Claim in brief | Verdict | Evidence |
|---|---|---|
| ~3,800–4,000 GB-series GPUs | **CONFIRMED (3,800)**; "4,000" and "about two months" are **REPORTED** (VentureBeat: "roughly two months on 4,000 Nvidia Grace Blackwell GPUs"; the same "~4,000 / ~two months" appears in a Decoder search snippet; the Decoder page itself says 3,800 and gives no duration) | Blog: "3,800 NVIDIA Grace Blackwell GPUs in Mistral's European datacenters"; The Register: "around 52 NVL72 racks" |
| >10T text tokens | **REPORTED, weak.** Not in the blog. kingy.ai cites a Mistral "training-content summary dated Oct 6" ("more than 10 trillion text tokens", crawler data through July 2026). I could not open that summary. Caution: the same "more than 10 trillion tokens" figure appears in the EU public training summaries for **Ministral 3**, so this may be conflation | https://kingy.ai/blog/mistral-large-4-specs-benchmarks-pricing/ |
| 160+ languages | **CONFIRMED** | Blog: "more than 160 languages, including every official language of the European Union" |
| Async RL, ~33B tokens/day at ~3k GPUs | **CONFIRMED** | Blog: "At our current scale (3k GPUs), a single training run produces roughly 33 billion tokens per day, of which around 16 billion are trainable completion tokens"; "autoscaling fleet of actors generates tens of thousands of rollouts in parallel while model training proceeds asynchronously"; "novel methods... minimize off-policy drift" |
| "Mistral Forge" environments | **CONFIRMED (name only)**: "same training, customization, and RL environment we offer our customers through Mistral Forge" | Blog |
| Hundreds of thousands of sandbox environments; "leaned less on distillation" | **REPORTED** (not in blog) | The Decoder / Pasquale Pillitteri summaries of Lample's remarks |
| Params | Blog/VentureBeat: 1T total, 49B active; The Decoder/kingy: 1.05T (52B incl. embeddings). Flag as a rounding discrepancy | |
| Context | Mistral card reportedly 1M; independent evals reportedly ~512K `[REPORTED]` | kingy.ai |

Other confirmed blog facts: natively multimodal MoE (text out only), hybrid instruct+reasoning; RL tasks span single-turn chat, scientific problem solving, safety alignment, factuality, long-horizon tool use; verification via "reward models, unit tests, LLM judges, and static checks"; RL still in flight at preview time. Their earlier Magistral paper (arXiv 2506.10910) is the best public source for Mistral's RL algorithm (KL-free, clip-higher 0.26–0.28, token-average loss, non-diverse-group filtering, async generators/trainers) `[REPORTED via search]`; Large 4's own algorithm is **not** disclosed.

---

## 7. Teaching notes: animation/toy ideas per concept

| Concept | Interactive idea (control → live output) |
|---|---|
| Cross-entropy / next-token | Slider for the probability assigned to the true next token → loss bar `-log p` and perplexity; click tokens in a sentence to see per-token loss "heat". |
| Tokenization | Type text, choose vocab size (32K/128K/200K) → colored BPE chunks, tokens per doc, embedding-table size in MB. |
| Scaling laws / over-training | Sliders for params and tokens → loss from a Chinchilla-style fit, a "tokens per param" gauge with markers (Chinchilla 20, Nemotron ~2,000, V4-Flash ~2,500), and lifetime cost with an inference-volume slider. |
| LR schedules | Toggle cosine / WSD / constant+cosine-tail / minus-sqrt tail → LR curve; draggable "branch an anneal here" marker showing why annealing from a stable checkpoint is cheap. |
| Muon vs Adam | A 2D weight-update visual: momentum matrix singular values before/after Newton–Schulz iterations (slider 0–10 iterations) → singular values converge to 1. DeepSeek's 8+2 coefficient switch as a toggle. |
| Stability | Attention-logit growth chart with/without QK-norm; "outlier" slider that triggers a loss spike; Anticipatory Routing toggle. |
| FP8/FP4 | Slider for bits → representable-value grid on a number line, with block-scale on/off to show why microscaling works; Nemotron's "which layers stay BF16" diagram. |
| Mid-training / context extension | Timeline bar with token budget per stage (GLM-5 32K/128K/200K); context window "grows" as you scrub, with a needle-in-haystack pass/fail. |
| SFT loss masking | A chat transcript where tokens are highlighted: prompt tokens grey (masked), assistant tokens green (loss); toggle "mask error segments" (GLM-5) and "tool outputs" to see which tokens get gradient. |
| Chat templates / thinking modes | Same conversation rendered under ChatML vs harmony vs `<think>` with Non-think/High/Max toggles. |
| RLHF/PPO clip | Slider for ratio r and advantage sign → clipped vs unclipped objective curve, shaded flat region where gradient is zero; add asymmetric ε_low/ε_high (0.2/0.28) to show clip-higher. |
| KL penalty | β slider with a reward-hacking toy (policy drifts to a degenerate high-reward string): reward vs KL curve; show why many 2026 recipes set β≈0 and rely on clipping/masking. |
| DPO | Pair of responses, slider for implicit reward margin → logistic loss; show reference-policy anchoring. |
| **GRPO group** | Prompt with N samples (slider 2–64) each with a 0/1 reward → mean, std, advantages (Dr.GRPO toggle: with/without std). Live: all-correct/all-wrong groups produce **zero gradient** (flash "filtered by dynamic sampling"). Sliders for pass-rate p to show signal peaking near p=0.5. |
| Token-level vs sequence-level loss | Two responses of different length, one right one wrong → show per-token weights under sample-mean vs token-mean (length bias in original GRPO). |
| CISPO vs PPO clip | Token importance ratio slider → PPO drops the token's gradient outside the clip; CISPO keeps the gradient scaled by a clipped weight (`[0,1+ε_high]`). |
| GSPO | Per-token ratios vs one geometric-mean sequence ratio; show variance of the ratio on long sequences. |
| Training/inference mismatch | Two bars of token probabilities (trainer vs rollout engine) with noise slider (BF16 vs FP16); IcePop mask turns tokens outside [1/β, β] grey. |
| Async RL | Gantt chart of GPU timelines: synchronous (idle during long rollout) vs async vs partial rollouts; staleness counter and IS-weight histogram. |
| Agentic RL | Replay of an agent trajectory (think → tool → observe → ...) with outcome reward at the end; toggle process-reward events (format error, language mix) and completion-time reward. |
| Generative reward model | Rubric builder: judge generates rubric, scores two candidates, tournament; verbosity-budget slider that makes the long answer auto-lose (K3). |
| Distillation / OPD | Teacher and student next-token bars per position; toggle SFT (teacher's text, fixed) vs OPD (student's own sample, teacher scores every token); slider for reverse vs forward KL (mode-seeking vs mean-seeking); "N teachers" with weights for the DeepSeek-V4 merge. |
| Whole pipeline | A "recipe builder" stacking the six stages with token/compute bars; preset buttons for DeepSeek-V4, GLM-5, Kimi K3, Olmo 3. |

### Existing explainers (all URLs seen in search results or fetched this session)

- Hugging Face, *The Smol Training Playbook* (pretraining decisions, ablations, loss spikes, post-training): https://huggingfacetb-smol-training-playbook.hf.space/
- Nathan Lambert, *RLHF Book* (reward modeling, policy gradients, RLVR, direct alignment, on-policy distillation): https://rlhfbook.com/
- Sebastian Raschka, *The State of Reinforcement Learning for LLM Reasoning* (GRPO, DAPO, Dr.GRPO, RLVR): https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training
- Jay Alammar, *The Illustrated DeepSeek-R1*: https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1
- Thinking Machines, *On-Policy Distillation*: https://thinkingmachines.ai/blog/on-policy-distillation/
- Karpathy, nanochat (tokenizer → pretrain → midtrain → SFT → optional GRPO on GSM8K, runnable end-to-end; check repo for current stage names): https://github.com/karpathy/nanochat
- UNIPO, interactive visual explanation of GRPO-family algorithms (paper, arXiv 2605.11549; I did not open the tool): https://arxiv.org/pdf/2605.11549
- Interactive GRPO chapter (group size, advantage stats, GRPO vs PPO; quality unchecked): https://learn-ml-visualized-interactive.onrender.com/chapter/group-relative-policy-optimization-grpo
- Papers worth linking from pages: DeepSeek-V4 https://arxiv.org/abs/2606.19348 (note: the arXiv ID looks odd for an April 2026 submission, but the abstract page I fetched is the V4 paper), DeepSeek-V3.2 https://arxiv.org/abs/2512.02556, GLM-5 https://arxiv.org/abs/2602.15763, MiMo-V2-Flash https://arxiv.org/abs/2601.02780, MiniMax-M2 series https://arxiv.org/abs/2605.26494, Kimi K3 https://github.com/MoonshotAI/Kimi-K3 (report PDF `k3_tech_report.pdf`), Nemotron 3 Super https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf, gpt-oss https://arxiv.org/abs/2508.10925, Olmo 3 https://arxiv.org/abs/2512.13961, GSPO https://arxiv.org/abs/2507.18071, FP16 mismatch https://arxiv.org/abs/2510.26788.

---

## 8. Open uncertainties

- Kimi K3 pretraining token count and exact RL loss (K2.5 lineage) are not public in the text I read.
- Proprietary labs' recipes (OpenAI, Anthropic, Google) are undisclosed; everything above is the open-weight frontier plus gpt-oss.
- Qwen3.5 recipe details (tokens, GSPO use, environment counts) rest on secondary sources; Qwen's blog was not fetchable.
- DPO/RLOO/REINFORCE++ adoption statements are "absent from the reports I read," not proof of non-use.
- Mistral Large 4 token count (>10T) and sandbox count are secondary-sourced; weights and fuller technical detail were promised for late October 2026.
- Several `[BG]` items (Kaplan, Chinchilla, PPO/DPO/R1/DAPO details, z-loss, soft-capping) were not re-fetched; follow the linked arXiv IDs before quoting numbers.
