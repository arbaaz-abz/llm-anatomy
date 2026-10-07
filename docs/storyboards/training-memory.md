# Where training memory goes (`training-memory`)

Track: training · Section: gpus · Prereqs: gpu-primer (matches `shared/concepts.json`)
Next: `parallelism` (the only slug whose `prereqs` list `training-memory`)
Status: approved (expert review)
Sources: 03 §3.1, §3.2, §3.3, §3.4, §4.1, §4.6, §7.1 (toy 1), §7.2 · 02 §1.5, §1.7 · 05 §1.1, §1.2, §2.
Expert review (Fable 5.1, 2026-10-07) applied; see §13.

Running example: GPT-3's shape (175B parameters, 96 blocks, d_model 12,288, 96 heads, 2,048-token
context; the `decoder-anatomy` preset) trained with today's standard recipe: BF16 weights, Adam with
FP32 states. GPT-3 itself (2020) predates BF16 training; the page says so in a visible line ("GPT-3's
shape, today's recipe"). The ZeRO paper's own example (7.5B parameters on 64 GPUs) is try-this 1, so
the learner can check the page against a published table. `parallelism` picks up GPT-3's 43.75 GB per
GPU and adds tensor and pipeline splits.

## 1. Learning objective
After this page you can add up what one parameter costs in training (2 bytes of weight, 2 of gradient,
8 of Adam state, 4 of FP32 master copy: 16 bytes; frames 1–4), estimate saved activations and what
recomputation buys (frames 5–7, try-this 2), compute per-GPU state under ZeRO stages 1–3 (frames 8–10,
try-this 1), and explain why a model with 1T parameters needs hundreds of GPUs just to hold its training
state, even when it is a mixture of experts that computes with a few percent of them (frame 11,
try-this 3).

## 2. Misconceptions to correct
- **Misconception:** "A model that fits in memory for inference fits for training." → **Reality:**
  inference needs the weights (2 bytes per parameter in BF16); mixed-precision training with Adam needs
  16: weights, gradients, two Adam moments and an FP32 master copy. GPT-3's 350 GB of weights become
  2.8 TB of training state. Corrected by frames 1–4. (03 §3.1)
- **Misconception:** "The weights are the big part." → **Reality:** the optimizer's FP32 state (master
  copy plus two moments) is 12 of the 16 bytes. That is why ZeRO's first stage, which shards only that,
  removes most of the memory. Corrected by frames 3–4 and 9, try-this 1. (03 §3.1, §3.4)
- **Misconception:** "Splitting training across more GPUs splits the memory." → **Reality:** plain data
  parallelism gives every GPU a full copy: 64 GPUs hold 64 copies of 2.8 TB. Only sharding (ZeRO / FSDP)
  divides the state, and ZeRO's three stages shard state, not activations, which belong to each GPU's
  own data. Corrected
  by frames 8–10 and try-this 2. (03 §3.4, §4.1)
- **Misconception:** "Activations are small next to the weights." → **Reality:** for one 2,048-token
  GPT-3 sequence with nothing recomputed, the 96 blocks save 275 GB, more than three H100s, and 70% of
  that is the attention-score grid. Recomputation trades it for compute: 4.8 GB at about a third more
  compute. Corrected by frames 5–7 and try-this 2. (03 §3.2)
- **Misconception:** "A mixture-of-experts model with 49B active parameters trains like a 49B model."
  → **Reality:** compute follows active parameters, memory follows total. At the 16-byte Adam recipe,
  DeepSeek-V4-Pro's 1.6T parameters need 25.6 TB of state, 320 H100s before a single activation (240
  with its actual recipe, Muon at 12 bytes). Corrected by frame 11
  and try-this 3. (03 §3.3)

## 3. Hook and intuition (final wording)
**Hook:** GPT-3's weights fit in 350 GB. Why does training it need 2.8 TB before it has seen a single
token?

Training keeps four things for every parameter. The weight itself, in BF16: 2 bytes. Its gradient,
which backpropagation writes once per step: 2 more. Adam's two running averages, which decide how far
each weight moves, kept in FP32 because they are tiny and must stay precise: 8 bytes. And an FP32 master
copy of the weight, because adding a small update to a BF16 number would often round it away: 4 bytes.
That is 16 bytes per parameter, eight times the 2 bytes that inference needs, and it answers the hook:
the other 2.45 TB are 0.35 TB of gradients and 2.1 TB of optimizer state (moments and master copy).

Then come activations, the intermediate results the forward pass saves because the backward pass needs
them. They grow with the tokens in flight, not with the parameters, and at long sequence lengths they
dominate. The cheapest fix is to save less and recompute: keep only each block's input and run its
forward again during backward. That costs about a third more compute and saves most of the memory.

Finally, sharding. Many GPUs training on different data each hold a full copy of the state, so adding
GPUs adds copies, not room. ZeRO (Zero Redundancy Optimizer; PyTorch's FSDP is the same idea) gives each
GPU one slice of the state and has the GPUs pass the rest around just in time. Shard the optimizer
state, then the gradients, then the weights, and per-GPU state falls from 2.8 TB to 44 GB on 64 GPUs.
The price is traffic: the fully sharded stage moves about 1.5 times the data of plain data parallelism,
and ZeRO's stages leave activations alone, so they still have to fit on each GPU (`parallelism` splits
those).

## 4. Visual metaphor
A `shareBar` of one parameter's bytes builds left to right in frames 1–4 (weight · gradient · Adam
moments · master copy), with each segment's bytes printed in it. Under it, a "GPT-3 needs" counter (GB
and H100s) and a short row of `gpu` glyphs, the row collapsed to at most four glyphs plus a plain text
"+ N more" (lesson 18). Frames 5–7 switch to a `blockStack` of GPT-3's 96 blocks with the activations
each block saves drawn as a plain filled bar beside it whose length is GB (printed). Frames 8–10 show
three GPUs side by side as `gpu` glyphs with their GB printed (GPU 1, GPU 2, "61 others", GPU 64); the
followed GPU ("GPU 1") carries the selection outline in every frame it appears, and only GPU 1's
composition is drawn, as one full-width `shareBar` (420 px) under the row, so its smallest slice (the
ZeRO-1 optimizer slice, 32.8 of 732.8 GB = 4.5%) is 18.8 px (README lesson 19); slice labels print
beside the bar with a leader line where they don't fit inside. The `gpu` glyph's memory bar is the
only fullness encoding (GB held ÷ 80 GB, clamped; an overflow prints "needs 2,800 GB of 80"), and the
`shareBar` shows composition only, so no quantity is encoded twice. Frame 1 prints that on screen: "bar:
what one parameter costs · GPUs: how full 80 GB gets" (lesson 21).

Glyphs used (from spec §5.1 and approved proposals): `gpu` (memory bar = fullness), `block` (frame 6:
"attention scores" as a labeled block inside one zoomed block), `flow` (frame 10: weights all-gathered
from the other GPUs; `carry: 'weight'`, accepted for the Training build), `shareBar` and `blockStack` (both proposed by
`decoder-anatomy`, which names this page as a reuser; this page adds no new stacked bar). Part fills use
`decoder-anatomy`'s proposed `--part-1 … --part-6` tokens: weights `--part-1`, gradients `--part-2`,
Adam moments `--part-3`, master copy `--part-4`, activations `--part-5`; every segment prints its label.
The `--part-*` tokens are a categorical palette whose meanings are per page (on `decoder-anatomy` they
mean embedding, attention, MLP, …); nothing course-wide should read `--part-2` as "gradient".

New glyphs proposed: none.

Plain labeled marks: "GPT-3's shape, today's recipe (GPT-3 itself trained in 2020)" (frame 1); the
"+ N more" GPU count; the "80 GB" capacity label on each `gpu`; "formula: Korthikanti et al. 2022,
micro-batch 1" (frame 5); "1/64" slice labels in frames 9–10 as text, not glyphs.

Terms introduced (one per frame): bytes per parameter (frame 1), gradient (2), optimizer state (3),
master weights (4), activations (5), selective recomputation (6), full recomputation (7), data parallel
(8), ZeRO (9), ZeRO-3 / FSDP (10), mixture of experts, defined on screen with a link to `moe` (11).
Terms assumed from `gpu-primer`: HBM, BF16, FP32, GB of HBM per chip. Terms assumed from general
background: parameter, forward and backward pass (linked to `decoder-anatomy`).

Indexing: GPUs are numbered from 1 on screen ("GPU 1 … GPU 64"); no memory addresses on this page.

## 5. Animation script
Numbers from `math/training-memory.js` (reproducer in §6). Decimal units throughout (1 GB = 10⁹
bytes); capacity math uses usable HBM where data has it and nominal otherwise, always labeled (H100 80 GB
nominal; H200 141 GB nominal; B200 180 GB usable of 192 nominal; B300 288 GB nominal).

| # | On screen | What moves | Caption (final wording) | Numbers shown |
|---|---|---|---|---|
| 1 | Empty `shareBar` labeled "one parameter". Text "GPT-3's shape, today's recipe" and "bar: what one parameter costs · GPUs: how full 80 GB gets". Counter "GPT-3 needs: 350 GB = 4.4 H100s"; five `gpu` glyphs (each ~96 px wide), four full and the fifth 38% full. | The first segment "weight 2 B" grows; the counter counts 0 → 350 GB; the GPUs fill one by one. | Training starts with the weights: 2 bytes per parameter in BF16. GPT-3's 175 billion parameters already need 350 GB, more than four H100s. | 175e9 × 2 = 350 GB · 350 ÷ 80 = 4.4 H100s |
| 2 | Same bar. | Segment "gradient 2 B" grows; counter 350 → 700 GB; the GPU row collapses to four full glyphs and the text "+ 5 more" (lesson 18). | Backpropagation writes a gradient for every weight, the same shape and size. That is 2 more bytes per parameter, 700 GB so far. | 2 + 2 = 4 B · 700 GB · 8.8 H100s |
| 3 | Same bar. | Segment "Adam moments 8 B" grows (the widest so far); counter 700 → 2,100 GB. | Adam keeps two running averages per parameter, called optimizer states, in FP32 so tiny updates stay precise. They add 8 bytes, the biggest share. | 2 × 4 B = 8 B · 2,100 GB · 26.3 H100s |
| 4 | Same bar, now complete, with the total "16 B" printed at its end. | Segment "master copy 4 B" grows; counter 2,100 → 2,800 GB; GPU row "+ 31 more". | The optimizer updates an FP32 master copy of each weight, then rounds it to BF16 for the next step. Total: 16 bytes per parameter, 2.8 TB for GPT-3. | 2 + 2 + 8 + 4 = 16 B · 2.8 TB · 35 H100s · weights alone are 2 of 16 (12.5%) |
| 5 | `blockStack` of 96 blocks (two drawn, "⋮ × 96"). Beside each drawn block, a bar "saved: 2.87 GB". Total "275 GB". Plain marks: "one 2,048-token sequence"; "formula: Korthikanti et al. 2022, micro-batch 1". | A `flow` dot (carry `activation`) runs down the stack; each block leaves its bar behind; the total counts up. | The backward pass needs numbers the forward pass computed, called activations, so each block saves them. For one 2,048-token sequence, GPT-3's 96 blocks save 275 GB. | per block: 2,048 × 12,288 × (34 + 80) bytes = 2.87 GB · × 96 = 275.4 GB |
| 6 | Zoom on one block: its 2.87 GB bar splits into "attention scores 2.01 GB" and "everything else 0.86 GB". | The scores segment fades out; the bar shrinks; the total recounts 275 → 82 GB. | Most of it is the attention-score grid. Recomputing it during the backward pass, or never storing it as FlashAttention does, cuts the save to 82 GB. | scores 80 of 114 parts (70%) · 34 × 2,048 × 12,288 = 0.86 GB per block · × 96 = 82.1 GB |
| 7 | Back to the stack. Each block's bar shrinks to a sliver "input 50 MB"; a second, dashed `flow` (carry `activation`) re-runs each block during the backward pass. Readout "extra compute: about +33%". | Bars shrink; the re-run dot passes once. | Full recomputation keeps only each block's input and reruns its forward pass during backward. Saved activations drop to 4.8 GB for about a third more compute. | 2 × 2,048 × 12,288 = 50.3 MB per block · × 96 = 4.83 GB · forward 1 + backward 2 + rerun 1 = 4 units instead of 3 |
| 8 | Three `gpu` glyphs labeled GPU 1 (outlined), GPU 2, GPU 64, with "61 others" between them, each with the overflow label "needs 2,800 GB of 80"; under the row, GPU 1's full-width `shareBar` (weight · gradient · moments · master). Each GPU gets a different data chip "batch 1", "batch 2", "batch 64". | Data chips drop into each GPU; the bars appear identical. | Sixty-four GPUs, each training on different data, is data parallelism. Every GPU keeps a full copy, so each one still needs all 2.8 TB. | per GPU 2,800 GB · 64 copies = 179.2 TB in total |
| 9 | Same GPUs, each printing its GB. On GPU 1's full-width bar the moments-and-master segment shrinks to an 18.8 px slice, labeled "slice 1 of 64" beside it with a leader line; GPU 2 and GPU 64 print "slice 2 of 64", "slice 64 of 64" under their glyphs. | GPU 1's optimizer segment splits; 63 pieces fly off toward the other GPUs; every GPU's GB recounts. | ZeRO shards the optimizer states: each GPU keeps one sixty-fourth and updates only that slice. Per-GPU memory falls from 2,800 GB to 733 GB. | 4 × 175e9 + 12 × 175e9 ÷ 64 = 700 + 32.8 = 732.8 GB · weights still 350 GB, gradients 350 GB |
| 10 | Same GPUs; gradient and weight segments shard too. GPU 1's bar (rescaled to its new total, full width): 5.5 + 5.5 + 32.8 = 43.75 GB; the `gpu` memory bar drops to 55%. For layer 1, `flow` arrows (carry `weight`) from GPU 2 and GPU 64 into GPU 1 carry that layer's weight slices (label "all-gather before the layer, forward and again backward"). A small readout: "+ activations 4.8 GB (full recompute) = 48.6 GB: fits". | Segments split and scatter; GPU 1's GB counts down; the all-gather arrows run once. | Shard gradients and weights too, and each GPU holds 44 GB of state. Before each layer runs, forward and backward, the GPUs gather its weights: that is ZeRO-3, or FSDP. | ZeRO-2 on the way: 388.3 GB · ZeRO-3: 2,800 ÷ 64 = 43.75 GB · traffic 1.5× plain data parallel · + 4.83 GB activations = 48.58 GB of 80 |
| 11 | A plain table: "1T parameters × 16 B = 16 TB" and rows: H100 (80 GB nominal) 200 · H200 (141 GB nominal) 114 · B200 (180 GB usable) 89 · B300 (288 GB nominal) 56 GPUs. Second line: "DeepSeek-V4-Pro's size at the 16-byte recipe: 1.6T total, 49B active → 25.6 TB → 320 H100s (its own recipe, Muon, is 12 B: 240)". On-screen definition: "mixture of experts: each token uses a few of many expert blocks (see `moe`)". | Rows type in; the DeepSeek line's "49B active" dims while "1.6T total" stays bright. | A trillion parameters need 16 TB of training state: 200 H100s before any activations. A mixture of experts pays here for every expert, even the ones a token skips. | 1e12 × 16 = 16 TB · ÷ 80 GB = 200 · ÷ 141 GB = 114 · ÷ 180 GB = 89 · ÷ 288 GB = 56 · V4-Pro at 16 B: 25.6 TB → 320 / 182 / 143 / 89; at 12 B: 240 / 137 / 107 / 67 |

Determinism: every frame is a pure function of (step, progress). Reduced motion shows each frame's end
state. Frames 1–4 draw `trainingBytesPerParam(TRAINING_RECIPES.adam)`; frames 5–7
`activationBytes({ layers: 96, seq: 2048, microBatch: 1, hidden: 12288, heads: 96, recompute })`;
frames 8–10 `zeroPerGpuBytes({ params: 175e9, recipe: adam, stage, dp: 64 })`; frame 11
`gpusToHoldStates`.

Caption check: counts in §12 (all ≤ 2 sentences, ≤ 30 words, no operators).

## 6. Toy
"Will it fit?" A training-memory calculator for one GPU. A visible line above the controls: "Per-GPU
memory counts training state and saved activations only; real runs also lose 20–50% of HBM to buffers,
fragmentation and communication workspaces (03 §3.3)."

**Controls**
| id | Label | Type | Range / values | Default | Presets |
|---|---|---|---|---|---|
| `model` | Model | Preset chips | ZeRO paper 7.5B · GPT-3 175B · Llama 3.1 405B · DeepSeek-V3 671B · Kimi K2 1.04T · DeepSeek-V4-Pro 1.6T | GPT-3 175B | parameter counts from `data/models.json` (§8); MoE presets use **total** parameters |
| `recipe` | Optimizer recipe | Segmented | Adam, FP32 states (16 B) · Adam, BF16 moments (12 B) · Muon, one FP32 momentum (12 B) | Adam 16 B | – |
| `stage` | ZeRO stage | Segmented | 0 (plain data parallel) · 1 · 2 · 3 (FSDP) | 0 | – |
| `dp` | Data-parallel GPUs | Slider (powers of 2) | 1 … 1,024 | 64 | – |
| `gpu` | GPU | Preset chips | H100 80 GB nominal · H200 141 GB nominal · B200 180 GB usable (192 nominal) · B300 288 GB nominal | H100 | `hardware.json/{h100,h200,b200,b300}`: `b200.hbm_usable_gb`, nominal `hbm_gb` for the rest (the B200 chip reads `b200`, not `gb200-nvl72`'s 186) |
| `seq` | Sequence length (GPT-3 only) | Slider (powers of 2) | 1,024 … 8,192 tokens | 2,048 | – |
| `recompute` | Saved activations (GPT-3 only) | Segmented | store everything · skip attention scores (selective) · block inputs only (full) | store everything | – |

Activations use the dense GPT layer formula, so they are computed only for GPT-3 (the one preset whose
shape is in `data/`, via `decoder-anatomy`'s keys). For every other preset the activation segment is
replaced by the visible note "activations: shape not modeled here (MoE / MLA layers); states only".

**Live outputs**
| Output | Formula / `math/` function | Units / format |
|---|---|---|
| Bytes per parameter, by part | `trainingBytesPerParam(TRAINING_RECIPES[recipe])` (`recipe` is a `TRAINING_RECIPES` entry, picked by the control) | B, in a one-parameter `shareBar` |
| Per-GPU state: weights, gradients, optimizer (master + moments) | `zeroPerGpuBytes({ params, recipe, stage, dp })` | GB (`formatBytes`, decimal), 2 decimals |
| Per-GPU activations (GPT-3 only) | `activationBytes({ layers: 96, seq, microBatch: 1, hidden: 12288, heads: 96, recompute })` | GB |
| **Per-GPU total and verdict** | state total + activations vs the chip's usable HBM | GB; "fits" (`--sem-ok`) / "does not fit" (`--sem-bad`), always with the numbers |
| Composition | the four or five parts | `shareBar` with labels and % |
| GPUs just to hold the state (any sharding, no activations) | `gpusToHoldStates({ params, bytesPerParam: total, hbmBytes })` | integer |
| Traffic vs plain data parallel | stage 3 → 1.5×, otherwise 1× (03 §3.4: 3Ψ vs 2Ψ words per step) | multiplier |
| Extra compute from recomputation | full → about +33% (one extra forward); selective → "small, not quantified here" (hover adds: "it recomputes only the attention-score grid"); none → 0 | % / text |

**Try this** (each leads to a named insight)
1. Pick "ZeRO paper 7.5B", keep 64 GPUs, and step ZeRO stage 0 → 1 → 2 → 3: 120 → 31.41 → 16.64 →
   1.88 GB per GPU, the ZeRO paper's table. → **Insight: optimizer state is 12 of the 16 bytes, so
   sharding it alone (stage 1) removes most of the memory; stage 3 divides all 16 bytes by the number of
   GPUs.**
2. GPT-3, ZeRO-3, 64 GPUs, H100: state is 43.75 GB. With "store everything" the total is 319.2 GB (does
   not fit); "skip attention scores" 125.9 GB (still no); "block inputs only" 48.6 GB (fits). Now set
   ZeRO stage 0: 2,804.8 GB even with full recomputation. → **Insight: ZeRO's stages shard state,
   never activations; recomputation shrinks activations, never state; you need both.** Activations are
   per GPU because each GPU holds its own data; `parallelism` splits them further with tensor and
   pipeline parallelism.
3. DeepSeek-V4-Pro's size (1.6T total, 49B active) at the Adam recipe, H100: "GPUs just to hold the
   state" reads 320. Switch to B200: 143; B300: 89. Switch the recipe to Muon (V4's own): 67. Then pick GPT-3 (175B dense): 35 H100s. → **Insight: memory
   follows total parameters, compute follows active ones, so a MoE that computes like a 49B model
   needs the training memory of a 1.6T one.** Muon's single momentum buffer saves a quarter of it.

**`math/training-memory.js`** (new module, owned by this page; settled: `math/memory.js` stays KV and
inference only, owned by `kv-cache`; `parallelism` imports from here; pure, no DOM, inputs never
mutated; tests first). Units: bytes.

```js
TRAINING_RECIPES   // frozen, bytes per parameter by part
//   adam:            { weight: 2, grad: 2, master: 4, optimizer: 8 }   // Adam m and v in FP32
//   adamBf16Moments: { weight: 2, grad: 2, master: 4, optimizer: 4 }   // DeepSeek-V3-style BF16 moments
//   muon:            { weight: 2, grad: 2, master: 4, optimizer: 4 }   // one FP32 momentum buffer
trainingBytesPerParam({ weight, grad, master, optimizer }) → { weight, grad, master, optimizer, total }
//   adam → total 16 · adamBf16Moments → 12 · muon → 12
//   { weight: 2, grad: 4, master: 0, optimizer: 0 } → total 6   (Kimi K2's resident params + FP32 grad buffer, §8)
zeroPerGpuBytes({ params, recipe, stage, dp, modelShards = 1 }) → { weights, grads, optimizer, total }
//   recipe: a TRAINING_RECIPES entry (an object, not its name); per-GPU bytes; `optimizer` = master + moments; stage 1 divides optimizer by dp, 2 adds grads, 3 adds weights;
//   modelShards (= tp · pp) divides everything first; `parallelism` passes it. Throws on stage ∉ {0,1,2,3}.
//   7.5e9, adam, dp 64: stage 0 → 120e9 · 1 → 31.41e9 · 2 → 16.64e9 · 3 → 1.875e9
//   175e9, adam, dp 64: stage 0 → 2800e9 · 1 → 732.81e9 · 2 → 388.28e9 · 3 → 43.75e9
activationBytesPerLayer({ seq, microBatch, hidden, heads, tp = 1, recompute = 'none' }) → number
//   'none': s·b·h·(34 + 5·a·s/h) / tp · 'selective': 34·s·b·h / tp · 'full': 2·s·b·h   (03 §3.2)
//   GPT-3 (2048, 1, 12288, 96): none 2,868,903,936 · selective 855,638,016 · full 50,331,648
activationBytes({ layers, ...perLayerArgs }) → number      // layers × activationBytesPerLayer
//   GPT-3, 96 layers: none 275.41e9 · selective 82.14e9 · full 4.83e9
gpusToHoldStates({ params, bytesPerParam, hbmBytes }) → number   // ceil(params · bytesPerParam / hbmBytes)
//   hbmBytes is the chip's HBM (usable for B200, nominal otherwise; settled)
//   (1e12, 16, 80e9) → 200 · (1e12, 16, 141e9) → 114 · (1e12, 16, 180e9) → 89 · (1e12, 16, 288e9) → 56
//   (175e9, 16, 80e9) → 35 · (1.6e12, 16, 80e9) → 320 · (1.6e12, 16, 180e9) → 143 · (1.6e12, 16, 288e9) → 89
//   (1.6e12, 12, 80e9) → 240 · (1.6e12, 12, 180e9) → 107 · (1.6e12, 12, 288e9) → 67
```
One definition each (README lesson 16): "bytes per parameter" is always `trainingBytesPerParam(...).total`;
"per-GPU memory" is always `zeroPerGpuBytes(...).total + activationBytes(...)` (activations only for
GPT-3); "GPUs to hold the state" is always `gpusToHoldStates` with the chip's HBM (usable for B200, nominal for the others), labeled
"usable" or "nominal". No prose
number on the page is computed any other way.

Reproducer (run 2026-10-07 against the scratch implementation; re-run after the review with usable
HBM; re-run once the module exists):
```sh
node -e '
import("./math/training-memory.js").then((m) => {
  const G = 1e9, f = (x) => Number(x.toFixed(2)), A = m.TRAINING_RECIPES.adam;
  for (const k of Object.keys(m.TRAINING_RECIPES)) console.log(k, m.trainingBytesPerParam(m.TRAINING_RECIPES[k]).total);
  console.log("k2", m.trainingBytesPerParam({ weight: 2, grad: 4, master: 0, optimizer: 0 }).total);
  for (const [P, N] of [[7.5e9, 64], [175e9, 64]]) for (const s of [0, 1, 2, 3]) { const z = m.zeroPerGpuBytes({ params: P, recipe: A, stage: s, dp: N }); console.log(P / G, s, f(z.weights / G), f(z.grads / G), f(z.optimizer / G), f(z.total / G)); }
  const g3 = { seq: 2048, microBatch: 1, hidden: 12288, heads: 96, layers: 96 };
  for (const r of ["none", "selective", "full"]) console.log(r, m.activationBytesPerLayer({ ...g3, recompute: r }), f(m.activationBytes({ ...g3, recompute: r }) / G));
  for (const [P, b] of [[1e12, 16], [1.6e12, 16], [1.6e12, 12], [175e9, 16]]) console.log(P, b, [80e9, 141e9, 180e9, 288e9].map((h) => m.gpusToHoldStates({ params: P, bytesPerParam: b, hbmBytes: h })).join(" "));
});'
```
Output (2026-10-07): adam 16, adamBf16Moments 12, muon 12, k2 6. 7.5B, dp 64 (weights / grads /
optimizer / total GB): stage 0 15 / 15 / 90 / 120 · 1 15 / 15 / 1.41 / 31.41 · 2 15 / 0.23 / 1.41 /
16.64 · 3 0.23 / 0.23 / 1.41 / 1.88. 175B, dp 64: 0 350 / 350 / 2,100 / 2,800 · 1 350 / 350 / 32.81 /
732.81 · 2 350 / 5.47 / 32.81 / 388.28 · 3 5.47 / 5.47 / 32.81 / 43.75. Activations per layer / total:
none 2,868,903,936 / 275.41 GB · selective 855,638,016 / 82.14 GB · full 50,331,648 / 4.83 GB. GPUs (H100 80 /
H200 141 / B200 180 usable / B300 288): 1T → 200 / 114 / 89 / 56; V4-Pro at 16 B → 320 / 182 / 143 /
89; at 12 B → 240 / 137 / 107 / 67; GPT-3 → 35 / 20 / 16 / 10. Kimi K2: 6e12 ÷ 256 = 23.4 GB per GPU. Derived: GPT-3 ZeRO-3 +
full = 48.58 GB, + selective = 125.89 GB, + none = 319.16 GB; ZeRO-0 + full = 2,804.83 GB; score share
80 ÷ 114 = 70.2%.

Tests to write first: the worked examples; `zeroPerGpuBytes` at `dp = 1` equals stage 0 for every stage;
`total` = sum of parts; stage totals are non-increasing in stage; `activationBytesPerLayer` 'full' never
exceeds 'selective' never exceeds 'none'; `TRAINING_RECIPES` frozen; `kvBytesPerToken` and
`kvBytesPerTokenMla` stay in `math/memory.js` (this module imports nothing from it).

## 7. Show me the math
```tex
\text{bytes per parameter} = \underbrace{\htmlClass{hl-w}{2}}_{\text{weight}} + \underbrace{\htmlClass{hl-g}{2}}_{\text{gradient}}
 + \underbrace{\htmlClass{hl-o}{8}}_{\text{Adam } m,\,v} + \underbrace{\htmlClass{hl-o}{4}}_{\text{FP32 master}} = 16
```
```tex
\text{ZeRO-1: } 4\Psi + \frac{12\Psi}{N} \qquad
\text{ZeRO-2: } 2\Psi + \frac{14\Psi}{N} \qquad
\text{ZeRO-3: } \frac{16\Psi}{N}
```
```tex
\htmlClass{hl-act}{A_{\text{layer}}} = s\,b\,h\left(34 + \frac{5\,a\,s}{h}\right) \ \text{bytes}
\quad\to\quad 34\,s\,b\,h \ (\text{selective}) \quad\to\quad 2\,s\,b\,h \ (\text{full})
```
Shapes: Ψ parameters; N data-parallel GPUs; s sequence length, b micro-batch, h hidden size, a heads
(GPT-3: 2,048, 1, 12,288, 96); the `5as/h` term is the attention-score grid [a × s × s] (softmax output,
dropout mask and input). Color links: `hl-w`, `hl-g`, `hl-o`, `hl-act` = the `shareBar` segments
(`--part-1`, `--part-2`, `--part-3`/`--part-4`, `--part-5`). Notes shown in the panel: (a) the factor
34 assumes BF16 activations and a 4h MLP; (b) recomputation's cost: a full extra forward pass is about a
third of a forward + backward step (backward ≈ 2 forwards); (c) ZeRO's traffic: stages 0–2 move about 2Ψ
numbers per step (reduce-scatter + all-gather), stage 3 about 3Ψ.

## 8. In today's models (Oct 2026)
| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| Mixed-precision Adam: 2 + 2 + 12 = 16 B per parameter (ZeRO paper, 2019); 16–18 B is the safe teaching number | none (mechanism; ZeRO paper arXiv 1910.02054) | 03 §3.1 |
| ZeRO paper example: 7.5B on 64 GPUs → 120 / 31.4 / 16.6 / 1.9 GB | `models.json/zero-paper-7.5b.total_params` = 7.5e9 *(proposed; a teaching preset, note "ZeRO paper example")* | 03 §3.4 |
| GPT-3 shape (preset): 175B, 96 blocks, d_model 12,288, 96 heads, 2,048 context | `models.json/gpt-3.total_params`, `.layers`, `.d_model`, `.n_heads`, `.context_length` *(proposed by `decoder-anatomy`; same keys)* | 01 §4 via `decoder-anatomy` |
| DeepSeek-V3 (2024) stored optimizer moments in BF16; trained with ZeRO-1 only, because 16-way pipeline and 64-way expert parallelism already shrink per-GPU state | `models.json/deepseek-v3.optimizer_state_format` = "BF16", `.zero_stage` = 1 *(proposed; entry proposed by `paged-attention`)* | 03 §3.1, §4.1 |
| Kimi K2 (2025): 1.04T parameters; BF16 weights + FP32 gradient buffer = 6 B per parameter, about 6 TB over a 256-GPU model-parallel group (about 23 GB per GPU), leaving about 30 GB per GPU for its share of optimizer state, which is sharded across data-parallel ranks. The page labels the 6 B "weights and gradient buffer only, not comparable with 16" | `models.json/kimi-k2.total_params` = 1.04e12, `.resident_bytes_per_param` = 6 *(proposed entry)* | 03 §3.1, §3.3, §4.6 |
| Muon (Kimi K2/K3, GLM-5, DeepSeek-V4) keeps one momentum buffer instead of Adam's two: about 4 B less per parameter | `models.json/deepseek-v4-pro.optimizer` (existing); `kimi-k3.optimizer` = "Per-Head Muon" *(proposed)* | 03 §3.1; 02 §1.5 |
| DeepSeek-V4-Pro: 1.6T total, 49B active (3.1%); its size at the 16-byte Adam recipe would be 25.6 TB (a what-if: V4 trained with Muon and FP8/FP4 weights; at Muon's 12 B, 19.2 TB) | `models.json/deepseek-v4-pro.total_params`, `.active_params`, `.optimizer` | 03 §3.3 (MoE caveat); 02 §1.5, §1.7 |
| Llama 3.1 405B (2024): FSDP sharding optimizer states and gradients, but not re-sharding weights after the forward pass (to avoid a second all-gather) | `models.json/llama-3.1-405b.sharding` *(proposed)* | 03 §4.1 |
| Modern activation tricks: DeepSeek-V3 recomputes RMSNorm and MLA up-projections; Kimi K3 combines recompute, FP8 activations and CPU / remote offload | `models.json/deepseek-v3.activation_tricks`, `kimi-k3.activation_tricks` *(proposed, strings)* | 03 §3.2 |
| Llama 3.1 405B, DeepSeek-V3 total parameters for the presets | `models.json/llama-3.1-405b.total_params`; `deepseek-v3.total_params` = 671e9 *(proposed entry)* | 03 §4.6 |

## 9. Takeaways
1. Mixed-precision training with Adam costs 16 bytes per parameter (weight 2, gradient 2, Adam moments
   8, FP32 master copy 4), eight times inference's 2: GPT-3's shape needs 2.8 TB of state.
2. Activations scale with tokens in flight and can exceed the state; recomputation trades about a third
   more compute for most of that memory (GPT-3, one sequence: 275 GB → 4.8 GB).
3. Data parallelism copies the state; ZeRO / FSDP shards it (2.8 TB → 44 GB on 64 GPUs) at up to 1.5×
   the traffic; its three stages leave activations alone. Memory follows total parameters, so a 1T MoE needs about
   200 H100s just to hold its state.

## 10. Next and go deeper
Next: `parallelism` (tensor and pipeline splits that cut activations and the state ZeRO cannot reach)
· Related: `gpu-primer` (HBM sizes), `moe` (total vs active parameters).
Go deeper (brief 03 §7.2, 05 §1.2): Microsoft, *ZeRO & DeepSpeed* blog with the animated sharding figure
(https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)
· EleutherAI, *Transformer Math 101* (https://blog.eleuther.ai/transformer-math/) · Hugging Face,
*Ultra-Scale Playbook* (https://huggingface.co/spaces/nanotron/ultrascale-playbook).

## 11. Key-frame sketch
Frame 10 end state (GPT-3, ZeRO-3, 64 GPUs), desktop width. Numbers from the §6 reproducer.
```text
Where training memory goes        10 / 11  [<] [Play] [>]
  (GPU 1)          GPU 2      61 others      GPU 64
 +--------+     +--------+                 +--------+
 |########|<----|########|  <-- all-gather |########|
 |[=====  ]|    |[=====  ]|  weights, fwd  |[=====  ]|
 +--------+     +--------+   and bwd       +--------+
  43.75 GB       43.75 GB                   43.75 GB
 GPU 1: [w 5.5 |g 5.5 |optimizer 32.8              ] 43.75 GB
 ZeRO-0 2,800 | ZeRO-1 732.8 | ZeRO-2 388.3 | ZeRO-3 43.75
 + activations 4.83 GB (full recompute) = 48.58 GB of 80
 traffic: 1.5x plain data parallel
 Shard gradients and weights too, and each GPU holds ...
```

## 12. Open questions for the reviewer
Caption check (2026-10-07, after the review, the `rlvr-grpo` counter adapted): frames 1–11 are
23/2, 23/2, 24/2, 28/2, 26/2, 26/2, 26/2, 24/2, 24/2, 30/2, 29/2 (words/sentences); no operators.

**Data-pass keys**
- `models.json/zero-paper-7.5b` (new teaching entry): `total_params = 7.5e9`, note "ZeRO paper
  (arXiv 1910.02054) example; per-GPU 120 / 31.4 / 16.6 / 1.9 GB at N = 64" (03 §3.4, CONFIRMED by
  reference).
- `models.json/kimi-k2` (new entry): `total_params = 1.04e12`, `active_params = 32e9`,
  `pretrain_tokens = 15.5e12`, `resident_bytes_per_param = 6` (note: "BF16 params + FP32 grad buffer,
  about 6 TB over 256 GPUs; about 30 GB per GPU left for optimizer state"), `model_parallel_gpus = 256`
  (03 §3.1, §3.3, §4.6; CONFIRMED arXiv 2507.20534).
- `models.json/deepseek-v3` (entry proposed by `paged-attention`): add `total_params = 671e9`,
  `active_params = 37e9`, `optimizer_state_format = "BF16"`, `zero_stage = 1`, `activation_tricks`
  (03 §3.1, §3.2, §4.1, CONFIRMED).
- `models.json/llama-3.1-405b.sharding = "FSDP: optimizer states + gradients; weights not resharded
  after forward"` (03 §4.1, CONFIRMED).
- `models.json/kimi-k3.optimizer = "Per-Head Muon"`, `.activation_tricks` (02 §1.5, 03 §3.2, CONFIRMED).
- Usable HBM per chip (settled: nominal and usable both stored): `b200.hbm_usable_gb = 180` (03 §1.4,
  §3.3); data holds no usable figure for H100, H200 or B300 (B300 and GB300 system pages imply 262–278 GB), so the page uses their nominal capacity and says "nominal".
- `models.json/gpt-3.*`: the keys `decoder-anatomy` proposed (shared, not re-proposed).
**Graph changes:** none.
**Judgment calls:** none open; all settled in §13.

## 13. Reviewer rulings (expert review, 2026-10-07)
Settled and applied (README lesson 20):
- **Module:** the training-memory functions live in `math/training-memory.js`, owned by this page;
  `math/memory.js` stays KV and inference only (`kv-cache`). `parallelism` imports `zeroPerGpuBytes`
  and `TRAINING_RECIPES` from here.
- **Capacity uses usable HBM, labeled:** B200 180 GB usable → 1T needs 89 GPUs (brief 03 §3.3's figure),
  V4-Pro's size at 16 B needs 143. Every chip label says "usable" or "nominal".
- **Kimi K2:** 6 B per parameter = weights + FP32 gradient buffer, about 6 TB over 256 GPUs (about 23 GB
  each), leaving about 30 GB per GPU for optimizer state.
- **DeepSeek-V4-Pro's 25.6 TB** is the 16-byte what-if, labeled; its own Muon recipe gives 240 H100s.
- **ZeRO and activations:** the page says "ZeRO's three stages shard state, not activations". ZeRO-R's
  activation partitioning (ZeRO paper §6) is not in the briefs and is not taught here.
- **Share-bar budget (lesson 19):** frames 8–10 draw only GPU 1's composition, full width (420 px), so
  the ZeRO-1 optimizer slice is 18.8 px; GPU 2 and GPU 64 are `gpu` glyphs with GB printed.
- **Brief's "20 B":** not used (2 + 4 + 4 + 4 + 4 = 18).
- **GPT-3 as the running example:** accepted, with the visible "today's recipe" line.
- **Weights in motion:** `carry: 'weight'` accepted for the Training build (frame 10).
- **Selective-recompute cost:** stated as "small, not quantified here", with a hover detail.
- **`--part-*`:** categorical hues, meanings per page (stated in §4).
- **Checkpoint id:** `llama-3.1-405b`.
- **Applied Shoulds:** H200 row in frame 11; hook names 0.35 TB of gradients and 2.1 TB of optimizer
  state; the `recipe` argument is a `TRAINING_RECIPES` entry; frame 10 names both passes (lesson 26);
  frame 1 prints what the bar and the GPUs each measure (lesson 21).
- Data pass 2026-10-07: chip memory labels now say nominal for H100, H200 and B300 (no usable figure in data); B200 stays 180 GB usable; no numbers changed.
