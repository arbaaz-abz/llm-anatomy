# Splitting a model across GPUs (`parallelism`)

Track: training · Section: gpus · Prereqs: training-memory, moe (matches `shared/concepts.json`)
Next: `cluster-topology`, `disaggregation` (the two slugs whose `prereqs` list `parallelism`)
Status: draft
Sources: 03 §2.3, §3.4, §4 (intro table, collective cost model), §4.1–§4.6, §7.1 (toys 3, 5, 6), §7.2 ·
05 §1.1, §1.2, §2

Running example: the course toy from `decoder-anatomy` and `attention`. Four tokens "The cat sat down"
(row "sat" followed, as everywhere), d_model 8, the toy's SwiGLU MLP (W_in and W_gate [8 × 16], W_out
[16 × 8], 384 parameters per block; 1,576 parameters in the whole toy model) and, for expert
parallelism, the `decoder-anatomy` MoE variant with 8 experts, top-2. The pipeline frames use a
stand-in 6-block model so three stages of two blocks each fit the stage budget. Real configurations
(Llama 3 405B, DeepSeek-V3) close the page; *where* each cut goes on the network is `cluster-topology`.

This page draws what each strategy splits and what it sends. It does not redo ZeRO (`training-memory`
owns it; frame 2 links to it) or the routing math of MoE (`moe` owns it; frame 10 uses hand-picked
routes).

## 1. Learning objective
After this page you can name the five ways to split a training step (data, tensor, pipeline, context,
expert; frame 1) and, for each, say what every GPU holds and what it must send: data parallelism
all-reduces gradients once per step (frame 2), tensor parallelism all-reduces partial sums inside every
layer (frames 3–4), pipeline parallelism passes activations between stages and pays a bubble you can
compute (frames 5–8, try-this 1–2), context parallelism passes keys and values along the sequence
(frame 9), expert parallelism sends tokens to their experts and back (frame 10); and multiply the
degrees of a real run into its GPU count (frame 11, try-this 3).

## 2. Misconceptions to correct
- **Misconception:** "Data parallelism splits the model." → **Reality:** it splits the *data*; every GPU
  runs the whole model and the GPUs agree by averaging gradients once per step. It is the only strategy
  here that splits nothing of the model (ZeRO then shards the state, `training-memory`). Corrected by
  frame 2. (03 §4 table, §4.1)
- **Misconception:** "Tensor parallelism is just more data parallelism inside a node." → **Reality:**
  it splits each weight matrix, so no GPU can finish a layer alone: an all-reduce of partial sums sits
  inside every layer, on the critical path, twice per layer in the forward pass. That is why it usually
  lives inside a fast NVLink domain, and why DeepSeek-V3 trained with no tensor parallelism at all.
  Corrected by frames 3–4. (03 §4.2)
- **Misconception:** "A pipeline of p stages runs p times faster." → **Reality:** stages wait for each
  other at the start and end of every step. With 3 stages and 4 micro-batches a third of the GPU time is
  idle; the bubble is (p − 1)/(m + p − 1) and only many micro-batches shrink it. Corrected by frames 6–7
  and try-this 1. (03 §4.4)
- **Misconception:** "1F1B removes the pipeline bubble." → **Reality:** with equal forward and backward
  times 1F1B has exactly GPipe's bubble; what it changes is memory: a stage holds at most p micro-batches
  of activations instead of m. Zero-bubble and DualPipe schedules attack the bubble itself (formulas in §7, §8; not demonstrated here). Corrected
  by frame 8 and try-this 2. (03 §4.4)
- **Misconception:** "Expert parallelism is free because each token only visits two experts." →
  **Reality:** each MoE layer sends every token to the GPUs that hold its experts and back (two
  all-to-alls), and a busy expert makes the others wait: in the toy one GPU gets 3 token-copies and
  another gets 1. Corrected by frame 10. (03 §4.5)

## 3. Hook and intuition (final wording)
**Hook:** A 405B model needs 6.5 TB of training state and an H100 holds 80 GB. When you spread it over
8,192 GPUs, what does each GPU actually hold, and what do they say to each other?

There are five ways to cut a training step, and they cut different things. Data parallelism cuts the
batch: every GPU runs the whole model on its own sequences and, once per step, they average gradients.
Tensor parallelism cuts each weight matrix: two GPUs each hold half of every matrix and must add up
their partial results inside every layer. Pipeline parallelism cuts the stack of blocks: each GPU runs
a few consecutive blocks and hands activations to the next, like stations on an assembly line, which
means stations sit idle while the line fills and drains. Context parallelism cuts one long sequence:
later tokens need the keys and values of earlier ones, so those travel. Expert parallelism puts
different experts on different GPUs, so tokens travel to their experts and back.

Each cut buys memory and costs communication of a particular shape: one big all-reduce per step,
small all-reduces inside every layer, point-to-point hand-offs, a ring of keys and values, or an
all-to-all shuffle. Real runs stack several cuts, and the GPU count is the product of their degrees:
Llama 3 405B ran 8-way tensor, 16-way pipeline and 64-way data parallelism, 8 × 16 × 64 = 8,192 GPUs.
That answers the hook: each of those GPUs holds one-eighth of every matrix in one-sixteenth of the
blocks, about 13 GB of state with ZeRO-1, and talks constantly to its 7 tensor partners, now and then
to its pipeline neighbors, and once per step to its 63 data-parallel peers. How chatty each cut is
decides where on the network it must live, which is the next lesson.

## 4. Visual metaphor
A "cut map" in frame 1: the toy's four token rows on the left, two blocks (attention, MLP) as `block`s,
and dashed cut lines along five axes, each labeled with its strategy. From frame 2 on, each strategy
gets its own GPU picture: two to four small `gpu` glyphs ("GPU 1" always on the left; the followed row
"sat" outlined wherever it appears), with what each GPU holds drawn on it and every transfer drawn as a
`flow` whose label prints the bytes. A plain counter "sent per GPU" sits at the bottom right in every
frame (`NUMBER_CELL` readout).

Glyphs used (from spec §5.1 and the built library): `token` (the four chips; "sat" outlined), `matrix`
(W_in, W_gate, W_out as shape grids, 20 px cells, no numbers; split halves tinted per GPU), `vector`
(`NUMBER_CELL` rows for the "sat" partial sums in frame 4: 8 cells × 43 px = 344 px, within the stage),
`block` (blocks, stages and experts; `active` = running now, `idle` = waiting), `gpu` (memory bar
unused here: `memFill` 0 is drawn but the page never reads it; see open question 3), `flow` (carry
`gradient` for the DP all-reduce, `activation` for TP partial sums, PP hand-offs and EP tokens, `kv` for
CP), `kvStack` (frame 9, K/V tiles of "The" and "cat"), `heatmap` (frame 9: the 4 × 4 causal mask split
between two GPUs, hatched cells masked).

From approved proposals: `laneTimeline` (proposed by `gpu-primer`) for the pipeline schedule (frames
6–8): one lane per stage, 40 px cells printing "F1", "B2" and so on, `forward` and `backward` fills,
idle hatched. With 3 stages × 4 micro-batches the schedule is 12 cells: 12 × 43 = 516 px plus a 52 px
lane-label gutter = 568 px (lesson 18). `blockStack` (proposed by `decoder-anatomy`) for the 6-block
stand-in model in frame 5.

New glyphs proposed: none. A `flow` carry value is proposed (open question 4): `carry: 'weight'` for
weights in motion (ZeRO-3 gathers here and in `training-memory`, W_O in `gpu-primer`).

Plain labeled marks: "partial sums are hand-picked stand-ins; they add up to `decoder-anatomy`'s MLP
output for sat" (frame 4); "stand-in 6-block model" (frame 5); "routes are hand-picked; how a router
picks them is `moe`" (frame 10); the "sent per GPU" counter; "bubble 33%" readouts.

Terms introduced (one per frame): degree, how many GPUs share one cut (frame 1, which shows the five
strategy names only as labels), all-reduce (2), tensor parallelism (3), partial sum (4), stage (5),
micro-batch (6), bubble (7), 1F1B (8), context parallelism (9), all-to-all (10); frame 11 introduces
none. Expert parallelism is named in frame 10's caption beside its term, all-to-all, because "expert"
is assumed from `moe`; data parallelism is assumed from `training-memory` (its frame 8). Terms assumed from prereqs: gradient,
optimizer state, activations, ZeRO, data parallel copy (`training-memory`); HBM, BF16 (`gpu-primer`);
expert, router, top-k (`moe`); keys, values, causal mask (`attention`, linked).

Indexing: GPUs, stages, micro-batches and experts count from 1 on screen; token positions from 1. No
memory addresses on this page.

## 5. Animation script
All byte counts in BF16 (2 bytes per number); collectives use the ring cost model (03 §4): an
all-reduce of S bytes over N GPUs sends 2(N − 1)/N · S bytes per GPU; an all-to-all sends (N − 1)/N.
Numbers from `math/parallel.js` (reproducer in §6).

| # | On screen | What moves | Caption (final wording) | Numbers shown |
|---|---|---|---|---|
| 1 | Cut map: the four token chips ("sat" outlined) entering two `block`s (attention, MLP), a second sequence's chips greyed below. Five dashed cut lines appear one by one, each with its label: between the two sequences (data), through each block's matrices (tensor), between the blocks (pipeline), between "cat" and "sat" (context), between experts inside the MLP (expert). | Each cut line draws in turn. | A training step can be cut by data, inside each matrix, between blocks, along the sequence or between experts. A cut's degree is how many GPUs share it. | 5 cuts |
| 2 | Two `gpu` glyphs, each with the full toy model ("1,576 parameters"). GPU 1 gets "The cat sat down", GPU 2 "another 4-token sequence" (greyed chips). After backward, `flow` arrows (carry `gradient`) run both ways; counter "sent per GPU: 3,152 B". Link chip: "ZeRO shards this state: `training-memory`". | Sequences drop in; a forward and backward pulse runs on each; then the gradient arrows run. | Data parallelism gives each GPU a full copy and different sequences. Once per step, an all-reduce averages their gradients so every copy takes the same update. | gradients 1,576 × 2 B = 3,152 B · all-reduce over 2 GPUs: 2 × 1/2 × 3,152 = 3,152 B per GPU · GPT-3 shape on 64 GPUs: 689 GB per GPU per step |
| 3 | Two GPUs. W_in and W_gate [8 × 16] each cut down the middle: columns 1–8 on GPU 1, 9–16 on GPU 2 (two tints). The four token rows go to both GPUs. Each GPU's hidden output [4 × 8]. | The cut slides through the matrices; the halves slide apart; each GPU's hidden half fills. Counter stays 0 B. | Tensor parallelism cuts each weight matrix between GPUs. Cut the MLP's first matrices by columns and each GPU computes half of the hidden numbers, without talking. | each GPU: W_in, W_gate [8 × 8] · hidden [4 × 8] · 0 B sent |
| 4 | W_out [16 × 8] cut by rows: rows 1–8 on GPU 1, 9–16 on GPU 2. Each GPU produces a full-width [4 × 8] partial output. The "sat" row of each is shown as a `NUMBER_CELL` `vector`; an all-reduce `flow` (carry `activation`) adds them into `decoder-anatomy`'s MLP output for sat. Visible stand-in label. | Partial rows appear; arrows cross; cells add column by column. Counter 0 → 64 B. | Cut the second matrix by rows and each GPU holds a partial sum of the output. An all-reduce adds them inside every layer before anything downstream can start. | GPU 1 sat: [0.25, 0.5, −0.25, −0.25, 0.25, 0.25, 0, 0] · GPU 2: [−0.25, −0.25, 0.25, −0.25, 0.25, −0.25, −0.25, 0.25] · sum [0, 0.25, 0, −0.5, 0.5, 0, −0.25, 0.25] · [4 × 8] × 2 B = 64 B; all-reduce over 2: 64 B per GPU · per GPU 192 of 384 MLP parameters · 2 all-reduces per block in the forward pass |
| 5 | `blockStack` of a stand-in 6-block model, cut into three stages of two blocks, one per `gpu` (GPU 1–3). One micro-batch's activation travels stage 1 → 2 → 3 (`flow`, carry `activation`). | The cuts drop; the activation `flow` hops between GPUs; the active stage lights, the others `idle`. | Pipeline parallelism gives each GPU a run of consecutive blocks, called a stage. Only the activations crossing a stage boundary travel, from one GPU to the next. | per hand-off: 4 tokens × 8 × 2 B = 64 B · 3 stages × 2 blocks |
| 6 | `laneTimeline`, three lanes (stage 1–3), forward half only: cells F1–F4 cascade diagonally (columns 1–6). | Cells appear column by column. | Split the batch into micro-batches so stages can work at the same time. Stage 2 starts micro-batch 1 while stage 1 moves on to micro-batch 2. | 4 micro-batches · forward done at column 6 |
| 7 | The full GPipe schedule, 12 columns: forwards then backwards; idle cells hatched; readout "bubble 33%". | Backward cells B1–B4 cascade up; idle cells hatch; the readout counts. | Stages still wait while the pipeline fills and drains. That idle time is the bubble: with 3 stages and 4 micro-batches, a third of every GPU's time. | 12 columns × 3 stages = 36 cells, 12 idle · (3 − 1) / (4 + 3 − 1) = 33.3% · stage 1 holds 4 micro-batches' activations at its peak |
| 8 | The 1F1B schedule, same 12 columns: F1 F2 F3 · · B1 F4 B2 · B3 · B4 on stage 1. Readout "bubble 33%" (unchanged) and "stage 1 holds at most 3". | Cells reorder from the GPipe layout into 1F1B; the in-flight counter on stage 1 peaks at 3 instead of 4. | The 1F1B schedule starts each backward as soon as it can. The bubble stays the same, but stage 1 never holds more than 3 micro-batches of activations. | bubble 33.3% · peak in flight per stage: GPipe 4, 4, 4 → 1F1B 3, 2, 1 · with 8 micro-batches: 8 vs 3, bubble 20% |
| 9 | Two GPUs. GPU 1 holds "The cat", GPU 2 "sat down" ("sat" outlined). A 4 × 4 causal `heatmap` split by GPU (rows = queries). `kvStack` tiles of The and cat travel from GPU 1 to GPU 2 (`flow`, carry `kv`); nothing travels back. | The K/V tiles slide across; GPU 2's two rows light; the masked upper cells stay hatched. | Context parallelism splits one long sequence between GPUs. Later tokens need the keys and values of earlier ones, so those travel; because of the mask, nothing flows back. | K and V per token per layer: 16 numbers × 2 B = 32 B · 2 tokens → 64 B per layer · score cells: GPU 1 3, GPU 2 7 |
| 10 | Four GPUs, each holding two experts (GPU 1: e1, e2 · GPU 2: e3, e4 · GPU 3: e5, e6 · GPU 4: e7, e8) and one token (The, cat, sat, down). Hand-picked top-2 routes: The → e1, e6 · cat → e2, e7 · sat → e4, e8 · down → e3, e8. Visible label "routes are hand-picked; how a router picks them is `moe`". | Token copies fly to their experts' GPUs (dispatch), the experts light, results fly back (combine). | Expert parallelism puts different experts on different GPUs. Each token is sent to its experts and back, an all-to-all twice per MoE layer, and the busiest GPU sets the pace. | 8 copies, 6 cross GPUs · 6 × 16 B = 96 B dispatch + 96 B combine · copies per GPU: 2, 2, 1, 3 |
| 11 | A plain product line and two rows: "Llama 3 405B (2024): tensor 8 × context 1 × pipeline 16 × data 64 = 8,192 H100s" and "DeepSeek-V3 (2024): pipeline 16 × expert 64 with ZeRO-1 data parallel, no tensor parallelism, 2,048 H800s". Readout under the first row: "state per GPU: 13.25 GB (ZeRO-1)". | The factors type in and multiply. | Real runs stack these cuts, and the GPU count is the product of their degrees. Llama 3 405B used 8-way tensor, 16-way pipeline and 64-way data parallelism: 8,192 GPUs. | 8 × 1 × 16 × 64 = 8,192 · 405e9 × 16 B ÷ (8 × 16) = 50.6 GB before ZeRO → 13.25 GB with ZeRO-1 over 64 |

Determinism: every frame is a pure function of (step, progress). Reduced motion shows each frame's end
state. Frames 6–8 draw `pipelineSchedule({ schedule, stages: 3, microBatches: 4 })`; frame 2 and 4 bytes
are `ringAllReduceBytes`; frame 10 is `allToAllBytes` on the fixed routes; frame 11 is `gpuCount` and
`zeroPerGpuBytes({ params: 405e9, recipe: adam, stage: 1, dp: 64, modelShards: 128 })`.

Caption check: counts in §12 (all ≤ 2 sentences, ≤ 30 words, no operators).

## 6. Toy
Two linked panels. A visible line above them: "Forward and backward each take one time unit here; real
backward passes take about twice as long, which changes the bubble's size a little but not its shape."

**Panel A: schedule a pipeline**
| id | Label | Type | Range / values | Default | Presets |
|---|---|---|---|---|---|
| `schedule` | Schedule | Segmented | GPipe · 1F1B | GPipe | – |
| `stages` | Pipeline stages p | Segmented | 2 · 3 · 4 · 8 | 4 | – |
| `micro` | Micro-batches m | Slider | 1 … 32 | 4 | chips `4`, `8`, `16`, `32` |

| Output | Formula / `math/` function | Units / format |
|---|---|---|
| Schedule grid | `pipelineSchedule({ schedule, stages, microBatches })` → `grid` | `laneTimeline`; cells print "F3" / "B3" at 40 px when the grid has ≤ 12 columns; wider grids (up to 2 × (32 + 8 − 1) = 78 columns) draw label-free cells sized to the 520 px track, and the micro-batch ids are not shown anywhere (no hover-only numbers; lesson 18). The readouts below carry every number. |
| **Bubble** | `pipelineBubble({ stages, microBatches })`; equals `grid` idle fraction (tested) | % (1 decimal) |
| Peak micro-batches in flight, stage 1 | `pipelineSchedule(...).peakInFlight[0]` | count; "activation memory ∝ this" |
| Step length | `pipelineSchedule(...).columns` | time units |

**Panel B: count the GPUs**
| id | Label | Type | Range / values | Default | Presets |
|---|---|---|---|---|---|
| `tp`, `cp`, `pp`, `dp` | Tensor · context · pipeline · data degree | Sliders (powers of 2) | tp 1–16 · cp 1–16 · pp 1–32 · dp 1–1,024 | 8 · 1 · 16 · 64 | Llama 3 405B: "8K GPUs" (8, 1, 16, 64), "16K GPUs" (8, 1, 16, 128), "long context" (8, 16, 16, 8) |
| `stage` | ZeRO stage on the data-parallel group | Segmented | 0 · 1 · 2 · 3 | 1 | – |

| Output | Formula / `math/` function | Units / format |
|---|---|---|
| GPU count | `gpuCount({ tp, cp, pp, dp })` | integer |
| State per GPU for a 405B model (Adam, 16 B) | `zeroPerGpuBytes({ params: 405e9, recipe: TRAINING_RECIPES.adam, stage, dp, modelShards: tp * pp })` (imported from `math/memory.js`; same definition as `training-memory`) | GB, 2 decimals; activations not modeled (visible note: "405B's layer shape is not in `data/`") |

**Try this** (each leads to a named insight)
1. GPipe, 4 stages: set micro-batches 4 → 8 → 16 → 32 and read the bubble: 42.9% → 27.3% → 15.8% →
   8.6%. Then 8 stages with 32 micro-batches: 17.9%. → **Insight: the bubble is (p − 1)/(m + p − 1),
   so a deep pipeline needs many more micro-batches than stages.**
2. 4 stages, 16 micro-batches: GPipe holds 16 micro-batches of activations at stage 1; switch to 1F1B:
   4, while the bubble stays 15.8%. → **Insight: 1F1B doesn't shrink the bubble; it caps activation
   memory at p micro-batches, which is what makes large m affordable.** Shrinking the bubble itself
   takes interleaved, zero-bubble or DualPipe schedules (§7, §8).
3. Load the Llama 3 "8K GPUs" preset: 8,192 GPUs, 13.25 GB of state each with ZeRO-1. Set all four
   degrees to 1: one GPU, 6,480 GB. Now set tp = pp = 1 and dp = 8,192 with ZeRO-3: also 0.79 GB, the
   same as tp 8 × pp 16 × dp 64 with ZeRO-3. → **Insight: for state alone, ZeRO-3 over all GPUs would
   do; tensor and pipeline splits are there for what ZeRO cannot touch: activations, and the traffic of
   gathering every weight from thousands of GPUs every layer** (03 §4.6 selection heuristic; 03 §4.1
   compute-bound condition).

**`math/parallel.js`** (new module, owned by this page; `cluster-topology` and `disaggregation` import
the collective costs; pure, no DOM, inputs never mutated; tests first). Bytes are bytes sent per GPU.

```js
ringAllReduceBytes(sizeBytes, ranks) → number   // 2(N − 1)/N · S   (reduce-scatter + all-gather)
//   (3152, 2) → 3152 · (64, 2) → 64 · (350e9, 64) → 689.06e9 · (3152, 64) → 6205.5
reduceScatterBytes(sizeBytes, ranks) → number   // (N − 1)/N · S
allGatherBytes(sizeBytes, ranks) → number       // (N − 1)/N · S
//   (350e9, 64) → 344.53e9
allToAllBytes(sizeBytes, ranks) → number        // (N − 1)/N · S: each GPU keeps 1/N of what it holds
//   (64, 4) → 48   (uniform traffic; frame 10 counts its fixed routes directly: 6 crossing copies × 16 B)
gpuCount({ tp = 1, cp = 1, pp = 1, dp = 1 }) → number
//   (8, 1, 16, 64) → 8192 · (8, 1, 16, 128) → 16384 · (8, 16, 16, 8) → 16384
pipelineBubble({ stages, microBatches, virtualStages = 1 }) → number
//   b = (p − 1)/v; returns b / (m + b)   ((p − 1)/(m + p − 1) for v = 1; interleaving divides the bubble by v)
//   (3, 4) → 0.3333 · (4, 4) → 0.4286 · (4, 8) → 0.2727 · (4, 16) → 0.1579 · (4, 32) → 0.0857 · (8, 32) → 0.1795
//   (4, 8, v 2) → 0.1579
pipelineSchedule({ schedule: 'gpipe' | '1f1b', stages, microBatches }) → { columns, grid, idleFraction, peakInFlight }
//   unit-time forward and backward; stage s runs its next op when its input is ready
//   (F needs F of stage s − 1 one step earlier; B needs B of stage s + 1, or F of the last stage);
//   1F1B order per stage: warm-up min(p − s, m) forwards (s from 1), then F/B alternating, then the remaining B.
//   grid[s] is an array of 'F1' … 'B4' or '.' (idle). columns = 2(m + p − 1).
//   gpipe (3, 4):  F1 F2 F3 F4 .  .  .  .  B1 B2 B3 B4 / . F1 F2 F3 F4 . . B1 B2 B3 B4 . / . . F1 F2 F3 F4 B1 B2 B3 B4 . .
//                  idleFraction 0.3333, peakInFlight [4, 4, 4]
//   1f1b (3, 4):   F1 F2 F3 .  .  B1 F4 B2 .  B3 .  B4 / . F1 F2 . B1 F3 B2 F4 B3 . B4 . / . . F1 B1 F2 B2 F3 B3 F4 B4 . .
//                  idleFraction 0.3333, peakInFlight [3, 2, 1]
//   (3, 8): gpipe peak [8, 8, 8], 1f1b [3, 2, 1], idle 0.2 · (4, 16): gpipe [16 ×4], 1f1b [4, 3, 2, 1], 38 columns
```
`zeroPerGpuBytes` and `TRAINING_RECIPES` are imported from `math/memory.js` (`training-memory`); this
page passes `modelShards = tp · pp`. Context parallelism does not divide the state (it replicates the
weights, like data parallelism), so it is not in `modelShards`.

Reproducer (run 2026-10-07 against the scratch implementation; re-run once the modules exist):
```sh
node -e '
Promise.all([import("./math/memory.js"), import("./math/parallel.js")]).then(([mm, p]) => {
  const G = 1e9, f = (x) => Number(x.toFixed(2)), A = mm.TRAINING_RECIPES.adam;
  for (const sch of ["gpipe", "1f1b"]) for (const [s, m] of [[3, 4], [4, 4], [3, 8], [4, 8], [4, 16], [8, 16]]) {
    const r = p.pipelineSchedule({ schedule: sch, stages: s, microBatches: m });
    console.log(sch, s, m, r.columns, f(100 * r.idleFraction), f(100 * p.pipelineBubble({ stages: s, microBatches: m })), r.peakInFlight.join(",")); }
  for (const sch of ["gpipe", "1f1b"]) p.pipelineSchedule({ schedule: sch, stages: 3, microBatches: 4 }).grid.forEach((row, i) => console.log(sch, i + 1, row.join(" ")));
  console.log(p.ringAllReduceBytes(3152, 2), p.ringAllReduceBytes(64, 2), p.ringAllReduceBytes(350e9, 64) / G, p.allGatherBytes(350e9, 64) / G, p.allToAllBytes(64, 4));
  console.log(p.gpuCount({ tp: 8, cp: 1, pp: 16, dp: 64 }), p.gpuCount({ tp: 8, pp: 16, dp: 128 }), p.gpuCount({ tp: 8, cp: 16, pp: 16, dp: 8 }));
  for (const [tp, pp, dp, st] of [[8, 16, 64, 0], [8, 16, 64, 1], [8, 16, 64, 3], [1, 1, 8192, 3], [1, 1, 1, 0]]) console.log(tp, pp, dp, st, f(mm.zeroPerGpuBytes({ params: 405e9, recipe: A, stage: st, dp, modelShards: tp * pp }).total / G));
  for (const [s, m] of [[4, 32], [8, 32]]) console.log(s, m, f(100 * p.pipelineBubble({ stages: s, microBatches: m })));
});'
```
Output (2026-10-07): gpipe 3/4 → 12 columns, 33.33% idle, formula 33.33%, peak 4,4,4 · 4/4 → 14,
42.86 · 3/8 → 20, 20 · 4/8 → 22, 27.27 · 4/16 → 38, 15.79, peak 16 · 8/16 → 46, 30.43. 1F1B: same
columns and bubbles; peaks 3,2,1 (p = 3) and 4,3,2,1 (p = 4) for every m ≥ p. Grids as in the
signature block. Collectives: 3,152 · 64 · 689.06 GB · 344.53 GB · 48. GPU counts 8,192 · 16,384 ·
16,384. 405B state per GPU: (8, 16, 64) ZeRO-0 50.63 GB, ZeRO-1 13.25, ZeRO-3 0.79; (1, 1, 8,192) ZeRO-3
0.79; (1, 1, 1) 6,480 GB. Bubbles (4, 32) 8.57%, (8, 32) 17.95%. Partial sums in frame 4: GPU 2 = sat's
MLP output − GPU 1's stand-in, element-wise (checked to the digit).

Tests to write first: grid idle fraction equals `pipelineBubble` for every p ∈ {2, 3, 4, 8}, m ∈ 1..32,
both schedules; every micro-batch appears exactly once as F and once as B on every stage; 1F1B
`peakInFlight[0] === min(p, m)`, GPipe's `=== m`; collectives are 0 at N = 1; `gpuCount` defaults to 1;
inputs are not mutated.

## 7. Show me the math
```tex
\text{all-reduce: } \htmlClass{hl-comm}{2\,\tfrac{N-1}{N}\,S}
\qquad
\text{all-gather, reduce-scatter, all-to-all: } \htmlClass{hl-comm}{\tfrac{N-1}{N}\,S}
\qquad\text{bytes sent per GPU}
```
```tex
\text{TP (Megatron MLP): } Y = \text{act}(X A),\; A = [A_1 \mid A_2]
\quad\Rightarrow\quad Z = Y B = Y_1 B_1 + Y_2 B_2 \ \ (\text{one all-reduce})
```
```tex
\htmlClass{hl-bubble}{\text{bubble}} = \frac{p-1}{m+p-1}\ (\text{GPipe, 1F1B})
\qquad \frac{(p-1)/v}{m + (p-1)/v}\ (\text{interleaved, } v \text{ chunks per GPU})
```
```tex
\text{bubble time (DeepSeek-V3 Table 2): }\;
\text{1F1B } (p-1)(F+B) \quad \text{ZB1P } (p-1)(F+B-2W) \quad
\text{DualPipe } \big(\tfrac{p}{2}-1\big)(F\&B + B - 3W)
```
```tex
\#\text{GPUs} = \text{TP}\times\text{CP}\times\text{PP}\times\text{DP}
\qquad \text{(EP usually reuses the data- or tensor-parallel GPUs)}
```
Shapes: X [tokens × d], A [d × h] split into A₁, A₂ [d × h/2]; B [h × d] split into B₁, B₂ [h/2 × d]
(toy: d = 8, h = 16). p stages, m micro-batches, v virtual stages; F, B, W = time of a forward chunk, a
backward-for-inputs chunk, a backward-for-weights chunk; F&B = an overlapped forward-and-backward pair.
Color links: `hl-comm` = the "sent per GPU" counter and the `flow` labels; `hl-bubble` = the hatched
idle cells in `laneTimeline`.

## 8. In today's models (Oct 2026)
| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| Llama 3 405B (2024 paper): TP 8 / CP 1 / PP 16 / DP 64 on 8,192 H100s; TP 8 / PP 16 / DP 128 on 16,384; long-context stage TP 8 / CP 16 / PP 16 / DP 8 on 16,384 | `models.json/llama-3-405b.parallelism` *(proposed: the three configs as strings)*; `.training_gpus` (existing, 16,384) | 03 §4.6 (Table 4) |
| Llama 3 places parallelism innermost to outermost as TP, CP, PP, DP: "innermost parallelism requires the highest network bandwidth" (details in `cluster-topology`) | `models.json/llama-3-405b.parallelism_order` = "TP, CP, PP, DP" *(proposed)* | 03 §2.3 |
| Llama 3 used all-gather context parallelism only in the final 128K-token stage (CP 16), since attention FLOPs dwarf the all-gather and GQA keeps K and V small | `models.json/llama-3-405b.cp_long_context` = 16 *(proposed)* | 03 §4.3 |
| DeepSeek-V3 (2024): 2,048 H800s, PP 16, EP 64 across 8 nodes, ZeRO-1 data parallel, **no tensor parallelism**; DualPipe schedule; communication kernels use only 20 of 132 SMs | `models.json/deepseek-v3.training_gpus` = 2048, `.parallelism` = "PP16 / EP64 / ZeRO-1 DP, no TP", `.pipeline_schedule` = "DualPipe" *(proposed; entry proposed by `paged-attention`)* | 03 §4.2, §4.5, §4.6 |
| DualPipe feeds micro-batches from both ends of the pipeline and overlaps all-to-all and pipeline traffic with compute; it keeps two copies of the parameters | (mechanism; DeepSeek-V3 Table 2, code github.com/deepseek-ai/DualPipe) | 03 §4.4 |
| Kimi K2 (2025): PP 16 with virtual stages, EP 16, ZeRO-1 data parallel, trainable on any multiple of 32 nodes (minimum 256 GPUs) | `models.json/kimi-k2.parallelism` *(proposed; entry proposed by `training-memory`)* | 03 §4.6 |
| Kimi K3 (2026): pipeline with virtual stages, MoonEP expert parallelism (dynamic redundant experts for perfect load balance), ZeRO-1 plus "Pipeline ZeRO-2", context parallelism that passes the linear-attention state | `models.json/kimi-k3.parallelism` *(proposed)* | 03 §4.1, §4.3, §4.5, §4.6 |
| DeepSeek-V4 (2026) fuses dispatch, expert GEMMs and combine into one mega-kernel so expert traffic hides behind compute (the condition is in `cluster-topology`) | `models.json/deepseek-v4-pro.ep_kernel` = "MegaMoE mega-kernel" *(proposed)* | 03 §4.5 |

Not shown: GPU counts for Kimi K3 and DeepSeek-V4 (not disclosed), "5D parallelism" as a product claim.

## 9. Takeaways
1. Five cuts, five kinds of traffic: data (one gradient all-reduce per step), tensor (an all-reduce of
   partial sums inside every layer), pipeline (activations handed between stages), context (keys and
   values along the sequence), expert (an all-to-all to the experts and back, where the busiest GPU sets
   the pace).
2. Pipelines idle while they fill and drain: the bubble is (p − 1)/(m + p − 1). 1F1B caps activation
   memory at p micro-batches without changing the bubble; interleaving, zero-bubble and DualPipe shrink
   the bubble itself (formulas in §7; not demonstrated on this page).
3. Real runs multiply cuts: Llama 3 405B's 8 × 16 × 64 = 8,192 H100s hold about 13 GB of state each;
   DeepSeek-V3 skipped tensor parallelism entirely. Where each cut runs on the network is the next page.

## 10. Next and go deeper
Next: `cluster-topology` (which link each cut needs, and why) · `disaggregation` (the same expert
parallelism, used for serving).
Go deeper (brief 03 §7.2, 05 §1.2): Hugging Face, *Ultra-Scale Playbook* (5D parallelism, interactive;
https://huggingface.co/spaces/nanotron/ultrascale-playbook) · Lilian Weng, *How to Train Really Large
Models on Many GPUs* (https://lilianweng.github.io/posts/2021-09-25-train-large/) · DeepSeek, *DualPipe*
(https://github.com/deepseek-ai/DualPipe).

## 11. Key-frame sketch
Frame 8 end state (1F1B, 3 stages, 4 micro-batches), desktop width. Rows printed from
`pipelineSchedule({ schedule: '1f1b', stages: 3, microBatches: 4 }).grid`; `..` = idle (hatched).
```text
Splitting a model across GPUs     8 / 11  [<] [Play] [>]
          1  2  3  4  5  6  7  8  9 10 11 12   time
stage 1  F1 F2 F3 .. .. B1 F4 B2 .. B3 .. B4   peak 3
stage 2  .. F1 F2 .. B1 F3 B2 F4 B3 .. B4 ..   peak 2
stage 3  .. .. F1 B1 F2 B2 F3 B3 F4 B4 .. ..   peak 1
bubble 12 of 36 cells = 33.3%   (GPipe: 33.3%, peak 4)
with 8 micro-batches: bubble 20%, peak 3 (GPipe 8)
 The 1F1B schedule starts each backward as soon as it
 can. The bubble stays the same, but stage 1 never ...
```

## 12. Open questions for the reviewer
Caption check (2026-10-07, the `rlvr-grpo` counter adapted): frames 1–11 are 28/2, 26/2, 26/2, 28/2,
27/2, 26/2, 27/2, 27/2, 28/2, 30/2, 29/2 (words/sentences); no operators.

**Data-pass keys**
- `models.json/llama-3-405b.parallelism = ["TP8/CP1/PP16/DP64 on 8,192", "TP8/CP1/PP16/DP128 on 16,384",
  "TP8/CP16/PP16/DP8 on 16,384 (long context)"]`, `.parallelism_order = "TP, CP, PP, DP"`,
  `.cp_long_context = 16` (03 §2.3, §4.3, §4.6; CONFIRMED arXiv 2407.21783 Table 4).
- `models.json/deepseek-v3`: `training_gpus = 2048` (H800), `parallelism = "PP16 / EP64 (8 nodes) /
  ZeRO-1 DP, no TP"`, `pipeline_schedule = "DualPipe"`, `comm_sms = "20 of 132"` (03 §4.5–§4.6,
  CONFIRMED).
- `models.json/kimi-k2.parallelism = "PP16 (virtual stages) / EP16 / ZeRO-1 DP; multiples of 32 nodes"`
  (03 §4.6, CONFIRMED).
- `models.json/kimi-k3.parallelism = "PP+VP / MoonEP / ZeRO-1 + Pipeline ZeRO-2 / KDA CP"` (03 §4.6,
  CONFIRMED).
- `models.json/deepseek-v4-pro.ep_kernel = "MegaMoE fused dispatch-GEMM-combine"` (03 §4.5, CONFIRMED).
**Graph changes:** none.
**Judgment calls**
1. **Pipeline example size.** The animation uses 3 stages × 4 micro-batches (12 numbered columns, 568 px
   with the gutter) because 4 × 4 needs 14 columns (602 px, over budget, lesson 18). The 1F1B memory
   saving is small at this size (3 vs 4); frame 8's Numbers line and try-this 2 carry the m = 8 and
   m = 16 cases (3 vs 8, 4 vs 16).
2. **Unit-time F and B.** Forward and backward each take one unit, so 1F1B's bubble equals GPipe's
   exactly; real backward ≈ 2 × forward. The visible line above the toy says so. DualPipe and zero-bubble
   appear only as Table 2 formulas in §7 and a row in §8, not in the toy, so the objective does not
   promise them.
3. **`gpu` glyph memory bar.** The glyph always draws its memory bar; this page has no fullness to show.
   Draw `memFill: 0` (an empty bar may read as "empty GPU") or add a `showMem: false` option to the
   glyph. Recommend the option.
4. **`flow` carry for weights.** Propose `carry: 'weight'` (a fourth carry color, e.g. `--part-1` from
   `decoder-anatomy`) for ZeRO-3 all-gathers (`training-memory` frame 10) and weight loads
   (`gpu-primer` frame 4). Until then those pages use labeled plain arrows.
5. **Training-track carry colors.** In `theme.css`, `--carry-activation` is `var(--accent)` and
   `--carry-gradient` is `var(--accent-train)`; on Training pages the accent *is* the training accent,
   so activation and gradient dots are the same color. This page needs them distinct (frame 2 vs frames
   4–5; `laneTimeline` forward vs backward fills). Suggest pinning `--carry-activation` to
   `--accent-arch` on every track.
6. **Frame 1 term.** "Degree" is the only new term; the five strategy names are shown as labels and
   defined in frames 2–10. If the reviewer counts five labels as five terms, frame 1's labels become
   generic ("data", "matrix", "blocks", "sequence", "experts").
7. **EP routes and load.** Frame 10's routes are hand-picked to show 6 of 8 copies crossing GPUs and an
   uneven load (2, 2, 1, 3); `moe` owns routing and balancing. Confirm with the `moe` author that its toy
   uses 8 experts, top-2 (as in `decoder-anatomy`'s MoE variant).
8. **DP traffic example.** Frame 2's "GPT-3 shape on 64 GPUs: 689 GB per GPU per step" is
   `ringAllReduceBytes(350e9, 64)` (BF16 gradients, no sharding). It is large on purpose: it motivates
   overlap (`scale-reliability`) and placement (`cluster-topology`).
