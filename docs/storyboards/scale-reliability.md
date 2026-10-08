# Training at 10,000 GPUs (`scale-reliability`)

Track: training · Section: gpus · Prereqs: cluster-topology, scaling-laws (matches `shared/concepts.json`)
Next: none (no slug lists `scale-reliability` as a prereq; the GPUs & scale section ends here) ·
Related: `scaling-laws` (how big N and D should be), `serving-calculator` (the same cost arithmetic for
inference)
Status: approved (expert review)
Sources: 03 §1.3, §5.1, §5.2, §5.3, §5.4, §5.5, §4.5, §4.6, §6, §7.1 (toys 8, 9), §7.2 · 02 §1.7 ·
05 §1.1, §1.2, §2. Expert review (Fable 5.1, 2026-10-07) applied; see §13.

Running example: Llama 3.1 405B (2024): 405B dense parameters, 15.6T tokens, up to 16,384 H100s, BF16,
38–43% MFU while training, 419 unexpected interruptions in 54 days; all CONFIRMED in the Llama 3 paper
("The Llama 3 Herd of Models") and the Llama 3.1 model card.
DeepSeek-V3 (2024) is the FP8 and cost example. Every number the learner sees comes from one chain
(§6, `runPlan`): FLOPs → useful GPU-hours at the MFU while training → failures and checkpoint loss →
GPU-hours, days, dollars.

Two utilization numbers, named apart everywhere on the page (README lesson 16): **MFU while training**
(the toy's slider; Llama's 38–43%) and **run-average MFU** (`mfuFrom`, FLOPs over all the GPU-hours
billed; Llama's 34.5%, DeepSeek-V3's 34.6%). They are related through the failure tax:
run-average = while-training × (1 − loss).

`scaling-laws` (Recipe track, a prereq) decides N and D and teaches the "6ND" intuition for compute
budgets; `math/scale.js` (this page) is the single exporter of `trainingFlops` and
`FLOPS_PER_PARAM_TOKEN` (settled), and `scaling-laws` and `cluster-topology` import them. This page
turns a budget into GPUs, days and money and does not re-teach scaling laws.

## 1. Learning objective
After this page you can turn a model size and token count into training FLOPs (6ND, frame 1), into
GPU-hours and days at a given MFU (frames 2–3, try-this 1), explain where the missing utilization goes
and how overlap and FP8 win some back (frames 4–7), estimate how often a 16,000-GPU run fails and what
checkpointing costs (frames 8–10, try-this 2–3), and price a run (frame 11).

## 2. Misconceptions to correct
- **Misconception:** "A cluster delivers its spec-sheet FLOPs." → **Reality:** Llama 3.1 405B's H100s
  delivered 38–43% of BF16 peak while training, and at least 34.5% averaged over the whole run (a floor:
  the model card's 30.84M GPU-hours also cover stages that 6ND over 15.6T tokens doesn't count). The 989 TFLOPS is a ceiling. Corrected by frames 2–3 and try-this 1.
  (03 §5.1, §5.5)
- **Misconception:** "Utilization numbers are comparable across runs." → **Reality:** they depend on
  the denominator. Llama 4 Behemoth's 390 TFLOPS per GPU in FP8 is 39% of an H100's BF16 peak or 20% of
  its FP8 peak; Meta gave no MFU. The page always names the peak it divides by. Corrected by frame 7.
  (03 §5.1)
- **Misconception:** "Failures are rare events you can ignore." → **Reality:** at 16,384 GPUs Llama 3
  saw an unexpected interruption every 3.1 hours, even though each GPU's share works out to one failure
  in about 6 years. The cluster's mean time between failures is a GPU's divided by the GPU count.
  Corrected by frame 8 and try-this 2. (03 §5.4)
- **Misconception:** "Checkpoint as often as possible to be safe." → **Reality:** every save costs time
  too. Too rarely loses work, too often loses time to saving; with the toy's stand-in save and restart
  times, saving every 5 minutes loses 13.0% of the run, every 60 minutes 18.6%, about every 14 minutes
  9.0%. Corrected by frame 9 and try-this 3. (03 §5.4, §7.1 toy 8; first principles)
- **Misconception:** "Adding GPUs always pays." → **Reality:** it shortens the run but raises the
  failure tax: the same Llama 3.1 run on 100,000 GPUs finishes in 15 days instead of 74, but 28% of its
  GPU-hours are lost to failures and restarts instead of 9%. Corrected by frame 10 and try-this 2.
  (03 §5.4; derived)

## 3. Hook and intuition (final wording)
**Hook:** Llama 3.1 405B needed 3.8 × 10²⁵ FLOPs, which 16,384 H100s at full speed would finish in 27
days. Why did it take three times as many GPU-hours as that?

Start with the arithmetic. Training costs about 6 FLOPs per parameter per token: 2 in the forward pass
and 4 in the backward pass. That rule ignores attention's own FLOPs, which matter at long context, but
for Llama 3.1 405B it gives 6 × 405 billion × 15.6 trillion = 3.8 × 10²⁵, the paper's own figure. At an
H100's 989 TFLOPS that is 10.6 million GPU-hours. Meta's model card reports 30.8 million. The card's
figure also covers the long-context and post-training stages, where 6ND undercounts, so the run-average
MFU it implies, 34.5%, is a floor.

The first gap is utilization. The share of peak a run actually delivers is its model FLOPs utilization,
MFU. Llama 3.1 ran at 38–43% MFU while training: the GPUs wait on memory-bound kernels (`gpu-primer`), on
communication that couldn't hide behind compute (`cluster-topology`) and on pipeline bubbles
(`parallelism`). Overlapping communication with compute and running matrix multiplies in FP8 win some
of it back. The second gap is reliability. With 16,384 GPUs, something fails every few hours; each
failure throws away the work since the last checkpoint plus the time to restart, and checkpoints
themselves cost time. Llama 3.1 kept over 90% of its time productive. Divide the 34.5% run-average by
that 0.9 and you are back inside the 38–43% band: utilization and failures together answer the hook.

The same arithmetic prices a run. GPU-hours times a price per hour is the bill: DeepSeek put V3's
2.79 million H800-hours at $2 each, $5.6 million, for the final run alone. The price of scale is the
failure tax: it grows with the GPU count, so on the biggest runs faster checkpointing and recovery can
be worth as much as faster chips (try-this 3).

## 4. Visual metaphor
One horizontal bar is the whole run, in GPU-hours, built up across the frames: ideal (at peak), then
the MFU gap, then the failure loss, each segment printed with its number. It is a `shareBar`
(`decoder-anatomy`'s proposal) with three named parts, "useful", "below peak", "lost to failures"; the
last is drawn hatched, the library's "excluded / doesn't count" hatch (README lesson 24), through a
`hatched: true` flag on a `shareBar` part (proposed below). Step
time is a two-lane `laneTimeline` (compute, communication) in frames 4–5; the failure frames use a
single-lane `laneTimeline` of the run (ruling P3-R9): `kind: 'compute'` segments, checkpoint saves as
`ticks: [{ t, label: 'save' }]`, the work since the last save as a `kind: 'lost'` segment (the compute
tint under the hatch: work that doesn't count, README lesson 24), and each restart as `gaps: [{ from, to, label: 'restart' }]`
after the failure mark. A failure is a plain labeled
mark ("✕ failure"), not a glyph. The followed run (Llama 3.1 405B) carries the selection outline on its bar in
every frame where several runs appear.

Glyphs used (from spec §5.1 and approved proposals): `rack` (frame 8: 16,384 GPUs collapsed to two
racks + "2,046 others", 8 GPUs each), `gpu` (frame 5: one GPU's SM grid with the S3 option `litSms: [10, 11]`
lighting 2 of its 12 drawn tiles, about 15%, for communication; the glyph's `<title>` says "2 of 12 tiles
lit" and the page's printed label carries the real count, "20 of 132 SMs"), `block` (frame 6: "GEMM in FP8", "accumulate in FP32" as labeled blocks), `matrix`
(frame 6: a 4 × 4 grid of tiles standing for 128 × 128 blocks, shape-only), `shareBar`
(`decoder-anatomy`), `laneTimeline` and `bitLayout` (both built in S3, from `gpu-primer`'s proposals).

New glyphs proposed: none. Glyph option built in S3: `shareBar` parts accept `hatched: true` (the
`heatmap` mask hatch, meaning excluded, never unknown; unknown parts keep the neutral fill, README lesson
19); reused by any page whose bar has a part that doesn't count. The lane's saves, lost work and restarts
use `laneTimeline`'s `ticks`, `lost` segments and `gaps`; frames 4–5's waiting compute lane is a hatched `idle` segment. Frame 9's lane shows a 90-minute window of the run
(about six saves), not all 54 days. Frame 10 prints "same scale: 36.97M GPU-h = full width" over its
two bars (lesson 21).

Plain labeled marks: "✕ failure" marks; "stand-in: 100 ms compute, 40 ms communication" (frames 4–5);
"save 30 s, restart 3 min: stand-ins chosen so Llama 3.1 lands near its reported >90%" (frames 9–10);
"measured against H100 peaks; Meta gave no MFU" (frame 7); "excludes research and ablations" (frame
11).

Terms introduced (one per frame): 6ND (frame 1), GPU-hour (2), MFU, named "while training" and
"run-average" on screen (3), exposed communication (4),
communication SMs (5), fine-grained scaling (6), none (7), MTBF, mean time between failures (8),
checkpoint interval (9), none (10), none (11). Terms assumed: TFLOPS, FP8, SM, memory-bound
(`gpu-primer`); overlap, all-reduce, all-to-all (`cluster-topology`, `parallelism`); bubble
(`parallelism`); N and D as parameters and tokens (`scaling-laws`).

Indexing: no positions or addresses on this page.

## 5. Animation script
Numbers from `math/scale.js` (reproducer in §6). H100 BF16 peak 989 TFLOPS, FP8 1,979 (reported).

| # | On screen | What moves | Caption (final wording) | Numbers shown |
|---|---|---|---|---|
| 1 | Plain equation card "6 × N × D" with Llama 3.1 405B's numbers dropping in; a small two-part `shareBar` "forward 2 · backward 4" (ruling P3-R19) and the line "why 6: each multiply-add is 2 FLOPs; backward does two matmuls for each forward one". | The factors drop into place; the product types in. | Training costs about 6 FLOPs per parameter per token: 2 in the forward pass, 4 in the backward. For Llama 3.1 405B that matches the paper's own total. | 6 × 405e9 × 15.6e12 = 3.79 × 10²⁵ FLOPs · ignores attention FLOPs, which grow at long context |
| 2 | The run bar appears with one segment, "at peak: 10.65M GPU-hours", and a plain labeled tick mark (page-drawn, 1 px ink) where the reported total ends: "model card: 30.84M" (no outline, no dashes: ruling P3-R11). | The segment grows; the tick drops in far to its right. | Divide by the H100's peak and the run needs 10.6 million GPU-hours, 27 days on 16,384 GPUs. Meta's model card reports 30.8 million GPU-hours. | 3.79e25 ÷ (989e12 × 3,600) = 10.65M GPU-h · ÷ 16,384 ÷ 24 = 27.1 days · reported 30.84M = 2.9× |
| 3 | The bar splits: "useful (MFU)" and "below peak". A readout "MFU while training: 38–43% (paper) · run-average MFU: 34.5% or more (card; covers more stages than 6ND counts)". | The "below peak" segment grows until the bar reaches 26.6M at 40%. | Model FLOPs utilization, MFU, is the share of peak a run actually delivers. Llama 3.1 ran at 38 to 43% while training, so most of the gap is utilization. | run-average 3.79e25 ÷ (30.84M × 3,600 × 989e12) = 34.5% (a floor) · at 40% while training: 26.62M GPU-h, 67.7 days · 34.5% ÷ 0.90 = 38.4% |
| 4 | Two-lane `laneTimeline` of one step: compute 100 ms, then communication 40 ms, in sequence; the compute lane idles (`kind: 'idle'`, hatched) during the communication. Stand-in label. | The communication segment draws after the compute segment; the idle hatch appears. | Exposed communication is time GPUs spend waiting on other GPUs. A step that computes for 100 ms, then waits 40 ms, keeps its tensor cores busy 71% of the time. | stand-in step: 100 + 40 = 140 ms · 100 ÷ 140 = 71% |
| 5 | The same lanes, the communication segment sliding under the compute segment. A `gpu` glyph whose SM grid shows a few SMs lit in a different shade, labeled "20 of 132 SMs run communication (DeepSeek-V3)". | The comm segment slides left until it sits under compute; the step shortens to 100 ms. | Overlap hides it: send one layer's gradients while computing the next. DeepSeek-V3 set aside 20 of each GPU's 132 SMs to run communication alongside the math. | overlapped step: max(100, 40) = 100 ms · 20 of 132 SMs (15%) |
| 6 | `bitLayout` FP8 E4M3 (1/4/3) beside BF16 (1/8/7) (`format: FORMATS.fp8_e4m3` and `format: FORMATS.bf16`; field texts S, E4, M3 / S, E8, M7). A `matrix` of tiles: activations scaled per 1 × 128 tile, weights per 128 × 128 block; one tile holds an outlier and only that tile's scale grows. A `block` "accumulate in FP32" under the GEMM block. | The outlier tile's scale label changes; the other tiles keep theirs; partial sums flow into the FP32 block. | DeepSeek-V3 ran its big matrix multiplies in FP8, with one scale per 128 numbers so an outlier spoils only its own tile. Its loss stayed within 0.25% of BF16. | tiles: 1 × 128 (activations), 128 × 128 (weights) · E4M3 everywhere · partial sums promoted to FP32 · loss error < 0.25% · kept in BF16/FP32: embeddings, output head, gating, norms, attention |
| 7 | Two runs side by side, each with its achieved TFLOPS per GPU and two percentages: "Llama 4 Behemoth: 390 TFLOPS per GPU (FP8): 39% of BF16 peak · 20% of FP8 peak" and "DeepSeek-V3: 343 TFLOPS per GPU (FP8), run-average: 35% · 17%". Plain mark: "measured against H100 peaks; neither lab gave an MFU". | The two percentages for each run type in, one per denominator. | With FP8, utilization depends on which peak you divide by. Behemoth's 390 TFLOPS per GPU is 39% of an H100's BF16 peak, or 20% of its FP8 peak. | 390 ÷ 989 = 39.4% · 390 ÷ 1,979 = 19.7% · DeepSeek-V3: 6 × 37e9 × 14.8e12 ÷ (2.664M × 3,600) = 343 TFLOPS → 34.6% / 17.3% |
| 8 | Two `rack`s + "2,046 others" (16,384 GPUs). "✕ failure" marks appear on random GPUs along a 54-day timeline: 419 of them, drawn as ticks. Readout "one every 3.09 h". | Ticks accumulate along the timeline; the readout counts. | Llama 3.1's 16,384 GPUs hit 419 unexpected interruptions in 54 days, one every 3.1 hours. The cluster's mean time between failures, MTBF, is one GPU's divided by the GPU count. | 54 × 24 ÷ 419 = 3.09 h · × 16,384 = 50,677 h per GPU (5.8 years) · 78% hardware, GPU issues 58.7% |
| 9 | Single-lane `laneTimeline` of a 90-minute window of the run: compute segments separated by `save` ticks every 13.6 min; a "✕ failure" mark; the work since the last save turns hatched (`lost`), then a `restart` gap. Stand-in label. | The run plays; at the failure the lost segment hatches and the restart gap opens. | Here the run saves a checkpoint every few minutes. A failure loses the work since the last save plus the restart; saving too often loses time to the saves. | save 30 s, restart 3 min (stand-ins) · best interval 13.6 min · loss 9.0% (save 3.7% + lost work 3.7% + restarts 1.6%) · Llama 3.1 reported >90% effective |
| 10 | The run bar for 16,384 GPUs (outlined) beside one for 100,000 GPUs, labeled "same scale: 36.97M GPU-h = full width": "useful 26.62M" the same; "lost to failures" (hatched) 2.62M vs 10.35M GPU-h. | The second bar's lost segment grows; the days readouts type in. | With 100,000 of the same GPUs, failures come every 30 minutes and checkpointing loses 28% of the run. Faster saves and restarts now matter as much as faster chips. | MTBF 3.09 h → 30.4 min · best interval 13.6 → 5.51 min · loss 9.0% → 28.0% · 74.4 → 15.4 days · 29.24M → 36.97M GPU-h |
| 11 | A plain line: "DeepSeek-V3: 2.788M H800-hours × $2 = $5.576M (DeepSeek's own price assumption; excludes research and ablations)". | The product types in. | The bill is GPU-hours times the price of an hour. DeepSeek priced V3's 2.79 million H800-hours at $2 each: $5.6 million, for the final run alone. | 2.788e6 × 2 = $5.576M · pre-training 2.664M H800-h = 180K per trillion tokens · 54.2 days on 2,048 GPUs |

Determinism: every frame is a pure function of (step, progress). Reduced motion shows each frame's end
state. Failure marks in frame 8 are placed by `mulberry32(54)` (seeded) and are illustrative; their
count (419) is the paper's. Frames 2–3 and 9–10 draw `runPlan` (§6).

Caption check: counts in §12.

## 6. Toy
"Plan a training run." A visible line above the controls: "Save and restart times are stand-ins chosen
so the Llama 3.1 405B preset lands near its reported >90% effective time; the per-GPU MTBF is derived from
Llama 3.1's 419 interruptions and includes non-GPU causes."

**Controls**
| id | Label | Type | Range / values | Default | Presets |
|---|---|---|---|---|---|
| `preset` | Run | Preset chips | Llama 3.1 405B · DeepSeek-V3 | Llama 3.1 405B | Llama: N 405e9, D 15.6e12, 16,384 GPUs, H100 BF16, MFU while training 40% · DeepSeek-V3: N 37e9 active, D 14.8e12, 2,048 GPUs, H100 BF16 peak as the visible stand-in for the H800, MFU while training 35.6% (exactly 0.3464 ÷ (1 − 0.0280) = 0.35637, back-solved so `runPlan` returns the reported 2.664M pre-training GPU-hours; visible note: "DeepSeek ran its matmuls in FP8; against the FP8 peak the run-average is 17.3%") |
| `params` | Active parameters N | Slider (log) | 1B … 1T | preset | – |
| `tokens` | Tokens D | Slider (log) | 100B … 50T | preset | – |
| `gpus` | GPUs | Slider (log, snapped) | 256 … 200,000 | preset | – (outside the loss formula's range, the readouts say so; see `valid`) |
| `peak` | Chip and precision | Preset chips | H100 BF16 989 · H100 FP8 1,979 · B200 BF16 2,250 · B300 FP8 5,000 | preset | `hardware.json/*_dense_tflops` |
| `mfu` | MFU while training | Slider | 15–60%, step 0.1 (a preset may set an unsnapped value) | preset (40% / 35.6%) | – |
| `mtbf` | MTBF per GPU | Slider (log) | 10,000 … 200,000 h | 50,677 h | derived from Llama 3.1 405B |
| `save` | Checkpoint save time | Slider | 5 s … 10 min | 30 s | stand-in |
| `restart` | Restart time | Slider | 1 … 30 min | 3 min | stand-in |
| `interval` | Checkpoint interval | Slider + "best" chip | 1 … 240 min | best | – |
| `price` | Price per GPU-hour | Slider | $1 … $6 | $2 | DeepSeek's assumption, not a market price (visible note) |

**Live outputs**
| Output | Formula / `math/` function | Units / format |
|---|---|---|
| Training FLOPs | `trainingFlops({ params, tokens })` | scientific, 3 s.f. |
| Useful GPU-hours, at the MFU while training | `gpuHoursAt({ flops, peakTflops, mfu })` | M GPU-h, 2 decimals |
| Cluster MTBF | `clusterMtbfHours({ perGpuMtbfHours, gpus })` | `formatDuration` ("3.09 h", "30.4 min") |
| Best interval (when "best") | `bestInterval({ saveH, mtbfH })` | `formatDuration` ("13.6 min", "5.51 min") |
| **Lost to checkpointing and failures** | `checkpointLossParts({ intervalH, saveH, restartH, mtbfH })` → `{ save, lostWork, restart, total }` (`checkpointLoss` is its `.total`) | %, 1 decimal |
| **GPU-hours, days, cost** | `runPlan(...)` → `gpuHours`, `days`, `cost`, `valid` | M GPU-h · days through `formatDuration` ("74.4 days") · $M (1 decimal); when `valid` is false the readouts are replaced by "outside the formula's range: interval plus restart exceed half the cluster MTBF" |
| Run-average MFU | `mfuFrom({ flops, gpuHours: runPlan(...).gpuHours, peakTflops })` = while-training × (1 − loss) | %, 1 decimal |
| Run bar | useful / below peak / lost | `shareBar` |
| Check against the record | Llama preset: reported 30.84M GPU-h (the card covers more stages than 6ND counts); DeepSeek preset: "reproduces 2.664M by construction (MFU while training set from the record)": useful 2.589M + failure tax 2.8% = 2.664M | text |

**Check my work:** none (the "Check against the record" output is this page's check, ruling P3-R16).

**Try this** (each leads to a named insight)
1. Llama 3.1 preset at MFU 40%: 29.24M GPU-hours, 74.4 days, against the model card's 30.84M. Slide MFU
   to 38%: 30.78M. → **Insight: 6ND ÷ (peak × MFU), plus a failure tax, reproduces a real run's
   GPU-hours to within a few percent; MFU is the number that turns FLOPs into days.** (The failure
   stand-ins were tuned to Llama 3.1's reported >90%, so this is a consistency check, not a prediction.)
2. Keep everything and raise GPUs 16,384 → 32,768 → 100,000: days 74.4 → 39.2 → 15.4, but the loss
   grows 9.0% → 13.6% → 28.0% and GPU-hours 29.24M → 30.81M → 36.97M. → **Insight: more GPUs finish
   sooner but pay a bigger failure tax, because the cluster's MTBF falls as the GPU count rises.**
3. At 16,384 GPUs set the interval to 5 min (loss 13.0%), 60 min (18.6%), then "best" (13.6 min,
   9.0%). Halve the save time to 15 s: best becomes 9.63 min and the loss 6.8%; at 100,000 GPUs the same
   change takes 28.0% to 22.7%. → **Insight: the best checkpoint interval balances save cost against
   lost work, and faster checkpointing is worth the most on the biggest clusters.**

**`math/scale.js`** (new module, owned by this page; pure, no DOM; tests first). Hours and TFLOPS in;
one definition per metric (README lesson 16): run-average MFU is always `mfuFrom(...)` =
`flops / (gpuHours · 3600 · peak)` over all GPU-hours, with the peak of the precision named on screen; MFU
while training is the slider; `runPlan` relates them: run-average = while-training × (1 − loss). 6ND has
one exporter: this module's `FLOPS_PER_PARAM_TOKEN` and `trainingFlops` (settled; `scaling-laws` and
`cluster-topology` import them).

```js
FLOPS_PER_PARAM_TOKEN = 6                                  // 2 forward + 4 backward
trainingFlops({ params, tokens }) → number                // FLOPS_PER_PARAM_TOKEN · N · D
//   (405e9, 15.6e12) → 3.7908e25 · (37e9, 14.8e12) → 3.2856e24
gpuHoursAt({ flops, peakTflops, mfu }) → number           // flops / (peak · 1e12 · mfu) / 3600
//   (3.7908e25, 989, 1) → 10.647e6 · (…, 0.40) → 26.62e6 · (…, 0.38) → 28.02e6
mfuFrom({ flops, gpuHours, peakTflops }) → number
//   (3.7908e25, 30.84e6, 989) → 0.3452 · (3.2856e24, 2.664e6, 989) → 0.3464 · (…, 1979) → 0.1731
achievedTflopsPerGpu({ flops, gpuHours }) → number
//   (3.7908e25, 30.84e6) → 341.4 · (3.2856e24, 2.664e6) → 342.6
wallClockDays({ gpuHours, gpus }) → number                // gpuHours / gpus / 24
//   (10.647e6, 16384) → 27.08 · (30.84e6, 16384) → 78.43 · (2.664e6, 2048) → 54.20
clusterMtbfHours({ perGpuMtbfHours, gpus }) → number      // perGpu / gpus
//   (50677, 16384) → 3.093 · (50677, 100000) → 0.507
checkpointLoss({ intervalH, saveH, restartH, mtbfH }) → number
//   first-order fraction of time lost: save/T + (T/2 + restart)/MTBF (valid when T, restart ≪ MTBF)
//   (13.6/60, 30/3600, 3/60, 3.093) → 0.0896 · (5/60, …) → 0.130 · (1, …) → 0.186
bestInterval({ saveH, mtbfH }) → number                   // √(2 · save · MTBF), the minimum of the formula above
//   (30/3600, 3.093) → 0.2270 h (13.6 min) · (30/3600, 0.507) → 0.0919 h (5.5 min) · (15/3600, 3.093) → 0.1605 h (9.6 min)
trainingStepMs({ computeMs, commMs, overlap }) → number   // overlap ? max : sum (P3-R4: "step time" alone is Serving's)
checkpointLossParts({ intervalH, saveH, restartH, mtbfH }) → { save, lostWork, restart, total }
//   (0.2270, 30/3600, 3/60, 3.093) → { 0.0367, 0.0367, 0.0162, 0.0896 }   (frame 9's 3.7 + 3.7 + 1.6 = 9.0%)
//   (100, 40, false) → 140 · (100, 40, true) → 100
runCost({ gpuHours, dollarsPerGpuHour }) → number
//   (2.788e6, 2) → 5.576e6
runPlan({ params, tokens, gpus, peakTflops, mfu, perGpuMtbfHours, saveH, restartH, intervalH?, dollarsPerGpuHour })
  → { flops, usefulGpuHours, mtbfH, intervalH, loss, valid, gpuHours, days, cost }
//   mfu = MFU while training; gpuHours = usefulGpuHours / (1 − loss); days and cost from gpuHours;
//   valid = intervalH + restartH ≤ mtbfH / 2 (the first-order loss formula's range); loss ≥ 1 → Infinity
//   100,000 GPUs at the defaults: 5.5 + 3 = 8.5 min ≤ 15.2 min → valid · 200,000 GPUs, restart 30 min → invalid
//   DeepSeek preset (37e9, 14.8e12, 2048 GPUs, 989, mfu 0.35637) → loss 0.0280, 2.664e6 GPU-h, 54.2 days
//   Llama preset, MFU 0.40, best interval → loss 0.090, 29.24e6 GPU-h, 74.4 days, $58.5M at $2
//   … 100,000 GPUs → loss 0.280, 36.97e6, 15.4 days · 32,768 → 0.136, 30.81e6, 39.2 days
//   … MFU 0.38 → 30.78e6 · save 15 s → loss 0.068, 28.56e6
```

Reproducer (run 2026-10-07 against the scratch implementation; re-run once the module exists):
```sh
node -e '
import("./math/scale.js").then((m) => {
  const f = (x, d = 2) => Number(x.toFixed(d)), per = 54 * 24 / 419 * 16384;
  const L = m.trainingFlops({ params: 405e9, tokens: 15.6e12 }), D = m.trainingFlops({ params: 37e9, tokens: 14.8e12 });
  console.log(L, D, m.gpuHoursAt({ flops: L, peakTflops: 989, mfu: 1 }) / 1e6, m.mfuFrom({ flops: L, gpuHours: 30.84e6, peakTflops: 989 }),
    m.mfuFrom({ flops: D, gpuHours: 2.664e6, peakTflops: 989 }), m.mfuFrom({ flops: D, gpuHours: 2.664e6, peakTflops: 1979 }),
    m.achievedTflopsPerGpu({ flops: D, gpuHours: 2.664e6 }), m.wallClockDays({ gpuHours: 2.664e6, gpus: 2048 }), m.runCost({ gpuHours: 2.788e6, dollarsPerGpuHour: 2 }), per);
  const base = { params: 405e9, tokens: 15.6e12, gpus: 16384, peakTflops: 989, mfu: 0.40, perGpuMtbfHours: per, saveH: 30 / 3600, restartH: 3 / 60, dollarsPerGpuHour: 2 };
  for (const [k, o] of [["base", {}], ["mfu38", { mfu: 0.38 }], ["32k", { gpus: 32768 }], ["100k", { gpus: 100000 }], ["T5", { intervalH: 5 / 60 }], ["T60", { intervalH: 1 }],
    ["save15", { saveH: 15 / 3600 }], ["save15 100k", { saveH: 15 / 3600, gpus: 100000 }]]) { const r = m.runPlan({ ...base, ...o });
    console.log(k, f(r.mtbfH, 3), f(r.intervalH * 60, 1), f(100 * r.loss, 1), f(r.gpuHours / 1e6), f(r.days, 1), f(r.cost / 1e6, 1)); }
  console.log(m.trainingStepMs({ computeMs: 100, commMs: 40, overlap: false }), m.trainingStepMs({ computeMs: 100, commMs: 40, overlap: true }));
  const ds = m.runPlan({ ...base, params: 37e9, tokens: 14.8e12, gpus: 2048, mfu: 0.35637 }); console.log("dsv3", f(ds.loss, 4), f(ds.usefulGpuHours / 1e6, 3), f(ds.gpuHours / 1e6, 3), f(ds.days, 1), ds.valid);
  console.log(m.runPlan({ ...base, gpus: 100000 }).valid, m.runPlan({ ...base, gpus: 200000, restartH: 0.5 }).valid);
});'
```
Output (2026-10-07): FLOPs 3.7908e25 and 3.2856e24; at peak 10.647M GPU-h; Llama MFU from the card
0.3452; DeepSeek-V3 0.3464 (BF16 peak) / 0.1731 (FP8 peak), 342.6 TFLOPS per GPU, 54.2 days on 2,048,
$5.576M; per-GPU MTBF 50,677 h. runPlan (MTBF h / interval min / loss % / M GPU-h / days / $M): base
3.093 / 13.6 / 9.0 / 29.24 / 74.4 / 58.5 · MFU 38% 30.78 / 78.3 · 32K 1.547 / 9.6 / 13.6 / 30.81 / 39.2
· 100K 0.507 / 5.5 / 28.0 / 36.97 / 15.4 / 73.9 · T 5 min 13.0 / 30.58 · T 60 min 18.6 / 32.71 · save
15 s 9.6 / 6.8 / 28.56 · save 15 s at 100K 3.9 / 22.7 / 34.43. Step 140 / 100 ms. After the review: DeepSeek preset (MFU while training 0.35637) → loss 0.0280,
useful 2.589M, total 2.664M GPU-h, 54.2 days, valid; `valid` at 100K GPUs true, at 200K with a 30-minute
restart false; Llama base run-average MFU 0.364 = 0.40 × (1 − 0.0896). Frame 9's split of
9.0%: save 0.00833 ÷ 0.2270 = 3.7%, lost work 0.1135 ÷ 3.093 = 3.7%, restarts 0.05 ÷ 3.093 = 1.6%.

Tests to write first: the worked examples; `runPlan(...).valid` is false when `intervalH + restartH >
mtbfH / 2`; the DeepSeek preset returns 2.664e6 GPU-hours to 3 significant figures; run-average MFU =
while-training × (1 − loss); `checkpointLoss` is minimized at `bestInterval` (checked by scanning ±1%); `runPlan` with `perGpuMtbfHours: Infinity` gives loss 0 and `gpuHours === usefulGpuHours`;
`mfuFrom(gpuHoursAt(x, mfu)) === mfu`; Infinity when loss ≥ 1; inputs not mutated.

## 7. Show me the math
```tex
\htmlClass{hl-flops}{C} \approx 6\,N\,D
\qquad
\text{GPU-hours} = \frac{C}{\htmlClass{hl-peak}{P_{\text{peak}}}\cdot \htmlClass{hl-mfu}{\text{MFU}} \cdot 3600}
\qquad
\text{MFU} = \frac{C}{\text{GPU-hours}\cdot 3600 \cdot P_{\text{peak}}}
```
```tex
\text{MTBF}_{\text{cluster}} = \frac{\text{MTBF}_{\text{GPU}}}{n_{\text{GPU}}}
\qquad
\htmlClass{hl-loss}{\text{loss}} \approx \frac{t_{\text{save}}}{T} + \frac{T/2 + t_{\text{restart}}}{\text{MTBF}_{\text{cluster}}}
\qquad
T^{*} = \sqrt{2\, t_{\text{save}}\, \text{MTBF}_{\text{cluster}}}
```
```tex
\text{cost} = \frac{\text{GPU-hours}_{\text{useful}}}{1 - \text{loss}} \times \text{price per GPU-hour}
```
Symbols: N parameters (active parameters for a MoE), D tokens, T checkpoint interval. Notes shown in
the panel: (a) 6ND counts the matmuls of forward (2ND) and backward (4ND); attention's own FLOPs grow
with context length and are ignored, so long-context stages cost more than 6ND says. (b) The loss
formula is first-order: one failure per MTBF on average, losing half an interval of work plus a
restart; T* follows by setting its derivative to zero. It overstates the loss when restart time
approaches the cluster MTBF, which is why the toy flags interval plus restart above half the MTBF. (c) Recomputation (`training-memory`) adds FLOPs
that MFU does not count (hardware FLOPs utilization, HFU, does). Color links: `hl-flops` = the equation
card, `hl-peak` = the "at peak" segment, `hl-mfu` = the "below peak" segment, `hl-loss` = the hatched
"lost" segments.

## 8. In today's models (Oct 2026)
| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| Llama 3.1 405B (2024): 15.6T tokens, 3.8 × 10²⁵ FLOPs, up to 16,384 H100s | `models.json/llama-3.1-405b.pretrain_tokens`, `.training_gpus` (existing); `.training_flops` = 3.8e25 *(proposed)* | 03 §5.5, §6 |
| Llama 3.1 405B model card: 30.84M H100 GPU-hours | `models.json/llama-3.1-405b.training_gpu_hours` = 30.84e6 *(proposed)* | 03 §5.5 |
| Llama 3.1 405B BF16 MFU: 43% (TP8/PP16/DP64, 8,192 GPUs, 430 TFLOPS), 41% (16,384 GPUs, 400 TFLOPS), 38% (long-context stage, 380 TFLOPS) | `models.json/llama-3.1-405b.mfu_bf16` = [0.38, 0.43] *(proposed; note lists the three configs)* | 03 §4.6, §5.1 |
| Llama 3.1 54-day window: 466 interruptions, 47 planned, 419 unexpected; 78% hardware; GPU issues 58.7% of unexpected; >90% effective training time; only 3 needed manual intervention | `models.json/llama-3.1-405b.interruptions_54d` = 419, `.interruptions_planned` = 47, `.interruptions_hardware_share` = 0.78, `.interruptions_gpu_share` = 0.587, `.effective_time` = ">90%" (confirmed) | 03 §5.4 |
| Llama 3.1 405B run: "even a single straggler can slow down thousands of other GPUs"; 1–2% diurnal throughput swing from temperature; 6 silent data corruptions | `models.json/llama-3.1-405b.reliability_notes` *(proposed, string)* | 03 §5.4 |
| DeepSeek-V3 (2024): 2.788M H800-hours total, 2.664M for pre-training (180K per trillion tokens, 3.7 days per trillion on 2,048 GPUs), "$5.576M at $2 per GPU-hour", excluding research and ablations | `models.json/deepseek-v3.training_gpu_hours` = 2.788e6, `.pretrain_gpu_hours` = 2.664e6, `.training_cost_usd_reported` = 5.576e6 *(proposed)* | 03 §5.5, §6 |
| DeepSeek-V3 FP8 recipe: all three GEMMs in FP8 (E4M3), 1 × 128 activation tiles, 128 × 128 weight blocks, FP32 promotion of partial sums, BF16 optimizer states; loss error vs BF16 < 0.25% | `models.json/deepseek-v3.fp8_recipe`, `.fp8_loss_error` = "<0.25%" *(proposed)* | 03 §5.3, §1.3 |
| DeepSeek-V3 reserves 20 of 132 SMs for communication; all-to-all and pipeline traffic "can be fully hidden" | `models.json/deepseek-v3.comm_sms` *(proposed by `parallelism`)* | 03 §4.5, §5.2 |
| Llama 4 Behemoth: 32K GPUs, FP8, 390 TFLOPS per GPU; Meta gave no MFU | `models.json/llama-4-behemoth.achieved_tflops_per_gpu` = 390, `.training_gpus` = "32K", `.pretrain_precision` = "FP8" *(the source says "32K", exact count unknown)* | 03 §5.1, §6 |
| NVIDIA Megatron Core on GB300 NVL72: 1,648 TFLOPS per GPU on DeepSeek-V3 pre-training with 256 GPUs, 2.72× GB200 NVL72's 606 (precision not stated, so no MFU is derivable) | `hardware.json/gb300-nvl72.megatron_dsv3_tflops_per_gpu` = 1648 *(proposed)* | 03 §5.1 |
| Low-precision pretraining in 2026: MiMo-V2-Flash trained in FP8 over 27T tokens; Nemotron 3 Super and Ultra, both pretrained in NVFP4 (confirmed by both arXiv abstracts); DeepSeek-V4 and Kimi K3 use FP4 quantization-aware training only in post-training | `models.json/mimo-v2-flash.pretrain_precision` = "FP8"; `models.json/nemotron-3-super.pretrain_precision` and `nemotron-3-ultra.pretrain_precision` = "NVFP4" (confirmed) | 02 §1.7; 03 §1.3 |
| DeepSeek-V4 warns that fully fused kernels make power throttling a limiter | `models.json/deepseek-v4-pro.notes_power` *(proposed, string)* | 03 §5.2 |
| Kimi K2 (2025): zero loss spikes over 15.5T tokens with MuonClip: training stability, not hardware reliability | `models.json/kimi-k2.loss_spikes` = 0 *(proposed)* | 03 §5.4 |

Not shown: Qwen3-Max's "+30% MFU" (low-quality source), xAI cluster sizes (UNVERIFIED), rental prices as
market facts (unsourced; the price slider is labeled an assumption).

## 9. Takeaways
1. A run's FLOPs are about 6 × parameters × tokens; dividing by peak × MFU gives GPU-hours. Llama 3.1
   405B: 3.8 × 10²⁵ FLOPs, 38–43% MFU while training, 30.84M H100-hours in all (a run-average of at least
   34.5%). Always name the peak an
   MFU divides by, especially with FP8.
2. Utilization is lost to memory-bound kernels, exposed communication and bubbles; overlap (DeepSeek-V3
   gave 20 SMs to communication) and FP8 matmuls with fine-grained scales win part of it back.
3. Failures scale with GPU count (Llama 3.1 405B: one every 3.1 hours on 16,384 GPUs); checkpoint interval
   trades save time against lost work, and the failure tax grows with scale (9% → 28% from 16K to 100K
   GPUs with the same parts). GPU-hours times price is the bill (DeepSeek-V3: $5.6M by its own
   assumption).

## 10. Next and go deeper
Next: none (end of the GPUs & scale section). Related: `scaling-laws` (choosing N and D), `serving-calculator` (pricing inference).
Go deeper (brief 03 §7.2, 05 §1.2): Meta, *The Llama 3 Herd of Models* (https://arxiv.org/abs/2407.21783),
the infrastructure and reliability sections · Stas Bekman, *ML Engineering Open Book*, fault tolerance
(https://github.com/stas00/ml-engineering) · EleutherAI, *Transformer Math 101*
(https://blog.eleuther.ai/transformer-math/).

## 11. Key-frame sketch
Frame 10 end state, desktop width. Numbers from the §6 reproducer; bars on one scale, 36.97M GPU-h = 40
characters; `/` = hatched "lost to failures".
```text
Training at 10,000 GPUs           10 / 11  [<] [Play] [>]
 same scale: 36.97M GPU-h = full width
(16,384 GPUs)  MTBF 3.09 h   checkpoint every 13.6 min
 [useful 26.62M GPU-h        |/lost 2.62/]   9.0%  74.4 days
 100,000 GPUs  MTBF 30.4 min checkpoint every 5.51 min
 [useful 26.62M GPU-h        |////lost 10.35M////] 28.0%  15.4 days
 save 30 s, restart 3 min (stand-ins)
 With 100,000 of the same GPUs, failures come every 30
 minutes and checkpointing loses 28% of the run. ...
```

## 12. Open questions for the reviewer
Caption check (2026-10-07, after the review, the `rlvr-grpo` counter adapted): frames 1–11 are
28/2, 24/2, 29/2, 30/2, 26/2, 29/2, 28/2, 30/2, 29/2, 29/2, 26/2 (words/sentences); no operators.

**Data-pass keys**
- `models.json/llama-3.1-405b`: `training_flops 3.8e25`, `training_gpu_hours 30.84e6` (model card; note
  "covers more than the 15.6T-token pretraining"), `mfu_bf16 [0.38, 0.43]` (note: MFU while training,
  with the three configs), `interruptions_54d 419`, `interruptions_planned 47`, `effective_time ">90%"`,
  `reliability_notes` (03 §4.6, §5.1, §5.4, §5.5; CONFIRMED).
- `models.json/deepseek-v3`: `training_gpu_hours 2.788e6`, `pretrain_gpu_hours 2.664e6`,
  `training_cost_usd_reported 5.576e6` (note "at $2/GPU-h, excludes research and ablations"),
  `fp8_recipe`, `fp8_loss_error "<0.25%"` (03 §5.3, §5.5; CONFIRMED).
- `models.json/llama-4-behemoth` (new entry): `training_gpus "32K"` (string: 32,000 vs 32,768 unknown),
  `pretrain_precision "FP8"`, `achieved_tflops_per_gpu 390` (03 §5.1, CONFIRMED Meta blog).
- `hardware.json/gb300-nvl72.megatron_dsv3_tflops_per_gpu 1648`, `gb200-nvl72.megatron_dsv3_tflops_per_gpu
  606` (03 §5.1, CONFIRMED figures, precision unstated).
- `models.json/mimo-v2-flash.pretrain_precision "FP8"` (02 §1.7, CONFIRMED); `models.json/kimi-k2.loss_spikes
  0` (03 §5.4); `models.json/deepseek-v4-pro.notes_power` (03 §5.2).
- `hardware.json/h800` (now in data, shared with `cluster-topology`): it holds NVLink only; dense BF16/FP8
  peaks are not in data. The DeepSeek-V3 preset uses H100 peaks with a visible note, and the page cites no H800 FLOPS.
- Nemotron NVFP4 pretraining (resolved): both `nemotron-3-super` and `nemotron-3-ultra` were pretrained in
  NVFP4 (`pretrain_precision`, confirmed by arXiv 2604.12374 and 2606.15007); there is no conflict.
**Graph changes:** none (no slug lists this page as a prereq; `Next:` is empty by design).
**Judgment calls:** none open; all settled in §13.

## 13. Reviewer rulings (expert review, 2026-10-07)
Settled and applied (README lesson 20):
- **Two MFUs, named apart (lesson 16):** "MFU while training" (the slider; Llama's 38–43%) and
  "run-average MFU" (`mfuFrom`; Llama's 34.5%, DeepSeek-V3's 34.6%), related by run-average =
  while-training × (1 − loss).
- **DeepSeek preset reproduces its record (lesson 27):** MFU while training 35.6% (0.35637), so `runPlan`
  returns 2.664M GPU-hours; the readout says "by construction".
- **34.5% is a floor:** the card's 30.84M GPU-hours cover the long-context and post-training stages too.
- **Model name:** Llama 3.1 405B everywhere the card's numbers appear; the paper title stays "The Llama 3
  Herd of Models"; id `llama-3.1-405b`.
- **Nemotron:** the on-page column may now say Super and Ultra (both NVFP4); the ids live in §8 and §12.
- **Validity (lesson 22):** `runPlan` returns `valid`; the toy replaces its readouts outside interval +
  restart ≤ half the cluster MTBF; the 100K case stays inside.
- **6ND exporter:** `math/scale.js` exports `FLOPS_PER_PARAM_TOKEN` and `trainingFlops`; `scaling-laws`
  and `cluster-topology` import them.
- **Behemoth:** kept, with the visible "measured against H100 peaks"; if the data pass can't confirm
  H100s, frame 7 shows DeepSeek-V3 alone.
- **H800 peak:** visible stand-in (H100 peaks).
- **Stand-in reliability constants:** kept and labeled; try-this 1 says it is a consistency check, not a
  prediction.
- **Checkpoint formula:** first principles (spec §7), with its validity caveat in §7.
- **Lost segments:** hatched ("excluded"), via the proposed `shareBar` `hatched: true` part flag; in the
  lane, `laneTimeline`'s `lost` kind.
- **Applied Shoulds and nice-to-haves:** frame 9 shows a 90-minute window; frame 10 prints its shared
  scale; frame 1 says why 6; the GB300 Megatron row says no MFU is derivable; §3's closing claim is
  scoped to what try-this 3 shows.
- Data pass 2026-10-07: `llama-4-behemoth` and `mimo-v2-flash` precision cite `pretrain_precision`; Nemotron 3 Super and Ultra are both NVFP4 (no conflict); H800 has NVLink only, so the stand-in H100 peaks stay labeled.

## 14. Plan 3 rulings applied (S3, 2026-10-08)
- P3-R4: `stepTime` → `trainingStepMs` (§6 signature, reproducer); unqualified "step time" is Serving's.
- P3-R19: frame 1's two-part bar is a `shareBar`.
- P3-R11: frame 2's dashed outline is a labeled tick on the bar.
- P3-R9 and S3-C's final API: frame 9's lane uses `ticks` (saves), `kind: 'lost'` segments and `gaps`
  (restarts); frame 4's waiting lane is `kind: 'idle'`.
- S3 options: `gpu` `litSms` (frame 5; it indexes the 12 drawn tiles, the page prints "20 of 132 SMs"),
  `shareBar` `hatched: true` parts (frames 3, 10).
- S3-B: the save / lost work / restart split is `checkpointLossParts`; `bestInterval`'s example reads 0.2270 h.
- X-3: "about 3× GB200 NVL72's 606" → "2.72×" (1,648 ÷ 606), through `formatRatio`.
- README lesson 35: MTBF, intervals and days print through `formatDuration` (3.09 h → 30.4 min; 13.6 →
  5.51 min; 9.63 min; 74.4 days); the reproducer and signatures keep raw numbers.
- X-1: the Llama 4 Behemoth row drops its year (no confirmed release date); "Llama 3.1 405B (2024)" and
  "Kimi K2 (2025)" stay with confirmed `release_date` keys added.
- P3-R14 (data gap 19): `llama-3.1-405b.interruptions_hardware_share` = 0.78 and
  `.interruptions_gpu_share` = 0.587 (confirmed, arXiv 2407.21783) back frame 8's "78% hardware, GPU issues
  58.7%".
- P3-R16: no "Check my work" (the "Check against the record" output is the check).
