# Racks and networks (`cluster-topology`)

Track: training · Section: gpus · Prereqs: parallelism (matches `shared/concepts.json`)
Next: `scale-reliability` (the only slug whose `prereqs` list `cluster-topology`)
Status: approved (expert review)
Sources: 03 §1.1 (hierarchy), §1.4 (rack-scale systems), §2.1, §2.2, §2.3, §4 (collective cost model),
§4.2, §4.5, §7.1 (toys 4, 7), §7.2 · 05 §1.1, §1.2, §2. Expert review (Fable 5.1, 2026-10-07) applied;
see §13.

Running example: GPT-3's shape (d_model 12,288, 96 blocks; the `decoder-anatomy` / `training-memory`
preset) split the way `parallelism` split it, on two systems: an H100 HGX cluster (8-GPU servers) and a
GB200 NVL72 rack (72 GPUs in one NVLink domain). Every "comm vs compute" percentage on the page has one
basis, a **full training step (forward and backward)**: the time to send a cut's bytes over a link
during one step divided by the time to do that step's math at the chip's dense BF16 peak (one
definition, §6; README lessons 16 and 25). The page names the basis once on screen. Llama 3's 24K
cluster and DeepSeek-V3's all-to-all are the real configurations.

## 1. Learning objective
After this page you can draw the two layers of a GPU cluster (a fast NVLink domain of 8 or 72 GPUs, and
a network about 9 times slower per GPU; frames 1–2), estimate how long each parallelism's traffic takes
relative to its compute on either layer (frames 3–5, try-this 1–2), state the placement rule (the
chattiest cut on the fastest link) and Llama 3's order (frames 6–7), and explain how rail-optimized
networks and 72-GPU racks change where expert parallelism can run (frames 8–10, try-this 3).

## 2. Misconceptions to correct
- **Misconception:** "A cluster is GPUs on one fast network." → **Reality:** it is islands: inside a
  server (or an NVL72 rack) GPUs share NVLink at 450 GB/s each way on H100s; between islands each GPU
  gets one network port, 50 GB/s each way at 400 Gb/s. That 9× gap decides where each cut can live. Corrected
  by frames 1–2. (03 §2.1)
- **Misconception:** "Faster GPUs make communication relatively cheaper." → **Reality:** the opposite:
  communication time is fixed by the link while compute time shrinks. GPT-3-shaped tensor parallelism
  over 8 GPUs costs 28% of its compute time on H100 NVLink and 35% on GB200 NVL72, whose compute grew
  faster than its NVLink. Corrected by try-this 1. (03 §1.4 headline trend; derived)
- **Misconception:** "Put the parallelism that sends the most bytes on the fastest link." → **Reality:**
  what matters is whether the traffic waits for, or hides behind, compute. Pipeline hand-offs are tiny;
  data-parallel syncs are large but happen once per step and overlap with the backward pass; tensor
  all-reduces sit inside every layer and block it. Llama 3's order (tensor, context, pipeline, data;
  innermost to outermost) follows that. Corrected by frames 3–6. (03 §2.3)
- **Misconception:** "A 72-GPU NVLink rack lets tensor parallelism scale to 72." → **Reality:** the
  link per GPU is still finite: for GPT-3's shape, 16-way tensor parallelism inside an NVL72 rack would
  spend 75% of its compute time communicating, still more than half. What the bigger domain really unlocks is expert
  parallelism, whose all-to-all can hide behind compute inside the rack (45%) but not over the network
  (407%). Corrected by frame 10 and try-this 3. (03 §2.2, §2.3, §4.5; derived)

## 3. Hook and intuition (final wording)
**Hook:** Llama 3.1 405B's 16,384 GPUs could have been wired as one flat network. Why does it matter
which GPUs share a server, and which parallelism runs where?

A GPU cluster is not one network but two. Inside a server, or inside a rack-scale system like NVL72,
GPUs connect through NVLink switches: on H100s, 900 GB/s per GPU counting both directions. Between
servers, each GPU gets one network port: 400 Gb/s on H100-era clusters, which is 50 GB/s each way, nine
times less than NVLink's 450. Every byte of communication runs on one of those two layers.

The parallelisms from the last lesson send very different traffic. Tensor parallelism all-reduces
partial sums inside every layer, and the next layer can't start until they arrive. For GPT-3's shape
that traffic costs 28% of the compute time over NVLink and two and a half times the compute time over
the network (counted over a full forward-and-backward step). Data parallelism sends a big gradient sync, but once per step, while the backward pass is
still running, so it can hide. Pipeline parallelism hands one activation across each stage boundary per
micro-batch (and its gradient back), about 1.5% of the compute even over the network. So the rule is: the chattiest cut, the one
that blocks compute, gets the fastest link. That is the hook's answer: Llama 3 kept tensor parallelism
inside each 8-GPU server and put pipeline and data parallelism on the network, data outermost.

Rack-scale systems widen the fast island. A GB200 NVL72 puts 72 GPUs on one NVLink domain, nine
servers' worth, which matters most for mixture-of-experts training: its all-to-all can hide behind
compute inside the rack but not across the network. The cost is that everything outside the island
still runs at network speed, and big clusters thin the network further between pods.

## 4. Visual metaphor
The stage is a map of links. Servers are `rack` glyphs (8 GPUs; 72 for NVL72 with `cols: 9`) drawn with
the default `linkWidth` (ruling P3-R8: link thickness never encodes an amount; the `rack` `labels` option
numbers the GPUs in frame 8); network links between racks use the same `rack` link style, and the
bandwidth is printed beside every link ("450 GB/s each way", "50 GB/s each way"), the one encoding of
link speed. Traffic is a `flow` dot on a link (carry
`activation` for tensor and pipeline traffic, `gradient` for data-parallel syncs, `token` for expert
dispatch). Each cut's cost is a two-lane `laneTimeline` (compute lane, comm lane) on one time axis,
with the percentage printed and the visible line "one full training step; both lanes in the same time
units" (lesson 21). The compute lane is drawn 120 px wide; a longer comm lane is cut at the 520 px track
(4.33× compute) with the `laneTimeline` options `scale: 1.2` (px per unit, so the compute lane's 100 units
are 120 px) and one global `cap: { at: 433.3, label: 'continues: N%' }` (e.g. 'continues: 537%' for tensor 16 over the network; 520 px ÷ 1.2; ruling P3-R9), and
ends in an arrow with its number printed before it; `w` is the lane-label gutter plus the 520 px track. Pods in frame 7 are faint filled regions (no outline: outlines mean
selection) labeled with their size. The followed GPU, "GPU 1" of server 1, has the selection outline in
every frame.

Glyphs used (from spec §5.1 and the built library): `rack` (frames 1–2, 6–10), `gpu` (frame 2:
one GPU, `showMem: false`), `flow`, `block` (frame 7:
leaf and spine switches as small labeled blocks), `token` (frame 8: one token chip traveling).
Shared glyph built in S3: `laneTimeline` (ruling P3-R9; segments `{ from, to, kind: 'compute' | 'comm',
label }`, `scale`, one global `cap`) for comm vs compute.

New glyphs proposed: none. Glyph options built in S3: `rack` `labels: ['1', …, '8']` so frame 8 can
number rails; `gpu` `showMem: false`.

Frame 10 layout (lesson 18): the 72-GPU `rack` at `cols: 9` is 244 × 218 px (the built geometry, ruling
P3-R8); it stands alone at the left with the printed line "= 9 HGX servers". The nine servers appear only
as the merge's start state, collapsed to three 8-GPU racks (each 114 × 62 px overall at `cols: 4`) and the text
"+ 6 other servers", so the frame stays inside 580 × 366.

Plain labeled marks: link bandwidth labels; "one full training step; compute at dense BF16 peak; real
kernels run slower, so real ratios are smaller" (frame 3); "tokens per replica per step: stand-in" (frame 4); "1:7" on the
oversubscribed uplinks (frame 7); "for DeepSeek-V4's expert shape" (frame 9); pod regions' labels.

Terms introduced (one per frame): scale-up domain (frame 1), scale-out network (2), comm-to-compute
ratio (3), overlap (4), none (5), placement order (6), oversubscription (7), rail (8), hiding condition
(9), NVL72 (10). Terms assumed: NVLink (named on the glyph, defined in frame 1's caption as the
in-server link), all-reduce, all-to-all, tensor / pipeline / data / expert parallelism, stage,
micro-batch (`parallelism`); HBM, TFLOPS, BF16 (`gpu-primer`); gradient (`training-memory`).

Indexing: GPUs within a server are numbered 1–8 on screen; servers, racks and pods from 1.

## 5. Animation script
All link figures per GPU and **each way**, read from the data keys `<chip>.nvlink_gb_s_each_way`
(settled; NVIDIA's published both-directions totals, H100 900 GB/s, Blackwell 1.8 TB/s, Rubin 3.6 TB/s,
live in each key's note, 03 §2.1) and the network keys `network-400g` (50 GB/s) and `network-800g`
(100 GB/s). Workload: GPT-3's shape; basis: one full training step; compute at dense BF16 peak (H100 989
TFLOPS). Numbers from `math/topology.js` (§6).

| # | On screen | What moves | Caption (final wording) | Numbers shown |
|---|---|---|---|---|
| 1 | One `rack` glyph of 8 GPUs labeled "H100 server (HGX)"; internal links (default width) labeled "NVLink: 900 GB/s per GPU, both directions = 450 each way". GPU 1 outlined. | Links draw from GPU 1 to every other GPU. | Inside a server, eight GPUs talk through NVLink switches at 450 GB/s each way on H100s. That fast island is called the scale-up domain. | 8 GPUs · 900 GB/s bidirectional = 450 GB/s each way |
| 2 | Four servers in a row; thin links from each GPU to a network switch, labeled "400 Gb/s = 50 GB/s each way". A side ladder, each way only: NVLink 450 → network 50 GB/s (HBM is a shared total and stays off this ladder, lesson 28; `gpu-primer` link). | The network links draw; the ladder's second bar shrinks to one ninth. | Between servers, each GPU gets one network port: 50 GB/s each way, nine times slower than NVLink. That second layer is the scale-out network. | 450 ÷ 50 = 9× |
| 3 | Tensor parallelism over 8 GPUs, drawn twice: inside one server, and spread over 8 servers. Under each, a `laneTimeline`: compute vs comm. Plain mark: "one full training step; compute at dense BF16 peak; real kernels run slower, so real ratios are smaller". | All-reduce dots run on both; the comm lanes grow to 28% and 250% of the compute lane (both fit under the 4.33× cap). | Tensor parallelism all-reduces inside every layer, and the next layer waits. Over NVLink that takes 28% as long as the math; over the network, two and a half times longer. | per layer per step (forward and backward): 352 MB sent per GPU vs 2,783 GFLOP per GPU · ratio 27.8% (NVLink) vs 250.4% (network) |
| 4 | Data parallelism across 64 servers' worth of GPUs (collapsed: "GPU 1 of servers 1, 2, …, 64", "61 others"). One gradient sync per step, its comm lane drawn *under* the backward part of the compute lane. Plain mark: "tokens per replica per step: 262,144 (stand-in)". | The sync dot runs while the backward segment of the compute lane is still filling. | Data parallelism syncs gradients once per step, and the sync can overlap with the backward pass that produces them. Over the network it costs 5% of the compute time here. | ratio 5.0% (network) · rises to 79.2% at 16,384 tokens per replica |
| 5 | Pipeline parallelism, 16 stages, one per server; a small activation dot hops forward between servers, then a gradient dot hops back. | The dots hop; the comm lane stays a sliver. | Pipeline stages hand over one activation per micro-batch at each boundary, and its gradient comes back. Even over the network that is about 1.5% of the compute. | ratio 1.5% (network), 0.2% (NVLink) at 16 stages |
| 6 | Llama 3's 8,192-GPU layout: tensor 8 inside each server (on NVLink, labeled), pipeline 16 across servers, data 64 across groups of pipelines. A plain list "innermost → outermost: tensor, context, pipeline, data". Visible note: "data parallelism sends more per unit of compute than pipeline here (5.0% vs 1.5%), but it overlaps with backward; pipeline hand-offs sit between stages". | The three cuts light in order, innermost first. | So the cut that blocks compute gets the fastest link. Llama 3 kept tensor parallelism inside each 8-GPU server and put pipeline, then data parallelism, on the network. | 8 × 16 × 64 = 8,192 · order TP, CP, PP, DP |
| 7 | Meta's 24K-GPU cluster: two `rack` glyphs of 16 GPUs (2 servers each) under a leaf `block`, "190 other racks"; a faint region "pod: 3,072 GPUs, full bandwidth"; eight pod regions side by side ("6 other pods") joined by thin uplinks labeled "1:7". | Pods tile in; uplinks thin out; a data-parallel sync dot crosses pods on a thin link. | Meta's 24,576-GPU Llama 3 cluster has full bandwidth only within pods of 3,072 GPUs. Above them, links are oversubscribed seven to one, so chatty traffic stays inside a pod. | 16 per rack × 192 racks = 3,072 per pod · 8 pods = 24,576 · 400 Gb/s per GPU · 1:7 above pods |
| 8 | Four servers; GPU 3 of every server labeled and joined to the same "rail 3" switch. A `token` chip on server 1, GPU 3, travels over rail 3 to server 4, GPU 3, then over NVLink to GPU 6 (its expert). | The token chip takes the two hops. | In a rail-optimized network, GPU 3 of every server shares one switch. DeepSeek sends each token over the network to the same-numbered GPU, then over NVLink to its expert. | DeepSeek-V3 (2024, H800): NVLink 160 GB/s vs network 50 GB/s (3.2×), DeepSeek's stated effective rates; the paper gives no direction, and 50 GB/s matches a 400 Gb/s port each way; the H800's NVLink is reduced vs the H100 · each token reaches at most 4 servers |
| 9 | A `laneTimeline` for expert parallelism on an H100 server: compute lane vs all-to-all lane, inside NVLink and over the network. Plain mark "for DeepSeek-V4's expert shape". | The all-to-all lane fits under the compute lane on NVLink (36%) and overflows it on the network (322%). | DeepSeek-V4 states when expert traffic can hide behind compute: roughly 1 GB/s of link for every 6 TFLOPS. An H100 needs 161 GB/s; the network gives 50. | 989 TFLOPS ÷ 6,144 FLOPs per byte = 161 GB/s · NVLink 450 → 35.8% · network 50 → 321.9% |
| 10 | Start state: three 8-GPU racks and "+ 6 other servers". End state: one `rack` with 72 GPUs (`cols: 9`, 244 × 218 px) at the left, labeled "GB200 NVL72: 72 GPUs + 36 Grace CPUs, one NVLink domain, 900 GB/s each way per GPU (1.8 TB/s both directions), 130 TB/s total" and "= 9 HGX servers". Expert groups light inside the rack. | The servers merge into one rack; expert-parallel `flow` dots stay inside it. | A GB200 NVL72 rack puts 72 GPUs in one NVLink domain, nine servers' worth. Expert groups bigger than 8 can now stay off the network. | 72 ÷ 8 = 9 · expert parallelism on GB200 (2,500 TFLOPS BF16): needs 407 GB/s; NVLink 900 → 45.2%; network 100 (800 Gb/s, reported) → 406.9% · tensor 16 inside the rack: 75.4% |

Determinism: every frame is a pure function of (step, progress). Reduced motion shows each frame's end
state. Frames 3–5 and 9–10 draw the ratio functions in §6 with GPT-3's shape, full-step basis.

Caption check: counts in §12.

## 6. Toy
"Put a cut on a link." Pick a system, a parallelism and its degree, and whether it runs inside the
NVLink domain or over the network; read the communication time as a share of the compute time. A
visible line above the controls: "One full training step (forward and backward). GPT-3's shape
(d_model 12,288, 96 blocks), compute at the chip's dense BF16 peak. Real kernels reach 35–55% of peak (`scale-reliability`), so real ratios are smaller,
but their order is the same."

**Controls**
| id | Label | Type | Range / values | Default | Presets |
|---|---|---|---|---|---|
| `system` | System | Preset chips | H100 HGX (8-GPU servers; NVLink 450, network 50 GB/s each way; 989 TFLOPS) · GB200 NVL72 (72-GPU domain; NVLink 900 each way; network 100 GB/s each way, 800 Gb/s ConnectX-8, reported; 2,500 TFLOPS) | H100 HGX | from `hardware.json` (§8): `*.nvlink_gb_s_each_way`, `network-400g`, `network-800g` |
| `cut` | Parallelism | Segmented | tensor · pipeline · data · expert | tensor | – |
| `degree` | Degree | Slider (powers of 2) | tensor 2–64 · pipeline 2–32 · data 2–1,024 · expert (not used) | 8 | – |
| `where` | Runs on | Segmented | inside the NVLink domain · over the network | inside | "inside" is disabled when the degree exceeds the domain (8 or 72) and the visible note says "this cut no longer fits in one NVLink domain" (ruling P3-R12: a single option the selection cannot use is disabled with a visible note) |
| `tokens` | Tokens per replica per step (data only) | Slider (powers of 2) | 4,096 … 1,048,576 | 262,144 | stand-in; visible label |

**Live outputs**
| Output | Formula / `math/` function | Units / format |
|---|---|---|
| **Comm as % of compute** | `tpCommRatio`, `ppCommRatio`, `dpCommRatio` or `epCommRatio` (below) | %, 1 decimal |
| Lanes | compute = 100 units, comm = ratio × 100; the compute lane is drawn 120 px wide and the comm lane is capped at the 520 px track (4.33× compute, `cap`), with an arrow "continues: 2,254%" and the number printed when it overflows; visible line "both lanes in the same time units; comm lane cut at 4.33× compute" | `laneTimeline`; the cap ratio through `formatRatio` |
| Can it hide? (text, from 03 §2.3, §5.2) | tensor: "no: on the critical path"; pipeline: "mostly, in 1F1B's steady state"; data: "yes, during backward"; expert: "only below the hiding line" | text |
| Link needed for expert traffic to hide | `epMinLinkGBps({ peakTflops })` | GB/s |
| Bytes per GPU behind the ratio (tensor) | `ringAllReduceBytes` (imported from `math/parallel.js`) × 4 per layer per step (2 forward, 2 backward) | MB per layer per step |

**Check my work** (default state: tensor, degree 8, H100 HGX, inside; templated for any state from the
cut's function in `math/topology.js` with its bytes and FLOPs per GPU; mono, `aria-live="polite"`; this
exact text appears on the page):
```text
bytes per GPU, one layer, full step
  = 4 × ring all-reduce of 2,048 × 12,288 × 2 B over 8 GPUs = 352.32 MB
FLOPs per GPU, one layer, full step
  = 72 × 2,048 × 12,288² ÷ 8 = 2,783.1 GFLOP
comm ÷ compute = (352.32 MB ÷ 450 GB/s) ÷ (2,783.1 GFLOP ÷ 989 TFLOPS) = 27.8%
```

**Try this** (each leads to a named insight)
1. Tensor, degree 8, H100 HGX: inside 27.8%, network 250.4%. Raise the degree to 16: it no longer
   fits in a server, so it runs over the network: 536.6%. Switch to GB200 NVL72, degree 8, inside:
   35.2%; degree 16, inside: 75.4%. → **Insight: tensor parallelism belongs inside the fastest domain
   and stays small (around 8–16), and newer chips make it harder, because compute grew faster than
   NVLink.**
2. Data, degree 64, over the network: 5.0% at 262,144 tokens per replica; slide tokens down to 16,384:
   79.2%. Then pipeline, degree 16, network: 1.5%. → **Insight: data-parallel syncs are cheap only when
   each replica processes many tokens per step; since they also overlap with the backward pass, they
   can live on the slowest, outermost links.** (03 §4.1: the per-device batch condition.)
3. Expert on H100 HGX: inside 35.8%, network 321.9%; the "link needed" readout says 161 GB/s. Switch
   to GB200 NVL72: needed 407 GB/s; inside 45.2%, network 406.9%. → **Insight: expert all-to-all hides
   only inside the NVLink domain, which is why expert groups stay inside a server or a 72-GPU rack** (and
   why DeepSeek-V3 capped each token at 4 servers, frame 8).

**`math/topology.js`** (new module, owned by this page; pure, no DOM; tests first). Peaks in TFLOPS,
links in GB/s each way. Each ratio is one definition on one basis: (bytes sent per GPU in one full
training step ÷ link) ÷ (that step's FLOPs per GPU ÷ peak).

```js
tpCommRatio({ tp, hidden, peakTflops, linkGBps }) → number
//   one layer, full step: 2 all-reduces of s·b·h BF16 numbers forward and 2 backward → 16(t − 1)/t · s·b·h
//   bytes (03 §4.2 gives the forward half), vs 72·s·b·h² / t FLOPs; s and b cancel: 2(t − 1) · peak / (9 · h · link)
//   (8, 12288, 989, 450) → 0.2782 · (8, 12288, 989, 50) → 2.5037 · (16, 12288, 989, 50) → 5.3660
//   (8, 12288, 2500, 900) → 0.3516 · (16, 12288, 2500, 900) → 0.7535
ppCommRatio({ pp, hidden, layers, peakTflops, linkGBps }) → number
//   one micro-batch at one stage boundary, full step: 2·s·b·h bytes forward + 2·s·b·h backward vs
//   72·s·b·h²·(layers/pp) FLOPs → pp · peak / (18 · h · layers · link)
//   (16, 12288, 96, 989, 50) → 0.0149 · (16, 12288, 96, 989, 450) → 0.0017
dpCommRatio({ dp, tokensPerReplica, peakTflops, linkGBps }) → number
//   one step: ring all-reduce of 2 bytes per parameter vs FLOPS_PER_PARAM_TOKEN (6, imported from
//   math/scale.js) per parameter per token → 2(N − 1)/N · 2 · peak / (6 · T · link)
//   (64, 262144, 989, 50) → 0.0497 · (64, 16384, 989, 50) → 0.7924 · (64, 262144, 989, 450) → 0.0055
epCommRatio({ peakTflops, linkGBps, flopsPerByte = 6144 }) → number   // DeepSeek-V4's hiding condition, C/B ≤ 6,144
//   (989, 450) → 0.3577 · (989, 50) → 3.2194 · (2500, 900) → 0.4521 · (2500, 100) → 4.069
epMinLinkGBps({ peakTflops, flopsPerByte = 6144 }) → number
//   989 → 160.97 · 2500 → 406.9
```
Collective costs are imported from `math/parallel.js` (`parallelism` owns them), not redefined.
`dpCommRatio` imports `FLOPS_PER_PARAM_TOKEN` from `math/scale.js` (settled; `scale-reliability` owns
it), so the factor 6 is typed once in the codebase.

Reproducer (re-run 2026-10-07 after the review on the full-step basis, against the scratch
implementation; re-run once the module exists):
```sh
node -e '
import("./math/topology.js").then((m) => {
  const f = (x) => Number((100 * x).toFixed(1));
  for (const [sys, peak, nv, net] of [["H100 HGX", 989, 450, 50], ["GB200 NVL72", 2500, 900, 100]]) for (const [ln, bw] of [["nvlink", nv], ["network", net]])
    console.log(sys, ln, [2, 4, 8, 16, 32, 64].map((t) => f(m.tpCommRatio({ tp: t, hidden: 12288, peakTflops: peak, linkGBps: bw }))).join(" "),
      "|", [4, 16].map((p) => f(m.ppCommRatio({ pp: p, hidden: 12288, layers: 96, peakTflops: peak, linkGBps: bw }))).join(" "),
      "|", [16384, 262144].map((T) => f(m.dpCommRatio({ dp: 64, tokensPerReplica: T, peakTflops: peak, linkGBps: bw }))).join(" "),
      "|", f(m.epCommRatio({ peakTflops: peak, linkGBps: bw })), m.epMinLinkGBps({ peakTflops: peak }).toFixed(1));
  console.log(2 * 8 * 7 / 8 * 2048 * 12288 / 1e6, "MB", 3 * 24 * 2048 * 12288 ** 2 / 8 / 1e9, "GFLOP");
});'
```
Output (2026-10-07, full step), tensor degrees 2/4/8/16/32/64 | pipeline 4/16 | data 64 at 16,384 /
262,144 tokens | expert | link needed: H100 NVLink 4.0 11.9 27.8 59.6 123.2 250.4 | 0.0 0.2 | 8.8 0.6 |
35.8 · H100 network 35.8 107.3 250.4 536.6 1,108.9 2,253.6 | 0.4 1.5 | 79.2 5.0 | 321.9 · 161.0 GB/s ·
GB200 NVLink 5.0 15.1 35.2 75.4 155.7 316.5 | 0.1 0.2 | 11.1 0.7 | 45.2 · GB200 network 45.2 135.6 316.5
678.2 1,401.5 2,848.3 | 0.5 1.9 | 100.1 6.3 | 406.9 · 406.9 GB/s. Frame 3 per layer per step: 352.32 MB
sent, 2,783.1 GFLOP per GPU (2,048 tokens, micro-batch 1). Ratios 450 ÷ 50 = 9; Llama pod 16 × 192 = 3,072; 8 ×
3,072 = 24,576; NVL72 72 ÷ 8 = 9.

Tests to write first: the worked examples; every ratio is linear in `peakTflops` and inverse in
`linkGBps`; `tpCommRatio` is 0 at tp = 1, `dpCommRatio` 0 at dp = 1; `epCommRatio(peak, epMinLinkGBps(peak))
=== 1`.

## 7. Show me the math
```tex
\text{ratio} = \frac{\htmlClass{hl-comm}{\text{bytes sent per GPU}} / \htmlClass{hl-link}{\text{link}}}{\htmlClass{hl-flops}{\text{FLOPs per GPU}} / P_{\text{peak}}}
```
```tex
\text{TP: } \frac{16\frac{t-1}{t}\,sbh / L}{72\,sbh^2 / (t P)} = \frac{2(t-1)\,P}{9\,h\,L}
\qquad
\text{PP: } \frac{p\,P}{18\,h\,n_{\text{layers}}\,L}
\qquad
\text{DP: } \frac{2(N-1)\,P}{3\,N\,T\,L}
\qquad
\text{EP (DeepSeek-V4): hidden if } \frac{P}{L} \le 6144
```
Shapes and symbols: t, p, N = tensor, pipeline, data degrees; s, b, h = tokens per sequence,
micro-batch, d_model (GPT-3: h = 12,288, 96 blocks); T = tokens per data-parallel replica per step;
P = peak FLOP/s; L = link bytes/s per direction. Assumptions shown in the panel: BF16 activations and
gradients (2 bytes); ring collectives; basis: one full training step (forward 24·s·b·h² matmul FLOPs
per layer, backward twice that; attention scores ignored); 6 FLOPs per parameter per token per step. Color links: `hl-comm` =
the comm lane, `hl-link` = the link label under the dot, `hl-flops` = the compute lane.

## 8. In today's models (Oct 2026)
| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| NVLink per GPU, each way: H100 450 GB/s, B200/B300 900 GB/s, Rubin 1,800 GB/s (NVIDIA publishes the both-directions totals 900 / 1,800 / 3,600; the product page lists 3,000 for Rubin) | `hardware.json/h100.nvlink_gb_s_each_way` = 450, `b200.…` = 900, `b300.…` = 900, `rubin.…` = 1800 or [1500, 1800] *(settled key; note carries the published total)* | 03 §2.1 |
| Scale-out per GPU, each way: 400 Gb/s (50 GB/s) in H100-era clusters; 800 Gb/s (100 GB/s) ConnectX-8 on Blackwell (reported); GB300 NVL72 uses ConnectX-8 at 800 Gb/s per GPU (confirmed) | `hardware.json/network-400g.gb_s_each_way` = 50, `network-800g.gb_s_each_way` = 100 (the serving track's ids) | 03 §2.1 |
| GB200 / GB300 NVL72: 72 GPUs + 36 Grace CPUs in one NVLink domain, about 130 TB/s aggregate | `hardware.json/gb200-nvl72.scale_up_domain`, `.rack_nvlink_tbps`; `gb300-nvl72.*` | 03 §1.4, §2.2 |
| Vera Rubin NVL72: 72 Rubin GPUs, 260 TB/s NVLink 6; Rubin Ultra NVL576 planned for H2 2027 (reported) | `hardware.json/rubin.scale_up_domain`; `rubin-ultra.scale_up_gpus` = 576 *(proposed, reported)* | 03 §1.4 |
| Meta's Llama 3 cluster (2024 paper): 16 GPUs per rack, 192 racks per pod = 3,072 GPUs with full bisection bandwidth, 8 pods = 24K GPUs, 1:7 oversubscribed above pods, 400 Gb/s per GPU, topology-aware scheduling | `hardware.json/meta-llama3-cluster.*` *(proposed entry: `gpus_per_rack` 16, `gpus_per_pod` 3072, `pods` 8, `oversubscription` "1:7", `per_gpu_gbps` 400)* | 03 §2.2 |
| Llama 3 order, innermost to outermost: TP, CP, PP, DP ("innermost parallelism requires the highest network bandwidth and lowest latency") | `models.json/llama-3.1-405b.parallelism_order` *(proposed by `parallelism`; the paper calls the model Llama 3, the checkpoint is Llama 3.1 405B)* | 03 §2.3 |
| DeepSeek-V3 (2024): H800s with NVLink 160 GB/s vs InfiniBand 50 GB/s per GPU (3.2×), DeepSeek's stated effective rates (direction not given; 50 GB/s matches a 400 Gb/s port each way); the H800's NVLink is reduced vs the H100 (200 vs 450 GB/s each way, `hardware.json/h800.nvlink_gb_s_each_way`); all-to-all goes over IB to the same-index GPU, then NVLink; each token reaches at most 4 nodes | `models.json/deepseek-v3.nvlink_effective_gb_s` = 160, `.ib_gb_s` = 50 (notes: "stated effective rates; direction not stated"), `.max_nodes_per_token` = 4 *(proposed)* | 03 §2.1, §2.2, §4.5 |
| DeepSeek-V4 (2026): expert traffic hides behind compute when compute ÷ bandwidth ≤ 6,144 FLOPs per byte (each GB/s hides about 6.1 TFLOPS) | `models.json/deepseek-v4-pro.ep_hiding_flops_per_byte` = 6144 *(proposed)* | 03 §4.5 |
| Kimi K2 (2025) servers: 8 GPUs, 2 TB RAM, 8 × 400 Gb/s RoCE | `models.json/kimi-k2.node` *(proposed; entry proposed by `training-memory`)* | 03 §2.2 |
| NVIDIA's Nemotron 3 RL report: expert-parallel groups must sit on the same rack to stay inside NVLink | `ep_colocation` on the data pass's Nemotron 3 entry for arXiv 2606.15007 *(proposed key; the entry id is the data pass's call; the page never names which Nemotron 3 model)* | 03 §2.2 |
| Tensor parallelism stays compute-bound only up to about 8–16-way (Scaling Book) | none (mechanism, Scaling Book training chapter) | 03 §2.3 |

## 9. Takeaways
1. A cluster is fast islands (NVLink: 8 GPUs per server, 72 per NVL72 rack, 450–900 GB/s each way per
   GPU) on a slower network (50–100 GB/s per GPU), and big clusters thin the network further between
   pods (Llama 3: 1:7).
2. Place by whether traffic blocks compute: tensor parallelism (every layer, on the critical path)
   inside the island; pipeline (tiny hand-offs) and data parallelism (one overlappable sync per step)
   on the network, data outermost. Faster chips make this harder, since compute outgrew links.
3. Expert all-to-all hides only inside the fast island (DeepSeek-V4's rule: about 1 GB/s per 6
   TFLOPS); DeepSeek exploits rail-optimized networks to route it, and NVIDIA's Nemotron 3 report keeps
   expert groups inside one rack.

## 10. Next and go deeper
Next: `scale-reliability` (overlap, utilization, failures and cost at 10,000+ GPUs).
Go deeper (brief 03 §7.2, 05 §1.2): Google, *How to Scale Your Model*, training chapter
(https://jax-ml.github.io/scaling-book/training/) · Stas Bekman, *ML Engineering Open Book*, networking
(https://github.com/stas00/ml-engineering) · DeepSeek, *DeepEP* (https://github.com/deepseek-ai/DeepEP).

## 11. Key-frame sketch
Frame 3 end state (tensor parallelism, degree 8, H100 HGX), desktop width. Ratios from the §6
reproducer (full step); lanes scaled to the compute lane = 10 characters, comm lane cut at the track.
```text
Racks and networks                 3 / 10  [<] [Play] [>]
 one full training step; both lanes in the same time units
 inside one server             | spread over 8 servers
 +-------------------+          | [o]  [o]  [o]  [o] ...
 |(1)=2==3==4        |          |  \    |    |   /
 | 5==6==7==8        |          |  ---- switch ----
 +-- NVLink 450 GB/s +          |    network 50 GB/s
 compute [==========]           | compute [==========]
 comm    [===]  27.8%           | comm    [=========================]  250.4%
 352 MB sent, 2,783 GFLOP per GPU per layer per step
 Tensor parallelism all-reduces inside every layer, ...
```

## 12. Open questions for the reviewer
Caption check (2026-10-07, after the review, the `rlvr-grpo` counter adapted): frames 1–10 are
24/2, 24/2, 30/2, 30/2, 27/2, 28/2, 29/2, 29/2, 27/2, 25/2 (words/sentences); no operators ("GB/s" is a unit).

**Data-pass keys**
- Link keys (settled names): `hardware.json/h100.nvlink_gb_s_each_way = 450`, `b200.… = 900`,
  `b300.… = 900`, `rubin.… = 1800` (or the range [1500, 1800], since the product page lists 3,000 both
  ways), each note "NVIDIA quotes the both-directions total: 900 / 1,800 / 3,600" (03 §2.1). Network:
  `network-400g.gb_s_each_way = 50`, `network-800g.gb_s_each_way = 100` (GB300 NVL72 CONFIRMED;
  ConnectX-8 on Blackwell generally REPORTED).
- `hardware.json/meta-llama3-cluster` (new entry): `gpus_per_rack 16`, `racks_per_pod 192`,
  `gpus_per_pod 3072`, `pods 8`, `oversubscription "1:7"`, `per_gpu_gbps 400` (03 §2.2, CONFIRMED arXiv
  2407.21783).
- `models.json/deepseek-v3`: `nvlink_effective_gb_s 160`, `ib_gb_s 50` (direction not given; notes:
  "stated effective rates"), `max_nodes_per_token 4` (03 §2.1, §4.5, CONFIRMED); `hardware.json/h800` (now in
  data: `nvlink_gb_s_each_way` 200; no peak FLOPS stored, so the page prints none).
- `models.json/deepseek-v4-pro.ep_hiding_flops_per_byte = 6144` (03 §4.5, CONFIRMED).
- `models.json/kimi-k2.node = "8 GPUs, 2 TB RAM, 8 × 400 Gb/s RoCE"` (03 §2.2, CONFIRMED).
- `hardware.json/rubin-ultra.scale_up_gpus = 576` (reported, 03 §1.4).
- `ep_colocation` (03 §2.2, CONFIRMED text of arXiv 2606.15007) on whichever Nemotron 3 entry
  (`nemotron-3-super` or `nemotron-3-ultra`) the data pass ties to that report.
**Graph changes:** none.
**Judgment calls:** none open; all settled in §13.

## 13. Reviewer rulings (expert review, 2026-10-07)
Settled and applied (README lesson 20):
- **One basis (lessons 16, 25):** every ratio is per full training step. Tensor and pipeline ratios are
  2/3 of the draft's forward-only values: TP 8 on H100 NVLink 27.8%, network 250.4%; NVL72 TP 16 75.4%;
  PP 16 over the network 1.5%. Data and expert ratios are unchanged. The ordering of cuts, and so the
  lesson, is unchanged.
- **Bandwidth convention:** links are stored and printed each way (`<chip>.nvlink_gb_s_each_way`, the
  published both-directions figure in the note; network keys per direction); in-server vs network is 9×.
  `disaggregation` uses the same convention.
- **HBM off the per-direction ladder (lesson 28):** frame 2's ladder is NVLink and network only.
- **Measured anchors (lesson 23):** DeepSeek's 160 vs 50 GB/s are labeled "stated effective rates". The
  reviewer asked for "each way"; the brief gives no direction for these figures, so the page says
  "direction not stated" and notes that 50 GB/s matches a 400 Gb/s port each way (rebuttal: printing
  "each way" would assert a convention the source doesn't state).
- **Stage budget (lesson 18):** frame 10 shows the 72-GPU rack alone with "= 9 HGX servers"; the nine
  servers appear collapsed only as the start state.
- **Peak-FLOPs ratios:** accepted with the visible line; no "typical MFU" before `scale-reliability`.
- **Placement and lesson 17:** frame 6's note states why data parallelism sits outermost despite 5.0%
  vs 1.5%.
- **Tokens per replica:** a labeled stand-in (262,144).
- **NVL72 emphasis:** expert parallelism, restated at TP 16 = 75% ("still more than half").
- **`rack` `labels`:** accepted.
- **Applied Shoulds:** frame 4 drops the impossible NVLink figure; `dpCommRatio` imports
  `FLOPS_PER_PARAM_TOKEN`; GB200's 800 Gb/s port is labeled reported; no `nemotron-3` id is proposed;
  checkpoint id `llama-3.1-405b`; lanes carry "both lanes in the same time units" (lesson 21).
- **Nice-to-haves applied:** misconception 1 says "decides where each cut can live"; try-this 3's
  4-server point is credited to frame 8.
- Data pass 2026-10-07: DeepSeek-V3 link keys renamed to `nvlink_effective_gb_s` and `ib_gb_s`; `h800` NVLink (200 GB/s each way) now cited; no H800 FLOPS printed.

## 14. Plan 3 rulings applied (S3, 2026-10-08)
- P3-R8: link thickness never encodes an amount: every `rack` uses the default `linkWidth` and bandwidths
  are printed (§4, frames 1 and 6); the rack sizes are the built ones (244 × 218 at `cols: 9`, 114 × 62 at
  `cols: 4`).
- P3-R9: the capped comm lane is `laneTimeline`'s `cap` option.
- S3-C final API (reconciled 2026-10-08): the lanes use `scale: 1.2` px per unit and one global `cap` at
  433.3 units (the 520 px track); an 8-GPU rack at `cols: 4` is 114 × 62 overall.
- X-3: "(4.3× compute)" → "(4.33× compute)" (§4, §6), through `formatRatio`.
- P3-R12: "inside" stays disabled with its visible note.
- P3-R13: `epCommRatio`'s default `flopsPerByte` (6,144) carries an equality test against
  `deepseek-v4-pro.ep_hiding_flops_per_byte` in `tests/topology.test.js` (the builder's module test).
- X-1 / P3-R14 (data gap 7): "Meta's Llama 3 cluster (2024 paper)": the year is the paper's,
  `hardware.meta-llama3-cluster.release_date` = "2024-07" (confirmed, arXiv 2407.21783); "Kimi K2 (2025)"
  stays with `kimi-k2.release_date` added.
- Conventions: the DeepSeek-V3 link keys' line says "direction not given" (`tests/training-conventions.test.js` (b)).
- X-2 / P3-R16: "Check my work" added (§6), from `tpCommRatio` and `ringAllReduceBytes`.
