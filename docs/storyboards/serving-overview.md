# One request's journey (`serving-overview`)

Track: serving · Section: serving · Prereqs: kv-cache
Status: draft
Sources: 04 §0, §1.1, §1.2, §1.3, §2.1, §2.4, §3.3, §7.1, §7.2, §10; 05 §1.1, §1.2 (llm-inference-explained link-out verdict). Rates in the toy come from `math/serving.js` (the `prefill-decode` model) and one measured anchor (InferenceX, 04 §1.3).

Running example (course-wide): request **A** is the prompt `The cat sat` (3 tokens), answered with `down` `.` and an end token. The same letters A–D, with the same `--req-1…4` hues, name requests in every serving lesson (`batching` and `paged-attention` use the A–D table from `math/paging.js`).

## 1. Learning objective
After this page you can trace one request from Enter to its last token: router, queue, scheduler, prefill (all prompt tokens at once, frame 4), decode (one token per step, frame 6), and name the two latencies a user feels, TTFT and TPOT (frames 5 and 7). You can also say why your request shares each GPU step with many others (frame 9), and point to the lesson in this track that speeds up each stop (frame 10).

## 2. Misconceptions to correct
- **Misconception:** "The model reads my prompt one token at a time, the way it writes the answer." → **Reality:** prefill pushes every prompt token through the model in one pass and stores all their keys and values at once; only the answer is produced one token per step. Corrected by frame 4 and frame 6 side by side. (04 §1.1)
- **Misconception:** "The answer is computed first and then sent." → **Reality:** each output token is sent as soon as it is picked, so the first word arrives after prefill and the rest stream at a steady beat. Corrected by frames 5 and 7. (04 §1.2)
- **Misconception:** "A long prompt is what makes a reply slow." → **Reality:** for chat-sized prompts, total time is dominated by the answer's length: in the toy's default, a 2,000-token prompt costs 0.14 s and a 500-token answer 7.3 s. Prompt length mostly sets TTFT. Corrected by try-this 1. (04 §1.2; numbers from `requestTimeline`)
- **Misconception:** "Each user gets a GPU, or a copy of the model, to themselves." → **Reality:** a GPU advances many requests in the same step, and one read of the weights serves all of them. Corrected by frame 9. (04 §0, §2.1)

## 3. Hook and intuition (final wording)
**Hook:** What happens between pressing Enter and the first word appearing, and why does the rest of the answer arrive at a steady beat instead of all at once?

Your text is cut into tokens and sent as one request. A router picks one of many identical copies of the model (replicas), and inside that replica a scheduler decides when your request joins the GPU's work. Then the model reads your whole prompt in a single pass. This is prefill: every prompt token flows through every layer together, and each one leaves its key and value in the KV cache. At the end of that pass the model picks the first word of the answer. The time until that word reaches you is TTFT, time to first token.

After that, the model can only move one token at a time. It feeds the newest token back in, adds one key and value to the cache, and picks the next token. This is decode, and it is why the answer streams: each step makes exactly one more token for you, and steps come at a steady pace, the time per output token (TPOT). A 500-token answer needs about 500 steps, so for most chats the answer, not the prompt, decides how long you wait in total.

The GPU is not working for you alone. Every decode step re-reads all of the model's weights, and the same read serves every request in the running batch, so a server keeps many requests in flight at once. That is the first of many tricks; the rest of this track takes the journey one stop at a time.

## 4. Visual metaphor
A left-to-right pipeline across the top half of the stage: `block` "you" → `block` "router" → three `block`s "replica 1–3" (the chosen one `active`, the others `dim`) → inside the chosen replica, `block` "scheduler" and a `gpu` with a `kvStack` beside it. The bottom half is a time axis with request A's `request` bar: a prefill segment, then decode ticks. The request's tokens are `token` chips that travel along `flow` arrows (carry `token`); new K/V tiles appear in the `kvStack`. The followed request A wears the serving accent frame in every frame (chip group, bar, and its cache tiles).

Layout for the 580 × 366 stage: pipeline row y ≈ 20–150 (five blocks of 80 × 36 with 20 px gaps fit in 500 px); `kvStack` (tile 14) to the right of the GPU, at most 6 tiles (≈ 96 px); timeline row y ≈ 220–330. At 400 px the pipeline wraps to two rows and the timeline stays full width.

Terms introduced (one per frame, defined on screen): request (1), replica (2), scheduler / queue (3), prefill (4), TTFT (5), decode step (6), TPOT (7), end token (8), batch (9). Terms assumed from `kv-cache`: token, key and value, KV cache, "decoding re-reads the past".

Glyphs used (from spec §5.1 and the built library): `token`, `block`, `gpu`, `kvStack`, `request`, `flow` (carry `token` and `kv`).

Plain text labels (not glyphs): "toy prompt: 3 tokens; real prompts are often thousands" (frame 1); "picks = samples from the model's probabilities (`sampling`)" (frame 4); "TTFT" and "TPOT" brackets on the time axis (frames 5, 7); "kept for reuse? see `prefix-caching`" (frame 8); lesson slugs on the map (frame 10).

New glyphs proposed:
- **`request` gains an `owner` option** (`request(parent, { …, owner })`): sets `data-req = requestSlot(owner)` so the bar takes the `--req-1…4` hue that matches the printed letter, exactly as `blockPool` slots do. Why: the dispatch asks every serving lesson to pair the request glyph with its letter and hue; today `request` has no hue parameter. Reused by every serving lesson that draws a request bar (`batching`, `prefill-decode`, `disaggregation`, `speculative-decoding`). The `batching` storyboard adds one more option to the same proposal (`idle`).

## 5. Animation script
| # | On screen | What moves | Caption (final wording) | Numbers shown |
|---|---|---|---|---|
| 1 | `block` "you" at the left. Three `token` chips `The` `cat` `sat` (positions 1–3) appear under it, grouped with the letter chip `A` (accent frame). Plain label: "toy prompt: 3 tokens; real prompts are often thousands". | Chips type in one by one, then the group slides into a `flow` toward the router. | You press Enter. Your text is cut into tokens, three here, and sent to the provider as one request, which this track calls A. | prompt 3 tokens · positions 1–3 |
| 2 | `block` "router" lights; three `block`s "replica 1", "replica 2", "replica 3" to its right, each a copy of the same model. | A `flow` dot runs router → replica 2; replica 2 turns `active`, the others `dim`. | A router sends the request to one of several identical copies of the model, called replicas. Which one it picks matters later, when some copies remember earlier prompts. | 3 replicas (a stand-in; providers run many) |
| 3 | Inside replica 2: `block` "scheduler" and a short queue lane holding A's `request` bar (no segments yet). The `gpu` beside it shows a memory fill. | A's bar waits one beat in the lane, then slides onto the GPU's time axis. | Inside the replica, a scheduler decides which requests join the GPU's next step. When the GPU is busy, a request waits in a queue. | queue wait: 0 steps here (toy) |
| 4 | **Prefill.** The three chips enter the `gpu` together. The `kvStack` gains 3 K/V tiles at once. A new chip `down` (position 4) leaves the GPU. Plain label: "picks = samples from the model's probabilities (`sampling`)". A's `request` bar draws its prefill segment. | Three chips move in parallel; three tiles appear in the same instant; `down` emerges. | Prefill: the model reads all three prompt tokens in one pass and stores their keys and values. At the end of that pass it picks the first answer token. | 3 tokens in, 3 K/V tiles written, 1 token out |
| 5 | `down` travels back along a `flow` to "you". On the time axis a bracket labeled `TTFT` spans from arrival to the end of the prefill segment. | The chip arrives; the bracket draws left to right. | The first token streams back right away. The wait so far is the time to first token, TTFT: time in the queue plus prefill. | toy default (§6): TTFT 0.141 s for a 2,000-token prompt |
| 6 | **Decode step.** `down` re-enters the GPU alone. The `kvStack` gains its 4th tile; chip `.` (position 5) leaves. A small text mark beside the GPU: "reads all weights + 4 cached tokens". | One chip in, one tile appears, one chip out. The bar gains its first decode tick. | Decode: the newest token goes back in, adds its key and value to the cache, and the model picks one more. Every step re-reads all the weights. | step 1: 1 token in, 1 tile, 1 token out · cache 4 tokens |
| 7 | Steps repeat: `.` goes in, the end token comes out; ticks land at equal spacing on the axis. A bracket `TPOT` spans two adjacent ticks. | Ticks appear at a steady rhythm; the bracket draws between two of them. | Tokens arrive at a steady beat, one per decode step. The gap between two of them is the time per output token, TPOT. | toy default: TPOT 14.7 ms (68 tokens/s) |
| 8 | An end chip `⟨end⟩` arrives at "you". A's 5 K/V tiles fade to faint; a text label beside them: "kept for reuse? see `prefix-caching`". | Tiles fade; the request bar's last tick lands. | An end token, or a length limit, stops the request. Its KV cache is freed for others, unless the server keeps it in case the next prompt starts the same way. | A: 3 prompt + 2 generated tokens in the cache at the end |
| 9 | Zoom out on the GPU's time axis: four `request` bars A, B, C, D (hues 1–4, letters printed) overlap in time; a vertical step marker sweeps across, crossing every running bar at once. | The step marker moves one step; every bar it crosses gains one tick in the same instant. | Zoom out: the GPU never serves you alone. Each step advances every running request by one token, so one read of the weights serves all of them, the batch. | step 3: 3 requests advance together (A–D from `batching`'s table) |
| 10 | The pipeline from frame 1 again, every stop tagged with a lesson slug in plain text: router → `prefix-caching`; scheduler → `batching`; prefill and decode → `prefill-decode`; KV cache → `paged-attention`, `prefix-caching`; decode loop → `speculative-decoding`; weights → `quantization`; many GPUs → `disaggregation`; all of it → `serving-calculator`. | Tags fade in in that order along the pipeline. | Each lesson in this track speeds up one stop on this journey. Next: why prefill and decode behave so differently on a GPU. | 8 lesson tags |

Determinism: every frame is a pure function of (step, progress). Reduced motion shows each frame's end state. Caption check (rlvr-grpo's counter run on this file): all captions ≤ 30 words and ≤ 2 sentences.

## 6. Toy
"Where does the time go?" One request on a time axis: queue, prefill, then decode ticks. A visible line above the controls: "Prefill and decode speeds are a 70B model on one H200, from `prefill-decode`; real servers vary with load."

**Controls**
| id | Label | Type | Range / values | Default | Presets |
|---|---|---|---|---|---|
| `prompt` | Prompt length | Slider (snapped) | 3, 100, 500, 2,000, 8,000, 20,000, 128,000 tokens | 2,000 | — |
| `output` | Answer length | Slider (snapped) | 2, 50, 200, 500, 1,000, 4,000 tokens | 500 | — |
| `queue` | Time in queue | Slider | 0–5 s, step 0.5 | 0 | — |
| `prefillRate` | Prefill speed | Readout chip (fixed) | 14,136 tokens/s | 14,136 | "70B on one H200, FP8 (ceiling)" |
| `decodeRate` | Decode speed per user | Preset chips | 67.9 tok/s "alone on the GPU" · 34.2 tok/s "sharing with 104 others" · 27 tok/s "a busy 2026 server (measured)" | 67.9 | first two from `prefill-decode`'s model; the third from `data/serving.json` `inferencex-v4-pro-gb300.interactivity_tok_s_user` (DeepSeek-V4-Pro on GB300, 8K in / 1K out, 2026-05-22) |

**Live outputs**
| Output | Formula / `math/` function | Units / format |
|---|---|---|
| Timeline bar (queue · prefill · decode ticks, one tick per 10 tokens past 200) | `requestTimeline({ queueS, promptTokens, prefillTokPerS, outputTokens, decodeTokPerS })` | drawn to scale |
| TTFT | `.ttftS` = queue + prompt ÷ prefill speed | s, 3 decimals |
| TPOT | `.tpotS` = 1 ÷ decode speed | ms, 1 decimal |
| Total time | `.e2eS` = TTFT + (answer − 1) × TPOT | s, 2 decimals |
| Share of the wait spent after the first token | `.decodeShare` | %, 1 decimal |

Rates behind the chips: `prefillTokPerSecCeiling({ activeParams: 70e9, peakFlops: 1979e12 })` → 14,135.7; `1 / stepTime(llama-3.1-70b FP8, H200, batch 1, context 2,048).timeS` → 67.92; the same at batch 105 → 34.17 (the batch where one H200's memory is full; see `prefill-decode`).

**Try this** (each leads to a named insight)
1. Predict first: which costs more time, a 10× longer prompt or a 10× longer answer? Set the prompt from 2,000 to 20,000: TTFT 0.141 → 1.415 s, total 7.49 → 8.76 s. Reset, then set the answer from 500 to 50: total 7.49 → 0.86 s. → **Insight: the prompt sets the wait for the first token; the answer's length sets the total.** Decode is 98.1% of the default request's time.
2. Switch decode speed from "alone" (67.9) to "busy server" (27): TTFT does not move (0.141 s), total goes 7.49 → 18.62 s. → **Insight: TTFT and TPOT are separate dials.** Sharing the GPU with more users slows each user's stream, not the first token, which is why servers track both and why `batching` and `disaggregation` treat them as two targets.
3. Set the queue to 2 s: TTFT 0.141 → 2.141 s, while TPOT stays 37.0 ms on the busy chip. → **Insight: a queue only delays the start.** Once your request is in the batch, its pace depends on the step time, not on how long it waited.

**`math/serving.js`** (pure, no DOM; owned by the serving track; full signature list in `prefill-decode` §6, which is where the step-time model is taught):
```js
requestTimeline({ queueS = 0, promptTokens, prefillTokPerS, outputTokens, decodeTokPerS })
  → { queueS, prefillS, ttftS, tpotS, e2eS, decodeShare }
  // ttftS = queueS + promptTokens / prefillTokPerS (prefill ends with the first token)
  // e2eS = ttftS + (outputTokens − 1) · tpotS; throws RangeError on non-positive rates or outputTokens < 1
prefillTokPerSecCeiling({ activeParams, peakFlops }) → number   // peakFlops / (2 · activeParams)
```
Worked examples (scratch implementation of these signatures, 2026-10-07):
```
requestTimeline({ promptTokens: 2000, prefillTokPerS: 14135.7, outputTokens: 500, decodeTokPerS: 67.92 })
  → ttftS 0.141, tpotS 0.0147, e2eS 7.49, decodeShare 0.981
promptTokens 20000 → ttftS 1.415, e2eS 8.76, decodeShare 0.839
outputTokens 50    → e2eS 0.86, decodeShare 0.836
decodeTokPerS 34.17 → tpotS 0.0293, e2eS 14.74
decodeTokPerS 27   → tpotS 0.0370, e2eS 18.62;  with queueS 2 → ttftS 2.141, e2eS 20.62
```
Reproducer (run once `math/serving.js` exists):
```
node -e "import('./math/serving.js').then(m => { const pre = m.prefillTokPerSecCeiling({ activeParams: 70e9, peakFlops: 1979e12 }); for (const [p, o, d, q] of [[2000,500,67.92,0],[20000,500,67.92,0],[2000,50,67.92,0],[2000,500,34.17,0],[2000,500,27,0],[2000,500,27,2]]) console.log(p, o, d, q, m.requestTimeline({ queueS: q, promptTokens: p, prefillTokPerS: pre, outputTokens: o, decodeTokPerS: d })) })"
```
Tests to write first: the table above; `e2eS ≥ ttftS`; `decodeShare` in [0, 1); outputTokens = 1 gives `e2eS === ttftS`; inputs not mutated.

## 7. Show me the math
```tex
\htmlClass{hl-ttft}{\text{TTFT}} = t_{\text{queue}} + \frac{\text{prompt tokens}}{\text{prefill speed}}
\qquad
\htmlClass{hl-tpot}{\text{TPOT}} = \frac{1}{\text{decode speed per user}}
```
```tex
t_{\text{total}} = \htmlClass{hl-ttft}{\text{TTFT}} + (\text{answer tokens} - 1)\cdot\htmlClass{hl-tpot}{\text{TPOT}}
```
Shapes: none (scalars). Definitions used across the serving track (one definition each, README lesson 16): TTFT = queue + prefill, ending when the first output token is picked; TPOT = time between two consecutive output tokens of one request = one decode step; tokens/s per user = 1 / TPOT. Color links: `hl-ttft` = the TTFT bracket (frame 5); `hl-tpot` = the TPOT bracket (frame 7).

## 8. In today's models (Oct 2026)
| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| A busy 2026 server, measured: DeepSeek-V4-Pro on GB300 NVL72 at 27 tokens/s per user (ISL 8K / OSL 1K, FP4, disaggregated Dynamo + vLLM, InferenceX, measured 2026-05-22) | `serving.json/inferencex-v4-pro-gb300.interactivity_tok_s_user = 27` *(proposed)* | 04 §1.3 CONFIRMED |
| A 128K-token prompt took 8.6 s to its first token: DeepSeek-R1 on GB300 NVL72 with chunked pipeline-parallel prefill (LMSYS, 2026-02-19) | `serving.json/lmsys-gb300-longctx.ttft_128k_s = 8.6` *(proposed; note names model, hardware, method)* | 04 §1.2, §7.5 CONFIRMED |
| Routers send a request to the replica already holding its prefix: 56.3% of DeepSeek's input tokens hit the KV cache (V3/R1 production, Feb 2025) | `serving.json/deepseek-v3-production.kv_hit_rate_pct = 56.3` *(proposed by `paged-attention`; same key)* | 04 §2.4, §6.3 CONFIRMED |
| The engines inside a replica: vLLM, SGLang and TensorRT-LLM; orchestration by NVIDIA Dynamo (1.0, GA 2026-03-16) or llm-d (CNCF Sandbox 2026-03-12) | `serving.json/engines.names` *(proposed, `reported`)*, `serving.json/dynamo.ga_date = "2026-03-16"` *(proposed, confirmed)*, `serving.json/llm-d.cncf_sandbox_date = "2026-03-12"` *(proposed, reported)* | 04 §7.1, §7.2 |
| Closed providers do not publish their serving stacks; this track uses open engines and published measurements | plain sentence, no key | 04 §10 item 5 |

Not shown: engine market share (04 §10 item 1).

## 9. Takeaways
1. A request goes router → queue → scheduler → GPU; prefill reads the whole prompt in one pass and ends with the first token (TTFT), then decode adds one token per step at a steady TPOT (frames 2–7).
2. For chat-sized prompts, the answer's length sets most of the total time; the prompt mostly sets TTFT (try-this 1).
3. Each decode step advances every request in the batch at once, which is why servers share GPUs and why the rest of this track is about keeping that step fast and full (frames 9–10).

## 10. Next and go deeper
Next: `prefill-decode` · Related: `kv-cache` (what the cache holds), `batching`, `prefix-caching`.

Go deeper (brief 05 §1.1–1.2, 04 §9.2): Lynskey, *LLM Inference Explained* (https://llm-inference-explained.vercel.app), the long-form companion for this whole track · Hugging Face, "Continuous batching from first principles" (https://huggingface.co/blog/continuous_batching).

## 11. Key-frame sketch
Frame 7 end state, desktop width; numbers are the toy defaults from §6.
```text
One request's journey          step 7 / 10   [<] [Play] [>]
[you]->[router]->[replica 2]->[scheduler]->[GPU] K V K V K V K V K V
   ^                                             (5 tiles: The cat sat down .)
   '-- down . <end>  streaming back
time ------------------------------------------------------>
A |#####| . . . . . . . . . . . . . . . . . . . .
  |-TTFT-|-TPOT-|
TTFT 0.141 s   TPOT 14.7 ms   total 7.49 s (2,000 in / 500 out)
```

## 12. Open questions for the reviewer
- **Mixed preset on the busy chip.** The "27 tok/s busy server" chip is a measured DeepSeek-V4-Pro / GB300 number, while the prefill speed stays the Llama-3.1-70B / H200 ceiling. The chip label names its source; if the reviewer prefers one model throughout, drop the chip and keep the two derived ones (try-this 2 then uses 34.2 tok/s: total 14.74 s).
- **`request` `owner` option** (glyph proposal, §4) needs adding to `shared/glyphs/systems.js` before any serving page is built.
- **Data-pass keys** listed in §8 (`inferencex-v4-pro-gb300.*`, `lmsys-gb300-longctx.*`, `engines.*`, `dynamo.*`, `llm-d.*`); `deepseek-v3-production.kv_hit_rate_pct` is shared with `paged-attention`.
