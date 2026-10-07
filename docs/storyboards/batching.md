# Continuous batching (`batching`)

Track: serving · Section: serving · Prereqs: prefill-decode
Status: approved (expert review)
Sources: 04 §1.2, §1.3, §2.1, §2.2, §2.3, §9.1 (slot-timeline and chunked-prefill toys), §9.2; 05 §1.1, §1.2. Beyond the briefs: Gordić, "Inside vLLM" (vllm.ai/blog/2025-09-05-anatomy-of-vllm), read by the storyboard author (Opus 5.5) on 2026-10-07 for one line only: V1 preempts by recompute ("swap preemption was supported in V0"). Expert review (Fable 5.1, 2026-10-07) applied; see §13.

Running example: the four requests A–D of `paged-attention` (`TOY_REQUESTS` in `math/paging.js`), unchanged, so the two lessons line up step for step. Step times come from `prefill-decode`'s running example (Llama-3.1-70B, FP8 weights, one H200), where a step with up to ~200 tokens takes 14.6 ms.

| Request | Arrives | Prompt | Output |
|---|---|---|---|
| A | step 0 | 8 | 4 |
| B | step 0 | 5 | 2 |
| C | step 0 | 10 | 6 |
| D | step 1 | 6 | 3 |

Step rule (the same as `paged-attention`'s): the step a request is admitted runs its prefill; each later step is one decode step that adds one token; a request with `output` o admitted at step a is done at the end of step a + o, and its seat is free from the next step. Steps count from 0; this page has no memory addresses.

## 1. Learning objective
After this page you can explain why static batching leaves seats idle and makes new requests wait for the slowest member (frames 2–4), how continuous batching refills a seat the step after it frees (frames 5–6), why one long prompt can stall everyone's stream and how chunked prefill bounds that stall (frames 8–9), and why the number of seats was really a memory limit in 2023 engines, the problem `paged-attention` solves (frame 10).

## 2. Misconceptions to correct
- **Misconception:** "A batch is a group of requests that start and finish together." → **Reality:** that is static batching. Continuous (iteration-level) batching re-forms the batch before every step: finished requests leave and waiting ones join, so with 3 seats D starts at step 3 instead of step 7 and the run ends at step 6 instead of 10. Corrected by frames 5–6 and try-this 1. (04 §2.1)
- **Misconception:** "The batch has to wait for the longest answer, so one long reply slows everyone." → **Reality:** only in static batching. Make C's answer 10 tokens long: static D waits until step 11, continuous D still runs steps 3–6. Corrected by try-this 2. (04 §2.1)
- **Misconception:** "Prefill and decode run in separate steps." → **Reality:** a step can mix D's prompt tokens with A and C's next tokens (frame 7). That is also what makes a long prompt dangerous: a 4,096-token prefill turns one step into 290 ms and stalls every stream in it (frame 8). Chunked prefill caps each step's tokens (frame 9). (04 §2.2)
- **Misconception:** "More seats is always better." → **Reality:** a seat only helps when someone is waiting for it, and in 2023 engines each seat cost a full reserved strip of KV memory. With 4 seats D starts at step 1, but seats are busy 68% of the time instead of 90%. Corrected by frame 10 and try-this 3. (04 §2.1, §3.2)

## 3. Hook and intuition (final wording)
**Hook:** If one read of the weights can serve dozens of users at once, why did early servers keep making new requests wait while seats sat empty?

In `prefill-decode` you saw that a decode step costs about the same whether it serves one user or dozens: the GPU reads the weights once and every user in the batch gets a token. So the server's job is to keep that batch full. The simplest way, static batching, gathers a group of requests, runs them together, and starts the next group only when the whole group is done. Answers have different lengths, though. When the short ones finish, their seats stay held and empty until the longest one is done, and requests that arrived in the meantime wait outside. Static batches also pad: each prompt is stretched to the longest one's length (adding a 100-token prompt to 8 running requests wastes 693 pad tokens in Hugging Face's worked example).

Continuous batching, introduced by Orca in 2022, fixes this by deciding the batch one step at a time. Before every step the scheduler drops finished requests and seats waiting ones, so a seat is reused the step after it frees, and sequences of different lengths sit side by side without padding (ragged batching). The major engines (vLLM, SGLang, TensorRT-LLM) all schedule this way; vLLM's core loop is literally schedule, run the model, sample, update, repeat.

Mixing requests in one step has a catch. A new request's whole prompt is prefilled in the step it joins, and a long prompt makes that step long. Everyone else's next token waits for it. Chunked prefill gives each step a token budget: running decodes go first, and the prompt fills the rest of the budget a slice at a time, so no step gets much longer than the budget allows. The price is that the long prompt's own first token comes a little later.

The scheduler has a few more knobs: a cap on seats, the token budget, priorities, and preemption, which pauses a request and recomputes it later when memory runs out (vLLM V1 recomputes; older versions could also swap to CPU memory). The price of continuous batching is a scheduling pass before every step and the chance that an admitted request runs out of memory mid-reply, which is what preemption is for. Memory is the knob this page holds fixed. In 2023 engines the seat cap was really a memory limit, because each seat reserved KV for the longest possible answer; you'll see how paging lifted it in `paged-attention`.

## 4. Visual metaphor
The stage is a **seat timeline**: three horizontal rows ("seat 1–3"), x = step (0–10, 40 px per step, so 11 steps fit in 440 px). Each request is a `request` bar on the row of the seat it holds: a one-step prefill segment at its admission step, then one decode tick per step, colored by its `--req` hue with its letter printed. Seat-steps that are **held but idle** are hatched, using the same hatch as `paged-attention`'s reserved-but-empty KV slots, because they mean the same thing: reserved and not used. Seats that are free and unheld are blank. Waiting requests sit in a "queue" box at the left with a `waiting` text badge. Two lanes ("static", "continuous") share the step axis in frames 6 and 10. The followed request is **D** (accent frame on its bar and queue chip in every frame).

In frames 8–9 the x-axis switches from steps to milliseconds (a visible axis label changes from "step" to "ms"), because steps no longer last the same time: a step's width is its `stepTime` on the running example.

Terms introduced (one per frame): seat (1), static batching (2), idle seat (3), seat utilization (4), continuous batching (5), tokens per step (6), mixed step (7), stall (8), chunked prefill / token budget (9), seat cap (10). Terms assumed: from `prefill-decode`: decode step, prefill, step time, memory-bound, TTFT/TPOT; from `serving-overview`: scheduler, queue, batch.

Glyphs used: `request` (with the `owner` hue option proposed in `serving-overview`), `token` (queue chips), `block` ("scheduler"), `flow` (carry `token`, queue → seat), `gpu` (frame 7).

Plain text labels: "seat = a place in the running batch" (frame 1); "waiting" badge; "1 step ≈ 14.6 ms here (Llama-3.1-70B, FP8, one H200)" (frame 1); "What if?" branch label (frames 8–9); axis label "step" / "ms"; "memory is not modeled on this page" (frame 1, under the seats).

New glyphs proposed:
- **`request` gains an `idle` option** (`request(parent, { …, owner, idle })`): `idle` hatched units drawn after the last decode tick, meaning "seat still held, doing nothing". Why: static batching's whole lesson is the held-but-empty seat, and the hatch reuses `blockPool`'s reserved-empty meaning. Extends the `owner` proposal from `serving-overview`; reused by `disaggregation` (frame 1).

## 5. Animation script
Numbers: `simulateStatic` / `simulateContinuous` (seats 3), and `scheduleTokens` + `stepTime` for frames 7–9 (§6).

| # | On screen | What moves | Caption (final wording) | Numbers shown |
|---|---|---|---|---|
| 1 | Three empty seat rows; queue box at the left with chips A, B, C, and D in its hue with the plain label "arrives at step 1". The request table above. Labels: "seat = a place in the running batch"; "1 step ≈ 14.6 ms here"; "memory is not modeled on this page, which is why `paged-attention` comes next". | The table rows fade in; seat rows draw. | Four requests and a GPU that runs at most three at a time. Each running request holds a seat, and every step each seated request gets one more token. | A 8/4 · B 5/2 · C 10/6 · D 6/3 (prompt/output) · D arrives at step 1 · 3 seats |
| 2 | **Static lane.** A, B, C move from the queue onto seats 1–3 at step 0 (prefill segments), then decode ticks grow step by step. | Chips slide to seats; bars extend one tick per step. | Static batching: A, B and C start together, and this batch runs until its longest member, C, is done. | A done step 4 · B done step 2 · C done step 6 |
| 3 | B's row hatches from step 3 to 6, A's from 5 to 6. D's chip (framed) sits in the queue with a `waiting` badge from step 1 on. | Hatch fills in step by step; the badge pulses once per step. | B is done after step 2 and A after step 4, but their seats stay held until C finishes. D arrived at step 1 and waits outside. | idle seat-steps: B 4, A 2 · D waiting steps 1–6 |
| 4 | D takes seat 1 at step 7, runs alone to step 10; seats 2–3 blank from step 7. A readout under the lane: busy seat-steps / all seat-steps. | D's bar draws steps 7–10; the readout counts up. | D finally starts at step 7, alone, and is done at step 10. Over the whole run, seats were busy only 58% of the time. | D 7 → 10 · busy 19 of 33 seat-steps = 57.6% · 15 tokens in 11 steps |
| 5 | **Continuous lane** below the static one (static dims). Same start. At the boundary between step 2 and 3, a `block` "scheduler" flashes; B's seat clears; D (framed) slides from the queue into seat 2 and prefills at step 3. | The scheduler flashes before each step; at step 3 D's chip moves into B's old seat. | Continuous batching runs the scheduler before every step. B leaves after step 2, so D takes its seat at step 3 instead of waiting for the whole batch. | D admitted step 3 (B's seat) · done step 6 |
| 6 | Both lanes, full height, end to end. Readouts beside each: last step, seat utilization, tokens per step. | The continuous lane's end marker settles at step 6; the static lane's at step 10. | Same four requests: all done by step 6 instead of 10, with seats busy 90% of the time. The GPU did the same work in fewer steps. | static: last step 10, 57.6%, 1.36 tokens/step · continuous: last step 6, 90.5%, 2.14 tokens/step |
| 7 | Zoom on step 3 of the continuous lane: a `gpu` with chips D×6 (prompt) and A, C (one decode token each) entering together. Step time readout. | The eight chips enter the GPU in one pass. | At step 3 the GPU runs D's six-token prefill and A's and C's next tokens in one pass. Prefill and decode can share a step. | 6 + 2 = 8 tokens · 14.6 ms (memory-bound, as in `prefill-decode`) |
| 8 | Branch label "What if D's prompt were 4,096 tokens?" over the continuous lane; the axis label switches to "ms", with a plain note "step 0 (23 prompt tokens) 14.7 ms, other short steps 14.6 ms". Step 3 widens to 290 ms; A's and C's next ticks land after it. | Step 3's block stretches from 14.6 to 290 ms; A and C's tick markers slide right with it. | What if D's prompt were 4,096 tokens? Step 3 now takes 290 ms, so A and C wait twenty times longer than usual for their next token. | step 3: 4,098 tokens, 289.9 ms · A done at 348.4 ms · C done at 377.5 ms · D's first token at 333.8 ms |
| 9 | Same branch, budget 512: D's prompt split into slices 510, 510, 511, 511, 512, 512, 512, 512, 6 across steps 3–11; A and C ride along until they finish. Readouts compared with frame 8. | The 290 ms block breaks into nine short blocks; A's and C's ticks slide left to their new times. | Chunked prefill caps each step at 512 tokens and feeds D's prompt in slices. A and C never wait over 36 ms; D's first token comes 14 ms later. | longest step 36.2 ms · A done 116.3 ms (was 348.4) · C done 188.7 ms (was 377.5) · D's first token 348.2 ms (was 333.8) |
| 10 | Back to the main timeline (branch label gone, axis "step"). A third lane, "continuous, 4 seats": D starts at step 1 in a fourth seat and is done at step 4. Text label: "2023 engines reserved KV for the longest answer per seat: see `paged-attention`". | The fourth seat row slides in; D's bar moves from step 3 to step 1. | In 2023 engines each seat reserved memory for the longest possible answer, which kept seats few. `paged-attention` makes room for a fourth, and D starts at step 1. | 3 seats: D 3 → 6 · 4 seats: D 1 → 4 (matches `paged-attention`'s before/after lanes) |

Determinism: every frame is a pure function of (step, progress); the simulators recompute from step 0. Reduced motion shows each frame's end state. Caption check: rlvr-grpo's counter, all ≤ 30 words and ≤ 2 sentences.

## 6. Toy
"Seat timeline." Static and continuous lanes, always both visible (stacked at 400 px). A visible line above the controls: "Four hand-picked requests (the same as `paged-attention`). Steps are timed with Llama-3.1-70B on one H200; memory is not modeled here."

**Controls**
| id | Label | Type | Range / values | Default | Presets |
|---|---|---|---|---|---|
| `seats` | Seats (max requests running at once) | Slider | 2, 3, 4 | 3 | — |
| `cOutput` | C's answer length | Slider | 2–10 tokens, step 1 | 6 | — |
| `dPrompt` | D's prompt | Preset chips | 6 tokens · 4,096 tokens | 6 | — |
| `budget` | Token budget per step (chunked prefill) | Preset chips | off · 2,048 · 512 | off | — |

`budget` affects the continuous lane only and is shown only when D's prompt is 4,096 (with 6 tokens no step comes near any budget). With the 4,096 chip the axis switches to milliseconds.

**Live outputs**
| Output | Formula / `math/` function | Units / format |
|---|---|---|
| Both lanes: bars per seat, idle hatch, queue | `simulateStatic({ requests, seats })`, `simulateContinuous({ requests, seats })` | drawn |
| Per request: admitted, done, waited | `live[]` | step numbers |
| Last step, seat utilization, tokens per step | `.lastStep`, `.utilizationPct` (busy seat-steps ÷ seats × steps), `.tokensPerStep` (output tokens ÷ steps) | step; %; 2 decimals |
| Step durations, request done times, longest wait between A's tokens, D's first token | `scheduleTokens({ requests, seats, budget })` → per-step tokens; each step `stepTime({ …running example…, tokens, seqs: decoders, context: 16 })` | ms, 1 decimal |

**Try this** (each leads to a named insight)
1. Defaults (3 seats). Compare the lanes: static ends at step 10 with seats busy 57.6% of the time; continuous ends at step 6 with 90.5%. → **Insight: refilling a seat the step after it frees is the whole trick.** Same work, 4 fewer steps, and D waits 2 steps instead of 6.
2. Predict first: what happens to D if C's answer grows from 6 to 10 tokens? Static: D now waits until step 11 and is done at 14 (seats busy 51.1%). Continuous: D still runs steps 3–6. → **Insight: in a static batch the longest answer sets everyone's schedule; in a continuous one it only sets its own.**
3. Set seats to 4. Continuous: D starts at step 1 and is done at step 4, but seat utilization drops to 67.9%, because nobody else is waiting for that fourth seat. Static with 4 seats: D still waits until step 7. → **Insight: a seat helps only if someone is waiting, and only continuous batching can hand it out mid-run.** In 2023 engines each seat also cost a full reserved strip of KV memory, which is why seats were scarce (`paged-attention`).
4. Pick D's 4,096-token prompt. Budget off: one 289.9 ms step, A done at 348.4 ms. Budget 2,048: longest step 144.9 ms, A done at 333.6 ms. Budget 512: longest step 36.2 ms, A done at 116.3 ms, C at 188.7 ms, while D's first token moves only from 333.8 to 348.2 ms. → **Insight: chunked prefill trades a little of the long prompt's TTFT for a bounded TPOT for everyone else.** The smaller the budget, the smoother the streams, until steps become too small to keep the GPU busy (not modeled here; `prefill-decode`'s 217-token crossover is the floor).

**`math/batching.js`** (pure, no DOM; `TOY_REQUESTS` is imported from `math/paging.js`, not redefined; `stepTime` from `math/serving.js`):
```js
simulateStatic({ requests, seats })
  → { kind, seats, live: [{ id, arrives, admitted, finishes, waited }], lastStep, steps, seatSteps, busySeatSteps,
      utilizationPct, outputTokens, tokensPerStep }
  // a batch starts only when the previous batch is entirely done; it takes up to `seats` arrived requests in arrival order
simulateContinuous({ requests, seats }) → same shape
  // before every step: drop requests done at the end of the previous step, then admit arrived requests in arrival order while seats are free
scheduleTokens({ requests, seats, budget = Infinity })
  → [{ step, decode: [ids], prefill: [{ id, tokens }], tokens }]
  // continuous; each step: running decodes first (one token each), then waiting prompt tokens up to budget − decodes;
  // a request decodes from the step after its last prefill slice
paddingWaste(newPromptTokens, runningRequests) → number          // (n − 1)(B − 1), Hugging Face's padded-batching example
```
Busy seat-steps count every step from a request's admission to its last step inclusive (prefill and decode).

Worked examples (scratch implementation, 2026-10-07; `R` = `TOY_REQUESTS`):
```
simulateStatic({ requests: R, seats: 3 })      → A 0–4, B 0–2, C 0–6, D 7–10; lastStep 10; busy 19 of 33 = 57.6%; 1.36 tokens/step
simulateContinuous({ requests: R, seats: 3 })  → A 0–4, B 0–2, C 0–6, D 3–6;  lastStep 6;  busy 19 of 21 = 90.5%; 2.14 tokens/step
seats 4: static D 7–10 (43.2%) · continuous D 1–4, lastStep 6 (67.9%)
seats 2: static A 0–4, B 0–2, C 5–11, D 5–8 (79.2%) · continuous C 3–9, D 5–8, lastStep 9 (95.0%)
C output 10, seats 3: static D 11–14, lastStep 14 (51.1%) · continuous D 3–6, lastStep 10 (69.7%)
paddingWaste(100, 8) → 693

D prompt 4,096, seats 3, step times from stepTime (Llama-3.1-70B FP8 on H200, context 16):
  budget off:   step 3 = 4,098 tokens, 289.9 ms; D's prefill ends 333.8 ms; A done 348.4; C done 377.5; longest A/C gap 289.9
  budget 2048:  steps 3–4 = 2,048 tokens, 144.9 ms each; D's prefill ends 348.2; A done 333.6; C 362.8
  budget 512:   slices 510, 510, 511, 511, 512, 512, 512, 512, 6 (steps 3–11); steps 36.2 ms; D's prefill ends 348.2; A done 116.3; C 188.7
  short prompts (defaults): step 0 (23 prompt tokens) 14.7 ms, every other step 14.6 ms
```
Reproducer (run once `math/batching.js` exists):
```
node -e "Promise.all([import('./math/batching.js'), import('./math/paging.js'), import('./math/serving.js')]).then(([b, p, s]) => { for (const seats of [2, 3, 4]) console.log(b.simulateStatic({ requests: p.TOY_REQUESTS, seats }), b.simulateContinuous({ requests: p.TOY_REQUESTS, seats })); const long = p.TOY_REQUESTS.map(r => r.id === 'D' ? { ...r, prompt: 4096 } : r); for (const budget of [Infinity, 2048, 512]) { let t = 0; for (const st of b.scheduleTokens({ requests: long, seats: 3, budget })) { t += s.stepTime({ activeParamsPerGpu: 70e9, weightBytesPerGpu: 70e9, dModel: 8192, actBytesPerElem: 1, kvBytesPerToken: 327680, peakTflops: 1979, bandwidthTBps: 4.8, tokens: st.tokens, seqs: st.decode.length, context: 16 }).timeS * 1e3; console.log(budget, st.step, st.tokens, st.decode.join(''), t.toFixed(1)); } } })"
```
Tests to write first: the table above; `simulateContinuous` with seats ≥ 4 admits every request on arrival; utilization ≤ 100%; continuous `lastStep` ≤ static `lastStep` for every control setting; `scheduleTokens` never exceeds the budget and processes every prompt token exactly once; with `budget = Infinity` and short prompts it reproduces `simulateContinuous`'s admitted/done steps; inputs not mutated (`Object.freeze` the request list).

## 7. Show me the math
```tex
\text{static: start}_{\text{next batch}} = \max_{r \in \text{batch}} \left(a_r + o_r\right) + 1
\qquad
\text{continuous: seat free at step } a_r + o_r + 1
```
```tex
\text{seat utilization} = \frac{\sum_r (o_r + 1)}{\text{seats}\times\text{steps}}
\qquad
\text{tokens per step} = \frac{\sum_r o_r}{\text{steps}}
\qquad
\text{pad waste} = (n-1)(B-1)
```
```tex
\text{chunked prefill: } \text{slice}_s = \min\!\left(\text{prompt left},\ \text{budget} - \#\text{decodes}_s\right),
\qquad t_s = \text{stepTime}(\#\text{decodes}_s + \text{slice}_s)
```
Shapes: none. `a_r` = admission step, `o_r` = output tokens (decode steps). Color links: `o_r + 1` (a request's held seat-steps) in its `--req` hue; the hatched idle term (`seats × steps − Σ`) in the reserved-empty hatch.

## 8. In today's models (Oct 2026)
| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| Padded static batching: adding a 100-token prompt to 8 running requests wastes 693 pad tokens (Hugging Face, Nov 2025) | `serving.json/hf-continuous-batching.pad_waste_example = 693` *(proposed; note "B = 8, n = 100")* | 04 §2.1 CONFIRMED |
| Continuous (iteration-level) batching: Orca, OSDI 2022; finished requests leave, waiting ones join every step | `serving.json/orca.venue = "OSDI 2022"` *(proposed)*; Orca's throughput gain is not shown (the brief marks it "prior, check number") | 04 §2.1 |
| vLLM's engine loop: schedule → run model → sample → update, every step | `serving.json/vllm.engine_loop` *(proposed, confirmed)* | 04 §2.1 CONFIRMED |
| Chunked prefill (Sarathi-Serve, OSDI 2024; the year is printed on the page): up to 2.6× serving throughput within SLO for Mistral-7B on one A100, up to 6.9× for Falcon-180B on 8 A100s, vs Orca and vLLM | `serving.json/sarathi-serve.gain_mistral7b_a100 = 2.6`, `.gain_falcon180b_8xa100 = 6.9` *(proposed)* | 04 §2.2 CONFIRMED (abstract) |
| Engine knobs: per-step token budget (`max_num_batched_tokens`), seat cap (`max_num_seqs`), FCFS by default, priorities, preemption; vLLM V1 preempts by recompute | `serving.json/vllm.scheduler_knobs` *(proposed, `reported`: brief marks it general knowledge)*; `serving.json/vllm.v1_preemption = "recompute"` *(proposed; source Gordić post, read 2026-10-07)* | 04 §2.3 *(prior)*; beyond the brief (header) |
| Chunks re-read the KV of earlier chunks; minimizing chunk overhead mattered for 64K-token prefill batches on GB200 (vLLM, 2026-02-03) | `serving.json/vllm-gb200-dsr1.note_chunking` *(proposed)* | 04 §2.2 CONFIRMED |

## 9. Takeaways
1. Static batching holds seats until the longest member is done and makes newcomers wait; continuous batching re-forms the batch every step, so a freed seat is reused at once (frames 2–6, try-this 1–2).
2. Prefill and decode can share a step, so a long prompt can stall everyone's stream; chunked prefill caps each step's tokens and bounds the stall at a small cost to that prompt's TTFT (frames 7–9, try-this 4).
3. A seat only helps if someone is waiting, and in 2023 engines each seat reserved KV for the longest answer, which kept seats scarce; `paged-attention` lifts that limit (frame 10, try-this 3).

## 10. Next and go deeper
Next: `paged-attention`, `disaggregation` · Related: `prefill-decode` (why a shared step is nearly free), `prefix-caching` (where KV-aware routing is covered).

Go deeper (brief 04 §9.2, 05 §1.2): Hugging Face, "Continuous batching from first principles" (https://huggingface.co/blog/continuous_batching) · Ashwin Giridharan, interactive vLLM guide with a token-by-token scheduler walkthrough (https://ashwing.github.io/vllm-guide/; the brief has not opened the guide itself) · Lynskey, *LLM Inference Explained* (https://llm-inference-explained.vercel.app).

## 11. Key-frame sketch
Frame 6 end state, desktop width; rows printed from `simulateStatic` / `simulateContinuous` (seats 3). `P` = prefill step, `#` = decode step, `/` = held but idle, `.` = free seat.
```text
Continuous batching             step 6 / 10   [<] [Play] [>]
step        0 1 2 3 4 5 6 7 8 9 10
--- static (3 seats) -----------------------------
seat 1   A  P # # # # / / D P # # #
seat 2   B  P # # / / / / . . . .
seat 3   C  P # # # # # # . . . .
queue: D waits steps 1-6  | busy 19/33 = 57.6% | last step 10
--- continuous (3 seats) -------------------------
seat 1   A  P # # # # . .
seat 2   B  P # # D P # # #
seat 3   C  P # # # # # #
D joins at step 3         | busy 19/21 = 90.5% | last step 6
```

## 12. Open questions for the reviewer
**Handoff answers**
- **`paged-attention` §12, `pool exhausted` contract.** Settled: neither `math/paging.js` nor `math/batching.js` simulates preemption. This page names it (§3 ¶4, §8) and states the V1 recompute behaviour from a primary source; `paged-attention`'s throw contract stands.
- **Same requests, same numbers.** With 3 seats, the continuous lane reproduces `paged-attention`'s before-lane (D 3 → 6), and with 4 seats its after-lane (D 1 → 4). The two pages tell one causal chain: seats limit the batch (here), memory limits the seats (`paged-attention`). `paged-attention` uses "slot" for one KV token; this page says "seat" for a place in the batch, so the words never collide.

**Graph and scope**
- **KV-aware routing (settled):** taught in `prefix-caching` (frame 9 there), since routing by KV state needs prefix caching; this page links to it. The spec §3.3 line moves accordingly (main session).

**Data-pass keys**
- `serving.json/hf-continuous-batching.*`, `orca.*`, `vllm.engine_loop`, `vllm.scheduler_knobs`, `vllm.v1_preemption`, `sarathi-serve.*`, `vllm-gb200-dsr1.note_chunking` as in §8.

**Judgment calls**
- **Memory-blind seats.** The toy's seats ignore KV memory on purpose (a visible line says so), so frame 10 can hand the memory story to `paged-attention`. The 4,096-token branch would not fit `paged-attention`'s 48-slot pool; it is labeled a "What if?" and timed in ms.
- **Chunk overheads not modeled.** Each chunk re-reads the KV of earlier chunks (brief 04 §2.2); at 4,096 tokens on the running example that is under 1.4 GB per step and is left out of `stepTime`'s `seqs`. The insight (bounded stall, slightly later TTFT) survives with it included.

## 13. Reviewer rulings (expert review, Fable 5.1, 2026-10-07)
- **Settled:** `batching` and `paged-attention` share `TOY_REQUESTS` and the step rule; the 3-seat and 4-seat continuous lanes reproduce paged's two lanes (reviewer verified).
- **Settled:** preemption is named here (vLLM V1 recomputes) and simulated nowhere; paged's throw contract stands.
- **Settled:** KV-aware routing is taught in `prefix-caching`; this page links to it.
- Applied: the engines sentence names vLLM, SGLang and TensorRT-LLM instead of "every major engine" (lesson 7); continuous batching's own cost (a scheduling pass per step, possible mid-reply memory exhaustion) is named (lesson 14); the ms axis notes step 0's 14.7 ms; try-this 4 marks the too-small-step limit as not modeled; Sarathi-Serve carries its year; frame 1's memory note points to `paged-attention`; D's "arrives at step 1" is a plain label, not greyed (grey reads as `dim`).

