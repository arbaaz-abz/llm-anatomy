# Speculative decoding (`speculative-decoding`)

Track: serving · Section: serving · Prereqs: prefill-decode, sampling
Status: approved (expert review)
Sources: 04 §1.1, §4.1, §4.2, §4.3, §9.1 (speculative-decoding toy), §10 item 8; 05 §1.1. Mechanism (draft, verify, rejection sampling) from first principles and the 2022–2023 papers the brief lists as *(prior)*; no paper speedups are printed except those the brief marks CONFIRMED. Expert review (Fable 5.1, 2026-10-07) applied; see §13.

Running example: the prompt `The cat sat` from `serving-overview`'s journey ("your request" there). The target is `prefill-decode`'s running example (Llama-3.1-70B, FP8 weights, one H200; each user has 1,024 tokens of context on this page). Draft guesses, probabilities and the drafter's cost are hand-picked stand-ins, and a visible line says so.

## 1. Learning objective
After this page you can explain how a cheap drafter's guesses let the big model emit several tokens per step (frames 2–5), why the output is exactly what the big model alone would sample (frame 6), how many tokens a round yields for a given acceptance rate and guess count (frame 7, try-this 1), and why the gain shrinks, and can turn into a loss, as the batch grows (frames 8–9, try-this 2).

## 2. Misconceptions to correct
- **Misconception:** "Speculative decoding trades quality for speed: you get the small model's answers." → **Reality:** every token is checked by the target and kept with a probability set by both models; a rejected position is resampled from the leftover probability. Tokens come out with exactly the target's distribution, because the leftover repays exactly the probability that rejection removed. Corrected by frame 6 and try-this 4. (04 §4.1, CONFIRMED via vLLM: "preserves the verifier model's output distribution exactly")
- **Misconception:** "Checking k guesses costs k target steps." → **Reality:** the target scores all k + 1 positions in one pass, and while decode is memory-bound that pass costs about one step: 14.7 ms for 4 positions and 14.7 ms for 1 at batch 1. Corrected by frame 3. (04 §4.1; `prefill-decode`)
- **Misconception:** "More guesses per round is always faster." → **Reality:** each extra guess is accepted only if all earlier ones were, so returns shrink: at acceptance 0.7, going from 3 to 5 guesses adds 0.41 tokens per round and the speedup peaks around 5 guesses (2.35×), then falls (8 guesses: 2.28×). Corrected by frame 7 and try-this 1. (04 §4.1)
- **Misconception:** "A 2–3× speedup means a server can handle 2–3× more traffic." → **Reality:** it is a per-user latency gain. At large batch the verify pass becomes compute-bound and the gain falls: 2.2× at one user, 2.14× at 64, 1.53× at 128, and with 5 guesses at 211 users it is 0.91 of the plain speed, slower than not speculating. EAGLE-3's own measurement fell from up to 6.5× to 1.38× at batch 64. Corrected by frame 9 and try-this 2. (04 §4.1, §10 item 8)

## 3. Hook and intuition (final wording)
**Hook:** How can a small model's guesses make a big model faster without changing a single word the big model would have written?

In `prefill-decode` you saw that a decode step mostly waits for the weights to arrive from memory: the arithmetic for one token is nearly free. Speculative decoding spends that free arithmetic. A small, fast drafter guesses the next few tokens, here `down` `on` `a`. Then the big model, the target, runs one pass over all of them at once: the guesses are already written down, so it can read them the way prefill reads a prompt. Because the step was waiting on memory anyway, checking four positions costs about the same as producing one.

The target then walks the guesses left to right. A guess survives with a probability that compares the two models: if the target likes the word at least as much as the drafter did, it is always kept; if the drafter was more confident than the target, it is kept only part of the time. At the first failure, the target picks a replacement from the probability the drafter left over, and everything after it is discarded. This rule, rejection sampling, makes the output follow the target's own probabilities exactly. With greedy decoding it reduces to "keep the guess if it is the target's top token". Either way the answer is the target's, only faster.

How much faster depends on how often guesses survive. With a 70% acceptance rate and three guesses, a round yields 2.53 tokens on average, and after paying for the drafter that is about 2.2× on the running example. The catch is the batch. With many users, checking four positions for each of them makes the verify pass large enough to become compute-bound, and the free arithmetic is gone. That is why speculative decoding is a per-user latency tool, strongest for interactive traffic at small batch. In 2026 open-model serving, the drafter is usually part of the model: multi-token prediction (MTP) heads or EAGLE-style heads that read the target's own hidden states, and newer parallel drafters guess all tokens in one pass. Many drafters also guess a small tree of alternatives instead of a single line, and the target checks every branch in the same pass.

## 4. Visual metaphor
A row of `token` chips for the answer so far (`The cat sat`, then the round's tokens). Two `block`s: a small "drafter" (left, narrow) and a large "target" (right). Draft chips use the library's `token` state `draft` (dashed border, defined once and shared with `sampling`'s MTP draft; hatch is never used, since hatch means "doesn't count" and a draft may be accepted); a `verdict` badge stamps each one after the verify pass: ✓ (`--sem-ok`) or ✗ (`--sem-bad`). The corrected token appears as a solid chip with a small plain label "target's pick". The step-time bar from `prefill-decode` (`stepBar`: `weights read`, `KV read` in the reading row, `arithmetic` in its own row) sits under the target so the learner sees the verify pass cost.

Frame 6 zooms on one position: two `vector`s at `NUMBER_CELL` size, `p` (target) and `q` (drafter), over the four candidate words `down` `on` `up` `big`, with printed probabilities; a third `vector` for the leftover (residual). Frames 7–9 use `curvePlot` (proposed in `prefill-decode`).

The followed item is the current guess under test (accent frame), the same in frames 4–6.

Terms introduced (one per frame): target (1), drafter (2), verify pass (3), accept / reject (4), corrected token (5), rejection sampling (6), acceptance rate α (7), drafter cost c (8), batch effect (9), self-drafting heads (10). Terms assumed: from `prefill-decode`: decode step, memory-bound, compute-bound, ridge, batch; from `sampling`: sampling from probabilities, greedy, MTP heads.

Glyphs used: `token` (state `draft` for unverified guesses), `block`, `verdict`, `vector`, `stepBar` (the step bar, P4-R8), `shareBar` (frame 6's two-part keep / reject bar, P4-R17), `curvePlot`, `flow` (carry `token`).

Plain text labels: "guesses, probabilities and drafter cost are hand-picked stand-ins" (frame 2); "two quantities on one y-axis: tokens per round (top curve) and speedup after drafting cost (lower curve); the gap is the drafter's cost" (frames 7–8); "EAGLE-3's measured 1.38× at batch 64 is lower than this toy's 2.14×; its setup had less KV to read per step" (frame 9); "target's pick" (frame 5); "What if?" is not used; "measured by the papers' authors, not this toy" over the 2026 numbers (frame 10).

New glyphs proposed: none.

## 5. Animation script
Numbers: `expectedTokens`, `batchSpeedup` (§6) on the running example with 1,024 tokens of context per user, drafter cost c = 0.05 of a target step.

| # | On screen | What moves | Caption (final wording) | Numbers shown |
|---|---|---|---|---|
| 1 | Chips `The cat sat` and the target `block`. One normal decode step: `down` appears; the step bar shows `weights read` 14.6 ms. | One chip in, one out; the bar draws. | Normal decoding: the big model, called the target here, makes one token per step. On the H200 example each step takes about 14.7 ms. | batch 1, context 1,024: 14.7 ms per token |
| 2 | Branch from `sat`: the drafter `block` emits three `draft` chips (dashed) `down` `on` `a`. Label: "guesses, probabilities and drafter cost are hand-picked stand-ins". | Three small, fast `flow` dots from the drafter; chips appear one by one. | A small drafter guesses the next three tokens: down, on, a. Each guess costs it about a twentieth of a target step. | k = 3 guesses · drafter cost c = 0.05 of a step each |
| 3 | The target takes `sat` + the three guesses in one pass (4 positions). The step bar for this pass (14.7 ms) sits under the one from frame 1 (14.7 ms); the verify pass's 283 µs `arithmetic` part widens to 18 px under a bracket with its value printed (README lesson 19). | Four chips enter the target together; the bar draws to the same length as frame 1's. | The target checks all three guesses in one pass, like a tiny prefill. Four positions cost about the same as one, because the step waits on memory. | verify pass: 4 tokens · 14.7 ms vs 14.7 ms for 1 token |
| 4 | `verdict` badges stamp left to right: `down` ✓, `on` ✓, `a` ✗. Chips after the ✗ (none here) would fade. | Badges stamp one per beat; the followed frame moves along the guesses. | The target checks the guesses left to right. down and on pass; a fails, and anything guessed after the first failure is thrown away. | accepted 2 of 3 |
| 5 | At `a`'s position a solid chip `the` appears with the label "target's pick". The round's result: `down on the`. | `a` fades out; `the` drops in. | At the failed position the target supplies its own token, the. This round made three tokens for about the price of one target step. | 2 accepted + 1 corrected = 3 tokens in 14.7 ms + 3 × 733 µs of drafting |
| 6 | Zoom on the position after `The cat sat`: `vector` p = [0.60, 0.25, 0.10, 0.05] and q = [0.70, 0.20, 0.05, 0.05] over `down` `on` `up` `big`; the drafter's guess `down` is followed. Readouts: "keep down with chance 0.60 / 0.70 = 0.857"; a third vector "leftover" = [0, 0.50, 0.50, 0]; under it "result = [0.60, 0.25, 0.10, 0.05], the target's own". | The keep-chance `shareBar` (two parts: keep 85.7% · reject 14.3%) fills to its keep part; then the leftover vector fills; then the result vector types in and matches p cell for cell. | A guess is kept with a chance set by both models, and a failure is redrawn from the probability the drafter left over. The result follows the target exactly. | keep down: 0.857 · reject mass 0.10 · leftover on 0.5, up 0.5 · result = p |
| 7 | `curvePlot`: x = guesses per round k (1–8), y = expected tokens per round, one curve per α (0.5, 0.7, 0.85, 0.9); the α = 0.7 point at k = 3 framed. The y-axis is labeled "tokens per round" (frame 8 adds a second quantity with its own visible line). | The curves draw left to right; the framed marker settles on 2.53. | Average over many positions and you get an acceptance rate. At 0.7 with three guesses, a round yields 2.53 tokens; each extra guess adds less. | α 0.7: k 1 → 1.70, 3 → 2.53, 5 → 2.94, 8 → 3.20 · α 0.9: k 3 → 3.44 |
| 8 | A second curve on the same axes: speedup after drafting cost (c = 0.05), peaking near k = 5. Visible line: "two quantities on one y-axis: tokens per round (top curve) and speedup after drafting cost (lower curve); the gap is the drafter's cost". | The speedup curve draws under the token curve and bends down after its peak. | Drafting is not free, so the speedup is smaller than the token count. With a drafter costing a twentieth of a step, three guesses give 2.2 times. | α 0.7, c 0.05, batch 1: k 1 → 1.62, 3 → 2.20, 5 → 2.35, 8 → 2.28 |
| 9 | `curvePlot`: x = users in the batch (1–211), y = speedup (k = 3, α 0.7). Step bars for 128 users: plain step 24 ms, verify pass 36.2 ms with `arithmetic` now the longest part. Plain mark under the plot: "EAGLE-3's measured 1.38× at batch 64 is lower than this toy's 2.14×; its setup had less KV to read per step". | The curve stays near 2.2 up to 64 users, then drops; the verify bar's arithmetic part overtakes the reading part. | With 128 users the verify pass checks 512 tokens, past the ridge, so it costs real math. The speedup falls from 2.2 to 1.53, and lower still as users grow. | 64 users 2.14× · 128 users 1.53× (verify 36.2 ms vs plain 24 ms) · 211 users 1.19× |
| 10 | Three labeled rows under "measured by the papers' authors, not this toy": MTP (DeepSeek-V3), EAGLE-3, P-EAGLE, each with its number and date. | Rows fade in one at a time. | In 2026 open-model serving, the drafter is usually part of the model: an MTP head or an EAGLE-style head. Parallel drafters guess every token in one pass. | each row prints its setup on screen: DeepSeek-V3 MTP (2024 report): second token accepted 85–90%, ~1.8× tokens/s · EAGLE-3 (2025, SGLang): up to 6.5× at small batch, 1.38× at batch 64 · P-EAGLE (vLLM, 2026-03-13): up to 1.69× over EAGLE-3 on B200 · closed providers do not publish theirs |

Determinism: every frame is a pure function of (step, progress). Reduced motion shows each frame's end state. Caption check: rlvr-grpo's counter, all ≤ 30 words and ≤ 2 sentences, no formulas.

## 6. Toy
"Guess and check." Two panels: a calculator (acceptance, guesses, drafter cost, batch) and one position (pick the drafter's guess, see the keep chance and the leftover). A visible line: "Acceptance rates, probabilities and drafter cost are stand-ins; real acceptance falls with depth and varies by task (code high, creative text low). Each user has 1,024 tokens of context, so 211 users fit; at 2,048 it would be 105 (`prefill-decode`)."

**Controls**
| id | Label | Type | Range / values | Default | Presets |
|---|---|---|---|---|---|
| `alpha` | Acceptance rate α | Slider | 0.50–0.95, step 0.05 | 0.70 | chips "MTP 0.85", "MTP 0.9" (with k = 1) from `serving.json/deepseek-v3-mtp.acceptance_pct` |
| `k` | Guesses per round | Slider | 1–8 | 3 | — |
| `c` | Drafter cost per guess (fraction of a target step) | Preset chips | 0 · 0.05 · 0.1 · 0.2 | 0.05 | — |
| `batch` | Users in the batch | Slider (snapped) | 1, 4, 16, 64, 128, 211 (max that fits at 1,024 tokens of context) | 1 | — |
| `guess` | Drafter's guess at the zoomed position | Chips | down · on · up · big | down | — |

**Live outputs**
| Output | Formula / `math/` function | Units / format |
|---|---|---|
| Expected tokens per round | `expectedTokens(alpha, k)` | 2 decimals |
| Speedup at batch 1, ignoring the batch | `simpleSpeedup(alpha, k, c)` = E ÷ (1 + k·c) | `formatRatio` |
| Speedup at the chosen batch, plain step, verify pass | `batchSpeedup({ alpha, k, c, batch, model })` → `speedup`, `plainMs`, `verifyMs` | `formatRatio`; `formatDuration` |
| Curves: tokens and speedup vs k; speedup vs batch | the same functions swept | `curvePlot`, current point framed |
| Zoomed position: keep chance, leftover, result | `verifyToken(p, q, guess)`, `outputDistribution(p, q)` with p = [0.60, 0.25, 0.10, 0.05], q = [0.70, 0.20, 0.05, 0.05] | 3 decimals; `vector` cells |
| Position's acceptance rate | `acceptanceRate(p, q)` | 0.900 |

**Check my work** (default state: α 0.7, k 3, c 0.05, batch 1):
```
E = (1 − α^(k+1)) / (1 − α) = (1 − 0.7^4) / (1 − 0.7) = 2.53 tokens per round
speedup = E / (1 + k·c) = 2.53 / (1 + 3 · 0.05) = 2.53 / 1.15 = 2.2× (rounded once, from the unrounded E)
```

**Try this** (each leads to a named insight)
1. α 0.7, c 0.05, batch 1. Slide k 1 → 3 → 5 → 8: tokens per round 1.70 → 2.53 → 2.94 → 3.20; speedup 1.62 → 2.20 → 2.35 → 2.28. → **Insight: guesses have diminishing returns.** A guess counts only if every earlier one survived, while each one costs drafting time, so the best k is small and depends on α and c.
2. k 3, batch 1 → 64 → 128 → 211: speedup 2.20 → 2.14 → 1.53 → 1.19. Now set k = 5 at 211 users: 0.91 of the plain speed. → **Insight: speculative decoding spends idle arithmetic, so it fades when the batch has none left.** It is a latency tool for small batches, not a free throughput multiplier.
3. Press "MTP 0.85" (k = 1): 1.85 tokens per round, 1.76× at batch 1 and still 1.73× at 128 users. → **Insight: one well-trained extra guess survives large batches,** because the verify pass only doubles the tokens. That is close to DeepSeek-V3's reported ~1.8× and why MTP heads are popular for serving.
4. In the position panel, switch the guess from `down` to `on`, then `up`: keep chance 0.857 → 1.000 → 1.000; the result vector stays [0.60, 0.25, 0.10, 0.05] every time. → **Insight: only guesses the drafter overrates are ever rejected, and the leftover repays exactly what rejection removed,** so the target's distribution is preserved whatever the drafter does.

**`math/specdec.js`** (pure, no DOM; `stepTime` from `math/serving.js`):
```js
expectedTokens(alpha, k) → number                 // (1 − α^(k+1)) / (1 − α); k + 1 when α = 1
simpleSpeedup(alpha, k, c) → number               // expectedTokens / (1 + k·c)
acceptanceRate(p, q) → number                     // Σ min(p_i, q_i)
verifyToken(p, q, guessIndex) → { accept, residual, rejectMass }   // accept = min(1, p/q); residual = normalize(max(0, p − q))
outputDistribution(p, q) → number[]               // q·accept + rejectMass·residual (equals p)
batchSpeedup({ alpha, k, c, batch, model })
  → { plainMs, verifyMs, roundMs, tokensPerRound, speedup }
  // plain = stepTime({ ...model, tokens: batch, seqs: batch }); verify = stepTime({ ...model, tokens: batch·(k+1), seqs: batch })
  // round = k·c·plain + verify; speedup = tokensPerRound · plain / round
  // model = { activeParamsPerGpu, weightBytesPerGpu, dModel, actBytesPerElem, kvBytesPerToken, peakTflops, bandwidthTBps, context }
```
Worked examples (scratch implementation, 2026-10-07; `M` = running example with `context: 1024`):
```
expectedTokens: α 0.5 → 1.50 1.75 1.88 1.97 (k 1, 2, 3, 5) · 0.7 → 1.70 2.19 2.53 2.94 · 0.85 → 1.85 2.57 3.19 4.15 · 0.9 → 1.90 2.71 3.44 4.69
expectedTokens(0.7, 8) → 3.20
simpleSpeedup(0.7, k, 0.05): k 1 → 1.62 · 2 → 1.99 · 3 → 2.20 · 5 → 2.35 · 8 → 2.28 ;  c 0.1: k 3 → 1.95, k 5 → 1.96
acceptanceRate(p, q) → 0.900 ; verifyToken(p, q, 0) → accept 0.857, residual [0, 0.5, 0.5, 0], rejectMass 0.10 ; verifyToken(p, q, 1) → accept 1
outputDistribution(p, q) → [0.600, 0.250, 0.100, 0.050]
batchSpeedup({ alpha: 0.7, k: 3, c: 0.05, model: M }): batch 1 → plain 14.66, verify 14.67, 2.20 · 16 → 15.76 / 15.93, 2.18 · 64 → 19.29 / 19.97, 2.14
   · 96 → 21.64 / 27.17, 1.80 · 128 → 23.99 / 36.22, 1.53 · 160 → 26.34 / 45.28, 1.36 · 211 → 30.08 / 59.71, 1.19
k 5: batch 64 → 1.77 · 128 → 1.17 · 211 → 0.91 ;  k 8: batch 64 → 1.27 · 96 → 0.99 · 211 → 0.66
MTP (k 1, c 0.05): α 0.85 → E 1.85, speedup 1.76 at batch 1, 1.73 at 128, 1.72 at 211 · α 0.90 → 1.90, 1.81 at batch 1 / 1.78 at 128
maxUsersPerGpu(71e9, kvCacheBytes({ bytesPerToken: 327680, tokens: 1024 })) → 211
```
Reproducer (run once `math/specdec.js` exists):
```
node -e "import('./math/specdec.js').then(m => { const M = { activeParamsPerGpu: 70e9, weightBytesPerGpu: 70e9, dModel: 8192, actBytesPerElem: 1, kvBytesPerToken: 327680, peakTflops: 1979, bandwidthTBps: 4.8, context: 1024 }; for (const a of [0.5, 0.7, 0.85, 0.9]) console.log(a, [1, 2, 3, 5, 8].map(k => m.expectedTokens(a, k).toFixed(2))); console.log([1, 2, 3, 5, 8].map(k => m.simpleSpeedup(0.7, k, 0.05).toFixed(2))); for (const k of [1, 3, 5]) for (const b of [1, 64, 128, 211]) console.log(k, b, m.batchSpeedup({ alpha: 0.7, k, c: 0.05, batch: b, model: M })); const p = [0.6, 0.25, 0.1, 0.05], q = [0.7, 0.2, 0.05, 0.05]; console.log(m.acceptanceRate(p, q), m.verifyToken(p, q, 0), m.outputDistribution(p, q)) })"
```
Tests to write first: the tables above; `expectedTokens(α, 0) === 1`; `expectedTokens` increases in k and in α and never exceeds k + 1; `outputDistribution(p, q)` equals p for random p, q (seeded with `mulberry32`); `verifyToken` accept is 1 whenever p ≥ q; `batchSpeedup` at batch 1 matches `simpleSpeedup` within 1% while verify is memory-bound; inputs not mutated.

## 7. Show me the math
```tex
P(\text{keep } x) = \min\!\left(1,\ \frac{\htmlClass{hl-p}{p(x)}}{\htmlClass{hl-q}{q(x)}}\right),
\qquad
x' \sim \operatorname{norm}\big(\max(0,\ \htmlClass{hl-p}{p} - \htmlClass{hl-q}{q})\big) \text{ on reject}
```
```tex
\Pr[\text{output}=x] = q(x)\min\!\left(1,\tfrac{p(x)}{q(x)}\right) + \Big(1-\sum_y \min(p(y),q(y))\Big)\frac{\max(0,p(x)-q(x))}{\sum_y \max(0,p(y)-q(y))} = \htmlClass{hl-p}{p(x)}
```
```tex
\alpha = \sum_x \min(p(x), q(x)),
\qquad
\mathbb{E}[\text{tokens per round}] = \frac{1-\alpha^{k+1}}{1-\alpha},
\qquad
\text{speedup} = \frac{\mathbb{E}\cdot t_{\text{step}}(B)}{k\,c\,t_{\text{step}}(B) + t_{\text{step}}\big(B(k+1)\big)}
```
Shapes: p, q ∈ ℝ^|V| (4 words in the zoom; the full vocabulary in a real model). `t_step(n)` is `stepTime` with n tokens and B sequences' KV. The constant-α formula assumes every guess is accepted independently with the same α; real acceptance falls with depth (04 §4.1). Color links: `hl-p` = the target's `p` vector and the result vector (frame 6); `hl-q` = the drafter's `q` vector.

## 8. In today's models (Oct 2026)
| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| vLLM: speculative decoding "preserves the verifier model's output distribution exactly via rejection sampling" | `serving.json/vllm-specdec.exact_distribution = true` *(proposed, confirmed)* | 04 §4.1 CONFIRMED |
| DeepSeek-V3's MTP module predicts one extra token; the second token is accepted 85–90% of the time, giving ~1.8× tokens per second | `serving.json/deepseek-v3-mtp.acceptance_pct = [85, 90]`, `.tps_gain = 1.8` *(proposed)* | 04 §4.2 CONFIRMED (via excerpt) |
| EAGLE-3 (2025): up to 6.5× vs plain decoding at small batch, about 1.4× better than EAGLE-2 (the abstract's words); 1.38× throughput at batch 64 in SGLang | `serving.json/eagle-3.peak_speedup = 6.5`, `.gain_vs_eagle2 = 1.4`, `.batch64_throughput_gain = 1.38`, `.release_date = "2025-03-03"` (the year prints only because this date is confirmed) | 04 §4.1, §4.2 CONFIRMED (abstract) |
| P-EAGLE parallel drafting (vLLM, 2026-03-13): the drafter emits all k tokens in one forward pass; up to 1.69× over EAGLE-3 on B200 | `serving.json/p-eagle.gain_vs_eagle3 = 1.69` *(proposed)* | 04 §4.2 CONFIRMED |
| vLLM (2026-07-28) frames parallel drafters (P-EAGLE, DFlash, DSpark) as the state of the art; they avoid retuning k as load changes | `serving.json/vllm-specdec.parallel_drafters_note` *(proposed)* | 04 §4.3 CONFIRMED |
| MTP raised per-user throughput 87% for DeepSeek-R1 on GB300 NVL72 at 128K in / 8K out while keeping peak system throughput (LMSYS, 2026-02-19) | `serving.json/lmsys-gb300-longctx.mtp_per_user_gain_pct = 87` *(proposed)* | 04 §4.3 CONFIRMED |
| Self-drafting heads (MTP, EAGLE-family) are the norm in open-model serving; separate draft models are a legacy option | `serving.json/vllm-specdec.mainstream_note` *(proposed, `reported`)* | 04 §4.3 (the brief's "safe claim") |

Not shown: the share of public API traffic using speculative decoding (UNVERIFIED, 04 §4.3).

## 9. Takeaways
1. A cheap drafter guesses k tokens and the target checks them all in one pass, which costs about one decode step while decode is memory-bound (frames 2–5).
2. Rejection sampling keeps a guess with chance min(1, p/q) and redraws failures from the leftover, so the output is exactly the target's; tokens per round grow with α but shrink per extra guess (frames 6–8, try-this 1, 4).
3. The gain is per-user latency: as the batch grows the verify pass becomes compute-bound and the speedup falls, even below 1× with many guesses; one high-acceptance MTP guess holds up best (frame 9, try-this 2–3).

## 10. Next and go deeper
Next: `serving-calculator` · Related: `sampling` (MTP heads, temperature), `prefill-decode` (why the verify pass is nearly free), `batching`.

Go deeper (brief 05 §1.2, 04 §9.2): PyTorch, "A Hitchhiker's Guide to Speculative Decoding" (https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/) · Li et al., EAGLE-3 (https://arxiv.org/abs/2503.01840) · vLLM, "P-EAGLE" (https://vllm.ai/blog/2026-03-13-p-eagle).

## 11. Key-frame sketch
Frame 9 end state, desktop width; numbers from the `batchSpeedup` rows in §6.
```text
Speculative decoding            step 9 / 10   [<] [Play] [>]
The cat sat [down v] [on v] [a x] -> the     (k 3, alpha 0.7)
speedup
 2.20 |*-----*
 2.14 |            *
 1.80 |               *
 1.53 |                  *
 1.19 |                         *
      +--+-----+-----+--+--+---+----------- users
         1    16    64 96 128 211
128 users: plain 24 ms | verify 36.2 ms (math-bound)
```

## 12. Open questions for the reviewer
**Data-pass keys**
- `serving.json/vllm-specdec.*`, `deepseek-v3-mtp.*`, `eagle-3.*`, `p-eagle.*`, `lmsys-gb300-longctx.mtp_per_user_gain_pct` as in §8.

**Cross-track**
- **`sampling` handoff.** This page assumes `sampling` introduces MTP heads as a preview and defines greedy vs sampled decoding; it uses the same running prompt (`The cat sat` → `down`) if `sampling` does. If `sampling` picks different candidate words, frame 6's four words should match its vocabulary.

**Judgment calls**
- **Drafter cost c as a fraction of a target step, constant with batch.** A real drafter's cost also grows with batch; keeping c fixed slightly flatters large batches, so the falling curve in frame 9 is, if anything, too kind (the lesson holds).
- **Where the drop starts.** On the running example the speedup stays near 2.2 up to 64 users because the KV read keeps the verify pass memory-bound; EAGLE-3 measured a much lower gain at batch 64 on its own setup. The page shows both and says which is measured.

## 13. Reviewer rulings (expert review, Fable 5.1, 2026-10-07)
- **Settled:** unverified guesses use the library's `token` state `draft` (dashed border), shared with `sampling`; hatch is never used for a draft (README lesson 24).
- **Settled:** frame 6's four words come from `sampling`'s vocabulary (`down` `on` `up` `big`); the probabilities and every number are unchanged.
- **Settled:** the constant drafter cost c is accepted; the page says it flatters large batches.
- **Settled:** cache size comes from `kvCacheBytes` (`kv-cache`).
- Applied: "in 2026 open-model serving" scopes the drafter claim (§3, frame 10); misconception 1 says why the result is exact; the 1,024-token context is explained on screen; frame 3 prints 14.67 vs 14.66 ms with the 0.28 ms math part in a zoomed bar; frames 7–8 carry the two-quantities line; frame 10 rows carry their setups; frame 9 notes why EAGLE-3's measured gain at batch 64 is lower.

