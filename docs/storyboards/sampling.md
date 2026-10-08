# Picking the next token (`sampling`)

Track: architecture · Section: architecture · Prereqs: decoder-recap
Next: `speculative-decoding` (the one slug whose `prereqs` list `sampling` in `shared/concepts.json`)
Status: approved (expert review)
Sources: 01 §1 ("Decoding" row), §5 (MTP paragraph), §7 · 04 §4.1 (formula, for the hand-off only), §4.2 (MTP row), §4.3, §9.1 (MTP preset) · 05 §1.1 (Sampling: "Build a small top-k/top-p/temperature toy"), §1.2. Sampling mechanics are first principles; nothing beyond the briefs.

Picks up exactly where `decoder-anatomy` frame 8 stops: the same 16-word vocabulary, the same scores
(on 2.0 · "." 1.5 · and 0.5 · the 0.0 · the other 12 words −1.0), the same five-cell collapse ("12 others")
and the same probabilities (on 0.390 · "." 0.237 · and 0.087 · the 0.053 · 0.019 each). It uses
`core.softmax(logits, temperature)`, which already exists. Multi-token prediction is only previewed; the
draft-and-verify math (expected tokens per step for k drafts) belongs to `speculative-decoding`.

## 1. Learning objective
After this page you can name the scores the model outputs (logits), turn them into a token by greedy choice
or by a random draw (frames 1–3), say what temperature, top-k and top-p each do to the distribution and
compute them on the five-cell row (frames 4–6, toy), explain why the same prompt gives different text
(frame 7, try-this 3), and describe how a multi-token-prediction head lets a model propose two tokens per
step (frames 8–9).

## 2. Misconceptions to correct
Each one names the frame or try-this that corrects it (README lesson 4).
- **Misconception:** "The model outputs a word." → **Reality:** it outputs one score per vocabulary entry;
  a separate, parameter-free sampler turns them into a token. Same weights, different sampler, different
  text. (first principles; `decoder-anatomy` misconception 4) · corrected by frames 1–3
- **Misconception:** "Temperature makes the model more creative or smarter." → **Reality:** it only
  rescales the scores before softmax. Below 1 the top token gains ("on" 0.390 → 0.682 at 0.5); above 1 the
  tail gains (the 12 unlikely words go from 0.233 together to 0.506 at 2). Nothing new is learned or
  computed. (first principles) · corrected by frame 4, try-this 1
- **Misconception:** "Top-k and top-p are the same filter." → **Reality:** top-k keeps a fixed number of
  tokens; top-p keeps however many it takes to reach a share of the probability. At p = 0.7, top-p keeps 2
  tokens at temperature 0.5, 3 at 1 and 9 at 2, while top-3 always keeps 3. (first principles) · corrected
  by frames 5–6, try-this 2
- **Misconception:** "Greedy decoding gives the best answer." → **Reality:** it gives the single most likely
  next token each step, which is deterministic and can repeat itself, because the most likely next token is
  often the one that continues the pattern already on the page; it is temperature → 0, one end of the
  same dial. (first principles) · corrected by frame 2 and try-this 1's greedy chip
- **Misconception:** "A model must produce exactly one token per forward pass." → **Reality:** a
  multi-token-prediction head proposes the token after next in the same pass; if the next pass agrees,
  two tokens come out of one step. DeepSeek-V3's second token is accepted 85–90% of the time.
  (04 §4.2 [C]; 01 §5 MTP) · corrected by frames 8–9

## 3. Hook and intuition (final wording)
**Hook:** The model ends every step with 163,840 scores in Kimi K3, one per possible token. Who decides
which one becomes the next word, and why does asking twice give two different answers?

The last step of `decoder-anatomy` gave one score per vocabulary word, the logits, and softmax turned them
into probabilities. That is where the model's job ends. Choosing a token is a separate step with no learned
parameters. The simplest choice is to always take the most likely token, greedy decoding. It is
predictable, and it tends to fall into loops.

Most chat systems draw instead: they pick a token at random in proportion to its probability, so "on" comes
out 39% of the time and "." 24% (over many draws). That is why the same prompt gives different replies. Three knobs shape the
draw. Temperature divides the scores before softmax: below 1 it sharpens toward the favorite, above 1 it
flattens toward the long tail. Top-k throws away everything but the k most likely tokens. Top-p keeps the
smallest set of tokens whose probabilities add up to p, so it keeps few tokens when the model is confident
and many when it is not. After any cut, the survivors are rescaled to add up to 1.

One more 2026 twist: some models predict more than one token per step. A small extra head, trained with the
model, guesses the token after next. On the following step the model checks that guess while computing
anyway, and if it agrees, two tokens come out of one step. The cost is the extra head and some wasted work
when the guess is wrong; `speculative-decoding` turns this into a full speed-up method.

## 4. Visual metaphor
Numbers are `decoder-anatomy`'s, unchanged (§6 below regenerates them). The row is five cells at
`NUMBER_CELL` (README lesson 18: 16 cells would need about 690 px): on · "." · and · the · "12 others". The
collapsed cell shows "each" and "together" values and, when a filter cuts through the tied 12, "n of 12
kept" (ties are broken by vocabulary order: The, cat, sat, down, mat, a, dog, ran, up, big, was, then). The
random number in frame 3 is a hand-picked stand-in, u = 0.55; the toy's draws use `mulberry32` seeds. A
visible line under the stage (README lesson 10): "The 16 scores are `decoder-anatomy`'s hand-picked
stand-ins; the random number in frame 3 is picked by hand, and the toy's draws are seeded."

**Terms introduced, one per frame** (README lesson 3): 1 logits · 2 greedy decoding · 3 sampling (a random
draw by probability) · 4 temperature · 5 top-k · 6 top-p · 7 seed · 8 multi-token-prediction (MTP) head ·
9 accept / reject of a draft. **Terms assumed from prereqs:** softmax, vocabulary, unembedding, score, the
generate-append loop (`decoder-anatomy`); the block and residual stream (`decoder-recap`). **Named and
deferred:** acceptance math for several drafts, EAGLE and parallel drafters (`speculative-decoding`); the
cross-entropy loss that trains these scores (`pretraining`); `attention`'s divisor slider is a different
temperature, inside attention (that page says so).

**Layout** (stage ≈ 580 × 366):
- Top: the four token chips "The cat sat down", then an empty slot "₅".
- Center: the five-cell row, scores, then probabilities beneath it (both `vector`s at `NUMBER_CELL`, 5 × 43 =
  215 px), "Σ = 1.000" as text.
- Frame 3: the probabilities redrawn as one `shareBar` (`decoder-anatomy`'s accepted glyph) from 0 to 1 at the
  full stage width (560 px, so "the" at 0.053 is 30 px and "and" at 0.087 is 49 px; README lesson 19), with
  five segments (on, ".", and, the, 12 others), each boundary (0.390, 0.627, 0.714, 0.767) printed under the
  bar, plus a plain text u marker.
- Frames 4–6: a second probability row under the first ("after") so before and after are both visible.
- Frame 7: `bars` (accepted on `moe`'s review, conditions (a)–(f)) of 20 draws per category; the
  numbered-cell fallback is documented there and not needed.
- Frames 8–9: a `blockStack` sliver (last block) with a second small `block` "MTP head" beside the
  unembedding; two output chips, the second a `token` in the library's `draft` state (dashed border, defined
  once and shared with `speculative-decoding`).

Glyphs used (from spec §5.1 and accepted elsewhere): token (including its `draft` state), vector, block,
flow (carry `token`), blockStack, shareBar, verdict, `bars` (accepted on `moe`).
New glyphs proposed: none. The cumulative strip in frame 3 is a `shareBar`
(segment length plus printed share, accepted on `decoder-anatomy`), not a new drawing. Small marks: "Σ =
1.000", "n of 12 kept" and "kept 0.714 → rescaled" are plain text labels; accept / reject uses the `verdict`
glyph, as on `speculative-decoding`.

Color: scores maxAbs 2, probabilities maxAbs 1 (cells on the value scale). Cut tokens are hatched (the
library's "excluded"). A draft may still count, so it is never hatched: it is the `token` `draft` state (README
lesson 24). The chosen token's cell gets the selection outline in frames 2, 3 and 9; the followed
position (slot ₅, then the MTP slot ₆) keeps the same outline.

## 5. Animation script
| # | On screen | What moves | Caption (final wording) | Numbers shown |
|---|---|---|---|---|
| 1 | Five score cells under the chips, then the five probability cells, "Σ = 1.000". A label "these scores are the logits" and the visible line "top: scores, any size · bottom: probabilities, add to 1". | The score row fills; the probability row interpolates in beneath it. | The model's last step gives one score per word in its vocabulary, called the logits. Softmax turns them into probabilities that add to 1. | logits: on 2.0 · "." 1.5 · and 0.5 · the 0.0 · 12 others −1.0 each · probabilities: 0.390 · 0.237 · 0.087 · 0.053 · 0.019 each (0.233 together) |
| 2 | The "on" cell gets the selection outline; a chip "on₅" drops into the slot. Plain label "greedy: always the top". | The outline appears; the chip falls into place. | Greedy decoding always takes the most likely token. It is predictable, but the same prompt always gives the same text, and it can repeat itself. | pick: on (0.390), every time |
| 3 | The probability row redraws as a `shareBar` from 0 to 1 at full stage width, boundaries 0.390, 0.627, 0.714, 0.767 printed under it; a marker at u = 0.55 lands in "."; the chip in the slot becomes ".₅". | The marker slides from 0 to 0.55; the "." slot lights. | Sampling draws a random number and picks the token whose slice of the strip it lands in. Here 0.55 lands in the slice for the period. | cumulative: on 0.390 · "." 0.627 · and 0.714 · the 0.767 · u = 0.55 → "." |
| 4 | "Before" row (T = 1) and "after" row; a temperature readout steps 1 → 0.5 → 2; the scores row shows the divided scores. | The after row interpolates to T = 0.5, holds, then to T = 2. | Temperature divides every logit before softmax. Below 1 the favorite takes more; above 1 the long tail takes more. | T = 0.5: logits 4, 3, 1, 0 → 0.682 · 0.251 · 0.034 · 0.012 · 0.002 each (0.020 together) · T = 2: 1, 0.75, 0.25, 0 → 0.189 · 0.147 · 0.089 · 0.069 · 0.042 each (0.506 together) |
| 5 | Back to T = 1. "the" and the 12 others hatch over; the surviving three rescale. Label "kept 0.714 → rescaled to 1". | Cells hatch from the right; the three survivors grow. | Top-k keeps only the k most likely tokens and rescales them to add up to 1. With k equal to 3, "on" now gets 55%. | top-3: 0.390 · 0.237 · 0.087 → 0.547 · 0.331 · 0.122 (÷ 0.714) |
| 6 | A running sum ticks along the sorted cells until it passes p = 0.7; the rest hatch. Below, small text: "at T = 0.5 it keeps 2; at T = 2 it keeps 9". | The sum counter climbs 0.390 → 0.627 → 0.714 and stops; the remaining cells hatch. | Top-p keeps the fewest tokens whose probabilities reach p, then rescales. It keeps few tokens when the model is sure and many when it is not. | p = 0.7: on + "." + and = 0.714 ≥ 0.7 → 3 kept · same three as top-3 here · at T = 0.5: 2 kept (0.933) · at T = 2: 9 kept (0.705) |
| 7 | `bars` of 20 draws at T = 1, seed 1, over the five categories; then the same at T = 0.5. The first eight drawn tokens print as text under the bars. | Bars grow draw by draw; then reset and regrow for T = 0.5. | Each draw uses a new random number, so the same prompt gives different text. A fixed seed repeats the same draws exactly. | T = 1, seed 1: on 6 · "." 8 · and 1 · the 1 · others 4 · first eight: and on . then was on . the · T = 0.5, seed 1: 15 · 2 · 0 · 1 · 2 |
| 8 | Branch label "in some 2026 models". The last block's output feeds two heads: the unembedding (→ "on₅") and a small `block` "MTP head" (→ "the₆", a `token` in the `draft` state, labeled "draft"). | Two `flow`s leave the block; "on₅" lands in the default state, "the₆" in the `draft` state. | A multi-token-prediction head guesses the token after next in the same step. Here the model outputs "on" and drafts "the". | 1 extra head · DeepSeek-V3 / V4: depth 1 · GLM-5: shares 3 MTP layers |
| 9 | The next step runs with "on₅" appended; its own pick for position 6 is compared with the draft; a `verdict` glyph (accept) appears beside it. A counter "2 tokens from 1 step". | The new pass's pick slides up to meet the draft chip; the `verdict` mark appears and the chip changes to the default state. | The next step checks the draft while it computes anyway. If it agrees, the draft is kept and two tokens come out of one step. | DeepSeek-V3: second token accepted 85–90% → 1.85–1.90 tokens per step on average (1 + acceptance, one draft) · full method: `speculative-decoding` |

Determinism: every frame is a pure function of (step, progress); draws are seeded. Reduced motion shows
each frame's end state. The five-cell rows keep their positions from frame 1 to frame 6; slot ₅ keeps its
outline.

Nothing load-bearing is behind hover (README lesson 5): every probability, the cumulative boundaries, the
kept counts, the draw counts and frame 7's first eight drawn tokens are printed; hover adds only more decimal
places.

Caption word counts (README lesson 2; ≤ 30 words, ≤ 2 sentences, no operators): 24 · 25 · 26 · 19 · 25 ·
26 · 22 · 20 · 25.

Absolutes checked (README lesson 7): frame 2's "the same prompt always gives the same text" holds for greedy
decoding on a fixed model and input (first principles; hardware nondeterminism is ignored and not claimed);
frame 7's "repeats the same draws exactly" is the seeded generator. Frame 9's acceptance range is DeepSeek-V3
specifically (04 §4.2 [C]).

Branches (README lesson 13): frames 8–9 are labeled "in some 2026 models" and come after the sampling thread
(frames 1–7) closes.

## 6. Toy
Title on page: "Shape the draw." One state object, one `render()`. The stand-in line from §4 is printed
above.

**Controls**
| id | Label | Type | Range / values | Default | Presets |
|---|---|---|---|---|---|
| `temperature` | Temperature | Slider (snapped) | [0.25, 0.5, 0.75, 1, 1.5, 2] | 1 | chip "greedy" (temperature → 0: argmax) |
| `topK` | Top-k | Slider (snapped) | [off, 1, 2, 3, 5, 8] | off | — |
| `topP` | Top-p | Slider (snapped) | [off, 0.95, 0.9, 0.75, 0.7, 0.5] | off | — |
| `seed` | Seed | stepper | 1–5 | 1 | "new seed" button |

Order of operations, printed beside the controls: "temperature → top-k → top-p → rescale → draw" (a common
order; libraries differ).

**Live outputs**
| Output | Formula / `math/` function | Units / format |
|---|---|---|
| Five-cell distribution after all filters | `samplingDistribution(LOGITS, { temperature, topK, topP })` then `collapseOthers(probs, NAMED)` | 3 d.p.; "n of 12 kept" |
| Tokens kept and probability kept before rescaling | `.kept`, `.mass` | count; 3 d.p. |
| 20 draws per category | `drawSamples(probs, 20, seed)` grouped by `collapseOthers` | `bars` with printed counts |
| First eight draws as text | `drawSamples(…).slice(0, 8)` mapped to words | mono line |
| "Check my work" | templated for the current temperature; at T = 0.5 it prints all five terms: "e^4 = 54.6 · e^3 = 20.1 · e^1 = 2.72 · e^0 = 1 · 12 × e^−2 = 1.62 · sum 80.03 · on = 54.6 ÷ 80.03 = 0.682" | mono, `aria-live="polite"` |

**Try this** (each leads to a named insight)
1. Temperature **0.5**: "on" 0.682, the 12 others 0.020 together; seed 1 draws "on" 15 times in 20.
   Temperature **2**: "on" 0.189, others 0.506; 8 of 20 draws come from the 12 unlikely words. Tap
   **greedy**: "on" every time. → **Insight: temperature is one dial from "always the favorite" to "almost
   uniform"; it changes how the draw spends probability, not what the model knows.**
2. Top-p **0.7** at temperature **1**: 3 kept. At **0.5**: 2 kept. At **2**: 9 kept (5 of the 12 tied words,
   chosen by vocabulary order). Now top-k **3** instead: 3 kept at every temperature. → **Insight: top-p
   adapts to how sure the model is; top-k does not.**
3. Temperature 1, no filters, step the **seed** 1 → 2 → 3: the first eight tokens change each time; return
   to seed 1 and they come back exactly ("and on . then was on . the"). → **Insight: the randomness lives
   in the sampler, not in the model; fix the seed and the text repeats.**

**`math/sampling.js`** (pure, no DOM, inputs never mutated; tests first; imports `softmax`, `mulberry32`
from `./core.js`).
```js
export const VOCAB = Object.freeze(['The','cat','sat','down','on','.','and','the','mat','a','dog','ran','up','big','was','then']);   // frozen, owned here; any page printing course-toy tokens picks from it
export const LOGITS = [-1,-1,-1,-1, 2, 1.5, 0.5, 0, -1,-1,-1,-1,-1,-1,-1,-1];   // decoder-anatomy frame 8
export const NAMED = [4, 5, 6, 7];                                              // on . and the

// Indices sorted by probability, highest first; ties by lower vocabulary index. Used by every filter and the draw.
rankTokens(probs) → number[]

// temperature 0 → one-hot on the top token (greedy); otherwise core.softmax(logits, temperature).
// The 0 case is special-cased on purpose: core.softmax throws on temperature ≤ 0. Do not "fix" it.
applyTemperature(logits, temperature) → number[]
//   (LOGITS, 1)[4]   → 0.390 · (LOGITS, 0.5)[4] → 0.682 · (LOGITS, 2)[4] → 0.189 · (LOGITS, 0) → one-hot at 4 (exact)

// Keep the k highest (ties by vocabulary order), rescale.
topK(probs, k) → { probs, kept, mass }
//   (applyTemperature(LOGITS, 1), 3) → probs[4,5,6] = [0.547, 0.331, 0.122] · kept 3 · mass 0.714
//   (…, 6) → keeps on . and the The cat · others cell "2 of 12 kept"

// Keep the shortest prefix of rankTokens whose mass reaches p, rescale.
topP(probs, p) → { probs, kept, mass }
//   (T 1, 0.7)   → kept 3 · mass 0.714 · [0.547, 0.331, 0.122]
//   (T 1, 0.5)   → kept 2 · mass 0.627 · [0.622, 0.378]
//   (T 1, 0.75)  → kept 4 · mass 0.767
//   (T 1, 0.9)   → kept 11 · mass 0.903 (cuts inside the tied 12: keeps The, cat, sat, down, mat, a, dog)
//   (T 0.5, 0.7) → kept 2 · mass 0.933 · (T 2, 0.7) → kept 9 · mass 0.705

// Temperature → top-k → top-p.
samplingDistribution(logits, { temperature = 1, topK = null, topP = null }) → { probs, kept, mass }

// Walk rankTokens(probs), return the first index whose cumulative probability exceeds u (0 ≤ u < 1).
sampleIndex(probs, u) → number
//   (T 1, 0.2) → 4 (on) · (T 1, 0.55) → 5 (".") · (T 1, 0.7) → 6 (and) · (T 1, 0.75) → 7 (the) · (T 1, 0.95) → 13 (big)

// n draws with u from mulberry32(seed).
drawSamples(probs, n, seed) → number[]
//   (T 1, 20, 1) → counts on 6 · "." 8 · and 1 · the 1 · others 4; first eight: and on . then was on . the
//   (T 0.5, 20, 1) → 15 · 2 · 0 · 1 · 2 · (T 2, 20, 1) → 4 · 2 · 2 · 4 · 8 · (top-p 0.7, 20, 1) → 13 · 4 · 3 · 0 · 0

// Display helper: the named cells plus one "others" cell { each, together, keptOf }.
collapseOthers(probs, named) → { named: number[], others: { each, together, kept, of } }
//   (T 1) → named [0.390, 0.237, 0.087, 0.053] · others { each: 0.019, together: 0.233, kept: 12, of: 12 }
```

Tests to write first: every filter's output sums to 1 (|Σ − 1| < 1e-12); `topK(p, 16)` and `topP(p, 1)` are
identities; `topP` keeps the same set as `topK` when the cut falls on a boundary (p = 0.7 ↔ k = 3 at T = 1);
temperature 0 equals `topK(…, 1)`; `drawSamples` is identical for equal seeds; the examples above to 3 d.p.;
inputs not mutated. README lesson 16: the prose claims "below 1 the favorite gains, above 1 the tail gains"
are asserted as monotonic tests on `applyTemperature`.

**Hand-off to `speculative-decoding`:** frame 9's "1.85–1.90 tokens per step" is brief 04 §9.1's MTP preset
(one draft, acceptance 0.85–0.90). The general expected-tokens function lives on `speculative-decoding`; at
one draft it must reproduce 1 + acceptance (consistency decision for the Task 12 review).

**Reproducer** (run from the repo root on 2026-10-07 against the shipped `math/core.js`; output matched
every number in §2, §5, §6 and §11):
```sh
node -e '
import("./math/core.js").then(({softmax,mulberry32})=>{
const V=["The","cat","sat","down","on",".","and","the","mat","a","dog","ran","up","big","was","then"];
const z=[-1,-1,-1,-1,2,1.5,.5,0,-1,-1,-1,-1,-1,-1,-1,-1],r=(x,d=3)=>Array.isArray(x)?x.map(v=>r(v,d)):Number(x.toFixed(d));
const rank=p=>p.map((v,i)=>({i,v})).sort((a,b)=>b.v-a.v||a.i-b.i).map(o=>o.i);
const five=p=>{const o=p.filter((_,i)=>![4,5,6,7].includes(i));return [...[4,5,6,7].map(i=>p[i]),o[0],o.reduce((s,x)=>s+x,0),o.filter(x=>x>0).length];};
const keep=(p,set)=>{const q=p.map((v,i)=>set.has(i)?v:0),s=q.reduce((a,b)=>a+b,0);return q.map(v=>v/s);};
const topk=(p,k)=>keep(p,new Set(rank(p).slice(0,k)));
const topp=(p,pp)=>{let c=0;const s=new Set();for(const i of rank(p)){s.add(i);c+=p[i];if(c>=pp-1e-12)break;}return{q:keep(p,s),kept:s.size,mass:r(c)};};
const p1=softmax(z);
for(const T of [1,0.5,2])console.log("T",T,JSON.stringify(r(five(softmax(z,T)))));
console.log("top3",JSON.stringify(r(five(topk(p1,3)))),"top6",JSON.stringify(r(five(topk(p1,6)))));
for(const [T,pp] of [[1,0.5],[1,0.7],[1,0.75],[1,0.9],[0.5,0.7],[2,0.7]]){const t=topp(softmax(z,T),pp);console.log("topp",T,pp,t.kept,t.mass,JSON.stringify(r(five(t.q))));}
let c=0;console.log("cum",rank(p1).slice(0,4).map(i=>(c+=p1[i]).toFixed(3)).join(" "));
const samp=(p,u)=>{let c=0;for(const i of rank(p)){c+=p[i];if(u<c)return i;}return rank(p).at(-1);};
console.log("u",[0.2,0.55,0.7,0.75,0.95].map(u=>V[samp(p1,u)]).join(" "));
const draws=(p,n,seed)=>{const g=mulberry32(seed);return Array.from({length:n},()=>samp(p,g()));};
for(const [nm,p] of [["T1",p1],["T0.5",softmax(z,0.5)],["T2",softmax(z,2)],["topp0.7",topp(p1,0.7).q]]){const d=draws(p,20,1);
 const cats=[4,5,6,7].map(i=>d.filter(x=>x===i).length);console.log("draws",nm,[...cats,20-cats.reduce((a,b)=>a+b,0)].join(" "),d.slice(0,8).map(i=>V[i]).join(" "));}
})'
# data pass 2026-10-07: Kimi K3 vocabulary, 163,840 (was 160,000)
node -e 'console.log(require("./data/models.json").entries.find((e) => e.id === "kimi-k3").facts.vocab_size.value)'   # 163840
```
Output on 2026-10-07: `T 1 [0.39,0.237,0.087,0.053,0.019,0.233,12]` · `T 0.5
[0.682,0.251,0.034,0.012,0.002,0.02,12]` · `T 2 [0.189,0.147,0.089,0.069,0.042,0.506,12]` · `top3
[0.547,0.331,0.122,0,0,0,0] top6 [0.484,0.294,0.108,0.066,0.024,0.048,2]` · `topp 1 0.5 2 0.627` ·
`topp 1 0.7 3 0.714 [0.547,0.331,0.122,0,0,0,0]` · `topp 1 0.75 4 0.767` · `topp 1 0.9 11 0.903` ·
`topp 0.5 0.7 2 0.933` · `topp 2 0.7 9 0.705` · `cum 0.390 0.627 0.714 0.767` · `u on . and the big` ·
`draws T1 6 8 1 1 4 and on . then was on . the` · `draws T0.5 15 2 0 1 2 …` · `draws T2 4 2 2 4 8 …` ·
`draws topp0.7 13 4 3 0 0 …`.

## 7. Show me the math
```tex
\htmlClass{hl-p}{p_v} = \frac{e^{\htmlClass{hl-z}{z_v} / \htmlClass{hl-t}{T}}}{\sum_u e^{z_u / T}},
\qquad \text{worked } (T = 0.5):\ \frac{e^{4}}{e^{4} + e^{3} + e^{1} + e^{0} + 12\,e^{-2}} = 0.682
```
```tex
\text{top-}k:\ \mathcal{K} = \text{the } k \text{ largest } p_v,\qquad
\text{top-}p:\ \mathcal{K} = \text{smallest prefix of the sorted } p \text{ with } \textstyle\sum_{v \in \mathcal{K}} p_v \ge p,
\qquad \tilde p_v = \frac{p_v}{\sum_{u \in \mathcal{K}} p_u}\ (v \in \mathcal{K})
```
```tex
\text{draw: } v = \min\Big\{\, v_j : \textstyle\sum_{i \le j} \tilde p_{v_i} > u \Big\},\quad u \sim \mathcal{U}[0, 1)
\qquad \text{worked: } u = 0.55 \in [0.390,\ 0.627) \Rightarrow \text{"."}
```
```tex
\text{MTP (one draft): } \mathbb{E}[\text{tokens per step}] = 1 + \alpha,\qquad \alpha = 0.85\text{–}0.90 \Rightarrow 1.85\text{–}1.90
```
Shapes: z, p [|V|] (16; 163,840 in Kimi K3); the MTP head reads the last block's output for position t and
predicts position t + 2. Greedy is the limit T → 0. Color links: `hl-z` → the logit row; `hl-t` → the
temperature readout; `hl-p` → the probability row. KaTeX with `trust: true, strict: false`.

## 8. In today's models (Oct 2026)
Framing paragraph on the page: "Sampling settings are chosen by whoever serves the model, not baked into it.
What is baked in, in several 2026 models, is a head that predicts more than one token per step."

| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| The softmax runs over the whole vocabulary: 163,840 entries in Kimi K3, 201,088 in gpt-oss-120b | `models.kimi-k3.vocab_size` = 163840, `models.gpt-oss-120b.vocab_size` = 201088 (in data; `decoder-anatomy` §12.3) | 01 §5 table [C]; gpt-oss config (re-verified by `decoder-anatomy`) |
| DeepSeek-V3 (Dec 2024): one MTP module predicts one extra token; the second token is accepted 85–90% of the time, about 1.8× tokens per second (for one user, as the V3 report states it) | `models.deepseek-v3.mtp_depth` = 1 (proposed); `serving.json/deepseek-v3-mtp.acceptance_pct` = [85, 90], `.tps_gain` = 1.8 (proposed by `speculative-decoding` §8; one key for both pages; arXiv 2412.19437) | 04 §4.2 [C via excerpt] |
| DeepSeek-V4 (2026): MTP depth 1, also used as an auxiliary training loss | `models.deepseek-v4-pro.mtp_depth` = 1 (proposed) | 01 §1, §5 [C] |
| GLM-5 (2026): shares 3 MTP layers; mean accepted length 2.76 tokens per step in its report | `models.glm-5.mtp_layers` = 3, `.mtp_accept_length` = 2.76 (proposed; same *new* `glm-5` entry as `kv-compression`) | 01 §1, §5 [C] |
| Kimi K3: one MTP layer, fine-tuned as an EAGLE-3-style draft | `models.kimi-k3.mtp_depth` = 1 (proposed) | 01 §5 [C] |
| In serving, MTP raised per-user speed by 87% for DeepSeek-R1 on GB300 NVL72 at 128K input / 8K output, with peak throughput kept (LMSYS, Feb 2026) | `serving.json/lmsys-gb300-longctx.mtp_per_user_gain_pct` = 87 (proposed by `speculative-decoding` §8; note names model, hardware, workload) | 04 §4.3 [C] |

Not shown: recommended sampling settings from model cards (not in the briefs).

## 9. Takeaways
1. The model ends with logits, one score per vocabulary entry; a parameter-free sampler turns them into a
   token: greedy takes the top, sampling draws by probability (frames 1–3).
2. Temperature rescales the logits (sharper below 1, flatter above); top-k keeps a fixed number of tokens;
   top-p keeps as many as it takes to reach p, so it adapts to how sure the model is (frames 4–6, try-this
   1–2). A seed makes the draws repeatable (frame 7, try-this 3).
3. Some 2026 models carry a multi-token-prediction head that drafts the token after next; when the next step
   agrees, one step yields two tokens (85–90% of the time in DeepSeek-V3). The full method is
   `speculative-decoding` (frames 8–9).

## 10. Next and go deeper
Next: `speculative-decoding` (draft, verify, and why it helps latency more than fleet throughput); it lists
`sampling` in `shared/concepts.json`. In-page links: `decoder-anatomy` (where the logits come from),
`pretraining` (the loss that shapes them), `attention` (its divisor slider, a temperature inside attention).

Go deeper (brief 05 §1.1–1.2): Transformer Explainer (https://poloclub.github.io/transformer-explainer/),
whose temperature slider runs on real GPT-2 · Gloeckle et al., *Better & Faster Large Language Models via
Multi-token Prediction* (https://arxiv.org/abs/2404.19737) · DeepSeek-AI, *DeepSeek-V3 Technical Report*
(https://arxiv.org/pdf/2412.19437), §MTP.

## 11. Key-frame sketch
Frame 6 (top-p at 0.7), desktop width; numbers from the §6 reproducer (`topp 1 0.7 3 0.714`, `cum`).
```text
┌──────────────────────────────────────────────────────────┐
│ [The]₁ [cat]₂ [sat]₃ [down]₄ ►[ ? ]₅                     │
│                                                          │
│           on      "."     and     the    12 others       │
│ before │ 0.390 │ 0.237 │ 0.087 │ 0.053 │ 0.019 each  │   │
│ sum      0.390 → 0.627 → 0.714 ≥ 0.7   stop              │
│ after  │ 0.547 │ 0.331 │ 0.122 │ ░░░░░ │ ░░░░░░░░░░  │   │
│                          kept 0.714 → rescaled to 1      │
│   at T = 0.5 it keeps 2; at T = 2 it keeps 9             │
├──────────────────────────────────────────────────────────┤
│ Top-p keeps the fewest tokens whose probabilities reach  │
│ p, then rescales. It keeps few tokens when the model is  │
│ sure and many when it is not.                            │
│ [◄] [Pause] [►]  ━━━━━━●━━━━━  6 / 9   speed [1×]        │
└──────────────────────────────────────────────────────────┘
```
"►" is the selection outline on the slot being filled (position 5). `░` cells are hatched (cut). At 400 px
the stage scrolls inside its container.

## 12. Open questions for the reviewer
**Data-pass keys** (*new*): `mtp_depth` (deepseek-v3 1, deepseek-v4-pro 1, kimi-k3 1), `mtp_layers` and
`mtp_accept_length` (glm-5). Reused, not re-proposed: `serving.json/deepseek-v3-mtp.acceptance_pct` and
`.tps_gain`, `serving.json/lmsys-gb300-longctx.mtp_per_user_gain_pct` (`speculative-decoding` §8);
`vocab_size` (`decoder-anatomy` §12.3).

**Graph changes:** none.

**Judgment calls:** all ruled by the expert review (§13) and applied; none remain open.

## 13. Expert review (2026-10-07) and what changed
Verdict: APPROVE WITH CHANGES (2 Must). Status is now "approved (expert review)". No number changed; the §6
reproducer was re-run and matches; the T = 0.5 "Check my work" terms were computed with `node -e`
(54.598, 20.086, 2.718, 1, 1.624; sum 80.03; 0.682).

Rulings applied (lesson 20): MTP formula ownership consistent (1 + α = `expectedTokens(α, 1)`); order of
filters keeps its hedge; ties by vocabulary order accepted; u = 0.55 accepted; `VOCAB` is frozen and owned
here (`speculative-decoding` frame 6 already draws `down on up big` from it).

Must (2/2): the draft is the library's `token` `draft` state, never hatched (README lesson 24); frame 3's
`shareBar` spans the full stage width so "the" (0.053) is 30 px, with the boundaries printed (lesson 19).
Should (8/8 applicable, no rebuttals): frame 1's "top: scores · bottom: probabilities" line (lesson 21);
§8's 1.8× names "for one user" (lesson 23); `VOCAB` frozen and recorded; frame 9 uses the `verdict` glyph;
misconception 4 says why greedy repeats; the "nothing behind hover" paragraph added, including frame 7's
eight tokens; `applyTemperature`'s 0 special case is commented; "(over many draws)" in §3. The order-of-filters
item needed no change.
Nice (1/2): "Check my work" prints all five T = 0.5 terms; no "common chat default" chip (no source).
- Data pass 2026-10-07: Kimi K3 vocabulary is 163,840 (was printed as 160,000); no computed value on this page depended on it (reproducer: `kimi-k3.vocab_size` = 163840).
