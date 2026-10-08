# From GPT-3 to 2026 (`decoder-recap`)

Track: architecture · Section: architecture · Prereqs: decoder-anatomy
Next: `moe`, `sampling`, `training-pipeline` (the three slugs whose `prereqs` list `decoder-recap` in `shared/concepts.json`)
Status: approved (expert review)
Sources: 01 §1 (every row), §2 (GQA/MLA, sliding window, hybrid bullets), §3 (RoPE, NoPE), §5 (MTP), §7 · 05 §1.1, §1.2 (Raschka gallery). Nothing beyond the briefs.

`decoder-anatomy` drew the 2026 block and taught the forward pass end to end; this page does not re-teach it.
It starts from GPT-3's block (2020) and swaps one part per frame, each with its *why*, and hands every part
off to the lesson that opens it. It owns two things `decoder-anatomy` deferred here: RMSNorm's formula
(worked on `decoder-anatomy`'s row `x_sat`, rms 0.586) and QK-norm (worked on `attention`'s head A row
"sat"). It names attention sinks and hybrid layers in one frame each and links to `long-context-attention`,
which owns their mechanics.

## 1. Learning objective
After this page you can list what changed between GPT-3's block and a 2026 block, part by part: RMSNorm
instead of LayerNorm (frame 2), RoPE instead of a position table (frame 3), a gated MLP (frame 4), experts
(frame 5), shared key/value heads or latents (frame 6), QK-norm and sinks around the softmax (frames 7–8) and
mixed layer types (frame 9), and give one reason for each; and you can compute what each swap does to GPT-3's
parameter count and its cache per token (toy).

## 2. Misconceptions to correct
Each one names the frame or try-this that corrects it (README lesson 4).
- **Misconception:** "2026 models are a new architecture." → **Reality:** the forward pass is GPT-3's:
  embed, N blocks of normalize-attend-add and normalize-MLP-add, unembed. Each part was swapped for a cheaper
  or more stable version; the shape stayed. (01 §1 table; `decoder-anatomy`) · corrected by frames 1 and 10
- **Misconception:** "RMSNorm is a different kind of normalization." → **Reality:** it is LayerNorm without
  subtracting the mean (and without the shift β): on the row "sat", LayerNorm gives
  [−0.630, 1.386, 0.378, …], RMSNorm divides by 0.586 and keeps zeros at zero. Cheaper, same quality. (01 §1
  "Norm" row) · corrected by frame 2
- **Misconception:** "SwiGLU makes the MLP bigger." → **Reality:** a gated MLP has three matrices instead of
  two, so models shrink the hidden width from 4 × d_model to about 8/3 × d_model; at GPT-3's width both are
  exactly 1,207,959,552 parameters per block. (01 §1 "MLP" row; first principles) · corrected by frame 4,
  try-this 1
- **Misconception:** "Most of the modern changes are about quality." → **Reality:** most are about cost.
  Sharing keys and values cuts GPT-3's cache from 4,718,592 B (4.72 MB) to 393,216 B (393 kB) per token, removing the position table
  removes a hard 2,048-token limit, and experts decouple stored parameters from per-token work. (01 §1
  "What changed and why" column) · corrected by frames 3, 5, 6 and try-this 2

## 3. Hook and intuition (final wording)
**Hook:** GPT-3's block from 2020 and a 2026 block do the same job: normalize, attend, add; normalize, MLP,
add. So what changed in each part, and why did each change win?

Lay GPT-3's block next to a 2026 one and almost every box has been replaced, but the wiring is the same.
The changes come in three kinds. Some make training stable at depth and scale: normalizing before each half
(GPT-3 already did), using the cheaper RMSNorm, and normalizing queries and keys so attention scores cannot
blow up. Some make the model better per unit of compute: a gated MLP, and experts that let a model store far
more than it uses per token. And many make it cheaper to run: queries share keys and values or read them
from a small latent, position is a rotation instead of a fixed table, and some attention layers look only
at a window or keep a fixed-size state.

The pattern is that serving cost shaped the 2026 block. The parts that changed most are the ones that set
what a model costs per token and per conversation: active parameters and the KV cache. Not every model takes
every change: gpt-oss keeps biases, Gemma normalizes before and after, Kimi K3 drops positions entirely in
some layers. And the newest changes are not in the block at all: models predict more than one token, train
with a new optimizer and store weights in fewer bits.

## 4. Visual metaphor
Two `blockStack`s (`decoder-anatomy`'s accepted glyph, `shown = 1` block each): left "GPT-3 (2020)", right
"2026" which starts as a copy of the left and changes one part per frame. Each part is a `block` labeled with
its component; a swapped part flips from GPT-3's label to the new one and carries a one-line plain text
"why". The followed part (the one being swapped) carries the selection outline; the stream lane and `adder`s
stay fixed.

**Numbers.** Frame 2 uses `decoder-anatomy`'s `x_sat = [0, 1, 0.5, 0, −0.5, 1, 0, 0.5]`; frame 7 uses
`attention`'s q_sat and head A keys for The, cat, sat; γ = 1 and β = 0 throughout. Parameter and cache counts
are exact, from `math/params.js` and `math/memory.js` on GPT-3's config (`decoder-anatomy` §6 preset `gpt3`).
A visible line under the stage (README lesson 10): "Rows are the earlier pages' hand-picked numbers;
parameter and cache counts are exact for GPT-3's shape."

**Terms introduced, one per frame** (README lesson 3): 1 none (GPT-3's block, the reference) · 2 RMSNorm ·
3 RoPE (named; opened in `rope`) · 4 gated MLP (SwiGLU) · 5 none (experts, from `decoder-anatomy`; opened in
`moe`) · 6 GQA / MLA (named; opened in `kv-compression`) · 7 QK-norm · 8 attention sink (named; opened in
`long-context-attention`) · 9 hybrid stack · 10 none (side by side). **Terms assumed from prereqs:** block,
residual stream, normalize, attention, MLP, expert, KV cache, active parameters (`decoder-anatomy`); query,
key, score, softmax are assumed from the course entry requirement (spec §1) and opened on `attention`
(a sibling page, linked in frame 7); frame 7's weights row is the one
`decoder-anatomy` frame 4 already showed ("head A of 2"), so it is familiar even before `attention`. **Named and
deferred:** mHC and Attention Residuals (residual redesigns, named only), MTP (`sampling`), Muon
(`scaling-laws`), FP8/FP4 (`quantization`, `gpu-primer`).

**Layout** (stage ≈ 580 × 366): the two block stacks side by side (each about 180 × 260 px) on the left two
thirds; the right third holds the frame's evidence: frame 6 a byte readout, frame 7 three 3-cell weight rows.

Stage budget (README lesson 18): frame 2's two 8-cell rows need 8 × 43 = 344 px at `NUMBER_CELL`, too wide
beside the stacks. Frame 2 therefore moves the left stack off-stage (it slides out and back in frame 3) and
prints both rows at `NUMBER_CELL` across the right 360 px. Frame 7's rows are 3 cells (129 px) and fit
beside the stacks.

Glyphs used (from spec §5.1 and accepted / proposed elsewhere): blockStack, block, adder, vector, flow (carry
`activation`), kvStack (frame 6), `dial` (frame 3; accepted on `rope`'s review; fallback if the gallery lands
later: a plain "↻" text mark on the q and k arrows).
New glyphs proposed: none. Small marks (README lesson 15): the "why" lines, the "×N" labels, the byte and
parameter readouts, frame 4's "×" gate symbol and the frame 10 page-text table are plain text labels.

Color: vector cells on the value scale (maxAbs 2 for the norm rows, 1 for weights). Swapped parts use the
Architecture accent `active` state once swapped; unchanged parts `idle`. The selection outline marks only the
part being swapped this frame.

## 5. Animation script
| # | On screen | What moves | Caption (final wording) | Numbers shown |
|---|---|---|---|---|
| 1 | Two identical blocks labeled "GPT-3 (2020)": input "+ learned position table [2,048 × 12,288]", LayerNorm → attention "96 heads, each with its own K and V" → ⊕, LayerNorm → MLP "GELU, 4 × wider" → ⊕; a "biases" tag on every matrix. | The left stack draws; the right copy fades in beside it. | This is GPT-3's block from 2020. A 2026 block keeps the same wiring, and almost every box has been swapped. | GPT-3: 96 blocks · d_model 12,288 · 174.6B parameters (`decoder-anatomy`'s count) · 4,718,592 B (4.72 MB) of cache per token |
| 2 | The left stack slides off. Row "sat" twice at `NUMBER_CELL`: "LayerNorm" (subtract the mean 0.3125, divide by the spread 0.496) and "RMSNorm" (divide by the root mean square 0.586). The right block's norm boxes relabel "RMSNorm". Plain label "why normalize: keeps the row's size steady as the stack gets deep". | The two rows type in; the zeros in the RMSNorm row stay zero while the LayerNorm row shifts them. | RMSNorm skips subtracting the mean and just rescales the row. It is cheaper and trains as well, so most 2026 models use it. | x = [0, 1, 0.5, 0, −0.5, 1, 0, 0.5] · LayerNorm: [−0.630, 1.386, 0.378, −0.630, −1.638, 1.386, −0.630, 0.378] · RMSNorm: [0, 1.706, 0.853, 0, −0.853, 1.706, 0, 0.853] |
| 3 | The left stack slides back. The "+ learned position table" box on the input fades out; a small `dial` appears on the q and k arrows inside the right attention box. "why: depends on distance, no 2,048 limit". | The table box fades; the dial turns once. | The position table is gone. RoPE turns each query and key by its position inside attention, so scores depend on distance and no table can run out. | GPT-3's table: 2,048 × 12,288 = 25,165,824 parameters, and nothing for position 2,049 · RoPE: 0 parameters (`rope`) |
| 4 | The right MLP box opens: GELU's two matrices "W_in [d × 4d], W_out [4d × d]" become SwiGLU's three "W_in, W_gate [d × 8/3 d], W_out"; a small "×" gate symbol joins the two branches. Parameter readout under each. | The box widens into two branches that multiply, then narrows. | The MLP gains a gate: one branch decides how much of the other passes through. It uses three matrices, so the hidden width shrinks to keep the size the same. | GPT-3 width: GELU 2 × 12,288 × 49,152 = 1,207,959,552 · SwiGLU 3 × 12,288 × 32,768 = 1,207,959,552 (no biases) |
| 5 | The right MLP box splits into a router and a grid of 64 small expert tiles, 8 lit (top-8). Label "most layers in 2026 models"; beside the readout the label "illustrative: not a real model". | The box splits; eight experts light. | In most 2026 models the MLP becomes a Mixture of Experts. The model stores far more than each token uses. | GPT-3's shape, 64 experts of hidden 4,096, top-8: 960B stored, 148B active, the dense version's 148B plus the router (exact text below) |
| 6 | The right attention box: 96 KV tiles collapse to 8 (`kvStack`s), each shared by 12 query heads; readout "cache per token 4,718,592 B → 393,216 B (4.72 MB → 393 kB)". Label "or a small latent (MLA)". | The 96 small stacks merge in twelves into 8. | Query heads now share their keys and values, or rebuild them from a small latent. GPT-3's shape with 8 shared sets stores 12 times less per token. | 2 × 96 × 96 × 128 × 2 B = 4,718,592 B → 2 × 96 × 8 × 128 × 2 B = 393,216 B (`kv-compression`) |
| 7 | Inside the right attention box: q_sat and the keys The, cat, sat. Row 1 "plain": weights [0.095, 0.703, 0.202]. Row 2 "q grew 10×": [0.000, 1.000, 0.000]. Row 3 "with QK-norm": [0.055, 0.765, 0.180], and the same when q is 10× larger. Visible line under the stage: "This row is `attention`'s hero row; that page computes it step by step." | The q vector inflates; row 2 snaps to one-hot; then a "norm" ring wraps q and k and row 3 settles and stays put as q inflates again. | QK-norm normalizes queries and keys before the dot product. However large they grow during training, the scores stay in a fixed range, so softmax stays soft. | plain scores ÷ 2: [−0.5, 1.5, 0.25] · q × 10: [−5, 15, 2.5] → [0.000, 1.000, 0.000] · normalized (rms q 1.031; k 0.612, 0.791, 0.612): [−0.792, 1.841, 0.396] → [0.055, 0.765, 0.180], unchanged at q × 10 |
| 8 | Inside the right attention box, an extra cell "sink" joins the softmax, labeled "a learned 'nothing here' score per head"; link chip `long-context-attention`. | The sink cell slides in beside the weights. | Some models add a learned sink score per head, so softmax can put its weight on "nothing". It matters most in window layers. | gpt-oss, MiMo-V2, DeepSeek-V4 use one per head · numbers and the window story: `long-context-attention` |
| 9 | The right stack grows to four blocks: three with attention relabeled "linear" or "window", one "full". Page text: "Qwen3.8: 3 linear : 1 full · Kimi K3: 69 linear + 24 full · gpt-oss: window and full alternate". | Blocks stack up; three of four attention boxes relabel. | Attention layers are no longer all alike. Many 2026 stacks mix full attention with window or linear layers that keep far less memory. | only full layers grow a cache · Qwen3.5-397B: 15 of 60 full · (`long-context-attention`) |
| 10 | **Key frame.** Both stacks at full size, side by side, every swapped box in the `active` state, the GPT-3 labels on the left. Page text under the stage: the "beyond the block" table and the visible residual note. | The swapped boxes pulse once in order 2 → 9. | Same wiring, new parts: RMSNorm, rotated positions, a gated MLP or experts, shared keys and values, and stabilized softmax. Most swaps cut what a token costs to run. | GPT-3 175B, 4.72 MB per token → same shape with every swap: 960B stored, 148B active, 393 kB per token, no position limit |

Stage cells print the norm rows and frame 7's scores at 2 d.p. (−0.63 … ; −0.79, 1.84, 0.40); the 3 d.p. values above are printed in the page text under the stage.

Frame 5's "Numbers shown", exactly: "GPT-3's shape with SwiGLU, no biases, 8 KV heads, then 64 experts of
hidden 4,096 (an eighth of 32,768), top-8: total 959,815,311,360 (960B), active 148,066,492,416 (148B) vs
147,990,994,944 dense; the difference is the router."

Frame 10 page text (under the stage):
```text
beyond the block (not drawn)        what                                     lesson
multi-token prediction              extra head drafts the token after next   sampling
Muon optimizer                      replaces AdamW in V4, Kimi, GLM-5        scaling-laws
FP8 training, FP4/MXFP4 weights     fewer bits per number                    quantization
residual redesigns (mHC, AttnRes)   the ⊕ itself is being redesigned (2026)  named only
```
Visible note on the residual row: "New in 2026 (DeepSeek-V4, Kimi K3); this course does not cover how they
work."

Determinism: every frame is a pure function of (step, progress). Reduced motion shows each frame's end state.
The right stack keeps its position from frame 1 to frame 10; the left stack is off-stage only in frame 2.

Caption word counts (README lesson 2; ≤ 30 words, ≤ 2 sentences, no operators): 20 · 23 · 27 · 30 · 20 ·
27 · 26 · 23 · 23 · 28.

Absolutes checked (README lesson 7): "most 2026 models use it" (frame 2; Gemma normalizes before and after,
01 §1); "most 2026 models" (frame 5); "Many 2026 stacks" (frame 9; DeepSeek, GLM, MiniMax use sparse or
compressed softmax attention instead, 01 §2); frame 7's "stays soft" is about the score scale QK-norm
bounds (01 §1 "Prevents attention-logit blowups"); frame 3's "no table to run out of" is the absence of a
learned table (stretching beyond the trained length is `rope`'s frame 8, linked).

Branches (README lesson 13): frame 2 moves the left stack off-stage and says so in its layout; frames 5 and
9 are labeled "most layers in 2026 models" and "many 2026 stacks".

## 6. Toy
Title on page: "Modernize GPT-3, one switch at a time." One state object, one `render()`. Each switch applies
to GPT-3's shape (96 blocks, d_model 12,288, 96 query heads × 128, vocabulary 50,257, tied embeddings).

**Controls**
| id | Label | Type | Range / values | Default | Presets |
|---|---|---|---|---|---|
| `norm` | Norm | toggle | LayerNorm · RMSNorm | LayerNorm | — |
| `position` | Position | toggle | learned table (2,048) · RoPE | learned | — |
| `mlp` | MLP | toggle | GELU, 4 × d · SwiGLU, 8/3 × d | GELU | — |
| `biases` | Biases | toggle | on · off | on | — |
| `kvHeads` | KV heads | segmented | 96 (GPT-3) · 8 · 1 | 96 | — |
| `experts` | Experts (illustrative: not a real model) | segmented | dense · 64 experts, top-8 (each an eighth of the SwiGLU width) | dense | experts requires SwiGLU; switching it on flips `mlp` |
| chips | — | preset chips | "GPT-3 (2020)" (all off) · "2026-style" (all on, 8 KV heads) | GPT-3 | — |

**Live outputs**
| Output | Formula / `math/` function | Units / format |
|---|---|---|
| Total and active parameters | `paramBreakdown(config)` from `math/params.js` (`.total`, `.active`) | `formatCount` and exact |
| The active definition under it, verbatim from `decoder-anatomy` §6 (README lesson 16) | — | text |
| Change vs GPT-3 | `publishedGap(total, gpt3Total)` from `math/params.js` (signed relative gap; one definition) | % |
| Attention, MLP / experts, norms, positional parameters | `.parts` | counts |
| Cache per token | `kvBytesPerToken({ layers: 96, kvHeads, headDim: 128, bytesPerElem: 2 })` from `math/memory.js` | exact bytes + `formatBytes` (decimal) |
| Longest input | learned → "2,048 (table size)"; RoPE → "set by training; see `rope`" | text |
| Parts table: which switch changed which number | templated | mono |

**Try this** (each leads to a named insight)
1. Flip **norm**: total 174,604,259,328 → 174,601,887,744 (2,371,584 fewer: the β vectors). Flip **MLP**:
   the MLP parts go 115,970,015,232 → 115,971,588,096 (the bias vectors differ; without biases both are
   96 × 1,207,959,552). → **Insight: RMSNorm and SwiGLU change how the numbers flow, not how many there are;
   they won on stability and quality per FLOP, not size.**
2. Flip **position** to RoPE: 25,165,824 fewer parameters and "longest input: set by training". Set **KV
   heads** to 8: cache 4,718,592 → 393,216 B per token (12× less). Turn biases off too, and the total is
   147,990,994,944 because W_K and W_V shrink. → **Insight: the big 2023–26 changes are about running cost: no position
   limit, a 12× smaller cache.** `kv-cache` lets you build this number by hand; `kv-compression` shows how
   the sharing works.
3. With every other swap on, turn **experts** on: total 147,990,994,944 → 959,815,311,360; active 147,990,994,944 → 148,066,492,416
   (+75,497,472: the router). Tap **2026-style**, then **GPT-3 (2020)**, and read the parts table. →
   **Insight: experts multiply what is stored, not what each token uses, and every other swap kept the block's
   shape.**

**Functions.** None new. `math/params.js` (`paramBreakdown` with `norm: 'layernorm' | 'rmsnorm'`,
`positional: 'learned' | 'rope'`, `mlp: { kind, hidden }`, `biases`, `attention: { kind: 'gqa', nHeads,
nKvHeads, dHead }`, `moe`) and `math/memory.js` (`kvBytesPerToken`) as specified on `decoder-anatomy` and
`kv-cache`. The two worked rows reuse `core.softmax` and an `rmsNorm(row)` / `layerNorm(row)` pair:

```js
// math/norms.js (accepted by the expert review; owned by this page):
rmsNorm(row, { gamma = 1 } = {}) → number[]          // row / sqrt(mean(row²)) · γ
//   ([0, 1, 0.5, 0, −0.5, 1, 0, 0.5]) → [0, 1.706, 0.853, 0, −0.853, 1.706, 0, 0.853]   (rms 0.586)
//   ([0, 2, 0.5, 0])                  → [0, 1.940, 0.485, 0]                             (rms 1.031; frame 7's q)
layerNorm(row, { gamma = 1, beta = 0 } = {}) → number[]  // (row − mean) / sqrt(var) · γ + β
//   ([0, 1, 0.5, 0, −0.5, 1, 0, 0.5]) → [−0.630, 1.386, 0.378, −0.630, −1.638, 1.386, −0.630, 0.378]   (mean 0.3125, sd 0.496)
```
QK-norm's worked row (frame 7): `softmax(rmsNorm(q) · rmsNorm(k_j) / 2)` for j = The, cat, sat →
[0.055, 0.765, 0.180]; identical for `q × 10`. Real QK-norm layers carry a learned γ; the toy uses γ = 1.

Worked `paramBreakdown` states (GPT-3 config, switches applied cumulatively in the order of the table):
GPT-3 174,604,259,328 · + RMSNorm 174,601,887,744 · + RoPE 174,576,721,920 · + SwiGLU 174,578,294,784 ·
+ no biases 174,566,105,088 · + 8 KV heads 147,990,994,944 · + 64 experts top-8 959,815,311,360 total /
148,066,492,416 active. Cache per token 4,718,592 B until KV heads change, then 393,216 B.

Tests to write first: `rmsNorm` output has root mean square 1 (γ = 1); `layerNorm` output has mean 0 and
variance 1; the examples above; GELU 4d and SwiGLU 8/3·d MLPs are equal without biases at d = 12,288 (the
frame 4 claim as a test, README lesson 16); with experts on, active − dense active equals the router
(96 × 12,288 × 64 = 75,497,472; lesson 17); the switch states reproduce the list above.

**Reproducer** (run from the repo root on 2026-10-07; `decoder-anatomy`'s parameter formulas inlined; output
matched every number in §2, §5, §6 and §11):
```sh
node -e '
import("./math/core.js").then(({softmax})=>{
const r=(x,d=3)=>Array.isArray(x)?x.map(v=>r(v,d)):Number(x.toFixed(d));
const x=[0,1,0.5,0,-0.5,1,0,0.5],mu=x.reduce((a,b)=>a+b)/8,sd=Math.sqrt(x.reduce((a,b)=>a+(b-mu)**2,0)/8),rms=v=>Math.sqrt(v.reduce((a,b)=>a+b*b,0)/v.length);
console.log("mu",mu,"sd",r(sd),"LN",JSON.stringify(r(x.map(v=>(v-mu)/sd))),"rms",r(rms(x)),"RMS",JSON.stringify(r(x.map(v=>v/rms(x)))));
const q=[0,2,0.5,0],K=[[1,-0.5,0,0.5],[0,1.5,0,-0.5],[-0.5,0,1,0.5]],dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0),rn=v=>v.map(t=>t/rms(v));
const row=(qq,KK)=>softmax(KK.map(k=>dot(qq,k)/2));
console.log("plain",r(row(q,K)),"x10",r(row(q.map(v=>v*10),K)),"rms",r(rms(q)),K.map(k=>r(rms(k))),"norm scores",r(K.map(k=>dot(rn(q),rn(k))/2)),"w",r(row(rn(q),K.map(rn))),"w x10",r(row(rn(q.map(v=>v*10)),K.map(rn))));
const gqa=(d,nH,nKV,dh,b)=>d*nH*dh+2*d*nKV*dh+nH*dh*d+(b?nH*dh+2*nKV*dh+d:0),mlp=(k,d,h,b)=>(k==="swiglu"?3:2)*d*h+(b?(k==="swiglu"?2*h:h)+d:0);
const bd=(c)=>{const d=c.d,L=c.L,b=!!c.b,nm=(c.ln?2:1)*d,attn=gqa(d,...c.gqa,b),moeL=c.moe?L:0,ex=c.moe?mlp("swiglu",d,c.moe.h,b):0;
 const p={emb:c.V*d,pos:c.pos?c.pos*d:0,attn:L*attn,mlp:(L-moeL)*mlp(c.k,d,c.h,b),ex:c.moe?moeL*c.moe.n*ex:0,router:c.moe?moeL*d*c.moe.n:0,norms:L*2*nm+nm};
 const total=Object.values(p).reduce((s,v)=>s+v,0);return [total,total-p.ex+(c.moe?moeL*c.moe.k*ex:0),p.mlp];};
let c={V:50257,d:12288,L:96,gqa:[96,96,128],k:"gelu",h:49152,ln:1,pos:2048,b:1};
for(const [n,ch] of [["GPT-3",{}],["RMSNorm",{ln:0}],["RoPE",{pos:0}],["SwiGLU",{k:"swiglu",h:32768}],["no biases",{b:0}],["GQA-8",{gqa:[96,8,128]}],["MoE",{moe:{n:64,k:8,h:4096}}]]){
 c={...c,...ch};console.log(n,bd(c).join(" "),"kv",2*96*c.gqa[1]*128*2);}
console.log("mlp",mlp("gelu",12288,49152,0),mlp("swiglu",12288,32768,0),"router",96*12288*64,"pos",2048*12288);
})'
```
Output on 2026-10-07: `mu 0.3125 sd 0.496 LN [-0.63,1.386,0.378,-0.63,-1.638,1.386,-0.63,0.378] rms 0.586
RMS [0,1.706,0.853,0,-0.853,1.706,0,0.853]` · `plain [0.095,0.703,0.202] x10 [0,1,0] rms 1.031
[0.612,0.791,0.612] norm scores [-0.792,1.841,0.396] w [0.055,0.765,0.18] w x10 [0.055,0.765,0.18]` ·
`GPT-3 174604259328 174604259328 115970015232 kv 4718592` · `RMSNorm 174601887744 …` · `RoPE 174576721920 …`
· `SwiGLU 174578294784 … 115971588096` · `no biases 174566105088 …` · `GQA-8 147990994944 147990994944 …
kv 393216` · `MoE 959815311360 148066492416 0 kv 393216` · `mlp 1207959552 1207959552 router 75497472 pos
25165824`. (Tied embeddings: the shared matrix is counted once, as on `decoder-anatomy`.)

## 7. Show me the math
```tex
\operatorname{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sigma} + \beta,
\qquad
\htmlClass{hl-rms}{\operatorname{RMSNorm}(x)} = \gamma \odot \frac{x}{\sqrt{\tfrac{1}{d}\sum_j x_j^2}},
\qquad \text{worked: } \operatorname{rms}(x_{\text{sat}}) = \sqrt{2.75/8} = 0.586
```
```tex
\operatorname{GELU\ MLP}(h) = \operatorname{GELU}(h W_{\text{in}})\, W_{\text{out}},\qquad
\htmlClass{hl-glu}{\operatorname{SwiGLU}(h)} = \big(\operatorname{SiLU}(h W_{\text{gate}}) \odot h W_{\text{in}}\big) W_{\text{out}},
\qquad 2 \cdot d \cdot 4d = 3 \cdot d \cdot \tfrac{8}{3}d = 8d^2
```
```tex
\htmlClass{hl-qk}{\text{QK-norm:}}\quad s_{ij} = \frac{\operatorname{RMSNorm}(q_i) \cdot \operatorname{RMSNorm}(k_j)}{\sqrt{d_{\text{head}}}},
\qquad |s_{ij}| \le \frac{d_{\text{head}}\,\gamma_q \gamma_k}{\sqrt{d_{\text{head}}}}
```
```tex
\text{pre-norm block: } x \leftarrow x + \operatorname{Attn}(\operatorname{Norm}(x)),\quad
x \leftarrow x + \operatorname{MLP}(\operatorname{Norm}(x))\qquad (\text{GPT-2/3 onward; the original 2017 transformer normalized after the add})
```
Shapes: x [d_model] (8 here; 12,288 in GPT-3); q, k [d_head] (4); SwiGLU W_in, W_gate [d × h], W_out [h × d].
The QK-norm bound follows from each normalized vector having squared length d_head (with γ = 1, the dot
product is at most d_head). Color links: `hl-rms` → frame 2's RMSNorm row; `hl-glu` → frame 4's gated
branches; `hl-qk` → frame 7's third row. KaTeX with `trust: true, strict: false`.

## 8. In today's models (Oct 2026)
Framing paragraph on the page: "Every part below is in at least one 2026 frontier model; almost none is in all of
them. Sebastian Raschka's architecture gallery compares 100+ models part by part (link below)."

| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| GPT-3 (2020): pre-norm LayerNorm, learned positions (2,048), GELU MLP 4× wide, full multi-head attention, biases, tied embeddings | `models.gpt-3.*` (proposed, `decoder-anatomy` §12.3, `kv-cache` §12) | 01 §1 baseline column (background) |
| Pre-norm RMSNorm in most 2026 models; Gemma normalizes both before and after | — (row of the 01 §1 table, no per-model key) | 01 §1 "Norm" row |
| RoPE in most models; Kimi K3's 24 MLA layers use no position encoding | `models.kimi-k3.attention` (existing, confirmed; "(NoPE)") | 01 §1 "Position" row, §3 [C] |
| QK-norm: Gemma 3, Qwen3, DeepSeek-V4 (V4 uses it instead of Kimi's QK-Clip) | `models.deepseek-v4-pro.qk_norm` = true (proposed, `attention` ruling 2) | 01 §1 "QK-norm" row [C for V4] |
| Biases mostly gone; gpt-oss keeps attention biases | `models.gpt-oss-120b.biases` = true (proposed, `decoder-anatomy` §12.3) | 01 §1 "Biases" row [C] |
| Learned sink per head: gpt-oss, MiMo-V2, DeepSeek-V4 | `models.*.attention_sink` (proposed, `attention` ruling 2) | 01 §1 "Attention sink" row [C] |
| GQA with 8 KV heads typical; MLA in DeepSeek, Kimi, GLM | `models.*.n_kv_heads`, `.mla_kv_rank` (proposed, `kv-compression` §12) | 01 §1 "Attention" row, §2 [C] |
| Hybrid stacks: Qwen3.8 3 Gated DeltaNet : 1 full; Kimi K3 69 linear + 24 MLA | `models.qwen3.8.attention`, `models.kimi-k3.attention` (existing, confirmed) | 01 §2 [C] |
| Residual redesigns in 2026: mHC (DeepSeek-V4), Attention Residuals (Kimi K3) | `models.deepseek-v4-pro.residual` = "mHC", `models.kimi-k3.residual` = "Attention Residuals" (proposed) | 01 §1 "Residual stream" row [C] |
| Multi-token prediction heads: V3/V4 depth 1, K3 1 layer, GLM-5 shares 3 | `models.deepseek-v4-pro.mtp_depth` (in data) | 01 §1 "Decoding" row [C] |
| Muon optimizer (V4, Kimi, GLM-5) | `models.deepseek-v4-pro.optimizer` = "Muon" (existing, confirmed) | 01 §1 "Optimizer" row [C] |
| Precision: FP8 training and FP4 experts (V4), MXFP4 (gpt-oss, K3) | `models.deepseek-v4-pro.pretrain_precision` = "FP8; FP4 experts (QAT)" | 01 §1 "Precision" row [C] |

## 9. Takeaways
1. The 2026 block has GPT-3's wiring: normalize, attend, add; normalize, MLP, add. Almost every box was
   swapped (frames 1, 10).
2. RMSNorm, SwiGLU and QK-norm are about stable, efficient training: same size, better behaved (frames 2, 4,
   7, try-this 1).
3. RoPE, shared keys and values or latents, window and linear layers, and experts are about running cost: no
   position table, a smaller cache, less work per token (frames 3, 5, 6, 9, try-this 2–3).

## 10. Next and go deeper
Next: `moe` (open the experts box), `sampling` (what happens after the last block), `training-pipeline` (how
a 2026 model is trained); all three list `decoder-recap` in `shared/concepts.json`. In-page links to the
lessons that open each box: `rope`, `kv-compression`, `long-context-attention`, `quantization`,
`scaling-laws`.

Go deeper (brief 05 §1.2; 01 §1, §8): Sebastian Raschka, LLM Architecture Gallery
(https://sebastianraschka.com/llm-architecture-gallery/), moved here from `decoder-anatomy` by its expert
review · Sebastian Raschka, *The Big LLM Architecture Comparison*
(https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) · Transformer Explainer
(https://poloclub.github.io/transformer-explainer/), to see GPT-2's block, the baseline this page starts from.

## 11. Key-frame sketch
Frame 10, desktop width; numbers from the §6 reproducer.
```text
┌──────────────────────────────────────────────────────────┐
│  GPT-3 (2020)                 2026                       │
│  + position table [2,048]     (no table)                 │
│  ║                            ║                          │
│  LayerNorm ─▶ attention       ■RMSNorm ─▶ attention      │
│     96 heads, own K,V          ■8 shared KV (or latent)  │
│                                ■QK-norm ■sink  ◔ RoPE    │
│  ⊕                            ⊕                          │
│  LayerNorm ─▶ MLP GELU 4×     ■RMSNorm ─▶ ■experts       │
│                                 (SwiGLU, top-8 of 64)    │
│  ⊕  × 96                      ⊕  × 96 (some: window or   │
│                                    linear attention)     │
│  175B · 4.72 MB/token         960B stored · 148B active  │
│                               393 kB/token               │
├──────────────────────────────────────────────────────────┤
│ Same wiring, new parts: RMSNorm, rotated positions, a    │
│ gated MLP or experts, shared keys and values, and        │
│ stabilized softmax. Most swaps cut what a token costs to │
│ run.                                                     │
│ [◄] [Pause] [►]  ━━━━━━━━━━●  10 / 10  speed [1×]        │
└──────────────────────────────────────────────────────────┘
```
■ marks a swapped part (`active` state), ◔ the `dial`. The "beyond the block" table is page text under the
stage. At 400 px the two stacks sit one above the other.

## 12. Open questions for the reviewer
**Data-pass keys** (*new* unless noted): `residual` (deepseek-v4-pro "mHC", kimi-k3 "Attention
Residuals"), `pretrain_precision` (deepseek-v4-pro), `qk_norm` and `attention_sink` (reuse `attention` ruling 2),
`biases` (reuse `decoder-anatomy`). Gemma 3 / Qwen3 QK-norm are background rows without entries.

**Graph changes:** none.

**Judgment calls:** all ruled by the expert review (§13) and applied; none remain open.

## 13. Expert review (2026-10-07) and what changed
Verdict: APPROVE WITH CHANGES (1 Must). Status is now "approved (expert review)". No number changed except
units; the §6 reproducer was re-run and still matches.

Rulings applied (lesson 20): `math/norms.js` is accepted as this page's module (`rmsNorm`, `layerNorm`);
frame 5's 64-expert GPT-3 is accepted as illustrative and labeled "not a real model" on stage and in the toy;
the SwiGLU paper (not in any brief) is swapped for Transformer Explainer (brief 05).

Must (1/1): decimal units (misconception 4, frames 1, 6, 10, the toy output, §11).
Should (6/6, no rebuttals): frame 7 visible line and `attention` named under terms assumed; frame 7 caption
"the scores stay in a fixed range, so softmax stays soft"; try-this 2 links to `kv-cache` and
`kv-compression`; "illustrative: not a real model" on frame 5 and the toy; §8 "almost none is in all of
them"; the `dial` fallback ("↻") written in §4.
Nice (2/2): frame 2 prints why normalizing helps; frame 4's "×" listed as a plain text mark.
- Data pass 2026-10-07: DeepSeek-V4-Pro precision cites `pretrain_precision`; no numbers changed.
