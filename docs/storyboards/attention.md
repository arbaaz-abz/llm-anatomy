# Attention, step by step (`attention`)

Track: architecture · Section: architecture · Prereqs: decoder-anatomy
Status: approved (expert review)
Sources: 01 §1, §2, §4, §5, §7, §8 · 05 §1–2

This is the course's reference page. Every later lesson that touches attention (`rope`, `kv-cache`,
`kv-compression`, `long-context-attention`, `multimodal`) reuses this page's four tokens, its glyph
layout and its vocabulary (scores → scale → mask → softmax → weighted sum: five steps).

## 1. Learning objective
After this page you can compute one row of attention by hand (query · keys, ÷ √d_head, mask the future
to −∞, softmax, weighted sum of values), read an attention heatmap (rows = queries, columns = keys), and
explain why a layer runs several heads: several attention patterns over the same tokens at once.

## 2. Misconceptions to correct
Each misconception names the frame or try-this that corrects it in the first build.
- **Misconception:** "Attention weights are parameters the model learned." → **Reality:** the weights
  (the heatmap) are recomputed for every input from Q·Kᵀ. The learned parts are the projection matrices
  W_Q, W_K, W_V, W_O. Frame 9 and try-this 3 show it: the same four tokens give a different heatmap in
  head B only because head B has different W matrices. (source: first principles; the learned per-head
  projections are named in brief 01 §1, "Attention" row) · corrected by frame 9, try-this 3
- **Misconception:** "The causal mask zeroes out the future scores." → **Reality:** it sets them to −∞
  *before* softmax, so their weight is exactly 0 and the visible weights still sum to 1. A score of 0
  still becomes e⁰ = 1 before dividing by the sum, so the future would keep a real share. Masking *after*
  softmax would leave the row summing to less than 1. (source: first principles; `core.softmax` test
  "masked entries are −Infinity") · corrected by frames 5–6 (the masked cell stays 0 while Σ counts to
  1.000), try-this 1
- **Misconception:** "Softmax picks the best token." → **Reality:** the output is a blend. In the worked
  row, "sat" takes 70% of cat's value but also 10% of The's and 20% of its own. Only an extreme divisor
  (try 0.5 in the toy) makes it near one-hot. (source: first principles) · corrected by frames 6–7,
  try-this 2
- **Misconception:** "Multi-head splits the tokens between heads." → **Reality:** every head sees the
  same tokens (all of them, in full attention). Heads differ in their projection matrices, so they
  produce different patterns; their outputs are concatenated and mixed by W_O. (source: brief 01 §1,
  "Attention" row; §2 MHA paragraph) · corrected by frames 9–10, try-this 3
- **Misconception:** "Earlier tokens' keys and values change as generation goes on." → **Reality:** with
  a causal mask, a token's K and V depend only on itself and its past, so they never change once
  computed (brief 01 §4: "K and V of earlier tokens never change (causal mask)"). That is the reason the
  KV cache works; you'll meet it in `kv-cache`. · corrected within one layer by frames 1 and 5: each
  token's K and V row comes from that token alone, and no masked cell feeds an earlier row

## 3. Hook and intuition (final wording)
**Hook:** When a model reads "The cat sat down", how does "sat" know to look at "cat", and why can it
never look at "down"?

Each token in an attention layer makes three small vectors from its current vector (in the first layer,
its embedding): a **query** (what it is looking for), a **key** (what other queries match against) and a
**value** (what it hands over if chosen). The three come from three learned matrices, W_Q, W_K and W_V,
multiplied with the same input vector. Each has d_head numbers; 4 in this toy.

Attention is one scoring pass and one blend. A token's query is dotted with every key, giving one score
per token. The scores are divided by √d_head, the future is masked out, and softmax turns the row into
weights that sum to 1. The token's output is the weighted sum of the values. Five steps, and you can do
them with a calculator; this page makes you do exactly that for one row.

Why forbid the future? The model learns by predicting each next word, and in training the whole sentence
goes in at once. If "sat" could see "down", it would copy the answer instead of learning to predict it.
When the model is generating, "down" doesn't exist yet, so the mask also keeps training and generation
the same.

A **head** is one such pattern over the tokens. A layer runs several heads at once, each with its own
W_Q, W_K, W_V, like looking at the same sentence through several lenses. Their outputs are concatenated
and mixed by one more matrix, W_O, and the result is added back into each token's vector (the residual
stream), where the next block starts. Most later lessons change what feeds these five steps (RoPE), what
is stored between them (KV cache, GQA, MLA) or which keys a query reads (sliding windows, sparse
attention). Two small additions sit around the softmax (QK-norm and a sink logit), and some 2026 layers
swap softmax attention for linear attention altogether (`long-context-attention`).

Footnote on the page (visible, under the prose): "We treat each word as one token here; how text is cut
into tokens is covered in `pretraining`."

## 4. Visual metaphor
Toy dimensions: n = 4 tokens ("The", "cat", "sat", "down"), d_model = 8, H = 2 heads, d_head = 4.
So √d_head = 2 and the concatenated head outputs are [4 × 8]. (Spec §4's "d_model 8, 2 heads, ≤ 8
tokens".)

**Numbers.** The Q, K, V matrices for both heads are hand-authored constants in `math/attention.js`
(`TOY`), chosen on a quarter grid (−1 … 3) so that every dot product is a one-line sum and the two heads
show visibly different patterns: head A puts most weight on "cat" for every later token; head B puts most
weight on the previous token. They are not RNG output and are not computed from an embedding on this
page; frame 1's caption and a visible line above the toy say so. A "from embeddings" preset (seeded
`randomMatrix` + `projectQKV`) is **deferred** until after the pilot build (§13, ruling 1); nothing on the
page depends on it.

**Terms introduced** (one new term or symbol per frame, defined on screen where it first appears):
query / key / value as one trio (frame 1, with "d_head = 4" as the shape label), score (frame 2), the
score grid S with rows = queries and columns = keys (frame 3), √d_head scaling (frame 4), causal mask and
−∞ (frame 5), attention weights (frame 6), output (frame 7), head and attention pattern (frame 8), a
second head (frame 9), joining heads and W_O (frame 10).
**Terms assumed from prereqs** (`decoder-anatomy`): token, embedding, the residual stream, block / layer,
d_model, softmax as "turns scores into probabilities that add to 1" (decoder-anatomy frame 8; frame 6
here shows the arithmetic).

**Layout** (stage ≈ 580 × 366, fixed-size SVG; scrolls inside its container at 400 px):
- Top: four `token` chips with subscripts 1–4. On-screen token positions are 1-based course-wide.
- Left: three `matrix` glyphs **Q**, **K**, **V** [4 × 4] at 20 px cells (values on hover/tap only),
  rows labeled by token, each with a label chip (Q / K / V) above it and the shape label
  "[4 tokens × d_head = 4]". Their numbers are never load-bearing on their own: every number the learner
  must read is printed at `NUMBER_CELL` in the lifted row, the frame-2 expansion line, the heatmap, or the
  toy's "Check my work" box.
- Middle: the score grid **S** [4 × 4] drawn with `heatmap` at `NUMBER_CELL` (40 px) so its 16 numbers
  print; rows = queries, columns = keys; masked cells hatched. It is the same glyph through frames 3–8;
  only its numbers change. Plain text axis labels "queries ↓" and "keys →".
- Right: the "row strip": the lifted-out row for the followed query (`vector`, orient `row`,
  `NUMBER_CELL`) in up to four states stacked vertically: score, scaled, exp, weight. In frame 2 the
  current K row is printed beside the lifted q at the same size, so the dot product is readable without
  hover.
- Bottom right: the output `vector` for the followed query; in frames 9–10, two `block` glyphs (head A,
  head B) with their heatmaps and a `block` labeled "W_O [8 × 8]".
- `flow` arrows carry the moving dot from the Q row to the K rows (frame 2), from the weights row to the
  V rows (frame 7), and from the joined outputs into W_O (frame 10).
- **The followed query is marked the same way in every frame:** row "sat" carries the selection outline
  (Architecture accent) in Q, in S / A, in the row strip and in O, frames 1–10. Outlines mean selection
  only; amounts are encoded by cell fill and printed numbers.
- Each quantity is encoded once: a weight appears as a colored cell in the weights row, and the V row it
  applies to is linked by a small chip in the same cell color (frame 7), not by re-drawing the row's
  opacity or outline.

Glyphs used (from spec §5.1): token, vector, matrix, heatmap (masked cells hatched), block (head / W_O),
flow (carry = activation).
New glyphs proposed: none. The "lift a row out of a matrix" motion is a transform of an existing `vector`
glyph, not a new drawing. The exp values in frame 6 are drawn as a row `vector` on the value scale, not
as bars. Small marks are all plain text labels: "÷ √d_head = ÷ √4 = ÷ 2" (frame 4), the "Σ = 1.000"
readout (frame 6), the weight chips beside V rows (frame 7: `vector` cells of length 1), the row-1
annotation (frame 8), the "→ added to the residual stream" label (frame 10).

Color: all numbers use the diverging value scale with `maxAbs = 4.5` for S and 3 for Q/K/V (so the one
score of 3.0 reads as strongly positive without saturating). Q/K/V/S/A/O identity is carried by label
chips and position, never by hue. Hovering a term in the math panel outlines the matching glyph with the
Architecture accent.

## 5. Animation script
Hero row: query "sat" (row 3), head A. Numbers are 3 d.p. unless exact. Token positions are 1-based.
Captions: one idea, at most two sentences and 30 words, no formulas or operators (formulas live in
"Numbers shown" and §7).

| # | On screen | What moves | Caption (final wording) | Numbers shown |
|---|---|---|---|---|
| 1 | Four token chips. Below them, Q, K, V matrices [4 × 4] of head A, each labeled, with a dim block "W_Q, W_K, W_V [8 × 4]" between the chips and the matrices. Shape label "[4 tokens × d_head = 4]". Row "sat" outlined. | A flow dot leaves each chip, passes the W block, and the chip's row in Q, K and V fills in (rows fill top to bottom, progress 0 → 1). | In a real model, three learned matrices turn each token's vector into a query, a key and a value. Here they are hand-picked so you can check every number. | Q = [[1,0,−0.5,0],[0.5,0.5,0,−1],[0,2,0.5,0],[0.5,2,0,0]] · K = [[1,−0.5,0,0.5],[0,1.5,0,−0.5],[−0.5,0,1,0.5],[0.5,−0.5,0.5,1]] · V = [[1,0,−1,0],[0,2,0,1],[−1,0,1,0.5],[0.5,−1,0,1]] · d_head = 4 |
| 2 | Row "sat" of Q lifts out to the row strip at `NUMBER_CELL`. The current K row is printed beside it at the same size. An empty score row (4 cells) under the K matrix. | The lifted q slides down the K rows; as it passes each row, that K row prints beside q, a flow dot lands in the score cell and the number appears (one key per quarter of progress). A mono line expands the "cat" product. | A score is the query dotted with a key. Against the four keys, "sat" scores −1.0, 3.0, 0.5 and −0.75; the biggest is "cat". | q_sat = [0, 2, 0.5, 0] · scores −1.0, 3.0, 0.5, −0.75 · expansion: 0·0 + 2·1.5 + 0.5·0 + 0·(−0.5) = 3.0 |
| 3 | The score row docks into row 3 of the heatmap S. The other three rows fill. Label "S = QKᵀ [4 × 4]". | Rows 1, 2, 4 of S fill left to right together (one matrix product, not three more passes). Axis labels "queries ↓" and "keys →" fade in. | Every query scores every key at once, in one matrix product. The grid has one row per query and one column per key. | S = QKᵀ = [[1,0,−1,0.25],[−0.25,1.25,−0.75,−1],[−1,3,0.5,−0.75],[−0.5,3,−0.25,−0.75]] |
| 4 | Same grid; a plain text label "÷ √d_head = ÷ √4 = ÷ 2" above it. | Every cell's number and color interpolate from the raw score to half of it. Nothing else moves. | Divide every score by the square root of the vector length: 4 numbers, so divide by 2. Longer vectors give bigger scores; dividing keeps softmax soft. | S/√d_head = S/2 = [[0.5,0,−0.5,0.125],[−0.125,0.625,−0.375,−0.5],[−0.5,1.5,0.25,−0.375],[−0.25,1.5,−0.125,−0.375]] |
| 5 | Same grid. | The six cells above the diagonal hatch in, one anti-diagonal at a time, and their numbers become −∞. | Causal mask: a token may only look at itself and earlier tokens. Future cells become −∞, so softmax will give them exactly 0. | masked S/2; cells (1,2),(1,3),(1,4),(2,3),(2,4),(3,4) = −∞ |
| 6 | Row "sat" lifts out again to the row strip, which now shows three stacked rows: scaled, exp, weight. | First the exp row fills (cells on the value scale, numbers typed in), then the weight row fills as each exp cell is divided by the sum; a plain text "Σ = 1.000" readout counts up to 1. The masked cell shows 0 throughout. | Softmax: exponentiate each score, then divide by the sum so the row adds to 1. "sat" now puts 70% of its attention on "cat". | scaled [−0.5, 1.5, 0.25, −∞] · exp [0.607, 4.482, 1.284, 0] · sum 6.372 · weights [0.095, 0.703, 0.202, 0] |
| 7 | The weights row stays. Beside V rows 1–3 a one-cell chip in the matching weight color (0.095, 0.703, 0.202); an empty output vector [1 × 4] at the right. | The chips appear beside V rows 1–3, then the three V rows slide together into the output vector, whose cells fill. Row 4 of V gets no chip (weight 0) and stays dim. | The output is the weighted sum of the values. "sat" leaves carrying mostly cat's value. | 0.095·[1,0,−1,0] + 0.703·[0,2,0,1] + 0.202·[−1,0,1,0.5] = [−0.106, 1.407, 0.106, 0.804] |
| 8 | The full heatmap A (weights) replaces the masked-score grid; a [4 × 4] output matrix O_A beside it. A permanent plain text annotation at row 1: "row 1 sees one key, so its weight is 1.0 whatever its score". | Rows 1, 2, 4 of the heatmap recolor from scores to weights and their output rows fill, all together. The annotation fades in with row 1. | Do the same for every row: one head is one attention pattern, the heatmap, and one output vector per token. | A = [[1,0,0,0],[0.321,0.679,0,0],[0.095,0.703,0.202,0],[0.114,0.656,0.129,0.101]] · O_A = [[1,0,−1,0],[0.321,1.358,−0.321,0.679],[−0.106,1.407,0.106,0.804],[0.035,1.212,0.015,0.821]] · every row of A sums to 1.000 |
| 9 | Head A (heatmap + O_A) inside a `block` labeled "head A" slides left; a second `block` "head B" appears beside it with an empty heatmap and O_B. | Head B's heatmap fills (sub-diagonal pattern), then its O_B fills. Nothing else moves. | A second head has its own W_Q, W_K and W_V, so it finds a different pattern: here, each token looks one step back. | B = [[1,0,0,0],[0.798,0.202,0,0],[0.168,0.664,0.168,0],[0.129,0.146,0.578,0.146]] · O_B = [[0,1,0,0],[0.202,0.798,0,0],[0.664,0.168,0,0.168],[0.146,0.129,0.146,0.578]] |
| 10 | Both head blocks stay. Below them an empty [4 × 8] strip, a `block` "W_O [8 × 8]" with a plain text line under it: "each of the 8 outputs is a weighted mix of all 8 joined numbers", and a plain text label "→ added to the residual stream". | O_A and O_B slide together into the [4 × 8] strip; the flow dot runs from the strip into W_O; the two labels fade in. | Heads run side by side. Their outputs are joined into one row of 8 numbers per token and mixed by one more matrix, W_O. | concat row "sat" = [−0.106, 1.407, 0.106, 0.804, 0.664, 0.168, 0, 0.168] · concat [4 × 8] · W_O [8 × 8] |

Determinism: every frame is a pure function of (step, progress). Reduced motion shows each frame's end
state. The heatmap keeps its position and orientation (rows = queries, columns = keys) from frame 3 to
frame 10; the row strip keeps its position from frame 2 to frame 7; the followed row "sat" keeps its
selection outline in every frame.

Nothing load-bearing is behind hover: the row-1 fact is a permanent annotation (frame 8), "Σ = 1.000"
is a printed readout (frame 6), the hand-picked-numbers note is in the frame-1 caption and above the
toy, and the residual-stream hand-off is a printed label (frame 10). Hover/tap on a 20 px Q/K/V cell
only repeats a number that is also printed elsewhere.

## 6. Toy
Title on page: "Compute one row yourself." A visible line directly under the title: "Q, K and V here
are hand-picked so every number checks by hand; a real model computes them from each token's vector
with W_Q, W_K, W_V." The toy shows the same tokens and head A by default, with the hero row selected,
so the learner starts exactly where the animation ended.

**Controls**
| id | Label | Type | Range / values | Default | Presets |
|---|---|---|---|---|---|
| `query` | Query token | segmented control (4 chips) | The₁ · cat₂ · sat₃ · down₄ | sat₃ | — |
| `divisor` | Divide scores by | Slider (snapped, `values`) | [0.5, 1, 2, 4, 8] | 2, labeled "√d_head (the model's value)" | chip "√d_head" resets to 2 |
| `causal` | Causal mask | toggle | on / off | on | — |
| `head` | Head | segmented control | A · B · both | A | — |

Deferred (§13, ruling 1; not in the first build): a "Numbers: hand-picked / from embeddings (seed 1)"
preset pair, where X = `randomMatrix(4, 8, 1)`, head A W_Q/W_K/W_V = `randomMatrix(8, 4, 2|3|4, 0.5)`,
head B seeds 5|6|7, then Q, K, V = `projectQKV(X, W)`. The four controls would apply unchanged.

A visible line under the divisor slider: "This slider is a temperature knob on the attention softmax.
Real models fix it at √d_head; the slider is here so you can feel why." Kept separate in wording from
sampling temperature (lesson `sampling`). Under it, one mono line that makes try-this 2's claim
checkable: "typical score size for random ±1 entries: 4 terms → 2; 128 terms → 11" (√4 = 2,
√128 = 11.3; `node -e 'console.log(Math.sqrt(4), Math.sqrt(128))'` → `2 11.313708498984761`).

**Live outputs** (one state object, one `render()`; every number is tabular mono)
| Output | Formula / `math/` function | Units / format |
|---|---|---|
| Score row for the selected query | `attentionHead(Q, K, V, {causal, divisor}).scores[i]` | 4 numbers, 2 d.p.; future cells dimmed when mask on |
| Scaled row | `.scaled[i]` | 4 numbers, 3 d.p. |
| Masked row | `.masked[i]` | 3 d.p.; masked cells show "−∞" and are hatched |
| Weights row + "Σ = 1.000" | `.weights[i]` | 3 d.p., bar-free (cells on the value scale) |
| Output vector | `.output[i]` | 4 numbers, 3 d.p. |
| Heatmap(s) | `.weights` for head A, B, or both side by side | `heatmap` glyph at `NUMBER_CELL`; numbers printed |
| Concat row (head = both) | `multiHead([A, B], {causal, divisor}).concat[i]` | 8 numbers, 3 d.p. |
| "Check my work" line | string built from `.scores[i]` substituted into the dot-product expansions, then "÷ 2 → exp → ÷ Σ" | mono, `aria-live="polite"` |

The "Check my work" box for the default state (this exact text appears on the page):
```text
q_sat · k_The  = 0·1    + 2·(−0.5) + 0.5·0   + 0·0.5    = −1.0
q_sat · k_cat  = 0·0    + 2·1.5    + 0.5·0   + 0·(−0.5) =  3.0
q_sat · k_sat  = 0·(−0.5) + 2·0    + 0.5·1   + 0·0.5    =  0.5
q_sat · k_down = masked (−∞)
÷ √4 = ÷ 2           →  −0.5     1.5      0.25
exp (calculator)     →   0.607   4.482    1.284     sum 6.372
÷ sum                →   0.095   0.703    0.202     (adds to 1.000)
output = 0.095·v_The + 0.703·v_cat + 0.202·v_sat
       = [−0.11, 1.41, 0.11, 0.80]
```
Rounding convention: the page shows 3 d.p. of the exact values. The exact third weight is 0.2015…, so a
learner working from rounded exp values may get 0.201 or 0.202; the page's note says "exp needs a
calculator; everything else is arithmetic, and ±0.001 is rounding, not a mistake."

**Try this** (each leads to a named insight)
1. Keep query = sat₃ and switch the causal mask **off** → the row becomes [0.086, 0.635, 0.182, 0.097]:
   0.097 of the weight now lands on "down", a token that comes *later*, and cat's share drops from 0.703
   to 0.635 because the row must still sum to 1 → insight: **The mask is the only thing that makes
   attention causal.** The scores know nothing about order; only the mask stops a query from reading
   later keys.
2. Mask back on. Drag the divisor to **0.5** → [0.000, 0.993, 0.007, 0]; now to **8** →
   [0.259, 0.428, 0.313, 0] → insight: **√d_head is a sharpness dial.** Small divisor: softmax collapses
   to "pick the max". Large divisor: it flattens to "average everything". Dot products are sums of d_head
   terms, so they grow with head size (see the mono line under the slider); dividing by √d_head holds
   them in the useful middle.
3. Divisor back to 2. Set query = down₄ and flip head **A → B** → head A gives [0.114, 0.656, 0.129,
   0.101] (most weight on "cat"); head B gives [0.129, 0.146, 0.578, 0.146] (most weight on "sat", the
   previous token). Choose **both** to see the two heatmaps side by side and the [1 × 8] concat row →
   insight: **Same tokens, different pattern: multi-head is several attention patterns at once.**
   The tokens did not change; only the W matrices did.

## 7. Show me the math
```tex
\htmlClass{hl-q}{Q} = X\,W_Q,\qquad
\htmlClass{hl-k}{K} = X\,W_K,\qquad
\htmlClass{hl-v}{V} = X\,W_V
\qquad
X \in \mathbb{R}^{n \times d_{\text{model}}},\;
W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{head}}}
```
```tex
\htmlClass{hl-s}{S} = \frac{\htmlClass{hl-q}{Q}\,\htmlClass{hl-k}{K}^{\top}}{\sqrt{d_{\text{head}}}} + M,
\qquad
M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}
```
```tex
\htmlClass{hl-a}{A}_{ij} = \frac{e^{S_{ij}}}{\sum_{k=1}^{n} e^{S_{ik}}},
\qquad
\htmlClass{hl-o}{O} = \htmlClass{hl-a}{A}\,\htmlClass{hl-v}{V}
```
```tex
\text{worked row: } A_{3,:} = \operatorname{softmax}\big([-0.5,\ 1.5,\ 0.25,\ -\infty]\big)
= [0.095,\ 0.703,\ 0.202,\ 0]
```
```tex
\text{head}_h = \operatorname{Attention}\!\left(X W_Q^{h},\, X W_K^{h},\, X W_V^{h}\right),
\qquad
\operatorname{MultiHead}(X) = \big[\text{head}_1 \,\|\, \cdots \,\|\, \text{head}_H\big]\, W_O
```
Shapes (toy values in parentheses): X [n × d_model] (4 × 8); W_Q, W_K, W_V [d_model × d_head] (8 × 4) per
head; Q, K, V [n × d_head] (4 × 4); S, A [n × n] (4 × 4); O [n × d_head] (4 × 4); concat [n × H·d_head]
(4 × 8); W_O [H·d_head × d_model] (8 × 8). The worked-row block is templated from the toy state (brief 05
§4.3: template the string, don't reparse).

Color links to the animation: `hl-q` → Q matrix and the lifted query row; `hl-k` → K matrix; `hl-v` → V
matrix; `hl-s` → the score heatmap in frames 3–5; `hl-a` → the weight heatmap and the weights row;
`hl-o` → the output vector / O matrix. Hovering a term outlines the glyph with the Architecture accent
(spec §5.2: identity by label and hover, not by hue). KaTeX is called with `trust: true, strict: false`
for `\htmlClass` (brief 05 §2, pattern 4).

**`math/attention.js` (pure; imports `matmul`, `transpose`, `softmax`, `causalMask` from `./core.js`).**
Tests compare numbers with `Math.abs(a − b) < 5e-4` unless the example is marked exact. Every number in
§5, §6 and §11 was regenerated on 2026-10-07 against the shipped `math/core.js` with the reproducer
recorded at the end of this section.

```js
export const TOY = {
  tokens: ['The', 'cat', 'sat', 'down'],
  dModel: 8, dHead: 4,
  heads: {
    A: { Q: [[1,0,-0.5,0],[0.5,0.5,0,-1],[0,2,0.5,0],[0.5,2,0,0]],
         K: [[1,-0.5,0,0.5],[0,1.5,0,-0.5],[-0.5,0,1,0.5],[0.5,-0.5,0.5,1]],
         V: [[1,0,-1,0],[0,2,0,1],[-1,0,1,0.5],[0.5,-1,0,1]] },
    B: { Q: [[1,0,0,0.5],[2,0,0.5,0],[0,2,0,0.5],[0.5,0,2,0]],
         K: [[1.5,0,-0.5,0],[0,1.5,0,-0.5],[-0.5,0,1.5,0],[0,-0.5,0,1.5]],
         V: [[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]] },
  },
};

// S = Q · Kᵀ  — [n × d] · [n × d] → [n × n]; throws RangeError (via matmul) on a shape mismatch
export function scores(Q, K) → number[][]
//   scores([[1, 2]], [[3, 4], [0, 1]])            → [[11, 2]]                       (exact)
//   scores(TOY.heads.A.Q, TOY.heads.A.K)[2]       → [-1, 3, 0.5, -0.75]              (exact)
//   scores(TOY.heads.B.Q, TOY.heads.B.K)[3]       → [-0.25, 0, 2.75, 0]              (exact)

// every cell ÷ divisor; throws RangeError('scaleScores: divisor must be > 0') otherwise
export function scaleScores(S, divisor) → number[][]
//   scaleScores([[-1, 3, 0.5, -0.75]], 2)         → [[-0.5, 1.5, 0.25, -0.375]]      (exact)
//   scaleScores([[1, 2]], 0.5)                    → [[2, 4]]                         (exact)
//   scaleScores([[1]], 0)                         → throws RangeError

// true = visible (same convention as core.causalMask); hidden cells become -Infinity
// throws RangeError('applyMask: mask shape …') if mask and S differ in shape
export function applyMask(S, mask) → number[][]
//   applyMask([[1, 2, 3]], [[true, true, false]]) → [[1, 2, -Infinity]]              (exact)
//   applyMask([[1, 2]], [[true, true]])           → [[1, 2]]                         (exact)
//   applyMask([[1, 2]], [[true]])                 → throws RangeError

// row-wise core.softmax; a fully masked row throws (core.softmax 'every logit is masked')
export function attentionWeights(S) → number[][]
//   attentionWeights([[-0.5, 1.5, 0.25, -Infinity]]) → [[0.095, 0.703, 0.202, 0]]
//   attentionWeights([[0.5, -Infinity, -Infinity, -Infinity]]) → [[1, 0, 0, 0]]        (exact)
//   attentionWeights([[-Infinity, -Infinity]])    → throws RangeError

// O = A · V  — [n × n] · [n × d] → [n × d]
export function weightedSum(A, V) → number[][]
//   weightedSum([[0.5, 0.5]], [[1, 0], [0, 2]])   → [[0.5, 1]]                       (exact)
//   weightedSum([[1, 0, 0, 0]], TOY.heads.A.V)    → [[1, 0, -1, 0]]                  (exact)
//   weightedSum(attentionWeights(applyMask(scaleScores(scores(A.Q, A.K), 2), causalMask(4))), A.V)[2]
//                                                 → [-0.106, 1.407, 0.106, 0.804]

// the whole pipeline for one head; every intermediate is returned so the animation and toy
// can show them. divisor defaults to √d_head = Math.sqrt(Q[0].length). causal=false → all-true mask.
export function attentionHead(Q, K, V, { causal = true, divisor = Math.sqrt(Q[0].length) } = {})
  → { scores, scaled, mask, masked, weights, output }   // each number[][] ([n×n] except output [n×d])
//   const A = TOY.heads.A;
//   attentionHead(A.Q, A.K, A.V).weights
//     → [[1,0,0,0],[0.321,0.679,0,0],[0.095,0.703,0.202,0],[0.114,0.656,0.129,0.101]]
//   attentionHead(A.Q, A.K, A.V).output[2]        → [-0.106, 1.407, 0.106, 0.804]
//   attentionHead(A.Q, A.K, A.V, { causal: false }).weights[2] → [0.086, 0.635, 0.182, 0.097]
//   attentionHead(A.Q, A.K, A.V, { divisor: 0.5 }).weights[2]  → [0.000, 0.993, 0.007, 0]
//   attentionHead(A.Q, A.K, A.V, { divisor: 8 }).weights[2]    → [0.259, 0.428, 0.313, 0]
//   every row of .weights sums to 1 (|Σ − 1| < 1e-9)
//   const B = TOY.heads.B;
//   attentionHead(B.Q, B.K, B.V).weights
//     → [[1,0,0,0],[0.798,0.202,0,0],[0.168,0.664,0.168,0],[0.129,0.146,0.578,0.146]]
//   attentionHead(B.Q, B.K, B.V).output
//     → [[0,1,0,0],[0.202,0.798,0,0],[0.664,0.168,0,0.168],[0.146,0.129,0.146,0.578]]

// several heads over the same tokens; concat row i = head outputs for row i joined left to right
export function multiHead(heads /* [{Q,K,V}, …] */, opts) → { heads: AttentionHeadResult[], concat: number[][] }
//   multiHead([TOY.heads.A, TOY.heads.B]).concat[2]
//     → [-0.106, 1.407, 0.106, 0.804, 0.664, 0.168, 0, 0.168]
//   multiHead([TOY.heads.A, TOY.heads.B]).concat[0] → [1, 0, -1, 0, 0, 1, 0, 0]     (exact)
//   multiHead([TOY.heads.A]).concat.map(r => r.length) → [4, 4, 4, 4]

// Q = X·W_Q etc.; used by the deferred "from embeddings" preset and by the gallery demo
export function projectQKV(X, { Wq, Wk, Wv }) → { Q, K, V }
//   projectQKV([[1,0],[0,1],[1,1]], { Wq: [[1,2],[3,4]], Wk: [[0,1],[1,0]], Wv: [[2,0],[0,2]] })
//     → { Q: [[1,2],[3,4],[4,6]], K: [[0,1],[1,0],[1,1]], V: [[2,0],[0,2],[2,2]] }    (exact)
//   const X = randomMatrix(4, 8, 1); const Wq = randomMatrix(8, 4, 2, 0.5);
//   projectQKV(X, { Wq, Wk: Wq, Wv: Wq }).Q[0][0]  → -0.287   (pins the seeded preset)
//   projectQKV([[1, 2]], { Wq: [[1]], Wk: [[1]], Wv: [[1]] }) → throws RangeError (via matmul)
```
Builder note: `W_O` is drawn as a shape-labeled block only; no numbers, no function. The page never
materializes it.

**Regeneration record** (run from the repo root on 2026-10-07; output matched every number in §5, §6
and §11, including the sketch):
```sh
node -e '
import("./math/core.js").then(({matmul,transpose,softmax,causalMask})=>{
const r=(x,d=3)=>Array.isArray(x)?x.map(v=>r(v,d)):(x===-Infinity?"-inf":Number(x.toFixed(d)));
const head=(Q,K,V,{causal=true,divisor=Math.sqrt(Q[0].length)}={})=>{const n=Q.length;
 const S=matmul(Q,transpose(K));const sc=S.map(row=>row.map(v=>v/divisor));
 const m=causal?causalMask(n):S.map(row=>row.map(()=>true));
 const mk=sc.map((row,i)=>row.map((v,j)=>m[i][j]?v:-Infinity));const W=mk.map(row=>softmax(row));
 return{S,sc,mk,W,O:matmul(W,V)}};
const A={Q:[[1,0,-0.5,0],[0.5,0.5,0,-1],[0,2,0.5,0],[0.5,2,0,0]],K:[[1,-0.5,0,0.5],[0,1.5,0,-0.5],[-0.5,0,1,0.5],[0.5,-0.5,0.5,1]],V:[[1,0,-1,0],[0,2,0,1],[-1,0,1,0.5],[0.5,-1,0,1]]};
const B={Q:[[1,0,0,0.5],[2,0,0.5,0],[0,2,0,0.5],[0.5,0,2,0]],K:[[1.5,0,-0.5,0],[0,1.5,0,-0.5],[-0.5,0,1.5,0],[0,-0.5,0,1.5]],V:[[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]]};
const a=head(A.Q,A.K,A.V),b=head(B.Q,B.K,B.V);
console.log("A.S",JSON.stringify(r(a.S)));console.log("A.S/2",JSON.stringify(r(a.sc)));
console.log("A.masked",JSON.stringify(r(a.mk)));console.log("A.W",JSON.stringify(r(a.W)));
console.log("A.O",JSON.stringify(r(a.O)));
console.log("exp row3",JSON.stringify(r(a.mk[2].map(Math.exp))),"sum",r(a.mk[2].map(Math.exp).reduce((s,x)=>s+x,0)));
console.log("B.W",JSON.stringify(r(b.W)));console.log("B.O",JSON.stringify(r(b.O)));
console.log("concat row3",JSON.stringify(r([...a.O[2],...b.O[2]])));
console.log("nomask row3",JSON.stringify(r(head(A.Q,A.K,A.V,{causal:false}).W[2])));
for(const d of[0.5,8])console.log("div",d,JSON.stringify(r(head(A.Q,A.K,A.V,{divisor:d}).W[2])));
console.log("rowsums",JSON.stringify(a.W.map(row=>row.reduce((s,x)=>s+x,0))));
})'
```
Output: `A.W [[1,0,0,0],[0.321,0.679,0,0],[0.095,0.703,0.202,0],[0.114,0.656,0.129,0.101]]`,
`A.O[2] [-0.106,1.407,0.106,0.804]`, `exp row3 [0.607,4.482,1.284,0] sum 6.372`,
`B.W [[1,0,0,0],[0.798,0.202,0,0],[0.168,0.664,0.168,0],[0.129,0.146,0.578,0.146]]`,
`concat row3 [-0.106,1.407,0.106,0.804,0.664,0.168,0,0.168]`, `nomask row3 [0.086,0.635,0.182,0.097]`,
`div 0.5 [0,0.993,0.007,0]`, `div 8 [0.259,0.428,0.313,0]`, row sums 1 ± 2e-16.

## 8. In today's models (Oct 2026)
Framing paragraph on the page: "These five steps are the 2017 formula, and they are unchanged in every
layer that still uses softmax attention. Real models often let several query heads share one set of keys
and values to save memory (`kv-compression`). What 2026 models change is (1) how many keys and values
each layer stores per token (→ `kv-compression`), (2) which past tokens a query reads, because every
query scoring every key gives a grid of n × n cells, so doubling the sequence quadruples the work
(→ `long-context-attention`), (3) two small additions around the softmax: normalizing Q and K before the
dot product, and a learned 'sink' logit that gives the row somewhere to put weight when nothing is
relevant, since the row must still sum to 1, and (4) in some models, swapping most softmax-attention
layers for linear attention (→ `long-context-attention`)." (Cost of the plain version: brief 01 §2, DSA
bullet, "O(L²)"; QK-norm and sinks: 01 §1; sharing and linear hybrids: 01 §2.)

The "Claim shown on page" column is the on-page wording. `renderFact` supplies each row's source link
and a "reported" chip; the raw `attention` string in the data file is not printed. Each row names its
year from `release_date`.

| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| GPT-3 (2020): full multi-head attention, 96 heads per layer, each 128 numbers wide, every head with its own keys and values; 96 layers | `models.gpt-3.attention` (expected value: "MHA, 96 heads × 128 dims, one K/V per head"), `models.gpt-3.layers` = 96, `models.gpt-3.release_date` | 01 §4 worked-numbers table; 01 §1 "Attention" row (untagged background; §13 ruling 3 cites the GPT-3 paper) |
| gpt-oss-120b (2025): 64 query heads share 8 sets of keys and values, each 64 numbers wide; layers alternate full attention with a 128-token window; a learned sink per head | `models.gpt-oss-120b.attention` (expected: "GQA 64Q/8KV, head_dim 64, alternating full + SWA-128, learned sinks, attention_bias"), `models.gpt-oss-120b.layers` = 36, `.release_date` | 01 §1 "Biases" and "Attention sink" rows [C]; §2 sliding-window bullet [C]; §4 table; §7 table [C] |
| DeepSeek-V4-Pro (2026): one set of keys and values per layer, 512 numbers wide, shared by all query heads, plus compression and sparsity (`long-context-attention`); Q and K normalized; a learned sink | `models.deepseek-v4-pro.attention` (expected: "1 KV head, head_dim 512; CSA/HCA + SWA-128; QK-norm; learned sink"), `.release_date` | 01 §1 "QK-norm" and "Attention sink" rows [C]; §2 MHA/MQA/GQA paragraph [C]; §7 table [C] |
| Kimi K3 (2026): 96 heads; only 24 of its 93 layers run this softmax attention, the other 69 are linear-attention layers (`long-context-attention`) | `models.kimi-k3.attention` (expected: "69 KDA + 24 MLA (NoPE), 96 heads"), `models.kimi-k3.layers` = 93, `.release_date` | 01 §2 hybrid bullet [C]; §5 table [C]; §7 table [C] |
| MiniMax-M3 (2026): 64 query heads, 4 sets of keys and values; from layer 3 on, each query reads only the 16 most relevant blocks of 128 past tokens | `models.minimax-m3.attention` (expected: "GQA 64Q/4KV + MSA block 128 top-16, sparse from layer 3"), `.release_date` | 01 §2 MSA bullet [C]; §7 table [C] |
| Qwen3.8-2.4T-A95B (2026): 64 query heads, 4 sets of keys and values; only 1 layer in 4 is softmax attention, the other 3 are linear-attention layers | `models.qwen3.8.attention` (expected: "3 GDN : 1 attention, 64Q/4KV heads"), `.release_date` | 01 §2 hybrid bullet [C]; §7 table [C] |

Prose beside the table, timeless and therefore not a data fact: "In production kernels the 4 × 4 grid
on this page is not written to GPU memory at full size: FlashAttention computes exactly these numbers
tile by tile in on-chip SRAM with an online softmax" (01 §2, FlashAttention paragraph; FA4's dated
numbers are deliberately left off the page, §13 ruling 4).

## 9. Takeaways
1. One row of attention is five hand-sized steps: dot the query with every key, divide by √d_head, set
   the future to −∞, softmax, then blend the values by those weights.
2. Attention weights are not parameters. They are recomputed from Q and K for every input; what training
   learns is W_Q, W_K, W_V and W_O.
3. A head is one pattern; a layer runs several patterns in parallel over the same tokens and joins them
   with W_O. Later lessons change what feeds these steps (RoPE), what is stored between them (KV cache,
   GQA, MLA) or which keys are read (sparse, sliding window). Some 2026 layers replace these steps with
   linear attention (`long-context-attention`).

## 10. Next and go deeper
Next: `rope`, `kv-cache`, `multimodal` (all list `attention` as their prerequisite in
`shared/concepts.json`) · Go deeper: 3Blue1Brown, Attention in transformers
(https://www.3blue1brown.com/lessons/attention) for the intuition on video; Transformer Explainer
(https://poloclub.github.io/transformer-explainer/) to run real GPT-2 on your own sentence; Jay Alammar,
The Illustrated Transformer (https://jalammar.github.io/illustrated-transformer/) for the canonical static
diagrams. (brief 05 §1.2)

## 11. Key-frame sketch
Frame 6 (softmax of the hero row), desktop width. Numbers checked against the §7 regeneration record
(`A.masked[2]`, `exp row3`, `sum`, `A.W[2]`).
```text
┌──────────────────────────────────────────────────────────┐
│ [The]₁   [cat]₂   [sat]₃   [down]₄        head A         │
│                                                          │
│  Q [4×4]   K [4×4]   V [4×4]    S/2, masked   keys →     │
│  The ▪▪▪▪  The ▪▪▪▪  The ▪▪▪▪   The  ▪ ░ ░ ░  queries ↓  │
│  cat ▪▪▪▪  cat ▪▪▪▪  cat ▪▪▪▪   cat  ▪ ▪ ░ ░             │
│ ►sat ▪▪▪▪  sat ▪▪▪▪  sat ▪▪▪▪  ►sat  ▪ ▪ ▪ ░ ──┐         │
│  down▪▪▪▪  down▪▪▪▪  down▪▪▪▪   down ▪ ▪ ▪ ▪   │         │
│                                                │         │
│ ►row "sat"                                     ▼         │
│  scaled  │ −0.5  │  1.5  │ 0.25  │  −∞  │                │
│  exp     │ 0.607 │ 4.482 │ 1.284 │   0  │  sum 6.372     │
│  weight  │ 0.095 │ 0.703 │ 0.202 │   0  │  Σ = 1.000     │
├──────────────────────────────────────────────────────────┤
│ Softmax: exponentiate each score, then divide by the sum │
│ so the row adds to 1. "sat" now puts 70% of its          │
│ attention on "cat".                                      │
│ [Back] [Pause] [Next]  ━━━━━●━━━━━  6 / 10   speed [1×]  │
└──────────────────────────────────────────────────────────┘
```
"►" is the selection outline that follows query "sat" in every frame. Q, K, V are 20 px grids (no printed
numbers); S/2 and the row strip are `NUMBER_CELL` grids with printed numbers. At 400px the stage scrolls
inside its container; the caption and controls stack below at full width.

## 12. Open questions for the reviewer
Resolved by the rulings in §13; kept for the record.
1. **Hand-authored Q/K/V vs seeded weights.** The spec says "fixed seeded weights"; this storyboard uses
   hand-picked constants as the default (quarter-grid numbers, visibly different head patterns, a
   hand-checkable hero row) and keeps seeded `randomMatrix` + `projectQKV` as an optional preset. The
   seeded version gives 4-decimal numbers and a near-uniform pattern (row "sat" → [0.363, 0.282, 0.356]),
   which nobody can check by hand. Confirm this trade-off, and whether the "from embeddings" preset is in
   the pilot build or deferred.
2. **`data/models.json` key granularity.** The task-4 key list has one `attention` string per model. This
   page would be better served by `n_heads`, `n_kv_heads`, `head_dim`, `attention_sink` (bool) and
   `qk_norm` (bool) keys, which `kv-compression` and `kv-cache` will also want for their calculators.
   Recommend adding them; the §8 table maps to `attention` until then, with the expected strings given so
   the data agent includes sinks and head counts.
3. **GPT-3 facts are untagged background in brief 01** (§4 table: 96 L, 96 heads × 128). The data file
   needs a `source_url`; the brief gives none. Suggest the data agent cites the GPT-3 paper and marks it
   `confirmed`, or we drop the GPT-3 row and open §8 with gpt-oss.
4. **FlashAttention.** Mentioned as timeless prose only. FA4's dated numbers (Mar 2026, ~1.6 PFLOP/s on
   B200, [R]) have no home in `models.json` or `hardware.json`. Leave off, or add a `techniques` entry
   type to `data/`?
5. **W_O.** Drawn as a shape-labeled block with no numbers (frame 10, toy "both" view). Enough for this
   page, or should the toy multiply the concat row by a seeded W_O to show the final [1 × 8] output?
6. **Caption length.** Some frames are two sentences, which wrap to three lines at 400px. The spec says
   "one-line caption"; I read that as one idea, not one visual line. Confirm.
7. **Control naming.** "Divide scores by" (not "temperature") to avoid a clash with `sampling`'s
   temperature slider. OK, or should the page name the equivalence explicitly in the control label?
8. **Tokenization is skipped.** "The cat sat down" is presented as four tokens with no mention of the
   tokenizer; `pretraining` covers that. A one-line footnote, or nothing?
9. **Scrubbable numbers.** Brief 05 recommends the Tangle pattern. Letting the learner drag one value of
   q_sat and watch the row recompute would be a strong fifth control but exceeds the 2–4 budget. Defer to
   a later revision?

## 13. Reviewer rulings (main session)
Every number in §5, §6 and §7 was re-run against the shipped `math/core.js`; all match.
1. Hand-picked `TOY` constants are the default; hand-checkability outranks the spec's "seeded" wording for this page. The "from embeddings" preset is deferred until after the pilot build.
   Expert review 2026-10-07 (partial overturn): the deferral stands only because nothing on the page now depends on the preset. Misconception 1 rests on frame 9 and try-this 3; frame 1's caption and a visible line above the toy say the numbers are hand-picked.
2. Add `n_heads`, `n_kv_heads`, `head_dim`, `attention_sink`, `qk_norm` to `data/models.json` in the data-extension pass that follows Task 12 (kv-cache and kv-compression need them too).
3. GPT-3: the data-extension pass cites the GPT-3 paper (arXiv 2005.14165) as confirmed. Keep the row.
4. FlashAttention: timeless prose only; no dated FA4 numbers.
5. W_O stays a shape-only block.
6. Captions: one idea each, at most two sentences. This is now a storyboard-wide rule.
   Expert review 2026-10-07 (extended): also at most 30 words and no formulas or operators in captions; formulas go in "Numbers shown" or §7. Old frame 9 carried two ideas and is split into frames 9 (second pattern) and 10 (join and W_O); the animation is now 10 frames.
7. "Divide scores by" stays; the equivalence to temperature is stated in a visible line under the slider (expert-review lesson 5: nothing load-bearing behind hover), not a hover note.
8. Tokenization: one footnote line linking to `pretraining` (text in §3).
9. Scrubbable numbers: deferred to a later revision.
10. Expert review 2026-10-07: `Prereqs` is `decoder-anatomy` (matches `shared/concepts.json`); "four steps" is "five steps" everywhere; the hook's *why* is answered in §3 ¶3; the row-1 fact is a permanent annotation; the residual-stream hand-off is printed in frame 10 and in §3 ¶4.
