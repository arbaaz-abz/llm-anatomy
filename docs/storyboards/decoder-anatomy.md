# The whole model, end to end (`decoder-anatomy`)

Track: architecture · Section: architecture · Prereqs: none (matches `shared/concepts.json`; the track's front door)
Next: `decoder-recap`, `attention` (the two slugs whose `prereqs` list `decoder-anatomy`)
Status: approved (expert review, 2026-10-07; changes applied, see §13)
Sources: 01 §1, §4, §5, §6, §7, §8 · 05 §1.1, §1.2, §2. Beyond the briefs: the gpt-oss-120b and
DeepSeek-V3 `config.json` files and the DeepSeek-V3 model card were fetched and read by the storyboard
author (Fable sub-agent) on 2026-10-07; the values used are listed in §6 and every one matched.

This page is the map. Every later Architecture lesson zooms into one box drawn here, so this page
introduces each box and hands it off (frame → lesson in §5), and never teaches what the neighbor owns.
It shares `attention`'s four tokens ("The cat sat down"), its d_model = 8 and its rows-are-positions
convention, and reuses one row of its attention weights as a static reference, not a step-through.

Differentiation (brief 05 §1.1): bbycroft's LLM Visualization and Transformer Explainer already walk a
GPT-2-style forward pass, and we link both. This page is 2D and phone-first, draws the **2026** block
(pre-norm RMSNorm, MoE in place of the MLP, where the KV cache sits, image patches entering the same
stream), and adds a toy neither has: "where do the parameters live?"

## 1. Learning objective
After this page you can draw the forward pass of a 2026 decoder from memory (pieces of input → one
vector per position → N blocks that each *add* an attention result and an MLP/MoE result onto a running
residual stream → final norm → one score per vocabulary entry → pick a token, append it, repeat), say
what a layer, a block, the KV cache and an expert are, and read a line like "1.6T total / 49B active,
61 layers" as a parameter breakdown you could reproduce with a calculator.

## 2. Misconceptions to correct
Each one names the frame or try-this that corrects it (README lesson 4).
- **Misconception:** "Attention is most of the model." → **Reality:** per block, attention holds 4·d²
  parameters and a GELU MLP 8·d², so two thirds of GPT-3's 175B is MLP (66.4% vs 33.2%, §6). In MoE
  models the experts hold over 90% (gpt-oss-120b: 98.2%; DeepSeek-V3: 97.8%) and attention under 2%.
  Attention is where the *cleverness* is, not where the parameters are. Corrected by frame 5's per-block
  readout and try-this 1. (source: 01 §1 "MLP" row; §6 worked counts)
- **Misconception:** "Each layer sees only the previous layer's output." → **Reality:** each block adds
  its result onto the same residual stream; the input to block 7 is the embedding plus the twelve deltas
  the first six blocks added (two per block), and the embedding is still in there at the final norm. This running sum is a large part of why
  deep stacks train, and the residual connection itself is now a design target (mHC, Attention
  Residuals; `decoder-recap`). Corrected by frames 3–5 (the ⊕ adds, cell by cell). (source: 01 §1
  "Residual stream" row; first principles)
- **Misconception:** "Image tokens go through a different model." → **Reality:** an image patch takes a
  longer road *into* the stream (vision encoder, then a projector) but then rides the same blocks as text;
  "native multimodal" means the backbone was trained on both from step 0, not that there are two
  backbones. Corrected by frame 2's side lane and the §8 modality rows. (source: 01 §6, both recipes;
  "both still have a separate vision encoder + MLP projector")
- **Misconception:** "The model outputs a word." → **Reality:** it outputs one score for *every*
  vocabulary entry (16 here, 163,840 in Kimi K3), and a separate sampling step picks one. Changing the
  sampler changes the text without touching a single parameter. Corrected by frames 8–9; the sampler's
  knobs are deferred to `sampling`. (source: first principles; `models.kimi-k3.vocab_size`)
- **Misconception:** "More total parameters means more compute per token." → **Reality:** with a Mixture
  of Experts, total parameters grow with the number of experts while the parameters each token *uses*
  stay nearly fixed: the toy goes 1,576 → 7,208 total while active moves 1,448 → 1,704 (the only growth
  is the router); DeepSeek-V4-Pro is 1.6T total and 49B active (3.1%). Corrected by frame 6 and
  try-this 3. (source: 01 §5;
  `models.deepseek-v4-pro.total_params`, `.active_params`)
- **Terminology, not a misconception:** "layer" on a model card ("layers: 61") counts *blocks*, each of
  which contains an attention sub-layer and an MLP/MoE sub-layer. The page says "block" for the repeated
  unit and "sub-layer" for its two halves, and frame 7's caption tells the learner that model cards say
  "layer" for block.

## 3. Hook and intuition (final wording)
**Hook:** Between "The cat sat down" and the next word, what does the model actually do, and where in
that machine do 1.6 trillion parameters sit?

The input is cut into pieces: words or word fragments for text, small squares of pixels for an image,
short slices for audio. Each piece becomes one vector of d_model numbers (8 on this page, 7,168 in
DeepSeek-V4-Pro and Kimi K3). A text token gets its vector by looking up a row of a table; an image
patch gets there through a small vision encoder and a projector. From then on the model does not care
where a vector came from.

The vectors, one per position, form the **residual stream**. Think of it as a highway with on-ramps:
each block reads the stream, computes something, and *adds* it back. Nothing is overwritten, so the
embedding is still in there at the end, with two deltas per block piled on top. A block has two halves.
First, **attention** lets each position read the other positions and pull in what it needs: this is the
only place positions talk to each other. Second, an **MLP** (in 2026, usually a **Mixture of Experts**)
transforms each position on its own: this is where most of the parameters live, because the MLP is
two or three wide matrices (d_model × hidden, with hidden two to four times d_model) against
attention's four matrices of about d_model × d_model (smaller when heads share keys and values), and a
Mixture of Experts splits the MLP into dozens or hundreds of smaller MLPs and keeps them all. Each half
is wrapped in "normalize, compute, add"; the normalize step keeps the numbers in a stable range as the
stack gets deep.

A model is that block repeated N times (61 in DeepSeek-V4-Pro, 93 in Kimi K3), each with its own
weights: talk, think, talk, think. After the last block, the last position's vector is normalized and
multiplied by one more matrix, the unembedding, giving one score per vocabulary entry. Softmax turns
the scores into probabilities, a sampler picks a token, the token is appended, and the whole thing runs
again for the next one. With a causal mask, earlier positions' keys and values do not change, so most
models keep them in a **KV cache** and read them rather than recompute them.

The price of this picture, which the rest of the course is about: each generated token multiplies by
roughly all the *active* parameters in one forward pass (so active parameters set compute), carries the whole residual
stream through N blocks one after another (depth means N steps in sequence, which sets a floor on
latency), and reads the KV cache of every earlier position (so context length sets memory traffic).
That is why 2026 models shrink active parameters with experts and shrink the cache with new attention
designs.

Footnote under the first paragraph, visible, not a hover: "How text is cut into tokens is its own topic:
see `pretraining`." (Four paragraphs; the frame hand-offs live in §5 and §10, not here.)

## 4. Visual metaphor
Toy dimensions: n = 4 tokens ("The", "cat", "sat", "down"), d_model = 8, 2 heads of d_head = 4, MLP
hidden 16 (SwiGLU, 384 parameters); in the MoE branch 8 experts of hidden 8 (192 parameters each, so
two active experts equal one dense MLP), N = 2 blocks, vocabulary of 16 words
`[The, cat, sat, down, on, ., and, the, mat, a, dog, ran, up, big, was, then]`. Rows are positions,
as on `attention` (X is [4 × 8]).

**Numbers.** The four embedding rows, the two per-row deltas and the final logits are hand-authored
constants on a quarter grid in `architecture/concepts/decoder-anatomy.js`, so every add on screen is
one line of arithmetic (the same trade-off `attention` made, its ruling 1). They are not derived from
the attention page's Q/K/V (which are themselves hand-picked), and `a_sat` stands in for
W_O · concat, which that page never materializes; a reviewer cannot reconcile the two, by design.
The 12 unused embedding rows are `randomMatrix(12, 8, 11, 1)` snapped to the half grid and are only
ever drawn dim, never read out. The toy in §6 is exact arithmetic on parameter counts and needs no
constants beyond the configs listed there. A visible line under the animation stage says so (README
lesson 10): "The eight-number vectors and the 16 scores are hand-picked stand-ins; the parameter counts
in the toy below are exact."

**Terms introduced, one per frame** (README lesson 3; each is defined in its caption or on screen the
first time): 1 piece / patch · 2 embedding table (vision encoder and projector are *named* here and
defined in `multimodal`) · 3 residual stream, with "block" defined in the same caption as "one repeated
unit of the model" (the reviewer's wording, Must 7: the name is needed before the block is opened) ·
4 normalize · 5 MLP · 6 Mixture of Experts, with "active vs total" as its on-screen readout · 7 N, the
number of blocks, which model cards call "layers" · 8 unembedding (the word on screen is "score";
"logits" waits for §7 and `sampling`) · 9 KV cache. **Terms assumed from the learner's background**
(the course's entry requirement, spec §1): token, vector, matrix, attention with queries, keys and
values, softmax. **Named and deferred:** router and load balancing (`moe`), RoPE (`rope`), RMSNorm's
formula (`decoder-recap`), temperature and sampling knobs (`sampling`), tokenizer (`pretraining`).

**Indexing** (README lesson 11): positions and experts are 1-based on screen (The₁ … down₄, E1 … E8);
this page shows no 0-based addresses.

- `x_The = [1, 0, −0.5, 0, 0.5, 0, 0, −0.5]` · `x_cat = [0.5, 0.5, 0, −1, 0, 1, 0.5, 0]`
- `x_sat = [0, 1, 0.5, 0, −0.5, 1, 0, 0.5]` · `x_down = [0.5, 1, 0, 0, 0, −0.5, 1, 0.5]`
- attention delta for sat (block 1): `a_sat = [0.25, 0.5, −0.25, 0.5, 0, −0.5, 0.25, 0]`
- MLP delta for sat (block 1): `m_sat = [0, 0.25, 0, −0.5, 0.5, 0, −0.25, 0.25]`
- logits for the last position after block 2: `on 2.0 · "." 1.5 · and 0.5 · the 0.0 · the other 12 words −1.0`
- RMSNorm gain γ = 1 throughout; the residual add uses the un-normalized x (pre-norm). The norm's
  numbers (rms 0.586, the rescaled row) stay in §7; on screen the stage is labeled "normalize: rescale
  the row to a standard size" and shows no square root (reviewer's Should).

**Layout** (desktop; at 400px the stream and the zoom stack vertically, the stack frame scrolls in its
own container):
- Top: four `token` chips with subscripts 1–4. In frame 2 a dim side lane at the right, labeled on
  screen as a branch "if the input had an image" (README lesson 13), holds one `patch` glyph (new,
  below) and its two `block`s; the lane never joins the four-row stream.
- Center: the residual stream, a `matrix` X [4 × 8] (`NUMBER_CELL` cells so the 8 numbers print),
  redrawn at each stage of a block; `flow` arrows (carry `activation`) and `adder` junctions (new,
  below) between stages. Zoom frames lift row "sat" out as a row `vector`, as `attention` lifts its
  query row.
- Frames 4–6: the block drawn as two `block`s ("attention", "MLP" / "router + 8 experts") beside the
  stream lane with an `adder` on the lane after each; `heatmap` (one row, masked cell hatched) as the
  static attention reference, labeled on screen "head A of 2; you'll compute this row in `attention`";
  small `matrix` shape chips W_in [8 × 16], W_out [16 × 8]. Frame 6 is a labeled branch ("in most
  2026 models"); frame 7 opens with the dense MLP block visibly back in place.
- Frame 7: `blockStack` (new, below) collapsing the two blocks to "× N". The model-card table sits
  **under the stage** as page text, not inside the 580 × 366 stage (reviewer's stage-budget ruling).
- Frame 8: a `vector` row of **five** cells at `NUMBER_CELL` (README lesson 18: 16 cells would need
  about 690 px): on · "." · and · the · one collapsed cell labeled "12 others". Scores first (maxAbs 2),
  then the same five cells as probabilities (maxAbs 0.4); the collapsed cell prints "−1.0 each" then
  "0.019 each". A visible line between the two states (README lesson 21): "same five words; scores can
  be any size, probabilities add to 1", because the two rows use different color scales.
- Frame 9: `kvStack` per block (count 4 → 5, `highlight: [4]`) beside the stack; the new `token`
  chip "on₅" joins the top row.
- Stage budget check (reviewer's Should): frame 2 holds E as a 16-row `matrix` at 18 px cells (288 px
  tall, dim, numbers on hover only since nothing in E is load-bearing), the [4 × 8] stream at
  `NUMBER_CELL` (344 px wide) and the side lane; frame 7 holds E (18 px), the stream and the stack. Both
  are to be mocked at 580 × 366 before build; if frame 2 overflows, E collapses to its four lit rows plus
  a "12 more rows" line.

Glyphs used (from spec §5.1): token, vector, matrix, heatmap, block, flow (carry activation and kv),
kvStack.

New glyphs proposed (none of the existing drawings can carry them; all four are needed by at least one
other page):
- `patch(parent, { x, y, pixels, index, state })`: a 24 × 24 rounded square holding a 4 × 4 mini
  pixel grid (`pixels` = 16 grey levels or an `rgb` triple each), subscript index like `token`. Depicts
  one image patch as an input piece. `token` draws text and cannot show "this piece is pixels".
  Reused heavily by `multimodal` (patch grid → tokens). Accepted with conditions: the greys are image
  content and never go through `valueColor`; `state` is restricted to `block`'s states (active fill,
  idle, dim), and the selection outline stays the only outline-as-mark. Builder note (reviewer's
  nice-to-have): use a real 4 × 4 crop of a recognizable picture, a cat's ear, to tie it to the sentence.
- `adder(parent, { x, y, r = 9 })`: a ⊕ junction (circle with a plus) sitting on the stream lane.
  Depicts the residual add and encodes no quantity. `block` is too large and has active/idle semantics
  that a sum does not have. Reused by `decoder-recap` (mHC and Attention Residuals redraw exactly this
  junction). Accepted.
- `blockStack(parent, { x, y, w, count, shown = 2, residual = true, halves = ['attention', 'MLP'],
  active })`: `shown` blocks drawn in full (each as two half-`block`s reusing `block`'s states, with an
  `adder` after each on a vertical lane), a "⋮ × N" collapse, then the last block; `count` is printed
  text, never encoded by stack height. Depicts "this block repeated N times with a residual lane through
  all of them". Reused by `kv-cache` (one cache per block), `parallelism` (pipeline stages),
  `model-card` (layers field). Accepted.
- `shareBar(parent, { x, y, w, h = 14, parts: [{ name, value, hue, unknown = false }], format,
  minSegment = 18, tail = 'zoom' })`: a categorical stacked bar with a legend and the percentage printed
  in or under each segment (segment length plus the printed percentage is a required label, not double
  encoding). Printable segments (README lesson 19): any part narrower than `minSegment` px on the main
  bar is folded into one "others" segment, and with `tail = 'zoom'` a second bar underneath, joined to
  "others" by a bracket, redraws the folded parts at a zoomed scale with its own printed percentages
  (so the learner can still follow attention's 0.82% in gpt-oss-120b). `unknown` parts are not on the
  scale at all: they are drawn as a neutral segment of fixed width (24 px) to the right of the bar with
  the printed label "not published". Widths at the toy's 300 px bar (`node -e`, 2026-10-07; parts
  grouped as mapped below): toy, embedding 24.4 · attention 97.5 · MLP 146.2 · other 7.6 · head 24.4
  px, so only "other" (2.54%) folds; GPT-3, embedding 1.1 · attention 99.6 · MLP 199.3 px, so
  embedding + other (0.37%) fold; gpt-oss-120b, MLP 294.5 px and everything else 0.0–2.5 px, so
  embedding + attention + other + head (1.82%) fold into "others" and the zoomed bar shows attention
  0.82 · head 0.50 · embedding 0.50 · other 0.01%; DeepSeek-V3 likewise (tail 1.99%: attention 1.70 ·
  embedding 0.14 · head 0.14 · other 0.02%). Accepted with three conditions: (a) the parts table has
  eight parts and the bar five hues plus one neutral; the mapping is
  `--part-1` embedding + positional · `--part-2` attention · `--part-3` MLP + experts (DeepSeek-V3 has
  both; they split only if a sixth hue passes the contrast check) · `--part-4` router + norms, labeled
  "other" · `--part-5` head · `unknown` segments use a neutral `--line`-tone fill at the fixed width
  above with the printed label "not published" (never hatch: hatch means "excluded / doesn't count"
  everywhere in the course, lesson 24);
  (b) `--part-1 … 5` are defined in both themes and run through the colorblind and contrast check, and
  are distinct from `--req-1 … 4` (request identity); (c) `memBar` is rebuilt as a thin wrapper over
  `shareBar`'s internals (one stacked-bar implementation) with its API unchanged. Reused by
  `training-memory` (weights / grads / optimizer / activations) and `serving-calculator`.

Color: cells use the diverging value scale (maxAbs 2 for the stream and logits, 0.4 for probabilities).
The active half of the block and the active experts use the Architecture accent `active` state (a
state, not an amount); idle experts are drawn in the library's `idle` state (outline), which is not a
selection mark. The KV tiles use `--sem-memory` through `kvStack`.
Hovering a term in the math panel outlines the matching glyph with the accent. Every quantity is encoded
once: the attention weights appear only as numbers in the heatmap row, so the three `flow` arrows into
the attention block in frame 4 are plain and equal, never weighted; the row being followed ("sat", then
"down") carries the same selection outline in every frame it appears. Small marks (README lesson 15):
the model-card table in frame 7, the "total / active" readout in frame 6 and the "Σ = 1.000" readout in
frame 8 are plain labeled text (the frame 7 table is page text under the stage); the loop in frame 9 is
a `flow`; the chosen probability cell in frame 9 gets the selection outline; the shape chips W_in,
W_gate, W_out, W_U are `block`s in the `dim` state, as `attention` draws W_O; the branch labels in
frames 2 and 6 and the "dense MLP restored" label in frame 7 are plain text labels.

## 5. Animation script
Hero row: "sat" (row 3), block 1, for frames 4–6; the last row, "down", for frames 8–9. Numbers are
exact (quarter grid) unless marked 3 d.p. Frame → hand-off: 2 → `multimodal` (and the `pretraining`
footnote) · 4 → `attention`, `rope` · 5 → `decoder-recap` · 6 → `moe` · 7 → `decoder-recap` ·
8 → `sampling` · 9 → `kv-cache`.

| # | On screen | What moves | Caption (final wording) | Numbers shown |
|---|---|---|---|---|
| 1 | Four `token` chips in a row at the top. At the right, dim: one `patch` (a 4 × 4 grey square) and, in prose only, the word "audio frame". Below, an empty [4 × 8] frame labeled "residual stream". | The chips slide in from the left one by one; the patch fades in beside them. | Whatever comes in is cut into pieces: a text token, a small square of an image, a slice of audio. Each piece will get exactly one row below. | n = 4 pieces · d_model = 8 · "a patch is 14 × 14 pixels in Kimi K3 and MiniMax-M3 (2026); 4 × 4 here" |
| 2 | The embedding table E [16 × 8] as a dim `matrix` at the left with its 16 word labels; a `flow` from each chip to its row; the row lights and copies into the stream. Side lane, labeled "if the input had an image": `patch` → `block` "vision encoder" → `block` "projector [→ 8]" → a `vector` of 8 cells, drawn dim, labeled "would be row 5". | Rows of E highlight one at a time and slide into the stream (top to bottom). Then the side lane's dot runs patch → encoder → projector → vector. | A text token looks up its row in the embedding table. An image patch passes through a vision encoder and a projector, and arrives as the same kind of vector. | `x_cat = E[cat] = [0.5, 0.5, 0, −1, 0, 1, 0.5, 0]` · E is 16 × 8 = 128 parameters · visible note: "Kimi K3's vision encoder (2026): 27 layers, 401M parameters, patch 14" |
| 3 | The full stream X [4 × 8] with numbers. A vertical lane extends down from it, with two `adder`s per block drawn faint (block 1 and 2 outlines beside the lane, each labeled "block"). | The lane draws downward; the four rows slide along it as one sheet. | These four rows are the residual stream. Each block (one repeated unit of the model) reads it and adds to it, so the embedding is still there at the end. | X rows as in §4 · shape [4 × 8] · visible note: "d_model is the row width: 8 here, 12,288 in GPT-3 (2020), 7,168 in DeepSeek-V4-Pro and Kimi K3 (2026)" |
| 4 | **Block 1, first half.** Row "sat" lifts out (selection outline). Three stages left to right: `norm`, labeled "normalize: rescale the row to a standard size" (no numbers) → `block` "attention" with a one-row `heatmap` [0.095, 0.703, 0.202, 0] (masked cell hatched) inside it, labeled "head A of 2; you'll compute this row in `attention`", three plain equal `flow` arrows from rows The, cat, sat into it → `adder` on the lane → the updated row. | The lifted row passes through norm (cells dim and brighten, no numbers change on screen), the three arrows run into the attention block, `a_sat` slides to the ⊕ and the sum row types in, with the cell sums printed under it. | First half of a block: normalize, let each row read the other rows (attention), add the result back. This is the one place where positions talk to each other. | weights row from `attention` `[0.095, 0.703, 0.202, 0]` · `a_sat = [0.25, 0.5, −0.25, 0.5, 0, −0.5, 0.25, 0]` · `x' = x + a = [0.25, 1.5, 0.25, 0.5, −0.5, 0.5, 0.25, 0.5]` · printed under cell 5: "−0.5 + 0 = −0.5" · (the norm's arithmetic, rms 0.586, is in §7 only) |
| 5 | **Block 1, second half.** The same lifted row (now x'). Stages: `norm` (same label, no numbers) → `block` "MLP" containing two shape chips W_in, W_gate [8 × 16] and W_out [16 × 8] → `adder` → x''. The other three rows sit still with no arrows into the MLP. A visible per-block readout "attention 256 · MLP 384 · norms 16". | The row passes through norm, widens to 16 cells inside the block, narrows back to 8, `m_sat` slides to the ⊕, the sum types in. | Second half: normalize, push each row alone through a small two-layer network, add back. Rows never see each other here, and this is where most parameters live. | MLP 8 → 16 → 8 · `m_sat = [0, 0.25, 0, −0.5, 0.5, 0, −0.25, 0.25]` · `x'' = [0.25, 1.75, 0.25, 0, 0, 0.5, 0, 0.75]` · readout: attention 256 · MLP 384 · norms 16 |
| 6 | **Branch, labeled on screen "in most 2026 models".** The MLP block swaps for a `block` "router" and eight small `block`s "E1 … E8" (each hidden 8); E3 and E6 `active`, the rest idle. A two-line text readout: "total: 8 × 192 = 1,536 · active: 2 × 192 = 384, the same as the dense MLP". | The MLP block splits into eight; a `flow` dot from the router lands on E3 then E6; the readout counts up. | In most 2026 models the MLP is a Mixture of Experts: a router sends each row to 2 of 8 small MLPs. All 8 count as total, 2 as active. | 8 experts × 192 = 1,536 · top-2 → 384 active (= one dense MLP of 384) · router 8 × 8 = 64 · visible note: "DeepSeek-V4-Pro (2026): 384 routed experts, 6 used per token" |
| 7 | **Key frame.** The branch closes: the dense MLP block is visibly back, labeled "dense MLP restored". The row docks back; the whole stream passes block 1, then an identical block 2 appears below it; the `blockStack` collapses the middle to "⋮ × N". Above: the four chips and E; below: a dim "final norm" and "unembedding" waiting. Under the stage, page text: "N on a model card: toy 2 · GPT-3 (2020) 96 · gpt-oss-120b (2025) 36 · DeepSeek-V4-Pro (2026) 61 · Kimi K3 (2026) 93" and the visible note "Real stacks mix a few block types: dense and MoE, full and sliding-window or linear attention (`decoder-recap`, `long-context-attention`)." | The MLP block fades back in; block 2 slides in under block 1; the stack folds to "× N"; the table's numbers type in. | A model is this block repeated N times, each with its own weights: talk, think, talk, think. "Layers: 61" on a model card counts these blocks. | N = 2 here · 96 · 36 · 61 · 93 · parameters per block (toy, dense): 256 + 384 + 16 = 656 |
| 8 | The last row ("down", selection outline) after block 2 lifts out → `norm` → `block` "unembedding W_U [8 × 16]" → a row `vector` of five cells labeled on · "." · and · the · "12 others", scores first, then the same five cells as probabilities. | The row passes norm and W_U; the five score cells fill (the collapsed cell prints "−1.0 each"); then the colors and numbers interpolate from scores to probabilities (collapsed cell "0.019 each") and a "Σ = 1.000" text readout appears. | After the last block, the last row is normalized and multiplied by the unembedding matrix: one score per vocabulary word. Softmax turns the 16 scores into probabilities: "on" 39%. | scores: on 2.0 · "." 1.5 · and 0.5 · the 0.0 · 12 others −1.0 each · probabilities: on 0.390 · "." 0.237 · and 0.087 · the 0.053 · 12 others 0.019 each (together 0.233) · Σ = 1.000 · W_U = 16 × 8 = 128 params |
| 9 | The "on" probability cell gets the selection outline; a `token` chip "on₅" slides up to join the four chips; the stream gains a fifth row; beside each block a `kvStack` with 4 tiles grows to 5 (`highlight: [4]`). A `flow` from the chips back to the top of the stream closes the loop. A visible note under the stacks. | The chip travels up; row 5 appends; one K and one V tile per block appear; the loop arrow draws. | Pick a token, append it, run again. Only the new row is computed; the earlier rows' keys and values were stored in the KV cache and are read, not recomputed. | 4 → 5 positions · KV tiles per block: 5 K + 5 V · visible note: "per position, GPT-3 (2020, at 2 bytes per number) stored 4,718,592 B ≈ 4.72 MB; DeepSeek-V3 (2024) 70,272 B ≈ 70.3 kB, derived from its config; 2026 designs go to a few kB: `kv-cache`. Sliding-window and linear-attention layers store less: `long-context-attention`" |

Determinism: every frame is a pure function of (step, progress). Reduced motion shows each frame's end
state. The stream matrix keeps its position from frame 3 to frame 9; the block lane keeps its position
from frame 3 to frame 7; the lifted row strip sits in the same place in frames 4, 5, 6 and 8.

Caption word counts (README lesson 2, ≤ 30 words, no operators; counted by splitting on spaces):
28 · 30 · 30 · 29 · 27 · 30 · 26 · 29 · 30. The one absolute in frame 4 ("the one place where positions
talk") holds for the softmax and linear-attention sub-layers alike, since hybrids replace this half, not
the MLP half (01 §2, hybrid bullet); frame 7's "repeated N times" is qualified by the visible
mixed-block-types note under the stage (Kimi K3 is 69 linear-attention + 24 MLA blocks; DeepSeek models
put dense blocks first); frame 9's "stored in the KV cache" is qualified on screen by the visible note.

Branches (README lesson 13): frame 2's side lane and frame 6's MoE swap are both labeled on screen as
branches, and frame 7 opens by restoring the dense MLP with a label, so the toy's dense 656-per-block
count in frame 7 is never confused with the branch's 1,536.

Nothing load-bearing is behind hover (README lesson 5): the cell-by-cell sum in frame 4, the per-block
counts in frame 5, the model-card numbers in frames 2, 3, 6, 7 and 9 and the Σ readout in frame 8 are
all printed. Hover adds only the three-decimal value of any cell (the `cell` glyph's tooltip) and the
dim embedding table's entries in frame 2, which nothing depends on.

## 6. Toy
Title on page: "Where do the parameters live?" The toy opens on the toy preset (the §5 model), so the
counts match the animation's 656-per-block readout, and one chip jumps to a real model.

**Controls**
| id | Label | Type | Range / values | Default | Presets |
|---|---|---|---|---|---|
| `preset` | Model | preset chips | toy · GPT-3 175B · gpt-oss-120b · DeepSeek-V3 · DeepSeek-V4-Pro (partial) | toy | real chips load `PRESETS[id]` (configs from `data/models.json`, see §8); the three sliders below are live only for `toy` and show the real model's values as read-only readouts otherwise. Five chips (Kimi K3 dropped by the reviewer's ruling: its bar would be 99.9% "not published"; its 0.08% appears as text in try-this 2) |
| `layers` | Blocks (N) | Slider | 1–8, step 1 | 2 | — |
| `dModel` | d_model | Slider (snapped, `values`) | [8, 16, 32, 64]; d_head = d_model / 2, dense MLP hidden = 2 · d_model and expert hidden = d_model follow it | 8 | — |
| `experts` | Routed experts (top-2, each half the dense hidden) | Slider (snapped, `values`) | [0 (dense MLP), 2, 4, 8, 16] | 0 | chips "dense" (0) and "8, like the animation" |

Fixed toy constants shown beside the controls: vocabulary 16, 2 heads, SwiGLU, RMSNorm, no biases,
untied embedding and unembedding. Slider validity (README lesson 22): every stop is a legal dense or
MoE decoder; the one edge is `experts = 2` with top-2, where every expert is active, and the readout
says so ("2 of 2 experts used: this is a dense MLP with a router"). Percentages are computed with
`sharePct(part, whole)` from `math/memory.js` (course settlement), never re-implemented here. For real presets a one-line spec sheet replaces this (vocab, d_model,
heads, experts, top-k, dense layers).

**Live outputs** (one state object, one `render()`; every number tabular mono)
| Output | Formula / `math/` function | Units / format |
|---|---|---|
| Parts table: embedding · positional · attention · MLP (dense) · experts · router · norms · head, with two columns: % of total and % of active | `paramBreakdown(config).parts`, `.share`, `.activeShare` | count (toy: exact integers; real: 3 significant figures, `formatCount`) and two percentages |
| `shareBar` of the same parts (hue mapping and the ≥ 18 px rule in §4) | `.share` via `sharePct` | percentages printed in segments; parts under 18 px fold into "others" with a bracketed zoomed tail bar (real presets: everything but the MLP/experts); the partial preset shows a fixed-width neutral "not published" segment off the scale, from `partialBreakdown` |
| Total | `.total` | count |
| Active, with the one convention printed under it (README lesson 16): "Active = the parameters multiplied for one token: every block parameter except unused experts, plus the unembedding. The embedding table is left out: looking up a row is not a multiplication. (If the table is shared with the unembedding, as in GPT-3, it is counted once.) Counting the lookup too gives N; labs differ: Mistral quotes 49B and 52B." | `.active`, `.activeWithEmbedding` (both always visible) | count and `active / total` as % |
| Printed beside the toy readout the first time (README lesson 12): "Active 1,448 of 1,576: active leaves out the 128-parameter embedding table." | `.parts.embedding` | text |
| Per-block line | `.perLayer` | "attention 256 · MLP 384 (or expert 192 × 8) · norms 16" |
| Published total / active (real presets) and the gap | `models.json` entry + `publishedGap(computed, published)` | "computed 116.83B vs published 116.8B (+0.02%)"; the partial V4-Pro preset prints its published 1.6T / 49B back exactly, since its remainder is by subtraction (README lesson 27) |
| "Check my work" box (toy) | templated from `.parts` and `.perLayer` | mono, `aria-live="polite"` |

The "Check my work" box for the default state (this exact text appears on the page):
```text
per block
  attention   W_Q, W_K, W_V, W_O        4 × (8 × 8)          =   256
  MLP         W_in, W_gate [8 × 16], W_out [16 × 8]   3 × 8 × 16   =   384
  norms       2 × 8                                          =    16
  block                                                      =   656
× 2 blocks                                                   = 1,312
embedding table      16 × 8                                  =   128
final norm                                                   =     8
unembedding          16 × 8                                  =   128
total                                                        = 1,576
active = total − embedding lookup                            = 1,448
```

**Try this** (each leads to a named insight)
1. Toy preset, read the shares: MLP 48.7%, attention 32.5%, embedding + head 16.2%, norms 2.5%. Tap
   **GPT-3**: MLP 66.4%, attention 33.2%, embedding 0.35%. Tap **gpt-oss-120b**: experts 98.2%,
   attention 0.8%. → **Insight: most parameters think per token; they don't talk between tokens.** A
   dense block's MLP is 8·d² against attention's 4·d² (1.5× in the toy, whose hidden is 2·d); turn the
   MLP into experts and attention shrinks to a rounding error.
2. Toy preset, set **Blocks = 1** and d_model = 8: embedding + head = 256 of 920, 27.8%. Now **Blocks =
   8, d_model = 64**: 0.6%. A text line under the toy adds the far end: "Kimi K3 (2026): 2.35B of
   2.78T, 0.08%" (from `models.kimi-k3.vocab_size` × `.d_model`). → **Insight: the embedding table is
   vocab × d_model, the blocks are about 10–12 · N · d_model² (10 in this toy, 12 in GPT-3); the table
   only matters in small models.** Read gpt-oss-120b's "% of active" column: its unembedding is 579M of
   5.13B active, 11%, so in a sparse model the head is a visible slice of what each token actually
   multiplies.
3. Toy preset, Blocks = 2, slide **Routed experts** 0 → 8 → 16: total 1,576 → 4,008 → 7,208; active
   1,448 → 1,576 → 1,704. Then tap **DeepSeek-V4-Pro**: 1.6T total, 49B active, 3.1%. → **Insight: MoE
   grows total with the number of experts; active barely moves.** Two active experts of hidden 8
   (2 × 192 = 384) are exactly one dense MLP of hidden 16, so the only active growth is the router: one
   row of d_model per expert per block (64 → 128 here). Real 2026 experts are fine-grained in the same
   way (01 §5).

**`math/params.js`** (pure, no DOM; tests first). All worked examples below were run with `node -e` on
2026-10-07 from this draft's reference implementation. Configs are plain objects; nothing is mutated.

```js
// Config shape (every field explicit so a preset can be audited against its source):
// { vocab, dModel, layers, norm: 'rmsnorm'|'layernorm', positional: 'rope'|'learned', maxPositions?,
//   tiedEmbeddings: bool, biases: bool,
//   attention: { kind: 'gqa', nHeads, nKvHeads, dHead }
//            | { kind: 'mla', nHeads, qLoraRank, kvLoraRank, qkNopeDim, qkRopeDim, vHeadDim },
//   mlp: { kind: 'swiglu'|'gelu', hidden },
//   moe: null | { routed, shared, topK, hidden, denseLayers } }

// Parameters of one attention sub-layer. GQA: d·nH·dh + 2·d·nKV·dh + nH·dh·d (+ biases).
// MLA: d·q_r + q_r·nH·(nope+rope) + d·(kv_r+rope) + kv_r·nH·(nope+v) + nH·v·d + (q_r + kv_r) latent norms.
attentionParams(attention, dModel, { biases = false } = {}) → number
//   ({ kind:'gqa', nHeads:2, nKvHeads:2, dHead:4 }, 8)                 → 256         (exact; 4 × 8 × 8)
//   ({ kind:'gqa', nHeads:96, nKvHeads:96, dHead:128 }, 12288, { biases:true }) → 604,028,928  (GPT-3: 4·d² + 4·d)
//   ({ kind:'gqa', nHeads:64, nKvHeads:8, dHead:64 }, 2880, { biases:true })    → 26,550,080   (gpt-oss-120b)
//   ({ kind:'mla', nHeads:128, qLoraRank:1536, kvLoraRank:512, qkNopeDim:128, qkRopeDim:64, vHeadDim:128 }, 7168)
//                                                                     → 187,107,328  (DeepSeek-V3)

// Parameters of one MLP: (3 for SwiGLU, 2 for GELU) · d · hidden (+ biases).
mlpParams({ kind, hidden }, dModel, { biases = false } = {}) → number
//   ({ kind:'swiglu', hidden:16 }, 8)        → 384            (exact; the toy's dense MLP)
//   ({ kind:'swiglu', hidden:8 }, 8)         → 192            (exact; one toy expert)
//   ({ kind:'gelu', hidden:49152 }, 12288, { biases:true }) → 1,208,020,992   (GPT-3: 8·d² + 5·d)
//   ({ kind:'swiglu', hidden:2048 }, 7168)   → 44,040,192     (one DeepSeek-V3 expert)
//   ({ kind:'swiglu', hidden:2880 }, 2880, { biases:true }) → 24,891,840     (one gpt-oss-120b expert)

// The whole breakdown. parts sum to total exactly. One definition of active (README lesson 16):
//   active = total − unused experts − (tiedEmbeddings ? 0 : embedding table)
// i.e. every block parameter that is multiplied for one token plus the unembedding; the lookup table is
// left out unless it is the unembedding (tied). activeWithEmbedding adds the lookup table back (untied only).
// activeShare[part] = that part's active parameters / active (experts: only the top-k + shared ones).
paramBreakdown(config) → { parts: { embedding, positional, attention, mlp, experts, router, norms, head },
                           total, active, activeWithEmbedding, perLayer: { attention, mlp, expert }, share, activeShare }
//   paramBreakdown(PRESETS.toy)
//     → parts { embedding:128, positional:0, attention:512, mlp:768, experts:0, router:0, norms:40, head:128 }
//       total 1576 · active 1448 · share.mlp 0.4873 · share.attention 0.3249            (exact)
//   paramBreakdown({ ...PRESETS.toy, moe: { routed:8, shared:0, topK:2, hidden:8, denseLayers:0 } })
//     → parts { …, experts:3072, router:128, mlp:0 } · total 4008 · active 1576 · perLayer.expert 192  (exact)
//   paramBreakdown({ ...PRESETS.toy, moe: { routed:16, …, hidden:8 } })  → total 7208 · active 1704   (exact)
//   paramBreakdown({ ...PRESETS.toy, layers:1 })               → total 920 · (embedding+head)/total 0.2783
//   paramBreakdown(PRESETS.gpt3)
//     → total 174,604,259,328 (174.60B) · active 174,604,259,328 (tied: the shared matrix is multiplied, so it counts)
//       attention 57.99B (33.21%) · mlp 115.97B (66.42%) · embedding 617.6M (0.35%)
//   paramBreakdown(PRESETS.gptOss120b)
//     → total 116,829,149,760 (116.83B) · experts 114.70B (98.18%) · attention 955.8M (0.82%) · head 579.1M
//       active 5,132,842,560 (5.13B) · activeWithEmbedding 5,711,976,000 (5.71B)
//   paramBreakdown(PRESETS.deepseekV3)
//     → total 671,026,404,352 (671.03B) · experts 656.46B (97.83%) · attention 11.41B (1.70%) · mlp 1.19B (3 dense layers)
//       active 36,625,603,584 (36.63B) · activeWithEmbedding 37.55B

// For a model whose config is only partly published: the derivable parts plus one neutral "not published" remainder.
partialBreakdown({ known: { [part]: number }, publishedTotal }) → { parts: { …known, unknown }, total, share }
//   ({ known: { experts: 385 * 61 * mlpParams({kind:'swiglu', hidden:3072}, 7168) }, publishedTotal: 1.6e12 })
//     → parts.experts 1,551,425,863,680 (97.0%) · parts.unknown 48.6B (3.0%)        (DeepSeek-V4-Pro, if all 61 layers are MoE)
//   ({ known: { embedding: 160000 * 7168, head: 160000 * 7168 }, publishedTotal: 2.78e12 })
//     → parts.embedding 1.147B · parts.head 1.147B · (embedding+head)/total 0.00083    (Kimi K3; feeds the try-this 2 text line, not a chip)

// Signed relative gap, computed vs published.
publishedGap(computed, published) → number
//   (174_604_259_328, 175e9)  → −0.0023 · (116_829_149_760, 116.8e9) → +0.0003 · (671_026_404_352, 671e9) → +0.00004
```

Preset configs (`PRESETS` in `math/params.js`; every number maps to a data key in §8):
- `toy`: vocab 16, d 8, 2 layers, 2 heads × 4, SwiGLU hidden 16, RMSNorm, RoPE, untied, no biases, dense;
  the experts slider adds `moe: { routed: N, shared: 0, topK: 2, hidden: 8, denseLayers: 0 }`.
- `gpt3`: vocab 50,257, d 12,288, 96 layers, 96 heads × 128, GELU hidden 49,152, LayerNorm, learned
  positions (2,048), **tied** embeddings (GPT-2's convention; untied would give 175.22B, and the paper's
  "175.0B" is consistent with either), biases, dense. Computed 174.60B vs published 175B: −0.23%, the
  paper's figure is rounded. Active = total (dense and tied: the one shared matrix is multiplied).
- `gptOss120b`: vocab 201,088, d 2,880, 36 layers, 64 heads × 64 with 8 KV heads, SwiGLU, 128 experts
  top-4 with hidden 2,880, no shared expert, no dense layers, RMSNorm, RoPE, untied
  (`tie_word_embeddings: false`), `attention_bias: true` and biases on the expert MLPs. **Computed
  116.83B vs published 116.8B (+0.02%); active 5.13B vs 5.1B (+0.6%).** The gap is rounding in the
  published 116.8B; the per-head sinks (64 × 36 = 2,304) and router biases (36 × 128 = 4,608) add under
  7,000 parameters, and including them gives 116,829,156,672, still 116.83B. This is the page's verified
  real-model check. Every config value above was re-verified against
  https://huggingface.co/openai/gpt-oss-120b/raw/main/config.json (fetched and read 2026-10-07 by the
  storyboard author); the published totals are `models.gpt-oss-120b.total_params` / `.active_params`.
- `deepseekV3`: vocab 129,280, d 7,168, 61 layers (first 3 dense, hidden 18,432), MLA (128 heads, q rank
  1,536, kv rank 512, nope 128, rope 64, v 128), 256 routed + 1 shared experts top-8 with hidden 2,048,
  untied, no biases, 1 MTP layer. **Computed 671.03B vs the paper's 671B (0.00%); active 36.63B vs
  37B.** The "explain the gap" line under this chip: the Hugging Face checkpoint is 685B because it
  also ships the 14B multi-token-prediction module (the card says so verbatim: "685B, which includes
  671B of the Main Model weights and 14B of the Multi-Token Prediction (MTP) Module weights"). Config
  re-verified against https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json and the card
  at https://huggingface.co/deepseek-ai/DeepSeek-V3 (both fetched and read 2026-10-07 by the storyboard
  author).
- `deepseekV4Pro` (partial): experts from confirmed facts only (384 + 1 experts, hidden 3,072, d 7,168,
  61 layers), remainder a neutral "not published" segment with the printed assumption "if all 61 layers
  are MoE". Active: 7 experts × 61 × 66.06M = 28.2B of the published 49B (58%), the rest "not published".
- Kimi K3 is not a preset (reviewer's ruling 2); its embedding + head (2.35B, 0.08%) is a text line in
  try-this 2 computed from `models.kimi-k3.vocab_size` × `.d_model` × 2.

Tests to write first: every `parts` sum equals `total`; `active ≤ total`; `active = total − embedding`
for every untied dense config and `active = total` for every tied dense config (lesson 16: the prose
claim "dense: every block parameter is used for every token" is asserted as `active − head ===
Σ block parts` for both toy and GPT-3); two toy experts of hidden 8 equal one dense MLP of hidden 16
(`2 · mlpParams({swiglu, 8}, 8) === mlpParams({swiglu, 16}, 8)`, lesson 17); the four full presets
reproduce the numbers above to the digit; `share` and `activeShare` each sum to 1 ± 1e-12;
`partialBreakdown` throws if `Σ known > publishedTotal`; configs passed in are not mutated
(`Object.freeze` the presets); `publishedGap` sign convention.

**Reproducer** (README lesson 6). Every number in §5, §6 and §11 was regenerated on 2026-10-07 with
this call (the same formulas `math/params.js` will implement; the full presets are the configs listed
above):
```sh
node -e '
const gqa=(d,nH,nKV,dh,b)=>d*nH*dh+2*d*nKV*dh+nH*dh*d+(b?nH*dh+2*nKV*dh+d:0);
const mla=(d,nH,qr,kvr,nope,rope,v)=>d*qr+qr*nH*(nope+rope)+d*(kvr+rope)+kvr*nH*(nope+v)+nH*v*d+qr+kvr;
const mlp=(k,d,h,b)=>(k==="swiglu"?3:2)*d*h+(b?(k==="swiglu"?2*h:h)+d:0);
const bd=(c)=>{const d=c.d,L=c.L,b=!!c.b,nm=(c.ln?2:1)*d,attn=c.mla?mla(d,...c.mla):gqa(d,...c.gqa,b);
 const moeL=c.moe?L-c.moe.dense:0,dl=L-moeL,ex=c.moe?mlp("swiglu",d,c.moe.h,b):0;
 const p={embedding:c.V*d,positional:c.pos?c.pos*d:0,attention:L*attn,mlp:dl*mlp(c.k,d,c.h,b),
  experts:c.moe?moeL*(c.moe.n+c.moe.s)*ex:0,router:c.moe?moeL*d*c.moe.n:0,norms:L*2*nm+nm,head:c.tied?0:c.V*d};
 const total=Object.values(p).reduce((s,v)=>s+v,0);
 const active=total-p.experts+(c.moe?moeL*(c.moe.k+c.moe.s)*ex:0)-(c.tied?0:p.embedding);
 return {p,total,active,activeEmb:active+(c.tied?0:p.embedding),perBlock:{attention:attn,mlp:mlp(c.k,d,c.h,b),expert:ex}};};
const toy={V:16,d:8,L:2,gqa:[2,2,4],k:"swiglu",h:16};
const show=(n,c)=>{const r=bd(c);console.log(n,r.total,r.active,r.activeEmb,JSON.stringify(r.p),JSON.stringify(r.perBlock))};
show("toy",toy);show("toy1L",{...toy,L:1});show("toy8x64",{...toy,L:8,d:64,gqa:[2,2,32],h:128});
show("moe8",{...toy,moe:{n:8,s:0,k:2,h:8,dense:0}});show("moe16",{...toy,moe:{n:16,s:0,k:2,h:8,dense:0}});
show("gpt3",{V:50257,d:12288,L:96,gqa:[96,96,128],k:"gelu",h:49152,ln:1,pos:2048,tied:1,b:1});
show("gptoss",{V:201088,d:2880,L:36,gqa:[64,8,64],k:"swiglu",h:2880,b:1,moe:{n:128,s:0,k:4,h:2880,dense:0}});
show("dsv3",{V:129280,d:7168,L:61,mla:[128,1536,512,128,64,128],k:"swiglu",h:18432,moe:{n:256,s:1,k:8,h:2048,dense:3}});
const e=3*7168*3072;console.log("v4 experts",385*61*e,385*61*e/1.6e12,"active",7*61*e,7*61*e/49e9);
console.log("k3 emb+head",2*160000*7168,2*160000*7168/2.78e12,"dsv3 kv B/pos",61*576*2,(61*576*2/1e3).toFixed(1)+" kB","gpt3 kv B/pos",2*96*96*128*2,(2*96*96*128*2/1e6).toFixed(2)+" MB");
console.log("ratios",1.6e12/49e9,2.78e12/104.2e9,2.4e12/95e9,753e9/40e9,428e9/23e9,116.8e9/5.1e9,1.05e12/49e9);
const rms=v=>Math.sqrt(v.reduce((s,x)=>s+x*x,0)/v.length);const x=[0,1,.5,0,-.5,1,0,.5],a=[.25,.5,-.25,.5,0,-.5,.25,0],m=[0,.25,0,-.5,.5,0,-.25,.25];
const x1=x.map((v,i)=>v+a[i]),x2=x1.map((v,i)=>v+m[i]);console.log(rms(x).toFixed(3),x1.join(","),rms(x1).toFixed(3),x2.join(","));
const z=[-1,-1,-1,-1,2,1.5,.5,0,-1,-1,-1,-1,-1,-1,-1,-1],ez=z.map(Math.exp),S=ez.reduce((s,v)=>s+v,0);const p=ez.map(v=>v/S);console.log(S.toFixed(2),p.map(v=>v.toFixed(3)).join(","),"12 others",(12*p[0]).toFixed(3),"5-cell sum",(p[4]+p[5]+p[6]+p[7]+12*p[0]).toFixed(3));
'
```
Data pass 2026-10-07 (Kimi K3 vocabulary 163,840, was 160,000; embedding + head 2.35B, was 2.29B):
```sh
node -e 'const e = 2 * 163840 * 7168; console.log(e, (e / 2.78e12 * 100).toFixed(3) + "%")'   # 2348810240 0.084%
```
Output on 2026-10-07 (after the expert review): toy 1576 / 1448 / 1576, parts {128, 0, 512, 768, 0, 0,
40, 128}, per block {256, 384}; toy1L 920; toy8x64 330816; moe8 4008 / 1576 / 1704, experts 3072,
router 128, expert 192; moe16 7208 / 1704 / 1832; gpt3 174604259328 / 174604259328 (tied); gptoss
116829149760 / 5132842560 / 5711976000; dsv3 671026404352 / 36625603584 / 37552282624; v4 experts
1551425863680 (0.9696), active 28207742976 (0.5757); k3 2293760000 (0.000825); dsv3 KV 70272 B =
70.3 kB per position, gpt3 KV 4718592 B = 4.72 MB (catch-up pass: decimal units); active/total ratios 32.65 (V4-Pro), 26.68 (K3), 25.26 (Qwen3.8), 18.83
(GLM-5.3), 18.61 (M3), 22.90 (gpt-oss), 21.43 (Mistral L4); rms 0.586, x′ = 0.25,1.5,0.25,0.5,−0.5,
0.5,0.25,0.5, rms(x′) 0.656, x″ = 0.25,1.75,0.25,0,0,0.5,0,0.75; Σe^z = 18.93, p(on) 0.390, p(.)
0.237, p(and) 0.087, p(the) 0.053, others 0.019 each, 12 others together 0.233, five-cell sum 1.000.
The §11 sketch's "256 + 384 + 16 = 656" is `perBlock` from the same output.

## 7. Show me the math
```tex
\htmlClass{hl-x}{x^{(0)}_i} = E[t_i] \quad (\text{text}),\qquad
\htmlClass{hl-x}{x^{(0)}_i} = \operatorname{Proj}\big(\operatorname{ViT}(\text{patch}_i)\big) \quad (\text{image}),
\qquad x^{(0)} \in \mathbb{R}^{n \times d_{\text{model}}}
```
```tex
\htmlClass{hl-a}{a^{(\ell)}} = \operatorname{Attn}\big(\operatorname{RMSNorm}(x^{(\ell)})\big),\quad
\tilde{x}^{(\ell)} = x^{(\ell)} + \htmlClass{hl-a}{a^{(\ell)}};\qquad
\htmlClass{hl-m}{m^{(\ell)}} = \operatorname{MLP}\big(\operatorname{RMSNorm}(\tilde{x}^{(\ell)})\big),\quad
x^{(\ell+1)} = \tilde{x}^{(\ell)} + \htmlClass{hl-m}{m^{(\ell)}}
```
```tex
\operatorname{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\tfrac{1}{d}\sum_j x_j^2}},\qquad
\text{worked: } \operatorname{rms}(x_{\text{sat}}) = \sqrt{2.75/8} = 0.586
```
```tex
\operatorname{MLP}(h) = \big(\sigma(h W_{\text{gate}}) \odot h W_{\text{in}}\big) W_{\text{out}},\qquad
\operatorname{MoE}(h) = \sum_{e \in \operatorname{top\text{-}k}(h W_{\text{router}})} g_e\, \operatorname{MLP}_e(h)
```
```tex
\htmlClass{hl-z}{z} = \operatorname{RMSNorm}\big(x^{(N)}_n\big)\, W_U \in \mathbb{R}^{|V|},\qquad
\htmlClass{hl-p}{p_v} = \frac{e^{z_v}}{\sum_{u} e^{z_u}},\qquad
\text{worked: } p_{\text{on}} = \frac{e^{2}}{18.93} = 0.390
```
```tex
\text{params per block} = \underbrace{d\,(n_H d_h) + 2d\,(n_{KV} d_h) + (n_H d_h)\,d}_{\text{attention}}
+ \underbrace{3\, d\, h}_{\text{SwiGLU MLP}}
\quad\text{or}\quad
\underbrace{N_e \cdot 3\, d\, h_e + d\, N_e}_{\text{experts + router}}
```
```tex
\text{total} = |V|\,d \;(+\,|V|\,d \text{ if untied}) + N \cdot \text{params per block} + \text{norms},\qquad
\text{active} = \text{total} - (N_e - k)\, N\, 3\, d\, h_e - |V|\, d\ \text{(the lookup table, untied models only)}
```
Shapes (toy in parentheses): x [n × d_model] (4 × 8); E, W_U [|V| × d_model] and [d_model × |V|]
(16 × 8, 8 × 16); W_in, W_gate [d_model × h] (8 × 16); W_out [h × d_model] (16 × 8); z, p [|V|] (16);
each toy expert has h_e = 8, so two active experts (2 · 192) equal the dense MLP (384).
Simplifications stated on the panel: γ = 1 everywhere; the toy's MLP hidden is 2·d (real SwiGLU models
use about 8/3·d and GPT-3 used 4·d), so the toy's MLP : attention ratio is 1.5 : 1 against GPT-3's
2 : 1; the one "active" convention (§6): every multiplied block parameter plus the unembedding, the
lookup table left out unless it is the unembedding. The RMSNorm worked line (rms 0.586) lives here, not
on the stage. Color links:
`hl-x` → the stream matrix and lifted row; `hl-a` → the attention block and `a_sat`; `hl-m` → the MLP
block and `m_sat`; `hl-z` → the logits row; `hl-p` → the probabilities row. KaTeX with
`trust: true, strict: false` for `\htmlClass` (brief 05 §2, pattern 4).

## 8. In today's models (Oct 2026)
Framing paragraph on the page: "Most 2026 frontier models are this picture with four knobs turned: the
MLP is a Mixture of Experts (so total and active parameters differ by 19–33×), attention stores far less
per position than GPT-3 did, some take images into the same stream, and the stack is 36–93 blocks deep.
The dense models in the table, GPT-3 (2020) and Llama 3.1 405B (2024), are the baseline." (The ratios
are computed from the table's entries: 18.6× MiniMax-M3 to 32.7× DeepSeek-V4-Pro; see the reproducer.)
One basis for every ratio on the page (README lesson 25): active = parameters multiplied in one
forward pass of one token, total = parameters stored; the line "total ÷ active, counted per forward
pass" sits once above the table.

| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| GPT-3 175B: 96 blocks, d_model 12,288, 96 heads × 128, vocab 50,257, 2,048 positions, dense; our count 174.6B | `models.gpt-3.total_params` = 175e9, `.layers` = 96, `.d_model` = 12288, `.n_heads` = 96, `.head_dim` = 128, `.vocab_size` = 50257, `.context_length` = 2048 (all proposed; source arXiv 2005.14165 per `attention` ruling 3) | 01 §4 worked-numbers table; §1 baseline column |
| gpt-oss-120b (2025): 116.8B total / 5.1B active (4.4%), 36 blocks, 128 experts top-4; config d 2,880, 64 Q / 8 KV heads × 64, vocab 201,088, expert hidden 2,880; our count 116.83B / 5.13B | `models.gpt-oss-120b.total_params`, `.active_params`, `.layers`, `.experts_total`, `.experts_active` (existing, confirmed); `.d_model`, `.n_heads`, `.n_kv_heads`, `.head_dim`, `.vocab_size`, `.expert_hidden`, `.biases`, `.tied_embeddings` (proposed; source = the entry's config.json URL, re-verified 2026-10-07, see §6) | 01 §5 table [C]; §7 [C] |
| DeepSeek-V3 (Dec 2024): 671B / 37B, 61 blocks (3 dense), 256 + 1 experts top-8, MLA; our count 671.03B / 36.6B; the HF checkpoint is 685B because it adds a 14B MTP module | `models.deepseek-v3.*` (entry proposed by `paged-attention`, accepted in its ruling; keys as for gpt-oss plus `.mla_q_rank`, `.mla_kv_rank`, `.mla_rope_dim`, `.mla_nope_dim`, `.mla_v_dim`, `.dense_layers`, `.experts_shared`, `.expert_hidden`, `.mlp_hidden`, `.mtp_params` = 14e9; sources arXiv 2412.19437, the HF config.json and model card, re-verified 2026-10-07, see §6) | 01 §5 table (background row); §2 MLA paragraph |
| DeepSeek-V4-Pro: 1.6T / 49B (3.1%), 61 blocks, 384 + 1 experts top-6, expert hidden 3,072, d 7,168; text only | `models.deepseek-v4-pro.total_params`, `.active_params`, `.layers`, `.experts_total`, `.experts_active`, `.modalities` (existing, confirmed); `.d_model` = 7168, `.expert_hidden` = 3072, `.experts_shared` = 1 (proposed; source arXiv 2606.19348) | 01 §5 table [C]; §6 "V4 is text-only" [C] |
| Kimi K3: 2.78T / 104.2B (3.7%), 93 blocks, 896 + 2 experts top-16, vocab 163,840, d 7,168; native multimodal with a 401M-parameter vision encoder | `models.kimi-k3.total_params`, `.active_params`, `.layers`, `.experts_total`, `.experts_active`, `.modalities` (existing, confirmed); `.d_model` = 7168, `.vocab_size` = 163840, `.vision_encoder_params` = 401e6 (in data; source arXiv 2607.24653) | 01 §5 table [C]; §6 vision-encoder sizes [C] |
| Qwen3.8-2.4T-A95B: 2.4T / 95B (4.0%), 92 blocks, 512 + 1 experts top-10; text only | `models.qwen3.8.total_params`, `.active_params`, `.layers`, `.experts_total`, `.experts_active`, `.modalities` (existing, confirmed) | 01 §5 table [C]; §7 [C] |
| GLM-5.3: 753B / 40B (5.3%), 78–80 blocks (sources conflict, shown as a range) | `models.glm-5.3.total_params` (confirmed), `.active_params` (reported chip), `.layers` (range) | 01 §5 table; §7; Open uncertainties |
| MiniMax-M3: ~428B / ~23B (5.4%), 60 blocks; native multimodal | `models.minimax-m3.total_params`, `.active_params`, `.layers`, `.modalities` (existing, confirmed) | 01 §5 table [C]; §6 [C] |
| Mistral Large 4: 1.05T / 49B routed-active, 52B including embeddings: labs differ on whether the embedding counts as active | `models.mistral-large-4.total_params`, `.active_params` (+ its `note`) (existing, `reported`: the blog says 52B active, the briefs 49B) | 01 §5 table; §7 verdicts |
| Llama 3.1 405B (2024): dense, so every block parameter is used for every token; a dense baseline on the shares bar with GPT-3 | `models.llama-3.1-405b.total_params` (existing entry, confirmed; renamed from the old id in the data pass per lesson 8) | 01 §7 (background) |
| 2026 norm: about 3–5% of parameters active per token (the table's entries run 3.1–5.4%), down from 9–28% in 2023–25 (Mixtral 8x7B, Dec 2023, was 28%; Qwen3-235B, 2025, 9%) | derived on the page from the `active_params / total_params` of the entries above (no new key) | 01 §5 "Active ratios" |
| KV per position: GPT-3 (2020, at 2 bytes per number) 4,718,592 B ≈ 4.72 MB → DeepSeek-V3 (2024) 70,272 B ≈ 70.3 kB, derived from its confirmed config (61 × 576 × 2, `kvBytesPerTokenMla` in `math/memory.js`); "2026 designs go to a few kB: `kv-cache`" (frame 9's visible note; decimal SI units, exact bytes printed, per the course settlement) | `models.gpt-3.kv_bytes_per_token` (4,718,592), `models.deepseek-v3.kv_bytes_per_token` (70,272; both computed from confirmed configs and stored `reported` with formula notes; same entry as `paged-attention` uses) | 01 §4 table |
| Patch size 14 × 14 pixels; Kimi K3's vision encoder is 27 layers / 401M parameters (frames 1–2 notes) | `models.kimi-k3.patch_size` = 14, `.vision_encoder_params` = 401e6, `.vision_encoder_layers` = 27; `models.minimax-m3.patch_size` = 14 (proposed; sources arXiv 2607.24653 and the M3 HF config) | 01 §6 mechanics and encoder sizes [C] |

Rendered with `renderFact` so each row carries its source link and a "reported" chip where the data
file says so. Not shown: Mistral Large 4's expert count (unpublished), Kimi K3's expert width (not in
the brief), audio token rates ([U]).

## 9. Takeaways
1. One forward pass: pieces of input become one vector per position; N blocks each add an attention
   result (positions talk) and an MLP/MoE result (each position thinks) onto a single residual stream;
   a final norm and the unembedding give one score per vocabulary entry; a sampler picks, the token is
   appended, and it runs again with earlier positions' K and V read from the cache.
2. Parameters live mostly in the MLP: two thirds of a dense model, over 90% of a Mixture of Experts.
   Attention is a third at most; the embedding table only matters in small models.
3. A model card is this picture in numbers: "layers" counts blocks, d_model is the row width, "total /
   active" is all experts versus the few each token uses, and the KV cache holds keys and values for
   every attention block and earlier position (newer designs store far less: `kv-cache`).

## 10. Next and go deeper
Next: `decoder-recap` (what changed in each box since GPT-3 and why) and `attention` (the first box
opened); both list `decoder-anatomy` as their prerequisite in `shared/concepts.json`. Frame hand-offs
that are not graph edges (`multimodal`, `rope`, `kv-cache`, `moe`, `sampling`, `model-card`) are
in-page links, not "Next". Prereqs: none; the `pretraining` footnote covers tokenization.

Go deeper (brief 05 §1.2): Brendan Bycroft, LLM Visualization (https://bbycroft.net/llm): a 3D walk
through every tensor of a tiny GPT, the best way to *see* the shapes · Transformer Explainer
(https://poloclub.github.io/transformer-explainer/): type your own sentence into GPT-2 in the browser ·
Jay Alammar, The Illustrated Transformer (https://jalammar.github.io/illustrated-transformer/): the
canonical static diagrams. (Three links, per spec §4; Raschka's LLM Architecture Gallery moves to
`decoder-recap`, where the model-by-model survey belongs.)

## 11. Key-frame sketch
Frame 7 (the stack collapses to × N), desktop width. `▪` = a value cell, `⊕` = adder on the lane. The
patch lane is not drawn in this frame (it appears only in frames 1–2); the model-card table is page
text under the stage.
```text
┌──────────────────────────────────────────────────────────┐
│ [The]₁ [cat]₂ [sat]₃ [down]₄                 E [16 × 8]  │
│        │       │       │                     (dim)       │
│        ▼       ▼       ▼                                 │
│ residual stream X [4 × 8]                                │
│  The  ▪▪▪▪▪▪▪▪    cat  ▪▪▪▪▪▪▪▪                          │
│  sat  ▪▪▪▪▪▪▪▪    down ▪▪▪▪▪▪▪▪                          │
│        ║                                                 │
│  ┌─────╫───────── block 1 ──────┐  256 + 384 + 16 = 656  │
│  │norm→│attention (rows talk)│→⊕│                        │
│  │norm→│MLP (think)          │→⊕│  dense MLP restored    │
│  └─────╫──────────────────────┘   [KV: K▪▪▪▪ V▪▪▪▪]     │
│        ⋮  × N                                            │
│  ┌─────╫───────── block N ──────┐                        │
│  │ … same shape, own weights …  │  (real stacks mix      │
│  └─────╫──────────────────────┘    block types)         │
│   final norm → W_U [8 × 16] → scores → softmax → "on"    │
├──────────────────────────────────────────────────────────┤
│ A model is this block repeated N times, each with its own│
│ weights: talk, think, talk, think. "Layers: 61" on a     │
│ model card counts these blocks.                          │
│ [Back] [Pause] [Next]  ━━━━━━━●━━  7 / 9   speed [1×]    │
└──────────────────────────────────────────────────────────┘
 N on a model card: toy 2 · GPT-3 (2020) 96 · gpt-oss-120b (2025) 36 ·
 DeepSeek-V4-Pro (2026) 61 · Kimi K3 (2026) 93
 Real stacks mix a few block types: dense and MoE, full and
 sliding-window or linear attention (decoder-recap, long-context-attention).
```
The stream is drawn as two rows of two [1 × 8] vectors here only so the stack fits the 366 px stage;
the builder may instead shrink the stream to 18 px cells in frames 7 and 9 (its numbers were read in
frames 3–5 and are not load-bearing here). At 400px the stack scrolls inside its own container; the
table and note stay under it as page text.

## 12. Settled questions (expert review 2026-10-07; stated as applied, per README lesson 20)
1. **Graph edges.** The header matches `shared/concepts.json` (`decoder-recap` and `attention` list
   this page). `multimodal`, `moe`, `sampling` and `kv-cache` reach it transitively; no edits.
2. **Presets.** Five chips: toy, GPT-3, gpt-oss-120b, DeepSeek-V3, DeepSeek-V4-Pro (partial: 97%
   derivable, remainder "not published"). Kimi K3 is not a chip; its 0.08% embedding share is a text
   line in try-this 2.
3. **Data keys and entries (data-pass item).** Keys to add: `d_model`, `vocab_size`, `mlp_hidden`,
   `expert_hidden`, `experts_shared`, `dense_layers` (not stored for V4-Pro), `tied_embeddings`, `biases`, `mla_q_rank`,
   `mla_kv_rank`, `mla_nope_dim`, `mla_rope_dim`, `mla_v_dim`, `vision_encoder_params`,
   `mtp_params`, `patch_size`, `vision_encoder_layers`, plus `attention`'s `n_heads`, `n_kv_heads`,
   `head_dim`. Entries: `gpt-3` (arXiv 2005.14165, plus `kv_bytes_per_token` 4,718,592 from 01 §4)
   and `deepseek-v3` (including `kv_bytes_per_token` 70,272 and `mtp_params` 14e9 with the card URL).
   The gpt-oss-120b and DeepSeek-V3 config values in §6 were re-verified against both config.json
   files on 2026-10-07; those URLs are the `source_url`, confidence `confirmed`. The Llama entry is
   `llama-3.1-405b` (exact checkpoint, lesson 8).
4. **DeepSeek-V4-Pro dense layers (a gap in the data).** The page prints the assumption "if all 61 layers
   are MoE" (97%) and labels it an assumption. The data pass found no dense-layer count (the V4-Pro config
   has no `first_k_dense_replace`; its `num_hash_layers` of 3 is not a dense count), so the page makes no
   dense-layer claim and cites no such key. (If 3 layers were dense it would give 1.475T, 92%; that is
   an illustration, not a fact.)
5. **"Active" convention.** One definition, §6: every multiplied block parameter plus the unembedding;
   the lookup table is left out unless it is the unembedding (tied). No fifth control; `active` and
   `activeWithEmbedding` are both visible. Data-pass item: confirm gpt-oss's own accounting in its
   model card and store the card's figures.
6. **DeepSeek-V3 685B vs 671B.** The HF card's "685B, which includes 671B of the Main Model weights
   and 14B of the Multi-Token Prediction (MTP) Module weights" (read 2026-10-07) is the "explain the
   gap" line under the V3 chip, keyed as `deepseek-v3.mtp_params` = 14e9.
7. **Hand-authored stream numbers.** `x_*`, `a_sat`, `m_sat` and the scores are constants (§4), with
   the visible stand-in line; `a_sat` does not reconcile with `attention`'s concat row (that page stops
   before W_O), and the frame 4 heatmap label says where the row comes from.
8. **Patch position.** The image patch stays in a labeled side-lane branch and never joins the
   four-row stream, so every number matches `attention`.
9. **Glyphs.** `patch`, `adder`, `blockStack` and `shareBar` are accepted with the conditions in §4;
   `shareBar` is its own glyph and `memBar` is a wrapper over it.
10. **Frame 9's KV note** names GPT-3 (4,718,592 B ≈ 4.72 MB) and DeepSeek-V3 (70,272 B ≈ 70.3 kB,
    derived from its confirmed config), then "2026 designs go to a few kB: `kv-cache`". These bytes per token
    are computed from confirmed configs; the data stores them as `reported` with formula notes.
11. **Toy MLP widths.** Dense hidden 2·d (stated simplification); experts hidden d (two active = one
    dense MLP).
12. **Captions.** Nine captions, each ≤ 30 words and ≤ 2 sentences, verified in §5.

## 13. Reviewer rulings (expert review, 2026-10-07) and what changed
Verdict: APPROVE WITH CHANGES; the reviewer recomputed the full reproducer and independently recounted
GPT-3, gpt-oss-120b and DeepSeek-V3 (all matched). Status is now "approved (expert review)". Every
number touched below was regenerated with the §6 reproducer, whose recorded output is the post-review run.

Must (all 10 applied):
1. Toy experts have hidden 8 (192 parameters), so two active experts equal one dense MLP; totals
   1,576 → 4,008 → 7,208 and active 1,448 → 1,576 → 1,704 in frame 6, try-this 3, misconception 5, the
   `paramBreakdown` examples and the reproducer. (Lesson 17.)
2. One "active" definition (lesson 16): tied GPT-3 keeps its shared matrix (active = total,
   174.60B); the Llama row says "dense: every block parameter is used for every token"; the toy prints
   "active leaves out the 128-parameter embedding table" beside its readout; the test reads "every
   untied dense config" plus a tied case.
3. §8 framing: dense baselines are GPT-3 and Llama 3.1 405B; the total/active ratio is 19–33× (18.6×
   MiniMax-M3 to 32.7× V4-Pro, computed in the reproducer); the 28% end (Mixtral 8x7B) is dated 2023,
   so "2023–25".
4. Deltas: two per block (twelve before block 7 in misconception 2; "two deltas per block" in §3 ¶2).
5. gpt-oss gap: rounding in the published 116.8B; sinks + router biases add under 7,000 parameters
   (116,829,156,672 with them). The data pass stores the card's own figure if it prints 116.83B / 5.13B.
6. Frame 8 shows five cells (on, ".", and, the, "12 others: −1.0 each → 0.019 each"); Σ = 1.000 holds
   (five-cell sum in the reproducer). (Lesson 18.)
7. "block" is defined in frame 3's caption ("one repeated unit of the model", 30 words); frame 7's new
   term is N, which model cards call "layers". (Lesson 3.)
8. Frame 6 is labeled on screen as a branch ("in most 2026 models") and frame 7 opens with the dense
   MLP visibly restored and labeled, so the 656-per-block count is the dense toy. (Lesson 13.)
9. Intuition cut to four paragraphs: the "That is the entire forward pass…" paragraph is gone (its
   hand-offs live in §5 and §10); the "price" paragraph stays (lesson 14).
10. A visible note under frame 7's table: "Real stacks mix a few block types: dense and MoE, full and
    sliding-window or linear attention (`decoder-recap`, `long-context-attention`)"; the §11 sketch
    carries the same qualifier. (Lesson 7.)

Should (all 13 applied, no rebuttals):
- Frames 4–5 show no rms numbers; the norm stage is labeled "normalize: rescale the row to a standard
  size" and the arithmetic (rms 0.586, 0.656) lives in §7.
- Frame 4's heatmap row is labeled "head A of 2; you'll compute this row in `attention`".
- Try-this 2 says "about 10–12 · N · d_model² (10 in this toy, 12 in GPT-3)".
- §3 ¶2: "four matrices of about d_model × d_model (smaller when heads share keys and values)"; a MoE
  "splits the MLP into dozens or hundreds of smaller MLPs and keeps them all".
- §3 ¶4: "depth means N steps in sequence, which sets a floor on latency".
- Takeaway 3: "the KV cache holds keys and values for every attention block and earlier position
  (newer designs store far less: `kv-cache`)".
- Frame 8 keeps "score" on screen; "logits" is left to §7 and `sampling`; the term list says so.
- §8: "about 3–5%" with the table's own 3.1–5.4% range stated.
- Stage budget: the frame 7 table is page text under the stage; frames 2 and 7 are to be mocked at
  580 × 366 before build, with the fallbacks written in §4 (E collapses to its four lit rows; the stream
  shrinks to 18 px cells in frames 7 and 9).
- §11 sketch no longer draws the patch lane, matching §5.
- Go deeper has three links; Raschka's gallery moves to `decoder-recap`.
- Frame 9's note says "GPT-3 (2020, at 2 bytes per number)".
- (Ruling 10, same note) names DeepSeek-V3's 70,272 B ≈ 70.3 kB instead of V4-Pro's reported range
  (unit made decimal in the catch-up pass).

Nice to have (both taken): the norm's purpose ("keeps the numbers in a stable range as the stack gets
deep") is in §3 ¶2; the `patch` builder note asks for a real 4 × 4 crop (a cat's ear).

Glyph conditions (all written into §4): `patch` greys never use `valueColor` and `state` is limited to
`block`'s states; `adder` encodes no quantity; `blockStack` halves reuse `block` states and "× N" is
printed text; `shareBar` has the five-hue-plus-neutral mapping, a non-hatched "not published" segment,
and `memBar` becomes a wrapper over it; `--part-1 … 5` go through the colorblind and contrast check.

Data-pass items collected from the rulings: add the §12 item 3 keys and the `deepseek-v3` entry
(including `kv_bytes_per_token` = 70,272 and `mtp_params` = 14e9 with the card URL); rename
`llama-3-405b` → `llama-3.1-405b` (done); V4-Pro's dense-layer count is not in data; confirm gpt-oss's active
accounting from its model card and store the card's own figures.

Catch-up pass 2026-10-07 (README lessons 19–28 and the course-wide settlements; Status unchanged):
- Lesson 19 / settlement (hatch): `shareBar` gains `minSegment = 18` and `tail = 'zoom'`; parts under
  18 px fold into "others" with a bracketed zoomed tail bar; "not published" is a fixed-width neutral
  segment off the scale. Segment widths at 300 px recorded in §4 from `node -e`.
- Lesson 20: §12 rewritten as settled statements; alternatives deleted.
- Lesson 21: frame 8 gets a visible "same five words; scores can be any size, probabilities add to 1"
  line between the score and probability states.
- Lesson 22: slider stops checked; the `experts = 2` stop is labeled "2 of 2 experts used: a dense MLP
  with a router".
- Lesson 23: no change (the page shows no measured anchors; published parameter counts carry the
  "active" convention in §6).
- Lesson 24: no change beyond the `shareBar` wording (the only hatch on the page is the masked cell in
  frame 4's heatmap row, which is "excluded").
- Lesson 25: one basis for every total/active ratio, printed once above the §8 table.
- Lesson 26: the cost paragraph says "in one forward pass".
- Lesson 27: the partial V4-Pro preset prints its published 1.6T / 49B back (remainder by
  subtraction), stated in §6.
- Lesson 28: no change (no bandwidth ladder on this page).
- Settlement (decimal units): frame 9's note, the §8 KV row, §12 item 10 and the reproducer now print
  "4,718,592 B ≈ 4.72 MB" and "70,272 B ≈ 70.3 kB" (`node -e`: `61*576*2` → 70272, `/1e3` → 70.3;
  `2*96*96*128*2` → 4718592, `/1e6` → 4.72); no KiB/MiB remain.
- Settlement (`math/memory.js`): the KV figures cite `kvBytesPerTokenMla`; percentages use
  `sharePct(part, whole)` from `math/memory.js`.
- Settlement (`--carry-activation` = `var(--accent-arch)`): no change to this page's `flow` usage
  (carry `activation` and `kv` only).
- Settlements (chip memory, FORMATS keys, draft tokens): no change; none appear on this page.
- Settlement (data ids): `llama-3.1-405b` already used.
- Data pass 2026-10-07: Kimi K3 vocabulary 163,840 (embedding + head 2.35B, was 2.29B; node check 2*163840*7168 = 2,348,810,240); no V4-Pro dense-layer claim (not in data); KV bytes per token are computed from confirmed configs and stored `reported`; Llama id `llama-3.1-405b`.
