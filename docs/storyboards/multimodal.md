# Seeing images (`multimodal`)

Track: architecture · Section: architecture · Prereqs: attention
Next: `model-card` (the one slug whose `prereqs` list `multimodal` in `shared/concepts.json`)
Status: approved (expert review)
Sources: 01 §6 (two recipes, mechanics, encoder sizes, 2026 reality check), §5 table, §7 · 05 §1.1 (Multimodal: "Build patch grid → token count toy; link ViT-Explainer"), §1.2. Nothing beyond the briefs.

Opens `decoder-anatomy` frame 2's side lane ("if the input had an image": `patch` → vision encoder →
projector → one vector) and its `patch` glyph, with the glyph's accepted conditions (greys are image
content and never go through `valueColor`; `state` is limited to `block`'s states). The words are
`attention`'s "The cat sat down"; the mask grid uses `attention`'s heatmap convention.

## 1. Learning objective
After this page you can trace an image into a language model: pixels → patches → patch vectors → vision
encoder → merged neighbors → projector → tokens in the same stream as text (frames 1–6); compute how many
tokens an image or a video costs from its size, the patch size and the merge factor (frames 7–9, toy); and
say what "native multimodal" means and does not mean (frame 10).

## 2. Misconceptions to correct
Each one names the frame or try-this that corrects it (README lesson 4).
- **Misconception:** "The language model looks at the pixels." → **Reality:** the language model only
  ever sees vectors. A vision encoder turns each patch into a vector, neighbors are merged, and a projector
  maps the result to d_model; from then on image tokens ride the same blocks as words. (01 §6 recipe 1 and
  mechanics) · corrected by frames 2–6
- **Misconception:** "Native multimodal means there is no vision encoder." → **Reality:** "native" means the
  backbone was trained on text and images together from the first step. Kimi K3 and MiniMax-M3 are native
  and both still have a separate vision encoder and an MLP projector. (01 §6: "'native' means joint
  pretraining, not 'no encoder'") · corrected by frame 10
- **Misconception:** "An image is about one token, like a word." → **Reality:** a 1,008 × 1,008 image cut
  into 14-pixel patches is 5,184 patches, 1,296 tokens after Kimi K3's 2 × 2 merge; its largest input,
  3,584 × 3,584, is 16,384 tokens. (01 §6 mechanics [C]) · corrected by frame 7, try-this 1
- **Misconception:** "Video is just a few images." → **Reality:** tokens grow with every frame. At this
  page's toy settings (448 × 448 frames, 2 per second, 2 × 2 merge) one hour of video is 1,843,200 tokens,
  more than a 1M context, before any pooling over time. (01 §6 video bullet: frames sampled, temporal pooling
  needed) · corrected by frame 9, try-this 3
- **Misconception:** "Every 2026 frontier model sees images." → **Reality:** several leaders are text-only:
  DeepSeek-V4, GLM-5.3, gpt-oss and Qwen3.8. (01 §6 "2026 reality check" [C]) · corrected by frame 10 (page text); §8 is the data source

## 3. Hook and intuition (final wording)
**Hook:** A language model only ever multiplies vectors. How does a photo become something it can read,
and why can one picture cost as many tokens as a short story?

An image is cut into a grid of small squares, patches, 14 pixels on a side in Kimi K3 and MiniMax-M3. Each
patch's pixels are flattened and multiplied by a matrix into one vector, and a vision encoder (a
transformer of its own, in which all patches of an image can see each other) turns those vectors into
descriptions of what each patch shows in context. Position works differently than for text: an image has rows and columns, so instead of one running count, models typically give each patch its row and column (and its frame, for video) through a 2D or 3D version of RoPE. Neighboring patches are then merged, four into one in
Kimi K3, and a small projector maps each merged vector to the language model's width. Those vectors join
the residual stream exactly where words do.

From there on nothing is special. Words after the image attend to the image tokens through the usual causal
mask, and the blocks treat them like any other position. The cost is the count. Tokens grow with the area of
the image divided by the patch area and the merge, so doubling an image's side quadruples its tokens. Models
keep each image's own shape (dynamic resolution) so they don't waste tokens on padding, and video pays per
frame, so encoders also pool over time.

Two ways to build this are in use. The older adapter recipe bolts a pretrained encoder and projector onto a
finished language model. The native recipe trains the encoder, projector and backbone on mixed text and
images from the first step; it still has an encoder, it just learns everything together. And not every
frontier model sees at all: several 2026 leaders are text-only.

## 4. Visual metaphor
**Toy:** a 16 × 16-pixel grey image (a real crop of a cat's ear, as `decoder-anatomy` §4's builder note
asks), patch 4 → a 4 × 4 grid of 16 patches; each patch has 16 grey values; encoder width 8 (as d_model);
merge 2 × 2 → 4 image tokens; projector → d_model 8. The words "The cat sat down" follow the image. A visible
line under the stage (README lesson 10): "The 16-pixel image, the 4-pixel patches and the width of 8 are toy
sizes; real models use 14-pixel patches on images up to thousands of pixels wide. Token counts are exact."

**Positions** (README lesson 11): in this frame's sequence the four image tokens take positions 1–4 and the
words 5–8; a visible label in frame 6 says so ("in this sequence the words sit at 5–8"). Patches are numbered
1–16, row by row. No memory addresses. `decoder-anatomy` frame 2 drew the patch as "would be row 5", after
the words; this page puts the image first because that is the real order for "describe this picture" (words
after an image can attend to it, words before it cannot), and frame 6's label says where the words moved.

**Terms introduced, one per frame** (README lesson 3): 1 patch size · 2 patch embedding · 3 vision encoder
(named on `decoder-anatomy`, opened here) · 4 merge · 5 projector · 6 interleaved sequence · 7 tokens per
image · 8 dynamic resolution · 9 temporal pooling (named) · 10 native vs adapter. **Terms assumed from
prereqs:** token, embedding, residual stream, d_model, patch (`decoder-anatomy`); attention, causal mask,
heatmap (`attention`). **Named and deferred:** 2D/3D position for patches (one line, linked to `rope`);
audio (not covered: the brief has no verified rates).

**Layout** (stage ≈ 580 × 366):
- Left: the image as a 4 × 4 grid of `patch` glyphs (24 px each plus gaps, about 112 px), patch 6 carrying
  the selection outline (the followed patch, same mark in every frame where it appears).
- Center: a `flow` to a dim `block` "patch embedding [16 → 8]", then a column of 16 `vector`s at 18 px
  (hover-only), then `block` "vision encoder", then the 2 × 2 merge (four `vector`s fuse into one), then
  `block` "projector [32 → 8]".
- Right: the stream, four image-token rows above four `token` chips; in frame 6 an 8 × 8 `heatmap` (18 px,
  pattern only, unread cells hatched).
- Frames 7–9: page-text counters and one `matrix` grid drawn as a coarse patch grid (cells encode nothing;
  the count is printed).

Glyphs used (from spec §5.1 and the accepted `decoder-anatomy` proposals): patch, block, vector, flow (carry
`activation`), token, heatmap, matrix.
New glyphs proposed: none. Small marks (README lesson 15): counters, "× 2 × 2 merge" labels, the "in this
sequence…" position note and the recipe labels in frame 10 are plain text labels.

Color: patch greys are image content (not the value scale). Vectors use the value scale with seeded values
from `randomMatrix` (hover-only; nothing load-bearing is in them). Image-token rows and word rows are told
apart by labels ("img₁ … img₄", word chips), never by hue. The followed patch 6 (and the merged token it
lands in, img₁) carries the selection outline.

## 5. Animation script
| # | On screen | What moves | Caption (final wording) | Numbers shown |
|---|---|---|---|---|
| 1 | The 16 × 16 image; grid lines draw at every 4 pixels; it separates into 16 `patch` glyphs numbered 1–16; patch 6 outlined. | Grid lines draw; patches slide apart slightly. | The image is cut into a grid of square patches. Here each patch is 4 by 4 pixels; real models use 14 by 14. | 16 × 16 pixels ÷ 4 × 4 = 16 patches · each patch 16 grey values · Kimi K3, MiniMax-M3: patch 14 |
| 2 | Patch 6's 16 greys flatten into a row; a `block` "patch embedding [16 → 8]"; out comes one 8-cell `vector`; then all 16 patches do the same. | Patch 6 flattens and passes the block; the other 15 follow together. | Each patch's pixels are flattened and multiplied by one learned matrix, giving one vector per patch. That is the patch embedding. | 16 greys → 8 numbers · 16 patch vectors · matrix [16 × 8] = 128 parameters |
| 3 | The 16 vectors enter a `block` "vision encoder"; inside, a small 16 × 16 pattern grid with every cell read (no hatching), labeled "within one image, every patch sees every patch". | Flow dots run between patch vectors inside the block; the vectors retint. | A vision encoder, a transformer of its own, lets every patch attend to every other one, with no causal mask. Each vector now describes its patch in context. | 16 × 16 = 256 patch pairs read · Kimi K3's encoder: 27 layers, 401M parameters |
| 4 | The 4 × 4 grid of encoded vectors groups into 2 × 2 blocks; each group of four concatenates into one 32-cell vector. Patch 6 (row 2, column 2) lands in group 1, the top-left block. | Groups of four slide together and fuse. | Neighboring patches are merged four into one, so the image costs a quarter of the tokens. Kimi K3 merges 2 by 2; DeepSeek's newest encoder merges 3 by 3. | 16 → 4 vectors of 4 × 8 = 32 numbers · K3 2 × 2 (÷ 4) · DeepSeek-ViT 3 × 3 (÷ 9, reported) |
| 5 | Each merged vector passes a `block` "projector [32 → 8]" and comes out 8 wide; labels img₁ … img₄; img₁ outlined. | Four flows through the projector; four rows appear at the stream's width. | A small projector maps each merged vector to the language model's width. Now the image is four tokens, the same shape as a word's vector. | 32 → 8 · 4 image tokens of d_model 8 · projector: a small MLP (DeepSeek-ViT's too, reported) |
| 6 | The stream: img₁ … img₄ then The cat sat down, eight rows; label "in this sequence the words sit at 5–8"; an 8 × 8 causal pattern grid beside it, the word rows reading all four image columns. | Word chips slide in under the image rows; the grid fills row by row. | Image tokens and words share one sequence. Under the causal mask, every word can attend to the image, and the blocks treat both alike. | 8 positions · each word row reads 4 image tokens + the words before it · row "sat" (position 7) reads 7 keys |
| 7 | A plain text prompt first, for one second: "How many tokens is a phone photo?" Then a coarse grid for a 1,008 × 1,008 photo at patch 14: "72 × 72 = 5,184 patches → 1,296 tokens"; then the largest input K3 accepts, 3,584 × 3,584: "65,536 → 16,384". | The grid's counters type in; then the grid redraws four times larger per side. | Tokens grow with the image's area. A 1,008-pixel square photo costs 1,296 tokens, and Kimi K3's largest input costs 16,384. | 1008 ÷ 14 = 72 · 72² = 5,184 · ÷ 4 = 1,296 (0.12% of 1,048,576) · 3584 ÷ 14 = 256 · 256² = 65,536 · ÷ 4 = 16,384 (1.6%) · visible line: "64 of these fill a 1M window" |
| 8 | A wide 1,008 × 504 photo: left, squashed into a square (1,296 tokens, label "stretched"); right, kept at its shape (648 tokens). | The square version deflates into the wide grid; the counter drops. | Dynamic resolution keeps each image's own shape instead of resizing it to a square. A wide photo then costs only the tokens it actually covers. | 1008 × 504 at patch 14: 72 × 36 = 2,592 patches → 648 tokens (vs 1,296 as a square) |
| 9 | A filmstrip of 448 × 448 frames, 2 per second: "256 tokens per frame"; a counter for 1 minute (30,720) and 1 hour (1,843,200) against a "1M context" line; label "toy settings (448², 2 fps, 2 × 2); real encoders also pool over time". | Frames stream in; the counter overtakes the 1M line. | Video pays for every frame. At these settings an hour of video would overflow a million-token context, so encoders also pool frames over time. | 448 ÷ 14 = 32 · 32² ÷ 4 = 256 per frame · 120 frames (1 min) → 30,720 · 7,200 frames (1 h) → 1,843,200 (175.8% of 1,048,576) |
| 10 | Two copies of the frame 5 pipeline side by side. Left "adapter": encoder and projector labeled "pretrained, attached later"; backbone "already trained on text". Right "native": all three labeled "trained together from step 0". Both keep an encoder. Page text under the stage: "Not every 2026 frontier model sees images: DeepSeek-V4-Pro (its paper lists images as future work), GLM-5.3, gpt-oss-120b and Qwen3.8 are text-only." | The labels type in; a single "trained together" bracket wraps the right side. | The adapter recipe attaches an encoder to a finished text model. Native models train encoder and backbone together from the start, and still keep an encoder. | adapter: LLaVA-style (2023) · native: Kimi K3 (encoder trained from scratch with next-token prediction), MiniMax-M3 ("mixed-modality training from the very first step") |

Determinism: every frame is a pure function of (step, progress). Reduced motion shows each frame's end
state. The patch grid keeps its position in frames 1–4; the stream keeps its position in frames 5–6; patch 6
and img₁ keep the selection outline.

Caption word counts (README lesson 2; ≤ 30 words, ≤ 2 sentences, no operators): 24 · 21 · 28 · 29 · 25 ·
24 · 20 · 25 · 24 · 26.

Absolutes checked (README lesson 7): frame 3's "every patch … no causal mask" describes vision encoders over
one image (01 §6 describes ViT/SigLIP encoders; K3's video encoder factorizes attention in space and time,
noted in §8). Frame 6's "every word can attend to the image" holds when the image comes first, as drawn.
Frame 10's "still keep an encoder" is 01 §6 [C] for K3 and M3.

Branches (README lesson 13): frames 7–9 switch from the toy image to real sizes and are labeled with the
real dimensions; frame 10 is labeled as a side-by-side comparison, not a timeline step.

## 6. Toy
Title on page: "What does a picture cost?" One state object, one `render()`. The stand-in line from §4 is
printed above.

**Controls**
| id | Label | Type | Range / values | Default | Presets |
|---|---|---|---|---|---|
| `width` | Width | Slider (snapped to multiples of patch × merge) | 112–3,584 px | 1,008 | — |
| `height` | Height | Slider (same snapping) | 112–3,584 px | 1,008 | — |
| `patch` | Patch size | segmented | 14 · 16 px | 14 | — |
| `merge` | Merge neighbors | segmented | none · 2 × 2 · 3 × 3 | 2 × 2 | — |
| `media` | Input | segmented | image · video | image | — |
| `seconds` | Video length (video only) | Slider (snapped) | [10, 60, 600, 3,600] s | 60 | — |
| `fps` | Frames per second (video only) | segmented | 1 · 2 | 2 | — |
| `context` | Context window | segmented | 262,144 · 1,048,576 tokens | 1,048,576 | — |
| `preset` | Model settings | chips | Kimi K3 (patch 14, 2 × 2, max 3,584) · "3 × 3 merge" (DeepSeek-ViT, reported) · toy (16 px image, patch 4) | Kimi K3 | — |

Snapping keeps both sides divisible by patch × merge, so every count is an integer; the slider shows the
snapped value.

**Live outputs**
| Output | Formula / `math/` function | Units / format |
|---|---|---|
| Patch grid (cols × rows) and patches | `patchGrid({ width, height, patch })` | "72 × 72 = 5,184" |
| Tokens per image or frame | `visionTokens({ width, height, patch, merge, frames: 1 }).tokensPerFrame` | count |
| Total tokens | `visionTokens({ …, frames: media === 'video' ? seconds · fps : 1 }).tokens` | count, `formatCount` |
| Share of the context window | `sharePct(tokens, context)` from `math/memory.js` (owned by `kv-cache`); a printed percentage, never a `shareBar` (a 0.12% segment could not be drawn) | %, one decimal; "does not fit" above 100% |
| Images of this size that fit | `Math.floor(context / tokensPerFrame)` | count |
| "Check my work" | templated: "1008 ÷ 14 = 72 · 72 × 72 = 5,184 · ÷ (2 × 2) = 1,296" | mono, `aria-live="polite"` |

**Try this** (each leads to a named insight)
1. Kimi K3 preset, 1,008 × 1,008: 1,296 tokens. Set merge to **none**: 5,184. Set it back and drag to
   **3,584 × 3,584**: 16,384 tokens, 1.6% of 1M; 64 such images fill the window. → **Insight: tokens grow
   with area; merging neighbors is the main lever on what an image costs.**
2. Set **1,008 × 504**: 648 tokens; the same photo squashed to 1,008 × 1,008 costs 1,296. → **Insight:
   keeping the aspect ratio (dynamic resolution) spends tokens only on real pixels.**
3. **Video**, 448 × 448, 2 fps: 1 minute 30,720 tokens; 10 minutes at 1 fps 153,600 (14.6% of 1M; 58.6%
   of 262,144); **1 hour** at 2 fps 1,843,200, "does not fit". → **Insight: video cost is per frame, so long
   video needs pooling over time or fewer frames; these counts are before any pooling.**

**`math/vision.js`** (pure, no DOM, inputs never mutated; tests first):
```js
// Patch grid. Throws RangeError('patchGrid: width must be a multiple of patch') (and for height).
patchGrid({ width, height, patch }) → { cols, rows, patches }
//   ({ width: 16,   height: 16,   patch: 4 })  → { cols: 4,   rows: 4,   patches: 16 }       (toy)
//   ({ width: 1008, height: 1008, patch: 14 }) → { cols: 72,  rows: 72,  patches: 5184 }
//   ({ width: 3584, height: 3584, patch: 14 }) → { cols: 256, rows: 256, patches: 65536 }
//   ({ width: 1008, height: 504,  patch: 14 }) → { cols: 72,  rows: 36,  patches: 2592 }
//   ({ width: 1000, height: 1000, patch: 14 }) → throws RangeError

// Tokens after merging merge × merge neighbors, for `frames` frames (no temporal pooling).
// Throws RangeError if cols or rows is not divisible by merge.
visionTokens({ width, height, patch, merge = 1, frames = 1 }) → { patches, tokensPerFrame, tokens }
//   ({ 16, 16, patch: 4, merge: 2 })               → { patches: 16,    tokensPerFrame: 4,     tokens: 4 }
//   ({ 1008, 1008, patch: 14, merge: 2 })          → { patches: 5184,  tokensPerFrame: 1296,  tokens: 1296 }
//   ({ 1008, 1008, patch: 14, merge: 3 })          → { …, tokensPerFrame: 576 }
//   ({ 3584, 3584, patch: 14, merge: 2 })          → { patches: 65536, tokensPerFrame: 16384, tokens: 16384 }
//   ({ 1008, 504, patch: 14, merge: 2 })           → { patches: 2592,  tokensPerFrame: 648 }
//   ({ 448, 448, patch: 14, merge: 2, frames: 120 })  → { tokensPerFrame: 256, tokens: 30720 }
//   ({ 448, 448, patch: 14, merge: 2, frames: 600 })  → { tokens: 153600 }
//   ({ 448, 448, patch: 14, merge: 2, frames: 7200 }) → { tokens: 1843200 }
```
Shares use `sharePct` from `math/memory.js` (one definition, shared with `kv-cache`): 1,296 / 1,048,576 =
0.12%; 16,384 → 1.56%; 30,720 → 2.93%; 153,600 → 14.65% (58.59% of 262,144); 1,843,200 → 175.78%.

Tests to write first: the examples above; `tokens === tokensPerFrame × frames`; doubling both sides
multiplies tokens by 4 (README lesson 16: the "grows with area" prose claim as a test); merge 2 divides by 4
and merge 3 by 9; throws on non-divisible sizes; inputs not mutated.

**Reproducer** (run from the repo root on 2026-10-07; output matched every number in §2, §5, §6 and §11):
```sh
node -e '
const grid=({width,height,patch})=>{if(width%patch||height%patch)throw new RangeError("patch");return{cols:width/patch,rows:height/patch,patches:width*height/patch/patch};};
const vt=({width,height,patch,merge=1,frames=1})=>{const g=grid({width,height,patch});if(g.cols%merge||g.rows%merge)throw new RangeError("merge");const t=g.patches/(merge*merge);return{patches:g.patches,tokensPerFrame:t,tokens:t*frames};};
const show=(n,o)=>{const v=vt(o);console.log(n,v.patches,v.tokensPerFrame,v.tokens,(100*v.tokens/1048576).toFixed(2)+"%",(100*v.tokens/262144).toFixed(2)+"%");};
show("toy",{width:16,height:16,patch:4,merge:2});show("1008 m1",{width:1008,height:1008,patch:14});
show("1008 m2",{width:1008,height:1008,patch:14,merge:2});show("1008 m3",{width:1008,height:1008,patch:14,merge:3});
show("3584 m2",{width:3584,height:3584,patch:14,merge:2});show("1008x504 m2",{width:1008,height:504,patch:14,merge:2});
show("2016 m1",{width:2016,height:2016,patch:14});show("vid 1min",{width:448,height:448,patch:14,merge:2,frames:120});
show("vid 10min 1fps",{width:448,height:448,patch:14,merge:2,frames:600});show("vid 1h",{width:448,height:448,patch:14,merge:2,frames:7200});
console.log("fit 3584",Math.floor(1048576/16384),"fit 1008",Math.floor(1048576/1296));
'
```
Output on 2026-10-07: `toy 16 4 4` · `1008 m1 5184 5184 5184 0.49% 1.98%` · `1008 m2 5184 1296 1296 0.12%
0.49%` · `1008 m3 5184 576 576 0.05% 0.22%` · `3584 m2 65536 16384 16384 1.56% 6.25%` ·
`1008x504 m2 2592 648 648 0.06% 0.25%` · `2016 m1 20736 20736 20736 1.98% 7.91%` ·
`vid 1min 1024 256 30720 2.93% 11.72%` · `vid 10min 1fps 1024 256 153600 14.65% 58.59%` ·
`vid 1h 1024 256 1843200 175.78% 703.13%` · `fit 3584 64 fit 1008 809`.

## 7. Show me the math
```tex
\text{patches} = \frac{W}{p}\cdot\frac{H}{p},\qquad
\htmlClass{hl-tok}{\text{tokens per image}} = \frac{W}{p}\cdot\frac{H}{p}\cdot\frac{1}{\htmlClass{hl-merge}{m}^2},
\qquad \text{video: } \times\ \text{frames}
```
```tex
\text{worked: } \frac{1008}{14}\cdot\frac{1008}{14}\cdot\frac{1}{2^2} = 72 \cdot 72 \cdot \tfrac{1}{4} = 1{,}296
```
```tex
z_i = \operatorname{flatten}(\text{patch}_i)\, W_{\text{patch}},\quad
Z = \operatorname{ViT}(z_1, \dots, z_N),\quad
\htmlClass{hl-proj}{x^{\text{img}}_j} = \operatorname{MLP}\big([\,Z_a \,\|\, Z_b \,\|\, Z_c \,\|\, Z_d\,]\big)\in\mathbb{R}^{d_{\text{model}}}
```
Shapes (toy in parentheses): patch [p × p × channels] (4 × 4 × 1 grey); W_patch [p²·channels × d_vit]
(16 × 8); Z [N × d_vit] (16 × 8); a merged group [m²·d_vit] (32); projector output [d_model] (8). Real
encoders are wider (MiniMax-M3's ViT: width 1,280, 32 layers). Patch positions are 2D (row, column; plus time
for video); Qwen2-VL's M-RoPE and MiniMax-M3's 3D RoPE extend `rope`'s rotation to those axes (01 §6). Color
links: `hl-tok` → the token counter; `hl-merge` → the merge labels (frames 4, 7); `hl-proj` → the projector
block (frame 5). KaTeX with `trust: true, strict: false`.

## 8. In today's models (Oct 2026)
Framing paragraph on the page: "In 2026 every vision-capable frontier model on this list keeps a separate
vision encoder and projector; they differ in whether the encoder was trained with the backbone, how large
images may be, and how neighbors are merged. Several leading models see no images at all."

| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| Kimi K3: native multimodal; vision encoder MoonViT-V2, 401M parameters, 27 layers, patch 14, trained from scratch with next-token prediction (a SigLIP-initialized version had unstable gradients); merges 2 × 2; images up to 3,584 × 3,584 | `models.kimi-k3.modalities` (existing, confirmed); `.vision_encoder_params` = 401e6, `.vision_encoder_layers` = 27, `.patch_size` = 14 (proposed, `decoder-anatomy` §12.3); `.vision_merge` = 2, `.max_image_side` = 3584 (proposed *new*) | 01 §6 recipes, mechanics, encoder sizes [C] |
| MiniMax-M3: native multimodal, "mixed-modality training from the very first step"; ViT of 32 layers, width 1,280, patch 14; images up to 2,016 × 2,016; 3D RoPE | `models.minimax-m3.modalities` (existing, confirmed); `.patch_size` = 14 (proposed, `decoder-anatomy`); `.vision_encoder_layers` = 32, `.vision_width` = 1280, `.max_image_side` = 2016 (proposed *new*) | 01 §6 [C] |
| Mistral Large 4 (2026): multimodal input, text output; a 1.6B-parameter vision encoder (reported) | `models.mistral-large-4.modalities` (existing, confirmed); `.vision_encoder_params` = 1.6e9 (proposed, **reported**) | 01 §6 encoder sizes [R], §7 |
| DeepSeek-V4.1-Flash: the first DeepSeek model with a vision encoder: 32 layers, merges 3 × 3, a small MLP projector (reported) | `models.deepseek-v4.1-flash.modalities` (existing, reported); `.vision_merge` = 3, `.vision_encoder_layers` = 32 (proposed, **reported**) | 01 §6 [R] |
| Text-only in 2026: DeepSeek-V4-Pro (multimodality is listed as future work), GLM-5.3, gpt-oss-120b, Qwen3.8 (its card says "text-only model") | `models.deepseek-v4-pro.modalities`, `models.glm-5.3.modalities`, `models.qwen3.8.modalities` (existing, confirmed); `models.gpt-oss-120b.modalities` = "text" (proposed) | 01 §6 "2026 reality check" [C] |
| The adapter recipe: LLaVA (2023): pretrained ViT + small MLP projector + an already-trained LLM | — (timeless, cited) | 01 §6 recipe 1 |
| Video in Kimi K3: frames sampled, attention factorized over space then time in the encoder, pooled over time | `models.kimi-k3.video` = "factorized spatial-temporal attention + temporal pooling" (proposed) | 01 §6 video bullet [C] |

Not shown: audio token rates (01 §6 [U]); MiniMax-M3's merge factor (not in the brief, so its token count
is not computed); a secondary claim that DeepSeek-V4 is natively multimodal (contradicted by its paper, 01 §6).

## 9. Takeaways
1. An image becomes tokens through patches → patch embedding → vision encoder → merge → projector; after
   that the language model treats image tokens like words (frames 1–6).
2. Tokens per image = (width ÷ patch) × (height ÷ patch) ÷ merge²: a 1,008-pixel square is 1,296 tokens in
   Kimi K3, its largest input 16,384, and video multiplies by frames (frames 7–9, try-this).
3. "Native" means trained together from step 0, not "no encoder"; and several 2026 leaders are text-only
   (frame 10, §8).

## 10. Next and go deeper
Next: `model-card` (reading "native multimodal (MoonViT-V2 401M)" on a spec sheet); it lists `multimodal`
in `shared/concepts.json`. In-page links: `decoder-anatomy` (the side lane this page opens), `rope` (2D/3D
positions), `prefill-decode` (image tokens are prefill work).

Go deeper (brief 05 §1.1–1.2; 01 §6): ViT-Explainer (https://arxiv.org/abs/2604.02182), an interactive
research prototype of a vision transformer (snippet-level in the brief; kept only if the live demo resolves at
release, otherwise the page keeps two links) · Liu et al., *LLaVA* (https://arxiv.org/abs/2304.08485), the adapter recipe · Wang et al., *Qwen2-VL*
(https://arxiv.org/abs/2409.12191), dynamic resolution and M-RoPE.

## 11. Key-frame sketch
Frame 6 (image and words in one sequence), desktop width; counts from the §6 reproducer (`toy 16 4 4`).
```text
┌──────────────────────────────────────────────────────────┐
│ ▦▦▦▦     vision     2×2      projector                   │
│ ▦►▦▦▦ →  encoder  → merge →  [32→8]  → ►img₁ ▪▪▪▪▪▪▪▪    │
│ ▦▦▦▦    (all see    16→4                img₂ ▪▪▪▪▪▪▪▪    │
│ ▦▦▦▦     all)                           img₃ ▪▪▪▪▪▪▪▪    │
│ 16 patches                              img₄ ▪▪▪▪▪▪▪▪    │
│                                        [The]₅ [cat]₆     │
│ keys →  1 2 3 4 5 6 7 8                [sat]₇ [down]₈    │
│ sat₇    ■ ■ ■ ■ ■ ■ ■ ░    in this sequence the words    │
│                            sit at positions 5–8          │
├──────────────────────────────────────────────────────────┤
│ Image tokens and words share one sequence. Under the     │
│ causal mask, every word can attend to the image, and the │
│ blocks treat both alike.                                 │
│ [◄] [Pause] [►]  ━━━━━●━━━━━━  6 / 10  speed [1×]        │
└──────────────────────────────────────────────────────────┘
```
"►" is the selection outline on patch 6 and the token it merged into (img₁). The sketch shows only row
"sat" of the 8 × 8 pattern grid; the stage draws all eight rows at 18 px. At 400 px the pipeline wraps into
two rows.

## 12. Open questions for the reviewer
**Data-pass keys** (*new* unless noted): `vision_merge` (kimi-k3 2; deepseek-v4.1-flash 3, reported),
`max_image_side` (kimi-k3 3584; minimax-m3 2016), `vision_width` (minimax-m3 1280), `vision_encoder_layers`
(minimax-m3 32; deepseek-v4.1-flash 32, reported), `vision_encoder_params` (mistral-large-4 1.6e9,
reported), `video` (kimi-k3), `gpt-oss-120b.modalities` = "text". Reused from `decoder-anatomy` §12.3:
`patch_size`, `vision_encoder_params`, `vision_encoder_layers` for kimi-k3.

**Graph changes:** none.

**Judgment calls:** all ruled by the expert review (§13) and applied; none remain open.

## 13. Expert review (2026-10-07) and what changed
Verdict: APPROVE WITH CHANGES (1 Must). Status is now "approved (expert review)". No number changed; the §6
reproducer was re-run and matches.

Rulings applied (lesson 20): image before words, accepted (§4 now says why it differs from
`decoder-anatomy`'s "would be row 5"); the hour of toy video kept with a toy-settings label; ViT-Explainer kept
only if the live demo resolves at release, otherwise two links.

Must (1/1): misconception 5 is corrected by frame 10's page text, not only by §8 (README lesson 31).
Should (7/7, no rebuttals): `sharePct` named as `memory.js`'s (no `shareBar`); §3 "14 pixels in Kimi K3 and
MiniMax-M3"; frame 4 caption "DeepSeek's newest encoder merges 3 by 3" (reported chip kept); frame 3 label
"within one image"; frame 9's toy-settings label; the positions note in §4; the go-deeper fallback.
Nice (2/2): a predict-then-reveal prompt opens frame 7; "64 of these fill a 1M window" on frame 7's stage.
