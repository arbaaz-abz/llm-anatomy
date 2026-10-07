# RoPE (`rope`)

Track: architecture · Section: architecture · Prereqs: attention
Next: `long-context-attention`, `midtraining` (the two slugs whose `prereqs` list `rope` in `shared/concepts.json`)
Status: approved (expert review)
Sources: 01 §1 ("Position" row), §3 (all), §2 (hybrid bullet: K3 NoPE), §7 · 05 §1.1 (RoPE: "Build"; gap: rotation per dimension pair, q·k depends only on relative distance), §2. Nothing beyond the briefs.

Reuses `attention`'s query "sat", its key "cat", head A's vectors and its five-step vocabulary: RoPE changes
step 1 (the scores) and nothing else. `decoder-recap` shows GPT-3's learned position table being swapped
out and links here; `midtraining` owns the staged context-extension *schedules* (8K → 1M) and links here
for the mechanism.

## 1. Learning objective
After this page you can explain why attention needs position information at all (frame 1), rotate a query
and a key pair by pair and compute their rotated score (frames 2–4), show that the score depends only on
how far apart the two tokens are (frame 5), say what the base θ controls (frames 6–7), and explain why a
model cannot simply read past its trained length and how position interpolation and YaRN stretch it
(frames 8–9), plus what partial RoPE and NoPE drop (frame 10).

## 2. Misconceptions to correct
Each one names the frame or try-this that corrects it (README lesson 4).
- **Misconception:** "RoPE adds a position vector to each token, like GPT-3 did." → **Reality:** RoPE
  adds nothing to the residual stream. Inside each attention layer it *rotates* the query and the key
  just before they are dotted; values are not rotated. (01 §3 RoPE bullet; first principles) · corrected
  by frames 2–4
- **Misconception:** "RoPE tells attention each token's absolute position." → **Reality:** the rotated
  score depends only on the offset between the two tokens: "sat" scores "cat" 1.596 at positions 3 and 2,
  at 13 and 12, and at 103 and 102. (01 §3: "The Q·K dot product then depends only on relative offset") ·
  corrected by frame 5, try-this 1
- **Misconception:** "All dimensions carry position the same way." → **Reality:** each pair turns at its
  own speed. Fast pairs complete a turn every few tokens and track nearby order; slow pairs take tens of
  thousands of tokens per turn and track long range. (01 §3: "Low-index pairs spin fast (local order),
  high-index pairs spin slowly (long range)") · corrected by frame 6, try-this 2
- **Misconception:** "A model trained on 4K tokens can read 128K if you just give it more." → **Reality:**
  past the trained length the slow pairs reach angles the model never saw. Extension squeezes positions
  back into the seen range (position interpolation, YaRN) and then trains briefly at the new length;
  gpt-oss stretched 4,096 to 131,072 tokens with YaRN (factor 32). (01 §3 context-extension bullet [C]) ·
  corrected by frames 8–9, try-this 3
- **Misconception:** "Every attention layer needs a position encoding." → **Reality:** a causal mask
  already leaks some order (Kazemnejad et al., 2023), and some layers use no position encoding at all
  (NoPE). Kimi K3's 24 MLA layers have none; its 69 linear-attention layers carry order through their decay,
  and the model reaches 1M tokens without any position-encoding change. (01 §2–§3 [C]) · corrected by
  frame 10

## 3. Hook and intuition (final wording)
**Hook:** In `attention`, "sat" scored "cat" 3.0 no matter where "cat" stood. How does a model learn that
"cat" came just before "sat", and why can it then not simply read ten times further than it was trained
on?

The five steps on `attention` never use positions. Swap two tokens and every score stays the same; only
the causal mask knows about order, and it only says "earlier or not". GPT-3 fixed this by learning one
extra vector per position and adding it to each token's embedding: a table with 2,048 rows, and nothing
for position 2,049 (`decoder-recap`).

RoPE puts position into the query and the key instead. Cut each vector into pairs of numbers and draw each
pair as a clock hand. Before the dot product, turn every hand by the token's position times a fixed speed:
the first pair turns fast, the last pair very slowly. A dot product of two hands only cares about the angle
*between* them, so after turning, the score depends on how far apart the two tokens are, not on where they
sit. That is exactly the information language needs: "the word just before me" means the same thing on
page 1 and page 300.

The slowest hand sets how far the model can tell positions apart, and a constant called the base sets how
slow it is: 2026 models raise it from 10,000 to as much as 10 million. Training fixes which angles the model
has seen. Feed it a longer text and the slow hands swing into angles it has never seen, so the scores stop
meaning anything. Context extension squeezes the positions back into the seen range, and YaRN squeezes only
the slow hands, which keeps the fast ones sharp. The price of squeezing is resolution: neighbors end up
closer together on the dial.

## 4. Visual metaphor
Toy: `attention`'s head A, d_head = 4, so **two pairs**: q_sat = [0, 2 | 0.5, 0], k_cat = [0, 1.5 | 0,
−0.5]. Pair 1 turns 1 radian per token; pair 2 turns 0.1 radian per token (base 100). Real models use
d_head 64–512 (32–256 pairs) and base 10,000 or more; a visible line under the stage says so (README lesson
10): "Two pairs and base 100 are toy choices so both hands visibly move; real heads have 32 to 256 pairs and
a base of 10,000 or more. Vectors are the attention page's numbers; cos and sin need a calculator, like exp
did there."

**Positions** (README lesson 11): token positions are 1-based on screen (The₁ cat₂ sat₃ down₄), and the
rotation uses that same number. Code usually counts from 0; frame 5 shows why it does not matter.

**Terms introduced, one per frame** (README lesson 3): 1 none (order-blindness, from `attention`) · 2
dimension pair · 3 rotation speed · 4 rotated score · 5 offset · 6 wavelength (tokens per full turn) · 7
base θ · 8 position interpolation · 9 YaRN · 10 partial RoPE and NoPE (one idea: "rotate fewer pairs, down
to none"). **Terms assumed from prereqs:** query, key, value, score, d_head, causal mask, the five steps
(`attention`). **Named and deferred:** GPT-3's learned table (`decoder-recap`), staged length training
(`midtraining`), why MLA keeps a separate position key (`kv-compression`), what cheap attention at 1M looks
like (`long-context-attention`).

**Layout** (stage ≈ 580 × 366):
- Top: the four token chips; "sat₃" outlined (the followed query, as on `attention`), "cat₂" labeled
  "key".
- Center-left: q_sat as a row `vector` at `NUMBER_CELL` with a divider between the pairs; under it two
  `dial`s (new glyph, below), "pair 1" and "pair 2", for the query.
- Center-right: the same for k_cat.
- Right: a plain text readout of the two pair dots and the score, and the unrotated 3.0 in gray.
- Frames 6–9: an 8-cell `vector` at `NUMBER_CELL` (8 × 43 = 344 px): the score at offsets 0–7.
- Frames 8–9: under each `dial`, a pale arc for "angles seen in training" (part of the `dial` glyph,
  below) and the hand's current angle.

Glyphs used (from spec §5.1): token, vector, flow (the turning motion has no flow; `flow` only carries q and
k into the score readout, carry `activation`).
New glyph proposed:
- `dial(parent, { x, y, r = 34, vector = null, angle = 0, seen = null, reached = null, label })`: a circle with one hand
  from the center. The hand is the pair (a, b) rotated by `angle`; its **length is the pair's magnitude**
  (printed under the dial as "|·| = 2.00") and its **direction is the rotation**, so each is encoded once.
  `seen` = [0, maxAngle] draws a pale filled sector "angles seen in training" (fill, never an outline; the
  outline stays selection-only); a full circle when the pair has completed a turn. Ticks at 0, ¼, ½, ¾ turn.
  Why: no glyph draws an angle. `vector` cells would print (−0.282, −1.980) but hide the one thing the lesson
  is about, the turn. Reused by: `midtraining` (stretching the context in stages), `multimodal` (one line on
  2D/3D position for image patches), `decoder-recap` (the RoPE swap frame shows one dial).
  **Accepted by the expert review with conditions (a)–(j):** (a) hand length = pair magnitude, printed under
  the dial; (b) the "seen" sector is a pale `--line` fill, never an outline, a full circle when the pair
  turned fully in training; (c) the hand uses `--fg`, nothing on a dial uses the value scale or the track
  accent; (d) the selection outline on a dial means "the followed query's pair" only; (e) ticks at 0, ¼, ½,
  ¾ turn, radians on hover only, degrees never load-bearing; (f) reduced motion draws the end angle; (g)
  `vector` is optional: without it the hand has unit length and no magnitude prints (`midtraining` frame 8);
  (h) `reached = [0, maxAngle]` draws a second, ghosted hand at the largest angle reached, with the plain
  label "never seen" when it leaves the sector (frame 8 here, `midtraining` frame 8); (i) angles past one
  turn print as "2.4 turns" beside the hand; (j) `angle` and `reached` interpolate linearly in `progress`.
  Added to the gallery.

Color: vector cells on the value scale (maxAbs 2). The dial hand uses `--fg`; the "seen" sector uses a
neutral `--line` tint; nothing on a dial uses the value scale. The followed query "sat" keeps the selection
outline (chip and its vector) in every frame. Hovering a term in the math panel outlines the matching glyph.

Small marks (README lesson 15): the pair-dot readouts, the "was 3.0" gray number, the "+10 tokens" label in
frame 5 and the "never seen" label in frame 8 are plain text labels.

## 5. Animation script
Hero: query "sat" (position 3) against key "cat" (position 2), head A. Angles in radians, 3 d.p. for
products.

| # | On screen | What moves | Caption (final wording) | Numbers shown |
|---|---|---|---|---|
| 1 | Four chips; the `attention` score row for "sat" [−1.0, 3.0, 0.5, masked]. A ghost copy of "cat" slides to position 9 (labeled "what if 'cat' were here?"); the "cat" score stays 3.0. | "cat" slides right and back; the score cell does not change. | So far attention cannot see order: "sat" scores "cat" 3.0 whether "cat" is next to it or far away. Only the mask knows anything about position. | q_sat · k_cat = 3.0 at any position · q_sat = [0, 2, 0.5, 0] · k_cat = [0, 1.5, 0, −0.5] |
| 2 | q_sat's vector gets a divider into two pairs; each pair becomes a `dial` hand: pair 1 (0, 2) points straight up, length 2; pair 2 (0.5, 0) points right, length 0.5. Same for k_cat beside it. | The vector splits; each pair flies down into its dial and becomes a hand. | RoPE cuts each query and key into pairs of numbers and treats each pair as a clock hand. Four numbers make two hands. | q pairs (0, 2) and (0.5, 0) · k pairs (0, 1.5) and (0, −0.5) · lengths 2, 0.5, 1.5, 0.5 |
| 3 | The query's dials turn: pair 1 by 3 radians (172°), pair 2 by 0.3 (17°). A plain label "speed: 1 per token" and "0.1 per token". | Both hands sweep to their new angles; the vector cells above retype. | Each hand turns by the token's position times its own speed. "sat" is token 3, so the fast hand turns 3 radians and the slow one 0.3. | q at 3: [−0.282, −1.980, 0.478, 0.148] · pair 1: 3 × 1 = 3 rad · pair 2: 3 × 0.1 = 0.3 rad |
| 4 | The key's dials turn by 2 and 0.2 (cat is token 2). Then a `flow` carries both into the readout: pair 1 dot, pair 2 dot, sum; the old 3.0 shown in gray beside it. | The key's hands sweep; the two pair dots type in, then the sum. | The score is still a dot product, now of the turned vectors. Each pair adds its own part, so the score drops from 3.0 to 1.596. | k at 2: [−1.364, −0.624, 0.099, −0.490] · pair dots 1.621 + (−0.025) = 1.596 · was 3.0 |
| 5 | A plain text prompt first, for one second: "Will the score change when both words move ten tokens later?" Then a "+10 tokens" label: "sat" to position 13, "cat" to 12. All four hands turn further; the angle between each q hand and its k hand stays the same (arc drawn between them). | Every hand sweeps by 10 more speeds' worth; the readout retypes 1.596. | Move both words ten tokens later: every hand turns further, but the angle between each pair stays the same. The score depends only on the offset. | q at 13: [−0.840, 1.815, 0.134, 0.482] · k at 12: [0.805, 1.266, 0.466, −0.181] · score 1.596 at (3, 2), (13, 12), (103, 102) · whole row at +10: [−1.353, 1.596, 0.5] |
| 6 | The 8-cell row "score of q_sat with k_cat at offset 0…7". Under it the two key dials, stepping back one offset at a time. | The cells fill left to right as the offset grows; the fast hand completes a turn just after offset 6, the slow hand barely moves. | The fast hand finishes a turn every 6.3 tokens, so it tracks nearby order; the slow one takes 63. Tokens per full turn is the wavelength. | offsets 0–7: [3, 1.596, −1.298, −3.044, −2.058, 0.731, 2.739, 2.101] · wavelengths 6.3 and 62.8 tokens |
| 7 | The slow dial alone, with a base readout; a page-text strip under the stage: "slowest turn, a head of 128 numbers (64 pairs): base 10,000 → 54,410 tokens · 150,000 (gpt-oss) → 782,338 · 10,000,000 → 48.8 million". | The base readout steps 100 → 10,000; the slow hand's speed drops to 0.01 per token; the score row barely changes. | A constant called the base sets how slow the slowest hand turns. A bigger base means a longer reach, so 2026 models raise it into the millions. | toy: base 100 → speed 0.1; base 10,000 → 0.01 · offsets 0–7 at base 10,000: [3, 1.618, −1.253, −2.977, −1.971, 0.838, 2.866, 2.244] · real 128-wide head: 54,410 tokens (base 10,000) to 48,843,285 (base 10,000,000) |
| 8 | Back to base 100. Label "trained on 16 tokens". Each dial shows its pale "seen" sector: full circle for pair 1, 0 to 1.5 rad for pair 2. Text "at 64 tokens": pair 2's hand swings to 6.3 rad, outside the sector, labeled "never seen". Then "position interpolation, ÷ 4": both speeds drop to a quarter; pair 2 stays inside its sector. | The slow hand overshoots, flashes "never seen", then the squeeze pulls it back; the score row redraws. | Past its training length, a slow hand reaches angles the model never saw. Position interpolation slows every hand by the stretch factor, so all angles look familiar again. | trained offsets 0–15: pair 2 up to 1.5 rad · at 64 tokens: up to 6.3 rad · after ÷ 4: up to 1.575 · new row: [3, 2.9, 2.62, 2.176, 1.596, 0.915, 0.175, −0.578] |
| 9 | Same frame; a branch label "YaRN-style": pair 1 returns to full speed (its sector is a full circle), pair 2 stays at a quarter. The score row redraws close to the original. Text: "cost of squeezing all: 'cat' must now be 4 tokens back to score what 1 back did". | Pair 1's hand speeds back up; the row's first cells snap back. | YaRN squeezes only the slow hands, which never finished a turn in training, and leaves the fast ones alone. Nearby words stay as distinct as before. | YaRN-style row: [3, 1.615, −1.261, −2.989, −1.986, 0.82, 2.843, 2.218] · PI row put 1.596 at offset 4 · gpt-oss (2025): 4,096 → 131,072 tokens, factor 32 |
| 10 | Two strips: "partial RoPE" with only pair 1 rotating (pair 2's dial grayed, label "position-free"), and "NoPE" with both dials grayed. Page text: "Qwen3.5: 25% of dimensions rotate · MiniMax-M3: 50% · Kimi K3's MLA layers: none". | Pair 2's hand stops and grays; then pair 1's. | Some models turn only part of each vector (partial RoPE), leaving the rest for content matching. Some layers turn none at all (NoPE); the causal mask still leaks order. | partial: pair 1 only · NoPE: score = 3.0 again · Qwen3.5 0.25 · MiniMax-M3 0.5 (64 of 128) · Kimi K3: 24 MLA layers without RoPE |

Determinism: every frame is a pure function of (step, progress); dial angles interpolate linearly in
`progress` from their start to end angle. Reduced motion shows each frame's end state. The token chips and
the q/k columns keep their positions from frame 1 to frame 5; the offset row keeps its position from frame 6
to frame 9; "sat" keeps its selection outline throughout.

Caption word counts (README lesson 2; ≤ 30 words, ≤ 2 sentences, no operators): 26 · 23 · 27 · 26 · 26 ·
26 · 27 · 28 · 26 · 29.

Absolutes checked (README lesson 7): frame 1's "cannot see order" is about the five steps as built on
`attention` (no positions anywhere), and the second sentence names the mask. Frame 5's "depends only on the
offset" is the RoPE identity (§7) and holds for every pair. Frame 10's "still leaks order" cites 01 §3
(Kazemnejad et al. 2023).

Branches (README lesson 13): frame 1's ghost "cat" at position 9 is labeled "what if"; frame 9 is labeled
"YaRN-style" and the stretch threads (frames 8–9) close before frame 10 starts.

## 6. Toy
Title on page: "Turn the hands." Two panels, one state object, one `render()`. The stand-in line from §4
is printed above the controls.

**Controls**
| id | Label | Type | Range / values | Default | Presets |
|---|---|---|---|---|---|
| `qPos` | Query position ("sat") | Slider | 1–64, step 1 | 3 | — |
| `kToken` | Key | segmented | The · cat · sat · down | cat | — |
| `kPos` | Key position | Slider | 1–64, step 1 (kept ≤ qPos; the mask) | 2 | chip "same offset, +10" moves both by 10 |
| `base` | Base θ | Slider (snapped) | [100, 10,000, 1,000,000] | 100 | — |
| `target` | Read up to (trained: 16 tokens) | Slider (snapped) | [16, 32, 64] tokens | 16 | — |
| `stretch` | Stretch method | segmented | none · squeeze all (PI) · squeeze slow pairs (YaRN-style) | none | — |

**Live outputs**
| Output | Formula / `math/` function | Units / format |
|---|---|---|
| Rotated q and k (cells and dials) | `rotatePairs(vec, pos, ropeFrequencies(4, base))` | 4 numbers, 3 d.p.; dial angles in radians and degrees |
| Pair dots and score | `ropeScore(q, k, { qPos, kPos, base })`, `.pairs` | 3 d.p.; the unrotated `q · k` in gray |
| "Same offset, 10 later" check | `ropeScore(q, k, { qPos: qPos + 10, kPos: kPos + 10, base })` | must print the same number |
| Score by offset 0–7 | `scoreByOffset(q, k, { freqs, offsets: [0…7] })` with `freqs` from the stretch method | 8-cell `vector` at `NUMBER_CELL` |
| Wavelengths | `wavelengths(ropeFrequencies(4, base))` | tokens per turn, 1 d.p. |
| Seen vs reached angle per pair | `angleCoverage(freqs, { trainedLength: 16, length: target })` | "seen up to 1.5 rad · reaches 6.3 rad · never seen" |
| Real-head readout | `wavelengths(ropeFrequencies(128, base)).at(-1)` for the preset bases in §8 | tokens, `formatCount` |

**Try this** (each leads to a named insight)
1. Key **cat**, positions 3 and 2: score 1.596. Tap **same offset, +10**: 13 and 12, still 1.596. Now set
   the key position to 1 (offset 2): −1.298. → **Insight: RoPE scores depend on how far apart two tokens
   are, never on where they are.**
2. Watch the offset row at base **100**: [3, 1.596, −1.298, −3.044, …]. Switch to **10,000**: [3, 1.618,
   −1.253, −2.977, …]; the slow hand nearly freezes (0.01 per token, a turn every 628 tokens), the fast one
   is unchanged. → **Insight: the base only changes the slow pairs, and the slow pairs are what reach far.**
   The page-text line shows the real version: a head of 128 numbers (64 pairs) turns its slowest pair once every 54,410 tokens
   at base 10,000 and once every 48.8 million at base 10,000,000.
3. Set **Read up to 64** with stretch **none**: pair 2 "reaches 6.3 rad, never seen". Pick **squeeze all**:
   pair 2 is back in range, but the offset row becomes [3, 2.9, 2.62, 2.176, 1.596, …]: offset 4 now scores
   what offset 1 did. Pick **squeeze slow pairs**: [3, 1.615, −1.261, −2.989, …], close to the original. →
   **Insight: stretching trades resolution for reach; YaRN spends that resolution only where the model had
   it to spare.**

**`math/rope.js`** (pure, no DOM, inputs never mutated; tests first). Pairs are adjacent dimensions (0,1),
(2,3), … as in the RoPE paper; many codebases pair dimension i with i + d/2 instead, which is the same math
with the dimensions reordered (§7 note).

```js
// Speed of each pair: base^(−2i/d) for i = 0 … d/2 − 1. Throws RangeError if d is odd or base ≤ 1.
ropeFrequencies(dHead, base) → number[]
//   (4, 100)    → [1, 0.1]                (exact)
//   (4, 10_000) → [1, 0.01]               (exact)
//   (128, 10_000).at(-1) → 1.155e-4       (slowest pair of a 128-wide head)

// Rotate each pair (x, y) by pos · freq: (x cos a − y sin a, x sin a + y cos a).
rotatePairs(vec, pos, freqs) → number[]
//   ([0, 2, 0.5, 0], 3, [1, 0.1])     → [−0.282, −1.980, 0.478, 0.148]
//   ([0, 1.5, 0, −0.5], 2, [1, 0.1])  → [−1.364, −0.624, 0.099, −0.490]
//   ([0, 2, 0.5, 0], 13, [1, 0.1])    → [−0.840, 1.815, 0.134, 0.482]
//   (v, 0, freqs)                     → v (exact; no rotation at position 0)
//   rotation preserves each pair's length (|(−0.282, −1.980)| = 2.000)

// Rotated dot product of q at qPos and k at kPos; pairs[i] is pair i's contribution.
ropeScore(q, k, { qPos, kPos, base, freqs = ropeFrequencies(q.length, base) }) → { score, pairs }
//   (q_sat, k_cat, { qPos: 3,   kPos: 2,   base: 100 })    → { score: 1.596, pairs: [1.621, −0.025] }
//   (q_sat, k_cat, { qPos: 13,  kPos: 12,  base: 100 })    → { score: 1.596, … }
//   (q_sat, k_cat, { qPos: 103, kPos: 102, base: 100 })    → { score: 1.596, … }
//   (q_sat, k_cat, { qPos: 3,   kPos: 2,   base: 10_000 }) → { score: 1.618, pairs: [1.621, −0.002] }
//   (q_sat, k_sat, { qPos: 3, kPos: 3, base: 100 })        → { score: 0.5 }   (offset 0: unchanged; exact)

// Score at each offset (query at offset + 1, key at 1).
scoreByOffset(q, k, { freqs, offsets }) → number[]
//   (q_sat, k_cat, { freqs: [1, 0.1],     offsets: [0…7] }) → [3, 1.596, −1.298, −3.044, −2.058, 0.731, 2.739, 2.101]
//   (q_sat, k_cat, { freqs: [1, 0.01],    offsets: [0…7] }) → [3, 1.618, −1.253, −2.977, −1.971, 0.838, 2.866, 2.244]
//   (q_sat, k_cat, { freqs: [0.25, 0.025], offsets: [0…7] }) → [3, 2.9, 2.62, 2.176, 1.596, 0.915, 0.175, −0.578]   (PI ÷ 4)
//   (q_sat, k_cat, { freqs: [1, 0.025],   offsets: [0…7] }) → [3, 1.615, −1.261, −2.989, −1.986, 0.82, 2.843, 2.218]  (YaRN-style)

// Tokens per full turn: 2π / freq.
wavelengths(freqs) → number[]
//   ([1, 0.1]) → [6.283, 62.832]
//   (ropeFrequencies(128, 10_000)).at(-1)     → 54_410 (rounded)
//   (ropeFrequencies(128, 150_000)).at(-1)    → 782_338
//   (ropeFrequencies(128, 10_000_000)).at(-1) → 48_843_285

// Stretch the trained range by `factor`. 'pi': every freq ÷ factor. 'yarn-simple': a pair keeps its freq if its
// wavelength ≤ trainedLength (it already turned fully in training), else freq ÷ factor. This is a stated
// simplification of YaRN, which ramps smoothly between the two cases and adds an attention-temperature fix (§7).
stretchFrequencies(freqs, { factor, method, trainedLength }) → number[]
//   ([1, 0.1], { factor: 4, method: 'pi',          trainedLength: 16 }) → [0.25, 0.025]
//   ([1, 0.1], { factor: 4, method: 'yarn-simple', trainedLength: 16 }) → [1, 0.025]
//   ([1, 0.1], { factor: 1, method: 'none',        trainedLength: 16 }) → [1, 0.1]

// For each pair: the largest angle seen in training (offsets 0 … trainedLength − 1), the largest at `length`,
// and whether every angle reached was seen (true if the pair turned fully in training or stays inside).
angleCoverage(freqs, { trainedLength, length, trainedFreqs = freqs }) → [{ seenMax, reachedMax, allSeen }]
//   ([1, 0.1],     { trainedLength: 16, length: 64 })                         → [{ 15, 63, true }, { 1.5, 6.3, false }]
//   ([0.25, 0.025], { trainedLength: 16, length: 64, trainedFreqs: [1, 0.1] }) → pair 2 reaches 1.575 (≈ seen 1.5)
```

Tests to write first: rotation preserves each pair's length; `ropeScore` is identical for (m, n) and
(m + s, n + s) for 20 random shifts (|Δ| < 1e-9; the frame 5 claim, README lesson 16); offset 0 gives the
unrotated dot; `scoreByOffset(…)[0] === q · k`; `stretchFrequencies` with factor 1 is the identity; every
example above to 3 d.p.; `ropeFrequencies` throws on odd d.

**Reproducer** (run from the repo root on 2026-10-07; output matched every number in §2, §5, §6 and §11):
```sh
node -e '
const freqs=(d,base)=>Array.from({length:d/2},(_,i)=>Math.pow(base,-2*i/d));
const rot=(v,pos,f)=>v.map((_,j)=>{const i=j>>1,a=pos*f[i],x=v[2*i],y=v[2*i+1];return j%2===0?x*Math.cos(a)-y*Math.sin(a):x*Math.sin(a)+y*Math.cos(a);});
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0),score=(q,k,m,n,f)=>dot(rot(q,m,f),rot(k,n,f));
const r=(x,d=3)=>Array.isArray(x)?x.map(v=>r(v,d)):Number(x.toFixed(d));
const q=[0,2,0.5,0],kc=[0,1.5,0,-0.5],kt=[1,-0.5,0,0.5],ks=[-0.5,0,1,0.5];
for(const base of [100,10000]){const f=freqs(4,base);
 console.log(base,"q@3",r(rot(q,3,f)),"k@2",r(rot(kc,2,f)),"q@13",r(rot(q,13,f)),"k@12",r(rot(kc,12,f)),
  "pairs",r(dot(rot(q,3,f).slice(0,2),rot(kc,2,f).slice(0,2))),r(dot(rot(q,3,f).slice(2),rot(kc,2,f).slice(2))),
  "score",r(score(q,kc,3,2,f)),r(score(q,kc,13,12,f)),r(score(q,kc,103,102,f)),
  "row+10",r([score(q,kt,13,11,f),score(q,kc,13,12,f),score(q,ks,13,13,f)]),
  "offsets",JSON.stringify(r(Array.from({length:8},(_,o)=>score(q,kc,o+1,1,f)))));}
console.log("identity pair2",(0.25*Math.sin(-0.1)).toFixed(3),"pair1",(3*Math.cos(-1)).toFixed(3));
for(const b of [10000,150000,5e6,8e6,1e7])console.log("d128",b,Math.round(2*Math.PI/freqs(128,b)[63]));
const f=freqs(4,100),pi=f.map(x=>x/4),ya=f.map(x=>2*Math.PI/x<=16?x:x/4);
console.log("seen",r(f.map(x=>x*15)),"reached",r(f.map(x=>x*63)),"pi reached",r(pi.map(x=>x*63)),
 "pi",JSON.stringify(r(Array.from({length:8},(_,o)=>score(q,kc,o+1,1,pi)))),"yarn",JSON.stringify(r(Array.from({length:8},(_,o)=>score(q,kc,o+1,1,ya)))));
'
```
Output on 2026-10-07: `100 q@3 [-0.282,-1.98,0.478,0.148] k@2 [-1.364,-0.624,0.099,-0.49] q@13
[-0.84,1.815,0.134,0.482] k@12 [0.805,1.266,0.466,-0.181] pairs 1.621 -0.025 score 1.596 1.596 1.596
row+10 [-1.353,1.596,0.5] offsets [3,1.596,-1.298,-3.044,-2.058,0.731,2.739,2.101]` ·
`10000 … pairs 1.621 -0.002 score 1.618 1.618 1.618 row+10 [-1.397,1.618,0.5] offsets
[3,1.618,-1.253,-2.977,-1.971,0.838,2.866,2.244]` · `d128 10000 54410` · `d128 150000 782338` ·
`d128 5000000 24687577` · `d128 8000000 39211104` · `d128 10000000 48843285` · `seen [15,1.5] reached
[63,6.3] pi reached [15.75,1.575] pi [3,2.9,2.62,2.176,1.596,0.915,0.175,-0.578] yarn
[3,1.615,-1.261,-2.989,-1.986,0.82,2.843,2.218]`. `identity pair2 -0.025 pair1 1.621` (the §7 worked line). Angles in degrees for frame 3: 3 rad = 171.9°,
0.3 rad = 17.2°.

## 7. Show me the math
```tex
\theta_i = \htmlClass{hl-base}{\beta}^{-2i/d},\quad i = 0,\dots,\tfrac{d}{2}-1
\qquad
R(a) = \begin{pmatrix}\cos a & -\sin a\\ \sin a & \cos a\end{pmatrix}
```
```tex
\tilde q^{(i)}_m = R(m\,\theta_i)\, q^{(i)},\qquad \tilde k^{(i)}_n = R(n\,\theta_i)\, k^{(i)}
\qquad (q^{(i)} = \text{pair } i)
```
```tex
\htmlClass{hl-score}{\tilde q_m \cdot \tilde k_n}
= \sum_i q^{(i)\top} R\big((n-m)\,\theta_i\big)\, k^{(i)}
\quad\text{depends only on } \htmlClass{hl-off}{n - m}
```
```tex
\text{worked } (n - m = -1): \underbrace{3\cos(-1)}_{\text{pair 1}} + \underbrace{0.25\,\sin(-0.1)}_{\text{pair 2}}
= 1.621 - 0.025 = 1.596
```
```tex
\text{wavelength}_i = \frac{2\pi}{\theta_i};\qquad
\text{PI: } \theta_i \to \theta_i / s;\qquad
\text{YaRN-style: } \theta_i \to \begin{cases}\theta_i & 2\pi/\theta_i \le L_{\text{train}}\\ \theta_i / s & \text{otherwise}\end{cases}
```
Shapes: q, k [d_head] (4), d_head / 2 pairs (2), one θ per pair. The pair-2 term in the worked line uses
the 2D identity (a, b)ᵀR(φ)(c, d) = (ac + bd) cos φ + (bc − ad) sin φ with q pair (0.5, 0), k pair (0, −0.5)
and φ = (n − m) θ₂ = −0.1 (the key is one token before the query): ac + bd = 0 and bc − ad = 0.25, so
0.25 · sin(−0.1) = −0.025. Pair 1 is unchanged: 3 cos(−1) = 3 cos(1) = 1.621. Notes printed on the panel: real YaRN ramps smoothly between "keep" and "squeeze" by each
pair's turns per trained length, and also rescales attention by a temperature (01 §3); NTK-aware scaling
raises the base instead of slowing positions (01 §3); pairing dimension i with i + d/2 is the same rotation
on reordered dimensions. Color links: `hl-base` → the base readout (frame 7); `hl-score` → the score
readout (frames 4–5); `hl-off` → the offset arc (frame 5). KaTeX with `trust: true, strict: false`.

## 8. In today's models (Oct 2026)
Framing paragraph on the page: "Almost every 2026 model rotates its queries and keys (in its softmax-attention layers). The differences are
the base, how much of each head rotates, which layers rotate at all, and how the model was stretched to its
final length. The mainstream recipe for 1M tokens is a very large base, staged length training
(`midtraining`), attention that is cheap at length (`long-context-attention`), and sometimes YaRN."

| Claim shown on page | data/*.json entry.key | Brief source |
|---|---|---|
| GPT-3 (2020): learned absolute positions, a table of 2,048 rows | `models.gpt-3.context_length` = 2048, `.positional` = "learned" (proposed; `decoder-anatomy` §12.3 gives `context_length`) | 01 §1 "Position" row (background) |
| RoPE base θ: DeepSeek-V4 10,000 (160,000 for its compressed streams) · gpt-oss 150,000 · MiniMax-M3 5,000,000 · GLM-5.3 8,000,000 · Qwen3.8 10,000,000 | `models.deepseek-v4-pro.rope_theta` = 10000, `.rope_theta_compressed` = 160000; `models.gpt-oss-120b.rope_theta` = 150000; `models.minimax-m3.rope_theta` = 5e6; `models.glm-5.3.rope_theta` = 8e6; `models.qwen3.8.rope_theta` = 1e7 (all proposed; sources the configs 01 §2–§3 link) | 01 §3 RoPE bullet [C] |
| Partial RoPE: MiniMax-M3 rotates 64 of 128 dimensions (0.5); Qwen3.5 rotates 25% | `models.minimax-m3.partial_rotary_factor` = 0.5; `models.qwen3.5-397b.partial_rotary_factor` = 0.25 (proposed; *new* entry `qwen3.5-397b`, source its HF config.json) | 01 §3 partial-RoPE bullet [C] |
| YaRN: gpt-oss stretched 4,096 → 131,072 tokens (factor 32); DeepSeek-V4 uses factor 16 | `models.gpt-oss-120b.yarn_factor` = 32, `.rope_original_context` = 4096, `.context_length` = 131072; `models.deepseek-v4-pro.yarn_factor` = 16 (proposed) | 01 §3 context-extension bullet [C]; 01 §7 (131K) |
| NoPE: Kimi K3's 24 MLA layers use no position encoding (order comes from its linear-attention layers' decay) and it reaches 1M tokens without any position-encoding change | `models.kimi-k3.attention` (existing, confirmed; string says "(NoPE)"), `.context_length` = 1,000,000 (existing) | 01 §3 NoPE bullet [C]; §2 hybrid bullet [C] |
| Most frontier open models list about 1M positions: DeepSeek-V4-Pro, Kimi K3, GLM-5.3, MiniMax-M3 | `models.*.context_length` (existing, confirmed) | 01 §3 "What is mainstream for 1M" [C] |

Not shown: Qwen3.8's method for going from 262K to about 1.01M (01 §3 [U]); Mistral Large 4's measured
context (`model-card` shows the conflict); Llama 4 iRoPE (background only, no data entry).

## 9. Takeaways
1. RoPE turns each pair of query and key numbers by position × the pair's speed, inside every attention
   layer; nothing is added to the token's vector (frames 2–4).
2. Because a dot product only sees the angle between two hands, the score depends only on how far apart two
   tokens are (frame 5, try-this 1). Fast pairs see nearby order, slow pairs far order, and the base sets
   how slow the slowest is (frames 6–7, try-this 2).
3. Past the trained length the slow pairs hit unseen angles. Position interpolation squeezes every pair and
   blurs neighbors; YaRN squeezes only the slow pairs. Some heads rotate only part of each vector, and some
   layers none (frames 8–10, try-this 3).

## 10. Next and go deeper
Next: `long-context-attention` (making 1M tokens affordable once positions reach that far) and
`midtraining` (the staged length training that teaches the stretched angles); both list `rope` in
`shared/concepts.json`. In-page links: `decoder-recap` (GPT-3's table → RoPE), `kv-compression` (MLA's
separate position key).

Go deeper (brief 05 §1.1, 01 §3): Su et al., *RoFormer* (https://arxiv.org/abs/2104.09864), the RoPE paper ·
Peng et al., *YaRN* (https://arxiv.org/abs/2309.00071) · Chen et al., *Position Interpolation*
(https://arxiv.org/abs/2306.15595). (Brief 05 lists only scattered RoPE demos, e.g. the ucalyptus
visualizer, at snippet level; none is linked.)

## 11. Key-frame sketch
Frame 5 (shift both by ten), desktop width. Numbers from the §6 reproducer (`q@13`, `k@12`, `score`).
```text
┌──────────────────────────────────────────────────────────┐
│ [The]₁ [cat]₂ ►[sat]₃ [down]₄         +10 tokens →       │
│                                    sat at 13, cat at 12  │
│  query "sat" @13               key "cat" @12             │
│ │−0.840│ 1.815│ 0.134│ 0.482│   │0.805│1.266│0.466│−0.181│
│   pair 1      pair 2             pair 1     pair 2       │
│    ( ↖ )       ( ↗ )              ( ↗ )      ( ↘ )       │
│   |q|=2.00    |q|=0.50           |k|=1.50   |k|=0.50     │
│                                                          │
│  angle between q and k hands: unchanged                  │
│  pair dots 1.621 + (−0.025) = 1.596   (unrotated: 3.0)   │
├──────────────────────────────────────────────────────────┤
│ Move both words ten tokens later: every hand turns       │
│ further, but the angle between each pair stays the same. │
│ The score depends only on the offset.                    │
│ [◄] [Pause] [►]  ━━━━━●━━━━━━  5 / 10  speed [1×]        │
└──────────────────────────────────────────────────────────┘
 Two pairs and base 100 are toy choices so both hands
 visibly move; real heads: 32 to 256 pairs, base 10,000+.
```
"►" is the selection outline on the followed query "sat". The rotated vectors print at `NUMBER_CELL`
(4 × 43 = 172 px each, two side by side = 360 px). At 400 px the q and k columns stack.

## 12. Open questions for the reviewer
**Data-pass keys** (all *new* unless noted): `rope_theta` for deepseek-v4-pro (10,000; plus
`rope_theta_compressed` 160,000), gpt-oss-120b (150,000), minimax-m3 (5,000,000), glm-5.3 (8,000,000),
qwen3.8 (10,000,000); `partial_rotary_factor` for minimax-m3 (0.5) and a new entry `qwen3.5-397b` (0.25,
its HF config.json, confirmed per 01 §3); `yarn_factor` for gpt-oss-120b (32, with `rope_original_context`
4,096) and deepseek-v4-pro (16); `gpt-oss-120b.context_length` = 131,072 (01 §7); `gpt-3.positional` =
"learned" (or reuse `decoder-anatomy`'s config field). All from 01 §3 [C]; the configs are linked in 01 §2–§3.

**Graph changes:** none.

**Judgment calls:** all ruled by the expert review (§13) and applied; none remain open.

## 13. Expert review (2026-10-07) and what changed
Verdict: APPROVE WITH CHANGES (1 Must). Status is now "approved (expert review)". The §6 reproducer was
re-run and now also prints the §7 identity's two terms (−0.025, 1.621).

Rulings applied (lesson 20): toy base 100 and the "YaRN-style" simplification (no ramp constants) accepted
with the §7 note; `dial` accepted with conditions (a)–(j), written into §4 and widened for `midtraining`;
1-based positions drive the arithmetic (frame 5 proves the origin does not matter; tests check 20 random
shifts); RoFormer, YaRN and PI links accepted (primary papers cited in brief 01 §3).

Must (1/1): §7's worked note uses the printed equation's conventions: identity (ac + bd) cos φ + (bc − ad)
sin φ with φ = (n − m) θ₂ = −0.1; the worked TeX line is 3 cos(−1) + 0.25 sin(−0.1) (README lesson 30).
Should (5/5 applicable, no rebuttals): misconception 5 separates the causal-mask reason from K3's decay;
frame 6 "just after offset 6"; frame 10 caption names partial RoPE and NoPE (29 words); "a head of 128
numbers (64 pairs)" in frame 7 and try-this 2; §8 framing "(in its softmax-attention layers)". The go-deeper
item needed no change.
Nice (2/2): a predict-then-reveal prompt opens frame 5; frame 7's strip prints base 150,000 (782,338 tokens).
