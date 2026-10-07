# 05 - Explainer prior art and technique brief

Date: 2026-10-07. Scope: what to link to, what to build ourselves, and which libraries/patterns to use for three static sites (Architecture, Training, Serving) plus a hub with an interactive concept map.

Verification note: every CDN URL in sections 3-4 was fetched and returned HTTP 200 on 2026-10-07 (sizes given). Explainer descriptions come from page/README/paper fetches, except ViT-Explainer/EL-VIT, Tiktokenizer, the Smol Playbook, the ucalyptus RoPE demo (https://ucalyptus-rope.static.hf.space/) and Ultra-Scale widget details, which are search-snippet-level only (URLs return 200 but I did not read them in depth).

---

## 1. Existing explainers

### 1.1 Concept -> "link out / build / both" matrix

Our site-specific differentiator is the **in-browser toy with sliders and live numbers**, so "link out" means "do not re-teach the full walkthrough; give a 1-paragraph summary plus a link, then build only the toy the explainer lacks".

| Concept (track) | Best existing resource | Verdict |
|---|---|---|
| Attention, full forward pass, next-token (Arch) | Transformer Explainer; Bycroft LLM Visualization; 3Blue1Brown ch.5-6 | **Link out** for the walkthrough; **build** a tiny attention-weights toy (Q/K/V with 3-4 tokens, hand-checkable numbers) |
| Sampling / temperature (Arch/Serving) | Transformer Explainer has a temperature slider | **Build** a small top-k/top-p/temperature toy; it is cheap and reused across tracks |
| RoPE (Arch) | Scattered demos only (ucalyptus RoPE viz, snippet-level) | **Build** (gap: rotation per dimension pair, q.k depends only on relative distance) |
| KV cache (Arch/Serving) | llm-inference-explained ch.2; Weng inference post | **Build** a memory-vs-context calculator; link for the generation-loop walkthrough |
| GQA / MLA / SWA / QK-Norm (Arch) | Raschka LLM Architecture Gallery (concept explainers + memory calculator + model diff) | **Link out** for model survey; **build** KV-bytes-per-token comparison slider (MHA vs GQA vs MLA) |
| MoE (Arch) | Grootendorst "Visual Guide to MoE" (static) | **Both**: link, plus build a router top-k toy with load-balance numbers |
| Multimodal / ViT patches (Arch) | ViT-Explainer (research prototype, arXiv 2604.02182), EL-VIT; no polished LLM-facing explainer found | **Build** patch grid -> token count toy (resolution x patch size -> tokens); link ViT-Explainer |
| Pretraining, scaling laws (Train) | Scaling Book, Ultra-Scale Playbook (text-first) | **Build** loss/compute/token-budget toy; link for depth |
| SFT / RLHF / RLVR / GRPO (Train) | UNIPO (poloclub, REINFORCE/PPO/GRPO/DAPO/Dr.GRPO, token-level coloring) | **Link out** for algorithm comparison; **build** a group-relative-advantage toy (8 sampled answers, rewards, normalised advantages) |
| GPUs, memory, distributed parallelism (Train) | Ultra-Scale Playbook; Scaling Book (roofline, sharding) | **Both**: link; build a memory-per-GPU calculator (params, optimizer states, ZeRO stage, TP/PP/DP sliders) and a pipeline-bubble animation |
| Batching, continuous batching, PagedAttention, spec decoding, quantization, disaggregation (Serve) | llm-inference-explained (14 chapters, MIT, Next.js + D3) covers nearly the whole track; vLLM blog; Grootendorst quantization | **Both**: link as "deep dive"; our toys must be smaller and standalone (slot timeline, block table, accept-rate speedup, bits-vs-bytes, TTFT/TPOT) |

### 1.2 Resource catalogue

Interactive, in-browser tools

| Resource | URL | Pedagogy / interaction | Stack (verified where stated) |
|---|---|---|---|
| Transformer Explainer (Georgia Tech Polo Club; CHI 2026 version) | https://poloclub.github.io/transformer-explainer/ ; paper https://arxiv.org/abs/2408.04619 ; code https://github.com/poloclub/transformer-explainer | Live GPT-2 in browser; type your own text; temperature slider; click an operation to expand it from overview to math ("smooth transitions across abstraction levels") | Svelte, Vite, TypeScript, Tailwind, ONNX Runtime (README) |
| LLM Visualization (Brendan Bycroft) | https://bbycroft.net/llm ; code https://github.com/bbycroft/llm-viz | 3D walkthrough of real tensors through a tiny minGPT (sorts letters), with GPT-2/3 scale views; guided narration steps | Next.js, TypeScript, WebGL, MIT (README) |
| AnimatedLLM (EACL 2026 TeachNLP) | https://animatedllm.github.io ; https://github.com/kasnerz/animated-llm ; https://arxiv.org/abs/2601.04213 | Step buttons through tokenization, embeddings, attention, layers; hover highlight; precomputed traces of open models, so no inference cost | React; runs fully client-side |
| UNIPO | https://poloclub.github.io/unipo/ ; https://github.com/poloclub/unipo | Three linked views (training explorer, step inspector with per-token green/pink coloring, algorithm explainer that decomposes objectives into aggregation / per-token term / constraint) | Svelte + D3, reads JSON logs |
| LLM Inference Explained (Lynskey) | https://llm-inference-explained.vercel.app ; https://github.com/BrendanJamesLynskey/llm-inference-explained | 14 chapters; every chapter has Concept / Maths / Code layers to toggle; simulator in a Web Worker | Next.js 14, TypeScript, D3 over React SVG, MDX, MIT |
| Raschka LLM Architecture Gallery | https://sebastianraschka.com/llm-architecture-gallery/ | 100+ model cards, filters, side-by-side diff, memory calculator, light/dark | Static page |
| TensorFlow Playground / MLU-Explain / Seeing Theory | https://playground.tensorflow.org/ ; https://mlu-explain.github.io/ ; https://seeing-theory.brown.edu/ | Classic slider-and-live-plot and scrollytelling patterns worth imitating in tone | - |
| Tiktokenizer | https://tiktokenizer.vercel.app/ | Type text, see tokens instantly | - |

Text/diagram-first (link, do not duplicate)

- Jay Alammar, Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/ (the canonical static-diagram style: one colour per tensor, shapes labelled).
- 3Blue1Brown Deep Learning ch.5-7: https://3blue1brown.com/lessons/attention , https://3blue1brown.com/lessons/mlp (video-first; excellent concrete examples).
- Grootendorst visual guides: https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts , https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization
- Raschka, Big LLM Architecture Comparison: https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison
- Lilian Weng: https://lilianweng.github.io/ ; Chip Huyen: https://huyenchip.com/blog/
- HF Ultra-Scale Playbook: https://huggingface.co/spaces/nanotron/ultrascale-playbook (reported to use interactive plots and widgets per https://the-decoder.com/hugging-face-explains-how-train-large-ai-models-in-the-ultra-scale-playbook/ ; I could not read widget details myself). HF Smol Training Playbook: https://huggingfacetb-smol-playbook-toc.static.hf.space/
- Google, How to Scale Your Model: https://jax-ml.github.io/scaling-book/ (12 chapters: roofline, sharding, training/inference of transformers; built around "problems to work for yourself").
- vLLM: https://blog.vllm.ai/2023/06/20/vllm.html (PagedAttention origin), https://vllm.ai/blog/2025-09-05-anatomy-of-vllm ; PyTorch speculative decoding guide https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/
- Explorable masters: Bartosz Ciechanowski https://ciechanow.ski/ ; Red Blob Games https://www.redblobgames.com/ ; Nicky Case https://ncase.me/ (e.g. https://ncase.me/loopy/ , https://ncase.me/trust/) ; Explorables index https://explorabl.es/ ; Distill https://distill.pub/
- 2025-2026 newcomers worth knowing: AnimatedLLM, UNIPO, llm-inference-explained, ViT-Explainer (all above).

Tech-stack observation: the polished newcomers use Svelte/React/Next + D3. We are constrained to vanilla JS, so copy their **patterns**, not their stack. The two that are closest in spirit to what we want (self-contained, trace/simulator driven) are AnimatedLLM (precomputed traces) and llm-inference-explained (pure-math sims in a Worker).

Biggest overlap risk: llm-inference-explained already covers the Serving track end-to-end. Our Serving site should position itself as shorter, phone-friendly, and calculator-centric, and link to it for the long form.

---

## 2. Pedagogical patterns for explorable explanations

Foundations
- Bret Victor, Explorable Explanations: http://worrydream.com/ExplorableExplanations/ . Three ideas: **reactive documents** (reader changes the author's assumptions and sees consequences), **explorable examples** (build intuition by watching results change), **contextual information** (just-in-time definitions). Goal: text as "an environment to think in".
- Victor, Up and Down the Ladder of Abstraction: http://worrydream.com/LadderOfAbstraction/ . Move between concrete instance and general rule; show the parameter space, not one run.
- Victor, Tangle (scrubbable inline numbers): http://worrydream.com/Tangle/ . The pattern is reimplementable in about 30 lines of vanilla JS (pointer-drag on a `<span>`, plus keyboard arrows for accessibility); do not load a library.
- Hohman, Conlen, Heer, Chau, Communicating with Interactive Articles (Distill 2020): https://distill.pub/2020/communicating-with-interactive-articles/ . Five affordances: connect people and data, make systems playful, prompt self-reflection, personalise reading, reduce cognitive load.
- Distill, Research Debt: https://distill.pub/2017/research-debt/ (motivation: explanation is a first-class contribution).

Concrete patterns to adopt (each is visible in the tools above)
1. **Step-through with transport controls**: play / pause / step / scrub / speed, and a keyboard map (space, arrows). AnimatedLLM and Bycroft use stepped narration; GSAP-style timeline seeking maps 1:1 onto a scrub bar.
2. **Concrete before abstract**: open each toy with 3-4 tokens, 2-4 dims, numbers small enough to verify with a calculator. Show the arithmetic next to the picture, then offer a "scale it up to Llama-3 8B" button that swaps in real dimensions.
3. **Progressive disclosure / "show me the math"**: collapsed panel under each toy. llm-inference-explained uses Concept / Maths / Code toggles; Transformer Explainer expands operations in place. Use native `<details>` for zero-JS, accessible disclosure.
4. **Linked highlighting between equation and diagram**: hover a term in the formula and the matching box lights up, and vice versa. With KaTeX, use `\htmlClass{hl-q}{Q}` (needs `trust: true` and `strict: false` for htmlExtension; documented at https://katex.org/docs/supported.html) and attach hover handlers by class.
5. **Live numeric readouts next to every slider** (tabular numerals, units, bytes -> GB conversion). Always show the derived quantity the lesson is about (e.g. KV GB/request), not only raw inputs.
6. **Predict-then-reveal**: ask the reader to guess (e.g. "will doubling GQA groups halve the cache?") before moving the slider. Scaling Book's "problems to work yourself" is the same idea.
7. **Presets, not blank sliders**: named presets (Llama-3 8B, DeepSeek-V3, a toy) so readers land on real numbers; a Reset button; shareable state in the URL hash.
8. **Linked views** (UNIPO): one change updates summary chart, detail table, and diagram together; keep a single state object and one `render()`.
9. **Overview -> detail ladder** (Transformer Explainer): every page begins with the whole pipeline diagram where each block links to its detail page; ties directly to the hub's concept map.
10. **Pause-on-demand and no autoplay** for anything longer than 5 seconds (WCAG 2.2.2 Pause, Stop, Hide: https://www.w3.org/WAI/WCAG21/Understanding/pause-stop-hide.html).

---

## 3. Concept-map / knowledge-graph UI for the hub

Requirement: about 30-45 nodes, 3 tracks, prerequisite edges (a DAG), readable on a phone, loadable from cdnjs / jsDelivr / unpkg.

### 3.1 Comparison (sizes measured on 2026-10-07)

| Option | Fit for ~40-node prerequisite DAG | Cost | Verdict |
|---|---|---|---|
| D3 force (d3 7.9.0, 280 KB raw / 92 KB gz) | Force layouts jitter and have no notion of "prerequisite flows left to right"; unstable between loads unless seeded | Hand-write pan/zoom/highlight | Reject for layout. (D3 is still fine as a scale/axis utility elsewhere) |
| Cytoscape.js 3.34.3 (436 KB raw / 136 KB gz) | Built-in pan/zoom/touch, selectors, `predecessors()`, `successors()`, `neighborhood()`, classes, compound nodes (tracks as groups) | One script | **Recommended runtime** |
| Cytoscape + dagre layout | Layered DAG layout, deterministic, edges flow in one direction | cytoscape-dagre 4.0.1 single file, 45 KB (bundles dagre) | **Recommended layout engine (design time or run time)** |
| Cytoscape + fcose | Good force-style, but still organic and prereq direction is lost | layout-base + cose-base + fcose (about 320 KB raw) | Skip |
| Cytoscape + ELK (cytoscape-elk + elkjs 0.12.0) | Best layered quality and port control | elk.bundled.js is 1.6 MB | Overkill at N=40; could be used **offline** only |
| Cytoscape + cola | Constraint force layout | Needs webcola too (not verified) | Dropped |
| vis-network 10.1.2 | Physics default is jittery; hierarchical option exists | 652 KB raw standalone | Reject (heavier, less control) |
| Sigma.js 3.0.3 + graphology 0.26.0 | WebGL, built for 10k+ nodes | 188 KB + 74 KB; labels/touch need extra work | Overkill |
| Hand-laid SVG with fixed coordinates | Perfect stability, zero dependencies, CSS theming, native a11y (each node a real `<a>`) | You write about 150 lines of JS for highlight and path logic | **Strong option; best for N about 40** |

### 3.2 Recommendation: precompute once, ship coordinates

Because prerequisites form a DAG and N is small, the most stable and lightest approach is a hybrid:
1. Run dagre (or ELK) **once offline** in Node on the node/edge list, with tracks as rank bands or compound groups. Hand-nudge a few nodes.
2. Commit the resulting `{id, x, y}` into the data file.
3. Render either (a) Cytoscape with `layout: {name: 'preset'}` (gets touch pan/zoom and graph queries free, about 136 KB gz), or (b) plain SVG (zero deps).
4. Produce **two** coordinate sets: landscape (rank left-to-right) and a narrow portrait set (rank top-to-bottom) selected with `matchMedia('(max-width: 640px)')`.
5. Also render a semantic **list view** (grouped by track, each item a link with "prerequisites: ..." text). A canvas or SVG graph is not keyboard or screen-reader navigable; the list is the accessible fallback and the best phone UX anyway.

Worth a 10-minute test before committing to dagre: Cytoscape's built-in `breadthfirst` layout (no extension) may already produce an acceptable layered result.

### 3.3 Exact CDN URLs (all returned 200)

Cytoscape (cdnjs hosts the core only; extensions are on jsDelivr/unpkg):
```html
<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.34.3/cytoscape.min.js"></script>
<!-- or -->
<script src="https://cdn.jsdelivr.net/npm/cytoscape@3.34.3/dist/cytoscape.min.js"></script>
<!-- dagre layout, self-contained build (45 KB); register with cytoscape.use(cytoscapeDagre) if it does not auto-register -->
<script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@4.0.1/dist/cytoscape-dagre.min.js"></script>
```
Avoid `cytoscape-dagre@2.5.0` on jsDelivr: its `.min.js` is a Terser-on-the-fly build and expects a separate `dagre` global (https://cdn.jsdelivr.net/npm/dagre@0.8.5/dist/dagre.min.js, also on cdnjs as dagre 0.8.5).

Others, if needed: fcose https://unpkg.com/cytoscape-fcose@2.2.0/cytoscape-fcose.js (load layout-base@2.0.1 and cose-base@2.2.0 first, both on unpkg); ELK https://cdn.jsdelivr.net/npm/elkjs@0.12.0/lib/elk.bundled.js ; D3 https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js ; vis-network https://cdnjs.cloudflare.com/ajax/libs/vis-network/10.1.2/standalone/umd/vis-network.min.js ; Sigma https://cdnjs.cloudflare.com/ajax/libs/sigma.js/3.0.3/sigma.min.js.

Docs: https://js.cytoscape.org/ ; ELK https://www.eclipse.org/elk/

### 3.4 Interactions and the APIs behind them

- **Hover to highlight neighbours**: `node.neighborhood().add(node)` gets `.hl`; everything else gets `.dim` (opacity 0.15). With SVG: toggle a class on a precomputed adjacency map.
- **Click to open**: node data carries `href`; on tap, open the page (second tap on touch screens, since first tap shows a preview card with 1-line summary, prerequisites, and an "Open" button; hover does not exist on phones).
- **"Show learning path to X"**: `node.predecessors()` returns all transitive prerequisites; order them with a topological sort for a numbered reading list ("1. Attention -> 2. KV cache -> 3. PagedAttention"). Also highlight `node.successors()` for "what this unlocks".
- **Filter by track**: classes `.arch`, `.train`, `.serve`; toggle chips set `display: none` or dim. Keep cross-track edges visible as dashed lines, since they are the hub's whole point.
- **Progress marking**: `localStorage` set of completed ids, wrapped in try/catch (private mode can throw); completed nodes get a check and a filled style; "next unlocked" nodes get a ring. Per-viewer convenience only.
- **Theming**: style via CSS variables read at init; re-apply on theme change (Cytoscape styles are JS objects, so rebuild them from `getComputedStyle`). SVG gets this free via CSS.
- **Stability**: fixed IDs and preset coordinates mean the map looks identical every visit; animate only highlight state, never layout.

---

## 4. Animation and math tech for vanilla-JS static pages

### 4.1 Options

| Option | Size (gz) | Strength | Weakness |
|---|---|---|---|
| **SVG + CSS + rAF (hand-rolled)** | 0 | Crisp at any DPI, themeable with CSS variables, nodes are real DOM (hover/focus/aria), tiny | Hundreds of animated elements get slow; you write your own timeline |
| **Canvas 2D** | 0 | Thousands of cells per frame (attention heatmaps, KV block grids, token streams) | No DOM semantics, manual hit testing, manual DPR scaling and theme colours |
| **GSAP 3.15.0** | 28 KB | `timeline.play/pause/reverse/seek/progress/timeScale` is exactly our transport bar; excellent SVG handling; matchMedia helper | Another dependency. **License: now 100% free including plugins** per https://gsap.com/pricing/ ("thanks to Webflow's support"); read the Terms of Use for any commercial edge case |
| anime.js 4.5.0 | 41 KB | Small, timelines, `createTimeline`, nice API | Less battle-tested for scrubbing than GSAP |
| D3 transitions | (D3 is 92 KB gz) | Great for data joins and scales | Not a timeline; not worth loading D3 only for this |
| Web Animations API / CSS animations | 0 | Native, GPU-friendly, `animation.currentTime` allows scrubbing | Sequencing many parts gets clumsy |

Recommendation: **default to SVG + rAF driven by a single `t` (0..1 or step index) state**, where every frame is a pure function `draw(t)`. That one design choice gives play/pause/step/scrub/reduced-motion for free (step = set `t`, scrub = bind `t` to an `<input type=range>`, reduced motion = jump to the end state). Use Canvas only for dense per-frame grids. Add **GSAP** as the one optional library on pages with long multi-part choreography. D3 only if already loaded for scales/axes.

CDN URLs (200):
```html
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.15.0/gsap.min.js"></script>   <!-- 73 KB raw, 28 KB gz -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.15.0/ScrollTrigger.min.js"></script> <!-- optional -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/animejs/4.5.0/anime.umd.min.js"></script>   <!-- alternative -->
```

### 4.2 Accessibility and performance rules

- `prefers-reduced-motion`: https://developer.mozilla.org/en-US/docs/Web/CSS/@media/prefers-reduced-motion . In JS, check `matchMedia('(prefers-reduced-motion: reduce)')`, skip autoplay, render the end state, keep manual step/scrub (user-initiated motion is fine).
- WCAG 2.2.2: any motion over 5 s needs a visible pause control.
- Pause animations when offscreen (`IntersectionObserver`) and when the tab is hidden (`visibilitychange`); this is the biggest battery win on phones.
- Cap devicePixelRatio at 2 for Canvas; redraw only on state change, not every frame, when nothing is animating.
- Animate `transform` and `opacity`, not layout properties; for SVG, set attributes rather than re-creating nodes each frame.
- Slider `input` events: throttle to one render per rAF.
- Touch: 44 px minimum targets; `touch-action: none` on draggable scrubbers only.
- Every toy needs a text equivalent: a live `aria-live="polite"` result line and labelled `<input type=range>` (native sliders are keyboard-accessible).
- File size: the 16 MB cap is per-site and our own files will be tiny; CDN scripts do not count against it but do cost phone bandwidth, so budget about 200-250 KB gz of third-party JS per page (KaTeX + one optional lib). Self-hosting is not needed.

### 4.3 Math rendering: KaTeX

KaTeX 0.19.0 (latest on both cdnjs and npm as of today). KaTeX fonts load relative to the CSS, so use the CSS from the same CDN folder.

```html
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.19.0/katex.min.css">
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.19.0/katex.min.js"></script>           <!-- 273 KB raw, 76 KB gz -->
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.19.0/contrib/auto-render.min.js"></script> <!-- 3.5 KB, optional -->
<!-- jsDelivr equivalents: https://cdn.jsdelivr.net/npm/katex@0.19.0/dist/katex.min.js , .../katex.min.css , .../contrib/auto-render.min.js -->
```
Prefer calling `katex.render(str, el, {throwOnError:false, trust:true, strict:false})` yourself for the toy formulas that need `\htmlClass` linking; `auto-render` is fine for static prose. Note `trust:true` is acceptable here because the formula strings are authored by us, never user input. MathJax 3.2.2 (1.17 MB) is available at https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.js but is 4x heavier; skip it. Re-render live numbers into formulas by templating the string (e.g. `d_k = ${dk}`) rather than reparsing the page.

---

## 5. Recommendations ("do this")

1. **Graph**: Cytoscape.js 3.34.3 for the hub, with layout **precomputed offline** (dagre or ELK in Node), shipped as `{id,x,y}` and rendered with `preset`. Hand-laid SVG is an equally valid zero-dependency alternative at N of about 40. No force layouts.
2. Make a **second, portrait coordinate set** for phone widths and a **semantic list view** (grouped by track, with prerequisites as text) as the accessible fallback.
3. Hub interactions: hover neighbourhood highlight, tap-to-preview then open, `predecessors()`-based learning path with topological order, track filter chips, `localStorage` progress (try/catch).
4. **Animation**: SVG (or Canvas for dense grids) driven by one pure `draw(t)` function and a shared transport bar (play, pause, step, scrub, speed, keyboard shortcuts). GSAP 3.15.0 is the single optional library.
5. Respect `prefers-reduced-motion`: no autoplay, jump to end state, keep manual stepping; pause offscreen and on hidden tabs.
6. **Math**: KaTeX 0.19.0 from cdnjs; use `\htmlClass` + hover handlers to link equation terms to diagram parts.
7. Every toy follows one template: **concrete tiny example -> sliders with live numeric readouts -> "scale up to a real model" preset -> collapsed "show me the math" `<details>`**.
8. Use **named presets** with real configs (Llama-3 8B/70B, DeepSeek-V3, Qwen) and show derived quantities (GB, tokens/s, speedup) next to the raw inputs.
9. **Link out, do not duplicate**: Transformer Explainer and Bycroft (full forward pass), Raschka gallery (model survey), UNIPO (RL algorithm comparison), Ultra-Scale Playbook and Scaling Book (parallelism and roofline), llm-inference-explained (serving long form), 3Blue1Brown (intuition video). Give a "Go deeper" box on each page with 2-3 curated links.
10. **Build the gaps**: RoPE rotation toy, KV-bytes calculator (MHA/GQA/MLA), MoE router toy, patch-to-token toy, GRPO advantage toy, per-GPU memory and pipeline-bubble toys, slot-timeline batching toy, paged block-table toy, speculative accept-rate toy.
11. Implement the **scrubbable-number (Tangle) pattern** in about 30 lines of vanilla JS with keyboard arrow support; use native `<input type=range>` for the main sliders.
12. **Share state in the URL hash** so a reader can send a configured toy to someone else; add a Reset button to every toy.
13. Keep a single state object and one `render()` per toy so linked views (diagram, chart, readout, formula) never drift; throttle input to rAF.
14. **Theming**: CSS custom properties for both light and dark; SVG/Canvas read colours from computed styles, and Cytoscape styles are rebuilt on theme change. Test at 360 px width.
15. Show hand-verifiable arithmetic: include a "check my work" line (the expression with the numbers substituted in) under every calculator, and unit-test the pure functions in Node so on-page numbers match published figures (for example, a Llama-3 8B KV cache of 128 KiB per token in fp16, derived from 32 layers x 8 KV heads x 128 dims x 2 (K and V) x 2 bytes).
