# Storyboards

Every lesson is designed here before any code is written. Fill `_template.md`.

Quality bar (a storyboard is rejected if any item fails):
- Concrete numbers a learner could check by hand (toy dims ≤ 8, seeded values).
- Every animation frame has a final-wording caption and one idea.
- Every dated fact maps to a `data/*.json` entry and a brief section; nothing UNVERIFIED.
- Each "try this" prompt leads to a named insight.
- Uses only glyphs from spec §5.1, or proposes a new one explicitly.
- At least one misconception is named and corrected.
- Prose is plain, short sentences, no filler, at most one analogy.

Flow: draft → reviewed (main session) → approved (independent expert review by a different model than the author, with a learner-advocate lens) → built (Plans 2–3).

Visual constraints from the built glyph library (gallery review, 2026-10-07):
- Numbers print inside cells only at ≥ 36 px; hero rows use `NUMBER_CELL` (40 px, 3 px gaps). Smaller grids (≈ 20 px) show values on hover only, so don't put a number the learner must read in a small grid.
- Numbers use a real minus sign and a leading zero ("−0.87").
- Byte units are decimal SI everywhere (kB, MB, GB, TB; `formatBytes` default), matching vendor HBM and bandwidth specs. Per-token sizes also print exact bytes with thousands separators ("70,272 B ≈ 70.3 kB"). Never mix binary (KiB/MiB/GiB) units on a page. Printed sizes use `formatBytes`' output: three significant figures ("328 kB", "70.3 kB", "4.72 MB", "42.9 GB"). (Ruling 2026-10-07.)
- At 1×, a step's transition takes 750 ms and auto-play dwells 1.8 s; captions are one idea, at most two sentences.
- Stages are fixed-size SVGs (≈ 580 × 366 in the gallery demo) that scroll inside their container at 400 px; plan layouts for that box.
- Outlines and accent frames mean **selection or focus only** ("the token we're following"), never a quantity. Encode amounts with fill, opacity, length or a printed number. (User feedback 2026-10-07: a weight drawn as outline thickness read as "selected".)
- Always mark the query or the item being followed the same way in every frame.
- Encode each quantity once. If the weights row already shows a weight by color, don't re-mark the rows it applies to; link them with a chip in the same color. (User feedback 2026-10-07.)
- Ratios "N× smaller / less / fewer" print 3 significant figures, trailing zeros dropped (12×, 56.9×, 1,180×); a ratio below 1 is never written as "N× smaller".

Lessons from the pilot expert review (2026-10-07). These apply to every storyboard:
1. **Header must match the graph.** `Prereqs:` equals the slug's `prereqs` in `shared/concepts.json`, and every `Next:` slug lists this one as a prereq.
2. **Captions:** at most two sentences and 30 words, one idea, no formulas or operators (names such as W_O are fine). Formulas and address rules go in "Numbers shown" or §7. A third sentence or a second idea means split the frame.
3. **One new term or symbol per frame.** Define it on screen the first time it appears, or link to the prereq page that defines it. §4 lists "Terms introduced: … / Terms assumed from prereqs: …".
4. **Promise only what ships.** Objective, misconceptions and takeaways claim only what a frame or try-this in the first build demonstrates. Each misconception names the frame or try-this that corrects it. Deferred topics are named as deferred ("you'll meet X in `slug`").
5. **Nothing load-bearing behind hover.** Hover and tap add detail only; caveats, footnotes and key facts are visible without pointer or focus.
6. **Every number is regenerated, sketches included.** Every number in §5, §6 and §11 comes from a `math/` function (or its reproducer), with the `node -e` call recorded, and the §11 sketch is checked against the same output.
7. **No unsupported absolutes.** "Every", "never", "always", "the same" need a brief line behind them; otherwise say "most" or "usually". Check against 2026 exceptions (linear attention, CISPO/GSPO, …).
8. **Dated numbers carry their date and conditions.** Classic-paper numbers say their year ("in the 2023 paper"). Throughput, concurrency and latency name model, hardware and context. A data id names the exact checkpoint (Llama-3 vs 3.1). Sources beyond the briefs record who re-verified them, when, and what was read.
9. **Answer the hook fully** — its *why*, not just its *how*.
10. **Say when numbers are stand-ins** (hand-picked Q, K, V; ratios) in a visible line.
11. **Indexing:** on-screen token positions are 1-based; memory addresses, block numbers and array slots are 0-based. A page showing both says so.
12. **Explain surprising quantities** (π_θ ≠ π_old; a toy topping out below the headline number) where they first appear.
13. **Order and branches.** Finish one idea thread before starting the next; label a what-if that leaves the main timeline as a branch on screen.
14. **Name the cost.** Each technique page states what it costs or when it breaks.
15. **Glyph discipline covers small marks too.** Bars, badges and stamps are a library glyph, a plain labeled text mark, or an explicit glyph proposal.
16. **One definition per derived metric.** Each derived metric (active params, bytes per token, waste %) has a single definition, applied to every preset and every prose claim, and tested.
17. **Check insights against the toy's own numbers.** A simplification in the toy must not reverse the real-world lesson it is meant to teach.
18. **Stage budget.** Numbered cells × 43 px must fit the ≈ 580 × 366 stage; otherwise collapse the row (named cells + "N others"), never shrink it to hover-only cells.
19. **Share bars need a printable segment.** Any segment the learner must read or follow is at least ~18 px wide; otherwise add a zoomed second bar for the tail (with a bracket) or collapse small parts into "others". Unknown parts use the neutral fill at a fixed width outside the scale, never hatch (hatch means masked / excluded).
20. **Settled rulings are recorded as settled.** When the main session rules on an open question, the storyboard states the ruling as applied and deletes the alternative.
21. **Compare like with like.** When two rows on one stage show quantities in different units or scales, a visible line says what the comparison is about (shape, not size; a multiplicative band).
22. **A toy's range stays inside its model's validity,** or the page says where it leaves it. Check every slider stop against the floor model the track teaches.
23. **Measured anchors carry their counting convention.** Tokens/s or $/M say output-only vs input + output; link speeds say each-way vs both directions; chip memory says nominal vs usable. Compare a floor and a measurement only in the same unit.
24. **One meaning per mark, course-wide.** Hatch means "excluded / doesn't count" everywhere (masked, reserved-empty, idle, gradient off, evicted). A state that may still count — a draft token — uses the library's `token` `draft` state (dashed border), defined once.
25. **One basis per compared ratio.** When a page compares a derived ratio across mechanisms, every mechanism uses the same unit of work (a full step, or a forward pass), named once on the page.
26. **Say which pass.** Statements about what travels, is saved or is sharded name the pass they hold for (forward, backward, or full step).
27. **A preset derived from a record reproduces it.** If a preset is back-solved from a published number, the toy prints that number back, or the page explains the gap where it first appears.
28. **HBM totals never sit on a per-direction bandwidth ladder.**
29. **Check captions against the page's own §8 table.** An absolute or a "both" in a caption or framing paragraph must hold for every model the same page lists.
30. **§7 worked lines use the printed equation's conventions** — same symbols, signs and order. Two compensating errors that give the right number are still a defect.
31. **§8 is a source, not a correction.** A misconception is corrected on the stage or in a try-this, never only by a §8 row.
32. **A `flow` dot must contrast with what it enters.** A dot entering an `active` (accent-filled) block gets a halo or stops at the block's edge.
33. **Lesson names in text.** On the stage, name a lesson by its title; in page text, use `[[slug]]` only where a title reads naturally ("… in [[x]]", "see [[x]]"), never as a possessive or a bare parenthetical slug.
34. **Try-this at the end of the toy.** Every page prints its storyboard's try-this list at the end of the toy section (`h4` "Try this" + `ol.try-this`, each item "prompt → Insight: …"), numbers computed, text pinned in the page test.
35. **One formatter per quantity.** Bytes print through `formatBytes`, counts through `formatCount`, shares through `sharePct` / `formatShare`, ratios through `formatRatio`, durations through `formatDuration`; two pages never print the same quantity in two formats. Exact integers (users, tokens, tok/s per GPU) print through `formatInt`; per-user rates through `formatCount`.
