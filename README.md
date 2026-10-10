# llm-anatomy

How a 2026 LLM is built, trained and served, as 34 small interactive lessons. Each one is an animation you step through plus a toy whose numbers you can check by hand. No framework, no build step, just ES modules and SVG.

- **Architecture** (11): attention, RoPE, the KV cache, MLA, MoE, long context, sampling, reading a model card.
- **Training** (14): pretraining through RL and distillation, then GPUs, memory, parallelism and running 10,000 of them.
- **Serving** (9): prefill vs decode, batching, paging, prefix caching, speculative decoding, quantization, and finally sizing a 1.6T model and what a million tokens costs.

## quick start

```sh
npm install
npm run serve
```

Open http://127.0.0.1:8080/.

Tests: `npm test` for the math and text, `npm run e2e` for the browser.

## notes

Every real-world number (model sizes, GPU specs, prices) lives in `data/` with its source and the date it was read. Each lesson's full design is in `docs/storyboards/`.

## license

MIT for code, CC BY 4.0 for the lessons.
