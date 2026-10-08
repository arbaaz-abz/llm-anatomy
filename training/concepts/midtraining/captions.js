// The nine captions of storyboard §5, verbatim (one idea each, at most two sentences and 30 words).
export const CAPTIONS = Object.freeze([
  "The learning rate sets how big each weight update is. Runs start it near zero and raise it over a short warmup, so early updates do not wreck the weights.",
  "A cosine schedule lowers the rate smoothly over the whole run. The finish line has to be chosen on day one, because the curve depends on it.",
  "Warmup-stable-decay keeps the rate flat and drops it only at the end. Any checkpoint on the plateau can be branched into its own short decay.",
  "As the steps shrink, the model settles into what it reads last. So labs put their best data in the decay: this is annealing.",
  "In 2026 this end phase is a named stage, mid-training. It is still next-token prediction on documents, with a richer mix and longer sequences.",
  "Context extension grows the sequence length in a few short stages. GLM-5 goes from 4K to 200K tokens using the last 5% of its run.",
  "Late and short, because each long-context token is costly: at 200K a new token attends to 50 times as many earlier tokens as at 4K. Long documents are scarce too.",
  "A model using RoPE has never seen rotations this large, so its rotations are rescaled to fit. Kimi K3's full-attention layers have no positional encoding and skip this step.",
  "All four 2026 reports here stage their context this way, ending near 200K or 1M. Kimi K3 says why: the extension is cheap because it is short.",
]);
