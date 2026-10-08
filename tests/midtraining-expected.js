// midtraining storyboard §5 captions and §6 "Check my work" / try-this text, exact. Imported by the page test and the e2e spec.
export const CAPTIONS = [
  "The learning rate sets how big each weight update is. Runs start it near zero and raise it over a short warmup, so early updates do not wreck the weights.",
  "A cosine schedule lowers the rate smoothly over the whole run. The finish line has to be chosen on day one, because the curve depends on it.",
  "Warmup-stable-decay keeps the rate flat and drops it only at the end. Any checkpoint on the plateau can be branched into its own short decay.",
  "As the steps shrink, the model settles into what it reads last. So labs put their best data in the decay: this is annealing.",
  "In 2026 this end phase is a named stage, mid-training. It is still next-token prediction on documents, with a richer mix and longer sequences.",
  "Context extension grows the sequence length in a few short stages. GLM-5 goes from 4K to 200K tokens using the last 5% of its run.",
  "Late and short, because each long-context token is costly: at 200K a new token attends to 50 times as many earlier tokens as at 4K. Long documents are scarce too.",
  "A model using RoPE has never seen rotations this large, so its rotations are rescaled to fit. Kimi K3's full-attention layers have no positional encoding and skip this step.",
  "All four 2026 reports here stage their context this way, ending near 200K or 1M. Kimi K3 says why: the extension is cheap because it is short.",
];

// The default state: WSD, decay 20%, stop at 60%.
export const CHECK_WORK = [
  "WSD: the decay starts at 100% − 20% = 80% of the run",
  "stop at 60%: before 80%, on the plateau → 1.000 of peak",
  "cosine at 60%: ½ × (1 + cos(π × 0.60)) = ½ × (1 − 0.309) = 0.345 of peak",
].join('\n');

// Inside the decay (stop at 90%): the WSD line reads the storyboard's example.
export const CHECK_WORK_IN_DECAY_LINE = "stop at 90%: inside the decay → 1 − (90% − 80%) ÷ 20% = 0.500 of peak";
