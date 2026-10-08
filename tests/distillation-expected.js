// distillation storyboard §5 captions, §6 "Check my work" and try-this text, verbatim. Imported by the page test and the e2e spec.
export const CAPTIONS = [
  "A teacher, here the math specialist RL produced, and a student both give probabilities for the next token. The student is less sure.",
  "The oldest way: fine-tune the student on text the teacher wrote, exactly like SFT. Only the one token the teacher picked counts.",
  "Logit distillation uses the teacher's whole row as the target, so the student also learns how unlikely 54 and 48 are.",
  "Branch: both methods only trained on text the teacher wrote. When the student samples its own mistake, it is somewhere it never practised.",
  "On-policy distillation lets the student write and the teacher grade every token it wrote. 54 earns minus 1.79: the teacher finds it six times less likely.",
  "GRPO gave every token of this wrong answer the same minus 0.58. The teacher's grades are dense: the blame lands on 54, not on the correct steps before it.",
  "To merge RL specialists, each prompt the student answers is graded by the teacher for its domain. One student learns every specialist's skill.",
  "Training one skill after another tends to erase earlier ones: catastrophic forgetting. Distilling from all specialists at once keeps them, which is why 2026 pipelines end this way.",
  "Small models are made the same way: pretrain, fine-tune on a big teacher's traces, then distill on-policy. The cost: the signal runs out once the student matches the teacher.",
];
export const STAND_IN = 'The teacher and student probabilities are hand-picked stand-ins; the losses and rewards computed from them are exact. Only four candidate tokens are shown, so every KL here sums over those four.';
export const CHECK_WORK = [
  "reward(54) = ln 0.05 − ln 0.30 = −2.996 − (−1.204) = −1.792",
  "expected reward = −KL(student ‖ teacher)",
  "  = −(0.40 ln(0.40/0.90) + 0.30 ln(0.30/0.05) + 0.20 ln(0.20/0.03) + 0.10 ln(0.10/0.02))",
  "  = −(−0.324 + 0.538 + 0.379 + 0.161) = −0.754",
].join('\n');
// Each item is the whole list entry as printed: prompt, " → ", "Insight: …", then the rest.
export const TRY_THIS = [
  "With on-policy, unsure and the math teacher, select each sampled token: 56 +0.811, 54 −1.792, 48 −1.897, 63 −1.609; the expected reward is −0.754. → Insight: the teacher grades the student's own choices, wrong ones included. Every token the student might write gets its own signed grade, which is what fixes the blind spot of frame 4.",
  "Switch student to close to the math teacher: every reward shrinks toward 0 (56 +0.057, 54 −0.336) and the expected reward is −0.013. Then switch to confident and wrong: 54 falls to −2.639 and the expected reward to −1.909. → Insight: the signal is largest where the student is most wrong and vanishes as it matches the teacher. The distillation reward stops pushing once the student matches its teacher; going past it needs another signal, such as MiMo-V2-Flash's outcome advantage (see the math panel).",
  "Keep close to the math teacher and switch teacher: math 0.013, chat 0.152, 50/50 mix 0.082. → Insight: averaging teachers pulls the student toward a blend. A student that already matches the math specialist is still pushed away from it by the mix, which is why Kimi K3 picks one teacher per prompt by domain and effort level; DeepSeek-V4 sums its teachers' KLs with weights.",
];
