// distillation's dated text (pure, no DOM): the §8 rows and framing, and the notes printed under the stage for frames 7–9.
// Dated numbers are {entry.key|format} placeholders filled from data/*.json; rows keep their placeholders so the scaffold
// adds each row's source link and "reported" chip.
export const FACT_ROWS = Object.freeze([
  'DeepSeek-V4 distills with {deepseek-v4-pro.opd}; the distillation replaces only the final mixed-RL stage, and its specialists are still trained with SFT and {deepseek-v4-pro.rl_algorithm}.',
  'Kimi K3 merges its experts with multi-teacher on-policy distillation: {kimi-k3.opd} ({kimi-k3.rl_experts|cite}).',
  'MiMo-V2-Flash adds an outcome advantage to the teacher signal, and reports that it keeps each teacher\'s peak instead of the usual trade-off: {mimo-v2-flash.opd}.',
  'GLM-5 ends with on-policy distillation across its own stages, to undo catastrophic forgetting: {glm-5.opd}.',
  'On-policy distillation was popularized by Thinking Machines and Qwen3; the claimed compute saving versus RL is {paper:on-policy-distillation-2025.compute_savings_vs_rl}.',
  'DeepSeek-R1 ({deepseek-r1.release_date|year}) was fine-tuned on about {deepseek-r1.sft_samples|count} SFT samples; reusing them to distill small Qwen and Llama models is background, not in the table.',
  'Olmo 3\'s SFT used about {olmo-3.sft_traces|count} reasoning traces distilled from QwQ-32B and DeepSeek-R1.',
]);

export const factRows = () => FACT_ROWS.map((claim) => ({ claim }));

export const FRAMING = 'Every lab below ends its recipe the same way: train specialists with RL, then merge them into one student by distillation. The rows say who uses which teacher signal, and what each lab reports it keeps.';

// Notes under the stage, by frame index (0-based); each is a list of paragraphs with placeholders.
export const BELOW = Object.freeze({
  6: ['DeepSeek-V4: {deepseek-v4-pro.opd}. Kimi K3: {kimi-k3.opd}.'],
  7: ['MiMo-V2-Flash: {mimo-v2-flash.opd}. GLM-5: {glm-5.opd}.'],
  8: ['DeepSeek-R1 ({deepseek-r1.release_date|year}): about {deepseek-r1.sft_samples|count} SFT samples, also used to distill small Qwen and Llama models (background).'],
});
