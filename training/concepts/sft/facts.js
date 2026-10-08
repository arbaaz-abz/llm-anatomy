// sft's dated text (pure, no DOM): the §8 rows, the framing and the notes printed under the stage.
// Dated numbers are {entry.key|format} placeholders filled from data/*.json; years only from confirmed release dates (X-1).

export const FRAMING = 'The chat templates differ per lab, and so do the SFT recipes. What the open reports share is the mask: the assistant\'s tokens are the targets, and the rest is context.';

// The ten rows of storyboard §8, in order. `|cite` prints nothing but attaches the fact's source link and its "reported" flag.
export const FACT_ROWS = Object.freeze([
  { claim: 'DeepSeek-V4 chat template: {deepseek-v4-pro.chat_template}. The team found that XML-style tool calls reduce escaping errors.' },
  { claim: 'gpt-oss chat template: {gpt-oss-120b.chat_template}.' },
  { claim: 'Kimi K3\'s chat template is called {kimi-k3.chat_template}; GLM-5\'s updated template adds {glm-5.chat_template}.' },
  { claim: 'GLM-5 keeps erroneous segments in its agent trajectories but masks them out of the loss, and excludes environment and tool outputs from it.{glm-5.sft_masking|cite}' },
  { claim: 'Kimi K3: the SFT stage establishes a high-quality cold-start policy for the subsequent RL stage, from trajectories by earlier Kimi specialists plus multi-stage verification and human annotation.{kimi-k3.sft_data|cite}' },
  { claim: 'GLM-5 builds SFT data by rejection sampling, filtering to problems its previous model finds hard.{glm-5.sft_data|cite}' },
  { claim: 'DeepSeek-R1 ({deepseek-r1.release_date|year}): about {deepseek-r1.sft_samples|count} SFT samples, reasoning and non-reasoning together.' },
  { claim: 'Olmo 3: about {olmo-3.sft_traces|count} reasoning traces distilled from QwQ-32B and DeepSeek-R1.' },
  { claim: 'Thinking modes: DeepSeek-V4 {deepseek-v4-pro.thinking_modes}; Kimi K3 {kimi-k3.thinking_modes}; gpt-oss {gpt-oss-120b.thinking_modes}.' },
  { claim: 'DeepSeek-V3.2 uses a {deepseek-v3.2.sft_cold_start}. Qwen3\'s recipe, as reported: {qwen3.post_training}.' },
]);

// Page text under the stage, one list per frame (index 0 = frame 1).
export const BELOW = Object.freeze([
  [],
  [],
  ['Products switch thinking modes by template and training, not by a different network: DeepSeek-V4 offers {deepseek-v4-pro.thinking_modes}.'],
  [],
  [],
  [],
  ['GLM-5 ({glm-5.release_date|year}) keeps a mistake in the text and masks it: {glm-5.sft_masking}.'],
  ['DeepSeek-R1 ({deepseek-r1.release_date|year}) used about {deepseek-r1.sft_samples|count} SFT examples. Olmo 3 used about {olmo-3.sft_traces|count} reasoning traces (reported).'],
  ['This is the zero-correct case in [[rlvr-grpo]]: with no right answers in a group there is no signal.'],
]);
