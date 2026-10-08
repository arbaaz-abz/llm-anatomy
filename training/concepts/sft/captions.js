// The nine captions of storyboard §5, verbatim (one idea each, at most two sentences and 30 words).
export const CAPTIONS = Object.freeze([
  'A pretrained base model only continues text. Asked a question, it may write more questions, because worksheets full of questions are common text.',
  'A chat template wraps each turn in special tokens that mark who is speaking. The model can now tell a question it was asked from an answer it should write.',
  'Reasoning models put their working between think tags before the answer. The tags are ordinary tokens; the model uses them because every example does.',
  'A loss mask decides which tokens are targets. The prompt and the template tags are context only, never targets.',
  'The model\'s tool call is trained. The tool\'s reply is masked, so the model learns to call tools without learning to invent their results.',
  'The loss is the same next-token cross-entropy as pretraining, counted only on the assistant\'s tokens.',
  'GLM-5 keeps mistakes in its agent traces but masks them. The model reads the error and the recovery, and is trained only on the recovery.',
  'Most 2026 SFT data in the open reports comes from other models. Rejection sampling keeps only the traces that pass a check, often on problems the previous model found hard.',
  'SFT gives RL its cold start: a model that answers in format and sometimes reasons its way to the right answer. RL can only reinforce what it already does sometimes.',
]);
