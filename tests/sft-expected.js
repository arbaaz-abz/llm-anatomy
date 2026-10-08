// Long expected strings for the sft page, exported once (the page test and the e2e spec import them; page code never does).
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

export const CHECK_WORK = [
  'trained = assistant 10 + error 5 = 15',
  'masked  = template 2 + user 6 + observation 3 = 11',
  'share   = 15 ÷ 26 = 57.7%',
].join('\n');

export const CHECK_WORK_NO_PROMPT_MASK = [
  'trained = template 2 + user 6 + assistant 10 + error 5 = 23',
  'masked  = observation 3 = 3',
  'share   = 23 ÷ 26 = 88.5%',
].join('\n');

export const CHECK_WORK_NO_OBSERVATION_MASK = [
  'trained = assistant 10 + error 5 + observation 3 = 18',
  'masked  = template 2 + user 6 = 8',
  'share   = 18 ÷ 26 = 69.2%',
].join('\n');

export const CHECK_WORK_MASK_ERROR = [
  'trained = assistant 10 = 10',
  'masked  = template 2 + user 6 + error 5 + observation 3 = 16',
  'share   = 10 ÷ 26 = 38.5%',
].join('\n');

export const CHECK_WORK_NOTHING_MASKED = [
  'trained = template 2 + user 6 + assistant 10 + error 5 + observation 3 = 26',
  'masked  = none = 0',
  'share   = 26 ÷ 26 = 100.0%',
].join('\n');

// The three try-this items, in order: { prompt, insight } as the toy prints them.
export const TRY_THIS = Object.freeze([
  {
    prompt: 'Turn off "Mask the prompt and template tags": trained 15 → 23 of 26 (88.5%), and the 6 user tokens plus 2 tags are now targets.',
    insight: 'without the mask, the model is also trained to write the user\'s side, so it spends capacity imitating questions instead of answering them.',
  },
  {
    prompt: 'Turn off "Mask the tool\'s reply" (prompt mask back on): trained 15 → 18 of 26 (69.2%), with <obs> 56 </obs> now a target.',
    insight: 'training on tool outputs teaches the model to predict results it should have waited for. That is how a model learns to invent a tool\'s answer; GLM-5 excludes tool outputs from the loss, and Agentic RL does the same.',
  },
  {
    prompt: 'Turn on "Mask the mistake in the trace (GLM-5)": trained 15 → 10 of 26 (38.5%); the five mistake chips stay visible but hatched.',
    insight: 'masking is how you show a mistake without teaching it. The model still reads 7 × 8 = 54 as context and is trained to write wait , check and the tool call after it.',
  },
]);

// What the toy prints for the default state.
export const DEFAULT_READOUTS = Object.freeze({
  trained: '15 of 26',
  masked: '11',
  share: '57.7%',
  selection: 'Selected "56", an assistant token: trained.',
});
