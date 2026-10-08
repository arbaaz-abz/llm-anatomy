// The page's hand-authored stand-ins (sft §5): the 26-token transcript, the template tag sets and the pipeline strip.
import { deepFreeze } from '@math/core.js';

// kind → tokens, whitespace-split. 26 tokens in eight segments (storyboard §5).
export const SEGMENTS = deepFreeze([
  { kind: 'template', tokens: ['<user>'] },
  { kind: 'user', tokens: ['What', 'is', '7', '×', '8', '?'] },
  { kind: 'template', tokens: ['<assistant>'] },
  { kind: 'assistant', tokens: ['<think>'] },
  { kind: 'error', tokens: ['7', '×', '8', '=', '54'] },
  { kind: 'assistant', tokens: ['wait', ',', 'check', '</think>', '<call>', 'calc(7*8)', '</call>'] },
  { kind: 'observation', tokens: ['<obs>', '56', '</obs>'] },
  { kind: 'assistant', tokens: ['56', '<end>'] },
]);

export const TOTAL_TOKENS = 26;
export const FOLLOWED = 24; // the final "56" (flat index): the token the page follows
export const FIRST_ASSISTANT_TOKEN = 8; // flat index of "<think>"
export const CALL_TOKENS = Object.freeze([18, 19, 20]); // <call> calc(7*8) </call>
export const OBS_TOKENS = Object.freeze([21, 22, 23]); // <obs> 56 </obs>
export const ERROR_TOKENS = Object.freeze([9, 10, 11, 12, 13]); // 7 × 8 = 54
export const THINK_TAGS = Object.freeze([8, 17]); // <think> and </think>

// Transcript lines: a role label (or none) and the segments each line holds.
export const LINES = deepFreeze([
  { role: 'user', segments: [0, 1] },
  { role: 'assistant', segments: [2, 3] },
  { role: null, segments: [4] },
  { role: null, segments: [5] },
  { role: 'tool', segments: [6] },
  { role: 'assistant', segments: [7] },
]);

// Frame 1: the base model's illustrative continuation, and the prompt it continues.
export const PROMPT = Object.freeze(['What', 'is', '7', '×', '8', '?']);
export const CONTINUATION = Object.freeze(['What', 'is', '9', '×', '6', '?', 'What', 'is', '4', '×', '7', '?']);

// Frame 8: four candidate traces; the checker keeps the first and third (illustrative).
export const CANDIDATES = Object.freeze([{ label: 'trace 1', ok: true }, { label: 'trace 2', ok: false }, { label: 'trace 3', ok: true }, { label: 'trace 4', ok: false }]);

// Frame 9: the six pipeline stages (training-pipeline); the RL stage lights.
export const STAGES = Object.freeze(['1 pretrain', '2 mid-train', '3 SFT', '4 specialist RL', '5 merge', '6 polish']);
export const RL_STAGE = 3;

// The toy's template chips are label-only: they change the tag text, never the counts.
// Only forms the data documents are shown: DeepSeek-V4's |DSML| tool calls and gpt-oss's role names.
export const TEMPLATES = deepFreeze({
  generic: {},
  deepseek: { '<call>': '|DSML|', '</call>': '|/DSML|' },
  harmony: { '<user>': 'User', '<assistant>': 'Assistant', '<obs>': 'Tool' },
});
export const TEMPLATE_OPTIONS = Object.freeze([
  { value: 'generic', label: 'generic' },
  { value: 'deepseek', label: 'DeepSeek-V4 <think>' },
  { value: 'harmony', label: 'gpt-oss harmony roles' },
]);
