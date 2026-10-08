// "Which tokens teach?" view model (storyboard §6): a pure function from the toy state to every string the toy prints.
// Unit-tested; toy.js only paints it. Every count comes from math/sft.js, every share from formatShare.
import { lossMask } from '@math/sft.js';
import { fillText } from '@shared/claims.js';
import { layoutTranscript } from './stage.js';
import { SEGMENTS, TEMPLATES } from './numbers.js';
import { INITIAL_STATE, checkWork, kindRows, optionsOf, selectionText, shareText, summaryOf, trainedText } from './format.js';

const TEMPLATE_NOTES = Object.freeze({
  generic: 'Tags are generic stand-ins; each lab\'s template differs.',
  deepseek: 'DeepSeek-V4: {deepseek-v4-pro.chat_template}. Only the tool-call tags change here; the counts do not.',
  harmony: 'gpt-oss: {gpt-oss-120b.chat_template}. Only the role names change here; the counts do not.',
});

export const templateNote = (state, data) => fillText(TEMPLATE_NOTES[state.template], data);

// The 26 chips as the toy draws them: layout for the chosen template, and whether each is a loss target.
export function chipsFor(state) {
  const mask = lossMask(SEGMENTS, optionsOf(state));
  return layoutTranscript(TEMPLATES[state.template]).map((chip) => ({ ...chip, trained: mask[chip.flat] }));
}

export function view(state, data) {
  const summary = summaryOf(state);
  const chips = chipsFor(state);
  return {
    chips,
    trained: trainedText(summary),
    masked: String(summary.masked),
    share: shareText(summary.trained, summary.total),
    kinds: kindRows(state),
    selection: selectionText(chips[state.selected], chips[state.selected].trained),
    checkWork: checkWork(state),
    templateNote: templateNote(state, data),
  };
}

// The three "try this" items ({ prompt, insight }), with every number computed from the same functions the toy uses.
export function tryThis() {
  const at = (patch) => {
    const s = summaryOf({ ...INITIAL_STATE, ...patch });
    return { trained: s.trained, total: s.total, share: shareText(s.trained, s.total), byKind: s.byKind };
  };
  const base = at({});
  const noPrompt = at({ maskPrompt: false });
  const noObs = at({ maskObservation: false });
  const maskErr = at({ maskError: true });
  return [
    {
      prompt: `Turn off "Mask the prompt and template tags": trained ${base.trained} → ${noPrompt.trained} of ${noPrompt.total} (${noPrompt.share}), and the ${base.byKind.user.masked} user tokens plus ${base.byKind.template.masked} tags are now targets.`,
      insight: 'without the mask, the model is also trained to write the user\'s side, so it spends capacity imitating questions instead of answering them.',
    },
    {
      prompt: `Turn off "Mask the tool's reply" (prompt mask back on): trained ${base.trained} → ${noObs.trained} of ${noObs.total} (${noObs.share}), with <obs> 56 </obs> now a target.`,
      insight: 'training on tool outputs teaches the model to predict results it should have waited for. That is how a model learns to invent a tool\'s answer; GLM-5 excludes tool outputs from the loss, and Agentic RL does the same.',
    },
    {
      prompt: `Turn on "Mask the mistake in the trace (GLM-5)": trained ${base.trained} → ${maskErr.trained} of ${maskErr.total} (${maskErr.share}); the five mistake chips stay visible but hatched.`,
      insight: 'masking is how you show a mistake without teaching it. The model still reads 7 × 8 = 54 as context and is trained to write wait , check and the tool call after it.',
    },
  ];
}
