// Storyboard §6 "Try this": four prompts, each leading to a named insight. Every number is computed from the toy's own
// view model (requestTimeline on RUNNING_EXAMPLE), printed through formatDuration / formatShare. Pure, no DOM.
import { formatDuration, formatInt, formatRatio } from '@math/core.js';
import { formatShare } from '@shared/glyphs.js';
import { arrowText, INITIAL_STATE } from './format.js';
import { speeds, timelineFor } from './toy-view.js';

function first(at) {
  const base = at({});
  const longPrompt = at({ prompt: 20_000 });
  const shortAnswer = at({ output: 50 });
  const tenfold = formatRatio(20_000 / INITIAL_STATE.prompt);
  return {
    prompt: `Predict first: which costs more time, a ${tenfold} longer prompt or a ${tenfold} longer answer? `
      + `Set Prompt length from ${formatInt(INITIAL_STATE.prompt)} to 20,000: TTFT ${arrowText(base.ttftS, longPrompt.ttftS)}, total ${arrowText(base.e2eS, longPrompt.e2eS)}. `
      + `Set Prompt length back to ${formatInt(INITIAL_STATE.prompt)}, then Answer length from ${formatInt(INITIAL_STATE.output)} to 50: total ${arrowText(base.e2eS, shortAnswer.e2eS)}.`,
    insight: 'the prompt sets the wait for the first token; the answer\'s length sets the total.',
    rest: ` Decode is ${formatShare(base.decodeShare)} of the default request's time.`,
  };
}

function second(at, chips) {
  const [alone, shared] = chips;
  const base = at({});
  const sharing = at({ decodeRate: 'shared' });
  const [aloneRate, aloneName] = alone.label.split(', ');
  const [sharedRate, sharedName] = shared.label.split(', ');
  return {
    prompt: `Switch Decode speed per user from "${aloneName}" (${aloneRate}) to "${sharedName}" (${sharedRate}): `
      + `TTFT does not move (${formatDuration(sharing.ttftS)}), total goes ${arrowText(base.e2eS, sharing.e2eS)}.`,
    insight: 'TTFT and TPOT are separate dials.',
    rest: ' Sharing the GPU with more users slows each user\'s stream, not the first token, which is why servers track both, and why [[batching]] and [[disaggregation]] treat them as two targets.',
  };
}

function third(at, chips) {
  const sharing = at({ decodeRate: 'shared' });
  const queued = at({ decodeRate: 'shared', queue: 2 });
  return {
    prompt: `On "${chips[1].label.split(', ')[1]}", set Time in queue to ${formatDuration(queued.queueS)}: `
      + `TTFT ${arrowText(sharing.ttftS, queued.ttftS)}, total ${arrowText(sharing.e2eS, queued.e2eS)}, while TPOT stays ${formatDuration(queued.tpotS)}.`,
    insight: 'a queue only delays the start.',
    rest: ' Once your request is in the batch, its pace depends on the step time, not on how long it waited.',
  };
}

function fourth(at) {
  const tiny = at({ prompt: 3 });
  return {
    prompt: `Slide Prompt length down to 3 tokens: TTFT is still ${formatDuration(tiny.ttftS)}, not near zero.`,
    insight: 'even a tiny prompt costs one full read of the weights,',
    rest: ' the same floor a decode step pays (see [[prefill-decode]]).',
  };
}

// [{ prompt, insight, rest }]; the page prints "prompt → Insight: insight rest", with [[slug]] as the lesson's title.
export function tryThis(data) {
  const { rates, options } = speeds(data);
  const at = (patch) => timelineFor({ ...INITIAL_STATE, ...patch }, rates);
  return [first(at), second(at, options), third(at, options), fourth(at)];
}
