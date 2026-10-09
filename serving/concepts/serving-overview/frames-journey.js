// Frames 1–4: your request travels from Enter to the GPU and is prefilled (storyboard §5). Each frame is a pure function
// of its progress p (0 → 1); its end state is the next frame's start, and a stop changes state only after p = 0.1.
import { PROMPT } from './numbers.js';
import {
  IDLE, CHOSEN, PROMPT_AT, QUEUE_LANE, CHIP_Y, GPU, seg, lerp, ease, leaving,
  pipeline, hop, promptChips, cacheTiles, yourBar, barSteps, outOfGpu, note, label, region,
} from './stage.js';

const TOY_NOTE = 'toy prompt: 3 tokens; real prompts are often thousands';
const REPLICA_NOTES = Object.freeze(['3 replicas (a stand-in; providers run many)', 'why the choice matters: Prefix caching']);
const QUEUE_NOTE = 'queue wait: 0 steps here (toy)';
const PICK_NOTE = 'picks = samples from the model\'s probabilities (Picking the next token)';
const PREFILL_NOTE = `${PROMPT.length} tokens in, ${PROMPT.length} K/V tiles written, 1 token out`;

// ---- frame 1: you press Enter; three tokens type in and leave as one request ----
const TYPE_AT = Object.freeze([0.08, 0.2, 0.32]);
export function drawFrame1(svg, p) {
  pipeline(svg, { ...IDLE, you: 'active' });
  cacheTiles(svg, 0);
  const slide = seg(p, 0.6, 0.95);
  hop(svg, 'toRouter', slide);
  promptChips(svg, { x: lerp(PROMPT_AT.you, PROMPT_AT.router, ease(slide)), typed: TYPE_AT.filter((t) => p >= t).length });
  note(svg, 0, TOY_NOTE, seg(p, 0.35, 0.5));
}

// ---- frame 2: a router sends it to one of three replicas ----
function frame2States(p) {
  if (p < 0.1) return { ...IDLE, you: 'active' };
  return p < 0.6 ? { ...IDLE, router: 'active' } : CHOSEN;
}
export function drawFrame2(svg, p) {
  pipeline(svg, frame2States(p));
  cacheTiles(svg, 0);
  const move = seg(p, 0.15, 0.6);
  hop(svg, 'toReplica', move);
  promptChips(svg, { x: lerp(PROMPT_AT.router, PROMPT_AT.replica, ease(move)) });
  note(svg, 0, TOY_NOTE, leaving(p));
  REPLICA_NOTES.forEach((text, row) => note(svg, row, text, seg(p, 0.6, 0.75)));
}

// ---- frame 3: inside the replica, the scheduler; a queue lane; your request reaches the GPU's time axis ----
export function drawFrame3(svg, p) {
  pipeline(svg, p < 0.1 ? CHOSEN : { ...CHOSEN, scheduler: 'active' });
  cacheTiles(svg, 0);
  const lane = seg(p, 0.05, 0.2) * (1 - seg(p, 0.6, 0.9));
  region(svg, QUEUE_LANE, lane);
  label(svg, QUEUE_LANE.x + QUEUE_LANE.w + 6, QUEUE_LANE.y + QUEUE_LANE.h / 2, 'queue', { opacity: lane });
  const enter = seg(p, 0.1, 0.35);
  const leave = seg(p, 0.6, 0.9);
  hop(svg, 'toScheduler', enter);
  hop(svg, 'toGpu', leave);
  const x = leave > 0 ? lerp(PROMPT_AT.queue, PROMPT_AT.gpu, ease(leave)) : lerp(PROMPT_AT.replica, PROMPT_AT.queue, ease(enter));
  promptChips(svg, { x });
  REPLICA_NOTES.forEach((text, row) => note(svg, row, text, leaving(p)));
  note(svg, 0, QUEUE_NOTE, seg(p, 0.35, 0.5));
  yourBar(svg, [], leave);
}

// ---- frame 4: prefill reads all three tokens in one pass, writes three K/V tiles and picks "down" ----
export function drawFrame4(svg, p) {
  pipeline(svg, p < 0.1 ? { ...CHOSEN, scheduler: 'active' } : CHOSEN);
  const rise = seg(p, 0.1, 0.4);
  if (rise < 1) {
    promptChips(svg, { x: PROMPT_AT.gpu, y: lerp(CHIP_Y, GPU.y + 18, ease(rise)), opacity: 1 - seg(rise, 0.55, 1), caption: rise === 0 });
  }
  cacheTiles(svg, p >= 0.4 ? PROMPT.length : 0);
  yourBar(svg, barSteps(seg(p, 0.1, 0.45), 0));
  outOfGpu(svg, 0, seg(p, 0.45, 0.75));
  note(svg, 0, QUEUE_NOTE, leaving(p));
  note(svg, 0, PICK_NOTE, seg(p, 0.45, 0.6));
  note(svg, 1, PREFILL_NOTE, seg(p, 0.75, 0.9));
}
