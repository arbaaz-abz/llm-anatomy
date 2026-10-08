// Pure formatters, the toy's starting state and the "Check my work" text for the midtraining page (storyboard §6). No DOM.
import { lrAt } from '@math/schedule.js';
import { sharePct } from '@math/memory.js';
import { formatShare } from '@shared/glyphs.js';

export const MINUS = '−';
export const SCHEDULES = Object.freeze([
  { value: 'wsd', label: 'WSD (linear decay)' },
  { value: 'wsd-minus-sqrt', label: 'WSD minus-sqrt' },
  { value: 'cosine', label: 'cosine' },
]);
export const DECAY_VALUES = Object.freeze([5, 10, 15, 20, 25, 30, 31.8, 35, 40]); // percent; 31.8 is MiniMax-M2's
export const STOP_RANGE = Object.freeze({ min: 50, max: 100, step: 5 });
export const RUN_IDS = Object.freeze(['glm-5', 'minimax-m2', 'deepseek-v4-pro', 'kimi-k3']);

// Where the animation ends: WSD, 20% of the run decaying, stop at 60%, GLM-5's stages.
export const INITIAL_STATE = Object.freeze({ schedule: 'wsd', decayFrac: 20, preset: 'custom', stopAt: 60, run: 'glm-5' });

export const SHARE_PCT_DIGITS = 2;
// Shares of the whole run print at two decimals ("94.57%"); the zoomed tail uses formatShare (one decimal from 1%).
export const mainShare = (share) => `${sharePct(share, 1, { decimals: SHARE_PCT_DIGITS }).toFixed(SHARE_PCT_DIGITS)}%`;
export const tailShare = formatShare;

export const fixed = (digits) => (v) => {
  const s = Math.abs(v).toFixed(digits);
  return `${v < 0 && Number(s) !== 0 ? MINUS : ''}${s}`;
};
export const fmt3 = fixed(3);
export const fmt1 = fixed(1);
export const trimNumber = (v) => String(Number(v.toFixed(3)));
// A fraction of the peak: three decimals, a zero printed as "0" (storyboard frame 2's "100% 0").
export const ofPeak = (v) => (v === 0 ? '0' : fmt3(v));

const SUPERSCRIPT = Object.freeze({ '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' });
// Scientific notation at 3 significant figures: 4.50 × 10⁻⁴.
export function sci(v) {
  const [mantissa, exponent] = v.toExponential(2).split('e');
  return `${mantissa} × 10${[...String(Number(exponent))].map((c) => SUPERSCRIPT[c]).join('')}`;
}

export const percentText = (v) => `${trimNumber(v)}%`;

// The schedule options the state names: cosine, or WSD with a linear or minus-sqrt decay that starts at 100% − decayFrac.
export function scheduleOptions(state, schedule = state.schedule) {
  if (schedule === 'cosine') return { kind: 'cosine', total: 1 };
  const decayStart = Number(((100 - state.decayFrac) / 100).toFixed(3));
  return { kind: 'wsd', total: 1, decayStart, decayShape: schedule === 'wsd-minus-sqrt' ? 'minus-sqrt' : 'linear' };
}
export const lrAtStop = (state, schedule = state.schedule) => lrAt(state.stopAt / 100, scheduleOptions(state, schedule));
// The schedule shown beside the chosen one (cosine against WSD, and the reverse).
export const otherSchedule = (schedule) => (schedule === 'cosine' ? 'wsd' : 'cosine');
export const scheduleName = (schedule) => SCHEDULES.find((s) => s.value === schedule).label;

// ---- "Check my work": the plateau, decay and cosine lines of storyboard §6, by where the stop falls ----
function wsdLines(state) {
  const wsd = state.schedule === 'wsd-minus-sqrt' ? 'wsd-minus-sqrt' : 'wsd';
  const start = Number((100 - state.decayFrac).toFixed(1));
  const { stopAt, decayFrac } = state;
  const name = wsd === 'wsd' ? 'WSD' : 'WSD minus-sqrt';
  const head = `${name}: the decay starts at 100% − ${percentText(decayFrac)} = ${percentText(start)} of the run`;
  const at = `stop at ${stopAt}%`;
  const value = fmt3(lrAtStop(state, wsd));
  if (stopAt < start) return [head, `${at}: before ${percentText(start)}, on the plateau → ${value} of peak`];
  if (stopAt === start) return [head, `${at}: where the decay begins → ${value} of peak`];
  const run = `(${stopAt}% − ${percentText(start)}) ÷ ${percentText(decayFrac)}`;
  const fall = wsd === 'wsd' ? `1 − ${run}` : `1 − √(${run})`;
  return [head, `${at}: inside the decay → ${fall} = ${value} of peak`];
}

function cosineLine(state) {
  const t = state.stopAt / 100;
  const c = Math.cos(Math.PI * t);
  const sign = c < 0 ? MINUS : '+';
  return `cosine at ${state.stopAt}%: ½ × (1 + cos(π × ${t.toFixed(2)})) = ½ × (1 ${sign} ${Math.abs(c).toFixed(3)}) = ${fmt3(lrAtStop(state, 'cosine'))} of peak`;
}

export const checkWork = (state) => [...wsdLines(state), cosineLine(state)].join('\n');
