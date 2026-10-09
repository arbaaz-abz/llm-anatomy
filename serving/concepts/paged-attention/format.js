// paged-attention's pure formatters and toy defaults (in the coverage gate). Shares go through sharePct at one decimal,
// exact integers through formatInt, bytes through formatBytes (README lesson 35).
import { formatInt, formatBytes, deepFreeze } from '@math/core.js';
import { sharePct } from '@math/memory.js';

export const INITIAL_STATE = deepFreeze({ blockSize: 4, step: 3, sharedPrefix: false, model: 'toy', follow: 'C' });
export const SLIDER_VALUES = deepFreeze({ blockSize: [2, 4, 8, 16] });
export const STEP_RANGE = Object.freeze([0, 6]);

export const shareText = (part, whole) => `${sharePct(part, whole).toFixed(1)}%`;
export const plural = (n, noun) => `${formatInt(n)} ${noun}${n === 1 ? '' : 's'}`;
export const exactBytes = (n) => `${formatInt(n)} B`;
export const bytesPair = (n) => ({ value: exactBytes(n), sub: `≈ ${formatBytes(n)}` });

// A request's status at the step shown: not yet arrived, waiting for memory, running, or done.
export function statusText(live, arrives, step) {
  if (live.running) return 'running';
  if (live.done) return 'done';
  return step >= arrives ? 'waiting' : 'not arrived';
}

export const stepText = (admitted) => (admitted == null ? 'not started' : `step ${admitted}`);

// Mean share of the pool wasted over the steps given (each step's `waste` in slots).
export const meanWaste = (wastes, poolSlots) => shareText(wastes.reduce((a, n) => a + n, 0), poolSlots * wastes.length);

// "30 slots" under a share: the share is the headline, the slot count its basis.
export const shareCell = (slots, poolSlots) => ({ value: shareText(slots, poolSlots), sub: plural(slots, 'slot') });
