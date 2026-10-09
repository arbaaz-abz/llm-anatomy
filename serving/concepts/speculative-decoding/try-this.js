// speculative-decoding try-this list (storyboard §6): prompt → insight → rest, in the course format (XT-2, README lesson 34).
// Numbers are computed from specdec.js; the one dated figure (DeepSeek-V3's reported gain) is a data placeholder.
import { fillText } from '@shared/claims.js';
import { batchSpeedup, expectedTokens, outputDistribution, simpleSpeedup, verifyToken } from '@math/specdec.js';
import { cellText, chanceText, ratioText, tokensText } from './format.js';
import { ALPHA, C, K, MODEL, P, Q, ZOOM_WORDS } from './numbers.js';

const list = (values, f) => values.map(f).join(' → ');
const speed = (alpha, k, c, batch) => batchSpeedup({ alpha, k, c, batch, model: MODEL }).speedup;

function first() {
  const ks = [1, 3, 5, 8];
  return {
    prompt: `With Acceptance rate α at 0.7, Drafter cost per guess at 0.05 and Users in the batch at 1, slide Guesses per round 1 → 3 → 5 → 8: tokens per round ${list(ks, (k) => tokensText(expectedTokens(ALPHA, k)))}; speedup ${list(ks, (k) => ratioText(simpleSpeedup(ALPHA, k, C)))}.`,
    insight: 'guesses have diminishing returns.',
    rest: ' A guess counts only if every earlier one survived, while each one costs drafting time, so the best number of guesses is small and depends on α and the drafter cost.',
  };
}

function second() {
  const users = [1, 64, 128, 211];
  return {
    prompt: `With Guesses per round at ${K}, slide Users in the batch 1 → 64 → 128 → 211: speedup ${list(users, (u) => ratioText(speed(ALPHA, K, C, u)))}. Now set Guesses per round to 5 at 211 users: ${ratioText(speed(ALPHA, 5, C, 211))}.`,
    insight: 'speculative decoding spends idle arithmetic, so it fades when the batch has none left.',
    rest: ' It is a latency tool for small batches, not a free throughput multiplier.',
  };
}

function third() {
  const mtp = 0.85;
  return {
    prompt: `Press MTP 0.85, which sets Guesses per round to 1: ${tokensText(expectedTokens(mtp, 1))} tokens per round, ${ratioText(speed(mtp, 1, C, 1))} at 1 user and still ${ratioText(speed(mtp, 1, C, 128))} at 128 users.`,
    insight: 'one well-trained extra guess survives large batches,',
    rest: " because the verify pass only doubles the tokens. That is close to DeepSeek-V3's reported {sv:deepseek-v3-mtp.tps_gain|raw}× tokens per second and why MTP heads are popular for serving.",
  };
}

function fourth() {
  const keep = (i) => chanceText(verifyToken(P, Q, i).accept);
  return {
    prompt: `In the position panel, switch Drafter's guess from ${ZOOM_WORDS[0]} to ${ZOOM_WORDS[1]}, then ${ZOOM_WORDS[2]}: keep chance ${keep(0)} → ${keep(1)} → ${keep(2)}; the result stays ${outputDistribution(P, Q).map(cellText).join(', ')} every time.`,
    insight: 'only guesses the drafter overrates are ever rejected, and the leftover repays exactly what rejection removed,',
    rest: " so the target's distribution is preserved whatever the drafter does.",
  };
}

export function tryThis(data) {
  return [first(), second(), third(), fourth()].map((t) => ({ ...t, rest: fillText(t.rest, data) }));
}
