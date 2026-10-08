// "Shape the draw" view model (storyboard §6): a pure function from the toy state to every string the toy prints.
// Unit-tested; toy.js only paints it. Every number comes from math/sampling.js.
import { LOGITS, NAMED, VOCAB, applyTemperature, collapseOthers, drawSamples, samplingDistribution, topP } from '@math/sampling.js';
import { INITIAL_STATE, checkWork, distributionOf, effectiveTemperature, fmt3, WORDS, quoted } from './format.js';

export const DRAWS = 20;
export const SEED_RANGE = Object.freeze({ min: 1, max: 5 });
const FIRST = 8;
const NAMES = Object.freeze(['on', 'dot', 'and', 'the', 'others']); // data-readout suffixes

const first = (draws) => draws.slice(0, FIRST).map((i) => VOCAB[i]).join(' ');
const categoryCounts = (draws) => {
  const named = NAMED.map((i) => draws.filter((d) => d === i).length);
  return [...named, draws.length - named.reduce((a, b) => a + b, 0)];
};

// One table row per displayed word: the probability after temperature and after the filters; "cut" for a token the filters removed.
function distributionRows(before, after) {
  return WORDS.map((word, i) => {
    const isOthers = i === NAMED.length;
    const [pre, post] = [before, after].map((c) => (isOthers ? c.others.each : c.named[i]));
    const sub = (c) => (isOthers ? `together ${fmt3(c.others.together)}` : null);
    const keptNote = isOthers ? `${sub(after)} · ${after.others.kept} of ${after.others.of} kept` : (post === 0 && pre > 0 ? 'cut' : null);
    return { name: NAMES[i], label: quoted(word), labelSub: isOthers ? 'value of each' : null, pre: fmt3(pre), preSub: sub(before), post: fmt3(post), postSub: keptNote };
  });
}

export function view(state) {
  const dist = distributionOf(state);
  const draws = drawSamples(dist.probs, DRAWS, state.seed);
  const rows = distributionRows(collapseOthers(applyTemperature(LOGITS, effectiveTemperature(state)), NAMED), collapseOthers(dist.probs, NAMED));
  return {
    rows,
    kept: `${dist.kept} of ${LOGITS.length}`,
    mass: fmt3(dist.mass),
    counts: categoryCounts(draws),
    drawsTitle: `${DRAWS} draws, seed ${state.seed}`,
    firstEight: first(draws),
    checkWork: checkWork(state),
    greedy: state.greedy,
  };
}

// The three "try this" suggestions, with every number computed from the same functions the toy uses.
export function tryThis() {
  const at = (temperature) => ({ probs: applyTemperature(LOGITS, temperature), dist: samplingDistribution(LOGITS, { temperature }) });
  const counts = (temperature) => categoryCounts(drawSamples(at(temperature).probs, DRAWS, INITIAL_STATE.seed));
  const [half, twice] = [0.5, 2].map((t) => ({ on: fmt3(at(t).probs[NAMED[0]]), others: fmt3(collapseOthers(at(t).probs, NAMED).others.together), counts: counts(t) }));
  const keptAt = (t) => topP(at(t).probs, 0.7);
  const tiedKept = collapseOthers(keptAt(2).probs, NAMED).others.kept;
  const seedText = (seed) => first(drawSamples(applyTemperature(LOGITS, 1), DRAWS, seed));
  return [
    `Temperature 0.5: "on" ${half.on}, the 12 others ${half.others} together; seed 1 draws "on" ${half.counts[0]} times in ${DRAWS}. Temperature 2: "on" ${twice.on}, others ${twice.others}; ${twice.counts[4]} of ${DRAWS} draws come from the 12 unlikely words. Tap Greedy: "on" every time. Temperature is one dial from "always the favorite" to "almost uniform"; it changes how the draw spends probability, not what the model knows.`,
    `Top-p 0.7 at temperature 1: ${keptAt(1).kept} kept. At 0.5: ${keptAt(0.5).kept} kept. At 2: ${keptAt(2).kept} kept (${tiedKept} of the 12 tied words, chosen by vocabulary order). Now top-k 3 instead: ${samplingDistribution(LOGITS, { topK: 3 }).kept} kept at every temperature. Top-p adapts to how sure the model is; top-k does not.`,
    `Temperature 1, no filters, step the seed 1 → 2 → 3: the first eight tokens change each time (seed 2: ${seedText(2)}). Return to seed 1 and they come back exactly ("${seedText(1)}"). The randomness lives in the sampler, not in the model; fix the seed and the text repeats.`,
  ];
}

export { INITIAL_STATE };
