// pretraining toy view model (pure, no DOM): state + data → every string the toy prints, and the try-this list.
// Losses, means and perplexities come from math/lm.js; Kimi K3's vocabulary from data/models.json.
import { tokenLoss, meanLoss, perplexity, uniformLoss } from '@math/lm.js';
import { lookupFact } from '@shared/claims.js';
import { STAND_INS, UNIFORM_VOCAB, PROB_CELLS, P_ON } from './numbers.js';
import { INITIAL_STATE, POSITIONS, checkWork, formatProb, formatLoss, formatPerplexity, selectPosition, setProb, allUniform, targetOf, wordOf } from './format.js';

const kimiVocab = (data) => lookupFact(data?.models, 'kimi-k3', 'vocab_size')?.value ?? null;
const grouped = (n) => n.toLocaleString('en-US');

// One entry per predicting position 1–7: the target chip it grades, its probability and its loss.
function strip(state) {
  return POSITIONS.map((pos) => {
    const p = state.probs[pos - 1];
    const loss = tokenLoss(p);
    const chip = targetOf(pos);
    return { pos, chip, word: wordOf(pos), label: `position ${chip}: ${wordOf(pos)}`, p, pText: formatProb(p), loss, lossText: formatLoss(loss), selected: pos === state.pos };
  });
}

export function toyView(state, data) {
  const cells = strip(state);
  const chosen = cells[state.pos - 1];
  const mean = meanLoss(state.probs);
  const vocab = kimiVocab(data);
  return {
    strip: cells,
    selected: { word: chosen.word, pText: chosen.pText, lossText: chosen.lossText },
    mean: formatLoss(mean),
    perplexity: formatPerplexity(perplexity(mean)),
    ref16: formatLoss(uniformLoss(UNIFORM_VOCAB)),
    refKimi: vocab ? formatLoss(uniformLoss(vocab)) : '—',
    kimiVocab: vocab ? grouped(vocab) : '—',
    checkWork: checkWork(state),
  };
}

const view = (state) => toyView(state, null);
const signed = (x) => `${x < 0 ? '−' : '+'}${Math.abs(x).toFixed(3)}`;

// Storyboard §6 "Try this", every number computed from the toy's own states. `x` marks code text.
export function tryThis(data) {
  const start = view(INITIAL_STATE);
  const [certain, miss] = [1, 0.01].map((p) => setProb(INITIAL_STATE, p));
  const [up, down] = [miss, certain].map((s) => meanLoss(s.probs) - meanLoss(STAND_INS));
  const on = view(selectPosition(INITIAL_STATE, 4));
  const uniform = view(allUniform(INITIAL_STATE));
  const vocab = kimiVocab(data);
  const near = vocab ? uniformLoss(vocab).toFixed(2) : '—';
  return [
    {
      prompt: `Predict first: select \`${start.selected.word}\` (${start.selected.pText}, loss ${start.selected.lossText}). Which moves the mean more: making it certain, or making it a confident miss? `
        + `Slide to 1.00: mean ${start.mean} → ${view(certain).mean} (down ${Math.abs(down).toFixed(3)}). Slide to 0.01: its loss is ${view(miss).selected.lossText} and the mean jumps to ${view(miss).mean}; perplexity ${start.perplexity} → ${view(miss).perplexity}.`,
      insight: 'the log punishes confident mistakes far more than it rewards certainty.',
      rest: ` One badly wrong token moves the average almost five times as much (${signed(up)}) as one perfect one does (${signed(down)}).`,
    },
    {
      prompt: `Select \`on\`, the model's top guess at ${P_ON.toFixed(3)} (against ${PROB_CELLS[1].toFixed(3)} for the runner-up \`.\`). Its loss is ${on.selected.lossText}, not 0.`,
      insight: 'there is no credit for being top-1, only for probability.',
      rest: ' The model keeps learning even on tokens it already ranks first, by making them more likely.',
    },
    {
      prompt: `Press \`uniform guess over 16\`: every loss becomes ${uniform.strip[0].lossText}, the mean is ${uniform.mean} and perplexity is exactly ${UNIFORM_VOCAB}. `
        + `Now read the reference mark: a model that knows nothing about Kimi K3's vocabulary of ${vocab ? grouped(vocab) : '—'} pieces would start near ${near}.`,
      insight: 'perplexity is the effective number of choices.',
      rest: ' Training a real model is the long walk from about ln(vocabulary size) down toward the loss of the text itself.',
    },
  ];
}
