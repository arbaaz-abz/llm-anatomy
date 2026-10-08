// training-pipeline toy view model (pure, no DOM): state + data → every string the toy prints.
// Shares come from math/pipeline.js budgetShares via formatShare; published figures from ctx.data.
import { budgetShares } from '@math/pipeline.js';
import { lookupFact, fillText } from '@shared/claims.js';
import { formatShare } from '@shared/glyphs/bars.js';
import { STAGES, STOPS } from './numbers.js';
import { RECIPES, RECIPE_ORDER } from './recipes.js';
import { checkWork, totalText } from './format.js';

export const MODEL_CHIPS = Object.freeze(RECIPE_ORDER.map((value) => ({ value, label: RECIPES[value].label })));
export const INITIAL_STATE = Object.freeze({ model: 'glm5', stage: 1 });
export const NOT_DESCRIBED = 'not described';

export function recipeOf(model) {
  const recipe = RECIPES[model];
  if (!recipe) throw new RangeError(`training-pipeline toy: unknown model "${model}"`);
  return recipe;
}

// The bar's parts for a model: published counts from data, { value: null } for anything unpublished.
export function tokenParts(model, data) {
  const recipe = recipeOf(model);
  return recipe.tokens.map(([name, key, hue]) => {
    if (key === null) return { name, value: null };
    const value = lookupFact(data?.models, recipe.id, key)?.value;
    if (!Number.isFinite(value)) throw new Error(`training-pipeline toy: models.${recipe.id}.${key} is not a published number`);
    return { name, value, hue };
  });
}

export const stageText = (model, stage, data) => {
  const template = recipeOf(model).stages[stage - 1];
  return template === null ? null : fillText(template, data);
};

export function stageStrip(model) {
  return STAGES.map((s) => ({ n: s.n, label: s.label, name: s.name, described: recipeOf(model).stages[s.n - 1] !== null }));
}



export function toyView(state, data) {
  const parts = tokenParts(state.model, data);
  const text = stageText(state.model, state.stage, data);
  const stage = STAGES[state.stage - 1];
  return {
    strip: stageStrip(state.model),
    inspector: `Stage ${stage.n}, ${stage.name}: ${text ?? `${NOT_DESCRIBED} in this report.`}`,
    stops: stopsText(state.stage),
    parts,
    knownTotal: totalText(budgetShares(parts).knownTotal),
    checkWork: checkWork(parts),
  };
}

const shareOf = (model, name, data) => {
  const found = budgetShares(tokenParts(model, data)).parts.find((q) => q.name === name);
  return formatShare(found.share);
};
const lit = (model, n) => recipeOf(model).stages[n - 1] !== null;
const join = (labels) => (labels.length > 1 ? `${labels.slice(0, -1).join(', ')} and ${labels.at(-1)}` : labels[0]);
export const stopsText = (stage) => `Taught in ${join(STOPS[stage - 1].map((slug) => `[[${slug}]]`))}.`;

// The storyboard's three try-this items; every number is computed (README lesson 34).
export function tryThis(data) {
  const name = (m) => recipeOf(m).label.replace(' (reported)', ''); // the storyboard's prose says "Olmo 3"
  const merged = RECIPE_ORDER.filter((m) => lit(m, 5)).map(name);
  const unmerged = RECIPE_ORDER.filter((m) => !lit(m, 5)).map(name);
  return [
    {
      prompt: `Step through the five presets and watch the merge stage: lit for ${join(merged)}, ${NOT_DESCRIBED} for ${join(unmerged)}.`,
      insight: 'merging specialists is the 2026 frontier step,',
      rest: ' used by the largest open MoE labs; smaller and fully open pipelines often stop at one RL stage ([[distillation]] explains the merge).',
    },
    {
      prompt: `With GLM-5, read the bar: pretrain ${shareOf('glm5', 'pretrain', data)}, mid-train ${shareOf('glm5', 'mid-train', data)}, post-training not published. Then select stage 4 and read its unit: environments, not tokens. Switch to Olmo 3: ${shareOf('olmo3', 'pretrain', data)} / ${shareOf('olmo3', 'mid-train', data)}, and its post-training stages are counted in traces, pairs and prompts.`,
      insight: 'tokens are the wrong ruler for post-training.',
      rest: ' It is reported in examples, environments and compute, because each of its tokens costs far more than a pretraining token.',
    },
    {
      prompt: 'Select Kimi K3: the bar holds only neutral segments labeled "no published shares", yet every stage is lit.',
      insight: 'a recipe can be fully described without its budget.',
      rest: ' Labs often publish the method but not the amounts, which is why the course labels unknowns rather than guessing.',
    },
  ];
}
