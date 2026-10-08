// model-card stage model (pure, no DOM): data/models.json + hardware.json → every string and number the stage prints.
// Frames only draw this; nothing on the stage is typed in. The followed card is DeepSeek-V4-Pro (storyboard §4).
import { decodeCard, activeShare, routedShare, activeWithEmbeddings, cachePerToken, cacheForConversation, formatByteRange, FIELD_GUIDE } from '@math/card.js';
import { sharePct } from '@math/memory.js';
import { formatBytes, formatCount } from '@math/core.js';
import { lookupFact } from '@shared/claims.js';
import { int, sharePercent, gpuShareText, cacheConversationText, hbmBytes, ZOOM_FRACTION } from './format.js';

export const CARD_ID = 'deepseek-v4-pro';
export const CARD_NAME = 'DeepSeek-V4-Pro';
// The stage cannot link, so an exit names its lesson by title (the page test pins these to shared/concepts.json).
export const EXIT_TITLES = Object.freeze({ 'scaling-laws': 'Scaling laws', pretraining: 'Pretraining', sampling: 'Picking the next token', quantization: 'Quantization' });
const SCALE_ID = 'llama-3.1-70b'; // frame 6's size comparison: a plain GQA model

const entryOf = (data, id) => data?.models?.entries?.find((e) => e.id === id) ?? null;
const factOf = (data, id, key) => lookupFact(data?.models, id, key)?.value ?? null;

// The ten chips of the card column, in order, with the field each reads (storyboard §5 frame 1).
const CHIP_KEYS = Object.freeze(['total_params', 'active_params', 'layers', 'experts_total', 'attention', 'context_length', 'modalities', 'optimizer', 'pretrain_tokens', 'license']);

function chipLabels(rows, f) {
  const shown = (key) => rows.find((r) => r.key === key).display;
  const heads = f('n_kv_heads');
  return {
    total_params: `${shown('total_params')} total`,
    active_params: `${shown('active_params')} active`,
    layers: `${shown('layers')} layers`,
    experts_total: `${shown('experts_total')} + ${f('experts_shared')} experts, top-${shown('experts_active')}`,
    attention: `${heads} KV head${heads === 1 ? '' : 's'} × ${f('head_dim')}, compressed`,
    context_length: `${shown('context_length')} context`,
    modalities: shown('modalities'),
    optimizer: shown('optimizer'),
    pretrain_tokens: `${shown('pretrain_tokens')} tokens`,
    license: shown('license'),
  };
}

// Chips from the paper or config, added in frames 4, 7 and 9 (labeled "from the paper / config" on the stage).
function extras(f) {
  const formats = String(f('weight_formats')).split(' (')[0];
  return {
    frame4: [`expert hidden ${int(f('expert_hidden'))}`],
    frame7: [`RoPE base ${int(f('rope_theta'))}, YaRN ×${f('yarn_factor')}`, `stages ${f('context_stages')}`],
    frame9: [`MTP depth ${f('mtp_depth')}`, formats],
  };
}

function cacheModel(data, f) {
  const entry = entryOf(data, CARD_ID);
  const perToken = cachePerToken(entry);
  const tokens = f('context_length');
  const conversation = cacheForConversation(entry, tokens);
  const hbm = hbmBytes(data);
  const scaleEntry = entryOf(data, SCALE_ID);
  const scaleBytes = cacheForConversation(scaleEntry, tokens);
  const ends = [conversation].flat();
  return {
    perTokenText: formatByteRange(perToken.bytes),
    tokens,
    tokensText: int(tokens),
    conversationText: cacheConversationText(conversation),
    scaleText: cacheConversationText(scaleBytes),
    gpuLabel: `${hbm / 1e9} GB GPU`,
    gpuFill: [sharePct(ends[0], hbm) / 100, sharePct(ends[ends.length - 1], hbm) / 100],
    gpuShareText: gpuShareText(conversation, hbm),
    lowShare: sharePct(ends[0], hbm).toFixed(1),
    highShare: sharePct(ends[ends.length - 1], hbm).toFixed(1),
    paperRatioText: `${sharePct(f('v32_kv_ratio_1m'), 1, { decimals: 0 })}%`,
  };
}

function expertsModel(f) {
  const [total, used, shared] = [f('experts_total'), f('experts_active'), f('experts_shared')];
  return { total, used, shared, others: total - used, routedText: sharePercent(used / total), hidden: int(f('expert_hidden')) };
}

function kvModel(f) {
  return { merge: f('csa_merge'), topK: int(f('csa_top_k')), hcaMerge: f('hca_merge'), window: f('window'), heads: f('n_kv_heads'), headDim: f('head_dim') };
}

// The four conflict rows of frame 10: both numbers with the source each came from. The source labels are checked
// against the data notes in the page test, so a reworded note fails loudly instead of mislabeling an end.
function conflictsModel(data, rows) {
  const row = (id, key) => decodeCard(entryOf(data, id)).find((r) => r.key === key);
  const mistral = entryOf(data, 'mistral-large-4');
  const [lowActive, highActive] = [factOf(data, 'mistral-large-4', 'active_params'), activeWithEmbeddings(mistral)];
  const total = factOf(data, 'mistral-large-4', 'total_params');
  const glm = row('glm-5.3', 'layers').value;
  const context = row('mistral-large-4', 'context_length').value;
  const cache = rows.find((r) => r.key === 'kv_bytes_per_token').value;
  return [
    { title: 'GLM-5.3 · layers', low: String(glm[0]), high: String(glm[1]), lowSource: 'config.json', highSource: 'GLM-5 paper', reported: false },
    { title: 'Mistral Large 4 · context', low: formatCount(context[0]), high: formatCount(context[1]), lowSource: 'evaluators', highSource: 'Mistral card', reported: true },
    { title: `${CARD_NAME} · cache per token`, low: formatBytes(cache[0]), high: formatBytes(cache[1]), lowSource: '1 : 1 mix (config)', highSource: '3 : 1 (blog summaries)', reported: true },
    {
      title: 'Mistral Large 4 · active', low: `${formatCount(lowActive)}`, high: `${formatCount(highActive)}`, lowSource: 'routed', highSource: 'with embeddings', reported: true,
      shares: `${sharePercent(lowActive / total)} · ${sharePercent(highActive / total)} of ${formatCount(total, { digits: 3 })}`,
    },
  ];
}

// null when the page has no data (the node page test validates the spec without it; a mounted page always has it).
export function stageModel(data) {
  const entry = entryOf(data, CARD_ID);
  if (!entry) return null;
  const rows = decodeCard(entry);
  const f = (key) => factOf(data, CARD_ID, key);
  const labels = chipLabels(rows, f);
  const [total, active] = [f('total_params'), f('active_params')];
  return {
    name: CARD_NAME,
    chips: CHIP_KEYS.map((key) => ({ key, label: labels[key] })),
    extras: extras(f),
    active: {
      total, active, notUsed: total - active, zoom: ZOOM_FRACTION, zoomText: `${sharePct(ZOOM_FRACTION, 1, { decimals: 0 })}%`,
      formula: `${rows.find((r) => r.key === 'active_params').display} ÷ ${rows.find((r) => r.key === 'total_params').display} = ${sharePercent(activeShare(entry))}`,
      line: `active ${sharePercent(activeShare(entry))} · not used ${sharePercent((total - active) / total)}`,
    },
    layers: String(f('layers')),
    experts: expertsModel(f),
    kv: kvModel(f),
    cache: cacheModel(data, f),
    context: {
      positions: int(f('context_length')), ropeBase: int(f('rope_theta')), ropeBaseCompressed: int(f('rope_theta_compressed')),
      yarn: f('yarn_factor'), stages: f('context_stages'),
    },
    modality: { text: `${f('modalities')}-only`, confidence: entry.facts.modalities.confidence },
    exits: [
      { chip: CHIP_KEYS.indexOf('optimizer'), lesson: FIELD_GUIDE.optimizer.lesson },
      { chip: CHIP_KEYS.indexOf('pretrain_tokens'), lesson: FIELD_GUIDE.pretrain_tokens.lesson },
      { extra: 0, lesson: 'sampling' },
      { extra: 1, lesson: 'quantization' },
    ].map((exit) => ({ ...exit, title: EXIT_TITLES[exit.lesson] })),
    conflicts: conflictsModel(data, rows),
  };
}
