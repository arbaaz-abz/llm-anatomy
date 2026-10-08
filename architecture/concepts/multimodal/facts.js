// multimodal §8: the facts rows and the dated notes under the stage. Every dated number is a {entry.key|format}
// placeholder filled from data/models.json (spec §7); nothing here types a model spec.

export const FRAMING = 'In 2026 every vision-capable frontier model on this list keeps a separate vision encoder and projector; they differ in whether the encoder was trained with the backbone, how large images may be, and how neighbors are merged. Several leading models see no images at all.';

export const ROWS = Object.freeze([
  { claim: 'Kimi K3 ({kimi-k3.release_date|year}): native multimodal; vision encoder MoonViT-V2, {kimi-k3.vision_encoder_params|count} parameters, {kimi-k3.vision_encoder_layers} layers, patch {kimi-k3.patch_size}, trained from scratch with next-token prediction (a SigLIP-initialized version had unstable gradients); merges {kimi-k3.vision_merge} × {kimi-k3.vision_merge}; images up to {kimi-k3.max_image_side} × {kimi-k3.max_image_side} pixels.' },
  { claim: 'MiniMax-M3 ({minimax-m3.release_date|year}): native multimodal, "mixed-modality training from the very first step"; ViT of {minimax-m3.vision_encoder_layers} layers, width {minimax-m3.vision_width}, patch {minimax-m3.patch_size}; images up to {minimax-m3.max_image_side} × {minimax-m3.max_image_side} pixels; 3D RoPE.' },
  { claim: 'Mistral Large 4 ({mistral-large-4.release_date|year}): multimodal input, text output; a {mistral-large-4.vision_encoder_params|count}-parameter vision encoder.' },
  { claim: 'DeepSeek-V4.1-Flash ({deepseek-v4.1-flash.release_date|year}): the first DeepSeek model with a vision encoder: {deepseek-v4.1-flash.vision_encoder_layers} layers, merges {deepseek-v4.1-flash.vision_merge} × {deepseek-v4.1-flash.vision_merge}, a small MLP projector.' },
  { claim: 'Text-only in 2026: DeepSeek-V4-Pro ({deepseek-v4-pro.modalities}; its paper lists multimodality as future work), GLM-5.3 ({glm-5.3.modalities}), gpt-oss-120b ({gpt-oss-120b.modalities}) and Qwen3.8 ({qwen3.8.modalities}; its card says "text-only model").' },
  // LLaVA (2023) is a classic paper with no data entry: cited in "Go deeper" below, so the row carries no source link of its own.
  { claim: 'The adapter recipe, LLaVA (2023): a pretrained ViT, a small MLP projector and an already-trained language model.', derived: true },
  { claim: 'Video in Kimi K3 ({kimi-k3.release_date|year}): frames are sampled, and the encoder uses {kimi-k3.video}.' },
]);

const PHONE = 'Kimi K3 and MiniMax-M3 both cut images into {kimi-k3.patch_size}-pixel patches.';

// Notes under the stage for the frames whose captions need a dated fact, by 0-based frame index.
export const BELOW = Object.freeze({
  0: [PHONE],
  2: ['Kimi K3\'s vision encoder has {kimi-k3.vision_encoder_layers} layers and {kimi-k3.vision_encoder_params|count} parameters.'],
  3: ['Kimi K3 merges {kimi-k3.vision_merge} × {kimi-k3.vision_merge}, so a quarter of the tokens; DeepSeek-ViT merges {deepseek-v4.1-flash.vision_merge} × {deepseek-v4.1-flash.vision_merge}, so a ninth (reported).'],
  4: ['The projector is a small MLP; DeepSeek-ViT\'s is a small MLP too (reported).'],
  6: ['Kimi K3\'s largest input is {kimi-k3.max_image_side} × {kimi-k3.max_image_side} pixels; at patch {kimi-k3.patch_size} and a {kimi-k3.vision_merge} × {kimi-k3.vision_merge} merge that is the count above.'],
  9: [
    'Native in practice: Kimi K3 trained its encoder from scratch with next-token prediction; MiniMax-M3 reports "mixed-modality training from the very first step". The adapter recipe is LLaVA-style (2023).',
    'Not every 2026 frontier model sees images: DeepSeek-V4-Pro (its paper lists images as future work), GLM-5.3, gpt-oss-120b and Qwen3.8 are text-only.',
  ],
});
