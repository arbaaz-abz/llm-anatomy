// multimodal storyboard §5 captions and §6 "Check my work" text, verbatim. Imported by the page test and the e2e spec.
export const CAPTIONS = [
  "The image is cut into a grid of square patches. Here each patch is 4 by 4 pixels; real models use 14 by 14.",
  "Each patch's pixels are flattened and multiplied by one learned matrix, giving one vector per patch. That is the patch embedding.",
  "A vision encoder, a transformer of its own, lets every patch attend to every other one, with no causal mask. Each vector now describes its patch in context.",
  "Neighboring patches are merged four into one, so the image costs a quarter of the tokens. Kimi K3 merges 2 by 2; DeepSeek's newest encoder merges 3 by 3.",
  "A small projector maps each merged vector to the language model's width. Now the image is four tokens, the same shape as a word's vector.",
  "Image tokens and words share one sequence. Under the causal mask, every word can attend to the image, and the blocks treat both alike.",
  "Tokens grow with the image's area. A 1,008-pixel square photo costs 1,296 tokens, and Kimi K3's largest input costs 16,384.",
  "Dynamic resolution keeps each image's own shape instead of resizing it to a square. A wide photo then costs only the tokens it actually covers.",
  "Video pays for every frame. At these settings an hour of video would overflow a million-token context, so encoders also pool frames over time.",
  "The adapter recipe attaches an encoder to a finished text model. Native models train encoder and backbone together from the start, and still keep an encoder.",
];
// The default toy state: Kimi K3 preset, 1,008 × 1,008 image.
export const CHECK_WORK = '1008 ÷ 14 = 72 · 72 × 72 = 5,184 · ÷ (2 × 2) = 1,296';
