// moe storyboard §5 captions, verbatim. Imported by the page test and the e2e spec. (The storyboard gives no "Check my work" box.)
export const CAPTIONS = [
  'In a Mixture-of-Experts layer, a router gives each expert a score for this token. It is one small learned matrix, 8 by 8 here.',
  'Only the top two experts run for this token: top-k with k equal to 2. The other six are skipped, not computed.',
  'The two chosen scores become gate weights that add to 1. The layer\'s output is the two experts\' outputs, blended by those weights.',
  'Every token is routed on its own. These four words land on seven different experts, and one expert gets nothing.',
  'Every expert is stored, but each token runs only two. Total parameters grow with the expert count; the work per token does not.',
  'Fine-grained experts cut each expert in half and pick twice as many. The work per token stays the same, but the router has far more combinations to choose from.',
  'A shared expert runs for every token and holds what all tokens need. Here one shared plus one routed expert does the same work as before.',
  'Routers drift: across a batch, a few experts get most tokens. Here the busiest expert has three times its fair share, and in practice the busiest expert sets the pace.',
  'DeepSeek\'s fix adds a per-expert bias to the scores, only for choosing. The gate weights still come from the raw scores.',
  'After each batch, idle experts\' biases go up a little and busy ones\' go down. Within six steps the loads are close to even.',
];
