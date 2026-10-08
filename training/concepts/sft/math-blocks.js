// The math panel's blocks (storyboard §7). The worked line is templated from math/sft.js, never typed.
import { maskSummary } from '@math/sft.js';
import { SEGMENTS } from './numbers.js';

const tex = String.raw;

export function mathBlocks() {
  const { trained, total } = maskSummary(SEGMENTS);
  return [
    {
      tex: tex`\mathcal{L}_{\text{SFT}} = -\frac{1}{\sum_t \htmlClass{hl-m}{m_t}} \sum_{t} \htmlClass{hl-m}{m_t} \ln p_\theta(x_t \mid x_{<t}),
\qquad
\htmlClass{hl-m}{m_t} = \begin{cases} 1 & x_t \text{ written by the assistant} \\ 0 & \text{prompt, template, tool output, masked error} \end{cases}`,
    },
    {
      tex: tex`\text{worked (default toggles):}\ \sum_t \htmlClass{hl-m}{m_t} = ${trained}\ \text{of}\ T = ${total}\ \Rightarrow\ \mathcal{L}_{\text{SFT}} \text{ averages over those } ${trained}\ \text{tokens}`,
      note: 'The same count as the toy\'s "Check my work".',
    },
  ];
}
