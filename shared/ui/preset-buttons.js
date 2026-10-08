// Preset chips as action buttons (Training review shared-6): they move a control (a slider, a size) and keep no
// "selected" state, because that control can sit between presets and a chip left pressed there would claim a value the
// toy is not showing. Same markup as mountChoice's chips (a labelled group of `.choice-option` buttons), minus aria-pressed.
//   mountPresetButtons(root, { id, label, options: [{ value, label }], onPick(value) }) → { destroy }
import { el } from './dom.js';

export function mountPresetButtons(root, { id, label, options, onPick }) {
  if (!Array.isArray(options) || options.length === 0) throw new RangeError('mountPresetButtons: options must be a non-empty array');
  const caption = el('span', { id: `${id}-label`, className: 'choice-label', textContent: label });
  const buttons = options.map((o) => {
    const b = el('button', { type: 'button', className: 'choice-option', textContent: o.label });
    b.dataset.value = String(o.value);
    return b;
  });
  const onClick = (event) => {
    const b = event.target.closest('.choice-option');
    if (b && root.contains(b)) onPick(options[buttons.indexOf(b)].value);
  };
  root.id = id;
  root.className = 'choice choice--chips';
  root.setAttribute('role', 'group');
  root.setAttribute('aria-labelledby', caption.id);
  root.replaceChildren(caption, ...buttons);
  root.addEventListener('click', onClick);
  return { destroy() { root.removeEventListener('click', onClick); root.replaceChildren(); } };
}
