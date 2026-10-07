// Segmented controls, preset chips and toggles: native buttons with aria-pressed (keyboard and focus for free).

export function checkOptions(options, value) {
  if (!Array.isArray(options) || options.length === 0) throw new RangeError('choice: options must be a non-empty array');
  const values = options.map((o) => o?.value);
  if (new Set(values).size !== values.length) throw new RangeError(`choice: option values must be unique, got ${values.join(', ')}`);
  if (options.some((o) => typeof o.label !== 'string' || o.label === '')) throw new RangeError('choice: every option needs a label');
  if (!values.includes(value)) throw new RangeError(`choice: value ${value} is not one of ${values.join(', ')}`);
  return true;
}

const button = (text, attrs = {}) => {
  const b = document.createElement('button');
  b.type = 'button';
  b.textContent = text;
  Object.entries(attrs).forEach(([k, v]) => b.setAttribute(k, v));
  return b;
};

// variant: 'segmented' (one of a few modes) or 'chips' (presets). onChange gets the option's value.
export function mountChoice(root, { id, label, options, value, onChange, variant = 'segmented' }) {
  checkOptions(options, value);
  let current = value;
  const caption = Object.assign(document.createElement('span'), { id: `${id}-label`, className: 'choice-label', textContent: label });
  const buttons = options.map((o) => button(o.label, { class: 'choice-option', 'data-value': String(o.value) }));
  const paint = () => buttons.forEach((b, i) => b.setAttribute('aria-pressed', String(options[i].value === current)));
  const choose = (next, notify) => {
    if (next === current) return;
    checkOptions(options, next);
    current = next;
    paint();
    if (notify) onChange?.(current);
  };
  const onClick = (event) => {
    const b = event.target.closest('.choice-option');
    if (b && root.contains(b)) choose(options[buttons.indexOf(b)].value, true);
  };
  root.id = id;
  root.className = `choice choice--${variant}`;
  root.setAttribute('role', 'group');
  root.setAttribute('aria-labelledby', caption.id);
  root.replaceChildren(caption, ...buttons);
  root.addEventListener('click', onClick);
  paint();
  return {
    get value() { return current; },
    set(next) { choose(next, true); },
    destroy() { root.removeEventListener('click', onClick); root.replaceChildren(); },
  };
}

export function mountToggle(root, { id, label, value = false, onChange }) {
  let on = Boolean(value);
  const b = button('', { id, class: 'toggle' });
  const paint = () => { b.setAttribute('aria-pressed', String(on)); b.textContent = `${label}: ${on ? 'on' : 'off'}`; };
  const onClick = () => { on = !on; paint(); onChange?.(on); };
  b.addEventListener('click', onClick);
  root.replaceChildren(b);
  paint();
  return {
    get value() { return on; },
    destroy() { b.removeEventListener('click', onClick); root.replaceChildren(); },
  };
}
