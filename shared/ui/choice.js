// Segmented controls, preset chips and toggles: native buttons with aria-pressed (keyboard and focus for free).
// An option { value, label, disabled: true, note } the current selection cannot use is a disabled button (out of the tab
// order, unclickable) with its note printed beside the control (ruling P3-R12). A note on an enabled option is kept, unshown.

const isOff = (o) => o.disabled === true;

function checkDisabled(options) {
  const bad = options.find((o) => 'disabled' in o && typeof o.disabled !== 'boolean');
  if (bad) throw new RangeError(`choice: option ${bad.value} disabled must be true or false`);
  const silent = options.find((o) => isOff(o) && (typeof o.note !== 'string' || o.note.trim() === ''));
  if (silent) throw new RangeError(`choice: disabled option ${silent.value} needs a note saying why (P3-R12)`);
  if (options.every(isOff)) throw new RangeError('choice: at least one option must be enabled');
}

export function checkOptions(options, value) {
  if (!Array.isArray(options) || options.length === 0) throw new RangeError('choice: options must be a non-empty array');
  const values = options.map((o) => o?.value);
  if (new Set(values).size !== values.length) throw new RangeError(`choice: option values must be unique, got ${values.join(', ')}`);
  if (options.some((o) => typeof o.label !== 'string' || o.label === '')) throw new RangeError('choice: every option needs a label');
  if (!values.includes(value)) throw new RangeError(`choice: value ${value} is not one of ${values.join(', ')}`);
  checkDisabled(options);
  const chosen = options[values.indexOf(value)];
  if (isOff(chosen)) throw new RangeError(`choice: value ${value} is disabled (${chosen.note})`);
  return true;
}

// One line per distinct note of the disabled options, led by the labels it covers: "BF16, FP8: no settled BF16/FP8 figure".
export function noteLines(options) {
  const byNote = options.filter(isOff).reduce((acc, o) => acc.set(o.note, [...(acc.get(o.note) ?? []), o.label]), new Map());
  return [...byNote].map(([note, labels]) => `${labels.join(', ')}: ${note}`);
}

export const firstEnabled = (options) => options.find((o) => !isOff(o)).value;

function checkSameOptions(before, after) {
  const key = (list) => list.map((o) => `${String(o?.value)}\u0000${o?.label}`).join('\u0001');
  if (!Array.isArray(after) || key(after) !== key(before)) throw new RangeError('choice: update() takes the same option values and labels, in order; only disabled and note change');
}

const button = (text, attrs = {}) => {
  const b = document.createElement('button');
  b.type = 'button';
  b.textContent = text;
  Object.entries(attrs).forEach(([k, v]) => b.setAttribute(k, v));
  return b;
};

// Segmented mode: the options sit in a pill track with one thumb that slides to the selected option. The thumb is pure
// decoration: until it has been measured (or if the track has no layout) CSS fills the selected option itself.
const THUMB_ATTR = 'data-thumb';
function mountTrack(buttons) {
  const node = Object.assign(document.createElement('div'), { className: 'choice-track' });
  node.append(Object.assign(document.createElement('span'), { className: 'choice-thumb' }), ...buttons);
  let selected = null;
  const place = (target = selected) => {
    selected = target;
    if (!selected || selected.offsetWidth === 0) { node.removeAttribute(THUMB_ATTR); return; }
    node.style.setProperty('--thumb-x', `${selected.offsetLeft}px`);
    node.style.setProperty('--thumb-w', `${selected.offsetWidth}px`);
    node.setAttribute(THUMB_ATTR, '');
  };
  // Fonts loading or a resize change the option widths; the observer also fires once the track first gets a layout.
  const observer = typeof ResizeObserver === 'function' ? new ResizeObserver(() => place()) : null;
  observer?.observe(node);
  return { node, place, stop: () => observer?.disconnect() };
}

// The note <p class="toy-note" data-choice-note> is in the DOM only while an option is disabled, so a choice with none
// renders exactly as before; disabled buttons point to it with aria-describedby.
function mountNote(root, id) {
  const note = Object.assign(document.createElement('p'), { id: `${id}-note`, className: 'toy-note choice-note' });
  note.setAttribute('data-choice-note', '');
  const line = (t) => Object.assign(document.createElement('span'), { className: 'choice-note-line', textContent: t });
  return {
    id: note.id,
    paint(options) {
      const lines = noteLines(options);
      if (lines.length === 0) { note.remove(); return; }
      note.replaceChildren(...lines.map(line));
      if (note.parentNode !== root) root.append(note);
    },
  };
}

function paintDisabled(b, off, noteId) {
  if (!off) { ['disabled', 'aria-disabled', 'aria-describedby'].forEach((a) => b.removeAttribute(a)); return; }
  b.setAttribute('disabled', '');
  b.setAttribute('aria-disabled', 'true');
  b.setAttribute('aria-describedby', noteId);
}

// variant: 'segmented' (one of a few modes) or 'chips' (presets). onChange gets the option's value.
// set(value) of a disabled or unknown value throws a RangeError. update(options) re-disables options (same values and
// labels, in order); if the selected option becomes disabled, the first enabled option is selected and onChange fires.
export function mountChoice(root, { id, label, options, value, onChange, variant = 'segmented' }) {
  checkOptions(options, value);
  let current = value;
  let opts = options;
  const caption = Object.assign(document.createElement('span'), { id: `${id}-label`, className: 'choice-label', textContent: label });
  const buttons = options.map((o) => button(o.label, { class: 'choice-option', 'data-value': String(o.value) }));
  const track = variant === 'segmented' ? mountTrack(buttons) : null;
  const note = mountNote(root, id);
  const paint = () => {
    buttons.forEach((b, i) => { b.setAttribute('aria-pressed', String(opts[i].value === current)); paintDisabled(b, isOff(opts[i]), note.id); });
    track?.place(buttons[opts.findIndex((o) => o.value === current)]);
    note.paint(opts);
  };
  const choose = (next, notify) => {
    if (next === current) return;
    checkOptions(opts, next);
    current = next;
    paint();
    if (notify) onChange?.(current);
  };
  const update = (next) => {
    checkSameOptions(opts, next);
    checkDisabled(next);
    opts = next;
    const selected = opts.find((o) => o.value === current);
    if (isOff(selected)) choose(firstEnabled(opts), true);
    else paint();
  };
  const onClick = (event) => {
    const b = event.target.closest('.choice-option');
    if (b && root.contains(b) && !b.disabled) choose(opts[buttons.indexOf(b)].value, true);
  };
  root.id = id;
  root.className = `choice choice--${variant}`;
  root.setAttribute('role', 'group');
  root.setAttribute('aria-labelledby', caption.id);
  root.replaceChildren(caption, ...(track ? [track.node] : buttons));
  root.addEventListener('click', onClick);
  paint();
  return {
    get value() { return current; },
    set(next) { choose(next, true); },
    update,
    destroy() { root.removeEventListener('click', onClick); track?.stop(); root.replaceChildren(); },
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
