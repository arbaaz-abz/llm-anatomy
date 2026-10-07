export function sliderValue(raw, { values } = {}) {
  if (!values) return Number(raw);
  const i = Math.min(Math.max(Math.trunc(Number(raw)) || 0, 0), values.length - 1);
  return values[i];
}

export function nearestIndex(values, v) {
  let best = 0;
  values.forEach((candidate, i) => {
    if (Math.abs(candidate - v) < Math.abs(values[best] - v)) best = i;
  });
  return best;
}

export function mountSlider(root, { id, label, min, max, step = 1, value, values, unit = '', format = String, onInput }) {
  root.classList.add('slider');
  root.innerHTML = `<label></label><input type="range"><output></output>`;
  const [labelEl, input, output] = ['label', 'input', 'output'].map((s) => root.querySelector(s));
  labelEl.textContent = label;
  labelEl.htmlFor = id;
  input.id = id;
  output.htmlFor = id;
  const bounds = values
    ? { min: 0, max: values.length - 1, step: 1, value: nearestIndex(values, value) }
    : { min, max, step, value };
  Object.entries(bounds).forEach(([k, v]) => { input[k] = String(v); });

  const current = () => sliderValue(input.value, { values });
  const show = () => {
    const text = unit ? `${format(current())} ${unit}` : format(current());
    output.textContent = text;
    input.setAttribute('aria-valuetext', text);
  };
  const handle = () => { show(); onInput?.(current()); };
  input.addEventListener('input', handle);
  show();

  return {
    get value() { return current(); },
    set(v) { input.value = String(values ? nearestIndex(values, v) : v); handle(); },
    destroy() { input.removeEventListener('input', handle); root.innerHTML = ''; },
  };
}
