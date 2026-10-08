// DOM helpers for the multimodal toy: controls that follow the state object. A control the user moves already
// matches the state; one a preset chip changed is mounted again with the new value (the shared controls have no silent setter).
import { mountChoice } from '@shared/ui/choice.js';
import { mountSlider } from '@shared/ui/slider.js';

export function syncedChoice(host, { id, label, options, variant = 'segmented', read, onChange }) {
  let control = null;
  const mount = (value) => {
    control?.destroy();
    control = mountChoice(host, { id, label, options, variant, value, onChange });
  };
  return {
    sync(state) {
      const value = read(state);
      if (!control || control.value !== value) mount(value);
    },
    destroy() { control?.destroy(); },
  };
}

// `valuesOf(state)` is the list of stops (it changes with patch × merge); the slider is mounted again when it does.
export function syncedSlider(host, { id, label, unit, valuesOf, read, format, onInput }) {
  let control = null;
  let key = '';
  const mount = (state) => {
    control?.destroy();
    const values = valuesOf(state);
    key = `${values.length}:${values[0]}:${values[values.length - 1]}`;
    control = mountSlider(host, { id, label, unit, values, value: read(state), format, onInput });
  };
  return {
    sync(state) {
      const values = valuesOf(state);
      const next = `${values.length}:${values[0]}:${values[values.length - 1]}`;
      if (!control || next !== key || control.value !== read(state)) mount(state);
    },
    destroy() { control?.destroy(); },
  };
}
