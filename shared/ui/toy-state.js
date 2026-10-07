// One frozen state object and one render() per toy (storyboard toys: "one state object, one render()").
// destroy() stops renders for good, so a toy left mid-interaction can never paint a detached page.
export function createToyState(initial, render) {
  if (typeof render !== 'function') throw new TypeError('createToyState: render must be a function');
  let state = Object.freeze({ ...initial });
  let alive = true;
  const set = (patch) => {
    if (!alive) return state;
    const next = typeof patch === 'function' ? patch(state) : patch;
    state = Object.freeze({ ...state, ...next });
    render(state);
    return state;
  };
  render(state);
  return {
    get: () => state,
    set,
    destroy() { alive = false; },
  };
}
