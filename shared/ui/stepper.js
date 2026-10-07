// The single animation control used on every page. The reducer is pure and tested;
// mountStepper only wires it to the DOM and requestAnimationFrame.

const SPEEDS = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2];
const TRANSITION_MS = 750; // at 1×; speed divides it
const DWELL_MS = 1800; // rest before auto-advancing, at 1×

const clampIndex = (value, last) => {
  const n = Math.trunc(Number(value));
  return Number.isFinite(n) ? Math.min(Math.max(n, 0), last) : 0;
};

export const initialStepperState = (count) => ({ index: 0, count, playing: false, speed: 1 });

export function stepperReducer(state, action) {
  const last = state.count - 1;
  switch (action.type) {
    case 'next': return { ...state, index: Math.min(state.index + 1, last) };
    case 'prev': return { ...state, index: Math.max(state.index - 1, 0), playing: false };
    case 'seek': return { ...state, index: clampIndex(action.index, last), playing: false };
    case 'play': return { ...state, playing: true, index: state.index === last ? 0 : state.index };
    case 'pause': return { ...state, playing: false };
    case 'toggle': return stepperReducer(state, { type: state.playing ? 'pause' : 'play' });
    case 'speed': return { ...state, speed: SPEEDS.includes(action.speed) ? action.speed : 1 };
    case 'stepDone':
      if (!state.playing) return state;
      return state.index === last ? { ...state, playing: false } : { ...state, index: state.index + 1 };
    default: throw new Error(`stepper: unknown action "${action.type}"`);
  }
}

export const easeInOut = (p) => (p < 0.5 ? 2 * p * p : 1 - (-2 * p + 2) ** 2 / 2);

const prefersReducedMotion = () =>
  globalThis.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;

const ICONS = {
  prev: '<svg viewBox="0 0 20 20" width="20" height="20" aria-hidden="true" focusable="false"><path d="M12.5 4 6.5 10l6 6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  next: '<svg viewBox="0 0 20 20" width="20" height="20" aria-hidden="true" focusable="false"><path d="m7.5 4 6 6-6 6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  play: '<svg viewBox="0 0 20 20" width="20" height="20" aria-hidden="true" focusable="false"><path d="M6 4.5v11l9-5.5z" fill="currentColor"/></svg>',
  pause: '<svg viewBox="0 0 20 20" width="20" height="20" aria-hidden="true" focusable="false"><path d="M5.5 4.5h3v11h-3zM11.5 4.5h3v11h-3z" fill="currentColor"/></svg>',
};

export function mountStepper(root, { steps, render, label = 'Animation', transitionMs = TRANSITION_MS, dwellMs = DWELL_MS }) {
  if (!steps?.length) throw new Error('mountStepper: steps must not be empty');
  let state = initialStepperState(steps.length);
  let frame = 0;
  let timer = 0;

  root.classList.add('stepper');
  root.tabIndex = 0;
  root.setAttribute('role', 'group');
  root.setAttribute('aria-roledescription', 'animation');
  root.setAttribute('aria-label', label);
  root.innerHTML = `
    <div class="stepper-stage"></div>
    <p class="stepper-caption" aria-live="polite"></p>
    <div class="stepper-bar" role="group" aria-label="Animation controls">
      <button type="button" class="stepper-icon" data-act="prev" aria-label="Previous step">${ICONS.prev}</button>
      <button type="button" class="stepper-play" data-act="toggle">${ICONS.play}<span class="stepper-play-label">Play</span></button>
      <button type="button" class="stepper-icon" data-act="next" aria-label="Next step">${ICONS.next}</button>
      <input type="range" min="0" max="${steps.length - 1}" step="1" value="0" aria-label="Step">
      <output class="stepper-count" aria-live="off"></output>
      <select aria-label="Speed">${SPEEDS.map((s) => `<option value="${s}"${s === 1 ? ' selected' : ''}>${s}×</option>`).join('')}</select>
    </div>`;
  const q = (sel) => root.querySelector(sel);
  const playButton = q('[data-act="toggle"]');
  const scrub = q('input[type="range"]');
  const counter = q('.stepper-count');
  const caption = q('.stepper-caption');
  const speedSelect = q('select');
  const stage = q('.stepper-stage');
  // The stage carries the frame it shows, so tests and tools read state through the public DOM.
  const paint = (index, progress) => {
    stage.dataset.step = String(index);
    stage.dataset.progress = String(progress);
    render(index, progress, stage);
  };

  const stop = () => { cancelAnimationFrame(frame); clearTimeout(timer); };

  function animate() {
    stop();
    const { index, speed } = state;
    const duration = prefersReducedMotion() ? 0 : transitionMs / speed;
    const start = performance.now();
    const tick = (now) => {
      const progress = duration === 0 ? 1 : Math.min((now - start) / duration, 1);
      paint(index, easeInOut(progress));
      if (progress < 1) {
        frame = requestAnimationFrame(tick);
      } else if (state.playing) {
        timer = setTimeout(() => dispatch({ type: 'stepDone' }), dwellMs / state.speed);
      }
    };
    frame = requestAnimationFrame(tick);
  }

  function sync() {
    playButton.innerHTML = `${state.playing ? ICONS.pause : ICONS.play}<span class="stepper-play-label">${state.playing ? 'Pause' : 'Play'}</span>`;
    scrub.value = String(state.index);
    counter.textContent = `${state.index + 1} / ${state.count}`;
    caption.textContent = steps[state.index].caption;
    scrub.setAttribute('aria-valuetext', `Step ${state.index + 1} of ${state.count}: ${steps[state.index].caption}`);
  }

  function dispatch(action) {
    const previous = state;
    state = stepperReducer(state, action);
    sync();
    if (state.index !== previous.index || (state.playing && !previous.playing)) animate();
    else if (!state.playing) clearTimeout(timer);
  }

  const onClick = (event) => {
    const button = event.target.closest('[data-act]');
    if (button && button.closest('.stepper-bar')) dispatch({ type: button.dataset.act });
  };
  const onScrub = () => dispatch({ type: 'seek', index: scrub.value });
  const onSpeed = () => dispatch({ type: 'speed', speed: Number(speedSelect.value) });
  const onKey = (event) => {
    // Only the stepper itself handles keys; buttons, the range and the select keep native behavior.
    if (event.target !== root || event.altKey || event.ctrlKey || event.metaKey) return;
    const keys = { ' ': 'toggle', ArrowRight: 'next', ArrowLeft: 'prev' };
    if (!keys[event.key]) return;
    event.preventDefault();
    dispatch({ type: keys[event.key] });
  };

  root.addEventListener('click', onClick);
  scrub.addEventListener('input', onScrub);
  speedSelect.addEventListener('change', onSpeed);
  root.addEventListener('keydown', onKey);

  sync();
  paint(0, 1); // complete at rest: the first frame is fully drawn before any interaction

  return {
    dispatch,
    getState: () => state,
    stage,
    destroy() {
      stop();
      root.removeEventListener('click', onClick);
      scrub.removeEventListener('input', onScrub);
      speedSelect.removeEventListener('change', onSpeed);
      root.removeEventListener('keydown', onKey);
      root.innerHTML = '';
    },
  };
}
