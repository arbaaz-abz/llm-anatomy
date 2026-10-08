// The round button that hides and shows the lesson list on wide screens (from 1440 px; CSS hides the button below that,
// where the Lessons menu does the job). The choice persists through safeStorage and degrades to "shown" without it.
import { safeStorage } from '../storage.js';
import { el } from './dom.js';

export const NAV_KEY = 'llm-anatomy:nav-collapsed';

export function createNavState(store = safeStorage()) {
  let collapsed = store.get(NAV_KEY, false) === true;
  return {
    collapsed: () => collapsed,
    toggle() {
      collapsed = !collapsed;
      store.set(NAV_KEY, collapsed);
      return collapsed;
    },
  };
}

const ICON = '<svg viewBox="0 0 20 20" width="18" height="18" aria-hidden="true" focusable="false"><rect x="2.75" y="3.75" width="14.5" height="12.5" rx="3" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M7.5 4v12" stroke="currentColor" stroke-width="1.6"/></svg>';

// onChange(collapsed) runs once now and after every click, so the caller can mirror the state onto the layout.
export function mountNavCollapse({ store, onChange, controls = 'lesson-nav' } = {}) {
  const state = createNavState(store);
  const button = el('button', { type: 'button', className: 'nav-collapse icon-button' });
  button.setAttribute('aria-controls', controls);
  const paint = () => {
    const collapsed = state.collapsed();
    const label = collapsed ? 'Show lessons' : 'Hide lessons';
    button.setAttribute('aria-label', label);
    button.title = label;
    button.setAttribute('aria-expanded', String(!collapsed));
    button.innerHTML = ICON;
    onChange?.(collapsed);
  };
  button.addEventListener('click', () => { state.toggle(); paint(); });
  paint();
  return button;
}
