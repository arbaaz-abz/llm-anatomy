// The light/dark button in every page header. The choice persists through safeStorage and is applied as
// html[data-theme]; before any choice the page follows the system preference (CSS) and the button reflects it.
import { safeStorage } from '../storage.js';
import { el } from './dom.js';

export const THEME_KEY = 'llm-anatomy:theme';
const THEMES = ['light', 'dark'];

export const resolveTheme = (stored, prefersDark) => (THEMES.includes(stored) ? stored : (prefersDark ? 'dark' : 'light'));

const systemPrefersDark = () => globalThis.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false;

export function createTheme({ store = safeStorage(), root = globalThis.document?.documentElement, prefersDark = systemPrefersDark } = {}) {
  const saved = store.get(THEME_KEY, null);
  let theme = resolveTheme(saved, prefersDark());
  if (THEMES.includes(saved)) root.setAttribute('data-theme', theme);
  return {
    current: () => theme,
    toggle() {
      theme = theme === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', theme);
      store.set(THEME_KEY, theme);
      return theme;
    },
  };
}

const ICONS = {
  light: '<svg viewBox="0 0 20 20" width="18" height="18" aria-hidden="true" focusable="false"><circle cx="10" cy="10" r="3.6" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="M10 2.5v2M10 15.5v2M2.5 10h2M15.5 10h2M4.7 4.7l1.4 1.4M13.9 13.9l1.4 1.4M4.7 15.3l1.4-1.4M13.9 6.1l1.4-1.4" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>',
  dark: '<svg viewBox="0 0 20 20" width="18" height="18" aria-hidden="true" focusable="false"><path d="M16.5 11.8A6.8 6.8 0 0 1 8.2 3.5a6.8 6.8 0 1 0 8.3 8.3z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/></svg>',
};

// A toggle button: aria-pressed is "dark mode is on"; the icon shows the mode a click switches to.
export function mountThemeToggle(options) {
  const theme = createTheme(options);
  const button = el('button', { type: 'button', className: 'theme-toggle icon-button', title: 'Switch between light and dark' });
  button.setAttribute('aria-label', 'Dark mode');
  const paint = () => {
    button.setAttribute('aria-pressed', String(theme.current() === 'dark'));
    button.innerHTML = ICONS[theme.current() === 'dark' ? 'light' : 'dark'];
  };
  button.addEventListener('click', () => { theme.toggle(); paint(); });
  paint();
  return button;
}
