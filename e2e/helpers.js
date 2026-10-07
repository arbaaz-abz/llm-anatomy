// Collects console errors and uncaught page errors. Warnings are excluded, so the
// expected "did not load; showing coming soon" warning is never counted.
export function collectConsoleErrors(page) {
  const errors = [];
  page.on('console', (msg) => { if (msg.type() === 'error') errors.push(msg.text()); });
  page.on('pageerror', (err) => errors.push(err.message));
  return errors;
}

export const TRACKS = [
  { id: 'architecture', count: 11 },
  { id: 'training', count: 14 },
  { id: 'serving', count: 9 },
];
