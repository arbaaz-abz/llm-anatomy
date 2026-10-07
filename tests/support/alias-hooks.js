// Test-only: mirrors the pages' import map ("@shared/" → shared/, "@math/" → math/) for node:test.
// Synchronous on purpose: module.registerHooks needs a sync hook; module.register accepts it too.
const ROOT = new URL('../../', import.meta.url);
const MAP = [['@shared/', new URL('shared/', ROOT).href], ['@math/', new URL('math/', ROOT).href]];

export function resolve(specifier, context, nextResolve) {
  const hit = MAP.find(([prefix]) => specifier.startsWith(prefix));
  return nextResolve(hit ? hit[1] + specifier.slice(hit[0].length) : specifier, context);
}
