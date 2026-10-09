// prefill-decode toy (stub until B6): mount(host, ctx) → destroy.
export function mount(host) {
  return () => host.replaceChildren();
}
