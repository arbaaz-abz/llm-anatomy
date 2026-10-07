// Zero-dependency static server for local development and visual QA.
import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, join, resolve, sep } from 'node:path';
import { pathToFileURL } from 'node:url';

const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml', '.png': 'image/png', '.gif': 'image/gif', '.woff2': 'font/woff2',
  '.md': 'text/markdown; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8', '.woff': 'font/woff', '.ttf': 'font/ttf', '.ico': 'image/x-icon',
};
const HOST = '127.0.0.1';

export const mimeFor = (file) => TYPES[extname(file)] ?? 'application/octet-stream';

// A directory served without its trailing slash gets the wrong base URI, so relative
// imports (e.g. concepts/<slug>.js) would resolve one level too high. Redirect instead.
export function redirectFor(urlPath, isDirectory) {
  const queryAt = urlPath.indexOf('?');
  const path = queryAt === -1 ? urlPath : urlPath.slice(0, queryAt);
  const query = queryAt === -1 ? '' : urlPath.slice(queryAt);
  if (!isDirectory || path.endsWith('/')) return null;
  // Collapse leading slashes: "//host/" as a Location would be a protocol-relative redirect.
  return `${path.replace(/^\/+/, '/')}/${query}`;
}

export function resolveSafe(root, urlPath) {
  let decoded;
  try {
    decoded = decodeURIComponent(urlPath.split('?')[0]);
  } catch {
    return null;
  }
  const full = resolve(root, `.${decoded}`);
  return full === root || full.startsWith(root + sep) ? full : null;
}

async function respond(root, req, res) {
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    res.writeHead(405, { allow: 'GET, HEAD' }).end('Method not allowed');
    return;
  }
  const target = resolveSafe(root, req.url ?? '/');
  if (!target) { res.writeHead(404).end('Not found'); return; }
  try {
    const info = await stat(target);
    const location = redirectFor(req.url ?? '/', info.isDirectory());
    if (location) { res.writeHead(301, { location }).end(); return; }
    const file = info.isDirectory() ? join(target, 'index.html') : target;
    const body = await readFile(file);
    res.writeHead(200, { 'content-type': mimeFor(file), 'cache-control': 'no-store' });
    res.end(req.method === 'HEAD' ? undefined : body);
  } catch (error) {
    if (error.code !== 'ENOENT') console.error(`serve: ${req.url}`, error);
    res.writeHead(404).end('Not found');
  }
}

export const createApp = (root) => createServer((req, res) => respond(root, req, res));

if (import.meta.url === pathToFileURL(process.argv[1] ?? '').href) {
  const root = resolve(process.argv[2] ?? '.');
  const port = Number(process.env.PORT ?? 8080);
  createApp(root).listen(port, HOST, () => {
    process.stdout.write(`Serving ${root} at http://localhost:${port}/\n`);
  });
}
