// Derives the claude.ai Artifact variant of a page from the standalone source page.
import { readFile, writeFile, mkdir, readdir } from 'node:fs/promises';
import { basename, dirname, extname, join, relative, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

export function toArtifactHtml(html) {
  const head = html.match(/<head[^>]*>([\s\S]*?)<\/head>/i)?.[1];
  const bodyTag = html.match(/<body([^>]*)>/i);
  const body = html.match(/<body[^>]*>([\s\S]*?)<\/body>/i)?.[1];
  if (head == null || body == null) throw new Error('build-artifact: the page needs a <head> and a <body>');
  if (bodyTag[1].trim()) throw new Error('build-artifact: put attributes on #app, not <body>; the artifact skeleton owns <body>');
  const cleanHead = head
    .replace(/<meta\s+charset[^>]*>\s*/i, '')
    .replace(/<meta\s+name="viewport"[^>]*>\s*/i, '');
  const merged = `<meta name="llm-anatomy-target" content="artifact">\n${cleanHead.trim()}\n${body.trim()}\n`;
  return rebase(merged);
}

// Rebase only what the browser resolves against the page: href/src attribute values and the
// import map. Inline script strings (e.g. loadJSON('../data/x.json')) are relative to their own
// module, so rewriting them would break the artifact.
const REBASE_ATTR = /\b(href|src)=(["'])\.\.\/(shared|math|data)\//gi;
const REBASE_SPECIFIER = /(["'])\.\.\/(shared|math|data)\//g;
const rebaseAttrs = (text) => text.replace(REBASE_ATTR, '$1=$2./$3/');

function rebase(html) {
  let out = '';
  let last = 0;
  for (const m of html.matchAll(/(<script\b[^>]*>)([\s\S]*?)(<\/script>)/gi)) {
    const isImportMap = /\btype=["']importmap["']/i.test(m[1]);
    out += rebaseAttrs(html.slice(last, m.index)) + rebaseAttrs(m[1])
      + (isImportMap ? m[2].replace(REBASE_SPECIFIER, '$1./$2/') : m[2]) + m[3];
    last = m.index + m[0].length;
  }
  return out + rebaseAttrs(html.slice(last));
}

const WEB_TYPES = new Set(['.js', '.json', '.css', '.woff2', '.svg', '.png', '.html']);
const posix = (path) => path.split('\\').join('/');

// A published entry: a plain source path, or { from, contentType } for extensionless text files.
function publishEntry(file, repoRoot) {
  const ext = extname(file);
  const source = posix(relative(repoRoot, file));
  if (WEB_TYPES.has(ext)) return source;
  if (ext === '' && !basename(file).startsWith('.')) return { from: source, contentType: 'text/plain' };
  return null;
}

async function listFiles(dir) {
  const entries = await readdir(dir, { withFileTypes: true }).catch((error) => {
    if (error.code === 'ENOENT') return [];
    throw error;
  });
  const nested = await Promise.all(entries.map((e) => (e.isDirectory() ? listFiles(join(dir, e.name)) : [join(dir, e.name)])));
  return nested.flat();
}

export async function artifactFiles(pageDir, repoRoot) {
  const entries = [];
  for (const dir of ['shared', 'math', 'data']) {
    for (const file of await listFiles(join(repoRoot, dir))) entries.push([posix(relative(repoRoot, file)), file]);
  }
  // Everything else in the page's own folder (page-local modules, concepts/**) except the page itself.
  for (const file of await listFiles(pageDir)) {
    const path = posix(relative(pageDir, file));
    if (path !== 'index.html') entries.push([path, file]);
  }
  const published = entries
    .map(([path, file]) => [path, publishEntry(file, repoRoot)])
    .filter(([, entry]) => entry !== null);
  return Object.fromEntries(published);
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? '').href) {
  const [pageDirArg, outDirArg] = process.argv.slice(2);
  if (!pageDirArg || !outDirArg) {
    process.stderr.write('usage: node scripts/build-artifact.js <pageDir> <outDir>\n');
    process.exit(2);
  }
  const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
  const pageDir = resolve(pageDirArg);
  const outDir = resolve(outDirArg);
  await mkdir(outDir, { recursive: true });
  const html = await readFile(join(pageDir, 'index.html'), 'utf8');
  await writeFile(join(outDir, 'index.html'), toArtifactHtml(html));
  const files = await artifactFiles(pageDir, repoRoot);
  await writeFile(join(outDir, 'files.json'), `${JSON.stringify(files, null, 2)}\n`);
  process.stdout.write(`Wrote ${join(outDir, 'index.html')} and files.json (${Object.keys(files).length} files)\n`);
}
