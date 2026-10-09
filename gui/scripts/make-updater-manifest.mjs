#!/usr/bin/env node
/**
 * Generate the auto-updater manifest (.updater/latest.json at the repo root)
 * after `pnpm tauri build`.
 *
 * The updater endpoint is the RAW master copy of this file — GitHub's
 * `releases/latest/download` shortcut skips pre-releases, so beta releases
 * would be invisible through it. Committing the manifest to master is the
 * publish step: installed apps poll it, verify the minisign signature, and
 * download the installer from the release tag named here.
 *
 * Usage: node scripts/make-updater-manifest.mjs <tag> [notes]
 *   <tag>   the git/GitHub release tag the assets live under, e.g. v0.2.1-beta.2
 *   [notes] optional one-line release notes shown by the updater
 */

import { readFileSync, writeFileSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { execSync } from 'child_process';

const __dirname = dirname(fileURLToPath(import.meta.url));
const guiDir = join(__dirname, '..');
const rootDir = join(guiDir, '..');

const tag = process.argv[2];
if (!tag) {
  console.error('Usage: node scripts/make-updater-manifest.mjs <release-tag> [notes]');
  process.exit(1);
}
const notes = process.argv[3] || `Nanna ${tag}`;

const conf = JSON.parse(readFileSync(join(guiDir, 'src-tauri', 'tauri.conf.json'), 'utf8'));
const version = conf.version;

// The repository the release assets live in: CI's own `GITHUB_REPOSITORY`,
// else the workspace manifest's `repository`. Never a literal — the repo moved
// from physics515 to basic-automation, and a hardcoded owner here wrote the old
// one into every manifest since, leaving each installed client's download one
// GitHub rename-redirect away from breaking.
function releaseRepo() {
  const fromCi = process.env.GITHUB_REPOSITORY;
  if (fromCi && /^[\w.-]+\/[\w.-]+$/.test(fromCi)) return fromCi;
  const cargo = readFileSync(join(rootDir, 'Cargo.toml'), 'utf8');
  const declared = /^repository\s*=\s*"https:\/\/github\.com\/([\w.-]+\/[\w.-]+?)\/?"/m.exec(cargo)?.[1];
  if (!declared) {
    console.error('No GitHub repository: set GITHUB_REPOSITORY or `repository` in Cargo.toml.');
    process.exit(1);
  }
  return declared;
}
const repo = releaseRepo();

// Same target-dir resolution as build-daemon.js: `<root>/target` is only the
// default and a global .cargo/config.toml target-dir moves it.
function resolveTargetDir() {
  try {
    const meta = execSync('cargo metadata --format-version 1 --no-deps', {
      cwd: rootDir,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    });
    const dir = JSON.parse(meta).target_directory;
    if (dir) return dir;
  } catch {
    // Fall through.
  }
  return join(rootDir, 'target');
}

// Every platform this can publish, and where its build leaves the signed
// installer. A platform is included only if its `.sig` is actually on disk —
// a local Windows-only build must still produce a valid Windows manifest, and
// inventing a Linux entry with no signature would ship a manifest that every
// Linux client rejects.
const PLATFORMS = [
  {
    key: 'windows-x86_64',
    asset: `Nanna_${version}_x64-setup.exe`,
    bundleDir: 'nsis',
  },
  {
    key: 'linux-x86_64',
    // The updater installs the AppImage, not the .deb — a .deb needs root and
    // cannot be swapped under a running app.
    asset: `Nanna_${version}_amd64.AppImage`,
    bundleDir: 'appimage',
  },
];

const platforms = {};
const missing = [];
for (const { key, asset, bundleDir } of PLATFORMS) {
  const sigPath = join(resolveTargetDir(), 'release', 'bundle', bundleDir, `${asset}.sig`);
  let signature;
  try {
    signature = readFileSync(sigPath, 'utf8').trim();
  } catch {
    missing.push(`${key} (no ${sigPath})`);
    continue;
  }
  // The signature embeds the file it was made for. A mismatch here means a
  // signature got paired with the wrong asset, which clients reject as
  // tampering — and the manifest would look perfectly fine to a reader.
  const signedFile = /file:(\S+)/.exec(Buffer.from(signature, 'base64').toString())?.[1];
  if (signedFile && signedFile !== asset) {
    console.error(`${key}: signature is for "${signedFile}", not "${asset}" — refusing to write.`);
    process.exit(1);
  }
  platforms[key] = {
    signature,
    url: `https://github.com/basic-automation/Nanna/releases/download/${tag}/${asset}`,
  };
}

if (Object.keys(platforms).length === 0) {
  console.error('No signed installer found for any platform; nothing to write.');
  console.error(missing.map((m) => `  missing: ${m}`).join('\n'));
  process.exit(1);
}
for (const m of missing) console.warn(`skipping ${m}`);

const manifest = {
  version,
  notes,
  pub_date: new Date().toISOString(),
  platforms,
};

const outDir = join(rootDir, '.updater');
mkdirSync(outDir, { recursive: true });
const outPath = join(outDir, 'latest.json');
writeFileSync(outPath, JSON.stringify(manifest, null, 2) + '\n');
console.log(`Wrote ${outPath} for ${version}:`);
for (const [key, p] of Object.entries(manifest.platforms)) {
  console.log(`  ${key} -> ${p.url}`);
}
