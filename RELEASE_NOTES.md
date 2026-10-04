# Nanna v0.3.37-beta.46 — Fewer ways to crash, faster recall

This release has no new screens. It removes several ways the daemon could take itself down, makes
memory search faster, and fixes where future updates are downloaded from.

## What's Changed

**Memory search is about twice as fast.** Nanna was using a memory allocator it inherited from
its database library rather than one anyone chose. Measured against the system allocator, the
search that runs on every recall and every new memory took 11.2 ms at 50,000 memories; it now
takes 4.7 ms. Loading memories at startup got slower (121 ms → 181 ms at 50,000, once per
launch). One C library is gone from the build. After startup, the daemon now returns the memory
that loading used to the system: on a real 3,730-memory store it settles at about 213 MB, down
from about 237 MB.

**A tool can no longer take the daemon down with it.** When a tool script reads a file, runs a
command, fetches a URL or calls a service, Nanna starts a small worker for that call. If the
worker could not start (for example because the system had run out of file handles), the whole
daemon used to stop. Now that one call fails with an error saying why, and everything else keeps
running. Similarly, if you edit the `discover_tools` tool so that it no longer parses, startup now
skips that tool with a warning instead of failing.

**Updates download from the repository's current home.** Since the project moved to
`basic-automation/Nanna`, each release's update manifest still pointed installers at the old
address. Downloads only worked because GitHub redirects the old address. The manifest generator
now uses the repository's actual name, and new installs check for updates there directly.
Existing installs keep working as before.

## Fixes

- **GPU vector search handles large stores.** Searching more vectors than the graphics card
  accepts in one batch was a hard crash. The search now splits the work into batches the card
  accepts. (Nanna does not use the GPU path today. This keeps it safe for when it does.)
- **macOS service install escapes its settings.** An install path or argument containing `&` or
  `<` produced a launchd file macOS refused to load. Values are now escaped, and that is tested on
  every platform.

## Under the hood

- The GPU benchmarks reported speed ratios 10^18 times too large, so they could never report a GPU
  win. Fixed. With correct numbers, SIMD wins across the tested range (4.3× faster at 10,000
  vectors).
- New tests: an MCP stream event split at every byte offset, the GPU batch limits (run on an RTX
  4070 Ti SUPER), the launchd escaping, and the tool-runtime helper.
- `tokio` 1.53.2, `mio` 1.2.4, `async-recursion` 1.2 and `@lucide/vue` 1.52. TypeScript 7 is still
  on hold until `vue-tsc` supports it, and `rten` 0.27 until `ocrs` supports it.

## Still open

- Chat is still there beside the board. Removing it is the next big step.
- Whether to keep the system allocator long term depends on how the daemon's memory use looks over
  a full day of running.
- The heartbeat is not a board card yet, and the board does not yet follow the Figma design's
  final styling.
