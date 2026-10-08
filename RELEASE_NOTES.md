# Nanna v0.3.39-beta.48 — A faster Memory page, and a board race closed

This release has no new screens. The Memory page loads faster on large stores, a timing gap that
could take a card away from an agent that had just started on it is closed, and a lot of code that
nothing used is gone.

## What's Changed

**The Memory page loads faster on a large store.** Opening it asks the daemon for every memory at
once. That reply used to be built twice in memory before it was sent: once as a tree of small
pieces, then a full copy of that tree. It is now written straight out once. On a store of 3,927
memories (a 15.8 MB reply) each request took about 79 ms before and about 67 ms now. What the app
receives is unchanged, byte for byte. Memory use after these requests is about the same as before.

**A card you start by hand can no longer be taken away at the same moment.** Every 5 minutes the
daemon looks for cards that an agent holds but nothing has worked on for 30 minutes, and hands
them back to the board's router. If you pressed Start on such a card at the exact moment of that
check, the card could be handed back while its new run was already working on it. Both now take
turns: either your Start comes first and the card stays with its agent, or the hand-back comes
first and the late Start is refused with "no longer assigned". A card someone commented on or
edited in that moment is also kept.

**Response-time histograms for monitoring.** `/metrics` on the health port now includes how long
each tool call and each model request took, in the standard histogram format. A Prometheus or
Grafana setup can chart percentiles over time from it. Before, it showed only a recent 95th
percentile.

**Less dead code.** About 3,200 lines that nothing called were removed:

- an unused second JavaScript engine (V8, through Deno). It was never compiled into the app, so
  the app does not change, but 83 packages leave the dependency list;
- 17 old built-in tools. The tools you use are the bundled skills, which are unchanged.

## Under the hood

- **Dependencies:** the turso database moves from 0.8.1 to 0.8.2, plus 37 other compatible
  updates. Three updates were held back because they do not build: `rustpython-ruff` 0.16.10,
  `rten` 0.27, and Nuxt 4.6.0, whose static build fails on Windows. The GUI stays on Nuxt 4.5.2.
- **Security audit:** four new advisories against `simple-git` are exempted, with the reasons
  recorded. It is used only by Nuxt's developer tools, which are switched off in the app you
  install. The fixed version cannot be used yet because it would break the development server.
- **Checked against outside implementations:** the MCP compatibility tests now run against the
  latest official Rust SDK (3.5.1) and TypeScript SDK (2.3.1). All 11 pass.

## Still open

- The built-in offline text recognition (OCR) is compiled into the app, but nothing calls it.
  Whether to connect it or remove it is the owner's call.
- The router still has no heartbeat card.
- Nanna does not yet run models itself. That waits on the Mummu model runner, which first needs to
  move to burn 0.22.0, released 2026-10-06.
