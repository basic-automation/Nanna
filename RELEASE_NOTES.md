# Nanna v0.3.33-beta.42 — The Board Gets a Team

The Task Management Agent is now switched on for cards you put on the board, and the board can
finally hold more than just you. Until this release the only member a card could go to was you:
nothing could add an agent to the roster. Now agents can be added, the router hears about new
cards, answered questions and recurring cards, and every workspace's board has its own router.
There is still no board screen in the app; everything here works through the daemon's
connection and is the groundwork the board client will sit on.

## What's Changed

**The router takes up cards you create on the board.** When a workspace or global card is
created through the daemon's connection (not by the chat's own model), that board's Task
Management Agent reads it and decides: assign it, split it into sub-tasks, ask you a question,
or park it. It uses its own model list if one is set, otherwise your chat models. Cards the chat makes for itself are left
alone on purpose, so a question from the router can never stall a chat mid-task. (The app's
current chat checklist makes chat-only cards, which are never routed.)

**Answering the router's question sends the card back to it.** When the router asks you
something, it makes a card for you and the original card waits. Before this release, finishing
that card unblocked the original but nobody picked it up again. Now the router decides again,
with your answer in front of it.

**A recurring card goes back to the router each round.** A recurring card used to reopen
straight to whoever had it last time. Now it is released and the router places it again.

**The router fills in what you left blank.** When it assigns a card it can add labels and a
"done when" check. A check you wrote yourself is never replaced.

**Agents can join the board.** The daemon can now list, add, edit and remove board members. An
agent belongs to one workspace's board, to the global board, or to you personally (a personal
agent follows you to every board). You and each board's router cannot be removed. Every change
is announced to all connected clients, so a second window sees a new agent straight away. You
can also set the router's own model list here; before, there was nowhere to set it.

## Fixes

- **Every workspace's board has a router.** Only workspaces that existed when the board was
  introduced got one. Any workspace opened since then had none, so its router's settings had
  nowhere to live. New workspaces now get one when opened, and older ones get theirs at the next
  start.
- **A card's labels and tool list have limits.** A card could carry any number of labels or tool
  names of any length. Now the limit is 32 labels of up to 64 bytes each, and 64 tool names of up
  to 64 bytes each (the longest tool name any model provider accepts).

## Under the hood

- Dependencies: `softaes` 0.1.7, plus `playwright-rs` 0.19, `bigdecimal` 0.4.11 and
  `tokio-rustls` 0.26.6. TypeScript 7 is still blocked (`vue-tsc` 3.3.11 has no support).
- Built with the Rust nightly of 2026-09-28 (rustc 1.101.0).
- `undici` 8.11.2 (pulled in by Nuxt), fixing a WebSocket denial-of-service advisory
  ([GHSA-3wwx-pv8p-q78v](https://github.com/advisories/GHSA-3wwx-pv8p-q78v)).

## Still open

- Assigning a card to an agent does not start work on it yet. That is the next stage.
- There is no board screen in the app yet.
- The router does not react yet to a failed acceptance check or to a card that has stalled.
  Both need agents that actually run cards first.
