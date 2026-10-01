# Nanna v0.3.34-beta.43 — Agents Work Their Cards

Until this release, assigning a card to an agent changed a field and nothing else. Now the
assignment is the start signal: the agent works the card straight away, writes its progress and
its result on the card's thread, and either closes the card or hands it back to the router with
the reason. There is still no board screen in the app; everything here works through the
daemon's connection and is what the board client will sit on.

## What's Changed

**Assigning a card to an agent starts the work.** Whether the router assigns it or you do, the
agent starts on the card at once, using the agent's own model list if it has one, otherwise
your chat models, in the card's own workspace. An agent works one card at a time. A card given
to a busy agent waits and starts as soon as the agent finishes. A card with a future date waits
for that date, and a card that waits on another card starts once that card is done. While it
works, the agent shows as busy.

**The work is on the card.** The agent's working notes appear as progress posts on the card,
and closing it leaves a verdict post: what the acceptance check said, or that no check ran. An
agent works only the card it was given and the sub-cards under it, never a card you add next to
it while it is busy.

**A card the agent cannot finish goes back to the router.** If the agent gives up, or runs out
of time, tokens or working models, the card is not cancelled. It returns to the router as an
open, unassigned card with the reason posted, and the router decides again (it may pick another
agent). If two attempts in a row fail, the router asks you what should change instead of
trying a third time.

**Agents ask you on the board.** When an agent working a card needs an answer, it creates a
question card for you, and its own card waits on that one. Nothing sits waiting for a reply.
Once you answer and complete the question card, the agent picks its card up again with your
answer in front of it. An agent cannot close its card on its own word while your answer is
still outstanding.

**Agents break work down on the board, and hand parts on.** When an agent splits its card
into sub-tasks, they appear as sub-cards under that card and the agent works through them
itself. It can also give a sub-card to another agent, who starts on it straight away while the
first agent leaves it alone; when the other agent finishes, the parent card carries on.
Sub-cards the router splits off for an agent start the same way. An agent can no longer clear
tasks in bulk.

**Stopping an agent pauses its card.** Cancelling an agent's run leaves the card with that
agent and says so on the thread, so nothing restarts it behind your back. Starting it again
picks up where it stopped. Work the daemon was doing when it shut down picks up again at the
next start.

## Fixes

- **Background work no longer reaches into your chats.** Work that ran in the background, outside
  any conversation, could have its "ask the user" questions and its to-do list land in whichever
  chat you used last. Each background run now has its own context, and a question from a board
  card becomes a question card instead.

## Under the hood

- New: starting, checking and stopping one card's run over the daemon's connection (`card_id`
  on `task.start_run`, `task.run_status` and `task.cancel_run`).
- Dependencies: the Tauri 2.12.1 patch line, tiptap 3.31.4, Lucide 1.49, Vitest 5.0.3. `libc`
  stays at 0.2.186 and `malachite-bigint` at 0.9.2 until RustPython releases past 0.5.0. Turso
  0.8.1 is out and is the next storage migration.

## Still open

- There is no board screen in the app yet.
- The router does not yet notice a card that sits in progress with nobody working it (for
  example one you paused), and the heartbeat is not a board card yet.
- Agents' capability tags do not yet adjust to how they actually perform.

# Also in this update

## Nanna v0.3.33-beta.42 — The Board Gets a Team

That release was prepared but never published, so it ships here.

### What's Changed

**The router takes up cards you create on the board.** When a workspace or global card is
created through the daemon's connection (not by the chat's own model), that board's Task
Management Agent reads it and decides: assign it, split it into sub-tasks, ask you a question,
or park it. It uses its own model list if one is set, otherwise your chat models. Cards the chat
makes for itself are left alone on purpose, so a question from the router can never stall a chat
mid-task.

**Answering the router's question sends the card back to it.** When the router asks you
something, it makes a card for you and the original card waits. Now, when you finish that card,
the router decides again with your answer in front of it.

**A recurring card goes back to the router each round.** A recurring card used to reopen
straight to whoever had it last time. Now it is released and the router places it again.

**The router fills in what you left blank.** When it assigns a card it can add labels and a
"done when" check. A check you wrote yourself is never replaced.

**Agents can join the board.** The daemon can now list, add, edit and remove board members. An
agent belongs to one workspace's board, to the global board, or to you personally (a personal
agent follows you to every board). You and each board's router cannot be removed. Every change
is announced to all connected clients. You can also set the router's own model list here.

### Fixes

- **Every workspace's board has a router**, including workspaces opened after boards were
  introduced (older ones get theirs at the next start).
- **A card's labels and tool list have limits**: 32 labels of up to 64 bytes each, and 64 tool
  names of up to 64 bytes each.

### Under the hood

- Dependencies: `softaes` 0.1.7, `playwright-rs` 0.19, `bigdecimal` 0.4.11, `tokio-rustls`
  0.26.6, and `undici` 8.11.2 (pulled in by Nuxt), fixing a WebSocket denial-of-service advisory
  ([GHSA-3wwx-pv8p-q78v](https://github.com/advisories/GHSA-3wwx-pv8p-q78v)).
- Built with the Rust nightly of 2026-09-28 (rustc 1.101.0).
