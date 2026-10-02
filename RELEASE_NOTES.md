# Nanna v0.3.35-beta.44 — The Board

Until this release the board existed only inside the daemon: cards, members, the router and
agents working cards were all real, but there was no screen to see them on. Now there is. Open
**Board** from the top of the left rail.

## What's Changed

**One line makes a card.** Type a card into the line at the top of the board. Tokens fill in
its fields, and everything else is the title:

- `#label` adds a label, `p1`–`p4` sets the priority, `@member` assigns it (`@me` is you).
- A date defers the card until that day: `today`, `tomorrow`, `friday`, `next friday`,
  `next week`, `in 3 days`, `march 30` or `2026-12-31`.
- A date in braces sets the deadline instead: `{friday}`, `{march 30}`.

For example, `Ship the fix #release p2 @builder tomorrow {friday}`. Anything you leave out, the
board's router fills in. If a token is wrong (an `@name` nobody on the board has, a deadline
that is not a date), the line says why and no card is made.

**The board.** Four columns: To do, Waiting (cards held up by another card, such as a question
for you), In progress and Done. Sub-cards sit inside their parent card, with a count, or on the
board as cards of their own. Filter by assignee, label, priority, or date (startable now,
deferred, overdue, no deadline). The board updates by itself as the router and agents work.

**A card's own view.** Click a card to see its thread: progress, questions and verdicts as
members post them, rendered as formatted text. From there you can post, reassign the card,
change its priority, date, deadline, labels and description, see what it waits on and what its
sub-cards are, add a sub-card, start, stop or resume an agent's work on it, and mark it done.
If its "done when" check fails, the card stays open and says why.

**Inbox and Upcoming.** Inbox lists the cards assigned to you that you can start now: no date,
or a date that has come. Upcoming lists the later ones by day. Both cover every board, and each
card says which board it is on.

**Members.** Add agents to a board, give each a list of models (best first), capabilities and
notes for the router, or make one your own so it follows you to every board. You can also set
the router's own model list here.

**Notifications.** You are notified when one of your cards' dates arrives, when its deadline
passes, and when the router or an agent hands you a card, such as a question about their work.

## Fixes

- **A date can be removed from a card.** Clearing a card's date, deadline or description used to
  be impossible; now it takes one click.
- **Listing a large folder gives the same answer on every computer.** When a folder had more
  entries than the project overview shows, which entries made the cut depended on the disk's
  filesystem. It now always keeps the first ones by name.

## Under the hood

- New daemon actions: `task.quick_add` (one line to one card, optionally as a sub-card) and
  `task.assigned` (a member's open cards on every board).
- The app now receives the daemon's card and roster change events, which it used to drop.
- RustPython 0.6. This lifts two version holds: `libc` (now 0.2.189) and `malachite-bigint`
  (now 0.12). Tauri's JavaScript packages now match the 2.12.1 Rust crates.
- `devalue` 5.9.4 (pulled in by Nuxt), fixing six advisories, three of them high
  ([GHSA-j22f-vq7h-c4qm](https://github.com/advisories/GHSA-j22f-vq7h-c4qm),
  [GHSA-mcm9-63f2-9j32](https://github.com/advisories/GHSA-mcm9-63f2-9j32),
  [GHSA-x5rw-q4pp-hg5g](https://github.com/advisories/GHSA-x5rw-q4pp-hg5g) and three lower).
  One `node-forge` advisory has no fix yet; it only affects Nuxt's development server, which is
  not part of the app.
- Built with the Rust nightly of 2026-10-02 (rustc 1.101.0).

## Still open

- The board shows the workspace selected at the top of the window, or the global board.
- The board does not yet follow the Figma design's final styling.
- The router does not yet notice a card that sits in progress with nobody working it (for
  example one you paused), and the heartbeat is not a board card yet.
- Chat is still there beside the board. Removing it is the next big step.

The previous release, [v0.3.34-beta.43](https://github.com/basic-automation/Nanna/releases/tag/v0.3.34-beta.43),
made agents start on the cards assigned to them.
