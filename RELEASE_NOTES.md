# Nanna v0.3.36-beta.45 — Cards don't get stuck

The board shipped in the last release. This one is about what happens when the work on a card
goes wrong: a card an agent stopped working on, a model provider that rejects the key, a
provider that says "slow down". Each now ends somewhere you can see, instead of in a card that
sits there.

## What's Changed

**A card nobody is working goes back to the router.** If an agent's card has been in progress
for half an hour with nothing working on it, and you did not stop it yourself, the board's
router takes it back. It says so on the card's thread, then decides again who should do it. A
card that keeps coming back this way ends with a question to you rather than going round
forever, the same as a card agents keep handing back. A card you stopped stays with its agent
until you resume or reassign it.

**A rejected API key becomes a question for you.** When the model provider refuses an agent's
run outright (an invalid key, an account out of credit, no key set), the card no longer fails or
goes to another agent on the same broken provider. You get a question card that names the error
and says what to fix. The work card waits for it, keeping its agent. Mark the question done once
it is fixed, and the agent starts again.

**A rate-limited run waits and tries again.** When the provider says too many requests, the card
stays with its agent and says when it will try again (the provider's own wait, else five
minutes). That does not count as a failure.

**Choose where Nanna keeps its data.** Settings → Data → **Data location** shows the folder the
daemon keeps its database, memories and logs in, and lets you choose another one or go back to
the default. The daemon reads this when it starts. Until it restarts, the page says it is still
using the old folder, and it never moves your data: copy it there first if you want to keep it.

## Fixes

- **A board shows its own cards.** With the app open twice on different workspaces, a board
  could list, and add cards to, whichever workspace the *other* window last picked. Each board
  now names its own workspace.

## Under the hood

- **turso 0.8.1** (from 0.7.2), built without its full-text search feature. That feature does
  not compile on our toolchain ([turso#9463](https://github.com/tursodatabase/turso/issues/9463)),
  and Nanna does not use it. Leaving it out removes 34 crates from the build, among them a C
  compression library and an `lru` version with a soundness advisory (RUSTSEC-2026-0253). Checked
  by starting the release daemon against a copy of a real 101 MB database.
- **25 app commands nothing called are gone**, among them commands that could write workspace
  files and tool code. The app no longer links Nanna's scripting engines: its dependency graph
  went from 862 crates to 740.
- The daemon's status now reports the data folder it is using, and `task.list` /
  `task.quick_add` accept a `workspace_id`.
- The unused `[memory] extraction_model` setting is removed. Old config files that still have it
  load as before.
- `uuid` 1.27 and `@lucide/vue` 1.51, plus routine lockfile updates. TypeScript 7 is still on
  hold until `vue-tsc` supports it.

## Still open

- Chat is still there beside the board. Removing it is the next big step.
- The heartbeat is not a board card yet, and the board does not yet follow the Figma design's
  final styling.
- There is no way yet to back up or export Nanna's data from the app.

The previous release, [v0.3.35-beta.44](https://github.com/basic-automation/Nanna/releases/tag/v0.3.35-beta.44),
added the board.
