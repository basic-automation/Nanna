# Nanna v0.3.42-beta.51 — Web pages can no longer reach your agent, and a long list of quiet fixes

This release closes a security hole: any web page you visited could connect to Nanna's background
service on your computer and run commands through it. It also fixes more than forty bugs found by
reviewing the board, memory, scheduler, file tools and settings code line by line. Most of them
gave a wrong answer without any error.

## Security

**Web pages can no longer control Nanna.** The background service listens only on your own
computer, but browsers allow any page to open a connection to it. A malicious page could have run
shell commands, read your settings and shut the service down. Connections from web pages are now
refused; the Nanna app and the `nanna` command line are unaffected.

**Settings no longer leak your API keys.** Reading or exporting the settings through the background
service returned your API keys and sign-in token in plain text. They are now left out.

**`nanna server` no longer answers web pages.** Its chat API, which can use tools, accepted requests
from any web site. It now accepts only requests from programs on your computer.

**Your rules for the agent are enforced everywhere.** A rule such as "never delete anything under
tests/" was not enforced for shell commands, and the agent could overwrite the file that holds those
rules. Both are fixed, and the agent can no longer erase the records its safety checks rely on.

## The board

- A card that the router split off, or that an agent handed on, is no longer stranded with nobody
  on it when its run gives up or stalls.
- A card no longer goes back to the router when its own agent asked you a question; the answer goes
  to that agent.
- The limit on retries before Nanna asks you now holds, however busy the card is.
- Text you are typing in a card's description or labels is no longer wiped when the board refreshes.
- A card opened from your Inbox can only be given to someone on that card's own board.
- Deleting an agent releases its open cards back to the router instead of leaving them pointing at
  an agent that is gone, and you can now unassign a card.
- Renaming, re-dating or deleting a card now updates every open view right away.
- Recurring cards open on the right day: not again on the day you finished them, and on the current
  round after the computer was off.
- Deadlines written with a time zone are no longer marked overdue before they pass.
- A dependency that could never be finished is now refused, dates that are not real dates are
  refused, and reopening or cancelling a card updates everything that follows it.

## Memory

- Memory now belongs to the right project. An agent working on one project's card no longer reads
  or writes another project's memories.
- Consolidation can no longer delete an unrelated memory when it re-folds a card's history.
- A fact you stated is no longer merged into an observation and reworded later.
- Saving a memory no longer fails when the embedding service is briefly unavailable.
- Recalling memories no longer strengthens ones that were never shown.

## Scheduled jobs

- A daily job that came due while Nanna was busy runs as soon as it is free, instead of skipping
  the day.
- Turning a job, or the scheduler, back on no longer runs it immediately off schedule.
- "Run now" no longer freezes Settings and reminders while the job runs.
- Editing a job now saves its new prompt.

## Files and tools

- Editing a file with Windows line endings no longer mixes in Unix line endings.
- Searches find matches at the end of lines in Windows-style files, and every match is marked.
- `find_files` patterns with folders (`src/**/*.rs`) work without an explicit path.
- PDF, image, audio and OCR tools can read files in projects outside your home folder (on a second
  drive, for example). Existing installs are updated automatically unless you changed these
  permissions yourself.
- Paths with `~` or an apostrophe no longer break the syntax checks, and `.env.local` beside `.env`
  is no longer refused as a copy.
- File history no longer fills up with temporary drafts, every saved checkpoint can be listed, and
  restoring over a file too large for history to keep is refused instead of losing it.
- Web pages in other character sets (Latin-1, Shift-JIS and others) are fetched with their text
  intact.
- The browser tools keep working after their first use. Before, every browser call after the
  first failed until Nanna was restarted.
- An MCP server that sends an enormous message can no longer use up Nanna's memory.

## Conversations with the model

- Requests sized for a small local model now count the project context, the tool list and the real
  size of attached images, so a request that looked small enough no longer overflows the model's
  window.
- A file the agent re-read to edit is no longer swapped for a "seen before" note on the next step.
- Empty-looking messages that Anthropic rejects are no longer sent, so a session that stored one no
  longer fails on every later turn.
- When a request is cut to fit, the agent is told; a message too long for the summarizer is no
  longer dropped after only its beginning was read.

## Settings

- A mistake in `config.toml` can no longer cause Nanna to overwrite the whole file with defaults.
  Settings are also saved safely if the computer loses power.
- Your custom system prompt and agent name are now used.
- "Remember my choice" in the close dialog is remembered after a restart.
- Saving one API key no longer copies keys from environment variables into your keyring.
- Changing two settings in quick succession no longer undoes the first.
- The Tools and Memory pages no longer show stale results after a quick switch, the scheduler switch
  shows its real state when a change is refused, and a failed settings import now says so.

## Command line

- `nanna config` no longer prints your API keys.
- `nanna daemon stop` waits until the daemon has exited, so `stop` followed by `start` works, and
  `nanna daemon start` reports a daemon that fails to start instead of claiming success.
- `--config` is honoured by `nanna credentials`, `init` and `status`.
- First-run key setup saves only your config file and the key you typed, not values from
  environment variables.

## Under the hood

- **Dependencies:** Tauri 2.12.3 and its plugins, Nuxt 4.6.1 (it now builds on Windows), uuid
  1.28, vue-router 5.4.0.
- **Held back:** `rten` 0.27 (waiting on `ocrs`) and the newer rustpython parser crates.
- The test suite no longer leaves temporary databases in `/tmp`.

## Still open

- Channel setup (Telegram, Discord and others) still copies environment-variable secrets into the
  keyring. Channels are due to be removed.
- `nanna daemon stop` on macOS and Windows cannot yet confirm that the recorded process is the
  daemon before stopping it.
- MCP: a question a server asks you times out after about a minute, and one slow server delays
  the tools of the others while Nanna starts.
