# Nanna v0.3.40-beta.49 — Ninety fixes from a full audit

This release has no new screens. It is the result of a read-through of the whole daemon, the app,
the chat-app connections and the command line, looking for places where Nanna said one thing and
did another. About 90 of them are fixed here, nearly all with a test that checks the fix. The rest
are listed in the roadmap.

## What's Changed

### The board and recurring cards

- **A recurring card moves to its next round.** Completing one used to leave its due and defer
  dates where they were, so it came back already overdue. Both dates now move forward by the
  recurrence. A recurrence that cannot be read is refused when you save it, and you can clear one.
- **Cron schedules fire the way cron does.** With both a day of the month and a day of the week
  set, a schedule ran only on days that matched both. Standard cron runs on either, and so does
  Nanna now.
- **"Run now" behaves like a scheduled run.** It is recorded the same way, and a task can no longer
  run twice at once because you pressed it during a scheduled run.
- **The board view holds steady.** Switching boards kept showing the old board's cards for a
  moment, a slow reply could overwrite a newer one, and sub-cards hidden by a filter vanished with
  their parent. All three are fixed.
- **Smaller board fixes.** A bare "defer until" date now actually defers the agent's run. A card
  split between agents checks every assignee before creating any part. The router can no longer
  take back a card an agent has just started. "Done" checks a card in its own board's workspace.
- **A long-running card is remembered in full.** When a card closes, its history becomes one
  memory. For a card with more than 1,000 events, only the first 1,000 were used, and later
  activity was never added. Now the start and the end are both used, and the memory is rebuilt
  when the card grows.

### Memory

- **Workspace memories stay in their workspace.** A new memory could be merged into a similar
  memory from another workspace, or into a global one, which made private text visible
  everywhere. Merges now happen only within the same scope.
- **Searching a small workspace finds its memories.** A search looked at the 3×k best matches in
  the whole store and then filtered by workspace, so a small workspace often got nothing back. The
  search now widens until it has enough matches from that workspace or has looked at everything.
- **Deleted memories stay deleted.** A delete that happened while a memory's search index was
  being saved could be undone on the next restart. A memory whose save to disk failed was reported
  as saved. Both are fixed.
- **Edited text is not searched by its old meaning.** A search index computed from a memory's old
  text could be attached to its new text. It is now discarded, and the old index is also wiped
  from disk when a memory is rewritten.

### Chat and models

- **Token counts for OpenAI-compatible providers.** Streamed replies from OpenAI, OpenRouter and
  GitHub Models were recorded as using 0 tokens. They now report their real usage, including
  cached prompt tokens.
- **A failing model is held back, then tried again.** A model with a poor record was taken out of
  use for good. It now waits for a cooldown that grows with each failure and gets tried again
  afterwards.
- **Rate limits are respected.** When a provider says how long to wait, Nanna now reads that time
  in every format providers use. The wait applies to chat and to card runs. A very large value can
  no longer crash the daemon.
- **A cut-off reply is retried, not used.** If a connection dropped in the middle of a tool call,
  the half-written call could be repaired and run. It is now treated as a failed request and
  retried.
- **A stopped turn leaves a valid conversation.** When a model wrote a tool call as plain text,
  Nanna runs it as a real call. If the turn was stopped or ran out of budget first, that call was
  saved without a result, which some providers reject on the next message. It is now saved with a
  "skipped" result.
- **Stop stops a running check.** Pressing Stop while a task's completion check was running used to
  wait for the check to finish, which could take up to 10 minutes. The check is now ended at once.

### Tools and scripts

- **Tool scripts stay in their folders.** A skill could use `..` to read or write outside the
  folders it was allowed, and a tool you write could replace a built-in tool with the same name.
  Both are refused now.
- **Tools work inside the open workspace.** The PDF, image and audio tools, the python tool and
  your own tools resolved file paths against the daemon's folder instead of the open project. They
  now use the project. A python run also changed the daemon's working folder for good. It now puts
  it back when it finishes.
- **Browser tools close their tabs.** Every browser call opened a tab and never closed it, so a
  long session could fill memory until Chromium was killed. Each call now closes its tab, and a
  browser that crashed is started again on the next call instead of failing until a restart.
  Browser errors now report what actually went wrong, and a screenshot reports where it was saved.
- **Web fetches are limited.** A fetch read the whole download before applying its size limit, and
  a network error made the tool fail outright. Downloads now stop at 64 MiB, and a network error
  is reported to the model.

### Chat apps

- **Telegram's allowed users apply to webhooks too.** With a webhook URL set, the `allowed_users`
  list was not checked, so anyone who messaged the bot could use it. It is now checked on every
  message.
- **Telegram replies are no longer lost to formatting.** A reply with an unmatched `_` or `*`,
  such as a file name, was refused by Telegram and never arrived. It is now sent again as plain
  text.
- **Editing a Telegram message no longer runs it again.** Fixing a typo used to start a second
  turn with a duplicate reply.

- **Discord slash commands carry what you typed.** `/ask question:"…"` reached Nanna as just
  "ask", and Discord showed the command as failed. The question is now the message, and Discord
  shows "Working on it…" until the reply arrives.

### MCP servers

- **Your token goes only to its own server.** An older-style MCP server could tell Nanna to send
  messages, together with your saved token, to a different web address. Nanna now refuses that.
- **Connections recover.** If an MCP server ends the session, Nanna starts a new one and retries,
  instead of failing every later call. A dropped connection now fails a call at once instead of
  after 60 seconds.
- **Shutdown closes everything a server started.** Closing a local MCP server now also stops any
  programs that server started.

### The app

- **Creating a tool in the Tools page works.** The page sent the tool's code under the wrong field
  name, so new tools were never created, and edits saved only the description. A new test checks
  every call the app makes to the daemon for this kind of mistake.
- **Settings pages stay in sync.** A change made by the daemon or another window now appears in
  the open page. A save that fails now shows an error instead of looking successful.
- **Requests fail fast while the daemon is down.** After the daemon dropped its connection, every
  request from the app failed with an unclear error and was left waiting in memory; one could hang
  for 5 minutes. They are now refused at once until the app reconnects.
- **Statistics are accurate.** The tool and model statistics include failed calls. The "all tools"
  average counts every call equally. "Slowest" shows the slowest tool. The usage chart for N days
  covers N days, not N+1.

### Command line and service

- **`nanna daemon restart` waits for the old daemon to stop.** It used to wait half a second, find
  the old daemon still running, start nothing and report success. It now waits until the old one
  exits, and reports an error if it does not.
- **Addresses are honored.** `nanna daemon status --port` and `nanna export --daemon` reach a daemon
  that is not on the default port. `nanna mcp serve` waits for a long tool call to finish instead
  of failing after 30 seconds, and reconnects after the daemon restarts.
- **The installed service is the daemon you set up.** `nanna-daemon install` ignored your port,
  data folder and config file. It also reported success when `systemctl` or `launchctl` refused
  the install. Both are fixed.
- **Settings that fail to save say so.** A setting that could not be written to the config file
  was reported as saved and then lost on restart. The reply now says it applies until restart
  only. `nanna doctor` now fails a config file that cannot be read.
- **A broken config file is not overwritten.** `nanna credentials import`, `refresh`, `setup`
  and `clear` replaced a `config.toml` with a typo in it with the default settings. They now leave
  it alone and say so, and every other command except `nanna doctor` stops with the parse error
  instead of running on the defaults. `nanna chat` exits at the end of input instead of spinning, and
  `nanna config` shows the file it actually read.
- **Saved keys are not lost.** Running a command such as `nanna mcp secret set` while the daemon
  was running could lose a saved key. Writes to the key file now wait for each other.

## Under the hood

- **Dependencies:** the `toml` crates and the app's Tauri plugins, `marked`, Lucide icons,
  Playwright and happy-dom are updated. Sixteen dependencies that nothing used were removed, and a
  new check (`cargo shear`) keeps unused ones out. Held back: `rustpython-ruff` 0.16.10 and
  `rten` 0.27 (they do not build), Tauri 2.12.2 (it would downgrade parts of the Windows app),
  Nuxt 4.6 (its Windows build fails) and TypeScript 7 (only development builds so far).
- **Status endpoint:** `/status` now reports the number of connected clients and the most recent
  error. Both used to be empty.
- **Sessions:** resuming a chat and the REST API read a conversation's latest messages, not its
  first ones. Session lists are sorted by time even when timestamps were saved in different
  formats.

## Still open

- While a python run with a working folder is in progress, every part of the daemon sees that
  folder.
- Two choices are the owner's to make: whether cron should catch up on runs missed while the
  computer was off, and whether board dates should use local time instead of UTC.
- The router still has no heartbeat card, and Nanna does not yet run models itself (that waits
  on the Mummu model runner).
