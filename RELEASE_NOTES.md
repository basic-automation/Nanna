# Nanna v0.3.43-beta.52 — Memories that stay put, tool calls that are never half-run, and channels that come back

This release fixes more than thirty bugs found by reviewing the memory, chat, board, channel, MCP,
browser and provider code line by line. The most serious could lose a memory's text, run a tool call
that had been cut off half-way, or leave a chat channel deaf until Nanna restarted.

## Memory

- **A memory can no longer lose its text during consolidation.** If saving a merged memory failed,
  Nanna still went on to delete the memory it had merged in, so after a restart that text existed
  nowhere. The merge is now undone and both memories are kept for the next pass.
- **Your most important facts are no longer summarised away.** Facts you marked as helpful could be
  rewritten into a summary and the original deleted, while less important facts were protected.
- **Only one consolidation runs at a time.** Pressing "Consolidate" while a scheduled one was running
  summarised the same memories twice and could remove more than the safety limit allows.
- Summaries now keep their search index across restarts, and a summary is no longer thrown away
  when you switch embedding providers in the middle of a pass.
- A memory deleted during consolidation no longer comes back after a restart.
- Recalling a memory now strengthens it as of the moment you recalled it, not the moment the next
  consolidation ran.
- A scheduled consolidation that failed everywhere is now reported as failed.

## Chat and the agent

- **A tool call cut off by the model's output limit is never run.** A reply that ran out of room
  in the middle of, say, writing a file had its half-written arguments patched up and executed —
  writing half a file. It is now retried instead.
- **Messages you send while the agent is working are no longer lost** if saving them fails, and a
  message sent just as a reply finishes now starts its own reply instead of waiting for your next
  one.
- Sub-agents now really use the models you configured for them; every attempt used to run on your
  main chat model.
- A paused chat waiting for a provider to come back no longer hammers a provider that is down.
- Provider checks now ask the model the run actually fell back to.
- A reply that stopped before the provider said it was done is reported as cut off, not complete.
- A rate limit returned by OpenRouter with a success status is recognised as a rate limit.
- A model that writes long runs of digits (such as a null hash) is no longer stopped as stuck.
- `gpt-oss:20b` and other tagged names go to Ollama, not OpenAI.
- Tool calls from servers that omit an index are no longer dropped.

## Chat channels

- **Telegram, Discord and Slack reconnect after a long outage.** After about eight minutes offline
  a channel gave up for good until Nanna restarted. It now keeps retrying every two minutes.
- A Discord bot set up without a required permission no longer reconnects twice a second (which can
  get its token reset); it stops and says why.
- A Slack connection that silently died is noticed and reconnected.
- Slack no longer answers its own replies, join messages and edits, or answers one @-mention twice.
- Telegram reactions with custom or paid emoji no longer block later updates (`nanna serve`).

## The board

- A card deferred to a time of day now starts that day; it used to never start.
- Cancelling or deleting a sub-card wakes the card that was waiting on it.
- A rate limit while working on a sub-card no longer cancels the sub-cards; the card waits it out.

## MCP servers and the browser

- **MCP servers start together**, and each one's tools are available as soon as it is up. One slow
  server used to hold back every other server's tools for minutes.
- A question an MCP server asks you is no longer cut off after about 75 seconds.
- Two servers whose tools end up with the same name no longer replace each other's tools.
- A server that is slow to start for the first time now connects instead of failing.
- MCP servers no longer load the app's own bundled libraries instead of your system's (AppImage).
- The browser tools report `undefined` results correctly, say when a selector matches nothing, close
  tabs whose page never finished loading, and no longer leave profile folders behind.

## Settings and storage

- Environment variables such as `PORT` are no longer written into your config file, and changing a
  setting the environment controls now says so instead of silently reverting.
- Reopening a project folder that was unavailable at startup keeps its sessions and tasks.
- Tool statistics remember when each tool last ran across restarts.

## Still open

Recorded in the roadmap: Discord and Slack have no sender allowlist yet, a card reassigned while
running can lose its new assignee, and a few smaller provider and storage issues.
