# Nanna v0.3.31-beta.40 — The Board Remembers

The last release taught the board to announce what happens to a card. This one makes it
**remember**. Every card and every post on a card's thread now becomes a memory, and when a
card is closed its whole history folds into one memory of what happened. Most of this release is
repair work, though: a long list of places where Nanna reported success for something that had
quietly failed, or kept running something it had said was stopped.

## What's Changed

**Cards and their threads are remembered.** Creating a card, posting on its thread, and each
change of state are written to memory as they happen, so an agent recalling "what did we decide
about the login bug?" finds the card and the conversation around it — not just chat.

**A closed card becomes one memory, not fifty.** When a card is done, its series of events is
folded into a single memory that keeps the turning points (who picked it up, where it got stuck,
how it ended) and drops the repetition. Reopen a card and close it again, and the memory is
rewritten to include the new chapter. The fold is deterministic and measured: a benchmark gate
now checks that it compresses at least as well as it does today and never loses a state change.

**The task router can see who is good at what.** The store now tallies, per member and per
label, how many verdicts passed and failed — and a verdict belongs to whoever held the card when
it was judged, not whoever holds it now. This is the number the router will use to pick a member
for a card.

## Fixes

**Things that said they worked and didn't:**
- Deleting or clearing memories could fail halfway and still report success; the "clear all"
  button did not reach storage at all. Deletes are now durable, or the app says they are not.
- Importing a settings file only rewrote the file: the running app ignored it until a restart,
  and the app's own copy lost every saved key. Imports now apply immediately and keep your keys.
- A provider error in the middle of a streamed reply (an overload, a rate limit) ended the reply
  as though the model had finished. It is now an error, retried or handed to the next model.
  The same for OpenRouter-style error messages inside a stream, which used to arrive as an empty
  answer.
- A Python script calling `sys.exit(1)` reported success.
- Edits to a scheduled job were lost on restart, and editing a job that did not exist "worked".
- A memory summary written by a local model could be cut off mid-generation, or include the
  model's private reasoning, and still be saved.
- A forked conversation came back empty after a restart, and a regenerated reply's old version
  came back too. Both are now stored properly.
- A crashed run whose conversation had been deleted was reported recovered — and its output
  thrown away.

**Things that kept running after they should have stopped:**
- A JavaScript tool stuck in a loop kept a CPU core busy for the life of the app after its
  timeout; it now stops. A tool returning a self-referencing object crashed the whole background
  service; it is now an error.
- Tools defined by a `tool.yaml` ignored their own timeout — the process and anything it started
  ran on. They are now stopped, children included.
- The browser tool could wait forever for a page that never loaded, left a tab open for every
  page it visited, and "waited" for an element by looking once.
- A helper agent that ran past its time limit, or was stopped, kept running in the background
  and made the app look permanently busy.

**Chat channels:**
- **Discord delivered no messages from people at all** — only from bots, which it then ignored —
  and was disconnected every forty seconds for not keeping its connection alive. Both fixed.
- Signal and WhatsApp connections were cut every two minutes, missing messages in between.
- Replies now thread under the message they answer on Telegram, Discord and Slack.
- Telegram error messages no longer include the bot's secret token.

**Smaller, but noticed:**
- Rate limits now wait as long as the provider says instead of guessing.
- A model-routing tier that names a different provider is skipped instead of failing every
  step.
- Image results from MCP servers no longer make the whole tool result disappear; MCP servers
  with many tools list all of them; a crashed MCP server fails its calls at once.
- Memory search snippets no longer crash on text with accented or wide characters.
- Several settings no longer freeze the app while the background service reloads, and adding
  a message no longer pauses every other conversation while it is written to disk.
- Text streamed while the app was catching up on a running reply is no longer lost.
- The **Recall Threshold** slider and **Enable Dreaming** switch are gone: neither ever did
  anything. Dreaming is always on, and recall keeps its tuned threshold.
- Upgrades that add a database column are now all-or-nothing, so an interrupted upgrade can no
  longer leave the database half-changed.

**Under the hood:** Tauri 2.12 (with its security fix), ARM64 builds compile again and their
vector code is now tested on every change, and the usual dependency sweep.

---

Updating from 0.3.29 or earlier? The
[0.3.30 release notes](https://github.com/basic-automation/Nanna/releases/tag/v0.3.30-beta.39) cover
the board's announcements and link back to earlier releases.
