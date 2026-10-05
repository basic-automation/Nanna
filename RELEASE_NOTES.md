# Nanna v0.3.38-beta.47 — Steadier memory use, fewer stuck cards

This release has no new screens. The daemon stops growing after large requests. Cards on the
board no longer get stuck in ways that need a restart. When the daemon dies unexpectedly, you
can now see how it died.

## What's Changed

**The daemon no longer grows with use.** Opening the Memory page asks the daemon for every
memory at once. On a real store of 3,730 memories that reply is 14 MB, and building it used to
leave the daemon permanently larger each time. In a test that repeated the request every two
minutes, memory use went from 142 MB at rest to 608 MB, and it was still climbing. Two changes
fix this:

- Listing memories, and the memory statistics on the Settings page, no longer copy every
  memory's search vectors just to read the text.
- After any reply over 1 MB is sent, the daemon hands the freed memory back to the system. This
  takes about 5 ms and runs at most once every 10 seconds.

In the same test, memory use was about 213 MB after the same number of requests that had taken the
old version to 439 MB. It still creeps up slowly, at about a third of the old rate.

**Board cards no longer get stuck.**

- **Stop keeps meaning stop.** After about 64 edits or reorders, a card you had stopped could be
  treated as abandoned and taken back from its agent. It now stays stopped for as long as you
  leave it.
- **A stopped card you give to someone else starts for them.** Its "stopped" note says it stays
  with the agent until it is restarted or reassigned. Reassigning it used to leave it stuck in
  progress with nobody working on it. Now the new agent picks it up.
- **A card that comes back unfinished in two different rounds no longer goes straight to you.**
  The count of failed attempts now starts over once a card has been finished. Previously, a
  repeating card that failed once one week and once the next asked you "2 runs could not finish
  it" right away.

**You can now see how an unexpected crash happened.** If the daemon is killed by something it
cannot catch, such as a crash signal or a forced kill, the app now records how it ended. The next
start reports it, for example "the app saw it end by signal 9 (SIGKILL)", where the daemon used to
say only that it had "died through a path no hook could see". `nanna doctor` has a new
`daemon.last_exit` check with the same information: whether the daemon is running, stopped
cleanly, or died, and where to look in the logs.

## Under the hood

- Toolchain: nightly-2026-10-03. Newer nightlies cannot build Nanna yet: Rust renamed an internal
  function, and a library our database depends on still uses the old name. That library has a fix
  pending.
- Dependencies: postcss 8.5.29, plus three small lockfile updates.

## Still open

- `memory.list` still builds its whole reply in memory before sending it. The memory is now
  handed back afterwards, but building the reply directly would also make it faster.
- These memory numbers come from a test on a copy of a real store, not from a day of real use.
- If you press Start on a card in the same moment the stall check is releasing it, the start
  can still lose. The window is a few milliseconds.
