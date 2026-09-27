# Nanna v0.3.32-beta.41 — Stopped Means Stopped

A repair release with one step forward on the task board. The headline fixes are about Python:
a timed-out `python.exec` used to keep running forever in the background, and any Python script
that imported `subprocess` quietly took Ctrl-C away from the daemon. Both are fixed, and both
were checked on a real running daemon, not only in tests. The board also gains the first half
of its Task Management Agent: the part that decides who works on a card.

## What's Changed

**A Python script that runs past its time limit is stopped.** The embedded interpreter can't be
stopped from outside, so when a script timed out Nanna reported the timeout and let it keep
running. A `while True:` loop used a whole CPU core until the daemon restarted. Nanna now
interrupts the script at its next step and keeps interrupting until it stops. On a running
daemon, a runaway loop given a 3-second limit stopped within a moment of timing out.

**Python can no longer take Ctrl-C away from the daemon.** The interpreter was allowed to install
its own Ctrl-C handler for the whole process. Any script that imported `signal` did this, and so
does `subprocess`. After that, Ctrl-C and `kill -INT` no longer shut the daemon down cleanly.
That permission is now off. Checked on a real daemon: after a script imported both modules,
`kill -INT` still shut it down cleanly.

**The board's router can decide who does a card (not switched on yet).** The Task Management
Agent now has its decision-making half. It reads a card, the card's recent thread, the members
of the board and how each has done on past cards, and gives one answer: assign the card, split
it into sub-tasks, ask you a question, or park it with a reason. Every answer is posted on the
card's thread. When it asks you something, it creates a card for you, and the original card waits
until you finish that card. A card that is being worked on is never reassigned. The router is
**not switched on yet**: while the chat still exists, the router would also pick up the chat's
own to-do cards, and a question from it could stall a chat mid-task. It will be switched on
together with the board.

**A card records who created it.** Until now the "created" entry named the card's assignee as
its author. That entry now names the actual creator, which the router needs to tell cards it made
itself from new work.

## Fixes

- **Command output with terminal links is clean.** `cargo` and `ls --hyperlink` wrap file paths
  in invisible terminal-link codes, and those codes used to leak into the output as
  `8;;file:///…` next to the path. Every kind of terminal escape code is now removed completely.
- **A missing working directory is reported as a missing directory.** Running a command in a
  directory that doesn't exist used to fail with "No such file or directory", which looks as if
  the command is missing. The error now names the directory.
- **Scripts can't read an endless file into memory.** A script reading `/dev/zero`, or a file that
  kept growing, read until the daemon ran out of memory. Reads now stop at 64 MiB and say so.
- **Git context is capped while it is being read.** Nanna used to read all of `git status`'s
  output before keeping the first 64 KiB, and a command that never stopped writing used up the
  whole timeout and produced nothing. It now stops reading at the limit.
- **Transparent screenshots are shrunk before sending.** An image with a transparent background
  couldn't be converted to a smaller JPEG, so it was sent over the provider's size limit and
  rejected. The "lower the quality" step also never took effect. Both are fixed.
- **The list of verified results in a long session no longer grows without limit.** It is the
  one part of the context that is never summarized. Simple look-around commands (`ls`, `cat`,
  `git status`) that succeeded are now counted on one line instead of listed one by one. The
  rest are listed newest first within a fixed size, and the list says how many it left out.
- **Debug copies of prompts are private and size-limited.** Copies of prompts saved for
  debugging are kept next to the daemon's logs, readable only by you and limited in size. They
  used to be written to the shared `/tmp` folder, where other users could read them, with no
  size limit.
- A tool's own `timeout` setting now applies no matter how the tool is loaded, and an absurdly
  large value can no longer overflow to a timeout of a few milliseconds.

## Dependencies

Tauri plugins updated (dialog 2.8, fs 2.6, notification 2.5, process 2.4, shell 2.4,
updater 2.13), with the JavaScript packages updated to match. The app was checked by launching it
with its own daemon and driving it through WebDriver.

## Still open

- The router is ready but not connected to the board. That connection comes with the board
  itself.
- Python that is waiting on the network or a file can't be interrupted until the wait ends, and
  a script that catches every exception can ignore the interrupt. After 5 seconds Nanna stops
  trying and logs a warning.
- TypeScript 7 is still blocked on `vue-tsc`.
