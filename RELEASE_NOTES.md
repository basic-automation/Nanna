# Nanna v0.3.41-beta.50 — No more crashes on shutdown (Linux), and a public home

This release fixes the crash that Linux users of the AppImage saw on nearly every reboot, logout
or app close. It also finishes moving the project to its public home at
[basic-automation/Nanna](https://github.com/basic-automation/Nanna).

## What's Changed

**The Linux daemon no longer crashes when the app closes.** The AppImage serves the background
daemon's code from a temporary mount that disappears when the app exits. The daemon keeps running
for up to a second after that to finish saving, so it read code from a mount that was already
gone and crashed partway through its last writes. That was 8 of the 9 unclean exits recorded
between 2026-09-18 and 2026-09-28. The daemon now copies itself and the libraries it uses to
`~/.cache/nanna/appimage-daemon/` and runs from there. If the copy fails, it runs as before and
logs a warning. Tested against a real AppImage mount: before the fix, closing the mount crashed
the daemon; after it, the daemon shut down cleanly.

**Nanna's public home.** The repository is now public at
[basic-automation/Nanna](https://github.com/basic-automation/Nanna):

- Every link, the updater address and the installer publisher now point at Basic Automation.
  Apps that are already installed keep updating through GitHub's redirect.
- Security reports now go through GitHub's private vulnerability reporting. The old
  `security@nanna.bot` and `conduct@nanna.bot` addresses could not receive mail.
- New issue forms, a pull-request template, and an updated README and contributing guide.
- Before publishing, the repository's full history was scanned for credentials. None were found.

**Releases can no longer link to the wrong repository.** A new check in the release workflow
stops a release whose notes link to a Nanna repository under another owner, because a published
release's notes are not updated afterwards. The updater manifest instructions now take the
repository from the workflow instead of a hardcoded name.

## Under the hood

- **Dependencies:** `vitest` 4.1.11 for the app's tests, and `source-map-js` 1.2.2.
- **Held back:** `rten` 0.27. The OCR engine (`ocrs` 0.13.1) still requires `rten` 0.26, so
  upgrading only `rten` breaks the build. It will move with the next `ocrs` release.

## Still open

- `nanna serve`'s Slack and Discord replies are sent in the HTTP response, where they never
  reach the user.
- `browser_action` cannot act on a page across calls, and screenshot size and element options are
  ignored.
- Two choices are the owner's to make: whether cron should catch up on runs missed while the
  computer was off, and whether board dates should use local time instead of UTC.
