# Release notes footer

`RELEASE_NOTES.md` is rewritten from scratch for every release, so its closing
footer is retyped by hand each time — including the repository URL. That is how
the move to `basic-automation` left 13 already-published release bodies pointing
at the old owner: a published body is not in version control, so the repo-wide
link fix could not reach them and each one had to be edited afterwards.

Copy the footer from here rather than from the previous release's prose, and the
owner stays right.

## The footer

```markdown
---

Updating from <PREVIOUS-MINOR> or earlier? The
[<LAST-MINOR> release notes](https://github.com/basic-automation/Nanna/releases/tag/<LAST-TAG>) cover
<one clause naming what that release changed for the reader> — and link back to <PREVIOUS-MINOR>'s.
```

- `<LAST-TAG>` is the full tag of the release immediately before this one, e.g.
  `v0.3.31-beta.40`. Take it from `gh release list`, not from the version
  number: the tag sequence has gaps, and one early tag (`v3.3-beta.8`) does not
  follow the scheme at all.
- `<LAST-MINOR>` / `<PREVIOUS-MINOR>` are the bare versions (`0.3.31`, `0.3.30`).
- The closing clause is written fresh each release. It describes what the reader
  gets, not which internals moved — the chain of footers is how someone updating
  across several versions finds what they skipped, so each link must actually
  point at notes that link onward in turn.

## Any link to this repository

Write the owner as `basic-automation`. Links to *other* repositories are fine as
they are — release bodies have always linked to things like `rust-lang/rust`.

The `check-release-notes` job in `.github/workflows/release.yml` enforces this:
it fails the release if `RELEASE_NOTES.md` links to a repository named `Nanna`
under any owner other than this one. It runs beside the builds, so it reports in
seconds instead of after the Windows build.
