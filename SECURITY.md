# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.2.x   | ✅ |
| < 0.2   | ❌ |

The public beta (`0.2.x`) is the only actively supported line. Security fixes
land on `master` and ship in the next beta/patch release; we do not back-port
to archive tags.

## Reporting a vulnerability

**Please do not open a public GitHub issue for security reports.**

Email **security@nanna.bot** (or, if that address is unreachable, open a
[private security advisory](https://github.com/basic-automation/Nanna/security/advisories/new)
on the GitHub repository).

Include, where you can:

- A short description of the issue and its impact
- Steps to reproduce, or a proof-of-concept
- Affected version / commit, OS, and install method (installer vs. source)
- Whether you plan to disclose publicly, and on what timeline

You should receive an acknowledgement within **72 hours**. We aim to ship a
fix or mitigation within **30 days** for high-severity issues; more complex
root causes may take longer, in which case we will keep you informed.

We are happy to credit reporters who would like to be named. Please let us
know your preferred credit line (name, handle, or anonymous).

## Scope

In scope:

- The Nanna daemon, CLI, and Tauri desktop app
- Default tool scripts shipped with the installer
- The WebSocket IPC protocol and the health/webhook HTTP surfaces
- Credential storage (OS keyring + encrypted file fallback)
- Channel webhook signature verification

Out of scope:

- Vulnerabilities that require physical access to an unlocked machine
- Issues in third-party services Nanna talks to (Anthropic, OpenAI, Telegram, …)
- Social-engineering the user into pasting a secret into the chat box
- Reports generated solely by automated scanners with no demonstrated impact

## Security posture (what we try to guarantee)

- **Secrets never touch `config.toml`.** API keys and tokens live in the OS
  keyring, with an AES-256-GCM encrypted file fallback only when no keyring is
  available.
- **Local control plane by default.** Health, IPC, and webhook servers bind
  `127.0.0.1`. Binding a non-loopback address logs a warning and is an explicit
  operator choice.
- **Webhook authenticity.** Discord (Ed25519), Slack (HMAC-SHA256 + replay
  window), and Telegram (`X-Telegram-Bot-Api-Secret-Token`) are verified when
  configured; unauthenticated delivery is rejected.
- **Tool policy is deny-wins** and is enforced *after* alias/fuzzy resolution,
  so a renamed tool cannot slip a denied action past the gate.
- **Path traversal** on user-authored tools and workspace files is rejected at
  the write chokepoints.

If you find a place where any of the above is not true, that is a valid report.

## Prefer private disclosure

We ask that you give us a reasonable window to fix serious issues before
public disclosure. Coordinated disclosure protects users who have not yet
updated. We will not pursue legal action against good-faith researchers who
follow this process.
