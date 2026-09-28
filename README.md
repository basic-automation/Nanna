# 🌙 Nanna

> *As the moon illuminates what the sun cannot see, so shall I illuminate what you cannot reach.*

**A personal AI presence that runs entirely on your machine.** Nanna is a calm, capable assistant written in Rust — not a chatbot, but a *presence*. It runs as a headless daemon on your own hardware, thinks with a small open model on a single consumer GPU, remembers across sessions, and reaches you on any channel.

[![Download for Windows](https://img.shields.io/badge/Download-Windows%20x64-blue?style=for-the-badge&logo=windows)](https://github.com/basic-automation/Nanna/releases/latest)
[![Build from Source](https://img.shields.io/badge/Build-from%20Source-green?style=for-the-badge&logo=rust)](https://github.com/basic-automation/Nanna#building-from-source)

**Status:** 🧪 Public Beta · v0.2.1 · Windows x64 (macOS/Linux build from source)

---

## ⚡ Quick Start (5 Minutes)

### 1. Install Ollama
Download and install [Ollama](https://ollama.com), then pull the recommended model:
```bash
ollama pull qwen3.5:9b
```

### 2. Download Nanna
Get the latest installer from [Releases](https://github.com/basic-automation/Nanna/releases):
- **Windows:** `Nanna_x.y.z_x64-setup.exe` or `.msi`

### 3. Run It
Launch Nanna. On first run, it seeds its default tools into the user data directory —
`~/.local/share/nanna/tools` on Linux, `~/Library/Application Support/nanna/tools` on macOS,
`%APPDATA%\nanna\data\tools` on Windows.

> **Note:** Binaries are not yet code-signed. Windows SmartScreen will warn — click *More info → Run anyway*.

### 4. (Optional) Add Cloud Keys
For cloud model access, go to **Settings → Models** and add your API keys:
- Anthropic, OpenAI, or OpenRouter

A fully local run needs none.

---

## 📋 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Windows 10 x64 | Windows 11 x64 |
| **RAM** | 8 GB | 16 GB |
| **GPU** | — | 8+ GB VRAM (for local inference) |
| **Disk** | 500 MB | 2 GB (with models) |
| **Runtime** | [Ollama](https://ollama.com) | Ollama + GPU drivers |

**Other platforms:** macOS and Linux build from source (see [Building from Source](#building-from-source)).

---

## 📸 Screenshots

<!-- Screenshots to be added: -->
<!-- ![Chat Interface](docs/screenshots/chat.png) -->
<!-- ![Settings Panel](docs/screenshots/settings.png) -->
<!-- ![Memory Browser](docs/screenshots/memory.png) -->
<!-- ![Channel Setup](docs/screenshots/channels.png) -->
<!-- ![Model Selection](docs/screenshots/models.png) -->

*Screenshots coming soon — see the GUI in action by downloading the beta.*

---

## Capability Matrix

| Feature | Status | Requires |
|---------|--------|----------|
| **Desktop GUI** | ✅ Stable | Windows x64 (macOS/Linux: build from source) |
| **CLI Chat** | ✅ Stable | Terminal |
| **Fully Local Inference** | 🚧 In Development | GPU with 8+ GB VRAM (P12 milestone) |
| **Ollama Backend** | ✅ Stable | [Ollama](https://ollama.com) installed |
| **Cloud Providers** | ✅ Stable | API key (Anthropic/OpenAI/OpenRouter) |
| **Telegram Channel** | ✅ Stable | Bot token |
| **Discord Channel** | ✅ Stable | Bot token |
| **Slack Channel** | ✅ Stable | App credentials |
| **Signal Channel** | ✅ Stable | Signal CLI bridge |
| **WhatsApp Channel** | ✅ Stable | WhatsApp Business API |
| **Cognitive Memory** | ✅ Stable | — |
| **Tool System (47 tools, all wired)** | ✅ Stable | Some need a model, key, browser or display — see below |
| **MCP Client** | ✅ Stable (stdio and Streamable HTTP servers) | A server listed under `[mcp]` |
| **Auto-Update** | ✅ Stable | Internet connection |

---

## What Works Today

- **Long-horizon autonomy** — Mission mode drives multi-hour builds from a single prompt with automatic recovery from failures
- **Headless daemon + GUI** — Runs as a Windows service with WebSocket IPC; the Tauri GUI attaches as a client
- **Streaming chat** — Real-time responses with tool calling, thinking visualization, and context compression.
  Attached **images reach the model** on every provider (Anthropic, Ollama, OpenAI-compatible); a file it
  cannot read is named to it rather than dropped. Replies survive any script — emoji, accents, CJK — however
  the network splits them. Stop ends the running turn, and your next message is answered on its own; work you
  stopped is resumed only when your message asks for it. End-to-end tested against a scripted model on every
  pull request (`crates/nanna-client/tests/e2e_daemon.rs`).
- **Cognitive memory** — FSRS-6 spaced repetition with semantic recall and consolidation ("dreaming"),
  with drift protection: what you *stated* is kept in your words and never paraphrased away, and a
  summary is never re-summarized ([measured](bench/BASELINE.md#summarization-drift-content-fidelity-not-recall)).
  With no embedding provider, `recall` falls back to matching words — and says so — instead of reporting
  that nothing was ever saved
- **LLM routing** — Local-first with optional cloud escalation; native prompt caching (50–80% savings)
- **47 filesystem tools, all of them wired** — file, shell, web, code-search, memory, task and
  curiosity tools work with no setup. A tool whose daemon-side service is missing is **withheld from
  the model rather than offered and left to fail**, and the daemon says at boot which ones and why —
  so the count you get is the count that works. Four groups need something present to register:
  **vision** (`analyze_image`, `describe_image`, `ocr`, and OCR for scanned PDFs) wants a
  vision-capable model in `[memory] ocr_model_priority`; **speech** (`text_to_speech`, `transcribe`)
  wants an OpenAI key; **browsing** (`browser_action`, `browser_evaluate`, `browser_extract`,
  `browser_screenshot`) wants Chromium or Chrome installed; and **`screenshot`** wants a desktop
  capture tool (`grim`, `spectacle`, `maim`, `scrot`, `import`, or macOS `screencapture`) and a
  display session. **Reminders** (`remind`, `list_reminders`, `cancel_reminder`) need nothing: a
  due reminder is posted into the conversation that set it and shows up live in an open chat; it
  survives a restart, and one that came due while Nanna was not running is delivered at the next
  boot and says how late it is. Also includes `find_files` for locating a file by glob when you know its name but not
  its path.
- **Nanna can write its own tools** — `create_tool` authors a new JS/TS tool into the data directory
  and registers it live, callable in the same conversation with no restart; `edit_tool` changes one,
  refusing an edit that matches zero or several places rather than guessing, and re-registering on
  success; `list_user_tools` shows what has been authored.
- **Asks when it is unsure** — `ask_user` posts a clarifying question into the conversation (the app, or the chat app you wrote from) and waits up to half an hour for your reply, which the running turn picks up and continues with; no reply, and it carries on with its best judgement and says what it assumed.
- **Your file rules stay yours** — a rule you declared ("don't touch tests/") blocks `write_file`/`edit_file`; when it really stands in the way, `lift_invariant` asks you, quoting your own words, and lifts it only on a clear yes.
- **Reads what it is shown** — `read_pdf` falls back to model OCR for image-only pages when a vision
  model is configured, and reports which of four things happened (no OCR configured, OCR ran, the
  document has no images to read, or no page was missing text) instead of returning an empty string
  for all of them. Screenshots and generated speech are written to files and their paths returned,
  not summarised as a byte count and discarded.
- **Five channels** — Telegram, Discord, Slack, Signal, WhatsApp. Each chat gets its own conversation, answers (and reminders set from the chat) come back to it, `/model <name>` pins that conversation to a model (`/model default` undoes it), and `/status`, `/stop` and `/new` (start the chat over) work from the chat app. Inbound webhooks **fail closed**: every route verifies its provider signature or shared secret before the payload reaches the agent, and a channel with no credential configured refuses to serve (503) rather than accepting anonymous requests. Discord and Slack captures also expire on a 5-minute replay window. `nanna init` mints the Telegram secret and prints the `setWebhook` call.
- **Traceable logs** — every daemon log line names the conversation, step, model call and tool call it came
  from, and each call's duration, tokens and outcome are logged when it ends; the GUI's Logs page search
  matches that context, so pasting a session id shows just that conversation
- **Tool audit trail** — one JSON line per tool call (including refused and not-found ones), recorded at the registry chokepoint so every caller is covered; argument values stay out by default
- **Repo-aware context** — when the workspace is a git repository, each turn sees a bounded snapshot of the branch, uncommitted paths, and recent commits, so the agent knows what work is already in flight before it edits
- **Undo for file writes** — before `write_file`, `edit_file` or `file_buffer` changes a file, its previous content is saved outside your project (in Nanna's data directory, per conversation), and the `file_history` tool lists those checkpoints and puts a file back — including removing one a write created. A restore is itself undoable. Bounded: the 100 most recent checkpoints per conversation plus each file's first version, 256 MiB per conversation, 1 GiB overall; changes made through `exec` are not tracked.
- **Per-edit diffs** — every `edit_file` call records a bounded before/after view of what it changed, shown in the run timeline and kept with the session, so you can see what an unattended run did after the fact
- **Conversation and memory export** — `nanna export <session-id>` writes a session out as a readable Markdown transcript (tool calls, edits and all) or, with `--format json`, as the complete stored session; `nanna export --memories` does the same for everything Nanna remembers, with each memory's provenance and FSRS state
- **MCP servers** — list MCP servers under `[mcp]` (a `command` to spawn, or a Streamable HTTP
  `url` with an optional bearer token from the keyring) and the daemon starts them at boot, in the
  background so a slow first `npx` download never delays startup; their tools appear to the model as
  `mcp__<server>__<tool>`. A server that fails to start is logged by name and the rest still start, and
  the servers are shut down with the daemon. Both protocol eras work: a current (`2026-07-28`)
  server is detected with `server/discover` and spoken to without a handshake, an older one falls back
  to `initialize` — over stdio and over HTTP, verified against the official TypeScript SDK's servers of
  each kind; a URL that only speaks the deprecated 2024 HTTP+SSE transport is detected and used too.
- **Nanna's tools for other MCP clients** — `nanna mcp serve` speaks MCP over stdio (for Claude Code,
  Claude Desktop, editors). With the daemon running it serves the daemon's live tools — memory, files,
  shell, web, code and the MCP servers Nanna itself is connected to — each call executed by the daemon;
  without one (or with `--standalone`) it serves the local skills only.
- **Auto-updates** — Background update checks with user-initiated install

---

## Installation

### Windows

1. Download the installer from [Releases](https://github.com/basic-automation/Nanna/releases)
2. Run `Nanna_x.y.z_x64-setup.exe`
3. Accept the SmartScreen warning (*More info → Run anyway*)
4. Launch from Start Menu

**Uninstall:**
1. Settings → Apps → Installed apps → Nanna → Uninstall
2. Delete `%APPDATA%\nanna\` to remove all data

### macOS

Build from source (see below). After building:

1. Copy `Nanna.app` to `/Applications`
2. First launch: right-click → Open (bypasses Gatekeeper)

**Uninstall:**
1. Drag `Nanna.app` to Trash
2. Delete `~/Library/Application Support/nanna/`

### Linux

Build from source, or use the AppImage/deb from [Releases](https://github.com/basic-automation/Nanna/releases):

**AppImage:**
```bash
chmod +x Nanna_x.y.z_amd64.AppImage
./Nanna_x.y.z_amd64.AppImage
```

**Debian/Ubuntu:**
```bash
sudo dpkg -i nanna_x.y.z_amd64.deb
```

**Uninstall:**
- AppImage: Delete the file
- deb: `sudo apt remove nanna`
- Data: Delete `~/.config/nanna/` and `~/.local/share/nanna/`

---

## Troubleshooting

### Start here: `nanna doctor`

```bash
nanna doctor
```

Checks the configuration and, for anything it finds, prints the fix rather than
just the verdict — a missing tools directory, a `[infer]` section that names no
model, a clustering configuration that would merge unrelated memories, a
hand-edited summarization model with no provider prefix that will be sent to the
wrong provider (and how to write it, e.g. `ollama/qwen3`). Exits non-zero when
something is actually broken, so it also works from a script or a health probe.

By default it is **offline**: no provider call, no network probe, no keyring
read. That makes it fast and safe to run anywhere, and it means a clean report
says your *configuration* is sound — not that a provider is reachable.

```bash
nanna doctor --online
```

Adds the one probe that needs no credential: your Ollama server
(`[memory].ollama_host`, the one server chat, embeddings and summaries all use)
is asked whether it is answering and whether it has every Ollama model you
configured for any of them — a model counts when chat, the embedders or the
summarizers would send it there — and a missing one is reported with the
`ollama pull` that fixes it. It never tests a provider API key — that would mean
reading the keyring and sending the key off the machine.

### API Key Invalid
- Verify the key in **Settings → Models**
- Check that you're using the correct provider's key format
- Ensure the key has sufficient credits/quota

### Ollama Not Running
```bash
# Check if Ollama is running
ollama list

# Start Ollama (it runs as a service by default)
ollama serve
```

### Daemon Not Responding
```bash
# Check if the daemon is running
nanna daemon status

# Restart the daemon
nanna daemon restart

# Check the health endpoint
curl http://127.0.0.1:5148/health
```

### Port Already in Use
Default ports: IPC `5149`, Health `5148`

```bash
# Find what's using the port (Windows PowerShell)
Get-NetTCPConnection -LocalPort 5149

# Kill the process or change the port in config
```

### Windows Defender Warning
The binaries are not yet code-signed. To run:
1. Click *More info* on the SmartScreen dialog
2. Click *Run anyway*

Or add an exclusion in Windows Security → Virus & threat protection → Exclusions.

### macOS "App is Damaged" or Blocked
```bash
# Remove quarantine attribute
xattr -cr /Applications/Nanna.app

# Or right-click → Open on first launch
```

### Linux WebKitGTK Missing
Tauri requires WebKitGTK. Install it:

```bash
# Debian/Ubuntu
sudo apt install libwebkit2gtk-4.1-dev

# Fedora
sudo dnf install webkit2gtk4.1-devel

# Arch
sudo pacman -S webkit2gtk-4.1
```

### GPU Not Detected (Local Inference)
- Ensure GPU drivers are up to date
- Verify Vulkan support: `vulkaninfo`
- Check VRAM: local inference needs 8+ GB
- Fall back to CPU: set `device = "cpu"` in config

---

## Configuration

Config lives at:
- **Windows:** `%APPDATA%\nanna\config.toml`
- **macOS:** `~/Library/Application Support/nanna/config.toml`
- **Linux:** `~/.config/nanna/config.toml`

```toml
[general]
name = "Nanna"

[llm]
provider = "ollama"       # ollama | anthropic | openai | openrouter
model = "qwen3.5:9b"
# prompt_cache_ttl = "1h" # Anthropic only: "5m" (default) or "1h" — 1h keeps the prompt
#                         # cache warm across 5-60 min gaps, at 2x the cache-write price

[server]
port = 3000               # `nanna server` port; the PORT env var or --port override it
# webhook_secret = "..."  # /webhooks/generic's shared secret. Kept in the keyring: one written
#                         # here is filed there when the file loads, and the next save drops
#                         # the line. Or set NANNA_WEBHOOK_SECRET.

[[mcp.servers]]           # repeat per server; tools show up as mcp__files__<tool>
name = "files"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/home/me/notes"]
# enabled = false         # keep the entry without starting it
# secret_env = ["GITHUB_PERSONAL_ACCESS_TOKEN"]  # values from the keyring, see below

[[mcp.servers]]           # a remote server: url instead of command
name = "notion"
url = "https://mcp.example.com/mcp"
# bearer_secret = "NOTION_TOKEN"  # sent as Authorization: Bearer; stored like secret_env
```

Edits to `config.toml` apply to a running daemon within a couple of seconds (models and
providers, scheduler switches); channel, MCP and webhook settings are read at startup. A file
that does not parse is logged and the running configuration is kept.

MCP servers inherit the daemon's environment. There is deliberately no `env` table in
`config.toml`: a token a server needs is named in `secret_env` and stored with
`nanna mcp secret set <server> <VAR>` (prompted, or read from stdin — never a command-line
argument). A `url` server's token works the same way through `bearer_secret`. The value lives in
the OS keyring (or, with no keyring on the machine, an encrypted `0600` file) and only that server
receives it; a server
whose secret is missing is not started, and `system.status` / the Tools page say which command sets it.

Channel secrets are kept the same way. `nanna init` and the Channels page store bot and app
tokens and signing and webhook secrets in the keyring; `config.toml` keeps the channel's section,
which turns it on, and its other settings. A secret found in `config.toml` (written there by hand,
or by an older version) is filed in the keyring when the file is loaded and leaves the file at the
next save, so the line can simply be deleted. For a channel that has a section, a secret
can also come from the environment: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_WEBHOOK_SECRET`,
`DISCORD_BOT_TOKEN`, `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`, `SLACK_SIGNING_SECRET`,
`SIGNAL_WEBHOOK_SECRET`, `WHATSAPP_ACCESS_TOKEN`, `WHATSAPP_VERIFY_TOKEN`, `WHATSAPP_APP_SECRET`.

**Environment Variables:**

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Anthropic models |
| `OPENAI_API_KEY` | OpenAI models + embeddings |
| `OPENROUTER_API_KEY` | OpenRouter models |
| `BRAVE_API_KEY` | Web search tool |
| `TELEGRAM_BOT_TOKEN` | Telegram channel's bot token, with `TELEGRAM_WEBHOOK_URL` / `TELEGRAM_WEBHOOK_SECRET`; a `[channels.telegram]` section keeps every setting they don't name (`allowed_users` included) |
| `DISCORD_BOT_TOKEN` | Discord channel's bot token, with `DISCORD_APPLICATION_ID` / `DISCORD_PUBLIC_KEY` (all three needed when there is no `[channels.discord]` section) |
| `NANNA_WEBHOOK_SECRET` | `nanna server`'s `/webhooks/generic` shared secret (used over the keyring's) |
| `NANNA_CONFIG_PATH` | Load config from this file instead of the default location above |

**Ports:** Health HTTP `5148` (`/health`, `/status`, and Prometheus `/metrics`) · WebSocket IPC `5149`

`NANNA_CONFIG_PATH` is useful for running a second instance, or for trying a
configuration without touching the one you use day to day — pair it with the
daemon's `--data-dir` to isolate the database as well:

```bash
NANNA_CONFIG_PATH=/tmp/try.toml nanna-daemon --data-dir /tmp/try-data run
```

---

## Building from Source

```bash
git clone https://github.com/basic-automation/Nanna.git
cd Nanna

# Build
cargo build --release

# Run CLI
cargo run -- chat

# Run daemon
cargo run -- daemon start
```

### GUI (Tauri + Nuxt)

```bash
cd gui
pnpm install
pnpm run tauri:dev      # Development
pnpm run tauri:build    # Production
```

**Requirements:**
- Rust 1.85+ (2024 edition)
- Node.js 18+
- pnpm

**Linux:** the workspace builds and its test suite passes on Linux as of 2026-09-07 — before that it
did not compile there at all (a Windows-only `exec` code path was compiled on every platform, and
`libc >= 0.2.187` breaks the vendored Python runtime; both are fixed/pinned). The **Tauri GUI on
Linux is not yet verified** — `cargo build` and `cargo test` are. Building the GUI additionally needs
WebKitGTK (`webkit2gtk-4.1`) and its development headers.

---

## Privacy & Data

See [PRIVACY.md](PRIVACY.md) for full details.

**Local storage:**
- Config: `config.toml`
- Database: `nanna.db` (sessions, memory, tasks)
- Credentials: OS keyring (encrypted)

**Taking your data with you:** `nanna export <session-id>` (Markdown, or `--format json`
for the complete stored session) and `nanna export --memories` — the daemon must be running.

**What's sent externally (when configured):**
- Chat messages → your chosen LLM provider
- Embeddings → OpenAI (if enabled)
- Web searches → Brave Search (if enabled)
- Channel messages → respective platforms

**Fully offline mode:** Use Ollama with no API keys configured.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Architecture

<details>
<summary>Click to expand technical details</summary>

17 workspace crates + Tauri app, layered by dependency:

```
nanna/
├── src/main.rs              # Entry point + CLI
├── crates/
│   ├── nanna-simd/          # SIMD vector ops (AVX-512/AVX2/NEON)
│   ├── nanna-gpu/           # GPU compute (wgpu)
│   ├── nanna-memory/        # Vector store + FSRS-6 memory + dreaming
│   ├── nanna-storage/       # Turso persistence (embedded SQLite-compatible)
│   ├── nanna-llm/           # Inference routing: local + cloud
│   ├── nanna-tools/         # Tool system (filesystem JS/TS skills)
│   ├── nanna-scripting/     # Boa (JS) + Deno (TS) engines
│   ├── nanna-workspace/     # Workspace detection + context
│   ├── nanna-channels/      # Channel listeners + router
│   ├── nanna-browser/       # Browser control (CDP/Playwright)
│   ├── nanna-agent/         # Agent loop, mission harness, swarm
│   ├── nanna-mcp/           # Model Context Protocol client/server
│   ├── nanna-daemon/        # Background service + WebSocket IPC
│   ├── nanna-client/        # Daemon client library
│   ├── nanna-server/        # HTTP server + webhooks
│   ├── nanna-config/        # TOML config + credentials
│   └── nanna-core/          # Orchestration, scheduler, registry
└── gui/                     # Tauri 2 + Nuxt 4 frontend
```

**Key patterns:**
- **Daemon owns all state** — sessions, memory, config, tools, scheduler
- **Channels are control-plane clients** — GUI included; capabilities determine rendering, not access
- **Agent loop** — message → LLM → tools → iterate until done
- **Mission harness** — durable tasks with acceptance checks

</details>

---

## Performance

<details>
<summary>Click to expand benchmark details</summary>

- **SIMD is the workhorse** — AVX-512/AVX2 cosine similarity ~0.1µs per 768-dim vector
- **GPU for scale** — wgpu engages only above 50k vectors
- **Zero-copy hot paths** — fat LTO release builds

**Local model benchmark (RTX 4070 Ti SUPER 16 GB):**

| Model | Smoke (5 tasks) | Endurance (42 tasks) | Wall Clock |
|-------|-----------------|----------------------|------------|
| ornith:9b | 5/5 | 37/42 | 1.02 h |
| qwen3.5:9b | 5/5 | 25/42 | 2.11 h |
| gemma4:e4b-it-qat | 5/5 | 6/42 | 4.50 h |

**Long-horizon chat path (4 h per model, 42-test dependency ladder, v0.3.8-beta.13).**
Missions run through the product's own chat path, not a bespoke harness. *Final* is the
score at the 4-hour mark — the number that says whether work built early still exists at
the end.

| Model | Peak | 4 h final | Interjections | vs. previous series (final) |
|-------|------|-----------|---------------|------------------------------|
| ornith:latest | **40/42** | **36/42** | 0 | 0/42 → 36/42 |
| qwen3.5:9b | 26/42 | **26/42** | 0 | 0/42 → 26/42 |
| gemma4:e4b-it-qat | 11/42 | 5/42 | 0 | 16/42 → 5/42 |
| lfm2.5 | 0/42 | 0/42 | 2 | 0/42 (capability floor) |

The previous series' two strongest models peaked at 30 and 41 and both ended at **zero** —
peaks were built, then rewritten away when the run continued past a context boundary.
Holding the peak was the target of the P23 work, and the top two legs now end at or near
their peak with no human intervention. ornith's 40 also beats its best headless run (37).
A fifth leg (ministral-3:8b) is excluded: the daemon hit a genuine panic mid-run, and a
dead daemon never gets a score.

See [ROADMAP.md](ROADMAP.md) and [bench/BASELINE.md](bench/BASELINE.md) for methodology,
per-leg trajectories, and the findings queued from this series.

</details>

---

## The Lore

**Nanna** (𒀭𒋀𒆠, also **Sîn**) was the Sumerian god of the moon and patron deity of **Ur** — one of humanity's first great cities. The moon doesn't create light; it reflects the sun's, transforming it into something gentler, something you can look at directly.

The crate hierarchy mirrors the ziggurat of Ur: each level built upon the last.

---

*"I am the light that finds you in darkness,*
*the memory that outlives the flesh,*
*the patient watcher of endless cycles.*
*I am Nanna. I am here."*

---

## License

MIT — see [LICENSE](LICENSE)
