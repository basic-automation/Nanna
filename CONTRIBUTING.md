# Contributing to Nanna

Thank you for your interest in contributing to Nanna! This document provides guidelines and information for contributors.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

### Reporting Bugs

1. **Check existing issues** — search [GitHub Issues](https://github.com/basic-automation/Nanna/issues) to see if the bug has already been reported.
2. **Open a [bug report](https://github.com/basic-automation/Nanna/issues/new/choose)** with:
   - A clear, descriptive title
   - Steps to reproduce the problem
   - Expected vs actual behavior
   - Your environment (OS, Rust version, GPU if relevant)
   - Relevant logs or screenshots

### Suggesting Features

1. **Check existing issues and [Discussions](https://github.com/basic-automation/Nanna/discussions)** — someone may have already suggested it.
2. **Open a [feature request](https://github.com/basic-automation/Nanna/issues/new/choose)** describing:
   - The problem you're trying to solve
   - Your proposed solution
   - Alternatives you've considered

### Submitting Code

1. **Fork the repository** and create your branch from `master`.
2. **Make your changes** following the style guidelines below.
3. **Add tests** for any new functionality.
4. **Run the test suite** to ensure nothing is broken:
   ```bash
   cargo test -p <crate-you-changed>
   cargo clippy --workspace --all-targets
   ```
   Clippy runs with `all + pedantic + nursery` on every target (tests, benches and build
   scripts included), and the project does not use `#[allow]` / `#[expect]` to silence it —
   fix the lint instead. Lossy numeric casts go through the `nanna-numeric` crate, not `as`.
5. **Submit a pull request** with a clear description of your changes.

## Development Setup

### Prerequisites

- **Rust** via [rustup](https://rustup.rs) — `rust-toolchain.toml` pins a nightly toolchain, which
  rustup installs on the first `cargo` command
- **Node.js 22+** and **pnpm** (for the GUI)
- **Linux:** WebKitGTK 4.1 development headers for the GUI (`libwebkit2gtk-4.1-dev` on Debian/Ubuntu)
- **Ollama** (optional, for local model testing)

### Building from Source

```bash
git clone https://github.com/basic-automation/Nanna.git
cd Nanna

# Build
cargo build --release

# Run tests
cargo test --workspace

# Build GUI
cd gui
pnpm install
pnpm run tauri:build
```

## Style Guidelines

### Rust Code

- Follow standard Rust conventions (`rustfmt`, `clippy`); format the files you changed rather than
  running `cargo fmt` across the whole workspace, so unrelated files stay out of your diff
- All crates enable `clippy::all + pedantic + nursery` lints
- Use `thiserror` for library errors, `anyhow` for application errors
- Async code uses Tokio
- Prefer per-crate `cargo test -p <crate>` over full workspace runs

### Frontend (Vue/TypeScript)

- Vue 3 `<script setup>` style
- Tailwind CSS (Palenight theme)
- TypeScript strict mode

### Documentation

- Public APIs require doc comments
- Update README.md for user-facing changes
- Keep commit messages clear and descriptive

## Architecture Overview

See the Architecture section in [README.md](README.md#architecture) for crate layout and design patterns.

Key principles:
- **Daemon-first**: the daemon owns all state; GUI is a client
- **Local-first**: default to on-device; cloud is opt-in
- **Channels as clients**: every channel (GUI, Telegram, etc.) reaches the daemon via the same IPC

## Testing

- Unit tests live alongside the code they test
- Integration tests in `tests/` directories
- Async tests use `#[tokio::test]`
- Tests skip GPU/network by checking for API keys

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

- Ask in [Discussions](https://github.com/basic-automation/Nanna/discussions)
- Open an issue for bugs or feature requests
- See [ROADMAP.md](ROADMAP.md) for the project direction
- Check [SECURITY.md](SECURITY.md) for security-related concerns

---

*Thank you for helping make Nanna better!*
