//! Secure credential storage.
//!
//! Primary backend: the OS keyring (Windows Credential Manager, macOS Keychain,
//! Linux Secret Service) via the `keyring` crate.
//!
//! Fallback backend: an **AES-256-GCM encrypted** file under the canonical
//! application data directory (`com.nanna.nanna`). The envelope key is kept in
//! the OS keyring when available; otherwise a per-machine `0600` key file sits
//! next to the encrypted envelope. Secrets never touch disk as plaintext JSON.
//!
//! Also reads Claude Code CLI OAuth tokens from `~/.claude/.credentials.json`
//! (that file is owned by Claude Code, not us).

use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Nonce};
use keyring::Entry;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use thiserror::Error;
use tracing::{debug, info, warn};
use zeroize::Zeroize;

/// Service name for keyring storage
const KEYRING_SERVICE: &str = "nanna";

/// Credential key names
pub mod keys {
    pub const ANTHROPIC_API_KEY: &str = "anthropic_api_key";
    /// Anthropic OAuth access token (bare string; what request auth sends).
    /// Same key `Config::load_secrets_from_store` hydrates from.
    pub const ANTHROPIC_OAUTH_TOKEN: &str = "anthropic_oauth_token";
    /// Full Anthropic OAuth credential as a JSON envelope.
    ///
    /// Carries the access token plus refresh token and expiry when the login
    /// flow had them — that is what makes startup token refresh possible. The
    /// bare-token key above is kept in lockstep for the config hydration path.
    pub const ANTHROPIC_OAUTH_CREDENTIAL: &str = "anthropic_oauth_credential";
    pub const OPENAI_API_KEY: &str = "openai_api_key";
    pub const OPENROUTER_API_KEY: &str = "openrouter_api_key";
    pub const GITHUB_TOKEN: &str = "github_token";
    pub const BRAVE_API_KEY: &str = "brave_api_key";
    pub const OLLAMA_API_KEY: &str = "ollama_api_key";
    /// The Ollama server [`OLLAMA_API_KEY`] was saved for (not a secret).
    ///
    /// The token is only ever loaded for that server: without the record it
    /// followed whatever address was configured, so pointing Nanna at another
    /// server sent it the previous server's token. A token with no record was
    /// saved by an older build and counts as the current server's.
    pub const OLLAMA_API_KEY_HOST: &str = "ollama_api_key_host";
    /// Set once the store's `[llm]` keys have been sorted out under a
    /// non-Anthropic `[llm].provider` (not a secret).
    ///
    /// Until 2026-09-18 `nanna init` filed that provider's key under
    /// [`ANTHROPIC_API_KEY`]; the first load under such a provider moves it
    /// to the provider's own entry. From then on only Anthropic's key is
    /// saved there, so it is never looked at again.
    pub const LLM_KEYS_FILED_BY_PROVIDER: &str = "llm_keys_filed_by_provider";
    // Channel secrets: `[channels]` in `config.toml` names the channel and
    // its other settings, and these hold the secrets (`channel_secrets.rs`).
    pub const TELEGRAM_BOT_TOKEN: &str = "telegram_bot_token";
    pub const TELEGRAM_WEBHOOK_SECRET: &str = "telegram_webhook_secret";
    pub const DISCORD_BOT_TOKEN: &str = "discord_bot_token";
    pub const SLACK_BOT_TOKEN: &str = "slack_bot_token";
    pub const SLACK_APP_TOKEN: &str = "slack_app_token";
    pub const SLACK_SIGNING_SECRET: &str = "slack_signing_secret";
    pub const SIGNAL_WEBHOOK_SECRET: &str = "signal_webhook_secret";
    pub const WHATSAPP_ACCESS_TOKEN: &str = "whatsapp_access_token";
    pub const WHATSAPP_VERIFY_TOKEN: &str = "whatsapp_verify_token";
    pub const WHATSAPP_APP_SECRET: &str = "whatsapp_app_secret";
    pub const ELEVENLABS_API_KEY: &str = "elevenlabs_api_key";
    /// `[server].webhook_secret`: the shared secret `nanna server`'s generic
    /// webhook requires (`server_secret.rs`).
    pub const SERVER_WEBHOOK_SECRET: &str = "server_webhook_secret";
}

/// Credential errors
#[derive(Error, Debug)]
pub enum CredentialError {
    #[error("Credentials not found")]
    NotFound,
    #[error("Credentials expired")]
    Expired,
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Token refresh failed: {0}")]
    RefreshFailed(String),
    #[error("Home directory not found")]
    NoHomeDir,
    #[error("Keyring error: {0}")]
    Keyring(String),
    /// AES-GCM envelope encrypt/decrypt failure.
    #[error("Credential crypto error: {0}")]
    Crypto(String),
}

impl From<keyring::Error> for CredentialError {
    fn from(e: keyring::Error) -> Self {
        match e {
            keyring::Error::NoEntry => Self::NotFound,
            _ => Self::Keyring(e.to_string()),
        }
    }
}

// =============================================================================
// Secure Keyring Storage
// =============================================================================

/// Cross-platform credential store using the OS keyring, with an AES-GCM encrypted file fallback.
#[derive(Debug, Clone, Default)]
pub struct SecureStore {
    /// Fallback to file storage if keyring unavailable
    pub allow_file_fallback: bool,
    /// Bypass the keyring entirely and use only the file store.
    ///
    /// Intended for headless/service contexts (and tests) where the OS keyring
    /// is inaccessible or non-deterministic — a keyring `set` can report success
    /// while a later `get` fails against a locked or absent secret service.
    file_only: bool,
    /// Override directory for the file store (default: platform data dir).
    file_dir: Option<PathBuf>,
}

impl SecureStore {
    /// Create a new secure store
    #[must_use]
    pub const fn new() -> Self {
        Self {
            allow_file_fallback: true,
            file_only: false,
            file_dir: None,
        }
    }

    /// Create without file fallback (keyring-only)
    #[must_use]
    pub const fn keyring_only() -> Self {
        Self {
            allow_file_fallback: false,
            file_only: false,
            file_dir: None,
        }
    }

    /// Create a file-only store rooted at `dir` (bypasses the OS keyring).
    ///
    /// Deterministic and self-contained: every `set`/`get`/`delete` reads and
    /// writes `dir/credentials.enc` only (AES-GCM). Used for headless deployments and for
    /// tests, so credential round-trips never depend on an interactive keyring.
    #[must_use]
    pub const fn file_only_at(dir: PathBuf) -> Self {
        Self {
            allow_file_fallback: true,
            file_only: true,
            file_dir: Some(dir),
        }
    }

    /// Get a credential from the keyring
    ///
    /// # Errors
    ///
    /// Returns [`CredentialError::NotFound`] when the key is not stored, and
    /// [`CredentialError::Keyring`] when the keyring entry cannot be opened, or
    /// cannot be read and file fallback is disabled. When the keyring has no
    /// entry or fails (with fallback enabled), or the store is file-only, the
    /// file store's errors are returned instead.
    /// The encrypted file store fails with [`CredentialError::NoHomeDir`] when no
    /// data directory can be resolved, [`CredentialError::Io`] or
    /// [`CredentialError::Json`] when its file (or a legacy plaintext file being
    /// migrated) cannot be read, written or parsed, and
    /// [`CredentialError::Crypto`] when it cannot be decrypted or encrypted.
    pub fn get(&self, key: &str) -> Result<String, CredentialError> {
        debug_assert!(!key.is_empty(), "credential key must not be empty");
        // File-only stores never touch the keyring.
        if self.file_only {
            return self.get_from_file(key);
        }
        let Some(entry) = self.open_entry(key)? else {
            return self.get_from_file(key);
        };
        match entry.get_password() {
            Ok(value) => {
                debug!("Retrieved credential '{}' from keyring", key);
                Ok(value)
            }
            Err(keyring::Error::NoEntry) => {
                // Try file fallback if allowed
                if self.allow_file_fallback {
                    self.get_from_file(key)
                } else {
                    Err(CredentialError::NotFound)
                }
            }
            Err(e) => {
                warn!("Keyring error for '{}': {}", key, e);
                // Try file fallback
                if self.allow_file_fallback {
                    self.get_from_file(key)
                } else {
                    Err(e.into())
                }
            }
        }
    }
    
    /// Store a credential in the keyring
    ///
    /// # Errors
    ///
    /// Returns [`CredentialError::Keyring`] when the keyring entry cannot be
    /// opened, or the write fails and file fallback is disabled. When the write
    /// fails with fallback enabled, or the store is file-only, the file store's
    /// errors are returned instead.
    /// The encrypted file store fails with [`CredentialError::NoHomeDir`] when no
    /// data directory can be resolved, [`CredentialError::Io`] or
    /// [`CredentialError::Json`] when its file (or a legacy plaintext file being
    /// migrated) cannot be read, written or parsed, and
    /// [`CredentialError::Crypto`] when it cannot be decrypted or encrypted.
    pub fn set(&self, key: &str, value: &str) -> Result<(), CredentialError> {
        debug_assert!(!key.is_empty(), "credential key must not be empty");
        // File-only stores never touch the keyring.
        if self.file_only {
            return self.set_to_file(key, value);
        }
        let Some(entry) = self.open_entry(key)? else {
            return self.set_to_file(key, value);
        };
        match entry.set_password(value) {
            Ok(()) => {
                info!("Stored credential '{}' in keyring", key);
                // Drop any file-fallback copy, as `delete` does. Left behind, it
                // is the OLD secret, and `get` reads it the moment the keyring
                // next fails — a rotated key quietly comes back.
                if self.allow_file_fallback {
                    match self.delete_from_file(key) {
                        Ok(()) | Err(CredentialError::NotFound) => {}
                        Err(e) => warn!("Could not drop the stale file copy of '{key}': {e}"),
                    }
                }
                Ok(())
            }
            Err(e) => {
                warn!("Keyring set error for '{}': {}", key, e);
                // Try file fallback
                if self.allow_file_fallback {
                    self.set_to_file(key, value)
                } else {
                    Err(e.into())
                }
            }
        }
    }
    
    /// Delete a credential from the keyring
    ///
    /// # Errors
    ///
    /// Returns [`CredentialError::NotFound`] when the key is not stored, and
    /// [`CredentialError::Keyring`] when the keyring entry cannot be opened or the
    /// keyring delete fails for a reason other than a missing entry. When the
    /// keyring has no entry (with fallback enabled), or the store is file-only,
    /// the file store's errors are returned instead.
    /// The encrypted file store fails with [`CredentialError::NoHomeDir`] when no
    /// data directory can be resolved, [`CredentialError::Io`] or
    /// [`CredentialError::Json`] when its file (or a legacy plaintext file being
    /// migrated) cannot be read, written or parsed, and
    /// [`CredentialError::Crypto`] when it cannot be decrypted or encrypted.
    pub fn delete(&self, key: &str) -> Result<(), CredentialError> {
        debug_assert!(!key.is_empty(), "credential key must not be empty");
        // File-only stores never touch the keyring.
        if self.file_only {
            return self.delete_from_file(key);
        }
        let Some(entry) = self.open_entry(key)? else {
            return self.delete_from_file(key);
        };
        match entry.delete_credential() {
            Ok(()) => {
                info!("Deleted credential '{}' from keyring", key);
                // Also delete from file fallback if it exists
                if self.allow_file_fallback {
                    let _ = self.delete_from_file(key);
                }
                Ok(())
            }
            Err(keyring::Error::NoEntry) => {
                // Try to delete from file
                if self.allow_file_fallback {
                    self.delete_from_file(key)
                } else {
                    Err(CredentialError::NotFound)
                }
            }
            Err(e) => Err(e.into()),
        }
    }
    
    /// Check if a credential exists
    #[must_use]
    pub fn exists(&self, key: &str) -> bool {
        self.get(key).is_ok()
    }
    
    /// List all stored credential keys (keyring doesn't support listing, so we check known keys)
    #[must_use]
    pub fn list_keys(&self) -> Vec<String> {
        let known_keys = [
            keys::ANTHROPIC_API_KEY,
            keys::ANTHROPIC_OAUTH_TOKEN,
            keys::ANTHROPIC_OAUTH_CREDENTIAL,
            keys::OPENAI_API_KEY,
            keys::OPENROUTER_API_KEY,
            keys::GITHUB_TOKEN,
            keys::BRAVE_API_KEY,
            keys::OLLAMA_API_KEY,
            keys::TELEGRAM_BOT_TOKEN,
            keys::TELEGRAM_WEBHOOK_SECRET,
            keys::DISCORD_BOT_TOKEN,
            keys::SLACK_BOT_TOKEN,
            keys::SLACK_APP_TOKEN,
            keys::SLACK_SIGNING_SECRET,
            keys::SIGNAL_WEBHOOK_SECRET,
            keys::WHATSAPP_ACCESS_TOKEN,
            keys::WHATSAPP_VERIFY_TOKEN,
            keys::WHATSAPP_APP_SECRET,
            keys::ELEVENLABS_API_KEY,
            keys::SERVER_WEBHOOK_SECRET,
        ];

        known_keys
            .iter()
            .filter(|k| self.exists(k))
            .map(std::string::ToString::to_string)
            .collect()
    }

    // =========================================================================
    // Anthropic OAuth credential (durable home for GUI/CLI OAuth logins)
    // =========================================================================

    /// Durably persist an Anthropic OAuth login.
    ///
    /// Writes the full JSON envelope (access + refresh token + expiry) under
    /// [`keys::ANTHROPIC_OAUTH_CREDENTIAL`] and the bare access token under
    /// [`keys::ANTHROPIC_OAUTH_TOKEN`] — the latter is what
    /// `Config::load_secrets_from_store` hydrates `llm.anthropic_oauth_token`
    /// from, so both keys must stay in lockstep.
    ///
    /// # Errors
    /// Returns an error when the credential cannot be serialized or the
    /// backing store rejects the write.
    pub fn save_anthropic_oauth(&self, cred: &OAuthCredential) -> Result<(), CredentialError> {
        debug_assert!(
            !cred.access_token.is_empty(),
            "OAuth access token must not be empty"
        );
        let json = serde_json::to_string(cred)?;
        self.set(keys::ANTHROPIC_OAUTH_CREDENTIAL, &json)?;
        self.set(keys::ANTHROPIC_OAUTH_TOKEN, &cred.access_token)
    }

    /// Load the stored Anthropic OAuth credential.
    ///
    /// Prefers the JSON envelope (has refresh token + expiry). Falls back to
    /// the bare access-token key — a token pasted from `claude setup-token`
    /// has no refresh token or expiry, and an unreadable envelope must not
    /// take a still-valid token down with it.
    ///
    /// # Errors
    /// Returns [`CredentialError::NotFound`] when neither key is stored, or
    /// the backing store's error when it cannot be read.
    pub fn load_anthropic_oauth(&self) -> Result<OAuthCredential, CredentialError> {
        match self.get(keys::ANTHROPIC_OAUTH_CREDENTIAL) {
            Ok(json) => match serde_json::from_str::<OAuthCredential>(&json) {
                Ok(cred) => return Ok(cred),
                Err(e) => warn!("Stored OAuth envelope unreadable ({e}); trying bare token"),
            },
            Err(CredentialError::NotFound) => {}
            Err(e) => warn!("Could not read OAuth envelope ({e}); trying bare token"),
        }
        let access_token = self.get(keys::ANTHROPIC_OAUTH_TOKEN)?;
        Ok(OAuthCredential {
            access_token,
            refresh_token: None,
            expires_at: None,
            subscription_type: None,
            account_id: None,
            organization_id: None,
        })
    }

    /// Remove every stored Anthropic OAuth key (logout). Absent keys are not
    /// an error — logout of a logged-out store is a no-op, not a failure.
    ///
    /// # Errors
    /// Returns the backing store's error when a present key cannot be removed.
    pub fn delete_anthropic_oauth(&self) -> Result<(), CredentialError> {
        for key in [
            keys::ANTHROPIC_OAUTH_CREDENTIAL,
            keys::ANTHROPIC_OAUTH_TOKEN,
        ] {
            match self.delete(key) {
                Ok(()) | Err(CredentialError::NotFound) => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    // =========================================================================
    // Ollama bearer token, bound to the server it was saved for
    // =========================================================================

    /// Save the Ollama bearer token for the server at `host`; a blank token
    /// removes the saved one ([`Self::delete_ollama_token`]).
    ///
    /// The steps are ordered so that no failure part-way leaves a token
    /// recorded for a server it was not saved for: the old token goes first,
    /// then the server is recorded, then the new token is written. A failure
    /// leaves no token at all — the error says so and saving again finishes
    /// the job — never the previous server's token filed under this one.
    ///
    /// # Errors
    /// Returns the backing store's error for the step that failed.
    pub fn save_ollama_token(&self, token: &str, host: &str) -> Result<(), CredentialError> {
        let token = token.trim();
        if token.is_empty() {
            return self.delete_ollama_token();
        }
        let host = crate::ollama::normalize_ollama_host(host);
        match self.delete(keys::OLLAMA_API_KEY) {
            Ok(()) | Err(CredentialError::NotFound) => {}
            Err(e) => return Err(e),
        }
        self.set(keys::OLLAMA_API_KEY_HOST, &host)?;
        self.set(keys::OLLAMA_API_KEY, token)
    }

    /// Remove the Ollama bearer token and the record of its server. Absent
    /// keys are not an error. The token goes first: a record left behind
    /// without one binds nothing.
    ///
    /// # Errors
    /// Returns the backing store's error when a present key cannot be removed.
    pub fn delete_ollama_token(&self) -> Result<(), CredentialError> {
        for key in [keys::OLLAMA_API_KEY, keys::OLLAMA_API_KEY_HOST] {
            match self.delete(key) {
                Ok(()) | Err(CredentialError::NotFound) => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    /// The saved Ollama bearer token, trimmed; `None` when there is none or
    /// it is blank.
    #[must_use]
    pub fn ollama_token(&self) -> Option<String> {
        self.get(keys::OLLAMA_API_KEY)
            .ok()
            .map(|token| token.trim().to_string())
            .filter(|token| !token.is_empty())
    }

    /// The server the saved Ollama token was saved for; `None` when no server
    /// is recorded (no token, or one saved by an older build).
    ///
    /// A blank record is still a record: it was written while the configured
    /// address was blank, names no server, and so matches none.
    ///
    /// # Errors
    /// Returns the store's error when it cannot say whether a server is
    /// recorded. That is not the same as none being recorded: a token with no
    /// record goes to the configured server, so a record that exists but
    /// could not be read must not pass for an absent one.
    pub fn ollama_token_host(&self) -> Result<Option<String>, CredentialError> {
        Ok(self
            .lookup(keys::OLLAMA_API_KEY_HOST)?
            .map(|host| crate::ollama::normalize_ollama_host(&host)))
    }

    /// Record `host` as the saved token's server when the token has none.
    ///
    /// Called with the address that is about to be replaced: a token saved by
    /// an older build counts as the configured server's, and unrecorded it
    /// would count as the next address's too. Returns whether it recorded.
    ///
    /// # Errors
    /// Returns the backing store's error when the record cannot be written,
    /// or when the store cannot say whether one is already there — nothing is
    /// written then, so a record that exists is never overwritten.
    pub fn bind_unbound_ollama_token(&self, host: &str) -> Result<bool, CredentialError> {
        // A blank host names no server; binding to it would retire the token.
        if host.trim().is_empty()
            || self.ollama_token().is_none()
            || self.ollama_token_host()?.is_some()
        {
            return Ok(false);
        }
        self.set(
            keys::OLLAMA_API_KEY_HOST,
            &crate::ollama::normalize_ollama_host(host),
        )?;
        Ok(true)
    }

    /// Read `key`, telling "not stored" apart from "could not be read".
    ///
    /// [`Self::get`] answers `NotFound` for a keyring that fails (locked, its
    /// service gone) whenever the file fallback lacks the key too. For a
    /// credential that is harmless — it is missing either way — but not for
    /// a record whose absence means something. `Ok(None)` here is only ever
    /// the definite answer of the store that holds the key.
    ///
    /// # Errors
    /// Returns the keyring's error when it could not answer and the file
    /// store holds no copy, and the file store's own errors.
    pub(crate) fn lookup(&self, key: &str) -> Result<Option<String>, CredentialError> {
        let from_file = || match self.get_from_file(key) {
            Ok(value) => Ok(Some(value)),
            Err(CredentialError::NotFound) => Ok(None),
            Err(e) => Err(e),
        };
        if self.file_only {
            return from_file();
        }
        let Some(entry) = self.open_entry(key)? else {
            return from_file();
        };
        match entry.get_password() {
            Ok(value) => Ok(Some(value)),
            Err(keyring::Error::NoEntry) if self.allow_file_fallback => from_file(),
            Err(keyring::Error::NoEntry) => Ok(None),
            Err(e) => match self.allow_file_fallback.then(from_file) {
                Some(Ok(Some(value))) => Ok(Some(value)),
                _ => Err(e.into()),
            },
        }
    }

    /// Open the keyring entry for `key`, or `None` when there is no keyring
    /// to open it in and the file fallback may stand in.
    ///
    /// Opening fails *before* any read or write when no keyring backend
    /// exists at all — measured 2026-09-18: with no Secret Service on the
    /// session bus, `keyring` 4 answers `NoDefaultStore` from `Entry::new`.
    /// That is exactly the machine the encrypted file fallback exists for
    /// (a headless server, a service without a login session), so it must
    /// not end the operation.
    ///
    /// # Errors
    ///
    /// Returns [`CredentialError::Keyring`] when the entry cannot be opened
    /// and the store is keyring-only.
    fn open_entry(&self, key: &str) -> Result<Option<Entry>, CredentialError> {
        debug_assert!(!self.file_only, "file-only stores never open the keyring");
        match Entry::new(KEYRING_SERVICE, key) {
            Ok(entry) => Ok(Some(entry)),
            Err(e) if self.allow_file_fallback => {
                debug!("No keyring for '{}' ({}); using the encrypted file", key, e);
                Ok(None)
            }
            Err(e) => Err(e.into()),
        }
    }

    // =========================================================================
    // File Fallback (for systems without keyring support)
    // =========================================================================

    fn credentials_file_path(&self) -> Result<PathBuf, CredentialError> {
        if let Some(dir) = &self.file_dir {
            return Ok(dir.join("credentials.enc"));
        }
        // Canonical identity shared with config/data (com.nanna.nanna).
        let data_dir = crate::project_dirs()
            .ok_or(CredentialError::NoHomeDir)?
            .data_dir()
            .to_path_buf();
        Ok(data_dir.join("credentials.enc"))
    }

    fn legacy_plaintext_path(&self) -> Result<PathBuf, CredentialError> {
        if let Some(dir) = &self.file_dir {
            return Ok(dir.join("credentials.json"));
        }
        let data_dir = crate::project_dirs()
            .ok_or(CredentialError::NoHomeDir)?
            .data_dir()
            .to_path_buf();
        Ok(data_dir.join("credentials.json"))
    }

    fn load_file_credentials(&self) -> Result<HashMap<String, String>, CredentialError> {
        self.migrate_plaintext_file_if_needed()?;
        let path = self.credentials_file_path()?;
        if !path.exists() {
            return Ok(HashMap::new());
        }
        let bytes = std::fs::read(&path)?;
        decrypt_credentials(&bytes, &self.file_encryption_key()?)
    }

    fn save_file_credentials(
        &self,
        creds: &HashMap<String, String>,
    ) -> Result<(), CredentialError> {
        let path = self.credentials_file_path()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let bytes = encrypt_credentials(creds, &self.file_encryption_key()?)?;
        let tmp = path.with_extension("enc.tmp");
        {
            // 0600 from the first byte, like the key file: the temp copy is
            // the store's full contents until the rename.
            let mut options = std::fs::OpenOptions::new();
            options.write(true).create(true).truncate(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            let mut f = options.open(&tmp)?;
            f.write_all(&bytes)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, &path)?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&path)?.permissions();
            perms.set_mode(0o600);
            std::fs::set_permissions(&path, perms)?;
        }
        if let Ok(legacy) = self.legacy_plaintext_path() {
            let _ = std::fs::remove_file(legacy);
        }
        Ok(())
    }

    /// One-shot: encrypt a leftover plaintext `credentials.json` and delete it.
    fn migrate_plaintext_file_if_needed(&self) -> Result<(), CredentialError> {
        let legacy = self.legacy_plaintext_path()?;
        let modern = self.credentials_file_path()?;
        if !legacy.exists() || modern.exists() {
            return Ok(());
        }
        let content = std::fs::read_to_string(&legacy)?;
        let creds: HashMap<String, String> = serde_json::from_str(&content)?;
        self.save_file_credentials(&creds)?;
        info!("Migrated plaintext credentials.json to AES-GCM credentials.enc");
        Ok(())
    }

    /// Resolve the 32-byte AES-256 key for `credentials.enc`.
    ///
    /// Two places can hold it — the OS keyring (`nanna/file-encryption-key`)
    /// and a 0600 `credentials.key` beside the store — and the old rule
    /// ("keyring, else file") picked by *availability*, so the source flipped:
    /// one keyring hiccup generated a new key file, the existing store stopped
    /// decrypting, and the next `set` re-encrypted it under the new key,
    /// orphaning every credential the old one held. The key is now the one
    /// that already OPENS the store ([`choose_file_key`]); a new one is only
    /// ever made for a store that does not exist yet.
    fn file_encryption_key(&self) -> Result<[u8; 32], CredentialError> {
        let key_path = self
            .credentials_file_path()?
            .with_file_name("credentials.key");
        let file_key = match std::fs::read(&key_path) {
            Ok(bytes) => <[u8; 32]>::try_from(bytes.as_slice()).ok(),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(e.into()),
        };
        // In file-only mode (tests/headless) never touch the OS keyring: the
        // whole point is determinism.
        let keyring = if self.file_only {
            KeyringKey::Absent
        } else {
            read_keyring_file_key()
        };
        let store_path = self.credentials_file_path()?;
        let ciphertext = match std::fs::read(&store_path) {
            Ok(bytes) => Some(bytes),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(e.into()),
        };
        match choose_file_key(file_key, &keyring, ciphertext.as_deref()) {
            KeyChoice::Use(key) => return Ok(key),
            KeyChoice::Refuse(why) => return Err(CredentialError::Crypto(why)),
            KeyChoice::Generate => {}
        }
        if !self.file_only
            && let Ok(entry) = Entry::new(KEYRING_SERVICE, "file-encryption-key")
        {
            let key = random_key()?;
            if entry.set_password(&base64_encode(&key)).is_ok() {
                return Ok(key);
            }
        }
        let key = random_key()?;
        if let Some(parent) = key_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // Created 0600 from the first byte: creating it with the default mode
        // and narrowing afterwards leaves a window where the key is readable.
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        {
            let mut f = options.open(&key_path)?;
            f.write_all(&key)?;
            f.sync_all()?;
        }
        #[cfg(unix)]
        {
            // A pre-existing file keeps its mode through `open`; narrow it too.
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&key_path)?.permissions();
            perms.set_mode(0o600);
            std::fs::set_permissions(&key_path, perms)?;
        }
        Ok(key)
    }
    
    fn get_from_file(&self, key: &str) -> Result<String, CredentialError> {
        let creds = self.load_file_credentials()?;
        creds.get(key).cloned().ok_or(CredentialError::NotFound)
    }

    fn set_to_file(&self, key: &str, value: &str) -> Result<(), CredentialError> {
        let mut creds = self.load_file_credentials()?;
        creds.insert(key.to_string(), value.to_string());
        self.save_file_credentials(&creds)?;
        info!("Stored credential '{}' in file fallback", key);
        Ok(())
    }

    fn delete_from_file(&self, key: &str) -> Result<(), CredentialError> {
        let mut creds = self.load_file_credentials()?;
        if creds.remove(key).is_some() {
            self.save_file_credentials(&creds)?;
            Ok(())
        } else {
            Err(CredentialError::NotFound)
        }
    }
}

// =============================================================================
// Claude CLI OAuth Credentials (Read-Only)
// =============================================================================

/// OAuth credential with refresh support (from Claude CLI)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthCredential {
    /// OAuth access token
    pub access_token: String,
    /// OAuth refresh token (for token renewal)
    pub refresh_token: Option<String>,
    /// Token expiration timestamp (milliseconds since epoch)
    pub expires_at: Option<i64>,
    /// Subscription type (e.g., "pro", "max", "free")
    pub subscription_type: Option<String>,
    /// Account ID
    pub account_id: Option<String>,
    /// Organization ID (for enterprise)
    pub organization_id: Option<String>,
}

impl OAuthCredential {
    /// Check if the token is expired (with 5-minute buffer)
    #[must_use]
    pub fn is_expired(&self) -> bool {
        // No expiry info - assume valid
        self.expires_at.is_some_and(|expires_at| {
            let now = chrono::Utc::now().timestamp_millis();
            // Add 5-minute buffer for safety
            expires_at < now + 5 * 60 * 1000
        })
    }

    /// Check if the token can be refreshed
    #[must_use]
    pub const fn can_refresh(&self) -> bool {
        self.refresh_token.is_some()
    }

    /// Get time until expiration in seconds (negative if expired)
    #[must_use]
    pub fn seconds_until_expiry(&self) -> Option<i64> {
        self.expires_at.map(|expires_at| {
            let now = chrono::Utc::now().timestamp_millis();
            (expires_at - now) / 1000
        })
    }
}

/// Claude CLI credentials file structure
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClaudeCredentialsFile {
    #[serde(rename = "claudeAiOauth")]
    claude_ai_oauth: Option<ClaudeOAuthData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClaudeOAuthData {
    #[serde(rename = "accessToken")]
    access_token: String,
    #[serde(rename = "refreshToken")]
    refresh_token: Option<String>,
    #[serde(rename = "expiresAt")]
    expires_at: Option<i64>,
    #[serde(rename = "subscriptionType")]
    subscription_type: Option<String>,
    #[serde(rename = "accountId")]
    account_id: Option<String>,
    #[serde(rename = "organizationId")]
    organization_id: Option<String>,
}

impl From<ClaudeOAuthData> for OAuthCredential {
    fn from(data: ClaudeOAuthData) -> Self {
        Self {
            access_token: data.access_token,
            refresh_token: data.refresh_token,
            expires_at: data.expires_at,
            subscription_type: data.subscription_type,
            account_id: data.account_id,
            organization_id: data.organization_id,
        }
    }
}

impl From<OAuthCredential> for ClaudeOAuthData {
    fn from(cred: OAuthCredential) -> Self {
        Self {
            access_token: cred.access_token,
            refresh_token: cred.refresh_token,
            expires_at: cred.expires_at,
            subscription_type: cred.subscription_type,
            account_id: cred.account_id,
            organization_id: cred.organization_id,
        }
    }
}

/// Credential source (where credentials were loaded from)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CredentialSource {
    /// Loaded from ~/.claude/.credentials.json
    File,
    /// Loaded from macOS Keychain
    MacOsKeychain,
    /// Loaded from Windows Credential Manager (via keyring)
    WindowsCredentialManager,
    /// Loaded from Linux Secret Service (via keyring)
    LinuxSecretService,
}

/// Result of loading credentials
#[derive(Debug, Clone)]
pub struct LoadedCredential {
    /// The credential
    pub credential: OAuthCredential,
    /// Where it was loaded from
    pub source: CredentialSource,
}

/// Claude CLI credential manager (for OAuth token reading)
#[derive(Debug, Clone)]
pub struct ClaudeCredentialManager {
    /// Home directory override (for testing)
    home_dir: Option<PathBuf>,
}

impl Default for ClaudeCredentialManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ClaudeCredentialManager {
    /// Create a new credential manager
    #[must_use]
    pub const fn new() -> Self {
        Self { home_dir: None }
    }

    /// Create with a custom home directory (for testing)
    #[must_use]
    pub const fn with_home_dir(home_dir: PathBuf) -> Self {
        Self {
            home_dir: Some(home_dir),
        }
    }

    /// Get the home directory
    fn get_home_dir(&self) -> Result<PathBuf, CredentialError> {
        if let Some(ref home) = self.home_dir {
            return Ok(home.clone());
        }

        // Try USERPROFILE (Windows) first, then HOME (Unix)
        std::env::var("USERPROFILE")
            .or_else(|_| std::env::var("HOME"))
            .map(PathBuf::from)
            .map_err(|_| CredentialError::NoHomeDir)
    }

    /// Get path to Claude credentials file
    fn credentials_path(&self) -> Result<PathBuf, CredentialError> {
        Ok(self.get_home_dir()?.join(".claude").join(".credentials.json"))
    }

    /// Load credentials from all available sources
    ///
    /// Priority:
    /// 1. macOS Keychain (if on macOS)
    /// 2. Windows Credential Manager (if on Windows, via keyring)
    /// 3. Linux Secret Service (if on Linux, via keyring)
    /// 4. Credentials file
    ///
    /// # Errors
    ///
    /// Any keyring failure falls through to the file, so only the file's errors
    /// are returned: see [`Self::load_from_file`].
    pub fn load(&self) -> Result<LoadedCredential, CredentialError> {
        // Try platform-specific secure storage via keyring first
        if let Ok(cred) = Self::load_from_keyring() {
            let source = if cfg!(target_os = "macos") {
                CredentialSource::MacOsKeychain
            } else if cfg!(target_os = "windows") {
                CredentialSource::WindowsCredentialManager
            } else {
                CredentialSource::LinuxSecretService
            };
            info!("Loaded Claude credentials from {:?}", source);
            return Ok(LoadedCredential {
                credential: cred,
                source,
            });
        }

        // Fall back to file
        let cred = self.load_from_file()?;
        info!("Loaded Claude credentials from file");
        Ok(LoadedCredential {
            credential: cred,
            source: CredentialSource::File,
        })
    }

    /// Load credentials from the keyring (cross-platform)
    fn load_from_keyring() -> Result<OAuthCredential, CredentialError> {
        let entry = Entry::new("Claude Code-credentials", "Claude Code")?;
        let json_str = entry.get_password()?;
        let data: ClaudeCredentialsFile = serde_json::from_str(&json_str)?;
        let oauth_data = data.claude_ai_oauth.ok_or(CredentialError::NotFound)?;
        Ok(oauth_data.into())
    }

    /// Load credentials from the file
    ///
    /// # Errors
    ///
    /// Returns [`CredentialError::NoHomeDir`] when neither `USERPROFILE` nor
    /// `HOME` is set (and no home override was given),
    /// [`CredentialError::NotFound`] when `~/.claude/.credentials.json` does not
    /// exist or has no `claudeAiOauth` entry, [`CredentialError::Io`] when it
    /// cannot be read, and [`CredentialError::Json`] when it is not valid JSON of
    /// the expected shape.
    pub fn load_from_file(&self) -> Result<OAuthCredential, CredentialError> {
        let path = self.credentials_path()?;

        if !path.exists() {
            debug!("Claude credentials file not found at {:?}", path);
            return Err(CredentialError::NotFound);
        }

        let content = std::fs::read_to_string(&path)?;
        let creds: ClaudeCredentialsFile = serde_json::from_str(&content)?;

        let oauth_data = creds.claude_ai_oauth.ok_or(CredentialError::NotFound)?;

        Ok(oauth_data.into())
    }

    /// Save credentials to file
    ///
    /// # Errors
    ///
    /// Returns [`CredentialError::NoHomeDir`] when neither `USERPROFILE` nor
    /// `HOME` is set (and no home override was given), [`CredentialError::Io`]
    /// when the `.claude` directory cannot be created or the credentials file
    /// cannot be read or written, and [`CredentialError::Json`] when the updated
    /// file cannot be serialized. An existing file that does not parse is
    /// replaced rather than reported.
    pub fn save_to_file(&self, credential: &OAuthCredential) -> Result<(), CredentialError> {
        let path = self.credentials_path()?;

        // Create directory if it doesn't exist
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Read existing file to preserve other fields
        let mut creds: ClaudeCredentialsFile = if path.exists() {
            let content = std::fs::read_to_string(&path)?;
            serde_json::from_str(&content).unwrap_or(ClaudeCredentialsFile {
                claude_ai_oauth: None,
            })
        } else {
            ClaudeCredentialsFile {
                claude_ai_oauth: None,
            }
        };

        // Update OAuth data
        creds.claude_ai_oauth = Some(credential.clone().into());

        // Write back
        let content = serde_json::to_string_pretty(&creds)?;
        std::fs::write(&path, content)?;

        info!("Saved Claude credentials to {:?}", path);
        Ok(())
    }

    /// Save credentials to keyring
    ///
    /// # Errors
    ///
    /// Returns [`CredentialError::Keyring`] when the Claude Code keyring entry
    /// cannot be opened or written, and [`CredentialError::Json`] when the
    /// credential cannot be serialized.
    pub fn save_to_keyring(&self, credential: &OAuthCredential) -> Result<(), CredentialError> {
        let entry = Entry::new("Claude Code-credentials", "Claude Code")?;
        
        let data = ClaudeCredentialsFile {
            claude_ai_oauth: Some(credential.clone().into()),
        };
        let json = serde_json::to_string(&data)?;
        
        entry.set_password(&json)?;
        info!("Saved Claude credentials to keyring");
        Ok(())
    }

    /// Save credentials back to the source they were loaded from
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::save_to_file`]: always for
    /// [`CredentialSource::File`], and for keyring sources only after the
    /// keyring write failed and the file fallback failed too.
    pub fn save(&self, credential: &OAuthCredential, source: CredentialSource) -> Result<(), CredentialError> {
        match source {
            CredentialSource::File => self.save_to_file(credential),
            CredentialSource::MacOsKeychain 
            | CredentialSource::WindowsCredentialManager 
            | CredentialSource::LinuxSecretService => {
                // Try keyring first, fall back to file
                if self.save_to_keyring(credential).is_err() {
                    warn!("Failed to save to keyring, falling back to file");
                    self.save_to_file(credential)
                } else {
                    Ok(())
                }
            }
        }
    }

    /// Refresh the OAuth token using Anthropic's token endpoint
    ///
    /// # Errors
    ///
    /// Returns [`CredentialError::RefreshFailed`] when `credential` has no refresh
    /// token, the token request cannot be sent, the endpoint answers a
    /// non-success status (its body is included), or the response is not a
    /// token object.
    pub async fn refresh_token(
        &self,
        credential: &OAuthCredential,
    ) -> Result<OAuthCredential, CredentialError> {
        const TOKEN_URL: &str = "https://console.anthropic.com/v1/oauth/token";

        #[derive(Deserialize)]
        struct TokenResponse {
            access_token: String,
            refresh_token: Option<String>,
            expires_in: Option<i64>,
            #[serde(rename = "subscriptionType")]
            subscription_type: Option<String>,
        }

        let refresh_token = credential
            .refresh_token
            .as_ref()
            .ok_or_else(|| CredentialError::RefreshFailed("No refresh token available".to_string()))?;

        // Build refresh request
        let client = reqwest::Client::new();
        let response = client
            .post(TOKEN_URL)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .form(&[
                ("grant_type", "refresh_token"),
                ("refresh_token", refresh_token.as_str()),
            ])
            .send()
            .await
            .map_err(|e| CredentialError::RefreshFailed(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(CredentialError::RefreshFailed(format!(
                "Token refresh failed with status {status}: {body}"
            )));
        }

        let token_resp: TokenResponse = response
            .json()
            .await
            .map_err(|e| CredentialError::RefreshFailed(e.to_string()))?;

        // Calculate new expiry
        let expires_at = token_resp.expires_in.map(|secs| {
            chrono::Utc::now().timestamp_millis() + secs * 1000
        });

        let new_credential = OAuthCredential {
            access_token: token_resp.access_token,
            refresh_token: token_resp.refresh_token.or_else(|| credential.refresh_token.clone()),
            expires_at,
            subscription_type: token_resp.subscription_type.or_else(|| credential.subscription_type.clone()),
            account_id: credential.account_id.clone(),
            organization_id: credential.organization_id.clone(),
        };

        info!(
            "Token refreshed, expires in {:?} seconds",
            new_credential.seconds_until_expiry()
        );

        Ok(new_credential)
    }

    /// Load credentials, refreshing if expired
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::load`]; [`CredentialError::Expired`] when the
    /// loaded token is expired and has no refresh token; and the errors of
    /// [`Self::refresh_token`] when a refresh is attempted and fails. Failing to
    /// save a refreshed token is only logged.
    pub async fn load_and_refresh(&self) -> Result<LoadedCredential, CredentialError> {
        let loaded = self.load()?;

        if loaded.credential.is_expired() {
            if loaded.credential.can_refresh() {
                info!("Token expired, attempting refresh...");
                let new_credential = self.refresh_token(&loaded.credential).await?;

                // Save the refreshed token
                if let Err(e) = self.save(&new_credential, loaded.source) {
                    warn!("Failed to save refreshed token: {}", e);
                }

                return Ok(LoadedCredential {
                    credential: new_credential,
                    source: loaded.source,
                });
            }
            return Err(CredentialError::Expired);
        }

        Ok(loaded)
    }

    /// Candidate program names for the Claude Code CLI, in resolution order.
    ///
    /// On Windows the CLI ships two ways: the native installer puts
    /// `claude.exe` on PATH (found by plain `claude` via `CreateProcess`'s
    /// implicit `.exe`), while the npm wrapper installs `claude.cmd` (which
    /// `Command::new("claude")` can NOT spawn — batch files need their
    /// extension). Trying only `claude.cmd` made the native install invisible.
    const fn claude_cli_candidates() -> &'static [&'static str] {
        if cfg!(windows) {
            &["claude", "claude.cmd"]
        } else {
            &["claude"]
        }
    }

    /// The first Claude CLI candidate that answers `--version`, if any.
    fn find_claude_cli() -> Option<&'static str> {
        Self::claude_cli_candidates().iter().copied().find(|cmd| {
            std::process::Command::new(cmd)
                .arg("--version")
                .output()
                .is_ok_and(|o| o.status.success())
        })
    }

    /// Check if Claude CLI is installed and available
    #[must_use]
    pub fn is_claude_cli_available() -> bool {
        Self::find_claude_cli().is_some()
    }

    /// Run `claude setup-token` interactively (inherited stdio).
    ///
    /// Suitable for a real terminal where the user can see the CLI's UI and
    /// complete the browser flow. NOTE: `claude setup-token` PRINTS the minted
    /// token for the user to copy (its success screen says to export it as
    /// `CLAUDE_CODE_OAUTH_TOKEN`); it does NOT write
    /// `~/.claude/.credentials.json` — verified against claude 2.1.71 on
    /// Windows. Callers must not assume the CLI credential store was updated.
    ///
    /// # Errors
    /// Returns error if the CLI is not available or the command fails.
    pub fn run_setup_token() -> Result<(), CredentialError> {
        let cmd = Self::find_claude_cli().ok_or_else(|| {
            CredentialError::RefreshFailed("Claude Code CLI not found on PATH".to_string())
        })?;

        let status = std::process::Command::new(cmd)
            .arg("setup-token")
            .status()
            .map_err(|e| {
                CredentialError::RefreshFailed(format!("Failed to run claude setup-token: {e}"))
            })?;

        if status.success() {
            Ok(())
        } else {
            Err(CredentialError::RefreshFailed(format!(
                "claude setup-token failed with exit code: {:?}",
                status.code()
            )))
        }
    }

    /// Run `claude setup-token` with captured output and parse the minted
    /// token out of it.
    ///
    /// This is the GUI-safe variant: a Tauri app on Windows has no console, so
    /// the CLI's terminal UI is invisible there anyway — capturing stdout is
    /// the only way the token (which the CLI only PRINTS, never persists) can
    /// reach us. stdin is explicitly null so the child can never block waiting
    /// for input nanna cannot provide.
    ///
    /// Returns `Ok(Some(credential))` when a token was parsed from the output
    /// (`setup-token` tokens are long-lived and carry no refresh token or
    /// expiry), `Ok(None)` when the command succeeded but printed no
    /// recognizable token.
    ///
    /// # Errors
    /// Returns error if the CLI is not available or exits non-zero (the
    /// stderr tail is included so the user sees the real reason).
    pub fn run_setup_token_captured() -> Result<Option<OAuthCredential>, CredentialError> {
        let cmd = Self::find_claude_cli().ok_or_else(|| {
            CredentialError::RefreshFailed("Claude Code CLI not found on PATH".to_string())
        })?;

        let output = std::process::Command::new(cmd)
            .arg("setup-token")
            .stdin(std::process::Stdio::null())
            .output()
            .map_err(|e| {
                CredentialError::RefreshFailed(format!("Failed to run claude setup-token: {e}"))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let tail: String = stderr
                .chars()
                .rev()
                .take(400)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            return Err(CredentialError::RefreshFailed(format!(
                "claude setup-token failed (exit {:?}): {}",
                output.status.code(),
                tail.trim()
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(
            extract_oauth_token(&stdout).map(|access_token| OAuthCredential {
                access_token,
                refresh_token: None,
                expires_at: None,
                subscription_type: None,
                account_id: None,
                organization_id: None,
            }),
        )
    }
}

/// Resolve a usable Anthropic OAuth credential from every durable source,
/// refreshing a stale one when a refresh token is available.
///
/// Order: nanna's own [`SecureStore`] (where GUI/CLI logins persist) first,
/// then the Claude Code CLI credential store (`claude login`). A credential
/// found in the CLI store is mirrored into the [`SecureStore`] so the next
/// boot no longer depends on the CLI. Fails closed: an expired credential
/// that cannot be refreshed is never returned.
///
/// # Errors
/// Returns the last source's error when no source yields a usable credential.
pub async fn resolve_anthropic_oauth(
    store: &SecureStore,
) -> Result<OAuthCredential, CredentialError> {
    match store.load_anthropic_oauth() {
        Ok(cred) if !cred.is_expired() => return Ok(cred),
        Ok(cred) if cred.can_refresh() => {
            match ClaudeCredentialManager::new().refresh_token(&cred).await {
                Ok(fresh) => {
                    if let Err(e) = store.save_anthropic_oauth(&fresh) {
                        warn!("Refreshed OAuth token but failed to persist it: {e}");
                    }
                    return Ok(fresh);
                }
                Err(e) => warn!("Stored OAuth token expired and refresh failed: {e}"),
            }
        }
        Ok(_) => warn!("Stored OAuth token expired with no refresh token; trying Claude CLI store"),
        Err(CredentialError::NotFound) => {}
        Err(e) => warn!("Could not read stored OAuth credential ({e}); trying Claude CLI store"),
    }

    // `load_and_refresh` also saves a refreshed token back to its CLI source.
    let loaded = ClaudeCredentialManager::new().load_and_refresh().await?;
    if let Err(e) = store.save_anthropic_oauth(&loaded.credential) {
        warn!("Failed to mirror Claude CLI OAuth credential into secure store: {e}");
    }
    Ok(loaded.credential)
}

/// Extract an Anthropic OAuth access token (`sk-ant-oat…`) from CLI output.
///
/// Strips ANSI escape sequences first (the CLI renders through a terminal UI
/// library), then takes the longest run of token characters starting at the
/// literal `sk-ant-oat` prefix. The 40-char floor rejects prose that merely
/// mentions the prefix; real tokens are ~100+ chars. Known limit: a token
/// hard-wrapped mid-string by a narrow terminal won't reassemble — callers
/// treat `None` as "fall back to other credential sources", never as failure.
fn extract_oauth_token(output: &str) -> Option<String> {
    const PREFIX: &str = "sk-ant-oat";
    const MIN_LEN: usize = 40;
    let clean = strip_ansi(output);
    let start = clean.find(PREFIX)?;
    let token: String = clean[start..]
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .collect();
    (token.len() >= MIN_LEN).then_some(token)
}

/// Drop ANSI CSI/OSC escape sequences, keeping printable text.
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\u{1b}' {
            out.push(c);
            continue;
        }
        match chars.peek() {
            // CSI: ESC [ … final byte in @–~
            Some('[') => {
                chars.next();
                for e in chars.by_ref() {
                    if ('\u{40}'..='\u{7e}').contains(&e) {
                        break;
                    }
                }
            }
            // OSC: ESC ] … terminated by BEL or ESC \
            Some(']') => {
                chars.next();
                while let Some(e) = chars.next() {
                    if e == '\u{07}' {
                        break;
                    }
                    if e == '\u{1b}' && chars.peek() == Some(&'\\') {
                        chars.next();
                        break;
                    }
                }
            }
            // Two-char escapes (ESC c, ESC 7, …): drop the next char.
            Some(_) => {
                chars.next();
            }
            None => {}
        }
    }
    out
}


// =============================================================================
// AES-256-GCM helpers
// =============================================================================

/// Envelope: magic (8) || nonce (12) || ciphertext+tag.
const ENC_MAGIC: &[u8; 8] = b"NANNAENC";
const NONCE_LEN: usize = 12;

/// Generate a fresh 32-byte AES-256 file-encryption key, **failing closed** if the
/// OS RNG does.
///
/// The old body fell back to a `SystemTime`-nanos-derived key on `getrandom`
/// failure — only ~30 bits of guessable entropy, so an attacker who obtained
/// `credentials.enc` and knew roughly when the key was created could brute-force
/// it. A getrandom failure is essentially never seen on a real system, and there
/// is no safe weak fallback for a long-lived encryption key, so we refuse to mint
/// one rather than mint a guessable one (matches [`random_nonce`]).
/// What the OS keyring said about the file-encryption key.
#[derive(Debug, Clone, PartialEq, Eq)]
enum KeyringKey {
    /// It holds a well-formed 32-byte key.
    Found([u8; 32]),
    /// It answered, and holds no usable key.
    Absent,
    /// It could not be asked (locked, no Secret Service, a D-Bus failure).
    Unavailable,
}

/// Ask the keyring for the file-encryption key, without ever writing one.
fn read_keyring_file_key() -> KeyringKey {
    let Ok(entry) = Entry::new(KEYRING_SERVICE, "file-encryption-key") else {
        return KeyringKey::Unavailable;
    };
    match entry.get_password() {
        Ok(b64) => base64_decode(&b64)
            .ok()
            .and_then(|bytes| <[u8; 32]>::try_from(bytes.as_slice()).ok())
            .map_or(KeyringKey::Absent, KeyringKey::Found),
        Err(keyring::Error::NoEntry) => KeyringKey::Absent,
        Err(_) => KeyringKey::Unavailable,
    }
}

/// Which file-encryption key to use.
#[derive(Debug, Clone, PartialEq, Eq)]
enum KeyChoice {
    Use([u8; 32]),
    /// No store exists yet, so a new key orphans nothing.
    Generate,
    /// A store exists and no key at hand opens it. Making a new one would
    /// orphan every credential in it, so this is an error, not a fallback.
    Refuse(String),
}

/// Choose the key for `credentials.enc`. Pure.
///
/// The key is the one that already opens the store, whichever of the two
/// places holds it; with no store yet, an existing key is reused (the key
/// file first — it is the one a keyring outage created, and the keyring may be
/// back without ever having held it) and only with no key anywhere is one
/// generated.
fn choose_file_key(
    file_key: Option<[u8; 32]>,
    keyring: &KeyringKey,
    ciphertext: Option<&[u8]>,
) -> KeyChoice {
    let keyring_key = match keyring {
        KeyringKey::Found(key) => Some(*key),
        KeyringKey::Absent | KeyringKey::Unavailable => None,
    };
    let candidates = [file_key, keyring_key];
    let Some(ciphertext) = ciphertext else {
        return candidates
            .into_iter()
            .flatten()
            .next()
            .map_or(KeyChoice::Generate, KeyChoice::Use);
    };
    if let Some(key) = candidates
        .into_iter()
        .flatten()
        .find(|key| decrypt_credentials(ciphertext, key).is_ok())
    {
        return KeyChoice::Use(key);
    }
    KeyChoice::Refuse(if *keyring == KeyringKey::Unavailable {
        "the OS keyring is unavailable and holds the key credentials.enc was written with; \
         not generating a new one, which would orphan every stored credential"
            .to_string()
    } else {
        "no key at hand opens credentials.enc; not generating a new one, which would orphan \
         every stored credential"
            .to_string()
    })
}

fn random_key() -> Result<[u8; 32], CredentialError> {
    let mut key = [0u8; 32];
    getrandom::fill(&mut key)
        .map_err(|e| CredentialError::Crypto(format!("RNG failure generating file key: {e}")))?;
    Ok(key)
}

/// Generate a fresh random AES-GCM nonce, **failing closed** if the OS RNG does.
///
/// The old body ignored `getrandom`'s error (`let _ = …`), so an RNG failure left
/// the nonce all-zeros. Under AES-GCM a repeated nonce with the same key is
/// catastrophic — it breaks confidentiality and forges authentication — and every
/// `encrypt_credentials` call reuses the file key, so a zero nonce means guaranteed
/// reuse. A getrandom failure is essentially never seen on a real system, so
/// refusing to encrypt is the correct, conservative response (unlike a key, a nonce
/// has no safe weak fallback: uniqueness is the whole contract).
fn random_nonce() -> Result<[u8; NONCE_LEN], CredentialError> {
    let mut n = [0u8; NONCE_LEN];
    getrandom::fill(&mut n)
        .map_err(|e| CredentialError::Crypto(format!("RNG failure generating nonce: {e}")))?;
    Ok(n)
}

fn encrypt_credentials(
    creds: &HashMap<String, String>,
    key: &[u8; 32],
) -> Result<Vec<u8>, CredentialError> {
    let plaintext = serde_json::to_vec(creds)?;
    let cipher = Aes256Gcm::new_from_slice(key)
        .map_err(|e| CredentialError::Crypto(e.to_string()))?;
    let nonce_bytes = random_nonce()?;
    let nonce = Nonce::try_from(nonce_bytes.as_slice())
        .map_err(|_| CredentialError::Crypto("nonce length mismatch".into()))?;
    let ciphertext = cipher
        .encrypt(&nonce, plaintext.as_ref())
        .map_err(|e| CredentialError::Crypto(e.to_string()))?;
    let mut out = Vec::with_capacity(8 + NONCE_LEN + ciphertext.len());
    out.extend_from_slice(ENC_MAGIC);
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ciphertext);
    Ok(out)
}

fn decrypt_credentials(
    bytes: &[u8],
    key: &[u8; 32],
) -> Result<HashMap<String, String>, CredentialError> {
    if bytes.len() < 8 + NONCE_LEN + 16 {
        return Err(CredentialError::Crypto("envelope too short".into()));
    }
    if &bytes[..8] != ENC_MAGIC {
        return Err(CredentialError::Crypto("bad envelope magic".into()));
    }
    let nonce = Nonce::try_from(&bytes[8..8 + NONCE_LEN])
        .map_err(|_| CredentialError::Crypto("nonce length mismatch".into()))?;
    let ciphertext = &bytes[8 + NONCE_LEN..];
    let cipher = Aes256Gcm::new_from_slice(key)
        .map_err(|e| CredentialError::Crypto(e.to_string()))?;
    let mut plaintext = cipher.decrypt(&nonce, ciphertext).map_err(|_| {
        CredentialError::Crypto("decryption failed (wrong key or corrupt file)".into())
    })?;
    let creds: HashMap<String, String> = serde_json::from_slice(&plaintext)?;
    plaintext.zeroize();
    Ok(creds)
}

fn base64_encode(bytes: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn base64_decode(s: &str) -> Result<Vec<u8>, ()> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.decode(s).map_err(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// The key-source flip: a store written under one key must keep being
    /// opened by that key whichever place holds it, and a store no key opens is
    /// refused rather than silently re-keyed — which would orphan it.
    #[test]
    fn the_file_key_is_the_one_that_opens_the_store() {
        let (a, b) = ([1u8; 32], [2u8; 32]);
        let creds = HashMap::from([("anthropic".to_string(), "sk-live".to_string())]);
        let under_a = encrypt_credentials(&creds, &a).expect("encrypt");
        let under_b = encrypt_credentials(&creds, &b).expect("encrypt");

        // A keyring outage once created a key file; the store is still under
        // the keyring's key. The file key must not win just by existing.
        assert_eq!(
            choose_file_key(Some(b), &KeyringKey::Found(a), Some(&under_a)),
            KeyChoice::Use(a)
        );
        // …and the other way round: the store moved to the file key.
        assert_eq!(
            choose_file_key(Some(b), &KeyringKey::Found(a), Some(&under_b)),
            KeyChoice::Use(b)
        );
        // The keyring is back and never held the key: the file key opens it.
        assert_eq!(
            choose_file_key(Some(b), &KeyringKey::Absent, Some(&under_b)),
            KeyChoice::Use(b)
        );
        // The keyring holds the only key and is locked right now: refuse,
        // never generate (the old code minted a file key here).
        assert!(matches!(
            choose_file_key(None, &KeyringKey::Unavailable, Some(&under_a)),
            KeyChoice::Refuse(why) if why.contains("unavailable")
        ));
        assert!(matches!(
            choose_file_key(Some(b), &KeyringKey::Absent, Some(&under_a)),
            KeyChoice::Refuse(_)
        ));
        // No store yet: reuse a key that exists, else make one.
        assert_eq!(
            choose_file_key(Some(b), &KeyringKey::Found(a), None),
            KeyChoice::Use(b)
        );
        assert_eq!(
            choose_file_key(None, &KeyringKey::Found(a), None),
            KeyChoice::Use(a)
        );
        assert_eq!(
            choose_file_key(None, &KeyringKey::Unavailable, None),
            KeyChoice::Generate
        );
    }

    #[test]
    fn test_credential_expiry() {
        let cred = OAuthCredential {
            access_token: "test".to_string(),
            refresh_token: Some("refresh".to_string()),
            expires_at: Some(chrono::Utc::now().timestamp_millis() + 3600 * 1000),
            subscription_type: None,
            account_id: None,
            organization_id: None,
        };

        assert!(!cred.is_expired());
        assert!(cred.can_refresh());
    }

    #[test]
    fn random_nonce_is_nonzero_and_unique() {
        // The old body swallowed getrandom errors and could yield an all-zeros
        // nonce; AES-GCM nonce reuse is catastrophic. Prove the happy path gives a
        // fresh, non-zero nonce each call.
        let a = random_nonce().expect("RNG available in tests");
        let b = random_nonce().expect("RNG available in tests");
        assert_ne!(a, [0u8; NONCE_LEN], "a real nonce is never all zeros");
        assert_ne!(
            a, b,
            "two nonces must differ — uniqueness is the GCM contract"
        );
    }

    #[test]
    fn encrypt_uses_a_fresh_nonce_each_call() {
        // Same plaintext + key must produce different envelopes, because the
        // random nonce is prepended — the observable proof a fixed nonce is not
        // being reused — yet both still decrypt back to the original.
        let key = [7u8; 32];
        let mut creds = HashMap::new();
        creds.insert("k".to_string(), "v".to_string());
        let a = encrypt_credentials(&creds, &key).expect("encrypt a");
        let b = encrypt_credentials(&creds, &key).expect("encrypt b");
        assert_ne!(a, b, "distinct nonces must yield distinct envelopes");
        assert_eq!(decrypt_credentials(&a, &key).expect("decrypt a"), creds);
        assert_eq!(decrypt_credentials(&b, &key).expect("decrypt b"), creds);
    }

    #[test]
    fn test_expired_credential() {
        let cred = OAuthCredential {
            access_token: "test".to_string(),
            refresh_token: Some("refresh".to_string()),
            expires_at: Some(chrono::Utc::now().timestamp_millis() - 1000),
            subscription_type: None,
            account_id: None,
            organization_id: None,
        };

        assert!(cred.is_expired());
    }

    #[test]
    fn test_save_and_load_file() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ClaudeCredentialManager::with_home_dir(temp_dir.path().to_path_buf());

        let cred = OAuthCredential {
            access_token: "test_token".to_string(),
            refresh_token: Some("refresh_token".to_string()),
            expires_at: Some(1_234_567_890_000),
            subscription_type: Some("pro".to_string()),
            account_id: None,
            organization_id: None,
        };

        // Save
        manager.save_to_file(&cred).unwrap();

        // Load
        let loaded = manager.load_from_file().unwrap();

        assert_eq!(loaded.access_token, "test_token");
        assert_eq!(loaded.refresh_token, Some("refresh_token".to_string()));
        assert_eq!(loaded.expires_at, Some(1_234_567_890_000));
        assert_eq!(loaded.subscription_type, Some("pro".to_string()));
    }
    
    #[test]
    fn test_secure_store_file_only_roundtrip() {
        // File-only store: deterministic, no dependency on an interactive keyring.
        let temp_dir = TempDir::new().unwrap();
        let store = SecureStore::file_only_at(temp_dir.path().to_path_buf());

        let key = "test_key_nanna_unit_test";
        let value = "test_value_12345";

        // Absent before any write.
        assert!(matches!(store.get(key), Err(CredentialError::NotFound)));

        // Set then get round-trips exactly.
        store.set(key, value).unwrap();
        assert_eq!(store.get(key).unwrap(), value);

        // Envelope exists and does NOT contain the secret in plaintext.
        let enc_path = temp_dir.path().join("credentials.enc");
        assert!(enc_path.exists(), "encrypted credentials file should exist");
        let raw = std::fs::read(&enc_path).unwrap();
        assert!(raw.starts_with(b"NANNAENC"), "envelope magic missing");
        let as_text = String::from_utf8_lossy(&raw);
        assert!(
            !as_text.contains(value),
            "secret leaked into the on-disk envelope in plaintext"
        );
        assert!(
            !temp_dir.path().join("credentials.json").exists(),
            "legacy plaintext file must not be written"
        );

        // Overwrite updates the stored value.
        store.set(key, "updated_value").unwrap();
        assert_eq!(store.get(key).unwrap(), "updated_value");

        // Delete removes it; a second delete reports NotFound.
        store.delete(key).unwrap();
        assert!(matches!(store.get(key), Err(CredentialError::NotFound)));
        assert!(matches!(store.delete(key), Err(CredentialError::NotFound)));
    }

    #[test]
    fn test_migrates_legacy_plaintext_json() {
        let temp_dir = TempDir::new().unwrap();
        let legacy = temp_dir.path().join("credentials.json");
        std::fs::write(&legacy, r#"{"legacy_key":"legacy_secret"}"#).unwrap();
        let store = SecureStore::file_only_at(temp_dir.path().to_path_buf());
        assert_eq!(store.get("legacy_key").unwrap(), "legacy_secret");
        assert!(
            !legacy.exists(),
            "legacy plaintext should be removed after migrate"
        );
        assert!(temp_dir.path().join("credentials.enc").exists());
        // And the envelope must not contain the secret in plaintext.
        let raw = std::fs::read(temp_dir.path().join("credentials.enc")).unwrap();
        assert!(raw.starts_with(b"NANNAENC"));
        assert!(!String::from_utf8_lossy(&raw).contains("legacy_secret"));
    }

    #[test]
    fn oauth_login_survives_restart_roundtrip() {
        // The bug this guards: OAuth logins lived only in the in-memory Config
        // (strip_secrets_for_disk blanks the token on every save), so a restart
        // logged the user out. The durable home is the SecureStore; prove a
        // fresh store instance over the same dir (= app restart) sees the login.
        let temp_dir = TempDir::new().unwrap();
        let cred = OAuthCredential {
            access_token: "sk-ant-oat01-roundtrip-access".to_string(),
            refresh_token: Some("refresh-me".to_string()),
            expires_at: Some(chrono::Utc::now().timestamp_millis() + 3600 * 1000),
            subscription_type: Some("max".to_string()),
            account_id: Some("acct_1".to_string()),
            organization_id: None,
        };

        SecureStore::file_only_at(temp_dir.path().to_path_buf())
            .save_anthropic_oauth(&cred)
            .unwrap();

        // "Restart": a brand-new store instance rooted at the same dir.
        let store = SecureStore::file_only_at(temp_dir.path().to_path_buf());
        let loaded = store.load_anthropic_oauth().unwrap();
        assert_eq!(loaded.access_token, cred.access_token);
        assert_eq!(loaded.refresh_token, cred.refresh_token);
        assert_eq!(loaded.expires_at, cred.expires_at);
        assert_eq!(loaded.subscription_type, cred.subscription_type);
        assert_eq!(loaded.account_id, cred.account_id);

        // The bare-token key (config hydration path) stays in lockstep.
        assert_eq!(
            store.get(keys::ANTHROPIC_OAUTH_TOKEN).unwrap(),
            cred.access_token
        );

        // Logout removes both keys; logging out twice is a no-op, not an error.
        store.delete_anthropic_oauth().unwrap();
        assert!(matches!(
            store.load_anthropic_oauth(),
            Err(CredentialError::NotFound)
        ));
        assert!(matches!(
            store.get(keys::ANTHROPIC_OAUTH_TOKEN),
            Err(CredentialError::NotFound)
        ));
        store.delete_anthropic_oauth().unwrap();
    }

    #[test]
    fn oauth_load_falls_back_to_bare_token() {
        // A token pasted from `claude setup-token` is stored bare (no envelope,
        // no refresh/expiry). Load must synthesize a usable credential from it.
        let temp_dir = TempDir::new().unwrap();
        let store = SecureStore::file_only_at(temp_dir.path().to_path_buf());
        store
            .set(keys::ANTHROPIC_OAUTH_TOKEN, "sk-ant-oat01-bare-token")
            .unwrap();

        let loaded = store.load_anthropic_oauth().unwrap();
        assert_eq!(loaded.access_token, "sk-ant-oat01-bare-token");
        assert_eq!(loaded.refresh_token, None);
        assert!(!loaded.is_expired(), "no expiry info means assumed valid");
        assert!(!loaded.can_refresh());
    }

    #[test]
    fn oauth_load_survives_corrupt_envelope() {
        // A garbled envelope must not take a still-valid bare token down with it.
        let temp_dir = TempDir::new().unwrap();
        let store = SecureStore::file_only_at(temp_dir.path().to_path_buf());
        store
            .set(keys::ANTHROPIC_OAUTH_CREDENTIAL, "{not json")
            .unwrap();
        store
            .set(keys::ANTHROPIC_OAUTH_TOKEN, "sk-ant-oat01-still-good")
            .unwrap();

        let loaded = store.load_anthropic_oauth().unwrap();
        assert_eq!(loaded.access_token, "sk-ant-oat01-still-good");
    }

    #[test]
    fn extract_oauth_token_parses_cli_output() {
        let token = format!("sk-ant-oat01-{}", "A".repeat(90));

        // Plain output.
        assert_eq!(
            extract_oauth_token(&format!("Your token:\n\n  {token}\n\nStore it safely.")),
            Some(token.clone())
        );

        // ANSI-styled output (the CLI renders through a terminal UI library).
        let styled = format!("\u{1b}[1m\u{1b}[32m{token}\u{1b}[0m");
        assert_eq!(extract_oauth_token(&styled), Some(token.clone()));

        // Trailing punctuation/quotes never join the token.
        assert_eq!(
            extract_oauth_token(&format!("token \"{token}\".")),
            Some(token)
        );

        // Prose that merely mentions the prefix is rejected (under the floor).
        assert_eq!(
            extract_oauth_token("tokens look like sk-ant-oat01-..."),
            None
        );
        assert_eq!(extract_oauth_token("no token here"), None);
    }

    #[test]
    fn strip_ansi_removes_csi_and_osc() {
        assert_eq!(strip_ansi("\u{1b}[1;32mhi\u{1b}[0m"), "hi");
        assert_eq!(strip_ansi("\u{1b}]0;title\u{07}body"), "body");
        assert_eq!(strip_ansi("\u{1b}]8;;url\u{1b}\\link"), "link");
        assert_eq!(strip_ansi("plain"), "plain");
    }

    #[test]
    fn test_secure_store_file_only_isolated_dirs() {
        // Two stores rooted at different dirs never see each other's secrets.
        let dir_a = TempDir::new().unwrap();
        let dir_b = TempDir::new().unwrap();
        let store_a = SecureStore::file_only_at(dir_a.path().to_path_buf());
        let store_b = SecureStore::file_only_at(dir_b.path().to_path_buf());

        store_a.set("shared_key", "in_a").unwrap();
        assert_eq!(store_a.get("shared_key").unwrap(), "in_a");
        assert!(matches!(store_b.get("shared_key"), Err(CredentialError::NotFound)));
    }

    #[test]
    fn an_unreadable_token_record_is_not_an_absent_one() {
        // "No server recorded" sends a token to the configured server, so a
        // store that cannot say must not answer "none" — and must not have a
        // record written over one it could not read.
        let dir = TempDir::new().unwrap();
        let store = SecureStore::file_only_at(dir.path().to_path_buf());
        assert!(
            matches!(store.ollama_token_host(), Ok(None)),
            "nothing stored yet"
        );
        store
            .save_ollama_token("token", "https://a.example")
            .unwrap();
        assert_eq!(
            store.ollama_token_host().unwrap().as_deref(),
            Some("https://a.example")
        );

        std::fs::write(dir.path().join("credentials.enc"), b"not an envelope").unwrap();
        assert!(
            store.ollama_token_host().is_err(),
            "unreadable is not absent"
        );
        assert!(
            !store
                .bind_unbound_ollama_token("https://b.example")
                .unwrap_or(false)
        );
        assert_eq!(
            std::fs::read(dir.path().join("credentials.enc")).unwrap(),
            b"not an envelope",
            "nothing was written over it"
        );
    }
}
