//! `credentials` subcommand handlers (Claude CLI OAuth).

use crate::CredentialsAction;
use nanna_config::Config;
use tracing::warn;

/// Print credentials status
fn print_credentials_status(
    manager: &nanna_config::ClaudeCredentialManager,
) {
    use nanna_config::{ClaudeCredentialManager, CredentialSource};

    println!("🔐 Claude CLI Credentials Status\n");

    if ClaudeCredentialManager::is_claude_cli_available() {
        println!("   ✓ Claude CLI installed");
    } else {
        println!("   ✗ Claude CLI not found");
        println!("     Install with: npm install -g @anthropic-ai/claude-code");
    }

    match manager.load() {
        Ok(loaded) => {
            let source = match loaded.source {
                CredentialSource::File => "file (~/.claude/.credentials.json)",
                CredentialSource::MacOsKeychain => "macOS Keychain",
                CredentialSource::WindowsCredentialManager => "Windows Credential Manager",
                CredentialSource::LinuxSecretService => "Linux Secret Service",
            };
            println!("   ✓ Credentials found ({source})");

            if let Some(secs) = loaded.credential.seconds_until_expiry() {
                if secs > 0 {
                    let hours = secs / 3600;
                    let mins = (secs % 3600) / 60;
                    if hours > 0 {
                        println!("   ⏱ Expires in {hours}h {mins}m");
                    } else {
                        println!("   ⏱ Expires in {mins}m");
                    }
                } else {
                    println!("   ⚠ Token expired ({} seconds ago)", -secs);
                    if loaded.credential.can_refresh() {
                        println!("     Run 'nanna credentials refresh' to renew");
                    }
                }
            }

            if let Some(ref sub) = loaded.credential.subscription_type {
                println!("   📋 Subscription: {sub}");
            }

            if loaded.credential.can_refresh() {
                println!("   🔄 Can auto-refresh: yes");
            } else {
                println!("   🔄 Can auto-refresh: no (no refresh token)");
            }
        }
        Err(e) => {
            println!("   ✗ No credentials found: {e}");
            println!("\n   To authenticate:");
            println!("   • Run 'nanna credentials setup' (requires Claude CLI)");
            println!("   • Or run 'claude login' directly, then 'nanna credentials import'");
        }
    }
}

/// Persist an OAuth credential durably and point the config at OAuth mode.
///
/// The `SecureStore` is the durable home — `Config::save` strips secrets from
/// config.toml, so a login that only touches the config dies with the process.
/// The config file as it is, to change a field of and save back — or `None`
/// when it cannot be read, which is said, and nothing is saved.
///
/// This used `Config::load().unwrap_or_default()`: a `config.toml` with one
/// TOML typo loaded as the DEFAULTS, and the save then wrote them over the
/// user's file — provider, models, channels, MCP servers and `data_dir` gone,
/// to set one OAuth field.
fn loaded_config_to_edit() -> Option<Config> {
    config_to_edit(Config::load())
}

fn config_to_edit<E: std::fmt::Display>(loaded: Result<Config, E>) -> Option<Config> {
    match loaded {
        Ok(config) => Some(config),
        Err(e) => {
            warn!("Not rewriting a config file that does not load: {e}");
            println!("⚠ config.toml could not be read ({e}); it was left as it is — fix it and re-run");
            None
        }
    }
}

fn persist_oauth_credential(credential: &nanna_config::OAuthCredential) {
    if let Err(e) = nanna_config::SecureStore::new().save_anthropic_oauth(credential) {
        warn!("Failed to persist OAuth token to secure store: {}", e);
        println!("⚠ Could not persist token to the secure store: {e}");
    }

    let Some(mut config) = loaded_config_to_edit() else {
        return;
    };
    config.llm.anthropic_oauth_token = Some(credential.access_token.clone());
    config.llm.anthropic_use_oauth = true;
    if let Err(e) = config.save() {
        warn!("Failed to save config: {}", e);
    } else {
        println!("   Config updated to use OAuth");
    }
}

/// Import Claude CLI credentials into Nanna config
async fn import_credentials(
    manager: &nanna_config::ClaudeCredentialManager,
) -> anyhow::Result<()> {
    use nanna_config::CredentialSource;

    println!("🔐 Importing Claude CLI Credentials...\n");

    match manager.load() {
        Ok(loaded) => {
            let source = match loaded.source {
                CredentialSource::File => "file",
                CredentialSource::MacOsKeychain => "macOS Keychain",
                CredentialSource::WindowsCredentialManager => "Windows Credential Manager",
                CredentialSource::LinuxSecretService => "Linux Secret Service",
            };

            let credential = if loaded.credential.is_expired() {
                println!("⚠ Warning: Token is expired");
                if loaded.credential.can_refresh() {
                    println!("   Attempting refresh...");
                    match manager.refresh_token(&loaded.credential).await {
                        Ok(new_cred) => {
                            if let Err(e) = manager.save(&new_cred, loaded.source) {
                                warn!("Failed to save refreshed token: {}", e);
                            }
                            println!("✅ Token refreshed and imported from {source}");
                            if let Some(ref sub) = new_cred.subscription_type {
                                println!("   Subscription: {sub}");
                            }
                            new_cred
                        }
                        Err(e) => {
                            println!("❌ Refresh failed: {e}");
                            println!("   Run 'claude login' to re-authenticate");
                            return Ok(());
                        }
                    }
                } else {
                    println!("   Cannot auto-refresh (no refresh token)");
                    println!("   Run 'claude login' to re-authenticate");
                    return Ok(());
                }
            } else {
                println!("✅ Credentials imported from {source}");
                if let Some(ref sub) = loaded.credential.subscription_type {
                    println!("   Subscription: {sub}");
                }
                if let Some(secs) = loaded.credential.seconds_until_expiry() {
                    let hours = secs / 3600;
                    println!("   Expires in: {hours}h");
                }
                loaded.credential
            };

            persist_oauth_credential(&credential);
        }
        Err(e) => {
            println!("❌ No credentials found: {e}");
            println!("\n   Run 'claude login' first, or use 'nanna credentials setup'");
        }
    }

    Ok(())
}

/// Run interactive Claude CLI setup and import credentials
fn setup_credentials(
    manager: &nanna_config::ClaudeCredentialManager,
) {
    use nanna_config::ClaudeCredentialManager;

    println!("🔐 Setting up Claude CLI Authentication...\n");

    if !ClaudeCredentialManager::is_claude_cli_available() {
        println!("❌ Claude CLI not found");
        println!("   Install with: npm install -g @anthropic-ai/claude-code");
        return;
    }

    println!("Running 'claude setup-token'...");
    println!("This will open your browser for authentication.\n");

    if let Err(e) = ClaudeCredentialManager::run_setup_token() {
        println!("❌ Setup failed: {e}");
        return;
    }

    // `claude setup-token` PRINTS the minted token for the user to copy; it
    // does NOT write the CLI credential store. Only a prior `claude login`
    // leaves something for `load()` to find — and it may be stale.
    match manager.load() {
        Ok(loaded) if !loaded.credential.is_expired() => {
            persist_oauth_credential(&loaded.credential);

            println!("\n✅ Authentication complete!");
            if let Some(ref sub) = loaded.credential.subscription_type {
                println!("   Subscription: {sub}");
            }
            println!("   Nanna is now configured to use OAuth");
        }
        Ok(_) => {
            println!("\n⚠ The CLI credential store only has an expired token.");
            println!("   Copy the token that `claude setup-token` printed above and run:");
            println!("   nanna with ANTHROPIC_OAUTH_TOKEN=<token>, or paste it in the GUI settings.");
        }
        Err(e) => {
            println!("\n⚠ Setup completed but couldn't import credentials: {e}");
            println!("   Copy the token that `claude setup-token` printed above and run:");
            println!("   nanna with ANTHROPIC_OAUTH_TOKEN=<token>, or paste it in the GUI settings.");
        }
    }
}

/// Refresh an existing OAuth token
async fn refresh_credentials(
    manager: &nanna_config::ClaudeCredentialManager,
) -> anyhow::Result<()> {
    println!("🔄 Refreshing OAuth Token...\n");

    match manager.load() {
        Ok(loaded) => {
            if !loaded.credential.can_refresh() {
                println!("❌ Cannot refresh: no refresh token available");
                println!("   Run 'nanna credentials setup' to re-authenticate");
                return Ok(());
            }

            match manager.refresh_token(&loaded.credential).await {
                Ok(new_cred) => {
                    if let Err(e) = manager.save(&new_cred, loaded.source) {
                        warn!("Failed to save to original source: {}", e);
                    }

                    persist_oauth_credential(&new_cred);

                    println!("✅ Token refreshed!");
                    if let Some(secs) = new_cred.seconds_until_expiry() {
                        let hours = secs / 3600;
                        println!("   New expiry: {hours}h from now");
                    }
                }
                Err(e) => {
                    println!("❌ Refresh failed: {e}");
                    println!("   You may need to re-authenticate with 'nanna credentials setup'");
                }
            }
        }
        Err(e) => {
            println!("❌ No credentials found: {e}");
        }
    }

    Ok(())
}

/// Clear stored OAuth credentials
fn clear_credentials() {
    println!("🗑 Clearing OAuth Credentials...\n");

    // The SecureStore is the durable home — clearing only the config would
    // log the user right back in at the next launch's hydration.
    if let Err(e) = nanna_config::SecureStore::new().delete_anthropic_oauth() {
        warn!("Failed to remove stored OAuth token: {}", e);
        println!("⚠ Could not remove the token from the secure store: {e}");
    }

    let Some(mut config) = loaded_config_to_edit() else {
        return;
    };
    config.llm.anthropic_oauth_token = None;
    config.llm.anthropic_use_oauth = false;
    if let Err(e) = config.save() {
        warn!("Failed to save config: {}", e);
    } else {
        println!("✅ Cleared OAuth token from Nanna config");
    }

    println!("\n   Note: Claude CLI credentials in ~/.claude/.credentials.json are not modified.");
    println!("   To fully log out, run 'claude logout' as well.");
}

/// Handle credentials subcommands
pub async fn handle_credentials_command(action: CredentialsAction) -> anyhow::Result<()> {
    use nanna_config::ClaudeCredentialManager;

    let manager = ClaudeCredentialManager::new();

    match action {
        CredentialsAction::Status => print_credentials_status(&manager),
        CredentialsAction::Import => import_credentials(&manager).await?,
        CredentialsAction::Setup => setup_credentials(&manager),
        CredentialsAction::Refresh => refresh_credentials(&manager).await?,
        CredentialsAction::Clear => clear_credentials(),
    }

    Ok(())
}

#[cfg(test)]
mod config_edit_tests {
    use super::config_to_edit;
    use nanna_config::Config;

    /// A config that does not load is never replaced by the defaults.
    #[test]
    fn a_config_that_does_not_load_is_not_rewritten() {
        assert!(config_to_edit::<String>(Err("expected `=` at line 3".to_string())).is_none());
        let mut mine = Config::default();
        mine.llm.model = "my-model".to_string();
        let kept = config_to_edit::<String>(Ok(mine)).expect("a loaded config is edited");
        assert_eq!(kept.llm.model, "my-model");
    }
}
