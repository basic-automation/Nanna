//! Config handlers for the [`ControlPlane`].

use super::{Arc, Config, ConfigAction, ControlPlane, Event, Value, info, json, warn};

impl ControlPlane {
    /// Push `[scheduler]` settings onto the **running** scheduler loop.
    ///
    /// The scheduler captures its config when it starts, so a config write
    /// alone would only take effect at the next daemon restart. That is the
    /// wrong latency for the heartbeat switch in particular: the heartbeat runs
    /// a full agent turn on the same model chat uses, so on a single-slot local
    /// backend it cancels an in-flight generation, and "turn it off" has to mean
    /// *now* — before the benchmark run, not after a restart.
    ///
    /// Takes a snapshot rather than reading `self.config`, so the caller can
    /// release its config guard first: the scheduler lock is only ever taken
    /// with no config lock held.
    async fn apply_scheduler_settings(&self, config: &Config) {
        let Some(scheduler) = self.scheduler.as_ref() else {
            return;
        };
        scheduler.write().await.apply_settings(
            config.scheduler.enabled,
            config.scheduler.heartbeat_enabled,
            std::time::Duration::from_secs(nanna_core::clamp_heartbeat_secs(
                config.scheduler.heartbeat_interval_secs,
            )),
        );
    }

    /// Make `new_config` the running configuration and push it everywhere a
    /// live change must reach: the agent, then [`Self::propagate_committed`].
    ///
    /// This is the reload the GUI triggers after saving a credential — the step
    /// that makes a post-boot login actually reach the router. The Scheduler
    /// tab rides the same hop, and so does a hand edit of `config.toml` (see
    /// `config_watch`). What it does NOT re-apply is anything wired only at
    /// boot (channels, MCP servers, the webhook server, and the embedding
    /// server and models — see `embedding_reload`).
    pub(super) async fn apply_loaded_config(&self, new_config: Config) {
        let mut config = self.config.write().await;
        *config = new_config;
        info!("Config reloaded from disk");
        if let Some(ref agent) = self.agent {
            agent.apply_llm_config(&config.llm).await;
        }
        let snapshot = config.clone();
        drop(config);
        self.propagate_committed(&snapshot).await;
    }

    /// Push a committed config to everything that holds live settings — the
    /// LLM router's provider set, the scheduler loop, the embedding clients'
    /// Ollama token — and then tell every client.
    ///
    /// The one tail every mutation path (set, reset, reload, import, the file
    /// watcher) shares, so a live setting cannot reach one path and miss
    /// another. The Ollama token did exactly that: every path pushed it to the
    /// chat router, none to the embedder, so a server that wanted the token
    /// refused every memory embed until a restart.
    ///
    /// Called with the config lock released: provider resolution can block on
    /// the keyring or refresh an expired Claude CLI token over the network.
    async fn propagate_committed(&self, config: &Config) {
        self.rebuild_llm_providers(config).await;
        self.apply_scheduler_settings(config).await;
        if let Some(ref live) = self.live_embedding {
            live.apply(config);
        }
        self.notify_config_changed();
    }

    /// Tell every connected client the config changed.
    ///
    /// Carries no payload — see [`Event::ConfigChanged`]. Fired once per
    /// COMMITTED mutation (the in-memory config was replaced and the router
    /// re-derived); a rejected write emits nothing. Fire-and-forget: a send
    /// error only means nobody is subscribed.
    pub(super) fn notify_config_changed(&self) {
        if let Some(ref tx) = self.event_tx {
            let _ = tx.send(Event::ConfigChanged);
        }
    }

    // =========================================================================
    // Config Handlers
    // =========================================================================

    /// Rebuild the LLM router's provider set from a config snapshot.
    ///
    /// Provider registration used to happen only at boot; the GUI would save a
    /// credential and call config-reload "so the daemon rebuilds its LLM
    /// client", but nothing did — every call to the newly-authenticated
    /// provider failed with "No provider available" until a daemon restart.
    /// Runs after every config mutation (set/reset/reload/import), with the
    /// config write lock already released — credential resolution may hit the
    /// OS keyring and refresh an expired Claude CLI token over the network.
    async fn rebuild_llm_providers(&self, config: &Config) {
        let Some(router) = self.router.as_ref() else {
            return;
        };
        let llm = crate::server::LlmConfig::from_nanna(config);
        let creds = crate::llm_router::ProviderCredentials::resolve(&llm).await;
        router.rebuild(&creds);
    }

    /// The live config with every secret blanked, for anything that leaves
    /// the daemon. The live copy holds the keys filled in from the secure
    /// store, and `config.get` / `config.export` serialized it as it was: any
    /// client of the socket read the user's API keys and OAuth token in
    /// plaintext, though the export was documented to carry none. An import of
    /// such an export keeps the stored secrets (absent fields refill).
    async fn config_without_secrets(&self) -> Config {
        let mut config = self.config.read().await.clone();
        config.strip_secrets_for_disk();
        debug_assert!(config.llm.api_key.is_none() && config.llm.anthropic_oauth_token.is_none());
        config
    }

    pub(super) async fn handle_config(&self, _client_id: &str, action: ConfigAction) -> Value {
        match action {
            ConfigAction::Get { path } => {
                let serialized = serde_json::to_value(&self.config_without_secrets().await);
                let config_value = match serialized {
                    Ok(v) => v,
                    Err(e) => return json!({ "error": "serialize_failed", "message": e.to_string() }),
                };
                
                if let Some(path) = path {
                    // Get nested value by path (e.g., "llm.model")
                    let parts: Vec<&str> = path.split('.').collect();
                    let mut current = &config_value;
                    for part in parts {
                        match current.get(part) {
                            Some(v) => current = v,
                            None => return json!({ "error": "path_not_found", "path": path })
                        }
                    }
                    json!({ "value": current, "path": path })
                } else {
                    json!({ "config": config_value })
                }
            }
            ConfigAction::Set { path, value } => self.config_set(path, value).await,
            ConfigAction::Reset { path: Some(_) } => {
                // Reset specific path - would need more complex logic
                json!({ "error": "partial_reset_not_supported", "hint": "Use Reset without path to reset all" })
            }
            ConfigAction::Reset { path: None } => self.config_reset().await,
            ConfigAction::Reload => self.config_reload().await,
            ConfigAction::Export => {
                let config = self.config_without_secrets().await;
                // Export as JSON (TOML export would require additional dependencies)
                match serde_json::to_value(&config) {
                    Ok(v) => json!({ "config": v, "format": "json" }),
                    Err(e) => json!({ "error": "export_failed", "message": e.to_string() })
                }
            }
            ConfigAction::Import { config: config_value } => {
                // Parse as JSON object (TOML parsing removed for simplicity)
                match serde_json::from_value::<Config>(config_value) {
                    Ok(cfg) => self.config_import(cfg).await,
                    Err(e) => json!({ "error": "import_failed", "message": e.to_string() }),
                }
            }
        }
    }

    /// `ConfigAction::Reset` of the whole config: the defaults, with the
    /// environment's overrides and every stored credential, persisted and
    /// propagated live.
    async fn config_reset(&self) -> Value {
        let mut config = self.config.write().await;
        let previous_ollama_host = config.memory.ollama_host.clone();
        let mut reset = Config::default().with_env_overrides_from(&*self.environment);
        if let Err(message) = self.file_brought_in_secrets(&config, &reset).await {
            warn!("config.reset refused: {message}");
            return json!({ "error": "secret_store_failed", "message": message });
        }
        // A reset is of the settings, not the credentials: the store keeps
        // every secret, and the reset config holds them as a restart would
        // load them.
        if let Err(message) = self
            .settle_left_out_secrets(&config, &mut reset, None)
            .await
        {
            warn!("config.reset refused: {message}");
            return json!({ "error": "secret_store_failed", "message": message });
        }
        *config = reset;

        // Save to disk
        let unsaved = self.save_config(&config);

        // Propagate to agent service
        if let Some(ref agent) = self.agent {
            agent.apply_llm_config(&config.llm).await;
        }

        let snapshot = config.clone();
        drop(config);
        // The reset moves the Ollama address like a `config.set` of it: a
        // token saved with no server recorded was the old address's.
        let snapshot = self
            .adopt_moved_ollama_token(&previous_ollama_host, snapshot)
            .await;
        self.propagate_committed(&snapshot).await;

        if let Some(message) = unsaved {
            return not_persisted("reset", &message);
        }
        json!({ "status": "reset" })
    }

    /// `ConfigAction::Import`: replace the whole config with `cfg`, persist,
    /// and propagate it live.
    async fn config_import(&self, cfg: Config) -> Value {
        let mut config = self.config.write().await;
        let previous_ollama_host = config.memory.ollama_host.clone();
        let mut imported = match self.import_over(&config, cfg) {
            Ok(imported) => imported,
            Err(refusal) => return refusal,
        };
        if let Err(message) = self.file_brought_in_secrets(&config, &imported).await {
            warn!("config.import refused: {message}");
            return json!({ "error": "secret_store_failed", "message": message });
        }
        // An export may carry no secrets: one the import lacks is kept, not
        // cleared.
        if let Err(message) = self
            .settle_left_out_secrets(&config, &mut imported, None)
            .await
        {
            warn!("config.import refused: {message}");
            return json!({ "error": "secret_store_failed", "message": message });
        }
        *config = imported;

        // Save to disk
        let unsaved = self.save_config(&config);

        info!("Config imported");

        // Import replaces the whole config, `[llm]` included — it propagates
        // for the same reason set/reset/reload do.
        if let Some(ref agent) = self.agent {
            agent.apply_llm_config(&config.llm).await;
        }

        let snapshot = config.clone();
        drop(config);
        // As for a reset: the address may have moved.
        let snapshot = self
            .adopt_moved_ollama_token(&previous_ollama_host, snapshot)
            .await;
        self.propagate_committed(&snapshot).await;

        if let Some(message) = unsaved {
            return not_persisted("imported", &message);
        }
        json!({ "status": "imported" })
    }

    /// `ConfigAction::Reload`: load `config.toml` again and apply it live.
    ///
    /// Loaded as the replacement for what is running, so a token saved with
    /// no server recorded stays the running server's rather than following an
    /// address edited in the file. Off the async runtime: keyring reads can
    /// block on an unlock prompt.
    async fn config_reload(&self) -> Value {
        let running_host = self.config.read().await.memory.ollama_host.clone();
        let store = self.credential_store.clone();
        let env = Arc::clone(&self.environment);
        let loaded = tokio::task::spawn_blocking(move || {
            Config::load_replacing_with(&running_host, &store, &*env)
        })
        .await;
        match loaded {
            Ok(Ok(new_config)) => {
                self.apply_loaded_config(new_config.with_env_overrides_from(&*self.environment))
                    .await;
                json!({ "status": "reloaded" })
            }
            Ok(Err(e)) => json!({ "error": "reload_failed", "message": e.to_string() }),
            Err(e) => json!({ "error": "reload_failed", "message": e.to_string() }),
        }
    }

    /// `ConfigAction::Set`: write one dotted path, persist, and propagate the change live.
    async fn config_set(&self, path: String, value: Value) -> Value {
        if let Some(instead) = retired_key_set(&path, &value) {
            return json!({ "error": "retired_key", "message": instead, "path": path });
        }
        let mut config = self.config.write().await;
        let mut config_value = match serde_json::to_value(&*config) {
            Ok(v) => v,
            Err(e) => return json!({ "error": "serialize_failed", "message": e.to_string() }),
        };

        // Set nested value by path using a helper function
        let parts: Vec<&str> = path.split('.').collect();
        if parts.is_empty() {
            return json!({ "error": "invalid_path", "path": path });
        }

        if let Err(e) = set_nested(&mut config_value, &parts, value.clone()) {
            return json!({ "error": "set_failed", "message": e, "path": path });
        }
        // Null or blank at a secret's own path is someone clearing it. A
        // blank inside a larger value (a whole section, an import) is not:
        // that only lacks the secret.
        let clears = (is_blank(&value) && Config::names_a_secret(&path)).then_some(path.as_str());

        // Deserialize back to config
        match serde_json::from_value::<Config>(config_value) {
            Ok(mut new_config) => {
                if let Some(refusal) = unknown_path_refusal(&new_config, &path, &value) {
                    return refusal;
                }
                // The Ollama token is bound to the server it was saved for;
                // edited in place, the config would carry the old server's
                // token to a new `memory.ollama_host`. Dropped here, under the
                // lock, so nothing ever reads it paired with the new address;
                // which token the new address gets is read from the secure
                // store below.
                let previous_ollama_host = config.memory.ollama_host.clone();
                let ollama_moved = nanna_config::ollama_server_changed(
                    &previous_ollama_host,
                    &new_config.memory.ollama_host,
                );
                if ollama_moved {
                    new_config.llm.ollama_api_key = None;
                }
                if let Err(mut refusal) = self.refuse_env_supplied_secrets(&config, &new_config) {
                    warn!("config.set {path} refused: {}", refusal["message"]);
                    refusal["path"] = json!(path);
                    return refusal;
                }
                if let Err(message) = self.file_brought_in_secrets(&config, &new_config).await {
                    warn!("config.set {path} refused: {message}");
                    return json!({ "error": "secret_store_failed", "message": message, "path": path });
                }
                if let Err(message) = self
                    .settle_left_out_secrets(&config, &mut new_config, clears)
                    .await
                {
                    warn!("config.set {path} refused: {message}");
                    return json!({ "error": "secret_store_failed", "message": message, "path": path });
                }
                // Cleared from the store, and still held: the environment
                // supplies it, and every load takes it from there.
                let still_supplied = clears.is_some() && new_config.holds_secret(&path);
                *config = new_config;

                // Save to disk if we have a path
                let unsaved = self.save_config(&config);

                // Propagate LLM config changes to agent service.
                // Whole-`[llm]` push, not just the model fields: a
                // `set` of e.g. `llm.summarization_priority` used to
                // land on disk and in `self.config` while the running
                // agent kept summarizing on the boot-time model.
                if path.starts_with("llm.")
                    && let Some(ref agent) = self.agent
                {
                    agent.apply_llm_config(&config.llm).await;
                }

                // Re-derive the router's provider set (registration is
                // not boot-only) and the rest of the live settings. Lock
                // released first: resolution can block on keyring/network.
                let snapshot = config.clone();
                drop(config);
                let snapshot = self
                    .adopt_moved_ollama_token(&previous_ollama_host, snapshot)
                    .await;
                self.propagate_committed(&snapshot).await;

                if let Some(message) = unsaved {
                    let mut reply = not_persisted("updated", &message);
                    reply["path"] = json!(path);
                    return reply;
                }
                if still_supplied {
                    let note = format!(
                        "{path} is deleted from the secure store, but the environment still \
                         supplies it; unset it there to clear it"
                    );
                    warn!("config.set {path}: {note}");
                    return json!({ "status": "updated", "path": path, "note": note });
                }
                json!({ "status": "updated", "path": path })
            }
            Err(e) => json!({ "error": "invalid_config", "message": e.to_string() })
        }
    }

    /// Save `config` to the daemon's config file, if it has one. `Some` is
    /// the reason a save failed: the change is live but will not survive a
    /// restart, which the caller must say rather than reply a clean success.
    pub(super) fn save_config(&self, config: &Config) -> Option<String> {
        let config_path = self.config_path.as_ref()?;
        match config.save_to(config_path) {
            Ok(()) => {
                info!("Config saved to {}", config_path.display());
                None
            }
            Err(e) => {
                warn!("Failed to save config to {}: {e}", config_path.display());
                Some(e.to_string())
            }
        }
    }

    /// After a `config.set`, reset or import that may have moved
    /// `[memory].ollama_host` away from `previous_host`, give the running
    /// config the token the new address is owed, if any, and return
    /// `changed` as the rest of the change should propagate it.
    ///
    /// Nothing to do when the address did not move, or when the new config
    /// names a token of its own (an imported one — the operator's say-so, as
    /// a token written into `config.toml` is). Read with no config lock held,
    /// because keyring reads can block on an unlock prompt. A stored token
    /// with no recorded server was `previous_host`'s, and is recorded as such
    /// (see `Config::rebind_ollama_token_if_moved`). The token only lands if
    /// the address is still the one it was resolved for: a change that moved
    /// it again meanwhile resolves its own.
    async fn adopt_moved_ollama_token(&self, previous_host: &str, changed: Config) -> Config {
        let names_its_own = changed
            .llm
            .ollama_api_key
            .as_deref()
            .is_some_and(|token| !token.trim().is_empty());
        if names_its_own
            || !nanna_config::ollama_server_changed(previous_host, &changed.memory.ollama_host)
        {
            return changed;
        }
        let store = self.credential_store.clone();
        let env = Arc::clone(&self.environment);
        let previous_host = previous_host.to_string();
        let mut resolved = changed.clone();
        let resolved = tokio::task::spawn_blocking(move || {
            resolved.rebind_ollama_token_if_moved_with(&previous_host, &store, &*env);
            resolved
        })
        .await;
        let token = match resolved {
            Ok(resolved) => resolved.llm.ollama_api_key,
            Err(e) => {
                warn!("Resolving the Ollama token for the new address failed: {e}");
                None
            }
        };
        let Some(token) = token else {
            return changed;
        };
        let mut config = self.config.write().await;
        if nanna_config::same_ollama_server(&config.memory.ollama_host, &changed.memory.ollama_host)
            && config.llm.ollama_api_key.is_none()
        {
            config.llm.ollama_api_key = Some(token);
        }
        config.clone()
    }

    /// Refuse a change from `previous` to `changed` that brings in a secret
    /// the daemon's environment supplies (`Config::secrets_the_environment_replaces`):
    /// `Err` is the response naming each one and its variable.
    ///
    /// Every load fills a secret from its variable before the store, and
    /// `with_env_overrides` replaces some outright. Made, such a change
    /// answered `updated`, ran the value set and filed it — and the next load
    /// (the config watcher reading the daemon's own save back, seconds later,
    /// or a restart) put the environment's back. Refused, nothing is applied,
    /// filed or saved, and whoever made it learns which variable to unset.
    ///
    /// As [`Self::file_brought_in_secrets`], only for a control plane that
    /// saves: with no config path nothing it changes outlives it. It reads
    /// the environment alone, never the store, so it runs under the lock. A
    /// reset is not checked: it brings in no secret but the environment's own.
    fn refuse_env_supplied_secrets(
        &self,
        previous: &Config,
        changed: &Config,
    ) -> Result<(), Value> {
        if self.config_path.is_none() {
            return Ok(());
        }
        let supplied = changed.secrets_the_environment_replaces(previous, &*self.environment);
        let Some(first) = supplied.first() else {
            return Ok(());
        };
        let taken = supplied
            .iter()
            .map(|secret| format!("{} from {}", secret.path, secret.variable))
            .collect::<Vec<_>>()
            .join(", ");
        let variables = supplied
            .iter()
            .map(|secret| secret.variable.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        Err(json!({
            "error": "env_overrides_secret",
            "secret": first.path,
            "variable": first.variable,
            "message": format!(
                "every load takes {taken} in the daemon's environment, so the value given here \
                 would be replaced by the environment's within seconds. To change it here, \
                 unset {variables} in the daemon's environment and restart the daemon."
            ),
        }))
    }

    /// `imported` as it would run in place of `running`: with the
    /// environment's overrides, unless it brings in a secret the environment
    /// supplies ([`Self::refuse_env_supplied_secrets`]; `Err` is the
    /// response). Checked as given, since the overrides would swap such a
    /// key for the environment's own unseen.
    ///
    /// Takes the import by value, so the async handler never borrows it: a
    /// borrowed local is kept in the handler's future across every later
    /// await, and the import is a whole `Config`.
    fn import_over(&self, running: &Config, imported: Config) -> Result<Config, Value> {
        if let Err(refusal) = self.refuse_env_supplied_secrets(running, &imported) {
            warn!("config.import refused: {}", refusal["message"]);
            return Err(refusal);
        }
        Ok(imported.with_env_overrides_from(&*self.environment))
    }

    /// File in the secure store the secrets `changed` brings in over
    /// `previous` (`Config::brings_in_secrets`), before the save that strips
    /// them from `config.toml`. Filed nowhere, a secret set, reset or imported
    /// here worked until the next load — a restart, or the config watcher
    /// reading the save back seconds later — and was gone.
    ///
    /// Only those, and none the daemon's environment supplies
    /// (`Config::file_secrets_brought_in`): the running config also holds the
    /// secrets the environment supplied at load, and a change can bring one
    /// in over another value the running config held, or none — a reset or an
    /// import rebuilt with the environment's overrides, or a set of the
    /// environment's own value. One of those filed with a change outlived its
    /// variable — unset or rotated, and the next start ran on the stale copy.
    ///
    /// Only for a control plane that saves: with no config path nothing it
    /// changes outlives it, secrets included. Called with the config write
    /// lock held, so filing and commit are one critical section and two
    /// changes of one secret cannot leave the store holding the one that
    /// lost. The store is written off the async runtime (a keyring write can
    /// wait on an unlock prompt), and only for a change that brings a secret
    /// in — one someone is entering as it happens. A change of anything else
    /// never touches the store.
    ///
    /// An `Err` says the store refused a secret, and the change must be
    /// refused with it: applied, it would work until the next load and then
    /// be gone. Secrets filed before the one refused stay filed.
    async fn file_brought_in_secrets(
        &self,
        previous: &Config,
        changed: &Config,
    ) -> Result<(), String> {
        if self.config_path.is_none() || !changed.brings_in_secrets(previous) {
            return Ok(());
        }
        let (filing, previous) = (changed.clone(), previous.clone());
        let store = self.credential_store.clone();
        let env = Arc::clone(&self.environment);
        let filed = tokio::task::spawn_blocking(move || {
            filing.file_secrets_brought_in(&previous, &store, &*env)
        });
        match filed.await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(format!("the secure store refused a secret: {e}")),
            Err(e) => Err(format!("filing secrets in the secure store failed: {e}")),
        }
    }

    /// Settle the secrets `changed` leaves out, before it is committed and
    /// saved: forget the one a `config.set` clears (`cleared`, a path
    /// `Config::names_a_secret`), and give `changed` back every one it
    /// merely drops (`Config::drops_secrets`), from where the next load
    /// fills them. Settled, `changed` holds what that load gives it.
    ///
    /// The save strips every secret, and the next load — a restart, or the
    /// config watcher reading the save back seconds later — fills each unset
    /// one from the environment, else the store. A clear made in the running
    /// config alone was back with that load. A change that merely lacked a
    /// secret — a reset, an import of an export carrying none, a set of a
    /// whole section — ran without it until then, chat included, while the
    /// store still held it.
    ///
    /// As [`Self::file_brought_in_secrets`]: only for a control plane that
    /// saves, called with the config write lock held so that the store and
    /// the commit agree, off the async runtime, and only for a change that
    /// clears or drops a secret. A change of anything else never touches the
    /// store.
    ///
    /// An `Err` says the store could not forget the cleared secret, and the
    /// change must be refused with it: applied, the secret would be back at
    /// the next load. `changed` is then left as it was.
    async fn settle_left_out_secrets(
        &self,
        previous: &Config,
        changed: &mut Config,
        cleared: Option<&str>,
    ) -> Result<(), String> {
        if self.config_path.is_none() || (cleared.is_none() && !changed.drops_secrets(previous)) {
            return Ok(());
        }
        let cleared = cleared.map(str::to_owned);
        let running_ollama_host = previous.memory.ollama_host.clone();
        let store = self.credential_store.clone();
        let env = Arc::clone(&self.environment);
        let mut settling = changed.clone();
        let settled = tokio::task::spawn_blocking(move || -> Result<Config, String> {
            if let Some(path) = cleared {
                settling
                    .forget_secret(&path, &store)
                    .map_err(|e| format!("the secure store could not delete {path}: {e}"))?;
            }
            settling.refill_secrets_replacing_with(&running_ollama_host, &store, &*env);
            Ok(settling)
        })
        .await;
        *changed = settled
            .map_err(|e| format!("settling secrets with the secure store failed: {e}"))??;
        Ok(())
    }
}

/// Whether a `config.set` value is null or a blank string.
fn is_blank(value: &Value) -> bool {
    value.is_null() || value.as_str().is_some_and(|text| text.trim().is_empty())
}

/// Keys the daemon once read and no longer does, each with the sentence
/// saying what replaced it.
///
/// A `config.set` of one would answer `updated` and change nothing: serde
/// drops an unknown key on the round trip, and the next save removes it from
/// disk. Refusing it by name is how a script that used it learns it stopped
/// working.
const RETIRED_KEYS: [(&str, &str); 1] = [(
    "llm.ollama_url",
    "[llm].ollama_url is no longer read: summaries go through chat's router, so an \
     `ollama/` summarization model uses chat's one Ollama server and its token. Set \
     memory.ollama_host to move it.",
)];

/// The refusal for a `set` that wrote a key `Config` does not have, or `None`.
///
/// `set_nested` inserts any key and serde drops the unknown ones on the way
/// back into `Config`, so `config.set {path: "llm.modle"}` answered `updated`,
/// announced `ConfigChanged` and changed nothing (`config.get` of the same
/// path then said `path_not_found`). Judged on the round trip itself: a value
/// that survives it is somewhere in `Config`. An empty value (`null`, `""`,
/// `[]`, `{}`) may legitimately leave nothing behind — optional and
/// `skip_serializing_if` empty fields are not written — and secrets are not
/// serialized at all, so none of those is judged here.
fn unknown_path_refusal(config: &Config, path: &str, value: &Value) -> Option<Value> {
    let empty = is_blank(value)
        || value.as_array().is_some_and(Vec::is_empty)
        || value.as_object().is_some_and(serde_json::Map::is_empty);
    if empty || Config::names_a_secret(path) {
        return None;
    }
    let written = serde_json::to_value(config).ok()?;
    let landed = path
        .split('.')
        .try_fold(&written, |node, part| node.get(part))
        .is_some();
    (!landed).then(|| {
        json!({
            "error": "unknown_path",
            "message": format!("'{path}' is not a configuration key; nothing was changed"),
            "path": path,
        })
    })
}

/// What replaced the retired key a `set` of `value` at `path` would write, or
/// `None` when it writes none.
///
/// Either the key's own path, or a parent of it given an object that carries
/// the key (`llm` with `{"ollama_url": …}`): the round trip drops the key the
/// same way in both, so both are refused the same way.
fn retired_key_set(path: &str, value: &Value) -> Option<&'static str> {
    RETIRED_KEYS.iter().find_map(|(key, instead)| {
        let written = *key == path
            || key
                .strip_prefix(path)
                .and_then(|rest| rest.strip_prefix('.'))
                .is_some_and(|rest| {
                    rest.split('.')
                        .try_fold(value, |node, part| node.get(part))
                        .is_some()
                });
        written.then_some(*instead)
    })
}

/// Set `value` at the dotted `parts` path inside `obj`, creating intermediate
/// objects as needed (pointer-based access for nested updates).
fn set_nested(obj: &mut Value, parts: &[&str], value: Value) -> Result<(), String> {
    if parts.is_empty() {
        return Err("Empty path".to_string());
    }
    
    if parts.len() == 1 {
        // Final part - set the value
        obj.as_object_mut().map_or_else(
            || Err("Parent is not an object".to_string()),
            |map| {
                map.insert(parts[0].to_string(), value);
                Ok(())
            },
        )
    } else {
        // Navigate deeper
        obj.as_object_mut().map_or_else(
            || Err("Parent is not an object".to_string()),
            |map| {
                let next = map.entry(parts[0]).or_insert(json!({}));
                set_nested(next, &parts[1..], value)
            },
        )
    }
}

/// The reply for a change that is live but was not saved: an `error` (so a
/// client that checks only for one cannot show it as saved), the `status` the
/// change reached, and why the save failed. Every config write used to log
/// the failure and reply success — a setting that silently reverted at the
/// next restart.
pub(super) fn not_persisted(status: &str, reason: &str) -> Value {
    debug_assert!(!reason.is_empty(), "a failed save has a reason");
    json!({
        "error": "not_persisted",
        "status": status,
        "message": format!(
            "applied for this run, but not saved to the config file ({reason}); \
             it will be lost on restart"
        ),
    })
}
