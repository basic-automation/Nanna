//! The non-secret settings the environment replaces at every load.
//!
//! [`Config::with_env_overrides`] puts `PORT` into `server.port` and a
//! channel's webhook URL or Discord IDs into its section. Every writer then
//! saved the running config as it was — so those values landed in
//! `config.toml` and outlived the variable (unset `PORT`, and the daemon still
//! moved to the port it once named), and a channel section the environment
//! built from a token alone was written as if the user had configured it.
//! A `config.set` of one answered `updated` and was put back by the next load
//! seconds later. Secrets already had both answers (stripped on save, refused
//! on set); this is the same pair for the rest.

use crate::Config;

/// A non-secret setting an environment variable replaces at every load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnvOverriddenSetting {
    /// The setting's dotted path, as `config.set` names it.
    pub path: &'static str,
    /// The variable it is taken from.
    pub variable: &'static str,
}

const PORT: EnvOverriddenSetting = EnvOverriddenSetting {
    path: "server.port",
    variable: "PORT",
};
const TELEGRAM_WEBHOOK_URL: EnvOverriddenSetting = EnvOverriddenSetting {
    path: "channels.telegram.webhook_url",
    variable: "TELEGRAM_WEBHOOK_URL",
};
const DISCORD_APPLICATION_ID: EnvOverriddenSetting = EnvOverriddenSetting {
    path: "channels.discord.application_id",
    variable: "DISCORD_APPLICATION_ID",
};
const DISCORD_PUBLIC_KEY: EnvOverriddenSetting = EnvOverriddenSetting {
    path: "channels.discord.public_key",
    variable: "DISCORD_PUBLIC_KEY",
};

impl Config {
    /// The non-secret settings `env` replaces on the next load of this
    /// config, by the same rules [`Config::with_env_overrides_from`] applies:
    /// a channel's fields only under its bot token, Discord's only into an
    /// existing section or a whole one. Pure over `env`.
    #[must_use]
    pub fn settings_the_environment_overrides(
        &self,
        env: impl Fn(&str) -> Option<String>,
    ) -> Vec<EnvOverriddenSetting> {
        let set = |name: &str| env(name).is_some_and(|value| !value.trim().is_empty());
        let mut overridden = Vec::with_capacity(4);
        if env(PORT.variable).is_some_and(|port| port.parse::<u16>().is_ok()) {
            overridden.push(PORT);
        }
        if set("TELEGRAM_BOT_TOKEN") && set(TELEGRAM_WEBHOOK_URL.variable) {
            overridden.push(TELEGRAM_WEBHOOK_URL);
        }
        let discord_applies = set("DISCORD_BOT_TOKEN")
            && (self.channels.discord.is_some()
                || (set(DISCORD_APPLICATION_ID.variable) && set(DISCORD_PUBLIC_KEY.variable)));
        if discord_applies {
            overridden.extend(
                [DISCORD_APPLICATION_ID, DISCORD_PUBLIC_KEY]
                    .into_iter()
                    .filter(|setting| set(setting.variable)),
            );
        }
        debug_assert!(overridden.len() <= 4);
        overridden
    }

    /// Give each setting `env` overrides the value `on_disk` holds, so a save
    /// writes what the file said rather than what the environment says; and
    /// drop a channel section the file never had that the environment built
    /// from a token alone (nothing in it is the user's).
    pub(crate) fn keep_file_values_of_env_settings(
        &mut self,
        on_disk: &Self,
        env: impl Fn(&str) -> Option<String>,
    ) {
        let set = |name: &str| env(name).is_some_and(|value| !value.trim().is_empty());
        for setting in self.settings_the_environment_overrides(&env) {
            match setting.path {
                "server.port" => self.server.port = on_disk.server.port,
                "channels.telegram.webhook_url" => {
                    if let Some(telegram) = &mut self.channels.telegram {
                        telegram.webhook_url = on_disk
                            .channels
                            .telegram
                            .as_ref()
                            .and_then(|file| file.webhook_url.clone());
                    }
                }
                "channels.discord.application_id" | "channels.discord.public_key" => {
                    if let (Some(discord), Some(file)) =
                        (&mut self.channels.discord, &on_disk.channels.discord)
                    {
                        discord.application_id.clone_from(&file.application_id);
                        discord.public_key.clone_from(&file.public_key);
                    }
                }
                other => debug_assert!(false, "an override with no file value: {other}"),
            }
        }
        if set("TELEGRAM_BOT_TOKEN")
            && on_disk.channels.telegram.is_none()
            && self.channels.telegram.as_ref().is_some_and(|telegram| {
                telegram.allowed_users.is_none() && telegram.webhook_url.is_none()
            })
        {
            self.channels.telegram = None;
        }
        if set("DISCORD_BOT_TOKEN") && on_disk.channels.discord.is_none() {
            // Built only whole from the three variables: nothing is the file's.
            self.channels.discord = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> + use<> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        move |name| map.get(name).cloned()
    }

    #[test]
    fn the_overridden_settings_follow_the_load_rules() {
        let config = Config::default();
        assert_eq!(config.settings_the_environment_overrides(env(&[])), []);
        assert_eq!(
            config.settings_the_environment_overrides(env(&[("PORT", "8123")])),
            [PORT]
        );
        assert!(
            config
                .settings_the_environment_overrides(env(&[("PORT", "not a port")]))
                .is_empty(),
            "an unparsable PORT overrides nothing"
        );
        assert!(
            config
                .settings_the_environment_overrides(env(&[("TELEGRAM_WEBHOOK_URL", "https://h")]))
                .is_empty(),
            "a channel's fields only under its token"
        );
        assert_eq!(
            config.settings_the_environment_overrides(env(&[
                ("TELEGRAM_BOT_TOKEN", "t"),
                ("TELEGRAM_WEBHOOK_URL", "https://h"),
            ])),
            [TELEGRAM_WEBHOOK_URL]
        );
        assert!(
            config
                .settings_the_environment_overrides(env(&[
                    ("DISCORD_BOT_TOKEN", "t"),
                    ("DISCORD_APPLICATION_ID", "1"),
                ]))
                .is_empty(),
            "no Discord section, and not all three: nothing is built"
        );
    }

    #[test]
    fn a_save_writes_the_files_values_not_the_environments() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("config.toml");
        let mut on_disk = Config::default();
        on_disk.server.port = 4000;
        on_disk.save_to(&path).expect("save");

        let environment = env(&[
            ("PORT", "8123"),
            ("TELEGRAM_BOT_TOKEN", "123:abc"),
            ("TELEGRAM_WEBHOOK_URL", "https://example.test/hook"),
        ]);
        let running = Config::load_from(&path)
            .expect("load")
            .with_env_overrides_from(&environment);
        assert_eq!(running.server.port, 8123, "the environment wins at run time");
        assert!(running.channels.telegram.is_some(), "built from the token");

        let mut changed = running;
        changed.agent.name = "Edited".to_string();
        changed.save_to_with(&path, &environment).expect("save");

        let saved = Config::load_from(&path).expect("load");
        assert_eq!(saved.server.port, 4000, "the file's port, not PORT");
        assert!(saved.channels.telegram.is_none(), "no section the user never made");
        assert_eq!(saved.agent.name, "Edited", "the real change is saved");
    }
}
