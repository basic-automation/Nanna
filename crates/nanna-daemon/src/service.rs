//! Platform service integration
//!
//! Handles running the daemon as a system service on different platforms:
//! - Windows: Windows Service
//! - macOS: launchd
//! - Linux: systemd

use std::path::PathBuf;

/// Service status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServiceStatus {
    Running,
    Stopped,
    Starting,
    Stopping,
    Unknown,
}

/// Service configuration
#[derive(Debug, Clone)]
pub struct ServiceConfig {
    pub name: String,
    pub display_name: String,
    pub description: String,
    pub executable: PathBuf,
    /// Arguments the platform's service supervisor passes when it launches
    /// `executable`. **Platform-dependent** — see `Default`.
    pub arguments: Vec<String>,
    pub working_directory: Option<PathBuf>,
}

/// The subcommand a service supervisor must invoke to start the daemon.
///
/// Windows is the odd one out and it is not cosmetic: the SCM requires the
/// process it launches to call `StartServiceCtrlDispatcher` and report status,
/// which is what the `service` subcommand does (`windows_service::run_as_service`).
/// Installing with `run` yields a console daemon that never reports to the SCM,
/// so Windows kills it as failed-to-start. launchd and systemd have no such
/// handshake — they supervise the foreground process `run` gives them.
const DEFAULT_SERVICE_ARGUMENT: &str = if cfg!(windows) { "service" } else { "run" };

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            name: "nanna-daemon".to_string(),
            display_name: "Nanna Daemon".to_string(),
            description: "Nanna AI assistant background service".to_string(),
            executable: std::env::current_exe().unwrap_or_else(|_| PathBuf::from("nanna-daemon")),
            arguments: vec![DEFAULT_SERVICE_ARGUMENT.to_string()],
            working_directory: None,
        }
    }
}

/// Platform-specific service operations
pub struct ServiceManager {
    config: ServiceConfig,
}

impl ServiceManager {
    #[must_use]
    pub const fn new(config: ServiceConfig) -> Self {
        Self { config }
    }
    
    /// Install the service
    ///
    /// # Errors
    ///
    /// Returns an error when the platform has no service backend, or the
    /// backend step fails: on Linux, creating or writing the systemd user unit
    /// or spawning `systemctl`; on macOS, writing the launchd plist or spawning
    /// `launchctl`; on Windows, connecting to the service manager or creating
    /// the service.
    pub fn install(&self) -> Result<(), String> {
        #[cfg(windows)]
        return self.install_windows();
        
        #[cfg(target_os = "macos")]
        return self.install_macos();
        
        #[cfg(target_os = "linux")]
        return self.install_linux();
        
        #[cfg(not(any(windows, target_os = "macos", target_os = "linux")))]
        Err("Service installation not supported on this platform".to_string())
    }
    
    /// Uninstall the service
    ///
    /// # Errors
    ///
    /// Returns an error when the platform has no service backend, or the
    /// backend step fails: on Linux, removing the systemd unit file or spawning
    /// `systemctl daemon-reload`; on macOS, removing the launchd plist; on
    /// Windows, connecting to the service manager or opening/deleting the
    /// service.
    pub fn uninstall(&self) -> Result<(), String> {
        #[cfg(windows)]
        return self.uninstall_windows();
        
        #[cfg(target_os = "macos")]
        return self.uninstall_macos();
        
        #[cfg(target_os = "linux")]
        return self.uninstall_linux();
        
        #[cfg(not(any(windows, target_os = "macos", target_os = "linux")))]
        Err("Service uninstallation not supported on this platform".to_string())
    }
    
    /// Start the service
    ///
    /// # Errors
    ///
    /// Returns an error when the platform has no service backend, or the
    /// service command could not be issued: spawning `systemctl` (Linux) or
    /// `launchctl` (macOS) failed, or the Windows service manager could not be
    /// reached or refused the start. A spawned command's non-zero exit status is
    /// not checked.
    pub fn start(&self) -> Result<(), String> {
        #[cfg(windows)]
        return self.start_windows();
        
        #[cfg(target_os = "macos")]
        return self.start_macos();
        
        #[cfg(target_os = "linux")]
        return self.start_linux();
        
        #[cfg(not(any(windows, target_os = "macos", target_os = "linux")))]
        Err("Service start not supported on this platform".to_string())
    }
    
    /// Stop the service
    ///
    /// # Errors
    ///
    /// Returns an error when the platform has no service backend, or the
    /// service command could not be issued: spawning `systemctl` (Linux) or
    /// `launchctl` (macOS) failed, or the Windows service manager could not be
    /// reached or refused the stop. A spawned command's non-zero exit status is
    /// not checked.
    pub fn stop(&self) -> Result<(), String> {
        #[cfg(windows)]
        return self.stop_windows();
        
        #[cfg(target_os = "macos")]
        return self.stop_macos();
        
        #[cfg(target_os = "linux")]
        return self.stop_linux();
        
        #[cfg(not(any(windows, target_os = "macos", target_os = "linux")))]
        Err("Service stop not supported on this platform".to_string())
    }
    
    /// Get service status
    #[must_use]
    pub fn status(&self) -> ServiceStatus {
        #[cfg(windows)]
        return self.status_windows();
        
        #[cfg(target_os = "macos")]
        return self.status_macos();
        
        #[cfg(target_os = "linux")]
        return self.status_linux();
        
        #[cfg(not(any(windows, target_os = "macos", target_os = "linux")))]
        ServiceStatus::Unknown
    }
    
    // =========================================================================
    // Windows implementation
    // =========================================================================
    
    // These delegate to `crate::windows_service`, which is where the SCM work
    // actually lives. They used to be `not yet implemented` stubs while
    // `windows_service` had a full implementation the CLI called directly — so
    // the platform abstraction reported a capability as missing that the binary
    // had been shipping all along. There is now one implementation per platform,
    // reached the same way.

    #[cfg(windows)]
    fn install_windows(&self) -> Result<(), String> {
        crate::windows_service::install_service(&self.config)
    }

    #[cfg(windows)]
    fn uninstall_windows(&self) -> Result<(), String> {
        crate::windows_service::uninstall_service(&self.config)
    }

    #[cfg(windows)]
    fn start_windows(&self) -> Result<(), String> {
        crate::windows_service::start_service(&self.config)
    }

    #[cfg(windows)]
    fn stop_windows(&self) -> Result<(), String> {
        crate::windows_service::stop_service(&self.config)
    }

    #[cfg(windows)]
    fn status_windows(&self) -> ServiceStatus {
        crate::windows_service::query_service_status(&self.config)
    }
    
    // =========================================================================
    // macOS implementation (launchd)
    // =========================================================================
    
    #[cfg(target_os = "macos")]
    fn install_macos(&self) -> Result<(), String> {
        let plist_path = self.launchd_plist_path();
        let plist_content = self.generate_launchd_plist();
        
        std::fs::create_dir_all(plist_path.parent().unwrap())
            .map_err(|e| e.to_string())?;
        std::fs::write(&plist_path, plist_content)
            .map_err(|e| e.to_string())?;
        
        // Load the service
        std::process::Command::new("launchctl")
            .args(["load", plist_path.to_str().unwrap()])
            .status()
            .map_err(|e| e.to_string())?;
        
        Ok(())
    }
    
    #[cfg(target_os = "macos")]
    fn uninstall_macos(&self) -> Result<(), String> {
        let plist_path = self.launchd_plist_path();
        
        // Unload first
        let _ = std::process::Command::new("launchctl")
            .args(["unload", plist_path.to_str().unwrap()])
            .status();
        
        // Remove plist
        if plist_path.exists() {
            std::fs::remove_file(&plist_path).map_err(|e| e.to_string())?;
        }
        
        Ok(())
    }
    
    #[cfg(target_os = "macos")]
    fn start_macos(&self) -> Result<(), String> {
        std::process::Command::new("launchctl")
            .args(["start", &format!("com.nanna.{}", self.config.name)])
            .status()
            .map_err(|e| e.to_string())?;
        Ok(())
    }
    
    #[cfg(target_os = "macos")]
    fn stop_macos(&self) -> Result<(), String> {
        std::process::Command::new("launchctl")
            .args(["stop", &format!("com.nanna.{}", self.config.name)])
            .status()
            .map_err(|e| e.to_string())?;
        Ok(())
    }
    
    #[cfg(target_os = "macos")]
    fn status_macos(&self) -> ServiceStatus {
        let output = std::process::Command::new("launchctl")
            .args(["list", &format!("com.nanna.{}", self.config.name)])
            .output();
        
        match output {
            Ok(o) if o.status.success() => ServiceStatus::Running,
            _ => ServiceStatus::Stopped,
        }
    }
    
    #[cfg(target_os = "macos")]
    fn launchd_plist_path(&self) -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(home)
            .join("Library/LaunchAgents")
            .join(format!("com.nanna.{}.plist", self.config.name))
    }
    
    #[cfg(target_os = "macos")]
    fn generate_launchd_plist(&self) -> String {
        let exe = self.config.executable.display();
        let args: String = self.config.arguments.iter()
            .map(|a| format!("        <string>{}</string>", a))
            .collect::<Vec<_>>()
            .join("\n");
        
        format!(r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.nanna.{}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{}</string>
{}
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/nanna-daemon.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/nanna-daemon.err</string>
</dict>
</plist>"#, self.config.name, exe, args)
    }
    
    // =========================================================================
    // Linux implementation (systemd)
    // =========================================================================
    
    #[cfg(target_os = "linux")]
    fn install_linux(&self) -> Result<(), String> {
        let unit_path = self.systemd_unit_path();
        let unit_content = self.generate_systemd_unit();
        
        std::fs::create_dir_all(unit_path.parent().unwrap())
            .map_err(|e| e.to_string())?;
        std::fs::write(&unit_path, unit_content)
            .map_err(|e| e.to_string())?;
        
        // Reload systemd
        std::process::Command::new("systemctl")
            .args(["--user", "daemon-reload"])
            .status()
            .map_err(|e| e.to_string())?;
        
        // Enable the service
        std::process::Command::new("systemctl")
            .args(["--user", "enable", &self.config.name])
            .status()
            .map_err(|e| e.to_string())?;
        
        Ok(())
    }
    
    #[cfg(target_os = "linux")]
    fn uninstall_linux(&self) -> Result<(), String> {
        // Disable and stop
        let _ = std::process::Command::new("systemctl")
            .args(["--user", "disable", "--now", &self.config.name])
            .status();
        
        // Remove unit file
        let unit_path = self.systemd_unit_path();
        if unit_path.exists() {
            std::fs::remove_file(&unit_path).map_err(|e| e.to_string())?;
        }
        
        // Reload systemd
        std::process::Command::new("systemctl")
            .args(["--user", "daemon-reload"])
            .status()
            .map_err(|e| e.to_string())?;
        
        Ok(())
    }
    
    #[cfg(target_os = "linux")]
    fn start_linux(&self) -> Result<(), String> {
        std::process::Command::new("systemctl")
            .args(["--user", "start", &self.config.name])
            .status()
            .map_err(|e| e.to_string())?;
        Ok(())
    }
    
    #[cfg(target_os = "linux")]
    fn stop_linux(&self) -> Result<(), String> {
        std::process::Command::new("systemctl")
            .args(["--user", "stop", &self.config.name])
            .status()
            .map_err(|e| e.to_string())?;
        Ok(())
    }
    
    #[cfg(target_os = "linux")]
    fn status_linux(&self) -> ServiceStatus {
        let output = std::process::Command::new("systemctl")
            .args(["--user", "is-active", &self.config.name])
            .output();
        
        match output {
            Ok(o) => {
                let status = String::from_utf8_lossy(&o.stdout).trim().to_string();
                match status.as_str() {
                    "active" => ServiceStatus::Running,
                    "inactive" => ServiceStatus::Stopped,
                    "activating" => ServiceStatus::Starting,
                    "deactivating" => ServiceStatus::Stopping,
                    _ => ServiceStatus::Unknown,
                }
            }
            _ => ServiceStatus::Unknown,
        }
    }
    
    #[cfg(target_os = "linux")]
    fn systemd_unit_path(&self) -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(home)
            .join(".config/systemd/user")
            .join(format!("{}.service", self.config.name))
    }
    
    #[cfg(target_os = "linux")]
    fn generate_systemd_unit(&self) -> String {
        // Each word quoted: systemd splits `ExecStart=` on whitespace and
        // expands `%` specifiers, so an install under a path with a space (or
        // a `%`) started the wrong program, or none.
        let exe = systemd_quote(&self.config.executable.to_string_lossy());
        let args = self
            .config
            .arguments
            .iter()
            .map(|arg| systemd_quote(arg))
            .collect::<Vec<_>>()
            .join(" ");
        
        format!(r"[Unit]
Description={}
After=network.target

[Service]
Type=simple
ExecStart={} {}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
", self.config.description, exe, args)
    }
}

/// One `ExecStart=` word, quoted the way systemd reads it: double quotes
/// around the whole word, `\` and `"` backslash-escaped inside them, and `%`
/// doubled so no specifier is expanded (systemd.service(5), "Command lines").
#[cfg(any(target_os = "linux", test))]
fn systemd_quote(word: &str) -> String {
    let mut quoted = String::with_capacity(word.len() + 2);
    quoted.push('"');
    for c in word.chars() {
        match c {
            '\\' | '"' => {
                quoted.push('\\');
                quoted.push(c);
            }
            '%' => quoted.push_str("%%"),
            _ => quoted.push(c),
        }
    }
    quoted.push('"');
    debug_assert!(quoted.len() >= word.len() + 2, "quoting only ever adds");
    quoted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exec_start_words_survive_spaces_quotes_and_specifiers() {
        assert_eq!(
            systemd_quote("/usr/bin/nanna-daemon"),
            "\"/usr/bin/nanna-daemon\""
        );
        assert_eq!(
            systemd_quote("/home/u/My Apps/nanna-daemon"),
            "\"/home/u/My Apps/nanna-daemon\"",
            "a space stays inside one word"
        );
        assert_eq!(systemd_quote(r#"a"b\c"#), r#""a\"b\\c""#);
        assert_eq!(systemd_quote("100%"), "\"100%%\"", "no specifier expansion");
    }

    /// The bug this guards: Windows needs the `service` subcommand (the SCM
    /// dispatcher), every other platform needs `run` (a supervised foreground
    /// process). Installing Windows with `run` yields a service the SCM kills for
    /// never reporting status — which is exactly why the Windows path used to
    /// bypass `ServiceConfig` and hardcode its own argument.
    #[test]
    fn default_service_argument_matches_the_platform_contract() {
        let config = ServiceConfig::default();
        if cfg!(windows) {
            assert_eq!(
                config.arguments,
                vec!["service".to_string()],
                "Windows must launch the SCM dispatcher, not the console runner"
            );
        } else {
            assert_eq!(
                config.arguments,
                vec!["run".to_string()],
                "launchd/systemd supervise the foreground process"
            );
        }
    }

    /// The supervisor is handed `executable` + `arguments`; a default that named
    /// no subcommand would start the interactive daemon under a service manager.
    #[test]
    fn default_config_is_installable() {
        let config = ServiceConfig::default();
        assert!(!config.name.is_empty(), "a service needs a name to register under");
        assert!(!config.display_name.is_empty(), "the SCM shows the display name");
        assert_eq!(config.arguments.len(), 1, "exactly one subcommand is passed");
        assert!(
            !config.arguments[0].is_empty(),
            "an empty argument would launch the default (interactive) mode"
        );
    }

    /// The systemd unit interpolates `arguments`; a regression there silently
    /// produces a unit that runs the wrong mode.
    #[cfg(target_os = "linux")]
    #[test]
    fn systemd_unit_execstart_carries_the_configured_argument() {
        let manager = ServiceManager::new(ServiceConfig::default());
        let unit = manager.generate_systemd_unit();
        assert!(
            unit.contains("ExecStart="),
            "unit must define ExecStart, got:\n{unit}"
        );
        assert!(
            unit.contains(" \"run\""),
            "ExecStart must pass the run subcommand as its own quoted word, got:\n{unit}"
        );
    }
}
