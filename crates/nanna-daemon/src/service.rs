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

/// Run a service-manager command and fail unless it exits zero.
///
/// `status()` only reported whether the program could be spawned: a `systemctl
/// --user enable` refused for a bad unit (or with no user session bus) printed
/// its error and `install` still reported success, so the operator learned the
/// daemon would not start at login only at the next login. The error carries
/// the command, its exit status and its stderr.
#[cfg(any(target_os = "macos", target_os = "linux", test))]
fn run_checked(command: &mut std::process::Command) -> Result<(), String> {
    let program = command.get_program().to_string_lossy().into_owned();
    let output = command
        .output()
        .map_err(|e| format!("could not run {program}: {e}"))?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stderr = stderr.trim();
    debug_assert!(!output.status.success());
    Err(if stderr.is_empty() {
        format!("{program} failed ({})", output.status)
    } else {
        format!("{program} failed ({}): {stderr}", output.status)
    })
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
    /// or running `systemctl`; on macOS, writing the launchd plist or running
    /// `launchctl` (a non-zero exit is a failure); on Windows, connecting to
    /// the service manager or creating
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
    /// reached or refused the start, or the command exited non-zero.
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
    /// reached or refused the stop, or the command exited non-zero.
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
        
        if let Some(parent) = plist_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(&plist_path, plist_content)
            .map_err(|e| e.to_string())?;
        
        // Load the service
        run_checked(std::process::Command::new("launchctl").arg("load").arg(&plist_path))
    }
    
    #[cfg(target_os = "macos")]
    fn uninstall_macos(&self) -> Result<(), String> {
        let plist_path = self.launchd_plist_path();
        
        // Unload first
        let _ = std::process::Command::new("launchctl")
            .arg("unload")
            .arg(&plist_path)
            .status();
        
        // Remove plist
        if plist_path.exists() {
            std::fs::remove_file(&plist_path).map_err(|e| e.to_string())?;
        }
        
        Ok(())
    }
    
    #[cfg(target_os = "macos")]
    fn start_macos(&self) -> Result<(), String> {
        run_checked(
            std::process::Command::new("launchctl")
                .args(["start", &format!("com.nanna.{}", self.config.name)]),
        )
    }
    
    #[cfg(target_os = "macos")]
    fn stop_macos(&self) -> Result<(), String> {
        run_checked(
            std::process::Command::new("launchctl")
                .args(["stop", &format!("com.nanna.{}", self.config.name)]),
        )
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
    
    /// The launchd job. Built on every host so its escaping is tested here,
    /// not only on a Mac.
    #[cfg(any(target_os = "macos", test))]
    fn generate_launchd_plist(&self) -> String {
        // Every interpolated string is XML-escaped: a plist is XML, so an
        // install path or argument with `&` or `<` (legal in both) produced a
        // file launchd refused to load — or, with a crafted `</string>`, a
        // different program list.
        let name = xml_escape(&self.config.name);
        let exe = xml_escape(&self.config.executable.to_string_lossy());
        let args: String = self.config.arguments.iter()
            .map(|a| format!("        <string>{}</string>", xml_escape(a)))
            .collect::<Vec<_>>()
            .join("\n");
        
        format!(r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.nanna.{name}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{exe}</string>
{args}
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
</plist>"#)
    }
    
    // =========================================================================
    // Linux implementation (systemd)
    // =========================================================================
    
    #[cfg(target_os = "linux")]
    fn install_linux(&self) -> Result<(), String> {
        let unit_path = self.systemd_unit_path();
        let unit_content = self.generate_systemd_unit();
        
        if let Some(parent) = unit_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(&unit_path, unit_content)
            .map_err(|e| e.to_string())?;
        
        // Reload systemd
        run_checked(std::process::Command::new("systemctl").args(["--user", "daemon-reload"]))?;
        
        // Enable the service
        run_checked(
            std::process::Command::new("systemctl").args(["--user", "enable", &self.config.name]),
        )
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
        run_checked(std::process::Command::new("systemctl").args(["--user", "daemon-reload"]))
    }
    
    #[cfg(target_os = "linux")]
    fn start_linux(&self) -> Result<(), String> {
        run_checked(
            std::process::Command::new("systemctl").args(["--user", "start", &self.config.name]),
        )
    }
    
    #[cfg(target_os = "linux")]
    fn stop_linux(&self) -> Result<(), String> {
        run_checked(
            std::process::Command::new("systemctl").args(["--user", "stop", &self.config.name]),
        )
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

/// `text` as XML character data or attribute content: the five characters
/// XML reserves become entity references, everything else passes through.
#[cfg(any(target_os = "macos", test))]
fn xml_escape(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&apos;"),
            _ => escaped.push(c),
        }
    }
    debug_assert!(escaped.len() >= text.len(), "escaping only ever adds");
    debug_assert!(!escaped.contains('<'), "no markup survives");
    escaped
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A service command that runs but refuses is a failure, with its stderr.
    #[cfg(unix)]
    #[test]
    fn a_service_command_that_exits_non_zero_fails() {
        let refused = run_checked(
            std::process::Command::new("sh").args(["-c", "echo 'Unit not found.' >&2; exit 5"]),
        );
        let message = refused.expect_err("exit 5 is a failure");
        assert!(message.contains("Unit not found."), "{message}");
        assert!(message.starts_with("sh failed"), "{message}");
        assert_eq!(run_checked(std::process::Command::new("sh").args(["-c", "exit 0"])), Ok(()));
        assert!(
            run_checked(&mut std::process::Command::new("/nonexistent/systemctl"))
                .is_err_and(|e| e.starts_with("could not run"))
        );
    }

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

    #[test]
    fn plist_strings_are_xml_escaped() {
        assert_eq!(xml_escape("/Applications/Nanna.app"), "/Applications/Nanna.app");
        assert_eq!(xml_escape(r#"a&b<c>d"e'f"#), "a&amp;b&lt;c&gt;d&quot;e&apos;f");

        let manager = ServiceManager::new(ServiceConfig {
            name: "R&D".to_string(),
            executable: PathBuf::from("/Users/u/Tools & Apps/nanna-daemon"),
            arguments: vec!["run".to_string(), "</string><string>/bin/sh".to_string()],
            ..ServiceConfig::default()
        });
        let plist = manager.generate_launchd_plist();
        assert!(plist.contains("<string>com.nanna.R&amp;D</string>"));
        assert!(plist.contains("<string>/Users/u/Tools &amp; Apps/nanna-daemon</string>"));
        assert!(plist.contains("<string>&lt;/string&gt;&lt;string&gt;/bin/sh</string>"));
        // One <string> per program word plus the label and two log paths:
        // an argument cannot open a new array element.
        assert_eq!(plist.matches("<string>").count(), 6);
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
