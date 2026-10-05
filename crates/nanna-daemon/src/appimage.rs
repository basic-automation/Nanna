//! Run the daemon from a copy outside the `AppImage`'s FUSE mount.
//!
//! The Linux GUI ships as an `AppImage`: its runtime mounts the image at
//! `/tmp/.mount_Nanna_*` and serves every page of `nanna-daemon`, and of the
//! bundled `libssl`/`libcrypto`/`libffi`/`libzstd`/`libbrotli*` `AppRun` puts
//! on `LD_LIBRARY_PATH`, through FUSE on demand. The mount lives exactly as
//! long as the GUI. The daemon does not: `--exit-with-parent` notices the
//! GUI's exit up to a second later, and its drain runs after that. A page the
//! drain faults in from a mount that is gone is a SIGBUS, so every reboot,
//! logout, or GUI kill turned the daemon's graceful shutdown into a crash in
//! the middle of its last database writes. That was eight of the nine unclean
//! exits on record from 2026-09-18 to 2026-09-28, and `systemd-coredump`
//! reported each one as `nanna-daemon` crashing.
//!
//! So a daemon started from inside the mount copies itself, and the bundle's
//! libraries it has mapped, to the user's cache, and re-execs from there.
//! `execve` keeps the PID, the parent, and the stdio pipes, so the GUI's
//! handle on its sidecar is unchanged. Staging is best-effort: any failure
//! leaves the daemon running from the mount, as before.

use std::ffi::{OsStr, OsString};
use std::os::unix::process::CommandExt as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest as _, Sha256};

/// Set in the staged copy's environment: this process already runs from the
/// staging directory, so it must not stage again.
const STAGED_MARKER: &str = "NANNA_APPIMAGE_STAGED";

/// The `LD_LIBRARY_PATH` the daemon was started with, handed across the
/// re-exec so the staged copy restores it for its own children. Absent when
/// the variable was unset.
const ORIGINAL_LIBRARY_PATH: &str = "NANNA_APPIMAGE_ORIGINAL_LD_LIBRARY_PATH";

/// Written last into a staging directory: its files are all there and the
/// copy was seen to start.
const COMPLETE_MARKER: &str = ".complete";

/// What [`run_outside_the_mount`] did, for `main` to log once tracing is up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// Not started from inside an `AppImage` mount: nothing to do.
    NotInAppImage,
    /// This process is the staged copy, running from `exe`, out of the mount.
    Staged { exe: PathBuf },
    /// Staging failed; the daemon runs from the mount, as before.
    Failed(String),
}

/// If this process runs from inside an `AppImage` mount, re-exec it from a
/// staged copy outside the mount; this does not return then.
///
/// Call it first thing in `main`, before any thread exists: the staged copy
/// rewrites its own environment here.
#[must_use]
pub fn run_outside_the_mount() -> Outcome {
    if std::env::var_os(STAGED_MARKER).is_some() {
        restore_environment();
        return std::env::current_exe().map_or_else(
            |e| {
                Outcome::Failed(format!(
                    "the staged copy cannot name its own executable: {e}"
                ))
            },
            |exe| Outcome::Staged { exe },
        );
    }
    let Some(appdir) = std::env::var_os("APPDIR").filter(|dir| !dir.is_empty()) else {
        return Outcome::NotInAppImage;
    };
    let appdir = PathBuf::from(appdir);
    let exe = match std::fs::canonicalize("/proc/self/exe") {
        Ok(exe) => exe,
        Err(e) => return Outcome::Failed(format!("cannot resolve /proc/self/exe: {e}")),
    };
    if !exe.starts_with(&appdir) {
        return Outcome::NotInAppImage;
    }
    let Err(reason) = stage_and_exec(&appdir, &exe);
    Outcome::Failed(reason)
}

/// Put back what the re-exec changed, so this process and its children see
/// the environment the daemon was started with.
fn restore_environment() {
    let original = std::env::var_os(ORIGINAL_LIBRARY_PATH);
    // SAFETY: called from the top of `main`, before this process has started
    // any thread, so nothing reads the environment concurrently.
    unsafe {
        match original {
            Some(value) => std::env::set_var("LD_LIBRARY_PATH", value),
            None => std::env::remove_var("LD_LIBRARY_PATH"),
        }
        std::env::remove_var(ORIGINAL_LIBRARY_PATH);
        std::env::remove_var(STAGED_MARKER);
    }
}

/// Stage the executable and the libraries mapped from the mount, then exec
/// the staged copy. Returns only on failure.
fn stage_and_exec(appdir: &Path, exe: &Path) -> Result<std::convert::Infallible, String> {
    let maps = std::fs::read_to_string("/proc/self/maps")
        .map_err(|e| format!("cannot read /proc/self/maps: {e}"))?;
    let libraries: Vec<PathBuf> = mapped_files_under(&maps, appdir)
        .into_iter()
        .filter(|path| path != exe)
        .collect();
    let root = cache_root().ok_or("neither XDG_CACHE_HOME nor HOME is set")?;
    let key = staging_key(exe, &libraries)?;
    let staged = root.join(&key);
    let original_library_path = std::env::var_os("LD_LIBRARY_PATH");
    let library_path = staged_library_path(
        &staged.join("lib"),
        appdir,
        original_library_path.as_deref(),
    )?;

    if !staged.join(COMPLETE_MARKER).is_file() {
        stage(
            &root,
            &key,
            exe,
            &libraries,
            appdir,
            original_library_path.as_deref(),
        )?;
    }
    prune_other_stagings(&root, &key);

    let mut command = Command::new(staged.join("nanna-daemon"));
    command
        .args(std::env::args_os().skip(1))
        .env("LD_LIBRARY_PATH", &library_path)
        .env(STAGED_MARKER, "1");
    match original_library_path {
        Some(original) => command.env(ORIGINAL_LIBRARY_PATH, original),
        None => command.env_remove(ORIGINAL_LIBRARY_PATH),
    };
    Err(format!(
        "exec of the staged copy failed: {}",
        command.exec()
    ))
}

/// Copy everything into a fresh directory next to its final place, check the
/// copy starts, and move it into place.
fn stage(
    root: &Path,
    key: &str,
    exe: &Path,
    libraries: &[PathBuf],
    appdir: &Path,
    original_library_path: Option<&OsStr>,
) -> Result<(), String> {
    let partial = root.join(format!(".{key}.partial-{}", std::process::id()));
    let result = fill(&partial, exe, libraries, appdir, original_library_path).and_then(|()| {
        match std::fs::rename(&partial, root.join(key)) {
            Ok(()) => Ok(()),
            // Another daemon staged the same build first: use its copy.
            Err(_) if root.join(key).join(COMPLETE_MARKER).is_file() => Ok(()),
            Err(e) => Err(format!("cannot move the staged copy into place: {e}")),
        }
    });
    // Gone after a successful rename; a failed stage leaves nothing behind.
    let _ = std::fs::remove_dir_all(&partial);
    result
}

/// The body of [`stage`]: `dir` gets `nanna-daemon`, `lib/`, and the marker.
fn fill(
    dir: &Path,
    exe: &Path,
    libraries: &[PathBuf],
    appdir: &Path,
    original_library_path: Option<&OsStr>,
) -> Result<(), String> {
    let lib_dir = dir.join("lib");
    std::fs::create_dir_all(&lib_dir)
        .map_err(|e| format!("cannot create {}: {e}", lib_dir.display()))?;
    let staged_exe = dir.join("nanna-daemon");
    copy(exe, &staged_exe)?;
    for library in libraries {
        let name = library
            .file_name()
            .ok_or_else(|| format!("{} has no file name", library.display()))?;
        let target = lib_dir.join(name);
        if target.exists() {
            return Err(format!(
                "two mapped libraries are named {}",
                name.to_string_lossy()
            ));
        }
        copy(library, &target)?;
        link_aliases(library, name, &lib_dir, appdir, original_library_path);
    }
    // A copy the loader cannot start would turn a crash at shutdown into a
    // daemon that never starts: prove it runs, against this directory's own
    // `lib/`, before anything execs it.
    let library_path = staged_library_path(&lib_dir, appdir, original_library_path)?;
    let check = Command::new(&staged_exe)
        .arg("--version")
        .env("LD_LIBRARY_PATH", library_path)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map_err(|e| format!("cannot run the staged copy: {e}"))?;
    if !check.success() {
        return Err(format!("the staged copy does not start ({check})"));
    }
    std::fs::write(dir.join(COMPLETE_MARKER), b"")
        .map_err(|e| format!("cannot mark the staged copy complete: {e}"))
}

/// `std::fs::copy`, with the paths in its error.
fn copy(from: &Path, to: &Path) -> Result<(), String> {
    std::fs::copy(from, to)
        .map(|_| ())
        .map_err(|e| format!("cannot copy {} to {}: {e}", from.display(), to.display()))
}

/// `/proc/self/maps` names the file a library resolved to; the loader looks
/// it up by its soname, which may be a symlink beside it. Give the copy every
/// name in the bundle's library directories that resolves to the same file.
fn link_aliases(
    library: &Path,
    name: &OsStr,
    lib_dir: &Path,
    appdir: &Path,
    original_library_path: Option<&OsStr>,
) {
    let Some(original) = original_library_path else {
        return;
    };
    for dir in std::env::split_paths(original).filter(|dir| dir.starts_with(appdir)) {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let alias = entry.file_name();
            if alias == name || lib_dir.join(&alias).exists() {
                continue;
            }
            if std::fs::canonicalize(entry.path()).is_ok_and(|real| real == library) {
                let _ = std::os::unix::fs::symlink(name, lib_dir.join(&alias));
            }
        }
    }
}

/// Remove the stagings of other builds. A daemon still running from one
/// keeps its files: Linux frees an unlinked file only once nothing maps it.
fn prune_other_stagings(root: &Path, keep: &str) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        if entry.file_name() != keep {
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
}

/// Where stagings live: `$XDG_CACHE_HOME/nanna/appimage-daemon`, or
/// `~/.cache/nanna/appimage-daemon`.
fn cache_root() -> Option<PathBuf> {
    let cache = std::env::var_os("XDG_CACHE_HOME")
        .filter(|dir| !dir.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .filter(|dir| !dir.is_empty())
                .map(|home| PathBuf::from(home).join(".cache"))
        })?;
    Some(cache.join("nanna").join("appimage-daemon"))
}

/// The files `maps` (the text of `/proc/<pid>/maps`) shows mapped from under
/// `root`, each once, in order. Pure.
fn mapped_files_under(maps: &str, root: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = Vec::new();
    for line in maps.lines() {
        // The path is the last column and the only one that starts with '/'.
        let Some(start) = line.find(" /") else {
            continue;
        };
        let path = &line[start + 1..];
        if path.ends_with(" (deleted)") {
            continue;
        }
        let path = PathBuf::from(path);
        if path.starts_with(root) && !files.contains(&path) {
            files.push(path);
        }
    }
    files
}

/// The staged copy's `LD_LIBRARY_PATH`: its own `lib_dir` first, then the
/// original entries that are not the bundle's (nothing may resolve to the
/// mount any more). Pure.
fn staged_library_path(
    lib_dir: &Path,
    appdir: &Path,
    original: Option<&OsStr>,
) -> Result<OsString, String> {
    let mut entries = vec![lib_dir.to_path_buf()];
    if let Some(original) = original {
        entries.extend(std::env::split_paths(original).filter(|dir| !dir.starts_with(appdir)));
    }
    std::env::join_paths(entries).map_err(|e| format!("cannot build LD_LIBRARY_PATH: {e}"))
}

/// Names one build's staging: this daemon's version plus the size and mtime
/// of every staged file. An update ships new files, so it gets a new key.
fn staging_key(exe: &Path, libraries: &[PathBuf]) -> Result<String, String> {
    let mut hasher = Sha256::new();
    hasher.update(env!("CARGO_PKG_VERSION").as_bytes());
    for file in std::iter::once(exe).chain(libraries.iter().map(PathBuf::as_path)) {
        let meta =
            std::fs::metadata(file).map_err(|e| format!("cannot stat {}: {e}", file.display()))?;
        let mtime = meta
            .modified()
            .ok()
            .and_then(|at| at.duration_since(std::time::UNIX_EPOCH).ok())
            .unwrap_or_default();
        hasher.update(file.file_name().unwrap_or_default().as_encoded_bytes());
        hasher.update(meta.len().to_le_bytes());
        hasher.update(mtime.as_nanos().to_le_bytes());
    }
    let digest = hex::encode(hasher.finalize());
    Ok(format!("{}-{}", env!("CARGO_PKG_VERSION"), &digest[..16]))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MOUNT: &str = "/tmp/.mount_Nanna_kMlAkL";

    #[test]
    fn only_files_mapped_from_the_mount_are_staged_once_each() {
        let maps = format!(
            "5c0000000000-5c0000001000 r--p 00000000 00:3a 12 {MOUNT}/usr/bin/nanna-daemon\n\
             5c0000001000-5c0000009000 r-xp 00001000 00:3a 12 {MOUNT}/usr/bin/nanna-daemon\n\
             7f0000000000-7f0000001000 r-xp 00000000 00:3a 40 {MOUNT}/usr/lib/libssl.so.3\n\
             7f0000002000-7f0000003000 r-xp 00000000 103:02 9 /usr/lib/libc.so.6\n\
             7f0000004000-7f0000005000 rw-p 00000000 00:00 0 \n\
             7ffd00000000-7ffd00021000 rw-p 00000000 00:00 0                          [stack]\n\
             7f0000006000-7f0000007000 r-xp 00000000 00:3a 41 {MOUNT}/usr/lib/libold.so (deleted)\n\
             7f0000008000-7f0000009000 r--p 00000000 103:02 9 {MOUNT}2/usr/lib/libz.so.1\n"
        );
        assert_eq!(
            mapped_files_under(&maps, Path::new(MOUNT)),
            vec![
                PathBuf::from(format!("{MOUNT}/usr/bin/nanna-daemon")),
                PathBuf::from(format!("{MOUNT}/usr/lib/libssl.so.3")),
            ],
            "a sibling mount whose name merely extends ours is not ours"
        );
    }

    #[test]
    fn a_path_with_spaces_is_kept_whole() {
        let maps = format!("7f00-7f01 r-xp 00000000 00:3a 40 {MOUNT}/usr/lib/my lib.so\n");
        assert_eq!(
            mapped_files_under(&maps, Path::new(MOUNT)),
            vec![PathBuf::from(format!("{MOUNT}/usr/lib/my lib.so"))]
        );
    }

    #[test]
    fn the_staged_library_path_leads_with_the_copy_and_drops_the_mount() {
        let original =
            format!("{MOUNT}/usr/lib/:/opt/cuda/lib64:{MOUNT}/usr/lib/x86_64-linux-gnu/");
        let path = staged_library_path(
            Path::new("/home/u/.cache/nanna/appimage-daemon/k/lib"),
            Path::new(MOUNT),
            Some(OsStr::new(&original)),
        );
        assert_eq!(
            path,
            Ok(OsString::from(
                "/home/u/.cache/nanna/appimage-daemon/k/lib:/opt/cuda/lib64"
            ))
        );
    }

    #[test]
    fn an_unset_library_path_stages_to_the_copy_alone() {
        let path = staged_library_path(Path::new("/c/lib"), Path::new(MOUNT), None);
        assert_eq!(path, Ok(OsString::from("/c/lib")));
    }

    #[test]
    fn the_key_names_the_build_and_changes_with_its_files() {
        let dir = tempfile::tempdir().expect("temp dir");
        let exe = dir.path().join("nanna-daemon");
        let lib = dir.path().join("libssl.so.3");
        std::fs::write(&exe, b"daemon").expect("write exe");
        std::fs::write(&lib, b"ssl").expect("write lib");
        let libraries = vec![lib.clone()];

        let first = staging_key(&exe, &libraries).expect("key");
        assert_eq!(
            staging_key(&exe, &libraries),
            Ok(first.clone()),
            "stable for one build"
        );
        assert!(first.starts_with(concat!(env!("CARGO_PKG_VERSION"), "-")));

        std::fs::write(&lib, b"ssl, patched").expect("rewrite lib");
        assert_ne!(
            staging_key(&exe, &libraries),
            Ok(first),
            "a changed library is a new build"
        );
    }
}
