#![warn(clippy::pedantic, clippy::nursery, clippy::all)]
//! A python run with a workdir leaves the process's working directory where it
//! found it. Its own test binary: the run moves the process cwd while it lasts,
//! which no other test in this process may observe.
#![cfg(feature = "python")]

use nanna_scripting::python::PythonEngine;

#[tokio::test]
async fn a_python_workdir_does_not_move_the_daemon() {
    let before = std::env::current_dir().expect("cwd");
    let workdir = tempfile::tempdir().expect("tempdir");
    let canonical = workdir.path().canonicalize().expect("canonical");
    let result = PythonEngine::new()
        .execute(
            "import os\nprint(os.getcwd())",
            Some(&canonical.to_string_lossy()),
            30,
        )
        .await
        .expect("runs");
    assert!(result.success, "{result:?}");
    assert_eq!(result.stdout.trim(), canonical.to_string_lossy());
    assert_eq!(
        std::env::current_dir().expect("cwd"),
        before,
        "the run's directory change was undone"
    );
}
