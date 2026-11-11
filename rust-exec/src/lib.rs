//! Deterministic execution boundary for pirl. Planning remains in Python;
//! this crate owns paths, argv execution, patch validation and journaling.

use serde::{Deserialize, Serialize};
use std::{fs, io::{self, Read, Write}, path::{Path, PathBuf}, process::{Command, Stdio}, time::Duration};
use wait_timeout::ChildExt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Limits {
    pub timeout_ms: u64,
    pub max_output_bytes: usize,
    pub max_memory_bytes: Option<u64>,
    pub max_cpu_seconds: Option<u64>,
    pub network: bool,
}

impl Default for Limits {
    fn default() -> Self { Self { timeout_ms: 60_000, max_output_bytes: 20_000, max_memory_bytes: None, max_cpu_seconds: None, network: false } }
}

#[derive(Debug)]
pub struct WorkspaceGuard { root: PathBuf }

impl WorkspaceGuard {
    pub fn new(root: impl AsRef<Path>) -> io::Result<Self> {
        let root = fs::canonicalize(root)?;
        if !root.is_dir() { return Err(io::Error::new(io::ErrorKind::InvalidInput, "workspace is not a directory")); }
        Ok(Self { root })
    }
    pub fn root(&self) -> &Path { &self.root }
    pub fn resolve(&self, raw: impl AsRef<Path>, must_exist: bool) -> io::Result<PathBuf> {
        let raw = raw.as_ref();
        let candidate = if raw.is_absolute() { raw.to_path_buf() } else { self.root.join(raw) };
        let checked = if must_exist { fs::canonicalize(candidate)? } else {
            let parent = candidate.parent().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "missing parent"))?;
            fs::canonicalize(parent)?.join(candidate.file_name().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "missing filename"))?)
        };
        if !checked.starts_with(&self.root) { return Err(io::Error::new(io::ErrorKind::PermissionDenied, "path escapes workspace")); }
        Ok(checked)
    }
    pub fn validate_patch_paths(&self, patch: &str) -> io::Result<()> {
        for line in patch.lines() {
            if let Some(raw) = line.strip_prefix("+++ ").or_else(|| line.strip_prefix("--- ")) {
                let raw = raw.split('\t').next().unwrap_or(raw);
                let path = raw.strip_prefix("a/").or_else(|| raw.strip_prefix("b/")).unwrap_or(raw);
                if path != "/dev/null" { self.resolve(path, false)?; }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct ExecResult { pub ok: bool, pub code: Option<i32>, pub stdout: String, pub stderr: String, pub timed_out: bool, pub truncated: bool }

pub struct Supervisor { guard: WorkspaceGuard, limits: Limits }

impl Supervisor {
    pub fn new(guard: WorkspaceGuard, limits: Limits) -> Self { Self { guard, limits } }
    pub fn run(&self, argv: &[String]) -> io::Result<ExecResult> {
        if argv.is_empty() { return Err(io::Error::new(io::ErrorKind::InvalidInput, "argv is empty")); }
        let mut command = Command::new(&argv[0]);
        command.args(&argv[1..]).current_dir(self.guard.root()).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
        #[cfg(unix)] {
            use std::os::unix::process::CommandExt;
            let limits = self.limits.clone();
            unsafe {
                command.pre_exec(move || {
                    if let Some(bytes) = limits.max_memory_bytes {
                        let lim = libc::rlimit { rlim_cur: bytes, rlim_max: bytes };
                        if libc::setrlimit(libc::RLIMIT_AS, &lim) != 0 { return Err(io::Error::last_os_error()); }
                    }
                    if let Some(seconds) = limits.max_cpu_seconds {
                        let lim = libc::rlimit { rlim_cur: seconds, rlim_max: seconds };
                        if libc::setrlimit(libc::RLIMIT_CPU, &lim) != 0 { return Err(io::Error::last_os_error()); }
                    }
                    if !limits.network {
                        // Fail closed: if the host cannot create a private
                        // network namespace, do not run a supposedly offline
                        // command with inherited network access.
                        if libc::unshare(libc::CLONE_NEWNET) != 0 { return Err(io::Error::last_os_error()); }
                    }
                    Ok(())
                });
            }
        }
        let mut child = command.spawn()?;
        #[cfg(unix)] unsafe { libc::setpgid(child.id() as libc::pid_t, child.id() as libc::pid_t); }
        let mut stdout = child.stdout.take().unwrap();
        let mut stderr = child.stderr.take().unwrap();
        let out_thread = std::thread::spawn(move || { let mut b = Vec::new(); let _ = stdout.read_to_end(&mut b); b });
        let err_thread = std::thread::spawn(move || { let mut b = Vec::new(); let _ = stderr.read_to_end(&mut b); b });
        let timeout = Duration::from_millis(self.limits.timeout_ms);
        let timed_out = child.wait_timeout(timeout)?.is_none();
        if timed_out {
            #[cfg(unix)] unsafe { libc::kill(-(child.id() as libc::pid_t), libc::SIGKILL); }
            #[cfg(windows)] { let _ = Command::new("taskkill").args(["/PID", &child.id().to_string(), "/T", "/F"]).status(); }
            let _ = child.wait();
        }
        let status = child.try_wait()?.or_else(|| Some(child.wait().ok()?));
        let out = out_thread.join().unwrap_or_default();
        let err = err_thread.join().unwrap_or_default();
        let (stdout, a) = cap(out, self.limits.max_output_bytes);
        let (stderr, b) = cap(err, self.limits.max_output_bytes);
        let success = status.as_ref().map(|s| s.success()).unwrap_or(false);
        let code = status.as_ref().and_then(|s| s.code());
        Ok(ExecResult { ok: !timed_out && success, code, stdout, stderr, timed_out, truncated: a || b })
    }
    pub fn apply_patch(&self, patch: &str) -> io::Result<()> {
        self.guard.validate_patch_paths(patch)?;
        let check = git_apply(self.guard.root(), patch, true)?;
        if !check.success() { return Err(io::Error::new(io::ErrorKind::InvalidData, "patch rejected")); }
        let applied = git_apply(self.guard.root(), patch, false)?;
        if !applied.success() { return Err(io::Error::new(io::ErrorKind::Other, "patch failed")); }
        Ok(())
    }
}

fn git_apply(root: &Path, patch: &str, check: bool) -> io::Result<std::process::ExitStatus> {
    let mut args = vec!["apply", "--whitespace=nowarn"];
    if check { args.push("--check"); }
    args.push("-");
    let mut child = Command::new("git").args(args).current_dir(root).stdin(Stdio::piped()).spawn()?;
    child.stdin.take().unwrap().write_all(patch.as_bytes())?;
    child.wait()
}

fn cap(bytes: Vec<u8>, limit: usize) -> (String, bool) { let truncated = bytes.len() > limit; (String::from_utf8_lossy(&bytes[..bytes.len().min(limit)]).to_string(), truncated) }

#[derive(Debug, Serialize, Deserialize)]
pub struct JournalEvent { pub run_id: String, pub kind: String, pub payload: serde_json::Value, pub ts: String }

pub struct CheckpointJournal { file: fs::File }
impl CheckpointJournal {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> { if let Some(parent) = path.as_ref().parent() { fs::create_dir_all(parent)?; } Ok(Self { file: fs::OpenOptions::new().create(true).append(true).open(path)? }) }
    pub fn append(&mut self, event: &JournalEvent) -> io::Result<()> {
        serde_json::to_writer(&mut self.file, event).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        self.file.write_all(b"\n")?;
        self.file.sync_data()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, path::PathBuf, time::{SystemTime, UNIX_EPOCH}};

    fn temp_dir(label: &str) -> PathBuf {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let path = std::env::temp_dir().join(format!("pirl-exec-{label}-{nonce}"));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn workspace_guard_accepts_inside_and_rejects_escape() {
        let root = temp_dir("guard");
        fs::write(root.join("file.txt"), "ok").unwrap();
        let guard = WorkspaceGuard::new(&root).unwrap();
        assert!(guard.resolve("file.txt", true).unwrap().starts_with(guard.root()));
        assert!(guard.resolve("../outside.txt", false).is_err());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn workspace_guard_rejects_patch_escape() {
        let root = temp_dir("patch-path");
        let guard = WorkspaceGuard::new(&root).unwrap();
        let patch = "--- a/../outside.txt\n+++ b/../outside.txt\n";
        assert!(guard.validate_patch_paths(patch).is_err());
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn supervisor_executes_argv_without_shell() {
        let root = temp_dir("exec");
        let guard = WorkspaceGuard::new(&root).unwrap();
        let limits = Limits { network: true, ..Limits::default() };
        let supervisor = Supervisor::new(guard, limits);
        let result = supervisor.run(&["printf".into(), "hello".into()]).unwrap();
        assert!(result.ok);
        assert_eq!(result.stdout, "hello");
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn supervisor_caps_output() {
        let root = temp_dir("output-cap");
        let guard = WorkspaceGuard::new(&root).unwrap();
        let limits = Limits { network: true, max_output_bytes: 4, ..Limits::default() };
        let supervisor = Supervisor::new(guard, limits);
        let result = supervisor.run(&["printf".into(), "123456789".into()]).unwrap();
        assert!(result.ok);
        assert_eq!(result.stdout, "1234");
        assert!(result.truncated);
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn supervisor_times_out_and_reports_failure() {
        let root = temp_dir("timeout");
        let guard = WorkspaceGuard::new(&root).unwrap();
        let limits = Limits { network: true, timeout_ms: 20, ..Limits::default() };
        let supervisor = Supervisor::new(guard, limits);
        let result = supervisor.run(&["sleep".into(), "1".into()]).unwrap();
        assert!(result.timed_out);
        assert!(!result.ok);
        let _ = fs::remove_dir_all(root);
    }
}
