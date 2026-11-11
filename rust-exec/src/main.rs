use pirl_exec::{CheckpointJournal, JournalEvent, Limits, Supervisor, WorkspaceGuard};
use serde::Deserialize;
use serde_json::json;
use std::io::{self, BufRead};

#[derive(Deserialize)] struct Request { op: String, workspace: Option<String>, argv: Option<Vec<String>>, patch: Option<String>, limits: Option<Limits>, journal: Option<String>, event: Option<JournalEvent> }

fn main() -> io::Result<()> {
    for line in io::stdin().lock().lines() {
        let req: Request = match serde_json::from_str(&line?) { Ok(v) => v, Err(e) => { println!("{}", json!({"ok":false,"error":e.to_string()})); continue; } };
        let result = handle(req).unwrap_or_else(|e| json!({"ok":false,"error":e}));
        println!("{}", result);
    }
    Ok(())
}

fn handle(req: Request) -> Result<serde_json::Value, String> {
    if req.op == "journal" { let mut j = CheckpointJournal::open(req.journal.ok_or("journal path required")?).map_err(|e|e.to_string())?; j.append(&req.event.ok_or("event required")?).map_err(|e|e.to_string())?; return Ok(json!({"ok":true})); }
    let guard = WorkspaceGuard::new(req.workspace.ok_or("workspace required")?).map_err(|e|e.to_string())?;
    let supervisor = Supervisor::new(guard, req.limits.unwrap_or_default());
    match req.op.as_str() {
        "exec" => Ok(serde_json::to_value(supervisor.run(&req.argv.ok_or("argv required")?).map_err(|e|e.to_string())?).unwrap()),
        "patch" => { supervisor.apply_patch(&req.patch.ok_or("patch required")?).map_err(|e|e.to_string())?; Ok(json!({"ok":true})) },
        _ => Err("unknown op".into()),
    }
}
