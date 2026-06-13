// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::{Arc, Mutex};
use tauri::{Emitter, Manager};
use tauri_plugin_shell::ShellExt;

/// Sidecar lifecycle manager state.
/// Holds the child process handle for graceful stdin-based shutdown.
struct SidecarState {
    child: Arc<Mutex<Option<tauri_plugin_shell::process::CommandChild>>>,
}

/// Spawns the backend sidecar process and wires up stdout/stderr event forwarding.
///
/// - **Dev mode** (`debug_assertions`): runs `python3 backend/main.py --dev` directly
///   via the OS command, expecting the backend source at `<project_root>/backend/main.py`.
/// - **Release mode**: spawns the bundled sidecar binary `bin/api/main` (configured in
///   `tauri.conf.json` → `externalBin`).
///
/// A dedicated thread reads `CommandEvent`s from the child's `Receiver` and emits
/// Tauri events (`sidecar-stdout`, `sidecar-stderr`, `sidecar-exit`) so the
/// frontend can react in real time.
fn spawn_sidecar(app: &tauri::AppHandle) -> Result<(), String> {
    use tauri_plugin_shell::process::CommandEvent;

    // Resolve the project root reliably.
    //
    // During `tauri dev`, Tauri sets the CWD to the *project root* (the directory
    // containing `package.json` / `src-tauri/`), so we can use it directly.
    //
    // We also try the canonical path derived from the binary location as a fallback.
    let project_root = {
        let cwd = std::env::current_dir().unwrap_or_default();
        // If `backend/` exists under the CWD, we're already at the project root.
        if cwd.join("backend").exists() {
            cwd
        } else if let Some(parent) = cwd.parent() {
            // One level up (e.g., if CWD somehow ended up inside src-tauri/)
            if parent.join("backend").exists() {
                parent.to_path_buf()
            } else {
                cwd // best-effort
            }
        } else {
            cwd
        }
    };

    let (mut rx, child) = if cfg!(debug_assertions) {
        // Dev: run the Python backend from the venv directly.
        // venv layout differs per OS: Windows uses Scripts\python.exe,
        // Unix uses bin/python3.
        let venv_python = if cfg!(windows) {
            project_root.join("backend/.venv/Scripts/python.exe")
        } else {
            project_root.join("backend/.venv/bin/python3")
        };
        let python_bin = if venv_python.exists() {
            venv_python.to_string_lossy().to_string()
        } else if cfg!(windows) {
            "python".to_string()
        } else {
            "python3".to_string()
        };

        app.shell()
            .command(&python_bin)
            .args(["backend/main.py", "--dev"])
            .current_dir(&project_root)
            .spawn()
            .map_err(|e| format!("Failed to spawn dev sidecar: {}", e))?
    } else {
        // Release: use the bundled sidecar binary.
        app.shell()
            .sidecar("bin/api/main")
            .map_err(|e| format!("Failed to create sidecar command: {}", e))?
            .spawn()
            .map_err(|e| format!("Failed to spawn release sidecar: {}", e))?
    };

    // Store the child process handle for later graceful shutdown.
    let state = app.state::<SidecarState>();
    *state.child.lock().unwrap() = Some(child);

    // Spawn an async task to forward sidecar output as Tauri events.
    // rx.recv() is async (tauri::async_runtime / tokio channel), so we must
    // run this inside tauri::async_runtime::spawn, not std::thread::spawn.
    let app_handle = app.clone();
    tauri::async_runtime::spawn(async move {
        while let Some(event) = rx.recv().await {
            match event {
                CommandEvent::Stdout(line) => {
                    let _ = app_handle.emit("sidecar-stdout", &line);
                }
                CommandEvent::Stderr(line) => {
                    let _ = app_handle.emit("sidecar-stderr", &line);
                }
                CommandEvent::Terminated(status) => {
                    // .code is a pub field (Option<i32>), not a method.
                    let code = status.code;
                    let _ = app_handle.emit("sidecar-exit", serde_json::json!({
                        "code": code,
                        "success": code == Some(0),
                    }));
                    break;
                }
                CommandEvent::Error(err) => {
                    let _ = app_handle.emit("sidecar-stderr", format!("[sidecar-error] {}", err));
                    break;
                }
                // Ignore other variants (e.g. DragDrop, etc.)
                _ => {}
            }
        }
        // Channel closed — emit exit event.
        let _ = app_handle.emit("sidecar-exit", serde_json::json!({
            "code": null,
            "success": false,
        }));
    });

    Ok(())
}

/// Sends a graceful shutdown message to the sidecar via stdin.
/// The backend is expected to handle the `"sidecar shutdown\n"` message
/// and exit cleanly. **NEVER** calls `process.kill()` — always use
/// stdin-based graceful shutdown.
fn shutdown_sidecar(app: &tauri::AppHandle) -> Result<(), String> {
    let state = app.state::<SidecarState>();
    let mut guard = state.child.lock().unwrap();

    if let Some(ref mut child) = *guard {
        child
            .write(b"sidecar shutdown\n")
            .map_err(|e| format!("Failed to send shutdown to sidecar: {}", e))?;
        *guard = None;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tauri commands — exposed to the frontend via invoke()
// ---------------------------------------------------------------------------

/// Manually start (or restart) the backend sidecar.
///
/// If a sidecar is already running it is shut down gracefully first.
#[tauri::command]
fn start_sidecar(app: tauri::AppHandle) -> Result<(), String> {
    // Shut down any existing instance first.
    let _ = shutdown_sidecar(&app);
    spawn_sidecar(&app)
}

/// Manually shut down the backend sidecar gracefully.
#[tauri::command]
fn stop_sidecar(app: tauri::AppHandle) -> Result<(), String> {
    shutdown_sidecar(&app)
}

/// Ping the sidecar's health endpoint.
///
/// Calls `GET http://localhost:8000/api/health` and returns the JSON body
/// so the frontend can verify the backend is alive and ready.
#[tauri::command]
fn ping_sidecar() -> Result<serde_json::Value, String> {
    let resp = ureq::get("http://localhost:8008/api/health")
        .timeout(std::time::Duration::from_secs(5))
        .call()
        .map_err(|e| format!("Health check failed: {}", e))?;

    let body: serde_json::Value = resp
        .into_json::<serde_json::Value>()
        .map_err(|e| format!("Failed to parse health response: {}", e))?;

    Ok(body)
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(SidecarState {
            child: Arc::new(Mutex::new(None)),
        })
        // Hook into window-close events at the builder level (correct Tauri v2 API).
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                let _ = shutdown_sidecar(&window.app_handle());
            }
        })
        .setup(|app| {
            // Auto-start the sidecar when the application launches.
            let handle = app.handle().clone();
            if let Err(e) = spawn_sidecar(&handle) {
                eprintln!("[LAIDocs] sidecar auto-start failed: {}", e);
                // Don't panic — the frontend can retry via `start_sidecar` command.
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![start_sidecar, stop_sidecar, ping_sidecar])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
