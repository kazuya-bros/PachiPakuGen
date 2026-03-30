use crate::error::AppError;
use ort::session::Session;
use std::path::{Path, PathBuf};
use tauri::{AppHandle, Manager};

/// Create an ONNX Runtime session with DirectML (GPU) and CPU fallback.
pub fn create_session(model_path: &Path) -> Result<Session, AppError> {
    if !model_path.exists() {
        return Err(AppError::General(format!(
            "Model file not found: {}",
            model_path.display()
        )));
    }

    eprintln!("[PachiPakuGen] Loading model: {}", model_path.display());

    // Try DirectML + CPU first
    match Session::builder()
        .and_then(|b| {
            b.with_execution_providers([
                ort::ep::DirectML::default().build(),
                ort::ep::CPU::default().build(),
            ])
        })
        .and_then(|b| b.commit_from_file(model_path))
    {
        Ok(session) => {
            eprintln!("[PachiPakuGen] Session created with DirectML+CPU");
            return Ok(session);
        }
        Err(e) => {
            eprintln!("[PachiPakuGen] DirectML failed ({}), trying CPU only...", e);
        }
    }

    // Fallback: CPU only
    let session = Session::builder()?
        .commit_from_file(model_path)?;

    eprintln!("[PachiPakuGen] Session created with CPU only");
    Ok(session)
}

/// Resolve model path: check resource dir first, then fallback to local models/ dir.
pub fn resolve_model_path(app: &AppHandle, filename: &str) -> Result<PathBuf, AppError> {
    // Try Tauri resource directory first (for bundled releases)
    if let Ok(path) = app
        .path()
        .resolve(format!("models/{}", filename), tauri::path::BaseDirectory::Resource)
    {
        if path.exists() {
            return Ok(path);
        }
    }

    // Fallback: models/ directory next to the executable (dev mode)
    let exe_dir = std::env::current_exe()
        .map_err(|e| AppError::General(format!("Failed to get exe path: {}", e)))?
        .parent()
        .unwrap()
        .to_path_buf();
    let local_path = exe_dir.join("models").join(filename);
    if local_path.exists() {
        return Ok(local_path);
    }

    // Fallback: src-tauri/models/ (for development)
    let dev_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join(filename);
    if dev_path.exists() {
        return Ok(dev_path);
    }

    Err(AppError::General(format!(
        "Model file '{}' not found. Place rife.onnx in the models/ directory.",
        filename
    )))
}
