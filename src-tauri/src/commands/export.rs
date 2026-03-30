use crate::error::AppError;
use crate::state::AppState;
use serde::Serialize;
use std::fs;
use std::path::Path;
use tauri::{AppHandle, Manager};

#[derive(Serialize)]
pub struct ExportResult {
    pub output_path: String,
    pub total_files: u32,
    pub pair_dirs: Vec<String>,
}

/// Export generated frames as per-part PNG sequences.
/// Output structure:
///   output_dir/
///     eye/frame_001.png, frame_002.png, ...
///     mouth_a/frame_001.png, ...
///     mouth_i/frame_001.png, ...
#[tauri::command]
pub async fn export_frames(
    app: AppHandle,
    output_path: String,
) -> Result<ExportResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || {
        export_frames_inner(app, output_path)
    })
    .await
    .map_err(|e| AppError::General(format!("Task join error: {}", e)))?
}

fn export_frames_inner(
    app: AppHandle,
    output_path: String,
) -> Result<ExportResult, AppError> {
    let state = app.state::<AppState>();
    let generated = state.generated.lock().unwrap();
    let raw_images = state.raw_images.lock().unwrap();

    if generated.is_empty() {
        return Err(AppError::General("フレームが生成されていません".into()));
    }

    let out_dir = Path::new(&output_path);
    fs::create_dir_all(out_dir)?;

    let mut total_files = 0u32;
    let mut pair_dirs = Vec::new();

    // Export body and hair from raw input images
    for key in &["body", "hair"] {
        if let Some(img) = raw_images.get(*key) {
            let path = out_dir.join(format!("{}.png", key));
            img.save(&path)?;
            total_files += 1;
        }
    }

    // Export generated frames per part
    for part_frames in generated.iter() {
        let part_dir = out_dir.join(&part_frames.name);
        fs::create_dir_all(&part_dir)?;

        for (i, frame) in part_frames.frames.iter().enumerate() {
            let filename = format!("frame_{:03}.png", i + 1);
            let path = part_dir.join(&filename);
            frame.save(&path)?;
            total_files += 1;
        }

        pair_dirs.push(part_frames.name.clone());
    }

    Ok(ExportResult {
        output_path: output_path.clone(),
        total_files,
        pair_dirs,
    })
}
