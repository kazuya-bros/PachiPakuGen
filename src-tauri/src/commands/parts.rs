use crate::error::AppError;
use crate::processing::image_utils;
use crate::state::AppState;
use serde::Serialize;
use tauri::{AppHandle, Manager};

#[derive(Serialize)]
pub struct PartPreview {
    pub preview: String,
    pub width: u32,
    pub height: u32,
    pub has_alpha: bool,
}

#[derive(Serialize)]
pub struct LoadPartsResult {
    pub canvas_width: u32,
    pub canvas_height: u32,
    pub part_count: usize,
}

/// Generate a thumbnail preview for a single part image (preserves alpha).
#[tauri::command]
pub async fn load_part_preview(path: String) -> Result<PartPreview, AppError> {
    tauri::async_runtime::spawn_blocking(move || {
        let img = image::open(&path)?;
        let has_alpha = img.color().has_alpha();
        let thumb = img.thumbnail(200, 200);
        let preview = image_utils::image_to_base64_png(&thumb);
        Ok(PartPreview {
            preview,
            width: img.width(),
            height: img.height(),
            has_alpha,
        })
    })
    .await
    .map_err(|e| AppError::General(format!("Task join error: {}", e)))?
}

/// Load all part images into state. Does NOT run segmentation.
/// Segmentation is handled separately in Step 2.
///
/// Expected parts JSON: { "body": "path", "hair": "path",
///   "eye_open": "path", "eye_closed": "path",
///   "mouth_closed": "path", "mouth_a": "path", ... }
#[tauri::command]
pub async fn load_parts(
    app: AppHandle,
    parts_json: String,
) -> Result<LoadPartsResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || load_parts_inner(app, parts_json))
        .await
        .map_err(|e| AppError::General(format!("Task join error: {}", e)))?
}

fn load_parts_inner(
    app: AppHandle,
    parts_json: String,
) -> Result<LoadPartsResult, AppError> {
    let state = app.state::<AppState>();

    let paths: std::collections::HashMap<String, String> = serde_json::from_str(&parts_json)
        .map_err(|e| AppError::General(format!("Invalid JSON: {}", e)))?;

    // body is required to determine canvas size
    let body_path = paths
        .get("body")
        .ok_or_else(|| AppError::General("素体(body)パーツが指定されていません".into()))?;
    let body = image::open(body_path)?;
    let (w, h) = (body.width(), body.height());

    // Load all raw images, resize to match body if needed
    let mut raw = std::collections::HashMap::new();
    for (key, path) in &paths {
        let img = image::open(path)?;
        let img = if img.width() != w || img.height() != h {
            img.resize_exact(w, h, image::imageops::FilterType::Lanczos3)
        } else {
            img
        };
        raw.insert(key.clone(), img);
    }

    let part_count = raw.len();

    // Store in state (no segmentation yet — that happens in Step 2)
    *state.raw_images.lock().unwrap() = raw;
    state.parts.lock().unwrap().clear();
    *state.raw_eye_mask.lock().unwrap() = None;
    *state.raw_mouth_mask.lock().unwrap() = None;
    *state.raw_eyebrow_mask.lock().unwrap() = None;
    *state.eye_mask.lock().unwrap() = None;
    *state.mouth_mask.lock().unwrap() = None;
    *state.eyebrow_mask.lock().unwrap() = None;
    *state.combined_mask.lock().unwrap() = None;
    *state.part_pairs.lock().unwrap() = Vec::new();
    state.generated.lock().unwrap().clear();
    *state.canvas_width.lock().unwrap() = w;
    *state.canvas_height.lock().unwrap() = h;

    eprintln!(
        "[PachiPakuGen] Loaded {} parts, canvas {}x{}",
        part_count, w, h
    );

    Ok(LoadPartsResult {
        canvas_width: w,
        canvas_height: h,
        part_count,
    })
}
