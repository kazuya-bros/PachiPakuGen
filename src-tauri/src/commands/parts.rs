use crate::error::AppError;
use crate::inference::neck_extract;
use crate::inference::rife::rife_interpolate;
use crate::inference::session::{create_session, resolve_model_path};
use crate::processing::composite::{extract_part_with_blended_alpha, premultiply_onto_body};
use crate::processing::image_utils;
use crate::state::AppState;
use base64::{engine::general_purpose::STANDARD, Engine};
use image::{DynamicImage, GrayImage, RgbaImage};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Cursor;
use std::path::Path;
use tauri::{AppHandle, Emitter, Manager};

// ── Layer mapping constants ──────────────────────────────────────────

/// Fixed mappings: NOT shown in the mapping UI. Absolutely certain assignments.
const FIXED_MAPPINGS: &[(&str, &str)] = &[
    ("face", "body"),
    ("neck", "body"),
    ("nose", "body"),
    ("irides", "eye"),
    ("eyewhite", "eye"),
    ("eyelash", "eye"),
    ("eyebrow", "eye"),
    ("mouth", "mouth"),
];

/// Adjustable mappings: shown in UI with defaults, user can change.
const ADJUSTABLE_DEFAULTS: &[(&str, &str)] = &[
    ("front_hair", "hair"),
    ("back_hair", "hair_back"),
    ("headwear", "hair"),
    ("ears", "body"),
    ("topwear", "body"),
    ("bottomwear", "body"),
    ("legwear", "body"),
    ("footwear", "body"),
    ("handwear", "body"),
    ("earwear", "body"),
    ("eyewear", "body"),
    ("neckwear", "body"),
    ("objects", "body"),
    ("wings", "body"),
    ("tail", "body"),
];

/// Layers that may have -l/-r variants.
const LR_SPLIT_LAYERS: &[&str] = &[
    "eyebrow", "eyelash", "irides", "eyewhite", "ears", "handwear",
];

const DEPTH_VISIBILITY_TOLERANCE: u8 = 2;
const DEPTH_VISIBILITY_FEATHER_SIGMA: f32 = 1.0;
const ARM_L_OVERLAY_PREFIX: &str = "arm_l_overlay_";
const ARM_R_OVERLAY_PREFIX: &str = "arm_r_overlay_";

/// 腕本体と同じ変形へ追従しつつ、独立したz位置で描画する切り出しパーツ。
pub(crate) fn arm_overlay_parent(name: &str) -> Option<&'static str> {
    if name.starts_with(ARM_L_OVERLAY_PREFIX) {
        Some("arm_l")
    } else if name.starts_with(ARM_R_OVERLAY_PREFIX) {
        Some("arm_r")
    } else {
        None
    }
}

pub(crate) fn is_arm_overlay_part_name(name: &str) -> bool {
    arm_overlay_parent(name).is_some()
}

fn arm_overlay_part_name(parent: &str, patch_id: &str) -> String {
    let suffix: String = patch_id
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .collect();
    format!("{parent}_overlay_{}", if suffix.is_empty() { "patch" } else { &suffix })
}

// ── Types ────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct AdjustableLayer {
    pub name: String,
    pub thumbnail: String,
    pub default_target: String,
}

#[derive(Serialize)]
pub struct SlotLoadResult {
    pub detected_layers: Vec<String>,
    pub adjustable_layers: Vec<AdjustableLayer>,
    pub canvas_width: u32,
    pub canvas_height: u32,
    pub source_type: String,
}

#[derive(Serialize)]
pub struct CreateBaseResult {
    pub output_path: String,
    pub composite_preview: String,
    pub base_eye_slot: String,
    pub base_mouth_slot: String,
    pub file_count: u32,
}

#[derive(Clone, Serialize)]
pub struct ProgressPayload {
    pub current: u32,
    pub total: u32,
    pub pair_name: String,
}

#[derive(Serialize)]
pub struct CreateDiffResult {
    pub output_path: String,
    pub pair_name: String,
    pub frame_count: u32,
    pub preview: String,
    pub previews: Vec<String>,
}

#[derive(Serialize)]
pub struct LayerInfo {
    pub name: String,        // layer name (e.g. "face", "irides")
    pub thumbnail: String,   // base64 PNG thumbnail
    pub bounds: LayerBounds, // non-transparent bounds on the full canvas
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct LayerBounds {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LayerPatch {
    pub id: String,
    pub source_layer: String,
    pub mask_png: String,
    pub cut_source: bool,
}

#[derive(Serialize)]
pub struct CategoryPreview {
    pub target: String,
    pub label: String,
    pub preview: String, // merged preview base64 PNG
    pub layer_names: Vec<String>,
    pub layers: Vec<LayerInfo>, // individual layer thumbnails for toggle UI
}

#[derive(Serialize)]
pub struct MappingPreviewResult {
    pub categories: Vec<CategoryPreview>,
    pub composite_preview: String,
}

#[derive(Serialize)]
pub struct RenderCategoryResult {
    pub preview: String,
}

#[derive(Serialize)]
pub struct ExportCorrectedLayerResult {
    pub output_path: String,
}

#[derive(Serialize)]
pub struct ImportCorrectionLayerResult {
    pub layer_name: String,
}

#[derive(Serialize)]
pub struct OriginalImageResult {
    pub original_preview: String,
    pub mouth_preview: Option<String>,
}

#[derive(Serialize)]
pub struct MouthMaskPreviewResult {
    pub mouth_preview: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Sam3SelectResult {
    pub mask_png: String,
}

// ── Commands ─────────────────────────────────────────────────────────

/// Load a See-Through output (PSD file or folder of PNGs) into the current slot.
#[tauri::command]
pub async fn load_slot(app: AppHandle, path: String) -> Result<SlotLoadResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || load_slot_inner(app, path))
        .await
        .map_err(|e| AppError::General(format!("Task join error: {}", e)))?
}

/// Create the base body: merge layers → export body/hair/hair_back + store base eye/mouth.
#[tauri::command]
pub async fn create_base(
    app: AppHandle,
    mapping_json: String,
    original_image_path: String,
    base_eye_slot: String,
    base_mouth_slot: String,
    body_layer_order: Vec<String>, // user's custom body layer order (top=front)
    body_layer_patches: Vec<LayerPatch>,
    hair_layer_order: Vec<String>, // user's custom hair layer order (top=front)
    hair_back_layer_order: Vec<String>, // user's custom hair_back layer order (top=front)
    output_path: String,
    // 胸を切出（オプション）: 切出ツールで塗ったマスクPNG。塗った範囲を body から
    // 除去し chest として独立出力する（See-Throughにchestレイヤーが無いための手動抽出導線）
    chest_mask_png: Option<String>,
) -> Result<CreateBaseResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || {
        create_base_inner(
            app,
            mapping_json,
            original_image_path,
            base_eye_slot,
            base_mouth_slot,
            body_layer_order,
            body_layer_patches,
            hair_layer_order,
            hair_back_layer_order,
            output_path,
            chest_mask_png,
        )
    })
    .await
    .map_err(|e| AppError::General(format!("Task join error: {}", e)))?
}

/// Create a diff: load PSD → extract eye or mouth → RIFE interpolate with base → export folder.
#[tauri::command]
pub async fn create_diff(
    app: AppHandle,
    path: String,
    diff_type: String, // "eye" or "mouth"
    slot_name: String, // e.g. "eye_closed", "mouth_a", "mouth_i", etc.
    frame_count: u32,
    output_path: String,
    original_image_path: String, // 元画像パス（mouth SAM3マスク適用用）
) -> Result<CreateDiffResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || {
        create_diff_inner(
            app,
            path,
            diff_type,
            slot_name,
            frame_count,
            output_path,
            original_image_path,
        )
    })
    .await
    .map_err(|e| AppError::General(format!("Task join error: {}", e)))?
}

/// Get a composite preview using current base parts.
#[tauri::command]
pub async fn get_base_preview(app: AppHandle) -> Result<String, AppError> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        let parts = state.parts.lock().unwrap();
        let w = *state.canvas_width.lock().unwrap();
        let h = *state.canvas_height.lock().unwrap();
        let draw_order = state.base_layer_group_order.lock().unwrap().clone();
        Ok(generate_composite_preview(&parts, w, h, &draw_order))
    })
    .await
    .map_err(|e| AppError::General(format!("Task join error: {}", e)))?
}

/// Get per-category merged previews based on current mapping.
#[tauri::command]
pub async fn get_mapping_preview(
    app: AppHandle,
    mapping_json: String,
) -> Result<MappingPreviewResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || get_mapping_preview_inner(app, mapping_json))
        .await
        .map_err(|e| AppError::General(format!("Task join error: {}", e)))?
}

/// Render a category preview with only specified layers enabled.
#[tauri::command]
pub async fn render_category(
    app: AppHandle,
    mapping_json: String,
    target: String,              // "body", "eye", "mouth", "hair", "hair_back"
    enabled_layers: Vec<String>, // layer names to include
    layer_patches: Vec<LayerPatch>,
    layer_opacities: HashMap<String, f32>,
    overlap_highlight: bool,
) -> Result<RenderCategoryResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || {
        render_category_inner(
            app,
            mapping_json,
            target,
            enabled_layers,
            layer_patches,
            layer_opacities,
            overlap_highlight,
        )
    })
    .await
    .map_err(|e| AppError::General(format!("Task join error: {}", e)))?
}

/// Get all loaded PSD layers for See-Through correction mode.
#[tauri::command]
pub async fn get_all_layers_preview(app: AppHandle) -> Result<MappingPreviewResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || get_all_layers_preview_inner(app))
        .await
        .map_err(|e| AppError::General(format!("Task join error: {}", e)))?
}

/// Add an external PNG/JPEG/WebP as a correction layer.
#[tauri::command]
pub async fn import_correction_layer(
    app: AppHandle,
    path: String,
    layer_name: String,
) -> Result<ImportCorrectionLayerResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || {
        import_correction_layer_inner(app, path, layer_name)
    })
    .await
    .map_err(|e| AppError::General(format!("Task join error: {}", e)))?
}

/// Segment a region from a click point using SAM3 (切出ツールのクリック選択モード).
/// `image_data_url` is the currently-displayed composite preview (what the user
/// clicked on); `points` are pixel coordinates on that same image.
#[tauri::command]
pub async fn sam3_select_region(
    image_data_url: String,
    points: Vec<(f64, f64)>,
) -> Result<Sam3SelectResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || sam3_select_region_inner(&image_data_url, &points))
        .await
        .map_err(|e| AppError::General(format!("Task join error: {}", e)))?
}

/// Export arbitrary corrected layer composition to an exact PNG path.
#[tauri::command]
pub async fn export_corrected_layer(
    app: AppHandle,
    output_path: String,
    enabled_layers: Vec<String>,
    layer_patches: Vec<LayerPatch>,
    layer_opacities: HashMap<String, f32>,
) -> Result<ExportCorrectedLayerResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || {
        export_corrected_layer_inner(
            app,
            output_path,
            enabled_layers,
            layer_patches,
            layer_opacities,
        )
    })
    .await
    .map_err(|e| AppError::General(format!("Task join error: {}", e)))?
}

/// Load original image and extract mouth mask via SAM3. Caches the result.
#[tauri::command]
pub async fn load_original_image(
    app: AppHandle,
    path: String,
    mouth_mask_dilate_radius: Option<i32>,
    mouth_mask_blur_radius: Option<i32>,
) -> Result<OriginalImageResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        let original = image::open(&path)?;
        let w = *state.canvas_width.lock().unwrap();
        let h = *state.canvas_height.lock().unwrap();
        let original = if w > 0 && h > 0 && (original.width() != w || original.height() != h) {
            fit_image_to_canvas(&original, w, h)
        } else {
            original
        };
        // Cache original image for later use (mouth extraction applies mask to this)
        *state.cached_original.lock().unwrap() = Some(original.clone());
        state.cached_mouth_originals.lock().unwrap().insert(path.clone(), original.clone());

        let sam3_ckpt = neck_extract::find_sam3_checkpoint();
        let radius = mouth_mask_dilate_radius.unwrap_or(15).clamp(0, 64);
        let blur = mouth_mask_blur_radius.unwrap_or(0).clamp(0, 32);
        let mut mouth_preview = None;

        // Extract and cache mouth mask from original image
        // (original image has clear mouth features even when closed, unlike PSD composite)
        match neck_extract::extract_mouth_raw_mask(&original, sam3_ckpt.as_deref()) {
            Ok(raw_mask) => {
                eprintln!(
                    "[PachiPakuGen] Raw mouth mask cached from original image via SAM3 (dilate radius={}, blur radius={})",
                    radius, blur
                );
                let mask = neck_extract::adjust_mask(&raw_mask, original.width(), original.height(), radius, blur);
                let masked = neck_extract::apply_mask_to_image(&original, &mask, original.width(), original.height());
                mouth_preview = Some(mouth_preview_to_base64(&masked, &mask, original.width(), original.height()));
                state.cached_mouth_raw_masks.lock().unwrap().insert(path.clone(), raw_mask.clone());
                *state.cached_mouth_raw_mask.lock().unwrap() = Some(raw_mask);
                *state.cached_mouth_mask.lock().unwrap() = Some(mask);
            }
            Err(e) => {
                eprintln!("[PachiPakuGen] SAM3 mouth from original failed: {}", e);
            }
        }

        Ok(OriginalImageResult {
            original_preview: image_utils::image_to_base64_png(&original),
            mouth_preview,
        })
    })
    .await
    .map_err(|e| AppError::General(format!("Task join error: {}", e)))?
}

#[tauri::command]
pub async fn update_mouth_mask_preview(
    app: AppHandle,
    path: String,
    mouth_mask_dilate_radius: i32,
    mouth_mask_blur_radius: i32,
) -> Result<MouthMaskPreviewResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        let original = state
            .cached_mouth_originals
            .lock()
            .unwrap()
            .get(&path)
            .cloned()
            .ok_or_else(|| {
                AppError::General(
                    "元画像が読み込まれていません。先に口マスク確認を実行してください".into(),
                )
            })?;
        let raw_mask = state
            .cached_mouth_raw_masks
            .lock()
            .unwrap()
            .get(&path)
            .cloned()
            .ok_or_else(|| {
                AppError::General(
                    "SAM3口マスクがまだ作成されていません。先に口マスク確認を実行してください"
                        .into(),
                )
            })?;
        let mask = neck_extract::adjust_mask(
            &raw_mask,
            original.width(),
            original.height(),
            mouth_mask_dilate_radius,
            mouth_mask_blur_radius,
        );
        let masked = neck_extract::apply_mask_to_image(
            &original,
            &mask,
            original.width(),
            original.height(),
        );
        let preview = mouth_preview_to_base64(&masked, &mask, original.width(), original.height());
        *state.cached_original.lock().unwrap() = Some(original);
        *state.cached_mouth_raw_mask.lock().unwrap() = Some(raw_mask);
        *state.cached_mouth_mask.lock().unwrap() = Some(mask);
        Ok(MouthMaskPreviewResult {
            mouth_preview: preview,
        })
    })
    .await
    .map_err(|e| AppError::General(format!("Task join error: {}", e)))?
}

// ── Implementation ───────────────────────────────────────────────────

fn mouth_preview_to_base64(masked: &DynamicImage, mask: &[u8], width: u32, height: u32) -> String {
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut found = false;

    for y in 0..height {
        for x in 0..width {
            if mask[(y * width + x) as usize] > 8 {
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
                found = true;
            }
        }
    }

    if !found {
        return image_utils::image_to_base64_png(masked);
    }

    let pad_x = (width / 12).max(48);
    let pad_y = (height / 12).max(48);
    let x0 = min_x.saturating_sub(pad_x);
    let y0 = min_y.saturating_sub(pad_y);
    let x1 = (max_x + pad_x).min(width.saturating_sub(1));
    let y1 = (max_y + pad_y).min(height.saturating_sub(1));
    let crop_w = (x1 - x0 + 1).max(1);
    let crop_h = (y1 - y0 + 1).max(1);
    let crop = masked.crop_imm(x0, y0, crop_w, crop_h);
    image_utils::image_to_base64_png(&crop)
}

fn part_previews_to_base64(frames: &[DynamicImage]) -> Vec<String> {
    if frames.is_empty() {
        return Vec::new();
    }

    let width = frames[0].width();
    let height = frames[0].height();
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut found = false;

    for frame in frames {
        let rgba = frame.to_rgba8();
        for y in 0..height {
            for x in 0..width {
                if rgba.get_pixel(x, y)[3] > 8 {
                    min_x = min_x.min(x);
                    min_y = min_y.min(y);
                    max_x = max_x.max(x);
                    max_y = max_y.max(y);
                    found = true;
                }
            }
        }
    }

    if !found {
        return frames
            .iter()
            .map(image_utils::image_to_base64_png)
            .collect();
    }

    let part_w = max_x - min_x + 1;
    let part_h = max_y - min_y + 1;
    let pad_x = (part_w / 2).max(24);
    let pad_y = (part_h / 2).max(24);
    let x0 = min_x.saturating_sub(pad_x);
    let y0 = min_y.saturating_sub(pad_y);
    let x1 = (max_x + pad_x).min(width.saturating_sub(1));
    let y1 = (max_y + pad_y).min(height.saturating_sub(1));
    frames
        .iter()
        .map(|frame| {
            let crop = frame.crop_imm(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
            image_utils::image_to_base64_png(&crop)
        })
        .collect()
}

fn fit_image_to_canvas(image: &DynamicImage, target_w: u32, target_h: u32) -> DynamicImage {
    let src_w = image.width();
    let src_h = image.height();
    if src_w == 0 || src_h == 0 || target_w == 0 || target_h == 0 {
        return image.clone();
    }

    let scale = (target_w as f32 / src_w as f32).min(target_h as f32 / src_h as f32);
    let resized_w = ((src_w as f32 * scale).round() as u32).max(1);
    let resized_h = ((src_h as f32 * scale).round() as u32).max(1);
    let resized = image
        .resize_exact(resized_w, resized_h, image::imageops::FilterType::Lanczos3)
        .to_rgba8();
    let mut canvas = image::RgbaImage::new(target_w, target_h);
    let x0 = (target_w - resized_w) / 2;
    let y0 = (target_h - resized_h) / 2;
    image::imageops::overlay(&mut canvas, &resized, x0.into(), y0.into());
    eprintln!(
        "[PachiPakuGen] Original image fitted without aspect distortion: {}x{} -> {}x{} on {}x{}",
        src_w, src_h, resized_w, resized_h, target_w, target_h
    );
    DynamicImage::ImageRgba8(canvas)
}

pub(crate) fn cache_original_image_for_canvas(
    app: &AppHandle,
    path: &Path,
) -> Result<(), AppError> {
    let state = app.state::<AppState>();
    let original = image::open(path)?;
    let w = *state.canvas_width.lock().unwrap();
    let h = *state.canvas_height.lock().unwrap();
    let original = if w > 0 && h > 0 && (original.width() != w || original.height() != h) {
        fit_image_to_canvas(&original, w, h)
    } else {
        original
    };
    *state.cached_original.lock().unwrap() = Some(original);
    Ok(())
}

pub(crate) fn load_slot_inner(app: AppHandle, path: String) -> Result<SlotLoadResult, AppError> {
    let p = Path::new(&path);
    if !p.is_file() {
        return Err(AppError::General(format!(
            "ファイルが見つかりません: {}",
            path
        )));
    }
    let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
    if ext.to_lowercase() != "psd" {
        return Err(AppError::General(
            "PSD形式のファイルを選択してください".into(),
        ));
    }
    let (layers, layer_order) = load_layers_from_psd(&path)?;
    let source_type = "psd".to_string();

    if layers.is_empty() {
        return Err(AppError::General(
            "レイヤーが1つも見つかりませんでした".into(),
        ));
    }

    let state = app.state::<AppState>();

    // Determine canvas size from first meaningful layer
    let (w, h) = {
        let body_layer = layers.get("face").or_else(|| layers.values().next());
        match body_layer {
            Some(img) => (img.width(), img.height()),
            None => return Err(AppError::General("画像が見つかりません".into())),
        }
    };
    *state.canvas_width.lock().unwrap() = w;
    *state.canvas_height.lock().unwrap() = h;

    // Resize all layers to match canvas
    let mut resized_layers: HashMap<String, DynamicImage> = HashMap::new();
    for (name, img) in &layers {
        let img = if img.width() != w || img.height() != h {
            img.resize_exact(w, h, image::imageops::FilterType::Lanczos3)
        } else {
            img.clone()
        };
        // 低アルファノイズ除去（852話氏 Anime2.5DRig の自動リグ前処理と同方針）。
        // See-Through分解レイヤーはキャンバス全面に低アルファの斑点を残すことがあり、
        // 放置すると合成のかすれ・目/口マスクの全面化・位置検出の破綻を招く
        resized_layers.insert(name.clone(), clean_layer_alpha_noise(&img));
    }
    let depth_maps = load_depth_maps_for_psd(&path, &resized_layers, w, h);

    // Detect adjustable layers
    let mut adjustable_layers = Vec::new();
    let detected_layers: Vec<String> = resized_layers.keys().cloned().collect();
    let mut seen_adjustable: Vec<String> = Vec::new();

    for &(layer_name, default_target) in ADJUSTABLE_DEFAULTS {
        let img = resized_layers
            .get(layer_name)
            .or_else(|| resized_layers.get(&format!("{}-l", layer_name)))
            .or_else(|| resized_layers.get(&format!("{}-r", layer_name)))
            .or_else(|| resized_layers.get(&format!("{}_l", layer_name)))
            .or_else(|| resized_layers.get(&format!("{}_r", layer_name)));

        if let Some(img) = img {
            if !seen_adjustable.contains(&layer_name.to_string()) {
                let thumb = img.thumbnail(120, 120);
                adjustable_layers.push(AdjustableLayer {
                    name: layer_name.to_string(),
                    thumbnail: image_utils::image_to_base64_png(&thumb),
                    default_target: default_target.to_string(),
                });
                seen_adjustable.push(layer_name.to_string());
            }
        }
    }

    // Detect unknown layers
    for layer_name in &detected_layers {
        if layer_name.starts_with('_') {
            continue;
        } // skip internal layers
        let base = normalize_layer_name(layer_name);
        let is_known = FIXED_MAPPINGS.iter().any(|(n, _)| *n == base)
            || ADJUSTABLE_DEFAULTS.iter().any(|(n, _)| *n == base);
        if !is_known && !seen_adjustable.contains(&base.to_string()) {
            if let Some(img) = resized_layers.get(layer_name.as_str()) {
                let thumb = img.thumbnail(120, 120);
                adjustable_layers.push(AdjustableLayer {
                    name: base.to_string(),
                    thumbnail: image_utils::image_to_base64_png(&thumb),
                    default_target: "body".to_string(),
                });
                seen_adjustable.push(base.to_string());
            }
        }
    }

    // Store in state (single slot — overwrite previous)
    *state.slot_layers.lock().unwrap() = {
        let mut m = HashMap::new();
        m.insert("current".to_string(), resized_layers);
        m
    };
    *state.slot_layer_order.lock().unwrap() = layer_order;
    *state.slot_depth_maps.lock().unwrap() = depth_maps;

    eprintln!(
        "[PachiPakuGen] Loaded from {} ({} layers, canvas {}x{})",
        source_type,
        detected_layers.len(),
        w,
        h
    );

    Ok(SlotLoadResult {
        detected_layers,
        adjustable_layers,
        canvas_width: w,
        canvas_height: h,
        source_type,
    })
}

fn create_base_inner(
    app: AppHandle,
    mapping_json: String,
    original_image_path: String,
    base_eye_slot: String,
    base_mouth_slot: String,
    body_layer_order: Vec<String>,
    body_layer_patches: Vec<LayerPatch>,
    hair_layer_order: Vec<String>,
    hair_back_layer_order: Vec<String>,
    output_path: String,
    chest_mask_png: Option<String>,
) -> Result<CreateBaseResult, AppError> {
    let state = app.state::<AppState>();

    // Build full mapping
    let user_mapping: HashMap<String, String> = serde_json::from_str(&mapping_json)
        .map_err(|e| AppError::General(format!("Invalid mapping JSON: {}", e)))?;
    let full_mapping = build_full_mapping(&user_mapping);

    // Store mapping for future diff operations
    *state.layer_mapping.lock().unwrap() = full_mapping.clone();

    let slot_layers = state.slot_layers.lock().unwrap();
    let current = slot_layers
        .get("current")
        .ok_or_else(|| AppError::General("PSDが読み込まれていません".into()))?;
    let source_layer_order = state.slot_layer_order.lock().unwrap().clone();
    let depth_maps = state.slot_depth_maps.lock().unwrap().clone();
    let w = *state.canvas_width.lock().unwrap();
    let h = *state.canvas_height.lock().unwrap();

    // Keep open-eye assets pixel-exact by cutting the original image with the
    // See-Through eye mask. Fall back to reconstructed layers when no original
    // image is available.
    let reconstructed_eye =
        merge_eye_layers_for_base(current, &full_mapping, &source_layer_order, w, h);
    let exact_original = if !original_image_path.is_empty() {
        let original = image::open(&original_image_path)
            .map_err(|e| AppError::General(format!("Base eye image load failed: {}", e)))?;
        Some(if original.width() != w || original.height() != h {
            fit_image_to_canvas(&original, w, h)
        } else {
            original
        })
    } else {
        state.cached_original.lock().unwrap().clone()
    };
    let exact_eye = exact_original.as_ref().and_then(|original| {
        extract_original_target_pixels(
            original,
            reconstructed_eye.as_ref()?,
            current,
            &depth_maps,
            &full_mapping,
            "eye",
            Some("hair"),
            w,
            h,
        )
    });
    let eye = match (exact_eye, reconstructed_eye) {
        (Some(exact), Some(reconstructed)) => {
            let exact_color = saturated_alpha_pixel_count(&exact);
            let reconstructed_color = saturated_alpha_pixel_count(&reconstructed);
            if reconstructed_color > 32 && exact_color.saturating_mul(3) < reconstructed_color {
                eprintln!(
                    "[PachiPakuGen] Base eye exact extraction lost colored pixels; using reconstructed eye layers (exact={}, reconstructed={})",
                    exact_color, reconstructed_color
                );
                Some(reconstructed)
            } else {
                Some(exact)
            }
        }
        (Some(exact), None) => Some(exact),
        (None, reconstructed) => reconstructed,
    };

    // mouth: use cached SAM3 mask applied to THIS base's original image
    // (SAM3 mask was detected from open original, but pixels come from base's own original)
    let mouth = {
        let cached_mask = state.cached_mouth_mask.lock().unwrap().clone();
        if let (Some(mask), true) = (cached_mask, !original_image_path.is_empty()) {
            let base_orig = image::open(&original_image_path)
                .map_err(|e| AppError::General(format!("Base元画像の読み込み失敗: {}", e)))?;
            let base_orig = if base_orig.width() != w || base_orig.height() != h {
                fit_image_to_canvas(&base_orig, w, h)
            } else {
                base_orig
            };
            eprintln!("[PachiPakuGen] Base mouth: SAM3 mask applied to base's original image");
            Some(neck_extract::apply_mask_to_image(&base_orig, &mask, w, h))
        } else {
            eprintln!("[PachiPakuGen] No cached mouth mask/original, using PSD mouth layer");
            merge_layers_for_target(
                current,
                &depth_maps,
                &full_mapping,
                "mouth",
                &source_layer_order,
                w,
                h,
            )
        }
    };

    let mut parts: HashMap<String, DynamicImage> = HashMap::new();
    let mut file_count = 0u32;

    let is_base_export = !output_path.is_empty();
    let uses_unified_layer_order = !body_layer_order.is_empty()
        && hair_layer_order.is_empty()
        && hair_back_layer_order.is_empty()
        && body_layer_order.iter().any(|layer_name| {
            layer_order_entry_target(layer_name, &body_layer_patches, &full_mapping)
                .is_some_and(|target| target == "hair" || target == "hair_back")
        });
    let body_order = if uses_unified_layer_order {
        filter_layer_order_for_target(
            &body_layer_order,
            &body_layer_patches,
            &full_mapping,
            "body",
        )
    } else {
        body_layer_order.clone()
    };
    // arm_l / arm_r / sway_* へ分離されたレイヤーは body 合成から除外（二重描画防止）
    let body_order: Vec<String> = body_order
        .into_iter()
        .filter(|layer_name| {
            layer_order_entry_target(layer_name, &body_layer_patches, &full_mapping)
                .map(|target| !target.starts_with("arm_") && !target.starts_with("sway_"))
                .unwrap_or(true)
        })
        .collect();
    let hair_order = if uses_unified_layer_order {
        filter_layer_order_for_target(
            &body_layer_order,
            &body_layer_patches,
            &full_mapping,
            "hair",
        )
    } else {
        hair_layer_order.clone()
    };
    let hair_back_order = if uses_unified_layer_order {
        filter_layer_order_for_target(
            &body_layer_order,
            &body_layer_patches,
            &full_mapping,
            "hair_back",
        )
    } else {
        hair_back_layer_order.clone()
    };

    // グループ間z順（背面→前面）を保持。save_codex_base_parts が layer-order.json
    // として素材フォルダへ出力し、Motion Lab / SpriTalk が固定z順の代わりに使う
    let base_layer_group_order = if uses_unified_layer_order {
        derive_group_draw_order(&body_layer_order, &body_layer_patches, &full_mapping)
    } else {
        Vec::new()
    };
    *state.base_layer_group_order.lock().unwrap() = base_layer_group_order.clone();

    // 腕分離（オプション）: mapping で arm_l / arm_r に割り当てられたレイヤーを
    // 独立パーツとして合成する。未割り当てなら None = 従来どおり body に統合
    for arm_target in ["arm_l", "arm_r"] {
        parts.extend(build_linked_arm_parts(
            current,
            &depth_maps,
            &full_mapping,
            &body_layer_order,
            &body_layer_patches,
            &source_layer_order,
            arm_target,
            w,
            h,
        )?);
    }

    // 汎用揺れパーツ分離: mapping で "sway_" から始まるターゲットへ割り当てた
    // レイヤーを独立パーツとして合成する（例: ears-l → sway_ear_l → sway_ear_l.png）
    let mut sway_targets: Vec<String> = full_mapping
        .values()
        .filter(|target| target.starts_with("sway_"))
        .cloned()
        .collect();
    sway_targets.sort();
    sway_targets.dedup();
    for sway_target in sway_targets {
        let sway_order = filter_layer_order_for_target(
            &body_layer_order,
            &body_layer_patches,
            &full_mapping,
            &sway_target,
        );
        let sway_img = if !sway_order.is_empty() {
            let mut order_reversed = sway_order;
            order_reversed.reverse();
            let render_layers =
                collect_ordered_render_layers(current, &order_reversed, &[], &HashMap::new(), None);
            Some(compose_depth_gated_layers(render_layers, &depth_maps, w, h))
        } else {
            merge_layers_for_target(
                current,
                &depth_maps,
                &full_mapping,
                &sway_target,
                &source_layer_order,
                w,
                h,
            )
        };
        if let Some(img) = sway_img {
            parts.insert(sway_target.clone(), img);
        }
    }

    if is_base_export {
        // === 素体モード: body/hair/hair_back を出力 ===
        // hair: merge layers using user's custom order
        let mut hair = if !hair_order.is_empty() {
            let mut order_reversed = hair_order.clone();
            order_reversed.reverse();
            let render_layers =
                collect_ordered_render_layers(current, &order_reversed, &[], &HashMap::new(), None);
            Some(compose_depth_gated_layers(render_layers, &depth_maps, w, h))
        } else {
            merge_layers_for_target(
                current,
                &depth_maps,
                &full_mapping,
                "hair",
                &source_layer_order,
                w,
                h,
            )
        };
        // hair_back: merge layers using user's custom order
        let mut hair_back = if !hair_back_order.is_empty() {
            let mut order_reversed = hair_back_order.clone();
            order_reversed.reverse();
            let render_layers =
                collect_ordered_render_layers(current, &order_reversed, &[], &HashMap::new(), None);
            Some(compose_depth_gated_layers(render_layers, &depth_maps, w, h))
        } else {
            merge_layers_for_target(
                current,
                &depth_maps,
                &full_mapping,
                "hair_back",
                &source_layer_order,
                w,
                h,
            )
        };

        // body: merge layers using user's custom order
        let mut body_img = if !body_order.is_empty() {
            let effective_order = if uses_unified_layer_order {
                body_order.clone()
            } else {
                ensure_core_body_layers_for_custom_order(
                    &body_order,
                    current,
                    &full_mapping,
                    &source_layer_order,
                )
            };
            let active_patches = active_patches_for_order(&body_layer_patches, &effective_order);
            let mut order_reversed = effective_order;
            order_reversed.reverse();
            let patch_masks = prepare_patch_masks(&active_patches, w, h)?;
            let render_layers = collect_ordered_render_layers(
                current,
                &order_reversed,
                &active_patches,
                &patch_masks,
                None,
            );
            compose_depth_gated_layers(render_layers, &depth_maps, w, h)
        } else {
            merge_layers_for_target(
                current,
                &depth_maps,
                &full_mapping,
                "body",
                &source_layer_order,
                w,
                h,
            )
            .ok_or_else(|| AppError::General("bodyに対応するレイヤーが見つかりません".into()))?
        };

        if let Some(hair_img) = hair.as_mut() {
            let promoted_pixels = promote_body_foreground_over_hair(
                &body_img,
                hair_img,
                current,
                &depth_maps,
                &full_mapping,
                w,
                h,
            );
            if promoted_pixels > 0 {
                eprintln!(
                    "[PachiPakuGen] Promoted {} body-over-hair pixels into hair overlay",
                    promoted_pixels
                );
            }
        }
        if let Some(hair_back_img) = hair_back.as_mut() {
            let promoted_pixels = promote_hair_back_foreground_over_body(
                &body_img,
                hair_back_img,
                &mut hair,
                current,
                &depth_maps,
                &full_mapping,
                w,
                h,
            );
            if promoted_pixels > 0 {
                eprintln!(
                    "[PachiPakuGen] Promoted {} hair_back-over-body pixels into front hair",
                    promoted_pixels
                );
            }
        }

        // 胸を切出（オプション・852話式: See-Throughにchestレイヤーが無いため手動抽出）。
        // 塗った範囲を body から除去し、chest として独立出力する
        body_img = cut_chest_from_body(body_img, chest_mask_png.as_deref(), &mut parts, w, h)?;

        parts.insert("body".to_string(), body_img);
        if let Some(img) = hair {
            parts.insert("hair".to_string(), img);
        }
        if let Some(img) = hair_back {
            parts.insert("hair_back".to_string(), img);
        }

        // Export static layers
        let out_dir = Path::new(&output_path);
        fs::create_dir_all(out_dir)?;
        for key in &["body", "hair", "hair_back", "arm_l", "arm_r", "chest"] {
            if let Some(img) = parts.get(*key) {
                img.save(out_dir.join(format!("{}.png", key)))?;
                file_count += 1;
            }
        }

        eprintln!(
            "[PachiPakuGen] Base body created ({}x{}), {}files",
            w, h, file_count
        );
    } else {
        // === フレーム補間モード: bodyはPSD合成でRIFE用に保持するだけ ===
        let body = if !body_order.is_empty() {
            let effective_order = if uses_unified_layer_order {
                body_order.clone()
            } else {
                ensure_core_body_layers_for_custom_order(
                    &body_order,
                    current,
                    &full_mapping,
                    &source_layer_order,
                )
            };
            let active_patches = active_patches_for_order(&body_layer_patches, &effective_order);
            let mut order_reversed = effective_order;
            order_reversed.reverse();
            let patch_masks = prepare_patch_masks(&active_patches, w, h)?;
            let render_layers = collect_ordered_render_layers(
                current,
                &order_reversed,
                &active_patches,
                &patch_masks,
                None,
            );
            compose_depth_gated_layers(render_layers, &depth_maps, w, h)
        } else {
            // Exclude neck from body when there is no user-edited layer order.
            merge_layers_for_target_excluding(
                current,
                &depth_maps,
                &full_mapping,
                "body",
                &["neck"],
                &source_layer_order,
                w,
                h,
            )
            .unwrap_or_else(|| DynamicImage::new_rgba8(w, h))
        };

        let mut hair = if !hair_order.is_empty() {
            let mut order_reversed = hair_order.clone();
            order_reversed.reverse();
            let render_layers =
                collect_ordered_render_layers(current, &order_reversed, &[], &HashMap::new(), None);
            Some(compose_depth_gated_layers(render_layers, &depth_maps, w, h))
        } else {
            merge_layers_for_target(
                current,
                &depth_maps,
                &full_mapping,
                "hair",
                &source_layer_order,
                w,
                h,
            )
        };

        let mut hair_back = if !hair_back_order.is_empty() {
            let mut order_reversed = hair_back_order.clone();
            order_reversed.reverse();
            let render_layers =
                collect_ordered_render_layers(current, &order_reversed, &[], &HashMap::new(), None);
            Some(compose_depth_gated_layers(render_layers, &depth_maps, w, h))
        } else {
            merge_layers_for_target(
                current,
                &depth_maps,
                &full_mapping,
                "hair_back",
                &source_layer_order,
                w,
                h,
            )
        };

        if let Some(hair_img) = hair.as_mut() {
            let promoted_pixels = promote_body_foreground_over_hair(
                &body,
                hair_img,
                current,
                &depth_maps,
                &full_mapping,
                w,
                h,
            );
            if promoted_pixels > 0 {
                eprintln!(
                    "[PachiPakuGen] Promoted {} body-over-hair pixels into hair overlay",
                    promoted_pixels
                );
            }
        }
        if let Some(hair_back_img) = hair_back.as_mut() {
            let promoted_pixels = promote_hair_back_foreground_over_body(
                &body,
                hair_back_img,
                &mut hair,
                current,
                &depth_maps,
                &full_mapping,
                w,
                h,
            );
            if promoted_pixels > 0 {
                eprintln!(
                    "[PachiPakuGen] Promoted {} hair_back-over-body pixels into front hair",
                    promoted_pixels
                );
            }
        }

        // 胸を切出（workspace flow=素体をsave_codex_base_partsで保存する経路もここ）
        let body = cut_chest_from_body(body, chest_mask_png.as_deref(), &mut parts, w, h)?;
        parts.insert("body".to_string(), body);
        if body_order.is_empty() {
            if let Some(neck) =
                merge_layers_for_names(current, &depth_maps, &["neck"], &source_layer_order, w, h)
            {
                parts.insert("neck".to_string(), neck);
            }
        }
        if let Some(img) = hair {
            parts.insert("hair".to_string(), img);
        }
        if let Some(img) = hair_back {
            parts.insert("hair_back".to_string(), img);
        }

        eprintln!(
            "[PachiPakuGen] Interp base loaded ({}x{}), body for premultiply (no neck)",
            w, h
        );
    }

    // eye/mouth
    if let Some(img) = eye {
        parts.insert(base_eye_slot.clone(), img);
    }
    if let Some(img) = mouth {
        parts.insert(base_mouth_slot.clone(), img);
    }

    let composite_preview = generate_composite_preview(&parts, w, h, &base_layer_group_order);

    // Store parts for future diff operations (keep base eye & mouth)
    drop(slot_layers);
    *state.parts.lock().unwrap() = parts;
    state.slot_layers.lock().unwrap().clear();
    state.slot_layer_order.lock().unwrap().clear();
    state.slot_depth_maps.lock().unwrap().clear();

    eprintln!(
        "[PachiPakuGen] Base created: {}files, eye={}, mouth={}",
        file_count, base_eye_slot, base_mouth_slot
    );

    Ok(CreateBaseResult {
        output_path: output_path.clone(),
        composite_preview,
        base_eye_slot,
        base_mouth_slot,
        file_count,
    })
}

fn create_diff_inner(
    app: AppHandle,
    path: String,
    diff_type: String,
    slot_name: String,
    frame_count: u32,
    output_path: String,
    original_image_path: String,
) -> Result<CreateDiffResult, AppError> {
    if frame_count < 2 || frame_count > 30 {
        return Err(AppError::General(
            "フレーム数は2〜30の範囲で指定してください".into(),
        ));
    }

    let state = app.state::<AppState>();

    // Load the diff PSD
    let p = Path::new(&path);
    if !p.is_file() {
        return Err(AppError::General(format!(
            "ファイルが見つかりません: {}",
            path
        )));
    }
    let (layers, source_layer_order) = load_layers_from_psd(&path)?;

    let mapping = state.layer_mapping.lock().unwrap().clone();
    if mapping.is_empty() {
        return Err(AppError::General("先に素体を作成してください".into()));
    }

    let w = *state.canvas_width.lock().unwrap();
    let h = *state.canvas_height.lock().unwrap();

    // Resize layers
    let mut resized: HashMap<String, DynamicImage> = HashMap::new();
    for (name, img) in &layers {
        let img = if img.width() != w || img.height() != h {
            img.resize_exact(w, h, image::imageops::FilterType::Lanczos3)
        } else {
            img.clone()
        };
        resized.insert(name.clone(), img);
    }

    let depth_maps = load_depth_maps_for_psd(&path, &resized, w, h);

    // Extract the target (eye or mouth) from the diff PSD
    let target = match diff_type.as_str() {
        "eye" => "eye",
        "mouth" => "mouth",
        _ => return Err(AppError::General(format!("不正なdiff_type: {}", diff_type))),
    };

    // Extract target from diff PSD. See-Through semantic layers are the primary
    // source for both eyes and mouths; masks are only a fallback when layer
    // classification fails.
    let diff_merged = if target == "mouth" {
        if let Some(merged) = merge_layers_for_target(
            &resized,
            &depth_maps,
            &mapping,
            target,
            &source_layer_order,
            w,
            h,
        ) {
            eprintln!("[PachiPakuGen] Diff mouth: using See-Through mouth layers");
            merged
        } else {
            let cached_mask = state.cached_mouth_mask.lock().unwrap().clone();
            if let (Some(mask), true) = (cached_mask, !original_image_path.is_empty()) {
                let diff_orig = image::open(&original_image_path)
                    .map_err(|e| AppError::General(format!("Diff元画像の読み込み失敗: {}", e)))?;
                let diff_orig = if diff_orig.width() != w || diff_orig.height() != h {
                    fit_image_to_canvas(&diff_orig, w, h)
                } else {
                    diff_orig
                };
                eprintln!("[PachiPakuGen] Diff mouth: See-Through mouth layer missing; fallback to SAM3 gate");
                neck_extract::apply_mask_to_image(&diff_orig, &mask, w, h)
            } else {
                return Err(AppError::General(
                    "mouthレイヤーが見つかりません。See-Through分類を補正するか、口マスクを作成してください"
                        .into(),
                ));
            }
        }
    } else {
        // Eye: PSD layers directly
        merge_layers_for_target(
            &resized,
            &depth_maps,
            &mapping,
            target,
            &source_layer_order,
            w,
            h,
        )
        .ok_or_else(|| AppError::General(format!("{}に対応するレイヤーが見つかりません", target)))?
    };

    let diff_merged = if target == "eye" && !original_image_path.is_empty() {
        let original = image::open(&original_image_path)
            .map_err(|e| AppError::General(format!("Diff eye image load failed: {}", e)))?;
        let original = if original.width() != w || original.height() != h {
            fit_image_to_canvas(&original, w, h)
        } else {
            original
        };
        extract_original_target_pixels(
            &original,
            &diff_merged,
            &resized,
            &depth_maps,
            &mapping,
            "eye",
            Some("hair"),
            w,
            h,
        )
        .unwrap_or(diff_merged)
    } else {
        diff_merged
    };

    // Get base frame from stored parts
    let parts = state.parts.lock().unwrap();
    let base_key = if diff_type == "eye" {
        // Find which eye slot is the base (eye_open or eye_closed)
        parts
            .keys()
            .find(|k| k.starts_with("eye_"))
            .cloned()
            .ok_or_else(|| AppError::General("素体のeyeが見つかりません".into()))?
    } else {
        parts
            .keys()
            .find(|k| k.starts_with("mouth_") || *k == "mouth_closed")
            .cloned()
            .ok_or_else(|| AppError::General("素体のmouthが見つかりません".into()))?
    };

    let base_frame = parts
        .get(&base_key)
        .ok_or_else(|| {
            AppError::General(format!("ベースフレーム '{}' が見つかりません", base_key))
        })?
        .clone();

    let body = parts
        .get("body")
        .ok_or_else(|| AppError::General("素体のbodyが見つかりません".into()))?
        .clone();
    let body_rgb = body.to_rgb8();
    drop(parts);

    // Initialize RIFE session if needed
    {
        let mut session = state.rife_session.lock().unwrap();
        if session.is_none() {
            let model_path = resolve_model_path(&app, "rife.onnx")?;
            *session = Some(create_session(&model_path)?);
        }
    }

    // RIFE interpolation: base_frame ↔ diff_merged
    let img_a_rgba = base_frame.to_rgba8();
    let img_b_rgba = diff_merged.to_rgba8();

    // Premultiply onto body (body already excludes neck in interp mode)
    let rife_a = premultiply_onto_body(&body_rgb, &img_a_rgba, w, h);
    let rife_b = premultiply_onto_body(&body_rgb, &img_b_rgba, w, h);

    let ratios: Vec<f32> = (0..frame_count)
        .map(|i| i as f32 / (frame_count - 1) as f32)
        .collect();

    let mut session_guard = state.rife_session.lock().unwrap();
    let session = session_guard.as_mut().unwrap();

    let mut frames = Vec::new();
    let pair_name = slot_name.clone();

    for (step, &ratio) in ratios.iter().enumerate() {
        let _ = app.emit(
            "generation-progress",
            ProgressPayload {
                current: (step + 1) as u32,
                total: frame_count,
                pair_name: pair_name.clone(),
            },
        );

        let part_frame = if step == 0 {
            DynamicImage::ImageRgba8(img_a_rgba.clone())
        } else if step + 1 == frame_count as usize {
            DynamicImage::ImageRgba8(img_b_rgba.clone())
        } else {
            let interpolated = rife_interpolate(session, &rife_a, &rife_b, ratio)?;
            extract_part_with_blended_alpha(&interpolated, &img_a_rgba, &img_b_rgba, ratio, w, h)
        };
        frames.push(part_frame);
    }
    drop(session_guard);

    // Export frames in SpriTalk order:
    // - eye: open -> closed
    // - mouth: closed -> open
    if diff_type == "eye" {
        frames.reverse();
    }
    let out_dir = Path::new(&output_path).join(&pair_name);
    fs::create_dir_all(&out_dir)?;

    for (i, frame) in frames.iter().enumerate() {
        let filename = format!("{:03}.png", i + 1);
        frame.save(out_dir.join(&filename))?;
    }

    let previews = part_previews_to_base64(&frames);
    let preview = previews.last().cloned().unwrap_or_default();

    eprintln!(
        "[PachiPakuGen] Diff created: {} ({} frames) → {}",
        pair_name,
        frame_count,
        out_dir.display()
    );

    Ok(CreateDiffResult {
        output_path: out_dir.to_string_lossy().into_owned(),
        pair_name,
        frame_count,
        preview,
        previews,
    })
}

// ── Helpers ──────────────────────────────────────────────────────────

pub(crate) fn get_mapping_preview_inner(
    app: AppHandle,
    mapping_json: String,
) -> Result<MappingPreviewResult, AppError> {
    let state = app.state::<AppState>();

    let user_mapping: HashMap<String, String> = serde_json::from_str(&mapping_json)
        .map_err(|e| AppError::General(format!("Invalid mapping JSON: {}", e)))?;
    let full_mapping = build_full_mapping(&user_mapping);

    let slot_layers = state.slot_layers.lock().unwrap();
    let current = slot_layers
        .get("current")
        .ok_or_else(|| AppError::General("PSD/フォルダが読み込まれていません".into()))?;
    let source_layer_order = state.slot_layer_order.lock().unwrap().clone();
    let depth_maps = state.slot_depth_maps.lock().unwrap().clone();
    let w = *state.canvas_width.lock().unwrap();
    let h = *state.canvas_height.lock().unwrap();

    let target_labels: &[(&str, &str)] = &[
        ("body", "Body (素体)"),
        ("eye", "Eye (目)"),
        ("mouth", "Mouth (口)"),
        ("hair", "Hair 前髪"),
        ("hair_back", "Hair 後髪"),
        ("arm_l", "Arm 左腕"),
        ("arm_r", "Arm 右腕"),
        ("skip", "スキップ"),
    ];

    let mut categories = Vec::new();

    for &(target, label) in target_labels {
        // Keep the PSD's top/front-first order so layer grouping does not
        // destroy See-Through's inferred depth relationships.
        let layer_names =
            ordered_layer_names_for_target(current, &full_mapping, target, &source_layer_order);

        if layer_names.is_empty() && target != "skip" {
            continue;
        }

        // Generate merged preview
        let preview = if target == "skip" {
            // For skip, just show a placeholder
            String::new()
        } else {
            let merged = merge_layers_for_target(
                current,
                &depth_maps,
                &full_mapping,
                target,
                &source_layer_order,
                w,
                h,
            );
            let merged = if target == "eye" {
                let cached_original = state.cached_original.lock().unwrap().clone();
                cached_original
                    .as_ref()
                    .and_then(|original| {
                        extract_original_target_pixels(
                            original,
                            merged.as_ref()?,
                            current,
                            &depth_maps,
                            &full_mapping,
                            "eye",
                            Some("hair"),
                            w,
                            h,
                        )
                    })
                    .or(merged)
            } else {
                merged
            };
            match merged {
                Some(img) => image_utils::image_to_base64_png(&img),
                None => String::new(),
            }
        };

        // Generate individual layer thumbnails
        let mut layers_info = Vec::new();
        for layer_name in &layer_names {
            if let Some(img) = current.get(layer_name.as_str()) {
                let thumb = img.thumbnail(120, 120);
                layers_info.push(LayerInfo {
                    name: layer_name.clone(),
                    thumbnail: image_utils::image_to_base64_png(&thumb),
                    bounds: alpha_bounds(&img.to_rgba8()).unwrap_or_default(),
                });
            }
        }

        if !preview.is_empty() || !layer_names.is_empty() {
            categories.push(CategoryPreview {
                target: target.to_string(),
                label: label.to_string(),
                preview,
                layer_names,
                layers: layers_info,
            });
        }
    }

    // Full composite preview
    let mut composite_parts: HashMap<String, DynamicImage> = HashMap::new();
    for target in &["body", "eye", "mouth", "hair", "hair_back", "arm_l", "arm_r"] {
        if let Some(mut img) = merge_layers_for_target(
            current,
            &depth_maps,
            &full_mapping,
            target,
            &source_layer_order,
            w,
            h,
        ) {
            if *target == "eye" {
                let cached_original = state.cached_original.lock().unwrap().clone();
                if let Some(exact_eye) = cached_original.as_ref().and_then(|original| {
                    extract_original_target_pixels(
                        original,
                        &img,
                        current,
                        &depth_maps,
                        &full_mapping,
                        "eye",
                        Some("hair"),
                        w,
                        h,
                    )
                }) {
                    img = exact_eye;
                }
            }
            // Map to expected keys for composite
            let key = match *target {
                "eye" => "eye_open",
                "mouth" => "mouth_closed",
                _ => target,
            };
            composite_parts.insert(key.to_string(), img);
        }
    }
    let body = composite_parts.get("body").cloned();
    if let (Some(body), Some(hair)) = (body.as_ref(), composite_parts.get_mut("hair")) {
        promote_body_foreground_over_hair(body, hair, current, &depth_maps, &full_mapping, w, h);
    }
    let composite_preview = generate_composite_preview(&composite_parts, w, h, &[]);

    Ok(MappingPreviewResult {
        categories,
        composite_preview,
    })
}

fn render_category_inner(
    app: AppHandle,
    _mapping_json: String,
    _target: String,
    enabled_layers: Vec<String>,
    layer_patches: Vec<LayerPatch>,
    layer_opacities: HashMap<String, f32>,
    overlap_highlight: bool,
) -> Result<RenderCategoryResult, AppError> {
    let state = app.state::<AppState>();

    let slot_layers = state.slot_layers.lock().unwrap();
    let current = slot_layers
        .get("current")
        .ok_or_else(|| AppError::General("PSDが読み込まれていません".into()))?;
    let w = *state.canvas_width.lock().unwrap();
    let h = *state.canvas_height.lock().unwrap();
    let depth_maps = state.slot_depth_maps.lock().unwrap().clone();

    // Composite in the exact order of enabled_layers (user-controlled order)
    let active_patches = active_patches_for_order(&layer_patches, &enabled_layers);
    let patch_masks = prepare_patch_masks(&active_patches, w, h)?;
    let render_layers = collect_ordered_render_layers(
        current,
        &enabled_layers,
        &active_patches,
        &patch_masks,
        Some(&layer_opacities),
    );
    let mut result_img = compose_depth_gated_layers(render_layers, &depth_maps, w, h).to_rgba8();
    if overlap_highlight {
        apply_overlap_highlight(
            &mut result_img,
            current,
            &enabled_layers,
            &active_patches,
            &patch_masks,
            &layer_opacities,
            w,
            h,
        )?;
    }

    let preview = image_utils::image_to_base64_png(&DynamicImage::ImageRgba8(result_img));
    Ok(RenderCategoryResult { preview })
}

fn get_all_layers_preview_inner(app: AppHandle) -> Result<MappingPreviewResult, AppError> {
    let state = app.state::<AppState>();

    let slot_layers = state.slot_layers.lock().unwrap();
    let current = slot_layers
        .get("current")
        .ok_or_else(|| AppError::General("PSDが読み込まれていません".into()))?;
    let w = *state.canvas_width.lock().unwrap();
    let h = *state.canvas_height.lock().unwrap();

    let mut compositing_order = state.slot_layer_order.lock().unwrap().clone();
    append_missing_layer_names(&mut compositing_order, current);
    let layer_names: Vec<String> = compositing_order.iter().rev().cloned().collect();

    let mut composite = image::RgbaImage::new(w, h);
    let mut layers = Vec::new();
    for name in &layer_names {
        if let Some(img) = current.get(name.as_str()) {
            let thumb = img.thumbnail(120, 120);
            layers.push(LayerInfo {
                name: name.clone(),
                thumbnail: image_utils::image_to_base64_png(&thumb),
                bounds: alpha_bounds(&img.to_rgba8()).unwrap_or_default(),
            });
        }
    }
    for name in &compositing_order {
        if let Some(img) = current.get(name.as_str()) {
            alpha_composite_onto(&mut composite, &img.to_rgba8(), w, h);
        }
    }

    let composite_preview =
        image_utils::image_to_base64_png(&DynamicImage::ImageRgba8(composite.clone()));
    Ok(MappingPreviewResult {
        categories: vec![CategoryPreview {
            target: "free".to_string(),
            label: "See-Through補正".to_string(),
            preview: composite_preview.clone(),
            layer_names,
            layers,
        }],
        composite_preview,
    })
}

fn import_correction_layer_inner(
    app: AppHandle,
    path: String,
    layer_name: String,
) -> Result<ImportCorrectionLayerResult, AppError> {
    let normalized_name = layer_name.trim().to_lowercase().replace(' ', "_");
    if normalized_name.is_empty() {
        return Err(AppError::General("追加レイヤー名が空です".into()));
    }

    let state = app.state::<AppState>();
    let w = *state.canvas_width.lock().unwrap();
    let h = *state.canvas_height.lock().unwrap();
    if w == 0 || h == 0 {
        return Err(AppError::General("先にPSDを読み込んでください".into()));
    }

    let img = image::open(&path)
        .map_err(|e| AppError::General(format!("追加レイヤー画像の読み込み失敗: {}", e)))?;
    let img = if img.width() != w || img.height() != h {
        img.resize_exact(w, h, image::imageops::FilterType::Lanczos3)
    } else {
        img
    };

    let mut slot_layers = state.slot_layers.lock().unwrap();
    let current = slot_layers
        .get_mut("current")
        .ok_or_else(|| AppError::General("PSDが読み込まれていません".into()))?;

    let mut final_name = normalized_name.clone();
    if current.contains_key(final_name.as_str()) {
        let mut idx = 1;
        loop {
            let candidate = format!("{}_{}", normalized_name, idx);
            if !current.contains_key(candidate.as_str()) {
                final_name = candidate;
                break;
            }
            idx += 1;
        }
    }
    current.insert(final_name.clone(), img);
    state
        .slot_layer_order
        .lock()
        .unwrap()
        .push(final_name.clone());
    eprintln!(
        "[PachiPakuGen] Imported correction layer '{}' from {}",
        final_name, path
    );

    Ok(ImportCorrectionLayerResult {
        layer_name: final_name,
    })
}

fn export_corrected_layer_inner(
    app: AppHandle,
    output_path: String,
    enabled_layers: Vec<String>,
    layer_patches: Vec<LayerPatch>,
    layer_opacities: HashMap<String, f32>,
) -> Result<ExportCorrectedLayerResult, AppError> {
    if enabled_layers.is_empty() {
        return Err(AppError::General("出力するレイヤーがありません".into()));
    }

    let state = app.state::<AppState>();
    let slot_layers = state.slot_layers.lock().unwrap();
    let current = slot_layers
        .get("current")
        .ok_or_else(|| AppError::General("PSDが読み込まれていません".into()))?;
    let w = *state.canvas_width.lock().unwrap();
    let h = *state.canvas_height.lock().unwrap();
    let depth_maps = state.slot_depth_maps.lock().unwrap().clone();

    let active_patches = active_patches_for_order(&layer_patches, &enabled_layers);
    let patch_masks = prepare_patch_masks(&active_patches, w, h)?;
    let render_layers = collect_ordered_render_layers(
        current,
        &enabled_layers,
        &active_patches,
        &patch_masks,
        Some(&layer_opacities),
    );
    let result_img = compose_depth_gated_layers(render_layers, &depth_maps, w, h).to_rgba8();

    let out_path = Path::new(&output_path);
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)?;
    }
    DynamicImage::ImageRgba8(result_img).save(out_path)?;

    Ok(ExportCorrectedLayerResult { output_path })
}

fn normalize_layer_name(name: &str) -> &str {
    for &base in LR_SPLIT_LAYERS {
        if name == format!("{}-l", base)
            || name == format!("{}-r", base)
            || name == format!("{}_l", base)
            || name == format!("{}_r", base)
        {
            return base;
        }
    }
    name
}

fn prepare_patch_masks(
    patches: &[LayerPatch],
    width: u32,
    height: u32,
) -> Result<HashMap<String, image::GrayImage>, AppError> {
    let mut masks = HashMap::new();
    for patch in patches {
        let mask = decode_mask_png(&patch.mask_png, width, height)?;
        masks.insert(patch.id.clone(), mask);
    }
    Ok(masks)
}

fn active_patches_for_order(patches: &[LayerPatch], enabled_layers: &[String]) -> Vec<LayerPatch> {
    patches
        .iter()
        .filter(|patch| enabled_layers.iter().any(|layer| layer == &patch.id))
        .cloned()
        .collect()
}

fn collect_ordered_render_layers(
    current: &HashMap<String, DynamicImage>,
    ordered_layer_names: &[String],
    patches: &[LayerPatch],
    patch_masks: &HashMap<String, GrayImage>,
    layer_opacities: Option<&HashMap<String, f32>>,
) -> Vec<(String, RgbaImage)> {
    let mut rendered = Vec::new();
    for layer_name in ordered_layer_names {
        let opacity = layer_opacities
            .and_then(|opacities| opacities.get(layer_name).copied())
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);

        if let Some(patch) = patches.iter().find(|patch| patch.id == *layer_name) {
            if let (Some(src), Some(mask)) = (
                current.get(patch.source_layer.as_str()),
                patch_masks.get(patch.id.as_str()),
            ) {
                let mut rgba = apply_mask_to_rgba(&src.to_rgba8(), mask, false);
                apply_opacity(&mut rgba, opacity);
                rendered.push((patch.source_layer.clone(), rgba));
            }
            continue;
        }

        let candidates = [
            layer_name.clone(),
            format!("{}-l", layer_name),
            format!("{}-r", layer_name),
            format!("{}_l", layer_name),
            format!("{}_r", layer_name),
        ];
        for candidate in &candidates {
            if let Some(img) = current.get(candidate.as_str()) {
                let mut rgba = img.to_rgba8();
                for patch in patches
                    .iter()
                    .filter(|patch| patch.cut_source && patch.source_layer == *candidate)
                {
                    if let Some(mask) = patch_masks.get(patch.id.as_str()) {
                        subtract_mask_from_rgba(&mut rgba, mask);
                    }
                }
                apply_opacity(&mut rgba, opacity);
                rendered.push((candidate.clone(), rgba));
            }
        }
    }
    rendered
}

fn apply_overlap_highlight(
    result: &mut image::RgbaImage,
    current: &HashMap<String, DynamicImage>,
    enabled_layers: &[String],
    patches: &[LayerPatch],
    patch_masks: &HashMap<String, image::GrayImage>,
    layer_opacities: &HashMap<String, f32>,
    width: u32,
    height: u32,
) -> Result<(), AppError> {
    let mut counts = vec![0u8; (width * height) as usize];
    for layer_name in enabled_layers {
        let opacity = layer_opacities
            .get(layer_name)
            .copied()
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);
        if opacity <= 0.0 {
            continue;
        }
        let images = body_order_item_alpha_images(current, layer_name, patches, patch_masks);
        for img in images {
            for y in 0..height {
                for x in 0..width {
                    if img.get_pixel(x, y)[3] > 8 {
                        let idx = (y * width + x) as usize;
                        counts[idx] = counts[idx].saturating_add(1);
                    }
                }
            }
        }
    }

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            if counts[idx] >= 2 {
                let dst = result.get_pixel_mut(x, y);
                let alpha = 150u16;
                let inv = 255u16 - alpha;
                let highlight = [255u16, 64u16, 180u16];
                dst[0] = ((highlight[0] * alpha + dst[0] as u16 * inv) / 255) as u8;
                dst[1] = ((highlight[1] * alpha + dst[1] as u16 * inv) / 255) as u8;
                dst[2] = ((highlight[2] * alpha + dst[2] as u16 * inv) / 255) as u8;
                dst[3] = dst[3].max(210);
            }
        }
    }
    Ok(())
}

fn body_order_item_alpha_images(
    current: &HashMap<String, DynamicImage>,
    layer_name: &str,
    patches: &[LayerPatch],
    patch_masks: &HashMap<String, image::GrayImage>,
) -> Vec<image::RgbaImage> {
    if let Some(patch) = patches.iter().find(|p| p.id == layer_name) {
        if let (Some(src), Some(mask)) = (
            current.get(patch.source_layer.as_str()),
            patch_masks.get(patch.id.as_str()),
        ) {
            return vec![apply_mask_to_rgba(&src.to_rgba8(), mask, false)];
        }
        return Vec::new();
    }

    let candidates = [
        layer_name.to_string(),
        format!("{}-l", layer_name),
        format!("{}-r", layer_name),
        format!("{}_l", layer_name),
        format!("{}_r", layer_name),
    ];
    let mut images = Vec::new();
    for candidate in &candidates {
        if let Some(img) = current.get(candidate.as_str()) {
            let mut rgba = img.to_rgba8();
            for patch in patches
                .iter()
                .filter(|p| p.cut_source && p.source_layer == *candidate)
            {
                if let Some(mask) = patch_masks.get(patch.id.as_str()) {
                    subtract_mask_from_rgba(&mut rgba, mask);
                }
            }
            images.push(rgba);
        }
    }
    images
}

/// 低アルファノイズ除去。
/// 1) alpha < 16 を透明化（実測: ノイズはalpha中央値1・p99=6、実体は128以上に集中）
/// 2) alpha < 128 かつ 3x3近傍に実体級(alpha>=64)の画素が2つ未満の孤立点を除去
///    （アンチエイリアスされた実体エッジは実体画素に隣接しているため保護される）
fn clean_layer_alpha_noise(image: &DynamicImage) -> DynamicImage {
    const ALPHA_FLOOR: u8 = 16;
    const SOLID_LEVEL: u8 = 64;
    const EDGE_LEVEL: u8 = 128;
    let mut rgba = image.to_rgba8();
    let (w, h) = rgba.dimensions();
    for pixel in rgba.pixels_mut() {
        if pixel[3] < ALPHA_FLOOR {
            pixel[3] = 0;
        }
    }
    let alpha: Vec<u8> = rgba.pixels().map(|pixel| pixel[3]).collect();
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let index = (y * w as i64 + x) as usize;
            if alpha[index] == 0 || alpha[index] >= EDGE_LEVEL {
                continue;
            }
            let mut solid_neighbors = 0;
            for dy in -1..=1i64 {
                for dx in -1..=1i64 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = x + dx;
                    let ny = y + dy;
                    if nx >= 0
                        && ny >= 0
                        && (nx as u32) < w
                        && (ny as u32) < h
                        && alpha[(ny * w as i64 + nx) as usize] >= SOLID_LEVEL
                    {
                        solid_neighbors += 1;
                    }
                }
            }
            if solid_neighbors < 2 {
                rgba.get_pixel_mut(x as u32, y as u32)[3] = 0;
            }
        }
    }
    DynamicImage::ImageRgba8(rgba)
}

fn get_mapping_target<'a>(
    layer_name: &str,
    full_mapping: &'a HashMap<String, String>,
) -> Option<&'a str> {
    // 完全一致を優先: "handwear-l" → arm_l のような左右別ターゲットを許可
    if let Some(target) = full_mapping.get(layer_name) {
        return Some(target.as_str());
    }
    let base = normalize_layer_name(layer_name);
    full_mapping.get(base).map(|s| s.as_str())
}

fn build_full_mapping(user_mapping: &HashMap<String, String>) -> HashMap<String, String> {
    let mut full: HashMap<String, String> = HashMap::new();
    for &(name, target) in FIXED_MAPPINGS {
        full.insert(name.to_string(), target.to_string());
    }
    for &(name, default_target) in ADJUSTABLE_DEFAULTS {
        let target = user_mapping
            .get(name)
            .cloned()
            .unwrap_or_else(|| default_target.to_string());
        full.insert(name.to_string(), target);
    }
    // ベース名以外のユーザー指定（"handwear-l" 等のサフィックス付きキー）も保持する
    for (name, target) in user_mapping {
        full.insert(name.clone(), target.clone());
    }
    full
}

// ── Layer loading ────────────────────────────────────────────────────

fn load_layers_from_psd(
    path: &str,
) -> Result<(HashMap<String, DynamicImage>, Vec<String>), AppError> {
    let bytes = fs::read(path)?;
    let psd = psd::Psd::from_bytes(&bytes)
        .map_err(|e| AppError::General(format!("PSD読み込みエラー: {:?}", e)))?;

    let mut layers = HashMap::new();
    let mut layer_order = Vec::new();
    let doc_width = psd.width();
    let doc_height = psd.height();

    for layer in psd.layers() {
        let name = layer.name().to_lowercase().replace(' ', "_");
        if name.is_empty() {
            continue;
        }

        // psd crate's layer.rgba() returns FULL document-sized RGBA data
        // (doc_width * doc_height * 4 bytes), already positioned on the canvas.
        let rgba_data = layer.rgba();
        let expected_len = (doc_width * doc_height * 4) as usize;

        if rgba_data.len() != expected_len {
            eprintln!(
                "[PachiPakuGen] PSD layer '{}': unexpected rgba size {} (expected {}), skipping",
                name,
                rgba_data.len(),
                expected_len
            );
            continue;
        }

        // Check if layer has any non-transparent pixels
        let has_content = rgba_data.chunks(4).any(|px| px[3] > 0);
        if !has_content {
            continue;
        }

        let canvas = image::RgbaImage::from_raw(doc_width, doc_height, rgba_data)
            .ok_or_else(|| AppError::General(format!("PSDレイヤー '{}' のRGBA変換に失敗", name)))?;

        layer_order.push(name.clone());
        layers.insert(name, DynamicImage::ImageRgba8(canvas));
    }
    Ok((layers, layer_order))
}

// ── Layer merging ────────────────────────────────────────────────────

/// Merge layers for a target, EXCLUDING specific base layer names.
fn load_depth_maps_for_psd(
    path: &str,
    layers: &HashMap<String, DynamicImage>,
    width: u32,
    height: u32,
) -> HashMap<String, GrayImage> {
    let psd_path = Path::new(path);
    let Some(parent) = psd_path.parent() else {
        return HashMap::new();
    };
    let Some(stem) = psd_path.file_stem().and_then(|value| value.to_str()) else {
        return HashMap::new();
    };
    let asset_dir = parent.join(stem);
    if !asset_dir.is_dir() {
        return HashMap::new();
    }

    let mut depth_maps = HashMap::new();
    for layer_name in layers.keys() {
        let depth_name = normalize_layer_name(layer_name).replace('_', " ");
        let depth_path = asset_dir.join(format!("{depth_name}_depth.png"));
        let Ok(depth) = image::open(&depth_path) else {
            continue;
        };
        let depth = if depth.width() != width || depth.height() != height {
            depth
                .resize_exact(width, height, image::imageops::FilterType::Nearest)
                .to_luma8()
        } else {
            depth.to_luma8()
        };
        depth_maps.insert(layer_name.clone(), depth);
    }

    eprintln!(
        "[PachiPakuGen] Loaded {} local depth maps from {}",
        depth_maps.len(),
        asset_dir.display()
    );
    depth_maps
}

fn merge_layers_for_target_excluding(
    slot_layers: &HashMap<String, DynamicImage>,
    depth_maps: &HashMap<String, GrayImage>,
    mapping: &HashMap<String, String>,
    target: &str,
    exclude: &[&str],
    source_layer_order: &[String],
    width: u32,
    height: u32,
) -> Option<DynamicImage> {
    let ordered_layers =
        ordered_layers_for_target(slot_layers, mapping, target, exclude, source_layer_order);

    if ordered_layers.is_empty() {
        return None;
    }
    Some(compose_depth_gated_layers(
        ordered_layers
            .into_iter()
            .map(|(name, layer)| (name, layer.to_rgba8()))
            .collect(),
        depth_maps,
        width,
        height,
    ))
}

fn merge_layers_for_target(
    slot_layers: &HashMap<String, DynamicImage>,
    depth_maps: &HashMap<String, GrayImage>,
    mapping: &HashMap<String, String>,
    target: &str,
    source_layer_order: &[String],
    width: u32,
    height: u32,
) -> Option<DynamicImage> {
    let ordered_layers =
        ordered_layers_for_target(slot_layers, mapping, target, &[], source_layer_order);

    if ordered_layers.is_empty() {
        return None;
    }
    Some(compose_depth_gated_layers(
        ordered_layers
            .into_iter()
            .map(|(name, layer)| (name, layer.to_rgba8()))
            .collect(),
        depth_maps,
        width,
        height,
    ))
}

fn merge_eye_layers_for_base(
    slot_layers: &HashMap<String, DynamicImage>,
    _mapping: &HashMap<String, String>,
    _source_layer_order: &[String],
    width: u32,
    height: u32,
) -> Option<DynamicImage> {
    let mut result = RgbaImage::new(width, height);
    let mut found = false;
    for target_name in ["eyewhite", "irides", "eyelash", "eyebrow"] {
        let mut matching_layers: Vec<_> = slot_layers
            .iter()
            .filter(|(layer_name, _)| normalize_layer_name(layer_name) == target_name)
            .collect();
        matching_layers.sort_by(|(left, _), (right, _)| left.cmp(right));
        for (_, layer) in matching_layers {
            let rgba = layer.to_rgba8();
            if !rgba.pixels().any(|pixel| pixel[3] > 0) {
                continue;
            }
            alpha_composite_onto(&mut result, &rgba, width, height);
            found = true;
        }
    }
    found.then_some(DynamicImage::ImageRgba8(result))
}

fn merge_layers_for_names(
    slot_layers: &HashMap<String, DynamicImage>,
    depth_maps: &HashMap<String, GrayImage>,
    normalized_names: &[&str],
    source_layer_order: &[String],
    width: u32,
    height: u32,
) -> Option<DynamicImage> {
    let mut ordered_layers = Vec::new();
    for layer_name in source_layer_order {
        let base = normalize_layer_name(layer_name);
        if normalized_names.contains(&base) {
            if let Some(img) = slot_layers.get(layer_name.as_str()) {
                ordered_layers.push((layer_name.clone(), img.to_rgba8()));
            }
        }
    }
    if ordered_layers.is_empty() {
        return None;
    }
    Some(compose_depth_gated_layers(
        ordered_layers,
        depth_maps,
        width,
        height,
    ))
}

fn ordered_layers_for_target<'a>(
    slot_layers: &'a HashMap<String, DynamicImage>,
    mapping: &HashMap<String, String>,
    target: &str,
    exclude: &[&str],
    source_layer_order: &[String],
) -> Vec<(String, &'a DynamicImage)> {
    let mut ordered_layers = Vec::new();
    let mut added_names = HashSet::new();

    // See-Through stores layers back-to-front. Keep that direction for alpha
    // compositing; local depth maps handle crossings that a global order cannot.
    for layer_name in source_layer_order {
        let base = normalize_layer_name(layer_name);
        if exclude.contains(&base) {
            continue;
        }
        if get_mapping_target(layer_name, mapping) == Some(target) {
            if let Some(img) = slot_layers.get(layer_name.as_str()) {
                ordered_layers.push((layer_name.clone(), img));
                added_names.insert(layer_name.clone());
            }
        }
    }

    // Imported/manual layers may not exist in the original PSD order.
    let mut remaining: Vec<_> = slot_layers
        .iter()
        .filter(|(layer_name, _)| {
            !added_names.contains(*layer_name)
                && !exclude.contains(&normalize_layer_name(layer_name))
                && get_mapping_target(layer_name, mapping) == Some(target)
        })
        .collect();
    remaining.sort_by(|(left, _), (right, _)| left.cmp(right));
    ordered_layers.extend(
        remaining
            .into_iter()
            .map(|(layer_name, img)| (layer_name.clone(), img)),
    );
    ordered_layers
}

fn ensure_core_body_layers_for_custom_order(
    body_layer_order: &[String],
    slot_layers: &HashMap<String, DynamicImage>,
    mapping: &HashMap<String, String>,
    source_layer_order: &[String],
) -> Vec<String> {
    let mut order = body_layer_order.to_vec();
    let mut normalized: HashSet<String> = order
        .iter()
        .map(|layer_name| normalize_layer_name(layer_name))
        .map(str::to_string)
        .collect();

    for required in ["face", "nose"] {
        if normalized.contains(required) {
            continue;
        }
        let source_name = source_layer_order
            .iter()
            .find(|layer_name| {
                slot_layers.contains_key(layer_name.as_str())
                    && normalize_layer_name(layer_name) == required
                    && get_mapping_target(layer_name, mapping) == Some("body")
            })
            .or_else(|| {
                slot_layers.keys().find(|layer_name| {
                    normalize_layer_name(layer_name) == required
                        && get_mapping_target(layer_name, mapping) == Some("body")
                })
            });
        if let Some(source_name) = source_name {
            order.insert(0, source_name.clone());
            normalized.insert(required.to_string());
        }
    }

    order
}

fn layer_order_entry_target<'a>(
    layer_name: &str,
    patches: &'a [LayerPatch],
    mapping: &'a HashMap<String, String>,
) -> Option<&'a str> {
    let source_name = patches
        .iter()
        .find(|patch| patch.id == layer_name)
        .map(|patch| patch.source_layer.as_str())
        .unwrap_or(layer_name);
    get_mapping_target(source_name, mapping)
}

/// unifiedレイヤー順（top=front）から出力パーツ間の描画順（背面→前面）を導出する。
/// 各パーツの深さは原則として所属レイヤーのインデックス平均（大きいほど背面）。
/// bodyだけは多数の顔・服・首レイヤーを含むため平均だと腕の前後操作が薄まる。
/// 腕と実際に重なる胴体の基準として topwear を優先し、素材に無ければ順次
/// bottomwear / neckwear / face / neck / nose を使う。
/// sway_* は個別の出力名を保持し、髪飾りと獣耳のように前後が異なるパーツを
/// 一括グループへ潰さない。eye/mouth/chest は body の直前面に固定挿入する。
pub(crate) fn derive_group_draw_order(
    unified_order: &[String],
    patches: &[LayerPatch],
    mapping: &HashMap<String, String>,
) -> Vec<String> {
    const GROUPS: &[&str] = &["hair_back", "hair", "body", "arm_l", "arm_r"];
    let mut sums: HashMap<String, (f64, u32)> = HashMap::new();
    for (index, layer_name) in unified_order.iter().enumerate() {
        let Some(target) = layer_order_entry_target(layer_name, patches, mapping) else {
            continue;
        };
        let patch = patches.iter().find(|patch| patch.id == *layer_name);
        // sway_* はそれぞれ独立した出力PNGなので、ターゲット名をそのまま保持する。
        // 腕から切り出したパッチも、親腕とは別のz位置を保持するリンクパーツにする。
        let group = if patch.is_some() && (target == "arm_l" || target == "arm_r") {
            patch.map(|patch| arm_overlay_part_name(target, &patch.id))
        } else if target.starts_with("sway_") {
            Some(target.to_string())
        } else {
            GROUPS
                .iter()
                .copied()
                .find(|group| *group == target)
                .map(str::to_string)
        };
        if let Some(group) = group {
            let entry = sums.entry(group).or_insert((0.0, 0));
            entry.0 += index as f64;
            entry.1 += 1;
        }
    }
    if sums.is_empty() {
        return Vec::new();
    }
    let mut order: Vec<(String, f64)> = sums
        .into_iter()
        .map(|(group, (sum, count))| (group, sum / count as f64))
        .collect();
    if let Some(body_depth) = preferred_body_anchor_depth(unified_order, patches, mapping) {
        if let Some((_, depth)) = order.iter_mut().find(|(group, _)| group == "body") {
            *depth = body_depth;
        }
    }
    // top=front なので深さの降順 = 背面→前面
    order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut result: Vec<String> = Vec::new();
    for (group, _) in order {
        result.push(group.clone());
        if group == "body" {
            result.push("chest".into());
            result.push("eye".into());
            result.push("mouth".into());
        }
    }
    result
}

/// 腕の前後判定に使うbody代表レイヤーの深さを返す。
/// STEP4ではユーザーが handwear と topwear の局所的な上下を見て調整するため、
/// body全体の平均ではなく、その見た目に対応する胴体レイヤーを基準にする。
fn preferred_body_anchor_depth(
    unified_order: &[String],
    patches: &[LayerPatch],
    mapping: &HashMap<String, String>,
) -> Option<f64> {
    const BODY_ANCHORS: &[&str] = &[
        "topwear",
        "bottomwear",
        "neckwear",
        "face",
        "neck",
        "nose",
    ];
    for anchor in BODY_ANCHORS {
        let mut sum = 0usize;
        let mut count = 0usize;
        for (index, layer_name) in unified_order.iter().enumerate() {
            // 切り出しパッチは独立した描画位置なので、body基準深度には混ぜない。
            if patches.iter().any(|patch| patch.id == *layer_name) {
                continue;
            }
            let source_name = patches
                .iter()
                .find(|patch| patch.id == layer_name.as_str())
                .map(|patch| patch.source_layer.as_str())
                .unwrap_or(layer_name);
            if normalize_layer_name(source_name) == *anchor
                && get_mapping_target(source_name, mapping) == Some("body")
            {
                sum += index;
                count += 1;
            }
        }
        if count > 0 {
            return Some(sum as f64 / count as f64);
        }
    }
    None
}

fn filter_layer_order_for_target(
    layer_order: &[String],
    patches: &[LayerPatch],
    mapping: &HashMap<String, String>,
    target: &str,
) -> Vec<String> {
    layer_order
        .iter()
        .filter(|layer_name| layer_order_entry_target(layer_name, patches, mapping) == Some(target))
        .cloned()
        .collect()
}

/// 腕本体と、STEP4で腕から切り出した前景パッチを別々の出力パーツにする。
/// パッチは描画順だけ独立し、Motion Labでは親腕と同じ変形へ追従する。
fn build_linked_arm_parts(
    current: &HashMap<String, DynamicImage>,
    depth_maps: &HashMap<String, GrayImage>,
    mapping: &HashMap<String, String>,
    unified_order: &[String],
    patches: &[LayerPatch],
    source_layer_order: &[String],
    arm_target: &str,
    width: u32,
    height: u32,
) -> Result<HashMap<String, DynamicImage>, AppError> {
    let mut result = HashMap::new();
    let arm_order = filter_layer_order_for_target(unified_order, patches, mapping, arm_target);
    if arm_order.is_empty() {
        if let Some(image) = merge_layers_for_target(
            current,
            depth_maps,
            mapping,
            arm_target,
            source_layer_order,
            width,
            height,
        ) {
            result.insert(arm_target.to_string(), image);
        }
        return Ok(result);
    }

    let active_patches = active_patches_for_order(patches, &arm_order);
    let patch_masks = prepare_patch_masks(&active_patches, width, height)?;
    let patch_ids: HashSet<&str> = active_patches
        .iter()
        .map(|patch| patch.id.as_str())
        .collect();

    let mut main_order: Vec<String> = arm_order
        .iter()
        .filter(|layer_name| !patch_ids.contains(layer_name.as_str()))
        .cloned()
        .collect();
    main_order.reverse();
    let main_layers = collect_ordered_render_layers(
        current,
        &main_order,
        &active_patches,
        &patch_masks,
        None,
    );
    if !main_layers.is_empty() {
        result.insert(
            arm_target.to_string(),
            compose_depth_gated_layers(main_layers, depth_maps, width, height),
        );
    }

    for patch in &active_patches {
        let overlay_layers = collect_ordered_render_layers(
            current,
            std::slice::from_ref(&patch.id),
            &active_patches,
            &patch_masks,
            None,
        );
        if !overlay_layers.is_empty() {
            result.insert(
                arm_overlay_part_name(arm_target, &patch.id),
                compose_depth_gated_layers(overlay_layers, depth_maps, width, height),
            );
        }
    }

    Ok(result)
}

fn ordered_layer_names_for_target(
    slot_layers: &HashMap<String, DynamicImage>,
    mapping: &HashMap<String, String>,
    target: &str,
    source_layer_order: &[String],
) -> Vec<String> {
    let mut names: Vec<String> = source_layer_order
        .iter()
        .rev()
        .filter(|layer_name| {
            slot_layers.contains_key(layer_name.as_str())
                && get_mapping_target(layer_name, mapping) == Some(target)
        })
        .cloned()
        .collect();
    let mut missing: Vec<String> = slot_layers
        .keys()
        .filter(|layer_name| {
            !names.contains(layer_name) && get_mapping_target(layer_name, mapping) == Some(target)
        })
        .cloned()
        .collect();
    missing.sort();
    missing.extend(names);
    names = missing;
    names
}

fn compose_depth_gated_layers(
    mut layers: Vec<(String, RgbaImage)>,
    depth_maps: &HashMap<String, GrayImage>,
    width: u32,
    height: u32,
) -> DynamicImage {
    if layers.is_empty() {
        return DynamicImage::new_rgba8(width, height);
    }

    let mut closest_depth = vec![u16::MAX; (width * height) as usize];
    let mut source_alpha = vec![0u8; (width * height) as usize];
    for (name, rgba) in &layers {
        for y in 0..height {
            for x in 0..width {
                let index = (y * width + x) as usize;
                source_alpha[index] = union_alpha(source_alpha[index], rgba.get_pixel(x, y)[3]);
            }
        }
        let Some(depth) = depth_maps.get(name) else {
            continue;
        };
        if depth.width() != width || depth.height() != height {
            continue;
        }
        for y in 0..height {
            for x in 0..width {
                if rgba.get_pixel(x, y)[3] == 0 {
                    continue;
                }
                let index = (y * width + x) as usize;
                closest_depth[index] = closest_depth[index].min(depth.get_pixel(x, y)[0] as u16);
            }
        }
    }

    let tolerance = DEPTH_VISIBILITY_TOLERANCE as u16;
    let mut result = RgbaImage::new(width, height);
    for (name, rgba) in &mut layers {
        if let Some(depth) = depth_maps.get(name) {
            if depth.width() == width && depth.height() == height {
                apply_feathered_depth_visibility(
                    rgba,
                    depth,
                    &closest_depth,
                    tolerance,
                    width,
                    height,
                );
            }
        }
        alpha_composite_onto(&mut result, rgba, width, height);
    }
    for y in 0..height {
        for x in 0..width {
            result.get_pixel_mut(x, y)[3] = source_alpha[(y * width + x) as usize];
        }
    }
    DynamicImage::ImageRgba8(result)
}

fn union_alpha(background: u8, foreground: u8) -> u8 {
    let background = background as u16;
    let foreground = foreground as u16;
    (foreground + ((background * (255 - foreground) + 127) / 255)).min(255) as u8
}

fn apply_feathered_depth_visibility(
    rgba: &mut RgbaImage,
    depth: &GrayImage,
    closest_depth: &[u16],
    tolerance: u16,
    width: u32,
    height: u32,
) {
    let mut visibility = GrayImage::from_pixel(width, height, image::Luma([255]));
    let mut visible_pixels = 0u64;
    let mut hidden_pixels = 0u64;

    for y in 0..height {
        for x in 0..width {
            if rgba.get_pixel(x, y)[3] == 0 {
                continue;
            }
            let index = (y * width + x) as usize;
            let hidden = closest_depth[index] != u16::MAX
                && depth.get_pixel(x, y)[0] as u16 > closest_depth[index].saturating_add(tolerance);
            if hidden {
                visibility.put_pixel(x, y, image::Luma([0]));
                hidden_pixels += 1;
            } else {
                visible_pixels += 1;
            }
        }
    }

    if hidden_pixels == 0 {
        return;
    }
    if visible_pixels == 0 {
        for pixel in rgba.pixels_mut() {
            pixel[3] = 0;
        }
        return;
    }

    let visibility = image::imageops::blur(&visibility, DEPTH_VISIBILITY_FEATHER_SIGMA);
    for y in 0..height {
        for x in 0..width {
            let alpha = rgba.get_pixel(x, y)[3] as u16;
            if alpha == 0 {
                continue;
            }
            let mask = visibility.get_pixel(x, y)[0] as u16;
            rgba.get_pixel_mut(x, y)[3] = ((alpha * mask + 127) / 255) as u8;
        }
    }
}

fn extract_original_target_pixels(
    original: &DynamicImage,
    target_mask: &DynamicImage,
    slot_layers: &HashMap<String, DynamicImage>,
    depth_maps: &HashMap<String, GrayImage>,
    mapping: &HashMap<String, String>,
    target: &str,
    occluding_target: Option<&str>,
    width: u32,
    height: u32,
) -> Option<DynamicImage> {
    if original.width() != width
        || original.height() != height
        || target_mask.width() != width
        || target_mask.height() != height
    {
        return None;
    }

    let original = original.to_rgba8();
    let target_mask = target_mask.to_rgba8();
    let target_depth =
        closest_depth_for_target(slot_layers, depth_maps, mapping, target, width, height);
    let occluding_depth = occluding_target.map(|occluder| {
        closest_depth_for_target(slot_layers, depth_maps, mapping, occluder, width, height)
    });
    let strong_bounds = alpha_bounds_above(&target_mask, 8)?;
    let padding = 24u32;
    let min_x = strong_bounds.x.saturating_sub(padding);
    let min_y = strong_bounds.y.saturating_sub(padding);
    let max_x = (strong_bounds.x + strong_bounds.width + padding).min(width);
    let max_y = (strong_bounds.y + strong_bounds.height + padding).min(height);
    let tolerance = DEPTH_VISIBILITY_TOLERANCE as u16;
    let mut result = RgbaImage::new(width, height);

    for y in min_y..max_y {
        for x in min_x..max_x {
            let index = (y * width + x) as usize;
            let mask_alpha = target_mask.get_pixel(x, y)[3];
            if mask_alpha == 0 {
                continue;
            }
            if let Some(occluding_depth) = occluding_depth.as_ref() {
                if target_depth[index] != u16::MAX
                    && occluding_depth[index] != u16::MAX
                    && occluding_depth[index].saturating_add(tolerance) < target_depth[index]
                {
                    continue;
                }
            }
            let mut pixel = *original.get_pixel(x, y);
            pixel[3] = ((pixel[3] as u16 * mask_alpha as u16 + 127) / 255) as u8;
            result.put_pixel(x, y, pixel);
        }
    }
    Some(DynamicImage::ImageRgba8(result))
}

fn alpha_bounds_above(src: &RgbaImage, threshold: u8) -> Option<LayerBounds> {
    let mut min_x = src.width();
    let mut min_y = src.height();
    let mut max_x = 0;
    let mut max_y = 0;
    let mut found = false;

    for (x, y, pixel) in src.enumerate_pixels() {
        if pixel[3] <= threshold {
            continue;
        }
        found = true;
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }

    found.then_some(LayerBounds {
        x: min_x,
        y: min_y,
        width: max_x - min_x + 1,
        height: max_y - min_y + 1,
    })
}

fn saturated_alpha_pixel_count(image: &DynamicImage) -> u64 {
    image
        .to_rgba8()
        .pixels()
        .filter(|pixel| {
            if pixel[3] <= 8 {
                return false;
            }
            let max = pixel[0].max(pixel[1]).max(pixel[2]);
            let min = pixel[0].min(pixel[1]).min(pixel[2]);
            max.saturating_sub(min) > 32
        })
        .count() as u64
}

/// Preserve cross-category crossings without changing the three-part export
/// contract. Body pixels that are locally closer than front hair are copied
/// into the hair overlay, which is composited last at runtime.
fn promote_body_foreground_over_hair(
    body: &DynamicImage,
    hair: &mut DynamicImage,
    slot_layers: &HashMap<String, DynamicImage>,
    depth_maps: &HashMap<String, GrayImage>,
    mapping: &HashMap<String, String>,
    width: u32,
    height: u32,
) -> u64 {
    let body_depth =
        closest_depth_for_target(slot_layers, depth_maps, mapping, "body", width, height);
    let hair_depth =
        closest_depth_for_target(slot_layers, depth_maps, mapping, "hair", width, height);
    let body_rgba = body.to_rgba8();
    let mut hair_rgba = hair.to_rgba8();
    let mut foreground_mask = GrayImage::new(width, height);
    let tolerance = DEPTH_VISIBILITY_TOLERANCE as u16;
    let mut promoted_pixels = 0u64;

    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            if body_rgba.get_pixel(x, y)[3] == 0
                || hair_rgba.get_pixel(x, y)[3] == 0
                || body_depth[index] == u16::MAX
                || hair_depth[index] == u16::MAX
            {
                continue;
            }
            if body_depth[index].saturating_add(tolerance) < hair_depth[index] {
                foreground_mask.put_pixel(x, y, image::Luma([255]));
                promoted_pixels += 1;
            }
        }
    }

    if promoted_pixels == 0 {
        return 0;
    }

    let foreground_mask = image::imageops::blur(&foreground_mask, DEPTH_VISIBILITY_FEATHER_SIGMA);
    let mut foreground = body_rgba;
    for y in 0..height {
        for x in 0..width {
            let mask = foreground_mask.get_pixel(x, y)[0] as u16;
            let pixel = foreground.get_pixel_mut(x, y);
            pixel[3] = ((pixel[3] as u16 * mask + 127) / 255) as u8;
        }
    }
    alpha_composite_onto(&mut hair_rgba, &foreground, width, height);
    *hair = DynamicImage::ImageRgba8(hair_rgba);
    promoted_pixels
}

/// 体より手前にある後ろ髪ピクセルを前髪カテゴリへ昇格する。
/// 肩の前へ垂れる房は深度上 body より手前だが、SpriTalkの固定z順
/// （hair_back < body）では body の裏に隠れて「髪が欠けた」ように見える。
/// 深度マップに基づき該当ピクセルを hair_back から除去し hair の下層へ移す
fn promote_hair_back_foreground_over_body(
    body: &DynamicImage,
    hair_back: &mut DynamicImage,
    hair: &mut Option<DynamicImage>,
    slot_layers: &HashMap<String, DynamicImage>,
    depth_maps: &HashMap<String, GrayImage>,
    mapping: &HashMap<String, String>,
    width: u32,
    height: u32,
) -> u64 {
    let body_depth =
        closest_depth_for_target(slot_layers, depth_maps, mapping, "body", width, height);
    let back_depth =
        closest_depth_for_target(slot_layers, depth_maps, mapping, "hair_back", width, height);
    let body_rgba = body.to_rgba8();
    let back_rgba = hair_back.to_rgba8();
    let mut foreground_mask = GrayImage::new(width, height);
    let tolerance = DEPTH_VISIBILITY_TOLERANCE as u16;
    let mut promoted_pixels = 0u64;

    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            if back_rgba.get_pixel(x, y)[3] == 0
                || body_rgba.get_pixel(x, y)[3] == 0
                || body_depth[index] == u16::MAX
                || back_depth[index] == u16::MAX
            {
                continue;
            }
            if back_depth[index].saturating_add(tolerance) < body_depth[index] {
                foreground_mask.put_pixel(x, y, image::Luma([255]));
                promoted_pixels += 1;
            }
        }
    }

    if promoted_pixels == 0 {
        return 0;
    }

    let foreground_mask = image::imageops::blur(&foreground_mask, DEPTH_VISIBILITY_FEATHER_SIGMA);
    // 非破壊コピー方式: hair_back からは削らず、前景分だけを hair 側へ複製する。
    // 削る方式は深度マップの誤りやフェザーの滲みで髪に透明穴を開けるため廃止。
    // 同一ピクセルの重ね描きになるので静止時の見た目は変わらず、
    // z順の問題（bodyの裏に隠れて髪が欠ける）だけが前髪側の複製で解消される
    let mut foreground = back_rgba.clone();
    for y in 0..height {
        for x in 0..width {
            let mask = foreground_mask.get_pixel(x, y)[0] as u16;
            let pixel = foreground.get_pixel_mut(x, y);
            pixel[3] = ((pixel[3] as u16 * mask + 127) / 255) as u8;
        }
    }
    // 昇格分は既存の前髪の「下」に敷く（前髪そのものが常に最前面）
    let mut combined = foreground;
    if let Some(existing_hair) = hair.take() {
        alpha_composite_onto(&mut combined, &existing_hair.to_rgba8(), width, height);
    }
    *hair = Some(DynamicImage::ImageRgba8(combined));
    let _ = back_rgba;
    promoted_pixels
}

fn closest_depth_for_target(
    slot_layers: &HashMap<String, DynamicImage>,
    depth_maps: &HashMap<String, GrayImage>,
    mapping: &HashMap<String, String>,
    target: &str,
    width: u32,
    height: u32,
) -> Vec<u16> {
    let mut closest_depth = vec![u16::MAX; (width * height) as usize];
    for (layer_name, layer) in slot_layers {
        if get_mapping_target(layer_name, mapping) != Some(target) {
            continue;
        }
        let Some(depth) = depth_maps.get(layer_name) else {
            continue;
        };
        if depth.width() != width || depth.height() != height {
            continue;
        }
        let rgba = layer.to_rgba8();
        if rgba.width() != width || rgba.height() != height {
            continue;
        }
        for y in 0..height {
            for x in 0..width {
                if rgba.get_pixel(x, y)[3] == 0 {
                    continue;
                }
                let index = (y * width + x) as usize;
                closest_depth[index] = closest_depth[index].min(depth.get_pixel(x, y)[0] as u16);
            }
        }
    }
    closest_depth
}

fn append_missing_layer_names(
    layer_names: &mut Vec<String>,
    slot_layers: &HashMap<String, DynamicImage>,
) {
    let mut missing: Vec<String> = slot_layers
        .keys()
        .filter(|name| !layer_names.contains(name))
        .cloned()
        .collect();
    missing.sort();
    layer_names.extend(missing);
}

fn alpha_composite_onto(
    dst: &mut image::RgbaImage,
    src: &image::RgbaImage,
    width: u32,
    height: u32,
) {
    for y in 0..height {
        for x in 0..width {
            let sp = src.get_pixel(x, y);
            let sa = sp[3] as f32 / 255.0;
            if sa > 0.0 {
                let dp = dst.get_pixel(x, y);
                let da = dp[3] as f32 / 255.0;
                let out_a = sa + da * (1.0 - sa);
                if out_a > 0.0 {
                    let r = (sp[0] as f32 * sa + dp[0] as f32 * da * (1.0 - sa)) / out_a;
                    let g = (sp[1] as f32 * sa + dp[1] as f32 * da * (1.0 - sa)) / out_a;
                    let b = (sp[2] as f32 * sa + dp[2] as f32 * da * (1.0 - sa)) / out_a;
                    dst.put_pixel(
                        x,
                        y,
                        image::Rgba([
                            r.clamp(0.0, 255.0) as u8,
                            g.clamp(0.0, 255.0) as u8,
                            b.clamp(0.0, 255.0) as u8,
                            (out_a * 255.0).clamp(0.0, 255.0) as u8,
                        ]),
                    );
                }
            }
        }
    }
}

fn sam3_select_region_inner(
    image_data_url: &str,
    points: &[(f64, f64)],
) -> Result<Sam3SelectResult, AppError> {
    let image = decode_data_url_image(image_data_url)?;
    let checkpoint = neck_extract::find_sam3_checkpoint();
    let mask = neck_extract::extract_mask_with_sam3_point(&image, points, checkpoint.as_deref())?;
    let (width, height) = (image.width(), image.height());
    // ブラシと同じ表示色(233,69,96)でRGBAマスクを作り、パッチマスクcanvasへ
    // そのまま合成できる形にする（アルファ=SAM3マスク値）
    let mut rgba = RgbaImage::new(width, height);
    for (index, pixel) in rgba.pixels_mut().enumerate() {
        let value = mask.get(index).copied().unwrap_or(0);
        *pixel = image::Rgba([233, 69, 96, value]);
    }
    Ok(Sam3SelectResult {
        mask_png: image_utils::image_to_base64_png(&DynamicImage::ImageRgba8(rgba)),
    })
}

/// 胸を切出（852話式・コピー方式）。塗ったマスク範囲を body から chest として複製する。
/// body からは**抜かない** — 胸の背後に体が残るため、揺れても穴が空かず自然に見える
/// （852話 Anime2.5DRig と同方式）。マスク無しなら body をそのまま返す
fn cut_chest_from_body(
    body: DynamicImage,
    chest_mask_png: Option<&str>,
    parts: &mut HashMap<String, DynamicImage>,
    width: u32,
    height: u32,
) -> Result<DynamicImage, AppError> {
    let Some(mask_png) = chest_mask_png.filter(|s| !s.is_empty()) else {
        return Ok(body);
    };
    let mask = decode_mask_png(mask_png, width, height)?;
    let body_rgba = body.to_rgba8();
    let chest_rgba = apply_mask_to_rgba(&body_rgba, &mask, false);
    parts.insert("chest".to_string(), DynamicImage::ImageRgba8(chest_rgba));
    Ok(body)
}

fn decode_data_url_image(data_uri: &str) -> Result<DynamicImage, AppError> {
    let encoded = data_uri
        .split_once(',')
        .map(|(_, data)| data)
        .unwrap_or(data_uri);
    let bytes = STANDARD
        .decode(encoded)
        .map_err(|e| AppError::General(format!("画像のデコードに失敗: {}", e)))?;
    image::load_from_memory(&bytes)
        .map_err(|e| AppError::General(format!("画像の読み込みに失敗: {}", e)))
}

fn decode_mask_png(data_uri: &str, width: u32, height: u32) -> Result<image::GrayImage, AppError> {
    let encoded = data_uri
        .split_once(',')
        .map(|(_, data)| data)
        .unwrap_or(data_uri);
    let bytes = STANDARD
        .decode(encoded)
        .map_err(|e| AppError::General(format!("マスクPNGのデコードに失敗: {}", e)))?;
    let img = image::load(Cursor::new(bytes), image::ImageFormat::Png)
        .map_err(|e| AppError::General(format!("マスクPNGの読み込みに失敗: {}", e)))?;
    let img = if img.width() != width || img.height() != height {
        img.resize_exact(width, height, image::imageops::FilterType::Nearest)
    } else {
        img
    };
    let rgba = img.to_rgba8();
    let mut mask = image::GrayImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let p = rgba.get_pixel(x, y);
            mask.put_pixel(x, y, image::Luma([p[3].max(p[0])]));
        }
    }
    Ok(mask)
}

fn apply_mask_to_rgba(
    src: &image::RgbaImage,
    mask: &image::GrayImage,
    invert: bool,
) -> image::RgbaImage {
    let mut out = src.clone();
    for (x, y, pixel) in out.enumerate_pixels_mut() {
        let m = mask.get_pixel(x, y)[0] as u16;
        let factor = if invert { 255 - m } else { m };
        pixel[3] = ((pixel[3] as u16 * factor) / 255) as u8;
    }
    out
}

fn subtract_mask_from_rgba(src: &mut image::RgbaImage, mask: &image::GrayImage) {
    for (x, y, pixel) in src.enumerate_pixels_mut() {
        let keep = 255u16.saturating_sub(mask.get_pixel(x, y)[0] as u16);
        pixel[3] = ((pixel[3] as u16 * keep) / 255) as u8;
    }
}

fn apply_opacity(src: &mut image::RgbaImage, opacity: f32) {
    if (opacity - 1.0).abs() < f32::EPSILON {
        return;
    }
    for pixel in src.pixels_mut() {
        pixel[3] = (pixel[3] as f32 * opacity).round().clamp(0.0, 255.0) as u8;
    }
}

fn alpha_bounds(src: &image::RgbaImage) -> Option<LayerBounds> {
    let mut min_x = src.width();
    let mut min_y = src.height();
    let mut max_x = 0;
    let mut max_y = 0;
    let mut found = false;

    for (x, y, pixel) in src.enumerate_pixels() {
        if pixel[3] == 0 {
            continue;
        }
        found = true;
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }

    if found {
        Some(LayerBounds {
            x: min_x,
            y: min_y,
            width: max_x - min_x + 1,
            height: max_y - min_y + 1,
        })
    } else {
        None
    }
}

fn generate_composite_preview(
    parts: &HashMap<String, DynamicImage>,
    width: u32,
    height: u32,
    draw_order: &[String],
) -> String {
    let result = compose_parts_preview(parts, width, height, draw_order);
    let composite = DynamicImage::ImageRgba8(result);
    image_utils::image_to_base64_png(&composite)
}

/// 素体パーツを出力時と同じ背面→前面順で合成する。
/// draw_order が空なら旧データ向けの固定順へフォールバックする。
fn compose_parts_preview(
    parts: &HashMap<String, DynamicImage>,
    width: u32,
    height: u32,
    draw_order: &[String],
) -> RgbaImage {
    const DEFAULT_ORDER: &[&str] = &[
        "hair_back",
        "body",
        "chest",
        "arm_l",
        "arm_r",
        "sways",
        "eye",
        "mouth",
        "hair",
    ];
    let mut result = image::RgbaImage::new(width, height);
    let order: Vec<&str> = if draw_order.is_empty() {
        DEFAULT_ORDER.to_vec()
    } else {
        draw_order.iter().map(String::as_str).collect()
    };
    let explicitly_ordered_sways: HashSet<&str> = order
        .iter()
        .copied()
        .filter(|key| key.starts_with("sway_"))
        .collect();
    let mut eye_keys: Vec<&String> = parts
        .keys()
        .filter(|key| key.starts_with("eye_"))
        .collect();
    eye_keys.sort();
    let mut mouth_keys: Vec<&String> = parts
        .keys()
        .filter(|key| key.starts_with("mouth_") || key.as_str() == "mouth_closed")
        .collect();
    mouth_keys.sort();

    for key in order {
        match key {
            "eye" => {
                if let Some(image) = eye_keys.first().and_then(|key| parts.get(*key)) {
                    alpha_composite_onto(&mut result, &image.to_rgba8(), width, height);
                }
            }
            "mouth" => {
                if let Some(image) = mouth_keys.first().and_then(|key| parts.get(*key)) {
                    alpha_composite_onto(&mut result, &image.to_rgba8(), width, height);
                }
            }
            "sways" => {
                let mut sway_keys: Vec<&String> = parts
                    .keys()
                    .filter(|key| {
                        key.starts_with("sway_")
                            && !explicitly_ordered_sways.contains(key.as_str())
                    })
                    .collect();
                sway_keys.sort();
                for sway_key in sway_keys {
                    if let Some(image) = parts.get(sway_key) {
                        alpha_composite_onto(&mut result, &image.to_rgba8(), width, height);
                    }
                }
            }
            _ => {
                if let Some(image) = parts.get(key) {
                    alpha_composite_onto(&mut result, &image.to_rgba8(), width, height);
                }
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derive_group_draw_order_reflects_unified_order() {
        // top=front: 前髪の前に犬耳(headwear→hair)、腕はbodyの背面
        let unified: Vec<String> = [
            "headwear",   // hair
            "front_hair", // hair
            "face",       // body
            "neck",       // body
            "back_hair",  // hair_back
            "handwear-l", // arm_l
            "handwear-r", // arm_r
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let mut mapping = HashMap::new();
        mapping.insert("headwear".to_string(), "hair".to_string());
        mapping.insert("front_hair".to_string(), "hair".to_string());
        mapping.insert("face".to_string(), "body".to_string());
        mapping.insert("neck".to_string(), "body".to_string());
        mapping.insert("back_hair".to_string(), "hair_back".to_string());
        mapping.insert("handwear-l".to_string(), "arm_l".to_string());
        mapping.insert("handwear-r".to_string(), "arm_r".to_string());

        let order = derive_group_draw_order(&unified, &[], &mapping);

        // 背面→前面: arm_r, arm_l, hair_back, body(+chest/eye/mouth), hair
        assert_eq!(
            order,
            vec![
                "arm_r", "arm_l", "hair_back", "body", "chest", "eye", "mouth", "hair"
            ]
        );
    }

    #[test]
    fn derive_group_draw_order_empty_without_known_targets() {
        let order = derive_group_draw_order(&["unknown".to_string()], &[], &HashMap::new());
        assert!(order.is_empty());
    }

    #[test]
    fn derive_group_draw_order_preserves_individual_sway_depths() {
        // top=front: 髪飾りは前髪より前、左右の獣耳は前髪より後ろ。
        let unified: Vec<String> = [
            "headwear",
            "front_hair",
            "ears-l",
            "ears-r",
            "face",
            "back_hair",
        ]
        .iter()
        .map(|name| name.to_string())
        .collect();
        let mapping = HashMap::from([
            ("headwear".to_string(), "sway_ear".to_string()),
            ("front_hair".to_string(), "hair".to_string()),
            ("ears-l".to_string(), "sway_ear_l".to_string()),
            ("ears-r".to_string(), "sway_ear_r".to_string()),
            ("face".to_string(), "body".to_string()),
            ("back_hair".to_string(), "hair_back".to_string()),
        ]);

        let order = derive_group_draw_order(&unified, &[], &mapping);

        assert_eq!(
            order,
            vec![
                "hair_back",
                "body",
                "chest",
                "eye",
                "mouth",
                "sway_ear_r",
                "sway_ear_l",
                "hair",
                "sway_ear",
            ]
        );
    }

    #[test]
    fn derive_group_draw_order_uses_topwear_for_arm_front_back() {
        // STEP4の表示順（top=front）。左腕は服より手前、右腕は服より奥。
        // body全体の平均では body=(2+4+5)/3=3.67 となり、index=3の右腕まで
        // 誤ってbody前面になるため、topwear(index=2)をbody基準にする。
        let unified: Vec<String> = [
            "front_hair",
            "handwear-l",
            "topwear",
            "handwear-r",
            "face",
            "neck",
            "back_hair",
        ]
        .iter()
        .map(|name| name.to_string())
        .collect();
        let mapping = HashMap::from([
            ("front_hair".to_string(), "hair".to_string()),
            ("handwear-l".to_string(), "arm_l".to_string()),
            ("topwear".to_string(), "body".to_string()),
            ("handwear-r".to_string(), "arm_r".to_string()),
            ("face".to_string(), "body".to_string()),
            ("neck".to_string(), "body".to_string()),
            ("back_hair".to_string(), "hair_back".to_string()),
        ]);

        let order = derive_group_draw_order(&unified, &[], &mapping);

        assert_eq!(
            order,
            vec![
                "hair_back",
                "arm_r",
                "body",
                "chest",
                "eye",
                "mouth",
                "arm_l",
                "hair",
            ]
        );
    }

    #[test]
    fn arm_patch_keeps_independent_depth_in_group_order() {
        // top=front: 指パッチはbody前、切り出し元の左腕はbody後ろ。
        let unified = vec![
            "finger_patch".to_string(),
            "topwear".to_string(),
            "handwear-l".to_string(),
        ];
        let patches = vec![LayerPatch {
            id: "finger_patch".to_string(),
            source_layer: "handwear-l".to_string(),
            mask_png: String::new(),
            cut_source: true,
        }];
        let mapping = HashMap::from([
            ("topwear".to_string(), "body".to_string()),
            ("handwear-l".to_string(), "arm_l".to_string()),
        ]);

        let order = derive_group_draw_order(&unified, &patches, &mapping);

        assert_eq!(
            order,
            vec![
                "arm_l",
                "body",
                "chest",
                "eye",
                "mouth",
                "arm_l_overlay_finger_patch",
            ]
        );
    }

    #[test]
    fn linked_arm_parts_cut_overlay_from_parent_image() {
        let mut arm_pixels = RgbaImage::new(2, 1);
        arm_pixels.put_pixel(0, 0, image::Rgba([255, 0, 0, 255]));
        arm_pixels.put_pixel(1, 0, image::Rgba([0, 255, 0, 255]));
        let current = HashMap::from([(
            "handwear-l".to_string(),
            DynamicImage::ImageRgba8(arm_pixels),
        )]);
        let mapping = HashMap::from([("handwear-l".to_string(), "arm_l".to_string())]);
        let unified = vec!["finger_patch".to_string(), "handwear-l".to_string()];
        let patches = vec![LayerPatch {
            id: "finger_patch".to_string(),
            source_layer: "handwear-l".to_string(),
            mask_png: encode_test_mask(&[0, 255], 2, 1),
            cut_source: true,
        }];

        let parts = build_linked_arm_parts(
            &current,
            &HashMap::new(),
            &mapping,
            &unified,
            &patches,
            &["handwear-l".to_string()],
            "arm_l",
            2,
            1,
        )
        .unwrap();
        let parent = parts.get("arm_l").unwrap().to_rgba8();
        let overlay = parts
            .get("arm_l_overlay_finger_patch")
            .unwrap()
            .to_rgba8();

        assert_eq!(parent.get_pixel(0, 0)[3], 255);
        assert_eq!(parent.get_pixel(1, 0)[3], 0);
        assert_eq!(overlay.get_pixel(0, 0)[3], 0);
        assert_eq!(overlay.get_pixel(1, 0).0, [0, 255, 0, 255]);
    }

    #[test]
    fn parts_preview_respects_explicit_arm_order() {
        let parts = HashMap::from([
            ("body".to_string(), solid(0, 0, 255)),
            ("arm_l".to_string(), solid(255, 0, 0)),
        ]);
        let behind = ["arm_l", "body"]
            .iter()
            .map(|key| key.to_string())
            .collect::<Vec<_>>();
        let front = ["body", "arm_l"]
            .iter()
            .map(|key| key.to_string())
            .collect::<Vec<_>>();

        let composite = compose_parts_preview(&parts, 1, 1, &behind);
        assert_eq!(composite.get_pixel(0, 0).0, [0, 0, 255, 255]);
        let composite = compose_parts_preview(&parts, 1, 1, &front);
        assert_eq!(composite.get_pixel(0, 0).0, [255, 0, 0, 255]);
    }

    fn solid(r: u8, g: u8, b: u8) -> DynamicImage {
        DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            1,
            1,
            image::Rgba([r, g, b, 255]),
        ))
    }

    fn encode_test_mask(alpha: &[u8], width: u32, height: u32) -> String {
        let mut image = RgbaImage::new(width, height);
        for (index, value) in alpha.iter().copied().enumerate() {
            let x = index as u32 % width;
            let y = index as u32 / width;
            image.put_pixel(x, y, image::Rgba([value, value, value, value]));
        }
        let mut cursor = Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(image)
            .write_to(&mut cursor, image::ImageFormat::Png)
            .unwrap();
        STANDARD.encode(cursor.into_inner())
    }

    #[test]
    fn target_merge_preserves_psd_back_to_front_order() {
        let layers = HashMap::from([
            ("face".to_string(), solid(255, 0, 0)),
            ("neck".to_string(), solid(0, 0, 255)),
        ]);
        let mapping = HashMap::from([
            ("face".to_string(), "body".to_string()),
            ("neck".to_string(), "body".to_string()),
        ]);
        let source_order = vec!["neck".to_string(), "face".to_string()];

        let merged = merge_layers_for_target(
            &layers,
            &HashMap::new(),
            &mapping,
            "body",
            &source_order,
            1,
            1,
        )
        .unwrap();

        assert_eq!(merged.to_rgba8().get_pixel(0, 0).0, [255, 0, 0, 255]);
    }

    #[test]
    fn category_names_are_front_first_for_ui() {
        let layers = HashMap::from([
            ("face".to_string(), solid(255, 0, 0)),
            ("neck".to_string(), solid(0, 0, 255)),
        ]);
        let mapping = HashMap::from([
            ("face".to_string(), "body".to_string()),
            ("neck".to_string(), "body".to_string()),
        ]);
        let source_order = vec!["neck".to_string(), "face".to_string()];

        assert_eq!(
            ordered_layer_names_for_target(&layers, &mapping, "body", &source_order),
            vec!["face".to_string(), "neck".to_string()]
        );
    }

    #[test]
    fn custom_body_order_keeps_core_face_parts_without_restoring_neck() {
        let layers = HashMap::from([
            ("face".to_string(), solid(255, 0, 0)),
            ("nose".to_string(), solid(0, 255, 0)),
            ("neck".to_string(), solid(0, 0, 255)),
            ("topwear".to_string(), solid(255, 255, 0)),
        ]);
        let mapping = HashMap::from([
            ("face".to_string(), "body".to_string()),
            ("nose".to_string(), "body".to_string()),
            ("neck".to_string(), "body".to_string()),
            ("topwear".to_string(), "body".to_string()),
        ]);
        let source_order = vec![
            "neck".to_string(),
            "face".to_string(),
            "nose".to_string(),
            "topwear".to_string(),
        ];

        let order = ensure_core_body_layers_for_custom_order(
            &["topwear".to_string()],
            &layers,
            &mapping,
            &source_order,
        );

        assert_eq!(
            order,
            vec![
                "nose".to_string(),
                "face".to_string(),
                "topwear".to_string()
            ]
        );
    }

    #[test]
    fn unified_layer_order_is_split_by_mapping_and_patch_source() {
        let mapping = HashMap::from([
            ("face".to_string(), "body".to_string()),
            ("neck".to_string(), "body".to_string()),
            ("front_hair".to_string(), "hair".to_string()),
            ("back_hair".to_string(), "hair_back".to_string()),
        ]);
        let patches = vec![LayerPatch {
            id: "neck_patch_1".to_string(),
            source_layer: "neck".to_string(),
            mask_png: String::new(),
            cut_source: true,
        }];
        let order = vec![
            "front_hair".to_string(),
            "neck_patch_1".to_string(),
            "neck".to_string(),
            "face".to_string(),
            "back_hair".to_string(),
        ];

        assert_eq!(
            filter_layer_order_for_target(&order, &patches, &mapping, "body"),
            vec![
                "neck_patch_1".to_string(),
                "neck".to_string(),
                "face".to_string()
            ]
        );
        assert_eq!(
            filter_layer_order_for_target(&order, &patches, &mapping, "hair"),
            vec!["front_hair".to_string()]
        );
        assert_eq!(
            filter_layer_order_for_target(&order, &patches, &mapping, "hair_back"),
            vec!["back_hair".to_string()]
        );
    }

    #[test]
    fn local_depth_masks_crossing_layers_per_pixel() {
        let layers = vec![
            (
                "back".to_string(),
                RgbaImage::from_pixel(9, 1, image::Rgba([0, 0, 255, 255])),
            ),
            (
                "front".to_string(),
                RgbaImage::from_pixel(9, 1, image::Rgba([255, 0, 0, 255])),
            ),
        ];
        let depth_maps = HashMap::from([
            (
                "back".to_string(),
                GrayImage::from_raw(9, 1, vec![10, 10, 10, 10, 20, 20, 20, 20, 20]).unwrap(),
            ),
            (
                "front".to_string(),
                GrayImage::from_raw(9, 1, vec![20, 20, 20, 20, 10, 10, 10, 10, 10]).unwrap(),
            ),
        ]);

        let merged = compose_depth_gated_layers(layers, &depth_maps, 9, 1).to_rgba8();

        assert_eq!(merged.get_pixel(0, 0).0, [0, 0, 255, 255]);
        assert_eq!(merged.get_pixel(8, 0).0, [255, 0, 0, 255]);
        assert_eq!(merged.get_pixel(3, 0)[3], 255);
        assert_eq!(merged.get_pixel(4, 0)[3], 255);
        assert!(merged.get_pixel(3, 0)[0] > 0);
        assert!(merged.get_pixel(4, 0)[2] > 0);
    }

    #[test]
    fn body_pixels_closer_than_hair_are_promoted_into_hair_overlay() {
        let width = 9;
        let body = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            width,
            1,
            image::Rgba([255, 0, 0, 255]),
        ));
        let mut hair = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            width,
            1,
            image::Rgba([0, 0, 255, 255]),
        ));
        let layers = HashMap::from([
            ("handwear-r".to_string(), body.clone()),
            ("front_hair".to_string(), hair.clone()),
        ]);
        let mapping = HashMap::from([
            ("handwear".to_string(), "body".to_string()),
            ("front_hair".to_string(), "hair".to_string()),
        ]);
        let depths = HashMap::from([
            (
                "handwear-r".to_string(),
                GrayImage::from_raw(width, 1, vec![10, 10, 10, 10, 20, 20, 20, 20, 20]).unwrap(),
            ),
            (
                "front_hair".to_string(),
                GrayImage::from_raw(width, 1, vec![20, 20, 20, 20, 10, 10, 10, 10, 10]).unwrap(),
            ),
        ]);

        let promoted = promote_body_foreground_over_hair(
            &body, &mut hair, &layers, &depths, &mapping, width, 1,
        );
        let hair = hair.to_rgba8();

        assert_eq!(promoted, 4);
        assert_eq!(hair.get_pixel(0, 0).0, [255, 0, 0, 255]);
        assert_eq!(hair.get_pixel(8, 0).0, [0, 0, 255, 255]);
        assert!(hair.get_pixel(3, 0)[0] > 0);
        assert!(hair.get_pixel(4, 0)[2] > 0);
    }

    #[test]
    fn original_eye_pixels_exclude_hair_that_is_locally_in_front() {
        let width = 3;
        let original = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            width,
            1,
            image::Rgba([10, 200, 240, 255]),
        ));
        let eye_mask = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            width,
            1,
            image::Rgba([255, 255, 255, 255]),
        ));
        let layers = HashMap::from([
            ("eyewhite".to_string(), eye_mask.clone()),
            (
                "front_hair".to_string(),
                DynamicImage::ImageRgba8(RgbaImage::from_pixel(
                    width,
                    1,
                    image::Rgba([20, 20, 30, 255]),
                )),
            ),
        ]);
        let mapping = HashMap::from([
            ("eyewhite".to_string(), "eye".to_string()),
            ("front_hair".to_string(), "hair".to_string()),
        ]);
        let depths = HashMap::from([
            (
                "eyewhite".to_string(),
                GrayImage::from_raw(width, 1, vec![10, 20, 10]).unwrap(),
            ),
            (
                "front_hair".to_string(),
                GrayImage::from_raw(width, 1, vec![20, 10, 20]).unwrap(),
            ),
        ]);

        let exact_eye = extract_original_target_pixels(
            &original,
            &eye_mask,
            &layers,
            &depths,
            &mapping,
            "eye",
            Some("hair"),
            width,
            1,
        )
        .unwrap()
        .to_rgba8();

        assert_eq!(exact_eye.get_pixel(0, 0).0, [10, 200, 240, 255]);
        assert_eq!(exact_eye.get_pixel(1, 0)[3], 0);
        assert_eq!(exact_eye.get_pixel(2, 0).0, [10, 200, 240, 255]);
    }
}
