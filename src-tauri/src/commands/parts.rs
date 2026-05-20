use crate::error::AppError;
use crate::inference::neck_extract;
use crate::inference::rife::rife_interpolate;
use crate::inference::session::{create_session, resolve_model_path};
use crate::processing::composite::premultiply_onto_body;
use crate::processing::image_utils;
use crate::state::AppState;
use base64::{engine::general_purpose::STANDARD, Engine};
use image::DynamicImage;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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

/// Default display order for body layers (top=front in UI).
/// This order is reversed when compositing (first=back, last=front).
const BODY_LAYER_ORDER: &[&str] = &[
    "nose", "face", "ears", "neck",
    "earwear", "eyewear", "neckwear",
    "topwear", "handwear", "bottomwear", "legwear", "footwear",
    "wings", "tail", "objects",
];

const EYE_LAYER_ORDER: &[&str] = &["eyewhite", "irides", "eyelash", "eyebrow"];

/// Layers that may have -l/-r variants.
const LR_SPLIT_LAYERS: &[&str] = &[
    "eyebrow", "eyelash", "irides", "eyewhite", "ears", "handwear",
];

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
    pub preview: String,           // merged preview base64 PNG
    pub layer_names: Vec<String>,
    pub layers: Vec<LayerInfo>,    // individual layer thumbnails for toggle UI
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

// ── Commands ─────────────────────────────────────────────────────────

/// Load a See-Through output (PSD file or folder of PNGs) into the current slot.
#[tauri::command]
pub async fn load_slot(
    app: AppHandle,
    path: String,
) -> Result<SlotLoadResult, AppError> {
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
    body_layer_order: Vec<String>,  // user's custom body layer order (top=front)
    body_layer_patches: Vec<LayerPatch>,
    hair_layer_order: Vec<String>,  // user's custom hair layer order (top=front)
    hair_back_layer_order: Vec<String>,  // user's custom hair_back layer order (top=front)
    output_path: String,
) -> Result<CreateBaseResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || {
        create_base_inner(app, mapping_json, original_image_path, base_eye_slot, base_mouth_slot, body_layer_order, body_layer_patches, hair_layer_order, hair_back_layer_order, output_path)
    })
    .await
    .map_err(|e| AppError::General(format!("Task join error: {}", e)))?
}

/// Create a diff: load PSD → extract eye or mouth → RIFE interpolate with base → export folder.
#[tauri::command]
pub async fn create_diff(
    app: AppHandle,
    path: String,
    diff_type: String,    // "eye" or "mouth"
    slot_name: String,    // e.g. "eye_closed", "mouth_a", "mouth_i", etc.
    frame_count: u32,
    output_path: String,
    original_image_path: String,  // 元画像パス（mouth SAM3マスク適用用）
) -> Result<CreateDiffResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || {
        create_diff_inner(app, path, diff_type, slot_name, frame_count, output_path, original_image_path)
    })
    .await
    .map_err(|e| AppError::General(format!("Task join error: {}", e)))?
}

/// Get a composite preview using current base parts.
#[tauri::command]
pub async fn get_base_preview(
    app: AppHandle,
) -> Result<String, AppError> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        let parts = state.parts.lock().unwrap();
        let w = *state.canvas_width.lock().unwrap();
        let h = *state.canvas_height.lock().unwrap();
        Ok(generate_composite_preview(&parts, w, h))
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
    tauri::async_runtime::spawn_blocking(move || {
        get_mapping_preview_inner(app, mapping_json)
    })
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
        render_category_inner(app, mapping_json, target, enabled_layers, layer_patches, layer_opacities, overlap_highlight)
    })
    .await
    .map_err(|e| AppError::General(format!("Task join error: {}", e)))?
}

/// Get all loaded PSD layers for See-Through correction mode.
#[tauri::command]
pub async fn get_all_layers_preview(
    app: AppHandle,
) -> Result<MappingPreviewResult, AppError> {
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
        export_corrected_layer_inner(app, output_path, enabled_layers, layer_patches, layer_opacities)
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
        let original = state.cached_mouth_originals.lock().unwrap().get(&path).cloned()
            .ok_or_else(|| AppError::General("元画像が読み込まれていません。先に口マスク確認を実行してください".into()))?;
        let raw_mask = state.cached_mouth_raw_masks.lock().unwrap().get(&path).cloned()
            .ok_or_else(|| AppError::General("SAM3口マスクがまだ作成されていません。先に口マスク確認を実行してください".into()))?;
        let mask = neck_extract::adjust_mask(
            &raw_mask,
            original.width(),
            original.height(),
            mouth_mask_dilate_radius,
            mouth_mask_blur_radius,
        );
        let masked = neck_extract::apply_mask_to_image(&original, &mask, original.width(), original.height());
        let preview = mouth_preview_to_base64(&masked, &mask, original.width(), original.height());
        *state.cached_original.lock().unwrap() = Some(original);
        *state.cached_mouth_raw_mask.lock().unwrap() = Some(raw_mask);
        *state.cached_mouth_mask.lock().unwrap() = Some(mask);
        Ok(MouthMaskPreviewResult { mouth_preview: preview })
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
        return frames.iter().map(image_utils::image_to_base64_png).collect();
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
    let resized = image.resize_exact(resized_w, resized_h, image::imageops::FilterType::Lanczos3).to_rgba8();
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

fn load_slot_inner(
    app: AppHandle,
    path: String,
) -> Result<SlotLoadResult, AppError> {
    let p = Path::new(&path);
    if !p.is_file() {
        return Err(AppError::General(format!("ファイルが見つかりません: {}", path)));
    }
    let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
    if ext.to_lowercase() != "psd" {
        return Err(AppError::General("PSD形式のファイルを選択してください".into()));
    }
    let layers = load_layers_from_psd(&path)?;
    let source_type = "psd".to_string();

    if layers.is_empty() {
        return Err(AppError::General("レイヤーが1つも見つかりませんでした".into()));
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
        resized_layers.insert(name.clone(), img);
    }

    // Detect adjustable layers
    let mut adjustable_layers = Vec::new();
    let detected_layers: Vec<String> = resized_layers.keys().cloned().collect();
    let mut seen_adjustable: Vec<String> = Vec::new();

    for &(layer_name, default_target) in ADJUSTABLE_DEFAULTS {
        let img = resized_layers.get(layer_name)
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
        if layer_name.starts_with('_') { continue; } // skip internal layers
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

    eprintln!(
        "[PachiPakuGen] Loaded from {} ({} layers, canvas {}x{})",
        source_type, detected_layers.len(), w, h
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
) -> Result<CreateBaseResult, AppError> {
    let state = app.state::<AppState>();

    // Build full mapping
    let user_mapping: HashMap<String, String> = serde_json::from_str(&mapping_json)
        .map_err(|e| AppError::General(format!("Invalid mapping JSON: {}", e)))?;
    let full_mapping = build_full_mapping(&user_mapping);

    // Store mapping for future diff operations
    *state.layer_mapping.lock().unwrap() = full_mapping.clone();

    let slot_layers = state.slot_layers.lock().unwrap();
    let current = slot_layers.get("current")
        .ok_or_else(|| AppError::General("PSDが読み込まれていません".into()))?;
    let w = *state.canvas_width.lock().unwrap();
    let h = *state.canvas_height.lock().unwrap();

    // eye: PSD layers directly
    let eye = merge_layers_for_target(current, &full_mapping, "eye", w, h);

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
            merge_layers_for_target(current, &full_mapping, "mouth", w, h)
        }
    };

    let mut parts: HashMap<String, DynamicImage> = HashMap::new();
    let mut file_count = 0u32;

    let is_base_export = !output_path.is_empty();

    if is_base_export {
        // === 素体モード: body/hair/hair_back を出力 ===
        // hair: merge layers using user's custom order
        let hair = if !hair_layer_order.is_empty() {
            let mut order_reversed = hair_layer_order.clone();
            order_reversed.reverse();
            let mut result = image::RgbaImage::new(w, h);
            for layer_name in &order_reversed {
                let candidates = [
                    layer_name.clone(),
                    format!("{}-l", layer_name), format!("{}-r", layer_name),
                    format!("{}_l", layer_name), format!("{}_r", layer_name),
                ];
                for candidate in &candidates {
                    if let Some(img) = current.get(candidate.as_str()) {
                        alpha_composite_onto(&mut result, &img.to_rgba8(), w, h);
                    }
                }
            }
            Some(DynamicImage::ImageRgba8(result))
        } else {
            merge_layers_for_target(current, &full_mapping, "hair", w, h)
        };
        // hair_back: merge layers using user's custom order
        let hair_back = if !hair_back_layer_order.is_empty() {
            let mut order_reversed = hair_back_layer_order.clone();
            order_reversed.reverse();
            let mut result = image::RgbaImage::new(w, h);
            for layer_name in &order_reversed {
                let candidates = [
                    layer_name.clone(),
                    format!("{}-l", layer_name), format!("{}-r", layer_name),
                    format!("{}_l", layer_name), format!("{}_r", layer_name),
                ];
                for candidate in &candidates {
                    if let Some(img) = current.get(candidate.as_str()) {
                        alpha_composite_onto(&mut result, &img.to_rgba8(), w, h);
                    }
                }
            }
            Some(DynamicImage::ImageRgba8(result))
        } else {
            merge_layers_for_target(current, &full_mapping, "hair_back", w, h)
        };

        // body: merge layers using user's custom order
        let body_img = if !body_layer_order.is_empty() {
            let mut order_reversed = body_layer_order.clone();
            order_reversed.reverse();
            let mut result = image::RgbaImage::new(w, h);
            let active_patches = active_patches_for_order(&body_layer_patches, &body_layer_order);
            let patch_masks = prepare_patch_masks(&active_patches, w, h)?;
            for layer_name in &order_reversed {
                composite_body_order_item(&mut result, current, layer_name, &active_patches, &patch_masks, None, w, h)?;
            }
            DynamicImage::ImageRgba8(result)
        } else {
            merge_layers_for_target(current, &full_mapping, "body", w, h)
                .ok_or_else(|| AppError::General("bodyに対応するレイヤーが見つかりません".into()))?
        };

        parts.insert("body".to_string(), body_img);
        if let Some(img) = hair { parts.insert("hair".to_string(), img); }
        if let Some(img) = hair_back { parts.insert("hair_back".to_string(), img); }

        // Export static layers
        let out_dir = Path::new(&output_path);
        fs::create_dir_all(out_dir)?;
        for key in &["body", "hair", "hair_back"] {
            if let Some(img) = parts.get(*key) {
                img.save(out_dir.join(format!("{}.png", key)))?;
                file_count += 1;
            }
        }

        eprintln!("[PachiPakuGen] Base body created ({}x{}), {}files", w, h, file_count);
    } else {
        // === フレーム補間モード: bodyはPSD合成でRIFE用に保持するだけ ===
        // Exclude neck from body (prevents neck bleeding into RIFE mouth frames)
        let body = merge_layers_for_target_excluding(
            current, &full_mapping, "body", &["neck"], w, h,
        ).unwrap_or_else(|| DynamicImage::new_rgba8(w, h));
        parts.insert("body".to_string(), body);

        eprintln!("[PachiPakuGen] Interp base loaded ({}x{}), body for premultiply (no neck)", w, h);
    }

    // eye/mouth
    if let Some(img) = eye { parts.insert(base_eye_slot.clone(), img); }
    if let Some(img) = mouth { parts.insert(base_mouth_slot.clone(), img); }

    let composite_preview = generate_composite_preview(&parts, w, h);

    // Store parts for future diff operations (keep base eye & mouth)
    drop(slot_layers);
    *state.parts.lock().unwrap() = parts;
    state.slot_layers.lock().unwrap().clear();

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
        return Err(AppError::General("フレーム数は2〜30の範囲で指定してください".into()));
    }

    let state = app.state::<AppState>();

    // Load the diff PSD
    let p = Path::new(&path);
    if !p.is_file() {
        return Err(AppError::General(format!("ファイルが見つかりません: {}", path)));
    }
    let layers = load_layers_from_psd(&path)?;

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

    // Extract the target (eye or mouth) from the diff PSD
    let target = match diff_type.as_str() {
        "eye" => "eye",
        "mouth" => "mouth",
        _ => return Err(AppError::General(format!("不正なdiff_type: {}", diff_type))),
    };

    // Extract target from diff PSD
    let diff_merged = if target == "mouth" {
        // Mouth: use cached SAM3 mask applied to diff's original image
        let cached_mask = state.cached_mouth_mask.lock().unwrap().clone();
        if let (Some(mask), true) = (cached_mask, !original_image_path.is_empty()) {
            // Load diff's original image and apply mouth mask
            let diff_orig = image::open(&original_image_path)
                .map_err(|e| AppError::General(format!("Diff元画像の読み込み失敗: {}", e)))?;
            let diff_orig = if diff_orig.width() != w || diff_orig.height() != h {
                fit_image_to_canvas(&diff_orig, w, h)
            } else {
                diff_orig
            };
            eprintln!("[PachiPakuGen] Diff mouth: SAM3 mask applied to original image");
            neck_extract::apply_mask_to_image(&diff_orig, &mask, w, h)
        } else {
            eprintln!("[PachiPakuGen] No cached mouth mask or original, using PSD layers");
            merge_layers_for_target(&resized, &mapping, target, w, h)
                .ok_or_else(|| AppError::General("mouthレイヤーが見つかりません".into()))?
        }
    } else {
        // Eye: PSD layers directly
        merge_layers_for_target(&resized, &mapping, target, w, h)
            .ok_or_else(|| AppError::General(format!(
                "{}に対応するレイヤーが見つかりません", target
            )))?
    };

    // Get base frame from stored parts
    let parts = state.parts.lock().unwrap();
    let base_key = if diff_type == "eye" {
        // Find which eye slot is the base (eye_open or eye_closed)
        parts.keys().find(|k| k.starts_with("eye_"))
            .cloned()
            .ok_or_else(|| AppError::General("素体のeyeが見つかりません".into()))?
    } else {
        parts.keys().find(|k| k.starts_with("mouth_") || *k == "mouth_closed")
            .cloned()
            .ok_or_else(|| AppError::General("素体のmouthが見つかりません".into()))?
    };

    let base_frame = parts.get(&base_key)
        .ok_or_else(|| AppError::General(format!("ベースフレーム '{}' が見つかりません", base_key)))?
        .clone();

    let body = parts.get("body")
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
        let _ = app.emit("generation-progress", ProgressPayload {
            current: (step + 1) as u32,
            total: frame_count,
            pair_name: pair_name.clone(),
        });

        let part_frame = if step == 0 {
            DynamicImage::ImageRgba8(img_a_rgba.clone())
        } else if step + 1 == frame_count as usize {
            DynamicImage::ImageRgba8(img_b_rgba.clone())
        } else {
            let interpolated = rife_interpolate(session, &rife_a, &rife_b, ratio)?;
            extract_part_with_blended_alpha(
                &interpolated, &img_a_rgba, &img_b_rgba, ratio, w, h,
            )
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
        pair_name, frame_count, out_dir.display()
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

fn get_mapping_preview_inner(
    app: AppHandle,
    mapping_json: String,
) -> Result<MappingPreviewResult, AppError> {
    let state = app.state::<AppState>();

    let user_mapping: HashMap<String, String> = serde_json::from_str(&mapping_json)
        .map_err(|e| AppError::General(format!("Invalid mapping JSON: {}", e)))?;
    let full_mapping = build_full_mapping(&user_mapping);

    let slot_layers = state.slot_layers.lock().unwrap();
    let current = slot_layers.get("current")
        .ok_or_else(|| AppError::General("PSD/フォルダが読み込まれていません".into()))?;
    let w = *state.canvas_width.lock().unwrap();
    let h = *state.canvas_height.lock().unwrap();

    let target_labels: &[(&str, &str)] = &[
        ("body", "Body (素体)"),
        ("eye", "Eye (目)"),
        ("mouth", "Mouth (口)"),
        ("hair", "Hair 前髪"),
        ("hair_back", "Hair 後髪"),
        ("skip", "スキップ"),
    ];

    let mut categories = Vec::new();

    for &(target, label) in target_labels {
        // Collect which layers map to this target, in BODY_LAYER_ORDER
        let mut layer_names: Vec<String> = Vec::new();

        // First: add in predefined order
        let order: &[&str] = match target {
            "body" => BODY_LAYER_ORDER,
            "eye" => EYE_LAYER_ORDER,
            _ => &[],
        };
        for &name in order {
            if !layer_names.contains(&name.to_string()) {
                // Check if this layer exists in current and maps to target
                let exists = current.keys().any(|k| normalize_layer_name(k) == name);
                if exists {
                    if let Some(mapped) = full_mapping.get(name) {
                        if mapped == target {
                            layer_names.push(name.to_string());
                        }
                    }
                }
            }
        }

        // Then: add remaining layers not in the order list
        for (layer_name, _) in current {
            let base = normalize_layer_name(layer_name);
            if let Some(mapped) = full_mapping.get(base) {
                if mapped == target && !layer_names.contains(&base.to_string()) {
                    layer_names.push(base.to_string());
                }
            }
        }

        if layer_names.is_empty() && target != "skip" {
            continue;
        }

        // Generate merged preview
        let preview = if target == "skip" {
            // For skip, just show a placeholder
            String::new()
        } else {
            match merge_layers_for_target(current, &full_mapping, target, w, h) {
                Some(img) => {
                    image_utils::image_to_base64_png(&img)
                }
                None => String::new(),
            }
        };

        // Generate individual layer thumbnails
        let mut layers_info = Vec::new();
        for base_name in &layer_names {
            // Find the actual image(s) for this base name (including L/R variants)
            let candidates = [
                base_name.clone(),
                format!("{}-l", base_name), format!("{}-r", base_name),
                format!("{}_l", base_name), format!("{}_r", base_name),
            ];
            for candidate in &candidates {
                if let Some(img) = current.get(candidate.as_str()) {
                    let thumb = img.thumbnail(120, 120);
                    layers_info.push(LayerInfo {
                        name: candidate.clone(),
                        thumbnail: image_utils::image_to_base64_png(&thumb),
                        bounds: alpha_bounds(&img.to_rgba8()).unwrap_or_default(),
                    });
                }
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
    for target in &["body", "eye", "mouth", "hair", "hair_back"] {
        if let Some(img) = merge_layers_for_target(current, &full_mapping, target, w, h) {
            // Map to expected keys for composite
            let key = match *target {
                "eye" => "eye_open",
                "mouth" => "mouth_closed",
                _ => target,
            };
            composite_parts.insert(key.to_string(), img);
        }
    }
    let composite_preview = generate_composite_preview(&composite_parts, w, h);

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
    let current = slot_layers.get("current")
        .ok_or_else(|| AppError::General("PSDが読み込まれていません".into()))?;
    let w = *state.canvas_width.lock().unwrap();
    let h = *state.canvas_height.lock().unwrap();

    // Composite in the exact order of enabled_layers (user-controlled order)
    let mut result_img = image::RgbaImage::new(w, h);
    let active_patches = active_patches_for_order(&layer_patches, &enabled_layers);
    let patch_masks = prepare_patch_masks(&active_patches, w, h)?;
    for layer_name in &enabled_layers {
        composite_body_order_item(&mut result_img, current, layer_name, &active_patches, &patch_masks, Some(&layer_opacities), w, h)?;
    }
    if overlap_highlight {
        apply_overlap_highlight(&mut result_img, current, &enabled_layers, &active_patches, &patch_masks, &layer_opacities, w, h)?;
    }

    let preview = image_utils::image_to_base64_png(&DynamicImage::ImageRgba8(result_img));
    Ok(RenderCategoryResult { preview })
}

fn get_all_layers_preview_inner(
    app: AppHandle,
) -> Result<MappingPreviewResult, AppError> {
    let state = app.state::<AppState>();

    let slot_layers = state.slot_layers.lock().unwrap();
    let current = slot_layers.get("current")
        .ok_or_else(|| AppError::General("PSD縺瑚ｪｭ縺ｿ霎ｼ縺ｾ繧後※縺・∪縺帙ｓ".into()))?;
    let w = *state.canvas_width.lock().unwrap();
    let h = *state.canvas_height.lock().unwrap();

    let mut layer_names: Vec<String> = current.keys().cloned().collect();
    layer_names.sort();

    let mut composite = image::RgbaImage::new(w, h);
    let mut layers = Vec::new();
    for name in &layer_names {
        if let Some(img) = current.get(name.as_str()) {
            let rgba = img.to_rgba8();
            alpha_composite_onto(&mut composite, &rgba, w, h);
            let thumb = img.thumbnail(120, 120);
            layers.push(LayerInfo {
                name: name.clone(),
                thumbnail: image_utils::image_to_base64_png(&thumb),
                bounds: alpha_bounds(&rgba).unwrap_or_default(),
            });
        }
    }

    let composite_preview = image_utils::image_to_base64_png(&DynamicImage::ImageRgba8(composite.clone()));
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
    let current = slot_layers.get_mut("current")
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
    eprintln!("[PachiPakuGen] Imported correction layer '{}' from {}", final_name, path);

    Ok(ImportCorrectionLayerResult { layer_name: final_name })
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
    let current = slot_layers.get("current")
        .ok_or_else(|| AppError::General("PSD縺瑚ｪｭ縺ｿ霎ｼ縺ｾ繧後※縺・∪縺帙ｓ".into()))?;
    let w = *state.canvas_width.lock().unwrap();
    let h = *state.canvas_height.lock().unwrap();

    let active_patches = active_patches_for_order(&layer_patches, &enabled_layers);
    let patch_masks = prepare_patch_masks(&active_patches, w, h)?;
    let mut result_img = image::RgbaImage::new(w, h);
    for layer_name in &enabled_layers {
        composite_body_order_item(
            &mut result_img,
            current,
            layer_name,
            &active_patches,
            &patch_masks,
            Some(&layer_opacities),
            w,
            h,
        )?;
    }

    let out_path = Path::new(&output_path);
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)?;
    }
    DynamicImage::ImageRgba8(result_img).save(out_path)?;

    Ok(ExportCorrectedLayerResult {
        output_path,
    })
}

fn normalize_layer_name(name: &str) -> &str {
    for &base in LR_SPLIT_LAYERS {
        if name == format!("{}-l", base) || name == format!("{}-r", base)
            || name == format!("{}_l", base) || name == format!("{}_r", base)
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

fn composite_body_order_item(
    result: &mut image::RgbaImage,
    current: &HashMap<String, DynamicImage>,
    layer_name: &str,
    patches: &[LayerPatch],
    patch_masks: &HashMap<String, image::GrayImage>,
    layer_opacities: Option<&HashMap<String, f32>>,
    width: u32,
    height: u32,
) -> Result<(), AppError> {
    let opacity = layer_opacities
        .and_then(|opacities| opacities.get(layer_name).copied())
        .unwrap_or(1.0)
        .clamp(0.0, 1.0);

    if let Some(patch) = patches.iter().find(|p| p.id == layer_name) {
        if let (Some(src), Some(mask)) = (current.get(patch.source_layer.as_str()), patch_masks.get(patch.id.as_str())) {
            let mut patch_img = apply_mask_to_rgba(&src.to_rgba8(), mask, false);
            apply_opacity(&mut patch_img, opacity);
            alpha_composite_onto(result, &patch_img, width, height);
        }
        return Ok(());
    }

    let candidates = [
        layer_name.to_string(),
        format!("{}-l", layer_name), format!("{}-r", layer_name),
        format!("{}_l", layer_name), format!("{}_r", layer_name),
    ];
    for candidate in &candidates {
        if let Some(img) = current.get(candidate.as_str()) {
            let mut rgba = img.to_rgba8();
            for patch in patches.iter().filter(|p| p.cut_source && p.source_layer == *candidate) {
                if let Some(mask) = patch_masks.get(patch.id.as_str()) {
                    subtract_mask_from_rgba(&mut rgba, mask);
                }
            }
            apply_opacity(&mut rgba, opacity);
            alpha_composite_onto(result, &rgba, width, height);
        }
    }

    Ok(())
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
        if let (Some(src), Some(mask)) = (current.get(patch.source_layer.as_str()), patch_masks.get(patch.id.as_str())) {
            return vec![apply_mask_to_rgba(&src.to_rgba8(), mask, false)];
        }
        return Vec::new();
    }

    let candidates = [
        layer_name.to_string(),
        format!("{}-l", layer_name), format!("{}-r", layer_name),
        format!("{}_l", layer_name), format!("{}_r", layer_name),
    ];
    let mut images = Vec::new();
    for candidate in &candidates {
        if let Some(img) = current.get(candidate.as_str()) {
            let mut rgba = img.to_rgba8();
            for patch in patches.iter().filter(|p| p.cut_source && p.source_layer == *candidate) {
                if let Some(mask) = patch_masks.get(patch.id.as_str()) {
                    subtract_mask_from_rgba(&mut rgba, mask);
                }
            }
            images.push(rgba);
        }
    }
    images
}

fn get_mapping_target<'a>(
    layer_name: &str,
    full_mapping: &'a HashMap<String, String>,
) -> Option<&'a str> {
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
    full
}

/// Extract part from RIFE output using blended alpha.
fn extract_part_with_blended_alpha(
    rife_output: &DynamicImage,
    img_a_rgba: &image::RgbaImage,
    img_b_rgba: &image::RgbaImage,
    ratio: f32,
    width: u32,
    height: u32,
) -> DynamicImage {
    let rgb = rife_output.to_rgb8();
    let mut result = image::RgbaImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let alpha_a = img_a_rgba.get_pixel(x, y)[3] as f32;
            let alpha_b = img_b_rgba.get_pixel(x, y)[3] as f32;
            let alpha_union = alpha_a.max(alpha_b);

            if alpha_union > 0.0 {
                let alpha_lerp = alpha_a * (1.0 - ratio) + alpha_b * ratio;
                let alpha = alpha_lerp.max(alpha_union * 0.5).min(alpha_union);
                let p = rgb.get_pixel(x, y);
                result.put_pixel(x, y, image::Rgba([
                    p[0], p[1], p[2],
                    alpha.clamp(0.0, 255.0) as u8,
                ]));
            }
        }
    }

    DynamicImage::ImageRgba8(result)
}

// ── Layer loading ────────────────────────────────────────────────────

fn load_layers_from_psd(path: &str) -> Result<HashMap<String, DynamicImage>, AppError> {
    let bytes = fs::read(path)?;
    let psd = psd::Psd::from_bytes(&bytes)
        .map_err(|e| AppError::General(format!("PSD読み込みエラー: {:?}", e)))?;

    let mut layers = HashMap::new();
    let doc_width = psd.width();
    let doc_height = psd.height();

    for layer in psd.layers() {
        let name = layer.name().to_lowercase().replace(' ', "_");
        if name.is_empty() { continue; }

        // psd crate's layer.rgba() returns FULL document-sized RGBA data
        // (doc_width * doc_height * 4 bytes), already positioned on the canvas.
        let rgba_data = layer.rgba();
        let expected_len = (doc_width * doc_height * 4) as usize;

        if rgba_data.len() != expected_len {
            eprintln!(
                "[PachiPakuGen] PSD layer '{}': unexpected rgba size {} (expected {}), skipping",
                name, rgba_data.len(), expected_len
            );
            continue;
        }

        // Check if layer has any non-transparent pixels
        let has_content = rgba_data.chunks(4).any(|px| px[3] > 0);
        if !has_content { continue; }

        let canvas = image::RgbaImage::from_raw(doc_width, doc_height, rgba_data)
            .ok_or_else(|| AppError::General(format!(
                "PSDレイヤー '{}' のRGBA変換に失敗", name
            )))?;

        layers.insert(name, DynamicImage::ImageRgba8(canvas));
    }
    Ok(layers)
}

// ── Layer merging ────────────────────────────────────────────────────

/// Merge layers for a target, EXCLUDING specific base layer names.
fn merge_layers_for_target_excluding(
    slot_layers: &HashMap<String, DynamicImage>,
    mapping: &HashMap<String, String>,
    target: &str,
    exclude: &[&str],
    width: u32,
    height: u32,
) -> Option<DynamicImage> {
    let order: &[&str] = match target {
        "body" => BODY_LAYER_ORDER,
        "eye" => EYE_LAYER_ORDER,
        _ => &[],
    };

    let mut ordered_layers: Vec<&DynamicImage> = Vec::new();
    let mut added_names: Vec<String> = Vec::new();

    for &base_name in order {
        if exclude.contains(&base_name) { continue; }
        if let Some(mapped_target) = get_mapping_target(base_name, mapping) {
            if mapped_target == target {
                let candidates = [
                    base_name.to_string(),
                    format!("{}-l", base_name), format!("{}-r", base_name),
                    format!("{}_l", base_name), format!("{}_r", base_name),
                ];
                for candidate in &candidates {
                    if let Some(img) = slot_layers.get(candidate.as_str()) {
                        ordered_layers.push(img);
                        added_names.push(candidate.clone());
                    }
                }
            }
        }
    }

    for (layer_name, img) in slot_layers {
        if added_names.contains(layer_name) { continue; }
        let base = normalize_layer_name(layer_name);
        if exclude.contains(&base) { continue; }
        if let Some(mapped_target) = get_mapping_target(layer_name, mapping) {
            if mapped_target == target {
                ordered_layers.push(img);
            }
        }
    }

    if ordered_layers.is_empty() { return None; }
    if ordered_layers.len() == 1 { return Some(ordered_layers[0].clone()); }

    let mut result = image::RgbaImage::new(width, height);
    for layer in &ordered_layers {
        let rgba = layer.to_rgba8();
        alpha_composite_onto(&mut result, &rgba, width, height);
    }
    Some(DynamicImage::ImageRgba8(result))
}

fn merge_layers_for_target(
    slot_layers: &HashMap<String, DynamicImage>,
    mapping: &HashMap<String, String>,
    target: &str,
    width: u32,
    height: u32,
) -> Option<DynamicImage> {
    let order: &[&str] = match target {
        "body" => BODY_LAYER_ORDER,
        "eye" => EYE_LAYER_ORDER,
        _ => &[],
    };

    let mut ordered_layers: Vec<&DynamicImage> = Vec::new();
    let mut added_names: Vec<String> = Vec::new();

    // Add layers from predefined order (including L/R variants)
    for &base_name in order {
        if let Some(mapped_target) = get_mapping_target(base_name, mapping) {
            if mapped_target == target {
                let candidates = [
                    base_name.to_string(),
                    format!("{}-l", base_name), format!("{}-r", base_name),
                    format!("{}_l", base_name), format!("{}_r", base_name),
                ];
                for candidate in &candidates {
                    if let Some(img) = slot_layers.get(candidate.as_str()) {
                        ordered_layers.push(img);
                        added_names.push(candidate.clone());
                    }
                }
            }
        }
    }

    // Add remaining layers mapped to this target
    for (layer_name, img) in slot_layers {
        if added_names.contains(layer_name) { continue; }
        if let Some(mapped_target) = get_mapping_target(layer_name, mapping) {
            if mapped_target == target {
                ordered_layers.push(img);
            }
        }
    }

    if ordered_layers.is_empty() { return None; }
    if ordered_layers.len() == 1 { return Some(ordered_layers[0].clone()); }

    let mut result = image::RgbaImage::new(width, height);
    for layer in &ordered_layers {
        let rgba = layer.to_rgba8();
        alpha_composite_onto(&mut result, &rgba, width, height);
    }
    Some(DynamicImage::ImageRgba8(result))
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
                    dst.put_pixel(x, y, image::Rgba([
                        r.clamp(0.0, 255.0) as u8,
                        g.clamp(0.0, 255.0) as u8,
                        b.clamp(0.0, 255.0) as u8,
                        (out_a * 255.0).clamp(0.0, 255.0) as u8,
                    ]));
                }
            }
        }
    }
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

fn apply_mask_to_rgba(src: &image::RgbaImage, mask: &image::GrayImage, invert: bool) -> image::RgbaImage {
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
) -> String {
    let mut result = image::RgbaImage::new(width, height);
    // Layer order: hair_back → body → eye → mouth → hair
    for key in &["hair_back", "body"] {
        if let Some(img) = parts.get(*key) {
            alpha_composite_onto(&mut result, &img.to_rgba8(), width, height);
        }
    }
    // Find first eye and mouth
    for (k, img) in parts {
        if k.starts_with("eye_") {
            alpha_composite_onto(&mut result, &img.to_rgba8(), width, height);
            break;
        }
    }
    for (k, img) in parts {
        if k.starts_with("mouth_") || k == "mouth_closed" {
            alpha_composite_onto(&mut result, &img.to_rgba8(), width, height);
            break;
        }
    }
    if let Some(img) = parts.get("hair") {
        alpha_composite_onto(&mut result, &img.to_rgba8(), width, height);
    }
    let composite = DynamicImage::ImageRgba8(result);
    image_utils::image_to_base64_jpeg(&composite, 80)
}
