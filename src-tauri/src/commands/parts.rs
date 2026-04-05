use crate::error::AppError;
use crate::inference::neck_extract;
use crate::inference::rife::rife_interpolate;
use crate::inference::session::{create_session, resolve_model_path};
use crate::processing::composite::premultiply_onto_body;
use crate::processing::image_utils;
use crate::state::AppState;
use image::DynamicImage;
use serde::Serialize;
use std::collections::HashMap;
use std::fs;
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
}

#[derive(Serialize)]
pub struct LayerInfo {
    pub name: String,        // layer name (e.g. "face", "irides")
    pub thumbnail: String,   // base64 PNG thumbnail
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
    hair_layer_order: Vec<String>,  // user's custom hair layer order (top=front)
    hair_back_layer_order: Vec<String>,  // user's custom hair_back layer order (top=front)
    output_path: String,
) -> Result<CreateBaseResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || {
        create_base_inner(app, mapping_json, original_image_path, base_eye_slot, base_mouth_slot, body_layer_order, hair_layer_order, hair_back_layer_order, output_path)
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
) -> Result<RenderCategoryResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || {
        render_category_inner(app, mapping_json, target, enabled_layers)
    })
    .await
    .map_err(|e| AppError::General(format!("Task join error: {}", e)))?
}


/// Load original image and extract neck via SAM3. Caches the result.
#[tauri::command]
pub async fn load_original_image(
    app: AppHandle,
    path: String,
) -> Result<String, AppError> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        let original = image::open(&path)?;
        let w = *state.canvas_width.lock().unwrap();
        let h = *state.canvas_height.lock().unwrap();
        let original = if w > 0 && h > 0 && (original.width() != w || original.height() != h) {
            original.resize_exact(w, h, image::imageops::FilterType::Lanczos3)
        } else {
            original
        };
        let original_rgba = original.to_rgba8();

        // Cache original image for later use (mouth extraction applies mask to this)
        *state.cached_original.lock().unwrap() = Some(original.clone());

        // Run SAM3 to extract neck
        let sam3_ckpt = neck_extract::find_sam3_checkpoint();
        let neck_img = match neck_extract::extract_neck_mask(&original, sam3_ckpt.as_deref()) {
            Ok(mask) => {
                let ow = original.width();
                let oh = original.height();
                let mut neck_rgba = image::RgbaImage::new(ow, oh);
                for y in 0..oh {
                    for x in 0..ow {
                        let mask_val = mask[(y * ow + x) as usize];
                        if mask_val > 128 {
                            let orig = original_rgba.get_pixel(x, y);
                            neck_rgba.put_pixel(x, y, *orig);
                        }
                    }
                }
                DynamicImage::ImageRgba8(neck_rgba)
            }
            Err(e) => {
                eprintln!("[PachiPakuGen] SAM3 neck failed: {}", e);
                return Err(e);
            }
        };

        // Cache neck and store in slot_layers for preview
        let preview = image_utils::image_to_base64_png(&neck_img);
        *state.cached_neck.lock().unwrap() = Some(neck_img.clone());

        // Also add to slot_layers["current"] so the layer sidebar shows the SAM3 neck
        if let Some(current) = state.slot_layers.lock().unwrap().get_mut("current") {
            current.insert("neck".to_string(), neck_img);
        }

        eprintln!("[PachiPakuGen] Neck extracted and cached via SAM3");

        // Also extract and cache mouth mask from original image
        // (original image has clear mouth features even when closed, unlike PSD composite)
        match neck_extract::extract_mouth_mask(&original, sam3_ckpt.as_deref()) {
            Ok(mask) => {
                eprintln!("[PachiPakuGen] Mouth mask cached from original image via SAM3");
                *state.cached_mouth_mask.lock().unwrap() = Some(mask);
            }
            Err(e) => {
                eprintln!("[PachiPakuGen] SAM3 mouth from original failed: {}", e);
            }
        }

        Ok(preview)
    })
    .await
    .map_err(|e| AppError::General(format!("Task join error: {}", e)))?
}

// ── Implementation ───────────────────────────────────────────────────

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
                base_orig.resize_exact(w, h, image::imageops::FilterType::Lanczos3)
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
                diff_orig.resize_exact(w, h, image::imageops::FilterType::Lanczos3)
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

    // Debug: save RIFE input frames
    let debug_dir = Path::new(&output_path).join(format!("{}_debug", slot_name));
    let _ = fs::create_dir_all(&debug_dir);
    let _ = DynamicImage::ImageRgba8(img_a_rgba.clone()).save(debug_dir.join("frame_a_base.png"));
    let _ = DynamicImage::ImageRgba8(img_b_rgba.clone()).save(debug_dir.join("frame_b_diff.png"));
    let _ = body.save(debug_dir.join("body_premultiply.png"));
    eprintln!("[PachiPakuGen] Debug frames saved to {}", debug_dir.display());

    // Premultiply onto body (body already excludes neck in interp mode)
    let rife_a = premultiply_onto_body(&body_rgb, &img_a_rgba, w, h);
    let rife_b = premultiply_onto_body(&body_rgb, &img_b_rgba, w, h);

    // Debug: save premultiplied frames
    let _ = rife_a.save(debug_dir.join("rife_a_premultiplied.png"));
    let _ = rife_b.save(debug_dir.join("rife_b_premultiplied.png"));

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

        let interpolated = rife_interpolate(session, &rife_a, &rife_b, ratio)?;
        let part_frame = extract_part_with_blended_alpha(
            &interpolated, &img_a_rgba, &img_b_rgba, ratio, w, h,
        );
        frames.push(part_frame);
    }
    drop(session_guard);

    // Export frames to folder (reversed: frame_001=closed, frame_N=open)
    frames.reverse();
    let out_dir = Path::new(&output_path).join(&pair_name);
    fs::create_dir_all(&out_dir)?;

    for (i, frame) in frames.iter().enumerate() {
        let filename = format!("frame_{:03}.png", i + 1);
        frame.save(out_dir.join(&filename))?;
    }

    // Preview: last frame
    let preview = image_utils::image_to_base64_png(frames.last().unwrap());

    eprintln!(
        "[PachiPakuGen] Diff created: {} ({} frames) → {}",
        pair_name, frame_count, out_dir.display()
    );

    Ok(CreateDiffResult {
        output_path: out_dir.to_string_lossy().into_owned(),
        pair_name,
        frame_count,
        preview,
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
) -> Result<RenderCategoryResult, AppError> {
    let state = app.state::<AppState>();

    let slot_layers = state.slot_layers.lock().unwrap();
    let current = slot_layers.get("current")
        .ok_or_else(|| AppError::General("PSDが読み込まれていません".into()))?;
    let w = *state.canvas_width.lock().unwrap();
    let h = *state.canvas_height.lock().unwrap();

    // Composite in the exact order of enabled_layers (user-controlled order)
    let mut result_img = image::RgbaImage::new(w, h);
    for layer_name in &enabled_layers {
        if let Some(img) = current.get(layer_name.as_str()) {
            alpha_composite_onto(&mut result_img, &img.to_rgba8(), w, h);
        }
    }

    let preview = image_utils::image_to_base64_png(&DynamicImage::ImageRgba8(result_img));
    Ok(RenderCategoryResult { preview })
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
