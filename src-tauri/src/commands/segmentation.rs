use base64::{engine::general_purpose::STANDARD, Engine};
use crate::error::AppError;
use crate::inference::sam3_python;
use crate::inference::segmentation::{
    apply_mask_to_rgba, create_overlay, extract_masks, run_segmentation,
};
use crate::inference::session::{create_session, resolve_model_path};
use crate::processing::image_utils;
use crate::state::{AppState, PartPair};
use image::DynamicImage;
use serde::Serialize;
use std::collections::HashSet;
use tauri::{AppHandle, Manager};

#[derive(Serialize)]
pub struct MaskInfo {
    pub eye_pixels: u32,
    pub mouth_pixels: u32,
    pub eyebrow_pixels: u32,
    pub has_eye: bool,
    pub has_mouth: bool,
    pub has_eyebrow: bool,
    pub overlay_preview: String,
}

#[derive(Serialize)]
pub struct ModelAvailability {
    pub unet: bool,
    pub sam3: bool,
}

/// Check which segmentation models are available.
#[tauri::command]
pub fn check_model_availability(app: AppHandle) -> ModelAvailability {
    let unet = resolve_model_path(&app, "unet_segmentation.onnx").is_ok();
    let sam3_path = sam3_python::resolve_sam3_checkpoint();
    let sam3 = sam3_path.is_some();
    eprintln!(
        "[PachiPakuGen] Model availability: unet={}, sam3={} (path={:?})",
        unet, sam3, sam3_path
    );
    ModelAvailability { unet, sam3 }
}

/// Run segmentation on the loaded eye_open image (composited onto white).
/// Uses either SAM3 (Python subprocess) or UNet (ONNX) depending on `model` param.
/// Targets: comma-separated "eye,mouth" or "eye,mouth,eyebrow" (eyebrow only for SAM3).
#[tauri::command]
pub async fn run_segmentation_cmd(
    app: AppHandle,
    model: String,
    targets: String,
) -> Result<MaskInfo, AppError> {
    tauri::async_runtime::spawn_blocking(move || {
        run_segmentation_inner(app, model, targets)
    })
    .await
    .map_err(|e| AppError::General(format!("Task join error: {}", e)))?
}

/// Helper to combine two Option<Vec<u8>> masks with per-pixel max.
fn combine_masks(a: Option<Vec<u8>>, b: Option<Vec<u8>>) -> Option<Vec<u8>> {
    match (a, b) {
        (Some(va), Some(vb)) => Some(va.iter().zip(vb.iter()).map(|(&x, &y)| x.max(y)).collect()),
        (Some(v), None) | (None, Some(v)) => Some(v),
        (None, None) => None,
    }
}

/// Merge all part masks into a single combined mask using per-pixel max.
fn combine_all_masks(
    eye: Option<&[u8]>,
    mouth: Option<&[u8]>,
    eyebrow: Option<&[u8]>,
    size: usize,
) -> Vec<u8> {
    let mut combined = vec![0u8; size];
    if let Some(e) = eye {
        for i in 0..size { combined[i] = combined[i].max(e[i]); }
    }
    if let Some(m) = mouth {
        for i in 0..size { combined[i] = combined[i].max(m[i]); }
    }
    if let Some(eb) = eyebrow {
        for i in 0..size { combined[i] = combined[i].max(eb[i]); }
    }
    combined
}

fn run_segmentation_inner(
    app: AppHandle,
    model: String,
    targets: String,
) -> Result<MaskInfo, AppError> {
    let state = app.state::<AppState>();

    // Parse enabled targets
    let enabled: HashSet<&str> = targets.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
    let want_eye = enabled.contains("eye");
    let want_mouth = enabled.contains("mouth");
    let want_eyebrow = enabled.contains("eyebrow");

    if !want_eye && !want_mouth && !want_eyebrow {
        return Err(AppError::General("少なくとも1つのターゲットを選択してください。".into()));
    }

    let raw_images = state.raw_images.lock().unwrap();
    let w = *state.canvas_width.lock().unwrap();
    let h = *state.canvas_height.lock().unwrap();

    if w == 0 || h == 0 {
        return Err(AppError::General("パーツが読み込まれていません。先にStep 1でパーツを読み込んでください。".into()));
    }

    // Use eye_open image as the primary segmentation source (has both eye and mouth visible).
    // Composite onto white for segmentation (transparent PNGs need solid background).
    let seg_source = raw_images.get("eye_open")
        .or_else(|| raw_images.get("mouth_closed"))
        .ok_or_else(|| AppError::General("目開き画像か口閉じ画像が必要です".into()))?;

    let seg_rgb = composite_onto_white(seg_source, w, h);

    // Run segmentation based on model choice
    let (eye_mask, mouth_mask, eyebrow_mask): (Option<Vec<u8>>, Option<Vec<u8>>, Option<Vec<u8>>) =
        if model == "sam3" || model == "sam" {
            // SAM3 via Python subprocess
            let checkpoint = sam3_python::resolve_sam3_checkpoint()
                .ok_or_else(|| AppError::General(
                    "SAM3チェックポイント(sam3.pt)が見つかりません。\n\
                     models/ フォルダに sam3.pt を配置してください。"
                        .into(),
                ))?;
            let script = sam3_python::resolve_script_path()?;

            eprintln!("[PachiPakuGen] Running SAM3: targets={}", targets);

            let (eye_a, mouth_a, eb_a) = sam3_python::run_sam3_via_python(
                &seg_rgb, &checkpoint, &script, &targets,
            )?;

            // Optionally also run on a second image and merge
            // For PachiPakuGen, the input images are very similar (same face, different expression)
            // so running on eye_open alone should be sufficient.
            // But if mouth_closed exists and is different, combine:
            let (eye_mask, mouth_mask, eyebrow_mask) = if let Some(mouth_src) = raw_images.get("mouth_closed") {
                let mouth_rgb = composite_onto_white(mouth_src, w, h);
                let (eye_b, mouth_b, eb_b) = sam3_python::run_sam3_via_python(
                    &mouth_rgb, &checkpoint, &script, &targets,
                )?;
                (combine_masks(eye_a, eye_b), combine_masks(mouth_a, mouth_b), combine_masks(eb_a, eb_b))
            } else {
                (eye_a, mouth_a, eb_a)
            };

            // Filter by what was requested
            (
                if want_eye { eye_mask } else { None },
                if want_mouth { mouth_mask } else { None },
                if want_eyebrow { eyebrow_mask } else { None },
            )
        } else {
            // UNet path (default)
            {
                let mut session = state.seg_session.lock().unwrap();
                if session.is_none() {
                    let model_path = resolve_model_path(&app, "unet_segmentation.onnx")?;
                    *session = Some(create_session(&model_path)?);
                }
            }

            let mut session = state.seg_session.lock().unwrap();
            let session = session.as_mut().unwrap();

            let seg_map = run_segmentation(session, &seg_rgb)?;
            let (eye_raw, mouth_raw) = extract_masks(&seg_map, w, h);

            let eye_mask = if want_eye { Some(eye_raw) } else { None };
            let mouth_mask = if want_mouth { Some(mouth_raw) } else { None };
            (eye_mask, mouth_mask, None) // UNet has no eyebrow class
        };

    drop(raw_images);

    let has_eye = eye_mask.is_some();
    let has_mouth = mouth_mask.is_some();
    let has_eyebrow = eyebrow_mask.is_some();

    // Count non-zero pixels
    let eye_pixels = eye_mask.as_ref().map(|m| m.iter().filter(|&&v| v > 0).count() as u32).unwrap_or(0);
    let mouth_pixels = mouth_mask.as_ref().map(|m| m.iter().filter(|&&v| v > 0).count() as u32).unwrap_or(0);
    let eyebrow_pixels = eyebrow_mask.as_ref().map(|m| m.iter().filter(|&&v| v > 0).count() as u32).unwrap_or(0);

    eprintln!(
        "[PachiPakuGen] Segmentation complete: eye={}px, mouth={}px, eyebrow={}px",
        eye_pixels, mouth_pixels, eyebrow_pixels
    );

    // Create overlay preview on the source image (white background)
    let raw_images = state.raw_images.lock().unwrap();
    let preview_base = raw_images.get("eye_open")
        .or_else(|| raw_images.get("mouth_closed"))
        .unwrap();
    let preview_rgb = composite_onto_white(preview_base, w, h);
    let overlay = create_overlay(&preview_rgb, eye_mask.as_deref(), mouth_mask.as_deref(), eyebrow_mask.as_deref(), w, h);
    let overlay_b64 = image_utils::image_to_base64_png(&overlay);
    drop(raw_images);

    // Store raw masks
    *state.raw_eye_mask.lock().unwrap() = eye_mask.clone();
    *state.raw_mouth_mask.lock().unwrap() = mouth_mask.clone();
    *state.raw_eyebrow_mask.lock().unwrap() = eyebrow_mask.clone();

    // Store adjusted masks (initially same as raw; brush editing updates these)
    *state.eye_mask.lock().unwrap() = eye_mask.clone();
    *state.mouth_mask.lock().unwrap() = mouth_mask.clone();
    *state.eyebrow_mask.lock().unwrap() = eyebrow_mask.clone();

    // Build combined mask
    let combined = combine_all_masks(eye_mask.as_deref(), mouth_mask.as_deref(), eyebrow_mask.as_deref(), (w * h) as usize);
    *state.combined_mask.lock().unwrap() = Some(combined);
    *state.mask_width.lock().unwrap() = w;
    *state.mask_height.lock().unwrap() = h;

    Ok(MaskInfo {
        eye_pixels,
        mouth_pixels,
        eyebrow_pixels,
        has_eye,
        has_mouth,
        has_eyebrow,
        overlay_preview: overlay_b64,
    })
}

/// Apply masks to separate raw images into eye-only and mouth-only parts,
/// then build part pairs for RIFE.
/// Call this after segmentation (and optional brush editing) is complete.
#[tauri::command]
pub async fn apply_masks_and_build_pairs(
    app: AppHandle,
) -> Result<ApplyMasksResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || {
        apply_masks_inner(app)
    })
    .await
    .map_err(|e| AppError::General(format!("Task join error: {}", e)))?
}

#[derive(Serialize)]
pub struct ApplyMasksResult {
    pub pair_count: usize,
    pub composite_preview: String,
}

fn apply_masks_inner(app: AppHandle) -> Result<ApplyMasksResult, AppError> {
    let state = app.state::<AppState>();
    let raw = state.raw_images.lock().unwrap();
    let w = *state.canvas_width.lock().unwrap();
    let h = *state.canvas_height.lock().unwrap();

    let eye_mask = state.eye_mask.lock().unwrap();
    let mouth_mask = state.mouth_mask.lock().unwrap();
    let eyebrow_mask = state.eyebrow_mask.lock().unwrap();

    let eye_detected = eye_mask.as_ref().map(|m| m.iter().any(|&v| v > 0)).unwrap_or(false);
    let mouth_detected = mouth_mask.as_ref().map(|m| m.iter().any(|&v| v > 0)).unwrap_or(false);
    let eyebrow_detected = eyebrow_mask.as_ref().map(|m| m.iter().any(|&v| v > 0)).unwrap_or(false);

    // Build combined eye+eyebrow mask for eye parts extraction.
    // Eyebrows are part of the eye region (they move with eye open/close),
    // so we merge them into the eye mask.
    let eye_combined_mask: Option<Vec<u8>> = if eye_detected || eyebrow_detected {
        let size = (w * h) as usize;
        let mut combined = vec![0u8; size];
        if let Some(em) = eye_mask.as_ref() {
            for i in 0..size { combined[i] = combined[i].max(em[i]); }
        }
        if let Some(ebm) = eyebrow_mask.as_ref() {
            for i in 0..size { combined[i] = combined[i].max(ebm[i]); }
        }
        Some(combined)
    } else {
        None
    };
    let eye_combined_detected = eye_combined_mask.as_ref().map(|m| m.iter().any(|&v| v > 0)).unwrap_or(false);

    eprintln!(
        "[PachiPakuGen] apply_masks: eye={}, mouth={}, eyebrow={}, eye+eyebrow={}",
        eye_detected, mouth_detected, eyebrow_detected, eye_combined_detected
    );

    // Separate parts using masks
    let mut separated = std::collections::HashMap::new();

    // body and hair pass through unchanged
    if let Some(img) = raw.get("body") {
        separated.insert("body".to_string(), img.clone());
    }
    if let Some(img) = raw.get("hair") {
        separated.insert("hair".to_string(), img.clone());
    }

    // Eye parts: extract eye + eyebrow region (eyebrows move with eye open/close)
    for key in &["eye_open", "eye_closed"] {
        if let Some(img) = raw.get(*key) {
            if eye_combined_detected {
                let eye_only = apply_mask_to_rgba(img, eye_combined_mask.as_ref().unwrap(), w, h);
                separated.insert(key.to_string(), eye_only);
            } else {
                separated.insert(key.to_string(), img.clone());
            }
        }
    }

    // Mouth parts: for each pair (mouth_closed + mouth_{vowel}),
    // build a per-pair mask = segmentation_mask ∩ (alpha_closed ∪ alpha_vowel).
    // This ensures each pair's mask covers both the closed and open mouth shapes.
    //
    // First, store mouth_closed separately (needed for every pair).
    let mouth_closed_img = raw.get("mouth_closed").cloned();

    // Always insert mouth_closed with just the seg mask (for composite preview)
    if let Some(ref img) = mouth_closed_img {
        if mouth_detected {
            let masked = apply_mask_to_rgba(img, mouth_mask.as_ref().unwrap(), w, h);
            separated.insert("mouth_closed".to_string(), masked);
        } else {
            separated.insert("mouth_closed".to_string(), img.clone());
        }
    }

    drop(eye_mask);
    drop(mouth_mask);
    drop(eyebrow_mask);

    // Build part pairs
    let mut pairs = Vec::new();

    if separated.contains_key("eye_open") && separated.contains_key("eye_closed") {
        pairs.push(PartPair {
            name: "eye".to_string(),
            image_a: separated["eye_open"].clone(),
            image_b: separated["eye_closed"].clone(),
        });
    }

    // For each mouth vowel, create a per-pair mask and masked images
    let mouth_vowels = ["a", "i", "u", "e", "o"];
    let seg_mouth_mask = state.mouth_mask.lock().unwrap();
    let size = (w * h) as usize;

    for vowel in &mouth_vowels {
        let vowel_key = format!("mouth_{}", vowel);
        let vowel_img = match raw.get(&vowel_key) {
            Some(img) => img,
            None => continue,
        };
        let closed_img = match mouth_closed_img.as_ref() {
            Some(img) => img,
            None => continue,
        };

        if mouth_detected {
            let seg_mask = seg_mouth_mask.as_ref().unwrap();

            // Build per-pair mask: segmentation_mask AND (alpha_closed OR alpha_vowel)
            // This covers the full mouth shape for both endpoints of the pair.
            let closed_rgba = closed_img.to_rgba8();
            let vowel_rgba = vowel_img.to_rgba8();
            let mut pair_mask = vec![0u8; size];
            for i in 0..size {
                let y = (i / w as usize) as u32;
                let x = (i % w as usize) as u32;
                let alpha_closed = closed_rgba.get_pixel(x, y)[3];
                let alpha_vowel = vowel_rgba.get_pixel(x, y)[3];
                let alpha_union = alpha_closed.max(alpha_vowel);
                // Intersection with segmentation mask, use min
                pair_mask[i] = seg_mask[i].min(alpha_union);
            }

            let masked_closed = apply_mask_to_rgba(closed_img, &pair_mask, w, h);
            let masked_vowel = apply_mask_to_rgba(vowel_img, &pair_mask, w, h);

            eprintln!(
                "[PachiPakuGen] mouth pair '{}': pair_mask non-zero={}px",
                vowel_key,
                pair_mask.iter().filter(|&&v| v > 0).count()
            );

            pairs.push(PartPair {
                name: vowel_key.clone(),
                image_a: masked_closed,
                image_b: masked_vowel,
            });
            // Also store for composite preview
            separated.insert(vowel_key, apply_mask_to_rgba(vowel_img, seg_mask, w, h));
        } else {
            // No segmentation mask — use raw images
            pairs.push(PartPair {
                name: vowel_key.clone(),
                image_a: closed_img.clone(),
                image_b: vowel_img.clone(),
            });
            separated.insert(vowel_key, vowel_img.clone());
        }
    }
    drop(seg_mouth_mask);

    let pair_count = pairs.len();

    // Generate composite preview
    let composite = build_composite_preview(&separated, w, h);
    let composite_preview = image_utils::image_to_base64_jpeg(&composite, 80);

    // Store
    *state.parts.lock().unwrap() = separated;
    *state.part_pairs.lock().unwrap() = pairs;
    state.generated.lock().unwrap().clear();

    drop(raw);

    Ok(ApplyMasksResult {
        pair_count,
        composite_preview,
    })
}

/// Composite a transparent image onto a white background for segmentation.
fn composite_onto_white(
    image: &DynamicImage,
    width: u32,
    height: u32,
) -> DynamicImage {
    let rgba = image.to_rgba8();
    let mut result = image::RgbImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let p = rgba.get_pixel(x, y);
            let a = p[3] as f32 / 255.0;
            let r = p[0] as f32 * a + 255.0 * (1.0 - a);
            let g = p[1] as f32 * a + 255.0 * (1.0 - a);
            let b = p[2] as f32 * a + 255.0 * (1.0 - a);
            result.put_pixel(
                x, y,
                image::Rgb([
                    r.clamp(0.0, 255.0) as u8,
                    g.clamp(0.0, 255.0) as u8,
                    b.clamp(0.0, 255.0) as u8,
                ]),
            );
        }
    }

    DynamicImage::ImageRgb8(result)
}

/// Build a composite preview from separated parts: body -> eye_open -> mouth_closed -> hair
fn build_composite_preview(
    parts: &std::collections::HashMap<String, DynamicImage>,
    width: u32,
    height: u32,
) -> DynamicImage {
    let mut result = image::RgbaImage::new(width, height);

    let layer_order = ["body", "eye_open", "mouth_closed", "hair"];

    for key in &layer_order {
        if let Some(img) = parts.get(*key) {
            let rgba = img.to_rgba8();
            for y in 0..height {
                for x in 0..width {
                    let sp = rgba.get_pixel(x, y);
                    let sa = sp[3] as f32 / 255.0;
                    if sa > 0.0 {
                        let dp = result.get_pixel(x, y);
                        let r = dp[0] as f32 * (1.0 - sa) + sp[0] as f32 * sa;
                        let g = dp[1] as f32 * (1.0 - sa) + sp[1] as f32 * sa;
                        let b = dp[2] as f32 * (1.0 - sa) + sp[2] as f32 * sa;
                        let a = dp[3] as f32 * (1.0 - sa) + 255.0 * sa;
                        result.put_pixel(x, y, image::Rgba([
                            r.clamp(0.0, 255.0) as u8,
                            g.clamp(0.0, 255.0) as u8,
                            b.clamp(0.0, 255.0) as u8,
                            a.clamp(0.0, 255.0) as u8,
                        ]));
                    }
                }
            }
        }
    }

    DynamicImage::ImageRgba8(result)
}

// ====== Brush editing commands ======

#[derive(Serialize)]
pub struct BlendResult {
    pub blend_image: String,
    pub width: u32,
    pub height: u32,
}

/// Return a background image for the brush canvas, with all relevant part images
/// composited semi-transparently onto white so the user can see where to paint.
///
/// - mouth target: overlay mouth_closed + all mouth vowels at 50% opacity
/// - eye target: overlay eye_open + eye_closed at 50% opacity
/// - eyebrow target: overlay eye_open at full opacity (eyebrows are on eye layer)
#[tauri::command]
pub async fn get_brush_background_cmd(app: AppHandle, target: String) -> Result<BlendResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        let raw = state.raw_images.lock().unwrap();
        let w = *state.canvas_width.lock().unwrap();
        let h = *state.canvas_height.lock().unwrap();

        // Collect the images to overlay based on target
        let keys: Vec<&str> = match target.as_str() {
            "mouth" => vec!["mouth_closed", "mouth_a", "mouth_i", "mouth_u", "mouth_e", "mouth_o"],
            "eye" => vec!["eye_open", "eye_closed"],
            "eyebrow" => vec!["eye_open"],
            _ => vec!["eye_open"],
        };

        // Start with white background
        let mut result = image::RgbImage::from_pixel(w, h, image::Rgb([255, 255, 255]));

        // Composite each image semi-transparently (50% opacity) onto white.
        // Multiple overlapping images create a blended view of all shapes.
        let blend_alpha = if keys.len() > 1 { 0.5f32 } else { 0.8f32 };

        for key in &keys {
            if let Some(img) = raw.get(*key) {
                let rgba = img.to_rgba8();
                for y in 0..h {
                    for x in 0..w {
                        let p = rgba.get_pixel(x, y);
                        let src_a = p[3] as f32 / 255.0 * blend_alpha;
                        if src_a > 0.0 {
                            let dp = result.get_pixel(x, y);
                            let r = dp[0] as f32 * (1.0 - src_a) + p[0] as f32 * src_a;
                            let g = dp[1] as f32 * (1.0 - src_a) + p[1] as f32 * src_a;
                            let b = dp[2] as f32 * (1.0 - src_a) + p[2] as f32 * src_a;
                            result.put_pixel(x, y, image::Rgb([
                                r.clamp(0.0, 255.0) as u8,
                                g.clamp(0.0, 255.0) as u8,
                                b.clamp(0.0, 255.0) as u8,
                            ]));
                        }
                    }
                }
            }
        }

        let bg = DynamicImage::ImageRgb8(result);
        let bg_b64 = image_utils::image_to_base64_png(&bg);

        Ok(BlendResult { blend_image: bg_b64, width: w, height: h })
    })
    .await
    .map_err(|e| AppError::General(format!("Task join error: {}", e)))?
}

/// Composite a transparent image onto a dark checkerboard pattern.
/// Uses dark gray (#2a2a2a) and medium gray (#3a3a3a) for good contrast
/// with the colored mask overlays (blue/red/green).
#[allow(dead_code)]
fn composite_onto_checkerboard(
    image: &DynamicImage,
    width: u32,
    height: u32,
) -> DynamicImage {
    let rgba = image.to_rgba8();
    let mut result = image::RgbImage::new(width, height);
    let check_size = 8u32; // 8x8 pixel checker squares

    for y in 0..height {
        for x in 0..width {
            // Dark checkerboard: alternate between two dark grays
            let checker = ((x / check_size) + (y / check_size)) % 2 == 0;
            let (bg_r, bg_g, bg_b): (f32, f32, f32) = if checker {
                (42.0, 42.0, 42.0)  // #2a2a2a
            } else {
                (58.0, 58.0, 58.0)  // #3a3a3a
            };

            let p = rgba.get_pixel(x, y);
            let a = p[3] as f32 / 255.0;
            let r = p[0] as f32 * a + bg_r * (1.0 - a);
            let g = p[1] as f32 * a + bg_g * (1.0 - a);
            let b = p[2] as f32 * a + bg_b * (1.0 - a);
            result.put_pixel(
                x, y,
                image::Rgb([
                    r.clamp(0.0, 255.0) as u8,
                    g.clamp(0.0, 255.0) as u8,
                    b.clamp(0.0, 255.0) as u8,
                ]),
            );
        }
    }

    DynamicImage::ImageRgb8(result)
}

#[derive(Serialize)]
pub struct TargetMaskResult {
    pub mask_image: String,
    pub width: u32,
    pub height: u32,
}

/// Return a single part's adjusted mask as grayscale PNG (for brush canvas init).
#[tauri::command]
pub async fn get_target_mask_cmd(app: AppHandle, target: String) -> Result<TargetMaskResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        let w = *state.mask_width.lock().unwrap();
        let h = *state.mask_height.lock().unwrap();
        let size = (w * h) as usize;

        let mask_data = match target.as_str() {
            "eye" => state.eye_mask.lock().unwrap().clone(),
            "mouth" => state.mouth_mask.lock().unwrap().clone(),
            "eyebrow" => state.eyebrow_mask.lock().unwrap().clone(),
            _ => return Err(AppError::General(format!("Unknown target: {}", target))),
        };

        let pixels = mask_data.unwrap_or_else(|| vec![0u8; size]);
        let gray = image::GrayImage::from_raw(w, h, pixels)
            .ok_or_else(|| AppError::General("mask size mismatch".into()))?;
        let img = DynamicImage::ImageLuma8(gray);
        let b64 = image_utils::image_to_base64_png(&img);

        Ok(TargetMaskResult { mask_image: b64, width: w, height: h })
    })
    .await
    .map_err(|e| AppError::General(format!("Task join error: {}", e)))?
}

#[derive(Serialize)]
pub struct BrushApplyResult {
    pub overlay_preview: String,
}

/// Helper: decode base64 PNG to grayscale Vec<u8> at mask dimensions.
fn decode_brush_png(brush_data: &str, w: u32, h: u32) -> Result<Vec<u8>, AppError> {
    let b64_str = brush_data
        .strip_prefix("data:image/png;base64,")
        .unwrap_or(brush_data);
    let decoded = STANDARD.decode(b64_str)
        .map_err(|e| AppError::General(format!("base64 decode: {}", e)))?;
    let brush_img = image::load_from_memory(&decoded)
        .map_err(|e| AppError::General(format!("image decode: {}", e)))?;

    let brush_resized = if brush_img.width() != w || brush_img.height() != h {
        brush_img.resize_exact(w, h, image::imageops::FilterType::Lanczos3)
    } else {
        brush_img
    };
    Ok(brush_resized.to_luma8().into_raw())
}

/// Helper: generate overlay from current state.
fn make_overlay_result(state: &AppState) -> Result<BrushApplyResult, AppError> {
    let raw = state.raw_images.lock().unwrap();
    let eye = state.eye_mask.lock().unwrap();
    let mouth = state.mouth_mask.lock().unwrap();
    let eyebrow = state.eyebrow_mask.lock().unwrap();
    let w = *state.mask_width.lock().unwrap();
    let h = *state.mask_height.lock().unwrap();

    let source = raw.get("eye_open")
        .or_else(|| raw.get("mouth_closed"))
        .ok_or_else(|| AppError::General("画像未読込".into()))?;
    let bg = composite_onto_white(source, w, h);
    let overlay = create_overlay(&bg, eye.as_deref(), mouth.as_deref(), eyebrow.as_deref(), w, h);

    // Rebuild combined mask
    let size = (w * h) as usize;
    let combined = combine_all_masks(eye.as_deref(), mouth.as_deref(), eyebrow.as_deref(), size);
    *state.combined_mask.lock().unwrap() = Some(combined);

    Ok(BrushApplyResult {
        overlay_preview: image_utils::image_to_base64_png(&overlay),
    })
}

/// Receive brush-edited mask for a specific part and overwrite the adjusted mask.
#[tauri::command]
pub async fn apply_brush_mask_cmd(
    app: AppHandle,
    target: String,
    brush_data: String,
) -> Result<BrushApplyResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        let w = *state.mask_width.lock().unwrap();
        let h = *state.mask_height.lock().unwrap();

        let brush_gray = decode_brush_png(&brush_data, w, h)?;

        // Overwrite the target part's adjusted mask
        match target.as_str() {
            "eye" => *state.eye_mask.lock().unwrap() = Some(brush_gray),
            "mouth" => *state.mouth_mask.lock().unwrap() = Some(brush_gray),
            "eyebrow" => *state.eyebrow_mask.lock().unwrap() = Some(brush_gray),
            _ => return Err(AppError::General(format!("Unknown target: {}", target))),
        }

        make_overlay_result(&state)
    })
    .await
    .map_err(|e| AppError::General(format!("Task join error: {}", e)))?
}

/// Delete a part's mask entirely (both raw and adjusted).
#[tauri::command]
pub async fn delete_mask_cmd(app: AppHandle, target: String) -> Result<BrushApplyResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();

        match target.as_str() {
            "eye" => {
                *state.raw_eye_mask.lock().unwrap() = None;
                *state.eye_mask.lock().unwrap() = None;
            }
            "mouth" => {
                *state.raw_mouth_mask.lock().unwrap() = None;
                *state.mouth_mask.lock().unwrap() = None;
            }
            "eyebrow" => {
                *state.raw_eyebrow_mask.lock().unwrap() = None;
                *state.eyebrow_mask.lock().unwrap() = None;
            }
            _ => return Err(AppError::General(format!("Unknown target: {}", target))),
        }

        make_overlay_result(&state)
    })
    .await
    .map_err(|e| AppError::General(format!("Task join error: {}", e)))?
}
