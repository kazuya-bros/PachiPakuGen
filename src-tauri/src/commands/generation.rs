use crate::error::AppError;
use crate::inference::rife::rife_interpolate;
use crate::inference::session::{create_session, resolve_model_path};
use crate::processing::composite::premultiply_onto_body;
use crate::processing::image_utils;
use crate::state::{AppState, PartFrames};
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};

#[derive(Clone, Serialize)]
pub struct ProgressPayload {
    pub current: u32,
    pub total: u32,
    pub pair_name: String,
    pub ratio: f32,
}

#[derive(Serialize)]
pub struct GenerationResult {
    pub total_frames: u32,
    pub pair_results: Vec<PairResult>,
}

#[derive(Serialize)]
pub struct PairResult {
    pub name: String,
    pub frame_count: u32,
    pub previews: Vec<String>,
}

/// Generate intermediate frames for all part pairs using RIFE.
#[tauri::command]
pub async fn generate_frames(
    app: AppHandle,
    frame_count: u32,
) -> Result<GenerationResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || {
        generate_frames_inner(app, frame_count)
    })
    .await
    .map_err(|e| AppError::General(format!("Task join error: {}", e)))?
}

fn generate_frames_inner(
    app: AppHandle,
    frame_count: u32,
) -> Result<GenerationResult, AppError> {
    if frame_count < 2 || frame_count > 30 {
        return Err(AppError::General(
            "フレーム数は2〜30の範囲で指定してください".into(),
        ));
    }

    let state = app.state::<AppState>();

    // Initialize RIFE session if needed
    {
        let mut session = state.rife_session.lock().unwrap();
        if session.is_none() {
            let model_path = resolve_model_path(&app, "rife.onnx")?;
            *session = Some(create_session(&model_path)?);
        }
    }

    let pairs = state.part_pairs.lock().unwrap().clone();
    if pairs.is_empty() {
        return Err(AppError::General(
            "パーツペアがありません。先にパーツを読み込んでください".into(),
        ));
    }

    // Get body for premultiplication (avoid black from transparency)
    let parts = state.parts.lock().unwrap();
    let body = parts.get("body")
        .ok_or_else(|| AppError::General("素体パーツが読み込まれていません".into()))?;
    let body_rgb = body.to_rgb8();
    let (w, h) = (body_rgb.width(), body_rgb.height());
    drop(parts);

    let ratios: Vec<f32> = (0..frame_count)
        .map(|i| i as f32 / (frame_count - 1) as f32)
        .collect();

    let total_steps = pairs.len() as u32 * frame_count;
    let mut current_step = 0u32;

    let mut session_guard = state.rife_session.lock().unwrap();
    let session = session_guard.as_mut().unwrap();

    let mut all_part_frames = Vec::new();
    let mut pair_results = Vec::new();

    for pair in &pairs {
        // Premultiply parts onto body for RIFE input
        let img_a_rgba = pair.image_a.to_rgba8();
        let img_b_rgba = pair.image_b.to_rgba8();
        let rife_a = premultiply_onto_body(&body_rgb, &img_a_rgba, w, h);
        let rife_b = premultiply_onto_body(&body_rgb, &img_b_rgba, w, h);

        let mut frames = Vec::new();
        let mut previews = Vec::new();

        for &ratio in &ratios {
            current_step += 1;
            let _ = app.emit(
                "generation-progress",
                ProgressPayload {
                    current: current_step,
                    total: total_steps,
                    pair_name: pair.name.clone(),
                    ratio,
                },
            );

            let interpolated = rife_interpolate(session, &rife_a, &rife_b, ratio)?;

            // Extract part using interpolated alpha from both image_a and image_b.
            // This is critical: mouth_closed (image_a) has thin alpha,
            // while mouth_open (image_b) has wider alpha for the open mouth shape.
            // Using only image_a's alpha would clip the open mouth frames.
            let part_frame = extract_part_with_blended_alpha(
                &interpolated, &img_a_rgba, &img_b_rgba, ratio, w, h,
            );

            let preview = image_utils::image_to_base64_png(&part_frame);
            previews.push(preview);
            frames.push(part_frame);
        }

        pair_results.push(PairResult {
            name: pair.name.clone(),
            frame_count,
            previews,
        });

        all_part_frames.push(PartFrames {
            name: pair.name.clone(),
            frames,
        });
    }

    *state.generated.lock().unwrap() = all_part_frames;
    *state.frame_count.lock().unwrap() = frame_count;

    Ok(GenerationResult {
        total_frames: total_steps,
        pair_results,
    })
}

/// Extract the part from RIFE output using blended alpha from both image_a and image_b.
///
/// Why both alphas are needed:
/// - mouth_closed (image_a) has alpha only on the thin closed-mouth line
/// - mouth_open (image_b) has alpha covering the wider open mouth shape
/// - Intermediate frames need alpha that transitions between these two shapes
/// - Using only image_a's alpha would clip open-mouth frames to the thin line
///
/// Alpha strategy: lerp between image_a and image_b alpha by ratio,
/// then use the max(lerped, union) to ensure full coverage.
fn extract_part_with_blended_alpha(
    rife_output: &image::DynamicImage,
    img_a_rgba: &image::RgbaImage,
    img_b_rgba: &image::RgbaImage,
    ratio: f32,
    width: u32,
    height: u32,
) -> image::DynamicImage {
    let rgb = rife_output.to_rgb8();
    let mut result = image::RgbaImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let alpha_a = img_a_rgba.get_pixel(x, y)[3] as f32;
            let alpha_b = img_b_rgba.get_pixel(x, y)[3] as f32;

            // Union alpha: any pixel that exists in either image_a or image_b
            let alpha_union = alpha_a.max(alpha_b);

            if alpha_union > 0.0 {
                // Lerp alpha for smooth transition between shapes
                let alpha_lerp = alpha_a * (1.0 - ratio) + alpha_b * ratio;
                // Use the larger of lerp and a threshold to avoid gaps
                // For intermediate frames, prefer the union to ensure coverage
                let alpha = alpha_lerp.max(alpha_union * 0.5).min(alpha_union);
                let p = rgb.get_pixel(x, y);
                result.put_pixel(x, y, image::Rgba([
                    p[0], p[1], p[2],
                    alpha.clamp(0.0, 255.0) as u8,
                ]));
            }
        }
    }

    image::DynamicImage::ImageRgba8(result)
}

/// Build a composite preview for a specific frame index.
/// Composites: body -> selected eye frame -> selected mouth frame -> hair
#[tauri::command]
pub async fn get_composite_preview(
    app: AppHandle,
    eye_frame: u32,
    mouth_pair: String,
    mouth_frame: u32,
) -> Result<String, AppError> {
    tauri::async_runtime::spawn_blocking(move || {
        get_composite_preview_inner(app, eye_frame, mouth_pair, mouth_frame)
    })
    .await
    .map_err(|e| AppError::General(format!("Task join error: {}", e)))?
}

fn get_composite_preview_inner(
    app: AppHandle,
    eye_frame: u32,
    mouth_pair: String,
    mouth_frame: u32,
) -> Result<String, AppError> {
    let state = app.state::<AppState>();
    let parts = state.parts.lock().unwrap();
    let generated = state.generated.lock().unwrap();
    let w = *state.canvas_width.lock().unwrap();
    let h = *state.canvas_height.lock().unwrap();

    eprintln!(
        "[PachiPakuGen] get_composite_preview: eye_frame={}, mouth_pair={}, mouth_frame={}, generated_pairs={}",
        eye_frame, mouth_pair, mouth_frame,
        generated.iter().map(|pf| pf.name.as_str()).collect::<Vec<_>>().join(",")
    );

    let body = parts.get("body")
        .ok_or_else(|| AppError::General("素体パーツがありません".into()))?;
    let hair = parts.get("hair")
        .ok_or_else(|| AppError::General("髪パーツがありません".into()))?;

    // Start with body
    let mut result = body.to_rgba8();

    // Composite eye frame
    if let Some(eye_frames) = generated.iter().find(|pf| pf.name == "eye") {
        let idx = (eye_frame as usize).min(eye_frames.frames.len().saturating_sub(1));
        eprintln!("[PachiPakuGen]   eye: idx={}, total_frames={}", idx, eye_frames.frames.len());
        let eye_img = &eye_frames.frames[idx];
        alpha_composite_onto(&mut result, &eye_img.to_rgba8(), w, h);
    } else {
        eprintln!("[PachiPakuGen]   eye: NOT FOUND in generated");
    }

    // Composite mouth frame
    // Try exact match first, then fall back to any available mouth pair
    let mouth_source = if !mouth_pair.is_empty() {
        generated.iter().find(|pf| pf.name == mouth_pair)
            .or_else(|| {
                eprintln!("[PachiPakuGen]   mouth '{}': NOT FOUND, trying fallback", mouth_pair);
                generated.iter().find(|pf| pf.name.starts_with("mouth_"))
            })
    } else {
        generated.iter().find(|pf| pf.name.starts_with("mouth_"))
    };
    if let Some(mouth_frames) = mouth_source {
        let idx = (mouth_frame as usize).min(mouth_frames.frames.len().saturating_sub(1));
        eprintln!("[PachiPakuGen]   mouth '{}': idx={}, total_frames={}", mouth_frames.name, idx, mouth_frames.frames.len());
        let mouth_img = &mouth_frames.frames[idx];
        alpha_composite_onto(&mut result, &mouth_img.to_rgba8(), w, h);
    } else {
        eprintln!("[PachiPakuGen]   mouth: NO mouth pairs found in generated at all");
    }

    // Composite hair on top
    let hair_rgba = hair.to_rgba8();
    alpha_composite_onto(&mut result, &hair_rgba, w, h);

    let composite = image::DynamicImage::ImageRgba8(result);
    Ok(image_utils::image_to_base64_jpeg(&composite, 80))
}

/// Alpha-composite src onto dst in-place.
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
                let r = dp[0] as f32 * (1.0 - sa) + sp[0] as f32 * sa;
                let g = dp[1] as f32 * (1.0 - sa) + sp[1] as f32 * sa;
                let b = dp[2] as f32 * (1.0 - sa) + sp[2] as f32 * sa;
                let a = dp[3] as f32 * (1.0 - sa) + 255.0 * sa;
                dst.put_pixel(x, y, image::Rgba([
                    r.clamp(0.0, 255.0) as u8,
                    g.clamp(0.0, 255.0) as u8,
                    b.clamp(0.0, 255.0) as u8,
                    a.clamp(0.0, 255.0) as u8,
                ]));
            }
        }
    }
}
