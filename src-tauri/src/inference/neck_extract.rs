use crate::error::AppError;
use image::DynamicImage;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::process::{Command, Stdio};

/// Extract a body part mask from an image using SAM3 via Python subprocess.
/// `prompt` is the text prompt for SAM3 (e.g. "neck", "mouth").
/// Returns a grayscale mask (Vec<u8>) where 255 = detected, 0 = not detected.
pub fn extract_mask_with_sam3(
    image: &DynamicImage,
    prompt: &str,
    sam3_checkpoint: Option<&Path>,
) -> Result<Vec<u8>, AppError> {
    let mut masks = extract_masks_with_sam3(image, &[prompt], sam3_checkpoint)?;
    masks
        .pop()
        .ok_or_else(|| AppError::General(format!("SAM3マスクが空です: {}", prompt)))
}

/// Extract multiple body part masks in a single SAM3 subprocess call.
pub fn extract_masks_with_sam3(
    image: &DynamicImage,
    prompts: &[&str],
    sam3_checkpoint: Option<&Path>,
) -> Result<Vec<Vec<u8>>, AppError> {
    let sam3_ckpt = sam3_checkpoint.ok_or_else(|| {
        AppError::General("sam3.pt が見つかりません。models/ に配置してください".into())
    })?;
    if prompts.is_empty() {
        return Err(AppError::General("SAM3プロンプトが空です".into()));
    }

    let temp_dir = std::env::temp_dir().join("pachipakugen_sam3");
    std::fs::create_dir_all(&temp_dir)
        .map_err(|e| AppError::General(format!("一時ディレクトリの作成に失敗: {}", e)))?;

    // Composite onto white background for SAM3 (transparent→black confuses detection)
    let temp_image = temp_dir.join("input.png");
    let rgba = image.to_rgba8();
    let w = image.width();
    let h = image.height();
    let mut rgb_img = image::RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let p = rgba.get_pixel(x, y);
            let a = p[3] as f32 / 255.0;
            rgb_img.put_pixel(
                x,
                y,
                image::Rgb([
                    (p[0] as f32 * a + 255.0 * (1.0 - a)) as u8,
                    (p[1] as f32 * a + 255.0 * (1.0 - a)) as u8,
                    (p[2] as f32 * a + 255.0 * (1.0 - a)) as u8,
                ]),
            );
        }
    }
    DynamicImage::ImageRgb8(rgb_img)
        .save(&temp_image)
        .map_err(|e| AppError::General(format!("一時画像の保存に失敗: {}", e)))?;

    let output_dir = temp_dir.join("output");
    std::fs::create_dir_all(&output_dir)
        .map_err(|e| AppError::General(format!("出力ディレクトリの作成に失敗: {}", e)))?;

    let script_path = resolve_script_path()?;
    let python = find_python()?;

    eprintln!(
        "[PachiPakuGen] SAM3 extraction: prompts='{}', script={}",
        prompts.join(","),
        script_path.display()
    );

    let mut child = Command::new(&python)
        .env("PYTHONIOENCODING", "utf-8")
        .arg(&script_path)
        .arg("--image")
        .arg(&temp_image)
        .arg("--checkpoint")
        .arg(sam3_ckpt)
        .arg("--output-dir")
        .arg(&output_dir)
        .arg("--prompts")
        .arg(prompts.join(","))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| AppError::General(format!("Pythonの実行に失敗: {}", e)))?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| AppError::General("SAM3 stdout を取得できません".into()))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| AppError::General("SAM3 stderr を取得できません".into()))?;

    let stdout_handle = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let mut reader = BufReader::new(stdout);
        let _ = reader.read_to_end(&mut buf);
        buf
    });

    let stderr_handle = std::thread::spawn(move || {
        let mut collected = String::new();
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            match line {
                Ok(line) => {
                    eprintln!("[PachiPakuGen] SAM3: {}", line);
                    collected.push_str(&line);
                    collected.push('\n');
                }
                Err(e) => {
                    let msg = format!("SAM3 stderr read failed: {}", e);
                    eprintln!("[PachiPakuGen] {}", msg);
                    collected.push_str(&msg);
                    collected.push('\n');
                    break;
                }
            }
        }
        collected
    });

    let status = child
        .wait()
        .map_err(|e| AppError::General(format!("Pythonの実行に失敗: {}", e)))?;
    let stdout_bytes = stdout_handle
        .join()
        .map_err(|_| AppError::General("SAM3 stdout reader が異常終了しました".into()))?;
    let stderr = stderr_handle
        .join()
        .map_err(|_| AppError::General("SAM3 stderr reader が異常終了しました".into()))?;
    if !status.success() {
        let stdout = String::from_utf8_lossy(&stdout_bytes);
        return Err(AppError::General(format!(
            "SAM3がエラーで終了:\n{}\n{}",
            stderr, stdout
        )));
    }

    let stdout = String::from_utf8_lossy(&stdout_bytes);
    if !stdout.trim().contains("OK") {
        return Err(AppError::General(format!("SAM3出力が不正: {}", stdout)));
    }

    let w = image.width();
    let h = image.height();
    let mut masks = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        let mask_path = output_dir.join(format!("{}_mask.png", prompt));
        masks.push(read_grayscale_mask(&mask_path, w, h)?);
    }

    let _ = std::fs::remove_dir_all(&temp_dir);
    Ok(masks)
}

pub fn extract_mouth_raw_mask(
    image: &DynamicImage,
    sam3_checkpoint: Option<&Path>,
) -> Result<Vec<u8>, AppError> {
    extract_mask_with_sam3(image, "mouth", sam3_checkpoint)
}

pub fn adjust_mask(
    mask: &[u8],
    width: u32,
    height: u32,
    dilate_radius: i32,
    blur_radius: i32,
) -> Vec<u8> {
    let radius = dilate_radius.clamp(0, 64);
    let blur = blur_radius.clamp(0, 32);
    let dilated = dilate_mask(mask, width, height, radius);
    if blur <= 0 {
        dilated
    } else {
        let blurred = blur_mask(&dilated, width, height, blur);
        dilated
            .iter()
            .zip(blurred)
            .map(|(hard, soft)| soft.min(*hard))
            .collect()
    }
}

/// Dilate a binary mask by the given radius (in pixels).
fn dilate_mask(mask: &[u8], width: u32, height: u32, radius: i32) -> Vec<u8> {
    if radius <= 0 {
        return mask.to_vec();
    }
    let mut result = vec![0u8; mask.len()];
    let r2 = radius * radius;
    for y in 0..height as i32 {
        for x in 0..width as i32 {
            if mask[(y as u32 * width + x as u32) as usize] > 128 {
                // Expand this pixel to surrounding area
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        if dx * dx + dy * dy <= r2 {
                            let nx = x + dx;
                            let ny = y + dy;
                            if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                                result[(ny as u32 * width + nx as u32) as usize] = 255;
                            }
                        }
                    }
                }
            }
        }
    }
    result
}

fn blur_mask(mask: &[u8], width: u32, height: u32, radius: i32) -> Vec<u8> {
    if radius <= 0 {
        return mask.to_vec();
    }
    let mut horizontal = vec![0u8; mask.len()];
    let diameter = radius * 2 + 1;
    for y in 0..height as i32 {
        let mut sum = 0u32;
        for x in -radius..=radius {
            let cx = x.clamp(0, width as i32 - 1);
            sum += mask[(y as u32 * width + cx as u32) as usize] as u32;
        }
        for x in 0..width as i32 {
            horizontal[(y as u32 * width + x as u32) as usize] = (sum / diameter as u32) as u8;
            let remove_x = (x - radius).clamp(0, width as i32 - 1);
            let add_x = (x + radius + 1).clamp(0, width as i32 - 1);
            sum = sum
                .saturating_sub(mask[(y as u32 * width + remove_x as u32) as usize] as u32)
                .saturating_add(mask[(y as u32 * width + add_x as u32) as usize] as u32);
        }
    }

    let mut result = vec![0u8; mask.len()];
    for x in 0..width as i32 {
        let mut sum = 0u32;
        for y in -radius..=radius {
            let cy = y.clamp(0, height as i32 - 1);
            sum += horizontal[(cy as u32 * width + x as u32) as usize] as u32;
        }
        for y in 0..height as i32 {
            result[(y as u32 * width + x as u32) as usize] = (sum / diameter as u32) as u8;
            let remove_y = (y - radius).clamp(0, height as i32 - 1);
            let add_y = (y + radius + 1).clamp(0, height as i32 - 1);
            sum = sum
                .saturating_sub(horizontal[(remove_y as u32 * width + x as u32) as usize] as u32)
                .saturating_add(horizontal[(add_y as u32 * width + x as u32) as usize] as u32);
        }
    }
    result
}

/// Apply a grayscale mask to an image, returning the masked RGBA result.
pub fn apply_mask_to_image(
    image: &DynamicImage,
    mask: &[u8],
    width: u32,
    height: u32,
) -> DynamicImage {
    let rgba = image.to_rgba8();
    let mut result = image::RgbaImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let mask_val = mask[(y * width + x) as usize];
            if mask_val > 0 {
                let mut p = *rgba.get_pixel(x, y);
                p[3] = ((p[3] as u16 * mask_val as u16) / 255) as u8;
                result.put_pixel(x, y, p);
            }
        }
    }
    DynamicImage::ImageRgba8(result)
}

fn read_grayscale_mask(path: &Path, target_w: u32, target_h: u32) -> Result<Vec<u8>, AppError> {
    if !path.exists() {
        return Err(AppError::General(format!(
            "マスクファイルが見つかりません: {}",
            path.display()
        )));
    }
    let img =
        image::open(path).map_err(|e| AppError::General(format!("マスク読み込み失敗: {}", e)))?;
    let gray = if img.width() != target_w || img.height() != target_h {
        img.resize_exact(target_w, target_h, image::imageops::FilterType::Nearest)
            .to_luma8()
    } else {
        img.to_luma8()
    };
    Ok(gray.into_raw())
}

fn find_python() -> Result<String, AppError> {
    if let Ok(path) = std::env::var("PACHIPAKUGEN_PYTHON") {
        let path = path.trim();
        if !path.is_empty() {
            let configured_path = std::path::PathBuf::from(path);
            let mut candidates = vec![configured_path.clone()];

            if configured_path.is_relative() {
                if let Ok(current_dir) = std::env::current_dir() {
                    candidates.push(current_dir.join(&configured_path));
                }

                let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
                candidates.push(manifest.join(&configured_path));
                if let Some(root) = manifest.parent() {
                    candidates.push(root.join(&configured_path));
                }
            }

            for candidate in candidates {
                if let Ok(output) = Command::new(&candidate).arg("--version").output() {
                    if output.status.success() {
                        let python = candidate.to_string_lossy().into_owned();
                        eprintln!(
                            "[PachiPakuGen] Using Python from PACHIPAKUGEN_PYTHON: {}",
                            python
                        );
                        return Ok(python);
                    }
                }
            }
            return Err(AppError::General(format!(
                "PACHIPAKUGEN_PYTHON の Python を実行できません: {}",
                path
            )));
        }
    }

    for name in &["python", "python3", "py"] {
        if let Ok(output) = Command::new(name).arg("--version").output() {
            if output.status.success() {
                return Ok(name.to_string());
            }
        }
    }
    Err(AppError::General(
        "Pythonが見つかりません。SAM3用のPythonをPATHに追加するか、PACHIPAKUGEN_PYTHON に Python 実行ファイルのパスを設定してください".into()
    ))
}

fn resolve_script_path() -> Result<std::path::PathBuf, AppError> {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest.parent().unwrap();
    let script = root.join("scripts").join("extract_neck_mask.py");
    if script.exists() {
        return Ok(script);
    }
    if let Ok(exe) = std::env::current_exe() {
        let s = exe
            .parent()
            .unwrap()
            .join("scripts")
            .join("extract_neck_mask.py");
        if s.exists() {
            return Ok(s);
        }
    }
    Err(AppError::General("SAM3スクリプトが見つかりません".into()))
}

pub fn find_sam3_checkpoint() -> Option<std::path::PathBuf> {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for dir in &[
        manifest.join("models"),
        manifest.parent().unwrap().join("models"),
    ] {
        let path = dir.join("sam3.pt");
        if path.exists() {
            return Some(path);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        let path = exe.parent().unwrap().join("models").join("sam3.pt");
        if path.exists() {
            return Some(path);
        }
    }
    None
}
