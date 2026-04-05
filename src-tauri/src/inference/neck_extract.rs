use crate::error::AppError;
use image::DynamicImage;
use std::path::Path;
use std::process::Command;

/// Extract a body part mask from an image using SAM3 via Python subprocess.
/// `prompt` is the text prompt for SAM3 (e.g. "neck", "mouth").
/// Returns a grayscale mask (Vec<u8>) where 255 = detected, 0 = not detected.
pub fn extract_mask_with_sam3(
    image: &DynamicImage,
    prompt: &str,
    sam3_checkpoint: Option<&Path>,
) -> Result<Vec<u8>, AppError> {
    let sam3_ckpt = sam3_checkpoint
        .ok_or_else(|| AppError::General("sam3.pt が見つかりません。models/ に配置してください".into()))?;

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
            rgb_img.put_pixel(x, y, image::Rgb([
                (p[0] as f32 * a + 255.0 * (1.0 - a)) as u8,
                (p[1] as f32 * a + 255.0 * (1.0 - a)) as u8,
                (p[2] as f32 * a + 255.0 * (1.0 - a)) as u8,
            ]));
        }
    }
    DynamicImage::ImageRgb8(rgb_img).save(&temp_image)
        .map_err(|e| AppError::General(format!("一時画像の保存に失敗: {}", e)))?;

    let output_dir = temp_dir.join("output");
    std::fs::create_dir_all(&output_dir)
        .map_err(|e| AppError::General(format!("出力ディレクトリの作成に失敗: {}", e)))?;

    let script_path = resolve_script_path()?;
    let python = find_python()?;

    eprintln!(
        "[PachiPakuGen] SAM3 extraction: prompt='{}', script={}",
        prompt, script_path.display()
    );

    let output = Command::new(&python)
        .env("PYTHONIOENCODING", "utf-8")
        .arg(&script_path)
        .arg("--image").arg(&temp_image)
        .arg("--checkpoint").arg(sam3_ckpt)
        .arg("--output-dir").arg(&output_dir)
        .arg("--prompts").arg(prompt)
        .output()
        .map_err(|e| AppError::General(format!("Pythonの実行に失敗: {}", e)))?;

    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.is_empty() {
        eprintln!("[PachiPakuGen] SAM3 stderr:\n{}", stderr);
    }

    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(AppError::General(format!("SAM3がエラーで終了:\n{}\n{}", stderr, stdout)));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    if !stdout.trim().contains("OK") {
        return Err(AppError::General(format!("SAM3出力が不正: {}", stdout)));
    }

    // Read mask: {prompt}_mask.png
    let mask_path = output_dir.join(format!("{}_mask.png", prompt));
    let w = image.width();
    let h = image.height();
    let mask = read_grayscale_mask(&mask_path, w, h)?;

    let _ = std::fs::remove_dir_all(&temp_dir);
    Ok(mask)
}

/// Convenience: extract neck mask
pub fn extract_neck_mask(
    image: &DynamicImage,
    sam3_checkpoint: Option<&Path>,
) -> Result<Vec<u8>, AppError> {
    extract_mask_with_sam3(image, "neck", sam3_checkpoint)
}

/// Extract mouth mask with extra dilation to cover closed lips.
/// SAM3 detects the open mouth cavity, but closed lips need a larger mask area.
pub fn extract_mouth_mask(
    image: &DynamicImage,
    sam3_checkpoint: Option<&Path>,
) -> Result<Vec<u8>, AppError> {
    let mask = extract_mask_with_sam3(image, "mouth", sam3_checkpoint)?;
    let w = image.width();
    let h = image.height();
    // Dilate mask to cover surrounding lip/chin area
    let dilated = dilate_mask(&mask, w, h, 15);
    Ok(dilated)
}

/// Dilate a binary mask by the given radius (in pixels).
fn dilate_mask(mask: &[u8], width: u32, height: u32, radius: i32) -> Vec<u8> {
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
            if mask_val > 128 {
                result.put_pixel(x, y, *rgba.get_pixel(x, y));
            }
        }
    }
    DynamicImage::ImageRgba8(result)
}

fn read_grayscale_mask(path: &Path, target_w: u32, target_h: u32) -> Result<Vec<u8>, AppError> {
    if !path.exists() {
        return Err(AppError::General(format!("マスクファイルが見つかりません: {}", path.display())));
    }
    let img = image::open(path)
        .map_err(|e| AppError::General(format!("マスク読み込み失敗: {}", e)))?;
    let gray = if img.width() != target_w || img.height() != target_h {
        img.resize_exact(target_w, target_h, image::imageops::FilterType::Nearest).to_luma8()
    } else {
        img.to_luma8()
    };
    Ok(gray.into_raw())
}

fn find_python() -> Result<String, AppError> {
    for name in &["python", "python3", "py"] {
        if let Ok(output) = Command::new(name).arg("--version").output() {
            if output.status.success() { return Ok(name.to_string()); }
        }
    }
    Err(AppError::General("Pythonが見つかりません".into()))
}

fn resolve_script_path() -> Result<std::path::PathBuf, AppError> {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest.parent().unwrap();
    let script = root.join("scripts").join("extract_neck_mask.py");
    if script.exists() { return Ok(script); }
    if let Ok(exe) = std::env::current_exe() {
        let s = exe.parent().unwrap().join("scripts").join("extract_neck_mask.py");
        if s.exists() { return Ok(s); }
    }
    Err(AppError::General("SAM3スクリプトが見つかりません".into()))
}

pub fn find_sam3_checkpoint() -> Option<std::path::PathBuf> {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for dir in &[manifest.join("models"), manifest.parent().unwrap().join("models")] {
        let path = dir.join("sam3.pt");
        if path.exists() { return Some(path); }
    }
    if let Ok(exe) = std::env::current_exe() {
        let path = exe.parent().unwrap().join("models").join("sam3.pt");
        if path.exists() { return Some(path); }
    }
    None
}
