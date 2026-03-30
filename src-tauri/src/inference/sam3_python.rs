use crate::error::AppError;
use image::DynamicImage;
use std::collections::HashSet;
use std::path::Path;
use std::process::Command;

/// Run SAM3 segmentation via Python subprocess.
///
/// Calls `export_sam3_masks.py` with the given image and checkpoint,
/// then reads back the generated mask PNGs.
///
/// `prompts` is a comma-separated string of targets, e.g. "eye,mouth" or "eye,mouth,eyebrow".
/// Returns (eye_mask, mouth_mask, eyebrow_mask) — each is Some only if requested.
pub fn run_sam3_via_python(
    image: &DynamicImage,
    sam3_pt_path: &Path,
    script_path: &Path,
    prompts: &str,
) -> Result<(Option<Vec<u8>>, Option<Vec<u8>>, Option<Vec<u8>>), AppError> {
    let temp_dir = std::env::temp_dir().join("pachipakugen_sam3");
    std::fs::create_dir_all(&temp_dir)
        .map_err(|e| AppError::General(format!("一時ディレクトリの作成に失敗: {}", e)))?;

    // Save input image to temp file
    let temp_image = temp_dir.join("input.png");
    image
        .save(&temp_image)
        .map_err(|e| AppError::General(format!("一時画像の保存に失敗: {}", e)))?;

    let output_dir = temp_dir.join("output");
    std::fs::create_dir_all(&output_dir)
        .map_err(|e| AppError::General(format!("出力ディレクトリの作成に失敗: {}", e)))?;

    // Find Python executable
    let python = find_python()?;

    eprintln!(
        "[PachiPakuGen] SAM3 Python: {} {} --image {} --checkpoint {} --output-dir {} --prompts {}",
        python,
        script_path.display(),
        temp_image.display(),
        sam3_pt_path.display(),
        output_dir.display(),
        prompts,
    );

    // Run Python script with UTF-8 encoding forced
    let output = Command::new(&python)
        .env("PYTHONIOENCODING", "utf-8")
        .arg(script_path)
        .arg("--image")
        .arg(&temp_image)
        .arg("--checkpoint")
        .arg(sam3_pt_path)
        .arg("--output-dir")
        .arg(&output_dir)
        .arg("--prompts")
        .arg(prompts)
        .output()
        .map_err(|e| {
            AppError::General(format!(
                "Pythonの実行に失敗しました。Pythonがインストールされているか確認してください。\nエラー: {}",
                e
            ))
        })?;

    // Log stderr (Python progress messages)
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.is_empty() {
        eprintln!("[PachiPakuGen] SAM3 Python stderr:\n{}", stderr);
    }

    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);

        // Parse specific error types from stderr for better user messages
        let stderr_str = stderr.to_string();
        let detail = if stderr_str.contains("Missing dependencies") {
            let dep_lines: Vec<&str> = stderr_str
                .lines()
                .filter(|l| l.contains("- ") || l.contains("pip install") || l.contains("git clone"))
                .collect();
            format!(
                "SAM3に必要なPythonパッケージが不足しています。\n\n{}",
                dep_lines.join("\n")
            )
        } else if stderr_str.contains("sam3") && stderr_str.contains("not installed") {
            "sam3パッケージがインストールされていません。\n\n\
             セットアップ手順:\n\
             1. git clone https://github.com/facebookresearch/sam3.git\n\
             2. cd sam3 && pip install -e ."
                .to_string()
        } else {
            format!(
                "SAM3 Pythonスクリプトがエラーで終了しました。\n\n詳細:\n{}{}",
                stderr, stdout
            )
        };

        return Err(AppError::General(detail));
    }

    // Check stdout for "OK"
    let stdout = String::from_utf8_lossy(&output.stdout);
    if !stdout.trim().contains("OK") {
        return Err(AppError::General(format!(
            "SAM3スクリプトの出力が不正です: {}",
            stdout
        )));
    }

    // Read output masks — only for requested prompts
    let prompt_set: HashSet<&str> = prompts.split(',').map(|s| s.trim()).collect();

    let eye_mask = if prompt_set.contains("eye") {
        Some(read_grayscale_mask(&output_dir.join("eye_mask.png"), image.width(), image.height())?)
    } else {
        None
    };

    let mouth_mask = if prompt_set.contains("mouth") {
        Some(read_grayscale_mask(&output_dir.join("mouth_mask.png"), image.width(), image.height())?)
    } else {
        None
    };

    let eyebrow_mask = if prompt_set.contains("eyebrow") {
        Some(read_grayscale_mask(&output_dir.join("eyebrow_mask.png"), image.width(), image.height())?)
    } else {
        None
    };

    // Cleanup temp files (best-effort)
    let _ = std::fs::remove_dir_all(&temp_dir);

    Ok((eye_mask, mouth_mask, eyebrow_mask))
}

/// Read a grayscale PNG mask and resize to target dimensions if needed.
fn read_grayscale_mask(path: &Path, target_w: u32, target_h: u32) -> Result<Vec<u8>, AppError> {
    if !path.exists() {
        return Err(AppError::General(format!(
            "マスクファイルが見つかりません: {}",
            path.display()
        )));
    }

    let img = image::open(path)
        .map_err(|e| AppError::General(format!("マスク画像の読み込みに失敗: {}", e)))?;

    let gray = if img.width() != target_w || img.height() != target_h {
        let resized = img.resize_exact(
            target_w,
            target_h,
            image::imageops::FilterType::Nearest,
        );
        resized.to_luma8()
    } else {
        img.to_luma8()
    };

    Ok(gray.into_raw())
}

/// Find a working Python executable.
fn find_python() -> Result<String, AppError> {
    for name in &["python", "python3", "py"] {
        if let Ok(output) = Command::new(name).arg("--version").output() {
            if output.status.success() {
                return Ok(name.to_string());
            }
        }
    }

    Err(AppError::General(
        "Pythonが見つかりません。SAM3を使用するにはPythonのインストールが必要です。\n\
         https://www.python.org/ からインストールしてください。"
            .into(),
    ))
}

/// Resolve the SAM3 Python script path.
/// Looks for export_sam3_masks.py in prototypes/ or next to the executable.
pub fn resolve_script_path() -> Result<std::path::PathBuf, AppError> {
    // Dev mode: prototypes/ directory relative to PachiPakuGen project root
    // CARGO_MANIFEST_DIR = PachiPakuGen-app/src-tauri
    // -> parent = PachiPakuGen-app
    // -> parent = PachiPakuGen
    let dev_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("prototypes")
        .join("export_sam3_masks.py");
    if dev_path.exists() {
        return Ok(dev_path);
    }

    // Installed: next to executable
    if let Ok(exe) = std::env::current_exe() {
        let exe_dir = exe.parent().unwrap();
        let script = exe_dir.join("scripts").join("export_sam3_masks.py");
        if script.exists() {
            return Ok(script);
        }
        let script = exe_dir.join("export_sam3_masks.py");
        if script.exists() {
            return Ok(script);
        }
    }

    Err(AppError::General(
        "SAM3変換スクリプト(export_sam3_masks.py)が見つかりません。\n\
         prototypes/ フォルダに配置してください。"
            .into(),
    ))
}

/// Resolve the SAM3 .pt checkpoint path.
/// Looks in models/ directory at multiple locations.
pub fn resolve_sam3_checkpoint() -> Option<std::path::PathBuf> {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    // 1. src-tauri/models/sam3.pt (same dir as Cargo.toml)
    let local_path = manifest.join("models").join("sam3.pt");
    if local_path.exists() {
        return Some(local_path);
    }

    // 2. PachiPakuGen-app/models/sam3.pt
    let app_path = manifest.parent().unwrap().join("models").join("sam3.pt");
    if app_path.exists() {
        return Some(app_path);
    }

    // 3. PachiPakuGen/models/sam3.pt (project root)
    let root_path = manifest.parent().unwrap().parent().unwrap().join("models").join("sam3.pt");
    if root_path.exists() {
        return Some(root_path);
    }

    // 4. Next to executable
    if let Ok(exe) = std::env::current_exe() {
        let exe_dir = exe.parent().unwrap();
        let path = exe_dir.join("models").join("sam3.pt");
        if path.exists() {
            return Some(path);
        }
    }

    None
}
