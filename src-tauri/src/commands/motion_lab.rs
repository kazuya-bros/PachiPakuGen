use crate::commands::parts::{arm_overlay_parent, is_arm_overlay_part_name};
use crate::commands::workspace::WorkspaceProject;
use crate::error::AppError;
use crate::processing::image_utils;
use image::DynamicImage;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

const MOUTH_FOLDERS: &[(&str, &str)] = &[
    ("a", "mouth_a"),
    ("i", "mouth_i"),
    ("u", "mouth_u"),
    ("e", "mouth_e"),
    ("o", "mouth_o"),
];

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MotionLabLinkedPartResult {
    /// このパーツが追従する腕（arm_l / arm_r）。描画順だけは独立している。
    pub parent: String,
    pub image: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MotionLabPartsResult {
    pub source_dir: String,
    pub width: u32,
    pub height: u32,
    pub body: String,
    pub hair: Option<String>,
    pub hair_back: Option<String>,
    pub arm_l: Option<String>,
    pub arm_r: Option<String>,
    pub chest: Option<String>,
    /// sway_<name>.png（汎用揺れパーツ）。キーはファイル名のstem（例: "sway_ribbon"）
    pub sways: HashMap<String, String>,
    /// 腕と同じ変形へ追従し、layer-order.json上では独立したz位置を持つ切り出し片。
    pub linked_parts: HashMap<String, MotionLabLinkedPartResult>,
    /// 独立した眉素材。Noneなら旧形式の目フレームが眉を保持する。
    pub eyebrow: Option<String>,
    /// 視線ドリフト用（§8.4）: 白目=クリップ領域、虹彩=ドリフト対象
    pub eyewhite: Option<String>,
    pub irides: Option<String>,
    /// 目ハイライト（微小ドリフト対象）
    pub highlight: Option<String>,
    pub eye_frames: Vec<String>,
    pub mouths: HashMap<String, Vec<String>>,
    /// layer-order.json（Step4由来のグループ描画順、背面→前面）。無ければ空=固定z順
    pub layer_order: Vec<String>,
    pub missing: Vec<String>,
    pub warnings: Vec<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SaveMotionLabManifestRequest {
    pub source_dir: String,
    pub manifest: Value,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SaveSpritalkMotionProfileRequest {
    pub source_dir: String,
    pub profile: Value,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MotionLabManifestResult {
    pub path: String,
    pub manifest: Value,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpritalkMotionProfileResult {
    pub path: String,
    pub profile: Value,
}

#[tauri::command]
pub async fn load_motion_lab_parts(dir: String) -> Result<MotionLabPartsResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || load_motion_lab_parts_inner(&dir))
        .await
        .map_err(|error| AppError::General(format!("Task join error: {error}")))?
}

#[tauri::command]
pub async fn save_motion_lab_manifest(
    request: SaveMotionLabManifestRequest,
) -> Result<MotionLabManifestResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || {
        save_motion_lab_manifest_inner(&request.source_dir, request.manifest)
    })
    .await
    .map_err(|error| AppError::General(format!("Task join error: {error}")))?
}

#[tauri::command]
pub async fn load_motion_lab_manifest(
    source_dir: String,
) -> Result<MotionLabManifestResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || load_motion_lab_manifest_inner(&source_dir))
        .await
        .map_err(|error| AppError::General(format!("Task join error: {error}")))?
}

#[tauri::command]
pub async fn load_spritalk_motion_profile(
    source_dir: String,
) -> Result<SpritalkMotionProfileResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || load_spritalk_motion_profile_inner(&source_dir))
        .await
        .map_err(|error| AppError::General(format!("Task join error: {error}")))?
}

#[tauri::command]
pub async fn save_spritalk_motion_profile(
    request: SaveSpritalkMotionProfileRequest,
) -> Result<SpritalkMotionProfileResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || {
        save_spritalk_motion_profile_inner(&request.source_dir, request.profile)
    })
    .await
    .map_err(|error| AppError::General(format!("Task join error: {error}")))?
}

fn load_motion_lab_parts_inner(dir: &str) -> Result<MotionLabPartsResult, AppError> {
    let root = PathBuf::from(dir);
    if !root.is_dir() {
        return Err(AppError::General(format!(
            "SpriTalk素材フォルダが見つかりません: {}",
            root.display()
        )));
    }
    ensure_workspace_parts_are_current(&root)?;

    let body_path = root.join("body.png");
    if !body_path.is_file() {
        return Err(AppError::General(format!(
            "body.png が見つかりません: {}",
            body_path.display()
        )));
    }

    let body_image = open_image(&body_path)?;
    let width = body_image.width();
    let height = body_image.height();
    let body = image_utils::image_to_base64_png(&body_image);
    let hair =
        read_optional_image_aliases(&root, &["hair.png", "front-hair.png", "front_hair.png"])?;
    let hair_back =
        read_optional_image_aliases(&root, &["hair_back.png", "back-hair.png", "back_hair.png"])?;
    let arm_l = read_optional_image_aliases(&root, &["arm_l.png", "arm-l.png", "arm_left.png"])?;
    let arm_r = read_optional_image_aliases(&root, &["arm_r.png", "arm-r.png", "arm_right.png"])?;
    let chest = read_optional_image_aliases(&root, &["chest.png"])?;
    let sways = read_sway_images(&root)?;
    let linked_parts = read_linked_arm_images(&root)?;
    let eyebrow_image =
        open_optional_image_aliases(&root, &["eyebrow.png", "eyebrows.png", "brow.png"])?;
    let eyebrow = eyebrow_image.as_ref().map(image_utils::image_to_base64_png);
    let eyebrow_cleanup_mask = eyebrow_image
        .as_ref()
        .map(|image| image_utils::eyebrow_cleanup_mask(image, width, height));
    let eyewhite = read_optional_image_aliases(&root, &["eyewhite.png", "eye_white.png"])?;
    let irides = read_optional_image_aliases(&root, &["irides.png", "iris.png"])?;
    let highlight = read_optional_image_aliases(&root, &["highlight.png", "eye_highlight.png"])?;
    // 旧出力の目フレームには眉が焼き込まれている場合がある。独立眉があるときは
    // 読み込み時だけその領域を抜き、動かす眉と固定眉が二重にならないようにする。
    let eye_frames =
        read_frame_source_with_alpha_mask(&root, "eye", eyebrow_cleanup_mask.as_ref())?;

    let mut mouths = HashMap::new();
    let mut missing = Vec::new();
    let mut warnings = Vec::new();

    let mut first_vowel_frame: Option<String> = None;
    for (key, folder) in MOUTH_FOLDERS {
        let frames = match *key {
            "a" => read_frame_source_aliases(
                &root,
                &[
                    folder,
                    "mouth_open",
                    "mouth-open",
                    "eyes-open-mouth-open",
                    "eyes_closed_mouth_open",
                ],
            )?,
            _ => read_frame_source(&root, folder)?,
        };
        if frames.is_empty() {
            missing.push(format!("{folder}/"));
        } else if first_vowel_frame.is_none() {
            first_vowel_frame = frames.first().cloned();
        }
        mouths.insert((*key).to_string(), frames);
    }

    let mut closed_frames = read_frame_source_aliases(
        &root,
        &[
            "mouth_closed",
            "mouth-closed",
            "eyes-open-mouth-closed",
            "eyes_closed_mouth_closed",
            "eyes-closed-mouth-closed",
        ],
    )?;
    if closed_frames.is_empty() {
        if let Some(frame) = first_vowel_frame {
            closed_frames.push(frame);
            // PachiPakuGenのRIFE出力では、各母音フォルダの先頭が閉じ口になる。
            // この形式では専用mouth_closedが無いのが正常なので警告しない。
            if !is_codex_rife_output(&root) {
                warnings.push(
                    "閉じ口素材がないため、母音フォルダの先頭フレームを閉じ口として使います".into(),
                );
            }
        } else {
            missing.push("mouth_closed".into());
        }
    }
    mouths.insert("closed".into(), closed_frames);

    let layer_order = read_layer_draw_order(&root, &mut warnings);

    Ok(MotionLabPartsResult {
        source_dir: root.to_string_lossy().into_owned(),
        width,
        height,
        body,
        hair,
        hair_back,
        arm_l,
        arm_r,
        chest,
        sways,
        linked_parts,
        eyebrow,
        eyewhite,
        irides,
        highlight,
        eye_frames,
        mouths,
        layer_order,
        missing,
        warnings,
    })
}

/// 任意の素材フォルダは従来どおり読み込めるが、現行ワークスペース配下の
/// `04_spritalk_parts`はproject.jsonのSTEP7到達を有効性マーカーとして扱う。
/// STEP4/5再編集後に残った旧画像をライブ表示へ誤って読み込ませない。
fn ensure_workspace_parts_are_current(root: &Path) -> Result<(), AppError> {
    if root.file_name().and_then(|name| name.to_str()) != Some("04_spritalk_parts") {
        return Ok(());
    }
    let Some(workspace_root) = root.parent() else {
        return Ok(());
    };
    let project_path = workspace_root.join("project.json");
    if !project_path.is_file() {
        return Ok(());
    }
    let project: WorkspaceProject = serde_json::from_slice(&fs::read(&project_path)?)
        .map_err(|error| AppError::General(format!("project.json解析失敗: {error}")))?;
    if project.current_step < 7 {
        return Err(AppError::General(
            "この作業フォルダは再編集後のフレーム生成が未完了です。STEP6でRIFE補完を再実行してください"
                .into(),
        ));
    }
    Ok(())
}

fn is_codex_rife_output(root: &Path) -> bool {
    fs::read(root.join("manifest.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .and_then(|manifest| {
            manifest
                .get("mode")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .is_some_and(|mode| mode == "codex-rife-output")
}

/// layer-order.json（{"drawOrder": ["hair_back", ...]}）を読む。無ければ空
fn read_layer_draw_order(root: &Path, warnings: &mut Vec<String>) -> Vec<String> {
    let path = root.join("layer-order.json");
    if !path.is_file() {
        return Vec::new();
    }
    let parsed = fs::read(&path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .and_then(|value| {
            value.get("drawOrder").and_then(|order| {
                order.as_array().map(|entries| {
                    entries
                        .iter()
                        .filter_map(|entry| entry.as_str().map(str::to_string))
                        .collect::<Vec<_>>()
                })
            })
        });
    match parsed {
        Some(order) if !order.is_empty() => order,
        _ => {
            warnings.push("layer-order.json を読めなかったため固定レイヤー順を使います".into());
            Vec::new()
        }
    }
}

/// sway_*.png（汎用揺れパーツ）をフォルダ直下から収集する
fn read_sway_images(root: &Path) -> Result<HashMap<String, String>, AppError> {
    let mut sways = HashMap::new();
    if !root.is_dir() {
        return Ok(sways);
    }
    let mut paths = fs::read_dir(root)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext.eq_ignore_ascii_case("png"))
                    .unwrap_or(false)
                && path
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .map(|stem| stem.to_ascii_lowercase().starts_with("sway_"))
                    .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    paths.sort();
    for path in paths {
        let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        sways.insert(
            stem.to_string(),
            image_utils::image_to_base64_png(&open_image(&path)?),
        );
    }
    Ok(sways)
}

/// arm_l_overlay_*.png / arm_r_overlay_*.png を親腕つきの独立描画パーツとして収集する。
/// 旧素材には該当ファイルがないため、空mapのまま従来どおり動作する。
fn read_linked_arm_images(
    root: &Path,
) -> Result<HashMap<String, MotionLabLinkedPartResult>, AppError> {
    let mut linked_parts = HashMap::new();
    if !root.is_dir() {
        return Ok(linked_parts);
    }
    let mut paths = fs::read_dir(root)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext.eq_ignore_ascii_case("png"))
                    .unwrap_or(false)
                && path
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .map(is_arm_overlay_part_name)
                    .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    paths.sort();
    for path in paths {
        let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        let Some(parent) = arm_overlay_parent(stem) else {
            continue;
        };
        linked_parts.insert(
            stem.to_string(),
            MotionLabLinkedPartResult {
                parent: parent.to_string(),
                image: image_utils::image_to_base64_png(&open_image(&path)?),
            },
        );
    }
    Ok(linked_parts)
}

fn save_motion_lab_manifest_inner(
    source_dir: &str,
    manifest: Value,
) -> Result<MotionLabManifestResult, AppError> {
    let root = validate_motion_lab_source_dir(source_dir)?;
    let path = root.join("motion-preview-manifest.json");
    fs::write(
        &path,
        serde_json::to_vec_pretty(&manifest)
            .map_err(|error| AppError::General(format!("motion manifest作成失敗: {error}")))?,
    )?;
    Ok(MotionLabManifestResult {
        path: path.to_string_lossy().into_owned(),
        manifest,
    })
}

fn load_motion_lab_manifest_inner(source_dir: &str) -> Result<MotionLabManifestResult, AppError> {
    let root = validate_motion_lab_source_dir(source_dir)?;
    let path = root.join("motion-preview-manifest.json");
    if !path.is_file() {
        return Err(AppError::General(format!(
            "motion-preview-manifest.json が見つかりません: {}",
            path.display()
        )));
    }
    let manifest = serde_json::from_slice::<Value>(&fs::read(&path)?)
        .map_err(|error| AppError::General(format!("motion manifest読込失敗: {error}")))?;
    Ok(MotionLabManifestResult {
        path: path.to_string_lossy().into_owned(),
        manifest,
    })
}

fn load_spritalk_motion_profile_inner(
    source_dir: &str,
) -> Result<SpritalkMotionProfileResult, AppError> {
    let root = validate_motion_lab_source_dir(source_dir)?;
    let path = root.join("spritalk-motion-profile.json");
    if !path.is_file() {
        return Err(AppError::General(format!(
            "spritalk-motion-profile.json が見つかりません: {}",
            path.display()
        )));
    }
    let profile = serde_json::from_slice::<Value>(&fs::read(&path)?)
        .map_err(|error| AppError::General(format!("SpriTalk motion profile読込失敗: {error}")))?;
    Ok(SpritalkMotionProfileResult {
        path: path.to_string_lossy().into_owned(),
        profile,
    })
}

fn save_spritalk_motion_profile_inner(
    source_dir: &str,
    profile: Value,
) -> Result<SpritalkMotionProfileResult, AppError> {
    let root = validate_motion_lab_source_dir(source_dir)?;
    let path = root.join("spritalk-motion-profile.json");
    fs::write(
        &path,
        serde_json::to_vec_pretty(&profile).map_err(|error| {
            AppError::General(format!("SpriTalk motion profile作成失敗: {error}"))
        })?,
    )?;
    Ok(SpritalkMotionProfileResult {
        path: path.to_string_lossy().into_owned(),
        profile,
    })
}

fn validate_motion_lab_source_dir(source_dir: &str) -> Result<PathBuf, AppError> {
    let root = PathBuf::from(source_dir);
    if !root.is_dir() {
        return Err(AppError::General(format!(
            "SpriTalk素材フォルダが見つかりません: {}",
            root.display()
        )));
    }
    Ok(root)
}

fn read_optional_image(path: &Path) -> Result<Option<String>, AppError> {
    if !path.is_file() {
        return Ok(None);
    }
    let image = open_image(path)?;
    Ok(Some(image_utils::image_to_base64_png(&image)))
}

fn open_optional_image_aliases(
    root: &Path,
    names: &[&str],
) -> Result<Option<DynamicImage>, AppError> {
    for name in names {
        let path = root.join(name);
        if path.is_file() {
            return Ok(Some(open_image(&path)?));
        }
    }
    Ok(None)
}

fn read_optional_image_aliases(root: &Path, names: &[&str]) -> Result<Option<String>, AppError> {
    for name in names {
        let image = read_optional_image(&root.join(name))?;
        if image.is_some() {
            return Ok(image);
        }
    }
    Ok(None)
}

fn read_frame_source(root: &Path, name: &str) -> Result<Vec<String>, AppError> {
    read_frame_source_with_alpha_mask(root, name, None)
}

fn read_frame_source_with_alpha_mask(
    root: &Path,
    name: &str,
    alpha_mask: Option<&DynamicImage>,
) -> Result<Vec<String>, AppError> {
    let dir = root.join(name);
    if dir.is_dir() {
        return read_frame_dir_with_alpha_mask(&dir, alpha_mask);
    }

    let file = root.join(format!("{name}.png"));
    if file.is_file() {
        let image = open_image(&file)?;
        let image = alpha_mask
            .map(|mask| erase_alpha_with_mask(&image, mask))
            .unwrap_or(image);
        return Ok(vec![image_utils::image_to_base64_png(&image)]);
    }

    Ok(Vec::new())
}

fn read_frame_source_aliases(root: &Path, names: &[&str]) -> Result<Vec<String>, AppError> {
    for name in names {
        let frames = read_frame_source(root, name)?;
        if !frames.is_empty() {
            return Ok(frames);
        }
    }
    Ok(Vec::new())
}

fn read_frame_dir_with_alpha_mask(
    dir: &Path,
    alpha_mask: Option<&DynamicImage>,
) -> Result<Vec<String>, AppError> {
    let mut files = fs::read_dir(dir)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext.eq_ignore_ascii_case("png"))
                    .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    files.sort_by_key(|path| {
        path.file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default()
    });

    files
        .iter()
        .map(|path| {
            let image = open_image(path)?;
            let image = alpha_mask
                .map(|mask| erase_alpha_with_mask(&image, mask))
                .unwrap_or(image);
            Ok(image_utils::image_to_base64_png(&image))
        })
        .collect()
}

fn erase_alpha_with_mask(image: &DynamicImage, mask: &DynamicImage) -> DynamicImage {
    image_utils::subtract_alpha_mask(image, mask)
}

fn open_image(path: &Path) -> Result<DynamicImage, AppError> {
    image::open(path).map_err(|error| {
        AppError::General(format!(
            "画像を読み込めません: {} ({error})",
            path.display()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine as _;
    use image::{Rgba, RgbaImage};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn load_motion_lab_parts_uses_first_vowel_frame_as_closed_fallback() {
        let root = std::env::temp_dir().join(format!(
            "pachipakugen_motion_lab_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(root.join("mouth_a")).unwrap();
        RgbaImage::from_pixel(4, 4, Rgba([10, 20, 30, 255]))
            .save(root.join("body.png"))
            .unwrap();
        RgbaImage::from_pixel(4, 4, Rgba([0, 0, 0, 0]))
            .save(root.join("mouth_a").join("001.png"))
            .unwrap();

        let result = load_motion_lab_parts_inner(&root.to_string_lossy()).unwrap();

        assert_eq!(result.width, 4);
        assert!(result
            .mouths
            .get("a")
            .is_some_and(|frames| frames.len() == 1));
        assert!(result
            .mouths
            .get("closed")
            .is_some_and(|frames| frames.len() == 1));
        assert!(result
            .warnings
            .iter()
            .any(|warning| warning.contains("閉じ口素材")));
        assert!(result.linked_parts.is_empty());

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn load_motion_lab_parts_accepts_native_closed_frame_without_warning() {
        let root = std::env::temp_dir().join(format!(
            "pachipakugen_motion_lab_native_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(root.join("mouth_a")).unwrap();
        fs::write(
            root.join("manifest.json"),
            br#"{"mode":"codex-rife-output"}"#,
        )
        .unwrap();
        RgbaImage::from_pixel(4, 4, Rgba([10, 20, 30, 255]))
            .save(root.join("body.png"))
            .unwrap();
        RgbaImage::from_pixel(4, 4, Rgba([0, 0, 0, 0]))
            .save(root.join("mouth_a").join("001.png"))
            .unwrap();

        let result = load_motion_lab_parts_inner(&root.to_string_lossy()).unwrap();

        assert!(result
            .mouths
            .get("closed")
            .is_some_and(|frames| frames.len() == 1));
        assert!(!result
            .warnings
            .iter()
            .any(|warning| warning.contains("閉じ口素材")));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn load_motion_lab_parts_accepts_purupuru_style_aliases() {
        let root = std::env::temp_dir().join(format!(
            "pachipakugen_motion_lab_alias_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(&root).unwrap();
        RgbaImage::from_pixel(4, 4, Rgba([10, 20, 30, 255]))
            .save(root.join("body.png"))
            .unwrap();
        RgbaImage::from_pixel(4, 4, Rgba([1, 2, 3, 128]))
            .save(root.join("front-hair.png"))
            .unwrap();
        RgbaImage::from_pixel(4, 4, Rgba([4, 5, 6, 128]))
            .save(root.join("back-hair.png"))
            .unwrap();
        RgbaImage::from_pixel(4, 4, Rgba([0, 0, 0, 0]))
            .save(root.join("eyes-open-mouth-closed.png"))
            .unwrap();
        RgbaImage::from_pixel(4, 4, Rgba([80, 30, 30, 255]))
            .save(root.join("eyes-open-mouth-open.png"))
            .unwrap();

        let result = load_motion_lab_parts_inner(&root.to_string_lossy()).unwrap();

        assert!(result.hair.is_some());
        assert!(result.hair_back.is_some());
        assert!(result
            .mouths
            .get("closed")
            .is_some_and(|frames| frames.len() == 1));
        assert!(result
            .mouths
            .get("a")
            .is_some_and(|frames| frames.len() == 1));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn load_motion_lab_parts_reads_arm_chest_and_sway_parts() {
        let root = std::env::temp_dir().join(format!(
            "pachipakugen_motion_lab_arm_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(root.join("mouth_a")).unwrap();
        RgbaImage::from_pixel(4, 4, Rgba([10, 20, 30, 255]))
            .save(root.join("body.png"))
            .unwrap();
        RgbaImage::from_pixel(4, 4, Rgba([0, 0, 0, 0]))
            .save(root.join("mouth_a").join("001.png"))
            .unwrap();
        RgbaImage::from_pixel(4, 4, Rgba([1, 1, 1, 255]))
            .save(root.join("arm_l.png"))
            .unwrap();
        RgbaImage::from_pixel(4, 4, Rgba([2, 2, 2, 255]))
            .save(root.join("arm_r.png"))
            .unwrap();
        RgbaImage::from_pixel(4, 4, Rgba([3, 3, 3, 255]))
            .save(root.join("chest.png"))
            .unwrap();
        RgbaImage::from_pixel(4, 4, Rgba([4, 4, 4, 255]))
            .save(root.join("sway_ribbon.png"))
            .unwrap();
        RgbaImage::from_pixel(4, 4, Rgba([5, 5, 5, 255]))
            .save(root.join("sway_necktie.png"))
            .unwrap();
        RgbaImage::from_pixel(4, 4, Rgba([6, 6, 6, 255]))
            .save(root.join("arm_l_overlay_patch_fingers.png"))
            .unwrap();
        RgbaImage::from_pixel(4, 4, Rgba([7, 7, 7, 255]))
            .save(root.join("arm_r_overlay_patch_sleeve.png"))
            .unwrap();
        RgbaImage::from_pixel(4, 4, Rgba([250, 250, 250, 255]))
            .save(root.join("eyewhite.png"))
            .unwrap();
        RgbaImage::from_pixel(4, 4, Rgba([35, 20, 15, 255]))
            .save(root.join("eyebrow.png"))
            .unwrap();
        RgbaImage::from_pixel(4, 4, Rgba([90, 60, 20, 255]))
            .save(root.join("irides.png"))
            .unwrap();
        RgbaImage::from_pixel(4, 4, Rgba([255, 255, 255, 128]))
            .save(root.join("highlight.png"))
            .unwrap();

        let result = load_motion_lab_parts_inner(&root.to_string_lossy()).unwrap();

        assert!(result.arm_l.is_some());
        assert!(result.arm_r.is_some());
        assert!(result.chest.is_some());
        assert_eq!(result.sways.len(), 2);
        assert!(result.sways.contains_key("sway_ribbon"));
        assert!(result.sways.contains_key("sway_necktie"));
        assert_eq!(result.linked_parts.len(), 2);
        assert_eq!(
            result
                .linked_parts
                .get("arm_l_overlay_patch_fingers")
                .map(|part| part.parent.as_str()),
            Some("arm_l")
        );
        assert!(result
            .linked_parts
            .get("arm_l_overlay_patch_fingers")
            .is_some_and(|part| part.image.starts_with("data:image/png;base64,")));
        assert_eq!(
            result
                .linked_parts
                .get("arm_r_overlay_patch_sleeve")
                .map(|part| part.parent.as_str()),
            Some("arm_r")
        );
        assert!(result.eyewhite.is_some());
        assert!(result.eyebrow.is_some());
        assert!(result.irides.is_some());
        assert!(result.highlight.is_some());

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn load_motion_lab_parts_removes_legacy_brow_from_eye_frames() {
        let root = std::env::temp_dir().join(format!(
            "pachipakugen_motion_lab_brow_cleanup_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(root.join("eye")).unwrap();
        RgbaImage::from_pixel(2, 1, Rgba([10, 20, 30, 255]))
            .save(root.join("body.png"))
            .unwrap();
        let mut eye = RgbaImage::from_pixel(2, 1, Rgba([80, 90, 100, 255]));
        eye.put_pixel(1, 0, Rgba([50, 25, 15, 255]));
        eye.save(root.join("eye").join("001.png")).unwrap();
        let mut eyebrow = RgbaImage::new(2, 1);
        eyebrow.put_pixel(1, 0, Rgba([35, 20, 15, 255]));
        eyebrow.save(root.join("eyebrow.png")).unwrap();

        let result = load_motion_lab_parts_inner(&root.to_string_lossy()).unwrap();
        let encoded = result.eye_frames[0]
            .strip_prefix("data:image/png;base64,")
            .unwrap();
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .unwrap();
        let cleaned = image::load_from_memory(&bytes).unwrap().to_rgba8();

        assert_eq!(cleaned.get_pixel(0, 0)[3], 255);
        assert_eq!(cleaned.get_pixel(1, 0)[3], 0);
        assert_eq!(
            image::open(root.join("eye").join("001.png"))
                .unwrap()
                .to_rgba8()
                .get_pixel(1, 0)[3],
            255
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn load_motion_lab_parts_preserves_saved_layer_order() {
        let root = std::env::temp_dir().join(format!(
            "pachipakugen_motion_lab_layer_order_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(&root).unwrap();
        RgbaImage::from_pixel(4, 4, Rgba([10, 20, 30, 255]))
            .save(root.join("body.png"))
            .unwrap();
        fs::write(
            root.join("layer-order.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "formatVersion": 1,
                "drawOrder": [
                    "hair_back",
                    "arm_r",
                    "arm_l",
                    "body",
                    "sway_ear_r",
                    "sway_ear_l",
                    "hair",
                    "sway_ear"
                ]
            }))
            .unwrap(),
        )
        .unwrap();

        let result = load_motion_lab_parts_inner(&root.to_string_lossy()).unwrap();

        assert_eq!(
            result.layer_order,
            vec![
                "hair_back",
                "arm_r",
                "arm_l",
                "body",
                "sway_ear_r",
                "sway_ear_l",
                "hair",
                "sway_ear"
            ]
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn motion_lab_manifest_round_trips_json() {
        let root = std::env::temp_dir().join(format!(
            "pachipakugen_motion_lab_manifest_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(&root).unwrap();
        let manifest = serde_json::json!({
            "schema": "pachipakugen.motionPreview.v1",
            "methods": {
                "lipTimelineSmoother": {
                    "enabled": true,
                    "attackMs": 90
                }
            }
        });

        let saved = save_motion_lab_manifest_inner(&root.to_string_lossy(), manifest.clone())
            .expect("save manifest");
        let loaded =
            load_motion_lab_manifest_inner(&root.to_string_lossy()).expect("load manifest");

        assert!(PathBuf::from(saved.path).is_file());
        assert_eq!(loaded.manifest, manifest);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn spritalk_motion_profile_saves_json() {
        let root = std::env::temp_dir().join(format!(
            "pachipakugen_motion_profile_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(&root).unwrap();
        let profile = serde_json::json!({
            "schema": "spritalk.motionProfile.v1",
            "lipSync": {
                "method": "bridge"
            }
        });

        let saved = save_spritalk_motion_profile_inner(&root.to_string_lossy(), profile.clone())
            .expect("save profile");
        let saved_path = PathBuf::from(saved.path);
        let saved_json = serde_json::from_slice::<Value>(&fs::read(&saved_path).unwrap()).unwrap();
        let loaded =
            load_spritalk_motion_profile_inner(&root.to_string_lossy()).expect("load profile");

        assert_eq!(
            saved_path.file_name().and_then(|name| name.to_str()),
            Some("spritalk-motion-profile.json")
        );
        assert_eq!(saved.profile, profile);
        assert_eq!(saved_json, profile);
        assert_eq!(loaded.profile, profile);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn spritalk_motion_profile_rejects_missing_or_invalid_json() {
        let root = std::env::temp_dir().join(format!(
            "pachipakugen_motion_profile_invalid_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(&root).unwrap();

        assert!(load_spritalk_motion_profile_inner(&root.to_string_lossy()).is_err());
        fs::write(root.join("spritalk-motion-profile.json"), b"{not-json").unwrap();
        assert!(load_spritalk_motion_profile_inner(&root.to_string_lossy()).is_err());

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn live_parts_reject_stale_workspace_output_until_step7() {
        let root = std::env::temp_dir().join(format!(
            "pachipakugen_stale_live_parts_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        let parts_dir = root.join("04_spritalk_parts");
        fs::create_dir_all(&parts_dir).unwrap();
        DynamicImage::new_rgba8(2, 2)
            .save(parts_dir.join("body.png"))
            .unwrap();

        let write_project = |current_step| {
            fs::write(
                root.join("project.json"),
                serde_json::to_vec_pretty(&WorkspaceProject {
                    version: 1,
                    created_at: 1,
                    updated_at: 2,
                    current_step,
                    source_image_path: None,
                    reference_image_path: None,
                    codex_prompt: None,
                    mouth_corner: Default::default(),
                })
                .unwrap(),
            )
            .unwrap();
        };

        write_project(6);
        let error = load_motion_lab_parts_inner(&parts_dir.to_string_lossy())
            .err()
            .expect("STEP7未到達なら旧ライブ素材を拒否する");
        assert!(format!("{error}").contains("RIFE補完を再実行"));

        write_project(7);
        assert!(load_motion_lab_parts_inner(&parts_dir.to_string_lossy()).is_ok());

        let _ = fs::remove_dir_all(root);
    }
}
