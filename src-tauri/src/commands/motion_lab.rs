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
pub struct MotionLabPartsResult {
    pub source_dir: String,
    pub width: u32,
    pub height: u32,
    pub body: String,
    pub hair: Option<String>,
    pub hair_back: Option<String>,
    pub eye_frames: Vec<String>,
    pub mouths: HashMap<String, Vec<String>>,
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
    let eye_frames = read_frame_source(&root, "eye")?;

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
            warnings.push(
                "mouth_closed がないため、母音フォルダの先頭フレームを閉口として使います".into(),
            );
        } else {
            missing.push("mouth_closed".into());
        }
    }
    mouths.insert("closed".into(), closed_frames);

    Ok(MotionLabPartsResult {
        source_dir: root.to_string_lossy().into_owned(),
        width,
        height,
        body,
        hair,
        hair_back,
        eye_frames,
        mouths,
        missing,
        warnings,
    })
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

fn save_spritalk_motion_profile_inner(
    source_dir: &str,
    profile: Value,
) -> Result<SpritalkMotionProfileResult, AppError> {
    let root = validate_motion_lab_source_dir(source_dir)?;
    let path = root.join("spritalk-motion-profile.json");
    fs::write(
        &path,
        serde_json::to_vec_pretty(&profile)
            .map_err(|error| AppError::General(format!("SpriTalk motion profile作成失敗: {error}")))?,
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
    let dir = root.join(name);
    if dir.is_dir() {
        return read_frame_dir(&dir);
    }

    let file = root.join(format!("{name}.png"));
    if file.is_file() {
        return Ok(vec![image_utils::image_to_base64_png(&open_image(&file)?)]);
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

fn read_frame_dir(dir: &Path) -> Result<Vec<String>, AppError> {
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
        .map(|path| Ok(image_utils::image_to_base64_png(&open_image(path)?)))
        .collect()
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
            .any(|warning| warning.contains("mouth_closed")));

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

        assert_eq!(
            saved_path.file_name().and_then(|name| name.to_str()),
            Some("spritalk-motion-profile.json")
        );
        assert_eq!(saved.profile, profile);
        assert_eq!(saved_json, profile);

        let _ = fs::remove_dir_all(root);
    }
}
