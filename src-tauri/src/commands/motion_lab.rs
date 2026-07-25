use crate::commands::parts::{arm_overlay_parent, is_arm_overlay_part_name};
use crate::commands::workspace::WorkspaceProject;
use crate::error::AppError;
use crate::processing::image_utils;
use base64::{engine::general_purpose::STANDARD, Engine};
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
/// rootが本物の7STEPワークスペースの`04_spritalk_parts`である場合のみ、
/// その親（project.jsonと同階層＝ジョブフォルダ直下）を返す。ワークスペース外の
/// 任意フォルダをMotion Labに直接指定された場合はNone＝root自身を使わせる
fn workspace_root_of(root: &Path) -> Option<PathBuf> {
    let parent = root.parent()?;
    if parent.join("project.json").is_file() {
        Some(parent.to_path_buf())
    } else {
        None
    }
}

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

/// manifest.jsonはジョブフォルダ直下(rootの親)が正だが、移行前の旧ワークスペースは
/// root(04_spritalk_parts)直下にまだ残っている場合があるため両方見る
fn is_codex_rife_output(root: &Path) -> bool {
    let candidates = [
        workspace_root_of(root).map(|parent| parent.join("manifest.json")),
        Some(root.join("manifest.json")),
    ];
    candidates
        .into_iter()
        .flatten()
        .find_map(|path| fs::read(path).ok())
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .and_then(|manifest| {
            manifest
                .get("mode")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .is_some_and(|mode| mode == "codex-rife-output")
}

fn parse_draw_order(value: &Value) -> Option<Vec<String>> {
    value.get("drawOrder").and_then(|order| {
        order.as_array().map(|entries| {
            entries
                .iter()
                .filter_map(|entry| entry.as_str().map(str::to_string))
                .collect::<Vec<_>>()
        })
    })
}

/// layer-order.json（{"drawOrder": ["hair_back", ...]}）を読む。無ければ空。
/// SpriTalk書き出し完了後は単体ファイルが消え、spritalk-motion-profile.jsonの
/// layerOrder.drawOrderへ統合されているためそちらも見る。
/// 再保存でprofileから落ちた壊れた成果物向けに、base_partsの原本も見る。
fn read_layer_draw_order(root: &Path, warnings: &mut Vec<String>) -> Vec<String> {
    let standalone = fs::read(root.join("layer-order.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .and_then(|value| parse_draw_order(&value));
    let parsed = standalone
        .or_else(|| {
            fs::read(root.join("spritalk-motion-profile.json"))
                .ok()
                .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
                .and_then(|value| value.get("layerOrder").and_then(parse_draw_order))
        })
        .or_else(|| {
            read_workspace_base_layer_order_document(root).and_then(|value| parse_draw_order(&value))
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

/// PachiPakuGen自身のSTEP7調整値の保存先。SpriTalkへ渡すフォルダ(root)ではなく
/// ジョブフォルダ直下(project.jsonと同階層)に置く。親が取れない場合のみroot直下へ
fn motion_preview_manifest_path(root: &Path) -> PathBuf {
    workspace_root_of(root)
        .unwrap_or_else(|| root.to_path_buf())
        .join("motion-preview-manifest.json")
}

fn legacy_motion_preview_manifest_path(root: &Path) -> PathBuf {
    root.join("motion-preview-manifest.json")
}

fn save_motion_lab_manifest_inner(
    source_dir: &str,
    manifest: Value,
) -> Result<MotionLabManifestResult, AppError> {
    let root = validate_motion_lab_source_dir(source_dir)?;
    let path = motion_preview_manifest_path(&root);
    fs::write(
        &path,
        serde_json::to_vec_pretty(&manifest)
            .map_err(|error| AppError::General(format!("motion manifest作成失敗: {error}")))?,
    )?;
    let legacy_path = legacy_motion_preview_manifest_path(&root);
    if legacy_path != path && legacy_path.is_file() {
        fs::remove_file(&legacy_path)?;
    }
    Ok(MotionLabManifestResult {
        path: path.to_string_lossy().into_owned(),
        manifest,
    })
}

fn load_motion_lab_manifest_inner(source_dir: &str) -> Result<MotionLabManifestResult, AppError> {
    let root = validate_motion_lab_source_dir(source_dir)?;
    let path = motion_preview_manifest_path(&root);
    let legacy_path = legacy_motion_preview_manifest_path(&root);
    let resolved_path = if path.is_file() { &path } else { &legacy_path };
    if !resolved_path.is_file() {
        return Err(AppError::General(format!(
            "motion-preview-manifest.json が見つかりません: {}",
            path.display()
        )));
    }
    let manifest = serde_json::from_slice::<Value>(&fs::read(resolved_path)?)
        .map_err(|error| AppError::General(format!("motion manifest読込失敗: {error}")))?;
    Ok(MotionLabManifestResult {
        path: resolved_path.to_string_lossy().into_owned(),
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

/// SpriTalkへ渡す成果物をspritalk-motion-profile.json 1本にまとめるため、
/// 併存していたlayer-order.json（描画順）とREADME.txt（案内文）の内容を
/// layerOrder/readmeフィールドへ吸収し、元の2ファイルは削除する。
///
/// フロントは再保存のたびに layerOrder/readme 無しの新規JSONを渡すため、
/// 単体ファイルが既に消えている場合は既存profileの同フィールドを引き継ぐ。
/// それも無い壊れた成果物向けに、ワークスペースの base_parts/layer-order.json
/// からも復旧する。
fn fold_companion_files_into_profile(root: &Path, mut profile: Value) -> Value {
    let layer_order_path = root.join("layer-order.json");
    let readme_path = root.join("README.txt");
    let existing_profile = fs::read(root.join("spritalk-motion-profile.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok());

    let layer_order = fs::read(&layer_order_path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .or_else(|| {
            existing_profile
                .as_ref()
                .and_then(|value| value.get("layerOrder").cloned())
        })
        .or_else(|| read_workspace_base_layer_order_document(root));
    if let Some(layer_order) = layer_order {
        if let Some(map) = profile.as_object_mut() {
            map.insert("layerOrder".to_string(), layer_order);
        }
    }

    let readme = fs::read_to_string(&readme_path).ok().or_else(|| {
        existing_profile
            .as_ref()
            .and_then(|value| value.get("readme").and_then(Value::as_str).map(str::to_owned))
    });
    if let Some(readme) = readme {
        if let Some(map) = profile.as_object_mut() {
            map.insert("readme".to_string(), Value::String(readme));
        }
    }

    for path in [layer_order_path, readme_path] {
        if path.is_file() {
            let _ = fs::remove_file(path);
        }
    }
    profile
}

/// 04_spritalk_parts から見て、STEP4が残した base_parts/layer-order.json を読む。
fn read_workspace_base_layer_order_document(root: &Path) -> Option<Value> {
    let workspace = workspace_root_of(root)?;
    [
        workspace
            .join("03_see_through")
            .join("base_parts")
            .join("layer-order.json"),
        workspace.join("base_parts").join("layer-order.json"),
    ]
    .into_iter()
    .find(|path| path.is_file())
    .and_then(|path| fs::read(path).ok())
    .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
}

fn save_spritalk_motion_profile_inner(
    source_dir: &str,
    profile: Value,
) -> Result<SpritalkMotionProfileResult, AppError> {
    let root = validate_motion_lab_source_dir(source_dir)?;
    let profile = fold_companion_files_into_profile(&root, profile);
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

// ===== ループ素材書き出し =====
// フロントのオフラインレンダラが1フレームずつPNG(base64)を送り、
// 完了時にPNG連番からAPNG（acTL/fcTL）を組み立てる。

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SaveMotionLoopFrameRequest {
    pub source_dir: String,
    pub frame_index: u32,
    /// dataURL（data:image/png;base64,...）または素のbase64
    pub png_base64: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FinalizeMotionLoopExportRequest {
    pub source_dir: String,
    pub fps: u32,
    pub frame_count: u32,
    pub make_apng: bool,
    pub make_gif: bool,
    pub keep_frames: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MotionLoopExportResult {
    pub export_dir: String,
    pub frames_dir: Option<String>,
    pub apng_path: Option<String>,
    pub gif_path: Option<String>,
    pub frame_count: u32,
    pub fps: u32,
}

const LOOP_EXPORT_DIR: &str = "loop_export";

/// 2枚のRGBA画像の間で実際に画素が変化した領域だけを矩形で返す（差分APNG用）。
/// 揺れ・呼吸のような局所的な動きでは、全画素のうち大部分が前フレームと
/// 完全に同一（背景・不動部分）なので、変化した矩形だけをfdATに書けば
/// 全面書き込みより大幅にファイルサイズを縮められる。完全に同一ならNone
fn frame_diff_bbox(
    previous: &image::RgbaImage,
    current: &image::RgbaImage,
) -> Option<(u32, u32, u32, u32)> {
    let (width, height) = current.dimensions();
    let prev_raw = previous.as_raw();
    let curr_raw = current.as_raw();
    let mut min_x = width;
    let mut max_x = 0u32;
    let mut min_y = height;
    let mut max_y = 0u32;
    let mut changed = false;
    for y in 0..height {
        let row_start = (y * width * 4) as usize;
        let row_end = row_start + (width * 4) as usize;
        if prev_raw[row_start..row_end] == curr_raw[row_start..row_end] {
            continue;
        }
        for x in 0..width {
            let start = (row_start + (x * 4) as usize) as usize;
            if prev_raw[start..start + 4] != curr_raw[start..start + 4] {
                changed = true;
                if x < min_x {
                    min_x = x;
                }
                if x > max_x {
                    max_x = x;
                }
                if y < min_y {
                    min_y = y;
                }
                if y > max_y {
                    max_y = y;
                }
            }
        }
    }
    if !changed {
        return None;
    }
    Some((min_x, min_y, max_x - min_x + 1, max_y - min_y + 1))
}

/// RGBA画像から矩形領域の生バイト列を行単位で切り出す（APNGのfdAT用）。
fn crop_rgba(image: &image::RgbaImage, x: u32, y: u32, w: u32, h: u32) -> Vec<u8> {
    let (full_width, _) = image.dimensions();
    let raw = image.as_raw();
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for row in y..y + h {
        let start = ((row * full_width + x) * 4) as usize;
        let end = start + (w * 4) as usize;
        out.extend_from_slice(&raw[start..end]);
    }
    out
}

#[tauri::command]
pub async fn save_motion_loop_frame(
    request: SaveMotionLoopFrameRequest,
) -> Result<(), AppError> {
    tauri::async_runtime::spawn_blocking(move || save_motion_loop_frame_inner(&request))
        .await
        .map_err(|error| AppError::General(format!("Task join error: {error}")))?
}

fn save_motion_loop_frame_inner(request: &SaveMotionLoopFrameRequest) -> Result<(), AppError> {
    let root = validate_motion_lab_source_dir(&request.source_dir)?;
    let export_dir = root.join(LOOP_EXPORT_DIR);
    if request.frame_index == 0 && export_dir.exists() {
        // 新しい書き出しの開始。前回の連番・APNGを混在させない
        fs::remove_dir_all(&export_dir)?;
    }
    let frames_dir = export_dir.join("frames");
    fs::create_dir_all(&frames_dir)?;
    let encoded = request
        .png_base64
        .rsplit(',')
        .next()
        .unwrap_or(&request.png_base64);
    let bytes = STANDARD
        .decode(encoded)
        .map_err(|error| AppError::General(format!("フレームPNGのbase64が不正です: {error}")))?;
    const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    if bytes.len() < 8 || bytes[..8] != PNG_SIGNATURE {
        return Err(AppError::General(
            "フレームデータがPNG形式ではありません".into(),
        ));
    }
    fs::write(
        frames_dir.join(format!("{:04}.png", request.frame_index)),
        bytes,
    )?;
    Ok(())
}

#[tauri::command]
pub async fn finalize_motion_loop_export(
    request: FinalizeMotionLoopExportRequest,
) -> Result<MotionLoopExportResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || finalize_motion_loop_export_inner(&request))
        .await
        .map_err(|error| AppError::General(format!("Task join error: {error}")))?
}

fn finalize_motion_loop_export_inner(
    request: &FinalizeMotionLoopExportRequest,
) -> Result<MotionLoopExportResult, AppError> {
    if request.frame_count == 0 {
        return Err(AppError::General("書き出すフレームがありません".into()));
    }
    if request.fps == 0 || request.fps > 120 {
        return Err(AppError::General("fpsは1〜120で指定してください".into()));
    }
    let root = validate_motion_lab_source_dir(&request.source_dir)?;
    let export_dir = root.join(LOOP_EXPORT_DIR);
    let frames_dir = export_dir.join("frames");
    let frame_path =
        |index: u32| frames_dir.join(format!("{index:04}.png"));
    for index in 0..request.frame_count {
        if !frame_path(index).is_file() {
            return Err(AppError::General(format!(
                "フレーム {index} が見つかりません。書き出しをやり直してください"
            )));
        }
    }

    let mut apng_path = None;
    if request.make_apng {
        let output_path = export_dir.join("loop.png");
        // 全フレームを一括で持たず、1枚ずつデコードして書き込む（数百フレーム対応）
        let first = image::open(frame_path(0))
            .map_err(|error| AppError::General(format!("フレーム0を読み込めません: {error}")))?
            .to_rgba8();
        let (width, height) = first.dimensions();
        let file = fs::File::create(&output_path)?;
        let writer = std::io::BufWriter::new(file);
        let mut encoder = png::Encoder::new(writer, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder
            .set_animated(request.frame_count, 0)
            .map_err(|error| AppError::General(format!("APNG設定に失敗: {error}")))?;
        encoder
            .set_frame_delay(1, request.fps as u16)
            .map_err(|error| AppError::General(format!("APNGフレーム間隔設定に失敗: {error}")))?;
        let mut png_writer = encoder
            .write_header()
            .map_err(|error| AppError::General(format!("APNGヘッダ書き込みに失敗: {error}")))?;
        // フレーム0は基準画像として全面書き込み（APNGの既定画像を兼ねる）
        png_writer
            .write_image_data(&first)
            .map_err(|error| AppError::General(format!("APNGフレーム0の書き込みに失敗: {error}")))?;
        let mut previous = first;
        for index in 1..request.frame_count {
            let frame = image::open(frame_path(index))
                .map_err(|error| {
                    AppError::General(format!("フレーム {index} を読み込めません: {error}"))
                })?
                .to_rgba8();
            if frame.dimensions() != (width, height) {
                return Err(AppError::General(format!(
                    "フレーム {index} のサイズが一致しません"
                )));
            }
            // 前フレームから変化した領域だけをfdATへ書き込む（差分エンコード）。
            // 揺れ・呼吸のような局所的な動きでは全面書き込みの数分の一まで
            // ファイルサイズを縮められる。並進順は必ずposition(0,0)リセット→
            // dimension→positionの順（pngクレートのfctl境界チェックが現在の
            // width/heightを参照するため、逆順だと前フレームの大きさで
            // 誤って範囲外エラーになることがある）
            let (x, y, patch_w, patch_h, patch) = match frame_diff_bbox(&previous, &frame) {
                Some((x, y, w, h)) => (x, y, w, h, crop_rgba(&frame, x, y, w, h)),
                // 前フレームと全く同じ場合でも1フレームとして記録する必要があるため、
                // 実害のない1x1の無変化パッチを書き込む
                None => (0, 0, 1, 1, frame.get_pixel(0, 0).0.to_vec()),
            };
            png_writer
                .reset_frame_position()
                .map_err(|error| AppError::General(format!("APNGフレーム {index} の位置初期化に失敗: {error}")))?;
            png_writer
                .set_frame_dimension(patch_w, patch_h)
                .map_err(|error| AppError::General(format!("APNGフレーム {index} のサイズ設定に失敗: {error}")))?;
            png_writer
                .set_frame_position(x, y)
                .map_err(|error| AppError::General(format!("APNGフレーム {index} の位置設定に失敗: {error}")))?;
            png_writer.write_image_data(&patch).map_err(|error| {
                AppError::General(format!("APNGフレーム {index} の書き込みに失敗: {error}"))
            })?;
            previous = frame;
        }
        png_writer
            .finish()
            .map_err(|error| AppError::General(format!("APNGの完了処理に失敗: {error}")))?;
        apng_path = Some(output_path.to_string_lossy().into_owned());
    }

    let mut gif_path = None;
    if request.make_gif {
        let output_path = export_dir.join("loop.gif");
        let first = image::open(frame_path(0))
            .map_err(|error| AppError::General(format!("フレーム0を読み込めません: {error}")))?
            .to_rgba8();
        let (width, height) = first.dimensions();
        let file = fs::File::create(&output_path)?;
        // GIFは256色パレット・完全透過/不透過の2値のみ。APNGより大幅に軽くなる一方、
        // 陰影のグラデーションでバンディングが出たり、髪の輪郭がやや硬くなる
        let mut gif_encoder = image::codecs::gif::GifEncoder::new(file);
        gif_encoder
            .set_repeat(image::codecs::gif::Repeat::Infinite)
            .map_err(|error| AppError::General(format!("GIFのループ設定に失敗: {error}")))?;
        let delay = image::Delay::from_numer_denom_ms(1000, request.fps.max(1));
        for index in 0..request.frame_count {
            let frame_image = if index == 0 {
                first.clone()
            } else {
                image::open(frame_path(index))
                    .map_err(|error| {
                        AppError::General(format!("フレーム {index} を読み込めません: {error}"))
                    })?
                    .to_rgba8()
            };
            if frame_image.dimensions() != (width, height) {
                return Err(AppError::General(format!(
                    "フレーム {index} のサイズが一致しません"
                )));
            }
            let frame = image::Frame::from_parts(frame_image, 0, 0, delay);
            gif_encoder.encode_frame(frame).map_err(|error| {
                AppError::General(format!("GIFフレーム {index} の書き込みに失敗: {error}"))
            })?;
        }
        drop(gif_encoder);
        gif_path = Some(output_path.to_string_lossy().into_owned());
    }

    let frames_dir_result = if request.keep_frames {
        Some(frames_dir.to_string_lossy().into_owned())
    } else {
        fs::remove_dir_all(&frames_dir)?;
        None
    };

    Ok(MotionLoopExportResult {
        export_dir: export_dir.to_string_lossy().into_owned(),
        frames_dir: frames_dir_result,
        apng_path,
        gif_path,
        frame_count: request.frame_count,
        fps: request.fps,
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

    fn write_test_project(root: &Path, current_step: u32) {
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
    }

    #[test]
    fn motion_preview_manifest_moves_to_job_root_inside_a_real_workspace() {
        let root = std::env::temp_dir().join(format!(
            "pachipakugen_manifest_relocation_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        let parts_dir = root.join("04_spritalk_parts");
        fs::create_dir_all(&parts_dir).unwrap();
        write_test_project(&root, 7);
        // 移行前の旧配置(04_spritalk_parts直下)に残っているファイルは新規保存時に消える
        fs::write(
            parts_dir.join("motion-preview-manifest.json"),
            b"{\"legacy\":true}",
        )
        .unwrap();

        let manifest = serde_json::json!({ "schema": "pachipakugen.motionPreview.v1" });
        let saved =
            save_motion_lab_manifest_inner(&parts_dir.to_string_lossy(), manifest.clone())
                .expect("save manifest");

        assert_eq!(
            PathBuf::from(&saved.path),
            root.join("motion-preview-manifest.json"),
            "本物のワークスペース内ではジョブ直下(project.jsonと同階層)に保存される"
        );
        assert!(
            !parts_dir.join("motion-preview-manifest.json").is_file(),
            "旧配置の残骸は削除される"
        );

        let loaded = load_motion_lab_manifest_inner(&parts_dir.to_string_lossy())
            .expect("load manifest");
        assert_eq!(loaded.manifest, manifest);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn motion_preview_manifest_stays_alongside_source_dir_without_a_real_workspace() {
        // project.jsonが無い任意フォルダ（ワークスペース外でMotion Labを直接使うケース）
        // では親フォルダへ書き出さず、指定フォルダ自身に留める
        let root = std::env::temp_dir().join(format!(
            "pachipakugen_manifest_no_workspace_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(&root).unwrap();

        let manifest = serde_json::json!({ "schema": "pachipakugen.motionPreview.v1" });
        let saved = save_motion_lab_manifest_inner(&root.to_string_lossy(), manifest)
            .expect("save manifest");

        assert_eq!(PathBuf::from(&saved.path), root.join("motion-preview-manifest.json"));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn saving_spritalk_profile_folds_layer_order_and_readme_and_removes_the_originals() {
        let root = std::env::temp_dir().join(format!(
            "pachipakugen_profile_fold_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("layer-order.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "formatVersion": 1,
                "drawOrder": ["hair_back", "body", "hair"]
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(root.join("README.txt"), "PachiPakuGen assets for SpriTalk\n").unwrap();

        let profile = serde_json::json!({ "schema": "spritalk.motionProfile.v2" });
        let saved = save_spritalk_motion_profile_inner(&root.to_string_lossy(), profile)
            .expect("save profile");

        assert_eq!(
            saved.profile["layerOrder"]["drawOrder"],
            serde_json::json!(["hair_back", "body", "hair"])
        );
        assert_eq!(
            saved.profile["readme"],
            serde_json::json!("PachiPakuGen assets for SpriTalk\n")
        );
        assert!(!root.join("layer-order.json").is_file(), "統合元は削除される");
        assert!(!root.join("README.txt").is_file(), "統合元は削除される");

        // SpriTalkへ渡すフォルダに残るファイルはspritalk-motion-profile.jsonのみ
        let remaining: Vec<_> = fs::read_dir(&root)
            .unwrap()
            .filter_map(Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(remaining, vec!["spritalk-motion-profile.json"]);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn layer_draw_order_falls_back_to_profile_after_export_removes_standalone_file() {
        let root = std::env::temp_dir().join(format!(
            "pachipakugen_layer_order_fallback_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(&root).unwrap();
        DynamicImage::new_rgba8(2, 2)
            .save(root.join("body.png"))
            .unwrap();
        fs::write(
            root.join("layer-order.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "formatVersion": 1,
                "drawOrder": ["hair_back", "body", "hair"]
            }))
            .unwrap(),
        )
        .unwrap();

        // エクスポート実行＝layer-order.jsonがspritalk-motion-profile.jsonへ統合され消える
        save_spritalk_motion_profile_inner(
            &root.to_string_lossy(),
            serde_json::json!({ "schema": "spritalk.motionProfile.v2" }),
        )
        .expect("save profile");
        assert!(!root.join("layer-order.json").is_file());

        let parts = load_motion_lab_parts_inner(&root.to_string_lossy()).expect("load parts");
        assert_eq!(parts.layer_order, vec!["hair_back", "body", "hair"]);
        assert!(parts.warnings.is_empty());

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn resaving_spritalk_profile_preserves_previously_folded_layer_order_and_readme() {
        let root = std::env::temp_dir().join(format!(
            "pachipakugen_profile_resave_preserve_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("layer-order.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "formatVersion": 1,
                "drawOrder": ["arm_r", "body", "arm_r_overlay_patch_hand", "arm_l", "hair"],
                "linkedParts": {
                    "arm_r_overlay_patch_hand": { "parent": "arm_r" }
                }
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(root.join("README.txt"), "keep me\n").unwrap();

        save_spritalk_motion_profile_inner(
            &root.to_string_lossy(),
            serde_json::json!({ "schema": "spritalk.motionProfile.v2", "motionScale": 1 }),
        )
        .expect("first save folds companions");
        assert!(!root.join("layer-order.json").is_file());

        // フロントは再保存時に layerOrder/readme 無しの新規JSONを渡す
        let resaved = save_spritalk_motion_profile_inner(
            &root.to_string_lossy(),
            serde_json::json!({
                "schema": "spritalk.motionProfile.v2",
                "motionScale": 1.2,
                "physics": { "arm": { "behindBody": false } }
            }),
        )
        .expect("second save must preserve folded fields");

        assert_eq!(resaved.profile["motionScale"], serde_json::json!(1.2));
        assert_eq!(
            resaved.profile["layerOrder"]["drawOrder"],
            serde_json::json!(["arm_r", "body", "arm_r_overlay_patch_hand", "arm_l", "hair"])
        );
        assert_eq!(
            resaved.profile["layerOrder"]["linkedParts"]["arm_r_overlay_patch_hand"]["parent"],
            serde_json::json!("arm_r")
        );
        assert_eq!(resaved.profile["readme"], serde_json::json!("keep me\n"));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn load_and_resave_recover_layer_order_from_workspace_base_parts() {
        let workspace = std::env::temp_dir().join(format!(
            "pachipakugen_layer_order_base_recover_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        let parts_dir = workspace.join("04_spritalk_parts");
        let base_dir = workspace.join("03_see_through").join("base_parts");
        fs::create_dir_all(&parts_dir).unwrap();
        fs::create_dir_all(&base_dir).unwrap();
        write_test_project(&workspace, 7);
        DynamicImage::new_rgba8(2, 2)
            .save(parts_dir.join("body.png"))
            .unwrap();
        fs::write(
            base_dir.join("layer-order.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "formatVersion": 1,
                "drawOrder": ["arm_r", "body", "arm_r_overlay_patch_hand", "hair"]
            }))
            .unwrap(),
        )
        .unwrap();
        // 再保存で落ちた壊れたprofile（layerOrder無し）
        fs::write(
            parts_dir.join("spritalk-motion-profile.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "schema": "spritalk.motionProfile.v2",
                "motionScale": 1
            }))
            .unwrap(),
        )
        .unwrap();

        let parts = load_motion_lab_parts_inner(&parts_dir.to_string_lossy()).expect("load");
        assert_eq!(
            parts.layer_order,
            vec!["arm_r", "body", "arm_r_overlay_patch_hand", "hair"]
        );
        assert!(parts.warnings.is_empty());

        let saved = save_spritalk_motion_profile_inner(
            &parts_dir.to_string_lossy(),
            serde_json::json!({ "schema": "spritalk.motionProfile.v2", "motionScale": 0.9 }),
        )
        .expect("resave heals profile");
        assert_eq!(
            saved.profile["layerOrder"]["drawOrder"],
            serde_json::json!(["arm_r", "body", "arm_r_overlay_patch_hand", "hair"])
        );

        let _ = fs::remove_dir_all(workspace);
    }

    fn encoded_test_frame(r: u8) -> String {
        let image = image::RgbaImage::from_pixel(4, 6, image::Rgba([r, 40, 60, 200]));
        let mut bytes = Vec::new();
        DynamicImage::ImageRgba8(image)
            .write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Png)
            .unwrap();
        STANDARD.encode(bytes)
    }

    #[test]
    fn loop_export_saves_frames_and_assembles_apng() {
        let root = std::env::temp_dir().join(format!(
            "pachipakugen-loop-export-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        ));
        fs::create_dir_all(&root).unwrap();

        for index in 0..3u32 {
            save_motion_loop_frame_inner(&SaveMotionLoopFrameRequest {
                source_dir: root.to_string_lossy().into_owned(),
                frame_index: index,
                png_base64: format!("data:image/png;base64,{}", encoded_test_frame(index as u8 * 80)),
            })
            .unwrap();
        }
        let result = finalize_motion_loop_export_inner(&FinalizeMotionLoopExportRequest {
            source_dir: root.to_string_lossy().into_owned(),
            fps: 30,
            frame_count: 3,
            make_apng: true,
            make_gif: false,
            keep_frames: true,
        })
        .unwrap();

        let apng_path = PathBuf::from(result.apng_path.expect("apng path"));
        assert!(apng_path.is_file());
        assert!(result.frames_dir.is_some());
        // APNGとして正しい構造か（acTLのフレーム数・無限ループ）をデコーダで検証
        let decoder = png::Decoder::new(std::io::BufReader::new(fs::File::open(&apng_path).unwrap()));
        let reader = decoder.read_info().unwrap();
        let animation = reader
            .info()
            .animation_control
            .expect("acTL chunk must exist");
        assert_eq!(animation.num_frames, 3);
        assert_eq!(animation.num_plays, 0);

        // 実際にcompositeした結果が、書き出し元の3フレームと一致することを確認する
        // （このテストのフィクスチャは各フレームが単色なので全面差分になり、
        // それでも位置・サイズ設定が壊れていないことの検証になる）
        let composited = decode_and_composite_apng(&apng_path);
        for (index, frame) in composited.iter().enumerate() {
            let expected = image::DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
                4,
                6,
                image::Rgba([index as u8 * 80, 40, 60, 200]),
            ))
            .to_rgba8();
            assert_eq!(*frame, expected, "frame {index} mismatch");
        }

        let _ = fs::remove_dir_all(root);
    }

    /// APNGを自前でcomposite（本実装が常にblend_op=Source/dispose_op=Noneを
    /// 使うことを前提に、各フレームのfcTL位置・サイズへ上書きするだけの
    /// 単純なモデルで十分）し、フレームごとの最終見た目を返す。
    fn decode_and_composite_apng(path: &Path) -> Vec<RgbaImage> {
        let decoder = png::Decoder::new(std::io::BufReader::new(fs::File::open(path).unwrap()));
        let mut reader = decoder.read_info().unwrap();
        let (full_width, full_height) = (reader.info().width, reader.info().height);
        let mut canvas = RgbaImage::new(full_width, full_height);
        let mut out = Vec::new();
        let frame_count = reader
            .info()
            .animation_control
            .expect("acTL chunk must exist")
            .num_frames;
        let apply_current_frame = |reader: &mut png::Reader<std::io::BufReader<fs::File>>, canvas: &mut RgbaImage| {
            let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
            reader.next_frame(&mut buf).unwrap();
            let fctl = *reader.info().frame_control.as_ref().unwrap();
            for row in 0..fctl.height {
                for col in 0..fctl.width {
                    let src = ((row * fctl.width + col) * 4) as usize;
                    let pixel = Rgba([buf[src], buf[src + 1], buf[src + 2], buf[src + 3]]);
                    canvas.put_pixel(fctl.x_offset + col, fctl.y_offset + row, pixel);
                }
            }
        };
        // read_info()は既にフレーム0のデータ位置まで読み進めているため、
        // フレーム0だけはnext_frame_infoを呼ばずに直接next_frameでdecodeする。
        // 2枚目以降だけnext_frame_infoで次のfcTLへ進める（呼びすぎるとフレーム数分
        // 進みきった後にPolledAfterEndOfImageになる）
        apply_current_frame(&mut reader, &mut canvas);
        out.push(canvas.clone());
        for _ in 1..frame_count {
            reader.next_frame_info().unwrap();
            apply_current_frame(&mut reader, &mut canvas);
            out.push(canvas.clone());
        }
        out
    }

    #[test]
    fn frame_diff_bbox_finds_tight_rectangle_and_none_when_identical() {
        let base = RgbaImage::from_pixel(20, 16, Rgba([10, 20, 30, 255]));
        let mut changed = base.clone();
        // (5,4)から幅3高さ2の矩形だけ書き換える
        for y in 4..6 {
            for x in 5..8 {
                changed.put_pixel(x, y, Rgba([200, 0, 0, 255]));
            }
        }
        let bbox = frame_diff_bbox(&base, &changed).expect("must detect a diff");
        assert_eq!(bbox, (5, 4, 3, 2));
        assert!(frame_diff_bbox(&base, &base).is_none());

        let cropped = crop_rgba(&changed, bbox.0, bbox.1, bbox.2, bbox.3);
        assert_eq!(cropped, vec![200, 0, 0, 255].repeat(6));
    }

    #[test]
    fn loop_export_apng_delta_encodes_and_composites_correctly() {
        let root = std::env::temp_dir().join(format!(
            "pachipakugen-loop-delta-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        ));
        fs::create_dir_all(&root).unwrap();
        let source_dir = root.to_string_lossy().into_owned();

        // 大きめのキャンバスで、ほとんど不動・一部だけ動く典型的なループ揺れを模す。
        // frame0: 背景一色。frame1: 左上に小さな四角が現れる。frame2: 四角が右へ移動。
        let (width, height) = (200u32, 160u32);
        let base = image::RgbaImage::from_pixel(width, height, image::Rgba([250, 250, 250, 255]));
        let mut frame1 = base.clone();
        for y in 10..30 {
            for x in 10..40 {
                frame1.put_pixel(x, y, image::Rgba([30, 120, 200, 255]));
            }
        }
        let mut frame2 = base.clone();
        for y in 10..30 {
            for x in 60..90 {
                frame2.put_pixel(x, y, image::Rgba([30, 120, 200, 255]));
            }
        }
        let expected = [base.clone(), frame1.clone(), frame2.clone()];
        for (index, image) in expected.iter().enumerate() {
            let mut bytes = Vec::new();
            image::DynamicImage::ImageRgba8(image.clone())
                .write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Png)
                .unwrap();
            save_motion_loop_frame_inner(&SaveMotionLoopFrameRequest {
                source_dir: source_dir.clone(),
                frame_index: index as u32,
                png_base64: STANDARD.encode(bytes),
            })
            .unwrap();
        }

        let result = finalize_motion_loop_export_inner(&FinalizeMotionLoopExportRequest {
            source_dir: source_dir.clone(),
            fps: 30,
            frame_count: 3,
            make_apng: true,
            make_gif: false,
            keep_frames: false,
        })
        .unwrap();
        let apng_path = PathBuf::from(result.apng_path.unwrap());

        // 合成結果が元の3フレームと完全一致すること（差分パッチが正しい位置へ適用されている）
        let composited = decode_and_composite_apng(&apng_path);
        assert_eq!(composited.len(), 3);
        for (index, frame) in composited.iter().enumerate() {
            assert_eq!(*frame, expected[index], "frame {index} mismatch");
        }

        // 全面書き込み（素朴な実装）と比べて大幅に小さいこと。
        // 3フレーム全面ならおよそ 3 * width*height*4 = 384000 バイト超になるはずだが、
        // 差分エンコードなら不動領域（大部分）を書かないため一桁小さくなる
        let apng_size = fs::metadata(&apng_path).unwrap().len();
        let naive_full_frame_size = (width * height * 4) as u64 * 3;
        assert!(
            apng_size < naive_full_frame_size / 4,
            "expected delta-encoded APNG well under naive size: {apng_size} vs {naive_full_frame_size}"
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn loop_export_gif_is_valid_animated_and_loops_infinitely() {
        let root = std::env::temp_dir().join(format!(
            "pachipakugen-loop-gif-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        ));
        fs::create_dir_all(&root).unwrap();
        let source_dir = root.to_string_lossy().into_owned();

        // 完全不透明の3フレーム（GIFの256色量子化・アルファ二値化を通しても
        // 判別できる、はっきり異なる原色にする）
        let colors = [[255u8, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]];
        for (index, color) in colors.iter().enumerate() {
            let image = RgbaImage::from_pixel(10, 8, Rgba(*color));
            let mut bytes = Vec::new();
            image::DynamicImage::ImageRgba8(image)
                .write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Png)
                .unwrap();
            save_motion_loop_frame_inner(&SaveMotionLoopFrameRequest {
                source_dir: source_dir.clone(),
                frame_index: index as u32,
                png_base64: STANDARD.encode(bytes),
            })
            .unwrap();
        }

        let result = finalize_motion_loop_export_inner(&FinalizeMotionLoopExportRequest {
            source_dir: source_dir.clone(),
            fps: 30,
            frame_count: 3,
            make_apng: false,
            make_gif: true,
            keep_frames: false,
        })
        .unwrap();
        let gif_path = PathBuf::from(result.gif_path.expect("gif path"));
        assert!(gif_path.is_file());
        assert!(result.apng_path.is_none());

        // 実際にGIFとしてデコードでき、無限ループ・正しいフレーム数・
        // 各フレームの支配的な色が入力と一致することを確認する
        let file = std::io::BufReader::new(fs::File::open(&gif_path).unwrap());
        let mut decoder_options = gif::DecodeOptions::new();
        decoder_options.set_color_output(gif::ColorOutput::RGBA);
        let mut decoder = decoder_options.read_info(file).unwrap();
        assert_eq!(decoder.repeat(), gif::Repeat::Infinite);
        let mut decoded_frames = 0;
        while let Some(frame) = decoder.read_next_frame().unwrap() {
            let expected = colors[decoded_frames];
            // 中心付近のピクセルで支配色を確認（256色量子化・透過二値化を経ても
            // 完全不透明の単色フレームなら色は保たれるはず）
            let center = ((frame.height as usize / 2) * frame.width as usize + frame.width as usize / 2) * 4;
            assert_eq!(&frame.buffer[center..center + 3], &expected[..3], "frame {decoded_frames} color mismatch");
            decoded_frames += 1;
        }
        assert_eq!(decoded_frames, 3);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn loop_export_restarts_cleanly_and_rejects_gaps() {
        let root = std::env::temp_dir().join(format!(
            "pachipakugen-loop-restart-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        ));
        fs::create_dir_all(&root).unwrap();
        let source_dir = root.to_string_lossy().into_owned();
        let save = |index: u32| {
            save_motion_loop_frame_inner(&SaveMotionLoopFrameRequest {
                source_dir: source_dir.clone(),
                frame_index: index,
                png_base64: encoded_test_frame(10),
            })
        };
        save(0).unwrap();
        save(1).unwrap();
        save(2).unwrap();
        // frame_index=0で前回の連番が消え、新しい書き出しとして始まる
        save(0).unwrap();
        save(1).unwrap();
        let error = finalize_motion_loop_export_inner(&FinalizeMotionLoopExportRequest {
            source_dir: source_dir.clone(),
            fps: 30,
            frame_count: 3,
            make_apng: false,
            make_gif: false,
            keep_frames: true,
        })
        .unwrap_err();
        assert!(error.to_string().contains("フレーム 2 が見つかりません"));

        // PNGでないデータは拒否する
        let invalid = save_motion_loop_frame_inner(&SaveMotionLoopFrameRequest {
            source_dir: source_dir.clone(),
            frame_index: 2,
            png_base64: STANDARD.encode(b"not a png"),
        })
        .unwrap_err();
        assert!(invalid.to_string().contains("PNG形式ではありません"));

        let _ = fs::remove_dir_all(root);
    }
}
