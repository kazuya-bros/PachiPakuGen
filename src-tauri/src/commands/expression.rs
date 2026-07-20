use crate::commands::parts::{
    arm_overlay_parent, get_mapping_preview_inner, is_arm_overlay_part_name, load_slot_inner,
    MappingPreviewResult, ProgressPayload, SlotLoadResult,
};
use crate::commands::see_through;
use crate::commands::workspace::{
    complete_workspace_edit, invalidate_workspace_before_edit, visual_image_fingerprint,
    WorkspaceProject,
};
use crate::error::AppError;
use crate::inference::neck_extract;
use crate::inference::rife::rife_interpolate;
use crate::inference::session::{create_session, resolve_model_path};
use crate::processing::composite::{extract_part_from_body_composite, premultiply_onto_body};
use crate::processing::image_utils;
use crate::state::AppState;
use base64::{engine::general_purpose::STANDARD, Engine};
use image::{DynamicImage, GrayImage, ImageFormat, RgbaImage};
use keyring_core::Entry;
use reqwest::blocking::multipart::{Form, Part};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter, Manager};

const EYE_LAYER_NAMES: &[&str] = &["eyewhite", "irides", "eyelash", "eyebrow"];
const EYE_ANIMATION_LAYER_NAMES: &[&str] = &["eyewhite", "irides", "eyelash"];
const KEYRING_SERVICE: &str = "com.kazuya.pachipakugen";
const OPENAI_CREDENTIAL: &str = "openai-api-key";
const GEMINI_CREDENTIAL: &str = "gemini-api-key";
const IMAGE_API_TIMEOUT: Duration = Duration::from_secs(10 * 60);
const IMAGE_API_CONNECT_TIMEOUT: Duration = Duration::from_secs(20);
const EXPRESSION_RIFE_DEFAULT_FRAME_COUNT: u32 = 8;
const MOUTH_VOWEL_TARGETS: &[&str] = &["mouth-a", "mouth-i", "mouth-u", "mouth-e", "mouth-o"];
const WORKSPACE_CODEX_REQUEST_DIR: &str = "01_codex_request";
const WORKSPACE_GENERATED_PARTS_DIR: &str = "02_generated_parts";
const WORKSPACE_SEE_THROUGH_DIR: &str = "03_see_through";
const WORKSPACE_SPRITALK_PARTS_DIR: &str = "04_spritalk_parts";
const GENERATED_PART_TARGETS: &[&str] = &[
    "mouth-closed",
    "mouth-a",
    "mouth-i",
    "mouth-u",
    "mouth-e",
    "mouth-o",
    "eyes-closed",
];
// STEP5ではsource由来のeyes-openも位置調整できる。Codex生成の必須成果物には含めないため、
// GENERATED_PART_TARGETSとは分けて管理する。
const ADJUSTABLE_PART_TARGETS: &[&str] = &[
    "eyes-open",
    "eyes-closed",
    "mouth-closed",
    "mouth-a",
    "mouth-i",
    "mouth-u",
    "mouth-e",
    "mouth-o",
];

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateExpressionSetRequest {
    pub engine: String,
    pub quality: String,
    pub targets: Vec<String>,
    pub reference_image_path: Option<String>,
    pub prompt: String,
    pub mouth_corner: String,
    pub mouth_size: String,
    pub output_path: String,
    pub rife_frame_count: Option<u32>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateExpressionSetResult {
    pub output_path: String,
    pub generated_files: Vec<String>,
    pub engine: String,
    pub model: String,
    pub rife_output_path: Option<String>,
    pub rife_directories: Vec<String>,
    pub rife_frame_count: u32,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PrepareCodexExpressionJobRequest {
    pub source_image_path: String,
    pub reference_image_path: Option<String>,
    pub targets: Vec<String>,
    pub prompt: String,
    pub mouth_corner: String,
    pub mouth_size: String,
    pub output_path: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PrepareCodexExpressionJobResult {
    pub job_path: String,
    pub source_path: String,
    pub reference_path: Option<String>,
    pub request_path: String,
    pub handoff_path: String,
    pub generated_parts_path: String,
    pub expected_parts: Vec<String>,
    pub missing_parts: Vec<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InspectCodexGeneratedPartsResult {
    pub generated_parts_path: String,
    pub expected_parts: Vec<String>,
    pub present_parts: Vec<String>,
    pub missing_parts: Vec<String>,
    pub size_mismatches: Vec<String>,
    pub ready: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct PartAdjustment {
    pub offset_x: i32,
    pub offset_y: i32,
    pub scale_percent: u32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExtractCodexGeneratedPartsResult {
    pub extracted_parts_path: String,
    pub extracted_parts: Vec<String>,
    pub warnings: Vec<String>,
    pub selected_profile: String,
    pub effective_options: Option<see_through::SeeThroughOptions>,
    pub split_parts: bool,
    /// パーツごとの現在の位置補正値（STEP5でパーツ切替時に実際の値を表示するため）
    pub part_adjustments: std::collections::BTreeMap<String, PartAdjustment>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkspaceSourceSeeThroughResult {
    pub psd_path: String,
    pub output_dir: String,
    pub selected_profile: String,
    pub slot_load: SlotLoadResult,
    pub mapping_preview: MappingPreviewResult,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CodexCompositePreviewItem {
    pub part: String,
    pub preview: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PreviewCodexCompositeResult {
    pub base_preview: String,
    pub previews: Vec<CodexCompositePreviewItem>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateCodexRifeOutputResult {
    pub output_path: String,
    pub directories: Vec<String>,
    pub frame_count: u32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SaveCodexBasePartsResult {
    pub base_parts_path: String,
    pub saved_parts: Vec<String>,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AdjustCodexExtractedPartsRequest {
    pub job_path: String,
    pub offset_x: i32,
    pub offset_y: i32,
    pub scale_percent: u32,
    /// 指定時はそのパーツだけを調整（パーツ個別補正）。None=全パーツ一括
    #[serde(default)]
    pub part: Option<String>,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CodexPartAdjustmentUpdate {
    pub part: String,
    pub offset_x: i32,
    pub offset_y: i32,
    pub scale_percent: u32,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AdjustCodexExtractedPartsBatchRequest {
    pub job_path: String,
    pub adjustments: Vec<CodexPartAdjustmentUpdate>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AdjustCodexExtractedPartsResult {
    pub extracted_parts_path: String,
    pub adjusted_parts: Vec<String>,
    pub offset_x: i32,
    pub offset_y: i32,
    pub scale_percent: u32,
    /// 適用後のパーツごとの位置補正値（全パーツ分。フロント側の表示同期用）
    pub part_adjustments: std::collections::BTreeMap<String, PartAdjustment>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LoadCodexExpressionJobResult {
    pub job: PrepareCodexExpressionJobResult,
    pub generated_parts: InspectCodexGeneratedPartsResult,
    pub extracted_parts: Option<ExtractCodexGeneratedPartsResult>,
    pub rife_output: Option<GenerateCodexRifeOutputResult>,
    pub resume_step: u32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExpressionApiStatus {
    pub openai_configured: bool,
    pub gemini_configured: bool,
    pub openai_source: Option<String>,
    pub gemini_source: Option<String>,
}

#[derive(Clone, Copy)]
enum ExpressionMaskKind {
    Eye,
    Mouth,
}

#[derive(Clone)]
struct ExpressionRifeJob {
    name: &'static str,
    start_key: &'static str,
    end_key: &'static str,
    mask_kind: ExpressionMaskKind,
}

#[tauri::command]
pub async fn prepare_codex_expression_job(
    app: AppHandle,
    request: PrepareCodexExpressionJobRequest,
) -> Result<PrepareCodexExpressionJobResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || prepare_codex_expression_job_inner(app, request))
        .await
        .map_err(|error| AppError::General(format!("Task join error: {error}")))?
}

#[tauri::command]
pub async fn inspect_codex_generated_parts(
    job_path: String,
) -> Result<InspectCodexGeneratedPartsResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || inspect_codex_generated_parts_inner(&job_path))
        .await
        .map_err(|error| AppError::General(format!("Task join error: {error}")))?
}

#[tauri::command]
pub async fn load_codex_expression_job(
    job_path: String,
) -> Result<LoadCodexExpressionJobResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || load_codex_expression_job_inner(&job_path))
        .await
        .map_err(|error| AppError::General(format!("Task join error: {error}")))?
}

#[tauri::command]
pub async fn extract_codex_generated_parts(
    app: AppHandle,
    job_path: String,
    profile: String,
    split_parts: bool,
    options: Option<see_through::SeeThroughOptions>,
) -> Result<ExtractCodexGeneratedPartsResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || {
        extract_codex_generated_parts_inner(app, &job_path, &profile, split_parts, options)
    })
    .await
    .map_err(|error| AppError::General(format!("Task join error: {error}")))?
}

#[tauri::command]
pub async fn cache_codex_source_see_through(
    job_path: String,
    psd_path: String,
) -> Result<String, AppError> {
    tauri::async_runtime::spawn_blocking(move || {
        cache_codex_source_see_through_inner(&job_path, &psd_path)
    })
    .await
    .map_err(|error| AppError::General(format!("Task join error: {error}")))?
}

#[tauri::command]
pub async fn load_codex_source_see_through(
    app: AppHandle,
    job_path: String,
) -> Result<WorkspaceSourceSeeThroughResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || {
        load_codex_source_see_through_inner(app, &job_path)
    })
    .await
    .map_err(|error| AppError::General(format!("Task join error: {error}")))?
}

#[tauri::command]
pub async fn preview_codex_composite(
    app: AppHandle,
    job_path: String,
    profile: String,
) -> Result<PreviewCodexCompositeResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || {
        preview_codex_composite_inner(app, &job_path, &profile)
    })
    .await
    .map_err(|error| AppError::General(format!("Task join error: {error}")))?
}

#[tauri::command]
pub async fn generate_codex_rife_outputs(
    app: AppHandle,
    job_path: String,
    frame_count: u32,
    profile: String,
) -> Result<GenerateCodexRifeOutputResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || {
        generate_codex_rife_outputs_inner(app, &job_path, frame_count, &profile)
    })
    .await
    .map_err(|error| AppError::General(format!("Task join error: {error}")))?
}

#[tauri::command]
pub async fn save_codex_base_parts(
    app: AppHandle,
    job_path: String,
) -> Result<SaveCodexBasePartsResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || save_codex_base_parts_inner(&app, &job_path))
        .await
        .map_err(|error| AppError::General(format!("Task join error: {error}")))?
}

#[tauri::command]
pub async fn adjust_codex_extracted_parts_batch(
    request: AdjustCodexExtractedPartsBatchRequest,
) -> Result<AdjustCodexExtractedPartsResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || adjust_codex_extracted_parts_batch_inner(request))
        .await
        .map_err(|error| AppError::General(format!("Task join error: {error}")))?
}

fn prepare_codex_expression_job_inner(
    app: AppHandle,
    request: PrepareCodexExpressionJobRequest,
) -> Result<PrepareCodexExpressionJobResult, AppError> {
    let output_root = PathBuf::from(&request.output_path);
    fs::create_dir_all(&output_root)?;
    let output_dir = output_root.join(format!(
        "pachipakugen_codex_job_{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_secs())
            .unwrap_or(0)
    ));
    fs::create_dir_all(&output_dir)?;

    let source = PathBuf::from(&request.source_image_path);
    if !source.is_file() {
        return Err(AppError::General(format!(
            "元画像が見つかりません: {}",
            source.display()
        )));
    }

    let source_image = image::open(&source)
        .map_err(|error| AppError::General(format!("元画像を読み込めません: {error}")))?;
    let source_path = output_dir.join("source.png");
    source_image.save(&source_path)?;

    let reference_path = request
        .reference_image_path
        .as_deref()
        .filter(|path| !path.trim().is_empty())
        .map(|path| copy_reference_image(path, &output_dir))
        .transpose()?;

    let generated_parts_dir = output_dir.join("generated_parts");
    fs::create_dir_all(&generated_parts_dir)?;
    let base_parts_dir = output_dir.join("base_parts");
    export_current_base_parts(&app, &base_parts_dir)?;

    let expected_parts = effective_expression_targets(&request.targets, &output_dir);
    let request_text = build_codex_request_text(
        &source_path,
        reference_path.as_deref(),
        &generated_parts_dir,
        &expected_parts,
        &request,
    );
    let request_path = output_dir.join("codex_request.md");
    fs::write(&request_path, request_text)?;
    let handoff_path = output_dir.join("codex_handoff.md");
    fs::write(
        &handoff_path,
        build_codex_handoff_text(&output_dir, &request_path, &generated_parts_dir),
    )?;

    let manifest = json!({
        "formatVersion": 1,
        "mode": "codex-generated-parts",
        "status": "waitingForGeneratedParts",
        "source": source_path.to_string_lossy(),
        "reference": reference_path.as_ref().map(|path| path.to_string_lossy().into_owned()),
        "generatedPartsDirectory": generated_parts_dir.to_string_lossy(),
        "expectedGeneratedParts": expected_parts,
        "mouthCorner": request.mouth_corner,
        "mouthSize": request.mouth_size,
        "prompt": request.prompt,
    });
    fs::write(
        output_dir.join("codex_job.json"),
        serde_json::to_vec_pretty(&manifest)
            .map_err(|error| AppError::General(format!("codex_job.json作成失敗: {error}")))?,
    )?;

    let status = inspect_generated_parts(&output_dir, &source_image, &expected_parts)?;
    Ok(PrepareCodexExpressionJobResult {
        job_path: output_dir.to_string_lossy().into_owned(),
        source_path: source_path.to_string_lossy().into_owned(),
        reference_path: reference_path.map(|path| path.to_string_lossy().into_owned()),
        request_path: request_path.to_string_lossy().into_owned(),
        handoff_path: handoff_path.to_string_lossy().into_owned(),
        generated_parts_path: generated_parts_dir.to_string_lossy().into_owned(),
        expected_parts,
        missing_parts: status.missing_parts,
    })
}

fn inspect_codex_generated_parts_inner(
    job_path: &str,
) -> Result<InspectCodexGeneratedPartsResult, AppError> {
    let job_dir = PathBuf::from(job_path);
    if !job_dir.is_dir() {
        return Err(AppError::General(format!(
            "Codexジョブフォルダが見つかりません: {}",
            job_dir.display()
        )));
    }
    let source_path = job_source_path(&job_dir);
    let source_image = image::open(&source_path).map_err(|error| {
        AppError::General(format!(
            "ジョブ内の source.png を読み込めません: {} ({error})",
            source_path.display()
        ))
    })?;
    let expected_parts = expected_parts_from_job(&job_dir)?;
    inspect_generated_parts(&job_dir, &source_image, &expected_parts)
}

fn load_codex_expression_job_inner(
    job_path: &str,
) -> Result<LoadCodexExpressionJobResult, AppError> {
    let job_dir = PathBuf::from(job_path);
    if !job_dir.is_dir() {
        return Err(AppError::General(format!(
            "Codexジョブフォルダが見つかりません: {}",
            job_dir.display()
        )));
    }
    let source_path = job_source_path(&job_dir);
    if !source_path.is_file() {
        return Err(AppError::General(format!(
            "source.png が見つかりません。PachiPakuGenのCodexジョブフォルダを選択してください: {}",
            source_path.display()
        )));
    }
    let generated_parts = inspect_codex_generated_parts_inner(job_path)?;
    let expected_parts = generated_parts.expected_parts.clone();
    let extracted_parts = read_extracted_parts_result(&job_dir);
    let rife_output = read_current_rife_output_result(&job_dir);
    let reference_path = job_dir
        .join("reference.png")
        .is_file()
        .then(|| job_dir.join("reference.png").to_string_lossy().into_owned());
    let job = PrepareCodexExpressionJobResult {
        job_path: job_dir.to_string_lossy().into_owned(),
        source_path: source_path.to_string_lossy().into_owned(),
        reference_path,
        request_path: job_request_path(&job_dir).to_string_lossy().into_owned(),
        handoff_path: job_handoff_path(&job_dir).to_string_lossy().into_owned(),
        generated_parts_path: generated_parts.generated_parts_path.clone(),
        expected_parts,
        missing_parts: generated_parts.missing_parts.clone(),
    };
    let resume_step = if rife_output.is_some() {
        8
    } else if extracted_parts.is_some() {
        7
    } else if generated_parts.ready {
        6
    } else {
        5
    };

    Ok(LoadCodexExpressionJobResult {
        job,
        generated_parts,
        extracted_parts,
        rife_output,
        resume_step,
    })
}

fn read_extracted_parts_result(job_dir: &Path) -> Option<ExtractCodexGeneratedPartsResult> {
    let extracted_dir = extracted_parts_dir(job_dir);
    if !extracted_dir.is_dir() {
        return None;
    }
    let mut extracted_parts: Vec<String> = GENERATED_PART_TARGETS
        .iter()
        .filter(|part| extracted_dir.join(format!("{part}.png")).is_file())
        .map(|part| (*part).to_string())
        .collect();
    if extracted_dir.join("eyes-open.png").is_file() {
        extracted_parts.insert(0, "eyes-open".to_string());
    }
    if extracted_parts.is_empty() {
        return None;
    }
    let warnings = fs::read(extracted_dir.join("manifest.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .and_then(|value| {
            value
                .get("warnings")
                .and_then(|warnings| warnings.as_array())
                .cloned()
        })
        .map(|warnings| {
            warnings
                .iter()
                .filter_map(|warning| warning.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    Some(ExtractCodexGeneratedPartsResult {
        extracted_parts_path: extracted_dir.to_string_lossy().into_owned(),
        extracted_parts,
        warnings,
        selected_profile: "cached".to_string(),
        effective_options: None,
        split_parts: true,
        part_adjustments: read_typed_part_adjustments(&extracted_dir),
    })
}

fn read_rife_output_result(job_dir: &Path) -> Option<GenerateCodexRifeOutputResult> {
    let output_dir = rife_output_dir(job_dir);
    if !output_dir.is_dir() {
        return None;
    }
    let manifest = fs::read(output_dir.join("manifest.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok());
    let frame_count = manifest
        .as_ref()
        .and_then(|value| value.get("frameCount"))
        .and_then(|value| value.as_u64())
        .unwrap_or(EXPRESSION_RIFE_DEFAULT_FRAME_COUNT as u64) as u32;
    let mut directories = Vec::new();
    for name in ["eye", "mouth_a", "mouth_i", "mouth_u", "mouth_e", "mouth_o"] {
        let path = output_dir.join(name);
        if path.is_dir() {
            directories.push(path.to_string_lossy().into_owned());
        }
    }
    if directories.is_empty() {
        return None;
    }
    Some(GenerateCodexRifeOutputResult {
        output_path: output_dir.to_string_lossy().into_owned(),
        directories,
        frame_count,
    })
}

/// 旧形式の単体Codexジョブにはproject.jsonが無いため従来どおり物理成果物を読む。
/// 現行ワークスペースではSTEP4/5再編集時にチェックポイントを5/6へ戻すので、
/// 物理的に残した旧RIFE出力を「つづきから」で現行成果物として復活させない。
fn read_current_rife_output_result(job_dir: &Path) -> Option<GenerateCodexRifeOutputResult> {
    let project_path = job_dir.join("project.json");
    if project_path.is_file() {
        let project = fs::read(&project_path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<WorkspaceProject>(&bytes).ok())?;
        if project.current_step < 7 {
            return None;
        }
    }
    read_rife_output_result(job_dir)
}

fn export_current_base_parts(app: &AppHandle, output_dir: &Path) -> Result<(), AppError> {
    let Some(parts) = current_base_parts(app) else {
        return Err(AppError::General(
            "表情セット用の素体が未作成です。Step 2で分解結果を承認してからジョブを書き出してください".into(),
        ));
    };
    save_base_parts(&parts, output_dir)
}

fn save_codex_base_parts_inner(
    app: &AppHandle,
    job_path: &str,
) -> Result<SaveCodexBasePartsResult, AppError> {
    let job_dir = PathBuf::from(job_path);
    if !job_dir.is_dir() {
        return Err(AppError::General(format!(
            "Codexジョブフォルダが見つかりません: {}",
            job_dir.display()
        )));
    }
    let parts = current_base_parts(app)
        .ok_or_else(|| AppError::General("保存できる素体パーツがまだありません".into()))?;
    let output_dir = base_parts_dir(&job_dir);
    let extracted_dir = extracted_parts_dir(&job_dir);
    // 完了済み案件の再編集では、最初の画像書込より先にSTEP4へ戻す。
    // 以降で保存に失敗・中断しても旧RIFEを現行成果物として復活させない。
    invalidate_workspace_before_edit(&job_dir, 4)?;
    save_base_parts(&parts, &output_dir)?;
    save_layer_draw_order(app, &output_dir)?;
    save_eyes_open_extracted_part(&job_dir, &parts, &extracted_dir)?;
    let existing_output = rife_output_dir(&job_dir);
    if existing_output.is_dir() {
        sync_dynamic_eye_assets(&output_dir, &extracted_dir, &existing_output)?;
    }
    complete_workspace_edit(&job_dir, 5)?;
    let mut saved_parts = parts.keys().cloned().collect::<Vec<_>>();
    saved_parts.sort();
    Ok(SaveCodexBasePartsResult {
        base_parts_path: output_dir.to_string_lossy().into_owned(),
        saved_parts,
    })
}

/// Step4のunifiedレイヤー順から導出したグループ描画順（背面→前面）を
/// layer-order.json として素材フォルダへ保存する。順序情報が無ければ
/// 古いファイルを消す（固定z順フォールバック）。
fn save_layer_draw_order(app: &AppHandle, output_dir: &Path) -> Result<(), AppError> {
    let group_order = app
        .state::<AppState>()
        .base_layer_group_order
        .lock()
        .unwrap()
        .clone();
    let path = output_dir.join("layer-order.json");
    if group_order.is_empty() {
        if path.exists() {
            fs::remove_file(path)?;
        }
        return Ok(());
    }
    fs::write(
        &path,
        serde_json::to_vec_pretty(&layer_order_document(&group_order))
            .map_err(|error| AppError::General(format!("layer-order.json作成失敗: {error}")))?,
    )?;
    Ok(())
}

fn layer_order_document(group_order: &[String]) -> serde_json::Value {
    let mut document = serde_json::Map::new();
    document.insert("formatVersion".into(), serde_json::Value::from(1));
    document.insert("drawOrder".into(), serde_json::json!(group_order));

    let mut linked_parts = serde_json::Map::new();
    for part_name in group_order {
        let Some(parent) = arm_overlay_parent(part_name) else {
            continue;
        };
        linked_parts.insert(part_name.clone(), serde_json::json!({ "parent": parent }));
    }
    if !linked_parts.is_empty() {
        document.insert(
            "linkedParts".into(),
            serde_json::Value::Object(linked_parts),
        );
    }
    serde_json::Value::Object(document)
}

#[cfg(test)]
fn adjust_codex_extracted_parts_inner(
    request: AdjustCodexExtractedPartsRequest,
) -> Result<AdjustCodexExtractedPartsResult, AppError> {
    adjust_codex_extracted_parts_core(request, true)
}

fn adjust_codex_extracted_parts_core(
    request: AdjustCodexExtractedPartsRequest,
    manage_workspace_checkpoint: bool,
) -> Result<AdjustCodexExtractedPartsResult, AppError> {
    if !(50..=150).contains(&request.scale_percent) {
        return Err(AppError::General(
            "scalePercent は 50 から 150 の範囲で指定してください".into(),
        ));
    }
    let job_dir = PathBuf::from(&request.job_path);
    if !job_dir.is_dir() {
        return Err(AppError::General(format!(
            "Codexジョブフォルダが見つかりません: {}",
            job_dir.display()
        )));
    }
    let extracted_dir = extracted_parts_dir(&job_dir);
    if !extracted_dir.is_dir() {
        return Err(AppError::General(
            "抽出済みパーツが見つかりません。先にSee-Through一括分解を実行してください".into(),
        ));
    }
    let target_parts: Vec<&str> = match request.part.as_deref() {
        Some(part) => {
            if !ADJUSTABLE_PART_TARGETS.contains(&part) {
                return Err(AppError::General(format!("調整対象外のパーツです: {part}")));
            }
            vec![part]
        }
        None => ADJUSTABLE_PART_TARGETS.to_vec(),
    };
    if !target_parts
        .iter()
        .any(|part| extracted_dir.join(format!("{part}.png")).is_file())
    {
        return Err(AppError::General(
            "調整対象の差分パーツが見つかりません".into(),
        ));
    }
    // 位置調整ファイルへ触れる前にSTEP5へ戻し、クラッシュ時も古いRIFEをfail-safeで無効化する。
    if manage_workspace_checkpoint {
        invalidate_workspace_before_edit(&job_dir, 5)?;
    }
    let originals_dir = extracted_dir.join("original_extracted_parts");
    fs::create_dir_all(&originals_dir)?;
    let updates_eyes_open = target_parts.contains(&"eyes-open");
    // パーツ個別の調整値を保持（v2）。一括適用時は全パーツを同値で上書きする
    let mut part_adjustments = read_part_adjustments(&extracted_dir);
    let mut adjusted_parts = Vec::new();
    let reset_to_original =
        request.offset_x == 0 && request.offset_y == 0 && request.scale_percent == 100;
    for part in target_parts {
        let current_path = extracted_dir.join(format!("{part}.png"));
        if !current_path.is_file() {
            continue;
        }
        let original_path = originals_dir.join(format!("{part}.png"));
        if !original_path.is_file() {
            fs::copy(&current_path, &original_path)?;
        }
        if reset_to_original {
            fs::copy(&original_path, &current_path)?;
            part_adjustments.remove(part);
        } else {
            let original = image::open(&original_path)?;
            let adjusted = transform_extracted_part(
                &original,
                request.offset_x,
                request.offset_y,
                request.scale_percent,
            );
            adjusted.save(&current_path)?;
            part_adjustments.insert(
                part.to_string(),
                json!({
                    "offsetX": request.offset_x,
                    "offsetY": request.offset_y,
                    "scalePercent": request.scale_percent,
                }),
            );
        }
        adjusted_parts.push(part.to_string());
    }
    // 旧バージョンが記録した既定値エントリも未調整として整理する。
    part_adjustments.retain(|_, value| {
        serde_json::from_value::<PartAdjustment>(value.clone())
            .map(|adjustment| !is_default_part_adjustment(&adjustment))
            .unwrap_or(false)
    });
    let active_adjusted_parts = part_adjustments.keys().cloned().collect::<Vec<_>>();
    let manifest_path = extracted_dir.join("adjustment.json");
    fs::write(
        manifest_path,
        serde_json::to_vec_pretty(&json!({
            "formatVersion": 2,
            "offsetX": request.offset_x,
            "offsetY": request.offset_y,
            "scalePercent": request.scale_percent,
            "adjustedParts": active_adjusted_parts,
            "parts": serde_json::Value::Object(part_adjustments.clone().into_iter().collect()),
        }))
        .map_err(|error| AppError::General(format!("adjustment.json作成失敗: {error}")))?,
    )?;
    let typed_adjustments: std::collections::BTreeMap<String, PartAdjustment> = part_adjustments
        .into_iter()
        .filter_map(|(part, value)| {
            serde_json::from_value::<PartAdjustment>(value)
                .ok()
                .map(|adjustment| (part, adjustment))
        })
        .collect();
    if updates_eyes_open {
        let existing_output = rife_output_dir(&job_dir);
        if existing_output.is_dir() {
            sync_dynamic_eye_assets(&base_parts_dir(&job_dir), &extracted_dir, &existing_output)?;
        }
    }
    if manage_workspace_checkpoint {
        complete_workspace_edit(&job_dir, 6)?;
    }

    Ok(AdjustCodexExtractedPartsResult {
        extracted_parts_path: extracted_dir.to_string_lossy().into_owned(),
        adjusted_parts,
        offset_x: request.offset_x,
        offset_y: request.offset_y,
        scale_percent: request.scale_percent,
        part_adjustments: typed_adjustments,
    })
}

fn adjust_codex_extracted_parts_batch_inner(
    request: AdjustCodexExtractedPartsBatchRequest,
) -> Result<AdjustCodexExtractedPartsResult, AppError> {
    if request.adjustments.is_empty() {
        return Err(AppError::General(
            "保存する差分位置の変更がありません".into(),
        ));
    }
    let job_dir = PathBuf::from(&request.job_path);
    let extracted_dir = extracted_parts_dir(&job_dir);
    if !extracted_dir.is_dir() {
        return Err(AppError::General(
            "抽出済みパーツが見つかりません。先にSee-Through一括分解を実行してください".into(),
        ));
    }

    // すべてを先に検証・退避する。途中の1件が失敗しても、完了ボタン1回を部分コミットにしない。
    let mut seen = HashSet::new();
    let mut file_snapshots = Vec::new();
    for adjustment in &request.adjustments {
        if !ADJUSTABLE_PART_TARGETS.contains(&adjustment.part.as_str()) {
            return Err(AppError::General(format!(
                "調整対象外のパーツです: {}",
                adjustment.part
            )));
        }
        if !(50..=150).contains(&adjustment.scale_percent) {
            return Err(AppError::General(
                "scalePercent は 50 から 150 の範囲で指定してください".into(),
            ));
        }
        if !seen.insert(adjustment.part.clone()) {
            return Err(AppError::General(format!(
                "同じパーツが複数指定されています: {}",
                adjustment.part
            )));
        }
        let path = extracted_dir.join(format!("{}.png", adjustment.part));
        if !path.is_file() {
            return Err(AppError::General(format!(
                "調整対象のパーツが見つかりません: {}",
                path.display()
            )));
        }
        file_snapshots.push((path.clone(), fs::read(path)?));
    }
    let manifest_path = extracted_dir.join("adjustment.json");
    let manifest_snapshot = match fs::read(&manifest_path) {
        Ok(bytes) => Some(bytes),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
    };
    let mut output_eye_snapshots = Vec::new();
    if request
        .adjustments
        .iter()
        .any(|adjustment| adjustment.part == "eyes-open")
    {
        let existing_output = rife_output_dir(&job_dir);
        if existing_output.is_dir() {
            for file_name in ["eyebrow.png", "eyewhite.png", "irides.png", "highlight.png"] {
                let path = existing_output.join(file_name);
                let snapshot = match fs::read(&path) {
                    Ok(bytes) => Some(bytes),
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                    Err(error) => return Err(error.into()),
                };
                output_eye_snapshots.push((path, snapshot));
            }
        }
    }

    // 全入力と復元用スナップショットの検証後、最初のPNG書込より先にSTEP5へ戻す。
    invalidate_workspace_before_edit(&job_dir, 5)?;
    let mut last_result = None;
    for adjustment in &request.adjustments {
        match adjust_codex_extracted_parts_core(
            AdjustCodexExtractedPartsRequest {
                job_path: request.job_path.clone(),
                offset_x: adjustment.offset_x,
                offset_y: adjustment.offset_y,
                scale_percent: adjustment.scale_percent,
                part: Some(adjustment.part.clone()),
            },
            false,
        ) {
            Ok(result) => last_result = Some(result),
            Err(error) => {
                if let Err(rollback_error) = restore_part_adjustment_batch(
                    &file_snapshots,
                    &manifest_path,
                    manifest_snapshot.as_deref(),
                    &output_eye_snapshots,
                ) {
                    return Err(AppError::General(format!(
                        "差分位置の保存に失敗し、変更前への復元にも失敗しました: {error} / {rollback_error}"
                    )));
                }
                return Err(error);
            }
        }
    }

    let mut result =
        last_result.ok_or_else(|| AppError::General("差分位置を保存できませんでした".into()))?;
    complete_workspace_edit(&job_dir, 6)?;
    result.adjusted_parts = request
        .adjustments
        .iter()
        .map(|adjustment| adjustment.part.clone())
        .collect();
    Ok(result)
}

fn restore_part_adjustment_batch(
    file_snapshots: &[(PathBuf, Vec<u8>)],
    manifest_path: &Path,
    manifest_snapshot: Option<&[u8]>,
    optional_file_snapshots: &[(PathBuf, Option<Vec<u8>>)],
) -> Result<(), AppError> {
    for (path, bytes) in file_snapshots {
        fs::write(path, bytes)?;
    }
    if let Some(bytes) = manifest_snapshot {
        fs::write(manifest_path, bytes)?;
    } else if manifest_path.is_file() {
        fs::remove_file(manifest_path)?;
    }
    for (path, snapshot) in optional_file_snapshots {
        if let Some(bytes) = snapshot {
            fs::write(path, bytes)?;
        } else if path.is_file() {
            fs::remove_file(path)?;
        }
    }
    Ok(())
}

fn is_default_part_adjustment(adjustment: &PartAdjustment) -> bool {
    adjustment.offset_x == 0 && adjustment.offset_y == 0 && adjustment.scale_percent == 100
}

/// adjustment.json（v2）からパーツごとの位置補正値を読む（STEP5でパーツ切替時の表示用）
fn read_typed_part_adjustments(
    extracted_dir: &Path,
) -> std::collections::BTreeMap<String, PartAdjustment> {
    read_part_adjustments(extracted_dir)
        .into_iter()
        .filter_map(|(part, value)| {
            serde_json::from_value::<PartAdjustment>(value)
                .ok()
                .map(|adjustment| (part, adjustment))
        })
        .collect()
}

/// adjustment.json（v2）からパーツ個別の調整値を読む（生のJSON値のまま）
fn read_part_adjustments(
    extracted_dir: &Path,
) -> std::collections::BTreeMap<String, serde_json::Value> {
    fs::read(extracted_dir.join("adjustment.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .and_then(|manifest| manifest.get("parts").cloned())
        .and_then(|parts| match parts {
            serde_json::Value::Object(map) => Some(map.into_iter().collect()),
            _ => None,
        })
        .unwrap_or_default()
}

fn transform_extracted_part(
    image: &DynamicImage,
    offset_x: i32,
    offset_y: i32,
    scale_percent: u32,
) -> DynamicImage {
    let source = image.to_rgba8();
    let (width, height) = source.dimensions();
    let (scaled, scaled_width, scaled_height) = if scale_percent == 100 {
        (source.clone(), width, height)
    } else {
        let scaled_width = ((width as u64 * scale_percent as u64) / 100).max(1) as u32;
        let scaled_height = ((height as u64 * scale_percent as u64) / 100).max(1) as u32;
        (
            image::imageops::resize(
                &source,
                scaled_width,
                scaled_height,
                image::imageops::FilterType::Lanczos3,
            ),
            scaled_width,
            scaled_height,
        )
    };
    let mut canvas = RgbaImage::new(width, height);
    let origin_x = ((width as i64 - scaled_width as i64) / 2) + offset_x as i64;
    let origin_y = ((height as i64 - scaled_height as i64) / 2) + offset_y as i64;
    for sy in 0..scaled_height {
        let dy = origin_y + sy as i64;
        if dy < 0 || dy >= height as i64 {
            continue;
        }
        for sx in 0..scaled_width {
            let dx = origin_x + sx as i64;
            if dx < 0 || dx >= width as i64 {
                continue;
            }
            canvas.put_pixel(dx as u32, dy as u32, *scaled.get_pixel(sx, sy));
        }
    }
    DynamicImage::ImageRgba8(canvas)
}

fn current_base_parts(app: &AppHandle) -> Option<HashMap<String, DynamicImage>> {
    let state = app.state::<AppState>();
    let parts = state.parts.lock().unwrap();
    if !parts.contains_key("body") {
        return None;
    }
    let mut cloned = HashMap::new();
    for key in [
        "body",
        "neck",
        "hair",
        "hair_back",
        "arm_l",
        "arm_r",
        "chest",
        "eye_open",
        "eyebrow",
        "eyewhite",
        "irides",
        "highlight",
        "mouth_closed",
    ] {
        if let Some(image) = parts.get(key) {
            cloned.insert(key.to_string(), image.clone());
        }
    }
    // 汎用揺れパーツと腕追従オーバーレイはマッピング由来で名前が動的
    for (key, image) in parts.iter() {
        if is_dynamic_base_part_name(key) {
            cloned.insert(key.clone(), image.clone());
        }
    }
    Some(cloned)
}

fn is_dynamic_base_part_name(name: &str) -> bool {
    name.starts_with("sway_") || is_arm_overlay_part_name(name)
}

fn save_base_parts(
    parts: &HashMap<String, DynamicImage>,
    output_dir: &Path,
) -> Result<(), AppError> {
    fs::create_dir_all(output_dir)?;
    for key in [
        "body",
        "neck",
        "hair",
        "hair_back",
        "arm_l",
        "arm_r",
        "chest",
        "eye_open",
        "eyebrow",
        "eyewhite",
        "irides",
        "highlight",
        "mouth_closed",
    ] {
        let path = output_dir.join(format!("{key}.png"));
        if path.is_file() {
            fs::remove_file(&path)?;
        }
        if let Some(image) = parts.get(key) {
            image.save(path)?;
        }
    }
    // 動的パーツ: 旧ファイルを掃除してから今回分を保存する。
    // 腕オーバーレイは本体とは別の前後関係を持つ一方、Motion Labでは親腕へ追従する。
    if let Ok(entries) = fs::read_dir(output_dir) {
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            let is_stale_dynamic_part = path.is_file()
                && path
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("png"))
                && path
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .is_some_and(|stem| is_dynamic_base_part_name(&stem.to_ascii_lowercase()));
            if is_stale_dynamic_part {
                fs::remove_file(path)?;
            }
        }
    }
    for (key, image) in parts.iter() {
        if is_dynamic_base_part_name(key) {
            image.save(output_dir.join(format!("{key}.png")))?;
        }
    }
    Ok(())
}

fn save_eyes_open_extracted_part(
    job_dir: &Path,
    parts: &HashMap<String, DynamicImage>,
    extracted_dir: &Path,
) -> Result<(), AppError> {
    let Some(base_eye) = parts.get("eye_open") else {
        return Ok(());
    };
    let (width, height) = parts
        .get("body")
        .map(|image| (image.width(), image.height()))
        .unwrap_or_else(|| (base_eye.width(), base_eye.height()));
    let eyes_open = if base_eye.width() != width || base_eye.height() != height {
        base_eye.resize_exact(width, height, image::imageops::FilterType::Nearest)
    } else {
        base_eye.clone()
    };
    fs::create_dir_all(extracted_dir)?;
    let existing_adjustment = read_typed_part_adjustments(extracted_dir)
        .get("eyes-open")
        .cloned();
    let displayed_eyes_open = existing_adjustment
        .as_ref()
        .filter(|adjustment| !is_default_part_adjustment(adjustment))
        .map(|adjustment| {
            transform_extracted_part(
                &eyes_open,
                adjustment.offset_x,
                adjustment.offset_y,
                adjustment.scale_percent,
            )
        })
        .unwrap_or_else(|| eyes_open.clone());
    // eyes-open は「素体の目（eye_open）そのもの」を正とする。他フレーム（口パク等）が
    // 表示する平常時の目と完全一致させ、閉じ目だけが差分になるようにする。
    // STEP5で既に位置補正済みなら、その補正を新しい素体の目にも再適用する。
    displayed_eyes_open.save(extracted_dir.join("eyes-open.png"))?;
    let originals_dir = extracted_dir.join("original_extracted_parts");
    fs::create_dir_all(&originals_dir)?;
    eyes_open.save(originals_dir.join("eyes-open.png"))?;
    let base_dir = base_parts_dir(job_dir);
    fs::create_dir_all(&base_dir)?;
    eyes_open.save(base_dir.join("eye_open.png"))?;
    Ok(())
}

fn cache_codex_source_see_through_inner(
    job_path: &str,
    psd_path: &str,
) -> Result<String, AppError> {
    let job_dir = PathBuf::from(job_path);
    if !job_dir.is_dir() {
        return Err(AppError::General(format!(
            "Codexジョブフォルダが見つかりません: {}",
            job_dir.display()
        )));
    }
    let source_psd = PathBuf::from(psd_path);
    if !source_psd.is_file() {
        return Err(AppError::General(format!(
            "See-Through PSDが見つかりません: {}",
            source_psd.display()
        )));
    }
    // source.psdを差し替える前にSTEP3へ戻し、旧抽出/RIFEを再利用させない。
    invalidate_workspace_before_edit(&job_dir, 3)?;
    let output_dir = job_dir.join(WORKSPACE_SEE_THROUGH_DIR);
    fs::create_dir_all(&output_dir)?;
    let cached_psd = output_dir.join("source.psd");
    fs::copy(&source_psd, &cached_psd)?;
    Ok(cached_psd.to_string_lossy().into_owned())
}

fn load_codex_source_see_through_inner(
    app: AppHandle,
    job_path: &str,
) -> Result<WorkspaceSourceSeeThroughResult, AppError> {
    let job_dir = PathBuf::from(job_path);
    if !job_dir.is_dir() {
        return Err(AppError::General(format!(
            "Codexジョブフォルダが見つかりません: {}",
            job_dir.display()
        )));
    }
    let psd_path = job_dir.join(WORKSPACE_SEE_THROUGH_DIR).join("source.psd");
    if !psd_path.is_file() {
        return Err(AppError::General(format!(
            "素体調整用のSee-Through PSDが見つかりません。Step3の一括分解をやり直してください: {}",
            psd_path.display()
        )));
    }
    let psd_path_string = psd_path.to_string_lossy().into_owned();
    let slot_load = load_slot_inner(app.clone(), psd_path_string.clone())?;
    let default_mapping: HashMap<String, String> = slot_load
        .adjustable_layers
        .iter()
        .map(|layer| (layer.name.clone(), layer.default_target.clone()))
        .collect();
    let mapping_json = serde_json::to_string(&default_mapping)
        .map_err(|error| AppError::General(format!("自動分類情報の作成に失敗しました: {error}")))?;
    let mapping_preview = get_mapping_preview_inner(app, mapping_json)?;
    Ok(WorkspaceSourceSeeThroughResult {
        psd_path: psd_path_string,
        output_dir: job_dir
            .join(WORKSPACE_SEE_THROUGH_DIR)
            .to_string_lossy()
            .into_owned(),
        selected_profile: "cached".to_string(),
        slot_load,
        mapping_preview,
    })
}

fn extract_codex_generated_parts_inner(
    app: AppHandle,
    job_path: &str,
    profile: &str,
    split_parts: bool,
    options: Option<see_through::SeeThroughOptions>,
) -> Result<ExtractCodexGeneratedPartsResult, AppError> {
    let job_dir = PathBuf::from(job_path);
    if !job_dir.is_dir() {
        return Err(AppError::General(format!(
            "Codexジョブフォルダが見つかりません: {}",
            job_dir.display()
        )));
    }
    let source_path = job_source_path(&job_dir);
    let source_image = image::open(&source_path).map_err(|error| {
        AppError::General(format!(
            "ジョブ内の source.png を読み込めません: {} ({error})",
            source_path.display()
        ))
    })?;
    let expected_parts = expected_parts_from_job(&job_dir)?;
    let status = inspect_generated_parts(&job_dir, &source_image, &expected_parts)?;
    if !status.ready {
        return Err(AppError::General(
            "Codex生成素材が揃っていません。先に不足とサイズ違いを解消してください".into(),
        ));
    }

    let extracted_dir = extracted_parts_dir(&job_dir);
    fs::create_dir_all(&extracted_dir)?;
    let generated_parts_dir = PathBuf::from(&status.generated_parts_path);
    let mut extracted_parts = Vec::new();
    let mut warnings = Vec::new();
    let total = status.expected_parts.len() as u32;
    let mut effective_profile = profile.to_string();
    let mut effective_options = options;
    let mut effective_split_parts = split_parts;

    // 直前に実行された source の See-Through 分解をスナップショット。
    // 生成素材の分解でグローバル状態が上書きされるため、eyes-open生成・
    // 位置合わせアンカー・後工程（素体調整）のすべてがこのスナップショットを正とする
    let source_snapshot = match snapshot_current_decomposition(&app) {
        Some(snapshot)
            if snapshot.width == source_image.width()
                && snapshot.height == source_image.height() =>
        {
            snapshot
        }
        _ => {
            let source_result = see_through::run_inference(
                &app,
                &source_path.to_string_lossy(),
                &effective_profile,
                effective_split_parts,
                effective_options.clone(),
            )?;
            effective_profile = source_result.selected_profile.clone();
            effective_options = source_result.effective_options.clone();
            if source_result.split_parts_fallback {
                effective_split_parts = false;
                warnings.push(
                    "元画像: 左右パーツ分解に失敗したため、左右分解なしで処理しました".into(),
                );
            }
            if let Some(note) = &source_result.oom_retry_note {
                warnings.push(format!(
                    "元画像: 推論エラーのため自動リトライしました（{note}）"
                ));
            }
            snapshot_current_decomposition(&app).ok_or_else(|| {
                AppError::General("元画像のSee-Through分解結果を取得できません".into())
            })?
        }
    };

    // STEP3を完了済み案件で再実行した場合も、抽出画像を上書きする前にSTEP3へ戻す。
    invalidate_workspace_before_edit(&job_dir, 3)?;

    // eyes-open は常に source から再生成（生成素材の分解結果に依存させない）。
    // 位置補正の対象外なので original_extracted_parts 側も同時に更新する
    regenerate_eyes_open_from_source(
        &source_image,
        &source_snapshot,
        &extracted_dir,
        &mut warnings,
    )?;

    // 位置合わせアンカー: source の目・口レイヤーのアルファ重心。
    // bbox中心より散在ピクセルの影響を受けにくい。
    let source_eye_anchor = extract_named_expression_layers(
        &source_snapshot.layers,
        EYE_ANIMATION_LAYER_NAMES,
        source_snapshot.width,
        source_snapshot.height,
    )
    .as_ref()
    .and_then(alpha_centroid);
    let source_mouth_anchor = extract_named_expression_layers(
        &source_snapshot.layers,
        &["mouth"],
        source_snapshot.width,
        source_snapshot.height,
    )
    .as_ref()
    .and_then(alpha_centroid);

    let previous_alignment = read_extraction_alignment(&extracted_dir);
    let previous_generated_fingerprints =
        read_extraction_generated_part_fingerprints(&extracted_dir);
    let mut alignment = serde_json::Map::new();
    let mut generated_part_fingerprints = serde_json::Map::new();

    for (index, part) in status.expected_parts.iter().enumerate() {
        app.emit(
            "generation-progress",
            ProgressPayload {
                current: index as u32 + 1,
                total,
                pair_name: format!("See-Through {part}"),
            },
        )
        .ok();

        let generated_path = generated_parts_dir.join(format!("{part}.png"));
        let output_path = extracted_dir.join(format!("{part}.png"));
        let generated_image = image::open(&generated_path).map_err(|error| {
            AppError::General(format!(
                "Codex生成素材を読み込めません: {} ({error})",
                generated_path.display()
            ))
        })?;
        let generated_fingerprint = visual_image_fingerprint(&generated_image);
        generated_part_fingerprints.insert(part.clone(), json!(generated_fingerprint.clone()));
        // 位置合わせ済み（alignment記録あり）かつ生成画像の視覚内容が同じ場合だけ再利用可。
        // ファイル時刻はWindowsコピーで維持されるため、再利用判定には使わない。
        if extracted_part_is_fresh(
            part,
            &output_path,
            previous_generated_fingerprints
                .get(part)
                .map(String::as_str),
            &generated_fingerprint,
        ) {
            if let Some(previous) = previous_alignment.get(part.as_str()) {
                alignment.insert(part.clone(), previous.clone());
                extracted_parts.push(part.clone());
                continue;
            }
        }

        let see_through_result = see_through::run_inference(
            &app,
            &generated_path.to_string_lossy(),
            &effective_profile,
            effective_split_parts,
            effective_options.clone(),
        )?;
        effective_profile = see_through_result.selected_profile.clone();
        effective_options = see_through_result.effective_options.clone();
        if see_through_result.split_parts_fallback {
            effective_split_parts = false;
            warnings.push(format!(
                "{part}: 左右パーツ分解に失敗したため、左右分解なしで処理しました"
            ));
        }
        if let Some(note) = &see_through_result.oom_retry_note {
            warnings.push(format!(
                "{part}: 推論エラーのため自動リトライしました（{note}）"
            ));
        }
        let state = app.state::<AppState>();
        let layers = state
            .slot_layers
            .lock()
            .unwrap()
            .get("current")
            .cloned()
            .ok_or_else(|| AppError::General("See-Through分解レイヤーを取得できません".into()))?;
        let width = *state.canvas_width.lock().unwrap();
        let height = *state.canvas_height.lock().unwrap();
        if width == 0 || height == 0 {
            return Err(AppError::General(
                "See-Through分解結果のキャンバスサイズが不正です".into(),
            ));
        }
        // 生成素材を作業解像度へ正規化（マスク切り出し・差分フォールバックの前提）
        let generated_image =
            if generated_image.width() != width || generated_image.height() != height {
                generated_image.resize_exact(width, height, image::imageops::FilterType::Lanczos3)
            } else {
                generated_image
            };
        let is_eye_part = part.starts_with("eyes-");
        // 1) See-Throughの名前付きレイヤー（eyewhite/irides等 or mouth）から抽出
        let extracted = extract_named_expression_layers(
            &layers,
            if is_eye_part {
                EYE_ANIMATION_LAYER_NAMES
            } else {
                &["mouth"]
            },
            width,
            height,
        )
        // 2) 名前付きが無ければ、See-Throughのマスクで生成素材から切り出し
        .or_else(|| {
            let mut mask = if is_eye_part {
                expression_mask(&layers, EYE_ANIMATION_LAYER_NAMES, width, height, 12, 3)
            } else {
                mouth_expression_mask(&layers, width, height)
            };
            if !is_eye_part {
                mask = refine_mouth_mask_with_difference(
                    &mask,
                    &source_image,
                    &generated_image,
                    width,
                    height,
                );
            }
            let min_area = if is_eye_part { 80 } else { 40 };
            if !mask_has_minimum_edit_area(&mask, min_area) {
                return None;
            }
            Some(cut_image_with_mask(&generated_image, &mask))
        })
        // 3) それでも失敗したら、元画像との差分領域から直接切り出す（See-Through非依存の保険）
        .or_else(|| {
            difference_fallback_extract(&source_image, &generated_image, width, height, is_eye_part)
        });
        let Some(extracted) = extracted else {
            warnings.push(format!(
                "{part}: 目/口の領域を抽出できず、この表情はスキップしました（元画像が分解しづらい可能性があります）"
            ));
            continue;
        };

        // 出力は source 解像度に正規化（生成素材が同アスペクト別解像度でも後段が揃う）
        let extracted = if extracted.width() != source_snapshot.width
            || extracted.height() != source_snapshot.height
        {
            extracted.resize_exact(
                source_snapshot.width,
                source_snapshot.height,
                image::imageops::FilterType::Lanczos3,
            )
        } else {
            extracted
        };

        // 自動位置合わせ: 抽出パーツのalpha bbox中心を source の対応領域中心へ平行移動
        let anchor = if part.starts_with("eyes-") {
            source_eye_anchor
        } else {
            source_mouth_anchor
        };
        let (aligned, dx, dy) =
            align_extracted_to_anchor(&extracted, anchor, source_snapshot.width);
        aligned.save(&output_path)?;
        alignment.insert(part.clone(), json!({ "dx": dx, "dy": dy }));
        extracted_parts.push(part.clone());
    }

    // 生成素材の分解で上書きされた状態を source の分解結果へ戻す
    // （この後の素体調整が source を対象に動くようにする）
    restore_decomposition_snapshot(&app, &source_snapshot);

    // eyes-open は生成素材ではなく source から別経路で作るため、通常の抽出ループには
    // 含まれない。実ファイルが作成（または有効な既存ファイルが維持）されていれば、
    // manifest と即時レスポンスの一覧にも必ず反映する。
    include_existing_eyes_open(&extracted_dir, &mut extracted_parts);

    let manifest = json!({
        "formatVersion": 3,
        "mode": "codex-generated-parts-extracted",
        "sourceJob": job_dir.to_string_lossy(),
        "extractedPartsDirectory": extracted_dir.to_string_lossy(),
        "extractedParts": extracted_parts,
        "alignment": serde_json::Value::Object(alignment),
        "generatedPartFingerprints": serde_json::Value::Object(generated_part_fingerprints),
        "eyesOpenSource": "source-image",
        "warnings": warnings,
    });
    fs::write(
        extracted_dir.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)
            .map_err(|error| AppError::General(format!("抽出manifest作成失敗: {error}")))?,
    )?;
    if ADJUSTABLE_PART_TARGETS
        .iter()
        .all(|required| extracted_parts.iter().any(|part| part == required))
    {
        complete_workspace_edit(&job_dir, 4)?;
    }

    Ok(ExtractCodexGeneratedPartsResult {
        extracted_parts_path: extracted_dir.to_string_lossy().into_owned(),
        extracted_parts,
        warnings,
        selected_profile: effective_profile,
        effective_options,
        split_parts: effective_split_parts,
        part_adjustments: read_typed_part_adjustments(&extracted_dir),
    })
}

struct DecompositionSnapshot {
    layers: HashMap<String, DynamicImage>,
    order: Vec<String>,
    depth_maps: HashMap<String, GrayImage>,
    width: u32,
    height: u32,
}

fn snapshot_current_decomposition(app: &AppHandle) -> Option<DecompositionSnapshot> {
    let state = app.state::<AppState>();
    let layers = state.slot_layers.lock().unwrap().get("current").cloned()?;
    if layers.is_empty() {
        return None;
    }
    let width = *state.canvas_width.lock().unwrap();
    let height = *state.canvas_height.lock().unwrap();
    if width == 0 || height == 0 {
        return None;
    }
    let order = state.slot_layer_order.lock().unwrap().clone();
    let depth_maps = state.slot_depth_maps.lock().unwrap().clone();
    Some(DecompositionSnapshot {
        layers,
        order,
        depth_maps,
        width,
        height,
    })
}

fn restore_decomposition_snapshot(app: &AppHandle, snapshot: &DecompositionSnapshot) {
    let state = app.state::<AppState>();
    state
        .slot_layers
        .lock()
        .unwrap()
        .insert("current".to_string(), snapshot.layers.clone());
    *state.slot_layer_order.lock().unwrap() = snapshot.order.clone();
    *state.slot_depth_maps.lock().unwrap() = snapshot.depth_maps.clone();
    *state.canvas_width.lock().unwrap() = snapshot.width;
    *state.canvas_height.lock().unwrap() = snapshot.height;
}

fn include_existing_eyes_open(extracted_dir: &Path, extracted_parts: &mut Vec<String>) {
    if extracted_dir.join("eyes-open.png").is_file()
        && !extracted_parts.iter().any(|part| part == "eyes-open")
    {
        extracted_parts.insert(0, "eyes-open".to_string());
    }
}

/// eyes-open.png を元画像から再生成する。RIFEの開き目始点は常に source 由来とし、
/// 生成素材（閉じ目）の分解結果が混入しないようにする
fn regenerate_eyes_open_from_source(
    source_image: &DynamicImage,
    snapshot: &DecompositionSnapshot,
    extracted_dir: &Path,
    warnings: &mut Vec<String>,
) -> Result<(), AppError> {
    let mask = expression_mask(
        &snapshot.layers,
        EYE_ANIMATION_LAYER_NAMES,
        snapshot.width,
        snapshot.height,
        12,
        3,
    );
    if !mask_has_minimum_edit_area(&mask, 80) {
        warnings.push(
            "eyes-open: 元画像の目レイヤーを十分に抽出できませんでした。既存のeyes-open.pngを維持します".into(),
        );
        return Ok(());
    }
    // サニティ: マスクが広すぎる場合は目領域として信用しない
    // （分解結果の目レイヤーが全面に漏れると eyes-open がフル画像化して以降の合成が壊れる）
    let mask_area = mask.pixels().filter(|pixel| pixel[0] > 0).count() as u64;
    let canvas_area = (snapshot.width as u64) * (snapshot.height as u64);
    let mask_bbox = mask_bounds(mask.as_raw(), snapshot.width, snapshot.height);
    let mask_too_large = mask_area * 100 > canvas_area * 15
        || mask_bbox.is_some_and(|(min_x, min_y, max_x, max_y)| {
            (max_x - min_x) > snapshot.width * 2 / 3 || (max_y - min_y) > snapshot.height / 2
        });
    if mask_too_large {
        warnings.push(
            "eyes-open: 目マスクが広すぎるため再生成をスキップしました。See-Through分解の目レイヤーを確認してください".into(),
        );
        return Ok(());
    }
    let source_resized =
        if source_image.width() != snapshot.width || source_image.height() != snapshot.height {
            source_image.resize_exact(
                snapshot.width,
                snapshot.height,
                image::imageops::FilterType::Lanczos3,
            )
        } else {
            source_image.clone()
        };
    // 抽出時点の暫定 eyes-open（base素体が未保存の段階の早期プレビュー用）。
    // STEP4の素体保存（save_eyes_open_extracted_part）で base の eye_open に上書きされ、
    // 最終的には平常時の目と完全一致する
    let eyes_open = cut_image_with_mask(&source_resized, &mask);
    eyes_open.save(extracted_dir.join("eyes-open.png"))?;
    let originals_dir = extracted_dir.join("original_extracted_parts");
    fs::create_dir_all(&originals_dir)?;
    eyes_open.save(originals_dir.join("eyes-open.png"))?;
    Ok(())
}

/// 既存 manifest.json から位置合わせ記録を読む（無ければ空 = 全パーツ再処理）
fn read_extraction_alignment(extracted_dir: &Path) -> serde_json::Map<String, serde_json::Value> {
    fs::read(extracted_dir.join("manifest.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .and_then(|manifest| manifest.get("alignment").cloned())
        .and_then(|alignment| match alignment {
            serde_json::Value::Object(map) => Some(map),
            _ => None,
        })
        .unwrap_or_default()
}

fn read_extraction_generated_part_fingerprints(
    extracted_dir: &Path,
) -> std::collections::BTreeMap<String, String> {
    fs::read(extracted_dir.join("manifest.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .and_then(|manifest| manifest.get("generatedPartFingerprints").cloned())
        .and_then(|fingerprints| match fingerprints {
            serde_json::Value::Object(map) => Some(map),
            _ => None,
        })
        .map(|fingerprints| {
            fingerprints
                .into_iter()
                .filter_map(|(part, value)| value.as_str().map(|value| (part, value.to_string())))
                .collect()
        })
        .unwrap_or_default()
}

/// 抽出パーツのアルファ重心を source 側アンカー（レイヤー重心）へ平行移動する。
/// アルファ重心はbbox中心より散在ピクセルに引きずられにくい。
/// ずれ量が width/6 を超える場合は検出不良とみなし補正しない
fn align_extracted_to_anchor(
    image: &DynamicImage,
    anchor: Option<(f64, f64)>,
    width: u32,
) -> (DynamicImage, i32, i32) {
    let Some((anchor_x, anchor_y)) = anchor else {
        return (image.clone(), 0, 0);
    };
    let Some((center_x, center_y)) = alpha_centroid(image) else {
        return (image.clone(), 0, 0);
    };
    let limit = (width as i32 / 6).max(1);
    let dx = (anchor_x - center_x).round() as i32;
    let dy = (anchor_y - center_y).round() as i32;
    if dx.abs() > limit || dy.abs() > limit {
        return (image.clone(), 0, 0);
    }
    if dx == 0 && dy == 0 {
        return (image.clone(), 0, 0);
    }
    (transform_extracted_part(image, dx, dy, 100), dx, dy)
}

/// アルファ加重の重心（位置合わせアンカー用）
fn alpha_centroid(image: &DynamicImage) -> Option<(f64, f64)> {
    let rgba = image.to_rgba8();
    let mut weight_sum = 0f64;
    let mut x_sum = 0f64;
    let mut y_sum = 0f64;
    for (x, y, pixel) in rgba.enumerate_pixels() {
        let alpha = pixel[3] as f64;
        if alpha > 8.0 {
            weight_sum += alpha;
            x_sum += x as f64 * alpha;
            y_sum += y as f64 * alpha;
        }
    }
    (weight_sum > 0.0).then(|| (x_sum / weight_sum, y_sum / weight_sum))
}

fn cut_image_with_mask(image: &DynamicImage, mask: &GrayImage) -> DynamicImage {
    let mut rgba = image.to_rgba8();
    for (x, y, pixel) in rgba.enumerate_pixels_mut() {
        let mask_alpha = mask.get_pixel(x, y)[0] as u16;
        pixel[3] = ((pixel[3] as u16 * mask_alpha) / 255) as u8;
    }
    DynamicImage::ImageRgba8(rgba)
}

fn preview_codex_composite_inner(
    app: AppHandle,
    job_path: &str,
    profile: &str,
) -> Result<PreviewCodexCompositeResult, AppError> {
    let job_dir = PathBuf::from(job_path);
    let early_extracted_dir = extracted_parts_dir(&job_dir);
    let include_expression_previews = profile != "base-only";
    if !early_extracted_dir.is_dir() {
        return Err(AppError::General(
            "extracted_parts が見つかりません。先に生成素材をSee-Throughで分解してください".into(),
        ));
    }
    if let Some(base_parts) = load_job_base_parts(&job_dir)? {
        return preview_from_base_parts(
            &base_parts,
            &early_extracted_dir,
            &base_parts_dir(&job_dir),
            include_expression_previews,
        );
    }
    if let Some(base_parts) = current_base_parts(&app) {
        let base_parts_dir = base_parts_dir(&job_dir);
        save_base_parts(&base_parts, &base_parts_dir)?;
        save_layer_draw_order(&app, &base_parts_dir)?;
        return preview_from_base_parts(
            &base_parts,
            &early_extracted_dir,
            &base_parts_dir,
            include_expression_previews,
        );
    }
    // 素体データが無い場合でも、ここで暗黙にSee-Through推論へフォールバックしない。
    // 推論はSTEP3の「一括分解を開始」ボタンだけがトリガー（つづきから復帰時に
    // プレビュー再構築経由で分解処理が勝手に走る事故の防止）
    Err(AppError::General(
        "素体データ（base_parts）が見つかりません。STEP4の素体調整を開いて保存すると合成プレビューを表示できます".into(),
    ))
}

fn load_job_base_parts(job_dir: &Path) -> Result<Option<HashMap<String, DynamicImage>>, AppError> {
    let base_dir = base_parts_dir(job_dir);
    if !base_dir.is_dir() {
        return Ok(None);
    }
    let mut parts = HashMap::new();
    for key in [
        "body",
        "neck",
        "hair",
        "hair_back",
        "arm_l",
        "arm_r",
        "chest",
        "eye_open",
        "eyebrow",
        "eyewhite",
        "irides",
        "highlight",
        "mouth_closed",
    ] {
        let path = base_dir.join(format!("{key}.png"));
        if path.is_file() {
            let image = image::open(&path).map_err(|error| {
                AppError::General(format!(
                    "base_parts を読み込めません: {} ({error})",
                    path.display()
                ))
            })?;
            parts.insert(key.to_string(), image);
        }
    }
    // 汎用揺れパーツ（sway_*、獣耳等）と腕追従オーバーレイも合成対象に含める
    if let Ok(entries) = fs::read_dir(&base_dir) {
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
                continue;
            };
            let is_dynamic_part_png = path.is_file()
                && is_dynamic_base_part_name(&stem.to_ascii_lowercase())
                && path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("png"));
            if is_dynamic_part_png {
                if let Ok(image) = image::open(&path) {
                    parts.insert(stem.to_string(), image);
                }
            }
        }
    }
    Ok(parts.contains_key("body").then_some(parts))
}

/// base_parts/layer-order.json のグループ描画順を読む（無ければ空=既定順）
fn read_base_layer_order(base_dir: &Path) -> Vec<String> {
    let path = base_dir.join("layer-order.json");
    let Ok(bytes) = fs::read(&path) else {
        return Vec::new();
    };
    serde_json::from_slice::<serde_json::Value>(&bytes)
        .ok()
        .and_then(|value| {
            value.get("drawOrder").and_then(|order| {
                order.as_array().map(|entries| {
                    entries
                        .iter()
                        .filter_map(|entry| entry.as_str().map(str::to_string))
                        .collect()
                })
            })
        })
        .unwrap_or_default()
}

/// layer-order.json の順序に既定順の不足グループ（neck/sways等）を補完する。
/// アルゴリズムはフロント drawMotionLabOrderedLayers と同じ（既定順で直前の要素の直後へ挿入）
fn resolve_base_draw_order(custom: &[String]) -> Vec<String> {
    const DEFAULT: [&str; 10] = [
        "hair_back",
        "body",
        "neck",
        "chest",
        "arm_l",
        "arm_r",
        "sways",
        "eye",
        "mouth",
        "hair",
    ];
    let mut order = Vec::new();
    let mut seen = HashSet::new();
    for key in custom {
        // 動的パーツはSTEP4で個別に前後関係を持つため、固定グループへ潰さず保持する。
        if (DEFAULT.contains(&key.as_str()) || is_dynamic_base_part_name(key))
            && seen.insert(key.clone())
        {
            order.push(key.clone());
        }
    }
    for (index, key) in DEFAULT.iter().enumerate() {
        if order.iter().any(|entry| entry == key) {
            continue;
        }
        let mut insert_at = order.len();
        for prev_index in (0..index).rev() {
            if let Some(pos) = order.iter().position(|entry| entry == DEFAULT[prev_index]) {
                insert_at = pos + 1;
                break;
            }
        }
        order.insert(insert_at, key.to_string());
    }
    order
}

fn preview_from_base_parts(
    base_parts: &HashMap<String, DynamicImage>,
    extracted_dir: &Path,
    base_dir: &Path,
    include_expression_previews: bool,
) -> Result<PreviewCodexCompositeResult, AppError> {
    let body = base_parts
        .get("body")
        .ok_or_else(|| AppError::General("base_parts/body.png が見つかりません".into()))?;
    let width = body.width();
    let height = body.height();
    // Step4のレイヤー調整で決めたグループ描画順（layer-order.json）を尊重する
    let draw_order = resolve_base_draw_order(&read_base_layer_order(base_dir));
    let mut part_names = Vec::new();
    if include_expression_previews {
        if extracted_dir.join("eyes-open.png").is_file() {
            part_names.push("eyes-open".to_string());
        }
        if extracted_dir.join("eyes-closed.png").is_file() {
            part_names.push("eyes-closed".to_string());
        }
        if extracted_dir.join("mouth-closed.png").is_file() {
            part_names.push("mouth-closed".to_string());
        }
        for part in MOUTH_VOWEL_TARGETS {
            if extracted_dir.join(format!("{part}.png")).is_file() {
                part_names.push((*part).to_string());
            }
        }
    }

    let independent_eyebrow = base_parts
        .get("eyebrow")
        .map(|eyebrow| image_utils::eyebrow_cleanup_mask(eyebrow, width, height));
    let clean_eye_overlay = |name: &str, image: RgbaImage| -> RgbaImage {
        if !name.starts_with("eyes-") {
            return image;
        }
        independent_eyebrow
            .as_ref()
            .map(|eyebrow| {
                erase_alpha_with_mask(&DynamicImage::ImageRgba8(image.clone()), eyebrow).to_rgba8()
            })
            .unwrap_or(image)
    };

    // 選択中パーツだけの「のっぺらぼう」表示にしない。目を調整中は閉じ口、口を
    // 調整中は開き目を確認用として重ねる。current PNGを読むため、STEP5の補正も反映される。
    // 独立眉がある案件では、旧目画像に焼き込まれた眉を消してから一枚だけ重ねる。
    let load_companion = |name: &str| -> Result<Option<RgbaImage>, AppError> {
        let path = extracted_dir.join(format!("{name}.png"));
        if !path.is_file() {
            return Ok(None);
        }
        let image = image::open(&path).map_err(|error| {
            AppError::General(format!(
                "確認用の差分パーツを読み込めません: {} ({error})",
                path.display()
            ))
        })?;
        Ok(Some(clean_eye_overlay(
            name,
            resized_rgba(&image, width, height),
        )))
    };
    let default_eye = include_expression_previews
        .then(|| load_companion("eyes-open"))
        .transpose()?
        .flatten();
    let default_mouth = include_expression_previews
        .then(|| load_companion("mouth-closed"))
        .transpose()?
        .flatten();

    let mut previews = Vec::new();
    for part in part_names {
        let part_path = extracted_dir.join(format!("{part}.png"));
        let part_image = image::open(&part_path).map_err(|error| {
            AppError::General(format!(
                "extracted part を読み込めません: {} ({error})",
                part_path.display()
            ))
        })?;
        let part_rgba = clean_eye_overlay(&part, resized_rgba(&part_image, width, height));
        let is_eye_part = part.starts_with("eyes-");
        let eye_overlay = if is_eye_part {
            Some(&part_rgba)
        } else {
            default_eye.as_ref()
        };
        let mouth_overlay = if is_eye_part {
            default_mouth.as_ref()
        } else {
            Some(&part_rgba)
        };
        let composite = compose_base_parts_ordered(
            base_parts,
            width,
            height,
            &draw_order,
            eye_overlay,
            mouth_overlay,
            true,
        );
        previews.push(CodexCompositePreviewItem {
            part,
            preview: image_data_url(&DynamicImage::ImageRgba8(composite))?,
        });
    }

    Ok(PreviewCodexCompositeResult {
        // base = 直接編集キャンバスの土台。目・口は乗せず、独立眉などの静的パーツは含める。
        base_preview: image_data_url(&DynamicImage::ImageRgba8(compose_base_parts_ordered(
            base_parts,
            width,
            height,
            &draw_order,
            None,
            None,
            false,
        )))?,
        previews,
    })
}

/// base_parts をグループ描画順（layer-order.json由来、背面→前面）で合成する。
/// eye/mouth スロットは overlay 指定時にそれで置き換え（差分パーツ・RIFEフレーム用）。
/// overlay が None のスロットは include_face_defaults=true なら eye_open / mouth_closed を、
/// false なら目・口は描かない。独立眉は直接編集キャンバスの静的土台として常に描く。
fn compose_base_parts_ordered(
    base_parts: &HashMap<String, DynamicImage>,
    width: u32,
    height: u32,
    order: &[String],
    eye_overlay: Option<&RgbaImage>,
    mouth_overlay: Option<&RgbaImage>,
    include_face_defaults: bool,
) -> RgbaImage {
    let mut result = RgbaImage::new(width, height);
    let explicitly_ordered_sways: HashSet<&str> = order
        .iter()
        .map(String::as_str)
        .filter(|key| key.starts_with("sway_"))
        .collect();
    let draw = |result: &mut RgbaImage, image: &DynamicImage| {
        alpha_composite_onto(result, &resized_rgba(image, width, height), width, height);
    };
    for key in order {
        match key.as_str() {
            "eye" => {
                if let Some(overlay) = eye_overlay {
                    alpha_composite_onto(&mut result, overlay, width, height);
                } else if include_face_defaults {
                    if let Some(image) = base_parts.get("eye_open") {
                        draw(&mut result, image);
                    }
                }
                if let Some(image) = base_parts.get("eyebrow") {
                    draw(&mut result, image);
                }
            }
            "mouth" => {
                if let Some(overlay) = mouth_overlay {
                    alpha_composite_onto(&mut result, overlay, width, height);
                } else if include_face_defaults {
                    if let Some(image) = base_parts.get("mouth_closed") {
                        draw(&mut result, image);
                    }
                }
            }
            "sways" => {
                let mut sway_keys: Vec<&String> = base_parts
                    .keys()
                    // 旧layer-order.jsonの`sways`は後方互換用。個別指定済みの
                    // swayはその位置で描画されるため、ここでは未指定分だけを補完する。
                    .filter(|key| {
                        key.starts_with("sway_") && !explicitly_ordered_sways.contains(key.as_str())
                    })
                    .collect();
                sway_keys.sort();
                for sway_key in sway_keys {
                    if let Some(image) = base_parts.get(sway_key) {
                        draw(&mut result, image);
                    }
                }
            }
            _ => {
                if let Some(image) = base_parts.get(key.as_str()) {
                    draw(&mut result, image);
                }
            }
        }
    }
    result
}

fn resized_rgba(image: &DynamicImage, width: u32, height: u32) -> RgbaImage {
    if image.width() != width || image.height() != height {
        image
            .resize_exact(width, height, image::imageops::FilterType::Lanczos3)
            .to_rgba8()
    } else {
        image.to_rgba8()
    }
}

fn image_data_url(image: &DynamicImage) -> Result<String, AppError> {
    Ok(format!(
        "data:image/png;base64,{}",
        STANDARD.encode(png_bytes(image)?)
    ))
}

fn extract_named_expression_layers(
    layers: &HashMap<String, DynamicImage>,
    names: &[&str],
    width: u32,
    height: u32,
) -> Option<DynamicImage> {
    let mut result = RgbaImage::new(width, height);
    let mut found = false;
    for target_name in names {
        let mut matching_layers: Vec<_> = layers
            .iter()
            .filter(|(layer_name, _)| normalize_layer_name(layer_name) == *target_name)
            .collect();
        matching_layers.sort_by(|(left, _), (right, _)| left.cmp(right));
        for (_, image) in matching_layers {
            let rgba = if image.width() != width || image.height() != height {
                image
                    .resize_exact(width, height, image::imageops::FilterType::Lanczos3)
                    .to_rgba8()
            } else {
                image.to_rgba8()
            };
            if !rgba.pixels().any(|pixel| pixel[3] > 0) {
                continue;
            }
            alpha_composite_onto(&mut result, &rgba, width, height);
            found = true;
        }
    }
    found.then_some(DynamicImage::ImageRgba8(result))
}

fn alpha_composite_onto(dst: &mut RgbaImage, src: &RgbaImage, width: u32, height: u32) {
    for y in 0..height {
        for x in 0..width {
            let sp = src.get_pixel(x, y);
            let sa = sp[3] as f32 / 255.0;
            if sa <= 0.0 {
                continue;
            }
            let dp = dst.get_pixel(x, y);
            let da = dp[3] as f32 / 255.0;
            let out_a = sa + da * (1.0 - sa);
            if out_a <= 0.0 {
                continue;
            }
            let out = image::Rgba([
                ((sp[0] as f32 * sa + dp[0] as f32 * da * (1.0 - sa)) / out_a).clamp(0.0, 255.0)
                    as u8,
                ((sp[1] as f32 * sa + dp[1] as f32 * da * (1.0 - sa)) / out_a).clamp(0.0, 255.0)
                    as u8,
                ((sp[2] as f32 * sa + dp[2] as f32 * da * (1.0 - sa)) / out_a).clamp(0.0, 255.0)
                    as u8,
                (out_a * 255.0).clamp(0.0, 255.0) as u8,
            ]);
            dst.put_pixel(x, y, out);
        }
    }
}

fn extracted_part_is_fresh(
    part: &str,
    output_path: &Path,
    previous_generated_fingerprint: Option<&str>,
    current_generated_fingerprint: &str,
) -> bool {
    let Ok(output_meta) = fs::metadata(output_path) else {
        return false;
    };
    if output_meta.len() == 0 {
        return false;
    }
    if part.starts_with("eyes-") {
        return false;
    }
    if part.starts_with("mouth-") && !mouth_extracted_alpha_is_reasonable(output_path) {
        return false;
    }
    previous_generated_fingerprint == Some(current_generated_fingerprint)
}

fn mouth_extracted_alpha_is_reasonable(path: &Path) -> bool {
    let Ok(image) = image::open(path) else {
        return false;
    };
    let rgba = image.to_rgba8();
    let (width, height) = rgba.dimensions();
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0;
    let mut max_y = 0;
    let mut count = 0usize;
    for y in 0..height {
        for x in 0..width {
            if rgba.get_pixel(x, y)[3] == 0 {
                continue;
            }
            count += 1;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }
    if count < 20 {
        return false;
    }
    let bounds_width = max_x.saturating_sub(min_x).saturating_add(1);
    let bounds_height = max_y.saturating_sub(min_y).saturating_add(1);
    bounds_width <= (width / 5).max(24) && bounds_height <= (height / 7).max(24)
}

fn generate_codex_rife_outputs_inner(
    app: AppHandle,
    job_path: &str,
    frame_count: u32,
    profile: &str,
) -> Result<GenerateCodexRifeOutputResult, AppError> {
    if !(2..=30).contains(&frame_count) {
        return Err(AppError::General(
            "RIFEフレーム数は2〜30の範囲で指定してください".into(),
        ));
    }
    let job_dir = PathBuf::from(job_path);
    if !job_dir.is_dir() {
        return Err(AppError::General(format!(
            "Codexジョブフォルダが見つかりません: {}",
            job_dir.display()
        )));
    }
    let extracted_dir = extracted_parts_dir(&job_dir);
    if !extracted_dir.is_dir() {
        return Err(AppError::General(
            "extracted_parts が見つかりません。先に生成素材をSee-Throughで分解してください".into(),
        ));
    }
    ensure_workspace_base_parts_ready(&job_dir)?;
    // RIFE出力や補助画像へ触れる前にSTEP6へ戻し、途中終了時も部分出力を復活させない。
    invalidate_workspace_before_edit(&job_dir, 6)?;
    ensure_eyes_open_part(&app, &job_dir, &extracted_dir, profile)?;

    let output_root = rife_output_dir(&job_dir);
    fs::create_dir_all(&output_root)?;
    let jobs = codex_rife_jobs(&extracted_dir);
    if jobs.is_empty() {
        return Err(AppError::General(
            "RIFE出力できる目・口パーツの組み合わせがありません".into(),
        ));
    }

    {
        let state = app.state::<AppState>();
        let mut session = state.rife_session.lock().unwrap();
        if session.is_none() {
            let model_path = resolve_model_path(&app, "rife.onnx")?;
            *session = Some(create_session(&model_path)?);
        }
    }

    let total = jobs.len() as u32 * frame_count;
    let ratios: Vec<f32> = (0..frame_count)
        .map(|index| index as f32 / (frame_count - 1) as f32)
        .collect();
    let mut done = 0u32;
    let mut directories = Vec::new();
    let state = app.state::<AppState>();
    let mut session_guard = state.rife_session.lock().unwrap();
    let session = session_guard.as_mut().unwrap();
    let body_image = image::open(base_parts_dir(&job_dir).join("body.png"))?;
    let eyebrow_image = image::open(base_parts_dir(&job_dir).join("eyebrow.png")).ok();
    let part_adjustments = read_typed_part_adjustments(&extracted_dir);

    for (folder, start_name, end_name) in jobs {
        let mut start = image::open(extracted_dir.join(format!("{start_name}.png")))?;
        let mut end = image::open(extracted_dir.join(format!("{end_name}.png")))?;
        // New extractions omit eyebrows. This also cleans legacy cached eye
        // parts so adding eyebrow.png never produces a fixed ghost underneath.
        if folder == "eye" {
            if let Some(eyebrow) = eyebrow_image.as_ref() {
                let adjusted_mask = |part_name: &str| {
                    part_adjustments
                        .get(part_name)
                        .filter(|adjustment| !is_default_part_adjustment(adjustment))
                        .map(|adjustment| {
                            transform_extracted_part(
                                eyebrow,
                                adjustment.offset_x,
                                adjustment.offset_y,
                                adjustment.scale_percent,
                            )
                        })
                        .unwrap_or_else(|| eyebrow.clone())
                };
                let start_mask = image_utils::eyebrow_cleanup_mask(
                    &adjusted_mask(start_name),
                    start.width(),
                    start.height(),
                );
                let end_mask = image_utils::eyebrow_cleanup_mask(
                    &adjusted_mask(end_name),
                    end.width(),
                    end.height(),
                );
                start = erase_alpha_with_mask(&start, &start_mask);
                end = erase_alpha_with_mask(&end, &end_mask);
            }
        }
        if start.width() != end.width() || start.height() != end.height() {
            return Err(AppError::General(format!(
                "RIFE入力サイズが一致しません: {start_name}.png と {end_name}.png"
            )));
        }
        let width = start.width();
        let height = start.height();
        let body_rgb = body_rgb_for_canvas(&body_image, width, height);
        let start_rgba = start.to_rgba8();
        let end_rgba = end.to_rgba8();
        let start_for_rife = premultiply_onto_body(&body_rgb, &start_rgba, width, height);
        let end_for_rife = premultiply_onto_body(&body_rgb, &end_rgba, width, height);
        let out_dir = output_root.join(folder);
        recreate_png_dir(&out_dir)?;
        for (index, &ratio) in ratios.iter().enumerate() {
            done += 1;
            app.emit(
                "generation-progress",
                ProgressPayload {
                    current: done,
                    total,
                    pair_name: format!("RIFE {folder}"),
                },
            )
            .ok();
            let frame = if index == 0 {
                start.clone()
            } else if index + 1 == frame_count as usize {
                end.clone()
            } else {
                let interpolated =
                    rife_interpolate(session, &start_for_rife, &end_for_rife, ratio)?;
                extract_part_from_body_composite(
                    &interpolated,
                    &body_rgb,
                    &start_rgba,
                    &end_rgba,
                    ratio,
                    width,
                    height,
                )
            };
            frame.save(out_dir.join(format!("{:03}.png", index + 1)))?;
        }
        directories.push(out_dir.to_string_lossy().into_owned());
    }

    let spritalk_assets =
        materialize_spritalk_static_assets(&job_dir, &extracted_dir, &output_root)?;

    let manifest = json!({
        "formatVersion": 1,
        "mode": "codex-rife-output",
        "sourceJob": job_dir.to_string_lossy(),
        "frameCount": frame_count,
        "spritalkImportDirectory": output_root.to_string_lossy(),
        "staticAssets": spritalk_assets,
        "directories": directories,
    });
    fs::write(
        output_root.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)
            .map_err(|error| AppError::General(format!("RIFE manifest作成失敗: {error}")))?,
    )?;
    complete_workspace_edit(&job_dir, 7)?;

    Ok(GenerateCodexRifeOutputResult {
        output_path: output_root.to_string_lossy().into_owned(),
        directories,
        frame_count,
    })
}

fn body_rgb_for_canvas(body: &DynamicImage, width: u32, height: u32) -> image::RgbImage {
    if body.width() == width && body.height() == height {
        body.to_rgb8()
    } else {
        body.resize_exact(width, height, image::imageops::FilterType::Lanczos3)
            .to_rgb8()
    }
}

fn erase_alpha_with_mask(image: &DynamicImage, mask: &DynamicImage) -> DynamicImage {
    image_utils::subtract_alpha_mask(image, mask)
}

fn codex_rife_jobs(extracted_dir: &Path) -> Vec<(&'static str, &'static str, &'static str)> {
    let exists = |name: &str| extracted_dir.join(format!("{name}.png")).is_file();
    let mut jobs = Vec::new();
    if exists("eyes-open") && exists("eyes-closed") {
        jobs.push(("eye", "eyes-open", "eyes-closed"));
    }
    if exists("mouth-closed") {
        for (folder, target) in [
            ("mouth_a", "mouth-a"),
            ("mouth_i", "mouth-i"),
            ("mouth_u", "mouth-u"),
            ("mouth_e", "mouth-e"),
            ("mouth_o", "mouth-o"),
        ] {
            if exists(target) {
                jobs.push((folder, "mouth-closed", target));
            }
        }
    }
    jobs
}

fn materialize_spritalk_static_assets(
    job_dir: &Path,
    extracted_dir: &Path,
    output_root: &Path,
) -> Result<Vec<String>, AppError> {
    let mut copied = Vec::new();
    let source_base_dir = base_parts_dir(job_dir);
    for legacy_dir in ["base_parts", "extracted_parts"] {
        let path = output_root.join(legacy_dir);
        if path.exists() {
            fs::remove_dir_all(path)?;
        }
    }
    if source_base_dir.is_dir() {
        copy_spritalk_root_assets(&source_base_dir, output_root, &mut copied)?;
        copied.extend(sync_dynamic_eye_assets(
            &source_base_dir,
            extracted_dir,
            output_root,
        )?);
    }
    fs::write(
        output_root.join("README.txt"),
        "PachiPakuGen assets for SpriTalk\nUse the image assets in this folder with the layer-import flow supported by your SpriTalk version.\nRequired: body.png\nOptional: hair.png, hair_back.png, arm_l.png, arm_r.png, chest.png, sway_*.png, arm_l_overlay_*.png, arm_r_overlay_*.png\nDynamic eyes: eyebrow.png, eyewhite.png, irides.png, highlight.png (optional)\nLayer linkage and draw order: layer-order.json\nAnimation folders: eye, mouth_a, mouth_i, mouth_u, mouth_e, mouth_o\nNote: spritalk-motion-profile.json schema v2 is used by PachiPakuGen live view and reserved for future SpriTalk integration; current SpriTalk does not import it.\n",
    )?;
    Ok(copied)
}

fn ensure_workspace_base_parts_ready(job_dir: &Path) -> Result<(), AppError> {
    if !job_dir.join(WORKSPACE_CODEX_REQUEST_DIR).is_dir() {
        return Ok(());
    }
    let base_dir = base_parts_dir(job_dir);
    if base_dir.join("body.png").is_file() {
        return Ok(());
    }
    Err(AppError::General(format!(
        "素体がまだ作成されていません。Step 4で素体調整を完了してください: {}",
        base_dir.display()
    )))
}

fn copy_spritalk_root_assets(
    source_dir: &Path,
    output_root: &Path,
    copied: &mut Vec<String>,
) -> Result<(), AppError> {
    fs::create_dir_all(output_root)?;
    for file_name in [
        "body.png",
        "hair.png",
        "hair_back.png",
        "arm_l.png",
        "arm_r.png",
        "chest.png",
        "layer-order.json",
    ] {
        let source_path = source_dir.join(file_name);
        if source_path.is_file() {
            let dest_path = output_root.join(file_name);
            fs::copy(&source_path, &dest_path)?;
            copied.push(dest_path.to_string_lossy().into_owned());
        } else {
            let stale_path = output_root.join(file_name);
            if stale_path.exists() {
                fs::remove_file(stale_path)?;
            }
        }
    }
    // 再出力で削除・改名された動的パーツを残さない。現行sourceを正本としてからコピーする。
    for entry in fs::read_dir(output_root)? {
        let entry = entry?;
        let path = entry.path();
        let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        let is_dynamic_part_png = path.is_file()
            && is_dynamic_base_part_name(&stem.to_ascii_lowercase())
            && path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("png"));
        if is_dynamic_part_png {
            fs::remove_file(path)?;
        }
    }
    // 汎用揺れパーツと腕追従オーバーレイをそのままルートへ伝搬する。
    if let Ok(entries) = fs::read_dir(source_dir) {
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
                continue;
            };
            if path.is_file()
                && is_dynamic_base_part_name(&stem.to_ascii_lowercase())
                && path
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("png"))
            {
                let dest_path = output_root.join(name);
                fs::copy(&path, &dest_path)?;
                copied.push(dest_path.to_string_lossy().into_owned());
            }
        }
    }
    Ok(())
}

/// RIFEフレームを再生成せず、開眼時だけ使う分離目素材を既存の出力へ同期する。
fn sync_dynamic_eye_assets(
    source_base_dir: &Path,
    extracted_dir: &Path,
    output_root: &Path,
) -> Result<Vec<String>, AppError> {
    fs::create_dir_all(output_root)?;
    let adjustment = read_typed_part_adjustments(extracted_dir)
        .get("eyes-open")
        .cloned();
    let mut synced = Vec::new();
    for file_name in ["eyebrow.png", "eyewhite.png", "irides.png", "highlight.png"] {
        let source_path = source_base_dir.join(file_name);
        let dest_path = output_root.join(file_name);
        if !source_path.is_file() {
            if dest_path.is_file() {
                fs::remove_file(dest_path)?;
            }
            continue;
        }
        if let Some(adjustment) = adjustment
            .as_ref()
            .filter(|adjustment| !is_default_part_adjustment(adjustment))
        {
            let image = image::open(&source_path)?;
            transform_extracted_part(
                &image,
                adjustment.offset_x,
                adjustment.offset_y,
                adjustment.scale_percent,
            )
            .save(&dest_path)?;
        } else {
            fs::copy(&source_path, &dest_path)?;
        }
        synced.push(dest_path.to_string_lossy().into_owned());
    }
    let eyebrow_path = output_root.join("eyebrow.png");
    if eyebrow_path.is_file() {
        let eyebrow = image::open(&eyebrow_path)?;
        sanitize_legacy_eye_frames(output_root, &eyebrow)?;
    }
    Ok(synced)
}

/// Older STEP6 outputs contain the eyebrow in every eye frame. Once an
/// independent eyebrow is available, remove that baked copy in place so both
/// Motion Lab and external SpriTalk consumers see exactly one moving eyebrow.
fn sanitize_legacy_eye_frames(output_root: &Path, eyebrow: &DynamicImage) -> Result<(), AppError> {
    let eye_dir = output_root.join("eye");
    if !eye_dir.is_dir() {
        return Ok(());
    }
    let mut frame_paths = fs::read_dir(&eye_dir)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("png"))
        })
        .collect::<Vec<_>>();
    frame_paths.sort();
    for frame_path in frame_paths {
        let frame = image::open(&frame_path)?;
        let cleanup_mask =
            image_utils::eyebrow_cleanup_mask(eyebrow, frame.width(), frame.height());
        image_utils::subtract_alpha_mask(&frame, &cleanup_mask).save(&frame_path)?;
    }
    Ok(())
}

fn ensure_eyes_open_part(
    app: &AppHandle,
    job_dir: &Path,
    extracted_dir: &Path,
    profile: &str,
) -> Result<(), AppError> {
    let eyes_open = extracted_dir.join("eyes-open.png");
    if eyes_open.is_file() {
        return Ok(());
    }
    // eyes-open は素体の目（base_parts/eye_open.png）を正とする。素体が保存済みなら
    // それをコピー＝平常時の目と完全一致させる（source切り出しへは素体が無い時だけ落ちる）
    let base_eye = base_parts_dir(job_dir).join("eye_open.png");
    if base_eye.is_file() {
        fs::copy(&base_eye, &eyes_open)?;
        let originals_dir = extracted_dir.join("original_extracted_parts");
        fs::create_dir_all(&originals_dir)?;
        fs::copy(&base_eye, originals_dir.join("eyes-open.png"))?;
        return Ok(());
    }
    let source_path = job_source_path(job_dir);
    if !source_path.is_file() {
        return Err(AppError::General(format!(
            "source.png が見つかりません: {}",
            source_path.display()
        )));
    }
    let source_image = image::open(&source_path)?;
    let _ = see_through::run_inference(app, &source_path.to_string_lossy(), profile, true, None)?;
    let state = app.state::<AppState>();
    let layers = state
        .slot_layers
        .lock()
        .unwrap()
        .get("current")
        .cloned()
        .ok_or_else(|| {
            AppError::General("元画像のSee-Through分解レイヤーを取得できません".into())
        })?;
    let width = *state.canvas_width.lock().unwrap();
    let height = *state.canvas_height.lock().unwrap();
    let mask = expression_mask(&layers, EYE_ANIMATION_LAYER_NAMES, width, height, 12, 3);
    if !mask_has_minimum_edit_area(&mask, 80) {
        return Err(AppError::General(
            "元画像から eyes-open を抽出できませんでした。分解結果の目レイヤーを確認してください"
                .into(),
        ));
    }
    let source_image = if source_image.width() != width || source_image.height() != height {
        source_image.resize_exact(width, height, image::imageops::FilterType::Lanczos3)
    } else {
        source_image
    };
    let raw_eyes_open = cut_image_with_mask(&source_image, &mask);
    raw_eyes_open.save(&eyes_open)?;
    let originals_dir = extracted_dir.join("original_extracted_parts");
    fs::create_dir_all(&originals_dir)?;
    raw_eyes_open.save(originals_dir.join("eyes-open.png"))?;
    Ok(())
}

/// 生成素材のサイズが立ち絵と異なる場合に、抽出時の自動リサイズで吸収可能か。
/// 抽出結果は最終的にresize_exactで立ち絵サイズへ正規化されるため、アスペクト比の
/// 歪みが1.25倍（25%）以内なら受け入れる。それを超える場合（縦横逆など）は
/// パーツ形状の歪みが大きすぎるためブロックする
pub(crate) fn generated_part_size_is_auto_fittable(
    source_width: u32,
    source_height: u32,
    part_width: u32,
    part_height: u32,
) -> bool {
    if source_height == 0 || part_height == 0 {
        return false;
    }
    let source_aspect = source_width as f64 / source_height as f64;
    let part_aspect = part_width as f64 / part_height as f64;
    if source_aspect <= 0.0 || part_aspect <= 0.0 {
        return false;
    }
    let ratio = if source_aspect > part_aspect {
        source_aspect / part_aspect
    } else {
        part_aspect / source_aspect
    };
    ratio <= 1.25
}

fn inspect_generated_parts(
    job_dir: &Path,
    source_image: &DynamicImage,
    expected_parts: &[String],
) -> Result<InspectCodexGeneratedPartsResult, AppError> {
    let generated_parts_dir = generated_parts_dir(job_dir);
    fs::create_dir_all(&generated_parts_dir)?;
    let mut present_parts = Vec::new();
    let mut missing_parts = Vec::new();
    let mut size_mismatches = Vec::new();

    for part in expected_parts {
        let path = generated_parts_dir.join(format!("{part}.png"));
        if !path.is_file() {
            missing_parts.push(part.clone());
            continue;
        }
        let image = image::open(&path).map_err(|error| {
            AppError::General(format!(
                "Codex生成素材を読み込めません: {} ({error})",
                path.display()
            ))
        })?;
        if (image.width() != source_image.width() || image.height() != source_image.height())
            && !generated_part_size_is_auto_fittable(
                source_image.width(),
                source_image.height(),
                image.width(),
                image.height(),
            )
        {
            size_mismatches.push(format!(
                "{}.png: {}x{} — 縦横比が立ち絵({}x{})と大きく異なります。近い縦横比で再生成してください",
                part,
                image.width(),
                image.height(),
                source_image.width(),
                source_image.height()
            ));
        }
        present_parts.push(part.clone());
    }

    Ok(InspectCodexGeneratedPartsResult {
        generated_parts_path: generated_parts_dir.to_string_lossy().into_owned(),
        expected_parts: expected_parts.to_vec(),
        ready: missing_parts.is_empty() && size_mismatches.is_empty(),
        present_parts,
        missing_parts,
        size_mismatches,
    })
}

fn generated_parts_dir(job_dir: &Path) -> PathBuf {
    let primary = job_dir.join("generated_parts");
    if primary.is_dir() {
        return primary;
    }
    let workspace = job_dir.join(WORKSPACE_GENERATED_PARTS_DIR);
    if workspace.is_dir() {
        return workspace;
    }
    let legacy = job_dir.join("donors");
    if legacy.is_dir() {
        return legacy;
    }
    primary
}

fn job_source_path(job_dir: &Path) -> PathBuf {
    let primary = job_dir.join("source.png");
    if primary.is_file() {
        return primary;
    }
    let workspace = job_dir.join(WORKSPACE_CODEX_REQUEST_DIR).join("source.png");
    if workspace.is_file() {
        return workspace;
    }
    primary
}

fn job_request_path(job_dir: &Path) -> PathBuf {
    let workspace = job_dir
        .join(WORKSPACE_CODEX_REQUEST_DIR)
        .join("codex_request.md");
    if workspace.is_file() {
        return workspace;
    }
    job_dir.join("codex_request.md")
}

fn job_handoff_path(job_dir: &Path) -> PathBuf {
    let workspace = job_dir
        .join(WORKSPACE_CODEX_REQUEST_DIR)
        .join("codex_handoff.md");
    if workspace.is_file() {
        return workspace;
    }
    job_dir.join("codex_handoff.md")
}

/// 抽出済みパーツの実際の保存先。旧式の単体ジョブ（job_dir直下にextracted_partsを持つ）と
/// 現行のワークスペース構成（job_dirが03_see_through配下へ直接保存）の両方を解決する。
/// この関数を経由せずにパスを組み立てると、書き込み側と食い違って読み取りが常に失敗する
/// （実際に発生したバグ: workspace.rsが独自に "03_see_through/extracted_parts/" を
/// 組み立てていたが、ここが実際に書き込むのは "03_see_through/" 直下だった）
pub(crate) fn extracted_parts_dir(job_dir: &Path) -> PathBuf {
    let primary = job_dir.join("extracted_parts");
    if primary.is_dir() {
        return primary;
    }
    let workspace = job_dir.join(WORKSPACE_SEE_THROUGH_DIR);
    if workspace.is_dir() {
        return workspace;
    }
    primary
}

fn base_parts_dir(job_dir: &Path) -> PathBuf {
    let primary = job_dir.join("base_parts");
    if primary.is_dir() {
        return primary;
    }
    let workspace = job_dir.join(WORKSPACE_SEE_THROUGH_DIR).join("base_parts");
    if workspace.parent().is_some_and(Path::is_dir) {
        return workspace;
    }
    primary
}

fn rife_output_dir(job_dir: &Path) -> PathBuf {
    let primary = job_dir.join("rife_output");
    if primary.is_dir() {
        return primary;
    }
    let workspace = job_dir.join(WORKSPACE_SPRITALK_PARTS_DIR);
    if workspace.is_dir() {
        return workspace;
    }
    primary
}

fn expected_parts_from_job(job_dir: &Path) -> Result<Vec<String>, AppError> {
    let job_json = job_dir.join("codex_job.json");
    if job_json.is_file() {
        let value: serde_json::Value = serde_json::from_slice(&fs::read(&job_json)?)
            .map_err(|error| AppError::General(format!("codex_job.json解析失敗: {error}")))?;
        if let Some(parts) = value
            .get("expectedGeneratedParts")
            .and_then(|value| value.as_array())
        {
            let parsed: Vec<String> = parts
                .iter()
                .filter_map(|value| value.as_str().map(str::to_string))
                .collect();
            if !parsed.is_empty() {
                return Ok(parsed);
            }
        }
    }
    Ok(GENERATED_PART_TARGETS
        .iter()
        .map(|target| (*target).to_string())
        .collect())
}

fn copy_reference_image(path: &str, output_dir: &Path) -> Result<PathBuf, AppError> {
    let reference = PathBuf::from(path);
    if !reference.is_file() {
        return Err(AppError::General(format!(
            "参照画像が見つかりません: {}",
            reference.display()
        )));
    }
    let image = image::open(&reference)
        .map_err(|error| AppError::General(format!("参照画像を読み込めません: {error}")))?;
    let output = output_dir.join("reference.png");
    image.save(&output)?;
    Ok(output)
}

fn build_codex_request_text(
    source_path: &Path,
    reference_path: Option<&Path>,
    generated_parts_dir: &Path,
    expected_parts: &[String],
    request: &PrepareCodexExpressionJobRequest,
) -> String {
    let targets = expected_parts
        .iter()
        .map(|target| {
            let prompt = generated_part_prompt(target, request);
            format!("- `{target}.png`: {prompt}")
        })
        .collect::<Vec<_>>()
        .join("\n");
    let reference = reference_path
        .map(|path| {
            format!(
                "- 参照画像: `{}`\n  口内、歯、舌、瞳、まぶたの描き方だけを参照してください。",
                path.display()
            )
        })
        .unwrap_or_else(|| "- 参照画像: なし".to_string());

    format!(
        r#"# Codex ImageGen 生成依頼

PachiPakuGen に戻すための Codex生成素材を作成してください。

- 元画像: `{}`
{}
- 保存先: `{}`

## 保存するファイル

{}

## 共通ルール

- 元画像と同じキャンバスサイズのフルフレームPNGで保存してください。
- 生成画像は完成品ではなく、PachiPakuGen が See-Through で目/口レイヤーだけ抽出するための素材です。
- キャラクター同一性、ポーズ、カメラ、輪郭、髪型、服、アクセサリー、首、チョーカー、背景をできるだけ維持してください。
- 鼻、首、顎、後ろ髪、服、アクセサリーを意図的に変更しないでください。
- 口差分では唇、歯、舌、口内だけを変えてください。
- 閉眼差分ではまぶたとまつ毛だけを変え、眉の位置や形は大きく変えないでください。
- 多少の周辺揺れは後段で捨てますが、目口の位置合わせを最優先してください。

## 追加指示

{}
"#,
        source_path.display(),
        reference,
        generated_parts_dir.display(),
        targets,
        request.prompt.trim()
    )
}

fn build_codex_handoff_text(
    job_dir: &Path,
    request_path: &Path,
    generated_parts_dir: &Path,
) -> String {
    format!(
        r#"# Codexへの渡し方

別のCodexチャット、または画像生成を頼むCodexセッションへ、次の文章をそのまま送ってください。

```text
`{}` の内容を読んで、指定されたCodex生成素材PNGを作ってください。
元画像は同じジョブフォルダ内の `source.png` です。
生成したPNGは `{}` に保存してください。
```

作成後、このPachiPakuGen画面へ戻って「生成素材を確認」を押します。

ジョブフォルダ:

```text
{}
```
"#,
        request_path.display(),
        generated_parts_dir.display(),
        job_dir.display()
    )
}

fn generated_part_prompt(target: &str, request: &PrepareCodexExpressionJobRequest) -> String {
    let request = GenerateExpressionSetRequest {
        engine: "codex".into(),
        quality: "standard".into(),
        targets: vec![target.to_string()],
        reference_image_path: None,
        prompt: request.prompt.clone(),
        mouth_corner: request.mouth_corner.clone(),
        mouth_size: request.mouth_size.clone(),
        output_path: String::new(),
        rife_frame_count: None,
    };
    target_prompt(target, &request).replace('\n', " ")
}

#[tauri::command]
pub fn get_expression_api_status() -> ExpressionApiStatus {
    expression_api_status()
}

#[tauri::command]
pub fn save_expression_api_key(
    provider: String,
    api_key: String,
) -> Result<ExpressionApiStatus, AppError> {
    let api_key = api_key.trim();
    if api_key.is_empty() {
        return Err(AppError::General("APIキーを入力してください".into()));
    }
    let entry = credential_entry(&provider)?;
    entry.set_password(api_key).map_err(|error| {
        AppError::General(format!("APIキーを安全に保存できませんでした: {error}"))
    })?;
    Ok(expression_api_status())
}

#[tauri::command]
pub fn delete_expression_api_key(provider: String) -> Result<ExpressionApiStatus, AppError> {
    let entry = credential_entry(&provider)?;
    if let Err(error) = entry.delete_credential() {
        if !matches!(error, keyring_core::Error::NoEntry) {
            return Err(AppError::General(format!(
                "保存済みAPIキーを削除できませんでした: {error}"
            )));
        }
    }
    Ok(expression_api_status())
}

fn expression_api_status() -> ExpressionApiStatus {
    let openai_source = api_key_source("gpt");
    let gemini_source = api_key_source("nano");
    ExpressionApiStatus {
        openai_configured: openai_source.is_some(),
        gemini_configured: gemini_source.is_some(),
        openai_source,
        gemini_source,
    }
}

#[tauri::command]
pub async fn generate_expression_set(
    app: AppHandle,
    request: GenerateExpressionSetRequest,
) -> Result<GenerateExpressionSetResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || generate_expression_set_inner(app, request))
        .await
        .map_err(|error| AppError::General(format!("Task join error: {error}")))?
}

fn generate_expression_set_inner(
    app: AppHandle,
    request: GenerateExpressionSetRequest,
) -> Result<GenerateExpressionSetResult, AppError> {
    if request.targets.is_empty() {
        return Err(AppError::General("生成対象が選択されていません".into()));
    }
    let output_dir = PathBuf::from(&request.output_path);
    fs::create_dir_all(&output_dir)?;
    let candidate_dir = output_dir.join("_candidates");
    let mask_dir = output_dir.join("_masks");
    fs::create_dir_all(&candidate_dir)?;
    fs::create_dir_all(&mask_dir)?;
    let generation_targets = effective_expression_targets(&request.targets, &output_dir);

    let state = app.state::<AppState>();
    let source = state
        .cached_original
        .lock()
        .unwrap()
        .clone()
        .ok_or_else(|| {
            AppError::General(
                "元画像が保持されていません。内蔵See-Throughを再実行してください".into(),
            )
        })?;
    let layers = state
        .slot_layers
        .lock()
        .unwrap()
        .get("current")
        .cloned()
        .ok_or_else(|| {
            AppError::General(
                "分解済みレイヤーがありません。内蔵See-Throughを再実行してください".into(),
            )
        })?;
    let width = source.width();
    let height = source.height();
    let mouth_mask = mouth_expression_mask(&layers, width, height);
    let eye_mask = expression_mask(&layers, EYE_ANIMATION_LAYER_NAMES, width, height, 36, 3);
    validate_selected_masks(&generation_targets, &mouth_mask, &eye_mask)?;
    let rife_frame_count = request
        .rife_frame_count
        .unwrap_or(EXPRESSION_RIFE_DEFAULT_FRAME_COUNT);
    if !(2..=30).contains(&rife_frame_count) {
        return Err(AppError::General(
            "RIFEフレーム数は2〜30の範囲で指定してください".into(),
        ));
    }
    let rife_jobs = planned_expression_rife_jobs(&generation_targets, &output_dir);
    let source_png = png_bytes(&source)?;
    let reference_image = request
        .reference_image_path
        .as_deref()
        .filter(|path| !path.trim().is_empty())
        .map(read_reference_image)
        .transpose()?;
    let reference_guide_dir = output_dir.join("_reference_guides");
    if reference_image.is_some() {
        fs::create_dir_all(&reference_guide_dir)?;
    }

    let (model, api_key) = match request.engine.as_str() {
        "gpt" => ("gpt-image-2".to_string(), resolve_api_key("gpt")?),
        "nano" => ("gemini-3-pro-image".to_string(), resolve_api_key("nano")?),
        other => return Err(AppError::General(format!("未対応の生成エンジン: {other}"))),
    };

    let mut generated_files = Vec::new();
    let mut output_images = HashMap::new();
    let mut progress_done = 0u32;
    let total = generation_targets.len() as u32 + rife_jobs.len() as u32 * rife_frame_count;
    let mut reference_guides = Vec::new();
    for (index, target) in generation_targets.iter().enumerate() {
        let mask = if target.starts_with("eyes-") {
            &eye_mask
        } else {
            &mouth_mask
        };
        let prompt = target_prompt(target, &request);
        let reference_guide_png = if target_uses_reference_guide(target) {
            reference_image
                .as_ref()
                .map(|reference| reference_feature_guide_png(reference, mask, width, height))
                .transpose()?
        } else {
            None
        };
        if let Some(reference_guide_png) = reference_guide_png.as_ref() {
            let reference_guide_path = reference_guide_dir.join(format!("{target}.png"));
            fs::write(&reference_guide_path, reference_guide_png)?;
            reference_guides.push(reference_guide_path.to_string_lossy().into_owned());
        }
        app.emit(
            "generation-progress",
            ProgressPayload {
                current: index as u32 + 1,
                total,
                pair_name: target.clone(),
            },
        )
        .ok();

        let candidate_bytes = if request.engine == "gpt" {
            generate_openai(
                &api_key,
                &model,
                &request.quality,
                &prompt,
                &source_png,
                reference_guide_png.as_deref(),
                mask,
            )?
        } else {
            generate_gemini(
                &api_key,
                &model,
                &request.quality,
                &prompt,
                &source_png,
                reference_guide_png.as_deref(),
            )?
        };
        let candidate_path = candidate_dir.join(format!("{target}.png"));
        fs::write(&candidate_path, &candidate_bytes)?;
        save_openai_mask(mask, &mask_dir.join(format!("{target}.png")))?;

        let candidate = image::load_from_memory(&candidate_bytes)
            .map_err(|error| AppError::General(format!("{target} の画像読込に失敗: {error}")))?;
        let final_image = composite_inside_mask(&source, &candidate, mask, width, height);
        let final_path = output_dir.join(format!("{target}.png"));
        final_image.save(&final_path)?;
        output_images.insert(target.clone(), final_image);
        generated_files.push(final_path.to_string_lossy().into_owned());
        progress_done = index as u32 + 1;
    }

    let eyes_open_path = output_dir.join("eyes-open.png");
    source.save(&eyes_open_path)?;
    output_images.insert("eyes-open".into(), source.clone());
    generated_files.push(eyes_open_path.to_string_lossy().into_owned());
    let rife_directories = generate_expression_rife_outputs(
        &app,
        &state,
        &output_dir,
        &output_images,
        &rife_jobs,
        &mouth_mask,
        &eye_mask,
        rife_frame_count,
        &mut progress_done,
        total,
    )?;
    let rife_output_path = if rife_directories.is_empty() {
        None
    } else {
        Some(output_dir.join("_rife").to_string_lossy().into_owned())
    };
    let manifest = json!({
        "formatVersion": 1,
        "engine": request.engine,
        "model": model,
        "quality": request.quality,
        "requestedTargets": request.targets,
        "targets": generation_targets,
        "mouthCorner": request.mouth_corner,
        "mouthSize": request.mouth_size,
        "prompt": request.prompt,
        "referenceImage": request.reference_image_path,
        "referenceGuides": reference_guides,
        "maskOutsideChangesDiscarded": true,
        "rifeFrameCount": rife_frame_count,
        "rifeOutputPath": rife_output_path,
        "rifeDirectories": rife_directories,
        "generatedFiles": generated_files,
    });
    fs::write(
        output_dir.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)
            .map_err(|error| AppError::General(format!("manifest作成失敗: {error}")))?,
    )?;

    Ok(GenerateExpressionSetResult {
        output_path: output_dir.to_string_lossy().into_owned(),
        generated_files,
        engine: request.engine,
        model,
        rife_output_path,
        rife_directories,
        rife_frame_count,
    })
}

fn planned_expression_rife_jobs(targets: &[String], output_dir: &Path) -> Vec<ExpressionRifeJob> {
    let has_target = |name: &str| targets.iter().any(|target| target == name);
    let mut jobs = Vec::new();
    if has_target("eyes-closed") {
        jobs.push(ExpressionRifeJob {
            name: "eyes-closed",
            start_key: "eyes-open",
            end_key: "eyes-closed",
            mask_kind: ExpressionMaskKind::Eye,
        });
    }
    if has_target("mouth-closed") || output_dir.join("mouth-closed.png").exists() {
        for name in ["mouth-a", "mouth-i", "mouth-u", "mouth-e", "mouth-o"] {
            if has_target(name) {
                jobs.push(ExpressionRifeJob {
                    name,
                    start_key: "mouth-closed",
                    end_key: name,
                    mask_kind: ExpressionMaskKind::Mouth,
                });
            }
        }
    }
    jobs
}

fn effective_expression_targets(requested: &[String], output_dir: &Path) -> Vec<String> {
    let mut targets = requested.to_vec();
    let needs_mouth_baseline = MOUTH_VOWEL_TARGETS
        .iter()
        .any(|name| targets.iter().any(|target| target == name));
    let has_mouth_baseline = targets.iter().any(|target| target == "mouth-closed")
        || output_dir.join("mouth-closed.png").exists();
    if needs_mouth_baseline && !has_mouth_baseline {
        targets.insert(0, "mouth-closed".into());
    }
    targets.dedup();
    targets
}

fn target_uses_reference_guide(target: &str) -> bool {
    MOUTH_VOWEL_TARGETS.contains(&target)
}

#[allow(clippy::too_many_arguments)]
fn generate_expression_rife_outputs(
    app: &AppHandle,
    state: &AppState,
    output_dir: &Path,
    output_images: &HashMap<String, DynamicImage>,
    jobs: &[ExpressionRifeJob],
    mouth_mask: &GrayImage,
    eye_mask: &GrayImage,
    frame_count: u32,
    progress_done: &mut u32,
    progress_total: u32,
) -> Result<Vec<String>, AppError> {
    if jobs.is_empty() {
        return Ok(Vec::new());
    }

    {
        let mut session = state.rife_session.lock().unwrap();
        if session.is_none() {
            let model_path = resolve_model_path(app, "rife.onnx")?;
            *session = Some(create_session(&model_path)?);
        }
    }

    let rife_root = output_dir.join("_rife");
    fs::create_dir_all(&rife_root)?;
    let mut directories = Vec::new();
    let ratios: Vec<f32> = (0..frame_count)
        .map(|index| index as f32 / (frame_count - 1) as f32)
        .collect();
    let mut session_guard = state.rife_session.lock().unwrap();
    let session = session_guard.as_mut().unwrap();

    for job in jobs {
        let start = expression_output_image(output_images, output_dir, job.start_key)?;
        let end = expression_output_image(output_images, output_dir, job.end_key)?;
        let mask = match job.mask_kind {
            ExpressionMaskKind::Eye => eye_mask,
            ExpressionMaskKind::Mouth => mouth_mask,
        };
        let out_dir = rife_root.join(job.name);
        recreate_png_dir(&out_dir)?;
        for (step, &ratio) in ratios.iter().enumerate() {
            *progress_done += 1;
            app.emit(
                "generation-progress",
                ProgressPayload {
                    current: (*progress_done).min(progress_total),
                    total: progress_total,
                    pair_name: format!("RIFE {}", job.name),
                },
            )
            .ok();

            let frame = if step == 0 {
                start.clone()
            } else if step + 1 == frame_count as usize {
                end.clone()
            } else {
                let interpolated = rife_interpolate(session, &start, &end, ratio)?;
                let interpolated = preserve_interpolated_alpha(&interpolated, &start, &end, ratio);
                composite_inside_mask(&start, &interpolated, mask, start.width(), start.height())
            };
            frame.save(out_dir.join(format!("{:03}.png", step + 1)))?;
        }
        directories.push(out_dir.to_string_lossy().into_owned());
    }
    Ok(directories)
}

fn expression_output_image(
    output_images: &HashMap<String, DynamicImage>,
    output_dir: &Path,
    key: &str,
) -> Result<DynamicImage, AppError> {
    if let Some(image) = output_images.get(key) {
        return Ok(image.clone());
    }
    let path = output_dir.join(format!("{key}.png"));
    image::open(&path).map_err(|error| {
        AppError::General(format!(
            "RIFE補間用の画像を読み込めません: {} ({error})",
            path.display()
        ))
    })
}

fn recreate_png_dir(path: &Path) -> Result<(), AppError> {
    fs::create_dir_all(path)?;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();
        if path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("png"))
        {
            fs::remove_file(path)?;
        }
    }
    Ok(())
}

fn preserve_interpolated_alpha(
    interpolated: &DynamicImage,
    start: &DynamicImage,
    end: &DynamicImage,
    ratio: f32,
) -> DynamicImage {
    let mut rgba = interpolated.to_rgba8();
    let start_alpha = start.to_rgba8();
    let end_alpha = end.to_rgba8();
    let ratio = ratio.clamp(0.0, 1.0);
    for y in 0..rgba.height() {
        for x in 0..rgba.width() {
            let a0 = start_alpha.get_pixel(x, y)[3] as f32;
            let a1 = end_alpha.get_pixel(x, y)[3] as f32;
            rgba.get_pixel_mut(x, y)[3] = (a0 * (1.0 - ratio) + a1 * ratio).round() as u8;
        }
    }
    DynamicImage::ImageRgba8(rgba)
}

fn credential_name(provider: &str) -> Result<&'static str, AppError> {
    match provider {
        "gpt" => Ok(OPENAI_CREDENTIAL),
        "nano" => Ok(GEMINI_CREDENTIAL),
        other => Err(AppError::General(format!(
            "未対応のAPIプロバイダー: {other}"
        ))),
    }
}

fn credential_entry(provider: &str) -> Result<Entry, AppError> {
    Entry::new(KEYRING_SERVICE, credential_name(provider)?)
        .map_err(|error| AppError::General(format!("Windows資格情報を開けませんでした: {error}")))
}

fn stored_api_key(provider: &str) -> Option<String> {
    credential_entry(provider)
        .ok()?
        .get_password()
        .ok()
        .filter(|value| !value.trim().is_empty())
}

fn api_key_source(provider: &str) -> Option<String> {
    if stored_api_key(provider).is_some() {
        return Some("app".into());
    }
    match provider {
        "gpt" if nonempty_env("OPENAI_API_KEY").is_ok() => Some("environment".into()),
        "nano"
            if nonempty_env("GEMINI_API_KEY").is_ok() || nonempty_env("GOOGLE_API_KEY").is_ok() =>
        {
            Some("environment".into())
        }
        _ => None,
    }
}

fn resolve_api_key(provider: &str) -> Result<String, AppError> {
    if let Some(key) = stored_api_key(provider) {
        return Ok(key);
    }
    match provider {
        "gpt" => nonempty_env("OPENAI_API_KEY").map_err(|_| {
            AppError::General(
                "GPT Image 2のAPIキーをアプリ内で保存するか、OPENAI_API_KEYを設定してください"
                    .into(),
            )
        }),
        "nano" => nonempty_env("GEMINI_API_KEY")
            .or_else(|_| nonempty_env("GOOGLE_API_KEY"))
            .map_err(|_| {
                AppError::General(
                    "Nano BananaのAPIキーをアプリ内で保存するか、GEMINI_API_KEYを設定してください"
                        .into(),
                )
            }),
        other => Err(AppError::General(format!(
            "未対応のAPIプロバイダー: {other}"
        ))),
    }
}

fn nonempty_env(name: &str) -> Result<String, std::env::VarError> {
    match std::env::var(name) {
        Ok(value) if !value.trim().is_empty() => Ok(value),
        Ok(_) => Err(std::env::VarError::NotPresent),
        Err(error) => Err(error),
    }
}

fn validate_selected_masks(
    targets: &[String],
    mouth_mask: &GrayImage,
    eye_mask: &GrayImage,
) -> Result<(), AppError> {
    if targets.iter().any(|target| target.starts_with("mouth-"))
        && !mask_has_minimum_edit_area(mouth_mask, 600)
    {
        return Err(AppError::General(
            "口マスクを取得できませんでした。分解結果で口レイヤーを確認してください".into(),
        ));
    }
    if targets.iter().any(|target| target.starts_with("eyes-"))
        && !mask_has_minimum_edit_area(eye_mask, 600)
    {
        return Err(AppError::General(
            "目マスクを取得できませんでした。分解結果で目レイヤーを確認してください".into(),
        ));
    }
    Ok(())
}

fn mask_has_minimum_edit_area(mask: &GrayImage, min_pixels: usize) -> bool {
    mask.pixels().filter(|pixel| pixel[0] > 0).count() >= min_pixels
}

fn normalize_layer_name(name: &str) -> &str {
    for base in ["eyewhite", "irides", "eyelash", "eyebrow", "mouth", "nose"] {
        if name == format!("{base}-l")
            || name == format!("{base}-r")
            || name == format!("{base}_l")
            || name == format!("{base}_r")
        {
            return base;
        }
    }
    name
}

fn expression_mask(
    layers: &HashMap<String, DynamicImage>,
    names: &[&str],
    width: u32,
    height: u32,
    dilate: i32,
    blur: i32,
) -> GrayImage {
    let raw = raw_expression_mask(layers, names, width, height);
    let adjusted = neck_extract::adjust_mask(&raw, width, height, dilate, blur);
    GrayImage::from_raw(width, height, adjusted).unwrap_or_else(|| GrayImage::new(width, height))
}

fn raw_expression_mask(
    layers: &HashMap<String, DynamicImage>,
    names: &[&str],
    width: u32,
    height: u32,
) -> Vec<u8> {
    let mut raw = vec![0u8; (width * height) as usize];
    for (layer_name, image) in layers {
        if !names.contains(&normalize_layer_name(layer_name)) {
            continue;
        }
        let rgba = image.to_rgba8();
        for y in 0..height {
            for x in 0..width {
                let index = (y * width + x) as usize;
                raw[index] = raw[index].max(rgba.get_pixel(x, y)[3]);
            }
        }
    }
    raw
}

fn mouth_expression_mask(
    layers: &HashMap<String, DynamicImage>,
    width: u32,
    height: u32,
) -> GrayImage {
    let raw = raw_expression_mask(layers, &["mouth"], width, height);
    let nose_raw = raw_expression_mask(layers, &["nose"], width, height);
    let eye_raw = raw_expression_mask(layers, EYE_LAYER_NAMES, width, height);
    let shaped = if mouth_raw_is_local(&raw, width, height) {
        mouth_mask_from_local_raw(&raw, width, height)
    } else {
        mouth_mask_from_face_anchors(&nose_raw, &eye_raw, width, height)
    };

    let mut adjusted = neck_extract::adjust_mask(&shaped, width, height, 0, 4);
    if nose_raw.iter().any(|value| *value > 0) {
        let nose_protect = neck_extract::adjust_mask(&nose_raw, width, height, 10, 2);
        for (editable, protect) in adjusted.iter_mut().zip(nose_protect) {
            if protect > 0 {
                *editable = 0;
            }
        }
    }
    GrayImage::from_raw(width, height, adjusted).unwrap_or_else(|| GrayImage::new(width, height))
}

fn mouth_raw_is_local(raw: &[u8], width: u32, height: u32) -> bool {
    let Some((min_x, min_y, max_x, max_y)) = mask_bounds(raw, width, height) else {
        return false;
    };
    let count = raw.iter().filter(|value| **value > 0).count() as u64;
    let area = width as u64 * height as u64;
    let bounds_width = max_x.saturating_sub(min_x).saturating_add(1);
    let bounds_height = max_y.saturating_sub(min_y).saturating_add(1);
    count < area / 20 && bounds_width < width / 3 && bounds_height < height / 4
}

fn mouth_mask_from_local_raw(raw: &[u8], width: u32, height: u32) -> Vec<u8> {
    let Some((min_x, min_y, max_x, max_y)) = mask_bounds(raw, width, height) else {
        return vec![0u8; (width * height) as usize];
    };
    let raw_width = max_x.saturating_sub(min_x).saturating_add(1);
    let raw_height = max_y.saturating_sub(min_y).saturating_add(1);
    let side_margin = (raw_width / 2).clamp(18, 48);
    let top_margin = (raw_height / 2).clamp(8, 24);
    let bottom_margin = raw_height.clamp(18, 48);
    rect_mask(
        width,
        height,
        min_x.saturating_sub(side_margin),
        min_y.saturating_sub(top_margin),
        max_x.saturating_add(side_margin),
        max_y.saturating_add(bottom_margin),
    )
}

fn mouth_mask_from_face_anchors(
    nose_raw: &[u8],
    eye_raw: &[u8],
    width: u32,
    height: u32,
) -> Vec<u8> {
    if let Some((nose_min_x, _nose_min_y, nose_max_x, nose_max_y)) = local_feature_bounds(
        nose_raw,
        width,
        height,
        (width / 6).max(12),
        (height / 5).max(12),
        40,
        4,
        5,
    ) {
        let cx = (nose_min_x + nose_max_x) / 2;
        let mouth_half_width = (width / 15).clamp(30, 96);
        let top_margin = (height / 192).clamp(3, 10);
        let mouth_height = (height / 12).clamp(44, 112);
        let top = nose_max_y.saturating_add(top_margin);
        return ellipse_mask(
            width,
            height,
            cx,
            top.saturating_add(mouth_height / 2),
            mouth_half_width,
            mouth_height / 2,
        );
    }
    if let Some((eye_min_x, _eye_min_y, eye_max_x, eye_max_y)) = local_feature_bounds(
        eye_raw,
        width,
        height,
        (width * 2 / 3).max(12),
        (height / 3).max(12),
        4,
        3,
        4,
    ) {
        let cx = (eye_min_x + eye_max_x) / 2;
        let mouth_half_width = (width / 15).clamp(30, 96);
        let top = eye_max_y.saturating_sub((height / 80).clamp(6, 16));
        let mouth_height = (height / 12).clamp(44, 112);
        return ellipse_mask(
            width,
            height,
            cx,
            top.saturating_add(mouth_height / 2),
            mouth_half_width,
            mouth_height / 2,
        );
    }
    ellipse_mask(
        width,
        height,
        width / 2,
        height.saturating_mul(19) / 40,
        (width / 15).clamp(30, 96),
        (height / 24).clamp(22, 56),
    )
}

/// See-Throughが目/口レイヤーを抽出できなかった時の保険。
/// 生成素材と元画像（立ち絵）の差分領域＝変化した目/口をそのまま切り出す。
/// See-Throughの分解結果に依存しないため、分解しづらい素材でも表情を落とさない。
fn difference_fallback_extract(
    source_image: &DynamicImage,
    generated_image: &DynamicImage,
    width: u32,
    height: u32,
    is_eye: bool,
) -> Option<DynamicImage> {
    if width == 0 || height == 0 {
        return None;
    }
    let source_rgba = resized_rgba(source_image, width, height);
    let generated_rgba = resized_rgba(generated_image, width, height);
    // 顔ゾーンで差分を検出（目=上寄り / 口=下寄り）。全体的なAIノイズの誤検出を抑える
    let (y_min, y_max) = if is_eye {
        (0u32, (height as f32 * 0.62) as u32)
    } else {
        ((height as f32 * 0.32) as u32, height)
    };
    let mut diff = vec![0u8; (width * height) as usize];
    let mut count = 0usize;
    for y in y_min..y_max.min(height) {
        for x in 0..width {
            let a = source_rgba.get_pixel(x, y).0;
            let b = generated_rgba.get_pixel(x, y).0;
            let rgb_delta = (a[0] as i16 - b[0] as i16).abs()
                + (a[1] as i16 - b[1] as i16).abs()
                + (a[2] as i16 - b[2] as i16).abs();
            let alpha_delta = (a[3] as i16 - b[3] as i16).abs();
            if rgb_delta >= 48 || alpha_delta >= 32 {
                diff[(y * width + x) as usize] = 255;
                count += 1;
            }
        }
    }
    if count < 30 {
        return None; // 差分がほぼ無い＝抽出不能
    }
    let (min_x, min_y, max_x, max_y) = mask_bounds(&diff, width, height)?;
    let bbox_w = max_x.saturating_sub(min_x) + 1;
    let bbox_h = max_y.saturating_sub(min_y) + 1;
    // 大きすぎる差分＝全体的な色シフト/ノイズで信頼できない
    if bbox_w > width * 3 / 4 || bbox_h > height * 3 / 5 {
        return None;
    }
    // 差分マスクを少し膨張＋ぼかしして輪郭をなじませる
    let adjusted = neck_extract::adjust_mask(&diff, width, height, 3, 3);
    let mask = GrayImage::from_raw(width, height, adjusted)?;
    let gen_dyn = DynamicImage::ImageRgba8(generated_rgba);
    Some(cut_image_with_mask(&gen_dyn, &mask))
}

fn refine_mouth_mask_with_difference(
    base_mask: &GrayImage,
    source_image: &DynamicImage,
    generated_image: &DynamicImage,
    width: u32,
    height: u32,
) -> GrayImage {
    if source_image.width() != width
        || source_image.height() != height
        || generated_image.width() != width
        || generated_image.height() != height
    {
        return base_mask.clone();
    }

    let source = source_image.to_rgba8();
    let generated = generated_image.to_rgba8();
    let mut diff = vec![0u8; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            if base_mask.get_pixel(x, y)[0] == 0 {
                continue;
            }
            let a = source.get_pixel(x, y).0;
            let b = generated.get_pixel(x, y).0;
            let rgb_delta = (a[0] as i16 - b[0] as i16).abs()
                + (a[1] as i16 - b[1] as i16).abs()
                + (a[2] as i16 - b[2] as i16).abs();
            let alpha_delta = (a[3] as i16 - b[3] as i16).abs();
            if rgb_delta >= 36 || alpha_delta >= 24 {
                diff[(y * width + x) as usize] = 255;
            }
        }
    }

    let Some((min_x, min_y, max_x, max_y)) = mask_bounds(&diff, width, height) else {
        return base_mask.clone();
    };
    let diff_count = diff.iter().filter(|value| **value > 0).count();
    if diff_count < 20 {
        return base_mask.clone();
    }

    let max_refined_width = (width / 5).max(24);
    let max_refined_height = (height / 7).max(24);
    if max_x.saturating_sub(min_x).saturating_add(1) > max_refined_width
        || max_y.saturating_sub(min_y).saturating_add(1) > max_refined_height
    {
        return base_mask.clone();
    }

    let expanded = rect_mask(
        width,
        height,
        min_x.saturating_sub((width / 96).clamp(6, 16)),
        min_y.saturating_sub((height / 128).clamp(5, 12)),
        max_x.saturating_add((width / 96).clamp(6, 16)),
        max_y.saturating_add((height / 96).clamp(8, 18)),
    );
    let adjusted = neck_extract::adjust_mask(&expanded, width, height, 2, 2);
    GrayImage::from_raw(width, height, adjusted).unwrap_or_else(|| base_mask.clone())
}

fn rect_mask(width: u32, height: u32, left: u32, top: u32, right: u32, bottom: u32) -> Vec<u8> {
    let mut mask = vec![0u8; (width * height) as usize];
    if width == 0 || height == 0 {
        return mask;
    }
    if left >= width || top >= height || right < left || bottom < top {
        return mask;
    }
    let right = right.min(width - 1);
    let bottom = bottom.min(height - 1);
    for y in top..=bottom {
        for x in left..=right {
            mask[(y * width + x) as usize] = 255;
        }
    }
    mask
}

fn ellipse_mask(width: u32, height: u32, cx: u32, cy: u32, rx: u32, ry: u32) -> Vec<u8> {
    let mut mask = vec![0u8; (width * height) as usize];
    if width == 0 || height == 0 || rx == 0 || ry == 0 || cx >= width || cy >= height {
        return mask;
    }
    let left = cx.saturating_sub(rx);
    let top = cy.saturating_sub(ry);
    let right = cx.saturating_add(rx).min(width - 1);
    let bottom = cy.saturating_add(ry).min(height - 1);
    let rx = rx as f32;
    let ry = ry as f32;
    for y in top..=bottom {
        for x in left..=right {
            let dx = (x as f32 - cx as f32) / rx;
            let dy = (y as f32 - cy as f32) / ry;
            if dx * dx + dy * dy <= 1.0 {
                mask[(y * width + x) as usize] = 255;
            }
        }
    }
    mask
}

fn local_feature_bounds(
    mask: &[u8],
    width: u32,
    height: u32,
    max_width: u32,
    max_height: u32,
    max_area_divisor: u64,
    max_y_ratio_num: u32,
    max_y_ratio_den: u32,
) -> Option<(u32, u32, u32, u32)> {
    let (min_x, min_y, max_x, max_y) = mask_bounds(mask, width, height)?;
    let bounds_width = max_x.saturating_sub(min_x).saturating_add(1);
    let bounds_height = max_y.saturating_sub(min_y).saturating_add(1);
    if bounds_width > max_width || bounds_height > max_height {
        return None;
    }
    let count = mask.iter().filter(|value| **value > 0).count() as u64;
    let area = width as u64 * height as u64;
    if max_area_divisor > 0 && count > area / max_area_divisor {
        return None;
    }
    let max_allowed_y = height.saturating_mul(max_y_ratio_num) / max_y_ratio_den;
    if max_y >= max_allowed_y {
        return None;
    }
    Some((min_x, min_y, max_x, max_y))
}

fn mask_bounds(mask: &[u8], width: u32, height: u32) -> Option<(u32, u32, u32, u32)> {
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0;
    let mut max_y = 0;
    let mut found = false;
    for y in 0..height {
        for x in 0..width {
            if mask[(y * width + x) as usize] == 0 {
                continue;
            }
            found = true;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }
    found.then_some((min_x, min_y, max_x, max_y))
}

fn gray_mask_bounds(mask: &GrayImage) -> Option<(u32, u32, u32, u32)> {
    let width = mask.width();
    let height = mask.height();
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0;
    let mut max_y = 0;
    let mut found = false;
    for y in 0..height {
        for x in 0..width {
            if mask.get_pixel(x, y)[0] == 0 {
                continue;
            }
            found = true;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }
    found.then_some((min_x, min_y, max_x, max_y))
}

fn target_prompt(target: &str, request: &GenerateExpressionSetRequest) -> String {
    let target_instruction = match target {
        "eyes-closed" => "Close both eyes naturally for a blink (FACS AU43). Preserve eyebrow shape and position.",
        "mouth-closed" => "Close the mouth with the lips together (FACS AU24), keeping a natural anime mouth line.",
        "mouth-a" => "Create Japanese vowel A / あ: jaw dropped and mouth vertically open (FACS AU26), natural visible mouth interior.",
        "mouth-i" => "Create Japanese vowel I / い: a narrow horizontal clenched-teeth mouth. Show mainly a clean white teeth strip, with minimal dark mouth interior and only a slight lip opening (FACS AU20).",
        "mouth-u" => "Create Japanese vowel U / う: a small clearly rounded puckered mouth (FACS AU18).",
        "mouth-e" => "Create Japanese vowel E / え: horizontally open mouth with some teeth visible, less stretched than I.",
        "mouth-o" => "Create Japanese vowel O / お: rounded vertically open mouth with lip funnel and visible mouth interior.",
        _ => "Edit only the requested facial feature.",
    };
    let corner = match request.mouth_corner.as_str() {
        "up" => "Raise the mouth corners.",
        "down" => "Lower the mouth corners.",
        _ => "Keep the mouth corners neutral.",
    };
    let size = match request.mouth_size.as_str() {
        "small" => "Use a smaller mouth.",
        "large" => "Use a larger mouth.",
        _ => "Use the character's natural mouth size.",
    };
    format!(
        "{}\n{}\n{}\n{}\nEdit only the eyes or mouth requested. Do not alter the nose; preserve the nose pixels exactly. Keep every other pixel, character identity, pose, face shape, hair, accessories, neck, choker, clothes, lighting, and background unchanged. The first image is the only edit canvas and must remain the base image. If a second image is provided, it is a transparent local feature guide, not a canvas; use it only for iris design, mouth-interior colors, teeth color, and local feature style. Never copy the second image's pose, face shape, background, crop, or full character.",
        request.prompt, target_instruction, corner, size
    )
}

fn generate_openai(
    api_key: &str,
    model: &str,
    quality: &str,
    prompt: &str,
    source_png: &[u8],
    reference_png: Option<&[u8]>,
    mask: &GrayImage,
) -> Result<Vec<u8>, AppError> {
    let mut form = Form::new()
        .text("model", model.to_string())
        .text("quality", quality.to_string())
        .text("output_format", "png")
        .text("prompt", prompt.to_string())
        .part(
            "image[]",
            Part::bytes(source_png.to_vec())
                .file_name("source.png")
                .mime_str("image/png")
                .map_err(|error| AppError::General(error.to_string()))?,
        )
        .part(
            "mask",
            Part::bytes(openai_mask_bytes(mask)?)
                .file_name("mask.png")
                .mime_str("image/png")
                .map_err(|error| AppError::General(error.to_string()))?,
        );
    if let Some(reference) = reference_png {
        form = form.part(
            "image[]",
            Part::bytes(reference.to_vec())
                .file_name("reference.png")
                .mime_str("image/png")
                .map_err(|error| AppError::General(error.to_string()))?,
        );
    }
    let response = image_api_client("OpenAI")?
        .post("https://api.openai.com/v1/images/edits")
        .bearer_auth(api_key)
        .multipart(form)
        .send()
        .map_err(|error| image_api_request_error("OpenAI", error))?;
    let status = response.status();
    let value: serde_json::Value = response
        .json()
        .map_err(|error| AppError::General(format!("OpenAI API応答の解析失敗: {error}")))?;
    if !status.is_success() {
        return Err(AppError::General(format!(
            "OpenAI API error: HTTP {status} {value}"
        )));
    }
    let encoded = value["data"][0]["b64_json"]
        .as_str()
        .ok_or_else(|| AppError::General(format!("OpenAI API応答に画像がありません: {value}")))?;
    STANDARD
        .decode(encoded)
        .map_err(|error| AppError::General(format!("OpenAI画像デコード失敗: {error}")))
}

fn generate_gemini(
    api_key: &str,
    model: &str,
    quality: &str,
    prompt: &str,
    source_png: &[u8],
    reference_png: Option<&[u8]>,
) -> Result<Vec<u8>, AppError> {
    let mut parts = vec![
        json!({"text": prompt}),
        json!({"inline_data": {"mime_type": "image/png", "data": STANDARD.encode(source_png)}}),
    ];
    if let Some(reference) = reference_png {
        parts.push(
            json!({"inline_data": {"mime_type": "image/png", "data": STANDARD.encode(reference)}}),
        );
    }
    let image_size = if quality == "high" { "4K" } else { "2K" };
    let body = json!({
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "responseFormat": {"image": {"imageSize": image_size}}
        }
    });
    let url =
        format!("https://generativelanguage.googleapis.com/v1/models/{model}:generateContent");
    let response = image_api_client("Gemini")?
        .post(url)
        .header("x-goog-api-key", api_key)
        .json(&body)
        .send()
        .map_err(|error| image_api_request_error("Gemini", error))?;
    let status = response.status();
    let value: serde_json::Value = response
        .json()
        .map_err(|error| AppError::General(format!("Gemini API応答の解析失敗: {error}")))?;
    if !status.is_success() {
        return Err(AppError::General(format!(
            "Gemini API error: HTTP {status} {value}"
        )));
    }
    let encoded = value["candidates"][0]["content"]["parts"]
        .as_array()
        .and_then(|parts| {
            parts.iter().find_map(|part| {
                part.get("inlineData")
                    .or_else(|| part.get("inline_data"))
                    .and_then(|inline| inline.get("data"))
                    .and_then(|data| data.as_str())
            })
        })
        .ok_or_else(|| AppError::General(format!("Gemini API応答に画像がありません: {value}")))?;
    STANDARD
        .decode(encoded)
        .map_err(|error| AppError::General(format!("Gemini画像デコード失敗: {error}")))
}

fn png_bytes(image: &DynamicImage) -> Result<Vec<u8>, AppError> {
    let mut cursor = Cursor::new(Vec::new());
    image
        .write_to(&mut cursor, ImageFormat::Png)
        .map_err(|error| AppError::General(format!("PNG変換失敗: {error}")))?;
    Ok(cursor.into_inner())
}

fn image_api_client(provider: &str) -> Result<reqwest::blocking::Client, AppError> {
    reqwest::blocking::Client::builder()
        .connect_timeout(IMAGE_API_CONNECT_TIMEOUT)
        .timeout(IMAGE_API_TIMEOUT)
        .build()
        .map_err(|error| {
            AppError::General(format!(
                "{provider} API用HTTPクライアントの初期化に失敗: {error}"
            ))
        })
}

fn image_api_request_error(provider: &str, error: reqwest::Error) -> AppError {
    if error.is_timeout() {
        return AppError::General(format!(
            "{provider} APIが10分以内に応答しませんでした。自動再送は二重課金防止のため行いません"
        ));
    }
    if error.is_connect() {
        return AppError::General(format!(
            "{provider} APIへ接続できませんでした。ネットワーク・TLS・プロキシ設定を確認してください: {error:?}"
        ));
    }
    AppError::General(format!(
        "{provider} APIとの通信に失敗しました。自動再送は行いません: {error:?}"
    ))
}

fn read_reference_image(path: &str) -> Result<DynamicImage, AppError> {
    image::open(path).map_err(|error| AppError::General(format!("参照画像の読込に失敗: {error}")))
}

fn reference_feature_guide_png(
    reference: &DynamicImage,
    mask: &GrayImage,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, AppError> {
    let Some((min_x, min_y, max_x, max_y)) = gray_mask_bounds(mask) else {
        return png_bytes(&DynamicImage::ImageRgba8(RgbaImage::new(width, height)));
    };
    let margin_x = (width / 32).clamp(24, 72);
    let margin_y = (height / 32).clamp(24, 72);
    let left = min_x.saturating_sub(margin_x);
    let top = min_y.saturating_sub(margin_y);
    let right = max_x.saturating_add(margin_x).min(width.saturating_sub(1));
    let bottom = max_y.saturating_add(margin_y).min(height.saturating_sub(1));

    let reference = reference
        .resize_exact(width, height, image::imageops::FilterType::Lanczos3)
        .to_rgba8();
    let mut guide = RgbaImage::new(width, height);
    for y in top..=bottom {
        for x in left..=right {
            guide.put_pixel(x, y, *reference.get_pixel(x, y));
        }
    }
    png_bytes(&DynamicImage::ImageRgba8(guide))
}

fn openai_mask_bytes(mask: &GrayImage) -> Result<Vec<u8>, AppError> {
    let mut rgba = RgbaImage::new(mask.width(), mask.height());
    for (x, y, pixel) in rgba.enumerate_pixels_mut() {
        let editable = mask.get_pixel(x, y)[0];
        *pixel = image::Rgba([0, 0, 0, 255u8.saturating_sub(editable)]);
    }
    png_bytes(&DynamicImage::ImageRgba8(rgba))
}

fn save_openai_mask(mask: &GrayImage, path: &Path) -> Result<(), AppError> {
    fs::write(path, openai_mask_bytes(mask)?)?;
    Ok(())
}

fn composite_inside_mask(
    source: &DynamicImage,
    candidate: &DynamicImage,
    mask: &GrayImage,
    width: u32,
    height: u32,
) -> DynamicImage {
    let source = source.to_rgba8();
    let candidate = candidate
        .resize_exact(width, height, image::imageops::FilterType::Lanczos3)
        .to_rgba8();
    let mut result = source.clone();
    for y in 0..height {
        for x in 0..width {
            let alpha = mask.get_pixel(x, y)[0] as f32 / 255.0;
            if alpha <= 0.0 {
                continue;
            }
            let original = source.get_pixel(x, y);
            let generated = candidate.get_pixel(x, y);
            result.put_pixel(
                x,
                y,
                image::Rgba([
                    (original[0] as f32 * (1.0 - alpha) + generated[0] as f32 * alpha) as u8,
                    (original[1] as f32 * (1.0 - alpha) + generated[1] as f32 * alpha) as u8,
                    (original[2] as f32 * (1.0 - alpha) + generated[2] as f32 * alpha) as u8,
                    original[3],
                ]),
            );
        }
    }
    DynamicImage::ImageRgba8(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma, Rgba};

    #[test]
    fn generated_part_auto_fit_accepts_mild_aspect_difference() {
        // 完全一致は当然OK
        assert!(generated_part_size_is_auto_fittable(1792, 2392, 1792, 2392));
        // 同アスペクト別解像度（従来から受け入れ）
        assert!(generated_part_size_is_auto_fittable(1792, 2392, 896, 1196));
        // 実例: Nano Banana立ち絵(1792x2392, 0.749) + Codex口差分(1024x1536, 0.667)
        // → 歪み比1.124で許容内
        assert!(generated_part_size_is_auto_fittable(1792, 2392, 1024, 1536));
        // 縦横逆転（歪み比>2）はブロック
        assert!(!generated_part_size_is_auto_fittable(1792, 2392, 1536, 1024));
        // 正方形 vs 3:4級（歪み比1.33）もブロック
        assert!(!generated_part_size_is_auto_fittable(1024, 1024, 768, 1024));
        // ゼロサイズは安全側で拒否
        assert!(!generated_part_size_is_auto_fittable(1792, 0, 1024, 1536));
        assert!(!generated_part_size_is_auto_fittable(1792, 2392, 1024, 0));
    }

    #[test]
    fn resolve_base_draw_order_inserts_missing_groups() {
        // layer-order.json は neck / sways を含まない → 既定の相対位置へ補完される
        let custom: Vec<String> = [
            "hair_back",
            "arm_l",
            "arm_r",
            "body",
            "chest",
            "eye",
            "mouth",
            "hair",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let order = resolve_base_draw_order(&custom);
        // sways は既定順の直前要素 arm_r の直後へ補完される
        // （sways が custom に無い = swayパーツ未使用なので実描画には影響しない）
        assert_eq!(
            order,
            vec![
                "hair_back",
                "arm_l",
                "arm_r",
                "sways",
                "body",
                "neck",
                "chest",
                "eye",
                "mouth",
                "hair"
            ]
        );
        // 空 = 既定順そのまま
        assert_eq!(
            resolve_base_draw_order(&[]),
            vec![
                "hair_back",
                "body",
                "neck",
                "chest",
                "arm_l",
                "arm_r",
                "sways",
                "eye",
                "mouth",
                "hair"
            ]
        );
    }

    #[test]
    fn resolve_base_draw_order_preserves_unique_individual_sways() {
        let custom: Vec<String> = [
            "hair_back",
            "body",
            "sway_ear_r",
            "sway_ear_l",
            "hair",
            "sway_ear",
            "sway_ear",
            "unknown",
        ]
        .iter()
        .map(|key| (*key).to_string())
        .collect();

        let order = resolve_base_draw_order(&custom);
        let position = |key: &str| order.iter().position(|entry| entry == key).unwrap();

        assert!(position("sway_ear_r") < position("sway_ear_l"));
        assert!(position("sway_ear_l") < position("hair"));
        assert!(position("hair") < position("sway_ear"));
        assert_eq!(order.iter().filter(|key| *key == "sway_ear").count(), 1);
        assert!(!order.iter().any(|key| key == "unknown"));
    }

    #[test]
    fn resolve_base_draw_order_preserves_arm_overlay_position() {
        let overlay = "arm_l_overlay_patch_fingers";
        let custom: Vec<String> = ["arm_l", "body", overlay, "hair"]
            .iter()
            .map(|key| (*key).to_string())
            .collect();

        let order = resolve_base_draw_order(&custom);
        let position = |key: &str| order.iter().position(|entry| entry == key).unwrap();

        assert!(position("arm_l") < position("body"));
        assert!(position("body") < position(overlay));
        assert!(position(overlay) < position("hair"));
        assert_eq!(order.iter().filter(|key| *key == overlay).count(), 1);
    }

    #[test]
    fn layer_order_document_describes_linked_arm_overlays() {
        let order: Vec<String> = [
            "arm_l",
            "body",
            "arm_l_overlay_patch_fingers",
            "arm_r_overlay_patch_sleeve",
            "hair",
        ]
        .iter()
        .map(|key| (*key).to_string())
        .collect();

        let document = layer_order_document(&order);

        assert_eq!(document["formatVersion"], 1);
        assert_eq!(document["drawOrder"], serde_json::json!(order));
        assert_eq!(
            document["linkedParts"]["arm_l_overlay_patch_fingers"]["parent"],
            "arm_l"
        );
        assert_eq!(
            document["linkedParts"]["arm_r_overlay_patch_sleeve"]["parent"],
            "arm_r"
        );

        let legacy = layer_order_document(&["body".into(), "hair".into()]);
        assert!(legacy.get("linkedParts").is_none());
    }

    #[test]
    fn save_and_load_base_parts_preserves_arm_overlays_and_removes_stale_files() {
        let root = std::env::temp_dir().join(format!(
            "pachipakugen_arm_overlay_parts_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        let base_dir = root.join("base_parts");
        fs::create_dir_all(&base_dir).unwrap();
        image::RgbaImage::from_pixel(2, 2, image::Rgba([255, 255, 0, 255]))
            .save(base_dir.join("arm_l_overlay_stale.png"))
            .unwrap();

        let mut parts = HashMap::new();
        parts.insert(
            "body".into(),
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(2, 2, Rgba([1, 2, 3, 255]))),
        );
        parts.insert(
            "arm_l_overlay_patch_fingers".into(),
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(2, 2, Rgba([4, 5, 6, 255]))),
        );
        parts.insert(
            "eyewhite".into(),
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(2, 2, Rgba([7, 8, 9, 255]))),
        );
        parts.insert(
            "irides".into(),
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(2, 2, Rgba([10, 11, 12, 255]))),
        );
        parts.insert(
            "eyebrow".into(),
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(2, 2, Rgba([13, 14, 15, 255]))),
        );

        save_base_parts(&parts, &base_dir).unwrap();
        let loaded = load_job_base_parts(&root).unwrap().unwrap();

        assert!(!base_dir.join("arm_l_overlay_stale.png").exists());
        assert!(base_dir.join("arm_l_overlay_patch_fingers.png").is_file());
        assert!(loaded.contains_key("body"));
        assert!(loaded.contains_key("arm_l_overlay_patch_fingers"));
        assert!(loaded.contains_key("eyewhite"));
        assert!(loaded.contains_key("irides"));
        assert!(loaded.contains_key("eyebrow"));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn base_only_preview_skips_expression_frame_decoding() {
        let root = std::env::temp_dir().join(format!(
            "pachipakugen_base_only_preview_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        let base_dir = root.join("base_parts");
        let extracted_dir = root.join("extracted_parts");
        fs::create_dir_all(&base_dir).unwrap();
        fs::create_dir_all(&extracted_dir).unwrap();
        // base-only復元では、存在する差分画像にも触れないことを壊れたPNGで保証する。
        fs::write(extracted_dir.join("eyes-closed.png"), b"not a png").unwrap();

        let mut parts = HashMap::new();
        parts.insert(
            "body".into(),
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(2, 2, Rgba([1, 2, 3, 255]))),
        );

        let preview = preview_from_base_parts(&parts, &extracted_dir, &base_dir, false).unwrap();

        assert!(preview.previews.is_empty());
        assert!(preview.base_preview.starts_with("data:image/png;base64,"));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn expression_previews_include_the_non_selected_face_part() {
        let root = std::env::temp_dir().join(format!(
            "pachipakugen_companion_preview_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        let base_dir = root.join("base_parts");
        let extracted_dir = root.join("extracted_parts");
        fs::create_dir_all(&base_dir).unwrap();
        fs::create_dir_all(&extracted_dir).unwrap();

        let transparent = Rgba([0, 0, 0, 0]);
        let mut parts = HashMap::new();
        parts.insert(
            "body".into(),
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(4, 4, transparent)),
        );
        let mut eyebrow = RgbaImage::from_pixel(4, 4, transparent);
        eyebrow.put_pixel(2, 0, Rgba([255, 0, 255, 255]));
        parts.insert("eyebrow".into(), DynamicImage::ImageRgba8(eyebrow));
        let mut eyes_open = RgbaImage::from_pixel(4, 4, transparent);
        eyes_open.put_pixel(0, 0, Rgba([255, 0, 0, 255]));
        eyes_open.put_pixel(2, 0, Rgba([70, 40, 20, 255]));
        eyes_open.save(extracted_dir.join("eyes-open.png")).unwrap();
        let mut eyes_closed = RgbaImage::from_pixel(4, 4, transparent);
        eyes_closed.put_pixel(1, 0, Rgba([0, 255, 0, 255]));
        eyes_closed.put_pixel(2, 0, Rgba([60, 30, 15, 255]));
        eyes_closed
            .save(extracted_dir.join("eyes-closed.png"))
            .unwrap();
        let mut mouth_closed = RgbaImage::from_pixel(4, 4, transparent);
        mouth_closed.put_pixel(0, 1, Rgba([0, 0, 255, 255]));
        mouth_closed
            .save(extracted_dir.join("mouth-closed.png"))
            .unwrap();
        let mut mouth_a = RgbaImage::from_pixel(4, 4, transparent);
        mouth_a.put_pixel(1, 1, Rgba([255, 255, 0, 255]));
        mouth_a.save(extracted_dir.join("mouth-a.png")).unwrap();

        let preview = preview_from_base_parts(&parts, &extracted_dir, &base_dir, true).unwrap();
        let decode = |data_url: &str| {
            let encoded = data_url.split_once(',').unwrap().1;
            let bytes = STANDARD.decode(encoded).unwrap();
            image::load_from_memory(&bytes).unwrap().to_rgba8()
        };
        let eye_preview = decode(
            &preview
                .previews
                .iter()
                .find(|item| item.part == "eyes-closed")
                .unwrap()
                .preview,
        );
        assert_eq!(eye_preview.get_pixel(1, 0), &Rgba([0, 255, 0, 255]));
        assert_eq!(eye_preview.get_pixel(0, 1), &Rgba([0, 0, 255, 255]));
        assert_eq!(eye_preview.get_pixel(2, 0), &Rgba([255, 0, 255, 255]));

        let mouth_preview = decode(
            &preview
                .previews
                .iter()
                .find(|item| item.part == "mouth-a")
                .unwrap()
                .preview,
        );
        assert_eq!(mouth_preview.get_pixel(0, 0), &Rgba([255, 0, 0, 255]));
        assert_eq!(mouth_preview.get_pixel(1, 1), &Rgba([255, 255, 0, 255]));
        assert_eq!(mouth_preview.get_pixel(2, 0), &Rgba([255, 0, 255, 255]));
        let base_preview = decode(&preview.base_preview);
        assert_eq!(base_preview.get_pixel(2, 0), &Rgba([255, 0, 255, 255]));
        assert_eq!(base_preview.get_pixel(0, 0)[3], 0);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn eyes_open_can_be_adjusted_without_becoming_a_generated_requirement() {
        assert!(!GENERATED_PART_TARGETS.contains(&"eyes-open"));
        assert!(ADJUSTABLE_PART_TARGETS.contains(&"eyes-open"));

        let root = std::env::temp_dir().join(format!(
            "pachipakugen_adjust_eyes_open_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        let extracted_dir = root.join("extracted_parts");
        let base_dir = root.join("base_parts");
        let rife_dir = root.join("rife_output");
        fs::create_dir_all(&extracted_dir).unwrap();
        fs::create_dir_all(&base_dir).unwrap();
        fs::create_dir_all(&rife_dir).unwrap();
        let mut original = RgbaImage::new(4, 4);
        original.put_pixel(1, 1, Rgba([10, 20, 30, 255]));
        original.save(extracted_dir.join("eyes-open.png")).unwrap();
        let mut iris = RgbaImage::new(4, 4);
        iris.put_pixel(1, 1, Rgba([0, 200, 255, 255]));
        iris.save(base_dir.join("irides.png")).unwrap();

        let result = adjust_codex_extracted_parts_inner(AdjustCodexExtractedPartsRequest {
            job_path: root.to_string_lossy().into_owned(),
            offset_x: 1,
            offset_y: 0,
            scale_percent: 100,
            part: Some("eyes-open".into()),
        })
        .unwrap();

        assert_eq!(result.adjusted_parts, vec!["eyes-open"]);
        assert_eq!(result.part_adjustments["eyes-open"].offset_x, 1);
        let adjusted = image::open(extracted_dir.join("eyes-open.png"))
            .unwrap()
            .to_rgba8();
        assert_eq!(adjusted.get_pixel(2, 1), &Rgba([10, 20, 30, 255]));
        let synced_iris = image::open(rife_dir.join("irides.png")).unwrap().to_rgba8();
        assert_eq!(synced_iris.get_pixel(2, 1), &Rgba([0, 200, 255, 255]));
        let preserved = image::open(
            extracted_dir
                .join("original_extracted_parts")
                .join("eyes-open.png"),
        )
        .unwrap()
        .to_rgba8();
        assert_eq!(preserved.get_pixel(1, 1), &Rgba([10, 20, 30, 255]));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn refreshing_eyes_open_preserves_existing_step5_adjustment() {
        let root = std::env::temp_dir().join(format!(
            "pachipakugen_refresh_eyes_open_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        let extracted_dir = root.join("extracted_parts");
        fs::create_dir_all(&extracted_dir).unwrap();
        fs::write(
            extracted_dir.join("adjustment.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "formatVersion": 2,
                "adjustedParts": ["eyes-open"],
                "parts": {
                    "eyes-open": { "offsetX": 1, "offsetY": 0, "scalePercent": 100 }
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let mut eye = RgbaImage::new(4, 4);
        eye.put_pixel(1, 1, Rgba([10, 20, 30, 255]));
        let mut parts = HashMap::new();
        parts.insert("eye_open".into(), DynamicImage::ImageRgba8(eye));

        save_eyes_open_extracted_part(&root, &parts, &extracted_dir).unwrap();

        let displayed = image::open(extracted_dir.join("eyes-open.png"))
            .unwrap()
            .to_rgba8();
        assert_eq!(displayed.get_pixel(2, 1), &Rgba([10, 20, 30, 255]));
        assert_eq!(displayed.get_pixel(1, 1), &Rgba([0, 0, 0, 0]));

        let original = image::open(
            extracted_dir
                .join("original_extracted_parts")
                .join("eyes-open.png"),
        )
        .unwrap()
        .to_rgba8();
        assert_eq!(original.get_pixel(1, 1), &Rgba([10, 20, 30, 255]));
        assert_eq!(
            read_typed_part_adjustments(&extracted_dir)["eyes-open"].offset_x,
            1
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn individual_adjustments_accumulate_and_reset_from_original() {
        let root = std::env::temp_dir().join(format!(
            "pachipakugen_adjust_reset_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        let extracted_dir = root.join("extracted_parts");
        fs::create_dir_all(&extracted_dir).unwrap();

        let mut eye = RgbaImage::new(4, 4);
        eye.put_pixel(1, 1, Rgba([10, 20, 30, 255]));
        eye.save(extracted_dir.join("eyes-open.png")).unwrap();
        let mut mouth = RgbaImage::new(4, 4);
        mouth.put_pixel(1, 1, Rgba([40, 50, 60, 255]));
        mouth.save(extracted_dir.join("mouth-closed.png")).unwrap();

        let batch =
            adjust_codex_extracted_parts_batch_inner(AdjustCodexExtractedPartsBatchRequest {
                job_path: root.to_string_lossy().into_owned(),
                adjustments: vec![
                    CodexPartAdjustmentUpdate {
                        part: "eyes-open".into(),
                        offset_x: 1,
                        offset_y: 0,
                        scale_percent: 100,
                    },
                    CodexPartAdjustmentUpdate {
                        part: "mouth-closed".into(),
                        offset_x: 0,
                        offset_y: 1,
                        scale_percent: 100,
                    },
                ],
            })
            .unwrap();
        assert_eq!(batch.adjusted_parts, vec!["eyes-open", "mouth-closed"]);
        let reset = adjust_codex_extracted_parts_inner(AdjustCodexExtractedPartsRequest {
            job_path: root.to_string_lossy().into_owned(),
            offset_x: 0,
            offset_y: 0,
            scale_percent: 100,
            part: Some("eyes-open".into()),
        })
        .unwrap();

        assert!(!reset.part_adjustments.contains_key("eyes-open"));
        assert_eq!(reset.part_adjustments["mouth-closed"].offset_y, 1);
        let restored_eye = image::open(extracted_dir.join("eyes-open.png"))
            .unwrap()
            .to_rgba8();
        assert_eq!(restored_eye.get_pixel(1, 1), &Rgba([10, 20, 30, 255]));
        assert_eq!(restored_eye.get_pixel(2, 1), &Rgba([0, 0, 0, 0]));
        let adjusted_mouth = image::open(extracted_dir.join("mouth-closed.png"))
            .unwrap()
            .to_rgba8();
        assert_eq!(adjusted_mouth.get_pixel(1, 2), &Rgba([40, 50, 60, 255]));

        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(extracted_dir.join("adjustment.json")).unwrap())
                .unwrap();
        assert_eq!(manifest["adjustedParts"], json!(["mouth-closed"]));
        assert!(manifest["parts"].get("eyes-open").is_none());

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn batch_adjustment_failure_restores_earlier_parts_and_manifest() {
        let root = std::env::temp_dir().join(format!(
            "pachipakugen_adjust_batch_rollback_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        let extracted_dir = root.join("extracted_parts");
        let base_dir = root.join("base_parts");
        let rife_dir = root.join("rife_output");
        fs::create_dir_all(&extracted_dir).unwrap();
        fs::create_dir_all(&base_dir).unwrap();
        fs::create_dir_all(&rife_dir).unwrap();
        let mut eye = RgbaImage::new(4, 4);
        eye.put_pixel(1, 1, Rgba([70, 80, 90, 255]));
        let eye_path = extracted_dir.join("eyes-open.png");
        eye.save(&eye_path).unwrap();
        let original_eye_bytes = fs::read(&eye_path).unwrap();
        let mut iris = RgbaImage::new(4, 4);
        iris.put_pixel(1, 1, Rgba([0, 200, 255, 255]));
        iris.save(base_dir.join("irides.png")).unwrap();
        let old_output_iris = RgbaImage::from_pixel(4, 4, Rgba([90, 0, 120, 255]));
        let output_iris_path = rife_dir.join("irides.png");
        old_output_iris.save(&output_iris_path).unwrap();
        let original_output_iris_bytes = fs::read(&output_iris_path).unwrap();
        // 1件目の目は保存できるが、2件目の口は画像デコードで失敗する。
        fs::write(extracted_dir.join("mouth-closed.png"), b"not a png").unwrap();

        let result =
            adjust_codex_extracted_parts_batch_inner(AdjustCodexExtractedPartsBatchRequest {
                job_path: root.to_string_lossy().into_owned(),
                adjustments: vec![
                    CodexPartAdjustmentUpdate {
                        part: "eyes-open".into(),
                        offset_x: 1,
                        offset_y: 0,
                        scale_percent: 100,
                    },
                    CodexPartAdjustmentUpdate {
                        part: "mouth-closed".into(),
                        offset_x: 1,
                        offset_y: 0,
                        scale_percent: 100,
                    },
                ],
            });

        assert!(result.is_err());
        assert_eq!(fs::read(&eye_path).unwrap(), original_eye_bytes);
        assert_eq!(
            fs::read(&output_iris_path).unwrap(),
            original_output_iris_bytes
        );
        assert!(!extracted_dir.join("adjustment.json").exists());

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn workspace_adjustment_updates_checkpoint_and_preflight_failure_preserves_it() {
        let root = std::env::temp_dir().join(format!(
            "pachipakugen_adjust_workspace_checkpoint_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        let extracted_dir = root.join(WORKSPACE_SEE_THROUGH_DIR);
        fs::create_dir_all(&extracted_dir).unwrap();
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
        let mut eye = RgbaImage::new(4, 4);
        eye.put_pixel(1, 1, Rgba([10, 20, 30, 255]));
        eye.save(extracted_dir.join("eyes-open.png")).unwrap();
        write_project(7);

        adjust_codex_extracted_parts_batch_inner(AdjustCodexExtractedPartsBatchRequest {
            job_path: root.to_string_lossy().into_owned(),
            adjustments: vec![CodexPartAdjustmentUpdate {
                part: "eyes-open".into(),
                offset_x: 1,
                offset_y: 0,
                scale_percent: 100,
            }],
        })
        .unwrap();
        let project: WorkspaceProject =
            serde_json::from_slice(&fs::read(root.join("project.json")).unwrap()).unwrap();
        assert_eq!(project.current_step, 6);

        fs::remove_file(extracted_dir.join("eyes-open.png")).unwrap();
        write_project(7);
        let error = adjust_codex_extracted_parts_inner(AdjustCodexExtractedPartsRequest {
            job_path: root.to_string_lossy().into_owned(),
            offset_x: 1,
            offset_y: 0,
            scale_percent: 100,
            part: Some("eyes-open".into()),
        })
        .unwrap_err();
        assert!(error.to_string().contains("調整対象の差分パーツ"));
        let project: WorkspaceProject =
            serde_json::from_slice(&fs::read(root.join("project.json")).unwrap()).unwrap();
        assert_eq!(project.current_step, 7);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn compose_base_parts_ordered_respects_arm_behind_body() {
        // 腕(赤)が body(青) の背面指定なら、重なり部は body の色になる
        let mut parts: HashMap<String, DynamicImage> = HashMap::new();
        parts.insert(
            "body".into(),
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(2, 2, Rgba([0, 0, 255, 255]))),
        );
        parts.insert(
            "arm_l".into(),
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(2, 2, Rgba([255, 0, 0, 255]))),
        );
        let behind: Vec<String> = ["arm_l", "body"].iter().map(|s| s.to_string()).collect();
        let composite = compose_base_parts_ordered(&parts, 2, 2, &behind, None, None, true);
        assert_eq!(composite.get_pixel(0, 0), &Rgba([0, 0, 255, 255]));
        let front: Vec<String> = ["body", "arm_l"].iter().map(|s| s.to_string()).collect();
        let composite = compose_base_parts_ordered(&parts, 2, 2, &front, None, None, true);
        assert_eq!(composite.get_pixel(0, 0), &Rgba([255, 0, 0, 255]));
    }

    #[test]
    fn compose_base_parts_ordered_draws_arm_overlay_at_its_own_depth() {
        let mut parts: HashMap<String, DynamicImage> = HashMap::new();
        parts.insert(
            "body".into(),
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(2, 2, Rgba([0, 0, 255, 255]))),
        );
        parts.insert(
            "arm_l".into(),
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(2, 2, Rgba([255, 0, 0, 255]))),
        );
        parts.insert(
            "arm_l_overlay_patch_fingers".into(),
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(2, 2, Rgba([0, 255, 0, 255]))),
        );

        let order: Vec<String> = ["arm_l", "body", "arm_l_overlay_patch_fingers"]
            .iter()
            .map(|key| (*key).to_string())
            .collect();
        let composite = compose_base_parts_ordered(&parts, 2, 2, &order, None, None, true);

        assert_eq!(composite.get_pixel(0, 0), &Rgba([0, 255, 0, 255]));
    }

    #[test]
    fn compose_base_parts_ordered_draws_explicit_sway_only_at_its_position() {
        let mut parts: HashMap<String, DynamicImage> = HashMap::new();
        parts.insert(
            "sway_ear".into(),
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(2, 2, Rgba([255, 0, 0, 255]))),
        );
        parts.insert(
            "hair".into(),
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(2, 2, Rgba([0, 0, 255, 255]))),
        );

        // 個別swayをhairの背面に指定した場合、後方互換`sways`で再描画されてはならない。
        let explicit: Vec<String> = ["sway_ear", "hair", "sways"]
            .iter()
            .map(|key| (*key).to_string())
            .collect();
        let composite = compose_base_parts_ordered(&parts, 2, 2, &explicit, None, None, true);
        assert_eq!(composite.get_pixel(0, 0), &Rgba([0, 0, 255, 255]));

        // 旧形式は`sways`位置で従来どおり全swayを描画する。
        let legacy: Vec<String> = ["hair", "sways"]
            .iter()
            .map(|key| (*key).to_string())
            .collect();
        let composite = compose_base_parts_ordered(&parts, 2, 2, &legacy, None, None, true);
        assert_eq!(composite.get_pixel(0, 0), &Rgba([255, 0, 0, 255]));
    }

    #[test]
    fn existing_eyes_open_is_reported_once_in_extracted_parts() {
        let root = std::env::temp_dir().join(format!(
            "pachipakugen_eyes_open_result_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("eyes-open.png"), []).unwrap();
        let mut parts = vec!["mouth-a".to_string()];

        include_existing_eyes_open(&root, &mut parts);
        include_existing_eyes_open(&root, &mut parts);

        assert_eq!(parts.first().map(String::as_str), Some("eyes-open"));
        assert_eq!(parts.iter().filter(|part| *part == "eyes-open").count(), 1);

        fs::remove_file(root.join("eyes-open.png")).unwrap();
        let mut missing = vec!["mouth-a".to_string()];
        include_existing_eyes_open(&root, &mut missing);
        assert_eq!(missing, vec!["mouth-a"]);

        let _ = fs::remove_dir_all(root);
    }

    fn request(targets: &[&str]) -> GenerateExpressionSetRequest {
        GenerateExpressionSetRequest {
            engine: "gpt".into(),
            quality: "medium".into(),
            targets: targets.iter().map(|target| (*target).into()).collect(),
            reference_image_path: None,
            prompt: "Preserve the character.".into(),
            mouth_corner: "neutral".into(),
            mouth_size: "normal".into(),
            output_path: String::new(),
            rife_frame_count: None,
        }
    }

    #[test]
    fn mouth_i_prompt_requires_visible_teeth() {
        let prompt = target_prompt("mouth-i", &request(&["mouth-i"]));
        assert!(prompt.contains("clean white teeth strip"));
        assert!(prompt.contains("minimal dark mouth interior"));
        assert!(prompt.contains("Do not alter the nose"));
        assert!(prompt.contains("first image is the only edit canvas"));
        assert!(prompt.contains("Keep every other pixel"));
    }

    #[test]
    fn mouth_vowel_requests_auto_include_closed_baseline() {
        let temp_dir = std::env::temp_dir().join("pachipakugen_effective_targets_test");
        let requested = vec!["mouth-a".to_string(), "mouth-i".to_string()];

        let targets = effective_expression_targets(&requested, &temp_dir);

        assert_eq!(targets.first().map(String::as_str), Some("mouth-closed"));
        assert!(targets.iter().any(|target| target == "mouth-a"));
        assert!(targets.iter().any(|target| target == "mouth-i"));
    }

    #[test]
    fn workspace_paths_are_used_as_codex_job_dirs() {
        let root = std::env::temp_dir().join(format!(
            "pachipakugen_workspace_paths_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(root.join(WORKSPACE_CODEX_REQUEST_DIR)).unwrap();
        fs::create_dir_all(root.join(WORKSPACE_GENERATED_PARTS_DIR)).unwrap();
        fs::create_dir_all(root.join(WORKSPACE_SEE_THROUGH_DIR)).unwrap();
        fs::create_dir_all(root.join(WORKSPACE_SPRITALK_PARTS_DIR)).unwrap();
        fs::write(
            root.join(WORKSPACE_CODEX_REQUEST_DIR).join("source.png"),
            [],
        )
        .unwrap();
        fs::write(
            root.join(WORKSPACE_CODEX_REQUEST_DIR)
                .join("codex_request.md"),
            [],
        )
        .unwrap();
        fs::write(
            root.join(WORKSPACE_CODEX_REQUEST_DIR)
                .join("codex_handoff.md"),
            [],
        )
        .unwrap();

        assert_eq!(
            job_source_path(&root),
            root.join(WORKSPACE_CODEX_REQUEST_DIR).join("source.png")
        );
        assert_eq!(
            job_request_path(&root),
            root.join(WORKSPACE_CODEX_REQUEST_DIR)
                .join("codex_request.md")
        );
        assert_eq!(
            job_handoff_path(&root),
            root.join(WORKSPACE_CODEX_REQUEST_DIR)
                .join("codex_handoff.md")
        );
        assert_eq!(
            generated_parts_dir(&root),
            root.join(WORKSPACE_GENERATED_PARTS_DIR)
        );
        assert_eq!(
            extracted_parts_dir(&root),
            root.join(WORKSPACE_SEE_THROUGH_DIR)
        );
        assert_eq!(
            rife_output_dir(&root),
            root.join(WORKSPACE_SPRITALK_PARTS_DIR)
        );
        assert_eq!(
            base_parts_dir(&root),
            root.join(WORKSPACE_SEE_THROUGH_DIR).join("base_parts")
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn spritalk_output_materializes_spritalk_import_assets_at_root() {
        let root = std::env::temp_dir().join(format!(
            "pachipakugen_spritalk_assets_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        let base_dir = root.join(WORKSPACE_SEE_THROUGH_DIR).join("base_parts");
        let extracted_dir = root.join(WORKSPACE_SEE_THROUGH_DIR);
        let output_dir = root.join(WORKSPACE_SPRITALK_PARTS_DIR);
        fs::create_dir_all(&base_dir).unwrap();
        fs::create_dir_all(&extracted_dir).unwrap();
        image::RgbaImage::from_pixel(2, 2, image::Rgba([1, 2, 3, 255]))
            .save(base_dir.join("body.png"))
            .unwrap();
        image::RgbaImage::from_pixel(2, 2, image::Rgba([7, 8, 9, 192]))
            .save(base_dir.join("hair.png"))
            .unwrap();
        image::RgbaImage::from_pixel(2, 2, image::Rgba([10, 20, 30, 255]))
            .save(base_dir.join("sway_ear.png"))
            .unwrap();
        image::RgbaImage::from_pixel(2, 2, image::Rgba([40, 50, 60, 255]))
            .save(base_dir.join("arm_l_overlay_patch_fingers.png"))
            .unwrap();
        let mut eyewhite = image::RgbaImage::new(4, 4);
        eyewhite.put_pixel(1, 1, image::Rgba([240, 240, 240, 255]));
        eyewhite.save(base_dir.join("eyewhite.png")).unwrap();
        let mut eyebrow = image::RgbaImage::new(4, 4);
        eyebrow.put_pixel(1, 1, image::Rgba([35, 20, 15, 255]));
        eyebrow.save(base_dir.join("eyebrow.png")).unwrap();
        let mut irides = image::RgbaImage::new(4, 4);
        irides.put_pixel(1, 1, image::Rgba([0, 200, 255, 255]));
        irides.save(base_dir.join("irides.png")).unwrap();
        let layer_order = serde_json::json!({
            "formatVersion": 1,
            "drawOrder": [
                "hair_back",
                "arm_l",
                "body",
                "arm_l_overlay_patch_fingers",
                "hair"
            ],
            "linkedParts": {
                "arm_l_overlay_patch_fingers": { "parent": "arm_l" }
            }
        });
        fs::write(
            base_dir.join("layer-order.json"),
            serde_json::to_vec_pretty(&layer_order).unwrap(),
        )
        .unwrap();
        image::RgbaImage::from_pixel(2, 2, image::Rgba([4, 5, 6, 128]))
            .save(extracted_dir.join("mouth-a.png"))
            .unwrap();
        fs::write(
            extracted_dir.join("adjustment.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "formatVersion": 2,
                "adjustedParts": ["eyes-open"],
                "parts": {
                    "eyes-open": { "offsetX": 1, "offsetY": 0, "scalePercent": 100 }
                }
            }))
            .unwrap(),
        )
        .unwrap();
        fs::create_dir_all(output_dir.join("base_parts")).unwrap();
        fs::create_dir_all(output_dir.join("extracted_parts")).unwrap();
        image::RgbaImage::from_pixel(2, 2, image::Rgba([255, 255, 0, 255]))
            .save(output_dir.join("sway_stale.png"))
            .unwrap();
        image::RgbaImage::from_pixel(2, 2, image::Rgba([255, 0, 255, 255]))
            .save(output_dir.join("arm_r_overlay_stale.png"))
            .unwrap();
        image::RgbaImage::from_pixel(2, 2, image::Rgba([255, 0, 0, 255]))
            .save(output_dir.join("base_parts").join("body.png"))
            .unwrap();

        let copied =
            materialize_spritalk_static_assets(&root, &extracted_dir, &output_dir).unwrap();

        assert!(output_dir.join("body.png").is_file());
        assert!(output_dir.join("hair.png").is_file());
        assert!(output_dir.join("sway_ear.png").is_file());
        assert!(output_dir.join("arm_l_overlay_patch_fingers.png").is_file());
        assert!(output_dir.join("eyewhite.png").is_file());
        assert!(output_dir.join("eyebrow.png").is_file());
        assert!(output_dir.join("irides.png").is_file());
        assert!(!output_dir.join("sway_stale.png").exists());
        assert!(!output_dir.join("arm_r_overlay_stale.png").exists());
        assert_eq!(
            image::open(output_dir.join("sway_ear.png"))
                .unwrap()
                .to_rgba8()
                .get_pixel(0, 0),
            &image::Rgba([10, 20, 30, 255])
        );
        assert_eq!(
            image::open(output_dir.join("arm_l_overlay_patch_fingers.png"))
                .unwrap()
                .to_rgba8()
                .get_pixel(0, 0),
            &image::Rgba([40, 50, 60, 255])
        );
        let copied_layer_order: serde_json::Value =
            serde_json::from_slice(&fs::read(output_dir.join("layer-order.json")).unwrap())
                .unwrap();
        assert_eq!(copied_layer_order, layer_order);
        assert_eq!(
            image::open(output_dir.join("eyewhite.png"))
                .unwrap()
                .to_rgba8()
                .get_pixel(2, 1),
            &image::Rgba([240, 240, 240, 255])
        );
        assert_eq!(
            image::open(output_dir.join("eyebrow.png"))
                .unwrap()
                .to_rgba8()
                .get_pixel(2, 1),
            &image::Rgba([35, 20, 15, 255])
        );
        assert_eq!(
            image::open(output_dir.join("irides.png"))
                .unwrap()
                .to_rgba8()
                .get_pixel(2, 1),
            &image::Rgba([0, 200, 255, 255])
        );
        assert!(!output_dir.join("base_parts").exists());
        assert!(!output_dir.join("extracted_parts").exists());
        assert!(!output_dir.join("mouth-a.png").is_file());
        assert!(output_dir.join("README.txt").is_file());
        assert!(fs::read_to_string(output_dir.join("README.txt"))
            .unwrap()
            .contains("arm_l_overlay_*.png"));
        assert_eq!(copied.len(), 8);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn regressed_workspace_does_not_restore_stale_rife_output() {
        let root = std::env::temp_dir().join(format!(
            "pachipakugen_stale_rife_checkpoint_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        let output_dir = root.join(WORKSPACE_SPRITALK_PARTS_DIR);
        fs::create_dir_all(output_dir.join("eye")).unwrap();
        fs::write(
            output_dir.join("manifest.json"),
            serde_json::to_vec_pretty(&json!({ "frameCount": 8 })).unwrap(),
        )
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

        write_project(7);
        assert!(read_current_rife_output_result(&root).is_some());

        // 再編集後は旧画像を物理的に残していても現行成果物としては扱わない。
        write_project(6);
        assert!(output_dir.join("eye").is_dir());
        assert!(read_rife_output_result(&root).is_some());
        assert!(read_current_rife_output_result(&root).is_none());

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn extracted_part_reuse_requires_matching_visual_fingerprint() {
        let root = std::env::temp_dir().join(format!(
            "pachipakugen_extracted_fingerprint_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(&root).unwrap();
        let output_path = root.join("mouth-a.png");
        let mut output = RgbaImage::new(64, 64);
        for y in 30..35 {
            for x in 30..35 {
                output.put_pixel(x, y, Rgba([120, 20, 30, 255]));
            }
        }
        output.save(&output_path).unwrap();

        let generated_before =
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(64, 64, Rgba([10, 20, 30, 255])));
        let generated_after =
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(64, 64, Rgba([11, 20, 30, 255])));
        let before = visual_image_fingerprint(&generated_before);
        let after = visual_image_fingerprint(&generated_after);

        assert!(extracted_part_is_fresh(
            "mouth-a",
            &output_path,
            Some(&before),
            &before,
        ));
        assert!(!extracted_part_is_fresh(
            "mouth-a",
            &output_path,
            Some(&before),
            &after,
        ));
        assert!(!extracted_part_is_fresh(
            "mouth-a",
            &output_path,
            None,
            &before,
        ));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn transform_extracted_part_offsets_pixels_from_original_canvas() {
        let mut source = RgbaImage::new(6, 6);
        source.put_pixel(2, 2, image::Rgba([255, 0, 0, 255]));
        let adjusted =
            transform_extracted_part(&DynamicImage::ImageRgba8(source), 1, -1, 100).to_rgba8();

        assert_eq!(adjusted.get_pixel(3, 1).0, [255, 0, 0, 255]);
        assert_eq!(adjusted.get_pixel(2, 2)[3], 0);
    }

    #[test]
    fn transform_extracted_part_scale_100_preserves_pixels() {
        let mut source = RgbaImage::new(4, 4);
        source.put_pixel(1, 1, image::Rgba([10, 20, 30, 255]));
        source.put_pixel(2, 1, image::Rgba([200, 210, 220, 255]));
        source.put_pixel(1, 2, image::Rgba([40, 50, 60, 128]));

        let adjusted =
            transform_extracted_part(&DynamicImage::ImageRgba8(source.clone()), 0, 0, 100)
                .to_rgba8();

        assert_eq!(adjusted, source);
    }

    #[test]
    fn mouth_vowels_plan_closed_to_vowel_rife_pairs() {
        let temp_dir = std::env::temp_dir().join("pachipakugen_rife_jobs_test");
        let targets = vec![
            "mouth-closed".to_string(),
            "mouth-a".to_string(),
            "mouth-o".to_string(),
        ];

        let jobs = planned_expression_rife_jobs(&targets, &temp_dir);

        assert_eq!(jobs.len(), 2);
        assert!(jobs.iter().any(|job| job.name == "mouth-a"
            && job.start_key == "mouth-closed"
            && job.end_key == "mouth-a"));
        assert!(jobs.iter().any(|job| job.name == "mouth-o"
            && job.start_key == "mouth-closed"
            && job.end_key == "mouth-o"));
    }

    #[test]
    fn reference_guide_is_used_only_for_open_mouth_vowels() {
        assert!(target_uses_reference_guide("mouth-a"));
        assert!(target_uses_reference_guide("mouth-i"));
        assert!(target_uses_reference_guide("mouth-u"));
        assert!(target_uses_reference_guide("mouth-e"));
        assert!(target_uses_reference_guide("mouth-o"));
        assert!(!target_uses_reference_guide("mouth-closed"));
        assert!(!target_uses_reference_guide("eyes-closed"));
    }

    #[test]
    fn reference_feature_guide_keeps_only_local_region() {
        let reference =
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(100, 100, Rgba([10, 20, 30, 255])));
        let mut mask = GrayImage::new(100, 100);
        mask.put_pixel(50, 50, Luma([255]));

        let encoded = reference_feature_guide_png(&reference, &mask, 100, 100).unwrap();
        let decoded = image::load_from_memory(&encoded).unwrap().to_rgba8();

        assert_eq!(decoded.get_pixel(0, 0)[3], 0);
        assert_eq!(decoded.get_pixel(50, 50), &Rgba([10, 20, 30, 255]));
    }

    #[test]
    fn mouth_mask_stays_local_and_protects_nose() {
        let mut layers = HashMap::new();
        let mut mouth = RgbaImage::new(120, 120);
        for y in 70..=72 {
            for x in 52..=68 {
                mouth.put_pixel(x, y, Rgba([255, 0, 0, 255]));
            }
        }
        let mut nose = RgbaImage::new(120, 120);
        for y in 48..=58 {
            for x in 57..=63 {
                nose.put_pixel(x, y, Rgba([255, 255, 255, 255]));
            }
        }
        layers.insert("mouth".into(), DynamicImage::ImageRgba8(mouth));
        layers.insert("nose".into(), DynamicImage::ImageRgba8(nose));

        let mask = mouth_expression_mask(&layers, 120, 120);

        assert_eq!(mask.get_pixel(60, 53)[0], 0);
        assert!(mask.get_pixel(60, 70)[0] > 0);
        assert!(mask.get_pixel(60, 90)[0] > 0);
        assert_eq!(mask.get_pixel(60, 105)[0], 0);
    }

    #[test]
    fn mouth_mask_refinement_uses_only_changed_pixels() {
        let mut base = GrayImage::new(120, 120);
        for y in 45..=95 {
            for x in 35..=85 {
                base.put_pixel(x, y, Luma([255]));
            }
        }
        let source =
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(120, 120, Rgba([20, 20, 20, 255])));
        let mut generated = RgbaImage::from_pixel(120, 120, Rgba([20, 20, 20, 255]));
        for y in 65..=72 {
            for x in 55..=68 {
                generated.put_pixel(x, y, Rgba([220, 40, 40, 255]));
            }
        }

        let refined = refine_mouth_mask_with_difference(
            &base,
            &source,
            &DynamicImage::ImageRgba8(generated),
            120,
            120,
        );

        assert!(refined.get_pixel(62, 68)[0] > 0);
        assert_eq!(refined.get_pixel(40, 50)[0], 0);
        assert_eq!(refined.get_pixel(62, 94)[0], 0);
    }

    #[test]
    fn expression_extraction_uses_named_mouth_layer_only() {
        let mut layers = HashMap::new();
        let mut mouth = RgbaImage::new(120, 120);
        for y in 65..=72 {
            for x in 55..=68 {
                mouth.put_pixel(x, y, Rgba([220, 40, 40, 255]));
            }
        }
        let face = RgbaImage::from_pixel(120, 120, Rgba([200, 160, 140, 255]));
        layers.insert("face".into(), DynamicImage::ImageRgba8(face));
        layers.insert("mouth".into(), DynamicImage::ImageRgba8(mouth));

        let extracted = extract_named_expression_layers(&layers, &["mouth"], 120, 120)
            .unwrap()
            .to_rgba8();

        assert_eq!(extracted.get_pixel(10, 10)[3], 0);
        assert_eq!(extracted.get_pixel(62, 68).0, [220, 40, 40, 255]);
        assert_eq!(extracted.get_pixel(62, 90)[3], 0);
    }

    #[test]
    fn full_canvas_mouth_layer_falls_back_to_local_face_anchor() {
        let mut layers = HashMap::new();
        let mouth = RgbaImage::from_pixel(160, 160, Rgba([0, 0, 0, 255]));
        let mut nose = RgbaImage::new(160, 160);
        for y in 62..=70 {
            for x in 76..=84 {
                nose.put_pixel(x, y, Rgba([255, 255, 255, 255]));
            }
        }
        layers.insert("mouth".into(), DynamicImage::ImageRgba8(mouth));
        layers.insert("nose".into(), DynamicImage::ImageRgba8(nose));

        let mask = mouth_expression_mask(&layers, 160, 160);
        let editable_count = mask.pixels().filter(|pixel| pixel[0] > 0).count();

        assert!(editable_count < 160 * 80);
        assert_eq!(mask.get_pixel(80, 66)[0], 0);
        assert!(mask.get_pixel(80, 82)[0] > 0);
        assert_eq!(mask.get_pixel(10, 10)[0], 0);
    }

    #[test]
    fn anchored_mouth_mask_avoids_side_hair_and_choker_band() {
        let mut layers = HashMap::new();
        let mouth = RgbaImage::from_pixel(160, 160, Rgba([0, 0, 0, 255]));
        let mut nose = RgbaImage::new(160, 160);
        for y in 62..=70 {
            for x in 76..=84 {
                nose.put_pixel(x, y, Rgba([255, 255, 255, 255]));
            }
        }
        layers.insert("mouth".into(), DynamicImage::ImageRgba8(mouth));
        layers.insert("nose".into(), DynamicImage::ImageRgba8(nose));

        let mask = mouth_expression_mask(&layers, 160, 160);

        assert!(mask.get_pixel(80, 91)[0] > 0);
        assert_eq!(mask.get_pixel(50, 91)[0], 0);
        assert_eq!(mask.get_pixel(80, 130)[0], 0);
    }

    #[test]
    fn corrupted_bottom_nose_anchor_uses_eye_fallback() {
        let mut layers = HashMap::new();
        let mouth = RgbaImage::from_pixel(160, 160, Rgba([0, 0, 0, 255]));
        let mut nose = RgbaImage::new(160, 160);
        for y in 150..=159 {
            for x in 76..=84 {
                nose.put_pixel(x, y, Rgba([255, 255, 255, 255]));
            }
        }
        let mut eyes = RgbaImage::new(160, 160);
        for y in 45..=68 {
            for x in 50..=110 {
                eyes.put_pixel(x, y, Rgba([255, 255, 255, 255]));
            }
        }
        layers.insert("mouth".into(), DynamicImage::ImageRgba8(mouth));
        layers.insert("nose".into(), DynamicImage::ImageRgba8(nose));
        layers.insert("eyewhite".into(), DynamicImage::ImageRgba8(eyes));

        let mask = mouth_expression_mask(&layers, 160, 160);

        assert!(mask.get_pixel(80, 76)[0] > 0);
        assert_eq!(mask.get_pixel(80, 158)[0], 0);
        assert!(mask_has_minimum_edit_area(&mask, 600));
    }

    #[test]
    fn out_of_bounds_rect_mask_does_not_create_bottom_row() {
        let mask = rect_mask(10, 10, 2, 12, 6, 18);

        assert!(mask.iter().all(|value| *value == 0));
    }

    #[test]
    fn selected_mask_validation_rejects_tiny_sliver() {
        let mut tiny = GrayImage::new(1280, 1280);
        for x in 520..=756 {
            tiny.put_pixel(x, 1279, Luma([255]));
        }

        let error =
            validate_selected_masks(&["mouth-a".into()], &tiny, &GrayImage::new(1280, 1280))
                .unwrap_err();

        assert!(error.to_string().contains("口マスクを取得できませんでした"));
    }

    #[test]
    fn empty_selected_mask_is_rejected_before_api_use() {
        let empty = GrayImage::new(2, 2);
        let error = validate_selected_masks(&["mouth-a".into()], &empty, &empty).unwrap_err();
        assert!(error.to_string().contains("口マスクを取得できませんでした"));
    }

    #[test]
    fn local_composite_preserves_every_pixel_outside_mask() {
        let source = DynamicImage::ImageRgba8(RgbaImage::from_pixel(2, 1, Rgba([10, 20, 30, 255])));
        let candidate =
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(2, 1, Rgba([200, 210, 220, 255])));
        let mut mask = GrayImage::new(2, 1);
        mask.put_pixel(1, 0, Luma([255]));

        let result = composite_inside_mask(&source, &candidate, &mask, 2, 1).to_rgba8();
        assert_eq!(result.get_pixel(0, 0), &Rgba([10, 20, 30, 255]));
        assert_eq!(result.get_pixel(1, 0), &Rgba([200, 210, 220, 255]));
    }

    #[test]
    fn openai_mask_marks_edit_region_transparent() {
        let mut mask = GrayImage::new(2, 1);
        mask.put_pixel(1, 0, Luma([255]));
        let encoded = openai_mask_bytes(&mask).unwrap();
        let decoded = image::load_from_memory(&encoded).unwrap().to_rgba8();
        assert_eq!(decoded.get_pixel(0, 0)[3], 255);
        assert_eq!(decoded.get_pixel(1, 0)[3], 0);
    }

    #[test]
    fn eyebrow_mask_removes_only_baked_brow_alpha() {
        let mut source = RgbaImage::from_pixel(2, 1, Rgba([20, 30, 40, 255]));
        source.put_pixel(1, 0, Rgba([20, 30, 40, 128]));
        let image = DynamicImage::ImageRgba8(source);
        let mut mask = RgbaImage::new(2, 1);
        mask.put_pixel(1, 0, Rgba([10, 10, 10, 128]));

        let result = erase_alpha_with_mask(&image, &DynamicImage::ImageRgba8(mask)).to_rgba8();

        assert_eq!(result.get_pixel(0, 0)[3], 255);
        assert_eq!(result.get_pixel(1, 0)[3], 0);
    }

    #[test]
    fn syncing_independent_eyebrow_sanitizes_legacy_eye_frames() {
        let root = std::env::temp_dir().join(format!(
            "pachipakugen_spritalk_brow_cleanup_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        let eye_dir = root.join("eye");
        fs::create_dir_all(&eye_dir).unwrap();

        let mut eyebrow = RgbaImage::new(256, 2);
        eyebrow.put_pixel(100, 0, Rgba([35, 20, 15, 160]));
        let eyebrow = DynamicImage::ImageRgba8(eyebrow);
        let mut legacy_eye = RgbaImage::new(256, 2);
        // The old antialiased edge sits one pixel outside the source mask.
        legacy_eye.put_pixel(101, 0, Rgba([35, 20, 15, 64]));
        legacy_eye.put_pixel(110, 0, Rgba([0, 200, 255, 255]));
        legacy_eye.save(eye_dir.join("001.png")).unwrap();

        sanitize_legacy_eye_frames(&root, &eyebrow).unwrap();

        let cleaned = image::open(eye_dir.join("001.png")).unwrap().to_rgba8();
        assert_eq!(cleaned.get_pixel(101, 0)[3], 0);
        assert_eq!(cleaned.get_pixel(110, 0)[3], 255);
        let _ = fs::remove_dir_all(root);
    }
}
