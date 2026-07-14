use crate::commands::parts::{
    cache_original_image_for_canvas, get_mapping_preview_inner, load_slot_inner,
    MappingPreviewResult, SlotLoadResult,
};
use crate::error::AppError;
use crate::processing::image_utils;
use crate::state::AppState;
use keyring_core::Entry;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter, Manager};

const SEE_THROUGH_REPO: &str = "https://github.com/shitagaki-lab/see-through.git";
const SEE_THROUGH_COMMIT: &str = "e4cb250dc69defe6f982168dab684aa461552b5b";
const KEYRING_SERVICE: &str = "com.kazuya.pachipakugen";
const HF_TOKEN_CREDENTIAL: &str = "huggingface-token";
const MODEL_MANIFEST_SCHEMA_VERSION: u32 = 1;
const MODEL_PREFETCH_SCRIPT: &str = include_str!("../../scripts/prepare_see_through_models.py");
const MODEL_REQUIREMENTS_JSON: &str =
    include_str!("../../scripts/see_through_model_requirements.json");
const MODEL_DOWNLOAD_WRAPPER: &str = include_str!("../../scripts/download_see_through_models.ps1");

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HfTokenStatus {
    pub configured: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SeeThroughModelDownloadLaunch {
    pub started: bool,
    pub message: String,
}

fn hf_token_entry() -> Result<Entry, AppError> {
    Entry::new(KEYRING_SERVICE, HF_TOKEN_CREDENTIAL)
        .map_err(|error| AppError::General(format!("Windows資格情報を開けませんでした: {error}")))
}

fn stored_hf_token() -> Option<String> {
    hf_token_entry()
        .ok()?
        .get_password()
        .ok()
        .filter(|value| !value.trim().is_empty())
}

#[tauri::command]
pub fn get_hf_token_status() -> HfTokenStatus {
    HfTokenStatus {
        configured: stored_hf_token().is_some(),
    }
}

/// HuggingFaceのアクセストークンをWindows資格情報に安全に保存する。
/// See-Throughのモデルダウンロードが匿名アクセスのレート制限を受けなくなる
#[tauri::command]
pub fn save_hf_token(token: String) -> Result<HfTokenStatus, AppError> {
    let trimmed = token.trim();
    if trimmed.is_empty() {
        return Err(AppError::General("トークンを入力してください".into()));
    }
    hf_token_entry()?.set_password(trimmed).map_err(|error| {
        AppError::General(format!("トークンを安全に保存できませんでした: {error}"))
    })?;
    Ok(HfTokenStatus { configured: true })
}

#[tauri::command]
pub fn delete_hf_token() -> Result<HfTokenStatus, AppError> {
    let entry = hf_token_entry()?;
    if let Err(error) = entry.delete_credential() {
        if !matches!(error, keyring_core::Error::NoEntry) {
            return Err(AppError::General(format!(
                "保存済みトークンを削除できませんでした: {error}"
            )));
        }
    }
    Ok(HfTokenStatus { configured: false })
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SeeThroughProgressPayload {
    pub stage: String,
    pub percent: u32,
    pub message: String,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SeeThroughRuntimeStatus {
    pub ready: bool,
    pub runtime_ready: bool,
    pub busy: bool,
    pub model_download_busy: bool,
    pub runtime_root: String,
    pub repo_path: String,
    pub python_path: String,
    pub pinned_commit: String,
    pub installed_commit: Option<String>,
    pub gpu_index: Option<u32>,
    pub gpu_name: Option<String>,
    pub gpu_memory_mb: Option<u32>,
    pub recommended_profile: String,
    pub selected_profile: String,
    pub message: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SeeThroughModelManifest {
    schema_version: u32,
    profile: String,
    repositories: Vec<String>,
    revisions: HashMap<String, String>,
    files: Vec<SeeThroughModelManifestFile>,
}

#[derive(Debug, Deserialize)]
struct SeeThroughModelManifestFile {
    path: String,
    size: u64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SeeThroughModelRequirements {
    schema_version: u32,
    profiles: HashMap<String, Vec<SeeThroughModelRepositoryRequirement>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SeeThroughModelRepositoryRequirement {
    repo_id: String,
    revision: String,
    files: Vec<SeeThroughRequiredModelFile>,
}

#[derive(Debug, Deserialize)]
struct SeeThroughRequiredModelFile {
    path: String,
    size: u64,
}

#[derive(Debug, PartialEq, Eq)]
enum ModelReadiness {
    Ready,
    Missing,
    Incomplete(String),
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SeeThroughRunResult {
    pub psd_path: String,
    pub output_dir: String,
    pub selected_profile: String,
    /// 自動回復を含め、実際に成功した推論設定。後続バッチへそのまま引き継ぐ。
    pub effective_options: Option<SeeThroughOptions>,
    pub slot_load: SlotLoadResult,
    pub mapping_preview: MappingPreviewResult,
    /// 左右パーツ分解に失敗し、左右分解なしで自動リトライした場合にtrue（UIで報告する）
    pub split_parts_fallback: bool,
    /// GPU/ネイティブ推論エラーで自動リトライした場合、その内容の説明文（UIで報告する）
    pub oom_retry_note: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SeeThroughOptions {
    pub seed: Option<i64>,
    pub resolution: Option<i64>,
    pub resolution_depth: Option<i64>,
    pub inference_steps: Option<i64>,
    pub inference_steps_depth: Option<i64>,
    pub group_offload: Option<String>,
    pub cpu_offload: Option<String>,
}

#[derive(Clone)]
struct GpuInfo {
    index: u32,
    uuid: String,
    name: String,
    memory_mb: u32,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SeeThroughGpuInfo {
    pub index: u32,
    pub name: String,
    pub memory_mb: u32,
}

#[tauri::command]
pub async fn list_see_through_gpus() -> Result<Vec<SeeThroughGpuInfo>, AppError> {
    tauri::async_runtime::spawn_blocking(|| {
        Ok(detect_gpus()
            .into_iter()
            .map(|gpu| SeeThroughGpuInfo {
                index: gpu.index,
                name: gpu.name,
                memory_mb: gpu.memory_mb,
            })
            .collect())
    })
    .await
    .map_err(|error| AppError::General(format!("GPU一覧の取得に失敗: {error}")))?
}

#[tauri::command]
pub fn set_see_through_gpu(app: AppHandle, gpu_index: Option<u32>) -> Result<(), AppError> {
    *app.state::<AppState>()
        .see_through_gpu_index
        .lock()
        .unwrap() = gpu_index;
    Ok(())
}

#[tauri::command]
pub async fn load_expression_source_preview(path: String) -> Result<String, AppError> {
    tauri::async_runtime::spawn_blocking(move || {
        let image = image::open(path)?;
        Ok(image_utils::image_to_base64_png(&image))
    })
    .await
    .map_err(|error| AppError::General(format!("元画像プレビューの読み込みに失敗: {error}")))?
}

#[tauri::command]
pub async fn get_see_through_runtime_status(
    app: AppHandle,
    profile: Option<String>,
) -> Result<SeeThroughRuntimeStatus, AppError> {
    tauri::async_runtime::spawn_blocking(move || {
        runtime_status(&app, profile.as_deref().unwrap_or("auto"))
    })
    .await
    .map_err(|error| AppError::General(format!("See-Through状態確認に失敗: {error}")))?
}

#[tauri::command]
pub async fn prepare_see_through_runtime(
    app: AppHandle,
    profile: Option<String>,
) -> Result<SeeThroughRuntimeStatus, AppError> {
    tauri::async_runtime::spawn_blocking(move || {
        prepare_runtime(&app, profile.as_deref().unwrap_or("auto"))
    })
    .await
    .map_err(|error| AppError::General(format!("See-Throughセットアップ処理に失敗: {error}")))?
}

#[tauri::command]
pub fn start_see_through_model_download(
    app: AppHandle,
    profile: Option<String>,
) -> Result<SeeThroughModelDownloadLaunch, AppError> {
    #[cfg(not(target_os = "windows"))]
    {
        let _ = (app, profile);
        return Err(AppError::General(
            "モデルの事前ダウンロード用コンソールはWindowsのみ対応しています".into(),
        ));
    }

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;

        const CREATE_NEW_CONSOLE: u32 = 0x0000_0010;

        let state = app.state::<AppState>();
        let _runtime_guard = state.see_through_runtime_lock.try_lock().map_err(|_| {
            AppError::General(
                "See-Throughのセットアップまたは推論を実行中です。完了後にモデルを取得してください"
                    .into(),
            )
        })?;
        let root = runtime_root(&app)?;
        let python = venv_python(&root);
        let selected_profile = model_profile_key(&select_profile(
            profile.as_deref().unwrap_or("auto"),
            resolve_gpu(&app).as_ref(),
        ))
        .to_string();
        let status = runtime_status(&app, &selected_profile)?;
        if !status.runtime_ready {
            return Err(AppError::General(
                "先にSee-Throughランタイムの初回セットアップを完了してください".into(),
            ));
        }
        if status.ready {
            return Ok(SeeThroughModelDownloadLaunch {
                started: false,
                message: format!(
                    "{}モデルはすでに検証済みです",
                    profile_label(&selected_profile)
                ),
            });
        }

        let (downloader_path, requirements_path, wrapper_path) =
            materialize_model_download_assets(&root)?;
        let manifest_path = model_manifest_path(&root, &selected_profile);
        if let Some(parent) = manifest_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let hf_home = root.join("huggingface");
        let hf_hub_cache = hf_home.join("hub");
        fs::create_dir_all(&hf_hub_cache)?;

        let mut active_pid = state.see_through_model_download_pid.lock().unwrap();
        if active_pid.is_some() {
            return Err(AppError::General(
                "モデル事前ダウンロード用コンソールはすでに起動しています".into(),
            ));
        }

        let mut command = Command::new("powershell.exe");
        command
            .args([
                "-NoLogo",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
            ])
            .arg(&wrapper_path)
            .arg("-Python")
            .arg(&python)
            .arg("-Downloader")
            .arg(&downloader_path)
            .arg("-Profile")
            .arg(&selected_profile)
            .arg("-Requirements")
            .arg(&requirements_path)
            .arg("-Manifest")
            .arg(&manifest_path)
            .current_dir(&root)
            .env("HF_HOME", &hf_home)
            .env("HF_HUB_CACHE", &hf_hub_cache)
            .env("HUGGINGFACE_HUB_CACHE", &hf_hub_cache)
            .env("TRANSFORMERS_CACHE", &hf_hub_cache)
            .env("DIFFUSERS_CACHE", &hf_hub_cache)
            .env("HF_HUB_DISABLE_XET", "1")
            .env("HF_HUB_DISABLE_PROGRESS_BARS", "0")
            .env("PYTHONIOENCODING", "utf-8")
            .env("PYTHONUNBUFFERED", "1")
            .creation_flags(CREATE_NEW_CONSOLE);
        if let Some(token) = stored_hf_token() {
            command.env("HF_TOKEN", &token);
            command.env("HUGGING_FACE_HUB_TOKEN", &token);
        }

        let mut child = command.spawn().map_err(|error| {
            AppError::General(format!(
                "モデル事前ダウンロード用コンソールを起動できません: {error}"
            ))
        })?;
        let pid = child.id();
        *active_pid = Some(pid);
        drop(active_pid);

        emit_progress(
            &app,
            "model-download-external",
            0,
            &format!(
                "{}モデルの事前ダウンロードを別コンソールで開始しました",
                profile_label(&selected_profile)
            ),
        );

        let wait_app = app.clone();
        let wait_profile = selected_profile.clone();
        std::thread::spawn(move || {
            let exit_status = child.wait();
            *wait_app
                .state::<AppState>()
                .see_through_model_download_pid
                .lock()
                .unwrap() = None;

            match exit_status {
                Ok(status) if status.success() => match runtime_status(&wait_app, &wait_profile) {
                    Ok(runtime) if runtime.ready => emit_progress(
                        &wait_app,
                        "model-download-complete",
                        100,
                        &format!(
                            "{}モデルの事前ダウンロードと完全性検証が完了しました",
                            profile_label(&wait_profile)
                        ),
                    ),
                    Ok(runtime) => emit_progress(
                        &wait_app,
                        "model-download-failed",
                        0,
                        &format!("モデル取得後の完全性検証に失敗しました: {}", runtime.message),
                    ),
                    Err(error) => emit_progress(
                        &wait_app,
                        "model-download-failed",
                        0,
                        &format!("モデル取得後の状態確認に失敗しました: {error}"),
                    ),
                },
                Ok(status) => emit_progress(
                    &wait_app,
                    "model-download-failed",
                    0,
                    &format!(
                        "モデル事前ダウンロードが完了しませんでした（終了コード: {}）。再実行すると続きから再開します",
                        status.code().unwrap_or(-1)
                    ),
                ),
                Err(error) => emit_progress(
                    &wait_app,
                    "model-download-failed",
                    0,
                    &format!("モデル事前ダウンロードの終了状態を確認できません: {error}"),
                ),
            }
        });

        Ok(SeeThroughModelDownloadLaunch {
            started: true,
            message: format!(
                "{}モデルの事前ダウンロード用コンソールを開きました",
                profile_label(&selected_profile)
            ),
        })
    }
}

#[tauri::command]
pub async fn run_see_through(
    app: AppHandle,
    source_path: String,
    profile: String,
    split_parts: bool,
    options: Option<SeeThroughOptions>,
) -> Result<SeeThroughRunResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || {
        run_inference(&app, &source_path, &profile, split_parts, options)
    })
    .await
    .map_err(|error| AppError::General(format!("See-Through実行処理に失敗: {error}")))?
}

#[tauri::command]
pub async fn cancel_see_through(app: AppHandle) -> Result<bool, AppError> {
    let state = app.state::<AppState>();
    let pid = *state.see_through_pid.lock().unwrap();
    let Some(pid) = pid else {
        return Ok(false);
    };

    #[cfg(target_os = "windows")]
    let status = Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .status()?;

    #[cfg(not(target_os = "windows"))]
    let status = Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .status()?;

    if status.success() {
        *state.see_through_pid.lock().unwrap() = None;
        emit_progress(&app, "cancelled", 0, "See-Through処理をキャンセルしました");
        Ok(true)
    } else {
        Err(AppError::General(format!(
            "See-Throughプロセスを停止できませんでした: PID {pid}"
        )))
    }
}

fn runtime_status(
    app: &AppHandle,
    requested_profile: &str,
) -> Result<SeeThroughRuntimeStatus, AppError> {
    let root = runtime_root(app)?;
    let repo = root.join("repo");
    let python = venv_python(&root);
    let marker = root.join("setup-complete.json");
    let installed_commit = git_commit(&repo);
    let gpu = resolve_gpu(app);
    let recommended_profile = select_profile("auto", gpu.as_ref());
    let selected_profile = select_profile(requested_profile, gpu.as_ref());
    let busy = app
        .state::<AppState>()
        .see_through_pid
        .lock()
        .unwrap()
        .is_some();
    let model_download_busy = app
        .state::<AppState>()
        .see_through_model_download_pid
        .lock()
        .unwrap()
        .is_some();
    let runtime_ready = marker.is_file()
        && repo.join("inference/scripts/inference_psd.py").is_file()
        && repo
            .join("inference/scripts/inference_psd_quantized.py")
            .is_file()
        && python.is_file()
        && installed_commit.as_deref() == Some(SEE_THROUGH_COMMIT);
    let model_readiness = if runtime_ready {
        validate_model_manifest(&root, &selected_profile)
    } else {
        ModelReadiness::Missing
    };
    let ready = runtime_ready && model_readiness == ModelReadiness::Ready;

    let message = if ready {
        format!(
            "See-Throughランタイムと{}モデルは使用できます",
            profile_label(&selected_profile)
        )
    } else if runtime_ready && model_download_busy {
        format!(
            "{}モデルを別コンソールで事前ダウンロードしています。完了後に環境を再確認してください",
            profile_label(&selected_profile)
        )
    } else if runtime_ready {
        match model_readiness {
            ModelReadiness::Ready => unreachable!("ready must include a ready model manifest"),
            ModelReadiness::Missing => format!(
                "{}モデルが未取得です。モデルの事前ダウンロードを別コンソールで実行してください",
                profile_label(&selected_profile)
            ),
            ModelReadiness::Incomplete(reason) => format!(
                "{}モデルの事前ダウンロードが未完了です: {reason}。もう一度起動すると続きから再開します",
                profile_label(&selected_profile)
            ),
        }
    } else if repo.is_dir() || python.is_file() {
        "See-Throughランタイムのセットアップが未完了です".to_string()
    } else {
        "ランタイムの初回セットアップが必要です。モデルはその後、別コンソールで事前ダウンロードします".to_string()
    };

    Ok(SeeThroughRuntimeStatus {
        ready,
        runtime_ready,
        busy,
        model_download_busy,
        runtime_root: root.to_string_lossy().into_owned(),
        repo_path: repo.to_string_lossy().into_owned(),
        python_path: python.to_string_lossy().into_owned(),
        pinned_commit: SEE_THROUGH_COMMIT.to_string(),
        installed_commit,
        gpu_index: gpu.as_ref().map(|gpu| gpu.index),
        gpu_name: gpu.as_ref().map(|gpu| gpu.name.clone()),
        gpu_memory_mb: gpu.as_ref().map(|gpu| gpu.memory_mb),
        recommended_profile,
        selected_profile,
        message,
    })
}

fn model_profile_key(profile: &str) -> &'static str {
    if profile == "low-vram" {
        "low-vram"
    } else {
        "standard"
    }
}

fn profile_label(profile: &str) -> &'static str {
    if model_profile_key(profile) == "low-vram" {
        "省VRAM"
    } else {
        "高VRAM"
    }
}

fn model_repository_requirements(
    profile: &str,
) -> Result<Vec<SeeThroughModelRepositoryRequirement>, String> {
    let mut requirements: SeeThroughModelRequirements =
        serde_json::from_str(MODEL_REQUIREMENTS_JSON)
            .map_err(|error| format!("モデル要件定義を読み込めません: {error}"))?;
    if requirements.schema_version != MODEL_MANIFEST_SCHEMA_VERSION {
        return Err("モデル要件定義の形式が未対応です".into());
    }
    requirements
        .profiles
        .remove(model_profile_key(profile))
        .filter(|repositories| !repositories.is_empty())
        .ok_or_else(|| "選択プロファイルのモデル要件がありません".into())
}

fn model_manifest_path(root: &Path, profile: &str) -> PathBuf {
    root.join("model-manifests")
        .join(format!("{}.json", model_profile_key(profile)))
}

fn model_cache_directory_name(repo_id: &str) -> String {
    format!("models--{}", repo_id.replace('/', "--"))
}

fn model_snapshot_manifest_path(
    repository: &SeeThroughModelRepositoryRequirement,
    relative_file: &str,
) -> String {
    format!(
        "hub/{}/snapshots/{}/{}",
        model_cache_directory_name(&repository.repo_id),
        repository.revision,
        relative_file.replace('\\', "/")
    )
}

fn model_ref_manifest_path(repository: &SeeThroughModelRepositoryRequirement) -> String {
    format!(
        "hub/{}/refs/main",
        model_cache_directory_name(&repository.repo_id)
    )
}

fn expected_model_file_map(
    repositories: &[SeeThroughModelRepositoryRequirement],
) -> Result<HashMap<String, u64>, String> {
    let mut expected = HashMap::new();
    let mut repo_ids = HashMap::new();
    for repository in repositories {
        if repository.repo_id.trim().is_empty()
            || repository.revision.len() != 40
            || repository.files.is_empty()
            || repo_ids
                .insert(repository.repo_id.clone(), repository.revision.clone())
                .is_some()
        {
            return Err("モデル要件定義に不正または重複したリポジトリがあります".into());
        }
        for file in &repository.files {
            let relative = Path::new(&file.path);
            let unsafe_path = relative.as_os_str().is_empty()
                || relative.components().any(|component| {
                    matches!(
                        component,
                        std::path::Component::ParentDir
                            | std::path::Component::RootDir
                            | std::path::Component::Prefix(_)
                    )
                });
            let manifest_path = model_snapshot_manifest_path(repository, &file.path);
            if unsafe_path || file.size == 0 || expected.insert(manifest_path, file.size).is_some()
            {
                return Err(format!(
                    "モデル要件定義に不正または重複したファイルがあります: {}/{}",
                    repository.repo_id, file.path
                ));
            }
        }
        let ref_path = model_ref_manifest_path(repository);
        if expected
            .insert(ref_path, repository.revision.len() as u64)
            .is_some()
        {
            return Err(format!(
                "モデル要件定義のrevision参照が重複しています: {}",
                repository.repo_id
            ));
        }
    }
    Ok(expected)
}

fn profile_has_incomplete_download(root: &Path, profile: &str) -> bool {
    model_repository_requirements(profile)
        .ok()
        .is_some_and(|repositories| {
            repositories.iter().any(|repository| {
                let blobs = root
                    .join("huggingface/hub")
                    .join(model_cache_directory_name(&repository.repo_id))
                    .join("blobs");
                fs::read_dir(blobs).ok().is_some_and(|entries| {
                    entries.filter_map(Result::ok).any(|entry| {
                        entry.path().extension().and_then(|value| value.to_str())
                            == Some("incomplete")
                    })
                })
            })
        })
}

fn invalid_model_manifest(root: &Path, profile: &str, reason: String) -> ModelReadiness {
    if profile_has_incomplete_download(root, profile) {
        ModelReadiness::Incomplete(
            "モデルのダウンロード途中ファイル（.incomplete）が残っています".into(),
        )
    } else {
        ModelReadiness::Incomplete(reason)
    }
}

fn validate_model_manifest(root: &Path, profile: &str) -> ModelReadiness {
    let profile_key = model_profile_key(profile);
    let repositories = match model_repository_requirements(profile_key) {
        Ok(repositories) => repositories,
        Err(reason) => return ModelReadiness::Incomplete(reason),
    };
    validate_model_manifest_against(root, profile_key, &repositories)
}

fn validate_model_manifest_against(
    root: &Path,
    profile: &str,
    repositories: &[SeeThroughModelRepositoryRequirement],
) -> ModelReadiness {
    let profile_key = model_profile_key(profile);
    let expected_files = match expected_model_file_map(repositories) {
        Ok(files) => files,
        Err(reason) => return ModelReadiness::Incomplete(reason),
    };
    let manifest_path = model_manifest_path(root, profile_key);
    let bytes = match fs::read(&manifest_path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return if profile_has_incomplete_download(root, profile_key) {
                ModelReadiness::Incomplete(
                    "モデルのダウンロード途中ファイル（.incomplete）が残っています".into(),
                )
            } else {
                ModelReadiness::Missing
            };
        }
        Err(error) => {
            return ModelReadiness::Incomplete(format!("モデル検証情報を読み込めません: {error}"));
        }
    };
    let manifest: SeeThroughModelManifest = match serde_json::from_slice(&bytes) {
        Ok(manifest) => manifest,
        Err(error) => {
            return invalid_model_manifest(
                root,
                profile_key,
                format!("モデル検証情報が壊れています: {error}"),
            );
        }
    };
    if manifest.schema_version != MODEL_MANIFEST_SCHEMA_VERSION {
        return invalid_model_manifest(
            root,
            profile_key,
            "モデル検証情報の形式が古いか未対応です".into(),
        );
    }
    if manifest.profile != profile_key {
        return invalid_model_manifest(
            root,
            profile_key,
            "別の実行プロファイル用モデルが記録されています".into(),
        );
    }

    let mut expected_ids = repositories
        .iter()
        .map(|repository| repository.repo_id.clone())
        .collect::<Vec<_>>();
    let mut actual_ids = manifest.repositories.clone();
    expected_ids.sort();
    actual_ids.sort();
    if actual_ids != expected_ids {
        return invalid_model_manifest(
            root,
            profile_key,
            "必要なモデルリポジトリの記録が揃っていません".into(),
        );
    }
    if manifest.revisions.len() != repositories.len() {
        return invalid_model_manifest(
            root,
            profile_key,
            "モデルの固定リビジョン記録数が一致しません".into(),
        );
    }
    for repository in repositories {
        if manifest
            .revisions
            .get(&repository.repo_id)
            .map(String::as_str)
            != Some(repository.revision.as_str())
        {
            return invalid_model_manifest(
                root,
                profile_key,
                format!(
                    "モデルの固定リビジョンが一致しません: {}",
                    repository.repo_id
                ),
            );
        }
    }

    let mut actual_files = HashMap::new();
    for file in &manifest.files {
        if actual_files.insert(file.path.clone(), file.size).is_some() {
            return invalid_model_manifest(
                root,
                profile_key,
                format!("モデル検証情報に重複ファイルがあります: {}", file.path),
            );
        }
    }
    if actual_files != expected_files {
        if let Some((path, size)) = expected_files
            .iter()
            .find(|(path, size)| actual_files.get(*path) != Some(*size))
        {
            return invalid_model_manifest(
                root,
                profile_key,
                format!("必須モデルファイルの記録が一致しません: {path} ({size} bytes)"),
            );
        }
        return invalid_model_manifest(
            root,
            profile_key,
            "モデル検証情報に不要なファイルが含まれています".into(),
        );
    }

    let hf_home = root.join("huggingface");
    for (path, expected_size) in &expected_files {
        let full_path = hf_home.join(path);
        let metadata = match fs::metadata(&full_path) {
            Ok(metadata) if metadata.is_file() => metadata,
            _ => {
                return invalid_model_manifest(
                    root,
                    profile_key,
                    format!("モデルファイルが見つかりません: {path}"),
                );
            }
        };
        if metadata.len() != *expected_size {
            return invalid_model_manifest(
                root,
                profile_key,
                format!(
                    "モデルファイルのサイズが一致しません: {}（期待: {} / 実際: {} bytes）",
                    path,
                    expected_size,
                    metadata.len()
                ),
            );
        }
    }
    for repository in repositories {
        let ref_path = hf_home.join(model_ref_manifest_path(repository));
        let actual_revision = match fs::read_to_string(&ref_path) {
            Ok(revision) => revision,
            Err(error) => {
                return invalid_model_manifest(
                    root,
                    profile_key,
                    format!(
                        "モデルrevision参照を読めません: {} ({error})",
                        ref_path.display()
                    ),
                );
            }
        };
        if actual_revision.trim() != repository.revision {
            return invalid_model_manifest(
                root,
                profile_key,
                format!(
                    "モデルrevision参照が固定値と一致しません: {}",
                    repository.repo_id
                ),
            );
        }
    }
    ModelReadiness::Ready
}

fn materialize_model_download_assets(root: &Path) -> Result<(PathBuf, PathBuf, PathBuf), AppError> {
    let script_path = root.join("scripts/prepare_see_through_models.py");
    let requirements_path = root.join("scripts/see_through_model_requirements.json");
    let wrapper_path = root.join("scripts/download_see_through_models.ps1");
    if let Some(parent) = script_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&script_path, MODEL_PREFETCH_SCRIPT).map_err(|error| {
        AppError::General(format!(
            "モデルセットアップスクリプトを準備できません: {error}"
        ))
    })?;
    fs::write(&requirements_path, MODEL_REQUIREMENTS_JSON)
        .map_err(|error| AppError::General(format!("モデル要件定義を準備できません: {error}")))?;
    fs::write(&wrapper_path, MODEL_DOWNLOAD_WRAPPER).map_err(|error| {
        AppError::General(format!(
            "モデル事前ダウンロード用スクリプトを準備できません: {error}"
        ))
    })?;
    Ok((script_path, requirements_path, wrapper_path))
}

fn prepare_runtime(
    app: &AppHandle,
    requested_profile: &str,
) -> Result<SeeThroughRuntimeStatus, AppError> {
    let state = app.state::<AppState>();
    let _runtime_guard = state.see_through_runtime_lock.try_lock().map_err(|_| {
        AppError::General(
            "See-Throughのセットアップまたは推論を実行中です。完了後に再実行してください".into(),
        )
    })?;
    if state
        .see_through_model_download_pid
        .lock()
        .unwrap()
        .is_some()
    {
        return Err(AppError::General(
            "モデル事前ダウンロード中はランタイムを更新できません。完了後に再実行してください"
                .into(),
        ));
    }
    let root = runtime_root(app)?;
    let repo = root.join("repo");
    let python = venv_python(&root);
    let gpu = resolve_gpu(app);
    let selected_profile = select_profile(requested_profile, gpu.as_ref());
    fs::create_dir_all(&root)?;

    emit_progress(
        app,
        "prepare",
        3,
        "See-Through専用ランタイムを準備しています",
    );
    let git = find_executable("PACHIPAKUGEN_GIT", &["git"])?;
    let uv = find_executable("PACHIPAKUGEN_UV", &["uv"])?;

    if !repo.join(".git").is_dir() {
        emit_progress(
            app,
            "prepare",
            8,
            "See-Through公式リポジトリを取得しています",
        );
        run_managed_command(
            app,
            &git,
            vec![
                "clone".into(),
                "--filter=blob:none".into(),
                "--no-checkout".into(),
                SEE_THROUGH_REPO.into(),
                repo.to_string_lossy().into_owned(),
            ],
            &root,
            "prepare",
            None,
            &root.join("huggingface"),
        )?;
    }

    emit_progress(
        app,
        "prepare",
        18,
        "検証済みのSee-Throughコミットへ固定しています",
    );
    run_managed_command(
        app,
        &git,
        vec![
            "fetch".into(),
            "origin".into(),
            SEE_THROUGH_COMMIT.into(),
            "--depth".into(),
            "1".into(),
        ],
        &repo,
        "prepare",
        None,
        &root.join("huggingface"),
    )?;
    run_managed_command(
        app,
        &git,
        vec![
            "checkout".into(),
            "--force".into(),
            SEE_THROUGH_COMMIT.into(),
        ],
        &repo,
        "prepare",
        None,
        &root.join("huggingface"),
    )?;

    let assets = repo.join("assets");
    if !assets.is_dir() {
        if assets.is_file() {
            fs::remove_file(&assets)?;
        }
        copy_dir_recursive(&repo.join("common/assets"), &assets)?;
    }
    apply_runtime_compatibility_patches(&repo, selected_profile != "low-vram")?;

    if !python.is_file() {
        emit_progress(app, "prepare", 25, "Python 3.12専用環境を作成しています");
        run_managed_command(
            app,
            &uv,
            vec![
                "venv".into(),
                root.join(".venv").to_string_lossy().into_owned(),
                "--python".into(),
                "3.12".into(),
            ],
            &root,
            "prepare",
            None,
            &root.join("huggingface"),
        )?;
    }

    emit_progress(app, "prepare", 35, "CUDA版PyTorchをインストールしています");
    run_managed_command(
        app,
        &uv,
        vec![
            "pip".into(),
            "install".into(),
            "--python".into(),
            python.to_string_lossy().into_owned(),
            "torch==2.8.0+cu128".into(),
            "torchvision==0.23.0+cu128".into(),
            "torchaudio==2.8.0+cu128".into(),
            "--index-url".into(),
            "https://download.pytorch.org/whl/cu128".into(),
        ],
        &root,
        "prepare",
        None,
        &root.join("huggingface"),
    )?;

    emit_progress(
        app,
        "prepare",
        55,
        "See-Through依存パッケージをインストールしています",
    );
    run_managed_command(
        app,
        &uv,
        vec![
            "pip".into(),
            "install".into(),
            "--python".into(),
            python.to_string_lossy().into_owned(),
            "-r".into(),
            repo.join("requirements.txt").to_string_lossy().into_owned(),
        ],
        &repo,
        "prepare",
        None,
        &root.join("huggingface"),
    )?;

    emit_progress(
        app,
        "prepare",
        80,
        "省VRAM向け依存パッケージを準備しています",
    );
    run_managed_command(
        app,
        &uv,
        vec![
            "pip".into(),
            "install".into(),
            "--python".into(),
            python.to_string_lossy().into_owned(),
            "-r".into(),
            repo.join("requirements-inference-bnb.txt")
                .to_string_lossy()
                .into_owned(),
        ],
        &repo,
        "prepare",
        None,
        &root.join("huggingface"),
    )?;

    emit_progress(app, "prepare", 92, "CUDA実行環境を確認しています");
    run_managed_command(
        app,
        &python.to_string_lossy(),
        vec![
            "-c".into(),
            "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
                .into(),
        ],
        &repo,
        "prepare",
        gpu.as_ref(),
        &root.join("huggingface"),
    )?;

    // 大容量モデルはアプリ内セットアップでは取得しない。ここでは外部コンソールで
    // ユーザーが事前ダウンロードするための固定スクリプトだけを準備する。
    materialize_model_download_assets(&root)?;

    let marker = serde_json::json!({
        "repository": SEE_THROUGH_REPO,
        "commit": SEE_THROUGH_COMMIT,
        "preparedAtUnix": unix_timestamp(),
    });
    fs::write(
        root.join("setup-complete.json"),
        serde_json::to_vec_pretty(&marker)
            .map_err(|error| AppError::General(format!("セットアップ情報の保存に失敗: {error}")))?,
    )?;

    let status = runtime_status(app, &selected_profile)?;
    if !status.runtime_ready {
        return Err(AppError::General(status.message));
    }
    emit_progress(
        app,
        "runtime-ready",
        100,
        "See-Throughランタイムを準備しました。モデルは別コンソールで事前ダウンロードしてください",
    );
    Ok(status)
}

pub(crate) fn run_inference(
    app: &AppHandle,
    source_path: &str,
    requested_profile: &str,
    split_parts: bool,
    options: Option<SeeThroughOptions>,
) -> Result<SeeThroughRunResult, AppError> {
    let state = app.state::<AppState>();
    let _runtime_guard = state.see_through_runtime_lock.try_lock().map_err(|_| {
        AppError::General(
            "See-Throughのセットアップまたは推論を実行中です。完了後に再実行してください".into(),
        )
    })?;
    if state
        .see_through_model_download_pid
        .lock()
        .unwrap()
        .is_some()
    {
        return Err(AppError::General(
            "モデル事前ダウンロード中は推論を開始できません。完了後に再実行してください".into(),
        ));
    }
    run_inference_with_recovery(
        app,
        source_path,
        requested_profile,
        split_parts,
        options,
        true,
    )
}

/// allow_oom_retry: VRAM不足での自動リトライを1回だけ許可するかどうか。
/// リトライ呼び出し自身はfalseを渡し、再帰の無限ループを防ぐ
fn run_inference_with_recovery(
    app: &AppHandle,
    source_path: &str,
    requested_profile: &str,
    split_parts: bool,
    options: Option<SeeThroughOptions>,
    allow_oom_retry: bool,
) -> Result<SeeThroughRunResult, AppError> {
    let source = Path::new(source_path);
    if !source.is_file() {
        return Err(AppError::General(format!(
            "元画像が見つかりません: {}",
            source.display()
        )));
    }

    let status = runtime_status(app, requested_profile)?;
    if !status.ready {
        return Err(AppError::General(status.message));
    }

    let root = PathBuf::from(&status.runtime_root);
    // 推論ごとに作られるジョブ作業フォルダ（input/output）は使い捨てのスクラッチ領域。
    // 完了後にPSD/画像を全てAppStateへ読み込み済みで再利用されないため、放置すると
    // 無限に増え続けるディスクリークになる。新規ジョブ作成前に古いものを間引く
    prune_stale_job_dirs(&root);
    let repo = PathBuf::from(&status.repo_path);
    let python = PathBuf::from(&status.python_path);
    let gpu = resolve_gpu(app);
    let selected_profile = select_profile(requested_profile, gpu.as_ref());
    apply_runtime_compatibility_patches(&repo, selected_profile != "low-vram")?;
    let job = root
        .join("jobs")
        .join(format!("expression-{}", unix_timestamp_millis()));
    let input_dir = job.join("input");
    let output_dir = job.join("output");
    fs::create_dir_all(&input_dir)?;
    fs::create_dir_all(&output_dir)?;
    let extension = source
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("png");
    let managed_source = input_dir.join(format!("source.{extension}"));
    fs::copy(source, &managed_source)?;

    let script_name = if selected_profile == "low-vram" {
        "inference_psd_quantized.py"
    } else {
        "inference_psd.py"
    };
    let script = repo.join("inference/scripts").join(script_name);
    let mut args = vec![
        script.to_string_lossy().into_owned(),
        "--srcp".into(),
        managed_source.to_string_lossy().into_owned(),
        "--save_dir".into(),
        output_dir.to_string_lossy().into_owned(),
        "--save_to_psd".into(),
    ];
    if split_parts {
        args.push("--tblr_split".into());
    }
    append_inference_options(&mut args, &selected_profile, options.as_ref())?;
    emit_progress(
        app,
        "inference",
        2,
        &format!("See-Through解析を開始します ({selected_profile})"),
    );
    let inference_result = run_managed_command(
        app,
        &python.to_string_lossy(),
        args,
        &repo,
        "inference",
        gpu.as_ref(),
        &root.join("huggingface"),
    );
    if let Err(error) = inference_result {
        if primary_psd(&output_dir)?.is_some() {
            emit_progress(
                app,
                "inference",
                90,
                "See-Through exited with an error after writing PSD; continuing with the saved PSD.",
            );
        } else if split_parts && is_lr_split_failure(&error) {
            // 左右分割（耳などのL/R分離）は素材依存で失敗することがある。
            // VRAM等の環境問題ではないため、左右分解なしで自動リトライして作業を止めない
            emit_progress(
                app,
                "inference",
                5,
                "左右パーツ分解に失敗したため、左右分解なしで再実行しています",
            );
            let mut result = run_inference_with_recovery(
                app,
                source_path,
                requested_profile,
                false,
                options,
                allow_oom_retry,
            )?;
            result.split_parts_fallback = true;
            return Ok(result);
        } else if allow_oom_retry && should_attempt_inference_recovery(&error, &selected_profile) {
            let quantization_kernel_issue =
                is_profile_quantization_kernel_failure(&error, &selected_profile);
            let native_crash = is_native_crash(&error);
            match inference_recovery_plan(
                &selected_profile,
                options.as_ref(),
                quantization_kernel_issue,
                native_crash,
            ) {
                Some((retry_profile, retry_options, note)) => {
                    let retry_status = runtime_status(app, &retry_profile)?;
                    if !retry_status.ready {
                        let original = augment_inference_error(error, quantization_kernel_issue);
                        return Err(AppError::General(format!(
                            "{original}\n自動リトライ先の{}モデルが未準備のため切り替えませんでした。STEP3で{}を選び、「モデルDL用コンソールを開く」を実行してください。詳細: {}",
                            profile_label(&retry_profile),
                            profile_label(&retry_profile),
                            retry_status.message
                        )));
                    }
                    emit_progress(
                        app,
                        "inference",
                        5,
                        &format!("推論エラーが発生したため、{note}して再実行しています"),
                    );
                    let mut result = run_inference_with_recovery(
                        app,
                        source_path,
                        &retry_profile,
                        split_parts,
                        retry_options,
                        false,
                    )?;
                    result.oom_retry_note = Some(note);
                    return Ok(result);
                }
                None => {
                    return Err(augment_inference_error(error, quantization_kernel_issue));
                }
            }
        } else if should_augment_inference_error(&error, &selected_profile) {
            let quantization_kernel_issue =
                is_profile_quantization_kernel_failure(&error, &selected_profile);
            return Err(augment_inference_error(error, quantization_kernel_issue));
        } else {
            return Err(error);
        }
    }

    emit_progress(app, "load", 92, "生成PSDを読み込み、自動分類しています");
    let psd = primary_psd(&output_dir)?.ok_or_else(|| {
        AppError::General(format!(
            "See-Throughは完了しましたがPSDが見つかりません: {}",
            output_dir.display()
        ))
    })?;
    let psd_path = psd.to_string_lossy().into_owned();
    let slot_load = load_slot_inner(app.clone(), psd_path.clone())?;
    cache_original_image_for_canvas(&app, &managed_source)?;
    let default_mapping: HashMap<String, String> = slot_load
        .adjustable_layers
        .iter()
        .map(|layer| (layer.name.clone(), layer.default_target.clone()))
        .collect();
    let mapping_json = serde_json::to_string(&default_mapping)
        .map_err(|error| AppError::General(format!("自動分類情報の作成に失敗: {error}")))?;
    let mapping_preview = get_mapping_preview_inner(app.clone(), mapping_json)?;
    emit_progress(app, "complete", 100, "自動分解とレイヤー分類が完了しました");

    Ok(SeeThroughRunResult {
        psd_path,
        output_dir: output_dir.to_string_lossy().into_owned(),
        selected_profile,
        effective_options: options.clone(),
        slot_load,
        mapping_preview,
        split_parts_fallback: false,
        oom_retry_note: None,
    })
}

/// 左右分割（--tblr_split）由来の失敗か。inference_utils.pyのlr_split系トレース
/// バックが「直前のログ」としてエラーメッセージに含まれることを利用する
/// 直近数件を残して古い推論ジョブ作業フォルダ（jobs/expression-<ミリ秒>）を削除する。
/// 各ジョブはrun_inference完了時点で内容が全てAppStateへ読み込み済みで、フォルダ自体は
/// 二度と参照されないスクラッチ領域のため安全に削除できる（実行のたびに増え続けるのを防止）
fn prune_stale_job_dirs(root: &Path) {
    const KEEP_RECENT: usize = 3;
    let jobs_dir = root.join("jobs");
    let Ok(entries) = fs::read_dir(&jobs_dir) else {
        return;
    };
    let mut dirs: Vec<_> = entries
        .filter_map(Result::ok)
        .filter(|entry| entry.path().is_dir())
        .collect();
    if dirs.len() <= KEEP_RECENT {
        return;
    }
    // フォルダ名が expression-<unixミリ秒> のため、名前の降順=新しい順になる
    dirs.sort_by_key(|entry| entry.file_name());
    dirs.reverse();
    for stale in dirs.into_iter().skip(KEEP_RECENT) {
        let _ = fs::remove_dir_all(stale.path());
    }
}

fn is_lr_split_failure(error: &AppError) -> bool {
    let text = error.to_string();
    text.contains("lr_split") || text.contains("further_extr")
}

/// Windowsのネイティブクラッシュ（アクセス違反 0xC0000005 等）か。
/// Pythonの例外文字列が出ないハードクラッシュで、exit code -1073741819 で判別する。
/// low-vram（量子化）プロファイルでこれが起きる場合、新しいGPU（Blackwell等）で
/// bitsandbytesの4bitカーネルが動かず落ちているケースが多い
fn is_native_crash(error: &AppError) -> bool {
    let text = error.to_string();
    text.contains("-1073741819") || text.contains("ネイティブアクセス違反")
}

/// run_managed_commandの無応答タイムアウトで打ち切られた失敗か
fn is_hang_timeout(error: &AppError) -> bool {
    let message = error.to_string();
    message.contains("応答しなかったため中断しました")
        || message.contains("完了しなかったため中断しました")
}

/// CUDA/cuBLASのメモリ確保失敗（cublasCreate等）由来の失敗か。
/// 注意: この失敗はVRAM不足そのものより、bitsandbytes 4bit量子化カーネルが
/// GPUアーキテクチャ/ドライバと非互換なケースで起きることが多い（is_quantization_kernel_failure参照）
fn is_cuda_oom_failure(error: &AppError) -> bool {
    let text = error.to_string();
    text.contains("CUBLAS_STATUS_ALLOC_FAILED")
        || text.contains("CUDA out of memory")
        || text.contains("OutOfMemoryError")
        || (text.contains("CUDA error") && text.contains("memory"))
}

/// bitsandbytesの4bit量子化演算（matmul_4bit等）の最中に失敗したか。
/// 新しいGPUアーキテクチャ（例: Blackwell）ではbitsandbytesの同梱カーネルが
/// 未対応で、テキストエンコーダのような小さな処理でも即座に失敗することがある。
/// この場合はVRAMを空けても直らず、量子化を使わない高VRAMプロファイルへの
/// 切り替えが最も有効な回避策になる
fn is_quantization_kernel_failure(error: &AppError) -> bool {
    let text = error.to_string();
    text.contains("bitsandbytes") && (text.contains("matmul_4bit") || text.contains("cublasCreate"))
}

fn is_profile_quantization_kernel_failure(error: &AppError, selected_profile: &str) -> bool {
    selected_profile == "low-vram" && is_quantization_kernel_failure(error)
}

/// OOM/CUDAエラー発生時に安全に一度だけ試せる、より成功しやすい実行設定を返す
/// (再試行プロファイル, 再試行オプション, 変更内容の説明文)。
/// これ以上緩和できる設定が無ければ None（自動リトライしない）
fn should_attempt_inference_recovery(error: &AppError, selected_profile: &str) -> bool {
    is_cuda_oom_failure(error)
        || is_profile_quantization_kernel_failure(error, selected_profile)
        || is_native_crash(error)
        || (is_hang_timeout(error) && selected_profile == "low-vram")
}

fn should_augment_inference_error(error: &AppError, selected_profile: &str) -> bool {
    is_cuda_oom_failure(error)
        || is_profile_quantization_kernel_failure(error, selected_profile)
        || is_native_crash(error)
        || is_hang_timeout(error)
}

fn inference_recovery_plan(
    selected_profile: &str,
    options: Option<&SeeThroughOptions>,
    quantization_kernel_issue: bool,
    native_crash: bool,
) -> Option<(String, Option<SeeThroughOptions>, String)> {
    if selected_profile == "low-vram" && quantization_kernel_issue {
        // bitsandbytesの4bit量子化カーネルがこのGPU/ドライバでは動作しない可能性が高い。
        // オフロードを強めても直らないため、量子化を使わない高VRAMプロファイルへ切り替える
        return Some((
            "standard".to_string(),
            options.cloned(),
            "量子化(bitsandbytes)を使わない高VRAMプロファイルへ切替".to_string(),
        ));
    }
    if selected_profile == "low-vram" && native_crash {
        // 原因をbitsandbytesと断定せず、実装経路が異なるstandardを一度だけ試す。
        return Some((
            "standard".to_string(),
            options.cloned(),
            "省VRAM側のネイティブ処理を避け、高VRAMプロファイルへ切替".to_string(),
        ));
    }
    if selected_profile != "low-vram" {
        // 高VRAMプロファイルでのOOMまたはネイティブクラッシュは、実装経路と
        // メモリ特性が異なる省VRAM（量子化）プロファイルを、実測で最小だった
        // group offload有効・CPU offload無効で一度だけ試す。
        let mut next = options.cloned().unwrap_or_default();
        next.group_offload = Some("on".to_string());
        next.cpu_offload = Some("off".to_string());
        return Some((
            "low-vram".to_string(),
            Some(next),
            "省VRAMプロファイルへ切り替え".to_string(),
        ));
    }
    // 既に省VRAMプロファイル: 実測で最小だったgroup offloadを明示的に有効化して
    // 一度だけ再試行する。CPU offloadはカスタムVAEをGPUに残す必要があり、OOM回復には使わない。
    let mut next = options.cloned().unwrap_or_default();
    let cpu_was_enabled = next.cpu_offload.as_deref() == Some("on");
    let group_was_disabled = next.group_offload.as_deref() == Some("off");
    if !cpu_was_enabled && !group_was_disabled {
        return None;
    }
    next.cpu_offload = Some("off".to_string());
    next.group_offload = Some("on".to_string());
    Some((
        "low-vram".to_string(),
        Some(next),
        "実測でVRAM使用量が少ないGroup offloadへ切り替え".to_string(),
    ))
}

/// 自動リトライ済み、または回復先を利用できない推論エラーに、原因別のヒントを添える。
/// ネイティブアクセス違反を根拠なくCUDA OOMと断定しないこと。
fn augment_inference_error(error: AppError, quantization_kernel_issue: bool) -> AppError {
    let hint = if quantization_kernel_issue {
        "bitsandbytesの4bit量子化演算がこのGPU（新しいアーキテクチャの可能性）で失敗している様子です。\
VRAM不足ではなく、量子化カーネルとGPU/ドライバの非互換が原因と考えられます。\
STEP3の実行プロファイルを「高VRAM」に切り替えて再実行してください（今回は自動切替も失敗しています。\
高VRAMプロファイルは量子化を使わないため、多くの場合これで解消します）。"
    } else if is_native_crash(&error) {
        "Python/PyTorchのネイティブモジュールでアクセス違反が発生しました。\
この終了コードだけではVRAM不足とは判定できません。実行プロファイルで異なる推論経路を試す、\
GPUを使用している他の処理を終了する、GPUドライバを確認する、の順で切り分けてください。"
    } else if is_hang_timeout(&error) {
        "推論処理が長時間応答しませんでした。VRAM不足とは限らないため、実行プロファイル、\
GPUドライバ、同時にGPUを使用している処理を確認してから再実行してください。"
    } else {
        "GPUのVRAM不足（CUDA/cuBLASのメモリ確保失敗）が原因の可能性があります。\
他にGPUを使用しているアプリ（ブラウザ・ゲーム・他のAI処理等）を終了する、\
STEP3の「使用GPU」でVRAMに余裕のあるGPUへ切り替える、のいずれかを試してから再実行してください。"
    };
    AppError::General(format!("{error}\n\n{hint}"))
}

fn append_inference_options(
    args: &mut Vec<String>,
    selected_profile: &str,
    options: Option<&SeeThroughOptions>,
) -> Result<(), AppError> {
    let Some(options) = options else {
        return Ok(());
    };

    validate_resolution_option(options.resolution, false, "resolution")?;
    validate_resolution_option(options.resolution_depth, true, "resolution_depth")?;
    push_positive_arg(args, "--seed", options.seed, 0, 999_999_999, "seed")?;
    push_positive_arg(
        args,
        "--resolution",
        options.resolution,
        256,
        4096,
        "resolution",
    )?;
    push_positive_arg(
        args,
        "--resolution_depth",
        options.resolution_depth,
        -1,
        4096,
        "resolution_depth",
    )?;

    if selected_profile == "low-vram" {
        push_positive_arg(
            args,
            "--num_inference_steps",
            options.inference_steps,
            1,
            150,
            "inference_steps",
        )?;
    } else {
        validate_auto_or_positive_option(
            options.inference_steps_depth,
            1,
            150,
            "inference_steps_depth",
        )?;
        push_positive_arg(
            args,
            "--inference_steps",
            options.inference_steps,
            1,
            150,
            "inference_steps",
        )?;
        push_positive_arg(
            args,
            "--inference_steps_depth",
            options.inference_steps_depth,
            -1,
            150,
            "inference_steps_depth",
        )?;
    }

    if selected_profile == "low-vram" {
        let cpu_mode = option_mode(options.cpu_offload.as_deref())?;
        if cpu_mode == "on" {
            // upstreamではCPU offloadとgroup offloadはif/elseの排他経路。
            // ユーザー指定のCPU offloadを優先し、Python側の既定group=trueも明示的に無効化する。
            args.push("--cpu_offload".into());
            args.push("--no_group_offload".into());
        } else {
            match option_mode(options.group_offload.as_deref())? {
                "on" => args.push("--group_offload".into()),
                "off" => args.push("--no_group_offload".into()),
                _ => {}
            }
            if cpu_mode == "off" {
                args.push("--no_cpu_offload".into());
            }
        }
    } else if option_mode(options.group_offload.as_deref())? == "on" {
        args.push("--group_offload".into());
    }

    Ok(())
}

fn validate_resolution_option(
    value: Option<i64>,
    allow_auto: bool,
    label: &str,
) -> Result<(), AppError> {
    let Some(value) = value else {
        return Ok(());
    };
    if allow_auto && value == -1 {
        return Ok(());
    }
    if !(256..=4096).contains(&value) || value % 64 != 0 {
        return Err(AppError::General(format!(
            "{label}は256〜4096の64刻み{}で指定してください: {value}",
            if allow_auto { "、または-1" } else { "" }
        )));
    }
    Ok(())
}

fn validate_auto_or_positive_option(
    value: Option<i64>,
    min: i64,
    max: i64,
    label: &str,
) -> Result<(), AppError> {
    let Some(value) = value else {
        return Ok(());
    };
    if value == -1 || (min..=max).contains(&value) {
        return Ok(());
    }
    Err(AppError::General(format!(
        "{label}は-1または{min}〜{max}で指定してください: {value}"
    )))
}

fn push_positive_arg(
    args: &mut Vec<String>,
    name: &str,
    value: Option<i64>,
    min: i64,
    max: i64,
    label: &str,
) -> Result<(), AppError> {
    let Some(value) = value else {
        return Ok(());
    };
    if value < min || value > max {
        return Err(AppError::General(format!(
            "See-Through設定 {label} は {min} から {max} の範囲で指定してください"
        )));
    }
    args.push(name.into());
    args.push(value.to_string());
    Ok(())
}

fn option_mode(value: Option<&str>) -> Result<&str, AppError> {
    match value.unwrap_or("default") {
        "default" => Ok("default"),
        "on" => Ok("on"),
        "off" => Ok("off"),
        other => Err(AppError::General(format!(
            "See-Through設定の指定が不正です: {other}"
        ))),
    }
}

fn default_runtime_root(app: &AppHandle) -> Result<PathBuf, AppError> {
    app.path()
        .app_local_data_dir()
        .map(|path| path.join("see-through"))
        .map_err(|error| AppError::General(format!("アプリデータ保存先を取得できません: {error}")))
}

/// インストール先設定ファイルのパス（Python環境・モデル本体とは別の、軽量な設定置き場）
fn install_location_config_path(app: &AppHandle) -> Result<PathBuf, AppError> {
    app.path()
        .app_config_dir()
        .map(|path| path.join("see-through-location.json"))
        .map_err(|error| AppError::General(format!("設定保存先を取得できません: {error}")))
}

/// ユーザーがSTEP3で指定したカスタムのインストール先（未設定ならNone）
fn read_custom_runtime_root(app: &AppHandle) -> Option<PathBuf> {
    let config_path = install_location_config_path(app).ok()?;
    let bytes = fs::read(config_path).ok()?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    let path = value.get("runtimeRoot")?.as_str()?;
    if path.trim().is_empty() {
        return None;
    }
    Some(PathBuf::from(path))
}

fn runtime_root(app: &AppHandle) -> Result<PathBuf, AppError> {
    if let Ok(path) = std::env::var("PACHIPAKUGEN_SEE_THROUGH_ROOT") {
        if !path.trim().is_empty() {
            return Ok(PathBuf::from(path));
        }
    }
    if let Some(custom) = read_custom_runtime_root(app) {
        return Ok(custom);
    }
    default_runtime_root(app)
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SeeThroughInstallLocation {
    pub path: String,
    pub is_default: bool,
}

#[tauri::command]
pub fn get_see_through_install_location(
    app: AppHandle,
) -> Result<SeeThroughInstallLocation, AppError> {
    let current = runtime_root(&app)?;
    let default_path = default_runtime_root(&app)?;
    Ok(SeeThroughInstallLocation {
        is_default: current == default_path,
        path: current.to_string_lossy().into_owned(),
    })
}

/// STEP3のインストール先変更。path=None（またはデフォルトと同一）で既定に戻す。
/// 既存の保存先にあるPython環境・モデルは移動しない
/// （新しい場所でランタイム構築とモデル事前DLが必要）
#[tauri::command]
pub fn set_see_through_install_location(
    app: AppHandle,
    path: Option<String>,
) -> Result<SeeThroughInstallLocation, AppError> {
    let config_path = install_location_config_path(&app)?;
    if let Some(parent) = config_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let default_path = default_runtime_root(&app)?;
    let trimmed = path.as_deref().map(str::trim).filter(|p| !p.is_empty());
    match trimmed {
        Some(custom) if PathBuf::from(custom) != default_path => {
            fs::write(
                &config_path,
                serde_json::to_vec_pretty(&serde_json::json!({ "runtimeRoot": custom }))
                    .map_err(|error| AppError::General(format!("設定の保存に失敗: {error}")))?,
            )?;
        }
        _ => {
            if config_path.is_file() {
                fs::remove_file(&config_path)?;
            }
        }
    }
    get_see_through_install_location(app)
}

fn venv_python(root: &Path) -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        root.join(".venv/Scripts/python.exe")
    }
    #[cfg(not(target_os = "windows"))]
    {
        root.join(".venv/bin/python")
    }
}

fn find_executable(env_name: &str, candidates: &[&str]) -> Result<String, AppError> {
    if let Ok(path) = std::env::var(env_name) {
        if Command::new(&path).arg("--version").output().is_ok() {
            return Ok(path);
        }
    }
    for candidate in candidates {
        if Command::new(candidate).arg("--version").output().is_ok() {
            return Ok((*candidate).to_string());
        }
    }
    Err(AppError::General(format!(
        "{}が見つかりません。PATHまたは{}を設定してください",
        candidates.join(" / "),
        env_name
    )))
}

fn run_managed_command(
    app: &AppHandle,
    program: &str,
    args: Vec<String>,
    cwd: &Path,
    stage: &str,
    gpu: Option<&GpuInfo>,
    hf_home: &Path,
) -> Result<(), AppError> {
    let mut command = Command::new(program);
    let hf_hub_cache = hf_home.join("hub");
    command
        .args(args)
        .current_dir(cwd)
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        .env("CUDA_MODULE_LOADING", "LAZY")
        .env("HF_HOME", hf_home)
        .env("HF_HUB_CACHE", &hf_hub_cache)
        .env("HUGGINGFACE_HUB_CACHE", &hf_hub_cache)
        .env("TRANSFORMERS_CACHE", &hf_hub_cache)
        .env("DIFFUSERS_CACHE", &hf_hub_cache)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if stage == "inference" {
        // 推論開始後の暗黙ダウンロードを禁止する。必要なモデルは別コンソールで
        // 事前DL・検証済みのため、欠損があればネットワーク待ちではなく即時エラーにする。
        command.env("HF_HUB_OFFLINE", "1");
        command.env("TRANSFORMERS_OFFLINE", "1");
        command.env("DIFFUSERS_OFFLINE", "1");
    }
    if let Some(gpu) = gpu {
        command.env("CUDA_VISIBLE_DEVICES", &gpu.uuid);
    }
    let mut child = command
        .spawn()
        .map_err(|error| AppError::General(format!("{program}を起動できません: {error}")))?;
    let pid = child.id();
    *app.state::<AppState>().see_through_pid.lock().unwrap() = Some(pid);

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| AppError::General("See-Through stdoutを取得できません".into()))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| AppError::General("See-Through stderrを取得できません".into()))?;
    let stdout_app = app.clone();
    let stderr_app = app.clone();
    let stdout_stage = stage.to_string();
    let stderr_stage = stage.to_string();

    let stdout_reader =
        std::thread::spawn(move || read_process_lines(stdout, &stdout_app, &stdout_stage));
    let stderr_reader =
        std::thread::spawn(move || read_process_lines(stderr, &stderr_app, &stderr_stage));

    // アプリ管理のセットアップ/推論だけを壁時計上限で保護する。数GB級モデル取得は
    // 可視コンソールへ分離しており、ここでは実行しない。
    let wall_clock_timeout = Duration::from_secs(30 * 60);
    let started = Instant::now();
    let process_status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if started.elapsed() > wall_clock_timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    *app.state::<AppState>().see_through_pid.lock().unwrap() = None;
                    let _ = stdout_reader.join();
                    let _ = stderr_reader.join();
                    let guidance = "GPUドライバまたはセットアップ処理が応答していません。実行プロファイル、GPU、ネットワーク接続を確認して再実行してください。";
                    return Err(AppError::General(format!(
                        "See-Through処理が{}分間完了しなかったため中断しました。{guidance}",
                        wall_clock_timeout.as_secs() / 60
                    )));
                }
                std::thread::sleep(Duration::from_millis(500));
            }
            Err(error) => return Err(AppError::from(error)),
        }
    };
    *app.state::<AppState>().see_through_pid.lock().unwrap() = None;
    let stdout_text = stdout_reader.join().unwrap_or_default();
    let stderr_text = stderr_reader.join().unwrap_or_default();
    if !process_status.success() {
        return Err(AppError::General(format_process_failure(
            process_status.code().unwrap_or(-1),
            &stdout_text,
            &stderr_text,
        )));
    }
    Ok(())
}

fn apply_runtime_compatibility_patches(
    repo: &Path,
    require_standard_compatibility: bool,
) -> Result<(), AppError> {
    if require_standard_compatibility {
        apply_bf16_loading_compatibility_patch(repo)?;
        apply_standard_pipeline_cleanup_compatibility_patch(repo)?;
    } else {
        apply_quantized_cpu_offload_compatibility_patch(repo)?;
    }
    Ok(())
}

/// 本家quantized版ではNF4 MarigoldだけCPU offload設定を参照せず、常にCUDAへ
/// 全モデルを配置する。LayerDiff側と同じ優先規則でmodel CPU offloadを適用する。
fn apply_quantized_cpu_offload_compatibility_patch(repo: &Path) -> Result<(), AppError> {
    let path = repo.join("inference/scripts/inference_psd_quantized.py");
    let original = fs::read_to_string(&path).map_err(|error| {
        AppError::General(format!(
            "See-Through省VRAM推論スクリプトの読み込みに失敗しました: {} ({error})",
            path.display()
        ))
    })?;
    let normalized = original.replace("\r\n", "\n");
    let patched = patch_quantized_layerdiff_cpu_offload(&normalized)?;
    let patched = patch_quantized_marigold_cpu_offload(&patched)?;
    let patched = patch_quantized_group_offload(&patched)?;
    if patched != original {
        fs::write(&path, patched).map_err(|error| {
            AppError::General(format!(
                "See-Through省VRAM推論のCPU offload設定を保存できませんでした: {} ({error})",
                path.display()
            ))
        })?;
    }

    for (relative_path, before, after, label) in [
        (
            "common/modules/layerdiffuse/diffusers_kdiffusion_sdxl.py",
            "        device = self.text_encoder.device",
            "        # PachiPakuGen: Accelerate moves the encoder after the offload hook runs.\n        device = self._execution_device",
            "LayerDiff",
        ),
        (
            "common/modules/layerdiffuse/diffusers_kdiffusion_sdxl.py",
            "    ):\n\n        device = self.unet.device\n        dtype = self.unet.dtype",
            "    ):\n\n        # PachiPakuGen: the UNet is on CPU until its Accelerate hook runs.\n        device = self._execution_device\n        dtype = self.unet.dtype",
            "LayerDiff推論",
        ),
        (
            "common/modules/layerdiffuse/diffusers_kdiffusion_sdxl.py",
            "                    group_index=group_index\n                )[0]",
            "                    group_index=group_index\n                )[0].to(device)",
            "LayerDiff出力",
        ),
        (
            "common/modules/marigold/marigold_depth_pipeline.py",
            "        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)",
            "        # PachiPakuGen: send inputs to Accelerate's execution device under CPU offload.\n        text_input_ids = text_inputs.input_ids.to(self._execution_device)",
            "Marigold",
        ),
        (
            "common/modules/marigold/marigold_depth_pipeline.py",
            "        is_3d = isinstance(self.unet, UNetFrameConditionModel)\n        device = self.device",
            "        is_3d = isinstance(self.unet, UNetFrameConditionModel)\n        # PachiPakuGen: the UNet is on CPU until its Accelerate hook runs.\n        device = self._execution_device",
            "Marigold推論",
        ),
        (
            "common/modules/marigold/marigold_depth_pipeline.py",
            "                noise_pred = self.unet(\n                    unet_input, t, encoder_hidden_states=batch_empty_text_embed\n                ).sample  # [B, 4, h, w]",
            "                noise_pred = self.unet(\n                    unet_input, t, encoder_hidden_states=batch_empty_text_embed\n                ).sample.to(device)  # [B, 4, h, w]",
            "Marigold出力",
        ),
    ] {
        let module_path = repo.join(relative_path);
        let module_source = fs::read_to_string(&module_path).map_err(|error| {
            AppError::General(format!(
                "See-Through省VRAM推論の{label}モジュールを読み込めませんでした: {} ({error})",
                module_path.display()
            ))
        })?;
        let module_patched = patch_cpu_offload_execution_device(
            &module_source.replace("\r\n", "\n"),
            before,
            after,
            label,
        )?;
        if module_patched != module_source {
            fs::write(&module_path, module_patched).map_err(|error| {
                AppError::General(format!(
                    "See-Through省VRAM推論の{label} CPU offload互換設定を保存できませんでした: {} ({error})",
                    module_path.display()
                ))
            })?;
        }
    }
    Ok(())
}

fn patch_quantized_layerdiff_cpu_offload(source: &str) -> Result<String, AppError> {
    const BEFORE: &str = "        if args.cpu_offload:\n            # VAE + TransparentVAE to bf16; quantized components handled by bnb\n            pipeline.vae.to(dtype=torch.bfloat16)\n            pipeline.trans_vae.to(dtype=torch.bfloat16)\n            pipeline.enable_model_cpu_offload()";
    const PREVIOUS_AFTER: &str = "        if args.cpu_offload:\n            # Keep custom VAE modules on CUDA: their direct method calls bypass Accelerate hooks.\n            pipeline.vae.to(dtype=torch.bfloat16)\n            pipeline.trans_vae.to(dtype=torch.bfloat16)\n            pipeline.model_cpu_offload_seq = 'unet'\n            pipeline._exclude_from_cpu_offload = ['vae', 'trans_vae']\n            pipeline.enable_model_cpu_offload()";
    const AFTER: &str = "        if args.cpu_offload:\n            # Keep custom VAE modules on CUDA: their direct method calls bypass Accelerate hooks.\n            pipeline.vae.to(dtype=torch.bfloat16)\n            pipeline.trans_vae.to(dtype=torch.bfloat16)\n            pipeline.model_cpu_offload_seq = 'unet'\n            pipeline._exclude_from_cpu_offload = ['vae', 'trans_vae']\n            pipeline.enable_model_cpu_offload()\n            pipeline.vae.to(device='cuda')\n            pipeline.trans_vae.to(device='cuda')";

    let normalized = source.replace("\r\n", "\n");
    if normalized.matches(AFTER).count() == 1 && !normalized.contains(BEFORE) {
        return Ok(normalized);
    }
    if normalized.matches(PREVIOUS_AFTER).count() == 1 && !normalized.contains(BEFORE) {
        return Ok(normalized.replacen(PREVIOUS_AFTER, AFTER, 1));
    }
    if normalized.matches(BEFORE).count() != 1 || normalized.contains(AFTER) {
        return Err(AppError::General(
            "See-Through省VRAM推論へLayerDiff CPU offload互換設定を適用できません。公式スクリプトの構造が変更されています"
                .into(),
        ));
    }
    Ok(normalized.replacen(BEFORE, AFTER, 1))
}

fn patch_quantized_marigold_cpu_offload(source: &str) -> Result<String, AppError> {
    const BEFORE: &str = "        marigold_pipe.vae.to(device='cuda')\n        marigold_pipe.unet.to(device='cuda')\n        # Text encoder may be quantized (from pre-quantized repo) — only move device, not dtype\n        if not getattr(marigold_pipe.text_encoder, 'is_quantized', False) and \\\n           not getattr(marigold_pipe.text_encoder, 'quantization_method', None):\n            marigold_pipe.text_encoder.to(device='cuda')\n        if getattr(args, 'group_offload', False):\n            marigold_pipe.enable_group_offload('cuda', num_blocks_per_group=1)";
    const AFTER: &str = "        # PachiPakuGen: honor CPU offload for the NF4 Marigold stage.\n        if args.cpu_offload:\n            # Keep the custom VAE on CUDA because it invokes encoder/decoder directly.\n            marigold_pipe.vae.to(dtype=torch.bfloat16)\n            marigold_pipe.model_cpu_offload_seq = 'unet'\n            marigold_pipe._exclude_from_cpu_offload = ['vae']\n            marigold_pipe.enable_model_cpu_offload()\n            marigold_pipe.vae.to(device='cuda')\n        else:\n            marigold_pipe.vae.to(device='cuda')\n            marigold_pipe.unet.to(device='cuda')\n            # Text encoder may be quantized (from pre-quantized repo) — only move device, not dtype\n            if not getattr(marigold_pipe.text_encoder, 'is_quantized', False) and \\\n               not getattr(marigold_pipe.text_encoder, 'quantization_method', None):\n                marigold_pipe.text_encoder.to(device='cuda')\n            if getattr(args, 'group_offload', False):\n                marigold_pipe.enable_group_offload('cuda', num_blocks_per_group=1)";
    const MARKER: &str = "        # PachiPakuGen: honor CPU offload for the NF4 Marigold stage.";
    const CACHE_ANCHOR: &str = "        marigold_pipe.cache_tag_embeds()";
    const GROUP_BEFORE: &str = "marigold_pipe.enable_group_offload('cuda', num_blocks_per_group=1)";
    const GROUP_AFTER: &str = "marigold_pipe.enable_group_offload(\n                    'cuda', num_blocks_per_group=1,\n                    exclude_modules=['unet', 'text_encoder'])";

    let normalized = source.replace("\r\n", "\n");
    if normalized.matches(MARKER).count() == 1 {
        let start = normalized.find(MARKER).expect("marker count was checked");
        let tail = &normalized[start..];
        let end = tail.find(CACHE_ANCHOR).ok_or_else(|| {
            AppError::General(
                "See-Through省VRAM推論のMarigold CPU offload設定が不完全です。ランタイムを再セットアップしてください"
                    .into(),
            )
        })?;
        let existing = &tail[..end];
        if !existing.contains("if args.cpu_offload:")
            || !existing.contains("marigold_pipe.enable_model_cpu_offload()")
            || !existing.contains("marigold_pipe.enable_group_offload(")
        {
            return Err(AppError::General(
                "See-Through省VRAM推論のMarigold CPU offload設定が不完全です。ランタイムを再セットアップしてください"
                .into(),
            ));
        }
        let canonical = match (
            existing.matches(GROUP_BEFORE).count(),
            existing.matches(GROUP_AFTER).count(),
        ) {
            (1, 0) => AFTER.to_string(),
            (0, 1) => AFTER.replacen(GROUP_BEFORE, GROUP_AFTER, 1),
            _ => {
                return Err(AppError::General(
                    "See-Through省VRAM推論のMarigold group offload設定が部分適用されています"
                        .into(),
                ));
            }
        };
        let mut patched = String::with_capacity(normalized.len());
        patched.push_str(&normalized[..start]);
        patched.push_str(&canonical);
        patched.push('\n');
        patched.push_str(&tail[end..]);
        return Ok(patched);
    }
    if normalized.matches(BEFORE).count() != 1 || normalized.contains(MARKER) {
        return Err(AppError::General(
            "See-Through省VRAM推論へMarigold CPU offload設定を適用できません。公式スクリプトの構造が変更されています"
                .into(),
        ));
    }
    Ok(normalized.replacen(BEFORE, AFTER, 1))
}

fn patch_cpu_offload_execution_device(
    source: &str,
    before: &str,
    after: &str,
    label: &str,
) -> Result<String, AppError> {
    let before_count = source.matches(before).count();
    let after_count = source.matches(after).count();
    if after_count == 1 {
        let expected_before_count = usize::from(after.contains(before));
        if before_count == expected_before_count {
            return Ok(source.to_string());
        }
    }
    if before_count != 1 || after_count != 0 {
        return Err(AppError::General(format!(
            "See-Through省VRAM推論へ{label} CPU offload実行デバイス設定を適用できません。公式スクリプトの構造が変更されています"
        )));
    }
    Ok(source.replacen(before, after, 1))
}

fn patch_quantized_group_offload(source: &str) -> Result<String, AppError> {
    const LAYERDIFF_BEFORE: &str = "pipeline.enable_group_offload('cuda', num_blocks_per_group=1)";
    const LAYERDIFF_AFTER: &str = "pipeline.enable_group_offload(\n                    'cuda', num_blocks_per_group=1,\n                    exclude_modules=['text_encoder', 'text_encoder_2'])";
    const MARIGOLD_BEFORE: &str =
        "marigold_pipe.enable_group_offload('cuda', num_blocks_per_group=1)";
    const MARIGOLD_AFTER: &str = "marigold_pipe.enable_group_offload(\n                    'cuda', num_blocks_per_group=1,\n                    exclude_modules=['unet', 'text_encoder'])";
    const NF4_LAYERDIFF_BEFORE: &str = "            # Don't manually .to(cuda) quantized components -- bnb handles device placement\n            if getattr(args, 'group_offload', False):\n                pipeline.enable_group_offload(\n                    'cuda', num_blocks_per_group=1,\n                    exclude_modules=['text_encoder', 'text_encoder_2'])";
    const NF4_LAYERDIFF_AFTER: &str = "            # Don't manually .to(cuda) quantized components -- bnb handles device placement\n            if getattr(args, 'group_offload', False):\n                # bitsandbytes 4-bit weights cannot be moved back to CPU by group hooks.\n                pipeline.enable_group_offload(\n                    'cuda', num_blocks_per_group=1,\n                    exclude_modules=['unet', 'text_encoder', 'text_encoder_2'])";

    let mut patched = source.to_string();
    let layerdiff_state = (
        patched.matches(LAYERDIFF_BEFORE).count(),
        patched.matches(LAYERDIFF_AFTER).count(),
        patched.matches(NF4_LAYERDIFF_AFTER).count(),
    );
    if layerdiff_state == (2, 0, 0) {
        patched = patched.replace(LAYERDIFF_BEFORE, LAYERDIFF_AFTER);
    } else if layerdiff_state != (0, 1, 1) {
        return Err(AppError::General(
            "See-Through省VRAM推論のLayerDiff group offload設定が変更または部分適用されています"
                .into(),
        ));
    }
    if patched.matches(NF4_LAYERDIFF_BEFORE).count() == 1 {
        patched = patched.replacen(NF4_LAYERDIFF_BEFORE, NF4_LAYERDIFF_AFTER, 1);
    } else if patched.matches(NF4_LAYERDIFF_AFTER).count() != 1 {
        return Err(AppError::General(
            "See-Through省VRAM推論へNF4 UNet除外設定を適用できません。公式スクリプトの構造が変更されています"
                .into(),
        ));
    }

    match (
        patched.matches(MARIGOLD_BEFORE).count(),
        patched.matches(MARIGOLD_AFTER).count(),
    ) {
        (2, 0) => patched = patched.replace(MARIGOLD_BEFORE, MARIGOLD_AFTER),
        (0, 2) => {}
        _ => {
            return Err(AppError::General(
                "See-Through省VRAM推論のMarigold group offload設定が変更または部分適用されています"
                    .into(),
            ));
        }
    }
    Ok(patched)
}

fn apply_bf16_loading_compatibility_patch(repo: &Path) -> Result<(), AppError> {
    let path = repo.join("common/utils/inference_utils.py");
    let original = fs::read_to_string(&path).map_err(|error| {
        AppError::General(format!(
            "See-Through互換設定の読み込みに失敗しました: {} ({error})",
            path.display()
        ))
    })?;
    // Windows(CRLF)でgit checkoutされたファイルだと、複数行にまたがるパターン
    // （改行を含むもの）がLF前提の比較と一致せず常にスキップされてしまうため、
    // 比較・書き戻しの前に改行をLFへ正規化する（Python実行には影響しない）
    let normalized = original.replace("\r\n", "\n");
    let patched = patch_bf16_loading(&normalized).map_err(|error| {
        AppError::General(format!(
            "See-Through標準推論へBF16互換設定を適用できません。ランタイムを再セットアップしてください: {error}"
        ))
    })?;
    if patched != original {
        fs::write(&path, patched).map_err(|error| {
            AppError::General(format!(
                "See-Through互換設定の保存に失敗しました: {} ({error})",
                path.display()
            ))
        })?;
    }
    Ok(())
}

/// 本家のquantized/blockswap版と同じく、LayerDiffとMarigoldを同時にGPUへ
/// 保持しないようstandard版のステージ境界へ明示解放を追加する。
/// upstreamの構造が変わって安全に適用できない場合は、危険なstandard推論を開始しない。
fn apply_standard_pipeline_cleanup_compatibility_patch(repo: &Path) -> Result<(), AppError> {
    let path = repo.join("inference/scripts/inference_psd.py");
    let original = fs::read_to_string(&path).map_err(|error| {
        AppError::General(format!(
            "See-Through標準推論スクリプトの読み込みに失敗しました: {} ({error})",
            path.display()
        ))
    })?;
    let normalized = original.replace("\r\n", "\n");
    let patched = patch_standard_pipeline_cleanup(&normalized)?;
    if patched != original {
        fs::write(&path, patched).map_err(|error| {
            AppError::General(format!(
                "See-Through標準推論のメモリ解放設定を保存できませんでした: {} ({error})",
                path.display()
            ))
        })?;
    }
    Ok(())
}

fn patch_standard_pipeline_cleanup(source: &str) -> Result<String, AppError> {
    const LAYERDIFF_CLEANUP: &str = "        # PachiPakuGen: release LayerDiff before loading Marigold.\n        inference_utils.layerdiff_pipeline = None\n        gc.collect()\n        torch.cuda.empty_cache()\n";
    const MARIGOLD_CLEANUP: &str = "        # PachiPakuGen: release Marigold before PSD assembly.\n        inference_utils.marigold_pipeline = None\n        gc.collect()\n        torch.cuda.empty_cache()\n";
    const MARIGOLD_ANCHOR: &str = "        print('running marigold...')";
    const PSD_ASSEMBLY_ANCHOR: &str = "        srcname = osp.basename(osp.splitext(srcp)[0])";

    let normalized = source.replace("\r\n", "\n");
    let has_layerdiff_cleanup = normalized.contains(LAYERDIFF_CLEANUP);
    let has_marigold_cleanup = normalized.contains(MARIGOLD_CLEANUP);
    let has_cleanup_fragment = normalized.contains("PachiPakuGen: release LayerDiff")
        || normalized.contains("PachiPakuGen: release Marigold")
        || normalized.contains("inference_utils.layerdiff_pipeline = None")
        || normalized.contains("inference_utils.marigold_pipeline = None");

    if has_layerdiff_cleanup && has_marigold_cleanup && normalized.contains("import gc\n") {
        validate_standard_pipeline_cleanup(&normalized)?;
        return Ok(normalized);
    }
    if has_cleanup_fragment {
        return Err(AppError::General(
            "See-Through標準推論のメモリ解放設定が不完全です。ランタイムを再セットアップしてください"
                .into(),
        ));
    }
    if !normalized.contains("apply_layerdiff(")
        || !normalized.contains("apply_marigold(")
        || !normalized.contains(MARIGOLD_ANCHOR)
        || !normalized.contains(PSD_ASSEMBLY_ANCHOR)
    {
        return Err(AppError::General(
            "See-Through標準推論へメモリ解放設定を適用できません。公式スクリプトの構造が変更されています"
                .into(),
        ));
    }

    let mut patched = normalized;
    if !patched.contains("import gc\n") {
        if !patched.contains("import os\n") {
            return Err(AppError::General(
                "See-Through標準推論へgc設定を追加できません。公式スクリプトのimport構造が変更されています"
                    .into(),
            ));
        }
        patched = patched.replacen("import os\n", "import os\nimport gc\n", 1);
    }
    patched = patched.replacen(
        MARIGOLD_ANCHOR,
        &format!("{LAYERDIFF_CLEANUP}\n{MARIGOLD_ANCHOR}"),
        1,
    );
    patched = patched.replacen(
        PSD_ASSEMBLY_ANCHOR,
        &format!("{MARIGOLD_CLEANUP}\n{PSD_ASSEMBLY_ANCHOR}"),
        1,
    );
    validate_standard_pipeline_cleanup(&patched)?;
    Ok(patched)
}

fn validate_standard_pipeline_cleanup(source: &str) -> Result<(), AppError> {
    const LAYERDIFF_RELEASE: &str = "inference_utils.layerdiff_pipeline = None";
    const MARIGOLD_RELEASE: &str = "inference_utils.marigold_pipeline = None";
    const MARIGOLD_START: &str = "print('running marigold...')";
    const PSD_ASSEMBLY: &str = "srcname = osp.basename(osp.splitext(srcp)[0])";

    let unique = [
        "import gc\n",
        "from utils import inference_utils\n",
        "apply_layerdiff(",
        LAYERDIFF_RELEASE,
        MARIGOLD_START,
        "apply_marigold(",
        MARIGOLD_RELEASE,
        PSD_ASSEMBLY,
    ];
    if unique
        .iter()
        .any(|needle| source.matches(needle).count() != 1)
        || source.matches("torch.cuda.empty_cache()").count() != 2
        || source.matches("gc.collect()").count() != 2
    {
        return Err(AppError::General(
            "See-Through標準推論のメモリ解放設定の個数またはimport構造が不正です".into(),
        ));
    }

    let layerdiff_call = source.find("apply_layerdiff(").unwrap();
    let layerdiff_release = source.find(LAYERDIFF_RELEASE).unwrap();
    let marigold_start = source.find(MARIGOLD_START).unwrap();
    let marigold_call = source.find("apply_marigold(").unwrap();
    let marigold_release = source.find(MARIGOLD_RELEASE).unwrap();
    let psd_assembly = source.find(PSD_ASSEMBLY).unwrap();
    if !(layerdiff_call < layerdiff_release
        && layerdiff_release < marigold_start
        && marigold_start < marigold_call
        && marigold_call < marigold_release
        && marigold_release < psd_assembly)
    {
        return Err(AppError::General(
            "See-Through標準推論のメモリ解放設定が正しいステージ境界にありません".into(),
        ));
    }
    Ok(())
}

fn patch_bf16_loading(source: &str) -> Result<String, AppError> {
    let replacements = [
        (
            "TransparentVAE.from_pretrained(pretrained, subfolder='trans_vae')",
            "TransparentVAE.from_pretrained(pretrained, subfolder='trans_vae', torch_dtype=torch.bfloat16)",
        ),
        (
            "UNetFrameConditionModel.from_pretrained(pretrained, subfolder='unet')",
            "UNetFrameConditionModel.from_pretrained(pretrained, subfolder='unet', torch_dtype=torch.bfloat16)",
        ),
        (
            "UNetFrameConditionModel.from_pretrained(unet_ckpt)",
            "UNetFrameConditionModel.from_pretrained(unet_ckpt, torch_dtype=torch.bfloat16)",
        ),
        (
            "scheduler=None\n        )",
            "scheduler=None, torch_dtype=torch.bfloat16\n        )",
        ),
        (
            "MarigoldDepthPipeline.from_pretrained(pretrained, unet=unet)",
            "MarigoldDepthPipeline.from_pretrained(pretrained, unet=unet, torch_dtype=torch.bfloat16)",
        ),
        (
            "layerdiff_pipeline.enable_group_offload('cuda', num_blocks_per_group=1)",
            "layerdiff_pipeline.enable_group_offload(\n                'cuda', num_blocks_per_group=1,\n                exclude_modules=['text_encoder', 'text_encoder_2'])",
        ),
        (
            "marigold_pipeline.enable_group_offload('cuda', num_blocks_per_group=1)",
            "marigold_pipeline.enable_group_offload(\n                'cuda', num_blocks_per_group=1,\n                exclude_modules=['text_encoder'])",
        ),
    ];
    let mut patched = source.to_string();
    for (before, after) in replacements {
        match (
            patched.matches(before).count(),
            patched.matches(after).count(),
        ) {
            (1, 0) => patched = patched.replacen(before, after, 1),
            (0, 1) => {}
            _ => {
                return Err(AppError::General(format!(
                    "See-Through互換設定を適用できません。公式スクリプトの構造が変更または部分適用されています: {before}"
                )));
            }
        }
    }
    Ok(patched)
}

fn format_process_failure(code: i32, stdout: &str, stderr: &str) -> String {
    let explanation = if code == -1_073_741_819 {
        "Windowsのネイティブアクセス違反が発生しました。Python/PyTorchのネイティブモジュールが異常終了しています。"
    } else {
        "See-Throughの処理に失敗しました。"
    };
    let combined = format!("{stdout}\n{stderr}");
    let tail = combined
        .lines()
        .filter(|line| !line.trim().is_empty())
        .rev()
        .take(20)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("\n");
    let tail = tail
        .chars()
        .rev()
        .take(4_000)
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>();
    format!("{explanation}\n終了コード: {code}\n\n直前のログ:\n{tail}")
}

/// 改行(\n)だけでなくtqdm等が使うカーソル復帰(\r)でも区切って読む。
/// BufRead::lines()は\nでしか区切らないため、tqdmのダウンロード進捗（\rでその場更新）は
/// 完了/中断までまとめて1行として溜め込まれ、ユーザーからは「何も表示されないまま
/// 何十分も止まっている」ように見えてしまっていた（実際は進んでいる場合も含めて）。
/// \r単位で都度flushすることで、ダウンロードが本当に進んでいるかをリアルタイムに可視化する
fn read_process_lines<R: std::io::Read>(reader: R, app: &AppHandle, stage: &str) -> String {
    let mut collected = String::new();
    let mut current: Vec<u8> = Vec::new();
    let mut last_reported = String::new();
    let mut flush = |current: &mut Vec<u8>| {
        if current.is_empty() {
            return;
        }
        let text = String::from_utf8_lossy(current).trim().to_string();
        current.clear();
        if text.is_empty() || text == last_reported {
            return;
        }
        eprintln!("[PachiPakuGen] See-Through: {text}");
        collected.push_str(&text);
        collected.push('\n');
        emit_progress(app, stage, progress_from_line(&text), &text);
        last_reported = text;
    };
    for byte in BufReader::new(reader).bytes().map_while(Result::ok) {
        if byte == b'\n' || byte == b'\r' {
            flush(&mut current);
        } else {
            current.push(byte);
        }
    }
    flush(&mut current);
    collected
}

fn progress_from_line(line: &str) -> u32 {
    let lower = line.to_lowercase();
    if lower.contains("running layerdiff") || lower.contains("layerdiff pipeline") {
        15
    } else if lower.contains("running marigold") || lower.contains("marigold pipeline") {
        65
    } else if lower.contains("psd") || lower.contains("further") {
        85
    } else {
        10
    }
}

fn emit_progress(app: &AppHandle, stage: &str, percent: u32, message: &str) {
    let _ = app.emit(
        "see-through-progress",
        SeeThroughProgressPayload {
            stage: stage.to_string(),
            percent,
            message: message.to_string(),
        },
    );
}

fn detect_gpus() -> Vec<GpuInfo> {
    let Ok(output) = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| {
            let mut fields = line.split(',').map(str::trim);
            Some(GpuInfo {
                index: fields.next()?.parse().ok()?,
                uuid: fields.next()?.to_string(),
                name: fields.next()?.to_string(),
                memory_mb: fields.next()?.parse().ok()?,
            })
        })
        .collect()
}

/// 実行に使うGPUの解決: ユーザー選択（see_through_gpu_index）を優先し、
/// 未選択または該当なしなら最大VRAMのGPUへフォールバック
fn resolve_gpu(app: &AppHandle) -> Option<GpuInfo> {
    let preferred = *app
        .state::<AppState>()
        .see_through_gpu_index
        .lock()
        .unwrap();
    let gpus = detect_gpus();
    if let Some(index) = preferred {
        if let Some(gpu) = gpus.iter().find(|gpu| gpu.index == index) {
            return Some(gpu.clone());
        }
    }
    gpus.into_iter().max_by_key(|gpu| gpu.memory_mb)
}

/// autoプロファイルで standard を推奨するVRAMしきい値（MB）。
/// 16GB以上あれば非量子化(standard)が収まり、bitsandbytesの4bitカーネル非互換
/// （新GPUでのクラッシュ）を根本的に避けられる
const STANDARD_PROFILE_VRAM_THRESHOLD_MB: u32 = 16_000;

fn select_profile(requested: &str, gpu: Option<&GpuInfo>) -> String {
    match requested {
        "standard" => return "standard".into(),
        "group-offload" => return "group-offload".into(),
        "low-vram" => return "low-vram".into(),
        _ => {}
    }
    // auto: VRAMに余裕があれば量子化(bitsandbytes)を使わない standard を選ぶ。
    // 新しいGPU（Blackwell等）ではbitsandbytesの4bitカーネルが未対応でクラッシュするため、
    // VRAMが足りるならstandardの方が安全かつ高速
    match gpu {
        Some(g) if g.memory_mb >= STANDARD_PROFILE_VRAM_THRESHOLD_MB => "standard".into(),
        _ => "low-vram".into(),
    }
}

fn git_commit(repo: &Path) -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(repo)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn copy_dir_recursive(source: &Path, destination: &Path) -> Result<(), AppError> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let from = entry.path();
        let to = destination.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_recursive(&from, &to)?;
        } else {
            fs::copy(from, to)?;
        }
    }
    Ok(())
}

fn newest_file_with_extension(root: &Path, extension: &str) -> Result<Option<PathBuf>, AppError> {
    fn visit(
        directory: &Path,
        extension: &str,
        newest: &mut Option<(SystemTime, PathBuf)>,
    ) -> Result<(), std::io::Error> {
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            let path = entry.path();
            if entry.file_type()?.is_dir() {
                visit(&path, extension, newest)?;
            } else if path
                .extension()
                .and_then(|value| value.to_str())
                .map_or(false, |value| value.eq_ignore_ascii_case(extension))
            {
                let modified = entry.metadata()?.modified().unwrap_or(UNIX_EPOCH);
                if newest
                    .as_ref()
                    .map_or(true, |(current, _)| modified > *current)
                {
                    *newest = Some((modified, path));
                }
            }
        }
        Ok(())
    }

    let mut newest = None;
    visit(root, extension, &mut newest)?;
    Ok(newest.map(|(_, path)| path))
}

fn primary_psd(output_dir: &Path) -> Result<Option<PathBuf>, AppError> {
    let expected = output_dir.join("source.psd");
    if expected.is_file() {
        return Ok(Some(expected));
    }
    Ok(
        newest_file_with_extension(output_dir, "psd")?.filter(|path| {
            !path
                .file_stem()
                .and_then(|value| value.to_str())
                .map_or(false, |value| value.ends_with("_depth"))
        }),
    )
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn unix_timestamp_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn standard_script_fixture() -> &'static str {
        "\
import os.path as osp
import os
import torch
from utils import inference_utils

for srcp in imglist:
        print('running layerdiff...')
        apply_layerdiff(srcp, args.repo_id_layerdiff)

        print('running marigold...')
        apply_marigold(srcp, args.repo_id_depth)

        srcname = osp.basename(osp.splitext(srcp)[0])
        further_extr(saved)
"
    }

    fn quantized_marigold_fixture() -> &'static str {
        r#"        pipeline.enable_group_offload('cuda', num_blocks_per_group=1)
        marigold_pipe.enable_group_offload('cuda', num_blocks_per_group=1)
        if args.cpu_offload:
            # VAE + TransparentVAE to bf16; quantized components handled by bnb
            pipeline.vae.to(dtype=torch.bfloat16)
            pipeline.trans_vae.to(dtype=torch.bfloat16)
            pipeline.enable_model_cpu_offload()
            # Don't manually .to(cuda) quantized components -- bnb handles device placement
            if getattr(args, 'group_offload', False):
                pipeline.enable_group_offload('cuda', num_blocks_per_group=1)
        marigold_pipe.vae.to(device='cuda')
        marigold_pipe.unet.to(device='cuda')
        # Text encoder may be quantized (from pre-quantized repo) — only move device, not dtype
        if not getattr(marigold_pipe.text_encoder, 'is_quantized', False) and \
           not getattr(marigold_pipe.text_encoder, 'quantization_method', None):
            marigold_pipe.text_encoder.to(device='cuda')
        if getattr(args, 'group_offload', False):
            marigold_pipe.enable_group_offload('cuda', num_blocks_per_group=1)
        marigold_pipe.cache_tag_embeds()
"#
    }

    fn gpu(memory_mb: u32) -> GpuInfo {
        GpuInfo {
            index: 0,
            uuid: "GPU-test".into(),
            name: "test".into(),
            memory_mb,
        }
    }

    fn model_test_root(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "pachipakugen-model-readiness-{name}-{}-{}",
            std::process::id(),
            unix_timestamp_millis()
        ))
    }

    fn test_model_requirements(profile: &str) -> Vec<SeeThroughModelRepositoryRequirement> {
        vec![
            SeeThroughModelRepositoryRequirement {
                repo_id: format!("test/{profile}-model"),
                revision: "a".repeat(40),
                files: vec![
                    SeeThroughRequiredModelFile {
                        path: "unet/config.json".into(),
                        size: 3,
                    },
                    SeeThroughRequiredModelFile {
                        path: "unet/model.safetensors".into(),
                        size: 7,
                    },
                ],
            },
            SeeThroughModelRepositoryRequirement {
                repo_id: "test/shared-scheduler".into(),
                revision: "b".repeat(40),
                files: vec![SeeThroughRequiredModelFile {
                    path: "scheduler/scheduler_config.json".into(),
                    size: 5,
                }],
            },
        ]
    }

    fn write_complete_test_model_manifest(
        root: &Path,
        profile: &str,
        requirements: &[SeeThroughModelRepositoryRequirement],
    ) -> Vec<PathBuf> {
        let profile_key = model_profile_key(profile);
        let mut manifest_files = Vec::new();
        let mut full_paths = Vec::new();
        for repository in requirements {
            for file in &repository.files {
                let relative = model_snapshot_manifest_path(repository, &file.path);
                let full_path = root.join("huggingface").join(&relative);
                fs::create_dir_all(full_path.parent().unwrap()).unwrap();
                fs::write(&full_path, vec![b'x'; file.size as usize]).unwrap();
                manifest_files.push(serde_json::json!({
                    "path": relative,
                    "size": file.size,
                }));
                full_paths.push(full_path);
            }
            let ref_relative = model_ref_manifest_path(repository);
            let ref_path = root.join("huggingface").join(&ref_relative);
            fs::create_dir_all(ref_path.parent().unwrap()).unwrap();
            fs::write(&ref_path, &repository.revision).unwrap();
            manifest_files.push(serde_json::json!({
                "path": ref_relative,
                "size": repository.revision.len(),
            }));
        }

        let repositories = requirements
            .iter()
            .map(|repository| repository.repo_id.clone())
            .collect::<Vec<_>>();
        let revisions = requirements
            .iter()
            .map(|repository| (repository.repo_id.clone(), repository.revision.clone()))
            .collect::<HashMap<_, _>>();
        let manifest = serde_json::json!({
            "schemaVersion": MODEL_MANIFEST_SCHEMA_VERSION,
            "profile": profile_key,
            "repositories": repositories,
            "revisions": revisions,
            "files": manifest_files,
        });
        let manifest_path = model_manifest_path(root, profile_key);
        fs::create_dir_all(manifest_path.parent().unwrap()).unwrap();
        fs::write(manifest_path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
        full_paths
    }

    fn edit_test_manifest(root: &Path, profile: &str, edit: impl FnOnce(&mut serde_json::Value)) {
        let manifest_path = model_manifest_path(root, profile);
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        edit(&mut manifest);
        fs::write(manifest_path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
    }

    fn create_arbitrary_manifest_file(root: &Path, relative: &str, size: usize) {
        let full_path = root.join("huggingface").join(relative);
        fs::create_dir_all(full_path.parent().unwrap()).unwrap();
        fs::write(full_path, vec![b'z'; size]).unwrap();
    }

    fn assert_incomplete_contains(readiness: ModelReadiness, expected: &str) {
        match readiness {
            ModelReadiness::Incomplete(reason) => assert!(
                reason.contains(expected),
                "expected `{expected}` in `{reason}`"
            ),
            other => panic!("expected incomplete model status, got {other:?}"),
        }
    }

    #[test]
    fn complete_model_manifest_is_ready() {
        let root = model_test_root("complete");
        let requirements = test_model_requirements("standard");
        write_complete_test_model_manifest(&root, "standard", &requirements);
        assert_eq!(
            validate_model_manifest_against(&root, "standard", &requirements),
            ModelReadiness::Ready
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn model_download_assets_are_materialized_without_credentials() {
        let root = model_test_root("download-assets");
        let (downloader, requirements, wrapper) = materialize_model_download_assets(&root).unwrap();
        assert_eq!(
            fs::read_to_string(downloader).unwrap(),
            MODEL_PREFETCH_SCRIPT
        );
        assert_eq!(
            fs::read_to_string(requirements).unwrap(),
            MODEL_REQUIREMENTS_JSON
        );
        let wrapper_text = fs::read_to_string(wrapper).unwrap();
        assert_eq!(wrapper_text, MODEL_DOWNLOAD_WRAPPER);
        assert!(wrapper_text.contains("HF_HUB_DISABLE_XET"));
        assert!(!wrapper_text.contains("HF_TOKEN"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn missing_model_manifest_is_not_ready() {
        let root = model_test_root("missing-manifest");
        assert_eq!(
            validate_model_manifest(&root, "low-vram"),
            ModelReadiness::Missing
        );
    }

    #[test]
    fn missing_model_file_is_reported() {
        let root = model_test_root("missing-file");
        let requirements = test_model_requirements("standard");
        let files = write_complete_test_model_manifest(&root, "standard", &requirements);
        fs::remove_file(&files[0]).unwrap();
        assert_incomplete_contains(
            validate_model_manifest_against(&root, "standard", &requirements),
            "モデルファイルが見つかりません",
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn model_file_size_mismatch_is_reported() {
        let root = model_test_root("size-mismatch");
        let requirements = test_model_requirements("standard");
        let files = write_complete_test_model_manifest(&root, "standard", &requirements);
        fs::write(&files[0], b"wrong-size").unwrap();
        assert_incomplete_contains(
            validate_model_manifest_against(&root, "standard", &requirements),
            "サイズが一致しません",
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn incomplete_download_is_reported_without_manifest() {
        let root = model_test_root("incomplete");
        let requirements = model_repository_requirements("standard").unwrap();
        let repo_id = &requirements[0].repo_id;
        let incomplete = root
            .join("huggingface/hub")
            .join(model_cache_directory_name(repo_id))
            .join("blobs/model.safetensors.incomplete");
        fs::create_dir_all(incomplete.parent().unwrap()).unwrap();
        fs::write(&incomplete, b"partial").unwrap();
        assert_incomplete_contains(validate_model_manifest(&root, "standard"), ".incomplete");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn incomplete_download_for_other_profile_does_not_block_selected_profile() {
        let root = model_test_root("other-profile-incomplete");
        let requirements = model_repository_requirements("standard").unwrap();
        let standard_repo = &requirements[0].repo_id;
        let incomplete = root
            .join("huggingface/hub")
            .join(model_cache_directory_name(standard_repo))
            .join("blobs/model.safetensors.incomplete");
        fs::create_dir_all(incomplete.parent().unwrap()).unwrap();
        fs::write(&incomplete, b"partial").unwrap();
        assert_eq!(
            validate_model_manifest(&root, "low-vram"),
            ModelReadiness::Missing
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn model_manifest_revision_must_match_pinned_revision() {
        let root = model_test_root("revision");
        let requirements = test_model_requirements("low-vram");
        write_complete_test_model_manifest(&root, "low-vram", &requirements);
        let repo_id = requirements[0].repo_id.clone();
        edit_test_manifest(&root, "low-vram", |manifest| {
            manifest["revisions"][&repo_id] = serde_json::Value::String("stale".into());
        });
        assert_incomplete_contains(
            validate_model_manifest_against(&root, "low-vram", &requirements),
            "固定リビジョンが一致しません",
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn installed_profile_does_not_mark_other_profile_ready() {
        let root = model_test_root("profile-separation");
        let requirements = test_model_requirements("standard");
        write_complete_test_model_manifest(&root, "standard", &requirements);
        assert_eq!(
            validate_model_manifest(&root, "low-vram"),
            ModelReadiness::Missing
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn arbitrary_manifest_paths_are_rejected() {
        let root = model_test_root("arbitrary-path");
        let requirements = test_model_requirements("standard");
        write_complete_test_model_manifest(&root, "standard", &requirements);
        create_arbitrary_manifest_file(&root, "hub/arbitrary/file.bin", 3);
        edit_test_manifest(&root, "standard", |manifest| {
            manifest["files"][0]["path"] =
                serde_json::Value::String("hub/arbitrary/file.bin".into());
        });
        assert_incomplete_contains(
            validate_model_manifest_against(&root, "standard", &requirements),
            "必須モデルファイルの記録が一致しません",
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn duplicate_manifest_paths_are_rejected() {
        let root = model_test_root("duplicate-path");
        let requirements = test_model_requirements("standard");
        write_complete_test_model_manifest(&root, "standard", &requirements);
        edit_test_manifest(&root, "standard", |manifest| {
            let duplicate = manifest["files"][0].clone();
            manifest["files"][1] = duplicate;
        });
        assert_incomplete_contains(
            validate_model_manifest_against(&root, "standard", &requirements),
            "重複ファイル",
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn main_revision_ref_must_match_pinned_revision() {
        let root = model_test_root("main-ref");
        let requirements = test_model_requirements("standard");
        write_complete_test_model_manifest(&root, "standard", &requirements);
        let ref_path = root
            .join("huggingface")
            .join(model_ref_manifest_path(&requirements[0]));
        fs::write(ref_path, "c".repeat(40)).unwrap();
        assert_incomplete_contains(
            validate_model_manifest_against(&root, "standard", &requirements),
            "revision参照が固定値と一致しません",
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn authoritative_model_requirements_have_expected_files_and_sizes() {
        for profile in ["low-vram", "standard"] {
            let requirements = model_repository_requirements(profile).unwrap();
            assert_eq!(requirements.len(), 3);
            assert_eq!(
                requirements
                    .iter()
                    .map(|repo| repo.files.len())
                    .sum::<usize>(),
                33
            );
            assert_eq!(expected_model_file_map(&requirements).unwrap().len(), 36);
        }
        let standard = model_repository_requirements("standard").unwrap();
        let unet = standard[0]
            .files
            .iter()
            .find(|file| file.path == "unet/diffusion_pytorch_model.safetensors")
            .unwrap();
        assert_eq!(unet.size, 8_142_919_672);
    }

    #[test]
    fn auto_profile_uses_standard_for_large_vram() {
        // 16GB以上は非量子化(standard)を選び、bitsandbytes 4bitカーネルの
        // 新GPU非互換クラッシュを根本的に避ける
        assert_eq!(select_profile("auto", Some(&gpu(24_000))), "standard");
        assert_eq!(select_profile("auto", Some(&gpu(16_000))), "standard");
    }

    #[test]
    fn auto_profile_uses_low_vram_for_mid_vram() {
        assert_eq!(select_profile("auto", Some(&gpu(12_000))), "low-vram");
    }

    #[test]
    fn auto_profile_uses_low_vram_without_supported_gpu() {
        assert_eq!(select_profile("auto", None), "low-vram");
        assert_eq!(select_profile("auto", Some(&gpu(8_000))), "low-vram");
    }

    #[test]
    fn native_crash_is_detected() {
        let err = AppError::General(
            "Windowsのネイティブアクセス違反が発生しました。\n終了コード: -1073741819".into(),
        );
        assert!(is_native_crash(&err));
        let ok = AppError::General("running layerdiff...".into());
        assert!(!is_native_crash(&ok));
    }

    #[test]
    fn requested_low_vram_overrides_gpu_size() {
        assert_eq!(select_profile("low-vram", Some(&gpu(24_000))), "low-vram");
    }

    #[test]
    fn requested_standard_overrides_safe_default() {
        assert_eq!(select_profile("standard", Some(&gpu(8_000))), "standard");
    }

    #[test]
    fn bf16_loading_patch_is_idempotent() {
        let source = "\
TransparentVAE.from_pretrained(pretrained, subfolder='trans_vae')
UNetFrameConditionModel.from_pretrained(pretrained, subfolder='unet')
UNetFrameConditionModel.from_pretrained(unet_ckpt)
scheduler=None
        )
MarigoldDepthPipeline.from_pretrained(pretrained, unet=unet)
layerdiff_pipeline.enable_group_offload('cuda', num_blocks_per_group=1)
marigold_pipeline.enable_group_offload('cuda', num_blocks_per_group=1)";
        let patched = patch_bf16_loading(source).unwrap();
        assert!(patched.contains("torch_dtype=torch.bfloat16"));
        assert!(patched.contains("exclude_modules=['text_encoder', 'text_encoder_2']"));
        assert!(patched.contains("exclude_modules=['text_encoder']"));
        assert_eq!(patch_bf16_loading(&patched).unwrap(), patched);

        let mixed =
            format!("{patched}\nTransparentVAE.from_pretrained(pretrained, subfolder='trans_vae')");
        assert!(patch_bf16_loading(&mixed).is_err());
    }

    #[test]
    fn standard_pipeline_cleanup_patch_matches_upstream_stage_boundaries() {
        let patched = patch_standard_pipeline_cleanup(standard_script_fixture()).unwrap();
        assert!(patched.contains("import gc\n"));
        assert_eq!(
            patched
                .matches("inference_utils.layerdiff_pipeline = None")
                .count(),
            1
        );
        assert_eq!(
            patched
                .matches("inference_utils.marigold_pipeline = None")
                .count(),
            1
        );
        assert_eq!(patched.matches("torch.cuda.empty_cache()").count(), 2);

        let layerdiff_call = patched.find("apply_layerdiff(").unwrap();
        let release_layerdiff = patched
            .find("inference_utils.layerdiff_pipeline = None")
            .unwrap();
        let marigold_start = patched.find("print('running marigold...')").unwrap();
        let marigold_call = patched.find("apply_marigold(").unwrap();
        let release_marigold = patched
            .find("inference_utils.marigold_pipeline = None")
            .unwrap();
        let psd_assembly = patched
            .find("srcname = osp.basename(osp.splitext(srcp)[0])")
            .unwrap();
        assert!(layerdiff_call < release_layerdiff);
        assert!(release_layerdiff < marigold_start);
        assert!(marigold_call < release_marigold);
        assert!(release_marigold < psd_assembly);
        assert_eq!(patch_standard_pipeline_cleanup(&patched).unwrap(), patched);
    }

    #[test]
    fn standard_pipeline_cleanup_patch_fails_closed_on_upstream_change() {
        let changed = "import os\napply_layerdiff(source)\napply_marigold(source)\n";
        let error = patch_standard_pipeline_cleanup(changed).unwrap_err();
        assert!(error.to_string().contains("公式スクリプトの構造が変更"));
    }

    #[test]
    fn low_vram_profile_does_not_require_standard_runtime_files() {
        let repo = std::env::temp_dir().join(format!(
            "pachipakugen-missing-standard-runtime-{}",
            unix_timestamp_millis()
        ));
        let quantized = repo.join("inference/scripts/inference_psd_quantized.py");
        fs::create_dir_all(quantized.parent().unwrap()).unwrap();
        fs::write(&quantized, quantized_marigold_fixture()).unwrap();
        let layerdiff_module =
            repo.join("common/modules/layerdiffuse/diffusers_kdiffusion_sdxl.py");
        fs::create_dir_all(layerdiff_module.parent().unwrap()).unwrap();
        fs::write(
            &layerdiff_module,
            "        device = self.text_encoder.device\n    ):\n\n        device = self.unet.device\n        dtype = self.unet.dtype\n                    group_index=group_index\n                )[0]\n",
        )
        .unwrap();
        let marigold_module = repo.join("common/modules/marigold/marigold_depth_pipeline.py");
        fs::create_dir_all(marigold_module.parent().unwrap()).unwrap();
        fs::write(
            &marigold_module,
            "        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)\n        is_3d = isinstance(self.unet, UNetFrameConditionModel)\n        device = self.device\n                noise_pred = self.unet(\n                    unet_input, t, encoder_hidden_states=batch_empty_text_embed\n                ).sample  # [B, 4, h, w]\n",
        )
        .unwrap();
        apply_runtime_compatibility_patches(&repo, false).unwrap();
        apply_runtime_compatibility_patches(&repo, false).unwrap();
        let patched = fs::read_to_string(quantized).unwrap();
        assert!(patched.contains("honor CPU offload for the NF4 Marigold stage"));
        assert!(patched.contains("pipeline.model_cpu_offload_seq = 'unet'"));
        assert!(patched.contains("pipeline._exclude_from_cpu_offload = ['vae', 'trans_vae']"));
        assert!(patched.contains("marigold_pipe.model_cpu_offload_seq = 'unet'"));
        assert!(patched.contains("marigold_pipe._exclude_from_cpu_offload = ['vae']"));
        assert!(patched.contains("exclude_modules=['unet', 'text_encoder', 'text_encoder_2']"));
        assert!(patched.contains("exclude_modules=['unet', 'text_encoder']"));
        assert!(fs::read_to_string(layerdiff_module)
            .unwrap()
            .contains("device = self._execution_device"));
        assert!(fs::read_to_string(marigold_module)
            .unwrap()
            .contains("input_ids.to(self._execution_device)"));
        assert!(!repo.join("common/utils/inference_utils.py").exists());
        fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn quantized_marigold_cpu_offload_patch_is_idempotent() {
        let patched = patch_quantized_marigold_cpu_offload(quantized_marigold_fixture()).unwrap();
        assert!(patched.contains("if args.cpu_offload:"));
        assert!(patched.contains("model_cpu_offload_seq = 'unet'"));
        assert!(patched.contains("_exclude_from_cpu_offload = ['vae']"));
        assert!(patched.contains("marigold_pipe.enable_model_cpu_offload()"));
        assert_eq!(
            patch_quantized_marigold_cpu_offload(&patched).unwrap(),
            patched
        );
    }

    #[test]
    fn cpu_offload_execution_device_patch_is_idempotent_and_fail_closed() {
        let before = "        device = self.text_encoder.device";
        let after = "        # compat\n        device = self._execution_device";
        let patched =
            patch_cpu_offload_execution_device(before, before, after, "LayerDiff").unwrap();
        assert_eq!(patched, after);
        assert_eq!(
            patch_cpu_offload_execution_device(&patched, before, after, "LayerDiff").unwrap(),
            patched
        );
        assert!(patch_cpu_offload_execution_device(
            "        device = something_else",
            before,
            after,
            "LayerDiff"
        )
        .is_err());
    }

    #[test]
    fn quantized_group_offload_patch_excludes_disposable_text_encoders() {
        let cpu_patched =
            patch_quantized_marigold_cpu_offload(quantized_marigold_fixture()).unwrap();
        assert!(
            cpu_patched.contains("pipeline.enable_group_offload('cuda', num_blocks_per_group=1)")
        );
        let patched = patch_quantized_group_offload(&cpu_patched).unwrap();
        assert!(patched.contains("exclude_modules=['unet', 'text_encoder', 'text_encoder_2']"));
        assert!(patched.contains("exclude_modules=['unet', 'text_encoder']"));
        assert_eq!(patch_quantized_group_offload(&patched).unwrap(), patched);
        let mixed =
            format!("{patched}\npipeline.enable_group_offload('cuda', num_blocks_per_group=1)");
        assert!(patch_quantized_group_offload(&mixed).is_err());
    }

    #[test]
    fn standard_detail_options_map_to_standard_script_arguments() {
        let options = SeeThroughOptions {
            seed: Some(314_159),
            resolution: Some(768),
            resolution_depth: Some(512),
            inference_steps: Some(4),
            inference_steps_depth: Some(2),
            group_offload: Some("on".into()),
            cpu_offload: Some("on".into()),
        };
        let mut args = Vec::new();
        append_inference_options(&mut args, "standard", Some(&options)).unwrap();
        assert_eq!(
            args,
            vec![
                "--seed",
                "314159",
                "--resolution",
                "768",
                "--resolution_depth",
                "512",
                "--inference_steps",
                "4",
                "--inference_steps_depth",
                "2",
                "--group_offload",
            ]
        );
        assert!(!args.iter().any(|arg| arg.contains("cpu_offload")));
    }

    #[test]
    fn low_vram_detail_options_use_quantized_arguments_and_cpu_precedence() {
        let options = SeeThroughOptions {
            seed: Some(54_321),
            resolution: Some(512),
            resolution_depth: Some(-1),
            inference_steps: Some(2),
            inference_steps_depth: Some(99),
            group_offload: Some("on".into()),
            cpu_offload: Some("on".into()),
        };
        let mut args = Vec::new();
        append_inference_options(&mut args, "low-vram", Some(&options)).unwrap();
        assert_eq!(
            args,
            vec![
                "--seed",
                "54321",
                "--resolution",
                "512",
                "--resolution_depth",
                "-1",
                "--num_inference_steps",
                "2",
                "--cpu_offload",
                "--no_group_offload",
            ]
        );
        assert!(!args.iter().any(|arg| arg == "--inference_steps_depth"));
    }

    #[test]
    fn low_vram_off_modes_are_forwarded_without_conflicting_flags() {
        let options = SeeThroughOptions {
            group_offload: Some("off".into()),
            cpu_offload: Some("off".into()),
            ..Default::default()
        };
        let mut args = Vec::new();
        append_inference_options(&mut args, "low-vram", Some(&options)).unwrap();
        assert_eq!(args, vec!["--no_group_offload", "--no_cpu_offload"]);
    }

    #[test]
    fn inference_detail_options_reject_invalid_manual_values() {
        for (field, options) in [
            (
                "resolution",
                SeeThroughOptions {
                    resolution: Some(257),
                    ..Default::default()
                },
            ),
            (
                "resolution_depth",
                SeeThroughOptions {
                    resolution_depth: Some(0),
                    ..Default::default()
                },
            ),
            (
                "inference_steps_depth",
                SeeThroughOptions {
                    inference_steps_depth: Some(0),
                    ..Default::default()
                },
            ),
        ] {
            let mut args = Vec::new();
            let error =
                append_inference_options(&mut args, "standard", Some(&options)).unwrap_err();
            assert!(error.to_string().contains(field));
        }

        let mut args = Vec::new();
        let invalid_mode = SeeThroughOptions {
            group_offload: Some("sometimes".into()),
            ..Default::default()
        };
        assert!(append_inference_options(&mut args, "low-vram", Some(&invalid_mode)).is_err());
    }

    #[test]
    fn standard_profile_fails_closed_when_bf16_patch_cannot_be_applied() {
        let repo = model_test_root("bf16-fail-closed");
        let inference_utils = repo.join("common/utils/inference_utils.py");
        fs::create_dir_all(inference_utils.parent().unwrap()).unwrap();
        fs::write(&inference_utils, "# incompatible upstream structure\n").unwrap();
        let error = apply_runtime_compatibility_patches(&repo, true).unwrap_err();
        assert!(error.to_string().contains("BF16互換設定を適用できません"));
        fs::remove_dir_all(repo).unwrap();
    }

    #[test]
    fn standard_pipeline_cleanup_patch_rejects_partial_patch() {
        let partial = "\
import os
import gc
apply_layerdiff(source)
inference_utils.layerdiff_pipeline = None
        print('running marigold...')
apply_marigold(source)
        srcname = osp.basename(osp.splitext(srcp)[0])
";
        let error = patch_standard_pipeline_cleanup(partial).unwrap_err();
        assert!(error.to_string().contains("メモリ解放設定が不完全"));
    }

    #[test]
    fn standard_pipeline_cleanup_validation_rejects_missing_import() {
        let patched = patch_standard_pipeline_cleanup(standard_script_fixture()).unwrap();
        let missing_import = patched.replace("from utils import inference_utils\n", "");
        let error = validate_standard_pipeline_cleanup(&missing_import).unwrap_err();
        assert!(error.to_string().contains("import構造が不正"));
    }

    #[test]
    fn standard_pipeline_cleanup_validation_rejects_wrong_stage_order() {
        let patched = patch_standard_pipeline_cleanup(standard_script_fixture()).unwrap();
        let release = "        # PachiPakuGen: release LayerDiff before loading Marigold.\n        inference_utils.layerdiff_pipeline = None\n        gc.collect()\n        torch.cuda.empty_cache()\n\n";
        let without_release = patched.replacen(release, "", 1);
        let misplaced = without_release.replacen(
            "        print('running layerdiff...')",
            &format!("{release}        print('running layerdiff...')"),
            1,
        );
        let error = validate_standard_pipeline_cleanup(&misplaced).unwrap_err();
        assert!(error.to_string().contains("正しいステージ境界"));
    }

    #[test]
    fn native_access_violation_is_not_classified_as_cuda_oom() {
        let native = AppError::General(
            "Windowsのネイティブアクセス違反が発生しました。\n終了コード: -1073741819".into(),
        );
        assert!(is_native_crash(&native));
        assert!(!is_cuda_oom_failure(&native));

        let unsupported =
            AppError::General("cublasCreate failed with CUBLAS_STATUS_NOT_SUPPORTED".into());
        assert!(!is_cuda_oom_failure(&unsupported));
        let alloc = AppError::General("cublasCreate failed with CUBLAS_STATUS_ALLOC_FAILED".into());
        assert!(is_cuda_oom_failure(&alloc));
    }

    #[test]
    fn inference_recovery_plan_switches_profile_for_native_crashes() {
        let standard =
            inference_recovery_plan("standard", None, false, true).expect("standard recovery");
        assert_eq!(standard.0, "low-vram");

        let low_vram =
            inference_recovery_plan("low-vram", None, false, true).expect("low-vram recovery");
        assert_eq!(low_vram.0, "standard");
        assert!(!low_vram.2.contains("bitsandbytes"));
    }

    #[test]
    fn standard_profile_does_not_treat_bitsandbytes_text_as_quantized_execution() {
        let error = AppError::General(
            "loaded bitsandbytes module\ncublasCreate failed with CUBLAS_STATUS_NOT_SUPPORTED"
                .into(),
        );
        assert!(is_quantization_kernel_failure(&error));
        assert!(!should_attempt_inference_recovery(&error, "standard"));
        assert!(should_attempt_inference_recovery(&error, "low-vram"));
        assert!(!should_augment_inference_error(&error, "standard"));
        assert!(should_augment_inference_error(&error, "low-vram"));
    }

    #[test]
    fn inference_recovery_plan_distinguishes_oom_and_quantization_failure() {
        let standard_oom =
            inference_recovery_plan("standard", None, false, false).expect("standard OOM");
        assert_eq!(standard_oom.0, "low-vram");

        assert!(inference_recovery_plan("low-vram", None, false, false).is_none());
        let group_disabled = SeeThroughOptions {
            group_offload: Some("off".into()),
            ..Default::default()
        };
        let low_vram_oom = inference_recovery_plan("low-vram", Some(&group_disabled), false, false)
            .expect("low-vram group recovery");
        assert_eq!(low_vram_oom.0, "low-vram");
        let options = low_vram_oom.1.unwrap();
        assert_eq!(options.cpu_offload.as_deref(), Some("off"));
        assert_eq!(options.group_offload.as_deref(), Some("on"));

        let cpu_enabled = SeeThroughOptions {
            cpu_offload: Some("on".into()),
            group_offload: Some("default".into()),
            ..Default::default()
        };
        let cpu_recovery = inference_recovery_plan("low-vram", Some(&cpu_enabled), false, false)
            .expect("low-vram CPU recovery");
        let cpu_recovery_options = cpu_recovery.1.unwrap();
        assert_eq!(cpu_recovery_options.cpu_offload.as_deref(), Some("off"));
        assert_eq!(cpu_recovery_options.group_offload.as_deref(), Some("on"));

        let quantization =
            inference_recovery_plan("low-vram", None, true, false).expect("quantization recovery");
        assert_eq!(quantization.0, "standard");

        let group_enabled = SeeThroughOptions {
            group_offload: Some("on".into()),
            ..Default::default()
        };
        assert!(inference_recovery_plan("low-vram", Some(&group_enabled), false, false).is_none());
    }

    #[test]
    fn native_access_violation_hint_does_not_claim_vram_shortage() {
        let error = AppError::General(
            "Windowsのネイティブアクセス違反が発生しました。\n終了コード: -1073741819".into(),
        );
        let message = augment_inference_error(error, false).to_string();
        assert!(message.contains("VRAM不足とは判定できません"));
        assert!(!message.contains("VRAM不足（CUDA/cuBLAS"));
    }

    #[test]
    fn access_violation_has_readable_error() {
        let message = format_process_failure(-1_073_741_819, "running layerdiff", "warning");
        assert!(message.contains("ネイティブアクセス違反"));
        assert!(message.contains("終了コード: -1073741819"));
        assert!(!message.contains("モデル読込時のメモリ不足"));
    }

    #[test]
    fn primary_psd_prefers_source_over_newer_depth_psd() {
        let root = std::env::temp_dir().join(format!(
            "pachipakugen-primary-psd-{}",
            unix_timestamp_millis()
        ));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("source.psd"), b"main").unwrap();
        fs::write(root.join("source_depth.psd"), b"depth").unwrap();
        assert_eq!(primary_psd(&root).unwrap(), Some(root.join("source.psd")));
        fs::remove_dir_all(root).unwrap();
    }
}
