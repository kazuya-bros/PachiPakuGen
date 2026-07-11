use crate::commands::parts::{
    cache_original_image_for_canvas, get_mapping_preview_inner, load_slot_inner,
    MappingPreviewResult, SlotLoadResult,
};
use crate::error::AppError;
use crate::processing::image_utils;
use crate::state::AppState;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter, Manager};

const SEE_THROUGH_REPO: &str = "https://github.com/shitagaki-lab/see-through.git";
const SEE_THROUGH_COMMIT: &str = "e4cb250dc69defe6f982168dab684aa461552b5b";

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
    pub busy: bool,
    pub runtime_root: String,
    pub repo_path: String,
    pub python_path: String,
    pub pinned_commit: String,
    pub installed_commit: Option<String>,
    pub gpu_index: Option<u32>,
    pub gpu_name: Option<String>,
    pub gpu_memory_mb: Option<u32>,
    pub recommended_profile: String,
    pub message: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SeeThroughRunResult {
    pub psd_path: String,
    pub output_dir: String,
    pub selected_profile: String,
    pub slot_load: SlotLoadResult,
    pub mapping_preview: MappingPreviewResult,
    /// 左右パーツ分解に失敗し、左右分解なしで自動リトライした場合にtrue（UIで報告する）
    pub split_parts_fallback: bool,
}

#[derive(Clone, Debug, Default, Deserialize)]
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
pub fn set_see_through_gpu(
    app: AppHandle,
    gpu_index: Option<u32>,
) -> Result<(), AppError> {
    *app.state::<AppState>().see_through_gpu_index.lock().unwrap() = gpu_index;
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
) -> Result<SeeThroughRuntimeStatus, AppError> {
    tauri::async_runtime::spawn_blocking(move || runtime_status(&app))
        .await
        .map_err(|error| AppError::General(format!("See-Through状態確認に失敗: {error}")))?
}

#[tauri::command]
pub async fn prepare_see_through_runtime(
    app: AppHandle,
) -> Result<SeeThroughRuntimeStatus, AppError> {
    tauri::async_runtime::spawn_blocking(move || prepare_runtime(&app))
        .await
        .map_err(|error| AppError::General(format!("See-Throughセットアップ処理に失敗: {error}")))?
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

fn runtime_status(app: &AppHandle) -> Result<SeeThroughRuntimeStatus, AppError> {
    let root = runtime_root(app)?;
    let repo = root.join("repo");
    let python = venv_python(&root);
    let marker = root.join("setup-complete.json");
    let installed_commit = git_commit(&repo);
    let gpu = resolve_gpu(app);
    let recommended_profile = select_profile("auto", gpu.as_ref());
    let busy = app
        .state::<AppState>()
        .see_through_pid
        .lock()
        .unwrap()
        .is_some();
    let ready = marker.is_file()
        && repo.join("inference/scripts/inference_psd.py").is_file()
        && python.is_file()
        && installed_commit.as_deref() == Some(SEE_THROUGH_COMMIT);

    let message = if ready {
        "See-Throughランタイムは使用できます".to_string()
    } else if repo.is_dir() || python.is_file() {
        "See-Throughランタイムのセットアップが未完了です".to_string()
    } else {
        "初回セットアップが必要です。Python環境とモデルはアプリ管理領域へ保存されます".to_string()
    };

    Ok(SeeThroughRuntimeStatus {
        ready,
        busy,
        runtime_root: root.to_string_lossy().into_owned(),
        repo_path: repo.to_string_lossy().into_owned(),
        python_path: python.to_string_lossy().into_owned(),
        pinned_commit: SEE_THROUGH_COMMIT.to_string(),
        installed_commit,
        gpu_index: gpu.as_ref().map(|gpu| gpu.index),
        gpu_name: gpu.as_ref().map(|gpu| gpu.name.clone()),
        gpu_memory_mb: gpu.as_ref().map(|gpu| gpu.memory_mb),
        recommended_profile,
        message,
    })
}

fn prepare_runtime(app: &AppHandle) -> Result<SeeThroughRuntimeStatus, AppError> {
    let root = runtime_root(app)?;
    let repo = root.join("repo");
    let python = venv_python(&root);
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
    apply_bf16_loading_compatibility_patch(&repo)?;

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
    let gpu = best_gpu();
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

    emit_progress(
        app,
        "ready",
        100,
        "See-Throughの初回セットアップが完了しました",
    );
    runtime_status(app)
}

pub(crate) fn run_inference(
    app: &AppHandle,
    source_path: &str,
    requested_profile: &str,
    split_parts: bool,
    options: Option<SeeThroughOptions>,
) -> Result<SeeThroughRunResult, AppError> {
    let source = Path::new(source_path);
    if !source.is_file() {
        return Err(AppError::General(format!(
            "元画像が見つかりません: {}",
            source.display()
        )));
    }

    let status = runtime_status(app)?;
    if !status.ready {
        return Err(AppError::General(
            "See-Throughの初回セットアップを先に実行してください".into(),
        ));
    }

    let root = PathBuf::from(&status.runtime_root);
    let repo = PathBuf::from(&status.repo_path);
    apply_bf16_loading_compatibility_patch(&repo)?;
    let python = PathBuf::from(&status.python_path);
    let gpu = resolve_gpu(app);
    let selected_profile = select_profile(requested_profile, gpu.as_ref());
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
            let mut result = run_inference(app, source_path, requested_profile, false, options)?;
            result.split_parts_fallback = true;
            return Ok(result);
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
        slot_load,
        mapping_preview,
        split_parts_fallback: false,
    })
}

/// 左右分割（--tblr_split）由来の失敗か。inference_utils.pyのlr_split系トレース
/// バックが「直前のログ」としてエラーメッセージに含まれることを利用する
fn is_lr_split_failure(error: &AppError) -> bool {
    let text = error.to_string();
    text.contains("lr_split") || text.contains("further_extr")
}

fn append_inference_options(
    args: &mut Vec<String>,
    selected_profile: &str,
    options: Option<&SeeThroughOptions>,
) -> Result<(), AppError> {
    let Some(options) = options else {
        return Ok(());
    };

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

    match option_mode(options.group_offload.as_deref())? {
        "on" => args.push("--group_offload".into()),
        "off" if selected_profile == "low-vram" => args.push("--no_group_offload".into()),
        _ => {}
    }

    if selected_profile == "low-vram" {
        match option_mode(options.cpu_offload.as_deref())? {
            "on" => args.push("--cpu_offload".into()),
            "off" => args.push("--no_cpu_offload".into()),
            _ => {}
        }
    }

    Ok(())
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

fn runtime_root(app: &AppHandle) -> Result<PathBuf, AppError> {
    if let Ok(path) = std::env::var("PACHIPAKUGEN_SEE_THROUGH_ROOT") {
        if !path.trim().is_empty() {
            return Ok(PathBuf::from(path));
        }
    }
    app.path()
        .app_local_data_dir()
        .map(|path| path.join("see-through"))
        .map_err(|error| AppError::General(format!("アプリデータ保存先を取得できません: {error}")))
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
    command
        .args(args)
        .current_dir(cwd)
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        .env("CUDA_MODULE_LOADING", "LAZY")
        .env("HF_HOME", hf_home)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
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
    let process_status = child.wait()?;
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

fn apply_bf16_loading_compatibility_patch(repo: &Path) -> Result<(), AppError> {
    let path = repo.join("common/utils/inference_utils.py");
    let original = fs::read_to_string(&path).map_err(|error| {
        AppError::General(format!(
            "See-Through互換設定の読み込みに失敗しました: {} ({error})",
            path.display()
        ))
    })?;
    let patched = match patch_bf16_loading(&original) {
        Ok(patched) => patched,
        Err(error) => {
            eprintln!("[PachiPakuGen] See-Through compatibility patch skipped: {error}");
            return Ok(());
        }
    };
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
    ];
    let mut patched = source.to_string();
    for (before, after) in replacements {
        if patched.contains(after) {
            continue;
        }
        if !patched.contains(before) {
            return Err(AppError::General(format!(
                "See-Through互換設定を適用できません。公式スクリプトの構造が変更されています: {before}"
            )));
        }
        patched = patched.replace(before, after);
    }
    Ok(patched)
}

fn format_process_failure(code: i32, stdout: &str, stderr: &str) -> String {
    let explanation = if code == -1_073_741_819 {
        "Windowsのネイティブアクセス違反が発生しました。モデル読込時のメモリ不足または互換性問題の可能性があります。"
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

fn read_process_lines<R: std::io::Read>(reader: R, app: &AppHandle, stage: &str) -> String {
    let mut collected = String::new();
    for line in BufReader::new(reader).lines().map_while(Result::ok) {
        eprintln!("[PachiPakuGen] See-Through: {line}");
        collected.push_str(&line);
        collected.push('\n');
        emit_progress(app, stage, progress_from_line(&line), &line);
    }
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

fn best_gpu() -> Option<GpuInfo> {
    detect_gpus().into_iter().max_by_key(|gpu| gpu.memory_mb)
}

/// 実行に使うGPUの解決: ユーザー選択（see_through_gpu_index）を優先し、
/// 未選択または該当なしなら最大VRAMのGPUへフォールバック
fn resolve_gpu(app: &AppHandle) -> Option<GpuInfo> {
    let preferred = *app.state::<AppState>().see_through_gpu_index.lock().unwrap();
    let gpus = detect_gpus();
    if let Some(index) = preferred {
        if let Some(gpu) = gpus.iter().find(|gpu| gpu.index == index) {
            return Some(gpu.clone());
        }
    }
    gpus.into_iter().max_by_key(|gpu| gpu.memory_mb)
}

fn select_profile(requested: &str, gpu: Option<&GpuInfo>) -> String {
    match requested {
        "standard" => return "standard".into(),
        "group-offload" => return "group-offload".into(),
        "low-vram" => return "low-vram".into(),
        _ => {}
    }
    let _ = gpu;
    "low-vram".into()
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

    fn gpu(memory_mb: u32) -> GpuInfo {
        GpuInfo {
            index: 0,
            uuid: "GPU-test".into(),
            name: "test".into(),
            memory_mb,
        }
    }

    #[test]
    fn auto_profile_uses_low_vram_for_large_vram() {
        assert_eq!(select_profile("auto", Some(&gpu(24_000))), "low-vram");
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
MarigoldDepthPipeline.from_pretrained(pretrained, unet=unet)";
        let patched = patch_bf16_loading(source).unwrap();
        assert!(patched.contains("torch_dtype=torch.bfloat16"));
        assert_eq!(patch_bf16_loading(&patched).unwrap(), patched);
    }

    #[test]
    fn access_violation_has_readable_error() {
        let message = format_process_failure(-1_073_741_819, "running layerdiff", "warning");
        assert!(message.contains("ネイティブアクセス違反"));
        assert!(message.contains("終了コード: -1073741819"));
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
