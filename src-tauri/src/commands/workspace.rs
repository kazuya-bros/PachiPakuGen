use crate::error::AppError;
use image::GenericImageView;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const PROJECT_FILE: &str = "project.json";
const CODEX_REQUEST_DIR: &str = "01_codex_request";
const GENERATED_PARTS_DIR: &str = "02_generated_parts";
const SEE_THROUGH_DIR: &str = "03_see_through";
const SPRITALK_PARTS_DIR: &str = "04_spritalk_parts";
const WORKSPACE_TARGETS: &[&str] = &[
    "mouth-closed",
    "mouth-a",
    "mouth-i",
    "mouth-u",
    "mouth-e",
    "mouth-o",
    "eyes-closed",
];

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkspaceProject {
    pub version: u32,
    pub created_at: u64,
    pub updated_at: u64,
    pub current_step: u32,
    pub source_image_path: Option<String>,
    pub reference_image_path: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExpressionWorkspaceResult {
    pub work_path: String,
    pub project_path: String,
    pub codex_request_path: String,
    pub generated_parts_path: String,
    pub see_through_path: String,
    pub spritalk_parts_path: String,
    pub project: WorkspaceProject,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PrepareWorkspaceCodexRequest {
    pub work_path: String,
    pub source_image_path: String,
    pub reference_image_path: Option<String>,
    pub prompt: String,
    pub mouth_corner: String,
    pub mouth_size: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkspaceGeneratedPartsStatus {
    pub request_path: String,
    pub handoff_path: String,
    pub generated_parts_path: String,
    pub expected_parts: Vec<String>,
    pub present_parts: Vec<String>,
    pub missing_parts: Vec<String>,
    pub size_mismatches: Vec<String>,
    pub ready: bool,
}

#[tauri::command]
pub async fn create_expression_workspace(
    work_path: String,
) -> Result<ExpressionWorkspaceResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || create_expression_workspace_inner(&work_path))
        .await
        .map_err(|error| AppError::General(format!("Task join error: {error}")))?
}

#[tauri::command]
pub async fn load_expression_workspace(
    work_path: String,
) -> Result<ExpressionWorkspaceResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || load_expression_workspace_inner(&work_path))
        .await
        .map_err(|error| AppError::General(format!("Task join error: {error}")))?
}

#[tauri::command]
pub async fn prepare_workspace_codex_request(
    request: PrepareWorkspaceCodexRequest,
) -> Result<WorkspaceGeneratedPartsStatus, AppError> {
    tauri::async_runtime::spawn_blocking(move || prepare_workspace_codex_request_inner(request))
        .await
        .map_err(|error| AppError::General(format!("Task join error: {error}")))?
}

#[tauri::command]
pub async fn inspect_workspace_generated_parts(
    work_path: String,
) -> Result<WorkspaceGeneratedPartsStatus, AppError> {
    tauri::async_runtime::spawn_blocking(move || {
        inspect_workspace_generated_parts_inner(&work_path)
    })
    .await
    .map_err(|error| AppError::General(format!("Task join error: {error}")))?
}

#[tauri::command]
pub async fn update_expression_workspace_step(
    work_path: String,
    current_step: u32,
) -> Result<ExpressionWorkspaceResult, AppError> {
    tauri::async_runtime::spawn_blocking(move || {
        update_expression_workspace_step_inner(&work_path, current_step)
    })
    .await
    .map_err(|error| AppError::General(format!("Task join error: {error}")))?
}

fn create_expression_workspace_inner(
    work_path: &str,
) -> Result<ExpressionWorkspaceResult, AppError> {
    let root = PathBuf::from(work_path);
    ensure_workspace_dirs(&root)?;
    let project_path = root.join(PROJECT_FILE);
    let now = unix_time();
    let project = if project_path.is_file() {
        read_project(&project_path)?
    } else {
        WorkspaceProject {
            version: 1,
            created_at: now,
            updated_at: now,
            current_step: 1,
            source_image_path: None,
            reference_image_path: None,
        }
    };
    write_project(&project_path, &project)?;
    workspace_result(root, project)
}

fn load_expression_workspace_inner(work_path: &str) -> Result<ExpressionWorkspaceResult, AppError> {
    let root = PathBuf::from(work_path);
    let project_path = root.join(PROJECT_FILE);
    if !project_path.is_file() {
        return Err(AppError::General(format!(
            "project.json が見つかりません: {}",
            project_path.display()
        )));
    }
    ensure_workspace_dirs(&root)?;
    let project = read_project(&project_path)?;
    workspace_result(root, project)
}

fn prepare_workspace_codex_request_inner(
    request: PrepareWorkspaceCodexRequest,
) -> Result<WorkspaceGeneratedPartsStatus, AppError> {
    let root = PathBuf::from(&request.work_path);
    ensure_workspace_dirs(&root)?;

    let source = PathBuf::from(&request.source_image_path);
    if !source.is_file() {
        return Err(AppError::General(format!(
            "元画像が見つかりません: {}",
            source.display()
        )));
    }

    let codex_dir = root.join(CODEX_REQUEST_DIR);
    let generated_dir = root.join(GENERATED_PARTS_DIR);
    let source_copy = codex_dir.join("source.png");
    image::open(&source)?.save(&source_copy)?;

    let reference_copy = request
        .reference_image_path
        .as_deref()
        .filter(|path| !path.trim().is_empty())
        .map(PathBuf::from)
        .filter(|path| path.is_file())
        .map(|path| {
            let dest = codex_dir.join("reference.png");
            image::open(path)?.save(&dest)?;
            Ok::<PathBuf, AppError>(dest)
        })
        .transpose()?;

    let expected_parts: Vec<String> = WORKSPACE_TARGETS
        .iter()
        .map(|part| part.to_string())
        .collect();
    let request_path = codex_dir.join("codex_request.md");
    let handoff_path = codex_dir.join("codex_handoff.md");
    fs::write(
        &request_path,
        workspace_codex_request_text(
            &source_copy,
            reference_copy.as_deref(),
            &generated_dir,
            &expected_parts,
            &request,
        ),
    )?;
    fs::write(
        &handoff_path,
        workspace_codex_handoff_text(&request_path, &generated_dir, &expected_parts),
    )?;
    fs::write(
        codex_dir.join("codex_manifest.json"),
        serde_json::to_vec_pretty(&json!({
            "formatVersion": 1,
            "mode": "workspace-codex-request",
            "source": source_copy.to_string_lossy(),
            "reference": reference_copy.as_ref().map(|path| path.to_string_lossy().into_owned()),
            "generatedPartsDirectory": generated_dir.to_string_lossy(),
            "expectedGeneratedParts": expected_parts,
            "mouthCorner": request.mouth_corner,
            "mouthSize": request.mouth_size,
            "prompt": request.prompt,
        }))
        .map_err(|error| AppError::General(format!("codex_manifest.json 作成失敗: {error}")))?,
    )?;

    let mut project = read_or_create_project(&root)?;
    project.updated_at = unix_time();
    project.current_step = project.current_step.max(2);
    project.source_image_path = Some(request.source_image_path);
    project.reference_image_path = request.reference_image_path;
    write_project(&root.join(PROJECT_FILE), &project)?;

    inspect_workspace_generated_parts_inner(&root.to_string_lossy())
}

fn inspect_workspace_generated_parts_inner(
    work_path: &str,
) -> Result<WorkspaceGeneratedPartsStatus, AppError> {
    let root = PathBuf::from(work_path);
    ensure_workspace_dirs(&root)?;
    let project = read_project(&root.join(PROJECT_FILE))?;
    let source_path = project
        .source_image_path
        .as_deref()
        .ok_or_else(|| AppError::General("先に立ち絵を選択してください".into()))?;
    let source = image::open(source_path)?;
    let expected_parts: Vec<String> = WORKSPACE_TARGETS
        .iter()
        .map(|part| part.to_string())
        .collect();
    let generated_dir = root.join(GENERATED_PARTS_DIR);
    let mut present_parts = Vec::new();
    let mut missing_parts = Vec::new();
    let mut size_mismatches = Vec::new();

    for part in &expected_parts {
        let path = generated_dir.join(format!("{part}.png"));
        if !path.is_file() {
            missing_parts.push(part.clone());
            continue;
        }
        present_parts.push(part.clone());
        match image::open(&path) {
            Ok(image) if image.dimensions() == source.dimensions() => {}
            Ok(image) => size_mismatches.push(format!(
                "{}.png: {}x{} (expected {}x{})",
                part,
                image.width(),
                image.height(),
                source.width(),
                source.height()
            )),
            Err(error) => size_mismatches.push(format!("{}.png: {error}", part)),
        }
    }

    let ready = missing_parts.is_empty() && size_mismatches.is_empty();
    if ready {
        update_project_step(&root, 3)?;
    }

    Ok(WorkspaceGeneratedPartsStatus {
        request_path: root
            .join(CODEX_REQUEST_DIR)
            .join("codex_request.md")
            .to_string_lossy()
            .into_owned(),
        handoff_path: root
            .join(CODEX_REQUEST_DIR)
            .join("codex_handoff.md")
            .to_string_lossy()
            .into_owned(),
        generated_parts_path: generated_dir.to_string_lossy().into_owned(),
        expected_parts,
        present_parts,
        ready,
        missing_parts,
        size_mismatches,
    })
}

fn update_expression_workspace_step_inner(
    work_path: &str,
    current_step: u32,
) -> Result<ExpressionWorkspaceResult, AppError> {
    let root = PathBuf::from(work_path);
    ensure_workspace_dirs(&root)?;
    update_project_step(&root, current_step)?;
    let project = read_project(&root.join(PROJECT_FILE))?;
    workspace_result(root, project)
}

fn update_project_step(root: &Path, current_step: u32) -> Result<(), AppError> {
    let mut project = read_or_create_project(root)?;
    project.current_step = project.current_step.max(current_step.clamp(1, 6));
    project.updated_at = unix_time();
    write_project(&root.join(PROJECT_FILE), &project)
}

fn ensure_workspace_dirs(root: &Path) -> Result<(), AppError> {
    fs::create_dir_all(root)?;
    for dir in [
        CODEX_REQUEST_DIR,
        GENERATED_PARTS_DIR,
        SEE_THROUGH_DIR,
        SPRITALK_PARTS_DIR,
    ] {
        fs::create_dir_all(root.join(dir))?;
    }
    Ok(())
}

fn workspace_result(
    root: PathBuf,
    project: WorkspaceProject,
) -> Result<ExpressionWorkspaceResult, AppError> {
    let project_path = root.join(PROJECT_FILE);
    Ok(ExpressionWorkspaceResult {
        work_path: root.to_string_lossy().into_owned(),
        project_path: project_path.to_string_lossy().into_owned(),
        codex_request_path: root.join(CODEX_REQUEST_DIR).to_string_lossy().into_owned(),
        generated_parts_path: root
            .join(GENERATED_PARTS_DIR)
            .to_string_lossy()
            .into_owned(),
        see_through_path: root.join(SEE_THROUGH_DIR).to_string_lossy().into_owned(),
        spritalk_parts_path: root.join(SPRITALK_PARTS_DIR).to_string_lossy().into_owned(),
        project,
    })
}

fn read_project(path: &Path) -> Result<WorkspaceProject, AppError> {
    let text = fs::read_to_string(path)?;
    serde_json::from_str(&text).map_err(|error| {
        AppError::General(format!("project.json の読み込みに失敗しました: {error}"))
    })
}

fn read_or_create_project(root: &Path) -> Result<WorkspaceProject, AppError> {
    let path = root.join(PROJECT_FILE);
    if path.is_file() {
        read_project(&path)
    } else {
        let now = unix_time();
        Ok(WorkspaceProject {
            version: 1,
            created_at: now,
            updated_at: now,
            current_step: 1,
            source_image_path: None,
            reference_image_path: None,
        })
    }
}

fn write_project(path: &Path, project: &WorkspaceProject) -> Result<(), AppError> {
    let text = serde_json::to_string_pretty(project).map_err(|error| {
        AppError::General(format!("project.json の作成に失敗しました: {error}"))
    })?;
    fs::write(path, text)?;
    Ok(())
}

fn workspace_codex_request_text(
    source_path: &Path,
    reference_path: Option<&Path>,
    generated_dir: &Path,
    expected_parts: &[String],
    request: &PrepareWorkspaceCodexRequest,
) -> String {
    let expected = expected_parts
        .iter()
        .map(|part| format!("- `{part}.png`"))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        r#"# PachiPakuGen Codex生成依頼

## 入力
- 元画像: `{}`
- 参照画像: `{}`

## 出力先
生成したPNGをこのフォルダへ保存してください。

`{}`

## 必要ファイル
{}

## 生成ルール
- 元画像と同じキャンバスサイズで出力してください。
- キャラクター、服、髪、ポーズ、カメラ、背景、照明は維持してください。
- 指定された目または口だけを変更してください。
- mouth-closed は閉じ口、mouth-a/i/u/e/o は日本語母音の口形、eyes-closed は閉眼です。
- 口角: `{}`
- 口サイズ: `{}`

## 追加指示
{}
"#,
        source_path.display(),
        reference_path
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "なし".to_string()),
        generated_dir.display(),
        expected,
        request.mouth_corner,
        request.mouth_size,
        request.prompt
    )
}

fn workspace_codex_handoff_text(
    request_path: &Path,
    generated_dir: &Path,
    expected_parts: &[String],
) -> String {
    let expected = expected_parts
        .iter()
        .map(|part| format!("{}.png", part))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "Codexへ渡すファイル: {}\n生成PNGの保存先: {}\n必要ファイル: {}\n",
        request_path.display(),
        generated_dir.display(),
        expected
    )
}

fn unix_time() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_workspace_writes_project_and_standard_dirs() {
        let root = unique_temp_dir("pachipakugen_workspace_create");
        let result = create_expression_workspace_inner(&root.to_string_lossy()).unwrap();

        assert!(root.join(PROJECT_FILE).is_file());
        assert!(root.join(CODEX_REQUEST_DIR).is_dir());
        assert!(root.join(GENERATED_PARTS_DIR).is_dir());
        assert!(root.join(SEE_THROUGH_DIR).is_dir());
        assert!(root.join(SPRITALK_PARTS_DIR).is_dir());
        assert_eq!(result.project.current_step, 1);
        assert_eq!(
            result.codex_request_path,
            root.join(CODEX_REQUEST_DIR).to_string_lossy()
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn load_workspace_requires_project_file() {
        let root = unique_temp_dir("pachipakugen_workspace_missing_project");
        fs::create_dir_all(&root).unwrap();

        let error = load_expression_workspace_inner(&root.to_string_lossy()).unwrap_err();
        assert!(format!("{error}").contains("project.json"));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn prepare_workspace_codex_request_uses_fixed_dirs() {
        let root = unique_temp_dir("pachipakugen_workspace_codex");
        let source = root.join("input.png");
        fs::create_dir_all(&root).unwrap();
        image::RgbaImage::from_pixel(8, 8, image::Rgba([1, 2, 3, 255]))
            .save(&source)
            .unwrap();
        create_expression_workspace_inner(&root.to_string_lossy()).unwrap();

        let status = prepare_workspace_codex_request_inner(PrepareWorkspaceCodexRequest {
            work_path: root.to_string_lossy().into_owned(),
            source_image_path: source.to_string_lossy().into_owned(),
            reference_image_path: None,
            prompt: "test prompt".into(),
            mouth_corner: "neutral".into(),
            mouth_size: "normal".into(),
        })
        .unwrap();

        assert_eq!(
            status.generated_parts_path,
            root.join(GENERATED_PARTS_DIR).to_string_lossy()
        );
        assert!(root
            .join(CODEX_REQUEST_DIR)
            .join("codex_request.md")
            .is_file());
        assert!(root.join(CODEX_REQUEST_DIR).join("source.png").is_file());
        assert_eq!(status.missing_parts.len(), WORKSPACE_TARGETS.len());

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn update_workspace_step_only_moves_forward() {
        let root = unique_temp_dir("pachipakugen_workspace_step");
        create_expression_workspace_inner(&root.to_string_lossy()).unwrap();

        let updated = update_expression_workspace_step_inner(&root.to_string_lossy(), 4).unwrap();
        assert_eq!(updated.project.current_step, 4);

        let updated = update_expression_workspace_step_inner(&root.to_string_lossy(), 2).unwrap();
        assert_eq!(updated.project.current_step, 4);

        let _ = fs::remove_dir_all(root);
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        std::env::temp_dir().join(format!("{prefix}_{}", unix_time()))
    }
}
