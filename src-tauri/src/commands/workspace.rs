use crate::error::AppError;
use image::GenericImageView;
use serde::{Deserialize, Serialize};
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WorkspaceMouthCornerMode {
    Source,
    Up,
    #[default]
    #[serde(alias = "neutral")]
    Flat,
    Down,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkspaceProject {
    pub version: u32,
    pub created_at: u64,
    pub updated_at: u64,
    pub current_step: u32,
    pub source_image_path: Option<String>,
    pub reference_image_path: Option<String>,
    #[serde(default)]
    pub codex_prompt: Option<String>,
    #[serde(default)]
    pub mouth_corner: WorkspaceMouthCornerMode,
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
    #[serde(default)]
    pub mouth_corner: WorkspaceMouthCornerMode,
    pub mouth_size: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkspaceGeneratedPartsStatus {
    pub request_path: String,
    pub generated_parts_path: String,
    pub expected_parts: Vec<String>,
    pub present_parts: Vec<String>,
    pub missing_parts: Vec<String>,
    pub stale_parts: Vec<String>,
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
            codex_prompt: None,
            mouth_corner: WorkspaceMouthCornerMode::Flat,
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
    let source_image = image::open(&source)?;
    let (source_width, source_height) = (source_image.width(), source_image.height());
    source_image.save(&source_copy)?;

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
    fs::write(
        &request_path,
        workspace_codex_request_text(
            &source_copy,
            reference_copy.as_deref(),
            &generated_dir,
            &expected_parts,
            &request,
            source_width,
            source_height,
        ),
    )?;
    // codex_request.mdを唯一の指示書にする。旧版が作った重複ファイルは
    // 再作成時に残さず、Codexが複数の契約を照合する状態を避ける。
    for obsolete_name in ["codex_handoff.md", "codex_manifest.json"] {
        let obsolete_path = codex_dir.join(obsolete_name);
        if obsolete_path.is_file() {
            fs::remove_file(obsolete_path)?;
        }
    }

    let mut project = read_or_create_project(&root)?;
    project.updated_at = unix_time();
    project.current_step = project.current_step.max(2);
    project.source_image_path = Some(request.source_image_path);
    project.reference_image_path = request.reference_image_path;
    project.codex_prompt = if request.prompt.trim().is_empty() {
        None
    } else {
        Some(request.prompt.trim().to_string())
    };
    project.mouth_corner = request.mouth_corner;
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
    let request_modified = fs::metadata(root.join(CODEX_REQUEST_DIR).join("codex_request.md"))
        .and_then(|metadata| metadata.modified())
        .ok();
    let mut present_parts = Vec::new();
    let mut missing_parts = Vec::new();
    let mut stale_parts = Vec::new();
    let mut size_mismatches = Vec::new();

    for part in &expected_parts {
        let path = generated_dir.join(format!("{part}.png"));
        if !path.is_file() {
            missing_parts.push(part.clone());
            continue;
        }
        let is_stale = request_modified.is_some_and(|request_time| {
            fs::metadata(&path)
                .and_then(|metadata| metadata.modified())
                .map(|part_time| part_time < request_time)
                .unwrap_or(false)
        });
        if is_stale {
            stale_parts.push(part.clone());
            continue;
        }
        present_parts.push(part.clone());
        match image::open(&path) {
            Ok(image) if image.dimensions() == source.dimensions() => {}
            Ok(image) => {
                // 同アスペクト比なら抽出時に自動リサイズするため受け入れる
                let source_aspect = source.width() as f64 / source.height() as f64;
                let part_aspect = image.width() as f64 / image.height() as f64;
                if (source_aspect - part_aspect).abs() > 0.01 {
                    size_mismatches.push(format!(
                        "{}.png: {}x{} (expected {}x{} または同アスペクト比)",
                        part,
                        image.width(),
                        image.height(),
                        source.width(),
                        source.height()
                    ));
                }
            }
            Err(error) => size_mismatches.push(format!("{}.png: {error}", part)),
        }
    }

    let ready = missing_parts.is_empty() && stale_parts.is_empty() && size_mismatches.is_empty();
    if ready {
        update_project_step(&root, 3)?;
    }

    Ok(WorkspaceGeneratedPartsStatus {
        request_path: root
            .join(CODEX_REQUEST_DIR)
            .join("codex_request.md")
            .to_string_lossy()
            .into_owned(),
        generated_parts_path: generated_dir.to_string_lossy().into_owned(),
        expected_parts,
        present_parts,
        ready,
        missing_parts,
        stale_parts,
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
    project.current_step = project.current_step.max(current_step.clamp(1, 7));
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
            codex_prompt: None,
            mouth_corner: WorkspaceMouthCornerMode::Flat,
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
    source_width: u32,
    source_height: u32,
) -> String {
    let expected = expected_parts
        .iter()
        .map(|part| format!("- `{part}.png`"))
        .collect::<Vec<_>>()
        .join("\n");
    let reference_instructions = match reference_path {
        Some(path) => format!(
            r#"- 描画参考: `{}`
- 描画参考から借りてよいのは、口内の配色と明度差、歯・舌・目・まぶたの線と塗り方だけです。
- 描画参考のポーズ、顔向き、表情、口角、牙、髪、服、背景、構図はコピーしないでください。"#,
            path.display()
        ),
        None => "- 描画参考: なし\n- 元画像の既存の線、塗り、色設計を基準に、必要最小限の編集にしてください。".to_string(),
    };
    let (mouth_corner_key, mouth_corner_label, mouth_corner_rule) = match request.mouth_corner {
        WorkspaceMouthCornerMode::Source => (
            "source",
            "元画像に合わせる",
            "固定の口角AUは追加しない。元画像の口角の向き・強さ・自然な左右差を読み取り、閉じ口の輪郭そのものをコピーせず、各口形へその傾向だけを引き継ぐ",
        ),
        WorkspaceMouthCornerMode::Up => (
            "up",
            "少し上げる",
            "AU12B。左右の口角だけを控えめに上げ、頬・目・眉は変えない",
        ),
        WorkspaceMouthCornerMode::Flat => (
            "flat",
            "普通・ニュートラル",
            "口角を上げ下げするAUは追加しない。顔向きと遠近に沿う自然な左右差を残しつつ、意図的な上げ下げを付けない",
        ),
        WorkspaceMouthCornerMode::Down => (
            "down",
            "少し下げる",
            "AU15B。左右の口角だけを控えめに下げ、頬・目・眉は変えない",
        ),
    };
    let additional_instructions = if request.prompt.trim().is_empty() {
        "なし".to_string()
    } else {
        request.prompt.trim().to_string()
    };
    format!(
        r#"# PachiPakuGen ImageGen生成依頼（唯一の指示書）

## 目的
元画像から、後続処理で目または口だけを切り出すための差分用フルフレームPNGを生成してください。

## 入力の役割
- 編集対象・正本: `{source}`
{reference_instructions}

各出力は必ず編集対象の元画像から個別に作ってください。生成済みの別口形を次の編集元にせず、7枚を連鎖編集しないでください。

## 共通の編集制約
- 新しいコード、スクリプト、手描き合成、SVG的な図形描画、ローカル画像処理は使わず、画像生成/画像編集だけで生成してください。
- 口差分では口だけ、閉眼差分では両まぶただけを変更してください。
- 編集対象の人物同一性、顔向き、目・眉・頬・鼻・輪郭、髪、耳、アクセサリー、肌色、服、手、ポーズ、照明、背景、構図を意図的に変更しないでください。
- 後続処理では対象部位だけを切り出すため、非対象領域の微細な生成揺れは合否対象外ですが、別の表情やポーズへ変更してはいけません。
- 生成結果は自然なラスターアニメ塗りにしてください。SVG風、ベクター風、単純な図形、硬い塗りつぶしは禁止です。
- 出力解像度は元画像と同じ {width}x{height} にしてください。
- テキストや透かしを追加しないでください。

## 出力
以下のPNGを1枚ずつ生成し、このフォルダへ保存してください。

`{generated_dir}`

{expected}

## 全口形の共通条件
- 口角モード: `{mouth_corner_key}`（{mouth_corner_label}）。{mouth_corner_rule}。
- この口角モードは `mouth-closed` と「あ・い・う・え・お」の6枚だけに共通適用します。各母音の形状差を優先しつつ、口角の方向と強さを統一してください。丸口の「う・お」は丸さと開口を壊さない範囲で効果を弱めてください。
- 口サイズ: `{mouth_size}`。元画像の顔に対して自然な大きさを保ってください。
- 口を変えても、目、眉、頬、顔の輪郭や感情表現を連動して変えないでください。
- 追加指示に口角方向の異なる指定がある場合は、この口角モードを優先してください。方向と矛盾しない強さの微調整だけ反映して構いません。

## 各差分の仕様
- `mouth-closed` — FACS: `AU25=0 + AU26=0 + AU27=0`。唇が柔らかく接する自然な閉じ口。強く結ばず、すぼめない。
- `mouth-a` — FACS: `AU25C + AU26C`。日本語「あ」の中程度の自然な縦開き。歯を描く場合は上の歯だけ。下の歯は描かない。
- `mouth-i` — FACS: `AU20C + AU25B`。日本語「い」の中程度の横開き。上下の白い歯を明確に分けて見せ、ピンク・赤の口内、舌、暗い空洞は描かない。口角は選択したモードに従い、頬や目まで笑顔表現へ連動させない。
- `mouth-u` — FACS: `AU18B + AU22B + AU25A`。日本語「う」の小さく柔らかい丸い楕円。真円よりわずかに横長。歯や白い歯状ハイライトは描かず、突き出しを弱くしてキス口・口笛口にしない。
- `mouth-e` — FACS: `AU20B + AU25B + AU26B`。日本語「え」の中程度に開いた横長口。「い」より縦にも開く。上の歯だけを見せ、下の歯は描かない。口角は選択したモードに従い、頬や目まで笑顔表現へ連動させない。
- `mouth-o` — FACS: `AU22C + AU25B + AU26B`。日本語「お」の丸く奥行きのある開口。「う」より大きく、縦にも十分に開き、歯・牙・白いハイライト・独立した舌の輪郭は描かない。
  - 口内は一色にせず、上側を中明度の温かい赤茶、下側をそれより明るいコーラル／サーモン系として、滑らかな明度差と奥行きを残してください。描画参考がある場合は、その口内の相対的な配色と明度差を優先してください。
  - 「歯・舌なし」を、口内全体の黒塗りや暗色一色という意味に解釈しないでください。純黒・黒に近い単色の穴、単色円、ベタ塗り、平坦なステッカー状の口は禁止です。濃色は細い外周線と上側の陰影だけに限定してください。
- `eyes-closed` — FACS: `AU45E`。編集対象の元画像から独立生成し、口角モードは適用しません。口の形・位置・線・口角と眉を元画像のまま維持し、両まぶただけを自然なアニメ調の弧で完全に閉じてください。目が残る場合だけ `AU43D + AU45E` でやり直してください。

## 完了前の目視検査（必須）
保存した7枚を実際に開き、次を確認してから完了報告してください。

1. 7枚すべてが指定名のPNGで、{width}x{height}である。
2. 各画像の口または両まぶた以外は、編集対象の元画像から意図的に変更されていない。
3. 「い」「う」「お」が明確に異なり、「う」は小さい丸口、「お」はそれより大きく奥行きのある丸口になっている。
4. `mouth-o` の口内に暖色の明暗2段階があり、黒または暗赤茶の単色の穴に見えない。単色に見える場合は失敗として `mouth-o` だけを再生成する。
5. 6種類の口画像で選択した口角の傾向が揃い、目・眉・頬へ表情変化が波及していない。「う・お」は口角より母音の丸さと開口が優先されている。
6. `eyes-closed` は両目が完全に閉じ、眉と口が元画像どおりであり、選択した口角モードの影響を受けていない。

条件を満たさない画像が1枚でもある場合、その画像だけを再生成し、合格するまで完了と報告しないでください。

## 追加指示
{additional_instructions}

追加指示が上記の入力役割、変更範囲、口形仕様、目視検査と矛盾する場合は、上記を優先してください。
"#,
        source = source_path.display(),
        reference_instructions = reference_instructions,
        width = source_width,
        height = source_height,
        generated_dir = generated_dir.display(),
        expected = expected,
        mouth_corner_key = mouth_corner_key,
        mouth_corner_label = mouth_corner_label,
        mouth_corner_rule = mouth_corner_rule,
        mouth_size = request.mouth_size,
        additional_instructions = additional_instructions
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
        assert_eq!(result.project.mouth_corner, WorkspaceMouthCornerMode::Flat);
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
        let codex_dir = root.join(CODEX_REQUEST_DIR);
        fs::write(codex_dir.join("codex_handoff.md"), "legacy handoff").unwrap();
        fs::write(codex_dir.join("codex_manifest.json"), "{}").unwrap();

        let status = prepare_workspace_codex_request_inner(PrepareWorkspaceCodexRequest {
            work_path: root.to_string_lossy().into_owned(),
            source_image_path: source.to_string_lossy().into_owned(),
            reference_image_path: None,
            prompt: "test prompt".into(),
            mouth_corner: WorkspaceMouthCornerMode::Up,
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
        assert!(!codex_dir.join("codex_handoff.md").exists());
        assert!(!codex_dir.join("codex_manifest.json").exists());
        let request_text = fs::read_to_string(codex_dir.join("codex_request.md")).unwrap();
        assert!(request_text.contains("唯一の指示書"));
        assert!(request_text.contains("AU22C + AU25B + AU26B"));
        assert!(request_text.contains("暖色の明暗2段階"));
        assert!(request_text.contains("暗赤茶の単色の穴"));
        assert!(request_text.contains("mouth-o` だけを再生成"));
        assert!(request_text.contains("## 追加指示\ntest prompt"));
        let saved_project = read_project(&root.join(PROJECT_FILE)).unwrap();
        assert_eq!(saved_project.codex_prompt.as_deref(), Some("test prompt"));
        assert_eq!(saved_project.mouth_corner, WorkspaceMouthCornerMode::Up);
        assert!(request_text.contains("口角モード: `up`（少し上げる）"));
        assert!(request_text.contains("AU12B"));
        assert_eq!(status.missing_parts.len(), WORKSPACE_TARGETS.len());

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn updating_request_marks_older_generated_parts_as_stale() {
        let root = unique_temp_dir("pachipakugen_workspace_stale_parts");
        let source = root.join("input.png");
        fs::create_dir_all(&root).unwrap();
        let image = image::RgbaImage::from_pixel(8, 8, image::Rgba([1, 2, 3, 255]));
        image.save(&source).unwrap();
        create_expression_workspace_inner(&root.to_string_lossy()).unwrap();
        image
            .save(root.join(GENERATED_PARTS_DIR).join("mouth-closed.png"))
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));

        let status = prepare_workspace_codex_request_inner(PrepareWorkspaceCodexRequest {
            work_path: root.to_string_lossy().into_owned(),
            source_image_path: source.to_string_lossy().into_owned(),
            reference_image_path: None,
            prompt: String::new(),
            mouth_corner: WorkspaceMouthCornerMode::Down,
            mouth_size: "normal".into(),
        })
        .unwrap();

        assert!(!status.ready);
        assert_eq!(status.stale_parts, vec!["mouth-closed"]);
        assert!(!status
            .present_parts
            .iter()
            .any(|part| part == "mouth-closed"));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn workspace_codex_request_limits_reference_scope_and_rejects_flat_o_mouth() {
        let request = PrepareWorkspaceCodexRequest {
            work_path: "C:\\work".into(),
            source_image_path: "C:\\work\\source.png".into(),
            reference_image_path: Some("C:\\work\\reference.png".into()),
            prompt: String::new(),
            mouth_corner: WorkspaceMouthCornerMode::Flat,
            mouth_size: "normal".into(),
        };
        let text = workspace_codex_request_text(
            Path::new("C:\\work\\source.png"),
            Some(Path::new("C:\\work\\reference.png")),
            Path::new("C:\\work\\generated"),
            &WORKSPACE_TARGETS
                .iter()
                .map(|target| target.to_string())
                .collect::<Vec<_>>(),
            &request,
            1254,
            1254,
        );

        assert!(text.contains("描画参考から借りてよいのは、口内の配色と明度差"));
        assert!(text.contains("口角、牙、髪、服、背景、構図はコピーしない"));
        assert!(text.contains("上側を中明度の温かい赤茶"));
        assert!(text.contains("下側をそれより明るいコーラル／サーモン系"));
        assert!(text.contains("純黒・黒に近い単色の穴"));
        assert!(text.contains("黒または暗赤茶の単色の穴に見えない"));
        assert!(text.contains("口角モード: `flat`（普通・ニュートラル）"));
        assert!(text.contains("顔向きと遠近に沿う自然な左右差"));
        assert!(text.contains("口角方向の異なる指定がある場合は、この口角モードを優先"));
        assert!(text.contains("口角モードは適用しません"));
        assert!(text.contains("## 追加指示\nなし"));
    }

    #[test]
    fn workspace_codex_request_describes_source_and_down_corner_modes() {
        let render = |mouth_corner| {
            let request = PrepareWorkspaceCodexRequest {
                work_path: "C:\\work".into(),
                source_image_path: "C:\\work\\source.png".into(),
                reference_image_path: None,
                prompt: String::new(),
                mouth_corner,
                mouth_size: "normal".into(),
            };
            workspace_codex_request_text(
                Path::new("C:\\work\\source.png"),
                None,
                Path::new("C:\\work\\generated"),
                &WORKSPACE_TARGETS
                    .iter()
                    .map(|target| target.to_string())
                    .collect::<Vec<_>>(),
                &request,
                1254,
                1254,
            )
        };

        let source = render(WorkspaceMouthCornerMode::Source);
        assert!(source.contains("口角モード: `source`（元画像に合わせる）"));
        assert!(source.contains("自然な左右差を読み取り"));

        let down = render(WorkspaceMouthCornerMode::Down);
        assert!(down.contains("口角モード: `down`（少し下げる）"));
        assert!(down.contains("AU15B"));
    }

    #[test]
    fn legacy_workspace_without_codex_prompt_still_loads() {
        let project: WorkspaceProject = serde_json::from_str(
            r#"{
                "version": 1,
                "createdAt": 1,
                "updatedAt": 2,
                "currentStep": 2,
                "sourceImagePath": null,
                "referenceImagePath": null
            }"#,
        )
        .unwrap();

        assert_eq!(project.codex_prompt, None);
        assert_eq!(project.mouth_corner, WorkspaceMouthCornerMode::Flat);
    }

    #[test]
    fn legacy_neutral_corner_value_loads_as_flat() {
        let project: WorkspaceProject = serde_json::from_str(
            r#"{
                "version": 1,
                "createdAt": 1,
                "updatedAt": 2,
                "currentStep": 2,
                "sourceImagePath": null,
                "referenceImagePath": null,
                "mouthCorner": "neutral"
            }"#,
        )
        .unwrap();

        assert_eq!(project.mouth_corner, WorkspaceMouthCornerMode::Flat);
    }

    #[test]
    fn update_workspace_step_only_moves_forward() {
        let root = unique_temp_dir("pachipakugen_workspace_step");
        create_expression_workspace_inner(&root.to_string_lossy()).unwrap();

        let updated = update_expression_workspace_step_inner(&root.to_string_lossy(), 4).unwrap();
        assert_eq!(updated.project.current_step, 4);

        let updated = update_expression_workspace_step_inner(&root.to_string_lossy(), 2).unwrap();
        assert_eq!(updated.project.current_step, 4);

        // STEP7（モーション調整）まで保存でき、範囲外は7へクランプされる
        let updated = update_expression_workspace_step_inner(&root.to_string_lossy(), 7).unwrap();
        assert_eq!(updated.project.current_step, 7);

        let updated = update_expression_workspace_step_inner(&root.to_string_lossy(), 99).unwrap();
        assert_eq!(updated.project.current_step, 7);

        let _ = fs::remove_dir_all(root);
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        std::env::temp_dir().join(format!("{prefix}_{}", unix_time()))
    }
}
