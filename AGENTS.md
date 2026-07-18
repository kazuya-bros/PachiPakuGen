# AGENTS.md

このファイルはPachiPakuGenを編集するコーディングエージェント向けの案内です。利用者向け仕様は[README.md](README.md)、第三者由来は[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)を優先してください。

## 現行プロダクト

PachiPakuGen v0.4系は、1枚の立ち絵からSpriTalk向けの表情・モーション素材を作るWindowsデスクトップアプリです。

```text
画像選択 → 表情素材 → See-Through → 素体調整 → 差分位置調整
         → RIFE補完 → モーション調整／ライブ表示
```

- スタートメニュー: はじめから／つづきから／ライブ表示
- 作業単位: `project.json`を持つ作業フォルダ
- 完成素材: `<作業フォルダ>/04_spritalk_parts`
- See-Through: 固定した公式コミットをユーザー操作後にアプリ管理領域へ取得
- RIFE: Practical-RIFE v4.9.2から監査済み手順で出力したONNXを同梱
- SAM3: v0.4.0で実行経路・Python環境・モデル要件を削除済み

## 技術スタック

- Frontend: React 19、TypeScript、Vite
- Desktop/backend: Tauri 2、Rust
- Image processing: `image`、`psd`
- Frame interpolation: ONNX Runtime、DirectML優先・CPUフォールバック
- See-Through runtime: Python 3.12、uv、CUDA版PyTorch（アプリ管理領域）
- Platform: Windows 10/11 x64

## 主な場所

```text
src/
├─ App.tsx / App.css / WorkspaceFlow.css  # 7STEP UIと共通レイアウト
├─ workspace/                             # 工程モデルと作業状態
└─ motionLab/                             # STEP 7、ライブ表示、描画・各効果

src-tauri/src/
├─ commands/workspace.rs                  # 作業フォルダ
├─ commands/expression.rs                 # 表情抽出、素体、差分、RIFE一括出力
├─ commands/see_through.rs                # 固定ランタイム、取得、互換パッチ、実行
├─ commands/motion_lab.rs                 # モーション設定とmanifest
├─ commands/parts.rs                      # PSD編集と現行表情パーツ構成
├─ inference/rife.rs                      # 64px padとRIFE推論
└─ inference/neck_extract.rs              # 汎用マスク膨張・ぼかし処理

src-tauri/scripts/
├─ see_through_model_requirements.json    # モデル配布元・revision・サイズ
└─ prepare_see_through_models.py           # 事前取得とmanifest検証
```

## 開発コマンド

```powershell
npm ci
npm run tauri -- dev
npm run build

npm run test:blink
npm run test:hair
npm run test:chest
npm run test:eyes
npm run test:ears
npm run test:templates
npm run test:workspace

Push-Location src-tauri
cargo fmt --check
cargo test --locked
Pop-Location

npm run tauri -- build
```

## 重要な設計条件

- 重い処理はTauriのUIスレッドで直接実行しない。
- 初回セットアップとモデル取得は明示ボタンからだけ開始する。
- See-Through取得元、コミット、モデルrevision、必要サイズを勝手に浮動化しない。
- 透過差分をRIFEへ直接入れず、素体へpremultiplyして補完後にalphaを戻す。
- RIFE入力は64px倍数へpadし、出力時に元キャンバスへcropする。
- STEP 5の再編集後は素体パーツを失わず、STEP 6を再実行できること。
- `spritalk-motion-profile.json` v2はPachiPakuGenライブ表示では利用可能だが、SpriTalk側の読み込みは未実装。
- UIの主操作はピンク、選択・編集・移動は青、完了は緑、エラーは赤。完了タブと現在タブを同色にしない。
- ヘッダー、7STEPタブ、本文、固定下部操作の基準線をそろえる。

## 第三者由来を変更するとき

- See-Throughの取得・互換パッチを変えたら、固定revision、モデル一覧、変更対象ファイル、ライセンスを再監査する。
- `src/motionLab/render.ts` のウェーブ式はPuruPuruPNGTuberのApache-2.0実装を変更・適応している。
- `src/motionLabPhysics.ts` の髪房検出・二重バネはAnime2.5DRigのMIT実装を変更・適応している。
- 上記コメントと`THIRD_PARTY_NOTICES.md`の出所・変更通知を削除しない。
- RIFEモデルを差し替える場合は、変換元、ライセンス、入力ハッシュ、変換スクリプト、出力SHA-256、DirectML実画像試験をそろえる。
- 依存を更新したら`npm run licenses:npm`と`npm run licenses:cargo`を再実行する。

## リリース

正式版を作る前に[RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md)を完了してください。版番号は少なくとも次を一致させます。

- `package.json` / `package-lock.json`
- `src-tauri/Cargo.toml` / `src-tauri/Cargo.lock`
- `src-tauri/tauri.conf.json`
- `CHANGELOG.md`

インストーラーへユーザー画像、トークン、ログ、作業フォルダ、See-Through大型モデル、旧SAM3関連ファイルを混入させないでください。
