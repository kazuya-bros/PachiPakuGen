# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

PachiPakuGen は、Qwen-Image-Layered等で生成した透過PNGパーツからRIFEで中間フレームをバッチ生成するデスクトップアプリ。
姉妹プロジェクト PachiPakuTween（2枚の完成画像→マスクベース補間）と異なり、パーツ画像を入力として受け取り、UNetセグメンテーションで目・口を自動分離してからRIFEで補間する。

**入力パーツ（Qwen-Image-Layered出力想定）:**
- 素体(body)、髪(hair) — そのまま使用
- 目開き / 目閉じ画像 — 目+口が同居した透過PNG → UNetで目だけ抽出
- 口閉じ / 口あ〜お画像 — 同上 → UNetで口だけ抽出
- 眉_通常 / 眉_上げ（将来拡張）

**処理フロー:**
1. パーツ読み込み → UNetセグメンテーションで目・口マスク検出
2. マスクで各画像から目だけ / 口だけの透過PNGに自動分割
3. 分割パーツでRIFEペア構築（目1ペア + 口最大5ペア）
4. 各ペアを一括RIFE中間フレーム生成
5. パーツ別透過PNG連番として出力（eye/, mouth_a/ 等）

## 技術スタック

- **Frontend:** React 19 + TypeScript + Vite（ポート1421）
- **Backend:** Rust + Tauri 2.0
- **Inference:** ONNX Runtime 2.0 + DirectML（GPU優先、CPU自動フォールバック）
- **Models:** rife.onnx（23MB）+ unet_segmentation.onnx（6MB）— src-tauri/models/ に配置
- **Platform:** Windows 10/11 + DirectX 12 GPU推奨

## ビルド・開発コマンド

```bash
# 依存パッケージインストール
cd PachiPakuGen-app && npm install

# 開発モード（Vite HMR + Rust）
npm run tauri dev

# Rust のみビルド（高速フィードバック）
cd src-tauri && cargo build

# リリースビルド
npm run tauri build

# PowerShellからのデバッグビルド
powershell -File build_dev.ps1
```

## アーキテクチャ

```
PachiPakuGen-app/
├── src/                    # React フロントエンド
│   ├── App.tsx             # メインUI（3ステップワークフロー）
│   ├── App.css             # ダークテーマスタイル
│   └── main.tsx            # エントリポイント
├── src-tauri/src/          # Rust バックエンド
│   ├── lib.rs              # Tauri初期化 + コマンド登録
│   ├── state.rs            # AppState（Mutex保護グローバル状態）
│   ├── error.rs            # AppError（thiserror + Serialize）
│   ├── commands/           # Tauriコマンドハンドラ
│   │   ├── parts.rs        # パーツ読み込み + UNetセグメント分離 + ペア構築
│   │   ├── generation.rs   # RIFE中間フレーム一括生成 + 合成プレビュー
│   │   └── export.rs       # パーツ別PNG連番出力
│   ├── inference/          # ONNX推論（PachiPakuTweenから流用）
│   │   ├── rife.rs         # RIFE補間（64px pad、RGB [0,1]テンソル）
│   │   ├── segmentation.rs # UNetセグメンテーション + マスク抽出・適用
│   │   └── session.rs      # セッション管理（DirectML+CPUフォールバック）
│   └── processing/         # 画像処理
│       ├── composite.rs    # アルファ合成・premultiply
│       └── image_utils.rs  # base64 PNG/JPEGエンコード
└── src-tauri/models/       # ONNXモデル（.gitignore対象）
    ├── rife.onnx           # 23MB — RIFE補間
    └── unet_segmentation.onnx # 6MB — 目・口セグメンテーション
```

## データフロー

1. **パーツ読み込み** → `load_parts` で全画像ロード
2. **自動セグメント分離** → eye_open画像を白背景合成→UNet推論→目/口マスク抽出→各画像にマスク適用して目だけ/口だけに分割
3. **ペア構築** → 分離済みパーツからPartPair自動構築（eye, mouth_a〜mouth_o）
4. **RIFE生成** → 各ペアを premultiply_onto_body → RIFE補間 → alpha抽出で透過PNG化
5. **プレビュー** → `get_composite_preview` で目・口スライダーに応じたレイヤー合成
6. **エクスポート** → パーツ別フォルダにframe_001.png〜連番出力

## 設計上の注意点

- **セグメント分離:** 入力画像は目+口が同居した透過PNG（Qwen-Image-Layered出力）。UNetで検出したマスクで目だけ・口だけに分割してからRIFEに渡す
- **白背景合成:** UNetはRGB入力前提。透過PNGを直接渡すと黒背景になり検出精度が落ちるため、白背景に合成してからセグメンテーション実行
- **premultiply:** RIFEに透過画像を直接渡すと黒ピクセルとの補間で色が崩れる。必ず素体に合成してからRIFEに通し、出力後にalphaチャンネルで切り出す
- **セッション再利用:** `rife_session`・`seg_session` は初回使用時にlazy init、以降キャッシュ
- **spawn_blocking:** 全Tauriコマンドでblocking taskをspawnしてUIスレッドを止めない
- **進捗イベント:** `generation-progress` イベントでフロントエンドにリアルタイム通知
- **モデル解決順序:** Tauriリソースディレクトリ → exe隣のmodels/ → CARGO_MANIFEST_DIR/models/

## 由来

- 推論・画像処理コードは姉妹プロジェクト PachiPakuTween から派生
- 本リポジトリは独立プロジェクトとして管理
