# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

PachiPakuGen は、See-Throughで分解済みのアニメキャラクターレイヤーPSDからRIFEで中間フレームをバッチ生成するデスクトップアプリ。
姉妹プロジェクト PachiPakuTween（2枚の完成画像→マスクベース補間）と異なり、See-Through出力のPSDを入力として受け取り、レイヤーマッピング→合成→RIFEで補間する。

**2つの独立モード:**
- **素体出力モード**: PSD + 元画像 → SAM3 neck抽出 → Hair/Bodyレイヤー編集 → body.png / hair.png / hair_back.png
- **フレーム補間モード**: 閉じPSD+元画像 ↔ 開きPSD+元画像 → SAM3 mouth+neck抽出 → RIFE中間フレーム生成

**入力（See-Through出力想定）:**
- See-Throughで分解したPSDファイル + See-Throughに入力した元画像のペア
- See-Throughレイヤー: face, neck, nose, irides, eyewhite, eyelash, eyebrow, mouth, front_hair, back_hair, headwear, ears等（最大22レイヤー）
- 固定マッピング（自動統合）+ ユーザー判断マッピング（UI表示）でSpriTalkレイヤーに統合

## 技術スタック

- **Frontend:** React 19 + TypeScript + Vite（ポート1421）
- **Backend:** Rust + Tauri 2.0
- **Inference:** ONNX Runtime 2.0 + DirectML（GPU優先、CPU自動フォールバック）
- **Models:** rife.onnx（23MB）— src-tauri/models/ に配置
- **SAM3:** Python subprocess連携（sam3.pt 3.5GB — src-tauri/models/）
- **PSD読み込み:** psd crate（See-Through PSD出力対応）
- **Platform:** Windows 10/11 + DirectX 12 GPU推奨

## ビルド・開発コマンド

```bash
# 開発モード（Vite HMR + Rust）
npm run tauri dev

# Rust のみビルド（高速フィードバック）
cd src-tauri && cargo build

# リリースビルド
npm run tauri build
```

## アーキテクチャ

```
PachiPakuGen/
├── src/                    # React フロントエンド
│   ├── App.tsx             # メインUI（3モード選択 + 各ワークフロー）
│   ├── App.css             # ダークテーマスタイル
│   └── main.tsx            # エントリポイント
├── src-tauri/src/          # Rust バックエンド
│   ├── lib.rs              # Tauri初期化 + コマンド登録
│   ├── state.rs            # AppState（Mutex保護グローバル状態）
│   ├── error.rs            # AppError（thiserror + Serialize）
│   ├── commands/
│   │   └── parts.rs        # 全コマンド: load_slot, create_base, create_diff, get_mapping_preview, render_category, load_original_image
│   ├── inference/
│   │   ├── rife.rs         # RIFE補間（64px pad、RGB [0,1]テンソル）
│   │   ├── session.rs      # セッション管理（DirectML+CPUフォールバック）
│   │   └── neck_extract.rs # SAM3連携（neck/mouth抽出、マスク膨張）
│   └── processing/
│       ├── composite.rs    # アルファ合成・premultiply
│       └── image_utils.rs  # base64 PNG/JPEGエンコード
├── scripts/
│   └── extract_neck_mask.py # SAM3 Pythonスクリプト
└── src-tauri/models/       # モデル（.gitignore対象）
    ├── rife.onnx           # 23MB — RIFE補間
    └── sam3.pt             # 3.5GB — SAM3セグメンテーション
```

## UIフロー

```
モード選択: [素体出力] [まばたき] [口パク]
                ↓            ↓         ↓
素体:   PSD+元画像 → SAM3 → Hair編集 → Body編集 → 出力
補間:   差分PSD+元画像ペア設定 → SAM3 → RIFE生成 → 出力
```

## データフロー

### 素体出力
1. `load_slot` でPSDからSee-Throughレイヤーを読み込み
2. `load_original_image` で元画像からSAM3 neck+mouth抽出、キャッシュ
3. Hair/Bodyレイヤーの並び替え・ON/OFF（UI）
4. `create_base` でマッピングに従いレイヤー合成 → body.png, hair.png, hair_back.png出力

### フレーム補間
1. `load_slot` で閉じPSD読み込み → `load_original_image` で開き元画像からSAM3抽出
2. `create_base` で閉じPSDのeye/mouth/bodyをpartsに保持（ファイル出力なし）
3. `create_diff` で開きPSD読み込み → premultiply → RIFE補間 → alpha抽出 → frame_001〜N出力
4. mouth: SAM3マスク（開き元画像から検出）を各元画像に適用（PSD合成画像のインペイント痕を回避）

## レイヤーマッピング

### 固定（変更不可、自動統合）
| See-Through | → SpriTalk |
|---|---|
| face, neck, nose | → body |
| ★irides(-l/-r), ★eyewhite(-l/-r), ★eyelash(-l/-r), ★eyebrow(-l/-r) | → eye |
| mouth | → mouth |

★ = 左右分離オプション（-l/-r サフィックス付きレイヤーが出力される場合あり）

### ユーザー判断（UI表示、デフォルト値あり）
| See-Through | デフォルト | 備考 |
|---|---|---|
| front_hair | hair | 前髪 |
| back_hair | hair_back | 後ろ髪 |
| headwear | hair | カチューシャ、獣耳等 |
| ★ears(-l/-r) | body | 獣耳の場合skip可 |
| topwear, bottomwear, legwear, footwear | body | 衣服類 |
| ★handwear(-l/-r) | body | 腕・手袋 |
| earwear, eyewear, neckwear, objects | body | 装飾類 |
| wings | body | 将来独立レイヤー候補 |
| tail | body | 将来独立レイヤー候補 |

## 設計上の注意点

- **SAM3 mouth検出**: 元画像に対して実行（PSD再合成画像はインペイント痕で検出品質低下）。開き口の元画像からマスク検出→キャッシュ→全表情に再利用
- **SAM3 mouthピクセル**: マスクは共通だが、ピクセルは各表情の元画像から取得（base=閉じ元画像、diff=開き元画像）
- **premultiply**: RIFEに透過画像を直接渡すと黒ピクセルとの補間で色が崩れる。必ず素体に合成してからRIFEに通し、出力後にalphaチャンネルで切り出す
- **素体/補間モード判定**: `create_base`の`output_path`空=補間モード（partsをメモリ保持のみ）、非空=素体モード（ファイル出力）
- **セッション再利用**: `rife_session` は初回使用時にlazy init、以降キャッシュ
- **spawn_blocking**: 全Tauriコマンドでblocking taskをspawnしてUIスレッドを止めない
- **進捗イベント**: `generation-progress` イベントでフロントエンドにリアルタイム通知
- **合成順序**: hair_back(-1) → body(0) → eye(1) → mouth(2) → hair(3)
- **フレーム出力順序**: frame_001=閉じ、frame_N=開き（reverseして出力）
- **cached_original**: load_original_imageで元画像をキャッシュ。mouth maskとは別管理

## 由来

- 推論・画像処理コードは姉妹プロジェクト PachiPakuTween から派生
- 本リポジトリは独立プロジェクトとして管理
