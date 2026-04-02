# PachiPakuGen

アニメキャラクターの**目パチ・口パク用中間フレーム**をパーツ単位でバッチ生成するデスクトップアプリです。

[SpriTalk](https://github.com/kazuya-bros/SpriTalk) などの口パクアプリ向けに、透過PNGレイヤー素材を自動生成できます。

## 特徴

- **パーツ単位の補間** -- 目・口を個別に RIFE 補間するため、クロスフェードの透過崩れやマスクずれが発生しない
- **自動セグメント分離** -- 目+口が同居した透過PNGから UNet セグメンテーションで目だけ・口だけを自動抽出
- **口パク5母音対応** -- 口閉じ / 口あ〜お の最大5ペアを一括生成
- **ブラシマスク編集** -- セグメンテーション結果をブラシで直接修正可能
- **SAM3 対応** -- より高精度なセグメンテーションが必要な場合、SAM3（Python連携）も使用可能
- **GPU アクセラレーション** -- ONNX Runtime + DirectML によるGPU推論（CPU自動フォールバック）

## 動作要件

- Windows 10/11
- DirectX 12 対応 GPU（推奨）
- Node.js 18+
- Rust 1.75+
- Python 3.8+（SAM3 使用時のみ）

## 処理フロー

```
パーツ画像読み込み
    ↓
UNet セグメンテーション（目+口画像 → 目だけ / 口だけに分離）
    ↓
RIFE ペア構築（目1ペア + 口最大5ペア）
    ↓
RIFE 中間フレーム一括生成
    ↓
パーツ別透過 PNG 連番として出力
  eye/frame_001.png〜
  mouth_a/frame_001.png〜
  mouth_i/frame_001.png〜
  ...
```

## 入力素材

Qwen-Image-Layered 等で生成した透過PNGパーツを想定しています。

| パーツ | 説明 | 必須 |
|--------|------|------|
| 素体（body） | 体のみの画像 | Yes |
| 髪（hair） | 髪レイヤー | Yes |
| 目開き | 目+口が同居した透過PNG | Yes |
| 目閉じ | 同上 | Yes |
| 口閉じ | 同上 | Yes |
| 口あ〜お | 各母音の口画像 | 任意 |

## インストール

### リリースビルドを使う場合

[Releases](../../releases) からインストーラーをダウンロードして実行してください。

### 開発ビルド

```bash
# 依存パッケージのインストール
npm install

# 開発モードで起動
npm run tauri dev

# リリースビルド
npm run tauri build
```

## モデルファイル

| ファイル | サイズ | 用途 | 配布 |
|----------|--------|------|------|
| `rife.onnx` | 23MB | RIFE 補間 | アプリに同梱 |
| `unet_segmentation.onnx` | 6MB | UNet セグメンテーション | 別途DL |
| `sam3.pt` | 3.5GB | SAM3 セグメンテーション（Python連携） | 別途DL |

開発時は `src-tauri/models/` に、インストール済みアプリでは `PachiPakuGen.exe` と同じ階層の `models/` フォルダに配置してください。

### UNet モデルのセットアップ

1. [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) の学習済みモデルを ONNX 形式に変換、または変換済みモデルを入手
2. `unet_segmentation.onnx` にリネームして以下に配置

```
PachiPakuGen.exe
models/
  unet_segmentation.onnx  <-- ここに配置
  rife.onnx                （自動バンドル済み）
```

### SAM3 のセットアップ（オプション）

SAM3 を使用する場合は、以下の手順でセットアップしてください。

1. [sam3.pt をダウンロード](https://github.com/facebookresearch/sam3)
2. アプリの `models/` フォルダに配置

```
PachiPakuGen.exe
models/
  sam3.pt                  <-- ここに配置
  unet_segmentation.onnx
  rife.onnx                （自動バンドル済み）
```

3. SAM3 Python パッケージをインストール

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3 && pip install -e .
```

## 使い方

### STEP 1: パーツ入力

パーツ画像（素体・髪・目開き・目閉じ・口閉じ〜口お）を読み込みます。

### STEP 2: セグメント

UNet（または SAM3）でセグメンテーションを実行し、目+口画像から目だけ・口だけを自動分離します。必要に応じてブラシでマスクを修正できます。

### STEP 3: フレーム生成

フレーム数を指定して RIFE 中間フレームを一括生成します。

### STEP 4: プレビュー・エクスポート

目・口のスライダーでリアルタイムプレビューし、パーツ別フォルダに透過 PNG 連番としてエクスポートします。

## 技術スタック

- **Frontend:** React 19 + TypeScript + Vite
- **Backend:** Rust + Tauri 2.0
- **Inference:** ONNX Runtime 2.0 (DirectML)
- **Image Processing:** image / imageproc crate

## ライセンス

本アプリケーションのソースコードは [MIT License](LICENSE) で提供されます。

### 同梱モデルのライセンス

| モデル | 元プロジェクト | ライセンス |
|--------|---------------|-----------|
| `rife.onnx` | [ECCV2022-RIFE](https://github.com/hzwer/ECCV2022-RIFE) (Huang et al.) | MIT License |

### 別途取得が必要なモデル

| モデル | 元プロジェクト | ライセンス |
|--------|---------------|-----------|
| `unet_segmentation.onnx` | [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) (BiSeNet) | MIT License（学習データ CelebAMask-HQ は非商用限定） |
| `sam3.pt` | [SAM 3](https://github.com/facebookresearch/sam3) (Meta) | SAM License |
