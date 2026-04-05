# PachiPakuGen

アニメキャラクターの**素体・目パチ・口パク用フレーム**をパーツ単位でバッチ生成するデスクトップアプリです。

[SpriTalk](https://github.com/kazuya-bros/SpriTalk) 専用の素材生成ツールです。透過PNGレイヤー素材を自動生成します。

## 特徴

- **See-Through PSD 入力** -- [See-Through](https://github.com/shitagaki-lab/see-through) で分解済みのPSD（最大22レイヤー）を直接読み込み
- **SAM3 による自動抽出** -- 元画像からSAM3で首・口領域を高精度に自動抽出（Python連携）
- **素体出力** -- body / hair / hair_back の3パーツを透過PNGで出力。レイヤー並び替え・ON/OFF対応
- **RIFE フレーム補間** -- 目パチ・口パク用の中間フレームをパーツ単位で自動生成
- **口パク5母音対応** -- 口閉じ / 口あ〜お の最大5ペアを一括生成
- **GPU アクセラレーション** -- ONNX Runtime 2.0 + DirectML によるGPU推論（CPU自動フォールバック）

## 動作要件

- Windows 10/11
- DirectX 12 対応 GPU（推奨）
- Node.js 18+
- Rust 1.75+
- Python 3.12+（SAM3 使用、必須）
- PyTorch 2.7+（SAM3 使用、CUDA 12.6+ 推奨）

## 処理フロー

### 素体出力モード

```
See-Through PSD + 元画像
    ↓
SAM3 首抽出（元画像から）
    ↓
Hair レイヤー編集（並び替え・ON/OFF）
    ↓
Body レイヤー編集（並び替え・ON/OFF）
    ↓
出力: body.png / hair.png / hair_back.png
```

### まばたき・口パク フレーム補間モード

```
閉じ PSD + 元画像  ↔  開き PSD + 元画像
    ↓
SAM3 口・首抽出（開き元画像から）
    ↓
RIFE 中間フレーム生成
    ↓
出力: eye/frame_001.png〜 / mouth_a/frame_001.png〜 ...
（frame_001 = 閉じ、frame_N = 開き）
```

## 入力素材

[See-Through](https://github.com/shitagaki-lab/see-through) で分解したPSDファイルと、See-Throughに入力した元画像のペアが必要です。

| 入力 | 説明 | 必須 |
|------|------|------|
| PSD | See-Through出力のPSDファイル | Yes |
| 元画像 | See-Throughに入力した元のイラスト画像 | Yes |

### See-Through レイヤー対応

| マッピング | レイヤー |
|-----------|---------|
| body (固定) | face, neck, nose, topwear, bottomwear 等 |
| eye (固定) | irides, eyewhite, eyelash, eyebrow (L/R対応) |
| mouth (固定) | mouth |
| hair (調整可) | front_hair, headwear |
| hair_back (調整可) | back_hair |

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
| `sam3.pt` | 約3.2GB | SAM3 首・口抽出（Python連携） | 別途DL |

開発時は `src-tauri/models/` に、インストール済みアプリでは `PachiPakuGen.exe` と同じ階層の `models/` フォルダに配置してください。

### SAM3 のセットアップ

1. [sam3.pt をダウンロード](https://github.com/facebookresearch/sam3)
2. `models/` フォルダに配置

```
src-tauri/models/
  sam3.pt                  <-- ここに配置
  rife.onnx                （自動バンドル済み）
```

3. PyTorch 2.7+ をインストール（未導入の場合）

```bash
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
```

4. SAM3 Python パッケージをインストール

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3 && pip install -e .
```

## 使い方

### 素体出力

1. モード選択画面で「素体出力」を選択
2. See-Through PSD と元画像を読み込み
3. Step 1/2: Hair レイヤーの並び替え・ON/OFF
4. Step 2/2: Body レイヤーの並び替え・ON/OFF
5. 出力先フォルダを選択して素体出力

### まばたき・口パク フレーム補間

1. モード選択画面で「まばたき」または「口パク」を選択
2. 閉じ/開きのPSD + 元画像ペアを設定
3. フレーム数を指定して一括生成

## 技術スタック

- **Frontend:** React 19 + TypeScript + Vite
- **Backend:** Rust + Tauri 2.0
- **Inference:** ONNX Runtime 2.0 (DirectML)
- **PSD読み込み:** psd crate
- **SAM3連携:** Python subprocess

## ライセンス

本アプリケーションのソースコードは [MIT License](LICENSE) で提供されます。

### 同梱モデルのライセンス

| モデル | 元プロジェクト | ライセンス |
|--------|---------------|-----------|
| `rife.onnx` | [ECCV2022-RIFE](https://github.com/hzwer/ECCV2022-RIFE) (Huang et al.) | MIT License |

### 別途取得が必要なモデル

| モデル | 元プロジェクト | ライセンス |
|--------|---------------|-----------|
| `sam3.pt` | [SAM 3](https://github.com/facebookresearch/sam3) (Meta) | SAM License |
