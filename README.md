# PachiPakuGen

[SpriTalk](https://kazuyabros.booth.pm/items/8102679) 専用の素材生成ツールです。

**RIFE（Real-Time Intermediate Flow Estimation）** によるフレーム補間で、滑らかな口パク・まばたきアニメーションを実現します。See-Through で分解したPSDから、SpriTalk のレイヤーモードに必要な透過PNGパーツを自動生成します。

## 特徴

- **See-Through PSD 入力** -- [See-Through](https://github.com/shitagaki-lab/see-through) で分解済みのPSDを直接読み込み
- **SAM3 による自動抽出** -- 元画像からSAM3で首・口領域を自動抽出（Python連携）。素体出力では首検出結果を確認し、プロンプトを変えて再検出できます
- **素体出力** -- body / hair / hair_back の3パーツを透過PNGで出力。レイヤー並び替え・ON/OFF対応
- **RIFE フレーム補間** -- 目パチ・口パク用の中間フレームをパーツ単位で自動生成
- **口パク5母音対応** -- 口閉じ / 口あ〜お の最大5ペアを一括生成
- **GPU 前提の高速推論** -- ONNX Runtime 2.0 + DirectML と CUDA 版 PyTorch によるGPU推論

## なぜ See-Through + SAM3 か

[See-Through](https://github.com/shitagaki-lab/see-through)（[ComfyUIノード](https://github.com/jtydhr88/ComfyUI-See-through)）は1枚のイラストから髪・顔・服などのセマンティックレイヤーをPSDとして自動分解するAIです。Qwen-Image-Layered と比べて低VRAMで動作するため、SpriTalk のレイヤーモード用素材が格段に作りやすくなりました。

ただし See-Through 単体では SpriTalk 素材として使う際に以下の問題があります。

- **首**: アウトペイント（塗り足し）で生成されるため、レイヤーの重ね順だけでは不自然な継ぎ目が残る
- **口**: mouth レイヤーの検出範囲が広く、口パク補間時にノイズが入りやすい

PachiPakuGen は **SAM3（Segment Anything Model 3）** を併用し、元画像から首・口領域を高精度に切り出すことでこれらの問題を解決します。

> **注意**: このアプリは GPU 搭載環境での利用を前提にしています。CPUでも一部処理は動作する場合がありますが、SAM3/RIFE ともに非常に低速で、通常利用・サポート対象はGPU環境です。SAM3 のモデルファイル（約3.2GB）の別途ダウンロードと、推論用の GPU VRAM が必要です。

## 動作要件

- Windows 10/11
- DirectX 12 対応 GPU（必須）
- NVIDIA GPU + CUDA 対応 PyTorch（SAM3 使用時は必須）
- Node.js 18+
- Rust 1.75+
- Python 3.12+（SAM3 使用、必須）
- CUDA版 PyTorch（SAM3 使用、`uv.lock` では CUDA 12.8 版を固定）

> CPU版 PyTorch はサポート対象外です。SAM3 実行時に CUDA が利用できない場合、処理を中止します。

## 処理フロー

### 素体出力モード

```
See-Through PSD + 元画像
    ↓
SAM3 首抽出（元画像から）
    ↓
首検出確認（プロンプト変更・再検出）
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
# フロントエンド依存のインストール
npm install

# uv が未導入の場合（Windows PowerShell）
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# SAM3用Python環境の作成（uv.lockに従って .venv を作成）
uv sync --locked

# 開発モードで起動
npm run tauri dev

# リリースビルド
npm run tauri build
```

## モデルファイル

| ファイル | サイズ | 用途 | 配布 |
|----------|--------|------|------|
| `rife.onnx` | 21MB | RIFE 補間 | リポジトリに同梱 |
| `sam3.pt` | 約3.2GB | SAM3 首・口抽出（Python連携） | 別途DL |

開発時は `src-tauri/models/` に、インストール済みアプリでは `PachiPakuGen.exe` と同じ階層の `models/` フォルダに配置してください。

### SAM3 のセットアップ

1. [sam3.pt をダウンロード](https://huggingface.co/facebook/sam3)
2. `models/` フォルダに配置

```
src-tauri/models/
  sam3.pt                  <-- ここに配置
  rife.onnx                （リポジトリに同梱）
```

3. SAM3用のPython環境を作成

PachiPakuGen は SAM3 用の Python 依存を `pyproject.toml` / `uv.lock` で管理しています。リポジトリ直下で以下を実行してください。

```bash
uv sync --locked
```

`uv` が未導入の場合は、先に以下を実行してください。

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

インストール直後のGit Bashなどで `uv: command not found` になる場合は、シェルを再起動するか `C:\Users\<ユーザー名>\.local\bin` を PATH に追加してください。

これにより `.venv/` が作成され、CUDA 12.8 版の `torch` / `torchvision` と、`opencv-python` / `psutil` / `sam3` / `triton-windows` が lock されたバージョンでインストールされます。既にCPU版PyTorchを入れた `.venv/` がある場合は、`.venv/` を削除してから `uv sync --locked` を実行してください。

CUDA が有効か確認する場合:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

`torch.cuda.is_available()` が `True` にならない場合、SAM3 は実行できません。

4. アプリから使うPythonを固定する（任意）

PachiPakuGen は通常 `python` / `python3` / `py` を順に探します。`uv sync` で作成した `.venv` を確実に使う場合は、アプリ起動前に `PACHIPAKUGEN_PYTHON` へPython実行ファイルを指定してください。

PowerShell例:

```powershell
$env:PACHIPAKUGEN_PYTHON = ".\.venv\Scripts\python.exe"
npm run tauri dev
```

SAM3 の `bpe_simple_vocab_16e6.txt.gz` は公式リポジトリの `sam3/assets/` 配下に含まれる tokenizer 語彙ファイルです。PyPI版 `sam3` でこのファイルが見つからない場合に備えて、PachiPakuGen では `scripts/assets/` に同梱した語彙ファイルへフォールバックします。

## 使い方

### 素体出力

1. モード選択画面で「素体出力」を選択
2. See-Through PSD と元画像を読み込み
3. 首検出確認画面で SAM3 の検出結果を確認
   - デフォルトのプロンプトは `neck`
   - 必要に応じて `neck, neckwear` などのカンマ区切りプロンプトで再検出
4. Step 1/2: Hair レイヤーの並び替え・ON/OFF
5. Step 2/2: Body レイヤーの並び替え・ON/OFF
6. 出力先フォルダを選択して素体出力

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
| `rife.onnx` | [TensorStack/RIFE](https://huggingface.co/TensorStack/RIFE) / [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) / [ECCV2022-RIFE](https://github.com/hzwer/ECCV2022-RIFE) (Huang et al.) | MIT License |

同梱している `rife.onnx` は、TensorStack/RIFE で公開されている `model.onnx` を PachiPakuGen 用に `src-tauri/models/rife.onnx` として配置したものです。ONNXメタデータ上の入出力は `img0`, `img1`, `timestep` → `output` で、PachiPakuGen の補間処理から利用します。

配布物の照合用SHA256は以下です。

```text
76E4CEF9AB42FA7DD4E8F6E4ABA47462051E3FAA969E4BCA6479784FBAB0AC6F
```

### 別途取得が必要なモデル

| モデル | 元プロジェクト | ライセンス |
|--------|---------------|-----------|
| `sam3.pt` | [SAM 3](https://github.com/facebookresearch/sam3) (Meta) | SAM License |
