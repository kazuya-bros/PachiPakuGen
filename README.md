# PachiPakuGen

[SpriTalk](https://kazuyabros.booth.pm/items/8102679) 専用の素材生成ツールです。

**RIFE（Real-Time Intermediate Flow Estimation）** によるフレーム補間で、滑らかな口パク・まばたきアニメーションを実現します。新規制作は立ち絵1枚から開始し、内蔵See-Throughで分解したPSDを既存のレイヤー解析・補正処理へ直接渡します。

## 特徴

- **内蔵See-Through** -- 立ち絵1枚からアプリ内で分解し、生成PSDを自動分類
- **分解結果の承認・補正** -- 自動分類をプレビューし、必要な時だけ既存のレイヤー補正画面で調整
- **外部PSD復旧経路** -- 既存のSee-Through PSDも引き続き直接読み込み可能
- **SAM3 による口検出** -- SAM3 は口パク補間の口領域抽出にのみ使用（素体出力・See-Through補正では使用しません）
- **素体出力** -- body / hair / hair_back の3パーツを透過PNGで出力。レイヤー並び替え・ON/OFF・切り出し対応
- **See-Through補正** -- PSDに含まれる全レイヤーと外部PNGを手動で並び替え・切り出し・保存
- **RIFE フレーム補間** -- RIFE はまばたき・口パクの中間フレーム生成にのみ使用（素体出力・See-Through補正では使用しません）
- **口パク5母音対応** -- 口閉じ / 口あ〜お の最大5ペアを一括生成
- **GPU 前提の高速推論** -- ONNX Runtime 2.0 + DirectML と CUDA 版 PyTorch によるGPU推論

## なぜ See-Through + SAM3 か

[See-Through](https://github.com/shitagaki-lab/see-through)（[ComfyUIノード](https://github.com/jtydhr88/ComfyUI-See-through)）は1枚のイラストから髪・顔・服などのセマンティックレイヤーをPSDとして自動分解するAIです。Qwen-Image-Layered と比べて低VRAMで動作するため、SpriTalk のレイヤーモード用素材が格段に作りやすくなりました。

ただし See-Through 単体では素材として使う際に以下の問題があります。

- **首・手などの重なり**: 1つのレイヤー内でも、部分ごとに正しい前後関係が異なる場合がある
- **口**: mouth レイヤーの検出範囲が広く、口パク補間時にノイズが入りやすい

PachiPakuGen は Body 編集や See-Through補正で任意レイヤーの一部をマスクで切り出し、別レイヤーとして並び替えられるようにすることで、首・手・服などの前後関係を調整できます。また、口パク補間では **SAM3（Segment Anything Model 3）** を併用し、元画像から口領域を切り出します。

> **注意**: 口パク・まばたきの生成は GPU 搭載環境での利用を前提にしています。SAM3 は口パクの口検出、RIFE は口パク・まばたきの中間フレーム生成にのみ使用します。素体出力と See-Through補正では SAM3/RIFE を実行しません。

## 動作要件

- Windows 10/11
- DirectX 12 対応 GPU（必須）
- NVIDIA GPU + CUDA 対応 PyTorch（内蔵See-Through・口パクのSAM3口検出時は必須）
- Node.js 18+
- Rust 1.75+
- Python 3.12+（口パクのSAM3口検出時に使用）
- CUDA版 PyTorch（口パクのSAM3口検出時に使用、`uv.lock` では CUDA 12.8 版を固定）

> 内蔵See-Throughは専用のPython 3.12環境をアプリ管理領域へ作成します。初回セットアップ時に依存パッケージ、最初の解析時にモデルをダウンロードします。既存PSDの素体出力・See-Through補正だけを使う場合、内蔵See-Through環境は不要です。

## 処理フロー

### 表情セット作成

```
開眼・口が確認できる立ち絵1枚
    ↓
内蔵See-Through（GPU検出 → 通常版 / オフロード版 / 量子化版）
    ↓
生成PSDを既存パーサーへ直接読込・自動分類
    ↓
分解プレビューを確認・承認（必要時だけ高度なレイヤー補正）
    ↓
差分設定・生成確認
```

初回セットアップと解析は明示ボタンからのみ開始します。解析中は進捗表示とキャンセルが利用でき、成果物はアプリのローカルデータ領域へ保持されます。

### 素体出力モード

```
See-Through PSD
    ↓
Hair レイヤー編集（並び替え・ON/OFF）
    ↓
Body レイヤー編集（マスク切り出し・並び替え・ON/OFF・透明度確認）
    ↓
出力: body.png / hair.png / hair_back.png
```

このモードでは SAM3 と RIFE は実行しません。

### See-Through補正モード

```
See-Through PSD
    ↓
必要に応じて中間素材PNGを追加（例: head.png）
    ↓
全レイヤーから必要なものだけON/OFF・並び替え・切り出し
    ↓
任意ファイル名でPNG保存
```

このモードでは SAM3 と RIFE は実行しません。

### まばたき・口パク フレーム補間モード

```
閉じ PSD  ↔  開き PSD
    ↓
口パク時のみ: 閉じ元画像 ↔ 開き元画像
    ↓
SAM3 口抽出（元画像から、口マスク余白を調整可能）
    ↓
RIFE 中間フレーム生成
    ↓
出力: eye/001.png〜 / mouth/001.png〜 / mouth_a/001.png〜 ...
（eye: 001 = 開き、N = 閉じ / mouth: 001 = 閉じ、N = 開き）
```

RIFE はこのフレーム補間モードでのみ使用します。SAM3 は口パク選択時の口検出にのみ使用し、まばたきでは使用しません。

## 入力素材

新規の表情セット作成では、開眼・口が確認できる立ち絵1枚だけが必須です。既存の個別ツールでは、See-Throughで分解済みのPSDを引き続き入力できます。

| 入力 | 説明 | 必須 |
|------|------|------|
| 立ち絵 | 開眼・口が確認できる元画像 | 表情セット作成 |
| 参照画像 | 瞳デザイン・口内色の任意参照 | 任意 |
| PSD | See-Through出力のPSDファイル | 既存個別ツールのみ |
| 元画像 | See-Throughに入力した元のイラスト画像 | 口パク補間のみ |

### See-Through レイヤー対応

| マッピング | レイヤー |
|-----------|---------|
| body (固定) | face, neck, nose, topwear, bottomwear 等 |
| eye (固定) | irides, eyewhite, eyelash, eyebrow (L/R対応) |
| mouth (固定) | mouth |
| hair (調整可) | front_hair, headwear |
| hair_back (調整可) | back_hair |

`See-Through補正` では上記マッピングに関係なく、PSDに含まれる全レイヤーを対象にできます。See-Throughの中間素材には存在するがPSDに出てこない `head.png` なども、外部PNGとして追加できます。

## インストール

### 内蔵See-Through

表情セット作成のSTEP 2で `初回セットアップ` を押すと、検証済みの公式See-Throughコミットと専用Python環境をアプリ管理領域へ準備します。GPUメモリに応じて、既定では次のプロファイルを自動選択します。

| GPUメモリ | 自動プロファイル |
|-----------|------------------|
| 16GB以上 | standard |
| 10GB以上 | group-offload |
| 10GB未満 / 未検出 | low-vram |

See-Through公式実装のライセンスはApache-2.0です。接続先は検証済みコミットへ固定しています。

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
| `sam3.pt` | 約3.2GB | SAM3 口抽出（Python連携） | 別途DL |

開発時は `src-tauri/models/` に、インストール済みアプリでは `PachiPakuGen.exe` と同じ階層の `models/` フォルダに配置してください。

`rife.onnx` はまばたき・口パクのフレーム補間でのみ使用します。`sam3.pt` は口パクの口検出でのみ使用します。素体出力・See-Through補正だけを使う場合、これらの推論処理は実行されません。

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

モード選択画面では以下を選べます。

| モード | 用途 |
|--------|------|
| 素体出力 | SpriTalk用の `body.png` / `hair.png` / `hair_back.png` を作成 |
| See-Through補正 | See-Through PSDや中間PNGを手動補正して任意PNGとして保存 |
| まばたき | `eye/001.png` 以降のまばたきフレームを生成 |
| 口パク mouthのみ | `mouth/001.png` 以降の口パクフレームを生成 |
| 口パク あ〜お | `mouth_a/` 〜 `mouth_o/` の母音別口パクフレームを生成 |

### 素体出力

1. モード選択画面で「素体出力」を選択
2. See-Through PSD を読み込み
3. Hair レイヤーを確認
   - `front_hair` / `headwear` / `back_hair` のON/OFFと並び順を調整
4. Body レイヤーを確認
   - レイヤーのON/OFF、並び替え、表示透明度を調整
   - `全ON` / `全OFF` で表示を一括変更
   - `切出` で任意範囲を塗り、別レイヤーとして切り出し
   - 切り出しパッチは元レイヤーから抜かれ、任意の順序へ移動可能
   - `重なり表示` で複数レイヤーが重なる部分を確認
5. `出力して確認へ` を押して出力
6. 完了画面で `出力フォルダを開く` または `モード選択へ`

出力先には以下が作成されます。

```text
body.png
hair.png
hair_back.png
```

### See-Through補正

1. モード選択画面で「See-Through補正」を選択
2. See-Through PSD を読み込み
3. 必要に応じて `PNG追加` から外部PNGを追加
   - 例: See-Throughの中間素材 `head.png` を `head` レイヤーとして追加
   - PSDに含まれない素材を救済したい場合に使います
4. レイヤーを調整
   - `全OFF` → 必要なレイヤーだけON、の流れで単体出力ができます
   - `front_hair` だけ、`headwear` だけ、切り出した耳だけ、など任意の組み合わせで保存可能
   - `切出` で一部だけを別レイヤー化し、元レイヤーから抜けます
5. `PNG保存` で任意のファイル名を指定して保存
6. 保存後は `出力フォルダを開く` / `再保存` / `モード選択へ` を選べます

このモードは自動候補検出を行いません。See-Throughの分類結果を人間が補正するための手動モードです。

### まばたき

1. モード選択画面で「まばたき」を選択
2. 閉じ目PSD / 開き目PSD を設定
3. フレーム数を指定して `中間フレーム作成`
4. 生成プレビューで確認
5. 完了画面で `出力フォルダを開く` または `モード選択へ`

出力例:

```text
eye/
  001.png  # 開き
  002.png
  ...
  008.png  # 閉じ
```

### 口パク mouthのみ

1. モード選択画面で「口パク mouthのみ」を選択
2. 閉じ口PSD / 開き口PSD を設定
3. それぞれの `元画像` を設定
   - 元画像は See-Through に渡した元イラストです
   - SAM3 はPSD合成画像ではなく元画像から口領域を抽出します
4. `口マスク作成` を実行
5. 右側のプレビューで口マスクを確認
   - `余白` と `ぼかし` を調整可能
   - 作り直したい場合は作成済み行の `×` で破棄して再作成
6. フレーム数を指定して `中間フレーム作成`
7. 生成プレビューで確認

出力例:

```text
mouth/
  001.png  # 閉じ
  002.png
  ...
  008.png  # 開き
```

### 口パク あ〜お

1. モード選択画面で「口パク あ〜お」を選択
2. `あ / い / う / え / お` のうち、作成したい母音ペアを設定
   - 最低1組あれば生成できます
   - 各母音ごとに閉じPSD / 開きPSD / 閉じ元画像 / 開き元画像を設定
3. `口マスク作成` を実行
   - 入力済みペアの口マスクを一括作成
4. 右側のタブで各母音の口マスクを確認
5. フレーム数を指定して `中間フレーム作成`
6. 生成プレビューで確認

出力例:

```text
mouth_a/
  001.png  # 閉じ
  ...
  008.png  # 開き
mouth_i/
mouth_u/
mouth_e/
mouth_o/
```

未入力の母音フォルダは生成されません。

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
