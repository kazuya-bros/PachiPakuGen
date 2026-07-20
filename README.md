# PachiPakuGen

[SpriTalk](https://kazuyabros.booth.pm/items/8102679) 向けの表情・モーション素材を、1枚の立ち絵から作るWindowsデスクトップアプリです。

PachiPakuGen v0.4.0は、表情差分の準備、[See-Through](https://github.com/shitagaki-lab/see-through)によるレイヤー分解、素体と差分パーツの補正、RIFEフレーム補完、モーション調整、マイク連動のライブ表示までを、1つの作業フォルダと7つのSTEPにまとめます。

> [!IMPORTANT]
> v0.4.0はv0.3系から制作フローを全面刷新しています。以前のREADMEにあった「素体出力／まばたき／口パク」の個別モードやSAM3前提の手順は、現行の7STEP制作フローには当てはまりません。

## v0.4.0で変わったこと

- 「はじめから」「つづきから」「ライブ表示」のスタートメニューを追加
- 立ち絵、生成物、See-Through成果物、SpriTalk向け出力を作業フォルダ単位で管理
- See-Through公式実装をアプリから自動セットアップし、検証済みコミットで実行
- Codex、ほかの画像編集AI、手作業のいずれでも表情素材を準備可能
- 立ち絵と7枚の表情素材をまとめてSee-Throughへ渡し、必要な目・口・髪・素体を自動構成
- 素体のレイヤー順、表示、腕・獣耳の分離、切り出しをアプリ内で補正
- 目・口の位置と拡大率をプレビュー上で直接調整
- RIFEで口パク・まばたき用の中間フレームを一括生成
- 揺れ、口パク、瞬き、視線、瞳、眉、髪、腕、胸、獣耳などをSTEP 7で調整
- マイク入力に反応するライブ表示とOBS向けクロマキー表示を追加
- STEP 3に、立ち絵1枚だけをレイヤー分解して獣耳・眼鏡などの抽出結果を高速確認できる「抽出ガチャ」を追加
- 表情素材が立ち絵と別解像度でも、縦横比が近ければ自動調整して受け付けるよう変更
- STEP 7で、つなぎ目のないループアニメ素材（PNG連番／APNG／AGIF）を書き出し可能に
- v0.4.0の7STEP制作フローではSAM3を使用せず、`sam3.pt`のダウンロードも不要

## 主な機能

### 表情素材を作る

立ち絵を基準に、次の7枚を用意します。

```text
eyes-closed.png
mouth-closed.png
mouth-a.png
mouth-i.png
mouth-u.png
mouth-e.png
mouth-o.png
```

アプリはCodex向けの作成ガイドと配置先を出力しますが、Codexの契約は必須ではありません。キャラクターの見た目・配置・キャンバスサイズを保てるなら、Nano Bananaなどの画像編集AI、ペイントソフト、外注素材でも進められます。

### See-Throughをアプリ内で扱う

PachiPakuGenはSee-Throughのソースコードや大型モデルをインストーラーへ同梱していません。初回セットアップ時に公式リポジトリの検証済みコミットをアプリ管理領域へ取得し、専用Python環境と必要なモデルを準備します。

- 取得元: [`shitagaki-lab/see-through`](https://github.com/shitagaki-lab/see-through)
- 固定コミット: [`e4cb250dc69defe6f982168dab684aa461552b5b`](https://github.com/shitagaki-lab/see-through/commit/e4cb250dc69defe6f982168dab684aa461552b5b)
- 実行プロファイル: `standard` / `low-vram`
- PachiPakuGenによる変更: Windows、BF16、CPU offload、推論ステージ間のVRAM解放に関する互換パッチ

初回セットアップとモデル取得は、ユーザーがボタンを押したときだけ開始します。推論中は事前取得済みモデルを使い、予期しないバックグラウンドダウンロードを行いません。

### 抽出ガチャで獣耳・眼鏡の取れ具合を確認する

獣耳や眼鏡は、Seedや解像度によってレイヤーとして抽出できたりできなかったりします。STEP 3の「抽出ガチャ」は、立ち絵1枚だけを深度推定・PSD組立を省いたレイヤー分解のみで処理し、抽出された各パーツをサムネイルで確認できる機能です。表情素材7枚を含む一括分解より大幅に短い時間で、Seedや詳細設定を変えながら納得のいく分解結果を探せます。気に入ったSeedのまま一括分解を開始すれば、その結果がそのまま使われます。

### SpriTalk向け素材を出力する

RIFE補完後の画像とモーション設定は、作業フォルダ内の `04_spritalk_parts` に集約されます。SpriTalkへ渡すときは、作業フォルダ全体ではなく、このフォルダを選びます。

`body.png`、`hair.png`、目・口のフレーム列などの基本画像は、SpriTalkのLayeredEmotion用素材として渡せる構成です。v0.4.0では、実際に生成した`04_spritalk_parts`をSpriTalk v1.1.2の`LayeredSpriteImporter`へ読み込み、基本画像と各フレーム列の検証に成功しています。

`spritalk-motion-profile.json` v2はSpriTalk側の将来連携用仕様で、v0.4.0時点ではSpriTalk本体に読み込み機能がありません。高度なモーション設定まで確実に再生できるのは、PachiPakuGenのライブ表示です。基本画像とv2プロファイルを同じ互換性として扱わないでください。

### ライブ表示する

完成素材をPachiPakuGen内でそのまま再生できます。

- マイク音量による口の開閉
- 開いた口として使う母音の選択
- 感度、ノイズゲート、開閉速度の調整
- 表示倍率の変更、ドラッグ移動、マウスホイール拡大縮小
- グリーン、ブルー、マゼンタ、ダークのOBS向け背景
- OBSのウィンドウキャプチャとクロマキーで利用できるキャプチャ表示

### ループアニメ素材を書き出す

STEP 7の調整内容のまま、つなぎ目なく繰り返せる透過アニメ素材を書き出せます。呼吸・体揺れ・髪の物理演算・まばたき・獣耳ピコピコなどの駆動を書き出し時間ぴったりの周期へそろえ、数周分ウォームアップしてから記録することで、動画編集ソフトやOBSでのループ再生に使えるシームレスな素材になります。

- ループ長（7.2 / 14.4 / 21.6秒）と、口の形（口パクなし、または「あ」〜「お」いずれか1つの開閉ループ）を選択
- 出力形式は次の3つから必要な分だけ選択
  - **PNG連番**: 最高画質。手元にffmpegがあれば案内するコマンドで透過付きWebMへ変換可能
  - **APNG**: 単一ファイルで透過ループを再生（対応ビューアが必要。ブラウザでの再生を推奨）
  - **AGIF**: 256色パレット・二値透過という制約はあるものの、最も軽量で対応ビューアが広い

## 動作要件

用途によって必要な環境が異なります。

| 用途 | 必要な環境 |
|---|---|
| 新しい立ち絵をSee-Throughで分解 | Windows 10/11 x64、NVIDIA CUDA対応GPU、Git、uv、PowerShell、インターネット接続 |
| 既存の作業フォルダを補正・RIFE補完 | Windows 10/11 x64。DirectML対応GPUを推奨。GPUが使えない場合はCPUへフォールバックしますが低速です |
| モーション調整・ライブ表示 | Windows 10/11 x64。ライブ口パクにはマイク入力が必要です |

See-Throughの目安は次のとおりです。画像サイズやGPU構成により実使用量は変わります。

| プロファイル | GPUメモリの目安 | 空き容量の目安 | 特徴 |
|---|---:|---:|---|
| `standard` | 16GB以上 | 約22GB | 高速。16GB以上のGPUでは自動選択 |
| `low-vram` | 8GB前後から | 約14GB | CPU offloadと量子化モデルでVRAM使用量を削減 |

Hugging Faceのモデルは匿名でも取得できますが、レート制限を避けるため無料のreadトークン設定を推奨します。トークンはWindows資格情報へ保存され、取得プロセスの環境変数にだけ渡されます。

## インストール

1. [GitHub Releases](https://github.com/kazuya-bros/PachiPakuGen/releases)からv0.4.0のNSISインストーラーを取得します。
2. インストーラーを実行します。
3. 新規制作を行う場合は、STEP 3でSee-Throughのランタイムとモデルを準備します。

現時点のWindowsインストーラーにはコード署名がありません。Windows SmartScreenが警告を表示した場合は、配布元とリリースページに掲載されたSHA-256を確認してから実行してください。

## 7STEP制作フロー

| STEP | 工程 | 内容 |
|---:|---|---|
| 1 | 画像選択 | 必須の立ち絵と、任意の参照画像を選択 |
| 2 | 表情素材 | Codex、画像編集AI、手作業のいずれかで7枚の表情素材を用意 |
| 3 | See-Through | 立ち絵と表情素材を一括分解し、必要なパーツを抽出 |
| 4 | 素体調整 | レイヤー順、表示、腕・獣耳分離、切り出しを確認・補正 |
| 5 | 差分位置調整 | 目・口の位置と拡大率を全表情で確認・補正 |
| 6 | RIFE補完 | 中間フレームを生成し、`04_spritalk_parts` へ集約 |
| 7 | モーション調整 | 口パク、瞬き、各部の揺れを調整して保存、またはライブ表示へ移動 |

`project.json` に進捗を保存するため、スタートメニューの「つづきから」で前回の作業フォルダを選ぶと途中から再開できます。

## 作業フォルダ

```text
<作業フォルダ>/
├─ project.json                 # 進捗と入力情報
├─ manifest.json                # PachiPakuGen自身の再開用内部状態（STEP6生成）
├─ motion-preview-manifest.json # PachiPakuGen自身の再開用内部状態（STEP7保存時に生成）
├─ 01_codex_request/            # 表情素材の作成ガイドと元画像
├─ 02_generated_parts/          # 用意した7枚の表情素材
├─ 03_see_through/              # See-ThroughのPSD・中間成果物
└─ 04_spritalk_parts/           # SpriTalkへ渡す完成素材
   ├─ body.png / hair.png / ...
   ├─ eye/                      # まばたき用RIFEフレーム列
   ├─ mouth_a/ ... mouth_o/     # 口閉じから各母音へ補完したRIFEフレーム列
   ├─ layer-order.json          # STEP 7で書き出すとspritalk-motion-profile.jsonへ統合され消える
   ├─ README.txt                # 同上（書き出し前のみ存在）
   ├─ spritalk-motion-profile.json  # STEP 7で書き出した場合に生成。layerOrder/readmeを含む
   └─ loop_export/                 # STEP 7でループ素材を書き出した場合に生成
      ├─ frames/                   # PNG連番（「PNG連番」を選んだ場合のみ残る）
      ├─ loop.png                  # APNG（「APNG」を選んだ場合）
      └─ loop.gif                  # AGIF（「AGIF」を選んだ場合）
```

`manifest.json`・`motion-preview-manifest.json`はPachiPakuGen自身が再開・再編集のために使う内部状態で、SpriTalkは読みません。STEP 7で「SpriTalk向けに保存」を実行すると、`04_spritalk_parts`内の`layer-order.json`と`README.txt`は`spritalk-motion-profile.json`へ統合されて削除され、SpriTalkへ渡すフォルダの成果物ファイルは`spritalk-motion-profile.json`1本（＋画像・フレーム列）になります。

ファイルを個別に移動すると再編集や再開ができなくなる場合があります。制作中は作業フォルダをまとめて保持し、SpriTalkへの取り込み時だけ `04_spritalk_parts` を指定してください。

## v0.3系からの移行

- v0.3系のPSDやPNGは、v0.4.0の作業フォルダへ自動変換されません。
- 新規制作は「はじめから」で作業フォルダを作成してください。
- 既に `04_spritalk_parts` 相当の完成素材がある場合は、STEP 7または「ライブ表示」から素材フォルダとして読み込めます。
- v0.4.0の標準制作フローはSAM3を使用しません。`sam3.pt`のダウンロードやSAM3用Python環境の構築は不要です。

## 開発

### 必要なもの

- Windows 10/11
- Node.js 20.19以上
- Rust stable
- Git
- uv（See-Throughの初回セットアップを試す場合）

### コマンド

```powershell
npm install

# 開発モード
npm run tauri -- dev

# フロントエンドのビルド
npm run build

# Rustのテスト
Push-Location src-tauri
cargo test
Pop-Location

# NSISインストーラー
npm run tauri -- build
```

Rust/Cargoが見つからない場合は、[rustup](https://rustup.rs/)でRustを導入し、新しいPowerShellを開いてから再実行してください。

## データとネットワーク

- 選択した立ち絵と生成素材は、ユーザーが指定した作業フォルダで処理します。
- See-Throughの公式コード、Python依存、モデルは、ユーザーがSTEP 3の準備操作を行ったときに各配布元から取得します。
- Hugging FaceトークンはWindows資格情報へ保存します。リポジトリ、ログ、コマンド行には書き出しません。
- Codexやほかの画像編集AIへ素材を渡す場合、そのサービスの利用規約と送信データの扱いは各サービス側の条件に従います。

## ライセンスと謝辞

PachiPakuGenの独自部分は[MIT License](LICENSE)です。第三者のコード、モデル、翻案・着想元にはそれぞれのライセンスが適用されます。詳細と変更内容は[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)を参照してください。

v0.4.0のモーション実装には、次のプロジェクトを参照して変更・適応した処理と、設計上の着想を得た処理があります。

- [Anime2.5DRig](https://github.com/852wa/Anime2.5DRig) — 852話氏。髪房の検出・二重バネをPachiPakuGen向けに変更・適応し、パララックス設計を参考
- [PuruPuruPNGTuber](https://github.com/rotejin/PuruPuruPNGTuber) — ウェーブ式の周期・位相設計をCanvas描画向けに変更・適応し、遅延追従などを参考

両プロジェクトの名称や作者名は謝辞・由来の説明であり、PachiPakuGenの共同著作権者、推奨者、提携先であることを意味しません。両プロジェクトのサンプル画像やキャラクター素材はPachiPakuGenへ収録していません。

See-ThroughはApache License 2.0で公開されている公式実装を取得し、PachiPakuGenが実行時に互換パッチを適用します。See-Throughおよび取得される各モデルの条件についても第三者通知を確認してください。

## 既知の制約

- See-Throughによる新規分解にはNVIDIA CUDA対応GPUと大容量の初回ダウンロードが必要です。
- AI分解と表情素材の品質によっては、STEP 4・5で手動補正が必要です。
- `spritalk-motion-profile.json` v2のSpriTalk側読み込みは未実装です。全効果を確認できるのは、v0.4.0時点ではPachiPakuGenのライブ表示です。
- RIFEのCPUフォールバックは利用できますが、生成には時間がかかります。
- 自動アップデーターとWindowsコード署名は未対応です。

## 関連文書

- [変更履歴](CHANGELOG.md)
- [第三者ソフトウェア・モデルの通知](THIRD_PARTY_NOTICES.md)

SpriTalk向けのモーション仕様・実装ハンドオフ資料は設計検討時の内部資料のため、`docs/_local-archive/`（非トラッキング）にのみ保管しています。
