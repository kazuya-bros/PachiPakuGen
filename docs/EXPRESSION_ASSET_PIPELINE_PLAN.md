# Single-Image Expression Asset Pipeline Plan

> [!NOTE]
> この文書は表情素材パイプライン検討時の履歴資料です。SAM3を補助利用する案を含みますが、v0.4.0の現行7STEP制作フローではSAM3を使用しません。現行仕様はREADMEを参照してください。

## 1. Goal

PachiPakuGen に、開眼・開口状態のアニメ調キャラクター画像を1枚入力すると、
以下を生成できる新しいワークフローを統合する。

- 目、口、眉を除去した `base.png`
- 透過差分パーツ
  - `eyes-open`
  - `eyes-closed`
  - `mouth-closed`
  - `mouth-a`
  - `mouth-i`
  - `mouth-u`
  - `mouth-e`
  - `mouth-o`
- RIFE で生成した、各差分パーツの中間フレーム

元画像1枚だけを新規制作の開始入力とする。PachiPakuGenへSee-Throughを内蔵し、
元画像から素体候補、目、眉、前髪レイヤーを半自動で構成する。
ベース画像を無確認で確定する完全自動化は行わず、自動構成結果を人間が確認・補正して
承認した `base.png` を後続工程の固定入力とする。

本計画は技術選定前の承認用文書である。技術選定は同一入力・同一評価基準で
プロトタイプを比較した後に確定する。

## 2. Product Boundary

統合先は PachiPakuGen とする。

- PachiPakuGen から再利用するもの
  - Tauri 2 + React 19 のデスクトップアプリ基盤
  - 内蔵See-Through成果物を確認・補正する `素体出力` と `See-Through補正`
  - RIFE ONNX + DirectML の補間処理
  - SAM3 Python subprocess 連携
  - 透過パーツの premultiply、補間、alpha 抽出処理
  - 母音別フォルダとフレーム出力規約
- `a-o_diff_maker` から再利用するもの
  - FACS ベースの口形状プリセット
  - 口角、口サイズ、追加指示による微調整
  - 参照画像
  - 対象差分の選択と個別再生成
  - Codex へ渡す生成依頼文の組み立て
- 原則として再利用しないもの
  - 現在の `a-o_diff_maker` Web UI
  - 完成画像同士を広いマスクで合成する現在のパイプライン
  - PachiPakuGen アプリ内から OpenAI / Gemini API を直接呼ぶ生成ルート
- 今回の技術選定対象外とするもの
  - See-Through 出力を人間の確認なしで `base.png` として確定する処理
  - 生成AIによる `base.png` の自動インペイントを必須工程にする処理

## 3. Input Contract

### Required

- 開眼・開口状態のキャラクター画像1枚

最終製品では、元画像の読み込み後に内蔵See-Throughを実行し、自動構成した
`base.png` 候補、目・眉・前髪レイヤーを同じワークフロー内で確認する。
必要な箇所だけ手動マージ・補正し、`base.png` の承認後に差分生成へ進む。
STEP2 のプロトタイプではベース作成方式を比較せず、用意済みの `base.png` を入力した。

差分パーツの抽出は、口も目も See-Through レイヤーを主力にする。外部生成した
あ・い・う・え・お・閉口・閉眼画像をそれぞれ See-Through に通し、元画像PSDへ
戻せる目/口レイヤーだけを採用する。SAM3 と差分マスクは、位置ゲート、検査、失敗時の
補助に下げる。どちらも自動処理に失敗した場合は手動マスクやレイヤーマッピングで
上書きできる設計にする。

### Optional

- 目や口内デザインを参照する追加画像
- 口の編集・抽出マスクの手動上書き
- 目のレイヤーマッピングまたは目マスクの手動上書き
- 髪、首、チョーカー、襟、アクセサリー等を絶対に変更しないための保護マスク
- 差分ごとの微調整
  - FACS
  - 口角
  - 口サイズ
  - 自然言語の追加指示
- 使用する生成エンジン

## 4. Part Ownership

ベース画像から目、口、眉を除去するため、差分パーツの責務を次のように固定する。

| Part | Contains |
|---|---|
| `base.png` | 目、口、眉以外の全要素 |
| `eyes-open` | See-Through が分離した左右の開いた目と眉。前髪は含めない |
| `eyes-closed` | 外部閉眼差分を See-Through で分離した左右の閉じた目と眉。前髪は含めない |
| `mouth-*` | 外部口差分を See-Through で分離した対象の口のみ |

`eyes-open` は原画像の See-Through レイヤーから抽出する。`eyes-closed` と
`mouth-*` は外部生成された完成差分画像を See-Through に通し、目または口レイヤーだけを
抽出する。前髪は常に別レイヤーとして目より上に合成する。差分画像そのものは成果物にせず、
元画像PSD由来のレイヤー構成へ部品として戻す。

## 5. Proposed Pipeline

```text
source.png
  |
  +-- embedded See-Through worker
  |     +-- automatic layer decomposition and mapping
  |     +-- PachiPakuGen review and correction
  |     +-- user approval: base.png
  |     +-- exact open-eye / eyebrow layers
  |     +-- separate front-hair layer
  |
  +-- eye preparation
  |     +-- eyes-open: See-Through exact layers
  |     +-- external eyes-closed image -> See-Through exact layers
  |     +-- optional SAM3 eye/eyebrow gate removes detached noise
  |     +-- front hair composited above eyes
  |
  +-- mouth preparation
  |     +-- external mouth variants -> See-Through exact layers
  |     +-- optional SAM3 mouth gate / manual layer mapping
  |     +-- reject non-mouth layers and detached noise
  |
  +-- validation
  |     +-- compare extracted parts with target ROI gates
  |     +-- inspect edge contamination, missing pixels, and layer misclassification
  |
  +-- interpolation
        +-- premultiply each endpoint onto base
        +-- RIFE interpolation
        +-- alpha extraction
```

生成モデルまたは外部ツールから返された完成画像をそのまま成果物にはしない。必ず
See-Through でレイヤー分解し、対象部位のレイヤーだけを透過差分パーツとして採用する。
差分マスク抽出は標準経路ではなく、See-Through分類が失敗した場合の診断・補助に限定する。
承認済み `base.png` は生成モデルへ再処理させず、その後の全工程で不変とする。

## 6. Prototype Candidates

### 6.1 Mask Proposal

| Candidate | Role | Prototype Decision |
|---|---|---|
| Manual mask | Auto-processing fallback and user override | Optional fallback |
| SAM3 mouth prompt | Mouth position gate and failure diagnostics | Assist |
| SAM3 eye + eyebrow prompt | Position gate, detached-noise removal, fallback locator | Assist |
| See-Through layers | Exact eye/eyebrow/mouth extraction and front-hair separation | Primary |

口と目の主力方式を See-Through に統一する。これまでの差分マスク抽出は、髪、首、
チョーカー、鼻、肌色パッチの巻き込みが残りやすかった。外部差分画像も See-Through に
通して semantic layer として回収し、うまく分類できない差分だけ手動マッピングまたは
SAM3/差分マスク補助へ落とす。

### 6.2 Semi-automatic Base Preparation

ベース作成は内蔵See-Throughの自動構成と、PachiPakuGenの既存補正機能を組み合わせた
半自動工程とする。

1. 元画像をPachiPakuGenへ読み込む
2. 内蔵See-Throughワーカーがレイヤー分解を実行する
3. レイヤー名と位置から素体、目、眉、前髪を自動分類する
4. PachiPakuGenが `base.png` 候補と各パーツを自動構成する
5. 目、口、眉の残存と、顔・髪・首・チョーカー・衣装・アクセサリーの欠損を確認する
6. 問題がある箇所だけ `素体出力` / `See-Through補正` 相当UIで修正する
7. ユーザーが `base.png` を承認する

STEP2 では、この工程を通したベース画像を評価用データセットに用意した。
STEP4で内蔵See-Through実行、自動分類、確認・補正画面への接続を実装した。
人間の承認を省略する完全自動確定は行わない。

後続工程へ渡す `base.png` は、次の条件を満たす必要がある。

- 元画像とキャンバスサイズが一致する
- 必要な透過情報を保持している
- 目、口、眉が残っていない
- 髪、顔、首、チョーカー、衣装、アクセサリーに意図しない欠損がない
- ユーザーがプレビューを確認して承認している
- 承認時の SHA-256 を `manifest.json` に保存する

差分生成とRIFE処理の終了後に再度SHA-256を検証し、承認済みベースが変更されていた場合は
成果物を不正として扱う。

### 6.3 Expression Generation

PachiPakuGen 本体は画像生成APIを直接呼ばない。生成はアプリ外の Codex セッションへ任せ、
アプリは `Codex生成素材` の依頼、受け取り、検査、See-Through分解、RIFE出力を担当する。

| Source | Role |
|---|---|
| Codex ImageGen | 標準の外部生成係。アプリが作成した `codex_request.md` に従って表情差分素材を作る |
| GPT Image / Gemini / Google AI Studio | Codex側またはユーザー側で使う候補。PachiPakuGen本体はAPIキーを保持しない |
| Qwen-Image-Edit / Illustrious / Anima 等 | 将来の任意生成手段。成果物は同じ `generated_parts/` 契約へ戻す |

生成モデルの品質だけでなく、目口レイヤーとしてSee-Throughで回収できるか、
キャラクター同一性、再現性、生成手順の説明しやすさを比較する。
API料金は PachiPakuGen 内の実行前判断ではなく、Codex依頼文や外部生成手段の注意書きとして扱う。

### 6.4 Frame Interpolation

PachiPakuGen に同梱済みの RIFE ONNX を第一候補として固定し、以下を検証する。

- 口パーツ境界での色にじみ
- 閉眼時のまつ毛や眉の変形
- 透過alphaのちらつき
- 2K以上の画像での処理時間とVRAM使用量

RIFE が評価基準を満たさない場合だけ、別補間モデルを比較対象に追加する。

## 7. Prototype Dataset

最低6枚を使用する。

| Case | Required Characteristic |
|---|---|
| A | 正面、単純な口、装飾なし |
| B | 褐色肌、鮮やかな瞳 |
| C | 口の近くにチョーカー、襟、アクセサリーがある |
| D | 目や眉に前髪が重なる |
| E | 横向きまたは顔が傾いている |
| F | 透過背景または複雑な背景 |

同一画像・同一要求で各候補を比較する。外部生成候補は確率的変動を確認するため、
重要ケースでは最低3回実行する。

## 8. Acceptance Metrics

### Mandatory

| Metric | Pass Condition |
|---|---|
| Outside-mask preservation | 許可マスク外の最終成果物ピクセル変更がゼロ |
| Approved base immutability | 承認済み `base.png` が後続工程で一切変更されない |
| Part isolation | 各透過パーツに首、髪、チョーカー、肌の不要領域が含まれない |
| Endpoint correctness | 開眼、閉眼、閉口、5母音を人間が識別できる |
| RIFE output | 中間フレームに重大な分裂、にじみ、ちらつきがない |
| Recoverability | マスク、設定、生成物を保存し、個別工程から再開できる |

### Scored

各項目を5段階評価し、候補ごとに比較表を作る。

- キャラクター同一性
- 口形状と目形状の品質
- マスク修正に必要な時間
- 生成時間
- VRAM使用量
- 外部生成に必要な手数
- 再現性
- ライセンスと配布容易性

## 9. Prototype Deliverables

STEP2 の成果物は、UIを持たない再実行可能なプロトタイプと比較報告書とする。

```text
prototype/
  inputs/
    source.png
    base.png
  masks/
  candidates/
  outputs/
  manifest.json
  evaluation.json
```

プロトタイプは以下を実行できること。

1. 1枚の元画像、See-Through レイヤー、必要なら手動上書きを読み込む
2. ユーザー承認済みの `base.png` を読み込み、変更せず保持する
3. `generated_parts/` から Codex生成素材を読み込む
4. Codex生成素材を See-Through に通し、対象の目/口レイヤーだけを透過差分パーツとして抽出する
5. PachiPakuGen の RIFE で中間フレームを作る
6. 評価用コンタクトシートと機械評価結果を出力する

## 10. Final Output Contract

```text
output/
  base.png
  parts/
    eyes-open.png
    eyes-closed.png
    mouth-closed.png
    mouth-a.png
    mouth-i.png
    mouth-u.png
    mouth-e.png
    mouth-o.png
  eye/
    001.png
    ...
  mouth_a/
    001.png
    ...
  mouth_i/
  mouth_u/
  mouth_e/
  mouth_o/
  masks/
    eyes-edit.png
    eyes-extract.png
    mouth-edit.png
    mouth-extract.png
    protect.png
  work/
    see-through-source/
    see-through-eyes-closed/
    see-through-mouth-closed/
    see-through-mouth-a/
    see-through-mouth-i/
    see-through-mouth-u/
    see-through-mouth-e/
    see-through-mouth-o/
    extracted-layer-report.json
    front-hair.png
    sam3-eye-gate.png
    sam3-mouth-gate.png
  manifest.json
```

`manifest.json` には、入力画像、承認済みベースのSHA-256、モデル、プロンプト、FACS、
マスク、費用見積もり、処理時間、出力ファイル、再生成に必要な設定を保存する。

フレーム系列の端点と順序は、既存の PachiPakuGen 規約を維持する。

- `eye/001.png` は開眼、最終フレームは閉眼
- `mouth_a/001.png` は閉口、最終フレームは「あ」
- `mouth_i`、`mouth_u`、`mouth_e`、`mouth_o` も同様に閉口から各母音へ補間

## 11. Implementation Architecture After Selection

技術選定後、PachiPakuGen に以下の境界を追加する。

```text
ExpressionPipeline
  LayerDecompositionProvider
    EmbeddedSeeThroughWorker
    ExternalPsdRecoveryProvider
  MaskProvider
    Sam3PositionGateProvider
    ManualMaskOverrideProvider
  EyeLayerProvider
    SeeThroughEyeLayerProvider
    Sam3EyeGateProvider
  MouthLayerProvider
    SeeThroughMouthLayerProvider
    Sam3MouthGateProvider
  BaseAssetProvider
    SemiAutomaticBaseProvider
    ApprovedBaseProvider
  ExternalExpressionSource
    GoogleAiStudioDiffFolder
    ApiGeneratedDiffFolder
    ManualDiffFolder
  PartExtractor
    SeeThroughDiffPartExtractor
    MaskFallbackPartExtractor
  PartValidator
  FrameInterpolator
    ExistingRifeInterpolator
```

プロバイダー固有処理を UI と RIFE 処理から分離し、後から別モデルを追加できる
構造にする。

## 12. Approval Gates

### Gate 1: Plan Approval

この文書の対象範囲、入力、出力、評価基準を承認する。

### Gate 2: Technology Approval

STEP2 の比較報告を確認し、マスク補助方式、差分生成方式、パーツ抽出方式を選択する。

### Gate 3: UI Approval

複数の DESIGN.md と画面案を比較し、実装するUIを承認する。

### Gate 4: Implementation Review

実装、テスト結果、既知の制約を確認する。

### Gate 5: User Acceptance

実画像による動作確認と修正を行う。

## 13. Decisions Requested for Gate 1

1. 統合先を PachiPakuGen とする。
2. 新規制作の必須入力は、開眼・口が確認できる立ち絵1枚だけとする。
3. See-ThroughはPachiPakuGen内蔵ワーカーとして実行し、自動構成結果を同じ画面で確認・補正する。
4. ユーザー承認済み `base.png` を後続工程で変更しない。
5. 目マスクは眉を含み、目差分パーツも眉を含む。
6. 口も目も See-Through レイヤー抽出を主力とする。SAM3 と手動入力は位置ゲート、
   失敗時補助、上書きとする。
7. `eyes-open` は原画像の See-Through レイヤーから切り出す。
8. STEP2 では UI やベース自動生成を作らず、差分生成以降の技術比較用プロトタイプを先に作った。
9. STEP4で内蔵See-Throughワーカー、自動分類、確認・補正フローを接続した。
10. 最終出力契約は本書の構造を基準とする。
