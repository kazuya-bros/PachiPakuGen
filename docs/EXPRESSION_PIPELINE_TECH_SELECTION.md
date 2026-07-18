# Expression Asset Pipeline STEP2 Technology Selection

> [!NOTE]
> この文書は技術選定時の調査記録です。SAM3を使う候補案は採用されておらず、v0.4.0の現行7STEP制作フローではSee-Throughと画像差分を使用します。

## 1. Gate 2 Recommendation

STEP2 の推奨構成は次の通り。

| Stage | Selected approach |
|---|---|
| Layer decomposition | See-ThroughをPachiPakuGen管理の内蔵Pythonワーカーとして実行し、PSD・PNG群を既存解析処理へ直接渡す |
| Base preparation | See-Through成果物を自動分類・自動構成し、既存の `素体出力` / `See-Through補正` 相当UIで必要箇所だけ修正。ユーザー承認済み `base.png` を固定 |
| Mouth preparation | Codex生成素材を See-Through に通し、口レイヤーだけを抽出。SAM3/手動マスクは位置ゲートと失敗時補助 |
| Eye preparation | See-Through の正確な目・眉レイヤーを主力とし、前髪を別レイヤーで保持。閉眼差分も See-Through で抽出 |
| Mask roles | See-Through レイヤー抽出を主力にし、編集・抽出・保護マスクは検査と補助に分離 |
| External generation | アプリ外の Codex ImageGen に依頼し、PachiPakuGen本体はAPIキーを保持しない |
| Generated parts folder | `generated_parts/` を標準フォルダ名、表示名を `Codex生成素材` とする |
| Part extraction | Codex生成素材を See-Through で分解し、対象の目/口レイヤーだけを透過パーツ化 |
| Validation | 承認済みベースの SHA-256、対象ROI外成分、レイヤー分類失敗を機械検証 |
| Interpolation | PachiPakuGen 既存の RIFE ONNX + DirectML を継続利用 |

Codex ImageGen やGoogle AI Studioなどの完成画像をそのまま成果物にはしない。外部生成系は
「差分候補を描く装置」として扱い、最終成果物の所有範囲は See-Through レイヤー抽出と
検証処理で保証する。差分マスク抽出は標準経路ではなく、See-Through分類の補助・診断に
限定する。

## 2. Prototype Deliverables

UI を持たない再実行可能なプロトタイプを追加した。

- `prototype/expression_assets/run_prototype.py`
  - 生成済み完成画像から透過差分パーツを抽出
  - 編集マスク、抽出マスク、保護マスクを個別指定
  - 承認済みベースの SHA-256 不変性を検証
  - 生成元と最終合成のマスク外変更を測定
  - コンタクトシート、`evaluation.json`、`manifest.json` を出力
- `src-tauri/examples/rife_parts.rs`
  - 2つの透過差分パーツを既存RIFEへ直接渡し、中間フレームを生成
- `prototype/expression_assets/tests/test_pipeline.py`
  - 保護領域、編集マスク外不変性、抽出マスク、ベース不変性を自動検証
- `prototype/expression_assets/sam3_mask_study.py`
  - SAM3 の口コアから非対称エンベロープ、膨張、ぼかしを生成
- `src-tauri/examples/psd_expression_layers.rs`
  - See-Through PSD から目・眉と前髪を別々に抽出
- `prototype/expression_assets/see_through_eye_study.py`
  - See-Through の目レイヤーを SAM3 の拡張位置ゲートで検査し、孤立ノイズだけを除去
- `prototype/expression_assets/see_through_diff_import_study.py`
  - 外部差分画像を一括で See-Through に通し、目/口レイヤーだけを元画像PSD座標へ戻す
  - 分類失敗、レイヤー位置ずれ、髪・首・鼻・チョーカー混入を評価する

## 3. Prototype Results

### 3.1 Important limitation

今回利用できた診断データは、承認済みの目・口・眉なしベースではなく、
`base.png = source.png` としている。したがって、最終的なパーツ品質の合否ではなく、
汚染検出、抽出、検証、RIFE接続の技術検証として評価した。

実装段階では、PachiPakuGen で手動作成した本物の承認済みベースを含む最低6ケースで
再評価する必要がある。今回の口自動マスクは1キャラクターの GPT / Nano 出力、目の
See-Through ハイブリッドは別の1キャラクターで検証した段階であり、量産適性は未確定。

### 3.2 GPT Image 2 / Nano Banana diagnostic assets

| Check | Result |
|---|---|
| Approved base immutability | GPT / Nano とも SHA-256 不変 |
| Final composite outside edit mask | 全差分で変更ピクセル数 0 |
| GPT source outside-mask drift | `mouth-i`: 12,668、`mouth-u`: 12,665、`mouth-o`: 12,679 ピクセル |
| Nano source outside-mask drift | 診断ファイルでは 0。既存処理で合成済みの可能性があり、API自体の保持性能とは断定しない |
| Wide-mask part isolation | 不合格。肌、髪、チョーカー片が一部混入 |

この結果から、生成モデルの「他を変えない」という指示だけでは不十分だと判断する。
最終合成のマスク外変更ゼロはローカル処理で保証できたが、広い編集マスクや差分マスクだけでは
パーツ内部への不要要素混入を防げなかった。以後の標準経路は、外部差分画像を
See-Through で再分解して対象部位レイヤーだけを採用する方式へ変更する。

### 3.3 Three mask roles

| Mask | Purpose | Typical shape |
|---|---|---|
| Edit mask | 最大開口や閉眼を生成するための余白をAPIへ与える | 広い |
| Extraction mask | 最終パーツへ取り込める領域を制限する | 狭い |
| Protect mask | 髪、首、チョーカー、襟等を明示的に除外する | 必要箇所だけ |

口を開くため編集マスクは広く取る必要がある。一方、その広さをそのまま切り出しに使うと
首やチョーカーまで差分へ混入する。三役の分離はオプションではなく必須設計とする。

診断画像へ狭い口抽出マスクを追加して再実行したところ、全口差分で
`part_outside_extraction_mask_pixels = 0` と
`composite_outside_mask_changed_pixels = 0` を維持しながら、髪、チョーカー、鼻の混入を
除去し、5母音の形状を保持できた。抽出マスクはキャラクターごとにユーザーがプレビューを
見ながら修正できる必要がある。

### 3.4 SAM3 + automatic mouth envelope

MouthSpriteExtractor-SAM3 の実装と、口位置を基準にした下方向寄りの楕円、
鼻側の上端制限、膨張、Gaussian feather の設計を確認した。SAM3 の生マスクへ
膨張・収縮と feather を適用する。

この考え方を使い、SAM3 の小さな口コアから自動マスクを作成した。

| Check | Result |
|---|---|
| Selected mouth extraction envelope | 左右16px、上20px、下36px |
| Mouth target recall | 5母音・閉口に対して 1.0 |
| Mouth target precision | 0.6812 |
| Manual broad mask outside pixels | 0 |
| Edit mask | 抽出マスクを12px膨張 |
| Feather | 15px |
| GPT / Nano final part outside extraction mask | 全差分 0 |
| GPT / Nano final composite outside edit mask | 全差分 0 |

広い手動マスクで抽出した GPT `mouth-a` と比較すると、alpha ピクセルは
5,253 から 3,826、マスク内再構成 MAE は 10.72 から 1.44 へ改善した。
髪、鼻、チョーカーの混入をある程度除去しつつ口形状を保持できた。ただし、後続検証では
口周辺の髪、肌色パッチ、首・後ろ髪の変化が残ったため、**口でもSAM3自動マスクを主力にはしない**。
SAM3 は See-Through で抽出した口レイヤーの位置ゲート、分類失敗時の候補表示、
手動補正の初期値として使う。

### 3.5 See-Through eye layers + SAM3 gate

目は SAM3 の輪郭を拡張するより、See-Through が分離した `eyewhite`、`irides`、
`eyelash`、`eyebrow` を正確な目パーツとして使う方が安定した。`front_hair` と
`headwear` は別パーツとして目より上に戻す。

SAM3 の `eye + eyebrow` を36px膨張した領域は、閉眼候補を生成する編集領域と、
See-Through の目レイヤーに含まれる孤立成分を除去する位置ゲートに使う。SAM3 だけで
作った目の抽出マスクは前髪と肌を巻き込みやすく、最終パーツ抽出には採用しない。

| Check | Result |
|---|---|
| Open eye | 10,782 / 10,782 alpha pixels retained |
| Closed eye | 3,834 / 3,933 alpha pixels retained |
| Closed-eye noise removal | 孤立2成分、99pxを除去 |
| Front hair | 目とは別レイヤーとして正常に再合成 |
| RIFE | 1024 x 1024、4フレーム、5.79秒、端点完全一致 |

したがって、**目では See-Through を主力、SAM3 を位置ゲート・ノイズ除去・失敗時の
位置推定にする**。レイヤー分類に失敗した場合だけ手動マッピングまたは手動マスクを使う。

### 3.6 RIFE

既存 `rife.onnx` を DirectML + CPU セッションで実行した。

- 1254 x 1254、4フレーム: 約8.43から10.13秒
- 先頭・末尾フレームは入力透過パーツとピクセル単位で完全一致
- 中間フレーム生成自体は成功
- 入力パーツに混入していた肌、髪、チョーカー片も補間された
- 狭い抽出マスクを適用した再実行では、髪とチョーカー片は中間フレームから除去できた
- 診断条件が `base.png = source.png` のため、口周辺の肌パッチは残る。本物の承認済み
  ベースで再評価する

RIFE は継続採用する。問題は補間方式ではなく、RIFEへ渡す前のパーツ分離品質にある。

## 4. Candidate Comparison

| Candidate | Editing / mask fit | Identity retention | Local / distribution | STEP2 decision |
|---|---|---|---|---|
| Codex ImageGen | アプリ外のCodexセッションで利用。PachiPakuGen本体にAPIキーを置かずに表情差分素材を作れる | 標準候補。生成結果は完成品にせずSee-Throughへ渡す | External | Primary |
| GPT Image 2 / Nano Banana Pro | Codex側またはユーザー側で使う生成候補。PachiPakuGen本体は直接呼ばない | 高い候補。ただしマスク外ドリフトを実測 | External | Optional |
| Qwen-Image-Edit-2511 | ローカル編集候補。位置ずれリスクを明記 | 評価が必要 | Apache 2.0。24GB GPUでは量子化等の追加検証が必要 | Experimental |
| Illustrious + ControlNet/Inpaint | アニメ生成には強いが、単体で編集パイプラインにならない | 構成依存 | ローカル構成と追加モデルが必要 | MVPから除外 |
| Anima | テキスト生成中心で編集用途に不向き | 未評価 | モデルカード上、商用利用禁止 | 除外 |
| SAM3 | 口/目の位置ゲート、分類失敗時の補助、手動補正の初期値に有用 | N/A | 既存ローカル環境で動作 | Assist |
| See-Through | 正確な目・眉・前髪・口の意味レイヤー抽出に有用 | 入力レイヤーを保持 | 既存ワークフロー | Primary |
| RIFE | 既存透過パーツ補間と統合済み | 入力品質を継承 | MIT、既存同梱 | Keep |

## 5. External Generation Guidance

PachiPakuGen の画面では API キー入力やAPI課金見積もりを通常導線へ出さない。
アプリは Codex へ渡す `codex_request.md` と `generated_parts/` を作る。
`eyes-open` は原画像から抽出するため、Codex生成対象は `eyes-closed` と6口差分の計7枚。

過去比較として、2026-06-14 時点の公開価格による出力画像だけの概算は次の通り。
これはアプリ内の課金UIではなく、Codex依頼文や外部生成手段の注意書きへ必要に応じて載せる。

| Provider / quality | Per image | 7 generated images |
|---|---:|---:|
| GPT Image 2, 1024x1024 medium | USD 0.034 | USD 0.238 |
| GPT Image 2, 1024x1024 high | USD 0.133 | USD 0.931 |
| Nano Banana Pro, up to 2K | USD 0.134 | USD 0.938 |
| Nano Banana Pro, 4K | USD 0.24 | USD 1.68 |

入力画像・テキストトークン、再生成、為替、税は別途発生する。

## 6. Implementation Rules Selected in STEP2

1. 新規制作の必須入力は、開眼・口が確認できる立ち絵1枚だけとする。
2. See-ThroughはPachiPakuGen内蔵ワーカーとして実行し、通常版・省VRAM版を環境に応じて選ぶ。
3. ベース作成は自動分類・自動構成後に、必要箇所だけ既存の手動マージ・補正を使う半自動工程とする。
4. 承認済みベースは外部生成へ渡しても、成果物として上書きしない。
5. 口差分は Codex生成素材を See-Through に通し、口レイヤーだけを抽出する。
6. 目は See-Through の目・眉レイヤーを使い、前髪を別レイヤーとして上から合成する。
7. SAM3 の口/目マスクは位置ゲート、分類失敗検出、手動補正の初期値に使い、最終抽出は
   原則 See-Through レイヤーで行う。
8. 保護マスクは補助抽出時に抽出マスクより優先し、該当ピクセルを必ず透明にする。
9. Codex生成素材は必ず See-Through レイヤー抽出・検証を通す。
10. `eyes-open` は元画像の See-Through レイヤーから抽出し、再生成しない。
11. RIFEは抽出・検証済みパーツだけを入力とする。
12. マスク、モデル、プロンプト、FACS、費用、ハッシュ、評価値をmanifestへ保存する。

## 7. Gate 2 Approval Items

STEP3へ進む前に、次の構成を承認対象とする。

1. 立ち絵1枚を唯一の新規制作入口にする。
2. See-Throughを内蔵ワーカーとして半自動ベース作成へ接続する。
3. 人間が承認したベースを固定入力にする。
4. 口も目も See-Through レイヤー抽出を主力とし、外部差分画像をそのまま成果物にしない。
5. SAM3 と手動指定は位置ゲート、分類失敗時の補助、上書きに使う。
6. 編集・抽出・保護の三種類のマスクを維持する。
7. アプリ内API生成ルートは本線から外し、Codex生成素材の受け渡しを第一候補にする。
8. Qwen-Image-Edit-2511はMVPの必須機能にせず、実験機能として保留する。
9. ローカル抽出・検証と既存RIFEを正式パイプラインにする。

## 8. Official References

- OpenAI Image Generation:
  https://developers.openai.com/api/docs/guides/image-generation
- OpenAI API Pricing:
  https://developers.openai.com/api/docs/pricing
- Gemini Image Generation:
  https://ai.google.dev/gemini-api/docs/image-generation
- Gemini API Pricing:
  https://ai.google.dev/gemini-api/docs/pricing
- Qwen-Image:
  https://github.com/QwenLM/Qwen-Image
- Qwen-Image-Edit-2511:
  https://huggingface.co/Qwen/Qwen-Image-Edit-2511
- SAM3:
  https://github.com/facebookresearch/sam3
- PuruPuruPNGTuber:
  https://github.com/rotejin/PuruPuruPNGTuber
- MouthSpriteExtractor-SAM3:
  https://github.com/kazuya-bros/MouthSpriteExtractor-SAM3
- RIFE:
  https://github.com/hzwer/ECCV2022-RIFE
- Anima:
  https://huggingface.co/circlestone-labs/Anima
- Illustrious XL early release:
  https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0
