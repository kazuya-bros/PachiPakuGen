# SpriTalk Motion Richness Research

Checked at: 2026-07-03

## 1. Scope

PachiPakuGen が出力した SpriTalk 用素材を使い、口パク、瞬き、髪、身体の動きを
簡単な設定でリッチにするための調査メモ。

最初のゴールは実装ではなく、PachiPakuGen 内で複数方式をプレビュー比較できる状態へ
進むための候補整理とする。最終的に残った方式だけを SpriTalk のランタイムへ移植する。

現状の前提:

- 瞬きは現時点で十分自然。
- 口パクは母音リップシンク時に形状が急に切り替わり、不自然さが出る。
- 髪と身体は小さな回転や上下移動に寄っており、Live2D 的な遅れ、しなり、部位差が足りない。
- PachiPakuGen の既存出力は `body.png`、`hair.png`、`hair_back.png`、`eye/`、
  `mouth_a/` から `mouth_o/` などの透過 PNG フレーム群。
- 既存の素材生成方針は、外部生成画像をそのまま成果物にせず、See-Through と検証を通して
  SpriTalk 用パーツへ戻す。

## 2. Source Findings

### 2.1 rotejin/PuruPuruPNGTuber

ユーザーが言及した対象は `rotejin/PuruPuruPNGTuber`。
表情差分 PNG に `front-hair.png` と `back-hair.png` を足し、ローカルブラウザ上で
顔向き、髪揺れ、口パク、まばたきを調整する PNGTuber アプリ。

重要な考え方:

- 必須素材は、前髪、後ろ髪、開眼/閉眼と閉口/半開き/開口を組み合わせた表情差分 PNG。
- 全 PNG は同一キャンバス、同一座標へ揃えることが重要。
- 大まかな描画順は、背面アイテム、後ろ髪、顔・表情、顔前アイテム、前髪、前髪前アイテム、最前面アイテム。
- `default-settings.json` には顔中心、目、鼻、口、顎、首支点、髪束ラインなどの初期設定を持つ。
- ユーザーはブラウザ上で、表情、顔向き、髪揺れ、アイテム位置を微調整する。
- アプリ側にはマイク音量/デモによる口パク、自動まばたき、マウス追従、カメラ顔トラッキング、前髪/後ろ髪の揺れ、顔向き、影/ハイライト調整、PNGアイテム追加、OBS向け透過表示がある。
- コード上は顔、前髪、後ろ髪の deformer を分け、顔向きに対して前髪と後ろ髪の追従量や曲がり方を別に扱う。

PachiPakuGen/SpriTalk への読み替え:

- PachiPakuGen の `body.png`、`hair.png`、`hair_back.png`、`eye/`、`mouth_*` 構成と近い。
- 口の急変対策として、母音直切替の前に `closed/half/open` 的な開口量レイヤーを挟む
  timeline smoother が有力。
- 髪と身体は、前髪/後ろ髪を同じ transform で動かすのではなく、顔向き・首支点・髪束ラインに
  対する追従量を分けるべき。
- PuruPuru の mesh/deformer 的な考え方は、PachiPakuGen 側では `Layered Spring` の次段に置く
  `Mesh Sway` 候補として扱う。

Source:
https://github.com/rotejin/PuruPuruPNGTuber

### 2.2 Live2D Cubism

Live2D 本家の標準パラメータから、SpriTalk-lite として取り込める概念は明確。

口:

- `ParamMouthOpenY`: 0 が閉口、1 が通常の開口。
- `ParamMouthForm`: -1 が怒り口方向、+1 が笑顔口方向。
- SDK の lip-sync は、音量を 0 から 1 に正規化して口開閉へ入れる方法、モーションに
  lip-sync 情報を埋める方法、口だけを別モーションマネージャで扱う方法がある。

身体と髪:

- `ParamBodyAngleX/Y/Z`、`ParamBreath`、`ParamHairFront/Side/Back/Fluffy` が標準候補。
- Live2D 物理は、入力パラメータ、振り子モデル、出力パラメータを分ける。
- 振り子には揺れの速さ、揺れやすさ、反応時間、収束速度がある。
- 出力側には影響率、反転、スケールがあり、計算結果をパラメータへ混ぜる。
- 髪揺れ自動生成は、ワープデフォーマを対象にして、横揺れ、縦揺れ、柔らかさ、拡縮を調整する。
- 回転デフォーマは形状を保った回転に向き、ワープデフォーマはメッシュ変形に向く。

PachiPakuGen/SpriTalk への読み替え:

- いきなり Cubism モデルを作る必要はない。
- PachiPakuGen では `MouthOpenY`、`MouthForm`、`BodyAngleX/Y/Z`、`Breath`、
  `HairFront/Back` 相当の仮想パラメータを持ち、既存 PNG レイヤーへ変換する。
- SpriTalk へ移すのは、Live2D のファイル形式ではなく、Live2D の制御構造。
- 口だけ別モーションとして扱う設計は、SpriTalk の会話、表情、アイドル動作を分離しやすい。

Sources:

- Standard Parameter List:
  https://docs.live2d.com/en/cubism-editor-manual/standard-parameter-list/
- Lip-sync:
  https://docs.live2d.com/en/cubism-sdk-manual/lipsync/
- Physics:
  https://docs.live2d.com/en/cubism-editor-manual/physics-operation/
- Auto Generation of Sway Motion:
  https://docs.live2d.com/en/cubism-editor-manual/auto-generation-of-sway-motion/
- Deformers:
  https://docs.live2d.com/en/cubism-editor-manual/deformer/

### 2.3 Rhubarb Lip Sync and Preston Blair style shapes

Rhubarb Lip Sync は音声から 2D 口アニメーション情報を出す CLI。

重要な考え方:

- 6 から 9 個の口形状を使う。
- 基本形状 A から F に加え、G/H/X は任意。
- `X` は休止口、`A` は P/B/M 用の閉じ口、`B` は子音や一部母音、`C/D/E/F` は開口、
  丸口、すぼめ口などを担当する。
- TSV、XML、JSON、Moho/OpenToonz 用 dat へ出力できる。
- 英語向けの PocketSphinx と、非英語向けの phonetic recognizer を持つ。

PachiPakuGen/SpriTalk への読み替え:

- 日本語 5 母音の直切替だけでは、子音、休止、半開きが足りない。
- `rest`、`closed speaking`、`half`、`wide`、`round`、`pucker` へ抽象化し、
  PachiPakuGen の `mouth_a/i/u/e/o` へ後段で写像する方が自然になる可能性が高い。
- Rhubarb をそのまま日本語リップシンク本線にするのではなく、口形状セットと
  JSON タイムラインの考え方をプレビュー候補として使う。

Source:
https://github.com/DanielSWolf/rhubarb-lip-sync

### 2.4 Single-image portrait animation repositories

単一画像を動画や音声で動かすリポジトリは、SpriTalk へ直接載せる候補というより、
参考動画、評価用比較、将来の自動モーション抽出候補として見る。

候補:

- LivePortrait:
  - 1枚画像とドライビング動画からポートレートを動かす。
  - 高品質だが、最終出力は動画であり、SpriTalk のレイヤー構造を保持しない。
  - モーションテンプレートやリターゲット制御は参考になる。
- SadTalker:
  - 音声と1枚画像から talking face 動画を生成する。
  - 口、表情、頭部姿勢がまとまった動画になるため、パーツ単位移植には向かない。
  - 音声から頭部揺れや表情を作る研究候補としては有用。
- First Order Motion Model:
  - ドライビング動画から相対キーポイント移動を取り、ソース画像をアニメーションする。
  - レイヤーを壊しやすいため、本線ではなく比較用。
- Thin-Plate Spline Motion Model:
  - TPS による柔軟なモーション推定を使う画像アニメーション。
  - PachiPakuGen 側の軽量メッシュワープ候補を考える時の参考になる。

PachiPakuGen/SpriTalk への読み替え:

- これらは「完成動画をSpriTalkへ入れる」より、「どの程度の頭部揺れ、口の遅れ、髪揺れが
  自然に見えるか」を見る評価参照として使う。
- 採用前にライセンス、モデル配布条件、商用可否、必要 VRAM を別途確認する。

Sources:

- LivePortrait:
  https://github.com/KlingAIResearch/LivePortrait
- SadTalker:
  https://github.com/OpenTalker/SadTalker
- First Order Motion Model:
  https://github.com/AliaksandrSiarohin/first-order-model
- Thin-Plate Spline Motion Model:
  https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model

### 2.5 See-through and the rigging gap

See-through 公式は、単一アニメ画像を最大23の意味レイヤーへ分解し、擬似深度と描画順を
推定する。これは PachiPakuGen の素材入口として非常に近い。一方、公式READMEは
「Image-to-Live2D」そのものではない理由として、より細かい芸術的なレイヤー分解、
変形メッシュ、物理パラメータ、モーションカーブ、全体の芸術的意図が別途必要だと整理している。

PachiPakuGen/SpriTalk への読み替え:

- PachiPakuGen の担当は、See-through 後の「軽量リグ候補を比較する場所」になる。
- 既存の `body.png`、`hair.png`、`hair_back.png` だけでは Live2D の細かい動きには足りないが、
  最初のプレビューではこの3パーツへ transform / spring / simple warp を掛けるだけでも比較価値がある。
- See-through のコミュニティ欄では PachiPakuGen 自体も SpriTalk 用素材生成ツールとして言及されているため、
  今回の調査は See-through 後段の自然な発展として扱える。
- StretchyStudio のような「PSDから自動リグ」系は注視対象。ただし、現時点では
  PachiPakuGen/SpriTalk へ直接依存させるのではなく、レイヤー配置と自動リグUIの比較対象に留める。

Sources:

- See-through:
  https://github.com/shitagaki-lab/see-through
- See-through paper:
  https://arxiv.org/abs/2602.03749
- ComfyUI-See-through:
  https://github.com/jtydhr88/ComfyUI-See-through
- CubismPartExtr:
  https://github.com/shitagaki-lab/CubismPartExtr

### 2.6 Advanced LivePortrait and audio-driven portrait tools

LivePortrait の派生・周辺ツールは、表情編集や音声駆動の観点で参考になる。

Candidate notes:

- ComfyUI-AdvancedLivePortrait:
  - 写真の表情編集、動画への表情挿入、複数表情からのアニメーション作成、サンプル写真からの表情抽出を扱う。
  - 2024-08-21 更新では、動画なしで動画を作る、ソース動画の顔追跡、リアルタイムプレビューが示されている。
  - `Motion index = Changing frame length : Length of frames waiting for next motion` という時間制御は、
    PachiPakuGen のプレビュータイムライン設計に近い。
- AniPortrait:
  - 参照ポートレートと音声から高品質アニメーションを作り、動画による face reenactment も可能。
  - audio2pose と head pose control の考え方は、SpriTalk の会話中の頭部/身体揺れ候補として参考になる。
- MuseTalk:
  - 顔領域 256 x 256 の latent inpainting で、多言語音声に対応するリアルタイム寄りの lip-sync。
  - 日本語を含む多言語対応と 30fps+ の記述は魅力的だが、出力は動画顔領域であり、
    SpriTalk の透過PNGレイヤー契約へ直接戻せない。

PachiPakuGen/SpriTalk への読み替え:

- これらは「候補を直接移植」ではなく、「自然なタイムライン設計」「表情/口/頭部姿勢の分離」
  「参考動画との横並び評価」に使う。
- PachiPakuGen の本線は、素材レイヤーを保持できる Live2D-lite / sprite timeline 系に置く。

Sources:

- ComfyUI-AdvancedLivePortrait:
  https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait
- AniPortrait:
  https://github.com/Zejun-Yang/AniPortrait
- MuseTalk:
  https://github.com/TMElyralab/MuseTalk

## 3. Candidate Approaches for PachiPakuGen Preview

### A. Sprite/viseme timeline smoother

Problem addressed:

- 母音が `a -> i -> u` のように直接切り替わる時の急変。

Core idea:

- 口を「選択中の母音画像」ではなく、連続パラメータとして扱う。
- `openY`、`formX`、`roundness`、`speakerClosed`、`rest` のような内部値を作る。
- 母音ターゲットが変わっても、攻撃時間、戻り時間、最小保持時間、ヒステリシス、
  先読みブレンドでなめらかに遷移させる。
- 画像は既存 `mouth_a/i/u/e/o` と `mouth_closed` を使い、短いクロスフェードか
  RIFE フレーム列内の位置を連続的に選ぶ。

Preview controls:

- `attack_ms`
- `release_ms`
- `hold_ms`
- `vowel_crossfade_ms`
- `rest_bias`
- `mouth_open_gain`
- `shape_smoothing`

Pros:

- SpriTalk へ移植しやすい。
- 既存素材をそのまま使える。
- PachiPakuGen のプレビューだけで評価できる。

Cons:

- 母音画像同士の形状差が大きい場合、クロスフェードだけでは二重口に見える可能性がある。
- 音声認識またはテキストタイミングがない場合、母音推定は別課題。

Recommended status:

- 最初に作るべき候補。

### B. RIFE transition graph

Problem addressed:

- 母音間の画像差が大きく、クロスフェードではなく「途中の口形状」が欲しいケース。

Core idea:

- `closed -> a` だけでなく、`a -> i`、`a -> u`、`i -> u` などの母音間遷移を
  RIFE で事前生成する。
- ランタイムでは現在形状と次形状に応じて、該当する遷移クリップを短く再生する。
- すべての 6 x 6 遷移を作ると重くなるため、PachiPakuGen では差が大きいペアだけを
  選んで生成する。

Preview controls:

- transition frame count
- generate only selected pairs
- snap threshold
- reverse reuse
- maximum transition duration

Pros:

- 現在の RIFE 資産を使える。
- 急激な形状変化を画像補間で直接緩和できる。

Cons:

- 遷移組み合わせが増えると出力サイズが増える。
- 透明パーツ補間の品質は、既存の premultiply/extract 経路を必ず使う必要がある。
- 遷移のたびにアニメ感が強くなりすぎる可能性がある。

Recommended status:

- A の次に、問題の強い母音ペアだけで試す。

### C. Live2D-lite layered rig

Problem addressed:

- 髪、身体、服が単純な回転と上下移動だけで硬く見える。

Core idea:

- Cubism 形式そのものではなく、標準パラメータと物理構造だけを借りる。
- `body`、`hair`、`hair_back`、任意の切り出しパッチへ、親子変換と軽量ワープをかける。
- 入力は `breath`、`baseX/Y`、`bodyAngleX/Y/Z`、`voiceEnergy`、`idleNoise`。
- 出力は髪前後、身体、肩、服、アクセサリーの個別 transform または mesh warp。
- 物理は 1 から 3 段の spring/pendulum として持ち、部位ごとに遅れを変える。

Preview controls:

- preset: calm / lively / bouncy
- breath amplitude
- body sway amplitude
- hair front delay
- hair back delay
- softness
- convergence
- wind noise
- pivot position

Pros:

- SpriTalk のレイヤー素材と相性がよい。
- Live2D より軽い。
- ユーザー設定をプリセット化しやすい。

Cons:

- mesh warp を SpriTalk へ移植する場合、レンダリング実装が必要。
- レイヤー境界や前後関係の破綻を PachiPakuGen で見つける必要がある。

Recommended status:

- 口の A/B と並行して、プレビュー専用で作る価値が高い。

### D. PuruPuru-style deformer and package approach

Problem addressed:

- PNG素材のまま、顔向き、前髪、後ろ髪、身体、アイテムの追従差を出したいケース。

Core idea:

- PuruPuruPNGTuber と同じく、同一キャンバスに揃えた PNG 素材を前提にする。
- 顔/前髪/後ろ髪を別 deformer として扱い、顔向きと髪の根元/毛先の追従差を作る。
- PachiPakuGen は、まず transform/spring で比較し、次に軽量 mesh warp を試す。
- SpriTalk 側へは `.purupuru` 互換ではなく、採用したパラメータ構造だけを移植する。

Preview controls:

- face center
- neck pivot
- front/back hair root line
- face follow strength
- tip delay
- warp strength
- body/item follow

Pros:

- PachiPakuGen の SpriTalk PNG 出力と素材契約が近い。
- 動画化せず、SpriTalk 側のレイヤー制御へ戻しやすい。
- ユーザーがブラウザ/アプリ上で微調整する思想が近い。

Cons:

- mesh warp を入れる場合、SpriTalk 側のレンダリング実装が増える。
- 顔中心、首支点、髪束ラインなどの初期推定またはUI指定が必要。
- キャラ固有補正を共通処理へ入れすぎると既存素材を壊しやすい。

Recommended status:

- Motion A の次段候補。まずは PachiPakuGen 内プレビューで `Mesh Sway` として比較する。

### E. AI portrait animation as reference

Problem addressed:

- どのくらいの頭部揺れ、口遅れ、身体揺れが自然か、目視基準が欲しい。

Core idea:

- LivePortrait、SadTalker、FOMM、TPSMM などで同じキャラクターを動かし、
  PachiPakuGen の軽量方式と横並び比較する。
- 採用するのは動画そのものではなく、動きの周波数、振幅、遅れ、部位差の観察結果。

Preview controls:

- reference video import
- side-by-side compare
- optional curve extraction

Pros:

- 目標品質を説明しやすい。
- 実装前に「豊かさ」の方向性を掴める。

Cons:

- レイヤー保持しない。
- 依存関係が重い。
- ライセンスと配布条件の確認が必要。

Recommended status:

- 調査・評価用。SpriTalk への直接移植候補ではない。

## 4. Proposed Preview Lab in PachiPakuGen

新しい画面を `Motion Preview Lab` として切り出す。

Inputs:

- `04_spritalk_parts` または同等の PachiPakuGen 出力フォルダ。
- 任意の音声ファイル、またはテスト用の母音タイムライン。
- 任意の参考動画。

Preview lanes:

1. Baseline
   - 現在の SpriTalk 相当の口パク、瞬き、髪身体揺れ。
2. Lip A: timeline smoother
   - 既存口フレームをなめらかに選択。
3. Lip B: RIFE transition graph
   - 母音間補間を使う。
4. Motion A: layered spring
   - PNG レイヤーへ transform と遅れを適用。
5. Motion B: mesh sway
   - hair/body に軽量 warp を適用。
6. Motion C: PuruPuru-style deformer
   - 前髪/後ろ髪/顔の追従差と軽量meshの比較用。

Metrics:

- mouth target changes per second
- mouth jerk score
- maximum per-frame bbox movement
- visible alpha popping count
- generated file size
- runtime CPU/GPU cost
- settings count
- SpriTalk export compatibility
- manual adjustment time

UI requirement:

- 1画面で方式を切り替えて同じ素材を比較できること。
- 設定はプリセットから始め、詳細値は折りたたむ。
- 最終採用前提なので、各方式の設定を `motion-preview-manifest.json` に保存する。

## 5. Recommended First Experiment Order

1. Lip A: timeline smoother
   - 最小実装で現在の最大問題に直撃する。
   - テスト用タイムラインは `closed -> a -> i -> u -> e -> o -> closed`。
   - 目標は「急変が減るが、発音形状の識別は残る」こと。
2. Lip B: RIFE transition graph
   - A でまだ不自然な母音ペアだけ生成する。
   - 最初は `a <-> i`、`a <-> u`、`i <-> u` を優先。
3. Motion A: layered spring
   - `hair`、`hair_back`、`body` だけで、Live2D-lite の遅れと振り子を確認する。
   - rotation、translation、scale まで。mesh は次段。
4. Motion B: mesh sway
   - hair/front/back の下端が遅れるワープを試す。
   - Live2D の `softness`、横揺れ、縦揺れ、拡縮に対応する値を持つ。
5. Motion C: PuruPuru-style deformer
   - `front-hair/back-hair` 的な根元固定と毛先遅れを、PachiPakuGen の `hair/hair_back` に写像する。

## 6. Current Decision

現時点で単一の正解には絞らない。

PachiPakuGen の次作業は、以下の3方式を同じ画面で比べられるプレビュー仕様へ落とすこと。

- Lip A: timeline smoother
- Lip B: RIFE transition graph
- Motion A: layered spring

PuruPuruPNGTuberを参考にした変形案、See-through後段の自動リグ系、AI portrait animation は、
評価参照および将来候補として残す。
SpriTalk へ移植するのは、PachiPakuGen のプレビューで素材破綻、設定量、実行コスト、
自然さを比較した後に残った方式だけにする。

次の仕様化対象は `docs/MOTION_PREVIEW_LAB_SPEC.md` とする。
