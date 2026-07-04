> 本書はSpriTalkリポジトリ（ブランチ claude/busy-haibt-144d8f）で作成された設計の正本。
> 本ブランチ（motion-preview-lab）実装との対応は [motion-lab-integration.md](motion-lab-integration.md)、
> 各リポジトリへのハンドオフは [handoff-spritalk.md](handoff-spritalk.md) / [handoff-webvoiceanimator.md](handoff-webvoiceanimator.md) を参照。

# アニメーション技術仕様 — A1調音結合 / B3メッシュ髪揺れ / 腕揺れ＋肩の弾み

本書はSpriTalk本体への移植を**別の実装者（AIモデルを含む）に引き継ぐ前提**で、
採用技術の決定経緯・仕様・参照実装・移植方針をまとめたものである。

関連: [reference/animation-lab/README.md](../reference/animation-lab/README.md)（使い方・比較方法）

---

## 0. 決定経緯（なぜこの構成に至ったか）

1. **課題**: PachiPakuGenのRIFE補間口パクが不自然 / See-Through分離レイヤーが
   髪・顔以外未活用で体のアニメーションが単調（2026-07-03 検討開始）
2. **原因分析**: 現行実装は母音が変わるたびに口を閉じて開き直す（調音結合の欠如）が
   不自然さの主因候補。RIFE品質・等速ステップ送りは副次要因
3. **比較検証**: `reference/animation-lab/` を構築し、口パク4方式（A0現行/A1調音結合/
   A2クロスフェード/A3メッシュモーフ）×体髪4方式（B0現行/B1遅延バネ/B2発話反応/
   B3メッシュ髪揺れ）を同一素材・同一モーラタイミングで並列比較
4. **ユーザー評価（2026-07-04）**:
   - **口パク: A1採用**（既存RIFE連番のまま最も自然。A3メッシュモーフは採用見送り・保留）
   - **髪: B3採用**（B1の遅延バネを多段化した効果が明確）
   - **顔向き軸: ノイズドリフト＋バネ頷きの駆動源が好評** → 既定として採用
   - **腕揺れ（角度チェーン）採用**。さらに「肩も一緒に弾む方が良いのでは」の
     指摘を受け肩の弾み（lift）を追加検証 → **採用**
5. **確定した含意**: PachiPakuGen（指示書生成→Codex→手動修正→RIFE→出力）は
   **一切変更不要**。素材追加は arm_l/arm_r の分離のみ。
   本体側の変更は「再生ロジック（A1）＋描画基盤（B3/腕のメッシュ）＋設定UI」

### 採用ステータス一覧

| 技術 | 判定 | 参照実装（reference/animation-lab/js/） |
|------|------|--------------------------------------|
| A1 調音結合 | **採用** | 20-mouth-methods.js（A1）、11-mora-player.js（エンベロープ） |
| A4 開度エンベロープ | **採用**（A1の駆動信号） | 11-mora-player.js |
| B3 メッシュ髪揺れ | **採用** | 30-body-methods.js（B3物理）、31-hair-mesh-b3.js（描画） |
| 腕揺れ＋肩の弾み | **採用** | 32-arm-sway.js |
| 顔向き軸（ノイズ/バネ頷き） | **採用**（軸素材は今後作成） | 12-axis-scrubber.js、23-axis-demos.js |
| B1 遅延バネ | 軽量フォールバックとして保持 | 30-body-methods.js（B1） |
| A2 クロスフェード | 見送り | — |
| A3 メッシュモーフ | 保留（口内表現が必要になったら再検討） | 21-mouth-mesh-a3.js、22-mesh-editor.js |

---

## 1. 技術の系譜（どこから来た手法か）

| 技術 | 源流 | animation-labでの位置づけ |
|------|------|--------------------------|
| 遅延追従バネ | ろてじん氏 PuruPuruPNGTuber（前髪/後ろ髪の遅延追従） | B1として実装・実証 |
| 多段角度チェーン＋メッシュ | Live2D物理演算（振り子チェーン→デフォーマ）/ Inochi2D / Spine系2Dボーン | B1の遅延バネを多段直列化したもの＝**B3** |
| 調音結合（母音間で閉じない） | 音声学の知見＋uLipSync系（開度パラメータの連続駆動） | **A1** |
| 開度エンベロープ（attack/release） | ADSR（シンセ）/ uLipSyncのSmoothDamp | A4（A1の駆動信号） |
| 焼き込みパラメータ軸 | Live2Dのパラメータ操作モデルを連番画像で代替する本プロジェクト独自の整理 | 軸スクラブ基盤 |

比較結果の含意: 「**遅延が生きて見える鍵**」（ろてじん氏の洞察）は正しく、
多段化（B3）でさらに効果が上がることが実証された。

---

## 2. A1: 調音結合リップシンク

### 2.1 原理

現行方式（A0）は母音が変わるたびに「閉じフレームまで戻す→新レイヤーで開き直す」。
実際の発話は母音間で口を閉じない（調音結合）ため、これがパクパク感の主因。

A1は**開度（openness ∈ 0..1）を母音レイヤーと独立に保持**し:

1. 母音が変わったら**開度を維持したままmouth_レイヤーだけ瞬時に差し替える**
2. フレーム番号は `round(openness × 最終フレーム番号)` で決める
3. 開度はA4エンベロープ＋SmoothDampで連続駆動する

### 2.2 A4開度エンベロープ

```
母音別目標開度: a=1.0 / o=0.85 / e=0.65 / i=0.5 / u=0.45 / N(ん)=0.1 / pause=0
cl(っ)・モーラ間: 直前の目標を維持

指数スムージング:
  tau = (目標 > 現在) ? attackMs : releaseMs     // 立ち上がり速く・閉じゆっくり
  openness += (目標 - openness) × (1 - exp(-dt/tau))
既定値: attack=40ms, release=90ms
```

その上でA1自身が `smoothDamp(openness, 目標, smoothTime=0.05s)` を重ねる
（エンベロープを「矩形」に切り替えると現行相当の駆動を再現できる）。

### 2.3 擬似コード（本体移植の中核）

```
update(dt, now, moraState):
  raw = vowelToMouthLayer(moraState.vowel)      // 既存ユーティリティ
  if raw not in {keep, null}:
    currentLayer = raw                           // ← 閉じずに即差し替え（A0との唯一の本質差）
  openness = smoothDamp(openness, moraState.openness, smoothTime, dt)
  frame = round(openness × (frameCount(currentLayer) - 1))
  表示: currentLayer の frame
```

### 2.4 移植方針

- **素材・PachiPakuGen: 変更ゼロ**。既存のRIFE連番をそのまま使う
- 変更箇所は `src/windows/character/components/CharacterApp.tsx` の
  `moraLipSyncLoop`（LayeredEmotion分岐）のみ
- エンベロープ生成は再生器側（音声開始時刻基準のrAFループ内）に置く
- 参照実装: `reference/animation-lab/js/20-mouth-methods.js`（A1）、
  `js/11-mora-player.js`（エンベロープ）

### 2.5 パラメータ（設定UI公開候補）

| パラメータ | 既定 | 範囲 | 意味 |
|-----------|------|------|------|
| attackMs | 40 | 5〜250 | 口が開く速さ |
| releaseMs | 90 | 10〜400 | 口が閉じる速さ |
| smoothTime | 0.05 | 0.01〜0.3 | 開度追従の滑らかさ |

---

## 3. B3: メッシュ髪揺れ（多段角度チェーン）

### 3.1 原理

hair.png を縦N分割（既定6）したメッシュにし、各行にバネ角θを持たせる:

```
θ_target[i] = (i == 0) ? drive + wind : θ[i-1]   // 親行に追従。1行分の伝播遅延が波を生む
k_i = k_root × (1 - 0.6 × i/N)                    // 毛先ほど柔らかい
ω[i] += (-k_i × (θ[i] - θ_target[i]) - c × ω[i]) × h   // 半陰的オイラー
θ[i] += ω[i] × h
θ[i] = clamp(θ[i], ±0.5rad)

drive = -頭のX速度 × 追従係数（頭が右へ動くと髪は左へ流れる）
wind  = sin波 ＋ 1Dノイズの常時微風
数値安定化: dt ≤ 100ms クランプ、dt > 1/60 でサブステップ2分割
```

### 3.2 描画（メッシュ変形）

行0（根元）固定。行iの基準点は「行i-1の基準点＋累積回転した段ベクトル」で
折れ線状に下り、行内の頂点は横オフセットを累積回転で回す。
pixi.js v8 では `MeshPlane({verticesX:4, verticesY:N+1})` の
`geometry.getBuffer('aPosition')` を毎フレーム書き換えて `buffer.update()`。

### 3.3 移植方針と技術判断

B3のみWebGL相当の描画基盤が必要。選択肢:

| 案 | 内容 | 評価 |
|----|------|------|
| (a) pixi.js導入 | npm依存追加（約450KB）。実装最短・品質最高 | Electronではサイズ誤差。**推奨** |
| (b) Canvas 2D三角形ワープ | MPNGモードの`drawWarpedSprite`（setTransform+clip）を多段化 | 依存ゼロ。髪1枚×縦6分割なら縫い目はほぼ不可視。負荷はCPU |

いずれの場合も**物理計算部（角度チェーン）は共通**で、
`reference/animation-lab/js/30-body-methods.js`（B3）が参照実装。
描画は `js/31-hair-mesh-b3.js`。
軽量フォールバックとしてB1（レイヤー全体の遅延バネ＝PuruPuru方式）を残す設計も可。

### 3.4 パラメータ

| パラメータ | 既定 | 範囲 | 意味 |
|-----------|------|------|------|
| segments | 6 | 3〜12 | 縦分割数 |
| k（根元剛性） | 70 | 10〜200 | 硬さ。低いほど遅延大 |
| c（減衰） | 7 | 1〜30 | 揺り戻しの収まり |
| wind | 0.012 | 0〜0.06 | 常時微風 |
| drive（頭追従） | 0.03 | 0〜0.2 | 体の動きへの反応量 |

---

## 4. 腕揺れ＋肩の弾み（待機モーション拡張・採用）

### 4.0 経緯

腕揺れ（角度チェーン）を実装・検証した際、「腕の回転だけでは肩が体に
固定されたままで、体が弾んでも腕がぶら下がって見える。肩も一緒に弾む方が
良いのでは」というユーザー指摘があり、**肩の弾み（lift）**を追加検証した。
結果、回転＋liftの組合せが採用となった。原理はB1/B3で実証済みの
「遅延追従が生きて見える」の縦方向への応用である。

### 4.1 目的と方針

呼吸で体がtranslateしても腕が胴体に焼き付いたままだと「一枚板」感が残る。
See-Throughは遮蔽部を補完済みでレイヤー分解するため、腕を分離しても
背後の胴体は既に描かれている＝**素材面の障害はない**。

段階設計のうち**段階2（B3と同じ角度チェーンを腕に適用）まで**を実装する。
腕ポーズ差分（GPT-Image2による軸化＝段階3）は待機モーションには過剰なので行わない。

### 4.2 素材規約（追加）

```
output/
  arm_l.png   任意（左腕、全キャンバスサイズの透過PNG）
  arm_r.png   任意（右腕）
```

レイヤー順: hair_back(-1) < body(0) < **arm_l/arm_r(1)** < eye(2) < mouth(3) < hair(4)

### 4.3 肩ピボットの自動推定

腕画像の**不透明ピクセルのバウンディングボックスを走査し、その上端中央を
肩ピボットとする**。設定不要で動く（手動オーバーライドは将来の設定UIで）。

### 4.4 物理と描画

- 物理: B3と同一の角度チェーン（既定3分割、肩固定・手先ほど柔らかい）。
  駆動 = 頭のX速度カップリング＋微小ノイズ。左右で位相をずらす
- **肩の弾み（lift）**: 体の上下動（呼吸・バウンス）のY速度に肩がバネ遅延で追従し、
  発話開始時に撃力（既定26px/s）で「ぽよん」と弾む。両肩共通の1本のバネ、
  最大±6px。回転のみ（OFF）と比較検証できるようトグルを用意
- 振幅は小さく（最大角 ±0.12rad ≈ ±7°を既定上限。継ぎ目対策）
- 描画:
  - DOMコンポジター: ピボット中心の**剛体回転**（チェーン角の平均を適用）＝段階1相当
  - pixiコンポジター: bbox切り出しテクスチャの**メッシュチェーン**＝段階2
- 体方式（B0〜B3）から独立したコントローラとして実装し、どの組合せでも併用可能

### 4.5 パラメータ

| パラメータ | 既定 | 意味 |
|-----------|------|------|
| enabled | on | 腕揺れ有効 |
| segments | 3 | チェーン分割数（pixi時） |
| k / c | 90 / 10 | 剛性・減衰 |
| coupling | 0.02 | 頭追従量 |
| noise | 0.008 | 常時微揺れ |
| maxAngle | 0.12 | 最大角（肩継ぎ目対策の上限） |
| liftEnabled | on | 肩の弾み有効 |
| liftCoupling | 0.08 | 体の上下動への追従量 |
| liftBounce | 26 | 発話開始の撃力（px/s） |
| liftMax | 6 | 肩上下の最大（px） |

---

## 5. SpriTalk移植UI案

> **重要（2026-07-05改訂）**: 本節の詳細スライダーUIは**PachiPakuGen側の
> チューニング画面仕様**として読み替えること。SpriTalk本体のUIは§9.1のとおり
> 3項目（ON/OFF・動きの強さ・口パク方式）に縮小し、詳細パラメータは
> アニメーションプロファイル（§9.2）で素材側に焼き込む。

### 5.1 配置

キャラクター設定（キャラクター中心設計に従う）の「アニメーション」セクションに
以下の3グループを追加。ライブプレビュー（設定変更の即時反映）は既存機構を利用。

```
キャラクター設定 > アニメーション
├─ 口パク
│   ├─ 方式: [従来（ステップ送り）| 調音結合（推奨）]
│   ├─ 開く速さ attack (ms)        [スライダー 5-250]
│   └─ 閉じる速さ release (ms)     [スライダー 10-400]
├─ 髪揺れ
│   ├─ 方式: [sin波（従来）| 遅延バネ（軽量）| メッシュ物理（推奨）]
│   ├─ 柔らかさ（剛性k・低いほど揺れる）
│   ├─ 収まり（減衰c）＋[自然（臨界）]ボタン
│   ├─ 風の強さ / 体への追従
│   └─ 分割数（メッシュ時のみ表示）
└─ 腕揺れ
    ├─ 有効 [トグル]（arm_l/arm_r 読込時のみ表示）
    ├─ 揺れ幅（maxAngle）
    ├─ 柔らかさ / 収まり
    └─ 肩の弾み [トグル] ＋ 弾み量 / 発話バウンス強度
```

### 5.2 保存データ構造案（ProceduralAnimationSettings拡張）

```ts
interface AnimationSettingsV2 {
  lipSync: {
    mode: 'step' | 'coarticulation';   // 既定 'coarticulation'、'step'は後方互換
    attackMs: number;                   // 40
    releaseMs: number;                  // 90
  };
  hairPhysics: {
    mode: 'sine' | 'spring' | 'mesh';  // sine=現行 / spring=B1 / mesh=B3
    segments: number;                   // 6
    stiffness: number;                  // 70
    damping: number;                    // 7
    wind: number;                       // 0.012
    drive: number;                      // 0.03
  };
  armSway: {
    enabled: boolean;
    stiffness: number;                  // 90
    damping: number;                    // 10
    maxAngle: number;                   // 0.12
  };
}
```

既存 `ProceduralAnimationSettings`（breathing/idleSway/hairSway）は温存し、
`hairPhysics.mode==='sine'` のとき従来動作＝完全後方互換とする。

> **改訂（§9適用後）**: 上記の詳細値はアニメーションプロファイル（§9.2）へ移動。
> SpriTalk側の保存データは最終的に
> `{ animationEnabled: boolean, motionScale: number, lipSyncMode: 'coarticulation'|'step' }`
> の3項目＋プロファイル参照のみとなる。

### 5.3 サンプルでの実証

animation-labの「🎛 移植UI案」ボタンで上記UIを実際に操作できる
（値はライブでプレビューセルに反映）。実装: `reference/animation-lab/js/52-port-ui.js`

---

## 6. 本体移植の実装チェックリスト（実装引き継ぎ用）

実装者は以下の順で進めること。各項目の参照実装は§0の表を参照。

### Phase A: A1調音結合（依存追加なし・最優先）

1. `src/windows/character/components/CharacterApp.tsx` の `moraLipSyncLoop` の
   LayeredEmotion分岐（L437-541相当）を A1ロジックに差し替える
   - 内部状態: `openness`（0..1）と `currentLayer` を保持
   - §2.3の擬似コードに従う。`getVowelAtTime`/`vowelToMouthLayer` は既存を流用
2. A4エンベロープ（§2.2）を同ループ内に実装（`NS.VOWEL_OPENNESS` 相当の
   母音別目標開度テーブルを `src/shared/utils/vowel-timing.ts` に追加）
3. 後方互換: `lipSync.mode === 'step'` で従来ロジックを維持（切替可能に）
4. 検証: animation-labを隣に並べ、同一テキスト・同一素材で見た目が一致すること

### Phase B: B3メッシュ髪揺れ＋腕揺れ（描画基盤の判断が必要）

1. 描画基盤を決定（§3.3）: pixi.js導入（推奨） or Canvas 2D三角形ワープ
2. 物理計算部を移植（`30-body-methods.js` B3 / `32-arm-sway.js` は
   コンポジター非依存の純ロジックなのでほぼコピー可能）
3. 髪: `31-hair-mesh-b3.js` 相当のメッシュ描画。hair_back はB1相当の回転追従
4. 腕: 肩ピボット自動推定（不透明bbox上端中央、`02-utils.js` の `alphaBBox`）、
   剛体回転＋lift（DOM/Canvasの場合）またはメッシュチェーン（WebGLの場合）
5. `ProceduralAnimator`（`src/shared/animation/procedural-animator.ts`）との統合:
   `hairPhysics.mode === 'sine'` で既存動作を完全維持（後方互換必須）
6. 数値安定化を必ず入れる: dt≤100msクランプ / dt>1/60でサブステップ2分割 /
   角度・liftのclamp（§3.1・§4.4の値）

### Phase C: 設定UI＋永続化

1. `AnimationSettingsV2`（§5.2）を `src/shared/types/store.ts` に追加
   （既存 `ProceduralAnimationSettings` は温存）
2. 設定画面に§5.1のUI（実装済みモックアップ: `52-port-ui.js`。
   animation-labの「🎛 移植UI案」ボタンで操作感を確認できる）
3. ライブプレビュー（設定変更の即時反映）は既存機構に接続
4. 腕揺れセクションは arm_l/arm_r 読込時のみ表示

### 実装時の注意（ハマりどころ）

- pixi.js v8は `await app.init()` 必須（v7と異なり同期コンストラクタ不可）
- MeshPlaneの頂点書き換えは `geometry.getBuffer('aPosition')` → `buffer.update()`
- 発話中の揺れ減衰（既存 `speechDamping`）とB3の頭追従駆動は干渉しうる。
  animation-labでは「体の揺れ幅減衰はB0仕様のまま、髪・腕は速度駆動」で両立させた
- 肩の弾みは両肩共通の1本のバネ（左右独立にすると不自然）。
  腕の回転チェーンは左右でノイズ位相をずらす（完全同期は機械的に見える）

## 7. 素材要件（確定分）

1. A1は既存RIFE連番をそのまま使用 — **PachiPakuGen変更不要**
2. 腕揺れは `arm_l.png` / `arm_r.png` の分離のみ — See-Through出力から切り出し
   （PachiPakuGenの素体出力に腕分離オプションを足すのが自然な拡張）
3. A3（メッシュモーフ）を将来採用する場合のみ、キーフレームに口周辺の肌を含める
   要件とMorphData JSON（README参照）が発生する — 今回は採用見送りのため保留

---

## 8. 将来候補と組み込み準備（PachiPakuGen／SpriTalk 役割分担）

2026-07-05 の調査（852話氏 Anime2.5DRig を含む）に基づく次期拡張の準備資料。
**素材工場＝PachiPakuGen / ランタイム＝SpriTalk** の役割分担で、
どちらに何を実装するかを実装引き継ぎ可能な粒度で定義する。

### 8.0 参考実装・ライセンス一覧

| 参照 | 内容 | ライセンス |
|------|------|-----------|
| [Anime2.5DRig](https://github.com/852wa/Anime2.5DRig)（852話氏） | PSD自動リグ: アンカー自動検出（虹彩中心・まぶた・口・首）、レイヤー深度テーブルによるパララックス＋シアー首振り、髪の房自動検出（毛先輪郭ピーク、最大6房/レイヤー）＋房ごと2重バネ、**胸揺れ**、WebGL1メッシュワープ | **MIT**（コード参照・流用可） |
| [PuruPuruPNGTuber](https://github.com/rotejin/PuruPuruPNGTuber)（ろてじん氏） | 前髪/後ろ髪の遅延追従揺れ（**採用済みB1/B3/腕揺れの源流原理**）、顔向き調整、目元演出（ハイライト・涙レンズ・影）、自動瞬き | **Apache-2.0**（帰属表示＋変更明示でコード参照・流用可） |
| [See-Through](https://github.com/shitagaki-lab/see-through) | 最大23レイヤー分解（眉・白目/虹彩・衣服・アクセサリ含む）＋深度マップ＋マスク | リポジトリ参照 |
| [THA3](https://github.com/pkhungurn/talking-head-anime-3-demo) / [THA4](https://pkhungurn.github.io/talking-head-anime-4/) | 1枚絵＋45パラメータ（表情39・回転6）でポーズ画像生成。軸連番のオフライン焼き込み工場候補 | リポジトリ参照 |

方針: Anime2.5DRig の口パクは簡易表現（差分切替＋音量）のため採用しない。
**口＝SpriTalk方式（A1モーラ同期）、首・視線・房物理＝852話方式**のいいとこ取りとする。

### 8.1 候補と役割分担マップ（優先順）

| # | 候補 | PachiPakuGen（素材） | SpriTalk（ランタイム） | 優先 |
|---|------|---------------------|----------------------|------|
| 0 | **登場・退場演出＋息継ぎ**（§8.6。SpriTalk特性に直結） | 追加素材**不要** | 登場撃力＋物理リセット＋位相ランダム化、pause_mora駆動の息継ぎ | ★★★ |
| 1 | レイヤー深度パララックス首振り | 追加素材**不要**（既存レイヤーのみ）。任意で depth 係数を meta.json 出力 | 深度係数×首振りパラメータの水平シフト＋シアー | ★★★ |
| 2 | 汎用揺れパーツ | `sway_*.png` の分離出力（指示書テンプレに「リボン・ネクタイ等を個別レイヤーで」追加） | 実装済みチェーン物理を sway_* に汎用適用（ピボットはbbox自動推定） | ★★★ |
| 3 | **胸揺れ**（§8.5。Anime2.5DRig由来・採用希望） | `chest.png` の分離出力（任意） | 縦バネ（肩の弾みと同系・低周波強減衰）＋呼吸・息継ぎ連動 | ★★★ |
| 4 | 視線・瞳ドリフト | `eyewhite.png`＋`irides.png` の分離出力（See-Throughは分離可能） | 虹彩中心自動検出→平行移動、白目でクリップ。**既定は正面基調**（§8.6） | ★★ |
| 5 | 房ごと髪物理（B3強化） | 任意: 房マスクの事前計算出力 | 毛先輪郭ピーク検出で房分割→房ごとチェーン（rigger.js参考） | ★★ |
| 6 | ハイライトドリフト | `highlight.png` 分離出力 | ノイズで±1〜2pxドリフト（PuruPuruの目元演出も参考） | ★ |
| 7 | 深度マップ視差（#1の精緻化） | See-Throughの深度マップを `depth.png` として出力に含める | 変位シェーダ（pixi DisplacementFilter） | 保留 |
| 8 | THA3軸焼き込み | 指示書生成の代替: THA3パラメータスイープで axis_* 連番を自動生成 | 既存の軸スクラブで再生（実装済み） | 保留 |
| — | ~~眉の感情変位~~ | — | — | **廃止**（§8.6: 感情別登録で表情は素材側にあるため不要。代わりに感情→物理倍率プリセットを採用） |

### 8.2 素材フォルダ規約の拡張案（PachiPakuGen出力 v2）

既存規約に追加する形（すべて任意。無ければ従来動作＝後方互換）:

```
output/                ※感情（LayeredEmotion）1つにつき1フォルダ。
                         ピボット・bbox・深度は感情ごとの素材で個別に自動推定する
  （既存: body.png / hair.png / hair_back.png / arm_l.png / arm_r.png /
          eye/ / mouth_a〜o/ / axis_*/）
  chest.png           胸部（胸揺れ対象。§8.5）
  eyewhite.png        白目（視線クリップ領域）
  irides.png          虹彩（視線ドリフト対象。中心はランタイムで自動検出）
  highlight.png       目ハイライト（微小ドリフト対象）
  sway_<name>.png     汎用揺れパーツ（例: sway_ribbon.png, sway_necktie.png）
  meta.json           任意メタ（下記。感情フォルダ単位）
```

`meta.json` 案（無ければ全て既定値で動くこと）:

```json
{
  "version": 1,
  "depth": { "hair_back": -0.6, "body": 0.0, "arm_l": 0.1, "arm_r": 0.1,
             "eye": 0.35, "brow": 0.4, "mouth": 0.35, "hair": 0.8 },
  "pivots": { "arm_l": [x, y], "sway_ribbon": [x, y] }
}
```

- `depth`: パララックス係数（-1..1、0=基準面。852話方式の名前ベース固定
  テーブルを既定値とし、meta.jsonで上書き可能にする）
- `pivots`: bbox自動推定を上書きしたい場合のみ

### 8.3 #1 パララックス首振りの実装仕様（SpriTalk側）

```
入力: headTurn ∈ [-1, 1]（駆動源はノイズドリフト＋発話頷きバネ = 採用済みの軸駆動源）
各レイヤーの変換:
  shiftX = headTurn × depth[layer] × SHIFT_MAX   // SHIFT_MAX ≈ キャンバス幅の2〜3%
  shear  = headTurn × depth[layer] × SHEAR_MAX   // SHEAR_MAX ≈ 0.06
  transform: translateX(shiftX) skewX(shear)     // CSS/WebGLどちらでも表現可
留意:
- 深度差の大きい隣接レイヤー（hair と body）の境界で絵の欠けが出る場合、
  headTurnの可動域を絞る（±0.5相当）か、hair_back側の描き足しを指示書要件にする
- 縦方向（headNod）は shiftY のみで shear 不要（頷きは既存バネと統合）
```

追加素材ゼロで動くため、**animation-labでの次の検証対象はこれが第一候補**。

### 8.4 #3 視線ドリフトの実装仕様（SpriTalk側）

```
1. irides.png の不透明領域から左右の虹彩を連結成分で分離し、各重心=虹彩中心を検出
   （参考: Anime2.5DRig lib/rigger.js、MIT）
2. eyewhite.png の不透明領域をクリップマスクにする
   （DOM: clip-path または mask-image / WebGL: ステンシル）
3. 駆動: gaze(x,y) をノイズドリフト（小振幅・低速）＋数秒ごとに正面(0,0)へ
   バネ復帰。発話開始時は正面を向く（話しかけている感）
4. まばたき（eye連番）とのレイヤー順: eyewhite < irides < highlight < eye(まぶた連番)
```

素材前提: PachiPakuGen側で See-Through の目パーツを eyewhite/irides に
分けて出力する改修（切り出しのみ、生成AIは不要）。

### 8.5 #3 胸揺れの実装仕様（Anime2.5DRig由来・SpriTalk側）

```
素材: chest.png（胸部レイヤー。z-index は body(0) と arm(1) の間 = 0.5相当。
      無ければ機能自体を非表示 = 後方互換）
物理: 縦バネ1本（肩の弾み lift と同系だが低周波・強減衰）
  駆動:
   - 体の上下動のY速度カップリング（呼吸で常時わずかに上下）
   - 発話開始・登場時の撃力（肩の弾みより遅れて・小さく = 二次揺れ）
   - 息継ぎ（§8.6）で吸気の持ち上がりに連動
  変形: translateY ＋ わずかな scaleY（bbox上端固定）。回転は不要
  既定値の目安: k=肩liftの0.5倍 / c=強め（臨界近く）/ 最大±4px
留意: 表現強度は好みが分かれるため、設定UIで 0（無効）〜強 のスライダー必須。
      感情別倍率（§8.6）の対象に含める
```

### 8.6 SpriTalk特性による見直し（重要・全候補に横断適用）

SpriTalkは「常時表示のアバター」ではなく
**「呼ばれた時に現れ、セリフをモーラ同期で喋り、感情別に登録された姿を
呼び出し側が選ぶ」**存在である。この特性から以下を全候補に適用する。

**特性A: 呼ばれた時だけ表示される**
- **登場・退場が最重要の演出機会**: 登場時に全物理（髪・腕・肩・胸・揺れパーツ）へ
  撃力を入れ、着地→揺れて収まる、で「呼ばれた感」を出す。退場も同様に軽い予備動作
- 表示開始時に物理状態を**必ずリセット**し、sin波・ノイズの**位相を毎回ランダム化**
  （短い表示が毎回同じ動きに見える問題の回避）
- 長周期アイドル（5〜7秒周期）は短い表示時間では1周期も見えないことがある。
  効果の主軸は「登場」「発話同期」「退場」に置き、アイドル揺れの既定周期は
  短め（2〜4秒）に寄せる

**特性B: モーラタイミングを事前に持っている（最大の差別化）**
- **息継ぎモーション**: pause_mora（読点・句点）の位置と長さを再生前に知れるため、
  ポーズ開始で肩＋胸が持ち上がり（吸気）、発話再開で戻る動きを事前スケジュール
  できる。マイク音量駆動の他ツールには原理的に不可能な「予知駆動」
- 文末（長めのpause）で頷き（バネ撃力）を自動発火するオプションも同様に可能

**特性C: 感情別登録・呼び出し側が感情を選ぶ**
- 物理パラメータは**キャラクター単位を基本**とし、感情ごとには**強度倍率のみ**を
  上書きする（例: 怒り=揺れ強く速く / 悲しみ=弱く沈む / 喜び=バウンス強め）。
  §5.2 の AnimationSettingsV2 に以下を追記:

  ```ts
  emotionOverrides?: Record<number /* emotionId */, {
    motionScale?: number;   // 揺れ全般の倍率（既定1.0）
    bounceScale?: number;   // 発話・登場バウンスの倍率（既定1.0）
  }>;
  ```

- ピボット・bbox・深度係数は**感情ごとの素材で個別に自動推定**する
  （感情ごとにポーズ・腕位置が違うため。meta.jsonも感情フォルダ単位）
- **感情切替時**（呼び出し側が別emotionIdを指定した瞬間）はレイヤーが丸ごと
  入れ替わるため物理状態は引き継がず、小さな切替撃力（バウンス）を入れて
  「表情が変わった」演出と物理の不連続の隠蔽を兼ねる
- **眉の感情変位は不要**: SpriTalkでは感情＝別スプライト登録で、表情は既に
  素材側で表現されている。ランタイムで眉を動かす意味が薄いため候補から除外
- **視線の既定は正面基調**: 呼ばれて視聴者（画面の向こう）に話しかける存在なので、
  視線ドリフトは「基本正面＋ごく小さな揺らぎ、発話開始時は正面へ復帰」とする。
  ジト目・伏し目などの意図的な視線は感情素材側で表現する

### 8.7 検証手順（実装前に行うこと）

1. **Anime2.5DRig実機確認**: 手持ちのSee-Through出力PSDを
   https://852wa.github.io/Anime2.5DRig/ にドロップし、パララックス首振り・
   房物理・視線の効きを自分の素材で確認する。
   レイヤー名は下表で読み替え（この変換表自体がPachiPakuGen改修の仕様になる）:

   | See-Through/SpriTalk | Anime2.5DRig |
   |----------------------|--------------|
   | body（顔ベース含む） | face |
   | eye（まばたき連番の元） | eye_close / eyelash |
   | 白目・虹彩（未分離） | eyewhite / irides |
   | hair | front hair_1.. |
   | hair_back | back hair_1.. |
   | mouth_a〜o | mouth_open / mouth_close |
   | 眉（未分離） | eyebrow |

2. 効きが確認できたら animation-lab に #1（パララックス）を追加実装して
   パラメータ（SHIFT_MAX/SHEAR_MAX/深度テーブル既定値）を確定
3. 確定値を本書 8.2/8.3 に反映してから PachiPakuGen v2 出力と
   SpriTalk本体実装に着手する

### 8.8 実装しないと決めたもの（再掲・理由付き）

- 音声スペクトラム解析リップシンク: SpriTalk本体ではモーラタイミング同期（A1）の方が
  高精度。息継ぎの「予知駆動」（§8.6 特性B）も事前タイミングがあってこそ
  ※PachiPakuGen単品ビューア（§9.4）ではマイク音量駆動を採用する（そこにはモーラがないため）
- MediaPipeカメラ追従: SpriTalk本体には入れない。**WebVoiceAnimatorの領分**
  （§10.2 カメラ感情推定）として再定義
- 眉の感情変位: 感情別スプライト登録と役割が重複（§8.6 特性C）
- 本格布シミュレーション: チェーン物理で十分

---

## 9. プロダクト間の設計原則 — 複雑さはPachiPakuGenが吸収する

### 9.1 原則

1. **SpriTalkは「再生専用・最小UI」**。キャラクターを使う人（配信者）に
   バネ定数や減衰を触らせない
2. **詳細パラメータは全て素材側（アニメーションプロファイル）に焼き込む**。
   チューニングはキャラクターを作る工程＝PachiPakuGenで行う
3. **プロファイルが無い素材でも既定値で必ず動く**（ゼロコンフィグ・後方互換）

この原則により、§5.1のUI案（スライダー多数）は**PachiPakuGen側のチューニング画面**へ
移動し、SpriTalk本体の設定UIは以下だけに縮小する:

```
キャラクター設定 > アニメーション（SpriTalk最終形）
├─ アニメーション [ON / OFF]
├─ 動きの強さ    [弱 ── 標準 ── 強]（プロファイル値への単一倍率 0.5〜1.5）
└─ 口パク       [自然（推奨） | 従来]
```

※§5.1の詳細UI案は「PachiPakuGenチューニング画面の仕様」として引き続き有効。
　animation-labの「🎛 移植UI案」はそのプロトタイプと位置づけ直す。

### 9.2 アニメーションプロファイル（anim_profile.json）

§8.2の meta.json を拡張し、詳細パラメータを全て含む**プロファイル**として
PachiPakuGenが素材と一緒に出力する（感情フォルダ単位）:

```json
{
  "version": 1,
  "depth":  { "hair_back": -0.6, "body": 0.0, "hair": 0.8 },
  "pivots": { "arm_l": [x, y] },
  "lipSync": { "attackMs": 40, "releaseMs": 90 },
  "parts": {
    "hair":   { "mode": "mesh", "segments": 6, "k": 70, "c": 7, "wind": 0.012, "drive": 0.03 },
    "arm":    { "k": 90, "c": 10, "maxAngle": 0.12, "lift": { "coupling": 0.08, "bounce": 26, "max": 6 } },
    "chest":  { "k": 45, "c": 12, "max": 4 },
    "sway_ribbon": { "k": 60, "c": 6 }
  },
  "presence": { "entryBounce": 1.0, "breathOnPause": 1.0 },
  "motionScale": 1.0
}
```

- SpriTalkの「動きの強さ」は `motionScale` への単一倍率としてのみ作用
- 感情別倍率（§8.6の emotionOverrides）は感情フォルダごとのプロファイルに
  `motionScale`/`bounceScale` を直接書く形で実現（SpriTalk側のUIは増えない）
- スキーマは additive にのみ変更する（旧バージョン読込可を維持）

### 9.3 PachiPakuGenのチューニング画面

- 手動修正UI（レイヤー順・重なり）の後工程として
  「アニメーション調整」タブを追加し、スライダー群（§5.1相当）＋
  ライブプレビューでプロファイルを作り込む
- **プレビューエンジンはanimation-labの流用を推奨**:
  `reference/animation-lab/js/` の物理・方式ロジック（11/12/20/30/32番台）は
  コンポジター非依存の素のJS（IIFE・依存なし）なので、PachiPakuGenの
  WebView/ブラウザUIにそのまま埋め込める。二重実装を避け、
  プレビューと本番（SpriTalk）の挙動一致も保証しやすい

### 9.4 PachiPakuGen単品の価値（スタンドアロン出力）

素材＋プロファイル＋ランタイムを持っているのだから、SpriTalkなしでも
動く形で出力すれば単品プロダクトとして成立する:

1. **ポータブルビューアHTML出力** ★推奨
   素材・プロファイル・アニメーションランタイムを1つの自己完結HTMLに
   パッケージして出力。ブラウザで開くだけで動き、**OBSのブラウザソースに
   ドロップすればマイク音量口パクの単体PNGTuberとして機能**する
   （PuruPuru / Anime2.5DRig と同じ土俵の配布物。技術基盤はanimation-labが
   ほぼそのまま使える）
2. **アイドルループのWebM/APNG/GIF書き出し**
   登場→待機→瞬きのループを動画素材として書き出し。配信サムネ・
   告知動画・Discordスタンプ等、SpriTalkと無関係な用途に使える
3. 位置づけ: 「PachiPakuGen＝1枚絵からリグ済みアバターを作るツール
   （単品でPNGTuber、SpriTalkと繋ぐとTTS同期の完全体）」という二段構えの訴求

### 9.5 役割分担の最終形

```
[See-Through]     1枚絵 → レイヤー分解PSD（＋深度）
      ↓
[PachiPakuGen]    指示書生成 → AI差分 → 手動修正 → RIFE/素材出力
                  ＋ アニメーション調整（プロファイル作成・複雑さはここで吸収）
                  ＋ 単品出力（ポータブルビューア / 動画書き出し）
      ↓  素材＋anim_profile.json（共通パッケージ）
      ├→ [SpriTalk]          TTS/モーラ駆動（AI・スクリプトの発話）§9.1
      └→ [WebVoiceAnimator]  生音声駆動（配信者・コラボ相手）＋カメラ感情推定 §10
```

---

## 10. WebVoiceAnimator統合 — 同じ素材を「生音声」で動かす

[WebVoiceAnimator](https://github.com/kazuya-bros/WebVoiceAnimator)（自社製・MIT・Chrome拡張）は
マイク音声駆動のPNGTuber。現状: 基本（1〜4枚絵）/ アドバンス（PixiJS・呼吸・微揺れ）/
MPNG（5段階母音口パク）の3モード、HQ Audioアルゴリズム
（エンベロープフォロワー・ノイズフロア自動検出・ヒステリシス）、
AudioWorkletによる約60fpsの音声解析＋母音判定、複数キャラクター登録対応済み。

### 10.1 統合方針: 駆動源とランタイムの分離

animation-labの方式ロジックは既に **`MoraState`（vowel / openness / speaking /
speechStarted）という駆動信号だけで動く**構造になっている。この駆動信号を
誰が作るかだけがプロダクトの違いになる:

| プロダクト | 駆動源 | vowel | openness | speechStarted |
|-----------|--------|-------|----------|---------------|
| SpriTalk | モーラタイミング（事前・予知駆動） | audio_query由来 | A4エンベロープ | バッチ再生開始 |
| WebVoiceAnimator | AudioWorklet実時間解析 | 既存の母音判定（音量＋周波数） | HQ Audioエンベロープフォロワー出力 | 発話検出の立ち上がり |
| （カメラ） | MediaPipe等 | — | — | 感情推定→emotionId切替（§10.3） |

つまり**アニメーションランタイム（A1調音結合・B3髪・腕・胸・パララックス・
揺れパーツ）は完全共通**にでき、WVA側はHQ Audio出力を `MoraState` に
アダプトする薄い層を書くだけでよい。WVAの母音判定は5母音口形状に
そのまま接続できる（MPNGモードで実証済みの技術）。

### 10.2 WVAに追加するもの

1. **レイヤーモード（第4のモード）**: PachiPakuGen共通パッケージ
   （素材フォルダ＋anim_profile.json）を読み込み、共通ランタイムで再生。
   §9.2のプロファイルがそのまま効くため、WVA側にも詳細設定UIは増やさない
   （既存のシンプルな設定思想と一致）
2. **複数話者対応（配信者＋コラボ相手）**: 複数キャラクター登録は実装済みなので、
   拡張点は「**キャラクターごとの音声ソース割当**」
   （マイク / タブ音声 / 仮想オーディオデバイス）。Discord等のコラボ音声を
   仮想デバイス経由で割り当てれば、相手の声で相手のキャラが喋る
   ※話者分離（1本の音声から誰が話しているか判定）はスコープ外とし、
   「音声ソース＝1キャラ」の割当方式で確実に動くものを先に作る
3. **カメラ感情推定（任意機能）**: MediaPipe（Apache-2.0、Anime2.5DRigでも採用）の
   顔ランドマーク/ブレンドシェイプから簡易感情分類→**emotionIdの自動切替**。
   感情別素材登録という共通概念（§8.6 特性C）の上に乗るため、
   切替先はSpriTalkと同じ感情フォルダ。切替時の撃力演出も共通ランタイム側で発火
   ※§8.8で「配信者モード候補として保留」としたMediaPipe連携は
   **WVAの領分**と再定義する（SpriTalk本体には入れない）

### 10.3 §9.4（単品ビューア）との関係

「ポータブルビューアHTML＝マイク駆動の単体PNGTuber」の役割は、
WVAレイヤーモードが実現するものと重なる。方針:

- **配信用途はWVAに一本化**（Chrome拡張としてOBS連携・バックグラウンド動作
  〔Offscreen Document API〕まで解決済みの資産を活かす）
- §9.4のポータブルHTML出力は「共有・確認用の軽量アーティファクト」
  （素材チェック・依頼者への納品プレビュー・SNS埋め込み）に役割を絞る

### 10.4 実装順序の提案

1. 共通ランタイムのパッケージ化（animation-lab の 11/12/20/30/32番台＋
   コンポジターを、SpriTalk/WVA/PachiPakuGenプレビューの3者から使える形に切り出す）
2. SpriTalk本体移植（§6 Phase A〜C）— モーラ駆動で先に完成させる
3. WVAレイヤーモード（HQ Audio→MoraStateアダプタ＋パッケージ読込）
4. WVA音声ソース割当（コラボ対応）
5. WVAカメラ感情推定（任意・最後）
