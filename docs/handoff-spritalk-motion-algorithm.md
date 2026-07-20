# SpriTalk向けハンドオフ — Gen動作アルゴリズムの厳密移植仕様

`spritalk-motion-profile.json`（[types.ts](../src/motionLab/types.ts) の `SpritalkMotionProfile`）は
**調整可能な設定値の受け渡し**が目的で、**計算式そのものは運びません**。
2026-07に実機比較（PachiPakuGen STEP7ライブ表示 vs SpriTalk）を行ったところ、
プロファイルの数値だけを読んでも一致しない箇所が複数見つかりました。原因は、
Gen側が「プロファイルにない派生値」や「素材ピクセルを実際に解析した結果」を
使っているためです。本書はその計算式を1つずつ、Gen実装（`src/motionLab/*.ts`）の
該当行を引用しながら転記します。[handoff-spritalk.md](handoff-spritalk.md) の
Phase B（物理揺れ）を実装済み・実装中の前提で書いています。

**方針**: プロファイルJSONの数値を右から左に流すのではなく、本書の式を
SpriTalk側のコードとして実装してください（＝Gen固有のアニメーションアルゴリズムを
SpriTalk側にも持つ）。プロファイルJSONは「どのエフェクトが有効か」「ユーザーが
調整したスカラー値（strength/scale等）」を運ぶだけと捉えてください。

---

## 0. 前提: 物理プリミティブ

以降の全ての式が依存する基礎関数です。[motionLabPhysics.ts](../src/motionLabPhysics.ts) より抜粋（完全一致で移植してください。定数を変えるとdt挙動が変わります）。

```ts
// Unity互換 SmoothDamp（臨界減衰バネによる滑らかな追従）
function smoothDamp(current: number, target: number, velRef: { v: number }, smoothTime: number, dt: number): number {
  smoothTime = Math.max(0.0001, smoothTime);
  const omega = 2 / smoothTime;
  const x = omega * dt;
  const exp = 1 / (1 + x + 0.48 * x * x + 0.235 * x * x * x);
  const change = current - target;
  const temp = (velRef.v + omega * change) * dt;
  velRef.v = (velRef.v - omega * temp) * exp;
  let output = target + (change + temp) * exp;
  if (target - current > 0 === output > target) {
    output = target;
    velRef.v = dt > 0 ? (output - target) / dt : 0;
  }
  return output;
}

// dtクランプ（巨大dtで発散させない）。全ての物理stepの前段で必須
function clampDt(dt: number): number {
  return Math.max(0, Math.min(dt, 0.1));
}

// バネ-ダンパー1本（半陰的オイラー＋dt>1/60でサブステップ2分割）
function springStep(state: { x: number; v: number }, target: number, k: number, c: number, dt: number): number {
  const dtc = clampDt(dt);
  const steps = dtc > 1 / 60 ? 2 : 1;
  const h = dtc / steps;
  for (let i = 0; i < steps; i += 1) {
    const a = -k * (state.x - target) - c * state.v;
    state.v += a * h;
    state.x += state.v * h;
  }
  return state.x;
}

// 1D Perlin風ノイズ（-1..1）。PERM配列はseed=1234567の決定的Fisher-Yatesシャッフル
// （完全一致は不要だが、周期性の無い滑らかなノイズであること）
function noise1d(x: number): number {
  const i0 = Math.floor(x) & 255;
  const i1 = (i0 + 1) & 255;
  const f = x - Math.floor(x);
  const u = f * f * f * (f * (f * 6 - 15) + 10); // smootherstep
  const g0 = PERM[i0] / 127.5 - 1;
  const g1 = PERM[i1] / 127.5 - 1;
  const n0 = g0 * f;
  const n1 = g1 * (f - 1);
  return (n0 + (n1 - n0) * u) * 2;
}
```

**⚠️ `randomizePhase`（`presence.randomizePhase`）について**: Genは表示開始のたびに
`breathPhase` / `swayPhase` / `noiseT` / `hairChain.t` / `arm.left.t` / `arm.right.t` /
`blinkWait` / `headTurnT` / `glanceWait` を**それぞれ独立して`Math.random()`で
初期化**します（[render.ts:948-991](../src/motionLab/render.ts#L948)）。
つまり同じプロファイルでも毎回違う位相で動き出すのが**Genの仕様**です。
SpriTalkとGenを並べて再生してもフレーム単位の完全一致は原理的にありえません。
比較は「振幅・周期・応答特性が同じか」で行ってください。

---

## 1. 視線（gaze）— 最優先で直してほしい箇所

### 1-1. 虹彩領域の検出（`irides.png`を実際に解析する）

Genの視線振幅は固定値ではなく、**`irides.png`のアルファチャンネルを解析して
左右の虹彩ブロブを検出し、その実測サイズから振幅を算出**します。
[eyeEffects.ts](../src/motionLab/eyeEffects.ts) `motionLabEyeRegions()`:

1. 各列(x)ごとに `alpha > threshold` のピクセル数をカウントし、列ヒストグラムを作る
2. `significantColumnPixels = max(2, ceil(maxColumnPixels * 0.08))` 未満の列は無視して
   有効な `minX..maxX` を求める（`maxX - minX < 3` なら領域なし扱い）
3. 中心 `center = (minX+maxX)/2` の付近30%〜70%区間で、列ヒストグラムが最も低い
   谷 `splitX` を探す（同値なら中心に近い方を優先）
4. `columns[splitX] > maxColumnPixels * 0.25` なら「谷が浅い＝目が1つに繋がって
   見えている」と判断し、**分割せず単一領域**として扱う
5. そうでなければ `splitX` で左右に分割し、各領域の alpha bboxを個別に返す
   （`{x, y, w, h, pixels}`。有効ピクセル数4未満は破棄）

### 1-2. 振幅の算出

```ts
// 100%までは控えめ。300%で最大4.8pxまで
function motionLabGazeRangePx(regions: EyeRegion[], strength: number): number {
  if (regions.length === 0 || strength <= 0) return 0;
  const averageWidth = regions.reduce((sum, r) => sum + r.w, 0) / regions.length;
  const baseRange = clamp(averageWidth * 0.03, 0.45, 1.6);
  return clamp(baseRange * strength, 0, 4.8); // MOTION_LAB_GAZE_MAX_RANGE_PX = 4.8
}
// 水平のみ。縦方向は常に0（意図的）
function motionLabHorizontalGazeAt(phase: number, regions: EyeRegion[], strength: number) {
  return { x: Math.sin(phase) * motionLabGazeRangePx(regions, strength), y: 0 };
}
```

`strength` = プロファイルの `physics.gaze.strength`（＝`gazeStrength`設定、0〜3）。
`phase` は `gazeT += dt * (2π / periodSeconds)` を積分した値
（`periodSeconds = 11.5`。プロファイル `physics.gaze.periodSeconds`）。

**プロファイルの `gaze.maxRangePx` はこの計算結果の上限クランプでしかなく、
実際の振幅そのものではありません**（[types.ts:471](../src/motionLab/types.ts#L471)
に「旧ランタイム互換。現行は虹彩実寸からpx上限を算出するため0」とコメント済み、
`rangeRatio`は常に0で無視してよい値）。**SpriTalk側でも同じ列ヒストグラム分割
アルゴリズムを実装し、読み込んだ`irides.png`から`averageWidth`を実測してください。**
そうしないと、キャラごとに異なるはずの視線振幅が全キャラ同じになるか、
プロファイルのmaxRangePxをそのまま振幅として使って動きすぎ／動かなすぎになります。

### 1-3. 描画（白目クリップ＋虹彩再配置）

検出した領域を個別にcrop＆再描画します（画像全体をtranslateするのではない）。
[render.ts:725-789](../src/motionLab/render.ts#L725) `drawMotionLabGaze()`:

```ts
scratchCtx.globalCompositeOperation = "source-over";
scratchCtx.drawImage(eyewhite, 0, 0, width, height); // 白目を土台に
scratchCtx.globalCompositeOperation = "source-atop"; // 以降は白目の不透明部分にしか描かれない
for (const region of regions) {
  const padding = 2;
  const x = Math.max(0, region.x - padding);
  const y = Math.max(0, region.y - padding);
  const w = Math.min(width - x, region.w + padding * 2);
  const h = Math.min(height - y, region.h + padding * 2);
  // irides.pngの同じ矩形を、gazeX/gazeY分だけずらして描く（左右の目が同じ方向へ動く）
  ctx.drawImage(irides, x*sx, y*sy, w*sx, h*sy, x + gazeX, y + gazeY, w, h);
}
```
（`sx`/`sy` は irides の naturalWidth/Height と表示widht/heightの比。regions が
空（=検出失敗）の時だけ、フォールバックとして画像全体を`gazeX,gazeY`でtranslateします。）

### 1-4. まばたき中の視線復帰（優先度2の項目）

`blinkFrame.phase !== "idle"`（＝まばたき進行中は常に）の間、
gazeターゲットを**強制的に(0,0)**にし、通常の`smoothTime=0.42s`ではなく
`centerSmoothTime = max(0.035, centerMs/3000)`（centerMs=190なので約0.063秒）
という速い収束で中央へ戻します。これは`randomGlance`設定と無関係に常時有効です。
`highlight`（ハイライト光沢）の位置は**常にgazeと同値**を使います
（旧実装の×0.6は廃止済み。[render.ts:1594](../src/motionLab/render.ts#L1594)）。

---

## 2. ランダムグランス（glance）— 「頭の向き」だけで「視線」は動かさない

[render.ts:1274-1288](../src/motionLab/render.ts#L1274):

```ts
if (randomGlance && glanceStrength > 0 && layerMode !== "simple") {
  if (!blinkFrame.sequenceActive) {          // まばたき準備中は新目標を選ばない
    glanceWait -= dt;
    if (glanceWait <= 0) {
      glanceWait = 1.4 + random() * 2.6;      // 次回まで1.4〜4.0秒
      glanceHeadTarget = (random()*2 - 1) * 0.45 * glanceStrength;
    }
  }
  glanceHead = smoothDamp(glanceHead, glanceHeadTarget, glanceHeadVel, 0.5, dt);
} else {
  glanceHeadTarget = 0;
  glanceHead = smoothDamp(glanceHead, 0, glanceHeadVel, 0.5, dt);
}
```

**まばたき中は「完全ブロック」ではなく「新しい目標選びを止めるだけ」**です。
既に選んだ目標へのsmoothDampはまばたき中も継続するため、頭は動き続けます
（止まって見えるのは目標選定タイミングとまばたきがたまたま重なった時だけ）。
`glanceHead`は視線(gaze)には一切影響せず、後述の`headTurn`（パララックス首振り）
にのみ加算されます。**視線側は影響を受けません**（型に`glanceGaze`フィールドは
存在しますが、現行実装では常に0で未使用の死んだ状態です — 移植不要）。

---

## 3. 腕の揺れ・肩リフト — 2つの隠れた依存式

### 3-1. 常時アイドルスイング（体が静止していても腕は揺れる）

[render.ts:1330](../src/motionLab/render.ts#L1330):

```
idleSwing = armMaxAngle * 0.45 * armSwayAmp   // ← プロファイルに直接の値は無い。要計算
```

これが無いと「体を動かさない限り腕が完全に静止する」状態になり、Genと明確に違って見えます。

### 3-2. 発話バウンス連動時の減衰（pyoko連動）

同じ行群、[render.ts:1333-1335](../src/motionLab/render.ts#L1333):

```
liftCoupling = ARM_DEFAULTS.lift.coupling * (pyokoBounce > 0 ? 2.8 : 1) * liftStrength
liftBounce   = ARM_DEFAULTS.lift.bounce   * (pyokoBounce > 0 ? 0.25 : 1) * liftStrength
```

**⚠️ 重要**: この判定は「発話バウンスのON/OFF」ではなく **`pyokoBounce`スライダーの
生値（0より大きいか）** で行われます。プロファイルが出す`physics.pyoko.amplitudePx`は
`effects.pyoko`がOFFなら0にゲート済みなので、「発話バウンスOFF・スライダーは
3のまま」という状態がプロファイル側からは判別できません（Gen内部では
`pyokoBounce > 0`かどうかだけを見ているため、この場合も2.8倍/0.25倍側の式が
使われます）。厳密一致が必要なら、Gen側に生の`pyokoBounce`値を別途出力する
対応を依頼してください（現時点のプロファイルには含まれていません）。実務上は
「発話バウンスがONならpyoko連動あり」とみなして問題ないはずです（OFFにして
スライダーだけ残すのは通常のUI操作では起きにくいため）。

### 3-3. 腕チェーン本体（B3と同じ`stepChain`、3分割）

[motionLabPhysics.ts:224-262](../src/motionLabPhysics.ts#L224) `updateArmSway()`:

```ts
vx = (rootX - prevRootX) / dt;  // 体（root）の速度
vy = (rootY - prevRootY) / dt;

// --- lift（肩の上下、両肩共通1本のバネ） ---
if (liftEnabled) {
  if (speechStarted) lift.v += liftBounce;               // 発話開始の撃力
  driveY = clamp(-liftCoupling * vy, -liftMax, liftMax);   // 体Y速度への遅延追従
  kL = k * 0.8; cL = c * 0.9;                              // k/c = MOTION_LAB_ARM_DEFAULTS (90/10)
  a = -kL * (lift.x - driveY) - cL * lift.v;
  lift.v += a * clampDt(dt);
  lift.x = clamp(lift.x + lift.v * clampDt(dt), -liftMax, liftMax);
} else {
  lift.x *= max(0, 1 - dt*8); lift.v = 0;                  // OFF時は滑らかにゼロへ
}

// --- 左右チェーン（駆動=体X速度カップリング＋アイドルスイング＋ノイズ） ---
drive = clamp(-coupling * vx * 0.05, -maxAngle, maxAngle);  // coupling = 0.02 * armSwayAmp
for (side of [left, right]) {
  noise = noise1d(chain.t*0.5 + (side==='left'?0:37)) * (0.008 * armSwayAmp); // noise定数
  idle = idleSwing > 0 ? sin(chain.t*0.9 + (side==='left'?0:2.1)) * idleSwing : 0;
  stepChain(chain, drive + noise + idle, k=90, c=10, dt, maxAngle);
  result[side] = { rigid: chainAverage(chain), lift: lift.x };
}
```

`stepChain`は0番のセグメント（根元）だけが`driveTarget`を直接追いかけ、以降は
1つ前のセグメントの角度を目標にする多段バネ（毛先ほど低剛性: `k_i = k*(1-0.6*i/n)`）。
`chainAverage`は全セグメント角の単純平均です（メッシュ描画しない場合の剛体回転近似）。

最終的な腕の描画角度は `angle = rigid * swingScale[part] + bodyTransform.rotationDeg`、
ピボットは `pivots[part]`未指定なら「腕bboxの上端中央 + armPivotRatio×bbox高さ」
（[render.ts:1380-1387](../src/motionLab/render.ts#L1380)）。

定数（`MOTION_LAB_ARM_DEFAULTS`, [constants.ts:224](../src/motionLab/constants.ts#L224)）:
```
k: 90, c: 10, coupling: 0.02, noise: 0.008, maxAngle既定: 0.12
lift: { coupling: 0.08, bounce: 26, max: 6 }
```

---

## 4. 胸部ワープ — chest.png無し時のフォールバック領域

[chestWarp.ts:20-42](../src/motionLab/chestWarp.ts#L20):

```ts
if (guideBounds有効（w>2 && h>2）) {
  centerX = guide.x + guide.w/2;  centerY = guide.y + guide.h/2;
  radiusX = max(width * 0.035, guide.w * 0.34);
  radiusY = max(height * 0.025, guide.h * 0.42);
} else {
  // chest.pngが無い場合: body.png全体のalpha bboxを使う（顔〜足先の全身bbox）
  centerX = bodyBounds.x + bodyBounds.w/2;  centerY = bodyBounds.y + bodyBounds.h/2;
  radiusX = max(width * 0.05, bodyBounds.w * 0.16);
  radiusY = max(height * 0.035, bodyBounds.h * 0.1);
}
```

`bodyBounds`は**body.png単体の不透明ピクセルbbox**（[render.ts:328](../src/motionLab/render.ts#L328)
`alphaBBox(body)`）。ガイド`chest.png`があればそちらの位置を使い範囲だけ既定推定より
広め、無ければ全身bboxの中心に控えめな範囲を置く、という2段構えです。
呼吸・発話バウンスによる縦オフセット`offsetY`が`|offsetY| < 0.02`のときは
ワープ自体を実行しません（早期リターン、[render.ts:302](../src/motionLab/render.ts#L302)）。

---

## 5. entryBounce — 効くのは「B3角度チェーン」だけ

[render.ts:998-1012](../src/motionLab/render.ts#L998) `resetMotionLabRuntime()`:

```ts
if (entryBounce > 0) {
  hairChain.omegas[0] += 0.9 * entryBounce;
  arm.lift.v += 26 * 0.8 * entryBounce;  // = 20.8 (entryBounce=1のとき)
  chest.v    += 4 * entryBounce;
}
```

**⚠️ 髪については`hairChain`（B3、根元セグメントの角速度）にしか撃力が入りません。**
`hairSpring`/`hairBackSpring`（B1スプリング）や、波モード（`hairEngine==="wave"`、
下記§6）の状態には**一切効きません**。つまり `hairEngine==="wave"` のキャラでは、
Gen実機上でも「登場時に髪が跳ねる」演出は発生せず、腕と胸だけが撃力を受けます。
SpriTalk側がこれを全エンジン共通で髪にも適用してしまうと、逆にGenより
派手な登場演出になり不一致になります。

---

## 6. 波モード髪（`hairEngine==="wave"`）— 波の式

`hairMotionStrength`/`hairWaveStrength`/`hairBackScale`が有効なのはこの式の中です。
[render.ts:659-718](../src/motionLab/render.ts#L659) `drawMotionLabWaveWarp()`
（前髪・後ろ髪とも同じ関数、パラメータだけ変える。後ろ髪は大波化: `spatialFreq=0.42`
`tempo=0.6`、[render.ts:1404-1412](../src/motionLab/render.ts#L1404)）:

```ts
TAU = 2π
beat = (timeMs/1000) * (160/60) * tempo    // PuruPuruPNGTuber同様のBPM160基準
px = (width/1024) * strength                // strength = hairWaveStrength*preset.hair*hairMotionStrength(*hairBackScale, 後ろ髪のみ)
voiceBoost = 1 + voice * 0.42

// stripCount本（既定24）の横帯に分割し、帯ごとに独立したdx/dyでずらして描画
for (i = 0..stripCount) {
  u = clamp((帯の中心Yratio - rootYRatio) / (1 - rootYRatio), 0, 1);
  mask = clamp(u*1.25, 0, 1) * u;            // 根元固定・毛先ほど強く効くマスク
  idleDrift = sin(TAU*(beat*0.42 + u*0.82*sf + seed));
  wave      = sin(TAU*(beat*0.72 + u*1.55*sf + 0.16 + seed*0.7));
  slow      = sin(TAU*(beat*0.5 - 0.255 + seed*0.3));
  idleFloat = cos(TAU*(beat*0.36 + u*0.38*sf + seed));
  dx = mask * px * voiceBoost * (5.2*idleDrift + 2.8*wave + 3.4*voice*slow);
  dy = mask * px * (1.5*idleFloat);
  // この帯だけ (dx, dy) だけずらして drawImage
}
```
前髪の`seed=0.6`目安、後ろ髪`seed=0.82`で位相をずらしています（同位相だと
前後髪が完全同期して不自然）。`voice`は現在の発話エンベロープ開度(0..1)。

**mesh・spring・wave の優先順位**（同じ髪でも layerMode と hairEngine の組で
描画経路が変わります。[render.ts:1391-1420](../src/motionLab/render.ts#L1391)）:
1. `hairEngine==="wave"` かつ `hairMotionEnabled` → 波ワープ（優先度最高。
   `layerMode==="mesh"`でも波が勝つ）
2. でなければ `layerMode==="mesh"` かつ房検出成功 → 房ごとソフトブレンドワープ
3. でなければ B1スプリング遅延追従（§3の腕と同系統の1本バネ。前髪
   `theta = springStep(hairSpring, drive, hairK, hairC, dt)`、
   `drive = clamp(-hairDrive*hairMotionStrength*rootVX, -0.18, 0.18)`）

---

## 7. 検証時の目安値（定数一覧）

| 定数 | 値 | 出典 |
|---|---|---|
| `MOTION_LAB_GAZE_DEFAULTS` | periodSeconds:11.5, smoothTime:0.42, maxRangePx:4.8 | constants.ts |
| `MOTION_LAB_BLINK_DEFAULTS` | centerMs:190, closeMs:90, openMs:130, settleMs:140, intervalMin:2, intervalMax:10(秒) | constants.ts |
| `MOTION_LAB_ARM_DEFAULTS` | k:90, c:10, coupling:0.02, noise:0.008, maxAngle:0.12, lift:{coupling:0.08, bounce:26, max:6} | constants.ts |
| `MOTION_LAB_NOD_DEFAULTS` | k:120, c:12, impulse:34, maxPx:8 | constants.ts |
| `MOTION_LAB_PARALLAX_DEFAULTS` | shiftRatio:0.045, shearMax:0.08, driftSpeed:0.35 | constants.ts |
| `MOTION_LAB_DEPTH_DEFAULTS` | hair_back:-0.6, body:0, arm_l/r:0.1, eye:0.35, mouth:0.35, hair:0.8 | constants.ts |
| `MOTION_LAB_HAIR_DEFAULTS` | k:70, c:7, wind:0.012, drive:0.03 | constants.ts |
| `MOTION_LAB_PRESENCE_DEFAULTS` | entryBounce:1.0, breathOnPause:1.0(SpriTalk専用), randomizePhase:true | constants.ts |
| gaze振幅式の係数 | `clamp(averageWidth*0.03, 0.45, 1.6)` | eyeEffects.ts |
| 波モードbeat基準 | BPM160 (`160/60`) | render.ts |

---

## 8. 比較不能・対応不要と確認済みの項目

- **`presence.breathOnPause`**: Gen側ライブレンダラに`pause_mora`概念自体が
  存在しないため、Gen実機とは比較できません。SpriTalk独自仕様として問題ありません
- **eyewhite/irides欠落時のフォールバック**: Genも同様に静的描画へフォールバックします
- **bodyレイヤー自体へのparallax再適用なし（`depth.body`=0）**: 仕様通りです。
  body transformは他レイヤーのオフセット基準そのものなので、parallaxは適用されません
- **highlightオフセット=gazeと同値**: 旧実装の×0.6は廃止済みで、現行は同値です

---

## 9. 今後の恒久対応（提案）

本書の§1・§3で明らかになった「プロファイルJSONから導出不可能な値」について、
Gen側で以下をプロファイルへ追加する対応も可能です（別途依頼があれば実施します）。
- 実測ベースのgaze基準振幅（`averageWidth`算出結果そのもの。ただしirides.pngの
  解像度依存なので、結局アルゴリズム自体の共有の方が安全）
- `armIdleSwing`（計算済みの`armMaxAngle*0.45*armSwayAmp`）
- 生の`pyokoBounce`スライダー値（`effects.pyoko`のゲートを通さないもの）

ただし今回は「②SpriTalk側にGen専用アルゴリズムを実装する」方針のため、
本書がその一次情報になります。プロファイル拡張が必要になった場合は
[handoff-spritalk.md](handoff-spritalk.md) と合わせて別途Issue化してください。
