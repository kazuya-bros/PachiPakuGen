# motion-preview-lab と検証済みアニメーション設計の統合メモ

本ブランチ（motion-preview-lab）の実装と、SpriTalk側 animation-lab
（比較検証アプリ、本リポジトリに `reference/animation-lab/` として同梱）で
確定した設計（[animation-lab-tech.md](animation-lab-tech.md)）の対応関係と統合方針。

両者は独立に開発されたが**同じ結論に収束している**。用語の対応:

| motion-preview-lab（本ブランチ） | animation-lab（検証済み設計） | 備考 |
|----------------------------------|-------------------------------|------|
| lipSyncRenderer: `directLayerSwitch` | **A1 調音結合**（採用決定） | 母音間で閉じずにレイヤー即差し替え |
| lipSyncRenderer: `smoothedFrameStepper` | A0 現行方式の改善版 | ベースライン |
| lipSyncRenderer: `neutralBridgeOpacityBlend` | A2 クロスフェード（見送り） | |
| lipSync.attackMs / releaseMs | **A4 開度エンベロープ**（採用決定） | 概念完全一致 |
| layerRenderer: `stripWarpExtension` | **B3 メッシュ髪揺れ**（採用決定） | 縦ストリップ変形 |
| layerRenderer: `existingProceduralAnimator` | B0 現行sin波（後方互換フォールバック） | |
| layerMotion.hairFront/BackDelayMs | B1 遅延追従の位相ずらし近似 | 下記「物理の置き換え」参照 |
| spritalk-motion-profile.json | anim_profile.json（§9.2） | **本ブランチの実装を正とする** |

## 統合方針

### 1. プロファイルは spritalk-motion-profile.json を正とする

設計書§9.2の `anim_profile.json` は本ブランチの
`spritalk-motion-profile.json`（schema: `spritalk.motionProfile.v1`）に統合する。
v1に対して**additiveに** v2 で追加すべきフィールド（検証済み・採用決定分）:

```jsonc
{
  "schema": "spritalk.motionProfile.v2",
  // ...v1の既存フィールドはそのまま...

  // 追加: 本物のバネ物理パラメータ（検証済み既定値）
  "physics": {
    "hair":  { "mode": "chain", "segments": 6, "k": 70, "c": 7,
               "wind": 0.012, "drive": 0.03 },          // B3角度チェーン
    "arm":   { "k": 90, "c": 10, "maxAngle": 0.12, "coupling": 0.02, "noise": 0.008,
               "lift": { "coupling": 0.08, "bounce": 26, "max": 6 } }, // 腕＋肩の弾み
    "chest": { "k": 45, "c": 12, "max": 4 }             // 胸揺れ
  },
  // 追加: SpriTalk特性連動（設計書§8.6）
  "presence": {
    "entryBounce": 1.0,       // 登場撃力の倍率
    "breathOnPause": 1.0,     // pause_mora息継ぎの倍率
    "randomizePhase": true    // 表示ごとにsin/ノイズ位相をランダム化
  },
  // 追加: パララックス首振り（設計書§8.3）
  "depth": { "hair_back": -0.6, "body": 0.0, "arm_l": 0.1, "arm_r": 0.1,
             "eye": 0.35, "mouth": 0.35, "hair": 0.8 },
  // 追加: 感情別倍率（感情フォルダごとのプロファイルに記載）
  "motionScale": 1.0,
  "bounceScale": 1.0
}
```

### 2. プレビュー物理の置き換え（sin位相ずらし → バネ-ダンパー）

現在のMotion Labプレビューは `Math.sin(time - delay)` の位相ずらしで
遅延追従を近似しているが、SpriTalk本番は**半陰的オイラーのバネ-ダンパー
チェーン**（検証済み）になる。プレビューと本番の挙動一致のため、
`reference/animation-lab/js/` の以下をMotion Labへ移植することを推奨:

- `02-utils.js` … smoothDamp / 1Dノイズ / alphaBBox（肩ピボット自動推定）
- `30-body-methods.js` … B3角度チェーン（springStep・サブステップ・クランプ）
- `32-arm-sway.js` … 腕チェーン＋肩の弾み（lift）
- `11-mora-player.js` … A4エンベロープ（attack/release指数スムージング）
- `12-axis-scrubber.js` … 軸スクラブ（ノイズドリフト/バネ頷き）

いずれも依存なしの素のJS（IIFE）なので、React側からそのまま import相当で
利用できる（グローバル名前空間 `AnimLab` を window に生やす方式）。

### 3. 素材規約の拡張（設計書§8.2 → 本リポジトリのローダーへ）

`load_motion_lab_parts` / SpriTalk出力（04_spritalk_parts）に追加対応するもの:

- `arm_l.png` / `arm_r.png`（腕揺れ。肩ピボットは不透明bbox上端中央を自動推定）
- `chest.png`（胸揺れ）
- `sway_<name>.png`（汎用揺れパーツ）
- `eyewhite.png` / `irides.png` / `highlight.png`（視線・ハイライト。将来）
- 感情フォルダ単位の `spritalk-motion-profile.json`（ピボット・深度も感情ごと）

### 4. 各リポジトリへのハンドオフ

- SpriTalk本体の改修設計: [handoff-spritalk.md](handoff-spritalk.md)
- WebVoiceAnimatorの改修設計: [handoff-webvoiceanimator.md](handoff-webvoiceanimator.md)
- 全体の経緯・数式・採用判断: [animation-lab-tech.md](animation-lab-tech.md)
- 検証用リファレンス実装: `reference/animation-lab/`（index.htmlをブラウザで
  開くだけで動く。モック素材内蔵。方式比較・パラメータ探索に使用）
