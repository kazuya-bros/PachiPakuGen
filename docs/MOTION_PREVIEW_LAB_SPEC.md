# Motion Preview Lab Specification

> [!NOTE]
> この文書はモーション調整機能を設計した際の履歴資料です。旧モード構成や独立した実験画面の案はv0.4.0で廃止され、モーション調整はSTEP 7、ライブ表示は制作ホームとSTEP 7から利用する現行フローへ統合されています。

Checked at: 2026-07-03

## 1. Purpose

`Motion Preview Lab` は、PachiPakuGen で作った `04_spritalk_parts` 相当の素材を読み込み、
複数の口パク/揺れ方式を同じ素材・同じタイムラインで比較するための実験画面。

この画面の目的は SpriTalk 本体への即時移植ではない。PachiPakuGen 内で見た目、設定量、
実行コスト、素材破綻を比べ、最後に残った方式だけを SpriTalk のランタイムへ移す。

## 2. Placement in Current App

現行 `src/App.tsx` の画面モードは次の3種類に整理されている。

```ts
type Mode = "select" | "workspace" | "live";
```

`Motion Preview Lab` の検証成果はSTEP 7のモーション調整へ統合済み。入力は現在のワークスペースの
`04_spritalk_parts`を使い、設定済み素材は制作ホームのライブ表示から直接再利用できる。

## 3. Input Contract

Required folder layout:

```text
04_spritalk_parts/
  body.png
  hair.png              # optional
  hair_back.png         # optional
  eye/                  # optional, existing blink sequence
    001.png
  mouth_a/
    001.png
  mouth_i/
  mouth_u/
  mouth_e/
  mouth_o/
```

Optional inputs:

- `manifest.json` from PachiPakuGen output.
- test audio file.
- test vowel timeline JSON.
- reference video or settings inspired by PuruPuruPNGTuber / LivePortrait / other tools.

Fallback if folders are missing:

- Missing `hair.png` or `hair_back.png`: disable hair lane.
- Missing one or more `mouth_*`: show the lane as incomplete; do not infer a final result.
- Missing `eye`: keep blink disabled; the current goal treats blink as already acceptable.

## 4. Preview Methods

### 4.1 Baseline

Purpose:

- Show current SpriTalk-like behavior for comparison.

Behavior:

- Mouth target changes directly according to the timeline.
- Body/hair use existing simple idle transform if available.

### 4.2 Lip A: Timeline Smoother

Purpose:

- Reduce abrupt vowel shape changes without generating new image assets.

Internal state:

```ts
type MotionLabMouthState = {
  openY: number;       // 0..1
  formX: number;       // -1..1, wide/smile/flat side
  roundness: number;   // 0..1
  target: "closed" | "a" | "i" | "u" | "e" | "o";
  displayedTarget: string;
  frameIndex: number;
  blend: number;       // 0..1 to next target
};
```

Controls:

- `attackMs`: 40-180, default 90
- `releaseMs`: 80-260, default 160
- `holdMs`: 30-140, default 70
- `crossfadeMs`: 0-120, default 50
- `restBias`: 0-1, default 0.25
- `openGain`: 0.5-1.5, default 1.0
- `shapeSmoothing`: 0-1, default 0.65
- `bridgeBias`: 0-0.85, default 0.45

Rendering:

- First implementation can use canvas 2D alpha compositing.
- Select frame from the existing `mouth_<vowel>/` sequence by smoothed `openY`.
- If crossfade is enabled, composite current and next mouth frame with `blend`.
- `bridge` mode inserts a weighted neutral/closed-mouth phase between vowel targets, reducing hard A-to-I style shape jumps without generating new files.
- Do not create new files unless user explicitly exports the preview result.

Acceptance:

- `closed -> a -> i -> u -> e -> o -> closed` no longer pops frame-to-frame.
- Vowel identity remains visually distinguishable.
- No extra dependency.

### 4.3 Lip B: RIFE Transition Graph

Purpose:

- Generate explicit in-between frames for difficult vowel-to-vowel transitions.

Transition set:

```text
a_i, i_a
a_u, u_a
i_u, u_i
a_o, o_a
closed_a, a_closed
```

Controls:

- `transitionFrames`: 3-8, default 4
- `selectedPairs`: manual checklist
- `reuseReverse`: default true
- `maxTransitionMs`: default 120
- `generateOnDemand`: default true

Backend fit:

- Reuse existing RIFE path.
- Reuse existing premultiply/extract handling. Transparent mouth PNGs must not go directly into RIFE.
- Store generated graph under lab work output, not in final SpriTalk folder at first.

Output draft:

```text
motion_lab/
  transition_graph/
    mouth_a_to_i/
      001.png
      ...
  motion-preview-manifest.json
```

Acceptance:

- Generated transition endpoints match input endpoint frames.
- No black halo around transparent mouth parts.
- User can compare Lip A against Lip B on the same timeline.

### 4.4 Motion A: Layered Spring

Purpose:

- Make hair/body movement richer while keeping SpriTalk-compatible PNG layers.

Layer model:

```ts
type MotionLabLayer = "body" | "hair" | "hair_back";

type LayerMotionState = {
  x: number;
  y: number;
  rotationDeg: number;
  scaleX: number;
  scaleY: number;
  pivotX: number;
  pivotY: number;
};
```

Virtual parameters:

- `breath`: slow 0..1 cycle
- `voiceEnergy`: optional from audio/test curve
- `bodyAngleX`
- `bodyAngleY`
- `bodyAngleZ`
- `hairFront`
- `hairBack`

Controls:

- preset: `calm`, `normal`, `lively`
- breath amplitude
- body sway amplitude
- body vertical amplitude
- hair front delay
- hair back delay
- softness
- convergence
- wind noise
- pivot position

Rendering:

- First implementation: canvas layer transforms only.
- Later implementation: optional mesh warp for `hair` and `hair_back`.
- Respect draw order: `hair_back -> body -> mouth/eye -> hair`.

Acceptance:

- Hair and body do not move as a single rigid block.
- Hair front/back have visibly different lag.
- Pivot defaults produce no obvious sliding around the neck.
- The motion can be disabled per layer.

## 5. Timeline

Use a deterministic test timeline before audio analysis.

```json
{
  "fps": 30,
  "durationMs": 4200,
  "events": [
    { "timeMs": 0, "mouth": "closed", "energy": 0.0 },
    { "timeMs": 450, "mouth": "a", "energy": 0.85 },
    { "timeMs": 950, "mouth": "i", "energy": 0.65 },
    { "timeMs": 1450, "mouth": "u", "energy": 0.7 },
    { "timeMs": 1950, "mouth": "e", "energy": 0.6 },
    { "timeMs": 2450, "mouth": "o", "energy": 0.75 },
    { "timeMs": 3200, "mouth": "closed", "energy": 0.1 }
  ]
}
```

This keeps first validation independent from speech recognition quality.

## 6. Manifest And Profile

Save preview settings beside the preview workspace.

```json
{
  "schema": "pachipakugen.motionPreview.v1",
  "sourcePartsDir": "04_spritalk_parts",
  "createdAt": "2026-07-03T00:00:00+09:00",
  "methods": {
    "baseline": { "enabled": true },
    "lipTimelineSmoother": {
      "enabled": true,
      "method": "bridge",
      "attackMs": 90,
      "releaseMs": 160,
      "holdMs": 70,
      "crossfadeMs": 50,
      "restBias": 0.25,
      "openGain": 1.0,
      "shapeSmoothing": 0.65,
      "bridgeBias": 0.45
    },
    "rifeTransitionGraph": {
      "enabled": false,
      "transitionFrames": 4,
      "reuseReverse": true,
      "selectedPairs": ["a_i", "a_u", "i_u"]
    },
    "layeredSpring": {
      "enabled": true,
      "layerMode": "mesh",
      "preset": "normal",
      "breathAmplitude": 1.0,
      "bodySwayAmplitude": 1.0,
      "hairFrontDelay": 0.18,
      "hairBackDelay": 0.28,
      "softness": 0.7,
      "convergence": 0.9
    }
  },
  "timeline": {
    "type": "builtInVowelTest"
  },
  "metrics": {},
  "review": {
    "verdict": "undecided",
    "scores": {
      "mouthSmoothness": 3,
      "vowelReadability": 3,
      "bodyNaturalness": 3,
      "hairBodySeparation": 3,
      "settingSimplicity": 3,
      "migrationConfidence": 3
    },
    "note": ""
  }
}
```

Export a smaller SpriTalk migration profile when the current candidate should be tried in SpriTalk:

```json
{
  "schema": "spritalk.motionProfile.v1",
  "sourcePartsDir": "04_spritalk_parts",
  "generatedBy": "PachiPakuGen Motion Lab",
  "blink": { "mode": "keepExisting" },
  "lipSync": {
    "method": "bridge",
    "attackMs": 90,
    "releaseMs": 160,
    "crossfadeMs": 50,
    "restBias": 0.25,
    "shapeSmoothing": 0.65,
    "bridgeBias": 0.45
  },
  "layerMotion": {
    "mode": "mesh",
    "preset": "normal",
    "breathAmplitude": 1.0,
    "bodySwayAmplitude": 1.0,
    "hairFrontDelayMs": 180,
    "hairBackDelayMs": 280
  },
  "spritalkProceduralAnimation": {
    "breathing": { "enabled": true, "amplitude": 4.5, "speed": 0.5 },
    "idleSway": { "enabled": true, "amplitudeX": 2.4, "amplitudeY": 1.4, "speed": 0.9, "reduceOnSpeech": true },
    "hairSway": { "enabled": true, "amplitude": 2.5, "speed": 0.77, "rotationAmount": 0.009 },
    "hairBackSway": { "enabled": true, "amplitude": 2.1, "speed": 0.54, "rotationAmount": 0.007 }
  },
  "runtimeRequirements": {
    "lipSyncRenderer": "neutralBridgeOpacityBlend",
    "layerRenderer": "stripWarpExtension"
  }
}
```

## 7. Evaluation Metrics

Mechanical:

- mouth target changes per second
- mouth frame index delta max
- mouth blend jerk score
- layer bbox movement max
- alpha popping count
- generated transition file count
- preview fps

Human review:

- mouth smoothness
- vowel readability
- body naturalness / stiffness reduction
- hair/body separation
- setting simplicity
- SpriTalk migration confidence
- candidate verdict: undecided / promising / hold / reject

## 8. First Implementation Slice

Implement only enough to compare real motion choices:

1. Add `motion_lab` mode and mode-select card.
2. Load `04_spritalk_parts` folder and validate required files.
3. Render a synchronized two-lane canvas preview: baseline direct switch/simple motion on the left, candidate smoother/layered spring on the right.
4. Add deterministic vowel timeline.
5. Add Layered Spring transform for `body`, `hair`, `hair_back`.
6. Add PuruPuru-style strip/mesh sway preview for `hair` and `hair_back`.
7. Save/load `motion-preview-manifest.json`.
8. Export `spritalk-motion-profile.json` as the minimal migration contract for SpriTalk.

Current implementation note:

- The first UI slice uses side-by-side preview rather than a single toggle-only canvas.
- Baseline lane intentionally keeps direct mouth switching and simpler unified motion.
- Candidate lane applies the selected mouth method plus selectable layer motion.
- Mouth methods are `direct`, `smooth`, and `bridge`; `bridge` mixes a neutral/closed mouth into vowel-to-vowel transitions to target the current abrupt lip-sync issue directly.
- `spring` mode uses whole-layer transforms for `body`, `hair`, and `hair_back`.
- `mesh` mode keeps the body/mouth alignment stable while drawing `hair` and `hair_back` as horizontal strips whose tips lag and sway more than their roots.
- PuruPuruPNGTuber-style file aliases such as `front-hair.png`, `back-hair.png`, `eyes-open-mouth-open.png`, and `eyes-open-mouth-closed.png` are accepted as preview inputs.
- Settings can be saved to and loaded from `motion-preview-manifest.json` in the selected parts folder.
- The selected candidate can be exported as `spritalk-motion-profile.json`; this keeps blink unchanged and carries lip-sync, layer-motion, and a SpriTalk-shaped `spritalkProceduralAnimation` object into the migration step.
- The concrete SpriTalk migration notes live in `docs/SPRITALK_MOTION_PROFILE_MIGRATION_PLAN.md`.
- Human review scores, verdict, and adoption notes are saved in the same manifest so the final SpriTalk migration candidate has a traceable reason.

Leave for later:

- RIFE transition graph generation.
- Audio file analysis.
- Reference video lane.
- SpriTalk export/import changes.

## 9. Open Questions

- SpriTalk currently imports PNG animation folders. Confirm whether it can accept per-frame transform metadata, or whether transforms must be baked to image sequences.
- Confirm whether the final SpriTalk runtime should evaluate springs live or consume baked preview frames.
- Confirm whether mouth smoothing should live in SpriTalk as a runtime behavior, or whether PachiPakuGen should export additional transition frames.
- Check commercial/distribution constraints before adding any AI portrait animation dependency.
