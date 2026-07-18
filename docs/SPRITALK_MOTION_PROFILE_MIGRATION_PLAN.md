# SpriTalk Motion Profile Migration Plan

## Purpose

PachiPakuGen Motion Lab now exports `spritalk-motion-profile.json`.
This file is the migration contract for bringing the selected Motion Lab result into SpriTalk without changing the existing blink behavior.

The target is SpriTalk LayeredEmotion, using the existing `04_spritalk_parts` layout:

- `body.png`
- `hair.png`
- `hair_back.png`
- `eye/`
- `mouth_a/`, `mouth_i/`, `mouth_u/`, `mouth_e/`, `mouth_o/`
- `spritalk-motion-profile.json`

## Current SpriTalk Evidence

Checked local repo: `E:\develop\spritalk`.

- Layered sprite import already accepts root `body.png`, `hair.png`, `hair_back.png`, and directories `eye/`, `mouth/`, `mouth_a/` through `mouth_o/`.
  Source: `src/main/settings/layered-sprite-importer.ts`
- LayeredEmotion currently switches mouth layers through `vowelToMouthLayer()` and `activeMouthLayer`.
  Sources: `src/shared/utils/vowel-timing.ts`, `src/windows/character/components/CharacterApp.tsx`
- Mouth rendering currently pre-renders all mouth frames but uses `visibility`, so it can show one mouth frame at a time.
  Source: `src/windows/character/components/CharacterApp.tsx`
- Existing procedural body/hair motion is configured through `Character.display.proceduralAnimation`.
  Sources: `src/shared/types/store.ts`, `src/shared/animation/procedural-animator.ts`
- Existing `ProceduralAnimator` applies whole-layer transforms to the wrapper, `hair`, and `hair_back`; it does not yet support strip/mesh deformation.
  Source: `src/shared/animation/procedural-animator.ts`

## Profile Schema

`spritalk-motion-profile.json` uses:

```json
{
  "schema": "spritalk.motionProfile.v1",
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
    "breathAmplitude": 1,
    "bodySwayAmplitude": 1,
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

## Migration Slice 1: Existing Procedural Motion

Goal: make SpriTalk consume the profile for body/hair motion with the current renderer.

1. Add profile type to SpriTalk shared types.
2. Extend layered import or settings import to detect `spritalk-motion-profile.json`.
3. When a profile exists, copy `spritalkProceduralAnimation` into `character.display.proceduralAnimation`.
4. Keep `blink.mode = keepExisting`; do not alter eye-frame logic.
5. Ignore `layerMotion.mode = mesh` initially unless `runtimeRequirements.layerRenderer` is supported.

This gives immediate richer breathing, idle sway, front hair, and back hair without changing render architecture.

## Migration Slice 2: Smooth Mouth State

Goal: reduce abrupt vowel-to-vowel switches for LayeredEmotion.

Add a small runtime state object in `CharacterApp.tsx`:

```ts
type SmoothMouthRuntime = {
  activeLayer: SpriteLayerName | null;
  previousLayer: SpriteLayerName | null;
  neutralLayer: SpriteLayerName | null;
  frameIndex: number;
  blend: number;
  bridgeAlpha: number;
  transitionStartMs: number;
};
```

For `lipSync.method`:

- `baseline`: current `vowelToMouthLayer()` behavior.
- `smooth`: keep single active layer, but use profile `attackMs`, `releaseMs`, and `shapeSmoothing` instead of fixed `stepDuration`.
- `bridge`: during vowel-to-vowel transitions, blend previous mouth, closed/default mouth, and target mouth using `bridgeBias`.

The current renderer uses `visibility`, so `bridge` needs opacity-based rendering:

- Render active previous mouth frame with opacity `(1 - blend) * (1 - bridgeAlpha)`.
- Render neutral/default mouth frame with opacity `bridgeAlpha`.
- Render target mouth frame with opacity `blend * (1 - bridgeAlpha)`.
- Keep all non-mouth layers unchanged.

## Migration Slice 3: Mesh/Strip Layer Motion

Goal: bring the PuruPuru-style `mesh` preview into SpriTalk.

The current `ProceduralAnimator` only applies one transform per element.
For `runtimeRequirements.layerRenderer = stripWarpExtension`, add a separate renderer path rather than overloading existing image transforms:

- Keep the current `existingProceduralAnimator` path for `spring`.
- For `mesh`, render `hair` and `hair_back` through a canvas or CSS clip-strip container.
- Root area should move less; lower strips should lag and sway more.
- Body and mouth alignment must remain stable, matching Motion Lab.

This is a later slice. It should not block Slice 1 or Slice 2.

## Verification Plan

In PachiPakuGen:

- Export `04_spritalk_parts`.
- Open Motion Lab and choose one candidate.
- Save `motion-preview-manifest.json`.
- Export `spritalk-motion-profile.json`.

In SpriTalk:

- Import the same folder as a LayeredEmotion.
- Confirm `body`, `hair`, `hair_back`, `eye`, and all `mouth_*` layers are detected.
- Confirm blink remains unchanged.
- Confirm `spritalkProceduralAnimation` affects body/hair/back hair.
- For `bridge`, confirm `a -> i -> u -> e -> o` no longer jumps directly between hard mouth shapes.

## Non-goals

- Do not replace SpriTalk's existing blink logic.
- Do not require RIFE transition graph output for the first SpriTalk migration.
- Do not make MPNG video generation a dependency of LayeredEmotion motion.
