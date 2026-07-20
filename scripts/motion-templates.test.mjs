import test, { after } from "node:test";
import assert from "node:assert/strict";
import { createServer } from "vite";

const vite = await createServer({
  logLevel: "silent",
  server: { middlewareMode: true },
  appType: "custom",
});
after(() => vite.close());

const {
  MOTION_LAB_TEMPLATE_LAYOUT,
  MOTION_LAB_TEMPLATES,
} = await vite.ssrLoadModule("/src/motionLab/constants.ts");
const {
  MOTION_LAB_DEFAULT_SETTINGS,
  motionLabSettingsReducer,
  toRenderSettings,
} = await vite.ssrLoadModule("/src/motionLab/useMotionLabSettings.ts");
const { buildSpritalkMotionProfile, buildMotionLabManifest } = await vite.ssrLoadModule("/src/motionLab/manifest.ts");

const templateKeys = Object.values(MOTION_LAB_TEMPLATE_LAYOUT)
  .flatMap(rows => rows.flatMap(row => [row.small, row.large]));

test("SpriTalk profile keeps the parts directory portable", () => {
  const profile = buildSpritalkMotionProfile(
    MOTION_LAB_DEFAULT_SETTINGS,
    "C:\\temporary\\04_spritalk_parts",
  );
  assert.equal(profile.sourcePartsDir, ".");
});

test("motion-preview-manifest keeps the parts directory portable (no local path leak)", () => {
  const manifest = buildMotionLabManifest(
    MOTION_LAB_DEFAULT_SETTINGS,
    "C:\\Users\\someone\\Desktop\\新しいフォルダー (13)\\04_spritalk_parts",
  );
  assert.equal(manifest.sourcePartsDir, ".");
});

test("each motion engine exposes two movement styles in small and large sizes", () => {
  assert.equal(templateKeys.length, 8);
  assert.equal(new Set(templateKeys).size, 8);

  for (const [engine, rows] of Object.entries(MOTION_LAB_TEMPLATE_LAYOUT)) {
    assert.equal(rows.length, 2);
    const keys = rows.flatMap(row => [row.small, row.large]);
    assert.equal(keys.length, 4);
    for (const key of keys) {
      const template = MOTION_LAB_TEMPLATES[key];
      assert.ok(template, `${engine}/${key} must exist`);
      assert.equal(template.engine, engine);
      assert.equal(template.hairEngine, engine === "wave" ? "wave" : "spring");
    }
  }
});

test("large variants are stronger than the matching small variants", () => {
  const magnitudeFields = [
    "breath", "bodySway", "pyokoBounce", "parallax", "hairWind", "hairDrive",
    "hairBackScale", "armSwayAmp", "armMaxAngle", "chestMax",
  ];

  for (const rows of Object.values(MOTION_LAB_TEMPLATE_LAYOUT)) {
    for (const row of rows) {
      const small = MOTION_LAB_TEMPLATES[row.small];
      const large = MOTION_LAB_TEMPLATES[row.large];
      let hasStrictIncrease = false;
      for (const field of magnitudeFields) {
        assert.ok(large[field] >= small[field], `${row.label}/${field} must not shrink`);
        hasStrictIncrease ||= large[field] > small[field];
      }
      assert.equal(hasStrictIncrease, true, `${row.label} needs a visible size difference`);
    }
  }
});

test("all eight templates apply their values while preserving material-specific pivots", () => {
  for (const key of templateKeys) {
    const template = MOTION_LAB_TEMPLATES[key];
    const source = {
      ...MOTION_LAB_DEFAULT_SETTINGS,
      effects: { ...MOTION_LAB_DEFAULT_SETTINGS.effects },
      pivots: { hair: { x: 123, y: 45 } },
      rangesDeg: { hair: 18 },
      swingScale: { hair: 1.4 },
    };
    const result = motionLabSettingsReducer(source, { type: "applyTemplate", key });

    assert.equal(result.templateName, key);
    assert.equal(result.engineFamily, template.engine);
    assert.equal(result.hairWind, template.hairWind);
    assert.equal(result.pyokoBounce, template.pyokoBounce);
    assert.equal(result.armSwayAmp, template.armSwayAmp);
    assert.deepEqual(result.pivots, source.pivots);
    assert.deepEqual(result.rangesDeg, source.rangesDeg);
    assert.deepEqual(result.swingScale, source.swingScale);
  }
});

test("the strength slider scales from the selected template instead of erasing its balance", () => {
  const selected = motionLabSettingsReducer(MOTION_LAB_DEFAULT_SETTINGS, {
    type: "applyTemplate",
    key: "yurari",
  });
  const stronger = motionLabSettingsReducer(selected, { type: "applyIntensity", value: 1.25 });
  const template = MOTION_LAB_TEMPLATES.yurari;

  assert.equal(stronger.breathAmplitude, Number((template.breath * 1.25).toFixed(3)));
  assert.equal(stronger.bodySwayAmplitude, Number((template.bodySway * 1.25).toFixed(3)));
  assert.equal(stronger.hairWind, Number((template.hairWind * 1.25).toFixed(4)));
  assert.equal(stronger.hairWaveStrength, Number((template.hairWaveStrength * 1.25).toFixed(3)));
  assert.equal(stronger.pyokoBounce, Number((template.pyokoBounce * 1.25).toFixed(2)));
  assert.equal(stronger.parallaxScale, Number((template.parallax * 1.25).toFixed(3)));
});

test("100 percent is the stable natural baseline for every template", () => {
  for (const key of templateKeys) {
    const template = MOTION_LAB_TEMPLATES[key];
    const selected = motionLabSettingsReducer(MOTION_LAB_DEFAULT_SETTINGS, {
      type: "applyTemplate",
      key,
    });
    const weaker = motionLabSettingsReducer(selected, { type: "applyIntensity", value: 0.5 });
    const stronger = motionLabSettingsReducer(weaker, { type: "applyIntensity", value: 1.5 });
    const restored = motionLabSettingsReducer(stronger, { type: "applyIntensity", value: 1 });

    assert.equal(restored.intensity, 1, key);
    assert.equal(restored.breathAmplitude, template.breath, key);
    assert.equal(restored.bodySwayAmplitude, template.bodySway, key);
    assert.equal(restored.hairWind, template.hairWind, key);
    assert.equal(restored.hairDrive, template.hairDrive, key);
    assert.equal(restored.hairWaveStrength, template.hairWaveStrength, key);
    assert.equal(restored.pyokoBounce, template.pyokoBounce, key);
    assert.equal(restored.armMaxAngle, template.armMaxAngle, key);
    assert.equal(restored.chestMax, template.chestMax, key);
    assert.equal(restored.parallaxScale, template.parallax, key);
    assert.equal(restored.hairK, template.hairK, key);
    assert.equal(restored.hairC, template.hairC, key);
    assert.equal(restored.hairBackScale, template.hairBackScale, key);
    assert.equal(restored.armSwayAmp, template.armSwayAmp, key);
  }
});

test("150 percent stays inside the renderer safety envelope", () => {
  for (const key of templateKeys) {
    const selected = motionLabSettingsReducer(MOTION_LAB_DEFAULT_SETTINGS, {
      type: "applyTemplate",
      key,
    });
    const result = motionLabSettingsReducer(selected, { type: "applyIntensity", value: 1.5 });

    assert.ok(result.breathAmplitude <= 1.6, key);
    assert.ok(result.bodySwayAmplitude <= 1.8, key);
    assert.ok(result.hairWind <= 0.06, key);
    assert.ok(result.hairDrive <= 0.2, key);
    assert.ok(result.hairWaveStrength <= 2, key);
    assert.ok(result.pyokoBounce <= 7, key);
    assert.ok(result.armMaxAngle <= 0.3, key);
    assert.ok(result.chestMax <= 8, key);
    assert.ok(result.parallaxScale <= 1.5, key);
  }
});

test("strength input is clamped instead of allowing accidental overdrive", () => {
  const selected = motionLabSettingsReducer(MOTION_LAB_DEFAULT_SETTINGS, {
    type: "applyTemplate",
    key: "breeze",
  });
  const tooHigh = motionLabSettingsReducer(selected, { type: "applyIntensity", value: 9 });
  const invalid = motionLabSettingsReducer(selected, { type: "applyIntensity", value: Number.NaN });

  assert.equal(tooHigh.intensity, 1.5);
  assert.equal(invalid.intensity, 1);
});

test("ear twitch follows effect-wide ON, OFF, and solo controls", () => {
  const allOn = motionLabSettingsReducer(MOTION_LAB_DEFAULT_SETTINGS, {
    type: "allEffects",
    value: true,
  });
  assert.ok(Object.values(allOn.effects).every(Boolean));

  const allOff = motionLabSettingsReducer(allOn, { type: "allEffects", value: false });
  assert.ok(Object.values(allOff.effects).every(value => !value));

  const solo = motionLabSettingsReducer(allOn, { type: "soloEffect", key: "earTwitch" });
  assert.equal(solo.effects.earTwitch, true);
  assert.ok(Object.entries(solo.effects).every(([key, value]) => key === "earTwitch" ? value : !value));
});

test("ear effect toggle gates rendering without losing its detailed motion settings", () => {
  const configured = {
    ...MOTION_LAB_DEFAULT_SETTINGS,
    effects: { ...MOTION_LAB_DEFAULT_SETTINGS.effects, earTwitch: false },
    earTwitchMode: "tilt",
    earTwitchScale: 1.35,
  };
  const disabled = toRenderSettings(configured, { pivotEditPart: null });
  const enabledSettings = motionLabSettingsReducer(configured, {
    type: "setEffect",
    key: "earTwitch",
    value: true,
  });
  const enabled = toRenderSettings(enabledSettings, { pivotEditPart: null });

  assert.equal(disabled.earTwitch, false);
  assert.equal(enabled.earTwitch, true);
  assert.equal(enabled.earTwitchMode, "tilt");
  assert.equal(enabled.earTwitchScale, 1.35);
  assert.equal(enabledSettings.earTwitchMode, configured.earTwitchMode);
  assert.equal(enabledSettings.earTwitchScale, configured.earTwitchScale);
});

test("SpriTalk profile carries every previously preview-only field", () => {
  const settings = {
    ...MOTION_LAB_DEFAULT_SETTINGS,
    hairEngine: "wave",
    hairWaveStrength: 1.4,
    hairBackScale: 0.72,
    hairMotionStrength: 1.2,
    pyokoBounce: 5,
    engineFamily: "wave",
    blinkRate: 2,
    liftStrength: 1.6,
    armSwayAmp: 1.3,
    armPivotRatio: 0.25,
    armBehindBody: true,
    glanceStrength: 1.8,
    intensity: 1.25,
    effects: { ...MOTION_LAB_DEFAULT_SETTINGS.effects, lift: true, glance: true },
  };
  const profile = buildSpritalkMotionProfile(settings, "C:\\workspace\\04_spritalk_parts");

  assert.equal(profile.physics.hair.engine, "wave");
  assert.equal(profile.physics.hair.waveMode, true);
  assert.equal(profile.physics.hair.waveStrength, 1.4);
  assert.equal(profile.physics.hair.backScale, 0.72);
  assert.equal(profile.physics.hair.motionStrength, 1.2);
  assert.equal(profile.physics.engineFamily, "wave");
  assert.equal(profile.physics.pyoko.enabled, true);
  assert.equal(profile.physics.pyoko.amplitudePx, 5);
  assert.equal(profile.physics.arm.swayAmp, 1.3);
  assert.equal(profile.physics.arm.pivotRatio, 0.25);
  assert.equal(profile.physics.arm.behindBody, true);
  assert.equal(profile.physics.arm.lift.strength, 1.6);
  assert.equal(profile.physics.glance.enabled, true);
  assert.equal(profile.physics.glance.strength, 1.8);
  assert.equal(profile.blink.rate, 2);
  assert.equal(profile.uiIntensity, 1.25);
  assert.deepEqual(profile.effects, settings.effects);
  // waveエンジン＋髪の揺れONなら、layerModeがmeshでなくても波レンダラーを要求する
  assert.equal(profile.runtimeRequirements.layerRenderer, "waveWarpRenderer");
});

test("SpriTalk profile zeroes strength fields when their effect toggle is off", () => {
  const settings = {
    ...MOTION_LAB_DEFAULT_SETTINGS,
    pyokoBounce: 5,
    liftStrength: 1.6,
    armSwayAmp: 1.3,
    glanceStrength: 1.8,
    hairBackScale: 0.72,
    hairMotionStrength: 1.2,
    blinkRate: 2,
    effects: {
      ...MOTION_LAB_DEFAULT_SETTINGS.effects,
      pyoko: false, lift: false, arm: false, glance: false,
      hairBack: false, hairMotion: false, blink: false,
    },
  };
  const profile = buildSpritalkMotionProfile(settings, "C:\\workspace\\04_spritalk_parts");

  assert.equal(profile.physics.pyoko.enabled, false);
  assert.equal(profile.physics.pyoko.amplitudePx, 0);
  assert.equal(profile.physics.arm.lift.strength, 0);
  assert.equal(profile.physics.arm.swayAmp, 0);
  assert.equal(profile.physics.glance.enabled, false);
  assert.equal(profile.physics.glance.strength, 0);
  assert.equal(profile.physics.hair.backScale, 0);
  assert.equal(profile.physics.hair.motionStrength, 0);
  assert.equal(profile.blink.rate, 0);
});

test("SpriTalk profile keeps mesh layer renderer when hair engine is spring", () => {
  const meshSpring = buildSpritalkMotionProfile(
    { ...MOTION_LAB_DEFAULT_SETTINGS, layerMode: "mesh", hairEngine: "spring" },
    "C:\\workspace\\04_spritalk_parts",
  );
  assert.equal(meshSpring.runtimeRequirements.layerRenderer, "stripWarpExtension");

  const simpleSpring = buildSpritalkMotionProfile(
    { ...MOTION_LAB_DEFAULT_SETTINGS, layerMode: "simple", hairEngine: "spring" },
    "C:\\workspace\\04_spritalk_parts",
  );
  assert.equal(simpleSpring.runtimeRequirements.layerRenderer, "existingProceduralAnimator");
});
