import type {
  MotionLabEffectKey,
  MotionLabEngineFamily,
  MotionLabMouthKey,
  MotionLabPreset,
  MotionLabReviewKey,
  MotionLabTemplate,
  MotionLabTimelineEvent,
} from "./types";
export { MOTION_LAB_EAR_TWITCH } from "./earTwitch";

export const MOTION_LAB_MOUTH_KEYS: MotionLabMouthKey[] = ["closed", "a", "i", "u", "e", "o"];
export const MOTION_LAB_VOWEL_KEYS: MotionLabMouthKey[] = ["a", "i", "u", "e", "o"];
export const MOTION_LAB_MOUTH_LABELS: Record<MotionLabMouthKey, string> = {
  closed: "閉",
  a: "あ",
  i: "い",
  u: "う",
  e: "え",
  o: "お",
};
// 母音別目標開度（animation-lab検証済み: NS.VOWEL_OPENNESS）
export const MOTION_LAB_TARGET_OPEN: Record<MotionLabMouthKey, number> = {
  closed: 0,
  a: 1,
  i: 0.5,
  u: 0.45,
  e: 0.65,
  o: 0.85,
};
export const MOTION_LAB_DURATION_MS = 3600;
// 各母音を均等に区切り、間に閉じを挟む（あ→閉→い→閉…の規則的な口パク）
export const MOTION_LAB_TIMELINE: MotionLabTimelineEvent[] = [
  { timeMs: 0, mouth: "closed", energy: 0 },
  { timeMs: 250, mouth: "a", energy: 0.9 },
  { timeMs: 600, mouth: "closed", energy: 0.05 },
  { timeMs: 850, mouth: "i", energy: 0.75 },
  { timeMs: 1200, mouth: "closed", energy: 0.05 },
  { timeMs: 1450, mouth: "u", energy: 0.8 },
  { timeMs: 1800, mouth: "closed", energy: 0.05 },
  { timeMs: 2050, mouth: "e", energy: 0.7 },
  { timeMs: 2400, mouth: "closed", energy: 0.05 },
  { timeMs: 2650, mouth: "o", energy: 0.9 },
  { timeMs: 3100, mouth: "closed", energy: 0 },
];

// かな→母音の簡易変換（テキスト口パク再生用）。清濁・拗音・カタカナ対応
export const MOTION_LAB_KANA_VOWEL: Record<string, MotionLabMouthKey> = (() => {
  const rows: Array<[string, MotionLabMouthKey]> = [
    ["あかがさざただなはばぱまやらわぁゃアカガサザタダナハバパマヤラワァャ", "a"],
    ["いきぎしじちぢにひびぴみりぃイキギシジチヂニヒビピミリィ", "i"],
    ["うくぐすずつづぬふぶぷむゆるぅゅゔウクグスズツヅヌフブプムユルゥュヴ", "u"],
    ["えけげせぜてでねへべぺめれぇエケゲセゼテデネヘベペメレェ", "e"],
    ["おこごそぞとどのほぼぽもよろをぉょオコゴソゾトドノホボポモヨロヲォョ", "o"],
  ];
  const map: Record<string, MotionLabMouthKey> = {};
  for (const [chars, vowel] of rows) {
    for (const ch of chars) map[ch] = vowel;
  }
  return map;
})();

/** テキスト→口パクタイムライン変換（1モーラ約140ms、読点等は300msの休止） */
export function motionLabTimelineFromText(text: string): { timeline: MotionLabTimelineEvent[]; durationMs: number } {
  const MORA_MS = 140;
  const PAUSE_MS = 300;
  const timeline: MotionLabTimelineEvent[] = [{ timeMs: 0, mouth: "closed", energy: 0 }];
  let t = 250; // 立ち上がりの間
  for (const ch of text) {
    const vowel = MOTION_LAB_KANA_VOWEL[ch];
    if (vowel) {
      timeline.push({ timeMs: t, mouth: vowel, energy: 0.75 });
      t += MORA_MS;
    } else if (ch === "ん" || ch === "ン") {
      timeline.push({ timeMs: t, mouth: "closed", energy: 0.4 });
      t += MORA_MS;
    } else if (ch === "っ" || ch === "ッ" || ch === "ー") {
      t += MORA_MS; // 直前の口形を維持して時間だけ進める
    } else if ("、。,.！？!? 　".includes(ch)) {
      timeline.push({ timeMs: t, mouth: "closed", energy: 0 });
      t += PAUSE_MS;
    }
    // その他の文字（漢字・英字）は読めないためスキップ
  }
  timeline.push({ timeMs: t, mouth: "closed", energy: 0 });
  return { timeline, durationMs: t + 600 };
}
export const MOTION_LAB_PRESET_FACTORS: Record<MotionLabPreset, { breath: number; body: number; hair: number }> = {
  calm: { breath: 0.65, body: 0.55, hair: 0.55 },
  normal: { breath: 1, body: 1, hair: 1 },
  lively: { breath: 1.25, body: 1.35, hair: 1.45 },
};


export const MOTION_LAB_EFFECT_DEFS: Array<{ key: MotionLabEffectKey; label: string; hint: string }> = [
  { key: "breath", label: "呼吸", hint: "体がゆっくり上下する（3.6秒周期）。頭・髪は少し遅れて追従" },
  { key: "bodySway", label: "体の揺れ", hint: "左右のアイドル揺れ＋わずかな傾き" },
  { key: "pyoko", label: "発話バウンス", hint: "話すたびに体がぴょこんと弾む" },
  { key: "hairMotion", label: "髪の揺れ", hint: "前髪・後ろ髪のバネ/波/房の揺れ全般" },
  { key: "hairBack", label: "後ろ髪の揺れ", hint: "後ろ髪だけの揺れ（髪の揺れON時に有効）" },
  { key: "parallax", label: "首振りパララックス", hint: "レイヤー深度差で顔がゆっくり左右を向く" },
  { key: "glance", label: "ランダム首振り", hint: "たまに顔向きだけがふっと変わる。瞳の位置は動かしません" },
  { key: "gaze", label: "視線の横揺れ", hint: "100%までは控えめ、最大300%では4.8pxまで左右へ揺れます。縦・ランダム移動はせず、瞬き直前に正面へ戻ります（要eyewhite/irides素材）" },
  { key: "irisBreath", label: "瞳の呼吸", hint: "中間値は控えめに保ち、最大では左右の虹彩全体を各中心で±4%伸縮します。瞳孔だけの変形ではありません（要irides素材）" },
  { key: "wetness", label: "目のうるみ", hint: "虹彩の下側へ淡い水面と細い三日月反射を加えます。最大では縮小表示でも分かる強さになります（要irides素材）" },
  { key: "brow", label: "眉の微動", hint: "待機中はごく小さく、発話中は少し持ち上げながら左右の角度をずらして自然に反応させます（要eyebrow素材）" },
  { key: "blink", label: "自動まばたき", hint: "2〜10秒間隔で自然に瞬きする" },
  { key: "arm", label: "腕揺れ", hint: "腕の振り子スイング（要arm素材）" },
  { key: "lift", label: "肩の弾み", hint: "体の弾みに少し遅れて肩がぽよん（発話バウンスの二次揺れ・要arm素材）" },
  { key: "chest", label: "胸部追従", hint: "体・衣服の胸周辺を、呼吸と体の弾みに遅れてやわらかく変形する" },
  { key: "earTwitch", label: "獣耳ピコピコ", hint: "獣耳をときどきピコッと動かします。動き方と付け根の位置は詳細調整で設定します（要sway_ear素材）" },
];

export const MOTION_LAB_EFFECT_DEFAULTS: Record<MotionLabEffectKey, boolean> = {
  breath: true, bodySway: true, pyoko: true, hairMotion: true, hairBack: true,
  parallax: true, glance: true, gaze: true, blink: true, arm: true, lift: true,
  irisBreath: false, wetness: false, brow: true, chest: false, earTwitch: false,
};

/** ソロ時に一緒にONにする依存エフェクト（単体では画面に現れないもの） */
export const MOTION_LAB_EFFECT_SOLO_DEPS: Partial<Record<MotionLabEffectKey, MotionLabEffectKey[]>> = {
  hairBack: ["hairMotion"],
  glance: ["parallax"],
};


export const MOTION_LAB_TEMPLATES: Record<string, MotionLabTemplate> = {
  // === スプリング式: 自然な追従（小 / 大） ===
  calm: {
    label: "おちつき",
    description: "呼吸と首振りに髪が小さく遅れてついてくる、控えめな自然追従",
    engine: "springRig",
    preset: "calm", layerMode: "spring", hairEngine: "spring", hairWaveStrength: 1,
    hairK: 84, hairC: 11, hairWind: 0.004, hairDrive: 0.018, hairBackScale: 0.35,
    breath: 0.85, bodySway: 0.7, pyokoBounce: 1.4, parallax: 0.5, randomGlance: true,
    strands: false, armSwayAmp: 0.6, armMaxAngle: 0.075, chestMax: 2.5, earTwitch: true,
  },
  standard: {
    label: "しなやか",
    description: "体と首の動きに髪・腕・胸が大きめに追従する、存在感のある自然追従",
    engine: "springRig",
    preset: "normal", layerMode: "mesh", hairEngine: "spring", hairWaveStrength: 1,
    hairK: 72, hairC: 9, hairWind: 0.006, hairDrive: 0.034, hairBackScale: 0.48,
    breath: 1, bodySway: 0.95, pyokoBounce: 2.4, parallax: 0.85, randomGlance: true,
    strands: true, armSwayAmp: 0.95, armMaxAngle: 0.11, chestMax: 4, earTwitch: true,
  },
  // === スプリング式: 風のなびき（小 / 大） ===
  softBreeze: {
    label: "そよかぜ",
    description: "体は控えめのまま、毛先と後ろ髪だけを小さく継続してなびかせる",
    engine: "springRig",
    preset: "calm", layerMode: "mesh", hairEngine: "spring", hairWaveStrength: 1,
    hairK: 72, hairC: 9, hairWind: 0.008, hairDrive: 0.025, hairBackScale: 0.42,
    breath: 0.82, bodySway: 0.65, pyokoBounce: 1.2, parallax: 0.5, randomGlance: true,
    strands: true, armSwayAmp: 0.55, armMaxAngle: 0.07, chestMax: 2.5, earTwitch: true,
  },
  breeze: {
    label: "なびき",
    description: "房分けした髪と後ろ髪を風で大きくなびかせる、髪を目立たせたい動き",
    engine: "springRig",
    preset: "normal", layerMode: "mesh", hairEngine: "spring", hairWaveStrength: 1,
    hairK: 70, hairC: 9, hairWind: 0.012, hairDrive: 0.035, hairBackScale: 0.45,
    breath: 0.9, bodySway: 0.8, pyokoBounce: 1.6, parallax: 0.7, randomGlance: true,
    strands: true, armSwayAmp: 0.75, armMaxAngle: 0.09, chestMax: 3.5, earTwitch: true,
  },
  // === ウェーブ式: ゆったり揺れ（小 / 大） ===
  yurari: {
    label: "ゆらり",
    description: "髪全体へ小さく長い波を流す、体の揺れを控えたゆったり動作",
    engine: "wave",
    preset: "calm", layerMode: "spring", hairEngine: "wave", hairWaveStrength: 0.8,
    hairK: 72, hairC: 9, hairWind: 0.006, hairDrive: 0.025, hairBackScale: 0.5,
    breath: 0.88, bodySway: 0.7, pyokoBounce: 1.6, parallax: 0.55, randomGlance: true,
    strands: false, armSwayAmp: 0.6, armMaxAngle: 0.075, chestMax: 2.5, earTwitch: true,
  },
  yurayura: {
    label: "ゆらゆら",
    description: "髪と体をゆっくり大きく揺らし、静かな会話中にも動きを見せる",
    engine: "wave",
    preset: "normal", layerMode: "spring", hairEngine: "wave", hairWaveStrength: 0.95,
    hairK: 68, hairC: 8, hairWind: 0.008, hairDrive: 0.04, hairBackScale: 0.62,
    breath: 0.95, bodySway: 1, pyokoBounce: 2.2, parallax: 0.8, randomGlance: true,
    strands: false, armSwayAmp: 0.9, armMaxAngle: 0.11, chestMax: 3.8, earTwitch: true,
  },
  // === ウェーブ式: 発話の弾み（小 / 大） ===
  pyokori: {
    label: "ぴょこり",
    description: "発話に合わせて体と髪を小さく弾ませる、落ち着いた話し方にも合う動き",
    engine: "wave",
    preset: "normal", layerMode: "mesh", hairEngine: "wave", hairWaveStrength: 0.6,
    hairK: 68, hairC: 8, hairWind: 0.008, hairDrive: 0.03, hairBackScale: 0.4,
    breath: 0.85, bodySway: 0.75, pyokoBounce: 2.6, parallax: 0.65, randomGlance: true,
    strands: false, armSwayAmp: 0.75, armMaxAngle: 0.09, chestMax: 3.2, earTwitch: true,
  },
  purupuru: {
    label: "ぷるぷる",
    description: "発話ごとに体・髪・腕を大きく弾ませる、元気で目立つ動き",
    engine: "wave",
    preset: "lively", layerMode: "mesh", hairEngine: "wave", hairWaveStrength: 0.72,
    hairK: 64, hairC: 8, hairWind: 0.01, hairDrive: 0.04, hairBackScale: 0.5,
    breath: 0.85, bodySway: 0.78, pyokoBounce: 4.2, parallax: 0.9, randomGlance: true,
    strands: false, armSwayAmp: 1.2, armMaxAngle: 0.135, chestMax: 4.8, earTwitch: true,
  },
};

/** 方式ごとのテンプレート配置。行=動きの性格、列=動きの大きさ。 */
export const MOTION_LAB_TEMPLATE_LAYOUT: Record<
  MotionLabEngineFamily,
  Array<{ label: string; small: string; large: string }>
> = {
  springRig: [
    { label: "自然な追従", small: "calm", large: "standard" },
    { label: "風のなびき", small: "softBreeze", large: "breeze" },
  ],
  wave: [
    { label: "ゆったり揺れ", small: "yurari", large: "yurayura" },
    { label: "発話の弾み", small: "pyokori", large: "purupuru" },
  ],
};

// ===== 検証済み物理パラメータ既定値（docs/animation-lab-tech.md §3.4/§4.5/§8.5） =====
export const MOTION_LAB_HAIR_SEGMENTS = 6;
export const MOTION_LAB_HAIR_DEFAULTS = { k: 70, c: 7, wind: 0.012, drive: 0.03 };
export const MOTION_LAB_ARM_DEFAULTS = {
  segments: 3,
  k: 90,
  c: 10,
  coupling: 0.02,
  noise: 0.008,
  maxAngle: 0.12,
  lift: { coupling: 0.08, bounce: 26, max: 6 },
};
export const MOTION_LAB_CHEST_DEFAULTS = { k: 45, c: 12, max: 3.5 };
// 自動瞬き（ろてじん氏 PuruPuruPNGTuber参考。animation-lab NS.P.blink と同値）
export const MOTION_LAB_BLINK_DEFAULTS = {
  centerMs: 190,
  closeMs: 90,
  openMs: 130,
  settleMs: 140,
  intervalMin: 2,
  intervalMax: 10,
};
// パララックス首振り（852話氏 Anime2.5DRig由来・設計書§8.3）
// shiftRatio ≈ キャンバス幅の2.5% / shear最大0.06 / 駆動=ノイズドリフト＋発話頷きバネ
export const MOTION_LAB_PARALLAX_DEFAULTS = { shiftRatio: 0.045, shearMax: 0.08, driftSpeed: 0.35 };
export const MOTION_LAB_NOD_DEFAULTS = { k: 120, c: 12, impulse: 34, maxPx: 8 };
// 視線の横揺れ（基本正面・瞬き直前に正面復帰）
export const MOTION_LAB_GAZE_DEFAULTS = { periodSeconds: 11.5, smoothTime: 0.42, maxRangePx: 4.8 };
// ハイライトドリフト（ろてじん氏の目元演出参考・§8.1 #6: ±1〜2px）
export const MOTION_LAB_HIGHLIGHT_DEFAULTS = { driftPx: 0, speed: 0 };
export const MOTION_LAB_SWAY_DEFAULTS = { segments: 3, k: 60, c: 6, noise: 0.008, maxAngle: 0.35 };
// 固定z順（layer-order.json が無い場合の既定描画順、背面→前面）
export const MOTION_LAB_DEFAULT_DRAW_ORDER: readonly string[] = [
  "hair_back", "body", "chest", "arm_l", "arm_r", "sways", "eye", "mouth", "hair",
];
export const MOTION_LAB_PRESENCE_DEFAULTS = { entryBounce: 1.0, breathOnPause: 1.0, randomizePhase: true };
// パララックス係数（設計書§8.3。プレビュー未適用、プロファイル出力のみ）
export const MOTION_LAB_DEPTH_DEFAULTS: Record<string, number> = {
  hair_back: -0.6,
  body: 0.0,
  arm_l: 0.1,
  arm_r: 0.1,
  eye: 0.35,
  mouth: 0.35,
  hair: 0.8,
};
export const MOTION_LAB_DEFAULT_REVIEW_SCORES: Record<MotionLabReviewKey, number> = {
  mouthSmoothness: 3,
  vowelReadability: 3,
  bodyNaturalness: 3,
  hairBodySeparation: 3,
  settingSimplicity: 3,
  migrationConfidence: 3,
};
export const MOTION_LAB_REVIEW_LABELS: Record<MotionLabReviewKey, string> = {
  mouthSmoothness: "口の滑らかさ",
  vowelReadability: "母音の読みやすさ",
  bodyNaturalness: "身体の自然さ",
  hairBodySeparation: "髪と身体の分離",
  settingSimplicity: "設定の少なさ",
  migrationConfidence: "移植しやすさ",
};
