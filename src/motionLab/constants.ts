import type {
  MotionLabEffectKey,
  MotionLabMouthKey,
  MotionLabPreset,
  MotionLabReviewKey,
  MotionLabTemplate,
  MotionLabTimelineEvent,
} from "./types";

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
  { key: "pyoko", label: "発話バウンス", hint: "話すたびに体がぴょこんと弾む（PuruPuru式）" },
  { key: "hairMotion", label: "髪の揺れ", hint: "前髪・後ろ髪のバネ/波/房の揺れ全般" },
  { key: "hairBack", label: "後ろ髪の揺れ", hint: "後ろ髪だけの揺れ（髪の揺れON時に有効）" },
  { key: "parallax", label: "首振りパララックス", hint: "レイヤー深度差で顔がゆっくり左右を向く（852話式）" },
  { key: "glance", label: "ランダムグランス", hint: "たまに顔向き・視線がふっと変わる（852話式）" },
  { key: "gaze", label: "視線ドリフト", hint: "瞳がわずかに泳ぎ、発話で正面に戻る（要eyewhite/irides素材）" },
  { key: "blink", label: "自動まばたき", hint: "2〜10秒間隔で瞬き（PuruPuru参考）" },
  { key: "arm", label: "腕揺れ", hint: "腕の振り子スイング（要arm素材）" },
  { key: "lift", label: "肩の弾み", hint: "体の弾みに少し遅れて肩がぽよん（発話バウンスの二次揺れ・要arm素材）" },
  { key: "chest", label: "胸揺れ", hint: "体の弾み・呼吸に遅れて追従する二次揺れ（要chest素材）" },
  { key: "earTwitch", label: "獣耳ピコピコ", hint: "たまに耳が上下にピコッと跳ねる（要sway_ear素材）" },
];

export const MOTION_LAB_EFFECT_DEFAULTS: Record<MotionLabEffectKey, boolean> = {
  breath: true, bodySway: true, pyoko: true, hairMotion: true, hairBack: true,
  parallax: true, glance: true, gaze: true, blink: true, arm: true, lift: true,
  chest: true, earTwitch: false,
};

/** ソロ時に一緒にONにする依存エフェクト（単体では画面に現れないもの） */
export const MOTION_LAB_EFFECT_SOLO_DEPS: Partial<Record<MotionLabEffectKey, MotionLabEffectKey[]>> = {
  hairBack: ["hairMotion"],
  glance: ["parallax", "gaze"],
};


export const MOTION_LAB_TEMPLATES: Record<string, MotionLabTemplate> = {
  // === 852話式（バネ・チェーンリグ）===
  // 配信画面の隅に置く「静かに生きてる」アバター向け。動きは小さく上品に
  calm: {
    label: "おちつき",
    description: "小さく上品な待機。呼吸と遅延追従が主役、後ろ髪は控えめ",
    engine: "hachigoni",
    preset: "calm", layerMode: "spring", hairEngine: "spring", hairWaveStrength: 1,
    hairK: 80, hairC: 10, hairWind: 0.006, hairDrive: 0.02, hairBackScale: 0.35,
    breath: 0.85, bodySway: 0.7, pyokoBounce: 1.5, parallax: 0.5, randomGlance: true,
    strands: false, armSwayAmp: 0.6, armMaxAngle: 0.08, chestMax: 3, earTwitch: true,
  },
  // 推奨バランス。mesh＋房分割で髪の質感を出しつつ、体の動きは自然な範囲
  standard: {
    label: "標準",
    description: "推奨バランス。mesh＋房分割の髪質感と自然な体の動き",
    engine: "hachigoni",
    preset: "normal", layerMode: "mesh", hairEngine: "spring", hairWaveStrength: 1,
    hairK: 70, hairC: 7, hairWind: 0.012, hairDrive: 0.03, hairBackScale: 0.5,
    breath: 1, bodySway: 1, pyokoBounce: 3, parallax: 1, randomGlance: true,
    strands: true, armSwayAmp: 1, armMaxAngle: 0.12, chestMax: 5, earTwitch: true,
  },
  // 髪を見せるための「風のある日」。房分割＋強め風、体は控えめ
  breeze: {
    label: "そよかぜ",
    description: "髪見せ用。房分割＋強め風で毛先がそよぐ、体は控えめ",
    engine: "hachigoni",
    preset: "normal", layerMode: "mesh", hairEngine: "spring", hairWaveStrength: 1,
    hairK: 55, hairC: 5, hairWind: 0.03, hairDrive: 0.08, hairBackScale: 0.6,
    breath: 0.95, bodySway: 0.9, pyokoBounce: 2, parallax: 0.8, randomGlance: true,
    strands: true, armSwayAmp: 0.9, armMaxAngle: 0.1, chestMax: 4, earTwitch: true,
  },
  // === ろてじん式（波揺れ・ぷるぷる）===
  // ゆったりした波が髪をたゆたわせる。体は静かめ
  yurari: {
    label: "ゆらり",
    description: "ゆったりした波が髪全体をたゆたわせる。体は静かめ",
    engine: "rotejin",
    preset: "calm", layerMode: "spring", hairEngine: "wave", hairWaveStrength: 0.8,
    hairK: 70, hairC: 8, hairWind: 0.01, hairDrive: 0.03, hairBackScale: 0.55,
    breath: 0.9, bodySway: 0.75, pyokoBounce: 2, parallax: 0.6, randomGlance: true,
    strands: false, armSwayAmp: 0.7, armMaxAngle: 0.08, chestMax: 3, earTwitch: true,
  },
  // PNGTuberらしい元気な弾み。波揺れ＋強めの発話バウンス
  purupuru: {
    label: "ぷるぷる",
    description: "PNGTuber風の元気な弾み。波揺れ＋強め発話バウンス＋大きめ腕振り",
    engine: "rotejin",
    preset: "lively", layerMode: "mesh", hairEngine: "wave", hairWaveStrength: 1.15,
    hairK: 60, hairC: 6, hairWind: 0.015, hairDrive: 0.05, hairBackScale: 0.45,
    breath: 1.1, bodySway: 1.15, pyokoBounce: 5.5, parallax: 1.15, randomGlance: true,
    strands: false, armSwayAmp: 1.7, armMaxAngle: 0.2, chestMax: 6.5, earTwitch: true,
  },
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
export const MOTION_LAB_CHEST_DEFAULTS = { k: 28, c: 6, max: 6 };
// 自動瞬き（ろてじん氏 PuruPuruPNGTuber参考。animation-lab NS.P.blink と同値）
export const MOTION_LAB_BLINK_DEFAULTS = { closeMs: 90, openMs: 130, intervalMin: 2, intervalMax: 10 };
// パララックス首振り（852話氏 Anime2.5DRig由来・設計書§8.3）
// shiftRatio ≈ キャンバス幅の2.5% / shear最大0.06 / 駆動=ノイズドリフト＋発話頷きバネ
export const MOTION_LAB_PARALLAX_DEFAULTS = { shiftRatio: 0.045, shearMax: 0.08, driftSpeed: 0.35 };
export const MOTION_LAB_NOD_DEFAULTS = { k: 120, c: 12, impulse: 34, maxPx: 8 };
// 視線ドリフト（852話氏由来＋SpriTalk特性§8.6: 基本正面・発話開始で正面復帰）
export const MOTION_LAB_GAZE_DEFAULTS = { rangeRatio: 0.008, driftSpeed: 0.1, smoothTime: 0.25 };
// ハイライトドリフト（ろてじん氏の目元演出参考・§8.1 #6: ±1〜2px）
export const MOTION_LAB_HIGHLIGHT_DEFAULTS = { driftPx: 1.5, speed: 0.35 };
export const MOTION_LAB_SWAY_DEFAULTS = { segments: 3, k: 60, c: 6, noise: 0.008, maxAngle: 0.35 };
// 獣耳ピコピコ: 数秒間隔の縦バネ撃力（上下に「ピコッ」と跳ねる）＋確率で短い連続ツイッチ。
// bounce=縦撃力(px/s)、k/c=速く少し弾む縦バネ、rotKick=ごく僅かな回転（有機感用、rad/s）
export const MOTION_LAB_EAR_TWITCH = {
  bounce: 110, k: 260, c: 13, maxPx: 9, rotKick: 1.2,
  intervalMin: 3, intervalRange: 6, doubleMin: 0.12, doubleRange: 0.12,
};
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
