/**
 * 埋め込みテスト用モーラタイミングデータ
 * VOICEVOXなしでも全方式を同一データで駆動できるようにする。
 * 形式は SpriTalk MoraTiming[] 互換（startTime=母音開始ms、子音はその前）。
 */
(function (NS) {
  'use strict';

  /**
   * 簡易ビルダー: [text, vowel, consonant?, consonantMs?, vowelMs?] の配列から
   * 累積 startTime を計算して MoraTiming[] を生成する。
   * consonantMs/vowelMs 省略時は defaults を使用。
   */
  function build(moras, defaults) {
    const dv = (defaults && defaults.vowelMs) || 110;
    const dc = (defaults && defaults.consonantMs) || 40;
    const timings = [];
    let t = 100; // 開始無音（prePhonemeLength相当）
    for (const m of moras) {
      const [text, vowel, consonant, cMs, vMs] = m;
      const consonantDuration = consonant ? (cMs != null ? cMs : dc) : undefined;
      const duration = vMs != null ? vMs : (vowel === 'pause' ? 260 : dv);
      const startTime = consonantDuration ? t + consonantDuration : t;
      const timing = { vowel, text, startTime, duration };
      if (consonant) {
        timing.consonant = consonant;
        timing.consonantDuration = consonantDuration;
      }
      timings.push(timing);
      t = startTime + duration;
    }
    return timings;
  }

  NS.MORA_PATTERNS = [
    {
      id: 'konnichiwa',
      label: 'こんにちは、今日もいい天気ですね',
      timings: build([
        ['こ', 'o', 'k'], ['ん', 'N'], ['に', 'i', 'n'], ['ち', 'i', 'ch', 60], ['は', 'a', 'h'],
        ['、', 'pause', null, null, 300],
        ['きょ', 'o', 'ky', 55], ['う', 'u'], ['も', 'o', 'm'],
        ['い', 'i'], ['い', 'i'],
        ['て', 'e', 't'], ['ん', 'N'], ['き', 'i', 'k'],
        ['で', 'e', 'd'], ['す', 'u', 's'], ['ね', 'e', 'n', null, 180],
      ]),
    },
    {
      id: 'aiueo',
      label: 'あいうえお（母音明瞭・ゆっくり）',
      timings: build([
        ['あ', 'a', null, null, 240], ['い', 'i', null, null, 240], ['う', 'u', null, null, 240],
        ['え', 'e', null, null, 240], ['お', 'o', null, null, 240],
        ['。', 'pause', null, null, 400],
        ['あ', 'a', null, null, 240], ['い', 'i', null, null, 240], ['う', 'u', null, null, 240],
        ['え', 'e', null, null, 240], ['お', 'o', null, null, 240],
      ]),
    },
    {
      id: 'hayakuchi',
      label: '早口: なまむぎなまごめなまたまご',
      timings: build([
        ['な', 'a', 'n'], ['ま', 'a', 'm'], ['む', 'u', 'm'], ['ぎ', 'i', 'g'],
        ['な', 'a', 'n'], ['ま', 'a', 'm'], ['ご', 'o', 'g'], ['め', 'e', 'm'],
        ['な', 'a', 'n'], ['ま', 'a', 'm'], ['た', 'a', 't'], ['ま', 'a', 'm'], ['ご', 'o', 'g'],
        ['っ', 'cl', null, null, 90], ['！', 'pause', null, null, 300],
      ], { vowelMs: 72, consonantMs: 28 }),
    },
  ];
})(window.AnimLab);
