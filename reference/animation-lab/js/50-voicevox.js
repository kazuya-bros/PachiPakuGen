/**
 * VOICEVOX任意連携（http://localhost:50021）
 * 起動していれば実音声＋実モーラタイミングで検証できる。未起動なら静かに無効化。
 * モーラ抽出は SpriTalk src/main/engines/mora-extractor.ts の移植。
 * （VOICEVOXは Access-Control-Allow-Origin: * を返すため file:// からもfetch可能）
 */
(function (NS) {
  'use strict';

  const BASE = 'http://127.0.0.1:50021';

  function toVowelType(vowel) {
    switch ((vowel || '').toLowerCase()) {
      case 'a': return 'a';
      case 'i': return 'i';
      case 'u': return 'u';
      case 'e': return 'e';
      case 'o': return 'o';
      case 'n': case 'nn': return 'N';
      case 'cl': return 'cl';
      case 'pau': return 'pause';
      default: return 'a';
    }
  }

  /** audio_query レスポンス → MoraTiming[]（mora-extractor.ts 互換） */
  function extractMoraTimings(audioQuery, speedScale) {
    const accentPhrases = audioQuery && audioQuery.accent_phrases;
    if (!Array.isArray(accentPhrases)) return [];
    speedScale = speedScale || 1.0;

    const timings = [];
    const prePhonemeLength = audioQuery.prePhonemeLength || 0;
    let currentTime = (prePhonemeLength / speedScale) * 1000;

    const processMora = (mora) => {
      const consonantDuration = mora.consonant_length
        ? (mora.consonant_length / speedScale) * 1000
        : undefined;
      const vowelDuration = (mora.vowel_length / speedScale) * 1000;
      const startTime = consonantDuration ? currentTime + consonantDuration : currentTime;
      return {
        vowel: toVowelType(mora.vowel),
        text: mora.text,
        startTime,
        duration: vowelDuration,
        consonant: mora.consonant || undefined,
        consonantDuration,
      };
    };

    for (const phrase of accentPhrases) {
      for (const mora of phrase.moras) {
        const t = processMora(mora);
        timings.push(t);
        currentTime += t.duration + (t.consonantDuration || 0);
      }
      if (phrase.pause_mora) {
        const p = processMora(phrase.pause_mora);
        p.vowel = 'pause';
        timings.push(p);
        currentTime += p.duration;
      }
    }
    return timings;
  }

  async function probe() {
    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 1500);
      const res = await fetch(BASE + '/version', { signal: ctrl.signal });
      clearTimeout(timer);
      if (!res.ok) return null;
      return await res.json();
    } catch (e) {
      return null;
    }
  }

  /**
   * テキストを合成し、モーラタイミング＋Audioを返す
   * @returns {Promise<{timings: MoraTiming[], audio: HTMLAudioElement}>}
   */
  async function synthesize(text, speaker = 1) {
    const qRes = await fetch(
      `${BASE}/audio_query?text=${encodeURIComponent(text)}&speaker=${speaker}`,
      { method: 'POST' }
    );
    if (!qRes.ok) throw new Error('audio_query失敗: ' + qRes.status);
    const query = await qRes.json();
    const timings = extractMoraTimings(query, query.speedScale || 1.0);

    const sRes = await fetch(`${BASE}/synthesis?speaker=${speaker}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(query),
    });
    if (!sRes.ok) throw new Error('synthesis失敗: ' + sRes.status);
    const blob = await sRes.blob();
    const audio = new Audio(URL.createObjectURL(blob));
    await new Promise((res) => {
      audio.addEventListener('loadedmetadata', res, { once: true });
      audio.addEventListener('error', res, { once: true });
    });
    return { timings, audio };
  }

  NS.Voicevox = { probe, synthesize, extractMoraTimings };
})(window.AnimLab);
