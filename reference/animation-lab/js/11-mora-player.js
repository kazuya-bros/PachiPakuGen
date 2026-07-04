/**
 * モーラタイムライン再生器
 * 移植元: SpriTalk src/shared/utils/vowel-timing.ts (getVowelAtTime)
 * 追加: A4開度エンベロープ（母音別目標開度 + attack/release スムージング）
 *
 * 全セル・全方式は本プレイヤーが毎フレーム出力する単一の MoraState で
 * 駆動される（同時刻・同駆動信号 = 比較の公平性を保証）。
 */
(function (NS) {
  'use strict';

  const U = NS.U;

  /**
   * 指定時刻の母音を返す（子音区間もそのモーラの母音として扱う）
   * SpriTalk getVowelAtTime の忠実移植
   */
  function getVowelAtTime(timings, currentTimeMs) {
    for (const timing of timings) {
      const effectiveStart = timing.consonantDuration
        ? timing.startTime - timing.consonantDuration
        : timing.startTime;
      const endTime = timing.startTime + timing.duration;
      if (currentTimeMs >= effectiveStart && currentTimeMs < endTime) {
        return timing.vowel;
      }
    }
    return undefined;
  }

  class MoraPlayer {
    constructor() {
      this.timings = [];
      this.durationMs = 0;
      this.playing = false;
      this.loop = true;
      this.audio = null;        // VOICEVOX連携時の HTMLAudioElement
      this._startPerf = 0;      // performance.now 基準の再生開始時刻
      this._pausedAt = 0;
      // MoraState 追跡
      this._vowel = undefined;
      this._prevVowel = undefined;
      this._vowelChangedAt = 0;
      this._speaking = false;
      this._openness = 0;
      this._envTarget = 0;
      this._onLoop = null;      // ループ時コールバック（方式リセット用）
    }

    /** @param {MoraTiming[]} timings  @param {HTMLAudioElement|null} audio */
    setTimings(timings, audio) {
      this.stop();
      this.timings = timings || [];
      this.audio = audio || null;
      let end = 0;
      for (const t of this.timings) end = Math.max(end, t.startTime + t.duration);
      this.durationMs = end + 200; // 終端に余韻
      if (audio && isFinite(audio.duration) && audio.duration > 0) {
        this.durationMs = Math.max(this.durationMs, audio.duration * 1000);
      }
    }

    play() {
      if (this.playing) return;
      this.playing = true;
      this._startPerf = performance.now() - this._pausedAt;
      if (this.audio) {
        this.audio.currentTime = this._pausedAt / 1000;
        this.audio.play().catch(() => {});
      }
    }

    pause() {
      if (!this.playing) return;
      this._pausedAt = performance.now() - this._startPerf;
      this.playing = false;
      if (this.audio) this.audio.pause();
    }

    stop() {
      this.playing = false;
      this._pausedAt = 0;
      this._vowel = undefined;
      this._prevVowel = undefined;
      this._speaking = false;
      this._openness = 0;
      this._envTarget = 0;
      if (this.audio) {
        this.audio.pause();
        this.audio.currentTime = 0;
      }
    }

    seek(ms) {
      this._pausedAt = U.clamp(ms, 0, this.durationMs);
      if (this.playing) this._startPerf = performance.now() - this._pausedAt;
      if (this.audio) this.audio.currentTime = this._pausedAt / 1000;
    }

    elapsed(now) {
      if (!this.playing) return this._pausedAt;
      // 音声があれば音声時刻を正とする（ズレ防止）
      if (this.audio && !this.audio.paused) return this.audio.currentTime * 1000;
      return now - this._startPerf;
    }

    /**
     * 毎フレーム呼び出し。MoraState を返す。
     * @param {number} now performance.now()
     * @param {number} dt 秒
     * @returns {MoraState}
     */
    update(now, dt) {
      let elapsed = this.elapsed(now);

      // 終端処理
      if (this.playing && elapsed >= this.durationMs) {
        if (this.loop) {
          this._pausedAt = 0;
          this._startPerf = now;
          if (this.audio) {
            this.audio.currentTime = 0;
            this.audio.play().catch(() => {});
          }
          elapsed = 0;
          if (this._onLoop) this._onLoop();
        } else {
          this.pause();
          this._pausedAt = this.durationMs;
          elapsed = this.durationMs;
        }
      }

      const vowel = this.playing ? getVowelAtTime(this.timings, elapsed) : undefined;

      // 母音変化の追跡
      if (vowel !== this._vowel && vowel !== undefined && vowel !== 'cl') {
        if (this._vowel !== undefined && this._vowel !== 'cl') this._prevVowel = this._vowel;
        this._vowel = vowel;
        this._vowelChangedAt = elapsed;
      } else if (vowel !== undefined && vowel !== 'cl') {
        this._vowel = vowel;
      }
      // vowel===undefined（モーラ間）は現在の母音を保持（本体仕様と同じ）

      const speakingNow = this.playing && vowel !== undefined && vowel !== 'pause';
      const speechStarted = speakingNow && !this._speaking;
      this._speaking = speakingNow;

      // ===== A4 開度エンベロープ =====
      const P = NS.P.envelope;
      let rawTarget;
      if (vowel === 'cl' || vowel === undefined) {
        rawTarget = this._envTarget; // 促音・モーラ間: 直前の目標を維持
      } else {
        rawTarget = NS.VOWEL_OPENNESS[vowel] != null ? NS.VOWEL_OPENNESS[vowel] : 0;
      }
      if (!this.playing) rawTarget = 0;
      this._envTarget = rawTarget;

      if (P.mode === 'rect') {
        this._openness = rawTarget;
      } else {
        // 指数スムージング: 立ち上がり attackMs / 立ち下がり releaseMs
        const tau = (rawTarget > this._openness ? P.attackMs : P.releaseMs) / 1000;
        const alpha = 1 - Math.exp(-dt / Math.max(0.001, tau));
        this._openness += (rawTarget - this._openness) * alpha;
      }

      return {
        elapsed,
        vowel: this.playing ? vowel : undefined,
        prevVowel: this._prevVowel,
        vowelChangedAt: this._vowelChangedAt,
        speaking: speakingNow,
        speechStarted,
        openness: U.clamp(this._openness, 0, 1),
        rawTarget,
        playing: this.playing,
      };
    }
  }

  NS.getVowelAtTime = getVowelAtTime;
  NS.MoraPlayer = MoraPlayer;
})(window.AnimLab);
