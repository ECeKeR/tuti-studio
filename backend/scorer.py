# Copyright (c) 2026 Efeberk Çeker
# SPDX-License-Identifier: AGPL-3.0-only

"""
Word/Phrase Scorer — v4 (Breathing-Aware)
N take arasından her kelime veya phrase için en iyi take'i seçer.

Skor kriterleri:
  - Whisper confidence        → yüksek = daha net telaffuz       (%40)
  - Enerji (RMS)              → çok düşük = fısıltı, çok yüksek = patlama  (%25)
  - Breathing / nefes kalitesi → doğal duraklama = dinlenebilir    (%20)
  - Pitch çeşitliliği         → doğal varyasyon = canlı, duygulu    (%15)
"""

import logging
import numpy as np
from aligner import BreathingAnalyzer

log = logging.getLogger(__name__)


class WordScorer:
    def __init__(self):
        pass

    # ------------------------------------------------------------------ #
    #  PHRASE MODE  (önerilen)
    # ------------------------------------------------------------------ #

    def select_best_phrases(
        self,
        aligned_takes: list[dict],
        chunk_text: str,
        crossfade_ms: int = 20,
    ) -> list[dict]:
        """
        Her take'i bir bütün olarak skorlar, en iyi take'i chunk için seçer.
        Aynı chunk'tan tek take kullanır → geçişler daha doğal.
        """
        scores = []
        for idx, take in enumerate(aligned_takes):
            score = self._score_take(take, chunk_text)
            scores.append((score, idx))
            log.debug(f"    Take {idx}: skor={score:.3f}")

        scores.sort(reverse=True)
        best_idx = scores[0][1]
        best_take = aligned_takes[best_idx]

        return [{
            "take_idx": best_idx,
            "audio_path": best_take["path"],
            "start": 0.0,
            "end": best_take["alignment"]["duration"],
            "text": chunk_text,
            "score": scores[0][0],
            "mode": "phrase",
        }]

    # ------------------------------------------------------------------ #
    #  WORD MODE  (daha granüler ama geçiş zor)
    # ------------------------------------------------------------------ #

    def select_best_words(
        self,
        aligned_takes: list[dict],
        chunk_text: str,
        min_segment_words: int = 2,
    ) -> list[dict]:
        """
        Her kelime için en iyi take'i seçer.
        Geçişleri minimize etmek için ardışık aynı take'leri birleştirir.

        min_segment_words: Tek take'den en az bu kadar ardışık kelime seçilmeli.
        Çok sık take değişimi = yapay ses.
        """
        # Her kelime için take skorlarını hesapla
        target_words = chunk_text.lower().split()
        # Noktalama temizle
        import re
        target_words = [re.sub(r"[^\w']", "", w) for w in target_words if w.strip()]

        per_word_scores = []  # [word_idx] -> {take_idx -> score}

        for word_idx, word in enumerate(target_words):
            word_scores = {}
            for take_idx, take in enumerate(aligned_takes):
                word_data = self._find_word(take["alignment"]["words"], word, word_idx)
                if word_data:
                    score = self._score_word(take["path"], word_data)
                else:
                    score = 0.0  # kelime bulunamazsa skip
                word_scores[take_idx] = score
            per_word_scores.append(word_scores)

        # En iyi take'i seç (ama geçişi minimize et)
        selected = self._smooth_selection(per_word_scores, len(aligned_takes), min_segment_words)

        # Sonuçları formatla
        segments = []
        for word_idx, (word, take_idx) in enumerate(zip(target_words, selected)):
            take = aligned_takes[take_idx]
            word_data = self._find_word(take["alignment"]["words"], word, word_idx)
            if word_data is None:
                # Fallback: en iyi genel take kullan
                best_take_idx = max(range(len(aligned_takes)),
                                    key=lambda i: self._score_take(aligned_takes[i], chunk_text))
                take = aligned_takes[best_take_idx]
                take_idx = best_take_idx
                word_data = self._find_word(take["alignment"]["words"], word, word_idx)
                if word_data is None:
                    log.warning(f"    '{word}' kelimesi hiçbir take'de bulunamadı, atlanıyor.")
                    continue

            segments.append({
                "take_idx": take_idx,
                "audio_path": take["path"],
                "start": word_data["start"],
                "end": word_data["end"],
                "text": word,
                "score": per_word_scores[word_idx].get(take_idx, 0.0),
                "mode": "word",
            })

        return segments

    # ------------------------------------------------------------------ #
    #  İÇ YARDIMCILAR
    # ------------------------------------------------------------------ #

    def _score_take(self, take: dict, expected_text: str | None = None) -> float:
        """Tüm take için ortalama skor, metin eşleşmesi cezasıyla birlikte."""
        words = take["alignment"].get("words", [])
        if not words:
            return 0.0
        confidences = [w.get("confidence", 0.5) for w in words]
        avg_confidence = np.mean(confidences)

        # Enerji ve pitch skoru
        energy_score = self._score_energy(take["path"])
        pitch_score = self._score_pitch_variety(take["path"])

        # Breathing skoru — nefessiz okuyan take'leri cezalandır
        # BreathingAnalyzer alignment dict'teki 'breathing' alanından okur
        breathing_score = BreathingAnalyzer.score_take_breathing(take["alignment"])

        # Ağırlıklı ortalama — breathing %20 ağırlıkla
        score = (
            avg_confidence * 0.40 +
            energy_score   * 0.25 +
            breathing_score * 0.20 +
            pitch_score    * 0.15
        )

        if expected_text:
            import difflib
            import re
            
            # Clean target words
            target_clean = re.sub(r"[^\w'\s]", "", expected_text.lower())
            target_words = [w for w in target_clean.split() if w]
            
            # Clean transcribed words
            transcribed_clean = re.sub(r"[^\w'\s]", "", take["alignment"].get("text", "").lower())
            transcribed_words = [w for w in transcribed_clean.split() if w]
            
            if target_words:
                matcher = difflib.SequenceMatcher(None, target_words, transcribed_words)
                match_ratio = matcher.ratio()
                
                # Strict verification for TTS verbatim reading:
                # If similarity is high (>= 0.90), no penalty at all.
                if match_ratio >= 0.90:
                    pass
                # Moderate similarity (0.80 to 0.90), apply a strict scaling penalty
                elif match_ratio >= 0.80:
                    # Map [0.80, 0.90] to [0.30, 1.00]
                    multiplier = 0.3 + (match_ratio - 0.80) * 7.0
                    score = score * multiplier
                # Anything below 0.80 has wrong words, slurs or hallucinations -> failed take
                else:
                    score = score * 0.05
            else:
                score = 0.0

        return float(score)

    def _score_word(self, audio_path: str, word_data: dict) -> float:
        """Tek kelime için skor."""
        confidence = word_data.get("confidence", 0.5)
        start = word_data["start"]
        end = word_data["end"]
        duration = end - start

        # Çok kısa veya çok uzun kelimeler cezalandır
        # İngilizce ortalama kelime ~0.2-0.6s
        duration_penalty = 1.0
        if duration < 0.05:
            duration_penalty = 0.5
        elif duration > 1.5:
            duration_penalty = 0.7

        energy = self._score_energy_segment(audio_path, start, end)
        score = (confidence * 0.6) + (energy * 0.25) + (duration_penalty * 0.15)
        return float(score)

    def _score_energy(self, audio_path: str) -> float:
        """Dosyanın genel enerji dengesi skoru (0-1).
        
        TTS modellerin tipik çıktı RMS aralığı: 0.015 – 0.08
        Eski normalizasyon (0.005-0.145) bu aralıkta %7–%51 veriyordu → yanlış.
        Yeni aralık: 0.010-0.090 → normal YouTuber sesi %40–%80+ alıyor.
        """
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None)
            rms = librosa.feature.rms(y=y)[0]
            mean_rms = np.mean(rms)
            if mean_rms < 0.003:
                return 0.0  # Silent/empty file
            elif mean_rms > 0.150:
                return 0.20  # Clipped/distorted
            else:
                # 0.020 - 0.080 is the healthy, natural speaker volume target range.
                # If mean_rms is within this range, it gets a perfect 1.0 score.
                if 0.020 <= mean_rms <= 0.080:
                    return 1.0
                elif mean_rms < 0.020:
                    # Scale from 0.003 (0.1) to 0.020 (1.0)
                    return float(0.1 + (mean_rms - 0.003) / 0.017 * 0.9)
                else:
                    # Scale from 0.080 (1.0) to 0.150 (0.2)
                    return float(1.0 - (mean_rms - 0.080) / 0.070 * 0.8)
        except Exception:
            return 0.5

    def _score_energy_segment(self, audio_path: str, start: float, end: float) -> float:
        """Belirli bir zaman segmentinin enerji skoru."""
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None, offset=start, duration=end - start)
            if len(y) == 0:
                return 0.5
            rms = np.sqrt(np.mean(y ** 2))
            return float(np.clip(rms / 0.1, 0, 1))
        except Exception:
            return 0.5

    def _score_pitch_variety(self, audio_path: str) -> float:
        """
        Pitch çeşitliliği — doğal konuşmada pitch ZATEN değişir.
        
        ESKİ HATA: Pitch 'stability' (düşük std) ödüllendiriliyordu.
        Bu, monoton/robotik take'lere avantaj sağlıyordu!
        
        YENİ: Makul seviyede pitch variety ödüllendiriliyor.
        - Çok düşük std (< 10 Hz) = monoton, robotik → düşük skor
        - Orta std (15-40 Hz) = doğal konuşma → yüksek skor  
        - Çok yüksek std (> 60 Hz) = tutarsız → düşük skor
        """
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None)
            f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr)
            voiced_f0 = f0[voiced > 0] if voiced is not None else np.array([])
            if len(voiced_f0) < 10:
                return 0.5
            
            std_hz = np.std(voiced_f0)
            mean_hz = np.mean(voiced_f0)
            
            # İdeal aralık: 15-40 Hz std (doğal konuşma)
            if std_hz < 10:
                # Çok monoton — cezalandır
                return float(np.clip(std_hz / 10.0, 0.2, 0.6))
            elif std_hz <= 40:
                # Doğal pitch variety — ödüllendir
                return float(np.clip(0.6 + (std_hz - 10) / 75.0, 0.6, 1.0))
            else:
                # Çok tutarsız — hafif cezalandır
                return float(np.clip(1.0 - (std_hz - 40) / 60.0, 0.4, 0.8))
        except Exception:
            return 0.5

    def _find_word(self, words: list[dict], target: str, hint_idx: int) -> dict | None:
        """
        Aligned word listesinde hedef kelimeyi bulur.
        hint_idx: beklenen pozisyon — aynı kelime birden fazla geçiyorsa
                  (ör: "I I don't") doğru olanı seçmek için kullanılır.
        
        ESKİ HATA: hint_idx hiç kullanılmıyordu → her zaman ilk eşleşme dönüyordu.
        "I I don't" gibi tekrarlarda hep ilk "I" seçiliyordu → yanlış timing.
        """
        import re
        target_clean = re.sub(r"[^\w']", "", target.lower())

        # Tüm tam eşleşmeleri bul
        exact_matches = []
        for i, w in enumerate(words):
            w_clean = re.sub(r"[^\w']", "", w["word"].lower())
            if w_clean == target_clean:
                exact_matches.append((i, w))

        if exact_matches:
            # hint_idx'e en yakın eşleşmeyi seç
            best = min(exact_matches, key=lambda x: abs(x[0] - hint_idx))
            return best[1]

        # Kısmi eşleşme (telaffuz farkı için)
        partial_matches = []
        for i, w in enumerate(words):
            w_clean = re.sub(r"[^\w']", "", w["word"].lower())
            if target_clean in w_clean or w_clean in target_clean:
                partial_matches.append((i, w))

        if partial_matches:
            best = min(partial_matches, key=lambda x: abs(x[0] - hint_idx))
            return best[1]

        return None

    def _smooth_selection(
        self,
        per_word_scores: list[dict],
        n_takes: int,
        min_segment: int,
    ) -> list[int]:
        """
        Her kelime için en iyi take'i seçer, ama çok sık geçiş yapmaz.
        Dynamic programming ile minimum geçişli, maksimum skorlu yol bulur.

        FIX: prev dizisinin -1 başlangıç değeri backtrack'te Python negatif
        index hatası yaratıyordu. Şimdi 0 ile başlatılıp doğru izleniyor.
        """
        n_words = len(per_word_scores)
        if n_words == 0:
            return []

        # Geçiş cezası
        SWITCH_PENALTY = 0.15

        NEG_INF = float("-inf")
        # dp[word][take] = toplam_skor
        dp = [[NEG_INF] * n_takes for _ in range(n_words)]
        # prev[word][take] = önceki take index (0 başlangıç — Python -1 index hatasını önler)
        prev = [[0] * n_takes for _ in range(n_words)]

        # İlk kelime — prev yok, sadece skor
        for t in range(n_takes):
            dp[0][t] = per_word_scores[0].get(t, 0.0)
            prev[0][t] = t  # İlk kelimede önceki yok, kendine işaret et

        # İleri geçiş
        for w in range(1, n_words):
            for t in range(n_takes):
                score = per_word_scores[w].get(t, 0.0)
                best_prev_score = NEG_INF
                best_prev_t = 0
                for pt in range(n_takes):
                    if dp[w-1][pt] == NEG_INF:
                        continue
                    penalty = 0.0 if pt == t else SWITCH_PENALTY
                    total = dp[w-1][pt] + score - penalty
                    if total > best_prev_score:
                        best_prev_score = total
                        best_prev_t = pt
                dp[w][t] = best_prev_score if best_prev_score > NEG_INF else score
                prev[w][t] = best_prev_t

        # Geri izleme — en iyi son take'i bul
        best_last_t = max(range(n_takes), key=lambda t: dp[n_words-1][t] if dp[n_words-1][t] > NEG_INF else NEG_INF)
        path = [best_last_t]
        for w in range(n_words - 1, 0, -1):
            path.append(prev[w][path[-1]])
        path.reverse()

        return path
