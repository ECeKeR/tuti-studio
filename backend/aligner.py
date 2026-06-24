# Copyright (c) 2026 Efeberk Çeker
# SPDX-License-Identifier: AGPL-3.0-only

"""
Word-Level Aligner + Breathing Analyzer
Whisper ile ses dosyasındaki her kelimenin tam zamanını (start/end) bulur.
BreathingAnalyzer: kelimeler arası gap'leri ölçerek doğal nefes/duraklama skoru üretir.
"""

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)


def to_native_types(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_native_types(x) for x in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


class WordAligner:
    def __init__(self, whisper_model: str = "base"):
        """
        Args:
            whisper_model: "tiny", "base", "small", "medium", "large"
                           TTS alignment için "base" yeterli ve hızlı.
        """
        self.whisper_model = whisper_model
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import whisper_timestamped as whisper
        log.info(f"Whisper '{self.whisper_model}' yükleniyor...")
        self._model = whisper.load_model(self.whisper_model)
        log.info("Whisper yüklendi.")

    def align(self, audio_path: str, expected_text: str | None = None) -> dict:
        """
        Ses dosyasını Whisper ile hizalar.
        
        Returns:
            {
                "text": "...",
                "words": [
                    {"word": "hello", "start": 0.1, "end": 0.5, "confidence": 0.99, "ending_punct": ""},
                    ...
                ],
                "duration": 3.2,
                "breathing": { ... }
            }
        """
        import whisper_timestamped as whisper
        import librosa

        self._load()

        audio, sr = librosa.load(audio_path, sr=16000)
        duration = len(audio) / sr

        result = whisper.transcribe(
            self._model,
            audio,
            language="en",
            detect_disfluencies=False,
        )

        words = []
        for segment in result.get("segments", []):
            for w in segment.get("words", []):
                raw_word = w["text"].strip()
                
                # Kelimenin sonundaki orijinal noktalamayı koru!
                # Çünkü BreathingAnalyzer bu noktalamaya bakarak gap doğrulaması yapacak.
                ending_punct = ""
                punct_match = re.search(r"([.,!?;:\—]|\.\.\.)$", raw_word)
                if punct_match:
                    ending_punct = punct_match.group(1)

                word_text = raw_word.strip(".,!?;:'\"-()[]{}…")
                word_text = word_text.strip()
                
                if word_text:
                    words.append({
                        "word": word_text,
                        "start": round(w["start"], 4),
                        "end": round(w["end"], 4),
                        "confidence": round(w.get("confidence", 1.0), 4),
                        "ending_punct": ending_punct
                    })

        # Breathing analizi — kelimeler arası gap'leri ölç
        breathing = BreathingAnalyzer.analyze_from_words(words, duration)

        alignment = {
            "text": result.get("text", "").strip(),
            "words": words,
            "duration": round(duration, 4),
            "audio_path": str(audio_path),
            "breathing": breathing,
        }

        log.debug(f"  Hizalama: {len(words)} kelime, {duration:.2f}s, "
                  f"breathing_score={breathing['score']:.3f}")
        return to_native_types(alignment)

    def align_batch(self, audio_paths: list[str], texts: list[str] | None = None) -> list[dict]:
        """Birden fazla dosyayı hizalar."""
        results = []
        texts = texts or [None] * len(audio_paths)
        for path, text in zip(audio_paths, texts):
            results.append(self.align(path, text))
        return results


class BreathingAnalyzer:
    """
    Whisper timing verisinden doğal nefes/duraklama kalitesini analiz eder.

    Neden önemli:
      TTS modeli bazen "hiç nefes almayacak gibi" okur — kelimeler arası gap < 20ms.
      Bu robotik ve yorucu bir dinleme deneyimi yaratır.

    Nasıl çalışır:
      1. Whisper'dan gelen word timestamps'ten kelimeler arası gap hesapla
      2. Her gap'i "doğal mı, çok kısa mı, çok uzun mu?" diye sınıflandır
      3. Noktalama ve clause boundary'lerde beklenen gap vs gerçek gap karşılaştır
      4. Genel bir breathing_score üret (0.0 = robotik, 1.0 = doğal)
    """

    # Doğal konuşmada kelimeler arası beklenen gap aralıkları (saniye)
    GAP_MIN_NATURAL = -0.020    # < -20ms = çakışma/robotik (akıcı konuşmada 0ms ve hafif çakışmalar doğaldır)
    GAP_IDEAL_WORD  = 0.060    # ~60ms = kelimeler arası ideal
    GAP_CLAUSE      = 0.150    # ~150ms = virgül
    GAP_SENTENCE    = 0.300    # ~300ms = cümle sonu
    GAP_MAX_NATURAL = 0.600    # > 600ms = çok uzun duraklama

    # Noktalama → beklenen minimum gap
    PUNCT_GAPS = {
        ",":  0.120,
        ";":  0.180,
        ":":  0.180,
        ".":  0.250,
        "!":  0.250,
        "?":  0.250,
        "—":  0.200,
        "...": 0.300,
    }

    # PLOSIVE/SIBILANT HARFLER (Anatomik Boşluk Yaratanlar)
    PLOSIVES_SIBILANTS = set(['p', 't', 'k', 'c', 'ch', 's', 'sh', 'f', 'x', 'z'])

    @classmethod
    def analyze_from_words(cls, words: list[dict], duration: float) -> dict:
        """
        Whisper word listesinden breathing analizi yapar.

        Returns:
            {
                "score": 0.72,          # 0=robotik, 1=doğal
                "gaps": [...],          # her kelime çiftinin gap'i
                "robotic_pairs": [...], # gap < 30ms olan kelime çiftleri
                "missing_pauses": [...],# noktalama olması gereken yerde gap yok
                "avg_gap_ms": 45.2,
                "details": "..."
            }
        """
        if len(words) < 2:
            return {"score": 0.5, "gaps": [], "robotic_pairs": [],
                    "missing_pauses": [], "avg_gap_ms": 0, "details": "too few words"}

        gaps = []
        robotic_pairs = []
        missing_pauses = []

        for i in range(len(words) - 1):
            w_curr = words[i]
            w_next = words[i + 1]
            gap = round(w_next["start"] - w_curr["end"], 4)
            gap_ms = gap * 1000

            # Patlamalı/Tıslamalı Harf Toleransı (Plosive Mitigation)
            # Eğer kelimelerin kesişim harfleri sertse robotik sınırını 15ms'e esnetiyoruz
            current_min_natural = cls.GAP_MIN_NATURAL
            last_char_curr = w_curr["word"][-1].lower() if w_curr["word"] else ""
            first_char_next = w_next["word"][0].lower() if w_next["word"] else ""
            
            if last_char_curr in cls.PLOSIVES_SIBILANTS or first_char_next in cls.PLOSIVES_SIBILANTS:
                current_min_natural = -0.035 # sert harflerde 35ms çakışmaya kadar tolerans tanıyalım

            gaps.append({
                "between": f"{w_curr['word']} → {w_next['word']}",
                "gap_ms": round(gap_ms, 1),
                "natural": current_min_natural <= gap <= cls.GAP_MAX_NATURAL,
            })

            # Robotik kontrol
            if gap < current_min_natural:
                robotic_pairs.append({
                    "words": f"'{w_curr['word']}' → '{w_next['word']}'",
                    "gap_ms": round(gap_ms, 1),
                })

            # missing_pauses Algoritmasını Çalıştırma
            # Kelimenin sonunda orijinal bir noktalama işareti var mı bakıyoruz
            punct = w_curr.get("ending_punct", "")
            if punct in cls.PUNCT_GAPS:
                expected_min = cls.PUNCT_GAPS[punct]
                # Eğer noktalama olan yerde bırakılan gap beklenen minimumdan küçükse yakala!
                if gap < expected_min:
                    missing_pauses.append({
                        "words": f"'{w_curr['word']}{punct}' → '{w_next['word']}'",
                        "expected_ms": expected_min * 1000,
                        "actual_ms": round(gap_ms, 1)
                    })

        avg_gap = sum(g["gap_ms"] for g in gaps) / len(gaps) if gaps else 0

        # Skor Hesaplama fonksiyonuna missing_pauses cezasını da dahil ediyoruz
        score = cls._compute_score(gaps, robotic_pairs, missing_pauses)

        details_parts = []
        if robotic_pairs:
            details_parts.append(f"{len(robotic_pairs)} robotik geçiş")
        if missing_pauses:
            details_parts.append(f"{len(missing_pauses)} kaçırılmış nefes noktası")
        if avg_gap < 40:
            details_parts.append("nefessiz okuma")
        elif avg_gap > 200:
            details_parts.append("ağır tempo")
        else:
            details_parts.append("doğal tempo")

        return {
            "score": round(score, 4),
            "gaps": gaps,
            "robotic_pairs": robotic_pairs,
            "missing_pauses": missing_pauses,
            "avg_gap_ms": round(avg_gap, 1),
            "details": ", ".join(details_parts) if details_parts else "normal",
        }

    @classmethod
    def _compute_score(cls, gaps: list[dict], robotic_pairs: list[dict], missing_pauses: list[dict]) -> float:
        """
        Geliştirilmiş Adalet Terazisi:
          - Robotik geçişler (< -20ms çakışma / sert harflerde -35ms) -> %50'ye kadar ceza
          - Kaçırılan noktalama nefesleri (missing_pauses) -> %30'a kadar ceza
          - İdeal aralık koruması (-20ms ile 300ms arası) -> Ödüllendirme
        """
        if not gaps:
            return 0.5

        n = len(gaps)
        score = 1.0

        # 1. Robotik geçiş cezası
        robotic_ratio = len(robotic_pairs) / n
        score -= robotic_ratio * 0.5

        # 2. Kaçırılan noktalama/nefes cezası
        if missing_pauses:
            missing_ratio = len(missing_pauses) / n
            score -= missing_ratio * 0.3

        # 3. Doğal gap oranı ödülü (-20ms ile 300ms arası, akıcı/sürekli konuşma dâhil)
        natural_gaps = sum(1 for g in gaps if -20.0 <= g["gap_ms"] <= 300.0)
        natural_ratio = natural_gaps / n
        score = score * 0.3 + natural_ratio * 0.7

        return float(max(0.0, min(1.0, score)))

    @classmethod
    def score_take_breathing(cls, alignment: dict) -> float:
        """
        Scorer tarafından çağrılır — alignment dict'inden breathing score döndürür.
        Mevcut alignment'ta breathing yoksa hesaplar.
        """
        breathing = alignment.get("breathing")
        if breathing:
            return breathing.get("score", 0.5)

        # Fallback: hesapla
        words = alignment.get("words", [])
        duration = alignment.get("duration", 0)
        result = cls.analyze_from_words(words, duration)
        return result.get("score", 0.5)
