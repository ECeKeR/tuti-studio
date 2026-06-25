# Copyright (c) 2026 Efeberk Çeker
# SPDX-License-Identifier: AGPL-3.0-only

"""
AudioStitcher — v5.2 "Acoustic Bridge, Tail Guard & Timbre Match"
Düzeltmeler: 
  1. AcousticBridge entegrasyonu, Parametre atamaları ve Konuşma sonu koruma sistemi.
  2. Geliştirme #2: Ambient Silence (Oda Tonu) üretiminde filter ringing (titreşim) önlendi.
  3. Geliştirme #3: Viterbi algoritmasına Pitch (F0) uyumluluk cezası eklendi (Timbre shift önlendi).
  4. Mastering: LUFS standardı -18'den -16'ya çekildi, Limiter oranı yumuşatıldı (0.3 -> 0.5).
"""

import logging
import re
import numpy as np
from pathlib import Path

log = logging.getLogger(__name__)

# Kritik: AcousticBridge'i import ediyoruz
try:
    from acoustic_bridge import AcousticBridge
except ImportError:
    log.warning("acoustic_bridge.py bulunamadı! Lokal matching kullanılacak.")
    AcousticBridge = None

TARGET_LUFS = -16.0  # YouTube standartları için seslendirmede ideal seviye
FALLBACK_SR = 24000


class AudioStitcher:
    def __init__(
        self,
        crossfade_ms: int = 25,              # Geçiş süresi (ms) — daha kısa = daha doğal (25ms geçişleri yumuşatır)
        target_lufs: float = TARGET_LUFS,
        ambient_level_db: float = -72.0,     # Oda tonu seviyesi (dBFS)
        pause_jitter_pct: float = 0.18,      # Duraklama rastgeleliği (±%18)
        use_acoustic_bridge: bool = True,    # Eklendi: self'e atanacak
        ollama_bridge: bool = False,          # Eklendi: self'e atanacak
        use_hybrid_comping: bool = False     # Eklendi: hibrit öbek comping
    ):
        self.crossfade_ms = crossfade_ms
        self.target_lufs = target_lufs
        self.ambient_level_db = ambient_level_db
        self.pause_jitter_pct = pause_jitter_pct
        
        # Kritik Düzeltme: Parametre atamaları yapıldı
        self.use_acoustic_bridge = use_acoustic_bridge
        self.ollama_bridge = ollama_bridge
        self.use_hybrid_comping = use_hybrid_comping
        
        # Bridge Nesnesi Başlatıldı
        if AcousticBridge and self.use_acoustic_bridge:
            self.bridge = AcousticBridge()
            log.info("  ✓ AcousticBridge aktif ve Stitcher'a bağlandı.")
        else:
            self.bridge = None
            
        self._sr = FALLBACK_SR  # stitch() başında güncellenir

    # ================================================================== #
    #  ANA STİTCH
    # ================================================================== #

    def stitch(self, all_segments: list[list[dict]], output_path: str) -> str:
        import soundfile as sf
        import json

        flat_segments = []
        for chunk_segs in all_segments:
            flat_segments.extend(chunk_segs)

        if not flat_segments:
            raise ValueError("Birleştirilecek segment yok!")

        log.info(f"  {len(flat_segments)} segment birleştiriliyor (Studio Engine v5.2 - Phrase Comping)...")

        # Referans sample rate
        self._sr = self._detect_sample_rate(flat_segments[0]["audio_path"])
        log.info(f"  Referans SR: {self._sr} Hz")

        # Her segmenti yükle
        audio_parts = []
        pause_durations = []

        for i, seg in enumerate(flat_segments):
            audio = None
            try:
                # take_*.wav ve take_*_aligned.json dosyalarını bulalım
                audio_path = Path(seg["audio_path"])
                seg_dir = audio_path.parent
                
                take_paths = list(seg_dir.glob("take_*.wav"))
                # Sadece take_ ile başlayan ve rakam içeren dosyaları seç ve sayısal olarak sırala
                take_paths = [p for p in take_paths if re.search(r"\d+", p.stem)]
                take_paths.sort(key=lambda p: int(re.search(r"\d+", p.stem).group()))
                
                alignments = []
                valid_paths = []
                
                for tp in take_paths:
                    ap = tp.with_name(f"{tp.stem}_aligned.json")
                    if ap.exists():
                        with open(ap, "r") as f:
                            alignments.append(json.load(f))
                            valid_paths.append(tp)
                            
                if self.use_hybrid_comping and len(valid_paths) > 1 and len(alignments) == len(valid_paths):
                    # Birden fazla take ve alignment var, Phrase Comping Engine çalıştır!
                    audio = self._stitch_hybrid_segment(seg, valid_paths, alignments)
                    if audio is not None:
                        log.info(f"    Seg {i}: Hibrit derleme yapıldı (Phrase Comping).")
            except Exception as e:
                log.warning(f"    Seg {i} hibrit derlenirken hata oluştu: {e}. Klasik birleştirmeye geçiliyor.")
                audio = None

            # Fallback: Seçili take'i doğrudan yükle — işlem yapma.
            if audio is None:
                audio = self._load_full_audio(seg["audio_path"])
                audio = self._trim_tail_only(audio)

            audio_parts.append(audio)

            # Pause süresi — ±%18 jitter ekle (insan gibi ritim)
            raw_pause = seg.get("pause_after", 0.3)
            jitter = np.random.uniform(-self.pause_jitter_pct, self.pause_jitter_pct)
            pause = max(0.05, raw_pause * (1.0 + jitter))
            pause_durations.append(pause)

            log.debug(
                f"    Seg {i}: '{seg.get('text','?')[:40]}' | "
                f"T{seg.get('take_idx','?')} | "
                f"{len(audio)/self._sr:.3f}s | pause={pause:.3f}s"
            )

        # Birleştir: pause ekle → zero-crossing + crossfade
        result = self._stitch_with_pauses(audio_parts, pause_durations, flat_segments)

        # Post-process
        result = self._trim_silence(result)
        result = self._master(result)

        # Apply global output speed if specified and different from 1.0
        if hasattr(self, "output_speed") and abs(self.output_speed - 1.0) > 0.005:
            log.info(f"  Applying global output speed: {self.output_speed:.2f}x using librosa time stretch")
            try:
                import librosa
                result = librosa.effects.time_stretch(result, rate=self.output_speed)
            except Exception as e:
                log.warning(f"  Failed to apply global output speed time stretch: {e}")

        sf.write(output_path, result, self._sr)
        duration = len(result) / self._sr
        log.info(f"  ✓ Kaydedildi: {output_path} ({duration:.2f}s) @ {self._sr}Hz")
        return output_path

    # ================================================================== #
    #  PHRASE COMPING ENGINE (v5.2) YARDIMCILARI
    # ================================================================== #

    def _build_prosodic_chunks(self, words: list[dict]) -> list[list[dict]]:
        chunks = []
        current_chunk = []
        for i, w in enumerate(words):
            current_chunk.append(w)
            should_split = False
            if i == len(words) - 1:
                should_split = True
            else:
                w_next = words[i + 1]
                gap = w_next["start"] - w["end"]
                if w.get("ending_punct") or gap > 0.100:
                    should_split = True
            if should_split:
                chunks.append(current_chunk)
                current_chunk = []
        return chunks

    def _find_chunk_range(self, take_words: list[dict], chunk_words: list[dict], search_start_idx: int) -> tuple[int, int] | None:
        n_take = len(take_words)
        n_chunk = len(chunk_words)
        if n_take == 0 or n_chunk == 0:
            return None

        chunk_clean = [re.sub(r"[^\w']", "", w["word"].lower()) for w in chunk_words if w.get("word")]
        if not chunk_clean:
            return None

        take_clean = [re.sub(r"[^\w']", "", w["word"].lower()) for w in take_words]

        best_match_idx = -1
        best_score = -1

        for start_idx in range(n_take):
            matches = 0
            end_idx = min(start_idx + len(chunk_clean), n_take)
            for i in range(start_idx, end_idx):
                chunk_w_idx = i - start_idx
                if take_clean[i] == chunk_clean[chunk_w_idx]:
                    matches += 1

            score = matches - 0.05 * abs(start_idx - search_start_idx)
            if score > best_score and matches > 0:
                best_score = score
                best_match_idx = start_idx

        if best_match_idx != -1:
            end_idx = min(best_match_idx + len(chunk_clean) - 1, n_take - 1)
            match_ratio = best_score / len(chunk_clean)
            if match_ratio >= 0.5:
                return best_match_idx, end_idx

        return None

    def _get_f0_mean(self, audio_path: str, start: float, end: float) -> float:
        """Geliştirme #3: Bir segmentin ortalama Pitch (F0) değerini güvenli ve hızlı bir şekilde hesaplar."""
        try:
            import librosa
            duration = max(0.05, end - start)
            y, sr = librosa.load(audio_path, sr=self._sr, offset=start, duration=duration)
            if len(y) < int(sr * 0.05):
                return 0.0
            f0 = librosa.yin(y, fmin=60, fmax=400, sr=sr)
            f0 = f0[f0 > 0]
            return float(np.mean(f0)) if len(f0) > 0 else 0.0
        except Exception:
            return 0.0

    def _score_chunk_segment(self, audio_path: str, start: float, end: float, words_in_range: list[dict]) -> float:
        if end <= start:
            return 0.0

        confidences = [w.get("confidence", 0.5) for w in words_in_range]
        avg_confidence = np.mean(confidences) if confidences else 0.5

        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=self._sr, offset=start, duration=end - start)
            if len(y) == 0:
                energy_score = 0.5
            else:
                rms = np.sqrt(np.mean(y ** 2))
                energy_score = float(np.clip(rms / 0.08, 0.0, 1.0))
        except Exception:
            energy_score = 0.5

        return float(avg_confidence * 0.6 + energy_score * 0.4)

    def _stitch_hybrid_segment(self, seg: dict, take_paths: list[Path], alignments: list[dict]) -> np.ndarray | None:
        intonation_trend = seg.get("intonation_trend", "stable")
        vowel_stretching = seg.get("vowel_stretching", "normal")

        ref_idx = 0
        target_name = f"take_{seg.get('take_idx', seg.get('selected_take', 0))}.wav"
        for idx, tp in enumerate(take_paths):
            if tp.name == target_name:
                ref_idx = idx
                break
            
        ref_alignment = alignments[ref_idx]
        ref_words = ref_alignment.get("words", [])
        if not ref_words:
            return None

        chunks = self._build_prosodic_chunks(ref_words)
        if not chunks:
            return None

        log.info(f"    Segment {Path(seg['audio_path']).parent.name} prosodik öbek sayısı: {len(chunks)}")

        n_takes = len(take_paths)
        n_chunks = len(chunks)
        
        chunk_take_info = [[None] * n_takes for _ in range(n_chunks)]
        search_hints = [0] * n_takes
        
        for c_idx, chunk in enumerate(chunks):
            for t_idx in range(n_takes):
                take_align = alignments[t_idx]
                take_words = take_align.get("words", [])
                
                match_range = self._find_chunk_range(take_words, chunk, search_hints[t_idx])
                if match_range:
                    start_w_idx, end_w_idx = match_range
                    search_hints[t_idx] = end_w_idx + 1
                    
                    start_time = take_words[start_w_idx]["start"]
                    end_time = take_words[end_w_idx]["end"]
                    words_in_range = take_words[start_w_idx:end_w_idx + 1]
                    
                    score = self._score_chunk_segment(str(take_paths[t_idx]), start_time, end_time, words_in_range)
                    
                    # Geliştirme #3: F0 (Pitch) hesapla ve kaydet
                    f0_mean = self._get_f0_mean(str(take_paths[t_idx]), start_time, end_time)
                    
                    chunk_take_info[c_idx][t_idx] = {
                        "start": start_time,
                        "end": end_time,
                        "score": score,
                        "f0_mean": f0_mean,
                        "words": words_in_range
                    }
                else:
                    ref_duration = ref_alignment.get("duration", 1.0)
                    take_duration = take_align.get("duration", 1.0)
                    ratio = take_duration / ref_duration if ref_duration > 0 else 1.0
                    
                    start_time = chunk[0]["start"] * ratio
                    end_time = chunk[-1]["end"] * ratio
                    
                    chunk_take_info[c_idx][t_idx] = {
                        "start": start_time,
                        "end": end_time,
                        "score": 0.05,
                        "f0_mean": 0.0,  # Bilinmiyorsa 0 ata
                        "words": []
                    }

        # Viterbi ile optimal take geçiş yolunu bul (SWITCH_PENALTY + Pitch Penalty)
        SWITCH_PENALTY = 0.40
        NEG_INF = float("-inf")
        SELECTED_BONUS = 0.55

        dp = [[NEG_INF] * n_takes for _ in range(n_chunks)]
        prev = [[0] * n_takes for _ in range(n_chunks)]
        
        for t in range(n_takes):
            bonus = SELECTED_BONUS if t == ref_idx else 0.0
            dp[0][t] = chunk_take_info[0][t]["score"] + bonus
            prev[0][t] = t
            
        for c in range(1, n_chunks):
            for t in range(n_takes):
                bonus = SELECTED_BONUS if t == ref_idx else 0.0
                score = chunk_take_info[c][t]["score"] + bonus
                best_prev_score = NEG_INF
                best_prev_t = 0
                for pt in range(n_takes):
                    if dp[c-1][pt] == NEG_INF:
                        continue
                    
                    penalty = 0.0
                    if pt != t:
                        penalty = SWITCH_PENALTY
                        # Geliştirme #3: Take değişiminde Pitch (F0) farkı ceza olarak eklenir
                        prev_f0 = chunk_take_info[c-1][pt]["f0_mean"]
                        curr_f0 = chunk_take_info[c][t]["f0_mean"]
                        if prev_f0 > 0 and curr_f0 > 0:
                            pitch_diff_semitones = abs(12 * np.log2(curr_f0 / prev_f0))
                            # Her yarım ton fark için 0.1 ek ceza
                            penalty += pitch_diff_semitones * 0.1

                    total = dp[c-1][pt] + score - penalty
                    if total > best_prev_score:
                        best_prev_score = total
                        best_prev_t = pt
                dp[c][t] = best_prev_score if best_prev_score > NEG_INF else score
                prev[c][t] = best_prev_t

        best_last_t = max(range(n_takes), key=lambda t: dp[n_chunks-1][t])
        best_path = [best_last_t]
        for c in range(n_chunks - 1, 0, -1):
            best_path.append(prev[c][best_path[-1]])
        best_path.reverse()

        log.info(f"    Optimal prosodik take seçim yolu: {best_path}")

        final_path = []
        for c in range(n_chunks):
            t_choice = best_path[c]
            if "chunks" in seg and c < len(seg["chunks"]):
                user_override = seg["chunks"][c].get("override_take")
                if user_override is not None and 0 <= user_override < n_takes:
                    t_choice = user_override
            final_path.append(t_choice)

        if final_path != best_path:
            log.info(f"    Kullanıcı override sonrası nihai seçim yolu: {final_path}")

        contiguous_blocks = []
        current_block = {
            "take_idx": final_path[0],
            "start_chunk_idx": 0,
            "end_chunk_idx": 0
        }
        
        for c in range(1, n_chunks):
            t_idx = final_path[c]
            if t_idx == current_block["take_idx"]:
                current_block["end_chunk_idx"] = c
            else:
                contiguous_blocks.append(current_block)
                current_block = {
                    "take_idx": t_idx,
                    "start_chunk_idx": c,
                    "end_chunk_idx": c
                }
        contiguous_blocks.append(current_block)

        if len(set(final_path)) == 1:
            single_t = final_path[0]
            try:
                import librosa
                y, _ = librosa.load(str(take_paths[single_t]), sr=self._sr)
                y = self._trim_tail_only(y)
                log.info(
                    f"    Single-take fast path: take_{single_t} doğrudan yüklendi "
                    f"({len(y)/self._sr:.2f}s) — stitching atlandı."
                )
                return y
            except Exception as e:
                log.warning(f"    Single-take fast path hatası: {e} — blok birleştirmeye geçiliyor.")

        block_audios = []
        cf_samples = int(self._sr * self.crossfade_ms / 1000)
        
        for b_idx, block in enumerate(contiguous_blocks):
            t_idx = block["take_idx"]
            audio_path = str(take_paths[t_idx])
            
            start_chunk = chunk_take_info[block["start_chunk_idx"]][t_idx]
            end_chunk = chunk_take_info[block["end_chunk_idx"]][t_idx]
            
            start_time = start_chunk["start"]
            end_time = end_chunk["end"]
            
            try:
                import librosa
                import soundfile as sf
                
                if block["start_chunk_idx"] == 0:
                    pad_start = 0.0
                else:
                    pad_start = max(0.0, start_time - 0.025)
                
                if block["end_chunk_idx"] == n_chunks - 1:
                    pad_end = sf.info(audio_path).duration
                else:
                    pad_end = min(end_time + 0.060, sf.info(audio_path).duration)
                
                duration = max(0.01, pad_end - pad_start)
                
                y, _ = librosa.load(audio_path, sr=self._sr, offset=pad_start, duration=duration)
                peak = np.max(np.abs(y))
                if peak > 0.98:
                    y = y * (0.95 / peak)
                block_audios.append({
                    "audio": y,
                    "start_ref": chunks[block["start_chunk_idx"]][0]["start"],
                    "end_ref": chunks[block["end_chunk_idx"]][-1]["end"]
                })
            except Exception as e:
                log.warning(f"Blok yükleme hatası: {e}")
                return None

        if not block_audios:
            return None
            
        result = block_audios[0]["audio"].copy()
        
        for i in range(1, len(block_audios)):
            prev_block = block_audios[i - 1]
            curr_block = block_audios[i]

            prev_audio = result
            curr_audio = curr_block["audio"]

            cf_ms = self.crossfade_ms
            if vowel_stretching == "high":
                cf_ms = cf_ms * 1.5
            elif vowel_stretching == "low":
                cf_ms = cf_ms * 0.7

            cf_samples = int(self._sr * cf_ms / 1000)

            zc_a = self._find_zero_crossing(prev_audio, len(prev_audio) - 1, window_ms=8.0)
            zc_b = self._find_zero_crossing(curr_audio, 0, window_ms=8.0)

            a_cut = prev_audio[:zc_a] if zc_a > 0 else prev_audio
            b_cut = curr_audio[zc_b:] if zc_b < len(curr_audio) else curr_audio

            result = self._crossfade_two(a_cut, b_cut, cf_samples)

        return result

    def build_segment_chunks(self, seg: dict, take_paths: list[Path], alignments: list[dict]) -> list[dict]:
        import re

        ref_idx = 0
        target_name = f"take_{seg.get('take_idx', seg.get('selected_take', 0))}.wav"
        for idx, tp in enumerate(take_paths):
            if tp.name == target_name:
                ref_idx = idx
                break
        
        if ref_idx >= len(alignments):
            ref_idx = 0

        ref_alignment = alignments[ref_idx]
        ref_words = ref_alignment.get("words", [])
        if not ref_words:
            return []

        chunks = self._build_prosodic_chunks(ref_words)
        if not chunks:
            return []

        n_takes = len(take_paths)
        n_chunks = len(chunks)
        
        chunk_take_info = [[None] * n_takes for _ in range(n_chunks)]
        search_hints = [0] * n_takes
        
        for c_idx, chunk in enumerate(chunks):
            for t_idx in range(n_takes):
                take_align = alignments[t_idx]
                take_words = take_align.get("words", [])
                
                match_range = self._find_chunk_range(take_words, chunk, search_hints[t_idx])
                if match_range:
                    start_w_idx, end_w_idx = match_range
                    search_hints[t_idx] = end_w_idx + 1
                    
                    start_time = take_words[start_w_idx]["start"]
                    end_time = take_words[end_w_idx]["end"]
                    words_in_range = take_words[start_w_idx:end_w_idx + 1]
                    
                    score = self._score_chunk_segment(str(take_paths[t_idx]), start_time, end_time, words_in_range)
                    f0_mean = self._get_f0_mean(str(take_paths[t_idx]), start_time, end_time)
                    
                    chunk_take_info[c_idx][t_idx] = {
                        "score": score,
                        "f0_mean": f0_mean
                    }
                else:
                    chunk_take_info[c_idx][t_idx] = {
                        "score": 0.05,
                        "f0_mean": 0.0
                    }

        SWITCH_PENALTY = 0.40
        NEG_INF = float("-inf")
        
        dp = [[NEG_INF] * n_takes for _ in range(n_chunks)]
        prev = [[0] * n_takes for _ in range(n_chunks)]
        
        for t in range(n_takes):
            dp[0][t] = chunk_take_info[0][t]["score"]
            prev[0][t] = t
            
        for c in range(1, n_chunks):
            for t in range(n_takes):
                score = chunk_take_info[c][t]["score"]
                best_prev_score = NEG_INF
                best_prev_t = 0
                for pt in range(n_takes):
                    if dp[c-1][pt] == NEG_INF:
                        continue
                    
                    penalty = 0.0
                    if pt != t:
                        penalty = SWITCH_PENALTY
                        prev_f0 = chunk_take_info[c-1][pt]["f0_mean"]
                        curr_f0 = chunk_take_info[c][t]["f0_mean"]
                        if prev_f0 > 0 and curr_f0 > 0:
                            pitch_diff_semitones = abs(12 * np.log2(curr_f0 / prev_f0))
                            penalty += pitch_diff_semitones * 0.1

                    total = dp[c-1][pt] + score - penalty
                    if total > best_prev_score:
                        best_prev_score = total
                        best_prev_t = pt
                dp[c][t] = best_prev_score if best_prev_score > NEG_INF else score
                prev[c][t] = best_prev_t

        best_last_t = max(range(n_takes), key=lambda t: dp[n_chunks-1][t])
        best_path = [best_last_t]
        for c in range(n_chunks - 1, 0, -1):
            best_path.append(prev[c][best_path[-1]])
        best_path.reverse()

        result_chunks = []
        for c_idx, chunk in enumerate(chunks):
            chunk_text = " ".join([w["word"] for w in chunk])
            auto_take = int(best_path[c_idx])
            
            old_override = None
            if "chunks" in seg and c_idx < len(seg["chunks"]):
                old_override = seg["chunks"][c_idx].get("override_take")
                
            override_take = old_override if old_override is not None else None
            active_take = override_take if (override_take is not None and override_take >= 0) else auto_take

            result_chunks.append({
                "idx": c_idx,
                "text": chunk_text,
                "auto_take": auto_take,
                "override_take": override_take,
                "active_take": active_take
            })
            
        return result_chunks

    # ================================================================== #
    #  SAMPLE RATE
    # ================================================================== #

    def _detect_sample_rate(self, audio_path: str) -> int:
        try:
            import soundfile as sf
            return sf.info(audio_path).samplerate
        except Exception:
            return FALLBACK_SR

    def _load_full_audio(self, audio_path: str) -> np.ndarray:
        import librosa
        y, _ = librosa.load(audio_path, sr=self._sr)
        return y

    # ================================================================== #
    #  GENTLE TRIM
    # ================================================================== #

    def _gentle_trim(self, audio: np.ndarray) -> np.ndarray:
        import librosa
        try:
            trimmed, _ = librosa.effects.trim(audio, top_db=30)
            if len(trimmed) < self._sr * 0.05:
                return audio
            return trimmed
        except Exception:
            return audio

    def _trim_tail_only(self, audio: np.ndarray, top_db: float = 30) -> np.ndarray:
        import librosa
        try:
            rev = audio[::-1].copy()
            trimmed_rev, _ = librosa.effects.trim(rev, top_db=top_db)
            result = trimmed_rev[::-1]
            if len(result) < self._sr * 0.05:
                return audio
            return result
        except Exception:
            return audio

    # ================================================================== #
    #  ZERO-CROSSING ALIGNMENT
    # ================================================================== #

    def _find_zero_crossing(
        self, audio: np.ndarray, target_sample: int, window_ms: float = 10.0
    ) -> int:
        window = int(self._sr * window_ms / 1000)
        lo = max(0, target_sample - window)
        hi = min(len(audio) - 2, target_sample + window)

        if lo >= hi:
            return max(0, min(target_sample, len(audio) - 1))

        best = target_sample
        best_dist = window + 1

        for i in range(lo, hi):
            if audio[i] * audio[i + 1] <= 0:
                dist = abs(i - target_sample)
                if dist < best_dist:
                    best_dist = dist
                    best = i + 1

        return max(0, min(best, len(audio)))

    # ================================================================== #
    #  GELİŞTİRİLMİŞ AMBIENT SİLENCE (Filter Ringing Önleme - #2)
    # ================================================================== #

    def _generate_ambient_silence(self, n_samples: int) -> np.ndarray:
        """
        Geliştirme #2: Filtre ringing (titreşim) sorununu çözmek için
        gürültüyü en az 0.5 saniye (veya 3 katı) uzunluğunda üretip, filtreledikten
        sonra tam ihtiyacımız olan kısmı ortasından kesiyoruz.
        Böylece filtrekenen kenar artefaktları (transient response) oluşmaz.
        """
        if n_samples <= 0:
            return np.array([], dtype=np.float32)

        level = 10 ** (self.ambient_level_db / 20)
        
        # Güvenli uzunluk hesapla
        safe_len = max(int(self._sr * 0.5), n_samples * 3)
        noise = np.random.normal(0, level, safe_len).astype(np.float32)

        try:
            from scipy.signal import butter, filtfilt
            nyq = self._sr / 2
            lo = max(80.0 / nyq, 1e-4)
            hi = min(4000.0 / nyq, 1 - 1e-4)
            b, a = butter(2, [lo, hi], btype="band")
            
            # Güvenli uzunlukta filtrele (Filtre kenar artefaktları burada birikir)
            filtered = filtfilt(b, a, noise).astype(np.float32)
            
            # Ortadaki en saf kısmı al
            start_idx = (len(filtered) - n_samples) // 2
            return filtered[start_idx : start_idx + n_samples]
            
        except Exception:
            # Filtreleme başarısız olursa ham noise'u kırpıp döndür
            return noise[:n_samples]

    # ================================================================== #
    #  ANA BİRLEŞTİRME
    # ================================================================== #

    def _stitch_with_pauses(
        self,
        parts: list[np.ndarray],
        pause_durations: list[float],
        segments: list[dict] = None
    ) -> np.ndarray:
        if not parts:
            return np.array([], dtype=np.float32)
        if len(parts) == 1:
            return parts[0].astype(np.float32)

        result = parts[0].copy()

        for i in range(1, len(parts)):
            next_part = parts[i]
            pause_secs = pause_durations[i - 1]

            cf_ms = self.crossfade_ms
            if segments and i < len(segments):
                seg = segments[i]
                vowel_stretching = seg.get("vowel_stretching", "normal")
                if vowel_stretching == "high":
                    cf_ms = cf_ms * 1.5
                elif vowel_stretching == "low":
                    cf_ms = cf_ms * 0.7

            result = self._stitch_two_blocks(result, next_part, pause_secs, pad_ms=30.0, crossfade_ms=cf_ms)

        return result.astype(np.float32)

    def _crossfade_two(
        self, a: np.ndarray, b: np.ndarray, cf_samples: int
    ) -> np.ndarray:
        cf = min(cf_samples, len(a), len(b))
        if cf == 0:
            return np.concatenate([a, b])

        fade_out = np.cos(np.linspace(0, np.pi / 2, cf)) ** 2
        fade_in = np.sin(np.linspace(0, np.pi / 2, cf)) ** 2

        cross = a[-cf:] * fade_out + b[:cf] * fade_in
        return np.concatenate([a[:-cf], cross, b[cf:]]).astype(np.float32)

    def _stitch_two_blocks(
        self, a: np.ndarray, b: np.ndarray, gap_secs: float, pad_ms: float = 60.0, crossfade_ms: float = 12.0
    ) -> np.ndarray:
        import numpy as np

        pad_samples = int(self._sr * pad_ms / 1000)
        cf_samples = int(self._sr * crossfade_ms / 1000)
        gap_samples = int(self._sr * gap_secs)

        if gap_samples < 2 * pad_samples:
            overlap = 2 * pad_samples - gap_samples
            overlap = min(overlap, len(a)//2, len(b)//2)
            
            zero_a = self._find_zero_crossing(a, len(a) - overlap)
            zero_b = self._find_zero_crossing(b, overlap)

            a_trimmed = a[:zero_a]
            b_trimmed = b[zero_b:]

            cf = min(cf_samples, len(a_trimmed), len(b_trimmed))
            if cf <= 0: return np.concatenate([a_trimmed, b_trimmed])

            fade_out = np.cos(np.linspace(0, np.pi / 2, cf)) ** 2
            fade_in = np.sin(np.linspace(0, np.pi / 2, cf)) ** 2

            cross = a_trimmed[-cf:] * fade_out + b_trimmed[:cf] * fade_in
            return np.concatenate([a_trimmed[:-cf], cross, b_trimmed[cf:]]).astype(np.float32)

        else:
            ambient_len = gap_samples 
            ambient = self._generate_ambient_silence(ambient_len)
            
            part1 = self._crossfade_two(a, ambient, cf_samples)
            return self._crossfade_two(part1, b, cf_samples)

    def _apply_bridge_or_local(self, prev, curr, seg_info, is_anchor_near):
        if self.bridge:
            return self.bridge.apply(
                prev, curr,
                intonation_trend=seg_info.get("intonation_trend", "stable"),
                vowel_stretching=seg_info.get("vowel_stretching", "normal"),
                is_anchor_near=is_anchor_near
            )
        else:
            return self._apply_local_ai_acoustic_matching(
                prev, curr,
                intonation_trend=seg_info.get("intonation_trend", "stable"),
                vowel_stretching=seg_info.get("vowel_stretching", "normal"),
                stress_anchor=seg_info.get("stress_anchor", ""),
                is_anchor_near=is_anchor_near
            )

    # ================================================================== #
    #  NORMALİZASYON
    # ================================================================== #

    def _normalize_segment(
        self, audio: np.ndarray, target_rms: float = 0.08
    ) -> np.ndarray:
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-6:
            return audio
        scale = min(target_rms / rms, 3.0)
        return audio * scale

    # ================================================================== #
    #  SESSIZLIK TEMİZLEME
    # ================================================================== #

    def _trim_silence(self, audio: np.ndarray, top_db: float = 40) -> np.ndarray:
        import librosa
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed

    # ================================================================== #
    #  MASTERING (Yumuşatılmış Limiter ve Yüksek LUFS)
    # ================================================================== #

    def _master(self, audio: np.ndarray) -> np.ndarray:
        """
        Geliştirme: Daha doğal sıkıştırma ve daha yüksek ses seviyesi.
        """
        audio = audio - np.mean(audio)

        try:
            import pyloudnorm as pyln
            meter = pyln.Meter(self._sr)
            loudness = meter.integrated_loudness(audio.astype(np.float64))
            if np.isfinite(loudness) and loudness > -70:
                audio = pyln.normalize.loudness(
                    audio.astype(np.float64), loudness, self.target_lufs
                ).astype(np.float32)
                log.debug(f"  LUFS: {loudness:.1f} → {self.target_lufs:.1f}")
        except ImportError:
            log.warning("pyloudnorm yok → pip install pyloudnorm")
        except Exception as e:
            log.warning(f"LUFS normalize başarısız: {e}")

        threshold = 0.95
        mask = np.abs(audio) > threshold
        if np.any(mask):
            over = np.abs(audio[mask]) - threshold
            # 2:1 sıkıştırma (0.5) - 3:1 yerine çok daha doğal
            compressed = threshold + over * 0.5  
            audio[mask] = np.sign(audio[mask]) * compressed

        peak = np.max(np.abs(audio))
        if peak > 0.01:
            audio = audio * (0.9 / peak)

        return audio.astype(np.float32)

    def _apply_local_ai_acoustic_matching(
        self,
        prev_block: np.ndarray,
        curr_block: np.ndarray,
        intonation_trend: str = "stable",
        vowel_stretching: str = "normal",
        stress_anchor: str = "",
        is_anchor_near: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        import librosa

        analysis_len = int(self._sr * 0.030)
        if len(prev_block) < analysis_len or len(curr_block) < analysis_len:
            return prev_block, curr_block

        prev_edge = prev_block[-analysis_len:]
        curr_edge = curr_block[:analysis_len]

        rms_prev = np.sqrt(np.mean(prev_edge ** 2))
        rms_curr = np.sqrt(np.mean(curr_edge ** 2))

        if rms_prev > 1e-4 and rms_curr > 1e-4:
            gain_ratio = rms_prev / rms_curr
            
            if is_anchor_near and gain_ratio < 1.0:
                gain_ratio = (gain_ratio + 1.0) / 2.0

            gain_curve = np.linspace(1.0, gain_ratio, analysis_len)
            curr_block[:analysis_len] = curr_block[:analysis_len] * gain_curve

        return prev_block, curr_block