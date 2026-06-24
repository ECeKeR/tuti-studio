# Copyright (c) 2026 Efeberk Çeker
# SPDX-License-Identifier: AGPL-3.0-only

"""
Flask Backend for AI Voice Pipeline v2
Provides API endpoints to interact with Speech Map, generate takes, view scores, override, and stitch.
"""

import sys

class SafeStream:
    def __init__(self, original):
        self.original = original

    def write(self, data):
        try:
            if self.original:
                self.original.write(data)
        except Exception:
            pass

    def flush(self):
        try:
            if self.original:
                self.original.flush()
        except Exception:
            pass

    def __getattr__(self, attr):
        return getattr(self.original, attr)

if sys.stdout is not None:
    sys.stdout = SafeStream(sys.stdout)
if sys.stderr is not None:
    sys.stderr = SafeStream(sys.stderr)

import os
os.environ["TQDM_DISABLE"] = "1"
os.environ["TQDM_MONITOR"] = "OFF"
try:
    from tqdm import tqdm
    tqdm.monitor_interval = 0
except Exception:
    pass
import json
import threading
import queue
import logging
from pathlib import Path
import numpy as np
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS

from speech_map import SpeechMapGenerator
from generator import TTSGenerator
from aligner import WordAligner
from scorer import WordScorer
from stitcher import AudioStitcher
from timbre_store import TimbreStore
import hf_downloader
from production_logger import ProductionLogger

# Configure logging with a file handler in pipeline_work folder
WORK_DIR = Path("./pipeline_work")
WORK_DIR.mkdir(parents=True, exist_ok=True)

log_format = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(WORK_DIR / "production.log", mode="a", encoding="utf-8")
    ]
)
log = logging.getLogger(__name__)

def run_startup_diagnostics():
    import sys
    log.info("=== STARTING PYTHON ENVIRONMENT DIAGNOSTICS ===")
    log.info(f"Python Executable: {sys.executable}")
    log.info(f"Python Version: {sys.version}")
    log.info(f"System Path (sys.path): {sys.path}")
    
    deps = ["numpy", "soundfile", "librosa", "pytsmod", "pyrubberband", "scipy", "pyloudnorm"]
    for dep in deps:
        try:
            mod = __import__(dep)
            path = getattr(mod, "__file__", "Built-in/Unknown")
            version = getattr(mod, "__version__", "Unknown")
            log.info(f"  ✓ {dep}: version={version}, path={path}")
        except ImportError as e:
            log.error(f"  ✗ {dep} IMPORT FAILED: {e}")
        except Exception as e:
            log.error(f"  ✗ {dep} FAILED WITH ERROR: {e}")
    log.info("================================================")

run_startup_diagnostics()

app = Flask(__name__, static_folder=None)
CORS(app)

# ── MLX WORKER THREAD MODEL ─────────────────────────────────────────────────
# MLX/Metal C++ runtime thread-safe değil. Aynı anda iki üretim başlarsa
# "PyThreadState_Get: GIL released" veya Metal device crash olur.
#
# ESKİ MODEL (senkron): threaded=False + handler içinde MLX üretimi. Bu, üretim
# sürerken ses dosyası servis edilememesine (play sessiz kalıyordu) neden oldu.
#
# YENİ MODEL (worker): threaded=True + MLX üretimi dedicated tek worker thread'de.
#   - generate_segment handler job'u queue'ya koyup hemen 202 döner (long-poll yok).
#   - Worker, job'ları sırayla işler; MLX'e sadece bu thread dokunur.
#   - Flask thread'leri I/O için serbesttir: ses dosyaları üretimle eş zamanlı servis edilir.
#
# _generation_lock artık "meşguliyet reddetme" değil, _generation_status dict'ine
# erişimi seri hale getirmek için kullanılır. MLX tek-thread garantisi queue'dan gelir.
_generation_lock = threading.Lock()
_generation_status = {
    "active": False,       # şu an bir job işleniyor mu
    "segment_idx": None,   # işlenen segment idx
    "queued": set(),       # enqueue edilmiş idx'ler (409 race kontrolü + status için)
    "queue_order": [],     # ordered snapshot (polling/status için)
}
_generation_queue = queue.Queue()

# Global models cached in memory
MODELS = {
    "generator": None,
    "aligner": None,
    "scorer": WordScorer(),
    "stitcher": AudioStitcher(),
    "current_backend": None,
    "current_model_size": None,
    "current_speaker": None,
    "current_ref_audio": None,
    "current_ref_text": None,
    "current_voice_mode": None,
    "current_design_prompt": None,
    "current_clone_alpha": None,
    "current_clone_topk": None,
    "current_clone_speaker": None,
}

STATE_FILE = WORK_DIR / "state.json"

def get_generator(backend="mlx", model_size="1.7B", speaker="Ethan", ref_audio=None, ref_text=None, voice_mode="preset", design_prompt=None, clone_alpha=1.0, clone_topk=3, clone_speaker="ryan"):
    """Lazy load and cache TTS model."""
    if (MODELS["generator"] is None or 
        MODELS["current_backend"] != backend or
        MODELS["current_model_size"] != model_size or 
        MODELS["current_speaker"] != speaker or
        MODELS["current_ref_audio"] != ref_audio or
        MODELS["current_ref_text"] != ref_text or
        MODELS["current_voice_mode"] != voice_mode or
        MODELS["current_design_prompt"] != design_prompt or
        MODELS["current_clone_alpha"] != clone_alpha or
        MODELS["current_clone_topk"] != clone_topk or
        MODELS["current_clone_speaker"] != clone_speaker):
        
        # Eski generator'ü bellekten atıp MLX cache temizliği yapıyoruz (RAM sızıntısını önlemek için)
        if MODELS["generator"] is not None:
            log.info("Eski TTS Generator bellekten temizleniyor (GC + MLX Cache)...")
            MODELS["generator"] = None
            import gc
            gc.collect()
            try:
                import mlx.core as mx
                mx.clear_cache()
            except ImportError:
                pass

        log.info(f"TTS Model yüklü değil veya farklı: backend={backend}, size={model_size}, speaker={speaker}, voice_mode={voice_mode}, ref={ref_audio}, clone_alpha={clone_alpha}, clone_topk={clone_topk}, clone_speaker={clone_speaker}. Yükleniyor...")
        MODELS["generator"] = TTSGenerator(
            backend=backend, 
            model_size=model_size, 
            speaker=speaker, 
            ref_audio=ref_audio, 
            ref_text=ref_text,
            voice_mode=voice_mode,
            design_prompt=design_prompt,
            clone_alpha=clone_alpha,
            clone_topk=clone_topk,
            clone_speaker=clone_speaker,
        )
        MODELS["current_backend"] = backend
        MODELS["current_model_size"] = model_size
        MODELS["current_speaker"] = speaker
        MODELS["current_ref_audio"] = ref_audio
        MODELS["current_ref_text"] = ref_text
        MODELS["current_voice_mode"] = voice_mode
        MODELS["current_design_prompt"] = design_prompt
        MODELS["current_clone_alpha"] = clone_alpha
        MODELS["current_clone_topk"] = clone_topk
        MODELS["current_clone_speaker"] = clone_speaker
    return MODELS["generator"]


def get_aligner():
    """Lazy load and cache Whisper aligner."""
    if MODELS["aligner"] is None:
        log.info("Whisper Aligner yükleniyor...")
        MODELS["aligner"] = WordAligner(whisper_model="base")
    return MODELS["aligner"]

import sqlite3
import datetime

DB_PATH = Path("./pipeline_work/tuti.db")

def _get_db_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
    except Exception as e:
        log.warning(f"Failed to set PRAGMAs on DB connection: {e}")
    return conn

def get_active_project_id():
    if not DB_PATH.exists():
        return None
    try:
        conn = _get_db_connection()
        c = conn.cursor()
        c.execute("SELECT value FROM settings WHERE key = 'active_project_id'")
        row = c.fetchone()
        conn.close()
        return int(row[0]) if row else None
    except Exception:
        return None

def load_project_state():
    project_id = get_active_project_id()
    if not project_id:
        if not DB_PATH.exists():
            return None
        try:
            conn = _get_db_connection()
            c = conn.cursor()
            c.execute("SELECT state_json FROM projects ORDER BY updated_at DESC LIMIT 1")
            row = c.fetchone()
            conn.close()
            return json.loads(row[0]) if row else None
        except Exception:
            return None

    try:
        conn = _get_db_connection()
        c = conn.cursor()
        c.execute("SELECT state_json FROM projects WHERE id = ?", (project_id,))
        row = c.fetchone()
        conn.close()
        return json.loads(row[0]) if row else None
    except Exception:
        return None

def save_project_state(state):
    project_id = state.get("id")
    if not project_id:
        return
    try:
        conn = _get_db_connection()
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('active_project_id', ?)", (str(project_id),))
        now = datetime.datetime.now().isoformat()
        c.execute(
            "UPDATE projects SET state_json = ?, updated_at = ? WHERE id = ?",
            (json.dumps(state, ensure_ascii=False), now, project_id)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.error(f"Error saving project state to SQLite: {e}")


def reset_stuck_generating_segments():
    """Resets any segments with status 'generating' back to 'pending' to prevent them from being stuck on startup."""
    if not DB_PATH.exists():
        return
    log.info("Checking for stuck 'generating' segments on startup...")
    try:
        conn = _get_db_connection()
        c = conn.cursor()
        c.execute("SELECT id, name, state_json FROM projects")
        rows = c.fetchall()
        for row in rows:
            p_id, p_name, state_json_str = row
            try:
                state = json.loads(state_json_str)
                updated = False
                if "segments" in state:
                    for seg in state["segments"]:
                        if seg.get("status") == "generating":
                            log.info(f"Resetting stuck segment {seg.get('idx')} in project '{p_name}' (ID: {p_id}) to 'pending'")
                            seg["status"] = "pending"
                            seg["error_msg"] = "Generation interrupted (server restart/crash)"
                            updated = True
                if updated:
                    now = datetime.datetime.now().isoformat()
                    c.execute(
                        "UPDATE projects SET state_json = ?, updated_at = ? WHERE id = ?",
                        (json.dumps(state, ensure_ascii=False), now, p_id)
                    )
            except Exception as e:
                log.error(f"Error parsing state_json for project ID {p_id}: {e}")
        conn.commit()
        conn.close()
        log.info("Stuck segment check complete.")
    except Exception as e:
        log.error(f"Failed to reset stuck generating segments: {e}")


reset_stuck_generating_segments()


def _rule_based_clean_punctuation(text: str, seg: dict | None = None) -> str:
    """
    Segment parametrelerine bakarak virgülleri ve rezonans döngülerini temizler veya korur.
    """
    import re

    is_sft = False
    if seg and (seg.get("lora_strength") is not None or seg.get("is_sft") is True):
        is_sft = True

    stress = float(seg.get("stress", 0.6)) if seg else 0.6
    intonation = seg.get("intonation_trend", "stable") if seg else "stable"
    pause_after = float(seg.get("pause_after", 0.45)) if seg else 0.45

    # Prosodic markup tespiti: Ollama büyük harf vurgu eklediyse virgüllere dokunma
    # "Well, we COULD..." veya "the ENTIRE SCRIPT AT ONCE?" gibi — virgüller kasıtlı
    has_caps_emphasis = bool(re.search(r'\b[A-Z]{2,}\b', text))  # 2+ harfli büyük kelime
    has_ellipsis = "..." in text  # Dramatik duraklama işareti

    # Prosodic markup varsa hiç comma temizliği yapma — Ollama'nın koyduğu virgüller kasıtlı
    if has_caps_emphasis or has_ellipsis:
        # Sadece fazla boşlukları temizle, başka dokunma
        result = re.sub(r'\s+', ' ', text).strip()
        log.info(f"  Prosodic markup tespit edildi (CAPS/ellipsis) — virgüller korundu: '{result}'")
        return result

    # SFT modunda veya YouTuber modunda virgüller şarkı ritmine (sing-songy) yol açar.
    # Bu sebeple SFT/YouTuber için virgül temizliği her zaman çalışmalı.
    if not is_sft and (stress >= 0.85 or intonation in ("rising", "falling")):
        return text

    result = text

    # Kural 1: Sayı aralıklarındaki veya listelerdeki virgüller — "six, seven" -> "six seven", "youtubers, reddit posts" -> "youtubers reddit posts"
    # Tüm virgül + boşluk + kelime durumlarında virgülü kaldırıp sadece boşluk bırakalım.
    # Örn: "youtubers, reddit posts, or just" -> "youtubers reddit posts or just"
    result = re.sub(r'(\b\w+),\s+(\b\w+)', r'\1 \2', result)

    # Kural 2: Liste sonundaki and/or önündeki Oxford virgülü
    # "A, B, and C" -> "A B and C"
    result = re.sub(r',\s+(and|or|but)\s+', r' \1 ', result, flags=re.IGNORECASE)

    # Kural 3: Kısa segmentte tüm virgülleri temizle
    word_count = len(result.split())
    if is_sft or (word_count <= 12 and pause_after < 0.6):
        result = result.replace(',', '')

    # Kural 4: Resonance Loop (Tekrarlayan Kelime Öbeği) Kontrolü ve rule-based düzeltme
    # Eğer "write a viral script" iki kez geçiyorsa ikincisini "do the same" yapalım (Kullanıcı Vaka 2)
    if "write a viral script" in result.lower():
        parts = result.lower().split("write a viral script")
        if len(parts) > 2:  # En az 2 kez geçiyor
            # İkinci ve sonraki geçişleri değiştir
            result = parts[0] + "write a viral script" + " while teaching you how to do the same" + "".join(parts[2:])
            log.info("  [Resonance Loop Düzeltildi]: 'write a viral script' rezonans döngüsü giderildi.")

    # Genel tekrarlayan öbek kontrolü (Loglama ve uyarı)
    words = re.sub(r"[^\w'\s]", "", result.lower()).split()
    n = len(words)
    detected_repeats = []
    for length in [5, 4, 3]:
        for i in range(n - length + 1):
            phrase = " ".join(words[i:i+length])
            for j in range(i + length, n - length + 1):
                compare_phrase = " ".join(words[j:j+length])
                if phrase == compare_phrase and phrase not in detected_repeats:
                    if not any(phrase in r for r in detected_repeats):
                        detected_repeats.append(phrase)

    if detected_repeats:
        log.warning(f"  [Resonance Loop Uyarısı]: Segmentte tekrarlayan öbekler tespit edildi: {detected_repeats}. "
                    f"Bu durum Autoregressive (Qwen3) modelde kelimelerin uzamasına (Attention Lock) sebep olabilir.")


    # Fazla boşlukları temizle
    result = re.sub(r'\s+', ' ', result).strip()
    log.info(f"  Noktalama / Metin temizleme sonucu: '{text}' → '{result}'")
    return result


def _smart_clean_punctuation(text: str, seg: dict | None = None, model: str = "qwen3:8b") -> str:
    """
    Önce kural tabanlı temizlik yapar (her zaman çalışır).
    Ollama açıksa ek iyileştirme ister.
    Prosodic markup (CAPS/ellipsis) varsa hiç dokunmaz.
    """
    import requests

    # 1. Her zaman: kural tabanlı temizlik
    rule_cleaned = _rule_based_clean_punctuation(text, seg)

    # 2. Prosodic markup varsa secondary Ollama call'unu atla
    # Ollama'nın speech map'te koyduğu CAPS vurguları ve "..." kasıtlı — değiştirme
    import re
    has_caps = bool(re.search(r'\b[A-Z]{2,}\b', rule_cleaned))
    has_ellipsis = "..." in rule_cleaned
    if has_caps or has_ellipsis:
        log.info(f"  Prosodic markup (CAPS/ellipsis) tespit edildi — secondary Ollama clean atlandı.")
        return rule_cleaned

    # 3. Ollama açıksa: daha iyi bir versiyon iste
    try:
        ping = requests.get("http://localhost:11434/api/tags", timeout=5)
        if ping.status_code != 200:
            return rule_cleaned
    except Exception:
        return rule_cleaned  # Ollama kapalı → kural tabanlı sonuç kullan

    stress = float(seg.get("stress", 0.6)) if seg else 0.6
    intonation = seg.get("intonation_trend", "stable") if seg else "stable"

    if stress >= 0.75 or intonation in ("rising", "falling"):
        fluency_note = "This segment needs emphasis — keep punctuation that creates dramatic pauses."
    else:
        fluency_note = "This segment should be read fluently and continuously — remove unnecessary commas."

    prompt = f"""You are a TTS text optimizer. Optimize the text below for natural text-to-speech reading by a fine-tuned voice.

Context: {fluency_note}

RULES:
1. Optimize commas: Remove commas that create unnatural list pauses (e.g. Oxford commas, list separators).
2. RESONANCE LOOP AVOIDANCE: If the text contains any repeated phrases of 3 or more words (e.g., repeating the same phrase twice like "write a viral script ... write a viral script"), rephrase the second occurrence of that phrase naturally to avoid repetition (e.g. "do the same", "do so", or "create one").
3. Do NOT change other words, capitalization, spelling, periods, question marks, or exclamation marks.
4. Return ONLY the optimized text, nothing else.

Text: "{rule_cleaned}"

Optimized text:"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "think": False,   # qwen3 thinking mode kapat
                "options": {"temperature": 0.1, "num_ctx": 2048},
            },
            timeout=30,
        )
        if response.status_code == 200:
            result = response.json().get("response", "").strip().strip('\'"')
            log.info(f"  Ollama smart clean raw response: '{result}'")
            # Güvenlik: kelime sayısı çok değişmişse kural tabanlı sonucu kullan
            if result and abs(len(result.split()) - len(text.split())) <= 3:
                log.info(f"  Ollama noktalama iyileştirildi: '{text}' → '{result}'")
                return result
    except Exception as e:
        log.warning(f"  Ollama noktalama iyileştirmesi başarısız ({e}), kural tabanlı sonuç kullanılıyor.")

    return rule_cleaned


def _find_zero_crossing_sample(audio: "np.ndarray", target: int, window_ms: float = 25.0, sr: int = 24000) -> int:
    """
    target sample etrafında window_ms içinde en yakın zero-crossing'i bulur.
    Dalga sıfır çizgisini keseceğimiz nokta → 'tık' sesi matematiksel sıfır.
    """
    import numpy as np
    window = int(sr * window_ms / 1000)
    lo = max(0, target - window)
    hi = min(len(audio) - 2, target + window)
    if lo >= hi:
        return max(0, min(target, len(audio) - 1))
    best, best_dist = target, window + 1
    for i in range(lo, hi):
        if audio[i] * audio[i + 1] <= 0:
            dist = abs(i - target)
            if dist < best_dist:
                best_dist = dist
                best = i + 1
    return max(0, min(best, len(audio)))


def _trim_context_from_takes(seg_dir, n_takes: int, seg_text: str, context_prefix: str, aligner, take_offset: int = 0) -> None:
    """
    Context padding ile üretilen take'lerden prefix kısmını keser.

    Süreç:
    1. Whisper ile tam alignment (context + asıl metin)
    2. Aligned words içinde asıl metnin ilk kelimesini bul → trim zamanı
    3. Zero-crossing noktasına yuvarla
    4. Trimlenmiş sesi take dosyasının üzerine yaz

    Eğer trim başarısız olursa orijinal dosya korunur (failsafe).
    take_offset: başlangıç index'i (0=take_0, 1=take_1 vb.)
    """
    import re
    import soundfile as sf
    import librosa
    import numpy as np

    seg_words_clean = [re.sub(r"[^\w']", "", w).lower() for w in seg_text.strip().split() if w]
    if not seg_words_clean:
        return

    context_words_clean = [re.sub(r"[^\w']", "", w).lower() for w in context_prefix.strip().split() if w]
    context_word_count = len(context_words_clean)

    for t_idx in range(take_offset, take_offset + n_takes):
        take_path = seg_dir / f"take_{t_idx}.wav"
        if not take_path.exists():
            continue

        try:
            # Ham audio yükle
            full_text = f"{context_prefix}... {seg_text}"
            alignment = aligner.align(str(take_path), full_text)
            words = alignment.get("words", [])
            sr = alignment.get("sample_rate", 24000)

            # Sequence-matching to find where the actual segment starts
            trim_time = None
            
            if words:
                aligned_words_clean = [re.sub(r"[^\w']", "", w["word"]).lower() for w in words]
                
                best_idx = None
                best_score = -999.0
                
                n_aligned = len(aligned_words_clean)
                seg_len = len(seg_words_clean)
                
                # Check all possible split positions s from 0 to n_aligned
                # s is the index in aligned_words_clean where the segment starts.
                # Words before s belong to the context prefix.
                for s in range(n_aligned + 1):
                    # Suffix matching: how well does the segment match starting at s?
                    suffix_matches = 0
                    K = min(4, seg_len)
                    if K > 0 and s < n_aligned:
                        actual_k = min(K, n_aligned - s)
                        for k in range(actual_k):
                            if aligned_words_clean[s + k] == seg_words_clean[k]:
                                suffix_matches += 1
                                
                    # Prefix matching: how well does the context match ending at s?
                    prefix_matches = 0
                    P = min(4, context_word_count)
                    if P > 0 and s > 0:
                        actual_p = min(P, s)
                        for p in range(actual_p):
                            if aligned_words_clean[s - 1 - p] == context_words_clean[context_word_count - 1 - p]:
                                prefix_matches += 1
                                
                    # Penalty for distance from expected context word count
                    dist_penalty = -0.05 * abs(s - context_word_count)
                    
                    score = suffix_matches + prefix_matches + dist_penalty
                    
                    # If we expected segment words but matched 0, heavily penalize this split
                    if K > 0 and suffix_matches == 0:
                        score -= 5.0
                        
                    if score > best_score:
                        best_score = score
                        best_idx = s
                        
                if best_idx is not None and best_idx < len(words) and best_score > -2.0:
                    # Load audio first to perform zero-crossing alignment & energy analysis
                    audio, file_sr = librosa.load(str(take_path), sr=None)
                    
                    if best_idx > 0:
                        prev_end = words[best_idx - 1]["end"]
                        curr_start = words[best_idx]["start"]
                        gap = curr_start - prev_end

                        if gap >= 0.040:
                            # Geniş gap (≥40ms): asıl metnin başına doğru yu — curr_start'ı hedefle.
                            # ESKİ: prev_end + gap * 0.80 — geniş gaplerde (>1s) asıl sözcüğe çok yakın kesiyordu.
                            # YENİ: curr_start'tan 50ms öncesine git → fricative kuyruğu sönümlenebilir,
                            #        ama asıl sözcüğü kesmeyiz.
                            trim_time = curr_start - 0.050
                        elif gap > 0.005:
                            # Küçük gap (5-40ms): curr_start'tan 30ms önce
                            trim_time = curr_start - 0.030
                        else:
                            # Sıfır veya negatif gap: curr_start'tan 20ms önce (onset güvencesi)
                            trim_time = curr_start - 0.020

                        # Asla curr_start'a ≤ 20ms yakına gitme (kelime başını yemez)
                        trim_time = min(trim_time, curr_start - 0.020)
                        trim_time = max(0.0, min(trim_time, len(audio) / file_sr))
                        log.info(f"    Take {t_idx}: context trim match at index {best_idx} (score {best_score:.2f}) -> trim_time={trim_time:.3f}s (gap={gap:.3f}s, prev_end={prev_end:.3f}s, curr_start={curr_start:.3f}s)")
                    else:
                        # best_idx == 0
                        if context_word_count > 0 and words[0]["start"] > 0.4:
                            trim_time = max(0.0, words[0]["start"] - 0.020) # 20ms safety margin
                            log.info(f"    Take {t_idx}: context trim match at index 0 (score {best_score:.2f}) with delay -> trim_time={trim_time:.3f}s")
                        else:
                            trim_time = 0.0
                            log.info(f"    Take {t_idx}: context trim match at index 0 (score {best_score:.2f}), no trim needed")
                            
            # Fallback pure energy-based trim if Whisper matching failed or words list was empty but we expected context
            if trim_time is None and context_word_count > 0:
                audio, file_sr = librosa.load(str(take_path), sr=None)
                duration = len(audio) / file_sr
                
                # Estimate context duration based on word count (0.35s per word average)
                expected_context_dur = context_word_count * 0.35
                search_start = max(0.1, expected_context_dur - 0.6)
                search_end = min(duration - 0.2, expected_context_dur + 0.8)
                
                if search_end > search_start + 0.100:
                    win_samples = int(file_sr * 0.100) # 100ms window
                    step = max(1, int(file_sr * 0.010)) # 10ms step
                    start_sample = int(search_start * file_sr)
                    end_sample = int(search_end * file_sr)
                    
                    min_energy = float('inf')
                    best_sample = start_sample
                    
                    for s_idx in range(start_sample, end_sample - win_samples + 1, step):
                        energy = np.sum(audio[s_idx : s_idx + win_samples] ** 2)
                        if energy < min_energy:
                            min_energy = energy
                            best_sample = s_idx + win_samples // 2
                            
                    trim_time = best_sample / file_sr
                    log.info(f"    Take {t_idx}: Whisper match failed/empty -> Fallback energy-based trim at quietest window: trim_time={trim_time:.3f}s")
                else:
                    trim_time = expected_context_dur
                    log.info(f"    Take {t_idx}: Whisper match failed/empty -> Fallback expected context duration: trim_time={trim_time:.3f}s")

            if trim_time is not None:
                if 'audio' not in locals():
                    audio, file_sr = librosa.load(str(take_path), sr=None)
                
                # Zero-crossing trim (using the loaded audio)
                trim_sample = int(trim_time * file_sr)
                zc_sample = _find_zero_crossing_sample(audio, trim_sample, window_ms=25.0, sr=file_sr)

                trimmed = audio[zc_sample:]
                if len(trimmed) < file_sr * 0.1:  # 100ms'den kısa ise trim çok ileri gitti
                    log.warning(f"    Take {t_idx}: trim sonrası çok kısa ({len(trimmed)/file_sr:.3f}s), atlandı")
                    continue

                # Trimlenmiş sesi kaydet
                sf.write(str(take_path), trimmed, file_sr)
                log.info(f"    Take {t_idx}: {len(audio)/file_sr:.3f}s → {len(trimmed)/file_sr:.3f}s (context {trim_time:.3f}s kesildi)")


        except Exception as e:
            log.warning(f"    Take {t_idx} context trim başarısız ({e}), orijinal korundu")




@app.route("/api/project/state", methods=["GET"])
def get_state():
    state = load_project_state()
    if state is None:
        return jsonify({"initialized": False})
    return jsonify(state)

@app.route("/api/project/list", methods=["GET"])
def list_projects_api():
    try:
        conn = _get_db_connection()
        c = conn.cursor()
        c.execute("SELECT id, name, script_text, created_at, updated_at FROM projects ORDER BY updated_at DESC")
        rows = c.fetchall()
        projects = []
        for r in rows:
            projects.append({
                "id": r[0],
                "name": r[1],
                "script_text": r[2],
                "created_at": r[3],
                "updated_at": r[4]
            })
        conn.close()
        return jsonify({"projects": projects})
    except Exception as e:
        log.error(f"Error listing projects: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/project/load", methods=["POST"])
def load_project_api():
    data = request.json or {}
    project_id = data.get("id")
    if not project_id:
        return jsonify({"error": "Project ID is required"}), 400
    try:
        conn = _get_db_connection()
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('active_project_id', ?)", (str(project_id),))
        c.execute("SELECT state_json FROM projects WHERE id = ?", (project_id,))
        row = c.fetchone()
        conn.commit()
        conn.close()
        if not row:
            return jsonify({"error": "Project not found"}), 404
        return jsonify(json.loads(row[0]))
    except Exception as e:
        log.error(f"Error loading project: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/project/delete", methods=["POST"])
def delete_project_api():
    project_id = request.args.get("id")
    if project_id is not None:
        project_id = int(project_id)
    else:
        data = request.json or {}
        project_id = data.get("id")
        if project_id is not None:
            project_id = int(project_id)

    if not project_id:
        return jsonify({"error": "Project ID is required"}), 400
    try:
        conn = _get_db_connection()
        c = conn.cursor()
        
        # Check active project ID
        c.execute("SELECT value FROM settings WHERE key = 'active_project_id'")
        row = c.fetchone()
        if row and row[0] == str(project_id):
            c.execute("DELETE FROM settings WHERE key = 'active_project_id'")
            
        c.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        conn.commit()
        conn.close()

        # Clean up project-specific workspace directory from disk
        import shutil
        project_dir = Path("pipeline_work") / f"project_{project_id}"
        if project_dir.exists() and project_dir.is_dir():
            shutil.rmtree(project_dir, ignore_errors=True)

        return jsonify({"deleted": True})
    except Exception as e:
        log.error(f"Error deleting project: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/project/reset", methods=["POST"])
def reset_project_api():
    try:
        conn = _get_db_connection()
        c = conn.cursor()
        c.execute("DELETE FROM settings WHERE key = 'active_project_id'")
        conn.commit()
        conn.close()
        return jsonify({"reset": True})
    except Exception as e:
        log.error(f"Error resetting project settings: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/ollama/models", methods=["GET"])
def get_ollama_models():
    """Fetches list of installed local models from Ollama."""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            return jsonify({"online": True, "models": models})
    except Exception as e:
        log.warning(f"Failed to connect to Ollama for models: {e}")
    return jsonify({"online": False, "models": []})

@app.route("/api/project/init", methods=["POST"])
def init_project():
    data = request.json or {}
    text = data.get("text", "").strip()
    use_case = data.get("use_case", "youtube_narration")
    tone = data.get("tone", "natural")
    speaker = data.get("speaker", "Ethan")
    model_size = data.get("model_size", "1.7B")
    n_takes = int(data.get("n_takes", 3))
    backend = data.get("backend", "mlx")
    ref_audio = data.get("ref_audio", "").strip() or None
    ref_text = data.get("ref_text", "").strip() or None
    voice_mode = data.get("voice_mode", "preset")  # "preset" | "clone" | "design"
    design_prompt = data.get("design_prompt", "").strip() or None
    project_name = data.get("project_name", "").strip() or data.get("name", "").strip() or None

    # Clone modda ref_audio boşsa, uploaded_ref.wav varsa otomatik kullan
    if voice_mode == "clone" and not ref_audio:
        fallback_ref = WORK_DIR / "uploaded_ref.wav"
        if fallback_ref.exists():
            ref_audio = str(fallback_ref)
            log.info(f"Clone modu: ref_audio otomatik algılandı → {ref_audio}")

    # Clone: Speaker Space Interpolation parametreleri
    clone_alpha = float(data.get("clone_alpha", 1.0))
    clone_topk  = int(data.get("clone_topk", 3))
    clone_speaker = data.get("clone_speaker", "ryan").strip().lower()

    llm_backend = data.get("llm_backend", "rule_based")
    ollama_model = data.get("ollama_model", "qwen3:8b").strip()

    if not text:
        return jsonify({"error": "Text script is required"}), 400

    log.info(f"Initializing new project: backend={backend}, voice_mode={voice_mode}, use_case={use_case}, tone={tone}, llm_backend={llm_backend}, ollama_model={ollama_model}")
    ProductionLogger.reset()
    
    # 1. Clear work dir structure but keep directory
    if WORK_DIR.exists():
        import shutil
        # delete old folders like seg_* and wav files
        for p in WORK_DIR.iterdir():
            if p.is_dir() and p.name.startswith("seg_"):
                shutil.rmtree(p)
            elif p.is_file() and p.suffix == ".wav" and p.name != "uploaded_ref.wav":
                p.unlink()

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Speech map generate
    character_profile = ""
    if voice_mode == "design" and design_prompt:
        character_profile = f"Voice Profile Design: {design_prompt}"
    elif voice_mode == "preset":
        character_profile = f"Preset Speaker: {speaker} ({tone} tone)"
    elif voice_mode == "clone":
        character_profile = "Cloned Speaker Voice"

    # SFT algılama: eğitilmiş bir adapter dosyası varsa SFT modü aktif
    is_sft = False
    if voice_mode in ("preset", "clone"):
        # Preset: speaker adı ile ara; Clone: ref_audio'ya bağlı isim ile ara
        candidate_name = speaker if voice_mode == "preset" else None
        if voice_mode == "clone" and ref_audio:
            try:
                _conn = _get_db_connection()
                _c = _conn.cursor()
                _c.execute("SELECT name FROM timbres WHERE ref_audio = ?", (ref_audio,))
                _row = _c.fetchone()
                _conn.close()
                if _row:
                    candidate_name = _row[0]
            except Exception as _e:
                log.warning(f"SFT algılama sorgusu başarısız: {_e}")
        if candidate_name:
            _adapter_path = WORK_DIR / f"{candidate_name}_adapter.safetensors"
            if _adapter_path.exists() and _adapter_path.stat().st_size > 0:
                is_sft = True
                log.info(f"SFT modu algılandı: '{candidate_name}' adaptör mevcut. Speech map SFT modunda üretilecek.")

    ProductionLogger.log_step(
        "Project Initialization",
        f"Project Name: {project_name or 'Untitled'}\n"
        f"Backend: {backend}\n"
        f"Voice Mode: {voice_mode}\n"
        f"Speaker: {speaker}\n"
        f"Tone: {tone}\n"
        f"LLM Backend: {llm_backend}\n"
        f"Ollama Model: {ollama_model}\n"
        f"SFT Mode Active: {is_sft}\n"
        f"Script Text:\n{text}"
    )

    map_gen = SpeechMapGenerator(llm_backend=llm_backend, model_name=ollama_model)
    speech_map = map_gen.generate(
        text=text,
        use_case=use_case,
        tone=tone,
        character_profile=character_profile,
        cache_path=None,
        is_sft=is_sft,
        voice_mode=voice_mode,
    )

    # 3. Create initial state
    segments = []
    for idx, seg_plan in enumerate(speech_map["segments"]):
        segments.append({
            "idx": idx,
            "text": seg_plan["text"],
            "emotion": seg_plan["emotion"],
            "stress": seg_plan["stress"],
            "speed": seg_plan["speed"],
            "pause_after": seg_plan["pause_after"],
            "tts_instruct": seg_plan["tts_instruct"],
            "intonation_trend": seg_plan.get("intonation_trend", "stable"),
            "vowel_stretching": seg_plan.get("vowel_stretching", "normal"),
            "stress_anchor": seg_plan.get("stress_anchor", ""),
            "pitch": seg_plan.get("pitch", 0.0),
            "thinking_enabled": bool(seg_plan.get("thinking_enabled", False)),
            "temperature": seg_plan.get("temperature", 0.6),
            "locked_seed": seg_plan.get("locked_seed", None),
            "lora_strength": seg_plan.get("lora_strength", 0.75),
            "status": "pending",  # pending, generating, completed, error
            "takes": [],          # takes list with audios & alignment/scores
            "selected_take": -1,
            "error_msg": None
        })

    if not project_name:
        project_name = text[:30] + "..." if len(text) > 30 else text
    if not project_name:
        project_name = "Untitled Project"

    state = {
        "initialized": True,
        "name": project_name,
        "config": {
            "backend": backend,
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            "voice_mode": voice_mode,
            "design_prompt": design_prompt,
            "llm_backend": llm_backend,
            "ollama_model": ollama_model,
            "use_case": use_case,
            "tone": tone,
            "speaker": speaker,
            "model_size": model_size,
            "n_takes": n_takes,
            "is_sft": is_sft,
            "clone_alpha": clone_alpha,
            "clone_topk": clone_topk,
            "clone_speaker": clone_speaker,
            "overall_emotion": speech_map["overall"]["emotion"]
        },
        "segments": segments,
        "stitched": False,
        "final_audio_path": None
    }

    # Insert new project into SQLite database
    try:
        conn = _get_db_connection()
        c = conn.cursor()
        now = datetime.datetime.now().isoformat()
        c.execute(
            "INSERT INTO projects (name, script_text, state_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (project_name, text, json.dumps(state, ensure_ascii=False), now, now)
        )
        project_id = c.lastrowid
        state["id"] = project_id
        c.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('active_project_id', ?)", (str(project_id),))
        conn.commit()
        conn.close()

        # Copy reference audio into project directory to preserve it across sessions
        project_dir = WORK_DIR / f"project_{project_id}"
        project_dir.mkdir(parents=True, exist_ok=True)
        ref_audio = state["config"].get("ref_audio")
        if ref_audio and os.path.exists(ref_audio):
            import shutil
            proj_ref_path = project_dir / "uploaded_ref.wav"
            shutil.copy2(ref_audio, str(proj_ref_path))
            state["config"]["ref_audio"] = str(proj_ref_path)
            log.info(f"Copied project reference audio to: {proj_ref_path}")
            
    except Exception as e:
        log.error(f"Error initializing project in SQLite: {e}")

    save_project_state(state)
    return jsonify(state)

@app.route("/api/project/append", methods=["POST"])
def append_project():
    try:
        raw_data = request.get_data()
        log.info(f"Raw request data: {raw_data}")
        data = json.loads(raw_data) if raw_data else {}
    except Exception as e:
        log.error(f"JSON decode error: {e}")
        data = {}

    text = data.get("text", "").strip()
    use_case = data.get("use_case", "youtube_narration")
    tone = data.get("tone", "natural")
    speaker = data.get("speaker", "Ethan")
    model_size = data.get("model_size", "1.7B")
    n_takes = int(data.get("n_takes", 3))
    backend = data.get("backend", "mlx")
    ref_audio = data.get("ref_audio", "").strip() or None
    ref_text = data.get("ref_text", "").strip() or None
    voice_mode = data.get("voice_mode", "preset")
    design_prompt = data.get("design_prompt", "").strip() or None
    llm_backend = data.get("llm_backend", "rule_based")
    ollama_model = data.get("ollama_model", "qwen3:8b").strip()

    if not text:
        return jsonify({"error": "Text script is required"}), 400

    state = load_project_state()
    if not state:
        return init_project()

    log.info(f"Appending to project: backend={backend}, voice_mode={voice_mode}, use_case={use_case}, tone={tone}")

    if voice_mode == "clone" and not ref_audio:
        fallback_ref = WORK_DIR / "uploaded_ref.wav"
        if fallback_ref.exists():
            ref_audio = str(fallback_ref)

    # Clone: Speaker Space Interpolation parametreleri
    clone_alpha = float(data.get("clone_alpha", 1.0))
    clone_topk  = int(data.get("clone_topk", 3))
    clone_speaker = data.get("clone_speaker", "ryan").strip().lower()

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    character_profile = ""
    if voice_mode == "design" and design_prompt:
        character_profile = f"Voice Profile Design: {design_prompt}"
    elif voice_mode == "preset":
        character_profile = f"Preset Speaker: {speaker} ({tone} tone)"
    elif voice_mode == "clone":
        character_profile = "Cloned Speaker Voice"

    # SFT algılama (append modü)
    is_sft = False
    if voice_mode in ("preset", "clone"):
        candidate_name = speaker if voice_mode == "preset" else None
        if voice_mode == "clone" and ref_audio:
            try:
                _conn = _get_db_connection()
                _c = _conn.cursor()
                _c.execute("SELECT name FROM timbres WHERE ref_audio = ?", (ref_audio,))
                _row = _c.fetchone()
                _conn.close()
                if _row:
                    candidate_name = _row[0]
            except Exception as _e:
                log.warning(f"SFT algılama sorgusu başarısız (append): {_e}")
        if candidate_name:
            _adapter_path = WORK_DIR / f"{candidate_name}_adapter.safetensors"
            if _adapter_path.exists() and _adapter_path.stat().st_size > 0:
                is_sft = True
                log.info(f"SFT modu algılandı (append): '{candidate_name}' adaptör mevcut.")

    map_gen = SpeechMapGenerator(llm_backend=llm_backend, model_name=ollama_model)
    speech_map = map_gen.generate(
        text=text,
        use_case=use_case,
        tone=tone,
        character_profile=character_profile,
        cache_path=None,
        is_sft=is_sft,
        voice_mode=voice_mode,
    )

    start_idx = len(state.get("segments", []))
    new_segments = []
    for i, seg_plan in enumerate(speech_map["segments"]):
        new_segments.append({
            "idx": start_idx + i,
            "text": seg_plan["text"],
            "emotion": seg_plan["emotion"],
            "stress": seg_plan["stress"],
            "speed": seg_plan["speed"],
            "pause_after": seg_plan["pause_after"],
            "tts_instruct": seg_plan["tts_instruct"],
            "intonation_trend": seg_plan.get("intonation_trend", "stable"),
            "vowel_stretching": seg_plan.get("vowel_stretching", "normal"),
            "stress_anchor": seg_plan.get("stress_anchor", ""),
            "pitch": seg_plan.get("pitch", 0.0),
            "thinking_enabled": bool(seg_plan.get("thinking_enabled", False)),
            "temperature": seg_plan.get("temperature", 0.6),
            "locked_seed": seg_plan.get("locked_seed", None),
            "lora_strength": seg_plan.get("lora_strength", 0.75),
            "status": "pending",
            "takes": [],
            "selected_take": -1,
            "error_msg": None
        })

    if "segments" not in state:
        state["segments"] = []
    state["segments"].extend(new_segments)
    
    state["config"].update({
        "backend": backend,
        "ref_audio": ref_audio,
        "ref_text": ref_text,
        "voice_mode": voice_mode,
        "design_prompt": design_prompt,
        "llm_backend": llm_backend,
        "ollama_model": ollama_model,
        "use_case": use_case,
        "tone": tone,
        "speaker": speaker,
        "model_size": model_size,
        "n_takes": n_takes,
        "is_sft": is_sft or state["config"].get("is_sft", False),
        "clone_alpha": clone_alpha,
        "clone_topk": clone_topk,
        "clone_speaker": clone_speaker,
        "overall_emotion": speech_map["overall"]["emotion"]
    })
    
    state["stitched"] = False

    try:
        project_id = state.get("id")
        if project_id:
            conn = _get_db_connection()
            c = conn.cursor()
            now = datetime.datetime.now().isoformat()
            
            c.execute("SELECT script_text FROM projects WHERE id = ?", (project_id,))
            row = c.fetchone()
            old_text = row[0] if row else ""
            new_text = old_text + "\n" + text if old_text else text
            
            c.execute(
                "UPDATE projects SET script_text = ?, state_json = ?, updated_at = ? WHERE id = ?",
                (new_text, json.dumps(state, ensure_ascii=False), now, project_id)
            )
            conn.commit()
            conn.close()
    except Exception as e:
        log.error(f"Error appending project in SQLite: {e}")

    save_project_state(state)
    return jsonify(state)

@app.route("/api/segment/update", methods=["POST"])
def update_segment():
    data = request.json or {}
    idx = int(data.get("idx", -1))
    text = data.get("text", "").strip()
    speed = float(data.get("speed", 1.0))
    stress = float(data.get("stress", 0.6))
    pause_after = float(data.get("pause_after", 0.3))
    tts_instruct = data.get("tts_instruct", "").strip()
    pitch = float(data.get("pitch", 0.0))
    intonation_trend = data.get("intonation_trend", "stable").strip()
    vowel_stretching = data.get("vowel_stretching", "normal").strip()
    thinking_enabled = bool(data.get("thinking_enabled", False))
    temperature = float(data.get("temperature", 0.6))
    lora_strength = float(data.get("lora_strength", 0.75))
    
    locked_seed_raw = data.get("locked_seed")
    locked_seed = int(locked_seed_raw) if locked_seed_raw is not None and str(locked_seed_raw).strip() != "" else None

    state = load_project_state()
    if not state or idx < 0 or idx >= len(state["segments"]):
        return jsonify({"error": "Invalid segment index or project not initialized"}), 400

    seg = state["segments"][idx]
    
    # Check if currently generating to prevent race conditions
    if seg.get("status") == "generating":
        return jsonify({"error": "Segment is currently generating. Please wait until it completes."}), 409

    seg["text"] = text
    seg["speed"] = speed
    seg["stress"] = stress
    seg["tts_instruct"] = tts_instruct
    seg["pitch"] = pitch
    seg["intonation_trend"] = intonation_trend
    seg["vowel_stretching"] = vowel_stretching
    seg["thinking_enabled"] = thinking_enabled
    seg["temperature"] = temperature
    seg["locked_seed"] = locked_seed
    seg["lora_strength"] = lora_strength
    seg["pause_after"] = pause_after

    if not seg.get("takes"):
        seg["status"] = "pending"

    state["stitched"] = False # Need to re-stitch since changes occurred
    save_project_state(state)
    return jsonify(state)

@app.route("/api/segment/delete", methods=["POST"])
def delete_segment():
    idx_val = request.args.get("idx")
    if idx_val is not None:
        idx = int(idx_val)
    else:
        data = request.json or {}
        idx = int(data.get("idx", -1))

    state = load_project_state()
    if not state or idx < 0 or idx >= len(state["segments"]):
        return jsonify({"error": "Invalid segment index or project not initialized"}), 400

    # Remove the segment
    del state["segments"][idx]
    
    # Update indices for remaining segments
    for i, seg in enumerate(state["segments"]):
        seg["idx"] = i

    state["stitched"] = False
    save_project_state(state)
    return jsonify(state)

@app.route("/api/segment/generate", methods=["POST"])
def generate_segment():
    """Segment üretimini worker queue'suna ekler ve hemen 202 döner.

    MLX üretimi dedicated worker thread'de çalışır (server altı açıklama).
    Handler lock'a dokunmaz — meşguliyet queue'nun doğal sıralamasıyla çözülür.
    Sadece race kontrolü: aynı segment zaten queued/generating ise 409.
    """
    data = request.json or {}
    idx = int(data.get("idx", -1))

    state = load_project_state()
    if not state or idx < 0 or idx >= len(state["segments"]):
        return jsonify({"error": "Invalid segment index or project not initialized"}), 400

    seg = state["segments"][idx]

    # Race kontrolü: aynı segment çift tıklama / Generate All içinde tekrar
    with _generation_lock:
        already_queued = idx in _generation_status["queued"] or seg["status"] == "generating"
        if already_queued:
            return jsonify({"error": "Segment is already being generated"}), 409

        # Enqueue: status'a işaretle ve queue'ya koy
        _generation_status["queued"].add(idx)
        _generation_status["queue_order"].append(idx)

    seg["status"] = "generating"
    state["stitched"] = False
    save_project_state(state)

    ProductionLogger.log_section(f"Segment #{idx} Generation")
    ProductionLogger.log_step(
        "Initiating Generation (queued)",
        f"Text: '{seg['text']}'\n"
        f"Parameters: speed={seg['speed']}, stress={seg['stress']}, pitch={seg['pitch']}, temperature={seg['temperature']}, lora_strength={seg.get('lora_strength', 0.75)}"
    )

    # Job'u worker'a devret — handler burada biter, uzun bekleme yok
    _generation_queue.put(idx)

    # 202 Accepted: üretim arka planda sürüyor, frontend poll edecek
    return jsonify(state), 202


def _do_generation(idx):
    """Worker thread'de çağrılır. MLX üretimini gerçekleştirir.

    _generation_lock burada acquire EDİLMEZ — worker tek thread olduğu için
    MLX tek-thread garantisi doğal olarak sağlanır. Lock sadece _generation_status
    dict erişimlerini seri hale getirmek için (çok kısa critical sections) kullanılır.
    """
    state = load_project_state()
    if not state or idx < 0 or idx >= len(state["segments"]):
        log.error(f"_do_generation: invalid segment idx={idx}")
        return
    seg = state["segments"][idx]

    try:
        config = state["config"]
        n_takes = config["n_takes"]
        speaker = config["speaker"]
        model_size = config["model_size"]
        backend = config.get("backend", "pytorch")
        ref_audio = config.get("ref_audio")
        ref_text = config.get("ref_text")
        voice_mode = config.get("voice_mode", "preset")
        design_prompt = config.get("design_prompt")
        clone_alpha = float(config.get("clone_alpha", 1.0))
        clone_topk  = int(config.get("clone_topk", 3))

        # Clone modda ref_audio yoksa uploaded_ref.wav'ı otomatik kullan
        if voice_mode == "clone" and not ref_audio:
            fallback_ref = WORK_DIR / "uploaded_ref.wav"
            if fallback_ref.exists():
                ref_audio = str(fallback_ref)
                log.info(f"Clone modu: ref_audio otomatik algılandı → {ref_audio}")

        project_id = state.get("id")
        seg_dir = WORK_DIR / f"project_{project_id}" / f"seg_{idx:03d}"
        seg_dir.mkdir(parents=True, exist_ok=True)

        generator = get_generator(
            backend=backend,
            model_size=model_size,
            speaker=speaker,
            ref_audio=ref_audio,
            ref_text=ref_text,
            voice_mode=voice_mode,
            design_prompt=design_prompt,
            clone_alpha=clone_alpha,
            clone_topk=clone_topk,
        )
        aligner = get_aligner()
        scorer = MODELS["scorer"]

        # ── NOKTALAMA PRE-PROCESSING ────────────────────────────────────
        # Her zaman kural tabanlı temizlik çalışır (Ollama kapalı olsa bile).
        # Ollama açıksa kural tabanlı sonucu daha da iyileştirebilir.
        llm_backend = config.get("llm_backend", "rule_based")
        ollama_model = config.get("ollama_model", "qwen3:8b")

        if llm_backend == "ollama":
            seg_text_for_tts = _smart_clean_punctuation(seg["text"], seg, ollama_model)
        else:
            # rule_based modda da yerel temizlik yap
            seg_text_for_tts = _rule_based_clean_punctuation(seg["text"], seg)

        ProductionLogger.log_step(
            "Text Normalization Complete",
            f"Original Script: '{seg['text']}'\n"
            f"TTS Target Text: '{seg_text_for_tts}'"
        )

        # ── CONTEXT PADDING ──────────────────────────────────────────
        # Bir önceki segmentin son 4 kelimesini prefix olarak kullan.
        # Model bir önceki cümlenin tonuyla üretime başlar → doğal geçiş.
        context_prefix = None
        if idx > 0:
            prev_seg = state["segments"][idx - 1]
            prev_text_raw = prev_seg.get("text", "").strip()

            # Prosodic markup'ı context prefix'ten temizle:
            # CAPS → lowercase (Whisper alignment için normalize),
            # "..." → " " (ellipsis sesli üretimde duraklama yaratır ama alignment'ı bozar)
            import re as _re
            prev_text_clean = _re.sub(r'\.\.\.', ' ', prev_text_raw)          # ellipsis sil
            prev_text_clean = _re.sub(r'\b([A-Z]{2,})\b', lambda m: m.group(1).lower(), prev_text_clean)  # CAPS→lower
            prev_text_clean = _re.sub(r'\s+', ' ', prev_text_clean).strip()

            prev_words = prev_text_clean.split()
            if len(prev_words) >= 2:
                ctx_words = prev_words[-4:]

                # Çakışma önleme: context prefix'in son kelimesi asıl segment'in
                # ilk kelimesiyle aynıysa o kelimeyi prefix'ten çıkar
                # (Whisper "so ... so" gibi çakışmalarda yanlış split yapıyor)
                seg_first_words = [
                    _re.sub(r"[^\w']", "", w).lower()
                    for w in seg_text_for_tts.strip().split()[:3]
                    if _re.sub(r"[^\w']", "", w)
                ]
                while ctx_words:
                    last = _re.sub(r"[^\w']", "", ctx_words[-1]).lower()
                    if last in seg_first_words:
                        ctx_words = ctx_words[:-1]
                    else:
                        break

                # Fonetik geçiş güvencesi: context prefix'in son kelimesi karmaşık
                # ünsüz kümesiyle (consonant cluster) veya sessiz-e öncesi fricative ile
                # bitiyorsa trim noktası bu sesin sönümlenme kuyruğuna denk gelir.
                # "chosen" → -ng; "chose" → -se (/z/ + silent e); "voice" → -ce (/s/)
                # Sessiz-e öncesi fricative: chose/close/nose → /z/; place/voice → /s/
                _HARSH_ENDINGS = (
                    'sh', 'th', 'ch', 'ck', 'xt', 'ft', 'sk', 'sp', 'st',
                    'ld', 'nd', 'nk', 'ng', 'nce', 'nts',  # ng: "everything","thing"
                    'ss', 'x',                               # "pass", "mix" → sibilant
                    'se', 'ce', 'ze', 'ge',                  # chose(/z/), voice(/s/), freeze(/z/), change(/dʒ/)
                )
                if ctx_words:
                    last_clean = _re.sub(r"[^\w']", "", ctx_words[-1]).lower()
                    if any(last_clean.endswith(e) for e in _HARSH_ENDINGS) and len(ctx_words) > 1:
                        ctx_words = ctx_words[:-1]
                        log.info(f"  Context prefix: son kelime fricative/cluster bitiyor → bir öncekinde bitti: '{' '.join(ctx_words)}'")

                # Kısa kelime patlayıcı guard: "but", "not", "yet", "at", "it", "up" gibi
                # ≤3 harf + patlayıcı sonu → function word, trailing vowel yok → click
                # NOT: ≤4 değil — "dark", "back" gibi içerik kelimeleri yakalanmasın
                # Örn: "videos is cool, but" → "but" → /t/ tıklama sesi başta kalıyor
                _SHORT_PLOSIVE_ENDS = ('t', 'd', 'k', 'p', 'b', 'g')
                if ctx_words:
                    last_clean = _re.sub(r"[^\w']", "", ctx_words[-1]).lower()
                    if len(last_clean) <= 3 and any(last_clean.endswith(e) for e in _SHORT_PLOSIVE_ENDS) and len(ctx_words) > 1:
                        ctx_words = ctx_words[:-1]
                        log.info(f"  Context prefix: kısa patlayıcı bitiş '{last_clean}' → bir öncekinde bitti: '{' '.join(ctx_words)}'")
                # Sarkan edat/article guard: "on", "in", "a", "the" gibi son kelimelerin
                # bir sonraki segmentin ilk kelimesiyle gramatik birleşme yaratmasını önle.
                # "human being on" + "I RAN" → Whisper: "on a rana" (yanlış birleşme)
                _DANGLING_FUNCTION_WORDS = {
                    'on', 'in', 'at', 'to', 'of', 'by', 'for', 'with', 'from',
                    'a', 'an', 'the', 'is', 'are', 'was', 'be', 'been',
                    'and', 'or', 'but', 'so',
                }
                if ctx_words:
                    last_clean = _re.sub(r"[^\w']", "", ctx_words[-1]).lower()
                    if last_clean in _DANGLING_FUNCTION_WORDS and len(ctx_words) > 1:
                        ctx_words = ctx_words[:-1]
                        log.info(f"  Context prefix: sarkan edat/article '{last_clean}' → bir öncekinde bitti: '{' '.join(ctx_words)}'")





                if len(ctx_words) >= 1:
                    context_prefix = " ".join(ctx_words)
                    log.info(f"  Context prefix: '{context_prefix}'")


        # Define callback to process each take immediately after it is written
        aligned_takes = []
        takes_data = []

        def on_take_generated(t_idx, take_path_str):
            take_path = Path(take_path_str)
            # 1. Trim context for this single take immediately!
            if context_prefix:
                _trim_context_from_takes(
                    seg_dir=seg_dir,
                    n_takes=1,
                    seg_text=seg["text"],
                    context_prefix=context_prefix,
                    aligner=aligner,
                    take_offset=t_idx,
                )
            
            # 2. Align (context trimlenmiş dosya üzerinde)
            alignment = aligner.align(str(take_path), seg["text"])
            align_path = seg_dir / f"take_{t_idx}_aligned.json"
            with open(align_path, "w") as f:
                json.dump(alignment, f, indent=2)

            # 3. Hallucination retry
            _transcription_text = alignment.get("text", "").strip()
            _duration = alignment.get("duration", 0.0)
            _is_hallucinating = (len(_transcription_text) == 0 and _duration >= 3.0)

            if _is_hallucinating:
                log.warning(
                    f"  Take {t_idx}: Hallucination tespit edildi (transcription boş, duration={_duration:.2f}s). "
                    f"3 strateji ile yeniden üretiliyor..."
                )

                _fallback_seeds = [42, 1337, 9999, 7777, 31337]
                _orig_seed = _fallback_seeds[t_idx % len(_fallback_seeds)]
                _retry_seed = _orig_seed + 500
                _retry_temp = 0.38
                log.info(f"  Take {t_idx} retry: Seed {_orig_seed} → {_retry_seed}, Temp → {_retry_temp}")

                import re as _re_local
                _retry_text = seg_text_for_tts
                _retry_text = _re_local.sub(
                    r'(?<![.,!?])\s+(and|but|so|or|because|when|that|which|if|as)\b',
                    r', \1',
                    _retry_text,
                    flags=_re_local.IGNORECASE
                )
                _retry_text = _re_local.sub(r',\s*,', ',', _retry_text)
                _retry_text = _re_local.sub(r'^\s*,\s*', '', _retry_text)

                _retry_full = (
                    f"{context_prefix}... {_retry_text}"
                    if context_prefix
                    else _retry_text
                )
                log.info(f"  Take {t_idx} retry: Metin → '{_retry_text[:80]}'")

                retry_take_path = seg_dir / f"take_{t_idx}_retry.wav"
                try:
                    generator.generate(
                        _retry_full,
                        str(retry_take_path),
                        instruct=seg.get("tts_instruct", "Natural delivery."),
                        language="English",
                        seed=_retry_seed,
                        temperature=_retry_temp,
                    )
                    import shutil
                    shutil.copy(str(retry_take_path), str(take_path))

                    if context_prefix:
                        _trim_context_from_takes(
                            seg_dir=seg_dir,
                            n_takes=1,
                            seg_text=seg["text"],
                            context_prefix=context_prefix,
                            aligner=aligner,
                            take_offset=t_idx,
                        )

                    alignment = aligner.align(str(take_path), seg["text"])
                    with open(align_path, "w") as f:
                        json.dump(alignment, f, indent=2)

                    _retry_transcript = alignment.get("text", "").strip()
                    if _retry_transcript:
                        log.info(f"  Take {t_idx}: Retry başarılı → '{_retry_transcript[:60]}'")
                    else:
                        log.warning(f"  Take {t_idx}: Retry sonrası hâlâ boş transcription.")
                except Exception as _retry_err:
                    log.warning(f"  Take {t_idx}: Retry başarısız ({_retry_err}), orijinal take korunuyor.")

            aligned_takes.append({"path": str(take_path), "alignment": alignment})

            # 4. Score
            from aligner import BreathingAnalyzer
            overall_score = scorer._score_take({"path": str(take_path), "alignment": alignment}, seg["text"])
            energy_score = scorer._score_energy(str(take_path))
            pitch_score = scorer._score_pitch_variety(str(take_path))
            breathing = alignment.get("breathing", {})
            breathing_score = BreathingAnalyzer.score_take_breathing(alignment)

            words = alignment.get("words", [])
            confidences = [w.get("confidence", 0.5) for w in words]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

            match_label = ""
            import difflib
            import re
            target_clean = re.sub(r"[^\w'\s]", "", seg["text"].lower())
            target_words = [w for w in target_clean.split() if w]
            transcribed_clean = re.sub(r"[^\w'\s]", "", alignment.get("text", "").lower())
            transcribed_words = [w for w in transcribed_clean.split() if w]
            if target_words:
                matcher = difflib.SequenceMatcher(None, target_words, transcribed_words)
                match_ratio = matcher.ratio()
                if match_ratio < 0.4:
                    match_label = " [Failed/Mismatch]"

            takes_data.append({
                "take_idx": t_idx,
                "audio_url": f"/api/audio?path={str(take_path)}",
                "score": round(overall_score, 3),
                "metrics": {
                    "energy": round(energy_score, 3),
                    "pitch_variety": round(pitch_score, 3),
                    "pronunciation_confidence": round(avg_confidence, 3),
                    "breathing": round(breathing_score, 3),
                    "avg_gap_ms": breathing.get("avg_gap_ms", 0),
                    "robotic_pairs": len(breathing.get("robotic_pairs", [])),
                },
                "duration": alignment["duration"],
                "transcription": alignment["text"] + match_label
            })

            # Save state immediately
            seg["takes"] = list(takes_data)
            seg["selected_take"] = -1
            save_project_state(state)

            ProductionLogger.log_step(
                f"Take #{t_idx} Analyzed & Saved Progressively",
                f"Score: {overall_score:.3f}\n"
                f"Transcription: '{alignment.get('text', '')}'\n"
                f"Duration: {alignment.get('duration', 0.0):.2f}s\n"
                f"Confidence: {avg_confidence:.3f}\n"
                f"Energy: {energy_score:.3f}\n"
                f"Pitch Variety: {pitch_score:.3f}\n"
                f"Breathing Score: {breathing_score:.3f}"
            )

        # Generate N takes (context ile) using callback
        ProductionLogger.log_step(
            "Generating N Takes (Progressive)",
            f"Context Prefix: '{context_prefix or 'None'}'\n"
            f"Number of takes: {n_takes}"
        )
        generator.generate_n_takes(
            text=seg_text_for_tts,
            output_dir=str(seg_dir),
            n=n_takes,
            speech_map_segment=seg,
            context_prefix=context_prefix,
            on_take_callback=on_take_generated,
        )

        # Select best take after all takes generated
        best_phrase_list = scorer.select_best_phrases(aligned_takes, seg["text"])
        best_take_idx = best_phrase_list[0]["take_idx"]

        seg["takes"] = takes_data
        seg["selected_take"] = best_take_idx

        ProductionLogger.log_step(
            "Segment Generation Completed",
            f"Selected Best Take: T{best_take_idx}\n"
            f"Best Score: {best_phrase_list[0]['score']:.3f}"
        )

        # Prosodik öbek haritasını oluştur
        try:
            stitcher = MODELS["stitcher"]
            take_paths = [Path(t["path"]) for t in aligned_takes]
            alignments = [t["alignment"] for t in aligned_takes]
            seg["chunks"] = stitcher.build_segment_chunks(seg, take_paths, alignments)
        except Exception as ex:
            log.warning(f"Segment {idx} için öbek haritası oluşturulamadı: {ex}")
            seg["chunks"] = []

        seg["status"] = "completed"
        seg["error_msg"] = None

    except Exception as e:
        log.exception(f"Error generating segment {idx}")
        seg["status"] = "error"
        seg["error_msg"] = str(e)
        ProductionLogger.log_step(
            "Segment Generation Failed",
            f"Error: {e}"
        )

    finally:
        # V3 ÇÖKME ÖNLEME: Her segmentten sonra derin bellek temizliği.
        # Take bazlı temizlik _generate_mlx finally'de zaten var, ama segment
        # başına 3 take + alignment + stitcher chunk'ları Metal memory pool'da
        # birikiyor. Uzun "Generate All" koşumlarında bu birikim çökme yapıyor.
        import gc as _gc_seg
        _gc_seg.collect()
        try:
            import mlx.core as _mx_seg
            _mx_seg.clear_cache()
        except Exception:
            pass

    state["stitched"] = False
    save_project_state(state)


def _generation_worker():
    """Dedicated MLX worker thread. Job'ları sırayla _do_generation'a devreder.

    Bu thread dışında MLX/Whisper'a erişen hiçbir thread yok — böylece Metal
    C++ runtime thread-safety kısıtlaması güvenle sağlanır. Flask thread'leri
    (threaded=True) sadece JSON + ses dosyası I/O yapar.
    """
    log.info("MLX generation worker thread started")
    while True:
        idx = _generation_queue.get()
        try:
            with _generation_lock:
                _generation_status["active"] = True
                _generation_status["segment_idx"] = idx
            _do_generation(idx)
        except Exception:
            log.exception("Worker error on segment %s", idx)
        finally:
            with _generation_lock:
                _generation_status["active"] = False
                _generation_status["segment_idx"] = None
                _generation_status["queued"].discard(idx)
                try:
                    _generation_status["queue_order"].remove(idx)
                except ValueError:
                    pass
            _generation_queue.task_done()


# Worker thread'i modül yüklenince daemon olarak başlat — ana process çıkınca ölür
_generation_worker_thread = threading.Thread(
    target=_generation_worker, name="mlx-generation-worker", daemon=True
)
_generation_worker_thread.start()

@app.route("/api/segment/create", methods=["POST"])
def create_segment():
    state = load_project_state()
    if not state:
        return jsonify({"error": "Project not initialized"}), 400

    new_idx = len(state.get("segments", []))
    new_seg = {
        "idx": new_idx,
        "text": "New segment text...",
        "emotion": "neutral",
        "stress": 0.6,
        "speed": 1.0,
        "pause_after": 0.3,
        "tts_instruct": "Speak naturally.",
        "intonation_trend": "stable",
        "vowel_stretching": "normal",
        "stress_anchor": "",
        "pitch": 0.0,
        "thinking_enabled": False,
        "temperature": 0.6,
        "locked_seed": None,
        "lora_strength": 0.75,
        "status": "pending",
        "takes": [],
        "selected_take": -1,
        "error_msg": None
    }

    state["segments"].append(new_seg)
    state["stitched"] = False
    save_project_state(state)
    return jsonify(state)

@app.route("/api/generation/status", methods=["GET"])
def generation_status():
    """Frontend'in MLX worker meşguliyetini ve kuyruğu sorgulaması için."""
    with _generation_lock:
        return jsonify({
            "busy": _generation_status["active"],
            "segment_idx": _generation_status["segment_idx"],
            "queued": sorted(_generation_status["queued"]),
            "queue_order": list(_generation_status["queue_order"]),
        })

@app.route("/api/segment/select_take", methods=["POST"])
def select_take():
    data = request.json or {}
    seg_idx = int(data.get("seg_idx", -1))
    take_idx = int(data.get("take_idx", -1))

    state = load_project_state()
    if not state or seg_idx < 0 or seg_idx >= len(state["segments"]):
        return jsonify({"error": "Invalid segment index"}), 400

    seg = state["segments"][seg_idx]
    if not seg.get("takes"):
        return jsonify({"error": "Segment has no generated takes yet"}), 400
    if take_idx < 0 or take_idx >= len(seg["takes"]):
        return jsonify({"error": "Invalid take index"}), 400

    seg["selected_take"] = take_idx

    # Yeni seçilen take'e göre öbek haritasını güncelle
    try:
        import re
        project_id = state.get("id")
        seg_dir = WORK_DIR / f"project_{project_id}" / f"seg_{seg_idx:03d}"
        take_paths = list(seg_dir.glob("take_*.wav"))
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
                    
        if len(valid_paths) > 0:
            stitcher = MODELS["stitcher"]
            seg["chunks"] = stitcher.build_segment_chunks(seg, valid_paths, alignments)
    except Exception as ex:
        log.warning(f"Error rebuilding chunks on select_take for segment {seg_idx}: {ex}")

    state["stitched"] = False
    save_project_state(state)
    return jsonify(state)

@app.route("/api/segment/override_chunk", methods=["POST"])
def override_chunk():
    data = request.json or {}
    seg_idx = int(data.get("seg_idx", -1))
    chunk_idx = int(data.get("chunk_idx", -1))
    take_idx = data.get("take_idx") # can be int or None

    state = load_project_state()
    if not state or seg_idx < 0 or seg_idx >= len(state["segments"]):
        return jsonify({"error": "Invalid segment index"}), 400

    seg = state["segments"][seg_idx]
    if "chunks" not in seg or chunk_idx < 0 or chunk_idx >= len(seg["chunks"]):
        return jsonify({"error": "Invalid chunk index or chunks not generated"}), 400

    chunk = seg["chunks"][chunk_idx]
    
    if take_idx is not None and int(take_idx) >= 0:
        take_idx = int(take_idx)
        if not seg.get("takes") or take_idx >= len(seg["takes"]):
            return jsonify({"error": "Invalid take index"}), 400
        chunk["override_take"] = take_idx
        chunk["active_take"] = take_idx
    else:
        chunk["override_take"] = None
        chunk["active_take"] = chunk["auto_take"]

    state["stitched"] = False
    save_project_state(state)
    return jsonify(state)

@app.route("/api/stitch", methods=["POST"])
def stitch_audio():
    state = load_project_state()
    if not state:
        return jsonify({"error": "Project not initialized"}), 400

    # Ensure all segments are completed
    for seg in state["segments"]:
        if seg["status"] != "completed":
            return jsonify({"error": f"Segment {seg['idx']} is not generated/completed yet."}), 400

    try:
        stitcher = MODELS["stitcher"]

        # Hız ayarı — istek gövdesinden al (varsayılan 1.0 = normal)
        data = request.get_json(silent=True) or {}
        output_speed = float(data.get("output_speed", 1.0))
        output_speed = max(0.5, min(2.0, output_speed))  # güvenli sınır
        stitcher.output_speed = output_speed
        if abs(output_speed - 1.0) > 0.005:
            log.info(f"Hız ayarı aktif: {output_speed:.2f}x")
        
        # Prepare segments for the stitcher.py
        # It expects list[list[dict]], where flat_segments represents the chosen takes
        all_segments_input = []
        for seg in state["segments"]:
            selected_take_idx = seg["selected_take"]
            take_data = seg["takes"][selected_take_idx]
            
            # Construct segment data matching what pipeline.py / scorer.py produces
            project_id = state.get("id")
            segment_item = {
                "take_idx": selected_take_idx,
                "audio_path": str(WORK_DIR / f"project_{project_id}" / f"seg_{seg['idx']:03d}" / f"take_{selected_take_idx}.wav"),
                "start": 0.0,
                "end": take_data["duration"],
                "text": seg["text"],
                "pause_after": seg["pause_after"],
                "intonation_trend": seg.get("intonation_trend", "stable"),
                "vowel_stretching": seg.get("vowel_stretching", "normal"),
                "stress_anchor": seg.get("stress_anchor", ""),
                "chunks": seg.get("chunks", []) # Öbek overrides bilgilerini stitcher'a paslıyoruz!
            }
            all_segments_input.append([segment_item])

        project_id = state.get("id")
        final_out_path = str(WORK_DIR / f"project_{project_id}" / "final_output.wav")
        stitcher.stitch(all_segments_input, final_out_path)

        state["stitched"] = True
        state["final_audio_path"] = final_out_path
        state["final_audio_url"] = f"/api/audio?path={final_out_path}"
        
        save_project_state(state)
        return jsonify(state)

    except Exception as e:
        log.exception("Error stitching segments")
        return jsonify({"error": f"Stitching failed: {str(e)}"}), 500

@app.route("/api/project/export", methods=["POST"])
def export_project():
    data = request.json or {}
    export_dir = data.get("export_dir", "").strip()
    if not export_dir:
        return jsonify({"error": "Export directory is required"}), 400

    state = load_project_state()
    if not state:
        return jsonify({"error": "Project not initialized"}), 400

    # Ensure all segments are completed
    for seg in state["segments"]:
        if seg["status"] != "completed":
            return jsonify({"error": f"Segment {seg['idx']} is not generated/completed yet."}), 400

    try:
        import shutil
        import re
        import soundfile as sf
        from pathlib import Path
        
        stitcher = MODELS["stitcher"]
        dest_path = Path(export_dir)
        dest_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Prepare segment parameters
        project_id = state.get("id")
        all_segments_input = []
        for seg in state["segments"]:
            selected_take_idx = seg["selected_take"]
            take_data = seg["takes"][selected_take_idx]
            
            segment_item = {
                "take_idx": selected_take_idx,
                "audio_path": str(WORK_DIR / f"project_{project_id}" / f"seg_{seg['idx']:03d}" / f"take_{selected_take_idx}.wav"),
                "start": 0.0,
                "end": take_data["duration"],
                "text": seg["text"],
                "pause_after": seg["pause_after"],
                "intonation_trend": seg.get("intonation_trend", "stable"),
                "vowel_stretching": seg.get("vowel_stretching", "normal"),
                "stress_anchor": seg.get("stress_anchor", ""),
                "chunks": seg.get("chunks", [])
            }
            all_segments_input.append([segment_item])

        # 2. Stitch the full version first to ensure final_output.wav is up to date
        project_id = state.get("id")
        final_out_path = str(WORK_DIR / f"project_{project_id}" / "final_output.wav")
        stitcher.stitch(all_segments_input, final_out_path)
        state["stitched"] = True
        state["final_audio_path"] = final_out_path
        state["final_audio_url"] = f"/api/audio?path={final_out_path}"
        save_project_state(state)

        # 3. Copy full version to target folder
        proj_name = state.get("name", "").strip()
        if not proj_name:
            proj_name = "full_narration"
        
        # Sanitize project name
        clean_proj_name = re.sub(r"[^\w\- ]", "", proj_name).strip().replace(" ", "_")
        full_dest_file = dest_path / f"{clean_proj_name}_full.wav"
        shutil.copy2(final_out_path, str(full_dest_file))

        # 4. Process and export each segment individually
        flat_segments = []
        for chunk_segs in all_segments_input:
            flat_segments.extend(chunk_segs)

        stitcher._sr = stitcher._detect_sample_rate(flat_segments[0]["audio_path"])

        for i, seg_item in enumerate(flat_segments):
            # Form clean text name snippet
            clean_text = re.sub(r"[^\w\- ]", "", seg_item["text"]).strip().lower().replace(" ", "_")
            words = [w for w in clean_text.split("_") if w][:5]
            short_text = "_".join(words)
            if not short_text:
                short_text = f"segment_{i+1:02d}"
                
            seg_file_name = f"{i+1:02d}_{short_text}.wav"
            seg_dest_file = dest_path / seg_file_name
            
            # Compile / Load the segment audio
            seg_audio = None
            try:
                audio_path = Path(seg_item["audio_path"])
                seg_dir = audio_path.parent
                import json
                take_paths = list(seg_dir.glob("take_*.wav"))
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
                
                if len(valid_paths) > 1 and len(alignments) == len(valid_paths):
                    # Hybrid stitch
                    seg_audio = stitcher._stitch_hybrid_segment(seg_item, valid_paths, alignments)
                    if seg_audio is not None:
                        seg_audio = stitcher._gentle_trim(seg_audio)
            except Exception as e:
                log.warning(f"Failed to stitch segment {i+1} hybridly for export: {e}")
                seg_audio = None

            if seg_audio is None:
                # Classic fallback loading
                seg_audio = stitcher._load_full_audio(seg_item["audio_path"])
                seg_audio = stitcher._gentle_trim(seg_audio)
                seg_audio = stitcher._normalize_segment(seg_audio)

            # Post-process segment audio (remove DC offset, peak-normalize)
            seg_audio = seg_audio - np.mean(seg_audio)
            peak = np.max(np.abs(seg_audio))
            if peak > 0.01:
                seg_audio = seg_audio * (0.9 / peak)
            
            sf.write(str(seg_dest_file), seg_audio, stitcher._sr)

        log.info(f"Project successfully exported to: {dest_path}")
        return jsonify({
            "success": True, 
            "export_dir": str(dest_path),
            "full_file": str(full_dest_file)
        })

    except Exception as e:
        log.exception("Error exporting project")
        return jsonify({"error": f"Export failed: {str(e)}"}), 500

@app.route("/api/audio", methods=["GET"])
def serve_audio():
    path_param = request.args.get("path", "")
    if not path_param:
        return jsonify({"error": "Missing path parameter"}), 400

    # Safety check: Resolve paths and check if it's within the pipeline_work directory
    target_path = Path(path_param).resolve()
    base_path = WORK_DIR.resolve()

    # FIX: str.startswith() symlink bypass'e açıktı — is_relative_to() ile güvenli kontrol
    try:
        target_path.relative_to(base_path)
    except ValueError:
        return jsonify({"error": "Unauthorized path access"}), 403

    if not target_path.exists():
        return jsonify({"error": "Audio file not found"}), 404

    return send_file(str(target_path), mimetype="audio/wav")

def denoise_reference_audio(file_path):
    """Applies high-pass filter to remove low rumble and gates silent parts to prevent background hum learning."""
    try:
        import soundfile as sf
        import numpy as np
        from scipy.signal import butter, lfilter
        
        data, samplerate = sf.read(str(file_path))
        
        # Convert to mono if stereo
        if len(data.shape) > 1:
            data_mono = data.mean(axis=1)
        else:
            data_mono = data
            
        # 1. Apply High-pass filter at 85Hz to cut off power line hum and low room rumble
        nyquist = 0.5 * samplerate
        cutoff = 85.0
        normal_cutoff = cutoff / nyquist
        b, a = butter(5, normal_cutoff, btype='high', analog=False)
        filtered_data = lfilter(b, a, data_mono)
        
        # 2. Apply a noise gate on rolling average amplitude to silence background static during pauses
        # 50ms window size
        window_size = int(samplerate * 0.05)
        amplitude_envelope = np.convolve(np.abs(filtered_data), np.ones(window_size)/window_size, mode='same')
        
        # Silence sections where average envelope is below 0.003 (about -50dB)
        gate = amplitude_envelope > 0.003
        gated_data = filtered_data * gate
        
        # Save clean mono version back to the file
        sf.write(str(file_path), gated_data, samplerate)
        log.info(f"Denoised uploaded reference audio: high-pass filtered (>85Hz) and noise-gated silent sections.")
    except Exception as e:
        log.error(f"Error while denoising reference audio: {e}")

@app.route("/api/project/upload_ref", methods=["POST"])
def upload_reference():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # FIX: MIME type ve uzantı kontrolü — sadece .wav ve .mp3 kabul et
    filename_lower = (file.filename or "").lower()
    if not (filename_lower.endswith(".wav") or filename_lower.endswith(".mp3")):
        return jsonify({"error": "Only .wav and .mp3 files are accepted as reference audio"}), 400

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    file_path = WORK_DIR / "uploaded_ref.wav"

    if filename_lower.endswith(".mp3"):
        temp_mp3_path = WORK_DIR / "temp_uploaded_ref.mp3"
        file.save(str(temp_mp3_path))
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(str(temp_mp3_path))
            audio.export(str(file_path), format="wav")
            if temp_mp3_path.exists():
                temp_mp3_path.unlink()
        except Exception as e:
            log.error(f"Error converting MP3 to WAV using pydub: {e}")
            try:
                import librosa
                import soundfile as sf
                y, sr = librosa.load(str(temp_mp3_path), sr=None)
                sf.write(str(file_path), y, sr)
                if temp_mp3_path.exists():
                    temp_mp3_path.unlink()
            except Exception as e2:
                log.error(f"Fallback conversion error: {e2}")
                if temp_mp3_path.exists():
                    temp_mp3_path.unlink()
                return jsonify({"error": f"Failed to convert MP3 to WAV: {str(e2)}"}), 500
    else:
        file.save(str(file_path))

    # Denoise reference audio before cloning to prevent hum/static replication
    denoise_reference_audio(file_path)

    log.info(f"Reference audio uploaded and saved to: {file_path}")
    return jsonify({
        "filePath": str(file_path),
        "fileName": file.filename
    })

# ═══════════════════════════════════════════════════════
#  TIMBRE REGISTRY API
# ═══════════════════════════════════════════════════════

_timbre_store = TimbreStore()

@app.route("/api/timbre/list", methods=["GET"])
def list_timbres():
    """Kaydedilmiş tüm ses kimliklerini listele."""
    return jsonify({"timbres": _timbre_store.list_timbres()})

@app.route("/api/timbre/save", methods=["POST"])
def save_timbre():
    """Yeni bir ses kimliği kaydet veya güncelle."""
    data = request.json or {}
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Timbre name is required"}), 400
    try:
        entry = _timbre_store.save_timbre(name, data)
        return jsonify({"saved": True, "timbre": entry})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/timbre/delete", methods=["POST"])
def delete_timbre():
    """Bir ses kimliğini sil."""
    data = request.json or {}
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Timbre name is required"}), 400
    deleted = _timbre_store.delete_timbre(name)
    return jsonify({"deleted": deleted})

@app.route("/api/timbre/finetune/prepare", methods=["GET"])
def prepare_finetune():
    voice_name = request.args.get("name", "").strip()
    if not voice_name:
        return jsonify({"error": "Timbre name is required"}), 400
        
    timbre = _timbre_store.load_timbre(voice_name)
    if not timbre:
        return jsonify({"error": f"Voice profile '{voice_name}' not found"}), 404
        
    ref_audio = timbre.get("ref_audio", "")
    ref_text = timbre.get("ref_text", "")
    if not ref_audio or not os.path.exists(ref_audio):
        return jsonify({"error": "Reference audio file not found"}), 400

    try:
        from aligner import WordAligner
        import soundfile as sf
        
        # Run Whisper word-level alignment
        aligner = WordAligner(whisper_model="base")
        alignment = aligner.align(ref_audio, ref_text)
        words = alignment.get("words", [])
        
        if not words:
            return jsonify({"error": "No words detected in reference audio"}), 400
            
        # Group words into segments of 4-8 seconds
        word_groups = []
        current_group = []
        current_dur = 0.0
        
        for i, w in enumerate(words):
            current_group.append(w)
            start_t = current_group[0]["start"]
            end_t = w["end"]
            current_dur = end_t - start_t
            
            should_split = False
            if current_dur >= 4.0:
                is_strong_punct = w.get("ending_punct") in [".", "?", "!", ";", "..."]
                has_large_gap = False
                if i < len(words) - 1:
                    gap = words[i+1]["start"] - w["end"]
                    if gap > 0.25:
                        has_large_gap = True
                if is_strong_punct or has_large_gap:
                    should_split = True
            if current_dur >= 6.0:
                should_split = True
                
            if should_split or i == len(words) - 1:
                word_groups.append(current_group)
                current_group = []
                current_dur = 0.0

        segments = []
        seg_idx = 1
        info = sf.info(ref_audio)
        samplerate = info.samplerate
        
        for g_idx, group in enumerate(word_groups):
            start_time = group[0]["start"]
            end_time = group[-1]["end"]
            
            # Default margins: 100ms padding at start, 150ms padding at end
            seg_start = start_time - 0.10
            seg_end = end_time + 0.15
            
            # Prevent bleed from previous segment
            if g_idx > 0:
                prev_end = word_groups[g_idx - 1][-1]["end"]
                midpoint = (prev_end + start_time) / 2
                if start_time > prev_end:
                    # Gap case: cap start at the midpoint of the gap
                    seg_start = max(seg_start, midpoint)
                else:
                    # Overlap case: split the difference with a 50ms overlap margin
                    seg_start = midpoint - 0.05
                
            # Prevent bleed into next segment
            if g_idx < len(word_groups) - 1:
                next_start = word_groups[g_idx + 1][0]["start"]
                midpoint = (end_time + next_start) / 2
                if next_start > end_time:
                    # Gap case: cap end at the midpoint of the gap
                    seg_end = min(seg_end, midpoint)
                else:
                    # Overlap case: split the difference with a 50ms overlap margin
                    seg_end = midpoint + 0.05
                
            seg_start = max(0.0, seg_start)
            seg_end = min(info.duration, seg_end)
            
            start_frame = int(seg_start * samplerate)
            end_frame = int(seg_end * samplerate)
            
            data, _ = sf.read(ref_audio, start=start_frame, stop=end_frame)
            
            seg_filename = f"finetune_seg_{voice_name}_{seg_idx}.wav"
            seg_path = WORK_DIR / seg_filename
            sf.write(str(seg_path), data, samplerate)
            
            confidences = [x.get("confidence", 1.0) for x in group]
            avg_conf = sum(confidences) / len(confidences) if confidences else 1.0
            
            status = "Green" if avg_conf >= 0.85 else "Yellow"
            
            segments.append({
                "id": seg_idx,
                "audio_path": str(seg_path),
                "audio_url": f"/api/audio?path={str(seg_path)}",
                "text": " ".join([x["word"] for x in group]),
                "start": round(seg_start, 2),
                "end": round(seg_end, 2),
                "confidence": round(avg_conf * 100, 1),
                "status": status
            })
            
            seg_idx += 1
                
        import json
        prep_path = WORK_DIR / f"{voice_name}_prepared_segments.json"
        with open(prep_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2, ensure_ascii=False)
            
        return jsonify({"segments": segments})
        
    except Exception as e:
        log.exception("Error preparing fine-tuning data")
        return jsonify({"error": f"Failed to prepare data: {str(e)}"}), 500

@app.route("/api/timbre/finetune/save", methods=["POST"])
def save_finetune_segments():
    try:
        import json
        data = request.json or {}
        voice_name = data.get("name", "").strip()
        segments = data.get("segments", [])
        
        if not voice_name:
            return jsonify({"error": "Voice name is required"}), 400
            
        prep_path = WORK_DIR / f"{voice_name}_prepared_segments.json"
        
        existing_segments = []
        if prep_path.exists():
            with open(prep_path, "r", encoding="utf-8") as f:
                existing_segments = json.load(f)
                
        existing_map = {s["id"]: s["audio_path"] for s in existing_segments}
        for s in segments:
            s["audio_path"] = existing_map.get(s["id"], "")
            s["audio_url"] = f"/api/audio?path={s['audio_path']}"
            
        with open(prep_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=2, ensure_ascii=False)
            
        return jsonify({"success": True})
        
    except Exception as e:
        log.exception("Error saving segments")
        return jsonify({"error": str(e)}), 500

@app.route("/api/timbre/finetune/stream")
def stream_finetune():
    voice_name = request.args.get("name", "").strip()
    if not voice_name:
        return jsonify({"error": "Timbre name is required"}), 400
        
    timbre = _timbre_store.load_timbre(voice_name)
    if not timbre:
        return jsonify({"error": f"Voice profile '{voice_name}' not found"}), 404
        
    ref_audio = timbre.get("ref_audio", "")
    ref_text = timbre.get("ref_text", "")
    
    # Read model from query args, fallback to mapping stored timbre model_size
    model_id = request.args.get("model", "").strip()
    if not model_id:
        db_model_size = timbre.get("model_size", "1.7B")
        model_id = f"mlx-community/Qwen3-TTS-12Hz-{db_model_size}-Base-bf16"
    
    lr = float(request.args.get("lr", 1e-5))
    r = int(request.args.get("r", 16))
    alpha = int(request.args.get("alpha", 16))
    steps = int(request.args.get("steps", 120))

    from finetune import run_finetune_generator
    
    def generate():
        for line in run_finetune_generator(voice_name, ref_audio, ref_text, model_id, lr=lr, r=r, alpha=alpha, steps=steps):
            yield f"data: {line}\n\n"
            
    return Response(generate(), mimetype="text/event-stream")

@app.route("/api/models/status", methods=["GET"])
def get_models_status():
    """Retrieve cache status for all Tuti registered models."""
    results = []
    for model in hf_downloader.MODELS_LIST:
        is_cached = hf_downloader.is_model_cached(model["id"])
        results.append({
            **model,
            "cached": is_cached
        })
    return jsonify({"models": results})

@app.route("/api/models/download/stream", methods=["GET"])
def stream_model_download():
    """Stream model download progress via SSE."""
    repo_id = request.args.get("repo_id", "").strip()
    if not repo_id:
        return jsonify({"error": "repo_id is required"}), 400

    # Ensure the repo_id is in our allowed models list to prevent downloading arbitrary repos
    allowed_repos = {m["id"] for m in hf_downloader.MODELS_LIST}
    if repo_id not in allowed_repos:
        return jsonify({"error": f"Repository '{repo_id}' is not in the allowed models registry."}), 403

    def generate():
        for event in hf_downloader.download_model_generator(repo_id):
            yield f"data: {json.dumps(event)}\n\n"

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5050)))
    args, unknown = parser.parse_known_args()

    # IMPORTANT: threaded=True artık güvenli — MLX üretimi dedicated worker thread
    # içinde çalışır (_generation_worker), Flask request thread'leri MLX'e hiç
    # dokunmaz; sadece JSON + ses dosyası I/O yapar. Bu sayede üretim sürerken
    # frontend ses dosyalarını eş zamanlı dinleyebilir.
    # Eski threaded=False ayarı üretim sırasında play tuşunun sessiz kalmasına
    # neden oluyordu (tek thread → audio isteği üretim bitene kadar kuyruktaydı).
    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False, threaded=True)
