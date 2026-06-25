# Copyright (c) 2026 Efeberk Çeker
# SPDX-License-Identifier: AGPL-3.0-only

"""
Speech Map Generator - LLM veya Rule-Based ile prosody planı oluşturur.
Supports Ollama (Qwen3:8b model) backend.
"""
import json
import logging
from pathlib import Path
import re

from speech_map_preset import PresetSpeechMapPlan
from speech_map_clone import CloneSpeechMapPlan

log = logging.getLogger(__name__)

class SpeechMapGenerator:
    def __init__(self, llm_backend: str = "rule_based", model_name: str = "qwen3:8b"):
        self.llm_backend = llm_backend
        self.model_name = model_name
        self.preset_planner = PresetSpeechMapPlan()
        self.clone_planner = CloneSpeechMapPlan()

    def generate(
        self,
        text: str,
        use_case: str = "youtube_narration",
        tone: str = "natural",
        character_profile: str = "",
        cache_path: str | None = None,
        is_sft: bool = False,
        voice_mode: str = "preset",
    ) -> dict:
        text = text.replace("—", "...").replace("--", "...")

        if cache_path and Path(cache_path).exists():
            try:
                with open(cache_path) as f:
                    log.info(f"Speech map önbellekten yüklendi: {cache_path}")
                    return json.load(f)
            except Exception as e:
                log.warning(f"Önbellek okunamadı: {e}")

        if self.llm_backend == "ollama":
            ollama_map = self._generate_via_ollama(text, use_case, tone, character_profile, is_sft=is_sft, voice_mode=voice_mode)
            if ollama_map:
                if cache_path:
                    try:
                        with open(cache_path, "w") as f:
                            json.dump(ollama_map, f, indent=2)
                    except Exception as e:
                        log.warning(f"Speech map kaydedilemedi: {e}")
                return ollama_map

        log.info("Kural tabanlı Speech Map üretiliyor...")
        segments = self._split_text(text)
        segments_plan = []

        for i, seg_text in enumerate(segments):
            if is_sft:
                plan = self.preset_planner.create_sft_segment_plan(seg_text, tone, seg_index=i, total_segments=len(segments))
            else:
                plan = self.clone_planner.create_clone_segment_plan(seg_text, use_case, tone, seg_index=i, total_segments=len(segments))
            segments_plan.append(plan)

        speech_map = {
            "overall": {
                "use_case": use_case,
                "tone": tone,
                "emotion": self._get_overall_emotion(tone),
            },
            "segments": segments_plan
        }

        if cache_path:
            try:
                with open(cache_path, "w") as f:
                    json.dump(speech_map, f, indent=2)
            except Exception as e:
                log.warning(f"Speech map kaydedilemedi: {e}")

        return speech_map

    def _generate_via_ollama(self, text: str, use_case: str, tone: str, character_profile: str = "", is_sft: bool = False, voice_mode: str = "preset") -> dict | None:
        import requests
        url = "http://localhost:11434/api/chat"

        if is_sft:
            sft_note = "- SFT MODE: Keep tts_instruct to 2-4 words describing general tone (e.g. \"Confident, clear narration.\")."
            clone_note = ""
        elif voice_mode == "clone":
            sft_note = ""
            clone_note = "- CLONE MODE: Describe style in 3-5 words (e.g., \"Warm, gentle conversational voice.\", \"Excited, energetic delivery.\"). Avoid words like \"sad\", \"tearful\", \"crying\"."
        else:
            sft_note = ""
            clone_note = ""

        prompt = f"""You are a YouTube TTS Voice Director.
Task: Split the input script into natural segments of 70-80 words (max 85 words).
For each segment, output:
- "text": The segment text (keep original words, do not rewrite).
- "tts_instruct": A very short emotion/tone prompt (2-4 words max). Examples: "Excited, energetic delivery.", "Somber, reflective tone."
- "pause_after": Pause after segment (0.15 for comma, 0.30 for period, 0.50 for topic change).
- "emotion", "stress" (0.60-0.90), "speed" (1.00-1.15), "temperature" (0.50-0.70), "pitch" (0.0), "lora_strength" (0.75), "intonation_trend" ("stable"/"rising"/"falling"), "vowel_stretching" ("normal"/"high"/"low"), "stress_anchor" (one key word), "thinking_enabled" (false).
{sft_note}
{clone_note}
Return ONLY valid JSON (no markdown fences).

Format:
{{
  "overall": {{"emotion": "excited"}},
  "segments": [
    {{
      "text": "First segment of 70-80 words here...",
      "emotion": "excited",
      "stress": 0.80,
      "speed": 1.08,
      "pause_after": 0.30,
      "pitch": 0.00,
      "thinking_enabled": false,
      "temperature": 0.60,
      "tts_instruct": "Excited, rising delivery.",
      "intonation_trend": "stable",
      "vowel_stretching": "normal",
      "stress_anchor": "word",
      "lora_strength": 0.75
    }}
  ]
}}

INPUT SCRIPT: "{text}"
Tone: "{tone}" | Use case: "{use_case}" | Character: "{character_profile}"
"""
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": True,
            "options": {
                "temperature": 0.2,
                "num_ctx": 8192,
                "num_predict": 4096
            }
        }
        
        try:
            log.info(f"Speech Map için Ollama ({self.model_name}) çağrılıyor (streaming)...")
            response = requests.post(url, json=payload, stream=True, timeout=(10, 120))
            if response.status_code == 200:
                content_parts = []
                for raw_line in response.iter_lines():
                    if not raw_line:
                        continue
                    try:
                        chunk = json.loads(raw_line)
                        delta = chunk.get("message", {}).get("content", "")
                        content_parts.append(delta)
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
                content = "".join(content_parts).strip()
                if content.startswith("```"):
                    content = re.sub(r"^```(?:json)?\n", "", content)
                    content = re.sub(r"\n```$", "", content)
                content = content.strip()
                parsed = json.loads(content)
                if "segments" in parsed and isinstance(parsed["segments"], list):
                    for seg in parsed["segments"]:
                        # V10 FIX: LLM "sad" dönerse "somber" yap, steady rhythm ekle
                        if "emotion" in seg and seg["emotion"].lower() in ("sad", "tearful", "crying"):
                            seg["emotion"] = "somber"
                        if "tts_instruct" in seg:
                            inst = seg["tts_instruct"]
                            inst = re.sub(r'\bsad\b', 'somber', inst, flags=re.IGNORECASE)
                            seg["tts_instruct"] = re.sub(r'\btearful\b', 'melancholic', inst, flags=re.IGNORECASE)
                            
                        if "emotion" not in seg: seg["emotion"] = "natural"
                        if "stress" not in seg: seg["stress"] = 0.6
                        else:
                            try: seg["stress"] = round(float(seg["stress"]), 2)
                            except: seg["stress"] = 0.6
                        if "speed" not in seg: seg["speed"] = 1.08
                        else:
                            try: seg["speed"] = round(float(seg["speed"]), 2)
                            except: seg["speed"] = 1.08
                        if "pause_after" not in seg: seg["pause_after"] = 0.3
                        else:
                            try: seg["pause_after"] = round(float(seg["pause_after"]), 2)
                            except: seg["pause_after"] = 0.3
                        if "pitch" not in seg: seg["pitch"] = 0.0
                        else:
                            try: seg["pitch"] = round(float(seg["pitch"]), 2)
                            except: seg["pitch"] = 0.0
                        if "thinking_enabled" not in seg: seg["thinking_enabled"] = False
                        else: seg["thinking_enabled"] = bool(seg["thinking_enabled"])
                        if "temperature" not in seg: seg["temperature"] = 0.6
                        else:
                            try: seg["temperature"] = round(float(seg["temperature"]), 2)
                            except: seg["temperature"] = 0.6
                        if "tts_instruct" not in seg: seg["tts_instruct"] = "Speak naturally."
                        if "intonation_trend" not in seg: seg["intonation_trend"] = "stable"
                        if "vowel_stretching" not in seg: seg["vowel_stretching"] = "normal"
                        if "stress_anchor" not in seg: seg["stress_anchor"] = ""
                    log.info(f"Ollama ({self.model_name}) ile Speech Map başarıyla üretildi.")
                    return parsed
            else:
                log.warning(f"Ollama returned HTTP {response.status_code}")
        except Exception as e:
            log.warning(f"Ollama ({self.model_name}) Speech Map üretimi başarısız oldu ({e}). Kural tabanlı fallback devreye giriyor.")
        return None

    def _split_text(self, text: str) -> list[str]:
        sentences = re.split(r'(?<=[.?!])\s+', text.strip())
        raw_segments = []
        
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            words = s.split()
            if len(words) <= 55:
                raw_segments.append(s)
            else:
                sub_segs = re.split(r'(?<=[,;:])\s+', s)
                merged = []
                buffer = ""
                for ss in sub_segs:
                    ss = ss.strip()
                    if not ss:
                        continue
                    if buffer:
                        candidate = buffer + " " + ss
                    else:
                        candidate = ss
                    if len(candidate.split()) < 15:
                        buffer = candidate
                    else:
                        merged.append(candidate)
                        buffer = ""
                if buffer:
                    if merged:
                        merged[-1] = merged[-1] + " " + buffer
                    else:
                        merged.append(buffer)
                raw_segments.extend(merged)
                
        if not raw_segments:
            return raw_segments
            
        final = [raw_segments[0]]
        for i in range(1, len(raw_segments)):
            seg = raw_segments[i]
            if len(seg.split()) <= 10 and len(final[-1].split()) < 55:
                final[-1] = final[-1] + " " + seg
            else:
                final.append(seg)
                
        return final

    def _create_segment_plan(self, text: str, use_case: str, tone: str, seg_index: int = 0, total_segments: int = 1) -> dict:
        emotion = "natural"
        stress = 0.6
        speed = 1.08
        pause_after = 0.3

        text_lower = text.lower()
        word_count = len(text.split())
        
        is_opening = seg_index == 0
        is_closing = seg_index == total_segments - 1
        position_ratio = seg_index / max(total_segments - 1, 1)
        
        # Kısa ve Aktör Odaklı Talimatlar
        if text.endswith("?"):
            emotion = "curious"
            stress = 0.75
            pause_after = 0.6
            tts_instruct = "Curious, rising tone."
        elif text.endswith("!"):
            emotion = "excited"
            stress = 0.9
            pause_after = 0.8
            tts_instruct = "Energetic, excited delivery."
        elif "welcome" in text_lower or "today we" in text_lower or "in this video" in text_lower:
            emotion = "welcoming"
            stress = 0.8
            speed = 0.95
            pause_after = 0.5
            tts_instruct = "Confident, welcoming host."
        elif any(kw in text_lower for kw in ["today", "reveal", "secret", "framework", "step", "guarantee"]):
            emotion = "confident"
            stress = 0.8
            speed = 0.95
            pause_after = 0.5
            tts_instruct = "Confident, authoritative tone."
        elif text.endswith("."):
            pause_after = 0.5
            if is_opening:
                stress = 0.75
                tts_instruct = "Confident, engaging opener."
            elif is_closing:
                stress = 0.65
                speed = 0.95
                pause_after = 0.8
                tts_instruct = "Warm, concluding delivery."
            elif position_ratio < 0.4:
                tts_instruct = "Natural narrative flow."
            else:
                stress = 0.7
                tts_instruct = "Steady, clear delivery."
        elif text.endswith(",") or text.endswith(";"):
            pause_after = 0.15
            tts_instruct = "Natural continuation."
        else:
            if is_opening:
                tts_instruct = "Energetic, engaging narration."
            elif is_closing:
                tts_instruct = "Natural, warm wrap-up."
            else:
                tts_instruct = "Natural, clear voice."

        # Ton ayarlamaları
        if tone == "energetic":
            speed = min(speed * 1.05, 1.2)
            stress = min(stress + 0.15, 1.0)
            tts_instruct = "High energy delivery."
        elif tone == "calm":
            speed = max(speed * 0.9, 0.8)
            stress = max(stress - 0.15, 0.4)
            pause_after = max(pause_after + 0.2, 0.4)
            tts_instruct = "Calm, relaxed pacing."
        elif tone == "warm":
            stress = min(stress + 0.05, 0.9)
            tts_instruct = "Warm, friendly delivery."
        elif tone == "serious":
            speed = max(speed * 0.95, 0.85)
            stress = min(stress + 0.1, 0.95)
            tts_instruct = "Serious, authoritative weight."

        # Pitch ve Diğer Parametreler
        pitch = 0.0
        if text.endswith("?"):
            pitch = 0.50
        elif text.endswith("!"):
            pitch = 0.80
        elif any(kw in text_lower for kw in ["reveal", "secret", "incredible", "never", "shout", "power"]):
            pitch = 0.60
        elif any(kw in text_lower for kw in ["grave", "deep", "serious", "authority", "somber", "dark"]):
            pitch = -0.60

        has_digits = any(c.isdigit() for c in text)
        has_math = any(c in text for c in ["+", "=", "-", "*", "/", "%", "$", "#", "@", "_"])
        has_complex_punctuation = text.count(",") + text.count(";") >= 2
        is_long = word_count > 20
        thinking_enabled = has_digits or has_math or has_complex_punctuation or is_long

        if tone == "energetic" or text.endswith("!"):
            temperature = 0.75  # Düzeltme: 0.80 -> 0.75
        elif tone == "serious" or tone == "calm":
            temperature = 0.55
        else:
            temperature = 0.65

        intonation_trend = "stable"
        if text.endswith("?"):
            intonation_trend = "rising"
        elif text.endswith("!"):
            intonation_trend = "falling"
 
        vowel_stretching = "normal"
        if tone == "energetic":
            vowel_stretching = "high"
        elif tone == "calm":
            vowel_stretching = "low"
 
        # KRİTİK DÜZELTME: Kelimeleri ayıkla ve stress_anchor'u belirle
        words_clean = [re.sub(r"[^\w']", "", w) for w in text.split()]
        words_clean = [w for w in words_clean if w]
        stress_anchor = max(words_clean, key=len) if words_clean else ""
        
        # KRİTİK DÜZELTME: Eğer metinde ALL CAPS yoksa, stress_anchor kelimesini büyüt!
        # Bu, modelin tekdüze (monoton) okumasını engeller, fonetik vurgu ekler.
        if stress_anchor and not any(w.isupper() for w in text.split()):
            # Stress anchor 4 harften uzunsa, anlamlı bir vurgu olur
            if len(stress_anchor) > 3:
                text = re.sub(r'\b' + re.escape(stress_anchor) + r'\b', stress_anchor.upper(), text, count=1, flags=re.IGNORECASE)
 
        return {
            "text": text,
            "emotion": emotion,
            "stress": round(stress, 2),
            "speed": round(speed, 2),
            "pause_after": round(pause_after, 2),
            "pitch": round(pitch, 2),
            "thinking_enabled": thinking_enabled,
            "temperature": round(temperature, 2),
            "tts_instruct": tts_instruct,
            "intonation_trend": intonation_trend,
            "vowel_stretching": vowel_stretching,
            "stress_anchor": stress_anchor
        }

    def _get_overall_emotion(self, tone: str) -> str:
        if tone == "energetic":
            return "excited"
        elif tone == "calm":
            return "peaceful"
        elif tone == "warm":
            return "friendly"
        elif tone == "serious":
            return "formal"
        return "natural"