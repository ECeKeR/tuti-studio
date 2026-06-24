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
            sft_note = """

SFT MODEL NOTE (parameters will be enforced by code — focus on TEXT MARKUP):
- Numeric params (temperature=0.40, speed=1.05, stress=0.70, pitch=0.00) are locked by the system.
- Your only creative job is the "text" field and "tts_instruct".
- PROSODIC MARKUP IS STILL REQUIRED IN SFT MODE:
  * Use ALL CAPS for 1-3 emphasized words per segment: "We COULD do it all at ONCE."
  * Use "..." for dramatic pauses before reveals: "One reason is... context."
  * Use "?!" for excited rhetorical questions: "But WHY?!"
  * Always use '...' for hesitations or mid-sentence pauses instead of em dashes (—) to maintain stable audio generation.
- DO NOT use [laughter], [gasp] or other paralinguistic tags.
- RESONANCE LOOP AVOIDANCE: rephrase any repeated phrase of 3+ words (second occurrence only).
"""
            clone_note = ""
        elif voice_mode == "clone":
            sft_note = ""
            clone_note = """

CLONE MODE NOTE (The target voice is cloned from reference audio):
- Qwen3-TTS instruction tags for cloned voices should specify descriptive style attributes rather than generic director notes.
- In "tts_instruct", describe the voice-acting qualities in detail:
  * Specify delivery style and mood (e.g. "Deliver with a sarcastic, assertive tone", "Speak in a warm, gentle voice", "with an energetic and excited expression").
  * Specify pacing and articulation (e.g. "crisp enunciation, steady conversational pace", "slow tempo, deliberate pauses").
  * Specify volume/intensity and pitch characteristics if appropriate (e.g. "controlled volume", "low pitch", "bright timbre").
- Format the instruction as a descriptive prompt, for example:
  "Speak in a sarcastic, assertive tone: crisp enunciation, controlled volume, conveying authority."
  "Deliver with extreme sadness and a slight cry: slow pace, low volume, trembling voice."
  "Bright and very happy delivery: crisp articulation, energetic pace, speaking with a smile."
- EMOTIONAL STABILITY RULE: NEVER use words like "sad", "tearful", or "crying" in tts_instruct, as they cause the model to cut off audio. Instead use "somber", "melancholic", or "reflective". ALWAYS add "Keep normal volume, clear pronunciation, and do not pause too long between sentences. Maintain a steady rhythm." to the instruction.
"""
        else:
            sft_note = ""
            clone_note = ""

        prompt = f"""You are a YouTube TTS Voice Director. Your task has TWO parts:

PART 1 — SPLIT THE SCRIPT
Split the input script into natural segments of roughly 40–50 words each, splitting at dramatic transition points (new topic, emotion shift, question→answer). Use commas, semicolons, and periods as your primary splitting points. Do NOT break natural phrasal verbs, idioms, or complete thoughts in half just to hit a word count. Maintain sentence integrity.
CRITICAL: Never exceed 50 words per segment — longer segments cause the TTS model to loop and hallucinate audio.

PART 2 — REWRITE EACH SEGMENT TEXT WITH PROSODIC MARKUP
You MUST transform the "text" field. Do NOT return the original words unchanged.
The TTS model reads the "text" exactly as written — the markup IS the vocal performance.

MANDATORY TRANSFORMATION RULES:
1. OPTIONAL: You MAY emphasize 1 key word per segment in ALL CAPS, but ONLY if it strengthens the delivery. Most segments should use natural capitalization. Do NOT force CAPS on every segment — it makes the delivery monotonous and robotic.
2. Add "..." before dramatic reveals or pauses mid-sentence.
3. Use "?!" for excited rhetorical questions instead of just "?".
4. Keep all other words exactly as in the original.

BEFORE / AFTER EXAMPLES (study these carefully):
  BEFORE: "How do you do that? Glad you asked."
  AFTER:  "How do YOU do that? Glad you ASKED."

  BEFORE: "But it's too linear, too predictable, boring."
  AFTER:  "But it's too LINEAR, too predictable... BORING."

  BEFORE: "Instead you should write like this happens but this happens."
  AFTER:  "Instead... you should write like THIS happens, but THAT happens."

  BEFORE: "It's way more interesting to listen to."
  AFTER:  "It's WAY more interesting to listen to."

  BEFORE: "And these guys made 28 seasons so I think they know."
  AFTER:  "And these guys made 28 SEASONS... so I think they KNOW."

  BEFORE: "You know the knowledge at this point but you need a clear structure."
  AFTER:  "You KNOW the knowledge at this point... but you need a CLEAR structure."

RULES FOR TEXT FIELD:
- Do NOT add words not in the original script.
- Do NOT use [SLOW], [FAST] or any bracket tags.
- Capitalize only content words (nouns, verbs, adjectives) — not articles, prepositions.
- CAPS is optional: use it sparingly, at most once per segment, only where it genuinely helps emphasis.

PARAMETERS TO SET FOR EACH SEGMENT:
- emotion: the emotional feel of this segment (avoid "sad", use "somber" instead)
- stress: 0.70–0.90 (higher for more energetic segments)
- speed: 1.00–1.15
- pause_after: 0.15 (comma), 0.30 (period), 0.50 (topic shift)
- pitch: 0.00 always
- temperature: 0.58–0.70
- thinking_enabled: false
- intonation_trend: "rising" | "falling" | "stable"
- vowel_stretching: "high" | "normal" | "low"
- stress_anchor: the single most stressed word (use the ALL CAPS version, e.g. "BORING")
- tts_instruct: a SHORT emotion direction (10-15 words max). Describe the FEELING, not mechanics. Examples: "Speak with bright curiosity and a teasing hook." / "Speak softly and candidly, like confiding." / "Speak with energetic excitement, fast and punchy." NEVER use words like "low energy", "slow build", "sad", "tearful".
- lora_strength: 0.75
{sft_note}
{clone_note}
OUTPUT: Return ONLY valid JSON, no markdown fences, no explanation.

EXAMPLE OUTPUT FORMAT:
{{
  "overall": {{"emotion": "excited"}},
  "segments": [
    {{
      "text": "How do YOU do that? Glad you ASKED.",
      "emotion": "excited",
      "stress": 0.80,
      "speed": 1.10,
      "pause_after": 0.30,
      "pitch": 0.00,
      "thinking_enabled": false,
      "temperature": 0.50,
      "tts_instruct": "Punchy and direct, rising on YOU, land hard on ASKED. Maintain a steady rhythm.",
      "intonation_trend": "rising",
      "vowel_stretching": "normal",
      "stress_anchor": "ASKED",
      "lora_strength": 0.75
    }},
    {{
      "text": "But it's too LINEAR, too predictable... BORING.",
      "emotion": "critical",
      "stress": 0.85,
      "speed": 1.05,
      "pause_after": 0.40,
      "pitch": 0.00,
      "thinking_enabled": false,
      "temperature": 0.50,
      "tts_instruct": "Sarcastic drop on LINEAR, dramatic pause before BORING. Keep normal volume and clear pronunciation.",
      "intonation_trend": "falling",
      "vowel_stretching": "low",
      "stress_anchor": "BORING",
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
                            if "steady rhythm" not in inst.lower():
                                inst += " Maintain a steady rhythm and clear pronunciation."
                            seg["tts_instruct"] = re.sub(r'\bsad\b', 'somber', inst, flags=re.IGNORECASE)
                            seg["tts_instruct"] = re.sub(r'\btearful\b', 'melancholic', seg["tts_instruct"], flags=re.IGNORECASE)
                            
                        if "emotion" not in seg: seg["emotion"] = "natural"
                        if "stress" not in seg: seg["stress"] = 0.6
                        else:
                            try: seg["stress"] = round(float(seg["stress"]), 2)
                            except: seg["stress"] = 0.6
                        if "speed" not in seg: seg["speed"] = 1.0
                        else:
                            try: seg["speed"] = round(float(seg["speed"]), 2)
                            except: seg["speed"] = 1.0
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
            if len(words) <= 25:
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
                    if len(candidate.split()) < 5:
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
            if len(seg.split()) <= 3 and len(final[-1].split()) < 25:
                final[-1] = final[-1] + " " + seg
            else:
                final.append(seg)
                
        return final

    def _create_segment_plan(self, text: str, use_case: str, tone: str, seg_index: int = 0, total_segments: int = 1) -> dict:
        emotion = "natural"
        stress = 0.6
        speed = 1.0
        pause_after = 0.3

        text_lower = text.lower()
        word_count = len(text.split())
        
        is_opening = seg_index == 0
        is_closing = seg_index == total_segments - 1
        position_ratio = seg_index / max(total_segments - 1, 1)
        
        # Kısa ve Aktörl Odaklı Talimatlar
        if text.endswith("?"):
            emotion = "curious"
            stress = 0.75
            pause_after = 0.6
            tts_instruct = "Speak with genuine curiosity, raising your pitch naturally towards the end."
        elif text.endswith("!"):
            emotion = "excited"
            stress = 0.9
            pause_after = 0.8
            tts_instruct = "Deliver with a burst of excitement. Emphasize the most impactful words naturally."
        elif "welcome" in text_lower or "today we" in text_lower or "in this video" in text_lower:
            emotion = "welcoming"
            stress = 0.8
            speed = 0.95
            pause_after = 0.5
            tts_instruct = "Speak like a confident, friendly host welcoming viewers. Vary your pitch naturally."
        elif any(kw in text_lower for kw in ["today", "reveal", "secret", "framework", "step", "guarantee"]):
            emotion = "confident"
            stress = 0.8
            speed = 0.95
            pause_after = 0.5
            tts_instruct = "Speak with confident authority. Slow down slightly on the key promise words."
        elif text.endswith("."):
            pause_after = 0.5
            if is_opening:
                stress = 0.75
                tts_instruct = "Open with confidence and energy, like the start of an engaging story."
            elif is_closing:
                stress = 0.65
                speed = 0.95
                pause_after = 0.8
                tts_instruct = "Wrap up with a sense of resolution and warmth. Let your pitch drop naturally."
            elif position_ratio < 0.4:
                tts_instruct = "Continue the narrative flow with natural energy. Vary your pitch to keep it interesting."
            else:
                stress = 0.7
                tts_instruct = "Build towards the conclusion with steady, clear delivery. Emphasize key concepts."
        elif text.endswith(",") or text.endswith(";"):
            pause_after = 0.15
            tts_instruct = "Continue naturally as if you're in the middle of a thought. Keep the momentum flowing smoothly."
        else:
            if is_opening:
                tts_instruct = "Start with energy and presence, as if beginning an exciting conversation."
            elif is_closing:
                tts_instruct = "Finish with a natural, warm tone that signals wrapping up."
            else:
                tts_instruct = "Speak naturally and clearly with varied pitch and pacing."

        # Ton ayarlamaları
        if tone == "energetic":
            speed = min(speed * 1.05, 1.2)
            stress = min(stress + 0.15, 1.0)
            tts_instruct += " High energy, lively and expressive delivery."
        elif tone == "calm":
            speed = max(speed * 0.9, 0.8)
            stress = max(stress - 0.15, 0.4)
            pause_after = max(pause_after + 0.2, 0.4)
            tts_instruct += " Calm, measured pace with a soothing, relaxed voice."
        elif tone == "warm":
            stress = min(stress + 0.05, 0.9)
            tts_instruct += " Add genuine warmth and friendliness to your voice."
        elif tone == "serious":
            speed = max(speed * 0.95, 0.85)
            stress = min(stress + 0.1, 0.95)
            tts_instruct += " Speak with authority and gravitas, steady and deliberate."

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