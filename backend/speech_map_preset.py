import re

class PresetSpeechMapPlan:
    def __init__(self):
        pass

    def create_sft_segment_plan(self, text: str, tone: str, seg_index: int = 0, total_segments: int = 1) -> dict:
        is_opening = seg_index == 0
        is_closing = seg_index == total_segments - 1

        # AKTÖR YÖNETMENİ TALİMATLARI
        # YouTuber akıcılığı ve kararlılığı odaklı basit acting cue'lar.
        # SFT için default stress 0.70'tir. (YouTube Set)
        if text.endswith("?"):
            tts_instruct = "Asking a question with natural curiosity and a rising intonation at the end."
            intonation_trend = "rising"
            pause_after = 0.55
            stress = 0.70
        elif text.endswith("!"):
            tts_instruct = "Energetic and lively YouTuber delivery with positive enthusiasm."
            intonation_trend = "falling"
            pause_after = 0.65
            stress = 0.70
        elif is_opening:
            tts_instruct = "Confident and engaging YouTuber opener, direct and clear."
            intonation_trend = "stable"
            pause_after = 0.25
            stress = 0.70
        elif is_closing:
            tts_instruct = "Warm and friendly YouTuber wrap-up, ending with confidence."
            intonation_trend = "falling"
            pause_after = 0.40
            stress = 0.70
        else:
            tts_instruct = "Clear, natural, and steady YouTuber narration with fluent forward momentum."
            intonation_trend = "stable"
            pause_after = 0.25
            stress = 0.70

        # Ton ayarı — Modele göre genel ruh halini ekle
        speed = 1.05  # YouTuber temel hızı (kullanıcı isteği: 1.05)
        if tone == "energetic":
            speed = 1.10
            stress = min(stress + 0.05, 0.85)
            tts_instruct += " High energy, speaking with a smile."
        elif tone == "calm":
            speed = 1.00
            stress = max(stress - 0.10, 0.60)
            tts_instruct += " Relaxed and calm, but still moving forward."
        elif tone == "serious":
            speed = 1.00
            stress = min(stress + 0.05, 0.80)
            tts_instruct += " Deliberate and authoritative, measured delivery."
        elif tone == "warm":
            speed = 1.05
            stress = min(stress + 0.05, 0.80)
            tts_instruct += " Friendly, warm smile audible in the voice."

        # Temperature: SFT için 0.40 en uygun değerdir.
        base_temp = 0.40
        if tone == "energetic":
            base_temp = 0.40  # SFT modunda temp'in 0.40 kalması istenir.
        elif tone in ("calm", "serious"):
            base_temp = 0.40

        # Pitch: SFT modeli pitch'e aşırı duyarlı — kesinlikle 0.00 kalmalı!
        pitch = 0.0

        # SFT Modunda Thinking devre dışı
        thinking_enabled = False
        vowel_stretching = "normal"

        words_clean = [re.sub(r"[^\w']", "", w) for w in text.split()]
        words_clean = [w for w in words_clean if w]
        stress_anchor = max(words_clean, key=len) if words_clean else ""

        return {
            "text": text,
            "emotion": "natural",
            "stress": round(stress, 2),
            "speed": round(speed, 2),
            "pause_after": round(pause_after, 2),
            "pitch": round(pitch, 2),
            "thinking_enabled": thinking_enabled,
            "temperature": round(base_temp, 2),
            "tts_instruct": tts_instruct.strip(),
            "intonation_trend": intonation_trend,
            "vowel_stretching": vowel_stretching,
            "stress_anchor": stress_anchor,
            "lora_strength": 0.0,  # Default to 0.0 to use stable target embedding injection and bypass corrupted LoRA perturbations
        }

    def create_preset_segment_plan(self, text: str, use_case: str, tone: str, seg_index: int = 0, total_segments: int = 1) -> dict:
        emotion = "natural"
        is_opening = seg_index == 0
        is_closing = seg_index == total_segments - 1

        if text.endswith("?"):
            tts_instruct = "Asking a question with natural curiosity and a rising intonation at the end."
            intonation_trend = "rising"
            pause_after = 0.60
            stress = 0.70
            emotion = "curious"
        elif text.endswith("!"):
            tts_instruct = "Energetic and lively YouTuber delivery with positive enthusiasm."
            intonation_trend = "falling"
            pause_after = 0.80
            stress = 0.90
            emotion = "excited"
        elif is_opening:
            tts_instruct = "Confident and engaging YouTuber opener, direct and clear."
            intonation_trend = "stable"
            pause_after = 0.50
            stress = 0.80
            emotion = "welcoming"
        elif is_closing:
            tts_instruct = "Warm and friendly YouTuber wrap-up, ending with confidence."
            intonation_trend = "falling"
            pause_after = 0.80
            stress = 0.65
            emotion = "natural"
        else:
            tts_instruct = "Clear, natural, and steady YouTuber narration with fluent forward momentum."
            intonation_trend = "stable"
            pause_after = 0.50
            stress = 0.70
            emotion = "natural"

        speed = 1.0
        if tone == "energetic":
            speed = min(speed * 1.05, 1.2)
            stress = min(stress + 0.15, 1.0)
            tts_instruct += " High energy and extra liveliness."
        elif tone == "calm":
            speed = max(speed * 0.9, 0.8)
            stress = max(stress - 0.15, 0.4)
            pause_after = max(pause_after + 0.2, 0.4)
            tts_instruct += " Calm, measured narration at a steady pace."
        elif tone == "warm":
            stress = min(stress + 0.05, 0.9)
            tts_instruct += " Warm and sincere narration."
        elif tone == "serious":
            speed = max(speed * 0.95, 0.85)
            stress = min(stress + 0.1, 0.95)
            tts_instruct += " Clear, direct, and deliberate narration."

        pitch = 0.0
        if text.endswith("?"):
            pitch = 0.50
        elif text.endswith("!"):
            pitch = 0.80
        elif any(kw in text.lower() for kw in ["reveal", "secret", "incredible", "never", "shout", "power"]):
            pitch = 0.60
        elif any(kw in text.lower() for kw in ["grave", "deep", "serious", "authority", "sad", "dark"]):
            pitch = -0.60

        word_count = len(text.split())
        has_digits = any(c.isdigit() for c in text)
        has_math = any(c in text for c in ["+", "=", "-", "*", "/", "%", "$", "#", "@", "_"])
        has_complex_punctuation = text.count(",") + text.count(";") >= 2
        is_long = word_count > 20
        thinking_enabled = has_digits or has_math or has_complex_punctuation or is_long

        if tone == "energetic" or text.endswith("!"):
            temperature = 0.80
        elif tone == "serious" or tone == "calm":
            temperature = 0.50
        else:
            temperature = 0.65

        vowel_stretching = "normal"
        if tone == "energetic":
            vowel_stretching = "high"
        elif tone == "calm":
            vowel_stretching = "low"
 
        words_clean = [re.sub(r"[^\w']", "", w) for w in text.split()]
        words_clean = [w for w in words_clean if w]
        stress_anchor = max(words_clean, key=len) if words_clean else ""
 
        return {
            "text": text,
            "emotion": emotion,
            "stress": round(stress, 2),
            "speed": round(speed, 2),
            "pause_after": round(pause_after, 2),
            "pitch": round(pitch, 2),
            "thinking_enabled": thinking_enabled,
            "temperature": round(temperature, 2),
            "tts_instruct": tts_instruct.strip(),
            "intonation_trend": intonation_trend,
            "vowel_stretching": vowel_stretching,
            "stress_anchor": stress_anchor
        }
