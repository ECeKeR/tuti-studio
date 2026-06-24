import re

class CloneSpeechMapPlan:
    def __init__(self):
        pass

    def create_clone_segment_plan(self, text: str, use_case: str, tone: str, seg_index: int = 0, total_segments: int = 1) -> dict:
        emotion = "natural"
        is_opening = seg_index == 0
        is_closing = seg_index == total_segments - 1

        # v3: KISA, duygu odaklı yönlendirmeler (9-12 kelime)
        if text.endswith("?"):
            tts_instruct = "Speak with genuine curiosity and rising intrigue, as if truly puzzled."
            intonation_trend = "rising"
            pause_after = 0.60
            stress = 0.70
            emotion = "curious"
        elif text.endswith("!"):
            tts_instruct = "Speak with energetic excitement, fast and punchy, like revealing a trick."
            intonation_trend = "falling"
            pause_after = 0.80
            stress = 0.90
            emotion = "excited"
        elif is_opening:
            tts_instruct = "Speak with bright curiosity and a teasing hook. Lean in, pull the viewer in."
            intonation_trend = "stable"
            pause_after = 0.50
            stress = 0.80
            emotion = "welcoming"
        elif is_closing:
            tts_instruct = "Speak with warm satisfaction, a grin audible in your voice. Land it."
            intonation_trend = "falling"
            pause_after = 0.80
            stress = 0.65
            emotion = "natural"
        else:
            tts_instruct = "Speak in an energetic, engaging YouTuber voiceover. Crisp, forward momentum."
            intonation_trend = "stable"
            pause_after = 0.50
            stress = 0.70
            emotion = "natural"

        speed = 1.0
        if tone == "energetic":
            speed = min(speed * 1.05, 1.2)
            stress = min(stress + 0.15, 1.0)
            tts_instruct += " Speak with high energy and bright excitement."
        elif tone == "calm":
            speed = max(speed * 0.9, 0.8)
            stress = max(stress - 0.15, 0.4)
            pause_after = max(pause_after + 0.2, 0.4)
            tts_instruct = "Speak calmly and softly, with gentle and relaxed pacing."
        elif tone == "warm":
            stress = min(stress + 0.05, 0.9)
            tts_instruct = "Speak with warm sincerity, a smile audible in your voice."
        elif tone == "serious":
            speed = max(speed * 0.95, 0.85)
            stress = min(stress + 0.1, 0.95)
            tts_instruct = "Speak with serious authority and deliberate, firm weight."

        pitch = 0.0
        if text.endswith("?"):
            pitch = 0.50
        elif text.endswith("!"):
            pitch = 0.80
        elif any(kw in text.lower() for kw in ["reveal", "secret", "incredible", "never", "shout", "power"]):
            pitch = 0.60
        elif any(kw in text.lower() for kw in ["grave", "deep", "serious", "authority", "somber", "dark"]): # sad yerine somber
            pitch = -0.60

        word_count = len(text.split())
        has_digits = any(c.isdigit() for c in text)
        has_math = any(c in text for c in ["+", "=", "-", "*", "/", "%", "$", "#", "@", "_"])
        has_complex_punctuation = text.count(",") + text.count(";") >= 2
        is_long = word_count > 20
        thinking_enabled = has_digits or has_math or has_complex_punctuation or is_long

        if tone == "energetic" or text.endswith("!"):
            temperature = 0.70
        elif tone == "serious" or tone == "calm":
            temperature = 0.58
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