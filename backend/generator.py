# Copyright (c) 2026 Efeberk Çeker
# SPDX-License-Identifier: AGPL-3.0-only

"""
Qwen3-TTS Generator Facade
Delegates presets to PresetTTSGenerator and voice clones to CloneTTSGenerator.
"""
import os
import logging
from pathlib import Path
import numpy as np
import soundfile as sf

from generator_preset import PresetTTSGenerator
from generator_clone import CloneTTSGenerator

log = logging.getLogger(__name__)

class TTSGenerator:
    def __init__(
        self, 
        backend: str = "pytorch",
        model_size: str = "1.7B", 
        speaker: str = "Ethan",
        ref_audio: str | None = None,
        ref_text: str | None = None,
        voice_mode: str = "preset",
        design_prompt: str | None = None,
        clone_alpha: float = 1.0,
        clone_topk: int = 3,
        clone_speaker: str = "ryan",
    ):
        self.backend = backend.lower()
        self.model_size = model_size
        self.speaker = speaker
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.voice_mode = voice_mode
        self.design_prompt = design_prompt
        self.clone_alpha = float(clone_alpha)
        self.clone_topk  = int(clone_topk)
        self.clone_speaker = clone_speaker

        self._sft_adapter_loaded = False
        
        self.is_sft_preset = False
        if self.voice_mode == "preset" and self.speaker:
            adapter_path = f"pipeline_work/{self.speaker}_adapter.safetensors"
            if os.path.exists(adapter_path) and os.path.getsize(adapter_path) > 0:
                self.is_sft_preset = True

        if self.voice_mode == "preset":
            self.generator = PresetTTSGenerator(
                model_size=self.model_size,
                speaker=self.speaker,
                backend=self.backend
            )
            self._sft_adapter_loaded = self.generator._sft_adapter_loaded
        else:
            self.generator = CloneTTSGenerator(
                model_size=self.model_size,
                backend=self.backend,
                voice_mode=self.voice_mode,
                ref_audio=self.ref_audio,
                ref_text=self.ref_text,
                design_prompt=self.design_prompt,
                clone_alpha=self.clone_alpha,
                clone_topk=self.clone_topk,
                clone_speaker=self.clone_speaker,
            )

    def generate(
        self,
        text: str,
        output_path: str,
        instruct: str = "Speak naturally with energy and forward momentum, like a confident person talking directly to someone.",
        language: str = "English",
        seed: int | None = None,
        temperature: float = 0.6,
    ) -> str:
        res_path = self.generator.generate(text, output_path, instruct, language, seed, temperature)
        if os.path.exists(res_path):
            size = os.path.getsize(res_path)
            if size <= 44:
                raise ValueError(f"Model generated empty or silent audio. Output file size is {size} bytes.")
        else:
            raise ValueError(f"Output audio file not found after generation at {res_path}")
        return res_path

    def generate_n_takes(
        self,
        text: str,
        output_dir: str,
        n: int = 3,
        speech_map_segment: dict | None = None,
        language: str = "English",
        context_prefix: str | None = None,
        on_take_callback = None,
    ) -> list[str]:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        base_instruct = "Casual, natural delivery — speaking with a slight smile, close to the microphone, forward momentum, no unnecessary pauses."
        if speech_map_segment:
            base_instruct = speech_map_segment.get("tts_instruct", base_instruct)
  
        instruct_variants = self._build_variants(base_instruct, n, speech_map_segment)
        temp_schedule = self._build_temp_schedule(n, speech_map_segment)
  
        final_text = self._prepare_text_for_base_model(text, speech_map_segment, self.voice_mode)

        if context_prefix and context_prefix.strip():
            prep_context = self._prepare_text_for_base_model(context_prefix, None, self.voice_mode)
            full_text = f"{prep_context.strip()}... {final_text.strip()}"
            log.info(f"  Context padding active: '{context_prefix[:30]}...' + main text")
        else:
            full_text = final_text
 
        paths = []
        locked_seed = speech_map_segment.get("locked_seed") if speech_map_segment else None
        if locked_seed is not None:
            try:
                base_seed = int(locked_seed)
            except:
                base_seed = 42
        else:
            base_seed = 42

        if speech_map_segment and hasattr(self.generator, "update_lora_strength"):
            lora_strength = float(speech_map_segment.get("lora_strength", 0.75))
            self.generator.update_lora_strength(lora_strength)

        _SEED_SPREAD = [42, 1337, 9999, 7777, 31337]

        for i in range(n):
            path = str(Path(output_dir) / f"take_{i}.wav")
            if speech_map_segment and speech_map_segment.get("locked_seed") is not None:
                take_seed = base_seed + i
            else:
                take_seed = _SEED_SPREAD[i % len(_SEED_SPREAD)]
            take_temp = temp_schedule[i]
            log.info(f"  Take {i+1}/{n}: seed={take_seed}, temp={take_temp}, '{instruct_variants[i][:60]}'")
            self.generate(
                full_text, path,
                instruct=instruct_variants[i],
                language=language,
                seed=take_seed,
                temperature=take_temp,
            )
            paths.append(path)
            if on_take_callback:
                on_take_callback(i, path)
 
        return paths

    def _build_temp_schedule(self, n: int, segment: dict | None = None) -> list[float]:
        user_temp = segment.get("temperature", 0.6) if segment else 0.6
        if n == 1:
            return [user_temp]

        sft_loaded = self._sft_adapter_loaded
        is_clone_flow = self.voice_mode == "clone"
        emotion = segment.get("emotion", "neutral") if segment else "neutral"
        is_excited = emotion in ("excited", "happy", "amused")

        # V3 KAZANAN BANT (youtube_test_script_v3 A_hookB_split2):
        # Dar ve daha yüksek sıcaklık → duyguyu korur, monotonluğu kırmaz.
        # Çok geniş spread (0.55-0.85) modeli "talimat takibi" moduna sokuyordu.
        if is_clone_flow or sft_loaded:
            if is_excited:
                lo = max(0.62, user_temp - 0.05)
                hi = min(0.72, user_temp + 0.10)
            else:
                lo = max(0.58, user_temp - 0.05)
                hi = min(0.68, user_temp + 0.10)
        else:
            if is_excited:
                lo = max(0.65, user_temp - 0.05)
                hi = min(0.75, user_temp + 0.10)
            else:
                lo = max(0.60, user_temp - 0.05)
                hi = min(0.72, user_temp + 0.10)

        temps = []
        for i in range(n):
            t = lo + ((hi - lo) * i / (n - 1))
            temps.append(round(t, 2))
        return temps

    def _build_variants(self, base_instruct: str, n: int, segment: dict | None) -> list[str]:
        import re

        # V10 DÜZELTMESİ: "Sad" ve "tearful" kelimeleri modelin enerjisini düşürüp
        # susmasına (early EOS) sebep oluyor. Bu koruma kalsın.
        emotion = segment.get("emotion", "neutral") if segment else "neutral"
        if emotion in ("sad", "melancholic"):
            base_instruct = re.sub(r'\bsad\b', 'somber and melancholic', base_instruct, flags=re.IGNORECASE)
            base_instruct = re.sub(r'\btearful\b', 'melancholic', base_instruct, flags=re.IGNORECASE)
            base_instruct = re.sub(r'\bcrying\b', 'melancholic', base_instruct, flags=re.IGNORECASE)

        # V3 KAZANAN STRATEJİ (A_hookB_split2): variant prefix YOK.
        # "Deliver in a snappy, fast-paced YouTuber voiceover..." gibi bürokratik
        # prefix'ler modeli emotion'a değil "talimatlara uymaya" yönlendiriyordu →
        # robotikleşme + monotonluk. Aynı KISA emotion instruction tüm take'lerde
        # kullanılır; çeşitlilik temperature spread + farklı seed'den gelir.
        # (v10 felsefesi: kısa, net, doğrudan duygu.)
        return [base_instruct.strip()] * n

    def _prepare_text_for_base_model(self, text: str, segment: dict | None, voice_mode: str | None = None) -> str:
        if not text:
            return text
        
        text = text.replace("[breath]", "...")
        text = text.replace("[sigh]", "...")
        text = text.replace("[gasp]", "...")
        text = text.replace("[laughter]", "...")

        if segment:
            intonation_trend = segment.get("intonation_trend", "stable")

            if voice_mode in ("clone", "design"):
                if intonation_trend == "rising" and not text.endswith("?"):
                    text = text.rstrip(".!?, ") + "?"
                elif intonation_trend == "falling" and not text.endswith("!"):
                    text = text.rstrip(".!?, ") + "!"

        return text