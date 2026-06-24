import os
import logging
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

log = logging.getLogger(__name__)

class PresetTTSGenerator:
    def __init__(self, model_size="1.7B", speaker="Ethan", backend="mlx", lora_strength=0.85, disable_hack=False):
        self.model_size = model_size
        self.speaker = speaker
        self.backend = backend
        self.lora_strength = lora_strength
        self.disable_hack = disable_hack
        self.model = None
        self._sft_adapter_loaded = False
        self._load_model()

    def _load_model(self):
        if self.backend == "mlx":
            self._load_mlx_model()
        else:
            self._load_pytorch_model()

    def _load_mlx_model(self):
        from mlx_audio.tts import load
        mlx_model_map = {
            "1.7B": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
            "0.6B": "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16",
        }
        model_id = mlx_model_map.get(self.model_size, mlx_model_map["1.7B"])
        log.info(f"[Preset MLX] Loading model: {model_id}")
        self.model = load(model_id)
        
        # Check if custom voice profile (SFT or Clone) is selected
        adapter_path = f"pipeline_work/{self.speaker}_adapter.safetensors"
        
        # Check database for reference audio
        ref_audio_path = None
        try:
            import sqlite3
            db_path = "pipeline_work/tuti.db"
            conn = sqlite3.connect(db_path, timeout=5.0)
            c = conn.cursor()
            c.execute("SELECT ref_audio FROM timbres WHERE LOWER(name) = ?", (self.speaker.lower().strip(),))
            row = c.fetchone()
            conn.close()
            if row and row[0] and os.path.exists(row[0]):
                ref_audio_path = row[0]
        except Exception as e:
            log.warning(f"Failed to query database for voice {self.speaker}: {e}")

        has_adapter = os.path.exists(adapter_path) and os.path.getsize(adapter_path) > 0
        has_ref_audio = ref_audio_path is not None

        if self.disable_hack:
            has_adapter = False
            has_ref_audio = False

        if has_adapter or has_ref_audio:
            log.info(f"Custom voice profile detected for '{self.speaker}'. (Adapter: {has_adapter}, Ref Audio: {has_ref_audio})")
            from production_logger import ProductionLogger
            ProductionLogger.log_step(
                "Custom Voice Profile Detected",
                f"Speaker: '{self.speaker}'\nAdapter: {has_adapter}\nRef Audio: {ref_audio_path}"
            )
            
            my_speaker_emb = None
            
            # If we have adapter, load LoRA and try to load speaker embedding from it
            if has_adapter:
                # Read rank and alpha dynamically from safetensors metadata
                r = 32
                alpha = 64.0
                try:
                    from safetensors import safe_open
                    with safe_open(adapter_path, framework="mlx") as f:
                        metadata = f.metadata()
                        if metadata:
                            r = int(metadata.get("rank", 32))
                            alpha = float(metadata.get("alpha", 64.0))
                    log.info(f"Loaded LoRA Config from metadata -> Rank (R): {r}, Alpha: {alpha}")
                    ProductionLogger.log_step("LoRA Config Extracted", f"Rank (R): {r}\nAlpha: {alpha}")
                except Exception as e:
                    log.warning(f"Could not read metadata from safetensors (falling back to R=32, Alpha=64.0): {e}")

                from finetune import apply_lora
                apply_lora(self.model, r=r, alpha=alpha)
                
                try:
                    import mlx.core as mx
                    flat_weights = mx.load(adapter_path)
                    my_speaker_emb = flat_weights.pop("my_speaker_embedding", None)
                    
                    # Load LoRA weights
                    total_layers = len(self.model.talker.model.layers)
                    num_lora_layers = 12
                    start_idx = max(0, total_layers - num_lora_layers)
                    
                    loaded = 0
                    for key, weight in flat_weights.items():
                        parts = key.split(".")
                        if len(parts) < 5 or parts[0] != "layers":
                            continue
                        layer_idx = int(parts[1])
                        proj_name = parts[3]
                        param_name = parts[4]
                        
                        if layer_idx < start_idx or layer_idx >= total_layers:
                            continue
                        if proj_name not in ("q_proj", "k_proj", "v_proj", "o_proj"):
                            continue
                        if param_name not in ("lora_a", "lora_b"):
                            continue
                            
                        proj = getattr(self.model.talker.model.layers[layer_idx].self_attn, proj_name)
                        if param_name == "lora_a":
                            proj.lora_A = weight
                        else:
                            proj.lora_B = weight
                        loaded += 1
                        
                    mx.eval(*[getattr(self.model.talker.model.layers[i].self_attn.q_proj, "lora_A")
                              for i in range(start_idx, total_layers)])
                    self._sft_adapter_loaded = True
                    log.info(f"SFT LoRA adapter weights loaded successfully onto the MLX model ({loaded} weights).")
                except Exception as e:
                    log.error(f"Failed to load SFT adapter weights using manual injection: {e}")

            # If speaker embedding not found in adapter, extract it dynamically from reference audio (segmented & averaged)
            if my_speaker_emb is None and has_ref_audio:
                log.info(f"Extracting speaker embedding dynamically from: {ref_audio_path} (segmented & averaged)")
                try:
                    import mlx.core as mx
                    from mlx_audio.utils import load_audio
                    import json
                    from finetune import segment_audio
                    
                    sr = self.model.sample_rate
                    audio_np = np.array(load_audio(ref_audio_path, sample_rate=sr))
                    
                    # Split reference audio into chunks of 3.0 to 8.0 seconds based on clean silences
                    audio_segments = segment_audio(audio_np, sr, min_sec=3.0, max_sec=8.0)
                    log.info(f"Split reference audio into {len(audio_segments)} silence-based segments for averaged embedding:")
                    for idx, seg in enumerate(audio_segments):
                        duration = len(seg) / sr
                        log.info(f"  Segment {idx+1}: (duration={duration:.2f}s)")
                        
                    log.info(f"Using {len(audio_segments)} silence-based segments for averaged embedding.")
                    
                    if hasattr(self.model, "speaker_encoder") and self.model.speaker_encoder is not None:
                        encoder_model = self.model
                    else:
                        log.info("Speaker encoder not present in CustomVoice model. Loading Base model temporarily for extraction...")
                        from mlx_audio.tts import load as load_base
                        base_size = "1.7B" if "1.7B" in self.model_size else "0.6B"
                        base_model_id = f"mlx-community/Qwen3-TTS-12Hz-{base_size}-Base-bf16"
                        encoder_model = load_base(base_model_id)
                    
                    embeddings = []
                    for seg in audio_segments:
                        seg_mx = mx.array(seg)
                        emb = encoder_model.extract_speaker_embedding(seg_mx, sr=sr)
                        embeddings.append(emb)
                    
                    if len(embeddings) > 0:
                        my_speaker_emb = mx.stack(embeddings).mean(axis=0)
                        log.info(f"Averaged {len(embeddings)} embeddings successfully.")
                    else:
                        raise ValueError("No segments generated from reference audio.")
                        
                    if not (hasattr(self.model, "speaker_encoder") and self.model.speaker_encoder is not None):
                        del encoder_model
                        mx.clear_cache()
                except Exception as emb_err:
                    log.warning(f"Failed to dynamically extract speaker embedding fallback: {emb_err}")

            if my_speaker_emb is not None:
                try:
                    import mlx.core as mx
                    emb_vector = my_speaker_emb.squeeze()
                    config = self.model.config.talker_config
                    emb_layer = self.model.talker.get_input_embeddings()
                    
                    # Override both 'serena' (female) and 'ryan' (male) dummy speaker tokens
                    for dummy_name in ["serena", "ryan", "ethan"]:
                        if dummy_name in config.spk_id:
                            spk_id = config.spk_id[dummy_name]
                            emb_layer.weight[spk_id] = emb_vector
                            log.info(f"🔥 SFT Model Hack: Token embedding weights for dummy speaker '{dummy_name}' replaced with target speaker embedding!")
                    
                    mx.eval(emb_layer.weight)  # Force evaluation on GPU (lazy eval prevention)
                    self._sft_adapter_loaded = True
                    
                    # Apply initial global lora_strength if LoRA was loaded
                    if has_adapter:
                        self.update_lora_strength(self.lora_strength)
                except Exception as override_err:
                    log.warning(f"Failed to override embedding weights: {override_err}")
            else:
                log.warning("Could not obtain speaker embedding. Generation might use default dummy speaker tics.")

    def update_lora_strength(self, strength: float):
        if not self._sft_adapter_loaded:
            return
        log.info(f"[Preset MLX] Dynamically updating SFT LoRA Strength to: {strength}")
        self.lora_strength = strength
        total_layers = len(self.model.talker.model.layers)
        num_layers = 12
        start_idx = max(0, total_layers - num_layers)
        for idx in range(start_idx, total_layers):
            layer = self.model.talker.model.layers[idx]
            for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                proj = getattr(layer.self_attn, proj_name)
                if hasattr(proj, "lora_strength"):
                    proj.lora_strength = strength

    def _load_pytorch_model(self):
        from qwen_tts import Qwen3TTSModel
        if torch.cuda.is_available():
            device = "cuda:0"
            dtype = torch.bfloat16
        elif torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32

        kwargs = {"device_map": device, "dtype": dtype}
        if torch.cuda.is_available():
            kwargs["attn_implementation"] = "flash_attention_2"

        model_id = f"Qwen/Qwen3-TTS-12Hz-{self.model_size}-CustomVoice"
        log.info(f"[Preset PyTorch] Loading model: {model_id} on {device}")
        try:
            self.model = Qwen3TTSModel.from_pretrained(model_id, **kwargs)
        except Exception as e:
            log.warning(f"Device map loading failed ({e}), falling back to standard...")
            kwargs.pop("device_map", None)
            self.model = Qwen3TTSModel.from_pretrained(model_id, **kwargs)
            try:
                self.model = self.model.to(device)
            except Exception as to_err:
                log.warning(f"Failed to move model to device: {to_err}")

    def generate(self, text: str, output_path: str, instruct: str, language: str, seed: int | None = None, temperature: float = 0.6) -> str:
        if self.backend == "mlx":
            return self._generate_mlx(text, output_path, instruct, language, seed, temperature)
        else:
            return self._generate_pytorch(text, output_path, instruct, language, seed, temperature)

    def _generate_mlx(self, text: str, output_path: str, instruct: str, language: str, seed: int | None = None, temperature: float = 0.6) -> str:
        import mlx.core as mx
        if seed is not None:
            np.random.seed(seed)
            mx.random.seed(seed)
        
        lang_code = "English" if language.lower() == "english" else language
        audio_chunks = []
        sr = 24000
        gen_kwargs = {"temperature": temperature}

        if self._sft_adapter_loaded:
            # Determine dummy speaker token base based on gender in instruct description
            dummy_speaker = "ryan"  # Default to male dummy speaker for male/general voices
            if instruct:
                ins_lower = instruct.lower()
                if any(w in ins_lower for w in ["female", "woman", "girl", "lady", "she", "her", "female voice", "woman's voice"]):
                    dummy_speaker = "serena"
            
            log.info(f"[SFT Preset MLX] Using dummy speaker token '{dummy_speaker}' for custom voice conditioning.")
            kwargs = {
                "language": lang_code,
                "instruct": instruct,
                "speaker": dummy_speaker,  # Dummy speaker to satisfy CustomVoice schema check
                **gen_kwargs,
            }
            log.info(f"[SFT Preset MLX] Generating with LoRA (CustomVoice mode), instruct: '{instruct[:60]}', seed: {seed}, temp: {temperature}")
            try:
                for result in self.model.generate_custom_voice(text, **kwargs):
                    audio_chunks.append(np.array(result.audio))
                    sr = result.sample_rate
            except Exception as e:
                log.error(f"[SFT Preset MLX] Failed to generate: {e}")
        else:
            tgt_speaker = self.speaker
            supported = []
            if hasattr(self.model, "get_supported_speakers"):
                try:
                    supported = self.model.get_supported_speakers()
                except:
                    pass
            if tgt_speaker and supported and tgt_speaker not in supported:
                fallback_speaker = "serena" if "serena" in supported else (supported[0] if len(supported) > 0 else "")
                log.info(f"Custom speaker '{self.speaker}' mapped to fallback '{fallback_speaker}'.")
                tgt_speaker = fallback_speaker

            log.info(f"[Preset MLX] Generating with CustomVoice: {self.speaker}, instruct: '{instruct[:60]}', seed: {seed}, temp: {temperature}")
            for result in self.model.generate_custom_voice(
                text=text,
                speaker=tgt_speaker,
                language=lang_code,
                instruct=instruct,
                **gen_kwargs,
            ):
                audio_chunks.append(np.array(result.audio))
                sr = result.sample_rate

        if audio_chunks:
            wav = np.concatenate(audio_chunks)
        else:
            wav = np.array([], dtype=np.float32)

        wav = np.asarray(wav, dtype=np.float32)
        max_val = np.max(np.abs(wav)) if len(wav) > 0 else 0
        if max_val > 1.0:
            wav = wav / max_val

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_p), wav, sr)

        try:
            import gc
            import mlx.core as mx
            gc.collect()
            mx.clear_cache()
            log.info("[Preset MLX] Cleared MLX cache and collected garbage.")
        except Exception as e:
            log.warning(f"[Preset MLX] Failed to clean memory cache: {e}")

        return output_path

    def _generate_pytorch(self, text: str, output_path: str, instruct: str, language: str, seed: int | None = None, temperature: float = 0.6) -> str:
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            
        lang_param = "English" if language.lower() == "english" else language
        try:
            wavs, sr = self.model.generate_custom_voice(
                text=text,
                language=lang_param,
                speaker=self.speaker,
                instruct=instruct,
                temperature=temperature
            )
        except TypeError:
            wavs, sr = self.model.generate_custom_voice(
                text=text,
                language=lang_param,
                speaker=self.speaker,
                instruct=instruct,
            )
        wav = wavs[0]
        wav = np.asarray(wav, dtype=np.float32)
        max_val = np.max(np.abs(wav)) if len(wav) > 0 else 0
        if max_val > 1.0:
            wav = wav / max_val

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_p), wav, sr)
        return output_path
