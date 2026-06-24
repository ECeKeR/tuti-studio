import os
import gc
import logging
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

log = logging.getLogger(__name__)


class CloneTTSGenerator:
    """
    Clone ve Design modlar için TTS üretici.

    MLX Clone modu — Speaker Space Interpolation (SLERP):
      SLERP ile vektörler küresel olarak kaynaştırılır (LERP yerine).
      alpha=1.0 (varsayılan) → tam ref embedding override.
      alpha=0.0 → saf base speaker token.
    """

    def __init__(
        self,
        model_size: str = "1.7B",
        backend: str = "mlx",
        voice_mode: str = "clone",
        ref_audio: str | None = None,
        ref_text: str | None = None,
        design_prompt: str | None = None,
        clone_alpha: float = 1.0,
        clone_topk: int = 3,
        clone_speaker: str = "ryan",
    ):
        self.model_size   = model_size
        self.backend      = backend
        self.voice_mode   = voice_mode
        self.ref_audio    = ref_audio
        self.ref_text     = ref_text
        self.design_prompt = design_prompt
        self.clone_alpha  = float(clone_alpha)
        self.clone_topk   = max(1, int(clone_topk))

        self.model        = None          # inference modeli (CustomVoice veya Base/Design)
        self.voice_prompt = None          # PyTorch clone prompt
        self._ref_centroid = None         # MLX: önceden hesaplanmış ref centroid
        self._target_speaker = clone_speaker or "ryan"  # MLX CustomVoice target speaker token

        self._load_model()

    # ── Model Yükleme ─────────────────────────────────────────────────────────

    def _load_model(self):
        if self.backend == "mlx":
            self._load_mlx_model()
        else:
            self._load_pytorch_model()

    def _load_mlx_model(self):
        from mlx_audio.tts import load as mlx_load

        if self.voice_mode == "design":
            mlx_model_map = {
                "1.7B": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
                "0.6B": "mlx-community/Qwen3-TTS-12Hz-0.6B-VoiceDesign-bf16",
            }
            model_id = mlx_model_map.get(self.model_size, mlx_model_map["1.7B"])
            log.info(f"[Design MLX] Loading model: {model_id}")
            self.model = mlx_load(model_id)
            return

        # ── Clone modu ────────────────────────────────────────────────────────
        ref_centroid = None
        if self.ref_audio and os.path.exists(self.ref_audio) and self.clone_alpha > 0.0:
            base_model_map = {
                "1.7B": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
                "0.6B": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
            }
            base_id = base_model_map.get(self.model_size, base_model_map["1.7B"])
            log.info(f"[Clone MLX] Loading BASE model for centroid extraction: {base_id}")
            try:
                base_model = mlx_load(base_id)
                ref_centroid, _ = self._build_ref_centroid(base_model)
                self._ref_centroid = ref_centroid
                log.info(f"[Clone MLX] Centroid extracted, norm={self._norm_val(ref_centroid):.4f}")
                
                del base_model
                self._mlx_cleanup()
            except Exception as e:
                log.error(f"[Clone MLX] BASE model loading/centroid extraction failed: {e}. Interpolation will be skipped.")
                self._ref_centroid = None
        else:
            self._ref_centroid = None

        custom_model_map = {
            "1.7B": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
            "0.6B": "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16",
        }
        custom_id = custom_model_map.get(self.model_size, custom_model_map["1.7B"])
        log.info(f"[Clone MLX] Loading CustomVoice inference model: {custom_id}")
        self.model = mlx_load(custom_id)

        if self._ref_centroid is not None:
            self._apply_speaker_interpolation_with_centroid(self._ref_centroid)
        elif self.clone_alpha == 0.0:
            log.info("[Clone MLX] alpha=0.0 — saf speaker token, interpolasyon yok.")
        else:
            log.warning(f"[Clone MLX] ref_audio bulunamadı ({self.ref_audio}) veya interpolasyon koşulları sağlanamadı, atlandı.")

    def _mlx_cleanup(self):
        gc.collect()
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass

    # ── Speaker Space Interpolation (V10 SLERP) ───────────────────────────────

    def _apply_speaker_interpolation_with_centroid(self, ref_centroid):
        try:
            speaker_id, speaker_vec = self._get_token_vec(self.model, self._target_speaker)
        except Exception as e:
            log.error(f"[Clone Interpolation] Speaker token alınamadı: {e}. Interpolasyon atlandı.")
            return

        log.info(
            f"[Clone Interpolation] speaker='{self._target_speaker}' id={speaker_id} "
            f"norm={self._norm_val(speaker_vec):.4f}  "
            f"cos(speaker,ref)={self._cosine(speaker_vec, ref_centroid):.5f}"
        )

        alpha = self.clone_alpha
        
        # V10 Testinden gelen SLERP matematiği
        mixed_vec = self._slerp(speaker_vec, ref_centroid, alpha)

        cos_ref = self._cosine(mixed_vec, ref_centroid)
        log.info(
            f"[Clone Interpolation] alpha={alpha:.2f} (SLERP)  "
            f"mixed_vec cos(ref)={cos_ref:.5f}  norm={self._norm_val(mixed_vec):.4f}"
        )

        patch_info = self._patch_token(self.model, self._target_speaker, mixed_vec)
        log.info(
            f"[Clone Interpolation] Patch: cos(orig vs new)={patch_info['cos_orig_vs_new']:.5f}  "
            f"final_norm={patch_info['final_norm']:.4f}"
        )

    # ── MLX Yardımcı Fonksiyonlar ─────────────────────────────────────────────

    def _norm_val(self, x):
        import mlx.core as mx
        return float(mx.linalg.norm(x))

    def _cosine(self, a, b):
        import mlx.core as mx
        a = a.astype(mx.float32).reshape(-1)
        b = b.astype(mx.float32).reshape(-1)
        return float(mx.sum(a * b) / (mx.linalg.norm(a) * mx.linalg.norm(b) + 1e-8))

    def _slerp(self, v1, v2, t, DOT_THRESHOLD=0.9995):
        """V10 Spherical Linear Interpolation"""
        import mlx.core as mx
        v1 = v1.astype(mx.float32).reshape(-1)
        v2 = v2.astype(mx.float32).reshape(-1)
        
        v1_norm = v1 / (mx.linalg.norm(v1) + 1e-8)
        v2_norm = v2 / (mx.linalg.norm(v2) + 1e-8)
        dot = mx.sum(v1_norm * v2_norm)
        
        if mx.abs(dot) > DOT_THRESHOLD:
            # Çok yakınsa LERP kullan (numerik stabilite)
            return (1.0 - t) * v1 + t * v2
            
        theta_0 = mx.arccos(dot)
        sin_theta_0 = mx.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = mx.sin(theta_t)
        
        s0 = mx.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        
        res = s0 * v1 + s1 * v2
        return res

    def _build_ref_centroid(self, base_model):
        import mlx.core as mx
        from mlx_audio.utils import load_audio
        from finetune import segment_audio

        sr = base_model.sample_rate
        audio = np.array(load_audio(str(self.ref_audio), sample_rate=sr))
        segments = segment_audio(audio, sr, min_sec=3.0, max_sec=8.0)

        if not segments:
            log.warning("[build_ref_centroid] 3s+ segment bulunamadı, tüm audio kullanılıyor.")
            segments = [audio]

        log.info(f"[build_ref_centroid] {len(segments)} segment bulundu.")
        embs = []
        for i, seg in enumerate(segments):
            emb = base_model.extract_speaker_embedding(mx.array(seg), sr=sr).squeeze()
            embs.append(emb)
            log.info(f"  Segment {i+1}: dur={len(seg)/sr:.2f}s  norm={self._norm_val(emb):.4f}")

        all_avg = mx.stack(embs).mean(axis=0).squeeze()

        scores = []
        for i, e1 in enumerate(embs):
            others = [self._cosine(e1, e2) for j, e2 in enumerate(embs) if j != i]
            avg_cos = float(np.mean(others)) if others else 1.0
            cos_avg = self._cosine(e1, all_avg)
            combined = avg_cos * 0.6 + cos_avg * 0.4
            scores.append({"idx": i, "combined": combined})

        scores.sort(key=lambda x: x["combined"], reverse=True)
        top_k = min(self.clone_topk, len(embs))
        selected = [embs[s["idx"]] for s in scores[:top_k]]
        centroid = mx.stack(selected).mean(axis=0).squeeze()

        log.info(
            f"[build_ref_centroid] Top-{top_k} centroid norm={self._norm_val(centroid):.4f}  "
            f"cos(all_avg, centroid)={self._cosine(all_avg, centroid):.5f}"
        )
        return centroid, all_avg

    def _get_token_vec(self, model, speaker: str):
        import mlx.core as mx
        config    = model.config.talker_config
        emb_layer = model.talker.get_input_embeddings()
        sid       = config.spk_id[speaker]
        vec       = mx.array(emb_layer.weight[sid]).squeeze()
        return int(sid), vec

    def _patch_token(self, model, speaker: str, new_vec):
        """Standart Norm Eşleştirme (Boost Yok - V10 Mantığı)"""
        import mlx.core as mx
        config    = model.config.talker_config
        emb_layer = model.talker.get_input_embeddings()
        sid       = config.spk_id[speaker]
        original  = mx.array(emb_layer.weight[sid]).squeeze()

        orig_norm = mx.linalg.norm(original)
        matched   = new_vec * (orig_norm / (mx.linalg.norm(new_vec) + 1e-8))

        emb_layer.weight[sid] = matched.reshape(original.shape).astype(original.dtype)
        mx.eval(emb_layer.weight)

        return {
            "speaker_id":      int(sid),
            "original_norm":   round(float(orig_norm), 4),
            "new_vec_norm":    round(float(mx.linalg.norm(new_vec)), 4),
            "final_norm":      round(float(mx.linalg.norm(matched)), 4),
            "cos_orig_vs_new": round(self._cosine(original, matched), 5),
        }

    # ── PyTorch Model Yükleme ─────────────────────────────────────────────────

    def _load_pytorch_model(self):
        from qwen_tts import Qwen3TTSModel
        if torch.cuda.is_available():
            device = "cuda:0"
            dtype  = torch.bfloat16
        elif torch.backends.mps.is_available():
            device = "mps"
            dtype  = torch.float16
        else:
            device = "cpu"
            dtype  = torch.float32

        kwargs = {"device_map": device, "dtype": dtype}
        if torch.cuda.is_available():
            kwargs["attn_implementation"] = "flash_attention_2"

        if self.ref_audio and os.path.exists(self.ref_audio):
            model_id = f"Qwen/Qwen3-TTS-12Hz-{self.model_size}-Base"
        else:
            model_id = f"Qwen/Qwen3-TTS-12Hz-{self.model_size}-CustomVoice"

        log.info(f"[Clone PyTorch] Loading model: {model_id} on {device}")
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

        if self.ref_audio and os.path.exists(self.ref_audio):
            log.info(f"[Clone PyTorch] Creating voice clone prompt: {self.ref_audio}")
            try:
                self.voice_prompt = self.model.create_voice_clone_prompt(
                    ref_audio=str(self.ref_audio),
                    ref_text=self.ref_text or "",
                    x_vector_only_mode=False,
                )
            except Exception as pe:
                log.error(f"[Clone PyTorch] Failed to create voice clone prompt: {pe}")

    # ── Üretim ────────────────────────────────────────────────────────────────

    def generate(
        self,
        text: str,
        output_path: str,
        instruct: str,
        language: str,
        seed: int | None = None,
        temperature: float = 0.6,
    ) -> str:
        if self.backend == "mlx":
            return self._generate_mlx(text, output_path, instruct, language, seed, temperature)
        else:
            return self._generate_pytorch(text, output_path, instruct, language, seed, temperature)

    def _generate_mlx(
        self,
        text: str,
        output_path: str,
        instruct: str,
        language: str,
        seed: int | None = None,
        temperature: float = 0.6,
    ) -> str:
        import mlx.core as mx

        if seed is not None:
            np.random.seed(seed)
            mx.random.seed(seed)

        lang_code    = "English" if language.lower() == "english" else language
        audio_chunks = []
        sr           = 24000
        # V10 testinde top_p kaldırıldı (boğulmayı önlemek için), o yüzden sadece temperature yollanıyor
        gen_kwargs   = {"temperature": temperature}

        if self.voice_mode == "design" and hasattr(self.model, "generate_voice_design"):
            design_instruct = self.design_prompt or instruct
            log.info(f"[Design MLX] Generating: '{design_instruct[:60]}'")
            for result in self.model.generate_voice_design(
                text=text,
                language=lang_code,
                instruct=design_instruct,
                **gen_kwargs,
            ):
                audio_chunks.append(np.array(result.audio))
                sr = result.sample_rate

        else:
            log.info(
                f"[Clone MLX] Generating with speaker='{self._target_speaker}' "
                f"alpha={self.clone_alpha:.2f} seed={seed} temp={temperature:.2f}"
            )
            for result in self.model.generate_custom_voice(
                text=text,
                speaker=self._target_speaker,
                language=lang_code,
                instruct=instruct,
                **gen_kwargs,
            ):
                audio_chunks.append(np.array(result.audio))
                sr = result.sample_rate

        wav = np.concatenate(audio_chunks) if audio_chunks else np.array([], dtype=np.float32)
        wav = np.asarray(wav, dtype=np.float32)
        max_val = float(np.max(np.abs(wav))) if len(wav) > 0 else 0.0
        if max_val > 1.0:
            wav = wav / max_val

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_p), wav, sr)

        try:
            import gc
            gc.collect()
            mx.clear_cache()
            log.info("[Clone MLX] Cleared MLX cache and collected garbage.")
        except Exception as e:
            log.warning(f"[Clone MLX] Failed to clean memory cache: {e}")

        return output_path

    def _generate_pytorch(
        self,
        text: str,
        output_path: str,
        instruct: str,
        language: str,
        seed: int | None = None,
        temperature: float = 0.6,
    ) -> str:
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

        lang_param = "English" if language.lower() == "english" else language

        if self.voice_prompt is not None:
            try:
                wavs, sr = self.model.generate_voice_clone(
                    text=text,
                    voice_clone_prompt=self.voice_prompt,
                    language=lang_param,
                    instruct=instruct,
                    temperature=temperature,
                )
            except TypeError:
                wavs, sr = self.model.generate_voice_clone(
                    text=text,
                    voice_clone_prompt=self.voice_prompt,
                    language=lang_param,
                    instruct=instruct,
                )
        else:
            try:
                wavs, sr = self.model.generate_custom_voice(
                    text=text,
                    language=lang_param,
                    speaker="Ethan",
                    instruct=instruct,
                    temperature=temperature,
                )
            except TypeError:
                wavs, sr = self.model.generate_custom_voice(
                    text=text,
                    language=lang_param,
                    speaker="Ethan",
                    instruct=instruct,
                )

        wav     = wavs[0]
        wav     = np.asarray(wav, dtype=np.float32)
        max_val = float(np.max(np.abs(wav))) if len(wav) > 0 else 0.0
        if max_val > 1.0:
            wav = wav / max_val

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_p), wav, sr)
        return output_path