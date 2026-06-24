# Copyright (c) 2026 Efeberk Çeker
# SPDX-License-Identifier: AGPL-3.0-only

#!/usr/bin/env python3
"""
VoiceDesign + Speaker Patch — Full Emotion Matrix Test
=======================================================

Mevcut VoiceDesign angry testinin eksikliğini kapatır.

Önceki testin sorunları:
  - Sadece 1 duygu (angry), 1 alpha, 0 beta sweep
  - Speaker patch spk_id manuel mapping (güvenilmez)
  - CustomVoice ile yan yana karşılaştırma yok

Bu test:
  - 2 model: CustomVoice vs VoiceDesign
  - 6 duygu × 3 intensity = 18 condition
  - 3 alpha: 0.0 (ham token), 0.5 (karışık), 1.0 (saf ref)
  - 5 instruct_scale (beta): 0.0, 0.4, 0.7, 1.0, 1.3
    (VoiceDesign'da instruct embed mean shift ile duygu şiddeti kontrolü)
  - Her satır: timbre cos_ref, f0, centroid, pause, rms

Sorular:
  Q1: VoiceDesign + patch hangi duygu için CustomVoice'tan daha iyi?
  Q2: Alpha=1.0 + beta=1.0 → timbre korunuyor mu?
  Q3: Beta<1.0 → duygu yumuşuyor ama ses kalitesi artıyor mu?
  Q4: En iyi emotion+timbre dengesi hangi model × alpha × beta?

Mimari notlar (_prepare_generation_inputs kaynağından):
  CustomVoice:
    input = [instruct_embed | role | codec+speaker | text]
    speaker → codec prefix (token weight[spk_id])

  VoiceDesign:
    input = [instruct_embed | role | text]  (speaker yok!)
    speaker embed → codec prefix'e manuel eklenir (_generate_with_instruct çağrısı)

  Beta injection:
    mixed_embed = neutral_embed + beta * (emotion_embed - neutral_embed)
    → beta=0.0: neutral ses
    → beta=1.0: tam emotion
    → beta=1.3: amplified emotion
"""

import os, sys, gc, json, time, logging, types
from pathlib import Path
from itertools import product as iproduct

import numpy as np
import soundfile as sf
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("vd_emotion_matrix")

# ── Config ────────────────────────────────────────────────────────────────────

CUSTOM_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"
DESIGN_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
BASE_ID   = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
REF_AUDIO = "pipeline_work/12345.mp3"

OUT = Path("pipeline_work/test_outputs/vd_emotion_matrix")
WAV = OUT / "wav"
OUT.mkdir(parents=True, exist_ok=True)
WAV.mkdir(parents=True, exist_ok=True)

LANG     = "English"
SEED     = 42
TOP_K    = 3
SPEAKER  = "ryan"
TEMP     = 0.45

# Sabit metin — duygusal içerik yok, tamamen instruct-driven
NEUTRAL_TEXT = (
    "The results came in this morning. "
    "I read through every line carefully before saying anything."
)

# ── Duygu × Intensity matrisi ─────────────────────────────────────────────────
EMOTIONS = [
    # Baseline
    {
        "emotion": "neutral", "intensity": "none",
        "instruct": "Speak naturally and clearly. Calm neutral delivery. No emotion.",
        "is_baseline": True,
    },
    # ANGRY
    {
        "emotion": "angry", "intensity": "low",
        "instruct": "Slightly irritated. A little edge in the voice. Mostly controlled.",
        "is_baseline": False,
    },
    {
        "emotion": "angry", "intensity": "mid",
        "instruct": "Clearly angry. Sharp consonants. Faster pace. Tight jaw.",
        "is_baseline": False,
    },
    {
        "emotion": "angry", "intensity": "high",
        "instruct": (
            "Furious. Barely controlled rage. Explosive emphasis on key words. "
            "Short punchy delivery. Voice grounded and low, not high-pitched. "
            "Sharp tense pauses before accusations."
        ),
        "is_baseline": False,
    },
    # SAD
    {
        "emotion": "sad", "intensity": "low",
        "instruct": "Slightly sad. Softer tone. A little slower. Quiet.",
        "is_baseline": False,
    },
    {
        "emotion": "sad", "intensity": "mid",
        "instruct": "Clearly sad. Low energy. Slow. Voice getting heavier.",
        "is_baseline": False,
    },
    {
        "emotion": "sad", "intensity": "high",
        "instruct": "Deep grief. On the verge of tears. Very slow. Weak voice. Long pauses.",
        "is_baseline": False,
    },
    # HAPPY
    {
        "emotion": "happy", "intensity": "low",
        "instruct": "Slightly happy. Warm tone. Small smile in the voice.",
        "is_baseline": False,
    },
    {
        "emotion": "happy", "intensity": "mid",
        "instruct": "Clearly happy. Brighter pitch. Energetic rhythm. Genuine warmth.",
        "is_baseline": False,
    },
    {
        "emotion": "happy", "intensity": "high",
        "instruct": "Extremely joyful. Almost laughing. Bright high pitch. Very energetic.",
        "is_baseline": False,
    },
    # FEAR
    {
        "emotion": "fear", "intensity": "mid",
        "instruct": "Clearly scared. Faster pace. Shaky voice. Short breaths between phrases.",
        "is_baseline": False,
    },
    {
        "emotion": "fear", "intensity": "high",
        "instruct": "Terrified. Barely able to speak. Trembling. Hushed and urgent.",
        "is_baseline": False,
    },
    # CALM/AUTHORITY
    {
        "emotion": "calm", "intensity": "mid",
        "instruct": "Very calm, controlled, authoritative. Deliberate pace. Each word placed carefully.",
        "is_baseline": False,
    },
    {
        "emotion": "calm", "intensity": "high",
        "instruct": "Deeply calm. Almost meditative. Very slow, wide resonant delivery.",
        "is_baseline": False,
    },
    # SARCASTIC
    {
        "emotion": "sarcastic", "intensity": "mid",
        "instruct": "Clearly sarcastic. Flat affect with a slight smirk. Dry, slightly elongated delivery.",
        "is_baseline": False,
    },
    {
        "emotion": "sarcastic", "intensity": "high",
        "instruct": "Very sarcastic. Exaggerated slow delivery on key words. Audible eye-roll quality.",
        "is_baseline": False,
    },
]

# Neutral baseline için instruct (beta injection referansı)
NEUTRAL_INSTRUCT = EMOTIONS[0]["instruct"]

ALPHAS = [0.0, 0.5, 1.0]

# Beta = instruct strength scale
# 0.0 = neutral embed, 1.0 = tam emotion, 1.3 = amplified
BETAS = [0.0, 0.4, 0.7, 1.0, 1.3]

MODELS = ["CustomVoice", "VoiceDesign"]

# ── Helpers ──────────────────────────────────────────────────────────────────

def clean():
    gc.collect()
    try: mx.clear_cache()
    except Exception: pass


def cosine(a, b):
    a = a.astype(mx.float32).reshape(-1)
    b = b.astype(mx.float32).reshape(-1)
    return float(mx.sum(a * b) / (mx.linalg.norm(a) * mx.linalg.norm(b) + 1e-8))


def norm_val(x):
    return float(mx.linalg.norm(x))


def save_wav(path, chunks, sr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(1, np.float32)
    wav = np.nan_to_num(wav)
    peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    if peak > 1.0:
        wav /= peak
    sf.write(str(path), wav, sr)


def analyze(path):
    import warnings; warnings.filterwarnings("ignore")
    wav, sr = sf.read(str(path))
    wav = np.asarray(wav, np.float32)
    if wav.ndim > 1: wav = wav.mean(axis=1)
    n = len(wav)
    if n == 0: return {}
    abs_wav = np.abs(wav)
    rms      = float(np.sqrt(np.mean(wav**2)))
    silence  = float(np.mean(abs_wav < 0.005))
    spec     = np.abs(np.fft.rfft(wav))
    freqs    = np.fft.rfftfreq(n, 1/sr)
    centroid = float(np.sum(freqs*spec) / (np.sum(spec)+1e-12))
    high_r   = float(np.sum(spec[freqs>=4000]**2) / (np.sum(spec**2)+1e-12))
    f0_mean = f0_std = voiced_ratio = 0.0
    try:
        import librosa
        y16, _ = librosa.load(str(path), sr=16000, mono=True)
        f0, _, _ = librosa.pyin(y16, fmin=60, fmax=500, sr=16000,
                                frame_length=1024, hop_length=256)
        voiced = f0[~np.isnan(f0)]
        if len(voiced) >= 4:
            f0_mean      = float(np.mean(voiced))
            f0_std       = float(np.std(voiced))
            voiced_ratio = float(len(voiced) / len(f0))
    except Exception: pass
    return {
        "duration":     round(n/sr, 3),
        "rms":          round(rms, 6),
        "silence":      round(silence, 4),
        "centroid":     round(centroid, 2),
        "high_ratio":   round(high_r, 5),
        "f0_mean":      round(f0_mean, 2),
        "f0_std":       round(f0_std, 2),
        "voiced_ratio": round(voiced_ratio, 4),
    }


# ── Speaker encoder helpers ───────────────────────────────────────────────────

def build_ref_centroid(base_model, audio_path, top_k=3):
    from mlx_audio.utils import load_audio
    from finetune import segment_audio
    sr    = base_model.sample_rate
    audio = np.array(load_audio(str(audio_path), sample_rate=sr))
    segs  = segment_audio(audio, sr, min_sec=3.0, max_sec=8.0)
    if not segs: raise RuntimeError("No reference segments")
    embs  = [base_model.extract_speaker_embedding(mx.array(s), sr=sr).squeeze() for s in segs]
    avg   = mx.stack(embs).mean(axis=0).squeeze()
    scores = [(i, float(np.mean([cosine(e,x) for j,x in enumerate(embs) if j!=i])) if len(embs)>1 else 1.0)
              for i,e in enumerate(embs)]
    scores.sort(key=lambda x:x[1], reverse=True)
    centroid = mx.stack([embs[i] for i,_ in scores[:top_k]]).mean(axis=0).squeeze()
    log.info(f"Ref centroid: norm={norm_val(centroid):.4f}")
    return centroid, avg


def reencode(base_model, wav_path):
    from mlx_audio.utils import load_audio
    from finetune import segment_audio
    sr    = base_model.sample_rate
    audio = np.array(load_audio(str(wav_path), sample_rate=sr))
    segs  = segment_audio(audio, sr, min_sec=3.0, max_sec=8.0) or [audio]
    embs  = [base_model.extract_speaker_embedding(mx.array(s), sr=sr).squeeze() for s in segs]
    return mx.stack(embs).mean(axis=0).squeeze(), len(embs)


# ── Speaker token patch ───────────────────────────────────────────────────────

def patch_speaker(model, speaker, ref_vec, alpha):
    """
    Norm-match interpolation (aynı interpolation testi yöntemi).
    VoiceDesign'da spk_id yoksa dinamik olarak ekler.
    """
    config    = model.config.talker_config
    emb_layer = model.talker.get_input_embeddings()

    # VoiceDesign'da spk_id olmayabilir
    if not hasattr(config, "spk_id") or config.spk_id is None:
        config.spk_id = {}

    if speaker not in config.spk_id:
        # CustomVoice'taki ryan token ID: 3066 — aynısını kullan
        config.spk_id[speaker] = 3066
        log.info(f"VoiceDesign: dynamically mapped '{speaker}' → token 3066")

    sid     = config.spk_id[speaker]
    src_vec = mx.array(emb_layer.weight[int(sid)]).squeeze()

    if alpha == 0.0:
        # Ham token, değiştirme
        return {"alpha": 0.0, "cos_final_ref": round(cosine(src_vec, ref_vec), 5)}

    mixed = (1.0 - alpha) * src_vec + alpha * ref_vec
    final = mixed * (mx.linalg.norm(src_vec) / (mx.linalg.norm(mixed) + 1e-8))
    emb_layer.weight[int(sid)] = final.reshape(src_vec.shape).astype(src_vec.dtype)
    mx.eval(emb_layer.weight)

    return {"alpha": alpha, "cos_final_ref": round(cosine(final, ref_vec), 5)}


# ── Instruct embed extraction ─────────────────────────────────────────────────

def get_instruct_embed(model, instruct_text: str) -> mx.array:
    """
    _prepare_generation_inputs'teki encoding'i birebir uygular:
        f"<|im_start|>user\n{instruct}<|im_end|>\n"
    """
    formatted = f"<|im_start|>user\n{instruct_text}<|im_end|>\n"
    ids       = mx.array(model.tokenizer.encode(formatted))[None, :]
    return model.talker.text_projection(
        model.talker.get_text_embeddings()(ids)
    )  # [1, seq_len, hidden]


def build_mixed_embed(neutral_embed, emotion_embed, beta):
    """
    beta=0.0 → neutral embed mean
    beta=1.0 → orijinal emotion embed
    beta>1.0 → amplified emotion direction

    Yöntem: emotion embed'in mean'ini beta × delta kaydır.
    Seq len farkı olduğu için mean-level shift kullanıyoruz.
    """
    neutral_mean = neutral_embed.mean(axis=1, keepdims=True)  # [1,1,hidden]
    emotion_mean = emotion_embed.mean(axis=1, keepdims=True)  # [1,1,hidden]
    delta        = emotion_mean - neutral_mean
    shift        = (1.0 - beta) * (-delta)
    return emotion_embed + shift  # [1, emo_len, hidden]


# ── Patched generation (her iki model için) ───────────────────────────────────

def generate_patched(
    model,
    model_type: str,
    text: str,
    speaker: str,
    neutral_embed: mx.array,
    emotion_embed: mx.array,
    beta: float,
    out_path: Path,
) -> dict:
    """
    _prepare_generation_inputs'i monkey-patch ederek
    instruct_embed'i mixed embed ile değiştirir.

    CustomVoice ve VoiceDesign için aynı path çalışır:
    çünkü her iki modelde de _prepare_generation_inputs mevcut.
    """
    if out_path.exists():
        log.info(f"      [SKIP] Already exists: {out_path.name}")
        try:
            metrics = analyze(str(out_path))
            metrics["latency"] = 0.0
            return metrics
        except Exception as e:
            log.warning(f"Failed to analyze existing file {out_path.name}, regenerating: {e}")

    np.random.seed(SEED)
    mx.random.seed(SEED)

    mixed_embed      = build_mixed_embed(neutral_embed, emotion_embed, beta)
    original_prepare = model._prepare_generation_inputs

    def patched_prepare(
        self_inner,
        text=text,
        language=LANG,
        speaker=speaker,
        ref_audio=None,
        ref_text=None,
        instruct=None,
    ):
        config = self_inner.config.talker_config

        # Text embed
        chat_text  = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        input_ids  = mx.array(self_inner.tokenizer.encode(chat_text))[None, :]
        text_embed = self_inner.talker.text_projection(
            self_inner.talker.get_text_embeddings()(input_ids)
        )

        # TTS special tokens
        tts_tokens = mx.array([[
            self_inner.config.tts_bos_token_id,
            self_inner.config.tts_eos_token_id,
            self_inner.config.tts_pad_token_id,
        ]])
        tts_embeds    = self_inner.talker.text_projection(
            self_inner.talker.get_text_embeddings()(tts_tokens)
        )
        tts_bos_embed = tts_embeds[:, 0:1, :]
        tts_eos_embed = tts_embeds[:, 1:2, :]
        tts_pad_embed = tts_embeds[:, 2:3, :]

        # Speaker embed (her iki modelde de artık spk_id mevcut)
        speaker_embed = None
        spk_id_map    = getattr(config, "spk_id", {}) or {}
        if speaker and speaker.lower() in spk_id_map:
            spk_ids       = mx.array([[spk_id_map[speaker.lower()]]])
            speaker_embed = self_inner.talker.get_input_embeddings()(spk_ids)

        # Language id
        language_id = None
        if language.lower() != "auto" and config.codec_language_id:
            if language.lower() in config.codec_language_id:
                language_id = config.codec_language_id[language.lower()]

        if language.lower() in ["chinese", "auto"] and speaker:
            if speaker.lower() in (getattr(config, "spk_is_dialect", {}) or {}):
                dialect = config.spk_is_dialect[speaker.lower()]
                if dialect in config.codec_language_id:
                    language_id = config.codec_language_id[dialect]

        # Codec prefix
        if language_id is None:
            codec_prefill = [
                config.codec_nothink_id,
                config.codec_think_bos_id,
                config.codec_think_eos_id,
            ]
        else:
            codec_prefill = [
                config.codec_think_id,
                config.codec_think_bos_id,
                language_id,
                config.codec_think_eos_id,
            ]

        codec_embed = self_inner.talker.get_input_embeddings()(mx.array([codec_prefill]))
        codec_suffix = self_inner.talker.get_input_embeddings()(
            mx.array([[config.codec_pad_id, config.codec_bos_id]])
        )

        if speaker_embed is not None:
            codec_embed = mx.concatenate(
                [codec_embed, speaker_embed.reshape(1,1,-1), codec_suffix], axis=1
            )
        else:
            codec_embed = mx.concatenate([codec_embed, codec_suffix], axis=1)

        # ── INJECTION: mixed instruct embed ──
        instruct_embed_to_use = mixed_embed

        role_embed = text_embed[:, :3, :]
        pad_count  = codec_embed.shape[1] - 2
        pad_embeds = mx.broadcast_to(tts_pad_embed, (1, pad_count, tts_pad_embed.shape[-1]))
        combined   = mx.concatenate([pad_embeds, tts_bos_embed], axis=1)
        combined   = combined + codec_embed[:, :-1, :]

        input_embeds = mx.concatenate(
            [instruct_embed_to_use, role_embed, combined], axis=1
        )

        first_text   = text_embed[:, 3:4, :] + codec_embed[:, -1:, :]
        input_embeds = mx.concatenate([input_embeds, first_text], axis=1)

        trailing = mx.concatenate([text_embed[:, 4:-5, :], tts_eos_embed], axis=1)

        return input_embeds, trailing, tts_pad_embed

    model._prepare_generation_inputs = types.MethodType(patched_prepare, model)

    try:
        chunks = []
        sr     = 24000
        t0     = time.time()

        # VoiceDesign model doesn't support generate_custom_voice directly, so we call _generate_with_instruct
        if model_type == "VoiceDesign":
            gen_iter = model._generate_with_instruct(
                text=text,
                speaker=speaker,
                language=LANG,
                instruct="__INJECTED__",
                temperature=TEMP,
                max_tokens=4096,
                top_k=50,
                top_p=1.0,
                repetition_penalty=1.05,
                verbose=False,
            )
        else:
            gen_iter = model.generate_custom_voice(
                text=text,
                speaker=speaker,
                language=LANG,
                instruct="__INJECTED__",
                temperature=TEMP,
            )

        for result in gen_iter:
            chunks.append(np.array(result.audio))
            sr = result.sample_rate

        latency = round(time.time() - t0, 3)
        save_wav(out_path, chunks, sr)
    finally:
        model._prepare_generation_inputs = original_prepare

    metrics          = analyze(str(out_path))
    metrics["latency"] = latency
    return metrics


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    from mlx_audio.tts import load as mlx_load

    report = {
        "timestamp":    time.strftime("%Y-%m-%d %H:%M:%S"),
        "goal":         "VoiceDesign vs CustomVoice — full emotion × alpha × beta matrix.",
        "neutral_text": NEUTRAL_TEXT,
        "speaker":      SPEAKER,
        "alphas":       ALPHAS,
        "betas":        BETAS,
        "temperature":  TEMP,
        "models":       MODELS,
        "rows":         [],
        "verdicts":     [],
    }

    # ── Phase 1: Ref centroid ──────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("PHASE 1 | Reference centroid")
    log.info("=" * 60)
    base = mlx_load(BASE_ID)
    ref_centroid, ref_avg = build_ref_centroid(base, REF_AUDIO, TOP_K)
    del base; clean()

    rows  = []
    total = len(MODELS) * len(ALPHAS) * len(EMOTIONS) * len(BETAS)
    # Baseline sadece beta=1.0 çalışır, diğerleri tüm beta'lar
    # Gerçek toplam:
    non_baseline = sum(1 for e in EMOTIONS if not e["is_baseline"])
    baseline_cnt = sum(1 for e in EMOTIONS if e["is_baseline"])
    total = len(MODELS) * len(ALPHAS) * (baseline_cnt + non_baseline * len(BETAS))
    count = 0

    model_ids = {
        "CustomVoice": CUSTOM_ID,
        "VoiceDesign": DESIGN_ID,
    }

    for model_name, alpha in iproduct(MODELS, ALPHAS):
        log.info(f"\n{'='*60}")
        log.info(f"MODEL={model_name}  ALPHA={alpha}")
        log.info(f"{'='*60}")

        model = mlx_load(model_ids[model_name])
        patch_info = patch_speaker(model, SPEAKER, ref_centroid, alpha)
        log.info(f"Patch: {patch_info}")

        # Neutral embed (beta injection referansı)
        neutral_embed = get_instruct_embed(model, NEUTRAL_INSTRUCT)
        log.info(f"Neutral embed: shape={neutral_embed.shape}  norm={norm_val(neutral_embed):.4f}")

        for emo in EMOTIONS:
            emo_key = f"{emo['emotion']}_{emo['intensity']}"

            # Baseline → sadece beta=1.0
            beta_list = [1.0] if emo["is_baseline"] else BETAS

            emotion_embed = get_instruct_embed(model, emo["instruct"])
            cos_ne = cosine(
                neutral_embed.mean(axis=1).squeeze(),
                emotion_embed.mean(axis=1).squeeze()
            )

            for beta in beta_list:
                count += 1
                log.info(
                    f"  [{count}/{total}] {model_name} a={alpha} "
                    f"{emo_key} b={beta:.1f}"
                )

                fname    = WAV / f"{model_name}_a{alpha:.1f}_{emo_key}_b{beta:.2f}.wav"
                metrics  = generate_patched(
                    model, model_name,
                    NEUTRAL_TEXT, SPEAKER,
                    neutral_embed, emotion_embed,
                    beta, fname
                )

                # Timbre re-encode
                base_m = mlx_load(BASE_ID)
                gen_emb, seg_count = reencode(base_m, fname)
                cos_ref = cosine(ref_centroid, gen_emb)
                del base_m; clean()

                row = {
                    "model":      model_name,
                    "alpha":      alpha,
                    "emotion":    emo["emotion"],
                    "intensity":  emo["intensity"],
                    "beta":       beta,
                    "is_baseline":emo["is_baseline"],
                    "instruct":   emo["instruct"],
                    "file":       str(fname),
                    "cos_ref":    round(cos_ref, 5),
                    "cos_ne":     round(cos_ne, 5),
                    "seg_count":  seg_count,
                    "patch":      patch_info,
                    **metrics,
                }
                rows.append(row)

                log.info(
                    f"    cos_ref={cos_ref:.5f}  "
                    f"f0={metrics.get('f0_mean',0):.1f}Hz  "
                    f"centroid={metrics.get('centroid',0):.0f}Hz  "
                    f"silence={metrics.get('silence',0):.3f}"
                )

        del model; clean()

    report["rows"] = rows

    # ── Verdicts ──────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("PHASE FINAL | Verdicts")
    log.info("=" * 60)

    verdicts  = []
    emotions  = list(set(r["emotion"] for r in rows if not r["is_baseline"]))

    # V1: VoiceDesign vs CustomVoice — timbre at alpha=1.0 beta=1.0
    for emo in emotions:
        for alpha in ALPHAS:
            cv = next((r for r in rows if r["model"]=="CustomVoice"
                       and r["emotion"]==emo and r["alpha"]==alpha
                       and r["beta"]==1.0), None)
            vd = next((r for r in rows if r["model"]=="VoiceDesign"
                       and r["emotion"]==emo and r["alpha"]==alpha
                       and r["beta"]==1.0), None)
            if not (cv and vd): continue

            baseline_cv = next((r for r in rows if r["model"]=="CustomVoice"
                                and r["is_baseline"] and r["alpha"]==alpha), None)
            if not baseline_cv: continue

            cv_loss = baseline_cv["cos_ref"] - cv["cos_ref"]
            vd_loss = baseline_cv["cos_ref"] - vd["cos_ref"]

            f0_diff_cv = abs(cv.get("f0_mean",0) - baseline_cv.get("f0_mean",0))
            f0_diff_vd = abs(vd.get("f0_mean",0) - baseline_cv.get("f0_mean",0))

            better_emo = "VoiceDesign" if f0_diff_vd > f0_diff_cv else "CustomVoice"
            better_tim = "VoiceDesign" if vd_loss < cv_loss else "CustomVoice"

            verdicts.append(
                f"[{emo}/α={alpha}] Emotion winner: {better_emo} "
                f"(Δf0 CV={f0_diff_cv:.1f} VD={f0_diff_vd:.1f}) | "
                f"Timbre winner: {better_tim} "
                f"(loss CV={cv_loss:.4f} VD={vd_loss:.4f})"
            )

    # V2: Beta sweep monotonicity per emotion
    for model_name in MODELS:
        for emo in emotions:
            alpha = 1.0
            beta_rows = sorted(
                [r for r in rows if r["model"]==model_name
                 and r["emotion"]==emo and r["alpha"]==alpha],
                key=lambda x: x["beta"]
            )
            if len(beta_rows) < 3: continue
            f0s  = [r.get("f0_mean",0) for r in beta_rows]
            betas_used = [r["beta"] for r in beta_rows]
            diffs = [f0s[i+1] - f0s[i] for i in range(len(f0s)-1)]
            monotone = all(d >= -5 for d in diffs) or all(d <= 5 for d in diffs)
            verdicts.append(
                f"[{model_name}/{emo}/α=1.0] Beta→f0 monotone: {monotone}  "
                f"f0 range: {min(f0s):.1f}→{max(f0s):.1f}Hz  "
                f"betas: {betas_used}"
            )

    # V3: Best configuration per emotion
    best_configs = []
    for emo in emotions:
        candidates = [
            r for r in rows
            if r["emotion"]==emo
            and r["alpha"]==1.0
            and r["beta"]==1.0
        ]
        if not candidates: continue

        baseline_rows = [r for r in rows if r["is_baseline"] and r["alpha"]==1.0]
        if not baseline_rows: continue
        baseline_cos = float(np.mean([r["cos_ref"] for r in baseline_rows]))

        def score(r):
            f0_shift = abs(r.get("f0_mean",0) - 140)   # 140Hz = rough neutral
            timbre   = r["cos_ref"]
            return f0_shift * timbre   # high f0 shift + high timbre = ideal

        candidates.sort(key=score, reverse=True)
        best = candidates[0]
        best_configs.append({
            "emotion": emo,
            "best_model": best["model"],
            "best_alpha": best["alpha"],
            "best_beta": best["beta"],
            "cos_ref": best["cos_ref"],
            "f0_mean": best.get("f0_mean",0),
        })
        verdicts.append(
            f"BEST [{emo}]: {best['model']} α={best['alpha']} β={best['beta']} "
            f"cos_ref={best['cos_ref']:.5f} f0={best.get('f0_mean',0):.1f}Hz"
        )

    report["verdicts"]     = verdicts
    report["best_configs"] = best_configs

    # ── Save ──────────────────────────────────────────────────────────────────
    json_path = OUT / "vd_emotion_matrix_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md_path = OUT / "vd_emotion_matrix_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# VoiceDesign vs CustomVoice — Emotion Matrix\n\n")
        f.write(f"**Timestamp:** {report['timestamp']}\n\n")

        f.write("## Verdicts\n\n")
        for v in verdicts:
            f.write(f"- {v}\n")

        f.write("\n## Best Configuration per Emotion\n\n")
        f.write("| Emotion | Model | Alpha | Beta | cos_ref | f0_mean |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        for b in best_configs:
            f.write(
                f"| {b['emotion']} | {b['best_model']} | {b['best_alpha']} | "
                f"{b['best_beta']} | {b['cos_ref']} | {b['f0_mean']:.1f} |\n"
            )

        f.write("\n## Full Matrix (alpha=1.0, beta=1.0)\n\n")
        f.write("| Model | Emotion | Intensity | cos_ref | f0_mean | centroid | silence |\n")
        f.write("|---|---|---|---:|---:|---:|---:|\n")
        for r in rows:
            if r["alpha"]==1.0 and r["beta"]==1.0:
                f.write(
                    f"| {r['model']} | {r['emotion']} | {r['intensity']} | "
                    f"{r['cos_ref']} | {r.get('f0_mean',0):.1f} | "
                    f"{r.get('centroid',0):.0f} | {r.get('silence',0):.3f} |\n"
                )

        f.write("\n## Listen Order — By Emotion\n\n")
        for emo_name in sorted(set(r["emotion"] for r in rows if not r["is_baseline"])):
            f.write(f"### {emo_name}\n\n")
            for model_name in MODELS:
                f.write(f"**{model_name}** (alpha=1.0)\n\n")
                emo_rows = sorted(
                    [r for r in rows if r["emotion"]==emo_name
                     and r["model"]==model_name and r["alpha"]==1.0],
                    key=lambda x: x["beta"]
                )
                for r in emo_rows:
                    f.write(
                        f"- beta={r['beta']:.2f}: `{r['file']}`  "
                        f"cos={r['cos_ref']}  f0={r.get('f0_mean',0):.1f}Hz\n"
                    )
                f.write("\n")

    # Terminal summary
    print("\n" + "=" * 70)
    print("VD EMOTION MATRIX — RESULTS")
    print("=" * 70)
    print(f"\n{'Model':>12}  {'Emotion':>10}  {'α':>4}  {'β':>4}  {'cos_ref':>9}  {'f0':>7}")
    print(f"{'-----':>12}  {'-------':>10}  {'-':>4}  {'-':>4}  {'-------':>9}  {'--':>7}")
    for r in sorted(rows, key=lambda x: (x["model"], x["emotion"], x["alpha"], x["beta"])):
        if r["alpha"] == 1.0 and not r["is_baseline"]:
            print(
                f"{r['model']:>12}  {r['emotion']:>10}  "
                f"{r['alpha']:>4.1f}  {r['beta']:>4.1f}  "
                f"{r['cos_ref']:>9.5f}  {r.get('f0_mean',0):>7.1f}"
            )
    print("\nVERDICTS:")
    for v in verdicts[:20]:
        print(f"  {v}")
    print(f"\nJSON: {json_path}")
    print(f"MD:   {md_path}")
    print(f"WAV:  {WAV}")
    print(f"Total samples: {len(rows)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
