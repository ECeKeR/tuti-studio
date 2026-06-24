# Copyright (c) 2026 Efeberk Çeker
# SPDX-License-Identifier: AGPL-3.0-only

#!/usr/bin/env python3
"""
Instruct Delta Injection Test
==============================
Kanıt sorusu: instruct_embed'e doğrudan delta inject ederek
              timbre'yi koruyup duyguyu kontrol edebilir miyiz?

Yöntem:
  neutral_embed  = encode_instruct("Speak naturally and clearly.")
  emotion_embed  = encode_instruct("Very angry, sharp consonants…")
  delta          = emotion_embed - neutral_embed

  final = neutral_embed + beta * delta

Beta sweep [0.0, 0.3, 0.5, 0.7, 1.0, 1.3, 1.6]:
  0.0 = saf neutral (referans)
  1.0 = orijinal emotion instruct
  >1.0 = amplified emotion
  <1.0 = softened emotion

Model kaynak koduna göre (_prepare_generation_inputs):

  instruct_text = f"<|im_start|>user\n{instruct}<|im_end|>\n"
  instruct_ids = tokenizer.encode(instruct_text)
  instruct_embed = talker.text_projection(
      talker.get_text_embeddings()(instruct_ids)
  )
  input_embeds = [instruct_embed, role_embed, combined_embed]

Bizim injection: bu pipeline'ı kopyalayıp instruct_embed'i
                 manipüle ettikten sonra direkt generation'a veriyoruz.
"""

import os, sys, gc, json, time, logging
from pathlib import Path

import numpy as np
import soundfile as sf
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("instruct_delta_injection")

# ── Config ────────────────────────────────────────────────────────────────────

CUSTOM_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"
BASE_ID   = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
REF_AUDIO = "pipeline_work/12345.mp3"

OUT = Path("pipeline_work/test_outputs/instruct_delta_injection")
WAV = OUT / "wav"
OUT.mkdir(parents=True, exist_ok=True)
WAV.mkdir(parents=True, exist_ok=True)

LANG = "English"
SEED = 42
TOP_K = 3
TARGET_SPEAKER = "ryan"

TEST_TEXT = (
    "The results came in this morning. "
    "I read through every line carefully before saying anything."
)

# Beta sweep — emotion strength multiplier
BETAS = [0.0, 0.3, 0.5, 0.7, 1.0, 1.3, 1.6]

# Alpha sweep — speaker timbre injection (sadece 1.0 ve 0.0 karşılaştırması)
ALPHAS = [0.0, 1.0]

# Duygular: neutral baseline + test emotions
EMOTIONS = [
    {
        "name": "neutral",
        "instruct": "Speak naturally and clearly. Calm neutral delivery.",
        "is_baseline": True,
    },
    {
        "name": "angry",
        "instruct": "Very angry and frustrated. Sharp consonants. Tight jaw. Explosive emphasis on key words. Fast, intense delivery.",
        "is_baseline": False,
    },
    {
        "name": "sad",
        "instruct": "Deep grief. On the verge of tears. Very slow. Weak voice. Long emotional pauses.",
        "is_baseline": False,
    },
    {
        "name": "happy",
        "instruct": "Extremely joyful and excited. Almost laughing. Bright high pitch. Very energetic.",
        "is_baseline": False,
    },
    {
        "name": "fear",
        "instruct": "Terrified. Barely able to speak. Trembling. Hushed and urgent.",
        "is_baseline": False,
    },
    {
        "name": "sarcastic",
        "instruct": "Very sarcastic. Exaggerated slow delivery on key words. Audible eye-roll quality. Dry flat affect.",
        "is_baseline": False,
    },
]

TEMP = 0.45

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
    wav = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(1, np.float32)
    wav = np.nan_to_num(wav)
    peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    if peak > 1.0:
        wav /= peak
    sf.write(str(path), wav, sr)


def analyze_audio(path):
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
    centroid = float(np.sum(freqs * spec) / (np.sum(spec) + 1e-12))
    high_r   = float(np.sum(spec[freqs >= 4000]**2) / (np.sum(spec**2) + 1e-12))
    # Pitch
    f0_mean = f0_std = voiced_ratio = 0.0
    try:
        import librosa
        y16, _ = librosa.load(str(path), sr=16000, mono=True)
        f0, _, _ = librosa.pyin(y16, fmin=60, fmax=500, sr=16000, frame_length=1024, hop_length=256)
        voiced = f0[~np.isnan(f0)]
        if len(voiced) >= 4:
            f0_mean = float(np.mean(voiced))
            f0_std  = float(np.std(voiced))
            voiced_ratio = float(len(voiced) / len(f0))
    except Exception: pass
    return {
        "duration":      round(n/sr, 3),
        "rms":           round(rms, 6),
        "silence":       round(silence, 4),
        "centroid":      round(centroid, 2),
        "high_ratio":    round(high_r, 5),
        "f0_mean":       round(f0_mean, 2),
        "f0_std":        round(f0_std, 2),
        "voiced_ratio":  round(voiced_ratio, 4),
    }


# ── Reference centroid ────────────────────────────────────────────────────────

def build_ref_centroid(base_model, audio_path, top_k=3):
    from mlx_audio.utils import load_audio
    from finetune import segment_audio
    sr    = base_model.sample_rate
    audio = np.array(load_audio(str(audio_path), sample_rate=sr))
    segs  = segment_audio(audio, sr, min_sec=3.0, max_sec=8.0)
    if not segs: raise RuntimeError("No segments")
    embs  = [base_model.extract_speaker_embedding(mx.array(s), sr=sr).squeeze() for s in segs]
    avg   = mx.stack(embs).mean(axis=0).squeeze()
    scores = [(i, float(np.mean([cosine(e, x) for j,x in enumerate(embs) if j!=i])) if len(embs)>1 else 1.0)
              for i,e in enumerate(embs)]
    scores.sort(key=lambda x: x[1], reverse=True)
    centroid = mx.stack([embs[i] for i,_ in scores[:top_k]]).mean(axis=0).squeeze()
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
    emb_layer = model.talker.get_input_embeddings()
    sid       = model.config.talker_config.spk_id[speaker]
    src_vec   = mx.array(emb_layer.weight[int(sid)]).squeeze()
    mixed     = (1.0 - alpha) * src_vec + alpha * ref_vec
    final     = mixed * (mx.linalg.norm(src_vec) / (mx.linalg.norm(mixed) + 1e-8))
    emb_layer.weight[int(sid)] = final.reshape(src_vec.shape).astype(src_vec.dtype)
    mx.eval(emb_layer.weight)
    return round(cosine(final, ref_vec), 5)


# ── Instruct embed extraction ─────────────────────────────────────────────────

def get_instruct_embed(model, instruct_text: str) -> mx.array:
    """
    Reproduces the exact instruct encoding from _prepare_generation_inputs:

        instruct_text = f"<|im_start|>user\n{instruct}<|im_end|>\n"
        instruct_ids  = tokenizer.encode(instruct_text)
        instruct_embed = talker.text_projection(
            talker.get_text_embeddings()(instruct_ids)
        )
    """
    formatted = f"<|im_start|>user\n{instruct_text}<|im_end|>\n"
    ids       = mx.array(model.tokenizer.encode(formatted))[None, :]
    embed     = model.talker.text_projection(
        model.talker.get_text_embeddings()(ids)
    )
    return embed  # shape: [1, seq_len, hidden]


# ── Patched generation ────────────────────────────────────────────────────────

def generate_with_patched_instruct(
    model,
    text: str,
    speaker: str,
    neutral_embed: mx.array,
    emotion_embed: mx.array,
    beta: float,
    out_path: Path,
) -> dict:
    """
    1. Build mixed instruct embed:
         mixed = neutral + beta * (emotion - neutral)
    2. Monkey-patch _prepare_generation_inputs to inject mixed embed
    3. Call generate_custom_voice normally
    4. Restore original method
    """
    np.random.seed(SEED)
    mx.random.seed(SEED)

    # Compute mixed embed (broadcast-safe: seq lens may differ → use emotion shape)
    delta = emotion_embed - neutral_embed   # [1, emo_len, hidden]

    # If seq lengths differ, pad/truncate delta to neutral length or just use emotion
    # Strategy: scale the whole emotion embed toward neutral
    #   mixed = (1-beta)*neutral + beta*emotion   when beta=1 → pure emotion
    # But shapes differ → we scale the emotion embed by beta and use neutral as base only for beta=0
    # Simpler robust approach: use emotion embed scaled, add beta-weighted delta from a mean vector
    
    # Robust: just scale the emotion embed itself
    # beta=0 → neutral instruct embed
    # beta=1 → original emotion embed
    # beta>1 → amplified (emotion direction × beta from neutral mean)
    
    neutral_mean = neutral_embed.mean(axis=1, keepdims=True)  # [1,1,hidden]
    emotion_mean = emotion_embed.mean(axis=1, keepdims=True)  # [1,1,hidden]
    
    # Mean-level delta
    mean_delta = emotion_mean - neutral_mean  # direction in mean space
    
    # Mixed: start from emotion embed, then lerp mean toward neutral×(1-beta)
    # When beta=0: embed that points toward neutral
    # When beta=1: original emotion
    # When beta>1: amplified
    
    # Method: use emotion embed, but shift its mean by (1-beta)*(-mean_delta)
    shift        = (1.0 - beta) * (-mean_delta)
    mixed_embed  = emotion_embed + shift   # shape: [1, emo_len, hidden]
    
    # Store original method
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
        """
        Identical to original _prepare_generation_inputs except
        instruct_embed is replaced with our mixed_embed.
        """
        config = self_inner.config.talker_config

        chat_text  = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        input_ids  = mx.array(self_inner.tokenizer.encode(chat_text))[None, :]
        text_embed = self_inner.talker.text_projection(
            self_inner.talker.get_text_embeddings()(input_ids)
        )

        tts_tokens = mx.array([[
            self_inner.config.tts_bos_token_id,
            self_inner.config.tts_eos_token_id,
            self_inner.config.tts_pad_token_id,
        ]])
        tts_embeds   = self_inner.talker.text_projection(
            self_inner.talker.get_text_embeddings()(tts_tokens)
        )
        tts_bos_embed = tts_embeds[:, 0:1, :]
        tts_eos_embed = tts_embeds[:, 1:2, :]
        tts_pad_embed = tts_embeds[:, 2:3, :]

        # Speaker embed
        speaker_embed = None
        if speaker and speaker.lower() in (config.spk_id or {}):
            spk_ids = mx.array([[config.spk_id[speaker.lower()]]])
            speaker_embed = self_inner.talker.get_input_embeddings()(spk_ids)

        # Language id
        language_id = None
        if language.lower() != "auto" and config.codec_language_id:
            if language.lower() in config.codec_language_id:
                language_id = config.codec_language_id[language.lower()]

        if language.lower() in ["chinese", "auto"] and speaker:
            if speaker.lower() in (config.spk_is_dialect or {}):
                dialect = config.spk_is_dialect[speaker.lower()]
                if dialect in config.codec_language_id:
                    language_id = config.codec_language_id[dialect]

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
        codec_embed_suffix = self_inner.talker.get_input_embeddings()(
            mx.array([[config.codec_pad_id, config.codec_bos_id]])
        )

        if speaker_embed is not None:
            codec_embed = mx.concatenate(
                [codec_embed, speaker_embed.reshape(1, 1, -1), codec_embed_suffix], axis=1
            )
        else:
            codec_embed = mx.concatenate([codec_embed, codec_embed_suffix], axis=1)

        # ── INJECTION: use mixed_embed instead of encoding instruct again ──
        instruct_embed_to_use = mixed_embed  # [1, emo_len, hidden]

        role_embed = text_embed[:, :3, :]

        pad_count   = codec_embed.shape[1] - 2
        pad_embeds  = mx.broadcast_to(tts_pad_embed, (1, pad_count, tts_pad_embed.shape[-1]))
        combined    = mx.concatenate([pad_embeds, tts_bos_embed], axis=1)
        combined    = combined + codec_embed[:, :-1, :]

        input_embeds = mx.concatenate(
            [instruct_embed_to_use, role_embed, combined], axis=1
        )

        first_text_embed = text_embed[:, 3:4, :] + codec_embed[:, -1:, :]
        input_embeds     = mx.concatenate([input_embeds, first_text_embed], axis=1)

        trailing_text_hidden = mx.concatenate(
            [text_embed[:, 4:-5, :], tts_eos_embed], axis=1
        )

        return input_embeds, trailing_text_hidden, tts_pad_embed

    # Monkey-patch (bound method style)
    import types
    model._prepare_generation_inputs = types.MethodType(patched_prepare, model)

    try:
        chunks = []
        sr     = 24000
        t0     = time.time()
        for result in model.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=LANG,
            instruct="INJECTED",   # placeholder — won't be used by our patch
            temperature=TEMP,
        ):
            chunks.append(np.array(result.audio))
            sr = result.sample_rate
        latency = round(time.time() - t0, 3)
        save_wav(out_path, chunks, sr)
    finally:
        # Always restore
        model._prepare_generation_inputs = original_prepare

    metrics = analyze_audio(str(out_path))
    metrics["latency"] = latency
    return metrics


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    from mlx_audio.tts import load as mlx_load

    report = {
        "timestamp":    time.strftime("%Y-%m-%d %H:%M:%S"),
        "goal":         "Test direct instruct_embed delta injection for emotion control.",
        "method":       "neutral + beta*(emotion-neutral) shift applied to instruct_embed mean",
        "custom_model": CUSTOM_ID,
        "base_model":   BASE_ID,
        "ref_audio":    REF_AUDIO,
        "speaker":      TARGET_SPEAKER,
        "text":         TEST_TEXT,
        "betas":        BETAS,
        "alphas":       ALPHAS,
        "temperature":  TEMP,
        "rows":         [],
        "verdicts":     [],
    }

    # ── Phase 1: ref centroid ──────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("PHASE 1 | Build reference centroid")
    log.info("=" * 60)
    base = mlx_load(BASE_ID)
    ref_centroid, _ = build_ref_centroid(base, REF_AUDIO, TOP_K)
    del base; clean()

    rows = []
    total = len(ALPHAS) * len(EMOTIONS) * (1 + (len(EMOTIONS)-1) * len(BETAS))
    # actually: for neutral no beta sweep; for others beta sweep
    total = len(ALPHAS) * (1 + (len(EMOTIONS)-1) * len(BETAS))
    count = 0

    for alpha in ALPHAS:
        log.info(f"\n{'='*60}")
        log.info(f"ALPHA = {alpha}")
        log.info(f"{'='*60}")

        # ── Load model once per alpha ──────────────────────────────────────
        custom = mlx_load(CUSTOM_ID)
        cos_spk = patch_speaker(custom, TARGET_SPEAKER, ref_centroid, alpha)
        log.info(f"Speaker patched: cos(final,ref)={cos_spk:.5f}")

        # ── Extract neutral embed (once) ───────────────────────────────────
        neutral_instruct = EMOTIONS[0]["instruct"]
        neutral_embed    = get_instruct_embed(custom, neutral_instruct)
        log.info(f"Neutral embed shape: {neutral_embed.shape}  norm={norm_val(neutral_embed):.4f}")

        # ── Baseline: beta=1.0 with neutral instruct (original method) ────
        count += 1
        log.info(f"\n[{count}] BASELINE neutral / alpha={alpha}")
        neutral_path = WAV / f"a{alpha:.1f}_neutral_baseline.wav"

        # For baseline: inject neutral_embed at beta=1.0 (no change)
        metrics = generate_with_patched_instruct(
            custom, TEST_TEXT, TARGET_SPEAKER,
            neutral_embed, neutral_embed, 1.0, neutral_path
        )

        base_model = mlx_load(BASE_ID)
        gen_emb, _ = reencode(base_model, neutral_path)
        cos_ref    = cosine(ref_centroid, gen_emb)
        del base_model; clean()

        baseline_row = {
            "alpha":    alpha,
            "emotion":  "neutral",
            "beta":     1.0,
            "label":    "baseline",
            "file":     str(neutral_path),
            "cos_ref":  round(cos_ref, 5),
            **metrics,
        }
        rows.append(baseline_row)
        log.info(f"  Baseline: cos_ref={cos_ref:.5f}  f0={metrics.get('f0_mean',0):.1f}  centroid={metrics.get('centroid',0):.0f}")

        # ── Beta sweep per emotion ─────────────────────────────────────────
        for emo in EMOTIONS:
            if emo["is_baseline"]:
                continue

            emotion_embed = get_instruct_embed(custom, emo["instruct"])
            log.info(f"\n  Emotion: {emo['name']}  embed_norm={norm_val(emotion_embed):.4f}")

            # Cosine between neutral and emotion embed means
            cos_n_e = cosine(neutral_embed.mean(axis=1).squeeze(),
                             emotion_embed.mean(axis=1).squeeze())
            log.info(f"  cos(neutral_mean, emotion_mean) = {cos_n_e:.5f}")

            for beta in BETAS:
                count += 1
                log.info(f"  [{count}] alpha={alpha} emo={emo['name']} beta={beta:.1f}")

                fname   = WAV / f"a{alpha:.1f}_{emo['name']}_b{beta:.2f}.wav"
                metrics = generate_with_patched_instruct(
                    custom, TEST_TEXT, TARGET_SPEAKER,
                    neutral_embed, emotion_embed, beta, fname
                )

                base_model = mlx_load(BASE_ID)
                gen_emb, _ = reencode(base_model, fname)
                cos_ref    = cosine(ref_centroid, gen_emb)
                del base_model; clean()

                row = {
                    "alpha":       alpha,
                    "emotion":     emo["name"],
                    "beta":        beta,
                    "label":       f"a{alpha:.1f}_{emo['name']}_b{beta:.2f}",
                    "file":        str(fname),
                    "cos_ref":     round(cos_ref, 5),
                    "cos_ne":      round(cos_n_e, 5),
                    **metrics,
                }
                rows.append(row)

                log.info(
                    f"    cos_ref={cos_ref:.5f}  "
                    f"f0={metrics.get('f0_mean',0):.1f}Hz  "
                    f"centroid={metrics.get('centroid',0):.0f}Hz  "
                    f"silence={metrics.get('silence',0):.3f}"
                )

        del custom; clean()

    report["rows"] = rows

    # ── Verdicts ──────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("PHASE FINAL | Verdicts")
    log.info("=" * 60)

    verdicts = []

    # V1: Does beta=0.0 sound like neutral? (f0 and centroid close to baseline)
    for alpha in ALPHAS:
        baseline = next((r for r in rows if r["alpha"]==alpha and r["emotion"]=="neutral"), None)
        if not baseline: continue
        for emo_name in set(r["emotion"] for r in rows if r["emotion"]!="neutral"):
            beta0   = next((r for r in rows if r["alpha"]==alpha and r["emotion"]==emo_name and r["beta"]==0.0), None)
            beta1   = next((r for r in rows if r["alpha"]==alpha and r["emotion"]==emo_name and r["beta"]==1.0), None)
            beta_hi = next((r for r in rows if r["alpha"]==alpha and r["emotion"]==emo_name and r["beta"]==1.6), None)
            if not (beta0 and beta1): continue

            f0_drift_b0 = abs(beta0.get("f0_mean",0) - baseline.get("f0_mean",0))
            f0_drift_b1 = abs(beta1.get("f0_mean",0) - baseline.get("f0_mean",0))
            timbre_loss = baseline["cos_ref"] - beta1["cos_ref"]

            if f0_drift_b0 < 10 and f0_drift_b1 > 15:
                verdicts.append(
                    f"✅ BETA KONTROL [{emo_name}/α={alpha}]: beta=0 neutral gibi (Δf0={f0_drift_b0:.1f}Hz), "
                    f"beta=1.0 farklı (Δf0={f0_drift_b1:.1f}Hz)"
                )
            elif f0_drift_b1 > 10:
                verdicts.append(
                    f"⚠️  BETA KISMI [{emo_name}/α={alpha}]: beta=1.0'da Δf0={f0_drift_b1:.1f}Hz var ama beta=0 tam neutral değil"
                )
            else:
                verdicts.append(
                    f"❌ BETA ETKISIZ [{emo_name}/α={alpha}]: beta değişiyor ama f0 değişmiyor (Δ={f0_drift_b1:.1f}Hz)"
                )

            if timbre_loss < 0.01:
                verdicts.append(f"✅ TİMBRE KORUNUYOR [{emo_name}/α={alpha}]: loss={timbre_loss:.5f}")
            elif timbre_loss < 0.03:
                verdicts.append(f"⚠️  TİMBRE KAYBI ORTA [{emo_name}/α={alpha}]: loss={timbre_loss:.5f}")
            else:
                verdicts.append(f"❌ TİMBRE KAYBI YÜKSEK [{emo_name}/α={alpha}]: loss={timbre_loss:.5f}")

    report["verdicts"] = verdicts

    # ── Save ──────────────────────────────────────────────────────────────────
    json_path = OUT / "instruct_delta_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md_path = OUT / "instruct_delta_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Instruct Delta Injection — Results\n\n")
        f.write(f"**Timestamp:** {report['timestamp']}\n\n")
        f.write("## Verdicts\n\n")
        for v in verdicts: f.write(f"- {v}\n")
        f.write("\n## Beta Sweep Table\n\n")
        f.write("| Alpha | Emotion | Beta | cos_ref | f0_mean | centroid | silence |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['alpha']} | {r['emotion']} | {r['beta']} | "
                f"{r['cos_ref']} | {r.get('f0_mean',0):.1f} | "
                f"{r.get('centroid',0):.0f} | {r.get('silence',0):.3f} |\n"
            )
        f.write("\n## Listen Order (beta 0.0 → 1.6)\n\n")
        for emo_name in set(r["emotion"] for r in rows if r["emotion"]!="neutral"):
            f.write(f"### {emo_name}\n")
            emo_rows = sorted([r for r in rows if r["emotion"]==emo_name and r["alpha"]==1.0],
                              key=lambda x: x["beta"])
            for r in emo_rows:
                f.write(f"- beta={r['beta']:.2f}: `{r['file']}`  cos={r['cos_ref']}  f0={r.get('f0_mean',0):.1f}Hz\n")
            f.write("\n")

    # Terminal
    print("\n" + "=" * 60)
    print("INSTRUCT DELTA INJECTION — RESULTS")
    print("=" * 60)
    print(f"\n{'Alpha':>6}  {'Emotion':>12}  {'Beta':>5}  {'cos_ref':>9}  {'f0':>7}  {'centroid':>9}")
    print(f"{'------':>6}  {'-------':>12}  {'----':>5}  {'-------':>9}  {'--':>7}  {'--------':>9}")
    for r in rows:
        if r["alpha"] == 1.0:
            print(
                f"{r['alpha']:>6.1f}  {r['emotion']:>12}  {r['beta']:>5.2f}  "
                f"{r['cos_ref']:>9.5f}  {r.get('f0_mean',0):>7.1f}  {r.get('centroid',0):>9.0f}"
            )
    print("\nVERDICTS:")
    for v in verdicts: print(f"  {v}")
    print(f"\nJSON:  {json_path}")
    print(f"MD:    {md_path}")
    print(f"WAV:   {WAV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
