# Copyright (c) 2026 Efeberk Çeker
# SPDX-License-Identifier: AGPL-3.0-only

#!/usr/bin/env python3
"""
Speaker Space Interpolation V12
===============================
Amaç:
  1. Alpha sweep ile en iyi adaptasyon noktasını bulmak
  2. Token-local adaptation kanıtı: Ryan/Serena/Dylan ayrı ayrı ref'e SLERP edilir
  3. SECS + duration + RMS + peak + zero-crossing ile kalite kontrolü
  4. Emotion preservation testi: neutral/sad/happy/angry/whisper/deep
  5. V11'de başarısız olan VoiceDirection yerine doğru deney: each token -> ref centroid
"""

import os, sys, gc, json, logging
from pathlib import Path
import numpy as np
import soundfile as sf
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("speaker_space_v12")

CUSTOM_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"
BASE_ID   = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"

OUT = Path("pipeline_work/test_outputs/speaker_space_interpolation_v12")
WAV = OUT / "wav"
OUT.mkdir(parents=True, exist_ok=True)
WAV.mkdir(parents=True, exist_ok=True)

REF_AUDIO = "pipeline_work/12345.mp3"

TEST_TEXT = (
    "I can't believe you actually did that. "
    "After all the promises we made, you just threw it all away like it meant nothing. "
    "But you know what? I'm not even angry anymore. I'm just disappointed."
)

LANG = "English"
SEED = 42
TOP_K = 5

TARGET_SPEAKERS = ["ryan", "serena", "dylan"]
ALPHAS = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0]

EMOTIONS = [
    {
        "name": "neutral",
        "instruct": "Speak naturally with energy and forward momentum, like a confident person talking directly to someone.",
        "temp": 0.60,
    },
    {
        "name": "sad",
        "instruct": "Speak with a somber and melancholic tone. Keep normal volume, clear pronunciation, and do not pause too long between sentences. Maintain a steady rhythm.",
        "temp": 0.65,
    },
    {
        "name": "happy",
        "instruct": "Speak with a very happy and cheerful tone, smiling while speaking.",
        "temp": 0.65,
    },
    {
        "name": "angry",
        "instruct": "Speak with an angry and frustrated tone, raising your volume slightly but keeping words sharp.",
        "temp": 0.60,
    },
    {
        "name": "whisper",
        "instruct": "Speak in a quiet whisper, but maintain clear articulation of every word.",
        "temp": 0.55,
    },
    {
        "name": "deep",
        "instruct": "Speak with a deep and resonant vocal register, maintaining a steady pace.",
        "temp": 0.50,
    },
]


def clean():
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass


def cosine(a, b):
    a = a.astype(mx.float32).reshape(-1)
    b = b.astype(mx.float32).reshape(-1)
    return float(mx.sum(a * b) / (mx.linalg.norm(a) * mx.linalg.norm(b) + 1e-8))


def slerp(v1, v2, t, DOT_THRESHOLD=0.9995):
    t = float(t)
    v1 = v1.astype(mx.float32).reshape(-1)
    v2 = v2.astype(mx.float32).reshape(-1)

    v1_norm = v1 / (mx.linalg.norm(v1) + 1e-8)
    v2_norm = v2 / (mx.linalg.norm(v2) + 1e-8)

    dot = mx.sum(v1_norm * v2_norm)
    dot = mx.clip(dot, -1.0, 1.0)

    if mx.abs(dot) > DOT_THRESHOLD:
        return (1.0 - t) * v1 + t * v2

    theta_0 = mx.arccos(dot)
    sin_theta_0 = mx.sin(theta_0)

    theta_t = theta_0 * t
    s0 = mx.sin(theta_0 - theta_t) / sin_theta_0
    s1 = mx.sin(theta_t) / sin_theta_0

    return s0 * v1 + s1 * v2


def norm_match(vec, target_norm):
    return vec * (target_norm / (mx.linalg.norm(vec) + 1e-8))


def get_speaker_vec(model, speaker):
    config = model.config.talker_config
    sid = config.spk_id[speaker]
    emb = model.talker.get_input_embeddings()
    return mx.array(emb.weight[sid]).squeeze()


def patch_speaker_vec(model, speaker, new_vec):
    config = model.config.talker_config
    sid = config.spk_id[speaker]
    emb = model.talker.get_input_embeddings()

    original = mx.array(emb.weight[sid]).squeeze()
    matched = norm_match(new_vec, mx.linalg.norm(original))

    emb.weight[sid] = matched.reshape(original.shape).astype(original.dtype)
    mx.eval(emb.weight)

    return {
        "speaker": speaker,
        "sid": int(sid),
        "orig_norm": round(float(mx.linalg.norm(original)), 5),
        "patched_norm": round(float(mx.linalg.norm(matched)), 5),
        "cos_orig_patch": round(cosine(original, matched), 5),
    }


def build_weighted_centroid(base_model, audio_path, top_k=5):
    from mlx_audio.utils import load_audio
    from finetune import segment_audio

    sr = base_model.sample_rate
    audio = np.array(load_audio(str(audio_path), sample_rate=sr))
    segments = segment_audio(audio, sr, min_sec=3.0, max_sec=8.0)

    if not segments:
        segments = [audio]

    embs = [
        base_model.extract_speaker_embedding(mx.array(seg), sr=sr).squeeze()
        for seg in segments
    ]

    all_avg = mx.stack(embs).mean(axis=0).squeeze()

    scores = []
    for i, e1 in enumerate(embs):
        others = [cosine(e1, e2) for j, e2 in enumerate(embs) if j != i]
        avg_cos = float(np.mean(others)) if others else 1.0
        cos_avg = cosine(e1, all_avg)
        combined = avg_cos * 0.6 + cos_avg * 0.4
        scores.append(combined)

    scores_np = np.array(scores, dtype=np.float32)
    weights = np.exp(scores_np / 0.1)
    weights = weights / np.sum(weights)

    top_indices = np.argsort(scores_np)[-top_k:]

    centroid = np.zeros_like(np.array(embs[0]))
    used = []

    for idx in top_indices:
        centroid += weights[idx] * np.array(embs[idx])
        used.append({
            "segment_index": int(idx),
            "score": round(float(scores_np[idx]), 5),
            "weight": round(float(weights[idx]), 5),
        })

    return mx.array(centroid), {
        "num_segments": len(segments),
        "top_k_used": used,
        "score_min": round(float(np.min(scores_np)), 5),
        "score_max": round(float(np.max(scores_np)), 5),
        "score_mean": round(float(np.mean(scores_np)), 5),
    }


def audio_metrics(path):
    wav, sr = sf.read(str(path), dtype="float32")

    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    duration = len(wav) / sr if sr else 0.0
    peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    rms = float(np.sqrt(np.mean(wav ** 2))) if len(wav) else 0.0

    signs = np.sign(wav)
    zcr = float(np.mean(signs[:-1] != signs[1:])) if len(wav) > 1 else 0.0

    silent_ratio = float(np.mean(np.abs(wav) < 1e-4)) if len(wav) else 1.0
    clipped_ratio = float(np.mean(np.abs(wav) > 0.99)) if len(wav) else 0.0

    return {
        "duration_sec": round(duration, 3),
        "peak": round(peak, 6),
        "rms": round(rms, 6),
        "zcr": round(zcr, 6),
        "silent_ratio": round(silent_ratio, 6),
        "clipped_ratio": round(clipped_ratio, 6),
        "sr": int(sr),
    }


def generate(model, speaker, text, instruct, temp, out_path):
    np.random.seed(SEED)
    mx.random.seed(SEED)

    chunks = []
    sr = 24000

    for result in model.generate_custom_voice(
        text=text,
        speaker=speaker,
        language=LANG,
        instruct=instruct,
        temperature=temp,
    ):
        chunks.append(np.array(result.audio))
        sr = result.sample_rate

    wav = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(1, dtype=np.float32)

    if np.isnan(wav).any() or np.isinf(wav).any():
        raise RuntimeError(f"Invalid wav generated: {out_path}")

    peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    if peak > 1.0:
        wav = wav / peak

    sf.write(str(out_path), wav, sr)
    return audio_metrics(out_path)


def evaluate_secs(base_model, generated_wav_path, ref_centroid):
    from mlx_audio.utils import load_audio
    from finetune import segment_audio

    sr = base_model.sample_rate
    audio = np.array(load_audio(str(generated_wav_path), sample_rate=sr))
    segments = segment_audio(audio, sr, min_sec=3.0, max_sec=8.0)

    if not segments:
        segments = [audio]

    embs = [
        base_model.extract_speaker_embedding(mx.array(seg), sr=sr).squeeze()
        for seg in segments
    ]

    gen_centroid = mx.stack(embs).mean(axis=0).squeeze()
    return cosine(ref_centroid, gen_centroid)


def score_candidate(secs, metrics):
    """
    Basit kalite skoru:
    - speaker similarity yüksek olsun
    - sessizlik çok olmasın
    - clipping olmasın
    - duration çok kısa olmasın
    """
    duration_penalty = 0.0
    if metrics["duration_sec"] < 3.0:
        duration_penalty += 0.10
    if metrics["duration_sec"] < 1.5:
        duration_penalty += 0.20

    silence_penalty = min(metrics["silent_ratio"] * 0.20, 0.20)
    clip_penalty = min(metrics["clipped_ratio"] * 2.0, 0.20)

    return round(float(secs - duration_penalty - silence_penalty - clip_penalty), 6)


def main():
    from mlx_audio.tts import load as mlx_load

    report = {
        "version": "v12",
        "purpose": "Alpha sweep, token-local adaptation, and emotion preservation validation",
        "config": {
            "custom_model": CUSTOM_ID,
            "base_model": BASE_ID,
            "ref_audio": REF_AUDIO,
            "target_speakers": TARGET_SPEAKERS,
            "alphas": ALPHAS,
            "top_k": TOP_K,
            "seed": SEED,
        },
        "reference": {},
        "alpha_sweep": {},
        "best_alpha": {},
        "emotion_preservation": {},
    }

    log.info("Loading base model...")
    base = mlx_load(BASE_ID)

    log.info("Building weighted reference centroid...")
    ref_centroid, centroid_info = build_weighted_centroid(base, REF_AUDIO, top_k=TOP_K)

    report["reference"] = {
        "centroid_norm": round(float(mx.linalg.norm(ref_centroid)), 5),
        "centroid_info": centroid_info,
    }

    log.info("Loading custom model...")
    custom = mlx_load(CUSTOM_ID)

    original_vectors = {
        spk: get_speaker_vec(custom, spk)
        for spk in TARGET_SPEAKERS
    }

    neutral_instruct = EMOTIONS[0]["instruct"]

    for speaker in TARGET_SPEAKERS:
        log.info(f"\n=== Alpha sweep for {speaker} ===")

        orig_vec = original_vectors[speaker]
        orig_norm = float(mx.linalg.norm(orig_vec))

        rows = []

        for alpha in ALPHAS:
            mixed = slerp(orig_vec, ref_centroid, alpha)
            patch_info = patch_speaker_vec(custom, speaker, mixed)

            out_path = WAV / f"sweep_{speaker}_alpha_{alpha:.2f}.wav"

            metrics = generate(
                custom,
                speaker,
                TEST_TEXT,
                neutral_instruct,
                0.60,
                out_path,
            )

            secs = evaluate_secs(base, out_path, ref_centroid)
            total_score = score_candidate(secs, metrics)

            row = {
                "speaker": speaker,
                "alpha": round(float(alpha), 3),
                "pre_patch_cos_token_ref": round(cosine(mixed, ref_centroid), 5),
                "patch_info": patch_info,
                "secs_ref": round(float(secs), 5),
                "quality_score": total_score,
                "audio_metrics": metrics,
                "file": str(out_path),
            }

            rows.append(row)

            log.info(
                f"{speaker} alpha={alpha:.2f} "
                f"SECS={secs:.5f} score={total_score:.5f} "
                f"dur={metrics['duration_sec']:.2f}s rms={metrics['rms']:.5f}"
            )

        report["alpha_sweep"][speaker] = rows

        best = max(rows, key=lambda x: x["quality_score"])
        report["best_alpha"][speaker] = best

    log.info("\n=== Emotion preservation test ===")

    for speaker in TARGET_SPEAKERS:
        best_alpha = report["best_alpha"][speaker]["alpha"]
        orig_vec = original_vectors[speaker]
        mixed = slerp(orig_vec, ref_centroid, best_alpha)
        patch_speaker_vec(custom, speaker, mixed)

        emotion_rows = []

        for emo in EMOTIONS:
            out_path = WAV / f"emotion_{speaker}_alpha_{best_alpha:.2f}_{emo['name']}.wav"

            metrics = generate(
                custom,
                speaker,
                TEST_TEXT,
                emo["instruct"],
                emo["temp"],
                out_path,
            )

            secs = evaluate_secs(base, out_path, ref_centroid)
            total_score = score_candidate(secs, metrics)

            row = {
                "speaker": speaker,
                "alpha": best_alpha,
                "emotion": emo["name"],
                "temperature": emo["temp"],
                "instruct": emo["instruct"],
                "secs_ref": round(float(secs), 5),
                "quality_score": total_score,
                "audio_metrics": metrics,
                "file": str(out_path),
            }

            emotion_rows.append(row)

            log.info(
                f"{speaker} emotion={emo['name']} "
                f"SECS={secs:.5f} score={total_score:.5f} "
                f"dur={metrics['duration_sec']:.2f}s"
            )

        report["emotion_preservation"][speaker] = emotion_rows

    json_path = OUT / "v12_report.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    del base
    del custom
    clean()

    log.info(f"\nV12 report saved: {json_path}")


if __name__ == "__main__":
    main()