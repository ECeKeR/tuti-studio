# Copyright (c) 2026 Efeberk Çeker
# SPDX-License-Identifier: AGPL-3.0-only

#!/usr/bin/env python3
"""
Speaker Space Interpolation V10 (Sad Hallucination Fix)
========================================================
Gelişmeler:
  1. "Sad" Sussyışı Çözüldü:
     - "sad" kelimesi yerine "somber and melancholic" kullanıldı (Model enerjisini kaybetmez).
     - Instruct'a "do not pause too long, maintain a steady rhythm" eklendi.
     - Temperature 0.55'ten 0.65'e çıkarıldı (Kelimeleri yutmasını ve susmasını engeller).
  2. Diğer tüm duygular (V9) korundu.
"""

import os, sys, gc, json, time, logging
from pathlib import Path
import numpy as np
import soundfile as sf
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("speaker_space_v10")

# ── Config ────────────────────────────────────────────────────────────────────
CUSTOM_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"
BASE_ID   = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"

OUT = Path("pipeline_work/test_outputs/speaker_space_interpolation_v10")
WAV = OUT / "wav"
OUT.mkdir(parents=True, exist_ok=True)
WAV.mkdir(parents=True, exist_ok=True)

REF_AUDIO = "pipeline_work/12345.mp3" # Kendi sesin

TEST_TEXT = "I can't believe you actually did that. After all the promises we made, you just threw it all away like it meant nothing. But you know what? I'm not even angry anymore. I'm just disappointed."

# Sadece Sad duygusu güncellendi. Diğerleri V9 ile aynı (korumalı).
EMOTIONS = [
    {
        "name": "neutral",
        "instruct": "Speak naturally with energy and forward momentum, like a confident person talking directly to someone.",
        "temp": 0.6
    },
    {
        "name": "sad",
        "instruct": "Speak with a somber and melancholic tone. Keep normal volume, clear pronunciation, and do not pause too long between sentences. Maintain a steady rhythm.",
        "temp": 0.65  # 0.55'ten 0.65'e çıkarıldı (Susmayı önler)
    },
    {
        "name": "happy",
        "instruct": "Speak with a very happy and cheerful tone, smiling while speaking.",
        "temp": 0.65
    },
    {
        "name": "angry",
        "instruct": "Speak with an angry and frustrated tone, raising your volume slightly but keeping words sharp.",
        "temp": 0.6
    },
    {
        "name": "whisper",
        "instruct": "Speak in a quiet whisper, but maintain clear articulation of every word.",
        "temp": 0.55
    },
    {
        "name": "slow",
        "instruct": "Speak at a slightly slow and deliberate pace, enunciating each word clearly.",
        "temp": 0.55
    },
    {
        "name": "deep",
        "instruct": "Speak with a deep and resonant vocal register, maintaining a steady pace.",
        "temp": 0.5
    }
]

SEED = 42
LANG = "English"
TARGET_SPEAKER = "ryan"
ALPHA = 1.0
TOP_K = 5

# ── Math & Helpers ────────────────────────────────────────────────────────────

def cosine(a, b):
    a = a.astype(mx.float32).reshape(-1)
    b = b.astype(mx.float32).reshape(-1)
    return float(mx.sum(a * b) / (mx.linalg.norm(a) * mx.linalg.norm(b) + 1e-8))

def slerp(v1, v2, t, DOT_THRESHOLD=0.9995):
    v1 = v1.astype(mx.float32).reshape(-1)
    v2 = v2.astype(mx.float32).reshape(-1)
    v1_norm = v1 / (mx.linalg.norm(v1) + 1e-8)
    v2_norm = v2 / (mx.linalg.norm(v2) + 1e-8)
    dot = mx.sum(v1_norm * v2_norm)
    
    if mx.abs(dot) > DOT_THRESHOLD:
        return (1.0 - t) * v1 + t * v2
        
    theta_0 = mx.arccos(dot)
    sin_theta_0 = mx.sin(theta_0)
    theta_t = theta_0 * t
    sin_theta_t = mx.sin(theta_t)
    
    s0 = mx.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    
    res = s0 * v1 + s1 * v2
    return res

def clean():
    gc.collect()
    try: mx.clear_cache()
    except: pass

def patch_token_standard(model, speaker, new_vec):
    config    = model.config.talker_config
    emb_layer = model.talker.get_input_embeddings()
    sid       = config.spk_id[speaker]
    original  = mx.array(emb_layer.weight[sid]).squeeze()

    orig_norm = mx.linalg.norm(original)
    matched   = new_vec * (orig_norm / (mx.linalg.norm(new_vec) + 1e-8))

    emb_layer.weight[sid] = matched.reshape(original.shape).astype(original.dtype)
    mx.eval(emb_layer.weight)
    return {"orig_norm": round(float(orig_norm), 4), "cos_sim": round(cosine(original, matched), 5)}

def build_weighted_centroid(base_model, audio_path, top_k=5):
    from mlx_audio.utils import load_audio
    from finetune import segment_audio

    sr  = base_model.sample_rate
    audio = np.array(load_audio(str(audio_path), sample_rate=sr))
    segments = segment_audio(audio, sr, min_sec=3.0, max_sec=8.0)
    if not segments: segments = [audio]

    embs = [base_model.extract_speaker_embedding(mx.array(s), sr=sr).squeeze() for s in segments]
    all_avg = mx.stack(embs).mean(axis=0).squeeze()

    scores = []
    for i, e1 in enumerate(embs):
        others = [cosine(e1, e2) for j, e2 in enumerate(embs) if j != i]
        avg_cos = float(np.mean(others)) if others else 1.0
        cos_avg = cosine(e1, all_avg)
        combined = avg_cos * 0.6 + cos_avg * 0.4
        scores.append(combined)
        
    scores = np.array(scores)
    weights = np.exp(scores / 0.1)
    weights = weights / np.sum(weights)
    
    top_indices = np.argsort(scores)[-top_k:]
    centroid = np.zeros_like(embs[0])
    for idx in top_indices:
        centroid += weights[idx] * np.array(embs[idx])
        
    centroid = mx.array(centroid)
    return centroid

def generate(model, speaker, text, instruct, temp, out_path):
    np.random.seed(SEED)
    mx.random.seed(SEED)
    chunks = []
    sr = 24000
    
    for result in model.generate_custom_voice(
        text=text, speaker=speaker, language=LANG, instruct=instruct, temperature=temp
    ):
        chunks.append(np.array(result.audio))
        sr = result.sample_rate
    
    wav = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(1, dtype=np.float32)
    log.info(f"Generated wav shape: {wav.shape}, dtype: {wav.dtype}, sr: {sr}")
    log.info(f"Any NaNs: {np.isnan(wav).any()}, Any Infs: {np.isinf(wav).any()}")
    peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    log.info(f"Wav peak value: {peak}")
    if peak > 1.0: wav = wav / peak
    try:
        sf.write(str(out_path), wav, sr)
    except Exception as e:
        log.error(f"Failed to write wav file. wav is nan/inf: {np.isnan(wav).any() or np.isinf(wav).any()}, shape: {wav.shape}, sr: {sr}")
        raise e

# ── Main Execution ────────────────────────────────────────────────────────────

def main():
    from mlx_audio.tts import load as mlx_load

    log.info("Building Weighted Centroid...")
    base = mlx_load(BASE_ID)
    ref_centroid = build_weighted_centroid(base, REF_AUDIO, top_k=TOP_K)
    del base; clean()

    log.info("Loading CustomVoice model...")
    custom = mlx_load(CUSTOM_ID)
    config = custom.config.talker_config
    sid = config.spk_id[TARGET_SPEAKER]
    target_vec = mx.array(custom.talker.get_input_embeddings().weight[sid]).squeeze()

    log.info(f"\n{'='*50}\nV10 Sad Fix Test | Target: {TARGET_SPEAKER} | Alpha {ALPHA:.2f}\n{'='*50}")
    
    mixed_vec = slerp(target_vec, ref_centroid, ALPHA)
    patch_info = patch_token_standard(custom, TARGET_SPEAKER, mixed_vec)
    log.info(f"Patch Info: {patch_info}")
    
    for emo in EMOTIONS:
        log.info(f"Generating Emotion: {emo['name']} (Temp: {emo['temp']})")
        out_path = WAV / f"V10_ryan_{emo['name']}.wav"
        generate(custom, TARGET_SPEAKER, TEST_TEXT, emo['instruct'], emo['temp'], out_path)
        log.info(f"Saved: {out_path}")

    log.info("\nV10 Test complete! Sad should no longer cut off or breathe endlessly.")

if __name__ == "__main__":
    main()