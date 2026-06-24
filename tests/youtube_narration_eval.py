# Copyright (c) 2026 Efeberk Çeker
# SPDX-License-Identifier: AGPL-3.0-only

#!/usr/bin/env python3
"""
YouTube Test Script V3 — s2 Refine (Hook + Hallüsinasyon Düzeltme)
====================================================================

V2 sonucu: s2_7blocks en iyi, iki sorun:
  1. hook çok cansız → "low energy" instruction'ı modeli öldürdü.
  2. explain(82kw→36sn) + proud(98kw→64.8sn) → hallüsinasyon (kw/sn > 0.40 = tehlikeli).

V3 yaklaşımı (minimum değişiklik, s2 tabanlı):
  - hook: "low energy" → ENERJİK + MERAK (hook = izleyiciyi yakala, öldürme).
  - 50 KELİME EŞIĞI: hiçbir parça 50 kw'i geçemez (hallüsinasyon sınırı).
  - s2'nin iyi çalışan emotion kütüphanesi (curious/confide/excited/reflective) korundu.

3 varyasyon dener — sadece düzeltme odaklı:
  A: hook_fixin_B + explain 2'ye böl + proud 2'ye böl  (8 parça)
  B: hook_fixin_C (alternatif) + aynı bölme            (8 parça)
  C: hook_fixin_B + explain 3'e böl + proud 3'e böl    (10 parça, en güvenli)

Çıktı: pipeline_work/youtube_test_v3/
"""

import os, sys, gc, time
import numpy as np
import soundfile as sf
from pathlib import Path
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

# ── v10 AYARLARI ──────────────────────────────────────────────────────────────
CUSTOM_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"
BASE_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
REF_AUDIO = "pipeline_work/12345.mp3"
SEED = 42
LANG = "English"
TARGET_SPEAKER = "ryan"
ALPHA = 1.0
TOP_K = 5

OUT = Path("pipeline_work/youtube_test_v3")
OUT.mkdir(parents=True, exist_ok=True)

# 50 kw eşik — v2 verisinden türetilen hallüsinasyon sınırı
MAX_KW = 50

# ── INSTRUCTION KÜTÜPHANESİ (v2'nin iyileri + hook düzeltmeleri) ──────────────
INSTR = {
    # V2 hook (ÖLDÜ): "Speak with low energy, quiet curiosity..." → cansız.
    # V3 hook varyantları: ENERJİK ama gizemli. İzleyiciyi yakala.
    "hook_B":     ("Speak with bright curiosity and a teasing hook. Lean in, pull the viewer in.", 0.65),
    "hook_C":     ("Speak with warm energy and playful mystery. Make them want to know what happens.", 0.68),

    # v2'nin doğrulanmış iyileri (olduğu gibi korundu)
    "curious":    ("Speak with genuine curiosity and rising intrigue, as if truly puzzled.", 0.65),
    "confide":    ("Speak softly and candidly, like confiding a personal realization.", 0.6),
    "excited":    ("Speak with energetic excitement, fast and punchy, like revealing a trick.", 0.7),
    "reflective": ("Speak thoughtfully and deliberately, weighing each word with care.", 0.58),
    "explain":    ("Speak in an energetic, engaging YouTuber voiceover. Crisp, forward momentum.", 0.62),
    "reveal":     ("Speak as if unveiling something exciting. Bright, building energy.", 0.68),
    "proud":      ("Speak with proud confidence and a playful grin. Slow the reveals.", 0.62),
    "story":      ("Speak naturally as a storyteller, warm and engaged with the tale.", 0.62),
    "close":      ("Speak with warm satisfaction, a grin audible in your voice. Land it.", 0.6),
}


# ── STRATEJİLER (s2 tabanlı + düzeltmeler) ────────────────────────────────────

STRATEGIES = {

    # ── A: hook_B + excited 2'ye + reflective 2'ye + explain 2'ye + proud 2'ye ─
    "A_hookB_split2": [
        ("About two weeks ago, I posted a YouTube video... and nobody watched it.",
         "hook_B"),
        ("What could have gone wrong? Was the product unappealing? The story? The editing? "
         "Or... was it the voice?",
         "curious"),
        ("Like most people, I use AI voiceovers, and yes... standard quality just doesn't cut it. "
         "But how do some people succeed so easily with AI voices?",
         "confide"),
        # excited 53kw → 2'ye böl (29/24 kw)
        ("One of the best examples is Isac. His method? Go to ElevenLabs, type the script, generate 3 free takes. "
         "Take the best parts, stitch them together, adjust the speed in Audacity, and boom... perfect voice.",
         "excited"),
        ("It works great. But there's a catch.",
         "dramatic"),
        # reflective 53kw → 2'ye böl (31/22 kw)
        ("I can't use ElevenLabs. I started this channel to showcase local AI projects. "
         "If I use a cloud model right out of the gate, what's the point?",
         "reflective"),
        ("Also... Isac's method is great, but I am too lazy for that. "
         "Instead, I need to build an app that does this entire process for me.",
         "confide"),
        # explain 82kw → 2'ye böl (44/38 kw) — duygu akışı: metod anlatımı
        ("I jumped straight into my open-source model. Took our Chinese princess, Qwen3-TTS. "
         "Had Ollama split the script into optimal segments. "
         "For each segment, I generated 3 to 5 takes, playing around with the seed values to get different variations.",
         "explain"),
        ("Then, I had Whisper analyze them to pick the best pronunciation and pacing. "
         "Finally, a custom stitcher merged them into the final, listenable audio file. "
         "It still has some flaws, but who doesn't? I ran thousands of tests to find the perfect pipeline.",
         "explain"),
        # proud 98kw → 3'e böl — duygu akışı: tanıtım → dramatik → son söz
        ("And finally... the final product. His name is Titu. "
         "He can't pronounce his own name, but whatever.",
         "story"),
        ("A storytelling parrot featured in One Thousand and One Nights. Who cares? Moving on. "
         "Here is the final product... Wait, there's a catch.",
         "dramatic"),
        ("We're still using the base model, and that's not good enough for us. "
         "Using SLERP, I went into the model, took my existing cloned voice, encoded it, "
         "and overwrote the custom voice with that encoded data.",
         "explain"),
        ("Alright, that's enough technical details. This is YouTube, not Reddit. "
         "Here's the app, link is in the description. That's it from me.",
         "close"),
    ],

    # ── B: hook_C (alternatif) + aynı bölme ───────────────────────────────────
    "B_hookC_split2": [
        ("About two weeks ago, I posted a YouTube video... and nobody watched it.",
         "hook_C"),
        ("What could have gone wrong? Was the product unappealing? The story? The editing? "
         "Or... was it the voice?",
         "curious"),
        ("Like most people, I use AI voiceovers, and yes... standard quality just doesn't cut it. "
         "But how do some people succeed so easily with AI voices?",
         "confide"),
        ("One of the best examples is Isac. His method? Go to ElevenLabs, type the script, generate 3 free takes. "
         "Take the best parts, stitch them together, adjust the speed in Audacity, and boom... perfect voice.",
         "excited"),
        ("It works great. But there's a catch.",
         "dramatic"),
        ("I can't use ElevenLabs. I started this channel to showcase local AI projects. "
         "If I use a cloud model right out of the gate, what's the point?",
         "reflective"),
        ("Also... Isac's method is great, but I am too lazy for that. "
         "Instead, I need to build an app that does this entire process for me.",
         "confide"),
        ("I jumped straight into my open-source model. Took our Chinese princess, Qwen3-TTS. "
         "Had Ollama split the script into optimal segments. "
         "For each segment, I generated 3 to 5 takes, playing around with the seed values to get different variations.",
         "explain"),
        ("Then, I had Whisper analyze them to pick the best pronunciation and pacing. "
         "Finally, a custom stitcher merged them into the final, listenable audio file. "
         "It still has some flaws, but who doesn't? I ran thousands of tests to find the perfect pipeline.",
         "explain"),
        ("And finally... the final product. His name is Titu. "
         "He can't pronounce his own name, but whatever.",
         "story"),
        ("A storytelling parrot featured in One Thousand and One Nights. Who cares? Moving on. "
         "Here is the final product... Wait, there's a catch.",
         "dramatic"),
        ("We're still using the base model, and that's not good enough for us. "
         "Using SLERP, I went into the model, took my existing cloned voice, encoded it, "
         "and overwrote the custom voice with that encoded data.",
         "explain"),
        ("Alright, that's enough technical details. This is YouTube, not Reddit. "
         "Here's the app, link is in the description. That's it from me.",
         "close"),
    ],

    # ── C: hook_B + excited 2'ye + reflective 2'ye + explain 3'e + proud 3'e ──
    "C_hookB_split3": [
        ("About two weeks ago, I posted a YouTube video... and nobody watched it.",
         "hook_B"),
        ("What could have gone wrong? Was the product unappealing? The story? The editing? "
         "Or... was it the voice?",
         "curious"),
        ("Like most people, I use AI voiceovers, and yes... standard quality just doesn't cut it. "
         "But how do some people succeed so easily with AI voices?",
         "confide"),
        ("One of the best examples is Isac. His method? Go to ElevenLabs, type the script, generate 3 free takes. "
         "Take the best parts, stitch them together, adjust the speed in Audacity, and boom... perfect voice.",
         "excited"),
        ("It works great. But there's a catch.",
         "dramatic"),
        ("I can't use ElevenLabs. I started this channel to showcase local AI projects. "
         "If I use a cloud model right out of the gate, what's the point?",
         "reflective"),
        ("Also... Isac's method is great, but I am too lazy for that. "
         "Instead, I need to build an app that does this entire process for me.",
         "confide"),
        # explain 82kw → 3'e böl (29/29/24 kw)
        ("I jumped straight into my open-source model. Took our Chinese princess, Qwen3-TTS. "
         "Had Ollama split the script into optimal segments.",
         "explain"),
        ("For each segment, I generated 3 to 5 takes, playing around with the seed values to get different variations. "
         "Then, I had Whisper analyze them to pick the best pronunciation and pacing.",
         "explain"),
        ("Finally, a custom stitcher merged them into the final, listenable audio file. "
         "It still has some flaws, but who doesn't? "
         "I ran thousands of tests to find the perfect pipeline.",
         "reveal"),
        # proud 98kw → 3'e böl (34/32/32 kw)
        ("And finally... the final product. His name is Titu. "
         "He can't pronounce his own name, but whatever.",
         "story"),
        ("A storytelling parrot featured in One Thousand and One Nights. Who cares? Moving on. "
         "Here is the final product... Wait, there's a catch.",
         "dramatic"),
        ("We're still using the base model, and that's not good enough for us. "
         "Using SLERP, I went into the model, took my existing cloned voice, encoded it, "
         "and overwrote the custom voice with that encoded data.",
         "explain"),
        ("Alright, that's enough technical details. This is YouTube, not Reddit. "
         "Here's the app, link is in the description. That's it from me.",
         "close"),
    ],
}

# v2'den eksik kalan dramatic instruction'ı ekle (C stratejisi kullanıyor)
INSTR["dramatic"] = ("Speak with sudden dramatic shift, dropping to a hush before the reveal.", 0.6)


# ── v10 MATH & HELPERS (birebir) ──────────────────────────────────────────────
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
    return s0 * v1 + s1 * v2

def clean():
    gc.collect()
    try: mx.clear_cache()
    except: pass

def patch_token_standard(model, speaker, new_vec):
    config = model.config.talker_config
    emb_layer = model.talker.get_input_embeddings()
    sid = config.spk_id[speaker]
    original = mx.array(emb_layer.weight[sid]).squeeze()
    orig_norm = mx.linalg.norm(original)
    matched = new_vec * (orig_norm / (mx.linalg.norm(new_vec) + 1e-8))
    emb_layer.weight[sid] = matched.reshape(original.shape).astype(original.dtype)
    mx.eval(emb_layer.weight)
    return {"orig_norm": round(float(orig_norm), 4),
            "cos_sim": round(cosine(original, matched), 5)}

def build_weighted_centroid(base_model, audio_path, top_k=5):
    from mlx_audio.utils import load_audio
    from finetune import segment_audio
    sr = base_model.sample_rate
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
    return mx.array(centroid)


# ── ÜRETIM ────────────────────────────────────────────────────────────────────
def generate_one(model, text, instruct, temp, out_path):
    np.random.seed(SEED)
    mx.random.seed(SEED)
    chunks = []
    sr = 24000
    t0 = time.time()
    for result in model.generate_custom_voice(
        text=text, speaker=TARGET_SPEAKER, language=LANG,
        instruct=instruct, temperature=temp
    ):
        chunks.append(np.array(result.audio))
        sr = result.sample_rate
    wav = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(1, dtype=np.float32)
    peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    if peak > 1.0: wav = wav / peak
    sf.write(str(out_path), wav, sr)
    dt = time.time() - t0
    dur = len(wav) / sr
    # Hallüsinasyon kontrolü: kw/sn oranı (ideal ~0.27, >0.40 = tehlikeli, >0.50 = kesin loop)
    kw = len(text.split())
    ratio = kw / dur if dur > 0 else 999
    status = "OK"
    if ratio > 0.50: status = "⚠️ HALLÜSINASYON (loop)"
    elif ratio > 0.40: status = "⚠️ riskli"
    elif dur < 1.0: status = "⚠️ çok kısa"
    return {"path": str(out_path), "dur_sec": round(dur, 2),
            "kw_sn": round(ratio, 3), "status": status,
            "temp": temp, "elapsed": round(dt, 1), "words": kw}

def concat_wavs(paths, out_path, gap_sec=0.0):
    arrays, sr = [], 24000
    for p in paths:
        a, sr = sf.read(str(p))
        if a.ndim > 1: a = a[:, 0]
        arrays.append(a.astype(np.float32))
        if gap_sec > 0:
            arrays.append(np.zeros(int(sr * gap_sec), dtype=np.float32))
    out = np.concatenate(arrays)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), out, sr)

def run_strategy(model, name, blocks, base_dir):
    sub = base_dir / name
    sub.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}\n[{name}] {len(blocks)} parça | hook düzeltmesi + bölme\n{'='*70}")
    paths = []
    results = []
    for i, (text, emo_key) in enumerate(blocks):
        instruct, temp = INSTR[emo_key]
        p = sub / f"{i:02d}_{emo_key}.wav"
        kw = len(text.split())
        over = " ⚠️ 50kw'den fazla!" if kw > MAX_KW else ""
        print(f"  [{i+1}/{len(blocks)}] {emo_key:<11} | {kw:>3} kw{over} | '{text[:50]}...'")
        info = generate_one(model, text, instruct, temp, p)
        info["emo"] = emo_key
        info["instruct"] = instruct
        results.append(info)
        paths.append(p)
        print(f"           → {info['dur_sec']}sn | kw/sn={info['kw_sn']} | {info['status']}")
    combined = sub / "combined.wav"
    concat_wavs(paths, combined, gap_sec=0.0)
    print(f"  ✓ Birleşik: {combined}")
    return {"strategy": name, "combined": str(combined), "parts": results}

def write_report(all_results):
    lines = ["YOUTUBE TEST V3 — Refine Raporu", "=" * 70,
             "V2 s2 baz alındı. Düzeltmeler:",
             "  • hook: 'low energy' → enerjik merak (hook_B / hook_C)",
             f"  • 50 kelime eşiği (MAX_KW={MAX_KW}) — hallüsinasyon sınırı",
             "  • explain(82kw) + proud(98kw) bölündü", ""]
    for r in all_results:
        name = r["strategy"]
        if "error" in r:
            lines.append(f"[{name}] HATA: {r['error']}")
            continue
        parts = r["parts"]
        total_dur = sum(p["dur_sec"] for p in parts)
        total_words = sum(p["words"] for p in parts)
        hallu = [p for p in parts if "HALLÜS" in p["status"]]
        risky = [p for p in parts if "riskli" in p["status"]]
        lines.append(f"[{name}] {len(parts)} parça | {total_words} kw | {total_dur:.1f}sn | "
                     f"hallü={len(hallu)} riskli={len(risky)}")
        lines.append(f"  Çıktı: {r['combined']}")
        for p in parts:
            lines.append(f"    {p['emo']:<11} | {p['words']:>3} kw | {p['dur_sec']:>5}sn | "
                         f"kw/sn={p['kw_sn']} | {p['status']}")
        lines.append("")
    lines += ["=" * 70,
              "Karşılaştırma:",
              "  • A (hook_B, split2): enerjik hook, 8 parça, dengeli",
              "  • B (hook_C, split2): alternatif hook, 8 parça",
              "  • C (hook_B, split3): 10 parça, en güvenli (en az hallü riski)",
              "",
              "Hook için: A/B'deki hook_B vs hook_C hangisi daha canlı?",
              "Bölme için: split2 (A/B) yeterli mi, yoksa split3 (C) daha mı temiz?"]
    (OUT / "REPORT.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n📄 Rapor: {OUT / 'REPORT.txt'}")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    only = [x.strip() for x in args.only.split(",")] if args.only else None

    print("=" * 70)
    print("YOUTUBE TEST V3 — s2 Refine (Hook + Hallüsinasyon Düzeltme)")
    print("=" * 70)
    print(f"Clone: {TARGET_SPEAKER} | Alpha: {ALPHA} | Seed: {SEED}")
    print(f"MAX_KW = {MAX_KW} (hallüsinasyon sınırı)")

    print("\nModel yükleniyor (v10 SLERP patch)...")
    from mlx_audio.tts import load as mlx_load
    t0 = time.time()
    print("  [1/3] Base → centroid...")
    base = mlx_load(BASE_ID)
    ref_centroid = build_weighted_centroid(base, REF_AUDIO, top_k=TOP_K)
    del base; clean()

    print("  [2/3] CustomVoice...")
    custom = mlx_load(CUSTOM_ID)
    config = custom.config.talker_config
    sid = config.spk_id[TARGET_SPEAKER]
    target_vec = mx.array(custom.talker.get_input_embeddings().weight[sid]).squeeze()

    print("  [3/3] SLERP patch...")
    mixed_vec = slerp(target_vec, ref_centroid, ALPHA)
    patch_info = patch_token_standard(custom, TARGET_SPEAKER, mixed_vec)
    print(f"  Patch: {patch_info}  ({time.time()-t0:.1f}s)\n")

    all_results = []
    for name, blocks in STRATEGIES.items():
        if only and name not in only: continue
        try:
            r = run_strategy(custom, name, blocks, OUT)
            all_results.append(r)
        except Exception as e:
            import traceback
            print(f"  ✗ {name} HATA: {e}")
            traceback.print_exc()
            all_results.append({"strategy": name, "error": str(e)})

    write_report(all_results)
    del custom; clean()
    print("\n" + "=" * 70)
    print("TAMAMLANDI. Her stratejinin combined.wav'ını dinle:")
    print(f"  {OUT}")
    print("Karar: hook_B mı hook_C mı? split2 mi split3 mü?")

if __name__ == "__main__":
    main()
