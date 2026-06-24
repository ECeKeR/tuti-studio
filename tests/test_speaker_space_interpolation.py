# Copyright (c) 2026 Efeberk Çeker
# SPDX-License-Identifier: AGPL-3.0-only

#!/usr/bin/env python3
"""
Speaker Space Interpolation Test
=================================
Kanıt sorusu: Embedding override gerçekten kontinüel bir uzayda mı çalışıyor?

Yöntem:
  mixed = normalize((1-alpha) * ryan_token + alpha * ref_centroid)

Alpha 0.0 → 1.0 arası 7 adım:
  0.00 = saf ryan
  0.15
  0.30
  0.45
  0.60
  0.80
  1.00 = saf ref (mevcut hack)

Her alpha için:
  - Ses üretilir
  - Speaker encoder ile re-encode edilir
  - ref centroid'e cosine hesaplanır
  - spectral_centroid ve high_freq_ratio ölçülür

Eğer uzay gerçekse:
  → cosine alpha ile monoton artar
  → spectral değerler smooth geçiş yapar
  → ses kulakta ara noktada durur

Eğer uzay yoksa (diskret switch):
  → cosine aniden atlar (alpha=0.3'te bile 1.0'a yakın ya da 0.0'a yakın)
  → spectral değerler smooth geçmez
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
log = logging.getLogger("speaker_space_interpolation")

# ── Config ────────────────────────────────────────────────────────────────────

CUSTOM_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"
BASE_ID   = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"

OUT = Path("pipeline_work/test_outputs/speaker_space_interpolation")
WAV = OUT / "wav"
OUT.mkdir(parents=True, exist_ok=True)
WAV.mkdir(parents=True, exist_ok=True)

# Referans ses — en yüksek production score'u olan ref2 kullanılıyor
REF_AUDIO = "pipeline_work/12345.mp3"

# Test metni: neutral, temiz, kısa — artefakt riskini minimize et
TEST_TEXT    = "OHH GİRLLL NO WHİTE MAN ONLY BLACK MAN "
TEST_INSTRUCT = "Speak as a sarcastic, assertive teenage girl: crisp enunciation, controlled volume, with vocal emphasis that conveys disdain and authority."
TEST_TEMP    = 0.35
SEED         = 42
LANG         = "English"

# İnterpolasyon adımları
ALPHAS = [0.00, 0.15, 0.30, 0.45, 0.60, 0.80, 1.00]

# Hangi speaker token üzerinde test edilecek
TARGET_SPEAKER = "serena"   # "serena" de denenebilir

TOP_K = 3   # centroid hesabı için

# ── Helpers ──────────────────────────────────────────────────────────────────

def clean():
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass


def norm_val(x):
    return float(mx.linalg.norm(x))


def cosine(a, b):
    a = a.astype(mx.float32).reshape(-1)
    b = b.astype(mx.float32).reshape(-1)
    return float(mx.sum(a * b) / (mx.linalg.norm(a) * mx.linalg.norm(b) + 1e-8))


def normalize_vec(v):
    return v / (mx.linalg.norm(v) + 1e-8)


def save_wav(path, chunks, sr):
    wav = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(1, dtype=np.float32)
    peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    if peak > 1.0:
        wav = wav / peak
    sf.write(str(path), wav, sr)


def analyze_audio(path):
    wav, sr = sf.read(str(path))
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    abs_wav = np.abs(wav)
    duration  = len(wav) / sr
    peak      = float(np.max(abs_wav)) if len(wav) else 0
    rms       = float(np.sqrt(np.mean(wav ** 2))) if len(wav) else 0
    silence   = float(np.mean(abs_wav < 0.005)) if len(wav) else 0

    spec   = np.abs(np.fft.rfft(wav)) if len(wav) else np.array([0.0])
    freqs  = np.fft.rfftfreq(len(wav), 1 / sr) if len(wav) else np.array([0.0])
    total  = float(np.sum(spec ** 2) + 1e-12)
    high   = float(np.sum(spec[freqs >= 4000] ** 2) / total)
    centroid = float(np.sum(freqs * spec) / (np.sum(spec) + 1e-12))

    return {
        "duration":          round(duration, 3),
        "peak":              round(peak, 5),
        "rms":               round(rms, 6),
        "silence_ratio":     round(silence, 4),
        "spectral_centroid": round(centroid, 2),
        "high_freq_ratio":   round(high, 5),
        "size":              Path(path).stat().st_size,
    }


# ── Speaker Embedding Extraction ─────────────────────────────────────────────

def build_ref_centroid(base_model, audio_path, top_k=3):
    from mlx_audio.utils import load_audio
    from finetune import segment_audio

    sr  = base_model.sample_rate
    audio = np.array(load_audio(str(audio_path), sample_rate=sr))
    segments = segment_audio(audio, sr, min_sec=3.0, max_sec=8.0)

    log.info(f"Reference audio: {len(segments)} segments found")

    embs = []
    for i, seg in enumerate(segments):
        emb = base_model.extract_speaker_embedding(mx.array(seg), sr=sr).squeeze()
        embs.append(emb)
        log.info(f"  Segment {i+1}: duration={len(seg)/sr:.2f}s  norm={norm_val(emb):.4f}")

    if not embs:
        raise RuntimeError("No segments extracted from reference audio")

    all_avg = mx.stack(embs).mean(axis=0).squeeze()

    # Centroid: her segmentin diğerlerine ortalama cosine + all_avg'e cosine
    scores = []
    for i, e1 in enumerate(embs):
        others = [cosine(e1, e2) for j, e2 in enumerate(embs) if j != i]
        avg_cos = float(np.mean(others)) if others else 1.0
        cos_avg = cosine(e1, all_avg)
        combined = avg_cos * 0.6 + cos_avg * 0.4
        scores.append({"idx": i, "combined": combined})

    scores.sort(key=lambda x: x["combined"], reverse=True)
    selected = [embs[s["idx"]] for s in scores[:top_k]]
    centroid = mx.stack(selected).mean(axis=0).squeeze()

    log.info(f"Centroid built from top-{top_k} segments: norm={norm_val(centroid):.4f}")
    log.info(f"cos(all_avg, centroid) = {cosine(all_avg, centroid):.5f}")

    return centroid, all_avg


def reencode(base_model, wav_path):
    from mlx_audio.utils import load_audio
    from finetune import segment_audio

    sr    = base_model.sample_rate
    audio = np.array(load_audio(str(wav_path), sample_rate=sr))
    segs  = segment_audio(audio, sr, min_sec=3.0, max_sec=8.0)

    if not segs:
        log.warning(f"Generated audio is too short for 3s segmentation. Falling back to using the entire audio as a single segment: {wav_path}")
        segs = [audio]

    embs = [base_model.extract_speaker_embedding(mx.array(s), sr=sr).squeeze() for s in segs]
    avg  = mx.stack(embs).mean(axis=0).squeeze()
    return avg, len(embs)


# ── Token & Patch ─────────────────────────────────────────────────────────────

def get_token_vec(model, speaker):
    config    = model.config.talker_config
    emb_layer = model.talker.get_input_embeddings()
    sid       = config.spk_id[speaker]
    vec       = mx.array(emb_layer.weight[sid]).squeeze()
    return int(sid), vec


def patch_token(model, speaker, new_vec):
    """
    Norm-match: yeni vektörün yönünü koru, normunu orijinal token normuyla eşleştir.
    Bu, önceki centroid testindeki yöntemle aynı.
    """
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
        "cos_orig_vs_new": round(cosine(original, matched), 5),
    }


# ── Generation ────────────────────────────────────────────────────────────────

def generate(model, speaker, text, instruct, temp, out_path):
    np.random.seed(SEED)
    mx.random.seed(SEED)

    chunks = []
    sr     = 24000
    t0     = time.time()

    for result in model.generate_custom_voice(
        text=text,
        speaker=speaker,
        language=LANG,
        instruct=instruct,
        temperature=temp,
    ):
        chunks.append(np.array(result.audio))
        sr = result.sample_rate

    save_wav(out_path, chunks, sr)
    elapsed = time.time() - t0

    metrics = analyze_audio(out_path)
    metrics["latency"] = round(elapsed, 3)
    metrics["rtf"]     = round(elapsed / metrics["duration"], 3) if metrics["duration"] > 0 else 0

    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    report = {
        "timestamp":       time.strftime("%Y-%m-%d %H:%M:%S"),
        "goal":            "Prove we operate in a continuous speaker embedding space via interpolation.",
        "hypothesis":      "If space is real: cosine(ref_centroid, generated) increases monotonically with alpha. Spectral values shift smoothly.",
        "custom_model":    CUSTOM_ID,
        "base_model":      BASE_ID,
        "ref_audio":       REF_AUDIO,
        "target_speaker":  TARGET_SPEAKER,
        "alphas":          ALPHAS,
        "seed":            SEED,
        "test_text":       TEST_TEXT,
        "test_instruct":   TEST_INSTRUCT,
        "test_temperature": TEST_TEMP,
        "ryan_baseline":   None,
        "ref_centroid_info": {},
        "interpolation_results": [],
        "verdict":         None,
    }

    # ── Phase 1: Build reference centroid ─────────────────────────────────────
    log.info("=" * 60)
    log.info("PHASE 1 | Build reference centroid from ref audio")
    log.info("=" * 60)

    from mlx_audio.tts import load as mlx_load

    base = mlx_load(BASE_ID)
    ref_centroid, ref_all_avg = build_ref_centroid(base, REF_AUDIO, top_k=TOP_K)

    report["ref_centroid_info"] = {
        "centroid_norm":       round(norm_val(ref_centroid), 4),
        "all_avg_norm":        round(norm_val(ref_all_avg), 4),
        "cos_centroid_vs_avg": round(cosine(ref_centroid, ref_all_avg), 5),
    }

    del base
    clean()

    # ── Phase 2: Get ryan token baseline ──────────────────────────────────────
    log.info("=" * 60)
    log.info("PHASE 2 | Get Ryan token vector & generate baseline")
    log.info("=" * 60)

    custom     = mlx_load(CUSTOM_ID)
    ryan_id, ryan_vec = get_token_vec(custom, TARGET_SPEAKER)

    log.info(f"Ryan token id: {ryan_id}  norm: {norm_val(ryan_vec):.4f}")
    log.info(f"cos(ryan, ref_centroid) = {cosine(ryan_vec, ref_centroid):.5f}")

    baseline_path = WAV / f"alpha_0.00_{TARGET_SPEAKER}_baseline.wav"
    baseline_metrics = generate(custom, TARGET_SPEAKER, TEST_TEXT, TEST_INSTRUCT, TEST_TEMP, baseline_path)

    report["ryan_baseline"] = {
        "token_id":            ryan_id,
        "token_norm":          round(norm_val(ryan_vec), 4),
        "cos_ryan_vs_ref":     round(cosine(ryan_vec, ref_centroid), 5),
        "file":                str(baseline_path),
        **baseline_metrics,
    }

    log.info(f"Baseline generated: {baseline_path}")
    log.info(f"  spectral_centroid={baseline_metrics['spectral_centroid']}  high_freq={baseline_metrics['high_freq_ratio']}")

    del custom
    clean()

    # ── Phase 3: Re-encode baseline ───────────────────────────────────────────
    log.info("Re-encoding baseline for closed-loop cosine...")
    base = mlx_load(BASE_ID)
    baseline_emb, _ = reencode(base, baseline_path)
    baseline_cl_cos  = cosine(ref_centroid, baseline_emb)
    report["ryan_baseline"]["closed_loop_cosine"] = round(baseline_cl_cos, 5)
    log.info(f"Baseline closed_loop_cosine vs ref_centroid: {baseline_cl_cos:.5f}")
    del base
    clean()

    # ── Phase 4: Interpolation sweep ─────────────────────────────────────────
    log.info("=" * 60)
    log.info("PHASE 4 | Interpolation sweep")
    log.info("=" * 60)

    results = []

    for alpha in ALPHAS:
        log.info(f"\n{'─'*50}")
        log.info(f"Alpha = {alpha:.2f}  |  ({1-alpha:.2f})*ryan + {alpha:.2f}*ref_centroid")

        # Interpolated vector (raw, before norm-match)
        mixed_raw = (1.0 - alpha) * ryan_vec + alpha * ref_centroid

        # Cos of mixed vector to both endpoints
        cos_mixed_ryan = cosine(mixed_raw, ryan_vec)
        cos_mixed_ref  = cosine(mixed_raw, ref_centroid)
        log.info(f"  mixed_vec → cos(ryan)={cos_mixed_ryan:.5f}  cos(ref)={cos_mixed_ref:.5f}  norm={norm_val(mixed_raw):.4f}")

        # Load model fresh each time (avoid state bleed between alphas)
        custom = mlx_load(CUSTOM_ID)

        patch_info = patch_token(custom, TARGET_SPEAKER, mixed_raw)
        log.info(f"  Patch: cos(orig_token vs new)={patch_info['cos_orig_vs_new']:.5f}  final_norm={patch_info['final_norm']:.4f}")

        out_path = WAV / f"alpha_{alpha:.2f}_{TARGET_SPEAKER}_interp.wav"
        metrics  = generate(custom, TARGET_SPEAKER, TEST_TEXT, TEST_INSTRUCT, TEST_TEMP, out_path)

        log.info(f"  Generated: spectral_centroid={metrics['spectral_centroid']}  high_freq={metrics['high_freq_ratio']}  silence={metrics['silence_ratio']}")

        del custom
        clean()

        # Re-encode
        base = mlx_load(BASE_ID)
        gen_emb, seg_count = reencode(base, out_path)
        cl_cos_centroid = cosine(ref_centroid, gen_emb)
        cl_cos_all_avg  = cosine(ref_all_avg,  gen_emb)
        cl_cos_ryan     = cosine(ryan_vec,      gen_emb)
        del base
        clean()

        row = {
            "alpha":                   alpha,
            "label":                   f"({1-alpha:.2f})*ryan + {alpha:.2f}*ref",
            "mixed_vec_cos_ryan":      round(cos_mixed_ryan, 5),
            "mixed_vec_cos_ref":       round(cos_mixed_ref, 5),
            "mixed_vec_norm_raw":      round(norm_val(mixed_raw), 4),
            "patch":                   patch_info,
            "file":                    str(out_path),
            **metrics,
            "closed_loop": {
                "segment_count":            seg_count,
                "cos_ref_centroid":         round(cl_cos_centroid, 5),
                "cos_ref_all_avg":          round(cl_cos_all_avg, 5),
                "cos_ryan_token":           round(cl_cos_ryan, 5),
            },
        }

        results.append(row)

        log.info(
            f"  ✓ closed_loop → cos(ref_centroid)={cl_cos_centroid:.5f}  "
            f"cos(ryan)={cl_cos_ryan:.5f}  "
            f"spectral={metrics['spectral_centroid']}"
        )

    report["interpolation_results"] = results

    # ── Phase 5: Verdict ──────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("PHASE 5 | Computing verdict")
    log.info("=" * 60)

    cl_cosines   = [r["closed_loop"]["cos_ref_centroid"] for r in results]
    spectrals    = [r["spectral_centroid"] for r in results]
    ryan_cosines = [r["closed_loop"]["cos_ryan_token"] for r in results]

    # Monotonicity check on closed_loop cosine
    diffs        = [cl_cosines[i+1] - cl_cosines[i] for i in range(len(cl_cosines)-1)]
    is_monotone  = all(d >= -0.003 for d in diffs)   # small tolerance for noise
    total_range  = max(cl_cosines) - min(cl_cosines)

    # Spectral shift check
    spectral_range   = max(spectrals) - min(spectrals)
    spectral_directional = spectrals[-1] != spectrals[0]

    # Ryan cosine should DECREASE as alpha increases (moving away from ryan)
    ryan_diffs       = [ryan_cosines[i+1] - ryan_cosines[i] for i in range(len(ryan_cosines)-1)]
    ryan_decreasing  = all(d <= 0.003 for d in ryan_diffs)

    verdict_lines = []

    if total_range > 0.005 and is_monotone:
        verdict_lines.append("✅ UZAY KANITI (Güçlü): closed_loop cosine alpha ile monoton artıyor.")
    elif total_range > 0.002:
        verdict_lines.append("⚠️  UZAY KANITI (Zayıf): cosine değişiyor ama monoton değil. Gürültü etkisi olabilir.")
    else:
        verdict_lines.append("❌ UZAY KANITI YOK: cosine değişmiyor. Model token'ı diskret switch gibi kullanıyor.")

    if spectral_range > 100 and spectral_directional:
        verdict_lines.append(f"✅ SPEKTRİK GEÇİŞ: {min(spectrals):.0f} → {max(spectrals):.0f} Hz smooth kayma ({spectral_range:.0f} Hz range).")
    elif spectral_range > 30:
        verdict_lines.append(f"⚠️  SPEKTRİK GEÇİŞ (Zayıf): {spectral_range:.0f} Hz range, muhtemelen anlamlı.")
    else:
        verdict_lines.append(f"❌ SPEKTRİK GEÇİŞ YOK: {spectral_range:.0f} Hz range, gürültü seviyesinde.")

    if ryan_decreasing:
        verdict_lines.append("✅ RYAN MESAFESİ: Alpha arttıkça üretilen ses ryan token'ından uzaklaşıyor.")
    else:
        verdict_lines.append("⚠️  RYAN MESAFESİ: Beklenen uzaklaşma monoton değil.")

    report["verdict"] = {
        "cl_cosine_range":   round(total_range, 5),
        "cl_cosine_values":  [round(c, 5) for c in cl_cosines],
        "spectral_range":    round(spectral_range, 2),
        "spectral_values":   [round(s, 2) for s in spectrals],
        "ryan_cos_values":   [round(c, 5) for c in ryan_cosines],
        "is_monotone":       is_monotone,
        "ryan_decreasing":   ryan_decreasing,
        "conclusions":       verdict_lines,
    }

    # ── Output ────────────────────────────────────────────────────────────────
    json_path = OUT / "speaker_space_interpolation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md_path = OUT / "speaker_space_interpolation_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Speaker Space Interpolation Test\n\n")
        f.write(f"**Soru:** Embedding override gerçekten kontinüel bir uzayda mı çalışıyor?\n\n")
        f.write(f"**Referans:** `{REF_AUDIO}`\n")
        f.write(f"**Hedef Token:** `{TARGET_SPEAKER}`\n\n")

        f.write("## Baseline\n\n")
        b = report["ryan_baseline"]
        f.write(f"- Ryan token closed_loop_cosine vs ref: `{b['closed_loop_cosine']}`\n")
        f.write(f"- Ryan spectral_centroid: `{b['spectral_centroid']}`\n\n")

        f.write("## Interpolation Results\n\n")
        f.write("| Alpha | Label | CL cos(ref) | CL cos(ryan) | spectral | high_freq | silence |\n")
        f.write("|-------|-------|------------|--------------|----------|-----------|--------|\n")
        for r in results:
            cl = r["closed_loop"]
            f.write(
                f"| {r['alpha']:.2f} | {r['label']} | "
                f"{cl['cos_ref_centroid']} | {cl['cos_ryan_token']} | "
                f"{r['spectral_centroid']} | {r['high_freq_ratio']} | {r['silence_ratio']} |\n"
            )

        f.write("\n## Verdict\n\n")
        v = report["verdict"]
        for line in v["conclusions"]:
            f.write(f"- {line}\n")

        f.write(f"\n**closed_loop cosine sweep:** {v['cl_cosine_values']}\n")
        f.write(f"**spectral sweep:** {v['spectral_values']}\n")
        f.write(f"**range:** cosine={v['cl_cosine_range']}  spectral={v['spectral_range']} Hz\n")

        f.write("\n## Listen Order\n\n")
        f.write("Dinleme sırası — alpha 0.0'dan 1.0'a:\n\n")
        for r in results:
            f.write(f"- alpha={r['alpha']:.2f} → `{r['file']}`\n")

    listen_path = OUT / "listen_order.txt"
    with open(listen_path, "w", encoding="utf-8") as f:
        f.write("DİNLEME SIRASI (alpha 0.0 = saf ryan, 1.0 = saf ref embedding)\n\n")
        f.write(f"BASELINE (hack yok): {report['ryan_baseline']['file']}\n\n")
        for r in results:
            cl = r["closed_loop"]
            f.write(
                f"alpha={r['alpha']:.2f}  cos_ref={cl['cos_ref_centroid']}  "
                f"cos_ryan={cl['cos_ryan_token']}  "
                f"spectral={r['spectral_centroid']}  "
                f"→  {r['file']}\n"
            )
        f.write("\nVERDICT:\n")
        for line in report["verdict"]["conclusions"]:
            f.write(f"  {line}\n")

    # Terminal summary
    print("\n" + "=" * 60)
    print("SPEAKER SPACE INTERPOLATION — SONUÇLAR")
    print("=" * 60)
    print(f"\nBaseline (ryan, hack yok):")
    print(f"  closed_loop cos(ref) = {report['ryan_baseline']['closed_loop_cosine']}")
    print(f"  spectral_centroid    = {report['ryan_baseline']['spectral_centroid']}")
    print(f"\nAlpha sweep:")
    print(f"  {'Alpha':>6}  {'cos(ref)':>10}  {'cos(ryan)':>10}  {'spectral':>10}")
    print(f"  {'------':>6}  {'--------':>10}  {'---------':>10}  {'--------':>10}")
    for r in results:
        cl = r["closed_loop"]
        print(
            f"  {r['alpha']:>6.2f}  "
            f"{cl['cos_ref_centroid']:>10.5f}  "
            f"{cl['cos_ryan_token']:>10.5f}  "
            f"{r['spectral_centroid']:>10.2f}"
        )
    print("\nVERDICT:")
    for line in report["verdict"]["conclusions"]:
        print(f"  {line}")
    print(f"\nJSON:  {json_path}")
    print(f"MD:    {md_path}")
    print(f"Listen: {listen_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()