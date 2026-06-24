# Copyright (c) 2026 Efeberk Çeker
# SPDX-License-Identifier: AGPL-3.0-only

#!/usr/bin/env python3
import os, json, math, warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

warnings.filterwarnings("ignore")

REF_AUDIO = "pipeline_work/12345.mp3"

GENERATED_DIRS = [
    "pipeline_work/test_outputs/speaker_space_interpolation/wav",
    "pipeline_work/test_outputs/closed_loop_centroid_quality/wav",
    "pipeline_work/test_outputs/hack_only_quality/wav",
]

OUT = Path("pipeline_work/test_outputs/cross_encoder_validation")
OUT.mkdir(parents=True, exist_ok=True)

TARGET_FILES = [
    # En önemli adayları buraya koyabilirsin
    "alpha_1.00_serena_interp.wav",
    "alpha_0.80_serena_interp.wav",
    "alpha_0.60_serena_interp.wav",
    "ref2_ryan_neutral_clean_centroid_normhack.wav",
    "ref2_serena_youtube_energy_centroid_normhack.wav",
    "ref2_ryan_emotional_control_centroid_normhack.wav",
]


def find_file(name):
    for d in GENERATED_DIRS:
        p = Path(d) / name
        if p.exists():
            return p
    return None


def load_audio_16k(path):
    wav, sr = librosa.load(str(path), sr=16000, mono=True)
    wav = wav.astype(np.float32)

    # Speaker encoders için çok düşük ses sorun çıkarabilir
    peak = np.max(np.abs(wav)) if len(wav) else 0
    if peak > 0:
        wav = wav / max(peak, 1e-8) * 0.95

    return wav, 16000


def cosine(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))


def save_temp_wav(path, wav, sr=16000):
    sf.write(str(path), wav, sr)


# ─────────────────────────────────────────────
# Encoder 1: Resemblyzer
# ─────────────────────────────────────────────

def try_resemblyzer(ref_path, gen_paths):
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav

        enc = VoiceEncoder()
        ref_emb = enc.embed_utterance(preprocess_wav(Path(ref_path)))

        rows = []
        for p in gen_paths:
            gen_emb = enc.embed_utterance(preprocess_wav(Path(p)))
            rows.append({
                "file": str(p),
                "score": round(cosine(ref_emb, gen_emb), 5)
            })

        return {
            "encoder": "resemblyzer",
            "status": "SUCCESS",
            "scores": rows
        }
    except Exception as e:
        return {
            "encoder": "resemblyzer",
            "status": "FAILED",
            "error": str(e)
        }


# ─────────────────────────────────────────────
# Encoder 2: SpeechBrain ECAPA
# ─────────────────────────────────────────────

def try_speechbrain_ecapa(ref_path, gen_paths):
    try:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(OUT / "speechbrain_ecapa")
        )

        def emb(path):
            wav, sr = load_audio_16k(path)
            tensor = torch.tensor(wav).unsqueeze(0)
            with torch.no_grad():
                e = classifier.encode_batch(tensor).squeeze().cpu().numpy()
            return e

        ref_emb = emb(ref_path)

        rows = []
        for p in gen_paths:
            gen_emb = emb(p)
            rows.append({
                "file": str(p),
                "score": round(cosine(ref_emb, gen_emb), 5)
            })

        return {
            "encoder": "speechbrain_ecapa_voxceleb",
            "status": "SUCCESS",
            "scores": rows
        }
    except Exception as e:
        return {
            "encoder": "speechbrain_ecapa_voxceleb",
            "status": "FAILED",
            "error": str(e)
        }


# ─────────────────────────────────────────────
# Encoder 3: SpeechBrain x-vector
# ─────────────────────────────────────────────

def try_speechbrain_xvector(ref_path, gen_paths):
    try:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir=str(OUT / "speechbrain_xvector")
        )

        def emb(path):
            wav, sr = load_audio_16k(path)
            tensor = torch.tensor(wav).unsqueeze(0)
            with torch.no_grad():
                e = classifier.encode_batch(tensor).squeeze().cpu().numpy()
            return e

        ref_emb = emb(ref_path)

        rows = []
        for p in gen_paths:
            gen_emb = emb(p)
            rows.append({
                "file": str(p),
                "score": round(cosine(ref_emb, gen_emb), 5)
            })

        return {
            "encoder": "speechbrain_xvector_voxceleb",
            "status": "SUCCESS",
            "scores": rows
        }
    except Exception as e:
        return {
            "encoder": "speechbrain_xvector_voxceleb",
            "status": "FAILED",
            "error": str(e)
        }


# ─────────────────────────────────────────────
# Encoder 4: PyAnnote verification
# HF_TOKEN gerektirir
# ─────────────────────────────────────────────

def try_pyannote(ref_path, gen_paths):
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        return {
            "encoder": "pyannote",
            "status": "SKIPPED",
            "error": "HF_TOKEN bulunamadı. PyAnnote için Hugging Face token gerekli."
        }

    try:
        from pyannote.audio import Model, Inference

        model = Model.from_pretrained(
            "pyannote/embedding",
            use_auth_token=token
        )
        inference = Inference(model, window="whole")

        ref_emb = inference(str(ref_path))

        rows = []
        for p in gen_paths:
            gen_emb = inference(str(p))
            rows.append({
                "file": str(p),
                "score": round(cosine(ref_emb, gen_emb), 5)
            })

        return {
            "encoder": "pyannote_embedding",
            "status": "SUCCESS",
            "scores": rows
        }

    except Exception as e:
        return {
            "encoder": "pyannote_embedding",
            "status": "FAILED",
            "error": str(e)
        }


def verdict(score):
    if score >= 0.85:
        return "STRONG"
    if score >= 0.75:
        return "GOOD"
    if score >= 0.65:
        return "WEAK"
    return "LOW"


def main():
    ref = Path(REF_AUDIO)
    if not ref.exists():
        raise FileNotFoundError(f"Reference not found: {REF_AUDIO}")

    gen_paths = []
    for name in TARGET_FILES:
        p = find_file(name)
        if p:
            gen_paths.append(p)
        else:
            print(f"Missing generated file, skipped: {name}")

    if not gen_paths:
        raise RuntimeError("Hiç generated wav bulunamadı. TARGET_FILES listesini kontrol et.")

    print("\nReference:", ref)
    print("Generated files:")
    for p in gen_paths:
        print(" -", p)

    results = {
        "reference": str(ref),
        "generated_files": [str(p) for p in gen_paths],
        "encoders": []
    }

    tests = [
        try_resemblyzer,
        try_speechbrain_ecapa,
        try_speechbrain_xvector,
        try_pyannote,
    ]

    for fn in tests:
        print(f"\nRunning {fn.__name__}...")
        r = fn(ref, gen_paths)
        results["encoders"].append(r)
        print(r["encoder"], r["status"])
        if r["status"] == "SUCCESS":
            for s in r["scores"]:
                print(" ", Path(s["file"]).name, s["score"], verdict(s["score"]))
        else:
            print(" ", r.get("error"))

    # Aggregate
    aggregate = {}
    for p in gen_paths:
        name = Path(p).name
        vals = []

        for enc in results["encoders"]:
            if enc["status"] != "SUCCESS":
                continue
            for row in enc["scores"]:
                if Path(row["file"]).name == name:
                    vals.append(row["score"])

        if vals:
            aggregate[name] = {
                "mean_score": round(float(np.mean(vals)), 5),
                "min_score": round(float(np.min(vals)), 5),
                "max_score": round(float(np.max(vals)), 5),
                "num_encoders": len(vals),
                "verdict": verdict(float(np.mean(vals)))
            }

    results["aggregate"] = dict(
        sorted(
            aggregate.items(),
            key=lambda x: x[1]["mean_score"],
            reverse=True
        )
    )

    json_path = OUT / "cross_encoder_validation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    md_path = OUT / "cross_encoder_validation_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Cross-Encoder Speaker Validation\n\n")
        f.write(f"Reference: `{ref}`\n\n")

        f.write("## Aggregate Ranking\n\n")
        f.write("| File | Mean | Min | Max | Encoders | Verdict |\n")
        f.write("|---|---:|---:|---:|---:|---|\n")
        for name, row in results["aggregate"].items():
            f.write(
                f"| `{name}` | {row['mean_score']} | {row['min_score']} | "
                f"{row['max_score']} | {row['num_encoders']} | {row['verdict']} |\n"
            )

        f.write("\n## Per Encoder Scores\n\n")
        for enc in results["encoders"]:
            f.write(f"### {enc['encoder']} — {enc['status']}\n\n")
            if enc["status"] == "SUCCESS":
                for s in enc["scores"]:
                    f.write(f"- `{Path(s['file']).name}`: {s['score']} ({verdict(s['score'])})\n")
            else:
                f.write(f"- {enc.get('error')}\n")
            f.write("\n")

    print("\nDONE")
    print("JSON:", json_path)
    print("MD:", md_path)

    print("\nAGGREGATE:")
    for name, row in results["aggregate"].items():
        print(row["mean_score"], row["verdict"], name)


if __name__ == "__main__":
    main()