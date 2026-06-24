# Copyright (c) 2026 Efeberk Çeker
# SPDX-License-Identifier: AGPL-3.0-only

"""
AI Voice Pipeline v2
Script → Speech Map → Qwen3-TTS ×N → Whisper Align → Skor → Birleştir → Doğal Ses

Kurulum:
    pip install qwen-tts whisper-timestamped librosa soundfile pydub numpy scipy transformers
    pip install flash-attn --no-build-isolation  # GPU varsa
"""

import os
import json
import logging
import argparse
from pathlib import Path

from speech_map import SpeechMapGenerator
from generator import TTSGenerator
from aligner import WordAligner
from scorer import WordScorer
from stitcher import AudioStitcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def run_pipeline(
    text: str,
    output_path: str = "final_output.wav",
    n_takes: int = 3,
    speaker: str = "Ethan",
    model_size: str = "0.6B",
    use_case: str = "youtube_narration",
    tone: str = "natural",
    llm_backend: str = "rule_based",   # "rule_based" | "ollama"
    ollama_model: str = "qwen3:8b",       # Ollama model seçimi
    use_phrase_mode: bool = True,
    work_dir: str = "./pipeline_work",
    backend: str = "pytorch",
    ref_audio: str | None = None,
    ref_text: str | None = None,
):
    """
    Ana pipeline.

    Args:
        text:         Seslendirme metni
        output_path:  Çıktı WAV
        n_takes:      Her segment için kaç take (3-5 önerilir)
        speaker:      Qwen3-TTS speaker
        model_size:   "0.6B" | "1.7B"
        use_case:     "youtube_intro" | "youtube_narration" | "advertisement" | "explainer" | "casual"
        tone:         "energetic" | "calm" | "warm" | "serious" | "natural"
        llm_backend:  "rule_based" (LLM gerekmez) | "ollama"
        ollama_model: Ollama backend için kullanılacak model adı
        use_phrase_mode: True=phrase, False=word seçimi
        work_dir:     Geçici dosya klasörü
    """
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("AI Voice Pipeline v2")
    log.info(f"  Use case: {use_case} | Tone: {tone} | LLM: {llm_backend} ({ollama_model})")
    log.info(f"  Takes: {n_takes} | Speaker: {speaker} | Model: {model_size}")
    log.info("=" * 60)

    # ── 1. SPEECH MAP ──────────────────────────────────────────────
    map_gen = SpeechMapGenerator(llm_backend=llm_backend, model_name=ollama_model)
    speech_map_path = work / "speech_map.json"
    speech_map = map_gen.generate(
        text=text,
        use_case=use_case,
        tone=tone,
        cache_path=str(speech_map_path),
    )

    segments_plan = speech_map.get("segments", [])
    log.info(f"Speech map: {len(segments_plan)} segment, emotion={speech_map['overall']['emotion']}")

    # ── 2. MODELLER ────────────────────────────────────────────────
    generator = TTSGenerator(
        backend=backend,
        model_size=model_size,
        speaker=speaker,
        ref_audio=ref_audio,
        ref_text=ref_text
    )
    aligner = WordAligner()
    scorer = WordScorer()

    # ── 3. HER SEGMENT → N TAKE → ALIGN → SKOR ────────────────────
    all_best_segments = []

    for seg_idx, seg_plan in enumerate(segments_plan):
        seg_text = seg_plan["text"]
        log.info(f"\n--- Segment {seg_idx+1}/{len(segments_plan)}: '{seg_text[:50]}' ---")
        log.info(f"    emotion={seg_plan.get('emotion')} stress={seg_plan.get('stress')} "
                 f"speed={seg_plan.get('speed')} pause={seg_plan.get('pause_after')}s")

        seg_dir = work / f"seg_{seg_idx:03d}"
        seg_dir.mkdir(exist_ok=True)

        # FIX: Tek seferde N take üret (çifte üretim bug'ı giderildi)
        # Mevcut take'leri kontrol et — zaten üretildiyse atla (resume desteği)
        existing_takes = [seg_dir / f"take_{i}.wav" for i in range(n_takes)]
        if not all(p.exists() for p in existing_takes):
            generator.generate_n_takes(
                text=seg_text,
                output_dir=str(seg_dir),
                n=n_takes,
                speech_map_segment=seg_plan,
            )

        takes = [str(seg_dir / f"take_{i}.wav") for i in range(n_takes)]

        # Alignment
        aligned_takes = []
        for t_idx, take_path in enumerate(takes):
            if not Path(take_path).exists():
                log.warning(f"  Take {t_idx} bulunamadı: {take_path}")
                continue
            align_path = seg_dir / f"take_{t_idx}_aligned.json"
            if not align_path.exists():
                alignment = aligner.align(take_path, seg_text)
                with open(align_path, "w") as f:
                    json.dump(alignment, f, indent=2)
            else:
                with open(align_path) as f:
                    alignment = json.load(f)
            aligned_takes.append({"path": take_path, "alignment": alignment})

        if not aligned_takes:
            log.error(f"  Segment {seg_idx} için aligned take yok, atlanıyor.")
            continue

        # En iyi seçim
        if use_phrase_mode:
            best = scorer.select_best_phrases(aligned_takes, seg_text)
        else:
            best = scorer.select_best_words(aligned_takes, seg_text)

        # Pause bilgisini ve akustik ipuçlarını ekle (speech map'ten)
        for b in best:
            b["pause_after"] = seg_plan.get("pause_after", 0.2)
            b["intonation_trend"] = seg_plan.get("intonation_trend", "stable")
            b["vowel_stretching"] = seg_plan.get("vowel_stretching", "normal")
            b["stress_anchor"] = seg_plan.get("stress_anchor", "")

        all_best_segments.append(best)
        log.info(f"  → Take {best[0]['take_idx']} seçildi (skor: {best[0]['score']:.3f})")

    # ── 4. BİRLEŞTİR ──────────────────────────────────────────────
    log.info("\nSesler birleştiriliyor...")
    stitcher = AudioStitcher()
    stitcher.stitch(all_best_segments, output_path)

    log.info(f"\n✅ Tamamlandı: {output_path}")
    log.info(f"   Speech map: {speech_map_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Voice Pipeline v2")
    parser.add_argument("text", help="Seslendirme metni")
    parser.add_argument("--output", default="final_output.wav")
    parser.add_argument("--takes", type=int, default=3)
    parser.add_argument("--speaker", default="Ethan")
    parser.add_argument("--model", default="0.6B", choices=["0.6B", "1.7B"])
    parser.add_argument("--use-case", default="youtube_narration",
                        choices=["youtube_intro", "youtube_narration", "advertisement", "explainer", "casual"])
    parser.add_argument("--tone", default="natural",
                        choices=["energetic", "calm", "warm", "serious", "natural"])
    parser.add_argument("--llm", default="rule_based",
                        choices=["rule_based", "ollama"],
                        help="Speech map için LLM backend")
    parser.add_argument("--ollama-model", default="qwen3:8b",
                        help="Ollama backend için model adı (örn: qwen3:8b, llama3)")
    parser.add_argument("--word-mode", action="store_true")
    parser.add_argument("--work-dir", default="./pipeline_work")
    parser.add_argument("--backend", default="pytorch", choices=["pytorch", "mlx"],
                        help="TTS Backend (pytorch veya mlx)")
    parser.add_argument("--ref-audio", default=None, help="Reference audio path for voice cloning")
    parser.add_argument("--ref-text", default=None, help="Reference text transcript for voice cloning")
    args = parser.parse_args()

    run_pipeline(
        text=args.text,
        output_path=args.output,
        n_takes=args.takes,
        speaker=args.speaker,
        model_size=args.model,
        use_case=args.use_case,
        tone=args.tone,
        llm_backend=args.llm,
        ollama_model=args.ollama_model,
        use_phrase_mode=not args.word_mode,
        work_dir=args.work_dir,
        backend=args.backend,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
    )
