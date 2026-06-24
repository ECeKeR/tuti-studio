# 🎙 Tuti Studio

<p align="center">
  <img src="frontend/logo_transparent.png" alt="Tuti Studio Logo" width="220">
</p>

<p align="center">
  <strong>🎙 Listen Before Reading</strong><br>
  This repository includes a short audio introduction generated with Tuti Studio.<br><br>
  <a href="https://github.com/user-attachments/files/29273011/repo_ses.MP3">
    🔊 Play Audio Introduction
  </a>
</p>


<p align="center">
  <a href="README.tr.md">🇹🇷 Türkçe Dokümantasyon</a>
</p>

**An open-source local AI voice studio powered by Qwen3-TTS.**

Generate natural voiceovers locally using Qwen3-TTS, Whisper alignment, and Apple Silicon MLX acceleration.

No cloud services.
No subscriptions.
No API keys.

Built for creators who want complete control over their voice generation pipeline.

---

### 📥 Download Latest Releases

| Platform | Architecture | Package Type | Link |
| :--- | :--- | :--- | :--- |
| **macOS** (Apple Silicon) | Apple Silicon (M1/M2/M3/M4) | DMG Installer | [Download Tuti.dmg (macOS)](bin/Tuti.dmg) |
| **Windows** (Intel/AMD) | x64 (AMD64) | Portable Executable | [Download Tuti-amd64-portable.exe (Windows)](bin/Tuti-amd64-portable.exe) |
| **Windows** (ARM64) | ARM64 | Portable Executable | [Download Tuti-arm64-portable.exe (Windows)](bin/Tuti-arm64-portable.exe) |

---

## Screenshots

<p align="center">
  <img src="assets/generation_screen.png" alt="Tuti Studio Voice Generation Screen" width="900">
  <br>
  <em>Voice Generation & Segment Editor</em>
</p>

<p align="center">
  <img src="assets/model_download_screen.png" alt="Tuti Studio Model Downloader" width="900">
  <br>
  <em>Local Model Management & Downloader</em>
</p>

<p align="center">
  <img src="assets/log_screen.png" alt="Tuti Studio Logs & Statistics" width="900">
  <br>
  <em>Real-time Synthesis Logs & Execution Flow</em>
</p>

---

## Why I Built This

This project started after my first YouTube video received almost no views.

While trying to understand what went wrong, I became obsessed with a simple question:

> Could the voice be part of the problem?

I tried many local text-to-speech solutions.

Some sounded robotic.

Some lacked consistency.

Some completely broke when generating longer narrations.

I wanted something that could produce natural voiceovers while staying fully local.

So I built Tuti Studio.

---

## What Makes Tuti Studio Different?

Many AI voice workflows look like this:

1. Generate 3-5 takes
2. Listen to all of them
3. Cut the best parts
4. Open Audacity
5. Fix timing manually
6. Export

Tuti Studio attempts to automate this process.

It can:

* Generate multiple takes
* Align speech using Whisper timestamps
* Score delivery quality
* Evaluate rhythm and timing
* Select the strongest performance automatically

The goal is simple:

> Create better narrations without relying on cloud services.

---

## Single-Pass First

While Tuti Studio supports advanced take generation and alignment workflows, one of the primary goals of the project is achieving high-quality narration in a single generation pass.

The best workflow is the one that disappears.

Write text.
Generate voice.
Publish.

---

## Wails & Ollama: Practical Architecture for Apple Silicon

AI voice generation already consumes significant memory and compute resources. I didn't want a heavy frontend or an extra LLM engine to hog the system.

### Why Wails Instead of Electron?
I didn't want the user interface to consume hundreds of additional megabytes just to display buttons and sliders. Tuti Studio uses **Wails (Go + HTML/JS)** to provide a lightweight desktop application, keeping system resources focused entirely on speech generation.

### Why Ollama for Speech Mapping?
Before generating audio, Tuti Studio creates a **Speech Map** (Prosody Plan) that splits the script into optimal segments (under 50 words to prevent TTS hallucinations), applies prosodic markup (CAPS for emphasis, `...` for pauses), and dynamically adjusts parameters (speed, stress, intonation, and `tts_instruct` directions) per segment.

Generating this map requires an LLM. Since local TTS models and Whisper already take up considerable disk space and RAM, downloading yet another heavy LLM runtime and its weights just for prosody planning felt wasteful. 

My MacBook was already running out of storage space, and Ollama was already installed. By bridging to your local Ollama instance (using models like `qwen3:8b`), we offloaded the text planning intelligently without adding any extra setup or memory overhead.

Sometimes, good engineering is about making practical decisions and utilizing what's already there.

## How to Build & Package

Tuti Studio uses **Wails v3** for its desktop application container. We provide production release scripts for both macOS and Windows.

### Prerequisites
* **Go** (1.21 or higher)
* **Node.js & npm**
* **Wails v3 CLI** (`go install github.com/wailsapp/wails/v3/cmd/wails3@latest`)
* **Python 3** (with MLX, Qwen3-TTS, and Whisper dependencies set up in your environment)

### Packaging for macOS
To generate a production `.app` bundle and a `.dmg` installer:
1. Ensure your signing identities are configured (optional, defaults to Ad-Hoc if not found).
2. Run the release script from the project root:
   ```bash
   ./release_macos.sh
   ```
This script automates frontend building, icon generation, Python resource copying, codesigning, notarization (if profile is configured), and DMG compilation.

### Packaging for Windows
To package the application for Windows (AMD64 & ARM64):
1. (Optional) Install NSIS via Homebrew (`brew install nsis`) to package installers on macOS.
2. Run the release script from the project root:
   ```bash
   ./release_windows.sh
   ```
This will compile both portable (`.exe`) and installer packages for Windows AMD64 and ARM64.

---

## Technology Stack

### Desktop

* Wails
* Go

### Frontend

* Vanilla JavaScript
* HTML
* CSS

### AI Pipeline

* Python
* Qwen3-TTS
* Whisper
* MLX
* SQLite

### Hardware Optimization

* Apple Silicon
* Metal Acceleration
* MLX Runtime

---

## Research

This repository also contains experimental research exploring speaker embedding interpolation and zero-shot adaptation techniques for closed-vocabulary TTS systems.

Current work includes:

* Speaker manifold interpolation
* SLERP-based embedding adaptation
* Emotion preservation testing
* Whisper alignment evaluation
* Voice design experiments

The research paper can be found in the [paper/](file:///Users/efeberkceker/Documents/develop/Tuti/tuti-studio/paper) directory, and the experimental source code/tests are in the [tests/](file:///Users/efeberkceker/Documents/develop/Tuti/tuti-studio/tests) directory.

---

## Open Source

Tuti Studio is licensed under AGPL-3.0.

If you improve it, build on top of it, or take it in a completely different direction, that's exactly what open source is for.

---

## Author

Created by Efeberk Çeker

Thanks for visiting the project.

I hope it helps someone build better local voice applications than I could find when I started.
