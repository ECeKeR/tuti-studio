# Copyright (c) 2026 Efeberk Çeker
# SPDX-License-Identifier: AGPL-3.0-only

import os
import datetime
from pathlib import Path

class ProductionLogger:
    _log_file = Path("./pipeline_work/production_map.txt")

    @classmethod
    def reset(cls):
        """Resets the log file with a clean header."""
        cls._log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cls._log_file, "w", encoding="utf-8") as f:
            f.write(f"================================================================================\n")
            f.write(f"  VOICE TTS PRODUCTION MAP - ACTIVE FLOW REPORT\n")
            f.write(f"  Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"================================================================================\n\n")

    @classmethod
    def log_section(cls, title: str):
        """Writes a clean section header."""
        cls._log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cls._log_file, "a", encoding="utf-8") as f:
            f.write(f"\n================================================================================\n")
            f.write(f"  {title.upper()}\n")
            f.write(f"================================================================================\n")

    @classmethod
    def log_step(cls, step: str, details: str = ""):
        """Writes a step with a timestamp and optional detailed indent lines."""
        cls._log_file.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        with open(cls._log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] -> {step}\n")
            if details:
                indented = "\n".join("   | " + line for line in details.strip().split("\n"))
                f.write(f"{indented}\n")
