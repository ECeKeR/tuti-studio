# Copyright (c) 2026 Efeberk Çeker
# SPDX-License-Identifier: AGPL-3.0-only

"""
Timbre Store — Ses Kimliği Kayıt Defteri
VoiceDesign ile oluşturulan, klonlanan veya preset ses kimliklerini SQLite veritabanında saklar.
"""

import sqlite3
import logging
from pathlib import Path
from datetime import datetime

log = logging.getLogger(__name__)

DB_PATH = Path("./pipeline_work/tuti.db")


class TimbreStore:
    def __init__(self, registry_path=None):
        # SQLite database tables are initialized by Go backend on launch
        pass

    def _get_connection(self):
        conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
        except Exception as e:
            log.warning(f"Failed to set PRAGMAs on DB connection: {e}")
        conn.row_factory = sqlite3.Row
        return conn

    def save_timbre(self, name: str, config: dict) -> dict:
        """Yeni bir ses kimliği kaydet veya mevcut olanı güncelle."""
        name = name.strip().lower().replace(" ", "_")
        if not name:
            raise ValueError("Timbre name cannot be empty")

        entry = {
            "name": name,
            "voice_mode": config.get("voice_mode", "preset"),
            "speaker": config.get("speaker", ""),
            "design_prompt": config.get("design_prompt", ""),
            "ref_audio": config.get("ref_audio", ""),
            "ref_text": config.get("ref_text", ""),
            "model_size": config.get("model_size", "1.7B"),
            "created_at": datetime.now().isoformat(),
        }

        conn = self._get_connection()
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO timbres (name, voice_mode, speaker, design_prompt, ref_audio, ref_text, model_size, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry["name"],
            entry["voice_mode"],
            entry["speaker"],
            entry["design_prompt"],
            entry["ref_audio"],
            entry["ref_text"],
            entry["model_size"],
            entry["created_at"]
        ))
        conn.commit()
        conn.close()
        log.info(f"Timbre kaydedildi: '{name}' (mode={entry['voice_mode']})")
        return entry

    def load_timbre(self, name: str) -> dict | None:
        """İsme göre ses kimliği yükle."""
        name = name.strip().lower()
        conn = self._get_connection()
        c = conn.cursor()
        c.execute("SELECT * FROM timbres WHERE name = ?", (name,))
        row = c.fetchone()
        conn.close()
        if row:
            return dict(row)
        return None

    def list_timbres(self) -> list[dict]:
        """Tüm kayıtlı ses kimliklerini listele."""
        conn = self._get_connection()
        c = conn.cursor()
        c.execute("SELECT * FROM timbres ORDER BY name ASC")
        rows = c.fetchall()
        conn.close()
        
        timbres = []
        for r in rows:
            t = dict(r)
            adapter_path = Path("./pipeline_work") / f"{t['name']}_adapter.safetensors"
            t["sft_enabled"] = adapter_path.exists()
            timbres.append(t)
        return timbres

    def delete_timbre(self, name: str) -> bool:
        """Ses kimliğini sil."""
        name = name.strip().lower()
        conn = self._get_connection()
        c = conn.cursor()
        c.execute("DELETE FROM timbres WHERE name = ?", (name,))
        deleted = c.rowcount > 0
        conn.commit()
        conn.close()
        if deleted:
            log.info(f"Timbre silindi: '{name}'")
        return deleted
