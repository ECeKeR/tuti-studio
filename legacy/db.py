# Copyright (c) 2026 Efeberk Çeker
# SPDX-License-Identifier: AGPL-3.0-only
#
# Legacy module.
# Kept for historical reference.
# Not used by the current application.

import sqlite3
import json
from pathlib import Path
import datetime

DB_PATH = Path("./pipeline_work/tuti.db")

def get_db_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
    except Exception:
        pass
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create tables if they do not exist. Go initializes them as well."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS timbres (
            name TEXT PRIMARY KEY,
            voice_mode TEXT NOT NULL,
            speaker TEXT,
            design_prompt TEXT,
            ref_audio TEXT,
            ref_text TEXT,
            model_size TEXT DEFAULT '1.7B',
            created_at TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            script_text TEXT,
            state_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.commit()
    conn.close()

def get_active_project_id():
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT value FROM settings WHERE key = 'active_project_id'")
        row = c.fetchone()
        conn.close()
        return int(row[0]) if row else None
    except Exception:
        return None

def load_project_state():
    project_id = get_active_project_id()
    conn = get_db_connection()
    c = conn.cursor()
    try:
        if not project_id:
            c.execute("SELECT state_json FROM projects ORDER BY updated_at DESC LIMIT 1")
            row = c.fetchone()
            return json.loads(row[0]) if row else None
        
        c.execute("SELECT state_json FROM projects WHERE id = ?", (project_id,))
        row = c.fetchone()
        return json.loads(row[0]) if row else None
    except Exception:
        return None
    finally:
        conn.close()

def save_project_state(state):
    project_id = state.get("id")
    if not project_id:
        return
    try:
        conn = get_db_connection()
        c = conn.cursor()
        # Set active project ID in settings
        c.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('active_project_id', ?)", (str(project_id),))
        now = datetime.datetime.now().isoformat()
        c.execute(
            "UPDATE projects SET state_json = ?, updated_at = ? WHERE id = ?",
            (json.dumps(state, ensure_ascii=False), now, project_id)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass
