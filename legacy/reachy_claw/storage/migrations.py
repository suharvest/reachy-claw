"""Schema versioning for the SQLite database.

Each migration is a SQL script that brings the DB from version N to N+1.
Schema version is stored via SQLite's PRAGMA user_version.
"""

from __future__ import annotations

import sqlite3

CURRENT_VERSION = 1

MIGRATIONS: dict[int, str] = {
    1: """
    CREATE TABLE asr_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts INTEGER NOT NULL,
        role TEXT NOT NULL,
        text TEXT NOT NULL,
        emotion TEXT
    );
    CREATE INDEX idx_asr_events_ts ON asr_events(ts);

    CREATE TABLE emotions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts INTEGER NOT NULL,
        value REAL,
        label TEXT
    );
    CREATE INDEX idx_emotions_ts ON emotions(ts);

    CREATE TABLE faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts INTEGER NOT NULL,
        count INTEGER NOT NULL,
        smile_count INTEGER DEFAULT 0,
        capture_path TEXT
    );
    CREATE INDEX idx_faces_ts ON faces(ts);

    CREATE TABLE thoughts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts INTEGER NOT NULL,
        text TEXT NOT NULL,
        emotion TEXT
    );
    CREATE INDEX idx_thoughts_ts ON thoughts(ts);

    CREATE TABLE sensors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts INTEGER NOT NULL,
        source TEXT NOT NULL,
        key TEXT NOT NULL,
        value_num REAL,
        value_text TEXT
    );
    CREATE INDEX idx_sensors_ts ON sensors(ts);
    CREATE INDEX idx_sensors_key_ts ON sensors(key, ts);

    CREATE TABLE diaries (
        date TEXT PRIMARY KEY,
        markdown TEXT NOT NULL,
        generated_at INTEGER NOT NULL,
        llm_model TEXT NOT NULL,
        prompt_version TEXT NOT NULL,
        published_at INTEGER
    );
    """,
}


def get_version(conn: sqlite3.Connection) -> int:
    return conn.execute("PRAGMA user_version").fetchone()[0]


def set_version(conn: sqlite3.Connection, version: int) -> None:
    # PRAGMA cannot use parameterized values; version is int so safe to format.
    conn.execute(f"PRAGMA user_version = {int(version)}")


def migrate(conn: sqlite3.Connection) -> None:
    """Run all pending migrations to bring DB to CURRENT_VERSION."""
    current = get_version(conn)
    for v in range(current + 1, CURRENT_VERSION + 1):
        sql = MIGRATIONS[v]
        conn.executescript(sql)
        set_version(conn, v)
    conn.commit()