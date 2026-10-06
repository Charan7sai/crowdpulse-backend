# ──────────────────────────────────────────────
# SQLite Logging Layer — Crowd Data Persistence
# ──────────────────────────────────────────────
# Stores every crowd reading with the new fields
# introduced in Checkpoint 1 (area-based capacity,
# Fruin level, breach predictions).

import sqlite3
import threading
from datetime import datetime, timezone

import config

_lock = threading.Lock()


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create (or migrate) the crowd_logs table."""
    with _lock:
        conn = _get_connection()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS crowd_logs (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp               TEXT    NOT NULL,
                current_count           INTEGER NOT NULL,
                smoothed_count          REAL    NOT NULL DEFAULT 0,
                zone_name               TEXT    NOT NULL DEFAULT '',
                zone_max_capacity       INTEGER NOT NULL DEFAULT 0,
                zone_safe_capacity      INTEGER NOT NULL DEFAULT 0,
                zone_area_m2            REAL    NOT NULL DEFAULT 0,
                occupancy_ratio         REAL    NOT NULL DEFAULT 0,
                occupancy_pct           TEXT    NOT NULL DEFAULT '',
                safe_slots_left         INTEGER NOT NULL DEFAULT 0,
                density_per_m2          REAL    NOT NULL DEFAULT 0,
                fruin_level             TEXT    NOT NULL DEFAULT '',
                growth_rate             REAL    NOT NULL DEFAULT 0,
                fill_rate_per_min       REAL    NOT NULL DEFAULT 0,
                time_to_safe_breach_sec INTEGER,
                time_to_max_breach_sec  INTEGER,
                risk_level              TEXT    NOT NULL,
                risk_score              REAL    NOT NULL,
                surge_flag              INTEGER NOT NULL,
                duration_high           REAL    NOT NULL
            )
            """
        )
        conn.commit()
        conn.close()


def log_reading(assessment: dict):
    """Insert a single crowd reading into the database."""
    with _lock:
        conn = _get_connection()
        conn.execute(
            """
            INSERT INTO crowd_logs (
                timestamp, current_count, smoothed_count,
                zone_name, zone_max_capacity, zone_safe_capacity,
                zone_area_m2, occupancy_ratio, occupancy_pct,
                safe_slots_left, density_per_m2, fruin_level,
                growth_rate, fill_rate_per_min,
                time_to_safe_breach_sec, time_to_max_breach_sec,
                risk_level, risk_score, surge_flag, duration_high
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                assessment.get("current_count", 0),
                assessment.get("smoothed_count", 0),
                assessment.get("zone_name", ""),
                assessment.get("zone_max_capacity", 0),
                assessment.get("zone_safe_capacity", 0),
                assessment.get("zone_area_m2", 0),
                assessment.get("occupancy_ratio", 0),
                assessment.get("occupancy_pct", ""),
                assessment.get("safe_slots_left", 0),
                assessment.get("density_per_m2", 0),
                assessment.get("fruin_level", ""),
                assessment.get("growth_rate", 0),
                assessment.get("fill_rate_per_min", 0),
                assessment.get("time_to_safe_breach_sec"),
                assessment.get("time_to_max_breach_sec"),
                assessment.get("risk_level", ""),
                assessment.get("risk_score", 0),
                int(assessment.get("surge_flag", False)),
                assessment.get("duration_in_high_state", 0),
            ),
        )
        conn.commit()
        conn.close()


def get_history(minutes: int = 2) -> list[dict]:
    """Return crowd log rows from the last N minutes."""
    with _lock:
        conn = _get_connection()
        rows = conn.execute(
            """
            SELECT
                timestamp, current_count, smoothed_count,
                zone_name, zone_max_capacity, zone_safe_capacity,
                zone_area_m2, occupancy_ratio, occupancy_pct,
                safe_slots_left, density_per_m2, fruin_level,
                growth_rate, fill_rate_per_min,
                time_to_safe_breach_sec, time_to_max_breach_sec,
                risk_level, risk_score, surge_flag, duration_high
            FROM crowd_logs
            WHERE timestamp >= datetime('now', ?)
            ORDER BY id ASC
            """,
            (f"-{minutes} minutes",),
        ).fetchall()
        conn.close()
    return [dict(r) for r in rows]