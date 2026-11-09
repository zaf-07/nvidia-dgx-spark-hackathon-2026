"""Campus Memory & Shared Sight Engine for GuideLens.

Provides on-campus, multi-student persistent memory for hazards, room
status, and delta change detection (What Changed?) backed by SQLite.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("guidelens.memory")

ALERT_TTL_SECONDS = float(os.getenv("CAMPUS_ALERT_TTL_HOURS", "12")) * 3600
DB_PATH = Path(os.getenv("CAMPUS_MEMORY_DB", str(Path(__file__).with_name("campus_memory.db"))))


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize empty campus memory tables; never invent campus reports."""
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS campus_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                location TEXT NOT NULL,
                category TEXT NOT NULL,
                label TEXT NOT NULL,
                description TEXT NOT NULL,
                severity TEXT NOT NULL DEFAULT 'medium',
                reported_by TEXT NOT NULL DEFAULT 'anonymous',
                created_at REAL NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1
            );
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_alerts_location
            ON campus_alerts(location, is_active);
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS location_baselines (
                location TEXT PRIMARY KEY,
                objects_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bridge_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                sender_role TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at REAL NOT NULL
            );
        """)
        conn.commit()


def _format_relative_time(timestamp: float) -> str:
    diff = max(0, time.time() - timestamp)
    if diff < 60:
        return "just now"
    minutes = int(diff // 60)
    if minutes < 60:
        return f"{minutes}m ago"
    hours = int(minutes // 60)
    if hours < 24:
        return f"{hours}h ago"
    days = int(hours // 24)
    return f"{days}d ago"


def get_active_alerts(location: str | None = None) -> list[dict[str, Any]]:
    """Retrieve active campus alerts, optionally filtered by location."""
    with _get_connection() as conn:
        if location and location.strip():
            cursor = conn.execute(
                """
                SELECT id, location, category, label, description, severity, reported_by, created_at, is_active
                FROM campus_alerts
                WHERE is_active = 1 AND created_at > ? AND (location LIKE ? OR location = 'All Campus')
                ORDER BY created_at DESC;
                """,
                (time.time() - ALERT_TTL_SECONDS, f"%{location.strip()}%",),
            )
        else:
            cursor = conn.execute(
                """
                SELECT id, location, category, label, description, severity, reported_by, created_at, is_active
                FROM campus_alerts
                WHERE is_active = 1 AND created_at > ?
                ORDER BY created_at DESC;
                """,
                (time.time() - ALERT_TTL_SECONDS,),
            )
        rows = cursor.fetchall()
        alerts = []
        for row in rows:
            alerts.append({
                "id": row["id"],
                "location": row["location"],
                "category": row["category"],
                "label": row["label"],
                "description": row["description"],
                "severity": row["severity"],
                "reported_by": row["reported_by"],
                "created_at": datetime.fromtimestamp(row["created_at"], timezone.utc).isoformat(),
                "relative_time": _format_relative_time(row["created_at"]),
                "is_active": bool(row["is_active"]),
            })
        return alerts


def add_alert(
    location: str,
    category: str,
    label: str,
    description: str,
    severity: str = "medium",
    reported_by: str = "anonymous",
) -> dict[str, Any]:
    """Record a new observation from any student phone into campus memory."""
    now = time.time()
    with _get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO campus_alerts
            (location, category, label, description, severity, reported_by, created_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1);
            """,
            (location, category, label, description, severity, reported_by, now),
        )
        conn.commit()
        alert_id = cursor.lastrowid

    logger.info("campus_alert_added id=%d location=%s label=%s", alert_id, location, label)
    return {
        "id": alert_id,
        "location": location,
        "category": category,
        "label": label,
        "description": description,
        "severity": severity,
        "reported_by": reported_by,
        "created_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        "relative_time": "just now",
        "is_active": True,
    }


def resolve_alert(alert_id: int) -> bool:
    """Mark a campus hazard as cleared/resolved."""
    with _get_connection() as conn:
        cursor = conn.execute(
            "UPDATE campus_alerts SET is_active = 0 WHERE id = ?;", (alert_id,)
        )
        conn.commit()
        return cursor.rowcount > 0


def detect_what_changed(
    location: str,
    current_objects: list[dict[str, Any]],
    update_baseline: bool = False,
) -> dict[str, Any]:
    """Compare currently detected objects against stored baseline for a room/location.

    Returns the geometric delta (what moved, what's new) instead of re-describing
    the static environment.
    """
    now = time.time()
    with _get_connection() as conn:
        cursor = conn.execute(
            "SELECT objects_json, updated_at FROM location_baselines WHERE location = ?;",
            (location,),
        )
        row = cursor.fetchone()

        if not row:
            # First time visiting this location: save current layout as baseline
            conn.execute(
                "INSERT INTO location_baselines (location, objects_json, updated_at) VALUES (?, ?, ?);",
                (location, json.dumps(current_objects), now),
            )
            conn.commit()
            return {
                "has_changed": False,
                "delta_summary": f"Baseline established for {location} with {len(current_objects)} objects.",
                "speech": f"Baseline established for {location}. I will only announce what changes from now on.",
                "added_objects": [],
                "removed_objects": [],
                "moved_objects": [],
            }

        baseline_objects = json.loads(row["objects_json"])

        if update_baseline:
            conn.execute(
                "UPDATE location_baselines SET objects_json = ?, updated_at = ? WHERE location = ?;",
                (json.dumps(current_objects), now, location),
            )
            conn.commit()

    # Compare current objects with baseline
    baseline_labels = [obj.get("label", "").lower() for obj in baseline_objects]
    current_labels = [obj.get("label", "").lower() for obj in current_objects]

    added: list[str] = []
    for label in current_labels:
        if label and baseline_labels.count(label) < current_labels.count(label) and label not in added:
            added.append(label)

    removed: list[str] = []
    for label in baseline_labels:
        if label and current_labels.count(label) < baseline_labels.count(label) and label not in removed:
            removed.append(label)

    # Detect moved obstacles in the central path (x ~ 0.35 to 0.65)
    moved: list[str] = []
    for curr in current_objects:
        label = curr.get("label", "").lower()
        curr_x = curr.get("x", 0.5)
        # Check if matched baseline object moved significantly
        matching_baseline = [b for b in baseline_objects if b.get("label", "").lower() == label]
        if matching_baseline:
            base_x = matching_baseline[0].get("x", 0.5)
            if abs(curr_x - base_x) > 0.20:
                moved.append(label)

    has_changed = bool(added or moved)

    if not has_changed:
        speech = f"Everything in {location} is in its usual place. Your path is clear."
        delta_summary = "No changes detected since earlier."
    else:
        parts = []
        if added:
            items_str = ", ".join(added[:2])
            parts.append(f"Something new is in your path: {items_str}")
        if moved:
            items_str = ", ".join(moved[:2])
            parts.append(f"A {items_str} on your usual path has moved")

        speech = f"Notice in {location}: " + ". ".join(parts) + "."
        delta_summary = "; ".join(parts)

    return {
        "has_changed": has_changed,
        "delta_summary": delta_summary,
        "speech": speech,
        "added_objects": added,
        "removed_objects": removed,
        "moved_objects": moved,
    }


def add_bridge_message(session_id: str, sender_role: str, text: str) -> dict[str, Any]:
    """Store a conversational message in the Deaf-Blind bridge."""
    now = time.time()
    with _get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO bridge_messages (session_id, sender_role, text, created_at)
            VALUES (?, ?, ?, ?);
            """,
            (session_id, sender_role, text.strip(), now),
        )
        conn.commit()
        msg_id = cursor.lastrowid

    return {
        "id": msg_id,
        "session_id": session_id,
        "sender_role": sender_role,
        "text": text.strip(),
        "created_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        "relative_time": "just now",
    }


def get_bridge_messages(session_id: str, limit: int = 20) -> list[dict[str, Any]]:
    """Retrieve recent bridge messages for a session."""
    with _get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT id, session_id, sender_role, text, created_at
            FROM bridge_messages
            WHERE session_id = ?
            ORDER BY created_at ASC
            LIMIT ?;
            """,
            (session_id, limit),
        )
        rows = cursor.fetchall()
        return [
            {
                "id": r["id"],
                "session_id": r["session_id"],
                "sender_role": r["sender_role"],
                "text": r["text"],
                "created_at": datetime.fromtimestamp(r["created_at"], timezone.utc).isoformat(),
                "relative_time": _format_relative_time(r["created_at"]),
            }
            for r in rows
        ]


# Initialize schema and seed data on module import
init_db()
