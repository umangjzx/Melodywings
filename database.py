"""
database.py — SQLite Database Backend
MelodyWings Guard | Real-Time Content Safety System

Manages persistent storage of all alerts using SQLite with automatic
schema creation and migration support.
"""

import sqlite3
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "melodywings_guard.db"


class AlertDatabase:
    """SQLite database for storing and retrieving alerts."""

    def __init__(self, db_path: Path = DB_PATH):
        """Initialize database connection and create schema if needed."""
        self.db_path = db_path
        self._init_schema()

    def _init_schema(self):
        """Create database tables if they don't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Main alerts table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        message TEXT,
                        flagged BOOLEAN NOT NULL DEFAULT 0,
                        severity TEXT DEFAULT 'medium',
                        category TEXT,
                        confidence REAL DEFAULT 0.0,
                        data JSON,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )

                # Chat analysis specific
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        alert_id INTEGER NOT NULL,
                        text TEXT NOT NULL,
                        has_profanity BOOLEAN,
                        has_pii BOOLEAN,
                        pii_types TEXT,
                        toxicity_score REAL,
                        toxicity_label TEXT,
                        sentiment TEXT,
                        sentiment_score REAL,
                        entities TEXT,
                        FOREIGN KEY(alert_id) REFERENCES alerts(id) ON DELETE CASCADE
                    )
                    """
                )

                # Video frame analysis
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS video_alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        alert_id INTEGER NOT NULL,
                        frame_number INTEGER,
                        timestamp_sec REAL,
                        nsfw_label TEXT,
                        nsfw_score REAL,
                        emotion TEXT,
                        FOREIGN KEY(alert_id) REFERENCES alerts(id) ON DELETE CASCADE
                    )
                    """
                )

                # Audio analysis
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audio_alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        alert_id INTEGER NOT NULL,
                        max_volume_db REAL,
                        mean_volume_db REAL,
                        silence_count INTEGER,
                        speech_rate_wpm REAL,
                        background_noise_db REAL,
                        speaker_count INTEGER,
                        FOREIGN KEY(alert_id) REFERENCES alerts(id) ON DELETE CASCADE
                    )
                    """
                )

                # Transcript confidence tracking
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS transcript_segments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        alert_id INTEGER,
                        segment_text TEXT,
                        confidence REAL,
                        start_time REAL,
                        end_time REAL,
                        FOREIGN KEY(alert_id) REFERENCES alerts(id) ON DELETE CASCADE
                    )
                    """
                )

                # Create indexes for performance
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_source ON alerts(source)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_timestamp ON alerts(timestamp)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_flagged ON alerts(flagged)"
                )

                conn.commit()
                logger.info(f"Database schema initialized: {self.db_path}")
        except sqlite3.Error as exc:
            logger.error(f"Database initialization failed: {exc}")
            raise

    def insert_alert(
        self,
        source: str,
        message: str = "",
        flagged: bool = False,
        severity: str = "medium",
        category: str = "",
        confidence: float = 0.0,
        data: Optional[dict] = None,
    ) -> int:
        """
        Insert a generic alert into the database.

        Args:
            source: "chat", "video_frame", "transcript", or "audio"
            message: Human-readable message
            flagged: Whether the alert is flagged
            severity: "low", "medium", "high", "critical"
            category: Alert category
            confidence: Confidence score (0-1)
            data: Additional JSON data

        Returns:
            The inserted alert ID.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO alerts
                    (source, timestamp, message, flagged, severity, category, confidence, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source,
                        datetime.now(timezone.utc).isoformat(),
                        message,
                        flagged,
                        severity,
                        category,
                        confidence,
                        json.dumps(data) if data else None,
                    ),
                )
                conn.commit()
                alert_id = cursor.lastrowid
                if alert_id is None:
                    raise sqlite3.Error("Insert succeeded but no alert ID was returned")
                return int(alert_id)
        except sqlite3.Error as exc:
            logger.error(f"Failed to insert alert: {exc}")
            raise

    def insert_chat_alert(
        self,
        alert_id: int,
        text: str,
        has_profanity: bool = False,
        has_pii: bool = False,
        pii_types: Optional[List[str]] = None,
        toxicity_score: float = 0.0,
        toxicity_label: str = "",
        sentiment: str = "",
        sentiment_score: float = 0.0,
        entities: Optional[List[str]] = None,
    ):
        """Insert chat-specific alert details."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO chat_alerts
                    (alert_id, text, has_profanity, has_pii, pii_types,
                     toxicity_score, toxicity_label, sentiment, sentiment_score, entities)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        alert_id,
                        text,
                        has_profanity,
                        has_pii,
                        json.dumps(pii_types) if pii_types else None,
                        toxicity_score,
                        toxicity_label,
                        sentiment,
                        sentiment_score,
                        json.dumps(entities) if entities else None,
                    ),
                )
                conn.commit()
        except sqlite3.Error as exc:
            logger.error(f"Failed to insert chat alert: {exc}")

    def insert_video_alert(
        self,
        alert_id: int,
        frame_number: int,
        timestamp_sec: float,
        nsfw_label: str,
        nsfw_score: float,
        emotion: str,
    ):
        """Insert video frame-specific alert details."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO video_alerts
                    (alert_id, frame_number, timestamp_sec, nsfw_label, nsfw_score, emotion)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (alert_id, frame_number, timestamp_sec, nsfw_label, nsfw_score, emotion),
                )
                conn.commit()
        except sqlite3.Error as exc:
            logger.error(f"Failed to insert video alert: {exc}")

    def insert_audio_alert(
        self,
        alert_id: int,
        max_volume_db: float,
        mean_volume_db: float,
        silence_count: int,
        speech_rate_wpm: float,
        background_noise_db: float,
        speaker_count: int,
    ):
        """Insert audio-specific alert details."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO audio_alerts
                    (alert_id, max_volume_db, mean_volume_db, silence_count,
                     speech_rate_wpm, background_noise_db, speaker_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        alert_id,
                        max_volume_db,
                        mean_volume_db,
                        silence_count,
                        speech_rate_wpm,
                        background_noise_db,
                        speaker_count,
                    ),
                )
                conn.commit()
        except sqlite3.Error as exc:
            logger.error(f"Failed to insert audio alert: {exc}")

    def get_all_alerts(
        self, source: Optional[str] = None, flagged_only: bool = False
    ) -> List[dict]:
        """
        Retrieve all alerts from the database.

        Args:
            source: Filter by source ("chat", "video_frame", "transcript", "audio")
            flagged_only: Only return flagged alerts

        Returns:
            List of alert dicts.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                query = "SELECT * FROM alerts WHERE 1=1"
                params = []

                if source:
                    query += " AND source = ?"
                    params.append(source)

                if flagged_only:
                    query += " AND flagged = 1"

                query += " ORDER BY timestamp DESC"

                cursor.execute(query, params)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as exc:
            logger.error(f"Failed to retrieve alerts: {exc}")
            return []

    def get_alerts_detailed(
        self, source: Optional[str] = None, flagged_only: bool = False
    ) -> List[dict]:
        """
        Retrieve alerts with joined source-specific detail tables.

        Args:
            source: Filter by source ("chat", "video_frame", "transcript", "audio")
            flagged_only: Only return flagged alerts

        Returns:
            List of joined alert dicts with normalized JSON fields.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                query = """
                    SELECT
                        a.id,
                        a.source,
                        a.timestamp,
                        a.message,
                        a.flagged,
                        a.severity,
                        a.category,
                        a.confidence,
                        a.data,
                        a.created_at,
                        ca.text AS chat_text,
                        ca.has_profanity,
                        ca.has_pii,
                        ca.pii_types,
                        ca.toxicity_score,
                        ca.toxicity_label,
                        ca.sentiment,
                        ca.sentiment_score,
                        ca.entities,
                        va.frame_number,
                        va.timestamp_sec,
                        va.nsfw_label,
                        va.nsfw_score,
                        va.emotion,
                        aa.max_volume_db,
                        aa.mean_volume_db,
                        aa.silence_count,
                        aa.speech_rate_wpm,
                        aa.background_noise_db,
                        aa.speaker_count
                    FROM alerts a
                    LEFT JOIN chat_alerts ca ON ca.alert_id = a.id
                    LEFT JOIN video_alerts va ON va.alert_id = a.id
                    LEFT JOIN audio_alerts aa ON aa.alert_id = a.id
                    WHERE 1=1
                """
                params = []

                if source:
                    query += " AND a.source = ?"
                    params.append(source)

                if flagged_only:
                    query += " AND a.flagged = 1"

                query += " ORDER BY a.timestamp DESC"

                cursor.execute(query, params)
                rows = [dict(row) for row in cursor.fetchall()]

                for row in rows:
                    row["flagged"] = bool(row.get("flagged"))
                    if row.get("has_profanity") is not None:
                        row["has_profanity"] = bool(row["has_profanity"])
                    if row.get("has_pii") is not None:
                        row["has_pii"] = bool(row["has_pii"])

                    for json_field in ("data", "pii_types", "entities"):
                        value = row.get(json_field)
                        if value is None or value == "":
                            row[json_field] = [] if json_field in ("pii_types", "entities") else None
                            continue
                        try:
                            row[json_field] = json.loads(value)
                        except (TypeError, json.JSONDecodeError):
                            row[json_field] = [] if json_field in ("pii_types", "entities") else None

                return rows
        except sqlite3.Error as exc:
            logger.error(f"Failed to retrieve detailed alerts: {exc}")
            return []

    def get_alert_stats(self) -> dict:
        """Get summary statistics of all alerts."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                total = cursor.execute(
                    "SELECT COUNT(*) FROM alerts"
                ).fetchone()[0]
                flagged = cursor.execute(
                    "SELECT COUNT(*) FROM alerts WHERE flagged = 1"
                ).fetchone()[0]
                by_source = cursor.execute(
                    "SELECT source, COUNT(*) as count FROM alerts GROUP BY source"
                ).fetchall()
                by_severity = cursor.execute(
                    "SELECT severity, COUNT(*) as count FROM alerts GROUP BY severity"
                ).fetchall()

                return {
                    "total_alerts": total,
                    "flagged_alerts": flagged,
                    "by_source": {row[0]: row[1] for row in by_source},
                    "by_severity": {row[0]: row[1] for row in by_severity},
                }
        except sqlite3.Error as exc:
            logger.error(f"Failed to retrieve stats: {exc}")
            return {}

    def clear_all_alerts(self):
        """Delete all alerts from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM alerts")
                cursor.execute("DELETE FROM chat_alerts")
                cursor.execute("DELETE FROM video_alerts")
                cursor.execute("DELETE FROM audio_alerts")
                cursor.execute("DELETE FROM transcript_segments")
                conn.commit()
                logger.info("All alerts cleared from database")
        except sqlite3.Error as exc:
            logger.error(f"Failed to clear alerts: {exc}")

    def export_to_json(self, output_path: Path) -> bool:
        """Export all alerts to JSON file."""
        try:
            alerts = self.get_all_alerts()
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(alerts, f, indent=2, ensure_ascii=False)
            logger.info(f"Alerts exported to {output_path}")
            return True
        except (sqlite3.Error, IOError) as exc:
            logger.error(f"Export failed: {exc}")
            return False


# Global instance
_db_instance: Optional[AlertDatabase] = None


def get_db() -> AlertDatabase:
    """Get or initialize the global database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = AlertDatabase()
    return _db_instance


def init_db(db_path: Path = DB_PATH) -> AlertDatabase:
    """Initialize a database instance."""
    global _db_instance
    _db_instance = AlertDatabase(db_path)
    return _db_instance
