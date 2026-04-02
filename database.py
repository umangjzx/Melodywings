"""
database.py — Database Backend
MelodyWings Guard | Real-Time Content Safety System

Supports SQLite and PostgreSQL backends via adapter interface.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_SQLITE_DB_PATH = Path(__file__).parent / "melodywings_guard.db"
DEFAULT_BACKEND = os.getenv("DB_BACKEND", "sqlite").strip().lower()
DEFAULT_SQLITE_PATH = Path(os.getenv("DB_SQLITE_PATH", str(DEFAULT_SQLITE_DB_PATH)))
DEFAULT_POSTGRES_DSN = os.getenv("DB_POSTGRES_DSN") or os.getenv("DATABASE_URL")


class DatabaseAdapter(ABC):
    """Backend adapter contract for query execution and schema bootstrap."""

    @abstractmethod
    def init_schema(self) -> None:
        pass

    @abstractmethod
    def ensure_column(self, table: str, column: str, column_definition: str) -> None:
        pass

    @abstractmethod
    def execute(self, query: str, params: Optional[Sequence[Any]] = None):
        pass

    @abstractmethod
    def fetchall(self, query: str, params: Optional[Sequence[Any]] = None) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def fetchone(self, query: str, params: Optional[Sequence[Any]] = None) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def commit(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class SQLiteAdapter(DatabaseAdapter):
    """SQLite implementation of database adapter."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._configure_connection()

    def _configure_connection(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.execute("PRAGMA journal_mode = WAL")
        cursor.execute("PRAGMA synchronous = NORMAL")
        cursor.execute("PRAGMA temp_store = MEMORY")
        self.conn.commit()

    def init_schema(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                source TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                message TEXT,
                flagged BOOLEAN NOT NULL DEFAULT 0,
                severity TEXT DEFAULT 'medium',
                category TEXT,
                confidence REAL DEFAULT 0.0,
                confidence_by_reason TEXT,
                data TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
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

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS video_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
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

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS audio_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
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

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS transcript_segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                alert_id INTEGER,
                segment_text TEXT,
                confidence REAL,
                start_time REAL,
                end_time REAL,
                FOREIGN KEY(alert_id) REFERENCES alerts(id) ON DELETE CASCADE
            )
            """
        )

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_source ON alerts(source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_flagged ON alerts(flagged)")
        self.conn.commit()

    def ensure_column(self, table: str, column: str, column_definition: str) -> None:
        cursor = self.conn.cursor()
        cursor.execute(f"PRAGMA table_info({table})")
        existing_columns = {str(row[1]).lower() for row in cursor.fetchall()}
        if column.lower() not in existing_columns:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_definition}")
            self.conn.commit()

    def execute(self, query: str, params: Optional[Sequence[Any]] = None):
        cursor = self.conn.cursor()
        cursor.execute(query, tuple(params or []))
        return cursor

    def fetchall(self, query: str, params: Optional[Sequence[Any]] = None) -> List[Dict[str, Any]]:
        cursor = self.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def fetchone(self, query: str, params: Optional[Sequence[Any]] = None) -> Optional[Dict[str, Any]]:
        cursor = self.execute(query, params)
        row = cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.close()
        except sqlite3.Error:
            pass


class PostgreSQLAdapter(DatabaseAdapter):
    """PostgreSQL implementation of database adapter."""

    def __init__(self, dsn: str):
        if not dsn:
            raise ValueError("PostgreSQL backend selected but no DSN was provided.")

        try:
            import psycopg2  # type: ignore[import-not-found]
            from psycopg2.extras import RealDictCursor  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "psycopg2 is required for PostgreSQL backend. Install psycopg2-binary."
            ) from exc

        self._psycopg2 = psycopg2
        self._real_dict_cursor = RealDictCursor
        self.conn = psycopg2.connect(dsn)
        self.conn.autocommit = False

    def _pg_query(self, query: str) -> str:
        return query.replace("?", "%s")

    def init_schema(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id SERIAL PRIMARY KEY,
                run_id TEXT,
                source TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                message TEXT,
                flagged BOOLEAN NOT NULL DEFAULT FALSE,
                severity TEXT DEFAULT 'medium',
                category TEXT,
                confidence DOUBLE PRECISION DEFAULT 0.0,
                confidence_by_reason JSONB,
                data JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_alerts (
                id SERIAL PRIMARY KEY,
                run_id TEXT,
                alert_id INTEGER NOT NULL,
                text TEXT NOT NULL,
                has_profanity BOOLEAN,
                has_pii BOOLEAN,
                pii_types JSONB,
                toxicity_score DOUBLE PRECISION,
                toxicity_label TEXT,
                sentiment TEXT,
                sentiment_score DOUBLE PRECISION,
                entities JSONB,
                CONSTRAINT fk_chat_alert FOREIGN KEY (alert_id) REFERENCES alerts(id) ON DELETE CASCADE
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS video_alerts (
                id SERIAL PRIMARY KEY,
                run_id TEXT,
                alert_id INTEGER NOT NULL,
                frame_number INTEGER,
                timestamp_sec DOUBLE PRECISION,
                nsfw_label TEXT,
                nsfw_score DOUBLE PRECISION,
                emotion TEXT,
                CONSTRAINT fk_video_alert FOREIGN KEY (alert_id) REFERENCES alerts(id) ON DELETE CASCADE
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS audio_alerts (
                id SERIAL PRIMARY KEY,
                run_id TEXT,
                alert_id INTEGER NOT NULL,
                max_volume_db DOUBLE PRECISION,
                mean_volume_db DOUBLE PRECISION,
                silence_count INTEGER,
                speech_rate_wpm DOUBLE PRECISION,
                background_noise_db DOUBLE PRECISION,
                speaker_count INTEGER,
                CONSTRAINT fk_audio_alert FOREIGN KEY (alert_id) REFERENCES alerts(id) ON DELETE CASCADE
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS transcript_segments (
                id SERIAL PRIMARY KEY,
                run_id TEXT,
                alert_id INTEGER,
                segment_text TEXT,
                confidence DOUBLE PRECISION,
                start_time DOUBLE PRECISION,
                end_time DOUBLE PRECISION,
                CONSTRAINT fk_segment_alert FOREIGN KEY (alert_id) REFERENCES alerts(id) ON DELETE CASCADE
            )
            """
        )

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_source ON alerts(source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_flagged ON alerts(flagged)")
        self.conn.commit()

    def ensure_column(self, table: str, column: str, column_definition: str) -> None:
        query = """
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = %s AND column_name = %s
            LIMIT 1
        """
        with self.conn.cursor() as cursor:
            cursor.execute(query, (table, column))
            exists = cursor.fetchone() is not None
            if not exists:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_definition}")
        self.conn.commit()

    def execute(self, query: str, params: Optional[Sequence[Any]] = None):
        with self.conn.cursor() as cursor:
            cursor.execute(self._pg_query(query), tuple(params or []))
            if cursor.description and query.strip().lower().startswith("insert"):
                row = cursor.fetchone()
                return row
        return None

    def fetchall(self, query: str, params: Optional[Sequence[Any]] = None) -> List[Dict[str, Any]]:
        with self.conn.cursor(cursor_factory=self._real_dict_cursor) as cursor:
            cursor.execute(self._pg_query(query), tuple(params or []))
            return [dict(row) for row in cursor.fetchall()]

    def fetchone(self, query: str, params: Optional[Sequence[Any]] = None) -> Optional[Dict[str, Any]]:
        with self.conn.cursor(cursor_factory=self._real_dict_cursor) as cursor:
            cursor.execute(self._pg_query(query), tuple(params or []))
            row = cursor.fetchone()
            if row is None:
                return None
            return dict(row)

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


class AlertDatabase:
    """Database for storing and retrieving alerts."""

    def __init__(
        self,
        backend: Optional[str] = None,
        sqlite_path: Optional[Path] = None,
        postgres_dsn: Optional[str] = None,
    ):
        self.backend = (backend or DEFAULT_BACKEND or "sqlite").strip().lower()
        self.sqlite_path = sqlite_path or DEFAULT_SQLITE_PATH
        self.postgres_dsn = postgres_dsn if postgres_dsn is not None else DEFAULT_POSTGRES_DSN
        self._lock = threading.RLock()

        if self.backend == "postgres":
            self._adapter: DatabaseAdapter = PostgreSQLAdapter(self.postgres_dsn or "")
            logger.info("Database backend initialized: postgres")
        else:
            self._adapter = SQLiteAdapter(self.sqlite_path)
            logger.info(f"Database backend initialized: sqlite ({self.sqlite_path})")

        self._init_schema()

    def _init_schema(self) -> None:
        try:
            with self._lock:
                self._adapter.init_schema()
                self._run_migrations()
        except Exception as exc:
            logger.error(f"Database initialization failed: {exc}")
            raise

    def _run_migrations(self) -> None:
        """Apply additive schema migrations for older DB files."""
        self._adapter.ensure_column("alerts", "run_id", "TEXT")
        self._adapter.ensure_column("alerts", "confidence_by_reason", "TEXT")
        self._adapter.ensure_column("chat_alerts", "run_id", "TEXT")
        self._adapter.ensure_column("video_alerts", "run_id", "TEXT")
        self._adapter.ensure_column("audio_alerts", "run_id", "TEXT")
        self._adapter.ensure_column("transcript_segments", "run_id", "TEXT")

        self._adapter.execute("CREATE INDEX IF NOT EXISTS idx_alerts_run_id ON alerts(run_id)")
        self._adapter.execute("CREATE INDEX IF NOT EXISTS idx_chat_alerts_run_id ON chat_alerts(run_id)")
        self._adapter.execute("CREATE INDEX IF NOT EXISTS idx_video_alerts_run_id ON video_alerts(run_id)")
        self._adapter.execute("CREATE INDEX IF NOT EXISTS idx_audio_alerts_run_id ON audio_alerts(run_id)")
        self._adapter.execute("CREATE INDEX IF NOT EXISTS idx_transcript_segments_run_id ON transcript_segments(run_id)")
        self._adapter.commit()

    @staticmethod
    def _parse_json(value: Any, default: Any) -> Any:
        if value is None or value == "":
            return default
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return default

    def _normalize_detailed_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize booleans and JSON payloads returned from joined detail queries."""
        for row in rows:
            row["flagged"] = bool(row.get("flagged"))
            if row.get("has_profanity") is not None:
                row["has_profanity"] = bool(row["has_profanity"])
            if row.get("has_pii") is not None:
                row["has_pii"] = bool(row["has_pii"])

            row["data"] = self._parse_json(row.get("data"), None)
            row["pii_types"] = self._parse_json(row.get("pii_types"), [])
            row["entities"] = self._parse_json(row.get("entities"), [])
            row["confidence_by_reason"] = self._parse_json(row.get("confidence_by_reason"), {})

        return rows

    def _fetch_alerts_detailed(
        self,
        source: Optional[str] = None,
        flagged_only: bool = False,
        alert_id: Optional[int] = None,
        run_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[dict]:
        """Fetch joined alert detail rows with optional predicates."""
        query = """
            SELECT
                a.id,
                a.run_id,
                a.source,
                a.timestamp,
                a.message,
                a.flagged,
                a.severity,
                a.category,
                a.confidence,
                a.confidence_by_reason,
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
                aa.speaker_count,
                ts.segment_text AS transcript_segment_text,
                ts.confidence AS segment_confidence,
                ts.start_time AS segment_start_time,
                ts.end_time AS segment_end_time
            FROM alerts a
            LEFT JOIN chat_alerts ca ON ca.alert_id = a.id
            LEFT JOIN video_alerts va ON va.alert_id = a.id
            LEFT JOIN audio_alerts aa ON aa.alert_id = a.id
            LEFT JOIN transcript_segments ts ON ts.alert_id = a.id
            WHERE 1=1
        """
        params: list[Any] = []

        if source:
            query += " AND a.source = ?"
            params.append(source)

        if flagged_only:
            query += " AND a.flagged = 1"

        if alert_id is not None:
            query += " AND a.id = ?"
            params.append(alert_id)

        if run_id:
            query += " AND a.run_id = ?"
            params.append(run_id)

        query += " ORDER BY a.timestamp DESC"

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        with self._lock:
            rows = self._adapter.fetchall(query, params)

        return self._normalize_detailed_rows(rows)

    @staticmethod
    def _derive_primary_reason(reasons: list[str]) -> Optional[str]:
        if not reasons:
            return None
        first = str(reasons[0]).lower()
        if first.startswith("toxicity"):
            return "toxicity"
        if first.startswith("pii:"):
            return "pii"
        if "profanity" in first:
            return "profanity"
        if "sentiment" in first:
            return "sentiment"
        if first.startswith("emotion:"):
            return "emotion"
        if first.startswith("nsfw"):
            return "nsfw"
        return first

    def insert_alert(
        self,
        source: str,
        message: str = "",
        flagged: bool = False,
        severity: str = "medium",
        category: str = "",
        confidence: float | dict | None = 0.0,
        confidence_by_reason: Optional[dict[str, float]] = None,
        reasons: Optional[list[str]] = None,
        run_id: Optional[str] = None,
        data: Optional[dict] = None,
        commit: bool = True,
    ) -> int:
        """Insert a generic alert into the database and return its ID."""
        if isinstance(confidence, dict) and confidence_by_reason is None:
            confidence_by_reason = confidence

        numeric_conf = 0.0
        if isinstance(confidence, (int, float)):
            numeric_conf = float(confidence)

        if confidence_by_reason:
            primary_reason = self._derive_primary_reason(reasons or [])
            if primary_reason and primary_reason in confidence_by_reason:
                numeric_conf = float(confidence_by_reason.get(primary_reason, numeric_conf) or 0.0)
            elif numeric_conf == 0.0:
                try:
                    numeric_conf = max(float(v) for v in confidence_by_reason.values())
                except Exception:
                    numeric_conf = 0.0

        payload_data = dict(data or {})
        if confidence_by_reason:
            payload_data.setdefault("confidence_by_reason", confidence_by_reason)

        with self._lock:
            if self.backend == "postgres":
                row = self._adapter.execute(
                    """
                    INSERT INTO alerts
                    (run_id, source, timestamp, message, flagged, severity, category, confidence, confidence_by_reason, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?::jsonb, ?::jsonb)
                    RETURNING id
                    """,
                    (
                        run_id,
                        source,
                        datetime.now(timezone.utc).isoformat(),
                        message,
                        flagged,
                        severity,
                        category,
                        numeric_conf,
                        json.dumps(confidence_by_reason) if confidence_by_reason else None,
                        json.dumps(payload_data) if payload_data else None,
                    ),
                )
                if commit:
                    self._adapter.commit()
                if not row:
                    raise RuntimeError("Insert succeeded but no alert ID was returned")
                return int(row[0])

            cursor = self._adapter.execute(
                """
                INSERT INTO alerts
                (run_id, source, timestamp, message, flagged, severity, category, confidence, confidence_by_reason, data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    source,
                    datetime.now(timezone.utc).isoformat(),
                    message,
                    flagged,
                    severity,
                    category,
                    numeric_conf,
                    json.dumps(confidence_by_reason) if confidence_by_reason else None,
                    json.dumps(payload_data) if payload_data else None,
                ),
            )
            if commit:
                self._adapter.commit()
            alert_id = getattr(cursor, "lastrowid", None)
            if alert_id is None:
                raise RuntimeError("Insert succeeded but no alert ID was returned")
            return int(alert_id)

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
        entities: Optional[List[Any]] = None,
        run_id: Optional[str] = None,
        commit: bool = True,
    ):
        """Insert chat-specific alert details."""
        with self._lock:
            self._adapter.execute(
                """
                INSERT INTO chat_alerts
                (run_id, alert_id, text, has_profanity, has_pii, pii_types,
                 toxicity_score, toxicity_label, sentiment, sentiment_score, entities)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    alert_id,
                    text,
                    has_profanity,
                    has_pii,
                    json.dumps(pii_types) if pii_types is not None else None,
                    toxicity_score,
                    toxicity_label,
                    sentiment,
                    sentiment_score,
                    json.dumps(entities) if entities is not None else None,
                ),
            )
            if commit:
                self._adapter.commit()

    def insert_video_alert(
        self,
        alert_id: int,
        frame_number: int,
        timestamp_sec: float,
        nsfw_label: str,
        nsfw_score: float,
        emotion: str,
        run_id: Optional[str] = None,
        commit: bool = True,
    ):
        """Insert video frame-specific alert details."""
        with self._lock:
            self._adapter.execute(
                """
                INSERT INTO video_alerts
                (run_id, alert_id, frame_number, timestamp_sec, nsfw_label, nsfw_score, emotion)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, alert_id, frame_number, timestamp_sec, nsfw_label, nsfw_score, emotion),
            )
            if commit:
                self._adapter.commit()

    def insert_audio_alert(
        self,
        alert_id: int,
        max_volume_db: float,
        mean_volume_db: float,
        silence_count: int,
        speech_rate_wpm: float,
        background_noise_db: float,
        speaker_count: int,
        run_id: Optional[str] = None,
        commit: bool = True,
    ):
        """Insert audio-specific alert details."""
        with self._lock:
            self._adapter.execute(
                """
                INSERT INTO audio_alerts
                (run_id, alert_id, max_volume_db, mean_volume_db, silence_count,
                 speech_rate_wpm, background_noise_db, speaker_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    alert_id,
                    max_volume_db,
                    mean_volume_db,
                    silence_count,
                    speech_rate_wpm,
                    background_noise_db,
                    speaker_count,
                ),
            )
            if commit:
                self._adapter.commit()

    def insert_transcript_segment(
        self,
        segment_text: str,
        confidence: float,
        start_time: Optional[float],
        end_time: Optional[float],
        alert_id: Optional[int] = None,
        run_id: Optional[str] = None,
        commit: bool = True,
    ) -> None:
        """Insert one transcript segment row."""
        with self._lock:
            self._adapter.execute(
                """
                INSERT INTO transcript_segments
                (run_id, alert_id, segment_text, confidence, start_time, end_time)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, alert_id, segment_text, confidence, start_time, end_time),
            )
            if commit:
                self._adapter.commit()

    def get_all_alerts(
        self,
        source: Optional[str] = None,
        flagged_only: bool = False,
        run_id: Optional[str] = None,
    ) -> List[dict]:
        """Retrieve all alerts from the database."""
        query = "SELECT * FROM alerts WHERE 1=1"
        params: list[Any] = []

        if source:
            query += " AND source = ?"
            params.append(source)
        if flagged_only:
            query += " AND flagged = 1"
        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)

        query += " ORDER BY timestamp DESC"

        with self._lock:
            rows = self._adapter.fetchall(query, params)
        return rows

    def get_alerts_detailed(
        self,
        source: Optional[str] = None,
        flagged_only: bool = False,
        run_id: Optional[str] = None,
    ) -> List[dict]:
        """Retrieve alerts with joined source-specific detail tables."""
        return self._fetch_alerts_detailed(source=source, flagged_only=flagged_only, run_id=run_id)

    def get_alert_detailed_by_id(self, alert_id: int) -> Optional[dict]:
        """Retrieve one detailed alert record by primary id."""
        rows = self._fetch_alerts_detailed(alert_id=alert_id, limit=1)
        if not rows:
            return None
        return rows[0]

    def get_alert_stats(self, run_id: Optional[str] = None) -> dict:
        """Get summary statistics of all alerts."""
        with self._lock:
            if run_id:
                total_row = self._adapter.fetchone("SELECT COUNT(*) AS count FROM alerts WHERE run_id = ?", (run_id,))
                flagged_row = self._adapter.fetchone(
                    "SELECT COUNT(*) AS count FROM alerts WHERE flagged = 1 AND run_id = ?", (run_id,)
                )
                by_source = self._adapter.fetchall(
                    "SELECT source, COUNT(*) AS count FROM alerts WHERE run_id = ? GROUP BY source", (run_id,)
                )
                by_severity = self._adapter.fetchall(
                    "SELECT severity, COUNT(*) AS count FROM alerts WHERE run_id = ? GROUP BY severity", (run_id,)
                )
            else:
                total_row = self._adapter.fetchone("SELECT COUNT(*) AS count FROM alerts")
                flagged_row = self._adapter.fetchone("SELECT COUNT(*) AS count FROM alerts WHERE flagged = 1")
                by_source = self._adapter.fetchall("SELECT source, COUNT(*) AS count FROM alerts GROUP BY source")
                by_severity = self._adapter.fetchall("SELECT severity, COUNT(*) AS count FROM alerts GROUP BY severity")

        total = int((total_row or {}).get("count", 0))
        flagged = int((flagged_row or {}).get("count", 0))
        return {
            "total_alerts": total,
            "flagged_alerts": flagged,
            "by_source": {str(row.get("source", "")): int(row.get("count", 0)) for row in by_source},
            "by_severity": {str(row.get("severity", "")): int(row.get("count", 0)) for row in by_severity},
        }

    def clear_all_alerts(self):
        """Delete all alerts from the database."""
        with self._lock:
            self._adapter.execute("DELETE FROM transcript_segments")
            self._adapter.execute("DELETE FROM chat_alerts")
            self._adapter.execute("DELETE FROM video_alerts")
            self._adapter.execute("DELETE FROM audio_alerts")
            self._adapter.execute("DELETE FROM alerts")
            self._adapter.commit()
        logger.info("All alerts cleared from database")

    def export_to_json(self, output_path: Path) -> bool:
        """Export all alerts to JSON file."""
        try:
            alerts = self.get_all_alerts()
            with open(output_path, "w", encoding="utf-8") as file_obj:
                json.dump(alerts, file_obj, indent=2, ensure_ascii=False)
            logger.info(f"Alerts exported to {output_path}")
            return True
        except Exception as exc:
            logger.error(f"Export failed: {exc}")
            return False

    def close(self):
        """Close the underlying shared database connection."""
        with self._lock:
            self._adapter.close()

    def commit(self):
        """Commit the current transaction."""
        with self._lock:
            self._adapter.commit()


# Global instance
_db_instance: Optional[AlertDatabase] = None


def get_db() -> AlertDatabase:
    """Get or initialize the global database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = AlertDatabase()
    return _db_instance


def init_db(
    db_path: Optional[Path] = None,
    backend: Optional[str] = None,
    postgres_dsn: Optional[str] = None,
) -> AlertDatabase:
    """Initialize a database instance with optional backend override."""
    global _db_instance
    if _db_instance is not None:
        _db_instance.close()

    resolved_backend = (backend or DEFAULT_BACKEND or "sqlite").strip().lower()
    if resolved_backend == "postgres":
        _db_instance = AlertDatabase(backend="postgres", postgres_dsn=postgres_dsn)
    else:
        sqlite_path = db_path or DEFAULT_SQLITE_PATH
        _db_instance = AlertDatabase(backend="sqlite", sqlite_path=sqlite_path)
    return _db_instance
