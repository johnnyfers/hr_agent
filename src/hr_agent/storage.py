"""Conversation, client, and job storage.

Tables:
- ``clients``       — one row per hiring client (e.g. Grupo Sazón).
- ``jobs``          — one row per screening flow. ``spec_json`` holds the
                      JobSpec body (fields, FAQ, service areas).
- ``conversations`` — one row per screening session. Linked to a job.
- ``messages``      — append-only turn log.

State, JobSpec body, and FAQ are stored as JSON in TEXT columns. We never
query *into* the JSON, so JSONB / relational decomposition would be premature.

Backend selection: ``DATABASE_URL=postgresql://...`` picks Postgres,
otherwise SQLite at ``HR_AGENT_DB_PATH``. Tests stay on SQLite.
"""

from __future__ import annotations

import abc
import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from .jobspec import Client, JobSpec
from .schema import Conversation, ConversationAnalytics, Decision, Message, ScreeningState

log = logging.getLogger(__name__)


# --- Abstract interface -----------------------------------------------------


class Storage(abc.ABC):
    """Backend-agnostic storage interface."""

    # Conversations
    @abc.abstractmethod
    def upsert_conversation(self, conv: Conversation) -> None: ...

    @abc.abstractmethod
    def append_message(self, conversation_id: str, msg: Message) -> None: ...

    @abc.abstractmethod
    def get_conversation(self, conv_id: str) -> Optional[Conversation]: ...

    @abc.abstractmethod
    def list_conversations(
        self,
        decision: Optional[Decision] = None,
        job_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]: ...

    # Clients
    @abc.abstractmethod
    def upsert_client(self, client: Client) -> None: ...

    @abc.abstractmethod
    def list_clients(self) -> list[Client]: ...

    # Jobs
    @abc.abstractmethod
    def upsert_job(self, job: JobSpec) -> None: ...

    @abc.abstractmethod
    def get_job(self, job_id: str) -> Optional[JobSpec]: ...

    @abc.abstractmethod
    def list_jobs(self, client_id: Optional[str] = None) -> list[JobSpec]: ...


def make_storage() -> Storage:
    """Pick a backend from env.

    ``DATABASE_URL=postgres[ql]://...``  → PostgresStorage
    otherwise                            → SqliteStorage at HR_AGENT_DB_PATH
    """
    url = (os.environ.get("DATABASE_URL") or "").strip()
    if url.startswith(("postgres://", "postgresql://")):
        log.info("storage: using PostgresStorage")
        return PostgresStorage(url)
    db_path = os.environ.get("HR_AGENT_DB_PATH", "./hr_agent.db")
    log.info("storage: using SqliteStorage(%s)", db_path)
    return SqliteStorage(db_path)


# --- SQLite backend ---------------------------------------------------------


_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS clients (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    client_id TEXT NOT NULL REFERENCES clients(id),
    title_es TEXT NOT NULL,
    title_en TEXT NOT NULL,
    description TEXT,
    spec_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_client ON jobs(client_id);

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    candidate_id TEXT,
    job_id TEXT NOT NULL REFERENCES jobs(id),
    state_json TEXT NOT NULL,
    decision TEXT NOT NULL,
    summary TEXT,
    analytics_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_conv_decision ON conversations(decision);
CREATE INDEX IF NOT EXISTS idx_conv_updated ON conversations(updated_at);
CREATE INDEX IF NOT EXISTS idx_conv_job ON conversations(job_id);
"""


class SqliteStorage(Storage):
    """Thread-safe SQLite store."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.environ.get("HR_AGENT_DB_PATH", "./hr_agent.db")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SQLITE_SCHEMA)
            cols = [r["name"] for r in conn.execute("PRAGMA table_info(conversations)").fetchall()]
            if "analytics_json" not in cols:
                conn.execute("ALTER TABLE conversations ADD COLUMN analytics_json TEXT")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # --- conversations ---

    def upsert_conversation(self, conv: Conversation) -> None:
        conv.recompute_analytics()
        with self._lock, self._connect() as conn:
            now = datetime.now(timezone.utc).isoformat()
            conv.updated_at = datetime.now(timezone.utc)
            conn.execute(
                """
                INSERT INTO conversations (id, candidate_id, job_id, state_json, decision, summary, analytics_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    candidate_id=excluded.candidate_id,
                    job_id=excluded.job_id,
                    state_json=excluded.state_json,
                    decision=excluded.decision,
                    summary=excluded.summary,
                    analytics_json=excluded.analytics_json,
                    updated_at=excluded.updated_at
                """,
                (
                    conv.id,
                    conv.candidate_id,
                    conv.state.job_id,
                    conv.state.model_dump_json(),
                    conv.state.decision.value,
                    conv.summary,
                    conv.analytics.model_dump_json(),
                    conv.created_at.isoformat(),
                    now,
                ),
            )

    def append_message(self, conversation_id: str, msg: Message) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (conversation_id, msg.role, msg.content, msg.timestamp.isoformat()),
            )

    def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ?", (conv_id,)
            ).fetchone()
            if not row:
                return None
            messages = [
                Message(
                    role=m["role"],
                    content=m["content"],
                    timestamp=datetime.fromisoformat(m["timestamp"]),
                )
                for m in conn.execute(
                    "SELECT role, content, timestamp FROM messages "
                    "WHERE conversation_id = ? ORDER BY id ASC",
                    (conv_id,),
                ).fetchall()
            ]
            return Conversation(
                id=row["id"],
                candidate_id=row["candidate_id"],
                state=ScreeningState.model_validate_json(row["state_json"]),
                summary=row["summary"],
                messages=messages,
                analytics=(
                    ConversationAnalytics.model_validate_json(row["analytics_json"])
                    if row["analytics_json"]
                    else ConversationAnalytics()
                ),
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )

    def list_conversations(
        self,
        decision: Optional[Decision] = None,
        job_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        with self._connect() as conn:
            clauses = []
            params: list = []
            if decision:
                clauses.append("decision = ?")
                params.append(decision.value)
            if job_id:
                clauses.append("job_id = ?")
                params.append(job_id)
            where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
            params.append(limit)
            rows = conn.execute(
                "SELECT id, decision, job_id, state_json, analytics_json, created_at, updated_at "
                "FROM conversations" + where +
                " ORDER BY updated_at DESC LIMIT ?",
                params,
            ).fetchall()
            return [dict(r) for r in rows]

    # --- clients ---

    def upsert_client(self, client: Client) -> None:
        with self._lock, self._connect() as conn:
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT INTO clients (id, name, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET name=excluded.name
                """,
                (client.id, client.name, now),
            )

    def list_clients(self) -> list[Client]:
        with self._connect() as conn:
            rows = conn.execute("SELECT id, name FROM clients ORDER BY name").fetchall()
            return [Client(id=r["id"], name=r["name"]) for r in rows]

    # --- jobs ---

    def upsert_job(self, job: JobSpec) -> None:
        with self._lock, self._connect() as conn:
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT INTO jobs (id, client_id, title_es, title_en, description, spec_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    client_id=excluded.client_id,
                    title_es=excluded.title_es,
                    title_en=excluded.title_en,
                    description=excluded.description,
                    spec_json=excluded.spec_json,
                    updated_at=excluded.updated_at
                """,
                (
                    job.job_id,
                    job.client.id,
                    job.title.get("es", ""),
                    job.title.get("en", ""),
                    job.description,
                    job.model_dump_json(),
                    now,
                    now,
                ),
            )

    def get_job(self, job_id: str) -> Optional[JobSpec]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT spec_json FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if not row:
                return None
            return JobSpec.model_validate_json(row["spec_json"])

    def list_jobs(self, client_id: Optional[str] = None) -> list[JobSpec]:
        with self._connect() as conn:
            if client_id:
                rows = conn.execute(
                    "SELECT spec_json FROM jobs WHERE client_id = ? ORDER BY id",
                    (client_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT spec_json FROM jobs ORDER BY id"
                ).fetchall()
            return [JobSpec.model_validate_json(r["spec_json"]) for r in rows]


# --- Postgres backend -------------------------------------------------------


_PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS clients (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    client_id TEXT NOT NULL REFERENCES clients(id),
    title_es TEXT NOT NULL,
    title_en TEXT NOT NULL,
    description TEXT,
    spec_json TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_client ON jobs(client_id);

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    candidate_id TEXT,
    job_id TEXT NOT NULL REFERENCES jobs(id),
    state_json TEXT NOT NULL,
    decision TEXT NOT NULL,
    summary TEXT,
    analytics_json TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id BIGSERIAL PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_conv_decision ON conversations(decision);
CREATE INDEX IF NOT EXISTS idx_conv_updated ON conversations(updated_at);
CREATE INDEX IF NOT EXISTS idx_conv_job ON conversations(job_id);
"""


class PostgresStorage(Storage):
    """psycopg3-backed Postgres store. Connection-per-operation."""

    def __init__(self, dsn: str, max_init_retries: int = 30, retry_delay: float = 1.0):
        self.dsn = dsn
        last_err: Optional[Exception] = None
        for attempt in range(max_init_retries):
            try:
                with self._connect() as conn, conn.cursor() as cur:
                    cur.execute(_PG_SCHEMA)
                    cur.execute("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS analytics_json TEXT")
                return
            except Exception as e:
                last_err = e
                if attempt < max_init_retries - 1:
                    log.info(
                        "postgres init failed (attempt %d): %s — retrying",
                        attempt + 1,
                        e,
                    )
                    time.sleep(retry_delay)
        raise RuntimeError(f"could not connect to postgres after {max_init_retries} attempts") from last_err

    @contextmanager
    def _connect(self):
        import psycopg

        conn = psycopg.connect(self.dsn, connect_timeout=5)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # --- conversations ---

    def upsert_conversation(self, conv: Conversation) -> None:
        conv.recompute_analytics()
        now = datetime.now(timezone.utc)
        conv.updated_at = now
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations (id, candidate_id, job_id, state_json, decision, summary, analytics_json, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    candidate_id = EXCLUDED.candidate_id,
                    job_id       = EXCLUDED.job_id,
                    state_json   = EXCLUDED.state_json,
                    decision     = EXCLUDED.decision,
                    summary      = EXCLUDED.summary,
                    analytics_json = EXCLUDED.analytics_json,
                    updated_at   = EXCLUDED.updated_at
                """,
                (
                    conv.id,
                    conv.candidate_id,
                    conv.state.job_id,
                    conv.state.model_dump_json(),
                    conv.state.decision.value,
                    conv.summary,
                    conv.analytics.model_dump_json(),
                    conv.created_at,
                    now,
                ),
            )

    def append_message(self, conversation_id: str, msg: Message) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (%s, %s, %s, %s)",
                (conversation_id, msg.role, msg.content, msg.timestamp),
            )

    def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, candidate_id, state_json, summary, analytics_json, created_at, updated_at "
                "FROM conversations WHERE id = %s",
                (conv_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            id_, candidate_id, state_json, summary, analytics_json, created_at, updated_at = row

            cur.execute(
                "SELECT role, content, timestamp FROM messages "
                "WHERE conversation_id = %s ORDER BY id ASC",
                (conv_id,),
            )
            messages = [
                Message(role=r, content=c, timestamp=t) for (r, c, t) in cur.fetchall()
            ]
            return Conversation(
                id=id_,
                candidate_id=candidate_id,
                state=ScreeningState.model_validate_json(state_json),
                summary=summary,
                messages=messages,
                analytics=(
                    ConversationAnalytics.model_validate_json(analytics_json)
                    if analytics_json
                    else ConversationAnalytics()
                ),
                created_at=created_at,
                updated_at=updated_at,
            )

    def list_conversations(
        self,
        decision: Optional[Decision] = None,
        job_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        clauses = []
        params: list = []
        if decision:
            clauses.append("decision = %s")
            params.append(decision.value)
        if job_id:
            clauses.append("job_id = %s")
            params.append(job_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, decision, job_id, state_json, analytics_json, created_at, updated_at "
                "FROM conversations" + where +
                " ORDER BY updated_at DESC LIMIT %s",
                params,
            )
            cols = [d[0] for d in cur.description]
            out: list[dict] = []
            for row in cur.fetchall():
                rec = dict(zip(cols, row))
                for k in ("created_at", "updated_at"):
                    v = rec.get(k)
                    if isinstance(v, datetime):
                        rec[k] = v.isoformat()
                out.append(rec)
            return out

    # --- clients ---

    def upsert_client(self, client: Client) -> None:
        now = datetime.now(timezone.utc)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO clients (id, name, created_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name
                """,
                (client.id, client.name, now),
            )

    def list_clients(self) -> list[Client]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT id, name FROM clients ORDER BY name")
            return [Client(id=r[0], name=r[1]) for r in cur.fetchall()]

    # --- jobs ---

    def upsert_job(self, job: JobSpec) -> None:
        now = datetime.now(timezone.utc)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO jobs (id, client_id, title_es, title_en, description, spec_json, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    client_id   = EXCLUDED.client_id,
                    title_es    = EXCLUDED.title_es,
                    title_en    = EXCLUDED.title_en,
                    description = EXCLUDED.description,
                    spec_json   = EXCLUDED.spec_json,
                    updated_at  = EXCLUDED.updated_at
                """,
                (
                    job.job_id,
                    job.client.id,
                    job.title.get("es", ""),
                    job.title.get("en", ""),
                    job.description,
                    job.model_dump_json(),
                    now,
                    now,
                ),
            )

    def get_job(self, job_id: str) -> Optional[JobSpec]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT spec_json FROM jobs WHERE id = %s", (job_id,))
            row = cur.fetchone()
            if not row:
                return None
            return JobSpec.model_validate_json(row[0])

    def list_jobs(self, client_id: Optional[str] = None) -> list[JobSpec]:
        with self._connect() as conn, conn.cursor() as cur:
            if client_id:
                cur.execute(
                    "SELECT spec_json FROM jobs WHERE client_id = %s ORDER BY id",
                    (client_id,),
                )
            else:
                cur.execute("SELECT spec_json FROM jobs ORDER BY id")
            return [JobSpec.model_validate_json(r[0]) for r in cur.fetchall()]
