"""SQLite-backed conversation storage.

Two tables: ``conversations`` (one row per screening) and ``messages``
(append-only turn log). State is stored as JSON in the conversations row;
this is fine at our scale (~200/week) and avoids schema churn while we
iterate on field shape.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from .schema import Conversation, Decision, Message, ScreeningState


_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    candidate_id TEXT,
    state_json TEXT NOT NULL,
    decision TEXT NOT NULL,
    summary TEXT,
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
"""


class Storage:
    """Thread-safe SQLite store. One instance per process is fine."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.environ.get("HR_AGENT_DB_PATH", "./hr_agent.db")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def upsert_conversation(self, conv: Conversation) -> None:
        with self._lock, self._connect() as conn:
            now = datetime.now(timezone.utc).isoformat()
            conv.updated_at = datetime.now(timezone.utc)
            conn.execute(
                """
                INSERT INTO conversations (id, candidate_id, state_json, decision, summary, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    candidate_id=excluded.candidate_id,
                    state_json=excluded.state_json,
                    decision=excluded.decision,
                    summary=excluded.summary,
                    updated_at=excluded.updated_at
                """,
                (
                    conv.id,
                    conv.candidate_id,
                    conv.state.model_dump_json(),
                    conv.state.decision.value,
                    conv.summary,
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
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )

    def list_conversations(
        self,
        decision: Optional[Decision] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Return conversation rows (without messages) for analytics/listing."""
        with self._connect() as conn:
            if decision:
                rows = conn.execute(
                    "SELECT id, decision, state_json, created_at, updated_at FROM conversations "
                    "WHERE decision = ? ORDER BY updated_at DESC LIMIT ?",
                    (decision.value, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, decision, state_json, created_at, updated_at FROM conversations "
                    "ORDER BY updated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]

    def export_json(self, conv_id: str, out_dir: str = "./conversations") -> Path:
        """Dump a conversation to a single JSON file (recruiter-readable)."""
        conv = self.get_conversation(conv_id)
        if not conv:
            raise KeyError(conv_id)
        out_path = Path(out_dir) / f"{conv_id}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(conv.model_dump(mode="json"), f, indent=2, ensure_ascii=False)
        return out_path
