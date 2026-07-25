import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

VALID_STATUSES = ("approved", "rejected", "timed_out")


class ApprovalStore:
    """
    SQLite-backed store for approval requests, persisted to disk so a
    decision survives process restarts and both the orchestrator process
    and the dashboard/API process (typically separate processes sharing
    this file) see the same state.
    """

    def __init__(self, db_path: str = "approvals.db"):
        self.db_path = db_path
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS approval_requests (
                    id TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    diff TEXT NOT NULL,
                    final_score REAL NOT NULL,
                    metrics TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    decided_at TEXT,
                    decision_note TEXT
                )
                """
            )
            conn.commit()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            yield conn
        finally:
            conn.close()

    def create_request(
        self, candidate_id: str, goal: str, diff: str, final_score: float, metrics: Dict[str, Any]
    ) -> str:
        request_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO approval_requests "
                "(id, candidate_id, goal, diff, final_score, metrics, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)",
                (
                    request_id,
                    candidate_id,
                    goal,
                    diff,
                    final_score,
                    json.dumps(metrics),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
        return request_id

    def get_request(self, request_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM approval_requests WHERE id = ?", (request_id,)
            ).fetchone()
            return self._row_to_dict(row) if row else None

    def list_pending(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM approval_requests WHERE status = 'pending' ORDER BY created_at ASC"
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def list_all(self, limit: int = 200) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM approval_requests ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def decide(self, request_id: str, status: str, note: Optional[str] = None) -> bool:
        """
        Records a decision. Only transitions a request that is currently
        'pending' - returns False (no-op) if it was already decided (e.g.
        already timed out), so a late human click can never override an
        automatic timeout hold, or vice versa.
        """
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid status: {status!r}, must be one of {VALID_STATUSES}")
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE approval_requests SET status = ?, decided_at = ?, decision_note = ? "
                "WHERE id = ? AND status = 'pending'",
                (status, datetime.now(timezone.utc).isoformat(), note, request_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def timeout_stale_requests(self, timeout_seconds: float) -> int:
        """
        Marks every pending request older than timeout_seconds as timed_out.
        A request is normally timed out by whichever process is awaiting it
        (see approval/gate.py:await_approval_decision) - but if that process
        crashed or was killed before its own deadline check ever fired, the
        request would otherwise stay 'pending' forever, blocking a reviewer's
        dashboard with a decision nothing is ever going to resolve. Meant to
        be run as part of startup/crash-recovery cleanup. Returns the count
        of requests timed out.
        """
        cutoff = datetime.now(timezone.utc).timestamp() - timeout_seconds
        count = 0
        for request in self.list_pending():
            created_at = datetime.fromisoformat(request["created_at"]).timestamp()
            if created_at <= cutoff:
                if self.decide(request["id"], "timed_out", note="Timed out during crash-recovery cleanup."):
                    count += 1
        return count

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        d["metrics"] = json.loads(d["metrics"])
        return d
