import os
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone

from approval.store import ApprovalStore


def test_create_and_get_request():
    with tempfile.TemporaryDirectory() as d:
        store = ApprovalStore(os.path.join(d, "approvals.db"))
        request_id = store.create_request("cand-1", "goal", "diff text", 0.9, {"k": 1})

        req = store.get_request(request_id)
        assert req["status"] == "pending"
        assert req["candidate_id"] == "cand-1"
        assert req["final_score"] == 0.9
        assert req["metrics"] == {"k": 1}


def test_decisions_are_persisted_across_a_fresh_store_instance():
    """Regression test: decisions must survive a process restart, not live only in memory."""
    with tempfile.TemporaryDirectory() as d:
        db_path = os.path.join(d, "approvals.db")
        store1 = ApprovalStore(db_path)
        request_id = store1.create_request("cand-1", "goal", "diff", 0.9, {})
        store1.decide(request_id, "approved", note="fine")

        store2 = ApprovalStore(db_path)  # simulates a separate/restarted process
        req = store2.get_request(request_id)
        assert req["status"] == "approved"
        assert req["decision_note"] == "fine"


def test_list_pending_excludes_decided_requests():
    with tempfile.TemporaryDirectory() as d:
        store = ApprovalStore(os.path.join(d, "approvals.db"))
        pending_id = store.create_request("cand-1", "goal", "diff", 0.9, {})
        decided_id = store.create_request("cand-2", "goal", "diff", 0.5, {})
        store.decide(decided_id, "rejected")

        pending = store.list_pending()
        assert len(pending) == 1
        assert pending[0]["id"] == pending_id


def test_decide_is_a_noop_once_already_decided():
    """A late human click must never override an automatic timeout hold, or vice versa."""
    with tempfile.TemporaryDirectory() as d:
        store = ApprovalStore(os.path.join(d, "approvals.db"))
        request_id = store.create_request("cand-1", "goal", "diff", 0.9, {})

        first = store.decide(request_id, "timed_out")
        second = store.decide(request_id, "approved")  # arrives too late

        assert first is True
        assert second is False
        assert store.get_request(request_id)["status"] == "timed_out"


def test_timeout_stale_requests_times_out_only_requests_older_than_the_window():
    """
    Wave 3 acceptance: crash recovery must reclaim a request left 'pending'
    forever because the process awaiting it crashed before its own deadline
    check ever fired - but must never touch a request still within its
    timeout window.
    """
    with tempfile.TemporaryDirectory() as d:
        store = ApprovalStore(os.path.join(d, "approvals.db"))
        stale_id = store.create_request("cand-stale", "goal", "diff", 0.9, {})
        fresh_id = store.create_request("cand-fresh", "goal", "diff", 0.9, {})

        # Backdate the stale request's created_at directly - simulates a
        # request that has genuinely been sitting pending past the timeout.
        # sqlite3's `with conn:` only commits/rolls back - it does not close
        # the connection, which would leave a file handle open on Windows
        # and block the TemporaryDirectory cleanup below. Close explicitly.
        old_created_at = (datetime.now(timezone.utc) - timedelta(seconds=10000)).isoformat()
        conn = sqlite3.connect(store.db_path)
        try:
            conn.execute(
                "UPDATE approval_requests SET created_at = ? WHERE id = ?", (old_created_at, stale_id)
            )
            conn.commit()
        finally:
            conn.close()

        timed_out_count = store.timeout_stale_requests(timeout_seconds=1800)

        assert timed_out_count == 1
        assert store.get_request(stale_id)["status"] == "timed_out"
        assert store.get_request(fresh_id)["status"] == "pending"


def test_timeout_stale_requests_never_overrides_an_already_decided_request():
    with tempfile.TemporaryDirectory() as d:
        store = ApprovalStore(os.path.join(d, "approvals.db"))
        request_id = store.create_request("cand-1", "goal", "diff", 0.9, {})
        store.decide(request_id, "approved")

        old_created_at = (datetime.now(timezone.utc) - timedelta(seconds=10000)).isoformat()
        conn = sqlite3.connect(store.db_path)
        try:
            conn.execute(
                "UPDATE approval_requests SET created_at = ? WHERE id = ?", (old_created_at, request_id)
            )
            conn.commit()
        finally:
            conn.close()

        timed_out_count = store.timeout_stale_requests(timeout_seconds=1800)

        assert timed_out_count == 0
        assert store.get_request(request_id)["status"] == "approved"


def test_decide_rejects_invalid_status():
    with tempfile.TemporaryDirectory() as d:
        store = ApprovalStore(os.path.join(d, "approvals.db"))
        request_id = store.create_request("cand-1", "goal", "diff", 0.9, {})
        try:
            store.decide(request_id, "maybe")
            assert False, "expected ValueError"
        except ValueError:
            pass
