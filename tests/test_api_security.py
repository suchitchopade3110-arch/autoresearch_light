import base64
import os
import sys

import pytest


def _fresh_api_main():
    for mod in ("api.main",):
        if mod in sys.modules:
            del sys.modules[mod]
    import api.main as api_main
    return api_main


def _basic_auth_header(username, password):
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


@pytest.fixture
def configure_paths(tmp_dir):
    os.environ["CHROMA_DB_PATH"] = os.path.join(tmp_dir, "chroma")
    os.environ["APPROVAL_DB_PATH"] = os.path.join(tmp_dir, "approvals.db")
    os.environ["EVOLUTION_REPORT_PATH"] = os.path.join(tmp_dir, "evolution_report.jsonl")
    yield
    for key in ("DASHBOARD_USERNAME", "DASHBOARD_PASSWORD", "DASHBOARD_AUTH_DISABLED"):
        os.environ.pop(key, None)


def test_dashboard_requires_auth_by_default_when_no_credentials_configured(configure_paths):
    """
    Wave 4 acceptance: fail-safe like the approval gate - if nobody has
    configured DASHBOARD_USERNAME/PASSWORD and auth hasn't been explicitly
    disabled, every request must be rejected rather than silently served.
    """
    for key in ("DASHBOARD_USERNAME", "DASHBOARD_PASSWORD", "DASHBOARD_AUTH_DISABLED"):
        os.environ.pop(key, None)
    api_main = _fresh_api_main()
    from fastapi.testclient import TestClient

    client = TestClient(api_main.app)
    r = client.get("/")
    assert r.status_code == 401


def test_dashboard_rejects_wrong_credentials(configure_paths):
    os.environ["DASHBOARD_USERNAME"] = "admin"
    os.environ["DASHBOARD_PASSWORD"] = "correct-horse"
    api_main = _fresh_api_main()
    from fastapi.testclient import TestClient

    client = TestClient(api_main.app)
    r = client.get("/", headers=_basic_auth_header("admin", "wrong-password"))
    assert r.status_code == 401


def test_dashboard_accepts_correct_credentials(configure_paths):
    os.environ["DASHBOARD_USERNAME"] = "admin"
    os.environ["DASHBOARD_PASSWORD"] = "correct-horse"
    api_main = _fresh_api_main()
    from fastapi.testclient import TestClient

    client = TestClient(api_main.app)
    r = client.get("/", headers=_basic_auth_header("admin", "correct-horse"))
    assert r.status_code == 200


def test_dashboard_auth_can_be_explicitly_disabled_for_local_use(configure_paths):
    os.environ["DASHBOARD_AUTH_DISABLED"] = "true"
    api_main = _fresh_api_main()
    from fastapi.testclient import TestClient

    client = TestClient(api_main.app)
    r = client.get("/")
    assert r.status_code == 200


def test_api_endpoints_also_require_auth(configure_paths):
    api_main = _fresh_api_main()
    from fastapi.testclient import TestClient

    client = TestClient(api_main.app)
    assert client.get("/api/pending").status_code == 401
    assert client.get("/api/history").status_code == 401
    assert client.get("/api/report").status_code == 401


def test_approve_without_csrf_token_is_rejected(configure_paths):
    """
    Wave 4 acceptance: the mutating approve/reject endpoints must reject a
    request that doesn't carry a matching CSRF token, even with valid auth -
    otherwise a cross-site auto-submitting form could trigger an approval
    using the operator's cached credentials.
    """
    os.environ["DASHBOARD_AUTH_DISABLED"] = "true"
    api_main = _fresh_api_main()
    from fastapi.testclient import TestClient

    client = TestClient(api_main.app)
    req_id = api_main.store.create_request("cand-1", "goal", "diff", 0.9, {})

    r = client.post(f"/approvals/{req_id}/approve", data={"note": "ship it"}, follow_redirects=False)
    assert r.status_code == 403
    assert api_main.store.get_request(req_id)["status"] == "pending"


def test_approve_with_mismatched_csrf_token_is_rejected(configure_paths):
    os.environ["DASHBOARD_AUTH_DISABLED"] = "true"
    api_main = _fresh_api_main()
    from fastapi.testclient import TestClient

    client = TestClient(api_main.app)
    req_id = api_main.store.create_request("cand-1", "goal", "diff", 0.9, {})

    dashboard = client.get("/")
    real_token = dashboard.cookies.get("csrf_token")
    assert real_token

    r = client.post(
        f"/approvals/{req_id}/approve",
        data={"note": "ship it", "csrf_token": "not-the-real-token"},
        follow_redirects=False,
    )
    assert r.status_code == 403
    assert api_main.store.get_request(req_id)["status"] == "pending"


def test_approve_with_matching_csrf_token_succeeds(configure_paths):
    os.environ["DASHBOARD_AUTH_DISABLED"] = "true"
    api_main = _fresh_api_main()
    from fastapi.testclient import TestClient

    client = TestClient(api_main.app)
    req_id = api_main.store.create_request("cand-1", "goal", "diff", 0.9, {})

    dashboard = client.get("/")
    real_token = dashboard.cookies.get("csrf_token")
    assert real_token
    assert f'value="{real_token}"' in dashboard.text

    r = client.post(
        f"/approvals/{req_id}/approve",
        data={"note": "ship it", "csrf_token": real_token},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert api_main.store.get_request(req_id)["status"] == "approved"
