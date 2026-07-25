import os
import secrets
from typing import Optional

from fastapi import Depends, FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from approval.store import ApprovalStore
from memory.db import ExperimentDB
from reporting.report_generator import compute_kpis

# Reads from the SAME files the orchestrator process writes to (when both
# are run from the same working directory, which is the intended setup) -
# not a forked/duplicated data store. Overridable via env vars in case the
# API is run from a different directory than the orchestrator.
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
APPROVAL_DB_PATH = os.environ.get("APPROVAL_DB_PATH", "approvals.db")
EVOLUTION_REPORT_PATH = os.environ.get("EVOLUTION_REPORT_PATH", "evolution_report.jsonl")

app = FastAPI(title="Autoresearch Dashboard")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

db = ExperimentDB(db_path=CHROMA_DB_PATH)
store = ApprovalStore(db_path=APPROVAL_DB_PATH)

_basic_auth = HTTPBasic(auto_error=False)


def require_auth(credentials: Optional[HTTPBasicCredentials] = Depends(_basic_auth)) -> None:
    """
    Fail-safe like the approval gate (approval/gate.py:resolve_approval_config):
    auth is required unless DASHBOARD_AUTH_DISABLED is explicitly set to
    "true". If it isn't disabled but no DASHBOARD_USERNAME/DASHBOARD_PASSWORD
    are configured either, every request is rejected rather than silently
    served - there is no way to authenticate, so nobody gets in.
    """
    if os.environ.get("DASHBOARD_AUTH_DISABLED") == "true":
        return

    expected_username = os.environ.get("DASHBOARD_USERNAME")
    expected_password = os.environ.get("DASHBOARD_PASSWORD")

    unauthorized = HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Basic"})

    if not expected_username or not expected_password:
        raise unauthorized

    if credentials is None:
        raise unauthorized

    # secrets.compare_digest for both fields - a naive == leaks timing
    # information proportional to how many leading characters match.
    valid_username = secrets.compare_digest(credentials.username, expected_username)
    valid_password = secrets.compare_digest(credentials.password, expected_password)
    if not (valid_username and valid_password):
        raise unauthorized


def _verify_csrf(request: Request, csrf_token: str) -> None:
    cookie_token = request.cookies.get("csrf_token")
    if not cookie_token or not secrets.compare_digest(csrf_token, cookie_token):
        raise HTTPException(status_code=403, detail="Invalid or missing CSRF token")


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, _auth: None = Depends(require_auth)):
    kpis = compute_kpis(db, store, evolution_report_path=EVOLUTION_REPORT_PATH)
    pending = store.list_pending()
    history = list(reversed(db.list_all_experiments(limit=50)))
    decided = [a for a in store.list_all(limit=50) if a["status"] != "pending"]

    # Double-submit-cookie CSRF token: reuse an existing one so repeat visits
    # (e.g. the page's own 10s auto-refresh) don't invalidate an in-flight
    # form submission from a previous render.
    csrf_token = request.cookies.get("csrf_token") or secrets.token_urlsafe(32)

    response = templates.TemplateResponse(
        request,
        "dashboard.html",
        {"kpis": kpis, "pending": pending, "history": history, "decided": decided, "csrf_token": csrf_token},
    )
    if not request.cookies.get("csrf_token"):
        response.set_cookie("csrf_token", csrf_token, httponly=False, samesite="strict")
    return response


@app.post("/approvals/{request_id}/approve")
def approve(request: Request, request_id: str, csrf_token: str = Form(""), note: Optional[str] = Form(None), _auth: None = Depends(require_auth)):
    _verify_csrf(request, csrf_token)
    store.decide(request_id, "approved", note=note)
    return RedirectResponse(url="/", status_code=303)


@app.post("/approvals/{request_id}/reject")
def reject(request: Request, request_id: str, csrf_token: str = Form(""), note: Optional[str] = Form(None), _auth: None = Depends(require_auth)):
    _verify_csrf(request, csrf_token)
    store.decide(request_id, "rejected", note=note)
    return RedirectResponse(url="/", status_code=303)


@app.get("/api/pending")
def api_pending(_auth: None = Depends(require_auth)):
    return store.list_pending()


@app.get("/api/approvals")
def api_approvals(limit: int = 100, _auth: None = Depends(require_auth)):
    return store.list_all(limit=limit)


@app.get("/api/history")
def api_history(limit: int = 100, _auth: None = Depends(require_auth)):
    return list(reversed(db.list_all_experiments(limit=limit)))


@app.get("/api/report")
def api_report(_auth: None = Depends(require_auth)):
    return compute_kpis(db, store, evolution_report_path=EVOLUTION_REPORT_PATH)
