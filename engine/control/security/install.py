"""Register auth routes and middleware on Command Center FastAPI app."""

from __future__ import annotations

from typing import Any

from scout_auto_os.engine.control.security.auth import AuthManager
from scout_auto_os.engine.control.security.config import SESSION_COOKIE_NAME
from scout_auto_os.engine.control.security.login_page import load_login_template

try:
    from fastapi import HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse, Response
    from pydantic import BaseModel
except ImportError:
    Request = object  # type: ignore[misc, assignment]
    HTMLResponse = JSONResponse = Response = object  # type: ignore[misc, assignment]
    HTTPException = Exception  # type: ignore[misc, assignment]
    BaseModel = object  # type: ignore[misc, assignment]


class LoginRequest(BaseModel):
    password: str


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def register_security(app: Any, auth: AuthManager) -> None:
    login_html = load_login_template()

    @app.post("/auth/login")
    async def post_login(body: LoginRequest, request: Request, response: Response) -> dict:
        ip = _client_ip(request)
        ok, reason, sid = auth.login(body.password, ip)
        if not ok:
            if reason == "locked":
                raise HTTPException(status_code=429, detail="login_locked_15m")
            raise HTTPException(status_code=401, detail="invalid_password")
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=sid,
            httponly=True,
            samesite="lax",
            secure=auth.cookie_secure,
            max_age=auth.sessions.ttl_seconds,
            path="/",
        )
        return {"ok": True}

    @app.post("/auth/logout")
    async def post_logout(request: Request, response: Response) -> dict:
        ip = _client_ip(request)
        sid = request.cookies.get(SESSION_COOKIE_NAME)
        auth.logout(sid, ip)
        response.delete_cookie(SESSION_COOKIE_NAME, path="/")
        return {"ok": True}

    @app.middleware("http")
    async def security_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        path = request.url.path
        ip = _client_ip(request)

        if path == "/auth/login" and request.method == "POST":
            return await call_next(request)

        if path == "/command-center" and request.method == "GET":
            sid = request.cookies.get(SESSION_COOKIE_NAME)
            ok, _ = auth.validate_session(sid, ip)
            if not ok:
                return HTMLResponse(content=login_html, status_code=200)
            return await call_next(request)

        if path.startswith("/control/") or path == "/auth/logout":
            sid = request.cookies.get(SESSION_COOKIE_NAME)
            ok, reason = auth.validate_session(sid, ip)
            if not ok:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "unauthorized", "reason": reason},
                )
            return await call_next(request)

        if path in ("/docs", "/redoc", "/openapi.json"):
            return JSONResponse(status_code=401, content={"detail": "unauthorized"})

        return await call_next(request)
