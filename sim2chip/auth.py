"""Login with Google or GitHub for a public server (OAuth 2 authorization code flow), stdlib only.

Configuration (environment or .env):
    SIM2CHIP_PUBLIC_URL        the server's address as browsers see it, e.g. https://eyeforge.app (redirects)
    SIM2CHIP_SESSION_SECRET    signs the session cookie (any long random string; changing it logs everyone out)
    GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET    a Google Cloud "Web application" OAuth client
    GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET    a GitHub OAuth App
A provider without its id and secret is not offered.

What a login gives: the provider's stable user id, an email, and the emails the provider says are verified
(the judge tier of accounts.py trusts only those).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

from . import config

PROVIDERS = {
    "google": {
        "authorize": "https://accounts.google.com/o/oauth2/v2/auth",
        "token": "https://oauth2.googleapis.com/token",
        "scope": "openid email profile",
    },
    "github": {
        "authorize": "https://github.com/login/oauth/authorize",
        "token": "https://github.com/login/oauth/access_token",
        "scope": "read:user user:email",
    },
}


class AuthError(Exception):
    pass


@dataclass
class Identity:
    provider: str
    subject: str
    email: str | None
    verified: list[str] = field(default_factory=list)
    name: str | None = None


def _http(url: str, data: dict | None = None, token: str | None = None) -> dict | list:
    headers = {"Accept": "application/json", "User-Agent": "EyeForge"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = urllib.parse.urlencode(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers, method="POST" if body else "GET")
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())
    except Exception as e:  # network, HTTP error or a body that is not JSON: the login fails, nothing else
        raise AuthError(f"{url.split('/')[2]}: {e}") from None


class Auth:
    def __init__(self, public_url: str, secret: str, clients: dict[str, tuple[str, str]], http=_http):
        """clients: provider -> (client id, client secret). http: (url, form data, bearer token) -> JSON."""
        if len(secret) < 16:
            raise ValueError("SIM2CHIP_SESSION_SECRET is too short (16 characters at least)")
        self.public_url = public_url.rstrip("/")
        self.secret = secret.encode()
        self.clients = clients
        self.http = http

    @classmethod
    def from_env(cls) -> "Auth | None":
        """The login of a public server, or None when it is not configured (no provider or no secret)."""
        clients = {p: (cid, sec) for p in PROVIDERS
                   if (cid := config.env(f"{p.upper()}_CLIENT_ID")) and (sec := config.env(f"{p.upper()}_CLIENT_SECRET"))}
        secret, url = config.env("SIM2CHIP_SESSION_SECRET"), config.env("SIM2CHIP_PUBLIC_URL")
        if not clients or not secret or not url:
            return None
        return cls(url, secret, clients)

    @property
    def providers(self) -> list[str]:
        return [p for p in PROVIDERS if p in self.clients]

    def redirect_uri(self, provider: str) -> str:
        return f"{self.public_url}/auth/{provider}/callback"

    # Signed values: "<base64 payload>.<base64 hmac>"; the payload carries its expiry.
    def sign(self, value: str, max_age: float) -> str:
        payload = base64.urlsafe_b64encode(json.dumps({"v": value, "exp": time.time() + max_age}).encode()).decode()
        mac = base64.urlsafe_b64encode(hmac.new(self.secret, payload.encode(), hashlib.sha256).digest()).decode()
        return f"{payload}.{mac}"

    def unsign(self, token: str | None) -> str | None:
        if not token or "." not in token:
            return None
        payload, mac = token.rsplit(".", 1)
        good = base64.urlsafe_b64encode(hmac.new(self.secret, payload.encode(), hashlib.sha256).digest()).decode()
        if not hmac.compare_digest(mac, good):
            return None
        try:
            data = json.loads(base64.urlsafe_b64decode(payload.encode()))
        except ValueError:
            return None
        return data["v"] if data.get("exp", 0) > time.time() else None

    def login_url(self, provider: str, state: str) -> str:
        p = PROVIDERS[provider]
        q = {"client_id": self.clients[provider][0], "redirect_uri": self.redirect_uri(provider),
             "response_type": "code", "scope": p["scope"], "state": state}
        if provider == "google":
            q["prompt"] = "select_account"
        return f"{p['authorize']}?{urllib.parse.urlencode(q)}"

    def identity(self, provider: str, code: str) -> Identity:
        """The user behind an authorization code."""
        cid, sec = self.clients[provider]
        tok = self.http(PROVIDERS[provider]["token"], {
            "client_id": cid, "client_secret": sec, "code": code, "redirect_uri": self.redirect_uri(provider),
            "grant_type": "authorization_code"}, None)
        access = tok.get("access_token") if isinstance(tok, dict) else None
        if not access:
            raise AuthError(f"{provider} gave no access token: {str(tok)[:200]}")
        if provider == "google":
            u = self.http("https://openidconnect.googleapis.com/v1/userinfo", None, access)
            email = u.get("email")
            return Identity("google", str(u["sub"]), email, [email] if email and u.get("email_verified") else [],
                            u.get("name"))
        u = self.http("https://api.github.com/user", None, access)
        emails = self.http("https://api.github.com/user/emails", None, access)
        emails = emails if isinstance(emails, list) else []
        verified = [e["email"] for e in emails if e.get("verified") and e.get("email")]
        primary = next((e["email"] for e in emails if e.get("primary")), None) or u.get("email")
        return Identity("github", str(u["id"]), primary, verified, u.get("name") or u.get("login"))
