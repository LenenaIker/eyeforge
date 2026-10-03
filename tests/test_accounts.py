"""Login with Google or GitHub, judge access and credits (docs/plan_login.md), with a fake OAuth provider."""
import json
import time
from urllib.parse import parse_qs, urlparse

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from sim2chip import interview  # noqa: E402
from sim2chip.accounts import Accounts  # noqa: E402
from sim2chip.auth import Auth  # noqa: E402
from sim2chip.server import create_app  # noqa: E402

from test_server import GOOD, FakeProc, Scripted  # noqa: E402

USERS = {  # access token -> what the provider says about the user
    "tok-alice": {"google": {"sub": "1", "email": "alice@gmail.com", "email_verified": True, "name": "Alice"}},
    "tok-judge": {"google": {"sub": "2", "email": "judge@nvidia.com", "email_verified": True}},
    "tok-fake": {"google": {"sub": "3", "email": "someone@nebius.com", "email_verified": False}},
    "tok-gh": {"github": ({"id": 7, "login": "octo"},
                          [{"email": "octo@users.noreply.github.com", "primary": True, "verified": True},
                           {"email": "octo@tavily.com", "primary": False, "verified": True}])},
}


def fake_http(url, data=None, token=None):
    if data is not None:  # token endpoint: the code is the access token's name
        return {"access_token": f"tok-{data['code']}"}
    who = USERS[token]
    if "googleapis" in url:
        return who["google"]
    user, emails = who["github"]
    return emails if url.endswith("/emails") else user


@pytest.fixture
def server(tmp_path):
    auth = Auth("https://eyeforge.test", "x" * 32, {"google": ("gid", "gsec"), "github": ("hid", "hsec")},
                http=fake_http)
    accounts = Accounts(tmp_path / "accounts.db")
    launched = []

    def interviewer():
        cats = interview.CategoryIndex(counts={"cow": 62})
        return interview.Session(Scripted([{"say": "Where is the camera?"}]), "m", cats)

    def launcher(home, job, request, backend, max_usd, start=None):
        launched.append(job.name)
        (job.logs / "pipeline.log").write_text("job x\n")
        return FakeProc()

    app = create_app(tmp_path, backend="nebius", interviewer=interviewer, launcher=launcher, public=True,
                     max_runs=5, auth=auth, accounts=accounts, contact="judges@eyeforge.test")
    app.accounts, app.launched = accounts, launched
    return app


def sign_in(app, provider: str, code: str) -> TestClient:
    """A browser that went through the provider's login and came back with `code`."""
    c = TestClient(app, base_url="https://eyeforge.test")
    r = c.get(f"/auth/{provider}/login?next=/app/job/x", follow_redirects=False)
    assert r.status_code == 302
    state = parse_qs(urlparse(r.headers["location"]).query)["state"][0]
    r = c.get(f"/auth/{provider}/callback?code={code}&state={state}", follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"] == "/app/job/x"
    return c


def new_job(c, what="cows"):
    """A run; a different `what` is a different setup (the same one twice is the running job, see create_job)."""
    spec = {**GOOD, "classes": [{**GOOD["classes"][0], "description": what}]}
    files = [("photos", ("a.jpg", b"x", "image/jpeg"))]
    return c.post("/api/jobs", data={"spec": json.dumps(spec)}, files=files)


def test_landing_open_api_needs_login(server):
    c = TestClient(server, base_url="https://eyeforge.test")
    assert c.get("/").status_code == 200
    cfg = c.get("/api/config").json()
    assert cfg["login"] == ["google", "github"] and cfg["locked"] is False
    assert c.get("/api/me").json()["signed_in"] is False
    assert c.get("/api/jobs").status_code == 401
    assert c.post("/api/interview").status_code == 401


def test_callback_needs_the_state_of_its_own_login(server):
    c = TestClient(server, base_url="https://eyeforge.test")
    c.get("/auth/google/login", follow_redirects=False)
    r = c.get("/auth/google/callback?code=alice&state=forged", follow_redirects=False)
    assert r.headers["location"] == "/app?login=failed"
    assert c.get("/api/me").json()["signed_in"] is False


def test_normal_account_pays_for_everything(server):
    c = sign_in(server, "google", "alice")
    me = c.get("/api/me").json()
    assert me["account"]["tier"] == "normal" and me["account"]["credits"] == 0
    assert c.post("/api/interview").status_code == 402
    assert new_job(c).status_code == 402
    assert server.launched == []
    acc = server.accounts.by_email("alice@gmail.com")[0]
    server.accounts.add(acc.id, 1.1, "grant")
    assert c.post("/api/interview").status_code == 200
    assert new_job(c).status_code == 200 and len(server.launched) == 1
    assert c.get("/api/me").json()["account"]["credits"] == 0
    again = new_job(c)  # the same setup twice (a double click) is the running job, not a second charge
    assert again.status_code == 200 and again.json().get("already_running") and len(server.launched) == 1
    assert new_job(c, "black cows").status_code == 402


def test_verified_judge_domain_is_free_unverified_is_not(server):
    judge = sign_in(server, "google", "judge")
    assert judge.get("/api/me").json()["account"]["tier"] == "judge"
    for i in range(3):
        assert new_job(judge, f"cam {i}").status_code == 200
    assert len(server.launched) == 3
    assert judge.get("/api/me").json()["account"]["credits"] == 0  # nothing charged
    fake = sign_in(server, "google", "fake")  # a nebius.com address Google did not verify
    assert fake.get("/api/me").json()["account"]["tier"] == "normal"
    gh = sign_in(server, "github", "gh")  # a verified secondary email of a judge domain counts
    assert gh.get("/api/me").json()["account"]["tier"] == "judge"


def test_judge_tier_ends_with_the_judging(tmp_path):
    accs = Accounts(tmp_path / "a.db", judge_until=time.time() - 1)
    acc = accs.login("google", "2", "judge@nvidia.com", ["judge@nvidia.com"])
    assert acc.judge == "domain" and accs.tier(acc) == "normal"


def test_manual_judge_and_ban(server):
    c = sign_in(server, "google", "alice")
    server.accounts.set_flag("alice@gmail.com", "judge", "manual")
    assert new_job(c).status_code == 200
    server.accounts.set_flag("alice@gmail.com", "banned", 1)
    r = c.get("/api/jobs")
    assert r.status_code == 403 and "judges@eyeforge.test" in r.json()["detail"]


def test_jobs_follow_the_account_to_another_device(server):
    pc = sign_in(server, "google", "judge")
    name = new_job(pc).json()["job"]
    phone = sign_in(server, "google", "judge")  # same account, another browser
    assert name in phone.get("/api/jobs").json()["jobs"]
    other = sign_in(server, "google", "alice")
    assert name not in other.get("/api/jobs").json()["jobs"]
    assert other.get(f"/api/jobs/{name}").status_code == 404


def test_a_run_that_cannot_start_is_refunded(tmp_path):
    auth = Auth("https://eyeforge.test", "x" * 32, {"google": ("gid", "gsec")}, http=fake_http)
    accounts = Accounts(tmp_path / "accounts.db")

    def broken(*a, **k):
        raise RuntimeError("no GPU")

    app = create_app(tmp_path, backend="nebius", launcher=broken, public=True, auth=auth, accounts=accounts)
    c = sign_in(app, "google", "alice")
    accounts.add(accounts.by_email("alice@gmail.com")[0].id, 1, "grant")
    with pytest.raises(RuntimeError):
        new_job(c)
    assert accounts.balance(accounts.by_email("alice@gmail.com")[0].id) == 1


def test_signed_values_expire_and_resist_tampering():
    a = Auth("https://x", "s" * 32, {"google": ("i", "s")})
    tok = a.sign("google:1", 60)
    assert a.unsign(tok) == "google:1"
    assert a.unsign(tok[:-2] + "AA") is None
    assert a.unsign(a.sign("google:1", -1)) is None
    assert Auth("https://x", "t" * 32, {}).unsign(tok) is None
