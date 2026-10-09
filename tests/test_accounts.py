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
    app.accounts, app.launched, app.home = accounts, launched, tmp_path
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


def used(app, name, cloud_usd=0.0, tokens=0):
    """The job used this much Nebius GPU time and Token Factory tokens; the meter charges it."""
    from sim2chip.workspace import Job
    job = Job(app.home / "jobs" / name)
    job.logs.mkdir(parents=True, exist_ok=True)
    (job.logs / "cloud_usage.jsonl").write_text(json.dumps({"hours": 0.2, "usd_estimate": cloud_usd}) + "\n")
    (job.logs / "llm_usage.jsonl").write_text(json.dumps({"model": "m", "purpose": "label", "prompt_tokens": tokens,
                                                          "completion_tokens": 0}) + "\n")
    (app.home / "cache").mkdir(exist_ok=True)
    (app.home / "cache" / "token_prices.json").write_text(json.dumps({"t": time.time(), "prices": {
        "m": {"input": 1e-6, "output": 2e-6}}}))
    app.meter_job(job, running=True)
    return job


def test_normal_account_pays_for_what_it_uses(server):
    c = sign_in(server, "google", "alice")
    me = c.get("/api/me").json()
    assert me["account"]["tier"] == "normal" and me["account"]["balance"] == 0
    assert c.post("/api/interview").status_code == 402
    assert new_job(c).status_code == 402
    assert server.launched == []
    acc = server.accounts.by_email("alice@gmail.com")[0]
    server.accounts.add(acc.id, 1.1, "grant")
    assert c.post("/api/interview").status_code == 200
    name = new_job(c).json()["job"]
    assert len(server.launched) == 1
    assert c.get("/api/me").json()["account"]["balance"] == 1.1  # nothing charged before it uses anything
    used(server, name, cloud_usd=0.2, tokens=50_000)  # 0.25 $ of cost, twice that charged
    assert c.get("/api/me").json()["account"]["balance"] == 0.6
    used(server, name, cloud_usd=0.2, tokens=50_000)  # the same usage read again is not charged twice
    assert c.get("/api/me").json()["account"]["balance"] == 0.6
    moves = c.get("/api/billing").json()["moves"]
    assert moves[0] == {**moves[0], "amount": -0.5, "reason": "run", "detail": f"job:{name}"}
    again = new_job(c)  # the same setup twice (a double click) is the running job
    assert again.status_code == 200 and again.json().get("already_running") and len(server.launched) == 1
    assert new_job(c, "black cows").status_code == 402  # 0.6 $ does not cover a usual run


def test_a_job_stops_when_the_balance_runs_out(server):
    c = sign_in(server, "google", "alice")
    acc = server.accounts.by_email("alice@gmail.com")[0]
    server.accounts.add(acc.id, 1.0, "grant")
    name = new_job(c).json()["job"]
    job = used(server, name, cloud_usd=0.6)
    assert server.accounts.balance(acc.id) == -0.2
    assert "the balance ran out" in (job.logs / "pipeline.log").read_text()


def test_verified_judge_domain_is_free_unverified_is_not(server):
    judge = sign_in(server, "google", "judge")
    assert judge.get("/api/me").json()["account"]["tier"] == "judge"
    for i in range(3):
        assert new_job(judge, f"cam {i}").status_code == 200
    assert len(server.launched) == 3
    name = judge.get("/api/jobs").json()["jobs"][0]
    used(server, name, cloud_usd=0.5)
    assert judge.get("/api/me").json()["account"]["balance"] == 0  # nothing charged
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


def test_a_run_that_cannot_start_costs_nothing(tmp_path):
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


def test_login_returns_only_to_a_page_of_this_site(server):
    for bad in ("//evil.example", "/\\evil.example", "https://evil.example", "/\\\\evil.example/x"):
        c = TestClient(server, base_url="https://eyeforge.test")
        r = c.get("/auth/google/login", params={"next": bad}, follow_redirects=False)
        state = parse_qs(urlparse(r.headers["location"]).query)["state"][0]
        r = c.get(f"/auth/google/callback?code=alice&state={state}", follow_redirects=False)
        assert r.headers["location"] == "/app", bad


def test_a_chat_and_its_photos_are_only_its_owners(server):
    alice, bob = sign_in(server, "google", "judge"), sign_in(server, "github", "gh")
    sid = alice.post("/api/interview").json()["session"]
    files = [("photos", ("a.jpg", b"x", "image/jpeg"))]
    assert alice.post(f"/api/interview/{sid}/photos", files=files, data={"more": "true"}).status_code == 200
    assert bob.post(f"/api/interview/{sid}", json={"text": "hi"}).status_code == 404
    assert bob.post(f"/api/interview/{sid}/photos", files=files).status_code == 404
    spec = {**GOOD, "user_data": {"raw": str(server.home / "uploads" / sid / "training")}}
    r = bob.post("/api/jobs", data={"spec": json.dumps(spec)})
    assert r.status_code == 400 and "uploaded here" in r.json()["detail"]
    assert alice.post("/api/jobs", data={"spec": json.dumps(spec)}).status_code == 200


def test_one_dollar_does_not_start_two_runs(server):
    c = sign_in(server, "google", "alice")
    server.accounts.add(server.accounts.by_email("alice@gmail.com")[0].id, 1.5, "grant")
    assert new_job(c, "cows").status_code == 200
    assert new_job(c, "black cows").status_code == 402  # the first one still runs, not charged yet


def test_huge_uploads_and_framing_are_refused(server, monkeypatch):
    from sim2chip import server as server_mod
    monkeypatch.setattr(server_mod, "MAX_PHOTO_BYTES", 10)
    c = sign_in(server, "google", "judge")
    sid = c.post("/api/interview").json()["session"]
    r = c.post(f"/api/interview/{sid}/photos", files=[("photos", ("a.jpg", b"x" * 11, "image/jpeg"))])
    assert r.status_code == 413
    h = c.get("/api/me").headers
    assert h["x-frame-options"] == "DENY" and h["x-content-type-options"] == "nosniff"


def test_deleting_an_account_removes_its_data_and_identity(server, monkeypatch):
    c = sign_in(server, "google", "alice")
    acc = server.accounts.by_email("alice@gmail.com")[0]
    server.accounts.add(acc.id, 5, "grant")
    name = new_job(c).json()["job"]
    other = sign_in(server, "google", "judge")
    kept = new_job(other).json()["job"]
    assert c.post("/api/account/delete", json={"confirm": "yes"}).status_code == 400
    assert c.post("/api/account/delete", json={"confirm": "delete"}).status_code == 409  # still training
    monkeypatch.setattr(FakeProc, "poll", lambda self: 0)  # the runs end
    r = c.post("/api/account/delete", json={"confirm": "delete"})
    assert r.status_code == 200 and r.json()["refunded"] == 0
    assert not (server.home / "jobs" / name).exists() and (server.home / "jobs" / kept).exists()
    assert server.accounts.by_email("alice@gmail.com") == []
    assert c.get("/api/me").json()["signed_in"] is False
    again = sign_in(server, "google", "alice")  # the same login makes a new, empty account
    assert again.get("/api/me").json()["account"]["balance"] == 0
    assert name not in again.get("/api/jobs").json()["jobs"]
