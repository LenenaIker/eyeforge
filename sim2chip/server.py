"""Web API and the web app: interview, photo upload, run, progress, review and download.

    sim2chip serve [--backend local|nebius] [--host 127.0.0.1] [--port 8000]

The app is a React build in web/dist (from the repo's web/ folder: `cd web && npm install && npm run build`;
the Docker image builds it). Any path outside /api serves the app, which routes itself.

Each run is a `sim2chip run` subprocess writing to the job's logs/pipeline.log, so the server never
blocks on the pipeline and the command line stays the single way to run a job. Progress is read back
from the job folder (done markers, the pipeline log and the cloud usage log).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from . import config, review, usage, warm as warm_mod
from .accounts import INTERVIEW_COST, RUN_COST
from . import spec as spec_mod
from .pipeline import Pipeline
from .workspace import IMAGE_EXT, Cache, Job, current_version, open_job, snapshot, versions

DOWNLOADS = {"model.tflite", "labels.txt", "model_int8.tflite", "model_data.h", "report.json"}
WEB = Path(__file__).parent / "web"
DIST = WEB / "dist"
OWNER = "owner"  # file in a public server's job folder: the visitor who started it
NO_BUILD = ("<p>EyeForge: the web app is not built. Run <code>cd web && npm install && npm run build</code> in the "
            "repository, then reload. The API under /api works without it.</p>")
OPEN_API = {"/api/config", "/api/key", "/api/me"}  # API calls open to a visitor without the key or a login
SESSION = "eyeforge_session"  # cookie: the signed account id (login mode)
STATE = "eyeforge_oauth"      # cookie: the signed nonce and return page of a login in progress
SESSION_DAYS = 30
ACCOUNTS_DB = "accounts.db"  # in the server's home
VISITOR = "eyeforge_visitor"  # cookie: one per browser, so visitors of a public server only see their own jobs
# 409 answers when the same job is open twice (phone and computer): the web shows them as news, not errors.
ALREADY_RUNNING = "This model is already being worked on, maybe from another device."
CHECK_OVER = "Training already started, maybe from another device."


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")
    return s[:48] or "job"


def default_interviewer(home: Path, backend: str = "local", public: bool = False):
    from . import interview, research
    from .llm import Client

    from dataclasses import replace

    base = Client(usage_log=home / "logs" / "llm_usage.jsonl")
    model, scene = base.model_for("text"), base.model_for("scene")
    objaverse_dir = home / "cache" / "objaverse"
    cats = interview.CategoryIndex(cache_dir=objaverse_dir, live=interview.live_searcher(objaverse_dir))
    researcher = research.searcher()  # None without a Tavily key

    def session():
        # Its own client, so its calls (the interview and the photo analysis) are logged apart: the job it
        # creates shows what its interview used (interview_start sets the log).
        client = replace(base)
        return interview.Session(client, model, cats, scene_check=interview.check_scene,
                                 photo_ranker=interview.photo_ranker(client, scene),
                                 scene_profiler=interview.scene_profiler(client, scene),
                                 photo_models=backend != "nebius", uploads_only=public, researcher=researcher)
    return session


MAX_PHOTOS = 400  # per upload on a public server
SESSION_HOURS = 6
CALIBRATION = "calibration.json"  # a new job whose teacher the user checks before the first run
CALIBRATION_PHOTOS = 6  # photos the teacher marks per calibration round
# Model calls are logged by purpose, which is the stage name except for qa's two-step strategy.
PURPOSE_STAGE = {"qa_describe": "qa_assets", "qa_decide": "qa_assets"}  # an interview left open longer is dropped


def create_app(home: Path, backend: str = "local", max_usd: float | None = None, interviewer=None,
               launcher=None, public: bool = False, key: str | None = None, max_runs: int = 2,
               daily_usd: float | None = None, auth=None, accounts=None, contact: str | None = None, warm=None):
    """public: a server others can reach. Specs may only use photos uploaded through it (no server paths),
    at most `max_runs` runs at a time, and, with `key`, the API needs it (a link with ?key=... or POST /api/key
    sets a cookie); the pages and the app's files stay open, so anyone sees the landing and the app asks for the
    key (user, 2026-10-03). Each browser gets a visitor cookie and only sees the jobs it started.
    daily_usd: no new run when the cloud GPU spent in the last 24 h plus one more run could pass it.
    auth (auth.Auth) and accounts (accounts.Accounts): login with Google or GitHub instead of the key; jobs
    belong to the account (so they follow it to the phone), and interviews and runs are paid with credits,
    free for judges (docs/plan_login.md). contact: the address judges write to for free access.
    warm (warm.WarmGPU): a cloud GPU asked for when a user uploads photos and kept while they check the teacher,
    so their runs skip the start-up (Nebius only)."""
    import hmac

    from fastapi.responses import JSONResponse

    from .interview import FIRST_QUESTION, spec_paths

    app = FastAPI(title="EyeForge")

    if public:  # added before the key check, so it runs after it (the last middleware added runs first)
        @app.middleware("http")
        async def visitor_cookie(request: Request, call_next):
            given = request.cookies.get(VISITOR, "")
            request.state.visitor = given if re.fullmatch(r"[0-9a-f]{32}", given) else uuid.uuid4().hex
            response = await call_next(request)
            if given != request.state.visitor:
                response.set_cookie(VISITOR, request.state.visitor, httponly=True, samesite="strict",
                                    max_age=30 * 86400)
            return response

    def key_cookie(response) -> None:
        response.set_cookie("eyeforge_key", key, httponly=True, samesite="strict", max_age=14 * 86400)

    if auth is not None:
        @app.middleware("http")
        async def require_login(request: Request, call_next):
            acc_id = auth.unsign(request.cookies.get(SESSION))
            acc = accounts.get(acc_id) if acc_id else None
            request.state.account = acc
            path = request.url.path
            if path.startswith("/api/") and path not in OPEN_API:
                if acc is None:
                    return JSONResponse({"detail": "sign in first"}, status_code=401)
                if accounts.tier(acc) == "banned":
                    return JSONResponse({"detail": paused_text()}, status_code=403)
            return await call_next(request)

    def paused_text() -> str:
        return ("Your account was paused by an automatic safety check. If this is a mistake, write to "
                f"{contact or 'us'} and we will sort it out.")

    if key and auth is None:
        @app.middleware("http")
        async def require_key(request: Request, call_next):
            given = request.query_params.get("key") or request.cookies.get("eyeforge_key") or \
                request.headers.get("x-eyeforge-key") or ""
            request.state.unlocked = hmac.compare_digest(given, key)
            path = request.url.path
            if not request.state.unlocked and path.startswith("/api/") and path not in OPEN_API:
                return JSONResponse({"detail": "this server needs the access key"}, status_code=401)
            response = await call_next(request)
            if request.state.unlocked and request.query_params.get("key"):
                key_cookie(response)
            return response

    @app.post("/api/key")
    async def unlock(request: Request):
        """The access key typed in the app: right, a cookie for the next requests; wrong, 401 after a pause
        that makes guessing slow."""
        given = str((await request.json()).get("key", "")) if key else ""
        if key and not hmac.compare_digest(given, key):
            await asyncio.sleep(1.0)
            raise HTTPException(401, "wrong access key")
        response = JSONResponse({"ok": True})
        if key:
            key_cookie(response)
        return response

    sessions: dict[str, object] = {}
    runs: dict[str, subprocess.Popen] = {}
    calibrations: dict[str, threading.Thread] = {}
    started: dict[str, float] = {}  # interview session -> start time
    new_session = interviewer
    launch = launcher or _launch

    def visitor(request: Request) -> str | None:
        if auth is not None:
            acc = getattr(request.state, "account", None)
            return acc.id if acc else None
        return getattr(request.state, "visitor", None) if public else None

    def pay(request: Request, cost: float, reason: str, detail: str | None = None):
        """Charge the signed-in account (free for judges); 402 without enough credits. None without accounts."""
        if accounts is None:
            return None
        acc = request.state.account
        if not accounts.charge(acc, cost, reason, detail):
            if accounts.tier(acc) == "banned":
                raise HTTPException(403, paused_text())
            raise HTTPException(402, f"not enough credits for this {reason} ({accounts.balance(acc.id):g} left)")
        return acc

    def mine(job: Job, request: Request) -> bool:
        if not public:
            return True
        owner = job.root / OWNER
        return owner.exists() and owner.read_text().strip() == visitor(request)

    def job_of(name: str, request: Request) -> Job:
        job = Job(home / "jobs" / slug(name))
        if not job.spec_path.exists() and not (job.root / "request.json").exists() or not mine(job, request):
            raise HTTPException(404, f"no job {name}")
        return job

    def calibrate(job: Job) -> None:
        """Before any training, the teacher marks a few of the user's photos. The user corrects them; the next
        round tunes the teacher to the corrections, marks the same photos again and a few new ones, so the user
        sees if it understood. The first run starts from these answers. The teacher (SAM 3) needs a GPU: this
        machine's when it has one, else a short Nebius job (the photos wait for the GPU to start)."""
        from . import teacher
        from .stages import Context
        from .stages.label import Label
        state = job.root / CALIBRATION

        def here():
            stage = Label()
            stage.sample = CALIBRATION_PHOTOS
            stage.run(Context(spec_mod.load(job.root / "request.json"), job, Cache(home / "cache")))

        def cloud():
            cmd = [sys.executable, "-m", "sim2chip.cli", "--home", str(home), "run", str(job.root / "request.json"),
                   "--job", job.name, "--backend", backend, "--only", "label"]
            if max_usd is not None:
                cmd += ["--max-usd", str(max_usd)]
            job.logs.mkdir(parents=True, exist_ok=True)
            with open(job.logs / "pipeline.log", "a") as log:
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=Path(__file__).resolve().parents[1],
                                        env={**os.environ, "SIM2CHIP_LABEL_SAMPLE": str(CALIBRATION_PHOTOS)})
                runs[job.name] = proc  # cancel stops it like a run
                if proc.wait() != 0:
                    lines = (job.logs / "pipeline.log").read_text(errors="replace").splitlines()
                    raise RuntimeError(next((l for l in reversed(lines) if "error" in l.lower() or "FAILED" in l),
                                            f"the teacher's cloud job failed (exit {proc.returncode})"))
            job.clear_done("label")  # a sample, not the whole folder: the first run labels every photo

        def answers_here() -> bool:
            """Every answer the round needs is cached on this machine (the warm GPU's prefetch brought them): only
            the threshold or the centres change with the corrections, which takes seconds without a GPU."""
            try:
                stage = Label()
                stage.sample = CALIBRATION_PHOTOS
                return not stage.gpu_needed(Context(spec_mod.load(job.root / "request.json"), job,
                                                    Cache(home / "cache")))
            except Exception as e:  # then the round asks the teacher for them
                print(f"calibration {job.name}: could not check the cached answers ({type(e).__name__}: {e})",
                      flush=True)
                return False

        def work():
            try:
                here() if teacher.available() or backend != "nebius" or answers_here() else cloud()
                state.write_text(json.dumps({"state": "ready"}))
                if warm is not None:
                    warm.touch(who=job.name)  # the user checks the photos now: keep the GPU for the next round or the run
                    prefetch(job)
            except Exception as e:  # shown on the job page, with the option to train anyway
                state.write_text(json.dumps({"state": "failed", "error": str(e)[:300]}))

        state.write_text(json.dumps({"state": "labeling"}))
        calibrations[job.name] = threading.Thread(target=work, daemon=True)
        calibrations[job.name].start()

    prefetching: set[str] = set()
    prefetch_lock = threading.Lock()  # calibration threads and requests may ask for the same job at once

    def prefetch(job: Job) -> None:
        """Count calibration: while the user checks the first photos, the warm GPU answers for the rest, so the
        next rounds and the first run need no GPU work for the teacher (warm.py, Label.prefetch)."""
        from .stages import Context
        from .stages.label import Label
        spec = spec_mod.load(job.root / "request.json")
        if spec.task != "count":
            return
        try:
            if not Label().gpu_needed(Context(spec, job, Cache(home / "cache"))):
                return  # every photo answered already
        except Exception as e:  # prefetch anyway: the warm GPU skips the photos it finds answered
            print(f"prefetch {job.name}: could not check the cached answers ({type(e).__name__}: {e})", flush=True)
        with prefetch_lock:
            if job.name in prefetching:
                return
            prefetching.add(job.name)

        def go():
            try:
                from .backends.nebius import NebiusBackend
                NebiusBackend(max_usd=max_usd).prefetch_warm(job, Cache(home / "cache"), spec, CALIBRATION_PHOTOS)
            except Exception as e:  # the rounds and the run then ask the teacher themselves
                print(f"prefetch {job.name} failed: {e}", flush=True)
            finally:
                prefetching.discard(job.name)
        threading.Thread(target=go, daemon=True).start()

    def warm_up(reason: str, who: str | None = None) -> None:
        """Ask for the warm GPU, unless today's budget could not pay for it."""
        if warm is None:
            return
        try:
            budget_left()
        except HTTPException:
            return
        warm.start(reason, who)

    def interview_usage(sid: str) -> Path:
        return home / "logs" / "interviews" / f"{sid}.usage.jsonl"

    def credit_interview(job: Job, sid: str | None) -> None:
        """The model calls of the interview that wrote this job's spec go with the job (usage panel)."""
        if not sid or not re.fullmatch(r"[0-9a-f]{32}", sid) or not interview_usage(sid).exists():
            return
        job.logs.mkdir(parents=True, exist_ok=True)
        with open(job.logs / "interview_usage.jsonl", "a") as f:
            f.write(interview_usage(sid).read_text())
        interview_usage(sid).unlink()  # counted once, even if the chat goes on

    def start_run(job: Job, request: Path, start: str | None = None, payer: Request | None = None):
        """payer: the HTTP request of the user who pays this run (accounts mode), refunded if it cannot start."""
        acc = pay(payer, RUN_COST, "run", job.name) if payer is not None else None
        try:
            (job.logs / "launched_at").write_text(f"{time.time():.3f}")
            runs[job.name] = launch(home, job, request, backend, max_usd, start=start)
        except Exception:
            if acc is not None:
                accounts.refund(acc, RUN_COST, job.name)
            raise

    def busy() -> bool:
        return public and sum(p.poll() is None for p in runs.values()) >= max_runs

    @app.get("/", include_in_schema=False)
    def index():
        page = DIST / "index.html"
        if not page.exists():
            return HTMLResponse(NO_BUILD, 503)
        return HTMLResponse(page.read_text())

    @app.get("/api/config")
    def config(request: Request):
        """What the app shows before a run: can this server start one now, and what is left of today's budget."""
        budget = None
        if daily_usd is not None and backend == "nebius":
            from .backends.nebius import USD_PER_HOUR, WORKER_HOURS
            spent = cloud_spent(home)
            budget = {"daily_usd": daily_usd, "spent_usd": round(spent, 2),
                      "can_run": spent + WORKER_HOURS * USD_PER_HOUR <= daily_usd}
        return {"backend": backend, "first_question": FIRST_QUESTION, "public": public, "busy": busy(),
                "max_runs": max_runs if public else None, "budget": budget,
                "locked": auth is None and bool(key) and not getattr(request.state, "unlocked", False),
                "login": auth.providers if auth is not None else None}

    @app.get("/api/me")
    def me(request: Request):
        """Who is signed in, their tier and credits; and where judges write for free access."""
        acc = getattr(request.state, "account", None)
        judging = accounts is not None and (accounts.judge_until is None or time.time() < accounts.judge_until)
        out = {"signed_in": acc is not None, "contact": contact, "judging": judging,
               "costs": {"run": RUN_COST, "interview": INTERVIEW_COST}}
        if acc is not None:
            out["account"] = {"email": acc.email, "name": acc.name, "provider": acc.provider,
                              "tier": accounts.tier(acc), "credits": accounts.balance(acc.id)}
        return out

    if auth is not None:
        @app.get("/auth/{provider}/login", include_in_schema=False)
        def login(provider: str, next: str = "/app"):
            if provider not in auth.providers:
                raise HTTPException(404, f"no login with {provider}")
            nonce = uuid.uuid4().hex
            back = next if next.startswith("/") and not next.startswith("//") else "/app"
            response = RedirectResponse(auth.login_url(provider, nonce), 302)
            response.set_cookie(STATE, auth.sign(json.dumps({"nonce": nonce, "next": back}), 600),
                                httponly=True, samesite="lax", secure=auth.public_url.startswith("https"),
                                max_age=600)
            return response

        @app.get("/auth/{provider}/callback", include_in_schema=False)
        def login_back(provider: str, request: Request, code: str = "", state: str = ""):
            from .auth import AuthError
            saved = auth.unsign(request.cookies.get(STATE))
            saved = json.loads(saved) if saved else None
            if provider not in auth.providers or not code or not saved \
                    or not hmac.compare_digest(state, saved["nonce"]):
                return RedirectResponse("/app?login=failed", 302)
            try:
                who = auth.identity(provider, code)
            except AuthError:
                return RedirectResponse("/app?login=failed", 302)
            acc = accounts.login(who.provider, who.subject, who.email, who.verified, who.name)
            response = RedirectResponse(saved["next"], 302)
            response.set_cookie(SESSION, auth.sign(acc.id, SESSION_DAYS * 86400), httponly=True, samesite="lax",
                                secure=auth.public_url.startswith("https"), max_age=SESSION_DAYS * 86400)
            response.delete_cookie(STATE)
            return response

        @app.post("/auth/logout", include_in_schema=False)
        def logout():
            response = JSONResponse({"ok": True})
            response.delete_cookie(SESSION)
            return response

    def check_paths(parsed: spec_mod.Spec, job: Job | None = None) -> None:
        """On a public server, only photos uploaded here (or, revising a job, the job's own folders): a path
        would let a spec read, and upload to the cloud, any file of the server."""
        if not public:
            return
        uploads = (home / "uploads").resolve()

        def ok(p: str) -> bool:
            if p == "photos" or (Path(p).is_absolute() and Path(p).resolve().is_relative_to(uploads)):
                return True
            return job is not None and not Path(p).is_absolute() and ".." not in Path(p).parts \
                and (job.input / p).exists() and not (job.input / p).is_symlink()
        bad = [p for p in spec_paths(parsed) if not ok(p)]
        if bad:
            raise HTTPException(400, f"invalid spec: only photos uploaded here can be used, not {bad}")

    @app.post("/api/interview")
    def interview_start(request: Request, job: str | None = None):
        """A new interview, or (job=<name>) one that revises a trained job's spec."""
        nonlocal new_session
        revising = job_of(job, request) if job else None
        pay(request, INTERVIEW_COST, "interview", revising.name if revising else None)
        if new_session is None:
            new_session = default_interviewer(home, backend, public)
        now = time.time()
        for old in [k for k, t in started.items() if now - t > SESSION_HOURS * 3600]:  # abandoned chats
            sessions.pop(old, None), started.pop(old, None), locks.pop(old, None)
        sid = uuid.uuid4().hex
        started[sid] = now
        sessions[sid] = new_session()
        sessions[sid].uploads = home / "uploads" / sid
        sessions[sid].transcript = home / "logs" / "interviews" / f"{sid}.json"
        if hasattr(getattr(sessions[sid], "client", None), "usage_log"):
            sessions[sid].client.usage_log = interview_usage(sid)
        if revising:
            turn = sessions[sid].revise(spec_mod.load(latest_spec(revising)))
            return {"session": sid, "text": turn.text, "job": revising.name}
        return {"session": sid, "text": FIRST_QUESTION}

    locks: dict[str, threading.Lock] = {}

    def guarded(sid: str, fn, *args, act: str | None = None):
        """One call at a time per interview (a double submit would interleave two model turns), and the
        right status: 502 when the model service fails, 409 when the interview is already over.
        act: the front end's id for this turn, under which its steps are shown (see interview_activity)."""
        from .llm import LLMError
        with locks.setdefault(sid, threading.Lock()):
            session = sessions.get(sid)
            if hasattr(session, "begin"):
                session.begin(act)
            try:
                return fn(*args)
            except LLMError as e:
                raise HTTPException(502, f"model service: {e}") from None
            except RuntimeError as e:
                raise HTTPException(409, str(e)) from None

    def turn_out(sid: str, turn) -> dict:
        out = {"kind": turn.kind, "text": turn.text}
        steps = getattr(sessions.get(sid), "activity", None)
        if steps:
            out["activity"] = [dict(s) for s in steps]
        if turn.spec is not None:
            out["spec"] = json.loads(turn.spec.to_json())
        if turn.request is not None:
            out["request"] = turn.request
        if turn.intro is not None:
            out["intro"] = turn.intro
        if turn.kind in ("done", "failed"):
            sessions.pop(sid, None)
            locks.pop(sid, None)
        return out

    @app.post("/api/interview/{sid}")
    def interview_reply(sid: str, body: dict, act: str | None = None):
        session = sessions.get(sid)
        if session is None:
            raise HTTPException(404, "unknown interview session")
        return turn_out(sid, guarded(sid, session.reply, str(body.get("text", "")), act=act))

    @app.get("/api/interview/{sid}/activity")
    def interview_activity(sid: str, act: str):
        """What the turn `act` is doing right now: its steps so far (thinking, searching, checking the setup),
        polled by the web while it waits. No lock: it must answer while the turn holds it. `now` lets the
        client time running steps with the server's clock."""
        session = sessions.get(sid)
        if session is None:
            raise HTTPException(404, "unknown interview session")
        steps = list(getattr(session, "activity", [])) if getattr(session, "activity_id", None) == act else []
        return {"steps": [dict(s) for s in steps], "now": time.time()}

    @app.post("/api/interview/{sid}/photos")
    def interview_photos(sid: str, purpose: str = Form("training"), cls: str = Form(""),
                               photos: list[UploadFile] = File(...), act: str | None = None,
                               more: bool = Form(False), total: int = Form(0)):
        """Photos uploaded in the chat, asked for by the interview or not. They wait in uploads/<session>
        until a job adopts them. Sync on purpose: the model calls after the upload block, and FastAPI runs
        sync handlers in a thread pool instead of the event loop (async stalled every other request).
        A big upload comes in batches (a proxy caps the request size): more=true only stores the batch, and
        the last one, with the total of all batches, makes the interview's turn."""
        session = sessions.get(sid)
        if session is None:
            raise HTTPException(404, "unknown interview session")
        purpose = "3d" if purpose == "3d" else "training"
        folder = home / "uploads" / sid / (f"3d_{slug(cls)}" if purpose == "3d" and cls else purpose)
        folder.mkdir(parents=True, exist_ok=True)
        n = 0
        for p in photos:
            if Path(p.filename or "").suffix.lower() in IMAGE_EXT:
                (folder / Path(p.filename).name).write_bytes(p.file.read())
                n += 1
        if not n and not total:
            raise HTTPException(400, "no images among the files")
        if purpose == "training":  # the GPU starts while the interview goes on
            warm_up("photos uploaded", f"session:{sid}")
        if more:
            return {"stored": n}
        n = max(n, total)
        return {**turn_out(sid, guarded(sid, session.add_photos, str(folder), n, purpose, cls or None,
                                         act=act)), "photos": n}

    @app.get("/api/interview/{sid}/thumb")
    def interview_thumb(sid: str, file: str):
        """Thumbnail of a photo the interview is proposing (only those: the path comes from the user)."""
        session = sessions.get(sid)
        if session is None or file not in {c["file"] for c in session.candidates}:
            raise HTTPException(404, "not a proposed photo")
        return FileResponse(photo_thumb(home / "uploads" / sid / "_thumbs", Path(file)))

    @app.post("/api/interview/{sid}/pick")
    def interview_pick(sid: str, body: dict, act: str | None = None):
        session = sessions.get(sid)
        if session is None:
            raise HTTPException(404, "unknown interview session")
        return turn_out(sid, guarded(sid, session.choose, [str(f) for f in body.get("files") or []],
                                     str(body.get("class", "")), act=act))

    def budget_left() -> None:
        if daily_usd is None or backend != "nebius":
            return
        from .backends.nebius import USD_PER_HOUR, WORKER_HOURS
        spent, worst = cloud_spent(home), WORKER_HOURS * USD_PER_HOUR
        if spent + worst > daily_usd:
            raise HTTPException(429, f"today's cloud budget is used up (${spent:.2f} of ${daily_usd:.2f}); "
                                     "try again tomorrow")

    @app.post("/api/jobs")
    def create_job(request: Request, spec: str = Form(...), photos: list[UploadFile] = File(default=[]),
                   calibrate_first: bool = Form(False, alias="calibrate"), session: str | None = Form(None)):
        """A new job. calibrate: when the teacher labels the user's raw photos, it first marks a few for the user
        to check (see calibrate); the run starts from the job page."""
        budget_left()
        if public:
            if busy():
                raise HTTPException(429, f"{max_runs} models are training right now; try again in a few minutes")
            if len(photos) > MAX_PHOTOS:
                raise HTTPException(413, f"at most {MAX_PHOTOS} photos per run")
        try:
            data = json.loads(spec)
            images = [p for p in photos if Path(p.filename or "").suffix.lower() in IMAGE_EXT]
            if images and not (data.get("user_data") or {}).get("labeled"):
                data.setdefault("user_data", {})["raw"] = "photos"  # the teacher model labels them
            parsed = spec_mod.from_dict(data)
        except (json.JSONDecodeError, spec_mod.SpecError) as e:
            raise HTTPException(400, f"invalid spec: {e}") from None
        check_paths(parsed)
        # The same setup sent twice (a double click, "Looks good" and Train together) must not pay for two runs.
        same = json.loads(parsed.to_json())
        for other, proc in runs.items():
            req = home / "jobs" / other / "request.json"
            if proc.poll() is None and req.exists() and {**json.loads(req.read_text()), "name": same["name"]} == same \
                    and mine(Job(home / "jobs" / other), request):
                return {"job": other, "photos": 0, "already_running": True}
        # A new request never reuses an existing job folder: it would overwrite that job's results and pick
        # up its review corrections. Reruns of a job go through the corrections endpoint instead.
        base = name = slug(parsed.name)
        n = 1
        while (home / "jobs" / name).exists():
            n += 1
            name = f"{base}_{n}"
        job = open_job(home, name)
        if public:
            (job.root / OWNER).write_text(visitor(request))
        parsed = adopt_uploads(parsed, home / "uploads", job)
        req = job.root / "request.json"
        req.write_text(json.dumps({**json.loads(parsed.to_json()), "name": name}, indent=2))
        save_photos(job.input / "photos", images)
        credit_interview(job, session)
        if session:
            warm_mod.note(home, "job", name, session=f"session:{session}")
        if calibrate_first and parsed.user_data.raw and not parsed.user_data.labeled:
            pay(request, RUN_COST, "run", name)  # the teacher's check is the start of the paid run
            warm_up("calibration", name)
            calibrate(job)
            return {"job": name, "photos": len(images), "calibrating": True}
        start_run(job, req, payer=request)
        return {"job": name, "photos": len(images)}

    @app.get("/api/jobs")
    def list_jobs(request: Request):
        root = home / "jobs"
        names = sorted(d.name for d in root.iterdir() if ((d / "spec.json").exists() or (d / "request.json").exists())
                       and mine(Job(d), request)) if root.exists() else []
        return {"jobs": names}

    @app.get("/api/jobs/{name}")
    def job_status(name: str, request: Request):
        job = job_of(name, request)
        return status(job, home, running=running(job), backend=backend)

    @app.get("/api/jobs/{name}/review")
    def review_data(name: str, request: Request):
        job = job_of(name, request)
        spec = spec_mod.load(job.spec_path if job.spec_path.exists() else job.root / "request.json")
        return review.gather(job, Cache(home / "cache"), spec.labels, task=spec.task)

    @app.get("/api/jobs/{name}/media/{path:path}")
    def media(name: str, path: str, request: Request, size: int = 360):
        """A small JPEG of an image inside the job folder (only the folders review shows); size 1024 for the
        point editor."""
        job = job_of(name, request)
        src = (job.root / path).resolve()
        if not src.is_relative_to(job.root.resolve()) or Path(path).parts[0] not in review.MEDIA_DIRS \
                or src.suffix.lower() not in IMAGE_EXT or not src.is_file():
            raise HTTPException(404, "no such image")
        return FileResponse(thumbnail(job, src, 1024 if size > 360 else 360))

    def running(job: Job) -> bool:
        """This server's run process, or a cloud job left by an earlier server process (still in time)."""
        proc, cal = runs.get(job.name), calibrations.get(job.name)
        return (proc is not None and proc.poll() is None) or cloud_job(job) is not None or \
            (cal is not None and cal.is_alive())

    @app.post("/api/jobs/{name}/cancel")
    def cancel(name: str, request: Request):
        """Stops the run: the local process and, on Nebius, the cloud job (which bills until it ends)."""
        job = job_of(name, request)
        proc = runs.get(job.name)
        if proc is not None and proc.poll() is None:
            proc.terminate()
        cj = cloud_job(job)
        if cj and cj.get("warm"):  # a task of the shared warm GPU: stop the task, not the GPU
            from .backends.nebius import NebiusBackend
            be = NebiusBackend()
            warm_mod.cancel(be.store, f"{be.prefix}-cache", cj["warm"], cj["task"])
            (job.logs / "cloud_job.json").unlink(missing_ok=True)
        elif cj:
            from .backends.nebius import NebiusCLI
            NebiusCLI().cancel(cj["id"])
            (job.logs / "cloud_job.json").unlink(missing_ok=True)
        with open(job.logs / "pipeline.log", "a") as f:
            f.write("FAILED: cancelled by the user\n")
        return {"cancelled": True, "cloud_job": cj["id"] if cj else None}

    @app.post("/api/jobs/{name}/train")
    def train_calibrated(name: str, body: dict, request: Request):
        """The user is happy with the teacher: keep the last fixes and start the first run. Its label stage
        learns from all the fixes and marks every photo."""
        job = job_of(name, request)
        if running(job):
            raise HTTPException(409, ALREADY_RUNNING)
        if not (job.root / CALIBRATION).exists():
            raise HTTPException(409, CHECK_OVER)
        budget_left()
        if public and busy():
            raise HTTPException(429, f"{max_runs} models are training right now; try again in a few minutes")
        review.apply(job, Cache(home / "cache"), body.get("labels") or {}, {}, body.get("points") or {},
                     share=not public)
        (job.root / CALIBRATION).unlink()
        start_run(job, job.root / "request.json")
        return {"rerun_from": None}

    @app.post("/api/jobs/{name}/corrections")
    def corrections(name: str, body: dict, request: Request):
        job = job_of(name, request)
        if running(job):
            raise HTTPException(409, ALREADY_RUNNING)
        # Fixes made on the teacher's check, sent after training started (from another device, or an old tab):
        # as fixes of a trained model they would start a run nobody asked for.
        if body.get("calibrating") and not (job.root / CALIBRATION).exists():
            raise HTTPException(409, CHECK_OVER)
        if (job.root / CALIBRATION).exists():  # still calibrating the teacher: another round, no training
            review.apply(job, Cache(home / "cache"), body.get("labels") or {}, {}, body.get("points") or {},
                         share=not public)
            calibrate(job)
            return {"rerun_from": "calibration"}
        budget_left()
        start = review.apply(job, Cache(home / "cache"), body.get("labels") or {}, body.get("assets") or {},
                             body.get("points") or {}, share=not public)
        if start is None:
            return {"rerun_from": None}
        snapshot(job, "corrected labels" if start == "label" else "reviewed 3D models")
        req = job.root / "request.json"
        if not req.exists():
            req.write_text(job.spec_path.read_text())
        start_run(job, req, start, payer=request)
        return {"rerun_from": start}

    @app.post("/api/jobs/{name}/revise")
    def revise(name: str, request: Request, spec: str = Form(...), dry: bool = Form(False),
               session: str | None = Form(None)):
        """A changed spec for a trained job: the job keeps its folder, the old model is kept as a version, and
        only the stages whose inputs changed run again. dry: just the stages that would run."""
        job = job_of(name, request)
        if running(job):
            raise HTTPException(409, f"job {job.name} is running")
        try:
            parsed = spec_mod.from_dict(json.loads(spec))
        except (json.JSONDecodeError, spec_mod.SpecError) as e:
            raise HTTPException(400, f"invalid spec: {e}") from None
        check_paths(parsed, job)
        old = spec_mod.load(latest_spec(job))
        changed = [k for k, v in json.loads(parsed.to_json()).items()
                   if k != "name" and json.loads(old.to_json()).get(k) != v]
        if not dry:
            parsed = adopt_uploads(parsed, home / "uploads", job)
            credit_interview(job, session)
        todo = [p.stage.name for p in Pipeline(parsed, job, Cache(home / "cache")).plan() if p.action == "run"]
        if dry or not todo:
            return {"stages": todo, "changed": changed, "rerun_from": None}
        budget_left()
        if public and busy():
            raise HTTPException(429, f"{max_runs} models are training right now; try again in a few minutes")
        req = job.root / "request.json"
        req.write_text(json.dumps({**json.loads(parsed.to_json()), "name": old.name}, indent=2))
        snapshot(job, "changed " + (", ".join(changed) or "the setup"))
        start_run(job, req, payer=request)
        return {"stages": todo, "changed": changed, "rerun_from": todo[0]}

    @app.post("/api/jobs/{name}/photos")
    def add_photos(name: str, request: Request, photos: list[UploadFile] = File(...), note: str = Form(""),
                   more: bool = Form(False), total: int = Form(0)):
        """More photos for a trained job, usually of what the model gets wrong. The teacher marks them, half
        go to test (the rest to training), and only the real-photo branch reruns: the synthetic data stays.
        In batches like the interview's photos: more=true stores, the last batch (total: all of them) reruns."""
        job = job_of(name, request)
        if running(job):
            raise HTTPException(409, f"job {job.name} is running")
        budget_left()
        if public and busy():
            raise HTTPException(429, f"{max_runs} models are training right now; try again in a few minutes")
        images = [p for p in photos if Path(p.filename or "").suffix.lower() in IMAGE_EXT]
        if not images:
            raise HTTPException(400, "no photos (jpg, png, bmp or webp)")
        req = latest_spec(job)
        data = json.loads(req.read_text())
        ud = data.setdefault("user_data", {})
        raw = ud.get("raw") or "photos"
        folder = job.input / raw
        # Only a folder of the job's own: a linked folder is the user's, and a path may be anywhere on disk.
        if Path(raw).is_absolute() or ".." in Path(raw).parts or folder.is_symlink():
            raise HTTPException(400, f"this job reads its photos from {raw}: add them there and rerun from label")
        have = sum(1 for f in folder.rglob("*") if f.suffix.lower() in IMAGE_EXT) if folder.is_dir() else 0
        if public and have + len(images) > MAX_PHOTOS:
            raise HTTPException(413, f"at most {MAX_PHOTOS} photos per model ({have} already)")
        ud["raw"] = raw
        saved = save_photos(folder / "added", images)
        if more:
            return {"stored": saved}
        saved = max(saved, total)
        (job.root / "request.json").write_text(json.dumps({**data, "name": data.get("name", job.name)}, indent=2))
        note = " ".join(note.split())[:120]
        snapshot(job, f"added {saved} photos" + (f": {note}" if note else ""))
        start_run(job, job.root / "request.json", "label", payer=request)
        return {"rerun_from": "label", "photos": saved}

    @app.get("/api/jobs/{name}/versions/{n}/files/{file}")
    def download_version(name: str, n: int, file: str, request: Request):
        if file not in DOWNLOADS:
            raise HTTPException(404, "not a downloadable file")
        job = job_of(name, request)
        path = job.root / "versions" / f"v{n}" / file
        if not path.exists():
            raise HTTPException(404, "no such version or file")
        return FileResponse(path, filename=f"{job.name}_v{n}_{file}")

    @app.get("/api/jobs/{name}/files/{file}")
    def download(name: str, file: str, request: Request):
        if file not in DOWNLOADS:
            raise HTTPException(404, "not a downloadable file")
        job = job_of(name, request)
        path = job.stage_dir("export") / file
        if not path.exists():
            raise HTTPException(404, "not ready")
        return FileResponse(path, filename=f"{job.name}_{file}")

    @app.get("/{path:path}", include_in_schema=False)
    def app_files(path: str):
        """The app's own files (scripts, icons, manifest, service worker); any other path is a page of the
        app, which routes itself."""
        if path.startswith("api/"):
            raise HTTPException(404, "no such endpoint")
        f = (DIST / path).resolve()
        if f.is_file() and f.is_relative_to(DIST.resolve()):
            return FileResponse(f)
        return index()

    return app


def _launch(home: Path, job: Job, request: Path, backend: str, max_usd: float | None,
            start: str | None = None) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "sim2chip.cli", "--home", str(home), "run", str(request), "--job", job.name,
           "--backend", backend]
    if start:
        cmd += ["--from", start]
    if max_usd is not None:
        cmd += ["--max-usd", str(max_usd)]
    job.logs.mkdir(parents=True, exist_ok=True)
    with open(job.logs / "pipeline.log", "a") as log:  # the child keeps its own copy of the file
        return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=Path(__file__).resolve().parents[1])


def cloud_spent(home: Path, hours: float = 24) -> float:
    """Cloud GPU dollars of every job's runs that ended in the last `hours`, plus the runs still going
    (counted at their worst case, the whole timeout). The warm GPU counts once, for its whole life (warm.py):
    its tasks' records in the jobs are left out."""
    from . import warm
    from .backends.nebius import USD_PER_HOUR
    since = time.time() - hours * 3600
    total = warm.spent(home, since)
    for logs in (home / "jobs").glob("*/logs"):
        try:
            for line in (logs / "cloud_usage.jsonl").read_text().splitlines():
                rec = json.loads(line)
                if rec.get("t", 0) >= since and not rec.get("warm"):
                    total += rec.get("usd_estimate", 0.0)
        except (OSError, ValueError):
            pass
        cj = cloud_job(Job(logs.parent))
        if cj and not cj.get("warm"):
            total += cj["hours"] * USD_PER_HOUR
    return total


def cloud_job(job: Job) -> dict | None:
    """The Nebius job a run left running, while it can still be running (its timeout plus the queue)."""
    f = job.logs / "cloud_job.json"
    try:
        cj = json.loads(f.read_text())
    except (OSError, ValueError):
        return None
    return cj if time.time() - cj["started"] < cj["hours"] * 3600 else None


def save_photos(folder: Path, images: list[UploadFile]) -> int:
    """Writes uploaded photos under their own names; a different photo with a name already there gets a
    numbered one (phones reuse names like image.jpg)."""
    folder.mkdir(parents=True, exist_ok=True)
    for p in images:
        data, name = p.file.read(), Path(p.filename or "photo.jpg").name
        dst, n = folder / name, 1
        while dst.exists() and dst.read_bytes() != data:
            n += 1
            dst = folder / f"{Path(name).stem}_{n}{Path(name).suffix}"
        dst.write_bytes(data)
    return len(images)


def latest_spec(job: Job) -> Path:
    """A relaunched job has a new request.json while spec.json is still the previous run's until the run
    starts: the newer one."""
    paths = [p for p in (job.spec_path, job.root / "request.json") if p.exists()]
    return max(paths, key=lambda p: p.stat().st_mtime)


def adopt_uploads(spec: spec_mod.Spec, uploads: Path, job: Job) -> spec_mod.Spec:
    """Move chat uploads the spec points to into the job's input folder and make the paths relative, so
    the photos travel with the job (to Nebius too) and outlive the interview session."""
    import shutil

    def adopt(path: str | None) -> str | None:
        if not path or not Path(path).is_absolute() or not Path(path).resolve().is_relative_to(uploads.resolve()):
            return path
        src = Path(path)
        name = f"{src.parent.name}_{src.name}"  # <session>_<purpose>: unique per job
        dst = job.input / name
        if src.is_dir():
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst)
        return name

    data = json.loads(spec.to_json())
    ud = data["user_data"]
    for k in ("raw", "labeled", "test"):
        ud[k] = adopt(ud.get(k))
    for c in data["classes"]:
        if c.get("assets"):
            c["assets"]["photos"] = adopt(c["assets"].get("photos"))
    return spec_mod.from_dict(data)


def thumbnail(job: Job, src: Path, size: int = 360) -> Path:
    """Cached JPEG thumbnail of an image in the job folder."""
    rel = src.relative_to(job.root.resolve())
    return photo_thumb(job.root / "_thumbs", src, rel.with_suffix(".jpg" if size == 360 else f".{size}.jpg"), size)


def photo_thumb(cache: Path, src: Path, rel: Path | None = None, size: int = 360) -> Path:
    """Cached JPEG thumbnail, upright (phone photos carry their orientation in EXIF)."""
    import hashlib

    from PIL import Image, ImageOps

    rel = rel or Path(hashlib.sha256(str(src).encode()).hexdigest()[:16] + ".jpg")
    out = cache / rel
    if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
        out.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((size, size))
            im.save(out, quality=85)
    return out


CLOUD_PHASES = {
    "PROVISIONING": "Waiting for Nebius to assign a GPU (usually 1 min, longer when the region is busy)",
    "STARTING": "Nebius is starting the GPU machine",
    "IMAGE_PULLING": "Downloading the Isaac Sim image onto the machine (about 2 min)",
    "RUNNING": "Machine ready: installing the environment and fetching the job data",
}


def phase(log: list[str], running: bool) -> str | None:
    """What the run is doing now, in words, from the pipeline log: is the cloud still getting a machine ready,
    or is the pipeline working on a stage? The stage list alone shows "pending" in both cases."""
    if not running:
        return None
    stage = cloud = None
    for line in log:
        if line.startswith("> "):
            stage = line[2:].split(" ")[0]
        elif line.startswith("  state "):
            cloud = line.split()[-1]
        elif line.startswith("FAILED"):
            return "The pipeline failed (see the log below); the cloud machine is shutting down"
    if stage:
        return f"Running {stage}" + (" on Nebius" if cloud else "")
    if cloud:
        return CLOUD_PHASES.get(cloud, f"Nebius job: {cloud.lower()}")
    return "Starting the run"


LOCAL_STEPS = ("upload job", "download results")


def read_timeline(job: Job) -> list[dict]:
    f = job.logs / "timeline.jsonl"
    try:
        return [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    except (OSError, ValueError):
        return []


def launched_at(job: Job) -> float | None:
    """When this server last started a run of the job (request.json's time for runs started elsewhere)."""
    for f in (job.logs / "launched_at", job.root / "request.json"):
        try:
            return float(f.read_text()) if f.name == "launched_at" else f.stat().st_mtime
        except (OSError, ValueError):
            continue
    return None


def typical_seconds(home: Path, where: str) -> dict[str, float]:
    """How long each step took the last time any job ran it on this backend: the stages and, on Nebius, the
    cloud's own steps (waiting for a GPU, pulling the image). Good enough to say "about 12 min left"."""
    out: dict[str, tuple[float, float]] = {}
    for f in (home / "jobs").glob("*/logs/timeline.jsonl"):
        for rec in read_timeline(Job(f.parents[1])):
            here = rec.get("where") == where or rec["step"] in LOCAL_STEPS  # the server's side of a cloud run
            if here and not rec.get("failed") and rec["t"] > out.get(rec["step"], (0, 0))[0]:
                out[rec["step"]] = (rec["t"], rec["seconds"])
    return {k: v[1] for k, v in out.items()}


def timing(job: Job, home: Path, stages: list[dict], running: bool, backend: str) -> dict:
    """Seconds per finished stage of the current run and, while it runs, the time so far and an estimate of
    the time left from earlier runs (the cloud's start-up steps included until they happen)."""
    started = launched_at(job)
    this_run = [r for r in read_timeline(job) if started is None or r["t"] >= started]
    took = {r["step"]: r["seconds"] for r in this_run}
    for st in stages:
        st["seconds"] = took.get(st["name"]) if st["state"] == "done" else None
    if not running or started is None:
        return {"started_at": started, "elapsed_s": None, "eta_s": None}
    now = time.time()
    typical = typical_seconds(home, backend)
    names = {st["name"] for st in stages}
    left = sum(sec for step, sec in typical.items() if step not in names and step not in took and (
        step.startswith("nebius job:") and step != "nebius job: running" or step.startswith("cloud:")
        or step in LOCAL_STEPS))
    since = max([r["t"] for r in this_run], default=started)  # the current stage started when the last step ended
    for st in stages:
        if st["state"] == "running":
            left += max(typical.get(st["name"], 0) - (now - since), 0)
        elif st["state"] == "pending":
            left += typical.get(st["name"], 0)
    return {"started_at": round(started, 1), "elapsed_s": round(now - started), "eta_s": round(left)}


def status(job: Job, home: Path, running: bool, backend: str = "local") -> dict:
    spec_path = latest_spec(job)
    log_path = job.logs / "pipeline.log"
    log = log_path.read_text(errors="replace").splitlines() if log_path.exists() else []
    try:
        spec = spec_mod.load(spec_path)
    except spec_mod.SpecError as e:  # e.g. an old job whose spec a newer version no longer accepts
        return {"job": job.name, "spec": None, "running": running, "stages": [], "log": log[-30:] + [f"error: {e}"],
                "cloud_usd": 0.0, "report": None, "timing": None, "version": None, "versions": []}
    current = next((l[2:].split(" ")[0] for l in reversed(log) if l.startswith("> ")), None)
    failed = any(l.startswith("FAILED") or l.startswith("error:") for l in log[-5:])
    # Token Factory model each stage really called (the usage log records it per call, by stage name).
    llm_log = job.logs / "llm_usage.jsonl"
    models = {PURPOSE_STAGE.get(r["purpose"], r["purpose"]): r["model"]
              for r in map(json.loads, llm_log.read_text().splitlines())} if llm_log.exists() else {}
    stages = []
    for p in Pipeline(spec, job, Cache(home / "cache")).plan():
        state = "done" if p.action == "skip" else "pending"
        if state == "pending" and p.stage.name == current:
            state = "running" if running else ("failed" if failed else "pending")
        stages.append({"name": p.stage.name, "runtime": p.stage.runtime, "state": state,
                       "powered_by": p.stage.powered_by(spec, backend, models.get(p.stage.name))})
    cloud_log = job.logs / "cloud_usage.jsonl"
    cost = sum(json.loads(l)["usd_estimate"] for l in cloud_log.read_text().splitlines()) if cloud_log.exists() else 0.0
    report = job.stage_dir("export") / "report.json"
    fresh = any(st["name"] == "export" and st["state"] == "done" for st in stages)  # not a stale model
    calibration = job.root / CALIBRATION
    return {"job": job.name, "spec": json.loads(spec.to_json()), "running": running,
            "calibration": json.loads(calibration.read_text()) if calibration.exists() else None,
            "stages": stages, "log": log[-30:], "cloud_usd": round(cost, 2), "phase": phase(log, running),
            "timing": timing(job, home, stages, running, backend),
            "usage": usage.summary(job, usage.saved_prices(home / "cache"), config.gpu_name()),
            "version": current_version(job), "versions": [{k: v for k, v in m.items() if k != "report_key"}
                                                          for m in versions(job)],
            "report": json.loads(report.read_text()) if fresh and report.exists() else None}


def serve(home: Path, backend: str, host: str, port: int, max_usd: float | None, public: bool = False,
          key: str | None = None, max_runs: int = 2, daily_usd: float | None = None, auth=None,
          judge_until: float | None = None, contact: str | None = None) -> None:
    """auth: login with Google or GitHub (auth.Auth); accounts live in <home>/accounts.db."""
    import uvicorn
    from .accounts import Accounts
    # Token Factory's prices for the usage panel, in the background: the server starts without waiting.
    threading.Thread(target=usage.refresh_prices, args=(home / "cache",), daemon=True).start()
    accounts = Accounts(home / ACCOUNTS_DB, judge_until) if auth is not None else None
    gpu = warm_mod.WarmGPU(home) if backend == "nebius" and warm_mod.enabled() else None
    if gpu is not None and warm_mod.current(home):  # one left by the previous server process: note its end
        gpu.watch()
    uvicorn.run(create_app(home, backend, max_usd, public=public, key=key, max_runs=max_runs, daily_usd=daily_usd,
                           auth=auth, accounts=accounts, contact=contact, warm=gpu),
                host=host, port=port, proxy_headers=True)
