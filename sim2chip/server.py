"""Web API and the web app: interview, photo upload, run, progress, review and download.

    sim2chip serve [--backend local|nebius] [--host 127.0.0.1] [--port 8000]

The app is a React build in web/dist (from the repo's web/ folder: `cd web && npm install && npm run build`;
the Docker image builds it). Any path outside /api serves the app, which routes itself.

Each run is a `sim2chip run` subprocess writing to the job's logs/pipeline.log, so the server never
blocks on the pipeline and the command line stays the single way to run a job. Progress is read back
from the job folder (done markers, the pipeline log and the cloud usage log).

create_app keeps the middleware, who is asking and what they may start; the routes of each part of the app (the
interview, jobs, the library, walk scenes, billing) are in sim2chip/api.
"""
from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from . import config, library, review, usage, warm as warm_mod
from .accounts import MARGIN, START
from .version import engine
from .job_service import CALIBRATION, LABELING, JobService, cloud_spent
from . import spec as spec_mod
from .pipeline import Pipeline
from .stages.render import no_synthetic
from .workspace import Cache, Job, current_version, last_engine, versions

DOWNLOADS = {"model.tflite", "labels.txt", "model_int8.tflite", "model_data.h", "model.bin", "report.json"}
WEB = Path(__file__).parent / "web"
DIST = WEB / "dist"
OWNER = "owner"  # file in a public server's job folder: the visitor who started it
PAYER = "payer"  # file in a job folder: the account paying what it uses (accounts mode)
METER_S = 30     # how often the usage of running jobs is charged
NO_BUILD = ("<p>EyeForge: the web app is not built. Run <code>cd web && npm install && npm run build</code> in the "
            "repository, then reload. The API under /api works without it.</p>")
OPEN_API = {"/api/config", "/api/key", "/api/me", "/api/stripe/webhook"}  # API calls open to a visitor without the key or a login
SESSION = "eyeforge_session"  # cookie: the signed account id (login mode)
STATE = "eyeforge_oauth"      # cookie: the signed nonce and return page of a login in progress
SESSION_DAYS = 30
ACCOUNTS_DB = "accounts.db"  # in the server's home
VISITOR = "eyeforge_visitor"  # cookie: one per browser, so visitors of a public server only see their own jobs
# 409 answers when the same job is open twice (phone and computer): the web shows them as news, not errors.
ALREADY_RUNNING = "This model is already being worked on, maybe from another device."
CHECK_OVER = "Training already started, maybe from another device."


def library_ref(v) -> str | None:
    """The library id a spec value names ("library:<id>", or the bare id the chat sometimes writes), else None."""
    if isinstance(v, str) and re.fullmatch(r"[0-9a-f]{16}", v):
        return v
    return v[len("library:"):] if isinstance(v, str) and v.startswith("library:") else None


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")
    return s[:48] or "job"


def default_interviewer(home: Path, backend: str = "local", public: bool = False):
    from . import interview, research
    from .llm import Client

    from dataclasses import replace

    base = Client(usage_log=home / "logs" / "llm_usage.jsonl")
    model, scene = base.model_for("interview"), base.model_for("scene")
    objaverse_dir = home / "cache" / "objaverse"
    cats = interview.CategoryIndex(cache_dir=objaverse_dir, live=interview.live_searcher(objaverse_dir))
    researcher = research.searcher()  # None without a Tavily key

    def session():
        # Its own client, so its calls (the interview and the photo analysis) are logged apart: the job it
        # creates shows what its interview used (interview_start sets the log).
        client = replace(base)
        return interview.Session(client, model, cats, photo_ranker=interview.photo_ranker(client, scene),
                                 scene_profiler=interview.scene_profiler(client, scene),
                                 photo_models=backend != "nebius", uploads_only=public, researcher=researcher)
    return session


def attach_library(session, lib, owner: str, folders: Path, kit: str | None = None, on_open=None) -> None:
    """The owner's library to an interview (interview.Session.use_library). folders: where a dataset the chat
    looks at is put as plain photos; on_open: called when that happens (the server asks for the warm GPU, as for
    an upload)."""
    chosen = next((k["name"] for k in lib.kits(owner) if k["id"] == kit), None) if kit else None

    def open_dataset(ds: str) -> Path | None:
        if not lib.owns(owner, "dataset", ds):
            return None
        if on_open is not None:
            on_open()
        lib.materialize_dataset(owner, ds, folders / f"library_{ds}")
        return folders / f"library_{ds}"
    session.use_library(*lib.for_chat(owner), kit=chosen, notes=functools.partial(lib.describe, owner),
                        open_dataset=open_dataset, datasets={d["id"] for d in lib.datasets(owner)},
                        dataset_card=functools.partial(lib.dataset_card, owner))


MAX_PHOTOS = 400  # per upload on a public server
MAX_PHOTO_BYTES = 30 << 20   # one photo: a 50 MP phone photo is about 20 MB
MAX_MODEL_BYTES = 200 << 20  # one GLB the user uploads
SENT = "sent.json"           # in a job folder: the spec as the web sent it (a double click is the same job)
SESSION_OWNER = ".owner"     # file in uploads/<session>: who started that chat
CHAT_VIEW = ".view.json"     # file in uploads/<session>: the chat as the web shows it, to open it again anywhere
MAX_VIEW_BYTES = 2_000_000
CHATS = "chats.json"         # in a job folder: the chats that set it up, [{"t", "revision", "msgs": [{"from", "text"}]}]
SESSION_HOURS = 6  # an interview left open longer is dropped
# Model calls are logged by purpose, which is the stage name except for qa's two-step strategy.
PURPOSE_STAGE = {"qa_describe": "qa_assets", "qa_decide": "qa_assets"}


def create_app(home: Path, backend: str = "local", max_usd: float | None = None, interviewer=None,
               launcher=None, public: bool = False, key: str | None = None, max_runs: int = 2,
               daily_usd: float | None = None, auth=None, accounts=None, contact: str | None = None, warm=None,
               describer=None, payments=None, scenes=None):
    """public: a server others can reach. Specs may only use photos uploaded through it (no server paths),
    at most `max_runs` runs at a time, and, with `key`, the API needs it (a link with ?key=... or POST /api/key
    sets a cookie); the pages and the app's files stay open, so anyone sees the landing and the app asks for the
    key (user, 2026-10-03). Each browser gets a visitor cookie and only sees the jobs it started.
    daily_usd: no new run when the cloud GPU spent in the last 24 h plus one more run could pass it.
    auth (auth.Auth) and accounts (accounts.Accounts): login with Google or GitHub instead of the key; jobs
    belong to the account (so they follow it to the phone), and interviews and runs are paid with credits,
    free for judges (docs/plan_login.md). contact: the address judges write to for free access.
    warm (warm.WarmGPU): a cloud GPU asked for when a user uploads photos and kept while they check the teacher,
    so their runs skip the start-up (Nebius only). scenes: the scene GPU, where the runs that render queue."""
    import hmac

    from fastapi.responses import JSONResponse

    from .interview import FIRST_QUESTION

    app = FastAPI(title="EyeForge")

    @app.exception_handler(review.StackedMarks)
    async def stacked_marks(request: Request, exc: review.StackedMarks):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.middleware("http")
    async def safety_headers(request: Request, call_next):
        """No page of ours inside another site's frame (a hidden "turn on auto top-up" click), no guessing of
        file types, no full addresses sent to other sites. The camera stays allowed for the live view."""
        response = await call_next(request)
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Content-Security-Policy", "frame-ancestors 'none'")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("Permissions-Policy", "camera=(self), microphone=(), geolocation=()")
        if auth is not None and auth.public_url.startswith("https"):
            response.headers.setdefault("Strict-Transport-Security", "max-age=31536000")
        return response

    @app.middleware("http")
    async def upload_timing(request: Request, call_next):
        """Photo uploads timed in the log and in Server-Timing (seen in the browser's network panel): bytes,
        the whole request, and the parts the handler marks with `timed` (reading the form is what comes
        before the handler starts). Tells how much of a slow upload is the network and how much is us."""
        if request.method != "POST" or not request.url.path.endswith("/photos") and \
                request.url.path != "/api/library/datasets":
            return await call_next(request)
        t0 = time.perf_counter()
        request.state.timing = {"t0": t0}
        response = await call_next(request)
        total = time.perf_counter() - t0
        parts = {k: v for k, v in request.state.timing.items() if k != "t0"}
        response.headers["Server-Timing"] = ", ".join(
            [f"{k};dur={v * 1000:.0f}" for k, v in parts.items()] + [f"total;dur={total * 1000:.0f}"])
        size = int(request.headers.get("content-length") or 0)
        print(f"upload {request.url.path} {response.status_code}: {size / 1e6:.1f} MB in {total:.2f} s ("
              + ", ".join(f"{k} {v:.2f}" for k, v in parts.items()) + ")", flush=True)
        return response

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

    lib = library.Library(home, home / "cache")
    jobs = JobService(home, backend, max_usd, launcher=launcher, warm=warm, public=public, max_runs=max_runs,
                      daily_usd=daily_usd, on_labels=lib.take_teacher, scenes=scenes)
    profiling: set[str] = set()  # datasets whose profile is being made
    from concurrent.futures import ThreadPoolExecutor
    profiler = ThreadPoolExecutor(1, thread_name_prefix="profile")

    def profile_soon(owner: str, ds: str, payer: str | None = None) -> None:
        """Make a dataset's profile in the background (describer: photos, classes -> descriptions). The automatic
        one is on us (user, 2026-10-09); one the user asks for (payer: their account) describes the photos anew
        and is charged by use."""
        if describer is None or ds in profiling:
            return
        profiling.add(ds)

        def work():
            from .scene_profile import GENERAL
            log = home / "logs" / "profiles" / f"{ds}_{int(time.time())}.jsonl" if payer else None
            run = (lambda photos: describer(photos, GENERAL, usage_log=log)) if log else \
                (lambda photos: describer(photos, GENERAL))
            try:
                lib.profile_dataset(owner, ds, run, getattr(describer, "model", None), fresh=log is not None)
            except Exception as e:  # a failed profile is made again on the next visit
                print(f"profile of dataset {ds} failed: {type(e).__name__}: {e}", flush=True)
            finally:
                profiling.discard(ds)
                if log is not None and accounts is not None:
                    accounts.meter(payer, f"profile:{log.stem}", "profile",
                                   usage.spent(usage.saved_prices(home / "cache"), [log]))
        profiler.submit(work)

    def owner_of(request: Request) -> str:
        return visitor(request) or "local"

    def visitor(request: Request) -> str | None:
        if auth is not None:
            acc = getattr(request.state, "account", None)
            return acc.id if acc else None
        return getattr(request.state, "visitor", None) if public else None

    def may_start(request: Request, reason: str):
        """The signed-in account, when it may start this (accounts.START); 402 when its balance cannot cover the
        usual price, 403 when paused. None without accounts. Nothing is charged here: usage is (meter)."""
        if accounts is None:
            return None
        acc = request.state.account
        root = home / "jobs"
        held = sum(START["label" if (d / LABELING).exists() else "run"] for d in (root.iterdir() if root.exists() else [])
                   if (d / PAYER).exists() and (d / PAYER).read_text().strip() == acc.id and jobs.running(Job(d)))
        if not accounts.can_start(acc, reason, held):
            if accounts.tier(acc) == "banned":
                raise HTTPException(403, paused_text())
            raise HTTPException(402, f"your balance (${accounts.balance(acc.id):.2f}) does not cover a {reason}, "
                                     f"usually about ${START[reason]:.2f}")
        return acc

    def bill_to(job: Job, acc) -> None:
        """Who pays what this job uses (the meter reads it)."""
        if acc is not None:
            (job.root / PAYER).write_text(acc.id)

    chat_payer: dict[str, str] = {}  # interview session -> account paying its model calls

    def meter_chat(sid: str) -> None:
        if accounts is not None and sid in chat_payer and interview_usage(sid).exists():
            cost = usage.spent(usage.saved_prices(home / "cache"), [interview_usage(sid)])
            charged(chat_payer[sid], accounts.meter(chat_payer[sid], f"chat:{sid}", "chat", cost))

    def charged(acc_id: str, balance: float) -> None:
        """Usage was charged: an auto top-up may be due (Stripe, in the background)."""
        if payments is None:
            return

        def auto():
            try:
                done = payments.maybe_auto(acc_id, balance)
                if done:
                    print(f"auto top-up {acc_id}: {done}", flush=True)
            except Exception as e:
                print(f"auto top-up {acc_id} failed: {type(e).__name__}: {e}", flush=True)
        threading.Thread(target=auto, daemon=True).start()

    def meter_job(job: Job, running: bool) -> None:
        """Charge what the job has used so far; a running job whose payer is out of money stops (it can go on
        from where it was after a top-up)."""
        payer = job.root / PAYER
        if accounts is None or not payer.exists():
            return
        cost = usage.spent(usage.saved_prices(home / "cache"), [job.logs / "llm_usage.jsonl"],
                           job.logs / "cloud_usage.jsonl")
        reason = "label" if (job.root / LABELING).exists() else "run"
        left = accounts.meter(payer.read_text().strip(), f"job:{job.name}", reason, cost)
        charged(payer.read_text().strip(), left)
        if running and left < 0:
            jobs.cancel(job)
            with open(job.logs / "pipeline.log", "a") as log:
                log.write("FAILED: stopped, the balance ran out. Top it up and run again: done steps are kept.\n")

    def meter_loop() -> None:
        while True:
            time.sleep(METER_S)
            root = home / "jobs"
            for d in root.iterdir() if root.exists() else []:
                if (d / PAYER).exists():
                    try:
                        job = Job(d)
                        meter_job(job, jobs.running(job))
                    except Exception as e:  # one job's broken logs never stop the meter
                        print(f"meter {d.name}: {type(e).__name__}: {e}", flush=True)

    if accounts is not None:
        threading.Thread(target=meter_loop, daemon=True, name="meter").start()

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

    def interview_usage(sid: str) -> Path:
        return home / "logs" / "interviews" / f"{sid}.usage.jsonl"

    def credit_interview(job: Job, sid: str | None, request: Request) -> None:
        """The model calls of the interview that wrote this job's spec go with the job (usage panel)."""
        if not sid or not owns_upload(sid, request) or not interview_usage(sid).exists():
            return
        meter_chat(sid)  # the chat pays its calls; the job's own meter leaves them out
        job.logs.mkdir(parents=True, exist_ok=True)
        with open(job.logs / "interview_usage.jsonl", "a") as f:
            f.write(interview_usage(sid).read_text())
        interview_usage(sid).unlink()  # counted once, even if the chat goes on

    def keep_chat(job: Job, sid: str | None, request: Request, revision: bool = False) -> None:
        """The chat that wrote this job's spec, as the web last showed it, goes with the job (its page shows it)."""
        if not sid or not owns_upload(sid, request):
            return
        try:
            view = json.loads((home / "uploads" / sid / CHAT_VIEW).read_text()).get("view") or {}
        except (OSError, ValueError):
            return
        add_chat(job, view, revision)

    def start_run(job: Job, request: Path, start: str | None = None, payer: Request | None = None):
        """payer: the HTTP request of the user who pays this run (accounts mode)."""
        if payer is not None:
            bill_to(job, may_start(payer, "run"))
        jobs.start(job, request, start)

    def room_for_a_run() -> None:
        if jobs.busy():
            raise HTTPException(429, f"{max_runs} models are training right now; try again in a few minutes")

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
        return {"backend": backend, "first_question": FIRST_QUESTION, "public": public, "busy": jobs.busy(),
                "max_runs": max_runs if public else None, "budget": budget,
                "locked": auth is None and bool(key) and not getattr(request.state, "unlocked", False),
                "login": auth.providers if auth is not None else None, "engine": engine()}

    @app.get("/api/me")
    def me(request: Request):
        """Who is signed in, their tier and credits; and where judges write for free access."""
        acc = getattr(request.state, "account", None)
        judging = accounts is not None and (accounts.judge_until is None or time.time() < accounts.judge_until)
        out = {"signed_in": acc is not None, "contact": contact, "judging": judging,
               "costs": {"margin": MARGIN, "start": START}, "topup": payments is not None}
        if acc is not None:
            out["account"] = {"email": acc.email, "name": acc.name, "provider": acc.provider,
                              "tier": accounts.tier(acc), "balance": accounts.balance(acc.id)}
        return out

    if auth is not None:
        @app.get("/auth/{provider}/login", include_in_schema=False)
        def login(provider: str, next: str = "/app"):
            if provider not in auth.providers:
                raise HTTPException(404, f"no login with {provider}")
            nonce = uuid.uuid4().hex
            # A page of this site only: "//x" and "/\\x" are other sites to a browser.
            back = next if re.fullmatch(r"/(?![/\\])[^\\\s]*", next) else "/app"
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

    def owns_upload(sid: str, request: Request) -> bool:
        """Is uploads/<sid> (a chat's photos) the requester's? Kept on disk, so it holds after a restart."""
        f = home / "uploads" / sid / SESSION_OWNER
        return re.fullmatch(r"[0-9a-f]{32}", sid or "") is not None and f.exists() \
            and f.read_text().strip() == owner_of(request)

    def budget_left() -> None:
        problem = jobs.over_budget()
        if problem:
            raise HTTPException(429, problem)

    def my_jobs(request: Request) -> list[str]:
        """The names of the requester's jobs."""
        root = home / "jobs"
        return sorted(d.name for d in root.iterdir() if ((d / "spec.json").exists() or (d / "request.json").exists())
                      and mine(Job(d), request)) if root.exists() else []

    LIB_KINDS = {"datasets": "dataset", "objects": "object", "scenes": "scene", "kits": "kit"}

    def owned(request: Request, kinds: str, item: str) -> tuple[str, str]:
        kind = LIB_KINDS.get(kinds)
        owner = owner_of(request)
        if kind is None or not lib.owns(owner, kind, item):
            raise HTTPException(404, f"no such {kind or 'item'}")
        return owner, kind


    # The routes of each part of the app (sim2chip/api), registered in this order: a generic path such as
    # /api/library/{kinds}/{item} comes after the specific ones it would also match. `ctx` is what they share.
    from .api import billing as billing_api, interview as interview_api, jobs as jobs_api, library as library_api
    from .api import scenes as scenes_api, specs as specs_api
    ctx = SimpleNamespace(
        home=home, lib=lib, jobs=jobs, public=public, backend=backend, accounts=accounts, payments=payments,
        contact=contact, describer=describer, interviewer=interviewer, profiling=profiling, profiler=profiler,
        chat_payer=chat_payer, owns_upload=owns_upload, owner_of=owner_of, visitor=visitor, may_start=may_start,
        bill_to=bill_to, meter_chat=meter_chat, charged=charged, mine=mine, job_of=job_of,
        interview_usage=interview_usage, credit_interview=credit_interview, keep_chat=keep_chat,
        start_run=start_run, room_for_a_run=room_for_a_run, budget_left=budget_left, owned=owned,
        profile_soon=profile_soon, my_jobs=my_jobs)
    specs_api.helpers(ctx)
    interview_api.add(app, ctx)
    jobs_api.add(app, ctx)
    library_api.add(app, ctx)
    scenes_api.add(app, ctx)
    billing_api.add(app, ctx)

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

    app.meter_job = meter_job  # tests charge a job without waiting for the meter
    return app


def capped(upload: UploadFile, limit: int) -> bytes:
    """An uploaded file's bytes, refused (413) past `limit`: a huge file would fill the disk or the memory."""
    data = upload.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"{Path(upload.filename or 'a file').name} is over {limit >> 20} MB")
    return data


@contextlib.contextmanager
def timed(request: Request, part: str):
    """Times a part of an upload's handler for upload_timing; the first one also marks how long reading
    the form took (from the request's arrival to the handler)."""
    timing = getattr(request.state, "timing", None)
    t = time.perf_counter()
    if timing is not None and "read" not in timing:
        timing["read"] = t - timing["t0"]
    try:
        yield
    finally:
        if timing is not None:
            timing[part] = timing.get(part, 0) + time.perf_counter() - t


def save_photos(folder: Path, images: list[UploadFile]) -> list[tuple[str, str]]:
    """Writes uploaded photos under their own names; a different photo with a name already there gets a
    numbered one (phones reuse names like image.jpg). Returns (upload's name, file name it got) per photo."""
    folder.mkdir(parents=True, exist_ok=True)
    got = []
    for p in images:
        data, name = capped(p, MAX_PHOTO_BYTES), Path(p.filename or "photo.jpg").name
        dst, n = folder / name, 1
        while dst.exists() and dst.read_bytes() != data:
            n += 1
            dst = folder / f"{Path(name).stem}_{n}{Path(name).suffix}"
        dst.write_bytes(data)
        got.append((name, dst.name))
    return got


def _photo_labels(text: str) -> dict[str, dict]:
    """The live test's labels sent with photos (add_photos), kept only when well formed."""
    try:
        raw = json.loads(text) if text else {}
    except ValueError:
        raise HTTPException(400, "labels must be JSON") from None
    out = {}
    for name, v in (raw.items() if isinstance(raw, dict) else []):
        if not isinstance(v, dict):
            continue
        if isinstance(v.get("label"), str):
            out[Path(str(name)).name] = {"label": v["label"]}
        elif isinstance(v.get("points"), list):
            pts = [p for p in v["points"] if isinstance(p, list) and len(p) == 3 and isinstance(p[2], str)
                   and all(isinstance(c, (int, float)) and 0 <= c <= 1 for c in p[:2])]
            out[Path(str(name)).name] = {"points": pts}
    return out


def add_chat(job: Job, view: dict, revision: bool = False) -> None:
    """Keep a chat's messages (who and what, no photos or setup cards) in the job folder."""
    msgs = [{"from": m["from"], "text": str(m.get("text", ""))} for m in view.get("msgs") or []
            if isinstance(m, dict) and m.get("from") in ("bot", "me", "sys") and m.get("text")]
    if not msgs:
        return
    f = job.root / CHATS
    try:
        chats = json.loads(f.read_text())
    except (OSError, ValueError):
        chats = []
    chats.append({"t": round(time.time()), "revision": revision, "msgs": msgs})
    f.write_text(json.dumps(chats, ensure_ascii=False))


def latest_spec(job: Job) -> Path:
    """A relaunched job has a new request.json while spec.json is still the previous run's until the run
    starts: the newer one."""
    paths = [p for p in (job.spec_path, job.root / "request.json") if p.exists()]
    return max(paths, key=lambda p: p.stat().st_mtime)


def default_describer(home: Path):
    """photos, classes -> one description per photo with the scene model (scene_profile.describe): what the
    library profiles datasets with. The client is made on first use."""
    from . import scene_profile
    from .llm import Client
    made: dict = {}

    def describe(photos: list[Path], classes: str, usage_log: Path | None = None) -> list[dict]:
        """usage_log: where this call's model usage goes (to charge it), else the server's log."""
        from dataclasses import replace
        if "client" not in made:
            made["client"] = Client(usage_log=home / "logs" / "llm_usage.jsonl")
        client = made["client"]
        if usage_log is not None:
            usage_log.parent.mkdir(parents=True, exist_ok=True)
            client = replace(client, usage_log=usage_log)
        return scene_profile.describe(photos, classes, client, client.model_for("scene"))
    return describe


def adopt_uploads(spec: spec_mod.Spec, uploads: Path, job: Job) -> spec_mod.Spec:
    """Move chat uploads the spec points to into the job's input folder and make the paths relative, so
    the photos travel with the job (to Nebius too) and outlive the interview session."""

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
        ud[k] = [adopt(v) for v in ud[k]] if isinstance(ud.get(k), list) else adopt(ud.get(k))
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
        why = job.dropped(p.stage.name) if state == "done" else None
        if why:
            state = "skipped"
        if state == "pending" and p.stage.name == current:
            state = "running" if running else ("failed" if failed else "pending")
        stages.append({"name": p.stage.name, "runtime": p.stage.runtime, "state": state,
                       "powered_by": p.stage.powered_by(spec, backend, models.get(p.stage.name)),
                       **({"reason": why} if why else {})})
    cloud_log = job.logs / "cloud_usage.jsonl"
    cost = sum(json.loads(l)["usd_estimate"] for l in cloud_log.read_text().splitlines()) if cloud_log.exists() else 0.0
    report = job.stage_dir("export") / "report.json"
    # The model export holds, unless a run is making a new one. Not running, it shows even when a newer
    # EyeForge would redo some stages (their code changed): it is still the model the user downloads.
    fresh = not running or any(st["name"] == "export" and st["state"] == "done" for st in stages)
    calibration = job.root / CALIBRATION
    title = job.root / library.TITLE
    return {"job": job.name, "title": title.read_text().strip() if title.exists() else None,
            "spec": json.loads(spec.to_json()), "running": running,
            "calibration": json.loads(calibration.read_text()) if calibration.exists() else None,
            "labeling": json.loads((job.root / LABELING).read_text()) if (job.root / LABELING).exists() else None,
            "stages": stages, "log": log[-30:], "cloud_usd": round(cost, 2), "phase": phase(log, running),
            "timing": timing(job, home, stages, running, backend),
            "usage": usage.summary(job, usage.saved_prices(home / "cache"), config.gpu_name()),
            "version": {**current_version(job), "engine": last_engine(job)}, "versions": [{k: v for k, v in m.items() if k != "report_key"}
                                                          for m in versions(job)],
            "report": json.loads(report.read_text()) if fresh and report.exists() else None,
            "no_synthetic": no_synthetic(spec, job)}


def serve(home: Path, backend: str, host: str, port: int, max_usd: float | None, public: bool = False,
          key: str | None = None, max_runs: int = 2, daily_usd: float | None = None, auth=None,
          judge_until: float | None = None, contact: str | None = None) -> None:
    """auth: login with Google or GitHub (auth.Auth); accounts live in <home>/accounts.db."""
    import uvicorn
    from .accounts import Accounts
    # Token Factory's prices for the usage panel, in the background: the server starts without waiting.
    threading.Thread(target=usage.refresh_prices, args=(home / "cache",), daemon=True).start()
    accounts = Accounts(home / ACCOUNTS_DB, judge_until) if auth is not None else None
    from .payments import Payments
    payments = Payments.from_env(accounts, auth.public_url) if accounts is not None else None
    gpu = warm_mod.WarmGPU(home) if backend == "nebius" and warm_mod.enabled() else None
    scenes = warm_mod.WarmGPU(home, kind="isaac") if gpu is not None and warm_mod.scene_queue() else None
    for g in (gpu, scenes):
        if g is not None and warm_mod.current(home, g.kind):  # one left by the previous server process: note its end
            g.watch()
    uvicorn.run(create_app(home, backend, max_usd, public=public, key=key, max_runs=max_runs, daily_usd=daily_usd,
                           auth=auth, accounts=accounts, contact=contact, warm=gpu, scenes=scenes,
                           describer=default_describer(home), payments=payments),
                host=host, port=port, proxy_headers=True)
