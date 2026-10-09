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
import functools
import json
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from . import config, library, review, usage, warm as warm_mod
from .accounts import MARGIN, START
from .version import engine
from .job_service import CALIBRATION, CALIBRATION_PHOTOS, LABELING, JobService, cloud_spent
from . import spec as spec_mod
from .pipeline import Pipeline
from .workspace import IMAGE_EXT, Cache, Job, current_version, last_engine, open_job, snapshot, versions

DOWNLOADS = {"model.tflite", "labels.txt", "model_int8.tflite", "model_data.h", "report.json"}
WEB = Path(__file__).parent / "web"
DIST = WEB / "dist"
OWNER = "owner"  # file in a public server's job folder: the visitor who started it
PAYER = "payer"  # file in a job folder: the account paying what it uses (accounts mode)
METER_S = 30     # how often the usage of running jobs is charged
NEW_SHARE = 0.25  # new photos, at most this share of a folder the user checked before, skip the check
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
MAX_PHOTO_BYTES = 30 << 20   # one photo: a 50 MP phone photo is about 20 MB
MAX_MODEL_BYTES = 200 << 20  # one GLB the user uploads
SESSION_OWNER = ".owner"     # file in uploads/<session>: who started that chat
SESSION_HOURS = 6
# Model calls are logged by purpose, which is the stage name except for qa's two-step strategy.
PURPOSE_STAGE = {"qa_describe": "qa_assets", "qa_decide": "qa_assets"}  # an interview left open longer is dropped


def create_app(home: Path, backend: str = "local", max_usd: float | None = None, interviewer=None,
               launcher=None, public: bool = False, key: str | None = None, max_runs: int = 2,
               daily_usd: float | None = None, auth=None, accounts=None, contact: str | None = None, warm=None,
               describer=None, payments=None):
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
    from .interview import sources as interview_sources

    app = FastAPI(title="EyeForge")

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
    started: dict[str, float] = {}  # interview session -> start time
    new_session = interviewer
    jobs = JobService(home, backend, max_usd, launcher=launcher, warm=warm, public=public, max_runs=max_runs,
                      daily_usd=daily_usd)

    lib = library.Library(home, home / "cache")
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

    def adopt_library(parsed: spec_mod.Spec, owner: str, job: Job) -> spec_mod.Spec:
        """Library references of a spec ("library:<id>", from a kit or the chat) become files in the job's input
        folder, so the job travels with them (to Nebius too) and keeps them if the library item is deleted."""
        def ref(v: str | None) -> str | None:
            if isinstance(v, str) and re.fullmatch(r"[0-9a-f]{16}", v):  # the chat wrote the bare id
                return v
            return v[len("library:"):] if isinstance(v, str) and v.startswith("library:") else None

        data = json.loads(parsed.to_json())
        ud = data["user_data"]

        def dataset(v, by_label: bool):
            ds = ref(v)
            if not ds:
                return v
            if not lib.owns(owner, "dataset", ds):
                raise HTTPException(400, f"invalid spec: no dataset {ds} in your library")
            lib.materialize_dataset(owner, ds, job.input / f"dataset_{ds}", by_label=by_label)
            return f"dataset_{ds}"
        for k in ("labeled", "test"):
            ud[k] = dataset(ud.get(k), True)
        raw = [dataset(v, False) for v in interview_sources(ud.get("raw"))]
        ud["raw"] = merge_sources(job, raw) if len(raw) > 1 else (raw[0] if raw else None)
        for item in [c["assets"] for c in data["classes"] if c.get("assets")] + data["props"]:
            ids = [ref(m) for m in item.get("models") or []]
            if not any(ids):
                continue
            missing = [i for i in ids if i and not lib.owns(owner, "object", i)]
            if missing or not all(ids):
                raise HTTPException(400, f"invalid spec: models must be objects of your library ({missing})")
            item["models"] = [f"objects/{n}" for n in lib.materialize_objects(owner, ids, job.input / "objects")]
        return spec_mod.from_dict(data)

    def reuse_corrections(parsed: spec_mod.Spec, owner: str, job: Job) -> None:
        """The user's earlier corrections of these photos (other jobs, the Datasets page) as this job's, so they
        are not asked again. Never over corrections the job already has."""
        from . import teacher
        f = job.input / review.CORRECTIONS
        raw = parsed.user_data.raw
        if f.exists() or not isinstance(raw, str):
            return
        try:
            folder = job.resolve_input(raw)
        except ValueError:
            return
        if not folder.is_dir():
            return
        phrases = {c: p[0] for c, p in teacher.phrases(parsed).items()} if parsed.task == "count" else {}
        found = lib.corrections_for(owner, folder, [c.id for c in parsed.classes] if phrases else [], phrases,
                                    parsed.labels if parsed.task != "count" else [])
        if found:
            f.write_text(json.dumps({"labels": {}, "assets": {}, "points": {}, **found}, indent=1))

    def unlabeled_as_raw(parsed: spec_mod.Spec, owner: str) -> spec_mod.Spec:
        """A library dataset without labels of its own given as "labeled" (the chat mixed it up with the
        teacher's labels, 2026-10-09: the job trained on 0 real photos) is raw photos for the teacher."""
        ud = parsed.user_data
        ds = re.fullmatch(r"(?:library:)?([0-9a-f]{16})", str(ud.labeled or ""))
        if not ds or not lib.owns(owner, "dataset", ds.group(1)):
            return parsed
        if next((d["labels"] for d in lib.datasets(owner) if d["id"] == ds.group(1)), None):
            return parsed
        data = json.loads(parsed.to_json())
        raw = interview_sources(data["user_data"].get("raw"))
        data["user_data"].update(labeled=None, class_map={}, raw=[ud.labeled, *raw] if raw else ud.labeled)
        return spec_mod.from_dict(data)

    def follow_labeling(parsed: spec_mod.Spec, owner: str) -> spec_mod.Spec:
        """A library dataset labeled before for the same task and class ids (library.labelings): the classes take
        that job's names and descriptions, so the teacher looks for the very same things and its kept answers
        and labels apply (user, 2026-10-09: a dataset's labels are not asked for again)."""
        ids = {c.id for c in parsed.classes}
        for src in interview_sources(parsed.user_data.raw):
            ds = re.fullmatch(r"(?:library:)?([0-9a-f]{16})", str(src))
            if not ds or not lib.owns(owner, "dataset", ds.group(1)):
                continue
            for done in lib.labelings(ds.group(1)):
                if done["task"] != parsed.task or set(done["classes"]) != ids \
                        or done["background"] != parsed.background_class:
                    continue
                before = {c["id"]: c for c in json.loads((home / "jobs" / done["job"] / "request.json").read_text())
                          ["classes"]}
                data = json.loads(parsed.to_json())
                for c in data["classes"]:
                    c["description"] = before[c["id"]].get("description") or c["description"]
                    c["names"] = before[c["id"]].get("names") or c.get("names")
                    if c.get("assets") and before[c["id"]].get("assets"):  # names come from the queries too
                        c["assets"]["queries"] = before[c["id"]]["assets"].get("queries") or c["assets"].get("queries")
                        c["assets"]["categories"] = before[c["id"]]["assets"].get("categories") or \
                            c["assets"].get("categories")
                return spec_mod.from_dict(data)
        return parsed

    def checked_before(parsed: spec_mod.Spec, owner: str, job: Job) -> bool:
        """Are most of these photos labeled already, so the few new ones need no check of their own (user,
        2026-10-09)? Either the user corrected at least a check round's worth (reuse_corrections), or an earlier
        job's teacher labeled them for these same classes and the user went on to train with those labels. The
        new ones, at most NEW_SHARE of the folder, are marked in the run, the teacher learning from the earlier
        corrections, and the user can still fix them after it."""
        from . import teacher
        raw = parsed.user_data.raw
        if not isinstance(raw, str):
            return False
        try:
            folder = job.resolve_input(raw)
        except ValueError:
            return False
        if not folder.is_dir():
            return False
        total = sum(1 for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXT and not p.name.startswith("."))
        f = job.input / review.CORRECTIONS
        fixes = json.loads(f.read_text()) if f.exists() else {}
        corrected = len(set(fixes.get("labels") or {}) | set(fixes.get("points") or {}))
        if corrected >= CALIBRATION_PHOTOS and total - corrected <= NEW_SHARE * total:
            return True
        phrases = [p[0] for p in teacher.phrases(parsed).values()] if parsed.task == "count" else []
        known = lib.known_labels(owner, folder, phrases, parsed.labels if parsed.task != "count" else [])
        return known > 0 and total - known <= NEW_SHARE * total

    def check_paths(parsed: spec_mod.Spec, request: Request, job: Job | None = None) -> None:
        """On a public server, only photos the requester uploaded in their chats (or, revising a job, the job's
        own folders): a path would let a spec read, and upload to the cloud, any file of the server, and another
        user's chat folder their photos."""
        if not public:
            return
        uploads = (home / "uploads").resolve()

        def ok(p: str) -> bool:
            if p == "photos":
                return True
            if Path(p).is_absolute() and Path(p).resolve().is_relative_to(uploads):
                rel = Path(p).resolve().relative_to(uploads).parts
                return bool(rel) and owns_upload(rel[0], request)
            if re.fullmatch(r"(library:)?[0-9a-f]{16}", p):  # the owner's library, checked when adopted
                return True
            return job is not None and not Path(p).is_absolute() and ".." not in Path(p).parts \
                and (job.input / p).exists() and not (job.input / p).is_symlink()
        bad = [p for p in spec_paths(parsed) if not ok(p)]
        if bad:
            raise HTTPException(400, f"invalid spec: only photos uploaded here can be used, not {bad}")

    @app.post("/api/interview")
    def interview_start(request: Request, job: str | None = None, kit: str | None = None):
        """A new interview, or (job=<name>) one that revises a trained job's spec; kit=<id>: a new model that
        starts from that kit of the user's library."""
        revising = job_of(job, request) if job else None
        acc = may_start(request, "chat")
        now = time.time()
        for old in [k for k, t in started.items() if now - t > SESSION_HOURS * 3600]:  # abandoned chats
            sessions.pop(old, None), started.pop(old, None), locks.pop(old, None)
        sid = uuid.uuid4().hex
        (home / "uploads" / sid).mkdir(parents=True, exist_ok=True)
        (home / "uploads" / sid / SESSION_OWNER).write_text(owner_of(request))
        open_session(sid, request, acc, kit)
        if revising:
            turn = sessions[sid].revise(spec_mod.load(latest_spec(revising)))
            return {"session": sid, "text": turn.text, "job": revising.name}
        return {"session": sid, "text": FIRST_QUESTION}

    def open_session(sid: str, request: Request, acc, kit: str | None = None) -> None:
        """The interview object of a chat, with its uploads folder, transcript, payer and library."""
        nonlocal new_session
        if new_session is None:
            new_session = default_interviewer(home, backend, public)
        started[sid] = time.time()
        sessions[sid] = new_session()
        if acc is not None:
            chat_payer[sid] = acc.id
        sessions[sid].uploads = home / "uploads" / sid
        sessions[sid].transcript = home / "logs" / "interviews" / f"{sid}.json"
        if hasattr(getattr(sessions[sid], "client", None), "usage_log"):
            sessions[sid].client.usage_log = interview_usage(sid)
        if hasattr(sessions[sid], "use_library"):  # the user's kits and datasets, to start from them
            owner = owner_of(request)
            chosen = next((k["name"] for k in lib.kits(owner) if k["id"] == kit), None) if kit else None
            folders = home / "uploads" / sid

            def open_dataset(ds: str, owner=owner, folders=folders) -> Path | None:
                if not lib.owns(owner, "dataset", ds):
                    return None
                jobs.warm_up("library photos", f"session:{sid}")  # as an upload: the GPU starts while the chat goes on
                lib.materialize_dataset(owner, ds, folders / f"library_{ds}", by_label=False)
                return folders / f"library_{ds}"
            sessions[sid].use_library(*lib.for_chat(owner), kit=chosen,
                                      notes=functools.partial(lib.describe, owner), open_dataset=open_dataset,
                                      labeled={d["id"] for d in lib.datasets(owner) if d["labels"]})

    locks: dict[str, threading.Lock] = {}

    def chat_of(sid: str, request: Request):
        """The interview session, only for who started it (its id alone is not enough)."""
        if not owns_upload(sid, request):
            raise HTTPException(404, "unknown interview session")
        session = sessions.get(sid)
        saved = home / "logs" / "interviews" / f"{sid}.json"
        if session is None and saved.exists() and not ended(sid).exists():
            # The server restarted (a deploy) while this chat was open: it goes on from its transcript.
            open_session(sid, request, getattr(request.state, "account", None))
            if hasattr(sessions[sid], "restore"):
                sessions[sid].restore(json.loads(saved.read_text()))
                session = sessions[sid]
            else:
                sessions.pop(sid, None)
        if session is None:
            raise HTTPException(404, "unknown interview session")
        return session

    def ended(sid: str) -> Path:
        return home / "uploads" / sid / ".ended"

    def owns_upload(sid: str, request: Request) -> bool:
        """Is uploads/<sid> (a chat's photos) the requester's? Kept on disk, so it holds after a restart."""
        f = home / "uploads" / sid / SESSION_OWNER
        return re.fullmatch(r"[0-9a-f]{32}", sid or "") is not None and f.exists() \
            and f.read_text().strip() == owner_of(request)

    def guarded(sid: str, fn, *args, act: str | None = None):
        """One call at a time per interview (a double submit would interleave two model turns), and the
        right status: 502 when the model service fails, 409 when the interview is already over.
        act: the front end's id for this turn, under which its steps are shown (see interview_activity)."""
        from .llm import LLMError
        with locks.setdefault(sid, threading.Lock()):
            session = sessions.get(sid)
            if accounts is not None and sid in chat_payer and accounts.balance(chat_payer[sid]) <= 0:
                acc = accounts.get(chat_payer[sid])
                if acc is not None and accounts.tier(acc) == "normal":
                    raise HTTPException(402, "your balance ran out; top it up to go on with this chat")
            if hasattr(session, "begin"):
                session.begin(act)
            try:
                return fn(*args)
            except LLMError as e:
                raise HTTPException(502, f"model service: {e}") from None
            except RuntimeError as e:
                raise HTTPException(409, str(e)) from None
            finally:
                meter_chat(sid)

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
            ended(sid).touch()  # over: never brought back from its transcript
        return out

    @app.post("/api/interview/{sid}")
    def interview_reply(sid: str, body: dict, request: Request, act: str | None = None):
        session = chat_of(sid, request)
        return turn_out(sid, guarded(sid, session.reply, str(body.get("text", "")), act=act))

    @app.get("/api/interview/{sid}/activity")
    def interview_activity(sid: str, act: str, request: Request):
        """What the turn `act` is doing right now: its steps so far (thinking, searching, checking the setup),
        polled by the web while it waits. No lock: it must answer while the turn holds it. `now` lets the
        client time running steps with the server's clock."""
        session = chat_of(sid, request)
        steps = list(getattr(session, "activity", [])) if getattr(session, "activity_id", None) == act else []
        return {"steps": [dict(s) for s in steps], "now": time.time()}

    @app.post("/api/interview/{sid}/photos")
    def interview_photos(sid: str, request: Request, purpose: str = Form("training"), cls: str = Form(""),
                               photos: list[UploadFile] = File(...), act: str | None = None,
                               more: bool = Form(False), total: int = Form(0)):
        """Photos uploaded in the chat, asked for by the interview or not. They wait in uploads/<session>
        until a job adopts them. Sync on purpose: the model calls after the upload block, and FastAPI runs
        sync handlers in a thread pool instead of the event loop (async stalled every other request).
        A big upload comes in batches (a proxy caps the request size): more=true only stores the batch, and
        the last one, with the total of all batches, makes the interview's turn."""
        session = chat_of(sid, request)
        purpose = "3d" if purpose == "3d" else "training"
        folder = home / "uploads" / sid / (f"3d_{slug(cls)}" if purpose == "3d" and cls else purpose)
        folder.mkdir(parents=True, exist_ok=True)
        saved = []
        for p in photos:
            if Path(p.filename or "").suffix.lower() in IMAGE_EXT:
                (folder / Path(p.filename).name).write_bytes(capped(p, MAX_PHOTO_BYTES))
                saved.append(folder / Path(p.filename).name)
        n = len(saved)
        # Every real photo belongs to a dataset: the chat's own, made with its first photos (user, 2026-10-08).
        lib.add_chat_photos(owner_of(request), sid, saved, "3d" if purpose == "3d" else "train", cls or None)
        if not n and not total:
            raise HTTPException(400, "no images among the files")
        if purpose == "training":  # the GPU starts while the interview goes on
            jobs.warm_up("photos uploaded", f"session:{sid}")
        if more:
            return {"stored": n}
        n = max(n, total)
        return {**turn_out(sid, guarded(sid, session.add_photos, str(folder), n, purpose, cls or None,
                                         act=act)), "photos": n}

    @app.get("/api/interview/{sid}/thumb")
    def interview_thumb(sid: str, file: str, request: Request):
        """Thumbnail of a photo the interview is proposing (only those: the path comes from the user)."""
        session = chat_of(sid, request)
        if file not in {c["file"] for c in session.candidates}:
            raise HTTPException(404, "not a proposed photo")
        return FileResponse(photo_thumb(home / "uploads" / sid / "_thumbs", Path(file)))

    @app.post("/api/interview/{sid}/pick")
    def interview_pick(sid: str, body: dict, request: Request, act: str | None = None):
        session = chat_of(sid, request)
        return turn_out(sid, guarded(sid, session.choose, [str(f) for f in body.get("files") or []],
                                     str(body.get("class", "")), act=act))

    def budget_left() -> None:
        problem = jobs.over_budget()
        if problem:
            raise HTTPException(429, problem)

    @app.post("/api/jobs")
    def create_job(request: Request, spec: str = Form(...), photos: list[UploadFile] = File(default=[]),
                   calibrate_first: bool = Form(False, alias="calibrate"), session: str | None = Form(None)):
        """A new job. calibrate: when the teacher labels the user's raw photos, it first marks a few for the user
        to check (see calibrate); the run starts from the job page."""
        budget_left()
        if public:
            room_for_a_run()
            if len(photos) > MAX_PHOTOS:
                raise HTTPException(413, f"at most {MAX_PHOTOS} photos per run")
        try:
            data = json.loads(spec)
            images = [p for p in photos if Path(p.filename or "").suffix.lower() in IMAGE_EXT]
            if images and not (data.get("user_data") or {}).get("labeled"):
                ud = data.setdefault("user_data", {})  # the teacher model labels them, with any other raw photos
                ud["raw"] = [*interview_sources(ud.get("raw")), "photos"] if ud.get("raw") not in (None, "photos") \
                    else "photos"
            parsed = spec_mod.from_dict(data)
        except (json.JSONDecodeError, spec_mod.SpecError) as e:
            raise HTTPException(400, f"invalid spec: {e}") from None
        check_paths(parsed, request)
        # The same setup sent twice (a double click, "Looks good" and Train together) must not pay for two runs.
        same = json.loads(parsed.to_json())
        for other, proc in jobs.runs.items():
            req = home / "jobs" / other / "request.json"
            if proc.poll() is None and req.exists() and {**json.loads(req.read_text()), "name": same["name"]} == same \
                    and mine(Job(home / "jobs" / other), request):
                return {"job": other, "photos": 0, "already_running": True}
        may_start(request, "run")  # before anything is written
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
        save_photos(job.input / "photos", images)  # before the library: raw may merge them with a dataset
        try:
            parsed = adopt_library(follow_labeling(unlabeled_as_raw(parsed, owner_of(request)), owner_of(request)),
                                   owner_of(request), job)
        except HTTPException:
            shutil.rmtree(job.root, ignore_errors=True)  # a refused spec leaves no empty job behind
            raise
        reuse_corrections(parsed, owner_of(request), job)
        req = job.root / "request.json"
        req.write_text(json.dumps({**json.loads(parsed.to_json()), "name": name}, indent=2))
        credit_interview(job, session, request)
        if session:
            warm_mod.note(home, "job", name, session=f"session:{session}")
        if calibrate_first and parsed.user_data.raw and not parsed.user_data.labeled \
                and not checked_before(parsed, owner_of(request), job):
            bill_to(job, may_start(request, "run"))  # the teacher's check is the start of the paid run
            jobs.warm_up("calibration", name)
            jobs.calibrate(job)
            return {"job": name, "photos": len(images), "calibrating": True}
        start_run(job, req, payer=request)
        return {"job": name, "photos": len(images)}

    @app.get("/api/jobs")
    def list_jobs(request: Request):
        root = home / "jobs"
        names = sorted(d.name for d in root.iterdir() if ((d / "spec.json").exists() or (d / "request.json").exists())
                       and mine(Job(d), request)) if root.exists() else []
        return {"jobs": names}

    @app.get("/api/library")
    def library_data(request: Request):
        """Everything the visitor owns: models (jobs), datasets of real photos, 3D objects, scenes and kits.
        Their jobs are read into the library first (only the new or changed ones)."""
        mine_jobs = [Job(home / "jobs" / n) for n in list_jobs(request)["jobs"]]
        owner = owner_of(request)
        lib.sync(owner, mine_jobs)
        datasets = lib.datasets(owner)
        for d in datasets:
            if lib.profile_stale(owner, d["id"]):
                profile_soon(owner, d["id"])
            d["profiling"] = d["id"] in profiling
        return {"models": [library.model_summary(j, jobs.running(j)) for j in mine_jobs
                           if not (j.root / LABELING).exists()],
                "datasets": datasets, "objects": lib.objects(owner), "scenes": lib.scenes(owner),
                "kits": lib.kits(owner)}

    @app.get("/api/library/photos/{key}")
    def library_photo(key: str, request: Request, size: int = 360):
        src = lib.photo_file(owner_of(request), key) if re.fullmatch(r"[0-9a-f]{16}", key) else None
        if src is None or not src.exists():
            raise HTTPException(404, "no such photo")
        return FileResponse(photo_thumb(lib.root / "_thumbs", src, Path(f"{key}.{size}.jpg"), 1024 if size > 360 else 360))

    @app.get("/api/library/{kind}/{item}/card")
    def library_card(kind: str, item: str, request: Request):
        src = lib.card_file(owner_of(request), kind, item) if kind in ("object", "scene") else None
        if src is None or not src.exists():
            raise HTTPException(404, "no picture")
        return FileResponse(src)

    LIB_KINDS = {"datasets": "dataset", "objects": "object", "scenes": "scene", "kits": "kit"}

    def owned(request: Request, kinds: str, item: str) -> tuple[str, str]:
        kind = LIB_KINDS.get(kinds)
        owner = owner_of(request)
        if kind is None or not lib.owns(owner, kind, item):
            raise HTTPException(404, f"no such {kind or 'item'}")
        return owner, kind

    def incoming(files: list[UploadFile], ext: set[str]) -> tuple[Path, list[Path]]:
        """Uploaded files into a fresh temporary folder (the library links or copies them from there)."""
        import tempfile
        (lib.root / "_incoming").mkdir(exist_ok=True)
        tmp = Path(tempfile.mkdtemp(dir=lib.root / "_incoming"))
        out = []
        for i, f in enumerate(files):
            suffix = Path(f.filename or "").suffix.lower()
            if suffix in ext:
                out.append(tmp / f"{i:05d}{suffix}")
                out[-1].write_bytes(capped(f, MAX_MODEL_BYTES if suffix == ".glb" else MAX_PHOTO_BYTES))
        return tmp, out

    @app.post("/api/library/datasets")
    def library_new_dataset(request: Request, name: str = Form(...), photos: list[UploadFile] = File(...)):
        """Photos uploaded on the Datasets page become a new dataset (more come with .../photos)."""
        import shutil
        if public and len(photos) > MAX_PHOTOS:
            raise HTTPException(413, f"at most {MAX_PHOTOS} photos per upload")
        tmp, files = incoming(photos, IMAGE_EXT)
        try:
            if not files:
                raise HTTPException(400, "no images among the files")
            return {"id": lib.new_dataset(owner_of(request), name.strip()[:80] or "My photos", files), "photos": len(files)}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @app.post("/api/library/datasets/{item}/photos")
    def library_add_photos(item: str, request: Request, photos: list[UploadFile] = File(...)):
        import shutil
        owner, _ = owned(request, "datasets", item)
        if public and len(photos) > MAX_PHOTOS:
            raise HTTPException(413, f"at most {MAX_PHOTOS} photos per upload")
        tmp, files = incoming(photos, IMAGE_EXT)
        try:
            return {"photos": lib.add_photos(owner, item, files)}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @app.post("/api/library/datasets/{item}/profile")
    def library_profile(item: str, request: Request):
        """Look at the dataset's photos again (new ones, or the last try failed)."""
        owner, _ = owned(request, "datasets", item)
        if describer is None:
            raise HTTPException(503, "no vision model on this server")
        acc = may_start(request, "profile")
        profile_soon(owner, item, acc.id if acc is not None else None)
        return {"profiling": True}

    @app.post("/api/library/datasets/{item}/label")
    def library_label(item: str, body: dict, request: Request):
        """Label a dataset without training a model: the teacher marks a few photos for the user to check, as
        before a first run, then every photo (label_all). Count: boxes of each thing named; classify: one of
        the names per photo. The marks and corrections stay with the photos, for any later model."""
        owner, _ = owned(request, "datasets", item)
        task = body.get("task")
        names = list(dict.fromkeys(str(n).strip()[:60] for n in body.get("classes") or [] if str(n).strip()))
        if task not in ("count", "classify") or not names or len(names) > 8:
            raise HTTPException(400, "say what to label: count or classify, and 1 to 8 names")
        if task == "classify" and len(names) < 2:
            raise HTTPException(400, "classify needs at least two names to choose among")
        ds_name = next((d["name"] for d in lib.datasets(owner) if d["id"] == item), "photos")
        classes = []
        for n in names:
            cid = re.sub(r"[^a-z0-9]+", "_", n.lower()).strip("_") or f"class_{len(classes)}"
            if cid not in [c["id"] for c in classes]:
                classes.append({"id": cid, "description": n})
        data = {"name": f"label {ds_name}", "task": "count" if task == "count" else "roi_classification",
                "classes": classes, "user_data": {"raw": f"library:{item}"}}
        try:
            parsed = spec_mod.from_dict(data)
        except spec_mod.SpecError as e:
            raise HTTPException(400, f"invalid labels: {e}") from None
        budget_left()
        base = name = slug(parsed.name)
        n = 1
        while (home / "jobs" / name).exists():
            n += 1
            name = f"{base}_{n}"
        acc = may_start(request, "label")
        job = open_job(home, name)
        bill_to(job, acc)
        if public:
            (job.root / OWNER).write_text(visitor(request))
        parsed = adopt_library(follow_labeling(parsed, owner), owner, job)
        reuse_corrections(parsed, owner, job)
        (job.root / "request.json").write_text(json.dumps({**json.loads(parsed.to_json()), "name": name}, indent=2))
        (job.root / LABELING).write_text(json.dumps({"dataset": item, "state": "checking"}))
        jobs.warm_up("calibration", name)
        if checked_before(parsed, owner, job):
            jobs.label_all(job)
        else:
            jobs.calibrate(job)
        return {"job": name}

    @app.get("/api/library/datasets/{item}/notes")
    def library_notes(item: str, request: Request):
        """What is known about each photo of the dataset: descriptions, the teacher's boxes, the user's marks
        and labels."""
        owner, _ = owned(request, "datasets", item)
        keys = [k for (k,) in lib.db.execute("SELECT DISTINCT key FROM photos WHERE dataset = ? ORDER BY key", (item,))]
        return {"keys": keys, "notes": lib.dataset_notes(owner, item), "profile": lib.profile(owner, item),
                "profiling": item in profiling}

    @app.post("/api/library/objects")
    def library_new_object(request: Request, name: str = Form(...), concept: str = Form(""),
                           model: UploadFile = File(...), card: UploadFile | None = File(None)):
        """A GLB the owner uploads; card: the picture the app rendered of it."""
        import shutil
        tmp, files = incoming([model], {".glb"})
        try:
            if not files:
                raise HTTPException(400, "send a .glb file")
            pic = None
            if card is not None:
                pic = tmp / "card.png"
                pic.write_bytes(capped(card, MAX_PHOTO_BYTES))
            name = name.strip()[:80] or "My object"
            return {"id": lib.new_object(owner_of(request), name, concept.strip()[:80] or name, files[0], pic)}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @app.patch("/api/library/{kinds}/{item}")
    async def library_rename(kinds: str, item: str, request: Request):
        owner, kind = owned(request, kinds, item)
        name = str((await request.json()).get("name", "")).strip()[:80]
        if not name:
            raise HTTPException(400, "a name is needed")
        lib.rename(owner, kind, item, name)
        return {"ok": True}

    @app.delete("/api/library/{kinds}/{item}")
    def library_delete(kinds: str, item: str, request: Request):
        owner, kind = owned(request, kinds, item)
        return {"kits": lib.delete(owner, kind, item)}

    @app.post("/api/library/kits")
    async def library_new_kit(request: Request):
        body = await request.json()
        name = str(body.get("name", "")).strip()[:80]
        if not name:
            raise HTTPException(400, "a kit needs a name")
        return {"id": lib.save_kit(owner_of(request), None, name, body)}

    @app.put("/api/library/kits/{item}")
    async def library_save_kit(item: str, request: Request):
        owner, _ = owned(request, "kits", item)
        body = await request.json()
        name = str(body.get("name", "")).strip()[:80]
        if not name:
            raise HTTPException(400, "a kit needs a name")
        return {"id": lib.save_kit(owner, item, name, body)}

    @app.get("/api/billing")
    def billing(request: Request):
        """Credits and their moves (a server with accounts); prices of a run and an interview."""
        acc = getattr(request.state, "account", None)
        out = {"enabled": accounts is not None, "costs": {"margin": MARGIN, "start": START}, "topup": None}
        if acc is not None:
            out.update(tier=accounts.tier(acc), balance=accounts.balance(acc.id), moves=accounts.moves(acc.id))
            if payments is not None and accounts.tier(acc) == "normal":
                out["topup"] = payments.settings(acc.id)
        return out

    def paying(request: Request):
        acc = getattr(request.state, "account", None)
        if payments is None or acc is None:
            raise HTTPException(404, "top-ups are not available on this server")
        return acc

    @app.post("/api/billing/topup")
    def topup(body: dict, request: Request):
        """A Stripe Checkout page that adds the chosen dollars once paid."""
        from .payments import PaymentError
        acc = paying(request)
        try:
            return {"url": payments.checkout(acc, float(body.get("usd") or 0))}
        except (PaymentError, ValueError) as e:
            raise HTTPException(400, str(e)) from None

    @app.put("/api/billing/auto")
    def auto_topup(body: dict, request: Request):
        from .payments import PaymentError
        acc = paying(request)
        try:
            return payments.set_auto(acc.id, bool(body.get("on")), float(body.get("below", 2)),
                                     float(body.get("amount", 10)))
        except (PaymentError, ValueError) as e:
            raise HTTPException(400, str(e)) from None

    @app.post("/api/stripe/webhook", include_in_schema=False)
    async def stripe_webhook(request: Request):
        """Stripe telling us a payment went through: signed, each payment counted once."""
        from .payments import PaymentError
        if payments is None:
            raise HTTPException(404, "no payments here")
        payload = await request.body()
        try:
            event = payments.verify(payload, request.headers.get("stripe-signature"))
        except PaymentError as e:
            raise HTTPException(400, str(e)) from None
        done = await asyncio.to_thread(payments.handle, event)
        print(f"stripe {event.get('type')} {event.get('id')}: {done}", flush=True)
        return {"ok": True}

    @app.post("/api/account/delete")
    def delete_account(body: dict, request: Request):
        """Close the signed-in account (privacy policy and terms): refund what is left of its paid balance, then
        delete its jobs (and their buckets in the cloud), chats, library and Stripe customer, and remove its name
        and email. Money moves stay, under an anonymous id, as tax records."""
        from .payments import PaymentError
        acc = getattr(request.state, "account", None)
        if accounts is None or acc is None:
            raise HTTPException(401, "sign in first")
        if body.get("confirm") != "delete":
            raise HTTPException(400, 'type "delete" to confirm')
        if accounts.tier(acc) == "banned":
            raise HTTPException(403, f"write to {contact} to close this account")
        root = home / "jobs"
        own = [Job(d) for d in sorted(root.iterdir()) if (d / OWNER).is_file()
               and (d / OWNER).read_text().strip() == acc.id] if root.exists() else []
        if any(jobs.running(j) for j in own):
            raise HTTPException(409, "a model is still running: cancel it or wait until it ends")
        refunded = 0.0
        if payments is not None:
            try:
                refunded = payments.close(acc.id)
            except PaymentError as e:
                raise HTTPException(502, f"could not refund your balance, nothing was deleted: {e}") from None
        in_cloud = [j for j in own if (j.root / "cloud_id").exists() or (j.logs / "cloud_usage.jsonl").exists()]
        if backend == "nebius" and in_cloud:
            from .backends.nebius import NebiusBackend
            be = NebiusBackend()
            for j in in_cloud:
                try:
                    be.delete_storage(j)
                except Exception as e:  # left for a hand cleanup: the log says which
                    print(f"account {acc.id}: could not delete the cloud storage of {j.name}: {e}", flush=True)
        for j in own:
            shutil.rmtree(j.root, ignore_errors=True)
        uploads = home / "uploads"
        for d in uploads.iterdir() if uploads.exists() else []:
            f = d / SESSION_OWNER
            if f.is_file() and f.read_text().strip() == acc.id:
                shutil.rmtree(d, ignore_errors=True)
                (home / "logs" / "interviews" / f"{d.name}.json").unlink(missing_ok=True)
        lib.forget(acc.id, [j.name for j in own])
        accounts.forget(acc.id)
        print(f"account closed: {len(own)} jobs, ${refunded:.2f} refunded", flush=True)
        response = JSONResponse({"ok": True, "refunded": refunded})
        response.delete_cookie(SESSION)
        return response

    @app.get("/api/jobs/{name}")
    def job_status(name: str, request: Request):
        job = job_of(name, request)
        return status(job, home, running=jobs.running(job), backend=backend)

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

    @app.post("/api/jobs/{name}/cancel")
    def cancel(name: str, request: Request):
        """Stops the run: the local process and, on Nebius, the cloud job (which bills until it ends)."""
        cj = jobs.cancel(job_of(name, request))
        return {"cancelled": True, "cloud_job": cj["id"] if cj else None}

    @app.post("/api/jobs/{name}/retry")
    def retry(name: str, request: Request):
        """Run a stopped or failed job again: the steps that finished are reused, it goes on from where it stopped
        (after a fix on our side, or a cloud hiccup)."""
        job = job_of(name, request)
        if jobs.running(job):
            raise HTTPException(409, ALREADY_RUNNING)
        if (job.root / CALIBRATION).exists():
            raise HTTPException(409, "check the teacher's marks first")
        budget_left()
        room_for_a_run()
        start_run(job, latest_spec(job), payer=request)
        return {"rerun_from": None}

    @app.post("/api/jobs/{name}/train")
    def train_calibrated(name: str, body: dict, request: Request):
        """The user is happy with the teacher: keep the last fixes and start the first run. Its label stage
        learns from all the fixes and marks every photo."""
        job = job_of(name, request)
        if jobs.running(job):
            raise HTTPException(409, ALREADY_RUNNING)
        if not (job.root / CALIBRATION).exists():
            raise HTTPException(409, CHECK_OVER)
        budget_left()
        room_for_a_run()
        may_start(request, "run")
        review.apply(job, Cache(home / "cache"), body.get("labels") or {}, {}, body.get("points") or {},
                     share=not public)
        (job.root / CALIBRATION).unlink()
        start_run(job, job.root / "request.json")
        return {"rerun_from": None}

    @app.post("/api/jobs/{name}/label_all")
    def label_all(name: str, body: dict, request: Request):
        """A labeling job: keep the last fixes of the check and let the teacher mark every photo."""
        job = job_of(name, request)
        if jobs.running(job):
            raise HTTPException(409, ALREADY_RUNNING)
        if not (job.root / LABELING).exists() or not (job.root / CALIBRATION).exists():
            raise HTTPException(409, CHECK_OVER)
        may_start(request, "label")
        review.apply(job, Cache(home / "cache"), body.get("labels") or {}, {}, body.get("points") or {},
                     share=not public)
        (job.root / CALIBRATION).unlink()
        jobs.label_all(job)
        return {"rerun_from": "label"}

    @app.post("/api/jobs/{name}/corrections")
    def corrections(name: str, body: dict, request: Request):
        job = job_of(name, request)
        if jobs.running(job):
            raise HTTPException(409, ALREADY_RUNNING)
        # Fixes made on the teacher's check, sent after training started (from another device, or an old tab):
        # as fixes of a trained model they would start a run nobody asked for.
        if body.get("calibrating") and not (job.root / CALIBRATION).exists():
            raise HTTPException(409, CHECK_OVER)
        if (job.root / CALIBRATION).exists():  # still calibrating the teacher: another round, no training
            review.apply(job, Cache(home / "cache"), body.get("labels") or {}, {}, body.get("points") or {},
                         share=not public)
            jobs.calibrate(job)
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
        if jobs.running(job):
            raise HTTPException(409, f"job {job.name} is running")
        try:
            parsed = spec_mod.from_dict(json.loads(spec))
        except (json.JSONDecodeError, spec_mod.SpecError) as e:
            raise HTTPException(400, f"invalid spec: {e}") from None
        check_paths(parsed, request, job)
        old = spec_mod.load(latest_spec(job))
        changed = [k for k, v in json.loads(parsed.to_json()).items()
                   if k != "name" and json.loads(old.to_json()).get(k) != v]
        if not dry:
            may_start(request, "run")
            parsed = adopt_uploads(parsed, home / "uploads", job)
            parsed = adopt_library(follow_labeling(unlabeled_as_raw(parsed, owner_of(request)), owner_of(request)),
                                   owner_of(request), job)
            reuse_corrections(parsed, owner_of(request), job)
            credit_interview(job, session, request)
        todo = [p.stage.name for p in Pipeline(parsed, job, Cache(home / "cache")).plan() if p.action == "run"]
        if dry or not todo:
            return {"stages": todo, "changed": changed, "rerun_from": None}
        budget_left()
        room_for_a_run()
        req = job.root / "request.json"
        req.write_text(json.dumps({**json.loads(parsed.to_json()), "name": old.name}, indent=2))
        snapshot(job, "changed " + (", ".join(changed) or "the setup"))
        start_run(job, req, payer=request)
        return {"stages": todo, "changed": changed, "rerun_from": todo[0]}

    @app.post("/api/jobs/{name}/photos")
    def add_photos(name: str, request: Request, photos: list[UploadFile] = File(...), note: str = Form(""),
                   more: bool = Form(False), total: int = Form(0), labels: str = Form("")):
        """More photos for a trained job, usually of what the model gets wrong. The teacher marks them, half
        go to test (the rest to training), and only the real-photo branch reruns: the synthetic data stays.
        In batches like the interview's photos: more=true stores, the last batch (total: all of them) reruns.
        labels: JSON {photo file name: {"label": class} or {"points": [[x, y, class], ...]}}, what the user said
        each photo is when they took it in the live test; it wins over the teacher's, as review corrections do."""
        job = job_of(name, request)
        if jobs.running(job):
            raise HTTPException(409, f"job {job.name} is running")
        budget_left()
        room_for_a_run()
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
        names = save_photos(folder / "added", images)
        said = _photo_labels(labels)
        fixed = {f"added/{names[n]}": v for n, v in said.items() if n in names}
        if fixed:  # keys as the label stage names photos: their path inside the raw folder
            review.apply(job, Cache(home / "cache"), {k: v["label"] for k, v in fixed.items() if "label" in v}, {},
                         {k: v["points"] for k, v in fixed.items() if "points" in v}, share=False)
        if more:
            return {"stored": len(names)}
        saved = max(len(names), total)
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

    app.meter_job = meter_job  # tests charge a job without waiting for the meter
    return app


def capped(upload: UploadFile, limit: int) -> bytes:
    """An uploaded file's bytes, refused (413) past `limit`: a huge file would fill the disk or the memory."""
    data = upload.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"{Path(upload.filename or 'a file').name} is over {limit >> 20} MB")
    return data


def save_photos(folder: Path, images: list[UploadFile]) -> dict[str, str]:
    """Writes uploaded photos under their own names; a different photo with a name already there gets a
    numbered one (phones reuse names like image.jpg). Returns each upload's name -> the file name it got."""
    folder.mkdir(parents=True, exist_ok=True)
    got = {}
    for p in images:
        data, name = capped(p, MAX_PHOTO_BYTES), Path(p.filename or "photo.jpg").name
        dst, n = folder / name, 1
        while dst.exists() and dst.read_bytes() != data:
            n += 1
            dst = folder / f"{Path(name).stem}_{n}{Path(name).suffix}"
        dst.write_bytes(data)
        got[name] = dst.name
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


def merge_sources(job: Job, sources: list[str]) -> str:
    """Several raw photo sources of a job (folders of its input) as one folder, input/raw, one subfolder each
    (hard links: no extra space). The same photo in two sources is kept once."""
    import os
    import shutil

    from .workspace import content_key
    out = job.input / "raw"
    seen = set()
    for i, src in enumerate(sources):
        base = job.resolve_input(src)
        for f in sorted(base.rglob("*")):
            if f.suffix.lower() not in IMAGE_EXT or (k := content_key(f)) in seen:
                continue
            seen.add(k)
            dst = out / f"{i}_{Path(src).name}" / f.relative_to(base)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                try:
                    os.link(f, dst)
                except OSError:
                    shutil.copy2(f, dst)
    return "raw"


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
    fresh = any(st["name"] == "export" and st["state"] == "done" for st in stages)  # not a stale model
    calibration = job.root / CALIBRATION
    return {"job": job.name, "spec": json.loads(spec.to_json()), "running": running,
            "calibration": json.loads(calibration.read_text()) if calibration.exists() else None,
            "labeling": json.loads((job.root / LABELING).read_text()) if (job.root / LABELING).exists() else None,
            "stages": stages, "log": log[-30:], "cloud_usd": round(cost, 2), "phase": phase(log, running),
            "timing": timing(job, home, stages, running, backend),
            "usage": usage.summary(job, usage.saved_prices(home / "cache"), config.gpu_name()),
            "version": {**current_version(job), "engine": last_engine(job)}, "versions": [{k: v for k, v in m.items() if k != "report_key"}
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
    from .payments import Payments
    payments = Payments.from_env(accounts, auth.public_url) if accounts is not None else None
    gpu = warm_mod.WarmGPU(home) if backend == "nebius" and warm_mod.enabled() else None
    if gpu is not None and warm_mod.current(home):  # one left by the previous server process: note its end
        gpu.watch()
    uvicorn.run(create_app(home, backend, max_usd, public=public, key=key, max_runs=max_runs, daily_usd=daily_usd,
                           auth=auth, accounts=accounts, contact=contact, warm=gpu,
                           describer=default_describer(home), payments=payments),
                host=host, port=port, proxy_headers=True)
