"""Web API and a minimal UI: interview, photo upload, run, progress and download.

    sim2chip serve [--backend local|nebius] [--host 127.0.0.1] [--port 8000]

Each run is a `sim2chip run` subprocess writing to the job's logs/pipeline.log, so the server never
blocks on the pipeline and the command line stays the single way to run a job. Progress is read back
from the job folder (done markers, the pipeline log and the cloud usage log).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from . import review
from . import spec as spec_mod
from .pipeline import Pipeline
from .workspace import IMAGE_EXT, Cache, Job, open_job

DOWNLOADS = {"model.tflite", "labels.txt", "model_int8.tflite", "model_data.h", "report.json"}
WEB = Path(__file__).parent / "web"


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")
    return s[:48] or "job"


def default_interviewer(home: Path, backend: str = "local", public: bool = False):
    from . import interview
    from .llm import Client

    client = Client(usage_log=home / "logs" / "llm_usage.jsonl")
    model = client.model_for("text")
    objaverse_dir = home / "cache" / "objaverse"
    cats = interview.CategoryIndex(cache_dir=objaverse_dir, live=interview.live_searcher(objaverse_dir))
    ranker = interview.photo_ranker(client, client.model_for("vision"))
    profiler = interview.scene_profiler(client, client.model_for("vision"))
    return lambda: interview.Session(client, model, cats, scene_check=interview.check_scene, photo_ranker=ranker,
                                     scene_profiler=profiler, photo_models=backend != "nebius", uploads_only=public)


MAX_PHOTOS = 400  # per upload on a public server
SESSION_HOURS = 6  # an interview left open longer is dropped


def create_app(home: Path, backend: str = "local", max_usd: float | None = None, interviewer=None,
               launcher=None, public: bool = False, key: str | None = None, max_runs: int = 2):
    """public: a server others can reach. Specs may only use photos uploaded through it (no server paths),
    at most `max_runs` runs at a time, and, with `key`, every request needs it (a link with ?key=... sets a
    cookie)."""
    import hmac

    from fastapi import Request
    from fastapi.responses import JSONResponse

    from .interview import FIRST_QUESTION, spec_paths

    app = FastAPI(title="EyeForge")

    if key:
        @app.middleware("http")
        async def require_key(request: Request, call_next):
            given = request.query_params.get("key") or request.cookies.get("eyeforge_key") or \
                request.headers.get("x-eyeforge-key") or ""
            if not hmac.compare_digest(given, key):
                if request.url.path.startswith("/api/"):
                    return JSONResponse({"detail": "this server needs the access link"}, status_code=401)
                return HTMLResponse("<p>EyeForge: open the link with the access key you were given.</p>", 401)
            response = await call_next(request)
            if request.query_params.get("key"):
                response.set_cookie("eyeforge_key", key, httponly=True, samesite="strict", max_age=14 * 86400)
            return response

    sessions: dict[str, object] = {}
    runs: dict[str, subprocess.Popen] = {}
    started: dict[str, float] = {}  # interview session -> start time
    new_session = interviewer
    launch = launcher or _launch

    def job_of(name: str) -> Job:
        job = Job(home / "jobs" / slug(name))
        if not job.spec_path.exists() and not (job.root / "request.json").exists():
            raise HTTPException(404, f"no job {name}")
        return job

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (WEB / "index.html").read_text()

    @app.get("/api/config")
    def config():
        return {"backend": backend, "first_question": FIRST_QUESTION}

    @app.post("/api/interview")
    def interview_start():
        nonlocal new_session
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
        return {"session": sid, "text": FIRST_QUESTION}

    locks: dict[str, threading.Lock] = {}

    def guarded(sid: str, fn, *args):
        """One call at a time per interview (a double submit would interleave two model turns), and the
        right status: 502 when the model service fails, 409 when the interview is already over."""
        from .llm import LLMError
        with locks.setdefault(sid, threading.Lock()):
            try:
                return fn(*args)
            except LLMError as e:
                raise HTTPException(502, f"model service: {e}")
            except RuntimeError as e:
                raise HTTPException(409, str(e))

    def turn_out(sid: str, turn) -> dict:
        out = {"kind": turn.kind, "text": turn.text}
        if turn.spec is not None:
            out["spec"] = json.loads(turn.spec.to_json())
        if turn.request is not None:
            out["request"] = turn.request
        if turn.kind in ("done", "failed"):
            sessions.pop(sid, None)
            locks.pop(sid, None)
        return out

    @app.post("/api/interview/{sid}")
    def interview_reply(sid: str, body: dict):
        session = sessions.get(sid)
        if session is None:
            raise HTTPException(404, "unknown interview session")
        return turn_out(sid, guarded(sid, session.reply, str(body.get("text", ""))))

    @app.post("/api/interview/{sid}/photos")
    def interview_photos(sid: str, purpose: str = Form("training"), cls: str = Form(""),
                               photos: list[UploadFile] = File(...)):
        """Photos uploaded in the chat, asked for by the interview or not. They wait in uploads/<session>
        until a job adopts them. Sync on purpose: the model calls after the upload block, and FastAPI runs
        sync handlers in a thread pool instead of the event loop (async stalled every other request)."""
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
        if not n:
            raise HTTPException(400, "no images among the files")
        return {**turn_out(sid, guarded(sid, session.add_photos, str(folder), n, purpose, cls or None)), "photos": n}

    @app.get("/api/interview/{sid}/thumb")
    def interview_thumb(sid: str, file: str):
        """Thumbnail of a photo the interview is proposing (only those: the path comes from the user)."""
        session = sessions.get(sid)
        if session is None or file not in {c["file"] for c in session.candidates}:
            raise HTTPException(404, "not a proposed photo")
        return FileResponse(photo_thumb(home / "uploads" / sid / "_thumbs", Path(file)))

    @app.post("/api/interview/{sid}/pick")
    def interview_pick(sid: str, body: dict):
        session = sessions.get(sid)
        if session is None:
            raise HTTPException(404, "unknown interview session")
        return turn_out(sid, guarded(sid, session.choose, [str(f) for f in body.get("files") or []],
                                     str(body.get("class", ""))))

    @app.post("/api/jobs")
    def create_job(spec: str = Form(...), photos: list[UploadFile] = File(default=[])):
        if public:
            if sum(p.poll() is None for p in runs.values()) >= max_runs:
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
            raise HTTPException(400, f"invalid spec: {e}")
        if public:  # only what was uploaded here: a path would let a spec read (and upload) any server file
            uploads = (home / "uploads").resolve()
            bad = [p for p in spec_paths(parsed) if p != "photos" and not (
                Path(p).is_absolute() and Path(p).resolve().is_relative_to(uploads))]
            if bad:
                raise HTTPException(400, f"invalid spec: only photos uploaded here can be used, not {bad}")
        # A new request never reuses an existing job folder: it would overwrite that job's results and pick
        # up its review corrections. Reruns of a job go through the corrections endpoint instead.
        base = name = slug(parsed.name)
        n = 1
        while (home / "jobs" / name).exists():
            n += 1
            name = f"{base}_{n}"
        job = open_job(home, name)
        parsed = adopt_uploads(parsed, home / "uploads", job)
        request = job.root / "request.json"
        request.write_text(json.dumps({**json.loads(parsed.to_json()), "name": name}, indent=2))
        folder = job.input / "photos"
        for p in images:
            folder.mkdir(parents=True, exist_ok=True)
            (folder / Path(p.filename).name).write_bytes(p.file.read())
        runs[name] = launch(home, job, request, backend, max_usd)
        return {"job": name, "photos": len(images)}

    @app.get("/api/jobs")
    def list_jobs():
        root = home / "jobs"
        names = sorted(d.name for d in root.iterdir() if (d / "spec.json").exists() or (d / "request.json").exists()) \
            if root.exists() else []
        return {"jobs": names}

    @app.get("/api/jobs/{name}")
    def job_status(name: str):
        job = job_of(name)
        return status(job, home, running=running(job), backend=backend)

    @app.get("/api/jobs/{name}/review")
    def review_data(name: str):
        job = job_of(name)
        spec = spec_mod.load(job.spec_path if job.spec_path.exists() else job.root / "request.json")
        return review.gather(job, Cache(home / "cache"), spec.labels, task=spec.task)

    @app.get("/api/jobs/{name}/media/{path:path}")
    def media(name: str, path: str, size: int = 360):
        """A small JPEG of an image inside the job folder (only the folders review shows); size 1024 for the
        point editor."""
        job = job_of(name)
        src = (job.root / path).resolve()
        if not src.is_relative_to(job.root.resolve()) or Path(path).parts[0] not in review.MEDIA_DIRS \
                or src.suffix.lower() not in IMAGE_EXT or not src.is_file():
            raise HTTPException(404, "no such image")
        return FileResponse(thumbnail(job, src, 1024 if size > 360 else 360))

    def running(job: Job) -> bool:
        """This server's run process, or a cloud job left by an earlier server process (still in time)."""
        proc = runs.get(job.name)
        return (proc is not None and proc.poll() is None) or cloud_job(job) is not None

    @app.post("/api/jobs/{name}/cancel")
    def cancel(name: str):
        """Stops the run: the local process and, on Nebius, the cloud job (which bills until it ends)."""
        job = job_of(name)
        proc = runs.get(job.name)
        if proc is not None and proc.poll() is None:
            proc.terminate()
        cj = cloud_job(job)
        if cj:
            from .backends.nebius import NebiusCLI
            NebiusCLI().cancel(cj["id"])
            (job.logs / "cloud_job.json").unlink(missing_ok=True)
        with open(job.logs / "pipeline.log", "a") as f:
            f.write("FAILED: cancelled by the user\n")
        return {"cancelled": True, "cloud_job": cj["id"] if cj else None}

    @app.post("/api/jobs/{name}/corrections")
    def corrections(name: str, body: dict):
        job = job_of(name)
        if running(job):
            raise HTTPException(409, f"job {job.name} is running")
        start = review.apply(job, Cache(home / "cache"), body.get("labels") or {}, body.get("assets") or {},
                             body.get("points") or {}, share=not public)
        if start is None:
            return {"rerun_from": None}
        request = job.root / "request.json"
        if not request.exists():
            request.write_text(job.spec_path.read_text())
        runs[job.name] = launch(home, job, request, backend, max_usd, start=start)
        return {"rerun_from": start}

    @app.get("/api/jobs/{name}/files/{file}")
    def download(name: str, file: str):
        if file not in DOWNLOADS:
            raise HTTPException(404, "not a downloadable file")
        path = job_of(name).stage_dir("export") / file
        if not path.exists():
            raise HTTPException(404, "not ready")
        return FileResponse(path, filename=f"{job_of(name).name}_{file}")

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
    log = open(job.logs / "pipeline.log", "a")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=Path(__file__).resolve().parents[1])


def cloud_job(job: Job) -> dict | None:
    """The Nebius job a run left running, while it can still be running (its timeout plus the queue)."""
    f = job.logs / "cloud_job.json"
    try:
        cj = json.loads(f.read_text())
    except (OSError, ValueError):
        return None
    return cj if time.time() - cj["started"] < cj["hours"] * 3600 else None


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


def status(job: Job, home: Path, running: bool, backend: str = "local") -> dict:
    # A relaunched job has a new request.json while spec.json is still the previous run's until the run
    # starts: read the newer one.
    paths = [p for p in (job.spec_path, job.root / "request.json") if p.exists()]
    spec_path = max(paths, key=lambda p: p.stat().st_mtime)
    log_path = job.logs / "pipeline.log"
    log = log_path.read_text(errors="replace").splitlines() if log_path.exists() else []
    try:
        spec = spec_mod.load(spec_path)
    except spec_mod.SpecError as e:  # e.g. an old job whose spec a newer version no longer accepts
        return {"job": job.name, "spec": None, "running": running, "stages": [], "log": log[-30:] + [f"error: {e}"],
                "cloud_usd": 0.0, "report": None}
    current = next((l[2:].split(" ")[0] for l in reversed(log) if l.startswith("> ")), None)
    failed = any(l.startswith("FAILED") or l.startswith("error:") for l in log[-5:])
    # Token Factory model each stage really called (the usage log records it per call, by stage name).
    llm_log = job.logs / "llm_usage.jsonl"
    models = {r["purpose"]: r["model"] for r in map(json.loads, llm_log.read_text().splitlines())} \
        if llm_log.exists() else {}
    stages = []
    for p in Pipeline(spec, job, Cache(home / "cache")).plan():
        state = "done" if p.action == "skip" else "pending"
        if state == "pending" and p.stage.name == current:
            state = "running" if running else ("failed" if failed else "pending")
        stages.append({"name": p.stage.name, "runtime": p.stage.runtime, "state": state,
                       "powered_by": p.stage.powered_by(spec, backend, models.get(p.stage.name))})
    usage = job.logs / "cloud_usage.jsonl"
    cost = sum(json.loads(l)["usd_estimate"] for l in usage.read_text().splitlines()) if usage.exists() else 0.0
    report = job.stage_dir("export") / "report.json"
    fresh = any(st["name"] == "export" and st["state"] == "done" for st in stages)  # not a stale model
    return {"job": job.name, "spec": json.loads(spec.to_json()), "running": running,
            "stages": stages, "log": log[-30:], "cloud_usd": round(cost, 2), "phase": phase(log, running),
            "report": json.loads(report.read_text()) if fresh and report.exists() else None}


def serve(home: Path, backend: str, host: str, port: int, max_usd: float | None, public: bool = False,
          key: str | None = None, max_runs: int = 2) -> None:
    import uvicorn
    uvicorn.run(create_app(home, backend, max_usd, public=public, key=key, max_runs=max_runs), host=host, port=port)
