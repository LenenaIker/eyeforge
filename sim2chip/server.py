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
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from . import spec as spec_mod
from .pipeline import Pipeline
from .workspace import Cache, Job, open_job

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DOWNLOADS = {"model_int8.tflite", "model_data.h", "report.json"}
WEB = Path(__file__).parent / "web"


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")
    return s[:48] or "job"


def default_interviewer(home: Path):
    from . import interview
    from .llm import Client

    client = Client(usage_log=home / "logs" / "llm_usage.jsonl")
    model = client.model_for("text")
    objaverse_dir = home / "cache" / "objaverse"
    cats = interview.CategoryIndex(cache_dir=objaverse_dir, live=interview.live_searcher(objaverse_dir))
    return lambda: interview.Session(client, model, cats, scene_check=interview.check_scene)


def create_app(home: Path, backend: str = "local", max_usd: float | None = None, interviewer=None,
               launcher=None):
    from .interview import FIRST_QUESTION

    app = FastAPI(title="Sim2Chip")
    sessions: dict[str, object] = {}
    runs: dict[str, subprocess.Popen] = {}
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
            new_session = default_interviewer(home)
        sid = uuid.uuid4().hex
        sessions[sid] = new_session()
        return {"session": sid, "text": FIRST_QUESTION}

    @app.post("/api/interview/{sid}")
    def interview_reply(sid: str, body: dict):
        session = sessions.get(sid)
        if session is None:
            raise HTTPException(404, "unknown interview session")
        try:
            turn = session.reply(str(body.get("text", "")))
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        out = {"kind": turn.kind, "text": turn.text}
        if turn.spec is not None:
            out["spec"] = json.loads(turn.spec.to_json())
        if turn.kind in ("done", "failed"):
            sessions.pop(sid, None)
        return out

    @app.post("/api/jobs")
    async def create_job(spec: str = Form(...), photos: list[UploadFile] = File(default=[])):
        try:
            data = json.loads(spec)
            images = [p for p in photos if Path(p.filename or "").suffix.lower() in IMAGE_EXT]
            if images and not (data.get("user_data") or {}).get("labeled"):
                data.setdefault("user_data", {})["raw"] = "photos"  # the teacher model labels them
            parsed = spec_mod.from_dict(data)
        except (json.JSONDecodeError, spec_mod.SpecError) as e:
            raise HTTPException(400, f"invalid spec: {e}")
        name = slug(parsed.name)
        if name in runs and runs[name].poll() is None:
            raise HTTPException(409, f"job {name} is already running")
        job = open_job(home, name)
        request = job.root / "request.json"
        request.write_text(parsed.to_json())
        folder = job.input / "photos"
        for p in images:
            folder.mkdir(parents=True, exist_ok=True)
            (folder / Path(p.filename).name).write_bytes(await p.read())
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
        proc = runs.get(job.name)
        return status(job, home, running=proc is not None and proc.poll() is None)

    @app.get("/api/jobs/{name}/files/{file}")
    def download(name: str, file: str):
        if file not in DOWNLOADS:
            raise HTTPException(404, "not a downloadable file")
        path = job_of(name).stage_dir("export") / file
        if not path.exists():
            raise HTTPException(404, "not ready")
        return FileResponse(path, filename=f"{job_of(name).name}_{file}")

    return app


def _launch(home: Path, job: Job, request: Path, backend: str, max_usd: float | None) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "sim2chip.cli", "--home", str(home), "run", str(request), "--job", job.name,
           "--backend", backend]
    if max_usd is not None:
        cmd += ["--max-usd", str(max_usd)]
    job.logs.mkdir(parents=True, exist_ok=True)
    log = open(job.logs / "pipeline.log", "a")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=Path(__file__).resolve().parents[1])


def status(job: Job, home: Path, running: bool) -> dict:
    spec_path = job.spec_path if job.spec_path.exists() else job.root / "request.json"
    spec = spec_mod.load(spec_path)
    log_path = job.logs / "pipeline.log"
    log = log_path.read_text(errors="replace").splitlines() if log_path.exists() else []
    current = next((l[2:].split(" ")[0] for l in reversed(log) if l.startswith("> ")), None)
    failed = any(l.startswith("FAILED") or l.startswith("error:") for l in log[-5:])
    stages = []
    for p in Pipeline(spec, job, Cache(home / "cache")).plan():
        state = "done" if p.action == "skip" else "pending"
        if state == "pending" and p.stage.name == current:
            state = "running" if running else ("failed" if failed else "pending")
        stages.append({"name": p.stage.name, "runtime": p.stage.runtime, "state": state})
    usage = job.logs / "cloud_usage.jsonl"
    cost = sum(json.loads(l)["usd_estimate"] for l in usage.read_text().splitlines()) if usage.exists() else 0.0
    report = job.stage_dir("export") / "report.json"
    fresh = any(st["name"] == "export" and st["state"] == "done" for st in stages)  # not a stale model
    return {"job": job.name, "spec": json.loads(spec.to_json()), "running": running,
            "stages": stages, "log": log[-30:], "cloud_usd": round(cost, 2),
            "report": json.loads(report.read_text()) if fresh and report.exists() else None}


def serve(home: Path, backend: str, host: str, port: int, max_usd: float | None) -> None:
    import uvicorn
    uvicorn.run(create_app(home, backend, max_usd), host=host, port=port)
