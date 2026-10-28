"""The routes of a job (a model): create, run again, check the teacher, correct, revise, add photos, download
and delete."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from .. import library, review, warm as warm_mod
from .. import spec as spec_mod
from ..interview import sources as interview_sources
from ..job_service import CALIBRATION, LABELING
from ..pipeline import Pipeline
from ..server import (ALREADY_RUNNING, CHAT_VIEW, CHATS, CHECK_OVER, DOWNLOADS, MAX_PHOTOS, OWNER, SENT, _photo_labels,
                      add_chat, adopt_uploads, latest_spec, save_photos, slug, status, thumbnail, timed)
from ..workspace import IMAGE_EXT, Cache, Job, content_key, open_job, snapshot


def add(app: FastAPI, ctx) -> None:
    lib, jobs, budget_left, public, room_for_a_run = ctx.lib, ctx.jobs, ctx.budget_left, ctx.public, ctx.room_for_a_run
    check_paths, home, mine, may_start, visitor = ctx.check_paths, ctx.home, ctx.mine, ctx.may_start, ctx.visitor
    adopt_library, owner_of, credit_interview = ctx.adopt_library, ctx.owner_of, ctx.credit_interview
    keep_chat, needs_check, bill_to, start_run = ctx.keep_chat, ctx.needs_check, ctx.bill_to, ctx.start_run
    my_jobs, backend, job_of, kept, owns_upload = ctx.my_jobs, ctx.backend, ctx.job_of, ctx.kept, ctx.owns_upload
    save_review, refresh = ctx.save_review, ctx.refresh

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
        if why := spec_mod.no_photos(parsed):
            raise HTTPException(400, f"invalid spec: {why}")
        check_paths(parsed, request)
        # The same setup sent twice (a double click, "Looks good" and Train together) must not pay for two runs.
        same = json.loads(parsed.to_json())
        for other, proc in jobs.runs.items():
            req = home / "jobs" / other / SENT  # as sent: request.json has the library's photos in it
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
        (job.root / SENT).write_text(json.dumps(same))
        parsed = adopt_uploads(parsed, home / "uploads", job)
        save_photos(job.input / "photos", images)  # before the library: raw may merge them with a dataset
        try:
            parsed = adopt_library(parsed, owner_of(request), job, session)
        except HTTPException:
            shutil.rmtree(job.root, ignore_errors=True)  # a refused spec leaves no empty job behind
            raise
        req = job.root / "request.json"
        req.write_text(json.dumps({**json.loads(parsed.to_json()), "name": name}, indent=2))
        credit_interview(job, session, request)
        keep_chat(job, session, request)
        if session:
            warm_mod.note(home, "job", name, session=f"session:{session}")
        if calibrate_first and parsed.user_data.raw and needs_check(job):
            bill_to(job, may_start(request, "run"))  # the teacher's check is the start of the paid run
            jobs.warm_up("calibration", name)
            jobs.calibrate(job)
            return {"job": name, "photos": len(images), "calibrating": True}
        start_run(job, req, payer=request)
        return {"job": name, "photos": len(images)}

    @app.get("/api/jobs")
    def list_jobs(request: Request):
        return {"jobs": my_jobs(request)}

    def forget_job(job: Job) -> None:
        """A model the owner deletes: its folder, its cloud storage and its records in the library. What it gave
        the library (photos, labels, objects, scenes) stays there."""
        if backend == "nebius" and ((job.root / "cloud_id").exists() or (job.logs / "cloud_usage.jsonl").exists()):
            from ..backends.nebius import NebiusBackend
            try:
                NebiusBackend().delete_storage(job)
            except Exception as e:  # left for a hand cleanup: the log says which
                print(f"could not delete the cloud storage of {job.name}: {e}", flush=True)
        shutil.rmtree(job.root, ignore_errors=True)
        lib.forget_jobs([job.name])

    @app.delete("/api/jobs/{name}")
    def delete_job(name: str, request: Request):
        job = job_of(name, request)
        if jobs.running(job):
            raise HTTPException(409, "the model is still running: cancel it or wait until it ends")
        forget_job(job)
        return {"ok": True}

    @app.patch("/api/jobs/{name}")
    async def rename_job(name: str, request: Request):
        """The name the user gives a model; its folder (and every link to it) keeps the first one."""
        job = job_of(name, request)
        title = str((await request.json()).get("title", "")).strip()[:80]
        if not title:
            raise HTTPException(400, "a name is needed")
        (job.root / library.TITLE).write_text(title)
        return {"ok": True}

    @app.get("/api/jobs/{name}")
    def job_status(name: str, request: Request):
        job = job_of(name, request)
        return status(job, home, running=jobs.running(job), backend=backend)

    @app.get("/api/jobs/{name}/chat")
    def job_chat(name: str, request: Request):
        """The chats that set this model up, oldest first."""
        job = job_of(name, request)
        if not (job.root / CHATS).exists():
            find_old_chat(job, request)
        try:
            return {"chats": json.loads((job.root / CHATS).read_text())}
        except (OSError, ValueError):
            return {"chats": []}

    def find_old_chat(job: Job, request: Request) -> None:
        """A job made before chats were kept with it: the requester's chat whose setup has its name, if the web
        still keeps that chat's view (a "New model" drops it)."""
        try:
            name = json.loads((job.root / SENT).read_text())["name"]
        except (OSError, ValueError, KeyError):
            return
        uploads = home / "uploads"
        for d in uploads.iterdir() if uploads.exists() else []:
            f = d / CHAT_VIEW
            if not f.is_file() or not owns_upload(d.name, request):
                continue
            try:
                view = json.loads(f.read_text()).get("view") or {}
            except ValueError:
                continue
            if name in {(view.get("spec") or {}).get("name"), (view.get("proposal") or {}).get("name")}:
                add_chat(job, view)
                return

    @app.get("/api/jobs/{name}/review")
    def review_data(name: str, request: Request):
        job = job_of(name, request)
        spec = spec_mod.load(job.spec_path if job.spec_path.exists() else job.root / "request.json")
        return review.gather(job, Cache(home / "cache"), spec.labels, task=spec.task)

    @app.get("/api/jobs/{name}/synthetic")
    def synthetic_data(name: str, request: Request):
        """The synthetic images of the job's last run: what went into training, what the filters left out and
        why, with a few pictures of each (review.synthetic)."""
        return review.synthetic(job_of(name, request))

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

    @app.get("/api/jobs/{name}/assets/{uid}/thumb")
    def asset_thumb(name: str, uid: str, request: Request):
        """A 3D model's card picture from Sketchfab, for the review (only the job's own models; kept in the cache)."""
        from .. import asset_search
        job = job_of(name, request)
        f = job.stage_dir("convert_assets") / "assets.json"
        known = {a["uid"] for a in json.loads(f.read_text())} if f.exists() else set()
        if uid not in known:
            raise HTTPException(404, "no such model")
        got = asset_search.thumbnail(uid, home / "cache" / "sketchfab_thumbs" / f"{uid}.jpg")
        if got is None:
            raise HTTPException(404, "no picture")
        return FileResponse(got, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})

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
        save_review(job, body, check=True)
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
        save_review(job, body, check=True)
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
            save_review(job, body, check=True)
            jobs.calibrate(job)
            return {"rerun_from": "calibration"}
        budget_left()
        # Marks go to the labeling (the run takes a fresh copy of it); 3D model decisions stay with the job.
        start = "label" if save_review(job, body) else None
        if review.apply(job, Cache(home / "cache"), {}, body.get("assets") or {}, share=not public):
            start = start or "qa_assets"
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
        if why := spec_mod.no_photos(parsed):
            raise HTTPException(400, f"invalid spec: {why}")
        check_paths(parsed, request, job)
        old = spec_mod.load(latest_spec(job))
        changed = [k for k, v in json.loads(parsed.to_json()).items()
                   if k != "name" and json.loads(old.to_json()).get(k) != v]
        if not dry:
            may_start(request, "run")
            parsed = adopt_uploads(parsed, home / "uploads", job)
            if not parsed.user_data.labeling and (snap := kept(job)):  # the chat kept the job's photos
                parsed.user_data.labeling = snap["labeling"]
            parsed = adopt_library(parsed, owner_of(request), job, session)
            credit_interview(job, session, request)
            keep_chat(job, session, request, revision=True)
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
        snap = kept(job)
        if snap is None or lib.labeling(snap["labeling"]) is None:
            raise HTTPException(400, "this model has no labeling of photos in your library to add them to")
        lab = lib.labeling(snap["labeling"])
        owner = owner_of(request)
        if public and lab["photos"] + len(images) > MAX_PHOTOS:
            raise HTTPException(413, f"at most {MAX_PHOTOS} photos per model ({lab['photos']} already)")
        import tempfile
        tmp = Path(tempfile.mkdtemp(dir=job.root, prefix="_incoming"))  # its own: batches come in parallel
        with timed(request, "save"):
            names = save_photos(tmp, images)
        files = {n: tmp / saved for n, saved in names}
        with timed(request, "library"):
            lib.add_photos(owner, lab["dataset"], [tmp / saved for _, saved in names])
        try:  # what the user said each photo is in the live test wins over the teacher's: reviewed photos
            for n, v in _photo_labels(labels).items():
                pid = lib.photo_id(lab["dataset"], content_key(files[n])) if n in files else None
                if pid is None:
                    continue
                if "label" in v:
                    lib.review_photo(lab["id"], pid, [{"class": v["label"]}])
                elif "points" in v:
                    lib.review_photo(lab["id"], pid, [{"class": c, "x": x, "y": y} for x, y, c, *_ in v["points"]])
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        refresh(job)
        if more:
            return {"stored": len(names)}
        saved = max(len(names), total)
        req = latest_spec(job)
        if req != job.root / "request.json":
            (job.root / "request.json").write_text(req.read_text())
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
