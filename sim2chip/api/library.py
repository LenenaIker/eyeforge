"""The library's routes: datasets of photos, their labelings and the teacher on them, 3D objects and kits."""
from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from .. import library, review
from .. import spec as spec_mod
from ..job_service import LABELING
from ..server import MAX_MODEL_BYTES, MAX_PHOTO_BYTES, MAX_PHOTOS, OWNER, capped, photo_thumb, slug, timed
from ..workspace import IMAGE_EXT, Job, open_job


def add(app: FastAPI, ctx) -> None:
    lib, visitor, jobs, home, my_jobs, owner_of = ctx.lib, ctx.visitor, ctx.jobs, ctx.home, ctx.my_jobs, ctx.owner_of
    profile_soon, profiling, public = ctx.profile_soon, ctx.profiling, ctx.public
    owned, describer = ctx.owned, ctx.describer
    may_start, budget_left, bill_to, adopt_library = ctx.may_start, ctx.budget_left, ctx.bill_to, ctx.adopt_library
    needs_check = ctx.needs_check

    @app.get("/api/library")
    def library_data(request: Request):
        """Everything the visitor owns: models (jobs), datasets of real photos, 3D objects, scenes and kits.
        Their jobs are read into the library first (only the new or changed ones)."""
        mine_jobs = [Job(home / "jobs" / n) for n in my_jobs(request)]
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

    @app.get("/api/library/photos/{pid}")
    def library_photo(pid: int, request: Request, size: int = 360):
        src = lib.photo_file(owner_of(request), pid)
        if src is None or not src.exists():
            raise HTTPException(404, "no such photo")
        big = 1024 if size > 360 else 360
        return FileResponse(photo_thumb(lib.root / "_thumbs", src, Path(f"{src.stem}.{big}.jpg"), big))

    @app.get("/api/library/{kind}/{item}/card")
    def library_card(kind: str, item: str, request: Request):
        src = lib.card_file(owner_of(request), kind, item) if kind in ("object", "scene") else None
        if src is None or not src.exists():
            raise HTTPException(404, "no picture")
        return FileResponse(src)

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
        with timed(request, "save"):
            tmp, files = incoming(photos, IMAGE_EXT)
        try:
            if not files:
                raise HTTPException(400, "no images among the files")
            with timed(request, "library"):
                made = lib.new_dataset(owner_of(request), name.strip()[:80] or "My photos", files)
            return {"id": made, "photos": len(files)}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @app.post("/api/library/datasets/{item}/photos")
    def library_add_photos(item: str, request: Request, photos: list[UploadFile] = File(...)):
        import shutil
        owner, _ = owned(request, "datasets", item)
        if public and len(photos) > MAX_PHOTOS:
            raise HTTPException(413, f"at most {MAX_PHOTOS} photos per upload")
        with timed(request, "save"):
            tmp, files = incoming(photos, IMAGE_EXT)
        try:
            with timed(request, "library"):
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

    def english_classes(names: list[str], task: str) -> list[dict]:
        """Class names as the user typed them, as a labeling's classes: keys in English whatever the language
        (user, 2026-10-10), and for counting the English names the teacher looks for."""
        from .. import teacher
        names = list(dict.fromkeys(str(n).strip()[:60] for n in names if str(n).strip()))
        english = teacher.english_phrases({f"c{i}": n for i, n in enumerate(names)}) or {}
        out = []
        for i, n in enumerate(names):
            phrases = english.get(f"c{i}") or []
            key = re.sub(r"[^a-z0-9]+", "_", (phrases[0] if phrases else n).lower()).strip("_") or f"class_{len(out)}"
            if key not in [c["key"] for c in out]:
                out.append({"key": key, "description": n, "phrases": phrases if task == "count" else []})
        return out

    def labeling_of(request: Request, lid: str) -> tuple[str, dict]:
        owner = owner_of(request)
        if not lib.owns_labeling(owner, lid):
            raise HTTPException(404, "no such labeling")
        return owner, lib.labeling(lid)

    def changed(fn, *args) -> dict:
        """A change to a labeling's classes: the labeling that has it now (a copy when a model uses the old one)."""
        try:
            return {"labeling": fn(*args)}
        except ValueError as e:
            raise HTTPException(400, str(e)) from None

    @app.get("/api/library/datasets/{item}")
    def library_dataset(item: str, request: Request):
        """A dataset: its photos (ids), its labelings and what its photos show."""
        owner, _ = owned(request, "datasets", item)
        d = next(d for d in lib.datasets(owner) if d["id"] == item)
        return {**d, "photo_ids": [p["id"] for p in lib.photos(item)], "profiling": item in profiling}

    @app.post("/api/library/datasets/{item}/labelings")
    def library_new_labeling(item: str, body: dict, request: Request):
        """A new labeling of the dataset: {name, task: count | classify, classes: [names]}; teach: start the
        teacher on it right away (a few photos to check first, then every photo)."""
        owner, _ = owned(request, "datasets", item)
        task = body.get("task")
        if task not in library.TASKS:
            raise HTTPException(400, "say what to label: count or classify")
        classes = english_classes(body.get("classes") or [], task)
        if not classes or len(classes) > 10 or (task == "classify" and len(classes) < 2):
            raise HTTPException(400, "count needs 1 to 10 classes, classify 2 to 10")
        name = str(body.get("name") or "").strip()[:80] or ", ".join(c["description"] for c in classes)[:80]
        lid = lib.create_labeling(item, name, task, classes)
        return {"labeling": lid, **(teach(lid, owner, request) if body.get("teach") else {})}

    def teach(lid: str, owner: str, request: Request) -> dict:
        """The teacher labels a labeling's photos without training a model: first a few for the user to check
        while they have reviewed fewer than a round, then every photo (label_all)."""
        lab = lib.labeling(lid)
        data = {"name": f"label {lab['name']}", "task": "count" if lab["task"] == "count" else "roi_classification",
                **({} if lab["task"] == "count" else {"background_class": None}),  # classify: only its classes
                "classes": [{"id": c["key"], "description": c["description"] or c["key"].replace("_", " "),
                             **({"names": c["phrases"]} if lab["task"] == "count" and c["phrases"] else {})}
                            for c in lab["classes"]],
                "user_data": {"raw": f"library:{lab['dataset']}", "labeling": lid}}
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
        parsed = adopt_library(parsed, owner, job)
        (job.root / "request.json").write_text(json.dumps({**json.loads(parsed.to_json()), "name": name}, indent=2))
        (job.root / LABELING).write_text(json.dumps({"dataset": lab["dataset"], "labeling": lid, "state": "checking"}))
        jobs.warm_up("calibration", name)
        if needs_check(job):
            jobs.calibrate(job)
        else:
            jobs.label_all(job)
        return {"job": name}

    @app.post("/api/library/labelings/{lid}/teach")
    def library_teach(lid: str, request: Request):
        owner, _ = labeling_of(request, lid)
        return teach(lid, owner, request)

    @app.get("/api/library/labelings/{lid}")
    def library_labeling(lid: str, request: Request):
        """A labeling with every photo of its dataset: its marks and whether the user reviewed it. The marks a
        finished teacher job made come in first."""
        owner, lab = labeling_of(request, lid)
        lib.sync(owner, [Job(home / "jobs" / n) for n in my_jobs(request)])
        photos = [{k: v for k, v in p.items() if k != "file"} for p in lib.labeling_photos(lid)]
        return {**lib.labeling(lid), "photos": photos}

    @app.patch("/api/library/labelings/{lid}")
    async def library_rename_labeling(lid: str, request: Request):
        labeling_of(request, lid)
        name = str((await request.json()).get("name", "")).strip()[:80]
        if not name:
            raise HTTPException(400, "give it a name")
        lib.rename_labeling(lid, name)
        return {"ok": True}

    @app.delete("/api/library/labelings/{lid}")
    def library_delete_labeling(lid: str, request: Request):
        labeling_of(request, lid)
        lib.delete_labeling(lid)
        return {"ok": True}

    @app.post("/api/library/labelings/{lid}/copy")
    def library_copy_labeling(lid: str, body: dict, request: Request):
        labeling_of(request, lid)
        return {"labeling": lib.copy_labeling(lid, str(body.get("name") or "").strip()[:80] or None)}

    @app.post("/api/library/labelings/{lid}/classes")
    def library_add_class(lid: str, body: dict, request: Request):
        """A new class ({name, description}): the reviewed photos are reviewed again for it."""
        _, lab = labeling_of(request, lid)
        c = english_classes([body.get("name") or ""], lab["task"])
        if not c:
            raise HTTPException(400, "give the class a name")
        return changed(lib.add_class, lid, c[0]["key"], str(body.get("description") or c[0]["description"]),
                       c[0]["phrases"] or None)

    @app.patch("/api/library/labelings/{lid}/classes/{cid}")
    def library_edit_class(lid: str, cid: int, body: dict, request: Request):
        """Rename a class (key, in English), or change its description or the English names the teacher looks
        for (names)."""
        labeling_of(request, lid)
        key = re.sub(r"[^a-z0-9]+", "_", str(body["key"]).lower()).strip("_") if body.get("key") else None
        names = [str(n) for n in body["names"]] if isinstance(body.get("names"), list) else None
        return changed(lib.edit_class, lid, cid, key, body.get("description"), names)

    @app.delete("/api/library/labelings/{lid}/classes/{cid}")
    def library_delete_class(lid: str, cid: int, request: Request):
        labeling_of(request, lid)
        return changed(lib.delete_class, lid, cid)

    @app.put("/api/library/labelings/{lid}/photos/{pid}")
    def library_review_photo(lid: str, pid: int, body: dict, request: Request):
        """The user's marks of a photo ({"marks": [{class, x, y}]} for count, [{class}] for classify): reviewed."""
        labeling_of(request, lid)
        try:
            lib.review_photo(lid, pid, [m for m in body.get("marks") or [] if isinstance(m, dict)])
        except review.StackedMarks:
            raise
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        return {"ok": True}

    @app.delete("/api/library/labelings/{lid}/photos/{pid}")
    def library_unreview_photo(lid: str, pid: int, request: Request):
        labeling_of(request, lid)
        lib.unreview_photo(lid, pid)
        return {"ok": True}

    @app.delete("/api/library/photos/{pid}")
    def library_delete_photo(pid: int, request: Request):
        if not lib.delete_photo(owner_of(request), pid):
            raise HTTPException(404, "no such photo")
        return {"ok": True}

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
