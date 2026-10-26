"""How a spec becomes a job's input: the library's datasets, labelings and 3D objects it names, the user's
reviews saved into its labeling, and the paths a public server lets it use (docs/plan_etiquetado.md)."""
from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import HTTPException, Request

from .. import library, review
from .. import spec as spec_mod
from ..interview import sources as interview_sources
from ..interview import spec_paths
from ..job_service import CALIBRATION_PHOTOS
from ..server import library_ref
from ..workspace import Cache, Job, content_key


def helpers(ctx) -> None:
    """Adds the spec helpers the routes share to the context `ctx`."""
    lib, public, home, owns_upload = ctx.lib, ctx.public, ctx.home, ctx.owns_upload

    def adopt_library(parsed: spec_mod.Spec, owner: str, job: Job, session: str | None = None) -> spec_mod.Spec:
        """The library and the job (docs/plan_etiquetado.md). The spec's 3D objects ("library:<id>") become files
        of the job. Its photos are one dataset of the owner: a library dataset it names (photos attached in the chat
        join it), else the chat's own dataset or a new one. They are labeled in one labeling of that dataset: the
        one the chat named (changed to the spec's classes, a copy when a model was trained with it), else one with
        the same classes, else a new one. The job takes a fixed copy of that labeling (library.export) as its raw
        photos; the teacher's marks come back from it (library.take_teacher)."""
        data = json.loads(parsed.to_json())
        ud = data["user_data"]
        names = {c["id"]: c.get("names") or [] for c in data["classes"]}
        for cid, item in [(c["id"], c["assets"]) for c in data["classes"] if c.get("assets")] + \
                [(p["id"], p) for p in data["props"]]:
            ids = [library_ref(m) for m in item.get("models") or []]
            if not any(ids):
                continue
            if not all(ids):
                raise HTTPException(400, f"invalid spec: models must be objects of your library ({item['models']})")
            # Not an object of the owner's (the chat gave a dataset's id, 2026-10-09): the run searches for models
            # instead of refusing the whole spec.
            ids = [i for i in ids if lib.owns(owner, "object", i)]
            item["models"] = [f"objects/{n}" for n in lib.materialize_objects(owner, ids, job.input / "objects")] \
                if ids else []
            if not item["models"] and item.get("source") == "library":
                item["source"] = "search"
                item["queries"] = item.get("queries") or names.get(cid) or [cid.replace("_", " ")]

        sources = interview_sources(ud.get("raw"))
        labeled = ud.get("labeled")
        if library_ref(labeled):  # a library dataset is photos; how they are labeled is in its labelings
            sources, labeled = [*sources, labeled], None
        dsets = list(dict.fromkeys(d for v in sources if (d := library_ref(v))))
        folders = [v for v in sources if not library_ref(v)]
        ud.update(test=None, class_map={} if not labeled else ud.get("class_map") or {})
        if not dsets and not folders and not labeled:  # synthetic images only
            ud.update(raw=None, labeled=None, labeling=None)
            return spec_mod.from_dict(data)
        if len(dsets) > 1:
            raise HTTPException(400, "invalid spec: one dataset per model (photos attached in the chat join it)")
        if dsets and not lib.owns(owner, "dataset", dsets[0]):
            raise HTTPException(400, f"invalid spec: no dataset {dsets[0]} in your library")
        files = [f for v in folders for f in library._images(job.resolve_input(v))]
        sorted_into: dict[str, list[Path]] = {}  # photos the user sorted into a folder per class
        if labeled:
            base = job.resolve_input(labeled)
            for label in parsed.labels:
                sorted_into[label] = [f for sub in parsed.user_data.class_map.get(label, [label])
                                      for f in library._images(base / sub)]
        every = files + [f for fs in sorted_into.values() for f in fs]
        if dsets:
            ds = dsets[0]
            lib.add_photos(owner, ds, every)
        elif session:
            ds = lib.add_chat_photos(owner, session, every)
        else:
            ds = lib.new_dataset(owner, parsed.name.replace("_", " "), every)
        lib.note_use(job.name, "dataset", ds)

        task = library.task_of(parsed.task)
        classes = labeling_classes(parsed)
        lid = ud.get("labeling")
        if lid and not (lib.owns_labeling(owner, lid) and lib.labeling(lid)["dataset"] == ds
                        and lib.labeling(lid)["task"] == task):
            lid = None
        try:
            if lid:
                lid = lib.fit_classes(lid, classes)
            else:
                lid = lib.find_labeling(ds, task, [c["key"] for c in classes]) or \
                    lib.create_labeling(ds, parsed.name.replace("_", " ").capitalize(), task, classes)
            for label, fs in sorted_into.items():
                for f in fs:
                    if (pid := lib.photo_id(ds, content_key(f))) is not None:
                        lib.review_photo(lid, pid, [{"class": label}])
        except ValueError as e:
            raise HTTPException(400, f"invalid spec: {e}") from None
        # The teacher looks for a class by the labeling's English names: they win over the chat's.
        if task == "count":
            phrases = {c["key"]: c["phrases"] for c in lib.classes(lid)}
            for c in data["classes"]:
                if phrases.get(c["id"]):
                    c["names"] = phrases[c["id"]]
        ud.update(raw=lib.export(lid, job.input), labeled=None, class_map={}, labeling=lid)
        gen = data.setdefault("generation", {})
        if gen.get("scenes"):  # the walk scenes the owner may use (theirs or public, ready) go with the job
            gen["scenes"] = lib.export_scenes(owner, list(gen["scenes"]), job.input)
        return spec_mod.from_dict(data)

    def labeling_classes(parsed: spec_mod.Spec) -> list[dict]:
        """A spec's classes as a labeling's: count, with the English names the teacher looks for; classify, every
        label the photo is chosen among (the "nothing" one too)."""
        if parsed.task == "count":
            from ..stages.label import phrases_of
            phrases = phrases_of(parsed, Cache(home / "cache"))
            return [{"key": c.id, "description": c.description, "phrases": phrases.get(c.id) or []}
                    for c in parsed.classes]
        desc = {c.id: c.description for c in parsed.classes}
        return [{"key": label, "description": desc.get(label, "none of the other classes")} for label in parsed.labels]

    def kept(job: Job) -> dict | None:
        """The job's copy of its labeling (library.export), or None for a job without one."""
        f = job.input / library.KEEP
        return json.loads(f.read_text()) if f.exists() else None

    def needs_check(job: Job) -> bool:
        """The teacher's check comes first while the user has reviewed fewer photos of the labeling than one round
        shows: then the teacher has learned nothing of their criterion yet."""
        snap = kept(job)
        return snap is not None and sum(p["reviewed"] for p in snap["photos"].values()) < CALIBRATION_PHOTOS

    def refresh(job: Job) -> None:
        """The job's copy of its labeling made again (new reviews, new photos), unless the labeling is gone."""
        snap = kept(job)
        if snap is not None and lib.labeling(snap["labeling"]) is not None:
            lib.export(snap["labeling"], job.input)

    def save_review(job: Job, body: dict, check: bool = False) -> int:
        """The user's marks of photos of the job ({"points": {photo: [[x, y, class], ...]}} for count, {"labels":
        {photo: class}} for classify, keyed as the job names its photos) into its labeling, as reviewed photos.
        check: a round of the teacher's check, whose photos the user looked at one by one: those left as they
        were count as reviewed too, with the teacher's marks (trash_counter, 2026-10-11: the user accepted every
        photo, nothing was saved, and the run trained and was scored on unchecked marks only). Returns how many
        photos."""
        snap = kept(job)
        points, labels = dict(body.get("points") or {}), dict(body.get("labels") or {})
        if check and snap is not None:
            shown = json.loads(f.read_text()) if (f := job.stage_dir("label") / "verdicts.json").exists() else {}
            for rel, v in shown.items():
                if rel in points or rel in labels or rel not in snap["photos"] or snap["photos"][rel]["reviewed"]:
                    continue
                if v.get("objects") is not None:
                    points[rel] = [[o["x"], o["y"], o["class"]] for o in v["objects"]]
                elif v.get("class") and v["class"] != "no_answer" and "objects" not in v:
                    labels[rel] = v["class"]
        if not points and not labels:
            return 0
        if snap is None or lib.labeling(snap["labeling"]) is None:
            raise HTTPException(409, "this model's labeling is no longer in your library")
        n = 0
        try:
            for rel, pts in points.items():
                if (p := snap["photos"].get(rel)) is not None:
                    lib.review_photo(snap["labeling"], p["photo_id"],
                                     [{"class": str(c), "x": x, "y": y} for x, y, c, *_ in pts])
                    n += 1
            for rel, label in labels.items():
                if (p := snap["photos"].get(rel)) is not None:
                    lib.review_photo(snap["labeling"], p["photo_id"], [{"class": str(label)}])
                    n += 1
        except ValueError as e:
            if isinstance(e, review.StackedMarks):
                raise
            raise HTTPException(400, str(e)) from None
        refresh(job)
        return n

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

    ctx.adopt_library = adopt_library
    ctx.labeling_classes = labeling_classes
    ctx.kept = kept
    ctx.needs_check = needs_check
    ctx.refresh = refresh
    ctx.save_review = save_review
    ctx.check_paths = check_paths
