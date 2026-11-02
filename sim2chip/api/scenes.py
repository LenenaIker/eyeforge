"""Walk scenes: the dataset page's proposals and the scene job (scene_job.py), apart from training."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Request


def add(app: FastAPI, ctx) -> None:
    lib, home, profiler, owned = ctx.lib, ctx.home, ctx.profiler, ctx.owned
    profiling, owner_of = ctx.profiling, ctx.owner_of
    may_start, jobs_held, accounts, charged = ctx.may_start, ctx.jobs_held, ctx.accounts, ctx.charged

    proposals: dict[str, tuple[float, list[dict]]] = {}  # dataset -> (profile time, proposals)
    proposing: set[str] = set()
    scene_runner = ThreadPoolExecutor(1, thread_name_prefix="scene")  # one scene at a time: one VM

    def fail_stale_scenes() -> None:
        """Scenes left building by a server that stopped (its queue and jobs went with it) fail with a reason, so
        the page can extract them again; a job still running (the CLI's) is left alone."""
        from .. import scene_job
        for sid in lib.building_scenes():
            if scene_job.stale(home, sid):
                lib.fail_scene(sid, "stopped when the server restarted: extract it again")
    fail_stale_scenes()

    def stop_vm_left_on() -> None:
        """First in the queue, so no new scene starts on a VM that is about to be stopped."""
        from .. import scene_job
        if scene_job.vm_left_on(home):
            print("the scene VM may have been left on by a job that is gone: stopping it", flush=True)
            scene_runner.submit(scene_job.vm_off, home)
    stop_vm_left_on()

    def propose_soon(owner: str, ds: str, made: float) -> None:
        if ds in proposing:
            return
        proposing.add(ds)

        def work():
            from .. import scene_pick
            try:
                proposals[ds] = (made, scene_pick.propose(lib.described_photos(owner, ds), home / "cache"))
            except Exception as e:
                print(f"scene proposals of dataset {ds} failed: {type(e).__name__}: {e}", flush=True)
            finally:
                proposing.discard(ds)
        profiler.submit(work)

    def scene_state(owner: str, sid: str) -> dict | None:
        from .. import scene_job
        st = lib.scene_status(owner, sid)
        if st is None:
            return None
        return {"id": sid, **st, **{k: v for k, v in (scene_job.progress(home, sid) or {}).items()
                                    if k in ("step", "done", "total")}}

    def proposals_of(owner: str, item: str) -> dict:
        """The dataset's photos proposed for a walk scene, each with its scene when one was started; computing
        while the profile or the proposals are being made (asked again later). Also for the chat."""
        prof = lib.profile(owner, item)
        if prof is None:
            return {"proposals": [], "computing": item in profiling}
        made, found = proposals.get(item, (None, None))
        if made != prof["created"]:
            propose_soon(owner, item, prof["created"])
            return {"proposals": [], "computing": True}
        return {"proposals": [{**p, "scene": scene_state(owner, lib.scene_of_photo(owner, p["photo_id"]))}
                              for p in found], "computing": False}
    ctx.scene_proposals = proposals_of

    @app.get("/api/library/datasets/{item}/scenes")
    def library_scene_proposals(item: str, request: Request):
        owner, _ = owned(request, "datasets", item)
        return proposals_of(owner, item)

    @app.post("/api/library/scenes/extract")
    async def library_scene_extract(request: Request):
        """Start the scene job for one of the owner's photos (two to three hours on a paid GPU)."""
        body = await request.json()
        owner = owner_of(request)
        pid = int(body.get("photo_id", -1))
        if lib.photo_file(owner, pid) is None:
            raise HTTPException(404, "no such photo")
        sid = lib.scene_of_photo(owner, pid)
        st = lib.scene_status(owner, sid)
        if st and st["status"] in ("building", "ready"):
            return scene_state(owner, sid)
        from .. import scene_job
        if why := scene_job.unavailable():
            raise HTTPException(503, why)
        acc = may_start(request, "scene")
        lib.start_scene(owner, pid)

        def work():
            # Hours may have passed in the queue. Only one scene runs at a time, so the others wait behind this one.
            now = accounts.get(acc.id) if acc is not None else None
            if acc is not None and not (now and accounts.can_start(now, "scene", jobs_held(acc.id))):
                lib.fail_scene(sid, "your balance no longer covers a scene: top up and try again")
                return
            try:
                scene_job.run(home, owner, pid)
            except Exception as e:  # never left building: that would hold the price of a scene for good
                print(f"scene {sid} failed: {type(e).__name__}: {e}", flush=True)
                lib.fail_scene(sid, f"the scene job stopped ({type(e).__name__})")
                return
            # What it used, ready or not. "Try again" keeps the scene's id, and a usage line only grows: each attempt
            # has its own line, or a second attempt would be free up to what the first one cost.
            p = scene_job.progress(home, sid) or {}
            if acc is not None and p.get("cost"):
                charged(acc.id, accounts.meter(acc.id, f"scene:{sid}:{int(p['started'])}", "scene", p["cost"]))
        scene_runner.submit(work)
        return scene_state(owner, sid)

    @app.get("/api/library/scenes/{item}")
    def library_scene(item: str, request: Request):
        st = scene_state(owner_of(request), item)
        if st is None:
            raise HTTPException(404, "no such scene")
        return st
