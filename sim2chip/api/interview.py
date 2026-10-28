"""The interview's routes: a chat's turns, its photos and proposed photos, and the chat as the web shows it, to
open it again elsewhere."""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from .. import library
from .. import spec as spec_mod
from ..interview import FIRST_QUESTION
from ..server import (CHAT_VIEW, MAX_VIEW_BYTES, SESSION_HOURS, SESSION_OWNER, attach_library, default_interviewer,
                      latest_spec, library_ref, photo_thumb, save_photos, slug, timed)
from ..workspace import IMAGE_EXT, content_key


def add(app: FastAPI, ctx) -> None:
    lib, interviewer, job_of, may_start, home = ctx.lib, ctx.interviewer, ctx.job_of, ctx.may_start, ctx.home
    owner_of, backend, public, chat_payer = ctx.owner_of, ctx.backend, ctx.public, ctx.chat_payer
    interview_usage, jobs, owns_upload, kept = ctx.interview_usage, ctx.jobs, ctx.owns_upload, ctx.kept
    accounts, meter_chat, owned = ctx.accounts, ctx.meter_chat, ctx.owned

    sessions: dict[str, object] = {}

    def scene_photos(owner: str, ds: str) -> dict:
        """For the chat's scene_photos action: the dataset's proposed photos, or why there are none now."""
        from .. import scene_job
        if not lib.owns(owner, "dataset", ds):
            return {"photos": [], "computing": False, "unavailable": None}
        got = ctx.scene_proposals(owner, ds)
        return {"photos": got["proposals"], "computing": got["computing"], "unavailable": scene_job.unavailable()}

    def spec_photos(owner: str, spec: spec_mod.Spec) -> list[Path]:
        """The photos a proposed spec names: its library datasets' and the folders attached in the chat."""
        raw = spec.user_data.raw if isinstance(spec.user_data.raw, list) else [spec.user_data.raw]
        out = []
        for src in raw:
            ds = library_ref(src)
            if ds and lib.owns(owner, "dataset", ds):
                out += [lib.files / p["file"] for p in lib.photos(ds)]
            elif src and not ds and Path(src).is_dir() and Path(src).resolve().is_relative_to(home / "uploads"):
                out += library._images(Path(src))
        return out
    started: dict[str, float] = {}  # interview session -> start time
    new_session = interviewer

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
            attach_library(sessions[sid], lib, owner, home / "uploads" / sid, kit,
                           on_open=lambda folder: jobs.prefetch_draft(owner, library._images(folder),
                                                                      reason="library photos"))
            if hasattr(sessions[sid], "use_scenes"):  # the chat offers photos for a scene of the user's own place
                sessions[sid].use_scenes(lambda ds: scene_photos(owner, ds))

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

    # The last answer of each chat's turn, {sid: {"act", "turn" | "error", "owner"}}: a proxy gives up on a turn
    # that runs long (Cloudflare: 100 s, HTTP 524; a library dataset measured anew took longer, 2026-10-10) while
    # the turn goes on here, and the web picks its answer up from interview_activity.
    answers: dict[str, dict] = {}
    # The turns each chat has seen arrive, by the web's id: a web whose request failed on the network asks
    # interview_activity whether it reached us, and a retry with the same id gets the first try's answer
    # instead of a second turn (it would answer the same message twice).
    arrived: dict[str, set[str]] = {}

    def kept_answer(sid: str, act: str | None, request: Request):
        """The answer already given to the turn `act` of this chat (a retry), or None."""
        done = answers.get(sid)
        if not act or not done or done["act"] != act or done["owner"] != owner_of(request):
            return None
        if "turn" in done:
            return done["turn"]
        raise HTTPException(done["status"], done["error"])

    def answered(sid: str, act: str | None, out: dict, request: Request) -> dict:
        if act:
            answers[sid] = {"act": act, "turn": out, "owner": owner_of(request)}
        return out

    def guarded(sid: str, fn, *args, act: str | None = None, request: Request | None = None):
        """One call at a time per interview (a double submit would interleave two model turns), and the
        right status: 502 when the model service fails, 409 when the interview is already over.
        act: the front end's id for this turn, under which its steps are shown (see interview_activity)."""
        from ..llm import LLMError
        seen = arrived.setdefault(sid, set())
        retry = bool(act) and act in seen
        if act:
            seen.add(act)
        with locks.setdefault(sid, threading.Lock()):
            if retry:  # the first try held the lock until it ended; its answer is kept right after
                until = time.time() + 10
                while (kept := kept_answer(sid, act, request)) is None and time.time() < until:
                    time.sleep(0.2)
                if kept is not None:
                    return kept
            session = sessions.get(sid)
            if accounts is not None and sid in chat_payer and accounts.balance(chat_payer[sid]) <= 0:
                acc = accounts.get(chat_payer[sid])
                if acc is not None and accounts.tier(acc) == "normal":
                    raise HTTPException(402, "your balance ran out; top it up to go on with this chat")
            if hasattr(session, "begin"):
                session.begin(act)
            try:
                try:
                    return fn(*args)
                except LLMError as e:
                    raise HTTPException(502, f"model service: {e}") from None
                except RuntimeError as e:
                    raise HTTPException(409, str(e)) from None
            except HTTPException as e:
                if act and request is not None:
                    answers[sid] = {"act": act, "error": e.detail, "status": e.status_code, "owner": owner_of(request)}
                raise
            finally:
                meter_chat(sid)

    def turn_out(sid: str, turn) -> dict:
        if isinstance(turn, dict):  # a retried turn: the answer as it was given (see guarded)
            return turn
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
        if turn.kind == "confirm" and turn.spec is not None:  # its classes known: SAM 3 answers before "Teach it"
            try:
                owner = (home / "uploads" / sid / SESSION_OWNER).read_text().strip()
                jobs.prefetch_draft(owner, spec_photos(owner, turn.spec), turn.spec, reason="setup proposed")
            except OSError:
                pass
        if turn.kind in ("done", "failed"):
            sessions.pop(sid, None)
            locks.pop(sid, None)
            arrived.pop(sid, None)
            ended(sid).touch()  # over: never brought back from its transcript
        return out

    @app.post("/api/interview/{sid}")
    def interview_reply(sid: str, body: dict, request: Request, act: str | None = None):
        if (kept := kept_answer(sid, act, request)) is not None:  # retried after the chat ended with it
            return kept
        session = chat_of(sid, request)
        return answered(sid, act, turn_out(sid, guarded(sid, session.reply, str(body.get("text", "")), act=act,
                                                        request=request)), request)

    @app.put("/api/interview/{sid}/view")
    async def interview_save_view(sid: str, request: Request):
        """The chat as the web shows it ({"key", "view"}), so it opens again in another tab, after a reload or on
        another device (interview_open)."""
        if not owns_upload(sid, request):
            raise HTTPException(404, "unknown interview session")
        body = await request.body()
        if len(body) > MAX_VIEW_BYTES:
            raise HTTPException(413, "chat too long to keep")
        data = json.loads(body)
        (home / "uploads" / sid / CHAT_VIEW).write_text(json.dumps(
            {"key": str(data.get("key", ""))[:120], "view": data.get("view"), "t": time.time()}))
        return {"ok": True}

    def open_chats(key: str, request: Request) -> list[tuple[float, str, dict]]:
        """The requester's chats for this key (a new model, a kit, a model being changed) still open: (when, id,
        view), newest first."""
        uploads, out = home / "uploads", []
        for d in uploads.iterdir() if uploads.exists() else []:
            f = d / CHAT_VIEW
            if f.is_file() and not ended(d.name).exists() and owns_upload(d.name, request):
                try:
                    v = json.loads(f.read_text())
                except ValueError:
                    continue
                if v.get("key") == key:
                    out.append((v.get("t", 0), d.name, v.get("view")))
        return sorted(out, key=lambda x: -x[0])

    @app.get("/api/interview/open")
    def interview_open(key: str, request: Request):
        """The requester's last open chat for this key, as the web last showed it."""
        found = open_chats(key, request)
        return {"session": found[0][1], "view": found[0][2]} if found else {}

    @app.delete("/api/interview/open")
    def interview_close(key: str, request: Request):
        """"New model": the open chats for this key are not brought back any more (their transcripts stay)."""
        for _, sid, _ in open_chats(key, request):
            (home / "uploads" / sid / CHAT_VIEW).unlink(missing_ok=True)
        return {"ok": True}

    @app.get("/api/interview/{sid}/activity")
    def interview_activity(sid: str, act: str, request: Request):
        """What the turn `act` is doing right now: its steps so far (thinking, searching, checking the setup),
        polled by the web while it waits. No lock: it must answer while the turn holds it. `now` lets the
        client time running steps with the server's clock. Once the turn is over, its answer too ("turn", or
        "error" and "status"), for a web whose request a proxy cut. arrived: whether the turn reached us at
        all (a request lost on the network never did: the web stops waiting and offers to send it again)."""
        done = answers.get(sid)
        if done and done["act"] == act and done["owner"] == owner_of(request):
            done = {k: v for k, v in done.items() if k in ("turn", "error", "status")}
            session = sessions.get(sid)  # gone once the chat is over
        else:
            done, session = {}, chat_of(sid, request)
        steps = list(getattr(session, "activity", [])) if getattr(session, "activity_id", None) == act else []
        return {"steps": [dict(s) for s in steps], "now": time.time(), "arrived": bool(done) or act in arrived.get(sid, ()),
                **done}

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
        with timed(request, "save"):  # a different photo with a name already there (image.jpg) gets its own
            saved = [folder / f for _, f in save_photos(
                folder, [p for p in photos if Path(p.filename or "").suffix.lower() in IMAGE_EXT])]
        n = len(saved)
        # Every training photo belongs to a dataset: the chat's own, made with its first photos (user, 2026-10-08).
        # Photos to build a class's 3D model from stay with the job that uses them.
        if purpose == "training" and saved:
            with timed(request, "library"):
                lib.add_chat_photos(owner_of(request), sid, saved)
        if not n and not total:
            raise HTTPException(400, "no images among the files")
        if purpose == "training" and more:  # the GPU starts while the rest of the photos come
            jobs.warm_up("photos uploaded", f"session:{sid}")
        if more:
            return {"stored": n}
        if purpose == "training":  # and gets the teacher's first work, while the interview goes on
            jobs.prefetch_draft(owner_of(request), library._images(folder))
        n = max(n, total)
        if (kept := kept_answer(sid, act, request)) is not None:
            return kept
        return answered(sid, act, {**turn_out(sid, guarded(sid, session.add_photos, str(folder), n, purpose,
                                                           cls or None, act=act, request=request)), "photos": n},
                        request)

    @app.post("/api/uploads/known")
    def uploads_known(body: dict, request: Request):
        """Which of these photos (content keys, as content_key makes them from the bytes the web would send)
        are already where they are going: a chat's folder ({"chat", "purpose", "cls"}), a library dataset
        ({"dataset"}) or a model's labeling ({"job"}). The web sends only the others: a retry after a cut or
        a folder sent again with a few new photos costs only what is new."""
        hashes = {str(h) for h in (body.get("hashes") or [])[:5000]}
        have: set[str] = set()
        if body.get("chat"):
            sid = str(body["chat"])
            if not owns_upload(sid, request):
                raise HTTPException(404, "unknown interview session")
            purpose, cls = ("3d" if body.get("purpose") == "3d" else "training"), str(body.get("cls") or "")
            folder = home / "uploads" / sid / (f"3d_{slug(cls)}" if purpose == "3d" and cls else purpose)
            have = {content_key(f) for f in folder.iterdir() if f.is_file()} if folder.is_dir() else set()
        elif body.get("dataset"):
            owned(request, "datasets", str(body["dataset"]))
            have = {p["hash"] for p in lib.photos(str(body["dataset"]))}
        elif body.get("job"):
            snap = kept(job_of(str(body["job"]), request))
            lab = lib.labeling(snap["labeling"]) if snap else None
            have = {p["hash"] for p in lib.photos(lab["dataset"])} if lab else set()
        return {"known": sorted(hashes & have)}

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
        return answered(sid, act, turn_out(sid, guarded(sid, session.choose, [str(f) for f in body.get("files") or []],
                                                        str(body.get("class", "")), act=act, request=request)),
                        request)
