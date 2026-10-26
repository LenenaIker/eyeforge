"""Credits, Stripe top-ups and closing an account."""
from __future__ import annotations

import asyncio
import shutil

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from ..accounts import MARGIN, START
from ..server import OWNER, SESSION, SESSION_OWNER
from ..workspace import Job


def add(app: FastAPI, ctx) -> None:
    accounts, payments, jobs, lib, contact, home = ctx.accounts, ctx.payments, ctx.jobs, ctx.lib, ctx.contact, ctx.home
    backend = ctx.backend

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
        from ..payments import PaymentError
        acc = paying(request)
        try:
            return {"url": payments.checkout(acc, float(body.get("usd") or 0))}
        except (PaymentError, ValueError) as e:
            raise HTTPException(400, str(e)) from None

    @app.put("/api/billing/auto")
    def auto_topup(body: dict, request: Request):
        from ..payments import PaymentError
        acc = paying(request)
        try:
            return payments.set_auto(acc.id, bool(body.get("on")), float(body.get("below", 2)),
                                     float(body.get("amount", 10)))
        except (PaymentError, ValueError) as e:
            raise HTTPException(400, str(e)) from None

    @app.post("/api/stripe/webhook", include_in_schema=False)
    async def stripe_webhook(request: Request):
        """Stripe telling us a payment went through: signed, each payment counted once."""
        from ..payments import PaymentError
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
        from ..payments import PaymentError
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
            from ..backends.nebius import NebiusBackend
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
