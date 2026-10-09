"""Top-ups of the balance with Stripe (user, 2026-10-09), stdlib only like auth.py.

Configuration (environment or .env):
    STRIPE_SECRET_KEY        sk_test_... (test mode) or sk_live_...
    STRIPE_WEBHOOK_SECRET    whsec_... of the webhook endpoint <public url>/api/stripe/webhook
    SIM2CHIP_STRIPE_TAX      1: Stripe Tax adds the buyer's VAT on the checkout page (set it up in Stripe first)
Without the first two there are no top-ups. Setup steps in docs/pagos.md.

- A top-up is a Stripe Checkout page: Stripe shows the card form, no card data reaches this server. The card is
  saved for later (setup_future_usage), so an auto top-up can charge it while the user is away.
- Money reaches the balance only through payments Stripe confirms: the signed webhook (checkout.session.completed,
  payment_intent.succeeded) or the answer to an auto top-up. Each payment counts once (its id is the move's detail).
- A refund made on the Stripe dashboard (charge.refunded) takes that money back from the balance, which may go
  below zero when it was already used.
- Auto top-up: when the balance falls below the account's threshold, its amount is charged to the saved card,
  at most once every AUTO_GAP_S. A refused card switches auto top-up off and says why. With Stripe Tax on it is
  not offered: a payment made without the checkout page would skip the VAT.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from . import config

API = "https://api.stripe.com/v1"
AMOUNTS = [5, 10, 25, 50]  # dollars offered on the Billing page
MIN_USD, MAX_USD = 5, 500  # a top-up, also an auto one
AUTO_GAP_S = 600           # one auto top-up per account in this time, whatever the meter says meanwhile
SIGNATURE_AGE_S = 300      # webhook calls older than this are refused (replays)

SCHEMA = """
CREATE TABLE IF NOT EXISTS stripe (
    account TEXT PRIMARY KEY, customer TEXT, payment_method TEXT, card TEXT, auto_on INTEGER NOT NULL DEFAULT 0,
    auto_below REAL NOT NULL DEFAULT 2, auto_amount REAL NOT NULL DEFAULT 10, auto_error TEXT, last_auto REAL);
"""


class PaymentError(Exception):
    pass


def _flat(data: dict, prefix: str = "") -> dict:
    """Nested data as Stripe's form fields: {"a": {"b": 1}, "c": [{"d": 2}]} -> a[b]=1, c[0][d]=2."""
    out = {}
    for k, v in data.items():
        key = f"{prefix}[{k}]" if prefix else str(k)
        if isinstance(v, dict):
            out |= _flat(v, key)
        elif isinstance(v, list):
            out |= _flat({str(i): x for i, x in enumerate(v)}, key)
        elif isinstance(v, bool):
            out[key] = "true" if v else "false"
        elif v is not None:
            out[key] = str(v)
    return out


def _http(method: str, url: str, secret: str, data: dict | None = None, idempotency: str | None = None) -> dict:
    headers = {"Authorization": f"Bearer {secret}", "User-Agent": "EyeForge"}
    if idempotency:
        headers["Idempotency-Key"] = idempotency
    body = urllib.parse.urlencode(_flat(data)).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:  # Stripe says why: a refused card, a bad request
        try:
            err = json.loads(e.read()).get("error", {})
        except ValueError:
            err = {}
        raise PaymentError(err.get("message") or f"Stripe answered {e.code}") from None
    except Exception as e:
        raise PaymentError(f"Stripe: {e}") from None


def configured() -> bool:
    return bool(config.env("STRIPE_SECRET_KEY") and config.env("STRIPE_WEBHOOK_SECRET"))


class Payments:
    def __init__(self, accounts, public_url: str, secret: str, webhook_secret: str, tax: bool = False, http=_http):
        """accounts: accounts.Accounts (its database also keeps each account's Stripe customer and settings).
        http: (method, url, secret key, form data, idempotency key) -> JSON."""
        self.accounts, self.public_url = accounts, public_url.rstrip("/")
        self.secret, self.webhook_secret, self.tax, self.http = secret, webhook_secret, tax, http
        self.db = accounts.db
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()

    @classmethod
    def from_env(cls, accounts, public_url: str) -> Payments | None:
        if not configured():
            return None
        return cls(accounts, public_url, config.env("STRIPE_SECRET_KEY"), config.env("STRIPE_WEBHOOK_SECRET"),
                   tax=config.env("SIM2CHIP_STRIPE_TAX", "") == "1")

    def _call(self, method: str, path: str, data: dict | None = None, idempotency: str | None = None) -> dict:
        return self.http(method, f"{API}/{path}", self.secret, data, idempotency)

    def _row(self, acc_id: str) -> dict:
        self.db.execute("INSERT OR IGNORE INTO stripe (account) VALUES (?)", (acc_id,))
        cur = self.db.execute("SELECT * FROM stripe WHERE account = ?", (acc_id,))
        return dict(zip([c[0] for c in cur.description], cur.fetchone()))

    def settings(self, acc_id: str) -> dict:
        """What the Billing page shows: the saved card and the auto top-up."""
        r = self._row(acc_id)
        return {"enabled": True, "amounts": AMOUNTS, "min": MIN_USD, "max": MAX_USD, "tax": self.tax,
                "card": r["card"], "auto": {"available": not self.tax, "on": bool(r["auto_on"]),
                                            "below": r["auto_below"], "amount": r["auto_amount"],
                                            "error": r["auto_error"]}}

    # Paying on the checkout page --------------------------------------------------------------------

    def _customer(self, acc) -> str:
        r = self._row(acc.id)
        if r["customer"]:
            return r["customer"]
        c = self._call("POST", "customers", {"email": acc.email, "name": acc.name, "metadata": {"account": acc.id}},
                       idempotency=f"customer-{acc.id}")
        self.db.execute("UPDATE stripe SET customer = ? WHERE account = ?", (c["id"], acc.id))
        return c["id"]

    def checkout(self, acc, usd: float) -> str:
        """The address of a Stripe Checkout page that adds `usd` to the balance once paid."""
        if not MIN_USD <= usd <= MAX_USD:
            raise PaymentError(f"a top-up is between ${MIN_USD} and ${MAX_USD}")
        cents = round(usd * 100)
        meta = {"account": acc.id, "usd": f"{cents / 100:.2f}", "kind": "checkout"}
        data = {"mode": "payment", "customer": self._customer(acc), "client_reference_id": acc.id,
                "line_items": [{"quantity": 1, "price_data": {
                    "currency": "usd", "unit_amount": cents, "tax_behavior": "exclusive" if self.tax else None,
                    "product_data": {"name": "EyeForge balance",
                                     "description": "Pay-as-you-go balance for models, labeling and chats"}}}],
                "payment_intent_data": {"setup_future_usage": "off_session", "metadata": meta},
                "metadata": meta,
                "success_url": f"{self.public_url}/app/billing?paid={{CHECKOUT_SESSION_ID}}",
                "cancel_url": f"{self.public_url}/app/billing"}
        if self.tax:
            data |= {"automatic_tax": {"enabled": True}, "customer_update": {"address": "auto", "name": "auto"},
                     "tax_id_collection": {"enabled": True}}
        return self._call("POST", "checkout/sessions", data)["url"]

    # Stripe telling us a payment went through ---------------------------------------------------------

    def verify(self, payload: bytes, header: str | None, now: float | None = None) -> dict:
        """The webhook's event, if Stripe signed it (Stripe-Signature: t=<time>,v1=<HMAC-SHA256 of t.payload>)."""
        pairs = [p.strip().split("=", 1) for p in (header or "").split(",") if "=" in p]
        signed = [v for k, v in pairs if k == "v1"]
        try:
            t = int(next(v for k, v in pairs if k == "t"))
        except (StopIteration, ValueError):
            raise PaymentError("unsigned webhook call") from None
        want = hmac.new(self.webhook_secret.encode(), f"{t}.".encode() + payload, hashlib.sha256).hexdigest()
        if not any(hmac.compare_digest(want, s) for s in signed):
            raise PaymentError("bad webhook signature")
        if abs((now or time.time()) - t) > SIGNATURE_AGE_S:
            raise PaymentError("old webhook call")
        return json.loads(payload)

    def handle(self, event: dict) -> str:
        """Act on a verified event; says what it did (for the log)."""
        obj = event.get("data", {}).get("object", {})
        if event.get("type") == "checkout.session.completed" and obj.get("payment_status") == "paid":
            meta = obj.get("metadata") or {}
            done = self._credit(meta, obj.get("payment_intent") or obj["id"])
            if obj.get("payment_intent"):
                self._keep_card(meta.get("account"), obj["payment_intent"])
            return done
        if event.get("type") == "payment_intent.succeeded" and (obj.get("metadata") or {}).get("kind") == "auto":
            return self._credit(obj["metadata"], obj["id"])
        if event.get("type") == "charge.refunded" and obj.get("payment_intent"):
            return self._refund(obj["payment_intent"], obj.get("amount_refunded") or 0)
        return "ignored"

    def _refund(self, payment: str, cents_refunded: int) -> str:
        """Take back what was refunded of `payment`. Stripe sends the total refunded so far, so a second partial
        refund (or the same event twice) only takes back the difference."""
        detail = f"refund {payment}"
        with self.accounts.lock:
            row = self.db.execute("SELECT account FROM credits WHERE detail = ? AND amount > 0", (payment,)).fetchone()
            if row is None:
                return "refund of an unknown payment"
            taken = -self.db.execute("SELECT COALESCE(SUM(amount), 0) FROM credits WHERE account = ? AND detail = ?",
                                     (row[0], detail)).fetchone()[0]
            more = round(cents_refunded / 100 - taken, 2)
            if more <= 0:
                return "already refunded"
            self.db.execute("INSERT INTO credits VALUES (?, ?, ?, ?, ?)", (row[0], -more, "refund", detail, time.time()))
        return f"refunded {more:.2f}"

    def _credit(self, meta: dict, payment: str) -> str:
        acc_id, usd = meta.get("account"), float(meta.get("usd") or 0)
        if not acc_id or usd <= 0 or self.accounts.get(acc_id) is None:
            return "no account"
        reason = "auto top-up" if meta.get("kind") == "auto" else "purchase"
        return "added" if self.accounts.add_once(acc_id, usd, reason, payment) else "already added"

    def _keep_card(self, acc_id: str | None, payment_intent: str) -> None:
        """The card of a checkout payment, for auto top-ups; shown as "visa 4242"."""
        if not acc_id:
            return
        try:
            pi = self._call("GET", f"payment_intents/{payment_intent}?expand[]=payment_method")
        except PaymentError as e:
            print(f"stripe: could not read the card of {payment_intent}: {e}", flush=True)
            return
        pm = pi.get("payment_method") or {}
        card = pm.get("card") or {}
        if pm.get("id"):
            self._row(acc_id)
            self.db.execute("UPDATE stripe SET payment_method = ?, card = ?, auto_error = NULL WHERE account = ?",
                            (pm["id"], f"{card.get('brand', 'card')} {card.get('last4', '')}".strip(), acc_id))

    # Auto top-up ----------------------------------------------------------------------------------------

    def set_auto(self, acc_id: str, on: bool, below: float, amount: float) -> dict:
        r = self._row(acc_id)
        if on and self.tax:
            raise PaymentError("auto top-up is not available with VAT on the checkout page")
        if on and not r["payment_method"]:
            raise PaymentError("top up once by card first: auto top-up uses that card")
        if not (0 <= below <= MAX_USD and MIN_USD <= amount <= MAX_USD):
            raise PaymentError(f"the threshold is up to ${MAX_USD} and the amount between ${MIN_USD} and ${MAX_USD}")
        self.db.execute("UPDATE stripe SET auto_on = ?, auto_below = ?, auto_amount = ?, auto_error = NULL "
                        "WHERE account = ?", (int(on), round(below, 2), round(amount, 2), acc_id))
        return self.settings(acc_id)

    def maybe_auto(self, acc_id: str, balance: float, now: float | None = None) -> str | None:
        """After usage was charged: below the threshold, charge the saved card (at most once per AUTO_GAP_S).
        Returns what happened, None when nothing was due."""
        now = now or time.time()
        with self.lock:
            r = self._row(acc_id)
            if not r["auto_on"] or self.tax or not r["payment_method"] or balance >= r["auto_below"] \
                    or now - (r["last_auto"] or 0) < AUTO_GAP_S:
                return None
            self.db.execute("UPDATE stripe SET last_auto = ? WHERE account = ?", (now, acc_id))
        cents = round(r["auto_amount"] * 100)
        meta = {"account": acc_id, "usd": f"{cents / 100:.2f}", "kind": "auto"}
        try:
            pi = self._call("POST", "payment_intents", {
                "amount": cents, "currency": "usd", "customer": r["customer"], "payment_method": r["payment_method"],
                "off_session": True, "confirm": True, "metadata": meta,
                "description": "EyeForge balance, auto top-up"}, idempotency=f"auto-{acc_id}-{r['payment_method']}-{int(now // AUTO_GAP_S)}")
        except PaymentError as e:  # refused or needs the user (3-D Secure): off until they act
            self.db.execute("UPDATE stripe SET auto_on = 0, auto_error = ? WHERE account = ?", (str(e)[:200], acc_id))
            return f"refused: {e}"
        if pi.get("status") == "succeeded":
            return self._credit(meta, pi["id"])
        self.db.execute("UPDATE stripe SET auto_on = 0, auto_error = ? WHERE account = ?",
                        (f"the payment is {pi.get('status')}; top up by hand", acc_id))
        return f"not paid: {pi.get('status')}"
