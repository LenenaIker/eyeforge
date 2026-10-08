"""Top-ups with Stripe (payments.py), against a fake Stripe."""
import hashlib
import hmac
import json
import time

import pytest

from sim2chip.accounts import Accounts
from sim2chip.payments import PaymentError, Payments, _flat

SECRET = "whsec_test"


class FakeStripe:
    def __init__(self):
        self.calls, self.refuse = [], None

    def __call__(self, method, url, secret, data=None, idempotency=None):
        path = url.split("/v1/", 1)[1]
        self.calls.append((method, path, data, idempotency))
        if path == "customers":
            return {"id": "cus_1"}
        if path == "checkout/sessions":
            return {"id": "cs_1", "url": "https://checkout.stripe.test/cs_1"}
        if path.startswith("payment_intents/"):
            return {"id": "pi_1", "payment_method": {"id": "pm_1", "card": {"brand": "visa", "last4": "4242"}}}
        if path == "payment_intents":
            if self.refuse:
                raise PaymentError(self.refuse)
            return {"id": f"pi_auto{len(self.calls)}", "status": "succeeded"}
        raise AssertionError(path)


@pytest.fixture
def pay(tmp_path):
    accs = Accounts(tmp_path / "a.db")
    acc = accs.login("google", "1", "alice@gmail.com", ["alice@gmail.com"], "Alice")
    stripe = FakeStripe()
    return Payments(accs, "https://eyeforge.test", "sk_test", SECRET, http=stripe), accs, acc, stripe


def signed(event: dict, t: int | None = None) -> tuple[bytes, str]:
    body = json.dumps(event).encode()
    t = t or int(time.time())
    sig = hmac.new(SECRET.encode(), f"{t}.".encode() + body, hashlib.sha256).hexdigest()
    return body, f"t={t},v1={sig}"


def paid(acc_id, usd="10.00", pi="pi_1"):
    return {"id": "evt_1", "type": "checkout.session.completed", "data": {"object": {
        "id": "cs_1", "payment_status": "paid", "payment_intent": pi,
        "metadata": {"account": acc_id, "usd": usd, "kind": "checkout"}}}}


def test_stripe_form_fields():
    assert _flat({"a": {"b": 1}, "c": [{"d": True}], "e": None}) == {"a[b]": "1", "c[0][d]": "true"}


def test_checkout_page_saves_the_card_for_later(pay):
    p, _, acc, stripe = pay
    assert p.checkout(acc, 10) == "https://checkout.stripe.test/cs_1"
    _, _, data, _ = stripe.calls[-1]
    item = data["line_items"][0]["price_data"]
    assert item["unit_amount"] == 1000 and item["currency"] == "usd"
    assert data["payment_intent_data"]["setup_future_usage"] == "off_session"
    assert data["metadata"] == {"account": acc.id, "usd": "10.00", "kind": "checkout"}
    p.checkout(acc, 5)
    assert [c[1] for c in stripe.calls].count("customers") == 1  # one Stripe customer per account
    with pytest.raises(PaymentError):
        p.checkout(acc, 1)


def test_a_payment_counts_once_and_only_when_signed(pay):
    p, accs, acc, _ = pay
    body, sig = signed(paid(acc.id))
    for _ in range(2):  # Stripe may send the same event twice
        p.handle(p.verify(body, sig))
    assert accs.balance(acc.id) == 10
    assert p.settings(acc.id)["card"] == "visa 4242"
    with pytest.raises(PaymentError):
        p.verify(body, sig.replace("v1=", "v1=0"))
    with pytest.raises(PaymentError):
        p.verify(*signed(paid(acc.id), t=int(time.time()) - 3600))
    with pytest.raises(PaymentError):
        p.verify(body, None)


def test_auto_top_up_charges_the_saved_card_once(pay):
    p, accs, acc, stripe = pay
    with pytest.raises(PaymentError):  # no card yet
        p.set_auto(acc.id, True, 2, 10)
    p.handle(paid(acc.id, "5.00"))
    p.set_auto(acc.id, True, 2, 10)
    assert p.maybe_auto(acc.id, 3.0) is None  # above the threshold
    assert p.maybe_auto(acc.id, 1.0) == "added"
    assert accs.balance(acc.id) == 15
    _, _, data, key = stripe.calls[-1]
    assert data["off_session"] is True and data["payment_method"] == "pm_1" and key.startswith("auto-")
    assert p.maybe_auto(acc.id, 1.0) is None  # not again right away
    stripe.refuse = "Your card was declined."
    assert p.maybe_auto(acc.id, 1.0, now=time.time() + 3600).startswith("refused")
    auto = p.settings(acc.id)["auto"]
    assert auto["on"] is False and auto["error"] == "Your card was declined."


def test_server_top_up_and_webhook(tmp_path):
    from test_accounts import fake_http, sign_in
    from fastapi.testclient import TestClient
    from sim2chip.auth import Auth
    from sim2chip.server import create_app
    auth = Auth("https://eyeforge.test", "x" * 32, {"google": ("gid", "gsec")}, http=fake_http)
    accs = Accounts(tmp_path / "accounts.db")
    stripe = FakeStripe()
    pays = Payments(accs, "https://eyeforge.test", "sk_test", SECRET, http=stripe)
    app = create_app(tmp_path, backend="nebius", public=True, auth=auth, accounts=accs, payments=pays)
    c = sign_in(app, "google", "alice")
    assert c.get("/api/billing").json()["topup"]["amounts"] == [5, 10, 25, 50]
    assert c.post("/api/billing/topup", json={"usd": 25}).json()["url"].startswith("https://checkout.stripe.test")
    acc = accs.by_email("alice@gmail.com")[0]
    body, sig = signed(paid(acc.id, "25.00"))
    anyone = TestClient(app, base_url="https://eyeforge.test")  # Stripe has no session
    assert anyone.post("/api/stripe/webhook", content=body, headers={"stripe-signature": "t=1,v1=x"}).status_code == 400
    assert anyone.post("/api/stripe/webhook", content=body, headers={"stripe-signature": sig}).status_code == 200
    assert c.get("/api/me").json()["account"]["balance"] == 25
