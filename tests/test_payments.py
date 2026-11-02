"""Top-ups with Stripe (payments.py), against a fake Stripe."""
import hashlib
import hmac
import json
import re
import sqlite3
import time
from pathlib import Path

import pytest

from sim2chip.accounts import MARGIN, Accounts
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
        if path == "refunds":
            return {"id": f"re_{len(self.calls)}", "status": "succeeded"}
        if path.startswith("customers/") and method == "DELETE":
            return {"id": path.split("/")[1], "deleted": True}
        if path == "payment_intents":
            if self.refuse:
                raise PaymentError(self.refuse)
            return {"id": f"pi_auto{len(self.calls)}", "status": "succeeded", "amount_received": data["amount"],
                    "currency": data["currency"]}
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
        "amount_subtotal": round(float(usd) * 100), "currency": "usd",
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
    assert data["allowed_payment_method_types"] == ["card"] and "payment_method_types" not in data
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


def test_an_account_being_closed_starts_nothing(tmp_path):
    from test_accounts import fake_http, new_job, sign_in
    from sim2chip.auth import Auth
    from sim2chip.server import create_app
    auth = Auth("https://eyeforge.test", "x" * 32, {"google": ("gid", "gsec")}, http=fake_http)
    accs = Accounts(tmp_path / "accounts.db")
    stripe, seen = FakeStripe(), []
    pays = Payments(accs, "https://eyeforge.test", "sk_test", SECRET, http=stripe)
    app = create_app(tmp_path, backend="nebius", public=True, auth=auth, accounts=accs, payments=pays)
    c = sign_in(app, "google", "alice")
    acc = accs.by_email("alice@gmail.com")[0]
    pays.checkout(acc, 10)
    pays.handle(paid(acc.id))

    def http(method, url, secret, data=None, idempotency=None):
        if url.endswith("/refunds"):  # half way through the closing, from another tab
            seen.append(new_job(c).status_code)
            seen.append(c.post("/api/account/delete", json={"confirm": "delete"}).status_code)
        return stripe(method, url, secret, data, idempotency)
    pays.http = http
    assert c.post("/api/account/delete", json={"confirm": "delete"}).json()["refunded"] == 10
    assert seen == [409, 409]


def refunded(pi="pi_1", cents=1000, paid_cents=1000):
    return {"id": "evt_r", "type": "charge.refunded", "data": {"object": {
        "id": "ch_1", "payment_intent": pi, "amount_refunded": cents, "amount": paid_cents}}}


def test_refund_takes_the_money_back_once(pay):
    p, accs, acc, _ = pay
    p.handle(paid(acc.id))
    assert p.handle(refunded(cents=400)) == "refunded 4.00"   # partial
    assert accs.balance(acc.id) == pytest.approx(6)
    assert p.handle(refunded(cents=400)) == "already refunded"  # told twice
    assert p.handle(refunded(cents=1000)) == "refunded 6.00"   # the rest: Stripe sends the running total
    assert accs.balance(acc.id) == pytest.approx(0)
    assert p.handle(refunded(pi="pi_other")) == "refund of an unknown payment"


def test_closing_refunds_the_paid_balance_left_and_drops_the_card(pay):
    p, accs, acc, stripe = pay
    p.checkout(acc, 10)  # makes its Stripe customer
    p.handle(paid(acc.id, "10.00", "pi_old"))
    p.handle(paid(acc.id, "5.00", "pi_new"))
    accs.add(acc.id, 3, "grant")  # given, not paid: counted as spent first, never refunded beyond what was paid
    accs.meter(acc.id, "job1", "run", 6.0)  # 7.8 $ used at the 1.3x margin: 10.2 $ left
    stripe.calls.clear()
    assert p.close(acc.id) == pytest.approx(10.2)
    refunds = [(c[2]["payment_intent"], c[2]["amount"]) for c in stripe.calls if c[1] == "refunds"]
    assert refunds == [("pi_new", 500), ("pi_old", 520)]  # newest first
    assert ("DELETE", "customers/cus_1", None, None) in stripe.calls
    assert p.handle(refunded("pi_new", 500, 500)) == "already refunded"  # its webhook comes later
    assert accs.balance(acc.id) == pytest.approx(0)


def test_a_payment_that_clears_later_counts_when_it_clears_and_once(pay):
    p, accs, acc, _ = pay
    event = paid(acc.id)
    event["data"]["object"]["payment_status"] = "unpaid"
    assert p.handle(event) == "ignored"
    assert accs.balance(acc.id) == 0
    event["type"], event["data"]["object"]["payment_status"] = "checkout.session.async_payment_succeeded", "paid"
    assert p.handle(event) == "added"
    assert p.handle(event) == "already added"
    assert accs.balance(acc.id) == 10


def test_a_charge_that_is_not_the_top_up_adds_nothing(pay):
    p, accs, acc, _ = pay
    for field, value in (("currency", "eur"), ("amount_subtotal", 500)):
        event = paid(acc.id)
        event["data"]["object"][field] = value
        assert p.handle(event).startswith("PAID BUT NOT ADDED")
    assert accs.balance(acc.id) == 0


def test_a_payment_for_a_closed_account_is_flagged(pay):
    p, accs, acc, _ = pay
    accs.forget(acc.id)
    assert p.handle(paid(acc.id)).startswith("PAID BUT THE ACCOUNT IS GONE")


def test_auto_top_up_checks_the_webhook_amount_too(pay):
    p, accs, acc, _ = pay
    event = {"type": "payment_intent.succeeded", "data": {"object": {
        "id": "pi_a", "amount_received": 1000, "currency": "usd",
        "metadata": {"account": acc.id, "usd": "10.00", "kind": "auto"}}}}
    assert p.handle(event) == "added"
    assert accs.balance(acc.id) == 10


def test_refund_with_vat_takes_only_what_was_added(pay):
    p, accs, acc, _ = pay
    p.handle(paid(acc.id))                                        # 10 $ added, 12.10 $ charged with VAT
    assert p.handle(refunded(cents=605, paid_cents=1210)) == "refunded 5.00"
    assert p.handle(refunded(cents=1210, paid_cents=1210)) == "refunded 5.00"
    assert accs.balance(acc.id) == pytest.approx(0)


def test_a_failed_refund_is_flagged_for_the_owner(pay):
    p, _, _, _ = pay
    done = p.handle({"type": "refund.failed", "data": {"object": {
        "id": "re_1", "payment_intent": "pi_1", "amount": 500, "failure_reason": "expired_or_canceled_card"}}})
    assert done.startswith("REFUND FAILED") and "re_1" in done


def dispute(what, status="needs_response", cents=1000, pi="pi_1", id="du_1", currency="usd"):
    return {"type": f"charge.dispute.{what}", "data": {"object": {
        "id": id, "payment_intent": pi, "amount": cents, "currency": currency, "status": status}}}


def test_a_dispute_stops_auto_top_up_but_an_inquiry_moves_no_money(pay):
    p, accs, acc, _ = pay
    p.handle(paid(acc.id))
    p.set_auto(acc.id, True, 2, 10)
    p.handle(dispute("created", "warning_needs_response"))
    assert accs.balance(acc.id) == 10
    auto = p.settings(acc.id)["auto"]
    assert auto["on"] is False and auto["error"] == "a payment was disputed"
    p.handle(dispute("closed", "warning_closed"))
    assert accs.balance(acc.id) == 10


def test_a_dispute_takes_the_money_once_and_a_win_returns_it_once(pay):
    p, accs, acc, _ = pay
    p.handle(paid(acc.id))
    accs.meter(acc.id, "job1", "run", 5.0)  # 6.5 $ used: the balance goes below zero, that blocks new work
    assert "10.00 taken" in p.handle(dispute("funds_withdrawn"))
    assert "already taken" in p.handle(dispute("funds_withdrawn"))
    assert accs.balance(acc.id) == pytest.approx(-6.5)
    assert "10.00 returned" in p.handle(dispute("funds_reinstated", "won"))
    assert "already returned" in p.handle(dispute("closed", "won"))
    assert accs.balance(acc.id) == pytest.approx(3.5)


def test_a_lost_dispute_returns_nothing(pay):
    p, accs, acc, _ = pay
    p.handle(paid(acc.id))
    p.handle(dispute("funds_withdrawn"))
    p.handle(dispute("funds_reinstated", "lost", cents=400))  # a partly refunded payment: not a win
    p.handle(dispute("closed", "lost"))
    assert accs.balance(acc.id) == pytest.approx(0)


def test_a_dispute_never_takes_more_than_is_left_of_the_payment(pay):
    p, accs, acc, _ = pay
    p.handle(paid(acc.id))
    p.handle(refunded(cents=400))
    assert "6.00 taken" in p.handle(dispute("funds_withdrawn", id="du_1"))
    assert "0.00 taken" in p.handle(dispute("funds_withdrawn", id="du_2"))  # two disputes on one payment
    assert accs.balance(acc.id) == pytest.approx(0)


def test_a_dispute_in_another_currency_takes_what_is_left(pay):
    p, accs, acc, _ = pay
    p.handle(paid(acc.id))
    p.handle(dispute("funds_withdrawn", cents=90000, currency="jpy"))
    assert accs.balance(acc.id) == pytest.approx(0)


def test_a_dispute_of_a_payment_we_do_not_know_changes_nothing(pay):
    p, accs, acc, _ = pay
    assert "unknown payment" in p.handle(dispute("funds_withdrawn"))
    assert "unknown payment" in p.handle(dispute("funds_withdrawn", pi=None))
    assert accs.balance(acc.id) == 0


def test_closing_skips_a_disputed_payment_and_refunds_the_others(pay):
    p, accs, acc, stripe = pay
    p.checkout(acc, 5)
    p.handle(paid(acc.id, "5.00", "pi_old"))
    p.handle(paid(acc.id, "10.00", "pi_new"))
    p.handle(dispute("funds_withdrawn", pi="pi_new"))
    stripe.calls.clear()
    assert p.close(acc.id) == pytest.approx(5)
    assert [c[2]["payment_intent"] for c in stripe.calls if c[1] == "refunds"] == ["pi_old"]


def test_closing_waits_when_a_dispute_holds_back_money_it_cannot_return(pay):
    p, accs, acc, stripe = pay
    p.checkout(acc, 10)
    p.handle(paid(acc.id))
    p.handle(dispute("created", "warning_needs_response"))  # an inquiry: the money is still in the balance
    stripe.calls.clear()
    with pytest.raises(PaymentError, match="under dispute"):
        p.close(acc.id)
    assert not [c for c in stripe.calls if c[1] == "refunds" or c[0] == "DELETE"]
    assert accs.balance(acc.id) == 10
    p.handle(dispute("closed", "warning_closed"))  # settled: it can go back now
    assert p.close(acc.id) == 10


def test_a_lost_dispute_does_not_keep_an_account_open(pay):
    p, accs, acc, stripe = pay
    p.checkout(acc, 10)
    p.handle(paid(acc.id))
    accs.add(acc.id, 3, "grant")  # given, not paid: nothing to send back
    p.handle(dispute("funds_withdrawn"))
    p.handle(dispute("closed", "lost"))
    stripe.calls.clear()
    assert p.close(acc.id) == 0
    assert not [c for c in stripe.calls if c[1] == "refunds"]


def test_closing_refunds_a_payment_whose_dispute_we_won(pay):
    p, accs, acc, stripe = pay
    p.checkout(acc, 10)
    p.handle(paid(acc.id))
    p.handle(dispute("funds_withdrawn"))
    p.handle(dispute("closed", "won"))
    assert p.close(acc.id) == 10


def test_closing_turns_auto_top_up_off_and_a_refusal_puts_it_back(pay):
    p, accs, acc, stripe = pay
    p.checkout(acc, 10)
    p.handle(paid(acc.id))
    p.set_auto(acc.id, True, 2, 10)
    real, charged = stripe.__call__, []

    def refuse(method, url, secret, data=None, idempotency=None):
        charged.append(p.maybe_auto(acc.id, 1.0))  # the balance is low now, while it closes
        raise PaymentError("refused")
    p.http = refuse
    with pytest.raises(PaymentError):
        p.close(acc.id)
    assert charged == [None]
    assert p.settings(acc.id)["auto"]["on"] is True
    p.http = real
    assert p.close(acc.id) == 10


@pytest.mark.parametrize("webhook_first", [True, False])
def test_closing_and_the_refund_webhook_count_once_in_either_order(pay, webhook_first):
    p, accs, acc, stripe = pay
    p.checkout(acc, 10)
    p.handle(paid(acc.id))
    real = stripe.__call__

    def http(method, url, secret, data=None, idempotency=None):
        assert not accs.lock.locked()  # Stripe's answer must not stop everyone else's usage
        out = real(method, url, secret, data, idempotency)
        if url.endswith("/refunds") and webhook_first:
            p.handle(refunded(cents=data["amount"]))
        return out
    p.http = http
    assert p.close(acc.id) == 10
    assert accs.balance(acc.id) == pytest.approx(0)
    assert p.handle(refunded(cents=1000)) == "already refunded"
    assert accs.balance(acc.id) == pytest.approx(0)


def test_the_ledger_refuses_a_payment_twice_but_not_a_second_partial_refund(pay):
    _, accs, acc, _ = pay
    insert = "INSERT INTO credits VALUES (?, ?, ?, ?, 0)"
    accs.db.execute(insert, (acc.id, 5, "purchase", "pi_1"))
    with pytest.raises(sqlite3.IntegrityError):
        accs.db.execute(insert, (acc.id, 5, "purchase", "pi_1"))
    for amount in (-1, -2):
        accs.db.execute(insert, (acc.id, amount, "refund", "refund pi_1"))
    assert accs.balance(acc.id) == 2


def test_a_ledger_with_a_payment_twice_still_starts(tmp_path, capsys):
    db = sqlite3.connect(tmp_path / "a.db")
    db.execute("CREATE TABLE credits (account TEXT NOT NULL, amount REAL NOT NULL, reason TEXT NOT NULL, detail TEXT, "
               "t REAL NOT NULL)")
    for _ in range(2):
        db.execute("INSERT INTO credits VALUES ('a', 5, 'purchase', 'pi_1', 0)")
    db.commit()
    db.close()
    assert Accounts(tmp_path / "a.db").balance("a") == 10
    assert "TWICE" in capsys.readouterr().out


def test_the_terms_state_the_margin_we_charge():
    terms = (Path(__file__).parents[1] / "web" / "src" / "pages" / "Legal.tsx").read_text()
    assert [int(n) for n in re.findall(r"plus (\d+) %", terms)] == [round((MARGIN - 1) * 100)]
    assert "twice" not in terms
