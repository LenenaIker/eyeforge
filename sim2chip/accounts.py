"""Accounts of a public server (login with Google or GitHub), their tier and their credits, in one SQLite file.

Tiers (docs/plan_login.md, user 2026-10-03):
  judge   free and unlimited until the judging ends (SIM2CHIP_JUDGE_UNTIL): an account with a verified email of
          the sponsors' or the organiser's domains, or one the owner marks by hand for a judge who wrote in
  normal  everything costs credits, the interview included; a new account has none
  banned  nothing works (the owner's hand brake against abuse)

The balance is in dollars (user, 2026-10-09). Things are charged by their use, not at a flat price: what a job,
a chat or a labeling really cost us (Token Factory tokens, Tavily searches, Nebius GPU time; usage.py) times
MARGIN, charged as it happens. Money in is a ledger of moves (grant, purchase); usage is one line per job or
chat that grows while it runs. Starting something only needs the balance to cover its usual price (START).
Run from the server with `sim2chip accounts ...` and `sim2chip credits ...`.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# Sponsor, organiser and judges' companies of the Nebius x NVIDIA hackathon: the rules want them free of charge.
JUDGE_DOMAINS = {"nvidia.com", "nebius.com", "tavily.com", "devpost.com"}
MARGIN = 2.0  # price = our cost x MARGIN: failed tries, the warm GPU's idle minutes, payment fees (user, 2026-10-09)
# Dollars the balance must hold to start each, about its usual price with the margin: a first Nebius run cost
# 0.56 $ (2026-09-27), a labeling a few GPU minutes, a chat a few cents of tokens.
START = {"run": 1.0, "label": 0.2, "chat": 0.05, "profile": 0.05}

SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    id TEXT PRIMARY KEY, provider TEXT NOT NULL, subject TEXT NOT NULL, email TEXT, name TEXT,
    judge TEXT, banned INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL, last_login REAL,
    UNIQUE (provider, subject));
CREATE TABLE IF NOT EXISTS credits (
    account TEXT NOT NULL, amount REAL NOT NULL, reason TEXT NOT NULL, detail TEXT, t REAL NOT NULL);
CREATE INDEX IF NOT EXISTS credits_account ON credits (account);
CREATE TABLE IF NOT EXISTS usage (
    account TEXT NOT NULL, key TEXT NOT NULL, reason TEXT NOT NULL, cost REAL NOT NULL, usd REAL NOT NULL,
    t REAL NOT NULL, PRIMARY KEY (account, key));
"""


@dataclass
class Account:
    id: str
    provider: str
    email: str | None
    name: str | None
    judge: str | None   # "domain" or "manual", or None
    banned: bool


def judge_domain(emails: list[str]) -> bool:
    return any(e.rsplit("@", 1)[-1].lower() in JUDGE_DOMAINS for e in emails if "@" in e)


class Accounts:
    def __init__(self, path: Path, judge_until: float | None = None):
        """judge_until: epoch seconds when the judge tier ends (None: it never does)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()
        self.judge_until = judge_until

    def _row(self, row) -> Account | None:
        if row is None:
            return None
        return Account(id=row[0], provider=row[1], email=row[2], name=row[3], judge=row[4], banned=bool(row[5]))

    def _select(self, where: str, *args) -> Account | None:
        return self._row(self.db.execute(
            f"SELECT id, provider, email, name, judge, banned FROM accounts WHERE {where}", args).fetchone())

    def login(self, provider: str, subject: str, email: str | None, verified: list[str],
              name: str | None = None) -> Account:
        """The account of a provider's user, made on the first login. A verified email of a judge domain makes
        it a judge (checked at every login: an email verified later counts too); never the other way round."""
        with self.lock:
            acc = self._select("provider = ? AND subject = ?", provider, subject)
            now = time.time()
            if acc is None:
                acc_id = f"{provider}:{subject}"
                self.db.execute("INSERT INTO accounts (id, provider, subject, email, name, created) "
                                "VALUES (?, ?, ?, ?, ?, ?)", (acc_id, provider, subject, email, name, now))
            else:
                acc_id = acc.id
            self.db.execute("UPDATE accounts SET email = COALESCE(?, email), name = COALESCE(?, name), "
                            "last_login = ? WHERE id = ?", (email, name, now, acc_id))
            if judge_domain(verified):
                self.db.execute("UPDATE accounts SET judge = 'domain' WHERE id = ? AND judge IS NULL", (acc_id,))
            return self._select("id = ?", acc_id)

    def get(self, acc_id: str) -> Account | None:
        return self._select("id = ?", acc_id)

    def by_email(self, email: str) -> list[Account]:
        rows = self.db.execute("SELECT id, provider, email, name, judge, banned FROM accounts "
                               "WHERE lower(email) = lower(?)", (email,)).fetchall()
        return [self._row(r) for r in rows]

    def all(self) -> list[Account]:
        rows = self.db.execute("SELECT id, provider, email, name, judge, banned FROM accounts "
                               "ORDER BY created").fetchall()
        return [self._row(r) for r in rows]

    def tier(self, acc: Account, now: float | None = None) -> str:
        if acc.banned:
            return "banned"
        if acc.judge and (self.judge_until is None or (now or time.time()) < self.judge_until):
            return "judge"
        return "normal"

    def set_flag(self, email: str, column: str, value) -> int:
        """judge / banned for every account with this email; how many changed."""
        assert column in ("judge", "banned")
        with self.lock:
            return self.db.execute(f"UPDATE accounts SET {column} = ? WHERE lower(email) = lower(?)",
                                   (value, email)).rowcount

    def balance(self, acc_id: str) -> float:
        got = self.db.execute("SELECT COALESCE(SUM(amount), 0) FROM credits WHERE account = ?", (acc_id,)).fetchone()[0]
        used = self.db.execute("SELECT COALESCE(SUM(usd), 0) FROM usage WHERE account = ?", (acc_id,)).fetchone()[0]
        return round(got - used, 4)

    def moves(self, acc_id: str, limit: int = 200) -> list[dict]:
        """The latest moves, newest first: money in, and what each job or chat has used (still growing while it
        runs)."""
        rows = self.db.execute("SELECT amount, reason, detail, t FROM credits WHERE account = ? UNION ALL "
                               "SELECT -usd, reason, key, t FROM usage WHERE account = ? ORDER BY 4 DESC LIMIT ?",
                               (acc_id, acc_id, limit)).fetchall()
        return [{"amount": round(a, 4), "reason": r, "detail": d, "t": t} for a, r, d, t in rows]

    def add(self, acc_id: str, amount: float, reason: str, detail: str | None = None) -> None:
        with self.lock:
            self.db.execute("INSERT INTO credits VALUES (?, ?, ?, ?, ?)", (acc_id, amount, reason, detail, time.time()))

    def can_start(self, acc: Account, reason: str) -> bool:
        """May this account start a run, a labeling, a chat...: free for a judge, never when banned, else when
        the balance covers its usual price."""
        tier = self.tier(acc)
        return tier == "judge" or (tier == "normal" and self.balance(acc.id) + 1e-9 >= START[reason])

    def meter(self, acc_id: str, key: str, reason: str, cost: float) -> float:
        """What `key` (a job, a chat) has cost us so far, in dollars: its usage line becomes cost x MARGIN (free
        for a judge, whose cost is still kept). Usage only grows. Returns the balance."""
        acc = self.get(acc_id)
        if acc is None:
            return 0.0
        usd = 0.0 if self.tier(acc) == "judge" else round(cost * MARGIN, 4)
        with self.lock:
            self.db.execute("INSERT INTO usage VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (account, key) DO UPDATE SET "
                            "cost = excluded.cost, usd = MAX(usd, excluded.usd), t = excluded.t "
                            "WHERE excluded.cost > cost", (acc_id, key, reason, round(cost, 6), usd, time.time()))
        return self.balance(acc_id)
