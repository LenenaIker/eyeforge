import { CreditCard } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { getBilling, setAutoTopup, startTopup, usd, type Billing as BillingData, type Topup } from "../api";
import { Badge, Button, Card, Notice, cx } from "../components/ui";

// The balance in dollars, how things are priced (by use) and every move: money in, and what each model, chat or
// labeling used (sim2chip/accounts.py).

const REASONS: Record<string, string> = {
  grant: "Balance given", purchase: "Top-up", "auto top-up": "Auto top-up", run: "Model", label: "Labeling", chat: "Chat", profile: "Dataset look",
  interview: "Chat", refund: "Refund",
};
// What a usage line is about: "job:<name>", "chat:<session>", "profile:<dataset>_<time>".
// Money in carries Stripe's payment id, which says nothing to the user.
const about = (d: string | null) => !d ? null : d.startsWith("job:") ? d.slice(4).replace(/_/g, " ")
  : /^(chat|profile):|^(pi|cs)_/.test(d) ? null : d.replace(/_/g, " ");
const small = "h-10 w-20 rounded-[var(--radius-control)] border border-line px-3 outline-none focus:border-brand-500";
const when = (t: number) => new Date(t * 1000).toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });

export default function Billing() {
  const [b, setB] = useState<BillingData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [params, setParams] = useSearchParams();
  const back = params.get("paid");  // back from Stripe's page: the balance grows when Stripe tells us
  const load = useCallback(() => getBilling().then(setB).catch((e) => setError(e.message)), []);
  useEffect(() => { load(); }, [load]);
  useEffect(() => {
    if (!back) return;
    const t = [2000, 5000, 10000].map((ms) => setTimeout(load, ms));
    const done = setTimeout(() => setParams({}, { replace: true }), 12000);
    return () => { t.forEach(clearTimeout); clearTimeout(done); };
  }, [back, load, setParams]);

  return (
    <div className="space-y-6 py-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Billing</h1>
        <p className="mt-1 text-sm text-muted">You pay for what you use, as you use it.</p>
      </div>
      {error && <Notice tone="bad">{error}</Notice>}
      {back && <Notice>Payment received. Your balance updates in a few seconds.</Notice>}
      {b && !b.enabled && <Notice>This server has no accounts, so nothing is charged here.</Notice>}
      {b && (
        <div className="grid gap-4 sm:grid-cols-2">
          <Card className="p-5">
            <div className="text-sm text-muted">Balance</div>
            <div className="mt-1 flex items-baseline gap-2">
              {b.tier === "judge" ? <><span className="text-3xl font-semibold tracking-tight">Free</span><Badge tone="brand">Judge</Badge></>
                : <span className="text-3xl font-semibold tracking-tight">{b.balance != null ? usd(b.balance) : "-"}</span>}
            </div>
            {b.tier === "normal" && (b.topup ? <TopUp t={b.topup} />
              : <Button className="mt-4 w-full" disabled title="Not on this server"><CreditCard className="h-4 w-4" /> Top up</Button>)}
          </Card>
          <Card className="p-5">
            <div className="text-sm text-muted">Prices</div>
            <p className="mt-2 text-sm">Each model, chat or labeling costs what it used: cloud GPU time on Nebius and
              AI model tokens, at {b.costs.margin}x what they cost us. It is charged as it runs and listed below.</p>
            <dl className="mt-3 space-y-2 text-sm">
              <div className="flex justify-between"><dt>A model, usually</dt><dd className="font-medium">about {usd(b.costs.start.run)}</dd></div>
              <div className="flex justify-between"><dt>Labeling a dataset</dt><dd className="font-medium">about {usd(b.costs.start.label)}</dd></div>
              <div className="flex justify-between"><dt>A chat</dt><dd className="font-medium">a few cents</dd></div>
            </dl>
          </Card>
        </div>
      )}
      {b?.topup && b.tier === "normal" && <AutoTopUp t={b.topup} onSaved={(t) => setB({ ...b, topup: t })} />}
      {b?.moves && (
        <Card>
          <div className="border-b border-line px-5 py-3 font-medium">History</div>
          {b.moves.length === 0 ? <p className="px-5 py-6 text-sm text-muted">Nothing yet.</p> : (
            <div className="divide-y divide-line">
              {b.moves.map((m, i) => (
                <div key={i} className="flex items-center justify-between gap-3 px-5 py-3 text-sm">
                  <div className="min-w-0">
                    <div className="font-medium">{REASONS[m.reason] ?? m.reason}</div>
                    <div className="truncate text-muted">{[about(m.detail), when(m.t)].filter(Boolean).join(" · ")}</div>
                  </div>
                  <span className={m.amount < 0 ? "text-ink" : "text-green-700"}>{m.amount > 0 ? "+" : ""}{usd(m.amount)}</span>
                </div>
              ))}
            </div>
          )}
        </Card>
      )}
    </div>
  );
}

// Pick an amount and pay it on Stripe's page.
function TopUp({ t }: { t: Topup }) {
  const [amount, setAmount] = useState(t.amounts[1] ?? t.amounts[0]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const go = async () => {
    setBusy(true);
    try {
      window.location.href = (await startTopup(amount)).url;
    } catch (e: any) {
      setError(e.message);
      setBusy(false);
    }
  };
  return (
    <div className="mt-4 space-y-3">
      <div className="grid grid-cols-4 gap-2">
        {t.amounts.map((a) => (
          <button key={a} onClick={() => setAmount(a)}
                  className={cx("h-10 rounded-[var(--radius-control)] border text-sm font-medium",
                                amount === a ? "border-brand-500 bg-brand-50 text-brand-700" : "border-line hover:border-brand-500")}>
            ${a}
          </button>
        ))}
      </div>
      {error && <Notice tone="bad">{error}</Notice>}
      <Button className="w-full" onClick={go} disabled={busy}>
        <CreditCard className="h-4 w-4" /> {busy ? "Opening Stripe..." : `Add ${usd(amount)}`}
      </Button>
      <p className="text-xs text-muted">Paid by card on Stripe{t.tax ? ", VAT added for your country" : ""}. Your card is kept by Stripe for auto top-ups, never by us.</p>
    </div>
  );
}

// Top up by itself with the saved card when the balance runs low, so a long run never stops halfway.
function AutoTopUp({ t, onSaved }: { t: Topup; onSaved: (t: Topup) => void }) {
  const [below, setBelow] = useState(String(t.auto.below));
  const [amount, setAmount] = useState(String(t.auto.amount));
  const [error, setError] = useState<string | null>(null);
  if (!t.auto.available) return null;
  const save = (on: boolean) => setAutoTopup(on, Number(below), Number(amount)).then((r) => { setError(null); onSaved(r); })
    .catch((e) => setError(e.message));
  return (
    <Card className="space-y-3 p-5">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="font-medium">Auto top-up</div>
          <p className="text-sm text-muted">{t.card ? `With your ${t.card}.` : "Top up once by card first: it uses that card."}</p>
        </div>
        <Badge tone={t.auto.on ? "good" : "neutral"}>{t.auto.on ? "On" : "Off"}</Badge>
      </div>
      <div className="flex flex-wrap items-center gap-2 text-sm">
        When the balance falls below $
        <input value={below} onChange={(e) => setBelow(e.target.value)} inputMode="decimal" aria-label="Threshold" className={small} />
        add $
        <input value={amount} onChange={(e) => setAmount(e.target.value)} inputMode="decimal" aria-label="Amount" className={small} />
      </div>
      {t.auto.error && <Notice tone="bad">Auto top-up was switched off: {t.auto.error}</Notice>}
      {error && <Notice tone="bad">{error}</Notice>}
      <div className="flex gap-2">
        <Button size="sm" onClick={() => save(true)} disabled={!t.card}>{t.auto.on ? "Save" : "Turn on"}</Button>
        {t.auto.on && <Button size="sm" variant="ghost" onClick={() => save(false)}>Turn off</Button>}
      </div>
    </Card>
  );
}
