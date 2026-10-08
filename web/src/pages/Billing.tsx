import { CreditCard } from "lucide-react";
import { useEffect, useState } from "react";
import { getBilling, usd, type Billing as BillingData } from "../api";
import { Badge, Button, Card, Notice } from "../components/ui";

// The balance in dollars, how things are priced (by use) and every move: money in, and what each model, chat or
// labeling used (sim2chip/accounts.py).

const REASONS: Record<string, string> = {
  grant: "Balance given", purchase: "Top-up", run: "Model", label: "Labeling", chat: "Chat", profile: "Dataset look",
  interview: "Chat", refund: "Refund",
};
// What a usage line is about: "job:<name>", "chat:<session>", "profile:<dataset>_<time>".
const about = (d: string | null) => !d ? null : d.startsWith("job:") ? d.slice(4).replace(/_/g, " ")
  : d.startsWith("chat:") || d.startsWith("profile:") ? null : d.replace(/_/g, " ");
const when = (t: number) => new Date(t * 1000).toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });

export default function Billing() {
  const [b, setB] = useState<BillingData | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { getBilling().then(setB).catch((e) => setError(e.message)); }, []);

  return (
    <div className="space-y-6 py-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Billing</h1>
        <p className="mt-1 text-sm text-muted">You pay for what you use, as you use it.</p>
      </div>
      {error && <Notice tone="bad">{error}</Notice>}
      {b && !b.enabled && <Notice>This server has no accounts, so nothing is charged here.</Notice>}
      {b && (
        <div className="grid gap-4 sm:grid-cols-2">
          <Card className="p-5">
            <div className="text-sm text-muted">Balance</div>
            <div className="mt-1 flex items-baseline gap-2">
              {b.tier === "judge" ? <><span className="text-3xl font-semibold tracking-tight">Free</span><Badge tone="brand">Judge</Badge></>
                : <span className="text-3xl font-semibold tracking-tight">{b.balance != null ? usd(b.balance) : "-"}</span>}
            </div>
            <Button className="mt-4 w-full" disabled title="Coming soon"><CreditCard className="h-4 w-4" /> Top up (soon)</Button>
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
