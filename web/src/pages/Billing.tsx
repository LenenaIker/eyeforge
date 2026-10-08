import { CreditCard } from "lucide-react";
import { useEffect, useState } from "react";
import { getBilling, type Billing as BillingData } from "../api";
import { Badge, Button, Card, Notice } from "../components/ui";

// Credits: the balance, what things cost and every move of the ledger (sim2chip/accounts.py).

const REASONS: Record<string, string> = {
  grant: "Credits given", purchase: "Credits bought", run: "Model run", interview: "Chat", refund: "Refund",
};
const when = (t: number) => new Date(t * 1000).toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
const credits = (n: number) => `${n % 1 ? n.toFixed(1) : n} credit${n === 1 ? "" : "s"}`;

export default function Billing() {
  const [b, setB] = useState<BillingData | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { getBilling().then(setB).catch((e) => setError(e.message)); }, []);

  return (
    <div className="space-y-6 py-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Billing</h1>
        <p className="mt-1 text-sm text-muted">Your credits and what you spent them on.</p>
      </div>
      {error && <Notice tone="bad">{error}</Notice>}
      {b && !b.enabled && <Notice>This server has no accounts, so nothing is charged here.</Notice>}
      {b && (
        <div className="grid gap-4 sm:grid-cols-2">
          <Card className="p-5">
            <div className="text-sm text-muted">Balance</div>
            <div className="mt-1 flex items-baseline gap-2">
              {b.tier === "judge" ? <><span className="text-3xl font-semibold tracking-tight">Free</span><Badge tone="brand">Judge</Badge></>
                : <span className="text-3xl font-semibold tracking-tight">{b.balance != null ? credits(b.balance) : "-"}</span>}
            </div>
            <Button className="mt-4 w-full" disabled title="Coming soon"><CreditCard className="h-4 w-4" /> Buy credits (soon)</Button>
          </Card>
          <Card className="p-5">
            <div className="text-sm text-muted">Prices</div>
            <dl className="mt-2 space-y-2 text-sm">
              <div className="flex justify-between"><dt>Build or retrain a model</dt><dd className="font-medium">{credits(b.costs.run)}</dd></div>
              <div className="flex justify-between"><dt>A chat to set it up</dt><dd className="font-medium">{credits(b.costs.interview)}</dd></div>
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
                    <div className="truncate text-muted">{[m.detail?.replace(/_/g, " "), when(m.t)].filter(Boolean).join(" · ")}</div>
                  </div>
                  <span className={m.amount < 0 ? "text-ink" : "text-green-700"}>{m.amount > 0 ? "+" : ""}{m.amount}</span>
                </div>
              ))}
            </div>
          )}
        </Card>
      )}
    </div>
  );
}
