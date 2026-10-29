import { Box, ChevronDown, CircleUser, CreditCard, Cpu, Images, KeyRound, LogOut, Menu, MessageSquare, Mountain, Package,
         Plus, Shapes, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { NavLink, Outlet, useLocation, useNavigate, useOutletContext } from "react-router-dom";
import { getConfig, getMe, loginUrl, logout, onAccountChange, unlock, usd, type Config, type Me } from "../api";
import InstallButton from "../components/InstallButton";
import { Badge, Button, Card, Logo, Notice, cx } from "../components/ui";

// me: the signed-in account on a server with accounts (null before it loads, or on a server without them).
// stamp: bumped by refreshJobs when a job starts, so the library pages load again.
export type AppContext = { config: Config | null; me: Me | null; refreshJobs: () => void; stamp: number };
export const useApp = () => useOutletContext<AppContext>();

export default function AppShell() {
  const [config, setConfig] = useState<Config | null>(null);
  const [me, setMe] = useState<Me | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [stamp, setStamp] = useState(0);
  const [menu, setMenu] = useState(false);  // the sidebar as a drawer, on a phone
  const { pathname } = useLocation();
  useEffect(() => setMenu(false), [pathname]);

  const refreshJobs = () => setStamp((n) => n + 1);
  const loadMe = () => getMe().then(setMe).catch(() => {});
  const load = () => {
    getConfig().then((c) => {
      setConfig(c);
      if (c.login) loadMe();
    }).catch((e) => setError(e.message));
  };
  useEffect(load, []);
  useEffect(() => onAccountChange(loadMe), []);  // a 401, 402 or 403 anywhere: the session or the balance changed

  const account = me?.account;
  const gate = config?.locked ? "key"
    : config?.login && me && !me.signed_in ? "login"
    : account?.tier === "banned" ? "paused"
    : config?.login && !me ? "wait" : null;

  return (
    <div className="flex min-h-dvh">
      {!gate && (
        <>
          <aside className="sticky top-0 hidden h-dvh w-60 shrink-0 border-r border-line bg-white lg:block"><Sidebar engine={config?.engine} /></aside>
          {menu && (
            <div className="fixed inset-0 z-40 lg:hidden" onClick={() => setMenu(false)}>
              <div className="absolute inset-0 bg-black/30" />
              <aside className="absolute inset-y-0 left-0 w-64 bg-white shadow-lg" onClick={(e) => e.stopPropagation()}>
                <Sidebar engine={config?.engine} onClose={() => setMenu(false)} />
              </aside>
            </div>
          )}
        </>
      )}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-20 border-b border-line bg-white/90 backdrop-blur">
          <div className="flex h-14 items-center justify-between gap-3 px-4">
            <div className="flex items-center gap-2">
              {!gate && (
                <button onClick={() => setMenu(true)} aria-label="Menu"
                        className="flex h-9 w-9 items-center justify-center rounded-lg text-muted hover:bg-gray-100 lg:hidden">
                  <Menu className="h-5 w-5" />
                </button>
              )}
              <Logo className={cx(!gate && "lg:hidden")} />
            </div>
            <div className="flex items-center gap-2">
              {config && (
                <span className="hidden sm:block">
                  <Badge tone={config.backend === "nebius" ? "brand" : "neutral"}>
                    {config.backend === "nebius" ? "Runs on Nebius AI Cloud" : "Runs on a local GPU"}
                  </Badge>
                </span>
              )}
              <InstallButton />
              {account && <AccountMenu me={me!} />}
            </div>
          </div>
        </header>
        {error && <div className="mx-auto w-full max-w-5xl px-4 pt-4"><Notice tone="bad">Cannot reach the server: {error}</Notice></div>}
        <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col px-4">
          {gate === "key" && <KeyGate onOpen={load} />}
          {gate === "login" && <SignIn me={me!} providers={config!.login!} />}
          {gate === "paused" && <Paused contact={me!.contact} />}
          {!gate && me?.signed_in && account?.tier === "normal" && account.balance < me.costs.start.run && <CreditsNotice me={me} />}
          {!gate && <Outlet context={{ config, me, refreshJobs, stamp } satisfies AppContext} />}
        </main>
      </div>
    </div>
  );
}

// An item without `to` heads the items marked sub under it (Assets: objects and scenes); a click folds them away.
const FOLDED = "eyeforge.assetsFolded";
type NavItem = { to?: string; label: string; icon: typeof MessageSquare; sub?: boolean };
const NAV: { title?: string; items: NavItem[] }[] = [
  { items: [{ to: "/app", label: "Chat", icon: MessageSquare }] },
  { title: "Library", items: [
    { to: "/app/models", label: "Models", icon: Cpu },
    { to: "/app/datasets", label: "Datasets", icon: Images },
    { label: "Assets", icon: Shapes },
    { to: "/app/assets/objects", label: "Objects", icon: Box, sub: true },
    { to: "/app/assets/scenes", label: "Scenes", icon: Mountain, sub: true },
    { to: "/app/kits", label: "Kits", icon: Package },
  ] },
  { title: "Manage", items: [{ to: "/app/billing", label: "Billing", icon: CreditCard }] },
];

function Sidebar({ engine, onClose }: { engine?: string; onClose?: () => void }) {
  const navigate = useNavigate();
  const { pathname } = useLocation();
  const inJob = pathname.startsWith("/app/job/");  // a model's own page belongs to Models
  const [folded, setFolded] = useState(() => { try { return localStorage.getItem(FOLDED) === "1"; } catch { return false; } });
  const fold = () => setFolded((f) => {
    try { localStorage.setItem(FOLDED, f ? "0" : "1"); } catch { /* private window: only this visit */ }
    return !f;
  });
  const inAssets = pathname.startsWith("/app/assets/");
  const link = ({ isActive }: { isActive: boolean }) => cx(
    "flex h-9 items-center gap-2.5 rounded-lg px-3 text-sm font-medium transition-colors",
    isActive ? "bg-brand-50 text-brand-700" : "text-muted hover:bg-gray-100 hover:text-ink");
  return (
    <nav className="flex h-full flex-col gap-1 overflow-y-auto p-3">
      <div className="flex h-11 items-center justify-between px-1">
        <Logo />
        {onClose && <button onClick={onClose} aria-label="Close menu" className="rounded-lg p-1.5 text-muted hover:bg-gray-100"><X className="h-5 w-5" /></button>}
      </div>
      <Button size="sm" className="my-3" onClick={() => navigate("/app", { state: { fresh: Date.now() } })}><Plus className="h-4 w-4" /> New model</Button>
      {NAV.map((group, i) => (
        <div key={i} className="mb-2">
          {group.title && <div className="px-3 pt-3 pb-1.5 text-xs font-medium tracking-wide text-muted/80 uppercase">{group.title}</div>}
          {group.items.map((it) => it.to ? (!(it.sub && folded) &&
            <NavLink key={it.to} to={it.to} end className={({ isActive }) => cx(link({ isActive: isActive || (inJob && it.to === "/app/models") }), it.sub && "ml-6")}>
              <it.icon className="h-4 w-4" strokeWidth={1.75} /> {it.label}
            </NavLink>
          ) : (
            // A folded heading stays marked while one of its pages is open, so the user still sees where they are.
            <button key={it.label} onClick={fold} aria-expanded={!folded}
                    className={cx(link({ isActive: folded && inAssets }), "w-full")}>
              <it.icon className="h-4 w-4" strokeWidth={1.75} /> {it.label}
              <ChevronDown className={cx("ml-auto h-4 w-4 transition-transform", folded && "-rotate-90")} strokeWidth={1.75} />
            </button>
          ))}
        </div>
      ))}
      {engine && <div className="mt-auto px-3 pt-4 text-[11px] text-muted/70" title="The EyeForge this server runs">EyeForge {engine}</div>}
    </nav>
  );
}

// A public server: the landing is open to everyone, building a model needs the access key given to the judges.
function KeyGate({ onOpen }: { onOpen: () => void }) {
  const [key, setKey] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await unlock(key.trim());
      onOpen();
    } catch {
      setError("That key is not right. Check the one in the instructions you were given.");
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="flex flex-1 items-center justify-center py-12">
      <Card className="w-full max-w-sm p-6">
        <div className="mb-4 flex items-center gap-2 text-lg font-semibold"><KeyRound className="h-5 w-5" /> Access key</div>
        <p className="mb-4 text-sm text-muted">
          Building a model uses cloud GPUs, so this demo asks for the access key from the instructions.
        </p>
        <form onSubmit={submit} className="flex flex-col gap-3">
          <input value={key} onChange={(e) => setKey(e.target.value)} autoFocus autoComplete="off"
                 placeholder="Access key" aria-label="Access key"
                 className="h-11 rounded-lg border border-line px-3 outline-none focus:border-brand-500" />
          {error && <Notice tone="bad">{error}</Notice>}
          <Button type="submit" disabled={!key.trim() || busy}>{busy ? "Checking..." : "Open the app"}</Button>
        </form>
      </Card>
    </div>
  );
}

const PROVIDER_NAMES: Record<string, string> = { google: "Google", github: "GitHub" };

// A public server with accounts: the landing is open, building a model needs an account (judges get it free).
function SignIn({ me, providers }: { me: Me; providers: string[] }) {
  const failed = new URLSearchParams(window.location.search).get("login") === "failed";
  const next = window.location.pathname.startsWith("/app") ? window.location.pathname : "/app";
  return (
    <div className="flex flex-1 items-center justify-center py-12">
      <Card className="w-full max-w-sm p-6">
        <div className="mb-2 text-lg font-semibold">Sign in to build a model</div>
        <p className="mb-5 text-sm text-muted">
          Training runs on cloud GPUs, so the app needs an account. We only read your name and email.
        </p>
        {failed && <div className="mb-4"><Notice tone="bad">The sign in did not finish. Please try again.</Notice></div>}
        <div className="flex flex-col gap-3">
          {providers.map((p) => (
            <a key={p} href={loginUrl(p, next)}
               className="inline-flex h-11 items-center justify-center gap-2.5 rounded-[var(--radius-control)] border border-line bg-white px-4 text-[15px] font-medium shadow-sm hover:bg-gray-50">
              <ProviderIcon provider={p} /> Continue with {PROVIDER_NAMES[p] ?? p}
            </a>
          ))}
        </div>
        {me.judging && me.contact && <p className="mt-5 text-xs text-muted"><JudgeLine contact={me.contact} /></p>}
      </Card>
    </div>
  );
}

function JudgeLine({ contact }: { contact: string }) {
  return (
    <>Judging this hackathon? Accounts with an <b>nvidia.com</b>, <b>nebius.com</b>, <b>tavily.com</b> or{" "}
      <b>devpost.com</b> email are free. With another account, write to{" "}
      <a className="text-brand-600 underline" href={`mailto:${contact}`}>{contact}</a> from its email and we will
      give it free access.</>
  );
}

function Paused({ contact }: { contact: string | null }) {
  return (
    <div className="flex flex-1 items-center justify-center py-12">
      <Card className="w-full max-w-md p-6">
        <div className="mb-2 text-lg font-semibold">Account paused</div>
        <p className="text-sm text-muted">
          Your account was paused by an automatic safety check. If this is a mistake, write to{" "}
          {contact ? <a className="text-brand-600 underline" href={`mailto:${contact}`}>{contact}</a> : "us"} and we
          will sort it out. Sorry for the trouble.
        </p>
        <div className="mt-5"><SignOutButton /></div>
      </Card>
    </div>
  );
}

function CreditsNotice({ me }: { me: Me }) {
  const balance = me.account!.balance;
  return (
    <div className="pt-4">
      <Notice>
        {balance > 0 ? `Your balance is ${usd(balance)}. ` : "Your balance is empty. "}
        You pay what you use: a model is usually about {usd(me.costs.start.run)}, a chat a few cents.{" "}
        {me.topup ? <NavLink className="text-brand-600 underline" to="/app/billing">Top up in Billing.</NavLink> : "Topping up is coming soon."}
        {me.judging && me.contact && <> <JudgeLine contact={me.contact} /></>}
      </Notice>
    </div>
  );
}

function AccountMenu({ me }: { me: Me }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const acc = me.account!;
  useEffect(() => {
    const close = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("click", close);
    return () => document.removeEventListener("click", close);
  }, []);
  return (
    <div ref={ref} className="relative">
      <button onClick={() => setOpen(!open)} aria-label="Account"
              className="flex h-9 items-center gap-1.5 rounded-lg px-2 text-sm font-medium text-muted hover:bg-gray-100 hover:text-ink">
        <CircleUser className="h-5 w-5" strokeWidth={1.75} />
        {acc.tier === "judge" ? <Badge tone="brand">Judge</Badge>
          : acc.tier === "normal" ? <Badge tone={acc.balance > 0 ? "neutral" : "ember"}>{usd(acc.balance)}</Badge> : null}
      </button>
      {open && (
        <div className="absolute right-0 mt-2 w-64 overflow-hidden rounded-[var(--radius-card)] border border-line bg-white py-1 shadow-lg">
          <div className="px-4 py-3 text-sm">
            <div className="truncate font-medium">{acc.name || acc.email}</div>
            {acc.name && <div className="truncate text-muted">{acc.email}</div>}
            <div className="mt-1 text-xs text-muted">
              {acc.tier === "judge" ? "Judge access: free while the judging lasts." : `Balance ${usd(acc.balance)}`}
            </div>
          </div>
          <div className="border-t border-line px-2 py-1"><SignOutButton /></div>
        </div>
      )}
    </div>
  );
}

function SignOutButton() {
  return (
    <Button variant="ghost" size="sm" onClick={() => logout().finally(() => window.location.assign("/"))}>
      <LogOut className="h-4 w-4" /> Sign out
    </Button>
  );
}

function ProviderIcon({ provider }: { provider: string }) {
  if (provider === "google") {
    return (
      <svg viewBox="0 0 48 48" className="h-5 w-5" aria-hidden="true">
        <path fill="#FFC107" d="M43.6 20.5H42V20H24v8h11.3C33.7 32.7 29.2 36 24 36c-6.6 0-12-5.4-12-12s5.4-12 12-12c3.1 0 5.8 1.2 7.9 3.1l5.7-5.7C34 6.1 29.3 4 24 4 12.9 4 4 12.9 4 24s8.9 20 20 20 20-8.9 20-20c0-1.3-.1-2.4-.4-3.5z" />
        <path fill="#FF3D00" d="m6.3 14.7 6.6 4.8C14.7 15.1 19 12 24 12c3.1 0 5.8 1.2 7.9 3.1l5.7-5.7C34 6.1 29.3 4 24 4 16.3 4 9.7 8.3 6.3 14.7z" />
        <path fill="#4CAF50" d="M24 44c5.2 0 9.9-2 13.4-5.2l-6.2-5.2C29.2 35.1 26.7 36 24 36c-5.2 0-9.6-3.3-11.3-8l-6.5 5C9.5 39.6 16.2 44 24 44z" />
        <path fill="#1976D2" d="M43.6 20.5H42V20H24v8h11.3c-.8 2.2-2.2 4.2-4.1 5.6l6.2 5.2C37 39.2 44 34 44 24c0-1.3-.1-2.4-.4-3.5z" />
      </svg>
    );
  }
  if (provider === "github") {
    return (
      <svg viewBox="0 0 16 16" className="h-5 w-5" fill="currentColor" aria-hidden="true">
        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z" />
      </svg>
    );
  }
  return null;
}
