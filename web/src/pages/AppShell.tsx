import { ChevronDown, FolderOpen, KeyRound, Plus, WifiOff } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Link, Outlet, useLocation, useNavigate, useOutletContext } from "react-router-dom";
import { getConfig, listJobs, unlock, type Config } from "../api";
import InstallButton from "../components/InstallButton";
import { savedJobs } from "../live/saved";
import { Badge, Button, Card, Logo, Notice, cx } from "../components/ui";

export type AppContext = { config: Config | null; refreshJobs: () => void };
export const useApp = () => useOutletContext<AppContext>();

export default function AppShell() {
  const [config, setConfig] = useState<Config | null>(null);
  const [jobs, setJobs] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  const refreshJobs = () => listJobs().then((r) => setJobs(r.jobs)).catch(() => {});
  const load = () => {
    getConfig().then((c) => { setConfig(c); if (!c.locked) refreshJobs(); }).catch((e) => setError(e.message));
  };
  useEffect(load, []);

  return (
    <div className="flex min-h-dvh flex-col">
      <header className="sticky top-0 z-20 border-b border-line bg-white/90 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-5xl items-center justify-between gap-3 px-4">
          <Logo />
          <div className="flex items-center gap-2">
            {config && (
              <span className="hidden sm:block">
                <Badge tone={config.backend === "nebius" ? "brand" : "neutral"}>
                  {config.backend === "nebius" ? "Nebius AI Cloud" : "Local GPU"}
                </Badge>
              </span>
            )}
            <InstallButton />
            {!config?.locked && <JobsMenu jobs={jobs} />}
          </div>
        </div>
      </header>
      {error && <div className="mx-auto w-full max-w-5xl px-4 pt-4"><Notice tone="bad">Cannot reach the server: {error}</Notice></div>}
      <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col px-4">
        {config?.locked
          ? <KeyGate onOpen={load} />
          : <Outlet context={{ config, refreshJobs } satisfies AppContext} />}
      </main>
    </div>
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

function JobsMenu({ jobs }: { jobs: string[] }) {
  const [open, setOpen] = useState(false);
  const [offline, setOffline] = useState<Set<string>>(new Set());  // models saved on this device for the live test
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => { if (open) savedJobs().then(setOffline).catch(() => {}); }, [open]);
  const navigate = useNavigate();
  const { pathname } = useLocation();
  useEffect(() => { setOpen(false); }, [pathname]);
  useEffect(() => {
    const close = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("click", close);
    return () => document.removeEventListener("click", close);
  }, []);

  return (
    <div ref={ref} className="relative">
      <button onClick={() => setOpen(!open)}
              className="flex h-9 items-center gap-1.5 rounded-lg px-3 text-sm font-medium whitespace-nowrap text-muted hover:bg-stone-100 hover:text-ink">
        <FolderOpen className="h-4 w-4" strokeWidth={1.75} /> <span className="hidden sm:inline">Your</span> models <ChevronDown className="h-4 w-4" />
      </button>
      {open && (
        <div className="absolute right-0 mt-2 w-64 overflow-hidden rounded-[var(--radius-card)] border border-line bg-white py-1 shadow-lg">
          <button onClick={() => navigate("/app")}
                  className="flex w-full items-center gap-2 px-4 py-2.5 text-left text-sm font-medium text-brand-600 hover:bg-brand-50">
            <Plus className="h-4 w-4" /> New model
          </button>
          <div className="max-h-72 overflow-y-auto border-t border-line">
            {jobs.length === 0 && <p className="px-4 py-3 text-sm text-muted">Nothing trained yet.</p>}
            {[...jobs].reverse().map((j) => (
              <Link key={j} to={`/app/job/${j}`}
                    className={cx("block truncate px-4 py-2.5 text-sm hover:bg-stone-50",
                                  pathname.endsWith(`/${j}`) && "bg-stone-50 font-medium")}>
                <span className="flex items-center justify-between gap-2">
                  <span className="truncate">{j}</span>
                  {offline.has(j) && <WifiOff className="h-3.5 w-3.5 shrink-0 text-muted" aria-label="works offline on this device" />}
                </span>
              </Link>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
