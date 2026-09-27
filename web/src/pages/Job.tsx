import { AlertTriangle, Camera, Check, ChevronRight, Clock, Cpu, Download, DollarSign, Loader2, Square, Zap } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { enterCameraMode, isHandheld } from "../live/device";
import * as API from "../api";
import type { Report, Stage, Status } from "../api";
import { Badge, Button, Card, Notice, cx } from "../components/ui";
import ReviewPanel from "./Review";

const FRIENDLY: Record<string, string> = {
  label: "Reading your photos", generate_assets: "Building 3D models from your photos", ingest: "Sorting your photos",
  fetch_assets: "Finding 3D models", convert_assets: "Preparing the models", preview_assets: "Photographing the models",
  qa_assets: "Checking the models", render: "Generating synthetic images", points: "Preparing training data",
  crops: "Preparing training data", train: "Training", evaluate: "Testing on your real photos", export: "Packing the model",
};
const POLL_MS = 5000;

export default function JobPage() {
  const { name = "" } = useParams();
  const [s, setS] = useState<Status | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reviewKey, setReviewKey] = useState(0);  // bumps when a run ends, to reload the review
  const [nonce, setNonce] = useState(0);  // bumps when a rerun starts, to poll again
  const wasRunning = useRef(false);

  useEffect(() => {
    let stop = false, timer: number | undefined;
    const poll = async () => {
      try {
        const st = await API.jobStatus(name);
        if (stop) return;
        setS(st);
        setError(null);
        if (wasRunning.current && !st.running) setReviewKey((k) => k + 1);
        wasRunning.current = st.running;
        if (st.running) timer = window.setTimeout(poll, POLL_MS);
      } catch (e) {
        if (!stop) { setError((e as Error).message); timer = window.setTimeout(poll, POLL_MS * 2); }
      }
    };
    setS(null);
    poll();
    return () => { stop = true; clearTimeout(timer); };
  }, [name, nonce]);

  const restart = () => { wasRunning.current = true; setNonce((n) => n + 1); };

  if (!s) {
    return <div className="py-10">{error ? <Notice tone="bad">{error}</Notice> : <Skeleton />}</div>;
  }
  const failed = s.stages.some((x) => x.state === "failed") || (!s.running && !s.report && s.log.some((l) => l.startsWith("FAILED")));
  return (
    <div className="space-y-6 py-6">
      <div>
        <p className="text-sm text-muted">Model</p>
        <h1 className="text-2xl font-bold tracking-tight">{(s.spec?.name ?? s.job).replace(/_/g, " ")}</h1>
      </div>
      {error && <Notice tone="bad">Lost contact with the server, retrying: {error}</Notice>}
      {(s.running || !s.report) && <Progress s={s} failed={failed} />}
      {s.report && <Result job={s.job} r={s.report} usd={s.cloud_usd} />}
      {!s.running && (s.report || failed) && <ReviewPanel key={reviewKey} job={s.job} onRerun={restart} />}
    </div>
  );
}

function Skeleton() {
  return <div className="animate-pulse space-y-4">
    <div className="h-8 w-1/3 rounded bg-stone-200" />
    <div className="h-40 rounded-[var(--radius-card)] bg-stone-200" />
    <div className="h-64 rounded-[var(--radius-card)] bg-stone-200" />
  </div>;
}

const minutes = (sec: number) => sec < 60 ? "under a minute" : `${Math.round(sec / 60)} min`;

function Progress({ s, failed }: { s: Status; failed: boolean }) {
  const done = s.stages.filter((x) => x.state === "done").length;
  const pct = s.stages.length ? Math.round((done / s.stages.length) * 100) : 0;
  const [stopping, setStopping] = useState(false);
  const t = s.timing;
  const stop = async () => {
    if (!confirm("Stop this run? The cloud GPU is released and the steps done so far are kept.")) return;
    setStopping(true);
    try { await API.cancelJob(s.job); } catch (e) { alert((e as Error).message); }
    setStopping(false);
  };
  return (
    <Card className="overflow-hidden">
      <div className="space-y-4 p-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className={cx("font-semibold", failed && "text-red-700")}>
              {failed ? "The run stopped" : s.running ? (s.phase ?? "Working") : "Not running"}
            </p>
            <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-sm text-muted">
              {t?.elapsed_s != null && <span className="flex items-center gap-1"><Clock className="h-3.5 w-3.5" /> {minutes(t.elapsed_s)} so far</span>}
              {s.running && t?.eta_s != null && t.eta_s > 0 && <span>about {minutes(t.eta_s)} left</span>}
              {s.cloud_usd > 0 && <span className="flex items-center gap-1"><DollarSign className="h-3.5 w-3.5" />{s.cloud_usd.toFixed(2)} of cloud GPU</span>}
            </div>
          </div>
          {s.running && <Button variant="danger" size="sm" onClick={stop} disabled={stopping}><Square className="h-3.5 w-3.5" /> Stop</Button>}
        </div>
        <div className="h-2 overflow-hidden rounded-full bg-stone-100">
          <div className={cx("h-full rounded-full transition-all duration-700", failed ? "bg-red-400" : "bg-gradient-to-r from-brand-500 to-ember-500")}
               style={{ width: `${Math.max(pct, s.running ? 4 : 0)}%` }} />
        </div>
      </div>
      <ol className="border-t border-line px-5 py-3">
        {s.stages.map((st) => <StageRow key={st.name} st={st} />)}
      </ol>
      <details className="border-t border-line px-5 py-3 text-sm">
        <summary className="cursor-pointer text-muted hover:text-ink">Details</summary>
        <pre className="mt-2 max-h-72 overflow-auto rounded-lg bg-stone-900 p-3 text-xs leading-relaxed text-stone-100">{s.log.join("\n")}</pre>
      </details>
    </Card>
  );
}

function StageRow({ st }: { st: Stage }) {
  return (
    <li className="flex items-center gap-3 py-2">
      <span className={cx("grid h-6 w-6 shrink-0 place-items-center rounded-full",
        st.state === "done" && "bg-green-500 text-white", st.state === "running" && "bg-ember-500 text-white animate-beat",
        st.state === "pending" && "border-2 border-line", st.state === "failed" && "bg-red-500 text-white")}>
        {st.state === "done" && <Check className="h-3.5 w-3.5" strokeWidth={3} />}
        {st.state === "running" && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
        {st.state === "failed" && <AlertTriangle className="h-3.5 w-3.5" />}
      </span>
      <div className="min-w-0 flex-1">
        <p className={cx("text-sm font-medium", st.state === "pending" && "text-muted")}>{FRIENDLY[st.name] ?? st.name}</p>
        <p className="truncate text-xs text-stone-400">{st.powered_by}</p>
      </div>
      {st.seconds != null && <span className="text-xs text-muted tabular-nums">{st.seconds < 60 ? `${Math.round(st.seconds)} s` : `${(st.seconds / 60).toFixed(1)} min`}</span>}
    </li>
  );
}

const pct = (v: number) => `${Math.round(v * 100)}%`;

function Result({ job, r, usd }: { job: string; r: Report; usd: number }) {
  const real = r.real_test ?? r.real_accuracy;
  const counting = r.task === "count";
  const headline = real ? pct(real.mean) : r.val_score != null ? pct(r.val_score) : "n/a";
  const explain = !real ? "on synthetic validation images (add real photos to score it on real ones)"
    : counting ? `of the objects in your real test photos found in the right place (F1, ${real.test_size ?? "?"} photos)`
    : `of your real test photos recognized correctly (${real.test_size ?? "?"} photos)`;
  const files = [["model_int8.tflite", "Model (int8)", true], ["model.tflite", "Portable model", !!r.portable_model_path],
                 ["labels.txt", "Labels", !!r.portable_model_path], ["model_data.h", "C array for the chip", r.c_array_path !== null],
                 ["report.json", "Full report", true]] as const;
  return (
    <div className="space-y-4">
      <Card className="p-6">
        <Badge tone="good"><Check className="h-3 w-3" /> Ready</Badge>
        {isHandheld() && r.portable_model_path && (
          <Link to={`/app/job/${job}/live`} onClick={() => { enterCameraMode(); }}
                className="mt-4 flex h-14 w-full items-center justify-center gap-2 rounded-[var(--radius-control)] bg-ink text-base font-semibold text-white shadow-sm">
            <Camera className="h-5 w-5" /> Try it live with your camera
          </Link>
        )}
        <div className="mt-4 flex flex-col gap-6 sm:flex-row sm:items-end">
          <div className="flex-1">
            <p className="text-6xl font-bold tracking-tight text-brand-600">{headline}</p>
            <p className="mt-2 max-w-md text-muted">{explain}</p>
          </div>
          <div className="grid grid-cols-2 gap-3 sm:w-72">
            {counting && real?.count_mae != null && <Mini label="count error per photo" value={real.count_mae.toFixed(1)} />}
            <Mini label={`int8, ${r.input[0]}x${r.input[1]} input`} value={`${r.model_kb} KB`} />
            <Mini label="device profile" value={r.hardware} />
            {usd > 0 && <Mini label="cloud GPU cost" value={`$${usd.toFixed(2)}`} />}
          </div>
        </div>
      </Card>
      <div className="grid gap-4 md:grid-cols-2">
        <Card className="p-5">
          <p className="font-semibold">Download</p>
          <div className="mt-3 space-y-1">
            {files.filter(([, , show]) => show).map(([f, label]) => (
              <a key={f} href={API.fileUrl(job, f)}
                 className="flex items-center justify-between rounded-lg px-3 py-2.5 text-sm hover:bg-stone-50">
                <span className="flex items-center gap-2"><Download className="h-4 w-4 text-muted" /> {label}</span>
                <span className="text-xs text-stone-400">{f}</span>
              </a>
            ))}
          </div>
        </Card>
        <Card className="bg-gradient-to-br from-ember-50 to-white p-5">
          <p className="flex items-center gap-2 font-semibold"><Cpu className="h-4 w-4 text-ember-700" /> Next: put it on your device</p>
          <p className="mt-2 text-sm text-muted">
            The int8 model runs without a network. On a microcontroller, the C array goes into the firmware; on a
            phone, Raspberry Pi or Jetson, the .tflite file loads with TensorFlow Lite.
          </p>
          <p className="mt-3 flex items-center gap-1 text-sm font-medium text-ember-700"><Zap className="h-4 w-4" /> Flashing from the browser is coming <ChevronRight className="h-4 w-4" /></p>
        </Card>
      </div>
    </div>
  );
}

function Mini({ label, value }: { label: string; value: string }) {
  return <div className="rounded-[var(--radius-control)] bg-stone-50 p-3">
    <p className="text-lg font-semibold tracking-tight">{value}</p>
    <p className="text-xs text-muted">{label}</p>
  </div>;
}
