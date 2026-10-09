import { AlertTriangle, Camera, Check, ChevronRight, Clock, Cpu, Download, DollarSign, ImagePlus, Loader2, Pencil, Sparkles, Square, X, Zap } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { enterCameraMode, isHandheld } from "../live/device";
import { captureFile, captureLabels, clearCaptures, listCaptures, removeCapture, type Capture } from "../live/captures";
import { prepareImages } from "../images";
import * as API from "../api";
import type { Report, Stage, Status, Usage, Version } from "../api";
import { Badge, Button, Card, Notice, cx } from "../components/ui";
import ReviewPanel from "./Review";

const FRIENDLY: Record<string, string> = {
  label: "Reading your photos", generate_assets: "Building 3D models from your photos", ingest: "Sorting your photos",
  fetch_assets: "Finding 3D models", convert_assets: "Preparing the models", preview_assets: "Photographing the models",
  qa_assets: "Checking the models", render: "Generating synthetic images", points: "Preparing training data",
  crops: "Preparing training data", train: "Training", evaluate: "Testing on your real photos", export: "Packing the model",
};
const POLL_MS = 5000;
const IDLE_POLL_MS = 15000;

export default function JobPage() {
  const { name = "" } = useParams();
  const [s, setS] = useState<Status | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reviewKey, setReviewKey] = useState(0);  // bumps when a run ends, to reload the review
  const [nonce, setNonce] = useState(0);  // bumps when a rerun starts, to poll again
  const [news, setNews] = useState<string | null>(null);  // the job moved on from another device
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
        // Idle, still slowly: the same job may be open on another device, which can start a run.
        timer = window.setTimeout(poll, st.running ? POLL_MS : IDLE_POLL_MS);
      } catch (e) {
        if (!stop) { setError((e as Error).message); timer = window.setTimeout(poll, POLL_MS * 2); }
      }
    };
    // Back to this tab or app (after using the phone, say): look now instead of waiting for the timer.
    const seen = () => { if (document.visibilityState === "visible") { clearTimeout(timer); poll(); } };
    setS(null);
    poll();
    document.addEventListener("visibilitychange", seen);
    return () => { stop = true; clearTimeout(timer); document.removeEventListener("visibilitychange", seen); };
  }, [name, nonce]);

  const restart = (msg?: string) => { setNews(msg ?? null); wasRunning.current = true; setNonce((n) => n + 1); };

  if (!s) {
    return <div className="py-10">{error ? <Notice tone="bad">{error}</Notice> : <Skeleton />}</div>;
  }
  const failed = s.stages.some((x) => x.state === "failed") || (!s.running && !s.report && s.log.some((l) => l.startsWith("FAILED")));
  return (
    <div className="space-y-6 py-6">
      <div>
        <p className="text-sm text-muted">{s.labeling ? "Labeling" : "Model"}</p>
        <h1 className="text-2xl font-bold tracking-tight">{(s.spec?.name ?? s.job).replace(/_/g, " ")}</h1>
      </div>
      {error && <Notice tone="bad">Lost contact with the server, retrying: {error}</Notice>}
      {news && <Notice>{news}</Notice>}
      {s.labeling && !s.calibration ? <Labeling s={s} /> : s.calibration ? <Calibration s={s} reviewKey={reviewKey} onRerun={restart} /> :
        (s.running || !s.report) && <Progress s={s} failed={failed} />}
      {s.report && <Result job={s.job} r={s.report} usd={s.cloud_usd} version={s.version} />}
      {!s.running && s.report && <Improve job={s.job} counting={s.report.task === "count"} onRerun={restart} />}
      {s.report && s.usage && (s.usage.models.length > 0 || s.usage.gpu.length > 0) && <UsagePanel u={s.usage} />}
      {s.report && s.versions.length > 0 && <Versions job={s.job} versions={s.versions} />}
      {!s.calibration && !s.labeling && !s.running && (s.report || failed) && <ReviewPanel key={reviewKey} job={s.job} onRerun={restart} />}
    </div>
  );
}

// Before the first run: the teacher marks a few photos and the user checks it understood what to look for.
function Calibration({ s, reviewKey, onRerun }: { s: Status; reviewKey: number; onRerun: (news?: string) => void }) {
  const [error, setError] = useState<string | null>(null);
  const train = () => API.trainCalibrated(s.job, { labels: {}, assets: {}, points: {} }).then(() => onRerun(), (e) =>
    e instanceof API.ApiError && e.status === 409 ? onRerun(e.message) : setError((e as Error).message));
  if (s.running) {
    return (
      <Card className="flex items-center gap-3 p-4 text-sm">
        <Loader2 className="h-4 w-4 shrink-0 animate-spin" />
        <span>The teacher is marking a few of your photos, so you can check it understood what to look for.</span>
      </Card>
    );
  }
  if (s.calibration?.state === "failed") {
    return (
      <div className="space-y-3">
        <Notice tone="bad">The teacher could not mark the photos: {s.calibration.error}</Notice>
        {error && <Notice tone="bad">{error}</Notice>}
        <Button onClick={train}>Train anyway</Button>
      </div>
    );
  }
  return <ReviewPanel key={reviewKey} job={s.job} onRerun={onRerun} calibrating labeling={!!s.labeling} />;
}

// A labeling job after the check: the teacher marks every photo; the labels then live in the dataset.
function Labeling({ s }: { s: Status }) {
  const l = s.labeling!;
  const back = <Link to={`/app/datasets/${l.dataset}`} className="font-medium text-brand-600 hover:underline">See them in the dataset</Link>;
  if (l.state === "failed") return <Notice tone="bad">The teacher could not label the photos: {l.error}</Notice>;
  if (l.state === "done") return <Card className="p-4 text-sm">Every photo is labeled and kept with the dataset, for any model you build with it. {back}</Card>;
  return (
    <Card className="flex items-center gap-3 p-4 text-sm">
      <Loader2 className="h-4 w-4 shrink-0 animate-spin" />
      <span>The teacher is labeling every photo. You can leave: the labels go to the dataset when it ends.</span>
    </Card>
  );
}

function Skeleton() {
  return <div className="animate-pulse space-y-4">
    <div className="h-8 w-1/3 rounded bg-gray-200" />
    <div className="h-40 rounded-[var(--radius-card)] bg-gray-200" />
    <div className="h-64 rounded-[var(--radius-card)] bg-gray-200" />
  </div>;
}

const minutes = (sec: number) => sec < 60 ? "under a minute" : `${Math.round(sec / 60)} min`;

function Progress({ s, failed }: { s: Status; failed: boolean }) {
  const done = s.stages.filter((x) => x.state === "done").length;
  const pct = s.stages.length ? Math.round((done / s.stages.length) * 100) : 0;
  const [stopping, setStopping] = useState(false);
  const [asking, setAsking] = useState(false);  // the Stop button asks once, in place
  const t = s.timing;
  const stop = async () => {
    setAsking(false);
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
          {s.running && (asking ? (
            <div className="flex shrink-0 gap-2">
              <Button variant="ghost" size="sm" onClick={() => setAsking(false)}>Keep going</Button>
              <Button variant="danger" size="sm" onClick={stop}><Square className="h-3.5 w-3.5" /> Stop run</Button>
            </div>
          ) : <Button variant="danger" size="sm" onClick={() => setAsking(true)} disabled={stopping}><Square className="h-3.5 w-3.5" /> Stop</Button>)}
        </div>
        <div className="h-2 overflow-hidden rounded-full bg-gray-100">
          <div className={cx("h-full rounded-full transition-all duration-700", failed ? "bg-red-400" : "bg-gradient-to-r from-brand-500 to-ember-500")}
               style={{ width: `${Math.max(pct, s.running ? 4 : 0)}%` }} />
        </div>
      </div>
      <ol className="border-t border-line px-5 py-3">
        {s.stages.map((st) => <StageRow key={st.name} st={st} />)}
      </ol>
      <details className="border-t border-line px-5 py-3 text-sm">
        <summary className="cursor-pointer text-muted hover:text-ink">Details</summary>
        <pre className="mt-2 max-h-72 overflow-auto rounded-lg bg-gray-900 p-3 text-xs leading-relaxed text-gray-100">{s.log.join("\n")}</pre>
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
        <p className="truncate text-xs text-gray-400">{st.powered_by}</p>
      </div>
      {st.seconds != null && <span className="text-xs text-muted tabular-nums">{st.seconds < 60 ? `${Math.round(st.seconds)} s` : `${(st.seconds / 60).toFixed(1)} min`}</span>}
    </li>
  );
}

const pct = (v: number) => `${Math.round(v * 100)}%`;

function Result({ job, r, usd, version }: { job: string; r: Report; usd: number; version: Version | null }) {
  const real = r.real_test ?? r.real_accuracy;
  const counting = r.task === "count";
  // The model the user downloads is the headline; the other seeds only say how steady training was.
  const shipped = r.shipped_test ?? null;
  const headline = shipped ? pct(shipped.score) : real ? pct(real.mean) : r.val_score != null ? pct(r.val_score) : "n/a";
  const seeds = Object.values(real?.per_seed ?? {});
  const spread = shipped && seeds.length > 1
    ? `${seeds.length} trainings scored ${pct(Math.min(...seeds))} to ${pct(Math.max(...seeds))}; yours was picked before looking at the test`
    : null;
  const countMae = shipped?.count_mae ?? real?.count_mae;
  const explain = !real ? "on synthetic validation images (add real photos to score it on real ones)"
    : counting ? `of the objects in your real test photos found in the right place (F1, ${real.test_size ?? "?"} photos)`
    : `of your real test photos recognized correctly (${real.test_size ?? "?"} photos)`;
  // A count detector ships one model, model.tflite (float16): it is the model, not a portable copy.
  const single = r.model_path === "model.tflite";
  const files = [["model_int8.tflite", "Model (int8)", !single],
                 ["model.tflite", single ? "Model" : "Portable model", single || !!r.portable_model_path],
                 ["labels.txt", "Labels", !!r.portable_model_path], ["model_data.h", "C array for the chip", r.c_array_path !== null],
                 ["report.json", "Full report", true]] as const;
  return (
    <div className="space-y-4">
      <Card className="p-6">
        <div className="flex flex-wrap items-center gap-2">
          <Badge tone="good"><Check className="h-3 w-3" /> Ready</Badge>
          {version && version.n > 1 && <Badge>v{version.n} · {version.reason}</Badge>}
        </div>
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
            {spread && <p className="mt-1 text-xs text-muted">{spread}</p>}
            {r.previous && <Compare p={r.previous} counting={counting} />}
            {r.advice?.map((a) => <p key={a} className="mt-3 max-w-md rounded-lg bg-ember-50 px-3 py-2 text-sm text-ember-700">{a}</p>)}
          </div>
          <div className="grid grid-cols-2 gap-3 sm:w-72">
            {counting && countMae != null && <Mini label="count error per photo" value={countMae.toFixed(1)} />}
            <Mini label={`${r.quantization ?? "int8"}, ${r.input[0]}x${r.input[1]} input`} value={`${r.model_kb} KB`} />
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
                 className="flex items-center justify-between rounded-lg px-3 py-2.5 text-sm hover:bg-gray-50">
                <span className="flex items-center gap-2"><Download className="h-4 w-4 text-muted" /> {label}</span>
                <span className="text-xs text-gray-400">{f}</span>
              </a>
            ))}
          </div>
        </Card>
        <Card className="bg-gradient-to-br from-ember-50 to-white p-5">
          <p className="flex items-center gap-2 font-semibold"><Cpu className="h-4 w-4 text-ember-700" /> Next: put it on your device</p>
          <p className="mt-2 text-sm text-muted">
            The model runs without a network. On a microcontroller, the C array goes into the firmware; on a
            phone, Raspberry Pi or Jetson, the .tflite file loads with TensorFlow Lite.
          </p>
          <p className="mt-3 flex items-center gap-1 text-sm font-medium text-ember-700"><Zap className="h-4 w-4" /> Flashing from the browser is coming <ChevronRight className="h-4 w-4" /></p>
        </Card>
      </div>
    </div>
  );
}

function Compare({ p, counting }: { p: NonNullable<Report["previous"]>; counting: boolean }) {
  if (p.score == null || p.now == null) {
    return <p className="mt-3 text-sm text-muted">Not compared with v{p.version}: {p.skipped ?? "no score"}.</p>;
  }
  const up = p.now - p.score;
  return (
    <p className="mt-3 inline-flex flex-wrap items-center gap-x-2 rounded-full bg-gray-50 px-3 py-1.5 text-sm">
      <span className="text-muted">v{p.version} {pct(p.score)}{p.then != null && Math.round(p.then * 100) !== Math.round(p.score * 100)
        ? ` (${pct(p.then)} before the test changed)` : ""}</span>
      <ChevronRight className="h-3.5 w-3.5 text-gray-400" />
      <span className="font-semibold">now {pct(p.now)}</span>
      <span className={cx("font-medium", up > 0 ? "text-green-700" : up < 0 ? "text-red-700" : "text-muted")}>
        ({up >= 0 ? "+" : ""}{Math.round(up * 100)} pts{counting && p.count_mae != null && p.now_count_mae != null
          ? `, count error ${p.count_mae.toFixed(1)} → ${p.now_count_mae.toFixed(1)}` : ""})
      </span>
      <span className="text-xs text-muted">same test photos, shipped models</span>
    </p>
  );
}

// The loop after a first model: photos of what it gets wrong (picked here or saved in the live test), the
// teacher marks them, and only the real-photo steps run again.
function Improve({ job, counting, onRerun }: { job: string; counting: boolean; onRerun: () => void }) {
  const [files, setFiles] = useState<File[]>([]);
  const [caps, setCaps] = useState<Capture[]>([]);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const camera = useRef<HTMLInputElement>(null), gallery = useRef<HTMLInputElement>(null);
  const card = useRef<HTMLDivElement>(null);
  useEffect(() => { listCaptures(job).then(setCaps); }, [job]);
  useEffect(() => {  // arriving from the live test's "Improve the model"
    if (window.location.hash === "#improve") card.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, []);
  const blobs = useMemo(() => [...caps.map((c) => c.blob), ...files], [caps, files]);
  const previews = useObjectUrls(blobs);
  const total = caps.length + files.length;

  const pick = async (list: FileList | null) => {
    if (list?.length) setFiles((f) => [...f, ...Array.from(list)]);
  };
  const drop = (i: number) => {
    if (i < caps.length) { removeCapture(caps[i].id); setCaps((c) => c.filter((_, k) => k !== i)); }
    else setFiles((f) => f.filter((_, k) => k !== i - caps.length));
  };
  const send = async () => {
    setBusy(true); setError(null);
    try {
      await API.addPhotos(job, await prepareImages([...caps.map(captureFile), ...files]), note, captureLabels(caps));
      await clearCaptures(job);
      onRerun();
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  };
  return (
    <div ref={card} id="improve" className="scroll-mt-4"><Card className="p-5">
      <p className="flex items-center gap-2 font-semibold"><Sparkles className="h-4 w-4 text-brand-600" /> Make it better</p>
      <p className="mt-1 text-sm text-muted">
        Missing something? Add photos of the cases it gets wrong. They are marked for you, half of them join the
        test so the fix is measured, and only your photos are processed again: the synthetic images are reused,
        so it takes minutes, not a full run.
      </p>
      <textarea value={note} onChange={(e) => setNote(e.target.value)} rows={2} maxLength={120}
                placeholder={counting ? "What does it get wrong? e.g. misses pots in the shade" : "What does it get wrong? e.g. dry patches on the lawn"}
                className="mt-3 w-full resize-none rounded-[var(--radius-control)] border border-line bg-white px-3 py-2 text-sm focus:border-brand-500 focus:outline-none" />
      {total > 0 && (
        <div className="mt-3 grid grid-cols-4 gap-2 sm:grid-cols-6">
          {previews.map((url, i) => (
            <div key={url} className="relative aspect-square overflow-hidden rounded-lg bg-gray-100">
              <img src={url} alt={i < caps.length ? "Frame saved in the live test" : "Photo to add"} className="h-full w-full object-cover" />
              {i < caps.length && <span className="absolute bottom-1 left-1 max-w-[90%] truncate rounded bg-black/60 px-1 text-[10px] font-medium text-white">
                {saidText(caps[i])}</span>}
              <button onClick={() => drop(i)} aria-label="Remove"
                      className="absolute right-1 top-1 grid h-6 w-6 place-items-center rounded-full bg-black/55 text-white"><X className="h-3.5 w-3.5" /></button>
            </div>
          ))}
        </div>
      )}
      <div className="mt-3 flex flex-wrap gap-2">
        {isHandheld() && <Button variant="secondary" size="sm" onClick={() => camera.current?.click()}><Camera className="h-4 w-4" /> Take photos</Button>}
        <Button variant="secondary" size="sm" onClick={() => gallery.current?.click()}><ImagePlus className="h-4 w-4" /> Choose photos</Button>
        <Link to={`/app?revise=${job}`}
              className="inline-flex h-9 items-center gap-1.5 rounded-[var(--radius-control)] px-3 text-sm font-medium text-muted hover:bg-gray-100 hover:text-ink">
          <Pencil className="h-4 w-4" /> Change the setup
        </Link>
        <Button size="sm" className="sm:ml-auto" onClick={send} disabled={!total || busy}>
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : null} {total ? `Add ${total} photo${total > 1 ? "s" : ""} and retrain` : "Add photos and retrain"}
        </Button>
      </div>
      <input ref={camera} type="file" accept="image/*" capture="environment" hidden onChange={(e) => { pick(e.target.files); e.target.value = ""; }} />
      <input ref={gallery} type="file" accept="image/*" multiple hidden onChange={(e) => { pick(e.target.files); e.target.value = ""; }} />
      {error && <div className="mt-3"><Notice tone="bad">{error}</Notice></div>}
    </Card></div>
  );
}

/** The tag on a live capture: what the user said it is, or "live" when the teacher will mark it. */
function saidText(c: Capture): string {
  if (!c.said) return "live";
  return "label" in c.said ? c.said.label.replace(/_/g, " ") : `${c.said.points.length} marked`;
}

/** Object URLs for blobs, revoked when the list changes or the card goes away. */
function useObjectUrls(blobs: Blob[]): string[] {
  const [urls, setUrls] = useState<string[]>([]);
  useEffect(() => {
    const u = blobs.map((b) => URL.createObjectURL(b));
    setUrls(u);
    return () => { u.forEach((x) => URL.revokeObjectURL(x)); };
  }, [blobs]);
  return urls;
}

function Versions({ job, versions }: { job: string; versions: Version[] }) {
  return (
    <Card className="p-5">
      <details>
        <summary className="cursor-pointer font-semibold">Earlier versions ({versions.length})</summary>
        <ul className="mt-3 divide-y divide-line">
          {[...versions].reverse().map((v) => (
            <li key={v.n} className="flex flex-wrap items-center justify-between gap-2 py-2.5 text-sm">
              <span><span className="font-medium">v{v.n}</span> <span className="text-muted">· {v.reason}</span>
                {(v.shipped ?? v.headline) != null && <span className="text-muted"> · {pct((v.shipped ?? v.headline)!)}{v.test_size ? ` on ${v.test_size} photos` : ""}</span>}
              </span>
              <a href={API.versionFileUrl(job, v.n, v.model ?? "model_int8.tflite")}
                 className="flex items-center gap-1 text-brand-600 hover:underline"><Download className="h-3.5 w-3.5" /> model</a>
            </li>
          ))}
        </ul>
      </details>
    </Card>
  );
}

// What this model took: each Token Factory model with the steps it did, and the GPU time, with their cost.
function UsagePanel({ u }: { u: Usage }) {
  const usd = (v: number | null) => v == null ? "" : v < 0.01 ? "< $0.01" : `$${v.toFixed(2)}`;
  const tokens = (n: number) => n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${Math.round(n / 1e3)}k` : String(n);
  return (
    <Card className="p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="font-semibold">How it was made</p>
        <p className="text-sm text-muted">{usd(u.usd)} in all{u.priced ? "" : " (some models without a price)"}</p>
      </div>
      <ul className="mt-3 divide-y divide-line text-sm">
        {u.models.map((m) => (
          <li key={m.model} className="flex flex-wrap items-center justify-between gap-2 py-2.5">
            <span className="flex flex-wrap items-center gap-2">
              <span className="font-medium">{m.name}</span>
              {m.nvidia && <Badge tone="good">NVIDIA</Badge>}
              <span className="text-muted">{m.service ?? "Nebius Token Factory"} · {m.steps.join(", ")}</span>
            </span>
            <span className="text-muted">{m.service === "Tavily" ? `${m.calls} searches` :
              `${m.calls} calls · ${tokens(m.input_tokens + m.output_tokens)} tokens`}{usd(m.usd) && ` · ${usd(m.usd)}`}</span>
          </li>
        ))}
        {u.gpu.map((g) => (
          <li key={g.where} className="flex flex-wrap items-center justify-between gap-2 py-2.5">
            <span className="flex items-center gap-2"><span className="font-medium">GPU</span><span className="text-muted">{g.where}</span></span>
            <span className="text-muted">{g.minutes.toFixed(1)} min · {g.usd > 0 ? usd(g.usd) : "free"}</span>
          </li>
        ))}
      </ul>
    </Card>
  );
}

function Mini({ label, value }: { label: string; value: string }) {
  return <div className="rounded-[var(--radius-control)] bg-gray-50 p-3">
    <p className="text-lg font-semibold tracking-tight">{value}</p>
    <p className="text-xs text-muted">{label}</p>
  </div>;
}
