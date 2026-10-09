import { Camera, Check, Image as ImageIcon, RefreshCw, SlidersHorizontal, X } from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { Link, Navigate, useNavigate, useParams } from "react-router-dom";
import { Spinner, cx } from "../components/ui";
import { addCapture, listCaptures, type Said } from "../live/captures";
import { anchors, decodeBoxes } from "../live/boxes";
import { decode, type LivePoint } from "../live/decode";
import { isHandheld } from "../live/device";
import { fit, toFrame, turnFromGravity, uprightRgb, uprightTransform, type Turn } from "../live/orient";
import { resizeBilinear } from "../live/resize";
import { loadModel, type LiveModel } from "../live/runtime";

// The trained model on the phone's camera, run on the phone: nothing leaves it and no network is needed once
// the model is saved. It behaves like a camera app: full screen, the picture fills it, the screen stays portrait
// while the labels turn with the phone, and the system back gesture leaves (no back button, except on iOS,
// where an installed app has none). The frame is turned upright before the model sees it and shrunk with
// Pillow's filter to the model's square input, as ingest did with the photos it was trained and scored on.
// The shutter freezes the frame (upright, full size) with what the model saw; the user says what it really is
// (one tap when the model was right) and it is kept for the job page's "Make it better", offline too.

// count: a detector's kept boxes, or a grid model's density output (steadier than its points in crowds; models
// before it: the points).
type Seen = { points: LivePoint[]; count: number; top: { label: string; p: number } | null; ms: number };
const SMOOTH = 5;  // frames in the median count shown, so the number does not flicker
const SAVE_SIDE = 1600;  // long side of a saved frame, as images.ts shrinks uploads
const IOS = /iphone|ipad|ipod/i.test(navigator.userAgent) || (/macintosh/i.test(navigator.userAgent) && navigator.maxTouchPoints > 1);

export default function Live() {
  const { name = "" } = useParams();
  const navigate = useNavigate();
  const [model, setModel] = useState<LiveModel | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [facing, setFacing] = useState<"environment" | "user">("environment");
  const [photo, setPhoto] = useState<string | null>(null);  // object URL of a gallery photo, instead of the camera
  const [frame, setFrame] = useState<{ w: number; h: number } | null>(null);
  const [view, setView] = useState({ w: window.innerWidth, h: window.innerHeight });
  const [seen, setSeen] = useState<Seen | null>(null);
  const [turn, setTurn] = useState<Turn>(0);
  const [threshold, setThreshold] = useState(0.5);
  const [tuning, setTuning] = useState(false);
  const [saved, setSaved] = useState(0);  // frames saved for improving the model (this job, all visits)
  const [flash, setFlash] = useState(false);
  const [checking, setChecking] = useState<{ blob: Blob; url: string; w: number; h: number; guess: Seen } | null>(null);
  const video = useRef<HTMLVideoElement>(null);
  const image = useRef<HTMLImageElement>(null);
  const counts = useRef<number[]>([]);
  const live = useRef({ threshold, turn });
  live.current = { threshold, turn };
  const handheld = isHandheld();

  // Camera mode while the page is open: black status bar, portrait lock (also when opened from a link or an
  // installed app, where the lock works without full screen), all undone on the way out.
  useEffect(() => {
    const meta = document.querySelector('meta[name="theme-color"]');
    const before = meta?.getAttribute("content");
    meta?.setAttribute("content", "#000000");
    (screen.orientation as any)?.lock?.("portrait-primary").catch(() => {});
    const onResize = () => setView({ w: window.innerWidth, h: window.innerHeight });
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      if (before) meta?.setAttribute("content", before);
      try { screen.orientation.unlock(); } catch { /* not locked */ }
      if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
    };
  }, []);

  // On Android the system back first leaves full screen instead of the page: leave both with one gesture.
  useEffect(() => {
    let was = !!document.fullscreenElement;
    const onChange = () => {
      if (was && !document.fullscreenElement) {  // back where the user came from, as the system back would
        if ((window.history.state?.idx ?? 0) > 0) navigate(-1); else navigate(`/app/job/${name}`, { replace: true });
      }
      was = !!document.fullscreenElement;
    };
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, [name]);

  // How the phone is held, from gravity. Only while the screen stays portrait: if the browser turned the page
  // itself (no lock), the camera frame is already upright.
  useEffect(() => {
    const sign = IOS ? -1 : 1;  // iOS reports gravity with the opposite sign
    const onMotion = (e: DeviceMotionEvent) => {
      const g = e.accelerationIncludingGravity;
      if (!g || g.x == null || g.y == null) return;
      const portrait = !screen.orientation || screen.orientation.angle === 0;
      const t = portrait ? turnFromGravity(sign * g.x, sign * g.y) : 0;
      setTurn((old) => (old === t ? old : t));
    };
    window.addEventListener("devicemotion", onMotion);
    return () => window.removeEventListener("devicemotion", onMotion);
  }, []);

  useEffect(() => {
    if (!handheld) return;
    let m: LiveModel | null = null, gone = false;
    loadModel(name).then((x) => {
      if (gone) { x.close(); return; }
      m = x;
      setThreshold(x.report.output?.threshold ?? 0.5);
      setModel(x);
    }).catch((e) => setError(e.message));
    return () => { gone = true; m?.close(); };
  }, [name]);

  // Camera stream, restarted when switching cameras; off while a photo is shown.
  useEffect(() => {
    if (!model || photo) return;
    if (!window.isSecureContext) { setError("The camera needs a secure connection (https, or localhost over USB)."); return; }
    let stream: MediaStream | null = null, gone = false;
    navigator.mediaDevices.getUserMedia({ video: { facingMode: facing, width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false })
      .then((s) => {
        if (gone) { s.getTracks().forEach((t) => t.stop()); return; }
        stream = s;
        const v = video.current!;
        v.srcObject = s;
        v.onloadedmetadata = () => { setFrame({ w: v.videoWidth, h: v.videoHeight }); v.play(); };
        v.onresize = () => setFrame({ w: v.videoWidth, h: v.videoHeight });
      })
      .catch((e) => setError(e.name === "NotAllowedError" ? "Camera access was denied. Allow it in the browser settings and try again."
                                                          : `Cannot open the camera: ${e.message}`));
    return () => { gone = true; stream?.getTracks().forEach((t) => t.stop()); };
  }, [model, facing, photo]);

  // Inference loop on the camera: one frame at a time, frames arriving meanwhile are skipped.
  useEffect(() => {
    if (!model || photo || !frame || checking) return;
    const canvas = document.createElement("canvas");
    let raf = 0, busy = false, gone = false;
    const tick = async () => {
      if (gone) return;
      const v = video.current;
      if (!busy && v && v.readyState >= 2) {
        busy = true;
        try {
          const r = await infer(model, v, canvas, live.current.threshold, live.current.turn);
          if (!gone) show(r);
        } catch (e) { setError(String(e)); }
        busy = false;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => { gone = true; cancelAnimationFrame(raf); };
  }, [model, photo, frame, checking]);

  // A photo is scored once (as it is: photos carry their own orientation), and again when the sensitivity changes.
  useEffect(() => {
    if (!model || !photo || !frame || !image.current) return;
    counts.current = [];
    infer(model, image.current, document.createElement("canvas"), threshold, 0).then(show).catch((e) => setError(String(e)));
  }, [model, photo, frame, threshold]);

  useEffect(() => { listCaptures(name).then((c) => setSaved(c.length)); }, [name]);

  // The shutter: the frame upright, scored again as saved (so the marks sit on the saved picture), to check.
  async function shoot() {
    const src = photo ? image.current : video.current;
    if (!src || !model) return;
    const blob = await uprightJpeg(src, photo ? 0 : turn);
    if (!blob) return;
    setFlash(true);
    setTimeout(() => setFlash(false), 180);
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.src = url;
    try {
      await img.decode();
      const guess = await infer(model, img, document.createElement("canvas"), threshold, 0);
      setChecking({ blob, url, w: img.naturalWidth, h: img.naturalHeight, guess });
    } catch (e) {
      URL.revokeObjectURL(url);
      setError(String(e));
    }
  }

  async function keep(said: Said | undefined) {
    if (!checking) return;
    if (await addCapture(name, checking.blob, said)) setSaved((n) => n + 1);
    else setError("Could not save the photo: this browser blocks storage for the site.");
    URL.revokeObjectURL(checking.url);
    setChecking(null);
  }

  function discard() {
    if (checking) URL.revokeObjectURL(checking.url);
    setChecking(null);
  }

  function show(r: Seen) {
    counts.current = [...counts.current.slice(-(SMOOTH - 1)), r.count];
    setSeen(r);
  }

  if (!handheld) return <Navigate to={`/app/job/${name}`} replace />;

  const counting = model?.report.task === "count";
  const sorted = [...counts.current].sort((a, b) => a - b);
  const count = sorted.length ? sorted[Math.floor(sorted.length / 2)] : 0;
  // Older reports have no "background" field: their background was always the first label.
  const report = model?.report;
  const bg = report && ("background" in report ? report.background : report.labels[0]);
  const what = report?.labels.filter((l) => l !== bg).map((l) => l.replace(/_/g, " ")).join(", ");
  // Where the frame is drawn: filling the screen for the camera, whole for a photo. Until its size is known it
  // is mounted out of sight, so the stream or the photo can report it.
  const box = frame ? fit(frame.w, frame.h, view.w, view.h, photo ? "contain" : "cover") : null;
  const face = { transform: `rotate(${turn}deg)`, transition: "transform 250ms ease" };  // labels face the user

  return (
    <div className="fixed inset-0 z-40 touch-none overflow-hidden bg-black text-white select-none">
      <div className={cx("pointer-events-none absolute inset-0 z-10 bg-white transition-opacity duration-150", flash ? "opacity-60" : "opacity-0")} />
      {model && !error && (
        <div className={box ? "absolute" : "invisible absolute"} style={box ?? undefined}>
          {photo
            ? <img ref={image} src={photo} alt="" className="block h-full w-full"
                   onLoad={(e) => setFrame({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })} />
            : <video ref={video} playsInline muted className="block h-full w-full" />}
          {counting && seen?.points.map((p, i) => (
            <span key={i} className="absolute h-5 w-5 -translate-x-1/2 -translate-y-1/2 rounded-full border-[3px] border-white bg-ember-500/80 shadow-lg"
                  style={{ left: `${p[0] * 100}%`, top: `${p[1] * 100}%` }} />
          ))}
        </div>
      )}

      {!model && !error && <Centered><p className="flex items-center gap-3 text-white/80"><Spinner /> Loading your model</p></Centered>}
      {error && (
        <Centered>
          <div className="max-w-sm space-y-4 px-6 text-center">
            <p className="text-white/90">{error}</p>
            <button onClick={() => window.location.reload()} className="h-11 rounded-full bg-white px-5 font-medium text-ink">Try again</button>
          </div>
        </Centered>
      )}

      {IOS && (
        <button onClick={() => navigate(`/app/job/${name}`)} aria-label="Close"
                className="absolute top-[max(0.75rem,env(safe-area-inset-top))] left-3 grid h-10 w-10 place-items-center rounded-full bg-black/35 backdrop-blur">
          <X className="h-5 w-5" />
        </button>
      )}
      {saved > 0 && !error && (
        <Link to={`/app/job/${name}#improve`} replace
              className="absolute top-[max(0.75rem,env(safe-area-inset-top))] left-1/2 z-20 -translate-x-1/2 rounded-full bg-black/45 px-3.5 py-2 text-xs font-medium whitespace-nowrap text-white backdrop-blur">
          {saved} photo{saved > 1 ? "s" : ""} · Improve the model
        </Link>
      )}
      {model && seen && (
        // A square slot, so the label turns inside it instead of past the screen edge.
        <div className="absolute top-[max(0.5rem,env(safe-area-inset-top))] right-1 grid h-20 w-20 place-items-center">
          <span style={face} className="rounded-full bg-black/35 px-2.5 py-1 text-[11px] whitespace-nowrap text-white/75 backdrop-blur">
            {model.accelerator === "webgpu" ? "GPU" : "CPU"} {Math.round(seen.ms)} ms
          </span>
        </div>
      )}

      {model && !error && (
        <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/55 to-transparent px-4 pt-16 pb-[max(1.25rem,env(safe-area-inset-bottom))]">
          {tuning && counting && (
            <input type="range" min={0.2} max={0.95} step={0.05} value={threshold} aria-label="Sensitivity"
                   onChange={(e) => setThreshold(+e.target.value)} className="mb-4 w-full touch-auto accent-ember-500" />
          )}
          <div className="flex items-center justify-between gap-3">
            <div style={face} className="grid h-16 min-w-16 place-items-center rounded-2xl bg-black/35 px-3 backdrop-blur">
              {counting ? (
                <div className="text-center leading-none">
                  <p className="text-3xl font-bold tabular-nums">{count}</p>
                  <p className="mt-1 max-w-24 truncate text-[10px] text-white/70">{what}</p>
                </div>
              ) : (
                <div className="text-center leading-tight">
                  <p className="max-w-32 truncate text-base font-semibold">{seen?.top?.label.replace(/_/g, " ") ?? "..."}</p>
                  <p className="text-[10px] text-white/70">{seen?.top ? `${Math.round(seen.top.p * 100)}%` : ""}</p>
                </div>
              )}
            </div>
            <div className="flex gap-3">
              <button onClick={shoot} aria-label="Take a photo to improve the model"
                      className="grid h-14 w-14 place-items-center rounded-full border-4 border-white bg-white/25 active:bg-white/60" />
              {counting && <RoundButton label="Sensitivity" on={tuning} style={face} onClick={() => setTuning(!tuning)}><SlidersHorizontal className="h-5 w-5" /></RoundButton>}
              {photo
                ? <RoundButton label="Camera" style={face} onClick={() => { setPhoto(null); setFrame(null); setSeen(null); counts.current = []; }}><Camera className="h-5 w-5" /></RoundButton>
                : <RoundButton label="Switch camera" style={face} onClick={() => { setFrame(null); setFacing(facing === "environment" ? "user" : "environment"); }}><RefreshCw className="h-5 w-5" /></RoundButton>}
              <label style={face} className="grid h-12 w-12 cursor-pointer place-items-center rounded-full bg-black/35 backdrop-blur" aria-label="Use a photo">
                <ImageIcon className="h-5 w-5" />
                <input type="file" accept="image/*" hidden onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) { setFrame(null); setSeen(null); setPhoto(URL.createObjectURL(f)); }
                  e.target.value = "";
                }} />
              </label>
            </div>
          </div>
        </div>
      )}
      {checking && model && <CheckPhoto c={checking} report={model.report} view={view} onKeep={keep} onDiscard={discard} />}
    </div>
  );
}

// The photo just taken, frozen, with what the model said: the user fixes it (a class, or the objects' points)
// before it is kept. Keeping it as it is takes one tap when the model was right.
function CheckPhoto({ c, report, view, onKeep, onDiscard }: {
  c: { url: string; w: number; h: number; guess: Seen }; report: LiveModel["report"]; view: { w: number; h: number };
  onKeep: (said: Said | undefined) => void; onDiscard: () => void;
}) {
  const counting = report.task === "count";
  const bg = "background" in report ? report.background : report.labels[0];
  const classes = counting ? report.labels.filter((l) => l !== bg) : report.labels;
  const [label, setLabel] = useState(c.guess.top?.label ?? classes[0]);
  const [points, setPoints] = useState<[number, number, string][]>(c.guess.points.map(([x, y, l]) => [x, y, l]));
  const [cls, setCls] = useState(classes[0]);
  const panel = counting ? (classes.length > 1 ? 190 : 140) : 120 + 48 * Math.ceil(classes.length / 2);
  const box = fit(c.w, c.h, view.w, Math.max(160, view.h - panel), "contain");
  const nice = (l: string) => l.replace(/_/g, " ");

  const tap = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!counting) return;
    const r = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
    const near = points.findIndex(([px, py]) => Math.hypot((px - x) * r.width, (py - y) * r.height) < 22);
    setPoints(near >= 0 ? points.filter((_, i) => i !== near)
                        : [...points, [Math.round(x * 1e4) / 1e4, Math.round(y * 1e4) / 1e4, cls]]);
  };

  return (
    <div className="absolute inset-0 z-50 flex flex-col bg-black touch-auto">
      <div className="relative" style={{ height: Math.max(160, view.h - panel) }}>
        <div className="absolute" style={box} onClick={tap}>
          <img src={c.url} alt="The photo just taken" className="block h-full w-full" />
          {counting && points.map(([x, y], i) => (
            <span key={i} className="pointer-events-none absolute h-5 w-5 -translate-x-1/2 -translate-y-1/2 rounded-full border-[3px] border-white bg-ember-500/80 shadow-lg"
                  style={{ left: `${x * 100}%`, top: `${y * 100}%` }} />
          ))}
        </div>
      </div>
      <div className="flex-1 space-y-3 px-4 pt-3 pb-[max(1rem,env(safe-area-inset-bottom))]">
        {counting ? (
          <>
            <p className="text-sm text-white/80">
              <b className="text-white">{points.length}</b> marked. Tap to add one, tap a mark to remove it.
            </p>
            {classes.length > 1 && (
              <div className="flex flex-wrap gap-2">
                {classes.map((l) => <Chip key={l} on={cls === l} onClick={() => setCls(l)}>{nice(l)}</Chip>)}
              </div>
            )}
          </>
        ) : (
          <>
            <p className="text-sm text-white/80">
              The model said <b className="text-white">{nice(c.guess.top?.label ?? "?")}</b>. What is it?
            </p>
            <div className="grid grid-cols-2 gap-2">
              {classes.map((l) => <Chip key={l} on={label === l} onClick={() => setLabel(l)}>{nice(l)}</Chip>)}
            </div>
          </>
        )}
        <div className="flex gap-3">
          <button onClick={onDiscard} className="h-11 flex-1 rounded-full bg-white/15 font-medium">Discard</button>
          <button onClick={() => onKeep(counting ? { points } : { label })}
                  className="flex h-11 flex-1 items-center justify-center gap-2 rounded-full bg-white font-medium text-ink">
            <Check className="h-4 w-4" /> Keep
          </button>
        </div>
      </div>
    </div>
  );
}

function Chip({ on, onClick, children }: { on: boolean; onClick: () => void; children: ReactNode }) {
  return <button onClick={onClick} className={cx("h-10 truncate rounded-full px-4 text-sm font-medium",
                                                 on ? "bg-ember-500 text-white" : "bg-white/15 text-white")}>{children}</button>;
}

function Centered({ children }: { children: ReactNode }) {
  return <div className="absolute inset-0 grid place-items-center">{children}</div>;
}

function RoundButton({ label, on, onClick, style, children }: {
  label: string; on?: boolean; onClick: () => void; style?: React.CSSProperties; children: ReactNode;
}) {
  return <button onClick={onClick} aria-label={label} style={style}
                 className={cx("grid h-12 w-12 place-items-center rounded-full backdrop-blur", on ? "bg-ember-500" : "bg-black/35")}>{children}</button>;
}

/** The frame as a JPEG, turned upright (as the model saw it) and at most SAVE_SIDE pixels on its long side. */
async function uprightJpeg(src: HTMLVideoElement | HTMLImageElement, turn: Turn): Promise<Blob | null> {
  const w0 = src instanceof HTMLVideoElement ? src.videoWidth : src.naturalWidth;
  const h0 = src instanceof HTMLVideoElement ? src.videoHeight : src.naturalHeight;
  const k = Math.min(1, SAVE_SIDE / Math.max(w0, h0));
  const w = Math.round(w0 * k), h = Math.round(h0 * k);
  const { width, height, m } = uprightTransform(w, h, turn);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(m[0], m[1], m[2], m[3], m[4], m[5]);
  ctx.drawImage(src, 0, 0, w, h);
  return new Promise((ok) => canvas.toBlob(ok, "image/jpeg", 0.9));
}

/** One frame (video or photo) through the model: turned upright, shrunk with Pillow's filter to the input square,
 * RGB 0-255 as float32. Points come back on the frame as the camera gives it. */
async function infer(model: LiveModel, src: HTMLVideoElement | HTMLImageElement, canvas: HTMLCanvasElement,
                     threshold: number, turn: Turn): Promise<Seen> {
  const t0 = performance.now();
  const s = model.size;
  const w = src instanceof HTMLVideoElement ? src.videoWidth : src.naturalWidth;
  const h = src instanceof HTMLVideoElement ? src.videoHeight : src.naturalHeight;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(src, 0, 0, w, h);
  // Stretching to a square and turning by 90 degree steps commute, so turn the small image: much cheaper.
  const small = uprightRgb(resizeBilinear(ctx.getImageData(0, 0, w, h).data, w, h, s), s, turn);
  const outs = await model.run(Float32Array.from(small));
  const ms = performance.now() - t0;
  const labels = model.report.labels;
  const output = model.report.output;
  if (model.report.task === "count" && output?.kind === "boxes") {
    // A detector: boxes and scores, told apart by size (their order when they are the same size, 4 classes).
    const nb = anchors(s) * 4;
    const boxes = outs.find((o) => o.length === nb) ?? outs[0];
    const scores = outs.find((o) => o !== boxes) ?? outs[1];
    const points = decodeBoxes(boxes, scores, s, labels, threshold, output.nms_iou ?? 0.6, output.max_objects ?? 1000)
      .map(([u, v, c]): LivePoint => [...toFrame(u, v, turn), c]);
    return { points, count: points.length, top: null, ms };
  }
  if (model.report.task === "count") {
    const g = s / (output?.stride_px ?? 8);
    // Told apart by size: the points grid has one channel per label, the density one less (no background).
    const grid = outs.find((o) => o.length === g * g * labels.length) ?? outs[0];
    const density = outs.find((o) => o.length === g * g * (labels.length - 1));
    const points = decode(grid, g, g, labels, threshold).map(([u, v, c]): LivePoint => [...toFrame(u, v, turn), c]);
    const count = density ? Math.round(density.reduce((a, b) => a + b, 0)) : points.length;
    return { points, count, top: null, ms };
  }
  const out = outs[0];
  let b = 0;
  for (let c = 1; c < labels.length; c++) if (out[c] > out[b]) b = c;
  return { points: [], count: 0, top: { label: labels[b], p: out[b] }, ms };
}
