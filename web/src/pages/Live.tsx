import { ArrowLeft, Camera, Image as ImageIcon, RefreshCw, SlidersHorizontal } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Link, Navigate, useParams } from "react-router-dom";
import { Spinner, cx } from "../components/ui";
import { decode, type LivePoint } from "../live/decode";
import { isHandheld } from "../live/device";
import { resizeBilinear } from "../live/resize";
import { loadModel, type LiveModel } from "../live/runtime";

// The trained model on the phone's camera, run on the phone: nothing leaves it and no network is needed once
// the model is saved. The whole frame is stretched to the model's square input, as ingest does with photos,
// so the model sees the geometry it was trained on and points map back to the full frame.

type Seen = { points: LivePoint[]; top: { label: string; p: number } | null; ms: number };
const SMOOTH = 5;  // frames in the median count shown, so the number does not flicker

export default function Live() {
  const { name = "" } = useParams();
  const [model, setModel] = useState<LiveModel | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [facing, setFacing] = useState<"environment" | "user">("environment");
  const [photo, setPhoto] = useState<string | null>(null);  // object URL of a gallery photo, instead of the camera
  const [frame, setFrame] = useState<{ w: number; h: number } | null>(null);
  const [seen, setSeen] = useState<Seen | null>(null);
  const [threshold, setThreshold] = useState(0.5);
  const [tuning, setTuning] = useState(false);
  const video = useRef<HTMLVideoElement>(null);
  const image = useRef<HTMLImageElement>(null);
  const counts = useRef<number[]>([]);
  const thresholdRef = useRef(threshold);
  thresholdRef.current = threshold;
  const handheld = isHandheld();

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
    navigator.mediaDevices.getUserMedia({ video: { facingMode: facing, width: { ideal: 1280 } }, audio: false })
      .then((s) => {
        if (gone) { s.getTracks().forEach((t) => t.stop()); return; }
        stream = s;
        const v = video.current!;
        v.srcObject = s;
        v.onloadedmetadata = () => { setFrame({ w: v.videoWidth, h: v.videoHeight }); v.play(); };
      })
      .catch((e) => setError(e.name === "NotAllowedError" ? "Camera access was denied. Allow it in the browser settings and reload."
                                                          : `Cannot open the camera: ${e.message}`));
    return () => { gone = true; stream?.getTracks().forEach((t) => t.stop()); };
  }, [model, facing, photo]);

  // Inference loop on the camera: one frame at a time, frames arriving meanwhile are skipped.
  useEffect(() => {
    if (!model || photo || !frame) return;
    const canvas = document.createElement("canvas");
    let raf = 0, busy = false, gone = false;
    const tick = async () => {
      if (gone) return;
      const v = video.current;
      if (!busy && v && v.readyState >= 2) {
        busy = true;
        try { const r = await infer(model, v, canvas, thresholdRef.current); if (!gone) show(r); } catch (e) { setError(String(e)); }
        busy = false;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => { gone = true; cancelAnimationFrame(raf); };
  }, [model, photo, frame]);

  // A photo is scored once, and again when the sensitivity changes.
  useEffect(() => {
    if (!model || !photo || !frame || !image.current) return;
    counts.current = [];
    infer(model, image.current, document.createElement("canvas"), threshold).then(show).catch((e) => setError(String(e)));
  }, [model, photo, frame, threshold]);

  function show(r: Seen) {
    counts.current = [...counts.current.slice(-(SMOOTH - 1)), r.points.length];
    setSeen(r);
  }

  if (!handheld) return <Navigate to={`/app/job/${name}`} replace />;

  const counting = model?.report.task === "count";
  const sorted = [...counts.current].sort((a, b) => a - b);
  const count = sorted.length ? sorted[Math.floor(sorted.length / 2)] : 0;
  const what = model?.report.labels.slice(1).map((l) => l.replace(/_/g, " ")).join(", ");

  return (
    <div className="fixed inset-0 z-40 flex flex-col bg-black text-white">
      <div className="flex items-center justify-between gap-2 px-3 pt-[max(0.5rem,env(safe-area-inset-top))] pb-2">
        <Link to={`/app/job/${name}`} className="flex h-10 items-center gap-1 rounded-full bg-white/10 px-3 text-sm" aria-label="Back">
          <ArrowLeft className="h-4 w-4" /> Back
        </Link>
        {model && <span className="rounded-full bg-white/10 px-3 py-1.5 text-xs text-white/80">
          on this phone, {model.accelerator === "webgpu" ? "GPU" : "CPU"}{seen ? `, ${Math.round(seen.ms)} ms` : ""}
        </span>}
      </div>

      <div className="relative flex min-h-0 flex-1 items-center justify-center overflow-hidden">
        {!model && !error && <p className="flex items-center gap-3 text-white/80"><Spinner /> Loading your model</p>}
        {error && (
          <div className="max-w-sm space-y-4 px-6 text-center">
            <p className="text-white/90">{error}</p>
            <button onClick={() => window.location.reload()} className="h-11 rounded-full bg-white px-5 font-medium text-ink">Try again</button>
          </div>
        )}
        {model && !error && (
          <div className="relative" style={frame ? { aspectRatio: `${frame.w} / ${frame.h}`,
                                                      width: `min(100vw, calc((100dvh - 190px) * ${frame.w / frame.h}))` } : undefined}>
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
      </div>

      {model && (
        <div className="space-y-3 px-4 pt-3 pb-[max(1rem,env(safe-area-inset-bottom))]">
          <div className="flex items-end justify-between gap-3">
            <div className="min-w-0">
              {counting ? (
                <>
                  <p className="text-5xl font-bold tracking-tight tabular-nums">{count}</p>
                  <p className="truncate text-sm text-white/70">{what} in view</p>
                </>
              ) : (
                <>
                  <p className="truncate text-3xl font-bold tracking-tight">{seen?.top?.label.replace(/_/g, " ") ?? "..."}</p>
                  <p className="text-sm text-white/70">{seen?.top ? `${Math.round(seen.top.p * 100)}% sure` : "looking"}</p>
                </>
              )}
            </div>
            <div className="flex gap-2">
              <RoundButton label="Sensitivity" on={tuning} onClick={() => setTuning(!tuning)}><SlidersHorizontal className="h-5 w-5" /></RoundButton>
              {photo
                ? <RoundButton label="Camera" onClick={() => { setPhoto(null); setFrame(null); setSeen(null); counts.current = []; }}><Camera className="h-5 w-5" /></RoundButton>
                : <RoundButton label="Switch camera" onClick={() => { setFrame(null); setFacing(facing === "environment" ? "user" : "environment"); }}><RefreshCw className="h-5 w-5" /></RoundButton>}
              <label className="grid h-12 w-12 cursor-pointer place-items-center rounded-full bg-white/15" aria-label="Use a photo">
                <ImageIcon className="h-5 w-5" />
                <input type="file" accept="image/*" hidden onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) { setFrame(null); setSeen(null); setPhoto(URL.createObjectURL(f)); }
                  e.target.value = "";
                }} />
              </label>
            </div>
          </div>
          {tuning && counting && (
            <label className="block text-sm text-white/80">
              Sensitivity: marks an object when the model is {Math.round(threshold * 100)}% sure
              <input type="range" min={0.2} max={0.95} step={0.05} value={threshold}
                     onChange={(e) => setThreshold(+e.target.value)} className="mt-2 w-full accent-ember-500" />
            </label>
          )}
        </div>
      )}
    </div>
  );
}

function RoundButton({ label, on, onClick, children }: { label: string; on?: boolean; onClick: () => void; children: React.ReactNode }) {
  return <button onClick={onClick} aria-label={label}
                 className={cx("grid h-12 w-12 place-items-center rounded-full", on ? "bg-ember-500" : "bg-white/15")}>{children}</button>;
}

/** One frame (video or photo) through the model: stretched to the input square, RGB 0-255 as float32. */
async function infer(model: LiveModel, src: HTMLVideoElement | HTMLImageElement, canvas: HTMLCanvasElement,
                     threshold: number): Promise<Seen> {
  const t0 = performance.now();
  // Read the frame at full size and shrink it with Pillow's filter, as ingest did with the photos the model was
  // trained and scored on: the browser's canvas scaling samples differently and moved some counts.
  const s = model.size;
  const w = src instanceof HTMLVideoElement ? src.videoWidth : src.naturalWidth;
  const h = src instanceof HTMLVideoElement ? src.videoHeight : src.naturalHeight;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(src, 0, 0, w, h);
  const small = resizeBilinear(ctx.getImageData(0, 0, w, h).data, w, h, s);
  const rgb = Float32Array.from(small);
  const out = await model.run(rgb);
  const ms = performance.now() - t0;
  const labels = model.report.labels;
  if (model.report.task === "count") {
    const g = s / (model.report.output?.stride_px ?? 8);
    return { points: decode(out, g, g, labels, threshold), top: null, ms };
  }
  let b = 0;
  for (let c = 1; c < labels.length; c++) if (out[c] > out[b]) b = c;
  return { points: [], top: { label: labels[b], p: out[b] }, ms };
}
