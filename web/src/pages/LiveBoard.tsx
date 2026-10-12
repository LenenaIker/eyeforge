import { AlertTriangle, Camera, Loader2, Usb, X, Zap } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { fileUrl, type Report } from "../api";
import { FlashCard, IPhoneNote } from "../components/FlashCard";
import { Spinner } from "../components/ui";
import { BOARDS, boardGuess, boardMismatch, fromBase64, modelCrc, parseLine, parseManifest, PHOTO_SIZE,
         type BoardEvent, type Manifest, type OnBoard } from "../flash/board";
import { flashRoute, pickPort, takePort, watch, type Watching } from "../flash/flasher";
import { addCapture, listCaptures, type Said } from "../live/captures";
import { fit } from "../live/orient";
import { Centered, CheckPhoto, Dots, type Guess } from "../live/overlay";

// The trained model live on the ESP32-CAM, seen from this page over the USB cable: the board runs the model and
// sends each result with the square it saw (a 96x96 JPEG, about one a second); this page draws them as the
// phone's live test does. On connecting, the board says which model it has (the CRC in model.bin's header): if
// it is not this job's, the page flashes it first. The shutter asks the board for a photo at a bigger size
// (PHOTO_SIZE), which the user checks and keeps for "Make it better", as on the phone.

type Ready = Extract<BoardEvent, { kind: "ready" }>;
type Frame = Extract<BoardEvent, { kind: "label" | "counts" }>;
type Photo = Extract<BoardEvent, { kind: "photo" }>;
type Step =
  | { step: "connect" }
  | { step: "starting" }
  | { step: "flash"; reason: string; onBoard: OnBoard }
  | { step: "live" }
  | { step: "stopped"; error?: string };

const ANSWER_MS = 8000;   // the board says "ready" about 2 s after a reset
const PHOTO_MS = 30000;   // a photo at the biggest size takes a few seconds through the cable

export default function LiveBoard() {
  const { name = "" } = useParams();
  const navigate = useNavigate();
  const [report, setReport] = useState<Report | null>(null);
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [crc, setCrc] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [step, setStep] = useState<Step>({ step: "connect" });
  const [ready, setReady] = useState<Ready | null>(null);
  const [update, setUpdate] = useState<string | null>(null);  // a newer firmware the board could get
  const [frame, setFrame] = useState<Frame | null>(null);
  const [frames, setFrames] = useState(0);
  const [saved, setSaved] = useState(0);
  const [taking, setTaking] = useState(false);
  const [lastPhoto, setLastPhoto] = useState<{ bytes: number; w: number; h: number; ms: number; captureMs: number } | null>(null);
  const [checking, setChecking] = useState<{ blob: Blob; url: string; w: number; h: number; guess: Guess } | null>(null);
  const [view, setView] = useState({ w: window.innerWidth, h: window.innerHeight });
  const port = useRef<any>(takePort());
  const board = useRef<Watching | null>(null);
  const answered = useRef<((r: Ready | null, message?: string) => void) | null>(null);
  const onBoard = useRef<OnBoard>(null);  // what the board said it runs, so a flash writes only what differs
  const photo = useRef<((p: Photo) => void) | null>(null);
  const route = flashRoute();

  useEffect(() => {
    const onResize = () => setView({ w: window.innerWidth, h: window.innerHeight });
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  // The job's report (labels, task), the board's manifest (firmware version, speed) and the model's fingerprint.
  useEffect(() => {
    let gone = false;
    (async () => {
      const r = await fetch(fileUrl(name, "report.json"), { cache: "no-store" });
      if (!r.ok) throw new Error("This model is not ready or no longer exists.");
      const rep: Report = await r.json();
      const dir = BOARDS[rep.hardware];
      if (!dir || !rep.device_blob_path) throw new Error("This model is not packed for the ESP32-CAM: retrain it to test it on the board.");
      const [m, bin] = await Promise.all([
        fetch(`${dir}/manifest.json`, { cache: "no-cache" }).then((x) => x.json()).then(parseManifest),
        fetch(fileUrl(name, rep.device_blob_path), { cache: "no-cache" }).then((x) => x.arrayBuffer()),
      ]);
      if (gone) return;
      setReport(rep);
      setManifest(m);
      setCrc(modelCrc(new Uint8Array(bin)));
    })().catch((e) => { if (!gone) setError(e.message); });
    return () => { gone = true; };
  }, [name]);

  useEffect(() => { listCaptures(name).then((c) => setSaved(c.length)); }, [name]);
  useEffect(() => () => { board.current?.stop(); }, []);

  // Start watching as soon as everything is known, when the board was picked on the job page.
  useEffect(() => {
    if (manifest && report && port.current && step.step === "connect") start();
  }, [manifest, report]);

  function onLine(line: string) {
    const ev = parseLine(line);
    if (ev.kind === "ready") {
      onBoard.current = { firmware: ev.firmware, crc: ev.crc };
      answered.current?.(ev);
    } else if (ev.kind === "error") {
      if (ev.firmware && !onBoard.current) onBoard.current = { firmware: ev.firmware };
      if (answered.current) answered.current(null, ev.message);
      else setError(`The board says: ${ev.message}`);
    } else if (ev.kind === "label" || ev.kind === "counts") {
      setFrame(ev);
      setFrames((n) => n + 1);
    } else if (ev.kind === "photo") photo.current?.(ev);
  }

  async function start() {
    if (!manifest || !report) return;
    await board.current?.stop();
    board.current = null;
    setStep({ step: "starting" });
    setFrame(null); setFrames(0); setReady(null); setUpdate(null); setError(null);
    onBoard.current = null;
    const answer = new Promise<{ r: Ready | null; message?: string }>((ok) => {
      answered.current = (r, message) => { answered.current = null; ok({ r, message }); };
      setTimeout(() => answered.current?.(null), ANSWER_MS);
    });
    try {
      board.current = await watch(port.current, manifest.monitorBaud, onLine, (e) => {
        board.current = null;
        setStep((s) => (s.step === "flash" ? s : { step: "stopped", error: e }));
      });
    } catch (e) {
      answered.current = null;
      setStep({ step: "stopped", error: (e as Error).message });
      return;
    }
    const { r, message } = await answer;
    const wrong = boardMismatch(r, crc, manifest.firmware);
    if (wrong && !wrong.optional) {
      await board.current?.stop();
      board.current = null;
      setStep({ step: "flash", reason: message && !r ? `The board says: ${message}.` : wrong.reason, onBoard: onBoard.current });
      return;
    }
    setReady(r);
    if (wrong) setUpdate(wrong.reason);
    const threshold = (report as any).output?.threshold;
    if (report.task === "count" && typeof threshold === "number") board.current?.send(`threshold ${threshold}`).catch(() => {});
    setStep({ step: "live" });
  }

  async function connect() {
    try {
      port.current = await pickPort();
      await start();
    } catch (e) {
      setStep({ step: "stopped", error: (e as Error).message });
    }
  }

  async function updateFirmware(reason: string) {
    await board.current?.stop();
    board.current = null;
    setStep({ step: "flash", reason, onBoard: onBoard.current });
  }

  // The shutter: a photo at a bigger size from the board, its centre square (what the model sees), to check.
  async function shoot() {
    if (!board.current || taking) return;
    setTaking(true);
    const guess = boardGuess(frame);
    const t0 = performance.now();
    try {
      const p = await new Promise<Photo>((ok, no) => {
        const timer = setTimeout(() => { photo.current = null; no(new Error("The board did not send the photo. Try again.")); }, PHOTO_MS);
        photo.current = (x) => { clearTimeout(timer); photo.current = null; ok(x); };
        board.current!.send(`capture ${PHOTO_SIZE}`).catch((e) => { clearTimeout(timer); photo.current = null; no(e); });
      });
      const ms = performance.now() - t0;
      setLastPhoto({ bytes: p.bytes, w: p.w, h: p.h, ms, captureMs: p.captureMs });
      const square = await centreSquare(new Blob([fromBase64(p.jpeg)], { type: "image/jpeg" }));
      setChecking({ ...square, url: URL.createObjectURL(square.blob), guess });
    } catch (e) {
      setError((e as Error).message);
    }
    setTaking(false);
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

  const close = () => navigate(`/app/job/${name}`);
  const counting = report?.task === "count";
  const bg = report && ("background" in report ? report.background : report.labels[0]);
  const what = report?.labels.filter((l) => l !== bg).map((l) => l.replace(/_/g, " ")).join(", ");
  const box = fit(1, 1, view.w, Math.max(160, view.h - 150), "contain");
  const count = frame?.kind === "counts" ? Object.values(frame.counts).reduce((a, b) => a + b, 0) : 0;

  return (
    <div className="fixed inset-0 z-40 overflow-hidden bg-black text-white select-none">
      {step.step === "live" && frame?.jpeg && (
        <div className="absolute" style={box}>
          <img src={`data:image/jpeg;base64,${frame.jpeg}`} alt="What the board's camera sees" className="block h-full w-full" />
          {counting && frame.kind === "counts" && <Dots points={frame.points ?? []} />}
        </div>
      )}

      {!route && (
        <Centered>
          <div className="max-w-sm space-y-4 rounded-2xl bg-white p-5 text-ink">
            <p>This browser cannot reach USB devices. Use Chrome on an Android phone, or Chrome or Edge on a computer.</p>
            <IPhoneNote />
          </div>
        </Centered>
      )}
      {route && error && !checking && (
        <div className="absolute inset-x-4 top-16 z-20 flex items-start gap-2 rounded-2xl bg-red-600/90 px-4 py-3 text-sm">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" /> <span className="flex-1">{error}</span>
          <button onClick={() => setError(null)} aria-label="Dismiss"><X className="h-4 w-4" /></button>
        </div>
      )}
      {route && !error && (!report || !manifest) && (
        <Centered><p className="flex items-center gap-3 text-white/80"><Spinner /> Loading your model</p></Centered>
      )}
      {route && report && manifest && step.step === "connect" && (
        <Centered>
          <div className="max-w-sm space-y-4 px-6 text-center">
            <p className="text-white/90">Plug the ESP32-CAM into this device with a data cable, then pick it in the list.</p>
            <button onClick={connect} className="inline-flex h-11 items-center gap-2 rounded-full bg-white px-5 font-medium text-ink">
              <Usb className="h-4 w-4" /> Connect the ESP32-CAM
            </button>
          </div>
        </Centered>
      )}
      {step.step === "starting" && (
        <Centered><p className="flex items-center gap-3 text-white/80"><Loader2 className="h-5 w-5 animate-spin" /> Starting the board</p></Centered>
      )}
      {step.step === "live" && !frame && (
        <Centered><p className="flex items-center gap-3 text-white/80"><Loader2 className="h-5 w-5 animate-spin" /> Waiting for the first picture</p></Centered>
      )}
      {step.step === "stopped" && (
        <Centered>
          <div className="max-w-sm space-y-4 px-6 text-center">
            <p className="text-white/90">{step.error ?? "Stopped."}</p>
            <button onClick={connect} className="inline-flex h-11 items-center gap-2 rounded-full bg-white px-5 font-medium text-ink">
              <Usb className="h-4 w-4" /> Connect again
            </button>
          </div>
        </Centered>
      )}
      {step.step === "flash" && report && (
        <div className="absolute inset-0 z-30 grid place-items-center overflow-auto bg-black/80 p-4">
          <div className="w-full max-w-md rounded-2xl bg-white p-5 text-ink">
            <p className="flex items-center gap-2 font-semibold"><Zap className="h-4 w-4 text-ember-700" /> Flash this model first</p>
            <p className="mt-2 text-sm text-muted">{step.reason} Only what the board is missing is written (the firmware and
              this model take about a minute and a half); then the live test starts.</p>
            <FlashCard job={name} hardware={report.hardware} modelUrl={fileUrl(name, report.device_blob_path ?? "model.bin")}
                       port={port.current} onBoard={step.onBoard} onFlashed={start}
                       label={step.onBoard?.crc && step.onBoard.crc === crc ? "Update the firmware" : "Flash this model first"} />
          </div>
        </div>
      )}

      <button onClick={close} aria-label="Close"
              className="absolute top-[max(0.75rem,env(safe-area-inset-top))] left-3 z-20 grid h-10 w-10 place-items-center rounded-full bg-black/35 backdrop-blur">
        <X className="h-5 w-5" />
      </button>
      {saved > 0 && (
        <Link to={`/app/job/${name}#improve`} replace
              className="absolute top-[max(0.75rem,env(safe-area-inset-top))] left-1/2 z-20 -translate-x-1/2 rounded-full bg-black/45 px-3.5 py-2 text-xs font-medium whitespace-nowrap text-white backdrop-blur">
          {saved} photo{saved > 1 ? "s" : ""} · Improve the model
        </Link>
      )}

      {step.step === "live" && (
        <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/55 to-transparent px-4 pt-16 pb-[max(1.25rem,env(safe-area-inset-bottom))]">
          {update && (
            <p className="mb-3 text-center text-xs text-white/70">{update}{" "}
              <button onClick={() => updateFirmware(update)} className="underline">Update it</button></p>
          )}
          {lastPhoto && (
            <p className="mb-3 text-center text-[11px] text-white/60">
              Last photo {lastPhoto.w}x{lastPhoto.h}, {Math.round(lastPhoto.bytes / 1024)} KB in {(lastPhoto.ms / 1000).toFixed(1)} s
            </p>
          )}
          <div className="mx-auto flex max-w-md items-center justify-between gap-3">
            <div className="grid h-16 min-w-16 place-items-center rounded-2xl bg-black/35 px-3 backdrop-blur">
              {counting ? (
                <div className="text-center leading-none">
                  <p className="text-3xl font-bold tabular-nums">{frame ? count : "..."}</p>
                  <p className="mt-1 max-w-24 truncate text-[10px] text-white/70">{what}</p>
                </div>
              ) : (
                <div className="text-center leading-tight">
                  <p className="max-w-32 truncate text-base font-semibold">{frame?.kind === "label" ? frame.label.replace(/_/g, " ") : "..."}</p>
                  <p className="text-[10px] text-white/70">{frame?.kind === "label" ? `${Math.round(frame.score * 100)}%` : ""}</p>
                </div>
              )}
            </div>
            <button onClick={shoot} disabled={taking || !frame} aria-label="Take a photo with the board to improve the model"
                    className="grid h-14 w-14 place-items-center rounded-full border-4 border-white bg-white/25 active:bg-white/60 disabled:opacity-60">
              {taking ? <Loader2 className="h-5 w-5 animate-spin" /> : <Camera className="h-5 w-5 opacity-0" />}
            </button>
            <div className="w-24 text-right text-[10px] leading-tight text-white/60">
              {frame && <p className="text-[11px] text-white/80">ESP32-CAM {(frame.ms / 1000).toFixed(1)} s</p>}
              <p>{frames} pictures</p>
              {ready && <p>firmware {ready.firmware}</p>}
            </div>
          </div>
        </div>
      )}
      {checking && report && <CheckPhoto c={checking} report={report} view={view} onKeep={keep} onDiscard={discard} />}
    </div>
  );
}

/** The centre square of a photo, as the board's model sees its frames, as a JPEG. */
async function centreSquare(jpeg: Blob): Promise<{ blob: Blob; w: number; h: number }> {
  const bmp = await createImageBitmap(jpeg);
  const side = Math.min(bmp.width, bmp.height);
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = side;
  canvas.getContext("2d")!.drawImage(bmp, (bmp.width - side) / 2, (bmp.height - side) / 2, side, side, 0, 0, side, side);
  bmp.close();
  const blob = await new Promise<Blob | null>((ok) => canvas.toBlob(ok, "image/jpeg", 0.92));
  if (!blob) throw new Error("Could not keep the photo.");
  return { blob, w: side, h: side };
}
