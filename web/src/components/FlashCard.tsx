// "Flash to my ESP32-CAM": writes our firmware and the job's model.bin onto the board over USB, then shows what the
// board sees, live, on the same cable. One tap: the device picker opens first (browsers only allow it straight
// from a tap), then the files download, the board says which firmware and model it runs, and only what differs
// is written (everything, about 80 s on a phone; the model alone, less). "Flash everything" skips the check.
import { Activity, AlertTriangle, Check, Cpu, Loader2, RotateCcw, Square, Usb, Zap } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { logLine, modelCrc, parseLine, planWrite, type BoardEvent, type OnBoard } from "../flash/board";
import { flash, flashRoute, FlashError, loadFiles, pickPort, probe, watch, type Files, type Progress, type Watching } from "../flash/flasher";
import { Button, Notice, cx } from "./ui";

type State =
  | { step: "idle" }
  | { step: "picking" }
  | { step: "loading" }
  | { step: "checking" }
  | { step: "flashing"; progress: Progress }
  | { step: "watching" }
  | { step: "stopped"; error?: string }
  | { step: "failed"; error: string; reset: boolean };

type Live = { ready?: { input: number; labels: string[] }; last?: BoardEvent; error?: string; frames: number };

const MAX_LOG = 400;

// port: a board already picked (the live test's "Flash this model first"); onBoard: what that board said it runs
// (no need to ask again); onFlashed: what to do once written, instead of watching the board here.
export function FlashCard({ job, hardware, modelUrl, port: given, onBoard: known, onFlashed, label }: {
  job: string; hardware: string; modelUrl: string; port?: any; onBoard?: OnBoard; onFlashed?: () => void; label?: string;
}) {
  const [state, setState] = useState<State>({ step: "idle" });
  const [live, setLive] = useState<Live>({ frames: 0 });
  const [log, setLog] = useState<string[]>([]);
  const [note, setNote] = useState<string | null>(null);  // what the last flash wrote, when not everything
  const port = useRef<any>(null);
  const files = useRef<(Files & { model: Uint8Array }) | null>(null);
  const watching = useRef<Watching | null>(null);
  const route = flashRoute();

  const addLog = (s: string) => setLog((l) => [...l.slice(-MAX_LOG + 1), logLine(s)]);

  useEffect(() => () => { watching.current?.stop(); }, []);
  useEffect(() => { files.current = null; }, [job, modelUrl]);

  // Keep the screen on and warn before leaving while the board is being written.
  useEffect(() => {
    if (state.step !== "flashing") return;
    let lock: any = null;
    (navigator as any).wakeLock?.request("screen").then((l: any) => { lock = l; }).catch(() => {});
    const leave = (e: BeforeUnloadEvent) => { e.preventDefault(); };
    window.addEventListener("beforeunload", leave);
    return () => { window.removeEventListener("beforeunload", leave); lock?.release?.().catch?.(() => {}); };
  }, [state.step === "flashing"]);

  const startWatch = async () => {
    setLive({ frames: 0 });
    try {
      watching.current = await watch(port.current, files.current?.manifest.monitorBaud ?? 115200, (line) => {
        addLog(line);
        const ev = parseLine(line);
        if (ev.kind === "ready") setLive((v) => ({ ...v, ready: { input: ev.input, labels: ev.labels }, error: undefined }));
        else if (ev.kind === "error") setLive((v) => ({ ...v, error: ev.message }));
        else if (ev.kind === "label" || ev.kind === "counts") setLive((v) => ({ ...v, last: ev, error: undefined, frames: v.frames + 1 }));
      }, (error) => { watching.current = null; setState({ step: "stopped", error }); });
      setState({ step: "watching" });
    } catch (e) {
      setState({ step: "failed", error: (e as Error).message, reset: false });
    }
  };

  const start = async (force = false) => {
    await watching.current?.stop();
    watching.current = null;
    setLog([]);
    setNote(null);
    setState({ step: "picking" });
    try {
      port.current = given ?? await pickPort();
      setState({ step: "loading" });
      try {
        files.current ??= await loadFiles(hardware, modelUrl);
      } catch (e) {
        if (e instanceof FlashError) throw e;
        throw new FlashError(`Could not get the files to write: ${(e as Error).message}. Check the connection and try again.`, "pick");
      }
      const f = files.current;
      let onBoard: OnBoard = null;
      if (!force) {
        setState({ step: "checking" });
        onBoard = known !== undefined ? known : await probe(port.current, f.manifest.monitorBaud).catch(() => null);
        addLog(onBoard ? `board runs firmware ${onBoard.firmware ?? "?"}, model ${onBoard.crc ?? "none"}` : "board did not say what it runs");
      }
      const plan = planWrite(onBoard, f.manifest.firmware, modelCrc(f.model), force);
      const parts = f.parts.filter((p) => (p.name === "model" ? plan.model : plan.firmware));
      if (!parts.length) {
        addLog("nothing to write: the board has this firmware and this model");
        setNote("The board already has this firmware and this model: nothing to write.");
      } else {
        setNote(plan.firmware && plan.model ? null : plan.model ? "Only the model was written: the firmware was up to date."
                                                                 : "Only the firmware was written: the model was already there.");
        const t0 = Date.now();
        await flash(port.current, { ...f, parts }, (progress) => setState({ step: "flashing", progress }), addLog);
        addLog(`written in ${((Date.now() - t0) / 1000).toFixed(0)} s`);
      }
      if (onFlashed) { setState({ step: "idle" }); onFlashed(); } else await startWatch();
    } catch (e) {
      const err = e as FlashError;
      addLog(`failed: ${err.message}`);
      setState({ step: "failed", error: err.message, reset: err.message.includes("IO0") });
    }
  };

  const stop = async () => {
    await watching.current?.stop();
    watching.current = null;
    setState({ step: "stopped" });
  };

  const busy = state.step === "picking" || state.step === "loading" || state.step === "checking" || state.step === "flashing";

  if (!route) {
    return (
      <div className="mt-3 space-y-2">
        <Notice>This browser cannot reach USB devices, so it cannot flash the board. Use Chrome on an Android phone, or
          Chrome or Edge on a computer.</Notice>
        <IPhoneNote />
      </div>
    );
  }

  return (
    <div className="mt-3 space-y-3">
      {state.step === "idle" && !given && (
        <p className="text-sm text-muted">
          Plug the ESP32-CAM, on its USB board, into this {route === "usb" ? "phone" : "computer"} with a data
          cable{route === "usb" ? " (and a USB OTG adapter if the phone needs one)" : ""}, then tap Flash and pick the board. It writes what the board is missing (the firmware and your model
          take about a minute and a half), then shows what the camera sees.
        </p>
      )}

      {(state.step === "idle" || state.step === "failed" || state.step === "stopped") && (
        <>
          <Button className="w-full" onClick={() => start()}>
            <Zap className="h-4 w-4" /> {state.step === "idle" ? label ?? "Flash to my ESP32-CAM" : state.step === "failed" ? "Try again" : "Flash again"}
          </Button>
          <button onClick={() => start(true)} className="w-full text-center text-xs text-muted underline hover:text-ink">
            Flash everything (for a board that does not start)
          </button>
        </>
      )}
      {note && state.step !== "failed" && !busy && <p className="flex items-center gap-2 text-sm text-muted"><Check className="h-4 w-4 text-green-600" /> {note}</p>}

      {busy && <Working state={state} />}

      {state.step === "failed" && (
        <div className="rounded-[var(--radius-control)] bg-red-50 px-4 py-3 text-sm text-red-800">
          <p className="flex items-start gap-2"><AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" /> {state.error}</p>
          {state.reset && (
            <p className="mt-2 text-xs text-red-700">On the ESP32-CAM-MB board, IO0 is the button on the USB board and RST
              the one on the camera board.</p>
          )}
        </div>
      )}

      {(state.step === "watching" || state.step === "stopped") && (
        <BoardView live={live} watching={state.step === "watching"} error={state.step === "stopped" ? state.error : undefined} />
      )}

      {state.step === "watching" && (
        <Button variant="secondary" size="sm" className="w-full" onClick={stop}><Square className="h-4 w-4" /> Stop watching</Button>
      )}
      {state.step === "stopped" && port.current && (
        <Button variant="secondary" size="sm" className="w-full" onClick={startWatch}><RotateCcw className="h-4 w-4" /> Watch the board again</Button>
      )}

      {log.length > 0 && (
        <details className="text-sm">
          <summary className="cursor-pointer text-muted hover:text-ink">Board log ({log.length} {log.length === 1 ? "line" : "lines"})</summary>
          <pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap break-all rounded-lg bg-gray-900 p-3 font-mono text-[11px] leading-snug text-green-300">
            {log.join("\n")}
          </pre>
        </details>
      )}
    </div>
  );
}

function Working({ state }: { state: State }) {
  if (state.step === "picking") {
    return <p className="flex items-center gap-2 text-sm text-muted"><Usb className="h-4 w-4" /> Pick the board in the list the browser shows.</p>;
  }
  if (state.step === "checking") {
    return <p className="flex items-center gap-2 text-sm text-muted"><Loader2 className="h-4 w-4 animate-spin" /> Checking what the board has</p>;
  }
  if (state.step === "loading") {
    return <p className="flex items-center gap-2 text-sm text-muted"><Loader2 className="h-4 w-4 animate-spin" /> Getting the firmware and your model</p>;
  }
  if (state.step !== "flashing") return null;
  const p = state.progress;
  const share = p.total ? p.done / p.total : 0;
  const text = p.step === "connect" ? "Talking to the board"
    : p.step === "restart" ? "Restarting the board"
    : `Writing ${p.part ?? ""}, do not unplug`;
  return (
    <div>
      <div className="flex items-center justify-between text-sm">
        <span className="flex items-center gap-2 font-medium"><Loader2 className="h-4 w-4 animate-spin text-brand-600" /> {text}</span>
        <span className="tabular-nums text-muted">{Math.round(share * 100)}%</span>
      </div>
      <div className="mt-2 h-2 overflow-hidden rounded-full bg-gray-100">
        <div className="h-full rounded-full bg-brand-600 transition-[width] duration-300" style={{ width: `${share * 100}%` }} />
      </div>
      {p.step === "connect" && (
        <p className="mt-2 text-xs text-muted">If this takes more than 20 seconds: hold IO0, tap RST, release IO0.</p>
      )}
    </div>
  );
}

const pct = (v: number) => `${Math.round(v * 100)}%`;

function BoardView({ live, watching, error }: { live: Live; watching: boolean; error?: string }) {
  const last = live.last;
  return (
    <div className="rounded-[var(--radius-control)] border border-line bg-white p-4">
      <div className="flex items-center justify-between text-xs text-muted">
        <span className="flex items-center gap-1.5">
          {watching ? <Activity className="h-3.5 w-3.5 text-green-600" /> : <Square className="h-3.5 w-3.5" />}
          {watching ? "Live from the board" : "Stopped"}
        </span>
        {live.ready && <span className="flex items-center gap-1"><Cpu className="h-3.5 w-3.5" /> model loaded, {live.ready.input} px</span>}
      </div>
      {!last && !live.error && watching && (
        <p className="mt-3 flex items-center gap-2 text-sm text-muted"><Loader2 className="h-4 w-4 animate-spin" />
          {live.ready ? "Waiting for the first picture" : "Starting the board"}</p>
      )}
      {last && "jpeg" in last && last.jpeg && (
        <img src={`data:image/jpeg;base64,${last.jpeg}`} alt="What the board's camera sees"
             className="mt-3 h-24 w-24 rounded-lg bg-gray-100 object-cover" />
      )}
      {last?.kind === "label" && (
        <div className="mt-2">
          <p className="flex items-baseline justify-between gap-3">
            <span className="truncate text-3xl font-bold tracking-tight">{last.label.replace(/_/g, " ")}</span>
            <span className="text-2xl font-semibold tabular-nums text-brand-600">{pct(last.score)}</span>
          </p>
          <div className="mt-3 space-y-1.5">
            {Object.entries(last.scores).sort((a, b) => b[1] - a[1]).map(([name, v]) => (
              <div key={name} className="flex items-center gap-2 text-xs">
                <span className={cx("w-24 truncate", name === last.label ? "font-semibold" : "text-muted")}>{name.replace(/_/g, " ")}</span>
                <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-gray-100">
                  <div className={cx("h-full rounded-full", name === last.label ? "bg-brand-600" : "bg-gray-300")} style={{ width: pct(v) }} />
                </div>
                <span className="w-9 text-right tabular-nums text-muted">{pct(v)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
      {last?.kind === "counts" && (
        <div className="mt-2 grid grid-cols-2 gap-2">
          {Object.entries(last.counts).map(([name, n]) => (
            <div key={name} className="rounded-lg bg-gray-50 px-3 py-2">
              <p className="text-3xl font-bold tabular-nums">{n}</p>
              <p className="truncate text-xs text-muted">{name.replace(/_/g, " ")}</p>
            </div>
          ))}
        </div>
      )}
      {last && "ms" in last && (
        <p className="mt-3 flex items-center gap-1 text-xs text-muted"><Check className="h-3.5 w-3.5 text-green-600" />
          {(last.ms / 1000).toFixed(1)} s per picture on the chip, {live.frames} pictures so far</p>
      )}
      {live.error && <p className="mt-3 text-sm text-red-700">The board says: {live.error}</p>}
      {error && <p className="mt-3 text-sm text-red-700">{error}</p>}
    </div>
  );
}

export function IPhoneNote() {
  return (
    <details className="text-sm">
      <summary className="cursor-pointer text-muted hover:text-ink">Why can't I flash my camera from an iPhone?</summary>
      <p className="mt-2 text-muted">
        Flashing goes over the USB cable, and Safari has no way for a web page to reach USB devices (no WebUSB, no Web
        Serial). On iPhone and iPad every browser, Chrome included, runs on Safari's engine, so none of them can. Use an
        Android phone with Chrome, or Chrome or Edge on a computer. Your model stays here; you can test it live with the
        iPhone's camera.
      </p>
    </details>
  );
}
